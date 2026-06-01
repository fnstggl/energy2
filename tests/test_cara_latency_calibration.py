"""Tests for the CARA latency calibration module.

Covers PHASE B (calibration classes) and PHASE E (final-status
classifier) of the mission spec. No real CARA data is required —
unit-level tests use synthetic distributions.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cara_latency_calibration import (  # noqa: E402
    FINAL_STATUS_VALUES,
    PROMOTION_THRESHOLDS,
    BaselineFallbackGate,
    ConservativeMultiplierCalibration,
    QuantileResidualCalibration,
    SplitConformalUpperBound,
    classify_tail_status,
    tail_safety_metrics,
    time_train_calibration_split,
    train_calibration_test_split,
)


class _MockPointPredictor:
    """Predicts a constant per-call (used in unit tests)."""

    def __init__(self, value):
        self._value = float(value)

    def predict(self, X):
        return np.full(X.shape[0], self._value, dtype=np.float64)


class _LinearPredictor:
    def __init__(self, slope=1.0, intercept=0.0):
        self.slope = slope
        self.intercept = intercept

    def predict(self, X):
        return self.slope * np.asarray(X)[:, 0] + self.intercept


# ---------- 1. Conservative multiplier --------------------------------------


def test_conservative_multiplier_calibrates_to_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.uniform(1.0, 3.0, size=1000)
    X = np.ones((1000, 4))
    base = _MockPointPredictor(value=1.0)
    cm = ConservativeMultiplierCalibration(
        target_quantile=0.95, base=base,
    ).fit(X, y)
    # Calibrated prediction should cover roughly 95% of training data.
    pred = cm.predict(X)
    coverage = float((y <= pred).mean())
    assert 0.94 <= coverage <= 0.99


def test_conservative_multiplier_clamps_at_min_multiplier():
    rng = np.random.default_rng(1)
    # base predicts much higher than y — raw_ratio_quantile would be < 1
    y = rng.uniform(0.1, 0.5, size=500)
    X = np.ones((500, 4))
    base = _MockPointPredictor(value=10.0)
    cm = ConservativeMultiplierCalibration(
        target_quantile=0.95, base=base, min_multiplier=1.0,
    ).fit(X, y)
    assert cm.multiplier == pytest.approx(1.0, abs=1e-9)
    diag = cm.diagnostics()
    assert diag["clamped_by_min_multiplier"] is True


def test_conservative_multiplier_diagnostics_complete():
    X = np.ones((100, 4))
    y = np.linspace(1.0, 2.0, 100)
    base = _MockPointPredictor(1.0)
    cm = ConservativeMultiplierCalibration(
        target_quantile=0.95, base=base,
    ).fit(X, y)
    d = cm.diagnostics()
    for k in ("method", "target_quantile", "multiplier",
              "empirical_calibration_coverage"):
        assert k in d
    assert d["method"] == "conservative_multiplier"


# ---------- 2. Quantile residual calibration -------------------------------


def test_quantile_residual_widens_when_residuals_positive():
    rng = np.random.default_rng(3)
    y = rng.exponential(scale=2.0, size=1000) + 1.0
    X = np.ones((1000, 4))
    base = _MockPointPredictor(value=0.0)  # always under-predicts
    qr = QuantileResidualCalibration(
        target_quantile=0.95, base=base,
    ).fit(X, y)
    assert qr.residual_quantile > 0
    assert qr.conservatism_widened is True


def test_quantile_residual_default_disallows_tighten():
    rng = np.random.default_rng(4)
    y = rng.uniform(0, 1, size=500)
    X = np.ones((500, 4))
    base = _MockPointPredictor(value=10.0)  # always over-predicts
    qr = QuantileResidualCalibration(
        target_quantile=0.95, base=base, allow_tighten=False,
    ).fit(X, y)
    # residuals would be negative -> clamped to 0
    assert qr.residual_quantile == pytest.approx(0.0)
    assert qr.conservatism_widened is False


def test_quantile_residual_predict_adds_offset():
    X = np.ones((10, 4))
    y = np.full(10, 5.0)
    base = _MockPointPredictor(value=2.0)
    qr = QuantileResidualCalibration(
        target_quantile=0.95, base=base, allow_tighten=False,
    ).fit(X, y)
    # residual_quantile = 3.0 (y - pred = 3)
    pred = qr.predict(X)
    np.testing.assert_allclose(pred, base.predict(X) + qr.residual_quantile)


# ---------- 3. Split-conformal --------------------------------------------


def test_split_conformal_empirical_coverage_at_least_alpha():
    rng = np.random.default_rng(5)
    n = 5000
    X = np.ones((n, 4))
    base = _MockPointPredictor(value=0.0)
    y = rng.exponential(scale=1.0, size=n)
    sc = SplitConformalUpperBound(alpha=0.95, base=base).fit(X, y)
    pred = sc.predict(X)
    # In-sample coverage on calibration data should be >= alpha.
    assert (y <= pred).mean() >= 0.94


def test_split_conformal_finite_sample_q_level():
    X = np.ones((100, 4))
    y = np.full(100, 1.0)
    base = _MockPointPredictor(value=0.0)
    sc = SplitConformalUpperBound(alpha=0.95, base=base).fit(X, y)
    expected_q = float(np.ceil(101 * 0.95) / 100)
    assert sc.q_level == pytest.approx(min(1.0, expected_q))


def test_split_conformal_rejects_invalid_alpha():
    X = np.ones((10, 4))
    y = np.ones(10)
    base = _MockPointPredictor(value=0.0)
    with pytest.raises(ValueError):
        SplitConformalUpperBound(alpha=0.0, base=base).fit(X, y)
    with pytest.raises(ValueError):
        SplitConformalUpperBound(alpha=1.0, base=base).fit(X, y)


# ---------- 4. Baseline fallback gate -------------------------------------


def test_floor_at_baseline_returns_max():
    ml = _MockPointPredictor(value=1.0)
    base = _MockPointPredictor(value=2.0)
    gate = BaselineFallbackGate(policy="floor_at_baseline", ml=ml, baseline=base)
    pred, used = gate.predict_with_fallback(np.ones((10, 4)))
    np.testing.assert_allclose(pred, 2.0)
    assert used.all()


def test_floor_returns_ml_when_ml_higher():
    ml = _MockPointPredictor(value=5.0)
    base = _MockPointPredictor(value=2.0)
    gate = BaselineFallbackGate(policy="floor_at_baseline", ml=ml, baseline=base)
    pred, used = gate.predict_with_fallback(np.ones((10, 4)))
    np.testing.assert_allclose(pred, 5.0)
    assert not used.any()


def test_baseline_when_ood_policy_fires_below_threshold():
    ml = _MockPointPredictor(value=0.3)
    base = _MockPointPredictor(value=1.0)
    gate = BaselineFallbackGate(
        policy="baseline_when_ood", ml=ml, baseline=base, ood_tolerance_x=0.5,
    )
    pred, used = gate.predict_with_fallback(np.ones((10, 4)))
    # 0.3 < 0.5 * 1.0, so fallback fires.
    assert used.all()
    np.testing.assert_allclose(pred, 1.0)


def test_baseline_fallback_gate_diagnostics():
    ml = _MockPointPredictor(value=1.0)
    base = _MockPointPredictor(value=2.0)
    gate = BaselineFallbackGate(policy="floor_at_baseline", ml=ml, baseline=base)
    _, used = gate.predict_with_fallback(np.ones((100, 4)))
    diag = gate.diagnostics(used)
    assert diag["method"] == "baseline_fallback_gate"
    assert diag["fallback_fired_rate"] == 1.0
    assert diag["fallback_total_count"] == 100


# ---------- 5. Train/calibration splits never overlap with test -----------


def test_train_calibration_split_disjoint():
    train = np.arange(1000)
    sub_train, cal = train_calibration_test_split(train, calibration_frac=0.25)
    assert set(sub_train.tolist()).isdisjoint(set(cal.tolist()))
    assert set(sub_train.tolist()) | set(cal.tolist()) == set(train.tolist())


def test_train_calibration_split_is_deterministic():
    train = np.arange(500)
    a = train_calibration_test_split(train, calibration_frac=0.25)
    b = train_calibration_test_split(train, calibration_frac=0.25)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_time_calibration_uses_recent_tail():
    train = np.arange(1000)
    ts = np.arange(2000, dtype=np.float64)  # later for later indices
    sub_train, cal = time_train_calibration_split(
        train, ts, calibration_frac=0.25,
    )
    # Calibration indices must have ts >= max train tail value.
    assert ts[cal].min() >= ts[sub_train].max()


def test_calibration_frac_out_of_range_rejected():
    train = np.arange(100)
    with pytest.raises(ValueError):
        train_calibration_test_split(train, calibration_frac=0.6)
    with pytest.raises(ValueError):
        time_train_calibration_split(train, np.arange(100, dtype=float),
                                     calibration_frac=0.0)


# ---------- 6. Tail-safety metrics + final-status classifier --------------


def test_tail_safety_metrics_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    m = tail_safety_metrics(y, pred, target_coverage=0.95)
    assert m["empirical_coverage"] == 1.0
    assert m["coverage_error"] == pytest.approx(0.05)
    assert m["undercoverage_rate"] == 0.0
    assert m["mean_conservatism"] == pytest.approx(0.5)


def test_promotion_thresholds_documented_for_required_keys():
    for key in (("ttft", 0.5), ("ttft", 0.95), ("ttft", 0.99),
                ("e2e", 0.95), ("e2e", 0.99)):
        assert key in PROMOTION_THRESHOLDS


def test_classify_tail_status_promotes_ttft_p50_when_gates_pass():
    status, _ = classify_tail_status(
        target_family="ttft", quantile=0.5,
        time_holdout_improvement_pct=42.0,
        random_holdout_improvement_pct=51.0,
        by_instance_holdout_improvement_pct=37.0,
        empirical_coverage=0.5,
        undercoverage_rate=0.0,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=True, no_test_label_calibration=True,
    )
    assert status == "shadow_ready"


def test_classify_tail_status_blocks_ttft_p99_when_time_holdout_regresses():
    status, reason = classify_tail_status(
        target_family="ttft", quantile=0.99,
        time_holdout_improvement_pct=-16.75,  # regression
        random_holdout_improvement_pct=22.6,
        by_instance_holdout_improvement_pct=79.4,
        empirical_coverage=0.99,
        undercoverage_rate=0.0,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=True, no_test_label_calibration=True,
    )
    assert status == "diagnostic_only"
    assert "time" in reason.lower()


def test_classify_tail_status_blocks_when_coverage_too_low():
    status, reason = classify_tail_status(
        target_family="ttft", quantile=0.95,
        time_holdout_improvement_pct=20.0,
        random_holdout_improvement_pct=20.0,
        by_instance_holdout_improvement_pct=20.0,
        empirical_coverage=0.85,  # below 0.93 threshold
        undercoverage_rate=0.10,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=True, no_test_label_calibration=True,
    )
    assert status == "diagnostic_only"
    assert "coverage" in reason.lower()


def test_classify_tail_status_signals_baseline_fallback():
    status, _ = classify_tail_status(
        target_family="ttft", quantile=0.99,
        time_holdout_improvement_pct=10.0,
        random_holdout_improvement_pct=10.0,
        by_instance_holdout_improvement_pct=10.0,
        empirical_coverage=0.98,
        undercoverage_rate=0.02,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=True,
        leakage_free=True, no_test_label_calibration=True,
    )
    assert status == "baseline_fallback"


def test_classify_tail_status_rejects_on_subgroup_regression():
    status, _ = classify_tail_status(
        target_family="ttft", quantile=0.95,
        time_holdout_improvement_pct=20.0,
        random_holdout_improvement_pct=20.0,
        by_instance_holdout_improvement_pct=20.0,
        empirical_coverage=0.95,
        undercoverage_rate=0.0,
        has_subgroup_regression=True,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=True, no_test_label_calibration=True,
    )
    assert status == "rejected_regression"


def test_classify_tail_status_rejects_on_leakage():
    status, _ = classify_tail_status(
        target_family="ttft", quantile=0.5,
        time_holdout_improvement_pct=99.0,
        random_holdout_improvement_pct=99.0,
        by_instance_holdout_improvement_pct=99.0,
        empirical_coverage=0.5,
        undercoverage_rate=0.0,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=False, no_test_label_calibration=True,
    )
    assert status == "rejected_regression"


def test_classify_tail_status_rejects_test_label_calibration():
    status, _ = classify_tail_status(
        target_family="ttft", quantile=0.5,
        time_holdout_improvement_pct=99.0,
        random_holdout_improvement_pct=99.0,
        by_instance_holdout_improvement_pct=99.0,
        empirical_coverage=0.5,
        undercoverage_rate=0.0,
        has_subgroup_regression=False,
        has_subgroup_undercoverage=False,
        fallback_required_on_time=False,
        leakage_free=True, no_test_label_calibration=False,
    )
    assert status == "rejected_regression"


# ---------- 7. Final-status enum closed -------------------------------------


def test_final_status_values_closed_enum():
    expected = {
        "shadow_ready", "shadow_ready_tail_candidate",
        "diagnostic_only", "needs_more_data",
        "baseline_fallback", "rejected_regression",
    }
    assert expected == set(FINAL_STATUS_VALUES)


# ---------- 8. No-test-label assertion ------------------------------------


def test_calibrators_do_not_see_test_labels_by_construction():
    """The fit() interface only accepts (X_cal, y_cal). There is no API
    that admits a separate y_test argument. We grep the module to enforce
    this contract."""
    import inspect

    from aurelius.forecasting import cara_latency_calibration as mod
    for cls in (mod.ConservativeMultiplierCalibration,
                mod.QuantileResidualCalibration,
                mod.SplitConformalUpperBound):
        sig = inspect.signature(cls.fit)
        params = list(sig.parameters)
        # Must be (self, X_cal, y_cal) — no test labels.
        assert "y_test" not in params and "X_test" not in params, (
            f"{cls.__name__}.fit signature must not accept test data"
        )
