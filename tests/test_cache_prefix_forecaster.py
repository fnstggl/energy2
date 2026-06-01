"""Tests for the cache / prefix-reuse forecaster baselines + ML
candidates + promotion classifier."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cache_prefix_forecaster import (  # noqa: E402
    ECONOMIC_PROMOTION_THRESHOLDS,
    FINAL_STATUS_VALUES,
    FallbackToBaselineWrapper,
    GlobalReuseRateBaseline,
    HistGradientBoostingReuseClassifier,
    HistGradientBoostingReuseRegressor,
    LogisticReuseClassifier,
    PerGroupReuseRateBaseline,
    PerSessionHistoryBaseline,
    RandomForestReuseClassifier,
    RecencyFrequencyBaseline,
    auprc,
    auroc,
    brier_score,
    calibration_error,
    classify_economic_status,
    mae,
    rmse,
)

# ---------- 1. Baselines -----------------------------------------------


def test_global_reuse_rate_predicts_training_mean():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    X = np.zeros((4, 3))
    b = GlobalReuseRateBaseline().fit(X, y)
    pred = b.predict(np.zeros((10, 3)))
    np.testing.assert_allclose(pred, 0.5)
    proba = b.predict_proba(np.zeros((10, 3)))
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba[:, 1], 0.5)


def test_per_group_baseline_falls_back_to_global_for_unseen():
    y = np.array([1.0] * 10 + [0.0] * 10)
    X = np.zeros((20, 1))
    groups = np.array(["A"] * 10 + ["B"] * 10, dtype=object)
    b = PerGroupReuseRateBaseline().fit(X, y, group_keys_train=groups)
    pred_seen = b.predict(np.zeros((2, 1)),
                          group_keys_predict=np.array(["A", "B"]))
    assert pred_seen[0] == pytest.approx(1.0)
    assert pred_seen[1] == pytest.approx(0.0)
    pred_unseen = b.predict(np.zeros((1, 1)),
                            group_keys_predict=np.array(["NEW"]))
    # global mean = 0.5
    assert pred_unseen[0] == pytest.approx(0.5)


def test_per_group_baseline_handles_none_in_group_keys():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    X = np.zeros((5, 1))
    groups = np.array([None, None, "A", "A", "A"], dtype=object)
    b = PerGroupReuseRateBaseline().fit(X, y, group_keys_train=groups)
    pred = b.predict(np.zeros((2, 1)),
                     group_keys_predict=np.array([None, "A"]))
    # No crash; predictions are finite.
    assert np.isfinite(pred).all()


def test_recency_frequency_baseline_branches_on_seen_count():
    rng = np.random.default_rng(0)
    n = 200
    seen = rng.integers(0, 5, size=n).astype(np.float64)
    # Force: y = 1 when seen >= 1, else 0
    y = (seen >= 1).astype(np.float64)
    X = np.zeros((n, 1))
    b = RecencyFrequencyBaseline(min_seen=1).fit(
        X, y, rolling_seen_count_train=seen)
    pred = b.predict(np.zeros((3, 1)),
                     rolling_seen_count_predict=np.array([0.0, 1.0, 5.0]))
    # First is unseen → low rate; rest are seen → high rate.
    assert pred[0] < 0.5
    assert pred[1] > 0.5
    assert pred[2] > 0.5


def test_per_session_history_returns_global_when_no_history_yet():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    X = np.zeros((4, 1))
    sids = np.array(["A", "A", "B", "B"], dtype=object)
    b = PerSessionHistoryBaseline().fit(X, y, session_keys_train=sids)
    pred = b.predict(
        np.zeros((1, 1)),
        session_keys_predict=np.array(["NEW"]),
        rolling_session_history=np.array([float("nan")]),
    )
    assert pred[0] == pytest.approx(0.5)


def test_per_session_history_uses_rolling_history_when_available():
    y = np.array([1.0, 0.0])
    X = np.zeros((2, 1))
    sids = np.array(["A", "A"], dtype=object)
    b = PerSessionHistoryBaseline().fit(X, y, session_keys_train=sids)
    pred = b.predict(
        np.zeros((1, 1)),
        session_keys_predict=np.array(["A"]),
        rolling_session_history=np.array([0.75]),
    )
    assert pred[0] == pytest.approx(0.75)


# ---------- 2. ML candidates -------------------------------------------


def test_logistic_classifier_learns_simple_pattern():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    # y = 1 iff X[:,0] > 0
    y = (X[:, 0] > 0).astype(np.float64)
    m = LogisticReuseClassifier().fit(X, y)
    p = m.predict(X)
    # Trivially separable → AUROC near 1.0
    assert auroc(y, p) > 0.95


def test_hist_gradient_boosting_classifier_runs():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    m = HistGradientBoostingReuseClassifier(max_iter=50).fit(X, y)
    p = m.predict(X)
    assert auroc(y, p) > 0.9


def test_random_forest_classifier_runs_on_small_data():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(np.float64)
    m = RandomForestReuseClassifier(n_estimators=20).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (n, 2)


def test_hist_gradient_boosting_regressor_predicts_continuous_target():
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (50.0 + 20.0 * X[:, 0] + 5.0 * X[:, 1])  # continuous reuse_pct-style
    m = HistGradientBoostingReuseRegressor(max_iter=50).fit(X, y)
    p = m.predict(X)
    assert mae(y, p) < 10.0


def test_fallback_to_baseline_wrapper_falls_back_when_uncertain():
    rng = np.random.default_rng(3)
    n = 100
    X = rng.normal(size=(n, 2))
    # Make an ML model that always predicts probability ~0.5 (low confidence).
    class _AlwaysHalf:
        def predict_proba(self, X):
            out = np.zeros((X.shape[0], 2))
            out[:, 0] = 0.5
            out[:, 1] = 0.5
            return out
    def _base_proba(X):
        out = np.zeros((X.shape[0], 2))
        out[:, 1] = 0.8
        out[:, 0] = 0.2
        return out
    w = FallbackToBaselineWrapper(
        ml_model=_AlwaysHalf(), baseline_proba=_base_proba, min_confidence=0.6)
    p = w.predict_proba(X)
    # All ML predictions are below the confidence threshold -> baseline used.
    np.testing.assert_allclose(p[:, 1], 0.8)


# ---------- 3. Metrics --------------------------------------------------


def test_auroc_perfect_score():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, s) == pytest.approx(1.0)


def test_auroc_random_around_0p5():
    rng = np.random.default_rng(4)
    n = 1000
    y = rng.integers(0, 2, size=n).astype(np.float64)
    s = rng.uniform(size=n)
    assert 0.4 < auroc(y, s) < 0.6


def test_auprc_perfect_score():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, s) == pytest.approx(1.0)


def test_brier_score_zero_when_perfect():
    y = np.array([0.0, 1.0])
    p = np.array([0.0, 1.0])
    assert brier_score(y, p) == pytest.approx(0.0)


def test_calibration_error_lower_than_one():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    ece = calibration_error(y, p, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_mae_rmse_compute_correctly():
    y = np.array([0.0, 10.0])
    p = np.array([1.0, 11.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(1.0)


# ---------- 4. Promotion classifier ------------------------------------


def test_status_values_are_canonical():
    expected = {
        "shadow_ready_for_integration_review",
        "promising_needs_validation",
        "diagnostic_only",
        "rejected_regression",
        "blocked_by_scorer_limitations",
        "needs_more_data",
    }
    assert FINAL_STATUS_VALUES == expected


def test_classify_economic_status_diagnostic_when_under_two_pct():
    s, reason = classify_economic_status(
        best_economic_improvement_pct=1.0,
        has_subgroup_regression=False, has_calibration_failure=False,
        leakage_free=True, scorer_supports_cache_value=True)
    assert s == "diagnostic_only"
    assert "2%" in reason or "1.00" in reason


def test_classify_economic_status_promising_when_between_2_and_5():
    s, reason = classify_economic_status(
        best_economic_improvement_pct=3.5,
        has_subgroup_regression=False, has_calibration_failure=False,
        leakage_free=True, scorer_supports_cache_value=True)
    assert s == "promising_needs_validation"


def test_classify_economic_status_shadow_ready_when_above_5_no_regression():
    s, _ = classify_economic_status(
        best_economic_improvement_pct=6.0,
        has_subgroup_regression=False, has_calibration_failure=False,
        leakage_free=True, scorer_supports_cache_value=True)
    assert s == "shadow_ready_for_integration_review"


def test_classify_economic_status_rejects_subgroup_regression_at_high_alpha():
    s, _ = classify_economic_status(
        best_economic_improvement_pct=6.0,
        has_subgroup_regression=True, has_calibration_failure=False,
        leakage_free=True, scorer_supports_cache_value=True)
    assert s == "rejected_regression"


def test_classify_economic_status_blocks_when_scorer_lacks_cache_value():
    # Mission spec: if scorer doesn't support cache value, integration is
    # blocked even at >5% shadow proxy improvement.
    s, _ = classify_economic_status(
        best_economic_improvement_pct=7.0,
        has_subgroup_regression=False, has_calibration_failure=False,
        leakage_free=True, scorer_supports_cache_value=False)
    assert s == "blocked_by_scorer_limitations"


def test_classify_economic_status_rejects_leakage_unconditionally():
    s, reason = classify_economic_status(
        best_economic_improvement_pct=20.0,
        has_subgroup_regression=False, has_calibration_failure=False,
        leakage_free=False, scorer_supports_cache_value=True)
    assert s == "rejected_regression"
    assert "leakage" in reason.lower()


def test_promotion_thresholds_match_mission_spec():
    assert ECONOMIC_PROMOTION_THRESHOLDS["diagnostic_only_max_pct"] == 2.0
    assert ECONOMIC_PROMOTION_THRESHOLDS["promising_max_pct"] == 5.0


# ---------- 5. Calibration failure flags downgrade ----------------------


def test_classify_economic_status_calibration_failure_downgrades_at_high_alpha():
    s, _ = classify_economic_status(
        best_economic_improvement_pct=6.0,
        has_subgroup_regression=False, has_calibration_failure=True,
        leakage_free=True, scorer_supports_cache_value=True)
    assert s == "diagnostic_only"
