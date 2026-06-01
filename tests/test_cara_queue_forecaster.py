"""Tests for the CARA queue-wait forecaster baselines + status classifier."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cara_queue_forecaster import (  # noqa: E402
    QUEUE_FINAL_STATUS_VALUES,
    QUEUE_PROMOTION_THRESHOLDS,
    NumWaitingBaseline,
    QueueDepthExtrapolationBaseline,
    classify_queue_status,
)

# ---------- 1. Baselines fit + predict -----------------------------------


def test_num_waiting_baseline_predicts_zero_when_queue_empty():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 0.5, size=500)
    X = np.zeros((500, 4))
    nw = np.zeros(500)  # queue ~always empty (CARA)
    b = NumWaitingBaseline().fit(X, y, num_waiting_train=nw)
    pred = b.predict(X[:10], num_waiting_predict=np.zeros(10))
    np.testing.assert_allclose(pred, 0.0)


def test_num_waiting_baseline_scales_with_waiting():
    # Need >= 5 rows with num_waiting >= 1 to trigger the per-waiting slope
    # branch (otherwise the baseline falls back to the global median).
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])  # qwait
    nw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])    # waiting
    X = np.zeros((6, 4))
    b = NumWaitingBaseline().fit(X, y, num_waiting_train=nw)
    # median(y/nw) = median([2,2,2,2,2,2]) = 2.0
    assert b.base_service_time_s == pytest.approx(2.0)
    pred = b.predict(np.zeros((1, 4)), num_waiting_predict=np.array([5.0]))
    assert pred[0] == pytest.approx(10.0)


def test_queue_depth_extrapolation_fits_slope():
    nr = np.arange(20, dtype=float)
    y = 0.1 * nr + 0.5  # linear in num_running
    X = np.zeros((20, 4))
    b = QueueDepthExtrapolationBaseline().fit(X, y, num_running_train=nr)
    assert b.per_running_service_time_s == pytest.approx(0.1, abs=0.02)
    pred = b.predict(np.zeros((1, 4)), num_running_predict=np.array([10.0]))
    assert pred[0] == pytest.approx(1.5, abs=0.1)


def test_queue_depth_extrapolation_clamps_nonnegative():
    b = QueueDepthExtrapolationBaseline()
    b.per_running_service_time_s = 0.0
    b.intercept_s = 0.0
    b._fitted = True
    pred = b.predict(np.zeros((1, 4)), num_running_predict=np.array([-5.0]))
    assert pred[0] >= 0.0


# ---------- 2. Status classifier ------------------------------------------


def test_queue_p50_shadow_ready_when_gates_pass():
    s, _ = classify_queue_status(
        quantile=0.50, time_improvement_pct=15.0,
        random_improvement_pct=15.0, by_instance_improvement_pct=12.0,
        empirical_coverage=0.5, undercoverage_rate=0.0, fallback_rate=0.0,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "shadow_ready"


def test_queue_p50_diagnostic_when_time_below_threshold():
    s, reason = classify_queue_status(
        quantile=0.50, time_improvement_pct=3.0,
        random_improvement_pct=15.0, by_instance_improvement_pct=12.0,
        empirical_coverage=0.5, undercoverage_rate=0.0, fallback_rate=0.0,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "diagnostic_only"
    assert "time" in reason.lower()


def test_queue_p95_tail_candidate_when_gates_pass():
    s, _ = classify_queue_status(
        quantile=0.95, time_improvement_pct=12.0,
        random_improvement_pct=12.0, by_instance_improvement_pct=12.0,
        empirical_coverage=0.95, undercoverage_rate=0.0, fallback_rate=0.1,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "shadow_ready_tail_candidate"


def test_queue_p95_diagnostic_when_coverage_low():
    s, reason = classify_queue_status(
        quantile=0.95, time_improvement_pct=12.0,
        random_improvement_pct=12.0, by_instance_improvement_pct=12.0,
        empirical_coverage=0.80, undercoverage_rate=0.13, fallback_rate=0.1,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "diagnostic_only"
    assert "coverage" in reason.lower()


def test_queue_p99_baseline_fallback_when_fallback_high():
    s, _ = classify_queue_status(
        quantile=0.99, time_improvement_pct=10.0,
        random_improvement_pct=10.0, by_instance_improvement_pct=10.0,
        empirical_coverage=0.99, undercoverage_rate=0.0, fallback_rate=0.5,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "baseline_fallback"


def test_queue_rejects_on_leakage():
    s, _ = classify_queue_status(
        quantile=0.95, time_improvement_pct=99.0,
        random_improvement_pct=99.0, by_instance_improvement_pct=99.0,
        empirical_coverage=0.99, undercoverage_rate=0.0, fallback_rate=0.0,
        has_subgroup_regression=False, has_subgroup_undercoverage=False,
        leakage_free=False)
    assert s == "rejected_regression"


def test_queue_rejects_on_subgroup_regression():
    s, _ = classify_queue_status(
        quantile=0.95, time_improvement_pct=12.0,
        random_improvement_pct=12.0, by_instance_improvement_pct=12.0,
        empirical_coverage=0.95, undercoverage_rate=0.0, fallback_rate=0.0,
        has_subgroup_regression=True, has_subgroup_undercoverage=False,
        leakage_free=True)
    assert s == "rejected_regression"


def test_queue_final_status_enum_closed():
    expected = {
        "shadow_ready", "shadow_ready_tail_candidate", "diagnostic_only",
        "baseline_fallback", "needs_more_data", "rejected_regression",
    }
    assert expected == set(QUEUE_FINAL_STATUS_VALUES)


def test_queue_promotion_thresholds_documented():
    for q in (0.50, 0.95, 0.99):
        assert q in QUEUE_PROMOTION_THRESHOLDS


# ---------- 3. Artefact validation ---------------------------------------


QUEUE_JSON = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_queue_wait_forecaster_v1", "queue_wait_model_comparison.json")


@pytest.fixture(scope="module")
def queue_summary():
    if not os.path.exists(QUEUE_JSON):
        pytest.skip("queue_wait_model_comparison.json not generated; run "
                    "scripts/run_cara_queue_wait_forecaster_v1.py first")
    with open(QUEUE_JSON) as fh:
        return json.load(fh)


def test_queue_summary_target_is_derived_not_measured(queue_summary):
    td = queue_summary["target_definition"]
    assert td["name"] == "derived_queue_wait_s"
    assert td["field_quality"] == "derived"
    assert td["measured_queue_wait_available"] is False
    assert td["is_real"] is False
    assert td["is_derived"] is True


def test_queue_summary_invariants(queue_summary):
    assert queue_summary["production_claim"] is False
    assert queue_summary["no_production_claim"] is True
    assert queue_summary["shadow_only"] is True
    assert queue_summary["modifies_controllers_or_defaults"] is False


def test_queue_summary_records_baselines(queue_summary):
    assert any("per_instance_type" in b for b in queue_summary["baselines"])


def test_queue_summary_has_three_holdouts(queue_summary):
    for q in ("p50", "p95", "p99"):
        holdouts = {c["holdout"] for c in
                    queue_summary["per_quantile"][q]["per_holdout"]}
        assert "random_holdout" in holdouts
        assert "time_holdout" in holdouts


def test_queue_summary_final_status_in_enum(queue_summary):
    for q in ("p50", "p95", "p99"):
        s = queue_summary["per_quantile"][q]["final_status"]
        assert s in QUEUE_FINAL_STATUS_VALUES


def test_queue_p95_p99_not_promoted_unless_time_gate_passes(queue_summary):
    """Binding rule: p95/p99 cannot be shadow_ready_tail_candidate unless
    the time-holdout improvement meets threshold."""
    for q, quant in (("p95", 0.95), ("p99", 0.99)):
        cell = queue_summary["per_quantile"][q]
        if cell["final_status"] == "shadow_ready_tail_candidate":
            time_cell = next(
                h for h in cell["per_holdout"] if h["holdout"] == "time_holdout")
            req = QUEUE_PROMOTION_THRESHOLDS[quant]["time_pinball_improvement_pct"]
            assert (time_cell["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
                    >= req)


def test_queue_subgroup_audit_present(queue_summary):
    for q in ("p50", "p95", "p99"):
        for h in queue_summary["per_quantile"][q]["per_holdout"]:
            assert "subgroup_audit" in h
            for key, subs in h["subgroup_audit"].items():
                for g, row in subs.items():
                    assert row["status"] in (
                        "PASS", "INSUFFICIENT_SAMPLE", "REGRESSION",
                        "UNDERCOVERED", "FALLBACK_REQUIRED")
