"""Tests for the TTFT-tail-with-queue-features experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1",
    "ttft_tail_with_queue_features_summary.json")


@pytest.fixture(scope="module")
def summary():
    if not os.path.exists(SUMMARY):
        pytest.skip("ttft_tail_with_queue_features_summary.json not generated; "
                    "run scripts/run_cara_ttft_tail_with_queue_features.py first")
    with open(SUMMARY) as fh:
        return json.load(fh)


# ---------- 1. Invariants -------------------------------------------------


def test_summary_invariants(summary):
    assert summary["doc_version"] == "cara_ttft_tail_with_queue_features_v1"
    assert summary["production_claim"] is False
    assert summary["no_production_claim"] is True
    assert summary["shadow_only"] is True
    assert summary["modifies_controllers_or_defaults"] is False
    assert summary["uses_oracle_as_headline"] is False


def test_queue_features_marked_out_of_fold(summary):
    assert summary["queue_features_are_out_of_fold"] is True
    assert summary["queue_target_is_derived_proxy"] is True


def test_queue_feature_names_listed(summary):
    for f in ("predicted_queue_p50", "predicted_queue_p95",
              "predicted_queue_p99", "queue_forecast_uncertainty",
              "queue_pressure_score"):
        assert f in summary["queue_feature_names"]


# ---------- 2. Comparison vs prior ----------------------------------------


def test_both_quantiles_present(summary):
    assert "p95" in summary["per_quantile"]
    assert "p99" in summary["per_quantile"]


def test_prior_calibrated_comparison_present(summary):
    for q in ("p95", "p99"):
        cell = summary["per_quantile"][q]
        assert "prior_calibrated_without_queue" in cell
        # The prior comparison should carry the prior status.
        prior = cell["prior_calibrated_without_queue"]
        if prior is not None:
            assert "final_status" in prior
            assert "calibrated_alpha_pct" in prior


def test_decision_table_has_prior_and_new(summary):
    for row in summary["final_decision_table"]:
        for k in ("prior_status", "new_status", "new_time_alpha_pct",
                  "time_alpha_delta_pct"):
            assert k in row


# ---------- 3. Promotion gates binding ------------------------------------


VALID_STATUS = frozenset({
    "shadow_ready", "shadow_ready_tail_candidate", "diagnostic_only",
    "needs_more_data", "baseline_fallback", "rejected_regression",
})


def test_new_status_in_closed_enum(summary):
    for q in ("p95", "p99"):
        assert summary["per_quantile"][q]["new_status_with_queue_features"] \
            in VALID_STATUS


def test_p95_promotes_only_if_gates_pass(summary):
    """p95 can only become shadow_ready_tail_candidate if time-holdout
    improvement >= 10% AND coverage >= 0.93 AND fallback <= 25%."""
    row = next(r for r in summary["final_decision_table"]
               if r["quantile"] == "p95")
    if row["new_status"] == "shadow_ready_tail_candidate":
        assert row["new_time_alpha_pct"] >= 10.0
        assert row["new_coverage"] >= 0.93
        assert row["new_fallback_rate"] <= 0.25


def test_p99_promotes_only_if_gates_pass(summary):
    """p99 can only move from baseline_fallback to
    shadow_ready_tail_candidate if time improvement >= 5%, coverage >=
    0.975, fallback <= 25%, no time regression."""
    row = next(r for r in summary["final_decision_table"]
               if r["quantile"] == "p99")
    if row["new_status"] == "shadow_ready_tail_candidate":
        assert row["new_time_alpha_pct"] >= 5.0
        assert row["new_coverage"] >= 0.975
        assert row["new_fallback_rate"] <= 0.25


def test_status_stays_diagnostic_or_fallback_when_no_improvement(summary):
    """If queue features do not improve the tail, the status must remain
    diagnostic_only or baseline_fallback — never silently promoted."""
    for q in ("p95", "p99"):
        cell = summary["per_quantile"][q]
        time_cell = next(
            (h for h in cell["per_holdout"] if h["holdout"] == "time_holdout"),
            None)
        if time_cell is None:
            continue
        time_alpha = time_cell["calibrated_with_queue_metrics"][
            "alpha_pct_vs_baseline"]
        fallback = time_cell["fallback_rate"]
        status = cell["new_status_with_queue_features"]
        # p95 needs >=10% to promote; p99 needs >=5% AND fallback<=25%.
        if q == "p95" and time_alpha < 10.0:
            assert status in ("diagnostic_only", "baseline_fallback",
                              "rejected_regression")
        if q == "p99" and (time_alpha < 5.0 or fallback > 0.25):
            assert status in ("diagnostic_only", "baseline_fallback",
                              "rejected_regression")


# ---------- 4. Subgroup audit present -------------------------------------


def test_subgroup_audit_present(summary):
    for q in ("p95", "p99"):
        for h in summary["per_quantile"][q]["per_holdout"]:
            assert "subgroup_audit" in h
            assert "instance_type" in h["subgroup_audit"]


# ---------- 5. No production claim + gitignore ---------------------------


def test_no_banned_phrase_in_script():
    path = os.path.join(REPO_ROOT, "scripts",
                        "run_cara_ttft_tail_with_queue_features.py")
    with open(path) as fh:
        src = fh.read().lower()
    for phrase in ("production savings", "hyperscaler-validated",
                   "guaranteed savings"):
        assert phrase not in src


def test_raw_and_analysis_sample_gitignored():
    for path in (
        os.path.join(REPO_ROOT, "data", "external", "hf",
                     "asdwb__cara_latency_prediction", "raw", "train.jsonl"),
        os.path.join(REPO_ROOT, "data", "external", "hf",
                     "asdwb__cara_latency_prediction", "train_flat",
                     "processed", "analysis_sample.jsonl"),
    ):
        if not os.path.exists(path):
            continue
        r = subprocess.run(["git", "check-ignore", path], cwd=REPO_ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"not gitignored: {path}"
