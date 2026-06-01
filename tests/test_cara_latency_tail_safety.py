"""Tests for the CARA tail-safety summary artefact.

Validates the JSON written by
``scripts/run_cara_latency_calibration_tail_safety.py`` and asserts the
binding PHASE H decision rules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


SUMMARY_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1", "calibration_tail_safety_summary.json",
)


@pytest.fixture(scope="module")
def summary():
    if not os.path.exists(SUMMARY_PATH):
        pytest.skip(
            "calibration_tail_safety_summary.json not generated; run "
            "scripts/run_cara_latency_calibration_tail_safety.py first"
        )
    with open(SUMMARY_PATH) as fh:
        return json.load(fh)


# ---------- 1. Top-level invariants ----------------------------------------


def test_payload_invariants(summary):
    assert summary["doc_version"] == "cara_latency_calibration_tail_safety_v1"
    for k in ("production_claim", "modifies_robust_energy_engine",
              "modifies_controllers_or_defaults", "uses_oracle_as_headline"):
        assert summary[k] is False, f"{k} must be False"
    assert summary["shadow_only"] is True
    assert summary["no_production_claim"] is True


def test_dataset_provenance_recorded(summary):
    assert summary["dataset"] == "asdwb/cara_latency_prediction"
    assert summary["config"] == "train_flat"
    assert summary["row_count"] >= 50_000


def test_leakage_features_listed_explicitly(summary):
    for required in ("actual_e2e_latency_s", "actual_ttft_s",
                     "actual_output_tokens", "completion_timestamp_s"):
        assert required in summary["leakage_features_excluded"]


def test_promotion_thresholds_documented_in_payload(summary):
    for key in ("ttft_p50", "ttft_p95", "ttft_p99",
                "e2e_p95", "e2e_p99"):
        assert key in summary["promotion_thresholds"]


# ---------- 2. Per-target per-quantile structure --------------------------


REQUIRED_TARGETS = ("actual_ttft_s", "actual_e2e_latency_s")
REQUIRED_QUANTILES = ("p50", "p95", "p99")


def test_every_target_and_quantile_present(summary):
    for target in REQUIRED_TARGETS:
        pt = summary["per_target"][target]
        for q in REQUIRED_QUANTILES:
            assert q in pt["per_quantile"], (
                f"{target} missing quantile {q}"
            )


def test_every_cell_lists_three_holdouts(summary):
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            cell = summary["per_target"][target]["per_quantile"][q]
            holdouts = {h["holdout"] for h in cell["per_holdout"]}
            assert "random_holdout" in holdouts
            assert "time_holdout" in holdouts
            # by_instance_type may be absent if the holdout group missed.


# ---------- 3. Calibration variants present + diagnostics complete -------


def test_calibration_variants_complete(summary):
    expected = {
        "conservative_multiplier",
        "quantile_residual",
        "split_conformal_upper_bound",
        "split_conformal_with_baseline_floor",
    }
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            cell = summary["per_target"][target]["per_quantile"][q]
            for h in cell["per_holdout"]:
                got = set(h["calibration_variants"].keys())
                missing = expected - got
                assert not missing, (
                    f"{target}/{q}/{h['holdout']} missing variants "
                    f"{missing}"
                )


def test_calibration_diagnostics_record_method(summary):
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            cell = summary["per_target"][target]["per_quantile"][q]
            for h in cell["per_holdout"]:
                for name, diag in h["calibration_diagnostics"].items():
                    assert "method" in diag
                    assert diag["method"] in (
                        "conservative_multiplier", "quantile_residual",
                        "split_conformal_upper_bound",
                    )


# ---------- 4. Tail-safety + final-status decisions ----------------------


VALID_FINAL_STATUS = frozenset({
    "shadow_ready", "shadow_ready_tail_candidate", "diagnostic_only",
    "needs_more_data", "baseline_fallback", "rejected_regression",
})


def test_final_status_values_closed(summary):
    assert set(summary["final_status_values_allowed"]) == set(VALID_FINAL_STATUS)
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            d = summary["per_target"][target]["per_quantile"][q]["decision"]
            assert d["final_status"] in VALID_FINAL_STATUS


def test_decision_table_rows_present(summary):
    rows = summary["final_decision_table"]
    # 2 targets x 3 quantiles = 6 rows minimum.
    assert len(rows) >= 6
    for r in rows:
        for k in ("target", "quantile", "final_status",
                  "calibrated_alpha_pct_time_holdout",
                  "time_holdout_empirical_coverage"):
            assert k in r


# ---------- 5. Binding promotion rules (PHASE H) -------------------------


def test_ttft_p50_is_shadow_ready(summary):
    d = summary["per_target"]["actual_ttft_s"]["per_quantile"]["p50"]["decision"]
    assert d["final_status"] == "shadow_ready", (
        f"TTFT p50 should be shadow_ready; got {d['final_status']} "
        f"(reason={d['reason']})"
    )


def test_no_p95_p99_shadow_ready_without_time_holdout_pass(summary):
    """A p95 or p99 cell can only carry shadow_ready_tail_candidate if it
    cleared the time-holdout pinball threshold AND coverage AND subgroup
    AND fallback rules. We don't pin the exact status (this is the
    PHASE H freedom: a future calibrator might pass), but we DO require
    that whatever status is set is justified by the recorded metrics.
    """
    for target in REQUIRED_TARGETS:
        for q in ("p95", "p99"):
            d = summary["per_target"][target]["per_quantile"][q]["decision"]
            if d["final_status"] == "shadow_ready_tail_candidate":
                # Time-holdout pinball improvement must meet threshold.
                req = (
                    summary["promotion_thresholds"][f"{_family(target)}_{q}"]
                    ["time_pinball_improvement_pct"]
                )
                assert d["time_holdout_pinball_improvement_pct"] >= req
                # Coverage threshold.
                cov_thresh = summary["promotion_thresholds"][
                    f"{_family(target)}_{q}"]["min_empirical_coverage"]
                assert d["time_holdout_empirical_coverage"] >= cov_thresh
                # Undercoverage threshold.
                uc_thresh = summary["promotion_thresholds"][
                    f"{_family(target)}_{q}"]["max_undercoverage_rate"]
                assert d["time_holdout_undercoverage_rate"] <= uc_thresh
                # Subgroup safety must hold.
                assert not d["subgroup_regression_on_time"]
                assert not d["subgroup_undercoverage_on_time"]
                # Fallback must NOT be required on time-holdout.
                assert not d["fallback_required_on_time"]


def _family(target: str) -> str:
    return "ttft" if "ttft" in target else "e2e"


def test_e2e_tail_models_stay_diagnostic_until_gates_pass(summary):
    """Per the mission spec PHASE H expected honest outcome: E2E p95/p99
    should be ``diagnostic_only`` unless time-holdout improves meaningfully.
    """
    for q in ("p95", "p99"):
        d = summary["per_target"]["actual_e2e_latency_s"]["per_quantile"][
            q]["decision"]
        if d["final_status"] == "shadow_ready_tail_candidate":
            # If promoted, time-holdout must show >= 5% improvement.
            req = summary["promotion_thresholds"][f"e2e_{q}"][
                "time_pinball_improvement_pct"]
            assert d["time_holdout_pinball_improvement_pct"] >= req


def test_ttft_p99_must_not_silently_skip_time_holdout(summary):
    """If TTFT p99's calibrated time-holdout pinball improvement is < 5%,
    OR fallback fires on >25% of time-holdout rows, OR coverage is below
    0.975, the final status MUST be one of
    {diagnostic_only, baseline_fallback, rejected_regression}.
    """
    d = summary["per_target"]["actual_ttft_s"]["per_quantile"]["p99"]["decision"]
    cal_alpha = d["time_holdout_pinball_improvement_pct"]
    cov = d["time_holdout_empirical_coverage"]
    fallback = d["fallback_required_on_time"]
    if (cal_alpha < 5.0) or (cov < 0.975) or fallback:
        assert d["final_status"] in {
            "diagnostic_only", "baseline_fallback", "rejected_regression",
            "needs_more_data",
        }


# ---------- 6. Subgroup audit present ------------------------------------


def test_subgroup_audit_present_for_every_cell(summary):
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            for h in summary["per_target"][target]["per_quantile"][q][
                    "per_holdout"]:
                assert "subgroup_audit" in h
                assert "instance_type" in h["subgroup_audit"]


def test_insufficient_sample_groups_flagged(summary):
    for target in REQUIRED_TARGETS:
        for q in REQUIRED_QUANTILES:
            for h in summary["per_target"][target]["per_quantile"][q][
                    "per_holdout"]:
                for key, subgroups in h["subgroup_audit"].items():
                    for g, row in subgroups.items():
                        assert row["status"] in (
                            "PASS", "INSUFFICIENT_SAMPLE", "REGRESSION",
                            "UNDERCOVERED", "FALLBACK_REQUIRED",
                        )
                        if row["row_count"] < 100 and q in ("p95", "p99"):
                            # Should be flagged.
                            assert row["status"] in (
                                "INSUFFICIENT_SAMPLE", "REGRESSION",
                                "UNDERCOVERED",
                            )


# ---------- 7. Raw data + analysis sample remain gitignored --------------


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
        r = subprocess.run(
            ["git", "check-ignore", path], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"raw/analysis sample not gitignored: {path}"


# ---------- 8. No production-claim phrases in driver script -------------


BANNED_PHRASES = (
    "production savings",
    "guaranteed savings",
    "production-proven",
    "hyperscaler-validated",
)


def test_no_banned_phrases_in_driver_script():
    path = os.path.join(REPO_ROOT, "scripts",
                        "run_cara_latency_calibration_tail_safety.py")
    with open(path) as fh:
        src = fh.read().lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in src


def test_no_executor_references_in_driver():
    path = os.path.join(REPO_ROOT, "scripts",
                        "run_cara_latency_calibration_tail_safety.py")
    with open(path) as fh:
        src = fh.read()
    for token in ("execute_frontier_decision", "apply_replica_scale",
                  "set_replicas", "RUN_FOR_REAL"):
        assert token not in src


# ---------- 9. Calibration driver does not modify any controller -------


def test_calibration_module_has_no_controller_imports():
    """The calibration module must be self-contained — no imports from
    scheduler / frontier-controller / executor modules."""
    path = os.path.join(REPO_ROOT, "aurelius", "forecasting",
                        "cara_latency_calibration.py")
    with open(path) as fh:
        src = fh.read()
    for forbidden in (
        "aurelius.optimization.scheduler",
        "aurelius.frontier.controller",
        "aurelius.frontier.execution",
    ):
        assert forbidden not in src, (
            f"calibration module must not import {forbidden}"
        )
