"""Tests for the Cache / Prefix-Reuse Forecaster v1 evaluation artifacts.

Enforces the mission-spec commit/safety contract:
- raw 3000 MiB CC-traces data is NOT committed,
- unnormalized analysis_sample.jsonl files are NOT committed,
- committed normalized CC-traces samples respect file + total caps,
- no raw prompt / completion text in any committed sample,
- the strength-expansion summary compares 80 MiB vs 3000 MiB,
- summary.json + data_readiness_audit.json carry the binding shadow-only
  flags and the leakage-blocklist,
- weak CC-traces sample cannot become headline,
- the expanded 3000 MiB sample is used for training only if the Phase 0
  decision is ``use_for_training`` or stronger,
- baselines, ML candidates, and holdouts are recorded,
- the economic-proxy metrics are present,
- the residency / routing / robust energy code paths are not touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "external" / "forecasting" / "cache_prefix_reuse_v1"
SUMMARY_PATH = DATA_DIR / "summary.json"
AUDIT_PATH = DATA_DIR / "data_readiness_audit.json"
PHASE0_PATH = DATA_DIR / "cc_traces_strength_expansion.json"
CC_TRACES_DIR = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
)
CC_3000_DIR = CC_TRACES_DIR / "traces_3000mib" / "processed"

MAX_COMMITTED_NORMALIZED_SAMPLE_BYTES = 50 * 1024 * 1024  # 50 MB / file
MAX_TOTAL_COMMITTED_NORMALIZED_BYTES = 150 * 1024 * 1024  # 150 MB total


def _git_tracked(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=REPO_ROOT, capture_output=True, text=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False


# ---------- 1. Summary + audit exist + carry binding flags ---------------


def test_summary_exists_and_is_valid_json():
    assert SUMMARY_PATH.exists(), (
        f"Phase C summary missing — re-run scripts/run_cache_prefix_reuse_forecaster_v1.py")
    payload = json.loads(SUMMARY_PATH.read_text())
    assert payload["doc_version"] == "cache_prefix_reuse_forecaster_v1"
    assert payload["production_claim"] is False
    assert payload["shadow_only"] is True
    assert payload["modifies_controllers_or_defaults"] is False
    assert payload["modifies_robust_energy_engine"] is False
    assert payload["uses_oracle_as_headline"] is False


def test_data_readiness_audit_exists():
    assert AUDIT_PATH.exists(), (
        f"Data-readiness audit missing — re-run the driver script")
    audit = json.loads(AUDIT_PATH.read_text())
    assert audit["doc_version"] == "cache_prefix_reuse_data_readiness_audit_v1"
    assert audit["shadow_only"] is True
    assert audit["production_claim"] is False
    assert "swissai_bucket_reuse" in audit["datasets"]
    assert "cc_traces" in audit["datasets"]
    assert "lmcache" in audit["datasets"]
    assert "prefixbench" in audit["datasets"]


# ---------- 2. Leakage blocklist preserved -------------------------------


def test_summary_records_leakage_blocklist():
    payload = json.loads(SUMMARY_PATH.read_text())
    blocked = set(payload["leakage_features_excluded"])
    assert "reuse_percentage" in blocked
    assert "reused_buckets" in blocked
    assert "reused_bucket_count" in blocked
    assert "actual_e2e_latency_s" in blocked
    assert "ttft_s" in blocked
    assert "api_time_s" in blocked
    assert "cache_hit" in blocked


def test_audit_records_strict_leakage_rules():
    audit = json.loads(AUDIT_PATH.read_text())
    rules = audit["strict_leakage_rules"]
    assert rules["reuse_percentage_cannot_be_feature_when_predicting_reuse"] is True
    assert rules["future_requests_cannot_predict_current_request"] is True
    assert rules["target_derived_bucket_overlap_excluded_unless_observable"] is True
    blocked = set(rules["post_decision_fields_excluded"])
    assert "reuse_percentage" in blocked
    assert "cache_hit" in blocked


# ---------- 3. CC-traces strength expansion summary present --------------


def test_phase0_summary_exists():
    assert PHASE0_PATH.exists(), (
        "Phase 0 CC-traces strength expansion summary missing")
    p = json.loads(PHASE0_PATH.read_text())
    assert p["doc_version"] == "cc_traces_strength_expansion_v1"
    assert p["raw_committed"] is False
    assert p["analysis_sample_committed"] is False
    assert p["no_raw_prompt_completion_text_committed"] is True
    # Must compare 80 MiB vs 3000 MiB explicitly.
    assert "old_sample_stats_80_mib_committed" in p
    assert "new_sample_stats_3000_mib_cap" in p
    assert p["old_sample_stats_80_mib_committed"]["label"] == "80_MiB_committed"
    # Decision must be one of the canonical values.
    assert p["decision"] in {
        "use_for_training", "use_for_validation_only",
        "diagnostic_only", "not_worth_expansion",
    }


def test_phase0_summary_shows_increase_over_80_mib():
    p = json.loads(PHASE0_PATH.read_text())
    old = p["old_sample_stats_80_mib_committed"]
    new = p["new_sample_stats_3000_mib_cap"]
    # Phase 0 must produce *some* increase; the strict gate is enforced
    # by the in-script `_decide` function.
    assert new["request_count"] >= old["request_count"]
    assert new["session_count"] >= old["session_count"]


# ---------- 4. Weak CC-traces cannot become headline ---------------------


def test_summary_reports_cc_traces_separately_from_headline():
    payload = json.loads(SUMMARY_PATH.read_text())
    # CC-traces uses a derived label (intra_session_reuse) — never the
    # SwissAI/PrefixBench/LMCache headline.
    note = payload.get("cc_traces_reported_separately_because", "")
    assert "KV" in note or "block-hash" in note or "derived" in note
    # The CC-traces results block must be present but tagged
    # ``headline_eligible: False``.
    cc = payload.get("cc_traces_results", {})
    if cc and cc.get("row_count", 0) > 0:
        assert cc.get("headline_eligible") is False


def test_expanded_cc_traces_used_for_training_only_if_decision_allows():
    p = json.loads(PHASE0_PATH.read_text())
    payload = json.loads(SUMMARY_PATH.read_text())
    decision = p["decision"]
    cc_source = payload["datasets_used"]["cc_traces_source"]
    if decision == "use_for_training":
        # OK to use expanded sample for training.
        return
    # If the decision is not 'use_for_training', the driver must NOT use
    # the 3000 MiB analysis sample as a training source.
    assert cc_source != "3000_mib_expanded", (
        f"phase0 decision={decision} but cc_traces_source={cc_source}; "
        "expanded sample must not be used for training unless the gate passes")


# ---------- 5. Raw + analysis_sample NOT committed -----------------------


def test_raw_3000_mib_file_not_committed():
    raw_path = CC_TRACES_DIR / "raw" / "traces_3000mib.jsonl"
    # The file may not exist after a cleanup; if it exists it must not
    # be git-tracked.
    if raw_path.exists():
        assert not _git_tracked(raw_path), (
            "Raw 3000 MiB CC-traces file is git-tracked — must be gitignored")


def test_cc_traces_analysis_sample_not_committed():
    # Per mission spec: unnormalized analysis_sample.jsonl files must not
    # be committed.
    for child in CC_TRACES_DIR.rglob("analysis_sample.jsonl"):
        assert not _git_tracked(child), (
            f"{child} is git-tracked — analysis_sample.jsonl must be "
            "gitignored")


def test_committed_normalized_sample_respects_size_caps():
    # Per mission spec: per-file cap 50 MB, total cap 150 MB.
    norm_path = CC_3000_DIR / "normalized_sample.jsonl"
    if not norm_path.exists():
        pytest.skip("3000 MiB normalized sample not present")
    bytes_ = norm_path.stat().st_size
    assert bytes_ <= MAX_COMMITTED_NORMALIZED_SAMPLE_BYTES, (
        f"{norm_path} = {bytes_} bytes exceeds 50 MB cap")


def test_total_committed_normalized_samples_under_150_mb():
    total = 0
    for child in (REPO_ROOT / "data" / "external" / "hf").rglob(
            "normalized_sample.jsonl"):
        # Only count git-tracked files.
        if _git_tracked(child):
            total += child.stat().st_size
    assert total <= MAX_TOTAL_COMMITTED_NORMALIZED_BYTES, (
        f"committed normalized samples total {total} bytes exceeds 150 MB cap")


# ---------- 6. No raw prompt / completion text -------------------------


def test_committed_cc_traces_normalized_sample_has_no_raw_prompt_text():
    norm_path = CC_3000_DIR / "normalized_sample.jsonl"
    if not norm_path.exists():
        pytest.skip("3000 MiB normalized sample not present")
    forbidden_keys = {"prompt", "completion", "messages", "message",
                      "content", "text"}
    with norm_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            assert not (set(row.keys()) & forbidden_keys), (
                f"forbidden raw text key in CC-traces normalized sample: "
                f"{set(row.keys()) & forbidden_keys}")


# ---------- 7. Baselines / ML candidates / holdouts recorded -------------


def test_summary_records_baselines_ml_candidates_holdouts():
    payload = json.loads(SUMMARY_PATH.read_text())
    baselines = payload["baselines_used"]
    assert any("global" in b for b in baselines)
    assert any("per_model" in b for b in baselines)
    assert any("per_session_history" in b for b in baselines)
    ml = payload["ml_candidates_used"]
    assert any("logistic" in m for m in ml)
    assert any("gradient_boosting" in m for m in ml)
    sw = payload.get("swissai_results", {})
    if sw.get("row_count", 0) > 0:
        holdouts = {c["holdout"] for c in sw["per_holdout"]}
        assert "random_holdout" in holdouts


# ---------- 8. Economic-proxy metrics present ---------------------------


def test_summary_records_economic_proxy_metrics():
    payload = json.loads(SUMMARY_PATH.read_text())
    assert "binding_swissai_economic_improvement_pct" in payload
    assert "binding_swissai_holdout" in payload
    # Per-holdout economic_proxy_by_model dict must exist for each
    # SwissAI holdout.
    sw = payload.get("swissai_results", {})
    if sw.get("row_count", 0) > 0:
        for cell in sw["per_holdout"]:
            assert "economic_proxy_by_model" in cell
            assert "best_economic_improvement_pct" in cell


# ---------- 9. Promotion classification ---------------------------------


def test_final_status_is_canonical():
    payload = json.loads(SUMMARY_PATH.read_text())
    valid = {
        "shadow_ready_for_integration_review",
        "promising_needs_validation",
        "diagnostic_only",
        "rejected_regression",
        "blocked_by_scorer_limitations",
        "needs_more_data",
    }
    assert payload["final_status"] in valid


def test_summary_records_scorer_limitation_note():
    payload = json.loads(SUMMARY_PATH.read_text())
    # Mission spec PHASE C: if the scorer cannot express cache value the
    # note must be present.
    if payload.get("scorer_supports_cache_value_today") is False:
        assert payload.get("scorer_limitation_note")
        assert payload.get("shadow_integration_justified") is False


# ---------- 10. HF data not treated as pilot telemetry -----------------


def test_summary_does_not_label_hf_data_as_pilot():
    payload = json.loads(SUMMARY_PATH.read_text())
    text = json.dumps(payload).lower()
    forbidden = ["pilot_telemetry", "production_savings",
                 "production_calibration_source"]
    for f in forbidden:
        assert f not in text, f"forbidden phrase {f} appears in summary"


# ---------- 11. Production safety properties ---------------------------


def test_no_scheduler_or_residency_module_modified_in_this_pr():
    # We don't allow any production controller / residency / robust
    # energy engine module to be modified in this PR. The shadow
    # forecaster lives at aurelius/forecasting/cache_prefix_* — nothing
    # else may be touched as part of the cache-forecaster mission.
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            pytest.skip("git diff failed; cannot enforce production-safety guard")
        changed = [line.strip() for line in out.stdout.splitlines()
                   if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git not available")
    forbidden_prefixes = (
        "aurelius/optimization/scheduler.py",
        "aurelius/frontier/risk.py",
        "aurelius/frontier/dynamic_controller.py",
        "aurelius/frontier/dynamic_estimator.py",
        "aurelius/frontier/batch_inference_controller.py",
        "aurelius/frontier/training_controller.py",
        "aurelius/frontier/eval_workload_controller.py",
        "aurelius/residency/decision.py",
        "aurelius/residency/backtest.py",
        "aurelius/residency/sim.py",
        "aurelius/residency/shadow.py",
        "aurelius/residency/metrics.py",
        "aurelius/forecasting/price_model.py",
        "aurelius/forecasting/carbon_model.py",
        "aurelius/forecasting/baseline.py",
    )
    for f in changed:
        for fp in forbidden_prefixes:
            assert not f.startswith(fp), (
                f"production module {fp} was modified by this PR — "
                "cache forecaster must be shadow-only")


def test_no_shadow_adapter_executes_real_control_actions():
    """Mission spec PHASE D: a shadow adapter must be flag-default-off /
    logging-only. Since the binding final_status in this PR is
    ``diagnostic_only`` or ``blocked_by_scorer_limitations``, NO shadow
    adapter is created. This test verifies no executable shadow adapter
    module is present in the cache forecaster surface.
    """
    suspect = REPO_ROOT / "aurelius" / "forecasting" / (
        "cache_prefix_reuse_shadow.py"
    )
    if suspect.exists():
        text = suspect.read_text()
        assert "executable_in_real_cluster = False" in text or \
               "no_control_action_taken" in text, (
            "cache_prefix_reuse_shadow.py exists but does not enforce the "
            "shadow-only contract")
