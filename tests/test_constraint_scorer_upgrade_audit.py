"""Tests for the constraint-aware scorer-upgrade audit JSON.

Enforces the mission-spec binding rules:
- shadow-only flags present,
- every input is signal-level-classified,
- no invented economic constants are stored anywhere in the audit,
- every $-denominated coefficient traces to operator policy or has
  ``invented_constants_introduced = False`` recorded,
- production scorer signature is captured,
- term coverage matrix lists every catalogued term × variant.
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

AUDIT_PATH = (REPO_ROOT / "data" / "external" / "forecasting"
              / "constraint_scorer_upgrade" / "scorer_path_audit.json")
TERM_COVERAGE_PATH = (REPO_ROOT / "data" / "external" / "forecasting"
                      / "constraint_scorer_upgrade" / "term_coverage_matrix.json")


# ---------- 1. Audit JSON exists + carries binding flags ----------------


def test_audit_json_exists():
    assert AUDIT_PATH.exists(), (
        "scorer_path_audit.json missing — re-run "
        "scripts/run_constraint_scorer_upgrade_audit.py")
    payload = json.loads(AUDIT_PATH.read_text())
    assert payload["doc_version"] == "constraint_scorer_upgrade_audit_v1"
    assert payload["audit_only"] is True
    assert payload["shadow_only"] is True
    assert payload["production_claim"] is False
    assert payload["modifies_controllers_or_defaults"] is False
    assert payload["modifies_robust_energy_engine"] is False
    assert payload["uses_oracle_as_headline"] is False


def test_term_coverage_matrix_exists():
    assert TERM_COVERAGE_PATH.exists()
    payload = json.loads(TERM_COVERAGE_PATH.read_text())
    assert payload["doc_version"] == "constraint_scorer_term_coverage_matrix_v1"
    assert payload["audit_only"] is True
    assert payload["shadow_only"] is True
    assert payload["production_claim"] is False


# ---------- 2. Signal hierarchy preserved -------------------------------


def test_audit_records_signal_hierarchy():
    payload = json.loads(AUDIT_PATH.read_text())
    hierarchy = payload["signal_hierarchy"]
    for level in ("level_1_measured", "level_2_derived",
                  "level_3_prior", "level_4_prohibited_or_uncalibrated"):
        assert level in hierarchy
        assert isinstance(hierarchy[level], str)
        assert len(hierarchy[level]) > 20  # non-trivial description


def test_every_input_is_signal_level_classified():
    payload = json.loads(AUDIT_PATH.read_text())
    for entry in payload["scoring_inputs_existing_vs_upgraded"]:
        # production_signal_level can be None (missing) but the field
        # must be present.
        assert "production_signal_level" in entry
        assert "upgraded_signal_level" in entry


def test_audit_includes_canonical_scorer_inputs():
    payload = json.loads(AUDIT_PATH.read_text())
    inputs = {e["input"] for e in payload["scoring_inputs_existing_vs_upgraded"]}
    must_have = {
        "TTFT", "TPOT", "E2E latency", "queue wait", "queue depth",
        "prefill cost", "decode cost", "cache-hit value", "prefix reuse",
        "cold-start penalty", "migration cache-loss penalty",
        "model load / unload", "GPU type", "model size",
        "VRAM / memory pressure", "GPU utilization", "energy per request",
        "power draw", "cloud cost", "energy cost ($)", "carbon cost ($)",
        "SLA risk", "timeout risk",
    }
    missing = must_have - inputs
    assert not missing, f"audit missing canonical inputs: {missing}"


# ---------- 3. Dollar coefficient audit ---------------------------------


def test_no_invented_dollar_coefficients_anywhere_in_audit():
    payload = json.loads(AUDIT_PATH.read_text())
    coefs = payload["dollar_coefficient_calibration_sources"]
    assert len(coefs) >= 3, "expected at least gpu_hour / energy / carbon coefs"
    for coef in coefs:
        assert coef.get("invented_constants_introduced") is False, (
            f"coefficient {coef.get('coefficient')!r} flagged as invented")


def test_audit_explicitly_records_no_cache_value_weight_constant():
    payload = json.loads(AUDIT_PATH.read_text())
    coefs = {c["coefficient"]: c
             for c in payload["dollar_coefficient_calibration_sources"]}
    assert "cache_value_weight" in coefs
    assert coefs["cache_value_weight"]["source"] == "DOES NOT EXIST IN THIS MODULE"
    assert coefs["cache_value_weight"]["invented_constants_introduced"] is False


def test_audit_explicitly_records_no_utility_weighted_composite():
    payload = json.loads(AUDIT_PATH.read_text())
    coefs = {c["coefficient"]: c
             for c in payload["dollar_coefficient_calibration_sources"]}
    assert "utility_weights_in_composite" in coefs
    assert coefs["utility_weights_in_composite"][
        "invented_constants_introduced"] is False


def test_no_invented_constants_in_feature_module():
    """Grep the feature module for invented-constant patterns."""
    src = (REPO_ROOT / "aurelius" / "forecasting"
           / "constraint_scorer_features.py").read_text()
    # Patterns that would be invented utility weights / penalties.
    forbidden = (
        "CACHE_VALUE =", "CACHE_WEIGHT =", "MIGRATION_PENALTY =",
        "UTILITY_SCORE", "0.4*latency", "0.4 * latency",
        "0.3*cache", "0.3 * cache",
    )
    for pat in forbidden:
        assert pat not in src, (
            f"feature module contains forbidden pattern {pat!r}")


def test_no_invented_constants_in_shadow_scorer_module():
    src = (REPO_ROOT / "aurelius" / "forecasting"
           / "constraint_shadow_scorer.py").read_text()
    forbidden = (
        "CACHE_VALUE =", "CACHE_WEIGHT =", "MIGRATION_PENALTY =",
        "UTILITY_SCORE", "0.4*latency", "0.4 * latency",
        "0.3*cache", "0.3 * cache",
    )
    for pat in forbidden:
        assert pat not in src, (
            f"shadow scorer module contains forbidden pattern {pat!r}")


# ---------- 4. Term coverage matrix --------------------------------------


def test_term_coverage_covers_five_variants():
    payload = json.loads(TERM_COVERAGE_PATH.read_text())
    expected = {
        "A_existing", "B_shadow_default_priors",
        "C_shadow_with_ttft_p50_prior",
        "D_shadow_with_cache_prefill", "E_shadow_full",
    }
    assert set(payload["variants"].keys()) == expected


def test_term_coverage_classifies_each_term_with_signal_level():
    payload = json.loads(TERM_COVERAGE_PATH.read_text())
    for variant_name, variant_doc in payload["variants"].items():
        for term, info in variant_doc["term_signal_levels"].items():
            assert "classification" in info, (
                f"term {term} in {variant_name} missing classification")
            # signal_level can be None for "missing" but the key must
            # exist.
            assert "signal_level" in info, (
                f"term {term} in {variant_name} missing signal_level")


def test_existing_variant_has_missing_terms_upgraded_variant_does_not():
    payload = json.loads(TERM_COVERAGE_PATH.read_text())
    existing = payload["variants"]["A_existing"]["term_coverage"]
    upgraded = payload["variants"]["E_shadow_full"]["term_coverage"]
    # Headline: per_gpu_hour_price was static_global_default in
    # existing, becomes operator_policy_or_operator_global_default in
    # upgraded.
    assert existing["per_gpu_hour_price"] == "static_global_default"
    assert upgraded["per_gpu_hour_price"] == (
        "operator_policy_or_operator_global_default")
    # Cache-hit-value: missing in existing, surfaced as the cache-prior
    # variant in shadow_full.
    assert existing["cache_hit_value"] == "missing"
    assert upgraded["cache_hit_value"] == (
        "forecasted_cache_prefix_reuse_v1_proxy")


def test_upgraded_variant_reclassifies_at_least_five_missing_to_derived():
    payload = json.loads(AUDIT_PATH.read_text())
    existing_counts = payload["counts_by_classification_existing"]
    upgraded_counts = payload["counts_by_classification_upgraded"]
    # Existing has many missings; upgraded should reduce them.
    n_missing_existing = existing_counts.get("missing", 0)
    n_missing_upgraded = upgraded_counts.get("missing", 0)
    assert n_missing_upgraded <= n_missing_existing - 5, (
        f"upgraded scorer didn't reduce missing-term count enough "
        f"(existing={n_missing_existing}, upgraded={n_missing_upgraded})")


# ---------- 5. Production safety ---------------------------------------


def test_no_production_module_modified_in_this_pr():
    """The audit + shadow scorer must not modify any production
    controller / scheduler / frontier / robust-energy module."""
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
    forbidden = (
        "aurelius/optimization/scheduler.py",
        "aurelius/optimization/objective.py",
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
        for fp in forbidden:
            assert not f.startswith(fp), (
                f"production module {fp} was modified — constraint "
                "scorer upgrade must be shadow-only")
