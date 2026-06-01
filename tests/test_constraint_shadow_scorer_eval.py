"""Tests for the constraint-aware shadow scorer evaluation summary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_PATH = (REPO_ROOT / "data" / "external" / "forecasting"
             / "constraint_scorer_upgrade" / "shadow_scorer_eval.json")


def _load():
    if not EVAL_PATH.exists():
        pytest.skip(
            "shadow_scorer_eval.json missing — re-run "
            "scripts/run_constraint_shadow_scorer_eval.py")
    return json.loads(EVAL_PATH.read_text())


# ---------- 1. Summary shape + binding flags ---------------------------


def test_eval_summary_exists_and_has_binding_flags():
    p = _load()
    assert p["doc_version"] == "constraint_shadow_scorer_eval_v1"
    assert p["shadow_only"] is True
    assert p["production_claim"] is False
    assert p["modifies_controllers_or_defaults"] is False
    assert p["modifies_robust_energy_engine"] is False
    assert p["uses_oracle_as_headline"] is False


def test_eval_runs_both_passes():
    p = _load()
    assert "pass_priors_only" in p
    assert "pass_with_operator_pricing" in p
    assert "headline" in p


def test_eval_records_honest_partitioning_note():
    p = _load()
    note = p["honest_partitioning_note"].lower()
    assert "do not invent economic constants" in note
    assert "operator" in note


# ---------- 2. All 5 variants present in BOTH passes -------------------


def test_each_pass_has_all_five_variants():
    p = _load()
    expected = {
        "A_existing", "B_shadow_default_priors",
        "C_shadow_with_ttft_p50_prior",
        "D_shadow_with_cache_prefill", "E_shadow_full",
    }
    for pass_key in ("pass_priors_only", "pass_with_operator_pricing"):
        assert set(p[pass_key]["aggregates"].keys()) == expected


# ---------- 3. Baseline (variant A) is the identity --------------------


def test_variant_A_existing_has_zero_change_rates():  # noqa: N802
    p = _load()
    for pass_key in ("pass_priors_only", "pass_with_operator_pricing"):
        a = p[pass_key]["aggregates"]["A_existing"]
        assert a["top1_change_rate"] == 0.0
        assert a["ranking_change_rate"] == 0.0
        assert a["sla_safe_goodput_per_dollar_improvement_pct"] == 0.0


# ---------- 4. SLA-safe count never regresses (binding contract) -------


def test_sla_safe_count_never_drops_below_baseline():
    p = _load()
    for pass_key in ("pass_priors_only", "pass_with_operator_pricing"):
        a = p[pass_key]["aggregates"]
        base_sla = a["A_existing"]["sla_safe_count"]
        for vname, vmetrics in a.items():
            assert vmetrics["sla_safe_count"] >= base_sla - 0, (
                f"variant {vname} ({pass_key}) drops SLA-safe count below "
                f"baseline {base_sla} -> {vmetrics['sla_safe_count']}")


# ---------- 5. Promotion classification --------------------------------


def test_final_status_values_canonical():
    p = _load()
    valid = {
        "shadow_ready_for_integration_review",
        "promising_needs_validation",
        "diagnostic_only",
        "rejected_regression",
        "proxy_promising_only",
        "blocked_by_pilot_telemetry",
    }
    assert p["pass_priors_only"]["final_status"] in valid
    assert p["pass_with_operator_pricing"]["final_status"] in valid
    assert p["headline"]["binding_status"] in valid


def test_binding_headline_is_priors_only_pass():
    p = _load()
    assert p["headline"]["binding_pass"] == "pass_priors_only"
    assert (p["headline"]["binding_status"]
            == p["pass_priors_only"]["final_status"])


# ---------- 6. Uncalibrated-term reporting ----------------------------


def test_uncalibrated_term_counts_present_per_variant():
    p = _load()
    for pass_key in ("pass_priors_only", "pass_with_operator_pricing"):
        for vname, agg in p[pass_key]["aggregates"].items():
            assert "uncalibrated_term_counts" in agg
            # Variant A_existing must not introduce any uncalibrated
            # terms (it doesn't run the shadow scorer).
            if vname == "A_existing":
                assert agg["uncalibrated_term_counts"] == {}


def test_priors_only_pass_records_uncalibrated_dollar_terms():
    p = _load()
    a = p["pass_priors_only"]["aggregates"]
    # When no operator policy is supplied, the energy USD term + cache
    # value USD term should appear uncalibrated in the shadow variants.
    e_uncal = a["E_shadow_full"]["uncalibrated_term_counts"]
    assert e_uncal.get("energy_cost_per_request_usd", 0) > 0


# ---------- 7. Pilot-only items recorded -------------------------------


def test_pilot_only_remaining_items_present():
    p = _load()
    items = p["pilot_only_remaining_items"]
    assert any("operator" in i.lower() and "energy_price" in i.lower()
               for i in items)
    assert any("per-gpu" in i.lower() and "/hr" in i.lower()
               for i in items)


# ---------- 8. No oracle / FIFO headline -------------------------------


def test_no_oracle_or_fifo_in_summary():
    p = _load()
    text = json.dumps(p).lower()
    forbidden = ["oracle_savings", "fifo_savings",
                 "production_savings_pct", "savings_dollars",
                 "savings_per_request_pct"]
    for f in forbidden:
        assert f not in text, (
            f"forbidden phrase {f!r} appears in eval summary")


def test_no_invented_economic_constants_in_eval_driver():
    src = (REPO_ROOT / "scripts"
           / "run_constraint_shadow_scorer_eval.py").read_text()
    # Patterns that would be invented utility coefficients.
    for pat in ("CACHE_VALUE =", "CACHE_WEIGHT =", "MIGRATION_PENALTY =",
                "UTILITY_SCORE", "0.4*latency", "0.4 * latency",
                "0.3*cache", "0.3 * cache"):
        assert pat not in src
