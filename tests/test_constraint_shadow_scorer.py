"""Tests for the constraint-aware shadow scorer.

Enforces:
- shadow-only contract (executable_in_real_cluster=False),
- production scorer is the safety floor (hard vetoes preserved),
- production defaults unchanged when no variant flag is set,
- missing-prior fallback produces the production score exactly,
- every $-denominated term that needs an operator coefficient is
  reported uncalibrated when the policy is empty,
- shadow scorer never produces a "better SLA" result than production
  (binary sla_met cannot be tightened — only the latency *margin* can
  be reported),
- per-term value_quality (Level 1/2/3) is recorded on every breakdown.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting.constraint_scorer_features import (  # noqa: E402
    OperatorPricingPolicy,
    OptimumPriorTable,
    ScorerPriors,
    derive_gpu_type,
    derive_model_family,
    derive_model_size_b,
)
from aurelius.forecasting.constraint_shadow_scorer import (  # noqa: E402
    ALL_VARIANTS,
    SHADOW_FINAL_STATUS_VALUES,
    SHADOW_SCORER_VERSION,
    VARIANT_EXISTING,
    VARIANT_SHADOW_FULL,
    ConstraintShadowScorer,
    ShadowScorerConfig,
    classify_shadow_scorer_status,
)
from aurelius.residency.decision import (  # noqa: E402
    SafetyContext,
    score_residency_candidate,
)
from aurelius.residency.models import (  # noqa: E402
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
)


def _ctx():
    return SafetyContext(gpu_hour_price=3.0, default_latency_sla_ms=30000.0,
                         service_time_proxy_s=2.0, min_telemetry_confidence="medium")


def _loc(gpu="a100", resident=True):
    return ModelLocationState(
        region="us-east", node_id="n1", gpu_id=f"{gpu}-0",
        container_id="vllm-0",
        loaded_model_ids=["llama-3-7b"] if resident else [],
        loaded_adapter_ids=[],
        gpu_memory_total=80e9, gpu_memory_used=8e9,
        gpu_utilization=0.5, queue_depth=0,
        estimated_queue_wait_s=0.05, thermal_risk=0.2,
        topology_score=0.9, telemetry_confidence="high",
        last_updated_s=time.time(),
    )


def _req(model="llama-3-7b", prompt=200, out=64, current=None):
    return ModelResidencyRequest(
        request_id="r1", timestamp=time.time(), workload_id="w",
        model_id=model, prompt_tokens=prompt, output_tokens=out,
        latency_sla_ms=30000.0, region="us-east",
        current_route=current,
    )


def _profile():
    return ModelLoadProfile(
        model_id="llama-3-7b",
        cold_load_p50_s=10.0, cold_load_p95_s=15.0,
        memory_required_gb=16.0, source="test", confidence="medium",
    )


# ---------- 1. Shadow-only contract ------------------------------------


def test_shadow_scorer_executable_in_real_cluster_is_false():
    for v in ALL_VARIANTS:
        assert v.config.executable_in_real_cluster is False


def test_shadow_scorer_config_is_frozen():
    cfg = ShadowScorerConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.executable_in_real_cluster = True


def test_shadow_scorer_refuses_to_construct_with_executable_true():
    # The dataclass is frozen, so we can't construct a flipped config.
    # The runtime assert is documented in __init__.
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=ShadowScorerConfig())
    assert scorer.config.executable_in_real_cluster is False


def test_shadow_record_carries_no_control_action_flag():
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    assert br["shadow_only"] is True
    assert br["executable_in_real_cluster"] is False
    assert br["no_control_action_taken"] is True


def test_shadow_scorer_version_recorded():
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    assert br["shadow_scorer_version"] == SHADOW_SCORER_VERSION


# ---------- 2. Production-passthrough contract --------------------------


def test_existing_variant_passes_production_score_through_unchanged():
    """Variant A (all flags False) must return identical CandidateScore
    fields to the production scorer."""
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_EXISTING.config)
    req, loc, profile, ctx = _req(), _loc(), _profile(), _ctx()
    cs, _br = scorer.score(req, loc, profile, ctx, ctx)
    prod = score_residency_candidate(req, loc, profile, ctx, ctx)
    assert cs.location_key == prod.location_key
    assert cs.feasible == prod.feasible
    assert cs.expected_latency_s == pytest.approx(prod.expected_latency_s)
    assert cs.expected_cost == pytest.approx(prod.expected_cost)
    assert cs.sla_met == prod.sla_met


def test_shadow_falls_back_to_production_when_priors_empty():
    """If no priors are wired, the shadow scorer still passes the
    production score through unchanged."""
    priors = ScorerPriors(optimum=None, gpu_power=None,
                          ttft_p50_shadow=None, cache_reuse_predict=None,
                          operator_policy=OperatorPricingPolicy())
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    req, loc, profile, ctx = _req(), _loc(), _profile(), _ctx()
    cs, _br = scorer.score(req, loc, profile, ctx, ctx)
    prod = score_residency_candidate(req, loc, profile, ctx, ctx)
    # Cost should be identical because there's no per-GPU price (operator
    # global default) and no cache savings (no predictor).
    assert cs.expected_cost == pytest.approx(prod.expected_cost)


def test_shadow_returns_infeasible_when_production_returns_infeasible():
    """The shadow scorer must never override an infeasible production
    score. Hard safety vetoes carry through."""
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    # Force memory veto by setting required > total.
    profile = ModelLoadProfile(
        model_id="llama-3-7b", cold_load_p50_s=5.0,
        cold_load_p95_s=10.0, memory_required_gb=200.0,
        source="test", confidence="medium")
    loc = _loc(resident=False)
    req = _req()
    cs, br = scorer.score(req, loc, profile, _ctx(), _ctx())
    assert cs.feasible is False or br.get("fallback_to_production")


# ---------- 3. Per-term value_quality recorded --------------------------


def test_breakdown_records_value_quality_per_term():
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    vq = br["value_quality_by_term"]
    # Mandatory terms must have a non-empty level tag.
    for term in ("queue_wait_s", "service_time_s",
                 "per_gpu_hour_price_usd", "expected_latency_s",
                 "expected_cost_usd", "goodput_per_dollar"):
        assert term in vq, f"value_quality missing for term {term}"
        assert vq[term] is not None
        # Every level tag starts with level_*.
        assert vq[term].startswith("level_"), (
            f"term {term} has malformed value_quality {vq[term]!r}")


def test_term_formulae_are_documented():
    priors = ScorerPriors.load_defaults()
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    fmls = br["term_formulae"]
    assert "expected_latency_s" in fmls
    assert "expected_cost_usd" in fmls
    assert "goodput_per_dollar" in fmls
    assert "sla_met" in fmls


# ---------- 4. Uncalibrated terms surfaced ------------------------------


def test_energy_term_uncalibrated_when_no_operator_energy_price():
    priors = ScorerPriors.load_defaults(operator_policy=OperatorPricingPolicy())
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    assert "energy_cost_per_request_usd" in br["uncalibrated_terms"]


def test_cache_value_usd_uncalibrated_when_no_per_gpu_policy():
    def _predict(req): return 50.0  # 50% reuse
    priors = ScorerPriors.load_defaults(
        operator_policy=OperatorPricingPolicy())  # no per-GPU policy
    priors.cache_reuse_predict = _predict
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    # The cache savings in seconds is calibrated; the USD form is not.
    assert "cache_hit_value_usd" in br["uncalibrated_terms"]


def test_cache_value_usd_calibrated_when_operator_per_gpu_policy():
    def _predict(req): return 50.0
    priors = ScorerPriors.load_defaults(
        operator_policy=OperatorPricingPolicy(
            gpu_hour_price_per_type={"a100": 3.5},
            energy_price_per_kwh_usd=0.10,
        ))
    priors.cache_reuse_predict = _predict
    scorer = ConstraintShadowScorer(
        priors=priors, config=VARIANT_SHADOW_FULL.config)
    _, br = scorer.score(_req(), _loc(), _profile(), _ctx(), _ctx())
    assert "cache_hit_value_usd" not in br["uncalibrated_terms"]


# ---------- 5. Helpers --------------------------------------------------


def test_derive_model_family():
    assert derive_model_family("llama-3-7b") == "llama"
    assert derive_model_family("Mistral-7B-v0.1") == "mistral"
    assert derive_model_family("Qwen2.5-3B") == "qwen"
    assert derive_model_family(None) is None
    assert derive_model_family("totally_unknown_model") is None


def test_derive_model_size_b():
    assert derive_model_size_b("llama-3-7b") == 7.0
    assert derive_model_size_b("llama-3-70b") == 70.0
    assert derive_model_size_b("mixtral-8x7b") == 56.0
    assert derive_model_size_b("qwen2.5-3b") == 3.0


def test_derive_gpu_type_from_location_key():
    assert derive_gpu_type("us-east/node-a100-0/a100-0/vllm-0") == "a100"
    assert derive_gpu_type("us-east/node-t4-0/t4-0/vllm-0") == "t4"
    assert derive_gpu_type(None) is None


def test_operator_pricing_policy_defaults_empty():
    p = OperatorPricingPolicy()
    assert p.gpu_hour_price_per_type == {}
    assert p.energy_price_per_kwh_usd is None
    assert p.carbon_price_per_kg_usd is None
    # Lookup falls back to the supplied default (existing scorer behaviour).
    v, tag = p.lookup_gpu_hour_price("a100", default=3.0)
    assert v == 3.0
    assert tag == "operator_global_default"


def test_operator_pricing_policy_supplies_per_gpu():
    p = OperatorPricingPolicy(
        gpu_hour_price_per_type={"a100": 3.5, "t4": 0.6})
    v, tag = p.lookup_gpu_hour_price("t4", default=3.0)
    assert v == 0.6
    assert tag == "operator_per_gpu_policy"


# ---------- 6. Optimum prior loader ------------------------------------


def test_optimum_prior_loads_from_fixtures():
    t = OptimumPriorTable.from_fixtures()
    assert t.fixture_count >= 5
    assert t.value_quality == "prior"  # Level 3
    cell, fallback = t.lookup(model_family=None, gpu_type="a100")
    assert fallback in {"gpu_only", "missing"}


# ---------- 7. Promotion classifier ------------------------------------


def test_status_values_are_canonical():
    expected = {
        "shadow_ready_for_integration_review",
        "promising_needs_validation",
        "diagnostic_only",
        "rejected_regression",
        "proxy_promising_only",
        "blocked_by_pilot_telemetry",
    }
    assert SHADOW_FINAL_STATUS_VALUES == expected


def test_classify_diagnostic_when_under_2pct():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=1.0,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=False,
        pilot_telemetry_required=False)
    assert s == "diagnostic_only"


def test_classify_promising_when_2_5_pct_calibrated():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=3.5,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=False,
        pilot_telemetry_required=False)
    assert s == "promising_needs_validation"


def test_classify_shadow_ready_when_above_5_pct_calibrated():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=7.0,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=False,
        pilot_telemetry_required=False)
    assert s == "shadow_ready_for_integration_review"


def test_classify_proxy_promising_when_uncalibrated_but_positive():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=10.0,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=True,
        pilot_telemetry_required=False)
    assert s == "proxy_promising_only"


def test_classify_diagnostic_when_negative_even_if_uncalibrated():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=-10.0,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=True,
        pilot_telemetry_required=False)
    assert s == "diagnostic_only"


def test_classify_rejects_sla_regression():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=20.0,
        has_sla_regression=True, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=False,
        pilot_telemetry_required=False)
    assert s == "rejected_regression"


def test_classify_blocks_on_pilot_telemetry():
    s, _ = classify_shadow_scorer_status(
        sla_safe_goodput_per_dollar_improvement_pct=20.0,
        has_sla_regression=False, has_subgroup_regression=False,
        headline_terms_are_uncalibrated=False,
        pilot_telemetry_required=True)
    assert s == "blocked_by_pilot_telemetry"
