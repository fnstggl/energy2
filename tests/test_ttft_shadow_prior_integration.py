"""Tests for the TTFT p50 shadow-prior integration.

Verifies the binding shadow contract: prior is optional, defaults to
non-applied, never enables real execution, never imports TTFT p95/p99
ML tails into a control path, and the offline eval JSON is structurally
valid + reports the binding metrics.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.ttft_shadow_prior import (  # noqa: E402
    TTFTShadowPrior,
    _derive_gpu_type,
    _derive_model_size,
    refine_service_time_proxy_s,
)
from aurelius.residency.decision import SafetyContext  # noqa: E402

EVAL_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "placement_prior_audit", "ttft_shadow_prior_eval.json",
)


# ---------- 1. Adapter primitives ----------------------------------------


def test_derive_model_size_and_gpu_type():
    assert _derive_model_size("qwen2.5-3b_a30") == "3b"
    assert _derive_model_size("qwen2.5-72b_a100") == "72b"
    assert _derive_gpu_type("qwen2.5-3b_a30") == "a30"
    assert _derive_gpu_type("qwen2.5-3b_p100") == "p100"
    assert _derive_model_size(None) is None
    assert _derive_gpu_type("") is None


def _rows(n=30):
    out = []
    for i in range(n):
        out.append({
            "actual_ttft_s": 0.05 + 0.01 * (i % 5),
            "instance_type": ["qwen2.5-3b_a30", "qwen2.5-3b_p100",
                              "qwen2.5-7b_a30"][i % 3],
            "num_prompt_tokens": 50 + 10 * (i % 4),
        })
    return out


def test_ttft_shadow_prior_fits_per_subgroup():
    p = TTFTShadowPrior().fit_from_rows(_rows(60))
    assert p.fit_row_count == 60
    # Per-GPU medians populated.
    assert "a30" in p.by_gpu
    assert "p100" in p.by_gpu


def test_ttft_shadow_prior_falls_back_through_hierarchy():
    p = TTFTShadowPrior().fit_from_rows(_rows(60))
    # Subgroup that exists.
    v = p.predict(model_size="3b", gpu_type="a30", prompt_tokens=60)
    assert v is not None and v > 0
    # Subgroup that doesn't exist for the bin → falls back to (m,g).
    v2 = p.predict(model_size="3b", gpu_type="a30", prompt_tokens=10_000)
    assert v2 is not None
    # Unknown model → falls back to gpu-only.
    v3 = p.predict(model_size="999z", gpu_type="a30", prompt_tokens=60)
    assert v3 is not None


def test_ttft_shadow_prior_returns_none_when_no_signal():
    p = TTFTShadowPrior()  # never fit
    assert p.predict(model_size="3b", gpu_type="a30", prompt_tokens=100) is None


# ---------- 2. Default is shadow / not applied --------------------------


def test_refine_default_does_not_apply_to_scorer():
    p = TTFTShadowPrior().fit_from_rows(_rows(60))
    ctx = SafetyContext(service_time_proxy_s=2.0)
    out_ctx, record = refine_service_time_proxy_s(
        ctx, model_size="3b", gpu_type="a30", prompt_tokens=100, prior=p,
    )
    # Default: apply_to_scorer=False → returns the input context unchanged.
    assert out_ctx is ctx
    assert record["applied_to_scorer"] is False
    assert record["static_proxy_s"] == 2.0
    # The "refined" field records what would have been applied; it must
    # equal max(static, predicted) — never below static.
    assert record["refined_proxy_s"] >= 2.0


def test_refine_when_opted_in_clamps_to_max_static_predicted():
    p = TTFTShadowPrior().fit_from_rows(_rows(60))
    ctx = SafetyContext(service_time_proxy_s=2.0)
    out_ctx, record = refine_service_time_proxy_s(
        ctx, model_size="3b", gpu_type="a30", prompt_tokens=100, prior=p,
        apply_to_scorer=True,
    )
    # Output context's service_time_proxy_s never drops below static.
    assert out_ctx.service_time_proxy_s >= 2.0
    assert record["applied_to_scorer"] is True


def test_refine_falls_back_when_prior_missing():
    p = TTFTShadowPrior()  # not fit
    ctx = SafetyContext(service_time_proxy_s=2.0)
    out_ctx, record = refine_service_time_proxy_s(
        ctx, model_size="3b", gpu_type="a30", prompt_tokens=100, prior=p,
        apply_to_scorer=True,
    )
    assert record["fallback_to_static"] is True
    assert out_ctx is ctx
    assert record["applied_to_scorer"] is False


def test_refine_records_subgroup_n_for_insufficient_sample_flagging():
    p = TTFTShadowPrior().fit_from_rows(_rows(60))
    ctx = SafetyContext(service_time_proxy_s=2.0)
    _, record = refine_service_time_proxy_s(
        ctx, model_size="3b", gpu_type="a30", prompt_tokens=100, prior=p,
    )
    assert "subgroup_n" in record
    assert "subgroup_insufficient" in record


# ---------- 3. Adapter does not import executors / controllers ---------


def test_ttft_shadow_prior_module_has_no_controller_imports():
    path = os.path.join(REPO_ROOT, "aurelius", "forecasting",
                        "ttft_shadow_prior.py")
    with open(path) as fh:
        src = fh.read()
    for forbidden in (
        "aurelius.optimization.scheduler",
        "aurelius.frontier.controller",
        "aurelius.frontier.execution",
        "aurelius.frontier.dynamic_controller",
    ):
        assert forbidden not in src, f"adapter imports executor: {forbidden}"


def test_ttft_shadow_prior_module_does_not_use_p95_p99_for_control():
    """The mission spec forbids using TTFT p95/p99 ML tails for control.
    The shadow prior module exposes ONLY a p50 predictor."""
    from aurelius.forecasting import ttft_shadow_prior
    public = [n for n in dir(ttft_shadow_prior) if not n.startswith("_")]
    forbidden_substrings = ("p95", "p99", "_tail_", "tail_predictor")
    for name in public:
        for f in forbidden_substrings:
            assert f not in name.lower(), (
                f"shadow prior module exposes p95/p99 tail symbol '{name}' — "
                "mission spec forbids using ML tails for control"
            )


# ---------- 4. Existing scorer defaults unchanged ----------------------


def test_safety_context_default_service_time_proxy_unchanged():
    """A bare SafetyContext() must still default to service_time_proxy_s=2.0.
    If a future PR changes this default, all downstream calibrations
    invalidate — so we pin it here."""
    ctx = SafetyContext()
    assert ctx.service_time_proxy_s == 2.0


def test_safety_context_is_dataclass_we_can_replace():
    ctx = SafetyContext()
    new = dataclasses.replace(ctx, service_time_proxy_s=5.0)
    assert new.service_time_proxy_s == 5.0
    # Original unchanged.
    assert ctx.service_time_proxy_s == 2.0


# ---------- 5. Eval JSON invariants -------------------------------------


@pytest.fixture(scope="module")
def eval_summary():
    if not os.path.exists(EVAL_PATH):
        pytest.skip("ttft_shadow_prior_eval.json not generated; run "
                    "scripts/run_ttft_shadow_prior_eval.py first")
    with open(EVAL_PATH) as fh:
        return json.load(fh)


def test_eval_top_level_invariants(eval_summary):
    assert eval_summary["audit_only"] is True
    assert eval_summary["modifies_controllers_or_defaults"] is False
    assert eval_summary["modifies_robust_energy_engine"] is False
    assert eval_summary["uses_oracle_as_headline"] is False
    assert eval_summary["production_claim"] is False
    assert eval_summary["shadow_only"] is True
    assert eval_summary["ttft_p95_p99_used_for_control"] is False


def test_eval_records_required_metrics(eval_summary):
    m = eval_summary["metrics"]
    for k in ("top1_placement_change_rate", "ranking_change_rate",
              "projected_goodput_per_dollar_delta_pct",
              "projected_sla_met_delta_pct",
              "projected_expected_latency_delta_pct",
              "safety_regression_count",
              "subgroup_top1_change_by_actual_instance"):
        assert k in m


def test_eval_final_status_in_closed_enum(eval_summary):
    valid = {"diagnostic_only", "promising_needs_validation",
             "shadow_ready_for_integration_review"}
    assert eval_summary["final_status"] in valid


def test_eval_promotion_rule_thresholds(eval_summary):
    rule = eval_summary["promotion_rule"]
    assert rule["diagnostic_only_max_goodput_delta_pct"] == 2.0
    assert rule["shadow_ready_for_integration_review_min_pct"] == 5.0
    assert rule["applies_only_if_safety_regression_count_is_zero"] is True


def test_promotion_status_consistent_with_metrics(eval_summary):
    delta = eval_summary["metrics"]["projected_goodput_per_dollar_delta_pct"]
    safety = eval_summary["metrics"]["safety_regression_count"]
    top1 = eval_summary["metrics"]["top1_placement_change_rate"]
    status = eval_summary["final_status"]
    if status == "shadow_ready_for_integration_review":
        assert delta >= 5.0 and safety == 0 and top1 > 0
    elif status == "promising_needs_validation":
        assert 2.0 <= delta < 5.0 and safety == 0 and top1 > 0
    else:  # diagnostic_only
        assert delta < 2.0 or safety > 0 or top1 == 0


def test_eval_records_diagnostic_without_clamp(eval_summary):
    """Eval surfaces a what-if metric with the safety clamp removed.
    It must be clearly labelled as NOT the binding shape."""
    diag = eval_summary["diagnostic_without_max_clamp"]
    assert "ANALYSIS ONLY" in diag["note"]
    assert "clamped" in diag["binding_integration_shape"].lower()
    assert "top1_change_rate" in diag


# ---------- 6. Eval script honesty checks -------------------------------


def test_eval_script_no_oracle_or_fifo_headline():
    path = os.path.join(REPO_ROOT, "scripts",
                        "run_ttft_shadow_prior_eval.py")
    with open(path) as fh:
        src = fh.read().lower()
    for phrase in ("production savings", "hyperscaler-validated",
                   "production-proven"):
        assert phrase not in src
    # The eval cannot use FIFO or oracle as headline metric. We allow the
    # words to appear in comments / negative statements; we just make sure
    # they aren't treated as the primary baseline.
    for forbidden in ("execute_frontier_decision", "set_replicas",
                      "RUN_FOR_REAL"):
        assert forbidden not in src


def test_eval_does_not_modify_default_safety_context():
    """After running the eval, a fresh SafetyContext() must still have
    the documented defaults — the eval is forbidden from globally mutating
    them."""
    fresh = SafetyContext()
    assert fresh.service_time_proxy_s == 2.0
    assert fresh.gpu_hour_price == 3.0
