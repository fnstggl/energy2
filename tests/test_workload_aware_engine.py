"""Tests for the workload-aware action eligibility + economic-gating fix.

The mission is to stop the engine from scaling batch / flexible / best-effort
workloads for marginal queue relief — the bug that made constraint_aware lose
the energy scenario on the canonical KPI. These tests pin down each spec
invariant:

A. batch queue mild pressure → no scale-up
B. critical inference queue pressure → can scale
C. batch deadline risk → may scale (deadline-driven)
D. energy scenario regression → goodput/$ improves
E. no gaming → an action that adds tokens but kills goodput/$ is rejected
F. workload-class propagation → SimWorkload → ISS → engine; JSON round-trip
G. explanations → blocked action includes workload class + economic reason
H. no regressions → thermal/queue_surge/underutilization wins preserved
"""

from __future__ import annotations

from datetime import datetime, timezone

from aurelius.benchmarks import ConstraintBenchmarkRunner
from aurelius.constraints.engine import (
    _STRONG_SLA_RISK_SCORE,
    _scale_eligible_for_class,
    _workload_class,
)
from aurelius.simulation.cluster import ClusterSimulator, load_scenario
from aurelius.state.models import InferenceServiceState, Provenance

_T = datetime(2026, 5, 28, tzinfo=timezone.utc)
_PROV = Provenance(source="test", fetched_at=_T, confidence="high")


def _svc(**overrides) -> InferenceServiceState:
    base = dict(
        service_id="svc",
        engine="vllm",
        timestamp=_T,
        provenance=_PROV,
        region="us-east",
    )
    base.update(overrides)
    return InferenceServiceState(**base)


# ---------------------------------------------------------------------------
# F. Workload-class propagation
# ---------------------------------------------------------------------------

def test_workload_class_resolved_from_priority_tier():
    assert _workload_class(_svc(priority_tier="critical")) == "critical_interactive"
    assert _workload_class(_svc(priority_tier="latency_sensitive")) == "standard_interactive"
    assert _workload_class(_svc(priority_tier="standard")) == "standard_interactive"
    assert _workload_class(_svc(priority_tier="batch")) == "batch_inference"
    assert _workload_class(_svc(priority_tier="best_effort")) == "best_effort"


def test_workload_class_resolved_from_workload_type():
    # batch/training workload_type alone is enough to classify as batch.
    assert _workload_class(_svc(workload_type="batch_training")) == "batch_inference"
    assert _workload_class(_svc(workload_type="fine_tuning")) == "training"
    assert _workload_class(_svc(workload_type="embedding")) == "embedding_offline"


def test_flexible_is_shiftability_not_class():
    # A `flexible` inference service should still be treated as interactive —
    # `flexible` is a shiftability flag, not a workload class. A `flexible`
    # *batch_training* job is still batch.
    assert _workload_class(
        _svc(priority_tier="flexible", workload_type="inference")
    ) == "standard_interactive"
    assert _workload_class(
        _svc(priority_tier="flexible", workload_type="batch_training")
    ) == "batch_inference"


def test_workload_class_propagates_through_simulator_to_iss():
    sc = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    sim.run(steps=4)
    st = sim.get_cluster_state()
    services = list(st.all_services.values())
    assert services, "scenario should produce services"
    east = next(s for s in services if s.service_id == "batch-llm-east")
    assert east.workload_type == "batch_training"
    assert east.priority_tier == "batch"
    assert east.latency_sensitive is False


def test_inference_service_state_roundtrips_workload_class():
    svc = _svc(
        workload_type="batch_training",
        priority_tier="batch",
        latency_sensitive=False,
        flexibility="high",
        migration_allowed=True,
        latency_sla_p99_ms=5000.0,
        queue_sla_p95_ms=2000.0,
        sla_policy_id="batch-default",
        deadline_s=3600.0,
        flexibility_window_minutes=60.0,
    )
    d = svc.to_dict()
    svc2 = InferenceServiceState.from_dict(d)
    for f in (
        "workload_type", "priority_tier", "latency_sensitive", "flexibility",
        "migration_allowed", "latency_sla_p99_ms", "queue_sla_p95_ms",
        "sla_policy_id", "deadline_s", "flexibility_window_minutes",
    ):
        assert getattr(svc2, f) == getattr(svc, f), f


# ---------------------------------------------------------------------------
# A. Batch mild queue → no scale-up (the headline bug)
# ---------------------------------------------------------------------------

def test_batch_class_blocks_scale_for_mild_queue_relief():
    sla_risk = 0.30  # exactly the marginal score that triggered the bug
    eligible, reason = _scale_eligible_for_class(
        "batch_inference", sla_risk_score=sla_risk, has_deadline_risk=False,
    )
    assert eligible is False
    assert "blocked_scale_for_low_value_queue_relief" in reason
    assert "batch_inference" in reason


def test_best_effort_class_blocks_scale_for_mild_queue_relief():
    eligible, _ = _scale_eligible_for_class(
        "best_effort", sla_risk_score=0.30, has_deadline_risk=False,
    )
    assert eligible is False


def test_energy_scenario_no_longer_scales_batch_workloads():
    res = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=24, seed=42,
    )
    ca = res.report.aggregated["constraint_aware"]
    # Engine must have blocked at least some scale-up attempts on batch workloads.
    assert ca.blocked_scale_for_low_value_queue_relief >= 1
    # And applied zero scale-ups in the energy scenario.
    assert ca.scale_up_applied == 0


# ---------------------------------------------------------------------------
# B. Critical / standard-interactive workloads can still scale
# ---------------------------------------------------------------------------

def test_critical_interactive_allows_scale_when_sla_risk_is_real():
    for sla_risk in (0.2, 0.4, 0.7, 0.95):
        eligible, _ = _scale_eligible_for_class(
            "critical_interactive", sla_risk_score=sla_risk, has_deadline_risk=False,
        )
        assert eligible is True, f"critical must remain eligible at sla_risk={sla_risk}"


def test_standard_interactive_remains_eligible():
    eligible, _ = _scale_eligible_for_class(
        "standard_interactive", sla_risk_score=0.3, has_deadline_risk=False,
    )
    assert eligible is True


def test_batch_class_allows_scale_under_strong_sla_risk():
    # Above the strong-evidence threshold, scaling becomes class-eligible even
    # for batch — but the economic gate still has to clear (handled elsewhere).
    eligible, reason = _scale_eligible_for_class(
        "batch_inference",
        sla_risk_score=_STRONG_SLA_RISK_SCORE + 0.01,
        has_deadline_risk=False,
    )
    assert eligible is True
    assert "strong_sla_risk" in reason


# ---------------------------------------------------------------------------
# C. Deadline-driven scale for batch
# ---------------------------------------------------------------------------

def test_batch_class_allows_scale_under_deadline_risk():
    eligible, reason = _scale_eligible_for_class(
        "batch_inference", sla_risk_score=0.1, has_deadline_risk=True,
    )
    assert eligible is True
    assert "deadline_risk_allows_scale" == reason


def test_embedding_class_allows_scale_only_under_deadline_or_strong_sla():
    assert _scale_eligible_for_class(
        "embedding_offline", sla_risk_score=0.3, has_deadline_risk=False,
    )[0] is False
    assert _scale_eligible_for_class(
        "embedding_offline", sla_risk_score=0.3, has_deadline_risk=True,
    )[0] is True


# ---------------------------------------------------------------------------
# D. Energy scenario goodput/$ regression: improved vs prior constraint_aware
# ---------------------------------------------------------------------------

# Before this fix, constraint_aware achieved 196,792 goodput/$ on the canonical
# energy scenario (see PR #85 progress notes). This test pins the floor and
# guards against future regressions of the workload-class fix.
_PRIOR_ENERGY_GOODPUT_PER_DOLLAR = 196_792


def test_constraint_aware_energy_goodput_per_dollar_improved():
    res = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=24, seed=42,
    )
    agg = res.report.aggregated
    ca_yield = agg["constraint_aware"].sla_safe_goodput_per_infra_dollar
    assert ca_yield is not None
    assert ca_yield > _PRIOR_ENERGY_GOODPUT_PER_DOLLAR, (
        f"energy goodput/$ regressed: {ca_yield} ≤ prior {_PRIOR_ENERGY_GOODPUT_PER_DOLLAR}"
    )


def test_energy_scenario_constraint_aware_infra_cost_matches_fifo():
    # The fix makes constraint_aware stop adding billable GPU-hours that buy
    # nothing in the energy scenario. GPU infra cost should drop back to the
    # FIFO baseline (no harmful scaling). Tiny energy-cost differences from
    # in-region spread are within tolerance.
    res = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=24, seed=42,
    )
    agg = res.report.aggregated
    ca, f = agg["constraint_aware"], agg["fifo"]
    assert ca.gpu_infra_cost == f.gpu_infra_cost
    assert ca.total_infrastructure_cost < f.total_infrastructure_cost * 1.001


# ---------------------------------------------------------------------------
# E. No gaming
# ---------------------------------------------------------------------------

def test_action_that_raises_tokens_but_kills_goodput_per_dollar_is_rejected():
    # batch workload + marginal sla_risk: scaling raises raw tokens (more
    # replicas, more capacity) but burns GPU-hours that don't translate into
    # SLA-compliant goodput — so the action is rejected by class gating.
    eligible, reason = _scale_eligible_for_class(
        "batch_inference", sla_risk_score=0.35, has_deadline_risk=False,
    )
    assert eligible is False
    assert "blocked_scale_for_low_value_queue_relief" in reason


# ---------------------------------------------------------------------------
# G. Explanations include workload class + reason
# ---------------------------------------------------------------------------

def test_blocked_actions_carry_workload_class_in_rejected_log():
    res = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=24, seed=42,
    )
    saw_class = False
    saw_reason = False
    for er in res.policy_results["constraint_aware"].engine_results:
        if er is None:
            continue
        for rj in er.rejected:
            reason = rj.get("reject_reason", "")
            if (
                "blocked_scale_for_low_value_queue_relief" in reason
                or "blocked_uneconomic_scale" in reason
            ):
                saw_reason = True
                if rj.get("workload_class"):
                    saw_class = True
                if saw_class and saw_reason:
                    break
    assert saw_reason, "engine must produce an explicit workload-aware block reason"
    assert saw_class, "blocked actions must carry a `workload_class` tag"


# ---------------------------------------------------------------------------
# H. No regressions across canonical wins / SLA
# ---------------------------------------------------------------------------

def test_thermal_hotspot_constraint_aware_still_wins():
    res = ConstraintBenchmarkRunner().run_scenario(
        "thermal_hotspot_mixed_cluster", steps=24, seed=42,
    )
    agg = res.report.aggregated
    # Significantly above FIFO — the thermal-spreading win. The factor varies
    # by Python execution environment (pytest vs interpreter) due to the same
    # set/dict iteration sensitivity that #84 patched for energy; 1.15× is a
    # conservative floor that holds in both.
    assert (
        agg["constraint_aware"].sla_safe_goodput_per_infra_dollar
        > agg["fifo"].sla_safe_goodput_per_infra_dollar * 1.15
    )


def test_underutilization_constraint_aware_still_wins():
    res = ConstraintBenchmarkRunner().run_scenario(
        "underutilization_stranded_capacity", steps=24, seed=42,
    )
    agg = res.report.aggregated
    assert (
        agg["constraint_aware"].sla_safe_goodput_per_infra_dollar
        > agg["fifo"].sla_safe_goodput_per_infra_dollar * 1.30
    )


def test_no_sla_regression_vs_fifo_across_canonical_scenarios():
    runner = ConstraintBenchmarkRunner()
    for scn in (
        "thermal_hotspot_mixed_cluster",
        "queue_surge_latency_sensitive",
        "energy_price_arbitrage_multiregion",
        "underutilization_stranded_capacity",
        "rack_density_overload_air",
    ):
        agg = runner.run_scenario(scn, steps=24, seed=42).report.aggregated
        assert (
            agg["constraint_aware"].total_sla_violations
            <= agg["fifo"].total_sla_violations
        ), scn
