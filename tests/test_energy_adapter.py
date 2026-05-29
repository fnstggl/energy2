"""Tests for the energy-arbitrage adapter (constraints/energy_adapter.py).

The adapter consumes the EXISTING energy engine's recommendations and routes
them through the constraint-aware SLA / KPI / risk gates. These tests prove:
  * the adapter calls the energy engine's PUBLIC entry point,
  * a high-flexibility batch workload receives a candidate,
  * a candidate is accepted only when SLA-safe AND KPI-positive,
  * a candidate is rejected when the destination is hot / full / stale / bad-topology,
  * explanations carry gross savings, basis risk, KPI delta, and rejection reason,
  * the adapter never produces a fundamentally different energy recommendation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aurelius.constraints.energy_adapter import (
    DestinationContext,
    EnergyArbitrageAdapter,
    EnergyCandidateAction,
    ExistingEnergyCandidate,
    GateDecision,
)
from aurelius.models import Job, OptimizationConfig

W = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)


def _job(jid, wtype, runtime, slack, mc=0.1, regions=("us-east", "us-west"), gpu=2, power=100.0):
    es = W
    return Job(
        job_id=jid, submit_time=es, runtime_hours=runtime,
        deadline=es + timedelta(hours=runtime + slack), power_kw=power,
        earliest_start=es, region_options=list(regions), gpu_count=gpu,
        workload_type=wtype, migration_cost_hours=mc,
    )


def _da_rt():
    da = {"us-east": {W: 120.0}, "us-west": {W: 40.0}}
    rt = {"us-east": {W: 122.0}, "us-west": {W: 45.0}}
    return da, rt


class _SpyScheduler:
    """Records that the adapter called the energy engine's public API."""

    def __init__(self, inner):
        self.inner = inner
        self.solve_called = False
        self.baseline_called = False

    def solve(self, *a, **k):
        self.solve_called = True
        return self.inner.solve(*a, **k)

    def create_baseline_schedule(self, *a, **k):
        self.baseline_called = True
        return self.inner.create_baseline_schedule(*a, **k)


def test_adapter_calls_existing_engine_public_api():
    from aurelius.optimization.scheduler import JobScheduler
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    spy = _SpyScheduler(JobScheduler(cfg))
    adapter = EnergyArbitrageAdapter(scheduler=spy, config=cfg)
    da, rt = _da_rt()
    jobs = [_job("j-batch", "llm_batch_inference", 4, 12)]
    cands = adapter.recommend(jobs, da, rt_price_data=rt, method="greedy")
    assert spy.solve_called and spy.baseline_called
    assert len(cands) == 1


def test_high_flexibility_batch_receives_candidate():
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    adapter = EnergyArbitrageAdapter(config=cfg)
    da, rt = _da_rt()
    cands = adapter.recommend([_job("j-batch", "llm_batch_inference", 4, 12)],
                              da, rt_price_data=rt, method="greedy")
    c = cands[0]
    assert c.is_region_move
    assert c.recommended_region == "us-west"  # the cheap region
    assert c.action == EnergyCandidateAction.SHIFT_BATCH_TO_CHEAPER_REGION
    assert c.gross_savings_usd > 0


def test_candidate_accepted_when_sla_safe_and_kpi_positive():
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    adapter = EnergyArbitrageAdapter(config=cfg)
    da, rt = _da_rt()
    cands = adapter.recommend([_job("j-batch", "llm_batch_inference", 4, 12)],
                              da, rt_price_data=rt, method="greedy")
    ctx = {"us-west": DestinationContext(region="us-west", spare_capacity_pct=40.0)}
    [v] = adapter.evaluate_all(cands, ctx)
    assert v.decision == GateDecision.ACCEPT
    assert v.candidate_goodput_per_dollar > v.baseline_goodput_per_dollar
    assert v.kpi_delta > 0


def test_critical_interactive_rejected_ineligible():
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    adapter = EnergyArbitrageAdapter(config=cfg)
    da, rt = _da_rt()
    # realtime inference: latency-pinned. The energy engine still places it in
    # the cheap region (initial placement), but the adapter must reject the move.
    cands = adapter.recommend([_job("j-rt", "realtime_inference", 1, 0, mc=None)],
                              da, rt_price_data=rt, method="greedy")
    ctx = {"us-west": DestinationContext(region="us-west", spare_capacity_pct=40.0)}
    [v] = adapter.evaluate_all(cands, ctx)
    assert v.decision == GateDecision.REJECT
    assert "ineligible_critical_interactive_inference" in v.reasons


def _accepted_batch_candidate():
    return ExistingEnergyCandidate(
        job_id="j", workload_type="llm_batch_inference",
        action=EnergyCandidateAction.SHIFT_BATCH_TO_CHEAPER_REGION,
        current_region="us-east", recommended_region="us-west",
        gross_savings_usd=10.0, gross_savings_pct=60.0,
        da_price_current_mwh=120.0, da_price_target_mwh=40.0, rt_price_target_mwh=45.0,
        da_rt_basis_risk_usd=0.2, forecast_confidence=0.7,
        window_start=W, baseline_start=W, runtime_hours=4.0, slack_hours=12.0,
        deadline=W + timedelta(hours=16), migration_allowed=True,
        migration_cost_hours=0.1, latency_sensitive=False, gpu_count=2, power_kw=100.0,
    )


def test_rejected_when_destination_hot():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(),
                         DestinationContext("us-west", spare_capacity_pct=40.0, is_hot=True))
    assert v.decision == GateDecision.REJECT
    assert "destination_unsafe_thermal_hot" in v.reasons


def test_rejected_when_destination_full():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(),
                         DestinationContext("us-west", spare_capacity_pct=1.0))
    assert v.decision == GateDecision.REJECT
    assert "destination_unsafe_full" in v.reasons


def test_rejected_when_destination_stale():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(),
                         DestinationContext("us-west", spare_capacity_pct=40.0,
                                            telemetry_confidence="low", is_stale=True))
    assert v.decision == GateDecision.REJECT
    assert "destination_unsafe_stale_or_low_confidence_telemetry" in v.reasons


def test_rejected_when_destination_bad_topology():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(),
                         DestinationContext("us-west", spare_capacity_pct=40.0,
                                            topology_fit_ok=False))
    assert v.decision == GateDecision.REJECT
    assert "destination_unsafe_bad_topology_for_workload" in v.reasons


def test_rejected_when_destination_telemetry_missing():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(), destination_context=None)
    assert v.decision == GateDecision.REJECT
    assert "destination_unsafe_missing_telemetry" in v.reasons


def test_explanation_has_savings_basis_kpi_and_reason():
    adapter = EnergyArbitrageAdapter()
    v = adapter.evaluate(_accepted_batch_candidate(),
                         DestinationContext("us-west", spare_capacity_pct=40.0, is_hot=True))
    e = v.explanation()
    for key in ("gross_forecasted_energy_savings_usd", "da_rt_basis_risk_usd",
                "net_savings_usd", "kpi_delta", "decision", "reasons",
                "baseline_goodput_per_dollar", "candidate_goodput_per_dollar",
                "workload_type", "recommended_region"):
        assert key in e, f"explanation missing {key}"
    assert e["decision"] == "reject"
    assert e["reasons"] == ["destination_unsafe_thermal_hot"]


def test_cache_destroying_energy_move_blocked():
    """A cache-sensitive workload with a high hit rate is blocked from moving."""
    adapter = EnergyArbitrageAdapter()
    c = _accepted_batch_candidate()
    c = ExistingEnergyCandidate(**{**c.__dict__, "cache_sensitive": True,
                                   "cache_hit_rate": 0.9})
    v = adapter.evaluate(c, DestinationContext("us-west", spare_capacity_pct=40.0))
    assert v.decision == GateDecision.REJECT
    assert "ineligible_cache_sensitive_high_hit_rate" in v.reasons


def test_adapter_does_not_change_the_energy_target():
    """The adapter ACCEPTS/REJECTS the engine's target; it never substitutes a
    different region."""
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    adapter = EnergyArbitrageAdapter(config=cfg)
    da, rt = _da_rt()
    cands = adapter.recommend([_job("j-batch", "llm_batch_inference", 4, 12)],
                              da, rt_price_data=rt, method="greedy")
    ctx = {"us-west": DestinationContext(region="us-west", spare_capacity_pct=40.0)}
    [v] = adapter.evaluate_all(cands, ctx)
    # Accepted => the applied region is exactly the engine's recommended region.
    assert v.applied_region == v.candidate.recommended_region == "us-west"
    # Rejected => fall back to the safe current region (no substitute target).
    hot = {"us-west": DestinationContext(region="us-west", spare_capacity_pct=40.0, is_hot=True)}
    [vr] = adapter.evaluate_all(cands, hot)
    assert vr.applied_region == v.candidate.current_region == "us-east"
