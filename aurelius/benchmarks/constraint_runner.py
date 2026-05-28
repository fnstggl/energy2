"""Constraint-aware benchmark runner for Phase 11.

Runs each scenario under multiple optimizer policies in a closed-loop simulation:
  1. Reset simulator with the same seed
  2. Run ticks tick-by-tick
  3. For each policy, optionally apply recommendations between ticks
  4. Collect TickMetrics for each policy
  5. Return full KPI comparison with metadata

Policies:
  fifo             – no optimizer intervention (FIFO scheduling)
  current_price_only – migrate flexible workloads to cheapest region this tick only
  greedy_energy    – always migrate flexible workloads to cheapest available region
  sla_aware        – constraint engine, but only applies ENERGY actions
  constraint_aware – full ConstraintAwareEngine; applies safe recommendations

The simulator is the same object (same YAML scenario, same seed, reset between runs),
so physical models (thermal, queues, KV cache) are identical except where the policy
changes workload placement via ClusterSimulator.migrate_workload().
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

from ..constraints.engine import ConstraintAwareEngine, EngineResult
from ..simulation.cluster.engine import ClusterSimulator, TickMetrics
from ..simulation.cluster.scenarios import ScenarioConfig, list_scenarios, load_scenario
from ..state.models import ClusterState
from .economics import (
    InfrastructureCostConfig,
    compute_cost_per_sla_compliant_token,
    compute_gpu_infra_cost,
    compute_network_cost,
    compute_sla_safe_goodput_per_infra_dollar,
    compute_total_infrastructure_cost,
)
from .report import (
    AggregatedKPI,
    BenchmarkMetadata,
    BenchmarkReport,
    OptimizationScorecard,
    TickKPI,
    build_scorecard,
)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

POLICY_FIFO = "fifo"
POLICY_PRICE_ONLY = "current_price_only"
POLICY_GREEDY_ENERGY = "greedy_energy"
POLICY_SLA_AWARE = "sla_aware"
POLICY_CONSTRAINT_AWARE = "constraint_aware"

ALL_POLICIES = [
    POLICY_FIFO,
    POLICY_PRICE_ONLY,
    POLICY_GREEDY_ENERGY,
    POLICY_SLA_AWARE,
    POLICY_CONSTRAINT_AWARE,
]


# ---------------------------------------------------------------------------
# Per-policy result
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """Raw tick-level KPI data for one policy run."""
    policy_name: str
    tick_kpis: list[TickKPI]
    engine_results: list[Optional[EngineResult]]   # None for fifo/price_only
    migration_log: list[dict[str, str]]            # {tick, workload_id, from, to}
    error: Optional[str] = None
    final_state: Optional[ClusterState] = None     # last observed state (for packing analysis)


# ---------------------------------------------------------------------------
# Full benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Complete benchmark result: metadata + all policy runs + report."""
    metadata: BenchmarkMetadata
    policy_results: dict[str, PolicyResult]
    report: BenchmarkReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "report": self.report.to_dict(),
            "policy_migration_counts": {
                p: len(r.migration_log)
                for p, r in self.policy_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Policy application helpers
# ---------------------------------------------------------------------------

def _apply_fifo(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """No-op: FIFO makes no optimizer decisions."""
    return None


def _apply_price_only(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """Migrate flexible workloads to cheapest region this tick only."""
    regions = state.regions
    if not regions:
        return None

    # Plan on the DAY-AHEAD price (the signal available at scheduling time).
    # Realized consumption settles at the real-time price, so a DA planner can be
    # wrong under an RT basis blowout — that is the realism this exposes.
    def price(r):
        if r.energy is not None:
            p = (r.energy.day_ahead_price_per_mwh
                 if r.energy.day_ahead_price_per_mwh is not None
                 else r.energy.real_time_price_per_mwh)
            if p is not None:
                return p
        # Fall back to sim cluster price for this region
        sim_r = sim._cluster.regions.get(r.region)
        if sim_r is not None:
            return sim_r.day_ahead_price
        return float("inf")

    cheapest = min(regions.values(), key=price)
    cheapest_price = price(cheapest)

    # Only migrate flexible workloads where current region is strictly more expensive
    for svc in state.all_services.values():
        current_region_id = getattr(svc, "region_id", None) or svc.region
        current_region = regions.get(current_region_id or "")
        if current_region is None:
            continue
        current_price = price(current_region)
        if current_price <= cheapest_price * 1.01:  # no meaningful saving
            continue
        if current_region_id == cheapest.region:
            continue
        # Find a workload in the sim matching this service; skip latency-sensitive
        for wl in sim._cluster.workloads.values():
            if (wl.service_id == svc.service_id
                    and wl.migration_allowed and not wl.latency_sensitive):
                src = wl.region_id
                migrated = sim.migrate_workload(wl.workload_id, cheapest.region)
                if migrated:
                    migration_log.append({
                        "tick": str(sim._cluster.tick),
                        "workload_id": wl.workload_id,
                        "from": src,
                        "to": cheapest.region,
                    })
                break
    return None


def _apply_greedy_energy(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """Always migrate any flexible workload to cheapest region (aggressive)."""
    regions = state.regions
    if not regions:
        return None

    def price(r):
        if r.energy is not None:
            p = (r.energy.day_ahead_price_per_mwh
                 if r.energy.day_ahead_price_per_mwh is not None
                 else r.energy.real_time_price_per_mwh)
            if p is not None:
                return p
        sim_r = sim._cluster.regions.get(r.region)
        if sim_r is not None:
            return sim_r.day_ahead_price
        return float("inf")

    cheapest = min(regions.values(), key=price)

    cheapest_price_val = price(cheapest)
    for wl in list(sim._cluster.workloads.values()):
        if not wl.migration_allowed:
            continue
        if wl.region_id == cheapest.region:
            continue
        if wl.latency_sensitive:
            continue
        # Get sim-level price for current region
        sim_region = sim._cluster.regions.get(wl.region_id)
        if sim_region is None:
            continue
        # Plan on the day-ahead price (settles at real-time).
        sim_price = sim_region.day_ahead_price
        if sim_price <= cheapest_price_val * 1.005:
            continue
        src = wl.region_id
        migrated = sim.migrate_workload(wl.workload_id, cheapest.region)
        if migrated:
            migration_log.append({
                "tick": str(sim._cluster.tick),
                "workload_id": wl.workload_id,
                "from": src,
                "to": cheapest.region,
            })
    return None


def _apply_sla_aware(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """Run constraint engine but only apply CHOOSE_CHEAPER_REGION (energy) actions.

    This isolates the SLA-aware optimization benefit without full constraint routing.
    rec.workload_id is the service_id; rec.target_region is the destination region.
    """
    er = engine.run(state)

    for rec in er.recommendations:
        if rec.is_noop:
            continue
        if rec.action_type != "choose_cheaper_region":  # ActionType.value (lowercase)
            continue
        target_region = rec.target_region
        if target_region is None:
            continue
        # Map service_id → workload in simulator
        svc_id = rec.workload_id
        for wl in sim._cluster.workloads.values():
            if wl.service_id == svc_id and wl.migration_allowed:
                src = wl.region_id
                migrated = sim.migrate_workload(wl.workload_id, target_region)
                if migrated:
                    migration_log.append({
                        "tick": str(sim._cluster.tick),
                        "workload_id": wl.workload_id,
                        "from": src,
                        "to": target_region,
                    })
                break
    return er


def _service_region(sim: ClusterSimulator, service_id: str) -> Optional[str]:
    for wl in sim._cluster.workloads.values():
        if wl.service_id == service_id or wl.workload_id == service_id:
            return wl.region_id
    return None


def _apply_constraint_aware(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """Run the full ConstraintAwareEngine and apply EVERY safe recommendation type.

    rec.workload_id is the service_id. Cross-region migrations use migrate_workload;
    operational actions (SCALE/SPREAD/DEFER/CONSOLIDATE/REROUTE) use the simulator's
    Mission-3 action methods so the constraint-aware policy is measured on
    thermal/queue/utilization/latency scenarios, not only energy migrations.

    NOTE: action_type values are ActionType.value (lowercase). The prior
    implementation compared against UPPERCASE names that never matched — so it
    silently applied nothing (constraint_aware was byte-identical to FIFO).
    """
    er = engine.run(state)

    migration_acts = {"choose_cheaper_region", "migrate_workload", "choose_lower_carbon_region"}

    for rec in er.recommendations:
        if rec.is_noop:
            continue
        at = rec.action_type
        svc_id = rec.workload_id
        applied = False

        if at in migration_acts and rec.target_region:
            for wl in sim._cluster.workloads.values():
                if wl.service_id == svc_id and wl.migration_allowed:
                    src = wl.region_id
                    if sim.migrate_workload(wl.workload_id, rec.target_region):
                        migration_log.append({
                            "tick": str(sim._cluster.tick), "workload_id": wl.workload_id,
                            "from": src, "to": rec.target_region, "action": at,
                        })
                        applied = True
                    break
        elif at == "change_placement" and rec.target_region \
                and rec.target_region != _service_region(sim, svc_id):
            for wl in sim._cluster.workloads.values():
                if wl.service_id == svc_id and wl.migration_allowed:
                    sim.migrate_workload(wl.workload_id, rec.target_region)
                    applied = True
                    break
        elif at == "scale_replicas":
            applied = sim.add_replica(svc_id)
        elif at == "spread_workloads":
            applied = sim.spread_workload(svc_id)
        elif at == "reroute_workload":
            if rec.target_region and rec.target_region != _service_region(sim, svc_id):
                for wl in sim._cluster.workloads.values():
                    if wl.service_id == svc_id and wl.migration_allowed:
                        sim.migrate_workload(wl.workload_id, rec.target_region)
                        applied = True
                        break
            else:
                applied = sim.spread_workload(svc_id)
        elif at == "defer_workload":
            applied = sim.defer_flexible_workload(svc_id)
        elif at == "consolidate_workloads":
            region = _service_region(sim, svc_id)
            if region:
                applied = sim.consolidate_low_priority(region, svc_id)

        if applied and at not in migration_acts:
            migration_log.append({
                "tick": str(sim._cluster.tick), "workload_id": svc_id, "action": at,
            })

    return er


_POLICY_APPLY_FNS = {
    POLICY_FIFO: _apply_fifo,
    POLICY_PRICE_ONLY: _apply_price_only,
    POLICY_GREEDY_ENERGY: _apply_greedy_energy,
    POLICY_SLA_AWARE: _apply_sla_aware,
    POLICY_CONSTRAINT_AWARE: _apply_constraint_aware,
}


# ---------------------------------------------------------------------------
# KPI aggregation
# ---------------------------------------------------------------------------

def _aggregate_kpis(
    policy_name: str,
    tick_kpis: list[TickKPI],
    cost_config: Optional[InfrastructureCostConfig] = None,
    engine_results: Optional[list] = None,
) -> AggregatedKPI:
    if cost_config is None:
        cost_config = InfrastructureCostConfig()
    if not tick_kpis:
        return AggregatedKPI(
            policy_name=policy_name,
            total_energy_cost=0.0,
            total_tokens=0,
            total_energy_kwh=0.0,
            mean_cost_per_token=None,
            mean_tokens_per_joule=None,
            mean_gpu_util_pct=0.0,
            p99_latency_ms=None,
            p95_latency_ms=None,
            p95_queue_wait_ms=None,
            total_sla_violations=0,
            total_thermal_throttle_ticks=0,
            total_migrations=0,
            mean_topology_score=1.0,
        )

    total_cost = sum(k.total_energy_cost for k in tick_kpis)
    total_tokens = sum(k.total_tokens for k in tick_kpis)
    total_kwh = sum(k.total_energy_kwh for k in tick_kpis)

    cpt_vals = [k.cost_per_token for k in tick_kpis if k.cost_per_token is not None]
    tpj_vals = [k.tokens_per_joule for k in tick_kpis if k.tokens_per_joule is not None]
    util_vals = [k.mean_gpu_util_pct for k in tick_kpis]
    p99_vals = [k.p99_latency_ms for k in tick_kpis if k.p99_latency_ms is not None]
    p95_vals = [k.p95_latency_ms for k in tick_kpis if k.p95_latency_ms is not None]
    qwait_vals = [k.queue_wait_p95_ms for k in tick_kpis if k.queue_wait_p95_ms is not None]
    topo_vals = [k.mean_topology_score for k in tick_kpis]

    kvp_vals = [k.kv_pressure_max for k in tick_kpis if k.kv_pressure_max is not None]
    hit_vals = [k.prefix_hit_rate_mean for k in tick_kpis if k.prefix_hit_rate_mean is not None]
    loc_vals = [
        k.locality_confidence_mean for k in tick_kpis if k.locality_confidence_mean is not None
    ]
    frag_vals = [
        k.cache_fragmentation_frac_mean for k in tick_kpis
        if k.cache_fragmentation_frac_mean is not None
    ]
    ttft99_vals = [k.ttft_p99_ms for k in tick_kpis if k.ttft_p99_ms is not None]

    beff_vals = [
        k.batch_efficiency_mean for k in tick_kpis if k.batch_efficiency_mean is not None
    ]
    psat_vals = [
        k.proxy_saturation_max for k in tick_kpis if k.proxy_saturation_max is not None
    ]
    startup_vals = [
        k.startup_latency_s_max for k in tick_kpis if k.startup_latency_s_max is not None
    ]
    gtemp_vals = [k.max_gpu_temp_c for k in tick_kpis if k.max_gpu_temp_c is not None]
    sth_vals = [
        k.thermal_slowdown_pct_mean for k in tick_kpis
        if k.thermal_slowdown_pct_mean is not None
    ]
    hot_vals2 = [k.hotspot_severity_max for k in tick_kpis if k.hotspot_severity_max is not None]
    rkw_vals = [k.rack_density_kw_max for k in tick_kpis if k.rack_density_kw_max is not None]
    tq_vals = [
        k.mean_topology_quality for k in tick_kpis if k.mean_topology_quality is not None
    ]
    fc_vals = [k.fabric_congestion_max for k in tick_kpis if k.fabric_congestion_max is not None]
    ca_vals = [
        k.collective_amplification_max for k in tick_kpis
        if k.collective_amplification_max is not None
    ]
    cp_vals = [
        k.comm_throughput_penalty_pct_mean for k in tick_kpis
        if k.comm_throughput_penalty_pct_mean is not None
    ]
    ss_vals = [
        k.sync_slowdown_pct_mean for k in tick_kpis if k.sync_slowdown_pct_mean is not None
    ]
    ns_vals = [k.nic_saturation_max for k in tick_kpis if k.nic_saturation_max is not None]
    tr_vals = [k.topology_risk_max for k in tick_kpis if k.topology_risk_max is not None]
    cl_vals = [
        k.comm_latency_p99_ms_max for k in tick_kpis if k.comm_latency_p99_ms_max is not None
    ]
    eu_vals = [k.mean_effective_util for k in tick_kpis if k.mean_effective_util is not None]
    da_vals = [k.dram_active_max for k in tick_kpis if k.dram_active_max is not None]
    fr_vals = [
        k.fragmentation_score_max for k in tick_kpis if k.fragmentation_score_max is not None
    ]
    pd_vals = [k.packing_density_max for k in tick_kpis if k.packing_density_max is not None]
    cr_vals = [
        k.consolidation_risk_max for k in tick_kpis if k.consolidation_risk_max is not None
    ]
    qa_vals = [
        k.queue_amplification_max for k in tick_kpis if k.queue_amplification_max is not None
    ]
    up_vals = [
        k.util_throughput_penalty_pct_mean for k in tick_kpis
        if k.util_throughput_penalty_pct_mean is not None
    ]
    bp_vals = [k.bin_packing_risk_max for k in tick_kpis if k.bin_packing_risk_max is not None]
    da_vals2 = [k.day_ahead_price_mean for k in tick_kpis if k.day_ahead_price_mean is not None]
    rt_vals2 = [k.real_time_price_mean for k in tick_kpis if k.real_time_price_mean is not None]
    basis_vals = [k.da_rt_basis_max for k in tick_kpis if k.da_rt_basis_max is not None]
    lc_vals = [k.lmp_congestion_max for k in tick_kpis if k.lmp_congestion_max is not None]
    ci_vals = [k.carbon_intensity_mean for k in tick_kpis if k.carbon_intensity_mean is not None]
    ns_vals2 = [k.net_savings_sum for k in tick_kpis if k.net_savings_sum is not None]
    gs_vals = [k.gross_savings_sum for k in tick_kpis if k.gross_savings_sum is not None]
    cpen_vals = [k.churn_penalty_max for k in tick_kpis if k.churn_penalty_max is not None]

    # --- Canonical primary KPI: SLA-safe goodput per infrastructure dollar ---
    # Computed from simulator-tracked sla_compliant_tokens + active_gpu_hours
    # and the existing energy_cost stream. No business-value weights, no
    # synthetic SLA penalty dollars folded in.
    cost_cfg = cost_config
    aggregated_by_type: dict[str, float] = {}
    for tk in tick_kpis:
        for gtype, hrs in (tk.active_gpu_hours_by_type or {}).items():
            aggregated_by_type[gtype] = aggregated_by_type.get(gtype, 0.0) + hrs
    active_gpu_hours = sum(aggregated_by_type.values())
    gpu_cost = compute_gpu_infra_cost(aggregated_by_type, cost_cfg)
    sla_goodput = sum(k.sla_compliant_tokens for k in tick_kpis)
    # network_cost is keyed on migrations only when the operator has configured
    # a per-migration cost; default is 0 per the spec.
    net_cost = compute_network_cost(
        migration_count=tick_kpis[-1].migration_count if tick_kpis else 0,
        config=cost_cfg,
    )
    total_infra_cost = compute_total_infrastructure_cost(gpu_cost, total_cost, net_cost)
    primary_kpi = compute_sla_safe_goodput_per_infra_dollar(sla_goodput, total_infra_cost)
    cpsct = compute_cost_per_sla_compliant_token(total_infra_cost, sla_goodput)
    goodput_per_gpu_hr = (
        sla_goodput / active_gpu_hours if active_gpu_hours > 0 else None
    )

    # Workload-aware action accounting (this PR): count SCALE_REPLICAS
    # recommendations and the rejections of each kind, by scanning engine
    # outputs. Pure parsing of existing `rejected.reject_reason` strings.
    scale_up_recommended = 0
    blk_low_value = 0
    blk_uneconomic = 0
    blk_dominated = 0
    if engine_results:
        for er in engine_results:
            if er is None:
                continue
            for rec in er.recommendations:
                if (not rec.is_noop) and rec.action_type == "scale_replicas":
                    scale_up_recommended += 1
            for rj in er.rejected:
                reason = rj.get("reject_reason", "")
                if reason.startswith("blocked_scale_for_low_value_queue_relief"):
                    blk_low_value += 1
                elif reason.startswith("blocked_uneconomic_scale"):
                    blk_uneconomic += 1
                elif reason.startswith("dominated"):
                    blk_dominated += 1

    return AggregatedKPI(
        policy_name=policy_name,
        total_energy_cost=total_cost,
        total_tokens=total_tokens,
        total_energy_kwh=total_kwh,
        mean_cost_per_token=sum(cpt_vals) / len(cpt_vals) if cpt_vals else None,
        mean_tokens_per_joule=sum(tpj_vals) / len(tpj_vals) if tpj_vals else None,
        mean_gpu_util_pct=sum(util_vals) / len(util_vals) if util_vals else 0.0,
        p99_latency_ms=max(p99_vals) if p99_vals else None,
        p95_latency_ms=max(p95_vals) if p95_vals else None,
        p95_queue_wait_ms=max(qwait_vals) if qwait_vals else None,
        total_sla_violations=sum(k.sla_violations for k in tick_kpis),
        sla_compliant_goodput=sla_goodput,
        gpu_infra_cost=gpu_cost,
        energy_cost=total_cost,
        network_cost=net_cost,
        total_infrastructure_cost=total_infra_cost,
        sla_safe_goodput_per_infra_dollar=primary_kpi,
        cost_per_sla_compliant_token=cpsct,
        active_gpu_hours=active_gpu_hours,
        active_gpu_hours_by_type=aggregated_by_type,
        goodput_per_gpu_hour=goodput_per_gpu_hr,
        scale_up_recommended=scale_up_recommended,
        scale_up_applied=0,  # filled in by the runner below from migration_log
        blocked_scale_for_low_value_queue_relief=blk_low_value,
        blocked_uneconomic_scale=blk_uneconomic,
        blocked_dominated=blk_dominated,
        total_thermal_throttle_ticks=sum(k.thermal_throttle_gpu_count for k in tick_kpis),
        total_migrations=tick_kpis[-1].migration_count if tick_kpis else 0,
        mean_topology_score=sum(topo_vals) / len(topo_vals) if topo_vals else 1.0,
        kv_pressure_max=max(kvp_vals) if kvp_vals else None,
        prefix_hit_rate_mean=sum(hit_vals) / len(hit_vals) if hit_vals else None,
        total_preemptions=sum(k.preemption_count for k in tick_kpis),
        total_cold_reroutes=tick_kpis[-1].cold_reroute_count if tick_kpis else 0,
        total_cache_evictions=sum(k.cache_eviction_count for k in tick_kpis),
        locality_confidence_mean=sum(loc_vals) / len(loc_vals) if loc_vals else None,
        cache_fragmentation_frac_mean=sum(frag_vals) / len(frag_vals) if frag_vals else None,
        ttft_p99_ms=max(ttft99_vals) if ttft99_vals else None,
        total_reroutes=tick_kpis[-1].reroute_count if tick_kpis else 0,
        total_migration_vetoes=tick_kpis[-1].migration_veto_count if tick_kpis else 0,
        batch_efficiency_mean=sum(beff_vals) / len(beff_vals) if beff_vals else None,
        proxy_saturation_max=max(psat_vals) if psat_vals else None,
        total_cold_starts=tick_kpis[-1].cold_start_count if tick_kpis else 0,
        total_rollbacks=tick_kpis[-1].rollback_count if tick_kpis else 0,
        total_overload_events=tick_kpis[-1].overload_events if tick_kpis else 0,
        startup_latency_s_max=max(startup_vals) if startup_vals else None,
        max_gpu_temp_c=max(gtemp_vals) if gtemp_vals else None,
        thermal_slowdown_pct_mean=sum(sth_vals) / len(sth_vals) if sth_vals else None,
        total_thermal_throttle_events=sum(k.thermal_throttle_events for k in tick_kpis),
        hotspot_severity_max=max(hot_vals2) if hot_vals2 else None,
        rack_density_kw_max=max(rkw_vals) if rkw_vals else None,
        total_thermal_excursions=tick_kpis[-1].thermal_excursions if tick_kpis else 0,
        total_thermal_migration_vetoes=sum(k.thermal_migration_vetoes for k in tick_kpis),
        mean_topology_quality=sum(tq_vals) / len(tq_vals) if tq_vals else None,
        min_topology_quality=min(tq_vals) if tq_vals else None,
        fabric_congestion_max=max(fc_vals) if fc_vals else None,
        collective_amplification_max=max(ca_vals) if ca_vals else None,
        comm_throughput_penalty_pct_mean=sum(cp_vals) / len(cp_vals) if cp_vals else None,
        sync_slowdown_pct_mean=sum(ss_vals) / len(ss_vals) if ss_vals else None,
        nic_saturation_max=max(ns_vals) if ns_vals else None,
        topology_risk_max=max(tr_vals) if tr_vals else None,
        total_collective_instability=sum(k.collective_instability_count for k in tick_kpis),
        total_topology_migration_vetoes=sum(k.topology_migration_vetoes for k in tick_kpis),
        comm_latency_p99_ms_max=max(cl_vals) if cl_vals else None,
        mean_effective_util=sum(eu_vals) / len(eu_vals) if eu_vals else None,
        dram_active_max=max(da_vals) if da_vals else None,
        fragmentation_score_max=max(fr_vals) if fr_vals else None,
        stranded_gpu_count_max=max((k.stranded_gpu_count for k in tick_kpis), default=0),
        packing_density_max=max(pd_vals) if pd_vals else None,
        consolidation_risk_max=max(cr_vals) if cr_vals else None,
        total_unsafe_consolidation=max(
            (k.unsafe_consolidation_count for k in tick_kpis), default=0
        ),
        queue_amplification_max=max(qa_vals) if qa_vals else None,
        util_throughput_penalty_pct_mean=sum(up_vals) / len(up_vals) if up_vals else None,
        total_utilization_paradox=max(
            (k.utilization_paradox_count for k in tick_kpis), default=0
        ),
        bin_packing_risk_max=max(bp_vals) if bp_vals else None,
        total_packing_migration_vetoes=sum(k.packing_migration_vetoes for k in tick_kpis),
        day_ahead_price_mean=sum(da_vals2) / len(da_vals2) if da_vals2 else None,
        real_time_price_mean=sum(rt_vals2) / len(rt_vals2) if rt_vals2 else None,
        da_rt_basis_max=max(basis_vals) if basis_vals else None,
        lmp_congestion_max=max(lc_vals) if lc_vals else None,
        carbon_intensity_mean=sum(ci_vals) / len(ci_vals) if ci_vals else None,
        total_net_savings=sum(ns_vals2) if ns_vals2 else None,
        total_gross_savings=sum(gs_vals) if gs_vals else None,
        total_energy_migration_vetoes=sum(k.energy_migration_vetoes for k in tick_kpis),
        total_energy_actions_rejected=sum(k.energy_actions_rejected for k in tick_kpis),
        churn_penalty_max=max(cpen_vals) if cpen_vals else None,
    )


def _tick_metrics_to_kpi(tm: TickMetrics) -> TickKPI:
    return TickKPI(
        tick=tm.tick,
        total_energy_cost=tm.total_energy_cost,
        total_tokens=tm.total_tokens,
        total_energy_kwh=tm.total_energy_kwh,
        cost_per_token=tm.cost_per_token,
        tokens_per_joule=tm.tokens_per_joule,
        mean_gpu_util_pct=tm.mean_gpu_util_pct,
        p95_latency_ms=tm.p95_latency_ms,
        p99_latency_ms=tm.p99_latency_ms,
        queue_wait_p95_ms=tm.queue_wait_p95_ms,
        sla_violations=tm.sla_violations,
        thermal_throttle_gpu_count=tm.thermal_throttle_gpu_count,
        migration_count=tm.migration_count,
        mean_topology_score=tm.mean_topology_score,
        sla_compliant_tokens=tm.sla_compliant_tokens,
        active_gpu_count=tm.active_gpu_count,
        active_gpu_hours_by_type=dict(tm.active_gpu_hours_by_type or {}),
        kv_pressure_max=tm.kv_pressure_max,
        prefix_hit_rate_mean=tm.prefix_hit_rate_mean,
        preemption_count=tm.preemption_count,
        cold_reroute_count=tm.cold_reroute_count,
        cache_eviction_count=tm.cache_eviction_count,
        locality_confidence_mean=tm.locality_confidence_mean,
        cache_fragmentation_frac_mean=tm.cache_fragmentation_frac_mean,
        ttft_p99_ms=tm.ttft_p99_ms,
        reroute_count=tm.reroute_count,
        migration_veto_count=tm.migration_veto_count,
        batch_efficiency_mean=tm.batch_efficiency_mean,
        proxy_saturation_max=tm.proxy_saturation_max,
        cold_start_count=tm.cold_start_count,
        rollback_count=tm.rollback_count,
        overload_events=tm.overload_events,
        startup_latency_s_max=tm.startup_latency_s_max,
        max_gpu_temp_c=tm.max_gpu_temp_c,
        thermal_slowdown_pct_mean=tm.thermal_slowdown_pct_mean,
        thermal_throttle_events=tm.thermal_throttle_events,
        hotspot_severity_max=tm.hotspot_severity_max,
        rack_density_kw_max=tm.rack_density_kw_max,
        thermal_excursions=tm.thermal_excursions,
        thermal_migration_vetoes=tm.thermal_migration_vetoes,
        mean_topology_quality=tm.mean_topology_quality,
        fabric_congestion_max=tm.fabric_congestion_max,
        collective_amplification_max=tm.collective_amplification_max,
        comm_throughput_penalty_pct_mean=tm.comm_throughput_penalty_pct_mean,
        sync_slowdown_pct_mean=tm.sync_slowdown_pct_mean,
        nic_saturation_max=tm.nic_saturation_max,
        topology_risk_max=tm.topology_risk_max,
        collective_instability_count=tm.collective_instability_count,
        topology_migration_vetoes=tm.topology_migration_vetoes,
        comm_latency_p99_ms_max=tm.comm_latency_p99_ms_max,
        mean_effective_util=tm.mean_effective_util,
        dram_active_max=tm.dram_active_max,
        fragmentation_score_max=tm.fragmentation_score_max,
        stranded_gpu_count=tm.stranded_gpu_count,
        packing_density_max=tm.packing_density_max,
        consolidation_risk_max=tm.consolidation_risk_max,
        unsafe_consolidation_count=tm.unsafe_consolidation_count,
        queue_amplification_max=tm.queue_amplification_max,
        util_throughput_penalty_pct_mean=tm.util_throughput_penalty_pct_mean,
        utilization_paradox_count=tm.utilization_paradox_count,
        bin_packing_risk_max=tm.bin_packing_risk_max,
        packing_migration_vetoes=tm.packing_migration_vetoes,
        day_ahead_price_mean=tm.day_ahead_price_mean,
        real_time_price_mean=tm.real_time_price_mean,
        da_rt_basis_max=tm.da_rt_basis_max,
        lmp_congestion_max=tm.lmp_congestion_max,
        carbon_intensity_mean=tm.carbon_intensity_mean,
        net_savings_sum=tm.net_savings_sum,
        gross_savings_sum=tm.gross_savings_sum,
        energy_migration_vetoes=tm.energy_migration_vetoes,
        energy_actions_rejected=tm.energy_actions_rejected,
        churn_penalty_max=tm.churn_penalty_max,
        low_energy_telemetry_count=tm.low_energy_telemetry_count,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class ConstraintBenchmarkRunner:
    """Multi-policy constraint-aware benchmark runner.

    Usage:
        runner = ConstraintBenchmarkRunner()
        result = runner.run_scenario("energy_price_arbitrage_multiregion", steps=24)
        print(result.report.to_text())
    """

    def __init__(
        self,
        policies: Optional[list[str]] = None,
        confidence_floor: float = 0.3,
        cost_config: Optional[InfrastructureCostConfig] = None,
    ):
        self._policies = policies or ALL_POLICIES
        from ..constraints.classifier import ConstraintConfig
        self._engine = ConstraintAwareEngine(
            classifier_config=ConstraintConfig(confidence_floor=confidence_floor)
        )
        # Infrastructure cost basis for the canonical KPI (operator-overridable).
        # Defaults are documented public-list cloud GPU prices; not production.
        self._cost_config = cost_config or InfrastructureCostConfig()

    def run_scenario(
        self,
        scenario_name: str,
        seed: Optional[int] = None,
        steps: int = 24,
        version: str = "v1",
    ) -> BenchmarkResult:
        """Run a scenario under all configured policies and return a BenchmarkResult."""
        scenario = load_scenario(scenario_name, version=version, seed_override=seed)
        effective_seed = scenario.config.seed

        metadata = BenchmarkMetadata.build(
            scenario_name=scenario_name,
            scenario_version=scenario.version,
            scenario_hash=scenario.scenario_hash,
            seed=effective_seed,
            simulator_version=scenario.config.simulator_version,
            steps=steps,
            config_dict={
                "scenario_name": scenario_name,
                "scenario_version": scenario.version,
                "seed": effective_seed,
                "tick_duration_hours": scenario.config.tick_duration_hours,
                "steps": steps,
            },
        )

        policy_results: dict[str, PolicyResult] = {}

        for policy_name in self._policies:
            pr = self._run_policy(scenario, policy_name, steps, effective_seed)
            policy_results[policy_name] = pr

        report = self._build_report(
            metadata=metadata,
            policy_results=policy_results,
            scenario=scenario,
            steps=steps,
        )

        return BenchmarkResult(
            metadata=metadata,
            policy_results=policy_results,
            report=report,
        )

    def run_all_scenarios(
        self,
        seed: Optional[int] = None,
        steps: int = 24,
        version: str = "v1",
    ) -> dict[str, BenchmarkResult]:
        """Run all available scenarios and return per-scenario results."""
        results: dict[str, BenchmarkResult] = {}
        for name in list_scenarios(version=version):
            try:
                results[name] = self.run_scenario(name, seed=seed, steps=steps, version=version)
            except Exception as exc:
                # Don't fail the whole suite on one bad scenario
                results[name] = _error_result(name, str(exc))
        return results

    def _run_policy(
        self,
        scenario: ScenarioConfig,
        policy_name: str,
        steps: int,
        seed: int,
    ) -> PolicyResult:
        """Run a single policy on the scenario and collect tick KPIs."""
        sim = ClusterSimulator(scenario.config, seed=seed)
        apply_fn = _POLICY_APPLY_FNS[policy_name]
        migration_log: list[dict[str, str]] = []
        engine_results: list[Optional[EngineResult]] = []
        tick_kpis: list[TickKPI] = []
        last_state: Optional[ClusterState] = None

        for _ in range(steps):
            tick_result = sim.tick()
            # Apply optimizer AFTER observing tick state (before next tick)
            state = tick_result.cluster_state
            last_state = state
            er = apply_fn(sim, state, self._engine, migration_log)
            engine_results.append(er)
            tick_kpis.append(_tick_metrics_to_kpi(tick_result.metrics))

        return PolicyResult(
            policy_name=policy_name,
            tick_kpis=tick_kpis,
            engine_results=engine_results,
            migration_log=migration_log,
            final_state=last_state,
        )

    def _build_report(
        self,
        metadata: BenchmarkMetadata,
        policy_results: dict[str, PolicyResult],
        scenario: ScenarioConfig,
        steps: int,
    ) -> BenchmarkReport:
        aggregated: dict[str, AggregatedKPI] = {}
        for policy_name, pr in policy_results.items():
            aggregated[policy_name] = _aggregate_kpis(
                policy_name, pr.tick_kpis, cost_config=self._cost_config,
                engine_results=pr.engine_results,
            )
            # Count applied scale-ups from the migration log so we can compare
            # applied vs recommended (an applied scale needs an idle GPU).
            applied_scale = sum(
                1 for m in pr.migration_log if m.get("action") == "scale_replicas"
            )
            aggregated[policy_name] = replace(
                aggregated[policy_name], scale_up_applied=applied_scale,
            )

        fifo_kpi = aggregated.get(POLICY_FIFO)
        ca_kpi = aggregated.get(POLICY_CONSTRAINT_AWARE)
        if fifo_kpi and ca_kpi:
            scorecard = build_scorecard(ca_kpi, fifo_kpi, steps)
        else:
            # Fallback scorecard when fifo or constraint_aware not in policies
            scorecard = OptimizationScorecard(
                net_cost_improvement=0.5,
                sla_preservation=1.0,
                utilization_improvement=0.5,
                latency_improvement=0.5,
                thermal_improvement=1.0,
                migration_stability=1.0,
                topology_quality=1.0,
                weighted_score=0.7,
                flags=["Scorecard requires fifo and constraint_aware policies"],
            )

        # Determine dominant observed constraint
        observed_dominant: Optional[str] = None
        ca_results = (policy_results.get(POLICY_CONSTRAINT_AWARE) or
                      policy_results.get(POLICY_SLA_AWARE))
        if ca_results and ca_results.engine_results:
            constraint_counts: dict[str, int] = {}
            for er in ca_results.engine_results:
                if er is None:
                    continue
                bc = er.assessment.binding_constraint
                if bc:
                    key = bc.value
                    constraint_counts[key] = constraint_counts.get(key, 0) + 1
            if constraint_counts:
                observed_dominant = max(constraint_counts, key=constraint_counts.__getitem__)

        expected = scenario.expected_primary_constraint
        # Normalize expected: YAML uses "energy_bound"/"memory_bound_indirect", enum uses "energy"/"memory"
        if expected == "memory_bound_indirect":
            expected_normalized: Optional[str] = "memory"
        else:
            expected_normalized = expected.removesuffix("_bound") if expected else None
        constraint_match = (
            observed_dominant == expected_normalized
            if expected_normalized and observed_dominant
            else True
        )

        regression_flags: list[str] = list(scorecard.flags)

        # Packing baseline frontier (analysis-only) for utilization/fragmentation scenarios.
        packing_frontier = self._packing_frontier(metadata.scenario_name, policy_results)

        # Per-workload baseline reporting (PR #87): pick the workload-relevant
        # strong baseline (NOT FIFO) and classify CA's outcome against it.
        # FIFO is now a sanity-only baseline.
        scenario_metadata = getattr(scenario, "metadata", None)
        headline_name: Optional[str] = None
        headline_rationale: Optional[str] = None
        outcome = None
        try:
            from .per_workload import (
                analyze_outcome,
                classify_scenario,
                select_headline_baseline,
            )
            if scenario_metadata is None:
                scenario_metadata = classify_scenario(
                    metadata.scenario_name,
                    scenario.expected_primary_constraint,
                    {},
                )
            headline_name, headline_rationale = select_headline_baseline(
                scenario_metadata, aggregated,
            )
            ca_for_outcome = aggregated.get(POLICY_CONSTRAINT_AWARE)
            headline_kpi = aggregated.get(headline_name) or aggregated.get(
                POLICY_FIFO
            )
            if ca_for_outcome is not None and headline_kpi is not None:
                outcome = analyze_outcome(
                    scenario_metadata, ca_for_outcome, headline_kpi, aggregated,
                    scorecard_flags=tuple(scorecard.flags),
                    headline_name=headline_name,
                )
        except Exception:
            # Never fail the benchmark on a reporting-layer issue.
            scenario_metadata = scenario_metadata
            headline_name = None
            headline_rationale = None
            outcome = None

        return BenchmarkReport(
            metadata=metadata,
            aggregated=aggregated,
            scorecard=scorecard,
            expected_primary_constraint=expected_normalized,
            observed_dominant_constraint=observed_dominant,
            constraint_match=constraint_match,
            regression_flags=regression_flags,
            is_valid=True,
            validity_notes=[],
            packing_frontier=packing_frontier,
            scenario_metadata=scenario_metadata,
            headline_baseline_name=headline_name,
            headline_baseline_rationale=headline_rationale,
            outcome=outcome,
        )

    @staticmethod
    def _packing_frontier(
        scenario_name: str,
        policy_results: dict[str, PolicyResult],
    ) -> Optional[list[dict[str, Any]]]:
        """Compute packing baselines for packing-relevant scenarios from FIFO state."""
        packing_keywords = ("fragmentation", "underutilization", "stranded",
                            "packing", "consolidation", "util")
        if not any(k in scenario_name for k in packing_keywords):
            return None
        fifo = policy_results.get(POLICY_FIFO)
        if fifo is None or fifo.final_state is None:
            return None
        try:
            from .packing import analyze_cluster_packing
            analyses = analyze_cluster_packing(fifo.final_state)
        except Exception:
            return None
        return [a.to_dict() for a in analyses] or None


def _error_result(scenario_name: str, error: str) -> BenchmarkResult:
    """Placeholder result for a scenario that failed to run."""
    metadata = BenchmarkMetadata(
        scenario_name=scenario_name,
        scenario_version="unknown",
        scenario_hash="error",
        seed=0,
        simulator_version="unknown",
        optimizer_version="unknown",
        config_hash="error",
        steps=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    scorecard = OptimizationScorecard(
        net_cost_improvement=0.0,
        sla_preservation=0.0,
        utilization_improvement=0.0,
        latency_improvement=0.0,
        thermal_improvement=0.0,
        migration_stability=0.0,
        topology_quality=0.0,
        weighted_score=0.0,
        flags=[f"ERROR: {error}"],
    )
    report = BenchmarkReport(
        metadata=metadata,
        aggregated={},
        scorecard=scorecard,
        expected_primary_constraint=None,
        observed_dominant_constraint=None,
        constraint_match=False,
        regression_flags=[f"ERROR: {error}"],
        is_valid=False,
        validity_notes=[error],
    )
    return BenchmarkResult(metadata=metadata, policy_results={}, report=report)
