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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..constraints.engine import ConstraintAwareEngine, EngineResult
from ..simulation.cluster.engine import ClusterSimulator, TickMetrics
from ..simulation.cluster.scenarios import ScenarioConfig, list_scenarios, load_scenario
from ..state.models import ClusterState
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

    # Find cheapest region by real_time_price or day_ahead_price from EnergyState
    def price(r):
        if r.energy is not None:
            p = (r.energy.real_time_price_per_mwh
                 if r.energy.real_time_price_per_mwh is not None
                 else r.energy.day_ahead_price_per_mwh)
            if p is not None:
                return p
        # Fall back to sim cluster price for this region
        sim_r = sim._cluster.regions.get(r.region)
        if sim_r is not None:
            return sim_r.current_energy_price
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
            p = (r.energy.real_time_price_per_mwh
                 if r.energy.real_time_price_per_mwh is not None
                 else r.energy.day_ahead_price_per_mwh)
            if p is not None:
                return p
        sim_r = sim._cluster.regions.get(r.region)
        if sim_r is not None:
            return sim_r.current_energy_price
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
        # Use sim region price directly (already updated by simulator tick)
        sim_price = sim_region.current_energy_price
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
        if rec.action_type != "CHOOSE_CHEAPER_REGION":
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


def _apply_constraint_aware(
    sim: ClusterSimulator,
    state: ClusterState,
    engine: ConstraintAwareEngine,
    migration_log: list,
) -> Optional[EngineResult]:
    """Run full ConstraintAwareEngine and apply all safe migration recommendations.

    rec.workload_id is the service_id; rec.target_region is the destination region.
    Only actions with a cross-region target_region trigger simulator migrations.
    """
    er = engine.run(state)

    for rec in er.recommendations:
        if rec.is_noop:
            continue
        # Only apply actions that result in a cross-region migration
        if rec.action_type not in ("CHOOSE_CHEAPER_REGION", "MIGRATE", "CHANGE_PLACEMENT"):
            continue
        target_region = rec.target_region
        if target_region is None:
            continue
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

def _aggregate_kpis(policy_name: str, tick_kpis: list[TickKPI]) -> AggregatedKPI:
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
        total_thermal_throttle_ticks=sum(k.thermal_throttle_gpu_count for k in tick_kpis),
        total_migrations=tick_kpis[-1].migration_count if tick_kpis else 0,
        mean_topology_score=sum(topo_vals) / len(topo_vals) if topo_vals else 1.0,
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
    ):
        self._policies = policies or ALL_POLICIES
        from ..constraints.classifier import ConstraintConfig
        self._engine = ConstraintAwareEngine(
            classifier_config=ConstraintConfig(confidence_floor=confidence_floor)
        )

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

        for _ in range(steps):
            tick_result = sim.tick()
            # Apply optimizer AFTER observing tick state (before next tick)
            state = tick_result.cluster_state
            er = apply_fn(sim, state, self._engine, migration_log)
            engine_results.append(er)
            tick_kpis.append(_tick_metrics_to_kpi(tick_result.metrics))

        return PolicyResult(
            policy_name=policy_name,
            tick_kpis=tick_kpis,
            engine_results=engine_results,
            migration_log=migration_log,
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
            aggregated[policy_name] = _aggregate_kpis(policy_name, pr.tick_kpis)

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
        # Normalize expected: YAML uses "energy_bound", enum uses "energy"
        expected_normalized = expected.removesuffix("_bound") if expected else None
        constraint_match = (
            observed_dominant == expected_normalized
            if expected_normalized and observed_dominant
            else True
        )

        regression_flags: list[str] = list(scorecard.flags)

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
        )


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
