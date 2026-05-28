"""Benchmark report models for Phase 11.

Contains metadata, scorecard, and KPI comparison structures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetadata:
    """Immutable metadata that must match exactly for a valid comparison."""
    scenario_name: str
    scenario_version: str
    scenario_hash: str          # SHA-256[:16] of the frozen YAML
    seed: int
    simulator_version: str
    optimizer_version: str      # semantic version of the constraint engine
    config_hash: str            # hash of serialized SimulatorConfig
    steps: int
    timestamp: str              # ISO-8601 UTC run time
    is_sandbox: bool = True     # always True — simulator output is never production

    @classmethod
    def build(
        cls,
        scenario_name: str,
        scenario_version: str,
        scenario_hash: str,
        seed: int,
        simulator_version: str,
        steps: int,
        config_dict: dict[str, Any],
    ) -> "BenchmarkMetadata":
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return cls(
            scenario_name=scenario_name,
            scenario_version=scenario_version,
            scenario_hash=scenario_hash,
            seed=seed,
            simulator_version=simulator_version,
            optimizer_version="1.0.0",
            config_hash=config_hash,
            steps=steps,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_version": self.scenario_version,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "simulator_version": self.simulator_version,
            "optimizer_version": self.optimizer_version,
            "config_hash": self.config_hash,
            "steps": self.steps,
            "timestamp": self.timestamp,
            "is_sandbox": self.is_sandbox,
        }

    def is_comparable_to(self, other: "BenchmarkMetadata") -> tuple[bool, list[str]]:
        """Return (compatible, list_of_mismatches) for regression gating."""
        mismatches: list[str] = []
        for field_name in (
            "scenario_name", "scenario_version", "scenario_hash",
            "seed", "simulator_version", "config_hash", "steps",
        ):
            v1 = getattr(self, field_name)
            v2 = getattr(other, field_name)
            if v1 != v2:
                mismatches.append(f"{field_name}: {v1!r} → {v2!r}")
        return (len(mismatches) == 0, mismatches)


# ---------------------------------------------------------------------------
# Per-tick KPI record
# ---------------------------------------------------------------------------

@dataclass
class TickKPI:
    """KPI snapshot for one simulated tick under one policy."""
    tick: int
    total_energy_cost: float
    total_tokens: int
    total_energy_kwh: float
    cost_per_token: Optional[float]
    tokens_per_joule: Optional[float]
    mean_gpu_util_pct: float
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    queue_wait_p95_ms: Optional[float]
    sla_violations: int
    thermal_throttle_gpu_count: int
    migration_count: int
    mean_topology_score: float
    # Canonical KPI inputs (per-tick).
    sla_compliant_tokens: int = 0
    active_gpu_count: int = 0
    active_gpu_hours_by_type: dict = field(default_factory=dict)
    # KV-cache / prefix-affinity / locality realism KPIs (optional)
    kv_pressure_max: Optional[float] = None
    prefix_hit_rate_mean: Optional[float] = None
    preemption_count: int = 0
    cold_reroute_count: int = 0
    cache_eviction_count: int = 0
    locality_confidence_mean: Optional[float] = None
    cache_fragmentation_frac_mean: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    # Migration / drain / cold-start realism KPIs (optional)
    reroute_count: int = 0
    migration_veto_count: int = 0
    batch_efficiency_mean: Optional[float] = None
    proxy_saturation_max: Optional[float] = None
    cold_start_count: int = 0
    rollback_count: int = 0
    overload_events: int = 0
    startup_latency_s_max: Optional[float] = None
    # Thermal / cooling / power realism KPIs (optional)
    max_gpu_temp_c: Optional[float] = None
    thermal_slowdown_pct_mean: Optional[float] = None
    thermal_throttle_events: int = 0
    hotspot_severity_max: Optional[float] = None
    rack_density_kw_max: Optional[float] = None
    thermal_excursions: int = 0
    thermal_migration_vetoes: int = 0
    # Topology / communication realism KPIs (optional)
    mean_topology_quality: Optional[float] = None
    fabric_congestion_max: Optional[float] = None
    collective_amplification_max: Optional[float] = None
    comm_throughput_penalty_pct_mean: Optional[float] = None
    sync_slowdown_pct_mean: Optional[float] = None
    nic_saturation_max: Optional[float] = None
    topology_risk_max: Optional[float] = None
    collective_instability_count: int = 0
    topology_migration_vetoes: int = 0
    comm_latency_p99_ms_max: Optional[float] = None
    # Utilization / fragmentation / bin-packing realism KPIs (optional)
    mean_effective_util: Optional[float] = None
    dram_active_max: Optional[float] = None
    fragmentation_score_max: Optional[float] = None
    stranded_gpu_count: int = 0
    packing_density_max: Optional[float] = None
    consolidation_risk_max: Optional[float] = None
    unsafe_consolidation_count: int = 0
    queue_amplification_max: Optional[float] = None
    util_throughput_penalty_pct_mean: Optional[float] = None
    utilization_paradox_count: int = 0
    bin_packing_risk_max: Optional[float] = None
    packing_migration_vetoes: int = 0
    # Energy / carbon / arbitrage realism KPIs (optional)
    day_ahead_price_mean: Optional[float] = None
    real_time_price_mean: Optional[float] = None
    da_rt_basis_max: Optional[float] = None
    lmp_congestion_max: Optional[float] = None
    carbon_intensity_mean: Optional[float] = None
    net_savings_sum: Optional[float] = None
    gross_savings_sum: Optional[float] = None
    energy_migration_vetoes: int = 0
    energy_actions_rejected: int = 0
    churn_penalty_max: Optional[float] = None
    low_energy_telemetry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "total_energy_cost": self.total_energy_cost,
            "total_tokens": self.total_tokens,
            "total_energy_kwh": self.total_energy_kwh,
            "cost_per_token": self.cost_per_token,
            "tokens_per_joule": self.tokens_per_joule,
            "mean_gpu_util_pct": self.mean_gpu_util_pct,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "queue_wait_p95_ms": self.queue_wait_p95_ms,
            "sla_violations": self.sla_violations,
            "thermal_throttle_gpu_count": self.thermal_throttle_gpu_count,
            "migration_count": self.migration_count,
            "mean_topology_score": self.mean_topology_score,
            "kv_pressure_max": self.kv_pressure_max,
            "prefix_hit_rate_mean": self.prefix_hit_rate_mean,
            "preemption_count": self.preemption_count,
            "cold_reroute_count": self.cold_reroute_count,
            "cache_eviction_count": self.cache_eviction_count,
            "locality_confidence_mean": self.locality_confidence_mean,
            "cache_fragmentation_frac_mean": self.cache_fragmentation_frac_mean,
            "ttft_p99_ms": self.ttft_p99_ms,
            "reroute_count": self.reroute_count,
            "migration_veto_count": self.migration_veto_count,
            "batch_efficiency_mean": self.batch_efficiency_mean,
            "proxy_saturation_max": self.proxy_saturation_max,
            "cold_start_count": self.cold_start_count,
            "rollback_count": self.rollback_count,
            "overload_events": self.overload_events,
            "startup_latency_s_max": self.startup_latency_s_max,
            "max_gpu_temp_c": self.max_gpu_temp_c,
            "thermal_slowdown_pct_mean": self.thermal_slowdown_pct_mean,
            "thermal_throttle_events": self.thermal_throttle_events,
            "hotspot_severity_max": self.hotspot_severity_max,
            "rack_density_kw_max": self.rack_density_kw_max,
            "thermal_excursions": self.thermal_excursions,
            "thermal_migration_vetoes": self.thermal_migration_vetoes,
            "mean_topology_quality": self.mean_topology_quality,
            "fabric_congestion_max": self.fabric_congestion_max,
            "collective_amplification_max": self.collective_amplification_max,
            "comm_throughput_penalty_pct_mean": self.comm_throughput_penalty_pct_mean,
            "sync_slowdown_pct_mean": self.sync_slowdown_pct_mean,
            "nic_saturation_max": self.nic_saturation_max,
            "topology_risk_max": self.topology_risk_max,
            "collective_instability_count": self.collective_instability_count,
            "topology_migration_vetoes": self.topology_migration_vetoes,
            "comm_latency_p99_ms_max": self.comm_latency_p99_ms_max,
            "mean_effective_util": self.mean_effective_util,
            "dram_active_max": self.dram_active_max,
            "fragmentation_score_max": self.fragmentation_score_max,
            "stranded_gpu_count": self.stranded_gpu_count,
            "packing_density_max": self.packing_density_max,
            "consolidation_risk_max": self.consolidation_risk_max,
            "unsafe_consolidation_count": self.unsafe_consolidation_count,
            "queue_amplification_max": self.queue_amplification_max,
            "util_throughput_penalty_pct_mean": self.util_throughput_penalty_pct_mean,
            "utilization_paradox_count": self.utilization_paradox_count,
            "bin_packing_risk_max": self.bin_packing_risk_max,
            "packing_migration_vetoes": self.packing_migration_vetoes,
            "day_ahead_price_mean": self.day_ahead_price_mean,
            "real_time_price_mean": self.real_time_price_mean,
            "da_rt_basis_max": self.da_rt_basis_max,
            "lmp_congestion_max": self.lmp_congestion_max,
            "carbon_intensity_mean": self.carbon_intensity_mean,
            "net_savings_sum": self.net_savings_sum,
            "gross_savings_sum": self.gross_savings_sum,
            "energy_migration_vetoes": self.energy_migration_vetoes,
            "energy_actions_rejected": self.energy_actions_rejected,
            "churn_penalty_max": self.churn_penalty_max,
            "low_energy_telemetry_count": self.low_energy_telemetry_count,
        }


@dataclass
class AggregatedKPI:
    """Aggregated KPI summary across all ticks for one policy."""
    policy_name: str
    total_energy_cost: float
    total_tokens: int
    total_energy_kwh: float
    mean_cost_per_token: Optional[float]      # None if no tokens were served
    mean_tokens_per_joule: Optional[float]    # None if no energy used
    mean_gpu_util_pct: float
    p99_latency_ms: Optional[float]           # max p99 across ticks
    p95_latency_ms: Optional[float]
    p95_queue_wait_ms: Optional[float]
    total_sla_violations: int
    total_thermal_throttle_ticks: int
    total_migrations: int
    mean_topology_score: float
    # --- Canonical primary KPI: SLA-safe goodput per infrastructure dollar ---
    # Numerator: SLA-compliant goodput (tokens that met their workload's SLO).
    # Denominator: gpu_infra_cost + energy_cost + network_cost.
    # Per the spec, secondary KPIs are NOT folded into this metric — they remain
    # diagnostics / constraints / vetoes below.
    sla_compliant_goodput: int = 0
    gpu_infra_cost: float = 0.0
    energy_cost: float = 0.0        # mirrors total_energy_cost for symmetry
    network_cost: float = 0.0
    total_infrastructure_cost: float = 0.0
    sla_safe_goodput_per_infra_dollar: Optional[float] = None
    cost_per_sla_compliant_token: Optional[float] = None
    active_gpu_hours: float = 0.0
    active_gpu_hours_by_type: dict = field(default_factory=dict)
    # Secondary derived KPI (diagnostic only, not part of the primary).
    goodput_per_gpu_hour: Optional[float] = None
    # KV-cache / prefix-affinity / locality realism KPIs (optional)
    kv_pressure_max: Optional[float] = None
    prefix_hit_rate_mean: Optional[float] = None
    total_preemptions: int = 0
    total_cold_reroutes: int = 0
    total_cache_evictions: int = 0
    locality_confidence_mean: Optional[float] = None
    cache_fragmentation_frac_mean: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    # Migration / drain / cold-start realism KPIs (optional)
    total_reroutes: int = 0
    total_migration_vetoes: int = 0
    batch_efficiency_mean: Optional[float] = None
    proxy_saturation_max: Optional[float] = None
    total_cold_starts: int = 0
    total_rollbacks: int = 0
    total_overload_events: int = 0
    startup_latency_s_max: Optional[float] = None
    # Thermal / cooling / power realism KPIs (optional)
    max_gpu_temp_c: Optional[float] = None
    thermal_slowdown_pct_mean: Optional[float] = None
    total_thermal_throttle_events: int = 0
    hotspot_severity_max: Optional[float] = None
    rack_density_kw_max: Optional[float] = None
    total_thermal_excursions: int = 0
    total_thermal_migration_vetoes: int = 0
    # Topology / communication realism KPIs (optional)
    mean_topology_quality: Optional[float] = None
    min_topology_quality: Optional[float] = None
    fabric_congestion_max: Optional[float] = None
    collective_amplification_max: Optional[float] = None
    comm_throughput_penalty_pct_mean: Optional[float] = None
    sync_slowdown_pct_mean: Optional[float] = None
    nic_saturation_max: Optional[float] = None
    topology_risk_max: Optional[float] = None
    total_collective_instability: int = 0
    total_topology_migration_vetoes: int = 0
    comm_latency_p99_ms_max: Optional[float] = None
    # Utilization / fragmentation / bin-packing realism KPIs (optional)
    mean_effective_util: Optional[float] = None
    dram_active_max: Optional[float] = None
    fragmentation_score_max: Optional[float] = None
    stranded_gpu_count_max: int = 0
    packing_density_max: Optional[float] = None
    consolidation_risk_max: Optional[float] = None
    total_unsafe_consolidation: int = 0
    queue_amplification_max: Optional[float] = None
    util_throughput_penalty_pct_mean: Optional[float] = None
    total_utilization_paradox: int = 0
    bin_packing_risk_max: Optional[float] = None
    total_packing_migration_vetoes: int = 0
    # Energy / carbon / arbitrage realism KPIs (optional)
    day_ahead_price_mean: Optional[float] = None
    real_time_price_mean: Optional[float] = None
    da_rt_basis_max: Optional[float] = None
    lmp_congestion_max: Optional[float] = None
    carbon_intensity_mean: Optional[float] = None
    total_net_savings: Optional[float] = None
    total_gross_savings: Optional[float] = None
    total_energy_migration_vetoes: int = 0
    total_energy_actions_rejected: int = 0
    churn_penalty_max: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        # Primary KPI (the canonical headline metric) and its cost breakdown go
        # FIRST in the JSON output so it's the first thing a reader sees.
        import math as _math
        cpsct = self.cost_per_sla_compliant_token
        return {
            "policy": self.policy_name,
            # --- Primary canonical KPI + components ---
            "sla_safe_goodput_per_infra_dollar": (
                None if self.sla_safe_goodput_per_infra_dollar is None
                else round(self.sla_safe_goodput_per_infra_dollar, 4)
            ),
            "cost_per_sla_compliant_token": (
                None if cpsct is None
                else (_math.inf if _math.isinf(cpsct) else round(cpsct, 10))
            ),
            "sla_compliant_goodput": self.sla_compliant_goodput,
            "total_infrastructure_cost": round(self.total_infrastructure_cost, 4),
            "gpu_infra_cost": round(self.gpu_infra_cost, 4),
            "network_cost": round(self.network_cost, 4),
            "active_gpu_hours": round(self.active_gpu_hours, 4),
            "active_gpu_hours_by_type": {
                k: round(v, 4) for k, v in self.active_gpu_hours_by_type.items()
            },
            "goodput_per_gpu_hour": (
                round(self.goodput_per_gpu_hour, 4)
                if self.goodput_per_gpu_hour is not None else None
            ),
            # --- Secondary KPIs (constraints, vetoes, diagnostics) ---
            "total_energy_cost": round(self.total_energy_cost, 4),
            "energy_cost": round(self.energy_cost, 4),
            "total_tokens": self.total_tokens,
            "total_energy_kwh": round(self.total_energy_kwh, 4),
            "mean_cost_per_token": (
                round(self.mean_cost_per_token, 8) if self.mean_cost_per_token is not None else None
            ),
            "mean_tokens_per_joule": (
                round(self.mean_tokens_per_joule, 6)
                if self.mean_tokens_per_joule is not None else None
            ),
            "mean_gpu_util_pct": round(self.mean_gpu_util_pct, 2),
            "p99_latency_ms": (
                round(self.p99_latency_ms, 1) if self.p99_latency_ms is not None else None
            ),
            "p95_latency_ms": (
                round(self.p95_latency_ms, 1) if self.p95_latency_ms is not None else None
            ),
            "p95_queue_wait_ms": (
                round(self.p95_queue_wait_ms, 1) if self.p95_queue_wait_ms is not None else None
            ),
            "total_sla_violations": self.total_sla_violations,
            "total_thermal_throttle_ticks": self.total_thermal_throttle_ticks,
            "total_migrations": self.total_migrations,
            "mean_topology_score": round(self.mean_topology_score, 3),
            "kv_pressure_max": (
                round(self.kv_pressure_max, 3) if self.kv_pressure_max is not None else None
            ),
            "prefix_hit_rate_mean": (
                round(self.prefix_hit_rate_mean, 3)
                if self.prefix_hit_rate_mean is not None else None
            ),
            "total_preemptions": self.total_preemptions,
            "total_cold_reroutes": self.total_cold_reroutes,
            "total_cache_evictions": self.total_cache_evictions,
            "locality_confidence_mean": (
                round(self.locality_confidence_mean, 3)
                if self.locality_confidence_mean is not None else None
            ),
            "cache_fragmentation_frac_mean": (
                round(self.cache_fragmentation_frac_mean, 4)
                if self.cache_fragmentation_frac_mean is not None else None
            ),
            "ttft_p99_ms": (
                round(self.ttft_p99_ms, 1) if self.ttft_p99_ms is not None else None
            ),
            "total_reroutes": self.total_reroutes,
            "total_migration_vetoes": self.total_migration_vetoes,
            "batch_efficiency_mean": (
                round(self.batch_efficiency_mean, 3)
                if self.batch_efficiency_mean is not None else None
            ),
            "proxy_saturation_max": (
                round(self.proxy_saturation_max, 3)
                if self.proxy_saturation_max is not None else None
            ),
            "total_cold_starts": self.total_cold_starts,
            "total_rollbacks": self.total_rollbacks,
            "total_overload_events": self.total_overload_events,
            "startup_latency_s_max": (
                round(self.startup_latency_s_max, 1)
                if self.startup_latency_s_max is not None else None
            ),
            "max_gpu_temp_c": (
                round(self.max_gpu_temp_c, 1) if self.max_gpu_temp_c is not None else None
            ),
            "thermal_slowdown_pct_mean": (
                round(self.thermal_slowdown_pct_mean, 2)
                if self.thermal_slowdown_pct_mean is not None else None
            ),
            "total_thermal_throttle_events": self.total_thermal_throttle_events,
            "hotspot_severity_max": (
                round(self.hotspot_severity_max, 3)
                if self.hotspot_severity_max is not None else None
            ),
            "rack_density_kw_max": (
                round(self.rack_density_kw_max, 2)
                if self.rack_density_kw_max is not None else None
            ),
            "total_thermal_excursions": self.total_thermal_excursions,
            "total_thermal_migration_vetoes": self.total_thermal_migration_vetoes,
            "mean_topology_quality": (
                round(self.mean_topology_quality, 3)
                if self.mean_topology_quality is not None else None
            ),
            "min_topology_quality": (
                round(self.min_topology_quality, 3)
                if self.min_topology_quality is not None else None
            ),
            "fabric_congestion_max": (
                round(self.fabric_congestion_max, 3)
                if self.fabric_congestion_max is not None else None
            ),
            "collective_amplification_max": (
                round(self.collective_amplification_max, 2)
                if self.collective_amplification_max is not None else None
            ),
            "comm_throughput_penalty_pct_mean": (
                round(self.comm_throughput_penalty_pct_mean, 2)
                if self.comm_throughput_penalty_pct_mean is not None else None
            ),
            "sync_slowdown_pct_mean": (
                round(self.sync_slowdown_pct_mean, 2)
                if self.sync_slowdown_pct_mean is not None else None
            ),
            "nic_saturation_max": (
                round(self.nic_saturation_max, 3)
                if self.nic_saturation_max is not None else None
            ),
            "topology_risk_max": (
                round(self.topology_risk_max, 3)
                if self.topology_risk_max is not None else None
            ),
            "total_collective_instability": self.total_collective_instability,
            "total_topology_migration_vetoes": self.total_topology_migration_vetoes,
            "comm_latency_p99_ms_max": (
                round(self.comm_latency_p99_ms_max, 1)
                if self.comm_latency_p99_ms_max is not None else None
            ),
            "mean_effective_util": (
                round(self.mean_effective_util, 3)
                if self.mean_effective_util is not None else None
            ),
            "dram_active_max": (
                round(self.dram_active_max, 3) if self.dram_active_max is not None else None
            ),
            "fragmentation_score_max": (
                round(self.fragmentation_score_max, 3)
                if self.fragmentation_score_max is not None else None
            ),
            "stranded_gpu_count_max": self.stranded_gpu_count_max,
            "packing_density_max": (
                round(self.packing_density_max, 3)
                if self.packing_density_max is not None else None
            ),
            "consolidation_risk_max": (
                round(self.consolidation_risk_max, 3)
                if self.consolidation_risk_max is not None else None
            ),
            "total_unsafe_consolidation": self.total_unsafe_consolidation,
            "queue_amplification_max": (
                round(self.queue_amplification_max, 2)
                if self.queue_amplification_max is not None else None
            ),
            "util_throughput_penalty_pct_mean": (
                round(self.util_throughput_penalty_pct_mean, 2)
                if self.util_throughput_penalty_pct_mean is not None else None
            ),
            "total_utilization_paradox": self.total_utilization_paradox,
            "bin_packing_risk_max": (
                round(self.bin_packing_risk_max, 3)
                if self.bin_packing_risk_max is not None else None
            ),
            "total_packing_migration_vetoes": self.total_packing_migration_vetoes,
            "day_ahead_price_mean": (
                round(self.day_ahead_price_mean, 2)
                if self.day_ahead_price_mean is not None else None
            ),
            "real_time_price_mean": (
                round(self.real_time_price_mean, 2)
                if self.real_time_price_mean is not None else None
            ),
            "da_rt_basis_max": (
                round(self.da_rt_basis_max, 2) if self.da_rt_basis_max is not None else None
            ),
            "lmp_congestion_max": (
                round(self.lmp_congestion_max, 2)
                if self.lmp_congestion_max is not None else None
            ),
            "carbon_intensity_mean": (
                round(self.carbon_intensity_mean, 1)
                if self.carbon_intensity_mean is not None else None
            ),
            "total_net_savings": (
                round(self.total_net_savings, 4)
                if self.total_net_savings is not None else None
            ),
            "total_gross_savings": (
                round(self.total_gross_savings, 4)
                if self.total_gross_savings is not None else None
            ),
            "total_energy_migration_vetoes": self.total_energy_migration_vetoes,
            "total_energy_actions_rejected": self.total_energy_actions_rejected,
            "churn_penalty_max": (
                round(self.churn_penalty_max, 4)
                if self.churn_penalty_max is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Optimization scorecard
# ---------------------------------------------------------------------------

# Weights must sum to 1.0
_SCORECARD_WEIGHTS = {
    "net_cost_improvement": 0.25,
    "sla_preservation": 0.25,
    "utilization_improvement": 0.15,
    "latency_improvement": 0.15,
    "thermal_improvement": 0.05,
    "migration_stability": 0.10,
    "topology_quality": 0.05,
}


@dataclass
class OptimizationScorecard:
    """Weighted scorecard for constraint-aware optimizer vs FIFO baseline.

    All sub-scores are in [0, 1]; 1 = best possible.
    Missing data (None) is treated as 0.5 (neutral, not a win or loss).
    """
    net_cost_improvement: float       # relative cost reduction vs fifo
    sla_preservation: float           # 1 - sla_violation_rate
    utilization_improvement: float    # relative GPU util gain vs fifo
    latency_improvement: float        # relative p99 improvement vs fifo
    thermal_improvement: float        # 1 - throttle_fraction
    migration_stability: float        # 1 - (migrations / max_migrations_threshold)
    topology_quality: float           # mean topology score
    weighted_score: float             # weighted combination
    flags: list[str] = field(default_factory=list)  # degradation warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "net_cost_improvement": round(self.net_cost_improvement, 3),
            "sla_preservation": round(self.sla_preservation, 3),
            "utilization_improvement": round(self.utilization_improvement, 3),
            "latency_improvement": round(self.latency_improvement, 3),
            "thermal_improvement": round(self.thermal_improvement, 3),
            "migration_stability": round(self.migration_stability, 3),
            "topology_quality": round(self.topology_quality, 3),
            "weighted_score": round(self.weighted_score, 3),
            "flags": self.flags,
        }


def build_scorecard(
    constraint_aware: AggregatedKPI,
    fifo: AggregatedKPI,
    steps: int,
) -> OptimizationScorecard:
    """Build a weighted scorecard comparing constraint_aware vs fifo."""
    flags: list[str] = []

    # Cost improvement: (fifo_cost - ca_cost) / fifo_cost, clipped [0,1]
    if fifo.total_energy_cost > 0:
        cost_delta = (
            (fifo.total_energy_cost - constraint_aware.total_energy_cost)
            / fifo.total_energy_cost
        )
        net_cost = max(0.0, min(1.0, 0.5 + cost_delta))  # center at 0.5
    else:
        net_cost = 0.5

    # Flag a COST regression on EFFICIENCY (cost per token), not absolute cost:
    # acting on a constraint (e.g. scaling replicas to clear a queue) legitimately
    # raises total energy while serving proportionally more tokens. Absolute cost
    # alone would false-flag those throughput-positive wins.
    def _cost_per_token(k: AggregatedKPI) -> Optional[float]:
        return k.total_energy_cost / k.total_tokens if k.total_tokens > 0 else None

    ca_cpt = _cost_per_token(constraint_aware)
    fifo_cpt = _cost_per_token(fifo)
    if ca_cpt is not None and fifo_cpt is not None and fifo_cpt > 0:
        if ca_cpt > fifo_cpt * 1.02:
            flags.append(
                f"COST_REGRESSION: constraint_aware cost/token {ca_cpt:.3e} "
                f"> fifo {fifo_cpt:.3e} (efficiency, throughput-normalized)"
            )
    elif constraint_aware.total_energy_cost > fifo.total_energy_cost * 1.02:
        # No token signal — fall back to absolute cost.
        flags.append("COST_REGRESSION: constraint_aware costs more than fifo")

    # SLA preservation: 1 - violation_rate (lower is better)
    # If constraint_aware has more violations than fifo → flag
    max_possible_violations = max(steps * 2, 1)
    sla_score = max(0.0, 1.0 - constraint_aware.total_sla_violations / max_possible_violations)
    if constraint_aware.total_sla_violations > fifo.total_sla_violations:
        flags.append(
            f"SLA_REGRESSION: constraint_aware SLA violations "
            f"({constraint_aware.total_sla_violations}) > fifo ({fifo.total_sla_violations})"
        )

    # Utilization improvement: (ca_util - fifo_util) / 100, clipped [0,1]
    util_delta = (constraint_aware.mean_gpu_util_pct - fifo.mean_gpu_util_pct) / 100.0
    util_score = max(0.0, min(1.0, 0.5 + util_delta * 2))

    # Latency improvement: relative p99 reduction vs fifo
    if fifo.p99_latency_ms and constraint_aware.p99_latency_ms:
        lat_ratio = fifo.p99_latency_ms / max(constraint_aware.p99_latency_ms, 1.0)
        lat_score = max(0.0, min(1.0, lat_ratio / 2.0))
        if constraint_aware.p99_latency_ms > fifo.p99_latency_ms * 1.10:
            flags.append(
                f"LATENCY_REGRESSION: p99 {constraint_aware.p99_latency_ms:.0f}ms "
                f"> fifo {fifo.p99_latency_ms:.0f}ms"
            )
    else:
        lat_score = 0.5

    # Thermal improvement: 1 - throttle_fraction
    max_possible_throttle = steps * max(fifo.mean_gpu_util_pct / 10, 1)
    thermal_score = max(
        0.0,
        1.0 - constraint_aware.total_thermal_throttle_ticks / max(max_possible_throttle, 1),
    )
    if constraint_aware.total_thermal_throttle_ticks > fifo.total_thermal_throttle_ticks:
        flags.append("THERMAL_REGRESSION: more throttle events than fifo")

    # Migration stability: penalise excessive churn
    max_migrations_threshold = steps * 2  # >2 migrations/tick is operationally unacceptable
    migration_stability = max(
        0.0,
        1.0 - constraint_aware.total_migrations / max(max_migrations_threshold, 1),
    )
    if constraint_aware.total_migrations > max_migrations_threshold:
        flags.append(
            f"MIGRATION_CHURN: {constraint_aware.total_migrations} migrations "
            f"exceeds threshold {max_migrations_threshold}"
        )

    # Topology quality: direct score
    topology_score = max(0.0, min(1.0, constraint_aware.mean_topology_score))
    if constraint_aware.mean_topology_score < fifo.mean_topology_score - 0.1:
        flags.append("TOPOLOGY_REGRESSION: mean topology score degraded vs fifo")

    # Weighted final score
    weights = _SCORECARD_WEIGHTS
    weighted = (
        weights["net_cost_improvement"] * net_cost
        + weights["sla_preservation"] * sla_score
        + weights["utilization_improvement"] * util_score
        + weights["latency_improvement"] * lat_score
        + weights["thermal_improvement"] * thermal_score
        + weights["migration_stability"] * migration_stability
        + weights["topology_quality"] * topology_score
    )

    return OptimizationScorecard(
        net_cost_improvement=net_cost,
        sla_preservation=sla_score,
        utilization_improvement=util_score,
        latency_improvement=lat_score,
        thermal_improvement=thermal_score,
        migration_stability=migration_stability,
        topology_quality=topology_score,
        weighted_score=weighted,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Full benchmark report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Full benchmark report: metadata + per-policy KPIs + scorecard."""
    metadata: BenchmarkMetadata
    aggregated: dict[str, AggregatedKPI]   # policy_name → AggregatedKPI
    scorecard: OptimizationScorecard
    expected_primary_constraint: Optional[str]
    observed_dominant_constraint: Optional[str]
    constraint_match: bool
    regression_flags: list[str]
    is_valid: bool                          # False when metadata or env changed
    validity_notes: list[str]
    # Packing baseline frontier (first-fit / best-fit / FFD / clairvoyant) for
    # utilization/fragmentation scenarios — analysis-only, never a deployable policy.
    packing_frontier: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "kpi_comparison": {k: v.to_dict() for k, v in self.aggregated.items()},
            "scorecard": self.scorecard.to_dict(),
            "expected_primary_constraint": self.expected_primary_constraint,
            "observed_dominant_constraint": self.observed_dominant_constraint,
            "constraint_match": self.constraint_match,
            "regression_flags": self.regression_flags,
            "is_valid": self.is_valid,
            "validity_notes": self.validity_notes,
            "packing_frontier": self.packing_frontier,
        }

    def to_text(self) -> str:
        """Human-readable benchmark report."""
        lines: list[str] = []
        m = self.metadata
        lines.append(f"Aurelius Constraint-Aware Benchmark — {m.timestamp[:19]}Z")
        lines.append(f"Scenario:          {m.scenario_name} ({m.scenario_version})")
        lines.append(f"Scenario hash:     {m.scenario_hash}")
        lines.append(f"Seed:              {m.seed}")
        lines.append(f"Steps:             {m.steps}")
        lines.append(f"Simulator version: {m.simulator_version}")
        lines.append(f"Optimizer version: {m.optimizer_version}")
        lines.append(f"Config hash:       {m.config_hash}")
        lines.append("[SANDBOX]          All outputs are synthetic. Not for production claims.")
        lines.append("")

        # Constraint validation
        exp = self.expected_primary_constraint or "N/A"
        obs = self.observed_dominant_constraint or "N/A"
        match_str = "MATCHES" if self.constraint_match else "MISMATCH"
        lines.append(f"Constraint check:  expected={exp!r}  observed={obs!r}  [{match_str}]")
        lines.append("")

        # KPI comparison table
        policies = ["fifo", "current_price_only", "greedy_energy", "sla_aware", "constraint_aware"]
        available = [p for p in policies if p in self.aggregated]

        col_w = 20
        header_parts = ["Metric".ljust(30)]
        for p in available:
            header_parts.append(p[:col_w].ljust(col_w))
        lines.append("  ".join(header_parts))
        lines.append("-" * (32 + col_w * len(available) + 2 * len(available)))

        def row(label: str, getter) -> str:
            parts = [label.ljust(30)]
            for p in available:
                kpi = self.aggregated[p]
                val = getter(kpi)
                parts.append((str(val) if val is not None else "N/A").ljust(col_w))
            return "  ".join(parts)

        # --- Primary canonical KPI (the headline benchmark metric) ---
        lines.append("Primary KPI: SLA-safe goodput per infrastructure dollar")
        lines.append("=" * (32 + col_w * len(available) + 2 * len(available)))
        lines.append(row(
            "goodput / $infra",
            lambda k: (
                f"{k.sla_safe_goodput_per_infra_dollar:,.0f}"
                if k.sla_safe_goodput_per_infra_dollar is not None else "N/A"
            ),
        ))
        import math as _math
        lines.append(row(
            "$ / SLA-compliant token",
            lambda k: (
                "inf" if k.cost_per_sla_compliant_token is not None
                and _math.isinf(k.cost_per_sla_compliant_token)
                else (f"{k.cost_per_sla_compliant_token:.3e}"
                      if k.cost_per_sla_compliant_token is not None else "N/A")
            ),
        ))
        lines.append(row("SLA-compliant goodput",
                         lambda k: f"{k.sla_compliant_goodput:,}"))
        lines.append(row("Total infra cost ($)",
                         lambda k: f"{k.total_infrastructure_cost:.2f}"))
        lines.append(row("  GPU infra ($)", lambda k: f"{k.gpu_infra_cost:.2f}"))
        lines.append(row("  Energy ($)", lambda k: f"{k.energy_cost:.4f}"))
        lines.append(row("  Network ($)", lambda k: f"{k.network_cost:.2f}"))
        lines.append(row("Active GPU-hours",
                         lambda k: f"{k.active_gpu_hours:.1f}"))
        lines.append("")
        lines.append("Secondary KPIs (diagnostics — NOT folded into the primary KPI):")
        lines.append("-" * (32 + col_w * len(available) + 2 * len(available)))
        lines.append(row("Total energy cost ($)", lambda k: f"{k.total_energy_cost:.4f}"))
        lines.append(row("Total raw tokens", lambda k: f"{k.total_tokens:,}"))
        lines.append(row("Mean GPU util (%)", lambda k: f"{k.mean_gpu_util_pct:.1f}"))
        lines.append(row("p99 latency (ms)", lambda k: f"{k.p99_latency_ms:.0f}" if k.p99_latency_ms else "N/A"))
        lines.append(row("p95 queue wait (ms)", lambda k: f"{k.p95_queue_wait_ms:.0f}" if k.p95_queue_wait_ms else "N/A"))
        lines.append(row("SLA violations", lambda k: str(k.total_sla_violations)))
        lines.append(row("Thermal throttle ticks", lambda k: str(k.total_thermal_throttle_ticks)))
        lines.append(row("Migrations", lambda k: str(k.total_migrations)))
        lines.append(row("Mean topology score", lambda k: f"{k.mean_topology_score:.3f}"))
        lines.append(row("Mean cost/token ($)", lambda k: f"{k.mean_cost_per_token:.6f}" if k.mean_cost_per_token else "N/A"))
        lines.append(row(
            "Goodput per GPU-hour",
            lambda k: (f"{k.goodput_per_gpu_hour:,.0f}"
                       if k.goodput_per_gpu_hour is not None else "N/A"),
        ))
        lines.append("")

        # Scorecard
        sc = self.scorecard
        lines.append("Optimization Scorecard (constraint_aware vs fifo):")
        lines.append(f"  Net cost improvement:   {sc.net_cost_improvement:.3f}")
        lines.append(f"  SLA preservation:       {sc.sla_preservation:.3f}")
        lines.append(f"  Utilization improvement:{sc.utilization_improvement:.3f}")
        lines.append(f"  Latency improvement:    {sc.latency_improvement:.3f}")
        lines.append(f"  Thermal improvement:    {sc.thermal_improvement:.3f}")
        lines.append(f"  Migration stability:    {sc.migration_stability:.3f}")
        lines.append(f"  Topology quality:       {sc.topology_quality:.3f}")
        lines.append("  ─────────────────────────────")
        lines.append(f"  Weighted score:         {sc.weighted_score:.3f}")
        lines.append("")

        if sc.flags:
            lines.append("Regression flags:")
            for flag in sc.flags:
                lines.append(f"  ⚠  {flag}")
            lines.append("")

        if self.regression_flags:
            lines.append("Cross-run regression flags:")
            for flag in self.regression_flags:
                lines.append(f"  ✗  {flag}")
            lines.append("")

        if self.packing_frontier:
            lines.append("")
            lines.append("Packing baselines (analysis-only — first-fit/best-fit/FFD/clairvoyant):")
            for pf in self.packing_frontier:
                region = pf.get("region", "?")
                cur = pf.get("current_active_nodes", "?")
                avail = pf.get("nodes_available", "?")
                cap = pf.get("bin_capacity", "?")
                lines.append(
                    f"  region {region}: {cur}/{avail} nodes active now "
                    f"(node capacity={cap} GPUs)"
                )
                for name, r in pf.get("results", {}).items():
                    lines.append(
                        f"     {name:<24} → {r['bins_used']} nodes, "
                        f"{r['stranded_gpus']} stranded, density={r['packing_density']}"
                    )
            lines.append("  (clairvoyant is an optimal floor for analysis, not deployable.)")

        lines.append("")
        validity_str = "VALID" if self.is_valid else "INVALID (environment changed)"
        lines.append(f"Comparison validity: {validity_str}")
        for note in self.validity_notes:
            lines.append(f"  → {note}")

        return "\n".join(lines)
