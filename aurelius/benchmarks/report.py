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

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "total_energy_cost": round(self.total_energy_cost, 4),
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

        lines.append(row("Total energy cost ($)", lambda k: f"{k.total_energy_cost:.4f}"))
        lines.append(row("Total tokens served", lambda k: str(k.total_tokens)))
        lines.append(row("Mean GPU util (%)", lambda k: f"{k.mean_gpu_util_pct:.1f}"))
        lines.append(row("p99 latency (ms)", lambda k: f"{k.p99_latency_ms:.0f}" if k.p99_latency_ms else "N/A"))
        lines.append(row("p95 queue wait (ms)", lambda k: f"{k.p95_queue_wait_ms:.0f}" if k.p95_queue_wait_ms else "N/A"))
        lines.append(row("SLA violations", lambda k: str(k.total_sla_violations)))
        lines.append(row("Thermal throttle ticks", lambda k: str(k.total_thermal_throttle_ticks)))
        lines.append(row("Migrations", lambda k: str(k.total_migrations)))
        lines.append(row("Mean topology score", lambda k: f"{k.mean_topology_score:.3f}"))
        lines.append(row("Mean cost/token ($)", lambda k: f"{k.mean_cost_per_token:.6f}" if k.mean_cost_per_token else "N/A"))
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

        validity_str = "VALID" if self.is_valid else "INVALID (environment changed)"
        lines.append(f"Comparison validity: {validity_str}")
        for note in self.validity_notes:
            lines.append(f"  → {note}")

        return "\n".join(lines)
