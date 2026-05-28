"""Aurelius canonical benchmark KPI — SLA-safe goodput per infrastructure dollar.

The previous primary metric (raw energy cost) is the wrong objective for full
constraint-aware orchestration: it rewards starving real customer SLOs to save
electricity and punishes safe consolidation that uses more energy now to prevent
SLA-violating thrash later. This module computes the spec's canonical metric:

    sla_safe_goodput_per_infrastructure_dollar =
        SLA_compliant_goodput
        / (gpu_infra_cost + energy_cost + network_cost)

Design rules (per the mission spec):
- No synthetic workload-value weights, no business-value multipliers, no hidden
  weighted SLA penalty dollars. SLA is a *filter on the numerator*, not a
  subtraction term in the cost denominator.
- Secondary KPIs (p99, queue wait, churn, thermal, topology …) remain
  constraints / vetoes / diagnostics — never folded into this headline KPI.
- All quantities are directly produced by the simulator/telemetry stack: queue
  timeout-rate (SLA filter), allocated GPU-hours (billable footprint), tracked
  energy cost, and a configurable network cost. No magic constants.
- Pure deterministic functions; no global state, no I/O.

Honest limits:
- GPU hour prices are documented public-list ballpark defaults; operators MUST
  override `InfrastructureCostConfig.gpu_hour_prices` with their actual contract
  rates before any external claim. The same applies to `network_cost_per_gb` /
  `network_cost_per_migration`.
- The simulator's SLA signal is per-queue `timeout_rate_pct` (engine.py:1758),
  which is an aggregate approximation of the share of work whose p99 exceeded the
  workload's configured SLO. Per-request SLA outcomes are not modelled; this
  module computes goodput as `tokens × (1 − timeout_rate_pct/100)` per queue per
  tick, which is the most faithful reading of the available signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Documented public-list ballpark on-demand cloud GPU prices ($/hr).
# These are DEFAULTS for the simulator benchmark only. Override per-deployment.
# Sources (last surveyed 2026-02): public cloud on-demand list pages — AWS p4d,
# p5, GCP a2/a3, Azure NDv4, Lambda Labs, CoreWeave. Treat as ±50% priors.
# ---------------------------------------------------------------------------

_DEFAULT_GPU_HOUR_PRICES: dict[str, float] = {
    "NVIDIA H100 SXM5 80GB": 3.00,   # H100 SXM5 — public-list on-demand ballpark
    "NVIDIA A100 SXM4 80GB": 2.00,   # A100 SXM4 80GB
    "NVIDIA A100 PCIe 80GB": 1.50,
    "NVIDIA L4": 0.75,
}

# Fallback when a GPU type is not in the table — deliberately conservative.
_FALLBACK_GPU_HOUR_PRICE: float = 2.00


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InfrastructureCostConfig:
    """Operator-overridable infrastructure cost basis.

    All defaults are PUBLIC-LIST PRIORS for the simulator only; production
    accounting requires the operator's real contract rates.
    """
    gpu_hour_prices: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_GPU_HOUR_PRICES)
    )
    fallback_gpu_hour_price: float = _FALLBACK_GPU_HOUR_PRICE
    # Network: 0.0 by default per the spec — do NOT invent network penalties.
    # Operators with real egress contracts populate these.
    network_cost_per_migration: float = 0.0
    network_cost_per_gb_egress: float = 0.0

    def gpu_price(self, gpu_type: str) -> float:
        return self.gpu_hour_prices.get(gpu_type, self.fallback_gpu_hour_price)


@dataclass(frozen=True)
class SLAFilterConfig:
    """Knobs for the SLA filter on the goodput numerator.

    The simulator's per-queue ``timeout_rate_pct`` already encodes the share of
    work whose p99 exceeded the workload's configured SLO; this config lets a
    caller add a noise/floor threshold (work that is *almost* SLA-compliant is
    still excluded — there is no partial credit by default) and clamp the
    compliant share to a safe range. No business-value weights.
    """
    # Minimum compliance share to count any tokens at all (floor). 0.0 = never
    # floor; 0.5 = exclude all tokens from a queue whose >50% of work violated.
    min_compliant_share: float = 0.0
    # Hard timeout-rate above which ALL of a queue's tokens are excluded
    # (everything timed out). The simulator caps timeout_rate_pct at 50 in
    # `engine.py`, so values above this default of 50.0 are a no-op.
    hard_exclude_timeout_rate_pct: float = 50.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EconomicKPIResult:
    """Canonical KPI result. All amounts are in dollars."""
    sla_compliant_goodput: int
    raw_tokens: int
    gpu_infra_cost: float
    energy_cost: float
    network_cost: float
    total_infrastructure_cost: float
    sla_safe_goodput_per_infra_dollar: Optional[float]
    cost_per_sla_compliant_token: Optional[float]
    active_gpu_hours: float
    # Diagnostics (not used inside the KPI; kept for honest breakdown).
    active_gpu_hours_by_type: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sla_compliant_goodput": self.sla_compliant_goodput,
            "raw_tokens": self.raw_tokens,
            "gpu_infra_cost": round(self.gpu_infra_cost, 4),
            "energy_cost": round(self.energy_cost, 4),
            "network_cost": round(self.network_cost, 4),
            "total_infrastructure_cost": round(self.total_infrastructure_cost, 4),
            "sla_safe_goodput_per_infra_dollar": (
                None if self.sla_safe_goodput_per_infra_dollar is None
                else round(self.sla_safe_goodput_per_infra_dollar, 4)
            ),
            "cost_per_sla_compliant_token": (
                None if self.cost_per_sla_compliant_token is None
                else (math.inf if math.isinf(self.cost_per_sla_compliant_token)
                      else round(self.cost_per_sla_compliant_token, 8))
            ),
            "active_gpu_hours": round(self.active_gpu_hours, 4),
            "active_gpu_hours_by_type": {
                k: round(v, 4) for k, v in self.active_gpu_hours_by_type.items()
            },
        }


# ---------------------------------------------------------------------------
# Pure functions (the spec's interface)
# ---------------------------------------------------------------------------

def compute_sla_compliant_goodput(
    tokens_per_tick: list[int],
    timeout_rate_pct_per_tick: list[float],
    *,
    sla_filter: Optional[SLAFilterConfig] = None,
) -> int:
    """Per-tick goodput = tokens × max(0, 1 − timeout_rate/100), summed.

    Per the spec, SLA-violating tokens MUST NOT count. This is the canonical
    SLA-filter form: timeout_rate_pct is the share of work that violated the SLO,
    so its complement is the share that met it.
    """
    if len(tokens_per_tick) != len(timeout_rate_pct_per_tick):
        raise ValueError("tokens_per_tick and timeout_rate_pct_per_tick lengths must match")
    cfg = sla_filter or SLAFilterConfig()
    total = 0
    for tokens, tr in zip(tokens_per_tick, timeout_rate_pct_per_tick):
        tr = max(0.0, min(100.0, tr))
        if tr >= cfg.hard_exclude_timeout_rate_pct:
            continue
        share = 1.0 - tr / 100.0
        if share < cfg.min_compliant_share:
            continue
        total += int(max(0, tokens) * share)
    return total


def compute_gpu_infra_cost(
    active_gpu_hours_by_type: dict[str, float],
    config: InfrastructureCostConfig,
) -> float:
    """gpu_hours_used × per-type $/hr, summed.

    `active_gpu_hours_by_type` accumulates allocated GPU-hours per model; a GPU
    whose `assigned_workload_id is None` is treated as de-provisioned (zero
    cost), which is the simulator's idle-node consolidation semantics.
    """
    if not active_gpu_hours_by_type:
        return 0.0
    cost = 0.0
    for gpu_type, hours in active_gpu_hours_by_type.items():
        cost += max(0.0, hours) * config.gpu_price(gpu_type)
    return cost


def compute_energy_cost(energy_cost_per_tick: list[float]) -> float:
    """Pass-through sum of simulator-tracked energy cost.

    The simulator computes per-tick cost from `kwh × realtime_price` so any DA/RT
    accounting already present (see engine.py:1992) is preserved here.
    """
    return sum(max(0.0, c) for c in energy_cost_per_tick)


def compute_network_cost(
    migration_count: int,
    *,
    egress_gb: float = 0.0,
    config: Optional[InfrastructureCostConfig] = None,
) -> float:
    """Configured per-migration + per-GB egress only. Default 0.0.

    Per the spec, we do NOT invent network penalties inside this headline KPI:
    if the operator has not configured network costs, network_cost is 0.
    """
    cfg = config or InfrastructureCostConfig()
    return (
        max(0, migration_count) * cfg.network_cost_per_migration
        + max(0.0, egress_gb) * cfg.network_cost_per_gb_egress
    )


def compute_total_infrastructure_cost(
    gpu_infra_cost: float, energy_cost: float, network_cost: float
) -> float:
    return max(0.0, gpu_infra_cost) + max(0.0, energy_cost) + max(0.0, network_cost)


def compute_sla_safe_goodput_per_infra_dollar(
    sla_compliant_goodput: int, total_infrastructure_cost: float
) -> Optional[float]:
    """The canonical primary KPI. None when there is no cost basis.

    Zero goodput with positive cost is reported as 0.0 (you spent money and
    delivered no SLA-compliant work — a real, defined value). Zero cost with
    zero goodput returns None (undefined: no run happened).
    """
    if total_infrastructure_cost <= 0.0:
        return None
    return sla_compliant_goodput / total_infrastructure_cost


def compute_cost_per_sla_compliant_token(
    total_infrastructure_cost: float, sla_compliant_goodput: int
) -> Optional[float]:
    """Reciprocal companion of the primary KPI.

    Returns math.inf when cost > 0 but no SLA-compliant tokens were delivered
    (you spent money and produced nothing within SLA); returns None when there
    is no cost AND no goodput.
    """
    if sla_compliant_goodput <= 0:
        return math.inf if total_infrastructure_cost > 0.0 else None
    return total_infrastructure_cost / sla_compliant_goodput


# ---------------------------------------------------------------------------
# Convenience: assemble a full EconomicKPIResult from tick-level series.
# ---------------------------------------------------------------------------

def compute_economic_kpi(
    *,
    tokens_per_tick: list[int],
    timeout_rate_pct_per_tick: list[float],
    energy_cost_per_tick: list[float],
    active_gpu_hours_by_type_per_tick: list[dict[str, float]],
    migration_count: int,
    egress_gb: float = 0.0,
    config: Optional[InfrastructureCostConfig] = None,
    sla_filter: Optional[SLAFilterConfig] = None,
) -> EconomicKPIResult:
    """Compose the full canonical KPI from per-tick simulator series."""
    cfg = config or InfrastructureCostConfig()

    goodput = compute_sla_compliant_goodput(
        tokens_per_tick, timeout_rate_pct_per_tick, sla_filter=sla_filter
    )
    energy_cost = compute_energy_cost(energy_cost_per_tick)

    # Aggregate per-type GPU-hours across ticks.
    by_type: dict[str, float] = {}
    for tick_map in active_gpu_hours_by_type_per_tick:
        for k, v in (tick_map or {}).items():
            by_type[k] = by_type.get(k, 0.0) + v
    active_gpu_hours = sum(by_type.values())

    gpu_cost = compute_gpu_infra_cost(by_type, cfg)
    network_cost = compute_network_cost(
        migration_count, egress_gb=egress_gb, config=cfg
    )
    total_cost = compute_total_infrastructure_cost(gpu_cost, energy_cost, network_cost)

    return EconomicKPIResult(
        sla_compliant_goodput=goodput,
        raw_tokens=sum(max(0, t) for t in tokens_per_tick),
        gpu_infra_cost=gpu_cost,
        energy_cost=energy_cost,
        network_cost=network_cost,
        total_infrastructure_cost=total_cost,
        sla_safe_goodput_per_infra_dollar=compute_sla_safe_goodput_per_infra_dollar(
            goodput, total_cost
        ),
        cost_per_sla_compliant_token=compute_cost_per_sla_compliant_token(
            total_cost, goodput
        ),
        active_gpu_hours=active_gpu_hours,
        active_gpu_hours_by_type=by_type,
    )
