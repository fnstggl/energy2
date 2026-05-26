"""Explicit, mutable energy / carbon / arbitrage state models.

First-class simulator states required by the energy-realism upgrade. Mutable
(updated each tick by the engine), separate from the frozen ClusterState.
``RegionEnergyState`` is attached per SimRegion; ``WorkloadEnergyState`` per
SimWorkload (shift window, churn, net-savings accounting).

All values are bounded proxies, NOT a market simulation. The DA/RT basis, LMP
component, spike, and carbon-forecast parameters are tunable engineering
heuristics (see calibration.py), not calibrated to a specific ISO/market. Do NOT
read any value here as production-accurate. By default basis + spikes are OFF
(RT == DA), preserving deterministic pricing for non-energy scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Per-region energy sub-states
# ---------------------------------------------------------------------------

@dataclass
class DayAheadState:
    """Day-ahead (scheduled) settlement price for a region/node."""
    price_per_mwh: float = 50.0     # planning signal


@dataclass
class RealTimeState:
    """Real-time (realized) settlement price for a region/node."""
    price_per_mwh: float = 50.0     # what realized consumption actually pays
    heavy_tail_active: bool = False


@dataclass
class BasisState:
    """DA/RT basis = p_RT - p_DA (mean-reverting, heteroskedastic)."""
    basis: float = 0.0
    vol_regime: str = "normal"      # normal | congested
    last_jump: float = 0.0


@dataclass
class LMPComponentState:
    """LMP decomposition: energy + congestion + loss components ($/MWh)."""
    energy_component: float = 50.0
    congestion_component: float = 0.0
    loss_component: float = 0.0
    constrained_interface: bool = False

    @property
    def total(self) -> float:
        return self.energy_component + self.congestion_component + self.loss_component


@dataclass
class CongestionState:
    """Nodal/interface congestion bookkeeping for a region."""
    active: bool = False
    severity: float = 0.0           # [0,1]
    persisted_ticks: int = 0


@dataclass
class CarbonForecastState:
    """Regional carbon intensity: forecast vs actual, with uncertainty."""
    forecast_gco2_per_kwh: float = 400.0
    actual_gco2_per_kwh: float = 400.0
    error_std_frac: float = 0.15
    provider_disagreement_frac: float = 0.12
    confidence: str = "high"        # high | medium | low


@dataclass
class ForecastUncertaintyState:
    """Price/carbon forecast error + confidence for a region."""
    price_error_std: float = 0.0
    carbon_error_std_frac: float = 0.0
    confidence: str = "high"        # high | medium | low


@dataclass
class SpareCapacityState:
    """Usable spare capacity for shifted load before price/queue rises."""
    free_gpus: int = 0
    usable_for_shift: int = 0       # limited low-cost window
    saturated: bool = False


@dataclass
class EnergyTelemetryConfidence:
    """Energy/price/carbon telemetry quality for a region."""
    tier: str = "high"              # high | medium | low
    price_visible: bool = True
    carbon_visible: bool = True
    stale_ticks: int = 0


@dataclass
class RegionEnergyState:
    """Composite per-region energy/carbon market state (all sub-states)."""
    day_ahead: DayAheadState = field(default_factory=DayAheadState)
    real_time: RealTimeState = field(default_factory=RealTimeState)
    basis: BasisState = field(default_factory=BasisState)
    lmp: LMPComponentState = field(default_factory=LMPComponentState)
    congestion: CongestionState = field(default_factory=CongestionState)
    carbon: CarbonForecastState = field(default_factory=CarbonForecastState)
    forecast: ForecastUncertaintyState = field(default_factory=ForecastUncertaintyState)
    spare: SpareCapacityState = field(default_factory=SpareCapacityState)
    telemetry: EnergyTelemetryConfidence = field(
        default_factory=EnergyTelemetryConfidence
    )
    basis_enabled: bool = False
    spikes_enabled: bool = False


# ---------------------------------------------------------------------------
# Per-workload energy / arbitrage / net-savings states
# ---------------------------------------------------------------------------

@dataclass
class ShiftWindowState:
    """Temporal-shift flexibility of a workload (nothing is infinitely deferrable)."""
    flexibility: str = "medium"     # low | medium | high
    max_shift_hours: float = 2.0
    spatial_shift: bool = True
    requires_locality: bool = True
    deferred_ticks: int = 0
    deadline_pressure: float = 0.0  # [0,1] rises as deferred work accumulates


@dataclass
class ChurnState:
    """Repeated-shifting churn bookkeeping (diminishing returns)."""
    recent_shifts: int = 0
    last_shift_tick: int = -999
    churn_penalty: float = 0.0


@dataclass
class NetSavingsState:
    """Net-vs-gross savings accounting for the workload's last evaluated action."""
    gross_energy_savings: float = 0.0
    gross_carbon_value: float = 0.0
    migration_cost: float = 0.0
    cache_cost: float = 0.0
    cold_start_cost: float = 0.0
    queue_penalty: float = 0.0
    sla_penalty: float = 0.0
    topology_penalty: float = 0.0
    thermal_penalty: float = 0.0
    forecast_error_cost: float = 0.0
    churn_penalty: float = 0.0
    net_savings: float = 0.0
    risk_adjusted_savings: float = 0.0
    required_margin: float = 0.0
    action_allowed: bool = False
    last_reason: str = ""


@dataclass
class WorkloadEnergyState:
    """Composite per-workload energy/arbitrage state."""
    shift: ShiftWindowState = field(default_factory=ShiftWindowState)
    churn: ChurnState = field(default_factory=ChurnState)
    net: NetSavingsState = field(default_factory=NetSavingsState)
    # Objective weights: objective = alpha*cost + beta*carbon.
    alpha_cost: float = 1.0
    beta_carbon: float = 0.0
