"""Explicit, mutable thermal / cooling / power state models.

First-class simulator states required by the thermal-realism upgrade. Mutable
(updated each tick by the engine), separate from the frozen ClusterState.
``GPUThermalState`` is attached per SimGPU; ``RackThermalState`` (with its
cooling-zone / airflow / hotspot / density sub-states) is attached per SimNode.
All values are bounded proxies, not a CFD/DCIM simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThermalInertiaState:
    """Lumped-capacitance thermal inertia bookkeeping for a GPU."""
    temp_c: float = 35.0
    last_power_w: float = 0.0


@dataclass
class ThermalThrottleState:
    """Continuous thermal slowdown (NOT a binary flag)."""
    slowdown_frac: float = 0.0
    throttle_events: int = 0


@dataclass
class PowerThrottleState:
    """Continuous power-cap slowdown."""
    slowdown_frac: float = 0.0
    power_cap_w: float = 0.0


@dataclass
class GPUThermalState:
    """Per-GPU thermal + power state."""
    inertia: ThermalInertiaState = field(default_factory=ThermalInertiaState)
    thermal_throttle: ThermalThrottleState = field(default_factory=ThermalThrottleState)
    power_throttle: PowerThrottleState = field(default_factory=PowerThrottleState)
    board_power_w: float = 0.0
    power_class: str = "a100"


@dataclass
class AirflowState:
    """Airflow quality for a rack/zone (1 = ideal, →0 = collapsed)."""
    quality: float = 1.0
    instability: float = 0.0


@dataclass
class HotspotState:
    """Persistent hotspot for a rack (lingers, recovers gradually)."""
    severity: float = 0.0          # [0,1]
    risk: float = 0.0              # instantaneous risk this tick
    persisted_ticks: int = 0


@dataclass
class RackDensityState:
    """Per-rack power density + regime."""
    rack_kw: float = 0.0
    regime: str = "normal"         # normal | elevated | critical


@dataclass
class CoolingRecoveryState:
    """Delayed-recovery bookkeeping (throttle risk stays elevated post-load)."""
    recovery_ticks_remaining: int = 0


@dataclass
class CoolingZoneState:
    """Cooling-zone (regime + utilization) bookkeeping for a rack."""
    regime: str = "air"
    zone_utilization: float = 0.0  # fraction of cooling capacity in use


@dataclass
class AmbientBoundaryState:
    """Slow-changing ambient/inlet boundary condition for a rack."""
    ambient_c: float = 22.0
    inlet_c: float = 22.0


@dataclass
class ThermalTelemetryConfidence:
    """Thermal telemetry quality for a rack/zone."""
    tier: str = "high"             # high | medium | low
    stale_ticks: int = 0


@dataclass
class ThermalViolationState:
    """Thermal excursions / cooling alarms bookkeeping."""
    excursions: int = 0
    cooling_alarms: int = 0
    last_tick_excursion: bool = False


@dataclass
class ThermalMigrationRiskState:
    """Thermal migration risk + veto bookkeeping for this rack as a destination."""
    risk: float = 0.0
    veto_count: int = 0


@dataclass
class RackThermalState:
    """Composite per-rack/node thermal state (all sub-states above)."""
    density: RackDensityState = field(default_factory=RackDensityState)
    airflow: AirflowState = field(default_factory=AirflowState)
    hotspot: HotspotState = field(default_factory=HotspotState)
    recovery: CoolingRecoveryState = field(default_factory=CoolingRecoveryState)
    zone: CoolingZoneState = field(default_factory=CoolingZoneState)
    ambient: AmbientBoundaryState = field(default_factory=AmbientBoundaryState)
    telemetry: ThermalTelemetryConfidence = field(default_factory=ThermalTelemetryConfidence)
    violation: ThermalViolationState = field(default_factory=ThermalViolationState)
    migration_risk: ThermalMigrationRiskState = field(
        default_factory=ThermalMigrationRiskState
    )
    cooling_regime: str = "air"
    peak_gpu_temp_c: float = 0.0
