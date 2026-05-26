"""Thermal / cooling / power realism for the cluster simulator.

Pure, deterministic functions (all randomness is caller-supplied via a
``random.Random`` → seedable) that replace the simulator's simplistic thermal
heuristics with a believable lumped-capacitance thermal model: saturating board
power, thermal inertia, rack-level heat accumulation, hotspot formation/
persistence, cooling regimes (air/liquid/hybrid/…), and CONTINUOUS thermal +
power slowdown (not a binary throttle flag).

Every magnitude comes from ``calibration.THERMAL_PARAMS`` /
``GPU_POWER_CLASSES`` / ``COOLING_REGIMES`` (inspectable provenance +
confidence) and is overridable via a per-run ``config`` dict. These are proxies,
NOT a CFD / DCIM simulation:

- the temperature ODE is a 1st-order lumped model, not a spatial thermal field;
- rack-density thresholds (~20 / ~30 kW) are operational heuristics, NOT
  universal limits;
- cooling-regime multipliers are engineering priors, not measured coefficients.

Do NOT read any value here as production-accurate. The goal is that dense
placements can fail, hotspots persist, throttling materially matters, and
cooling regimes matter.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .calibration import (
    CoolingRegime,
    GPUPowerClass,
    power_class_for_model,
    resolve_cooling_regime,
    thermal_value,
)

__all__ = [
    "RackDensityRegime",
    "board_power_watts",
    "workload_power_multiplier",
    "temperature_step",
    "rack_heat_kw",
    "rack_density_regime",
    "hotspot_risk",
    "hotspot_step",
    "inlet_temperature",
    "thermal_slowdown_frac",
    "power_slowdown_frac",
    "throughput_factor",
    "thermal_telemetry_confidence",
    "thermal_migration_blocked",
    "resolve_cooling_regime",
    "power_class_for_model",
]


class RackDensityRegime:
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Board power (saturating curve)
# ---------------------------------------------------------------------------

def workload_power_multiplier(workload_kind: str, config: Optional[dict] = None) -> float:
    """Relative board-power draw by workload type at equal utilization."""
    key = {
        "inference": "power_mult_inference",
        "embedding": "power_mult_inference",
        "batch_training": "power_mult_training",
        "fine_tuning": "power_mult_training",
        "training": "power_mult_training",
        "memory_bound": "power_mult_memory_bound",
    }.get(workload_kind, "power_mult_inference")
    return thermal_value(key, config)


def board_power_watts(
    util_frac: float,
    power_class: GPUPowerClass,
    workload_kind: str = "inference",
    config: Optional[dict] = None,
) -> float:
    """Saturating board power: P = P_idle + (P_max−P_idle)·(1−exp(−k·u))·w_mult.

    Power rises rapidly with utilization then saturates near the board limit;
    the workload multiplier shifts the curve (training is power-denser). Utilization
    alone does NOT linearly predict power.
    """
    u = max(0.0, min(1.0, util_frac))
    k = thermal_value("power_curve_k", config)
    idle_frac = thermal_value("power_idle_frac", config)
    p_idle = idle_frac * power_class.p_max_w
    w = workload_power_multiplier(workload_kind, config)
    p = p_idle + (power_class.p_max_w - p_idle) * (1.0 - math.exp(-k * u)) * w
    return max(0.0, min(power_class.p_max_w * 1.05, p))  # tiny headroom for w>1


# ---------------------------------------------------------------------------
# Temperature evolution (thermal inertia)
# ---------------------------------------------------------------------------

def temperature_step(
    temp_c: float,
    power_w: float,
    ambient_c: float,
    power_class: GPUPowerClass,
    regime: CoolingRegime,
    rng: random.Random,
    config: Optional[dict] = None,
) -> float:
    """One step of the lumped-capacitance thermal ODE.

    T_{t+1} = T_t + a·P − b·(T_t − T_ambient) + ε_t

    a = thermal_alpha (per-class), b = thermal_beta_air × regime.beta_mult,
    ε ~ N(0, thermal_noise_c). Recovery is NOT instantaneous: small b means
    temperatures (and hotspots) decay slowly after load drops.
    """
    a = power_class.alpha if power_class.alpha > 0 else thermal_value("thermal_alpha", config)
    b = thermal_value("thermal_beta_air", config) * regime.beta_mult
    noise = rng.gauss(0.0, thermal_value("thermal_noise_c", config))
    # a is °C per watt per tick (calibrated so full-power settles ~50°C above
    # inlet under air cooling); b sets the recovery time constant.
    heating = a * power_w
    cooling = b * (temp_c - ambient_c)
    t_next = temp_c + heating - cooling + noise
    return max(ambient_c - 2.0, min(power_class.max_temp_c + 5.0, t_next))


# ---------------------------------------------------------------------------
# Rack-level heat + hotspots
# ---------------------------------------------------------------------------

def rack_heat_kw(
    job_powers_w: list[float],
    density_regime: str,
    config: Optional[dict] = None,
) -> float:
    """Rack heat load in kW = Σ job power + airflow + recirculation penalties.

    The penalties grow in elevated/critical density regimes (poor air management
    recirculates hot exhaust).
    """
    base_kw = sum(max(0.0, p) for p in job_powers_w) / 1000.0
    airflow_pen = thermal_value("airflow_penalty_c", config) / 1000.0  # ~kW-equivalent
    recirc_pen = thermal_value("hotspot_recirc_penalty_c", config) / 1000.0
    if density_regime == RackDensityRegime.CRITICAL:
        base_kw += airflow_pen + recirc_pen
    elif density_regime == RackDensityRegime.ELEVATED:
        base_kw += 0.5 * airflow_pen
    return base_kw


def rack_density_regime(
    rack_kw: float, regime: CoolingRegime, config: Optional[dict] = None
) -> str:
    """Classify a rack's density into normal / elevated / critical.

    Thresholds scale with the cooling regime's density tolerance (liquid cooling
    tolerates much higher kW). Operational heuristics, NOT universal limits.
    """
    elevated = thermal_value("rack_density_elevated_kw", config) * regime.density_mult
    critical = thermal_value("rack_density_critical_kw", config) * regime.density_mult
    if rack_kw >= critical:
        return RackDensityRegime.CRITICAL
    if rack_kw >= elevated:
        return RackDensityRegime.ELEVATED
    return RackDensityRegime.NORMAL


def hotspot_risk(
    rack_kw: float,
    airflow_quality: float,
    sustained_power_frac: float,
    regime: CoolingRegime,
    config: Optional[dict] = None,
) -> float:
    """Instantaneous hotspot risk in [0, 1].

    Rises with rack density (past the elevated threshold), poor airflow, and
    sustained high power; scaled by the regime's hotspot multiplier. Liquid
    cooling lowers it but never to zero.
    """
    elevated = thermal_value("rack_density_elevated_kw", config) * regime.density_mult
    critical = thermal_value("rack_density_critical_kw", config) * regime.density_mult
    if rack_kw <= elevated:
        density_term = 0.0
    else:
        density_term = min(1.0, (rack_kw - elevated) / max(1e-6, critical - elevated))
    airflow_term = 1.0 - max(0.0, min(1.0, airflow_quality))
    sustained_term = max(0.0, min(1.0, sustained_power_frac))
    raw = (0.5 * density_term + 0.3 * airflow_term + 0.2 * sustained_term)
    return max(0.0, min(1.0, raw * regime.hotspot_mult))


def hotspot_step(
    current_hotspot: float, instantaneous_risk: float, config: Optional[dict] = None
) -> float:
    """Persistent hotspot EMA: hotspots linger and recover gradually.

    h_{t+1} = max(persistence·h_t, risk) — an existing hotspot decays slowly
    (recovery lag) but a fresh high risk can raise it immediately.
    """
    persistence = thermal_value("hotspot_persistence", config)
    decayed = persistence * max(0.0, min(1.0, current_hotspot))
    return max(0.0, min(1.0, max(decayed, instantaneous_risk)))


def inlet_temperature(
    ambient_c: float,
    hotspot: float,
    regime: CoolingRegime,
    rng: random.Random,
    config: Optional[dict] = None,
) -> float:
    """Local inlet temperature = ambient + recirculation·hotspot + local variance.

    Recirculation adds hot-exhaust °C scaled by hotspot severity; local variance
    is regime-dependent (liquid/contained racks vary less).
    """
    recirc = thermal_value("hotspot_recirc_penalty_c", config) * max(0.0, min(1.0, hotspot))
    var = thermal_value("inlet_variance_c", config) * regime.inlet_variance_mult
    return ambient_c + recirc + rng.gauss(0.0, var)


# ---------------------------------------------------------------------------
# Continuous thermal + power slowdown
# ---------------------------------------------------------------------------

def thermal_slowdown_frac(
    temp_c: float, power_class: GPUPowerClass, config: Optional[dict] = None
) -> float:
    """Continuous thermal slowdown fraction in [0, thermal_slowdown_max].

    Zero below the throttle onset; ramps linearly to the max as temperature goes
    from onset → max_temp. NOT a binary flag.
    """
    onset = power_class.throttle_onset_c
    top = power_class.max_temp_c
    if temp_c <= onset or top <= onset:
        return 0.0
    frac = (temp_c - onset) / (top - onset)
    smax = thermal_value("thermal_slowdown_max", config)
    return max(0.0, min(smax, smax * frac))


def power_slowdown_frac(
    power_w: float, power_cap_w: float, config: Optional[dict] = None
) -> float:
    """Continuous power-cap slowdown fraction in [0, power_slowdown_max].

    Zero below ~90% of the cap; ramps to the max as power pins at the cap.
    """
    if power_cap_w <= 0:
        return 0.0
    load = power_w / power_cap_w
    if load < 0.9:
        return 0.0
    smax = thermal_value("power_slowdown_max", config)
    frac = min(1.0, (load - 0.9) / 0.1)
    return max(0.0, min(smax, smax * frac))


def throughput_factor(s_thermal: float, s_power: float) -> float:
    """Combined throughput multiplier = 1 − s_thermal − s_power, clamped to [0.05,1]."""
    return max(0.05, min(1.0, 1.0 - max(0.0, s_thermal) - max(0.0, s_power)))


# ---------------------------------------------------------------------------
# Telemetry confidence + thermal migration governor
# ---------------------------------------------------------------------------

def thermal_telemetry_confidence(
    has_temp: bool, has_power: bool, stale_ticks: int
) -> str:
    """Map thermal telemetry availability/staleness to a confidence tier.

    HIGH   fresh temp + power. MEDIUM  one missing or mildly stale. LOW  both
    missing or very stale. Missing telemetry LOWERS confidence — it must NOT be
    read as 'cool / safe'.
    """
    if has_temp and has_power and stale_ticks <= 1:
        return "high"
    if (has_temp or has_power) and stale_ticks <= 3:
        return "medium"
    return "low"


def thermal_migration_blocked(
    dest_temp_c: float,
    dest_hotspot: float,
    telemetry_tier: str,
    config: Optional[dict] = None,
) -> bool:
    """Veto migration INTO a hot zone (thermal governor).

    Blocks when the destination is above the hot-veto temperature OR has a strong
    persistent hotspot. With LOW telemetry confidence the effective threshold is
    lowered (missing telemetry ≠ safe → be conservative).
    """
    veto_c = thermal_value("thermal_migration_hot_veto_c", config)
    if telemetry_tier == "low":
        veto_c -= thermal_value("thermal_telemetry_missing_risk", config) * 10.0
    if dest_temp_c >= veto_c:
        return True
    return dest_hotspot >= 0.7
