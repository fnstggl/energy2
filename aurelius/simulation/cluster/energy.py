"""Energy / carbon / arbitrage realism for the cluster simulator.

Pure, deterministic functions (all randomness is caller-supplied via a
``random.Random`` → seedable) that replace clean price minimization with
constrained energy-aware optimization: separate day-ahead / real-time
settlement, a mean-reverting + heteroskedastic DA/RT basis with congestion
jumps, an LMP component decomposition (energy + congestion + loss), a
heavy-tailed RT spike process, regional carbon-intensity forecasts with
uncertainty + provider disagreement, conservative forecast-error buffers,
net-vs-gross savings accounting with explicit penalties, and diminishing
returns / churn from repeated shifting.

Every magnitude comes from ``calibration.ENERGY_PARAMS`` /
``ENERGY_FLEX_PROFILES`` (inspectable provenance + confidence) and is overridable
via a per-run ``config`` dict. These are proxies, NOT a market simulation:

- DA/RT settlement uses the standard two-settlement formula but the basis is a
  tunable OU+jump prior, NOT calibrated to a market;
- LMP = energy + congestion + loss is the documented decomposition with
  heuristic component magnitudes;
- the spike process and carbon-forecast errors are configurable priors.

Do NOT read any value here as production-accurate. The goal is that DA planning
can be wrong under RT settlement, nodal/regional arbitrage is not clean, gross
savings != net savings, tiny spreads do not justify action, forecast error and
churn bias toward no-op, and carbon-cheap is not price-cheap.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .calibration import energy_value, resolve_energy_flex

__all__ = [
    "settlement_bill",
    "basis_step",
    "lmp_total",
    "spike_increment",
    "real_time_price",
    "carbon_intensity",
    "carbon_forecast",
    "objective",
    "risk_adjusted_savings",
    "net_savings",
    "churn_penalty",
    "required_margin",
    "energy_action_allowed",
    "shift_window_hours",
    "deadline_pressure",
    "energy_telemetry_confidence",
]


# ---------------------------------------------------------------------------
# Day-ahead / real-time settlement
# ---------------------------------------------------------------------------

def settlement_bill(q_da: float, q_rt: float, p_da: float, p_rt: float) -> float:
    """Two-settlement bill = q_DA·p_DA + (q_RT − q_DA)·p_RT.

    Scheduled quantity settles at the day-ahead price; the deviation between
    realized and scheduled quantity settles at the real-time price. A planner
    that commits q_DA at p_DA but realizes q_RT under a divergent p_RT pays the
    basis on the deviation.
    """
    return q_da * p_da + (q_rt - q_da) * p_rt


# ---------------------------------------------------------------------------
# DA/RT basis (mean-reverting, heteroskedastic, congestion jumps)
# ---------------------------------------------------------------------------

def basis_step(
    basis: float,
    congested: bool,
    rng: random.Random,
    config: Optional[dict] = None,
) -> tuple[float, float]:
    """One step of the DA/RT basis OU process with congestion heteroskedasticity.

    b_{t+1} = b_t·(1−theta) + N(0, vol·vol_mult) + jump. Mean-reverting toward 0,
    with higher volatility and a jump probability during congestion. Returns
    (new_basis, jump_increment).
    """
    theta = energy_value("basis_mean_reversion", config)
    vol = energy_value("basis_vol", config)
    if congested:
        vol *= energy_value("basis_vol_congestion_mult", config)
    drift = basis * (1.0 - theta)
    innovation = rng.gauss(0.0, vol)
    jump = 0.0
    jump_prob = energy_value("basis_jump_prob", config)
    if congested:
        jump_prob = min(1.0, jump_prob * 3.0)
    if rng.random() < jump_prob:
        mag = energy_value("basis_jump_magnitude", config)
        jump = abs(rng.gauss(mag, mag * 0.4))  # right-skewed (RT above DA)
    return drift + innovation + jump, jump


# ---------------------------------------------------------------------------
# LMP component decomposition
# ---------------------------------------------------------------------------

def lmp_total(
    energy_component: float,
    congestion_active: bool,
    config: Optional[dict] = None,
) -> tuple[float, float, float]:
    """LMP = energy + congestion + loss. Returns (congestion, loss, total).

    Regional/nodal arbitrage relies on the congestion + loss components, which
    are NOT clean: a destination congestion event can erase the spread.
    """
    congestion = energy_value("lmp_congestion_base", config)
    if congestion_active:
        congestion += energy_value("lmp_congestion_event_adder", config)
    loss = max(0.0, energy_component) * energy_value("lmp_loss_frac", config)
    total = max(0.0, energy_component) + congestion + loss
    return congestion, loss, total


# ---------------------------------------------------------------------------
# RT spike process (heavy tails) + assembled real-time price
# ---------------------------------------------------------------------------

def spike_increment(
    stress: bool, rng: random.Random, config: Optional[dict] = None
) -> float:
    """Heavy-tailed RT price spike increment ($/MWh), 0 if no spike this tick.

    RT prices have heavier tails than DA. Spike probability rises in a grid-stress
    regime. Magnitudes/probabilities are configurable, NOT hardcoded universally.
    """
    prob = energy_value("spike_prob", config)
    if stress:
        prob = min(1.0, prob * energy_value("spike_stress_mult", config))
    if rng.random() >= prob:
        return 0.0
    mag = energy_value("spike_magnitude", config)
    return abs(rng.gauss(mag, mag * 0.6))


def real_time_price(
    day_ahead: float,
    basis: float,
    congestion: float,
    loss: float,
    spike: float,
) -> float:
    """Realized RT price = DA + basis + congestion + loss + spike (clamped ≥ 0).

    When basis/spike are disabled (both 0 and no congestion/loss) this equals the
    day-ahead price, preserving deterministic DA==RT pricing.
    """
    return max(0.0, day_ahead + basis + congestion + loss + spike)


# ---------------------------------------------------------------------------
# Carbon intensity + forecast uncertainty
# ---------------------------------------------------------------------------

def carbon_intensity(
    baseline: float, hour: int, config: Optional[dict] = None
) -> float:
    """Actual regional carbon intensity with a diurnal cycle (gCO2/kWh).

    Lower mid-day (solar) — but this does NOT imply price-cheap (see
    carbon_price_correlation).
    """
    amp = energy_value("carbon_diurnal_amplitude", config)
    # Trough near mid-day (hour 12), peak near evening.
    diurnal = -amp * math.cos(math.pi * (hour - 18) / 12.0)
    return max(0.0, baseline * (1.0 + diurnal))


def carbon_forecast(
    actual: float, rng: random.Random, config: Optional[dict] = None
) -> tuple[float, float, float]:
    """Forecast carbon intensity = actual·(1 + N(0, error_std)).

    Returns (forecast, error_std_frac, provider_disagreement_frac). Forecast
    carbon is NOT ground truth and providers disagree.
    """
    err = energy_value("carbon_forecast_error_std", config)
    disagreement = energy_value("carbon_provider_disagreement", config)
    forecast = max(0.0, actual * (1.0 + rng.gauss(0.0, err)))
    return forecast, err, disagreement


def objective(cost: float, carbon: float, alpha: float, beta: float) -> float:
    """Weighted objective = alpha·cost + beta·carbon.

    Carbon-minimizing windows are NOT necessarily price-minimizing windows;
    beta>0 trades price for carbon.
    """
    return alpha * cost + beta * carbon


# ---------------------------------------------------------------------------
# Net-vs-gross savings + forecast buffer + churn
# ---------------------------------------------------------------------------

def risk_adjusted_savings(
    expected_savings: float, forecast_error_std: float, config: Optional[dict] = None
) -> float:
    """risk_adjusted = expected − k·forecast_error_std (conservative buffer)."""
    k = energy_value("forecast_error_buffer_k", config)
    return expected_savings - k * max(0.0, forecast_error_std)


def churn_penalty(
    recent_shifts: int, tick_energy_cost: float, config: Optional[dict] = None
) -> float:
    """Super-linear churn penalty growing with recent shift count.

    penalty = base·tick_cost·(recent_shifts^growth). Repeated shifting does NOT
    keep producing linear savings.
    """
    if recent_shifts <= 0:
        return 0.0
    base = energy_value("churn_penalty_base", config)
    growth = energy_value("churn_penalty_growth", config)
    return base * max(0.0, tick_energy_cost) * (recent_shifts ** growth)


def net_savings(
    gross_energy_savings: float,
    *,
    gross_carbon_value: float = 0.0,
    migration_cost: float = 0.0,
    cache_cost: float = 0.0,
    cold_start_cost: float = 0.0,
    queue_penalty: float = 0.0,
    sla_penalty: float = 0.0,
    topology_penalty: float = 0.0,
    thermal_penalty: float = 0.0,
    forecast_error_cost: float = 0.0,
    churn_penalty: float = 0.0,
) -> float:
    """Net savings = gross (energy + carbon value) − all operational penalties.

    Energy savings must NEVER be reported without these penalties. If the result
    is ≤ 0 the action should be a KEEP / no-op.
    """
    return (
        gross_energy_savings
        + gross_carbon_value
        - migration_cost
        - cache_cost
        - cold_start_cost
        - queue_penalty
        - sla_penalty
        - topology_penalty
        - thermal_penalty
        - forecast_error_cost
        - churn_penalty
    )


def required_margin(
    tick_energy_cost: float, forecast_confidence: str, config: Optional[dict] = None
) -> float:
    """Required net-savings margin before an energy action is allowed.

    A fraction of the workload's tick energy cost, inflated when forecast
    confidence is missing/low (bias toward no-op). Tiny arbitrage must NOT act.
    """
    frac = energy_value("required_margin_frac", config)
    margin = frac * max(0.0, tick_energy_cost)
    if forecast_confidence in ("low", "medium"):
        mult = energy_value("missing_forecast_margin_mult", config)
        if forecast_confidence == "medium":
            mult = 1.0 + 0.5 * (mult - 1.0)
        margin *= mult
    return margin


def energy_action_allowed(
    net: float, risk_adjusted: float, margin: float
) -> bool:
    """Allow an energy action only if BOTH net and risk-adjusted savings clear
    the required margin. Otherwise recommend KEEP / no-op."""
    return net > margin and risk_adjusted > margin


# ---------------------------------------------------------------------------
# Workload flexibility / shift window
# ---------------------------------------------------------------------------

def shift_window_hours(flexibility: str | None, config: Optional[dict] = None) -> float:
    """Max temporal-shift window (hours) for a flexibility class. Never infinite."""
    return resolve_energy_flex(flexibility).max_shift_hours


def deadline_pressure(
    deferred_ticks: int, max_shift_hours: float, tick_hours: float
) -> float:
    """Deadline pressure [0,1] rising as deferred work approaches its window.

    Deferred work accumulates a deadline; once the shift window is exhausted the
    work can no longer be deferred (pressure 1.0).
    """
    if max_shift_hours <= 0:
        return 1.0
    deferred_hours = deferred_ticks * max(0.0, tick_hours)
    return max(0.0, min(1.0, deferred_hours / max_shift_hours))


# ---------------------------------------------------------------------------
# Telemetry confidence
# ---------------------------------------------------------------------------

def energy_telemetry_confidence(
    price_visible: bool, carbon_visible: bool, stale_ticks: int
) -> str:
    """Map price/carbon telemetry visibility/staleness to a confidence tier.

    Missing forecast confidence must reduce confidence, increase the required
    margin, and bias toward no-op — it must NOT be read as a safe opportunity.
    """
    if price_visible and carbon_visible and stale_ticks <= 1:
        return "high"
    if (price_visible or carbon_visible) and stale_ticks <= 3:
        return "medium"
    return "low"
