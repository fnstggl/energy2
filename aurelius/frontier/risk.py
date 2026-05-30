"""Deterministic / statistical risk estimation for the Dynamic Safe
Frontier Estimator (v1).

This is **NOT trained ML**. The risk estimator turns a recent telemetry
window plus a candidate rho into two probability-like scores in [0, 1]:

- ``estimate_sla_risk`` — probability that the SLA-safe goodput rate
  collapses at the candidate rho (timeout rises past the configured
  threshold).
- ``estimate_queue_blowup_risk`` — probability that the queue tail
  (queue p99) blows past the configured threshold at the candidate rho.

Signals used (all deterministic, all observable in the window — no
future leakage):

- recent timeout share / trend
- queue p99 trend (mean + slope)
- proximity to safety thresholds
- delta between current rho and candidate rho
- burstiness (coefficient of variation of RPS)
- scale-event / churn density
- telemetry-confidence label

Defaults are pre-registered; nothing is tuned to force a win on a
particular trace.

Documented in ``docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md``. Directional
shadow-mode evidence only — NOT production savings (``docs/RESULTS.md``
§8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .dynamic_models import ServingTelemetryTick

# Confidence ordering (mirrors safety.py / controller.py).
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class RiskConfig:
    """Pre-registered risk-estimator settings.

    Every threshold is transparent and adjustable per call. ``ema_alpha``
    weighs recent ticks more heavily for the trend signals.
    """

    # Safety thresholds the dynamic estimator scores against (mirror the
    # static SafetyConfig defaults).
    max_timeout_pct: float = 10.0
    max_queue_p99_ms: float = 2000.0
    max_latency_p99_ms: Optional[float] = None
    # Window-shape settings.
    ema_alpha: float = 0.4
    # Recent-window length to compute trends (in ticks, not seconds —
    # the caller controls tick_seconds).
    trend_window_ticks: int = 6
    # When the candidate rho exceeds the observed current rho by more
    # than this, the SLA / queue risk scores get an additive bump.
    rho_jump_threshold: float = 0.10
    # Heuristic weights for the SLA-risk score components.
    sla_proximity_weight: float = 0.55
    sla_trend_weight: float = 0.25
    sla_rho_jump_weight: float = 0.20
    # Heuristic weights for the queue-blowup-risk score components.
    queue_proximity_weight: float = 0.45
    queue_trend_weight: float = 0.30
    queue_rho_jump_weight: float = 0.20
    queue_burstiness_weight: float = 0.05
    # Required minimum telemetry confidence to emit a non-fallback risk
    # estimate. Below this we return ``None`` to signal INSUFFICIENT.
    min_telemetry_confidence: str = "low"

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")
        for name in ("max_timeout_pct", "max_queue_p99_ms"):
            v = getattr(self, name)
            if v is None or v <= 0:
                raise ValueError(f"{name} must be > 0; got {v}")


# ---------------------------------------------------------------------------
# Internal helpers — deterministic, stdlib-only.
# ---------------------------------------------------------------------------

def _ema(values: Sequence[Optional[float]], alpha: float
         ) -> Optional[float]:
    """Exponential moving average over the non-None values. Returns
    ``None`` if no value is present."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    s = present[0]
    for v in present[1:]:
        s = alpha * v + (1.0 - alpha) * s
    return s


def _ema_slope(values: Sequence[Optional[float]], alpha: float
               ) -> Optional[float]:
    """Coarse trend signal — difference between the EMA of the second
    half of the present values and the first half. Positive ⇒ rising.
    Normalized by max(|first|, |second|) so the result is bounded."""
    present = [v for v in values if v is not None]
    if len(present) < 4:
        return None
    mid = len(present) // 2
    a = _ema(present[:mid], alpha)
    b = _ema(present[mid:], alpha)
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1e-9)
    return (b - a) / denom  # in [-1, 1] roughly


def _cv(values: Sequence[Optional[float]]) -> Optional[float]:
    """Coefficient of variation (std / mean) over the non-None values.
    Bursty workloads have high CV; smooth workloads have CV ≈ 0."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    mean = sum(present) / len(present)
    if mean <= 0:
        return None
    var = sum((v - mean) ** 2 for v in present) / len(present)
    return math.sqrt(var) / mean


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _telemetry_confidence_ok(window: Sequence[ServingTelemetryTick],
                              min_conf: str) -> bool:
    needed = _CONF_RANK.get(min_conf, 0)
    if not window:
        return False
    # Take the *minimum* confidence across the window (worst tick wins).
    return min(_CONF_RANK.get(t.telemetry_confidence or "unknown", 0)
               for t in window) >= needed


def _current_rho_from_window(window: Sequence[ServingTelemetryTick]
                              ) -> Optional[float]:
    """Best-effort estimate of the *observed* current rho from the
    window. Returns ``None`` if the window has no usable signal."""
    if not window:
        return None
    rhos = [t.mean_utilization for t in window
            if t.mean_utilization is not None]
    if rhos:
        return sum(rhos) / len(rhos)
    return None


# ---------------------------------------------------------------------------
# Public risk-estimator API.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskEstimate:
    """Probability-like score plus reason codes + confidence.

    ``probability`` is in [0, 1] when produced; ``None`` when
    telemetry is insufficient to score. ``reason_codes`` enumerates
    every component that contributed to a non-zero risk.
    """

    probability: Optional[float]
    reason_codes: tuple
    confidence: str

    def to_dict(self) -> dict:
        return {"probability": self.probability,
                "reason_codes": list(self.reason_codes),
                "confidence": self.confidence}


def estimate_sla_risk(
    *,
    candidate_rho: float,
    current_rho: Optional[float],
    window: Sequence[ServingTelemetryTick],
    config: RiskConfig,
) -> RiskEstimate:
    """Deterministic SLA-risk score in [0, 1].

    Components (clipped to [0,1] then weighted-summed):

    1. *proximity*: how close the recent (EMA) timeout share is to the
       configured ``max_timeout_pct`` threshold.
    2. *trend*:    how fast the timeout share is rising.
    3. *rho jump*: penalty when the candidate rho exceeds the observed
       current rho by more than ``rho_jump_threshold``.
    """
    if not _telemetry_confidence_ok(window, config.min_telemetry_confidence):
        return RiskEstimate(probability=None,
                            reason_codes=("low_telemetry_confidence",),
                            confidence="low")

    timeouts = [t.timeout_pct for t in window]
    timeout_ema = _ema(timeouts, config.ema_alpha)
    timeout_slope = _ema_slope(timeouts, config.ema_alpha)

    reasons: list[str] = []
    components = []

    # 1. Proximity component
    if timeout_ema is None:
        proximity = 0.0
        reasons.append("timeout_telemetry_missing")
    else:
        proximity = _clip(timeout_ema / config.max_timeout_pct)
        if proximity >= 0.8:
            reasons.append("timeout_near_threshold")
    components.append(proximity * config.sla_proximity_weight)

    # 2. Trend component (positive slope ⇒ risk rising)
    if timeout_slope is None:
        trend = 0.0
    else:
        # slope in [-1, 1]; positive slope adds risk, negative subtracts a bit
        trend = _clip((timeout_slope + 1.0) / 2.0)
        if timeout_slope > 0.3:
            reasons.append("timeout_trend_rising")
    components.append(trend * config.sla_trend_weight)

    # 3. Rho-jump component
    if current_rho is None or candidate_rho <= current_rho:
        jump = 0.0
    else:
        delta = candidate_rho - current_rho
        jump = _clip(delta / max(config.rho_jump_threshold, 1e-6))
        if delta > config.rho_jump_threshold:
            reasons.append("rho_jump_exceeds_threshold")
    components.append(jump * config.sla_rho_jump_weight)

    score = _clip(sum(components))

    # Confidence: blend of the window's worst confidence and the trend
    # observability.
    worst_conf = (min((t.telemetry_confidence or "unknown" for t in window),
                       key=lambda c: _CONF_RANK.get(c, 0))
                  if window else "unknown")
    if timeout_slope is None and len(window) < config.trend_window_ticks:
        worst_conf = (worst_conf if _CONF_RANK.get(worst_conf, 0) <= 1
                      else "low")
    return RiskEstimate(probability=score,
                        reason_codes=tuple(reasons),
                        confidence=worst_conf)


def estimate_queue_blowup_risk(
    *,
    candidate_rho: float,
    current_rho: Optional[float],
    window: Sequence[ServingTelemetryTick],
    config: RiskConfig,
) -> RiskEstimate:
    """Deterministic queue-blowup-risk score in [0, 1]."""
    if not _telemetry_confidence_ok(window, config.min_telemetry_confidence):
        return RiskEstimate(probability=None,
                            reason_codes=("low_telemetry_confidence",),
                            confidence="low")

    q99 = [t.queue_p99_ms for t in window]
    rps = [t.observed_rps for t in window]
    q99_ema = _ema(q99, config.ema_alpha)
    q99_slope = _ema_slope(q99, config.ema_alpha)
    burstiness = _cv(rps)

    reasons: list[str] = []
    components = []

    # 1. Proximity to max_queue_p99_ms
    if q99_ema is None:
        proximity = 0.0
        reasons.append("queue_p99_telemetry_missing")
    else:
        proximity = _clip(q99_ema / config.max_queue_p99_ms)
        if proximity >= 0.8:
            reasons.append("queue_p99_near_threshold")
    components.append(proximity * config.queue_proximity_weight)

    # 2. Trend component
    if q99_slope is None:
        trend = 0.0
    else:
        trend = _clip((q99_slope + 1.0) / 2.0)
        if q99_slope > 0.3:
            reasons.append("queue_p99_trend_rising")
    components.append(trend * config.queue_trend_weight)

    # 3. Rho-jump component
    if current_rho is None or candidate_rho <= current_rho:
        jump = 0.0
    else:
        delta = candidate_rho - current_rho
        jump = _clip(delta / max(config.rho_jump_threshold, 1e-6))
        if delta > config.rho_jump_threshold:
            reasons.append("rho_jump_exceeds_threshold")
    components.append(jump * config.queue_rho_jump_weight)

    # 4. Burstiness component (more variance = more queue-tail risk)
    if burstiness is None:
        burst = 0.0
    else:
        burst = _clip(burstiness / 2.0)  # CV >= 2 saturates
        if burstiness > 1.0:
            reasons.append("workload_burstiness_high")
    components.append(burst * config.queue_burstiness_weight)

    score = _clip(sum(components))
    worst_conf = (min((t.telemetry_confidence or "unknown" for t in window),
                       key=lambda c: _CONF_RANK.get(c, 0))
                  if window else "unknown")
    return RiskEstimate(probability=score,
                        reason_codes=tuple(reasons),
                        confidence=worst_conf)


def estimate_required_headroom(
    window: Sequence[ServingTelemetryTick],
    *,
    config: RiskConfig,
) -> Optional[float]:
    """Deterministic estimate of the rho headroom required to absorb
    recent burstiness without queue blowup. Returns ``None`` if the
    signal can't be computed.

    Heuristic: use the recent CV of RPS as a proxy. CV ≈ 0 (smooth) ⇒
    0.05 headroom; CV ≈ 1+ (bursty) ⇒ 0.25+ headroom. The score is
    capped at 0.35 (no workload should reserve more than that as
    headroom without explicit ops intervention).
    """
    rps = [t.observed_rps for t in window]
    cv = _cv(rps)
    if cv is None:
        return None
    # Smooth: 0.05 floor; bursty: up to 0.35 ceiling.
    return _clip(0.05 + 0.2 * cv, 0.0, 0.35)


def estimate_churn_risk(
    window: Sequence[ServingTelemetryTick],
    *,
    config: RiskConfig,
) -> RiskEstimate:
    """Heuristic churn score in [0, 1] used by the dynamic controller to
    *suppress* aggressive rho moves when the workload is unstable."""
    if not _telemetry_confidence_ok(window, config.min_telemetry_confidence):
        return RiskEstimate(probability=None,
                            reason_codes=("low_telemetry_confidence",),
                            confidence="low")
    scale_events = [t.scale_events_delta for t in window
                    if t.scale_events_delta is not None]
    churn = [t.churn_delta for t in window
             if t.churn_delta is not None]
    reasons: list[str] = []
    if not scale_events and not churn:
        return RiskEstimate(probability=None,
                            reason_codes=("churn_telemetry_missing",),
                            confidence="low")
    n = max(1, len(window))
    # Per-tick events; normalize by an arbitrary saturation point so the
    # score lives in [0,1]. The saturation point is opt-in via config in
    # later versions; v1 hard-codes a conservative 2 events/tick.
    se_rate = (sum(scale_events) / n) if scale_events else 0.0
    ch_rate = (sum(churn) / n) if churn else 0.0
    score = _clip((se_rate / 2.0) + (ch_rate / 4.0))
    if se_rate > 1.0:
        reasons.append("scale_events_high")
    if ch_rate > 2.0:
        reasons.append("churn_high")
    worst_conf = min((t.telemetry_confidence or "unknown" for t in window),
                      key=lambda c: _CONF_RANK.get(c, 0))
    return RiskEstimate(probability=score,
                        reason_codes=tuple(reasons),
                        confidence=worst_conf)
