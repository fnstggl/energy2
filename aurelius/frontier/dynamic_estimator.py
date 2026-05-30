"""Dynamic Safe Frontier Estimator — telemetry-driven frontier estimation.

Given a recent telemetry window of :class:`ServingTelemetryTick` and the
current rho operating point, produce a frontier estimate over a set of
candidate rhos. Each candidate carries a predicted goodput/$,
GPU-hours, timeout %, queue p99, plus deterministic SLA-risk and
queue-blowup-risk probabilities (in [0, 1]).

Hard rules (asserted by tests):

- **No future leakage.** The estimator may only read the window passed
  in by the caller. The streaming-replay benchmark wires it into a
  rolling window so the t-th decision only sees t' ≤ t telemetry.
- **No invented data.** When a required signal is missing, the
  candidate's safety_status is INSUFFICIENT_TELEMETRY and the
  ``predicted_*`` field stays ``None``.
- **No ML in v1.** Predictions come from the unchanged serving physics
  (Erlang-C tail multipliers via
  :mod:`aurelius.simulation.cluster.serving`) plus the deterministic
  risk estimator in :mod:`aurelius.frontier.risk`.
- **Output is recommendation-only.** Estimates are consumed by the
  dynamic controller (:mod:`aurelius.frontier.dynamic_controller`),
  which emits a recommendation-only
  :class:`DynamicFrontierDecision`. Real execution requires the static
  ``execute_frontier_decision`` opt-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .dynamic_models import (
    DynamicFrontierCandidate,
    DynamicFrontierEstimate,
    ServingTelemetryTick,
)
from .dynamic_telemetry import validate_dynamic_window
from .models import (
    DEFAULT_CANDIDATE_RHOS,
    SafetyStatus,
    WorkloadFrontierProfile,
)
from .risk import (
    RiskConfig,
    _current_rho_from_window,
    estimate_queue_blowup_risk,
    estimate_required_headroom,
    estimate_sla_risk,
)
from .safety import SafetyConfig

# Confidence ordering shared with the static controller.
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class DynamicEstimatorConfig:
    """Settings for :func:`estimate_dynamic_frontier`. All values are
    pre-registered defaults — none are tuned per trace."""

    # Local-window length used when scanning rho candidates around the
    # observed current rho.
    local_delta: float = 0.10
    # Minimum candidate-grid points around current rho (in addition to
    # the global grid).
    n_local_candidates: int = 2
    # Global candidate grid; mirrors the static frontier audit defaults.
    global_candidate_rhos: tuple = DEFAULT_CANDIDATE_RHOS
    # When the SLA-risk OR queue-blowup-risk score exceeds this, the
    # candidate is marked UNSAFE even if hard thresholds haven't tripped
    # yet. Pre-registered at 0.75; never tuned per trace.
    unsafe_risk_threshold: float = 0.75
    # Minimum window length (in ticks) to attempt estimation.
    min_window_ticks: int = 8
    # Required telemetry-confidence label.
    min_telemetry_confidence: str = "low"
    # Conservative margin — step back from the safety boundary even
    # when the candidate passes every gate.
    conservative_margin_enabled: bool = True
    # When True, the estimator emits its prediction even if a candidate
    # has missing telemetry fields (treated as INSUFFICIENT_TELEMETRY).
    emit_insufficient_candidates: bool = True

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")
        if not (0.0 < self.local_delta < 1.0):
            raise ValueError(
                f"local_delta must be in (0,1); got {self.local_delta}")
        if not (0.0 <= self.unsafe_risk_threshold <= 1.0):
            raise ValueError(
                f"unsafe_risk_threshold must be in [0,1]; got "
                f"{self.unsafe_risk_threshold}")


# ---------------------------------------------------------------------------
# Prediction helpers (deterministic, stdlib-only).
# ---------------------------------------------------------------------------

def _ema_optional(values, alpha: float = 0.4) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    s = present[0]
    for v in present[1:]:
        s = alpha * v + (1.0 - alpha) * s
    return s


def _mean_optional(values) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _erlang_c_tail(rho: float) -> float:
    """Closed-form tail multiplier for a single-server M/M/1-style
    queue (the same family the engine's ``serving.saturation_amplifier``
    uses, applied here at the rho granularity). Saturates near rho → 1.
    """
    rho = max(1e-6, min(0.999, rho))
    return 1.0 / (1.0 - rho)


def _predict_for_rho(
    *,
    rho: float,
    current_rho: Optional[float],
    window: Sequence[ServingTelemetryTick],
) -> dict:
    """Deterministic per-rho prediction using the recent window.

    Method (M/M/1-style Erlang-C calibration from the observed window):

    1. The recent window was operated at the *observed* mean rho
       (``observed_rho``). We use the observed (queue p99 EMA, observed
       rho) pair to **calibrate** a constant
       ``C_queue = observed_queue_p99 * (1 - observed_rho)``; predicted
       queue p99 at any candidate rho ``R`` is then
       ``C_queue / (1 - R)`` — i.e., a workload-specific Erlang-C tail.
    2. The same calibration is applied to the observed timeout share.
       Above the safe-timeout threshold the prediction grows quickly;
       below it it shrinks toward zero. (We deliberately do NOT clip
       this at the observed value — see also the safety filter, which
       vetoes any candidate that exceeds the threshold.)
    3. GPU-hours scale with replicas, which scale ~ 1 / rho when the
       offered load is held constant (the autoscaler resizes to hold
       the target rho).
    4. Goodput/$ proxies as (1 / predicted_gpu_hours) — the directional
       sign matches the engine's KPI (more rho ⇒ fewer replicas ⇒
       higher KPI) when the SLA still holds.

    The static frontier audit (``aurelius/frontier/estimator.py``) uses
    the same closed-form Erlang-C family.
    """
    cur = current_rho if current_rho is not None else 0.65
    cur = max(0.05, min(0.99, cur))

    # Recent observed signals (EMAs).
    timeout_ema = _ema_optional([t.timeout_pct for t in window])
    queue_ema = _ema_optional([t.queue_p99_ms for t in window])
    latency_ema = _ema_optional([t.latency_p99_ms for t in window])
    replicas_ema = _ema_optional([t.active_replicas for t in window])
    observed_rho_ema = _ema_optional([t.mean_utilization for t in window])
    gpu_h_total = sum((t.gpu_hours_delta or 0.0) for t in window) or None

    # Calibration rho — prefer the observed mean rho from the window,
    # fall back to the passed-in current rho. Clamp to (0.05, 0.99) so
    # the tail constant stays finite.
    calib_rho = observed_rho_ema if observed_rho_ema is not None else cur
    calib_rho = max(0.05, min(0.99, calib_rho))

    # Erlang-C tail at the candidate rho, calibrated to the observed
    # window. ``C_queue`` and ``C_timeout`` are workload-specific
    # constants extracted from the recent operating point.
    if queue_ema is not None:
        C_queue = queue_ema * (1.0 - calib_rho)
        predicted_queue_p99 = C_queue / max(1e-6, (1.0 - rho))
    else:
        predicted_queue_p99 = None

    if timeout_ema is not None:
        C_timeout = timeout_ema * (1.0 - calib_rho)
        predicted_timeout = C_timeout / max(1e-6, (1.0 - rho))
    else:
        predicted_timeout = None

    if latency_ema is not None:
        # Latency scales with the queue-tail multiplier (service time
        # changes slowly and is treated as constant in this v1 model).
        C_lat = latency_ema * (1.0 - calib_rho)
        predicted_latency = C_lat / max(1e-6, (1.0 - rho))
    else:
        predicted_latency = None

    # Replica & GPU-hour predictions: the autoscaler resizes to hold the
    # target rho, so replicas scale ~ (calib_rho / rho).
    if replicas_ema is not None:
        predicted_replicas = max(1.0, replicas_ema * (calib_rho / rho))
    else:
        predicted_replicas = None

    if gpu_h_total is not None:
        predicted_gpu_hours = gpu_h_total * (calib_rho / rho)
    else:
        predicted_gpu_hours = None

    # Goodput/$ — inverse GPU-hour proxy (same direction as engine KPI).
    if predicted_gpu_hours and predicted_gpu_hours > 0:
        predicted_goodput = (1.0 / predicted_gpu_hours)
    else:
        predicted_goodput = None

    return {
        "predicted_replicas": predicted_replicas,
        "predicted_goodput_per_dollar": predicted_goodput,
        "predicted_gpu_hours": predicted_gpu_hours,
        "predicted_queue_p99_ms": predicted_queue_p99,
        "predicted_timeout_pct": predicted_timeout,
        "predicted_latency_p99_ms": predicted_latency,
    }


def _safety_status(
    *,
    prediction: dict,
    sla_risk: Optional[float],
    queue_risk: Optional[float],
    safety: SafetyConfig,
    unsafe_risk_threshold: float,
) -> tuple[str, tuple]:
    """Categorize a candidate as SAFE / UNSAFE / INSUFFICIENT_TELEMETRY."""
    hard: list[str] = []
    missing: list[str] = []

    if safety.max_timeout_pct is not None:
        v = prediction["predicted_timeout_pct"]
        if v is None:
            missing.append("timeout_telemetry_missing")
        elif v > safety.max_timeout_pct:
            hard.append("timeout_exceeds_threshold")

    if safety.max_queue_p99_ms is not None:
        v = prediction["predicted_queue_p99_ms"]
        if v is None:
            missing.append("queue_p99_telemetry_missing")
        elif v > safety.max_queue_p99_ms:
            hard.append("queue_p99_exceeds_threshold")

    if safety.max_latency_p99_ms is not None:
        v = prediction["predicted_latency_p99_ms"]
        if v is None:
            missing.append("latency_p99_telemetry_missing")
        elif v > safety.max_latency_p99_ms:
            hard.append("latency_p99_exceeds_threshold")

    # Risk gate: a known SLA risk above the threshold is UNSAFE even if
    # the hard threshold hasn't tripped yet. This is the "lean back from
    # the boundary" rule.
    for name, score in (("predicted_sla_risk", sla_risk),
                        ("predicted_queue_blowup_risk", queue_risk)):
        if score is None:
            missing.append(f"{name}_telemetry_missing")
        elif score > unsafe_risk_threshold:
            hard.append(f"{name}_score_exceeds_threshold")

    if hard:
        return SafetyStatus.UNSAFE, tuple(hard + missing)
    if missing:
        return SafetyStatus.INSUFFICIENT_TELEMETRY, tuple(missing)
    return SafetyStatus.SAFE, ()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def estimate_dynamic_frontier(
    *,
    workload_profile: WorkloadFrontierProfile,
    telemetry_window: Sequence[ServingTelemetryTick],
    current_rho: Optional[float],
    candidate_rhos: Optional[Iterable[float]] = None,
    config: Optional[DynamicEstimatorConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
) -> DynamicFrontierEstimate:
    """Estimate the safe-utilization frontier from recent telemetry."""
    cfg = config or DynamicEstimatorConfig()
    safety = safety_config or SafetyConfig()
    risk_cfg = risk_config or RiskConfig(
        max_timeout_pct=safety.max_timeout_pct or 10.0,
        max_queue_p99_ms=safety.max_queue_p99_ms or 2000.0,
        max_latency_p99_ms=safety.max_latency_p99_ms,
        min_telemetry_confidence=cfg.min_telemetry_confidence)

    # 1 — window validation
    val = validate_dynamic_window(telemetry_window,
                                  min_ticks=cfg.min_window_ticks)
    if not val.ok:
        return DynamicFrontierEstimate(
            workload_id=workload_profile.workload_id,
            window_start_s=(telemetry_window[0].timestamp_s
                            if telemetry_window else 0.0),
            window_end_s=(telemetry_window[-1].timestamp_s
                          if telemetry_window else 0.0),
            current_rho_estimate=None, estimated_safe_rho=None,
            recommended_rho=None, confidence="low",
            frontier_slope=None, risk_at_current_rho=None,
            risk_at_recommended_rho=None, required_headroom=None,
            candidate_points=(), prediction_method="dynamic_v1",
            fallback_reason=val.reason,
            notes=(f"window_validation_failed:{val.reason}",))

    # 2 — current rho estimate
    obs_rho = _current_rho_from_window(telemetry_window)
    cur_rho = current_rho if current_rho is not None else obs_rho

    # 3 — candidate-rho construction (local + global, clamped)
    candidate_set: set = set()
    if candidate_rhos is not None:
        candidate_set.update(round(float(r), 4) for r in candidate_rhos)
    else:
        candidate_set.update(round(float(r), 4)
                             for r in cfg.global_candidate_rhos)
    if cur_rho is not None:
        step = cfg.local_delta / max(1, cfg.n_local_candidates)
        for i in range(-cfg.n_local_candidates, cfg.n_local_candidates + 1):
            candidate_set.add(round(cur_rho + i * step, 4))
    # Clamp to profile band; drop anything outside (min_rho, max_rho].
    candidate_list = sorted(r for r in candidate_set
                            if workload_profile.min_rho <= r
                            <= workload_profile.max_rho)
    if not candidate_list:
        return DynamicFrontierEstimate(
            workload_id=workload_profile.workload_id,
            window_start_s=telemetry_window[0].timestamp_s,
            window_end_s=telemetry_window[-1].timestamp_s,
            current_rho_estimate=cur_rho, estimated_safe_rho=None,
            recommended_rho=None, confidence="low",
            frontier_slope=None, risk_at_current_rho=None,
            risk_at_recommended_rho=None, required_headroom=None,
            candidate_points=(), prediction_method="dynamic_v1",
            fallback_reason="no_candidates_in_profile_band",
            notes=())

    # 4 — per-candidate prediction + risk scoring
    candidates: list[DynamicFrontierCandidate] = []
    for rho in candidate_list:
        pred = _predict_for_rho(rho=rho, current_rho=cur_rho,
                                window=telemetry_window)
        sla_risk = estimate_sla_risk(candidate_rho=rho, current_rho=cur_rho,
                                     window=telemetry_window,
                                     config=risk_cfg)
        queue_risk = estimate_queue_blowup_risk(
            candidate_rho=rho, current_rho=cur_rho,
            window=telemetry_window, config=risk_cfg)
        status, vetoes = _safety_status(
            prediction=pred, sla_risk=sla_risk.probability,
            queue_risk=queue_risk.probability, safety=safety,
            unsafe_risk_threshold=cfg.unsafe_risk_threshold)

        # Combined risk-reason codes (deduped, ordered).
        reason_codes = tuple(sorted(set(sla_risk.reason_codes +
                                         queue_risk.reason_codes)))
        # Confidence: the worst of the two risk estimates.
        confidence = (sla_risk.confidence
                      if _CONF_RANK[sla_risk.confidence]
                      <= _CONF_RANK[queue_risk.confidence]
                      else queue_risk.confidence)

        candidates.append(DynamicFrontierCandidate(
            rho_target=rho,
            predicted_goodput_per_dollar=pred["predicted_goodput_per_dollar"],
            predicted_gpu_hours=pred["predicted_gpu_hours"],
            predicted_timeout_pct=pred["predicted_timeout_pct"],
            predicted_queue_p99_ms=pred["predicted_queue_p99_ms"],
            predicted_latency_p99_ms=pred["predicted_latency_p99_ms"],
            predicted_churn_score=None,
            predicted_sla_risk_probability=sla_risk.probability,
            predicted_queue_blowup_probability=queue_risk.probability,
            safety_status=status, safety_vetoes=tuple(vetoes),
            confidence=confidence, risk_reason_codes=reason_codes))

    # 5 — frontier slope (marginal goodput/$ at the current rho)
    slope = _frontier_slope(candidates, cur_rho)

    # 6 — choose best safe candidate
    safe = [c for c in candidates if c.safety_status == SafetyStatus.SAFE]
    if safe:
        # Best goodput/$ among safe candidates. Handles None KPI by
        # treating None as -inf so safe-but-unmeasured points lose.
        def _kpi(c):
            return (c.predicted_goodput_per_dollar
                    if c.predicted_goodput_per_dollar is not None
                    else float("-inf"))
        best = max(safe, key=_kpi)

        # 7 — conservative margin: prefer next-lower safe if best is
        # adjacent to UNSAFE.
        if cfg.conservative_margin_enabled:
            above = sorted([c for c in candidates
                            if c.rho_target > best.rho_target],
                           key=lambda c: c.rho_target)
            for c in above:
                if c.safety_status == SafetyStatus.UNSAFE:
                    lower = sorted([s for s in safe
                                    if s.rho_target < best.rho_target],
                                   key=lambda c: c.rho_target,
                                   reverse=True)
                    if lower:
                        best = lower[0]
                    break
        recommended = best.rho_target
        estimated_safe = max(s.rho_target for s in safe)
        risk_at_rec = max(best.predicted_sla_risk_probability or 0.0,
                          best.predicted_queue_blowup_probability or 0.0)
    else:
        # No safe candidate — recommend the lowest tested rho.
        recommended = min(c.rho_target for c in candidates)
        estimated_safe = None
        risk_at_rec = None

    # 8 — risk at current rho
    risk_at_cur = None
    cur_cand = next((c for c in candidates if cur_rho is not None
                     and abs(c.rho_target - cur_rho) < 0.05), None)
    if cur_cand is not None:
        if (cur_cand.predicted_sla_risk_probability is not None
                or cur_cand.predicted_queue_blowup_probability is not None):
            risk_at_cur = max(cur_cand.predicted_sla_risk_probability or 0.0,
                              cur_cand.predicted_queue_blowup_probability
                              or 0.0)

    headroom = estimate_required_headroom(telemetry_window, config=risk_cfg)

    # Overall confidence: worst tick confidence on the window.
    if telemetry_window:
        overall_conf = min((t.telemetry_confidence or "unknown"
                            for t in telemetry_window),
                           key=lambda c: _CONF_RANK.get(c, 0))
    else:
        overall_conf = "unknown"

    return DynamicFrontierEstimate(
        workload_id=workload_profile.workload_id,
        window_start_s=telemetry_window[0].timestamp_s,
        window_end_s=telemetry_window[-1].timestamp_s,
        current_rho_estimate=cur_rho, estimated_safe_rho=estimated_safe,
        recommended_rho=recommended, confidence=overall_conf,
        frontier_slope=slope, risk_at_current_rho=risk_at_cur,
        risk_at_recommended_rho=risk_at_rec, required_headroom=headroom,
        candidate_points=tuple(candidates),
        prediction_method="dynamic_v1_erlang_c_tail",
        notes=tuple(val.reason for _ in [val] if not val.ok))


def _frontier_slope(candidates, cur_rho: Optional[float]) -> Optional[float]:
    """Marginal Δgoodput/$ per +0.01 rho near ``cur_rho``. Returns
    ``None`` if not enough adjacent safe points exist."""
    if cur_rho is None or len(candidates) < 2:
        return None
    safe_pts = [(c.rho_target, c.predicted_goodput_per_dollar)
                for c in candidates
                if c.predicted_goodput_per_dollar is not None]
    if len(safe_pts) < 2:
        return None
    # Two points straddling cur_rho.
    above = [p for p in safe_pts if p[0] > cur_rho]
    below = [p for p in safe_pts if p[0] < cur_rho]
    if not above or not below:
        # Use the closest two points instead.
        s = sorted(safe_pts, key=lambda p: abs(p[0] - cur_rho))[:2]
        if len(s) < 2:
            return None
        a, b = s
    else:
        a = min(above, key=lambda p: p[0])
        b = max(below, key=lambda p: p[0])
    dr = a[0] - b[0]
    if dr == 0:
        return None
    return (a[1] - b[1]) / (dr * 100.0)  # per +0.01 rho
