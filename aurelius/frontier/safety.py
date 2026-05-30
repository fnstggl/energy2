"""Safety filter for frontier points.

Pre-registered, transparent thresholds — never folded into a KPI weight
(``docs/RESULTS.md`` §1-§2). A point that breaches *any* configured gate is
UNSAFE; a point that lacks the telemetry needed to evaluate a configured gate
is INSUFFICIENT_TELEMETRY (never auto-pass).

Defaults mirror the diagnostic safety ceilings in
``docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md``:

- ``max_timeout_pct = 10.0`` (timeout share)
- ``max_queue_p99_ms = 2000.0`` (queue p99)

Thermal / topology / memory / scale-churn / telemetry-confidence gates are
opt-in (``None`` disables the gate). Latency p99 is enforced only if the
estimator reports it; an unreported latency does NOT auto-pass when an SLA
budget is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import FrontierPoint, SafetyStatus

# Confidence ordering (mirrors ``aurelius/residency/decision.py``).
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class SafetyConfig:
    """Pre-registered safety ceilings + telemetry minimums.

    Every threshold is explicit; ``None`` disables the gate. Defaults mirror
    the Azure 2024 audit thresholds.
    """

    max_timeout_pct: Optional[float] = 10.0
    max_queue_p99_ms: Optional[float] = 2000.0
    max_queue_p95_ms: Optional[float] = None
    max_latency_p99_ms: Optional[float] = None
    max_latency_p95_ms: Optional[float] = None
    min_telemetry_confidence: str = "low"
    max_thermal_risk: Optional[float] = None
    min_topology_score: Optional[float] = None
    max_memory_pressure: Optional[float] = None
    max_scale_events: Optional[int] = None
    max_churn_score: Optional[float] = None

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}; "
                f"expected one of {sorted(_CONF_RANK)}")


def _missing(value) -> bool:
    return value is None


def _vetoes_for_point(point: FrontierPoint, cfg: SafetyConfig,
                      telemetry_confidence: str) -> tuple[list, list]:
    """Return (hard_unsafe_vetoes, missing_telemetry_vetoes)."""
    hard: list[str] = []
    missing: list[str] = []

    # --- timeout gate (required when configured) ---
    if cfg.max_timeout_pct is not None:
        if _missing(point.predicted_timeout_pct):
            missing.append("timeout_telemetry_missing")
        elif point.predicted_timeout_pct > cfg.max_timeout_pct:
            hard.append("timeout_exceeds_threshold")

    # --- queue p99 gate ---
    if cfg.max_queue_p99_ms is not None:
        if _missing(point.predicted_queue_p99_ms):
            missing.append("queue_p99_telemetry_missing")
        elif point.predicted_queue_p99_ms > cfg.max_queue_p99_ms:
            hard.append("queue_p99_exceeds_threshold")

    # --- queue p95 gate (opt-in) ---
    if cfg.max_queue_p95_ms is not None:
        if _missing(point.predicted_queue_p95_ms):
            missing.append("queue_p95_telemetry_missing")
        elif point.predicted_queue_p95_ms > cfg.max_queue_p95_ms:
            hard.append("queue_p95_exceeds_threshold")

    # --- latency p99 gate (opt-in; unreported -> missing, not auto-pass) ---
    if cfg.max_latency_p99_ms is not None:
        if _missing(point.predicted_latency_p99_ms):
            missing.append("latency_p99_telemetry_missing")
        elif point.predicted_latency_p99_ms > cfg.max_latency_p99_ms:
            hard.append("latency_p99_exceeds_threshold")

    if cfg.max_latency_p95_ms is not None:
        if _missing(point.predicted_latency_p95_ms):
            missing.append("latency_p95_telemetry_missing")
        elif point.predicted_latency_p95_ms > cfg.max_latency_p95_ms:
            hard.append("latency_p95_exceeds_threshold")

    # --- telemetry-confidence gate (always evaluated when configured) ---
    needed = _CONF_RANK.get(cfg.min_telemetry_confidence, 0)
    have = _CONF_RANK.get(telemetry_confidence or "unknown", 0)
    if have < needed:
        missing.append("low_telemetry_confidence")

    # --- thermal / topology / memory / churn / scale (opt-in) ---
    # The estimator does not always report these. ``None`` for an opt-in gate
    # means the gate is disabled, so a missing prediction stays as missing.
    if cfg.max_thermal_risk is not None:
        thermal = _point_attr(point, "predicted_thermal_risk")
        if _missing(thermal):
            missing.append("thermal_telemetry_missing")
        elif thermal > cfg.max_thermal_risk:
            hard.append("thermal_risk_exceeds_threshold")

    if cfg.min_topology_score is not None:
        topo = _point_attr(point, "predicted_topology_score")
        if _missing(topo):
            missing.append("topology_telemetry_missing")
        elif topo < cfg.min_topology_score:
            hard.append("topology_score_below_threshold")

    if cfg.max_memory_pressure is not None:
        mem = _point_attr(point, "predicted_memory_pressure")
        if _missing(mem):
            missing.append("memory_telemetry_missing")
        elif mem > cfg.max_memory_pressure:
            hard.append("memory_pressure_exceeds_threshold")

    if cfg.max_scale_events is not None:
        if _missing(point.predicted_scale_events):
            missing.append("scale_events_telemetry_missing")
        elif point.predicted_scale_events > cfg.max_scale_events:
            hard.append("scale_events_exceeds_threshold")

    if cfg.max_churn_score is not None:
        if _missing(point.predicted_churn_score):
            missing.append("churn_telemetry_missing")
        elif point.predicted_churn_score > cfg.max_churn_score:
            hard.append("churn_exceeds_threshold")

    return hard, missing


def _point_attr(point: FrontierPoint, name: str):
    """Best-effort optional attribute fetch (the estimator may not provide
    every field; we treat absent attributes as missing telemetry)."""
    return getattr(point, name, None)


def is_frontier_point_safe(point: FrontierPoint, cfg: SafetyConfig, *,
                           telemetry_confidence: Optional[str] = None) -> bool:
    """Return True iff ``point`` passes every configured safety gate.

    A point with ``safety_status == INSUFFICIENT_TELEMETRY`` (e.g. missing
    timeout or queue telemetry) is NOT safe; the controller treats it as
    INSUFFICIENT_TELEMETRY rather than UNSAFE.
    """
    hard, missing = _vetoes_for_point(point, cfg, telemetry_confidence or "unknown")
    return not hard and not missing


def classify_point_safety(point: FrontierPoint, cfg: SafetyConfig, *,
                          telemetry_confidence: Optional[str] = None
                          ) -> tuple[str, tuple]:
    """Return the (``safety_status``, ``safety_vetoes``) pair for ``point``.

    - SAFE if all gates pass.
    - UNSAFE if any hard gate breaches.
    - INSUFFICIENT_TELEMETRY if any required input is missing but no hard
      gate breaches (UNSAFE wins over INSUFFICIENT_TELEMETRY when both apply
      — a known breach is a known breach).
    """
    hard, missing = _vetoes_for_point(point, cfg, telemetry_confidence or "unknown")
    if hard:
        return SafetyStatus.UNSAFE, tuple(hard + missing)
    if missing:
        return SafetyStatus.INSUFFICIENT_TELEMETRY, tuple(missing)
    return SafetyStatus.SAFE, ()
