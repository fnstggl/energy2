"""Telemetry-window builders for the Dynamic Safe Frontier Estimator.

This module accepts three kinds of inputs and normalizes them into a
sequence of :class:`ServingTelemetryTick`:

1. **Azure 2024 aggregated arrival ticks** (the ``ArrivalTick`` dataclass
   produced by ``aurelius/traces/replay.py``) — used by the dynamic
   benchmark.
2. **Existing serving / backtest tick summaries** (per-tick dicts with
   ``goodput_per_dollar`` / ``queue_p99_ms`` / etc.) — used when the
   caller already has aggregated metrics.
3. **Fixture telemetry dicts** — used by tests and pilot bring-up.

Missing fields stay ``None`` (the contract from
``docs/PILOT_TELEMETRY_CONTRACT.md`` §1). The validator returns a
structured result that names every missing required field; it never
silently zero-fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .dynamic_models import ServingTelemetryTick

# Minimum fields required to estimate the dynamic frontier. ``rps`` +
# at least one queue percentile + an active-replica count is the
# pre-registered minimum.
DEFAULT_REQUIRED_FIELDS = ("observed_rps", "queue_p99_ms", "active_replicas")


@dataclass(frozen=True)
class TelemetryWindowValidation:
    """Result of validating a telemetry window.

    ``ok`` is True iff the window has at least ``min_ticks`` ticks AND
    every required field is present on at least ``min_field_coverage``
    fraction of ticks.
    """

    ok: bool
    reason: str
    n_ticks: int
    min_ticks: int
    required_fields: tuple
    missing_field_coverage: dict  # field -> ticks missing it (count)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "n_ticks": self.n_ticks,
                "min_ticks": self.min_ticks,
                "required_fields": list(self.required_fields),
                "missing_field_coverage": dict(self.missing_field_coverage)}


def _coerce_optional_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN/inf
        return None
    return out


def _coerce_optional_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def telemetry_tick_from_arrival_tick(
    arrival_tick,
    *,
    active_replicas: Optional[int] = None,
    queue_p50_ms: Optional[float] = None,
    queue_p95_ms: Optional[float] = None,
    queue_p99_ms: Optional[float] = None,
    latency_p99_ms: Optional[float] = None,
    timeout_pct: Optional[float] = None,
    sla_violation_pct: Optional[float] = None,
    mean_utilization: Optional[float] = None,
    scale_events_delta: Optional[int] = None,
    churn_delta: Optional[float] = None,
    gpu_hours_delta: Optional[float] = None,
    telemetry_confidence: str = "medium",
    source: str = "arrival_tick",
) -> ServingTelemetryTick:
    """Convert one ``aurelius/traces/replay.ArrivalTick`` to a serving
    telemetry tick. Caller may supply optional observed signals (queue
    percentiles, timeouts, etc.) — anything not supplied stays ``None``.
    """
    duration = max(1e-9, getattr(arrival_tick, "duration_s", 60.0))
    tokens_total = (getattr(arrival_tick, "total_prompt_tokens", 0)
                    + getattr(arrival_tick, "total_output_tokens", 0))
    prompt_tokens_per_s = (getattr(arrival_tick, "total_prompt_tokens", 0)
                           / duration) if duration else None
    output_tokens_per_s = (getattr(arrival_tick, "total_output_tokens", 0)
                           / duration) if duration else None
    # If gpu_hours_delta wasn't supplied but active_replicas is, infer
    # gpu_hours_delta = replicas * (duration / 3600).
    if gpu_hours_delta is None and active_replicas is not None:
        gpu_hours_delta = float(active_replicas) * (duration / 3600.0)
    return ServingTelemetryTick(
        timestamp_s=float(getattr(arrival_tick, "start_s", 0.0)),
        observed_rps=_coerce_optional_float(
            getattr(arrival_tick, "arrival_rate_rps", None)),
        prompt_tokens_per_s=prompt_tokens_per_s,
        output_tokens_per_s=output_tokens_per_s,
        total_tokens_per_s=(tokens_total / duration) if duration else None,
        active_replicas=active_replicas,
        gpu_hours_delta=gpu_hours_delta,
        mean_utilization=mean_utilization,
        queue_p50_ms=queue_p50_ms,
        queue_p95_ms=queue_p95_ms,
        queue_p99_ms=queue_p99_ms,
        latency_p99_ms=latency_p99_ms,
        timeout_pct=timeout_pct,
        sla_violation_pct=sla_violation_pct,
        scale_events_delta=scale_events_delta,
        churn_delta=churn_delta,
        telemetry_confidence=telemetry_confidence,
        source=source,
    )


def build_serving_telemetry_window(
    ticks: Iterable,
    *,
    source: str = "unknown",
    telemetry_confidence: str = "medium",
) -> list[ServingTelemetryTick]:
    """Normalize ``ticks`` (ArrivalTick, dict, or ServingTelemetryTick) into
    a list of :class:`ServingTelemetryTick`. Order is preserved.
    """
    out: list[ServingTelemetryTick] = []
    for t in ticks:
        if isinstance(t, ServingTelemetryTick):
            out.append(t)
            continue
        if isinstance(t, dict):
            d = dict(t)
            d.setdefault("telemetry_confidence", telemetry_confidence)
            d.setdefault("source", source)
            # Drop any keys not in the dataclass to stay strict.
            allowed = set(ServingTelemetryTick.__dataclass_fields__)
            payload = {k: d.get(k) for k in allowed}
            payload["timestamp_s"] = float(payload.get("timestamp_s") or 0.0)
            for k in ("observed_rps", "prompt_tokens_per_s",
                      "output_tokens_per_s", "total_tokens_per_s",
                      "gpu_hours_delta", "mean_utilization",
                      "queue_p50_ms", "queue_p95_ms", "queue_p99_ms",
                      "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                      "timeout_pct", "sla_violation_pct", "churn_delta"):
                payload[k] = _coerce_optional_float(payload.get(k))
            for k in ("active_replicas", "scale_events_delta"):
                payload[k] = _coerce_optional_int(payload.get(k))
            payload.setdefault("telemetry_confidence", telemetry_confidence)
            payload.setdefault("source", source)
            out.append(ServingTelemetryTick(**payload))
            continue
        # Assume an aurelius/traces/replay.ArrivalTick-like object
        out.append(telemetry_tick_from_arrival_tick(
            t, telemetry_confidence=telemetry_confidence, source=source))
    return out


def validate_dynamic_window(
    window: Sequence[ServingTelemetryTick],
    *,
    min_ticks: int = 8,
    required_fields: Sequence[str] = DEFAULT_REQUIRED_FIELDS,
    min_field_coverage: float = 0.5,
) -> TelemetryWindowValidation:
    """Validate that the window has enough ticks and field coverage."""
    n = len(window) if window is not None else 0
    if n < min_ticks:
        return TelemetryWindowValidation(
            ok=False,
            reason=f"n_ticks={n} below minimum {min_ticks}",
            n_ticks=n, min_ticks=min_ticks,
            required_fields=tuple(required_fields),
            missing_field_coverage={})
    missing_cov: dict = {}
    for f in required_fields:
        missing = sum(1 for t in window
                      if getattr(t, f, None) is None)
        if missing:
            missing_cov[f] = missing
        if (n - missing) / max(1, n) < min_field_coverage:
            return TelemetryWindowValidation(
                ok=False,
                reason=(f"field {f!r} present on only "
                        f"{n - missing}/{n} ticks "
                        f"({(n - missing) / n:.2%} < required "
                        f"{min_field_coverage:.0%})"),
                n_ticks=n, min_ticks=min_ticks,
                required_fields=tuple(required_fields),
                missing_field_coverage=missing_cov)
    return TelemetryWindowValidation(
        ok=True, reason="ok", n_ticks=n, min_ticks=min_ticks,
        required_fields=tuple(required_fields),
        missing_field_coverage=missing_cov)
