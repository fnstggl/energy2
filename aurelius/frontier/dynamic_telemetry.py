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
from .telemetry_provenance import (
    FieldOrigin,
    FieldProvenance,
    TickProvenance,
    TimeoutFallback,
    derive_sla_violation_pct,
    make_derived,
    make_missing,
    make_proxy,
    make_real,
    resolve_timeout_pct,
)

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


def telemetry_tick_with_provenance_from_inference_service_state(
    state,
    *,
    timestamp_s: Optional[float] = None,
    tick_duration_s: Optional[float] = None,
    mean_utilization: Optional[float] = None,
    mean_utilization_source: str = "dcgm.DCGM_FI_DEV_GPU_UTIL",
    gpu_hours_delta: Optional[float] = None,
    scale_events_delta: Optional[int] = None,
    churn_delta: Optional[float] = None,
    scale_delta_source: str = "k8s.K8sReplicaDelta",
    latency_sla_p99_ms: Optional[float] = None,
    explicit_timeout_counter_rate: Optional[float] = None,
    deadline_exceeded_rate: Optional[float] = None,
    total_request_rate: Optional[float] = None,
    telemetry_confidence: Optional[str] = None,
    source: str = "inference_service_state",
) -> tuple[ServingTelemetryTick, TickProvenance]:
    """Bridge: build a :class:`ServingTelemetryTick` PLUS its
    :class:`TickProvenance` from a real
    :class:`aurelius.state.models.InferenceServiceState` observation.

    Hard rules:

    - Every populated field carries a :class:`FieldProvenance` entry so
      the dynamic-frontier shadow log and calibration record can tell
      REAL from DERIVED from PROXY.
    - Missing connector fields stay ``None`` AND emit a ``MISSING``
      provenance entry — never silent zero-fill.
    - ``timeout_pct`` runs through the
      :func:`resolve_timeout_pct` fallback chain (A → B → C → D); the
      level used is recorded on the field's provenance entry.
    - ``sla_violation_pct`` is DERIVED when ``latency_sla_p99_ms`` is
      provided AND ``p99_latency_ms`` is observed — never invented.
    - The engine of the underlying :class:`InferenceServiceState`
      decides which queue / latency mappings are REAL vs PROXY: vLLM's
      ``ttft_p99_ms`` is a PROXY for queue p99; Ray Serve's e2e p99 is
      a PROXY for queue p99; queue percentiles only count as REAL when
      the connector populated ``queue_time_p*_ms``.
    """
    entries: list[FieldProvenance] = []

    if timestamp_s is None:
        ts = getattr(state, "timestamp", None)
        timestamp_s = (ts.timestamp() if ts is not None else 0.0)

    engine = (getattr(state, "engine", None) or "unknown").lower()

    # ----- queue percentiles -----
    queue_p99 = getattr(state, "queue_time_p99_ms", None)
    queue_p99_origin = FieldOrigin.MISSING
    queue_p99_src = ""
    queue_p99_notes = ""
    if queue_p99 is not None:
        queue_p99_origin = FieldOrigin.REAL
        queue_p99_src = f"{engine}.queue_time_p99_ms"
        queue_p99_notes = "engine-reported queue wait p99"
    else:
        # Fall back to TTFT (vLLM) / e2e p99 — clearly PROXY.
        ttft_p99 = getattr(state, "ttft_p99_ms", None)
        e2e_p99 = getattr(state, "p99_latency_ms", None)
        if engine == "vllm" and ttft_p99 is not None:
            queue_p99 = ttft_p99
            queue_p99_origin = FieldOrigin.PROXY
            queue_p99_src = "vllm.ttft_p99_ms"
            queue_p99_notes = ("TTFT is the time to first token; used as "
                                "queue p99 proxy because vLLM does not "
                                "expose a pure queue-wait histogram")
        elif e2e_p99 is not None:
            queue_p99 = e2e_p99
            queue_p99_origin = FieldOrigin.PROXY
            queue_p99_src = f"{engine}.p99_latency_ms"
            queue_p99_notes = ("e2e latency p99 used as queue p99 proxy "
                                "(engine does not expose pure queue "
                                "histogram)")

    queue_p95 = getattr(state, "queue_time_p95_ms", None)
    queue_p95_origin = FieldOrigin.MISSING
    queue_p95_src = ""
    queue_p95_notes = ""
    if queue_p95 is not None:
        queue_p95_origin = FieldOrigin.REAL
        queue_p95_src = f"{engine}.queue_time_p95_ms"
    else:
        e2e_p95 = getattr(state, "p95_latency_ms", None)
        if e2e_p95 is not None:
            queue_p95 = e2e_p95
            queue_p95_origin = FieldOrigin.PROXY
            queue_p95_src = f"{engine}.p95_latency_ms"
            queue_p95_notes = "e2e p95 used as queue p95 proxy"

    queue_p50 = getattr(state, "queue_time_p50_ms", None)
    queue_p50_origin = FieldOrigin.MISSING
    queue_p50_src = ""
    queue_p50_notes = ""
    if queue_p50 is not None:
        # Triton specifically: queue_time_p50_ms is a derived AVERAGE
        # (queue_duration_us / exec_count), not a true p50.
        if engine == "triton":
            queue_p50_origin = FieldOrigin.DERIVED
            queue_p50_src = ("triton.nv_inference_queue_duration_us / "
                              "nv_inference_exec_count")
            queue_p50_notes = ("derived average — Triton default metrics "
                                "do not expose a queue histogram")
        else:
            queue_p50_origin = FieldOrigin.REAL
            queue_p50_src = f"{engine}.queue_time_p50_ms"

    # ----- replicas + RPS -----
    active_replicas = getattr(state, "replicas", None)
    replicas_origin = (FieldOrigin.MISSING if active_replicas is None
                        else FieldOrigin.REAL)
    replicas_src = (f"{engine}.replicas" if active_replicas is not None
                     else "")
    replicas_notes = ""

    rps_running = getattr(state, "requests_running", None)
    tokens_per_s = getattr(state, "tokens_per_s", None)
    # Choose the most-honest observed_rps source available.
    rps_value: Optional[float] = None
    rps_origin = FieldOrigin.MISSING
    rps_src = ""
    rps_notes = ""
    if rps_running is not None:
        rps_value = float(rps_running)
        rps_origin = FieldOrigin.PROXY
        rps_src = f"{engine}.requests_running"
        rps_notes = ("in-flight request count used as RPS proxy; the "
                      "REAL RPS source is rate(*request_success_total"
                      "[1m]) which the bridge does not see from a "
                      "single InferenceServiceState snapshot")

    # ----- latency p50 / p95 / p99 -----
    latency_p50 = getattr(state, "p50_latency_ms", None)
    latency_p95 = getattr(state, "p95_latency_ms", None)
    latency_p99 = getattr(state, "p99_latency_ms", None)

    def _lat_origin(v, engine_):
        if v is None:
            return FieldOrigin.MISSING
        # Triton p50 is documented as the cumulative AVERAGE — not a
        # true percentile — when histograms are disabled.
        if engine_ == "triton":
            return FieldOrigin.DERIVED
        return FieldOrigin.REAL

    latency_p50_origin = _lat_origin(latency_p50, engine)
    latency_p95_origin = _lat_origin(latency_p95, engine)
    latency_p99_origin = _lat_origin(latency_p99, engine)

    # ----- timeout_pct fallback chain -----
    err_rate = getattr(state, "error_rate_pct", None)
    timeout_res = resolve_timeout_pct(
        explicit_timeout_counter_rate=explicit_timeout_counter_rate,
        deadline_exceeded_rate=deadline_exceeded_rate,
        total_request_rate=total_request_rate,
        error_rate_pct=err_rate,
        latency_p99_ms=latency_p99,
        latency_sla_p99_ms=latency_sla_p99_ms,
    )
    timeout_value = timeout_res.value
    if timeout_value is None:
        timeout_origin = FieldOrigin.MISSING
        timeout_notes = "no timeout signal available"
    elif timeout_res.level in (TimeoutFallback.A_TIMEOUT_COUNTER,
                                TimeoutFallback.B_DEADLINE_COUNTER):
        timeout_origin = FieldOrigin.REAL
        timeout_notes = timeout_res.notes
    elif timeout_res.level == TimeoutFallback.C_ERROR_RATE:
        timeout_origin = FieldOrigin.PROXY
        timeout_notes = timeout_res.notes
    else:  # D_SLA_RISK_PROXY
        timeout_origin = FieldOrigin.PROXY
        timeout_notes = timeout_res.notes

    # ----- sla_violation_pct (derived) -----
    sla_violation_value = derive_sla_violation_pct(
        latency_p99_ms=latency_p99,
        latency_sla_p99_ms=latency_sla_p99_ms,
    )
    sla_violation_origin = (FieldOrigin.MISSING
                             if sla_violation_value is None
                             else FieldOrigin.DERIVED)
    sla_violation_src = (
        "derived: latency_p99_ms > latency_sla_p99_ms (per-tick binary)"
        if sla_violation_value is not None else "")

    # ----- gpu_hours_delta (derived) -----
    if (gpu_hours_delta is None and active_replicas is not None
            and tick_duration_s is not None and tick_duration_s > 0):
        gpu_hours_delta = float(active_replicas) * (
            float(tick_duration_s) / 3600.0)
        gpu_hours_origin = FieldOrigin.DERIVED
        gpu_hours_src = "active_replicas * tick_duration_s / 3600"
    elif gpu_hours_delta is not None:
        gpu_hours_origin = FieldOrigin.REAL
        gpu_hours_src = "caller-provided"
    else:
        gpu_hours_origin = FieldOrigin.MISSING
        gpu_hours_src = ""

    # ----- scale_events / churn (derived externally) -----
    if scale_events_delta is None:
        scale_events_origin = FieldOrigin.MISSING
        scale_events_src = ""
    else:
        scale_events_origin = FieldOrigin.DERIVED
        scale_events_src = scale_delta_source

    if churn_delta is None:
        churn_origin = FieldOrigin.MISSING
        churn_src = ""
    else:
        churn_origin = FieldOrigin.DERIVED
        churn_src = scale_delta_source

    # ----- mean_utilization -----
    if mean_utilization is None:
        mean_util_origin = FieldOrigin.MISSING
        mean_util_src_used = ""
    else:
        mean_util_origin = FieldOrigin.REAL
        mean_util_src_used = mean_utilization_source

    # ----- telemetry_confidence -----
    if telemetry_confidence is None:
        prov = getattr(state, "provenance", None)
        if prov is not None and getattr(prov, "confidence", None) in (
                "low", "medium", "high", "unknown"):
            telemetry_confidence = prov.confidence
        else:
            telemetry_confidence = "medium"

    # ----- assemble provenance entries -----
    def _add(field_name: str, origin: FieldOrigin, src: str = "",
             notes: str = "",
             fallback_level: Optional[TimeoutFallback] = None):
        entries.append(FieldProvenance(
            field=field_name, origin=origin, source=src,
            confidence=telemetry_confidence, notes=notes,
            fallback_level=fallback_level))

    _add("observed_rps", rps_origin, rps_src, rps_notes)
    _add("active_replicas", replicas_origin, replicas_src, replicas_notes)
    _add("queue_p99_ms", queue_p99_origin, queue_p99_src, queue_p99_notes)
    _add("queue_p95_ms", queue_p95_origin, queue_p95_src, queue_p95_notes)
    _add("queue_p50_ms", queue_p50_origin, queue_p50_src, queue_p50_notes)
    _add("latency_p99_ms", latency_p99_origin,
         (f"{engine}.p99_latency_ms" if latency_p99 is not None else ""),
         "" if latency_p99_origin != FieldOrigin.DERIVED
         else "Triton default p50 is cumulative average; p99 unavailable")
    _add("latency_p95_ms", latency_p95_origin,
         (f"{engine}.p95_latency_ms" if latency_p95 is not None else ""))
    _add("latency_p50_ms", latency_p50_origin,
         (f"{engine}.p50_latency_ms" if latency_p50 is not None else ""),
         "" if latency_p50_origin != FieldOrigin.DERIVED
         else "Triton derived average")
    _add("timeout_pct", timeout_origin,
         timeout_res.source, timeout_notes,
         fallback_level=timeout_res.level)
    _add("sla_violation_pct", sla_violation_origin,
         sla_violation_src,
         "binary per-tick; aggregate over a window for a meaningful share"
         if sla_violation_value is not None else "")
    _add("mean_utilization", mean_util_origin, mean_util_src_used,
         "DCGM gpu.util_pct / 100" if mean_util_origin == FieldOrigin.REAL
         else "")
    _add("gpu_hours_delta", gpu_hours_origin, gpu_hours_src)
    _add("scale_events_delta", scale_events_origin, scale_events_src)
    _add("churn_delta", churn_origin, churn_src)

    tick = ServingTelemetryTick(
        timestamp_s=float(timestamp_s),
        observed_rps=_coerce_optional_float(rps_value),
        prompt_tokens_per_s=None,
        output_tokens_per_s=None,
        total_tokens_per_s=_coerce_optional_float(tokens_per_s),
        active_replicas=_coerce_optional_int(active_replicas),
        gpu_hours_delta=_coerce_optional_float(gpu_hours_delta),
        mean_utilization=_coerce_optional_float(mean_utilization),
        queue_p50_ms=_coerce_optional_float(queue_p50),
        queue_p95_ms=_coerce_optional_float(queue_p95),
        queue_p99_ms=_coerce_optional_float(queue_p99),
        latency_p50_ms=_coerce_optional_float(latency_p50),
        latency_p95_ms=_coerce_optional_float(latency_p95),
        latency_p99_ms=_coerce_optional_float(latency_p99),
        timeout_pct=_coerce_optional_float(timeout_value),
        sla_violation_pct=_coerce_optional_float(sla_violation_value),
        scale_events_delta=_coerce_optional_int(scale_events_delta),
        churn_delta=_coerce_optional_float(churn_delta),
        telemetry_confidence=telemetry_confidence,
        source=source,
    )
    return tick, TickProvenance(entries=tuple(entries))


def telemetry_tick_from_inference_service_state(
    state,
    *,
    timestamp_s: Optional[float] = None,
    tick_duration_s: Optional[float] = None,
    mean_utilization: Optional[float] = None,
    gpu_hours_delta: Optional[float] = None,
    scale_events_delta: Optional[int] = None,
    churn_delta: Optional[float] = None,
    telemetry_confidence: Optional[str] = None,
    source: str = "inference_service_state",
    latency_sla_p99_ms: Optional[float] = None,
    explicit_timeout_counter_rate: Optional[float] = None,
    deadline_exceeded_rate: Optional[float] = None,
    total_request_rate: Optional[float] = None,
) -> ServingTelemetryTick:
    """Backwards-compatible wrapper around
    :func:`telemetry_tick_with_provenance_from_inference_service_state`.

    Returns ONLY the tick. Callers that want the per-field provenance
    record should use the ``with_provenance`` variant. The new optional
    keyword arguments enable the documented timeout-fallback chain and
    the SLA-violation derivation.
    """
    tick, _prov = telemetry_tick_with_provenance_from_inference_service_state(
        state,
        timestamp_s=timestamp_s,
        tick_duration_s=tick_duration_s,
        mean_utilization=mean_utilization,
        gpu_hours_delta=gpu_hours_delta,
        scale_events_delta=scale_events_delta,
        churn_delta=churn_delta,
        latency_sla_p99_ms=latency_sla_p99_ms,
        explicit_timeout_counter_rate=explicit_timeout_counter_rate,
        deadline_exceeded_rate=deadline_exceeded_rate,
        total_request_rate=total_request_rate,
        telemetry_confidence=telemetry_confidence,
        source=source,
    )
    return tick


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
