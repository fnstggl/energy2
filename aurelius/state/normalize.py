"""Normalization utilities for the Aurelius state layer.

These helpers adapt raw telemetry dicts into typed state model instances.
They are designed to be called by connectors (Prometheus, Kubernetes, etc.)
and the synthetic simulator identically — same code path for both.

Key invariants:
- Missing / unavailable metrics → None, never 0
- Timestamps are always UTC-aware on output
- Percentage values validated to [0, 100]
- Byte counts, rates, latencies validated to >= 0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    GPUState,
    InferenceServiceState,
    Provenance,
    QueueState,
    RuntimeType,
)

# ---------------------------------------------------------------------------
# Primitive validators (exported for test use)
# ---------------------------------------------------------------------------

def ensure_utc(ts: datetime) -> datetime:
    """Convert naive datetime to UTC-aware; pass through if already aware."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def validate_percentage(value: Optional[float], field_name: str = "") -> Optional[float]:
    """Return value if in [0, 100], else raise ValueError. None passes through."""
    if value is None:
        return None
    if not (0.0 <= value <= 100.0):
        raise ValueError(
            f"Percentage {field_name!r} must be in [0, 100], got {value}"
        )
    return value


def validate_non_negative(value: Optional[float], field_name: str = "") -> Optional[float]:
    """Return value if >= 0, else raise ValueError. None passes through."""
    if value is None:
        return None
    if value < 0.0:
        raise ValueError(f"Field {field_name!r} must be >= 0, got {value}")
    return value


def _maybe_float(raw: Any) -> Optional[float]:
    """Convert to float or return None on failure/None input."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _maybe_int(raw: Any) -> Optional[int]:
    """Convert to int or return None on failure/None input."""
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _safe_pct(raw: Any) -> Optional[float]:
    """Convert to float and clamp to [0, 100] or return None."""
    v = _maybe_float(raw)
    if v is None:
        return None
    if v < 0.0 or v > 100.0:
        return None
    return v


# ---------------------------------------------------------------------------
# GPUState normalization
# ---------------------------------------------------------------------------

def normalize_gpu_state(
    raw: dict[str, Any],
    provenance: Optional[Provenance] = None,
) -> GPUState:
    """Build a GPUState from a flat dict of raw metric values.

    Unknown or unparseable metrics are set to None, never 0.

    Expected raw keys (all optional except gpu_id):
        gpu_id, uuid, node_id, model, index, pci_bus_id,
        utilization_pct, sm_activity_pct,
        memory_used_bytes, memory_total_bytes, memory_bandwidth_util_pct,
        power_watts, power_limit_watts,
        temperature_c, thermal_throttle_active, thermal_slowdown_active,
        xid_error_count,
        nvlink_rx_bytes_per_sec, nvlink_tx_bytes_per_sec,
        pcie_rx_bytes_per_sec, pcie_tx_bytes_per_sec,
        assigned_workload_ids
    """
    def _bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes")
        return None

    return GPUState(
        gpu_id=str(raw["gpu_id"]),
        uuid=raw.get("uuid"),
        node_id=raw.get("node_id"),
        model=raw.get("model"),
        index=_maybe_int(raw.get("index")),
        pci_bus_id=raw.get("pci_bus_id"),
        utilization_pct=_safe_pct(raw.get("utilization_pct")),
        sm_activity_pct=_safe_pct(raw.get("sm_activity_pct")),
        memory_used_bytes=_maybe_int(raw.get("memory_used_bytes")),
        memory_total_bytes=_maybe_int(raw.get("memory_total_bytes")),
        memory_bandwidth_util_pct=_safe_pct(raw.get("memory_bandwidth_util_pct")),
        power_watts=_maybe_float(raw.get("power_watts")),
        power_limit_watts=_maybe_float(raw.get("power_limit_watts")),
        temperature_c=_maybe_float(raw.get("temperature_c")),
        thermal_throttle_active=_bool(raw.get("thermal_throttle_active")),
        thermal_slowdown_active=_bool(raw.get("thermal_slowdown_active")),
        xid_error_count=_maybe_int(raw.get("xid_error_count")),
        nvlink_rx_bytes_per_sec=_maybe_float(raw.get("nvlink_rx_bytes_per_sec")),
        nvlink_tx_bytes_per_sec=_maybe_float(raw.get("nvlink_tx_bytes_per_sec")),
        pcie_rx_bytes_per_sec=_maybe_float(raw.get("pcie_rx_bytes_per_sec")),
        pcie_tx_bytes_per_sec=_maybe_float(raw.get("pcie_tx_bytes_per_sec")),
        assigned_workload_ids=list(raw.get("assigned_workload_ids") or []),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# InferenceServiceState normalization
# ---------------------------------------------------------------------------

def normalize_inference_service(
    raw: dict[str, Any],
    provenance: Optional[Provenance] = None,
) -> InferenceServiceState:
    """Build an InferenceServiceState from a raw metric dict.

    Expected keys (all optional except service_id):
        service_id, runtime, node_ids, gpu_ids,
        requests_per_second, tokens_per_second,
        ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
        tpot_p50_ms, tpot_p95_ms, tpot_p99_ms,
        latency_p50_ms, latency_p95_ms, latency_p99_ms,
        queue_depth, queue_wait_p95_ms,
        active_sequences, batch_size,
        timeout_rate_pct, error_rate_pct,
        kv_cache_usage_pct, prefix_cache_hit_rate_pct
    """
    runtime_raw = raw.get("runtime", "unknown")
    try:
        runtime = RuntimeType(runtime_raw)
    except ValueError:
        runtime = RuntimeType.UNKNOWN

    return InferenceServiceState(
        service_id=str(raw["service_id"]),
        runtime=runtime,
        node_ids=list(raw.get("node_ids") or []),
        gpu_ids=list(raw.get("gpu_ids") or []),
        requests_per_second=_maybe_float(raw.get("requests_per_second")),
        tokens_per_second=_maybe_float(raw.get("tokens_per_second")),
        ttft_p50_ms=_maybe_float(raw.get("ttft_p50_ms")),
        ttft_p95_ms=_maybe_float(raw.get("ttft_p95_ms")),
        ttft_p99_ms=_maybe_float(raw.get("ttft_p99_ms")),
        tpot_p50_ms=_maybe_float(raw.get("tpot_p50_ms")),
        tpot_p95_ms=_maybe_float(raw.get("tpot_p95_ms")),
        tpot_p99_ms=_maybe_float(raw.get("tpot_p99_ms")),
        latency_p50_ms=_maybe_float(raw.get("latency_p50_ms")),
        latency_p95_ms=_maybe_float(raw.get("latency_p95_ms")),
        latency_p99_ms=_maybe_float(raw.get("latency_p99_ms")),
        queue_depth=_maybe_int(raw.get("queue_depth")),
        queue_wait_p95_ms=_maybe_float(raw.get("queue_wait_p95_ms")),
        active_sequences=_maybe_int(raw.get("active_sequences")),
        batch_size=_maybe_int(raw.get("batch_size")),
        timeout_rate_pct=_safe_pct(raw.get("timeout_rate_pct")),
        error_rate_pct=_safe_pct(raw.get("error_rate_pct")),
        kv_cache_usage_pct=_safe_pct(raw.get("kv_cache_usage_pct")),
        prefix_cache_hit_rate_pct=_safe_pct(raw.get("prefix_cache_hit_rate_pct")),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# QueueState normalization
# ---------------------------------------------------------------------------

def normalize_queue_state(
    raw: dict[str, Any],
    provenance: Optional[Provenance] = None,
) -> QueueState:
    """Build a QueueState from a raw metric dict."""
    return QueueState(
        queue_id=str(raw["queue_id"]),
        service_id=raw.get("service_id"),
        pending_jobs=_maybe_int(raw.get("pending_jobs")),
        queue_depth=_maybe_int(raw.get("queue_depth")),
        oldest_pending_age_sec=_maybe_float(raw.get("oldest_pending_age_sec")),
        p95_wait_ms=_maybe_float(raw.get("p95_wait_ms")),
        arrival_rate_per_sec=_maybe_float(raw.get("arrival_rate_per_sec")),
        service_rate_per_sec=_maybe_float(raw.get("service_rate_per_sec")),
        provenance=provenance,
    )
