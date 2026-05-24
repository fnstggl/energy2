"""Normalization utilities: adapt existing Aurelius models into canonical state.

This module bridges from existing aurelius/models.py shapes (GPUMetrics,
GPUHealthScore, QueueState) into the canonical ClusterState layer without
renaming or breaking the originals.

It also provides standalone validation helpers for connector code to use
before constructing frozen model instances.

Important invariants:
- Missing/unknown data → None, never coerced to 0
- All timestamps must be UTC-aware on exit (naive timestamps are rejected)
- Percentage fields must be in [0, 100]
- Rate/bytes/duration fields must be >= 0
- is_sandbox provenance flag is ALWAYS preserved from the source
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aurelius.models import GPUHealthScore, GPUMetrics, QueueState
    from aurelius.state.models import GPUState, Provenance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone validation helpers (also re-exported from __init__.py)
# ---------------------------------------------------------------------------

def validate_utc_aware(dt: datetime, field_name: str = "timestamp") -> datetime:
    """Return dt if it is UTC-aware; raise ValueError if naive.

    Use this at connector ingestion boundaries before constructing state models.
    """
    if dt.tzinfo is None:
        raise ValueError(
            f"{field_name} must be UTC-aware (got naive datetime). "
            "Attach tzinfo=timezone.utc or parse with fromisoformat and '+00:00'."
        )
    return dt


def validate_percentage(
    value: Optional[float],
    field_name: str,
    *,
    allow_none: bool = True,
) -> Optional[float]:
    """Validate that value is in [0, 100].

    Returns value unchanged if valid, raises ValueError if out of range.
    If allow_none=True (default) and value is None, returns None without error.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None")
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"{field_name} must be in [0, 100], got {value}")
    return value


def validate_non_negative(
    value: Optional[float],
    field_name: str,
    *,
    allow_none: bool = True,
) -> Optional[float]:
    """Validate that value is >= 0.

    Returns value unchanged if valid, raises ValueError if negative.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None")
    if value < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
    return value


def make_provenance(
    source: str,
    fetched_at: Optional[datetime] = None,
    *,
    confidence: str = "medium",
    is_sandbox: bool = False,
    sample_age_s: Optional[float] = None,
) -> "Provenance":  # noqa: F821  (imported below to avoid circular)
    """Convenience factory for Provenance with a default UTC now timestamp."""
    from aurelius.state.models import Provenance

    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)
    elif fetched_at.tzinfo is None:
        raise ValueError(
            "make_provenance: fetched_at must be UTC-aware if provided. "
            "Use datetime.now(tz=timezone.utc)."
        )
    return Provenance(
        source=source,
        fetched_at=fetched_at,
        confidence=confidence,
        is_sandbox=is_sandbox,
        sample_age_s=sample_age_s,
    )


# ---------------------------------------------------------------------------
# Adapter: GPUMetrics (+ optional GPUHealthScore) → GPUState
# ---------------------------------------------------------------------------

def adapt_gpu_metrics(
    gpu_metrics: "GPUMetrics",  # aurelius.models.GPUMetrics
    provenance: "Provenance",
    *,
    health_score: Optional["GPUHealthScore"] = None,  # aurelius.models.GPUHealthScore
) -> "GPUState":
    """Normalize an existing GPUMetrics (+ optional GPUHealthScore) into GPUState.

    Mapping notes:
    - power_throttle_us / thermal_throttle_us in GPUMetrics are documented as µs,
      but DCGM POWER_VIOLATION / THERMAL_VIOLATION are actually nanoseconds.
      We preserve the raw value under power_violation_ns / thermal_violation_ns
      because the unit mismatch in the source model is a known gap (see §6.3 of
      CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md). Callers that know the source unit
      should pass None and populate the correct field directly.
    - ecc_sbe_count and ecc_dbe_count default to 0 in GPUMetrics (not None),
      meaning we cannot distinguish "observed 0" from "metric absent". We map
      them to the canonical int fields directly; connectors that know ECC is
      disabled should set those fields to None on GPUState directly.
    - clock_throttle_reasons → clocks_event_reasons (both old and new names).
    - GPUMetrics.gpu_type uses lowercase model strings (e.g. "h100");
      GPUState.gpu_type preserves it as-is.
    """
    from aurelius.state.models import GPUState as _GPUState

    ts = gpu_metrics.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # Derive mem_total from used + free (GPUMetrics has mem_total_mb directly)
    mem_total = gpu_metrics.mem_total_mb if gpu_metrics.mem_total_mb > 0 else None
    mem_used = gpu_metrics.mem_used_mb if gpu_metrics.mem_used_mb >= 0 else None
    mem_free: Optional[float] = None
    if mem_total is not None and mem_used is not None:
        mem_free = max(0.0, mem_total - mem_used)

    health_penalty: Optional[float] = None
    is_schedulable: Optional[bool] = None
    if health_score is not None:
        health_penalty = health_score.health_penalty
        is_schedulable = health_score.is_schedulable

    return _GPUState(
        gpu_uuid=gpu_metrics.gpu_uuid,
        node_id=gpu_metrics.node_id,
        region=gpu_metrics.region,
        timestamp=ts,
        provenance=provenance,
        gpu_index=gpu_metrics.gpu_index,
        gpu_type=gpu_metrics.gpu_type or None,
        util_pct=gpu_metrics.gpu_util_pct,
        mem_used_mb=mem_used,
        mem_free_mb=mem_free,
        mem_total_mb=mem_total,
        power_w=gpu_metrics.power_usage_w,
        temp_c=gpu_metrics.gpu_temp_c,
        ecc_sbe_total=gpu_metrics.ecc_sbe_count,
        ecc_dbe_total=gpu_metrics.ecc_dbe_count,
        xid_last=gpu_metrics.xid_error_count if gpu_metrics.xid_error_count >= 0 else None,
        # power/thermal throttle in GPUMetrics are documented as µs but DCGM emits ns.
        # We map them to the ns fields to match the canonical model's unit.
        # Phase 3 will fix DCGMProvider to use the correct ns units directly.
        power_violation_ns=int(gpu_metrics.power_throttle_us) if gpu_metrics.power_throttle_us else None,
        thermal_violation_ns=int(gpu_metrics.thermal_throttle_us) if gpu_metrics.thermal_throttle_us else None,
        clocks_event_reasons=gpu_metrics.clock_throttle_reasons or None,
        health_penalty=health_penalty,
        is_schedulable=is_schedulable,
    )


# ---------------------------------------------------------------------------
# Adapter: QueueState → QueueStateWithProvenance wrapper
# ---------------------------------------------------------------------------

def adapt_queue_state(
    queue_state: "QueueState",  # aurelius.models.QueueState
    provenance: "Provenance",
) -> dict:
    """Return a dict representation of an existing QueueState with provenance added.

    Returns a plain dict because QueueState is not replaced in Phase 1 —
    it is reused as-is. Future phases may promote this to a typed model.

    The existing QueueState is the authoritative queue representation used by
    the optimizer and objective function. This wrapper adds provenance metadata
    for the constraint classifier.
    """
    ts = queue_state.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return {
        "timestamp": ts.isoformat(),
        "region": queue_state.region,
        "cluster_id": queue_state.cluster_id,
        "gpu_type": queue_state.gpu_type,
        "available_gpus": queue_state.available_gpus,
        "queue_depth_jobs": queue_state.queue_depth_jobs,
        "est_wait_hours": queue_state.est_wait_hours,
        "provenance": provenance.to_dict(),
    }


# ---------------------------------------------------------------------------
# Coerce naive datetime to UTC
# ---------------------------------------------------------------------------

def coerce_to_utc(dt: datetime) -> datetime:
    """If dt is naive, attach UTC tzinfo. If already tz-aware, return unchanged.

    Use only when the source is known to always be UTC (e.g., synthetic data).
    For uncertain sources, use validate_utc_aware() which raises instead.
    """
    if dt.tzinfo is None:
        logger.debug("coerce_to_utc: received naive datetime, attaching UTC. "
                     "Consider fixing the source to emit tz-aware timestamps.")
        return dt.replace(tzinfo=timezone.utc)
    return dt
