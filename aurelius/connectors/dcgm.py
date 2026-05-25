"""DCGM/dcgm-exporter adapter for Aurelius.

Normalizes Prometheus metrics from dcgm-exporter into canonical GPUState objects.

Key design decisions:
- All optional DCGM metrics default to None when not available/disabled
- Supports both per-GPU (UUID label) and per-node aggregation
- GPU UUID is the canonical identity key
- clocks_event_reasons supports both metric name variants:
    DCGM_FI_DEV_CLOCKS_EVENT_REASONS (newer dcgm-exporter >= 3.x)
    DCGM_FI_DEV_CLOCK_THROTTLE_REASONS (older, deprecated)
- thermal_violation_ns / power_violation_ns are nanoseconds (not µs)
  as documented in the plan §6.3; the old dcgm_provider.py had a µs bug
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aurelius.connectors.base import TelemetrySnapshot
from aurelius.connectors.metric_mapping import MetricMappingRegistry, dcgm_registry
from aurelius.state.models import GPUState, Provenance

logger = logging.getLogger(__name__)

# DCGM label names for GPU identity
_LABEL_GPU_INDEX = "gpu"
_LABEL_GPU_UUID = "UUID"
_LABEL_NODE = "node"
_LABEL_MODEL = "modelName"

# Canonical DCGM fields (matches metric_mapping._DCGM_BUILTIN keys)
_F_UTIL = "gpu.util_pct"
_F_SM_ACTIVE = "gpu.sm_active_ratio"
_F_TENSOR_ACTIVE = "gpu.tensor_active_ratio"
_F_DRAM_ACTIVE = "gpu.dram_active_ratio"
_F_MEM_USED = "gpu.mem_used_mb"
_F_MEM_FREE = "gpu.mem_free_mb"
_F_MEM_RESERVED = "gpu.mem_reserved_mb"
_F_POWER = "gpu.power_w"
_F_POWER_LIMIT = "gpu.power_limit_w"
_F_TEMP = "gpu.temp_c"
_F_MEM_TEMP = "gpu.mem_temp_c"
_F_SM_CLOCK = "gpu.sm_clock_mhz"
_F_CLOCKS_EVENT = "gpu.clocks_event_reasons"
_F_POWER_VIOLATION = "gpu.power_violation_ns"
_F_THERMAL_VIOLATION = "gpu.thermal_violation_ns"
_F_ECC_SBE = "gpu.ecc_sbe_total"
_F_ECC_DBE = "gpu.ecc_dbe_total"
_F_XID = "gpu.xid_last"
_F_XID_WINDOW = "gpu.xid_count_window"
_F_PCIE_TX = "gpu.pcie_tx_bytes_per_s"
_F_PCIE_RX = "gpu.pcie_rx_bytes_per_s"
_F_NVLINK_TX = "gpu.nvlink_tx_bytes_per_s"
_F_NVLINK_RX = "gpu.nvlink_rx_bytes_per_s"


def _get_scalar(
    snapshot: TelemetrySnapshot,
    field: str,
    **label_filters: str,
) -> Optional[float]:
    """Extract a single float value from a snapshot field, optionally filtered by labels."""
    result = snapshot.get(field)
    if result is None or result.missing:
        return None
    if label_filters:
        return result.value_for_labels(**label_filters)
    return result.first_value


def _get_int(
    snapshot: TelemetrySnapshot,
    field: str,
    **label_filters: str,
) -> Optional[int]:
    v = _get_scalar(snapshot, field, **label_filters)
    if v is None:
        return None
    return int(v)


def _get_mem_total_mb(snapshot: TelemetrySnapshot, **label_filters: str) -> Optional[float]:
    """Derive mem_total_mb from used + free + reserved (DCGM convention)."""
    used = _get_scalar(snapshot, _F_MEM_USED, **label_filters)
    free = _get_scalar(snapshot, _F_MEM_FREE, **label_filters)
    reserved = _get_scalar(snapshot, _F_MEM_RESERVED, **label_filters)

    if used is not None and free is not None:
        total = used + free
        if reserved is not None:
            total += reserved
        return total
    return None


def _health_penalty(gpu_state: GPUState) -> Optional[float]:
    """Compute a 0-1 health penalty from GPU state (matches existing DCGMProvider logic)."""
    if gpu_state.util_pct is None and gpu_state.temp_c is None:
        return None

    penalty = 0.0
    _UTIL_WARN = 80.0
    _TEMP_SAFE = 70.0
    _TEMP_CRITICAL = 95.0
    _THROTTLE_MAX_NS = 1_000_000_000  # 1 second in ns

    if gpu_state.util_pct is not None and gpu_state.util_pct > _UTIL_WARN:
        penalty += (gpu_state.util_pct - _UTIL_WARN) / (100.0 - _UTIL_WARN) * 0.3

    if gpu_state.temp_c is not None:
        if gpu_state.temp_c >= _TEMP_CRITICAL:
            penalty += 0.5
        elif gpu_state.temp_c > _TEMP_SAFE:
            penalty += (gpu_state.temp_c - _TEMP_SAFE) / (_TEMP_CRITICAL - _TEMP_SAFE) * 0.4

    if gpu_state.thermal_violation_ns is not None and gpu_state.thermal_violation_ns > 0:
        throttle_ratio = min(1.0, gpu_state.thermal_violation_ns / _THROTTLE_MAX_NS)
        penalty += throttle_ratio * 0.3

    if gpu_state.ecc_dbe_total is not None and gpu_state.ecc_dbe_total > 0:
        penalty += min(0.5, 0.1 * gpu_state.ecc_dbe_total)

    return min(1.0, penalty)


class DCGMAdapter:
    """Normalizes DCGM Prometheus metrics → GPUState objects.

    All adapters share the same interface: they accept a TelemetrySnapshot
    (from either PrometheusClient or FakePrometheusClient) and return
    canonical state objects. The same code runs in sandbox and production.
    """

    def __init__(self, registry: Optional[MetricMappingRegistry] = None) -> None:
        self._registry = registry or dcgm_registry()

    def all_gpu_uuids(self, snapshot: TelemetrySnapshot) -> list[str]:
        """Extract all unique GPU UUIDs from the snapshot."""
        result = snapshot.get(_F_UTIL)
        if result is None or result.missing:
            result = snapshot.get(_F_POWER)
        if result is None or result.missing:
            return []
        uuids = []
        for mv in result.values:
            uuid = mv.labels.get(_LABEL_GPU_UUID)
            if uuid and uuid not in uuids:
                uuids.append(uuid)
        return uuids

    def normalize_gpu_state(
        self,
        snapshot: TelemetrySnapshot,
        gpu_uuid: str,
        node_id: str,
        region: str,
        timestamp: Optional[datetime] = None,
        gpu_type: Optional[str] = None,
        gpu_index: Optional[int] = None,
    ) -> GPUState:
        """Normalize a single GPU's metrics from a snapshot into GPUState.

        Uses UUID-filtered label lookups so each GPU is correctly extracted
        from multi-GPU node snapshots.
        """
        ts = timestamp or snapshot.fetched_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        lf = {_LABEL_GPU_UUID: gpu_uuid}

        # Derive gpu_index from snapshot labels if not provided
        if gpu_index is None:
            idx_str = None
            util_result = snapshot.get(_F_UTIL)
            if util_result and not util_result.missing:
                for mv in util_result.values:
                    if mv.labels.get(_LABEL_GPU_UUID) == gpu_uuid:
                        idx_str = mv.labels.get(_LABEL_GPU_INDEX)
                        if gpu_type is None:
                            gpu_type = mv.labels.get(_LABEL_MODEL)
                        break
            if idx_str is not None:
                try:
                    gpu_index = int(idx_str)
                except (ValueError, TypeError):
                    pass

        # Derive node_id from labels if not provided
        if node_id == "__auto__":
            util_result = snapshot.get(_F_UTIL)
            if util_result and not util_result.missing:
                for mv in util_result.values:
                    if mv.labels.get(_LABEL_GPU_UUID) == gpu_uuid:
                        node_id = mv.labels.get(_LABEL_NODE) or node_id
                        break

        mem_total = _get_mem_total_mb(snapshot, **lf)

        provenance = Provenance(
            source=snapshot.source,
            fetched_at=snapshot.fetched_at,
            confidence="high" if snapshot.coverage_pct() > 70 else "medium" if snapshot.coverage_pct() > 40 else "low",
            is_sandbox=snapshot.is_sandbox,
        )

        gpu = GPUState(
            gpu_uuid=gpu_uuid,
            node_id=node_id,
            region=region,
            timestamp=ts,
            provenance=provenance,
            gpu_index=gpu_index,
            gpu_type=gpu_type,
            util_pct=_get_scalar(snapshot, _F_UTIL, **lf),
            sm_active_ratio=_get_scalar(snapshot, _F_SM_ACTIVE, **lf),
            tensor_active_ratio=_get_scalar(snapshot, _F_TENSOR_ACTIVE, **lf),
            dram_active_ratio=_get_scalar(snapshot, _F_DRAM_ACTIVE, **lf),
            mem_used_mb=_get_scalar(snapshot, _F_MEM_USED, **lf),
            mem_free_mb=_get_scalar(snapshot, _F_MEM_FREE, **lf),
            mem_reserved_mb=_get_scalar(snapshot, _F_MEM_RESERVED, **lf),
            mem_total_mb=mem_total,
            power_w=_get_scalar(snapshot, _F_POWER, **lf),
            power_limit_w=_get_scalar(snapshot, _F_POWER_LIMIT, **lf),
            temp_c=_get_scalar(snapshot, _F_TEMP, **lf),
            mem_temp_c=_get_scalar(snapshot, _F_MEM_TEMP, **lf),
            sm_clock_mhz=_get_scalar(snapshot, _F_SM_CLOCK, **lf),
            clocks_event_reasons=_get_int(snapshot, _F_CLOCKS_EVENT, **lf),
            power_violation_ns=_get_int(snapshot, _F_POWER_VIOLATION, **lf),
            thermal_violation_ns=_get_int(snapshot, _F_THERMAL_VIOLATION, **lf),
            ecc_sbe_total=_get_int(snapshot, _F_ECC_SBE, **lf),
            ecc_dbe_total=_get_int(snapshot, _F_ECC_DBE, **lf),
            xid_last=_get_int(snapshot, _F_XID, **lf),
            xid_count_window=_get_int(snapshot, _F_XID_WINDOW, **lf),
            pcie_tx_bytes_per_s=_get_scalar(snapshot, _F_PCIE_TX, **lf),
            pcie_rx_bytes_per_s=_get_scalar(snapshot, _F_PCIE_RX, **lf),
            nvlink_tx_bytes_per_s=_get_scalar(snapshot, _F_NVLINK_TX, **lf),
            nvlink_rx_bytes_per_s=_get_scalar(snapshot, _F_NVLINK_RX, **lf),
        )

        # Compute health penalty (same logic as existing DCGMProvider)
        hp = _health_penalty(gpu)
        is_schedulable = (hp is not None and hp < 1.0) or hp is None

        # Return a new instance with health_penalty set (frozen dataclass workaround)
        return GPUState(
            **{
                **{f: getattr(gpu, f) for f in gpu.__dataclass_fields__},
                "health_penalty": hp,
                "is_schedulable": is_schedulable,
            }
        )

    def normalize_gpus(
        self,
        snapshot: TelemetrySnapshot,
        node_id: str,
        region: str,
        timestamp: Optional[datetime] = None,
    ) -> list[GPUState]:
        """Normalize all GPUs found in a snapshot for a given node."""
        uuids = self.all_gpu_uuids(snapshot)
        if not uuids:
            logger.info("DCGMAdapter: no GPU UUIDs found in snapshot from %s", snapshot.source)
            return []

        states = []
        for uuid in uuids:
            try:
                gpu = self.normalize_gpu_state(
                    snapshot=snapshot,
                    gpu_uuid=uuid,
                    node_id=node_id,
                    region=region,
                    timestamp=timestamp,
                )
                states.append(gpu)
            except Exception as exc:
                logger.warning("DCGMAdapter: failed to normalize GPU %s: %s", uuid, exc)

        return states

    def unknown_metrics(self, snapshot: TelemetrySnapshot) -> list[str]:
        """Return list of DCGM metrics not available in this snapshot."""
        return snapshot.unknown_metrics
