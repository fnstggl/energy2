"""DCGM / dcgm-exporter Prometheus metrics adapter.

Parses raw Prometheus text-format /metrics output from dcgm-exporter
and normalizes it into GPUState instances.

Does not require a live DCGM installation — works from fixture text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import GPUState, Provenance
from ...state.normalize import normalize_gpu_state

# Map from DCGM metric name → canonical GPUState field
_DCGM_FIELD_MAP: dict[str, str] = {
    "DCGM_FI_DEV_GPU_UTIL": "utilization_pct",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE": "utilization_pct",  # fallback, *100 needed
    "DCGM_FI_PROF_SM_ACTIVE": "sm_activity_pct",        # *100 needed
    "DCGM_FI_DEV_FB_USED": "memory_used_bytes",
    "DCGM_FI_DEV_FB_FREE": "_fb_free",
    "DCGM_FI_PROF_DRAM_ACTIVE": "memory_bandwidth_util_pct",  # *100 needed
    "DCGM_FI_DEV_POWER_USAGE": "power_watts",
    "DCGM_FI_DEV_GPU_TEMP": "temperature_c",
    "DCGM_FI_DEV_POWER_VIOLATION": "_power_violation",
    "DCGM_FI_DEV_THERMAL_VIOLATION": "_thermal_violation",
    "DCGM_FI_DEV_XID_ERRORS": "xid_error_count",
    "DCGM_FI_PROF_NVLINK_RX_BYTES": "nvlink_rx_bytes_per_sec",
    "DCGM_FI_PROF_NVLINK_TX_BYTES": "nvlink_tx_bytes_per_sec",
    "DCGM_FI_PROF_PCIE_RX_BYTES": "pcie_rx_bytes_per_sec",
    "DCGM_FI_PROF_PCIE_TX_BYTES": "pcie_tx_bytes_per_sec",
}

# Fields that need *100 multiplication (fraction → percentage)
_FRACTION_TO_PCT = {
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
}

# Fields stored in MiB that need *1024*1024 to convert to bytes
_MIB_TO_BYTES = {
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_FREE",
}


@dataclass
class DCGMParseResult:
    gpus: dict[str, GPUState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class DCGMAdapter:
    """Parses dcgm-exporter Prometheus text metrics into GPUState objects.

    Usage::

        adapter = DCGMAdapter()
        result = adapter.parse_text(metrics_text, node_id="node-1")
        for gpu_id, gpu_state in result.gpus.items():
            ...
    """

    def parse_text(
        self,
        metrics_text: str,
        node_id: Optional[str] = None,
        source: str = "dcgm-exporter",
    ) -> DCGMParseResult:
        """Parse raw Prometheus text-format output from dcgm-exporter.

        Parameters
        ----------
        metrics_text:
            Raw /metrics text from dcgm-exporter endpoint.
        node_id:
            Node identifier to attach to all GPUState instances.
        source:
            Provenance source label.
        """
        result = DCGMParseResult()
        collected_at = datetime.now(timezone.utc)

        # gpu_key → raw metric dict
        gpu_raw: dict[str, dict[str, Any]] = {}
        seen_metric_names: set[str] = set()

        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                continue

            metric_name, labels, value = parsed
            seen_metric_names.add(metric_name)

            if metric_name not in _DCGM_FIELD_MAP:
                result.unknown_metrics.append(metric_name)
                continue

            gpu_key = self._gpu_key(labels)
            if gpu_key not in gpu_raw:
                gpu_raw[gpu_key] = {
                    "gpu_id": gpu_key,
                    "node_id": node_id,
                    "uuid": labels.get("UUID") or labels.get("uuid"),
                    "model": labels.get("modelName") or labels.get("model_name"),
                    "index": labels.get("gpu"),
                    "pci_bus_id": labels.get("pciId") or labels.get("pci_bus_id"),
                }

            canonical = _DCGM_FIELD_MAP[metric_name]

            if metric_name in _FRACTION_TO_PCT:
                value = value * 100.0
            elif metric_name in _MIB_TO_BYTES:
                value = value * 1024 * 1024

            if canonical == "_fb_free":
                gpu_raw[gpu_key]["_fb_free"] = value
            elif canonical == "_power_violation":
                if value > 0:
                    gpu_raw[gpu_key]["thermal_throttle_active"] = True
            elif canonical == "_thermal_violation":
                if value > 0:
                    gpu_raw[gpu_key]["thermal_throttle_active"] = True
            else:
                # Don't overwrite utilization_pct if already set by primary metric
                if canonical == "utilization_pct" and canonical in gpu_raw[gpu_key]:
                    pass
                else:
                    gpu_raw[gpu_key][canonical] = value

        # Derive memory_total_bytes from used + free
        for raw in gpu_raw.values():
            fb_free = raw.pop("_fb_free", None)
            if fb_free is not None and "memory_used_bytes" in raw:
                raw["memory_total_bytes"] = raw["memory_used_bytes"] + fb_free

        # Build GPUState objects
        for gpu_key, raw in gpu_raw.items():
            prov = Provenance(
                source=source,
                collected_at=collected_at,
                confidence=1.0 if not result.unknown_metrics else 0.8,
            )
            try:
                result.gpus[gpu_key] = normalize_gpu_state(raw, prov)
            except Exception as exc:
                result.parse_errors.append(f"GPU {gpu_key}: {exc}")

        return result

    def _parse_line(
        self, line: str
    ) -> Optional[tuple[str, dict[str, str], float]]:
        """Parse a single Prometheus text-format metric line.

        Returns (metric_name, labels_dict, float_value) or None.
        """
        # Pattern: metric_name{label="val",...} value [timestamp]
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)', line)
        if not match:
            return None

        name = match.group(1)
        labels_str = match.group(2) or ""
        value_str = match.group(3)

        try:
            value = float(value_str)
        except ValueError:
            return None

        labels: dict[str, str] = {}
        if labels_str:
            for kv in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[kv.group(1)] = kv.group(2)

        return name, labels, value

    def _gpu_key(self, labels: dict[str, str]) -> str:
        return (
            labels.get("UUID")
            or labels.get("uuid")
            or f"gpu_{labels.get('gpu', 'unknown')}"
        )
