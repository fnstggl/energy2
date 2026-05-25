"""Triton Inference Server Prometheus metrics adapter.

Triton exposes Prometheus metrics by default at localhost:8002/metrics.
Key metrics: nv_inference_request_success, nv_inference_request_duration_us, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import InferenceServiceState, Provenance
from ...state.normalize import normalize_inference_service

_TRITON_FIELD_MAP: dict[str, tuple[str, float]] = {
    "nv_inference_request_success": ("_req_success_total", 1.0),
    "nv_inference_request_failure": ("_req_failure_total", 1.0),
    "nv_inference_count": ("_inference_count", 1.0),
    "nv_inference_exec_count": ("_exec_count", 1.0),
    "nv_inference_pending_request_count": ("queue_depth", 1.0),
}

_HISTOGRAM_PREFIXES = [
    "nv_inference_request_duration_us",
    "nv_inference_queue_duration_us",
    "nv_inference_compute_input_duration_us",
    "nv_inference_compute_infer_duration_us",
    "nv_inference_compute_output_duration_us",
]


@dataclass
class TritonParseResult:
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class TritonAdapter:
    """Parses Triton Prometheus text metrics into InferenceServiceState."""

    def parse_text(
        self,
        metrics_text: str,
        source: str = "triton",
    ) -> TritonParseResult:
        result = TritonParseResult()
        collected_at = datetime.now(timezone.utc)
        svc_raw: dict[str, dict[str, Any]] = {}

        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = self._parse_line(line)
            if parsed is None:
                continue
            name, labels, value = parsed
            svc_key = self._svc_key(labels)
            if svc_key not in svc_raw:
                svc_raw[svc_key] = {
                    "service_id": svc_key,
                    "runtime": "triton",
                }
            self._apply_metric(name, labels, value, svc_raw[svc_key], result)

        # Derive latency from histogram sum/count (mean approximation only)
        for raw in svc_raw.values():
            for prefix in _HISTOGRAM_PREFIXES:
                s = raw.pop(f"_hist_{prefix}_sum", None)
                c = raw.pop(f"_hist_{prefix}_count", None)
                if s is not None and c is not None and c > 0:
                    mean_us = s / c
                    mean_ms = mean_us / 1000.0
                    if prefix == "nv_inference_request_duration_us":
                        if "latency_p50_ms" not in raw:
                            raw["latency_p50_ms"] = mean_ms
                    elif prefix == "nv_inference_queue_duration_us":
                        if "queue_wait_p95_ms" not in raw:
                            raw["queue_wait_p95_ms"] = mean_ms

        for svc_key, raw in svc_raw.items():
            prov = Provenance(
                source=source,
                collected_at=collected_at,
                confidence=0.9 if not result.unknown_metrics else 0.7,
            )
            try:
                result.services[svc_key] = normalize_inference_service(raw, prov)
            except Exception as exc:
                result.parse_errors.append(f"Service {svc_key}: {exc}")

        return result

    def _apply_metric(
        self,
        name: str,
        labels: dict[str, str],
        value: float,
        raw: dict[str, Any],
        result: TritonParseResult,
    ) -> None:
        if name in _TRITON_FIELD_MAP:
            canonical, mult = _TRITON_FIELD_MAP[name]
            if not canonical.startswith("_"):
                raw[canonical] = value * mult
        elif any(name.startswith(p) for p in _HISTOGRAM_PREFIXES):
            for prefix in _HISTOGRAM_PREFIXES:
                if name == f"{prefix}_sum":
                    raw[f"_hist_{prefix}_sum"] = value
                elif name == f"{prefix}_count":
                    raw[f"_hist_{prefix}_count"] = value
        else:
            result.unknown_metrics.append(name)

    def _parse_line(self, line: str) -> Optional[tuple[str, dict[str, str], float]]:
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

    def _svc_key(self, labels: dict[str, str]) -> str:
        return (
            labels.get("model")
            or labels.get("model_name")
            or "triton_default"
        )
