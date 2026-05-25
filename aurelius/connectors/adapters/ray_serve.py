"""Ray Serve Prometheus metrics adapter.

Ray Serve exposes Prometheus metrics. Key metrics include:
ray_serve_num_router_requests_total, ray_serve_deployment_queued_requests, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import InferenceServiceState, Provenance
from ...state.normalize import normalize_inference_service

_RAY_FIELD_MAP: dict[str, tuple[str, float]] = {
    "ray_serve_num_router_requests_total": ("_router_requests_total", 1.0),
    "ray_serve_num_deployment_errors_total": ("_error_total", 1.0),
    "ray_serve_deployment_queued_requests": ("queue_depth", 1.0),
    "ray_serve_replica_processing_latency_ms_sum": ("_latency_sum", 1.0),
    "ray_serve_replica_processing_latency_ms_count": ("_latency_count", 1.0),
    "ray_serve_num_ongoing_requests": ("active_sequences", 1.0),
}

_HISTOGRAM_PREFIXES = [
    "ray_serve_deployment_processing_latency_ms",
    "ray_serve_deployment_queuing_latency_ms",
]


@dataclass
class RayParseResult:
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class RayServeAdapter:
    """Parses Ray Serve Prometheus text metrics into InferenceServiceState."""

    def parse_text(
        self,
        metrics_text: str,
        source: str = "ray_serve",
    ) -> RayParseResult:
        result = RayParseResult()
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
                    "runtime": "ray_serve",
                    "_router_requests_total": None,
                    "_error_total": None,
                }
            self._apply_metric(name, labels, value, svc_raw[svc_key], result)

        # Post-process
        for raw in svc_raw.values():
            # Derive error_rate_pct from totals
            req_total = raw.pop("_router_requests_total", None)
            err_total = raw.pop("_error_total", None)
            if req_total is not None and req_total > 0 and err_total is not None:
                raw["error_rate_pct"] = min(100.0, (err_total / req_total) * 100.0)

            # Derive mean latency from _sum/_count (approximate p50 proxy)
            lat_sum = raw.pop("_latency_sum", None)
            lat_count = raw.pop("_latency_count", None)
            if lat_sum is not None and lat_count is not None and lat_count > 0:
                raw.setdefault("latency_p50_ms", lat_sum / lat_count)

            # Clean histogram remnants
            for prefix in _HISTOGRAM_PREFIXES:
                s = raw.pop(f"_hist_{prefix}_sum", None)
                c = raw.pop(f"_hist_{prefix}_count", None)
                if s is not None and c is not None and c > 0:
                    mean_ms = s / c
                    if prefix == "ray_serve_deployment_processing_latency_ms":
                        raw.setdefault("latency_p50_ms", mean_ms)
                    elif prefix == "ray_serve_deployment_queuing_latency_ms":
                        raw.setdefault("queue_wait_p95_ms", mean_ms)

        for svc_key, raw in svc_raw.items():
            # Remove internal keys before normalization
            clean = {k: v for k, v in raw.items() if not k.startswith("_")}
            prov = Provenance(
                source=source,
                collected_at=collected_at,
                confidence=0.9 if not result.unknown_metrics else 0.7,
            )
            try:
                result.services[svc_key] = normalize_inference_service(clean, prov)
            except Exception as exc:
                result.parse_errors.append(f"Service {svc_key}: {exc}")

        return result

    def _apply_metric(
        self,
        name: str,
        labels: dict[str, str],
        value: float,
        raw: dict[str, Any],
        result: RayParseResult,
    ) -> None:
        if name in _RAY_FIELD_MAP:
            canonical, mult = _RAY_FIELD_MAP[name]
            if not canonical.startswith("_"):
                raw[canonical] = value * mult
            else:
                # accumulate totals
                existing = raw.get(canonical)
                if existing is None:
                    raw[canonical] = value * mult
                else:
                    raw[canonical] = existing + value * mult
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
            labels.get("deployment")
            or labels.get("model_name")
            or labels.get("replica_tag", "").split("#")[0]
            or "ray_default"
        )
