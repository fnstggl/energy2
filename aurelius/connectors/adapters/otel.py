"""OpenTelemetry (OTLP) metrics adapter.

Parses OTLP-like JSON metric payloads and normalizes into InferenceServiceState.

Strategy:
OpenTelemetry Collector is vendor-agnostic — it can receive/process/export
telemetry across open formats (OTLP, Prometheus, StatsD, etc.) and various
backends. In production, OTLP can flow through OpenTelemetry Collector into
Prometheus (where the Prometheus connector handles it) or via direct OTLP
receiver.

This adapter handles direct OTLP JSON payloads for cases where Prometheus
is not the intermediary.

OTLP JSON schema reference:
https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import InferenceServiceState, Provenance
from ...state.normalize import normalize_inference_service


@dataclass
class OTelParseResult:
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


# Map OTLP metric name → canonical InferenceServiceState field
_OTEL_FIELD_MAP: dict[str, str] = {
    "inference.request_rate": "requests_per_second",
    "inference.token_rate": "tokens_per_second",
    "inference.ttft_p50": "ttft_p50_ms",
    "inference.ttft_p95": "ttft_p95_ms",
    "inference.ttft_p99": "ttft_p99_ms",
    "inference.latency_p50": "latency_p50_ms",
    "inference.latency_p95": "latency_p95_ms",
    "inference.latency_p99": "latency_p99_ms",
    "inference.queue_depth": "queue_depth",
    "inference.queue_wait_p95": "queue_wait_p95_ms",
    "inference.active_sequences": "active_sequences",
    "inference.error_rate": "error_rate_pct",
    "inference.kv_cache_usage": "kv_cache_usage_pct",
    "inference.prefix_cache_hit_rate": "prefix_cache_hit_rate_pct",
}


class OTelAdapter:
    """Parses OTLP-like JSON metric payloads into InferenceServiceState.

    Accepts either:
    1. Raw OTLP protobuf-JSON format (resourceMetrics array)
    2. Simplified flat dict format used by test fixtures

    Missing metrics are recorded in result.unknown_metrics, not fabricated.
    """

    def parse_otlp_json(
        self,
        payload: dict[str, Any],
        source: str = "otel",
    ) -> OTelParseResult:
        """Parse an OTLP ExportMetricsServiceRequest JSON payload."""
        result = OTelParseResult()
        collected_at = datetime.now(timezone.utc)
        svc_raw: dict[str, dict[str, Any]] = {}

        for resource_metric in payload.get("resourceMetrics", []):
            resource_attrs = self._extract_attrs(
                resource_metric.get("resource", {}).get("attributes", [])
            )
            service_name = resource_attrs.get("service.name", "unknown")

            for scope_metric in resource_metric.get("scopeMetrics", []):
                for metric in scope_metric.get("metrics", []):
                    self._process_metric(
                        metric, service_name, svc_raw, result
                    )

        for svc_key, raw in svc_raw.items():
            prov = Provenance(
                source=source,
                collected_at=collected_at,
                confidence=1.0 if not result.unknown_metrics else 0.8,
            )
            try:
                result.services[svc_key] = normalize_inference_service(raw, prov)
            except Exception as exc:
                result.parse_errors.append(f"Service {svc_key}: {exc}")

        return result

    def parse_flat_dict(
        self,
        payload: dict[str, Any],
        source: str = "otel",
    ) -> OTelParseResult:
        """Parse a simplified flat metric dict (used in tests and fixtures).

        Expected format::

            {
                "service_id": "my-service",
                "runtime": "vllm",
                "inference.request_rate": 42.0,
                "inference.ttft_p95": 150.0,
                ...
            }
        """
        result = OTelParseResult()
        collected_at = datetime.now(timezone.utc)

        raw: dict[str, Any] = {
            "service_id": payload.get("service_id", "otel_default"),
            "runtime": payload.get("runtime", "unknown"),
        }

        for key, value in payload.items():
            if key in ("service_id", "runtime"):
                continue
            if key in _OTEL_FIELD_MAP:
                raw[_OTEL_FIELD_MAP[key]] = value
            else:
                result.unknown_metrics.append(key)

        prov = Provenance(
            source=source,
            collected_at=collected_at,
            confidence=1.0 if not result.unknown_metrics else 0.8,
        )
        try:
            result.services[raw["service_id"]] = normalize_inference_service(raw, prov)
        except Exception as exc:
            result.parse_errors.append(f"Service normalization: {exc}")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_attrs(self, attrs: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for attr in attrs:
            key = attr.get("key", "")
            value_wrapper = attr.get("value", {})
            if "stringValue" in value_wrapper:
                out[key] = value_wrapper["stringValue"]
            elif "intValue" in value_wrapper:
                out[key] = str(value_wrapper["intValue"])
            elif "doubleValue" in value_wrapper:
                out[key] = str(value_wrapper["doubleValue"])
        return out

    def _process_metric(
        self,
        metric: dict[str, Any],
        service_name: str,
        svc_raw: dict[str, dict[str, Any]],
        result: OTelParseResult,
    ) -> None:
        name = metric.get("name", "")
        if service_name not in svc_raw:
            svc_raw[service_name] = {
                "service_id": service_name,
                "runtime": "unknown",
            }

        canonical = _OTEL_FIELD_MAP.get(name)
        if canonical is None:
            result.unknown_metrics.append(name)
            return

        value = self._extract_metric_value(metric)
        if value is not None:
            svc_raw[service_name][canonical] = value

    def _extract_metric_value(self, metric: dict[str, Any]) -> Optional[float]:
        """Extract the most recent data point value from an OTLP metric."""
        for data_key in ("gauge", "sum", "histogram"):
            data = metric.get(data_key)
            if data is None:
                continue
            data_points = data.get("dataPoints", [])
            if not data_points:
                continue
            dp = data_points[-1]
            if "asDouble" in dp:
                return float(dp["asDouble"])
            if "asInt" in dp:
                return float(dp["asInt"])
            if data_key == "histogram":
                # Return sum/count mean as a proxy value
                s = dp.get("sum")
                c = dp.get("count")
                if s is not None and c and c > 0:
                    return float(s) / float(c)
        return None
