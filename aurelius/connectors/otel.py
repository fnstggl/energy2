"""OpenTelemetry (OTLP) fixture adapter for Aurelius.

This module provides a conceptual OTLP metric ingestion path for Aurelius.
It is fixture-based only — production OTLP integration should route through
the OpenTelemetry Collector → Prometheus or OTLP Receiver, not directly here.

Architecture intent:
    Real telemetry sources → OTel Collector → Prometheus (Prometheus exporter)
                                            → Aurelius via PrometheusConnector
    OR:
    Real telemetry sources → OTel Collector → OTLP receiver → this adapter (future)

This adapter parses a simplified OTLP-like JSON format as produced by the OTel
SDK or Collector and normalizes it into InferenceServiceState.

Field mapping from OTLP → InferenceServiceState (using existing model field names):
  request.latency.p50/p95/p99 → p50/p95/p99_latency_ms
  ttft.p50/p95/p99            → ttft_p50/p95/p99_ms
  queue.depth                 → requests_waiting
  token.rate                  → tokens_per_s
  kv_cache.usage_pct          → kv_cache_usage [clamped to 0-1 fraction]
  prefix_cache.hit_rate_pct   → prefix_cache_hit_rate [clamped to 0-1 fraction]
  error.rate                  → error_rate_pct
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from aurelius.state.models import InferenceServiceState, Provenance

logger = logging.getLogger(__name__)

# OTLP metric name → InferenceServiceState field mapping
_OTLP_METRIC_MAP: dict[str, str] = {
    "request.latency.p50": "p50_latency_ms",
    "request.latency.p95": "p95_latency_ms",
    "request.latency.p99": "p99_latency_ms",
    "request.rate": "tokens_per_s",      # rough approximation if tokens/s not available
    "token.rate": "tokens_per_s",
    "queue.depth": "requests_waiting",
    "ttft.p50": "ttft_p50_ms",
    "ttft.p95": "ttft_p95_ms",
    "ttft.p99": "ttft_p99_ms",
    "kv_cache.usage_pct": "kv_cache_usage",            # stored as 0-1 fraction
    "prefix_cache.hit_rate_pct": "prefix_cache_hit_rate",  # stored as 0-1 fraction
    "error.rate": "error_rate_pct",
}


def _parse_attribute_value(attr_value: dict[str, Any]) -> Any:
    """Extract value from OTLP attribute value struct."""
    for key in ("stringValue", "doubleValue", "intValue", "boolValue"):
        if key in attr_value:
            return attr_value[key]
    return None


def _parse_attributes(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse OTLP attributes list into a flat dict."""
    result = {}
    for attr in attributes:
        k = attr.get("key", "")
        v = _parse_attribute_value(attr.get("value", {}))
        if k:
            result[k] = v
    return result


def _extract_service_name(resource: dict[str, Any]) -> Optional[str]:
    """Extract service.name from OTLP resource attributes."""
    attrs = _parse_attributes(resource.get("attributes", []))
    return attrs.get("service.name")


def _extract_data_points(metric: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract data points from any OTLP metric type."""
    for kind in ("gauge", "sum", "histogram", "summary"):
        if kind in metric:
            return metric[kind].get("dataPoints", [])
    return []


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_otlp_json(payload: dict[str, Any]) -> dict[str, dict[str, Optional[float]]]:
    """Parse OTLP JSON export into a flat {service_name: {model_field: value}} dict.

    This is a best-effort parser for the simplified OTLP JSON format.
    Unknown metric names are skipped (logged at DEBUG level).

    Returns:
        Dict mapping service_name → {InferenceServiceState_field → float or None}
    """
    services: dict[str, dict[str, Optional[float]]] = {}

    resource_metrics = payload.get("resourceMetrics", [])
    for rm in resource_metrics:
        service_name = _extract_service_name(rm.get("resource", {})) or "__unknown__"

        if service_name not in services:
            services[service_name] = {}

        for scope_metrics in rm.get("scopeMetrics", []):
            for metric in scope_metrics.get("metrics", []):
                metric_name = metric.get("name", "")
                model_field = _OTLP_METRIC_MAP.get(metric_name)
                if model_field is None:
                    logger.debug("OTelAdapter: unknown metric %r — skipping", metric_name)
                    continue

                data_points = _extract_data_points(metric)
                if not data_points:
                    continue

                # Take the last data point's value
                dp = data_points[-1]
                value = _safe_float(dp.get("asDouble") or dp.get("asInt"))
                services[service_name][model_field] = value

    return services


class OTelAdapter:
    """Normalizes OTLP JSON payload into InferenceServiceState objects.

    This is a fixture/sandbox adapter. For production OTLP ingestion,
    route through the OpenTelemetry Collector → Prometheus → PrometheusConnector.
    """

    def normalize_from_otlp_json(
        self,
        payload: dict[str, Any],
        service_id_prefix: str = "otel",
        timestamp: Optional[datetime] = None,
        is_sandbox: bool = True,
    ) -> list[InferenceServiceState]:
        """Parse an OTLP JSON payload and normalize into InferenceServiceState list."""
        ts = timestamp or datetime.now(tz=timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        services = parse_otlp_json(payload)
        states = []

        for service_name, fields in services.items():
            service_id = f"{service_id_prefix}/{service_name}"
            provenance = Provenance(
                source="otel",
                fetched_at=ts,
                confidence="medium",
                is_sandbox=is_sandbox,
            )

            def _clamp_fraction(v: Optional[float]) -> Optional[float]:
                if v is None:
                    return None
                # If value is a percentage (0-100), convert to fraction (0-1)
                if v > 1.0:
                    v = v / 100.0
                return max(0.0, min(1.0, v))

            def _clamp_pct(v: Optional[float]) -> Optional[float]:
                if v is None:
                    return None
                return max(0.0, min(100.0, v))

            def _nn(v: Optional[float]) -> Optional[float]:
                if v is None:
                    return None
                return max(0.0, v)

            try:
                svc = InferenceServiceState(
                    service_id=service_id,
                    engine="unknown",
                    timestamp=ts,
                    provenance=provenance,
                    tokens_per_s=_nn(fields.get("tokens_per_s")),
                    requests_waiting=_nn(fields.get("requests_waiting")),
                    ttft_p50_ms=_nn(fields.get("ttft_p50_ms")),
                    ttft_p95_ms=_nn(fields.get("ttft_p95_ms")),
                    ttft_p99_ms=_nn(fields.get("ttft_p99_ms")),
                    p50_latency_ms=_nn(fields.get("p50_latency_ms")),
                    p95_latency_ms=_nn(fields.get("p95_latency_ms")),
                    p99_latency_ms=_nn(fields.get("p99_latency_ms")),
                    kv_cache_usage=_clamp_fraction(fields.get("kv_cache_usage")),
                    prefix_cache_hit_rate=_clamp_fraction(fields.get("prefix_cache_hit_rate")),
                    error_rate_pct=_clamp_pct(fields.get("error_rate_pct")),
                )
                states.append(svc)
            except Exception as exc:
                logger.warning(
                    "OTelAdapter: failed to normalize service %s: %s",
                    service_name, exc,
                )

        return states
