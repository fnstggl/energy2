"""Telemetry connectors for Aurelius constraint-aware GPU orchestration.

This package provides:
- Generic Prometheus HTTP client (real + sandbox/fixture)
- Metric mapping layer (canonical field → Prometheus query)
- Vendor adapters: DCGM, vLLM, Triton, Ray Serve, OTel

All connectors share the same interface and normalize into the canonical
ClusterState model from aurelius.state. Missing metrics → None, never 0.
"""

from aurelius.connectors.base import (
    AuthConfig,
    AuthType,
    ConnectorConfig,
    MetricLabel,
    MetricValue,
    RawMetricResult,
    TelemetrySnapshot,
)
from aurelius.connectors.metric_mapping import (
    MetricMapping,
    MetricMappingRegistry,
    UnitConversion,
    load_mapping_yaml,
)
from aurelius.connectors.prometheus import (
    FakePrometheusClient,
    PrometheusClient,
    PrometheusTelemetryConnector,
    parse_prometheus_text,
)

__all__ = [
    # Base types
    "AuthConfig",
    "AuthType",
    "ConnectorConfig",
    "MetricLabel",
    "MetricValue",
    "RawMetricResult",
    "TelemetrySnapshot",
    # Metric mapping
    "MetricMapping",
    "MetricMappingRegistry",
    "UnitConversion",
    "load_mapping_yaml",
    # Prometheus
    "FakePrometheusClient",
    "PrometheusClient",
    "PrometheusTelemetryConnector",
    "parse_prometheus_text",
]
