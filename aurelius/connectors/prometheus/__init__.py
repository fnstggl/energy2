"""Prometheus-native telemetry ingestion connector."""

from .client import PrometheusClient, PrometheusQueryError
from .connector import PrometheusTelemetryConnector, TelemetrySnapshot
from .mappings import DEFAULT_DCGM_MAPPINGS, DEFAULT_VLLM_MAPPINGS, MappingRegistry, MetricMapping

__all__ = [
    "PrometheusClient",
    "PrometheusQueryError",
    "PrometheusTelemetryConnector",
    "TelemetrySnapshot",
    "MetricMapping",
    "MappingRegistry",
    "DEFAULT_DCGM_MAPPINGS",
    "DEFAULT_VLLM_MAPPINGS",
]
