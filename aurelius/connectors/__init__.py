"""Telemetry connectors for Aurelius constraint-aware GPU orchestration.

This package provides:
- Generic Prometheus HTTP client (real + sandbox/fixture)
- Metric mapping layer (canonical field → Prometheus query)
- Vendor adapters: DCGM, vLLM, Triton, Ray Serve, OTel
- Kubernetes read-only placement connector (Phase 4)
- GPU topology collector and placement scorer (Phase 5)

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
from aurelius.connectors.kubernetes import (
    FakeKubernetesConnector,
    K8sPlacementSnapshot,
    K8sReplicaDelta,
    KubernetesConnector,
    KubernetesConnectorConfig,
    PodPlacement,
    compute_k8s_scale_delta,
    normalize_node_dict,
    normalize_pod_dict,
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
from aurelius.connectors.topology import (
    FakeTopologyCollector,
    NvidiaSmiTopologyCollector,
    PlacementScore,
    PlacementWorkloadSpec,
    build_topology_state,
    parse_nvidia_smi_list,
    parse_nvidia_smi_topo,
    rank_placements,
    score_placement,
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
    # Kubernetes (Phase 4)
    "FakeKubernetesConnector",
    "K8sPlacementSnapshot",
    "K8sReplicaDelta",
    "KubernetesConnector",
    "KubernetesConnectorConfig",
    "PodPlacement",
    "compute_k8s_scale_delta",
    "normalize_node_dict",
    "normalize_pod_dict",
    # Topology (Phase 5)
    "FakeTopologyCollector",
    "NvidiaSmiTopologyCollector",
    "PlacementScore",
    "PlacementWorkloadSpec",
    "build_topology_state",
    "parse_nvidia_smi_list",
    "parse_nvidia_smi_topo",
    "rank_placements",
    "score_placement",
]
