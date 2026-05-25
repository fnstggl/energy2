"""Metric mapping configuration for Prometheus connector.

Maps canonical Aurelius field names to Prometheus query strings.
Supports multiple fallback queries and unit conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricMapping:
    """Maps a canonical field to one or more Prometheus queries."""

    canonical_field: str
    queries: list[str]
    unit_conversion: float = 1.0
    label_map: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def primary_query(self) -> str:
        return self.queries[0]

    @property
    def fallback_queries(self) -> list[str]:
        return self.queries[1:]


class MappingRegistry:
    """Registry of MetricMappings for a specific adapter/source."""

    def __init__(self, name: str, mappings: list[MetricMapping]) -> None:
        self.name = name
        self._by_field: dict[str, MetricMapping] = {m.canonical_field: m for m in mappings}

    def get(self, canonical_field: str) -> Optional[MetricMapping]:
        return self._by_field.get(canonical_field)

    def all_fields(self) -> list[str]:
        return list(self._by_field.keys())

    def all_mappings(self) -> list[MetricMapping]:
        return list(self._by_field.values())


# ---------------------------------------------------------------------------
# DCGM / dcgm-exporter default mappings
# ---------------------------------------------------------------------------

DEFAULT_DCGM_MAPPINGS = MappingRegistry(
    name="dcgm",
    mappings=[
        MetricMapping(
            canonical_field="gpu.utilization_pct",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_GPU_UTIL)",
                "avg by (gpu, node, UUID) (DCGM_FI_PROF_GR_ENGINE_ACTIVE * 100)",
            ],
            description="GPU utilization percentage",
        ),
        MetricMapping(
            canonical_field="gpu.sm_activity_pct",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_PROF_SM_ACTIVE * 100)",
            ],
            description="SM active percentage",
        ),
        MetricMapping(
            canonical_field="gpu.memory_used_bytes",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_FB_USED * 1024 * 1024)",
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_FB_USED)",
            ],
            unit_conversion=1.0,
            description="GPU framebuffer used (bytes)",
        ),
        MetricMapping(
            canonical_field="gpu.memory_total_bytes",
            queries=[
                "avg by (gpu, node, UUID) ((DCGM_FI_DEV_FB_FREE + DCGM_FI_DEV_FB_USED) * 1024 * 1024)",
            ],
            description="GPU framebuffer total (bytes)",
        ),
        MetricMapping(
            canonical_field="gpu.memory_bandwidth_util_pct",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_PROF_DRAM_ACTIVE * 100)",
            ],
            description="Memory bandwidth utilization percentage",
        ),
        MetricMapping(
            canonical_field="gpu.power_watts",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_POWER_USAGE)",
            ],
            description="GPU power draw in watts",
        ),
        MetricMapping(
            canonical_field="gpu.temperature_c",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_GPU_TEMP)",
            ],
            description="GPU temperature in Celsius",
        ),
        MetricMapping(
            canonical_field="gpu.thermal_throttle_active",
            queries=[
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_POWER_VIOLATION > 0)",
                "avg by (gpu, node, UUID) (DCGM_FI_DEV_THERMAL_VIOLATION > 0)",
            ],
            description="Whether GPU is thermally throttling",
        ),
        MetricMapping(
            canonical_field="gpu.xid_error_count",
            queries=[
                "sum by (gpu, node, UUID) (DCGM_FI_DEV_XID_ERRORS)",
            ],
            description="XID error count",
        ),
        MetricMapping(
            canonical_field="gpu.nvlink_rx_bytes_per_sec",
            queries=[
                "sum by (gpu, node, UUID) (rate(DCGM_FI_PROF_NVLINK_RX_BYTES[1m]))",
            ],
            description="NVLink receive bytes/sec",
        ),
        MetricMapping(
            canonical_field="gpu.nvlink_tx_bytes_per_sec",
            queries=[
                "sum by (gpu, node, UUID) (rate(DCGM_FI_PROF_NVLINK_TX_BYTES[1m]))",
            ],
            description="NVLink transmit bytes/sec",
        ),
        MetricMapping(
            canonical_field="gpu.pcie_rx_bytes_per_sec",
            queries=[
                "sum by (gpu, node, UUID) (rate(DCGM_FI_PROF_PCIE_RX_BYTES[1m]))",
            ],
            description="PCIe receive bytes/sec",
        ),
        MetricMapping(
            canonical_field="gpu.pcie_tx_bytes_per_sec",
            queries=[
                "sum by (gpu, node, UUID) (rate(DCGM_FI_PROF_PCIE_TX_BYTES[1m]))",
            ],
            description="PCIe transmit bytes/sec",
        ),
    ],
)


# ---------------------------------------------------------------------------
# vLLM default mappings
# ---------------------------------------------------------------------------

DEFAULT_VLLM_MAPPINGS = MappingRegistry(
    name="vllm",
    mappings=[
        MetricMapping(
            canonical_field="inference.requests_per_second",
            queries=[
                "sum by (model_name) (rate(vllm:request_success_total[1m]))",
                "sum by (model_name) (rate(vllm_requests_total[1m]))",
            ],
            description="Requests per second",
        ),
        MetricMapping(
            canonical_field="inference.tokens_per_second",
            queries=[
                "sum by (model_name) (rate(vllm:generation_tokens_total[1m]))",
            ],
            description="Token generation throughput",
        ),
        MetricMapping(
            canonical_field="inference.ttft_p50_ms",
            queries=[
                "histogram_quantile(0.5, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name)) * 1000",
            ],
            description="Time-to-first-token p50 in ms",
        ),
        MetricMapping(
            canonical_field="inference.ttft_p95_ms",
            queries=[
                "histogram_quantile(0.95, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name)) * 1000",
            ],
            description="Time-to-first-token p95 in ms",
        ),
        MetricMapping(
            canonical_field="inference.ttft_p99_ms",
            queries=[
                "histogram_quantile(0.99, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name)) * 1000",
            ],
            description="Time-to-first-token p99 in ms",
        ),
        MetricMapping(
            canonical_field="inference.latency_p95_ms",
            queries=[
                "histogram_quantile(0.95, sum(rate(vllm:request_generation_tokens_bucket[5m])) by (le, model_name)) * 1000",
                "histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le, model_name)) * 1000",
            ],
            description="End-to-end request latency p95 in ms",
        ),
        MetricMapping(
            canonical_field="inference.latency_p99_ms",
            queries=[
                "histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le, model_name)) * 1000",
            ],
            description="End-to-end request latency p99 in ms",
        ),
        MetricMapping(
            canonical_field="inference.queue_depth",
            queries=[
                "sum by (model_name) (vllm:num_requests_waiting)",
            ],
            description="Number of requests waiting in queue",
        ),
        MetricMapping(
            canonical_field="inference.active_sequences",
            queries=[
                "sum by (model_name) (vllm:num_requests_running)",
            ],
            description="Active running sequences",
        ),
        MetricMapping(
            canonical_field="inference.kv_cache_usage_pct",
            queries=[
                "avg by (model_name) (vllm:gpu_cache_usage_perc * 100)",
                "avg by (model_name) (vllm:gpu_cache_usage_perc)",
            ],
            description="KV cache usage percentage",
        ),
        MetricMapping(
            canonical_field="inference.prefix_cache_hit_rate_pct",
            queries=[
                "avg by (model_name) (vllm:prefix_cache_hit_rate * 100)",
            ],
            description="Prefix cache hit rate percentage",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Triton default mappings
# ---------------------------------------------------------------------------

DEFAULT_TRITON_MAPPINGS = MappingRegistry(
    name="triton",
    mappings=[
        MetricMapping(
            canonical_field="inference.requests_per_second",
            queries=[
                "sum by (model) (rate(nv_inference_request_success[1m]))",
            ],
            description="Triton inference requests per second",
        ),
        MetricMapping(
            canonical_field="inference.latency_p95_ms",
            queries=[
                "histogram_quantile(0.95, sum(rate(nv_inference_request_duration_us_bucket[5m])) by (le, model)) / 1000",
            ],
            description="Inference duration p95 in ms (from microseconds)",
        ),
        MetricMapping(
            canonical_field="inference.latency_p99_ms",
            queries=[
                "histogram_quantile(0.99, sum(rate(nv_inference_request_duration_us_bucket[5m])) by (le, model)) / 1000",
            ],
            description="Inference duration p99 in ms",
        ),
        MetricMapping(
            canonical_field="inference.queue_depth",
            queries=[
                "sum by (model) (nv_inference_pending_request_count)",
            ],
            description="Pending requests",
        ),
        MetricMapping(
            canonical_field="inference.queue_wait_p95_ms",
            queries=[
                "histogram_quantile(0.95, sum(rate(nv_inference_queue_duration_us_bucket[5m])) by (le, model)) / 1000",
            ],
            description="Queue wait time p95 in ms",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Ray Serve default mappings
# ---------------------------------------------------------------------------

DEFAULT_RAY_MAPPINGS = MappingRegistry(
    name="ray_serve",
    mappings=[
        MetricMapping(
            canonical_field="inference.requests_per_second",
            queries=[
                "sum by (deployment) (rate(ray_serve_num_router_requests_total[1m]))",
            ],
            description="Ray Serve request rate",
        ),
        MetricMapping(
            canonical_field="inference.latency_p95_ms",
            queries=[
                "histogram_quantile(0.95, sum(rate(ray_serve_deployment_processing_latency_ms_bucket[5m])) by (le, deployment))",
            ],
            description="Ray Serve processing latency p95 in ms",
        ),
        MetricMapping(
            canonical_field="inference.latency_p99_ms",
            queries=[
                "histogram_quantile(0.99, sum(rate(ray_serve_deployment_processing_latency_ms_bucket[5m])) by (le, deployment))",
            ],
            description="Ray Serve processing latency p99 in ms",
        ),
        MetricMapping(
            canonical_field="inference.queue_depth",
            queries=[
                "sum by (deployment) (ray_serve_deployment_queued_requests)",
            ],
            description="Queued requests",
        ),
        MetricMapping(
            canonical_field="inference.error_rate_pct",
            queries=[
                "100 * sum by (deployment) (rate(ray_serve_num_deployment_errors_total[1m])) / (sum by (deployment) (rate(ray_serve_num_router_requests_total[1m])) + 0.001)",
            ],
            description="Error rate percentage",
        ),
    ],
)
