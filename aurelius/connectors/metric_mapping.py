"""Metric mapping layer for Aurelius Prometheus connectors.

Maps canonical Aurelius field names to Prometheus query expressions,
with support for fallback queries, unit conversions, and label extraction.

This layer lets deployments customize metric names without touching adapter
code. A mapping file might look like:

    gpu.util_pct:
      query: "DCGM_FI_DEV_GPU_UTIL"
      fallback_queries:
        - "dcgm_fi_dev_gpu_util"
      unit: pct
      labels: [gpu, node, UUID]

    inference.ttft_p95_ms:
      query: >
        histogram_quantile(0.95,
          sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name))
      unit: seconds_to_ms
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class UnitConversion:
    """Supported unit conversions from Prometheus raw values."""
    NONE = "none"
    PCT = "pct"                   # already a percentage, no conversion
    RATIO_TO_PCT = "ratio_to_pct" # multiply by 100
    SECONDS_TO_MS = "seconds_to_ms"
    MS_TO_MS = "ms_to_ms"        # already ms
    BYTES = "bytes"               # already bytes
    MB_TO_BYTES = "mb_to_bytes"  # multiply by 1_048_576
    WATTS = "watts"               # already watts

    _FACTORS: dict[str, float] = {
        NONE: 1.0,
        PCT: 1.0,
        RATIO_TO_PCT: 100.0,
        SECONDS_TO_MS: 1_000.0,
        MS_TO_MS: 1.0,
        BYTES: 1.0,
        MB_TO_BYTES: 1_048_576.0,
        WATTS: 1.0,
    }

    @classmethod
    def apply(cls, value: Optional[float], unit: str) -> Optional[float]:
        if value is None:
            return None
        factor = cls._FACTORS.get(unit, 1.0)
        return value * factor


@dataclass
class MetricMapping:
    """Mapping from a canonical Aurelius field name to Prometheus queries.

    Attributes:
        canonical_field:   Aurelius field name, e.g. "gpu.util_pct"
        query:             Primary PromQL query or metric name
        fallback_queries:  Tried in order if primary returns no data
        unit:              Unit conversion (see UnitConversion)
        labels_to_keep:    Label keys to preserve in MetricValue.labels
        description:       Human-readable description
    """
    canonical_field: str
    query: str
    fallback_queries: list[str] = field(default_factory=list)
    unit: str = UnitConversion.NONE
    labels_to_keep: list[str] = field(default_factory=list)
    description: str = ""

    def convert(self, value: Optional[float]) -> Optional[float]:
        return UnitConversion.apply(value, self.unit)

    @classmethod
    def from_dict(cls, canonical_field: str, d: dict[str, Any]) -> "MetricMapping":
        return cls(
            canonical_field=canonical_field,
            query=d["query"],
            fallback_queries=d.get("fallback_queries", []),
            unit=d.get("unit", UnitConversion.NONE),
            labels_to_keep=d.get("labels", []),
            description=d.get("description", ""),
        )


class MetricMappingRegistry:
    """Registry of canonical_field → MetricMapping for a connector type."""

    def __init__(self, mappings: dict[str, MetricMapping]) -> None:
        self._mappings = mappings

    def get(self, canonical_field: str) -> Optional[MetricMapping]:
        return self._mappings.get(canonical_field)

    def all_fields(self) -> list[str]:
        return list(self._mappings.keys())

    def all_mappings(self) -> list[MetricMapping]:
        return list(self._mappings.values())

    def __len__(self) -> int:
        return len(self._mappings)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricMappingRegistry":
        mappings = {}
        for field_name, field_def in d.items():
            if isinstance(field_def, str):
                # shorthand: just the query string
                field_def = {"query": field_def}
            mappings[field_name] = MetricMapping.from_dict(field_name, field_def)
        return cls(mappings)

    @classmethod
    def empty(cls) -> "MetricMappingRegistry":
        return cls({})


def load_mapping_yaml(path: str) -> MetricMappingRegistry:
    """Load a MetricMappingRegistry from a YAML file."""
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required to load mapping YAML files")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metric mapping file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metric mapping file must be a YAML dict: {path}")
    return MetricMappingRegistry.from_dict(data)


def load_mapping_dict(d: dict[str, Any]) -> MetricMappingRegistry:
    """Load a MetricMappingRegistry from a dict (for inline/programmatic use)."""
    return MetricMappingRegistry.from_dict(d)


# ---------------------------------------------------------------------------
# Built-in registries (can be overridden by user-provided YAML)
# ---------------------------------------------------------------------------

_DCGM_BUILTIN: dict[str, Any] = {
    "gpu.util_pct": {
        "query": "DCGM_FI_DEV_GPU_UTIL",
        "unit": "pct",
        "labels": ["gpu", "node", "UUID", "modelName"],
        "description": "GPU utilization %",
    },
    "gpu.sm_active_ratio": {
        "query": "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
        "fallback_queries": [],
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
        "description": "SM active ratio 0-1",
    },
    "gpu.tensor_active_ratio": {
        "query": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.dram_active_ratio": {
        "query": "DCGM_FI_PROF_DRAM_ACTIVE",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.mem_used_mb": {
        "query": "DCGM_FI_DEV_FB_USED",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
        "description": "GPU frame buffer used (MiB)",
    },
    "gpu.mem_free_mb": {
        "query": "DCGM_FI_DEV_FB_FREE",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
        "description": "GPU frame buffer free (MiB)",
    },
    "gpu.mem_reserved_mb": {
        "query": "DCGM_FI_DEV_FB_RESERVED",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.power_w": {
        "query": "DCGM_FI_DEV_POWER_USAGE",
        "unit": "watts",
        "labels": ["gpu", "node", "UUID"],
        "description": "GPU power consumption (W)",
    },
    "gpu.power_limit_w": {
        "query": "DCGM_FI_DEV_POWER_MGMT_LIMIT",
        "unit": "watts",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.temp_c": {
        "query": "DCGM_FI_DEV_GPU_TEMP",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
        "description": "GPU temperature (°C)",
    },
    "gpu.mem_temp_c": {
        "query": "DCGM_FI_DEV_MEMORY_TEMP",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.sm_clock_mhz": {
        "query": "DCGM_FI_DEV_SM_CLOCK",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.clocks_event_reasons": {
        "query": "DCGM_FI_DEV_CLOCKS_EVENT_REASONS",
        "fallback_queries": ["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"],
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
        "description": "Clock throttle reason bitmask",
    },
    "gpu.power_violation_ns": {
        "query": "DCGM_FI_DEV_POWER_VIOLATION",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.thermal_violation_ns": {
        "query": "DCGM_FI_DEV_THERMAL_VIOLATION",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.ecc_sbe_total": {
        "query": "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.ecc_dbe_total": {
        "query": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.xid_last": {
        "query": "DCGM_FI_DEV_XID_ERRORS",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.xid_count_window": {
        "query": "DCGM_EXP_XID_ERRORS_COUNT",
        "unit": "none",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.pcie_tx_bytes_per_s": {
        "query": "DCGM_FI_PROF_PCIE_TX_BYTES",
        "unit": "bytes",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.pcie_rx_bytes_per_s": {
        "query": "DCGM_FI_PROF_PCIE_RX_BYTES",
        "unit": "bytes",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.nvlink_tx_bytes_per_s": {
        "query": "DCGM_FI_PROF_NVLINK_TX_BYTES",
        "unit": "bytes",
        "labels": ["gpu", "node", "UUID"],
    },
    "gpu.nvlink_rx_bytes_per_s": {
        "query": "DCGM_FI_PROF_NVLINK_RX_BYTES",
        "unit": "bytes",
        "labels": ["gpu", "node", "UUID"],
    },
}

_VLLM_BUILTIN: dict[str, Any] = {
    "inference.requests_per_second": {
        "query": "rate(vllm:request_success_total[1m])",
        "fallback_queries": ["rate(vllm_request_success_total[1m])"],
        "unit": "none",
        "labels": ["model_name"],
        "description": "Requests per second",
    },
    "inference.tokens_per_second": {
        "query": "rate(vllm:generation_tokens_total[1m])",
        "fallback_queries": ["rate(vllm_generation_tokens_total[1m])"],
        "unit": "none",
        "labels": ["model_name"],
    },
    "inference.ttft_p50_ms": {
        "query": "histogram_quantile(0.50, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
        "description": "Time to first token p50 (ms)",
    },
    "inference.ttft_p95_ms": {
        "query": "histogram_quantile(0.95, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.ttft_p99_ms": {
        "query": "histogram_quantile(0.99, sum(rate(vllm:request_first_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.tpot_p50_ms": {
        "query": "histogram_quantile(0.50, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
        "description": "Time per output token p50 (ms)",
    },
    "inference.tpot_p95_ms": {
        "query": "histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.tpot_p99_ms": {
        "query": "histogram_quantile(0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.e2e_p50_ms": {
        "query": "histogram_quantile(0.50, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.e2e_p95_ms": {
        "query": "histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.e2e_p99_ms": {
        "query": "histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le, model_name))",
        "unit": "seconds_to_ms",
        "labels": ["model_name"],
    },
    "inference.queue_depth": {
        "query": "vllm:num_requests_waiting",
        "fallback_queries": ["vllm_num_requests_waiting"],
        "unit": "none",
        "labels": ["model_name"],
    },
    "inference.active_sequences": {
        "query": "vllm:num_requests_running",
        "fallback_queries": ["vllm_num_requests_running"],
        "unit": "none",
        "labels": ["model_name"],
    },
    "inference.kv_cache_usage_pct": {
        "query": "vllm:gpu_cache_usage_perc * 100",
        "fallback_queries": ["vllm_gpu_cache_usage_perc * 100"],
        "unit": "pct",
        "labels": ["model_name"],
        "description": "KV cache usage %",
    },
    "inference.prefix_cache_hit_rate_pct": {
        "query": "vllm:gpu_prefix_cache_hit_rate * 100",
        "fallback_queries": ["vllm_gpu_prefix_cache_hit_rate * 100"],
        "unit": "pct",
        "labels": ["model_name"],
    },
    # KV-pressure / preemption rate. vLLM emits this as a cumulative
    # counter; the rate is the per-second preemption count. Used by the
    # bridge as an input to risk diagnostics, NOT to ``timeout_pct`` —
    # preemptions are restarts, not SLA timeouts.
    "inference.preemptions_per_second": {
        "query": "rate(vllm:num_preemptions_total[1m])",
        "fallback_queries": ["rate(vllm_num_preemptions_total[1m])"],
        "unit": "none",
        "labels": ["model_name"],
        "description": (
            "vLLM preemption rate — KV-cache-pressure signal; NOT a "
            "timeout counter."),
    },
}

_TRITON_BUILTIN: dict[str, Any] = {
    "triton.inference_count": {
        "query": "nv_inference_count",
        "unit": "none",
        "labels": ["model", "version"],
        "description": "Triton inference request count",
    },
    "triton.inference_exec_count": {
        "query": "nv_inference_exec_count",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.inference_queue_duration_us": {
        "query": "nv_inference_queue_duration_us",
        "unit": "none",
        "labels": ["model", "version"],
        "description": "Triton inference queue duration (μs)",
    },
    "triton.inference_compute_duration_us": {
        "query": "nv_inference_compute_infer_duration_us",
        "fallback_queries": ["nv_inference_compute_duration_us"],
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.inference_input_duration_us": {
        "query": "nv_inference_compute_input_duration_us",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.inference_output_duration_us": {
        "query": "nv_inference_compute_output_duration_us",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.request_success": {
        "query": "nv_inference_request_success",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.request_failure": {
        "query": "nv_inference_request_failure",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.pending_request_count": {
        "query": "nv_inference_pending_request_count",
        "unit": "none",
        "labels": ["model", "version"],
    },
    "triton.gpu_utilization_pct": {
        "query": "nv_gpu_utilization",
        "unit": "pct",
        "labels": ["gpu_uuid"],
    },
    "triton.gpu_memory_used_bytes": {
        "query": "nv_gpu_memory_used_bytes",
        "unit": "bytes",
        "labels": ["gpu_uuid"],
    },
    "triton.gpu_memory_total_bytes": {
        "query": "nv_gpu_memory_total_bytes",
        "unit": "bytes",
        "labels": ["gpu_uuid"],
    },
}

_RAY_SERVE_BUILTIN: dict[str, Any] = {
    "ray.serve.num_replicas": {
        "query": "sum by (deployment) (ray_serve_deployment_replica_count)",
        "fallback_queries": ["ray_serve_deployment_replica_count"],
        "unit": "none",
        "labels": ["deployment"],
        "description": "Ray Serve replica count",
    },
    "ray.serve.requests_per_second": {
        "query": "rate(ray_serve_num_ongoing_requests_total[1m])",
        "fallback_queries": [
            "ray_serve_num_ongoing_requests_total",
            "rate(ray_serve_num_http_requests_total[1m])",
        ],
        "unit": "none",
        "labels": ["deployment"],
    },
    "ray.serve.queue_len": {
        "query": "ray_serve_deployment_queued_queries",
        "unit": "none",
        "labels": ["deployment"],
    },
    "ray.serve.request_latency_p50_ms": {
        "query": "histogram_quantile(0.50, sum(rate(ray_serve_request_latency_ms_bucket[5m])) by (le, deployment))",
        "fallback_queries": ["ray_serve_request_latency_ms_bucket"],
        "unit": "ms_to_ms",
        "labels": ["deployment"],
    },
    "ray.serve.request_latency_p95_ms": {
        "query": "histogram_quantile(0.95, sum(rate(ray_serve_request_latency_ms_bucket[5m])) by (le, deployment))",
        "fallback_queries": ["ray_serve_request_latency_ms_bucket"],
        "unit": "ms_to_ms",
        "labels": ["deployment"],
    },
    "ray.serve.request_latency_p99_ms": {
        "query": "histogram_quantile(0.99, sum(rate(ray_serve_request_latency_ms_bucket[5m])) by (le, deployment))",
        "fallback_queries": ["ray_serve_request_latency_ms_bucket"],
        "unit": "ms_to_ms",
        "labels": ["deployment"],
    },
    "ray.serve.error_rate_pct": {
        "query": "100 * rate(ray_serve_num_http_error_requests_total[1m]) / (rate(ray_serve_num_http_requests_total[1m]) + 0.0001)",
        "fallback_queries": ["ray_serve_num_http_error_requests_total"],
        "unit": "pct",
        "labels": ["deployment"],
    },
}


def dcgm_registry() -> MetricMappingRegistry:
    """Return the built-in DCGM metric mapping registry."""
    return MetricMappingRegistry.from_dict(_DCGM_BUILTIN)


def vllm_registry() -> MetricMappingRegistry:
    """Return the built-in vLLM metric mapping registry."""
    return MetricMappingRegistry.from_dict(_VLLM_BUILTIN)


def triton_registry() -> MetricMappingRegistry:
    """Return the built-in Triton metric mapping registry."""
    return MetricMappingRegistry.from_dict(_TRITON_BUILTIN)


def ray_serve_registry() -> MetricMappingRegistry:
    """Return the built-in Ray Serve metric mapping registry."""
    return MetricMappingRegistry.from_dict(_RAY_SERVE_BUILTIN)
