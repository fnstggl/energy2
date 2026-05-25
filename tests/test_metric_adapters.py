"""Tests for DCGM, vLLM, Triton, Ray Serve, and OTel adapters — Phase 3.

All tests use fixture files, no live services required.
"""

import pathlib
import pytest
from datetime import timezone

from aurelius.connectors.adapters.dcgm import DCGMAdapter
from aurelius.connectors.adapters.vllm import VLLMAdapter
from aurelius.connectors.adapters.triton import TritonAdapter
from aurelius.connectors.adapters.ray_serve import RayServeAdapter
from aurelius.connectors.adapters.otel import OTelAdapter

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "prometheus"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# DCGMAdapter
# ---------------------------------------------------------------------------

class TestDCGMAdapter:
    def test_parses_dcgm_fixture(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text, node_id="node-1")
        assert len(result.gpus) == 2
        assert len(result.parse_errors) == 0

    def test_gpu_fields_populated(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text, node_id="node-1")
        gpu = result.gpus.get("GPU-abc123")
        assert gpu is not None
        assert gpu.utilization_pct == pytest.approx(75.5)
        assert gpu.power_watts == pytest.approx(380.5)
        assert gpu.temperature_c == pytest.approx(72.0)

    def test_xid_errors_zero_not_fabricated(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        # xid_error_count = 0 is explicitly set in fixture, not fabricated
        assert gpu.xid_error_count == 0

    def test_sm_activity_pct_fraction_to_percent_conversion(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        # 0.72 * 100 = 72.0
        assert gpu.sm_activity_pct == pytest.approx(72.0)

    def test_memory_bytes_mib_conversion(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        # 40960 MiB * 1024 * 1024 = 42949672960 bytes
        assert gpu.memory_used_bytes == 40960 * 1024 * 1024

    def test_memory_total_derived_from_used_plus_free(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        # used=40960MiB free=40960MiB total=81920MiB
        assert gpu.memory_total_bytes == (40960 + 40960) * 1024 * 1024

    def test_memory_bandwidth_pct_conversion(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        # 0.45 * 100 = 45.0
        assert gpu.memory_bandwidth_util_pct == pytest.approx(45.0)

    def test_nvlink_bytes_populated(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-abc123")
        assert gpu.nvlink_tx_bytes_per_sec is not None
        assert gpu.nvlink_rx_bytes_per_sec is not None

    def test_node_id_attached(self):
        adapter = DCGMAdapter()
        text = load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text, node_id="test-node")
        for gpu in result.gpus.values():
            assert gpu.node_id == "test-node"

    def test_unknown_metric_recorded(self):
        adapter = DCGMAdapter()
        text = "CUSTOM_METRIC_XYZ 42.0\n" + load_fixture("dcgm_metrics.txt")
        result = adapter.parse_text(text)
        assert "CUSTOM_METRIC_XYZ" in result.unknown_metrics

    def test_missing_metric_does_not_crash(self):
        adapter = DCGMAdapter()
        # Minimal text with only GPU utilization
        text = 'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-xyz",node="n1"} 50.0\n'
        result = adapter.parse_text(text)
        assert "GPU-xyz" in result.gpus
        gpu = result.gpus["GPU-xyz"]
        assert gpu.power_watts is None
        assert gpu.temperature_c is None
        assert len(result.parse_errors) == 0

    def test_empty_metrics_text(self):
        adapter = DCGMAdapter()
        result = adapter.parse_text("")
        assert len(result.gpus) == 0

    def test_comment_lines_ignored(self):
        adapter = DCGMAdapter()
        text = (
            "# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization\n"
            "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
            'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc",node="n1"} 60.0\n'
        )
        result = adapter.parse_text(text)
        assert "GPU-abc" in result.gpus

    def test_malformed_line_skipped(self):
        adapter = DCGMAdapter()
        text = (
            'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-ok",node="n1"} 60.0\n'
            "this_is_not_valid_prometheus_format\n"
        )
        result = adapter.parse_text(text)
        assert "GPU-ok" in result.gpus

    def test_labels_map_workload_and_node(self):
        adapter = DCGMAdapter()
        text = 'DCGM_FI_DEV_GPU_UTIL{gpu="2",UUID="GPU-node2",node="gpu-node-2"} 80.0\n'
        result = adapter.parse_text(text)
        gpu = result.gpus.get("GPU-node2")
        assert gpu.node_id is None  # not injected via parse_text without node_id param
        result2 = adapter.parse_text(text, node_id="gpu-node-2")
        assert result2.gpus["GPU-node2"].node_id == "gpu-node-2"


# ---------------------------------------------------------------------------
# VLLMAdapter
# ---------------------------------------------------------------------------

class TestVLLMAdapter:
    def test_parses_vllm_fixture(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        assert len(result.services) >= 1
        assert len(result.parse_errors) == 0

    def test_kv_cache_usage_populated(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        svc_key = next(iter(result.services))
        svc = result.services[svc_key]
        # 0.68 → 68.0
        assert svc.kv_cache_usage_pct == pytest.approx(68.0)

    def test_prefix_cache_hit_rate_populated(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        svc_key = next(iter(result.services))
        svc = result.services[svc_key]
        # 0.45 → 45.0
        assert svc.prefix_cache_hit_rate_pct == pytest.approx(45.0)

    def test_queue_depth_populated(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        svc = next(iter(result.services.values()))
        assert svc.queue_depth == 12

    def test_active_sequences_populated(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        svc = next(iter(result.services.values()))
        assert svc.active_sequences == 8

    def test_tokens_per_second_populated(self):
        adapter = VLLMAdapter()
        text = load_fixture("vllm_metrics.txt")
        result = adapter.parse_text(text)
        svc = next(iter(result.services.values()))
        assert svc.tokens_per_second == pytest.approx(3200.5)

    def test_missing_optional_metrics_none_not_zero(self):
        adapter = VLLMAdapter()
        text = 'vllm:num_requests_running{model_name="test-model"} 5\n'
        result = adapter.parse_text(text)
        svc = result.services.get("test-model")
        assert svc is not None
        assert svc.active_sequences == 5
        assert svc.kv_cache_usage_pct is None
        assert svc.ttft_p50_ms is None

    def test_runtime_set_to_vllm(self):
        adapter = VLLMAdapter()
        text = 'vllm:num_requests_running{model_name="test-model"} 3\n'
        result = adapter.parse_text(text)
        svc = result.services.get("test-model")
        from aurelius.state.models import RuntimeType
        assert svc.runtime == RuntimeType.VLLM

    def test_unknown_metrics_recorded(self):
        adapter = VLLMAdapter()
        text = (
            'vllm:num_requests_running{model_name="m"} 5\n'
            'custom_nonstandard_metric{model_name="m"} 1.0\n'
        )
        result = adapter.parse_text(text)
        assert "custom_nonstandard_metric" in result.unknown_metrics

    def test_empty_text_returns_empty_result(self):
        adapter = VLLMAdapter()
        result = adapter.parse_text("")
        assert len(result.services) == 0

    def test_model_name_label_as_service_key(self):
        adapter = VLLMAdapter()
        text = 'vllm:num_requests_running{model_name="meta-llama/Llama-2-7b"} 4\n'
        result = adapter.parse_text(text)
        assert "meta-llama/Llama-2-7b" in result.services


# ---------------------------------------------------------------------------
# TritonAdapter
# ---------------------------------------------------------------------------

class TestTritonAdapter:
    def test_parses_triton_fixture(self):
        adapter = TritonAdapter()
        text = load_fixture("triton_metrics.txt")
        result = adapter.parse_text(text)
        assert len(result.services) >= 1

    def test_queue_depth_populated(self):
        adapter = TritonAdapter()
        text = load_fixture("triton_metrics.txt")
        result = adapter.parse_text(text)
        svc = next(iter(result.services.values()))
        assert svc.queue_depth == 5

    def test_latency_derived_from_histogram(self):
        adapter = TritonAdapter()
        text = load_fixture("triton_metrics.txt")
        result = adapter.parse_text(text)
        svc = next(iter(result.services.values()))
        # sum=6500000us, count=1000 → mean=6500us → 6.5ms
        assert svc.latency_p50_ms == pytest.approx(6.5)

    def test_model_label_as_service_key(self):
        adapter = TritonAdapter()
        text = 'nv_inference_pending_request_count{model="resnet50",version="1"} 3\n'
        result = adapter.parse_text(text)
        assert "resnet50" in result.services

    def test_missing_metrics_none_not_zero(self):
        adapter = TritonAdapter()
        text = 'nv_inference_pending_request_count{model="m1",version="1"} 3\n'
        result = adapter.parse_text(text)
        svc = result.services.get("m1")
        assert svc.ttft_p50_ms is None
        assert svc.tokens_per_second is None

    def test_unknown_metrics_recorded(self):
        adapter = TritonAdapter()
        text = (
            'nv_inference_pending_request_count{model="m1",version="1"} 3\n'
            'custom_triton_metric{model="m1"} 1.0\n'
        )
        result = adapter.parse_text(text)
        assert "custom_triton_metric" in result.unknown_metrics

    def test_runtime_set_to_triton(self):
        adapter = TritonAdapter()
        text = 'nv_inference_pending_request_count{model="m1",version="1"} 1\n'
        result = adapter.parse_text(text)
        from aurelius.state.models import RuntimeType
        assert result.services["m1"].runtime == RuntimeType.TRITON


# ---------------------------------------------------------------------------
# RayServeAdapter
# ---------------------------------------------------------------------------

class TestRayServeAdapter:
    RAY_FIXTURE = (
        'ray_serve_deployment_queued_requests{deployment="my-llm"} 7\n'
        'ray_serve_num_ongoing_requests{deployment="my-llm"} 4\n'
        'ray_serve_num_router_requests_total{deployment="my-llm"} 1000\n'
        'ray_serve_num_deployment_errors_total{deployment="my-llm"} 5\n'
        'ray_serve_deployment_processing_latency_ms_sum{deployment="my-llm"} 50000\n'
        'ray_serve_deployment_processing_latency_ms_count{deployment="my-llm"} 1000\n'
    )

    def test_parses_ray_fixture(self):
        adapter = RayServeAdapter()
        result = adapter.parse_text(self.RAY_FIXTURE)
        assert "my-llm" in result.services

    def test_queue_depth_populated(self):
        adapter = RayServeAdapter()
        result = adapter.parse_text(self.RAY_FIXTURE)
        svc = result.services["my-llm"]
        assert svc.queue_depth == 7

    def test_active_sequences_populated(self):
        adapter = RayServeAdapter()
        result = adapter.parse_text(self.RAY_FIXTURE)
        svc = result.services["my-llm"]
        assert svc.active_sequences == 4

    def test_error_rate_derived_from_totals(self):
        adapter = RayServeAdapter()
        result = adapter.parse_text(self.RAY_FIXTURE)
        svc = result.services["my-llm"]
        # 5/1000 * 100 = 0.5%
        assert svc.error_rate_pct == pytest.approx(0.5)

    def test_latency_derived_from_sum_count(self):
        adapter = RayServeAdapter()
        result = adapter.parse_text(self.RAY_FIXTURE)
        svc = result.services["my-llm"]
        # sum=50000ms, count=1000 → mean=50ms
        assert svc.latency_p50_ms == pytest.approx(50.0)

    def test_missing_optional_metrics_none_not_zero(self):
        adapter = RayServeAdapter()
        text = 'ray_serve_deployment_queued_requests{deployment="dep-1"} 3\n'
        result = adapter.parse_text(text)
        svc = result.services.get("dep-1")
        assert svc.ttft_p50_ms is None
        assert svc.tokens_per_second is None
        assert svc.kv_cache_usage_pct is None

    def test_runtime_set_to_ray_serve(self):
        adapter = RayServeAdapter()
        text = 'ray_serve_deployment_queued_requests{deployment="dep-1"} 1\n'
        result = adapter.parse_text(text)
        from aurelius.state.models import RuntimeType
        assert result.services["dep-1"].runtime == RuntimeType.RAY_SERVE

    def test_deployment_label_as_service_key(self):
        adapter = RayServeAdapter()
        text = 'ray_serve_num_ongoing_requests{deployment="classifier-v2"} 2\n'
        result = adapter.parse_text(text)
        assert "classifier-v2" in result.services


# ---------------------------------------------------------------------------
# OTelAdapter
# ---------------------------------------------------------------------------

class TestOTelAdapter:
    def test_parse_flat_dict_minimal(self):
        adapter = OTelAdapter()
        result = adapter.parse_flat_dict({
            "service_id": "my-service",
            "runtime": "vllm",
            "inference.request_rate": 42.0,
        })
        assert "my-service" in result.services
        svc = result.services["my-service"]
        assert svc.requests_per_second == 42.0

    def test_parse_flat_dict_unknown_fields_recorded(self):
        adapter = OTelAdapter()
        result = adapter.parse_flat_dict({
            "service_id": "s1",
            "unknown.custom.metric": 1.0,
        })
        assert "unknown.custom.metric" in result.unknown_metrics

    def test_parse_flat_dict_missing_metrics_none_not_zero(self):
        adapter = OTelAdapter()
        result = adapter.parse_flat_dict({"service_id": "s1"})
        svc = result.services["s1"]
        assert svc.requests_per_second is None
        assert svc.kv_cache_usage_pct is None

    def test_parse_otlp_json(self):
        adapter = OTelAdapter()
        payload = {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "llm-service"}}
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "inference.request_rate",
                                    "gauge": {
                                        "dataPoints": [
                                            {"asDouble": 55.0, "timeUnixNano": "1705320000000000000"}
                                        ]
                                    },
                                },
                                {
                                    "name": "inference.ttft_p95",
                                    "gauge": {
                                        "dataPoints": [
                                            {"asDouble": 180.0}
                                        ]
                                    },
                                },
                            ]
                        }
                    ],
                }
            ]
        }
        result = adapter.parse_otlp_json(payload)
        assert "llm-service" in result.services
        svc = result.services["llm-service"]
        assert svc.requests_per_second == 55.0
        assert svc.ttft_p95_ms == 180.0

    def test_otlp_unknown_metric_recorded(self):
        adapter = OTelAdapter()
        payload = {
            "resourceMetrics": [
                {
                    "resource": {"attributes": [
                        {"key": "service.name", "value": {"stringValue": "svc"}}
                    ]},
                    "scopeMetrics": [{
                        "metrics": [{"name": "custom.nonstandard.metric", "gauge": {"dataPoints": []}}]
                    }],
                }
            ]
        }
        result = adapter.parse_otlp_json(payload)
        assert "custom.nonstandard.metric" in result.unknown_metrics
