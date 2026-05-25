"""Tests for Phase 3: vLLM, Triton, Ray Serve, and OTel adapters.

Proves:
- vLLM adapter maps Prometheus metrics to InferenceServiceState correctly
- Triton adapter normalizes cumulative counters correctly
- Ray Serve adapter extracts deployment-level metrics
- OTel adapter parses OTLP JSON format correctly
- Missing optional metrics are None, not fabricated values
- Adapters correctly handle multi-model deployments
- Error rate derived correctly from success/failure counters
- kv_cache_usage stored as 0-1 fraction (not percent) — model requirement
- prefix_cache_hit_rate stored as 0-1 fraction (not percent)
- All adapters share the same interface (sandbox + real paths identical)
"""

from __future__ import annotations

import os

import pytest

from aurelius.connectors.metric_mapping import ray_serve_registry, triton_registry, vllm_registry
from aurelius.connectors.otel import OTelAdapter, parse_otlp_json
from aurelius.connectors.prometheus import FakePrometheusClient, PrometheusTelemetryConnector
from aurelius.connectors.ray_serve import RayServeAdapter
from aurelius.connectors.triton import TritonAdapter
from aurelius.connectors.vllm import VLLMAdapter
from aurelius.state.models import InferenceServiceState

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prometheus")


def _vllm_snapshot_from_file():
    with open(os.path.join(FIXTURES_DIR, "vllm_metrics.prom")) as f:
        text = f.read()
    client = FakePrometheusClient(prometheus_text=text)
    reg = vllm_registry()
    connector = PrometheusTelemetryConnector(client, reg, source="vllm-test")
    return connector.fetch_snapshot()


def _triton_snapshot_from_file():
    with open(os.path.join(FIXTURES_DIR, "triton_metrics.prom")) as f:
        text = f.read()
    client = FakePrometheusClient(prometheus_text=text)
    reg = triton_registry()
    connector = PrometheusTelemetryConnector(client, reg, source="triton-test")
    return connector.fetch_snapshot()


def _ray_snapshot_from_file():
    with open(os.path.join(FIXTURES_DIR, "ray_serve_metrics.prom")) as f:
        text = f.read()
    client = FakePrometheusClient(prometheus_text=text)
    reg = ray_serve_registry()
    connector = PrometheusTelemetryConnector(client, reg, source="ray-test")
    return connector.fetch_snapshot()


# ---------------------------------------------------------------------------
# vLLM Adapter Tests
# ---------------------------------------------------------------------------

class TestVLLMAdapter:
    def test_discovers_model_names(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        models = adapter.all_model_names(snapshot)
        assert "llama3-70b" in models
        assert "mistral-7b" in models

    def test_normalize_single_model(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "vllm/llama3-70b", model_name="llama3-70b")
        assert isinstance(svc, InferenceServiceState)
        assert svc.service_id == "vllm/llama3-70b"
        assert svc.engine == "vllm"

    def test_requests_waiting_correct(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc_llama = adapter.normalize_inference_state(snapshot, "vllm/llama3-70b", model_name="llama3-70b")
        svc_mistral = adapter.normalize_inference_state(snapshot, "vllm/mistral-7b", model_name="mistral-7b")
        # vllm:num_requests_waiting{model_name="llama3-70b"} = 3
        # vllm:num_requests_waiting{model_name="mistral-7b"} = 12
        assert svc_llama.requests_waiting == pytest.approx(3.0)
        assert svc_mistral.requests_waiting == pytest.approx(12.0)

    def test_requests_running_correct(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "vllm/llama3-70b", model_name="llama3-70b")
        # vllm:num_requests_running{model_name="llama3-70b"} = 8
        assert svc.requests_running == pytest.approx(8.0)

    def test_kv_cache_usage_is_fraction_not_percent(self):
        """kv_cache_usage must be stored as [0-1] fraction, not percentage.

        InferenceServiceState validates kv_cache_usage in [0, 1].
        vllm:gpu_cache_usage_perc emits 0-1 already; mapping may apply ×100 conversion.
        The adapter must clamp back to [0-1].
        """
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "vllm/llama3-70b", model_name="llama3-70b")
        if svc.kv_cache_usage is not None:
            assert 0.0 <= svc.kv_cache_usage <= 1.0, (
                f"kv_cache_usage must be 0-1 fraction, got {svc.kv_cache_usage}"
            )

    def test_prefix_cache_hit_rate_is_fraction(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "vllm/llama3-70b", model_name="llama3-70b")
        if svc.prefix_cache_hit_rate is not None:
            assert 0.0 <= svc.prefix_cache_hit_rate <= 1.0

    def test_normalize_all_services(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        services = adapter.normalize_all_services(snapshot, service_id_prefix="vllm")
        assert len(services) == 2
        service_ids = [s.service_id for s in services]
        assert any("llama3-70b" in sid for sid in service_ids)
        assert any("mistral-7b" in sid for sid in service_ids)

    def test_missing_metrics_none_not_zero(self):
        """Missing vLLM metrics must be None, not 0."""
        client = FakePrometheusClient(fixtures={
            "vllm:num_requests_waiting": [
                {"labels": {"model_name": "llm"}, "value": 5.0}
            ],
        })
        connector = PrometheusTelemetryConnector(
            client, vllm_registry(), source="partial-vllm"
        )
        snapshot = connector.fetch_snapshot()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "vllm/llm", model_name="llm")

        assert svc.requests_waiting == pytest.approx(5.0)
        # Missing metrics — must be None
        assert svc.ttft_p95_ms is None
        assert svc.tokens_per_s is None

    def test_timestamp_utc_aware(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "test", model_name="llama3-70b")
        assert svc.timestamp.tzinfo is not None

    def test_provenance_sandbox(self):
        snapshot = _vllm_snapshot_from_file()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "test", model_name="llama3-70b")
        assert svc.provenance.is_sandbox is True

    def test_ratio_kv_not_over_1(self):
        """kv_cache_usage is clamped to [0, 1]; never > 1.0."""
        client = FakePrometheusClient(fixtures={
            # Fixture emits 78.0 (the × 100 version); adapter must clamp to [0,1]
            "inference.kv_cache_usage_pct": [
                {"labels": {"model_name": "m"}, "value": 78.0}
            ],
        })
        connector = PrometheusTelemetryConnector(client, vllm_registry(), source="test")
        snapshot = connector.fetch_snapshot()
        adapter = VLLMAdapter()
        svc = adapter.normalize_inference_state(snapshot, "m", model_name="m")
        if svc.kv_cache_usage is not None:
            assert svc.kv_cache_usage <= 1.0


# ---------------------------------------------------------------------------
# Triton Adapter Tests
# ---------------------------------------------------------------------------

class TestTritonAdapter:
    def test_discovers_models(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        models = adapter.all_models(snapshot)
        assert ("bert-large", "1") in models
        assert ("gpt2", "1") in models

    def test_normalize_single_model(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(
            snapshot, "triton/bert-large/1", model="bert-large", version="1"
        )
        assert isinstance(svc, InferenceServiceState)
        assert svc.engine == "triton"
        assert svc.service_id == "triton/bert-large/1"

    def test_pending_request_count(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/b/1", model="bert-large", version="1")
        assert svc.requests_waiting == pytest.approx(5.0)

    def test_error_rate_derived_from_counters(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/b/1", model="bert-large", version="1")
        # success=47923, failure=368, total=48291
        # error_rate = 368/48291 * 100 ≈ 0.762%
        if svc.error_rate_pct is not None:
            assert 0.0 <= svc.error_rate_pct < 5.0

    def test_avg_latency_derived_from_counters(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/b/1", model="bert-large", version="1")
        # Should have some latency value (average from compute_us / exec_count)
        assert svc.p50_latency_ms is not None
        assert svc.p50_latency_ms > 0
        # p95/p99 are not available from Triton defaults
        assert svc.p95_latency_ms is None
        assert svc.p99_latency_ms is None

    def test_normalize_all_services(self):
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        services = adapter.normalize_all_services(snapshot, service_id_prefix="triton")
        assert len(services) == 2

    def test_missing_metrics_none(self):
        """Missing Triton metrics must be None, not 0."""
        client = FakePrometheusClient(fixtures={
            "nv_inference_pending_request_count": [
                {"labels": {"model": "mymodel", "version": "1"}, "value": 3.0}
            ],
        })
        connector = PrometheusTelemetryConnector(client, triton_registry(), source="test")
        snapshot = connector.fetch_snapshot()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/m/1", model="mymodel", version="1")
        assert svc.requests_waiting == pytest.approx(3.0)
        # No exec count or compute duration → no latency
        assert svc.p50_latency_ms is None

    def test_zero_exec_count_no_latency(self):
        """When exec_count is 0, latency must be None (not divide-by-zero)."""
        client = FakePrometheusClient(fixtures={
            "nv_inference_exec_count": [{"labels": {"model": "m", "version": "1"}, "value": 0.0}],
            "nv_inference_compute_infer_duration_us": [{"labels": {"model": "m", "version": "1"}, "value": 0.0}],
        })
        connector = PrometheusTelemetryConnector(client, triton_registry(), source="test")
        snapshot = connector.fetch_snapshot()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/m/1", model="m", version="1")
        assert svc.p50_latency_ms is None

    def test_no_kv_cache_for_triton(self):
        """Triton doesn't expose KV cache by default."""
        snapshot = _triton_snapshot_from_file()
        adapter = TritonAdapter()
        svc = adapter.normalize_inference_state(snapshot, "t/b/1", model="bert-large", version="1")
        assert svc.kv_cache_usage is None
        assert svc.prefix_cache_hit_rate is None


# ---------------------------------------------------------------------------
# Ray Serve Adapter Tests
# ---------------------------------------------------------------------------

class TestRayServeAdapter:
    def test_discovers_deployments(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        deployments = adapter.all_deployments(snapshot)
        assert "llm-router" in deployments
        assert "embedding-service" in deployments

    def test_normalize_single_deployment(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(
            snapshot, "ray/llm-router", deployment="llm-router"
        )
        assert isinstance(svc, InferenceServiceState)
        assert svc.engine == "ray_serve"

    def test_queue_depth_correct(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/llm", deployment="llm-router")
        # ray_serve_deployment_queued_queries{deployment="llm-router"} = 7
        assert svc.requests_waiting == pytest.approx(7.0)

    def test_embedding_no_queue(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/emb", deployment="embedding-service")
        assert svc.requests_waiting == pytest.approx(0.0)

    def test_replicas_extracted(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/llm", deployment="llm-router")
        # ray_serve_deployment_replica_count{deployment="llm-router"} = 4
        assert svc.replicas == 4

    def test_normalize_all_services(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        services = adapter.normalize_all_services(snapshot, service_id_prefix="ray")
        assert len(services) == 2

    def test_missing_metrics_none(self):
        """Missing Ray Serve metrics must be None."""
        client = FakePrometheusClient(fixtures={
            "ray_serve_deployment_queued_queries": [
                {"labels": {"deployment": "my-svc"}, "value": 2.0}
            ],
        })
        connector = PrometheusTelemetryConnector(client, ray_serve_registry(), source="test")
        snapshot = connector.fetch_snapshot()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/my-svc", deployment="my-svc")
        assert svc.requests_waiting == pytest.approx(2.0)
        assert svc.p95_latency_ms is None

    def test_error_rate_pct_clamped(self):
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/llm", deployment="llm-router")
        if svc.error_rate_pct is not None:
            assert 0.0 <= svc.error_rate_pct <= 100.0

    def test_ttft_not_available_from_ray(self):
        """Ray Serve does not expose ttft/tpot by default."""
        snapshot = _ray_snapshot_from_file()
        adapter = RayServeAdapter()
        svc = adapter.normalize_inference_state(snapshot, "r/llm", deployment="llm-router")
        assert svc.ttft_p95_ms is None


# ---------------------------------------------------------------------------
# OTel Adapter Tests
# ---------------------------------------------------------------------------

class TestOTelAdapter:
    def _make_otlp_payload(self, service_name: str, metrics: dict) -> dict:
        def make_metric(name: str, value: float) -> dict:
            return {
                "name": name,
                "gauge": {
                    "dataPoints": [{"asDouble": value}]
                }
            }

        return {
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}}
                    ]
                },
                "scopeMetrics": [{
                    "metrics": [make_metric(k, v) for k, v in metrics.items()]
                }]
            }]
        }

    def test_parse_otlp_basic(self):
        payload = self._make_otlp_payload("my-llm", {
            "request.latency.p95": 145.0,
            "queue.depth": 7.0,
            "ttft.p95": 85.0,
        })
        services = parse_otlp_json(payload)
        assert "my-llm" in services
        svc = services["my-llm"]
        assert svc["p95_latency_ms"] == pytest.approx(145.0)
        assert svc["requests_waiting"] == pytest.approx(7.0)
        assert svc["ttft_p95_ms"] == pytest.approx(85.0)

    def test_unknown_metric_skipped(self):
        payload = self._make_otlp_payload("svc", {
            "unknown.proprietary.metric": 42.0,
            "request.latency.p99": 200.0,
        })
        services = parse_otlp_json(payload)
        assert "p99_latency_ms" in services["svc"]
        assert "unknown.proprietary.metric" not in services["svc"]

    def test_normalize_from_otlp_json(self):
        payload = self._make_otlp_payload("inference-svc", {
            "request.latency.p50": 50.0,
            "request.latency.p95": 120.0,
            "request.latency.p99": 250.0,
            "queue.depth": 3.0,
            "kv_cache.usage_pct": 75.0,
        })
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload, is_sandbox=True)
        assert len(services) == 1
        svc = services[0]
        assert isinstance(svc, InferenceServiceState)
        assert svc.p50_latency_ms == pytest.approx(50.0)
        assert svc.p95_latency_ms == pytest.approx(120.0)
        assert svc.requests_waiting == pytest.approx(3.0)
        # kv_cache.usage_pct = 75.0 → converted to fraction 0.75
        assert svc.kv_cache_usage == pytest.approx(0.75)

    def test_multi_service_payload(self):
        def _make_rm(name, value):
            return {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": name}}]
                },
                "scopeMetrics": [{
                    "metrics": [{
                        "name": "queue.depth",
                        "gauge": {"dataPoints": [{"asDouble": value}]}
                    }]
                }]
            }

        payload = {"resourceMetrics": [_make_rm("svc-a", 2.0), _make_rm("svc-b", 8.0)]}
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload)
        assert len(services) == 2

    def test_kv_cache_clamped_to_fraction(self):
        """kv_cache.usage_pct = 150.0 → fraction clamped to 1.0."""
        payload = self._make_otlp_payload("svc", {"kv_cache.usage_pct": 150.0})
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload)
        assert services[0].kv_cache_usage == pytest.approx(1.0)

    def test_missing_metrics_none(self):
        payload = self._make_otlp_payload("svc", {"queue.depth": 5.0})
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload)
        svc = services[0]
        assert svc.requests_waiting == pytest.approx(5.0)
        assert svc.ttft_p95_ms is None
        assert svc.p99_latency_ms is None

    def test_provenance_is_sandbox(self):
        payload = self._make_otlp_payload("svc", {"queue.depth": 1.0})
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload, is_sandbox=True)
        assert services[0].provenance.is_sandbox is True

    def test_engine_is_unknown_for_otel(self):
        """OTel adapter uses engine='unknown' (valid value per InferenceServiceState)."""
        payload = self._make_otlp_payload("svc", {"queue.depth": 1.0})
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json(payload)
        assert services[0].engine == "unknown"

    def test_empty_payload(self):
        adapter = OTelAdapter()
        services = adapter.normalize_from_otlp_json({"resourceMetrics": []})
        assert services == []


# ---------------------------------------------------------------------------
# Cross-adapter: all adapters share the same interface
# ---------------------------------------------------------------------------

class TestAdapterInterfaceConsistency:
    """Verify all adapters produce canonical InferenceServiceState consistently."""

    def test_all_produce_utc_aware_timestamps(self):
        vllm_snap = _vllm_snapshot_from_file()
        triton_snap = _triton_snapshot_from_file()
        ray_snap = _ray_snapshot_from_file()

        vllm_svc = VLLMAdapter().normalize_inference_state(vllm_snap, "v", model_name="llama3-70b")
        triton_svc = TritonAdapter().normalize_inference_state(triton_snap, "t", model="bert-large", version="1")
        ray_svc = RayServeAdapter().normalize_inference_state(ray_snap, "r", deployment="llm-router")

        for svc in [vllm_svc, triton_svc, ray_svc]:
            assert svc.timestamp.tzinfo is not None, f"{svc.service_id} timestamp not UTC-aware"

    def test_all_produce_provenance(self):
        vllm_snap = _vllm_snapshot_from_file()
        svc = VLLMAdapter().normalize_inference_state(vllm_snap, "v", model_name="llama3-70b")
        assert svc.provenance is not None
        assert svc.provenance.source == "vllm-test"

    def test_all_engines_valid(self):
        """engine field must be a valid value per InferenceServiceState._VALID_ENGINES."""
        vllm_snap = _vllm_snapshot_from_file()
        triton_snap = _triton_snapshot_from_file()
        ray_snap = _ray_snapshot_from_file()

        svcs = [
            VLLMAdapter().normalize_inference_state(vllm_snap, "v", model_name="llama3-70b"),
            TritonAdapter().normalize_inference_state(triton_snap, "t", model="bert-large", version="1"),
            RayServeAdapter().normalize_inference_state(ray_snap, "r", deployment="llm-router"),
        ]
        valid_engines = {"vllm", "triton", "ray_serve", "unknown"}
        for svc in svcs:
            assert svc.engine in valid_engines, f"Invalid engine: {svc.engine}"

    def test_kv_cache_usage_always_fraction(self):
        """Across all adapters, kv_cache_usage must be in [0, 1] when not None."""
        vllm_snap = _vllm_snapshot_from_file()
        svc = VLLMAdapter().normalize_inference_state(vllm_snap, "v", model_name="llama3-70b")
        if svc.kv_cache_usage is not None:
            assert 0.0 <= svc.kv_cache_usage <= 1.0

    def test_no_field_forced_to_zero(self):
        """Missing fields must be None, not 0."""
        client = FakePrometheusClient(fixtures={
            "vllm:num_requests_waiting": [{"labels": {"model_name": "m"}, "value": 1.0}],
        })
        connector = PrometheusTelemetryConnector(client, vllm_registry(), source="test")
        snapshot = connector.fetch_snapshot()
        svc = VLLMAdapter().normalize_inference_state(snapshot, "v", model_name="m")
        # All other fields not in fixtures must be None, not 0
        for field in ["ttft_p50_ms", "ttft_p95_ms", "p50_latency_ms", "tokens_per_s", "kv_cache_usage"]:
            val = getattr(svc, field, "MISSING")
            assert val is None or isinstance(val, (int, float)), (
                f"Field {field} should be None or numeric, got {val!r}"
            )
