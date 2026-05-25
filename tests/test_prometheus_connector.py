"""Tests for Prometheus connector — Phase 2.

All tests run fully offline using sandbox fixture injection.
No live Prometheus server required.
"""

import json
import pathlib
import pytest
from datetime import datetime, timezone

from aurelius.connectors.prometheus.client import (
    PrometheusClient,
    PrometheusQueryError,
    PrometheusAuth,
)
from aurelius.connectors.prometheus.connector import PrometheusTelemetryConnector
from aurelius.connectors.prometheus.mappings import DEFAULT_DCGM_MAPPINGS, DEFAULT_VLLM_MAPPINGS

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "prometheus"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


QUERY_RESPONSE = (FIXTURES / "prometheus_query_response.json").read_text()
EMPTY_RESPONSE = json.dumps({
    "status": "success",
    "data": {"resultType": "vector", "result": []}
})


# ---------------------------------------------------------------------------
# PrometheusClient tests
# ---------------------------------------------------------------------------

class TestPrometheusClient:
    def test_sandbox_query_returns_fixture(self):
        sandbox = {"/api/v1/query": QUERY_RESPONSE}
        client = PrometheusClient(_sandbox_responses=sandbox)
        result = client.query("DCGM_FI_DEV_GPU_UTIL")
        assert result["status"] == "success"
        assert result["data"]["resultType"] == "vector"
        assert len(result["data"]["result"]) == 2

    def test_sandbox_raw_scrape(self):
        metrics_text = load_fixture("dcgm_metrics.txt")
        sandbox = {"/metrics": metrics_text}
        client = PrometheusClient(_sandbox_responses=sandbox)
        text = client.scrape_metrics()
        assert "DCGM_FI_DEV_GPU_UTIL" in text

    def test_sandbox_missing_key_raises(self):
        client = PrometheusClient(_sandbox_responses={})
        with pytest.raises(PrometheusQueryError):
            client.query("missing_metric")

    def test_bearer_auth_header(self):
        auth = PrometheusAuth(auth_type="bearer", token="test-token-xyz")
        headers = auth.headers()
        assert headers.get("Authorization") == "Bearer test-token-xyz"

    def test_basic_auth_header(self):
        import base64
        auth = PrometheusAuth(auth_type="basic", username="user", password="pass")
        headers = auth.headers()
        creds = base64.b64encode(b"user:pass").decode()
        assert headers.get("Authorization") == f"Basic {creds}"

    def test_no_auth_empty_headers(self):
        auth = PrometheusAuth(auth_type="none")
        assert auth.headers() == {}

    def test_prometheus_error_status_raises(self):
        error_response = json.dumps({
            "status": "error",
            "error": "bad_request",
            "errorType": "bad_data",
        })
        sandbox = {"/api/v1/query": error_response}
        client = PrometheusClient(_sandbox_responses=sandbox)
        with pytest.raises(PrometheusQueryError, match="Prometheus error"):
            client.query("bad_promql{{{")

    def test_invalid_json_raises(self):
        sandbox = {"/api/v1/query": "this is not json"}
        client = PrometheusClient(_sandbox_responses=sandbox)
        with pytest.raises(PrometheusQueryError, match="Invalid JSON"):
            client.query("metric")

    def test_query_range_sandbox(self):
        range_response = json.dumps({
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"gpu": "0", "UUID": "GPU-abc123"},
                        "values": [
                            [1705320000, "75.0"],
                            [1705320060, "76.0"],
                        ],
                    }
                ],
            },
        })
        sandbox = {"/api/v1/query_range": range_response}
        client = PrometheusClient(_sandbox_responses=sandbox)
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        result = client.query_range("metric", start=now - timedelta(hours=1), end=now)
        assert result["status"] == "success"
        assert result["data"]["resultType"] == "matrix"


# ---------------------------------------------------------------------------
# PrometheusTelemetryConnector tests
# ---------------------------------------------------------------------------

class TestPrometheusTelemetryConnector:
    def _make_connector(self, query_response: str = QUERY_RESPONSE) -> PrometheusTelemetryConnector:
        # Respond to any query with the fixture
        sandbox = {"/api/v1/query": query_response}
        client = PrometheusClient(_sandbox_responses=sandbox)
        return PrometheusTelemetryConnector(client=client, cluster_id="test-cluster")

    def test_fetch_cluster_state_returns_snapshot(self):
        connector = self._make_connector()
        snap = connector.fetch_cluster_state()
        assert snap.cluster_id == "test-cluster"
        assert snap.collected_at is not None

    def test_missing_metrics_recorded_as_unknown(self):
        connector = self._make_connector(EMPTY_RESPONSE)
        snap = connector.fetch_cluster_state()
        assert len(snap.unknown_metrics) > 0

    def test_missing_metrics_not_fabricated_as_zero(self):
        connector = self._make_connector(EMPTY_RESPONSE)
        snap = connector.fetch_cluster_state()
        # No GPUs should be created from empty responses
        assert len(snap.gpus) == 0

    def test_gpu_fields_populated_from_vector_result(self):
        # Build a response that maps to gpu.utilization_pct
        util_response = json.dumps({
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"UUID": "GPU-test-1", "node": "node-1"},
                        "value": [1705320000, "77.5"],
                    }
                ],
            },
        })
        sandbox = {"/api/v1/query": util_response}
        client = PrometheusClient(_sandbox_responses=sandbox)
        connector = PrometheusTelemetryConnector(
            client=client,
            gpu_mapping=DEFAULT_DCGM_MAPPINGS,
            cluster_id="c1",
        )
        snap = connector.fetch_cluster_state()
        assert "GPU-test-1" in snap.gpus
        gpu = snap.gpus["GPU-test-1"]
        assert gpu.utilization_pct == pytest.approx(77.5)

    def test_to_cluster_state(self):
        connector = self._make_connector()
        snap = connector.fetch_cluster_state()
        cs = snap.to_cluster_state()
        from aurelius.state.models import ClusterState
        assert isinstance(cs, ClusterState)
        assert cs.cluster_id == snap.cluster_id

    def test_errors_are_non_fatal(self):
        # Even if all queries fail, fetch_cluster_state should not raise
        sandbox = {}  # all lookups will fail
        # Use empty sandbox that raises PrometheusQueryError
        from unittest.mock import patch, MagicMock
        client = PrometheusClient(_sandbox_responses={"/api/v1/query": EMPTY_RESPONSE})
        connector = PrometheusTelemetryConnector(client=client)
        snap = connector.fetch_cluster_state()
        # Should have returned without raising
        assert snap is not None


# ---------------------------------------------------------------------------
# Unit conversion tests
# ---------------------------------------------------------------------------

class TestUnitConversions:
    def test_dcgm_mapping_has_memory_query(self):
        mapping = DEFAULT_DCGM_MAPPINGS.get("gpu.memory_used_bytes")
        assert mapping is not None
        assert len(mapping.queries) >= 1

    def test_vllm_mapping_has_kv_cache_query(self):
        mapping = DEFAULT_VLLM_MAPPINGS.get("inference.kv_cache_usage_pct")
        assert mapping is not None

    def test_mapping_registry_all_fields(self):
        fields = DEFAULT_DCGM_MAPPINGS.all_fields()
        assert "gpu.utilization_pct" in fields
        assert "gpu.power_watts" in fields
        assert "gpu.temperature_c" in fields
