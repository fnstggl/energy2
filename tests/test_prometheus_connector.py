"""Tests for Phase 2: Prometheus-native telemetry ingestion.

Proves:
- parse_prometheus_text correctly parses all Prometheus text format variations
- FakePrometheusClient works offline with fixture data
- PrometheusClient sends correct auth headers (mocked)
- query() and query_range() parse Prometheus JSON API correctly
- Missing metrics → missing=True, not fabricated values
- Unit conversions work (seconds_to_ms, ratio_to_pct, etc.)
- Connector can run fully offline in sandbox mode
- TelemetrySnapshot coverage_pct and missing_metrics work correctly
- YAML metric mapping loads correctly
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aurelius.connectors.base import (
    AuthConfig,
    AuthType,
    ConnectorConfig,
    TelemetrySnapshot,
)
from aurelius.connectors.metric_mapping import (
    MetricMapping,
    MetricMappingRegistry,
    UnitConversion,
    dcgm_registry,
    load_mapping_dict,
    load_mapping_yaml,
    ray_serve_registry,
    triton_registry,
    vllm_registry,
)
from aurelius.connectors.prometheus import (
    _REQUESTS_AVAILABLE,
    FakePrometheusClient,
    PrometheusClient,
    PrometheusTelemetryConnector,
    _parse_prometheus_json_result,
    parse_prometheus_text,
)

try:
    import yaml as _yaml_unused  # noqa: F401
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_skip_requests = pytest.mark.skipif(
    not _REQUESTS_AVAILABLE,
    reason="requests not installed in test environment",
)
_skip_yaml = pytest.mark.skipif(
    not _YAML_AVAILABLE,
    reason="PyYAML not installed in test environment",
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prometheus")


# ---------------------------------------------------------------------------
# parse_prometheus_text
# ---------------------------------------------------------------------------

class TestParsePrometheusText:
    def test_basic_gauge(self):
        text = 'my_gauge{label="a"} 42.5'
        result = parse_prometheus_text(text)
        assert "my_gauge" in result
        r = result["my_gauge"]
        assert len(r.values) == 1
        assert r.values[0].value == pytest.approx(42.5)
        assert r.values[0].labels["label"] == "a"

    def test_comment_and_help_ignored(self):
        text = """
# HELP my_metric A metric
# TYPE my_metric gauge
my_metric 1.0
"""
        result = parse_prometheus_text(text)
        assert "my_metric" in result
        assert result["my_metric"].values[0].value == pytest.approx(1.0)

    def test_multiple_label_sets(self):
        text = """
node_cpu{cpu="0",mode="idle"} 100.0
node_cpu{cpu="1",mode="idle"} 95.5
node_cpu{cpu="0",mode="user"} 0.0
"""
        result = parse_prometheus_text(text)
        assert "node_cpu" in result
        assert len(result["node_cpu"].values) == 3

    def test_no_labels(self):
        text = "simple_counter 12345"
        result = parse_prometheus_text(text)
        assert "simple_counter" in result
        assert result["simple_counter"].values[0].value == pytest.approx(12345.0)
        assert result["simple_counter"].values[0].labels == {}

    def test_nan_value(self):
        text = 'broken_metric{foo="bar"} NaN'
        result = parse_prometheus_text(text)
        assert result["broken_metric"].values[0].value is None

    def test_inf_value(self):
        text = 'histogram_bucket{le="+Inf"} Inf'
        result = parse_prometheus_text(text)
        assert result["histogram_bucket"].values[0].value == float("inf")

    def test_scientific_notation(self):
        text = "big_metric 1.23e+09"
        result = parse_prometheus_text(text)
        assert result["big_metric"].values[0].value == pytest.approx(1.23e9)

    def test_empty_text(self):
        result = parse_prometheus_text("")
        assert result == {}

    def test_only_comments(self):
        result = parse_prometheus_text("# HELP nothing\n# TYPE nothing gauge")
        assert result == {}

    def test_dcgm_fixture_file(self):
        """Parse the real DCGM fixture file."""
        with open(os.path.join(FIXTURES_DIR, "dcgm_metrics.prom")) as f:
            text = f.read()
        result = parse_prometheus_text(text)
        assert "DCGM_FI_DEV_GPU_UTIL" in result
        util = result["DCGM_FI_DEV_GPU_UTIL"]
        assert len(util.values) == 3
        # Check specific values
        uuids = {mv.labels["UUID"]: mv.value for mv in util.values}
        assert uuids["GPU-aaaa-bbbb-cccc-0000"] == pytest.approx(72.5)
        assert uuids["GPU-aaaa-bbbb-cccc-0001"] == pytest.approx(88.0)

    def test_vllm_fixture_file(self):
        """Parse the vLLM fixture file."""
        with open(os.path.join(FIXTURES_DIR, "vllm_metrics.prom")) as f:
            text = f.read()
        result = parse_prometheus_text(text)
        assert "vllm:num_requests_waiting" in result
        waiting = result["vllm:num_requests_waiting"]
        by_model = {mv.labels.get("model_name"): mv.value for mv in waiting.values}
        assert by_model["llama3-70b"] == pytest.approx(3.0)
        assert by_model["mistral-7b"] == pytest.approx(12.0)

    def test_triton_fixture_file(self):
        """Parse the Triton fixture file."""
        with open(os.path.join(FIXTURES_DIR, "triton_metrics.prom")) as f:
            text = f.read()
        result = parse_prometheus_text(text)
        assert "nv_inference_count" in result
        assert "nv_inference_request_success" in result
        assert "nv_inference_request_failure" in result

    def test_timestamp_parsing(self):
        text = "my_metric 1.0 1716600000000"
        result = parse_prometheus_text(text)
        assert result["my_metric"].values[0].timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# MetricMapping and UnitConversion
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_none_passthrough(self):
        assert UnitConversion.apply(None, UnitConversion.NONE) is None

    def test_seconds_to_ms(self):
        assert UnitConversion.apply(1.5, UnitConversion.SECONDS_TO_MS) == pytest.approx(1500.0)

    def test_ratio_to_pct(self):
        assert UnitConversion.apply(0.75, UnitConversion.RATIO_TO_PCT) == pytest.approx(75.0)

    def test_none_value_returns_none(self):
        assert UnitConversion.apply(None, UnitConversion.SECONDS_TO_MS) is None

    def test_unknown_unit_no_conversion(self):
        assert UnitConversion.apply(5.0, "unknown_unit") == pytest.approx(5.0)

    def test_pct_no_conversion(self):
        assert UnitConversion.apply(85.0, UnitConversion.PCT) == pytest.approx(85.0)

    def test_mb_to_bytes(self):
        assert UnitConversion.apply(1.0, UnitConversion.MB_TO_BYTES) == pytest.approx(1_048_576.0)


class TestMetricMapping:
    def test_from_dict_basic(self):
        m = MetricMapping.from_dict("gpu.util_pct", {
            "query": "DCGM_FI_DEV_GPU_UTIL",
            "unit": "pct",
            "labels": ["gpu", "node"],
        })
        assert m.canonical_field == "gpu.util_pct"
        assert m.query == "DCGM_FI_DEV_GPU_UTIL"
        assert m.unit == "pct"
        assert m.labels_to_keep == ["gpu", "node"]

    def test_from_dict_fallbacks(self):
        m = MetricMapping.from_dict("gpu.clocks", {
            "query": "DCGM_FI_DEV_CLOCKS_EVENT_REASONS",
            "fallback_queries": ["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"],
        })
        assert len(m.fallback_queries) == 1

    def test_convert_applies_unit(self):
        m = MetricMapping.from_dict("inference.ttft_ms", {
            "query": "...",
            "unit": "seconds_to_ms",
        })
        assert m.convert(0.5) == pytest.approx(500.0)
        assert m.convert(None) is None


class TestMetricMappingRegistry:
    def test_dcgm_registry_has_expected_fields(self):
        reg = dcgm_registry()
        assert reg.get("gpu.util_pct") is not None
        assert reg.get("gpu.power_w") is not None
        assert reg.get("gpu.temp_c") is not None
        assert reg.get("gpu.clocks_event_reasons") is not None
        assert reg.get("gpu.nvlink_tx_bytes_per_s") is not None

    def test_dcgm_registry_clocks_has_fallback(self):
        reg = dcgm_registry()
        clocks = reg.get("gpu.clocks_event_reasons")
        assert clocks is not None
        assert len(clocks.fallback_queries) >= 1

    def test_vllm_registry_has_expected_fields(self):
        reg = vllm_registry()
        assert reg.get("inference.ttft_p95_ms") is not None
        assert reg.get("inference.kv_cache_usage_pct") is not None
        assert reg.get("inference.prefix_cache_hit_rate_pct") is not None

    def test_triton_registry(self):
        reg = triton_registry()
        assert reg.get("triton.inference_count") is not None

    def test_ray_registry(self):
        reg = ray_serve_registry()
        assert reg.get("ray.serve.num_replicas") is not None

    def test_from_dict(self):
        d = {
            "custom.field": {"query": "my_metric", "unit": "pct"},
        }
        reg = MetricMappingRegistry.from_dict(d)
        assert reg.get("custom.field") is not None
        assert len(reg) == 1

    def test_load_mapping_dict(self):
        d = {
            "gpu.util_pct": {"query": "MY_GPU_UTIL", "unit": "pct"},
        }
        reg = load_mapping_dict(d)
        assert reg.get("gpu.util_pct") is not None

    @pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed in test environment")
    def test_load_mapping_yaml_dcgm(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "connectors", "dcgm_mapping.yaml")
        reg = load_mapping_yaml(path)
        assert reg.get("gpu.util_pct") is not None
        assert reg.get("gpu.clocks_event_reasons") is not None

    @pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed in test environment")
    def test_load_mapping_yaml_vllm(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "connectors", "vllm_mapping.yaml")
        reg = load_mapping_yaml(path)
        assert reg.get("inference.ttft_p95_ms") is not None


# ---------------------------------------------------------------------------
# FakePrometheusClient
# ---------------------------------------------------------------------------

class TestFakePrometheusClient:
    def test_query_from_fixtures_dict(self):
        client = FakePrometheusClient(fixtures={
            "DCGM_FI_DEV_GPU_UTIL": [
                {"labels": {"gpu": "0", "UUID": "abc"}, "value": 75.0},
                {"labels": {"gpu": "1", "UUID": "def"}, "value": 50.0},
            ]
        })
        result = client.query("DCGM_FI_DEV_GPU_UTIL", canonical_field="gpu.util_pct")
        assert not result.missing
        assert len(result.values) == 2
        by_uuid = {mv.labels["UUID"]: mv.value for mv in result.values}
        assert by_uuid["abc"] == pytest.approx(75.0)

    def test_query_missing_returns_missing(self):
        client = FakePrometheusClient(fixtures={})
        result = client.query("NONEXISTENT_METRIC")
        assert result.missing
        assert result.first_value is None

    def test_query_from_prometheus_text(self):
        text = """
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-abc",node="n1"} 65.0
DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="GPU-def",node="n1"} 71.5
"""
        client = FakePrometheusClient(prometheus_text=text)
        result = client.query("DCGM_FI_DEV_GPU_TEMP")
        assert not result.missing
        assert len(result.values) == 2

    def test_query_applies_unit_conversion(self):
        client = FakePrometheusClient(fixtures={
            "my_seconds": [{"labels": {}, "value": 1.5}]
        })
        mapping = MetricMapping(
            canonical_field="my_field",
            query="my_seconds",
            unit=UnitConversion.SECONDS_TO_MS,
        )
        result = client.query("my_seconds", canonical_field="my_field", mapping=mapping)
        assert result.values[0].value == pytest.approx(1500.0)

    def test_scrape_metrics_from_text(self):
        with open(os.path.join(FIXTURES_DIR, "dcgm_metrics.prom")) as f:
            text = f.read()
        client = FakePrometheusClient(prometheus_text=text)
        raw = client.scrape_metrics()
        assert "DCGM_FI_DEV_GPU_UTIL" in raw
        assert "DCGM_FI_DEV_POWER_USAGE" in raw

    def test_fetch_snapshot_marks_unknown(self):
        client = FakePrometheusClient(fixtures={
            "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"UUID": "abc"}, "value": 72.0}],
        })
        reg = MetricMappingRegistry.from_dict({
            "gpu.util_pct": {"query": "DCGM_FI_DEV_GPU_UTIL"},
            "gpu.power_w": {"query": "DCGM_FI_DEV_POWER_USAGE"},  # not in fixtures
        })
        snapshot = client.fetch_snapshot(reg, source="test")
        assert "gpu.util_pct" in snapshot.metrics
        assert "gpu.power_w" in snapshot.unknown_metrics
        assert not snapshot.metrics["gpu.util_pct"].missing

    def test_fetch_snapshot_coverage_pct(self):
        client = FakePrometheusClient(fixtures={
            "M1": [{"labels": {}, "value": 1.0}],
        })
        reg = MetricMappingRegistry.from_dict({
            "f1": {"query": "M1"},
            "f2": {"query": "M2"},
            "f3": {"query": "M3"},
        })
        snapshot = client.fetch_snapshot(reg)
        assert snapshot.coverage_pct() == pytest.approx(100.0 / 3.0)

    def test_query_range_returns_same_as_query(self):
        client = FakePrometheusClient(fixtures={
            "my_metric": [{"labels": {}, "value": 42.0}]
        })
        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        result = client.query_range("my_metric", now - timedelta(hours=1), now)
        assert result.values[0].value == pytest.approx(42.0)

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed in test environment")
    def test_no_network_calls(self):
        """FakePrometheusClient must never make network calls."""
        with patch("requests.Session") as mock_session:
            client = FakePrometheusClient(fixtures={"m": [{"labels": {}, "value": 1.0}]})
            client.query("m")
            mock_session.assert_not_called()


# ---------------------------------------------------------------------------
# PrometheusClient (real HTTP — mocked)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed in test environment")
class TestPrometheusClientMocked:
    def _make_client(self, base_url="http://prometheus.test:9090"):
        config = ConnectorConfig(base_url=base_url)
        return PrometheusClient(config)

    def test_query_sends_correct_url(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {"resultType": "vector", "result": [
                {"metric": {"gpu": "0"}, "value": [1716600000.0, "75.0"]}
            ]}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            result = client.query("DCGM_FI_DEV_GPU_UTIL")
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "/api/v1/query" in url
            params = mock_get.call_args[1].get("params") or mock_get.call_args[0][1] if len(mock_get.call_args[0]) > 1 else mock_get.call_args[1].get("params")
            assert params is not None
            assert params["query"] == "DCGM_FI_DEV_GPU_UTIL"

        assert not result.missing
        assert result.values[0].value == pytest.approx(75.0)

    def test_bearer_token_sent(self):
        config = ConnectorConfig(
            base_url="http://prom.test:9090",
            auth=AuthConfig(type=AuthType.BEARER, token_env="TEST_PROM_TOKEN"),
        )
        client = PrometheusClient(config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "data": {"resultType": "vector", "result": []}}
        mock_resp.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"TEST_PROM_TOKEN": "secret-token-value"}):
            with patch.object(client._session, "get", return_value=mock_resp):
                client.query("some_metric")
                assert client._session.headers.get("Authorization") == "Bearer secret-token-value"

    def test_missing_prometheus_result_returns_missing(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {"resultType": "vector", "result": []}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.query("NONEXISTENT_METRIC", canonical_field="my.field")
            assert result.missing
            assert result.first_value is None

    def test_http_error_returns_missing_with_error_message(self):
        client = self._make_client()
        import requests as req_lib
        with patch.object(client._session, "get", side_effect=req_lib.ConnectionError("refused")):
            result = client.query("some_metric", canonical_field="my.field")
            assert result.missing
            assert result.error is not None

    def test_prometheus_error_response(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "error", "error": "query parse error"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.query("invalid{query}")
            assert result.missing
            assert "parse error" in (result.error or "")

    def test_query_range_url(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {}, "values": [[1716600000.0, "42.0"], [1716600060.0, "45.0"]]}
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()

        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            result = client.query_range("my_metric", now - timedelta(hours=1), now)
            url = mock_get.call_args[0][0]
            assert "/api/v1/query_range" in url

        assert not result.missing
        # Should take the last value from the range
        assert result.first_value == pytest.approx(45.0)

    def test_basic_auth_credentials(self):
        config = ConnectorConfig(
            base_url="http://prom.test:9090",
            auth=AuthConfig(
                type=AuthType.BASIC,
                username="aurelius",
                password_env="TEST_PROM_PASSWORD",
            ),
        )
        PrometheusClient(config)  # verify construction succeeds
        assert config.auth.basic_credentials() is None  # env not set

        with patch.dict(os.environ, {"TEST_PROM_PASSWORD": "s3cr3t"}):
            creds = config.auth.basic_credentials()
            assert creds == ("aurelius", "s3cr3t")


# ---------------------------------------------------------------------------
# _parse_prometheus_json_result
# ---------------------------------------------------------------------------

class TestParsePrometheusJsonResult:
    def test_vector_result(self):
        data = {
            "resultType": "vector",
            "result": [
                {"metric": {"gpu": "0", "UUID": "abc"}, "value": [1716600000.0, "75.5"]},
            ]
        }
        r = _parse_prometheus_json_result("UTIL", "gpu.util_pct", data, datetime.now(tz=timezone.utc))
        assert not r.missing
        assert r.values[0].value == pytest.approx(75.5)
        assert r.values[0].labels["UUID"] == "abc"

    def test_empty_result_is_missing(self):
        data = {"resultType": "vector", "result": []}
        r = _parse_prometheus_json_result("METRIC", "field", data, datetime.now(tz=timezone.utc))
        assert r.missing

    def test_matrix_result_takes_last_value(self):
        data = {
            "resultType": "matrix",
            "result": [{
                "metric": {"node": "n1"},
                "values": [[1716600000.0, "10.0"], [1716600060.0, "20.0"]]
            }]
        }
        r = _parse_prometheus_json_result("METRIC", "field", data, datetime.now(tz=timezone.utc))
        assert r.first_value == pytest.approx(20.0)

    def test_name_label_stripped(self):
        data = {
            "resultType": "vector",
            "result": [{"metric": {"__name__": "METRIC", "gpu": "0"}, "value": [0.0, "1.0"]}]
        }
        r = _parse_prometheus_json_result("METRIC", "field", data, datetime.now(tz=timezone.utc))
        assert "__name__" not in r.values[0].labels

    def test_unit_conversion_applied(self):
        data = {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [0.0, "1.5"]}]
        }
        mapping = MetricMapping(
            canonical_field="my.field",
            query="...",
            unit=UnitConversion.SECONDS_TO_MS,
        )
        r = _parse_prometheus_json_result("METRIC", "my.field", data, datetime.now(tz=timezone.utc), mapping=mapping)
        assert r.values[0].value == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# PrometheusTelemetryConnector
# ---------------------------------------------------------------------------

class TestPrometheusTelemetryConnector:
    def test_sandbox_connector_fetch_snapshot(self):
        text = open(os.path.join(FIXTURES_DIR, "dcgm_metrics.prom")).read()
        client = FakePrometheusClient(prometheus_text=text)
        reg = dcgm_registry()
        connector = PrometheusTelemetryConnector(client, reg, source="dcgm-sandbox")
        snapshot = connector.fetch_snapshot()
        assert snapshot.source == "dcgm-sandbox"
        assert snapshot.is_sandbox
        assert snapshot.coverage_pct() > 0

    def test_scrape_snapshot_from_fixture(self):
        text = open(os.path.join(FIXTURES_DIR, "dcgm_metrics.prom")).read()
        client = FakePrometheusClient(prometheus_text=text)
        reg = dcgm_registry()
        connector = PrometheusTelemetryConnector(client, reg, source="dcgm-scrape")
        snapshot = connector.scrape_snapshot()
        assert "DCGM_FI_DEV_GPU_UTIL" in snapshot.metrics

    def test_missing_metrics_not_fabricated(self):
        """Connector must not fabricate values for missing metrics."""
        client = FakePrometheusClient(fixtures={
            "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"UUID": "abc"}, "value": 80.0}],
        })
        # Registry includes many fields not in the fixtures
        reg = dcgm_registry()
        connector = PrometheusTelemetryConnector(client, reg, source="partial")
        snapshot = connector.fetch_snapshot()

        # Util should be present
        util_result = snapshot.get("gpu.util_pct")
        assert util_result is not None
        assert not util_result.missing

        # Power should be missing (not in fixtures)
        power_result = snapshot.get("gpu.power_w")
        assert power_result is None or "gpu.power_w" in snapshot.unknown_metrics

    def test_is_sandbox_with_fake_client(self):
        client = FakePrometheusClient(is_sandbox=True)
        reg = MetricMappingRegistry.empty()
        connector = PrometheusTelemetryConnector(client, reg)
        assert connector.is_sandbox


# ---------------------------------------------------------------------------
# ConnectorConfig and AuthConfig
# ---------------------------------------------------------------------------

class TestConnectorConfig:
    def test_from_dict(self):
        d = {
            "base_url": "http://prom.test:9090",
            "auth": {"type": "bearer", "token_env": "MY_TOKEN"},
            "tls_verify": False,
            "timeout_s": 15.0,
            "max_retries": 5,
            "is_sandbox": True,
        }
        config = ConnectorConfig.from_dict(d)
        assert config.base_url == "http://prom.test:9090"
        assert config.auth.type == AuthType.BEARER
        assert config.auth.token_env == "MY_TOKEN"
        assert config.tls_verify is False
        assert config.timeout_s == 15.0
        assert config.is_sandbox is True

    def test_bearer_token_reads_env(self):
        auth = AuthConfig(type=AuthType.BEARER, token_env="MY_BEARER_TOKEN")
        with patch.dict(os.environ, {"MY_BEARER_TOKEN": "tok123"}):
            assert auth.bearer_token() == "tok123"

    def test_missing_env_returns_none(self):
        auth = AuthConfig(type=AuthType.BEARER, token_env="NONEXISTENT_VAR_XYZ")
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        assert auth.bearer_token() is None

    def test_no_auth(self):
        auth = AuthConfig(type=AuthType.NONE)
        assert auth.bearer_token() is None
        assert auth.basic_credentials() is None


# ---------------------------------------------------------------------------
# TelemetrySnapshot helpers
# ---------------------------------------------------------------------------

class TestTelemetrySnapshot:
    def _make_snapshot(self, has_metrics: int, has_unknown: int) -> TelemetrySnapshot:
        from aurelius.connectors.base import MetricValue, RawMetricResult
        metrics = {}
        for i in range(has_metrics):
            field = f"field_{i}"
            mv = MetricValue(
                metric_name=field,
                labels={},
                value=float(i),
                timestamp=datetime.now(tz=timezone.utc),
            )
            metrics[field] = RawMetricResult(
                metric_name=field,
                query=field,
                values=[mv],
                fetched_at=datetime.now(tz=timezone.utc),
            )
        unknown = [f"unknown_{i}" for i in range(has_unknown)]
        return TelemetrySnapshot(
            source="test",
            fetched_at=datetime.now(tz=timezone.utc),
            metrics=metrics,
            unknown_metrics=unknown,
        )

    def test_coverage_pct_all_known(self):
        snap = self._make_snapshot(5, 0)
        assert snap.coverage_pct() == pytest.approx(100.0)

    def test_coverage_pct_all_unknown(self):
        snap = self._make_snapshot(0, 5)
        assert snap.coverage_pct() == pytest.approx(0.0)

    def test_coverage_pct_mixed(self):
        snap = self._make_snapshot(3, 7)
        assert snap.coverage_pct() == pytest.approx(30.0)

    def test_value_returns_none_for_missing(self):
        snap = self._make_snapshot(0, 0)
        assert snap.value("nonexistent") is None

    def test_value_for_labels(self):
        from aurelius.connectors.base import MetricValue, RawMetricResult
        ts = datetime.now(tz=timezone.utc)
        mv1 = MetricValue("m", {"gpu": "0"}, 10.0, ts)
        mv2 = MetricValue("m", {"gpu": "1"}, 20.0, ts)
        r = RawMetricResult("m", "m", [mv1, mv2], ts)
        snap = TelemetrySnapshot("test", ts, metrics={"field": r})
        assert snap.value_for_labels("field", gpu="0") == pytest.approx(10.0)
        assert snap.value_for_labels("field", gpu="1") == pytest.approx(20.0)
        assert snap.value_for_labels("field", gpu="99") is None
