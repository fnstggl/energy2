"""Generic Prometheus HTTP client and telemetry connector for Aurelius.

Supports:
- Prometheus HTTP API (/api/v1/query, /api/v1/query_range)
- Direct /metrics endpoint scrape (Prometheus text format)
- Sandbox/fixture mode (FakePrometheusClient) — no network required
- Bearer token, basic auth, custom headers, TLS verify toggle
- Retries with exponential backoff

Usage (real Prometheus):
    config = ConnectorConfig(base_url="http://prometheus:9090", ...)
    client = PrometheusClient(config)
    result = client.query("DCGM_FI_DEV_GPU_UTIL")

Usage (sandbox/test):
    client = FakePrometheusClient(fixtures={
        "DCGM_FI_DEV_GPU_UTIL": [{"labels": {"gpu": "0"}, "value": 75.0}]
    })
    result = client.query("DCGM_FI_DEV_GPU_UTIL")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from aurelius.connectors.base import (
    ConnectorConfig,
    MetricValue,
    RawMetricResult,
    TelemetrySnapshot,
)
from aurelius.connectors.metric_mapping import MetricMapping, MetricMappingRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus text format parser
# ---------------------------------------------------------------------------

_PROM_SAMPLE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+'
    r'(?P<value>[+-]?(?:(?i:inf|nan)|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))'
    r'(?:\s+(?P<ts>\d+))?$'
)

_LABEL_PAIR_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus_text(text: str, fetched_at: Optional[datetime] = None) -> dict[str, RawMetricResult]:
    """Parse Prometheus exposition text format into RawMetricResult dict.

    Handles: HELP, TYPE, comment lines, multi-label metrics, NaN/Inf values.

    Returns:
        dict mapping metric_name → RawMetricResult (one per unique metric name,
        with multiple MetricValue entries for each label set).
    """
    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)

    results: dict[str, RawMetricResult] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _PROM_SAMPLE_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        raw_labels_str = m.group("labels") or ""
        raw_value_str = m.group("value")

        # Parse value
        try:
            if raw_value_str.lower() in ("inf", "+inf"):
                value = float("inf")
            elif raw_value_str.lower() in ("-inf",):
                value = float("-inf")
            elif raw_value_str.lower() == "nan":
                value = None
            else:
                value = float(raw_value_str)
        except ValueError:
            value = None

        # Parse labels
        labels: dict[str, str] = {}
        if raw_labels_str:
            for k, v in _LABEL_PAIR_RE.findall(raw_labels_str):
                labels[k] = v

        # Parse timestamp (Prometheus uses milliseconds, convert to UTC)
        ts_ms = m.group("ts")
        if ts_ms:
            ts = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
        else:
            ts = fetched_at

        mv = MetricValue(
            metric_name=name,
            labels=labels,
            value=value,
            timestamp=ts,
            raw_value_str=raw_value_str,
        )

        if name not in results:
            results[name] = RawMetricResult(
                metric_name=name,
                query=name,
                values=[],
                fetched_at=fetched_at,
                missing=False,
            )
        results[name].values.append(mv)

    return results


# ---------------------------------------------------------------------------
# Prometheus JSON API response parser
# ---------------------------------------------------------------------------

def _parse_prometheus_json_result(
    query: str,
    canonical_field: str,
    data: dict[str, Any],
    fetched_at: datetime,
    mapping: Optional[MetricMapping] = None,
) -> RawMetricResult:
    """Parse a Prometheus JSON API /api/v1/query response result."""
    result_type = data.get("resultType", "vector")
    raw_results = data.get("result", [])

    if not raw_results:
        return RawMetricResult(
            metric_name=canonical_field,
            query=query,
            values=[],
            fetched_at=fetched_at,
            missing=True,
        )

    values: list[MetricValue] = []

    if result_type == "vector":
        for item in raw_results:
            labels = item.get("metric", {})
            ts_val = item.get("value", [None, None])
            ts_unix = ts_val[0]
            raw_str = str(ts_val[1]) if ts_val[1] is not None else ""

            if ts_unix is not None:
                ts = datetime.fromtimestamp(float(ts_unix), tz=timezone.utc)
            else:
                ts = fetched_at

            try:
                fv: Optional[float] = float(raw_str) if raw_str else None
            except (ValueError, TypeError):
                fv = None

            if mapping is not None:
                fv = mapping.convert(fv)

            mv = MetricValue(
                metric_name=canonical_field,
                labels={k: str(v) for k, v in labels.items() if k != "__name__"},
                value=fv,
                timestamp=ts,
                raw_value_str=raw_str,
            )
            values.append(mv)

    elif result_type == "matrix":
        # For range queries: take the last value for each series
        for item in raw_results:
            labels = item.get("metric", {})
            vals = item.get("values", [])
            if not vals:
                continue
            ts_unix, raw_str = vals[-1]
            ts = datetime.fromtimestamp(float(ts_unix), tz=timezone.utc)
            try:
                fv = float(raw_str) if raw_str else None
            except (ValueError, TypeError):
                fv = None

            if mapping is not None:
                fv = mapping.convert(fv)

            mv = MetricValue(
                metric_name=canonical_field,
                labels={k: str(v) for k, v in labels.items() if k != "__name__"},
                value=fv,
                timestamp=ts,
                raw_value_str=str(raw_str),
            )
            values.append(mv)

    return RawMetricResult(
        metric_name=canonical_field,
        query=query,
        values=values,
        fetched_at=fetched_at,
        missing=(len(values) == 0),
    )


# ---------------------------------------------------------------------------
# PrometheusClient — real HTTP
# ---------------------------------------------------------------------------

class PrometheusClient:
    """Generic Prometheus HTTP client.

    Communicates with a Prometheus server via the HTTP API or scrapes a
    /metrics endpoint directly. Supports bearer/basic auth, TLS toggle,
    retries with exponential backoff.

    Security: secrets are read from environment variables at call time,
    never stored in the client object. The config only stores env var names.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for PrometheusClient. "
                "Install it with: pip install requests"
            )
        self._config = config
        self._session = requests.Session()
        self._session.verify = config.tls_verify
        if config.extra_headers:
            self._session.headers.update(config.extra_headers)

    def _auth_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        token = self._config.auth.bearer_token()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
            return kwargs
        creds = self._config.auth.basic_credentials()
        if creds:
            kwargs["auth"] = creds
        return kwargs

    def _get_with_retry(self, url: str, params: Optional[dict] = None) -> dict[str, Any]:
        last_exc: Exception = RuntimeError("no attempts made")
        delay = 1.0
        for attempt in range(self._config.max_retries + 1):
            try:
                auth_kwargs = self._auth_kwargs()
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.timeout_s,
                    **auth_kwargs,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self._config.max_retries:
                    logger.warning(
                        "PrometheusClient: request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._config.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
        raise last_exc

    def query(
        self,
        expr: str,
        canonical_field: str = "",
        mapping: Optional[MetricMapping] = None,
        at: Optional[datetime] = None,
    ) -> RawMetricResult:
        """Execute an instant query against the Prometheus HTTP API."""
        fetched_at = datetime.now(tz=timezone.utc)
        params: dict[str, Any] = {"query": expr}
        if at is not None:
            params["time"] = at.timestamp()

        try:
            resp = self._get_with_retry(
                f"{self._config.base_url.rstrip('/')}/api/v1/query",
                params=params,
            )
        except Exception as exc:
            logger.warning("PrometheusClient.query failed for %r: %s", expr, exc)
            return RawMetricResult(
                metric_name=canonical_field or expr,
                query=expr,
                fetched_at=fetched_at,
                missing=True,
                error=str(exc),
            )

        if resp.get("status") != "success":
            err = resp.get("error", "unknown Prometheus error")
            return RawMetricResult(
                metric_name=canonical_field or expr,
                query=expr,
                fetched_at=fetched_at,
                missing=True,
                error=err,
            )

        return _parse_prometheus_json_result(
            query=expr,
            canonical_field=canonical_field or expr,
            data=resp.get("data", {}),
            fetched_at=fetched_at,
            mapping=mapping,
        )

    def query_range(
        self,
        expr: str,
        start: datetime,
        end: datetime,
        step: str = "60s",
        canonical_field: str = "",
        mapping: Optional[MetricMapping] = None,
    ) -> RawMetricResult:
        """Execute a range query against the Prometheus HTTP API."""
        fetched_at = datetime.now(tz=timezone.utc)
        params = {
            "query": expr,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }
        try:
            resp = self._get_with_retry(
                f"{self._config.base_url.rstrip('/')}/api/v1/query_range",
                params=params,
            )
        except Exception as exc:
            logger.warning("PrometheusClient.query_range failed for %r: %s", expr, exc)
            return RawMetricResult(
                metric_name=canonical_field or expr,
                query=expr,
                fetched_at=fetched_at,
                missing=True,
                error=str(exc),
            )

        if resp.get("status") != "success":
            err = resp.get("error", "unknown Prometheus error")
            return RawMetricResult(
                metric_name=canonical_field or expr,
                query=expr,
                fetched_at=fetched_at,
                missing=True,
                error=err,
            )

        return _parse_prometheus_json_result(
            query=expr,
            canonical_field=canonical_field or expr,
            data=resp.get("data", {}),
            fetched_at=fetched_at,
            mapping=mapping,
        )

    def scrape_metrics(self, path: str = "/metrics") -> dict[str, RawMetricResult]:
        """Scrape a raw /metrics endpoint and parse Prometheus text format."""
        fetched_at = datetime.now(tz=timezone.utc)
        url = f"{self._config.base_url.rstrip('/')}{path}"
        auth_kwargs = self._auth_kwargs()

        try:
            resp = self._session.get(url, timeout=self._config.timeout_s, **auth_kwargs)
            resp.raise_for_status()
            return parse_prometheus_text(resp.text, fetched_at=fetched_at)
        except Exception as exc:
            logger.warning("PrometheusClient.scrape_metrics failed for %s: %s", url, exc)
            return {}

    def fetch_snapshot(
        self,
        registry: MetricMappingRegistry,
        source: str = "prometheus",
    ) -> TelemetrySnapshot:
        """Query all fields in the registry and return a TelemetrySnapshot.

        Tries the primary query, then falls back to fallback_queries.
        Missing metrics are recorded in snapshot.unknown_metrics.
        """
        fetched_at = datetime.now(tz=timezone.utc)
        metrics: dict[str, RawMetricResult] = {}
        unknown: list[str] = []

        for mapping in registry.all_mappings():
            queries = [mapping.query] + mapping.fallback_queries
            result: Optional[RawMetricResult] = None

            for q in queries:
                r = self.query(
                    expr=q,
                    canonical_field=mapping.canonical_field,
                    mapping=mapping,
                )
                if not r.missing:
                    result = r
                    break

            if result is not None and not result.missing:
                metrics[mapping.canonical_field] = result
            else:
                unknown.append(mapping.canonical_field)

        return TelemetrySnapshot(
            source=source,
            fetched_at=fetched_at,
            is_sandbox=self._config.is_sandbox,
            metrics=metrics,
            unknown_metrics=unknown,
        )


# ---------------------------------------------------------------------------
# FakePrometheusClient — offline fixture-based client for tests
# ---------------------------------------------------------------------------

@dataclass
class FakePrometheusClient:
    """Offline Prometheus client for tests and sandbox mode.

    Accepts pre-loaded fixture data as a dict mapping:
        metric_name_or_expr → list of {labels: dict, value: float}

    Or as raw Prometheus text format (prometheus_text).

    No network connections are made. Provides the same interface as
    PrometheusClient so tests use the same code paths.
    """

    fixtures: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prometheus_text: str = ""
    is_sandbox: bool = True
    _text_cache: dict[str, RawMetricResult] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prometheus_text:
            self._text_cache = parse_prometheus_text(self.prometheus_text)

    def query(
        self,
        expr: str,
        canonical_field: str = "",
        mapping: Optional[MetricMapping] = None,
        at: Optional[datetime] = None,
    ) -> RawMetricResult:
        fetched_at = datetime.now(tz=timezone.utc)
        field_name = canonical_field or expr

        # Try text fixtures first (from prometheus_text)
        if expr in self._text_cache:
            r = self._text_cache[expr]
            result = RawMetricResult(
                metric_name=field_name,
                query=expr,
                values=r.values,
                fetched_at=fetched_at,
                missing=r.missing,
            )
            if mapping:
                result = _apply_mapping_to_result(result, mapping)
            return result

        # Try dict fixtures
        raw_items = self.fixtures.get(expr) or self.fixtures.get(field_name)
        if raw_items is not None:
            values = []
            for item in raw_items:
                raw_str = str(item.get("value", ""))
                fv: Optional[float] = None
                try:
                    fv = float(raw_str) if raw_str else None
                except (ValueError, TypeError):
                    fv = None
                if mapping is not None:
                    fv = mapping.convert(fv)
                mv = MetricValue(
                    metric_name=field_name,
                    labels=dict(item.get("labels", {})),
                    value=fv,
                    timestamp=fetched_at,
                    raw_value_str=raw_str,
                )
                values.append(mv)
            return RawMetricResult(
                metric_name=field_name,
                query=expr,
                values=values,
                fetched_at=fetched_at,
                missing=(len(values) == 0),
            )

        return RawMetricResult(
            metric_name=field_name,
            query=expr,
            fetched_at=fetched_at,
            missing=True,
        )

    def query_range(
        self,
        expr: str,
        start: datetime,
        end: datetime,
        step: str = "60s",
        canonical_field: str = "",
        mapping: Optional[MetricMapping] = None,
    ) -> RawMetricResult:
        return self.query(expr=expr, canonical_field=canonical_field, mapping=mapping)

    def scrape_metrics(self, path: str = "/metrics") -> dict[str, RawMetricResult]:
        if self._text_cache:
            return dict(self._text_cache)
        return {}

    def fetch_snapshot(
        self,
        registry: MetricMappingRegistry,
        source: str = "fake-prometheus",
    ) -> TelemetrySnapshot:
        fetched_at = datetime.now(tz=timezone.utc)
        metrics: dict[str, RawMetricResult] = {}
        unknown: list[str] = []

        for mapping in registry.all_mappings():
            queries = [mapping.query] + mapping.fallback_queries
            result: Optional[RawMetricResult] = None

            for q in queries:
                r = self.query(expr=q, canonical_field=mapping.canonical_field, mapping=mapping)
                if not r.missing:
                    result = r
                    break

            if result is not None and not result.missing:
                metrics[mapping.canonical_field] = result
            else:
                unknown.append(mapping.canonical_field)

        return TelemetrySnapshot(
            source=source,
            fetched_at=fetched_at,
            is_sandbox=self.is_sandbox,
            metrics=metrics,
            unknown_metrics=unknown,
        )


def _apply_mapping_to_result(result: RawMetricResult, mapping: MetricMapping) -> RawMetricResult:
    """Apply unit conversion from a mapping to all values in a result."""
    converted_values = []
    for mv in result.values:
        converted_values.append(MetricValue(
            metric_name=mv.metric_name,
            labels=mv.labels,
            value=mapping.convert(mv.value),
            timestamp=mv.timestamp,
            raw_value_str=mv.raw_value_str,
        ))
    result.values.clear()
    result.values.extend(converted_values)
    return result


# ---------------------------------------------------------------------------
# PrometheusTelemetryConnector — orchestrates client + adapters → ClusterState
# ---------------------------------------------------------------------------

class PrometheusTelemetryConnector:
    """High-level connector: fetches and normalizes a TelemetrySnapshot.

    This connector accepts either a real PrometheusClient or a
    FakePrometheusClient (for sandbox/test mode). Adapters (DCGMAdapter,
    VLLMAdapter, etc.) then convert the snapshot into canonical state objects.

    Usage:
        config = ConnectorConfig(base_url="http://prometheus:9090")
        connector = PrometheusTelemetryConnector(
            client=PrometheusClient(config),
            registry=dcgm_registry(),
            source="dcgm-prod",
        )
        snapshot = connector.fetch_snapshot()
        # Then pass snapshot to DCGMAdapter.normalize_gpus(...)
    """

    def __init__(
        self,
        client: "PrometheusClient | FakePrometheusClient",
        registry: MetricMappingRegistry,
        source: str = "prometheus",
    ) -> None:
        self._client = client
        self._registry = registry
        self._source = source

    @property
    def is_sandbox(self) -> bool:
        if isinstance(self._client, FakePrometheusClient):
            return self._client.is_sandbox
        return self._client._config.is_sandbox

    def fetch_snapshot(self) -> TelemetrySnapshot:
        """Fetch all registered metrics and return a normalized snapshot."""
        return self._client.fetch_snapshot(self._registry, source=self._source)

    def scrape_snapshot(self, path: str = "/metrics") -> TelemetrySnapshot:
        """Scrape a raw /metrics endpoint and return a TelemetrySnapshot.

        Uses the raw Prometheus text format parser. Only metric_name keys
        are available (no canonical field mapping); useful for debugging
        and for connectors that expose direct /metrics endpoints.
        """
        fetched_at = datetime.now(tz=timezone.utc)
        raw = self._client.scrape_metrics(path=path)

        return TelemetrySnapshot(
            source=self._source,
            fetched_at=fetched_at,
            is_sandbox=self.is_sandbox,
            metrics={name: result for name, result in raw.items()},
            unknown_metrics=[],
            raw_text=None,
        )
