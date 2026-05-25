"""Base types for Aurelius telemetry connectors.

Design rules:
- missing/unknown values → None, never fabricated
- auth secrets read from env, never stored in plain config
- connectors must work identically in sandbox (fixture) and real modes
- timestamps always UTC-aware
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class AuthType(str, Enum):
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"


@dataclass
class AuthConfig:
    """Authentication configuration for a connector.

    Secrets are read from environment variables at use time, not stored here.

    Attributes:
        type:        authentication scheme
        token_env:   env var name for bearer token (AuthType.BEARER)
        username:    basic-auth username (AuthType.BASIC)
        password_env: env var name for basic-auth password (AuthType.BASIC)
    """
    type: AuthType = AuthType.NONE
    token_env: Optional[str] = None
    username: Optional[str] = None
    password_env: Optional[str] = None

    def bearer_token(self) -> Optional[str]:
        if self.type != AuthType.BEARER or self.token_env is None:
            return None
        return os.environ.get(self.token_env)

    def basic_credentials(self) -> Optional[tuple[str, str]]:
        if self.type != AuthType.BASIC:
            return None
        if self.username is None or self.password_env is None:
            return None
        password = os.environ.get(self.password_env)
        if password is None:
            return None
        return (self.username, password)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuthConfig":
        auth_type = AuthType(d.get("type", "none"))
        return cls(
            type=auth_type,
            token_env=d.get("token_env"),
            username=d.get("username"),
            password_env=d.get("password_env"),
        )


@dataclass
class ConnectorConfig:
    """Configuration for a telemetry connector endpoint.

    Attributes:
        base_url:              Prometheus server or exporter URL
        auth:                  authentication config
        tls_verify:            verify TLS certificates (default True)
        timeout_s:             request timeout in seconds
        max_retries:           max retry attempts with exponential backoff
        scrape_interval_s:     expected scrape interval (for staleness detection)
        extra_headers:         additional HTTP headers (values NOT secrets)
        is_sandbox:            if True, connector is fixture/sandbox — excluded from production claims
        custom_labels:         extra labels attached to all metrics from this connector
    """
    base_url: str
    auth: AuthConfig = field(default_factory=AuthConfig)
    tls_verify: bool = True
    timeout_s: float = 30.0
    max_retries: int = 3
    scrape_interval_s: int = 30
    extra_headers: dict[str, str] = field(default_factory=dict)
    is_sandbox: bool = False
    custom_labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConnectorConfig":
        auth_dict = d.get("auth", {})
        return cls(
            base_url=d["base_url"],
            auth=AuthConfig.from_dict(auth_dict),
            tls_verify=d.get("tls_verify", True),
            timeout_s=float(d.get("timeout_s", 30.0)),
            max_retries=int(d.get("max_retries", 3)),
            scrape_interval_s=int(d.get("scrape_interval_s", 30)),
            extra_headers=d.get("extra_headers", {}),
            is_sandbox=d.get("is_sandbox", False),
            custom_labels=d.get("custom_labels", {}),
        )


# ---------------------------------------------------------------------------
# Metric value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricLabel:
    """A single label key-value pair from a Prometheus metric."""
    key: str
    value: str


@dataclass(frozen=True)
class MetricValue:
    """A single time-series point from a Prometheus result.

    Attributes:
        metric_name:    Prometheus metric name
        labels:         label set (dict for fast lookup)
        value:          float value (None if metric was missing/parse error)
        timestamp:      UTC-aware observation timestamp
        raw_value_str:  raw string from Prometheus (before unit conversion)
    """
    metric_name: str
    labels: dict[str, str]
    value: Optional[float]
    timestamp: datetime
    raw_value_str: str = ""

    def label(self, key: str) -> Optional[str]:
        return self.labels.get(key)


@dataclass
class RawMetricResult:
    """Result of a Prometheus query — may contain multiple time series.

    Attributes:
        metric_name:    name of the queried metric / expression
        query:          Prometheus query string used
        values:         list of MetricValues (one per label set)
        fetched_at:     UTC time when query was executed
        missing:        True if query returned no data (metric not available)
        error:          error message if query failed
    """
    metric_name: str
    query: str
    values: list[MetricValue] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    missing: bool = False
    error: Optional[str] = None

    @property
    def first_value(self) -> Optional[float]:
        """Return the first value if there is exactly one time series."""
        return self.values[0].value if self.values else None

    def value_for_labels(self, **label_filters: str) -> Optional[float]:
        """Find the value where all given labels match."""
        for mv in self.values:
            if all(mv.labels.get(k) == v for k, v in label_filters.items()):
                return mv.value
        return None

    def values_by_label(self, key: str) -> dict[str, Optional[float]]:
        """Group values by a label key → {label_value: metric_value}."""
        result: dict[str, Optional[float]] = {}
        for mv in self.values:
            lv = mv.labels.get(key)
            if lv is not None:
                result[lv] = mv.value
        return result


@dataclass
class TelemetrySnapshot:
    """Normalized collection of metric results from a single connector scrape.

    Attributes:
        source:          connector/adapter name
        fetched_at:      UTC timestamp of the scrape
        is_sandbox:      True if from simulator or fixture
        metrics:         canonical_field_name → RawMetricResult
        unknown_metrics: canonical fields that were attempted but missing
        raw_text:        raw /metrics text if scraped directly (not via API)
    """
    source: str
    fetched_at: datetime
    is_sandbox: bool = False
    metrics: dict[str, RawMetricResult] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    raw_text: Optional[str] = None

    def get(self, canonical_field: str) -> Optional[RawMetricResult]:
        return self.metrics.get(canonical_field)

    def value(self, canonical_field: str) -> Optional[float]:
        result = self.metrics.get(canonical_field)
        if result is None or result.missing:
            return None
        return result.first_value

    def value_for_labels(self, canonical_field: str, **label_filters: str) -> Optional[float]:
        result = self.metrics.get(canonical_field)
        if result is None or result.missing:
            return None
        return result.value_for_labels(**label_filters)

    def coverage_pct(self) -> float:
        """Fraction of attempted metrics that returned data."""
        total = len(self.metrics) + len(self.unknown_metrics)
        if total == 0:
            return 0.0
        return 100.0 * len(self.metrics) / total
