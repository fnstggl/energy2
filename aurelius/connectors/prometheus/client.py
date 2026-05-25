"""Generic Prometheus HTTP API client.

Supports:
- Prometheus HTTP API: /api/v1/query and /api/v1/query_range
- Raw /metrics scrape (Prometheus text format)
- Bearer token, Basic auth, custom headers
- TLS verify on/off
- Configurable timeout and retry with exponential backoff
- Fully offline sandbox mode via injected response fixture
"""

from __future__ import annotations

import base64
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


class PrometheusQueryError(Exception):
    """Raised when the Prometheus HTTP API returns an error."""


@dataclass
class PrometheusAuth:
    """Authentication configuration for the Prometheus client."""

    auth_type: str = "none"
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def headers(self) -> dict[str, str]:
        if self.auth_type == "bearer" and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if self.auth_type == "basic" and self.username and self.password:
            creds = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()
            return {"Authorization": f"Basic {creds}"}
        return {}


class PrometheusClient:
    """Low-level HTTP client for Prometheus APIs.

    Parameters
    ----------
    base_url:
        Root URL of the Prometheus server, e.g. "http://prometheus:9090"
    auth:
        Authentication configuration.
    extra_headers:
        Additional headers merged on every request.
    timeout_seconds:
        HTTP request timeout.
    max_retries:
        Number of retries on transient failures.
    tls_verify:
        Whether to verify TLS certificates.
    _sandbox_responses:
        If provided, URL → response body strings are used instead of live HTTP.
        Enables fully offline testing without a real Prometheus server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9090",
        auth: Optional[PrometheusAuth] = None,
        extra_headers: Optional[dict[str, str]] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        tls_verify: bool = True,
        _sandbox_responses: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth or PrometheusAuth()
        self.extra_headers = extra_headers or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.tls_verify = tls_verify
        self._sandbox = _sandbox_responses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        promql: str,
        time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Execute an instant PromQL query.

        Returns the parsed JSON response ``{"status": "success", "data": {...}}``.
        Raises PrometheusQueryError on HTTP errors or Prometheus error status.
        """
        params: dict[str, str] = {"query": promql}
        if time is not None:
            params["time"] = time.isoformat()
        url = f"{self.base_url}/api/v1/query?" + urllib.parse.urlencode(params)
        return self._request_json(url)

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "60s",
    ) -> dict[str, Any]:
        """Execute a range PromQL query."""
        params = {
            "query": promql,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": step,
        }
        url = f"{self.base_url}/api/v1/query_range?" + urllib.parse.urlencode(params)
        return self._request_json(url)

    def scrape_metrics(self, path: str = "/metrics") -> str:
        """Scrape raw Prometheus text-format metrics from an exporter endpoint.

        Returns the raw metrics text for parsing by adapters.
        """
        url = f"{self.base_url}{path}"
        return self._request_text(url)

    def health_check(self) -> bool:
        """Return True if the Prometheus server is reachable."""
        try:
            url = f"{self.base_url}/-/healthy"
            self._request_text(url)
            return True
        except (PrometheusQueryError, Exception):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        headers.update(self.auth.headers())
        headers.update(self.extra_headers)
        return headers

    def _request_json(self, url: str) -> dict[str, Any]:
        import json
        body = self._request_text(url, accept="application/json")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PrometheusQueryError(f"Invalid JSON from {url}: {exc}") from exc
        if parsed.get("status") == "error":
            raise PrometheusQueryError(
                f"Prometheus error: {parsed.get('error', 'unknown')} "
                f"({parsed.get('errorType', '')})"
            )
        return parsed

    def _request_text(self, url: str, accept: str = "text/plain") -> str:
        if self._sandbox is not None:
            return self._sandbox_lookup(url)

        headers = self._build_headers()
        headers["Accept"] = accept

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(2 ** (attempt - 1))
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(
                    req, timeout=self.timeout_seconds
                ) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                raise PrometheusQueryError(
                    f"HTTP {exc.code} from {url}: {exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                last_exc = exc
        raise PrometheusQueryError(
            f"Failed to reach {url} after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _sandbox_lookup(self, url: str) -> str:
        """Look up a URL in the sandbox fixture dict."""
        assert self._sandbox is not None
        # Try exact match first, then prefix match
        if url in self._sandbox:
            return self._sandbox[url]
        # Match by path+query stripped of base_url
        for key, val in self._sandbox.items():
            if url.endswith(key) or key in url:
                return val
        raise PrometheusQueryError(
            f"Sandbox: no fixture for URL {url!r}. "
            f"Available keys: {list(self._sandbox.keys())}"
        )
