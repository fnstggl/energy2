"""Kubernetes API client — read-only.

Supports:
- In-cluster service account token
- Kubeconfig path
- Direct base URL with bearer token
- Fully offline sandbox mode via fixture injection

IMPORTANT: This client never writes to the cluster.
Only GET requests are issued.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional


class KubernetesClientError(Exception):
    pass


class KubernetesClient:
    """Read-only Kubernetes API client.

    Parameters
    ----------
    base_url:
        Kubernetes API server URL.
    token:
        Bearer token for authentication.
    ca_cert_path:
        Path to CA certificate for TLS verification.
    tls_verify:
        Whether to verify TLS certificates.
    namespace_allowlist:
        If non-empty, limit pod queries to these namespaces.
    _sandbox_responses:
        Fixture dict {path: json_response} for offline testing.
    """

    def __init__(
        self,
        base_url: str = "https://kubernetes.default.svc",
        token: Optional[str] = None,
        ca_cert_path: Optional[str] = None,
        tls_verify: bool = True,
        namespace_allowlist: Optional[list[str]] = None,
        _sandbox_responses: Optional[dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._ca_cert_path = ca_cert_path
        self._tls_verify = tls_verify
        self._namespace_allowlist = namespace_allowlist or []
        self._sandbox = _sandbox_responses

    @classmethod
    def from_in_cluster(cls) -> "KubernetesClient":
        """Build a client from in-cluster service account credentials."""
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        base_url = f"https://{host}:{port}"
        token = None
        if os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
        return cls(
            base_url=base_url,
            token=token,
            ca_cert_path=ca_path if os.path.exists(ca_path) else None,
        )

    def list_nodes(self) -> dict[str, Any]:
        """List all cluster nodes."""
        return self._get("/api/v1/nodes")

    def list_pods(self, namespace: Optional[str] = None) -> dict[str, Any]:
        """List pods in a namespace (or all namespaces if namespace=None)."""
        if namespace:
            path = f"/api/v1/namespaces/{namespace}/pods"
        else:
            path = "/api/v1/pods"
        return self._get(path)

    def list_pods_all_allowed(self) -> dict[str, Any]:
        """List pods from all allowed namespaces, merged into one list."""
        if not self._namespace_allowlist:
            return self.list_pods()

        all_items: list[Any] = []
        for ns in self._namespace_allowlist:
            result = self.list_pods(ns)
            all_items.extend(result.get("items", []))

        return {
            "apiVersion": "v1",
            "kind": "PodList",
            "items": all_items,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        if self._sandbox is not None:
            return self._sandbox_get(path)
        return self._live_get(path)

    def _sandbox_get(self, path: str) -> dict[str, Any]:
        assert self._sandbox is not None
        if path in self._sandbox:
            data = self._sandbox[path]
            if isinstance(data, str):
                return json.loads(data)
            return data
        raise KubernetesClientError(
            f"Sandbox: no fixture for path {path!r}. "
            f"Available: {list(self._sandbox.keys())}"
        )

    def _live_get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            raise KubernetesClientError(
                f"HTTP {exc.code} from {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise KubernetesClientError(
                f"Connection failed to {url}: {exc.reason}"
            ) from exc
