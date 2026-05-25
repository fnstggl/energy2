"""High-level Prometheus telemetry connector.

Orchestrates the Prometheus client and metric mappings to produce a
TelemetrySnapshot — a partial ClusterState populated from Prometheus data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import (
    ClusterState,
    GPUState,
    InferenceServiceState,
    Provenance,
)
from ...state.normalize import normalize_gpu_state, normalize_inference_service
from .client import PrometheusClient, PrometheusQueryError
from .mappings import DEFAULT_DCGM_MAPPINGS, DEFAULT_VLLM_MAPPINGS, MappingRegistry


@dataclass
class TelemetrySnapshot:
    """Partial ClusterState derived from Prometheus telemetry.

    Connectors populate whichever fields are available. Missing fields
    remain None so the state layer can distinguish unknown from zero.
    """

    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cluster_id: str = "default"
    gpus: dict[str, GPUState] = field(default_factory=dict)
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_results: dict[str, Any] = field(default_factory=dict)

    def to_cluster_state(self) -> ClusterState:
        """Convert to a minimal ClusterState with only collected fields."""
        return ClusterState(
            timestamp=self.collected_at,
            cluster_id=self.cluster_id,
            gpus=self.gpus,
            services=self.services,
        )


class PrometheusTelemetryConnector:
    """Fetches telemetry from Prometheus and normalizes into ClusterState.

    Parameters
    ----------
    client:
        PrometheusClient instance (may be sandbox).
    gpu_mapping:
        MappingRegistry for GPU metrics (defaults to DCGM mappings).
    inference_mapping:
        MappingRegistry for inference service metrics (defaults to vLLM).
    cluster_id:
        Identifier for the cluster being monitored.
    """

    def __init__(
        self,
        client: PrometheusClient,
        gpu_mapping: Optional[MappingRegistry] = None,
        inference_mapping: Optional[MappingRegistry] = None,
        cluster_id: str = "default",
    ) -> None:
        self.client = client
        self.gpu_mapping = gpu_mapping or DEFAULT_DCGM_MAPPINGS
        self.inference_mapping = inference_mapping or DEFAULT_VLLM_MAPPINGS
        self.cluster_id = cluster_id

    def fetch_cluster_state(self) -> TelemetrySnapshot:
        """Query Prometheus and return a TelemetrySnapshot.

        Missing metrics are recorded in snapshot.unknown_metrics.
        Errors are recorded in snapshot.errors and do not raise.
        """
        snap = TelemetrySnapshot(
            collected_at=datetime.now(timezone.utc),
            cluster_id=self.cluster_id,
        )
        self._collect_gpu_metrics(snap)
        self._collect_inference_metrics(snap)
        return snap

    def _collect_gpu_metrics(self, snap: TelemetrySnapshot) -> None:
        """Query all GPU mapping fields and group by GPU identifier."""
        gpu_raw: dict[str, dict[str, Any]] = {}

        for mapping in self.gpu_mapping.all_mappings():
            result = self._query_first_available(mapping.queries, snap)
            if result is None:
                snap.unknown_metrics.append(mapping.canonical_field)
                continue

            snap.raw_results[mapping.canonical_field] = result
            for series in self._iter_vector_series(result):
                gpu_key = self._extract_gpu_key(series.get("metric", {}))
                if gpu_key not in gpu_raw:
                    gpu_raw[gpu_key] = {"gpu_id": gpu_key}
                    gpu_raw[gpu_key].update(self._extract_node_labels(series.get("metric", {})))

                short_field = mapping.canonical_field.split(".", 1)[-1]
                value = self._extract_value(series, mapping.unit_conversion)
                gpu_raw[gpu_key][short_field] = value

        for gpu_key, raw in gpu_raw.items():
            prov = Provenance(
                source=f"prometheus:{self.gpu_mapping.name}",
                collected_at=snap.collected_at,
                confidence=1.0 if not snap.unknown_metrics else 0.7,
            )
            try:
                snap.gpus[gpu_key] = normalize_gpu_state(raw, prov)
            except Exception as exc:
                snap.errors.append(f"GPU {gpu_key} normalization failed: {exc}")

    def _collect_inference_metrics(self, snap: TelemetrySnapshot) -> None:
        """Query all inference mapping fields and group by service identifier."""
        svc_raw: dict[str, dict[str, Any]] = {}

        for mapping in self.inference_mapping.all_mappings():
            result = self._query_first_available(mapping.queries, snap)
            if result is None:
                snap.unknown_metrics.append(mapping.canonical_field)
                continue

            snap.raw_results[mapping.canonical_field] = result
            for series in self._iter_vector_series(result):
                svc_key = self._extract_service_key(series.get("metric", {}))
                if svc_key not in svc_raw:
                    svc_raw[svc_key] = {
                        "service_id": svc_key,
                        "runtime": self.inference_mapping.name,
                    }

                short_field = mapping.canonical_field.split(".", 1)[-1]
                value = self._extract_value(series, mapping.unit_conversion)
                svc_raw[svc_key][short_field] = value

        for svc_key, raw in svc_raw.items():
            prov = Provenance(
                source=f"prometheus:{self.inference_mapping.name}",
                collected_at=snap.collected_at,
                confidence=1.0 if not snap.unknown_metrics else 0.7,
            )
            try:
                snap.services[svc_key] = normalize_inference_service(raw, prov)
            except Exception as exc:
                snap.errors.append(f"Service {svc_key} normalization failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _query_first_available(
        self,
        queries: list[str],
        snap: TelemetrySnapshot,
    ) -> Optional[dict[str, Any]]:
        """Try each query in order; return first successful non-empty result."""
        for query in queries:
            try:
                result = self.client.query(query)
                if self._has_data(result):
                    return result
            except PrometheusQueryError as exc:
                snap.errors.append(f"Query failed ({query!r}): {exc}")
        return None

    def _has_data(self, result: dict[str, Any]) -> bool:
        data = result.get("data", {})
        return bool(data.get("result"))

    def _iter_vector_series(self, result: dict[str, Any]):
        """Yield vector result series from a Prometheus instant query response."""
        data = result.get("data", {})
        result_type = data.get("resultType", "")
        if result_type == "vector":
            yield from data.get("result", [])
        elif result_type == "matrix":
            for series in data.get("result", []):
                values = series.get("values", [])
                if values:
                    yield {"metric": series.get("metric", {}), "value": values[-1]}

    def _extract_gpu_key(self, labels: dict[str, str]) -> str:
        return (
            labels.get("UUID")
            or labels.get("uuid")
            or labels.get("gpu")
            or labels.get("GPU_I_ID")
            or "gpu_unknown"
        )

    def _extract_service_key(self, labels: dict[str, str]) -> str:
        return (
            labels.get("model_name")
            or labels.get("deployment")
            or labels.get("model")
            or labels.get("service")
            or "service_unknown"
        )

    def _extract_node_labels(self, labels: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "node" in labels:
            result["node_id"] = labels["node"]
        if "UUID" in labels:
            result["uuid"] = labels["UUID"]
        return result

    def _extract_value(self, series: dict[str, Any], unit_conversion: float) -> Optional[float]:
        value_pair = series.get("value")
        if not value_pair:
            return None
        try:
            raw_val = float(value_pair[1])
            return raw_val * unit_conversion
        except (TypeError, ValueError, IndexError):
            return None
