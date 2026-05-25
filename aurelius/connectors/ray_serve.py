"""Ray Serve Prometheus metrics adapter for Aurelius.

Normalizes Ray Serve Prometheus metrics into canonical InferenceServiceState.

Field mapping from Ray Serve → InferenceServiceState (using existing model fields):
  ray_serve_deployment_queued_queries    → requests_waiting
  ray_serve_request_latency_ms p50/p95/p99 → p50/p95/p99_latency_ms
  error rate derived from http_requests  → error_rate_pct
  ray_serve_deployment_replica_count     → replicas

Missing from InferenceServiceState (Ray Serve default metrics):
  - ttft_* (LLM-specific, not in standard Ray Serve metrics)
  - kv_cache_usage, prefix_cache_hit_rate (model-runtime specific)
  - tokens_per_s (not exposed by Ray Serve by default)

Reference:
  https://docs.ray.io/en/latest/serve/production-guide/monitoring.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aurelius.connectors.base import TelemetrySnapshot
from aurelius.connectors.metric_mapping import MetricMappingRegistry, ray_serve_registry
from aurelius.state.models import InferenceServiceState, Provenance

logger = logging.getLogger(__name__)

_F_REPLICAS = "ray.serve.num_replicas"
_F_RPS = "ray.serve.requests_per_second"
_F_QUEUE = "ray.serve.queue_len"
_F_LAT_P50 = "ray.serve.request_latency_p50_ms"
_F_LAT_P95 = "ray.serve.request_latency_p95_ms"
_F_LAT_P99 = "ray.serve.request_latency_p99_ms"
_F_ERROR_RATE = "ray.serve.error_rate_pct"


def _get_scalar(snapshot: TelemetrySnapshot, field: str, **label_filters: str) -> Optional[float]:
    result = snapshot.get(field)
    if result is None or result.missing:
        return None
    if label_filters:
        return result.value_for_labels(**label_filters)
    return result.first_value


def _clamp_pct(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, min(100.0, v))


def _clamp_non_negative(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, v)


class RayServeAdapter:
    """Normalizes Ray Serve Prometheus metrics → InferenceServiceState objects.

    One InferenceServiceState per deployment label.
    """

    def __init__(self, registry: Optional[MetricMappingRegistry] = None) -> None:
        self._registry = registry or ray_serve_registry()

    def all_deployments(self, snapshot: TelemetrySnapshot) -> list[str]:
        """Extract all unique deployment labels from the snapshot."""
        for field_name in [_F_REPLICAS, _F_RPS, _F_QUEUE]:
            result = snapshot.get(field_name)
            if result and not result.missing:
                deployments = []
                for mv in result.values:
                    dep = mv.labels.get("deployment")
                    if dep and dep not in deployments:
                        deployments.append(dep)
                if deployments:
                    return deployments
        return []

    def normalize_inference_state(
        self,
        snapshot: TelemetrySnapshot,
        service_id: str,
        deployment: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> InferenceServiceState:
        """Normalize Ray Serve metrics for a single deployment."""
        ts = timestamp or snapshot.fetched_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        lf: dict[str, str] = {}
        if deployment:
            lf = {"deployment": deployment}

        coverage = snapshot.coverage_pct()
        provenance = Provenance(
            source=snapshot.source,
            fetched_at=snapshot.fetched_at,
            confidence="high" if coverage > 70 else "medium" if coverage > 40 else "low",
            is_sandbox=snapshot.is_sandbox,
        )

        replicas_val = _get_scalar(snapshot, _F_REPLICAS, **lf)
        replicas_int: Optional[int] = int(replicas_val) if replicas_val is not None else None

        return InferenceServiceState(
            service_id=service_id,
            engine="ray_serve",
            timestamp=ts,
            provenance=provenance,
            requests_waiting=_get_scalar(snapshot, _F_QUEUE, **lf),
            p50_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_LAT_P50, **lf)),
            p95_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_LAT_P95, **lf)),
            p99_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_LAT_P99, **lf)),
            error_rate_pct=_clamp_pct(_get_scalar(snapshot, _F_ERROR_RATE, **lf)),
            replicas=replicas_int,
        )

    def normalize_all_services(
        self,
        snapshot: TelemetrySnapshot,
        service_id_prefix: str = "ray",
        timestamp: Optional[datetime] = None,
    ) -> list[InferenceServiceState]:
        """Normalize all deployment services found in a snapshot."""
        deployments = self.all_deployments(snapshot)
        if not deployments:
            logger.info(
                "RayServeAdapter: no deployment labels found in snapshot from %s",
                snapshot.source,
            )
            svc = self.normalize_inference_state(
                snapshot=snapshot,
                service_id=service_id_prefix,
                deployment=None,
                timestamp=timestamp,
            )
            return [svc]

        states = []
        for dep in deployments:
            service_id = f"{service_id_prefix}/{dep}"
            try:
                svc = self.normalize_inference_state(
                    snapshot=snapshot,
                    service_id=service_id,
                    deployment=dep,
                    timestamp=timestamp,
                )
                states.append(svc)
            except Exception as exc:
                logger.warning(
                    "RayServeAdapter: failed to normalize deployment %s: %s",
                    dep, exc,
                )

        return states
