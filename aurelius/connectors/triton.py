"""NVIDIA Triton Inference Server Prometheus adapter for Aurelius.

Normalizes Triton /metrics into canonical InferenceServiceState objects.

Triton exposes Prometheus metrics by default at localhost:8002/metrics.
Metrics use the nv_ prefix. Latency statistics come from cumulative counters
(duration_us / exec_count) so derived p95/p99 are not directly available
from Triton's default Prometheus metrics without histograms enabled.

Field mapping from Triton → InferenceServiceState (using existing model fields):
  nv_inference_pending_request_count → requests_waiting
  nv_inference_queue_duration_us / exec_count → queue_time_p50_ms (average, not percentile)
  nv_inference_compute_duration_us / exec_count → p50_latency_ms (average, not percentile)
  nv_inference_request_failure / total → error_rate_pct

Missing from InferenceServiceState (Triton-specific fields not in model):
  - tokens_per_s (Triton is not LLM-specific by default)
  - ttft_* (LLM-specific Triton extensions only)
  - kv_cache_usage, prefix_cache_hit_rate (model-runtime specific)
  - p95/p99 latency (not in Triton default Prometheus metrics without histograms)

Reference:
  https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aurelius.connectors.base import TelemetrySnapshot
from aurelius.connectors.metric_mapping import MetricMappingRegistry, triton_registry
from aurelius.state.models import InferenceServiceState, Provenance

logger = logging.getLogger(__name__)

_F_INFER_COUNT = "triton.inference_count"
_F_EXEC_COUNT = "triton.inference_exec_count"
_F_QUEUE_US = "triton.inference_queue_duration_us"
_F_COMPUTE_US = "triton.inference_compute_duration_us"
_F_INPUT_US = "triton.inference_input_duration_us"
_F_OUTPUT_US = "triton.inference_output_duration_us"
_F_SUCCESS = "triton.request_success"
_F_FAILURE = "triton.request_failure"
_F_PENDING = "triton.pending_request_count"
_F_GPU_UTIL = "triton.gpu_utilization_pct"
_F_GPU_MEM_USED = "triton.gpu_memory_used_bytes"
_F_GPU_MEM_TOTAL = "triton.gpu_memory_total_bytes"


def _get_scalar(snapshot: TelemetrySnapshot, field: str, **label_filters: str) -> Optional[float]:
    result = snapshot.get(field)
    if result is None or result.missing:
        return None
    if label_filters:
        return result.value_for_labels(**label_filters)
    return result.first_value


def _derive_avg_latency_ms(
    total_duration_us: Optional[float],
    exec_count: Optional[float],
) -> Optional[float]:
    """Derive average latency from cumulative counters (us → ms, not a percentile)."""
    if total_duration_us is None or exec_count is None or exec_count <= 0:
        return None
    return (total_duration_us / exec_count) / 1000.0  # µs → ms


def _derive_error_rate_pct(
    success: Optional[float],
    failure: Optional[float],
) -> Optional[float]:
    if success is None or failure is None:
        return None
    total = success + failure
    if total <= 0:
        return 0.0
    return min(100.0, 100.0 * failure / total)


class TritonAdapter:
    """Normalizes Triton Prometheus metrics → InferenceServiceState objects.

    One InferenceServiceState is produced per (model, version) label pair.

    Note: latency fields are averages (not percentiles) since Triton's default
    metrics don't expose histogram buckets for queue/compute duration.
    p50_latency_ms is set to the cumulative average; p95/p99 remain None.
    """

    def __init__(self, registry: Optional[MetricMappingRegistry] = None) -> None:
        self._registry = registry or triton_registry()

    def all_models(self, snapshot: TelemetrySnapshot) -> list[tuple[str, str]]:
        """Extract all unique (model, version) tuples from the snapshot."""
        for field_name in [_F_INFER_COUNT, _F_SUCCESS, _F_PENDING]:
            result = snapshot.get(field_name)
            if result and not result.missing:
                pairs: list[tuple[str, str]] = []
                for mv in result.values:
                    model = mv.labels.get("model", "")
                    version = mv.labels.get("version", "1")
                    pair = (model, version)
                    if model and pair not in pairs:
                        pairs.append(pair)
                if pairs:
                    return pairs
        return []

    def normalize_inference_state(
        self,
        snapshot: TelemetrySnapshot,
        service_id: str,
        model: str = "",
        version: str = "1",
        timestamp: Optional[datetime] = None,
    ) -> InferenceServiceState:
        """Normalize Triton metrics for a single model/version into InferenceServiceState."""
        ts = timestamp or snapshot.fetched_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        lf: dict[str, str] = {}
        if model:
            lf = {"model": model, "version": version}

        coverage = snapshot.coverage_pct()
        provenance = Provenance(
            source=snapshot.source,
            fetched_at=snapshot.fetched_at,
            confidence="high" if coverage > 70 else "medium" if coverage > 40 else "low",
            is_sandbox=snapshot.is_sandbox,
        )

        exec_count = _get_scalar(snapshot, _F_EXEC_COUNT, **lf)
        queue_us = _get_scalar(snapshot, _F_QUEUE_US, **lf)
        compute_us = _get_scalar(snapshot, _F_COMPUTE_US, **lf)
        success = _get_scalar(snapshot, _F_SUCCESS, **lf)
        failure = _get_scalar(snapshot, _F_FAILURE, **lf)

        # Average latency from cumulative counters; not a percentile estimate
        avg_queue_ms = _derive_avg_latency_ms(queue_us, exec_count)
        avg_compute_ms = _derive_avg_latency_ms(compute_us, exec_count)
        avg_total_ms: Optional[float] = None
        if avg_queue_ms is not None and avg_compute_ms is not None:
            avg_total_ms = avg_queue_ms + avg_compute_ms
        elif avg_compute_ms is not None:
            avg_total_ms = avg_compute_ms

        return InferenceServiceState(
            service_id=service_id,
            engine="triton",
            timestamp=ts,
            provenance=provenance,
            requests_waiting=_get_scalar(snapshot, _F_PENDING, **lf),
            # p50 is the average (not p50); p95/p99 not available from default Triton metrics
            p50_latency_ms=avg_total_ms,
            p95_latency_ms=None,
            p99_latency_ms=None,
            queue_time_p50_ms=avg_queue_ms,
            error_rate_pct=_derive_error_rate_pct(success, failure),
        )

    def normalize_all_services(
        self,
        snapshot: TelemetrySnapshot,
        service_id_prefix: str = "triton",
        timestamp: Optional[datetime] = None,
    ) -> list[InferenceServiceState]:
        """Normalize all (model, version) services found in a snapshot."""
        models = self.all_models(snapshot)
        if not models:
            logger.info(
                "TritonAdapter: no model labels found in snapshot from %s",
                snapshot.source,
            )
            svc = self.normalize_inference_state(
                snapshot=snapshot,
                service_id=service_id_prefix,
                timestamp=timestamp,
            )
            return [svc]

        states = []
        for model, version in models:
            service_id = f"{service_id_prefix}/{model}/{version}"
            try:
                svc = self.normalize_inference_state(
                    snapshot=snapshot,
                    service_id=service_id,
                    model=model,
                    version=version,
                    timestamp=timestamp,
                )
                states.append(svc)
            except Exception as exc:
                logger.warning(
                    "TritonAdapter: failed to normalize model %s v%s: %s",
                    model, version, exc,
                )

        return states
