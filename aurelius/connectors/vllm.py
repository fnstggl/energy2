"""vLLM Prometheus metrics adapter for Aurelius.

Normalizes vLLM /metrics into canonical InferenceServiceState objects.

Field mapping from vLLM → InferenceServiceState (using existing model field names):
  vllm:num_requests_running         → requests_running
  vllm:num_requests_waiting         → requests_waiting
  vllm:generation_tokens_total rate → tokens_per_s
  vllm:gpu_cache_usage_perc         → kv_cache_usage  [0-1 fraction, NOT percent]
  vllm:gpu_prefix_cache_hit_rate    → prefix_cache_hit_rate  [0-1 fraction]
  vllm:request_first_token_seconds  → ttft_p50/p95/p99_ms
  vllm:e2e_request_latency_seconds  → p50/p95/p99_latency_ms

vLLM V0→V1 metric naming change (both supported via fallback_queries):
  V0: vllm_* (underscore namespace)
  V1: vllm:* (colon namespace)

Reference:
  https://docs.vllm.ai/en/stable/serving/metrics.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aurelius.connectors.base import TelemetrySnapshot
from aurelius.connectors.metric_mapping import MetricMappingRegistry, vllm_registry
from aurelius.state.models import InferenceServiceState, Provenance

logger = logging.getLogger(__name__)

# Canonical field names (matches metric_mapping._VLLM_BUILTIN keys)
_F_RPS = "inference.requests_per_second"
_F_TPS = "inference.tokens_per_second"
_F_TTFT_P50 = "inference.ttft_p50_ms"
_F_TTFT_P95 = "inference.ttft_p95_ms"
_F_TTFT_P99 = "inference.ttft_p99_ms"
_F_TPOT_P50 = "inference.tpot_p50_ms"
_F_TPOT_P95 = "inference.tpot_p95_ms"
_F_TPOT_P99 = "inference.tpot_p99_ms"
_F_E2E_P50 = "inference.e2e_p50_ms"
_F_E2E_P95 = "inference.e2e_p95_ms"
_F_E2E_P99 = "inference.e2e_p99_ms"
_F_QUEUE_DEPTH = "inference.queue_depth"
_F_ACTIVE_SEQ = "inference.active_sequences"
_F_KV_CACHE = "inference.kv_cache_usage_pct"
_F_PREFIX_HIT = "inference.prefix_cache_hit_rate_pct"


def _get_scalar(snapshot: TelemetrySnapshot, field: str, **label_filters: str) -> Optional[float]:
    result = snapshot.get(field)
    if result is None or result.missing:
        return None
    if label_filters:
        return result.value_for_labels(**label_filters)
    return result.first_value


def _clamp_fraction(v: Optional[float]) -> Optional[float]:
    """Clamp a fraction to [0, 1].

    vLLM gpu_cache_usage_perc and prefix_cache_hit_rate are already 0-1 fractions.
    The metric_mapping may multiply by 100 for display; InferenceServiceState stores 0-1.
    """
    if v is None:
        return None
    # If the mapping emitted a percentage (0-100), convert back to fraction
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _clamp_non_negative(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, v)


class VLLMAdapter:
    """Normalizes vLLM Prometheus metrics → InferenceServiceState objects.

    One InferenceServiceState is produced per model_name label (vLLM deployment).

    Uses the existing InferenceServiceState model fields:
      engine="vllm", requests_running, requests_waiting,
      tokens_per_s, ttft_*_ms, p*_latency_ms,
      kv_cache_usage [0-1], prefix_cache_hit_rate [0-1]
    """

    def __init__(self, registry: Optional[MetricMappingRegistry] = None) -> None:
        self._registry = registry or vllm_registry()

    def all_model_names(self, snapshot: TelemetrySnapshot) -> list[str]:
        """Extract all unique model_name labels from the snapshot."""
        for field_name in [_F_ACTIVE_SEQ, _F_QUEUE_DEPTH, _F_KV_CACHE, _F_RPS]:
            result = snapshot.get(field_name)
            if result and not result.missing:
                models = []
                for mv in result.values:
                    name = mv.labels.get("model_name")
                    if name and name not in models:
                        models.append(name)
                if models:
                    return models
        return []

    def normalize_inference_state(
        self,
        snapshot: TelemetrySnapshot,
        service_id: str,
        model_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> InferenceServiceState:
        """Normalize vLLM metrics for a single service/model.

        If model_name is None, uses aggregate values (single-model deployments).
        """
        ts = timestamp or snapshot.fetched_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        lf: dict[str, str] = {}
        if model_name is not None:
            lf = {"model_name": model_name}

        coverage = snapshot.coverage_pct()
        provenance = Provenance(
            source=snapshot.source,
            fetched_at=snapshot.fetched_at,
            confidence="high" if coverage > 70 else "medium" if coverage > 40 else "low",
            is_sandbox=snapshot.is_sandbox,
        )

        # kv_cache_usage_pct mapping emits pct (×100); convert back to 0-1 fraction
        kv_raw = _get_scalar(snapshot, _F_KV_CACHE, **lf)
        prefix_raw = _get_scalar(snapshot, _F_PREFIX_HIT, **lf)

        return InferenceServiceState(
            service_id=service_id,
            engine="vllm",
            timestamp=ts,
            provenance=provenance,
            tokens_per_s=_clamp_non_negative(_get_scalar(snapshot, _F_TPS, **lf)),
            requests_running=_clamp_non_negative(_get_scalar(snapshot, _F_ACTIVE_SEQ, **lf)),
            requests_waiting=_clamp_non_negative(_get_scalar(snapshot, _F_QUEUE_DEPTH, **lf)),
            ttft_p50_ms=_clamp_non_negative(_get_scalar(snapshot, _F_TTFT_P50, **lf)),
            ttft_p95_ms=_clamp_non_negative(_get_scalar(snapshot, _F_TTFT_P95, **lf)),
            ttft_p99_ms=_clamp_non_negative(_get_scalar(snapshot, _F_TTFT_P99, **lf)),
            p50_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_E2E_P50, **lf)),
            p95_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_E2E_P95, **lf)),
            p99_latency_ms=_clamp_non_negative(_get_scalar(snapshot, _F_E2E_P99, **lf)),
            kv_cache_usage=_clamp_fraction(kv_raw),
            prefix_cache_hit_rate=_clamp_fraction(prefix_raw),
        )

    def normalize_all_services(
        self,
        snapshot: TelemetrySnapshot,
        service_id_prefix: str = "vllm",
        timestamp: Optional[datetime] = None,
    ) -> list[InferenceServiceState]:
        """Normalize all model_name services found in a snapshot."""
        models = self.all_model_names(snapshot)
        if not models:
            logger.info(
                "VLLMAdapter: no model_name labels found in snapshot from %s",
                snapshot.source,
            )
            svc = self.normalize_inference_state(
                snapshot=snapshot,
                service_id=service_id_prefix,
                model_name=None,
                timestamp=timestamp,
            )
            return [svc]

        states = []
        for model in models:
            service_id = f"{service_id_prefix}/{model}"
            try:
                svc = self.normalize_inference_state(
                    snapshot=snapshot,
                    service_id=service_id,
                    model_name=model,
                    timestamp=timestamp,
                )
                states.append(svc)
            except Exception as exc:
                logger.warning("VLLMAdapter: failed to normalize model %s: %s", model, exc)

        return states
