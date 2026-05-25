"""vLLM Prometheus metrics adapter.

Parses raw Prometheus text-format /metrics from a vLLM server and
normalizes into InferenceServiceState.

Reference: vLLM exposes metrics at /metrics on the OpenAI-compatible server.
Key metrics: vllm:gpu_cache_usage_perc, vllm:num_requests_running, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ...state.models import InferenceServiceState, Provenance
from ...state.normalize import normalize_inference_service

# Map vLLM metric name → canonical InferenceServiceState field
_VLLM_FIELD_MAP: dict[str, tuple[str, float]] = {
    # (canonical_field, multiplier)
    "vllm:num_requests_running": ("active_sequences", 1.0),
    "vllm:num_requests_waiting": ("queue_depth", 1.0),
    "vllm:gpu_cache_usage_perc": ("kv_cache_usage_pct", 100.0),
    "vllm:prefix_cache_hit_rate": ("prefix_cache_hit_rate_pct", 100.0),
    "vllm:avg_generation_throughput_toks_per_s": ("tokens_per_second", 1.0),
    "vllm:avg_prompt_throughput_toks_per_s": ("_prompt_toks_per_s", 1.0),
    # Histogram summaries — handled separately via _parse_histogram
    "vllm:request_success_total": ("_request_success_total", 1.0),
    "vllm:request_failure_total": ("_request_failure_total", 1.0),
}

# Histogram metrics that we'll extract _sum and _count from to compute rates
_HISTOGRAM_PREFIXES: list[str] = [
    "vllm:e2e_request_latency_seconds",
    "vllm:request_first_token_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:time_to_first_token_seconds",
]


@dataclass
class VLLMParseResult:
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    unknown_metrics: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class VLLMAdapter:
    """Parses vLLM Prometheus text metrics into InferenceServiceState objects.

    The adapter groups metrics by model_name label to support multi-model
    deployments on the same server.

    Usage::

        adapter = VLLMAdapter()
        result = adapter.parse_text(metrics_text)
        for svc_id, service in result.services.items():
            ...
    """

    def parse_text(
        self,
        metrics_text: str,
        source: str = "vllm",
    ) -> VLLMParseResult:
        result = VLLMParseResult()
        collected_at = datetime.now(timezone.utc)

        # svc_key → raw dict
        svc_raw: dict[str, dict[str, Any]] = {}
        seen_metrics: set[str] = set()

        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                continue

            name, labels, value = parsed
            seen_metrics.add(name)
            svc_key = self._svc_key(labels)

            if svc_key not in svc_raw:
                svc_raw[svc_key] = {
                    "service_id": svc_key,
                    "runtime": "vllm",
                }

            self._apply_metric(name, labels, value, svc_raw[svc_key], result)

        # Derive percentages from raw fractions
        for raw in svc_raw.values():
            for frac_field, pct_field in [
                ("kv_cache_usage_pct", "kv_cache_usage_pct"),
                ("prefix_cache_hit_rate_pct", "prefix_cache_hit_rate_pct"),
            ]:
                v = raw.get(frac_field)
                if v is not None and v <= 1.0:
                    raw[frac_field] = v * 100.0

        for svc_key, raw in svc_raw.items():
            prov = Provenance(
                source=source,
                collected_at=collected_at,
                confidence=1.0 if not result.unknown_metrics else 0.8,
            )
            try:
                result.services[svc_key] = normalize_inference_service(raw, prov)
            except Exception as exc:
                result.parse_errors.append(f"Service {svc_key}: {exc}")

        return result

    def _apply_metric(
        self,
        name: str,
        labels: dict[str, str],
        value: float,
        raw: dict[str, Any],
        result: VLLMParseResult,
    ) -> None:
        if name in _VLLM_FIELD_MAP:
            canonical, mult = _VLLM_FIELD_MAP[name]
            if not canonical.startswith("_"):
                raw[canonical] = value * mult
        elif any(name.startswith(p) for p in _HISTOGRAM_PREFIXES):
            # Store histogram _sum and _count for potential mean computation
            for prefix in _HISTOGRAM_PREFIXES:
                if name == f"{prefix}_sum":
                    raw[f"_hist_{prefix}_sum"] = value
                elif name == f"{prefix}_count":
                    raw[f"_hist_{prefix}_count"] = value
        else:
            result.unknown_metrics.append(name)

    def _parse_line(
        self, line: str
    ) -> Optional[tuple[str, dict[str, str], float]]:
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)', line)
        if not match:
            return None
        name = match.group(1)
        labels_str = match.group(2) or ""
        value_str = match.group(3)
        try:
            value = float(value_str)
        except ValueError:
            return None
        labels: dict[str, str] = {}
        if labels_str:
            for kv in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[kv.group(1)] = kv.group(2)
        return name, labels, value

    def _svc_key(self, labels: dict[str, str]) -> str:
        return (
            labels.get("model_name")
            or labels.get("model")
            or labels.get("deployment")
            or "vllm_default"
        )
