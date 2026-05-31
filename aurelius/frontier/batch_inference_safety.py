"""Batch Inference Frontier — safety filter.

Pre-registered, transparent thresholds — never folded into a KPI weight
(``docs/RESULTS.md`` §1-§2). A point that breaches any configured gate is
UNSAFE; a point that lacks the telemetry needed to evaluate a configured
gate is INSUFFICIENT_TELEMETRY (never auto-pass).

The v1 safety floor:

- ``max_deadline_miss_rate_pct`` — batch-class deadline-miss ceiling
  (default 2.0% — conservative). The user-spec safety constraint.
- ``max_timeout_pct`` — the existing serving timeout ceiling (default 10%).
- ``max_queue_p99_ms`` — existing serving queue ceiling (default 2000 ms).
- ``no_sla_regression_vs_interactive_baseline`` — when an interactive
  baseline p99 is provided, the candidate's queue + timeout MUST stay
  inside the baseline; the gate is otherwise inactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .batch_inference_models import (
    BatchInferenceFrontierPoint,
    BatchInferenceSafetyStatus,
    BatchInferenceWorkloadProfile,
)

_BATCH_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class BatchInferenceSafetyConfig:
    max_deadline_miss_rate_pct: Optional[float] = 2.0
    max_timeout_pct: Optional[float] = 10.0
    max_queue_p99_ms: Optional[float] = 2000.0
    max_queue_p95_ms: Optional[float] = None
    max_latency_p99_ms: Optional[float] = None
    min_telemetry_confidence: str = "low"
    # The user-spec requires "no SLA regression vs the strongest non-batch
    # baseline". When this is enabled and the profile carries
    # interactive_baseline_p99_ms / interactive_baseline_timeout_pct, the
    # candidate's queue p99 + timeout must not exceed those baselines by
    # more than the configured tolerance.
    enforce_interactive_baseline_floor: bool = True
    interactive_queue_p99_tolerance_ms: float = 0.0
    interactive_timeout_tolerance_pct: float = 0.0

    def __post_init__(self):
        if self.min_telemetry_confidence not in _BATCH_CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")


def _vetoes_for_point(
    point: BatchInferenceFrontierPoint,
    cfg: BatchInferenceSafetyConfig,
    profile: Optional[BatchInferenceWorkloadProfile],
    telemetry_confidence: str,
) -> tuple[list, list]:
    hard: list[str] = []
    missing: list[str] = []

    if cfg.max_deadline_miss_rate_pct is not None:
        if point.predicted_deadline_miss_rate_pct is None:
            missing.append("deadline_miss_telemetry_missing")
        elif (point.predicted_deadline_miss_rate_pct
              > cfg.max_deadline_miss_rate_pct):
            hard.append("deadline_miss_exceeds_threshold")

    if cfg.max_timeout_pct is not None:
        if point.predicted_timeout_rate_pct is None:
            missing.append("timeout_telemetry_missing")
        elif point.predicted_timeout_rate_pct > cfg.max_timeout_pct:
            hard.append("timeout_exceeds_threshold")

    if cfg.max_queue_p99_ms is not None:
        if point.predicted_queue_p99_ms is None:
            missing.append("queue_p99_telemetry_missing")
        elif point.predicted_queue_p99_ms > cfg.max_queue_p99_ms:
            hard.append("queue_p99_exceeds_threshold")

    if cfg.max_queue_p95_ms is not None:
        if point.predicted_queue_p95_ms is None:
            missing.append("queue_p95_telemetry_missing")
        elif point.predicted_queue_p95_ms > cfg.max_queue_p95_ms:
            hard.append("queue_p95_exceeds_threshold")

    if cfg.max_latency_p99_ms is not None:
        if point.predicted_latency_p99_ms is None:
            missing.append("latency_p99_telemetry_missing")
        elif point.predicted_latency_p99_ms > cfg.max_latency_p99_ms:
            hard.append("latency_p99_exceeds_threshold")

    needed = _BATCH_CONF_RANK.get(cfg.min_telemetry_confidence, 0)
    have = _BATCH_CONF_RANK.get(telemetry_confidence or "unknown", 0)
    if have < needed:
        missing.append("low_telemetry_confidence")

    # Cross-baseline veto: no SLA regression vs the interactive baseline
    # the operator already runs.
    if (cfg.enforce_interactive_baseline_floor
            and profile is not None
            and profile.interactive_baseline_p99_ms is not None
            and point.predicted_queue_p99_ms is not None):
        limit = (profile.interactive_baseline_p99_ms
                 + cfg.interactive_queue_p99_tolerance_ms)
        if point.predicted_queue_p99_ms > limit:
            hard.append("queue_p99_regresses_vs_interactive_baseline")
    if (cfg.enforce_interactive_baseline_floor
            and profile is not None
            and profile.interactive_baseline_timeout_pct is not None
            and point.predicted_timeout_rate_pct is not None):
        limit = (profile.interactive_baseline_timeout_pct
                 + cfg.interactive_timeout_tolerance_pct)
        if point.predicted_timeout_rate_pct > limit:
            hard.append("timeout_regresses_vs_interactive_baseline")

    # Profile-declared SLA overrides config when stricter.
    if (profile is not None
            and profile.deadline_miss_rate_sla_pct is not None
            and point.predicted_deadline_miss_rate_pct is not None
            and point.predicted_deadline_miss_rate_pct
            > profile.deadline_miss_rate_sla_pct):
        hard.append("deadline_miss_exceeds_profile_sla")
    if (profile is not None
            and profile.queue_wait_sla_p99_ms is not None
            and point.predicted_queue_p99_ms is not None
            and point.predicted_queue_p99_ms > profile.queue_wait_sla_p99_ms):
        hard.append("queue_p99_exceeds_profile_sla")

    return hard, missing


def classify_batch_point_safety(
    point: BatchInferenceFrontierPoint,
    cfg: BatchInferenceSafetyConfig,
    *,
    profile: Optional[BatchInferenceWorkloadProfile] = None,
    telemetry_confidence: Optional[str] = None,
) -> tuple[str, tuple]:
    hard, missing = _vetoes_for_point(
        point, cfg, profile,
        telemetry_confidence or (
            profile.telemetry_confidence if profile is not None else "unknown"
        ))
    if hard:
        return BatchInferenceSafetyStatus.UNSAFE, tuple(hard + missing)
    if missing:
        return BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY, tuple(missing)
    return BatchInferenceSafetyStatus.SAFE, ()


def is_batch_point_safe(
    point: BatchInferenceFrontierPoint,
    cfg: BatchInferenceSafetyConfig,
    *,
    profile: Optional[BatchInferenceWorkloadProfile] = None,
    telemetry_confidence: Optional[str] = None,
) -> bool:
    status, _ = classify_batch_point_safety(
        point, cfg, profile=profile,
        telemetry_confidence=telemetry_confidence)
    return status == BatchInferenceSafetyStatus.SAFE
