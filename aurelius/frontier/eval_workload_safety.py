"""Eval Workload Frontier — safety filter.

Pre-registered, transparent thresholds — never folded into a KPI weight.
The mixed-fleet veto is the critical eval-class safety gate: a candidate
that runs the eval workload alongside interactive serving must NOT degrade
the interactive baseline.

Gates:

- ``max_deadline_miss_rate_pct`` — eval-class deadline-miss ceiling
  (default 1.0% — evals tolerate slack but must still complete).
- ``max_eval_suite_completion_hours`` — completion-deadline ceiling when
  the workload profile declares one.
- ``mixed_fleet_veto`` — when the candidate runs on a SHARED fleet
  (``candidate.dedicated_fleet=False``), the interactive baseline thresholds
  on the profile are REQUIRED; if missing, the candidate is
  INSUFFICIENT_TELEMETRY. If the predicted interactive deltas exceed the
  configured tolerance, the candidate is UNSAFE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .eval_workload_models import (
    EvalWorkloadFrontierPoint,
    EvalWorkloadProfile,
    EvalWorkloadSafetyStatus,
)

_EVAL_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class EvalWorkloadSafetyConfig:
    max_deadline_miss_rate_pct: Optional[float] = 1.0
    max_eval_suite_completion_hours: Optional[float] = None
    max_queue_p99_ms: Optional[float] = None
    max_latency_p99_ms: Optional[float] = None
    min_telemetry_confidence: str = "low"
    # Mixed-fleet veto: when the candidate runs on a SHARED fleet, the
    # predicted interactive-baseline delta must stay within these
    # tolerances. The thresholds are tight by default (0 ms / 0 pct
    # tolerance) — eval workloads must NEVER make interactive worse.
    enforce_mixed_fleet_veto: bool = True
    interactive_p99_tolerance_ms: float = 0.0
    interactive_timeout_tolerance_pct: float = 0.0

    def __post_init__(self):
        if self.min_telemetry_confidence not in _EVAL_CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")


def _vetoes_for_point(
    point: EvalWorkloadFrontierPoint,
    cfg: EvalWorkloadSafetyConfig,
    profile: Optional[EvalWorkloadProfile],
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

    # Eval-suite completion deadline (config OR profile).
    completion_cap = cfg.max_eval_suite_completion_hours
    if profile is not None and profile.eval_suite_completion_deadline_hours is not None:
        if (completion_cap is None
                or profile.eval_suite_completion_deadline_hours
                < completion_cap):
            completion_cap = profile.eval_suite_completion_deadline_hours
    if completion_cap is not None:
        if point.predicted_eval_suite_completion_hours is None:
            missing.append("eval_completion_telemetry_missing")
        elif point.predicted_eval_suite_completion_hours > completion_cap:
            hard.append("eval_completion_exceeds_deadline")

    if cfg.max_queue_p99_ms is not None:
        if point.predicted_queue_p99_ms is None:
            missing.append("queue_p99_telemetry_missing")
        elif point.predicted_queue_p99_ms > cfg.max_queue_p99_ms:
            hard.append("queue_p99_exceeds_threshold")

    if cfg.max_latency_p99_ms is not None:
        if point.predicted_latency_p99_ms is None:
            missing.append("latency_p99_telemetry_missing")
        elif point.predicted_latency_p99_ms > cfg.max_latency_p99_ms:
            hard.append("latency_p99_exceeds_threshold")

    needed = _EVAL_CONF_RANK.get(cfg.min_telemetry_confidence, 0)
    have = _EVAL_CONF_RANK.get(telemetry_confidence or "unknown", 0)
    if have < needed:
        missing.append("low_telemetry_confidence")

    # Mixed-fleet veto.
    is_dedicated = (point.candidate.dedicated_fleet
                    if point.candidate.dedicated_fleet is not None else None)
    if cfg.enforce_mixed_fleet_veto and is_dedicated is False:
        # interactive baselines are REQUIRED in mixed-fleet mode.
        if profile is None:
            missing.append("mixed_fleet_baseline_missing")
        else:
            if profile.interactive_baseline_p99_ms is None:
                missing.append("interactive_baseline_p99_ms_missing")
            if profile.interactive_baseline_timeout_pct is None:
                missing.append("interactive_baseline_timeout_pct_missing")
            # And the predicted deltas must stay within tolerances.
            if (profile.interactive_baseline_p99_ms is not None
                    and point.predicted_interactive_p99_delta_ms is not None
                    and (point.predicted_interactive_p99_delta_ms
                         > cfg.interactive_p99_tolerance_ms)):
                hard.append("interactive_p99_regresses_under_shared_fleet")
            if (profile.interactive_baseline_timeout_pct is not None
                    and point.predicted_interactive_timeout_delta_pct is not None
                    and (point.predicted_interactive_timeout_delta_pct
                         > cfg.interactive_timeout_tolerance_pct)):
                hard.append("interactive_timeout_regresses_under_shared_fleet")

    return hard, missing


def classify_eval_point_safety(
    point: EvalWorkloadFrontierPoint,
    cfg: EvalWorkloadSafetyConfig,
    *,
    profile: Optional[EvalWorkloadProfile] = None,
    telemetry_confidence: Optional[str] = None,
) -> tuple[str, tuple]:
    hard, missing = _vetoes_for_point(
        point, cfg, profile,
        telemetry_confidence or (
            profile.telemetry_confidence if profile is not None else "unknown"
        ))
    if hard:
        return EvalWorkloadSafetyStatus.UNSAFE, tuple(hard + missing)
    if missing:
        return (EvalWorkloadSafetyStatus.INSUFFICIENT_TELEMETRY,
                tuple(missing))
    return EvalWorkloadSafetyStatus.SAFE, ()


def is_eval_point_safe(
    point: EvalWorkloadFrontierPoint,
    cfg: EvalWorkloadSafetyConfig,
    *,
    profile: Optional[EvalWorkloadProfile] = None,
    telemetry_confidence: Optional[str] = None,
) -> bool:
    status, _ = classify_eval_point_safety(
        point, cfg, profile=profile,
        telemetry_confidence=telemetry_confidence)
    return status == EvalWorkloadSafetyStatus.SAFE
