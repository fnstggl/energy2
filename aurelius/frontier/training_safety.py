"""Training Safe Utilization Frontier — safety filter.

Pre-registered, transparent thresholds — never folded into a KPI weight
(``docs/RESULTS.md`` §1-§2). A point that breaches *any* configured gate
is UNSAFE; a point that lacks the telemetry needed to evaluate a
configured gate is INSUFFICIENT_TELEMETRY (never auto-pass).

Defaults mirror what the existing public-trace scheduling/packing
benchmarks (Philly + Alibaba GPU v2023) report as the baseline-safe
operating regimes. Per-tenant SLAs must override these explicitly via
the workload profile or the safety config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .training_models import TrainingFrontierPoint, TrainingSafetyStatus

# Structured veto reason codes (the controller filter exposes these
# verbatim — they are NOT scored or weighted).
VETO_QUEUE_WAIT_P95 = "queue_wait_p95_exceeded"
VETO_QUEUE_WAIT_P99 = "queue_wait_p99_exceeded"
VETO_STARVATION = "starvation_rate_exceeded"
VETO_FRAGMENTATION = "fragmentation_budget_exceeded"
VETO_GANG_FAILURE = "gang_scheduling_failure_exceeded"
VETO_COMPLETION = "completion_time_risk_exceeded"
VETO_RETRY_WASTE = "retry_waste_exceeded"
VETO_LOW_COMPLETED_WORK = "low_completed_work"
VETO_LOW_BACKFILL_SUCCESS = "low_backfill_success_rate"
VETO_INSUFFICIENT_TELEMETRY = "insufficient_telemetry"
VETO_UNSUPPORTED_WORKLOAD = "unsupported_workload_type"

ALL_TRAINING_VETOES = frozenset({
    VETO_QUEUE_WAIT_P95, VETO_QUEUE_WAIT_P99, VETO_STARVATION,
    VETO_FRAGMENTATION, VETO_GANG_FAILURE, VETO_COMPLETION,
    VETO_RETRY_WASTE, VETO_LOW_COMPLETED_WORK, VETO_LOW_BACKFILL_SUCCESS,
    VETO_INSUFFICIENT_TELEMETRY, VETO_UNSUPPORTED_WORKLOAD,
})

# Confidence ordering (mirrors serving safety.py).
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class TrainingSafetyConfig:
    """Pre-registered safety ceilings + telemetry minimums for training.

    Every threshold is explicit; ``None`` disables the gate.

    Defaults are pre-registered from the existing Philly + Alibaba GPU
    backtest summaries; they are NOT tuned per dataset to force a win.
    """

    # Queue-wait gates (Philly-style scheduling; Alibaba may not report).
    max_queue_wait_p95_s: Optional[float] = 6 * 3600.0
    max_queue_wait_p99_s: Optional[float] = 12 * 3600.0
    # Starvation: fraction of jobs that wait > STARVATION_WAIT_S
    # (Philly default 6 h; recorded as ``starvation_events`` in the
    # existing summary). Stated as a percentage of submitted jobs.
    max_starvation_rate_pct: Optional[float] = 5.0
    # Fragmentation: fraction of placement attempts blocked despite
    # aggregate free GPUs. Both Philly and Alibaba report this.
    max_fragmentation_block_rate_pct: Optional[float] = 25.0
    # Gang-scheduling failure: fraction of multi-GPU jobs that fail to
    # schedule atomically. May be ``None`` when the trace doesn't
    # distinguish (the controller falls back to INSUFFICIENT_TELEMETRY).
    max_gang_scheduling_failure_pct: Optional[float] = 10.0
    # Retry waste: tolerated wasted GPU-hours from retries. Philly's
    # ``attempt_analysis.wasted_gpu_hours_from_retries`` populates this.
    max_retry_waste_gpu_hours: Optional[float] = None
    # Checkpoint overhead — opt-in.
    max_checkpoint_overhead_pct: Optional[float] = None
    # Backfill success — opt-in floor.
    min_backfill_success_rate_pct: Optional[float] = None
    # Minimum completed-work ratio (placed_work / total_work) — caps
    # any "win by under-running" pathology.
    min_completed_work_ratio: Optional[float] = 0.50
    # Telemetry-confidence floor.
    min_telemetry_confidence: str = "low"

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")


def _missing(v) -> bool:
    return v is None


def _vetoes_for_point(point: TrainingFrontierPoint,
                       config: TrainingSafetyConfig,
                       telemetry_confidence: str) -> tuple[list, list]:
    """Return (hard_unsafe_vetoes, missing_telemetry_vetoes).

    The split lets the caller distinguish UNSAFE (a gate is breached
    with measured data) from INSUFFICIENT_TELEMETRY (a required gate
    can't be evaluated because the data is missing).
    """
    hard: list[str] = []
    missing: list[str] = []

    if config.max_queue_wait_p95_s is not None:
        v = point.predicted_queue_wait_p95_s
        if _missing(v):
            missing.append("queue_wait_p95_telemetry_missing")
        elif v > config.max_queue_wait_p95_s:
            hard.append(VETO_QUEUE_WAIT_P95)

    if config.max_queue_wait_p99_s is not None:
        v = point.predicted_queue_wait_p99_s
        if _missing(v):
            missing.append("queue_wait_p99_telemetry_missing")
        elif v > config.max_queue_wait_p99_s:
            hard.append(VETO_QUEUE_WAIT_P99)

    if config.max_starvation_rate_pct is not None:
        v = point.predicted_starvation_rate_pct
        if _missing(v):
            missing.append("starvation_telemetry_missing")
        elif v > config.max_starvation_rate_pct:
            hard.append(VETO_STARVATION)

    if config.max_fragmentation_block_rate_pct is not None:
        v = point.predicted_fragmentation_block_rate_pct
        if _missing(v):
            missing.append("fragmentation_telemetry_missing")
        elif v > config.max_fragmentation_block_rate_pct:
            hard.append(VETO_FRAGMENTATION)

    if config.max_gang_scheduling_failure_pct is not None:
        v = point.predicted_gang_scheduling_failure_pct
        if _missing(v):
            missing.append("gang_failure_telemetry_missing")
        elif v > config.max_gang_scheduling_failure_pct:
            hard.append(VETO_GANG_FAILURE)

    if config.max_retry_waste_gpu_hours is not None:
        v = point.predicted_retry_waste_gpu_hours
        if _missing(v):
            missing.append("retry_waste_telemetry_missing")
        elif v > config.max_retry_waste_gpu_hours:
            hard.append(VETO_RETRY_WASTE)

    if config.min_backfill_success_rate_pct is not None:
        v = point.predicted_backfill_success_rate_pct
        if _missing(v):
            missing.append("backfill_success_telemetry_missing")
        elif v < config.min_backfill_success_rate_pct:
            hard.append(VETO_LOW_BACKFILL_SUCCESS)

    if config.min_completed_work_ratio is not None:
        # Completed work ratio is computed by the estimator and folded
        # into ``predicted_completed_work`` (absolute). The estimator
        # compares against the per-trace baseline; here we only check
        # that the point HAS a completed-work prediction.
        if _missing(point.predicted_completed_work):
            missing.append("completed_work_telemetry_missing")

    # Telemetry-confidence gate (always evaluated when configured).
    needed = _CONF_RANK.get(config.min_telemetry_confidence, 0)
    have = _CONF_RANK.get(telemetry_confidence or "unknown", 0)
    if have < needed:
        missing.append("low_telemetry_confidence")

    return hard, missing


def classify_training_frontier_point(
    point: TrainingFrontierPoint,
    config: TrainingSafetyConfig,
    *,
    telemetry_confidence: Optional[str] = None,
) -> tuple[str, tuple]:
    """Return the (``safety_status``, ``safety_vetoes``) pair.

    - SAFE if every configured gate passes.
    - UNSAFE if any configured gate is breached with measured data.
    - INSUFFICIENT_TELEMETRY if any required signal is missing but no
      hard gate breaches. UNSAFE wins over INSUFFICIENT_TELEMETRY when
      both apply — a known breach is a known breach.
    """
    hard, missing = _vetoes_for_point(point, config,
                                       telemetry_confidence or "unknown")
    if hard:
        return TrainingSafetyStatus.UNSAFE, tuple(hard + missing)
    if missing:
        return TrainingSafetyStatus.INSUFFICIENT_TELEMETRY, tuple(missing)
    return TrainingSafetyStatus.SAFE, ()


def is_training_frontier_point_safe(
    point: TrainingFrontierPoint,
    config: TrainingSafetyConfig,
    *,
    telemetry_confidence: Optional[str] = None,
) -> bool:
    """Convenience wrapper — returns True iff ``point`` is SAFE."""
    status, _ = classify_training_frontier_point(
        point, config, telemetry_confidence=telemetry_confidence)
    return status == TrainingSafetyStatus.SAFE
