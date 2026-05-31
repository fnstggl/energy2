"""Batch Inference Frontier — controller.

Given a list of :class:`BatchInferenceFrontierPoint` (produced by the
estimator), pick the safe point with the highest predicted goodput/$ as the
recommendation. Apply a deadband to avoid churn around the current
candidate. Recommendation-only at construction.

Hard rules (mirrored from the serving + training controllers):

- If the safe set is empty AND the current candidate is the only available
  insufficient-telemetry point → INSUFFICIENT_TELEMETRY.
- If the current candidate is UNSAFE → LOWER_BATCH_PRESSURE (recommend the
  highest safe rho strictly below the current rho).
- Otherwise pick the safe point with the highest predicted goodput/$.
- Deadband: if the selected candidate is within `deadband` of the current
  candidate on (rho, deadline_slack, concurrency) AND the KPI delta is
  small → KEEP_CURRENT_BATCH_POLICY.

``BatchInferenceFrontierDecision.executable_in_real_cluster=False`` at
construction. Real execution requires `allow_real_execution=True` on the
``execute_batch_inference_frontier_decision`` shim (which ships only a
no-op stub for v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .batch_inference_models import (
    BatchInferenceFrontierAction,
    BatchInferenceFrontierCandidate,
    BatchInferenceFrontierDecision,
    BatchInferenceFrontierPoint,
    BatchInferenceFrontierSchemaError,
    BatchInferenceSafetyStatus,
    BatchInferenceWorkloadProfile,
    EXECUTION_MODE_SHADOW,
)


@dataclass
class BatchInferenceFrontierControllerConfig:
    """Controller settings. All deadbands are transparent / opt-in."""

    rho_deadband: float = 0.05
    slack_deadband_seconds: float = 60.0
    concurrency_deadband: int = 1
    kpi_deadband_pct: float = 0.01
    default_execution_mode: str = EXECUTION_MODE_SHADOW


def _is_within_deadband(
    a: BatchInferenceFrontierCandidate,
    b: BatchInferenceFrontierCandidate,
    cfg: BatchInferenceFrontierControllerConfig,
) -> bool:
    if a is None or b is None:
        return False
    if (a.target_rho is not None and b.target_rho is not None
            and abs(a.target_rho - b.target_rho) > cfg.rho_deadband):
        return False
    if (a.deadline_slack_seconds is not None
            and b.deadline_slack_seconds is not None
            and abs(a.deadline_slack_seconds - b.deadline_slack_seconds)
            > cfg.slack_deadband_seconds):
        return False
    if (a.batch_concurrency is not None and b.batch_concurrency is not None
            and abs(a.batch_concurrency - b.batch_concurrency)
            > cfg.concurrency_deadband):
        return False
    return True


def choose_batch_inference_frontier_target(
    profile: BatchInferenceWorkloadProfile,
    points: Sequence[BatchInferenceFrontierPoint],
    *,
    current_candidate: Optional[BatchInferenceFrontierCandidate] = None,
    controller_config: Optional[BatchInferenceFrontierControllerConfig] = None,
) -> BatchInferenceFrontierDecision:
    """Pick the batch-frontier recommendation for one workload.

    Recommendation-only. Real execution path NEVER returned actionable
    here — that requires the explicit-opt-in shim.
    """
    cfg = controller_config or BatchInferenceFrontierControllerConfig()
    pts = list(points)

    if not pts:
        return BatchInferenceFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(),
            action=BatchInferenceFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="empty_frontier_point_set",
            confidence=profile.telemetry_confidence,
        )

    safe = [p for p in pts if p.is_safe]
    insufficient = [p for p in pts if p.is_insufficient_telemetry]

    # Find the current candidate's safety status (if known).
    current_point = None
    if current_candidate is not None:
        for p in pts:
            cc = p.candidate
            if (cc.target_rho == current_candidate.target_rho
                    and cc.deadline_slack_seconds
                    == current_candidate.deadline_slack_seconds
                    and cc.batch_concurrency
                    == current_candidate.batch_concurrency):
                current_point = p
                break

    # No safe points + telemetry incomplete -> INSUFFICIENT_TELEMETRY.
    if not safe and insufficient and not [p for p in pts if not p.is_safe
                                          and not p.is_insufficient_telemetry]:
        return BatchInferenceFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=BatchInferenceFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="no_safe_point_and_telemetry_incomplete",
            confidence=profile.telemetry_confidence,
        )

    # Current candidate is UNSAFE -> LOWER_BATCH_PRESSURE.
    if (current_point is not None
            and current_point.safety_status
            == BatchInferenceSafetyStatus.UNSAFE):
        # Find the highest-rho safe point strictly below current rho.
        below_safe = [
            p for p in safe
            if p.candidate.target_rho is not None
            and current_candidate.target_rho is not None
            and p.candidate.target_rho < current_candidate.target_rho
        ]
        if below_safe:
            target = max(
                below_safe,
                key=lambda p: (p.candidate.target_rho or 0.0))
            return BatchInferenceFrontierDecision(
                workload_id=profile.workload_id,
                selected_candidate=target.candidate,
                current_candidate=current_candidate,
                selected_point=target,
                frontier_points=tuple(pts),
                action=BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE,
                reason="current_candidate_unsafe_lower_to_safe_below",
                confidence=profile.telemetry_confidence,
                safety_vetoes=current_point.safety_vetoes,
            )
        # No safe point at all below current — emit LOWER on lowest available.
        if safe:
            target = min(safe, key=lambda p: (p.candidate.target_rho or 0.0))
            return BatchInferenceFrontierDecision(
                workload_id=profile.workload_id,
                selected_candidate=target.candidate,
                current_candidate=current_candidate,
                selected_point=target,
                frontier_points=tuple(pts),
                action=BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE,
                reason="current_candidate_unsafe_lower_to_lowest_safe",
                confidence=profile.telemetry_confidence,
                safety_vetoes=current_point.safety_vetoes,
            )
        return BatchInferenceFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE,
            reason="current_unsafe_no_safe_alternative",
            confidence=profile.telemetry_confidence,
            safety_vetoes=current_point.safety_vetoes,
        )

    if not safe:
        return BatchInferenceFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=BatchInferenceFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="no_safe_point_in_frontier",
            confidence=profile.telemetry_confidence,
        )

    # Pick the highest-goodput/$ safe point.
    best = max(safe, key=lambda p: (
        p.predicted_goodput_per_dollar or 0.0))

    # Deadband vs current.
    if current_candidate is not None and current_point is not None and current_point.is_safe:
        if _is_within_deadband(best.candidate, current_candidate, cfg):
            cur_kpi = current_point.predicted_goodput_per_dollar or 0.0
            best_kpi = best.predicted_goodput_per_dollar or 0.0
            if (cur_kpi <= 0 or
                    abs(best_kpi - cur_kpi) / cur_kpi
                    <= cfg.kpi_deadband_pct):
                return BatchInferenceFrontierDecision(
                    workload_id=profile.workload_id,
                    selected_candidate=current_candidate,
                    current_candidate=current_candidate,
                    selected_point=current_point,
                    frontier_points=tuple(pts),
                    action=BatchInferenceFrontierAction.KEEP_CURRENT_BATCH_POLICY,
                    reason="deadband_around_current_safe_candidate",
                    expected_goodput_per_dollar_delta=best_kpi - cur_kpi,
                    confidence=profile.telemetry_confidence,
                )

    delta = None
    if current_point is not None:
        cur_kpi = current_point.predicted_goodput_per_dollar or 0.0
        best_kpi = best.predicted_goodput_per_dollar or 0.0
        delta = best_kpi - cur_kpi

    return BatchInferenceFrontierDecision(
        workload_id=profile.workload_id,
        selected_candidate=best.candidate,
        current_candidate=current_candidate,
        selected_point=best,
        frontier_points=tuple(pts),
        action=BatchInferenceFrontierAction.RECOMMEND_BATCH_FRONTIER,
        reason="highest_safe_goodput_per_dollar_point",
        expected_goodput_per_dollar_delta=delta,
        confidence=profile.telemetry_confidence,
    )


def execute_batch_inference_frontier_decision(
    decision: BatchInferenceFrontierDecision,
    *,
    allow_real_execution: bool = False,
    executor=None,
):
    """Stub real-execution shim.

    Mirrors the serving + training siblings: the v1 ships only a no-op
    stub. Real execution requires BOTH ``allow_real_execution=True`` AND a
    non-stub ``executor`` callable. The decision's
    ``executable_in_real_cluster`` flag stays False at construction; this
    shim never sets it.
    """
    if not allow_real_execution:
        return {"mode": "shadow",
                "executed": False,
                "reason": "real_execution_disabled_by_default"}
    if executor is None:
        return {"mode": "real_disabled",
                "executed": False,
                "reason": "no_real_executor_supplied"}
    # Defer entirely to the caller-supplied executor — we do not write
    # any real cluster code in v1.
    return executor(decision)  # pragma: no cover - requires non-stub executor
