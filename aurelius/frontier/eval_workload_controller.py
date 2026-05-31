"""Eval Workload Frontier — controller.

Pick the safe candidate with the highest predicted goodput/$. Apply a
deadband on (rho, deadline_slack, concurrency) to avoid churn. The
mixed-fleet-veto failure mode emits ``ISOLATE_FROM_INTERACTIVE`` (move the
eval workload to a dedicated fleet) rather than just LOWER.

Recommendation-only at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .eval_workload_models import (
    EvalWorkloadFrontierAction,
    EvalWorkloadFrontierCandidate,
    EvalWorkloadFrontierDecision,
    EvalWorkloadFrontierPoint,
    EvalWorkloadFrontierSchemaError,
    EvalWorkloadProfile,
    EvalWorkloadSafetyStatus,
    EXECUTION_MODE_SHADOW,
)


@dataclass
class EvalWorkloadFrontierControllerConfig:
    rho_deadband: float = 0.05
    slack_deadband_hours: float = 1.0
    concurrency_deadband: int = 1
    kpi_deadband_pct: float = 0.01
    default_execution_mode: str = EXECUTION_MODE_SHADOW


def _is_within_deadband(
    a: EvalWorkloadFrontierCandidate,
    b: EvalWorkloadFrontierCandidate,
    cfg: EvalWorkloadFrontierControllerConfig,
) -> bool:
    if a is None or b is None:
        return False
    if (a.target_rho is not None and b.target_rho is not None
            and abs(a.target_rho - b.target_rho) > cfg.rho_deadband):
        return False
    if (a.deadline_slack_hours is not None
            and b.deadline_slack_hours is not None
            and abs(a.deadline_slack_hours - b.deadline_slack_hours)
            > cfg.slack_deadband_hours):
        return False
    if (a.concurrency is not None and b.concurrency is not None
            and abs(a.concurrency - b.concurrency)
            > cfg.concurrency_deadband):
        return False
    if a.dedicated_fleet != b.dedicated_fleet:
        return False
    return True


def _mixed_fleet_veto_in_play(p: EvalWorkloadFrontierPoint) -> bool:
    """Return True if the point's safety vetoes include the mixed-fleet
    interactive-degradation codes."""
    veto_codes = {"interactive_p99_regresses_under_shared_fleet",
                  "interactive_timeout_regresses_under_shared_fleet"}
    return any(v in veto_codes for v in p.safety_vetoes)


def choose_eval_workload_frontier_target(
    profile: EvalWorkloadProfile,
    points: Sequence[EvalWorkloadFrontierPoint],
    *,
    current_candidate: Optional[EvalWorkloadFrontierCandidate] = None,
    controller_config: Optional[EvalWorkloadFrontierControllerConfig] = None,
) -> EvalWorkloadFrontierDecision:
    """Pick the eval-frontier recommendation for one workload."""
    cfg = controller_config or EvalWorkloadFrontierControllerConfig()
    pts = list(points)

    if not pts:
        return EvalWorkloadFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(),
            action=EvalWorkloadFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="empty_frontier_point_set",
            confidence=profile.telemetry_confidence,
        )

    safe = [p for p in pts if p.is_safe]
    insufficient = [p for p in pts if p.is_insufficient_telemetry]

    current_point = None
    if current_candidate is not None:
        for p in pts:
            cc = p.candidate
            if (cc.target_rho == current_candidate.target_rho
                    and cc.deadline_slack_hours
                    == current_candidate.deadline_slack_hours
                    and cc.concurrency == current_candidate.concurrency
                    and cc.dedicated_fleet
                    == current_candidate.dedicated_fleet):
                current_point = p
                break

    # No safe points and only insufficient -> INSUFFICIENT_TELEMETRY.
    if not safe and insufficient and not [p for p in pts if not p.is_safe
                                          and not p.is_insufficient_telemetry]:
        return EvalWorkloadFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=EvalWorkloadFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="no_safe_point_and_telemetry_incomplete",
            confidence=profile.telemetry_confidence,
        )

    # Mixed-fleet veto: every UNSAFE candidate's veto codes include the
    # mixed-fleet interactive-regression codes -> ISOLATE_FROM_INTERACTIVE.
    unsafe = [p for p in pts if not p.is_safe and not p.is_insufficient_telemetry]
    if (unsafe and all(_mixed_fleet_veto_in_play(p) for p in unsafe)
            and safe):
        # Recommend the safe candidate with the highest goodput/$ that
        # uses dedicated_fleet=True (the user-spec "isolate" verb).
        dedicated_safe = [
            p for p in safe
            if p.candidate.dedicated_fleet is True
        ]
        if dedicated_safe:
            target = max(dedicated_safe, key=lambda p: (
                p.predicted_goodput_per_dollar or 0.0))
            return EvalWorkloadFrontierDecision(
                workload_id=profile.workload_id,
                selected_candidate=target.candidate,
                current_candidate=current_candidate,
                selected_point=target,
                frontier_points=tuple(pts),
                action=EvalWorkloadFrontierAction.ISOLATE_FROM_INTERACTIVE,
                reason="mixed_fleet_veto_isolate_to_dedicated_fleet",
                confidence=profile.telemetry_confidence,
            )

    # Current candidate UNSAFE -> LOWER_EVAL_CONCURRENCY.
    if (current_point is not None
            and current_point.safety_status
            == EvalWorkloadSafetyStatus.UNSAFE):
        below_safe = [
            p for p in safe
            if p.candidate.concurrency is not None
            and current_candidate.concurrency is not None
            and p.candidate.concurrency < current_candidate.concurrency
        ]
        if below_safe:
            target = max(
                below_safe,
                key=lambda p: (p.candidate.concurrency or 0))
        elif safe:
            target = min(safe,
                         key=lambda p: (p.candidate.concurrency or 0))
        else:
            target = None
        if target is not None:
            return EvalWorkloadFrontierDecision(
                workload_id=profile.workload_id,
                selected_candidate=target.candidate,
                current_candidate=current_candidate,
                selected_point=target,
                frontier_points=tuple(pts),
                action=EvalWorkloadFrontierAction.LOWER_EVAL_CONCURRENCY,
                reason="current_candidate_unsafe_lower_concurrency",
                confidence=profile.telemetry_confidence,
                safety_vetoes=current_point.safety_vetoes,
            )

    if not safe:
        return EvalWorkloadFrontierDecision(
            workload_id=profile.workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=EvalWorkloadFrontierAction.INSUFFICIENT_TELEMETRY,
            reason="no_safe_point_in_frontier",
            confidence=profile.telemetry_confidence,
        )

    best = max(safe, key=lambda p: (
        p.predicted_goodput_per_dollar or 0.0))

    if (current_candidate is not None and current_point is not None
            and current_point.is_safe):
        if _is_within_deadband(best.candidate, current_candidate, cfg):
            cur_kpi = current_point.predicted_goodput_per_dollar or 0.0
            best_kpi = best.predicted_goodput_per_dollar or 0.0
            if (cur_kpi <= 0 or
                    abs(best_kpi - cur_kpi) / cur_kpi
                    <= cfg.kpi_deadband_pct):
                return EvalWorkloadFrontierDecision(
                    workload_id=profile.workload_id,
                    selected_candidate=current_candidate,
                    current_candidate=current_candidate,
                    selected_point=current_point,
                    frontier_points=tuple(pts),
                    action=EvalWorkloadFrontierAction.KEEP_CURRENT_EVAL_POLICY,
                    reason="deadband_around_current_safe_candidate",
                    expected_goodput_per_dollar_delta=best_kpi - cur_kpi,
                    confidence=profile.telemetry_confidence,
                )

    delta = None
    if current_point is not None:
        cur_kpi = current_point.predicted_goodput_per_dollar or 0.0
        best_kpi = best.predicted_goodput_per_dollar or 0.0
        delta = best_kpi - cur_kpi

    return EvalWorkloadFrontierDecision(
        workload_id=profile.workload_id,
        selected_candidate=best.candidate,
        current_candidate=current_candidate,
        selected_point=best,
        frontier_points=tuple(pts),
        action=EvalWorkloadFrontierAction.RECOMMEND_EVAL_FRONTIER,
        reason="highest_safe_goodput_per_dollar_point",
        expected_goodput_per_dollar_delta=delta,
        confidence=profile.telemetry_confidence,
    )


def execute_eval_workload_frontier_decision(
    decision: EvalWorkloadFrontierDecision,
    *,
    allow_real_execution: bool = False,
    executor=None,
):
    if not allow_real_execution:
        return {"mode": "shadow", "executed": False,
                "reason": "real_execution_disabled_by_default"}
    if executor is None:
        return {"mode": "real_disabled", "executed": False,
                "reason": "no_real_executor_supplied"}
    return executor(decision)  # pragma: no cover
