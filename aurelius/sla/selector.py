"""SLA-aware action selection.

Ranks candidate optimization actions using:

    score = expected_savings
            - migration_penalty
            - latency_risk_penalty
            - queue_risk_penalty
            - availability_risk_penalty
            - soft_sla_penalty

subject to: hard_sla_constraints satisfied.

If the cheapest (highest-savings) action violates a hard SLA, the selector
picks the next-best SLA-safe action, or the no-op (keep current placement) if
nothing safe beats keeping put.

Also implements the soft "preferred region" tradeoff rule: a more expensive
preferred-region option is chosen over a cheaper non-preferred one when the
savings gap is within ``max_acceptable_savings_tradeoff_pct``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .actions import OptimizationAction, keep_current
from .evaluator import SLAEvaluation, evaluate_action_against_sla
from .schema import SLAPolicy
from .telemetry import HeuristicPredictor, RegionContext, WorkloadState

logger = logging.getLogger(__name__)


@dataclass
class ScoredAction:
    action: OptimizationAction
    evaluation: SLAEvaluation
    score: float
    sla_safe: bool


@dataclass
class SLADecision:
    """Outcome of SLA-aware selection for one workload."""

    workload_id: str
    unconstrained_action: OptimizationAction
    chosen_action: OptimizationAction
    unconstrained_savings_pct: float
    chosen_savings_pct: float
    was_corrected: bool
    blocked_reasons: list[str] = field(default_factory=list)
    risk_avoided: float = 0.0
    scored_actions: list[ScoredAction] = field(default_factory=list)
    explanation: str = ""

    @property
    def savings_sacrificed_pct(self) -> float:
        return max(0.0, self.unconstrained_savings_pct - self.chosen_savings_pct)

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "unconstrained_action": self.unconstrained_action.action_type.value,
            "unconstrained_target_region": self.unconstrained_action.target_region,
            "unconstrained_savings_pct": round(self.unconstrained_savings_pct, 4),
            "chosen_action": self.chosen_action.action_type.value,
            "chosen_target_region": self.chosen_action.target_region,
            "chosen_savings_pct": round(self.chosen_savings_pct, 4),
            "was_corrected": self.was_corrected,
            "savings_sacrificed_pct": round(self.savings_sacrificed_pct, 4),
            "blocked_reasons": list(self.blocked_reasons),
            "risk_avoided": round(self.risk_avoided, 4),
            "explanation": self.explanation,
        }


class SLAAwareActionSelector:
    """Selects the best SLA-safe action from a set of candidates."""

    def __init__(self, predictor: Optional[HeuristicPredictor] = None):
        self.predictor = predictor or HeuristicPredictor()

    def select(
        self,
        workload,
        candidate_actions: list[OptimizationAction],
        current_state: WorkloadState,
        sla_policy: Optional[SLAPolicy],
        region_contexts: Optional[dict[str, RegionContext]] = None,
        predicted_states: Optional[dict[int, WorkloadState]] = None,
        now: Optional[datetime] = None,
        block_on_unknown: bool = False,
    ) -> SLADecision:
        """Rank candidate actions and choose the best SLA-safe one.

        Args:
            workload: The workload/job.
            candidate_actions: Actions to consider. A no-op is always added.
            current_state: Current observed state.
            sla_policy: Governing policy (None/disabled => pure savings ranking).
            region_contexts: {region: RegionContext} for the predictor.
            predicted_states: Optional pre-computed {id(action): WorkloadState}.
                When absent, the HeuristicPredictor is used.
            now: Current time (for no_migration_windows).
            block_on_unknown: Fail-closed on unknown metrics.
        """
        region_contexts = region_contexts or {}
        workload_id = getattr(workload, "job_id", getattr(workload, "id", "unknown"))

        # Always include the no-op so "do nothing" is a real option.
        actions = list(candidate_actions)
        if not any(a.is_noop for a in actions):
            actions.append(keep_current(region=current_state.region))

        # Unconstrained recommendation = highest expected savings (ignores SLA).
        unconstrained = max(actions, key=lambda a: a.expected_savings_pct)

        scored: list[ScoredAction] = []
        for action in actions:
            dest_ctx = (
                region_contexts.get(action.target_region)
                if action.target_region
                else None
            )
            if predicted_states is not None and id(action) in predicted_states:
                pred = predicted_states[id(action)]
            else:
                pred = self.predictor.predict(action, current_state, dest_ctx)

            evaluation = evaluate_action_against_sla(
                action=action,
                workload=workload,
                current_state=current_state,
                predicted_state=pred,
                sla_policy=sla_policy,
                now=now,
                block_on_unknown=block_on_unknown,
            )

            # score = savings - risk penalties - soft penalty
            score = (
                action.expected_savings_pct
                - evaluation.risk_breakdown.total
                - evaluation.soft_penalty_score
            )
            scored.append(
                ScoredAction(
                    action=action,
                    evaluation=evaluation,
                    score=score,
                    sla_safe=evaluation.allowed,
                )
            )

        # Among SLA-safe actions, pick the highest score.
        safe = [s for s in scored if s.sla_safe]

        # Soft preferred-region tradeoff: if the top safe action is NOT in
        # preferred regions but a preferred-region safe action exists whose
        # savings gap is within max_acceptable_savings_tradeoff_pct, prefer it.
        chosen_scored = self._apply_preferred_region_tradeoff(safe, sla_policy)

        if chosen_scored is None:
            # No safe action beats keeping put; fall back to no-op.
            noop = next((s for s in scored if s.action.is_noop), None)
            chosen_scored = noop or max(scored, key=lambda s: s.score)

        chosen = chosen_scored.action

        # Determine correction + blocked reasons relative to unconstrained.
        unconstrained_eval = next(
            (s.evaluation for s in scored if s.action is unconstrained), None
        )
        blocked_reasons: list[str] = []
        risk_avoided = 0.0
        was_corrected = chosen is not unconstrained
        if unconstrained_eval is not None and not unconstrained_eval.allowed:
            blocked_reasons = list(unconstrained_eval.violated_hard_constraints)
            risk_avoided = unconstrained_eval.risk_score
        elif was_corrected and unconstrained_eval is not None:
            # Corrected for soft/risk reasons, not a hard block.
            risk_avoided = unconstrained_eval.risk_breakdown.total

        explanation = self._build_explanation(
            workload_id, unconstrained, chosen, blocked_reasons, scored, sla_policy
        )

        return SLADecision(
            workload_id=workload_id,
            unconstrained_action=unconstrained,
            chosen_action=chosen,
            unconstrained_savings_pct=unconstrained.expected_savings_pct,
            chosen_savings_pct=chosen.expected_savings_pct,
            was_corrected=was_corrected,
            blocked_reasons=blocked_reasons,
            risk_avoided=risk_avoided,
            scored_actions=scored,
            explanation=explanation,
        )

    def _apply_preferred_region_tradeoff(
        self,
        safe: list[ScoredAction],
        sla_policy: Optional[SLAPolicy],
    ) -> Optional[ScoredAction]:
        if not safe:
            return None
        top = max(safe, key=lambda s: s.score)

        if sla_policy is None or not sla_policy.soft.preferred_regions:
            return top

        preferred = set(sla_policy.soft.preferred_regions)
        tradeoff = sla_policy.soft.max_acceptable_savings_tradeoff_pct
        if tradeoff is None:
            return top

        top_region = top.action.target_region
        if top_region in preferred:
            return top

        # Find the best safe action that IS in a preferred region.
        preferred_safe = [
            s for s in safe if (s.action.target_region in preferred)
        ]
        if not preferred_safe:
            return top
        best_preferred = max(preferred_safe, key=lambda s: s.action.expected_savings_pct)

        savings_gap = top.action.expected_savings_pct - best_preferred.action.expected_savings_pct
        if savings_gap <= tradeoff:
            logger.info(
                "SLA soft preference: choosing preferred region %s (savings %.1f%%) over "
                "%s (savings %.1f%%); gap %.1f%% <= tradeoff %.1f%%",
                best_preferred.action.target_region,
                best_preferred.action.expected_savings_pct,
                top_region,
                top.action.expected_savings_pct,
                savings_gap,
                tradeoff,
            )
            return best_preferred
        return top

    @staticmethod
    def _build_explanation(
        workload_id: str,
        unconstrained: OptimizationAction,
        chosen: OptimizationAction,
        blocked_reasons: list[str],
        scored: list[ScoredAction],
        sla_policy: Optional[SLAPolicy],
    ) -> str:
        lines = [f"workload={workload_id}"]
        lines.append(
            f"unconstrained={unconstrained.action_type.value}"
            f"->{unconstrained.target_region} savings={unconstrained.expected_savings_pct:.1f}%"
        )
        if blocked_reasons:
            lines.append("blocked: " + "; ".join(blocked_reasons))
        lines.append(
            f"chosen={chosen.action_type.value}->{chosen.target_region} "
            f"savings={chosen.expected_savings_pct:.1f}%"
        )
        if chosen is not unconstrained:
            sacrificed = unconstrained.expected_savings_pct - chosen.expected_savings_pct
            lines.append(f"savings_sacrificed={sacrificed:.1f}%")
        return " | ".join(lines)
