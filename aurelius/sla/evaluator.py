"""SLA-aware optimization correction engine.

Core entry point: :func:`evaluate_action_against_sla`.

Given a candidate optimization action plus the current and predicted workload
states and the governing SLA policy, it decides whether the action is allowed
under HARD constraints, scores SOFT-constraint penalties, computes a risk
score, and (when blocked) proposes a corrected action.

Hard violation => ``allowed=False`` (the action MUST be blocked).
Soft violation => action allowed but ``soft_penalty_score`` increases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .actions import ActionType, OptimizationAction, keep_current
from .schema import OptimizationAggressiveness, SLAPolicy
from .telemetry import WorkloadState

# Aggressiveness scales how much risk reduces an action's attractiveness and
# how strict soft penalties are. Conservative => penalties amplified.
_AGGRESSIVENESS_RISK_MULT = {
    OptimizationAggressiveness.CONSERVATIVE: 1.5,
    OptimizationAggressiveness.BALANCED: 1.0,
    OptimizationAggressiveness.AGGRESSIVE: 0.5,
}


@dataclass
class RiskBreakdown:
    """Per-dimension risk penalties (in 'savings-percent-equivalent' units).

    These feed the optimizer's ranking formula:
        score = expected_savings
                - migration_penalty
                - latency_risk_penalty
                - queue_risk_penalty
                - availability_risk_penalty
                - soft_sla_penalty
    """

    migration_penalty: float = 0.0
    latency_risk_penalty: float = 0.0
    queue_risk_penalty: float = 0.0
    availability_risk_penalty: float = 0.0
    capacity_risk_penalty: float = 0.0
    thermal_risk_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.migration_penalty
            + self.latency_risk_penalty
            + self.queue_risk_penalty
            + self.availability_risk_penalty
            + self.capacity_risk_penalty
            + self.thermal_risk_penalty
        )


@dataclass
class SLAEvaluation:
    """Result of evaluating one action against one SLA policy."""

    action: OptimizationAction
    allowed: bool
    violated_hard_constraints: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    soft_penalty_score: float = 0.0
    risk_score: float = 0.0
    risk_breakdown: RiskBreakdown = field(default_factory=RiskBreakdown)
    corrected_action: Optional[OptimizationAction] = None
    explanation: str = ""
    unknown_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action.action_type.value,
            "target_region": self.action.target_region,
            "expected_savings_pct": round(self.action.expected_savings_pct, 4),
            "allowed": self.allowed,
            "violated_hard_constraints": list(self.violated_hard_constraints),
            "soft_violations": list(self.soft_violations),
            "soft_penalty_score": round(self.soft_penalty_score, 4),
            "risk_score": round(self.risk_score, 4),
            "risk_breakdown": {
                "migration_penalty": round(self.risk_breakdown.migration_penalty, 4),
                "latency_risk_penalty": round(self.risk_breakdown.latency_risk_penalty, 4),
                "queue_risk_penalty": round(self.risk_breakdown.queue_risk_penalty, 4),
                "availability_risk_penalty": round(self.risk_breakdown.availability_risk_penalty, 4),
                "capacity_risk_penalty": round(self.risk_breakdown.capacity_risk_penalty, 4),
                "thermal_risk_penalty": round(self.risk_breakdown.thermal_risk_penalty, 4),
            },
            "corrected_action": (
                self.corrected_action.action_type.value if self.corrected_action else None
            ),
            "explanation": self.explanation,
            "unknown_metrics": list(self.unknown_metrics),
        }


def _headroom_penalty(value: Optional[float], limit: Optional[float]) -> float:
    """A soft penalty that grows as ``value`` approaches ``limit`` from below.

    0 when far below limit; rises toward 1.0 as value -> limit. Used for risk
    scoring of metrics that are within hard bounds but getting close.
    """
    if value is None or limit is None or limit <= 0:
        return 0.0
    frac = value / limit
    if frac <= 0.5:
        return 0.0
    if frac >= 1.0:
        return 1.0
    return (frac - 0.5) / 0.5


def _in_no_migration_window(windows: Optional[list[list[str]]], now: Optional[datetime]) -> bool:
    if not windows or now is None:
        return False
    for w in windows:
        try:
            start = datetime.fromisoformat(w[0])
            end = datetime.fromisoformat(w[1])
        except (ValueError, IndexError, TypeError):
            continue
        # Compare naively if tz-awareness mismatches.
        n = now
        if (start.tzinfo is None) != (n.tzinfo is None):
            n = n.replace(tzinfo=None)
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        if start <= n <= end:
            return True
    return False


def evaluate_action_against_sla(
    action: OptimizationAction,
    workload,  # aurelius.models.Job-like or anything with .job_id/.workload_type
    current_state: WorkloadState,
    predicted_state: WorkloadState,
    sla_policy: Optional[SLAPolicy],
    now: Optional[datetime] = None,
    block_on_unknown: bool = False,
) -> SLAEvaluation:
    """Evaluate a candidate optimization action against an SLA policy.

    Args:
        action: The candidate action.
        workload: The workload/job (used for id/type/migration accounting).
        current_state: Observed current state.
        predicted_state: Predicted post-action state (e.g. from HeuristicPredictor).
        sla_policy: Governing policy. If None or disabled, the action is allowed
            with zero penalties (preserves pre-SLA behavior).
        now: Current time, for no_migration_windows checks.
        block_on_unknown: If True, a hard constraint whose metric is unknown
            BLOCKS the action (strict fail-closed). Default False: unknown
            metrics are reported but do not block — Aurelius will not claim an
            SLA is met on data it does not have, nor fabricate a violation.

    Returns:
        SLAEvaluation.
    """
    # No policy / disabled => preserve legacy behavior (allow, no penalty).
    if sla_policy is None or not sla_policy.enabled:
        return SLAEvaluation(
            action=action,
            allowed=True,
            explanation="No SLA policy in effect; action allowed unchanged.",
        )

    hard = sla_policy.hard
    soft = sla_policy.soft
    target_region = action.target_region or predicted_state.region or current_state.region

    violated: list[str] = []
    unknown: list[str] = []

    # ---------------- HARD: placement / residency ----------------
    if action.target_region is not None:
        if hard.allowed_regions is not None and target_region not in hard.allowed_regions:
            violated.append(
                f"region '{target_region}' not in allowed_regions {hard.allowed_regions}"
            )
        if hard.forbidden_regions and target_region in hard.forbidden_regions:
            violated.append(f"region '{target_region}' is forbidden")
        if (
            hard.data_residency_region is not None
            and target_region != hard.data_residency_region
        ):
            violated.append(
                f"data residency requires region '{hard.data_residency_region}', "
                f"action targets '{target_region}'"
            )

    # ---------------- HARD: latency ----------------
    def _check_max(metric_name: str, value: Optional[float], limit: Optional[float]):
        if limit is None:
            return
        if value is None:
            unknown.append(metric_name)
            if block_on_unknown:
                violated.append(f"{metric_name} unknown and block_on_unknown=True")
            return
        if value > limit:
            violated.append(f"predicted {metric_name} {value:.1f} > limit {limit:.1f}")

    _check_max("p95_latency_ms", predicted_state.p95_latency_ms, hard.max_p95_latency_ms)
    _check_max("p99_latency_ms", predicted_state.p99_latency_ms, hard.max_p99_latency_ms)
    _check_max("queue_wait_ms", predicted_state.queue_wait_ms, hard.max_queue_wait_ms)
    _check_max("error_rate_pct", predicted_state.error_rate_pct, hard.max_error_rate_pct)
    _check_max("timeout_rate_pct", predicted_state.timeout_rate_pct, hard.max_timeout_rate_pct)

    # ---------------- HARD: availability (min) ----------------
    if hard.min_availability_pct is not None:
        if predicted_state.availability_pct is None:
            unknown.append("availability_pct")
            if block_on_unknown:
                violated.append("availability_pct unknown and block_on_unknown=True")
        elif predicted_state.availability_pct < hard.min_availability_pct:
            violated.append(
                f"predicted availability {predicted_state.availability_pct:.3f}% "
                f"< min {hard.min_availability_pct:.3f}%"
            )

    # ---------------- HARD: capacity buffer (min) ----------------
    if hard.required_capacity_buffer_pct is not None:
        if predicted_state.capacity_buffer_pct is None:
            unknown.append("capacity_buffer_pct")
            if block_on_unknown:
                violated.append("capacity_buffer_pct unknown and block_on_unknown=True")
        elif predicted_state.capacity_buffer_pct < hard.required_capacity_buffer_pct:
            violated.append(
                f"predicted capacity buffer {predicted_state.capacity_buffer_pct:.1f}% "
                f"< required {hard.required_capacity_buffer_pct:.1f}%"
            )

    # ---------------- HARD: migration governance ----------------
    if action.is_migration and action.target_region not in (None, current_state.region):
        if hard.migration_allowed is False:
            violated.append("migration_allowed=false but action migrates the workload")
        if hard.max_migrations_per_hour is not None:
            # current count + this migration
            projected = current_state.migration_count_last_hour + 1
            if projected > hard.max_migrations_per_hour:
                violated.append(
                    f"migration would be #{projected} this hour > "
                    f"max_migrations_per_hour {hard.max_migrations_per_hour}"
                )
        if _in_no_migration_window(hard.no_migration_windows, now):
            violated.append("current time is inside a no_migration_window")

    # ---------------- RISK breakdown (savings-pct-equivalent units) ----------------
    agg = sla_policy.aggressiveness
    risk_mult = _AGGRESSIVENESS_RISK_MULT.get(agg, 1.0)
    rb = RiskBreakdown()

    if action.is_migration and action.target_region not in (None, current_state.region):
        rb.migration_penalty = 2.0 * risk_mult  # # HEURISTIC base migration friction
    if action.action_type == ActionType.CONSOLIDATE:
        rb.migration_penalty = 1.0 * risk_mult

    # Latency headroom risk (how close predicted is to the cap).
    lat_pen = max(
        _headroom_penalty(predicted_state.p99_latency_ms, hard.max_p99_latency_ms),
        _headroom_penalty(predicted_state.p95_latency_ms, hard.max_p95_latency_ms),
    )
    rb.latency_risk_penalty = lat_pen * 5.0 * risk_mult  # scale to savings-pct units

    rb.queue_risk_penalty = (
        _headroom_penalty(predicted_state.queue_wait_ms, hard.max_queue_wait_ms)
        * 5.0
        * risk_mult
    )

    # Availability risk: how close to the floor.
    if hard.min_availability_pct is not None and predicted_state.availability_pct is not None:
        # 0 when at 100%, grows as it nears the floor.
        margin = predicted_state.availability_pct - hard.min_availability_pct
        floor_gap = 100.0 - hard.min_availability_pct
        if floor_gap > 0:
            closeness = 1.0 - max(0.0, min(1.0, margin / floor_gap))
            rb.availability_risk_penalty = closeness * 5.0 * risk_mult

    # Capacity risk.
    if (
        hard.required_capacity_buffer_pct is not None
        and predicted_state.capacity_buffer_pct is not None
    ):
        req = hard.required_capacity_buffer_pct
        if predicted_state.capacity_buffer_pct < req * 1.5:  # within 50% of floor
            rb.capacity_risk_penalty = 3.0 * risk_mult

    # Thermal risk: predicted p99 inflated relative to current implies stress.
    if (
        current_state.p99_latency_ms
        and predicted_state.p99_latency_ms
        and predicted_state.p99_latency_ms > current_state.p99_latency_ms * 1.3
    ):
        rb.thermal_risk_penalty = 1.5 * risk_mult

    # ---------------- SOFT penalties ----------------
    soft_penalty = 0.0
    soft_violations: list[str] = []

    # Preferred regions: penalize leaving the preferred set.
    if soft.preferred_regions and target_region not in soft.preferred_regions:
        soft_penalty += 1.0 * risk_mult
        soft_violations.append(
            f"target region '{target_region}' not in preferred {soft.preferred_regions}"
        )

    # Carbon preference.
    if (
        soft.preferred_carbon_intensity is not None
        and predicted_state.carbon_intensity is not None
        and predicted_state.carbon_intensity > soft.preferred_carbon_intensity
    ):
        over = predicted_state.carbon_intensity - soft.preferred_carbon_intensity
        soft_penalty += min(3.0, over / max(1.0, soft.preferred_carbon_intensity) * 3.0) * risk_mult
        soft_violations.append(
            f"carbon {predicted_state.carbon_intensity:.0f} > preferred "
            f"{soft.preferred_carbon_intensity:.0f} gCO2/kWh"
        )

    # Energy price percentile preference (lower percentile = cheaper = better).
    if (
        soft.preferred_energy_price_percentile is not None
        and predicted_state.energy_price_percentile is not None
        and predicted_state.energy_price_percentile > soft.preferred_energy_price_percentile
    ):
        soft_penalty += 1.0 * risk_mult
        soft_violations.append(
            f"energy price percentile {predicted_state.energy_price_percentile:.0f} > "
            f"preferred {soft.preferred_energy_price_percentile:.0f}"
        )

    # GPU utilization target (under-utilization is wasteful; over is risky).
    if (
        soft.target_gpu_utilization_pct is not None
        and predicted_state.gpu_utilization_pct is not None
    ):
        diff = abs(predicted_state.gpu_utilization_pct - soft.target_gpu_utilization_pct)
        if diff > 15.0:  # # HEURISTIC tolerance band
            soft_penalty += min(2.0, diff / 50.0) * risk_mult
            soft_violations.append(
                f"GPU util {predicted_state.gpu_utilization_pct:.0f}% off target "
                f"{soft.target_gpu_utilization_pct:.0f}%"
            )

    # Latency headroom preference.
    if (
        soft.preferred_latency_headroom_pct is not None
        and hard.max_p99_latency_ms is not None
        and predicted_state.p99_latency_ms is not None
    ):
        headroom_pct = (
            (hard.max_p99_latency_ms - predicted_state.p99_latency_ms)
            / hard.max_p99_latency_ms
            * 100.0
        )
        if headroom_pct < soft.preferred_latency_headroom_pct:
            soft_penalty += 1.5 * risk_mult
            soft_violations.append(
                f"latency headroom {headroom_pct:.0f}% < preferred "
                f"{soft.preferred_latency_headroom_pct:.0f}%"
            )

    # cost_per_token / tokens_per_joule targets (efficiency).
    if (
        soft.target_cost_per_token is not None
        and predicted_state.cost_per_token is not None
        and predicted_state.cost_per_token > soft.target_cost_per_token
    ):
        soft_penalty += 1.0 * risk_mult
        soft_violations.append(
            f"cost/token {predicted_state.cost_per_token:.4g} > target "
            f"{soft.target_cost_per_token:.4g}"
        )
    if (
        soft.target_tokens_per_joule is not None
        and predicted_state.tokens_per_joule is not None
        and predicted_state.tokens_per_joule < soft.target_tokens_per_joule
    ):
        soft_penalty += 1.0 * risk_mult
        soft_violations.append(
            f"tokens/joule {predicted_state.tokens_per_joule:.4g} < target "
            f"{soft.target_tokens_per_joule:.4g}"
        )

    risk_score = rb.total
    allowed = len(violated) == 0

    # ---------------- corrected action ----------------
    corrected: Optional[OptimizationAction] = None
    if not allowed and not action.is_noop:
        # The safe correction for a blocked move is to keep current placement.
        corrected = keep_current(region=current_state.region)

    # ---------------- explanation ----------------
    parts: list[str] = []
    parts.append(
        f"action={action.action_type.value} target={target_region} "
        f"savings={action.expected_savings_pct:.1f}% tier={sla_policy.tier.value} "
        f"aggr={agg.value}"
    )
    if violated:
        parts.append("HARD VIOLATIONS: " + "; ".join(violated))
    else:
        parts.append("hard constraints satisfied")
    if soft_violations:
        parts.append("soft: " + "; ".join(soft_violations))
    if unknown:
        parts.append(f"unknown metrics (not blocking): {unknown}")
    parts.append(
        f"risk={risk_score:.2f} (mig={rb.migration_penalty:.2f} "
        f"lat={rb.latency_risk_penalty:.2f} q={rb.queue_risk_penalty:.2f} "
        f"avail={rb.availability_risk_penalty:.2f} cap={rb.capacity_risk_penalty:.2f})"
    )
    if corrected is not None:
        parts.append(f"corrected_action={corrected.action_type.value}")

    return SLAEvaluation(
        action=action,
        allowed=allowed,
        violated_hard_constraints=violated,
        soft_violations=soft_violations,
        soft_penalty_score=round(soft_penalty, 4),
        risk_score=round(risk_score, 4),
        risk_breakdown=rb,
        corrected_action=corrected,
        explanation=" | ".join(parts),
        unknown_metrics=unknown,
    )
