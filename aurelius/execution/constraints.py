"""Execution constraint profiles for Aurelius.

This module provides constraint evaluation for execution decisions.
Constraints do NOT modify decisions - they evaluate whether a decision
satisfies the specified constraint profile.

Constraint Profiles:
- batch_optimized: Allow full optimization flexibility (default)
- latency_safe: Enforce zero-slack, start-time preservation, resource integrity

The latency_safe profile guarantees:
1. Zero-slack: |optimized_start - baseline_start| <= latency_slack_threshold_hours
2. Start-time preservation: Jobs run at their originally scheduled time
3. Resource integrity: No CPU/memory reduction (power_fraction >= 1.0)

When constraints cannot be satisfied, the evaluator returns a fallback
recommendation to use baseline execution.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional
import json
import logging

from ..models import ScheduleDecision
from .base import ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class ConstraintEvaluation:
    """Result of evaluating a decision against constraint profile.

    Attributes:
        job_id: The job being evaluated
        decision: The optimized decision being evaluated
        baseline_decision: The baseline decision for comparison (if provided)
        constraint_profile: The profile used for evaluation
        passed: True if all constraints are satisfied
        violations: List of constraint violations (empty if passed)
        fallback_to_baseline: True if baseline execution should be used
        slack_hours: Actual slack between optimized and baseline (if applicable)
        power_fraction: Power fraction from the decision
    """
    job_id: str
    decision: ScheduleDecision
    baseline_decision: Optional[ScheduleDecision]
    constraint_profile: Literal["batch_optimized", "latency_safe"]
    passed: bool
    violations: list[str]
    fallback_to_baseline: bool
    slack_hours: Optional[float] = None
    power_fraction: float = 1.0

    def to_audit_dict(self) -> dict:
        """Convert to dictionary for audit logging."""
        result = {
            "job_id": self.job_id,
            "constraint_profile": self.constraint_profile,
            "passed": self.passed,
            "violations": self.violations,
            "fallback_to_baseline": self.fallback_to_baseline,
            "power_fraction": self.power_fraction,
            "optimized_start": self.decision.start_time.isoformat(),
            "optimized_region": self.decision.region,
        }
        if self.slack_hours is not None:
            result["slack_hours"] = self.slack_hours
        if self.baseline_decision is not None:
            result["baseline_start"] = self.baseline_decision.start_time.isoformat()
            result["baseline_region"] = self.baseline_decision.region
        return result

    def to_audit_json(self) -> str:
        """Convert to JSON string for audit logging."""
        return json.dumps(self.to_audit_dict())


class ConstraintEvaluator:
    """Evaluates execution decisions against constraint profiles.

    This evaluator checks whether optimized scheduling decisions satisfy
    the specified constraint profile. It does NOT modify decisions.

    For latency_safe profile, evaluates:
    - Zero-slack requirement: start time deviation within threshold
    - Resource integrity: power_fraction >= 1.0 (no throttling)

    Usage:
        evaluator = ConstraintEvaluator()
        config = ExecutionConfig(constraint_profile="latency_safe")

        # With baseline decisions for comparison
        evaluations = evaluator.evaluate(
            optimized_decisions,
            baseline_decisions,
            config,
        )

        # Check which decisions can proceed
        for eval in evaluations:
            if eval.passed:
                # Execute optimized decision
                pass
            elif eval.fallback_to_baseline:
                # Execute baseline decision instead
                pass
    """

    def evaluate(
        self,
        decisions: list[ScheduleDecision],
        baseline_decisions: Optional[list[ScheduleDecision]],
        config: ExecutionConfig,
    ) -> list[ConstraintEvaluation]:
        """Evaluate a list of decisions against constraint profile.

        Args:
            decisions: Optimized scheduling decisions to evaluate
            baseline_decisions: Baseline decisions for comparison (required for latency_safe)
            config: ExecutionConfig with constraint_profile setting

        Returns:
            List of ConstraintEvaluation results, one per decision
        """
        # Build baseline lookup map
        baseline_map: dict[str, ScheduleDecision] = {}
        if baseline_decisions:
            for bd in baseline_decisions:
                baseline_map[bd.job_id] = bd

        evaluations = []
        for decision in decisions:
            baseline = baseline_map.get(decision.job_id)
            evaluation = self._evaluate_single(decision, baseline, config)
            evaluations.append(evaluation)

            # Log audit record
            self._log_evaluation_audit(evaluation, config)

        return evaluations

    def _evaluate_single(
        self,
        decision: ScheduleDecision,
        baseline_decision: Optional[ScheduleDecision],
        config: ExecutionConfig,
    ) -> ConstraintEvaluation:
        """Evaluate a single decision against constraint profile.

        Args:
            decision: The optimized decision to evaluate
            baseline_decision: The baseline decision for comparison
            config: ExecutionConfig with constraint settings

        Returns:
            ConstraintEvaluation result
        """
        # batch_optimized: Always passes, no constraints
        if config.is_batch_optimized():
            return ConstraintEvaluation(
                job_id=decision.job_id,
                decision=decision,
                baseline_decision=baseline_decision,
                constraint_profile="batch_optimized",
                passed=True,
                violations=[],
                fallback_to_baseline=False,
                slack_hours=None,
                power_fraction=decision.power_fraction,
            )

        # latency_safe: Enforce strict constraints
        violations = []
        slack_hours = None

        # Constraint 1: Zero-slack (start time preservation)
        if baseline_decision is not None:
            slack_seconds = abs(
                (decision.start_time - baseline_decision.start_time).total_seconds()
            )
            slack_hours = slack_seconds / 3600
            threshold = config.latency_slack_threshold_hours

            if slack_hours > threshold:
                violations.append(
                    f"slack_violation: {slack_hours:.4f}h > threshold {threshold}h"
                )
        else:
            # No baseline to compare - cannot verify zero-slack
            # Conservative: treat as violation
            violations.append("no_baseline: cannot verify zero-slack constraint")

        # Constraint 2: Resource integrity (no power reduction)
        if decision.power_fraction < 1.0:
            violations.append(
                f"power_reduction: {decision.power_fraction:.2f} < 1.0 (no throttling allowed)"
            )

        # Determine outcome
        passed = len(violations) == 0
        fallback_to_baseline = not passed and baseline_decision is not None

        return ConstraintEvaluation(
            job_id=decision.job_id,
            decision=decision,
            baseline_decision=baseline_decision,
            constraint_profile="latency_safe",
            passed=passed,
            violations=violations,
            fallback_to_baseline=fallback_to_baseline,
            slack_hours=slack_hours,
            power_fraction=decision.power_fraction,
        )

    def _log_evaluation_audit(
        self,
        evaluation: ConstraintEvaluation,
        config: ExecutionConfig,
    ) -> None:
        """Log a structured audit record for constraint evaluation.

        Args:
            evaluation: The constraint evaluation result
            config: The execution configuration
        """
        audit_record = {
            "event": "constraint_evaluation",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": config.mode,
            "constraint_profile": config.constraint_profile,
            "latency_slack_threshold_hours": config.latency_slack_threshold_hours,
            "evaluation": evaluation.to_audit_dict(),
        }

        logger.info(f"AUDIT: {json.dumps(audit_record)}")


def apply_constraint_filter(
    optimized_decisions: list[ScheduleDecision],
    baseline_decisions: list[ScheduleDecision],
    config: ExecutionConfig,
) -> list[ScheduleDecision]:
    """Apply constraint profile and return executable decisions.

    This is a convenience function that:
    1. Evaluates all decisions against the constraint profile
    2. Returns optimized decisions that pass constraints
    3. Substitutes baseline decisions for those that don't pass

    For batch_optimized: Returns all optimized decisions unchanged
    For latency_safe: Returns baseline for any decision that violates constraints

    Args:
        optimized_decisions: List of optimized scheduling decisions
        baseline_decisions: List of baseline scheduling decisions
        config: ExecutionConfig with constraint_profile

    Returns:
        List of decisions safe to execute (optimized or baseline fallback)
    """
    evaluator = ConstraintEvaluator()
    evaluations = evaluator.evaluate(optimized_decisions, baseline_decisions, config)

    # Build baseline lookup
    baseline_map = {bd.job_id: bd for bd in baseline_decisions}

    result = []
    for evaluation in evaluations:
        if evaluation.passed:
            # Use optimized decision
            result.append(evaluation.decision)
        elif evaluation.fallback_to_baseline:
            # Use baseline decision
            baseline = baseline_map.get(evaluation.job_id)
            if baseline is not None:
                result.append(baseline)
            else:
                # Should not happen, but include optimized as last resort
                result.append(evaluation.decision)
        else:
            # No fallback available - include optimized
            result.append(evaluation.decision)

    return result


# Inline tests
if __name__ == "__main__":
    from datetime import timedelta

    print("=" * 60)
    print("ConstraintEvaluator Inline Tests")
    print("=" * 60)

    # Test data
    now = datetime.utcnow()

    # Test 1: batch_optimized always passes
    print("\n[Test 1] batch_optimized profile - should always pass")
    config_batch = ExecutionConfig(constraint_profile="batch_optimized")
    decision1 = ScheduleDecision(
        job_id="job-001",
        start_time=now + timedelta(hours=2),  # Different from baseline
        region="us-west",
        power_fraction=0.5,  # Throttled
        actual_runtime_hours=2.0,
    )
    baseline1 = ScheduleDecision(
        job_id="job-001",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evaluator = ConstraintEvaluator()
    evals = evaluator.evaluate([decision1], [baseline1], config_batch)
    assert len(evals) == 1
    assert evals[0].passed is True
    assert evals[0].violations == []
    assert evals[0].fallback_to_baseline is False
    print(f"  PASSED: {evals[0].to_audit_dict()}")

    # Test 2: latency_safe with valid decision (same start time, full power)
    print("\n[Test 2] latency_safe profile - valid decision")
    config_latency = ExecutionConfig(
        constraint_profile="latency_safe",
        latency_slack_threshold_hours=0.05,  # ~3 minutes
    )
    decision2 = ScheduleDecision(
        job_id="job-002",
        start_time=now + timedelta(minutes=1),  # Within 3 min threshold
        region="us-west",
        power_fraction=1.0,  # Full power
        actual_runtime_hours=1.0,
    )
    baseline2 = ScheduleDecision(
        job_id="job-002",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision2], [baseline2], config_latency)
    assert len(evals) == 1
    assert evals[0].passed is True
    assert evals[0].violations == []
    assert evals[0].slack_hours < 0.05
    print(f"  PASSED: slack={evals[0].slack_hours:.4f}h")

    # Test 3: latency_safe with slack violation
    print("\n[Test 3] latency_safe profile - slack violation")
    decision3 = ScheduleDecision(
        job_id="job-003",
        start_time=now + timedelta(hours=1),  # 1 hour deviation
        region="us-west",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )
    baseline3 = ScheduleDecision(
        job_id="job-003",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision3], [baseline3], config_latency)
    assert len(evals) == 1
    assert evals[0].passed is False
    assert any("slack_violation" in v for v in evals[0].violations)
    assert evals[0].fallback_to_baseline is True
    print(f"  PASSED: violations={evals[0].violations}")

    # Test 4: latency_safe with power reduction violation
    print("\n[Test 4] latency_safe profile - power reduction violation")
    decision4 = ScheduleDecision(
        job_id="job-004",
        start_time=now,  # Same as baseline
        region="us-west",
        power_fraction=0.8,  # Throttled
        actual_runtime_hours=1.25,
    )
    baseline4 = ScheduleDecision(
        job_id="job-004",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision4], [baseline4], config_latency)
    assert len(evals) == 1
    assert evals[0].passed is False
    assert any("power_reduction" in v for v in evals[0].violations)
    assert evals[0].fallback_to_baseline is True
    print(f"  PASSED: violations={evals[0].violations}")

    # Test 5: latency_safe with multiple violations
    print("\n[Test 5] latency_safe profile - multiple violations")
    decision5 = ScheduleDecision(
        job_id="job-005",
        start_time=now + timedelta(hours=2),  # Slack violation
        region="us-west",
        power_fraction=0.5,  # Power violation
        actual_runtime_hours=2.0,
    )
    baseline5 = ScheduleDecision(
        job_id="job-005",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision5], [baseline5], config_latency)
    assert len(evals) == 1
    assert evals[0].passed is False
    assert len(evals[0].violations) == 2
    assert evals[0].fallback_to_baseline is True
    print(f"  PASSED: violations={evals[0].violations}")

    # Test 6: latency_safe with no baseline (conservative failure)
    print("\n[Test 6] latency_safe profile - no baseline available")
    decision6 = ScheduleDecision(
        job_id="job-006",
        start_time=now,
        region="us-west",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision6], [], config_latency)
    assert len(evals) == 1
    assert evals[0].passed is False
    assert any("no_baseline" in v for v in evals[0].violations)
    assert evals[0].fallback_to_baseline is False  # No baseline to fall back to
    print(f"  PASSED: violations={evals[0].violations}")

    # Test 7: apply_constraint_filter convenience function
    print("\n[Test 7] apply_constraint_filter - mixed results")
    config_filter = ExecutionConfig(
        constraint_profile="latency_safe",
        latency_slack_threshold_hours=0.05,
    )

    # Decision A: passes constraints
    decision_a = ScheduleDecision(
        job_id="job-A",
        start_time=now + timedelta(minutes=1),
        region="us-west",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )
    baseline_a = ScheduleDecision(
        job_id="job-A",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    # Decision B: fails constraints (should fall back to baseline)
    decision_b = ScheduleDecision(
        job_id="job-B",
        start_time=now + timedelta(hours=1),
        region="us-west",
        power_fraction=0.5,
        actual_runtime_hours=2.0,
    )
    baseline_b = ScheduleDecision(
        job_id="job-B",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    result = apply_constraint_filter(
        [decision_a, decision_b],
        [baseline_a, baseline_b],
        config_filter,
    )

    assert len(result) == 2
    # First decision should be optimized (passed)
    assert result[0].job_id == "job-A"
    assert result[0].region == "us-west"  # Optimized region
    # Second decision should be baseline (failed)
    assert result[1].job_id == "job-B"
    assert result[1].region == "us-east"  # Baseline region
    print(f"  PASSED: result regions={[d.region for d in result]}")

    # Test 8: Exact threshold boundary
    print("\n[Test 8] latency_safe profile - exact threshold boundary")
    config_threshold = ExecutionConfig(
        constraint_profile="latency_safe",
        latency_slack_threshold_hours=0.05,  # 3 minutes exactly
    )
    # At exactly 3 minutes - should pass
    decision8a = ScheduleDecision(
        job_id="job-008a",
        start_time=now + timedelta(minutes=3),
        region="us-west",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )
    baseline8 = ScheduleDecision(
        job_id="job-008a",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision8a], [baseline8], config_threshold)
    # 3 minutes = 0.05 hours, threshold is 0.05 hours, should pass
    assert evals[0].passed is True
    print(f"  PASSED: boundary at exactly threshold passes")

    # Test 9: Just over threshold
    decision8b = ScheduleDecision(
        job_id="job-008b",
        start_time=now + timedelta(minutes=4),  # 4 min > 3 min threshold
        region="us-west",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )
    baseline8b = ScheduleDecision(
        job_id="job-008b",
        start_time=now,
        region="us-east",
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision8b], [baseline8b], config_threshold)
    assert evals[0].passed is False
    print(f"  PASSED: just over threshold fails")

    # Test 10: Region change allowed in latency_safe (only start time and power matter)
    print("\n[Test 10] latency_safe - region change allowed if timing/power OK")
    decision10 = ScheduleDecision(
        job_id="job-010",
        start_time=now,  # Same as baseline
        region="eu-west",  # Different region
        power_fraction=1.0,  # Full power
        actual_runtime_hours=1.0,
    )
    baseline10 = ScheduleDecision(
        job_id="job-010",
        start_time=now,
        region="us-east",  # Original region
        power_fraction=1.0,
        actual_runtime_hours=1.0,
    )

    evals = evaluator.evaluate([decision10], [baseline10], config_latency)
    assert evals[0].passed is True
    print(f"  PASSED: region change allowed (start/power constraints satisfied)")

    print("\n" + "=" * 60)
    print("All 10 tests passed!")
    print("=" * 60)
