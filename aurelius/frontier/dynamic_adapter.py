"""Adapter — turn a dynamic decision into a static-compatible
:class:`FrontierDecision`.

The existing ``constraint_aware`` ↔ frontier integration shim
(``aurelius/constraints/frontier_integration.py``) consumes a static
:class:`FrontierDecision`. The dynamic estimator emits a
:class:`DynamicFrontierDecision`. This module provides the one-way
conversion so the constraint-aware integration can opt into the
dynamic path without changing its public API.

Hard rules:

- The static decision returned here is *recommendation-only*
  (``executable_in_real_cluster=False``).
- ``execution_mode`` defaults to ``shadow``; the caller controls
  promotion via the existing ``execute_frontier_decision`` opt-in.
- The dynamic action mapping is lossless and explicit:
  ``RAISE_RHO`` / ``LOWER_RHO`` → ``RECOMMEND_RHO`` (or ``LOWER_RHO``
  when the recommended rho is strictly below current);
  ``KEEP_RHO`` → ``KEEP_RHO``; ``INSUFFICIENT_TELEMETRY`` →
  ``INSUFFICIENT_TELEMETRY``.
"""

from __future__ import annotations

from typing import Optional

from .dynamic_models import (
    DynamicFrontierCandidate,
    DynamicFrontierDecision,
)
from .models import (
    EXECUTION_MODE_SHADOW,
    FrontierAction,
    FrontierDecision,
    FrontierPoint,
    SafetyStatus,
)


def _candidate_to_point(c: DynamicFrontierCandidate) -> FrontierPoint:
    """Convert a dynamic candidate to a static :class:`FrontierPoint`."""
    return FrontierPoint(
        rho_target=c.rho_target,
        predicted_goodput_per_dollar=c.predicted_goodput_per_dollar,
        predicted_gpu_hours=c.predicted_gpu_hours,
        predicted_timeout_pct=c.predicted_timeout_pct,
        predicted_queue_p99_ms=c.predicted_queue_p99_ms,
        predicted_latency_p99_ms=c.predicted_latency_p99_ms,
        predicted_churn_score=c.predicted_churn_score,
        safety_status=c.safety_status,
        safety_vetoes=tuple(c.safety_vetoes),
        notes=tuple(c.risk_reason_codes),
    )


def dynamic_estimate_to_frontier_decision(
    decision: DynamicFrontierDecision,
    *,
    candidate_points: Optional[list] = None,
    execution_mode: str = EXECUTION_MODE_SHADOW,
) -> FrontierDecision:
    """Convert a dynamic decision to a static :class:`FrontierDecision`.

    ``candidate_points`` is the dynamic estimate's candidate list (a
    sequence of :class:`DynamicFrontierCandidate`). When omitted, the
    static decision carries an empty frontier_points tuple.
    """
    if decision.action == "INSUFFICIENT_TELEMETRY":
        static_action = FrontierAction.INSUFFICIENT_TELEMETRY
    elif decision.action == "KEEP_RHO":
        static_action = FrontierAction.KEEP_RHO
    elif decision.action == "LOWER_RHO":
        static_action = FrontierAction.LOWER_RHO
    elif decision.action == "RAISE_RHO":
        static_action = FrontierAction.RECOMMEND_RHO
    else:  # pragma: no cover - DYNAMIC_ACTIONS guards the input
        static_action = FrontierAction.RECOMMEND_RHO

    static_points = tuple(_candidate_to_point(c)
                          for c in (candidate_points or ()))
    selected_point = None
    if static_points and decision.recommended_rho is not None:
        for p in static_points:
            if abs(p.rho_target - decision.recommended_rho) < 1e-9:
                selected_point = p
                break

    return FrontierDecision(
        workload_id=decision.workload_id,
        selected_rho=decision.recommended_rho,
        selected_point=selected_point,
        frontier_points=static_points,
        action=static_action,
        reason=decision.reason,
        previous_rho=decision.current_rho,
        expected_goodput_per_dollar_delta=
            decision.expected_goodput_per_dollar_delta,
        expected_gpu_hour_delta=decision.expected_gpu_hour_delta,
        expected_sla_risk_delta=decision.expected_sla_risk_delta,
        confidence=decision.confidence,
        execution_mode=execution_mode,
        executable_in_simulator=decision.executable_in_simulator,
        # Recommendation-only at construction (static-decision invariant).
        executable_in_real_cluster=False,
        safety_vetoes=tuple(decision.safety_vetoes),
        source=decision.source,
    )
