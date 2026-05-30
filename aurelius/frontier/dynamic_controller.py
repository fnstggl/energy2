"""Dynamic frontier controller — turns an estimate into a decision.

Given a :class:`DynamicFrontierEstimate` and the current rho operating
point, emit a recommendation-only :class:`DynamicFrontierDecision`.
The controller is conservative-by-default:

- INSUFFICIENT_TELEMETRY when the estimator already returned a fallback.
- LOWER_RHO when the risk at the current rho exceeds the configured
  unsafe-risk threshold.
- KEEP_RHO when the recommended rho is within the deadband of the
  current rho or the expected KPI delta is below the configured
  minimum (hysteresis).
- LOWER_RHO when the recommendation is strictly below current.
- RAISE_RHO when the recommendation is strictly above current.

Real-cluster execution stays disabled by default
(``executable_in_real_cluster=False``); the dynamic decision flows into
the existing ``execute_frontier_decision`` shim with the same opt-in
guard as the static controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .dynamic_models import (
    DynamicFrontierDecision,
    DynamicFrontierEstimate,
)


@dataclass
class DynamicControllerConfig:
    """Settings for :func:`choose_dynamic_rho`."""

    # Rho changes within this band collapse to KEEP_RHO.
    deadband_rho: float = 0.05
    # Expected goodput/$ delta (fraction of current) below which a
    # within-deadband change is also treated as KEEP_RHO. The dynamic
    # estimator's goodput/$ is in 1/gpu-hour units, so this is a
    # relative-fraction threshold.
    deadband_kpi_pct: float = 0.02
    # Above this risk score at the current rho, recommend LOWER_RHO.
    lower_rho_risk_threshold: float = 0.75
    # Hysteresis: once the controller has just changed rho in one
    # direction, suppress the opposite-direction recommendation unless
    # the magnitude of change exceeds this multiplier of the deadband.
    hysteresis_multiplier: float = 2.0
    # Churn suppression: when the workload is "churn-y" (the estimator
    # supplies risk_reason_codes containing "scale_events_high" or
    # "churn_high"), suppress RAISE_RHO.
    churn_suppresses_raise: bool = True


def choose_dynamic_rho(
    estimate: DynamicFrontierEstimate,
    *,
    current_rho: Optional[float],
    config: Optional[DynamicControllerConfig] = None,
    previous_action: Optional[str] = None,
) -> DynamicFrontierDecision:
    """Pick a recommendation-only :class:`DynamicFrontierDecision`."""
    cfg = config or DynamicControllerConfig()

    # 1. Insufficient telemetry — pass through the estimator's fallback.
    if estimate.recommended_rho is None or estimate.fallback_reason:
        return DynamicFrontierDecision(
            workload_id=estimate.workload_id,
            current_rho=current_rho,
            recommended_rho=None,
            action="INSUFFICIENT_TELEMETRY",
            reason=(estimate.fallback_reason
                    or "estimator returned no recommendation"),
            confidence="low",
            fallback_reason=(estimate.fallback_reason
                             or "no_recommended_rho"),
            notes=tuple(estimate.notes))

    cur = (current_rho if current_rho is not None
           else estimate.current_rho_estimate)

    # 2. Current rho unsafe → LOWER_RHO.
    if (estimate.risk_at_current_rho is not None
            and estimate.risk_at_current_rho >= cfg.lower_rho_risk_threshold):
        return DynamicFrontierDecision(
            workload_id=estimate.workload_id,
            current_rho=cur,
            recommended_rho=min(estimate.recommended_rho, cur or 0.65)
                            if cur is not None else estimate.recommended_rho,
            action="LOWER_RHO",
            reason=(f"risk_at_current_rho={estimate.risk_at_current_rho:.3f} "
                    f">= lower_rho_risk_threshold "
                    f"{cfg.lower_rho_risk_threshold}"),
            expected_sla_risk_delta=-1.0,
            confidence=estimate.confidence,
            safety_vetoes=tuple(
                v for c in estimate.candidate_points
                if cur is not None and abs(c.rho_target - cur) < 0.05
                for v in c.safety_vetoes))

    # 3. Deadband — small rho change with small KPI delta → KEEP_RHO.
    if cur is not None:
        rho_delta = estimate.recommended_rho - cur
        if abs(rho_delta) <= cfg.deadband_rho:
            # KPI delta evaluation
            kpi_delta_pct = 0.0
            best = next((c for c in estimate.candidate_points
                         if abs(c.rho_target - estimate.recommended_rho)
                         < 1e-9), None)
            cur_cand = next((c for c in estimate.candidate_points
                              if abs(c.rho_target - cur) < 0.05), None)
            if (best is not None and cur_cand is not None
                    and best.predicted_goodput_per_dollar
                    and cur_cand.predicted_goodput_per_dollar):
                kpi_delta_pct = (abs(best.predicted_goodput_per_dollar
                                     - cur_cand.predicted_goodput_per_dollar)
                                 / cur_cand.predicted_goodput_per_dollar)
            if kpi_delta_pct <= cfg.deadband_kpi_pct:
                return DynamicFrontierDecision(
                    workload_id=estimate.workload_id,
                    current_rho=cur, recommended_rho=cur,
                    action="KEEP_RHO",
                    reason=(f"|Δρ|={abs(rho_delta):.4f} ≤ deadband "
                            f"{cfg.deadband_rho} and ΔKPI≈"
                            f"{kpi_delta_pct:.4f} ≤ "
                            f"{cfg.deadband_kpi_pct}"),
                    expected_goodput_per_dollar_delta=0.0,
                    expected_gpu_hour_delta=0.0,
                    expected_sla_risk_delta=0.0,
                    confidence=estimate.confidence,
                    hysteresis_applied=True,
                    notes=("deadband_collapsed_to_keep",))

    # 4. Churn suppression — don't raise rho on a churn-y workload.
    if (cfg.churn_suppresses_raise and cur is not None
            and estimate.recommended_rho > cur):
        any_churn = any(
            ("churn_high" in (c.risk_reason_codes or ())
             or "scale_events_high" in (c.risk_reason_codes or ()))
            for c in estimate.candidate_points)
        if any_churn:
            return DynamicFrontierDecision(
                workload_id=estimate.workload_id,
                current_rho=cur, recommended_rho=cur,
                action="KEEP_RHO",
                reason="churn signals suppress RAISE_RHO (conservative)",
                expected_goodput_per_dollar_delta=0.0,
                confidence=estimate.confidence,
                hysteresis_applied=True,
                notes=("churn_suppressed_raise",))

    # 5. Hysteresis vs previous_action — flip suppression.
    if previous_action and cur is not None:
        flip = ((previous_action == "RAISE_RHO"
                 and estimate.recommended_rho < cur)
                or (previous_action == "LOWER_RHO"
                    and estimate.recommended_rho > cur))
        if flip:
            magnitude = abs(estimate.recommended_rho - cur)
            if magnitude < cfg.deadband_rho * cfg.hysteresis_multiplier:
                return DynamicFrontierDecision(
                    workload_id=estimate.workload_id,
                    current_rho=cur, recommended_rho=cur,
                    action="KEEP_RHO",
                    reason=(f"hysteresis: flip from {previous_action} "
                            f"with |Δρ|={magnitude:.4f} below "
                            f"{cfg.deadband_rho * cfg.hysteresis_multiplier:.4f}"),
                    confidence=estimate.confidence,
                    hysteresis_applied=True,
                    notes=("hysteresis_suppressed_flip",))

    # 6. Direction
    if cur is None:
        action = "RAISE_RHO" if estimate.recommended_rho > 0.65 else "LOWER_RHO"
    elif estimate.recommended_rho > cur:
        action = "RAISE_RHO"
    elif estimate.recommended_rho < cur:
        action = "LOWER_RHO"
    else:
        action = "KEEP_RHO"

    return DynamicFrontierDecision(
        workload_id=estimate.workload_id,
        current_rho=cur,
        recommended_rho=estimate.recommended_rho,
        action=action,
        reason=(f"highest SLA-safe goodput/$ at rho "
                f"{estimate.recommended_rho:.4f} via deterministic dynamic "
                f"estimator (Erlang-C tail + risk heuristic)"),
        expected_goodput_per_dollar_delta=None,
        expected_gpu_hour_delta=None,
        expected_sla_risk_delta=(
            (estimate.risk_at_recommended_rho or 0.0)
            - (estimate.risk_at_current_rho or 0.0)
            if estimate.risk_at_recommended_rho is not None
            and estimate.risk_at_current_rho is not None
            else None),
        confidence=estimate.confidence,
        hysteresis_applied=False,
        safety_vetoes=())
