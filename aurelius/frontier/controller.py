"""Safe Utilization Frontier Controller — selection logic.

Given a candidate frontier (a list of :class:`FrontierPoint`), pick the
highest *SLA-safe* goodput/$ point — not the highest utilization. Apply a
conservative margin to step back from the safety boundary, a deadband to
avoid churn around the current rho, and a hard LOWER_RHO branch when the
current rho is itself unsafe.

Selection rules (binding):

1. Telemetry-confidence gate → INSUFFICIENT_TELEMETRY.
2. Filter to safe points only.
3. If the current rho is UNSAFE → LOWER_RHO.
4. Else pick the safe point with the highest predicted goodput/$.
5. Apply optional conservative margin: if the best point is *adjacent* to a
   first-unsafe point, prefer the next-lower safe rho (transparent).
6. Deadband: if the selected rho is within ``deadband`` of the current rho
   and the KPI delta is small → KEEP_RHO.
7. If no safe points → LOWER_RHO (with the lowest available candidate as
   the recommendation), or INSUFFICIENT_TELEMETRY if telemetry is also
   inadequate.

The controller emits ``execution_mode=shadow`` by default; the caller
chooses ``simulator`` or ``real_disabled`` / ``real_enabled`` via
``execute_frontier_decision`` (see ``execution.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .models import (
    EXECUTION_MODE_SHADOW,
    FrontierAction,
    FrontierDecision,
    FrontierPoint,
    SafetyStatus,
    WorkloadFrontierProfile,
)

_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class FrontierControllerConfig:
    """Controller settings (transparent, all configurable per workload).

    - ``conservative_margin``: if > 0, the controller prefers the next-lower
      safe rho when the best point sits at the safety boundary (the
      adjacent candidate is UNSAFE / INSUFFICIENT_TELEMETRY).
    - ``deadband_rho``: rho changes smaller than this are KEEP_RHO when the
      KPI delta is also small.
    - ``deadband_kpi_pct``: KPI delta fraction (e.g. 0.01 = 1 %) below which
      a rho change within the rho deadband collapses to KEEP_RHO.
    - ``min_telemetry_confidence``: workload telemetry-confidence label
      required to act; below this the controller returns
      INSUFFICIENT_TELEMETRY.
    """

    conservative_margin: bool = False
    deadband_rho: float = 0.05
    deadband_kpi_pct: float = 0.01
    min_telemetry_confidence: str = "low"
    default_execution_mode: str = EXECUTION_MODE_SHADOW


def _safe_points(points: Iterable[FrontierPoint]) -> list[FrontierPoint]:
    return [p for p in points if p.is_safe]


def _point_at_rho(points: Iterable[FrontierPoint], rho: Optional[float]
                  ) -> Optional[FrontierPoint]:
    if rho is None:
        return None
    for p in points:
        if abs(p.rho_target - rho) < 1e-9:
            return p
    return None


def _next_lower_safe(points: list[FrontierPoint], best: FrontierPoint
                     ) -> Optional[FrontierPoint]:
    """Next lower rho whose point is SAFE (skipping non-safe points)."""
    candidates = sorted([p for p in points if p.rho_target < best.rho_target
                         and p.is_safe], key=lambda p: p.rho_target,
                        reverse=True)
    return candidates[0] if candidates else None


def _adjacent_first_unsafe(all_points: list[FrontierPoint], best: FrontierPoint
                           ) -> Optional[FrontierPoint]:
    """The first not-SAFE point at rho > best.rho_target, if any."""
    above = sorted([p for p in all_points if p.rho_target > best.rho_target],
                   key=lambda p: p.rho_target)
    for p in above:
        if not p.is_safe:
            return p
    return None


def _kpi(point: Optional[FrontierPoint]) -> Optional[float]:
    return point.predicted_goodput_per_dollar if point is not None else None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _conf_blend(*labels: str) -> str:
    rank = min((_CONF_RANK.get(x or "unknown", 0) for x in labels), default=0)
    for name, r in _CONF_RANK.items():
        if r == rank:
            return name
    return "unknown"


def choose_safe_utilization_target(profile: WorkloadFrontierProfile,
                                   frontier_points: Iterable[FrontierPoint],
                                   current_rho: Optional[float],
                                   controller_config: Optional[
                                       FrontierControllerConfig] = None,
                                   ) -> FrontierDecision:
    """Pick the highest-KPI safe rho for ``profile``.

    Returns a recommendation-only :class:`FrontierDecision`
    (``executable_in_real_cluster=False``); ``execution_mode`` defaults to
    ``shadow``. The caller decides whether to mutate via
    ``execute_frontier_decision``.
    """
    cfg = controller_config or FrontierControllerConfig()
    points = list(frontier_points)

    # --- 1. telemetry-confidence gate ---
    needed = _CONF_RANK.get(cfg.min_telemetry_confidence, 0)
    have = _CONF_RANK.get(profile.telemetry_confidence or "unknown", 0)
    if have < needed or not points or all(p.is_insufficient_telemetry
                                          for p in points):
        return FrontierDecision(
            workload_id=profile.workload_id, selected_rho=None,
            selected_point=None, frontier_points=tuple(points),
            action=FrontierAction.INSUFFICIENT_TELEMETRY,
            reason=("workload telemetry confidence "
                    f"{profile.telemetry_confidence!r} below required "
                    f"{cfg.min_telemetry_confidence!r}"
                    if have < needed
                    else "no frontier points have sufficient telemetry"),
            previous_rho=current_rho, confidence="low",
            execution_mode=cfg.default_execution_mode,
            safety_vetoes=tuple(sorted({v for p in points for v in p.safety_vetoes})),
            executable_in_simulator=False)

    safe = _safe_points(points)
    current_point = _point_at_rho(points, current_rho)

    # --- 3. current rho UNSAFE → LOWER_RHO ---
    if (current_point is not None
            and current_point.safety_status == SafetyStatus.UNSAFE):
        target = _next_lower_safe(points, current_point)
        if target is None:
            # no safe rho lower than current: still LOWER_RHO toward profile.min_rho
            below = sorted([p for p in points if p.rho_target < current_point.rho_target],
                           key=lambda p: p.rho_target)
            target = below[0] if below else None
        if target is None:
            return FrontierDecision(
                workload_id=profile.workload_id, selected_rho=profile.min_rho,
                selected_point=None, frontier_points=tuple(points),
                action=FrontierAction.LOWER_RHO,
                reason=("current rho is unsafe and no lower safe candidate is "
                        "available; recommend the workload floor"),
                previous_rho=current_rho,
                expected_sla_risk_delta=-1.0,
                safety_vetoes=tuple(current_point.safety_vetoes),
                confidence=_conf_blend(profile.telemetry_confidence),
                execution_mode=cfg.default_execution_mode)
        return FrontierDecision(
            workload_id=profile.workload_id, selected_rho=target.rho_target,
            selected_point=target, frontier_points=tuple(points),
            action=FrontierAction.LOWER_RHO,
            reason=("current rho violates safety gates "
                    f"({', '.join(current_point.safety_vetoes) or 'unsafe'}); "
                    f"recommend lowering to nearest safe rho {target.rho_target}"),
            previous_rho=current_rho,
            expected_goodput_per_dollar_delta=_delta(_kpi(target), _kpi(current_point)),
            expected_gpu_hour_delta=_delta(target.predicted_gpu_hours,
                                           current_point.predicted_gpu_hours),
            expected_sla_risk_delta=-1.0,
            safety_vetoes=tuple(current_point.safety_vetoes),
            confidence=_conf_blend(profile.telemetry_confidence),
            execution_mode=cfg.default_execution_mode)

    # --- 7. no safe points → LOWER_RHO toward the smallest candidate ---
    if not safe:
        # if telemetry is the dominant gap, return INSUFFICIENT_TELEMETRY
        if all(p.is_insufficient_telemetry for p in points):
            return FrontierDecision(
                workload_id=profile.workload_id, selected_rho=None,
                selected_point=None, frontier_points=tuple(points),
                action=FrontierAction.INSUFFICIENT_TELEMETRY,
                reason="every candidate rho has insufficient telemetry",
                previous_rho=current_rho, confidence="low",
                safety_vetoes=tuple(sorted({v for p in points
                                            for v in p.safety_vetoes})),
                execution_mode=cfg.default_execution_mode,
                executable_in_simulator=False)
        lowest = min(points, key=lambda p: p.rho_target)
        return FrontierDecision(
            workload_id=profile.workload_id, selected_rho=lowest.rho_target,
            selected_point=lowest, frontier_points=tuple(points),
            action=FrontierAction.LOWER_RHO,
            reason="no safe candidate rho; recommend lowering to the smallest tested rho",
            previous_rho=current_rho,
            expected_sla_risk_delta=-1.0,
            safety_vetoes=tuple(sorted({v for p in points for v in p.safety_vetoes})),
            confidence=_conf_blend(profile.telemetry_confidence),
            execution_mode=cfg.default_execution_mode)

    # --- 4. choose highest goodput/$ among safe points ---
    best = max(safe, key=lambda p: (p.predicted_goodput_per_dollar or 0.0))

    # --- 5. conservative margin: step back from boundary if adjacent unsafe ---
    margin_notes = []
    if cfg.conservative_margin:
        adj = _adjacent_first_unsafe(points, best)
        if adj is not None and abs(adj.rho_target - best.rho_target) <= 0.10 + 1e-9:
            lower = _next_lower_safe(points, best)
            if lower is not None:
                margin_notes.append(
                    f"conservative_margin: best safe rho {best.rho_target} is "
                    f"adjacent to unsafe rho {adj.rho_target}; stepping back to "
                    f"{lower.rho_target}")
                best = lower

    # --- 6. deadband / churn avoidance vs current rho ---
    action = FrontierAction.RECOMMEND_RHO
    reason = (f"highest SLA-safe goodput/$ at rho {best.rho_target} "
              f"(predicted {best.predicted_goodput_per_dollar:,.2f} "
              "across all SLA / queue / latency / telemetry gates)")
    gpd_delta = (None if current_point is None
                 else _delta(_kpi(best), _kpi(current_point)))
    if current_point is not None and current_rho is not None:
        if abs(best.rho_target - current_rho) <= cfg.deadband_rho:
            kpi_now = _kpi(current_point) or 0.0
            kpi_delta_pct = (abs(gpd_delta or 0.0) / kpi_now) if kpi_now else 0.0
            if kpi_delta_pct <= cfg.deadband_kpi_pct:
                action = FrontierAction.KEEP_RHO
                reason = (
                    f"selected rho {best.rho_target} within deadband "
                    f"{cfg.deadband_rho} of current rho {current_rho} and KPI "
                    f"delta {kpi_delta_pct:.4f} ≤ {cfg.deadband_kpi_pct}; "
                    "keep current rho to avoid churn")

    notes = tuple(margin_notes) if margin_notes else ()
    confidence = _conf_blend(profile.telemetry_confidence)
    selected_rho = best.rho_target if action != FrontierAction.KEEP_RHO else current_rho
    selected_point = (best if action != FrontierAction.KEEP_RHO else current_point)

    return FrontierDecision(
        workload_id=profile.workload_id, selected_rho=selected_rho,
        selected_point=selected_point,
        frontier_points=tuple(points),
        action=action,
        reason=reason + (f" [{'; '.join(notes)}]" if notes else ""),
        previous_rho=current_rho,
        expected_goodput_per_dollar_delta=gpd_delta,
        expected_gpu_hour_delta=(None if current_point is None
                                 else _delta(best.predicted_gpu_hours,
                                             current_point.predicted_gpu_hours)),
        expected_sla_risk_delta=0.0,
        safety_vetoes=(),
        confidence=confidence,
        execution_mode=cfg.default_execution_mode)
