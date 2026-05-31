"""Training Safe Utilization Frontier — unified controller (v1).

Given a sequence of :class:`TrainingFrontierPoint` (built by the
Philly / Alibaba GPU estimators), pick the **highest SLA-safe
goodput/$** candidate — not the densest packing, not the highest
occupancy. The controller is conservative-by-default:

1. ``INSUFFICIENT_TELEMETRY`` when every candidate has insufficient
   telemetry.
2. ``LOWER_PACKING_PRESSURE`` when the current policy violates
   fragmentation or starvation gates AND a safer candidate exists.
3. ``RESERVE_FOR_LARGE_JOBS`` when the workload exhibits large-job
   starvation AND a candidate with ``large_job_reservation_fraction``
   > 0 is the safe peak.
4. ``KEEP_CURRENT_POLICY`` when the current candidate is within the
   configured KPI / dimension deadbands of the selected candidate.
5. ``RECOMMEND_TRAINING_FRONTIER`` when a strictly better safe
   candidate exists outside the deadbands.

The training controller does **NOT** import the serving rho
controller. Tests assert this hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .training_models import (
    EXECUTION_MODE_SHADOW,
    TrainingFrontierAction,
    TrainingFrontierCandidate,
    TrainingFrontierDecision,
    TrainingFrontierPoint,
    TrainingSafetyStatus,
)

_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class TrainingControllerConfig:
    """Controller settings (transparent, all configurable per workload).

    Defaults mirror the conservative posture of the serving frontier
    controller's deadband / hysteresis logic; they are pre-registered
    and NOT tuned per dataset.
    """

    # KPI deadband — recommended ↔ current change with |Δgoodput/$|
    # below this collapses to KEEP_CURRENT_POLICY.
    deadband_kpi_pct: float = 0.01
    # Packing-density deadband — recommended ↔ current change with
    # |Δpacking_density| below this collapses to KEEP_CURRENT_POLICY.
    deadband_packing_density: float = 0.05
    # If True, when the best safe candidate is adjacent to an UNSAFE
    # point, the controller prefers the next-lower-pressure safe
    # candidate (mirrors the serving controller's
    # conservative_margin).
    conservative_margin: bool = False
    # When True, the controller emits LOWER_PACKING_PRESSURE if the
    # *current* candidate is UNSAFE, even when a higher-KPI safe
    # candidate exists at a different occupancy point.
    prefer_lower_pressure_on_current_unsafe: bool = True
    # Default execution mode.
    default_execution_mode: str = EXECUTION_MODE_SHADOW
    # Minimum telemetry confidence required to act.
    min_telemetry_confidence: str = "low"

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}")


def _safe_points(points: Iterable[TrainingFrontierPoint]
                 ) -> list[TrainingFrontierPoint]:
    return [p for p in points
            if p.safety_status == TrainingSafetyStatus.SAFE]


def _point_for_candidate(points: Iterable[TrainingFrontierPoint],
                          candidate: Optional[TrainingFrontierCandidate]
                          ) -> Optional[TrainingFrontierPoint]:
    if candidate is None:
        return None
    for p in points:
        if (p.candidate.source_policy is not None
                and p.candidate.source_policy
                == candidate.source_policy):
            return p
    return None


def _kpi(point: Optional[TrainingFrontierPoint]) -> Optional[float]:
    return point.predicted_goodput_per_dollar if point is not None else None


def _delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def _has_large_job_starvation(
    point: TrainingFrontierPoint,
    starvation_threshold_pct: float = 5.0,
) -> bool:
    """True if the point's starvation rate exceeds the threshold AND the
    workload's gpu_count distribution is consistent with large-job
    pressure (proxied here by the descriptor's
    ``gang_scheduling_strictness`` ≥ 0.5 — strict gang scheduling makes
    large jobs the dominant source of starvation)."""
    if point.predicted_starvation_rate_pct is None:
        return False
    if point.predicted_starvation_rate_pct < starvation_threshold_pct:
        return False
    if point.candidate.gang_scheduling_strictness is None:
        return False
    return point.candidate.gang_scheduling_strictness >= 0.5


def choose_training_frontier_target(
    points: Iterable[TrainingFrontierPoint],
    *,
    current_candidate: Optional[TrainingFrontierCandidate] = None,
    config: Optional[TrainingControllerConfig] = None,
    workload_id: str = "training_workload",
    telemetry_confidence: str = "medium",
) -> TrainingFrontierDecision:
    """Pick the highest-KPI safe candidate for the training workload."""
    cfg = config or TrainingControllerConfig()
    pts = list(points)

    # 1 — telemetry-confidence gate
    needed = _CONF_RANK.get(cfg.min_telemetry_confidence, 0)
    have = _CONF_RANK.get(telemetry_confidence or "unknown", 0)
    if have < needed or not pts or all(
            p.safety_status == TrainingSafetyStatus.INSUFFICIENT_TELEMETRY
            for p in pts):
        return TrainingFrontierDecision(
            workload_id=workload_id,
            selected_candidate=None,
            current_candidate=current_candidate,
            selected_point=None,
            frontier_points=tuple(pts),
            action=TrainingFrontierAction.INSUFFICIENT_TELEMETRY,
            reason=(
                f"workload telemetry_confidence={telemetry_confidence!r} "
                f"below required {cfg.min_telemetry_confidence!r}"
                if have < needed
                else "every candidate has insufficient telemetry"),
            confidence="low",
            execution_mode=cfg.default_execution_mode,
            safety_vetoes=tuple(sorted({v for p in pts
                                         for v in p.safety_vetoes})),
            executable_in_simulator=False)

    safe = _safe_points(pts)
    current_point = _point_for_candidate(pts, current_candidate)

    # 2 — current policy UNSAFE → LOWER_PACKING_PRESSURE
    if (current_point is not None
            and current_point.safety_status == TrainingSafetyStatus.UNSAFE
            and cfg.prefer_lower_pressure_on_current_unsafe):
        # Pick the highest-KPI safe candidate with **strictly lower
        # packing density** (or occupancy) than the current.
        cur_dens = (current_point.predicted_packing_density
                    if current_point.predicted_packing_density is not None
                    else current_point.predicted_gpu_occupancy)
        if cur_dens is not None:
            candidates_below = [p for p in safe
                                 if ((p.predicted_packing_density
                                       or p.predicted_gpu_occupancy or 0.0)
                                      < cur_dens)]
        else:
            candidates_below = list(safe)
        target = (max(candidates_below,
                      key=lambda p: (p.predicted_goodput_per_dollar or 0.0))
                  if candidates_below else
                  (max(safe,
                       key=lambda p: (p.predicted_goodput_per_dollar or 0.0))
                   if safe else None))
        if target is None:
            return TrainingFrontierDecision(
                workload_id=workload_id,
                selected_candidate=None,
                current_candidate=current_candidate,
                selected_point=None,
                frontier_points=tuple(pts),
                action=TrainingFrontierAction.LOWER_PACKING_PRESSURE,
                reason=("current policy is UNSAFE and no safer candidate "
                        "is available"),
                expected_goodput_per_dollar_delta=None,
                confidence=telemetry_confidence,
                safety_vetoes=tuple(current_point.safety_vetoes),
                execution_mode=cfg.default_execution_mode)
        return TrainingFrontierDecision(
            workload_id=workload_id,
            selected_candidate=target.candidate,
            current_candidate=current_candidate,
            selected_point=target,
            frontier_points=tuple(pts),
            action=TrainingFrontierAction.LOWER_PACKING_PRESSURE,
            reason=(f"current policy "
                    f"{current_point.candidate.source_policy!r} violates "
                    f"{', '.join(current_point.safety_vetoes) or 'safety'};"
                    f" recommend lower-pressure safe candidate "
                    f"{target.candidate.source_policy!r}"),
            expected_goodput_per_dollar_delta=_delta(
                _kpi(target), _kpi(current_point)),
            expected_gpu_hour_delta=_delta(target.predicted_gpu_hours,
                                            current_point.predicted_gpu_hours),
            expected_queue_wait_delta_s=_delta(
                target.predicted_queue_wait_p99_s,
                current_point.predicted_queue_wait_p99_s),
            expected_fragmentation_delta_pct=_delta(
                target.predicted_fragmentation_block_rate_pct,
                current_point.predicted_fragmentation_block_rate_pct),
            expected_starvation_delta_pct=_delta(
                target.predicted_starvation_rate_pct,
                current_point.predicted_starvation_rate_pct),
            confidence=telemetry_confidence,
            safety_vetoes=tuple(current_point.safety_vetoes),
            execution_mode=cfg.default_execution_mode)

    # 3 — no safe points → INSUFFICIENT_TELEMETRY or LOWER_PACKING_PRESSURE
    if not safe:
        if all(p.safety_status == TrainingSafetyStatus.INSUFFICIENT_TELEMETRY
                for p in pts):
            return TrainingFrontierDecision(
                workload_id=workload_id,
                selected_candidate=None,
                current_candidate=current_candidate,
                selected_point=None,
                frontier_points=tuple(pts),
                action=TrainingFrontierAction.INSUFFICIENT_TELEMETRY,
                reason="every candidate has insufficient telemetry",
                confidence="low",
                execution_mode=cfg.default_execution_mode,
                safety_vetoes=tuple(sorted({v for p in pts
                                             for v in p.safety_vetoes})),
                executable_in_simulator=False)
        # Pick the lowest-packing-pressure candidate as the LOWER recommendation
        ordered = sorted(pts,
                          key=lambda p: (p.predicted_packing_density
                                          or p.predicted_gpu_occupancy
                                          or 1.0))
        target = ordered[0] if ordered else None
        return TrainingFrontierDecision(
            workload_id=workload_id,
            selected_candidate=(target.candidate
                                  if target is not None else None),
            current_candidate=current_candidate,
            selected_point=target,
            frontier_points=tuple(pts),
            action=TrainingFrontierAction.LOWER_PACKING_PRESSURE,
            reason="no candidate is SAFE under the configured gates",
            confidence=telemetry_confidence,
            safety_vetoes=tuple(sorted({v for p in pts
                                         for v in p.safety_vetoes})),
            execution_mode=cfg.default_execution_mode)

    # 4 — choose highest goodput/$ among safe candidates
    best = max(safe, key=lambda p: (p.predicted_goodput_per_dollar or 0.0))

    # 5 — conservative margin: step back from boundary if adjacent unsafe
    notes: list[str] = []
    if cfg.conservative_margin:
        # Identify candidates with density just above ``best``; if any of
        # them is UNSAFE, prefer the next safer candidate strictly below
        # ``best``'s density.
        best_dens = (best.predicted_packing_density
                     or best.predicted_gpu_occupancy or 0.0)
        above = [p for p in pts
                 if (p.predicted_packing_density
                      or p.predicted_gpu_occupancy or 0.0) > best_dens]
        adj_unsafe = [p for p in above
                       if p.safety_status == TrainingSafetyStatus.UNSAFE]
        if adj_unsafe:
            below = [p for p in safe
                     if (p.predicted_packing_density
                          or p.predicted_gpu_occupancy or 0.0) < best_dens]
            if below:
                # Pick the highest-KPI safe candidate strictly below
                step = max(below,
                            key=lambda p: (p.predicted_goodput_per_dollar
                                            or 0.0))
                notes.append(
                    f"conservative_margin: best "
                    f"{best.candidate.source_policy!r} adjacent to UNSAFE "
                    f"{adj_unsafe[0].candidate.source_policy!r}; stepping "
                    f"down to {step.candidate.source_policy!r}")
                best = step

    # 6 — RESERVE_FOR_LARGE_JOBS path: if the best safe point exhibits
    # large-job starvation AND a candidate with a non-zero reservation
    # fraction exists, prefer the latter.
    if _has_large_job_starvation(best):
        reservers = [p for p in safe
                      if (p.candidate.large_job_reservation_fraction or 0.0)
                      > 0.0]
        if reservers:
            res = max(reservers,
                       key=lambda p: (p.predicted_goodput_per_dollar or 0.0))
            return TrainingFrontierDecision(
                workload_id=workload_id,
                selected_candidate=res.candidate,
                current_candidate=current_candidate,
                selected_point=res,
                frontier_points=tuple(pts),
                action=TrainingFrontierAction.RESERVE_FOR_LARGE_JOBS,
                reason=(f"best safe candidate "
                        f"{best.candidate.source_policy!r} shows large-job "
                        f"starvation; prefer reserving for large jobs via "
                        f"{res.candidate.source_policy!r}"),
                expected_goodput_per_dollar_delta=_delta(
                    _kpi(res), _kpi(current_point)),
                expected_gpu_hour_delta=_delta(
                    res.predicted_gpu_hours,
                    current_point.predicted_gpu_hours
                    if current_point else None),
                expected_starvation_delta_pct=_delta(
                    res.predicted_starvation_rate_pct,
                    current_point.predicted_starvation_rate_pct
                    if current_point else None),
                confidence=telemetry_confidence,
                execution_mode=cfg.default_execution_mode,
                notes=tuple(notes) + (
                    f"reserve_fraction={res.candidate.large_job_reservation_fraction}",
                ))

    # 7 — KEEP vs RECOMMEND
    action = TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER
    if (current_point is not None
            and current_point.safety_status == TrainingSafetyStatus.SAFE):
        cur_kpi = _kpi(current_point) or 0.0
        best_kpi = _kpi(best) or 0.0
        kpi_delta_pct = (
            abs(best_kpi - cur_kpi) / cur_kpi if cur_kpi else 0.0)
        cur_dens = (current_point.predicted_packing_density
                    or current_point.predicted_gpu_occupancy or 0.0)
        best_dens = (best.predicted_packing_density
                     or best.predicted_gpu_occupancy or 0.0)
        dens_delta = abs(best_dens - cur_dens)
        if (kpi_delta_pct <= cfg.deadband_kpi_pct
                and dens_delta <= cfg.deadband_packing_density):
            action = TrainingFrontierAction.KEEP_CURRENT_POLICY
            return TrainingFrontierDecision(
                workload_id=workload_id,
                selected_candidate=current_point.candidate,
                current_candidate=current_candidate,
                selected_point=current_point,
                frontier_points=tuple(pts),
                action=action,
                reason=(
                    f"current candidate "
                    f"{current_point.candidate.source_policy!r} within "
                    f"KPI deadband ({kpi_delta_pct:.4f} ≤ "
                    f"{cfg.deadband_kpi_pct}) and packing-density deadband "
                    f"({dens_delta:.4f} ≤ {cfg.deadband_packing_density})"),
                expected_goodput_per_dollar_delta=0.0,
                expected_gpu_hour_delta=0.0,
                expected_queue_wait_delta_s=0.0,
                expected_fragmentation_delta_pct=0.0,
                expected_starvation_delta_pct=0.0,
                confidence=telemetry_confidence,
                execution_mode=cfg.default_execution_mode,
                notes=tuple(notes))

    return TrainingFrontierDecision(
        workload_id=workload_id,
        selected_candidate=best.candidate,
        current_candidate=current_candidate,
        selected_point=best,
        frontier_points=tuple(pts),
        action=action,
        reason=(f"highest SLA-safe goodput/$ at policy "
                f"{best.candidate.source_policy!r} "
                f"(predicted {best.predicted_goodput_per_dollar})"),
        expected_goodput_per_dollar_delta=_delta(
            _kpi(best), _kpi(current_point)),
        expected_gpu_hour_delta=_delta(
            best.predicted_gpu_hours,
            current_point.predicted_gpu_hours if current_point else None),
        expected_queue_wait_delta_s=_delta(
            best.predicted_queue_wait_p99_s,
            current_point.predicted_queue_wait_p99_s
            if current_point else None),
        expected_fragmentation_delta_pct=_delta(
            best.predicted_fragmentation_block_rate_pct,
            current_point.predicted_fragmentation_block_rate_pct
            if current_point else None),
        expected_starvation_delta_pct=_delta(
            best.predicted_starvation_rate_pct,
            current_point.predicted_starvation_rate_pct
            if current_point else None),
        confidence=telemetry_confidence,
        execution_mode=cfg.default_execution_mode,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# Real-execution stub — by default raises so production paths can't fire
# accidentally.
# ---------------------------------------------------------------------------

class TrainingRealExecutionDisabledError(RuntimeError):
    """Raised when a caller asks for real-cluster execution without the
    explicit opt-in + a non-stub executor."""


def execute_training_frontier_decision(
    decision: TrainingFrontierDecision,
    *,
    mode: str = EXECUTION_MODE_SHADOW,
    executor=None,
    allow_real_execution: bool = False,
):
    """Recommendation-only execution shim for training-frontier decisions.

    Behaviour by mode:

    - ``shadow`` / ``real_disabled`` — log the recommendation, mutate
      nothing. Default.
    - ``simulator`` — return a side-effect-free
      :class:`SimulatorEffect`-like dict capturing the recommended
      candidate (the caller's simulator may apply it).
    - ``real_enabled`` — requires BOTH ``allow_real_execution=True``
      AND a non-stub executor. With neither, raises
      :class:`TrainingRealExecutionDisabledError`. The v1 training
      executor is a deliberate stub.
    """
    if mode == EXECUTION_MODE_SHADOW or mode == "real_disabled":
        return {"mode": mode, "mutated": False,
                "recommended_candidate": (
                    decision.selected_candidate.to_dict()
                    if decision.selected_candidate is not None else None)}
    if mode == EXECUTION_MODE_SHADOW:  # pragma: no cover (covered above)
        return None
    if mode == "simulator":
        return {"mode": "simulator", "mutated": True,
                "applied_candidate": (
                    decision.selected_candidate.to_dict()
                    if decision.selected_candidate is not None else None)}
    if mode == "real_enabled":
        if not allow_real_execution:
            raise TrainingRealExecutionDisabledError(
                "real-cluster execution requires allow_real_execution=True")
        if executor is None:
            return {"mode": "real_enabled", "mutated": False,
                    "notes": ("not_implemented_real_executor",)}
        return executor(decision)
    raise ValueError(f"unknown execution mode {mode!r}")
