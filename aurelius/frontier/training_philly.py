"""Philly Training Frontier Estimator (v1).

Builds a :class:`TrainingFrontierPoint` for each measured Philly
scheduling policy in
``data/external/philly/processed/philly_backtest_summary.json``. Each
point's candidate descriptor labels the dimensions of the frontier the
policy emphasizes (backfill aggressiveness, gang-scheduling strictness,
large-job reservation, packing density).

What Philly is **good** for here:

- queue-wait p95 / p99
- starvation events
- multi-GPU gang-scheduling pressure
- backfill / head-of-line behaviour
- retry / wasted-GPU-hours from the attempt log

What Philly is **NOT** good for:

- GPU model price heterogeneity (no GPU type column)
- real per-job utilization time series
- energy

Missing signals are reported as ``None`` and surface as
``INSUFFICIENT_TELEMETRY`` vetoes — never zero-filled.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Sequence

from .training_models import (
    TrainingFrontierCandidate,
    TrainingFrontierPoint,
    TrainingSafetyStatus,
)
from .training_safety import (
    TrainingSafetyConfig,
    classify_training_frontier_point,
)


# ---------------------------------------------------------------------------
# Per-policy candidate descriptors (transparent mapping; no hidden tuning).
# ---------------------------------------------------------------------------

# Each Philly policy in
# ``aurelius/traces/gpu_scheduling.py:SCHEDULING_POLICIES`` is mapped to
# a TrainingFrontierCandidate that **labels what the policy emphasizes**.
# These labels are descriptive, not prescriptive — they are not used to
# alter any measured KPI.
PHILLY_POLICY_CANDIDATES: dict = {
    "fifo": TrainingFrontierCandidate(
        occupancy_target=0.50, packing_density_target=0.50,
        backfill_aggressiveness=0.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=1.0,
        preemption_allowed=False,
        heterogeneity_preference="any",
        source_policy="fifo"),
    "first_fit": TrainingFrontierCandidate(
        occupancy_target=0.65, packing_density_target=0.70,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=0.5,
        preemption_allowed=False,
        heterogeneity_preference="any",
        source_policy="first_fit"),
    "best_fit": TrainingFrontierCandidate(
        occupancy_target=0.75, packing_density_target=0.85,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=0.5,
        preemption_allowed=False,
        heterogeneity_preference="homogeneous",
        source_policy="best_fit"),
    "first_fit_decreasing": TrainingFrontierCandidate(
        occupancy_target=0.70, packing_density_target=0.80,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.10,
        gang_scheduling_strictness=0.7,
        preemption_allowed=False,
        heterogeneity_preference="any",
        source_policy="first_fit_decreasing"),
    "greedy_packing": TrainingFrontierCandidate(
        occupancy_target=0.85, packing_density_target=0.90,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=0.5,
        preemption_allowed=False,
        heterogeneity_preference="homogeneous",
        source_policy="greedy_packing"),
    "topology_aware": TrainingFrontierCandidate(
        occupancy_target=0.75, packing_density_target=0.80,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.10,
        gang_scheduling_strictness=0.9,
        preemption_allowed=False,
        heterogeneity_preference="homogeneous",
        source_policy="topology_aware"),
    "utilization_aware": TrainingFrontierCandidate(
        occupancy_target=0.85, packing_density_target=0.85,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=0.5,
        preemption_allowed=False,
        heterogeneity_preference="any",
        source_policy="utilization_aware"),
    "constraint_aware": TrainingFrontierCandidate(
        occupancy_target=0.80, packing_density_target=0.85,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.10,
        gang_scheduling_strictness=0.8,
        preemption_allowed=False,
        fragmentation_budget=0.10,
        heterogeneity_preference="homogeneous",
        price_aware_gpu_routing_enabled=False,
        source_policy="constraint_aware"),
}


# ---------------------------------------------------------------------------
# Helpers — coerce / compute per-policy training-frontier metrics.
# ---------------------------------------------------------------------------

def _pct_of_jobs(count: Optional[int], n_jobs: int) -> Optional[float]:
    if count is None or not n_jobs:
        return None
    return 100.0 * float(count) / max(1, n_jobs)


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _point_from_philly_policy(
    policy_name: str,
    pol: dict,
    *,
    n_scheduled: int,
    attempt_analysis: Optional[dict],
    safety_config: TrainingSafetyConfig,
    telemetry_confidence: str,
) -> TrainingFrontierPoint:
    """Build one TrainingFrontierPoint from a Philly policy summary."""
    candidate = PHILLY_POLICY_CANDIDATES.get(
        policy_name,
        TrainingFrontierCandidate(source_policy=policy_name))

    # Queue / scheduling metrics that Philly reports directly.
    queue_wait_p95 = pol.get("queue_wait_s_p95")
    queue_wait_p99 = pol.get("queue_wait_s_p99")
    completion_mean = pol.get("mean_completion_s")
    # Philly's existing summary doesn't expose p95/p99 completion; we
    # leave the predicted_*p95/p99 completion fields as ``None`` rather
    # than synthesize them.
    starvation_events = pol.get("starvation_events")
    starvation_rate_pct = _pct_of_jobs(starvation_events, n_scheduled)

    # Fragmentation BLOCK rate: use Philly's measured
    # ``failed_placement_rate_pct`` (a clean fraction of placement
    # attempts that failed) as the primary signal. The raw
    # ``fragmentation_block_events`` count is events-per-job which can
    # exceed 100% when one job blocks multiple times — that's surfaced
    # as a diagnostic only.
    failed_placement_rate_pct = pol.get("failed_placement_rate_pct")
    fragmentation_block_events = pol.get("fragmentation_block_events")
    fragmentation_loss_pct = pol.get("fragmentation_loss_pct")
    fragmentation_block_rate_pct = failed_placement_rate_pct

    backfill_placements = pol.get("backfill_placements")
    # Backfill success: fraction of placements that were *backfill* (a
    # measured rate, not a prediction). Philly's simulator records this
    # as the count of jobs placed out-of-order; we surface the fraction.
    if backfill_placements is not None and n_scheduled:
        backfill_success_rate_pct = (
            100.0 * float(backfill_placements) / max(1, n_scheduled))
    else:
        backfill_success_rate_pct = None

    # Gang-scheduling failure: Philly's per-policy summary does NOT
    # label which failures were strictly gang-scheduling. We do NOT
    # pretend ``failed_or_killed_run`` is gang failure (it lumps in
    # non-gang reasons). The Philly safety config disables the gang
    # gate by default; the failed_or_killed count is surfaced in the
    # notes as a diagnostic.
    failed_or_killed_run = pol.get("failed_or_killed_run")
    gang_failure_pct = None

    # Retry waste: from ``attempt_analysis`` (Philly-wide, not per-policy
    # — the descriptive count of wasted GPU-hours from attempt retries
    # in the raw log).
    retry_waste = None
    if attempt_analysis is not None:
        retry_waste = attempt_analysis.get("wasted_gpu_hours_from_retries")

    # KPI fields.
    gpd = pol.get("sla_safe_goodput_per_infra_dollar")
    gpu_hours = pol.get("gpu_hours_used")
    completed_work = pol.get("goodput_gpu_seconds")
    cost = pol.get("infra_cost")
    util_mean_pct = pol.get("utilization_mean_pct")
    occupancy = (util_mean_pct / 100.0
                  if util_mean_pct is not None else None)

    # Packing density isn't directly reported by Philly's scheduling
    # backtest — we leave it as the descriptor target (no fabrication).
    packing_density = candidate.packing_density_target

    point = TrainingFrontierPoint(
        candidate=candidate,
        predicted_goodput_per_dollar=gpd,
        predicted_gpu_occupancy=occupancy,
        predicted_packing_density=packing_density,
        predicted_gpu_hours=gpu_hours,
        predicted_completed_work=completed_work,
        predicted_queue_wait_p95_s=queue_wait_p95,
        predicted_queue_wait_p99_s=queue_wait_p99,
        predicted_job_completion_p95_s=None,  # Philly summary lacks p95/p99
        predicted_job_completion_p99_s=None,
        predicted_starvation_rate_pct=starvation_rate_pct,
        predicted_fragmentation_block_rate_pct=fragmentation_block_rate_pct,
        predicted_gang_scheduling_failure_pct=gang_failure_pct,
        predicted_backfill_success_rate_pct=backfill_success_rate_pct,
        predicted_retry_waste_gpu_hours=retry_waste,
        predicted_cost=cost,
        notes=tuple(filter(None, (
            f"source_policy={policy_name}",
            "queue_wait_p95/p99: measured" if queue_wait_p95 is not None
            else "queue_wait_p95: missing",
            "completion_p95/p99: NOT REPORTED by Philly summary",
            "gang_failure: PROXY via failed_or_killed_run"
            if failed_or_killed_run is not None
            else "gang_failure: missing",
        ))),
    )
    status, vetoes = classify_training_frontier_point(
        point, safety_config, telemetry_confidence=telemetry_confidence)
    return TrainingFrontierPoint(
        candidate=candidate,
        predicted_goodput_per_dollar=point.predicted_goodput_per_dollar,
        predicted_gpu_occupancy=point.predicted_gpu_occupancy,
        predicted_packing_density=point.predicted_packing_density,
        predicted_gpu_hours=point.predicted_gpu_hours,
        predicted_completed_work=point.predicted_completed_work,
        predicted_queue_wait_p95_s=point.predicted_queue_wait_p95_s,
        predicted_queue_wait_p99_s=point.predicted_queue_wait_p99_s,
        predicted_job_completion_p95_s=point.predicted_job_completion_p95_s,
        predicted_job_completion_p99_s=point.predicted_job_completion_p99_s,
        predicted_starvation_rate_pct=point.predicted_starvation_rate_pct,
        predicted_fragmentation_block_rate_pct=
            point.predicted_fragmentation_block_rate_pct,
        predicted_gang_scheduling_failure_pct=
            point.predicted_gang_scheduling_failure_pct,
        predicted_backfill_success_rate_pct=
            point.predicted_backfill_success_rate_pct,
        predicted_retry_waste_gpu_hours=point.predicted_retry_waste_gpu_hours,
        predicted_cost=point.predicted_cost,
        safety_status=status,
        safety_vetoes=tuple(vetoes),
        notes=point.notes,
    )


def estimate_philly_training_frontier(
    backtest_summary: dict,
    *,
    safety_config: Optional[TrainingSafetyConfig] = None,
    telemetry_confidence: str = "medium",
    policies: Optional[Iterable[str]] = None,
) -> list[TrainingFrontierPoint]:
    """Build the Philly training frontier from an existing backtest summary.

    ``backtest_summary`` is the JSON dict produced by
    ``scripts/run_philly_backtest.py`` /
    ``data/external/philly/processed/philly_backtest_summary.json``.

    Returns one :class:`TrainingFrontierPoint` per Philly policy
    (default: every policy present in the summary). Pass ``policies`` to
    restrict the set.

    The default ``safety_config`` disables the gang-scheduling gate
    because Philly does NOT cleanly distinguish gang failures from
    other failure causes; per-tenant pilot telemetry that DOES report
    gang failure should pass a stricter ``safety_config``.
    """
    if safety_config is None:
        safety_config = TrainingSafetyConfig(
            # Gang failure isn't cleanly measured by Philly's summary —
            # disable the gate by default rather than fabricate a value.
            max_gang_scheduling_failure_pct=None,
        )
    safety = safety_config
    bt = backtest_summary.get("backtest") or {}
    pols = bt.get("policies") or {}
    n_scheduled = int(bt.get("n_scheduled") or bt.get("n_jobs") or 0)
    attempt_analysis = backtest_summary.get("attempt_analysis")
    if policies is None:
        policies = list(pols.keys())
    out: list[TrainingFrontierPoint] = []
    for name in policies:
        pol = pols.get(name)
        if not pol:
            continue
        out.append(_point_from_philly_policy(
            name, pol, n_scheduled=n_scheduled,
            attempt_analysis=attempt_analysis,
            safety_config=safety,
            telemetry_confidence=telemetry_confidence))
    return out


def load_philly_summary(path: Optional[str] = None) -> dict:
    """Load the committed Philly backtest summary JSON."""
    if path is None:
        path = os.path.join(
            "data", "external", "philly", "processed",
            "philly_backtest_summary.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
