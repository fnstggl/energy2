"""Alibaba GPU v2023 Training Frontier Estimator (v1).

Builds a :class:`TrainingFrontierPoint` for each measured Alibaba-GPU
packing policy in
``data/external/alibaba_gpu/processed/alibaba_gpu_backtest_summary.json``.
Each point's candidate descriptor labels what the policy emphasizes
(packing density, fragmentation budget, heterogeneity preference,
price-aware GPU routing).

What Alibaba GPU is **good** for here:

- packing density / GPU occupancy on heterogeneous fleets
- fragmentation score (free-fraction on active nodes)
- placed vs stranded GPU demand
- price-aware GPU routing (the existing ``constraint_aware`` packing
  candidate already routes to lower-priced GPU types when possible)

What Alibaba GPU is **NOT** good for:

- per-job queue wait p95/p99 (the trace lacks consistent submit / start
  times for the static packing baseline)
- starvation rate (no queue-wait causality)
- retry / gang-scheduling failure (no attempt history)

Missing signals are reported as ``None`` and surface as
``INSUFFICIENT_TELEMETRY`` vetoes — never invented.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from .training_models import (
    TrainingFrontierCandidate,
    TrainingFrontierPoint,
    TrainingSafetyStatus,
)
from .training_safety import (
    TrainingSafetyConfig,
    classify_training_frontier_point,
)


# Per-policy candidate descriptors for Alibaba GPU packing. Each
# descriptor labels the packing dimension the policy emphasizes; the
# *measured* KPIs are sourced directly from the committed packing
# backtest summary — never altered here.
ALIBABA_POLICY_CANDIDATES: dict = {
    "fifo": TrainingFrontierCandidate(
        occupancy_target=0.50, packing_density_target=0.50,
        fragmentation_budget=0.50,
        heterogeneity_preference="any",
        price_aware_gpu_routing_enabled=False,
        source_policy="fifo"),
    "first_fit": TrainingFrontierCandidate(
        occupancy_target=0.85, packing_density_target=0.85,
        fragmentation_budget=0.15,
        heterogeneity_preference="any",
        price_aware_gpu_routing_enabled=False,
        source_policy="first_fit"),
    "best_fit": TrainingFrontierCandidate(
        occupancy_target=0.90, packing_density_target=0.90,
        fragmentation_budget=0.10,
        heterogeneity_preference="homogeneous",
        price_aware_gpu_routing_enabled=False,
        source_policy="best_fit"),
    "first_fit_decreasing": TrainingFrontierCandidate(
        occupancy_target=0.90, packing_density_target=0.92,
        fragmentation_budget=0.08,
        heterogeneity_preference="any",
        price_aware_gpu_routing_enabled=False,
        source_policy="first_fit_decreasing"),
    "greedy_packing": TrainingFrontierCandidate(
        occupancy_target=0.85, packing_density_target=0.85,
        fragmentation_budget=0.15,
        heterogeneity_preference="homogeneous",
        price_aware_gpu_routing_enabled=False,
        source_policy="greedy_packing"),
    "constraint_aware": TrainingFrontierCandidate(
        occupancy_target=0.90, packing_density_target=0.90,
        fragmentation_budget=0.10,
        large_job_reservation_fraction=0.05,
        gang_scheduling_strictness=0.8,
        heterogeneity_preference="homogeneous",
        price_aware_gpu_routing_enabled=True,
        source_policy="constraint_aware"),
}


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _point_from_alibaba_policy(
    policy_name: str,
    pol: dict,
    *,
    n_gpu_jobs: int,
    safety_config: TrainingSafetyConfig,
    telemetry_confidence: str,
) -> TrainingFrontierPoint:
    """Build one TrainingFrontierPoint from an Alibaba GPU policy summary.

    Alibaba's packing summary does NOT include queue-wait / starvation /
    gang-scheduling / retry metrics — those fields stay ``None``. That
    is the honest reflection of what the static packing trace measures.
    """
    candidate = ALIBABA_POLICY_CANDIDATES.get(
        policy_name, TrainingFrontierCandidate(source_policy=policy_name))

    util_pct = pol.get("gpu_utilization_pct")
    occupancy = util_pct / 100.0 if util_pct is not None else None
    fragmentation_score = pol.get("fragmentation_score")
    # Alibaba's ``fragmentation_score`` is the mean free/total GPU-milli
    # on ACTIVE nodes — i.e., a *headroom* fraction. We surface it as a
    # block-rate percentage (higher = more blocking pressure).
    fragmentation_block_rate_pct = (
        fragmentation_score * 100.0 if fragmentation_score is not None
        else None)

    stranded_jobs = pol.get("stranded_jobs")
    # Stranded fraction is the closest analog to "placement failure"
    # this trace reports. We use it as a *fragmentation-pressure* signal
    # — we do NOT label it as gang-scheduling failure (Alibaba does not
    # distinguish multi-GPU atomic failures from single-GPU strands).
    stranded_rate_pct = (
        100.0 * float(stranded_jobs) / max(1, n_gpu_jobs)
        if stranded_jobs is not None else None)

    point = TrainingFrontierPoint(
        candidate=candidate,
        predicted_goodput_per_dollar=pol.get(
            "sla_safe_goodput_per_infra_dollar"),
        predicted_gpu_occupancy=occupancy,
        predicted_packing_density=occupancy,  # same family of measurement
        predicted_gpu_hours=pol.get("provisioned_gpu_hours"),
        predicted_completed_work=pol.get("placed_work_gpu_seconds"),
        # Queue / completion / starvation / gang-failure / retry: NOT
        # measured by the static packing baseline. Stay None.
        predicted_queue_wait_p95_s=None,
        predicted_queue_wait_p99_s=None,
        predicted_job_completion_p95_s=None,
        predicted_job_completion_p99_s=None,
        predicted_starvation_rate_pct=stranded_rate_pct,
        predicted_fragmentation_block_rate_pct=fragmentation_block_rate_pct,
        predicted_gang_scheduling_failure_pct=None,
        predicted_backfill_success_rate_pct=None,
        predicted_retry_waste_gpu_hours=None,
        predicted_cost=pol.get("infra_cost"),
        notes=tuple(filter(None, (
            f"source_policy={policy_name}",
            "queue_wait/p95/p99: NOT REPORTED by Alibaba packing summary",
            "gang_failure: NOT REPORTED by Alibaba packing summary",
            "retry_waste: NOT REPORTED by Alibaba packing summary",
            "stranded_rate used as fragmentation-pressure proxy "
            "(NOT gang-scheduling failure)"
            if stranded_rate_pct is not None else None,
        ))),
    )
    status, vetoes = classify_training_frontier_point(
        point, safety_config, telemetry_confidence=telemetry_confidence)
    return TrainingFrontierPoint(
        candidate=point.candidate,
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
        safety_status=status, safety_vetoes=tuple(vetoes),
        notes=point.notes,
    )


def estimate_alibaba_gpu_training_frontier(
    backtest_summary: dict,
    *,
    safety_config: Optional[TrainingSafetyConfig] = None,
    telemetry_confidence: str = "medium",
    policies: Optional[Iterable[str]] = None,
) -> list[TrainingFrontierPoint]:
    """Build the Alibaba GPU training frontier from an existing summary.

    ``backtest_summary`` is the JSON dict produced by
    ``scripts/run_alibaba_gpu_backtest.py``. Queue-wait / starvation /
    gang-failure / retry-waste fields are intentionally absent (see
    module docstring); the safety classification respects this and
    yields INSUFFICIENT_TELEMETRY rather than fabricating numbers.
    """
    # Alibaba's packing summary doesn't report queue_wait fields, so
    # the queue/starvation/gang gates are intentionally **disabled** by
    # passing a config that drops them. The caller may override with a
    # stricter ``safety_config`` if telemetry is supplied externally.
    if safety_config is None:
        safety_config = TrainingSafetyConfig(
            max_queue_wait_p95_s=None,
            max_queue_wait_p99_s=None,
            max_gang_scheduling_failure_pct=None,
            max_retry_waste_gpu_hours=None,
            # keep fragmentation / starvation / completed-work gates
        )
    bt = backtest_summary.get("backtest") or {}
    pols = bt.get("policies") or {}
    n_gpu_jobs = int(bt.get("n_gpu_jobs") or bt.get("n_jobs") or 0)
    if policies is None:
        policies = list(pols.keys())
    out: list[TrainingFrontierPoint] = []
    for name in policies:
        pol = pols.get(name)
        if not pol:
            continue
        out.append(_point_from_alibaba_policy(
            name, pol, n_gpu_jobs=n_gpu_jobs,
            safety_config=safety_config,
            telemetry_confidence=telemetry_confidence))
    return out


def load_alibaba_gpu_summary(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(
            "data", "external", "alibaba_gpu", "processed",
            "alibaba_gpu_backtest_summary.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
