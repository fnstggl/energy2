"""Batch Inference Frontier — estimator.

Reuses the **unchanged** serving physics in ``aurelius/traces/backtest.py``
to evaluate every (rho, deadline_slack) candidate point on a sequence of
arrival ticks (Azure LLM 2024, BurstGPT, or any future trace that yields
``ArrivalTick`` objects).

The estimator NEVER reads a real deadline from the source serving trace —
the deadline is a **synthetic scenario knob** the caller passes in via the
profile (``synthetic_scenario_label``) and the candidate
(``deadline_slack_seconds``). The deadline-miss-rate is computed per-tick
as the fraction of the tick's predicted p99 latency that exceeds the
candidate's deadline budget.

Pure / deterministic / stdlib-only. No ML training. No optimizer constant
is tuned. Real cluster execution is disabled by default at the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .batch_inference_models import (
    BatchInferenceFrontierCandidate,
    BatchInferenceFrontierPoint,
    BatchInferenceFrontierSchemaError,
    BatchInferenceSafetyStatus,
    BatchInferenceWorkloadProfile,
)
from .batch_inference_safety import (
    BatchInferenceSafetyConfig,
    classify_batch_point_safety,
)


ANTICIPATORY = "anticipatory"
REACTIVE = "reactive"
_MODES = (ANTICIPATORY, REACTIVE)


@dataclass
class BatchInferenceEstimatorConfig:
    """Estimator settings.

    ``mode`` selects reactive vs anticipatory replay (mirrors the serving
    frontier estimator). ``tick_seconds`` matches the canonical replay
    cadence. ``prefill_savings`` mirrors the canonical backtest default.
    ``rho_baseline`` is the operator's current rho (used by the controller
    to compute deltas).
    """

    mode: str = ANTICIPATORY
    tick_seconds: float = 60.0
    prefill_savings: float = 0.0
    rho_baseline: float = 0.65

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(
                f"unknown estimator mode {self.mode!r}; "
                f"expected one of {_MODES}")


def _bt():  # pragma: no cover - import indirection (mirrors estimator.py)
    from aurelius.traces import backtest as bt
    return bt


def _evaluate_for_candidate(
    ticks,
    candidate: BatchInferenceFrontierCandidate,
    *,
    mode: str,
    tick_seconds: float,
    prefill_savings: float,
) -> dict:
    """Replay ``ticks`` at ``candidate.target_rho`` and compute KPI + the
    candidate's predicted deadline-miss rate against
    ``candidate.deadline_slack_seconds``.

    Mirrors ``aurelius/frontier/estimator.py::_evaluate_for_rho`` but adds:
    - the deadline-slack synthetic-scenario knob;
    - a per-tick deadline-miss computation that compares the tick's
      predicted p99 latency to the deadline budget (sla + slack).
    """
    bt = _bt()
    tick_hours = tick_seconds / 3600.0
    rho = candidate.target_rho if candidate.target_rho is not None else 0.65
    deadline_slack_ms = (1000.0 * candidate.deadline_slack_seconds
                         if candidate.deadline_slack_seconds is not None
                         else None)

    if mode == ANTICIPATORY:
        sizer_cls = _AnticipatorySizer
        sizer = sizer_cls(rho, tick_hours=tick_hours)
    else:
        sizer = _ReactiveSizer(rho)

    evals = []
    prev_r = None
    deadline_misses = 0  # ticks where p99 exceeded deadline budget
    miss_weighted_requests = 0
    weighted_requests = 0
    for t in ticks:
        r = sizer.size(t)
        ev = bt.evaluate_tick(t, r, prefill_savings=prefill_savings,
                              tick_hours=tick_hours)
        if prev_r is not None and ev.replicas != prev_r:
            ev.scale_event = True
        prev_r = ev.replicas
        evals.append(ev)
        # Deadline-miss: the predicted p99 latency must stay <= sla_ms + slack
        if deadline_slack_ms is not None and t.request_count > 0:
            budget = ev.sla_ms + deadline_slack_ms
            if ev.latency_p99_ms > budget:
                deadline_misses += 1
                miss_weighted_requests += t.request_count
            weighted_requests += t.request_count

    res = bt._aggregate(f"{mode}@rho={rho}_slack={candidate.deadline_slack_seconds}",
                        evals, cache_aware=False, ticks=ticks)
    active = [(e, t) for e, t in zip(evals, ticks) if t.request_count > 0]
    aw = sum(t.request_count for _, t in active) or 1
    mean_rho = sum(e.rho * t.request_count for e, t in active) / aw

    deadline_miss_pct = (
        100.0 * miss_weighted_requests / weighted_requests
        if weighted_requests > 0 else None
    )

    return {
        "predicted_goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "predicted_sla_safe_goodput": float(res.kpi.sla_compliant_goodput),
        "predicted_deadline_miss_rate_pct": deadline_miss_pct,
        "predicted_timeout_rate_pct": float(res.timeout_rate_pct_mean),
        "predicted_queue_p95_ms": float(res.queue_p95_ms),
        "predicted_queue_p99_ms": float(res.queue_p99_ms),
        "predicted_latency_p95_ms": float(res.latency_p95_ms),
        "predicted_latency_p99_ms": float(res.latency_p99_ms),
        "predicted_gpu_hours": float(res.kpi.active_gpu_hours),
        "predicted_mean_utilization": float(mean_rho),
        "predicted_cost_per_sla_compliant_token": float(
            res.kpi.cost_per_sla_compliant_token or 0.0) or None,
    }


# Lightweight sizers — local copies of the patterns in ``estimator.py`` so
# this module does NOT import the serving rho controller.

class _ReactiveSizer:

    def __init__(self, R: float):
        self.R = R
        self.prev = None

    def size(self, t):
        bt = _bt()
        src = self.prev if self.prev is not None else t
        r = bt._size_for_target(
            src.arrival_rate_rps, max(1.0, src.output_tokens_mean),
            bt._tick_throughput_tokps(src), self.R)
        self.prev = t
        return r


class _AnticipatorySizer:

    def __init__(self, R: float, *, tick_hours: float):
        self.R = R
        self.tick_hours = tick_hours
        self.ewma_r = 0.0
        self.ewma_o = 0.0
        self.prev_replicas = None

    def size(self, t):
        bt = _bt()
        a = 0.5
        if t.request_count > 0:
            self.ewma_r = (a * t.arrival_rate_rps + (1 - a) * self.ewma_r
                           if self.ewma_r else t.arrival_rate_rps)
            self.ewma_o = (a * t.output_tokens_mean + (1 - a) * self.ewma_o
                           if self.ewma_o else t.output_tokens_mean)
        plan_rate = max(t.arrival_rate_rps, self.ewma_r)
        plan_out = (max(t.output_tokens_mean, self.ewma_o) if t.request_count
                    else self.ewma_o)
        base = bt._size_for_target(
            plan_rate, max(1.0, plan_out),
            bt._tick_throughput_tokps(t), self.R)
        r = bt._constraint_trim(t, base, 0.0, self.tick_hours,
                                self.prev_replicas)
        self.prev_replicas = r
        return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_batch_inference_frontier(
    profile: BatchInferenceWorkloadProfile,
    telemetry_window,
    candidates: Iterable[BatchInferenceFrontierCandidate],
    *,
    estimator_config: Optional[BatchInferenceEstimatorConfig] = None,
    safety_config: Optional[BatchInferenceSafetyConfig] = None,
) -> list[BatchInferenceFrontierPoint]:
    """Estimate the batch frontier for ``profile`` over ``telemetry_window``.

    ``telemetry_window`` is a sequence of aggregated ``ArrivalTick`` objects
    (as used by ``aurelius/traces/backtest.py``). Empty window → all points
    return INSUFFICIENT_TELEMETRY; the estimator does not invent data.

    Each candidate's ``target_rho`` and ``deadline_slack_seconds`` drive the
    replay; the safety classifier produces the categorical verdict.
    """
    cfg = estimator_config or BatchInferenceEstimatorConfig()
    safety = safety_config or BatchInferenceSafetyConfig()
    ticks = list(telemetry_window) if telemetry_window is not None else []
    cand_list = list(candidates)

    if not ticks:
        return [
            BatchInferenceFrontierPoint(
                candidate=c,
                safety_status=BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY,
                safety_vetoes=("empty_telemetry_window",),
                notes=("estimator received empty telemetry window",),
            )
            for c in cand_list
        ]

    points: list[BatchInferenceFrontierPoint] = []
    for c in cand_list:
        if c.target_rho is None:
            # Cannot replay without a rho; record as insufficient telemetry.
            points.append(BatchInferenceFrontierPoint(
                candidate=c,
                safety_status=BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY,
                safety_vetoes=("candidate_target_rho_missing",),
                notes=("candidate target_rho is None",),
            ))
            continue
        metrics = _evaluate_for_candidate(
            ticks, c, mode=cfg.mode, tick_seconds=cfg.tick_seconds,
            prefill_savings=cfg.prefill_savings)
        provisional = BatchInferenceFrontierPoint(
            candidate=c, safety_status=BatchInferenceSafetyStatus.SAFE,
            **metrics)
        status, vetoes = classify_batch_point_safety(
            provisional, safety, profile=profile,
            telemetry_confidence=profile.telemetry_confidence)
        points.append(BatchInferenceFrontierPoint(
            candidate=c, safety_status=status, safety_vetoes=vetoes,
            notes=(cfg.mode,), **metrics))
    return points
