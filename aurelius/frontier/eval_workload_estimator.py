"""Eval Workload Frontier — estimator.

Eval-class traces (ShareGPT, LMSYS Chatbot Arena conversations) carry
*conversation shape* — turn counts + token-estimate proxies — but typically
do NOT carry arrival timestamps or measured latency. The estimator therefore
operates on a different model than the serving / batch estimators: it takes
the total work (effective_total_tokens summed across the eval suite) and
computes, for each candidate, the predicted goodput/$ at a given concurrency
+ target rho, plus the deadline-miss rate against a SYNTHETIC scenario
deadline.

The estimator is pure / deterministic / stdlib-only. No ML. No real cluster
execution path. No oracle as headline.

Mathematical model (transparent, documented constants):

- per-replica decode tokens/s ~= ``per_replica_decode_tokens_per_s``
  (default 2500.0; mirrors the public-benchmark prior in
  ``aurelius/traces/backtest.py``).
- effective batching efficiency at target rho R ~= ``min(1.0, R + slack)``,
  where ``slack`` is a small efficiency bonus for higher rho. The model is
  deliberately CONSERVATIVE — efficiency saturates and does not blow up.
- predicted GPU-hours = total_tokens / (concurrency * per_replica * R * eff)
  / 3600. Higher concurrency + higher R = fewer GPU-hours, same work.
- predicted goodput/$ = sla_safe_goodput / (gpu_infra_cost + energy_cost).
- predicted eval_suite_completion_hours = total_tokens / (concurrency *
  per_replica * R * eff) / 3600 — the same shape as GPU-hours but expressed
  as wall-clock time.
- predicted deadline_miss_rate_pct = fraction of (per-request projected
  end-to-end time) that exceeds the candidate's deadline-slack budget,
  using a conservative per-request projection: token_count / per_replica /
  R. The model assumes each request is served by one replica at the target
  rho; ``deadline_slack_hours`` is the budget.

Honesty:
- Per-replica throughput is the SAME public-benchmark constant the canonical
  backtest uses — there is no per-class tuning.
- The model has no measured latency to compare against, so the
  deadline-miss rate is a STRUCTURAL prediction, not a calibrated one. The
  user-spec calls this exact pattern out: "If deadlines are synthetic
  assumptions, label them synthetic_scenario, not real trace fields."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..traces.eval_schema import EvalWorkloadRequest
from .eval_workload_models import (
    EvalWorkloadFrontierCandidate,
    EvalWorkloadFrontierPoint,
    EvalWorkloadProfile,
    EvalWorkloadSafetyStatus,
)
from .eval_workload_safety import (
    EvalWorkloadSafetyConfig,
    classify_eval_point_safety,
)


# Public-benchmark priors. Mirrored verbatim from
# ``aurelius/traces/backtest.py`` so the eval frontier and the serving
# replay share a consistent cost basis.
PER_REPLICA_DECODE_TOKENS_PER_S = 2500.0
GPU_TYPE = "NVIDIA A100 SXM4 80GB"
GPU_POWER_KW = 0.40
ELECTRICITY_PRICE_PER_KWH = 0.10
# Aurelius infra cost: see ``aurelius/benchmarks/economics.py``
# ``InfrastructureCostConfig``. The default for an A100 list rate is
# documented there; we use the public-list prior $2.04 / GPU-hour.
GPU_HOURLY_USD = 2.04


@dataclass
class EvalWorkloadEstimatorConfig:
    """Estimator settings. Transparent, no tuned ML constants."""

    per_replica_decode_tokens_per_s: float = PER_REPLICA_DECODE_TOKENS_PER_S
    gpu_hourly_usd: float = GPU_HOURLY_USD
    gpu_power_kw: float = GPU_POWER_KW
    electricity_price_per_kwh: float = ELECTRICITY_PRICE_PER_KWH
    # Conservative efficiency model: efficiency at rho R is min(1, R + slack)
    # so higher rho gives slightly higher batching efficiency, but it
    # SATURATES at 1.0 — never blows up.
    efficiency_slack: float = 0.10


def _eff_at_rho(R: float, slack: float) -> float:
    if R <= 0:
        return 0.0
    return min(1.0, R + slack)


def _total_eval_tokens(requests: Sequence[EvalWorkloadRequest]) -> int:
    """Sum effective_total_tokens across requests; missing -> 0 per record."""
    total = 0
    for r in requests:
        eff = r.effective_total_tokens
        if eff is not None:
            total += int(eff)
    return total


def _per_request_completion_times_s(
    requests: Sequence[EvalWorkloadRequest],
    *,
    per_replica: float,
    R: float,
    slack: float,
    concurrency: int,
) -> list[float]:
    """Project per-request completion times under the candidate.

    Conservative single-replica per-request projection at the target rho —
    NOT a real serving-physics simulation, since eval traces have no real
    arrival times.
    """
    eff = _eff_at_rho(R, slack)
    if eff <= 0 or concurrency <= 0 or per_replica <= 0:
        return []
    # Each request's per-replica completion time at rho R:
    out = []
    for r in requests:
        n = r.effective_total_tokens
        if n is None or n <= 0:
            continue
        out.append(float(n) / (per_replica * R * eff))
    return out


def _evaluate_for_candidate(
    requests: Sequence[EvalWorkloadRequest],
    candidate: EvalWorkloadFrontierCandidate,
    *,
    cfg: EvalWorkloadEstimatorConfig,
    profile: EvalWorkloadProfile,
) -> dict:
    """Compute predicted KPI + deadline-miss for ``candidate``."""
    R = candidate.target_rho or 0.65
    concurrency = candidate.concurrency or 1
    slack = cfg.efficiency_slack
    eff = _eff_at_rho(R, slack)
    per_replica = cfg.per_replica_decode_tokens_per_s

    total_tokens = _total_eval_tokens(requests)
    n_requests = len(requests)
    if total_tokens <= 0 or concurrency <= 0 or per_replica <= 0 or eff <= 0:
        return {
            "predicted_goodput_per_dollar": 0.0,
            "predicted_sla_safe_goodput": 0.0,
            "predicted_deadline_miss_rate_pct": None,
            "predicted_eval_suite_completion_hours": None,
            "predicted_interactive_p99_delta_ms": None,
            "predicted_interactive_timeout_delta_pct": None,
            "predicted_queue_p99_ms": None,
            "predicted_latency_p99_ms": None,
            "predicted_gpu_hours": 0.0,
            "predicted_mean_utilization": float(R),
        }

    # Fleet-wide effective tokens/s = concurrency * per_replica * R * eff
    fleet_tokens_per_s = concurrency * per_replica * R * eff
    completion_seconds = total_tokens / fleet_tokens_per_s
    completion_hours = completion_seconds / 3600.0

    # GPU-hours = concurrency * completion_hours.
    gpu_hours = concurrency * completion_hours
    gpu_cost = gpu_hours * cfg.gpu_hourly_usd
    energy_cost = (gpu_hours * cfg.gpu_power_kw
                   * cfg.electricity_price_per_kwh)
    total_cost = gpu_cost + energy_cost

    # SLA-safe goodput (numerator): tokens that completed BEFORE the
    # deadline. If no deadline configured, all tokens count.
    deadline_seconds = (
        candidate.deadline_slack_hours * 3600.0
        if candidate.deadline_slack_hours is not None else None)
    if deadline_seconds is None:
        sla_safe_tokens = float(total_tokens)
        deadline_miss_pct = None
    else:
        per_req_times = _per_request_completion_times_s(
            requests, per_replica=per_replica, R=R, slack=slack,
            concurrency=concurrency)
        if not per_req_times:
            sla_safe_tokens = 0.0
            deadline_miss_pct = None
        else:
            n_miss = sum(1 for t in per_req_times if t > deadline_seconds)
            miss_rate = 100.0 * n_miss / len(per_req_times)
            deadline_miss_pct = miss_rate
            sla_safe_fraction = max(0.0, 1.0 - miss_rate / 100.0)
            sla_safe_tokens = total_tokens * sla_safe_fraction

    goodput_per_dollar = (sla_safe_tokens / total_cost
                          if total_cost > 0 else 0.0)

    # Interactive-baseline deltas: in dedicated_fleet=True case both deltas
    # are zero (eval workload runs on its own fleet, no interference). In
    # mixed-fleet mode we model the worst-case impact as proportional to
    # concurrency / fleet-headroom — but we LEAVE THIS as None unless the
    # caller provided a baseline rho, so we never invent a number.
    is_dedicated = (candidate.dedicated_fleet
                    if candidate.dedicated_fleet is not None
                    else True)
    if is_dedicated:
        # Eval runs on its own fleet — by construction zero interference.
        interactive_p99_delta_ms = 0.0
        interactive_timeout_delta_pct = 0.0
    else:
        # Shared fleet: structural model. The eval workload steals fleet
        # tokens/s proportional to (R - baseline_rho). When baseline data is
        # missing we report None — the safety gate then treats this as
        # INSUFFICIENT_TELEMETRY.
        if profile.interactive_baseline_p99_ms is None:
            interactive_p99_delta_ms = None
        else:
            # Conservative proportional model: extra contention pushes the
            # baseline p99 up by R fraction.
            interactive_p99_delta_ms = (
                profile.interactive_baseline_p99_ms * max(0.0, R - 0.5)
            )
        if profile.interactive_baseline_timeout_pct is None:
            interactive_timeout_delta_pct = None
        else:
            # Similar conservative model: 10x amplification on timeout.
            interactive_timeout_delta_pct = (
                profile.interactive_baseline_timeout_pct * max(0.0, R - 0.5)
                * 10.0
            )

    return {
        "predicted_goodput_per_dollar": float(goodput_per_dollar),
        "predicted_sla_safe_goodput": float(sla_safe_tokens),
        "predicted_deadline_miss_rate_pct": deadline_miss_pct,
        "predicted_eval_suite_completion_hours": float(completion_hours),
        "predicted_interactive_p99_delta_ms": interactive_p99_delta_ms,
        "predicted_interactive_timeout_delta_pct": (
            interactive_timeout_delta_pct),
        # Eval-shape traces have no measured queue/latency — we leave
        # these None rather than invent values.
        "predicted_queue_p99_ms": None,
        "predicted_latency_p99_ms": None,
        "predicted_gpu_hours": float(gpu_hours),
        "predicted_mean_utilization": float(R),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_eval_workload_frontier(
    profile: EvalWorkloadProfile,
    eval_requests: Sequence[EvalWorkloadRequest],
    candidates: Iterable[EvalWorkloadFrontierCandidate],
    *,
    estimator_config: Optional[EvalWorkloadEstimatorConfig] = None,
    safety_config: Optional[EvalWorkloadSafetyConfig] = None,
) -> list[EvalWorkloadFrontierPoint]:
    """Estimate the eval frontier for ``profile`` over ``eval_requests``.

    Empty ``eval_requests`` → every point returns INSUFFICIENT_TELEMETRY.
    """
    cfg = estimator_config or EvalWorkloadEstimatorConfig()
    safety = safety_config or EvalWorkloadSafetyConfig()
    cand_list = list(candidates)

    if not eval_requests:
        return [
            EvalWorkloadFrontierPoint(
                candidate=c,
                safety_status=EvalWorkloadSafetyStatus.INSUFFICIENT_TELEMETRY,
                safety_vetoes=("empty_eval_request_set",),
                notes=("estimator received empty eval request set",),
            )
            for c in cand_list
        ]

    points: list[EvalWorkloadFrontierPoint] = []
    for c in cand_list:
        metrics = _evaluate_for_candidate(
            eval_requests, c, cfg=cfg, profile=profile)
        provisional = EvalWorkloadFrontierPoint(
            candidate=c, safety_status=EvalWorkloadSafetyStatus.SAFE,
            **metrics)
        status, vetoes = classify_eval_point_safety(
            provisional, safety, profile=profile,
            telemetry_confidence=profile.telemetry_confidence)
        points.append(EvalWorkloadFrontierPoint(
            candidate=c, safety_status=status, safety_vetoes=vetoes,
            notes=("structural_eval_model_v1",), **metrics))
    return points
