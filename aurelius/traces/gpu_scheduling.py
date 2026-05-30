"""Temporal GPU-cluster scheduling simulator + scheduler-pressure diagnostics.

Philly is primarily a **scheduler-pressure** benchmark, so beyond static packing
(``gpu_packing.py``) this module replays jobs through a deterministic
discrete-event scheduler over a fixed fleet and measures the queueing,
saturation, fragmentation, fairness and backfill behaviour the mission asks for.

Reuses ``gpu_packing._NodeState`` / ``_select_node`` (same placement policies)
plus ``release`` for the time dimension, and ``economics.py`` for the canonical
KPI. Pure / deterministic / stdlib only.

Model (honest):
- Jobs arrive at ``submit_time_s`` and need ``gpu_count`` whole GPUs on a single
  node (Philly is whole-GPU). When placed they hold the GPUs for their trace
  ``duration_s`` then release. ``fifo`` is strict head-of-line (no backfill);
  every other policy backfills (places any queued job that fits); the head-of-
  line job is tracked so genuine *backfill* placements are counted.
- SLA-safe goodput = ``gpu_seconds_work`` of jobs that (a) are not Failed/Killed
  in the trace and (b) start within a queue-wait budget. Failed/Killed jobs that
  run still consume GPU-hours → that is the "wasted GPU-hours" signal.
- Cost bills every node that is *ever* powered, for the makespan, at the
  documented per-GPU-type price (``gpu_packing.gpu_price``). Same fleet, prices
  and jobs across policies — only the scheduling decision differs.
- Retries/failures are **trace-observed** (attempt history), not re-simulated;
  we report them and whether constraint_aware reduces the queueing/fragmentation
  that drives real preemption-retries — directional, not a re-simulation.

NOT a production claim; no constants tuned to favour a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from aurelius.benchmarks.economics import (
    compute_cost_per_sla_compliant_token,
    compute_sla_safe_goodput_per_infra_dollar,
)

from .gpu_packing import (
    HEADLINE_CANDIDATES,
    GPUNode,
    _NodeState,
    _select_node,
    gpu_price,
)
from .schema import NormalizedGPUJob, percentile

# Scheduling policies for Philly (fifo = sanity baseline, NOT the headline).
SCHEDULING_POLICIES = (
    "fifo",
    "first_fit",
    "best_fit",
    "first_fit_decreasing",
    "greedy_packing",
    "topology_aware",
    "utilization_aware",
    "constraint_aware",
)

# Queue-wait SLA budget: a job is SLA-safe if it starts within max(absolute,
# multiple × its own runtime) of submit. Documented prior, identical across
# policies. Generous so only genuinely starved jobs are excluded.
SLA_WAIT_ABS_S = 3600.0           # 1 hour grace
SLA_WAIT_RUNTIME_MULT = 2.0       # or 2× the job's runtime, whichever is larger
STARVATION_WAIT_S = 6 * 3600.0    # waited > 6h => starvation event
QUEUE_COLLAPSE_DEPTH = 50         # ready-queue depth crossing => collapse event


def _size_class(gpu_count: int) -> str:
    if gpu_count <= 1:
        return "1"
    if gpu_count <= 4:
        return "2-4"
    if gpu_count <= 8:
        return "5-8"
    return "9+"


@dataclass
class _RunJob:
    job: NormalizedGPUJob
    node_idx: int
    token: dict
    start: float
    end: float
    backfilled: bool


@dataclass
class SchedResult:
    policy: str
    completed_jobs: int
    failed_or_killed_run: int
    unplaceable_jobs: int
    goodput_gpu_seconds: float
    gpu_hours_used: float
    provisioned_gpu_hours: float
    infra_cost: float
    goodput_per_dollar: Optional[float]
    cost_per_unit_work: Optional[float]
    makespan_s: float
    queue_wait_p50: float
    queue_wait_p95: float
    queue_wait_p99: float
    mean_completion_s: float
    mean_slowdown: float
    utilization_mean_pct: float
    utilization_p95_pct: float
    fragmentation_block_events: int
    fragmentation_loss_pct: float
    failed_placement_rate_pct: float
    backfill_placements: int
    queue_collapse_events: int
    starvation_events: int
    wait_by_size_class: dict = field(default_factory=dict)
    slowdown_by_size_class: dict = field(default_factory=dict)

    def summary(self) -> dict:
        import math as _m
        return {
            "policy": self.policy,
            "goodput_unit": "gpu_seconds_work (completed_gpu_job_work)",
            "sla_safe_goodput_per_infra_dollar": (
                None if self.goodput_per_dollar is None
                else round(self.goodput_per_dollar, 4)),
            "cost_per_unit_work": (
                None if self.cost_per_unit_work is None
                else (_m.inf if _m.isinf(self.cost_per_unit_work)
                      else round(self.cost_per_unit_work, 8))),
            "completed_jobs": self.completed_jobs,
            "failed_or_killed_run": self.failed_or_killed_run,
            "unplaceable_jobs": self.unplaceable_jobs,
            "goodput_gpu_seconds": round(self.goodput_gpu_seconds, 1),
            "gpu_hours_used": round(self.gpu_hours_used, 2),
            "provisioned_gpu_hours": round(self.provisioned_gpu_hours, 2),
            "infra_cost": round(self.infra_cost, 2),
            "makespan_s": round(self.makespan_s, 1),
            "queue_wait_s_p50": round(self.queue_wait_p50, 1),
            "queue_wait_s_p95": round(self.queue_wait_p95, 1),
            "queue_wait_s_p99": round(self.queue_wait_p99, 1),
            "mean_completion_s": round(self.mean_completion_s, 1),
            "mean_slowdown": round(self.mean_slowdown, 3),
            "utilization_mean_pct": round(self.utilization_mean_pct, 2),
            "utilization_p95_pct": round(self.utilization_p95_pct, 2),
            "fragmentation_block_events": self.fragmentation_block_events,
            "fragmentation_loss_pct": round(self.fragmentation_loss_pct, 3),
            "failed_placement_rate_pct": round(self.failed_placement_rate_pct, 3),
            "backfill_placements": self.backfill_placements,
            "queue_collapse_events": self.queue_collapse_events,
            "starvation_events": self.starvation_events,
            "wait_by_size_class": {k: round(v, 1) for k, v in self.wait_by_size_class.items()},
            "slowdown_by_size_class": {k: round(v, 3)
                                       for k, v in self.slowdown_by_size_class.items()},
        }


def _aggregate_free_gpus(states) -> int:
    return sum(sum(1 for f in s.gpu_free if f == 1000) for s in states)


def run_scheduling(
    jobs: Sequence[NormalizedGPUJob],
    nodes: Sequence[GPUNode],
    policy: str,
) -> SchedResult:
    """Deterministic discrete-event scheduling replay of ``jobs`` on ``nodes``."""
    # schedulable jobs: have submit time, >=1 GPU, positive duration
    sched = [j for j in jobs
             if j.submit_time_s is not None and (j.gpu_count or 0) >= 1
             and j.duration_s and j.duration_s > 0]
    sched.sort(key=lambda j: (j.submit_time_s, j.job_id))
    states = [_NodeState(n) for n in nodes]
    total_gpu = sum(n.gpu_count for n in nodes) or 1
    fifo_cursor = [0]

    running: list[_RunJob] = []
    ready: list[NormalizedGPUJob] = []
    si = 0
    n = len(sched)
    per_job: dict = {}          # job_id -> {wait, completion, slowdown, gpu_count, failed}
    frag_block_events = 0
    frag_attempts = 0
    backfill_placements = 0
    queue_collapse_events = 0
    starvation_events = 0
    util_integral = 0.0
    util_samples: list[float] = []
    prev_t = sched[0].submit_time_s if sched else 0.0
    max_end = prev_t
    prev_queue_collapsed = False

    def _advance_util(now: float):
        nonlocal util_integral, prev_t
        dt = now - prev_t
        if dt > 0:
            alloc = total_gpu - _aggregate_free_gpus(states)
            frac = alloc / total_gpu
            util_integral += dt * frac
            util_samples.append(100.0 * frac)
        prev_t = now

    def _place(job, now, head_blocked):
        nonlocal backfill_placements, max_end
        idx = _select_node(states, job, policy, fifo_cursor)
        if idx is None:
            return False
        token = states[idx].place(job)
        end = now + (job.duration_s or 0.0)
        max_end = max(max_end, end)
        running.append(_RunJob(job, idx, token, now, end, head_blocked))
        if head_blocked:
            backfill_placements += 1
        per_job[job.job_id] = {
            "wait": now - job.submit_time_s,
            "completion": end - job.submit_time_s,
            "duration": job.duration_s,
            "gpu_count": job.gpu_count,
            "failed": job.is_failed,
        }
        return True

    def _schedule_pass(now):
        nonlocal frag_block_events, frag_attempts, queue_collapse_events
        nonlocal prev_queue_collapsed, starvation_events
        if not ready:
            return
        # queue-collapse detection (depth crosses threshold, edge-triggered)
        collapsed = len(ready) >= QUEUE_COLLAPSE_DEPTH
        if collapsed and not prev_queue_collapsed:
            queue_collapse_events += 1
        prev_queue_collapsed = collapsed

        if policy == "fifo":
            # strict head-of-line: place from the front while the head fits
            while ready:
                head = ready[0]
                frag_attempts += 1
                if _place(head, now, head_blocked=False):
                    ready.pop(0)
                else:
                    if _aggregate_free_gpus(states) >= head.gpu_count:
                        frag_block_events += 1  # blocked despite aggregate
                    break
            return

        # backfill policies: scan ready in submit order, place any that fit
        head_blocked = False
        remaining = []
        for job in ready:
            frag_attempts += 1
            if _place(job, now, head_blocked=head_blocked and len(remaining) > 0):
                continue
            # could not place this job
            if _aggregate_free_gpus(states) >= job.gpu_count:
                frag_block_events += 1
            if not head_blocked:
                head_blocked = True  # earliest unplaced job => later fits = backfill
            remaining.append(job)
        ready[:] = remaining

    # event loop
    guard = 0
    max_iters = (n + 1) * 4 + 100000
    while (si < n or ready or running) and guard < max_iters:
        guard += 1
        # next event time
        cand = []
        if si < n:
            cand.append(sched[si].submit_time_s)
        if running:
            cand.append(min(r.end for r in running))
        if not cand:
            break
        now = min(cand)
        _advance_util(now)
        # releases first
        still = []
        for r in running:
            if r.end <= now:
                states[r.node_idx].release(r.token)
            else:
                still.append(r)
        running[:] = still
        # admit submissions
        while si < n and sched[si].submit_time_s <= now:
            ready.append(sched[si])
            si += 1
        # schedule
        _schedule_pass(now)
        # stall guard: ready jobs but nothing running and no more submits => unplaceable
        if ready and not running and si >= n:
            break

    # any job never placed = unplaceable (e.g. needs more GPUs than any node has)
    placed_ids = set(per_job)
    unplaceable = [j for j in sched if j.job_id not in placed_ids]

    makespan = max(0.0, max_end - (sched[0].submit_time_s if sched else 0.0))

    waits = [v["wait"] for v in per_job.values()]
    completions = [v["completion"] for v in per_job.values()]
    slowdowns = [(v["wait"] + v["duration"]) / max(1.0, v["duration"])
                 for v in per_job.values()]
    starvation_events = sum(1 for w in waits if w > STARVATION_WAIT_S)

    # goodput: completed, not failed/killed, started within SLA wait budget
    goodput = 0.0
    completed = 0
    failed_run = 0
    gpu_seconds_used = 0.0
    for v in per_job.values():
        gpu_seconds_used += v["gpu_count"] * v["duration"]
        if v["failed"]:
            failed_run += 1
            continue
        budget = max(SLA_WAIT_ABS_S, SLA_WAIT_RUNTIME_MULT * v["duration"])
        if v["wait"] <= budget:
            goodput += v["gpu_count"] * v["duration"]
            completed += 1

    ever = [s for s in states if s.ever_active]
    makespan_h = max(1.0 / 3600.0, makespan / 3600.0)
    provisioned_gpu_hours = sum(s.node.gpu_count for s in ever) * makespan_h
    infra_cost = sum(s.node.gpu_count * gpu_price(s.node.gpu_model) * makespan_h
                     for s in ever)

    goodput_int = int(goodput)
    gpd = compute_sla_safe_goodput_per_infra_dollar(goodput_int, infra_cost)
    cpu = compute_cost_per_sla_compliant_token(infra_cost, goodput_int)

    # by size class
    wait_by: dict = {}
    slow_by: dict = {}
    cnt_by: dict = {}
    for v in per_job.values():
        sc = _size_class(v["gpu_count"])
        wait_by[sc] = wait_by.get(sc, 0.0) + v["wait"]
        slow_by[sc] = slow_by.get(sc, 0.0) + (v["wait"] + v["duration"]) / max(1.0, v["duration"])
        cnt_by[sc] = cnt_by.get(sc, 0) + 1
    wait_by = {k: wait_by[k] / cnt_by[k] for k in wait_by}
    slow_by = {k: slow_by[k] / cnt_by[k] for k in slow_by}

    placed_n = len(per_job)
    return SchedResult(
        policy=policy,
        completed_jobs=completed,
        failed_or_killed_run=failed_run,
        unplaceable_jobs=len(unplaceable),
        goodput_gpu_seconds=goodput,
        gpu_hours_used=gpu_seconds_used / 3600.0,
        provisioned_gpu_hours=provisioned_gpu_hours,
        infra_cost=infra_cost,
        goodput_per_dollar=gpd,
        cost_per_unit_work=cpu,
        makespan_s=makespan,
        queue_wait_p50=percentile(waits, 50) if waits else 0.0,
        queue_wait_p95=percentile(waits, 95) if waits else 0.0,
        queue_wait_p99=percentile(waits, 99) if waits else 0.0,
        mean_completion_s=sum(completions) / len(completions) if completions else 0.0,
        mean_slowdown=sum(slowdowns) / len(slowdowns) if slowdowns else 0.0,
        utilization_mean_pct=100.0 * util_integral / makespan if makespan > 0 else 0.0,
        utilization_p95_pct=percentile(util_samples, 95) if util_samples else 0.0,
        fragmentation_block_events=frag_block_events,
        fragmentation_loss_pct=100.0 * frag_block_events / frag_attempts
        if frag_attempts else 0.0,
        failed_placement_rate_pct=100.0 * len(unplaceable) / placed_n
        if placed_n else 0.0,
        backfill_placements=backfill_placements,
        queue_collapse_events=queue_collapse_events,
        starvation_events=starvation_events,
        wait_by_size_class=dict(sorted(wait_by.items())),
        slowdown_by_size_class=dict(sorted(slow_by.items())),
    )


# ---------------------------------------------------------------------------
# Outcome classification — headline is a real scheduling baseline, NOT fifo
# ---------------------------------------------------------------------------

@dataclass
class SchedOutcome:
    outcome: str
    margin_pct: float
    headline: str
    safety_evidence: list = field(default_factory=list)
    loss_reasons: list = field(default_factory=list)
    notes: str = ""
    beats_fifo: bool = True
    fifo_margin_pct: float = 0.0


def select_headline(results: dict) -> str:
    cands = {k: v for k, v in results.items() if k in HEADLINE_CANDIDATES}
    if not cands:
        return "best_fit"
    return max(cands.items(), key=lambda kv: (kv[1].goodput_per_dollar or 0.0))[0]


def classify(results: dict) -> SchedOutcome:
    ca = results.get("constraint_aware")
    headline_name = select_headline(results)
    headline = results.get(headline_name)
    if ca is None or headline is None:
        return SchedOutcome("TIE", 0.0, headline_name, notes="missing policy")
    ca_g = ca.goodput_per_dollar or 0.0
    base_g = headline.goodput_per_dollar or 0.0
    margin = ((ca_g - base_g) / base_g * 100.0) if base_g > 0 else 0.0

    safety = []
    if headline.queue_wait_p95 > 0 and ca.queue_wait_p95 <= 0.5 * headline.queue_wait_p95:
        safety.append("queue_wait_p95<=0.5x_headline")
    if headline.fragmentation_block_events > 0 and \
            ca.fragmentation_block_events <= 0.5 * headline.fragmentation_block_events:
        safety.append("fragmentation<=0.5x_headline")
    if headline.starvation_events > 0 and ca.starvation_events < headline.starvation_events:
        safety.append("fewer_starvation_events")

    fifo = results.get("fifo")
    fifo_g = (fifo.goodput_per_dollar or 0.0) if fifo else 0.0
    fifo_margin = ((ca_g - fifo_g) / fifo_g * 100.0) if fifo_g > 0 else 0.0

    if margin > 1.0:
        out = SchedOutcome("ALPHA_WIN", margin, headline_name, safety_evidence=safety)
    elif abs(margin) <= 1.0 and safety:
        out = SchedOutcome("SAFETY_WIN", margin, headline_name, safety_evidence=safety)
    elif abs(margin) <= 1.0:
        out = SchedOutcome("TIE", margin, headline_name)
    else:
        out = SchedOutcome("LOSS", margin, headline_name,
                           loss_reasons=["weaker_than_scheduling_baseline"],
                           notes=f"constraint_aware below {headline_name} on goodput/$")
    out.beats_fifo = ca_g >= fifo_g
    out.fifo_margin_pct = fifo_margin
    return out


@dataclass
class PhillyBacktestResult:
    n_jobs: int
    n_scheduled: int
    n_nodes: int
    fleet_gpu_count: int
    policy_results: dict
    outcome: SchedOutcome

    def to_summary_dict(self) -> dict:
        return {
            "primary_kpi": "sla_safe_goodput_per_infrastructure_dollar",
            "goodput_unit": "gpu_seconds_work (completed_gpu_job_work)",
            "headline_baseline": self.outcome.headline,
            "headline_is_scheduling_baseline": self.outcome.headline in HEADLINE_CANDIDATES,
            "n_jobs": self.n_jobs,
            "n_scheduled": self.n_scheduled,
            "n_nodes": self.n_nodes,
            "fleet_gpu_count": self.fleet_gpu_count,
            "policies": {p: r.summary() for p, r in self.policy_results.items()},
            "outcome": {
                "constraint_aware_vs_headline": self.outcome.outcome,
                "margin_pct": round(self.outcome.margin_pct, 4),
                "safety_evidence": self.outcome.safety_evidence,
                "loss_reasons": self.outcome.loss_reasons,
                "notes": self.outcome.notes,
                "beats_fifo_sanity_baseline": self.outcome.beats_fifo,
                "fifo_margin_pct": round(self.outcome.fifo_margin_pct, 4),
            },
        }


def run_backtest(
    jobs: Sequence[NormalizedGPUJob],
    nodes: Sequence[GPUNode],
    *,
    policies: Sequence[str] = SCHEDULING_POLICIES,
) -> PhillyBacktestResult:
    results = {p: run_scheduling(jobs, nodes, p) for p in policies}
    outcome = classify(results)
    n_sched = sum(1 for j in jobs if j.submit_time_s is not None
                  and (j.gpu_count or 0) >= 1 and j.duration_s and j.duration_s > 0)
    return PhillyBacktestResult(
        n_jobs=len(jobs), n_scheduled=n_sched, n_nodes=len(nodes),
        fleet_gpu_count=sum(n.gpu_count for n in nodes),
        policy_results=results, outcome=outcome,
    )
