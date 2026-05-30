"""GPU bin-packing / fragmentation backtest engine.

Executable packing baselines (closing the prior analysis-only gap) for GPU
cluster traces such as Alibaba ``cluster-trace-gpu-v2023``. Jobs (with whole-GPU
or fractional ``gpu_milli`` requests, plus cpu/mem) are packed onto a fixed,
**heterogeneous** node fleet; the canonical KPI from ``docs/RESULTS.md`` §1 —
SLA-safe goodput per infrastructure dollar — is scored via
``aurelius/benchmarks/economics.py``.

Framing (honest): this is a **static fractional bin-packing** benchmark in the
spirit of Alibaba's openb scheduling benchmark — jobs are placed onto the fleet
in policy order; a job that does not fit is **stranded** (fragmentation), which
is the SLA-violation analogue here. Goodput is ``completed_gpu_job_work``
(``token_equivalent`` = effective_GPU × duration), explicitly NOT inference
output tokens. Infra cost bills every **active** node (one with ≥1 placed job)
for the full trace window at a documented per-GPU-type price — so spreading /
fragmentation (more active, under-filled nodes) costs more per unit work, and
consolidation is rewarded. No temporal migration is modelled (churn = 0).

Nothing here is a production claim, no constants are tuned to favour a policy:
the packing physics, fleet, prices and trace window are identical across every
policy — only the placement decision differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from aurelius.benchmarks.economics import (
    compute_cost_per_sla_compliant_token,
    compute_sla_safe_goodput_per_infra_dollar,
)

from .schema import NormalizedGPUJob

# Documented per-GPU-hour price priors by Alibaba model label ($/GPU-hr). Public
# ballpark priors (±50%), identical across policies. Alibaba anonymizes some
# models as G1/G2/G3; priced as mid/high-tier. Override before any external use.
GPU_PRICE_PER_HR: dict[str, float] = {
    "V100M32": 2.5, "V100M16": 2.2, "P100": 1.2, "T4": 0.55, "A10": 1.0,
    "G1": 1.0, "G2": 1.5, "G3": 2.0,
}
FALLBACK_GPU_PRICE_PER_HR: float = 1.5
GPU_MILLI_PER_GPU: int = 1000


def gpu_price(model: Optional[str]) -> float:
    return GPU_PRICE_PER_HR.get(model or "", FALLBACK_GPU_PRICE_PER_HR)


@dataclass(frozen=True)
class GPUNode:
    """A fixed-fleet node (capacity to pack onto)."""

    node_id: str
    gpu_count: int
    gpu_model: str
    cpu_milli: int
    memory_mib: int


# ---------------------------------------------------------------------------
# Job demand helpers
# ---------------------------------------------------------------------------

def effective_gpu(job: NormalizedGPUJob) -> float:
    """Effective whole-GPU equivalents requested (fractional for sharing)."""
    if job.gpu_count and job.gpu_count >= 1:
        # multi/whole-GPU: gpu_milli is per-GPU (≈1000); treat as whole GPUs
        if job.gpu_count == 1 and job.gpu_milli and job.gpu_milli < GPU_MILLI_PER_GPU:
            return job.gpu_milli / GPU_MILLI_PER_GPU
        return float(job.gpu_count)
    if job.gpu_milli and job.gpu_milli > 0:
        return job.gpu_milli / GPU_MILLI_PER_GPU
    return 0.0


def _is_fractional(job: NormalizedGPUJob) -> bool:
    return (job.gpu_count or 0) <= 1 and bool(job.gpu_milli) and job.gpu_milli < GPU_MILLI_PER_GPU


def job_work(job: NormalizedGPUJob) -> float:
    """``completed_gpu_job_work`` (token_equivalent): effective_GPU × duration_s."""
    dur = job.duration_s if (job.duration_s and job.duration_s > 0) else 0.0
    return effective_gpu(job) * dur


# ---------------------------------------------------------------------------
# Mutable node state during a pack
# ---------------------------------------------------------------------------

class _NodeState:
    __slots__ = ("node", "gpu_free", "cpu_free", "mem_free", "placed", "ever_active")

    def __init__(self, node: GPUNode):
        self.node = node
        self.gpu_free = [GPU_MILLI_PER_GPU] * node.gpu_count  # milli free per GPU
        self.cpu_free = node.cpu_milli
        self.mem_free = node.memory_mib
        self.placed = 0
        self.ever_active = False  # powered at least once (for temporal cost)

    @property
    def active(self) -> bool:
        return self.placed > 0

    @property
    def total_gpu_milli(self) -> int:
        return self.node.gpu_count * GPU_MILLI_PER_GPU

    @property
    def free_gpu_milli(self) -> int:
        return sum(self.gpu_free)

    def can_place(self, job: NormalizedGPUJob) -> bool:
        cpu = job.cpu_milli or 0
        mem = job.memory_mib or 0
        if cpu > self.cpu_free or mem > self.mem_free:
            return False
        gc = job.gpu_count or 0
        if gc == 0 and not (job.gpu_milli and job.gpu_milli > 0):
            return True  # cpu-only fits if cpu/mem ok
        if _is_fractional(job):
            need = job.gpu_milli
            return any(free >= need for free in self.gpu_free)
        # whole GPUs: need `gc` fully-free GPUs
        return sum(1 for free in self.gpu_free if free == GPU_MILLI_PER_GPU) >= max(1, gc)

    def place(self, job: NormalizedGPUJob) -> dict:
        """Allocate resources for ``job``; return an allocation token that
        ``release`` can later return to the pool (used by the temporal
        scheduler). The static packer simply ignores the return value."""
        cpu = job.cpu_milli or 0
        mem = job.memory_mib or 0
        self.cpu_free -= cpu
        self.mem_free -= mem
        gc = job.gpu_count or 0
        gpu_alloc: list[tuple[int, int]] = []
        if _is_fractional(job):
            need = job.gpu_milli
            idx = min((i for i, f in enumerate(self.gpu_free) if f >= need),
                      key=lambda i: self.gpu_free[i], default=None)
            if idx is not None:
                self.gpu_free[idx] -= need
                gpu_alloc.append((idx, need))
        elif gc >= 1:
            taken = 0
            for i, f in enumerate(self.gpu_free):
                if f == GPU_MILLI_PER_GPU:
                    self.gpu_free[i] = 0
                    gpu_alloc.append((i, GPU_MILLI_PER_GPU))
                    taken += 1
                    if taken >= gc:
                        break
        self.placed += 1
        self.ever_active = True
        return {"cpu": cpu, "mem": mem, "gpu": gpu_alloc}

    def release(self, token: dict) -> None:
        self.cpu_free += token.get("cpu", 0)
        self.mem_free += token.get("mem", 0)
        for idx, milli in token.get("gpu", []):
            self.gpu_free[idx] += milli
        self.placed = max(0, self.placed - 1)


# ---------------------------------------------------------------------------
# Placement policies
# ---------------------------------------------------------------------------

PACKING_POLICIES = (
    "fifo",
    "first_fit",
    "best_fit",
    "first_fit_decreasing",
    "greedy_packing",
    "constraint_aware",
)
# Headline candidates are the real packing/scheduling baselines — NOT fifo
# (docs/RESULTS.md §3; mission requirement). topology_aware / utilization_aware
# are opt-in (not in the default PACKING_POLICIES) but, when run (e.g. Philly),
# are eligible as the headline. select_headline only considers candidates that
# were actually run, so the Alibaba default set is unaffected.
HEADLINE_CANDIDATES = ("best_fit", "first_fit_decreasing", "greedy_packing",
                       "topology_aware", "utilization_aware")


def _job_order(jobs: Sequence[NormalizedGPUJob], policy: str) -> list[NormalizedGPUJob]:
    if policy in ("first_fit_decreasing", "greedy_packing"):
        # decreasing GPU demand (big jobs first — classic FFD)
        return sorted(jobs, key=lambda j: (-effective_gpu(j), j.submit_time_s or 0.0,
                                           j.job_id))
    # arrival order (fifo / first_fit / best_fit / constraint_aware). CA stays in
    # arrival order so it never strands more work than best_fit — its edge is
    # heterogeneous price-aware node SELECTION, not a different job order.
    return sorted(jobs, key=lambda j: (j.submit_time_s or 0.0, j.job_id))


def _select_node(states: list[_NodeState], job: NormalizedGPUJob, policy: str,
                 fifo_cursor: list[int]) -> Optional[int]:
    fitting = [i for i, s in enumerate(states) if s.can_place(job)]
    if not fitting:
        return None
    if policy == "fifo":
        # naive spread: round-robin among fitting nodes (no consolidation)
        start = fifo_cursor[0]
        ordered = [i for i in fitting if i >= start] + [i for i in fitting if i < start]
        chosen = ordered[0]
        fifo_cursor[0] = (chosen + 1) % len(states)
        return chosen
    if policy == "first_fit":
        return fitting[0]  # lowest index that fits
    if policy in ("best_fit", "first_fit_decreasing", "greedy_packing"):
        if policy == "first_fit_decreasing":
            return fitting[0]
        # best_fit / greedy_packing: tightest remaining GPU capacity that fits,
        # preferring already-active nodes (consolidation).
        return min(fitting, key=lambda i: (
            0 if states[i].active else 1,
            states[i].free_gpu_milli,
            i,
        ))
    if policy == "constraint_aware":
        # Aurelius: consolidate (prefer active) + cheapest adequate GPU type
        # (heterogeneous-aware) + tightest fit; and reserve fully-free GPUs on
        # big-GPU nodes for whole/multi-GPU jobs (don't fragment them with tiny
        # fractional shares when a smaller/partial node fits).
        frac = _is_fractional(job)
        def score(i: int):
            s = states[i]
            big_node = s.node.gpu_count >= 4
            # penalty for opening a large node's fresh GPUs for a fractional job
            reserve_penalty = 1 if (frac and big_node and not s.active) else 0
            return (
                0 if s.active else 1,        # consolidate onto active nodes
                reserve_penalty,             # keep big nodes free for big jobs
                gpu_price(s.node.gpu_model), # cheapest adequate GPU type
                s.free_gpu_milli,            # tightest fit
                i,
            )
        return min(fitting, key=score)
    if policy == "topology_aware":
        # Right-size the node to the job to preserve large contiguous GPU blocks
        # (locality) for multi-GPU training jobs: a small job goes to the
        # smallest node that fits (don't fragment an 8-GPU node for a 1-GPU job);
        # consolidate onto active nodes first.
        need = max(1, job.gpu_count or 1)
        return min(fitting, key=lambda i: (
            0 if states[i].active else 1,
            states[i].node.gpu_count < need,       # must hold the whole job
            states[i].node.gpu_count,              # smallest adequate node
            states[i].free_gpu_milli,
            i,
        ))
    if policy == "utilization_aware":
        # Pack onto the most-utilised node that still fits (maximise per-node
        # utilisation before powering a fresh node).
        return min(fitting, key=lambda i: (
            0 if states[i].active else 1,
            states[i].free_gpu_milli / states[i].total_gpu_milli,  # least free frac
            i,
        ))
    raise ValueError(f"unknown policy {policy}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Result + simulator
# ---------------------------------------------------------------------------

@dataclass
class PackResult:
    policy: str
    placed_jobs: int
    stranded_jobs: int
    placed_work: float          # gpu-seconds (token_equivalent goodput)
    stranded_gpu: float         # effective-GPU demand that could not be placed
    active_nodes: int
    active_gpu_count: int
    provisioned_gpu_hours: float
    infra_cost: float
    goodput_per_dollar: Optional[float]
    cost_per_unit_work: Optional[float]
    gpu_utilization_pct: float
    fragmentation_score: float  # mean free/total GPU-milli on ACTIVE nodes (lower=better)
    allocated_gpu_hours: float

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "placed_jobs": self.placed_jobs,
            "stranded_jobs": self.stranded_jobs,
            "placed_work_gpu_seconds": round(self.placed_work, 2),
            "stranded_gpu_demand": round(self.stranded_gpu, 4),
            "active_nodes": self.active_nodes,
            "active_gpu_count": self.active_gpu_count,
            "provisioned_gpu_hours": round(self.provisioned_gpu_hours, 2),
            "infra_cost": round(self.infra_cost, 2),
            "goodput_unit": "completed_gpu_job_work (token_equivalent)",
            "sla_safe_goodput_per_infra_dollar": (
                None if self.goodput_per_dollar is None
                else round(self.goodput_per_dollar, 4)),
            "cost_per_unit_work": (
                None if self.cost_per_unit_work is None
                else (math.inf if math.isinf(self.cost_per_unit_work)
                      else round(self.cost_per_unit_work, 8))),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 3),
            "fragmentation_score": round(self.fragmentation_score, 4),
            "allocated_gpu_hours": round(self.allocated_gpu_hours, 2),
        }


def _trace_window_hours(jobs: Sequence[NormalizedGPUJob]) -> float:
    subs = [j.submit_time_s for j in jobs if j.submit_time_s is not None]
    ends = [j.end_time_s for j in jobs if j.end_time_s is not None]
    if not subs or not ends:
        # fall back to max duration
        durs = [j.duration_s or 0.0 for j in jobs]
        return max(1.0, max(durs) / 3600.0) if durs else 1.0
    window_s = max(ends) - min(subs)
    return max(1.0 / 3600.0, window_s / 3600.0)


def run_packing(
    jobs: Sequence[NormalizedGPUJob],
    nodes: Sequence[GPUNode],
    policy: str,
) -> PackResult:
    """Statically pack ``jobs`` onto ``nodes`` under ``policy``; score the KPI."""
    gpu_jobs = [j for j in jobs if effective_gpu(j) > 0]
    states = [_NodeState(n) for n in nodes]
    window_h = _trace_window_hours(jobs)

    placed = 0
    placed_work = 0.0
    stranded = 0
    stranded_gpu = 0.0
    allocated_gpu_seconds = 0.0
    fifo_cursor = [0]

    for job in _job_order(gpu_jobs, policy):
        idx = _select_node(states, job, policy, fifo_cursor)
        if idx is None:
            stranded += 1
            stranded_gpu += effective_gpu(job)
            continue
        states[idx].place(job)
        placed += 1
        placed_work += job_work(job)
        allocated_gpu_seconds += effective_gpu(job) * (job.duration_s or 0.0)

    active = [s for s in states if s.active]
    active_nodes = len(active)
    active_gpu = sum(s.node.gpu_count for s in active)
    provisioned_gpu_hours = active_gpu * window_h
    infra_cost = sum(s.node.gpu_count * gpu_price(s.node.gpu_model) * window_h
                     for s in active)

    # utilisation: allocated GPU-milli / active GPU-milli capacity
    active_capacity_milli = sum(s.total_gpu_milli for s in active) or 1
    used_milli = sum(s.total_gpu_milli - s.free_gpu_milli for s in active)
    gpu_util = 100.0 * used_milli / active_capacity_milli
    # fragmentation: mean free/total on active nodes (powered-but-idle capacity)
    frag = (sum(s.free_gpu_milli / s.total_gpu_milli for s in active) / active_nodes
            if active_nodes else 0.0)

    goodput = int(placed_work)
    gpd = compute_sla_safe_goodput_per_infra_dollar(goodput, infra_cost)
    cpu = compute_cost_per_sla_compliant_token(infra_cost, goodput)

    return PackResult(
        policy=policy, placed_jobs=placed, stranded_jobs=stranded,
        placed_work=placed_work, stranded_gpu=stranded_gpu,
        active_nodes=active_nodes, active_gpu_count=active_gpu,
        provisioned_gpu_hours=provisioned_gpu_hours, infra_cost=infra_cost,
        goodput_per_dollar=gpd, cost_per_unit_work=cpu,
        gpu_utilization_pct=gpu_util, fragmentation_score=frag,
        allocated_gpu_hours=allocated_gpu_seconds / 3600.0,
    )


# ---------------------------------------------------------------------------
# Outcome classification (docs/RESULTS.md §6) — headline is a PACKING baseline
# ---------------------------------------------------------------------------

@dataclass
class PackingOutcome:
    outcome: str
    margin_pct: float
    headline: str
    safety_evidence: list = field(default_factory=list)
    loss_reasons: list = field(default_factory=list)
    notes: str = ""
    beats_fifo: bool = True
    fifo_margin_pct: float = 0.0


def select_headline(results: dict) -> str:
    """Strongest packing baseline by goodput/$ (best_fit / FFD / greedy)."""
    cands = {k: v for k, v in results.items() if k in HEADLINE_CANDIDATES}
    if not cands:
        return "best_fit"
    return max(cands.items(),
              key=lambda kv: (kv[1].goodput_per_dollar or 0.0))[0]


def classify(results: dict) -> PackingOutcome:
    ca = results.get("constraint_aware")
    headline_name = select_headline(results)
    headline = results.get(headline_name)
    if ca is None or headline is None:
        return PackingOutcome("TIE", 0.0, headline_name, notes="missing policy")

    ca_g = ca.goodput_per_dollar or 0.0
    base_g = headline.goodput_per_dollar or 0.0
    margin = ((ca_g - base_g) / base_g * 100.0) if base_g > 0 else 0.0

    # safety evidence: fewer stranded jobs / lower fragmentation than headline
    safety = []
    if headline.stranded_jobs > 0 and ca.stranded_jobs <= 0.5 * headline.stranded_jobs:
        safety.append("stranded_jobs<=0.5x_headline")
    if headline.fragmentation_score > 0 and \
            ca.fragmentation_score <= 0.5 * headline.fragmentation_score:
        safety.append("fragmentation<=0.5x_headline")

    fifo = results.get("fifo")
    fifo_g = (fifo.goodput_per_dollar or 0.0) if fifo else 0.0
    fifo_margin = ((ca_g - fifo_g) / fifo_g * 100.0) if fifo_g > 0 else 0.0

    if margin > 1.0:
        out = PackingOutcome("ALPHA_WIN", margin, headline_name, safety_evidence=safety)
    elif abs(margin) <= 1.0 and safety:
        out = PackingOutcome("SAFETY_WIN", margin, headline_name, safety_evidence=safety)
    elif abs(margin) <= 1.0:
        out = PackingOutcome("TIE", margin, headline_name)
    else:
        out = PackingOutcome("LOSS", margin, headline_name,
                             loss_reasons=["weaker_than_packing_baseline"],
                             notes=f"constraint_aware below {headline_name} on goodput/$")
    out.beats_fifo = ca_g >= fifo_g
    out.fifo_margin_pct = fifo_margin
    return out


@dataclass
class GPUBacktestResult:
    n_jobs: int
    n_gpu_jobs: int
    n_nodes: int
    fleet_gpu_count: int
    policy_results: dict
    outcome: PackingOutcome

    def to_summary_dict(self) -> dict:
        return {
            "primary_kpi": "sla_safe_goodput_per_infrastructure_dollar",
            "goodput_unit": "completed_gpu_job_work (token_equivalent)",
            "headline_baseline": self.outcome.headline,
            "headline_is_packing_baseline": self.outcome.headline in HEADLINE_CANDIDATES,
            "n_jobs": self.n_jobs,
            "n_gpu_jobs": self.n_gpu_jobs,
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
    policies: Sequence[str] = PACKING_POLICIES,
) -> GPUBacktestResult:
    results = {p: run_packing(jobs, nodes, p) for p in policies}
    outcome = classify(results)
    n_gpu = sum(1 for j in jobs if effective_gpu(j) > 0)
    return GPUBacktestResult(
        n_jobs=len(jobs), n_gpu_jobs=n_gpu, n_nodes=len(nodes),
        fleet_gpu_count=sum(n.gpu_count for n in nodes),
        policy_results=results, outcome=outcome,
    )
