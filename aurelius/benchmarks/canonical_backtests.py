"""Canonical CAISO / PJM / ERCOT 1000-job energy backtest — frozen benchmark.

This module FREEZES a single, fully-deterministic energy backtest so that every
future optimizer / forecasting / adapter change can be compared apples-to-apples
against a fixed reference. See docs/BACKTESTS.md for the human-readable spec and
docs/ENERGY_SYSTEM_MAP.md for what is "core energy code" (and must not change).

The backtest lives in the EXISTING energy engine's world (Job lists + energy
price data + ScheduleDecisions), NOT the cluster simulator. It exercises:

  * the existing robust energy optimizer standalone
    (aurelius.optimization.scheduler.JobScheduler — UNCHANGED),
  * the deterministic baselines (fifo / current_price_only / greedy_energy),
  * the constraint-aware energy adapter
    (aurelius.constraints.energy_adapter.EnergyArbitrageAdapter — gates only).

Determinism guarantees (docs/BACKTESTS.md §"stability"):
  * Fixed seed, fixed job count (1000), fixed workload mix, fixed deadlines /
    flexibility / migration flags / region eligibility.
  * Fixed energy data windows + fixed data file paths (DA + RT, all 3 ISOs).
  * No PyYAML / optional dependency on the result.
  * No dict/set-ordering dependence (jobs and regions are processed in a stable
    sorted order; the golden summary rounds all floats).

Primary KPI: SLA-safe goodput per infrastructure dollar (docs/RESULTS.md §1).
Goodput unit: ``token_equivalent`` (gpu_count * runtime_hours job-progress proxy
for batch/training, per docs/RESULTS.md §5). Deadline miss => zero goodput (SLA
filter on the numerator). No business-value or revenue weights anywhere.

Simulator/benchmark result only — NOT a production-savings claim
(docs/RESULTS.md §8).
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..backtesting.baselines import current_price_only_policy
from ..backtesting.evaluator import evaluate_schedule
from ..constraints.energy_adapter import (
    DestinationContext,
    EnergyArbitrageAdapter,
    GateDecision,
)
from ..models import Job, OptimizationConfig, ScheduleDecision

# ---------------------------------------------------------------------------
# FROZEN canonical constants — changing any of these changes the benchmark and
# MUST be accompanied by a regenerated golden snapshot + a PR-body explanation.
# ---------------------------------------------------------------------------

CANONICAL_SEED: int = 20260201
CANONICAL_JOB_COUNT: int = 1000

# ISO -> internal region mapping (see docs/ENERGY_SYSTEM_MAP.md).
REGION_CAISO = "us-west"
REGION_PJM = "us-east"
REGION_ERCOT = "us-south"
CANONICAL_REGIONS: tuple[str, ...] = (REGION_CAISO, REGION_PJM, REGION_ERCOT)
ISO_FOR_REGION: dict[str, str] = {
    REGION_CAISO: "CAISO",
    REGION_PJM: "PJM",
    REGION_ERCOT: "ERCOT",
}

# Fixed evaluation window. Chosen to sit fully inside the DA + RT data ranges of
# ALL THREE ISOs (ERCOT DAM starts 2026-01-28, ends 2026-03-01) with headroom.
CANONICAL_WINDOW_START = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
CANONICAL_WINDOW_END = datetime(2026, 2, 27, 0, 0, tzinfo=timezone.utc)

# Fixed data file paths (relative to repo root). DA = planning price; RT =
# realized settlement price (what the optimizer is actually charged).
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
CANONICAL_DA_FILES: dict[str, str] = {
    REGION_CAISO: os.path.join(_DATA_DIR, "caiso_us_west_dam.csv"),
    REGION_PJM: os.path.join(_DATA_DIR, "pjm_us_east_dam.csv"),
    REGION_ERCOT: os.path.join(_DATA_DIR, "ercot_us_south_dam.csv"),
}
CANONICAL_RT_FILES: dict[str, str] = {
    REGION_CAISO: os.path.join(_DATA_DIR, "caiso_us_west_rt.csv"),
    REGION_PJM: os.path.join(_DATA_DIR, "pjm_us_east_rt.csv"),
    REGION_ERCOT: os.path.join(_DATA_DIR, "ercot_us_south_rt.csv"),
}

# Fixed job mix (workload_type, fraction, migration_cost_hours, latency_pinned).
# Fractions sum to 1.0; counts are computed deterministically so the total is
# exactly CANONICAL_JOB_COUNT.
@dataclass(frozen=True)
class _WorkloadClassSpec:
    workload_type: str
    fraction: float
    migration_cost_hours: Optional[float]   # None => cannot migrate
    runtime_choices: tuple[int, ...]
    slack_choices: tuple[int, ...]
    power_choices: tuple[float, ...]
    gpu_choices: tuple[int, ...]
    single_region_prob: float               # data-residency pin probability


CANONICAL_WORKLOAD_MIX: tuple[_WorkloadClassSpec, ...] = (
    _WorkloadClassSpec("llm_batch_inference", 0.35, 0.10,
                       (2, 4, 6, 8), (6, 12, 24), (50.0, 100.0, 200.0), (1, 2, 4), 0.15),
    _WorkloadClassSpec("data_processing", 0.15, 0.05,
                       (1, 2, 4, 6), (6, 12, 24), (20.0, 50.0, 100.0), (1, 2), 0.10),
    _WorkloadClassSpec("scheduled_batch", 0.15, 0.10,
                       (2, 4, 8), (8, 16, 24), (20.0, 100.0), (1, 2), 0.10),
    _WorkloadClassSpec("fine_tuning", 0.10, 0.25,
                       (4, 8, 12), (12, 24), (100.0, 300.0), (2, 4), 0.20),
    _WorkloadClassSpec("training", 0.10, 0.50,
                       (8, 12, 24), (24, 48), (200.0, 400.0), (4, 8), 0.20),
    _WorkloadClassSpec("realtime_inference", 0.15, None,
                       (1, 2), (0,), (5.0, 20.0, 80.0), (1, 2), 0.10),
)

# Fixed cost basis for the canonical KPI (public-list directional rate, NOT a
# production procurement rate — docs/RESULTS.md §8).
CANONICAL_GPU_HOUR_USD: float = 2.0
CANONICAL_MIGRATION_NETWORK_USD: float = 0.5

# Optimizer method used for the standalone energy engine. Frozen.
CANONICAL_METHOD: str = "greedy"

# Committed golden snapshot path (regenerate deliberately — see docs/BACKTESTS.md).
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden", "canonical_energy_backtest.json",
)

# Fixed, healthy destination contexts for the canonical run (all three ISOs have
# ample capacity, no thermal hotspot, high-confidence telemetry). Destination
# REJECTIONS (hot/full/stale/bad-topology) are proven in the adapter unit tests,
# not forced into the canonical trace. Keeping these healthy keeps the canonical
# delta attributable to ELIGIBILITY + SLA + KPI gates, which is the honest story.
def _canonical_destination_contexts() -> dict[str, DestinationContext]:
    return {
        r: DestinationContext(
            region=r, spare_capacity_pct=40.0, is_hot=False,
            queue_p95_ms=200.0, telemetry_confidence="high",
            is_stale=False, topology_fit_ok=True, is_cold=False,
        )
        for r in CANONICAL_REGIONS
    }


POLICY_FIFO = "fifo"
POLICY_CURRENT_PRICE_ONLY = "current_price_only"
POLICY_GREEDY_ENERGY = "greedy_energy"
POLICY_ROBUST_STANDALONE = "robust_energy_standalone"
POLICY_CONSTRAINT_AWARE_ADAPTER = "constraint_aware_with_energy_adapter"
POLICY_SLA_AWARE = "sla_aware"

CANONICAL_POLICIES: tuple[str, ...] = (
    POLICY_FIFO,
    POLICY_CURRENT_PRICE_ONLY,
    POLICY_GREEDY_ENERGY,
    POLICY_ROBUST_STANDALONE,
    POLICY_SLA_AWARE,
    POLICY_CONSTRAINT_AWARE_ADAPTER,
)


# ---------------------------------------------------------------------------
# Deterministic job trace
# ---------------------------------------------------------------------------

def build_canonical_jobs(
    seed: int = CANONICAL_SEED,
    count: int = CANONICAL_JOB_COUNT,
) -> list[Job]:
    """Build the FROZEN 1000-job CAISO/PJM/ERCOT workload trace.

    Fully deterministic given (seed, count). Job ids are stable (``job-00000``);
    no uuid is used so repeated builds are byte-identical.
    """
    rng = random.Random(seed)
    # Deterministic per-class counts that sum exactly to ``count``.
    counts = [int(round(spec.fraction * count)) for spec in CANONICAL_WORKLOAD_MIX]
    counts[0] += count - sum(counts)  # absorb rounding into the first (largest) class
    window_hours = (CANONICAL_WINDOW_END - CANONICAL_WINDOW_START).total_seconds() / 3600.0

    # Build an ordered (class, index) plan, then sort jobs by submit_time so the
    # trace order is stable regardless of class iteration order.
    plan: list[_WorkloadClassSpec] = []
    for spec, n in zip(CANONICAL_WORKLOAD_MIX, counts):
        plan.extend([spec] * n)

    jobs: list[Job] = []
    for i, spec in enumerate(plan):
        runtime = float(rng.choice(spec.runtime_choices))
        slack = float(rng.choice(spec.slack_choices))
        # Submit early enough that runtime + slack fits inside the window.
        latest_submit = max(1.0, window_hours - runtime - slack - 1.0)
        submit = CANONICAL_WINDOW_START + timedelta(hours=rng.uniform(0.0, latest_submit))
        submit = submit.replace(minute=0, second=0, microsecond=0)
        earliest_start = submit
        deadline = earliest_start + timedelta(hours=runtime + slack)
        power = float(rng.choice(spec.power_choices))
        gpu = int(rng.choice(spec.gpu_choices))
        if rng.random() < spec.single_region_prob:
            region_options = [rng.choice(CANONICAL_REGIONS)]
        else:
            region_options = list(CANONICAL_REGIONS)
        jobs.append(Job(
            job_id=f"job-{i:05d}",
            submit_time=submit,
            runtime_hours=runtime,
            deadline=deadline,
            power_kw=power,
            earliest_start=earliest_start,
            region_options=region_options,
            workload_type=spec.workload_type,
            gpu_count=gpu,
            migration_cost_hours=spec.migration_cost_hours,
        ))
    jobs.sort(key=lambda j: (j.submit_time, j.job_id))
    return jobs


# ---------------------------------------------------------------------------
# Price data loading (DA + RT) — pure csv, no DB / ingestion layer
# ---------------------------------------------------------------------------

def _load_price_csv(path: str) -> dict[datetime, float]:
    out: dict[datetime, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            if CANONICAL_WINDOW_START <= ts <= CANONICAL_WINDOW_END:
                out[ts] = float(row["price_per_mwh"])
    return out


def load_canonical_price_data() -> tuple[
    dict[str, dict[datetime, float]], dict[str, dict[datetime, float]]
]:
    """Return ``(da_price_data, rt_price_data)`` sliced to the canonical window."""
    da = {r: _load_price_csv(CANONICAL_DA_FILES[r]) for r in CANONICAL_REGIONS}
    rt = {r: _load_price_csv(CANONICAL_RT_FILES[r]) for r in CANONICAL_REGIONS}
    return da, rt


# ---------------------------------------------------------------------------
# Per-policy metrics
# ---------------------------------------------------------------------------

@dataclass
class PolicyMetrics:
    policy: str
    realized_energy_cost_usd: float
    da_planned_cost_usd: float
    da_rt_basis_usd: float                 # realized - DA-planned (>0 = adverse)
    gpu_infra_cost_usd: float
    network_cost_usd: float
    total_infra_cost_usd: float
    migrations: int
    deadline_misses: int                   # SLA violations
    sla_compliant_goodput: float           # token_equivalent
    sla_safe_goodput_per_infra_dollar: float
    cost_per_sla_compliant_job: float
    gross_energy_savings_vs_fifo_usd: float = 0.0
    net_energy_savings_vs_fifo_usd: float = 0.0
    # Adapter-only diagnostics.
    candidates_generated: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    candidates_deferred: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "realized_energy_cost_usd": round(self.realized_energy_cost_usd, 2),
            "da_planned_cost_usd": round(self.da_planned_cost_usd, 2),
            "da_rt_basis_usd": round(self.da_rt_basis_usd, 2),
            "gpu_infra_cost_usd": round(self.gpu_infra_cost_usd, 2),
            "network_cost_usd": round(self.network_cost_usd, 2),
            "total_infra_cost_usd": round(self.total_infra_cost_usd, 2),
            "migrations": self.migrations,
            "deadline_misses": self.deadline_misses,
            "sla_compliant_goodput": round(self.sla_compliant_goodput, 2),
            "sla_safe_goodput_per_infra_dollar": round(
                self.sla_safe_goodput_per_infra_dollar, 6
            ),
            "cost_per_sla_compliant_job": round(self.cost_per_sla_compliant_job, 4),
            "gross_energy_savings_vs_fifo_usd": round(
                self.gross_energy_savings_vs_fifo_usd, 2
            ),
            "net_energy_savings_vs_fifo_usd": round(
                self.net_energy_savings_vs_fifo_usd, 2
            ),
            "candidates_generated": self.candidates_generated,
            "candidates_accepted": self.candidates_accepted,
            "candidates_rejected": self.candidates_rejected,
            "candidates_deferred": self.candidates_deferred,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


def _score_schedule(
    policy: str,
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    baseline_regions: dict[str, str],
    da: dict[str, dict[datetime, float]],
    rt: dict[str, dict[datetime, float]],
) -> PolicyMetrics:
    """Compute frozen metrics for a schedule, scored on REALIZED (RT) prices."""
    job_by_id = {j.job_id: j for j in jobs}
    realized = evaluate_schedule(schedule, jobs, rt, {r: {} for r in da},
                                 warn_on_missing=False)
    planned = evaluate_schedule(schedule, jobs, da, {r: {} for r in da},
                                warn_on_missing=False)

    gpu_infra = 0.0
    migrations = 0
    deadline_misses = 0
    goodput = 0.0
    for d in schedule:
        job = job_by_id.get(d.job_id)
        if job is None:
            continue
        gpu_infra += CANONICAL_GPU_HOUR_USD * max(0, job.gpu_count) * job.runtime_hours
        moved = d.region != baseline_regions.get(d.job_id, d.region)
        if moved:
            migrations += 1
        # Warmup-aware deadline check: relocating a workload to a non-home region
        # incurs migration_cost_hours of warmup the WARMUP-BLIND energy engine
        # does not budget for. Scoring it consistently with the adapter exposes
        # exactly the deadline-edge placements the constraint-aware wrapper
        # reverts (a safety win the energy engine alone cannot see).
        completion = d.end_time
        if moved and job.migration_cost_hours:
            completion += timedelta(hours=job.migration_cost_hours)
        if completion > job.deadline:
            deadline_misses += 1
        else:
            unit = max(0.0, job.gpu_count * job.runtime_hours)
            goodput += unit if unit > 0 else job.runtime_hours

    network_cost = migrations * CANONICAL_MIGRATION_NETWORK_USD
    energy = realized.total_energy_cost_usd
    total_infra = energy + gpu_infra + network_cost
    compliant_jobs = len(schedule) - deadline_misses
    return PolicyMetrics(
        policy=policy,
        realized_energy_cost_usd=energy,
        da_planned_cost_usd=planned.total_energy_cost_usd,
        da_rt_basis_usd=energy - planned.total_energy_cost_usd,
        gpu_infra_cost_usd=gpu_infra,
        network_cost_usd=network_cost,
        total_infra_cost_usd=total_infra,
        migrations=migrations,
        deadline_misses=deadline_misses,
        sla_compliant_goodput=goodput,
        sla_safe_goodput_per_infra_dollar=(goodput / total_infra if total_infra > 0 else 0.0),
        cost_per_sla_compliant_job=(total_infra / compliant_jobs if compliant_jobs > 0 else 0.0),
    )


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _greedy_energy_policy(
    jobs: list[Job],
    da: dict[str, dict[datetime, float]],
) -> list[ScheduleDecision]:
    """Aggressive energy baseline: cheapest (region, start-hour) in each job's
    feasible window on the DA price, ignoring migration cost / basis / SLA.

    This is a deterministic COMPARISON BASELINE (like FIFO), not the energy
    engine — it has no safety awareness. It exists to show what naive
    energy-greedy would do.
    """
    out: list[ScheduleDecision] = []
    for job in sorted(jobs, key=lambda j: (j.submit_time, j.job_id)):
        best_region = job.region_options[0]
        best_start = job.earliest_start
        best_price = float("inf")
        # Feasible start hours: earliest_start .. latest_start (deadline-runtime).
        latest_start = job.deadline - timedelta(hours=job.runtime_hours)
        hour = job.earliest_start.replace(minute=0, second=0, microsecond=0)
        while hour <= latest_start:
            for region in job.region_options:
                price = da.get(region, {}).get(hour)
                if price is not None and price < best_price:
                    best_price = price
                    best_region = region
                    best_start = hour
            hour += timedelta(hours=1)
        out.append(ScheduleDecision(
            job_id=job.job_id, start_time=best_start, region=best_region,
            power_fraction=1.0, actual_runtime_hours=job.runtime_hours,
        ))
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class CanonicalBacktestSummary:
    seed: int
    job_count: int
    regions: tuple[str, ...]
    window_start: str
    window_end: str
    method: str
    policies: dict[str, PolicyMetrics]
    standalone_vs_wrapped_delta: dict
    data_files: dict[str, str]

    def golden_dict(self) -> dict:
        """Stable, fully-rounded, sort-ordered dict for the golden snapshot."""
        return {
            "schema_version": 1,
            "seed": self.seed,
            "job_count": self.job_count,
            "regions": list(self.regions),
            "iso_for_region": dict(sorted(ISO_FOR_REGION.items())),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "method": self.method,
            "goodput_unit": "token_equivalent",
            "primary_kpi": "sla_safe_goodput_per_infrastructure_dollar",
            "cost_basis": {
                "gpu_hour_usd": CANONICAL_GPU_HOUR_USD,
                "migration_network_usd": CANONICAL_MIGRATION_NETWORK_USD,
            },
            "data_files": dict(sorted(self.data_files.items())),
            "policies": {
                name: self.policies[name].to_dict()
                for name in sorted(self.policies)
            },
            "standalone_vs_wrapped_delta": self.standalone_vs_wrapped_delta,
            "disclaimer": (
                "Simulator/benchmark result. Directional only — not production "
                "savings. Requires live telemetry calibration."
            ),
        }


def run_canonical_backtest(
    seed: int = CANONICAL_SEED,
    job_count: int = CANONICAL_JOB_COUNT,
    method: str = CANONICAL_METHOD,
) -> CanonicalBacktestSummary:
    """Run the frozen canonical backtest under all policies and summarize."""
    jobs = build_canonical_jobs(seed, job_count)
    da, rt = load_canonical_price_data()
    carbon = {r: {} for r in CANONICAL_REGIONS}
    cfg = OptimizationConfig(default_region=REGION_PJM, min_power_fraction=1.0)

    # --- the EXISTING energy engine, via its public API ---
    adapter = EnergyArbitrageAdapter(
        config=cfg,
        gpu_hour_usd=CANONICAL_GPU_HOUR_USD,
        migration_network_usd=CANONICAL_MIGRATION_NETWORK_USD,
    )
    from ..optimization.scheduler import JobScheduler
    scheduler = JobScheduler(cfg)
    standalone_result = scheduler.solve(jobs, da, carbon, method=method)
    standalone_schedule = standalone_result.schedule
    # ASAP home placement = "no optimization" sanity baseline (FIFO) + the safe
    # no-move fallback for rejected candidates. Each job runs at its own
    # earliest_start in the default region; no cross-job queueing artifact (the
    # energy evaluator scores each job's window independently), so it has zero
    # deadline misses and is the honest "did we make things worse?" reference.
    asap_baseline = scheduler.create_baseline_schedule(jobs)
    fifo_schedule = asap_baseline
    baseline_regions = {d.job_id: d.region for d in asap_baseline}

    # Energy candidates (verbatim from the engine), then gate them.
    candidates = adapter.candidates_from_schedules(
        jobs=jobs, baseline_schedule=asap_baseline,
        optimized_schedule=standalone_schedule, da_price_data=da, rt_price_data=rt,
    )
    verdicts = adapter.evaluate_all(candidates, _canonical_destination_contexts())

    # Wrapped schedule: accepted -> engine's decision; rejected/deferred ->
    # safe baseline placement. Standalone decisions are indexed by job_id.
    standalone_by_id = {d.job_id: d for d in standalone_schedule}
    fifo_by_id = {d.job_id: d for d in fifo_schedule}
    wrapped_schedule: list[ScheduleDecision] = []
    accepted = rejected = deferred = 0
    rejection_reasons: dict[str, int] = {}
    for v in verdicts:
        jid = v.candidate.job_id
        if v.decision in (GateDecision.ACCEPT, GateDecision.MODIFY):
            accepted += 1
            wrapped_schedule.append(standalone_by_id[jid])
        else:
            if v.decision is GateDecision.DEFER:
                deferred += 1
            else:
                rejected += 1
            primary = v.primary_reason
            rejection_reasons[primary] = rejection_reasons.get(primary, 0) + 1
            wrapped_schedule.append(fifo_by_id[jid])

    # sla_aware = energy engine + deadline/latency safety ONLY (no destination /
    # KPI gates). Latency-pinned jobs and deadline-unsafe moves revert to FIFO.
    sla_schedule: list[ScheduleDecision] = []
    for d in standalone_schedule:
        job = next((j for j in jobs if j.job_id == d.job_id), None)
        if job is None:
            sla_schedule.append(d)
            continue
        unsafe = (
            job.workload_type in ("realtime_inference",)
            or d.end_time > job.deadline
        )
        sla_schedule.append(fifo_by_id[d.job_id] if unsafe else d)

    schedules = {
        POLICY_FIFO: fifo_schedule,
        POLICY_CURRENT_PRICE_ONLY: current_price_only_policy(jobs, da, carbon, cfg),
        POLICY_GREEDY_ENERGY: _greedy_energy_policy(jobs, da),
        POLICY_ROBUST_STANDALONE: standalone_schedule,
        POLICY_SLA_AWARE: sla_schedule,
        POLICY_CONSTRAINT_AWARE_ADAPTER: wrapped_schedule,
    }

    metrics: dict[str, PolicyMetrics] = {}
    for name in CANONICAL_POLICIES:
        metrics[name] = _score_schedule(
            name, schedules[name], jobs, baseline_regions, da, rt
        )

    # Savings vs FIFO (gross = energy only; net = total infra incl. migration).
    fifo_m = metrics[POLICY_FIFO]
    for name, m in metrics.items():
        m.gross_energy_savings_vs_fifo_usd = (
            fifo_m.realized_energy_cost_usd - m.realized_energy_cost_usd
        )
        m.net_energy_savings_vs_fifo_usd = (
            fifo_m.total_infra_cost_usd - m.total_infra_cost_usd
        )

    # Adapter diagnostics on the wrapped policy.
    wm = metrics[POLICY_CONSTRAINT_AWARE_ADAPTER]
    wm.candidates_generated = len(candidates)
    wm.candidates_accepted = accepted
    wm.candidates_rejected = rejected
    wm.candidates_deferred = deferred
    wm.rejection_reasons = rejection_reasons

    standalone_m = metrics[POLICY_ROBUST_STANDALONE]
    delta = {
        "standalone_energy_cost_usd": round(standalone_m.realized_energy_cost_usd, 2),
        "wrapped_energy_cost_usd": round(wm.realized_energy_cost_usd, 2),
        "wrapped_minus_standalone_energy_usd": round(
            wm.realized_energy_cost_usd - standalone_m.realized_energy_cost_usd, 2
        ),
        "standalone_deadline_misses": standalone_m.deadline_misses,
        "wrapped_deadline_misses": wm.deadline_misses,
        "standalone_goodput_per_dollar": round(
            standalone_m.sla_safe_goodput_per_infra_dollar, 6
        ),
        "wrapped_goodput_per_dollar": round(
            wm.sla_safe_goodput_per_infra_dollar, 6
        ),
        "candidates_generated": len(candidates),
        "candidates_accepted": accepted,
        "candidates_rejected": rejected,
        "candidates_deferred": deferred,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }

    return CanonicalBacktestSummary(
        seed=seed,
        job_count=job_count,
        regions=CANONICAL_REGIONS,
        window_start=CANONICAL_WINDOW_START.isoformat(),
        window_end=CANONICAL_WINDOW_END.isoformat(),
        method=method,
        policies=metrics,
        standalone_vs_wrapped_delta=delta,
        data_files={
            **{f"da_{r}": os.path.relpath(CANONICAL_DA_FILES[r], _DATA_DIR)
               for r in CANONICAL_REGIONS},
            **{f"rt_{r}": os.path.relpath(CANONICAL_RT_FILES[r], _DATA_DIR)
               for r in CANONICAL_REGIONS},
        },
    )
