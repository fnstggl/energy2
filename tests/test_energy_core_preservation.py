"""Guard test — the EXISTING robust energy engine output is unchanged by this PR.

This PR integrates the energy engine into the constraint-aware optimizer via a
thin adapter; it must NOT change the energy engine's algorithms or constants
(see docs/ENERGY_SYSTEM_MAP.md "core energy code"). These snapshot assertions
pin the standalone engine output for a fixed input so any future drift in the
core (scheduler / objective / constraints / baselines) is caught immediately.

The frozen reference values were produced by the unmodified energy core. If a
future change legitimately alters them, that change is — by definition — a
change to the energy engine and must be justified explicitly (the whole point
of this guard).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from aurelius.backtesting.baselines import current_price_only_policy, fifo_policy
from aurelius.backtesting.evaluator import evaluate_schedule
from aurelius.models import Job, OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler

W = datetime(2026, 2, 1, tzinfo=timezone.utc)

# --- frozen reference values (unmodified energy core) ---
STANDALONE_COST = 153.0
STANDALONE_REGIONS_HASH = (
    "6a5a7078d315b2715ee45499469662a2473369814352d879dd9b023ae4ad12e0"
)
CPO_COST = 216.0


def _price_curve(base: float) -> dict[datetime, float]:
    """Deterministic diurnal curve: +30 $/MWh during the 08:00–20:00 peak."""
    return {
        W + timedelta(hours=h): base + 30.0 * (8 <= (h % 24) < 20)
        for h in range(0, 72)
    }


def _fixture():
    da = {
        "us-west": _price_curve(30.0),
        "us-east": _price_curve(60.0),
        "us-south": _price_curve(45.0),
    }
    carbon = {r: {} for r in da}
    jobs = []
    for i in range(12):
        rt_h = [2, 4, 6][i % 3]
        slack = [6, 12, 24][i % 3]
        es = W + timedelta(hours=i)
        jobs.append(Job(
            job_id=f"job-{i:03d}", submit_time=es, runtime_hours=rt_h,
            deadline=es + timedelta(hours=rt_h + slack), power_kw=100.0,
            earliest_start=es, region_options=["us-west", "us-east", "us-south"],
            gpu_count=2, workload_type="llm_batch_inference", migration_cost_hours=0.1,
        ))
    return jobs, da, carbon


def _regions_hash(schedule) -> str:
    payload = [
        (d.job_id, d.region, d.start_time.isoformat(), round(d.power_fraction, 3))
        for d in schedule
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_standalone_energy_engine_output_unchanged():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    res = JobScheduler(cfg).solve(jobs, da, carbon, method="greedy")
    cost = round(evaluate_schedule(res.schedule, jobs, da, carbon).total_energy_cost_usd, 6)
    assert cost == STANDALONE_COST, (
        f"energy engine realized cost changed: {cost} != {STANDALONE_COST}. "
        "The energy core must not be modified by this integration PR."
    )
    assert _regions_hash(res.schedule) == STANDALONE_REGIONS_HASH, (
        "energy engine placement decisions changed — core energy code drifted."
    )


def test_standalone_energy_engine_is_deterministic():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    sch = JobScheduler(cfg)
    h1 = _regions_hash(sch.solve(jobs, da, carbon, method="greedy").schedule)
    jobs2, da2, carbon2 = _fixture()
    h2 = _regions_hash(sch.solve(jobs2, da2, carbon2, method="greedy").schedule)
    assert h1 == h2 == STANDALONE_REGIONS_HASH


def test_current_price_only_baseline_unchanged():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    sched = current_price_only_policy(jobs, da, carbon, cfg)
    cost = round(evaluate_schedule(sched, jobs, da, carbon).total_energy_cost_usd, 6)
    assert cost == CPO_COST
    # current_price_only picks the cheapest region at earliest_start: us-west here.
    assert all(d.region == "us-west" for d in sched)


def test_fifo_baseline_unchanged():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    sched = fifo_policy(jobs, da, carbon, cfg)
    # FIFO is price-blind: every job stays in the default region.
    assert all(d.region == "us-east" for d in sched)


def test_adapter_consumes_engine_output_without_changing_the_target():
    """The adapter wraps the engine; running the engine THROUGH the adapter must
    reproduce the engine's own region choices verbatim (it only gates them)."""
    from aurelius.constraints.energy_adapter import EnergyArbitrageAdapter
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    sch = JobScheduler(cfg)
    direct = {d.job_id: d.region for d in sch.solve(jobs, da, carbon, method="greedy").schedule}
    adapter = EnergyArbitrageAdapter(config=cfg)
    cands = adapter.recommend(jobs, da, carbon_data=carbon, method="greedy")
    for c in cands:
        assert c.recommended_region == direct[c.job_id], (
            "adapter altered the energy engine's recommended region"
        )
