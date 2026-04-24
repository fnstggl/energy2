"""Deterministic baseline scheduling policies for backtesting comparison.

All 7 policies share the same signature:
    policy(jobs, price_data, carbon_data, config) -> list[ScheduleDecision]

where:
    jobs       – list[Job]
    price_data – {region: {timestamp: price_per_mwh}}
    carbon_data– {region: {timestamp: gco2_per_kwh}}
    config     – OptimizationConfig

All policies are deterministic given the same inputs.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Callable, Optional

from aurelius.models import Job, ScheduleDecision, OptimizationConfig

BaselinePolicy = Callable[
    [list[Job], dict, dict, OptimizationConfig],
    list[ScheduleDecision],
]


def _default_region(job: Job, config: OptimizationConfig) -> str:
    """Return the job's preferred region, falling back to first option."""
    if config.default_region in job.region_options:
        return config.default_region
    return job.region_options[0]


# ---------------------------------------------------------------------------
# 1. FIFO — jobs run in submission order, queued per region
# ---------------------------------------------------------------------------

def fifo_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Schedule jobs strictly in submission order (FIFO), no price awareness."""
    sorted_jobs = sorted(jobs, key=lambda j: j.submit_time)
    region_free: dict[str, datetime] = {}
    schedule: list[ScheduleDecision] = []

    for job in sorted_jobs:
        region = _default_region(job, config)
        free_at = region_free.get(region, job.earliest_start)
        start = max(job.earliest_start, free_at)
        region_free[region] = start + timedelta(hours=job.runtime_hours)
        schedule.append(ScheduleDecision(
            job_id=job.job_id,
            start_time=start,
            region=region,
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        ))

    return schedule


# ---------------------------------------------------------------------------
# 2. Peak-blind ASAP — start immediately, no price awareness
# ---------------------------------------------------------------------------

def peak_blind_asap_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Start every job at its earliest_start in the default region."""
    return [
        ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=_default_region(job, config),
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        for job in jobs
    ]


# ---------------------------------------------------------------------------
# 3. Latency-first — minimise wall-clock time: full power, earliest start
# ---------------------------------------------------------------------------

def latency_first_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Always start at earliest_start at full power to minimise latency."""
    return [
        ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=_default_region(job, config),
            power_fraction=config.max_power_fraction,
            actual_runtime_hours=job.runtime_hours / config.max_power_fraction,
        )
        for job in jobs
    ]


# ---------------------------------------------------------------------------
# 4. Closest-region — pick the region with the most remaining options
#    (proxy for lowest latency / nearest availability).
#    Ties broken by alphabetical order for determinism.
# ---------------------------------------------------------------------------

def closest_region_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Pick the first (alphabetically) region from each job's allowed list."""
    return [
        ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=sorted(job.region_options)[0],
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        for job in jobs
    ]


# ---------------------------------------------------------------------------
# 5. Fixed-primary-region — always use config.default_region
# ---------------------------------------------------------------------------

def fixed_primary_region_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Always schedule in config.default_region (or first region if unavailable)."""
    return [
        ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=_default_region(job, config),
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        )
        for job in jobs
    ]


# ---------------------------------------------------------------------------
# 6. Current-price-only — pick the cheapest region at earliest_start
# ---------------------------------------------------------------------------

def current_price_only_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Choose region with the lowest price at earliest_start (no forecasting)."""
    schedule: list[ScheduleDecision] = []

    for job in sorted(jobs, key=lambda j: j.submit_time):
        anchor = job.earliest_start.replace(minute=0, second=0, microsecond=0)
        best_region = _default_region(job, config)
        best_price = float("inf")

        for region in job.region_options:
            region_prices = price_data.get(region, {})
            price = region_prices.get(anchor, float("inf"))
            if price < best_price:
                best_price = price
                best_region = region

        schedule.append(ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=best_region,
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        ))

    return schedule


# ---------------------------------------------------------------------------
# 7. Round-robin — cycle through available regions in alphabetical order
# ---------------------------------------------------------------------------

def round_robin_policy(
    jobs: list[Job],
    price_data: dict,
    carbon_data: dict,
    config: OptimizationConfig,
) -> list[ScheduleDecision]:
    """Distribute jobs across regions in round-robin order."""
    all_regions = sorted({r for job in jobs for r in job.region_options})
    region_cycle = itertools.cycle(all_regions) if all_regions else itertools.cycle(["default"])

    schedule: list[ScheduleDecision] = []
    for job in sorted(jobs, key=lambda j: j.submit_time):
        # Advance cycle to a region the job can actually run in
        for _ in range(len(all_regions) + 1):
            candidate = next(region_cycle)
            if candidate in job.region_options:
                break
        else:
            candidate = job.region_options[0]

        schedule.append(ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=candidate,
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        ))

    return schedule


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_BASELINES: dict[str, BaselinePolicy] = {
    "fifo": fifo_policy,
    "peak_blind_asap": peak_blind_asap_policy,
    "latency_first": latency_first_policy,
    "closest_region": closest_region_policy,
    "fixed_primary_region": fixed_primary_region_policy,
    "current_price_only": current_price_only_policy,
    "round_robin": round_robin_policy,
}
