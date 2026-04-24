"""Tests for the job scheduler: power cap enforcement in greedy and local_search."""

import pytest
from datetime import datetime, timezone, timedelta

from aurelius.models import Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler
from aurelius.optimization.constraints import ConstraintBuilder

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 0, 0, tzinfo=UTC)


def _h(n):
    return timedelta(hours=n)


def _make_price(regions, hours=48, price=50.0):
    data = {}
    for r in regions:
        data[r] = {}
        for h in range(hours):
            ts = (T0 + _h(h)).replace(minute=0, second=0, microsecond=0)
            data[r][ts] = price
    return data


def _make_carbon(regions, hours=48, carbon=300.0):
    data = {}
    for r in regions:
        data[r] = {}
        for h in range(hours):
            ts = (T0 + _h(h)).replace(minute=0, second=0, microsecond=0)
            data[r][ts] = carbon
    return data


# ---------------------------------------------------------------------------
# ConstraintBuilder.would_violate_power_cap
# ---------------------------------------------------------------------------

class TestWouldViolatePowerCap:
    def setup_method(self):
        self.config = OptimizationConfig(region_power_caps={"us-west": 200.0})
        self.cb = ConstraintBuilder(self.config)

    def _job(self, job_id, power_kw, start_offset=0):
        return Job(
            job_id=job_id,
            submit_time=T0,
            runtime_hours=2.0,
            deadline=T0 + _h(24),
            power_kw=power_kw,
            earliest_start=T0 + _h(start_offset),
            region_options=["us-west"],
        )

    def _decision(self, job, region="us-west", power_fraction=1.0):
        return ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=region,
            power_fraction=power_fraction,
            actual_runtime_hours=job.runtime_hours,
        )

    def test_no_violation_when_no_existing_schedule(self):
        job = self._job("j1", power_kw=100.0)
        dec = self._decision(job)
        assert not self.cb.would_violate_power_cap(job, dec, [], [job])

    def test_no_violation_under_cap(self):
        job1 = self._job("j1", power_kw=100.0, start_offset=0)
        job2 = self._job("j2", power_kw=80.0, start_offset=0)
        dec1 = self._decision(job1)
        dec2 = self._decision(job2)
        # 100 + 80 = 180 < 200 cap
        assert not self.cb.would_violate_power_cap(job2, dec2, [dec1], [job1, job2])

    def test_violation_over_cap(self):
        job1 = self._job("j1", power_kw=150.0, start_offset=0)
        job2 = self._job("j2", power_kw=100.0, start_offset=0)
        dec1 = self._decision(job1)
        dec2 = self._decision(job2)
        # 150 + 100 = 250 > 200 cap
        assert self.cb.would_violate_power_cap(job2, dec2, [dec1], [job1, job2])

    def test_no_cap_configured_returns_false(self):
        config = OptimizationConfig(region_power_caps={})
        cb = ConstraintBuilder(config)
        job = self._job("j1", power_kw=999999.0)
        dec = self._decision(job)
        assert not cb.would_violate_power_cap(job, dec, [], [job])

    def test_non_overlapping_jobs_no_violation(self):
        # job1 runs 00-02, job2 runs 04-06 → no overlap
        job1 = self._job("j1", power_kw=150.0, start_offset=0)
        job2 = self._job("j2", power_kw=150.0, start_offset=4)
        dec1 = self._decision(job1)
        dec2 = self._decision(job2)
        assert not self.cb.would_violate_power_cap(job2, dec2, [dec1], [job1, job2])


# ---------------------------------------------------------------------------
# Greedy solver power cap enforcement
# ---------------------------------------------------------------------------

class TestGreedyPowerCap:
    def test_greedy_respects_power_cap(self):
        """With a 200 kW cap, scheduler must not overlap 150 kW + 150 kW jobs."""
        cap = 200.0
        config = OptimizationConfig(
            region_power_caps={"us-west": cap},
            default_region="us-west",
        )
        scheduler = JobScheduler(config)

        # Two jobs both wanting to start at T0 at 150 kW – would exceed 200 kW cap
        job1 = Job(
            job_id="j1", submit_time=T0, runtime_hours=2.0, deadline=T0 + _h(24),
            power_kw=150.0, earliest_start=T0, region_options=["us-west"],
        )
        job2 = Job(
            job_id="j2", submit_time=T0, runtime_hours=2.0, deadline=T0 + _h(24),
            power_kw=150.0, earliest_start=T0, region_options=["us-west"],
        )

        price_data = _make_price(["us-west"])
        carbon_data = _make_carbon(["us-west"])

        result = scheduler.solve([job1, job2], price_data, carbon_data, method="greedy")
        schedule = result.schedule

        cb = ConstraintBuilder(config)
        violations = cb.check_schedule_constraints([job1, job2], schedule)
        power_violations = [v for v in violations if v.constraint_type == "power_cap"]
        assert power_violations == [], (
            f"Greedy solver produced power cap violations: {power_violations}"
        )

    def test_local_search_respects_power_cap(self):
        """local_search must also produce a schedule with no power cap violations."""
        cap = 200.0
        config = OptimizationConfig(
            region_power_caps={"us-west": cap},
            default_region="us-west",
        )
        scheduler = JobScheduler(config)

        jobs = [
            Job(
                job_id=f"j{i}", submit_time=T0, runtime_hours=2.0, deadline=T0 + _h(24),
                power_kw=150.0, earliest_start=T0, region_options=["us-west"],
            )
            for i in range(2)
        ]

        price_data = _make_price(["us-west"])
        carbon_data = _make_carbon(["us-west"])

        result = scheduler.solve(jobs, price_data, carbon_data, method="local_search", time_limit_seconds=5)
        schedule = result.schedule

        cb = ConstraintBuilder(config)
        violations = cb.check_schedule_constraints(jobs, schedule)
        power_violations = [v for v in violations if v.constraint_type == "power_cap"]
        assert power_violations == []
