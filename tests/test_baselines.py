"""Tests for all 7 deterministic baseline scheduling policies."""

import pytest
from datetime import datetime, timezone, timedelta

from aurelius.models import Job, ScheduleDecision, OptimizationConfig
from aurelius.backtesting.baselines import (
    fifo_policy,
    peak_blind_asap_policy,
    latency_first_policy,
    closest_region_policy,
    fixed_primary_region_policy,
    current_price_only_policy,
    round_robin_policy,
    ALL_BASELINES,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)


def _h(n):
    return timedelta(hours=n)


def _make_jobs(n=4, regions=("us-west", "us-east")):
    jobs = []
    for i in range(n):
        jobs.append(Job(
            job_id=f"job-{i}",
            submit_time=T0 + _h(i),
            runtime_hours=2.0,
            deadline=T0 + _h(48),
            power_kw=100.0,
            earliest_start=T0 + _h(i),
            region_options=list(regions),
            priority=1,
        ))
    return jobs


def _make_price_data(regions, hours=48):
    price_data = {}
    for region in regions:
        price_data[region] = {}
        for h in range(hours):
            ts = T0 + _h(h)
            ts = ts.replace(minute=0, second=0, microsecond=0)
            price_data[region][ts] = 50.0 + h % 24
    return price_data


def _make_carbon_data(regions, hours=48):
    carbon_data = {}
    for region in regions:
        carbon_data[region] = {}
        for h in range(hours):
            ts = T0 + _h(h)
            carbon_data[region][ts] = 300.0
    return carbon_data


REGIONS = ("us-west", "us-east")
CONFIG = OptimizationConfig(default_region="us-west")


@pytest.fixture
def jobs():
    return _make_jobs(regions=REGIONS)


@pytest.fixture
def price_data():
    return _make_price_data(REGIONS)


@pytest.fixture
def carbon_data():
    return _make_carbon_data(REGIONS)


def _check_schedule(schedule, jobs):
    assert len(schedule) == len(jobs)
    for dec in schedule:
        assert isinstance(dec, ScheduleDecision)
        assert dec.job_id.startswith("job-")
        assert dec.power_fraction > 0
        assert dec.actual_runtime_hours > 0
        assert dec.region in REGIONS


# ---------------------------------------------------------------------------
# 1. FIFO
# ---------------------------------------------------------------------------

class TestFifoPolicy:
    def test_produces_one_decision_per_job(self, jobs, price_data, carbon_data):
        schedule = fifo_policy(jobs, price_data, carbon_data, CONFIG)
        _check_schedule(schedule, jobs)

    def test_respects_submission_order(self, jobs, price_data, carbon_data):
        schedule = fifo_policy(jobs, price_data, carbon_data, CONFIG)
        submit_order = sorted(jobs, key=lambda j: j.submit_time)
        for dec, job in zip(schedule, submit_order):
            assert dec.job_id == job.job_id

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = fifo_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = fifo_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.job_id for d in s1] == [d.job_id for d in s2]
        assert [d.start_time for d in s1] == [d.start_time for d in s2]

    def test_start_times_respect_earliest_start(self, jobs, price_data, carbon_data):
        schedule = fifo_policy(jobs, price_data, carbon_data, CONFIG)
        job_map = {j.job_id: j for j in jobs}
        for dec in schedule:
            job = job_map[dec.job_id]
            assert dec.start_time >= job.earliest_start


# ---------------------------------------------------------------------------
# 2. Peak-blind ASAP
# ---------------------------------------------------------------------------

class TestPeakBlindAsapPolicy:
    def test_starts_at_earliest_start(self, jobs, price_data, carbon_data):
        schedule = peak_blind_asap_policy(jobs, price_data, carbon_data, CONFIG)
        job_map = {j.job_id: j for j in jobs}
        for dec in schedule:
            assert dec.start_time == job_map[dec.job_id].earliest_start

    def test_full_power(self, jobs, price_data, carbon_data):
        schedule = peak_blind_asap_policy(jobs, price_data, carbon_data, CONFIG)
        for dec in schedule:
            assert dec.power_fraction == 1.0

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = peak_blind_asap_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = peak_blind_asap_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.start_time for d in s1] == [d.start_time for d in s2]


# ---------------------------------------------------------------------------
# 3. Latency-first
# ---------------------------------------------------------------------------

class TestLatencyFirstPolicy:
    def test_max_power(self, jobs, price_data, carbon_data):
        schedule = latency_first_policy(jobs, price_data, carbon_data, CONFIG)
        for dec in schedule:
            assert dec.power_fraction == CONFIG.max_power_fraction

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = latency_first_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = latency_first_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.power_fraction for d in s1] == [d.power_fraction for d in s2]


# ---------------------------------------------------------------------------
# 4. Closest-region
# ---------------------------------------------------------------------------

class TestClosestRegionPolicy:
    def test_picks_alphabetically_first_region(self, jobs, price_data, carbon_data):
        schedule = closest_region_policy(jobs, price_data, carbon_data, CONFIG)
        for dec in schedule:
            # "us-east" < "us-west" alphabetically
            assert dec.region == "us-east"

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = closest_region_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = closest_region_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.region for d in s1] == [d.region for d in s2]


# ---------------------------------------------------------------------------
# 5. Fixed-primary-region
# ---------------------------------------------------------------------------

class TestFixedPrimaryRegionPolicy:
    def test_uses_default_region(self, jobs, price_data, carbon_data):
        schedule = fixed_primary_region_policy(jobs, price_data, carbon_data, CONFIG)
        for dec in schedule:
            assert dec.region == CONFIG.default_region

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = fixed_primary_region_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = fixed_primary_region_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.region for d in s1] == [d.region for d in s2]


# ---------------------------------------------------------------------------
# 6. Current-price-only
# ---------------------------------------------------------------------------

class TestCurrentPriceOnlyPolicy:
    def test_picks_region_with_lower_price(self, jobs, price_data, carbon_data):
        # Make us-east cheaper at T0
        price_data["us-east"][T0] = 10.0
        price_data["us-west"][T0] = 100.0
        schedule = current_price_only_policy(jobs, price_data, carbon_data, CONFIG)
        # The first job (earliest_start == T0) should pick us-east
        first = next(d for d in schedule if d.job_id == "job-0")
        assert first.region == "us-east"

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = current_price_only_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = current_price_only_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.region for d in s1] == [d.region for d in s2]


# ---------------------------------------------------------------------------
# 7. Round-robin
# ---------------------------------------------------------------------------

class TestRoundRobinPolicy:
    def test_distributes_across_regions(self, price_data, carbon_data):
        jobs = _make_jobs(n=6, regions=REGIONS)
        schedule = round_robin_policy(jobs, price_data, carbon_data, CONFIG)
        regions_used = [d.region for d in schedule]
        # Should see both regions used
        assert len(set(regions_used)) > 1

    def test_deterministic(self, jobs, price_data, carbon_data):
        s1 = round_robin_policy(jobs, price_data, carbon_data, CONFIG)
        s2 = round_robin_policy(jobs, price_data, carbon_data, CONFIG)
        assert [d.region for d in s1] == [d.region for d in s2]

    def test_all_jobs_scheduled(self, jobs, price_data, carbon_data):
        schedule = round_robin_policy(jobs, price_data, carbon_data, CONFIG)
        assert len(schedule) == len(jobs)


# ---------------------------------------------------------------------------
# ALL_BASELINES registry
# ---------------------------------------------------------------------------

class TestAllBaselinesRegistry:
    def test_all_seven_present(self):
        assert len(ALL_BASELINES) == 7
        expected = {
            "fifo", "peak_blind_asap", "latency_first", "closest_region",
            "fixed_primary_region", "current_price_only", "round_robin",
        }
        assert set(ALL_BASELINES.keys()) == expected

    def test_each_policy_callable(self, jobs, price_data, carbon_data):
        for name, policy in ALL_BASELINES.items():
            schedule = policy(jobs, price_data, carbon_data, CONFIG)
            assert isinstance(schedule, list), f"{name} returned non-list"
            assert len(schedule) == len(jobs), f"{name} wrong schedule length"
