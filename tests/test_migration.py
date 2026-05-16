"""Tests for mid-job region migration support.

Covers:
- ScheduleSegment / ScheduleDecision.all_segments / migration_count
- Evaluator scores multi-segment schedules and accounts for migration overhead
- JobScheduler --method greedy_migrate produces multi-segment schedules
  when the price gradient justifies the migration cost
- Non-migratable jobs (migration_cost_hours=None) are not segmented
- Migration is rejected when deadline doesn't fit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aurelius.models import (
    Job,
    OptimizationConfig,
    ScheduleDecision,
    ScheduleSegment,
)
from aurelius.optimization.scheduler import JobScheduler
from aurelius.backtesting.evaluator import evaluate_schedule


# Anchor times — far enough apart to allow long jobs
START = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _hourly_prices(region: str, hours: int, base: float = 50.0, peak_hour: int = 12,
                   peak_amplitude: float = 50.0, offset_hours: int = 0) -> dict:
    """Build a hourly price dict with a daily cycle, optionally phase-shifted."""
    out = {}
    for h in range(hours):
        ts = START + timedelta(hours=h)
        # Simple peak around peak_hour each day
        hour_of_day = (h + offset_hours) % 24
        # Cosine bump centered at peak_hour
        import math
        amp = peak_amplitude * math.cos(2 * math.pi * (hour_of_day - peak_hour) / 24)
        out[ts] = base + amp
    return out


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TestScheduleDecisionSegments:
    def test_single_segment_back_compat(self):
        """Old-style ScheduleDecision (no segments) yields a 1-element list."""
        d = ScheduleDecision(
            job_id="j1",
            start_time=START,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=10.0,
        )
        assert d.segments is None
        assert d.migration_count == 0
        segs = d.all_segments
        assert len(segs) == 1
        assert segs[0].region == "us-west"
        assert segs[0].start_time == START
        assert segs[0].end_time == START + timedelta(hours=10)

    def test_multi_segment_migration_count(self):
        d = ScheduleDecision(
            job_id="j1",
            start_time=START,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=20.5,  # 10h + 0.5h migration + 10h
            segments=[
                ScheduleSegment(START, START + timedelta(hours=10), "us-west"),
                ScheduleSegment(
                    START + timedelta(hours=10),
                    START + timedelta(hours=20.5),
                    "us-east",
                ),
            ],
        )
        assert d.migration_count == 1
        assert d.end_time == START + timedelta(hours=20.5)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluatorMultiSegment:
    def test_single_segment_unchanged(self):
        """Single-segment decision scores identically to old behavior."""
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=4.0,
            deadline=START + timedelta(hours=24), power_kw=100.0,
            earliest_start=START, region_options=["us-west"],
        )
        # Flat $100/MWh in us-west
        prices = {"us-west": {START + timedelta(hours=h): 100.0 for h in range(10)}}
        d = ScheduleDecision(
            job_id="j1", start_time=START, region="us-west",
            power_fraction=1.0, actual_runtime_hours=4.0,
        )
        m = evaluate_schedule([d], [job], prices, {}, warn_on_missing=False)
        # 100 kW * 4 h * $100/MWh / 1000 = $40
        assert m.total_energy_cost_usd == pytest.approx(40.0, rel=1e-6)
        assert m.missing_price_hours == 0

    def test_two_segment_split_cost(self):
        """Multi-segment decision sums per-segment costs at the right region."""
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=6.0,
            deadline=START + timedelta(hours=24), power_kw=100.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            migration_cost_hours=0.5,
        )
        # us-west $200/MWh, us-east $50/MWh (so migrating to us-east mid-job saves money)
        prices = {
            "us-west": {START + timedelta(hours=h): 200.0 for h in range(10)},
            "us-east": {START + timedelta(hours=h): 50.0 for h in range(10)},
        }
        # 3h in us-west, then 0.5h migration + 3h useful in us-east = 3.5h in us-east
        seg1 = ScheduleSegment(START, START + timedelta(hours=3), "us-west")
        seg2 = ScheduleSegment(
            START + timedelta(hours=3),
            START + timedelta(hours=6.5),
            "us-east",
        )
        d = ScheduleDecision(
            job_id="j1", start_time=START, region="us-west",
            power_fraction=1.0, actual_runtime_hours=6.5,
            segments=[seg1, seg2],
        )
        m = evaluate_schedule([d], [job], prices, {}, warn_on_missing=False)
        # Seg1: 100kW * 3h * $200/MWh / 1000 = $60
        # Seg2: 100kW * 3.5h * $50/MWh / 1000 = $17.50
        # Total: $77.50
        assert m.total_energy_cost_usd == pytest.approx(77.50, rel=1e-6)


# ---------------------------------------------------------------------------
# Scheduler — greedy_migrate end-to-end
# ---------------------------------------------------------------------------

class TestGreedyMigrate:
    def test_migration_taken_when_profitable(self):
        """A long job with a big inter-region price spread should get a migration."""
        # us-west cheap for 24h then expensive for 24h
        # us-east opposite phase: expensive for 24h then cheap for 24h
        prices = {
            "us-west": {START + timedelta(hours=h): (30.0 if h < 24 else 200.0)
                        for h in range(48)},
            "us-east": {START + timedelta(hours=h): (200.0 if h < 24 else 30.0)
                        for h in range(48)},
        }
        job = Job(
            job_id="j1",
            submit_time=START,
            runtime_hours=40.0,  # spans both regimes
            deadline=START + timedelta(hours=48),
            power_kw=500.0,
            earliest_start=START,
            region_options=["us-west", "us-east"],
            workload_type="training",
            migration_cost_hours=0.5,
        )

        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate")
        assert len(result.schedule) == 1
        decision = result.schedule[0]
        # Greedy starts in us-west (cheap at t=0), then should migrate to us-east
        # around hour 24 (where us-east becomes much cheaper than us-west)
        assert decision.migration_count == 1, (
            f"Expected 1 migration with a 170-point inter-region spread, "
            f"got migration_count={decision.migration_count}, segments={decision.segments}"
        )
        # First segment should be us-west, second us-east
        segs = decision.all_segments
        assert segs[0].region == "us-west"
        assert segs[1].region == "us-east"

    def test_no_migration_when_not_profitable(self):
        """If both regions have the same flat price, migration is not worthwhile."""
        prices = {
            "us-west": {START + timedelta(hours=h): 100.0 for h in range(48)},
            "us-east": {START + timedelta(hours=h): 100.0 for h in range(48)},
        }
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=24.0,
            deadline=START + timedelta(hours=48), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            migration_cost_hours=0.5,
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate")
        # Flat prices → no profitable migration
        assert result.schedule[0].migration_count == 0

    def test_non_migratable_job_stays_single_segment(self):
        """Jobs with migration_cost_hours=None never migrate, even if profitable."""
        prices = {
            "us-west": {START + timedelta(hours=h): (30.0 if h < 12 else 500.0)
                        for h in range(24)},
            "us-east": {START + timedelta(hours=h): (500.0 if h < 12 else 30.0)
                        for h in range(24)},
        }
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=20.0,
            deadline=START + timedelta(hours=24), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            workload_type="realtime_inference",
            migration_cost_hours=None,  # cannot migrate
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate")
        assert result.schedule[0].migration_count == 0

    def test_migration_rejected_when_deadline_too_tight(self):
        """If migration cost would push end past deadline, leave single-segment."""
        prices = {
            "us-west": {START + timedelta(hours=h): (30.0 if h < 5 else 500.0)
                        for h in range(10)},
            "us-east": {START + timedelta(hours=h): (500.0 if h < 5 else 30.0)
                        for h in range(10)},
        }
        # 8h runtime, 8h deadline → no room for 0.5h migration overhead
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=8.0,
            deadline=START + timedelta(hours=8), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            migration_cost_hours=0.5,
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate")
        assert result.schedule[0].migration_count == 0


# ---------------------------------------------------------------------------
# Multi-migration DP — greedy_migrate_dp
# ---------------------------------------------------------------------------

class TestMultiMigrationDP:
    def test_dp_finds_multiple_migrations_on_multi_cycle_job(self):
        """A 4-day job on a daily price flip should take ~3 migrations."""
        # 96-hour window. us-west cheap odd days, us-east cheap even days.
        # Flip every 24h.
        prices_w, prices_e = {}, {}
        for h in range(120):  # plenty of buffer
            ts = START + timedelta(hours=h)
            day = h // 24
            if day % 2 == 0:
                prices_w[ts] = 20.0
                prices_e[ts] = 200.0
            else:
                prices_w[ts] = 200.0
                prices_e[ts] = 20.0
        prices = {"us-west": prices_w, "us-east": prices_e}

        # 96-hour job (4 days). Should migrate ~3 times to always be in cheap region.
        job = Job(
            job_id="j1",
            submit_time=START,
            runtime_hours=96.0,
            deadline=START + timedelta(hours=120),
            power_kw=500.0,
            earliest_start=START,
            region_options=["us-west", "us-east"],
            workload_type="training",
            migration_cost_hours=0.5,
        )

        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate_dp")
        decision = result.schedule[0]
        # Optimal is start in us-west (cheap day 0), migrate to us-east (day 1 cheap),
        # back to us-west (day 2 cheap), to us-east (day 3 cheap) — 3 migrations.
        # With 0.5h migration cost, total wallclock = 96 + 3*0.5 = 97.5h, fits in 120h budget.
        assert decision.migration_count >= 2, (
            f"DP should chase daily cycles with multiple migrations, "
            f"got migration_count={decision.migration_count}"
        )

    def test_dp_strictly_better_than_single_on_multi_cycle(self):
        """On a multi-cycle setup, DP should produce strictly lower forecast cost
        than single-migration."""
        # 3-day window with daily flips
        prices_w, prices_e = {}, {}
        for h in range(96):
            ts = START + timedelta(hours=h)
            day = h // 24
            if day % 2 == 0:
                prices_w[ts] = 30.0
                prices_e[ts] = 180.0
            else:
                prices_w[ts] = 180.0
                prices_e[ts] = 30.0
        prices = {"us-west": prices_w, "us-east": prices_e}

        job = Job(
            job_id="j1", submit_time=START, runtime_hours=72.0,
            deadline=START + timedelta(hours=96), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            workload_type="training", migration_cost_hours=0.5,
        )
        sched = JobScheduler(OptimizationConfig())

        single = sched.solve([job], prices, {}, method="greedy_migrate")
        dp = sched.solve([job], prices, {}, method="greedy_migrate_dp")

        # Score both with the evaluator (forecast == actual in this test)
        single_metrics = evaluate_schedule(
            single.schedule, [job], prices, {}, warn_on_missing=False,
        )
        dp_metrics = evaluate_schedule(
            dp.schedule, [job], prices, {}, warn_on_missing=False,
        )

        assert dp.schedule[0].migration_count >= single.schedule[0].migration_count
        # DP must be at least as good as single (it's strictly more expressive)
        assert dp_metrics.total_energy_cost_usd <= single_metrics.total_energy_cost_usd + 1e-6
        # In this multi-cycle scenario it should be strictly better
        if dp.schedule[0].migration_count > single.schedule[0].migration_count:
            assert dp_metrics.total_energy_cost_usd < single_metrics.total_energy_cost_usd

    def test_dp_no_migration_when_flat(self):
        """DP should produce 0 migrations when prices are flat (matches single)."""
        prices = {
            "us-west": {START + timedelta(hours=h): 100.0 for h in range(48)},
            "us-east": {START + timedelta(hours=h): 100.0 for h in range(48)},
        }
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=24.0,
            deadline=START + timedelta(hours=48), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            migration_cost_hours=0.5,
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate_dp")
        assert result.schedule[0].migration_count == 0

    def test_dp_respects_migration_cap(self):
        """When K_max is bounded by deadline, DP cannot exceed it."""
        # 24-hour job with 24.5h deadline → only ~1 migration fits (0.5h overhead)
        prices_w, prices_e = {}, {}
        # Hourly flip: forces DP to want lots of migrations
        for h in range(48):
            ts = START + timedelta(hours=h)
            prices_w[ts] = 20.0 if h % 2 == 0 else 200.0
            prices_e[ts] = 200.0 if h % 2 == 0 else 20.0
        prices = {"us-west": prices_w, "us-east": prices_e}

        job = Job(
            job_id="j1", submit_time=START, runtime_hours=24.0,
            deadline=START + timedelta(hours=25),  # only 1h slack
            power_kw=500.0, earliest_start=START,
            region_options=["us-west", "us-east"],
            migration_cost_hours=0.5,
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate_dp")
        # Deadline gives (25 - 24) / 0.5 = 2 migrations max
        assert result.schedule[0].migration_count <= 2

    def test_dp_non_migratable_unchanged(self):
        """Non-migratable jobs (realtime) get no migration even under DP."""
        prices = {
            "us-west": {START + timedelta(hours=h): (10.0 if h % 2 == 0 else 1000.0)
                        for h in range(48)},
            "us-east": {START + timedelta(hours=h): (1000.0 if h % 2 == 0 else 10.0)
                        for h in range(48)},
        }
        job = Job(
            job_id="j1", submit_time=START, runtime_hours=24.0,
            deadline=START + timedelta(hours=48), power_kw=500.0,
            earliest_start=START, region_options=["us-west", "us-east"],
            workload_type="realtime_inference", migration_cost_hours=None,
        )
        sched = JobScheduler(OptimizationConfig())
        result = sched.solve([job], prices, {}, method="greedy_migrate_dp")
        assert result.schedule[0].migration_count == 0
