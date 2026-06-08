"""Phase 8/11.10 — realized carbon replay & per-baseline savings (no 400 fallback)."""

from datetime import datetime, timedelta, timezone

import pytest

from aurelius.backtesting.baselines import ALL_BASELINES
from aurelius.carbon.accounting import CarbonDataKind
from aurelius.carbon.replay import compare_baselines_carbon, replay_schedule_carbon
from aurelius.models import Job, OptimizationConfig, ScheduleDecision

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _jobs():
    return [
        Job(
            job_id=f"j{i}", submit_time=T0, runtime_hours=2.0,
            deadline=T0 + timedelta(hours=24), power_kw=100.0,
            earliest_start=T0, region_options=["us-west", "us-east"],
            gpu_count=8,
        )
        for i in range(3)
    ]


def _hours(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def _moer_full(regions, value_by_region, n=6):
    return {r: {h: value_by_region[r] for h in _hours(T0, n)} for r in regions}


def _place_all(jobs, region):
    return [
        ScheduleDecision(
            job_id=j.job_id, start_time=T0, region=region,
            power_fraction=1.0, actual_runtime_hours=2.0,
        )
        for j in jobs
    ]


class TestBaselineCarbonSavings:
    def test_optimized_clean_beats_dirty_baselines(self):
        jobs = _jobs()
        # us-west clean (100), us-east dirty (600). Price makes east cheaper so a
        # price-only baseline (and fifo on default us-east) pick the dirty region.
        moer = _moer_full(["us-west", "us-east"], {"us-west": 100.0, "us-east": 600.0})
        price_data = {
            "us-west": {h: 100.0 for h in _hours(T0, 6)},
            "us-east": {h: 50.0 for h in _hours(T0, 6)},
        }
        cfg = OptimizationConfig(default_region="us-east")

        # Optimized (Aurelius): everything in the clean region.
        opt_schedule = _place_all(jobs, "us-west")
        opt_result = replay_schedule_carbon(
            opt_schedule, jobs, moer, scheduler_name="aurelius",
            carbon_data_kind=CarbonDataKind.HISTORICAL,
        )

        # Baselines: fifo, current_price_only, round_robin (random), + no_migration.
        baseline_results = {}
        for name in ("fifo", "current_price_only", "round_robin"):
            sched = ALL_BASELINES[name](jobs, price_data, moer, cfg)
            baseline_results[name] = replay_schedule_carbon(
                sched, jobs, moer, scheduler_name=name,
                carbon_data_kind=CarbonDataKind.HISTORICAL,
            )
        # Explicit no-migration baseline: pin all to dirty default region.
        baseline_results["no_migration"] = replay_schedule_carbon(
            _place_all(jobs, "us-east"), jobs, moer, scheduler_name="no_migration",
            carbon_data_kind=CarbonDataKind.HISTORICAL,
        )

        cmp = compare_baselines_carbon(
            opt_result, baseline_results,
            optimized_schedule=opt_schedule,
            baseline_schedules=None, jobs=jobs,
        )

        # Optimized emissions: 3 jobs * 100kW * 2h * 100/1000 = 60 kg.
        assert opt_result.total_emissions_kgco2 == pytest.approx(60.0)
        # FIFO/price-only/no_migration land in us-east (dirty) -> 360 kg -> savings.
        for name in ("fifo", "current_price_only", "no_migration"):
            entry = cmp["per_baseline"][name]
            assert entry["carbon_savings_kgco2_vs_baseline"] > 0
            assert entry["carbon_savings_pct_vs_baseline"] > 0
            assert entry["carbon_savings_is_real_historical"] is True
        # Goodput-per-kgCO2 surfaced.
        assert cmp["per_baseline"]["fifo"]["optimized_goodput_per_kgco2"] is not None

    def test_random_and_pricewise_baselines_present(self):
        jobs = _jobs()
        moer = _moer_full(["us-west", "us-east"], {"us-west": 100.0, "us-east": 600.0})
        price_data = {
            "us-west": {h: 100.0 for h in _hours(T0, 6)},
            "us-east": {h: 50.0 for h in _hours(T0, 6)},
        }
        cfg = OptimizationConfig(default_region="us-east")
        opt = replay_schedule_carbon(_place_all(jobs, "us-west"), jobs, moer)
        baselines = {
            n: replay_schedule_carbon(ALL_BASELINES[n](jobs, price_data, moer, cfg), jobs, moer)
            for n in ("fifo", "current_price_only", "round_robin")
        }
        cmp = compare_baselines_carbon(opt, baselines)
        assert {"fifo", "current_price_only", "round_robin"} <= set(cmp["per_baseline"])


class TestNoSilentFallback:
    def test_missing_moer_excluded_and_flagged(self):
        jobs = _jobs()[:1]
        # Only hour 0 has MOER; hour 1 is missing -> coverage 50%, NOT 400-filled.
        moer = {"us-west": {T0: 300.0}}
        result = replay_schedule_carbon(
            _place_all(jobs, "us-west"), jobs, moer,
            carbon_data_kind=CarbonDataKind.HISTORICAL,
        )
        # Only the covered hour contributes: 100kW * 1h * 300/1000 = 30 kg.
        assert result.total_emissions_kgco2 == pytest.approx(30.0)
        assert result.carbon_data_coverage_pct == pytest.approx(50.0)
        # Incomplete coverage => the comparison must not call it real historical.
        baseline = replay_schedule_carbon(_place_all(jobs, "us-east"), jobs, moer)
        cmp = compare_baselines_carbon(result, {"b": baseline})
        assert cmp["per_baseline"]["b"]["carbon_savings_is_real_historical"] is False


class TestForecastNotReportedAsRealized:
    def test_forecast_kind_is_not_historical(self):
        jobs = _jobs()[:1]
        moer = {"us-west": {T0: 300.0, T0 + timedelta(hours=1): 300.0}}
        result = replay_schedule_carbon(
            _place_all(jobs, "us-west"), jobs, moer,
            carbon_data_kind=CarbonDataKind.FORECAST,
        )
        assert result.records[0].carbon_data_is_forecast is True
        assert result.records[0].carbon_data_is_historical is False
        cmp = compare_baselines_carbon(result, {})
        assert cmp["carbon_signal"] == "non_historical"
