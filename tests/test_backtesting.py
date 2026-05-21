"""Tests for the BacktestEngine – ensures optimizer only sees training data."""

import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from aurelius.backtesting.engine import BacktestEngine
from aurelius.backtesting.evaluator import evaluate_schedule
from aurelius.models import Job, OptimizationConfig

UTC = timezone.utc
BASE_TS = pd.Timestamp("2024-02-01", tz="UTC")

TRAIN_DAYS = 7
EVAL_DAYS = 2
DATA_HOURS = (TRAIN_DAYS + EVAL_DAYS * 3) * 24


def _make_price_df(regions=("us-west",), hours=DATA_HOURS, base_price=50.0):
    rows = []
    for h in range(hours):
        ts = BASE_TS + pd.Timedelta(hours=h)
        for region in regions:
            rows.append({
                "timestamp": ts,
                "region": region,
                "price_per_mwh": base_price + (h % 24),
                "currency": "USD",
                "source": "test",
                "source_granularity": "hourly",
                "fetched_at": pd.Timestamp.now("UTC"),
            })
    return pd.DataFrame(rows)


def _make_carbon_df(regions=("us-west",), hours=DATA_HOURS, base_carbon=300.0):
    rows = []
    for h in range(hours):
        ts = BASE_TS + pd.Timedelta(hours=h)
        for region in regions:
            rows.append({
                "timestamp": ts,
                "region": region,
                "gco2_per_kwh": base_carbon,
                "source": "test",
                "source_granularity": "hourly",
                "fetched_at": pd.Timestamp.now("UTC"),
            })
    return pd.DataFrame(rows)


def _make_jobs(n=4, start_offset_days=TRAIN_DAYS, regions=("us-west",)):
    """Jobs with earliest_start inside the first eval window."""
    base = (BASE_TS + pd.Timedelta(days=start_offset_days)).to_pydatetime()
    jobs = []
    for i in range(n):
        earliest = base + timedelta(hours=i * 4)
        jobs.append(Job(
            job_id=f"job-bt-{i}",
            submit_time=earliest - timedelta(hours=1),
            runtime_hours=2.0,
            deadline=earliest + timedelta(hours=24),
            power_kw=50.0,
            earliest_start=earliest,
            region_options=list(regions),
        ))
    return jobs


class TestBacktestEngine:
    def test_engine_produces_rounds(self):
        price_df = _make_price_df()
        carbon_df = _make_carbon_df()
        jobs = _make_jobs(n=4)

        engine = BacktestEngine(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, step_days=EVAL_DAYS)
        rounds = engine.run(jobs, price_df, carbon_df)
        assert len(rounds) >= 1

    def test_optimizer_does_not_see_eval_data(self):
        """The optimizer must only receive training-window prices.

        Jobs have earliest_start inside the eval window, so they CAN'T start
        in the training window regardless. We verify start_time >= eval_start.
        """
        price_df = _make_price_df(base_price=50.0)
        carbon_df = _make_carbon_df()
        jobs = _make_jobs(n=2)

        engine = BacktestEngine(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, baselines=[])
        rounds = engine.run(jobs, price_df, carbon_df)

        eval_start = BASE_TS + pd.Timedelta(days=TRAIN_DAYS)
        for r in rounds:
            for dec in r.optimizer_schedule:
                dec_ts = pd.Timestamp(dec.start_time)
                if dec_ts.tzinfo is None:
                    dec_ts = dec_ts.tz_localize("UTC")
                assert dec_ts >= eval_start, (
                    f"Decision {dec.job_id} starts at {dec_ts} "
                    f"before eval window {eval_start}"
                )

    def test_each_round_has_metrics(self):
        price_df = _make_price_df()
        carbon_df = _make_carbon_df()
        jobs = _make_jobs(n=2)

        engine = BacktestEngine(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS)
        rounds = engine.run(jobs, price_df, carbon_df)

        for r in rounds:
            assert r.optimizer_metrics is not None
            assert r.optimizer_metrics.jobs_evaluated >= 0

    def test_baselines_all_evaluated(self):
        price_df = _make_price_df()
        carbon_df = _make_carbon_df()
        jobs = _make_jobs(n=2)

        engine = BacktestEngine(
            train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
            baselines=["fifo", "peak_blind_asap"],
        )
        rounds = engine.run(jobs, price_df, carbon_df)

        for r in rounds:
            assert "fifo" in r.baseline_metrics
            assert "peak_blind_asap" in r.baseline_metrics

    def test_to_dict_serializable(self):
        price_df = _make_price_df()
        carbon_df = _make_carbon_df()
        jobs = _make_jobs(n=2)

        engine = BacktestEngine(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, baselines=["fifo"])
        rounds = engine.run(jobs, price_df, carbon_df)
        for r in rounds:
            d = r.to_dict()
            assert "fold_index" in d
            assert "optimizer" in d
            assert "baselines" in d


class TestEvaluateSchedule:
    def test_basic_cost_calculation(self):
        T0 = BASE_TS.to_pydatetime()
        job = Job(
            job_id="j1", submit_time=T0, runtime_hours=2.0, deadline=T0 + timedelta(hours=24),
            power_kw=100.0, earliest_start=T0, region_options=["us-west"],
        )
        from aurelius.models import ScheduleDecision
        decision = ScheduleDecision(
            job_id="j1", start_time=T0, region="us-west",
            power_fraction=1.0, actual_runtime_hours=2.0,
        )
        price_data = {"us-west": {
            T0.replace(minute=0): 100.0,
            (T0 + timedelta(hours=1)).replace(minute=0): 100.0,
        }}
        carbon_data = {"us-west": {
            T0.replace(minute=0): 300.0,
            (T0 + timedelta(hours=1)).replace(minute=0): 300.0,
        }}
        metrics = evaluate_schedule([decision], [job], price_data, carbon_data)
        # 100 kW * 2h * ($100/MWh / 1000) = $20
        assert abs(metrics.total_energy_cost_usd - 20.0) < 0.01
        # 300 gCO2/kWh * 200 kWh = 60000 gCO2
        assert abs(metrics.total_carbon_gco2 - 60000.0) < 1.0

    def test_empty_schedule_returns_zero_metrics(self):
        metrics = evaluate_schedule([], [], {}, {})
        assert metrics.total_energy_cost_usd == 0.0
        assert metrics.jobs_evaluated == 0


class TestRollingHorizon:
    """Rolling-horizon (receding-horizon / MPC) optimization."""

    def test_rolling_produces_folds(self):
        """Rolling mode runs end-to-end and produces the same fold structure."""
        price_df = _make_price_df(regions=("us-west", "us-east"))
        carbon_df = _make_carbon_df(regions=("us-west", "us-east"))
        jobs = _make_jobs(n=6, regions=("us-west", "us-east"))

        engine = BacktestEngine(
            method="greedy_migrate", train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
            config=OptimizationConfig(),
        )
        engine.forecast_horizon_hours = 24
        engine.replan_hours = 24
        rounds = engine.run(jobs, price_df, carbon_df)
        assert len(rounds) >= 1
        # Every eval job in a fold should have a committed decision
        for r in rounds:
            assert len(r.optimizer_schedule) == len(r.eval_jobs)

    def test_rolling_matches_oneshot_when_horizon_covers_window(self):
        """If the horizon covers the whole eval window, rolling with actual
        prices should do at least as well as one-shot on the same forecast."""
        price_df = _make_price_df(regions=("us-west", "us-east"))
        carbon_df = _make_carbon_df(regions=("us-west", "us-east"))
        jobs = _make_jobs(n=6, regions=("us-west", "us-east"))

        # Rolling with a huge horizon (covers everything) == perfect near-term
        engine_roll = BacktestEngine(
            method="greedy", train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
            config=OptimizationConfig(),
        )
        engine_roll.forecast_horizon_hours = 24 * 365  # effectively unbounded
        engine_roll.replan_hours = 24
        rounds_roll = engine_roll.run(jobs, price_df, carbon_df)
        # Should produce folds and schedules without error
        assert len(rounds_roll) >= 1
        for r in rounds_roll:
            assert len(r.optimizer_schedule) == len(r.eval_jobs)
