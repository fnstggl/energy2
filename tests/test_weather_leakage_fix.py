"""Tests for the Phase-1 weather leakage fix in the backtest engine.

Verifies that:
  * the leakage-assertion helper detects perfect-foresight (observed) eval weather,
  * it passes when eval weather is sourced from a day-ahead forecast,
  * BacktestEngine accepts forecast_weather_df and uses it leakage-free for the
    eval window while training still uses observed history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurelius.backtesting.engine import (
    BacktestEngine,
    WeatherLeakageError,
    _assert_no_observed_weather_in_eval,
)
from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster

REGIONS = ["us-west", "us-east", "us-south"]
EVAL_START = pd.Timestamp("2026-01-20", tz="UTC")


def _weather(start, hours, regions, temp_fn):
    rows = []
    for r in regions:
        for i in range(hours):
            ts = start + pd.Timedelta(hours=i)
            t = temp_fn(r, i)
            tf = t * 9 / 5 + 32
            rows.append({
                "timestamp": ts, "region": r, "temperature_c": t,
                "humidity_pct": 50.0, "wind_speed_ms": 3.0,
                "hdd_f": max(0.0, 65 - tf), "cdd_f": max(0.0, tf - 65),
                "temp_rolling_24h_c": t, "temp_delta_24h_c": 0.0, "source": "test",
            })
    return pd.DataFrame(rows)


def test_assertion_detects_observed_eval_weather():
    start = EVAL_START - pd.Timedelta(hours=48)
    obs = _weather(start, 96, REGIONS, lambda r, i: 10.0 + i * 0.1)
    # predict frame == observed everywhere → perfect-foresight leakage
    with pytest.raises(WeatherLeakageError):
        _assert_no_observed_weather_in_eval(obs.copy(), obs, EVAL_START)


def test_assertion_passes_for_forecast_eval_weather():
    start = EVAL_START - pd.Timedelta(hours=48)
    obs = _weather(start, 96, REGIONS, lambda r, i: 10.0 + i * 0.1)
    fc = _weather(start, 96, REGIONS, lambda r, i: 10.0 + i * 0.1 + 1.3)  # forecast error
    predict = pd.concat(
        [obs[pd.to_datetime(obs.timestamp, utc=True) < EVAL_START],
         fc[pd.to_datetime(fc.timestamp, utc=True) >= EVAL_START]],
        ignore_index=True,
    )
    _assert_no_observed_weather_in_eval(predict, obs, EVAL_START)  # must not raise


def test_assertion_noop_when_no_eval_rows():
    start = EVAL_START - pd.Timedelta(hours=48)
    obs = _weather(start, 24, REGIONS, lambda r, i: 5.0)  # all before eval_start
    _assert_no_observed_weather_in_eval(obs, obs, EVAL_START)  # no eval rows → no-op


def _prices(start, hours, regions, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in regions:
        base = {"us-west": 30, "us-east": 45, "us-south": 35}[r]
        for i in range(hours):
            ts = start + pd.Timedelta(hours=i)
            rows.append({"timestamp": ts, "region": r,
                         "price_per_mwh": float(base + 10 * np.sin(i / 12) + rng.normal(0, 3))})
    return pd.DataFrame(rows)


def test_engine_accepts_forecast_weather_df_without_leakage():
    start = pd.Timestamp("2025-12-01", tz="UTC")
    hours = 24 * 60
    price = _prices(start, hours, REGIONS)
    obs = _weather(start, hours, REGIONS, lambda r, i: 8.0 + 5 * np.sin(i / 24))
    fc = _weather(start, hours, REGIONS, lambda r, i: 8.0 + 5 * np.sin(i / 24) + 1.0)
    carbon = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])

    from aurelius.ingestion.job_logs import JobLogIngester
    jobs = JobLogIngester().generate_synthetic(
        start_time=start.to_pydatetime(), duration_hours=hours,
        num_jobs=20, regions=REGIONS, seed=1,
        workload_mix="realistic", workload_filter="training")

    engine = BacktestEngine(
        method="greedy_migrate", train_days=30, eval_days=7,
        price_forecaster_cls=PriceQuantileForecaster,
        price_forecaster_config=PriceModelConfig(seed=42, n_estimators=50,
                                                 include_weather_features=True),
        context_hours=192,
        weather_df=obs,
        forecast_weather_df=fc,
    )
    # Must run without raising WeatherLeakageError (eval weather = forecast).
    rounds = engine.run(jobs, price, carbon)
    assert len(rounds) > 0
    for r in rounds:
        assert r.forecast_quality is not None
