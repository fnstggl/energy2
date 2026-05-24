"""Phase-3 test: the live shadow runner consumes weather leakage-free.

Uses a recording fake forecaster to verify that LiveShadowRunner:
  * passes only PRE-decision-time observed weather to fit(),
  * passes the day-ahead FORECAST weather to predict(),
  * never passes future observed weather (no perfect-foresight leakage).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from aurelius.shadow.runner import LiveShadowRunner
from aurelius.ingestion.job_logs import JobLogIngester

REGIONS = ["us-west", "us-east", "us-south"]
DECISION = datetime(2026, 1, 20, tzinfo=timezone.utc)


class _RecordingForecaster:
    """Records weather passed to fit/predict; returns flat forecasts."""
    last = {}

    def __init__(self, config=None):
        pass

    def fit(self, prices, weather_df=None):
        _RecordingForecaster.last["fit_weather"] = weather_df
        return self

    def predict(self, region, timestamps, recent_prices=None, weather_df=None):
        _RecordingForecaster.last.setdefault("predict_weather", {})[region] = weather_df

        class _F:
            def __init__(self, ts, region):
                self.timestamp = ts
                self.region = region
                self.p50 = 50.0
                self.p90 = 60.0
        return [_F(ts, region) for ts in timestamps]


def _df(value_fn, start, hours, cols):
    rows = []
    for r in REGIONS:
        for i in range(hours):
            ts = start + timedelta(hours=i)
            row = {"timestamp": ts, "region": r}
            row.update(value_fn(r, i))
            rows.append(row)
    return pd.DataFrame(rows)


def _weather(start, hours, bias=0.0):
    def vf(r, i):
        t = 8.0 + bias
        tf = t * 9 / 5 + 32
        return {"temperature_c": t, "humidity_pct": 50.0, "wind_speed_ms": 3.0,
                "hdd_f": max(0.0, 65 - tf), "cdd_f": max(0.0, tf - 65),
                "temp_rolling_24h_c": t, "temp_delta_24h_c": 0.0, "source": "test"}
    return _df(vf, start, hours, None)


def test_shadow_runner_threads_weather_leakage_free():
    start = DECISION - timedelta(hours=24 * 40)
    total_hours = 24 * 48
    price = _df(lambda r, i: {"price_per_mwh": 40.0 + (i % 24)}, start, total_hours, None)
    obs = _weather(start, total_hours, bias=0.0)            # observed (incl. future)
    fc = _weather(start, total_hours, bias=1.5)             # day-ahead forecast (offset)

    jobs = JobLogIngester().generate_synthetic(
        start_time=DECISION.replace(tzinfo=timezone.utc), duration_hours=200,
        num_jobs=6, regions=REGIONS, seed=1, workload_mix="realistic",
        workload_filter="training")

    runner = LiveShadowRunner(
        regions=REGIONS, train_days=30, horizon_hours=48,
        price_forecaster_cls=_RecordingForecaster,
        weather_df=obs, forecast_weather_df=fc,
        enable_safety_gate=False,
    )
    runner.run(price_df=price, jobs=jobs, decision_time=DECISION)

    fit_w = _RecordingForecaster.last["fit_weather"]
    assert fit_w is not None and not fit_w.empty
    # training weather must be strictly before decision_time (no leakage)
    assert (pd.to_datetime(fit_w["timestamp"], utc=True) < pd.Timestamp(DECISION)).all()

    # predict weather must be the FORECAST frame (bias=1.5 → temp 9.5, not 8.0)
    pw = _RecordingForecaster.last["predict_weather"]["us-south"]
    assert pw is not None
    assert abs(float(pw["temperature_c"].iloc[0]) - 9.5) < 1e-6
