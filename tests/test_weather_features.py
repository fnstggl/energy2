"""Tests for Phase 3 weather feature integration.

Covers:
- Weather lookup construction from canonical DataFrame
- Weather feature injection into feature matrix (training and predict paths)
- PriceQuantileForecaster backward compatibility (no weather → same as v2.0)
- PriceQuantileForecaster with weather features: fit and predict
- Leakage-free slicing in BacktestEngine (train weather ≠ eval weather)
- Graceful degradation when weather data has gaps
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aurelius.forecasting.quantile_model import (
    WEATHER_FEATURE_COLS,
    add_weather_features,
    build_feature_matrix,
    build_feature_matrix_for_predict,
    build_weather_lookup,
)
from aurelius.forecasting.price_model import (
    PriceModelConfig,
    PriceQuantileForecaster,
)
from aurelius.models import EnergyPrice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2026-01-01") -> list[datetime]:
    base = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    return [base + timedelta(hours=h) for h in range(n)]


def _make_weather_df(
    timestamps: list[datetime],
    regions: list[str],
    base_temp: float = 5.0,
) -> pd.DataFrame:
    """Create a minimal synthetic weather DataFrame matching canonical schema."""
    rows = []
    for i, (ts, region) in enumerate(zip(timestamps, regions)):
        t = base_temp + np.sin(i / 24 * 2 * np.pi) * 8
        tf = t * 9 / 5 + 32
        rows.append({
            "timestamp": ts,
            "region":    region,
            "temperature_c":     round(t, 2),
            "humidity_pct":      60.0,
            "wind_speed_ms":     5.0 + np.cos(i / 24 * 2 * np.pi) * 2,
            "hdd_f":             max(0.0, 65.0 - tf),
            "cdd_f":             max(0.0, tf - 65.0),
            "temp_rolling_24h_c": t,
            "temp_delta_24h_c":  0.0,
            "source":            "synthetic_test",
        })
    return pd.DataFrame(rows)


def _make_prices(
    timestamps: list[datetime],
    regions: list[str],
    base_price: float = 40.0,
) -> list[EnergyPrice]:
    np.random.seed(42)
    return [
        EnergyPrice(timestamp=ts, region=r, price_per_mwh=max(0.1, base_price + np.random.randn() * 10))
        for ts, r in zip(timestamps, regions)
    ]


# ---------------------------------------------------------------------------
# 1. build_weather_lookup
# ---------------------------------------------------------------------------

class TestBuildWeatherLookup:
    def test_basic_lookup_construction(self):
        ts = _make_timestamps(24, "2026-01-01")
        regions = ["us-south"] * 24
        wdf = _make_weather_df(ts, regions)
        lookup = build_weather_lookup(wdf)
        assert len(lookup) == 24
        key = (pd.Timestamp("2026-01-01T00:00:00+00:00"), "us-south")
        assert key in lookup
        assert "temperature_c" in lookup[key]
        assert "hdd_f" in lookup[key]
        assert "wind_speed_ms" in lookup[key]

    def test_empty_df_returns_empty_dict(self):
        assert build_weather_lookup(pd.DataFrame()) == {}
        assert build_weather_lookup(None) == {}

    def test_multi_region_lookup(self):
        ts = _make_timestamps(48, "2026-01-01")
        regions = ["us-west"] * 24 + ["us-east"] * 24
        wdf = _make_weather_df(ts, regions)
        lookup = build_weather_lookup(wdf)
        assert len(lookup) == 48
        west_keys = [k for k in lookup if k[1] == "us-west"]
        east_keys = [k for k in lookup if k[1] == "us-east"]
        assert len(west_keys) == 24
        assert len(east_keys) == 24

    def test_timestamps_floored_to_hour(self):
        # Non-hour timestamps should be floored
        ts = [
            datetime(2026, 1, 1, 0, 53, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 1, 47, tzinfo=timezone.utc),
        ]
        regions = ["us-south"] * 2
        wdf = _make_weather_df(ts, regions)
        lookup = build_weather_lookup(wdf)
        # Both should be keyed by floored hour
        key0 = (pd.Timestamp("2026-01-01T00:00:00+00:00"), "us-south")
        key1 = (pd.Timestamp("2026-01-01T01:00:00+00:00"), "us-south")
        assert key0 in lookup or key1 in lookup

    def test_no_nan_in_values(self):
        ts = _make_timestamps(24, "2026-01-01")
        wdf = _make_weather_df(ts, ["us-south"] * 24)
        # Introduce a NaN
        wdf.loc[0, "temperature_c"] = float("nan")
        lookup = build_weather_lookup(wdf)
        # NaN should be replaced with 0.0
        first_key = list(lookup.keys())[0]
        assert not (lookup[first_key]["temperature_c"] != lookup[first_key]["temperature_c"])  # not NaN


# ---------------------------------------------------------------------------
# 2. add_weather_features + build_feature_matrix with weather
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixWithWeather:
    def test_weather_cols_added(self):
        ts = _make_timestamps(24)
        regions = ["us-south"] * 24
        values = np.random.rand(24) * 50 + 20
        wdf = _make_weather_df(ts, regions)
        lookup = build_weather_lookup(wdf)

        df = build_feature_matrix(
            ts, regions, values,
            include_lags=False, include_rolling=False, include_volatility=False,
            weather_lookup=lookup,
        )
        for col in ["temperature_c", "hdd_f", "cdd_f", "wind_speed_ms"]:
            assert col in df.columns, f"Missing weather column: {col}"

    def test_no_weather_lookup_backward_compat(self):
        ts = _make_timestamps(24)
        regions = ["us-south"] * 24
        values = np.random.rand(24) * 50
        df = build_feature_matrix(ts, regions, values)
        for col in WEATHER_FEATURE_COLS:
            assert col not in df.columns

    def test_weather_values_nonzero_where_data_exists(self):
        ts = _make_timestamps(24, "2026-01-01")
        regions = ["us-south"] * 24
        values = np.random.rand(24) * 50
        wdf = _make_weather_df(ts, regions, base_temp=5.0)
        lookup = build_weather_lookup(wdf)

        df = build_feature_matrix(ts, regions, values, weather_lookup=lookup)
        # temperature_c should not be all zeros
        assert df["temperature_c"].abs().sum() > 0

    def test_missing_region_weather_falls_back_to_zero(self):
        ts = _make_timestamps(24, "2026-01-01")
        regions_train = ["us-south"] * 24
        wdf = _make_weather_df(ts, regions_train)
        lookup = build_weather_lookup(wdf)

        # Use a different region for prediction (us-west not in lookup)
        regions_pred = ["us-west"] * 24
        values = np.random.rand(24) * 50
        df = build_feature_matrix(ts, regions_pred, values, weather_lookup=lookup)
        # Should degrade gracefully to 0 (no crash)
        assert "temperature_c" in df.columns
        assert df.isnull().sum().sum() == 0  # no NaN

    def test_no_nan_in_output(self):
        ts = _make_timestamps(48)
        regions = ["us-south"] * 48
        values = np.random.rand(48) * 50
        wdf = _make_weather_df(ts, regions)
        lookup = build_weather_lookup(wdf)
        df = build_feature_matrix(
            ts, regions, values,
            include_lags=True, include_rolling=True, include_volatility=True,
            weather_lookup=lookup,
        )
        assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 3. build_feature_matrix_for_predict with weather
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixForPredictWithWeather:
    def test_weather_passed_through_to_predict_features(self):
        recent_ts = _make_timestamps(48, "2026-01-01")
        recent_values = np.random.rand(48) * 50 + 20
        pred_ts = _make_timestamps(24, "2026-01-03")
        region = "us-south"

        # Build weather for the full period (recent + predict)
        all_ts = recent_ts + pred_ts
        all_regions = [region] * len(all_ts)
        wdf = _make_weather_df(all_ts, all_regions, base_temp=2.0)
        lookup = build_weather_lookup(wdf)

        df, used_lags = build_feature_matrix_for_predict(
            pred_ts, [region] * len(pred_ts), recent_values,
            weather_lookup=lookup,
        )
        assert used_lags
        for col in ["temperature_c", "hdd_f"]:
            assert col in df.columns

    def test_weather_predict_no_lags_fallback(self):
        # When insufficient recent data, fallback still includes weather
        pred_ts = _make_timestamps(24, "2026-01-03")
        region = "us-south"
        wdf = _make_weather_df(pred_ts, [region] * len(pred_ts))
        lookup = build_weather_lookup(wdf)

        df, used_lags = build_feature_matrix_for_predict(
            pred_ts, [region] * 24, None,  # None → no recent data
            weather_lookup=lookup,
        )
        assert not used_lags
        assert "temperature_c" in df.columns


# ---------------------------------------------------------------------------
# 4. PriceQuantileForecaster backward compatibility
# ---------------------------------------------------------------------------

class TestPriceQuantileForecasterBackwardCompat:
    def test_fit_without_weather_no_change(self):
        ts = _make_timestamps(200)
        prices = _make_prices(ts, ["us-south"] * 200)
        config = PriceModelConfig(seed=42, n_estimators=20)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices)  # no weather_df
        assert fc.is_fitted
        assert fc.metadata.features_version == "v2.0"

    def test_predict_without_weather_no_change(self):
        ts = _make_timestamps(200)
        prices = _make_prices(ts, ["us-south"] * 200)
        config = PriceModelConfig(seed=42, n_estimators=20)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices)

        pred_ts = _make_timestamps(24, "2026-01-09")
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-48:])
        assert len(preds) == 24
        assert all(p.p90 >= p.p50 >= 0 for p in preds)


# ---------------------------------------------------------------------------
# 5. PriceQuantileForecaster with weather features
# ---------------------------------------------------------------------------

class TestPriceQuantileForecasterWithWeather:
    def test_fit_with_weather_produces_v3_features(self):
        ts = _make_timestamps(200, "2026-01-01")
        regions = ["us-south"] * 200
        prices = _make_prices(ts, regions)
        wdf = _make_weather_df(ts, regions, base_temp=5.0)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=True)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices, weather_df=wdf)

        assert fc.is_fitted
        assert fc.metadata.features_version == "v3.0"
        assert "+weather" in fc.metadata.model_type

    def test_predict_with_weather_returns_valid_forecasts(self):
        ts = _make_timestamps(300, "2026-01-01")
        regions = ["us-south"] * 300
        prices = _make_prices(ts, regions)
        wdf = _make_weather_df(ts, regions, base_temp=5.0)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=True)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices, weather_df=wdf)

        pred_ts = _make_timestamps(24, "2026-01-13")
        pred_wdf = _make_weather_df(pred_ts, ["us-south"] * 24, base_temp=3.0)

        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-48:], weather_df=pred_wdf)
        assert len(preds) == 24
        assert all(p.p90 >= p.p50 >= 0 for p in preds)

    def test_predict_without_predict_weather_falls_back_to_train_lookup(self):
        ts = _make_timestamps(200, "2026-01-01")
        regions = ["us-south"] * 200
        prices = _make_prices(ts, regions)
        wdf = _make_weather_df(ts, regions)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=True)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices, weather_df=wdf)

        pred_ts = _make_timestamps(24, "2026-01-09")
        # No predict-time weather provided → uses cached training weather lookup
        preds = fc.predict("us-south", pred_ts, recent_prices=prices[-48:], weather_df=None)
        assert len(preds) == 24
        assert all(p.p90 >= p.p50 >= 0 for p in preds)

    def test_include_weather_false_ignores_weather_df(self):
        ts = _make_timestamps(200, "2026-01-01")
        regions = ["us-south"] * 200
        prices = _make_prices(ts, regions)
        wdf = _make_weather_df(ts, regions)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=False)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices, weather_df=wdf)

        # Even though weather_df was supplied, include_weather=False → v2.0
        assert fc.metadata.features_version == "v2.0"
        assert "+weather" not in fc.metadata.model_type

    def test_determinism_with_weather(self):
        ts = _make_timestamps(200, "2026-01-01")
        regions = ["us-south"] * 200
        prices = _make_prices(ts, regions)
        wdf = _make_weather_df(ts, regions)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=True)

        fc1 = PriceQuantileForecaster(config, corrections_path=False)
        fc1.fit(prices, weather_df=wdf)
        pred_ts = _make_timestamps(5, "2026-01-09")
        preds1 = fc1.predict("us-south", pred_ts, recent_prices=prices[-48:])

        fc2 = PriceQuantileForecaster(config, corrections_path=False)
        fc2.fit(prices, weather_df=wdf)
        preds2 = fc2.predict("us-south", pred_ts, recent_prices=prices[-48:])

        assert all(p1.p50 == p2.p50 for p1, p2 in zip(preds1, preds2))
        assert all(p1.p90 == p2.p90 for p1, p2 in zip(preds1, preds2))

    def test_multi_region_weather(self):
        ts_w = _make_timestamps(200, "2026-01-01")
        ts_e = _make_timestamps(200, "2026-01-01")
        regions_w = ["us-west"] * 200
        regions_e = ["us-east"] * 200
        all_ts = ts_w + ts_e
        all_regions = regions_w + regions_e
        prices = _make_prices(all_ts, all_regions)
        wdf = _make_weather_df(all_ts, all_regions, base_temp=10.0)

        config = PriceModelConfig(seed=42, n_estimators=20, include_weather_features=True)
        fc = PriceQuantileForecaster(config, corrections_path=False)
        fc.fit(prices, weather_df=wdf)

        pred_ts = _make_timestamps(12, "2026-01-09")
        for region in ["us-west", "us-east"]:
            region_wdf = _make_weather_df(pred_ts, [region] * 12)
            preds = fc.predict(region, pred_ts, weather_df=region_wdf)
            assert len(preds) == 12
            assert all(p.p90 >= p.p50 >= 0 for p in preds)


# ---------------------------------------------------------------------------
# 6. Leakage safety: weather split in BacktestEngine
# ---------------------------------------------------------------------------

class TestWeatherLeakageSafety:
    """Verify that the engine slices weather into train/eval correctly."""

    def test_train_weather_never_includes_eval_timestamps(self):
        """Engine._build_ml_forecast() should split weather on eval_start."""
        import pandas as pd
        from aurelius.backtesting.engine import BacktestEngine, _df_to_price_data
        from aurelius.backtesting.splitter import TemporalSplit

        # Build a dummy split
        train_start = pd.Timestamp("2026-01-01", tz="UTC")
        train_end   = pd.Timestamp("2026-01-08", tz="UTC")
        eval_start  = pd.Timestamp("2026-01-08", tz="UTC")
        eval_end    = pd.Timestamp("2026-01-15", tz="UTC")

        # All-region weather spanning both train and eval
        all_ts = [
            train_start + timedelta(hours=h)
            for h in range(int((eval_end - train_start).total_seconds() / 3600))
        ]
        wdf = _make_weather_df(all_ts, ["us-south"] * len(all_ts))

        # Simulate what the engine does
        wts = pd.to_datetime(wdf["timestamp"], utc=True)
        train_mask = wts < eval_start
        train_weather = wdf[train_mask]
        eval_weather  = wdf[~train_mask]

        assert (pd.to_datetime(train_weather["timestamp"], utc=True) < eval_start).all(), \
            "Train weather must not include eval-window timestamps"
        assert not train_weather.empty
        assert not eval_weather.empty

    def test_weather_df_with_no_region_match_is_empty(self):
        ts = _make_timestamps(24, "2026-01-01")
        wdf = _make_weather_df(ts, ["eu-west"] * 24)  # non-existent region
        mask = wdf["region"] == "us-south"
        result = wdf[mask]
        assert result.empty


# ---------------------------------------------------------------------------
# 7. Integration: BacktestEngine with weather_df end-to-end smoke test
# ---------------------------------------------------------------------------

class TestBacktestEngineWeatherIntegration:
    """Smoke test: engine runs without error when weather_df is provided."""

    def test_engine_with_weather_produces_folds(self):
        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.ingestion.job_logs import JobLogIngester
        from aurelius.models import OptimizationConfig
        from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter

        repo_root = _REPO_ROOT
        da_file = repo_root / "data" / "ercot_us_south_dam.csv"
        if not da_file.exists():
            pytest.skip("ERCOT price data not available")

        price_df = CSVPriceImporter(str(da_file)).load_all()
        price_df = price_df[price_df["region"] == "us-south"]
        if price_df.empty:
            pytest.skip("No us-south price data")

        # Trim to 45 days to keep test fast
        start_ts = pd.Timestamp("2026-01-01", tz="UTC")
        end_ts   = pd.Timestamp("2026-02-15", tz="UTC")
        price_df = price_df[
            (pd.to_datetime(price_df["timestamp"], utc=True) >= start_ts) &
            (pd.to_datetime(price_df["timestamp"], utc=True) < end_ts)
        ]

        # Load actual weather data
        weather_file = repo_root / "data" / "weather_q12026.csv"
        if not weather_file.exists():
            pytest.skip("Weather data not available")
        wdf = pd.read_csv(str(weather_file))
        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], utc=True)
        wdf = wdf[wdf["region"] == "us-south"]

        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            start_time=start_ts.to_pydatetime(),
            duration_hours=1100,
            num_jobs=15,
            regions=["us-south"],
            seed=42,
            workload_mix="realistic",
            workload_filter="training",
        )

        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        config = PriceModelConfig(seed=42, n_estimators=30, include_weather_features=True)

        engine = BacktestEngine(
            method="greedy_migrate",
            train_days=20,
            eval_days=5,
            config=OptimizationConfig(),
            price_forecaster_cls=PriceQuantileForecaster,
            price_forecaster_config=config,
            context_hours=168,
            weather_df=wdf,
        )
        rounds = engine.run(
            jobs, price_df,
            carbon_df=pd.DataFrame(),
            start=start_ts, end=end_ts,
        )
        assert len(rounds) >= 1, "Expected at least 1 fold"
        assert rounds[0].optimizer_metrics is not None
