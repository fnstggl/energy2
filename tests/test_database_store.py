"""Tests for aurelius.database.store.TimeSeriesStore.

All tests use SQLite in-memory databases — no live Postgres required.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from aurelius.database import TimeSeriesStore
from aurelius.database.store import _empty_carbon_df, _empty_price_df, _to_utc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Fresh in-memory SQLite TimeSeriesStore for each test."""
    s = TimeSeriesStore("sqlite:///:memory:")
    yield s
    s.close()


@pytest.fixture
def disabled_store():
    """Store with no DATABASE_URL set (no-op mode)."""
    return TimeSeriesStore(url="")


def _price_df(region: str = "us-west", n: int = 5, source: str = "caiso_da") -> pd.DataFrame:
    """Build a minimal canonical price DataFrame."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [t0 + timedelta(hours=i) for i in range(n)],
            "region": [region] * n,
            "price_per_mwh": [30.0 + i for i in range(n)],
            "currency": ["USD"] * n,
            "source": [source] * n,
            "source_granularity": ["hourly"] * n,
        }
    )


def _carbon_df(region: str = "us-west", n: int = 5) -> pd.DataFrame:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [t0 + timedelta(hours=i) for i in range(n)],
            "region": [region] * n,
            "gco2_per_kwh": [200.0 + i * 5 for i in range(n)],
            "source": ["watttime_moer"] * n,
            "source_granularity": ["hourly"] * n,
        }
    )


# ---------------------------------------------------------------------------
# TestTimeSeriesStoreInit
# ---------------------------------------------------------------------------

class TestTimeSeriesStoreInit:
    def test_sqlite_memory_is_enabled(self):
        s = TimeSeriesStore("sqlite:///:memory:")
        assert s.enabled is True
        s.close()

    def test_empty_url_is_disabled(self):
        s = TimeSeriesStore(url="")
        assert s.enabled is False

    def test_no_url_is_disabled(self):
        import os

        old = os.environ.pop("DATABASE_URL", None)
        try:
            s = TimeSeriesStore()
            assert s.enabled is False
        finally:
            if old is not None:
                os.environ["DATABASE_URL"] = old

    def test_bad_url_is_disabled(self):
        # Bad URL: no such driver → should not raise, just disable
        s = TimeSeriesStore("baddriver://nope")
        assert s.enabled is False

    def test_dialect_disabled_when_no_url(self):
        s = TimeSeriesStore(url="")
        assert s.dialect == "disabled"

    def test_dialect_sqlite_for_memory(self):
        s = TimeSeriesStore("sqlite:///:memory:")
        assert s.dialect == "sqlite"
        s.close()

    def test_tables_created_on_init(self, store):
        # row_counts should work immediately after init
        counts = store.row_counts()
        assert "energy_prices" in counts
        assert "carbon_intensity" in counts
        assert "benchmark_runs" in counts

    def test_row_counts_initially_zero(self, store):
        counts = store.row_counts()
        assert counts["energy_prices"] == 0
        assert counts["carbon_intensity"] == 0
        assert counts["benchmark_runs"] == 0


# ---------------------------------------------------------------------------
# TestUpsertPrices
# ---------------------------------------------------------------------------

class TestUpsertPrices:
    def test_upsert_basic(self, store):
        df = _price_df(n=5)
        n = store.upsert_prices(df)
        assert n == 5
        assert store.row_counts()["energy_prices"] == 5

    def test_upsert_returns_zero_when_disabled(self, disabled_store):
        df = _price_df(n=5)
        n = disabled_store.upsert_prices(df)
        assert n == 0

    def test_upsert_returns_zero_for_empty_df(self, store):
        n = store.upsert_prices(pd.DataFrame())
        assert n == 0

    def test_upsert_idempotent_same_rows(self, store):
        df = _price_df(n=5)
        store.upsert_prices(df)
        n2 = store.upsert_prices(df)
        # Second upsert: 0 new rows inserted (duplicates ignored)
        assert n2 == 0
        assert store.row_counts()["energy_prices"] == 5

    def test_upsert_mixed_new_and_dup(self, store):
        df1 = _price_df(n=3)
        df2 = _price_df(n=5)  # first 3 are duplicates
        store.upsert_prices(df1)
        n2 = store.upsert_prices(df2)
        # Only 2 new rows
        assert n2 == 2
        assert store.row_counts()["energy_prices"] == 5

    def test_upsert_multiple_regions(self, store):
        df_west = _price_df(region="us-west", n=3)
        df_east = _price_df(region="us-east", n=3)
        store.upsert_prices(pd.concat([df_west, df_east], ignore_index=True))
        assert store.row_counts()["energy_prices"] == 6

    def test_upsert_handles_naive_timestamps(self, store):
        """Timestamps without tzinfo should be accepted and stored as UTC."""
        df = _price_df(n=2)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        n = store.upsert_prices(df)
        assert n == 2

    def test_upsert_different_sources_are_separate_rows(self, store):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df_da = pd.DataFrame({
            "timestamp": [t0],
            "region": ["us-west"],
            "price_per_mwh": [30.0],
            "currency": ["USD"],
            "source": ["caiso_da"],
        })
        df_rt = pd.DataFrame({
            "timestamp": [t0],
            "region": ["us-west"],
            "price_per_mwh": [32.0],
            "currency": ["USD"],
            "source": ["caiso_rt"],
        })
        store.upsert_prices(df_da)
        store.upsert_prices(df_rt)
        assert store.row_counts()["energy_prices"] == 2


# ---------------------------------------------------------------------------
# TestGetPrices
# ---------------------------------------------------------------------------

class TestGetPrices:
    def test_get_returns_correct_rows(self, store):
        df = _price_df(region="us-west", n=24, source="caiso_da")
        store.upsert_prices(df)

        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_prices("us-west", t0, t0 + timedelta(hours=5))
        assert len(result) == 6  # hours 0-5 inclusive

    def test_get_empty_when_disabled(self, disabled_store):
        result = disabled_store.get_prices(
            "us-west", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_empty_when_no_data_for_region(self, store):
        df = _price_df(region="us-west", n=5)
        store.upsert_prices(df)
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_prices("us-east", t0, t0 + timedelta(hours=5))
        assert result.empty

    def test_get_returns_sorted_by_timestamp(self, store):
        # Insert shuffled timestamps
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "timestamp": [t0 + timedelta(hours=i) for i in [2, 0, 4, 1, 3]],
            "region": ["us-west"] * 5,
            "price_per_mwh": [30.0] * 5,
            "source": ["caiso_da"] * 5,
        })
        store.upsert_prices(df)
        result = store.get_prices("us-west", t0, t0 + timedelta(hours=4))
        assert list(result["timestamp"]) == sorted(result["timestamp"])

    def test_get_correct_columns(self, store):
        df = _price_df(n=2)
        store.upsert_prices(df)
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_prices("us-west", t0, t0 + timedelta(hours=2))
        for col in ["timestamp", "region", "price_per_mwh", "currency", "source"]:
            assert col in result.columns

    def test_get_filter_by_source(self, store):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df_da = pd.DataFrame({
            "timestamp": [t0, t0 + timedelta(hours=1)],
            "region": ["us-west"] * 2,
            "price_per_mwh": [30.0, 31.0],
            "currency": ["USD"] * 2,
            "source": ["caiso_da"] * 2,
        })
        df_rt = pd.DataFrame({
            "timestamp": [t0],
            "region": ["us-west"],
            "price_per_mwh": [32.0],
            "currency": ["USD"],
            "source": ["caiso_rt"],
        })
        store.upsert_prices(pd.concat([df_da, df_rt], ignore_index=True))

        result = store.get_prices("us-west", t0, t0 + timedelta(hours=2), source="caiso_da")
        assert len(result) == 2
        assert set(result["source"]) == {"caiso_da"}

    def test_get_naive_timestamp_query_works(self, store):
        """Query with naive start/end datetimes is handled correctly."""
        df = _price_df(n=5)
        store.upsert_prices(df)
        t0_naive = datetime(2026, 1, 1)  # no tzinfo
        result = store.get_prices("us-west", t0_naive, t0_naive + timedelta(hours=4))
        assert len(result) == 5

    def test_get_prices_empty_df_schema(self, store):
        """Empty result should have correct column schema."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_prices("no-such-region", t0, t0 + timedelta(hours=5))
        assert list(result.columns) == ["timestamp", "region", "price_per_mwh", "currency", "source"]


# ---------------------------------------------------------------------------
# TestUpsertCarbon
# ---------------------------------------------------------------------------

class TestUpsertCarbon:
    def test_upsert_basic(self, store):
        df = _carbon_df(n=5)
        n = store.upsert_carbon(df)
        assert n == 5
        assert store.row_counts()["carbon_intensity"] == 5

    def test_upsert_idempotent(self, store):
        df = _carbon_df(n=5)
        store.upsert_carbon(df)
        n2 = store.upsert_carbon(df)
        assert n2 == 0

    def test_upsert_returns_zero_disabled(self, disabled_store):
        assert disabled_store.upsert_carbon(_carbon_df()) == 0

    def test_upsert_empty_returns_zero(self, store):
        assert store.upsert_carbon(pd.DataFrame()) == 0


# ---------------------------------------------------------------------------
# TestGetCarbon
# ---------------------------------------------------------------------------

class TestGetCarbon:
    def test_get_basic(self, store):
        df = _carbon_df(n=10)
        store.upsert_carbon(df)
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_carbon("us-west", t0, t0 + timedelta(hours=4))
        assert len(result) == 5

    def test_get_correct_columns(self, store):
        store.upsert_carbon(_carbon_df(n=2))
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_carbon("us-west", t0, t0 + timedelta(hours=2))
        for col in ["timestamp", "region", "gco2_per_kwh", "source"]:
            assert col in result.columns

    def test_get_empty_when_disabled(self, disabled_store):
        result = disabled_store.get_carbon(
            "us-west", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)
        )
        assert result.empty

    def test_get_empty_schema(self, store):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = store.get_carbon("no-region", t0, t0 + timedelta(hours=5))
        assert list(result.columns) == ["timestamp", "region", "gco2_per_kwh", "source"]


# ---------------------------------------------------------------------------
# TestBenchmarkRuns
# ---------------------------------------------------------------------------

class TestBenchmarkRuns:
    def test_save_and_retrieve(self, store):
        store.save_benchmark_run(
            run_id="20260523T200730Z",
            forecaster="ml_quantile_recovery",
            region_combo="caiso_pjm_ercot_da_rt",
            workload="training",
            savings_vs_cpo=15.0,
            folds=5,
            miss_pct=0.0,
        )
        history = store.get_benchmark_history()
        assert len(history) == 1
        row = history[0]
        assert row["run_id"] == "20260523T200730Z"
        assert row["forecaster"] == "ml_quantile_recovery"
        assert row["workload"] == "training"
        assert abs(row["savings_vs_cpo"] - 15.0) < 0.001
        assert row["folds"] == 5

    def test_overwrite_same_cell(self, store):
        store.save_benchmark_run(
            run_id="run1",
            forecaster="ml_quantile",
            region_combo="caiso_pjm",
            workload="training",
            savings_vs_cpo=14.0,
            folds=5,
        )
        store.save_benchmark_run(
            run_id="run1",
            forecaster="ml_quantile",
            region_combo="caiso_pjm",
            workload="training",
            savings_vs_cpo=15.5,
            folds=5,
        )
        history = store.get_benchmark_history()
        assert len(history) == 1
        assert abs(history[0]["savings_vs_cpo"] - 15.5) < 0.001

    def test_save_multiple_workloads(self, store):
        workloads = ["training", "fine_tuning", "llm_batch_inference", "background_maintenance"]
        for wl in workloads:
            store.save_benchmark_run(
                run_id="20260523T200730Z",
                forecaster="ml_quantile_recovery",
                region_combo="caiso_pjm_ercot_da_rt",
                workload=wl,
                savings_vs_cpo=20.0,
                folds=5,
            )
        assert store.row_counts()["benchmark_runs"] == len(workloads)

    def test_filter_by_region_combo(self, store):
        store.save_benchmark_run("r1", "f1", "region_A", "training", 10.0, 5)
        store.save_benchmark_run("r1", "f1", "region_B", "training", 20.0, 5)

        history = store.get_benchmark_history(region_combo="region_A")
        assert len(history) == 1
        assert history[0]["region_combo"] == "region_A"

    def test_filter_by_workload(self, store):
        store.save_benchmark_run("r1", "f1", "combo", "training", 10.0, 5)
        store.save_benchmark_run("r1", "f1", "combo", "fine_tuning", 20.0, 5)

        history = store.get_benchmark_history(workload="fine_tuning")
        assert len(history) == 1
        assert history[0]["workload"] == "fine_tuning"

    def test_filter_by_forecaster(self, store):
        store.save_benchmark_run("r1", "ml_quantile", "combo", "training", 10.0, 5)
        store.save_benchmark_run("r2", "ml_quantile_recovery", "combo", "training", 15.0, 5)

        history = store.get_benchmark_history(forecaster="ml_quantile_recovery")
        assert len(history) == 1
        assert history[0]["forecaster"] == "ml_quantile_recovery"

    def test_meta_json_round_trip(self, store):
        meta = {"oracle_ceiling": 29.9, "notes": "cold snap period"}
        store.save_benchmark_run("r1", "f1", "combo", "training", 15.0, 5, meta=meta)
        history = store.get_benchmark_history()
        assert history[0]["meta"] == meta

    def test_history_limit(self, store):
        for i in range(10):
            store.save_benchmark_run(f"run{i}", "f1", "combo", f"wl{i}", float(i), 5)
        history = store.get_benchmark_history(limit=3)
        assert len(history) == 3

    def test_history_empty_when_disabled(self, disabled_store):
        history = disabled_store.get_benchmark_history()
        assert history == []

    def test_save_noop_when_disabled(self, disabled_store):
        # Should not raise
        disabled_store.save_benchmark_run("r1", "f1", "combo", "training", 15.0, 5)


# ---------------------------------------------------------------------------
# TestRowCounts
# ---------------------------------------------------------------------------

class TestRowCounts:
    def test_counts_reflect_inserts(self, store):
        store.upsert_prices(_price_df(n=10))
        store.upsert_carbon(_carbon_df(n=7))
        store.save_benchmark_run("r1", "f1", "c1", "training", 15.0, 5)

        counts = store.row_counts()
        assert counts["energy_prices"] == 10
        assert counts["carbon_intensity"] == 7
        assert counts["benchmark_runs"] == 1

    def test_counts_when_disabled(self, disabled_store):
        counts = disabled_store.row_counts()
        assert counts == {"enabled": False}


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_to_utc_naive(self):
        dt = datetime(2026, 1, 1, 12, 0, 0)
        result = _to_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_to_utc_aware(self):
        from datetime import timedelta
        from datetime import timezone as tz_module

        est = tz_module(timedelta(hours=-5))
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=est)
        result = _to_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # EST = UTC-5

    def test_empty_price_df_schema(self):
        df = _empty_price_df()
        assert list(df.columns) == ["timestamp", "region", "price_per_mwh", "currency", "source"]
        assert len(df) == 0

    def test_empty_carbon_df_schema(self):
        df = _empty_carbon_df()
        assert list(df.columns) == ["timestamp", "region", "gco2_per_kwh", "source"]
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_disables_store(self):
        s = TimeSeriesStore("sqlite:///:memory:")
        assert s.enabled is True
        s.close()
        assert s.enabled is False

    def test_close_idempotent(self):
        s = TimeSeriesStore("sqlite:///:memory:")
        s.close()
        s.close()  # Should not raise


# ---------------------------------------------------------------------------
# TestIntegrationPriceRoundTrip
# ---------------------------------------------------------------------------

class TestIntegrationPriceRoundTrip:
    def test_insert_then_retrieve_preserves_values(self, store):
        t0 = datetime(2026, 2, 15, 6, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "timestamp": [t0 + timedelta(hours=i) for i in range(3)],
            "region": ["us-west"] * 3,
            "price_per_mwh": [45.2, 51.8, 38.9],
            "currency": ["USD"] * 3,
            "source": ["caiso_da"] * 3,
        })
        store.upsert_prices(df)
        result = store.get_prices("us-west", t0, t0 + timedelta(hours=2))

        assert len(result) == 3
        prices = sorted(result["price_per_mwh"].tolist())
        assert abs(prices[0] - 38.9) < 0.01
        assert abs(prices[1] - 45.2) < 0.01
        assert abs(prices[2] - 51.8) < 0.01

    def test_three_region_combo(self, store):
        """Simulate a 3-region benchmark data scenario."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for region, source, price in [
            ("us-west", "caiso_da", 30.0),
            ("us-east", "pjm_da", 45.0),
            ("us-south", "ercot_da", 25.0),
        ]:
            df = pd.DataFrame({
                "timestamp": [t0 + timedelta(hours=i) for i in range(24)],
                "region": [region] * 24,
                "price_per_mwh": [price + i * 0.5 for i in range(24)],
                "currency": ["USD"] * 24,
                "source": [source] * 24,
            })
            store.upsert_prices(df)

        assert store.row_counts()["energy_prices"] == 72

        west = store.get_prices("us-west", t0, t0 + timedelta(hours=23))
        east = store.get_prices("us-east", t0, t0 + timedelta(hours=23))
        south = store.get_prices("us-south", t0, t0 + timedelta(hours=23))

        assert len(west) == 24
        assert len(east) == 24
        assert len(south) == 24
        assert west["price_per_mwh"].mean() < east["price_per_mwh"].mean()
