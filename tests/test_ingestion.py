"""Tests for the ingestion layer: CSV importers and provider key guards."""

from datetime import timezone

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import (
    CARBON_COLUMNS,
    PRICE_COLUMNS,
    ProviderConfigError,
    empty_carbon_df,
    empty_price_df,
    normalize_carbon_df,
    normalize_price_df,
)
from aurelius.ingestion.grid_apis.csv_importer import CSVCarbonImporter, CSVPriceImporter
from aurelius.ingestion.grid_apis.eia import EIAPriceProvider
from aurelius.ingestion.grid_apis.electricitymaps import ElectricityMapsCarbonProvider
from aurelius.ingestion.grid_apis.entsoe import ENTSOEPriceProvider
from aurelius.ingestion.grid_apis.market_registry import UnsupportedMarketPriceError

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Base module
# ---------------------------------------------------------------------------

def test_empty_price_df_schema():
    df = empty_price_df()
    assert list(df.columns) == PRICE_COLUMNS
    assert len(df) == 0


def test_empty_carbon_df_schema():
    df = empty_carbon_df()
    assert list(df.columns) == CARBON_COLUMNS
    assert len(df) == 0


def test_normalize_price_df():
    raw = pd.DataFrame([
        {"timestamp": "2024-01-01T00:00:00Z", "region": "us-west", "price_per_mwh": 45.0},
        {"timestamp": "2024-01-01T01:00:00Z", "region": "us-west", "price_per_mwh": 50.0},
    ])
    df = normalize_price_df(raw, source="test", currency="USD", granularity="hourly")
    assert list(df.columns) == PRICE_COLUMNS
    assert len(df) == 2
    assert df["source"].iloc[0] == "test"
    assert df["currency"].iloc[0] == "USD"
    assert df["timestamp"].dtype.tz is not None  # UTC-aware


def test_normalize_price_df_drops_nan_prices():
    raw = pd.DataFrame([
        {"timestamp": "2024-01-01T00:00:00Z", "region": "us-west", "price_per_mwh": None},
        {"timestamp": "2024-01-01T01:00:00Z", "region": "us-west", "price_per_mwh": 50.0},
    ])
    df = normalize_price_df(raw, source="test", currency="USD")
    assert len(df) == 1


def test_normalize_carbon_df():
    raw = pd.DataFrame([
        {"timestamp": "2024-01-01T00:00:00Z", "region": "us-west", "gco2_per_kwh": 300.0},
    ])
    df = normalize_carbon_df(raw, source="test", granularity="hourly")
    assert list(df.columns) == CARBON_COLUMNS
    assert df["gco2_per_kwh"].iloc[0] == 300.0


# ---------------------------------------------------------------------------
# CSV importers
# ---------------------------------------------------------------------------

def test_csv_price_importer_load_all(price_csv_path, sample_regions):
    importer = CSVPriceImporter(price_csv_path)
    df = importer.load_all()
    assert not df.empty
    assert list(df.columns) == PRICE_COLUMNS
    assert set(df["region"].unique()) == set(sample_regions)


def test_csv_price_importer_fetch_prices(price_csv_path, t0, sample_regions):
    importer = CSVPriceImporter(price_csv_path)
    region = sample_regions[0]
    start = t0
    end = t0 + pd.Timedelta(hours=24)
    df = importer.fetch_prices(region=region, start=start, end=end)
    assert not df.empty
    assert (df["region"] == region).all()
    start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).tz_localize("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end).tz_localize("UTC")
    assert (df["timestamp"] >= start_ts).all()
    assert (df["timestamp"] < end_ts).all()


def test_csv_price_importer_unknown_region(price_csv_path, t0):
    importer = CSVPriceImporter(price_csv_path)
    df = importer.fetch_prices("eu-east", t0, t0 + pd.Timedelta(hours=24))
    assert df.empty
    assert list(df.columns) == PRICE_COLUMNS


def test_csv_price_importer_missing_file(tmp_path, t0):
    importer = CSVPriceImporter(tmp_path / "nonexistent.csv")
    df = importer.fetch_prices("us-west", t0, t0 + pd.Timedelta(hours=1))
    assert df.empty


def test_csv_carbon_importer_load_all(carbon_csv_path, sample_regions):
    importer = CSVCarbonImporter(carbon_csv_path)
    df = importer.load_all()
    assert not df.empty
    assert list(df.columns) == CARBON_COLUMNS


def test_csv_carbon_importer_fetch_carbon(carbon_csv_path, t0, sample_regions):
    importer = CSVCarbonImporter(carbon_csv_path)
    region = sample_regions[0]
    df = importer.fetch_carbon(region, t0, t0 + pd.Timedelta(hours=12))
    assert not df.empty
    assert (df["region"] == region).all()


# ---------------------------------------------------------------------------
# Provider key guards (no API keys in CI → must raise ProviderConfigError)
# ---------------------------------------------------------------------------

def test_eia_raises_unsupported_for_any_region(t0):
    """EIA never provides wholesale prices — must raise UnsupportedMarketPriceError."""
    provider = EIAPriceProvider(api_key="")
    with pytest.raises(UnsupportedMarketPriceError):
        provider.fetch_prices("us-west", t0, t0 + pd.Timedelta(hours=1))


def test_eia_raises_unsupported_even_with_key(t0):
    """EIA raises UnsupportedMarketPriceError regardless of whether a key is present."""
    provider = EIAPriceProvider(api_key="dummy-key")
    with pytest.raises(UnsupportedMarketPriceError):
        provider.fetch_prices("us-east", t0, t0 + pd.Timedelta(hours=1))


def test_eia_error_message_guides_to_real_providers(t0):
    """Error message must name the correct alternatives."""
    provider = EIAPriceProvider()
    with pytest.raises(UnsupportedMarketPriceError, match="CAISOPriceProvider|PJMPriceProvider"):
        provider.fetch_prices("us-west", t0, t0 + pd.Timedelta(hours=1))


def test_entsoe_raises_without_key(monkeypatch, t0):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    provider = ENTSOEPriceProvider(api_key="")
    with pytest.raises(ProviderConfigError, match="ENTSOE_API_KEY"):
        provider.fetch_prices("eu-west", t0, t0 + pd.Timedelta(hours=1))


def test_electricitymaps_raises_without_key(monkeypatch, t0):
    monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
    provider = ElectricityMapsCarbonProvider(api_key="")
    with pytest.raises(ProviderConfigError, match="ELECTRICITYMAPS_API_KEY"):
        provider.fetch_carbon("us-west", t0, t0 + pd.Timedelta(hours=1))


def test_electricitymaps_unknown_region_returns_empty(t0):
    provider = ElectricityMapsCarbonProvider(api_key="dummy")
    df = provider.fetch_carbon("xx-unknown", t0, t0 + pd.Timedelta(hours=1))
    assert df.empty


# ---------------------------------------------------------------------------
# Job file round-trip fidelity (JSON/CSV) and timezone normalization
# ---------------------------------------------------------------------------

def _sample_jobs():
    from aurelius.ingestion.job_logs import JobLogIngester
    start = pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime()
    return JobLogIngester().generate_synthetic(
        start_time=start, duration_hours=240, num_jobs=20,
        regions=["us-west", "us-east", "us-south"], seed=7, workload_mix="realistic",
    )


def test_json_round_trip_preserves_migration_and_workload(tmp_path):
    from aurelius.ingestion.job_logs import JobLogIngester
    ing = JobLogIngester()
    jobs = _sample_jobs()
    p = tmp_path / "jobs.json"
    ing.save_to_json(jobs, p)
    loaded = ing.load_from_json(p)
    assert len(loaded) == len(jobs)
    for a, b in zip(jobs, loaded):
        assert a.migration_cost_hours == b.migration_cost_hours
        assert a.workload_type == b.workload_type
        assert a.earliest_start == b.earliest_start
        assert a.region_options == b.region_options


def test_csv_round_trip_preserves_migration_and_workload(tmp_path):
    from aurelius.ingestion.job_logs import JobLogIngester
    ing = JobLogIngester()
    jobs = _sample_jobs()
    p = tmp_path / "jobs.csv"
    ing.save_to_csv(jobs, p)
    loaded = ing.load_from_csv(p)
    assert len(loaded) == len(jobs)
    for a, b in zip(jobs, loaded):
        assert a.migration_cost_hours == b.migration_cost_hours
        assert a.workload_type == b.workload_type


def test_load_json_normalizes_naive_datetimes_to_utc(tmp_path):
    import json

    from aurelius.ingestion.job_logs import JobLogIngester
    naive = [{
        "job_id": "j1", "submit_time": "2026-01-15T00:00:00",
        "runtime_hours": 2.0, "deadline": "2026-01-16T00:00:00",
        "power_kw": 50.0, "earliest_start": "2026-01-15T03:00:00",
        "region_options": ["us-west"],
    }]
    p = tmp_path / "naive.json"
    p.write_text(json.dumps(naive))
    job = JobLogIngester().load_from_json(p)[0]
    assert job.earliest_start.tzinfo is not None
    assert job.submit_time.utcoffset().total_seconds() == 0
