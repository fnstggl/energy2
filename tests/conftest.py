"""Shared fixtures for the Aurelius test suite."""

import os
import io
from datetime import datetime, timedelta, timezone
from typing import Generator

import pandas as pd
import pytest

from aurelius.models import Job, OptimizationConfig, ScheduleDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc

_T0 = datetime(2024, 1, 15, 0, 0, tzinfo=UTC)


def _hours(n: int) -> timedelta:
    return timedelta(hours=n)


# ---------------------------------------------------------------------------
# Canonical price CSV fixture (in-memory)
# ---------------------------------------------------------------------------

def _make_price_csv(regions=("us-west", "us-east"), hours=72, base_price=50.0) -> str:
    rows = ["timestamp,region,price_per_mwh"]
    for h in range(hours):
        ts = (_T0 + _hours(h)).isoformat()
        for region in regions:
            price = base_price + (h % 24) * 2  # mild diurnal pattern
            rows.append(f"{ts},{region},{price:.2f}")
    return "\n".join(rows)


def _make_carbon_csv(regions=("us-west", "us-east"), hours=72, base_carbon=300.0) -> str:
    rows = ["timestamp,region,gco2_per_kwh"]
    for h in range(hours):
        ts = (_T0 + _hours(h)).isoformat()
        for region in regions:
            carbon = base_carbon + (h % 12) * 5
            rows.append(f"{ts},{region},{carbon:.2f}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_regions():
    return ["us-west", "us-east"]


@pytest.fixture
def t0():
    return _T0


@pytest.fixture
def price_csv_content(sample_regions):
    return _make_price_csv(regions=sample_regions)


@pytest.fixture
def carbon_csv_content(sample_regions):
    return _make_carbon_csv(regions=sample_regions)


@pytest.fixture
def price_csv_path(tmp_path, price_csv_content):
    p = tmp_path / "prices.csv"
    p.write_text(price_csv_content)
    return p


@pytest.fixture
def carbon_csv_path(tmp_path, carbon_csv_content):
    p = tmp_path / "carbon.csv"
    p.write_text(carbon_csv_content)
    return p


@pytest.fixture
def price_df(price_csv_path):
    from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
    return CSVPriceImporter(price_csv_path).load_all()


@pytest.fixture
def carbon_df(carbon_csv_path):
    from aurelius.ingestion.grid_apis.csv_importer import CSVCarbonImporter
    return CSVCarbonImporter(carbon_csv_path).load_all()


@pytest.fixture
def sample_jobs(t0, sample_regions):
    """A small set of test jobs whose windows fit inside the 72h price fixture."""
    jobs = []
    for i in range(5):
        jobs.append(Job(
            job_id=f"job-{i}",
            submit_time=t0 + _hours(i),
            runtime_hours=2.0,
            deadline=t0 + _hours(24 + i * 4),
            power_kw=100.0,
            earliest_start=t0 + _hours(i),
            region_options=sample_regions,
            priority=1,
        ))
    return jobs


@pytest.fixture
def opt_config(sample_regions):
    return OptimizationConfig(
        region_power_caps={r: 10_000.0 for r in sample_regions},
        default_region="us-west",
    )


@pytest.fixture
def price_data_dict(price_df):
    """Convert canonical price_df to the {region: {ts: price}} dict the scheduler expects."""
    result = {}
    for _, row in price_df.iterrows():
        region = row["region"]
        ts = row["timestamp"].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        result.setdefault(region, {})[ts] = float(row["price_per_mwh"])
    return result


@pytest.fixture
def carbon_data_dict(carbon_df):
    result = {}
    for _, row in carbon_df.iterrows():
        region = row["region"]
        ts = row["timestamp"].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        result.setdefault(region, {})[ts] = float(row["gco2_per_kwh"])
    return result


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------

def make_decision(
    job_id: str = "job-0",
    start: datetime | None = None,
    region: str = "us-west",
    power: float = 1.0,
    runtime: float = 2.0,
    forecast: dict | None = None,
) -> ScheduleDecision:
    return ScheduleDecision(
        job_id=job_id,
        start_time=start or _T0,
        region=region,
        power_fraction=power,
        actual_runtime_hours=runtime,
        forecast=forecast,
    )
