"""Live integration test for WattTime carbon provider – skipped without credentials."""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("WATTTIME_USERNAME") and os.environ.get("WATTTIME_PASSWORD")),
    reason="WATTTIME_USERNAME + WATTTIME_PASSWORD not set; skipping live WattTime test",
)


def test_watttime_fetch_us_west():
    from aurelius.ingestion.grid_apis.base import CARBON_COLUMNS
    from aurelius.ingestion.grid_apis.watttime import WattTimeCarbonProvider

    provider = WattTimeCarbonProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=48)

    df = provider.fetch_carbon("us-west", start, end)

    assert list(df.columns) == CARBON_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-west").all()
        assert (df["gco2_per_kwh"] >= 0).all()
        # source column must identify MOER signal
        assert "moer" in df["source"].iloc[0].lower()


def test_watttime_fetch_us_east():
    from aurelius.ingestion.grid_apis.base import CARBON_COLUMNS
    from aurelius.ingestion.grid_apis.watttime import WattTimeCarbonProvider

    provider = WattTimeCarbonProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=24)

    df = provider.fetch_carbon("us-east", start, end)

    assert list(df.columns) == CARBON_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-east").all()
        assert (df["gco2_per_kwh"] >= 0).all()
