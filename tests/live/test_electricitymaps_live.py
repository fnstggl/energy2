"""Live integration test for ElectricityMaps – skipped without ELECTRICITYMAPS_API_KEY."""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ELECTRICITYMAPS_API_KEY"),
    reason="ELECTRICITYMAPS_API_KEY not set; skipping live ElectricityMaps test",
)


def test_electricitymaps_fetch_us_west():
    from aurelius.ingestion.grid_apis.base import CARBON_COLUMNS
    from aurelius.ingestion.grid_apis.electricitymaps import ElectricityMapsCarbonProvider

    provider = ElectricityMapsCarbonProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=48)

    df = provider.fetch_carbon("us-west", start, end)

    assert list(df.columns) == CARBON_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-west").all()
        assert df["gco2_per_kwh"].notna().all()
