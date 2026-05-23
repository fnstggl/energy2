"""Live integration test for EIA provider – skipped without EIA_API_KEY."""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("EIA_API_KEY"),
    reason="EIA_API_KEY not set; skipping live EIA test",
)


def test_eia_fetch_us_west():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.eia import EIAPriceProvider

    provider = EIAPriceProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=48)

    df = provider.fetch_prices("us-west", start, end)

    # May be empty if EIA doesn't have recent data yet, but schema must be correct
    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-west").all()
        assert df["price_per_mwh"].notna().all()


def test_eia_unknown_region_returns_empty():
    from datetime import datetime, timezone

    from aurelius.ingestion.grid_apis.eia import EIAPriceProvider

    provider = EIAPriceProvider()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=1)
    df = provider.fetch_prices("xx-invalid", start, end)
    assert df.empty
