"""Live integration test for ENTSO-E provider – skipped without ENTSOE_API_KEY."""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ENTSOE_API_KEY"),
    reason="ENTSOE_API_KEY not set; skipping live ENTSO-E test",
)


def test_entsoe_fetch_eu_west():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.entsoe import ENTSOEPriceProvider

    provider = ENTSOEPriceProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=48)

    df = provider.fetch_prices("eu-west", start, end)

    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["region"] == "eu-west").all()
        assert (df["currency"] == "EUR").all()
