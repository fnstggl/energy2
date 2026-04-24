"""Live integration test for CAISO OASIS provider – no auth required.

CAISO OASIS is a public API so no skip condition is needed. However, the test
is network-dependent and may be slow (~5-30s per request). It is placed in the
live/ directory so CI can skip the entire directory if needed.
"""

from datetime import datetime, timedelta, timezone

import pytest


def test_caiso_fetch_us_west():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider

    provider = CAISOPriceProvider()
    # Use a fixed historical date well in the past to avoid data-not-yet-published errors
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)

    df = provider.fetch_prices("us-west", start, end)

    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-west").all()
        assert (df["currency"] == "USD").all()
        assert (df["source"] == "caiso_oasis_dam").all()
        # Day-ahead prices should be in a plausible range
        assert (df["price_per_mwh"] > -500).all()
        assert (df["price_per_mwh"] < 10000).all()


def test_caiso_unknown_region_returns_empty():
    """No network call should happen for an unmapped region."""
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider

    provider = CAISOPriceProvider()
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    df = provider.fetch_prices("eu-west", start, end)

    assert df.empty
    assert list(df.columns) == PRICE_COLUMNS
