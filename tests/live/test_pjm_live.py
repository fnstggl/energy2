"""Live integration test for PJM Data Miner provider – skipped without PJM_API_KEY."""

import os
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PJM_API_KEY"),
    reason="PJM_API_KEY not set; skipping live PJM test",
)


def test_pjm_fetch_us_east():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider

    provider = PJMPriceProvider()
    # Use a fixed historical date to avoid data-not-yet-published issues
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)

    df = provider.fetch_prices("us-east", start, end)

    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-east").all()
        assert (df["currency"] == "USD").all()
        assert (df["source"] == "pjm_da_lmp").all()
        assert (df["price_per_mwh"] > -500).all()
        assert (df["price_per_mwh"] < 10000).all()
