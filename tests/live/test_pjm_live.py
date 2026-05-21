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


def _recent_utc_window(days_ago=5, hours=2):
    from datetime import timedelta
    end = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(days=days_ago)
    return end - timedelta(hours=hours), end


def test_pjm_fetch_rt_5min_us_east():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.pjm import PJMRealtimePriceProvider

    provider = PJMRealtimePriceProvider()
    # RT five-minute data archives after ~6 months — use a recent window.
    start, end = _recent_utc_window()

    df = provider.fetch_prices("us-east", start, end)

    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["region"] == "us-east").all()
        assert (df["currency"] == "USD").all()
        assert (df["source"] == "pjm_rt_lmp").all()
        assert (df["source_granularity"] == "5min").all()
        assert (df["price_per_mwh"] > -2000).all()
        assert (df["price_per_mwh"] < 100000).all()
        # 5-minute precision must survive (not all timestamps on the hour)
        assert {ts.minute for ts in df["timestamp"]} != {0}


def test_pjm_fetch_rt_hourly_us_east():
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.pjm import PJMRealtimePriceProvider

    provider = PJMRealtimePriceProvider(hourly=True)
    start, end = _recent_utc_window()

    df = provider.fetch_prices("us-east", start, end)

    assert list(df.columns) == PRICE_COLUMNS
    if not df.empty:
        assert (df["source"] == "pjm_rt_lmp").all()
        assert (df["source_granularity"] == "hourly").all()
        assert {ts.minute for ts in df["timestamp"]} == {0}
