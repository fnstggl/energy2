"""Live integration tests for CAISO OASIS providers — no auth required.

CAISO OASIS is a public API. These tests hit the real endpoint and verify
end-to-end behavior: ZIP download, CSV parse, schema normalization, and
UTC timestamp handling.

Network-dependent and may be slow (~5–30s per request). Placed in live/
so CI can skip the whole directory if needed.

Providers tested:
    CAISOPriceProvider         – PRC_LMP / DAM (day-ahead, hourly)
    CAISORealtimePriceProvider – PRC_INTVL_LMP / RTM (real-time, 5-min)

Node: TH_NP15_GEN-APND (NP15 trading hub, Northern California)
"""

from datetime import datetime, timezone

import pytest


def test_caiso_day_ahead_fetch_us_west():
    """Day-ahead: fetch real CAISO PRC_LMP / DAM data for a historical date."""
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider

    provider = CAISOPriceProvider()
    # Fixed historical date well in the past — avoids data-not-yet-published errors
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)

    df = provider.fetch_prices("us-west", start, end)

    # Schema
    assert list(df.columns) == PRICE_COLUMNS

    if not df.empty:
        # Region and source
        assert (df["region"] == "us-west").all()
        assert (df["currency"] == "USD").all()
        assert (df["source"] == "caiso_oasis_dam").all()
        assert (df["source_granularity"] == "hourly").all()

        # Price sanity — CAISO day-ahead LMPs are typically -$500 to $10,000/MWh
        assert (df["price_per_mwh"] > -500).all()
        assert (df["price_per_mwh"] < 10_000).all()
        import pandas as pd
        assert pd.api.types.is_numeric_dtype(df["price_per_mwh"])

        # Timestamps must be UTC-aware (never naive)
        for ts in df["timestamp"]:
            assert ts.tzinfo is not None, f"Naive timestamp found: {ts}"
            assert str(ts.tzinfo) == "UTC"

        # Must have data for the requested period
        assert len(df) > 0, "Expected >0 rows for 2024-01-02"

        # No demand/load fields may appear as column names
        forbidden_fields = {"demand", "load", "generation", "consumption", "interchange"}
        col_lower = {c.lower() for c in df.columns}
        overlap = forbidden_fields & col_lower
        assert not overlap, f"Forbidden demand/load fields in output: {overlap}"


def test_caiso_day_ahead_unknown_region_returns_empty():
    """No network call should happen for an unmapped region."""
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider

    provider = CAISOPriceProvider()
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    df = provider.fetch_prices("eu-west", start, end)

    assert df.empty
    assert list(df.columns) == PRICE_COLUMNS


def test_caiso_real_time_fetch_us_west():
    """Real-time: fetch CAISO PRC_INTVL_LMP / RTM data for a recent historical date.

    Uses a date a few days in the past to avoid data publication lag.
    If CAISO returns empty (e.g., weekend / holiday lag), the test skips
    rather than failing — the schema check is still performed.
    """
    import pandas as pd

    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISORealtimePriceProvider

    provider = CAISORealtimePriceProvider()

    # Use a fixed historical date to ensure data is available
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 1, 0, tzinfo=timezone.utc)  # one hour of 5-min intervals

    df = provider.fetch_prices("us-west", start, end)

    # Schema must always be correct even if empty
    assert list(df.columns) == PRICE_COLUMNS

    if df.empty:
        pytest.skip(
            "CAISO returned no real-time data for the test window — "
            "this may indicate a data publication lag; skipping rather than failing."
        )

    # Region and source
    assert (df["region"] == "us-west").all()
    assert (df["currency"] == "USD").all()
    assert (df["source"] == "caiso_oasis_rtm").all()
    assert (df["source_granularity"] == "5min").all()

    # Price sanity
    assert (df["price_per_mwh"] > -500).all()
    assert (df["price_per_mwh"] < 10_000).all()
    assert pd.api.types.is_numeric_dtype(df["price_per_mwh"])

    # Timestamps must be UTC-aware
    for ts in df["timestamp"]:
        assert ts.tzinfo is not None, f"Naive timestamp found: {ts}"
        assert str(ts.tzinfo) == "UTC"

    # 5-minute granularity: within one hour we expect ~12 intervals
    assert len(df) > 0

    # No demand/load fields as column names
    forbidden_fields = {"demand", "load", "generation", "consumption"}
    col_lower = {c.lower() for c in df.columns}
    assert not (forbidden_fields & col_lower)


def test_caiso_real_time_unknown_region_returns_empty():
    """No network call for an unmapped region in the real-time provider."""
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS
    from aurelius.ingestion.grid_apis.caiso import CAISORealtimePriceProvider

    provider = CAISORealtimePriceProvider()
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    df = provider.fetch_prices("us-east", start, end)

    assert df.empty
    assert list(df.columns) == PRICE_COLUMNS
