"""Tests for the provenance-aware market-data provider abstraction."""

from datetime import datetime, timezone

import pytest

from aurelius.ingestion.market_data_provider import (
    BenchmarkDataError,
    CarbonPoint,
    MarketPricePoint,
    MarketType,
    Provenance,
    ProviderCapability,
    Signal,
    assert_benchmark_admissible,
    filter_benchmark_admissible,
    points_to_carbon_df,
    points_to_price_df,
)
from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS, CARBON_COLUMNS

UTC = timezone.utc
T0 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Provenance / point invariants
# ---------------------------------------------------------------------------

class TestProvenanceFields:
    def test_price_point_default_provenance_is_source_of_truth(self):
        p = MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0)
        assert p.provenance == Provenance.SOURCE_OF_TRUTH
        assert p.is_sandbox is False
        assert p.benchmark_admissible is True

    def test_sandbox_flag_forces_sandbox_provenance(self):
        p = MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0, is_sandbox=True)
        assert p.provenance == Provenance.SANDBOX
        assert p.benchmark_admissible is False

    def test_sandbox_provenance_forces_flag(self):
        c = CarbonPoint(timestamp=T0, region="us-west", gco2_per_kwh=200.0,
                        provenance=Provenance.SANDBOX)
        assert c.is_sandbox is True
        assert c.benchmark_admissible is False

    def test_estimated_provenance_sets_estimated_flag(self):
        c = CarbonPoint(timestamp=T0, region="us-west", gco2_per_kwh=200.0,
                        provenance=Provenance.ESTIMATED)
        assert c.is_estimated is True
        # estimated is not sandbox, but also not benchmark-admissible? It is real
        # data that is modelled; we allow it but it is NOT source-of-truth.
        assert c.is_sandbox is False

    def test_invalid_provenance_rejected(self):
        with pytest.raises(ValueError):
            MarketPricePoint(timestamp=T0, region="x", price_per_mwh=1.0, provenance="bogus")

    def test_invalid_market_type_rejected(self):
        with pytest.raises(ValueError):
            MarketPricePoint(timestamp=T0, region="x", price_per_mwh=1.0, market_type="bogus")


# ---------------------------------------------------------------------------
# Benchmark gate
# ---------------------------------------------------------------------------

class TestBenchmarkGate:
    def test_assert_passes_for_real_data(self):
        pts = [MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0)]
        assert_benchmark_admissible(pts)  # should not raise

    def test_assert_rejects_sandbox_data(self):
        pts = [
            MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0),
            MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=99.0, is_sandbox=True),
        ]
        with pytest.raises(BenchmarkDataError):
            assert_benchmark_admissible(pts)

    def test_filter_drops_sandbox_only(self):
        good = MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0)
        bad = MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=99.0, is_sandbox=True)
        kept = filter_benchmark_admissible([good, bad])
        assert kept == [good]


# ---------------------------------------------------------------------------
# Capability schema
# ---------------------------------------------------------------------------

class TestCapabilitySchema:
    def test_capability_valid(self):
        cap = ProviderCapability(
            provider="caiso", signal=Signal.PRICE, regions=("us-west",),
            granularity="hourly", history_supported=True, forecast_supported=False,
            sandbox_supported=False, production_supported=True, auth_required=False,
            market_types=(MarketType.DAY_AHEAD_LMP,),
        )
        assert cap.supports_region("us-west")
        assert not cap.supports_region("us-east")

    def test_capability_rejects_unknown_signal(self):
        with pytest.raises(ValueError):
            ProviderCapability(
                provider="x", signal="weather", regions=(), granularity="hourly",
                history_supported=True, forecast_supported=False,
                sandbox_supported=False, production_supported=True, auth_required=False,
            )


# ---------------------------------------------------------------------------
# Conversion to canonical DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameConversion:
    def test_price_df_keeps_canonical_columns_first(self):
        pts = [MarketPricePoint(timestamp=T0, region="us-west", price_per_mwh=40.0,
                                provider="caiso", source="CAISO")]
        df = points_to_price_df(pts)
        assert df.columns[:len(PRICE_COLUMNS)].tolist() == PRICE_COLUMNS
        assert "is_sandbox" in df.columns and "provenance" in df.columns
        assert df.iloc[0]["price_per_mwh"] == 40.0

    def test_carbon_df_keeps_canonical_columns_first(self):
        pts = [CarbonPoint(timestamp=T0, region="us-west", gco2_per_kwh=200.0,
                           provider="electricitymaps")]
        df = points_to_carbon_df(pts)
        assert df.columns[:len(CARBON_COLUMNS)].tolist() == CARBON_COLUMNS
        assert bool(df.iloc[0]["is_sandbox"]) is False

    def test_empty_points_yield_empty_canonical_df(self):
        assert list(points_to_price_df([]).columns) == PRICE_COLUMNS
        assert list(points_to_carbon_df([]).columns) == CARBON_COLUMNS
