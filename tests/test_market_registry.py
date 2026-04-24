"""Tests for the market price source registry and demand-guard helpers."""

import pytest

from aurelius.ingestion.grid_apis.market_registry import (
    MARKET_REGISTRY,
    MarketRegistryEntry,
    UnsupportedMarketPriceError,
    assert_price_type_not_demand,
    get_price_provider_for_region,
    get_registry_entry,
    list_supported_regions,
)


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------

class TestRegistryEntries:
    def test_us_west_resolves_to_caiso(self):
        entry = get_registry_entry("us-west")
        assert entry.provider == "caiso_oasis"
        assert entry.auth_required is False
        assert entry.auth_env_var is None

    def test_us_east_resolves_to_pjm(self):
        entry = get_registry_entry("us-east")
        assert entry.provider == "pjm"
        assert entry.auth_required is True
        assert entry.auth_env_var == "PJM_API_KEY"

    def test_eu_regions_resolve_to_entsoe(self):
        for region in ("eu-west", "eu-north", "eu-central"):
            entry = get_registry_entry(region)
            assert entry.provider == "entsoe", f"{region} should use entsoe"
            assert entry.auth_env_var == "ENTSOE_API_KEY"

    def test_every_registry_entry_has_price_unit(self):
        """Every entry must document a real price unit, never raw energy."""
        allowed_units = {"USD/MWh", "EUR/MWh"}
        for region, entry in MARKET_REGISTRY.items():
            assert entry.unit in allowed_units, (
                f"Registry entry for '{region}' has unit='{entry.unit}' "
                f"which is not a price unit. Demand/load units (MWh, MW) are forbidden."
            )

    def test_every_registry_entry_has_currency(self):
        for region, entry in MARKET_REGISTRY.items():
            assert entry.currency in ("USD", "EUR"), (
                f"Entry for '{region}' has unexpected currency '{entry.currency}'"
            )

    def test_every_registry_entry_has_price_type_not_demand(self):
        """price_type field must not contain 'demand', 'load', 'generation', or 'consumption'."""
        forbidden = {"demand", "load", "generation", "consumption"}
        for region, entry in MARKET_REGISTRY.items():
            for word in forbidden:
                assert word not in entry.price_type.lower(), (
                    f"Entry for '{region}' has price_type='{entry.price_type}' "
                    f"which contains forbidden word '{word}'"
                )

    def test_all_entries_are_frozen_dataclasses(self):
        for region, entry in MARKET_REGISTRY.items():
            assert isinstance(entry, MarketRegistryEntry)
            with pytest.raises((AttributeError, TypeError)):
                entry.unit = "broken"  # frozen dataclass should reject mutation

    def test_list_supported_regions_returns_all_keys(self):
        regions = list_supported_regions()
        assert set(regions) == set(MARKET_REGISTRY.keys())
        assert len(regions) >= 5  # at minimum: us-west, us-east, us-south, eu-west, eu-north


# ---------------------------------------------------------------------------
# UnsupportedMarketPriceError
# ---------------------------------------------------------------------------

class TestUnsupportedMarketPriceError:
    def test_unknown_region_raises(self):
        with pytest.raises(UnsupportedMarketPriceError):
            get_registry_entry("xx-unknown")

    def test_us_north_raises_not_yet_implemented(self):
        """us-north (MISO) is explicitly listed as unsupported."""
        with pytest.raises(UnsupportedMarketPriceError, match="us-north"):
            get_registry_entry("us-north")

    def test_error_message_includes_region(self):
        with pytest.raises(UnsupportedMarketPriceError, match="xx-bad-region"):
            get_registry_entry("xx-bad-region")

    def test_get_price_provider_price_type_mismatch_raises(self):
        """Requesting a price_type that doesn't match the registry entry raises."""
        with pytest.raises(UnsupportedMarketPriceError, match="real_time"):
            get_price_provider_for_region("us-west", price_type="real_time")

    def test_get_price_provider_matching_type_returns_entry(self):
        entry = get_price_provider_for_region("us-west", price_type="day_ahead")
        assert entry.provider == "caiso_oasis"

    def test_get_price_provider_no_type_filter_returns_entry(self):
        entry = get_price_provider_for_region("us-east")
        assert entry.provider == "pjm"


# ---------------------------------------------------------------------------
# assert_price_type_not_demand
# ---------------------------------------------------------------------------

class TestAssertPriceTypeNotDemand:
    @pytest.mark.parametrize("label", [
        "demand",
        "load",
        "net_generation",
        "generation",
        "consumption",
        "forecast_demand",
        "DEMAND",       # case-insensitive
        "total_load",
    ])
    def test_demand_labels_rejected(self, label):
        with pytest.raises(ValueError, match="demand|load|generation|consumption"):
            assert_price_type_not_demand(label)

    @pytest.mark.parametrize("label", [
        "day_ahead",
        "lmp",
        "total_lmp_da",
        "price_per_mwh",
        "da_lmp",
        "hub_price",
    ])
    def test_price_labels_accepted(self, label):
        assert_price_type_not_demand(label)  # must not raise


# ---------------------------------------------------------------------------
# EIA cannot supply price data (via UnsupportedMarketPriceError)
# ---------------------------------------------------------------------------

class TestEIANotAPrice:
    def test_eia_price_provider_raises_unsupported(self):
        """EIAPriceProvider must raise UnsupportedMarketPriceError, not return demand data."""
        import pandas as pd
        from datetime import datetime, timezone
        from aurelius.ingestion.grid_apis.eia import EIAPriceProvider

        provider = EIAPriceProvider(api_key="any-key")
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(UnsupportedMarketPriceError):
            provider.fetch_prices("us-west", t0, t0 + pd.Timedelta(hours=24))

    def test_eia_never_returns_demand_as_price(self):
        """Ensure fetch_prices never silently returns a DataFrame (demand as price)."""
        import pandas as pd
        from datetime import datetime, timezone
        from aurelius.ingestion.grid_apis.eia import EIAPriceProvider

        provider = EIAPriceProvider()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        try:
            result = provider.fetch_prices("us-east", t0, t0 + pd.Timedelta(hours=1))
            # If we reach here, it must NOT be a non-empty DataFrame
            assert result is None or (hasattr(result, "empty") and result.empty), (
                "EIAPriceProvider returned a non-empty DataFrame — this means demand/load "
                "data was silently mapped to price_per_mwh, which is WRONG."
            )
        except UnsupportedMarketPriceError:
            pass  # expected
