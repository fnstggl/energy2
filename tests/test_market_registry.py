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
    _REAL_TIME_REGISTRY,
)


# ---------------------------------------------------------------------------
# Day-ahead registry entries
# ---------------------------------------------------------------------------

class TestRegistryEntries:
    def test_us_west_resolves_to_caiso(self):
        entry = get_registry_entry("us-west")
        assert entry.provider == "caiso_oasis"
        assert entry.auth_required is False
        assert entry.auth_env_var is None

    def test_us_west_node_is_np15_trading_hub(self):
        """us-west must use TH_NP15_GEN-APND — the standard CAISO NP15 trading hub."""
        entry = get_registry_entry("us-west")
        assert entry.hub_or_zone == "TH_NP15_GEN-APND"

    def test_us_west_timezone_is_america_los_angeles(self):
        entry = get_registry_entry("us-west")
        assert entry.timezone == "America/Los_Angeles"

    def test_us_west_price_type_is_day_ahead_lmp(self):
        entry = get_registry_entry("us-west")
        assert entry.price_type == "day_ahead_lmp"

    def test_us_west_endpoint_hint_has_resultformat_6(self):
        """resultformat=6 must be in the endpoint hint (ZIP/CSV output)."""
        entry = get_registry_entry("us-west")
        assert "resultformat=6" in entry.endpoint_hint

    def test_us_west_endpoint_hint_uses_prc_lmp_dam(self):
        entry = get_registry_entry("us-west")
        assert "PRC_LMP" in entry.endpoint_hint
        assert "DAM" in entry.endpoint_hint

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

    def test_caiso_is_not_universal_price_source(self):
        """CAISO covers us-west only; other regions must use their own providers."""
        west = get_registry_entry("us-west")
        east = get_registry_entry("us-east")
        assert west.provider != east.provider, "CAISO must not be the universal price source"
        assert east.provider == "pjm"


# ---------------------------------------------------------------------------
# Real-time registry entries
# ---------------------------------------------------------------------------

class TestRealTimeRegistry:
    def test_us_west_has_real_time_entry(self):
        rt = get_price_provider_for_region("us-west", price_type="real_time_lmp")
        assert rt.provider == "caiso_oasis"
        assert rt.price_type == "real_time_lmp"

    def test_us_west_real_time_granularity_is_5min(self):
        rt = get_price_provider_for_region("us-west", price_type="real_time_lmp")
        assert rt.granularity == "5min"

    def test_us_west_real_time_node_is_np15_trading_hub(self):
        rt = get_price_provider_for_region("us-west", price_type="real_time_lmp")
        assert rt.hub_or_zone == "TH_NP15_GEN-APND"

    def test_us_west_real_time_endpoint_uses_prc_intvl_lmp(self):
        rt = get_price_provider_for_region("us-west", price_type="real_time_lmp")
        assert "PRC_INTVL_LMP" in rt.endpoint_hint
        assert "RTM" in rt.endpoint_hint
        assert "resultformat=6" in rt.endpoint_hint

    def test_us_west_real_time_auth_not_required(self):
        rt = get_price_provider_for_region("us-west", price_type="real_time_lmp")
        assert rt.auth_required is False
        assert rt.auth_env_var is None

    def test_real_time_registry_unit_is_usd_per_mwh(self):
        for region, entry in _REAL_TIME_REGISTRY.items():
            assert entry.unit in {"USD/MWh", "EUR/MWh"}, (
                f"Real-time entry for '{region}' has bad unit '{entry.unit}'"
            )

    def test_real_time_registry_price_type_not_demand(self):
        forbidden = {"demand", "load", "generation", "consumption"}
        for region, entry in _REAL_TIME_REGISTRY.items():
            for word in forbidden:
                assert word not in entry.price_type.lower()

    def test_rtm_alias_accepted(self):
        """'rtm' is an accepted alias for real_time_lmp."""
        rt = get_price_provider_for_region("us-west", price_type="rtm")
        assert rt.price_type == "real_time_lmp"

    def test_real_time_alias_accepted(self):
        """'real_time' is an accepted alias for real_time_lmp."""
        rt = get_price_provider_for_region("us-west", price_type="real_time")
        assert rt.price_type == "real_time_lmp"

    def test_us_east_has_real_time_entry(self):
        """PJM now exposes a real-time five-minute LMP entry for us-east."""
        rt = get_price_provider_for_region("us-east", price_type="real_time_lmp")
        assert rt.provider == "pjm"
        assert rt.price_type == "real_time_lmp"
        assert rt.granularity == "5min"
        assert rt.auth_required is True
        assert rt.auth_env_var == "PJM_API_KEY"
        assert "rt_fivemin_hrl_lmps" in rt.endpoint_hint


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
        """Requesting a price_type that doesn't match any entry raises."""
        with pytest.raises(UnsupportedMarketPriceError):
            # eu-west has day_ahead_lmp only — no real-time entry in the RT registry
            get_price_provider_for_region("eu-west", price_type="real_time_lmp")

    def test_get_price_provider_matching_day_ahead_type_returns_entry(self):
        entry = get_price_provider_for_region("us-west", price_type="day_ahead_lmp")
        assert entry.provider == "caiso_oasis"

    def test_get_price_provider_no_type_filter_returns_entry(self):
        entry = get_price_provider_for_region("us-east")
        assert entry.provider == "pjm"

    def test_unsupported_price_type_string_raises(self):
        """A completely unknown price_type must raise, not silently return."""
        with pytest.raises(UnsupportedMarketPriceError):
            get_price_provider_for_region("us-west", price_type="spot_price_unknown")


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
        "day_ahead_lmp",
        "real_time_lmp",
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
            assert result is None or (hasattr(result, "empty") and result.empty), (
                "EIAPriceProvider returned a non-empty DataFrame — this means demand/load "
                "data was silently mapped to price_per_mwh, which is WRONG."
            )
        except UnsupportedMarketPriceError:
            pass  # expected
