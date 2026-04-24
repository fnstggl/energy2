"""Market price source registry for Aurelius.

Maps canonical Aurelius regions to authoritative wholesale electricity price
provider metadata. Each entry documents exactly what price type, endpoint,
currency, and known limitations apply.

Usage:
    from aurelius.ingestion.grid_apis.market_registry import (
        get_registry_entry,
        get_price_provider_for_region,
        UnsupportedMarketPriceError,
        MARKET_REGISTRY,
    )

    entry = get_registry_entry("us-west")
    print(entry["hub_or_zone"])  # NP15_7_N001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class UnsupportedMarketPriceError(Exception):
    """Raised when a region has no supported real wholesale price source.

    This is raised (never silently swallowed) so callers know they cannot
    proceed with price-backed backtesting for the requested region.
    """


@dataclass(frozen=True)
class MarketRegistryEntry:
    """Full metadata for a supported market/region price source."""
    canonical_region: str
    market: str
    operator: str
    provider: str            # internal provider key
    price_type: str          # day_ahead | real_time | hub_price | zonal_lmp
    currency: str
    unit: str                # must be "USD/MWh" or "EUR/MWh" (never MWh, MW, etc.)
    granularity: str
    timezone: str
    supported_from: str      # earliest reliable data (ISO date)
    hub_or_zone: str
    endpoint_hint: str       # human-readable endpoint reference
    auth_required: bool
    auth_env_var: Optional[str]
    limitations: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MARKET_REGISTRY: dict[str, MarketRegistryEntry] = {
    "us-west": MarketRegistryEntry(
        canonical_region="us-west",
        market="CAISO",
        operator="California ISO",
        provider="caiso_oasis",
        price_type="day_ahead",
        currency="USD",
        unit="USD/MWh",
        granularity="hourly",
        timezone="US/Pacific",
        supported_from="2010-01-01",
        hub_or_zone="NP15_7_N001",
        endpoint_hint="https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&market_run_id=DAM",
        auth_required=False,
        auth_env_var=None,
        limitations=(
            "NP15 trading-hub day-ahead LMP. CAISO OASIS returns a ZIP/CSV file per request. "
            "Free public access; no API key required. Maximum ~31-day window per request. "
            "Does not cover SP15 (Southern CA) or DLAP zones."
        ),
    ),
    "us-east": MarketRegistryEntry(
        canonical_region="us-east",
        market="PJM",
        operator="PJM Interconnection",
        provider="pjm",
        price_type="day_ahead",
        currency="USD",
        unit="USD/MWh",
        granularity="hourly",
        timezone="US/Eastern",
        supported_from="1998-01-01",
        hub_or_zone="WESTERN HUB (pnode_id=1)",
        endpoint_hint="https://api.pjm.com/api/v1/da_hrl_lmps",
        auth_required=True,
        auth_env_var="PJM_API_KEY",
        limitations=(
            "PJM Western Hub day-ahead hourly LMP. "
            "Requires PJM_API_KEY (free registration at developer.pjm.com). "
            "Ocp-Apim-Subscription-Key header required. "
            "Does not cover nodal prices or other hubs."
        ),
    ),
    "us-south": MarketRegistryEntry(
        canonical_region="us-south",
        market="ERCOT",
        operator="Electric Reliability Council of Texas",
        provider="ercot",
        price_type="real_time",
        currency="USD",
        unit="USD/MWh",
        granularity="15min",
        timezone="US/Central",
        supported_from="2010-01-01",
        hub_or_zone="Houston Hub",
        endpoint_hint="https://api.ercot.com/api/public-reports/",
        auth_required=True,
        auth_env_var="ERCOT_API_KEY",
        limitations=(
            "NOT YET IMPLEMENTED. ERCOT API requires separate registration and uses "
            "settlement-point prices (SPPs) rather than LMPs. Houston Hub is the most "
            "liquid reference point. 15-min granularity; hourly aggregation needed."
        ),
    ),
    "eu-west": MarketRegistryEntry(
        canonical_region="eu-west",
        market="EPEX SPOT",
        operator="ENTSO-E / EnBW TSO",
        provider="entsoe",
        price_type="day_ahead",
        currency="EUR",
        unit="EUR/MWh",
        granularity="hourly",
        timezone="Europe/Berlin",
        supported_from="2015-01-01",
        hub_or_zone="DE bidding zone (EIC: 10YDE-ENBW-----N)",
        endpoint_hint="https://web-api.tp.entsoe.eu/api?documentType=A44",
        auth_required=True,
        auth_env_var="ENTSOE_API_KEY",
        limitations=(
            "Germany EnBW bidding-zone day-ahead prices from ENTSO-E Transparency Platform. "
            "Prices are in EUR/MWh. Requires ENTSOE_API_KEY. "
            "In-sample coverage depends on ENTSO-E data availability (~2015 onward)."
        ),
    ),
    "eu-north": MarketRegistryEntry(
        canonical_region="eu-north",
        market="Nord Pool",
        operator="ENTSO-E / Statnett TSO",
        provider="entsoe",
        price_type="day_ahead",
        currency="EUR",
        unit="EUR/MWh",
        granularity="hourly",
        timezone="Europe/Oslo",
        supported_from="2015-01-01",
        hub_or_zone="Norway NO1 bidding zone (EIC: 10YNO-1--------2)",
        endpoint_hint="https://web-api.tp.entsoe.eu/api?documentType=A44",
        auth_required=True,
        auth_env_var="ENTSOE_API_KEY",
        limitations=(
            "Norway NO1 bidding-zone day-ahead prices from ENTSO-E Transparency Platform. "
            "Prices in EUR/MWh. Requires ENTSOE_API_KEY."
        ),
    ),
    "eu-central": MarketRegistryEntry(
        canonical_region="eu-central",
        market="EPEX SPOT",
        operator="ENTSO-E / RTE TSO",
        provider="entsoe",
        price_type="day_ahead",
        currency="EUR",
        unit="EUR/MWh",
        granularity="hourly",
        timezone="Europe/Paris",
        supported_from="2015-01-01",
        hub_or_zone="France bidding zone (EIC: 10YFR-RTE------C)",
        endpoint_hint="https://web-api.tp.entsoe.eu/api?documentType=A44",
        auth_required=True,
        auth_env_var="ENTSOE_API_KEY",
        limitations=(
            "France bidding-zone day-ahead prices from ENTSO-E Transparency Platform. "
            "Prices in EUR/MWh. Requires ENTSOE_API_KEY."
        ),
    ),
}

# Regions explicitly NOT supported with real price data
_UNSUPPORTED_REGIONS = {
    "us-north": (
        "MISO (us-north) real-time or day-ahead LMP is not yet implemented. "
        "MISO market data requires registration at misoenergy.org. "
        "No public unauthenticated API is available."
    ),
}


def get_registry_entry(region: str) -> MarketRegistryEntry:
    """Return the registry entry for a canonical region.

    Args:
        region: Canonical Aurelius region (e.g. "us-west", "eu-west").

    Returns:
        MarketRegistryEntry with full provider metadata.

    Raises:
        UnsupportedMarketPriceError: If the region has no supported price source.
    """
    entry = MARKET_REGISTRY.get(region)
    if entry is not None:
        return entry
    reason = _UNSUPPORTED_REGIONS.get(region, f"Region '{region}' is not in the market registry.")
    raise UnsupportedMarketPriceError(
        f"No supported real wholesale price source for region '{region}'. {reason}"
    )


def get_price_provider_for_region(region: str, price_type: Optional[str] = None) -> MarketRegistryEntry:
    """Return registry metadata for a region, optionally filtered by price_type.

    Args:
        region:     Canonical Aurelius region identifier.
        price_type: Optional filter (e.g. "day_ahead", "real_time").
                    If provided and does not match, raises UnsupportedMarketPriceError.

    Returns:
        MarketRegistryEntry.

    Raises:
        UnsupportedMarketPriceError: Region not supported, or price_type mismatch.
    """
    entry = get_registry_entry(region)
    if price_type is not None and entry.price_type != price_type:
        raise UnsupportedMarketPriceError(
            f"Region '{region}' has price_type='{entry.price_type}', "
            f"not the requested '{price_type}'. "
            f"Use price_type='{entry.price_type}' or None to accept any type."
        )
    return entry


def list_supported_regions() -> list[str]:
    """Return all canonical regions with at least one supported price source."""
    return list(MARKET_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# These field names indicate demand/load/generation data — never acceptable
# as price_per_mwh values.
_REJECTED_DATA_TYPE_KEYWORDS = frozenset({
    "demand", "load", "generation", "consumption",
    "net_generation", "interchange", "forecast_demand",
})


def assert_price_type_not_demand(data_type_label: str) -> None:
    """Raise ValueError if a data type label indicates demand/load data.

    Args:
        data_type_label: Lower-cased label from the data source (e.g. "demand").

    Raises:
        ValueError: If the label is in the rejected set.
    """
    label = data_type_label.lower().strip()
    for keyword in _REJECTED_DATA_TYPE_KEYWORDS:
        if keyword in label:
            raise ValueError(
                f"Data type '{data_type_label}' is demand/load/generation data and "
                f"must NOT be mapped to price_per_mwh. "
                f"Use a real wholesale electricity price endpoint."
            )
