"""Grid data provider package for Aurelius ingestion.

Price providers (real wholesale electricity prices only):
    CAISOPriceProvider     – us-west  (CAISO NP15 day-ahead LMP, no auth)
    PJMPriceProvider       – us-east  (PJM Western Hub DA LMP, requires PJM_API_KEY)
    ENTSOEPriceProvider    – eu-*     (ENTSO-E day-ahead prices, requires ENTSOE_API_KEY)
    CSVPriceImporter       – any region from CSV file

Carbon providers:
    ElectricityMapsCarbonProvider  – requires ELECTRICITYMAPS_API_KEY
    WattTimeCarbonProvider         – requires WATTTIME_USERNAME + WATTTIME_PASSWORD
    CSVCarbonImporter              – any region from CSV file

Deprecated:
    EIAPriceProvider  – Raises UnsupportedMarketPriceError; EIA v2 provides demand
                        (MWh), not wholesale electricity prices ($/MWh).

Market registry:
    MARKET_REGISTRY              – full metadata per region
    get_registry_entry(region)   – raises UnsupportedMarketPriceError if unknown
    UnsupportedMarketPriceError  – raised for unsupported regions
"""

from .base import (
    PriceProvider,
    CarbonProvider,
    ProviderConfigError,
    PRICE_COLUMNS,
    CARBON_COLUMNS,
    empty_price_df,
    empty_carbon_df,
    normalize_price_df,
    normalize_carbon_df,
)
from .csv_importer import CSVPriceImporter, CSVCarbonImporter
from .eia import EIAPriceProvider
from .entsoe import ENTSOEPriceProvider
from .electricitymaps import ElectricityMapsCarbonProvider
from .caiso import CAISOPriceProvider
from .pjm import PJMPriceProvider
from .watttime import WattTimeCarbonProvider
from .market_registry import (
    MARKET_REGISTRY,
    MarketRegistryEntry,
    UnsupportedMarketPriceError,
    get_registry_entry,
    get_price_provider_for_region,
    list_supported_regions,
    assert_price_type_not_demand,
)

__all__ = [
    "PriceProvider",
    "CarbonProvider",
    "ProviderConfigError",
    "PRICE_COLUMNS",
    "CARBON_COLUMNS",
    "empty_price_df",
    "empty_carbon_df",
    "normalize_price_df",
    "normalize_carbon_df",
    "CSVPriceImporter",
    "CSVCarbonImporter",
    "CAISOPriceProvider",
    "PJMPriceProvider",
    "ENTSOEPriceProvider",
    "EIAPriceProvider",
    "ElectricityMapsCarbonProvider",
    "WattTimeCarbonProvider",
    "MARKET_REGISTRY",
    "MarketRegistryEntry",
    "UnsupportedMarketPriceError",
    "get_registry_entry",
    "get_price_provider_for_region",
    "list_supported_regions",
    "assert_price_type_not_demand",
]
