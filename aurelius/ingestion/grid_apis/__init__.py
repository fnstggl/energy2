"""Grid data provider package for Aurelius ingestion.

Price providers (real wholesale electricity prices only):
    CAISOPriceProvider         – us-west day-ahead LMP  (CAISO NP15, no auth)
    CAISORealtimePriceProvider – us-west real-time 5-min LMP  (CAISO NP15, no auth)
    PJMPriceProvider           – us-east  (PJM-RTO day-ahead LMP, requires PJM_API_KEY)
    PJMRealtimePriceProvider   – us-east  (PJM-RTO real-time LMP, requires PJM_API_KEY)
    ERCOTPriceProvider         – us-south (ERCOT HB_HOUSTON day-ahead SPP, requires ERCOT creds)
    ERCOTRealtimePriceProvider – us-south (ERCOT HB_HOUSTON real-time 15-min SPP, requires ERCOT creds)
    ENTSOEPriceProvider        – eu-*     (ENTSO-E day-ahead prices, requires ENTSOE_API_KEY)
    CSVPriceImporter           – any region from CSV file

Carbon providers:
    ElectricityMapsCarbonProvider  – requires ELECTRICITYMAPS_API_KEY
    WattTimeCarbonProvider         – requires WATTTIME_USERNAME + WATTTIME_PASSWORD
    CSVCarbonImporter              – any region from CSV file

Deprecated:
    EIAPriceProvider  – Raises UnsupportedMarketPriceError; EIA v2 provides demand
                        (MWh), not wholesale electricity prices ($/MWh).

Market registry:
    MARKET_REGISTRY              – day-ahead metadata per region
    get_registry_entry(region)   – raises UnsupportedMarketPriceError if unknown
    get_price_provider_for_region(region, price_type) – supports real_time_lmp lookup
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
from .caiso import CAISOPriceProvider, CAISORealtimePriceProvider
from .pjm import PJMPriceProvider, PJMRealtimePriceProvider
from .ercot import ERCOTPriceProvider, ERCOTRealtimePriceProvider
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
    "CAISORealtimePriceProvider",
    "PJMPriceProvider",
    "PJMRealtimePriceProvider",
    "ERCOTPriceProvider",
    "ERCOTRealtimePriceProvider",
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
