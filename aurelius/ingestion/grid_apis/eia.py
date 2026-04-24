"""EIA (U.S. Energy Information Administration) API v2 adapter.

IMPORTANT — DATA AVAILABILITY NOTICE:
    The EIA API v2 /electricity/rto/region-data/ endpoint provides:
        D  = Demand (MWh load)         ← NOT a price
        NG = Net Generation (MWh)      ← NOT a price
        TI = Total Interchange (MWh)   ← NOT a price

    EIA API v2 does NOT provide hourly day-ahead or real-time LMP/wholesale
    electricity prices. For authoritative hourly wholesale prices use:
        CAISO (us-west):   CAISOPriceProvider  (OASIS API, no key needed)
        PJM   (us-east):   PJMPriceProvider    (Data Miner API, requires PJM_API_KEY)
        ENTSOE (eu-*):     ENTSOEPriceProvider (ENTSO-E, requires ENTSOE_API_KEY)

    This module is retained for informational/demand-signal uses only.
    The EIAPriceProvider class raises UnsupportedMarketPriceError when called
    because it cannot provide true wholesale electricity price data.

Environment variable:
    EIA_API_KEY  –  register free at https://www.eia.gov/opendata/
"""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
)
from .market_registry import UnsupportedMarketPriceError

logger = logging.getLogger(__name__)


class EIAPriceProvider(PriceProvider):
    """EIA API v2 price provider – NOT suitable for wholesale price data.

    EIA /electricity/rto/region-data/ provides regional demand (MWh), NOT prices
    ($/MWh). This adapter explicitly refuses to map demand data to price_per_mwh.

    For real wholesale electricity price data use:
        CAISOPriceProvider  – us-west (CAISO NP15 day-ahead LMP)
        PJMPriceProvider    – us-east (PJM Western Hub day-ahead LMP)
        ENTSOEPriceProvider – eu-west, eu-central, eu-north

    Raises:
        UnsupportedMarketPriceError: Always, for all regions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("EIA_API_KEY", "")
        self._region_map = region_map or {}

    @property
    def source_name(self) -> str:
        return "eia_v2"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Raises UnsupportedMarketPriceError — EIA does not provide hourly LMP data.

        EIA API v2 RTO data endpoints provide demand (MWh), generation (MWh),
        and interchange (MWh). None of these are wholesale electricity prices.
        Mapping demand/load to price_per_mwh produces meaningless cost estimates.

        Use CAISOPriceProvider, PJMPriceProvider, or ENTSOEPriceProvider instead.

        Raises:
            UnsupportedMarketPriceError: Always.
        """
        raise UnsupportedMarketPriceError(
            f"EIAPriceProvider does not supply real wholesale electricity prices for "
            f"region='{region}'. "
            "EIA API v2 /electricity/rto/region-data/ provides demand (MWh), not "
            "prices ($/MWh). "
            "Use CAISOPriceProvider for us-west, PJMPriceProvider for us-east, "
            "or ENTSOEPriceProvider for EU regions."
        )
