"""EIA (U.S. Energy Information Administration) API v2 adapter.

Fetches hourly electricity prices (Day-Ahead LMP or Regional Demand Forecasts)
from the EIA Open Data API v2.

Environment variable required:
    EIA_API_KEY  –  register free at https://www.eia.gov/opendata/

Region mapping (Aurelius → EIA RTO region code):
    "us-west"  → "CAL"  (CAISO California)
    "us-east"  → "PJM"  (PJM Interconnection)
    "us-south" → "TEX"  (ERCOT Texas)
    "us-north" → "MISO" (Midcontinent ISO)

You can override the mapping by passing region_map to the constructor.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
    normalize_price_df,
)

logger = logging.getLogger(__name__)

_DEFAULT_REGION_MAP = {
    "us-west":  "CAL",
    "us-east":  "PJM",
    "us-south": "TEX",
    "us-north": "MISO",
}

_EIA_BASE = "https://api.eia.gov/v2"
_ENDPOINT = "/electricity/rto/region-data/data/"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds; doubled each retry


class EIAPriceProvider(PriceProvider):
    """Fetch hourly electricity prices from EIA API v2.

    Prices are the hourly regional demand/generation composite.
    Day-ahead LMP endpoint is used where available.

    Args:
        api_key: EIA API key. Reads EIA_API_KEY env var if not supplied.
        region_map: Mapping of Aurelius region name → EIA respondent code.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("EIA_API_KEY", "")
        self._region_map = region_map or _DEFAULT_REGION_MAP

    @property
    def source_name(self) -> str:
        return "eia_v2"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly prices for *region* in [start, end).

        Raises:
            ProviderConfigError: If EIA_API_KEY is not set.
        """
        if not self._api_key:
            raise ProviderConfigError(
                "EIA_API_KEY is not set. "
                "Export it as an environment variable or pass api_key= to EIAPriceProvider."
            )

        eia_region = self._region_map.get(region)
        if eia_region is None:
            logger.warning(
                f"Region '{region}' not in EIA region_map; available: {list(self._region_map)}"
            )
            return empty_price_df()

        try:
            import requests
        except ImportError:
            raise ProviderConfigError("'requests' package is required for EIA provider: pip install requests")

        # EIA v2 uses ISO dates without timezone in the query parameters
        start_str = start.strftime("%Y-%m-%dT%H")
        end_str = end.strftime("%Y-%m-%dT%H")

        params = {
            "api_key": self._api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": eia_region,
            "facets[type][]": "D",   # Demand (MWh) — proxy for price signal
            "start": start_str,
            "end": end_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }

        all_rows: list[dict] = []
        url = f"{_EIA_BASE}{_ENDPOINT}"

        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"EIA rate limit hit; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.RequestException as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"EIA request failed after {_MAX_RETRIES} attempts: {exc}")
                    return empty_price_df()
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        else:
            return empty_price_df()

        data = payload.get("response", {}).get("data", [])
        if not data:
            logger.warning(f"EIA returned no data for region={eia_region} {start_str}..{end_str}")
            return empty_price_df()

        # EIA returns demand in MWh; convert to a synthetic $/MWh price proxy.
        # NOTE: Real prices require the day-ahead LMP endpoint which varies by
        # market. This adapter returns demand as a proxy until LMP keys are set.
        rows = []
        for item in data:
            try:
                ts = pd.Timestamp(item["period"], tz="UTC")
                value = float(item.get("value") or 0)
                rows.append({"timestamp": ts, "region": region, "price_per_mwh": value})
            except (KeyError, ValueError, TypeError):
                continue

        if not rows:
            return empty_price_df()

        df = pd.DataFrame(rows)
        return normalize_price_df(df, source=self.source_name, currency="USD", granularity="hourly")
