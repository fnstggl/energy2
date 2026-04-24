"""ElectricityMaps carbon intensity adapter.

Fetches historical carbon intensity (gCO₂eq/kWh) from the ElectricityMaps API.

Environment variable required:
    ELECTRICITYMAPS_API_KEY  –  register at https://api.electricitymap.org

Zone mapping (Aurelius → ElectricityMaps zone key):
    "us-west"  → "US-CAL-CISO"
    "us-east"  → "US-MIDA-PJM"
    "us-south" → "US-TEX-ERCO"
    "eu-west"  → "DE"
    "eu-north" → "NO-NO1"

Override with zone_map= constructor argument.

Rate limits:
    Free tier: 30 requests/min. The adapter inserts a short sleep between
    pages and retries with exponential back-off on 429 responses.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .base import (
    CarbonProvider,
    ProviderConfigError,
    empty_carbon_df,
    normalize_carbon_df,
)

logger = logging.getLogger(__name__)

_DEFAULT_ZONE_MAP = {
    "us-west":  "US-CAL-CISO",
    "us-east":  "US-MIDA-PJM",
    "us-south": "US-TEX-ERCO",
    "eu-west":  "DE",
    "eu-north": "NO-NO1",
}

_EM_BASE = "https://api.electricitymap.org/v3"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_PAGE_SLEEP = 0.5      # seconds between paginated requests
_MAX_HISTORY_DAYS = 90  # free tier limit; paid tiers allow more


class ElectricityMapsCarbonProvider(CarbonProvider):
    """Fetch historical carbon intensity from ElectricityMaps.

    Args:
        api_key: ElectricityMaps auth token.
        zone_map: Mapping of Aurelius region → ElectricityMaps zone key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        zone_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ELECTRICITYMAPS_API_KEY", "")
        self._zone_map = zone_map or _DEFAULT_ZONE_MAP

    @property
    def source_name(self) -> str:
        return "electricitymaps"

    def fetch_carbon(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly carbon intensity for *region* in [start, end).

        Raises:
            ProviderConfigError: If ELECTRICITYMAPS_API_KEY is not set.
        """
        if not self._api_key:
            raise ProviderConfigError(
                "ELECTRICITYMAPS_API_KEY is not set. "
                "Export it or pass api_key= to ElectricityMapsCarbonProvider."
            )

        zone = self._zone_map.get(region)
        if zone is None:
            logger.warning(f"Region '{region}' not in ElectricityMaps zone map; available: {list(self._zone_map)}")
            return empty_carbon_df()

        try:
            import requests
        except ImportError:
            raise ProviderConfigError("'requests' package is required: pip install requests")

        # Ensure UTC-aware datetimes
        def _to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        headers = {"auth-token": self._api_key}
        all_rows: list[dict] = []

        # ElectricityMaps /past-range supports up to ~10-day windows per call.
        # We chunk into ≤10-day windows and paginate.
        chunk_days = 10
        current = start_utc
        while current < end_utc:
            chunk_end = min(current + timedelta(days=chunk_days), end_utc)
            params = {
                "zone": zone,
                "start": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            for attempt in range(_MAX_RETRIES):
                try:
                    resp = requests.get(
                        f"{_EM_BASE}/carbon-intensity/past-range",
                        params=params,
                        headers=headers,
                        timeout=30,
                    )
                    if resp.status_code == 429:
                        wait = _RETRY_BACKOFF * (2 ** attempt)
                        logger.warning(f"ElectricityMaps rate limit; retrying in {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    if resp.status_code in (401, 403):
                        raise ProviderConfigError(
                            f"ElectricityMaps API key rejected ({resp.status_code}). "
                            "Check ELECTRICITYMAPS_API_KEY."
                        )
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except ProviderConfigError:
                    raise
                except Exception as exc:
                    if attempt == _MAX_RETRIES - 1:
                        logger.error(f"ElectricityMaps request failed: {exc}")
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            else:
                current = chunk_end
                time.sleep(_PAGE_SLEEP)
                continue

            for item in payload.get("history", []):
                try:
                    ts = pd.Timestamp(item["datetime"], tz="UTC")
                    ci = float(item.get("carbonIntensity") or 0)
                    all_rows.append({"timestamp": ts, "region": region, "gco2_per_kwh": ci})
                except (KeyError, ValueError, TypeError):
                    continue

            current = chunk_end
            time.sleep(_PAGE_SLEEP)

        if not all_rows:
            logger.warning(f"ElectricityMaps returned no data for zone={zone} {start_utc}..{end_utc}")
            return empty_carbon_df()

        df = pd.DataFrame(all_rows)
        return normalize_carbon_df(df, source=self.source_name, granularity="hourly")
