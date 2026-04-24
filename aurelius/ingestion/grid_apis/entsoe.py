"""ENTSO-E Transparency Platform adapter.

Fetches day-ahead electricity prices (Document Type A44) for European
bidding zones.

Environment variable required:
    ENTSOE_API_KEY  –  register at https://transparency.entsoe.eu/

Bidding zone mapping (Aurelius → ENTSO-E EIC code):
    "eu-west"    → "10YDE-ENBW-----N"  (Germany, EnBW)
    "eu-central" → "10YFR-RTE------C"  (France)
    "eu-north"   → "10YNO-1--------2"  (Norway NO1)

Override with bidding_zone_map= constructor argument.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
    normalize_price_df,
)

logger = logging.getLogger(__name__)

_DEFAULT_ZONE_MAP = {
    "eu-west":    "10YDE-ENBW-----N",
    "eu-central": "10YFR-RTE------C",
    "eu-north":   "10YNO-1--------2",
}

_ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


class ENTSOEPriceProvider(PriceProvider):
    """Fetch day-ahead electricity prices from ENTSO-E Transparency Platform.

    Args:
        api_key: ENTSO-E security token.
        bidding_zone_map: Mapping of Aurelius region → EIC bidding zone code.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        bidding_zone_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ENTSOE_API_KEY", "")
        self._zone_map = bidding_zone_map or _DEFAULT_ZONE_MAP

    @property
    def source_name(self) -> str:
        return "entsoe_dah"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly day-ahead prices.

        Raises:
            ProviderConfigError: If ENTSOE_API_KEY is not set.
        """
        if not self._api_key:
            raise ProviderConfigError(
                "ENTSOE_API_KEY is not set. "
                "Export it as an environment variable or pass api_key= to ENTSOEPriceProvider."
            )

        zone = self._zone_map.get(region)
        if zone is None:
            logger.warning(f"Region '{region}' not in ENTSO-E zone map; available: {list(self._zone_map)}")
            return empty_price_df()

        try:
            import requests
            from xml.etree import ElementTree as ET
        except ImportError:
            raise ProviderConfigError("'requests' package is required: pip install requests")

        # ENTSO-E expects UTC times in YYYYMMDDHHmm format
        start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
        period_start = start_utc.strftime("%Y%m%d%H%M")
        period_end = end_utc.strftime("%Y%m%d%H%M")

        params = {
            "securityToken": self._api_key,
            "documentType": "A44",   # Day-ahead prices
            "in_Domain": zone,
            "out_Domain": zone,
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(_ENTSOE_BASE, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"ENTSO-E rate limit; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if resp.status_code == 401:
                    raise ProviderConfigError("ENTSO-E API key rejected (401). Check ENTSOE_API_KEY.")
                resp.raise_for_status()
                xml_text = resp.text
                break
            except ProviderConfigError:
                raise
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"ENTSO-E request failed after {_MAX_RETRIES} attempts: {exc}")
                    return empty_price_df()
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        else:
            return empty_price_df()

        rows = self._parse_xml(xml_text, region)
        if not rows:
            logger.warning(f"ENTSO-E returned no parseable data for {region} {period_start}..{period_end}")
            return empty_price_df()

        df = pd.DataFrame(rows)
        return normalize_price_df(df, source=self.source_name, currency="EUR", granularity="hourly")

    @staticmethod
    def _parse_xml(xml_text: str, region: str) -> list[dict]:
        """Parse ENTSO-E GL_MarketDocument XML into price rows."""
        from xml.etree import ElementTree as ET

        NS = {
            "gl": "urn:iec62325.351:tc57wg16:451-6:generateddocument:2:0",
        }
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error(f"ENTSO-E XML parse error: {exc}")
            return []

        rows = []
        for ts_elem in root.findall(".//gl:TimeSeries", NS):
            try:
                period = ts_elem.find("gl:Period", NS)
                if period is None:
                    continue
                time_interval = period.find("gl:timeInterval", NS)
                if time_interval is None:
                    continue
                start_str = time_interval.find("gl:start", NS).text.strip()
                resolution = period.find("gl:resolution", NS).text.strip()  # e.g. PT60M
                if resolution != "PT60M":
                    continue  # Only hourly for now

                period_start_dt = pd.Timestamp(start_str, tz="UTC")
                for idx, point in enumerate(period.findall("gl:Point", NS)):
                    position = int(point.find("gl:position", NS).text)
                    price_elem = point.find("gl:price.amount", NS)
                    if price_elem is None:
                        continue
                    price = float(price_elem.text)
                    ts = period_start_dt + pd.Timedelta(hours=position - 1)
                    rows.append({"timestamp": ts, "region": region, "price_per_mwh": price})
            except Exception as exc:
                logger.debug(f"Skipping ENTSO-E TimeSeries element: {exc}")
                continue

        return rows
