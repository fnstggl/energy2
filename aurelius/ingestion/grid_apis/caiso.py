"""CAISO OASIS day-ahead LMP price adapter.

Fetches hourly day-ahead locational marginal prices (DA-LMP) from the
California ISO OASIS public API. No API key required.

CAISO OASIS API:
    https://oasis.caiso.com/oasisapi/SingleZip
    queryname=PRC_LMP, market_run_id=DAM

Supported regions (Aurelius → CAISO pricing node):
    "us-west" → NP15_7_N001  (Northern California trading hub)

Override with hub_map= constructor argument.

Price unit:
    USD/MWh (LMP = Locational Marginal Price)

    The "LMP" type in CAISO OASIS data is the total price including:
        - MCE (Market Clearing Energy component)
        - MCC (Congestion component)
        - MLC (Loss component)
    This adapter fetches LMP_TYPE=LMP (total) and maps it to price_per_mwh.

Notes:
    - CAISO OASIS returns a ZIP file containing a CSV; this is parsed in-memory.
    - Requests are chunked into ≤31-day windows.
    - Page sleep and retry logic guard against 503 responses.
    - CAISO OASIS can be slow; timeout is set to 120 seconds per request.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
    normalize_price_df,
)

logger = logging.getLogger(__name__)

_OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 3.0
_PAGE_SLEEP = 1.0
_CHUNK_DAYS = 31

_DEFAULT_HUB_MAP: dict[str, str] = {
    "us-west": "NP15_7_N001",
}


class CAISOPriceProvider(PriceProvider):
    """Fetch hourly day-ahead LMP from CAISO OASIS (no auth required).

    Args:
        hub_map: Mapping of Aurelius region → CAISO pricing node ID.
    """

    def __init__(
        self,
        hub_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._hub_map = hub_map or _DEFAULT_HUB_MAP

    @property
    def source_name(self) -> str:
        return "caiso_oasis_dam"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly day-ahead LMP for *region* in [start, end).

        Returns:
            DataFrame with canonical PRICE_COLUMNS.
            Returns empty_price_df() on failure.
        """
        node = self._hub_map.get(region)
        if node is None:
            logger.warning(
                f"Region '{region}' not in CAISO hub map; "
                f"available: {list(self._hub_map)}"
            )
            return empty_price_df()

        def _to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        all_rows: list[dict] = []

        current = start_utc
        while current < end_utc:
            chunk_end = min(current + timedelta(days=_CHUNK_DAYS), end_utc)

            # CAISO OASIS datetime format: YYYYMMDDThhmm-0000 (UTC offset)
            start_str = current.strftime("%Y%m%dT%H%M") + "-0000"
            end_str = chunk_end.strftime("%Y%m%dT%H%M") + "-0000"

            params = {
                "queryname": "PRC_LMP",
                "market_run_id": "DAM",
                "startdatetime": start_str,
                "enddatetime": end_str,
                "version": "1",
                "node": node,
            }

            for attempt in range(_MAX_RETRIES):
                try:
                    resp = requests.get(
                        _OASIS_URL,
                        params=params,
                        timeout=120,
                    )
                    if resp.status_code == 429 or resp.status_code == 503:
                        wait = _RETRY_BACKOFF * (2 ** attempt)
                        logger.warning(
                            f"CAISO OASIS throttled ({resp.status_code}); "
                            f"retrying in {wait:.0f}s"
                        )
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()

                    rows = self._parse_zip_response(resp.content, region, node)
                    all_rows.extend(rows)
                    break

                except ProviderConfigError:
                    raise
                except Exception as exc:
                    if attempt == _MAX_RETRIES - 1:
                        logger.error(f"CAISO OASIS request failed: {exc}")
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))

            current = chunk_end
            time.sleep(_PAGE_SLEEP)

        if not all_rows:
            logger.warning(
                f"CAISO OASIS returned no data for node={node} "
                f"{start_utc}..{end_utc}"
            )
            return empty_price_df()

        df = pd.DataFrame(all_rows)
        return normalize_price_df(df, source=self.source_name, currency="USD", granularity="hourly")

    @staticmethod
    def _parse_zip_response(
        content: bytes,
        region: str,
        node: str,
    ) -> list[dict]:
        """Extract price rows from a CAISO OASIS ZIP/CSV response.

        CAISO OASIS returns:
            - A ZIP file containing one or more CSV files.
            - Each CSV has columns including INTERVALSTARTTIME_GMT, LMP_TYPE, and
              a value column (named "MW" in some OASIS versions — despite the name
              this column holds USD/MWh price values for PRC_LMP queries).

        We filter rows where LMP_TYPE == "LMP" (total price, not congestion/loss).

        Args:
            content: Raw bytes of the HTTP response (a ZIP file).
            region:  Aurelius region label for the output DataFrame.
            node:    CAISO pricing node (for logging).

        Returns:
            List of dicts with keys: timestamp, region, price_per_mwh.
        """
        rows: list[dict] = []

        try:
            zip_buf = io.BytesIO(content)
            with zipfile.ZipFile(zip_buf, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    # CAISO sometimes returns an XML error inside the ZIP
                    xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
                    if xml_names:
                        with zf.open(xml_names[0]) as f:
                            err_text = f.read(2000).decode("utf-8", errors="replace")
                        logger.error(f"CAISO OASIS returned XML error for node={node}: {err_text[:500]}")
                    else:
                        logger.error(f"CAISO OASIS ZIP contains no CSV for node={node}")
                    return rows

                for csv_name in csv_names:
                    with zf.open(csv_name) as f:
                        try:
                            df = pd.read_csv(f)
                        except Exception as exc:
                            logger.error(f"CAISO CSV parse error ({csv_name}): {exc}")
                            continue

                        rows.extend(_extract_lmp_rows(df, region, csv_name))

        except zipfile.BadZipFile:
            # Response may be a plain XML/HTML error page
            snippet = content[:500].decode("utf-8", errors="replace")
            logger.error(f"CAISO OASIS response is not a ZIP (node={node}): {snippet}")

        return rows


def _extract_lmp_rows(df: pd.DataFrame, region: str, source_name: str) -> list[dict]:
    """Parse price rows from a CAISO OASIS CSV DataFrame.

    CAISO OASIS PRC_LMP CSVs have these key columns:
        INTERVALSTARTTIME_GMT  – ISO 8601 UTC timestamp
        LMP_TYPE               – LMP | MCE | MCC | MLC
        MW                     – Price in USD/MWh (column name is "MW" by CAISO convention
                                  but the value IS a price for PRC_LMP queries)

    We keep only LMP_TYPE == "LMP" rows (total price).
    """
    rows: list[dict] = []

    # Normalise column names (CAISO occasionally uses different casing)
    df.columns = [c.strip().upper() for c in df.columns]

    ts_col = next((c for c in df.columns if "INTERVALSTARTTIME" in c and "GMT" in c), None)
    type_col = "LMP_TYPE" if "LMP_TYPE" in df.columns else None
    # The price value column is called "MW" in CAISO OASIS for PRC_LMP
    val_col = "MW" if "MW" in df.columns else None

    if ts_col is None or val_col is None:
        logger.error(
            f"CAISO CSV ({source_name}) missing expected columns. "
            f"Found: {list(df.columns)}"
        )
        return rows

    # Filter to total LMP only (not congestion/loss components)
    if type_col:
        df = df[df[type_col].astype(str).str.upper() == "LMP"]

    for _, row in df.iterrows():
        try:
            ts = pd.Timestamp(row[ts_col]).tz_convert("UTC")
            # Round to hour (CAISO DAM is already hourly but be safe)
            ts = ts.floor("h")
            price = float(row[val_col])
            rows.append({"timestamp": ts, "region": region, "price_per_mwh": price})
        except (ValueError, TypeError, KeyError):
            continue

    return rows
