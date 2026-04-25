"""CAISO OASIS LMP price adapters (day-ahead and real-time).

Both providers use the public CAISO OASIS SingleZip endpoint — no auth required.

Endpoint:
    https://oasis.caiso.com/oasisapi/SingleZip

Day-ahead (PRC_LMP / DAM):
    URL template:
        https://oasis.caiso.com/oasisapi/SingleZip
            ?queryname=PRC_LMP
            &market_run_id=DAM
            &startdatetime=<YYYYMMDDThhmm-0000>
            &enddatetime=<YYYYMMDDThhmm-0000>
            &node=TH_NP15_GEN-APND
            &resultformat=6
            &version=1

Real-time 5-min intervals (PRC_INTVL_LMP / RTM):
    URL template:
        https://oasis.caiso.com/oasisapi/SingleZip
            ?queryname=PRC_INTVL_LMP
            &market_run_id=RTM
            &startdatetime=<YYYYMMDDThhmm-0000>
            &enddatetime=<YYYYMMDDThhmm-0000>
            &node=TH_NP15_GEN-APND
            &resultformat=6
            &version=1

Reference node:
    TH_NP15_GEN-APND — NP15 trading-hub aggregate price node (Northern California).
    This is the standard liquid reference hub used for California LMP backtesting.

resultformat=6:
    Requests ZIP/CSV output explicitly. Without it CAISO may return XML.

CSV columns (both PRC_LMP and PRC_INTVL_LMP):
    INTERVALSTARTTIME_GMT  — ISO 8601 UTC timestamp (field is already UTC/GMT)
    LMP_TYPE               — LMP | MCE | MCC | MLC  (we keep LMP = total price only)
    MW                     — price in USD/MWh (column name is "MW" by CAISO convention
                             but the value IS a price for PRC_LMP / PRC_INTVL_LMP queries)

Timestamp handling:
    CAISO OASIS timestamps in INTERVALSTARTTIME_GMT are UTC.
    All output timestamps are UTC-aware pd.Timestamps.
    No naive timestamps are ever produced.

Price field:
    LMP_TYPE == "LMP" row → MW column → price_per_mwh (USD/MWh).
    MCE / MCC / MLC component rows are filtered out.

Demand/load guard:
    The MW column from PRC_LMP / PRC_INTVL_LMP queries holds prices (USD/MWh).
    It is never mapped to a demand or load value.
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

# TH_NP15_GEN-APND: NP15 trading-hub aggregate price node (Northern California).
# Standard reference hub for CAISO LMP backtesting and live monitoring.
_DEFAULT_HUB_MAP: dict[str, str] = {
    "us-west": "TH_NP15_GEN-APND",
}


class CAISOPriceProvider(PriceProvider):
    """Fetch hourly day-ahead LMP from CAISO OASIS (no auth required).

    Queries PRC_LMP with market_run_id=DAM. Returns hourly prices for the
    NP15 trading hub (TH_NP15_GEN-APND) — the standard California reference hub.

    Args:
        hub_map: Override mapping of Aurelius region → CAISO pricing node ID.
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
            DataFrame with canonical PRICE_COLUMNS; empty on failure.
        """
        node = self._hub_map.get(region)
        if node is None:
            logger.warning(
                f"Region '{region}' not in CAISO hub map; "
                f"available: {list(self._hub_map)}"
            )
            return empty_price_df()

        return _fetch_lmp(
            node=node,
            region=region,
            start=start,
            end=end,
            queryname="PRC_LMP",
            market_run_id="DAM",
            source_name=self.source_name,
            granularity="hourly",
            floor_to="h",
        )


class CAISORealtimePriceProvider(PriceProvider):
    """Fetch 5-minute real-time interval LMP from CAISO OASIS (no auth required).

    Queries PRC_INTVL_LMP with market_run_id=RTM. Returns 5-minute interval
    prices for TH_NP15_GEN-APND — used for live/shadow monitoring of California LMP.

    Args:
        hub_map: Override mapping of Aurelius region → CAISO pricing node ID.
    """

    def __init__(
        self,
        hub_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._hub_map = hub_map or _DEFAULT_HUB_MAP

    @property
    def source_name(self) -> str:
        return "caiso_oasis_rtm"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch 5-minute real-time LMP for *region* in [start, end).

        Returns:
            DataFrame with canonical PRICE_COLUMNS; empty on failure.
            source_granularity == "5min".
        """
        node = self._hub_map.get(region)
        if node is None:
            logger.warning(
                f"Region '{region}' not in CAISO hub map; "
                f"available: {list(self._hub_map)}"
            )
            return empty_price_df()

        return _fetch_lmp(
            node=node,
            region=region,
            start=start,
            end=end,
            queryname="PRC_INTVL_LMP",
            market_run_id="RTM",
            source_name=self.source_name,
            granularity="5min",
            floor_to=None,  # preserve 5-minute interval precision
        )


def _fetch_lmp(
    node: str,
    region: str,
    start: datetime,
    end: datetime,
    queryname: str,
    market_run_id: str,
    source_name: str,
    granularity: str,
    floor_to: Optional[str],
) -> pd.DataFrame:
    """Core CAISO OASIS fetch loop used by both day-ahead and real-time providers."""

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
            "queryname": queryname,
            "market_run_id": market_run_id,
            "startdatetime": start_str,
            "enddatetime": end_str,
            "version": "1",
            "node": node,
            "resultformat": "6",  # ZIP/CSV output (explicit; avoids XML default)
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

                rows = _parse_zip_response(resp.content, region, node, floor_to)
                all_rows.extend(rows)
                break

            except ProviderConfigError:
                raise
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(
                        f"CAISO OASIS request failed "
                        f"(queryname={queryname}, node={node}): {exc}"
                    )
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))

        current = chunk_end
        time.sleep(_PAGE_SLEEP)

    if not all_rows:
        logger.warning(
            f"CAISO OASIS returned no data "
            f"(queryname={queryname}, node={node}, {start_utc}..{end_utc})"
        )
        return empty_price_df()

    df = pd.DataFrame(all_rows)
    return normalize_price_df(df, source=source_name, currency="USD", granularity=granularity)


def _parse_zip_response(
    content: bytes,
    region: str,
    node: str,
    floor_to: Optional[str] = "h",
) -> list[dict]:
    """Extract price rows from a CAISO OASIS ZIP/CSV response.

    CAISO OASIS returns a ZIP file containing one or more CSV files.
    Each CSV has INTERVALSTARTTIME_GMT (UTC), LMP_TYPE, and MW (price USD/MWh).

    Args:
        content:  Raw bytes of the HTTP response (a ZIP file).
        region:   Aurelius region label for the output DataFrame.
        node:     CAISO pricing node (for logging).
        floor_to: pd.Timestamp floor unit for timestamps ("h" for hourly,
                  None to preserve raw interval precision).

    Returns:
        List of dicts with keys: timestamp, region, price_per_mwh.
    """
    rows: list[dict] = []

    try:
        zip_buf = io.BytesIO(content)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
                if xml_names:
                    with zf.open(xml_names[0]) as f:
                        err_text = f.read(2000).decode("utf-8", errors="replace")
                    logger.error(
                        f"CAISO OASIS returned XML error for node={node}: {err_text[:500]}"
                    )
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

                    rows.extend(_extract_lmp_rows(df, region, csv_name, floor_to))

    except zipfile.BadZipFile:
        snippet = content[:500].decode("utf-8", errors="replace")
        logger.error(f"CAISO OASIS response is not a ZIP (node={node}): {snippet}")

    return rows


def _extract_lmp_rows(
    df: pd.DataFrame,
    region: str,
    source_name: str,
    floor_to: Optional[str] = "h",
) -> list[dict]:
    """Parse price rows from a CAISO OASIS CSV DataFrame.

    CAISO OASIS PRC_LMP and PRC_INTVL_LMP CSVs share the same key columns:
        INTERVALSTARTTIME_GMT  – ISO 8601 UTC timestamp (already UTC)
        LMP_TYPE               – LMP | MCE | MCC | MLC
        MW                     – Price in USD/MWh (column name is "MW" by CAISO
                                  convention but holds a $/MWh price for PRC_LMP
                                  and PRC_INTVL_LMP queries — NOT energy in MWh)

    We keep only LMP_TYPE == "LMP" rows (total LMP; filters out congestion/loss).

    Args:
        floor_to: Timestamp floor unit ("h" for day-ahead hourly data, None for
                  5-min real-time interval data).
    """
    rows: list[dict] = []

    df.columns = [c.strip().upper() for c in df.columns]

    ts_col = next((c for c in df.columns if "INTERVALSTARTTIME" in c and "GMT" in c), None)
    type_col = "LMP_TYPE" if "LMP_TYPE" in df.columns else None
    # MW is the price column in CAISO PRC_LMP / PRC_INTVL_LMP responses
    val_col = "MW" if "MW" in df.columns else None

    if ts_col is None or val_col is None:
        logger.error(
            f"CAISO CSV ({source_name}) missing expected columns. "
            f"Found: {list(df.columns)}"
        )
        return rows

    # Filter to total LMP only — exclude congestion (MCC), loss (MLC), energy (MCE) components
    if type_col:
        df = df[df[type_col].astype(str).str.upper() == "LMP"]

    for _, row in df.iterrows():
        try:
            ts = pd.Timestamp(row[ts_col]).tz_convert("UTC")
            if floor_to:
                ts = ts.floor(floor_to)
            price = float(row[val_col])
            rows.append({"timestamp": ts, "region": region, "price_per_mwh": price})
        except (ValueError, TypeError, KeyError):
            continue

    return rows
