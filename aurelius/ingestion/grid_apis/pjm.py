"""PJM Data Miner API 2 LMP price adapters (day-ahead and real-time).

Fetches locational marginal prices from PJM Interconnection via the PJM
Data Miner 2 REST API.

Environment variable required:
    PJM_API_KEY  –  register free at https://developer.pjm.com/

Supported regions (Aurelius → PJM pricing node):
    "us-east" → pnode_id=1 (PJM-RTO system-wide aggregate)

Override with node_map= constructor argument.

Providers:
    PJMPriceProvider          – hourly day-ahead LMP   (/da_hrl_lmps, total_lmp_da)
    PJMRealtimePriceProvider  – real-time LMP          (/rt_fivemin_hrl_lmps 5-min
                                                         total_lmp_rt; or /rt_hrl_lmps
                                                         hourly when hourly=True)

Price unit:
    USD/MWh (Total LMP = system energy price + congestion + loss)

Datetime filtering:
    PJM Data Miner 2's datetime_beginning_ept filter requires timestamps in
    Eastern Prevailing Time (EPT, DST-aware), format "MM/DD/YYYY HH:MM", and a
    range separator " to " (with spaces). ISO/bracket notation returns HTTP 400.
    Response timestamps are read from datetime_beginning_utc (UTC) — that field
    is a bare ISO string with no "UTC" suffix, so it is parsed as UTC directly.

API reference:
    https://api.pjm.com/api/v1/da_hrl_lmps
    https://api.pjm.com/api/v1/rt_fivemin_hrl_lmps
    https://api.pjm.com/api/v1/rt_hrl_lmps
    Authentication header: Ocp-Apim-Subscription-Key

Rate limits:
    Subscription key tier determines rate limits. Free tier: 30 requests/min.
    Retry with exponential back-off on 429 responses.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+ stdlib
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import pandas as pd
import requests

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
    normalize_price_df,
)

logger = logging.getLogger(__name__)

_PJM_BASE = "https://api.pjm.com/api/v1"
_DA_LMP_ENDPOINT = "/da_hrl_lmps"
_RT_5MIN_ENDPOINT = "/rt_fivemin_hrl_lmps"
_RT_HRL_ENDPOINT = "/rt_hrl_lmps"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_MAX_ROWS_PER_PAGE = 5000
_EASTERN = ZoneInfo("America/New_York")  # PJM datetime_beginning_ept is EPT, DST-aware

_MISSING_KEY_MSG = (
    "PJM_API_KEY is not set. "
    "Register free at https://developer.pjm.com/ and export PJM_API_KEY."
)

# pnode_id=1 is the PJM-RTO system-wide aggregate price node — the most liquid
# RTO-level reference. The same node is used for day-ahead and real-time so the
# DA/RT spread is measured on a consistent location.
_DEFAULT_NODE_MAP: dict[str, dict] = {
    "us-east": {
        "pnode_id": "1",
        "pnode_name": "PJM-RTO",
    },
}


def _parse_pjm_timestamp(item: dict) -> Optional[pd.Timestamp]:
    """Return a UTC-aware timestamp from a PJM row, or None if unparseable.

    Prefers datetime_beginning_utc (a bare ISO string that IS UTC despite having
    no 'Z'/'UTC' suffix). Falls back to datetime_beginning_ept, localized as
    US/Eastern with DST-safe handling for the spring-forward / fall-back hours.
    """
    utc_str = item.get("datetime_beginning_utc")
    if utc_str:
        ts = pd.Timestamp(utc_str)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    ept_str = item.get("datetime_beginning_ept")
    if ept_str:
        ts = pd.Timestamp(ept_str)
        if ts.tzinfo is None:
            # nonexistent='shift_forward': 2:00 AM on spring-forward day becomes
            # 3:00 AM. ambiguous='infer': use PJM's ordering for fall-back hours.
            return ts.tz_localize(
                "US/Eastern", nonexistent="shift_forward", ambiguous="infer"
            ).tz_convert("UTC")
        return ts.tz_convert("UTC")

    return None


def _fetch_pjm_lmp(
    *,
    api_key: str,
    region: str,
    start: datetime,
    end: datetime,
    node_spec: dict,
    endpoint: str,
    price_field: str,
    source_name: str,
    granularity: str,
    floor_to: Optional[str],
) -> pd.DataFrame:
    """Core PJM Data Miner fetch + pagination loop shared by DA and RT providers.

    Args:
        price_field: Response field holding the $/MWh price
                     ("total_lmp_da" for day-ahead, "total_lmp_rt" for real-time).
        floor_to:    pd.Timestamp floor unit ("h" for hourly, None to preserve
                     5-minute interval precision).
    """

    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    start_ept = _to_utc(start).astimezone(_EASTERN)
    end_ept = _to_utc(end).astimezone(_EASTERN)
    start_str = start_ept.strftime("%m/%d/%Y %H:%M")
    end_str = end_ept.strftime("%m/%d/%Y %H:%M")

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }

    all_rows: list[dict] = []
    offset = 1  # PJM uses 1-based row numbering

    while True:
        params = {
            "startRow": offset,
            "rowCount": _MAX_ROWS_PER_PAGE,
            "datetime_beginning_ept": f"{start_str} to {end_str}",
            "pnode_id": node_spec["pnode_id"],
            "fields": f"datetime_beginning_utc,datetime_beginning_ept,pnode_name,{price_field}",
        }

        payload = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(
                    f"{_PJM_BASE}{endpoint}",
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"PJM rate limit; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if resp.status_code in (401, 403):
                    raise ProviderConfigError(
                        f"PJM API key rejected ({resp.status_code}). Check PJM_API_KEY."
                    )
                if 400 <= resp.status_code < 500:
                    # Bad-request family: don't retry. Parse PJM's structured
                    # errors[] so the most relevant message surfaces.
                    body = resp.text[:2000]
                    try:
                        errs = resp.json().get("errors", [])
                    except Exception:
                        errs = []
                    err_msgs = [
                        f"{e.get('field', '?')}: {e.get('message', '?')} "
                        f"(detail={e.get('detail')})"
                        for e in errs
                    ]
                    is_archive_error = any(
                        "archived data" in (e.get("message", "") or "").lower()
                        for e in errs
                    )
                    if is_archive_error:
                        logger.error(
                            "PJM rejected query as 'archived data' (HTTP %d). "
                            "Older data is moved to PJM's archive feed which does "
                            "not accept pnode_id/fields filters. Shift --start/--end "
                            "into the recent window or query the archive feed "
                            "separately. PJM errors: %s",
                            resp.status_code, err_msgs,
                        )
                    else:
                        logger.error(
                            "PJM rejected request (HTTP %d). Errors: %s\n"
                            "Params: %s\n--- PJM response (first 2KB) ---\n%s\n--- END ---",
                            resp.status_code, err_msgs, params, body,
                        )
                    return empty_price_df()
                resp.raise_for_status()
                payload = resp.json()
                break
            except ProviderConfigError:
                raise
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"PJM request failed ({endpoint}): {exc}")
                    return empty_price_df()
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        else:
            return empty_price_df()

        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            try:
                ts = _parse_pjm_timestamp(item)
                if ts is None:
                    continue
                if floor_to:
                    ts = ts.floor(floor_to)
                # total_lmp_* is the sum of energy + congestion + loss in $/MWh
                price = float(item[price_field])
                all_rows.append({
                    "timestamp": ts,
                    "region": region,
                    "price_per_mwh": price,
                })
            except (KeyError, ValueError, TypeError):
                continue

        total_rows = payload.get("totalRows", 0) or 0
        offset += len(items)
        if len(items) < _MAX_ROWS_PER_PAGE or (total_rows and offset > total_rows):
            break

    if not all_rows:
        logger.warning(
            f"PJM returned no data ({endpoint}, pnode={node_spec['pnode_id']}, "
            f"{start_str}..{end_str} EPT)"
        )
        return empty_price_df()

    df = pd.DataFrame(all_rows)
    return normalize_price_df(df, source=source_name, currency="USD", granularity=granularity)


class PJMPriceProvider(PriceProvider):
    """Fetch hourly day-ahead LMP from PJM Data Miner API 2.

    Args:
        api_key:  PJM subscription key. Reads PJM_API_KEY env var if not supplied.
        node_map: Mapping of Aurelius region → PJM node spec dict.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        node_map: Optional[dict[str, dict]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("PJM_API_KEY", "")
        self._node_map = node_map or _DEFAULT_NODE_MAP

    @property
    def source_name(self) -> str:
        return "pjm_da_lmp"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly day-ahead LMP for *region* in [start, end).

        Raises:
            ProviderConfigError: If PJM_API_KEY is not set.
        """
        if not self._api_key:
            raise ProviderConfigError(_MISSING_KEY_MSG)

        node_spec = self._node_map.get(region)
        if node_spec is None:
            logger.warning(
                f"Region '{region}' not in PJM node map; available: {list(self._node_map)}"
            )
            return empty_price_df()

        return _fetch_pjm_lmp(
            api_key=self._api_key,
            region=region,
            start=start,
            end=end,
            node_spec=node_spec,
            endpoint=_DA_LMP_ENDPOINT,
            price_field="total_lmp_da",
            source_name=self.source_name,
            granularity="hourly",
            floor_to="h",
        )


class PJMRealtimePriceProvider(PriceProvider):
    """Fetch real-time LMP from PJM Data Miner API 2.

    Defaults to 5-minute granularity via /rt_fivemin_hrl_lmps. Set ``hourly=True``
    to fetch the hourly-integrated real-time LMP via /rt_hrl_lmps (aligns with the
    hourly day-ahead series for DA/RT spread analysis). Maps total_lmp_rt →
    price_per_mwh.

    Args:
        api_key:  PJM subscription key. Reads PJM_API_KEY env var if not supplied.
        node_map: Mapping of Aurelius region → PJM node spec dict.
        hourly:   When True, use the hourly RT feed instead of the 5-minute feed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        node_map: Optional[dict[str, dict]] = None,
        hourly: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("PJM_API_KEY", "")
        self._node_map = node_map or _DEFAULT_NODE_MAP
        self._hourly = hourly

    @property
    def source_name(self) -> str:
        return "pjm_rt_lmp"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch real-time LMP for *region* in [start, end).

        Returns 5-minute interval prices by default (source_granularity="5min"),
        or hourly prices when constructed with hourly=True.

        Raises:
            ProviderConfigError: If PJM_API_KEY is not set.
        """
        if not self._api_key:
            raise ProviderConfigError(_MISSING_KEY_MSG)

        node_spec = self._node_map.get(region)
        if node_spec is None:
            logger.warning(
                f"Region '{region}' not in PJM node map; available: {list(self._node_map)}"
            )
            return empty_price_df()

        endpoint = _RT_HRL_ENDPOINT if self._hourly else _RT_5MIN_ENDPOINT
        granularity = "hourly" if self._hourly else "5min"
        floor_to = "h" if self._hourly else None

        return _fetch_pjm_lmp(
            api_key=self._api_key,
            region=region,
            start=start,
            end=end,
            node_spec=node_spec,
            endpoint=endpoint,
            price_field="total_lmp_rt",
            source_name=self.source_name,
            granularity=granularity,
            floor_to=floor_to,
        )
