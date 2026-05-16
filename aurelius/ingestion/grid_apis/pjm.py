"""PJM Data Miner API 2 day-ahead LMP price adapter.

Fetches hourly day-ahead locational marginal prices from PJM Interconnection
via the PJM Data Miner 2 REST API.

Environment variable required:
    PJM_API_KEY  –  register free at https://developer.pjm.com/

Supported regions (Aurelius → PJM pricing node):
    "us-east" → Western Hub (pnode_id=1)

Override with node_map= constructor argument.

Price unit:
    USD/MWh (Total LMP = system energy price + congestion + loss)

API reference:
    https://api.pjm.com/api/v1/da_hrl_lmps
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
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_MAX_ROWS_PER_PAGE = 5000
_EASTERN = ZoneInfo("America/New_York")  # PJM datetime_beginning_ept is EPT, DST-aware

# pnode_id=1 is PJM Western Hub, the most liquid reference price
_DEFAULT_NODE_MAP: dict[str, dict] = {
    "us-east": {
        "pnode_id": "1",
        "pnode_name": "WESTERN HUB",
    },
}


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
            raise ProviderConfigError(
                "PJM_API_KEY is not set. "
                "Register free at https://developer.pjm.com/ and export PJM_API_KEY."
            )

        node_spec = self._node_map.get(region)
        if node_spec is None:
            logger.warning(
                f"Region '{region}' not in PJM node map; "
                f"available: {list(self._node_map)}"
            )
            return empty_price_df()

        def _to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        # PJM Data Miner 2 datetime_beginning_ept filter requires:
        #   - timestamps in Eastern Prevailing Time (EPT, DST-aware), not UTC
        #   - format MM/DD/YYYY HH:MM
        #   - range separator " to " (with spaces)
        # ISO format with bracket notation returns 400 "DateTime filter
        # field value range is invalid".
        start_ept = start_utc.astimezone(_EASTERN)
        end_ept = end_utc.astimezone(_EASTERN)
        start_str = start_ept.strftime("%m/%d/%Y %H:%M")
        end_str = end_ept.strftime("%m/%d/%Y %H:%M")

        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
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
                "fields": "datetime_beginning_utc,pnode_name,total_lmp_da",
            }

            for attempt in range(_MAX_RETRIES):
                try:
                    resp = requests.get(
                        f"{_PJM_BASE}{_DA_LMP_ENDPOINT}",
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
                        # errors[] so the most relevant message surfaces (the feed
                        # metadata body also includes the word "archived" in its
                        # description, which would fool a substring match).
                        body = resp.text[:2000]
                        try:
                            errs = resp.json().get("errors", [])
                            err_msgs = [
                                f"{e.get('field','?')}: {e.get('message','?')} "
                                f"(detail={e.get('detail')})"
                                for e in errs
                            ]
                        except Exception:
                            err_msgs = []
                        is_archive_error = any(
                            "archived data" in (e.get("message", "") or "").lower()
                            for e in (errs if 'errs' in locals() else [])
                        )
                        if is_archive_error:
                            logger.error(
                                "PJM rejected query as 'archived data' (HTTP %d). "
                                "Data older than ~24 months is moved to PJM's archive "
                                "feed which does not accept pnode_id/fields filters. "
                                "Shift your --start/--end into the last ~24 months "
                                "or query the archive feed separately. "
                                "PJM errors: %s",
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
                        logger.error(f"PJM request failed: {exc}")
                        return empty_price_df()
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            else:
                return empty_price_df()

            items = payload.get("items", [])
            if not items:
                break

            for item in items:
                try:
                    # Use the FIELD NAME to determine the timezone, not the value's
                    # content. PJM returns datetime_beginning_utc as a bare ISO string
                    # like "2026-01-01T05:00:00" (no 'Z' / 'UTC' suffix), so naive
                    # string sniffing falls through to the wrong branch and tries to
                    # localize a UTC value as Eastern — which crashes on the DST
                    # spring-forward hour (e.g. 2026-03-08T02:00).
                    utc_str = item.get("datetime_beginning_utc")
                    ept_str = item.get("datetime_beginning_ept")
                    if utc_str:
                        ts = pd.Timestamp(utc_str)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        else:
                            ts = ts.tz_convert("UTC")
                    elif ept_str:
                        ts = pd.Timestamp(ept_str)
                        if ts.tzinfo is None:
                            # nonexistent='shift_forward': 2:00 AM on spring-forward
                            # day becomes 3:00 AM. ambiguous='infer': use PJM's
                            # ordering for fall-back duplicate hours.
                            ts = ts.tz_localize(
                                "US/Eastern",
                                nonexistent="shift_forward",
                                ambiguous="infer",
                            ).tz_convert("UTC")
                        else:
                            ts = ts.tz_convert("UTC")
                    else:
                        continue

                    # total_lmp_da is the sum of energy + congestion + loss in $/MWh
                    price = float(item["total_lmp_da"])
                    all_rows.append({
                        "timestamp": ts.floor("h"),
                        "region": region,
                        "price_per_mwh": price,
                    })
                except (KeyError, ValueError, TypeError):
                    continue

            total_rows = payload.get("totalRows", 0)
            offset += len(items)
            if offset > total_rows or len(items) < _MAX_ROWS_PER_PAGE:
                break

        if not all_rows:
            logger.warning(
                f"PJM returned no data for pnode={node_spec['pnode_id']} "
                f"{start_str}..{end_str}"
            )
            return empty_price_df()

        df = pd.DataFrame(all_rows)
        return normalize_price_df(df, source=self.source_name, currency="USD", granularity="hourly")
