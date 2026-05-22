"""ERCOT Public API settlement-point-price adapters (day-ahead and real-time).

ERCOT publishes Settlement Point Prices (SPP) rather than LMPs; the SPP is the
ERCOT equivalent of a nodal/hub price in USD/MWh and is what loads settle
against. We use the Houston trading hub (HB_HOUSTON) as the us-south reference.

Authentication (OAuth2 ROPC + APIM subscription key):
    The ERCOT Public API requires BOTH an Azure-B2C id_token (obtained from a
    username/password "resource owner password credential" flow) AND an API
    Management subscription key. Register at https://apiexplorer.ercot.com,
    subscribe to "Public API", then set:
        ERCOT_API_KEY   – primary subscription key (Ocp-Apim-Subscription-Key)
        ERCOT_USERNAME  – portal sign-up email (ROPC username)
        ERCOT_PASSWORD  – portal password (ROPC password)
    Optionally set ERCOT_ID_TOKEN to supply a pre-fetched id_token directly and
    skip the ROPC call (id_tokens last ~1 hour and cannot be refreshed).

    Token endpoint (POST, application/x-www-form-urlencoded):
        https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/
            B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token
    Body: grant_type=password, username, password, response_type=id_token,
          scope="openid <client_id> offline_access", client_id=<client_id>.
    The id_token field of the JSON response is the bearer credential.

Endpoints (base https://api.ercot.com/api/public-reports):
    ERCOTPriceProvider          – day-ahead hourly SPP
                                  /np4-190-cd/dam_stlmnt_pnt_prices
                                  (deliveryDate, hourEnding, settlementPoint,
                                   settlementPointPrice, DSTFlag)
    ERCOTRealtimePriceProvider  – real-time 15-minute SPP
                                  /np6-905-cd/spp_node_zone_hub
                                  (deliveryDate, deliveryHour, deliveryInterval,
                                   settlementPointName, settlementPointPrice,
                                   DSTFlag)

Query / response:
    Date filter params: deliveryDateFrom / deliveryDateTo (YYYY-MM-DD, ERCOT
    local = US/Central calendar dates). Settlement point filtered server-side via
    settlementPoint=HB_HOUSTON. Paginated via page (1-based) + size, with
    _meta.totalPages. Response body is {"fields":[{"name":...}], "data":[[...]],
    "_meta":{...}} — column order comes from fields[].name and each data row is a
    positional array.

Timestamps:
    ERCOT delivery dates/hours/intervals are US/Central wall-clock. We build the
    naive local timestamp (hourEnding-1 for the hour beginning; deliveryHour-1
    plus (deliveryInterval-1)*15 minutes for RT), localize to US/Central
    (nonexistent="shift_forward" for spring-forward; ambiguous resolved from the
    DSTFlag repeated-hour marker), and convert to UTC. Output is always
    UTC-aware, matching the canonical price schema.

Price unit:
    USD/MWh (settlementPointPrice).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+ stdlib
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import numpy as np
import pandas as pd
import requests

from .base import (
    PriceProvider,
    ProviderConfigError,
    empty_price_df,
    normalize_price_df,
)

logger = logging.getLogger(__name__)

_PUBLIC_BASE = "https://api.ercot.com/api/public-reports"
_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
# Public client id of the ERCOT Public API B2C application (documented constant,
# identical for all users — not a secret).
_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
_SCOPE = f"openid {_CLIENT_ID} offline_access"

_DAM_SPP_ENDPOINT = "/np4-190-cd/dam_stlmnt_pnt_prices"
_RT_SPP_ENDPOINT = "/np6-905-cd/spp_node_zone_hub"

_MAX_RETRIES = 4
_RETRY_BACKOFF = 2.0
_PAGE_SIZE = 50000
_CHUNK_DAYS = 30
_TOKEN_TTL_SECONDS = 3300  # id_tokens last ~1h; refresh a little early
_CENTRAL = "America/Chicago"  # ERCOT operating timezone (US Central)

# HB_HOUSTON: Houston trading hub — the most liquid us-south reference point.
_DEFAULT_HUB_MAP: dict[str, str] = {
    "us-south": "HB_HOUSTON",
}

_MISSING_CREDS_MSG = (
    "ERCOT credentials are not set. Register at https://apiexplorer.ercot.com "
    "(subscribe to 'Public API'), then export ERCOT_API_KEY (subscription key) "
    "and either ERCOT_USERNAME + ERCOT_PASSWORD (for the OAuth ROPC flow) or a "
    "pre-fetched ERCOT_ID_TOKEN."
)

# Module-level token cache keyed by username so repeated provider calls within an
# hour reuse one id_token. Value: (id_token, expiry_epoch_seconds).
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


class ERCOTAuthError(ProviderConfigError):
    """Raised when ERCOT authentication (ROPC token or subscription key) fails."""


def _get_id_token(username: str, password: str, *, force_refresh: bool = False) -> str:
    """Return a valid ERCOT id_token, using the ROPC flow (cached for ~1h).

    Raises ERCOTAuthError if the token endpoint rejects the credentials.
    """
    explicit = os.environ.get("ERCOT_ID_TOKEN", "")
    if explicit and not force_refresh:
        return explicit

    now = time.time()
    if not force_refresh:
        cached = _TOKEN_CACHE.get(username)
        if cached and cached[1] > now:
            return cached[0]

    body = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "response_type": "id_token",
        "scope": _SCOPE,
        "client_id": _CLIENT_ID,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(_TOKEN_URL, data=body, headers=headers, timeout=60)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise ERCOTAuthError(f"ERCOT token request failed: {exc}") from exc
            time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            continue

        if resp.status_code == 200:
            payload = resp.json()
            token = payload.get("id_token") or payload.get("access_token")
            if not token:
                raise ERCOTAuthError(
                    "ERCOT token response contained no id_token/access_token."
                )
            _TOKEN_CACHE[username] = (token, now + _TOKEN_TTL_SECONDS)
            return token
        if resp.status_code in (400, 401, 403):
            raise ERCOTAuthError(
                f"ERCOT rejected credentials ({resp.status_code}). Check "
                f"ERCOT_USERNAME / ERCOT_PASSWORD. Response: {resp.text[:300]}"
            )
        if attempt == _MAX_RETRIES - 1:
            raise ERCOTAuthError(
                f"ERCOT token endpoint error {resp.status_code}: {resp.text[:300]}"
            )
        time.sleep(_RETRY_BACKOFF * (2 ** attempt))

    raise ERCOTAuthError("ERCOT token request exhausted retries.")


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_field_indices(fields: list[dict]) -> dict[str, int]:
    """Map lower-cased field names → positional index in each data row."""
    return {str(f.get("name", "")).strip().lower(): i for i, f in enumerate(fields)}


def _parse_hour_ending(value) -> int:
    """Parse an ERCOT hourEnding value to an int hour (1..24).

    Accepts "1".."24", "01:00".."24:00", "0100", etc.
    """
    s = str(value).strip()
    if ":" in s:
        s = s.split(":")[0]
    s = s.lstrip("0") or "0"
    return int(float(s))


def _localize_central_to_utc(
    naive: list[pd.Timestamp],
    dst_flags: list[str],
) -> pd.DatetimeIndex:
    """Localize naive US/Central wall-clock timestamps to UTC.

    DSTFlag marks the repeated fall-back hour: "Y" → the second (standard-time)
    occurrence, otherwise the first (DST) occurrence. nonexistent="shift_forward"
    pushes the skipped spring-forward hour to the next valid instant.
    """
    idx = pd.DatetimeIndex(naive)
    # ERCOT marks the repeated fall-back hour with DSTFlag truthy (JSON boolean
    # True, or "Y"/"true"); that occurrence is standard time (the second reading),
    # so ambiguous=False. All other rows take the DST/first reading (ambiguous=True).
    repeated = {"Y", "TRUE", "1"}
    ambiguous = np.array(
        [str(f).strip().upper() not in repeated for f in dst_flags], dtype=bool
    )
    localized = idx.tz_localize(
        _CENTRAL, ambiguous=ambiguous, nonexistent="shift_forward"
    )
    return localized.tz_convert("UTC")


def _fetch_ercot_spp(
    *,
    api_key: str,
    username: str,
    password: str,
    region: str,
    settlement_point: str,
    start: datetime,
    end: datetime,
    endpoint: str,
    is_realtime: bool,
    source_name: str,
    granularity: str,
    floor_to: Optional[str],
) -> pd.DataFrame:
    """Core ERCOT SPP fetch: auth, date-chunked pagination, parse, normalize."""
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)

    # Fail fast on bad credentials before entering the pagination loop.
    _get_id_token(username, password)

    all_rows: list[dict] = []
    current = start_utc
    while current < end_utc:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS), end_utc)

        # ERCOT filters by US/Central calendar date. Widen by a day on each side
        # so boundary intervals aren't dropped; we filter precisely by UTC below.
        central = ZoneInfo(_CENTRAL)
        from_date = (current.astimezone(central) - timedelta(days=1)).strftime("%Y-%m-%d")
        to_date = (chunk_end.astimezone(central) + timedelta(days=1)).strftime("%Y-%m-%d")

        page = 1
        total_pages = 1
        while page <= total_pages:
            params = {
                "deliveryDateFrom": from_date,
                "deliveryDateTo": to_date,
                "settlementPoint": settlement_point,
                "page": page,
                "size": _PAGE_SIZE,
            }
            payload = _request_page(endpoint, params, api_key, username, password)
            if payload is None:
                break

            fields = payload.get("fields", []) or []
            data = payload.get("data", []) or []
            meta = payload.get("_meta", {}) or {}
            total_pages = int(meta.get("totalPages", 1) or 1)

            if fields and data:
                all_rows.extend(
                    _rows_from_page(fields, data, region, settlement_point, is_realtime)
                )
            if not data:
                break
            page += 1

        current = chunk_end

    if not all_rows:
        logger.warning(
            "ERCOT returned no data (endpoint=%s, point=%s, %s..%s)",
            endpoint, settlement_point, start_utc, end_utc,
        )
        return empty_price_df()

    df = pd.DataFrame(all_rows)
    # Localize the naive Central timestamps in one vectorized pass, then floor.
    utc_index = _localize_central_to_utc(
        list(df["_naive"]), list(df["_dst"])
    )
    df["timestamp"] = utc_index
    if floor_to:
        df["timestamp"] = df["timestamp"].dt.floor(floor_to)
    # Keep only the requested window (we widened the query by a day each side).
    mask = (df["timestamp"] >= pd.Timestamp(start_utc)) & (df["timestamp"] < pd.Timestamp(end_utc))
    df = df.loc[mask, ["timestamp", "region", "price_per_mwh"]]
    # Adjacent date-chunks overlap by the ±1-day widening, so the same interval
    # can be returned twice; drop duplicate (timestamp, region) pairs.
    df = df.drop_duplicates(subset=["timestamp", "region"], keep="last")
    if df.empty:
        return empty_price_df()
    return normalize_price_df(df, source=source_name, currency="USD", granularity=granularity)


def _request_page(
    endpoint: str,
    params: dict,
    api_key: str,
    username: str,
    password: str,
) -> Optional[dict]:
    """GET one page with retries; refresh token once on 401/403."""
    url = f"{_PUBLIC_BASE}{endpoint}"
    token = _get_id_token(username, password)
    refreshed = False

    for attempt in range(_MAX_RETRIES):
        headers = {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": api_key,
            "Accept": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=120)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.error("ERCOT request failed (%s): %s", endpoint, exc)
                return None
            time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.error("ERCOT returned non-JSON (%s): %s", endpoint, resp.text[:300])
                return None
        if resp.status_code in (401, 403) and not refreshed:
            # Expired/invalid token — force a refresh once, then retry.
            token = _get_id_token(username, password, force_refresh=True)
            refreshed = True
            continue
        if resp.status_code in (401, 403):
            raise ERCOTAuthError(
                f"ERCOT API rejected request ({resp.status_code}). Check "
                f"ERCOT_API_KEY subscription and token. Body: {resp.text[:300]}"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = _RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "ERCOT throttled/error %s on %s; retry %d/%d in %.0fs",
                resp.status_code, endpoint, attempt + 1, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue
        logger.error(
            "ERCOT request rejected (HTTP %s) on %s. Params=%s Body=%s",
            resp.status_code, endpoint, params, resp.text[:500],
        )
        return None

    return None


def _rows_from_page(
    fields: list[dict],
    data: list[list],
    region: str,
    settlement_point: str,
    is_realtime: bool,
) -> list[dict]:
    """Convert one page's positional rows into intermediate price-row dicts.

    Output dicts carry a naive Central timestamp ('_naive') and DST flag ('_dst')
    that the caller localizes to UTC in a single vectorized pass.
    """
    col = _resolve_field_indices(fields)

    date_i = col.get("deliverydate")
    price_i = col.get("settlementpointprice")
    dst_i = col.get("dstflag")
    point_i = col.get("settlementpoint", col.get("settlementpointname"))

    if date_i is None or price_i is None:
        logger.error(
            "ERCOT response missing deliveryDate/settlementPointPrice. Fields: %s",
            [f.get("name") for f in fields],
        )
        return []

    if is_realtime:
        hour_i = col.get("deliveryhour")
        interval_i = col.get("deliveryinterval")
    else:
        he_i = col.get("hourending")

    target = settlement_point.strip().upper()
    rows: list[dict] = []
    for r in data:
        try:
            # Defensive server-side filter (in case the API ignores the param).
            if point_i is not None and r[point_i] is not None:
                if str(r[point_i]).strip().upper() != target:
                    continue
            day = pd.Timestamp(r[date_i]).normalize()
            if is_realtime:
                hour = int(r[hour_i])          # 1..24
                interval = int(r[interval_i])  # 1..4
                naive = day + pd.Timedelta(hours=hour - 1, minutes=(interval - 1) * 15)
            else:
                hour = _parse_hour_ending(r[he_i])  # 1..24
                naive = day + pd.Timedelta(hours=hour - 1)
            price = float(r[price_i])
            dst = r[dst_i] if dst_i is not None else "N"
            rows.append({
                "_naive": naive,
                "_dst": dst,
                "region": region,
                "price_per_mwh": price,
            })
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return rows


class _ERCOTBase(PriceProvider):
    """Shared credential resolution + region→hub mapping for ERCOT providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        hub_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ERCOT_API_KEY", "")
        self._username = username or os.environ.get("ERCOT_USERNAME", "")
        self._password = password or os.environ.get("ERCOT_PASSWORD", "")
        self._hub_map = hub_map or _DEFAULT_HUB_MAP

    def _check_creds(self) -> None:
        has_token = bool(os.environ.get("ERCOT_ID_TOKEN"))
        if not self._api_key or not (has_token or (self._username and self._password)):
            raise ProviderConfigError(_MISSING_CREDS_MSG)

    def _resolve_point(self, region: str) -> Optional[str]:
        point = self._hub_map.get(region)
        if point is None:
            logger.warning(
                "Region '%s' not in ERCOT hub map; available: %s",
                region, list(self._hub_map),
            )
        return point


class ERCOTPriceProvider(_ERCOTBase):
    """Fetch hourly day-ahead settlement-point prices from the ERCOT Public API.

    Queries /np4-190-cd/dam_stlmnt_pnt_prices for HB_HOUSTON (us-south).
    """

    @property
    def source_name(self) -> str:
        return "ercot_dam_spp"

    def fetch_prices(self, region: str, start: datetime, end: datetime) -> pd.DataFrame:
        self._check_creds()
        point = self._resolve_point(region)
        if point is None:
            return empty_price_df()
        return _fetch_ercot_spp(
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            region=region,
            settlement_point=point,
            start=start,
            end=end,
            endpoint=_DAM_SPP_ENDPOINT,
            is_realtime=False,
            source_name=self.source_name,
            granularity="hourly",
            floor_to="h",
        )


class ERCOTRealtimePriceProvider(_ERCOTBase):
    """Fetch 15-minute real-time settlement-point prices from the ERCOT Public API.

    Queries /np6-905-cd/spp_node_zone_hub for HB_HOUSTON (us-south).
    source_granularity == "15min" (resample to hourly mean for hourly settlement).
    """

    @property
    def source_name(self) -> str:
        return "ercot_rt_spp"

    def fetch_prices(self, region: str, start: datetime, end: datetime) -> pd.DataFrame:
        self._check_creds()
        point = self._resolve_point(region)
        if point is None:
            return empty_price_df()
        return _fetch_ercot_spp(
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            region=region,
            settlement_point=point,
            start=start,
            end=end,
            endpoint=_RT_SPP_ENDPOINT,
            is_realtime=True,
            source_name=self.source_name,
            granularity="15min",
            floor_to=None,  # preserve 15-minute interval precision
        )
