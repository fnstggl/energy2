"""ElectricityMaps carbon intensity adapter.

Fetches historical carbon intensity (gCO₂eq/kWh) from the ElectricityMaps API.

Electricity Maps is treated as an OPTIONAL aggregator/sandbox/fallback — never
as the source of truth for wholesale prices. Electricity Maps does not publish
US ISO wholesale prices, and its European prices come from the same ENTSO-E
Transparency Platform Aurelius already reads directly. See
docs/ELECTRICITYMAPS_CONTRIB_AUDIT.md.

Environment variables:
    ELECTRICITYMAPS_API_KEY  – register at https://api.electricitymap.org
    ELECTRICITYMAPS_SANDBOX  – if "true"/"1"/"yes", all returned data is flagged
                               is_sandbox=True and is INADMISSIBLE for benchmark
                               or savings claims.

Zone mapping is sourced from aurelius.ingestion.region_registry (single source
of truth). Override with the zone_map= constructor argument.

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
from ..market_data_provider import (
    CarbonPoint,
    MarketDataProvider,
    MarketPricePoint,
    MarketType,
    Provenance,
    ProviderCapability,
    Signal,
)
from ..region_registry import electricitymaps_zone_map

logger = logging.getLogger(__name__)


def _registry_zone_map() -> dict[str, str]:
    """Build the Aurelius-region → EM-zone map from the region registry."""
    try:
        return electricitymaps_zone_map()
    except Exception:  # pragma: no cover - registry import/use should not fail
        return dict(_FALLBACK_ZONE_MAP)


# Used only if the registry is unavailable for some reason.
_FALLBACK_ZONE_MAP = {
    "us-west":  "US-CAL-CISO",
    "us-east":  "US-MIDA-PJM",
    "us-south": "US-TEX-ERCO",
    "eu-west":  "DE",
    "eu-north": "NO-NO1",
}

_DEFAULT_ZONE_MAP = _registry_zone_map()

_EM_BASE = "https://api.electricitymap.org/v3"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_PAGE_SLEEP = 0.5      # seconds between paginated requests
_MAX_HISTORY_DAYS = 90  # free tier limit; paid tiers allow more

_SANDBOX_TRUTHY = frozenset({"1", "true", "yes", "on"})


def sandbox_enabled() -> bool:
    """Return True if ELECTRICITYMAPS_SANDBOX requests sandbox/randomized data."""
    return os.environ.get("ELECTRICITYMAPS_SANDBOX", "").strip().lower() in _SANDBOX_TRUTHY


def _redact(token: str) -> str:
    """Never expose the token. Show only that one is present."""
    return "<set>" if token else "<unset>"


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
        sandbox: Optional[bool] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ELECTRICITYMAPS_API_KEY", "")
        self._zone_map = zone_map or _DEFAULT_ZONE_MAP
        self._sandbox = sandbox_enabled() if sandbox is None else bool(sandbox)

    @property
    def source_name(self) -> str:
        return "electricitymaps"

    @property
    def is_sandbox(self) -> bool:
        return self._sandbox

    def __repr__(self) -> str:  # never leak the token
        return (
            f"ElectricityMapsCarbonProvider(api_key={_redact(self._api_key)}, "
            f"sandbox={self._sandbox})"
        )

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


# ---------------------------------------------------------------------------
# Provenance-aware provider (implements the MarketDataProvider interface)
# ---------------------------------------------------------------------------

_PROVIDER_NAME = "electricitymaps"


class ElectricityMapsProvider(MarketDataProvider):
    """Provenance-aware Electricity Maps adapter.

    Capabilities:
      * CARBON: historical carbon intensity for all mapped zones (aggregated;
        flagged ESTIMATED when EM marks the value estimated).
      * PRICE: Electricity Maps does NOT publish US ISO wholesale prices and
        does NOT publish nodal LMP anywhere. Price requests therefore return an
        empty series with a clear log; use the direct ISO/TSO providers
        (CAISO/PJM/ERCOT/ENTSO-E) for prices. Requesting a *_lmp market_type
        from this provider is always refused because EM only ever has zonal /
        bidding-zone prices, never true nodal LMP.

    Sandbox: when ELECTRICITYMAPS_SANDBOX is enabled (or sandbox=True), every
    point is flagged is_sandbox=True / provenance=SANDBOX and is therefore
    inadmissible for benchmark/savings claims (see market_data_provider).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        zone_map: Optional[dict[str, str]] = None,
        sandbox: Optional[bool] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ELECTRICITYMAPS_API_KEY", "")
        self._zone_map = zone_map or _DEFAULT_ZONE_MAP
        self._sandbox = sandbox_enabled() if sandbox is None else bool(sandbox)
        self._carbon = ElectricityMapsCarbonProvider(
            api_key=self._api_key, zone_map=self._zone_map, sandbox=self._sandbox,
        )

    def __repr__(self) -> str:  # never leak the token
        return f"ElectricityMapsProvider(api_key={_redact(self._api_key)}, sandbox={self._sandbox})"

    def is_sandbox(self) -> bool:
        return self._sandbox

    def validate_credentials(self) -> bool:
        """True if a key is present. Does not call the network when absent."""
        return bool(self._api_key)

    def get_capabilities(self) -> list[ProviderCapability]:
        regions = tuple(self._zone_map.keys())
        return [
            ProviderCapability(
                provider=_PROVIDER_NAME,
                signal=Signal.CARBON,
                regions=regions,
                granularity="hourly",
                history_supported=True,
                forecast_supported=False,
                sandbox_supported=True,
                production_supported=True,
                auth_required=True,
                market_types=(),
            ),
            ProviderCapability(
                provider=_PROVIDER_NAME,
                signal=Signal.PRICE,
                regions=(),  # EM is not a price source-of-truth for Aurelius
                granularity="hourly",
                history_supported=False,
                forecast_supported=False,
                sandbox_supported=True,
                production_supported=False,
                auth_required=True,
                market_types=(MarketType.DAY_AHEAD_PRICE,),
            ),
        ]

    def fetch_carbon_series(
        self,
        region: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[CarbonPoint]:
        df = self._carbon.fetch_carbon(region, start_dt, end_dt)
        provenance = Provenance.SANDBOX if self._sandbox else Provenance.AGGREGATED
        points: list[CarbonPoint] = []
        for _, row in df.iterrows():
            points.append(CarbonPoint(
                timestamp=row["timestamp"].to_pydatetime(),
                region=region,
                gco2_per_kwh=float(row["gco2_per_kwh"]),
                source="ENTSO-E/operator (via Electricity Maps)",
                provider=_PROVIDER_NAME,
                source_granularity="hourly",
                provenance=provenance,
                is_sandbox=self._sandbox,
                is_estimated=False,
                fetched_at=row["fetched_at"].to_pydatetime(),
            ))
        return points

    def fetch_price_series(
        self,
        region: str,
        start_dt: datetime,
        end_dt: datetime,
        market_type: str = MarketType.DAY_AHEAD_LMP,
    ) -> list[MarketPricePoint]:
        if market_type in (MarketType.DAY_AHEAD_LMP, MarketType.REAL_TIME_LMP):
            logger.warning(
                "Electricity Maps does not provide nodal LMP for any zone; "
                "use the direct ISO/TSO LMP providers (CAISO/PJM/ERCOT). "
                "Returning empty series for market_type=%s region=%s.",
                market_type, region,
            )
            return []
        logger.warning(
            "Electricity Maps is not a wholesale price source-of-truth for "
            "Aurelius (no US prices; EU prices come from ENTSO-E directly). "
            "Returning empty price series for region=%s.", region,
        )
        return []


__all__ = [
    "ElectricityMapsCarbonProvider",
    "ElectricityMapsProvider",
    "sandbox_enabled",
]
