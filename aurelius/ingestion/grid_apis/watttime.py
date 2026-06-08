"""WattTime carbon intensity adapter (API v3).

Fetches marginal operating emissions rate (MOER) from the WattTime API v3.
MOER represents the carbon intensity of the MARGINAL generating unit, not the
average grid carbon intensity. This distinction is important: MOER is the
correct metric for evaluating the incremental carbon cost of flexible load
shifting (which is Aurelius's use case).

Environment variables required:
    WATTTIME_USERNAME  – WattTime account username
    WATTTIME_PASSWORD  – WattTime account password

Optional:
    WATTTIME_API_TOKEN – Pre-cached bearer token (skips login request if fresh)

Region mapping (Aurelius → WattTime balancing authority):
    Resolved from the ONE authoritative map in
    ``aurelius.carbon.regions.watttime_ba_map`` (derived from the canonical
    ``region_registry``), so the client, optimizer and evaluator never diverge.
    Current mapped regions: us-west→CAISO_NP15, us-east→PJM_DOM,
    us-south→ERCOT_HOUSTON, us-north→MISO_INDIANAPOLIS.

Override with ba_map= constructor argument (e.g. for tests).

Units:
    WattTime MOER is returned in lbs CO2 / MWh.
    This adapter converts to gCO2 / kWh (canonical Aurelius unit):
        gCO2/kWh = lbs_CO2/MWh × (453.592 g/lb) / (1000 kWh/MWh)
                 = lbs_CO2/MWh × 0.453592

Signal type note:
    This adapter uses signal_type=co2_moer (marginal). For average intensity use
    signal_type=co2_aoer if your WattTime plan includes it. The source column in
    the output DataFrame is set to "watttime_moer" to make the signal type visible.

Rate limits:
    Free tier: limited to ~3 months historical data and rate limits apply.
    Retry with exponential back-off on 429 responses.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from .base import (
    CarbonProvider,
    ProviderConfigError,
    empty_carbon_df,
    normalize_carbon_df,
)

logger = logging.getLogger(__name__)

_WATTTIME_LOGIN_URL = "https://api.watttime.org/login"
_WATTTIME_HIST_URL = "https://api.watttime.org/v3/historical"
_WATTTIME_FORECAST_URL = "https://api.watttime.org/v3/forecast"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_PAGE_SLEEP = 0.5
_CHUNK_DAYS = 30        # WattTime free-tier historical window

# Conversion: lbs CO2/MWh → gCO2/kWh
_LBS_PER_MWH_TO_GCO2_PER_KWH = 453.592 / 1000.0


def _default_ba_map() -> dict[str, str]:
    """The ONE authoritative Aurelius-region → WattTime BA map.

    Sourced from ``aurelius.carbon.regions`` (derived from region_registry) so
    there is no second, divergent BA table living in this client (the prior bug).
    """
    from ...carbon.regions import watttime_ba_map
    return watttime_ba_map()


class WattTimeCarbonProvider(CarbonProvider):
    """Fetch marginal carbon intensity (MOER) from WattTime API v3.

    Args:
        username: WattTime account username.
        password: WattTime account password.
        ba_map:   Override mapping of Aurelius region → WattTime BA code.
        signal_type: "co2_moer" (marginal, default) or "co2_aoer" (average).
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ba_map: Optional[dict[str, str]] = None,
        signal_type: str = "co2_moer",
    ) -> None:
        self._username = username or os.environ.get("WATTTIME_USERNAME", "")
        self._password = password or os.environ.get("WATTTIME_PASSWORD", "")
        self._ba_map = ba_map or _default_ba_map()
        self._signal_type = signal_type
        self._token: Optional[str] = None

    @property
    def source_name(self) -> str:
        return f"watttime_{self._signal_type}"

    def _get_token(self) -> str:
        """Obtain or return cached bearer token."""
        if self._token:
            return self._token

        try:
            resp = requests.get(
                _WATTTIME_LOGIN_URL,
                auth=(self._username, self._password),
                timeout=30,
            )
            if resp.status_code in (401, 403):
                raise ProviderConfigError(
                    "WattTime authentication failed. "
                    "Check WATTTIME_USERNAME and WATTTIME_PASSWORD."
                )
            resp.raise_for_status()
            self._token = resp.json()["token"]
            return self._token
        except ProviderConfigError:
            raise
        except Exception as exc:
            raise ProviderConfigError(f"WattTime login failed: {exc}") from exc

    def fetch_carbon(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly marginal carbon intensity for *region* in [start, end).

        Raises:
            ProviderConfigError: If WATTTIME_USERNAME or WATTTIME_PASSWORD is not set,
                                 or if authentication fails.
        """
        if not self._username or not self._password:
            raise ProviderConfigError(
                "WATTTIME_USERNAME and WATTTIME_PASSWORD are required. "
                "Export them as environment variables or pass username=/password= "
                "to WattTimeCarbonProvider."
            )

        ba = self._ba_map.get(region)
        if ba is None:
            logger.warning(
                f"Region '{region}' not in WattTime BA map; "
                f"available: {list(self._ba_map)}"
            )
            return empty_carbon_df()

        def _to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        all_rows: list[dict] = []

        current = start_utc
        while current < end_utc:
            chunk_end = min(current + timedelta(days=_CHUNK_DAYS), end_utc)

            params = {
                "region": ba,
                "start": current.strftime("%Y-%m-%dT%H:%M+00:00"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M+00:00"),
                "signal_type": self._signal_type,
            }

            for attempt in range(_MAX_RETRIES):
                try:
                    resp = requests.get(
                        _WATTTIME_HIST_URL,
                        params=params,
                        headers=headers,
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        wait = _RETRY_BACKOFF * (2 ** attempt)
                        logger.warning(f"WattTime rate limit; retrying in {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    if resp.status_code in (401, 403):
                        # Token may have expired – clear and re-auth on next call
                        self._token = None
                        raise ProviderConfigError(
                            f"WattTime auth rejected ({resp.status_code}). "
                            "Check WATTTIME_USERNAME and WATTTIME_PASSWORD."
                        )
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except ProviderConfigError:
                    raise
                except Exception as exc:
                    if attempt == _MAX_RETRIES - 1:
                        logger.error(f"WattTime request failed: {exc}")
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            else:
                current = chunk_end
                time.sleep(_PAGE_SLEEP)
                continue

            # WattTime returns 5-min intervals; resample to hourly mean
            chunk_rows: list[dict] = []
            for point in payload.get("data", []):
                try:
                    ts = pd.Timestamp(point["point_time"]).tz_convert("UTC")
                    lbs_per_mwh = float(point["value"])
                    gco2_per_kwh = lbs_per_mwh * _LBS_PER_MWH_TO_GCO2_PER_KWH
                    chunk_rows.append({"timestamp": ts, "gco2_per_kwh": gco2_per_kwh})
                except (KeyError, ValueError, TypeError):
                    continue

            if chunk_rows:
                chunk_df = pd.DataFrame(chunk_rows)
                # Resample to hourly mean (WattTime provides 5-min resolution)
                chunk_df = chunk_df.set_index("timestamp").resample("h").mean().reset_index()
                for _, row in chunk_df.iterrows():
                    if pd.notna(row["gco2_per_kwh"]):
                        all_rows.append({
                            "timestamp": row["timestamp"],
                            "region": region,
                            "gco2_per_kwh": row["gco2_per_kwh"],
                        })

            current = chunk_end
            time.sleep(_PAGE_SLEEP)

        if not all_rows:
            logger.warning(
                f"WattTime returned no data for ba={ba} "
                f"{start_utc}..{end_utc} signal={self._signal_type}"
            )
            return empty_carbon_df()

        df = pd.DataFrame(all_rows)
        return normalize_carbon_df(df, source=self.source_name, granularity="hourly")

    def fetch_forecast_carbon(
        self,
        region: str,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """Fetch forward-looking marginal carbon (MOER) for scheduling decisions.

        Uses WattTime's ``/v3/forecast`` endpoint, which returns the predicted
        MOER for the next ``horizon_hours`` from now. This is the signal the
        scheduler must use to make FORWARD decisions; historical MOER
        (``fetch_carbon``) is for realized replay/reporting — never mix them.

        Units are normalized to gCO2/kWh and timestamps to UTC, identically to
        ``fetch_carbon``. The source column is suffixed ``_forecast`` so a
        forecast value can never be mistaken for a settled historical value.

        Raises:
            ProviderConfigError: If credentials are missing or auth fails. Never
            silently substitutes a default emissions rate.
        """
        if not self._username or not self._password:
            raise ProviderConfigError(
                "WATTTIME_USERNAME and WATTTIME_PASSWORD are required for forecast MOER. "
                "Export them or pass username=/password=."
            )

        ba = self._ba_map.get(region)
        if ba is None:
            logger.warning(
                f"Region '{region}' has no WattTime BA (carbon_unavailable); "
                f"available: {list(self._ba_map)}"
            )
            return empty_carbon_df()

        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "region": ba,
            "signal_type": self._signal_type,
            "horizon_hours": int(horizon_hours),
        }

        payload = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(
                    _WATTTIME_FORECAST_URL, params=params, headers=headers, timeout=60
                )
                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"WattTime rate limit; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if resp.status_code in (401, 403):
                    self._token = None
                    raise ProviderConfigError(
                        f"WattTime auth rejected ({resp.status_code}) on forecast. "
                        "Check WATTTIME_USERNAME and WATTTIME_PASSWORD."
                    )
                resp.raise_for_status()
                payload = resp.json()
                break
            except ProviderConfigError:
                raise
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"WattTime forecast request failed: {exc}")
                    raise ProviderConfigError(
                        f"WattTime forecast fetch failed: {exc}"
                    ) from exc
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))

        rows: list[dict] = []
        # v3 forecast payloads put the series under "forecast" or "data".
        series = (payload or {}).get("forecast") or (payload or {}).get("data") or []
        for point in series:
            try:
                ts = pd.Timestamp(point["point_time"]).tz_convert("UTC")
                lbs_per_mwh = float(point["value"])
                rows.append({
                    "timestamp": ts,
                    "region": region,
                    "gco2_per_kwh": lbs_per_mwh * _LBS_PER_MWH_TO_GCO2_PER_KWH,
                })
            except (KeyError, ValueError, TypeError):
                continue

        if not rows:
            logger.warning(
                f"WattTime forecast returned no data for ba={ba} signal={self._signal_type}"
            )
            return empty_carbon_df()

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp").resample("h").mean(numeric_only=True)
        df["region"] = region
        df = df.reset_index()
        return normalize_carbon_df(
            df, source=f"{self.source_name}_forecast", granularity="hourly"
        )
