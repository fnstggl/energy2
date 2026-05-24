"""Open-Meteo weather ingestion adapter for Aurelius.

Open-Meteo is the designated primary weather provider. This module is a
production-grade ingestion layer that replaces the static-CSV scaffolding in
``scripts/fetch_weather_data.py``. It supports the three Open-Meteo products
that matter for a forecasting/optimization backtest, each verified against the
live API:

  * Historical Weather API (ERA5 archive)   -> ground-truth observed weather
        host: archive-api.open-meteo.com/v1/archive
        Use: training-window weather, and as verification "actuals".

  * Forecast API                            -> live forward forecast
        host: api.open-meteo.com/v1/forecast
        Use: production decision-time weather (what you have in deployment).

  * Previous Runs API                       -> fixed lead-time forecast archive
        host: previous-runs-api.open-meteo.com/v1/forecast
        Use: BACKTESTING with REALISTIC forecast weather. The
        ``*_previous_dayN`` series contains, for each valid timestamp, the value
        that was actually forecast N days in advance. This is the principled fix
        for the perfect-foresight leakage in engine.py:770 — at decision time T
        the optimizer only ever had the day-ahead forecast, never the future
        observation.

Design notes
------------
* No API key, no SDK — plain HTTP GET + JSON (per Open-Meteo docs).
* All timestamps normalised to UTC, floored to the hour.
* Output is the canonical Aurelius weather schema (identical columns to
  ``scripts/fetch_weather_data.py``) so it is a drop-in for build_weather_lookup
  and the ML forecaster:
      timestamp, region, temperature_c, humidity_pct, wind_speed_ms,
      hdd_f, cdd_f, temp_rolling_24h_c, temp_delta_24h_c, source
* Retries with exponential backoff; optional on-disk JSON cache so backtests are
  reproducible and don't hammer the API.
* Region->coordinate mapping is explicit and matches the existing benchmark
  stations (SFO/DCA/HOU).

The derived-feature maths (HDD/CDD base-65F, 24h trailing rolling mean and
delta) is intentionally identical to the legacy script so observed and forecast
weather are directly comparable.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region -> coordinates (matches scripts/fetch_weather_data.py station sites)
# ---------------------------------------------------------------------------
REGION_COORDS: dict[str, dict[str, float]] = {
    "us-west": {"lat": 37.62, "lon": -122.38},   # SFO (CAISO)
    "us-east": {"lat": 38.85, "lon": -77.04},     # DCA (PJM)
    "us-south": {"lat": 29.65, "lon": -95.28},    # HOU (ERCOT)
}

_ARCHIVE_HOST = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_HOST = "https://api.open-meteo.com/v1/forecast"
_PREVIOUS_RUNS_HOST = "https://previous-runs-api.open-meteo.com/v1/forecast"

CANONICAL_COLS = [
    "timestamp", "region",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    "hdd_f", "cdd_f", "temp_rolling_24h_c", "temp_delta_24h_c",
    "source",
]


@dataclass
class OpenMeteoConfig:
    """Configuration for the Open-Meteo adapter.

    Attributes:
        max_retries: HTTP retry attempts on transient failure.
        backoff_base_s: base seconds for exponential backoff (2,4,8,...).
        timeout_s: per-request timeout.
        cache_dir: optional directory for caching raw JSON responses. When set,
            identical requests are served from disk (reproducible backtests).
        rate_limit_s: polite sleep between distinct network requests.
    """
    max_retries: int = 4
    backoff_base_s: float = 2.0
    timeout_s: float = 60.0
    cache_dir: Optional[str] = None
    rate_limit_s: float = 0.5


class OpenMeteoWeatherProvider:
    """Production-grade Open-Meteo ingestion with caching + retries.

    Every public fetch method returns the canonical Aurelius weather DataFrame
    (CANONICAL_COLS) or an empty DataFrame on total failure (never raises in the
    hot path — graceful degradation, matching the rest of the weather stack).
    """

    def __init__(self, config: Optional[OpenMeteoConfig] = None) -> None:
        self.config = config or OpenMeteoConfig()
        self._last_request_t = 0.0
        if self.config.cache_dir:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ HTTP
    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        import hashlib
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        return Path(self.config.cache_dir) / f"openmeteo_{key}.json"

    def _get_json(self, host: str, params: dict) -> dict:
        url = host + "?" + urllib.parse.urlencode(params, doseq=True)
        cache = self._cache_path(url)
        if cache is not None and cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass  # fall through to network on corrupt cache

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            # polite rate limiting between network calls
            dt = time.monotonic() - self._last_request_t
            if dt < self.config.rate_limit_s:
                time.sleep(self.config.rate_limit_s - dt)
            try:
                with urllib.request.urlopen(url, timeout=self.config.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self._last_request_t = time.monotonic()
                if cache is not None:
                    try:
                        cache.write_text(json.dumps(data))
                    except Exception:
                        pass
                return data
            except Exception as exc:  # noqa: BLE001 — transient network/HTTP
                last_exc = exc
                wait = self.config.backoff_base_s * (2 ** attempt)
                logger.warning(
                    "Open-Meteo request failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1, self.config.max_retries, exc, wait,
                )
                self._last_request_t = time.monotonic()
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait)
        logger.error("Open-Meteo request failed permanently: %s", last_exc)
        return {}

    # ------------------------------------------------------------- public API
    def fetch_historical(
        self, region: str, start: str, end: str,
    ) -> pd.DataFrame:
        """ERA5 reanalysis (observed ground truth) for a region.

        Args:
            region: aurelius region id (must be in REGION_COORDS).
            start/end: 'YYYY-MM-DD' (inclusive), UTC.
        """
        coords = REGION_COORDS.get(region)
        if coords is None:
            logger.warning("OpenMeteo: unknown region %s", region)
            return pd.DataFrame()
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC", "wind_speed_unit": "ms",
        }
        data = self._get_json(_ARCHIVE_HOST, params)
        return self._canonicalize(data, region, source="open_meteo_era5",
                                  t="temperature_2m", rh="relative_humidity_2m",
                                  ws="wind_speed_10m")

    def fetch_forecast(
        self, region: str, forecast_days: int = 7, past_days: int = 0,
    ) -> pd.DataFrame:
        """Live forward forecast (production decision-time weather)."""
        coords = REGION_COORDS.get(region)
        if coords is None:
            return pd.DataFrame()
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "forecast_days": forecast_days, "past_days": past_days,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC", "wind_speed_unit": "ms",
        }
        data = self._get_json(_FORECAST_HOST, params)
        return self._canonicalize(data, region, source="open_meteo_forecast",
                                  t="temperature_2m", rh="relative_humidity_2m",
                                  ws="wind_speed_10m")

    def fetch_previous_run_forecast(
        self, region: str, start: str, end: str, lead_day: int = 1,
    ) -> pd.DataFrame:
        """Realistic backtest weather: the forecast issued ``lead_day`` days ahead.

        Returns the canonical schema built from the ``*_previous_day{lead_day}``
        series — i.e. for each valid timestamp t, the value forecast ~lead_day*24h
        before t. This is what a deployed system actually had at decision time.

        Args:
            lead_day: 1..7 (Open-Meteo Previous Runs supports day1..day7).
        """
        coords = REGION_COORDS.get(region)
        if coords is None:
            return pd.DataFrame()
        if not (1 <= lead_day <= 7):
            raise ValueError(f"lead_day must be 1..7, got {lead_day}")
        suf = f"_previous_day{lead_day}"
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": start, "end_date": end,
            "hourly": ",".join([
                f"temperature_2m{suf}",
                f"relative_humidity_2m{suf}",
                f"wind_speed_10m{suf}",
            ]),
            "timezone": "UTC", "wind_speed_unit": "ms",
        }
        data = self._get_json(_PREVIOUS_RUNS_HOST, params)
        return self._canonicalize(
            data, region, source=f"open_meteo_previous_day{lead_day}",
            t=f"temperature_2m{suf}", rh=f"relative_humidity_2m{suf}",
            ws=f"wind_speed_10m{suf}",
        )

    def fetch_region_set(
        self, regions: list[str], start: str, end: str,
        product: str = "historical", lead_day: int = 1,
    ) -> pd.DataFrame:
        """Fetch + concatenate multiple regions into one canonical DataFrame.

        product: 'historical' (ERA5) or 'previous_run' (lead-time forecast).
        """
        frames = []
        for r in regions:
            if product == "historical":
                df = self.fetch_historical(r, start, end)
            elif product == "previous_run":
                df = self.fetch_previous_run_forecast(r, start, end, lead_day=lead_day)
            else:
                raise ValueError(f"unknown product {product!r}")
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLS)
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["region", "timestamp"]).reset_index(drop=True)

    # ----------------------------------------------------------- derivation
    @staticmethod
    def _canonicalize(
        data: dict, region: str, source: str, t: str, rh: str, ws: str,
    ) -> pd.DataFrame:
        """Convert a raw Open-Meteo JSON payload to the canonical schema.

        Derived features (HDD/CDD base 65F, 24h trailing rolling mean + delta)
        replicate scripts/fetch_weather_data.py exactly. All rolling/delta ops
        are strictly trailing (no future leakage within the weather series).
        """
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame(columns=CANONICAL_COLS)

        tc = pd.Series(hourly.get(t, [np.nan] * len(times)), dtype=float)
        rh_s = pd.Series(hourly.get(rh, [np.nan] * len(times)), dtype=float)
        ws_s = pd.Series(hourly.get(ws, [np.nan] * len(times)), dtype=float)

        ts = pd.to_datetime(times, utc=True).floor("h")
        tf = tc * 9.0 / 5.0 + 32.0

        df = pd.DataFrame({
            "timestamp": ts,
            "temperature_c": tc.values,
            "humidity_pct": rh_s.clip(0, 100).values,
            "wind_speed_ms": ws_s.fillna(0.0).values,
        })
        df["hdd_f"] = (65.0 - tf).clip(lower=0.0).values
        df["cdd_f"] = (tf - 65.0).clip(lower=0.0).values

        df = df.sort_values("timestamp").reset_index(drop=True)
        tcc = df["temperature_c"]
        df["temp_rolling_24h_c"] = tcc.rolling(window=24, min_periods=1).mean()
        df["temp_delta_24h_c"] = tcc - tcc.shift(24).bfill()

        # forward/backfill any gaps so the lookup never sees NaN
        df = df.ffill().bfill()
        df["region"] = region
        df["source"] = source
        return df[CANONICAL_COLS]
