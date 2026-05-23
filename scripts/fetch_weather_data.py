#!/usr/bin/env python3
"""Fetch historical hourly weather data for Aurelius benchmark regions.

Primary source: Iowa Environmental Mesonet (IEM) ASOS network — free, no API key.
  - METAR observation data for major airports in each region
  - Covers KHOU (Houston/ERCOT), KSFO (San Francisco/CAISO), KDCA (DC/PJM)
  - Full hourly temperature, dew point, wind speed coverage

Fallback: Open-Meteo ERA5 archive (https://archive-api.open-meteo.com) — tried
  automatically when IEM fails.

Usage:
    python scripts/fetch_weather_data.py --all
    python scripts/fetch_weather_data.py --start 2026-01-01 --end 2026-03-31 \
        --output data/weather_q12026.csv

Output schema (canonical Aurelius weather format):
    timestamp,region,temperature_c,humidity_pct,wind_speed_ms,
    hdd_f,cdd_f,temp_rolling_24h_c,temp_delta_24h_c,source

Notes:
    - hdd_f: heating degree days proxy = max(0, 65°F - temp_f) per hour
    - cdd_f: cooling degree days proxy = max(0, temp_f - 65°F) per hour
    - temp_rolling_24h_c: 24h rolling mean temperature (regime detection)
    - temp_delta_24h_c: temperature change vs 24h ago (cold snap detection)
    - wind_speed_ms: wind speed in m/s (ERCOT wind generation proxy)
    - All timestamps are UTC (floored to the hour)
    - Missing hours are forward-filled then backward-filled
    - source column: 'iem_asos_metar' or 'open_meteo_archive'
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Station mappings: Aurelius region → airport ICAO code
# ---------------------------------------------------------------------------
REGION_STATIONS = {
    "us-west":  "KSFO",   # San Francisco International
    "us-east":  "KDCA",   # Reagan National, Washington DC
    "us-south": "KHOU",   # Houston Hobby Airport (ERCOT core demand node)
}

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

STANDARD_PERIODS = [
    {
        "name": "q12026",
        "start": "2025-12-15",   # extra context window before backtest start 2026-01-01
        "end": "2026-03-31",
        "output": "data/weather_q12026.csv",
    },
    {
        "name": "summer2025",
        "start": "2025-05-01",   # extra context window before backtest start 2025-06-01
        "end": "2025-09-01",
        "output": "data/summer2025/weather_summer2025.csv",
    },
]


# ---------------------------------------------------------------------------
# IEM ASOS fetcher (primary source)
# ---------------------------------------------------------------------------

def _fetch_iem(station: str, start: str, end: str) -> pd.DataFrame:
    """Fetch METAR hourly observations from IEM ASOS for one station.

    Args:
        station: ICAO station code (e.g. "KHOU")
        start:   "YYYY-MM-DD"
        end:     "YYYY-MM-DD"

    Returns:
        DataFrame with columns: timestamp, tmpf, dwpf, sknt (raw IEM fields)
        Timestamps are UTC datetime objects floored to the hour.
        Empty on error.
    """
    import urllib.request, urllib.parse

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    params = {
        "station":       station,
        "data":          "tmpf,dwpf,sknt",
        "year1":         str(s.year),
        "month1":        str(s.month),
        "day1":          str(s.day),
        "year2":         str(e.year),
        "month2":        str(e.month),
        "day2":          str(e.day),
        "tz":            "UTC",
        "format":        "onlycomma",
        "latlon":        "no",
        "direct":        "no",
        "report_type":   "2",  # METAR only — on-the-hour, full observations
    }
    url = IEM_ASOS_URL + "?" + urllib.parse.urlencode(params, doseq=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"    IEM fetch failed for {station}: {exc}")
        return pd.DataFrame()

    lines = [l for l in raw.strip().split("\n") if not l.startswith("#") and "," in l]
    if len(lines) <= 1:
        return pd.DataFrame()

    # header row is "station,valid,tmpf,dwpf,sknt"
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ts = pd.Timestamp(parts[1].strip(), tz="UTC")
            tmpf = float(parts[2]) if parts[2].strip() not in ("M", "") else float("nan")
            dwpf = float(parts[3]) if parts[3].strip() not in ("M", "") else float("nan")
            sknt = float(parts[4]) if len(parts) > 4 and parts[4].strip() not in ("M", "") else float("nan")
            rows.append({"timestamp": ts, "tmpf": tmpf, "dwpf": dwpf, "sknt": sknt})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Floor to hour (METAR times are at :53 or similar minutes)
    df["timestamp"] = df["timestamp"].dt.floor("h")
    # Average observations within the same hour (in case of duplicates)
    df = df.groupby("timestamp").mean().reset_index()
    return df


# ---------------------------------------------------------------------------
# Open-Meteo fallback
# ---------------------------------------------------------------------------

def _fetch_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo ERA5 archive as fallback.

    Returns DataFrame with columns: timestamp, tmpf, dwpf, sknt (same schema).
    """
    import urllib.request, urllib.parse

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone":   "UTC",
    }
    url = OPENMETEO_ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"    Open-Meteo fallback failed: {exc}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    tc_arr = hourly.get("temperature_2m", [float("nan")] * len(times))
    rh_arr = hourly.get("relative_humidity_2m", [float("nan")] * len(times))
    ws_ms_arr = hourly.get("wind_speed_10m", [float("nan")] * len(times))  # already m/s

    tc = pd.Series(tc_arr, dtype=float)
    tf = tc * 9.0 / 5.0 + 32.0  # Celsius → Fahrenheit

    # Estimate dew point from RH using Magnus approximation
    rh = pd.Series(rh_arr, dtype=float).clip(1, 100)
    a, b = 17.62, 243.12
    gamma = np.log(rh / 100.0) + (a * tc) / (b + tc)
    td_c = (b * gamma) / (a - gamma)
    td_f = td_c * 9.0 / 5.0 + 32.0

    ws_knots = pd.Series(ws_ms_arr, dtype=float) / 0.514444  # m/s → knots

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(times, utc=True),
        "tmpf":      tf.values,
        "dwpf":      td_f.values,
        "sknt":      ws_knots.values,
    })
    return df


# ---------------------------------------------------------------------------
# Physics utilities
# ---------------------------------------------------------------------------

REGION_COORDS = {
    "us-west":  {"lat": 37.62, "lon": -122.38},   # SFO airport
    "us-east":  {"lat": 38.85, "lon": -77.04},    # DCA airport
    "us-south": {"lat": 29.65, "lon": -95.28},    # HOU airport
}

_KNOTS_TO_MS = 0.514444


def _fahrenheit_to_celsius(tf: pd.Series) -> pd.Series:
    return (tf - 32.0) * 5.0 / 9.0


def _dewpoint_to_rh(tc: pd.Series, td_c: pd.Series) -> pd.Series:
    """Estimate relative humidity from air temperature and dew point (Magnus formula)."""
    a, b = 17.62, 243.12
    num = np.exp(a * td_c / (b + td_c))
    den = np.exp(a * tc  / (b + tc))
    rh = 100.0 * num / den
    return rh.clip(0, 100)


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw IEM columns to canonical Aurelius weather schema."""
    tc = _fahrenheit_to_celsius(df["tmpf"])
    tf = df["tmpf"]

    # Dew point → humidity
    td_f = df["dwpf"].fillna(tf - 20)  # rough fallback: ~40% RH
    td_c = _fahrenheit_to_celsius(td_f)
    rh = _dewpoint_to_rh(tc, td_c)

    ws_ms = df["sknt"].fillna(0.0) * _KNOTS_TO_MS  # knots → m/s

    df2 = pd.DataFrame({
        "timestamp":     df["timestamp"],
        "temperature_c": tc.values,
        "humidity_pct":  rh.values,
        "wind_speed_ms": ws_ms.values,
    })

    df2["hdd_f"] = (65.0 - tf).clip(lower=0.0).values
    df2["cdd_f"] = (tf - 65.0).clip(lower=0.0).values

    # 24h rolling mean and delta — strictly trailing, no leakage
    df2 = df2.sort_values("timestamp").reset_index(drop=True)
    tc2 = df2["temperature_c"]
    df2["temp_rolling_24h_c"] = tc2.rolling(window=24, min_periods=1).mean()
    df2["temp_delta_24h_c"]   = tc2 - tc2.shift(24).bfill()

    return df2


def _hourly_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Create a complete hourly UTC timestamp grid from start to end (inclusive)."""
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        freq="h",
    )


def _fill_to_hourly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Resample to a regular hourly UTC grid; forward-fill then backward-fill gaps."""
    grid = _hourly_grid(start, end)
    df = df.set_index("timestamp")
    df = df.reindex(grid)
    df = df.ffill().bfill()
    df.index.name = "timestamp"
    return df.reset_index()


# ---------------------------------------------------------------------------
# Per-region fetcher (IEM primary, Open-Meteo fallback)
# ---------------------------------------------------------------------------

def fetch_region(region: str, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather for one region and return canonical DataFrame."""
    station = REGION_STATIONS.get(region)
    if station is None:
        print(f"    WARNING: no station defined for region '{region}'")
        return pd.DataFrame()

    print(f"    Trying IEM ASOS ({station}) ...", flush=True)
    raw = _fetch_iem(station, start, end)
    source = "iem_asos_metar"

    if raw.empty:
        coords = REGION_COORDS.get(region, {})
        lat = coords.get("lat", 0)
        lon = coords.get("lon", 0)
        print(f"    IEM failed, trying Open-Meteo ({lat},{lon}) ...", flush=True)
        raw = _fetch_openmeteo(lat, lon, start, end)
        source = "open_meteo_archive"

    if raw.empty:
        print(f"    ERROR: all sources failed for {region}")
        return pd.DataFrame()

    # Ensure timestamp column exists
    if "timestamp" not in raw.columns and raw.index.name == "timestamp":
        raw = raw.reset_index()

    df = _compute_derived(raw)
    df = _fill_to_hourly(df, start, end)
    df["region"] = region
    df["source"] = source

    n_orig = len(raw)
    n_expected = len(_hourly_grid(start, end))
    pct = 100 * n_orig / n_expected if n_expected > 0 else 0
    print(f"    OK: {n_orig} raw obs → {len(df)} hourly rows ({pct:.0f}% direct obs)")

    return df


# ---------------------------------------------------------------------------
# Main fetch-and-save
# ---------------------------------------------------------------------------

CANONICAL_COLS = [
    "timestamp", "region",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    "hdd_f", "cdd_f", "temp_rolling_24h_c", "temp_delta_24h_c",
    "source",
]


def fetch_and_save(
    start: str,
    end: str,
    output: str,
    regions: list[str] | None = None,
) -> bool:
    """Fetch weather data for all regions and save to CSV.

    Returns:
        True if at least partial data was saved.
    """
    if regions is None:
        regions = list(REGION_STATIONS.keys())

    all_dfs: list[pd.DataFrame] = []
    success_count = 0

    for i, region in enumerate(regions):
        print(f"  Fetching {region} ({start} → {end}) ...", flush=True)
        if i > 0:
            time.sleep(1.5)  # polite rate limiting for IEM
        df = fetch_region(region, start, end)
        if df.empty:
            print(f"    WARNING: empty result for {region}")
            continue
        all_dfs.append(df)
        success_count += 1

    if not all_dfs:
        print("ERROR: no data fetched for any region — output not written")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["region", "timestamp"]).reset_index(drop=True)
    combined = combined[CANONICAL_COLS]
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    out_path = _REPO_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows → {out_path}")
    print(f"Coverage: {success_count}/{len(regions)} regions")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical weather data for Aurelius benchmark regions"
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output CSV path (relative to repo root)")
    parser.add_argument(
        "--all", action="store_true", help="Fetch all standard benchmark periods"
    )
    parser.add_argument(
        "--regions", nargs="*", default=None,
        help="Regions to fetch (default: all 3 regions)"
    )
    args = parser.parse_args()

    if args.all:
        for period in STANDARD_PERIODS:
            print(f"\n=== Fetching weather for {period['name']} ===")
            fetch_and_save(
                start=period["start"],
                end=period["end"],
                output=period["output"],
                regions=args.regions,
            )
    elif args.start and args.end and args.output:
        fetch_and_save(
            start=args.start,
            end=args.end,
            output=args.output,
            regions=args.regions,
        )
    else:
        parser.error("Provide --start/--end/--output or --all")


if __name__ == "__main__":
    main()
