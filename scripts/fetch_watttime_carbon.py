#!/usr/bin/env python3
"""Fetch WattTime MOER (marginal carbon intensity) for Aurelius benchmark regions.

Fetches co2_moer for CAISO_NP15, PJM, and ERCOT for the date ranges used in
backtesting benchmarks and saves hourly carbon CSVs.

Usage:
    python scripts/fetch_watttime_carbon.py --start 2026-01-01 --end 2026-03-15 \
        --output data/watttime_carbon_q12026.csv

    python scripts/fetch_watttime_carbon.py --start 2025-06-01 --end 2025-09-01 \
        --output data/summer2025/watttime_carbon_summer2025.csv

    python scripts/fetch_watttime_carbon.py --all  # fetch all standard periods

Environment variables required:
    WATTTIME_USERNAME
    WATTTIME_PASSWORD

Output schema (canonical Aurelius carbon format):
    timestamp,region,gco2_per_kwh,source

Note: WattTime free tier provides ~3 months of historical data.
      Older data requires a paid plan. Missing data → rows are omitted
      (caller should check coverage before using for benchmark claims).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

import pandas as pd

from aurelius.ingestion.grid_apis.watttime import WattTimeCarbonProvider

# Aurelius region → WattTime region key (for fetch_carbon)
REGION_MAP = {
    "us-west":  "us-west",
    "us-east":  "us-east",
    "us-south": "us-south",
}

STANDARD_PERIODS = [
    {
        "name": "q12026",
        "start": "2026-01-01",
        "end": "2026-03-15",
        "output": "data/watttime_carbon_q12026.csv",
    },
    {
        "name": "summer2025",
        "start": "2025-06-01",
        "end": "2025-09-01",
        "output": "data/summer2025/watttime_carbon_summer2025.csv",
    },
]


def fetch_and_save(
    start: str,
    end: str,
    output: str,
    regions: list[str] | None = None,
) -> bool:
    """Fetch WattTime carbon data for all regions in [start, end) and save to CSV.

    Returns:
        True if at least partial data was fetched, False if all regions failed.
    """
    if regions is None:
        regions = list(REGION_MAP.keys())

    provider = WattTimeCarbonProvider()

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    all_dfs: list[pd.DataFrame] = []
    success_count = 0

    for region in regions:
        print(f"  Fetching {region} ({start} → {end}) ...", flush=True)
        try:
            df = provider.fetch_carbon(region=region, start=start_dt, end=end_dt)
            if df.empty:
                print(f"    WARNING: empty result for {region}")
                continue
            # Resample to hourly (WattTime provides 5-min data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            hourly = df.resample("h").mean(numeric_only=True)
            hourly = hourly.reset_index()
            # Drop hours with zero gco2 — these are resampling artifacts from
            # periods where the API returned no data (not real zero-carbon periods)
            hourly = hourly[hourly["gco2_per_kwh"] > 0.0]
            hourly["region"] = region
            hourly["source"] = "watttime_moer"
            all_dfs.append(hourly)
            success_count += 1
            print(f"    OK: {len(hourly)} hourly rows")
        except Exception as exc:
            print(f"    ERROR fetching {region}: {exc}")

    if not all_dfs:
        print("ERROR: no data fetched for any region — output not written")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    # Ensure canonical schema
    combined = combined[["timestamp", "region", "gco2_per_kwh", "source"]]
    combined = combined.sort_values(["region", "timestamp"]).reset_index(drop=True)
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    out_path = _REPO_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    print(f"Coverage: {success_count}/{len(regions)} regions")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch WattTime carbon data for Aurelius benchmarks")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output CSV path (relative to repo root)")
    parser.add_argument("--all", action="store_true", help="Fetch all standard benchmark periods")
    parser.add_argument(
        "--regions", nargs="*", default=None,
        help="Regions to fetch (default: us-west us-east us-south)"
    )
    args = parser.parse_args()

    if args.all:
        for period in STANDARD_PERIODS:
            print(f"\n=== Fetching {period['name']} ===")
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
