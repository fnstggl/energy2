"""Build combined multi-period price dataset for extended-window benchmarks.

Merges existing price CSVs across multiple time periods into a single
combined dataset suitable for 90-day training windows.

Combines:
    data/summer2025/      → Summer 2025 (Jun-Aug 2025, ~92 days)
    data/fall2025/        → Fall 2025 (Sep-Dec 2025, ~122 days)
    data/q12026_*         → Q1 2026 (Jan-Mar 2026, ~73 days)

Output:
    data/combined_2025_2026/caiso_us_west_dam.csv   CAISO DA (Jun 2025 - Mar 2026)
    data/combined_2025_2026/pjm_us_east_dam.csv     PJM DA
    data/combined_2025_2026/ercot_us_south_dam.csv  ERCOT DA
    data/combined_2025_2026/caiso_us_west_rt.csv    CAISO RT
    data/combined_2025_2026/pjm_us_east_rt.csv      PJM RT
    data/combined_2025_2026/ercot_us_south_rt.csv   ERCOT RT
    data/combined_2025_2026/3region_dam.csv         3-region DA (all regions)
    data/combined_2025_2026/3region_rt.csv          3-region RT

Usage:
    python scripts/build_combined_dataset.py
    python scripts/build_combined_dataset.py --no-rt  # DA only (for regions without RT)

NOTE:
    This script merges REAL price data from 3 ISOs (CAISO, PJM, ERCOT).
    The combined dataset is suitable for savings benchmark claims, provided
    each source dataset is from the authoritative ISO API.
    If fall2025/ data is missing, the script will error with a clear message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

import pandas as pd

from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Run the fetch script first:\n"
            f"  python scripts/fetch_caiso_pjm_prices.py --start <START> --end <END> "
            f"--out-dir {path.parent}"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Normalize timestamp to UTC-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    print(f"  {label}: {len(df)} rows  "
          f"[{df['timestamp'].min().date()} → {df['timestamp'].max().date()}]")
    return df[PRICE_COLUMNS]


def _merge_and_dedup(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp", "region"], keep="last")
    combined = combined.sort_values(["region", "timestamp"]).reset_index(drop=True)
    dedup = before - len(combined)
    print(f"  {label}: {before} rows merged → {len(combined)} after dedup "
          f"({dedup} duplicates removed)")
    return combined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-rt", action="store_true", help="Skip RT price files")
    ap.add_argument("--out-dir", default="data/combined_2025_2026", help="Output directory")
    args = ap.parse_args()

    data = Path("data")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Building combined 2025-2026 price dataset ===\n")

    # -------------------------------------------------------------------------
    # CAISO us-west day-ahead
    # -------------------------------------------------------------------------
    print("CAISO us-west day-ahead:")
    caiso_da_frames = []
    for label, path in [
        ("summer2025", data / "summer2025" / "caiso_us_west_dam.csv"),
        ("fall2025",   data / "fall2025"   / "caiso_us_west_dam.csv"),
        ("q12026",     data / "caiso_us_west_dam.csv"),
    ]:
        caiso_da_frames.append(_load_csv(path, label))
    caiso_da = _merge_and_dedup(caiso_da_frames, "caiso_da_combined")
    caiso_da.to_csv(out / "caiso_us_west_dam.csv", index=False)

    # -------------------------------------------------------------------------
    # PJM us-east day-ahead
    # -------------------------------------------------------------------------
    print("\nPJM us-east day-ahead:")
    pjm_da_frames = []
    for label, path in [
        ("summer2025", data / "summer2025" / "pjm_us_east_dam.csv"),
        ("fall2025",   data / "fall2025"   / "pjm_us_east_dam.csv"),
        ("q12026",     data / "pjm_us_east_dam.csv"),
    ]:
        pjm_da_frames.append(_load_csv(path, label))
    pjm_da = _merge_and_dedup(pjm_da_frames, "pjm_da_combined")
    pjm_da.to_csv(out / "pjm_us_east_dam.csv", index=False)

    # -------------------------------------------------------------------------
    # ERCOT us-south day-ahead
    # -------------------------------------------------------------------------
    print("\nERCOT us-south day-ahead:")
    ercot_da_frames = []
    for label, path in [
        ("summer2025", data / "summer2025" / "ercot_us_south_dam.csv"),
        ("fall2025",   data / "fall2025"   / "ercot_us_south_dam.csv"),
        ("q12026",     data / "ercot_us_south_dam.csv"),
    ]:
        ercot_da_frames.append(_load_csv(path, label))
    ercot_da = _merge_and_dedup(ercot_da_frames, "ercot_da_combined")
    ercot_da.to_csv(out / "ercot_us_south_dam.csv", index=False)

    if not args.no_rt:
        # -------------------------------------------------------------------------
        # CAISO us-west real-time
        # -------------------------------------------------------------------------
        print("\nCAISO us-west real-time:")
        caiso_rt_frames = []
        for label, path in [
            ("summer2025", data / "summer2025" / "caiso_us_west_rt.csv"),
            ("fall2025",   data / "fall2025"   / "caiso_us_west_rt.csv"),
        ]:
            # Q1 2026 CAISO RT not always present; skip gracefully
            if path.exists():
                caiso_rt_frames.append(_load_csv(path, label))
        # Q1 2026 CAISO RT
        q1_caiso_rt = data / "caiso_us_west_rt.csv"
        if q1_caiso_rt.exists():
            caiso_rt_frames.append(_load_csv(q1_caiso_rt, "q12026"))
        if caiso_rt_frames:
            caiso_rt = _merge_and_dedup(caiso_rt_frames, "caiso_rt_combined")
            caiso_rt.to_csv(out / "caiso_us_west_rt.csv", index=False)

        # -------------------------------------------------------------------------
        # PJM us-east real-time
        # -------------------------------------------------------------------------
        print("\nPJM us-east real-time:")
        pjm_rt_frames = []
        for label, path in [
            ("summer2025", data / "summer2025" / "pjm_us_east_rt.csv"),
            ("fall2025",   data / "fall2025"   / "pjm_us_east_rt.csv"),
            ("q12026",     data / "pjm_us_east_rt.csv"),
        ]:
            if path.exists():
                pjm_rt_frames.append(_load_csv(path, label))
        if pjm_rt_frames:
            pjm_rt = _merge_and_dedup(pjm_rt_frames, "pjm_rt_combined")
            pjm_rt.to_csv(out / "pjm_us_east_rt.csv", index=False)

        # -------------------------------------------------------------------------
        # ERCOT us-south real-time
        # -------------------------------------------------------------------------
        print("\nERCOT us-south real-time:")
        ercot_rt_frames = []
        for label, path in [
            ("summer2025", data / "summer2025" / "ercot_us_south_rt.csv"),
            ("fall2025",   data / "fall2025"   / "ercot_us_south_rt.csv"),
            ("q12026",     data / "ercot_us_south_rt.csv"),
        ]:
            if path.exists():
                ercot_rt_frames.append(_load_csv(path, label))
        if ercot_rt_frames:
            ercot_rt = _merge_and_dedup(ercot_rt_frames, "ercot_rt_combined")
            ercot_rt.to_csv(out / "ercot_us_south_rt.csv", index=False)

    # -------------------------------------------------------------------------
    # 3-region combined DA (all regions in one file)
    # -------------------------------------------------------------------------
    print("\n3-region combined DA:")
    all_da = pd.concat([caiso_da, pjm_da, ercot_da], ignore_index=True)
    all_da = all_da.drop_duplicates(subset=["timestamp", "region"], keep="last")
    all_da = all_da.sort_values(["region", "timestamp"]).reset_index(drop=True)
    all_da.to_csv(out / "3region_dam.csv", index=False)
    print(f"  3region_dam: {len(all_da)} rows total")
    for reg, g in all_da.groupby("region"):
        print(f"    {reg}: {len(g)} rows  "
              f"[{g['timestamp'].min().date()} → {g['timestamp'].max().date()}]")

    if not args.no_rt:
        rt_parts = []
        for rt_path in [out / "caiso_us_west_rt.csv", out / "pjm_us_east_rt.csv", out / "ercot_us_south_rt.csv"]:
            if rt_path.exists():
                rt_df = pd.read_csv(rt_path, parse_dates=["timestamp"])
                if rt_df["timestamp"].dt.tz is None:
                    rt_df["timestamp"] = pd.to_datetime(rt_df["timestamp"], utc=True)
                else:
                    rt_df["timestamp"] = rt_df["timestamp"].dt.tz_convert("UTC")
                rt_parts.append(rt_df[PRICE_COLUMNS])
        if rt_parts:
            all_rt = pd.concat(rt_parts, ignore_index=True)
            all_rt = all_rt.drop_duplicates(subset=["timestamp", "region"], keep="last")
            all_rt = all_rt.sort_values(["region", "timestamp"]).reset_index(drop=True)
            all_rt.to_csv(out / "3region_rt.csv", index=False)
            print(f"  3region_rt: {len(all_rt)} rows total")

    print(f"\n[OK] Combined dataset written to {out}/")
    print("\nDate coverage summary:")
    print(f"  CAISO DA: {caiso_da['timestamp'].min().date()} → {caiso_da['timestamp'].max().date()}")
    print(f"  PJM DA:   {pjm_da['timestamp'].min().date()} → {pjm_da['timestamp'].max().date()}")
    print(f"  ERCOT DA: {ercot_da['timestamp'].min().date()} → {ercot_da['timestamp'].max().date()}")
    total_days = (all_da["timestamp"].max() - all_da["timestamp"].min()).days
    print(f"\n  Combined span: {total_days} days")
    if total_days >= 180:
        print("  [OK] ≥ 180 days — sufficient for 90-day training windows")
    else:
        print(f"  [WARNING] Only {total_days} days — may be insufficient for 90-day windows")
        print("  Fetch more data to enable extended benchmarks.")


if __name__ == "__main__":
    main()
