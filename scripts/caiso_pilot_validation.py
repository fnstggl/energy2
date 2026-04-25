"""CAISO pilot validation — fetch and validate real CAISO LMP data.

Supports two modes:

1. Day-ahead backtest (historical PRC_LMP / DAM):
       python scripts/caiso_pilot_validation.py --mode day_ahead \\
           --start 2024-01-01 --end 2024-02-01

2. Real-time monitoring snapshot (recent PRC_INTVL_LMP / RTM):
       python scripts/caiso_pilot_validation.py --mode real_time \\
           --hours 1

Outputs are saved to reports/pilot_validation/caiso/.

Carbon:
    If ELECTRICITYMAPS_API_KEY or WATTTIME credentials are absent the script
    runs in cost_only mode and prints:
        "Carbon data unavailable — running in cost_only mode."

CAISO endpoint:
    https://oasis.caiso.com/oasisapi/SingleZip
    No API key required.

Node: TH_NP15_GEN-APND (NP15 trading hub, Northern California)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "pilot_validation" / "caiso"


def _carbon_available() -> bool:
    em_key = os.environ.get("ELECTRICITYMAPS_API_KEY", "").strip()
    wt_user = os.environ.get("WATTTIME_USERNAME", "").strip()
    wt_pass = os.environ.get("WATTTIME_PASSWORD", "").strip()
    return bool(em_key) or (bool(wt_user) and bool(wt_pass))


def _validate_df(df: pd.DataFrame, mode: str) -> None:
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS

    if df.empty:
        print(f"[WARN] No data returned for mode={mode}. CAISO may have no data for this window.")
        return

    assert list(df.columns) == PRICE_COLUMNS, f"Schema mismatch: {list(df.columns)}"
    assert (df["region"] == "us-west").all(), "region must be us-west"
    assert (df["currency"] == "USD").all(), "currency must be USD"
    assert pd.api.types.is_numeric_dtype(df["price_per_mwh"]), "price_per_mwh must be numeric"
    assert not df["price_per_mwh"].isna().any(), "no NaN prices allowed"

    # Timestamps must be UTC-aware
    for ts in df["timestamp"]:
        assert ts.tzinfo is not None, f"Naive timestamp: {ts}"

    # No demand/load fields
    forbidden = {"demand", "load", "generation", "consumption"}
    col_lower = {c.lower() for c in df.columns}
    overlap = forbidden & col_lower
    assert not overlap, f"Forbidden fields in output: {overlap}"

    if mode == "day_ahead":
        assert (df["source"] == "caiso_oasis_dam").all()
        assert (df["source_granularity"] == "hourly").all()
    elif mode == "real_time":
        assert (df["source"] == "caiso_oasis_rtm").all()
        assert (df["source_granularity"] == "5min").all()

    print(f"[OK] Validation passed: {len(df)} rows, "
          f"price range [{df['price_per_mwh'].min():.2f}, {df['price_per_mwh'].max():.2f}] USD/MWh")


def run_day_ahead(start_str: str, end_str: str) -> None:
    """Fetch and validate CAISO day-ahead LMP for the given date range.

    Command:
        python scripts/caiso_pilot_validation.py --mode day_ahead \\
            --start YYYY-MM-DD --end YYYY-MM-DD
    """
    from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider

    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    print(f"\n[CAISO DAM] Fetching day-ahead LMP for us-west, {start_str} → {end_str}")
    print(f"  Endpoint: https://oasis.caiso.com/oasisapi/SingleZip"
          f"  queryname=PRC_LMP, market_run_id=DAM, node=TH_NP15_GEN-APND, resultformat=6")

    provider = CAISOPriceProvider()
    df = provider.fetch_prices("us-west", start, end)

    _validate_df(df, "day_ahead")

    if not df.empty:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"day_ahead_{start_str}_{end_str}.csv"
        df.to_csv(out, index=False)
        print(f"[OK] Saved to {out}")

    if not _carbon_available():
        print("\n[INFO] Carbon data unavailable — running in cost_only mode.")
        print("       Set ELECTRICITYMAPS_API_KEY or WATTTIME_USERNAME/PASSWORD "
              "to enable carbon-aware optimization.")


def run_real_time(hours: int) -> None:
    """Fetch and validate recent CAISO real-time interval LMP.

    Command:
        python scripts/caiso_pilot_validation.py --mode real_time --hours 1
    """
    from aurelius.ingestion.grid_apis.caiso import CAISORealtimePriceProvider

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)

    print(f"\n[CAISO RTM] Fetching real-time LMP for us-west, last {hours}h "
          f"({start.isoformat()} → {end.isoformat()})")
    print(f"  Endpoint: https://oasis.caiso.com/oasisapi/SingleZip"
          f"  queryname=PRC_INTVL_LMP, market_run_id=RTM, node=TH_NP15_GEN-APND, resultformat=6")

    provider = CAISORealtimePriceProvider()
    df = provider.fetch_prices("us-west", start, end)

    if df.empty:
        print("[WARN] CAISO returned no real-time data for the requested window.")
        print("       Real-time data may have a publication lag of a few minutes.")
        print("       Try using --hours 2 or a specific historical window.")
    else:
        _validate_df(df, "real_time")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_tag = start.strftime("%Y%m%dT%H%M")
        out = REPORTS_DIR / f"real_time_{ts_tag}_{hours}h.csv"
        df.to_csv(out, index=False)
        print(f"[OK] Saved to {out}")

    if not _carbon_available():
        print("\n[INFO] Carbon data unavailable — running in cost_only mode.")
        print("       Set ELECTRICITYMAPS_API_KEY or WATTTIME_USERNAME/PASSWORD "
              "to enable carbon-aware optimization.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAISO OASIS pilot validation — fetch and validate real LMP data."
    )
    parser.add_argument(
        "--mode",
        choices=["day_ahead", "real_time"],
        required=True,
        help="'day_ahead' for PRC_LMP/DAM backtest; 'real_time' for PRC_INTVL_LMP/RTM snapshot.",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date YYYY-MM-DD (day_ahead mode). Default: 2024-01-01",
    )
    parser.add_argument(
        "--end",
        default="2024-02-01",
        help="End date YYYY-MM-DD exclusive (day_ahead mode). Default: 2024-02-01",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Number of hours to look back (real_time mode). Default: 1",
    )
    args = parser.parse_args()

    if args.mode == "day_ahead":
        run_day_ahead(args.start, args.end)
    else:
        run_real_time(args.hours)


if __name__ == "__main__":
    main()
