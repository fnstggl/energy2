"""Fetch CAISO + PJM (+ optional ERCOT) day-ahead and real-time prices.

Produces the DA-plan / RT-settle backtest inputs per region:
    data/caiso_us_west_dam.csv      CAISO us-west day-ahead   (hourly)
    data/caiso_us_west_rt.csv       CAISO us-west real-time   (5-min -> hourly mean)
    data/pjm_us_east_dam.csv        PJM   us-east  day-ahead   (hourly)
    data/pjm_us_east_rt.csv         PJM   us-east  real-time   (rt_hrl_lmps hourly)
    data/ercot_us_south_dam.csv     ERCOT us-south day-ahead SPP (hourly)  [if creds]
    data/ercot_us_south_rt.csv      ERCOT us-south real-time SPP (15-min -> hourly mean)
    data/plan_da_caiso_pjm.csv      PLANNING prices: all fetched regions' day-ahead
    data/settle_rt_caiso_pjm.csv    SETTLEMENT prices: all fetched regions' real-time

ERCOT (us-south) is included automatically when ERCOT credentials are present:
ERCOT_API_KEY plus ERCOT_USERNAME/ERCOT_PASSWORD (or ERCOT_ID_TOKEN). Without
them the script fetches CAISO + PJM only. ERCOT SPP is the Houston-hub
settlement-point price (USD/MWh); its 15-minute real-time series is resampled to
an hourly mean, matching the CAISO 5-minute handling, because the backtest
engine operates at hourly resolution.

Backtest usage (RT-exposed customer):
    aurelius backtest --price-provider csv --price-file data/plan_da_caiso_pjm.csv \
        --settlement-price-file data/settle_rt_caiso_pjm.csv \
        --regions us-west,us-east,us-south --forecaster ml_quantile \
        --forecast-horizon-hours 24

Usage:
    PJM_API_KEY=... ERCOT_API_KEY=... ERCOT_USERNAME=... ERCOT_PASSWORD=... \
        python scripts/fetch_caiso_pjm_prices.py \
        --start 2026-01-01 --end 2026-03-15 --out-dir data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make aurelius importable when running the script directly from any working dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

import pandas as pd

from aurelius.ingestion.grid_apis.base import normalize_price_df
from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider, CAISORealtimePriceProvider
from aurelius.ingestion.grid_apis.ercot import ERCOTPriceProvider, ERCOTRealtimePriceProvider
from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider, PJMRealtimePriceProvider


def _ercot_creds_present() -> bool:
    has_key = bool(os.environ.get("ERCOT_API_KEY"))
    has_auth = bool(os.environ.get("ERCOT_ID_TOKEN")) or (
        bool(os.environ.get("ERCOT_USERNAME")) and bool(os.environ.get("ERCOT_PASSWORD"))
    )
    return has_key and has_auth


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _to_hourly_mean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Resample a sub-hourly price series to an hourly mean per region."""
    if df.empty:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("h")
    agg = (
        d.groupby(["region", "timestamp"], as_index=False)["price_per_mwh"]
        .mean()
    )
    return normalize_price_df(agg, source=source, currency="USD", granularity="hourly")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-01", help="ISO start date (UTC)")
    ap.add_argument("--end", default="2026-03-15", help="ISO end date (UTC)")
    ap.add_argument("--out-dir", default="data", help="Output directory")
    args = ap.parse_args()

    start, end = _parse_utc(args.start), _parse_utc(args.end)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    log("CAISO us-west day-ahead (hourly)...")
    caiso_da = CAISOPriceProvider().fetch_prices("us-west", start, end)
    caiso_da.to_csv(out / "caiso_us_west_dam.csv", index=False)
    log(f"  rows={len(caiso_da)}")

    log("CAISO us-west real-time (5-min -> hourly mean)...")
    caiso_rt_5min = CAISORealtimePriceProvider().fetch_prices("us-west", start, end)
    caiso_rt = _to_hourly_mean(caiso_rt_5min, source="caiso_oasis_rtm_hourly")
    caiso_rt.to_csv(out / "caiso_us_west_rt.csv", index=False)
    log(f"  rows={len(caiso_rt)} (from {len(caiso_rt_5min)} 5-min intervals)")

    log("PJM us-east day-ahead (hourly)...")
    pjm_da = PJMPriceProvider().fetch_prices("us-east", start, end)
    pjm_da.to_csv(out / "pjm_us_east_dam.csv", index=False)
    log(f"  rows={len(pjm_da)}")

    log("PJM us-east real-time (hourly)...")
    pjm_rt = PJMRealtimePriceProvider(hourly=True).fetch_prices("us-east", start, end)
    pjm_rt.to_csv(out / "pjm_us_east_rt.csv", index=False)
    log(f"  rows={len(pjm_rt)}")

    plan_parts = [caiso_da, pjm_da]
    settle_parts = [caiso_rt, pjm_rt]

    if _ercot_creds_present():
        log("ERCOT us-south day-ahead SPP (hourly)...")
        ercot_da = ERCOTPriceProvider().fetch_prices("us-south", start, end)
        ercot_da.to_csv(out / "ercot_us_south_dam.csv", index=False)
        log(f"  rows={len(ercot_da)}")

        log("ERCOT us-south real-time SPP (15-min -> hourly mean)...")
        ercot_rt_15min = ERCOTRealtimePriceProvider().fetch_prices("us-south", start, end)
        ercot_rt = _to_hourly_mean(ercot_rt_15min, source="ercot_rt_spp_hourly")
        ercot_rt.to_csv(out / "ercot_us_south_rt.csv", index=False)
        log(f"  rows={len(ercot_rt)} (from {len(ercot_rt_15min)} 15-min intervals)")

        if not ercot_da.empty:
            plan_parts.append(ercot_da)
        if not ercot_rt.empty:
            settle_parts.append(ercot_rt)
    else:
        log("ERCOT skipped (no ERCOT_API_KEY + ERCOT_USERNAME/PASSWORD or ERCOT_ID_TOKEN).")

    plan = pd.concat(plan_parts, ignore_index=True)
    settle = pd.concat(settle_parts, ignore_index=True)
    plan.to_csv(out / "plan_da_caiso_pjm.csv", index=False)
    settle.to_csv(out / "settle_rt_caiso_pjm.csv", index=False)

    print("\n--- summary (mean $/MWh) ---", flush=True)
    for label, df in (("PLAN (day-ahead)", plan), ("SETTLE (real-time)", settle)):
        print(f"{label}:", flush=True)
        for reg, g in df.groupby("region"):
            print(f"  {reg}: {len(g)} rows  mean=${g['price_per_mwh'].mean():.2f}  "
                  f"max=${g['price_per_mwh'].max():.2f}", flush=True)


if __name__ == "__main__":
    main()
