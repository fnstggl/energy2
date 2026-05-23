"""PJM pilot validation — fetch and validate real PJM Data Miner 2 LMP data.

Mirror of scripts/caiso_pilot_validation.py for the PJM us-east region.

Day-ahead backtest:
    python scripts/pjm_pilot_validation.py --start 2024-01-01 --end 2024-04-01

Requires:
    export PJM_API_KEY=<your_data_miner_2_subscription_key>

Outputs are saved to reports/pilot_validation/pjm/.

PJM Data Miner 2 endpoint (used by aurelius/ingestion/grid_apis/pjm.py):
    https://api.pjm.com/api/v1/da_hrl_lmps
    auth header: Ocp-Apim-Subscription-Key

Default node: pnode_id=1 (Western Hub) — the most liquid PJM reference price.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stderr,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "pilot_validation" / "pjm"


def _validate_df(df: pd.DataFrame) -> None:
    from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS

    if df.empty:
        print("[WARN] No data returned. PJM may have no data for this window or key was rejected.")
        return

    assert list(df.columns) == PRICE_COLUMNS, f"Schema mismatch: {list(df.columns)}"
    assert (df["region"] == "us-east").all(), "region must be us-east"
    assert (df["currency"] == "USD").all(), "currency must be USD"
    assert pd.api.types.is_numeric_dtype(df["price_per_mwh"]), "price_per_mwh must be numeric"
    assert not df["price_per_mwh"].isna().any(), "no NaN prices allowed"

    for ts in df["timestamp"]:
        assert ts.tzinfo is not None, f"Naive timestamp: {ts}"

    forbidden = {"demand", "load", "generation", "consumption"}
    overlap = forbidden & {c.lower() for c in df.columns}
    assert not overlap, f"Forbidden fields in output: {overlap}"

    assert (df["source"] == "pjm_da_lmp").all()
    assert (df["source_granularity"] == "hourly").all()

    print(f"[OK] Validation passed: {len(df)} rows, "
          f"price range [{df['price_per_mwh'].min():.2f}, {df['price_per_mwh'].max():.2f}] USD/MWh")


def run_day_ahead(start_str: str, end_str: str) -> None:
    from aurelius.ingestion.grid_apis.base import ProviderConfigError
    from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider

    if not os.environ.get("PJM_API_KEY", "").strip():
        print("ERROR: PJM_API_KEY env var is not set.", file=sys.stderr)
        print("Get your key at https://dataminer2.pjm.com/profile and export:", file=sys.stderr)
        print("  export PJM_API_KEY=<your_key>", file=sys.stderr)
        sys.exit(1)

    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    print(f"\n[PJM DAM] Fetching day-ahead LMP for us-east, {start_str} → {end_str}")
    print("  Endpoint: https://api.pjm.com/api/v1/da_hrl_lmps  pnode_id=1 (Western Hub)")

    provider = PJMPriceProvider()
    try:
        df = provider.fetch_prices("us-east", start, end)
    except ProviderConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    _validate_df(df)

    if not df.empty:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"day_ahead_{start_str}_{end_str}.csv"
        df.to_csv(out, index=False)
        print(f"[OK] Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PJM Data Miner 2 pilot validation — fetch and validate real LMP data."
    )
    parser.add_argument(
        "--start", default="2026-01-01",
        help="Start date YYYY-MM-DD. Default: 2026-01-01 "
             "(PJM moves data older than ~12 months to an archive feed that does "
             "not accept pnode_id/fields filters — keep --start within ~12 months)",
    )
    parser.add_argument(
        "--end", default="2026-04-08",
        help="End date YYYY-MM-DD exclusive. Default: 2026-04-08 (Q1 + 7-day buffer)",
    )
    args = parser.parse_args()
    run_day_ahead(args.start, args.end)


if __name__ == "__main__":
    main()
