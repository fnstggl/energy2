"""Merge multiple canonical-schema price CSVs into one for multi-region backtest.

Usage:
    python scripts/merge_price_csvs.py \\
        --inputs reports/pilot_validation/caiso/day_ahead_2024-01-01_2024-04-08.csv \\
                 reports/pilot_validation/pjm/day_ahead_2024-01-01_2024-04-08.csv \\
        --output reports/pilot_validation/combined_q1_2024.csv

All input CSVs must share the canonical price schema produced by the
pilot-validation scripts (timestamp, region, price_per_mwh, currency,
source, source_granularity, fetched_at).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge canonical price CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(p, parse_dates=["timestamp"])
        missing = set(PRICE_COLUMNS) - set(df.columns)
        if missing:
            print(f"ERROR: {p} missing columns {missing}", file=sys.stderr)
            sys.exit(1)
        df = df[PRICE_COLUMNS]
        print(f"  {p.name}: {len(df)} rows, regions={sorted(df['region'].unique())}, "
              f"range=[{df['timestamp'].min()}, {df['timestamp'].max()}]")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "region"], keep="last")
    combined = combined.sort_values(["region", "timestamp"]).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\n[OK] Merged {len(args.inputs)} files → {len(combined)} rows, "
          f"{len(combined['region'].unique())} regions → {out}")


if __name__ == "__main__":
    main()
