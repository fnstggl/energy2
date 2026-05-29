#!/usr/bin/env python3
"""Run the canonical CAISO/PJM/ERCOT 1000-job energy backtest.

Usage:
    python scripts/run_canonical_backtests.py                 # print summary
    python scripts/run_canonical_backtests.py --json          # print golden JSON
    python scripts/run_canonical_backtests.py --write-golden  # (re)write golden snapshot

The golden snapshot lives at:
    aurelius/benchmarks/golden/canonical_energy_backtest.json

Regenerating it is a DELIBERATE act: it changes the frozen benchmark and must be
explained in the PR body (see docs/BACKTESTS.md). The pytest suite
(tests/test_canonical_energy_backtest.py) fails if the live run diverges from
the committed golden snapshot.

Simulator/benchmark result only — not a production-savings claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.benchmarks.canonical_backtests import (  # noqa: E402
    GOLDEN_PATH,
    run_canonical_backtest,
)


def _print_human(golden: dict) -> None:
    print("=" * 78)
    print("CANONICAL CAISO/PJM/ERCOT 1000-JOB ENERGY BACKTEST")
    print("=" * 78)
    print(f"seed={golden['seed']}  jobs={golden['job_count']}  "
          f"regions={golden['regions']}  method={golden['method']}")
    print(f"window: {golden['window_start']} .. {golden['window_end']}")
    print(f"primary KPI: {golden['primary_kpi']} (unit: {golden['goodput_unit']})")
    print("-" * 78)
    header = (f"{'policy':<40}{'energy$':>11}{'infra$':>12}"
             f"{'goodput/$':>11}{'misses':>8}{'migs':>7}")
    print(header)
    for name, m in golden["policies"].items():
        print(f"{name:<40}{m['realized_energy_cost_usd']:>11,.0f}"
              f"{m['total_infra_cost_usd']:>12,.0f}"
              f"{m['sla_safe_goodput_per_infra_dollar']:>11.5f}"
              f"{m['deadline_misses']:>8d}{m['migrations']:>7d}")
    print("-" * 78)
    print("standalone vs constraint-aware-wrapped delta:")
    for k, v in golden["standalone_vs_wrapped_delta"].items():
        print(f"  {k}: {v}")
    print("-" * 78)
    print(golden["disclaimer"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the golden JSON")
    ap.add_argument("--write-golden", action="store_true",
                    help="(re)write the committed golden snapshot")
    args = ap.parse_args()

    summary = run_canonical_backtest()
    golden = summary.golden_dict()

    if args.write_golden:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(golden, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote golden snapshot to {GOLDEN_PATH}")
        return 0

    if args.json:
        print(json.dumps(golden, indent=2, sort_keys=True))
    else:
        _print_human(golden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
