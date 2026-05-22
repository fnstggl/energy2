#!/usr/bin/env python3
"""
Aurelius Benchmark Regression Checker
======================================

Compares two benchmark JSON outputs and fails if savings vs current_price_only
regressed by more than the threshold.

Usage:
    python benchmarks/compare_against_previous.py \\
        --baseline benchmarks/results/baseline_benchmark.json \\
        --current  benchmarks/results/benchmark_<timestamp>.json \\
        [--threshold 2.0]

Exit codes:
    0   no regressions
    1   regression(s) detected
    2   input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRIMARY_BASELINE = "current_price_only"
DEFAULT_THRESHOLD = 2.0  # pct regression allowed


def load_results(path: str) -> list[dict]:
    """Load results list from a benchmark JSON file."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(2)
    with open(p) as f:
        data = json.load(f)
    # Accept both wrapped {"results": [...]} and bare [...]
    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        print(f"ERROR: unexpected JSON structure in {p}", file=sys.stderr)
        sys.exit(2)
    return [r for r in results if "error" not in r]


def compare(
    baseline_results: list[dict],
    current_results: list[dict],
    threshold_pct: float,
) -> tuple[list[str], list[str]]:
    """Return (regressions, improvements)."""
    prev_index = {
        (r.get("workload_type"), r.get("region_combo")): r
        for r in baseline_results
    }
    regressions = []
    improvements = []
    for r in current_results:
        key = (r.get("workload_type"), r.get("region_combo"))
        prev = prev_index.get(key)
        if prev is None:
            continue
        cur_pct = r.get("primary_savings_pct")
        prev_pct = prev.get("primary_savings_pct")
        if cur_pct is None or prev_pct is None:
            continue
        delta = cur_pct - prev_pct
        if delta < -threshold_pct:
            regressions.append(
                f"REGRESSION  {key[0]:<28}@{key[1]:<22}  "
                f"{prev_pct:+.1f}% → {cur_pct:+.1f}%  (Δ {delta:.1f}%, threshold -{threshold_pct:.1f}%)"
            )
        elif delta > 0.5:
            improvements.append(
                f"IMPROVEMENT {key[0]:<28}@{key[1]:<22}  "
                f"{prev_pct:+.1f}% → {cur_pct:+.1f}%  (Δ +{delta:.1f}%)"
            )
    return regressions, improvements


def missing_baselines(results: list[dict]) -> list[str]:
    """Return list of cells missing current_price_only in savings."""
    missing = []
    for r in results:
        savings = r.get("savings", {})
        if PRIMARY_BASELINE not in savings:
            key = f"{r.get('workload_type')}@{r.get('region_combo')}"
            missing.append(f"MISSING {PRIMARY_BASELINE}: {key}")
    return missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="Path to baseline benchmark JSON")
    p.add_argument("--current", required=True, help="Path to current benchmark JSON")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Regression threshold in %% (default: {DEFAULT_THRESHOLD})")
    args = p.parse_args()

    baseline_results = load_results(args.baseline)
    current_results = load_results(args.current)

    if not baseline_results:
        print("ERROR: baseline results list is empty", file=sys.stderr)
        return 2
    if not current_results:
        print("ERROR: current results list is empty", file=sys.stderr)
        return 2

    # Check that current_price_only is present in all cells
    missing = missing_baselines(current_results)
    regressions, improvements = compare(baseline_results, current_results, args.threshold)

    print(f"Comparing {len(current_results)} current cells against {len(baseline_results)} baseline cells")
    print(f"Threshold: -{args.threshold:.1f}%  Primary baseline: {PRIMARY_BASELINE}\n")

    if improvements:
        print(f"Improvements ({len(improvements)}):")
        for imp in improvements:
            print(f"  {imp}")

    if missing:
        print(f"\nMissing {PRIMARY_BASELINE} data ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    if regressions:
        print(f"\n✗ REGRESSIONS ({len(regressions)}) — BENCHMARK FAILED:")
        for reg in regressions:
            print(f"  {reg}")
        return 1

    print(f"✓ No regressions detected (threshold -{args.threshold:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
