#!/usr/bin/env python3
"""Savings-level weather-alpha test with bootstrap CIs over job mixes.

The prior audit showed the single-run savings metric flips sign with job count,
i.e. it is noise-dominated. This harness drives the REAL BacktestEngine +
optimizer over the full multi-season window with N independent job-generation
seeds, weather OFF vs weather ON (REALISTIC day-ahead forecast, not observed
actuals), and bootstraps a 95% CI on the per-seed savings delta.

Verdict rule: weather creates real savings alpha only if the 95% CI on the
mean savings delta (ON - OFF) excludes 0.

Deterministic given --seeds. Emits a JSON artifact.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from aurelius.backtesting.engine import BacktestEngine
from aurelius.backtesting.baselines import ALL_BASELINES
from aurelius.ingestion.job_logs import JobLogIngester
from aurelius.models import OptimizationConfig
from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig

REPO = Path(__file__).resolve().parent.parent
PRICE_FILE = REPO / "data/combined_2025_2026/3region_dam.csv"
W_FC1 = REPO / "data/weather_openmeteo/forecast_day1.csv"   # honest: day-ahead forecast
REGIONS = ["us-west", "us-east", "us-south"]
PRIMARY_BASELINE = "current_price_only"
START = "2025-09-01"   # 90-day training context available before this
END = "2026-03-05"
WORKLOAD = "training"   # most price-sensitive flexible workload


def _cfg(use_weather):
    return PriceModelConfig(seed=42, n_estimators=200, learning_rate=0.05,
                            include_volatility_features=True, num_leaves=63,
                            include_weather_features=use_weather, include_rank_features=False)


def savings_for(jobs, price_df, weather_df, use_weather):
    engine = BacktestEngine(
        method="greedy_migrate", train_days=90, eval_days=14,
        config=OptimizationConfig(),
        price_forecaster_cls=PriceQuantileForecaster,
        price_forecaster_config=_cfg(use_weather),
        context_hours=336,
        weather_df=weather_df if use_weather else None,
    )
    empty_carbon = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])
    rounds = engine.run(jobs, price_df, empty_carbon,
                        start=pd.Timestamp(START, tz="UTC"),
                        end=pd.Timestamp(END, tz="UTC"))
    if not rounds:
        return None
    opt = [r.optimizer_metrics.total_energy_cost_usd for r in rounds if r.optimizer_metrics]
    mean_opt = float(np.mean(opt)) if opt else 0.0
    bl_costs = [r.baseline_metrics[PRIMARY_BASELINE].total_energy_cost_usd
                for r in rounds if PRIMARY_BASELINE in r.baseline_metrics]
    mean_bl = float(np.mean(bl_costs)) if bl_costs else 0.0
    return (mean_bl - mean_opt) / mean_bl * 100 if mean_bl > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12, help="number of job-mix seeds")
    ap.add_argument("--num-jobs", type=int, default=50)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/weather_savings_eval.json"))
    args = ap.parse_args()

    price_df = pd.read_csv(PRICE_FILE, parse_dates=["timestamp"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
    price_df = price_df[price_df.region.isin(REGIONS)]
    weather_df = pd.read_csv(W_FC1, parse_dates=["timestamp"])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)

    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")
    backtest_hours = int((end_ts - start_ts).total_seconds() / 3600)
    duration_hours = int(backtest_hours / 0.7) + 24
    ingester = JobLogIngester()

    print(f"Savings bootstrap: {WORKLOAD} @ 3-region | {args.seeds} job seeds x {args.num_jobs} jobs")
    print(f"Window {START}..{END} | weather OFF vs ON(day-ahead forecast)\n")

    deltas, offs, ons = [], [], []
    for seed in range(args.seeds):
        jobs = ingester.generate_synthetic(
            start_time=start_ts.to_pydatetime(), duration_hours=duration_hours,
            num_jobs=args.num_jobs, regions=REGIONS, seed=seed,
            workload_mix="realistic", workload_filter=WORKLOAD)
        off = savings_for(jobs, price_df, weather_df, False)
        on = savings_for(jobs, price_df, weather_df, True)
        if off is None or on is None:
            continue
        deltas.append(on - off); offs.append(off); ons.append(on)
        print(f"  seed {seed:2d}: OFF={off:6.2f}%  ON={on:6.2f}%  delta={on-off:+6.2f}pp")

    deltas = np.array(deltas)
    rng = np.random.default_rng(0)
    boot = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(10000)]
    mean_d, lo, hi = float(deltas.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    sig = "SIGNIFICANT (+)" if lo > 0 else ("SIGNIFICANT (-)" if hi < 0 else "NOT SIGNIFICANT (within noise)")

    print("\n" + "=" * 70)
    print("SAVINGS DELTA (weather ON - OFF), bootstrap 95% CI over job seeds")
    print("=" * 70)
    print(f"  mean OFF savings = {np.mean(offs):.2f}%  (sd {np.std(offs):.2f})")
    print(f"  mean ON  savings = {np.mean(ons):.2f}%  (sd {np.std(ons):.2f})")
    print(f"  mean delta       = {mean_d:+.3f}pp   95%CI[{lo:+.3f}, {hi:+.3f}]")
    print(f"  VERDICT: {sig}")

    out = {"workload": WORKLOAD, "n_seeds": len(deltas), "num_jobs": args.num_jobs,
           "mean_off": float(np.mean(offs)), "mean_on": float(np.mean(ons)),
           "mean_delta_pp": mean_d, "ci_lo": lo, "ci_hi": hi, "verdict": sig,
           "per_seed_delta": deltas.tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nArtifact: {args.out}")


if __name__ == "__main__":
    main()
