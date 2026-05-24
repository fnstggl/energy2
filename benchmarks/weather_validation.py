#!/usr/bin/env python3
"""Production-readiness validation for weather-aware optimization.

Validates the caveats from the weather PR with runnable evidence:
  --mode rt_vs_dam   : weather savings under DAM scoring vs RT-settlement scoring
  --mode gating      : weather enabled none/full/PJM-only/winter-only/PJM+winter
  --mode sensitivity : robustness to job count / duration / deadline / migration

All modes bootstrap a 95% CI on the savings delta (weather ON - OFF) over
independent job-mix seeds and run the REAL BacktestEngine + optimizer.

RT scoring: the optimizer PLANS against day-ahead prices (price_df) + day-ahead
forecast weather, and is SCORED against real-time settlement prices (settle_df).
This is the realistic deployment setup and the true test of weather alpha under
DA->RT basis risk.

Deterministic given seeds. Emits a JSON artifact per mode.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from aurelius.backtesting.engine import BacktestEngine
from aurelius.ingestion.job_logs import JobLogIngester
from aurelius.models import OptimizationConfig
from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig

REPO = Path(__file__).resolve().parent.parent
DAM = REPO / "data/combined_2025_2026/3region_dam.csv"
RT = REPO / "data/combined_2025_2026/3region_rt.csv"
W_FC1 = REPO / "data/weather_openmeteo/forecast_day1.csv"
REGIONS = ["us-west", "us-east", "us-south"]
PRIMARY = "current_price_only"
START, END = "2025-09-01", "2026-03-05"
WINTER_MONTHS = (12, 1, 2)
EMPTY_CARBON = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])


def _load_prices(path):
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d[d.region.isin(REGIONS)]


def _load_weather():
    w = pd.read_csv(W_FC1, parse_dates=["timestamp"])
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
    return w


def _gate_weather(w, gating):
    """Return the weather subset for a gating rule; None => weather off."""
    if gating == "none":
        return None
    if gating == "full":
        return w
    if gating == "pjm":
        return w[w.region == "us-east"]
    if gating == "winter":
        return w[w.timestamp.dt.month.isin(WINTER_MONTHS)]
    if gating == "pjm_winter":
        return w[(w.region == "us-east") & (w.timestamp.dt.month.isin(WINTER_MONTHS))]
    raise ValueError(gating)


def _cfg(use_weather):
    return PriceModelConfig(seed=42, n_estimators=200, learning_rate=0.05,
                            include_volatility_features=True, num_leaves=63,
                            include_weather_features=use_weather, include_rank_features=False)


def _gen_jobs(seed, num_jobs, workload, duration_div=0.7, max_delay=None):
    start_ts = pd.Timestamp(START, tz="UTC"); end_ts = pd.Timestamp(END, tz="UTC")
    backtest_hours = int((end_ts - start_ts).total_seconds() / 3600)
    duration_hours = int(backtest_hours / duration_div) + 24
    jobs = JobLogIngester().generate_synthetic(
        start_time=start_ts.to_pydatetime(), duration_hours=duration_hours,
        num_jobs=num_jobs, regions=REGIONS, seed=seed,
        workload_mix="realistic", workload_filter=workload)
    if max_delay is not None:
        for j in jobs:
            if getattr(j, "max_delay_hours", None) is not None:
                j.max_delay_hours = max_delay
    return jobs


def _savings(jobs, da_df, weather_subset, settle_df, no_migrate=False):
    """Run engine; return savings% vs current_price_only. settle_df=None => DAM scoring."""
    use_weather = weather_subset is not None and len(weather_subset) > 0
    if no_migrate:
        for j in jobs:
            j.migration_cost_hours = None  # disable migration
    engine = BacktestEngine(
        method="greedy_migrate", train_days=90, eval_days=14,
        config=OptimizationConfig(),
        price_forecaster_cls=PriceQuantileForecaster,
        price_forecaster_config=_cfg(use_weather),
        context_hours=336,
        weather_df=weather_subset if use_weather else None)
    rounds = engine.run(jobs, da_df, EMPTY_CARBON,
                        start=pd.Timestamp(START, tz="UTC"),
                        end=pd.Timestamp(END, tz="UTC"),
                        settle_price_df=settle_df)
    if not rounds:
        return None, None
    opt = [r.optimizer_metrics.total_energy_cost_usd for r in rounds if r.optimizer_metrics]
    bl = [r.baseline_metrics[PRIMARY].total_energy_cost_usd for r in rounds
          if PRIMARY in r.baseline_metrics]
    if not opt or not bl:
        return None, None
    mo, mb = float(np.mean(opt)), float(np.mean(bl))
    sav = (mb - mo) / mb * 100 if mb > 0 else 0.0
    # per-fold savings for concentration analysis
    per_fold = []
    for r in rounds:
        if r.optimizer_metrics and PRIMARY in r.baseline_metrics:
            b = r.baseline_metrics[PRIMARY].total_energy_cost_usd
            o = r.optimizer_metrics.total_energy_cost_usd
            per_fold.append(b - o)
    return sav, per_fold


def _ci(deltas, n=10000, seed=0):
    if len(deltas) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.asarray(deltas, float)
    b = [rng.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _verdict(lo, hi):
    return "SIGNIFICANT (+)" if lo > 0 else ("SIGNIFICANT (-)" if hi < 0 else "NOT SIGNIFICANT")


def mode_rt_vs_dam(args, da, rt, w):
    print("=== RT vs DAM: does weather alpha survive DA->RT basis? ===\n")
    out = {}
    for scoring, settle in [("DAM", None), ("RT", rt)]:
        d_off, d_on = [], []
        for s in range(args.seeds):
            jobs = _gen_jobs(s, args.num_jobs, "training")
            off, _ = _savings(jobs, da, None, settle)
            on, _ = _savings(jobs, da, w, settle)
            if off is None or on is None:
                continue
            d_off.append(off); d_on.append(on)
            print(f"  [{scoring}] seed {s}: OFF={off:6.2f}% ON={on:6.2f}% d={on-off:+6.2f}pp")
        deltas = np.array(d_on) - np.array(d_off)
        m, lo, hi = _ci(deltas)
        out[scoring] = {"mean_off": float(np.mean(d_off)), "mean_on": float(np.mean(d_on)),
                        "mean_delta_pp": m, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(lo, hi),
                        "per_seed_delta": deltas.tolist()}
        print(f"  [{scoring}] mean OFF={np.mean(d_off):.2f}% ON={np.mean(d_on):.2f}% "
              f"delta={m:+.2f}pp 95%CI[{lo:+.2f},{hi:+.2f}] {_verdict(lo,hi)}\n")
    return out


def mode_gating(args, da, rt, w):
    print("=== Gating (RT-scored): safest production rule ===\n")
    configs = ["full", "pjm", "winter", "pjm_winter"]
    out = {}
    # OFF baseline per seed (shared)
    off_by_seed, foldconc = {}, {}
    for s in range(args.seeds):
        jobs = _gen_jobs(s, args.num_jobs, "training")
        off, _ = _savings(jobs, da, None, rt)
        off_by_seed[s] = off
    for cfg in configs:
        wsub = _gate_weather(w, cfg)
        d, per_fold_last = [], None
        for s in range(args.seeds):
            jobs = _gen_jobs(s, args.num_jobs, "training")
            on, pf = _savings(jobs, da, wsub, rt)
            if on is None or off_by_seed[s] is None:
                continue
            d.append(on - off_by_seed[s])
            if s == 0:
                per_fold_last = pf
        d = np.array(d)
        m, lo, hi = _ci(d)
        # concentration: top fold share of total positive extra savings (seed 0 proxy)
        out[cfg] = {"mean_delta_pp": m, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(lo, hi),
                    "per_seed_delta": d.tolist()}
        print(f"  {cfg:11s} delta={m:+6.2f}pp 95%CI[{lo:+.2f},{hi:+.2f}] {_verdict(lo,hi)}")
    out["_off_mean"] = float(np.mean([v for v in off_by_seed.values() if v is not None]))
    return out


def mode_sensitivity(args, da, rt, w):
    print("=== Sensitivity (RT-scored, weather=full): robustness of the delta ===\n")
    out = {}
    base = dict(num_jobs=50, workload="training", duration_div=0.7, max_delay=None, no_migrate=False)
    variants = {
        "base": {},
        "jobs_25": {"num_jobs": 25},
        "jobs_100": {"num_jobs": 100},
        "short_dur": {"duration_div": 1.4},
        "tight_deadline": {"max_delay": 12},
        "loose_deadline": {"max_delay": 240},
        "no_migration": {"no_migrate": True},
        "mixed_workload": {"workload": None},
    }
    for name, ov in variants.items():
        p = {**base, **ov}
        d = []
        for s in range(args.seeds):
            jobs = _gen_jobs(s, p["num_jobs"], p["workload"], p["duration_div"], p["max_delay"])
            off, _ = _savings(jobs, da, None, rt, no_migrate=p["no_migrate"])
            jobs2 = _gen_jobs(s, p["num_jobs"], p["workload"], p["duration_div"], p["max_delay"])
            on, _ = _savings(jobs2, da, w, rt, no_migrate=p["no_migrate"])
            if off is None or on is None:
                continue
            d.append(on - off)
        d = np.array(d)
        m, lo, hi = _ci(d)
        out[name] = {"mean_delta_pp": m, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(lo, hi)}
        print(f"  {name:16s} delta={m:+6.2f}pp 95%CI[{lo:+.2f},{hi:+.2f}] {_verdict(lo,hi)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["rt_vs_dam", "gating", "sensitivity"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--num-jobs", type=int, default=50)
    args = ap.parse_args()

    da, rt, w = _load_prices(DAM), _load_prices(RT), _load_weather()
    print(f"Window {START}..{END} | seeds={args.seeds} | mode={args.mode}\n")

    fn = {"rt_vs_dam": mode_rt_vs_dam, "gating": mode_gating, "sensitivity": mode_sensitivity}[args.mode]
    result = fn(args, da, rt, w)

    outp = REPO / f"benchmarks/results/weather_validation_{args.mode}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nArtifact: {outp}")


if __name__ == "__main__":
    main()
