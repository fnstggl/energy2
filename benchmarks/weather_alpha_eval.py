#!/usr/bin/env python3
"""Statistically rigorous weather-alpha evaluation for Aurelius.

Answers ONE question honestly: does weather create real, stable economic alpha
under REALISTIC (forecast, not perfect-foresight) deployment conditions?

It separates three things the prior benchmark conflated:
  1. forecast improvement (price-model MAE/RMSE/pinball),
  2. realistic vs leaky weather (forecast-day-ahead vs observed-actuals),
  3. benchmark noise (bootstrap CIs across folds + seeds).

Weather modes compared, all else identical:
  - none      : price-only baseline (v2.0 forecaster).
  - observed  : eval-window weather = OBSERVED actuals (perfect foresight —
                the leaky mode the engine currently uses).
  - forecast  : eval-window weather = DAY-AHEAD forecast (Open-Meteo Previous
                Runs previous_day1 — what a deployed system actually had).

Statistics:
  - Walk-forward folds spanning summer + winter (multi-season).
  - Per (fold x region) MAE/RMSE/pinball collected for every mode.
  - Bootstrap 95% CI on the mean delta (mode - none) across folds.
  - A claim of "weather helps region R" requires the CI upper bound < 0.

Deterministic: fixed forecaster seed; bootstrap uses a fixed RNG seed.

Usage:
  python benchmarks/weather_alpha_eval.py            # full multi-season run
  python benchmarks/weather_alpha_eval.py --quick    # 4 folds, fast smoke
"""
from __future__ import annotations
import argparse, json, sys, warnings
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from aurelius.models import EnergyPrice
from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig

REPO = Path(__file__).resolve().parent.parent
PRICE_FILE = REPO / "data/combined_2025_2026/3region_dam.csv"
W_OBS = REPO / "data/weather_openmeteo/observed_era5.csv"
W_FC1 = REPO / "data/weather_openmeteo/forecast_day1.csv"
REGIONS = ["us-west", "us-east", "us-south"]
CONTEXT_HOURS = 336
P90 = 0.90


def _load():
    p = pd.read_csv(PRICE_FILE, parse_dates=["timestamp"])
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p[p.region.isin(REGIONS)].sort_values(["region", "timestamp"]).reset_index(drop=True)
    wobs = pd.read_csv(W_OBS, parse_dates=["timestamp"]); wobs["timestamp"] = pd.to_datetime(wobs["timestamp"], utc=True)
    wfc = pd.read_csv(W_FC1, parse_dates=["timestamp"]); wfc["timestamp"] = pd.to_datetime(wfc["timestamp"], utc=True)
    return p, wobs, wfc


def _recs(df):
    return [EnergyPrice(timestamp=r.timestamp.to_pydatetime(), region=r.region,
                        price_per_mwh=float(r.price_per_mwh)) for r in df.itertuples()]


def _pinball(actual, pred, q):
    d = actual - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _season(ts):
    m = ts.month
    return "summer" if m in (6, 7, 8) else ("winter" if m in (12, 1, 2) else "shoulder")


def run_fold(p, wobs, wfc, train_start, eval_start, eval_end, seed):
    """Fit each mode once; return per-region metrics for none/observed/forecast."""
    train = p[(p.timestamp >= train_start) & (p.timestamp < eval_start)]
    ev = p[(p.timestamp >= eval_start) & (p.timestamp < eval_end)]
    if len(train) == 0 or len(ev) == 0:
        return None
    train_recs = _recs(train)
    train_w = wobs[wobs.timestamp < eval_start].copy()  # training always uses observed history

    def fit(use_weather):
        cfg = PriceModelConfig(seed=seed, n_estimators=200, learning_rate=0.05,
                               include_volatility_features=True, num_leaves=63,
                               include_weather_features=use_weather, include_rank_features=False)
        fc = PriceQuantileForecaster(config=cfg)
        fc.fit(train_recs, weather_df=(train_w if use_weather else None))
        return fc

    fc_none = fit(False)
    fc_w = fit(True)  # same fitted model serves observed/forecast predict (only predict weather differs)

    out = {}
    for region in REGIONS:
        evr = ev[ev.region == region].sort_values("timestamp")
        if len(evr) == 0:
            continue
        ts = [t.to_pydatetime() for t in evr.timestamp]
        actual = evr.price_per_mwh.values
        ctx = _recs(train[train.region == region].sort_values("timestamp").tail(CONTEXT_HOURS))

        def metrics(fc, pw):
            f = fc.predict(region, ts, recent_prices=ctx, weather_df=pw)
            p50 = np.array([x.p50 for x in f]); p90 = np.array([x.p90 for x in f])
            return {"mae": float(np.mean(np.abs(p50 - actual))),
                    "rmse": float(np.sqrt(np.mean((p50 - actual) ** 2))),
                    "pinball90": _pinball(actual, p90, P90)}

        wo = wobs[wobs.region == region]
        wf = wfc[wfc.region == region]
        out[region] = {
            "none": metrics(fc_none, None),
            "observed": metrics(fc_w, wo),
            "forecast": metrics(fc_w, wf),
        }
    return out


def bootstrap_ci(deltas, n=5000, seed=0):
    """95% CI on the mean of per-fold deltas via fold resampling."""
    if len(deltas) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return (float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "benchmarks/results/weather_alpha_eval.json"))
    args = ap.parse_args()

    p, wobs, wfc = _load()
    pmin, pmax = p.timestamp.min(), p.timestamp.max()

    # Walk-forward folds: 90d train, 14d eval, step 14d, spanning the full range.
    train_days, eval_days, step_days = 90, 14, 14
    folds = []
    cur = pmin + timedelta(days=train_days)
    while cur + timedelta(days=eval_days) <= pmax:
        folds.append((cur - timedelta(days=train_days), cur, cur + timedelta(days=eval_days)))
        cur += timedelta(days=step_days)
    if args.quick:
        folds = folds[:2] + folds[-2:]
    seeds = list(range(42, 42 + args.seeds))

    print(f"Price {pmin.date()}..{pmax.date()} | {len(folds)} folds x {len(seeds)} seeds")
    print(f"Modes: none | observed(leaky perfect-foresight) | forecast(day-ahead, honest)\n")

    # collect per (fold,region,seed) metrics
    records = []
    for fi, (ts_, es, ee) in enumerate(folds):
        for seed in seeds:
            r = run_fold(p, wobs, wfc, ts_, es, ee, seed)
            if r is None:
                continue
            for region, modes in r.items():
                rec = {"fold": fi, "seed": seed, "region": region,
                       "season": _season(es), "eval_start": str(es.date())}
                for mode, mm in modes.items():
                    for k, v in mm.items():
                        rec[f"{mode}_{k}"] = v
                records.append(rec)
        print(f"  fold {fi+1}/{len(folds)} {es.date()}..{ee.date()} [{_season(es)}] done")

    df = pd.DataFrame(records)
    summary = {"n_folds": len(folds), "n_seeds": len(seeds), "results": {}}

    print("\n" + "=" * 90)
    print("FORECAST QUALITY: delta vs price-only baseline (negative = weather helps)")
    print("95% bootstrap CI over folds. Claim valid only if CI upper bound < 0.")
    print("=" * 90)
    for metric in ["mae", "rmse", "pinball90"]:
        print(f"\n--- {metric.upper()} ---")
        for scope in ["overall"] + REGIONS:
            sub = df if scope == "overall" else df[df.region == scope]
            for mode in ["observed", "forecast"]:
                d = (sub[f"{mode}_{metric}"] - sub[f"none_{metric}"]).values
                mean, lo, hi = bootstrap_ci(d)
                sig = "SIGNIFICANT" if hi < 0 else ("hurts" if lo > 0 else "n.s.")
                tag = "leaky " if mode == "observed" else "honest"
                print(f"  {scope:9s} {tag} d{metric}={mean:+7.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {sig}")
                summary["results"][f"{metric}_{scope}_{mode}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "sig": sig}

    # season split for forecast mode (the honest one)
    print("\n" + "=" * 90)
    print("HONEST (forecast) dMAE by season x region")
    print("=" * 90)
    for season in ["summer", "winter", "shoulder"]:
        sub = df[df.season == season]
        if len(sub) == 0:
            continue
        for region in REGIONS:
            s2 = sub[sub.region == region]
            if len(s2) == 0:
                continue
            d = (s2["forecast_mae"] - s2["none_mae"]).values
            mean, lo, hi = bootstrap_ci(d)
            sig = "SIGNIFICANT" if hi < 0 else ("hurts" if lo > 0 else "n.s.")
            print(f"  {season:8s} {region:9s} dMAE={mean:+7.3f} 95%CI[{lo:+.3f},{hi:+.3f}] {sig}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nArtifact: {args.out}")


if __name__ == "__main__":
    main()
