#!/usr/bin/env python3
"""
Aurelius Standardized Benchmark Runner
=======================================

Runs the full benchmark suite across all workload types and region combinations,
comparing the optimizer against all baselines (primary: current_price_only).

Usage:
    python benchmarks/run_benchmark.py [--quick] [--output-dir benchmarks/results]

Options:
    --quick           Run a reduced suite (fewer folds, smaller job count)
                      for CI smoke-testing. Results are NOT valid for claims.
    --output-dir DIR  Output directory for JSON results (default: benchmarks/results)
    --workload WTYPE  Run a single workload type only
    --region-combo    Run a specific region combination (caiso_pjm | us-west | us-east)
    --oracle          Also run oracle diagnostics (ceiling analysis)
    --compare-baseline FILE  Compare against a previous benchmark JSON; fail if regression

Outputs:
    benchmarks/results/benchmark_<timestamp>.json   Full results
    benchmarks/results/summary_<timestamp>.txt      Human-readable summary
    stdout                                          Table + regression check

Exit codes:
    0  success (all checks passed)
    1  regression detected or constraint violation
    2  input/config error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add repo root to sys.path so `aurelius` is importable without install
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

import pandas as pd

from aurelius.backtesting.engine import BacktestEngine
from aurelius.backtesting.baselines import ALL_BASELINES
from aurelius.ingestion.job_logs import JobLogIngester
from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
from aurelius.models import OptimizationConfig

# Optional DB persistence — no-op when DATABASE_URL is not set
try:
    from aurelius.database import TimeSeriesStore as _TSStore
except ImportError:
    _TSStore = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Benchmark suite definitions
# ---------------------------------------------------------------------------

WORKLOAD_TYPES = [
    "training",
    "fine_tuning",
    "llm_batch_inference",
    "data_processing",
    "scheduled_batch",
    "background_maintenance",
    "realtime_inference",
]

# Each region combo is (label, regions_list, da_price_file, rt_price_file_or_None)
REGION_COMBOS = [
    # --- Q1 2026: single-region baselines ---
    {
        "name": "us-west-only",
        "regions": ["us-west"],
        "da_price_file": "data/caiso_us_west_dam.csv",
        "rt_price_file": None,
        # CAISO data: 2026-01-01 → 2026-03-14 (1752 hourly rows)
        "date_start": "2026-01-01",
        "date_end": "2026-03-10",
    },
    {
        "name": "us-east-only",
        "regions": ["us-east"],
        "da_price_file": "data/pjm_us_east_dam.csv",
        "rt_price_file": None,
        # PJM data: 2026-01-01 → 2026-03-15 (1753 hourly rows)
        "date_start": "2026-01-01",
        "date_end": "2026-03-10",
    },
    {
        "name": "us-south-only",
        "regions": ["us-south"],
        "da_price_file": "data/ercot_us_south_dam.csv",
        "rt_price_file": "data/ercot_us_south_rt.csv",
        # ERCOT data: 2026-01-01 → 2026-03-14 (1728 hourly rows)
        "date_start": "2026-01-01",
        "date_end": "2026-03-10",
    },
    # --- Q1 2026: 2-region CAISO+PJM with DA plan + RT settle ---
    {
        "name": "caiso_pjm_da_rt",
        "regions": ["us-west", "us-east"],
        "da_price_file": "data/plan_da_caiso_pjm.csv",
        "rt_price_file": "data/settle_rt_caiso_pjm.csv",
        # Merged DA+RT data: 2026-01-01 → 2026-03-15
        "date_start": "2026-01-01",
        "date_end": "2026-03-10",
    },
    # --- Q1 2026: 3-region CAISO+PJM+ERCOT (anti-correlation test) ---
    {
        "name": "caiso_pjm_ercot_da_rt",
        "regions": ["us-west", "us-east", "us-south"],
        "da_price_file": "data/q12026_3region_dam.csv",
        "rt_price_file": "data/q12026_3region_rt.csv",
        # 3-region merged: 2026-01-01 → 2026-03-14
        "date_start": "2026-01-01",
        "date_end": "2026-03-10",
    },
    # --- Summer 2025: 3-region (seasonal diversification test) ---
    {
        "name": "summer2025_3region",
        "regions": ["us-west", "us-east", "us-south"],
        "da_price_file": "data/summer2025/3region_dam.csv",
        "rt_price_file": "data/summer2025/3region_rt.csv",
        # 3-region summer data: 2025-06-01 → 2025-08-30
        "date_start": "2025-06-01",
        "date_end": "2025-08-25",
    },
]

# Extended region combos — require the combined 2025-2026 dataset.
# Build it first with: python scripts/build_combined_dataset.py
# (needs fall2025 data from: python scripts/fetch_caiso_pjm_prices.py
#  --start 2025-09-01 --end 2026-01-01 --out-dir data/fall2025)
# These combos use 90-day training windows to validate per-region forecaster
# with sufficient per-region data (≥2160 records/region vs 720 with 30-day windows).
EXTENDED_REGION_COMBOS = [
    {
        "name": "combined_2025_2026_3region",
        "regions": ["us-west", "us-east", "us-south"],
        "da_price_file": "data/combined_2025_2026/3region_dam.csv",
        "rt_price_file": "data/combined_2025_2026/3region_rt.csv",
        # Combined: Jun 2025 → Mar 2026 (~270–290 days)
        # Use start=2025-09-01 so 90-day training window starts in full fall data.
        "date_start": "2025-09-01",
        "date_end": "2026-03-10",
        "recommended_train_days": 90,
    },
]

QUICK_REGION_COMBOS = [
    {
        "name": "caiso_pjm_da_rt",
        "regions": ["us-west", "us-east"],
        "da_price_file": "data/plan_da_caiso_pjm.csv",
        "rt_price_file": "data/settle_rt_caiso_pjm.csv",
        # Quick: 5 weeks of data — gives ≥2 folds with 10d train + 5d eval
        "date_start": "2026-01-15",
        "date_end": "2026-02-28",
    },
    {
        "name": "caiso_pjm_ercot_da_rt",
        "regions": ["us-west", "us-east", "us-south"],
        "da_price_file": "data/q12026_3region_dam.csv",
        "rt_price_file": "data/q12026_3region_rt.csv",
        # Quick 3-region: 5 weeks gives ≥2 folds
        "date_start": "2026-01-15",
        "date_end": "2026-02-28",
    },
]

QUICK_WORKLOAD_TYPES = ["training", "llm_batch_inference", "realtime_inference"]

# Minimum savings vs current_price_only to flag (not hard fail)
SAVINGS_FLOORS: dict[str, float] = {
    "training": 3.0,
    "fine_tuning": 2.0,
    "llm_batch_inference": 2.0,
    "data_processing": 2.0,
    "scheduled_batch": 1.0,
    "background_maintenance": 3.0,
    "realtime_inference": 0.0,
}

PRIMARY_BASELINE = "current_price_only"
REGRESSION_THRESHOLD_PCT = 2.0   # allowed degradation vs archived baseline
MAX_MISSING_PRICE_PCT = 5.0      # max % of hours using fallback price


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def _get_ml_forecaster_cls():
    """Import ML forecaster class; returns None if unavailable."""
    try:
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        return PriceQuantileForecaster, PriceModelConfig
    except ImportError:
        return None, None


def _get_per_region_forecaster_cls():
    """Import per-region forecaster class; returns None if unavailable."""
    try:
        from aurelius.forecasting.price_model import (
            PerRegionForecaster,
            PerRegionForecasterConfig,
            PriceModelConfig,
        )
        return PerRegionForecaster, PerRegionForecasterConfig, PriceModelConfig
    except ImportError:
        return None, None, None


def _load_carbon_df(region_combo: dict, repo_root: Path) -> pd.DataFrame:
    """Auto-detect and load a carbon CSV for the given region_combo.

    Looks for carbon files co-located with the price data (same date directory).
    Returns an empty DataFrame if no carbon file is found — the optimizer then
    runs price-only (which is the correct fallback, not a silent error).

    Admissible sources: watttime_moer (CAISO only on free tier for now).
    """
    from aurelius.ingestion.grid_apis.csv_importer import CSVCarbonImporter

    carbon_candidates: list[Path] = []

    # Look next to the DA price file
    da_path = repo_root / region_combo["da_price_file"]
    carbon_candidates.append(da_path.parent / "watttime_carbon_q12026.csv")
    carbon_candidates.append(da_path.parent / "watttime_carbon_summer2025.csv")

    # Also check top-level data/ dir
    data_dir = repo_root / "data"
    carbon_candidates.append(data_dir / "watttime_carbon_q12026.csv")

    for candidate in carbon_candidates:
        if candidate.exists():
            try:
                df = CSVCarbonImporter(str(candidate)).load_all()
                # Filter to regions in this combo that have carbon data
                regions_with_carbon = df["region"].unique().tolist() if not df.empty else []
                combo_regions = region_combo["regions"]
                overlap = [r for r in combo_regions if r in regions_with_carbon]
                if overlap:
                    df_filtered = df[df["region"].isin(combo_regions)]
                    return df_filtered
            except Exception:
                pass

    return pd.DataFrame()


def _load_weather_df(region_combo: dict, repo_root: Path) -> pd.DataFrame:
    """Auto-detect and load a weather CSV for the given region_combo.

    Looks for weather files co-located with the price data (same date directory)
    and in the top-level data/ directory.  Returns empty DataFrame if not found —
    the ML forecaster then falls back to price-only mode (no crash).

    Schema: timestamp, region, temperature_c, hdd_f, cdd_f, wind_speed_ms,
            temp_rolling_24h_c, temp_delta_24h_c, source.
    """
    weather_candidates: list[Path] = []

    da_path = repo_root / region_combo["da_price_file"]
    weather_candidates.append(da_path.parent / "weather_q12026.csv")
    weather_candidates.append(da_path.parent / "weather_summer2025.csv")

    data_dir = repo_root / "data"
    weather_candidates.append(data_dir / "weather_q12026.csv")

    for candidate in weather_candidates:
        if candidate.exists():
            try:
                df = pd.read_csv(str(candidate))
                if df.empty or "timestamp" not in df.columns or "region" not in df.columns:
                    continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                combo_regions = region_combo["regions"]
                df_filtered = df[df["region"].isin(combo_regions)]
                if not df_filtered.empty:
                    return df_filtered.reset_index(drop=True)
            except Exception:
                pass

    return pd.DataFrame()


def run_single_benchmark(
    *,
    region_combo: dict,
    workload_type: str,
    train_days: int = 30,
    eval_days: int = 7,
    num_jobs: int = 50,
    method: str = "greedy_migrate",
    oracle: bool = False,
    forecaster: str = "seasonal_naive",
    carbon_file: Optional[str] = None,
    weather_file: Optional[str] = None,
    queue_file: Optional[str] = None,
    queue_delay_cost_per_gpu_hour: float = 0.0,
    gpu_file: Optional[str] = None,
    gpu_health_cost_per_hour: float = 0.0,
    repo_root: Path,
) -> dict:
    """Run one (region_combo × workload_type) benchmark cell.

    Args:
        forecaster:   "seasonal_naive" (default, no leakage risk) or
                      "ml_quantile" (LightGBM quantile, re-fit per fold) or
                      "ml_quantile_weather" (ML quantile + weather features).
        carbon_file:  Path to carbon CSV (relative to repo root). If None,
                      auto-detects from co-located files. If no carbon data is
                      found, the optimizer runs price-only (correct fallback).
        weather_file: Path to weather CSV (relative to repo root). If None,
                      auto-detects from co-located files. If no weather data is
                      found, the ML forecaster runs price-only (correct fallback).

    Returns a result dict with savings vs each baseline.
    """
    da_file = repo_root / region_combo["da_price_file"]
    rt_file = repo_root / region_combo["rt_price_file"] if region_combo["rt_price_file"] else None
    regions = region_combo["regions"]

    # Check that data file exists before attempting to load it — gives a clear
    # SKIPPED message instead of a confusing file-not-found stack trace.
    if not da_file.exists():
        return {
            "region_combo": region_combo["name"],
            "workload_type": workload_type,
            "skipped": True,
            "skip_reason": f"DA price file not found: {region_combo['da_price_file']} "
                           f"— run scripts/build_combined_dataset.py first",
        }

    price_df = CSVPriceImporter(str(da_file)).load_all()
    price_df = price_df[price_df["region"].isin(regions)]
    if price_df.empty:
        return {"error": f"Empty price data for {region_combo['name']} / {regions}"}

    settle_df = None
    if rt_file and rt_file.exists():
        settle_df = CSVPriceImporter(str(rt_file)).load_all()
        settle_df = settle_df[settle_df["region"].isin(regions)]
        if settle_df.empty:
            settle_df = None

    # Carbon data: explicit file takes precedence, then auto-detect
    if carbon_file:
        from aurelius.ingestion.grid_apis.csv_importer import CSVCarbonImporter
        carbon_df = CSVCarbonImporter(str(repo_root / carbon_file)).load_all()
        carbon_df = carbon_df[carbon_df["region"].isin(regions)] if not carbon_df.empty else pd.DataFrame()
    else:
        carbon_df = _load_carbon_df(region_combo, repo_root)

    carbon_regions = carbon_df["region"].unique().tolist() if not carbon_df.empty else []
    if carbon_regions:
        print(f"  Carbon signal: {carbon_regions} (watttime_moer)")

    # Weather data for ML forecaster:
    # - ml_quantile_weather: joint model with all-region weather features
    # - ml_quantile_perregion: per-region model; weather applied selectively
    #   (ERCOT/us-south gets weather; CAISO/PJM remain price-only)
    # - plain ml_quantile / ml_quantile_recovery: price-only v2.0, no weather
    weather_df_loaded: pd.DataFrame = pd.DataFrame()
    use_weather = forecaster in ("ml_quantile_weather", "ml_quantile_perregion")
    if use_weather:
        if weather_file and weather_file != "none":
            try:
                weather_df_loaded = pd.read_csv(str(repo_root / weather_file))
                weather_df_loaded["timestamp"] = pd.to_datetime(
                    weather_df_loaded["timestamp"], utc=True
                )
                weather_df_loaded = weather_df_loaded[
                    weather_df_loaded["region"].isin(regions)
                ].reset_index(drop=True)
            except Exception as exc:
                print(f"  WARNING: failed to load weather file {weather_file}: {exc}")
        else:
            weather_df_loaded = _load_weather_df(region_combo, repo_root)

    weather_regions = (
        weather_df_loaded["region"].unique().tolist()
        if not weather_df_loaded.empty else []
    )
    if weather_regions:
        print(f"  Weather signal: {weather_regions} (iem_asos_metar)")

    start_ts = pd.Timestamp(region_combo["date_start"], tz="UTC")
    end_ts = pd.Timestamp(region_combo["date_end"], tz="UTC")

    # Synthetic jobs spanning backtest window
    ingester = JobLogIngester()
    backtest_hours = int((end_ts - start_ts).total_seconds() / 3600)
    duration_hours = int(backtest_hours / 0.7) + 24
    sim_start = start_ts.to_pydatetime()

    jobs = ingester.generate_synthetic(
        start_time=sim_start,
        duration_hours=duration_hours,
        num_jobs=num_jobs,
        regions=regions,
        seed=42,
        workload_mix="realistic",
        workload_filter=workload_type,
    )

    config = OptimizationConfig()

    # Wire ML forecaster if requested; fall back gracefully if unavailable
    price_forecaster_cls = None
    price_forecaster_config = None
    apply_recovery_correction = False
    effective_forecaster = forecaster
    if forecaster in ("ml_quantile", "ml_quantile_weather", "ml_quantile_recovery"):
        PriceQuantileForecaster, PriceModelConfig = _get_ml_forecaster_cls()
        if PriceQuantileForecaster is not None:
            price_forecaster_cls = PriceQuantileForecaster
            # v3.0: weather features enabled when weather data is available
            # include_rank_features=False preserves exact v2.0 baseline
            price_forecaster_config = PriceModelConfig(
                seed=42,
                n_estimators=200,
                learning_rate=0.05,
                include_volatility_features=True,
                num_leaves=63,
                include_weather_features=True,
                include_rank_features=False,
            )
        else:
            print("  WARNING: ml_quantile unavailable, falling back to seasonal_naive")
            effective_forecaster = "seasonal_naive"
        # ml_quantile_recovery = v2.0 + regime-aware recovery correction.
        # Training workloads are excluded: the exponential decay distorts long-horizon
        # (96-200h) start-time decisions, causing a -2.7pp regression. Recovery
        # correction helps flexible/maintenance workloads but hurts training.
        if forecaster == "ml_quantile_recovery" and price_forecaster_cls is not None:
            apply_recovery_correction = True
    elif forecaster == "ml_quantile_v5":
        # v5.0: price rank/percentile features + lag_336h (bi-weekly lag)
        PriceQuantileForecaster, PriceModelConfig = _get_ml_forecaster_cls()
        if PriceQuantileForecaster is not None:
            price_forecaster_cls = PriceQuantileForecaster
            price_forecaster_config = PriceModelConfig(
                seed=42,
                n_estimators=200,
                learning_rate=0.05,
                include_volatility_features=True,
                num_leaves=63,
                include_weather_features=False,
                include_rank_features=True,
            )
        else:
            print("  WARNING: ml_quantile_v5 unavailable, falling back to seasonal_naive")
            effective_forecaster = "seasonal_naive"
    elif forecaster == "ml_quantile_perregion":
        PerRegionForecaster, PerRegionForecasterConfig, PriceModelConfig = _get_per_region_forecaster_cls()
        if PerRegionForecaster is not None:
            price_forecaster_cls = PerRegionForecaster
            # Base config: applied to all regions (price-only volatility features)
            base_cfg = PriceModelConfig(
                seed=42,
                n_estimators=200,
                learning_rate=0.05,
                include_volatility_features=True,
                num_leaves=63,
                include_weather_features=True,
            )
            # ERCOT (us-south) gets more capacity: higher num_leaves for spike patterns
            ercot_cfg = PriceModelConfig(
                seed=42,
                n_estimators=250,
                learning_rate=0.05,
                include_volatility_features=True,
                num_leaves=127,
                include_weather_features=True,
            )
            price_forecaster_config = PerRegionForecasterConfig(
                base_config=base_cfg,
                weather_regions=["us-south"],   # ERCOT gets weather; CAISO/PJM don't
                region_configs={"us-south": ercot_cfg},
            )
        else:
            print("  WARNING: ml_quantile_perregion unavailable, falling back to seasonal_naive")
            effective_forecaster = "seasonal_naive"

    # Load queue state CSV if provided
    queue_df_loaded: Optional[pd.DataFrame] = None
    if queue_file and queue_file != "none":
        try:
            _qpath = repo_root / queue_file
            queue_df_loaded = pd.read_csv(str(_qpath))
            queue_df_loaded["timestamp"] = pd.to_datetime(
                queue_df_loaded["timestamp"], utc=True
            )
            print(f"  Queue signal: {queue_df_loaded['region'].nunique()} regions, "
                  f"{len(queue_df_loaded)} rows from {queue_file}")
        except Exception as exc:
            print(f"  WARNING: failed to load queue file {queue_file}: {exc}")

    # Apply queue delay cost to optimizer config when queue data is provided
    if queue_df_loaded is not None and queue_delay_cost_per_gpu_hour > 0.0:
        config = OptimizationConfig(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            min_power_fraction=config.min_power_fraction,
            max_power_fraction=config.max_power_fraction,
            region_power_caps=config.region_power_caps,
            default_region=config.default_region,
            carbon_objective=config.carbon_objective,
            carbon_threshold_gco2_per_kwh=config.carbon_threshold_gco2_per_kwh,
            data_transfer_cost_per_gb=config.data_transfer_cost_per_gb,
            sla_risk_thresholds=config.sla_risk_thresholds,
            queue_delay_cost_per_gpu_hour=queue_delay_cost_per_gpu_hour,
        )
        print(f"  Queue-aware routing: cost_per_gpu_hour=${queue_delay_cost_per_gpu_hour:.2f}")

    # Load GPU telemetry CSV if provided
    gpu_df_loaded: Optional[pd.DataFrame] = None
    if gpu_file and gpu_file != "none":
        try:
            _gpath = repo_root / gpu_file
            gpu_df_loaded = pd.read_csv(str(_gpath))
            gpu_df_loaded["timestamp"] = pd.to_datetime(gpu_df_loaded["timestamp"], utc=True)
            print(f"  GPU telemetry: {gpu_df_loaded['region'].nunique()} regions, "
                  f"{len(gpu_df_loaded)} rows from {gpu_file}")
        except Exception as exc:
            print(f"  WARNING: failed to load GPU telemetry file {gpu_file}: {exc}")

    # Apply GPU health cost to optimizer config when GPU data is provided
    if gpu_df_loaded is not None and gpu_health_cost_per_hour > 0.0:
        config = OptimizationConfig(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            min_power_fraction=config.min_power_fraction,
            max_power_fraction=config.max_power_fraction,
            region_power_caps=config.region_power_caps,
            default_region=config.default_region,
            carbon_objective=config.carbon_objective,
            carbon_threshold_gco2_per_kwh=config.carbon_threshold_gco2_per_kwh,
            data_transfer_cost_per_gb=config.data_transfer_cost_per_gb,
            sla_risk_thresholds=config.sla_risk_thresholds,
            queue_delay_cost_per_gpu_hour=config.queue_delay_cost_per_gpu_hour,
            gpu_health_cost_per_hour=gpu_health_cost_per_hour,
        )
        print(f"  GPU-health-aware routing: cost_per_hour=${gpu_health_cost_per_hour:.2f}")

    engine = BacktestEngine(
        method=method,
        train_days=train_days,
        eval_days=eval_days,
        config=config,
        rt_risk_lambda=1.0 if settle_df is not None else None,
        price_forecaster_cls=price_forecaster_cls,
        price_forecaster_config=price_forecaster_config,
        context_hours=336,  # 2 weeks for lag_168h to work across the full eval horizon
        weather_df=weather_df_loaded if not weather_df_loaded.empty else None,
        queue_df=queue_df_loaded,
        gpu_df=gpu_df_loaded,
        apply_recovery_correction=apply_recovery_correction,
        recovery_excluded_workload_types=frozenset({"training"}) if apply_recovery_correction else frozenset(),
    )
    if oracle:
        engine.oracle_forecast = True

    rounds = engine.run(
        jobs,
        price_df,
        carbon_df=carbon_df,
        start=start_ts,
        end=end_ts,
        settle_price_df=settle_df,
    )

    if not rounds:
        return {"error": "no folds produced", "region": region_combo["name"], "workload": workload_type}

    # Aggregate across folds
    opt_costs = [r.optimizer_metrics.total_energy_cost_usd for r in rounds if r.optimizer_metrics]
    missing_hours = sum(r.optimizer_metrics.missing_price_hours for r in rounds if r.optimizer_metrics)
    # Total eval jobs across folds (used to normalise missing-hour rate)
    total_eval_jobs = sum(r.optimizer_metrics.jobs_evaluated for r in rounds if r.optimizer_metrics)

    mean_opt = sum(opt_costs) / len(opt_costs) if opt_costs else 0.0

    # Build per-baseline savings
    savings: dict[str, dict] = {}
    for bl_name in ALL_BASELINES:
        bl_costs = []
        for r in rounds:
            if bl_name in r.baseline_metrics:
                m = r.baseline_metrics[bl_name]
                bl_costs.append(m.total_energy_cost_usd)
        if not bl_costs:
            continue
        mean_bl = sum(bl_costs) / len(bl_costs)
        pct = (mean_bl - mean_opt) / mean_bl * 100 if mean_bl > 0 else 0.0
        abs_usd = mean_bl - mean_opt
        savings[bl_name] = {"savings_pct": round(pct, 3), "savings_usd": round(abs_usd, 4),
                             "mean_opt_cost": round(mean_opt, 4), "mean_baseline_cost": round(mean_bl, 4)}

    # missing_price_pct: fraction of missing hour-lookups relative to a rough
    # estimate of total scheduled hours (jobs × avg 12h runtime proxy).
    # This is intentionally approximate — what matters is whether it's near-zero.
    estimated_total_hours = max(1, total_eval_jobs * 12)
    missing_pct = min(100.0, (missing_hours / (estimated_total_hours + missing_hours)) * 100)

    # Collect per-fold forecast quality if ML mode was used
    forecast_quality_summary = None
    if effective_forecaster in ("ml_quantile", "ml_quantile_perregion", "ml_quantile_v5",
                                 "ml_quantile_recovery"):
        fq_records = [r.forecast_quality.to_dict() for r in rounds if r.forecast_quality is not None]
        if fq_records:
            import math
            valid_mapes = [f["mape"] for f in fq_records if f.get("mape") is not None and not math.isnan(f["mape"])]
            valid_covgs = [f["p90_coverage"] for f in fq_records if f.get("p90_coverage") is not None and not math.isnan(f["p90_coverage"])]
            forecast_quality_summary = {
                "forecaster": effective_forecaster,
                "folds_with_quality": len(fq_records),
                "mean_mape": round(sum(valid_mapes) / len(valid_mapes), 4) if valid_mapes else None,
                "mean_p90_coverage": round(sum(valid_covgs) / len(valid_covgs), 4) if valid_covgs else None,
            }

    result = {
        "region_combo": region_combo["name"],
        "regions": regions,
        "workload_type": workload_type,
        "folds": len(rounds),
        "num_eval_jobs": sum(len(r.eval_jobs) for r in rounds),
        "method": method,
        "forecaster": effective_forecaster,
        "oracle": oracle,
        "settle_model": rt_file is not None and settle_df is not None,
        "carbon_regions": carbon_regions,
        "missing_price_hours": missing_hours,
        "missing_price_pct": round(missing_pct, 2),
        "savings": savings,
        "primary_savings_pct": savings.get(PRIMARY_BASELINE, {}).get("savings_pct"),
        "date_range": {"start": region_combo["date_start"], "end": region_combo["date_end"]},
        "train_days": train_days,
        "eval_days": eval_days,
    }
    if forecast_quality_summary:
        result["forecast_quality"] = forecast_quality_summary
    return result


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(rounds_data: list[dict]) -> list[str]:
    """Check for known leakage patterns in result dicts. Returns list of issues."""
    issues = []
    for r in rounds_data:
        if r.get("oracle") and r.get("primary_savings_pct") is not None:
            if r["primary_savings_pct"] > 0:
                issues.append(
                    f"[LEAKAGE-RISK] Oracle result for {r['workload_type']}@{r['region_combo']} "
                    f"shows {r['primary_savings_pct']:.1f}% savings — "
                    f"DIAGNOSTIC ONLY, must not appear in real savings claims"
                )
        if r.get("missing_price_pct", 0) > MAX_MISSING_PRICE_PCT:
            issues.append(
                f"[DATA-QUALITY] {r['workload_type']}@{r['region_combo']}: "
                f"{r['missing_price_pct']:.1f}% missing price hours "
                f"(threshold {MAX_MISSING_PRICE_PCT}%) — results may be unreliable"
            )
    return issues


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def compare_against_baseline(
    current: list[dict],
    previous: list[dict],
    threshold_pct: float = REGRESSION_THRESHOLD_PCT,
) -> list[str]:
    """Compare current results against previous benchmark. Returns regression list."""
    regressions = []
    prev_index = {
        (r["workload_type"], r["region_combo"]): r
        for r in previous
        if "workload_type" in r and "region_combo" in r
    }
    for r in current:
        key = (r.get("workload_type"), r.get("region_combo"))
        prev = prev_index.get(key)
        if prev is None:
            continue  # new entry, no regression possible
        cur_pct = r.get("primary_savings_pct")
        prev_pct = prev.get("primary_savings_pct")
        if cur_pct is None or prev_pct is None:
            continue
        if prev_pct - cur_pct > threshold_pct:
            regressions.append(
                f"REGRESSION: {key[0]}@{key[1]}: "
                f"savings vs {PRIMARY_BASELINE} dropped {prev_pct:.1f}% → {cur_pct:.1f}% "
                f"(delta {cur_pct - prev_pct:.1f}%, threshold -{threshold_pct}%)"
            )
    return regressions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Reduced suite for CI smoke testing")
    p.add_argument("--extended-data", action="store_true",
                   help="Include EXTENDED_REGION_COMBOS (combined_2025_2026 dataset, ~270 days). "
                        "Requires: python scripts/build_combined_dataset.py first. "
                        "Use with --train-days 90 for per-region forecaster validation.")
    p.add_argument("--output-dir", default="benchmarks/results", help="Output directory")
    p.add_argument("--workload", help="Run a single workload type")
    p.add_argument("--region-combo", help="Run a single region combo by name")
    p.add_argument("--oracle", action="store_true", help="Include oracle diagnostics (ceiling analysis)")
    p.add_argument("--compare-baseline", help="Path to previous benchmark JSON for regression check")
    p.add_argument("--train-days", type=int, default=30, help="Training window per fold (days)")
    p.add_argument("--eval-days", type=int, default=7, help="Eval window per fold (days)")
    p.add_argument("--num-jobs", type=int, default=50, help="Synthetic jobs per run")
    p.add_argument("--method", default="greedy_migrate",
                   choices=["greedy", "greedy_migrate", "local_search", "local_search_migrate",
                            "greedy_migrate_dp", "local_search_migrate_dp"],
                   help="Optimizer method")
    p.add_argument("--forecaster", default="seasonal_naive",
                   choices=["seasonal_naive", "ml_quantile", "ml_quantile_weather",
                            "ml_quantile_perregion", "ml_quantile_v5", "ml_quantile_recovery"],
                   help="Forecasting method: 'seasonal_naive' (default, no ML), "
                        "'ml_quantile' (LightGBM v2.0 — volatility features, preserved baseline), "
                        "'ml_quantile_weather' (joint model + weather features), "
                        "'ml_quantile_perregion' (one model per region; ERCOT gets weather, "
                        "CAISO/PJM price-only — eliminates cross-region feature stealing), "
                        "'ml_quantile_v5' (v5.0 — adds price rank features + lag_336h for "
                        "better cheap-regime routing), "
                        "'ml_quantile_recovery' (v2.0 + regime-aware recovery correction: "
                        "reduces forecast bias when recent prices << training mean). "
                        "NEVER mix oracle with ml_quantile for savings claims.")
    p.add_argument("--carbon-file", default=None,
                   help="Path to carbon CSV (relative to repo root). If omitted, auto-detects "
                        "from co-located files. Admissible: watttime_moer (production data only).")
    p.add_argument("--weather-file", default=None,
                   help="Path to weather CSV (relative to repo root). If omitted, auto-detects "
                        "from co-located files (data/weather_q12026.csv etc.). "
                        "Pass 'none' to explicitly disable weather features.")
    p.add_argument("--queue-file", default=None,
                   help="Path to queue-state CSV (relative to repo root). Schema: "
                        "timestamp,region,cluster_id,gpu_type,available_gpus,"
                        "queue_depth_jobs,est_wait_hours. Enables queue-aware routing.")
    p.add_argument("--queue-delay-cost", type=float, default=0.0,
                   help="Opportunity cost per GPU-hour lost to queue waiting ($/GPU-hour). "
                        "Requires --queue-file. Typical value for H100: 2.0–4.0.")
    p.add_argument("--gpu-file", default=None,
                   help="Path to GPU telemetry CSV (relative to repo root). "
                        "Canonical schema: timestamp,region,node_id,gpu_index,gpu_uuid,"
                        "gpu_type,gpu_util_pct,mem_used_mb,mem_total_mb,power_usage_w,"
                        "gpu_temp_c,ecc_sbe_count,ecc_dbe_count,xid_error_count,"
                        "power_throttle_us,thermal_throttle_us,clock_throttle_reasons. "
                        "Enables GPU-health-aware placement (Tier 3). "
                        "Use DCGMProvider.generate_fixture() for synthetic demo data or "
                        "DCGMProvider.from_prometheus_live() for production data. "
                        "NOTE: SYNTHETIC fixture data must NOT be used for savings claims.")
    p.add_argument("--gpu-health-cost", type=float, default=0.0,
                   help="Penalty per degraded-GPU-hour ($/GPU-hour). "
                        "Requires --gpu-file. Set to 0 to disable GPU-health routing. "
                        "Typical value for H100: 1.0–3.0.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    workloads = QUICK_WORKLOAD_TYPES if args.quick else WORKLOAD_TYPES
    region_combos = QUICK_REGION_COMBOS if args.quick else REGION_COMBOS

    # Add extended combos when requested (e.g. --extended-data --train-days 90)
    if getattr(args, "extended_data", False) and not args.quick:
        region_combos = region_combos + EXTENDED_REGION_COMBOS

    if args.workload:
        if args.workload not in WORKLOAD_TYPES:
            print(f"ERROR: unknown workload type '{args.workload}'. Valid: {WORKLOAD_TYPES}", file=sys.stderr)
            return 2
        workloads = [args.workload]

    if args.region_combo:
        all_combo_names = [c["name"] for c in REGION_COMBOS + EXTENDED_REGION_COMBOS]
        matches = [c for c in REGION_COMBOS + EXTENDED_REGION_COMBOS if c["name"] == args.region_combo]
        if not matches:
            print(f"ERROR: unknown region combo '{args.region_combo}'. Valid: {all_combo_names}", file=sys.stderr)
            return 2
        region_combos = matches

    results: list[dict] = []
    warnings: list[str] = []
    total_cells = len(workloads) * len(region_combos)
    cell_n = 0

    print(f"\n{'='*70}")
    print(f"AURELIUS BENCHMARK SUITE  ({run_ts})")
    print(f"{'='*70}")
    print(f"Workloads:  {workloads}")
    print(f"Regions:    {[c['name'] for c in region_combos]}")
    print(f"Method:     {args.method}")
    print(f"Forecaster: {args.forecaster}")
    print(f"Quick:      {args.quick}")
    print(f"Oracle:     {args.oracle}")
    print(f"{'='*70}\n")

    # Quick mode uses shorter windows to fit in the reduced date range
    effective_train_days = 10 if args.quick else args.train_days
    effective_eval_days = 5 if args.quick else args.eval_days
    effective_num_jobs = 20 if args.quick else args.num_jobs

    for combo in region_combos:
        for wtype in workloads:
            cell_n += 1
            label = f"[{cell_n}/{total_cells}] {wtype} @ {combo['name']} [{args.forecaster}]"
            print(f"Running {label} ...", flush=True)
            try:
                result = run_single_benchmark(
                    region_combo=combo,
                    workload_type=wtype,
                    train_days=effective_train_days,
                    eval_days=effective_eval_days,
                    num_jobs=effective_num_jobs,
                    method=args.method,
                    oracle=False,
                    forecaster=args.forecaster,
                    carbon_file=args.carbon_file,
                    weather_file=getattr(args, "weather_file", None),
                    queue_file=getattr(args, "queue_file", None),
                    queue_delay_cost_per_gpu_hour=getattr(args, "queue_delay_cost", 0.0),
                    gpu_file=getattr(args, "gpu_file", None),
                    gpu_health_cost_per_hour=getattr(args, "gpu_health_cost", 0.0),
                    repo_root=repo_root,
                )
                result["run_ts"] = run_ts
                results.append(result)

                if result.get("skipped"):
                    print(f"  SKIPPED: {result.get('skip_reason', 'data file missing')}")
                    continue

                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                    warnings.append(f"{label}: {result['error']}")
                    continue

                pct = result.get("primary_savings_pct")
                floor = SAVINGS_FLOORS.get(wtype, 0.0)
                indicator = ""
                if pct is not None and pct < floor and floor > 0:
                    indicator = f"  ⚠ BELOW FLOOR ({floor:.1f}%)"
                    warnings.append(f"{label}: savings {pct:.1f}% below floor {floor:.1f}%")
                print(f"  vs current_price_only: {pct:.1f}%{indicator}  [folds={result['folds']}]")

            except Exception as exc:
                print(f"  FAILED: {exc}")
                results.append({"region_combo": combo["name"], "workload_type": wtype,
                                 "error": str(exc), "run_ts": run_ts})
                warnings.append(f"{label}: exception: {exc}")

    # Oracle diagnostic runs (separate, clearly labeled)
    if args.oracle:
        print(f"\n{'='*70}")
        print("ORACLE DIAGNOSTICS  (ceiling analysis — NEVER present as real savings)")
        print(f"{'='*70}\n")
        oracle_results = []
        for combo in region_combos[:1]:  # only first combo for oracle
            for wtype in workloads[:3]:  # only first 3 workloads for oracle
                label = f"[ORACLE] {wtype} @ {combo['name']}"
                print(f"Running {label} ...", flush=True)
                try:
                    r = run_single_benchmark(
                        region_combo=combo,
                        workload_type=wtype,
                        train_days=effective_train_days,
                        eval_days=effective_eval_days,
                        num_jobs=effective_num_jobs,
                        method=args.method,
                        oracle=True,
                        forecaster="seasonal_naive",  # oracle always uses naive (ceiling test, not ML test)
                        carbon_file=args.carbon_file,
                        weather_file=None,  # oracle doesn't use weather (it sees actual prices)
                        repo_root=repo_root,
                    )
                    r["run_ts"] = run_ts
                    oracle_results.append(r)
                    pct = r.get("primary_savings_pct")
                    print(f"  CEILING vs current_price_only: {pct:.1f}%  [DIAGNOSTIC]")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    oracle_results.append({"error": str(exc), "oracle": True})

        if oracle_results:
            oracle_file = output_dir / f"oracle_{run_ts}.json"
            with open(oracle_file, "w") as f:
                json.dump(oracle_results, f, indent=2, default=str)
            print(f"\nOracle results: {oracle_file}  [DIAGNOSTIC ONLY — never cite as savings]")

    # Leakage audit
    leakage_issues = leakage_audit(results)

    # Regression check
    regressions = []
    if args.compare_baseline and Path(args.compare_baseline).exists():
        with open(args.compare_baseline) as f:
            previous = json.load(f)
        regressions = compare_against_baseline(results, previous)

    # Print summary table
    non_error = [r for r in results if "error" not in r and not r.get("skipped")]
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY  —  primary baseline: {PRIMARY_BASELINE}")
    print(f"{'='*70}")
    print(f"{'Workload':<28}  {'Region':<22}  {'vs current_price_only':>22}  {'Folds':>6}  {'MissPct':>8}")
    print(f"{'-'*90}")
    for r in non_error:
        pct = r.get("primary_savings_pct")
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        floor = SAVINGS_FLOORS.get(r.get("workload_type", ""), 0.0)
        flag = " ⚠" if pct is not None and pct < floor and floor > 0 else ""
        print(f"  {r.get('workload_type','?'):<26}  {r.get('region_combo','?'):<22}  "
              f"{pct_str:>21}{flag}  {r.get('folds',0):>6}  {r.get('missing_price_pct',0):>7.1f}%")

    if non_error:
        all_pcts = [r["primary_savings_pct"] for r in non_error if r.get("primary_savings_pct") is not None]
        if all_pcts:
            mean_pct = sum(all_pcts) / len(all_pcts)
            print(f"\n  Mean savings vs {PRIMARY_BASELINE}: {mean_pct:.1f}%")
            print(f"  Min: {min(all_pcts):.1f}%   Max: {max(all_pcts):.1f}%")

    # Report warnings, leakage issues, regressions
    exit_code = 0
    if warnings:
        print(f"\n⚠  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   {w}")

    if leakage_issues:
        print(f"\n⚠  LEAKAGE / DATA-QUALITY ISSUES ({len(leakage_issues)}):")
        for issue in leakage_issues:
            print(f"   {issue}")

    if regressions:
        print(f"\n✗  REGRESSIONS DETECTED ({len(regressions)}) — BENCHMARK FAILED:")
        for reg in regressions:
            print(f"   {reg}")
        exit_code = 1
    elif args.compare_baseline:
        print(f"\n✓  No regressions vs {args.compare_baseline}")

    # Optional: persist benchmark results to TimeSeriesStore (no-op if DATABASE_URL absent)
    if _TSStore is not None and not args.quick:
        _db_url = os.environ.get("DATABASE_URL", "")
        if _db_url:
            _store = _TSStore(_db_url)
            if _store.enabled:
                for r in non_error:
                    pct = r.get("primary_savings_pct")
                    if pct is not None:
                        _store.save_benchmark_run(
                            run_id=run_ts,
                            forecaster=args.forecaster,
                            region_combo=r.get("region_combo", ""),
                            workload=r.get("workload_type", ""),
                            savings_vs_cpo=float(pct),
                            folds=r.get("folds", 0),
                            miss_pct=float(r.get("missing_price_pct", 0.0)),
                        )
                _store.close()

    # Save results
    result_file = output_dir / f"benchmark_{run_ts}.json"
    full_output = {
        "run_ts": run_ts,
        "quick": args.quick,
        "method": args.method,
        "primary_baseline": PRIMARY_BASELINE,
        "results": results,
        "warnings": warnings,
        "leakage_issues": leakage_issues,
        "regressions": regressions,
    }
    with open(result_file, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"\nResults saved: {result_file}")

    # Save summary text
    summary_file = output_dir / f"summary_{run_ts}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Aurelius Benchmark Summary — {run_ts}\n")
        f.write(f"Primary baseline: {PRIMARY_BASELINE}\n")
        f.write(f"Method: {args.method}\n\n")
        for r in non_error:
            pct = r.get("primary_savings_pct")
            f.write(f"{r.get('workload_type','?')}@{r.get('region_combo','?')}: "
                    f"{pct:.1f}% vs {PRIMARY_BASELINE}  (folds={r.get('folds',0)})\n")
        if non_error and all_pcts:
            f.write(f"\nMean: {mean_pct:.1f}%  Min: {min(all_pcts):.1f}%  Max: {max(all_pcts):.1f}%\n")
        if regressions:
            f.write(f"\nREGRESSIONS:\n")
            for reg in regressions:
                f.write(f"  {reg}\n")
    print(f"Summary saved:  {summary_file}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
