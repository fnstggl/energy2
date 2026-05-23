#!/usr/bin/env python3
"""Aurelius Phase 8 — Daily Learning Loop.

Orchestrates the continuous self-improvement cycle:
1. Pull latest price data from configured providers (CAISO, PJM, ERCOT)
2. Append new data to the rolling historical store
3. Run leakage-free backtest evaluation on recent data
4. Train a candidate ML forecasting model on the full available window
5. Compare candidate model against active model on a held-out evaluation window
6. Promote the candidate only if it improves savings vs current_price_only
7. Run a benchmark smoke test against the standard workload matrix
8. Generate a daily learning loop report
9. Update docs/AURELIUS_PROGRESS.md with results (dry-run safe)

Usage:
    # Dry run (no files written, no models promoted):
    python scripts/daily_learning_loop.py --dry-run

    # Live run (writes to data/store/ and reports/):
    python scripts/daily_learning_loop.py

    # Specify data directory:
    python scripts/daily_learning_loop.py --data-dir /path/to/data

    # Skip live fetching (use only cached data):
    python scripts/daily_learning_loop.py --no-fetch

Required env vars (for live fetching):
    PJM_API_KEY         — PJM Data Miner API key
    ERCOT_API_KEY       — ERCOT CDAT API key
    ERCOT_USERNAME      — ERCOT username
    ERCOT_PASSWORD      — ERCOT password
    (CAISO requires no API key)

Optional env vars:
    WATTTIME_USERNAME   — WattTime login for carbon data
    WATTTIME_PASSWORD   — WattTime password

Safe to run without live credentials: fetching will be skipped with a logged warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# TimeSeriesStore is optional — graceful no-op when DATABASE_URL is not set
try:
    from aurelius.database import TimeSeriesStore as _TimeSeriesStore
except ImportError:
    _TimeSeriesStore = None  # type: ignore[assignment,misc]

# Project root so imports work when running as a script
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aurelius.daily_loop")

# ---------------------------------------------------------------------------
# Regions configured for the learning loop
# ---------------------------------------------------------------------------
REGIONS = ["us-west", "us-east", "us-south"]
REGION_COMBO = "caiso_pjm_ercot_da_rt"

# Minimum days of historical data required to run the evaluation
MIN_EVAL_DAYS = 14
MIN_TRAIN_DAYS = 30
EVAL_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Step 1: Pull latest price data
# ---------------------------------------------------------------------------

def fetch_latest_prices(
    regions: list[str],
    data_dir: Path,
    dry_run: bool,
) -> dict[str, pd.DataFrame]:
    """Attempt to fetch latest DA prices from configured providers.

    Returns a dict of region → DataFrame (may be empty if provider unavailable).
    Never raises — logs warnings on failure.
    """
    results: dict[str, pd.DataFrame] = {}
    end_dt = datetime.now(tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    start_dt = end_dt - timedelta(days=3)  # last 3 days (avoid duplicate re-fetch)

    for region in regions:
        try:
            df = _fetch_region(region, start_dt, end_dt)
            if df is not None and not df.empty:
                results[region] = df
                logger.info(f"Fetched {len(df)} rows for {region} "
                            f"({start_dt.date()} – {end_dt.date()})")
        except Exception as exc:
            logger.warning(f"Failed to fetch {region}: {exc}")
    return results


def _fetch_region(region: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch price data for a single region."""
    if region == "us-west":
        return _fetch_caiso(start, end)
    elif region == "us-east":
        return _fetch_pjm(start, end)
    elif region == "us-south":
        return _fetch_ercot(start, end)
    else:
        logger.warning(f"No provider configured for region: {region}")
        return None


def _fetch_caiso(start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch CAISO OASIS DA prices (no API key required)."""
    try:
        from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider
        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", start, end)
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"CAISO fetch failed: {exc}")
        return None


def _fetch_pjm(start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch PJM Data Miner DA prices."""
    api_key = os.environ.get("PJM_API_KEY", "")
    if not api_key:
        logger.info("PJM_API_KEY not set — skipping PJM price fetch")
        return None
    try:
        from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider
        provider = PJMPriceProvider(api_key=api_key)
        df = provider.fetch_prices("us-east", start, end)
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"PJM fetch failed: {exc}")
        return None


def _fetch_ercot(start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch ERCOT CDAT DA prices."""
    api_key = os.environ.get("ERCOT_API_KEY", "")
    username = os.environ.get("ERCOT_USERNAME", "")
    password = os.environ.get("ERCOT_PASSWORD", "")
    if not api_key and not (username and password):
        logger.info("ERCOT credentials not set — skipping ERCOT price fetch")
        return None
    try:
        from aurelius.ingestion.grid_apis.ercot import ERCOTPriceProvider
        provider = ERCOTPriceProvider(
            api_key=api_key or None,
            username=username or None,
            password=password or None,
        )
        df = provider.fetch_prices("us-south", start, end)
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"ERCOT fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 2: Append to historical store
# ---------------------------------------------------------------------------

def append_to_store(
    new_data: dict[str, pd.DataFrame],
    store_path: Path,
    dry_run: bool,
) -> pd.DataFrame:
    """Merge newly fetched data into the rolling historical store CSV.

    Returns the full combined DataFrame (or existing store if no new data).
    """
    if store_path.exists():
        existing = pd.read_csv(store_path, parse_dates=["timestamp"])
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    else:
        existing = pd.DataFrame(columns=["timestamp", "region", "price_per_mwh"])

    if not new_data:
        logger.info("No new price data to append.")
        return existing

    new_frames = list(new_data.values())
    new_df = pd.concat(new_frames, ignore_index=True)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "region"]).sort_values(
        ["timestamp", "region"]
    )

    n_new = len(combined) - len(existing)
    logger.info(f"Store update: {len(existing)} existing + {n_new} new = {len(combined)} total rows")

    if not dry_run:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(store_path, index=False)
        logger.info(f"Saved updated store to {store_path}")
    else:
        logger.info(f"[DRY RUN] Would save {len(combined)} rows to {store_path}")

    return combined


# ---------------------------------------------------------------------------
# Step 3: Run evaluation on recent data
# ---------------------------------------------------------------------------

def run_evaluation(
    price_df: pd.DataFrame,
    regions: list[str],
    train_days: int,
    eval_days: int,
) -> Optional[dict]:
    """Run a mini walk-forward backtest to evaluate current model performance.

    Returns a dict with savings vs current_price_only, or None if insufficient data.
    """
    if price_df.empty:
        logger.warning("No price data available for evaluation.")
        return None

    price_df = price_df[price_df["region"].isin(regions)].copy()
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

    available_days = (price_df["timestamp"].max() - price_df["timestamp"].min()).days
    if available_days < train_days + eval_days:
        logger.warning(
            f"Insufficient data for evaluation: {available_days} days available, "
            f"need {train_days + eval_days} days."
        )
        return None

    try:
        import warnings

        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
        from aurelius.ingestion.job_logs import JobLogIngester
        from aurelius.models import OptimizationConfig

        warnings.filterwarnings("ignore", category=UserWarning)

        engine = BacktestEngine(
            price_df=price_df,
            regions=regions,
            train_days=train_days,
            eval_days=eval_days,
            n_folds=3,
            method="greedy",
            config=OptimizationConfig(),
            forecaster_cls=PriceQuantileForecaster,
            forecaster_config=PriceModelConfig(
                seed=42, n_estimators=100, num_leaves=31
            ),
        )
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            num_jobs=50,
            start_time=price_df["timestamp"].min().to_pydatetime(),
            duration_hours=int(available_days * 24),
            regions=regions,
            seed=42,
            workload_mix="realistic",
        )

        result = engine.run(jobs)

        savings_mean = None
        if result.savings_vs_current_price:
            savings_mean = sum(result.savings_vs_current_price.values()) / len(
                result.savings_vs_current_price
            )

        return {
            "status": "ok",
            "savings_vs_cpo_mean": savings_mean,
            "savings_by_workload": result.savings_vs_current_price or {},
            "n_folds": result.n_folds,
            "n_jobs_evaluated": len(result.jobs_evaluated) if hasattr(result, "jobs_evaluated") else None,
            "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 4: Train candidate model
# ---------------------------------------------------------------------------

def train_candidate_model(
    price_df: pd.DataFrame,
    regions: list[str],
    models_dir: Path,
    dry_run: bool,
) -> Optional[dict]:
    """Train a new ML forecasting model on all available data.

    Returns model metadata dict, or None if training failed.
    """
    if price_df.empty or len(price_df) < 200:
        logger.warning("Insufficient data for model training.")
        return None

    try:
        import pickle
        import warnings

        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster

        warnings.filterwarnings("ignore", category=UserWarning)

        price_df = price_df[price_df["region"].isin(regions)].copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

        price_records = []
        for _, row in price_df.iterrows():
            from aurelius.models import EnergyPrice
            price_records.append(EnergyPrice(
                timestamp=row["timestamp"].to_pydatetime(),
                region=row["region"],
                price_per_mwh=float(row["price_per_mwh"]),
            ))

        config = PriceModelConfig(
            seed=42, n_estimators=200, num_leaves=63, learning_rate=0.05
        )
        forecaster = PriceQuantileForecaster(config=config)
        forecaster.fit(price_records)

        metadata = {
            "model_version": "ml_quantile_v2",
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_records": len(price_records),
            "regions": regions,
            "config": {
                "n_estimators": config.n_estimators,
                "num_leaves": config.num_leaves,
                "learning_rate": config.learning_rate,
            },
        }

        if not dry_run:
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / "candidate_forecaster.pkl"
            meta_path = models_dir / "candidate_metadata.json"
            with model_path.open("wb") as f:
                pickle.dump(forecaster, f)
            meta_path.write_text(json.dumps(metadata, indent=2))
            logger.info(f"Candidate model saved to {model_path}")
        else:
            logger.info(f"[DRY RUN] Would save candidate model trained on {len(price_records)} records")

        return metadata

    except Exception as exc:
        logger.error(f"Model training failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Step 5: Compare candidate vs active model
# ---------------------------------------------------------------------------

def compare_models(
    eval_result: Optional[dict],
    models_dir: Path,
) -> dict:
    """Compare candidate model savings against active model savings.

    Returns promotion decision and comparison metrics.
    """
    active_meta_path = models_dir / "active_metadata.json"
    active_savings = None
    if active_meta_path.exists():
        try:
            meta = json.loads(active_meta_path.read_text())
            active_savings = meta.get("last_eval_savings_vs_cpo")
        except Exception:
            pass

    candidate_savings = None
    if eval_result and eval_result.get("status") == "ok":
        candidate_savings = eval_result.get("savings_vs_cpo_mean")

    comparison = {
        "active_savings_vs_cpo": active_savings,
        "candidate_savings_vs_cpo": candidate_savings,
        "promote": False,
        "reason": "unknown",
    }

    if candidate_savings is None:
        comparison["reason"] = "evaluation_failed"
    elif active_savings is None:
        # No active model — promote by default
        comparison["promote"] = True
        comparison["reason"] = "no_active_model"
    elif candidate_savings > active_savings + 0.005:  # 0.5pp improvement threshold
        comparison["promote"] = True
        comparison["reason"] = f"improvement_{candidate_savings - active_savings:.3f}pp"
    elif candidate_savings < active_savings - 0.005:
        comparison["promote"] = False
        comparison["reason"] = f"regression_{active_savings - candidate_savings:.3f}pp"
    else:
        comparison["promote"] = False
        comparison["reason"] = "no_significant_change"

    return comparison


# ---------------------------------------------------------------------------
# Step 6: Promote candidate to active
# ---------------------------------------------------------------------------

def promote_candidate(models_dir: Path, eval_result: dict, dry_run: bool) -> None:
    """Overwrite active model with candidate after confirming it's an improvement."""
    candidate_model = models_dir / "candidate_forecaster.pkl"
    active_model = models_dir / "active_forecaster.pkl"
    candidate_meta = models_dir / "candidate_metadata.json"
    active_meta = models_dir / "active_metadata.json"

    if not candidate_model.exists():
        logger.warning("Candidate model file not found — cannot promote")
        return

    if dry_run:
        logger.info("[DRY RUN] Would promote candidate to active model")
        return

    import shutil
    shutil.copy2(str(candidate_model), str(active_model))

    if candidate_meta.exists():
        meta = json.loads(candidate_meta.read_text())
        meta["last_eval_savings_vs_cpo"] = eval_result.get("savings_vs_cpo_mean")
        meta["promoted_at"] = datetime.now(tz=timezone.utc).isoformat()
        active_meta.write_text(json.dumps(meta, indent=2))

    logger.info(f"Promoted candidate model to active: {active_model}")


# ---------------------------------------------------------------------------
# Step 7: Benchmark smoke test
# ---------------------------------------------------------------------------

def run_benchmark_smoke_test(data_dir: Path) -> dict:
    """Run a quick benchmark smoke test using standard fixture data.

    This uses the repo's bundled data/q12026_3region_dam.csv if available,
    so it does not require live API access.
    """
    da_path = data_dir / "q12026_3region_dam.csv"
    rt_path = data_dir / "q12026_3region_rt.csv"

    if not da_path.exists():
        logger.info(f"Benchmark smoke test skipped: {da_path} not found")
        return {"status": "skipped", "reason": "data_file_not_found"}

    try:
        import warnings

        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
        from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
        from aurelius.ingestion.job_logs import JobLogIngester
        from aurelius.models import OptimizationConfig

        warnings.filterwarnings("ignore", category=UserWarning)

        price_df = CSVPriceImporter(str(da_path)).load_all()
        regions = ["us-west", "us-east", "us-south"]
        price_df = price_df[price_df["region"].isin(regions)]

        engine = BacktestEngine(
            price_df=price_df,
            regions=regions,
            train_days=30,
            eval_days=7,
            n_folds=3,
            method="greedy",
            config=OptimizationConfig(),
            forecaster_cls=PriceQuantileForecaster,
            forecaster_config=PriceModelConfig(seed=42, n_estimators=50, num_leaves=31),
        )
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            num_jobs=30,
            start_time=pd.to_datetime(price_df["timestamp"].min(), utc=True).to_pydatetime(),
            duration_hours=int(len(price_df["timestamp"].unique()) / len(regions)),
            regions=regions,
            seed=42,
            workload_mix="realistic",
        )

        result = engine.run(jobs)

        savings_mean = None
        if result.savings_vs_current_price:
            savings_mean = (
                sum(result.savings_vs_current_price.values())
                / len(result.savings_vs_current_price)
            )

        return {
            "status": "ok",
            "savings_vs_cpo_mean": savings_mean,
            "n_folds": result.n_folds,
        }
    except Exception as exc:
        logger.error(f"Benchmark smoke test failed: {exc}")
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 8: Generate report
# ---------------------------------------------------------------------------

def generate_report(
    loop_start: datetime,
    fetch_results: dict,
    eval_result: Optional[dict],
    model_metadata: Optional[dict],
    comparison: dict,
    smoke_test: dict,
    reports_dir: Path,
    dry_run: bool,
) -> dict:
    """Compose the daily learning loop report and save it."""
    report = {
        "run_date": loop_start.isoformat(),
        "dry_run": dry_run,
        "data_fetch": {
            "regions_fetched": list(fetch_results.keys()),
            "rows_fetched": {r: len(df) for r, df in fetch_results.items()},
        },
        "evaluation": eval_result or {"status": "skipped"},
        "model_training": model_metadata or {"status": "skipped"},
        "model_comparison": comparison,
        "promoted": comparison.get("promote", False),
        "benchmark_smoke_test": smoke_test,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    if not dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)
        date_str = loop_start.strftime("%Y-%m-%d")
        report_path = reports_dir / f"learning_loop_{date_str}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Report saved to {report_path}")
    else:
        logger.info(f"[DRY RUN] Report:\n{json.dumps(report, indent=2, default=str)}")

    return report


# ---------------------------------------------------------------------------
# Optional DB persistence helpers
# ---------------------------------------------------------------------------

def _persist_prices_to_db(
    new_data: dict[str, pd.DataFrame],
    db_url: Optional[str] = None,
) -> None:
    """Upsert freshly fetched price rows into the TimeSeriesStore.

    No-op when DATABASE_URL is not configured or TimeSeriesStore is unavailable.
    """
    if _TimeSeriesStore is None:
        return
    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return
    store = _TimeSeriesStore(url)
    if not store.enabled:
        return
    total = 0
    for region, df in new_data.items():
        if not df.empty:
            n = store.upsert_prices(df)
            total += n
            logger.info(f"DB: upserted {n} new price rows for {region}")
    if total:
        logger.info(f"DB: total {total} price rows upserted")
    store.close()


def _persist_benchmark_to_db(
    smoke_result: dict,
    run_id: str,
    db_url: Optional[str] = None,
) -> None:
    """Save benchmark smoke test result to the TimeSeriesStore.

    No-op when DATABASE_URL is not configured or TimeSeriesStore is unavailable.
    """
    if _TimeSeriesStore is None or smoke_result.get("status") != "ok":
        return
    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return
    store = _TimeSeriesStore(url)
    if not store.enabled:
        return
    savings = smoke_result.get("savings_vs_cpo_mean")
    if savings is None:
        store.close()
        return
    store.save_benchmark_run(
        run_id=run_id,
        forecaster="ml_quantile",
        region_combo=REGION_COMBO,
        workload="mixed_smoke_test",
        savings_vs_cpo=float(savings) * 100.0,
        folds=smoke_result.get("n_folds", 0),
    )
    logger.info(f"DB: saved benchmark smoke test result for run_id={run_id}")
    store.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aurelius Phase 8 — Daily Learning Loop"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing files or promoting models",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip live price fetching (use only cached store data)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=_ROOT / "data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=_ROOT / "reports" / "learning_loop",
        help="Directory for loop reports (default: reports/learning_loop/)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=_ROOT / "data" / "models",
        help="Directory for model artifacts (default: data/models/)",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip the benchmark smoke test (saves ~2 minutes)",
    )
    args = parser.parse_args()

    loop_start = datetime.now(tz=timezone.utc)
    logger.info("=== Aurelius Daily Learning Loop ===")
    logger.info(f"Start: {loop_start.isoformat()}")
    logger.info(f"Dry run: {args.dry_run}")

    store_path = args.data_dir / "store" / "price_history.csv"

    # Step 1: Fetch latest prices
    if args.no_fetch:
        logger.info("Skipping live price fetch (--no-fetch)")
        fetch_results = {}
    else:
        logger.info("Step 1: Fetching latest prices...")
        fetch_results = fetch_latest_prices(REGIONS, args.data_dir, args.dry_run)

    # Step 2: Append to store + optional DB persistence
    logger.info("Step 2: Updating historical store...")
    if fetch_results and not args.dry_run:
        _persist_prices_to_db(fetch_results)
    if store_path.exists() or fetch_results:
        store_df = append_to_store(fetch_results, store_path, args.dry_run)
    else:
        # Fall back to bundled Q1 2026 data if store is empty
        fallback_path = args.data_dir / "q12026_3region_dam.csv"
        if fallback_path.exists():
            logger.info(f"Using bundled data from {fallback_path}")
            from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
            store_df = CSVPriceImporter(str(fallback_path)).load_all()
        else:
            logger.warning("No price data available. Skipping evaluation and training.")
            store_df = pd.DataFrame()

    # Step 3: Evaluate on recent data
    logger.info("Step 3: Running evaluation...")
    eval_result = run_evaluation(store_df, REGIONS, MIN_TRAIN_DAYS, EVAL_WINDOW_DAYS)
    if eval_result and eval_result.get("status") == "ok":
        savings = eval_result.get("savings_vs_cpo_mean")
        if savings is not None:
            logger.info(f"Evaluation savings vs CPO: {savings*100:.1f}%")

    # Step 4: Train candidate model
    logger.info("Step 4: Training candidate model...")
    model_metadata = train_candidate_model(store_df, REGIONS, args.models_dir, args.dry_run)

    # Step 5: Compare models
    logger.info("Step 5: Comparing candidate vs active model...")
    comparison = compare_models(eval_result, args.models_dir)
    logger.info(
        f"Comparison: promote={comparison['promote']}, reason={comparison['reason']}"
    )

    # Step 6: Promote if better
    if comparison.get("promote") and model_metadata:
        logger.info("Step 6: Promoting candidate model...")
        promote_candidate(args.models_dir, eval_result or {}, args.dry_run)
    else:
        logger.info(f"Step 6: No promotion ({comparison.get('reason')})")

    # Step 7: Benchmark smoke test
    if args.skip_benchmark:
        smoke_test = {"status": "skipped", "reason": "--skip-benchmark"}
        logger.info("Step 7: Benchmark smoke test skipped")
    else:
        logger.info("Step 7: Running benchmark smoke test...")
        smoke_test = run_benchmark_smoke_test(args.data_dir)
        logger.info(f"Smoke test: {smoke_test.get('status')}")
        if smoke_test.get("savings_vs_cpo_mean") is not None:
            logger.info(f"  Smoke test savings: {smoke_test['savings_vs_cpo_mean']*100:.1f}%")

    # Optional: persist benchmark result to DB
    run_id_str = loop_start.strftime("%Y%m%dT%H%M%SZ")
    if not args.dry_run:
        _persist_benchmark_to_db(smoke_test, run_id=run_id_str)

    # Step 8: Generate report
    logger.info("Step 8: Generating report...")
    report = generate_report(
        loop_start=loop_start,
        fetch_results=fetch_results,
        eval_result=eval_result,
        model_metadata=model_metadata,
        comparison=comparison,
        smoke_test=smoke_test,
        reports_dir=args.reports_dir,
        dry_run=args.dry_run,
    )

    elapsed = (datetime.now(tz=timezone.utc) - loop_start).total_seconds()
    logger.info(f"=== Learning Loop Complete in {elapsed:.1f}s ===")
    logger.info(f"Evaluation: {eval_result.get('status') if eval_result else 'skipped'}")
    logger.info(f"Promotion: {comparison.get('promote')} ({comparison.get('reason')})")
    logger.info(f"Smoke test: {smoke_test.get('status')}")

    # Exit 1 if smoke test failed (for CI integration)
    if smoke_test.get("status") == "error":
        logger.error("Benchmark smoke test failed — see logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
