#!/usr/bin/env python3
"""Aurelius — Daily Learning Loop.

Orchestrates the continuous self-improvement cycle (single-host locked):
1. Pull latest price data from configured providers (CAISO, PJM, ERCOT)
2. Append new data to the rolling historical store (CSV + Postgres)
3. Model update: train a candidate on a train split, compare it against the
   ACTIVE model (loaded from the registry + artifact store) on a leakage-free
   holdout window, and promote only if it genuinely wins. Promotion decisions
   and model versions are persisted; rollback is supported (--rollback).
4. Run a benchmark smoke test against the standard workload matrix
5. Read realized-outcome feedback back from the store (predicted vs realized)
6. Generate a daily learning loop report

Persistence is gated by DATABASE_URL (Postgres/SQLite via SQLAlchemy); model
binaries live in the artifact store (ARTIFACT_STORE_URI: file://, s3://). All
writes are no-op safe when those are unset.

Usage:
    # Dry run (no files written, no models promoted):
    python scripts/daily_learning_loop.py --dry-run

    # Live run for a specific pilot (writes to store + registry):
    DATABASE_URL=postgresql://... python scripts/daily_learning_loop.py \
        --customer-id acme --pilot-id pilot-q1

    # Roll the active model back to the previous version:
    DATABASE_URL=postgresql://... python scripts/daily_learning_loop.py \
        --rollback --customer-id acme --pilot-id pilot-q1

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

# Project root so imports work when running as a script (must precede any
# aurelius.* import below — running `python scripts/...` only puts the script
# directory on sys.path, not the repo root).
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# TimeSeriesStore is optional — graceful no-op when DATABASE_URL is not set
try:
    from aurelius.database import TimeSeriesStore as _TimeSeriesStore
except ImportError:
    _TimeSeriesStore = None  # type: ignore[assignment,misc]

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

# Length of the leakage-free holdout window used to compare candidate vs active
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
# Savings aggregation helper
# ---------------------------------------------------------------------------

def _mean_savings_vs_cpo(rounds: list) -> Optional[float]:
    """Compute mean fractional savings vs current_price_only across folds.

    Aggregates optimizer vs current_price_only baseline energy cost over all
    folds in a BacktestEngine.run() result. Returns a fraction (e.g. 0.25 for
    25%), or None when no comparable folds are present.
    """
    cpo = "current_price_only"
    opt_costs = [
        r.optimizer_metrics.total_energy_cost_usd
        for r in rounds
        if r.optimizer_metrics is not None
    ]
    bl_costs = [
        r.baseline_metrics[cpo].total_energy_cost_usd
        for r in rounds
        if cpo in r.baseline_metrics
    ]
    if not opt_costs or not bl_costs:
        return None
    mean_opt = sum(opt_costs) / len(opt_costs)
    mean_bl = sum(bl_costs) / len(bl_costs)
    if mean_bl <= 0:
        return None
    return (mean_bl - mean_opt) / mean_bl


# ---------------------------------------------------------------------------
# Legacy scalar comparison (retained for backward-compat tests; main() uses
# the registry-backed candidate-vs-active comparison in aurelius.learning)
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
# Benchmark smoke test
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
        from aurelius.ingestion.grid_apis.base import empty_carbon_df
        from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
        from aurelius.ingestion.job_logs import JobLogIngester
        from aurelius.models import OptimizationConfig

        warnings.filterwarnings("ignore", category=UserWarning)

        price_df = CSVPriceImporter(str(da_path)).load_all()
        regions = ["us-west", "us-east", "us-south"]
        price_df = price_df[price_df["region"].isin(regions)]

        engine = BacktestEngine(
            method="greedy",
            train_days=30,
            eval_days=7,
            config=OptimizationConfig(),
            price_forecaster_cls=PriceQuantileForecaster,
            price_forecaster_config=PriceModelConfig(seed=42, n_estimators=50, num_leaves=31),
            context_hours=336,
        )
        ts_min = pd.to_datetime(price_df["timestamp"].min(), utc=True)
        ts_max = pd.to_datetime(price_df["timestamp"].max(), utc=True)
        span_hours = int((ts_max - ts_min).total_seconds() / 3600)
        ingester = JobLogIngester()
        jobs = ingester.generate_synthetic(
            num_jobs=30,
            start_time=ts_min.to_pydatetime(),
            duration_hours=span_hours,
            regions=regions,
            seed=42,
            workload_mix="realistic",
        )

        rounds = engine.run(jobs, price_df, carbon_df=empty_carbon_df())
        savings_mean = _mean_savings_vs_cpo(rounds)

        return {
            "status": "ok",
            "savings_vs_cpo_mean": savings_mean,
            "n_folds": len(rounds),
        }
    except Exception as exc:
        logger.error(f"Benchmark smoke test failed: {exc}")
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    loop_start: datetime,
    fetch_results: dict,
    smoke_test: dict,
    reports_dir: Path,
    dry_run: bool,
    eval_result: Optional[dict] = None,
    model_metadata: Optional[dict] = None,
    comparison: Optional[dict] = None,
    outcomes_summary: Optional[dict] = None,
    model_update: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Compose the daily learning loop report and save it.

    `model_update` is the registry-backed candidate-vs-active result (the
    trustworthy path used by main()). `eval_result`/`model_metadata`/`comparison`
    are retained for backward compatibility with callers/tests.
    """
    comparison = comparison or {}
    promoted = (
        model_update.get("promoted", False) if model_update else comparison.get("promote", False)
    )
    report = {
        "run_id": run_id,
        "run_date": loop_start.isoformat(),
        "dry_run": dry_run,
        "data_fetch": {
            "regions_fetched": list(fetch_results.keys()),
            "rows_fetched": {r: len(df) for r, df in fetch_results.items()},
        },
        "model_update": model_update or {"status": "skipped"},
        "evaluation": eval_result or {"status": "skipped"},
        "model_training": model_metadata or {"status": "skipped"},
        "model_comparison": comparison,
        "promoted": promoted,
        "benchmark_smoke_test": smoke_test,
        "realized_outcomes_feedback": outcomes_summary or {"status": "skipped"},
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


def read_realized_outcomes_summary(db_url: Optional[str] = None) -> dict:
    """Read historical realized outcomes from the store and summarise feedback.

    This is the loop's read-back of the data moat: realized pilot/shadow
    outcomes (predicted vs realized savings) recorded by shadow mode. Returns
    a summary dict; no-op friendly when the store is disabled.
    """
    if _TimeSeriesStore is None:
        return {"status": "store_unavailable"}
    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return {"status": "disabled"}
    store = _TimeSeriesStore(url)
    if not store.enabled:
        return {"status": "disabled"}
    try:
        rows = store.get_realized_outcomes(limit=5000)
    finally:
        store.close()

    if not rows:
        return {"status": "ok", "n_outcomes": 0}

    realized = [r["realized_savings_pct"] for r in rows if r.get("realized_savings_pct") is not None]
    deltas = [
        r["realized_savings_pct"] - r["predicted_savings_pct"]
        for r in rows
        if r.get("realized_savings_pct") is not None and r.get("predicted_savings_pct") is not None
    ]
    summary = {
        "status": "ok",
        "n_outcomes": len(rows),
        "mean_realized_savings_pct": (sum(realized) / len(realized)) if realized else None,
        "mean_forecast_error_pp": (
            sum(abs(d) for d in deltas) / len(deltas) if deltas else None
        ),
        "customers": sorted({r.get("customer_id", "unknown") for r in rows}),
    }
    logger.info(
        "Realized-outcome feedback: %d outcomes, mean realized savings=%s, "
        "mean |forecast error|=%s pp",
        summary["n_outcomes"],
        f"{summary['mean_realized_savings_pct']:.1f}%" if summary["mean_realized_savings_pct"] is not None else "n/a",
        f"{summary['mean_forecast_error_pp']:.1f}" if summary["mean_forecast_error_pp"] is not None else "n/a",
    )
    return summary


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
    parser.add_argument(
        "--customer-id", default="global",
        help="Customer scope for the model registry (default: global)",
    )
    parser.add_argument(
        "--pilot-id", default="unknown",
        help="Pilot identifier for the model registry (default: unknown)",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Roll the active model back to the previous version, then exit",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="Skip the single-host run lock (NOT recommended for cron/Railway)",
    )
    args = parser.parse_args()

    # --- Rollback path (operator action; exits immediately) ---
    if args.rollback:
        _run_rollback(scope=args.customer_id, pilot_id=args.pilot_id, dry_run=args.dry_run)
        return

    # --- Single-host lock: prevent overlapping cron/Railway runs ---
    lock = None
    if not args.no_lock:
        from aurelius.learning.locking import FileLock, LockNotAcquiredError
        lock = FileLock(str(args.data_dir / ".learning_loop.lock"))
        try:
            lock.acquire()
        except LockNotAcquiredError as exc:
            logger.warning("Another learning-loop run is in progress — exiting. (%s)", exc)
            return

    try:
        _run_loop(args)
    finally:
        if lock is not None:
            lock.release()


def _run_rollback(scope: str, pilot_id: str, dry_run: bool) -> None:
    """Roll the active price model back to the previous version."""
    if _TimeSeriesStore is None or not os.environ.get("DATABASE_URL"):
        logger.error("Rollback requires DATABASE_URL (a persistent model registry).")
        sys.exit(1)
    store = _TimeSeriesStore()
    if not store.enabled:
        logger.error("Model registry unavailable — cannot roll back.")
        sys.exit(1)
    try:
        if dry_run:
            active = store.get_active_model("price", scope, pilot_id)
            logger.info("[DRY RUN] Would roll back active model %s",
                        active["model_id"] if active else "(none)")
            return
        restored = store.rollback_active("price", scope, pilot_id)
        if restored is None:
            logger.warning("Nothing to roll back to (no previous archived model).")
            return
        store.record_promotion_decision(
            decision="rollback", model_type="price", scope=scope, pilot_id=pilot_id,
            model_id=restored["model_id"], reason="operator_rollback",
        )
        logger.info("Rolled back. Active model is now %s", restored["model_id"])
    finally:
        store.close()


def _run_loop(args) -> None:
    import uuid

    loop_start = datetime.now(tz=timezone.utc)
    run_id = uuid.uuid4().hex
    logger.info("=== Aurelius Daily Learning Loop ===")
    logger.info(f"Start: {loop_start.isoformat()}  run_id={run_id}")
    logger.info(f"Dry run: {args.dry_run}  scope={args.customer_id} pilot={args.pilot_id}")

    store_path = args.data_dir / "store" / "price_history.csv"

    # Open the registry/lifecycle store once (no-op when DATABASE_URL unset).
    store = None
    if _TimeSeriesStore is not None and os.environ.get("DATABASE_URL"):
        store = _TimeSeriesStore()
        if store.enabled and not args.dry_run:
            store.start_learning_run(run_id, scope=args.customer_id, pilot_id=args.pilot_id)

    run_state = "completed"
    try:
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
            fallback_path = args.data_dir / "q12026_3region_dam.csv"
            if fallback_path.exists():
                logger.info(f"Using bundled data from {fallback_path}")
                from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
                store_df = CSVPriceImporter(str(fallback_path)).load_all()
            else:
                logger.warning("No price data available. Skipping model update.")
                store_df = pd.DataFrame()

        # Step 3: Model update — train candidate, compare vs ACTIVE on a held-out
        # window, promote only if genuinely better (registry + artifact store).
        logger.info("Step 3: Model update (candidate vs active)...")
        model_update = _run_model_update_step(store_df, store, run_id, args)
        logger.info(
            "Model update: status=%s promoted=%s reason=%s",
            model_update.get("status"), model_update.get("promoted"),
            model_update.get("reason"),
        )

        # Step 4: Benchmark smoke test
        if args.skip_benchmark:
            smoke_test = {"status": "skipped", "reason": "--skip-benchmark"}
            logger.info("Step 4: Benchmark smoke test skipped")
        else:
            logger.info("Step 4: Running benchmark smoke test...")
            smoke_test = run_benchmark_smoke_test(args.data_dir)
            logger.info(f"Smoke test: {smoke_test.get('status')}")
            if smoke_test.get("savings_vs_cpo_mean") is not None:
                logger.info(f"  Smoke test savings: {smoke_test['savings_vs_cpo_mean']*100:.1f}%")

        if not args.dry_run:
            _persist_benchmark_to_db(smoke_test, run_id=run_id)

        # Step 5: Read realized-outcome feedback from the store (data moat).
        logger.info("Step 5: Reading realized-outcome feedback from store...")
        outcomes_summary = read_realized_outcomes_summary()

        # Step 6: Generate report
        logger.info("Step 6: Generating report...")
        generate_report(
            loop_start=loop_start,
            fetch_results=fetch_results,
            smoke_test=smoke_test,
            reports_dir=args.reports_dir,
            dry_run=args.dry_run,
            outcomes_summary=outcomes_summary,
            model_update=model_update,
            run_id=run_id,
        )

        elapsed = (datetime.now(tz=timezone.utc) - loop_start).total_seconds()
        logger.info(f"=== Learning Loop Complete in {elapsed:.1f}s ===")
        logger.info(f"Model update: {model_update.get('status')} "
                    f"(promoted={model_update.get('promoted')})")
        logger.info(f"Smoke test: {smoke_test.get('status')}")

        if smoke_test.get("status") == "error":
            run_state = "failed"
    except Exception:
        run_state = "failed"
        raise
    finally:
        if store is not None and store.enabled and not args.dry_run:
            store.finish_learning_run(run_id, state=run_state)
            store.close()

    if run_state == "failed":
        logger.error("Benchmark smoke test failed — see logs above")
        sys.exit(1)


def _run_model_update_step(store_df, store, run_id, args) -> dict:
    """Run the registry-backed candidate-vs-active model update."""
    if store_df.empty:
        return {"status": "skipped", "reason": "no_price_data"}
    try:
        import warnings

        from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
        from aurelius.learning.promotion import run_model_update
        from aurelius.storage import get_artifact_store

        warnings.filterwarnings("ignore", category=UserWarning)
        return run_model_update(
            price_df=store_df,
            regions=REGIONS,
            forecaster_cls=PriceQuantileForecaster,
            forecaster_config=PriceModelConfig(seed=42, n_estimators=200, num_leaves=63),
            store=store,
            artifact_store=get_artifact_store(),
            eval_days=EVAL_WINDOW_DAYS,
            scope=args.customer_id,
            pilot_id=args.pilot_id,
            run_id=run_id,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.error(f"Model update failed: {exc}")
        logger.debug(traceback.format_exc())
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    main()
