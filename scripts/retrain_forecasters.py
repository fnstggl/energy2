#!/usr/bin/env python3
"""Daily retraining script for Aurelius price AND carbon quantile forecasters.

Runs a leakage-free temporal split over the specified date window, trains
candidate price and carbon models on the training portion, evaluates each on
the held-out validation portion, and promotes candidates only when they beat
the currently active model on the primary metric.

Usage:
    python -m aurelius.scripts.retrain_forecasters \\
        --start 2023-01-01 --end 2024-01-01

    python scripts/retrain_forecasters.py \\
        --start 2023-06-01 --end 2024-01-01 \\
        --data-csv-price data/prices.csv \\
        --data-csv-carbon data/carbon.csv \\
        --store-root /var/aurelius/models \\
        --holdout-days 30 \\
        --dry-run

Options:
    --start             ISO date for the start of the data window (required)
    --end               ISO date for the end of the data window (required)
    --data-csv-price    Path to price CSV (timestamp,region,price_per_mwh)
    --data-csv-carbon   Path to carbon CSV (timestamp,region,gco2_per_kwh)
    --store-root        Path to model store root directory
    --holdout-days      Days to reserve as holdout (default: 30)
    --min-train-days    Minimum training days required (default: 60)
    --primary-metric    Metric for model comparison: mape | rmse (default: mape)
    --min-improvement   Minimum %% improvement required to promote (default: 1.0)
    --dry-run           Evaluate but do not save or promote
    --seed              Random seed (default: 42)
    --log-level         DEBUG | INFO | WARNING (default: INFO)

Exit codes:
    0  Success (models trained; promoted if improved)
    1  Error
    2  No improvement (evaluation complete; candidates not promoted)

LEAKAGE GUARANTEE:
    Training data:  timestamps < holdout_start
    Holdout data:   timestamps >= holdout_start
    The temporal split is enforced before any model is created.
    Holdout data is never passed to .fit().
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Ensure package root is on path when run as script
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aurelius.forecasting.carbon_model import CarbonModelConfig, CarbonQuantileForecaster
from aurelius.forecasting.price_model import PriceModelConfig, PriceQuantileForecaster
from aurelius.ml.forecast_evaluator import EvaluationResult, ForecastEvaluator, compare_models
from aurelius.ml.model_store import ModelStore
from aurelius.models import CarbonIntensity, EnergyPrice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_price_csv(path: Path) -> list[EnergyPrice]:
    """Load EnergyPrice records from a CSV file.

    Expected columns (case-insensitive): timestamp, region, price_per_mwh
    Optional: currency, source
    """
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        ts_col = cols.get("timestamp", cols.get("datetime", cols.get("time")))
        region_col = cols.get("region", cols.get("zone", cols.get("location")))
        price_col = cols.get("price_per_mwh", cols.get("price", cols.get("lmp")))

        if not all([ts_col, region_col, price_col]):
            raise ValueError(
                f"Price CSV {path} must have timestamp, region, price_per_mwh columns. "
                f"Found: {list(cols.keys())}"
            )

        for row in reader:
            try:
                ts_str = row[ts_col].strip()
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                records.append(EnergyPrice(
                    timestamp=ts,
                    region=row[region_col].strip(),
                    price_per_mwh=float(row[price_col]),
                ))
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed price CSV row: {e}")
    logger.info(f"Loaded {len(records)} price records from {path}")
    return records


def load_carbon_csv(path: Path) -> list[CarbonIntensity]:
    """Load CarbonIntensity records from a CSV file.

    Expected columns (case-insensitive): timestamp, region, gco2_per_kwh
    """
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        ts_col = cols.get("timestamp", cols.get("datetime", cols.get("time")))
        region_col = cols.get("region", cols.get("zone"))
        carbon_col = cols.get("gco2_per_kwh", cols.get("carbon", cols.get("intensity")))

        if not all([ts_col, region_col, carbon_col]):
            raise ValueError(
                f"Carbon CSV {path} must have timestamp, region, gco2_per_kwh columns. "
                f"Found: {list(cols.keys())}"
            )

        for row in reader:
            try:
                ts_str = row[ts_col].strip()
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                records.append(CarbonIntensity(
                    timestamp=ts,
                    region=row[region_col].strip(),
                    gco2_per_kwh=float(row[carbon_col]),
                ))
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed carbon CSV row: {e}")
    logger.info(f"Loaded {len(records)} carbon records from {path}")
    return records


def generate_synthetic_price_data(
    start: datetime,
    end: datetime,
    regions: list[str],
    seed: int = 42,
) -> list[EnergyPrice]:
    """Generate synthetic price data for testing when no CSV is provided."""
    import random
    rng = random.Random(seed)
    records = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        for region in regions:
            base = {"us-east": 55.0, "us-west": 45.0, "pjm": 50.0}.get(region, 50.0)
            hour_factor = 1.0 + 0.3 * abs((current.hour - 14) / 14 - 0.5)
            price = base * hour_factor * (1 + rng.gauss(0, 0.05))
            records.append(EnergyPrice(
                timestamp=current,
                region=region,
                price_per_mwh=max(0.1, price),
            ))
        current += timedelta(hours=1)
    return records


def generate_synthetic_carbon_data(
    start: datetime,
    end: datetime,
    regions: list[str],
    seed: int = 42,
) -> list[CarbonIntensity]:
    """Generate synthetic carbon data for testing."""
    import random
    rng = random.Random(seed + 1)
    records = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        for region in regions:
            base = {"us-east": 450.0, "us-west": 300.0, "pjm": 400.0}.get(region, 380.0)
            solar_factor = max(0.0, 1 - 0.2 * max(0, 8 - abs(current.hour - 12)))
            carbon = base * (1 - 0.15 * solar_factor) * (1 + rng.gauss(0, 0.03))
            records.append(CarbonIntensity(
                timestamp=current,
                region=region,
                gco2_per_kwh=max(10.0, carbon),
            ))
        current += timedelta(hours=1)
    return records


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    records: list,
    holdout_start: datetime,
) -> tuple[list, list]:
    """Split records into train (< holdout_start) and holdout (>= holdout_start).

    LEAKAGE GUARANTEE: every holdout timestamp > every training timestamp.
    """
    train = [r for r in records if r.timestamp < holdout_start]
    holdout = [r for r in records if r.timestamp >= holdout_start]
    return train, holdout


def _dataset_hash(records: list) -> str:
    """Compute a short SHA-256 hash of a record list for reproducibility logging."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: (str(x.timestamp), x.region)):
        h.update(f"{r.timestamp.isoformat()},{r.region}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_price_model(
    model: PriceQuantileForecaster,
    holdout: list[EnergyPrice],
    evaluator: ForecastEvaluator,
) -> Optional[EvaluationResult]:
    """Run holdout evaluation on a price model. Returns EvaluationResult or None."""
    if not holdout:
        return None
    try:
        return evaluator.evaluate_from_model(model, holdout)
    except Exception as e:
        logger.error(f"Price model evaluation failed: {e}")
        return None


def _evaluate_carbon_model(
    model: CarbonQuantileForecaster,
    holdout: list[CarbonIntensity],
    evaluator: ForecastEvaluator,
) -> Optional[EvaluationResult]:
    """Run holdout evaluation on a carbon model. Returns EvaluationResult or None."""
    if not holdout:
        return None
    try:
        return evaluator.evaluate_from_model(model, holdout)
    except Exception as e:
        logger.error(f"Carbon model evaluation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main retraining logic
# ---------------------------------------------------------------------------

def retrain_forecasters(
    start: datetime,
    end: datetime,
    price_records: list[EnergyPrice],
    carbon_records: list[CarbonIntensity],
    store: ModelStore,
    holdout_days: int = 30,
    min_train_days: int = 60,
    primary_metric: str = "mape",
    min_improvement_pct: float = 1.0,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """Run the full retraining pipeline for price and carbon forecasters.

    Args:
        start: Start of data window.
        end: End of data window.
        price_records: Historical price records within [start, end].
        carbon_records: Historical carbon records within [start, end].
        store: ModelStore for saving and promoting models.
        holdout_days: Days at the end of the window to use as holdout.
        min_train_days: Minimum training days required.
        primary_metric: Metric for promotion decision (mape | rmse).
        min_improvement_pct: Minimum % improvement to promote.
        dry_run: If True, evaluate but do not promote.
        seed: Random seed.

    Returns:
        Summary dict with metrics, promotion decisions, and dataset hashes.
    """
    holdout_start = end - timedelta(days=holdout_days)

    if (holdout_start - start).days < min_train_days:
        raise ValueError(
            f"Insufficient training window: need {min_train_days} days before holdout, "
            f"got {(holdout_start - start).days} days. "
            f"Extend --start or reduce --holdout-days."
        )

    # Temporal split (leakage-free by construction)
    price_train, price_holdout = temporal_split(price_records, holdout_start)
    carbon_train, carbon_holdout = temporal_split(carbon_records, holdout_start)

    # Verify no leakage (defensive assertion)
    if price_train and price_holdout:
        max_train_ts = max(r.timestamp for r in price_train)
        min_holdout_ts = min(r.timestamp for r in price_holdout)
        assert max_train_ts < min_holdout_ts, (
            f"DATA LEAKAGE: max train ts {max_train_ts} >= min holdout ts {min_holdout_ts}"
        )

    dataset_hash_price = _dataset_hash(price_records)
    dataset_hash_carbon = _dataset_hash(carbon_records)

    logger.info(f"Price:  {len(price_train)} train, {len(price_holdout)} holdout records")
    logger.info(f"Carbon: {len(carbon_train)} train, {len(carbon_holdout)} holdout records")
    logger.info(f"Dataset hashes — price:{dataset_hash_price}, carbon:{dataset_hash_carbon}")

    evaluator = ForecastEvaluator()
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_days": holdout_days,
        "dry_run": dry_run,
        "dataset_hash_price": dataset_hash_price,
        "dataset_hash_carbon": dataset_hash_carbon,
        "price": {},
        "carbon": {},
    }

    # ------------------------------------------------------------------
    # Price model
    # ------------------------------------------------------------------
    if len(price_train) >= 24:
        logger.info("\n--- Training price model ---")
        config = PriceModelConfig(seed=seed)
        candidate_price = PriceQuantileForecaster(config, corrections_path=False)
        candidate_price.fit(price_train)

        price_metrics = _evaluate_price_model(candidate_price, price_holdout, evaluator)
        coverage_result = candidate_price.validate_coverage(price_holdout) if price_holdout else {}

        if price_metrics:
            logger.info(
                f"Price candidate — MAPE: {price_metrics.mape:.4f}, "
                f"RMSE: {price_metrics.rmse:.4f}, "
                f"p90_coverage: {coverage_result.get('empirical_p90_coverage', 'n/a')}"
            )

        summary["price"] = {
            "n_train": len(price_train),
            "n_holdout": len(price_holdout),
            "candidate_metrics": price_metrics.to_dict() if price_metrics else None,
            "coverage": coverage_result,
        }

        if not dry_run and price_metrics:
            # Check if candidate beats active model
            active_metrics = None
            if store.has_active("price"):
                try:
                    active_model = store.load_active("price")
                    active_metrics = _evaluate_price_model(active_model, price_holdout, evaluator)
                except Exception as e:
                    logger.warning(f"Could not load active price model: {e}")

            promote = False
            if active_metrics is None:
                promote = True
                logger.info("No active price model — promoting candidate")
            else:
                comparison = compare_models(
                    price_metrics, active_metrics, primary_metric, min_improvement_pct
                )
                promote = comparison.promote
                logger.info(
                    f"Price improvement vs active: {comparison.improvement_pct:.2f}% "
                    f"(need {min_improvement_pct}%) → {'PROMOTE' if promote else 'SKIP'}"
                )

            if promote:
                version_id = store.save(
                    candidate_price, "price",
                    metadata={
                        "mape": price_metrics.mape,
                        "rmse": price_metrics.rmse,
                        "p90_coverage": coverage_result.get("empirical_p90_coverage"),
                        "dataset_hash": dataset_hash_price,
                        "n_train": len(price_train),
                        "holdout_start": holdout_start.isoformat(),
                    }
                )
                store.promote("price", version_id)
                summary["price"]["promoted"] = True
                summary["price"]["version_id"] = version_id
                logger.info(f"Price model promoted as version {version_id}")
            else:
                summary["price"]["promoted"] = False
    else:
        logger.warning(f"Skipping price model: only {len(price_train)} training records (need 24+)")
        summary["price"]["skipped"] = "insufficient_data"

    # ------------------------------------------------------------------
    # Carbon model
    # ------------------------------------------------------------------
    if len(carbon_train) >= 24:
        logger.info("\n--- Training carbon model ---")
        config = CarbonModelConfig(seed=seed)
        candidate_carbon = CarbonQuantileForecaster(config, corrections_path=False)
        candidate_carbon.fit(carbon_train)

        carbon_metrics = _evaluate_carbon_model(candidate_carbon, carbon_holdout, evaluator)
        coverage_result = candidate_carbon.validate_coverage(carbon_holdout) if carbon_holdout else {}

        if carbon_metrics:
            logger.info(
                f"Carbon candidate — MAPE: {carbon_metrics.mape:.4f}, "
                f"RMSE: {carbon_metrics.rmse:.4f}, "
                f"p90_coverage: {coverage_result.get('empirical_p90_coverage', 'n/a')}"
            )

        summary["carbon"] = {
            "n_train": len(carbon_train),
            "n_holdout": len(carbon_holdout),
            "candidate_metrics": carbon_metrics.to_dict() if carbon_metrics else None,
            "coverage": coverage_result,
        }

        if not dry_run and carbon_metrics:
            active_metrics = None
            if store.has_active("carbon"):
                try:
                    active_model = store.load_active("carbon")
                    active_metrics = _evaluate_carbon_model(active_model, carbon_holdout, evaluator)
                except Exception as e:
                    logger.warning(f"Could not load active carbon model: {e}")

            promote = False
            if active_metrics is None:
                promote = True
                logger.info("No active carbon model — promoting candidate")
            else:
                comparison = compare_models(
                    carbon_metrics, active_metrics, primary_metric, min_improvement_pct
                )
                promote = comparison.promote
                logger.info(
                    f"Carbon improvement vs active: {comparison.improvement_pct:.2f}% "
                    f"(need {min_improvement_pct}%) → {'PROMOTE' if promote else 'SKIP'}"
                )

            if promote:
                version_id = store.save(
                    candidate_carbon, "carbon",
                    metadata={
                        "mape": carbon_metrics.mape,
                        "rmse": carbon_metrics.rmse,
                        "p90_coverage": coverage_result.get("empirical_p90_coverage"),
                        "dataset_hash": dataset_hash_carbon,
                        "n_train": len(carbon_train),
                        "holdout_start": holdout_start.isoformat(),
                    }
                )
                store.promote("carbon", version_id)
                summary["carbon"]["promoted"] = True
                summary["carbon"]["version_id"] = version_id
                logger.info(f"Carbon model promoted as version {version_id}")
            else:
                summary["carbon"]["promoted"] = False
    else:
        logger.warning(f"Skipping carbon model: only {len(carbon_train)} training records (need 24+)")
        summary["carbon"]["skipped"] = "insufficient_data"

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Date must be YYYY-MM-DD, got: {date_str}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain Aurelius price and carbon quantile forecasters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start", required=True, type=_parse_date,
        help="Start of data window (YYYY-MM-DD, UTC)",
    )
    parser.add_argument(
        "--end", required=True, type=_parse_date,
        help="End of data window (YYYY-MM-DD, UTC)",
    )
    parser.add_argument(
        "--data-csv-price", type=Path, default=None,
        help="Path to price CSV (timestamp, region, price_per_mwh)",
    )
    parser.add_argument(
        "--data-csv-carbon", type=Path, default=None,
        help="Path to carbon CSV (timestamp, region, gco2_per_kwh)",
    )
    parser.add_argument(
        "--store-root", type=Path, default=None,
        help="Model store root directory (default: aurelius/data/model_store)",
    )
    parser.add_argument(
        "--holdout-days", type=int, default=30,
        help="Days to reserve as holdout (default: 30)",
    )
    parser.add_argument(
        "--min-train-days", type=int, default=60,
        help="Minimum training days (default: 60)",
    )
    parser.add_argument(
        "--primary-metric", default="mape", choices=["mape", "rmse"],
        help="Metric for promotion decision (default: mape)",
    )
    parser.add_argument(
        "--min-improvement", type=float, default=1.0,
        help="Minimum %% improvement to promote (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate but do not save or promote models",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--regions", nargs="+", default=["us-east", "us-west"],
        help="Regions for synthetic data generation (default: us-east us-west)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.start >= args.end:
        logger.error("--start must be before --end")
        return 1

    logger.info("=" * 60)
    logger.info("Aurelius Forecaster Retraining")
    logger.info("=" * 60)
    logger.info(f"Window:      {args.start.date()} → {args.end.date()}")
    logger.info(f"Holdout:     last {args.holdout_days} days")
    logger.info(f"Primary metric: {args.primary_metric}")
    logger.info(f"Dry run:     {args.dry_run}")
    logger.info(f"Seed:        {args.seed}")

    # Load data
    try:
        if args.data_csv_price:
            price_records = load_price_csv(args.data_csv_price)
        else:
            logger.warning("No --data-csv-price provided; using synthetic price data")
            price_records = generate_synthetic_price_data(
                args.start, args.end, args.regions, seed=args.seed
            )

        if args.data_csv_carbon:
            carbon_records = load_carbon_csv(args.data_csv_carbon)
        else:
            logger.warning("No --data-csv-carbon provided; using synthetic carbon data")
            carbon_records = generate_synthetic_carbon_data(
                args.start, args.end, args.regions, seed=args.seed
            )
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    # Filter to requested window
    price_records = [
        r for r in price_records if args.start <= r.timestamp < args.end
    ]
    carbon_records = [
        r for r in carbon_records if args.start <= r.timestamp < args.end
    ]

    # Initialise model store
    store = ModelStore(store_root=args.store_root)

    # Run retraining
    try:
        summary = retrain_forecasters(
            start=args.start,
            end=args.end,
            price_records=price_records,
            carbon_records=carbon_records,
            store=store,
            holdout_days=args.holdout_days,
            min_train_days=args.min_train_days,
            primary_metric=args.primary_metric,
            min_improvement_pct=args.min_improvement,
            dry_run=args.dry_run,
            seed=args.seed,
        )
    except ValueError as e:
        logger.error(f"Retraining failed: {e}")
        return 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(json.dumps(summary, indent=2, default=str))

    # Return code: 2 if neither model was promoted
    price_promoted = summary.get("price", {}).get("promoted", False)
    carbon_promoted = summary.get("carbon", {}).get("promoted", False)
    if not args.dry_run and not price_promoted and not carbon_promoted:
        if (
            "skipped" not in summary.get("price", {}) or
            "skipped" not in summary.get("carbon", {})
        ):
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
