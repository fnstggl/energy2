#!/usr/bin/env python3
"""Daily retraining script for Aurelius price and carbon quantile forecasters.

Performs a leakage-free temporal split, trains a candidate model on the
training window, evaluates on a held-out validation window, and promotes
the candidate only if it beats the currently active model's holdout metrics.

Usage:
    python scripts/retrain_forecaster.py [options]

    # Retrain price model using CSV fixture data
    python scripts/retrain_forecaster.py --model-type price \\
        --data-csv data/prices.csv --holdout-days 14

    # Retrain carbon model with custom store
    python scripts/retrain_forecaster.py --model-type carbon \\
        --data-csv data/carbon.csv --store-root /var/aurelius/models

    # Dry-run: evaluate but do not promote
    python scripts/retrain_forecaster.py --model-type price \\
        --data-csv data/prices.csv --dry-run

Options:
    --model-type      price | carbon  (required)
    --data-csv        Path to CSV file with historical data (required if no --data-jsonl)
    --data-jsonl      Path to JSONL file with serialised EnergyPrice / CarbonIntensity records
    --store-root      Path to model store root directory
    --holdout-days    Number of most-recent days to use as holdout (default: 14)
    --min-train-days  Minimum days of training data required (default: 30)
    --primary-metric  Metric for model comparison: mape | rmse | mae (default: mape)
    --min-improvement Minimum % improvement required to promote (default: 1.0)
    --dry-run         Evaluate but do not save or promote anything
    --seed            Random seed (default: 42)
    --log-level       Logging level: DEBUG | INFO | WARNING (default: INFO)

Exit codes:
    0  Success (model trained; promoted if improved)
    1  Error (missing data, training failure, etc.)
    2  No improvement (evaluation complete; candidate not promoted)

LEAKAGE GUARANTEE:
    Training data: timestamps < holdout_start
    Holdout data:  timestamps >= holdout_start
    The temporal split is enforced before any model object is created.
    Holdout data is never passed to .fit().
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Ensure the package root is on the path when run as a script
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aurelius.models import EnergyPrice, CarbonIntensity
from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
from aurelius.forecasting.carbon_model import CarbonQuantileForecaster, CarbonModelConfig
from aurelius.ml.forecast_evaluator import ForecastEvaluator, compare_models
from aurelius.ml.model_store import ModelStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_price_csv(path: Path) -> list[EnergyPrice]:
    """Load EnergyPrice records from a CSV file.

    Expected columns (case-insensitive):
        timestamp, region, price_per_mwh

    The timestamp column must be ISO-8601 parseable.
    Currency and source columns are optional and ignored.
    """
    records: list[EnergyPrice] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")
        headers = {h.strip().lower(): h for h in reader.fieldnames}
        _require_cols(headers, {"timestamp", "region", "price_per_mwh"}, path)

        for lineno, row in enumerate(reader, start=2):
            try:
                ts = _parse_timestamp(row[headers["timestamp"]].strip())
                region = row[headers["region"]].strip()
                price = float(row[headers["price_per_mwh"]].strip())
                records.append(EnergyPrice(timestamp=ts, region=region, price_per_mwh=price))
            except (ValueError, KeyError) as exc:
                logger.warning(f"{path}:{lineno}: skipping malformed row: {exc}")

    logger.info(f"Loaded {len(records)} price records from {path}")
    return records


def load_carbon_csv(path: Path) -> list[CarbonIntensity]:
    """Load CarbonIntensity records from a CSV file.

    Expected columns (case-insensitive):
        timestamp, region, gco2_per_kwh
    """
    records: list[CarbonIntensity] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")
        headers = {h.strip().lower(): h for h in reader.fieldnames}
        _require_cols(headers, {"timestamp", "region", "gco2_per_kwh"}, path)

        for lineno, row in enumerate(reader, start=2):
            try:
                ts = _parse_timestamp(row[headers["timestamp"]].strip())
                region = row[headers["region"]].strip()
                gco2 = float(row[headers["gco2_per_kwh"]].strip())
                records.append(CarbonIntensity(timestamp=ts, region=region, gco2_per_kwh=gco2))
            except (ValueError, KeyError) as exc:
                logger.warning(f"{path}:{lineno}: skipping malformed row: {exc}")

    logger.info(f"Loaded {len(records)} carbon records from {path}")
    return records


def load_price_jsonl(path: Path) -> list[EnergyPrice]:
    """Load EnergyPrice records from a JSONL file."""
    records: list[EnergyPrice] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = _parse_timestamp(obj["timestamp"])
                records.append(EnergyPrice(
                    timestamp=ts,
                    region=obj["region"],
                    price_per_mwh=float(obj["price_per_mwh"]),
                ))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.warning(f"{path}:{lineno}: skipping malformed record: {exc}")
    logger.info(f"Loaded {len(records)} price records from {path}")
    return records


def load_carbon_jsonl(path: Path) -> list[CarbonIntensity]:
    """Load CarbonIntensity records from a JSONL file."""
    records: list[CarbonIntensity] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = _parse_timestamp(obj["timestamp"])
                records.append(CarbonIntensity(
                    timestamp=ts,
                    region=obj["region"],
                    gco2_per_kwh=float(obj["gco2_per_kwh"]),
                ))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.warning(f"{path}:{lineno}: skipping malformed record: {exc}")
    logger.info(f"Loaded {len(records)} carbon records from {path}")
    return records


# ---------------------------------------------------------------------------
# Temporal split (leakage-free)
# ---------------------------------------------------------------------------

def temporal_split(
    records: list,
    holdout_days: int,
) -> tuple[list, list, datetime]:
    """Split records into train and holdout sets with a strict temporal boundary.

    INVARIANT: max(train_timestamps) < min(holdout_timestamps)

    Args:
        records: list[EnergyPrice] or list[CarbonIntensity], sorted by timestamp.
        holdout_days: Number of most-recent days to hold out.

    Returns:
        (train_records, holdout_records, holdout_start) where all timestamps
        in holdout_records >= holdout_start and all in train_records < holdout_start.

    Raises:
        ValueError: If there are no records, or the split leaves either
                    split empty.
    """
    if not records:
        raise ValueError("Cannot split empty records list")

    records_sorted = sorted(records, key=lambda r: r.timestamp)
    max_ts = records_sorted[-1].timestamp

    # holdout_start = max_ts - holdout_days, rounded down to midnight UTC
    if max_ts.tzinfo is None:
        max_ts_utc = max_ts.replace(tzinfo=timezone.utc)
    else:
        max_ts_utc = max_ts.astimezone(timezone.utc)

    holdout_start_utc = (max_ts_utc - timedelta(days=holdout_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Normalize record timestamps for comparison
    def _ts_utc(r):
        ts = r.timestamp
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    train = [r for r in records_sorted if _ts_utc(r) < holdout_start_utc]
    holdout = [r for r in records_sorted if _ts_utc(r) >= holdout_start_utc]

    if not train:
        raise ValueError(
            f"No training data before holdout_start={holdout_start_utc.isoformat()}. "
            "Increase data history or decrease --holdout-days."
        )
    if not holdout:
        raise ValueError(
            f"No holdout data on or after holdout_start={holdout_start_utc.isoformat()}. "
            "Decrease --holdout-days or provide more recent data."
        )

    # Enforce hard leakage invariant
    train_max_utc = max(_ts_utc(r) for r in train)
    holdout_min_utc = min(_ts_utc(r) for r in holdout)
    assert train_max_utc < holdout_min_utc, (
        f"LEAKAGE BUG: train_max={train_max_utc} >= holdout_min={holdout_min_utc}"
    )

    return train, holdout, holdout_start_utc


# ---------------------------------------------------------------------------
# Main retraining pipeline
# ---------------------------------------------------------------------------

def retrain(
    model_type: str,
    records: list,
    store: ModelStore,
    holdout_days: int = 14,
    min_train_days: int = 30,
    primary_metric: str = "mape",
    min_improvement_pct: float = 1.0,
    seed: int = 42,
    dry_run: bool = False,
) -> int:
    """Run one retraining cycle.

    Returns:
        0  — success, model promoted
        2  — no improvement (candidate not promoted)
        1  — error (raised as exception upstream; this value reserved)
    """
    logger.info(f"=== Aurelius Retraining: model_type={model_type} ===")

    # --- temporal split (leakage-free) ---
    train_records, holdout_records, holdout_start = temporal_split(records, holdout_days)
    train_days_actual = (
        holdout_start - min(_ts_utc(r) for r in train_records)
    ).days

    logger.info(
        f"Split: train={len(train_records)} records ({train_days_actual}d), "
        f"holdout={len(holdout_records)} records ({holdout_days}d), "
        f"holdout_start={holdout_start.isoformat()}"
    )

    if train_days_actual < min_train_days:
        raise ValueError(
            f"Only {train_days_actual} days of training data; "
            f"need at least {min_train_days}. Provide more history."
        )

    # --- training mean for savings_lift (from train split only) ---
    if model_type == "price":
        training_mean = sum(r.price_per_mwh for r in train_records) / len(train_records)
    else:
        training_mean = sum(r.gco2_per_kwh for r in train_records) / len(train_records)

    # --- train candidate model ---
    logger.info("Training candidate model...")
    if model_type == "price":
        config = PriceModelConfig(seed=seed)
        candidate = PriceQuantileForecaster(config)
        candidate.fit(train_records)
    else:
        config = CarbonModelConfig(seed=seed)
        candidate = CarbonQuantileForecaster(config)
        candidate.fit(train_records)

    # --- evaluate candidate on holdout ---
    logger.info("Evaluating candidate on holdout...")
    evaluator = ForecastEvaluator()

    # Pass the last 48 hours of train as recent context for lag features
    context_records = sorted(train_records, key=lambda r: r.timestamp)[-48:]
    candidate_result = evaluator.evaluate_from_model(
        forecaster=candidate,
        holdout_actuals=holdout_records,
        training_mean=training_mean,
        recent_context=context_records,
    )

    logger.info(
        f"Candidate holdout metrics: "
        f"MAPE={candidate_result.mape:.4f} "
        f"RMSE={candidate_result.rmse:.4f} "
        f"MAE={candidate_result.mae:.4f} "
        f"p90_coverage={candidate_result.p90_coverage:.4f} "
        f"cal_error={candidate_result.calibration_error:.4f} "
        f"savings_lift={candidate_result.savings_lift * 100:.2f}%"
    )

    # --- compare vs active model (if any) ---
    if not dry_run and store.has_active(model_type):
        logger.info("Loading active model for comparison...")
        try:
            active_forecaster = store.load_active(model_type)
            active_result = evaluator.evaluate_from_model(
                forecaster=active_forecaster,
                holdout_actuals=holdout_records,
                training_mean=training_mean,
                recent_context=context_records,
            )
            logger.info(
                f"Active model holdout metrics: "
                f"MAPE={active_result.mape:.4f} "
                f"RMSE={active_result.rmse:.4f} "
                f"MAE={active_result.mae:.4f} "
                f"p90_coverage={active_result.p90_coverage:.4f}"
            )

            comparison = compare_models(
                candidate=candidate_result,
                current=active_result,
                primary_metric=primary_metric,
                min_improvement_pct=min_improvement_pct,
            )
            logger.info(f"Model comparison: {comparison.reason}")

            if not comparison.promote:
                logger.info("Candidate NOT promoted. Active model retained.")
                return 2

        except Exception as exc:
            logger.warning(
                f"Could not evaluate active model ({exc}); "
                "proceeding with candidate promotion (first-run or corrupt active)"
            )

    if dry_run:
        logger.info("[dry-run] Skipping save and promote")
        return 0

    # --- save and promote candidate ---
    metadata = {
        "holdout_days": holdout_days,
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "holdout_start": holdout_start.isoformat(),
        "training_mean": training_mean,
        "eval_metrics": candidate_result.to_dict(),
        "primary_metric": primary_metric,
    }
    version_id = store.save(candidate, model_type=model_type, metadata=metadata)
    store.promote(model_type=model_type, version_id=version_id)
    logger.info(f"Promoted {model_type}/{version_id} to active.")
    return 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_cols(headers: dict, required: set, path: Path) -> None:
    missing = required - set(headers.keys())
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")


def _parse_timestamp(s: str) -> datetime:
    """Parse ISO-8601 timestamp string to timezone-aware datetime (UTC)."""
    s = s.strip()
    # Handle common formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {s!r}")


def _ts_utc(record) -> datetime:
    ts = record.timestamp
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aurelius daily forecaster retraining script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["price", "carbon"],
        help="Which model to retrain",
    )
    parser.add_argument("--data-csv", type=Path, help="Path to input CSV file")
    parser.add_argument("--data-jsonl", type=Path, help="Path to input JSONL file")
    parser.add_argument("--store-root", type=Path, default=None, help="Model store root")
    parser.add_argument("--holdout-days", type=int, default=14, help="Days to hold out")
    parser.add_argument("--min-train-days", type=int, default=30, help="Min training days")
    parser.add_argument(
        "--primary-metric", default="mape", choices=["mape", "rmse", "mae"],
        help="Metric for model comparison"
    )
    parser.add_argument(
        "--min-improvement", type=float, default=1.0,
        help="Min %% improvement required to promote (default: 1.0)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but do not promote")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not args.data_csv and not args.data_jsonl:
        logger.error("Must provide --data-csv or --data-jsonl")
        return 1

    # Load records
    try:
        if args.data_csv:
            if args.model_type == "price":
                records = load_price_csv(args.data_csv)
            else:
                records = load_carbon_csv(args.data_csv)
        else:
            if args.model_type == "price":
                records = load_price_jsonl(args.data_jsonl)
            else:
                records = load_carbon_jsonl(args.data_jsonl)
    except Exception as exc:
        logger.error(f"Failed to load data: {exc}")
        return 1

    if not records:
        logger.error("No records loaded; cannot retrain")
        return 1

    store = ModelStore(store_root=args.store_root)

    try:
        exit_code = retrain(
            model_type=args.model_type,
            records=records,
            store=store,
            holdout_days=args.holdout_days,
            min_train_days=args.min_train_days,
            primary_metric=args.primary_metric,
            min_improvement_pct=args.min_improvement,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        logger.error(f"Retraining failed: {exc}")
        return 1
    except Exception as exc:
        logger.exception(f"Unexpected error during retraining: {exc}")
        return 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
