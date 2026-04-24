"""Walk-forward backtesting engine.

The engine runs a full backtest by stepping through historical data with
strict train/eval separation. The optimizer is trained on data *before* each
evaluation window; it is never shown eval-window actuals.

Usage:
    engine = BacktestEngine(method="greedy")
    results = engine.run(
        jobs=jobs,
        price_df=price_df,
        carbon_df=carbon_df,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-03-01", tz="UTC"),
    )
    for r in results:
        print(r.fold_index, r.optimizer_metrics.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from aurelius.backtesting.evaluator import RealizedMetrics, evaluate_schedule
from aurelius.backtesting.splitter import TemporalSplit, TemporalSplitter
from aurelius.backtesting.baselines import ALL_BASELINES, BaselinePolicy
from aurelius.models import Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler

logger = logging.getLogger(__name__)


@dataclass
class BacktestRound:
    """Result of a single backtest fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    # Jobs that have their earliest_start inside the eval window
    eval_jobs: list[Job] = field(default_factory=list)

    # Optimizer schedule and metrics
    optimizer_schedule: list[ScheduleDecision] = field(default_factory=list)
    optimizer_metrics: Optional[RealizedMetrics] = None

    # Baseline schedules and metrics keyed by policy name
    baseline_schedules: dict[str, list[ScheduleDecision]] = field(default_factory=dict)
    baseline_metrics: dict[str, RealizedMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        baselines = {
            name: m.to_dict()
            for name, m in self.baseline_metrics.items()
        }
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "n_eval_jobs": len(self.eval_jobs),
            "optimizer": self.optimizer_metrics.to_dict() if self.optimizer_metrics else {},
            "baselines": baselines,
        }


def _df_to_price_data(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> dict[str, dict[datetime, float]]:
    """Convert a canonical price DataFrame to the {region: {ts: price}} dict."""
    result: dict[str, dict[datetime, float]] = {}
    for _, row in df.iterrows():
        region = row["region"]
        ts = row[ts_col].to_pydatetime() if hasattr(row[ts_col], "to_pydatetime") else row[ts_col]
        ts = ts.replace(minute=0, second=0, microsecond=0)
        result.setdefault(region, {})[ts] = float(row["price_per_mwh"])
    return result


def _df_to_carbon_data(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> dict[str, dict[datetime, float]]:
    """Convert a canonical carbon DataFrame to the {region: {ts: gco2}} dict."""
    result: dict[str, dict[datetime, float]] = {}
    for _, row in df.iterrows():
        region = row["region"]
        ts = row[ts_col].to_pydatetime() if hasattr(row[ts_col], "to_pydatetime") else row[ts_col]
        ts = ts.replace(minute=0, second=0, microsecond=0)
        result.setdefault(region, {})[ts] = float(row["gco2_per_kwh"])
    return result


class BacktestEngine:
    """Walk-forward backtesting engine.

    For each fold the engine:
      1. Slices training data (strictly before eval window).
      2. Determines which jobs fall in the eval window.
      3. Runs the optimizer using *only* training-window data as signal.
      4. Evaluates the optimizer schedule against *actual* eval-window data.
      5. Runs all baseline policies and evaluates them the same way.

    The optimizer never sees eval-window price or carbon data.

    Args:
        method:      Optimizer method ("greedy", "local_search", "milp").
        train_days:  Training window length in days.
        eval_days:   Evaluation window length in days.
        step_days:   Step between folds (default = eval_days).
        config:      OptimizationConfig (uses defaults if None).
        baselines:   Which baseline policy names to run alongside optimizer.
                     Defaults to all 7 policies.
    """

    def __init__(
        self,
        method: str = "greedy",
        train_days: int = 30,
        eval_days: int = 7,
        step_days: int = 0,
        config: Optional[OptimizationConfig] = None,
        baselines: Optional[list[str]] = None,
    ) -> None:
        self.method = method
        self.config = config or OptimizationConfig()
        self.splitter = TemporalSplitter(
            train_days=train_days,
            eval_days=eval_days,
            step_days=step_days,
        )
        self.scheduler = JobScheduler(self.config)
        self.baseline_names = baselines if baselines is not None else list(ALL_BASELINES.keys())

    def run(
        self,
        jobs: list[Job],
        price_df: pd.DataFrame,
        carbon_df: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> list[BacktestRound]:
        """Run the full walk-forward backtest.

        Args:
            jobs:      All jobs. The engine assigns jobs to folds by
                       earliest_start timestamp.
            price_df:  Canonical price DataFrame (columns: timestamp, region, price_per_mwh).
            carbon_df: Canonical carbon DataFrame (columns: timestamp, region, gco2_per_kwh).
            start:     Backtest start timestamp (defaults to min timestamp in price_df).
            end:       Backtest end timestamp (defaults to max timestamp in price_df + 1h).

        Returns:
            List of BacktestRound objects, one per fold.
        """
        if price_df.empty:
            logger.warning("BacktestEngine.run: price_df is empty; returning no folds")
            return []

        splits = self.splitter.split(price_df, start=start, end=end)
        if not splits:
            logger.warning("BacktestEngine.run: no valid splits produced")
            return []

        rounds: list[BacktestRound] = []

        for split in splits:
            round_ = self._run_fold(split, jobs, price_df, carbon_df)
            if round_ is not None:
                rounds.append(round_)

        logger.info(f"BacktestEngine finished: {len(rounds)} folds evaluated")
        return rounds

    def _run_fold(
        self,
        split: TemporalSplit,
        all_jobs: list[Job],
        price_df: pd.DataFrame,
        carbon_df: pd.DataFrame,
    ) -> Optional[BacktestRound]:
        """Execute a single fold."""
        # Jobs whose earliest_start falls in the eval window
        eval_jobs = [
            j for j in all_jobs
            if split.eval_start <= _to_ts(j.earliest_start) < split.eval_end
        ]

        if not eval_jobs:
            logger.debug(f"Fold {split.fold_index}: no jobs in eval window, skipping")
            return None

        # Build signal dicts from TRAINING data only (leakage guard)
        train_price_data = _df_to_price_data(split.train_df)
        train_carbon_data: dict[str, dict[datetime, float]] = {}
        if not carbon_df.empty:
            ts_col = "timestamp"
            carbon_mask = (
                (pd.to_datetime(carbon_df[ts_col]) >= split.train_start)
                & (pd.to_datetime(carbon_df[ts_col]) < split.train_end)
            )
            train_carbon_df = carbon_df[carbon_mask]
            train_carbon_data = _df_to_carbon_data(train_carbon_df)

        # Actual eval-window data for scoring
        eval_price_mask = (
            (pd.to_datetime(price_df["timestamp"]) >= split.eval_start)
            & (pd.to_datetime(price_df["timestamp"]) < split.eval_end)
        )
        eval_price_data = _df_to_price_data(price_df[eval_price_mask])

        eval_carbon_data: dict[str, dict[datetime, float]] = {}
        if not carbon_df.empty:
            ts_col = "timestamp"
            eval_carbon_mask = (
                (pd.to_datetime(carbon_df[ts_col]) >= split.eval_start)
                & (pd.to_datetime(carbon_df[ts_col]) < split.eval_end)
            )
            eval_carbon_data = _df_to_carbon_data(carbon_df[eval_carbon_mask])

        # Run optimizer (sees only training signal)
        try:
            opt_result = self.scheduler.solve(
                eval_jobs,
                train_price_data,
                train_carbon_data,
                method=self.method,
            )
            opt_schedule = opt_result.schedule
        except Exception as exc:
            logger.error(f"Fold {split.fold_index}: optimizer failed: {exc}")
            opt_schedule = []

        # Evaluate optimizer against actuals
        opt_metrics = evaluate_schedule(
            opt_schedule, eval_jobs, eval_price_data, eval_carbon_data
        )

        # Run and evaluate all baseline policies
        baseline_schedules: dict[str, list[ScheduleDecision]] = {}
        baseline_metrics: dict[str, RealizedMetrics] = {}

        for name in self.baseline_names:
            policy = ALL_BASELINES.get(name)
            if policy is None:
                logger.warning(f"Unknown baseline policy '{name}', skipping")
                continue
            try:
                bl_schedule = policy(eval_jobs, train_price_data, train_carbon_data, self.config)
                bl_metrics = evaluate_schedule(
                    bl_schedule, eval_jobs, eval_price_data, eval_carbon_data
                )
                baseline_schedules[name] = bl_schedule
                baseline_metrics[name] = bl_metrics
            except Exception as exc:
                logger.error(f"Fold {split.fold_index}, baseline '{name}': {exc}")

        return BacktestRound(
            fold_index=split.fold_index,
            train_start=split.train_start,
            train_end=split.train_end,
            eval_start=split.eval_start,
            eval_end=split.eval_end,
            eval_jobs=eval_jobs,
            optimizer_schedule=opt_schedule,
            optimizer_metrics=opt_metrics,
            baseline_schedules=baseline_schedules,
            baseline_metrics=baseline_metrics,
        )


def _to_ts(dt: datetime) -> pd.Timestamp:
    """Convert a datetime to a UTC-aware pd.Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts
