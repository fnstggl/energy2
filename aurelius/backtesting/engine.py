"""Walk-forward backtesting engine.

The engine runs a full backtest by stepping through historical data with
strict train/eval separation. The optimizer is trained on data *before* each
evaluation window; it is never shown eval-window actuals.

Forecast modes:
  "seasonal_naive" (default)
    - Uses hour-of-day means from the training window as the price/carbon signal.
    - No ML model required.

  "ml_quantile" (when price_forecaster_cls is supplied)
    - At each fold, fits a fresh PriceQuantileForecaster on training records.
    - Uses p50 predictions as the optimizer signal.
    - Records per-fold forecast quality metrics (MAPE, p90 coverage, calibration).
    - LEAKAGE INVARIANT: forecaster.fit() is called with train-only records.
      recent_context passed to predict() is the last 48 h of training data.
      The forecaster never sees eval-window actuals.

Usage:
    # Naive mode (backward-compatible)
    engine = BacktestEngine(method="greedy")

    # ML mode
    from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
    engine = BacktestEngine(
        method="greedy",
        price_forecaster_cls=PriceQuantileForecaster,
        price_forecaster_config=PriceModelConfig(seed=42, n_estimators=50),
    )
    results = engine.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
    for r in results:
        print(r.forecast_method, r.forecast_metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Type

import pandas as pd

from aurelius.backtesting.evaluator import RealizedMetrics, evaluate_schedule
from aurelius.backtesting.splitter import TemporalSplit, TemporalSplitter
from aurelius.backtesting.baselines import ALL_BASELINES
from aurelius.models import EnergyPrice, CarbonIntensity, Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler

logger = logging.getLogger(__name__)


@dataclass
class ForecastQuality:
    """Per-fold forecast accuracy metrics.

    Populated when the engine is configured with an ML forecaster.
    Measures how well the forecaster predicted eval-window prices/carbon
    (using actuals as ground truth, without contaminating the training set).
    """
    n_eval_hours: int = 0
    mape: float = float("nan")
    rmse: float = float("nan")
    p90_coverage: float = float("nan")
    calibration_error: float = float("nan")
    forecast_method: str = "seasonal_naive"

    def to_dict(self) -> dict:
        import math
        return {
            "forecast_method": self.forecast_method,
            "n_eval_hours": self.n_eval_hours,
            "mape": round(self.mape, 4) if not math.isnan(self.mape) else None,
            "rmse": round(self.rmse, 4) if not math.isnan(self.rmse) else None,
            "p90_coverage": round(self.p90_coverage, 4) if not math.isnan(self.p90_coverage) else None,
            "calibration_error": round(self.calibration_error, 4) if not math.isnan(self.calibration_error) else None,
        }


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

    # Forecast quality for this fold (populated in ML mode)
    forecast_quality: Optional[ForecastQuality] = None

    def to_dict(self) -> dict:
        baselines = {
            name: m.to_dict()
            for name, m in self.baseline_metrics.items()
        }
        result = {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "n_eval_jobs": len(self.eval_jobs),
            "optimizer": self.optimizer_metrics.to_dict() if self.optimizer_metrics else {},
            "baselines": baselines,
        }
        if self.forecast_quality is not None:
            result["forecast_quality"] = self.forecast_quality.to_dict()
        return result


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


def _df_to_price_records(df: pd.DataFrame) -> list[EnergyPrice]:
    """Convert price DataFrame to list[EnergyPrice] for forecaster.fit()."""
    records = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        records.append(EnergyPrice(
            timestamp=ts,
            region=str(row["region"]),
            price_per_mwh=float(row["price_per_mwh"]),
        ))
    return records


def _df_to_carbon_records(df: pd.DataFrame) -> list[CarbonIntensity]:
    """Convert carbon DataFrame to list[CarbonIntensity] for forecaster.fit()."""
    records = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        records.append(CarbonIntensity(
            timestamp=ts,
            region=str(row["region"]),
            gco2_per_kwh=float(row["gco2_per_kwh"]),
        ))
    return records


class BacktestEngine:
    """Walk-forward backtesting engine with optional ML forecaster support.

    For each fold the engine:
      1. Slices training data (strictly before eval window).
      2. Determines which jobs fall in the eval window.
      3a. [naive] Builds hour-of-day mean forecasts from training data.
      3b. [ML]    Fits ML forecaster on training data, predicts eval window.
          Both paths use ONLY training data — eval-window actuals never seen.
      4. Runs the optimizer with these price/carbon signals.
      5. Evaluates optimizer schedule against eval-window actuals.
      6. Runs all baseline policies and evaluates them the same way.
      7. [ML] Records per-fold forecast quality metrics.

    Args:
        method:                    Optimizer method ("greedy", "local_search", "milp").
        train_days:                Training window length in days.
        eval_days:                 Evaluation window length in days.
        step_days:                 Step between folds (default = eval_days).
        config:                    OptimizationConfig (uses defaults if None).
        baselines:                 Baseline policy names to run (all 7 by default).
        price_forecaster_cls:      Class for price forecasting (e.g. PriceQuantileForecaster).
                                   If None, seasonal-naive forecasting is used.
        price_forecaster_config:   Config object passed to price_forecaster_cls().
        carbon_forecaster_cls:     Class for carbon forecasting (e.g. CarbonQuantileForecaster).
        carbon_forecaster_config:  Config object passed to carbon_forecaster_cls().
        context_hours:             Hours of training tail to use as lag-feature context (default 48).
    """

    def __init__(
        self,
        method: str = "greedy",
        train_days: int = 30,
        eval_days: int = 7,
        step_days: int = 0,
        config: Optional[OptimizationConfig] = None,
        baselines: Optional[list[str]] = None,
        price_forecaster_cls: Optional[Type] = None,
        price_forecaster_config: Optional[Any] = None,
        carbon_forecaster_cls: Optional[Type] = None,
        carbon_forecaster_config: Optional[Any] = None,
        context_hours: int = 48,
        recorder_path: Optional[Path] = None,
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

        # ML forecaster class references (not instances — re-fitted per fold)
        self.price_forecaster_cls = price_forecaster_cls
        self.price_forecaster_config = price_forecaster_config
        self.carbon_forecaster_cls = carbon_forecaster_cls
        self.carbon_forecaster_config = carbon_forecaster_config
        self.context_hours = context_hours

        # Optional PostExecutionRecorder for learning-loop data accumulation
        self._recorder = None
        if recorder_path is not None:
            from aurelius.execution.post_execution import PostExecutionRecorder
            self._recorder = PostExecutionRecorder(output_path=str(recorder_path))

    @property
    def uses_ml_forecaster(self) -> bool:
        return self.price_forecaster_cls is not None

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

        method_label = "ml_quantile" if self.uses_ml_forecaster else "seasonal_naive"
        logger.info(
            f"BacktestEngine finished: {len(rounds)} folds, forecast_method={method_label}"
        )
        return rounds

    def _run_fold(
        self,
        split: TemporalSplit,
        all_jobs: list[Job],
        price_df: pd.DataFrame,
        carbon_df: pd.DataFrame,
    ) -> Optional[BacktestRound]:
        """Execute a single fold."""
        eval_jobs = [
            j for j in all_jobs
            if split.eval_start <= _to_ts(j.earliest_start) < split.eval_end
        ]
        if not eval_jobs:
            logger.debug(f"Fold {split.fold_index}: no jobs in eval window, skipping")
            return None

        # --- Build training data dicts (never contains eval-window rows) ---
        train_price_data = _df_to_price_data(split.train_df)
        train_carbon_data: dict[str, dict[datetime, float]] = {}
        if not carbon_df.empty:
            carbon_mask = (
                (pd.to_datetime(carbon_df["timestamp"]) >= split.train_start)
                & (pd.to_datetime(carbon_df["timestamp"]) < split.train_end)
            )
            train_carbon_data = _df_to_carbon_data(carbon_df[carbon_mask])

        # --- Actual eval-window data (used only for scoring, never for forecasting) ---
        # Use the FULL price_df (not eval-window-bounded) so jobs whose runtime
        # extends past split.eval_end get real prices instead of the $50/MWh
        # fallback. Scoring with actual realized prices outside the eval window
        # is NOT a leakage concern — the realized prices are public ground truth
        # by the time the job runs, and the forecaster only sees train_price_data.
        eval_price_data = _df_to_price_data(price_df)
        eval_carbon_data: dict[str, dict[datetime, float]] = {}
        if not carbon_df.empty:
            eval_carbon_data = _df_to_carbon_data(carbon_df)

        # --- Build forecast signals for the optimizer ---
        forecast_quality: Optional[ForecastQuality] = None

        if self.uses_ml_forecaster:
            forecast_price_data, forecast_carbon_data, forecast_quality = (
                self._build_ml_forecast(
                    split, train_price_data, train_carbon_data,
                    eval_price_data, eval_carbon_data,
                )
            )
        else:
            forecast_price_data = _build_hourly_price_forecast(
                train_price_data, split.eval_start, split.eval_end
            )
            forecast_carbon_data = _build_hourly_carbon_forecast(
                train_carbon_data, split.eval_start, split.eval_end
            )

        n_fc = sum(len(v) for v in forecast_price_data.values())
        logger.debug(
            f"Fold {split.fold_index}: {n_fc} forecast hours for optimizer"
        )

        # --- Run optimizer ---
        try:
            opt_result = self.scheduler.solve(
                eval_jobs, forecast_price_data, forecast_carbon_data,
                method=self.method,
            )
            opt_schedule = opt_result.schedule
        except Exception as exc:
            logger.error(f"Fold {split.fold_index}: optimizer failed: {exc}")
            opt_schedule = []

        opt_metrics = evaluate_schedule(opt_schedule, eval_jobs, eval_price_data, eval_carbon_data)
        if opt_metrics.missing_price_hours > 0:
            logger.warning(
                f"Fold {split.fold_index}: {opt_metrics.missing_price_hours} optimizer hours "
                "had no actual price data"
            )

        # --- Run baselines ---
        baseline_schedules: dict[str, list[ScheduleDecision]] = {}
        baseline_metrics: dict[str, RealizedMetrics] = {}
        for name in self.baseline_names:
            policy = ALL_BASELINES.get(name)
            if policy is None:
                logger.warning(f"Unknown baseline policy '{name}', skipping")
                continue
            try:
                bl_schedule = policy(eval_jobs, forecast_price_data, forecast_carbon_data, self.config)
                bl_metrics = evaluate_schedule(bl_schedule, eval_jobs, eval_price_data, eval_carbon_data)
                baseline_schedules[name] = bl_schedule
                baseline_metrics[name] = bl_metrics
            except Exception as exc:
                logger.error(f"Fold {split.fold_index}, baseline '{name}': {exc}")

        round_ = BacktestRound(
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
            forecast_quality=forecast_quality,
        )

        if self._recorder is not None:
            self._record_fold_decisions(
                fold_index=split.fold_index,
                opt_schedule=opt_schedule,
                baseline_schedules=baseline_schedules,
                eval_jobs=eval_jobs,
                forecast_price_data=forecast_price_data,
                forecast_carbon_data=forecast_carbon_data,
                eval_price_data=eval_price_data,
                eval_carbon_data=eval_carbon_data,
            )

        return round_

    # ------------------------------------------------------------------
    # ML forecaster path
    # ------------------------------------------------------------------

    def _build_ml_forecast(
        self,
        split: TemporalSplit,
        train_price_data: dict,
        train_carbon_data: dict,
        eval_price_data: dict,
        eval_carbon_data: dict,
    ) -> tuple[dict, dict, ForecastQuality]:
        """Fit ML forecasters on training data and predict the eval window.

        LEAKAGE GUARANTEE:
          - forecaster.fit() receives only records with timestamp < split.eval_start
          - recent_context (last self.context_hours of training) has
            max timestamp < split.eval_start
          - eval_price_data / eval_carbon_data are used ONLY for computing
            forecast_quality metrics AFTER the forecast is produced

        Returns:
            (forecast_price_data, forecast_carbon_data, ForecastQuality)
        """
        # Convert training price data back to EnergyPrice records for the forecaster
        train_price_records: list[EnergyPrice] = []
        for region, ts_map in train_price_data.items():
            for ts, price in ts_map.items():
                train_price_records.append(EnergyPrice(timestamp=ts, region=region, price_per_mwh=price))

        # Fit price forecaster on training records
        try:
            if self.price_forecaster_config is not None:
                price_fc = self.price_forecaster_cls(self.price_forecaster_config)
            else:
                price_fc = self.price_forecaster_cls()
            price_fc.fit(train_price_records)
        except Exception as exc:
            logger.error(f"ML price forecaster fit failed: {exc}; falling back to naive")
            naive_price = _build_hourly_price_forecast(train_price_data, split.eval_start, split.eval_end)
            naive_carbon = _build_hourly_carbon_forecast(train_carbon_data, split.eval_start, split.eval_end)
            return naive_price, naive_carbon, ForecastQuality(forecast_method="seasonal_naive_fallback")

        # Recent context: last context_hours of training records (strictly before eval window)
        context_sorted = sorted(train_price_records, key=lambda r: r.timestamp)
        recent_context = context_sorted[-self.context_hours:]

        # Predict eval window for each region using p50 as optimizer signal
        forecast_price_data: dict[str, dict[datetime, float]] = {}
        regions = list(set(r.region for r in train_price_records))

        eval_start_dt = split.eval_start.to_pydatetime()
        eval_end_dt = split.eval_end.to_pydatetime()
        if eval_start_dt.tzinfo is None:
            eval_start_dt = eval_start_dt.replace(tzinfo=timezone.utc)
        if eval_end_dt.tzinfo is None:
            eval_end_dt = eval_end_dt.replace(tzinfo=timezone.utc)

        n_hours = int((eval_end_dt - eval_start_dt).total_seconds() / 3600)
        eval_timestamps = [eval_start_dt + timedelta(hours=h) for h in range(n_hours)]

        for region in regions:
            region_context = [r for r in recent_context if r.region == region]
            try:
                preds = price_fc.predict(region, eval_timestamps, region_context)
                # Use p50 as the optimizer signal (median forecast)
                forecast_price_data[region] = {fc.timestamp: fc.p50 for fc in preds}
            except Exception as exc:
                logger.warning(f"Price prediction failed for region={region}: {exc}")
                # Fallback to naive for this region only
                naive = _build_hourly_price_forecast(
                    {region: train_price_data.get(region, {})},
                    split.eval_start, split.eval_end
                )
                forecast_price_data[region] = naive.get(region, {})

        # Carbon forecasting (if class provided; otherwise fallback to naive)
        forecast_carbon_data: dict[str, dict[datetime, float]] = {}
        if self.carbon_forecaster_cls is not None and train_carbon_data:
            train_carbon_records: list[CarbonIntensity] = []
            for region, ts_map in train_carbon_data.items():
                for ts, gco2 in ts_map.items():
                    train_carbon_records.append(CarbonIntensity(timestamp=ts, region=region, gco2_per_kwh=gco2))
            try:
                if self.carbon_forecaster_config is not None:
                    carbon_fc = self.carbon_forecaster_cls(self.carbon_forecaster_config)
                else:
                    carbon_fc = self.carbon_forecaster_cls()
                carbon_fc.fit(train_carbon_records)
                carbon_context = sorted(train_carbon_records, key=lambda r: r.timestamp)[-self.context_hours:]
                carbon_regions = list(set(r.region for r in train_carbon_records))
                for region in carbon_regions:
                    rc = [r for r in carbon_context if r.region == region]
                    try:
                        preds = carbon_fc.predict(region, eval_timestamps, rc)
                        forecast_carbon_data[region] = {fc.timestamp: fc.p50 for fc in preds}
                    except Exception as exc:
                        logger.warning(f"Carbon prediction failed for region={region}: {exc}")
            except Exception as exc:
                logger.warning(f"Carbon forecaster fit failed: {exc}; using naive carbon")
        if not forecast_carbon_data:
            forecast_carbon_data = _build_hourly_carbon_forecast(
                train_carbon_data, split.eval_start, split.eval_end
            )

        # --- Compute forecast quality using eval-window actuals ---
        # This is MEASUREMENT only: actuals are not fed back to the forecaster
        fq = self._measure_forecast_quality(
            price_fc, regions, eval_timestamps, recent_context,
            eval_price_data,
        )

        return forecast_price_data, forecast_carbon_data, fq

    def _measure_forecast_quality(
        self,
        price_fc: Any,
        regions: list[str],
        eval_timestamps: list[datetime],
        recent_context: list[EnergyPrice],
        eval_price_data: dict[str, dict[datetime, float]],
    ) -> ForecastQuality:
        """Compare p50/p90 predictions vs eval-window actuals for metrics.

        The forecaster is NOT re-fitted here and eval actuals are NOT passed
        to the forecaster — this function is read-only with respect to the model.
        """
        from aurelius.ml.forecast_evaluator import ForecastEvaluator, ForecastPoint
        import math

        all_actuals = []
        all_p50 = []
        all_p90 = []

        for region in regions:
            region_actuals = eval_price_data.get(region, {})
            if not region_actuals:
                continue
            region_context = [r for r in recent_context if r.region == region]
            try:
                preds = price_fc.predict(region, eval_timestamps, region_context)
            except Exception:
                continue
            pred_map = {fc.timestamp: fc for fc in preds}
            for ts, actual_price in region_actuals.items():
                fc = pred_map.get(ts)
                if fc is None:
                    # Normalise timezone
                    if ts.tzinfo is None:
                        ts_utc = ts.replace(tzinfo=timezone.utc)
                    else:
                        ts_utc = ts.astimezone(timezone.utc)
                    fc = pred_map.get(ts_utc)
                if fc is None:
                    continue
                all_actuals.append(ForecastPoint(ts, region, actual_price))
                all_p50.append(ForecastPoint(ts, region, fc.p50))
                all_p90.append(ForecastPoint(ts, region, fc.p90))

        if not all_actuals:
            return ForecastQuality(forecast_method="ml_quantile")

        try:
            result = ForecastEvaluator().evaluate(all_actuals, all_p50, all_p90)
            return ForecastQuality(
                n_eval_hours=result.n_samples,
                mape=result.mape,
                rmse=result.rmse,
                p90_coverage=result.p90_coverage,
                calibration_error=result.calibration_error,
                forecast_method="ml_quantile",
            )
        except Exception as exc:
            logger.warning(f"Forecast quality measurement failed: {exc}")
            return ForecastQuality(forecast_method="ml_quantile")


    # ------------------------------------------------------------------
    # PostExecutionRecorder wiring
    # ------------------------------------------------------------------

    def _record_fold_decisions(
        self,
        fold_index: int,
        opt_schedule: list[ScheduleDecision],
        baseline_schedules: dict[str, list[ScheduleDecision]],
        eval_jobs: list[Job],
        forecast_price_data: dict,
        forecast_carbon_data: dict,
        eval_price_data: dict,
        eval_carbon_data: dict,
    ) -> None:
        """Write one PostExecutionRecord per optimizer decision for this fold.

        Uses synthetic ExecutionResult/ExecutionConfig (status="dry_run") so
        the recorder never touches live infrastructure.  Forecast values come
        from the optimizer's price/carbon signal; realized values come from
        eval-window actuals (ground truth for that fold).
        """
        from aurelius.execution.post_execution import (
            ForecastSnapshot,
            PostExecutionRecord,
            RealizedOutcome,
        )
        from aurelius.execution.base import ExecutionConfig, ExecutionResult

        job_map: dict[str, Job] = {j.job_id: j for j in eval_jobs}

        # Use the first available baseline for comparison (deterministic order)
        ref_baseline_name = next(iter(baseline_schedules), None)
        ref_baseline_map: dict[str, ScheduleDecision] = {}
        if ref_baseline_name:
            for d in baseline_schedules[ref_baseline_name]:
                ref_baseline_map[d.job_id] = d

        exec_config = ExecutionConfig(mode="dry_run", constraint_profile="batch_optimized")

        for decision in opt_schedule:
            try:
                job = job_map.get(decision.job_id)
                baseline_decision = ref_baseline_map.get(decision.job_id)

                # --- Build ForecastSnapshot from the optimizer's price signal ---
                fc_price = None
                fc_baseline_price = None
                region_fc = forecast_price_data.get(decision.region, {})
                start_key = decision.start_time
                if start_key.tzinfo is None:
                    start_key = start_key.replace(tzinfo=timezone.utc)
                if region_fc:
                    fc_price = region_fc.get(start_key)
                    if fc_price is None:
                        # Round to nearest hour
                        rounded = start_key.replace(minute=0, second=0, microsecond=0)
                        fc_price = region_fc.get(rounded)

                if baseline_decision is not None and region_fc:
                    bl_key = baseline_decision.start_time
                    if bl_key.tzinfo is None:
                        bl_key = bl_key.replace(tzinfo=timezone.utc)
                    fc_baseline_price = region_fc.get(bl_key)
                    if fc_baseline_price is None:
                        bl_rounded = bl_key.replace(minute=0, second=0, microsecond=0)
                        fc_baseline_price = region_fc.get(bl_rounded)

                power_kw = job.power_kw if job else 1.0
                runtime = decision.actual_runtime_hours or (job.runtime_hours if job else 1.0)

                fc_energy_cost_p50 = None
                if fc_price is not None:
                    fc_energy_cost_p50 = fc_price * power_kw * runtime / 1000.0

                fc_energy_cost_baseline = None
                if fc_baseline_price is not None:
                    fc_energy_cost_baseline = fc_baseline_price * power_kw * runtime / 1000.0

                snapshot = ForecastSnapshot(
                    energy_cost_p50=fc_energy_cost_p50,
                    energy_cost_p90=None,
                    energy_cost_baseline=fc_energy_cost_baseline,
                )

                # --- Build RealizedOutcome from eval-window actuals ---
                real_price = None
                real_energy_cost = None
                region_eval = eval_price_data.get(decision.region, {})
                if region_eval:
                    real_price = region_eval.get(start_key)
                    if real_price is None:
                        rounded = start_key.replace(minute=0, second=0, microsecond=0)
                        real_price = region_eval.get(rounded)
                if real_price is not None:
                    real_energy_cost = real_price * power_kw * runtime / 1000.0

                realized = RealizedOutcome(
                    realized_start_time=decision.start_time,
                    realized_energy_price=real_price,
                    realized_energy_cost=real_energy_cost,
                )

                exec_result = ExecutionResult(
                    job_id=decision.job_id,
                    submitted=False,
                    aws_job_id=None,
                    region=decision.region,
                    submit_time=decision.start_time,
                    status="dry_run",
                )

                self._recorder.record(
                    decision=decision,
                    baseline_decision=baseline_decision,
                    execution_result=exec_result,
                    config=exec_config,
                    forecast=snapshot,
                    realized=realized,
                )
            except Exception as exc:
                logger.debug(
                    "Fold %d: failed to record decision for job %s: %s",
                    fold_index,
                    decision.job_id,
                    exc,
                )


def _to_ts(dt: datetime) -> pd.Timestamp:
    """Convert a datetime to a UTC-aware pd.Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _build_hourly_price_forecast(
    train_price_data: dict[str, dict[datetime, float]],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> dict[str, dict[datetime, float]]:
    """Build a leakage-free hour-of-day mean price forecast for the eval window.

    Uses ONLY training-window data. This is the "seasonal naive" fallback:
    for each region, computes mean price at each hour-of-day (0-23) across
    all training days, then projects those means onto every eval-window hour.
    """
    from datetime import timezone as _tz
    forecast: dict[str, dict[datetime, float]] = {}

    sample_tz = None
    for ts_map in train_price_data.values():
        for ts in ts_map.keys():
            sample_tz = ts.tzinfo
            break
        if sample_tz is not None:
            break
    key_tz = sample_tz if sample_tz is not None else _tz.utc

    eval_start_dt = _as_tz(eval_start, key_tz)
    eval_end_dt = _as_tz(eval_end, key_tz)

    for region, ts_map in train_price_data.items():
        if not ts_map:
            continue
        hour_sums: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_counts: dict[int, int] = {h: 0 for h in range(24)}
        for ts, price in ts_map.items():
            h = ts.hour
            hour_sums[h] += price
            hour_counts[h] += 1
        overall_mean = sum(ts_map.values()) / len(ts_map)
        hour_means = {
            h: (hour_sums[h] / hour_counts[h] if hour_counts[h] > 0 else overall_mean)
            for h in range(24)
        }
        region_forecast: dict[datetime, float] = {}
        cur = eval_start_dt
        while cur < eval_end_dt:
            region_forecast[cur] = hour_means[cur.hour]
            cur += timedelta(hours=1)
        forecast[region] = region_forecast

    return forecast


def _build_hourly_carbon_forecast(
    train_carbon_data: dict[str, dict[datetime, float]],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> dict[str, dict[datetime, float]]:
    """Build a leakage-free hour-of-day mean carbon forecast for the eval window."""
    from datetime import timezone as _tz
    forecast: dict[str, dict[datetime, float]] = {}

    sample_tz = None
    for ts_map in train_carbon_data.values():
        for ts in ts_map.keys():
            sample_tz = ts.tzinfo
            break
        if sample_tz is not None:
            break
    key_tz = sample_tz if sample_tz is not None else _tz.utc

    eval_start_dt = _as_tz(eval_start, key_tz)
    eval_end_dt = _as_tz(eval_end, key_tz)

    for region, ts_map in train_carbon_data.items():
        if not ts_map:
            continue
        hour_sums: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_counts: dict[int, int] = {h: 0 for h in range(24)}
        for ts, gco2 in ts_map.items():
            h = ts.hour
            hour_sums[h] += gco2
            hour_counts[h] += 1
        overall_mean = sum(ts_map.values()) / len(ts_map)
        hour_means = {
            h: (hour_sums[h] / hour_counts[h] if hour_counts[h] > 0 else overall_mean)
            for h in range(24)
        }
        region_forecast: dict[datetime, float] = {}
        cur = eval_start_dt
        while cur < eval_end_dt:
            region_forecast[cur] = hour_means[cur.hour]
            cur += timedelta(hours=1)
        forecast[region] = region_forecast

    return forecast


def _as_tz(ts: pd.Timestamp, tz) -> datetime:
    """Convert a pd.Timestamp to a datetime with the given timezone."""
    dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)
