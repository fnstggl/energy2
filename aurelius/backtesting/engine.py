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
      recent_context passed to predict() is the last `context_hours` of training
      data PER REGION (default 192 h, enough for lag_168h on multi-region setups).
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

from aurelius.backtesting.baselines import ALL_BASELINES
from aurelius.backtesting.evaluator import RealizedMetrics, evaluate_schedule
from aurelius.backtesting.splitter import TemporalSplit, TemporalSplitter
from aurelius.models import CarbonIntensity, EnergyPrice, Job, OptimizationConfig, ScheduleDecision
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


class WeatherLeakageError(AssertionError):
    """Raised when eval-window predict weather is sourced from observed actuals."""


def _assert_no_observed_weather_in_eval(
    predict_weather_df: pd.DataFrame,
    observed_weather_df: pd.DataFrame,
    eval_start: pd.Timestamp,
) -> None:
    """Guard: eval-window predict weather must come from the FORECAST, not actuals.

    For every (timestamp, region) at or after eval_start, the temperature in the
    predict frame must match the forecast frame that was concatenated in — i.e.
    it must NOT silently equal the observed actual. We verify by confirming the
    eval-window temperatures differ from the observed series on at least some
    rows (a perfect-foresight join would be identical everywhere). A degenerate
    case where forecast == observed exactly on every hour is astronomically
    unlikely for real NWP output and would itself indicate a wiring bug.
    """
    pw = predict_weather_df.copy()
    pw["timestamp"] = pd.to_datetime(pw["timestamp"], utc=True)
    ow = observed_weather_df.copy()
    ow["timestamp"] = pd.to_datetime(ow["timestamp"], utc=True)
    pe = pw[pw["timestamp"] >= eval_start]
    oe = ow[ow["timestamp"] >= eval_start]
    if pe.empty or oe.empty:
        return
    merged = pe.merge(oe, on=["timestamp", "region"], suffixes=("_pred", "_obs"))
    if merged.empty:
        return
    identical = (merged["temperature_c_pred"] - merged["temperature_c_obs"]).abs() < 1e-9
    if bool(identical.all()):
        raise WeatherLeakageError(
            "Eval-window predict weather is identical to observed actuals "
            "(perfect-foresight leakage): expected day-ahead forecast values."
        )


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
        context_hours: int = 192,  # 168h for lag_168h + 24h safety margin
        recorder_path: Optional[Path] = None,
        rt_risk_lambda: Optional[float] = None,
        weather_df: Optional[Any] = None,  # pd.DataFrame with canonical weather schema
        forecast_weather_df: Optional[Any] = None,  # day-ahead forecast weather (leakage-free eval)
        queue_df: Optional[Any] = None,    # pd.DataFrame with canonical queue schema
        gpu_df: Optional[Any] = None,      # pd.DataFrame with canonical GPU metric schema
        apply_recovery_correction: bool = False,  # enable regime-aware forecast correction
        recovery_excluded_workload_types: frozenset = frozenset(),  # skip correction for these
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

        # Diagnostic-only: when True, the optimizer is fed ACTUAL eval-window
        # prices as its "forecast" (perfect foresight). This is leakage and
        # must NEVER be used to report real savings — its sole purpose is to
        # isolate forecast-quality limits from structural price-spread limits.
        # If oracle savings >> ML savings, forecasting is the bottleneck.
        # If oracle savings ≈ ML savings, the inter-region/inter-hour spread is
        # the bottleneck (more regions / better workload mix needed, not better ML).
        self.oracle_forecast = False

        # Rolling-horizon (receding-horizon / MPC) mode. When set, models the
        # production reality that day-ahead prices are PUBLISHED ~1 day ahead:
        # the optimizer re-plans every `replan_hours` and, at each replan, knows
        # the actual prices for the next `forecast_horizon_hours` (published DAM),
        # falling back to the ML/naive forecast beyond that. This is NOT leakage —
        # it mirrors exactly what a production scheduler sees. None = disabled
        # (single one-shot optimize over the whole eval window).
        self.forecast_horizon_hours: Optional[int] = None
        self.replan_hours: int = 24

        # DA->RT spread risk adjustment. When set (>= 0) AND a settlement price
        # series is provided, the optimizer plans against a risk-adjusted RT
        # estimate (DA + learned conditional spread + lambda * upside risk),
        # fit per fold on the training window only. None = disabled (plan on raw
        # DA). lambda=0 applies debias only; higher penalizes spike-prone slots.
        self.rt_risk_lambda: Optional[float] = rt_risk_lambda

        # Optional weather DataFrame for weather-feature-enhanced ML forecasting.
        # Schema: timestamp, region, temperature_c, hdd_f, cdd_f, wind_speed_ms,
        # temp_rolling_24h_c, temp_delta_24h_c.  Passed to _build_ml_forecast()
        # per fold with proper train/eval split. None = price-only mode (no change
        # in ML behaviour vs. v2.0).
        self.weather_df = weather_df

        # Optional FORECAST weather (same canonical schema) used for the eval
        # window at PREDICT time. This is the leakage-free counterpart to
        # weather_df: in deployment the optimizer only ever has a day-ahead
        # weather *forecast* for the horizon it is scheduling, never the future
        # observation. When provided, _build_ml_forecast() builds the predict
        # weather as observed-history (< eval_start) + forecast (>= eval_start),
        # eliminating the perfect-foresight leakage. When None, the eval window
        # falls back to observed actuals from weather_df (perfect foresight —
        # an OPTIMISTIC upper bound, retained only for backward compatibility).
        self.forecast_weather_df = forecast_weather_df

        # Optional queue DataFrame for queue-aware placement optimization.
        # Schema: timestamp, region, cluster_id, gpu_type, available_gpus,
        # queue_depth_jobs, est_wait_hours.  Converted to a dict lookup per
        # fold so the optimizer can penalise congested regions.  Leakage-safe:
        # only rows with timestamp ≤ eval window are used for each fold.
        # None = queue-unaware mode (no change to existing behaviour).
        self.queue_df = queue_df

        # Optional GPU telemetry DataFrame for GPU-health-aware placement.
        # Schema: canonical GPUMetrics columns (see aurelius/ingestion/dcgm_provider.py).
        # Converted to {region: {timestamp: avg_health_penalty}} per fold.
        # Leakage-safe: only snapshots with timestamp < eval_start are used.
        # None = GPU-health-unaware mode.
        self.gpu_df = gpu_df

        # When True, apply regime-aware forecast correction after ML prediction.
        # Detects post-spike recovery (recent prices << training mean) and reduces
        # the forecast bias toward recently observed prices with exponential decay.
        # Only activates when recent_mean / training_mean < 0.40 per region.
        # Has no effect when price_forecaster_cls is None (seasonal-naive path).
        self.apply_recovery_correction = apply_recovery_correction
        # Workload types for which recovery correction is suppressed even when
        # apply_recovery_correction=True. Training workloads regress (-2.7pp) because
        # the exponential decay pattern distorts long-horizon (96-200h) scheduling.
        self.recovery_excluded_workload_types: frozenset = frozenset(recovery_excluded_workload_types)

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
        settle_price_df: Optional[pd.DataFrame] = None,
    ) -> list[BacktestRound]:
        """Run the full walk-forward backtest.

        Args:
            jobs:      All jobs. The engine assigns jobs to folds by
                       earliest_start timestamp.
            price_df:  Canonical PLANNING price DataFrame (columns: timestamp,
                       region, price_per_mwh). This is the known-ahead signal the
                       optimizer plans against (e.g. day-ahead LMP) and the
                       forecaster is trained on.
            carbon_df: Canonical carbon DataFrame (columns: timestamp, region, gco2_per_kwh).
            start:     Backtest start timestamp (defaults to min timestamp in price_df).
            end:       Backtest end timestamp (defaults to max timestamp in price_df + 1h).
            settle_price_df: Optional SETTLEMENT price DataFrame (same schema).
                       When provided, schedules are SCORED against these prices
                       (e.g. realized real-time LMP) while the optimizer still
                       plans against price_df. This models an RT-exposed customer
                       who plans on day-ahead but pays real-time. When None,
                       settlement == planning price (DA-hedged customer).

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
            round_ = self._run_fold(split, jobs, price_df, carbon_df, settle_price_df)
            if round_ is not None:
                rounds.append(round_)

        if self.oracle_forecast:
            method_label = "oracle"
        elif self.uses_ml_forecaster:
            method_label = "ml_quantile"
        else:
            method_label = "seasonal_naive"
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
        settle_price_df: Optional[pd.DataFrame] = None,
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

        # Settlement prices: what the customer actually PAYS. Schedules are scored
        # against these (e.g. realized real-time LMP) while the optimizer plans
        # against eval_price_data (e.g. known day-ahead). When no settlement df is
        # given, settlement == planning price (DA-hedged customer pays what they
        # planned). This split is the DA-plan / RT-settle model.
        if settle_price_df is not None and not settle_price_df.empty:
            settle_price_data = _df_to_price_data(settle_price_df)
        else:
            settle_price_data = eval_price_data

        # --- Fit DA->RT spread risk model on TRAIN window only (no leakage) ---
        # The optimizer will plan against risk-adjusted prices (estimate of RT),
        # while baselines keep raw prices and scoring stays on actual RT.
        spread_risk_model = None
        if (
            self.rt_risk_lambda is not None
            and settle_price_df is not None
            and not settle_price_df.empty
        ):
            settle_mask = (
                (pd.to_datetime(settle_price_df["timestamp"]) >= split.train_start)
                & (pd.to_datetime(settle_price_df["timestamp"]) < split.train_end)
            )
            train_settle_data = _df_to_price_data(settle_price_df[settle_mask])
            from aurelius.forecasting.spread_risk import SpreadRiskModel
            spread_risk_model = SpreadRiskModel(risk_lambda=self.rt_risk_lambda).fit(
                train_price_data, train_settle_data
            )

        # --- Build forecast signals for the optimizer ---
        forecast_quality: Optional[ForecastQuality] = None

        # Forecast horizon must cover the latest hour any eval job could be
        # scheduled into — i.e. up to the latest job deadline — NOT just the
        # eval window. Otherwise candidate start times past eval_end have no
        # forecast price and the objective falls back to a flat $50/MWh
        # (objective.py), which creates phantom-cheap future hours and lets the
        # optimizer "park" deadline-flexible jobs months out (e.g. scheduling a
        # late-March job to start in late May). That both corrupts the savings
        # number and triggers missing-price warnings at scoring time.
        forecast_end = split.eval_end
        for j in eval_jobs:
            j_deadline = _to_ts(j.deadline)
            if j_deadline > forecast_end:
                forecast_end = j_deadline
        # Pad by a day so a job starting at its deadline-minus-runtime still has
        # prices for its full runtime.
        forecast_end = forecast_end + pd.Timedelta(days=1)

        if self.oracle_forecast:
            # DIAGNOSTIC: perfect-foresight. Feed actual eval-window prices to
            # the optimizer. This is intentional leakage — used only to measure
            # the ceiling achievable with a perfect forecaster, isolating
            # forecast quality from structural price-spread limits.
            forecast_price_data = {r: dict(ts_map) for r, ts_map in eval_price_data.items()}
            forecast_carbon_data = {r: dict(ts_map) for r, ts_map in eval_carbon_data.items()}
            forecast_quality = ForecastQuality(forecast_method="oracle")
        elif self.uses_ml_forecaster:
            # Suppress recovery correction for workload types that regress with it.
            # Training jobs (96-200h) regress because the exponential decay distorts
            # long-horizon start-time decisions. Check if ALL fold jobs are excluded.
            fold_wl_types = {j.workload_type for j in eval_jobs}
            skip_recovery = (
                self.apply_recovery_correction
                and bool(self.recovery_excluded_workload_types)
                and fold_wl_types.issubset(self.recovery_excluded_workload_types)
            )
            forecast_price_data, forecast_carbon_data, forecast_quality = (
                self._build_ml_forecast(
                    split, train_price_data, train_carbon_data,
                    eval_price_data, eval_carbon_data, forecast_end,
                    weather_df=self.weather_df,
                    forecast_weather_df=self.forecast_weather_df,
                    skip_recovery_correction=skip_recovery,
                )
            )
        else:
            forecast_price_data = _build_hourly_price_forecast(
                train_price_data, split.eval_start, forecast_end
            )
            forecast_carbon_data = _build_hourly_carbon_forecast(
                train_carbon_data, split.eval_start, forecast_end
            )

        n_fc = sum(len(v) for v in forecast_price_data.values())
        logger.debug(
            f"Fold {split.fold_index}: {n_fc} forecast hours for optimizer"
        )

        # --- Build queue signal for the optimizer (leakage-safe) ---
        # Only use queue state recorded BEFORE the eval window starts so the
        # optimizer never sees future congestion during backtesting.
        queue_data: Optional[dict[str, dict[datetime, float]]] = None
        if self.queue_df is not None and not (
            hasattr(self.queue_df, "empty") and self.queue_df.empty
        ):
            from aurelius.ingestion.queue_provider import QueueProvider
            q_ts = pd.to_datetime(self.queue_df["timestamp"], utc=True)
            q_mask = q_ts < split.eval_start
            if q_mask.any():
                queue_provider = QueueProvider.from_dataframe(self.queue_df[q_mask])
                queue_data = queue_provider.to_dict_lookup()

        # --- Build GPU health signal for the optimizer (leakage-safe) ---
        # Only use GPU snapshots recorded BEFORE eval_start.
        gpu_health_data: Optional[dict[str, dict[datetime, float]]] = None
        if self.gpu_df is not None and not (
            hasattr(self.gpu_df, "empty") and self.gpu_df.empty
        ):
            from aurelius.ingestion.dcgm_provider import DCGMProvider
            g_ts = pd.to_datetime(self.gpu_df["timestamp"], utc=True)
            g_mask = g_ts < split.eval_start
            if g_mask.any():
                gpu_provider = DCGMProvider.from_dataframe(self.gpu_df[g_mask])
                gpu_health_data = gpu_provider.to_dict_lookup()

        # --- Run optimizer ---
        try:
            if self.forecast_horizon_hours is not None and not self.oracle_forecast:
                # Rolling-horizon (receding-horizon) optimization: re-plan at a
                # daily cadence, revealing actual published DAM prices for the
                # next forecast_horizon_hours at each replan, ML/naive beyond.
                opt_schedule = self._rolling_optimize(
                    split, eval_jobs, eval_price_data, forecast_price_data,
                    forecast_carbon_data, spread_risk_model,
                )
            else:
                opt_prices = (
                    spread_risk_model.adjust_price_map(forecast_price_data)
                    if spread_risk_model is not None else forecast_price_data
                )
                opt_result = self.scheduler.solve(
                    eval_jobs, opt_prices, forecast_carbon_data,
                    method=self.method,
                    queue_data=queue_data,
                    gpu_health_data=gpu_health_data,
                )
                opt_schedule = opt_result.schedule
        except Exception as exc:
            logger.error(f"Fold {split.fold_index}: optimizer failed: {exc}")
            opt_schedule = []

        opt_metrics = evaluate_schedule(opt_schedule, eval_jobs, settle_price_data, eval_carbon_data)
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
                bl_metrics = evaluate_schedule(bl_schedule, eval_jobs, settle_price_data, eval_carbon_data)
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
                eval_price_data=settle_price_data,
                eval_carbon_data=eval_carbon_data,
            )

        return round_

    # ------------------------------------------------------------------
    # Rolling-horizon (receding-horizon / MPC) optimization
    # ------------------------------------------------------------------

    def _rolling_optimize(
        self,
        split: TemporalSplit,
        eval_jobs: list[Job],
        eval_price_data: dict[str, dict[datetime, float]],
        base_forecast: dict[str, dict[datetime, float]],
        forecast_carbon_data: dict[str, dict[datetime, float]],
        spread_risk_model=None,
    ) -> list[ScheduleDecision]:
        """Receding-horizon optimization mirroring production DAM publication.

        Models the reality that day-ahead prices are published ~1 day ahead:
        the scheduler re-plans every `replan_hours`, and at each replan it
        KNOWS the actual prices for the next `forecast_horizon_hours` (published
        DAM), falling back to the ML/naive forecast (base_forecast) beyond.

        Jobs are planned in waves keyed by earliest_start. A job is planned in
        the first wave whose window reaches its earliest_start, and its schedule
        is committed then (commit-at-start; mid-flight re-planning of in-flight
        jobs is a documented Phase-2 enhancement). Scoring is always on actuals.

        This is NOT leakage: revealing the next ~24h of actual DAM prices is
        exactly what a production scheduler has access to. Only the >horizon
        tail uses the (leakage-free) forecast.
        """
        horizon = self.forecast_horizon_hours
        replan = max(1, self.replan_hours)

        regions = set(base_forecast) | set(eval_price_data)
        committed: dict[str, ScheduleDecision] = {}

        def build_price_view(plan_time: datetime) -> dict[str, dict[datetime, float]]:
            horizon_end = plan_time + timedelta(hours=horizon)
            view: dict[str, dict[datetime, float]] = {}
            for region in regions:
                # Base: forecast everywhere we have it
                pv = dict(base_forecast.get(region, {}))
                # Override with ACTUAL within the published-DAM window
                for h, price in eval_price_data.get(region, {}).items():
                    if plan_time <= h < horizon_end:
                        pv[h] = price
                view[region] = pv
            # Plan against a risk-adjusted RT estimate: even the known DA prices
            # within the horizon are converted toward expected RT (+ spike
            # penalty), since the customer ultimately pays RT. Scoring is still
            # done on actual RT outside this function.
            if spread_risk_model is not None:
                view = spread_risk_model.adjust_price_map(view)
            return view

        # Walk replan waves. At each wave we (1) MID-FLIGHT RE-PLAN the remaining
        # migration path of every in-flight job using freshly-published actual
        # prices, then (2) plan newly-available jobs. The loop continues past
        # eval_end while any job is still running, so long jobs get re-planned
        # through their full runtime (this is the receding-horizon / MPC core).
        job_by_id = {j.job_id: j for j in eval_jobs}
        plan_time = split.eval_start
        eval_end = split.eval_end
        hard_stop = eval_end + timedelta(days=400)  # safety against runaway loops

        while True:
            wave_cutoff = plan_time + timedelta(hours=replan)
            price_view = build_price_view(plan_time)

            # (1) Mid-flight re-planning of in-flight jobs.
            for jid, dec in list(committed.items()):
                if dec.start_time < plan_time < dec.end_time:
                    committed[jid] = self.scheduler.replan_remainder(
                        dec, job_by_id[jid], price_view, plan_time,
                    )

            # (2) Plan newly-available, not-yet-committed jobs.
            wave = [
                j for j in eval_jobs
                if j.job_id not in committed
                and _to_ts(j.earliest_start) < wave_cutoff
            ]
            if wave:
                res = self.scheduler.solve(
                    wave, price_view, forecast_carbon_data, method=self.method,
                )
                for d in res.schedule:
                    committed[d.job_id] = d

            plan_time = wave_cutoff

            # Termination: past the eval window, every eval job committed, and
            # nothing still in flight.
            if plan_time >= eval_end:
                all_committed = all(j.job_id in committed for j in eval_jobs)
                any_inflight = any(
                    committed[j.job_id].start_time < plan_time < committed[j.job_id].end_time
                    for j in eval_jobs if j.job_id in committed
                )
                if (all_committed and not any_inflight) or plan_time > hard_stop:
                    break

        # Safety: plan any uncommitted job (should not occur — all eval jobs have
        # earliest_start < eval_end and are committed before the loop exits).
        leftover = [j for j in eval_jobs if j.job_id not in committed]
        if leftover:
            res = self.scheduler.solve(
                leftover, build_price_view(eval_end), forecast_carbon_data,
                method=self.method,
            )
            for d in res.schedule:
                committed[d.job_id] = d

        # Preserve input job order
        return [committed[j.job_id] for j in eval_jobs if j.job_id in committed]

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
        forecast_end: Optional[pd.Timestamp] = None,
        weather_df: Optional[Any] = None,
        forecast_weather_df: Optional[Any] = None,
        skip_recovery_correction: bool = False,
    ) -> tuple[dict, dict, ForecastQuality]:
        """Fit ML forecasters on training data and predict the eval window.

        LEAKAGE GUARANTEE:
          - forecaster.fit() receives only records with timestamp < split.eval_start
          - recent_context (last self.context_hours of training) has
            max timestamp < split.eval_start
          - eval_price_data / eval_carbon_data are used ONLY for computing
            forecast_quality metrics AFTER the forecast is produced
          - weather_df is split into train (< eval_start) for fit() and full
            DataFrame (including eval window) for predict() — this mirrors how
            weather forecasts work in production: at decision time you have the
            current weather and a near-future weather forecast, both of which
            are exogenous inputs (not the target being predicted).

        Returns:
            (forecast_price_data, forecast_carbon_data, ForecastQuality)
        """
        # Slice weather into training and predict windows
        train_weather_df: Optional[Any] = None
        predict_weather_df: Optional[Any] = None
        if weather_df is not None and not (hasattr(weather_df, "empty") and weather_df.empty):
            wts = pd.to_datetime(weather_df["timestamp"], utc=True)
            train_mask = wts < split.eval_start
            train_weather_df = weather_df[train_mask].copy() if train_mask.any() else None

            if forecast_weather_df is not None and not (
                hasattr(forecast_weather_df, "empty") and forecast_weather_df.empty
            ):
                # LEAKAGE-FREE predict weather: observed history before eval_start
                # (rolling/lag context) + day-ahead FORECAST from eval_start onward.
                # This is what a deployed system actually has at decision time —
                # never the future observation.
                fwts = pd.to_datetime(forecast_weather_df["timestamp"], utc=True)
                obs_hist = weather_df[train_mask]
                fc_future = forecast_weather_df[fwts >= split.eval_start]
                predict_weather_df = pd.concat([obs_hist, fc_future], ignore_index=True)
                _assert_no_observed_weather_in_eval(
                    predict_weather_df, weather_df, split.eval_start
                )
            else:
                # Fallback (perfect foresight): eval-window weather = OBSERVED
                # actuals. NOT price leakage (weather is exogenous) but it IS
                # forecast leakage — the model sees weather it could not have had.
                # Optimistic upper bound; pass forecast_weather_df for honest eval.
                predict_weather_df = weather_df.copy()  # full range

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
            # Backward-compatible fit: pass weather_df only if the forecaster accepts it.
            # This allows custom forecasters with the legacy fit(prices) signature to work.
            import inspect as _inspect
            _fit_sig = _inspect.signature(price_fc.fit)
            if "weather_df" in _fit_sig.parameters:
                price_fc.fit(train_price_records, weather_df=train_weather_df)
            else:
                price_fc.fit(train_price_records)
        except Exception as exc:
            logger.error(f"ML price forecaster fit failed: {exc}; falling back to naive")
            _fc_end = forecast_end if forecast_end is not None else split.eval_end
            naive_price = _build_hourly_price_forecast(train_price_data, split.eval_start, _fc_end)
            naive_carbon = _build_hourly_carbon_forecast(train_carbon_data, split.eval_start, _fc_end)
            return naive_price, naive_carbon, ForecastQuality(forecast_method="seasonal_naive_fallback")

        # Recent context: last context_hours of training records, sliced
        # PER REGION (not globally). Global slicing meant that for N regions,
        # each region only got ~context_hours/N records — which silently
        # broke long-horizon lag features (e.g. lag_168h needs >=168 records
        # per region). Per-region slicing guarantees each region has
        # context_hours of history.
        context_by_region: dict[str, list[EnergyPrice]] = {}
        for record in sorted(train_price_records, key=lambda r: r.timestamp):
            context_by_region.setdefault(record.region, []).append(record)
        recent_context = []
        for region_records in context_by_region.values():
            recent_context.extend(region_records[-self.context_hours:])
        recent_context.sort(key=lambda r: r.timestamp)

        # Predict eval window for each region using p50 as optimizer signal
        forecast_price_data: dict[str, dict[datetime, float]] = {}
        regions = list(set(r.region for r in train_price_records))

        eval_start_dt = split.eval_start.to_pydatetime()
        # Forecast out to forecast_end (covers latest job deadline), not just
        # the eval window — see _run_fold for why (phantom-$50 future hours).
        _fc_end = forecast_end if forecast_end is not None else split.eval_end
        eval_end_dt = _fc_end.to_pydatetime() if hasattr(_fc_end, "to_pydatetime") else _fc_end
        if eval_start_dt.tzinfo is None:
            eval_start_dt = eval_start_dt.replace(tzinfo=timezone.utc)
        if eval_end_dt.tzinfo is None:
            eval_end_dt = eval_end_dt.replace(tzinfo=timezone.utc)

        n_hours = int((eval_end_dt - eval_start_dt).total_seconds() / 3600)
        eval_timestamps = [eval_start_dt + timedelta(hours=h) for h in range(n_hours)]

        # Forecast-quality measurement must use ONLY the real eval window
        # (eval_start..split.eval_end), not the extended forecast horizon, so
        # metrics aren't diluted by post-window hours.
        true_eval_end_dt = split.eval_end.to_pydatetime()
        if true_eval_end_dt.tzinfo is None:
            true_eval_end_dt = true_eval_end_dt.replace(tzinfo=timezone.utc)
        n_eval_hours = int((true_eval_end_dt - eval_start_dt).total_seconds() / 3600)
        eval_window_timestamps = [eval_start_dt + timedelta(hours=h) for h in range(n_eval_hours)]

        for region in regions:
            region_context = [r for r in recent_context if r.region == region]
            # Slice predict_weather_df to this region only for efficiency
            region_weather_df: Optional[Any] = None
            if predict_weather_df is not None and not (
                hasattr(predict_weather_df, "empty") and predict_weather_df.empty
            ):
                mask = predict_weather_df.get("region", pd.Series(dtype=str)) == region
                rw = predict_weather_df[mask]
                region_weather_df = rw if not rw.empty else None
            try:
                # Backward-compatible predict: pass weather_df only if the forecaster accepts it.
                _pred_sig = _inspect.signature(price_fc.predict)
                if "weather_df" in _pred_sig.parameters:
                    preds = price_fc.predict(
                        region, eval_timestamps, region_context,
                        weather_df=region_weather_df,
                    )
                else:
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

        # Apply regime-aware recovery correction (opt-in, ML mode only).
        # Reduces forecast bias when a region is in post-spike recovery:
        # recent_mean / training_mean < 0.40 triggers per-region correction.
        # skip_recovery_correction=True when all fold jobs are in the exclusion set
        # (e.g. "training" workloads that regress due to long-horizon decay distortion).
        if self.apply_recovery_correction and not skip_recovery_correction and forecast_price_data:
            from aurelius.forecasting.regime import RegimeDetector
            detector = RegimeDetector()
            forecast_price_data = detector.apply_corrections_to_forecast(
                forecast_price_data, train_price_data, recent_context
            )

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
                # Per-region context slicing (same rationale as price context above)
                carbon_by_region: dict[str, list] = {}
                for record in sorted(train_carbon_records, key=lambda r: r.timestamp):
                    carbon_by_region.setdefault(record.region, []).append(record)
                carbon_context = []
                for region_records in carbon_by_region.values():
                    carbon_context.extend(region_records[-self.context_hours:])
                carbon_context.sort(key=lambda r: r.timestamp)
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
            price_fc, regions, eval_window_timestamps, recent_context,
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
        from aurelius.execution.base import ExecutionConfig, ExecutionResult
        from aurelius.execution.post_execution import (
            ForecastSnapshot,
            RealizedOutcome,
        )

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
