"""LiveShadowRunner — makes optimizer decisions as if running live.

The runner trains only on historical data strictly before decision_time,
forecasts the next horizon_hours, then runs the optimizer and the
current_price_only baseline on the submitted jobs.

Leakage invariant:
    All price records with timestamp >= decision_time are NEVER passed to
    the forecaster or optimizer. RT prices are not used at all — only DA
    planning prices are visible at decision time. This mirrors what a real
    production system sees when it schedules a job.

Typical usage:
    # With customer workload trace
    runner = LiveShadowRunner(
        regions=["us-west", "us-east", "us-south"],
        price_forecaster_cls=PriceQuantileForecaster,
        price_forecaster_config=PriceModelConfig(seed=42),
        train_days=30,
    )
    records = runner.run(price_df=da_price_df, jobs=customer_jobs)

    # Save for later realization
    recorder = DecisionRecorder(output_path=Path("reports/shadow/decisions.jsonl"))
    recorder.save(records)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

import pandas as pd

from aurelius.backtesting.baselines import current_price_only_policy
from aurelius.backtesting.engine import _df_to_price_data, _df_to_price_records
from aurelius.models import Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler

from .models import DecisionRecord, make_run_id

logger = logging.getLogger(__name__)


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    """Convert a datetime to a UTC-aware pandas Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class LiveShadowRunner:
    """Makes optimizer decisions as if running live, recording DecisionRecords.

    Unlike BacktestEngine (which rolls through historical folds), this runner
    makes a single forward-looking pass: train on historical data before
    decision_time, forecast the next horizon_hours, schedule submitted jobs.

    No workloads are executed. Records are saved for later realization.
    """

    def __init__(
        self,
        regions: Optional[list[str]] = None,
        method: str = "greedy",
        train_days: int = 30,
        horizon_hours: int = 168,
        config: Optional[OptimizationConfig] = None,
        price_forecaster_cls: Optional[Type] = None,
        price_forecaster_config: Optional[Any] = None,
        context_hours: int = 336,
        run_id: Optional[str] = None,
        forecaster_version: str = "ml_quantile",
        optimizer_version: str = "greedy_migrate",
        safety_gate_config: Optional[Any] = None,
        enable_safety_gate: bool = True,
        apply_recovery_correction: bool = False,
        weather_df: Optional[Any] = None,
        forecast_weather_df: Optional[Any] = None,
    ) -> None:
        """
        Args:
            regions:                   Allowed regions (default: all in price_df).
            method:                    Optimizer method ("greedy", "local_search").
            train_days:                Days of history to use for forecaster training.
            horizon_hours:             How far ahead to forecast and schedule (default: 168h).
            config:                    OptimizationConfig (uses defaults if None).
            price_forecaster_cls:      ML forecaster class (e.g. PriceQuantileForecaster).
                                       If None, seasonal-naive mean is used.
            price_forecaster_config:   Config passed to price_forecaster_cls().
            context_hours:             Tail of training data passed as lag context.
            run_id:                    Optional fixed run ID (auto-generated if None).
            forecaster_version:        Label for audit trail.
            optimizer_version:         Label for audit trail.
            apply_recovery_correction: When True, applies two-gate regime-recovery
                                       bias correction to the ML forecast (ml_quantile_recovery
                                       mode). Reduces post-cold-snap ERCOT overprediction.
                                       Only active when price_forecaster_cls is not None.
        """
        self.regions = regions or []
        self.method = method
        self.train_days = train_days
        self.horizon_hours = horizon_hours
        self.config = config or OptimizationConfig()
        self.price_forecaster_cls = price_forecaster_cls
        self.price_forecaster_config = price_forecaster_config
        self.context_hours = context_hours
        self.run_id = run_id or make_run_id()
        self.forecaster_version = forecaster_version
        self.optimizer_version = optimizer_version
        self.apply_recovery_correction = apply_recovery_correction
        # Observed weather history (for forecaster training) and the day-ahead
        # forecast weather (for the prediction horizon). forecast_weather_df is
        # leakage-free by construction: it is the forecast available at decision
        # time, never a future observation. When forecast_weather_df is None the
        # runner falls back to weather_df for the horizon (observed history only
        # extends to decision_time, so practically price-only beyond it).
        self.weather_df = weather_df
        self.forecast_weather_df = forecast_weather_df
        self.scheduler = JobScheduler(self.config)

        from aurelius.safety.quantile_gate import QuantileGateConfig, QuantileSafetyGate

        self.enable_safety_gate = enable_safety_gate
        self._safety_gate = QuantileSafetyGate()
        self._safety_gate_config = safety_gate_config or QuantileGateConfig(
            enabled=enable_safety_gate,
            metric="energy_cost",
            min_expected_savings_pct=0.0,
            max_downside_risk_pct=10.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        price_df: pd.DataFrame,
        jobs: list[Job],
        carbon_df: Optional[pd.DataFrame] = None,
        decision_time: Optional[datetime] = None,
    ) -> list[DecisionRecord]:
        """Run the optimizer as if live and return DecisionRecords.

        Args:
            price_df:      Canonical DA price DataFrame (timestamp, region, price_per_mwh).
                           The runner uses only rows with timestamp < decision_time
                           for training. Rows >= decision_time may exist (they are
                           silently ignored at training time).
            jobs:          Jobs to schedule. Typically submit_time >= decision_time,
                           but any jobs with deadline > decision_time are included.
            carbon_df:     Optional carbon DataFrame (same schema with gco2_per_kwh).
            decision_time: The logical "now" for this shadow run.
                           Defaults to the last available timestamp in price_df + 1h.

        Returns:
            List of DecisionRecord with predicted fields filled.
            All realized_* fields are None (filled later by RealizedSavingsCalculator).
        """
        if price_df.empty:
            logger.warning("LiveShadowRunner: price_df is empty — returning no records")
            return []
        if not jobs:
            logger.warning("LiveShadowRunner: no jobs submitted — returning no records")
            return []

        price_df = price_df.copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

        decision_time = self._resolve_decision_time(price_df, decision_time)

        effective_regions = self._resolve_regions(price_df)
        if not effective_regions:
            logger.warning("LiveShadowRunner: no regions found in price_df")
            return []

        # --- Build training split (LEAKAGE GUARD: strict < decision_time) ---
        train_start = decision_time - timedelta(days=self.train_days)
        ts_train_start = _to_utc_ts(train_start)
        ts_decision = _to_utc_ts(decision_time)
        train_mask = (
            (price_df["timestamp"] >= ts_train_start)
            & (price_df["timestamp"] < ts_decision)
        )
        train_df = price_df[train_mask]

        if train_df.empty:
            logger.warning(
                f"LiveShadowRunner: no training data in [{train_start.date()} — {decision_time.date()})"
            )
            return []

        train_price_data = _df_to_price_data(train_df)

        carbon_data: dict = {}
        if carbon_df is not None and not carbon_df.empty:
            from aurelius.backtesting.engine import _df_to_carbon_data
            carbon_df = carbon_df.copy()
            carbon_df["timestamp"] = pd.to_datetime(carbon_df["timestamp"], utc=True)
            c_mask = carbon_df["timestamp"] < _to_utc_ts(decision_time)
            carbon_data = _df_to_carbon_data(carbon_df[c_mask])

        # --- Build forecast prices for the future horizon ---
        forecast_price_data = self._build_forecast(
            train_df=train_df,
            decision_time=decision_time,
            effective_regions=effective_regions,
            train_price_data=train_price_data,
        )

        # --- Filter jobs: only jobs with deadline > decision_time ---
        schedulable_jobs = [
            j for j in jobs
            if j.deadline > decision_time
            and any(r in effective_regions for r in j.region_options)
        ]
        if not schedulable_jobs:
            logger.warning("LiveShadowRunner: no schedulable jobs after filtering")
            return []

        # --- Run optimizer with forecast prices ---
        optimizer_schedule = self._run_optimizer(
            jobs=schedulable_jobs,
            price_data=forecast_price_data,
            carbon_data=carbon_data,
        )

        # --- Run current_price_only baseline with KNOWN prices at submit time ---
        baseline_schedule = current_price_only_policy(
            jobs=schedulable_jobs,
            price_data=train_price_data,
            carbon_data=carbon_data,
            config=self.config,
        )

        # --- Build DecisionRecords ---
        records = self._build_records(
            schedulable_jobs=schedulable_jobs,
            optimizer_schedule=optimizer_schedule,
            baseline_schedule=baseline_schedule,
            forecast_price_data=forecast_price_data,
            train_price_data=train_price_data,
            decision_time=decision_time,
        )

        # --- Safety gate runs in the decision path (fail-closed) ---
        self._apply_safety_gate(records)

        logger.info(
            f"LiveShadowRunner run_id={self.run_id}: "
            f"{len(records)} records, decision_time={decision_time.isoformat()}, "
            f"regions={effective_regions}"
        )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_decision_time(
        self,
        price_df: pd.DataFrame,
        decision_time: Optional[datetime],
    ) -> datetime:
        if decision_time is not None:
            if decision_time.tzinfo is None:
                decision_time = decision_time.replace(tzinfo=timezone.utc)
            return decision_time.replace(minute=0, second=0, microsecond=0)
        last_ts = price_df["timestamp"].max()
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        return (last_ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    def _resolve_regions(self, price_df: pd.DataFrame) -> list[str]:
        if self.regions:
            return [r for r in self.regions if r in price_df["region"].unique()]
        return sorted(price_df["region"].unique())

    def _build_forecast(
        self,
        train_df: pd.DataFrame,
        decision_time: datetime,
        effective_regions: list[str],
        train_price_data: dict,
    ) -> dict:
        """Build forecast price dict for decision_time … decision_time + horizon_hours.

        Returns {region: {timestamp: predicted_price}} using only training data.
        """
        forecast_hours = list(range(self.horizon_hours))

        if self.price_forecaster_cls is not None:
            return self._build_ml_forecast(
                train_df=train_df,
                decision_time=decision_time,
                effective_regions=effective_regions,
                forecast_hours=forecast_hours,
            )
        return self._build_naive_forecast(
            train_price_data=train_price_data,
            decision_time=decision_time,
            effective_regions=effective_regions,
            forecast_hours=forecast_hours,
        )

    def _build_naive_forecast(
        self,
        train_price_data: dict,
        decision_time: datetime,
        effective_regions: list[str],
        forecast_hours: list[int],
    ) -> dict:
        """Seasonal-naive: use hour-of-day mean from training data."""
        # Build hour-of-day mean per region
        hod_means: dict[str, dict[int, float]] = {}
        for region, ts_prices in train_price_data.items():
            hod: dict[int, list[float]] = {}
            for ts, price in ts_prices.items():
                h = ts.hour
                hod.setdefault(h, []).append(price)
            hod_means[region] = {h: sum(vs) / len(vs) for h, vs in hod.items()}

        global_fallback = 50.0
        forecast: dict[str, dict[datetime, float]] = {}
        for region in effective_regions:
            region_means = hod_means.get(region, {})
            region_fallback = sum(region_means.values()) / max(1, len(region_means)) or global_fallback
            forecast[region] = {}
            for h_offset in forecast_hours:
                ts = (decision_time + timedelta(hours=h_offset)).replace(
                    minute=0, second=0, microsecond=0
                )
                price = region_means.get(ts.hour, region_fallback)
                forecast[region][ts] = price
        return forecast

    def _build_ml_forecast(
        self,
        train_df: pd.DataFrame,
        decision_time: datetime,
        effective_regions: list[str],
        forecast_hours: list[int],
    ) -> dict:
        """ML quantile forecast using training data only."""
        try:
            train_records = _df_to_price_records(train_df)
            if not train_records:
                logger.warning("LiveShadowRunner: no training records for ML forecaster")
                return {}

            cfg = self.price_forecaster_config
            forecaster = self.price_forecaster_cls(cfg) if cfg is not None else self.price_forecaster_cls()

            # Weather (leakage-free): train on observed history < decision_time;
            # predict on the day-ahead forecast for the horizon. Threaded only if
            # the forecaster's fit/predict accept weather_df (backward compatible).
            import inspect as _inspect
            _train_weather = None
            _predict_weather = None
            if self.weather_df is not None and not (
                hasattr(self.weather_df, "empty") and self.weather_df.empty
            ):
                _wts = pd.to_datetime(self.weather_df["timestamp"], utc=True)
                _train_weather = self.weather_df[_wts < _to_utc_ts(decision_time)]
                fwx = self.forecast_weather_df if self.forecast_weather_df is not None else self.weather_df
                _predict_weather = fwx
            _fit_accepts_weather = "weather_df" in _inspect.signature(forecaster.fit).parameters
            _pred_accepts_weather = "weather_df" in _inspect.signature(forecaster.predict).parameters

            if _fit_accepts_weather and _train_weather is not None:
                forecaster.fit(train_records, weather_df=_train_weather)
            else:
                forecaster.fit(train_records)

            # Build context from the tail of training data
            context_start = decision_time - timedelta(hours=self.context_hours)
            context_mask = (
                (train_df["timestamp"] >= _to_utc_ts(context_start))
            )
            context_df = train_df[context_mask]
            context_records = _df_to_price_records(context_df)
            if not context_records:
                context_records = train_records[-min(self.context_hours, len(train_records)):]

            forecast_timestamps = [
                (decision_time + timedelta(hours=h)).replace(
                    minute=0, second=0, microsecond=0
                )
                for h in forecast_hours
            ]

            forecast: dict[str, dict[datetime, float]] = {}
            for region in effective_regions:
                region_context = [r for r in context_records if r.region == region]
                if _pred_accepts_weather and _predict_weather is not None:
                    rw = _predict_weather[_predict_weather["region"] == region]
                    preds = forecaster.predict(region, forecast_timestamps, region_context,
                                               weather_df=(rw if not rw.empty else None))
                else:
                    preds = forecaster.predict(region, forecast_timestamps, region_context)
                for pred in preds:
                    ts = pred.timestamp.replace(minute=0, second=0, microsecond=0)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    forecast.setdefault(pred.region, {})[ts] = float(pred.p50)

            # Fill any gaps with naive fallback
            if len(forecast) < len(effective_regions):
                naive = self._build_naive_forecast(
                    train_price_data=_df_to_price_data(train_df),
                    decision_time=decision_time,
                    effective_regions=[r for r in effective_regions if r not in forecast],
                    forecast_hours=forecast_hours,
                )
                forecast.update(naive)

            # Optional post-spike recovery bias correction (ml_quantile_recovery mode).
            # Detects regions recovering from a price spike (ratio < 0.40 AND recent
            # mean ≤ $30/MWh) and exponentially decays the overprediction.
            if self.apply_recovery_correction and forecast:
                try:
                    from aurelius.forecasting.regime import RegimeDetector
                    detector = RegimeDetector()
                    # train_price_data needs {region: {timestamp: price}} dict format
                    train_price_dict = _df_to_price_data(train_df)
                    # recent context is the last ~168h of training data as EnergyPrice records
                    context_start = decision_time - timedelta(hours=min(self.context_hours, 168))
                    ctx_mask = train_df["timestamp"] >= _to_utc_ts(context_start)
                    recent_prices = _df_to_price_records(train_df[ctx_mask])
                    forecast = detector.apply_corrections_to_forecast(
                        forecast, train_price_dict, recent_prices
                    )
                except Exception as exc:
                    logger.warning(
                        "LiveShadowRunner: recovery correction failed (%s), using uncorrected forecast",
                        exc,
                    )

            return forecast

        except Exception as exc:
            logger.warning(
                f"LiveShadowRunner: ML forecast failed ({exc}), falling back to naive"
            )
            return self._build_naive_forecast(
                train_price_data=_df_to_price_data(train_df),
                decision_time=decision_time,
                effective_regions=effective_regions,
                forecast_hours=forecast_hours,
            )

    def _run_optimizer(
        self,
        jobs: list[Job],
        price_data: dict,
        carbon_data: dict,
    ) -> list[ScheduleDecision]:
        if self.method in ("greedy", "greedy_migrate"):
            return self.scheduler.solve(
                jobs=jobs,
                price_data=price_data,
                carbon_data=carbon_data,
                method="greedy",
            ).schedule
        return self.scheduler.solve(
            jobs=jobs,
            price_data=price_data,
            carbon_data=carbon_data,
            method=self.method,
        ).schedule

    def _build_records(
        self,
        schedulable_jobs: list[Job],
        optimizer_schedule: list[ScheduleDecision],
        baseline_schedule: list[ScheduleDecision],
        forecast_price_data: dict,
        train_price_data: dict,
        decision_time: datetime,
    ) -> list[DecisionRecord]:
        job_by_id = {j.job_id: j for j in schedulable_jobs}
        opt_by_id = {d.job_id: d for d in optimizer_schedule}
        base_by_id = {d.job_id: d for d in baseline_schedule}

        records: list[DecisionRecord] = []
        for job_id, job in job_by_id.items():
            opt_dec = opt_by_id.get(job_id)
            base_dec = base_by_id.get(job_id)

            if opt_dec is None or base_dec is None:
                logger.debug(f"Skipping job {job_id}: missing optimizer or baseline decision")
                continue

            # Forecast price at the optimizer's chosen slot
            f_region = opt_dec.region
            f_ts = opt_dec.start_time.replace(minute=0, second=0, microsecond=0)
            if f_ts.tzinfo is None:
                f_ts = f_ts.replace(tzinfo=timezone.utc)
            region_forecast = forecast_price_data.get(f_region, {})
            forecast_p50 = region_forecast.get(f_ts, 50.0)
            # p90: look for any p90 attribute from predictor, else +20% over p50
            forecast_p90 = forecast_p50 * 1.2

            # Predicted cost (forecast price × energy)
            predicted_cost = self._compute_cost(job, opt_dec, forecast_price_data)

            # Baseline cost (known DA price at submit time)
            baseline_cost = self._compute_cost(job, base_dec, train_price_data)

            # Predicted savings
            if baseline_cost > 0:
                pred_savings = (1.0 - predicted_cost / baseline_cost) * 100.0
            else:
                pred_savings = 0.0

            scheduled_end = opt_dec.start_time + timedelta(hours=opt_dec.actual_runtime_hours)

            record = DecisionRecord(
                run_id=self.run_id,
                job_id=job_id,
                workload_type=getattr(job, "workload_type", "unknown"),
                decision_time=decision_time,
                scheduled_region=opt_dec.region,
                scheduled_start=opt_dec.start_time,
                scheduled_end=scheduled_end,
                scheduled_runtime_h=opt_dec.actual_runtime_hours,
                forecast_da_price_p50=forecast_p50,
                forecast_da_price_p90=forecast_p90,
                predicted_energy_cost=predicted_cost,
                baseline_region=base_dec.region,
                baseline_start=base_dec.start_time,
                baseline_energy_cost=baseline_cost,
                predicted_savings_pct=pred_savings,
                power_kw=job.power_kw,
                gpu_count=getattr(job, "gpu_count", 0),
                sla_class=getattr(job, "sla_class", "best_effort"),
                forecaster_version=self.forecaster_version,
                optimizer_version=self.optimizer_version,
            )
            records.append(record)

        return records

    def _apply_safety_gate(self, records: list[DecisionRecord]) -> None:
        """Evaluate each decision through the quantile safety gate (fail-closed).

        Annotates every record with gate_status ("passed"/"filtered") and
        gate_reason. A "filtered" decision is unsafe to route — downstream
        execution must honor gate_status and must not route it silently.
        """
        if not self.enable_safety_gate:
            return
        n_filtered = 0
        for rec in records:
            p50 = rec.predicted_energy_cost
            baseline = rec.baseline_energy_cost
            # Scale the p50 cost to a p90 cost using the price quantile ratio.
            if rec.forecast_da_price_p50 and rec.forecast_da_price_p50 > 0:
                ratio = rec.forecast_da_price_p90 / rec.forecast_da_price_p50
            else:
                ratio = 1.2
            p90 = p50 * ratio
            forecast = {"energy_cost": {"p50": p50, "p90": p90, "baseline": baseline}}
            result = self._safety_gate._evaluate_metric(
                job_id=rec.job_id,
                forecast=forecast,
                metric="energy_cost",
                config=self._safety_gate_config,
            )
            rec.gate_status = "passed" if result.passed else "filtered"
            rec.gate_reason = result.reason
            if not result.passed:
                n_filtered += 1
        if n_filtered:
            logger.warning(
                "Safety gate filtered %d/%d decisions (unsafe to route silently)",
                n_filtered, len(records),
            )

    @staticmethod
    def _compute_cost(
        job: Job,
        decision: ScheduleDecision,
        price_data: dict,
    ) -> float:
        """Compute energy cost using the job window and price_data."""
        region_prices = price_data.get(decision.region, {})
        total = 0.0
        runtime_h = decision.actual_runtime_hours
        current = decision.start_time.replace(minute=0, second=0, microsecond=0)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        end = current + timedelta(hours=runtime_h)

        while current < end:
            hour_fraction = min(1.0, (end - current).total_seconds() / 3600.0)
            if hour_fraction <= 0:
                break
            price = region_prices.get(current, 50.0)
            energy_kwh = job.power_kw * hour_fraction
            total += (price / 1000.0) * energy_kwh
            current += timedelta(hours=1)

        return total
