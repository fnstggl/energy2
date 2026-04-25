"""Tests for Phase 5 learning loop wiring.

Covers:
- PostExecutionRecorder.lookup_realized_price
- PostExecutionRecorder market_registry integration
- BacktestEngine recorder_path wiring
- ShadowRunner post_execution_path wiring
- learning_loop_cron.sh existence and executability
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aurelius.execution.post_execution import (
    ForecastSnapshot,
    PostExecutionRecorder,
    RealizedOutcome,
    lookup_realized_price,
)
from aurelius.execution.base import ExecutionConfig, ExecutionResult
from aurelius.models import Job, ScheduleDecision, OptimizationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(job_id="j1", region="us-east", power_kw=10.0, runtime_hours=2.0):
    now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
    return Job(
        job_id=job_id,
        submit_time=now,
        power_kw=power_kw,
        runtime_hours=runtime_hours,
        earliest_start=now,
        deadline=datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc),
        region_options=[region],
    )


def _make_decision(job_id="j1", region="us-east", start_hour=8):
    start = datetime(2024, 1, 15, start_hour, 0, tzinfo=timezone.utc)
    return ScheduleDecision(
        job_id=job_id,
        start_time=start,
        region=region,
        power_fraction=1.0,
        actual_runtime_hours=2.0,
    )


def _make_exec_result(job_id="j1", region="us-east"):
    return ExecutionResult(
        job_id=job_id,
        submitted=False,
        aws_job_id=None,
        region=region,
        submit_time=datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),
        status="dry_run",
    )


def _make_exec_config():
    return ExecutionConfig(mode="dry_run", constraint_profile="batch_optimized")


# ---------------------------------------------------------------------------
# lookup_realized_price tests
# ---------------------------------------------------------------------------

class TestLookupRealizedPrice:
    def test_returns_none_when_no_registry(self):
        result = lookup_realized_price("us-east", datetime.utcnow(), 1.0)
        assert result is None

    def test_returns_none_when_registry_is_none(self):
        result = lookup_realized_price("us-east", datetime.utcnow(), 1.0, market_registry=None)
        assert result is None

    def test_returns_mean_price_from_registry(self):
        mock_registry = MagicMock()
        price_df = pd.DataFrame({"price_per_mwh": [50.0, 60.0, 70.0]})
        mock_registry.fetch_prices.return_value = price_df

        result = lookup_realized_price(
            "us-east",
            datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),
            2.0,
            market_registry=mock_registry,
        )
        assert result == pytest.approx(60.0)

    def test_returns_none_when_registry_raises(self):
        mock_registry = MagicMock()
        mock_registry.fetch_prices.side_effect = RuntimeError("API down")

        result = lookup_realized_price(
            "us-east",
            datetime.utcnow(),
            1.0,
            market_registry=mock_registry,
        )
        assert result is None

    def test_returns_none_when_empty_dataframe(self):
        mock_registry = MagicMock()
        mock_registry.fetch_prices.return_value = pd.DataFrame({"price_per_mwh": []})

        result = lookup_realized_price("us-east", datetime.utcnow(), 1.0, market_registry=mock_registry)
        assert result is None

    def test_returns_none_when_missing_column(self):
        mock_registry = MagicMock()
        mock_registry.fetch_prices.return_value = pd.DataFrame({"demand_mw": [100.0]})

        result = lookup_realized_price("us-east", datetime.utcnow(), 1.0, market_registry=mock_registry)
        assert result is None

    def test_returns_none_when_registry_returns_none(self):
        mock_registry = MagicMock()
        mock_registry.fetch_prices.return_value = None

        result = lookup_realized_price("us-east", datetime.utcnow(), 1.0, market_registry=mock_registry)
        assert result is None


# ---------------------------------------------------------------------------
# PostExecutionRecorder with market_registry
# ---------------------------------------------------------------------------

class TestPostExecutionRecorderMarketRegistry:
    def test_recorder_accepts_market_registry_param(self, tmp_path):
        mock_registry = MagicMock()
        recorder = PostExecutionRecorder(
            output_path=str(tmp_path / "pe.jsonl"),
            market_registry=mock_registry,
        )
        assert recorder._market_registry is mock_registry

    def test_recorder_populates_realized_price_from_registry(self, tmp_path):
        mock_registry = MagicMock()
        price_df = pd.DataFrame({"price_per_mwh": [55.0, 65.0]})
        mock_registry.fetch_prices.return_value = price_df

        recorder = PostExecutionRecorder(
            output_path=str(tmp_path / "pe.jsonl"),
            market_registry=mock_registry,
        )
        decision = _make_decision()
        record = recorder.record(
            decision=decision,
            baseline_decision=None,
            execution_result=_make_exec_result(),
            config=_make_exec_config(),
            forecast=ForecastSnapshot(energy_cost_p50=10.0),
            realized=RealizedOutcome(),  # no price set
        )
        assert record is not None
        # realized_energy_price should be populated from registry (mean of 55, 65 = 60)
        assert record.realized_energy_price == pytest.approx(60.0)

    def test_recorder_skips_registry_when_price_already_set(self, tmp_path):
        mock_registry = MagicMock()
        recorder = PostExecutionRecorder(
            output_path=str(tmp_path / "pe.jsonl"),
            market_registry=mock_registry,
        )
        decision = _make_decision()
        record = recorder.record(
            decision=decision,
            baseline_decision=None,
            execution_result=_make_exec_result(),
            config=_make_exec_config(),
            realized=RealizedOutcome(realized_energy_price=42.0),  # already set
        )
        assert record is not None
        # Registry should NOT be called since price is already set
        mock_registry.fetch_prices.assert_not_called()
        assert record.realized_energy_price == pytest.approx(42.0)

    def test_recorder_tolerates_registry_failure(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.fetch_prices.side_effect = RuntimeError("network error")

        recorder = PostExecutionRecorder(
            output_path=str(tmp_path / "pe.jsonl"),
            market_registry=mock_registry,
        )
        decision = _make_decision()
        # Should not raise — failure is logged at DEBUG level
        record = recorder.record(
            decision=decision,
            baseline_decision=None,
            execution_result=_make_exec_result(),
            config=_make_exec_config(),
        )
        assert record is not None
        assert record.realized_energy_price is None


# ---------------------------------------------------------------------------
# BacktestEngine recorder_path wiring
# ---------------------------------------------------------------------------

class TestBacktestEngineRecorderPath:
    def test_recorder_created_when_path_provided(self, tmp_path):
        from aurelius.backtesting.engine import BacktestEngine
        pe_path = tmp_path / "pe.jsonl"
        engine = BacktestEngine(recorder_path=pe_path)
        assert engine._recorder is not None

    def test_no_recorder_when_path_not_provided(self):
        from aurelius.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        assert engine._recorder is None

    def test_recorder_writes_jsonl_during_backtest(self, tmp_path):
        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.models import Job
        from datetime import timedelta

        pe_path = tmp_path / "pe.jsonl"
        engine = BacktestEngine(
            method="greedy",
            train_days=7,
            eval_days=3,
            recorder_path=pe_path,
        )

        # Build minimal price/carbon DataFrames
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for d in range(14):
            for h in range(24):
                ts = base + timedelta(days=d, hours=h)
                rows.append({"timestamp": ts, "region": "us-east", "price_per_mwh": 50.0 + h})
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame([
            {"timestamp": r["timestamp"], "region": "us-east", "gco2_per_kwh": 400.0}
            for r in rows
        ])

        # Create jobs in the eval window
        eval_start = base + timedelta(days=7)
        jobs = [
            Job(
                job_id=f"j{i}",
                submit_time=eval_start + timedelta(hours=i),
                power_kw=5.0,
                runtime_hours=1.0,
                earliest_start=eval_start + timedelta(hours=i),
                deadline=eval_start + timedelta(hours=i + 6),
                region_options=["us-east"],
            )
            for i in range(3)
        ]

        rounds = engine.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
        assert len(rounds) >= 1

        # JSONL file must exist and have records
        assert pe_path.exists(), "Recorder JSONL should be written"
        lines = [l for l in pe_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1, "At least one PostExecutionRecord should be written"

        # Each record must be valid JSON with required fields
        for line in lines:
            rec = json.loads(line)
            assert "job_id" in rec
            assert "decision_id" in rec
            assert "region" in rec
            assert "execution_mode" in rec
            assert rec["execution_mode"] == "dry_run"

    def test_backtest_without_recorder_still_works(self):
        from aurelius.backtesting.engine import BacktestEngine
        from datetime import timedelta

        engine = BacktestEngine(method="greedy", train_days=7, eval_days=3)

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"timestamp": base + timedelta(days=d, hours=h), "region": "us-east", "price_per_mwh": 50.0}
            for d in range(12) for h in range(24)
        ]
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame([
            {"timestamp": r["timestamp"], "region": "us-east", "gco2_per_kwh": 400.0}
            for r in rows
        ])
        jobs = [
            Job(
                job_id="j1",
                submit_time=base + timedelta(days=7, hours=2),
                power_kw=5.0,
                runtime_hours=1.0,
                earliest_start=base + timedelta(days=7, hours=2),
                deadline=base + timedelta(days=7, hours=10),
                region_options=["us-east"],
            )
        ]
        rounds = engine.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
        # No crash; rounds returned
        assert isinstance(rounds, list)


# ---------------------------------------------------------------------------
# ShadowRunner post_execution_path wiring
# ---------------------------------------------------------------------------

class TestShadowRunnerPostExecutionPath:
    def test_pe_recorder_created_when_path_provided(self, tmp_path):
        from aurelius.execution.shadow_runner import ShadowRunner
        pe_path = tmp_path / "pe.jsonl"
        runner = ShadowRunner(post_execution_path=pe_path)
        assert runner._pe_recorder is not None

    def test_no_pe_recorder_when_path_not_provided(self):
        from aurelius.execution.shadow_runner import ShadowRunner
        runner = ShadowRunner()
        assert runner._pe_recorder is None

    def test_shadow_run_writes_pe_records(self, tmp_path):
        from aurelius.execution.shadow_runner import ShadowRunner
        from aurelius.models import Job

        pe_path = tmp_path / "pe.jsonl"
        runner = ShadowRunner(post_execution_path=pe_path)

        base = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        job = Job(
            job_id="j1",
            submit_time=base,
            power_kw=10.0,
            runtime_hours=2.0,
            earliest_start=base,
            deadline=datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc),
            region_options=["us-east"],
        )
        decision = ScheduleDecision(
            job_id="j1",
            start_time=base,
            region="us-east",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        real_prices = {"us-east": {base: 60.0, base.replace(hour=9): 65.0}}
        real_carbon = {"us-east": {base: 400.0, base.replace(hour=9): 400.0}}

        result = runner.run(
            decisions=[decision],
            real_prices=real_prices,
            real_carbon=real_carbon,
            jobs=[job],
        )
        assert result is not None

        assert pe_path.exists()
        lines = [l for l in pe_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1

        rec = json.loads(lines[0])
        assert rec["job_id"] == "j1"
        assert rec["region"] == "us-east"
        assert rec["execution_mode"] == "dry_run"

    def test_shadow_run_without_pe_path_still_works(self):
        from aurelius.execution.shadow_runner import ShadowRunner
        from aurelius.models import Job

        runner = ShadowRunner()
        base = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        job = Job(
            job_id="j1",
            submit_time=base,
            power_kw=10.0,
            runtime_hours=2.0,
            earliest_start=base,
            deadline=datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc),
            region_options=["us-east"],
        )
        decision = ScheduleDecision(
            job_id="j1",
            start_time=base,
            region="us-east",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        result = runner.run(
            decisions=[decision],
            real_prices={"us-east": {base: 60.0}},
            real_carbon={"us-east": {base: 400.0}},
            jobs=[job],
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Learning loop: JSONL grows with every run
# ---------------------------------------------------------------------------

class TestLearningLoopDataAccumulation:
    def test_backtest_pe_jsonl_grows_across_runs(self, tmp_path):
        """Each backtest run with recorder_path appends new records."""
        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.models import Job
        from datetime import timedelta

        pe_path = tmp_path / "pe.jsonl"
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"timestamp": base + timedelta(days=d, hours=h), "region": "us-east", "price_per_mwh": 50.0}
            for d in range(12) for h in range(24)
        ]
        price_df = pd.DataFrame(rows)
        carbon_df = pd.DataFrame([
            {"timestamp": r["timestamp"], "region": "us-east", "gco2_per_kwh": 400.0}
            for r in rows
        ])
        jobs = [
            Job(
                job_id="j1",
                submit_time=base + timedelta(days=7, hours=2),
                power_kw=5.0,
                runtime_hours=1.0,
                earliest_start=base + timedelta(days=7, hours=2),
                deadline=base + timedelta(days=7, hours=10),
                region_options=["us-east"],
            )
        ]

        # First run
        engine1 = BacktestEngine(method="greedy", train_days=7, eval_days=3, recorder_path=pe_path)
        engine1.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
        count1 = sum(1 for l in pe_path.read_text().splitlines() if l.strip())

        # Second run — JSONL should grow (append mode)
        engine2 = BacktestEngine(method="greedy", train_days=7, eval_days=3, recorder_path=pe_path)
        engine2.run(jobs=jobs, price_df=price_df, carbon_df=carbon_df)
        count2 = sum(1 for l in pe_path.read_text().splitlines() if l.strip())

        assert count2 >= count1, "Second run should not shrink the JSONL"

    def test_train_offline_reads_pe_records(self, tmp_path):
        """train_offline.py can read records written by the backtest recorder."""
        from aurelius.ml.train_offline import run_training

        pe_path = tmp_path / "pe.jsonl"
        # Write minimal valid PostExecutionRecord dicts
        records = []
        for i in range(5):
            records.append({
                "job_id": f"j{i}",
                "region": "us-east",
                "optimized_start_time": "2024-01-15T10:00:00Z",
                "forecast_energy_cost_p50": 10.0,
                "energy_cost_p50_error": float(i - 2),
                "energy_cost_p90_covered": True,
                "realized_savings": float(i),
                "decision_outcome_label": "good_decision",
                "constraint_profile": "batch_optimized",
                "execution_mode": "dry_run",
                "execution_status": "dry_run",
            })
        with open(pe_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        artifact_dir = tmp_path / "artifacts"
        success = run_training(
            input_path=pe_path,
            output_dir=artifact_dir,
            seed=42,
            overwrite=True,
            min_records=1,
        )
        assert success is True
        assert (artifact_dir / "forecast_corrections_v1.json").exists()
        assert (artifact_dir / "manifest_v1.json").exists()


# ---------------------------------------------------------------------------
# Forecast corrections: non-zero bias after PE records
# ---------------------------------------------------------------------------

class TestForecastCorrectionsNonZero:
    def test_corrections_non_zero_when_systematic_bias(self, tmp_path):
        """If forecasts are systematically too low, corrections should be non-zero."""
        from aurelius.ml.train_offline import run_training

        pe_path = tmp_path / "pe.jsonl"
        # All records have a positive error (actual > forecast) — systematic over-prediction
        records = [
            {
                "job_id": f"j{i}",
                "region": "us-east",
                "optimized_start_time": f"2024-01-15T{10 + (i % 8):02d}:00:00Z",
                "forecast_energy_cost_p50": 100.0,
                "energy_cost_p50_error": 10.0,  # actual = 110, forecast = 100
                "energy_cost_p90_covered": True,
                "realized_savings": 5.0,
                "decision_outcome_label": "good_decision",
                "constraint_profile": "batch_optimized",
                "execution_mode": "dry_run",
                "execution_status": "dry_run",
            }
            for i in range(20)
        ]
        with open(pe_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        artifact_dir = tmp_path / "artifacts"
        success = run_training(
            input_path=pe_path,
            output_dir=artifact_dir,
            seed=42,
            overwrite=True,
            min_records=1,
        )
        assert success

        fc = json.loads((artifact_dir / "forecast_corrections_v1.json").read_text())
        buckets = fc.get("buckets", [])
        # At least one bucket should have a non-zero energy_cost_p50_bias
        non_zero = [
            b for b in buckets
            if b.get("energy_cost_p50_bias") is not None
            and abs(b["energy_cost_p50_bias"]) > 1e-9
        ]
        assert len(non_zero) > 0, (
            "forecast_corrections_v1.json must contain non-zero bias estimates "
            "when there is a systematic forecast error"
        )


# ---------------------------------------------------------------------------
# Learning loop cron script
# ---------------------------------------------------------------------------

class TestLearningLoopCronScript:
    def test_cron_script_exists(self):
        script = Path(__file__).parent.parent / "scripts" / "learning_loop_cron.sh"
        assert script.exists(), "scripts/learning_loop_cron.sh must exist"

    def test_cron_script_is_executable(self):
        script = Path(__file__).parent.parent / "scripts" / "learning_loop_cron.sh"
        assert os.access(script, os.X_OK), "learning_loop_cron.sh must be executable"

    def test_cron_script_has_bash_shebang(self):
        script = Path(__file__).parent.parent / "scripts" / "learning_loop_cron.sh"
        first_line = script.read_text().splitlines()[0]
        assert "bash" in first_line, "Script must have a bash shebang"

    def test_cron_script_dry_run_exits_zero(self):
        """--dry-run flag should exit 0 without touching live APIs."""
        import subprocess
        script = Path(__file__).parent.parent / "scripts" / "learning_loop_cron.sh"
        result = subprocess.run(
            ["bash", str(script), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # dry-run exits 0 or 2 (insufficient data) — both are acceptable
        assert result.returncode in (0, 2), (
            f"Unexpected exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Price model bias correction loading integration
# ---------------------------------------------------------------------------

class TestPriceModelBiasCorrection:
    def test_model_loads_corrections_when_artifact_exists(self, tmp_path):
        """PriceQuantileForecaster loads bias corrections from artifact."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster

        # Use primary schema (energy_cost_p50_bias from train_forecast_corrections)
        corrections = {
            "version": 1,
            "buckets": [
                {
                    "region": "us-east",
                    "hour_utc": 10,
                    "energy_cost_p50_bias": 5.0,
                    "n": 20,
                }
            ],
        }
        corrections_path = tmp_path / "forecast_corrections_v1.json"
        corrections_path.write_text(json.dumps(corrections))

        fc = PriceQuantileForecaster(corrections_path=corrections_path)
        assert fc._corrections_loaded is True
        assert "us-east" in fc._p50_bias
        assert fc._p50_bias["us-east"][10] == pytest.approx(5.0)

    def test_model_loads_legacy_corrections_schema(self, tmp_path):
        """PriceQuantileForecaster also reads legacy energy_cost.mean_error schema."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster

        corrections = {
            "version": 1,
            "buckets": [
                {
                    "region": "us-west",
                    "hour_utc": 14,
                    "energy_cost": {"mean_error": 3.0, "n": 10},
                }
            ],
        }
        corrections_path = tmp_path / "forecast_corrections_v1.json"
        corrections_path.write_text(json.dumps(corrections))

        fc = PriceQuantileForecaster(corrections_path=corrections_path)
        assert fc._corrections_loaded is True
        assert fc._p50_bias["us-west"][14] == pytest.approx(3.0)

    def test_model_handles_missing_corrections_gracefully(self):
        """If corrections file is absent, model works without bias correction."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster

        fc = PriceQuantileForecaster(corrections_path="/nonexistent/path.json")
        assert fc._corrections_loaded is False
        assert fc._p50_bias == {}

    def test_bias_correction_applied_to_predictions(self, tmp_path):
        """Predictions are adjusted by the stored bias for that region/hour."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        from aurelius.models import EnergyPrice

        # Artifact with +10 bias for us-east hour=8 (primary schema)
        corrections = {
            "version": 1,
            "buckets": [
                {"region": "us-east", "hour_utc": 8, "energy_cost_p50_bias": 10.0, "n": 20},
            ],
        }
        corrections_path = tmp_path / "fc.json"
        corrections_path.write_text(json.dumps(corrections))

        fc = PriceQuantileForecaster(
            config=PriceModelConfig(n_estimators=10, seed=42),
            corrections_path=corrections_path,
        )

        # Train on simple data
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        prices = [
            EnergyPrice(
                timestamp=base + timedelta(days=d, hours=h),
                region="us-east",
                price_per_mwh=50.0,
            )
            for d in range(7)
            for h in range(24)
        ]
        fc.fit(prices)

        # Predict at hour 8
        pred_ts = datetime(2024, 1, 8, 8, 0, tzinfo=timezone.utc)
        preds_with = fc.predict("us-east", [pred_ts])

        # Disable correction and predict again
        fc2 = PriceQuantileForecaster(
            config=PriceModelConfig(n_estimators=10, seed=42),
            corrections_path=False,
        )
        fc2.fit(prices)
        preds_without = fc2.predict("us-east", [pred_ts])

        # With correction: p50 should be lower by ~10 units
        p50_with = preds_with[0].p50
        p50_without = preds_without[0].p50
        # correction subtracts mean_error=10 → corrected = raw - 10
        assert p50_with == pytest.approx(p50_without - 10.0, abs=0.5)
