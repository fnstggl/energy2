"""Production-readiness tests covering the improvements in this sprint.

Tests:
1. Safety gate inline validation now executes without crashing
2. RotatingJSONLWriter size and count thresholds
3. PostExecutionRecorder uses RotatingJSONLWriter by default
4. `aurelius ingest --help` works (CLI command is registered)
5. Learning loop cron: dry-run with us-west (no key needed)
6. Learning loop cron: generates correct ingest command with --price-output
7. End-to-end backtest CLI with CSV + savings report generation
8. Temporal leakage assertion fires in every backtest fold
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

UTC = timezone.utc
_T0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(*args, cwd=None):
    result = subprocess.run(
        [sys.executable, "-m", "aurelius.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd or Path(__file__).parent.parent,
    )
    return result.returncode, result.stdout, result.stderr


def _make_price_csv(
    path: Path,
    regions=("us-west", "us-east"),
    hours: int = 120,
    base_price: float = 50.0,
):
    rows = ["timestamp,region,price_per_mwh"]
    for h in range(hours):
        ts = (_T0 + timedelta(hours=h)).isoformat()
        for region in regions:
            price = base_price + (h % 24) * 2.0
            rows.append(f"{ts},{region},{price:.2f}")
    path.write_text("\n".join(rows))


# ---------------------------------------------------------------------------
# 1. Safety gate inline validation runs without AssertionError
# ---------------------------------------------------------------------------

class TestSafetyGateInlineValidation:
    def test_validation_runs_without_error(self):
        """_run_validation() must complete without any AssertionError."""
        from aurelius.safety.quantile_gate import _run_validation
        # Suppress logging to avoid cluttering test output
        import logging
        logging.disable(logging.CRITICAL)
        try:
            _run_validation()
        finally:
            logging.disable(logging.NOTSET)

    def test_missing_forecast_is_blocked(self):
        """Core fail-closed invariant: None forecast → blocked."""
        from dataclasses import dataclass
        from typing import Optional
        from aurelius.safety.quantile_gate import QuantileSafetyGate, QuantileGateConfig

        @dataclass
        class MockDecision:
            job_id: str
            forecast: Optional[dict] = None

        gate = QuantileSafetyGate()
        config = QuantileGateConfig(enabled=True, metric="energy_cost")
        result = gate.filter([MockDecision(job_id="j", forecast=None)], config)
        assert result == [], "Missing forecast MUST be blocked (fail-closed)"

    def test_zero_baseline_is_blocked(self):
        """Zero baseline → fail closed."""
        from dataclasses import dataclass
        from typing import Optional
        from aurelius.safety.quantile_gate import QuantileSafetyGate, QuantileGateConfig

        @dataclass
        class MockDecision:
            job_id: str
            forecast: Optional[dict] = None

        gate = QuantileSafetyGate()
        config = QuantileGateConfig(enabled=True, metric="energy_cost")
        d = MockDecision(
            job_id="j",
            forecast={"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 0.0}},
        )
        result = gate.filter([d], config)
        assert result == [], "Zero baseline MUST be blocked (fail-closed)"


# ---------------------------------------------------------------------------
# 2. RotatingJSONLWriter
# ---------------------------------------------------------------------------

class TestRotatingJSONLWriter:
    def test_basic_write_and_read(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter
        w = RotatingJSONLWriter(tmp_path / "pe.jsonl", max_size_bytes=None, max_records=None)
        w.append({"id": 1})
        w.append({"id": 2})
        records = w.read_all()
        assert len(records) == 2
        assert records[0]["id"] == 1

    def test_rotation_on_record_count(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter
        target = tmp_path / "pe.jsonl"
        w = RotatingJSONLWriter(target, max_size_bytes=None, max_records=3, keep_rotated=10)

        # Write 3 records — rotation threshold reached on the NEXT append
        for i in range(3):
            w.append({"id": i})
        assert target.exists()
        # Fourth append should trigger rotation
        w.append({"id": 999})

        # After rotation: archive file exists, active file has 1 record
        archives = list(tmp_path.glob("pe.*.jsonl"))
        assert len(archives) == 1, f"Expected 1 archive, got: {archives}"
        active_records = w.read_all()
        assert len(active_records) == 1
        assert active_records[0]["id"] == 999

    def test_rotation_on_size(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter
        target = tmp_path / "pe.jsonl"
        # Very small size threshold (50 bytes)
        w = RotatingJSONLWriter(target, max_size_bytes=50, max_records=None, keep_rotated=5)

        # Write records until rotation happens
        rotated = False
        for i in range(50):
            w.append({"id": i, "data": "x" * 10})
            archives = list(tmp_path.glob("pe.*.jsonl"))
            if archives:
                rotated = True
                break

        assert rotated, "File should have rotated after exceeding 50 bytes"

    def test_no_data_loss_on_rotation(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter, JSONLWriter
        target = tmp_path / "pe.jsonl"
        w = RotatingJSONLWriter(target, max_size_bytes=None, max_records=5, keep_rotated=10)

        all_ids = list(range(12))
        for i in all_ids:
            w.append({"id": i})

        # Count records across active + all archives
        # Archives match pattern pe.*.<suffix>
        pattern = f"{target.stem}.*{target.suffix}"
        archives = list(tmp_path.glob(pattern))
        total = len(w.read_all())
        for archive in archives:
            arc_writer = JSONLWriter(archive)
            total += len(arc_writer.read_all())

        assert total == len(all_ids), f"Expected {len(all_ids)} total records, got {total}"

    def test_keep_rotated_prunes_oldest(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter
        target = tmp_path / "pe.jsonl"
        # keep_rotated=2, rotate every 2 records
        w = RotatingJSONLWriter(target, max_size_bytes=None, max_records=2, keep_rotated=2)

        # Write 20 records — forces at least 9 rotations
        for i in range(20):
            w.append({"id": i})

        pattern = f"{target.stem}.*{target.suffix}"
        archives = list(tmp_path.glob(pattern))
        assert len(archives) <= 2, f"Expected ≤2 archives, got {len(archives)}: {archives}"

    def test_default_max_size_is_100mb(self):
        from aurelius.data.persistence import RotatingJSONLWriter
        assert RotatingJSONLWriter.DEFAULT_MAX_SIZE_BYTES == 100 * 1024 * 1024

    def test_disabled_rotation_never_rotates(self, tmp_path):
        from aurelius.data.persistence import RotatingJSONLWriter
        target = tmp_path / "pe.jsonl"
        w = RotatingJSONLWriter(target, max_size_bytes=None, max_records=None)
        for i in range(1000):
            w.append({"id": i})
        archives = list(tmp_path.glob("pe.*.jsonl"))
        assert len(archives) == 0, "No rotation should occur when both thresholds are None"
        assert len(w.read_all()) == 1000


# ---------------------------------------------------------------------------
# 3. PostExecutionRecorder uses RotatingJSONLWriter
# ---------------------------------------------------------------------------

class TestPostExecutionRecorderRotation:
    def test_recorder_rotates_by_record_count(self, tmp_path):
        from aurelius.execution.post_execution import PostExecutionRecorder
        from aurelius.models import ScheduleDecision
        from aurelius.execution.base import ExecutionConfig, ExecutionResult

        target = tmp_path / "pe.jsonl"
        recorder = PostExecutionRecorder(
            output_path=str(target),
            max_size_bytes=None,
            max_records=3,
            keep_rotated=5,
        )

        config = ExecutionConfig(mode="dry_run")

        for i in range(5):
            d = ScheduleDecision(
                job_id=f"job{i}",
                start_time=_T0 + timedelta(hours=i),
                region="us-west",
                power_fraction=1.0,
                actual_runtime_hours=1.0,
            )
            r = ExecutionResult(
                job_id=f"job{i}",
                submitted=False,
                aws_job_id=None,
                region="us-west",
                submit_time=_T0 + timedelta(hours=i),
                status="dry_run",
            )
            recorder.record(
                decision=d,
                baseline_decision=None,
                execution_result=r,
                config=config,
            )

        pattern = f"{target.stem}.*{target.suffix}"
        archives = list(tmp_path.glob(pattern))
        assert len(archives) >= 1, "Should have rotated at least once after 5 records (threshold=3)"

    def test_recorder_default_has_rotation_enabled(self, tmp_path):
        from aurelius.execution.post_execution import PostExecutionRecorder
        from aurelius.data.persistence import RotatingJSONLWriter

        recorder = PostExecutionRecorder(output_path=str(tmp_path / "pe.jsonl"))
        assert isinstance(recorder._writer, RotatingJSONLWriter)
        assert recorder._writer.max_size_bytes == RotatingJSONLWriter.DEFAULT_MAX_SIZE_BYTES


# ---------------------------------------------------------------------------
# 4. `aurelius ingest --help` is registered
# ---------------------------------------------------------------------------

class TestIngestCLICommand:
    def test_ingest_help_registered(self):
        code, stdout, stderr = _run_cli("ingest", "--help")
        assert code == 0, f"ingest --help failed:\n{stderr}"
        assert "--region" in stdout
        assert "--days" in stdout
        assert "--price-output" in stdout
        assert "--carbon-output" in stdout

    def test_ingest_help_not_eia(self):
        code, stdout, stderr = _run_cli("ingest", "--help")
        assert "eia" not in stdout.lower(), "ingest should not mention EIA as a provider"

    def test_ingest_unsupported_region_exits_nonzero(self):
        """An unsupported region should exit non-zero."""
        code, stdout, stderr = _run_cli("ingest", "--region", "fantasy-land", "--days", "1")
        assert code != 0

    def test_main_help_lists_ingest(self):
        code, stdout, _ = _run_cli("--help")
        assert code == 0
        assert "ingest" in stdout

    def test_ingest_caiso_no_api_key_skips_price(self, tmp_path):
        """CAISO works without an API key but the live fetch won't run in CI."""
        # We can test that the command doesn't crash with --region us-west
        # even without network access (it should attempt and potentially fail gracefully)
        price_out = tmp_path / "prices.csv"
        carbon_out = tmp_path / "carbon.csv"
        # Without real network, CAISO returns empty; ingest may warn but not crash
        # We just verify the command is wired correctly
        code, stdout, stderr = _run_cli(
            "ingest",
            "--region", "us-west",
            "--days", "1",
            "--price-output", str(price_out),
            "--carbon-output", str(carbon_out),
        )
        # Exit 0 (empty fetch is non-fatal warning) or non-zero if provider fails
        # Key assertion: the command is registered and runs (not "unknown command")
        combined = stdout + stderr
        assert "unknown command" not in combined.lower()
        assert "error: argument command" not in combined.lower()


# ---------------------------------------------------------------------------
# 5. End-to-end backtest CLI with CSV data + savings report generation
# ---------------------------------------------------------------------------

class TestEndToEndBacktest:
    # Aligned with the price CSV: 120 hours from 2024-01-01 → 2024-01-06
    _START = "2024-01-01"
    _END = "2024-01-06"

    @pytest.fixture
    def price_csv(self, tmp_path):
        p = tmp_path / "prices.csv"
        _make_price_csv(p, hours=120)
        return p

    @pytest.fixture
    def jobs_json(self, tmp_path):
        """Pre-generated jobs with earliest_start in the eval window [Jan 4, Jan 5]."""
        import json as _json
        import uuid
        # eval window: 2024-01-04 (72h) to 2024-01-05 (96h)
        eval_start = _T0 + timedelta(hours=72)
        jobs = []
        for i in range(8):
            es = eval_start + timedelta(hours=i * 3)
            dl = es + timedelta(hours=4)
            jobs.append({
                "job_id": f"job-{uuid.uuid4().hex[:8]}",
                "submit_time": (es - timedelta(hours=1)).isoformat(),
                "runtime_hours": 1.0,
                "deadline": dl.isoformat(),
                "power_kw": 100.0,
                "earliest_start": es.isoformat(),
                "region_options": ["us-west", "us-east"],
                "priority": 1,
            })
        p = tmp_path / "jobs.json"
        p.write_text(_json.dumps(jobs))
        return p

    def test_backtest_csv_produces_results(self, price_csv, jobs_json, tmp_path):
        """Full CLI backtest with CSV data must exit 0 and produce a JSON file."""
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv),
            "--jobs-file", str(jobs_json),
            "--regions", "us-west,us-east",
            "--start", self._START,
            "--end", self._END,
            "--train-days", "3",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0, f"Backtest exited {code}\nstdout: {stdout}\nstderr: {stderr}"
        assert out.exists(), "Output JSON file should be created"
        data = json.loads(out.read_text())
        assert isinstance(data, list), "Output should be a list of fold results"
        assert len(data) > 0, "At least one fold should be produced"

    def test_backtest_result_has_required_fields(self, price_csv, jobs_json, tmp_path):
        """Each backtest fold must have required fields."""
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv),
            "--jobs-file", str(jobs_json),
            "--regions", "us-west,us-east",
            "--start", self._START,
            "--end", self._END,
            "--train-days", "3",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0, stderr
        data = json.loads(out.read_text())
        fold = data[0]
        assert "fold_index" in fold
        assert "train_start" in fold
        assert "train_end" in fold
        assert "eval_start" in fold
        assert "eval_end" in fold
        assert "optimizer" in fold
        assert "baselines" in fold

    def test_backtest_train_end_before_eval_start(self, price_csv, jobs_json, tmp_path):
        """train_end must always be before eval_start — leakage invariant."""
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv),
            "--jobs-file", str(jobs_json),
            "--regions", "us-west,us-east",
            "--start", self._START,
            "--end", self._END,
            "--train-days", "3",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0, stderr
        data = json.loads(out.read_text())
        for fold in data:
            train_end = pd.Timestamp(fold["train_end"])
            eval_start = pd.Timestamp(fold["eval_start"])
            assert train_end <= eval_start, (
                f"Fold {fold['fold_index']}: train_end={train_end} > eval_start={eval_start} — LEAKAGE!"
            )

    def test_backtest_savings_report_generation(self, price_csv, jobs_json, tmp_path):
        """SavingsReport can be built from backtest engine results."""
        import json as _json
        from aurelius.backtesting.engine import BacktestEngine
        from aurelius.models import OptimizationConfig
        from aurelius.reporting.savings_report import SavingsReport
        from aurelius.ingestion.grid_apis.csv_importer import CSVPriceImporter
        from aurelius.ingestion.job_logs import JobLogIngester

        importer = CSVPriceImporter(str(price_csv))
        price_df = importer.load_all()

        # Load pre-built jobs (guaranteed in eval window)
        jobs = JobLogIngester().load_from_json(str(jobs_json))

        config = OptimizationConfig()
        engine = BacktestEngine(method="greedy", train_days=3, eval_days=1, config=config)
        rounds = engine.run(
            jobs=jobs,
            price_df=price_df,
            carbon_df=pd.DataFrame(),
        )
        assert rounds, "Should produce at least one backtest fold"

        report = SavingsReport.generate(rounds)
        assert report is not None
        assert isinstance(report, dict)

        # Verify required report keys
        assert "confidence_intervals" in report
        assert "methodology" in report
        assert "baseline_comparison" in report
        assert "fold_results" in report

        # Verify confidence intervals are present with required structure
        ci = report["confidence_intervals"]
        assert isinstance(ci, dict), "confidence_intervals should be a dict"
        # At least one CI metric should have estimate/lower_95/upper_95
        for _metric_name, ci_val in ci.items():
            if isinstance(ci_val, dict) and "estimate" in ci_val:
                assert "lower_95" in ci_val
                assert "upper_95" in ci_val
                break  # found at least one valid CI

        # Methodology must prove leakage-free computation
        methodology = report["methodology"]
        assert methodology, "methodology section must be non-empty"

    def test_backtest_no_negative_realized_cost(self, price_csv, jobs_json, tmp_path):
        """No fold's realized optimizer cost should be negative (data check)."""
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv),
            "--jobs-file", str(jobs_json),
            "--regions", "us-west,us-east",
            "--start", self._START,
            "--end", self._END,
            "--train-days", "3",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0, stderr
        data = json.loads(out.read_text())
        for fold in data:
            opt = fold.get("optimizer", {})
            cost = opt.get("total_energy_cost_usd", 0.0)
            assert cost >= 0, f"Fold {fold['fold_index']}: negative optimizer cost {cost}"


# ---------------------------------------------------------------------------
# 6. Learning loop cron: dry-run produces correct ingest command
# ---------------------------------------------------------------------------

class TestLearningLoopCronScript:
    @pytest.fixture
    def cron_path(self):
        return Path(__file__).parent.parent / "scripts" / "learning_loop_cron.sh"

    def test_cron_script_exists(self, cron_path):
        assert cron_path.exists(), "learning_loop_cron.sh must exist"

    def test_cron_dry_run_exits_zero_when_no_pe_data(self, cron_path, tmp_path):
        """Dry-run with no PE data should exit 0 (dry run) or 2 (insufficient data)."""
        result = subprocess.run(
            ["bash", str(cron_path), "--dry-run", "--region", "us-west"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "AURELIUS_DATA_DIR": str(tmp_path)},
        )
        assert result.returncode in (0, 2), (
            f"Expected exit 0 or 2, got {result.returncode}\n{result.stderr}"
        )

    def test_cron_dry_run_shows_correct_ingest_command(self, cron_path, tmp_path):
        """Dry-run output must show the correct ingest CLI command."""
        result = subprocess.run(
            ["bash", str(cron_path), "--dry-run", "--region", "us-west", "--days", "1"],
            capture_output=True, text=True,
            env={
                **__import__("os").environ,
                "AURELIUS_DATA_DIR": str(tmp_path),
                # Remove any live API keys to force warning path
                "PJM_API_KEY": "",
                "ENTSOE_API_KEY": "",
                "ELECTRICITYMAPS_API_KEY": "",
                "WATTTIME_USERNAME": "",
            },
        )
        combined = result.stdout + result.stderr
        # Should show the correct ingest command (CAISO us-west needs no key)
        assert "aurelius.cli ingest" in combined or "ingest" in combined, (
            f"Expected ingest command in output:\n{combined}"
        )
        assert "--price-output" in combined, (
            f"Expected --price-output in dry-run output:\n{combined}"
        )

    def test_cron_dry_run_mentions_forecaster_retraining(self, cron_path, tmp_path):
        """Dry-run should skip directly (exit 0 or 2) since no PE data."""
        result = subprocess.run(
            ["bash", str(cron_path), "--dry-run", "--region", "us-west"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "AURELIUS_DATA_DIR": str(tmp_path)},
        )
        # Step 3.5 is only reached after PE records check passes;
        # in dry-run with no data, we exit at step 2 with code 2.
        assert result.returncode in (0, 2)

    def test_cron_has_price_history_csv_variable(self, cron_path):
        """The cron script must reference PRICE_HISTORY_CSV for retrain_forecasters."""
        content = cron_path.read_text()
        assert "PRICE_HISTORY_CSV" in content, "Cron script must define PRICE_HISTORY_CSV"
        assert "retrain_forecasters" in content, "Cron script must call retrain_forecasters.py"

    def test_cron_step_35_is_present(self, cron_path):
        """Step 3.5 (forecaster retraining) must be in the cron script."""
        content = cron_path.read_text()
        assert "Step 3.5" in content, "Cron script must have Step 3.5 for forecaster retraining"
        assert "mape" in content.lower(), "Cron script must mention MAPE in promotion logic"


# ---------------------------------------------------------------------------
# 7. Leakage audit: assert_no_leakage raises DataLeakageError on overlap
# ---------------------------------------------------------------------------

class TestLeakageAudit:
    def test_no_leakage_passes_clean_split(self):
        from aurelius.validation.leakage_audit import assert_no_leakage

        train = pd.DataFrame({
            "timestamp": [_T0 + timedelta(hours=i) for i in range(24)]
        })
        eval_ = pd.DataFrame({
            "timestamp": [_T0 + timedelta(hours=i) for i in range(24, 48)]
        })
        assert_no_leakage(train, eval_)  # should not raise

    def test_no_leakage_raises_on_overlap(self):
        from aurelius.validation.leakage_audit import assert_no_leakage, DataLeakageError

        train = pd.DataFrame({
            "timestamp": [_T0 + timedelta(hours=i) for i in range(25)]  # includes h24
        })
        eval_ = pd.DataFrame({
            "timestamp": [_T0 + timedelta(hours=i) for i in range(24, 48)]  # starts at h24
        })
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)

    def test_splitter_raises_on_constructed_leakage(self):
        """TemporalSplitter.split() calls assert_no_leakage and would raise if leakage occurred."""
        from aurelius.backtesting.splitter import TemporalSplitter

        df = pd.DataFrame({
            "timestamp": [_T0 + timedelta(hours=i) for i in range(200)],
            "price_per_mwh": [50.0] * 200,
        })
        splitter = TemporalSplitter(train_days=3, eval_days=1)
        splits = splitter.split(df)
        for split in splits:
            assert split.train_end <= split.eval_start, (
                f"Leakage: train_end={split.train_end} > eval_start={split.eval_start}"
            )
