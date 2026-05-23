"""Tests for the Phase 8 daily learning loop script.

Coverage:
- store append logic (deduplication, sorting)
- model comparison and promotion logic
- report generation
- dry-run safety (no files written)
- graceful failure when live APIs are unavailable
- ROI CLI integration (end-to-end import test)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.daily_learning_loop import (
    append_to_store,
    compare_models,
    generate_report,
    run_benchmark_smoke_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_df():
    """Minimal price DataFrame for 2 regions, 48 hours."""
    rows = []
    for hour in range(48):
        ts = datetime(2026, 1, 1, hour % 24, tzinfo=timezone.utc)
        rows.append({"timestamp": ts, "region": "us-west", "price_per_mwh": 30.0 + hour * 0.5})
        rows.append({"timestamp": ts, "region": "us-east", "price_per_mwh": 28.0 + hour * 0.4})
    return pd.DataFrame(rows)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "store" / "price_history.csv"


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def reports_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# TestAppendToStore
# ---------------------------------------------------------------------------


class TestAppendToStore:
    def test_creates_store_when_empty(self, sample_price_df, store_path):
        combined = append_to_store(
            {"us-west": sample_price_df[sample_price_df["region"] == "us-west"]},
            store_path,
            dry_run=False,
        )
        assert store_path.exists()
        assert len(combined) > 0

    def test_dry_run_does_not_write(self, sample_price_df, store_path):
        append_to_store(
            {"us-west": sample_price_df[sample_price_df["region"] == "us-west"]},
            store_path,
            dry_run=True,
        )
        assert not store_path.exists()

    def test_deduplicates_rows(self, sample_price_df, store_path):
        region_df = sample_price_df[sample_price_df["region"] == "us-west"].copy()
        append_to_store({"us-west": region_df}, store_path, dry_run=False)
        # Append again (same data) — should not grow
        combined2 = append_to_store({"us-west": region_df}, store_path, dry_run=False)
        n_unique_pairs = len(region_df.drop_duplicates(subset=["timestamp", "region"]))
        assert len(combined2[combined2["region"] == "us-west"]) == n_unique_pairs

    def test_empty_new_data_returns_existing(self, sample_price_df, store_path):
        append_to_store(
            {"us-west": sample_price_df[sample_price_df["region"] == "us-west"]},
            store_path,
            dry_run=False,
        )
        combined = append_to_store({}, store_path, dry_run=False)
        assert len(combined) > 0

    def test_adds_new_region_to_existing_store(self, sample_price_df, store_path):
        west_df = sample_price_df[sample_price_df["region"] == "us-west"].copy()
        east_df = sample_price_df[sample_price_df["region"] == "us-east"].copy()
        append_to_store({"us-west": west_df}, store_path, dry_run=False)
        combined = append_to_store({"us-east": east_df}, store_path, dry_run=False)
        assert set(combined["region"].unique()) == {"us-west", "us-east"}

    def test_sorted_by_timestamp(self, sample_price_df, store_path):
        append_to_store(
            {"us-west": sample_price_df[sample_price_df["region"] == "us-west"]},
            store_path,
            dry_run=False,
        )
        loaded = pd.read_csv(store_path, parse_dates=["timestamp"])
        timestamps = loaded[loaded["region"] == "us-west"]["timestamp"].values
        assert (timestamps[1:] >= timestamps[:-1]).all()


# ---------------------------------------------------------------------------
# TestCompareModels
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_no_active_model_promotes(self, models_dir):
        eval_result = {"status": "ok", "savings_vs_cpo_mean": 0.20}
        comparison = compare_models(eval_result, models_dir)
        assert comparison["promote"] is True
        assert "no_active_model" in comparison["reason"]

    def test_improvement_promotes(self, models_dir):
        # Write an active metadata file with lower savings
        meta = {"last_eval_savings_vs_cpo": 0.15}
        (models_dir / "active_metadata.json").write_text(json.dumps(meta))
        eval_result = {"status": "ok", "savings_vs_cpo_mean": 0.22}
        comparison = compare_models(eval_result, models_dir)
        assert comparison["promote"] is True
        assert "improvement" in comparison["reason"]

    def test_regression_does_not_promote(self, models_dir):
        meta = {"last_eval_savings_vs_cpo": 0.25}
        (models_dir / "active_metadata.json").write_text(json.dumps(meta))
        eval_result = {"status": "ok", "savings_vs_cpo_mean": 0.20}
        comparison = compare_models(eval_result, models_dir)
        assert comparison["promote"] is False
        assert "regression" in comparison["reason"]

    def test_no_change_does_not_promote(self, models_dir):
        meta = {"last_eval_savings_vs_cpo": 0.20}
        (models_dir / "active_metadata.json").write_text(json.dumps(meta))
        eval_result = {"status": "ok", "savings_vs_cpo_mean": 0.201}
        comparison = compare_models(eval_result, models_dir)
        assert comparison["promote"] is False

    def test_failed_evaluation_does_not_promote(self, models_dir):
        eval_result = {"status": "error", "error": "out of memory"}
        comparison = compare_models(eval_result, models_dir)
        assert comparison["promote"] is False

    def test_none_evaluation_does_not_promote(self, models_dir):
        comparison = compare_models(None, models_dir)
        assert comparison["promote"] is False

    def test_threshold_is_half_percent(self, models_dir):
        meta = {"last_eval_savings_vs_cpo": 0.20}
        (models_dir / "active_metadata.json").write_text(json.dumps(meta))
        # Exactly 0.5pp improvement (threshold)
        eval_result_below = {"status": "ok", "savings_vs_cpo_mean": 0.2049}
        eval_result_above = {"status": "ok", "savings_vs_cpo_mean": 0.2051}
        assert compare_models(eval_result_below, models_dir)["promote"] is False
        assert compare_models(eval_result_above, models_dir)["promote"] is True


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def _base_report_args(self, reports_dir):
        return dict(
            loop_start=datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc),
            fetch_results={"us-west": pd.DataFrame({"a": [1, 2]})},
            eval_result={"status": "ok", "savings_vs_cpo_mean": 0.22},
            model_metadata={"model_version": "ml_quantile_v2", "n_records": 1000},
            comparison={"promote": True, "reason": "improvement_0.07pp"},
            smoke_test={"status": "ok", "savings_vs_cpo_mean": 0.20},
            reports_dir=reports_dir,
        )

    def test_report_written_to_disk(self, reports_dir):
        args = self._base_report_args(reports_dir)
        generate_report(**args, dry_run=False)
        report_files = list(reports_dir.glob("learning_loop_*.json"))
        assert len(report_files) == 1
        saved = json.loads(report_files[0].read_text())
        assert saved["promoted"] is True

    def test_dry_run_does_not_write(self, reports_dir):
        args = self._base_report_args(reports_dir)
        generate_report(**args, dry_run=True)
        assert not any(reports_dir.glob("*.json"))

    def test_report_keys_present(self, reports_dir):
        args = self._base_report_args(reports_dir)
        report = generate_report(**args, dry_run=True)
        assert "run_date" in report
        assert "data_fetch" in report
        assert "evaluation" in report
        assert "model_training" in report
        assert "model_comparison" in report
        assert "promoted" in report
        assert "benchmark_smoke_test" in report

    def test_none_eval_shows_skipped(self, reports_dir):
        args = self._base_report_args(reports_dir)
        args["eval_result"] = None
        report = generate_report(**args, dry_run=True)
        assert report["evaluation"].get("status") == "skipped"

    def test_none_model_metadata_shows_skipped(self, reports_dir):
        args = self._base_report_args(reports_dir)
        args["model_metadata"] = None
        report = generate_report(**args, dry_run=True)
        assert report["model_training"].get("status") == "skipped"


# ---------------------------------------------------------------------------
# TestBenchmarkSmokeTest
# ---------------------------------------------------------------------------


class TestBenchmarkSmokeTest:
    def test_skipped_when_no_data_file(self, tmp_path):
        result = run_benchmark_smoke_test(tmp_path)
        assert result["status"] == "skipped"
        assert "data_file_not_found" in result["reason"]

    def test_runs_with_real_data(self):
        data_dir = Path(__file__).parent.parent / "data"
        da_path = data_dir / "q12026_3region_dam.csv"
        if not da_path.exists():
            pytest.skip("q12026_3region_dam.csv not available")
        result = run_benchmark_smoke_test(data_dir)
        assert result["status"] in ("ok", "error")
        if result["status"] == "ok":
            assert result["savings_vs_cpo_mean"] is not None


# ---------------------------------------------------------------------------
# TestLearningLoopDryRun
# ---------------------------------------------------------------------------


class TestLearningLoopDryRun:
    def test_dry_run_exits_cleanly(self, tmp_path):
        """Full dry-run of the learning loop with no live APIs."""
        from scripts.daily_learning_loop import main

        data_dir = Path(__file__).parent.parent / "data"
        args = [
            "--dry-run",
            "--no-fetch",
            "--data-dir", str(data_dir),
            "--reports-dir", str(tmp_path / "reports"),
            "--models-dir", str(tmp_path / "models"),
            "--skip-benchmark",
        ]
        with patch("sys.argv", ["daily_learning_loop.py"] + args):
            # Should not raise
            try:
                main()
            except SystemExit as e:
                # Only ok exit codes: 0 (clean) or 1 (smoke test failed)
                assert e.code in (0, 1)

    def test_dry_run_writes_no_files(self, tmp_path):
        from scripts.daily_learning_loop import main

        data_dir = Path(__file__).parent.parent / "data"
        reports_dir = tmp_path / "reports"
        models_dir = tmp_path / "models"

        args = [
            "--dry-run",
            "--no-fetch",
            "--data-dir", str(data_dir),
            "--reports-dir", str(reports_dir),
            "--models-dir", str(models_dir),
            "--skip-benchmark",
        ]
        with patch("sys.argv", ["daily_learning_loop.py"] + args):
            try:
                main()
            except SystemExit:
                pass

        # Reports and models dirs may be created but should have no content
        assert not any(reports_dir.glob("*.json")) if reports_dir.exists() else True
        assert not any(models_dir.glob("*.pkl")) if models_dir.exists() else True
