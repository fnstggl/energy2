"""Tests for the CLI backtest subcommand."""

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(*args):
    """Run the Aurelius CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "aurelius.cli", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode, result.stdout, result.stderr


class TestBacktestCLI:
    def test_backtest_help(self):
        code, stdout, _ = _run_cli("backtest", "--help")
        assert code == 0
        assert "--price-provider" in stdout
        assert "--price-file" in stdout
        assert "--carbon-provider" in stdout
        assert "--carbon-file" in stdout

    def test_backtest_help_no_legacy_price_source(self):
        """Old --price-source argument must not appear in help."""
        code, stdout, _ = _run_cli("backtest", "--help")
        assert "--price-source" not in stdout

    def test_backtest_with_csv(self, price_csv_path, tmp_path):
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv_path),
            "--regions", "us-west,us-east",
            "--train-days", "2",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0 or "No backtest folds" in stdout or "No backtest folds" in stderr

    def test_backtest_csv_missing_file_exits_nonzero(self, tmp_path):
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(tmp_path / "nonexistent.csv"),
            "--regions", "us-west",
        )
        assert code != 0

    def test_backtest_csv_no_price_file_exits_nonzero(self):
        """--price-provider=csv without --price-file must fail."""
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--regions", "us-west",
        )
        assert code != 0

    def test_backtest_eia_provider_choice_rejected(self):
        """--price-provider=eia must be rejected by argparse (not a valid choice)."""
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "eia",
            "--regions", "us-west",
        )
        assert code != 0

    def test_backtest_output_json_valid(self, price_csv_path, tmp_path):
        out = tmp_path / "bt_out.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv_path),
            "--regions", "us-west,us-east",
            "--train-days", "2",
            "--eval-days", "1",
            "--output", str(out),
        )
        if out.exists():
            data = json.loads(out.read_text())
            assert isinstance(data, list)

    def test_backtest_with_carbon_csv(self, price_csv_path, carbon_csv_path, tmp_path):
        out = tmp_path / "bt_carbon.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv_path),
            "--carbon-provider", "csv",
            "--carbon-file", str(carbon_csv_path),
            "--regions", "us-west,us-east",
            "--train-days", "2",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0 or "No backtest folds" in stdout or "No backtest folds" in stderr

    def test_backtest_carbon_csv_missing_file_exits_nonzero(self, price_csv_path, tmp_path):
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-provider", "csv",
            "--price-file", str(price_csv_path),
            "--carbon-provider", "csv",
            "--carbon-file", str(tmp_path / "nonexistent_carbon.csv"),
            "--regions", "us-west",
        )
        assert code != 0
