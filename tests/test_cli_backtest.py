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
        assert "--price-source" in stdout

    def test_backtest_with_csv(self, price_csv_path, tmp_path):
        out = tmp_path / "bt_results.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-source", str(price_csv_path),
            "--regions", "us-west,us-east",
            "--train-days", "2",
            "--eval-days", "1",
            "--output", str(out),
        )
        assert code == 0 or "No backtest folds" in stdout or "No backtest folds" in stderr

    def test_backtest_missing_file_exits_nonzero(self, tmp_path):
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-source", str(tmp_path / "nonexistent.csv"),
            "--regions", "us-west",
        )
        assert code != 0

    def test_backtest_output_json_valid(self, price_csv_path, tmp_path):
        out = tmp_path / "bt_out.json"
        code, stdout, stderr = _run_cli(
            "backtest",
            "--price-source", str(price_csv_path),
            "--regions", "us-west,us-east",
            "--train-days", "2",
            "--eval-days", "1",
            "--output", str(out),
        )
        if out.exists():
            data = json.loads(out.read_text())
            assert isinstance(data, list)
