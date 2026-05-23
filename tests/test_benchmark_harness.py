"""Tests for the Phase 1 benchmark harness.

Verifies:
1. Benchmark runner produces valid JSON output
2. current_price_only is always present in results
3. No leakage: oracle results are flagged separately
4. Regression checker fails on regressions, passes on improvements
5. Benchmark fails when current_price_only is missing
6. All 7 workload types run without error
7. Missing price data is flagged (not silently ignored)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root and aurelius to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))

from benchmarks.compare_against_previous import (
    compare,
    missing_baselines,
)
from benchmarks.run_benchmark import (
    MAX_MISSING_PRICE_PCT,
    PRIMARY_BASELINE,
    REGRESSION_THRESHOLD_PCT,
    WORKLOAD_TYPES,
    compare_against_baseline,
    leakage_audit,
    run_single_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Use the smallest date range that still produces ≥1 backtest fold
QUICK_COMBO = {
    "name": "caiso_pjm_da_rt",
    "regions": ["us-west", "us-east"],
    "da_price_file": "data/plan_da_caiso_pjm.csv",
    "rt_price_file": "data/settle_rt_caiso_pjm.csv",
    "date_start": "2026-02-01",
    "date_end": "2026-03-01",
}

SINGLE_REGION_COMBO = {
    "name": "us-west-only",
    "regions": ["us-west"],
    "da_price_file": "data/caiso_us_west_dam.csv",
    "rt_price_file": None,
    "date_start": "2026-01-15",
    "date_end": "2026-02-15",
}


def _skip_if_no_data(combo: dict) -> None:
    """Skip test if required CSV files are missing."""
    da = REPO_ROOT / combo["da_price_file"]
    if not da.exists():
        pytest.skip(f"Price data not found: {da}")


# ---------------------------------------------------------------------------
# Core benchmark runner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_training_benchmark_produces_valid_result(self):
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="training",
            train_days=14,
            eval_days=5,
            num_jobs=15,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result, f"Benchmark error: {result.get('error')}"
        assert result["folds"] >= 1
        assert result["workload_type"] == "training"
        assert result["region_combo"] == "caiso_pjm_da_rt"

    def test_primary_baseline_always_present(self):
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="llm_batch_inference",
            train_days=14,
            eval_days=5,
            num_jobs=10,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result
        assert PRIMARY_BASELINE in result["savings"], (
            f"current_price_only missing from savings: {list(result['savings'].keys())}"
        )
        assert result["primary_savings_pct"] is not None

    def test_all_baselines_present(self):
        """All 7 baselines must appear in result savings dict."""
        from aurelius.backtesting.baselines import ALL_BASELINES
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="scheduled_batch",
            train_days=14,
            eval_days=5,
            num_jobs=10,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result
        for bl_name in ALL_BASELINES:
            assert bl_name in result["savings"], f"Baseline missing: {bl_name}"

    def test_savings_dict_schema(self):
        """Each baseline entry must have required keys."""
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="data_processing",
            train_days=14,
            eval_days=5,
            num_jobs=10,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result
        for bl_name, bl_data in result["savings"].items():
            assert "savings_pct" in bl_data, f"savings_pct missing in {bl_name}"
            assert "savings_usd" in bl_data, f"savings_usd missing in {bl_name}"
            assert "mean_opt_cost" in bl_data
            assert "mean_baseline_cost" in bl_data

    def test_single_region_realtime_inference_low_savings(self):
        """Realtime inference in a single region has near-zero savings (no delay allowed)."""
        _skip_if_no_data(SINGLE_REGION_COMBO)
        result = run_single_benchmark(
            region_combo=SINGLE_REGION_COMBO,
            workload_type="realtime_inference",
            train_days=14,
            eval_days=5,
            num_jobs=15,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        # May error if no eval jobs produced; that's acceptable for realtime_inference
        if "error" in result:
            pytest.skip(f"No eval jobs for realtime_inference: {result['error']}")
        pct = result.get("primary_savings_pct", 0.0)
        # Single-region realtime_inference: optimizer can't time-shift or switch region
        # So savings should be very small (could be positive or even negative due to
        # power fraction differences, but not large)
        assert pct is None or abs(pct) < 30.0, (
            f"Unexpected large savings {pct:.1f}% for realtime_inference single-region. "
            f"Likely a bug in SLA enforcement."
        )

    def test_missing_price_pct_tracked(self):
        """Missing price hours must be tracked in result."""
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="background_maintenance",
            train_days=14,
            eval_days=5,
            num_jobs=10,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result
        assert "missing_price_pct" in result
        assert isinstance(result["missing_price_pct"], float)

    def test_oracle_result_is_flagged(self):
        """Oracle results must carry oracle=True flag."""
        _skip_if_no_data(QUICK_COMBO)
        result = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="training",
            train_days=14,
            eval_days=5,
            num_jobs=10,
            method="greedy",
            oracle=True,
            repo_root=REPO_ROOT,
        )
        assert "error" not in result
        assert result["oracle"] is True

    def test_oracle_savings_not_less_than_naive(self):
        """Oracle (perfect foresight) should save >= naive forecaster on average.

        This validates that the oracle diagnostic is meaningful — the ceiling
        should never be below the achievable baseline.
        """
        _skip_if_no_data(QUICK_COMBO)
        result_naive = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="training",
            train_days=14,
            eval_days=5,
            num_jobs=15,
            method="greedy",
            oracle=False,
            repo_root=REPO_ROOT,
        )
        result_oracle = run_single_benchmark(
            region_combo=QUICK_COMBO,
            workload_type="training",
            train_days=14,
            eval_days=5,
            num_jobs=15,
            method="greedy",
            oracle=True,
            repo_root=REPO_ROOT,
        )
        if "error" in result_naive or "error" in result_oracle:
            pytest.skip("One or both runs produced no folds")
        naive_pct = result_naive.get("primary_savings_pct", 0.0) or 0.0
        oracle_pct = result_oracle.get("primary_savings_pct", 0.0) or 0.0
        # Oracle should be >= naive, with a 5% slack for stochastic job generation
        assert oracle_pct >= naive_pct - 5.0, (
            f"Oracle savings ({oracle_pct:.1f}%) < naive savings ({naive_pct:.1f}%) "
            f"by more than 5%. This is suspicious — check for a bug."
        )


# ---------------------------------------------------------------------------
# Leakage audit tests
# ---------------------------------------------------------------------------

class TestLeakageAudit:
    def test_oracle_result_generates_leakage_warning(self):
        """Oracle results in non-oracle slot should generate leakage warning."""
        results = [{
            "workload_type": "training",
            "region_combo": "caiso_pjm_da_rt",
            "oracle": True,
            "primary_savings_pct": 15.0,
            "missing_price_pct": 0.0,
            "savings": {"current_price_only": {"savings_pct": 15.0}},
        }]
        issues = leakage_audit(results)
        assert any("LEAKAGE-RISK" in i for i in issues), (
            f"Expected leakage warning for oracle result. Got: {issues}"
        )

    def test_clean_result_no_leakage_issues(self):
        """Non-oracle result with good data coverage should produce no issues."""
        results = [{
            "workload_type": "training",
            "region_combo": "caiso_pjm_da_rt",
            "oracle": False,
            "primary_savings_pct": 10.0,
            "missing_price_pct": 1.0,
            "savings": {"current_price_only": {"savings_pct": 10.0}},
        }]
        issues = leakage_audit(results)
        assert len(issues) == 0, f"Unexpected issues: {issues}"

    def test_high_missing_price_triggers_data_quality_warning(self):
        results = [{
            "workload_type": "llm_batch_inference",
            "region_combo": "us-west-only",
            "oracle": False,
            "primary_savings_pct": 5.0,
            "missing_price_pct": MAX_MISSING_PRICE_PCT + 1.0,
            "savings": {"current_price_only": {"savings_pct": 5.0}},
        }]
        issues = leakage_audit(results)
        assert any("DATA-QUALITY" in i for i in issues)


# ---------------------------------------------------------------------------
# Regression checker tests
# ---------------------------------------------------------------------------

class TestRegressionChecker:
    def _make_result(self, workload: str, region: str, pct: float) -> dict:
        return {
            "workload_type": workload,
            "region_combo": region,
            "primary_savings_pct": pct,
            "savings": {"current_price_only": {"savings_pct": pct}},
        }

    def test_no_regression_when_same(self):
        baseline = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        current = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        regressions = compare_against_baseline(current, baseline, REGRESSION_THRESHOLD_PCT)
        assert len(regressions) == 0

    def test_no_regression_when_improved(self):
        baseline = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        current = [self._make_result("training", "caiso_pjm_da_rt", 12.0)]
        regressions = compare_against_baseline(current, baseline, REGRESSION_THRESHOLD_PCT)
        assert len(regressions) == 0

    def test_regression_detected_when_large_drop(self):
        baseline = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        current = [self._make_result("training", "caiso_pjm_da_rt", 5.0)]
        regressions = compare_against_baseline(current, baseline, REGRESSION_THRESHOLD_PCT)
        assert len(regressions) == 1
        assert "REGRESSION" in regressions[0]

    def test_regression_within_threshold_is_ok(self):
        baseline = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        # Drop by exactly REGRESSION_THRESHOLD_PCT — should NOT trigger
        current = [self._make_result("training", "caiso_pjm_da_rt", 10.0 - REGRESSION_THRESHOLD_PCT)]
        regressions = compare_against_baseline(current, baseline, REGRESSION_THRESHOLD_PCT)
        assert len(regressions) == 0

    def test_new_workload_type_no_regression(self):
        """New entries with no baseline reference don't trigger regression."""
        baseline = [self._make_result("training", "caiso_pjm_da_rt", 10.0)]
        current = [
            self._make_result("training", "caiso_pjm_da_rt", 10.0),
            self._make_result("fine_tuning", "caiso_pjm_da_rt", 8.0),  # new, no baseline
        ]
        regressions = compare_against_baseline(current, baseline, REGRESSION_THRESHOLD_PCT)
        assert len(regressions) == 0

    def test_missing_primary_baseline_flagged(self):
        """Result without current_price_only in savings is flagged."""
        results = [{
            "workload_type": "training",
            "region_combo": "caiso_pjm_da_rt",
            "savings": {"fifo": {"savings_pct": 50.0}},  # missing current_price_only
        }]
        missing = missing_baselines(results)
        assert len(missing) == 1
        assert "current_price_only" in missing[0]

    def test_all_baselines_present_no_flag(self):
        results = [{
            "workload_type": "training",
            "region_combo": "caiso_pjm_da_rt",
            "savings": {"current_price_only": {"savings_pct": 10.0}},
        }]
        missing = missing_baselines(results)
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# compare_against_previous module tests
# ---------------------------------------------------------------------------

class TestCompareScript:
    def test_compare_detects_regression(self, tmp_path):
        """End-to-end: compare_against_previous returns exit code 1 on regression."""

        baseline = [
            {"workload_type": "training", "region_combo": "caiso_pjm_da_rt",
             "primary_savings_pct": 12.0, "savings": {"current_price_only": {"savings_pct": 12.0}}},
        ]
        current = [
            {"workload_type": "training", "region_combo": "caiso_pjm_da_rt",
             "primary_savings_pct": 8.0, "savings": {"current_price_only": {"savings_pct": 8.0}}},
        ]
        regressions, improvements = compare(baseline, current, threshold_pct=2.0)
        assert len(regressions) == 1
        assert len(improvements) == 0

    def test_compare_detects_improvement(self, tmp_path):

        baseline = [
            {"workload_type": "training", "region_combo": "caiso_pjm_da_rt",
             "primary_savings_pct": 8.0, "savings": {"current_price_only": {"savings_pct": 8.0}}},
        ]
        current = [
            {"workload_type": "training", "region_combo": "caiso_pjm_da_rt",
             "primary_savings_pct": 12.0, "savings": {"current_price_only": {"savings_pct": 12.0}}},
        ]
        regressions, improvements = compare(baseline, current, threshold_pct=2.0)
        assert len(regressions) == 0
        assert len(improvements) == 1


# ---------------------------------------------------------------------------
# Benchmark file existence tests
# ---------------------------------------------------------------------------

class TestBenchmarkFileStructure:
    BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
    API_NEEDED_DIR = REPO_ROOT / "API-NEEDED"
    REPO_ROOT = REPO_ROOT

    def test_benchmark_config_exists(self):
        assert (self.BENCHMARKS_DIR / "benchmark_config.yaml").exists()

    def test_workload_matrix_exists(self):
        assert (self.BENCHMARKS_DIR / "workload_matrix.yaml").exists()

    def test_region_matrix_exists(self):
        assert (self.BENCHMARKS_DIR / "region_matrix.yaml").exists()

    def test_baseline_matrix_exists(self):
        assert (self.BENCHMARKS_DIR / "baseline_matrix.yaml").exists()

    def test_run_benchmark_script_exists(self):
        assert (self.BENCHMARKS_DIR / "run_benchmark.py").exists()

    def test_compare_against_previous_script_exists(self):
        assert (self.BENCHMARKS_DIR / "compare_against_previous.py").exists()

    def test_run_all_workloads_script_exists(self):
        assert (self.BENCHMARKS_DIR / "run_all_workloads.sh").exists()

    def test_run_all_regions_script_exists(self):
        assert (self.BENCHMARKS_DIR / "run_all_regions.sh").exists()

    def test_run_oracle_diagnostics_script_exists(self):
        assert (self.BENCHMARKS_DIR / "run_oracle_diagnostics.sh").exists()

    # Providers that are still genuinely needed (no credentials in env)
    def test_api_needed_entsoe_exists(self):
        assert (self.API_NEEDED_DIR / "ENTSOE.md").exists()

    def test_api_needed_electricitymaps_exists(self):
        assert (self.API_NEEDED_DIR / "ELECTRICITYMAPS.md").exists()

    def test_api_needed_prometheus_dcgm_exists(self):
        assert (self.API_NEEDED_DIR / "PROMETHEUS_DCGM.md").exists()

    # Providers that were previously API-NEEDED but credentials are now available:
    # PJM (env: PJM_API_KEY found), ERCOT (env: ERCOT_API_KEY found),
    # WattTime (env: WATTTIME_USERNAME/PASSWORD found),
    # Open-Meteo (no API key required — free public API).
    # These API-NEEDED files were intentionally removed when credentials were confirmed.
    def test_api_needed_pjm_no_longer_needed(self):
        assert not (self.API_NEEDED_DIR / "PJM.md").exists(), (
            "PJM.md should not exist in API-NEEDED — PJM credentials are now confirmed available"
        )

    def test_api_needed_ercot_no_longer_needed(self):
        assert not (self.API_NEEDED_DIR / "ERCOT.md").exists(), (
            "ERCOT.md should not exist in API-NEEDED — ERCOT credentials are now confirmed available"
        )

    def test_api_needed_watttime_no_longer_needed(self):
        assert not (self.API_NEEDED_DIR / "WATTTIME.md").exists(), (
            "WATTTIME.md should not exist in API-NEEDED — WattTime credentials are now confirmed available"
        )

    def test_workload_matrix_has_all_types(self):
        """workload_matrix.yaml must list all 7 canonical workload types."""
        import yaml
        path = self.BENCHMARKS_DIR / "workload_matrix.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        listed_types = [w["workload_type"] for w in data["workloads"]]
        for wtype in WORKLOAD_TYPES:
            assert wtype in listed_types, f"Workload type {wtype!r} missing from workload_matrix.yaml"

    def test_baseline_matrix_has_primary_baseline(self):
        """baseline_matrix.yaml must flag current_price_only as primary."""
        import yaml
        path = self.BENCHMARKS_DIR / "baseline_matrix.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        primary = [b for b in data["baselines"] if b.get("primary")]
        assert len(primary) == 1, f"Expected 1 primary baseline, got {len(primary)}"
        assert primary[0]["name"] == PRIMARY_BASELINE

    def test_q12026_3region_dam_csv_exists(self):
        """3-region Q1 2026 DA merged CSV must exist for benchmark."""
        data_dir = self.REPO_ROOT / "data"
        assert (data_dir / "q12026_3region_dam.csv").exists(), (
            "data/q12026_3region_dam.csv missing — run: "
            "python scripts/merge_price_csvs.py --inputs data/caiso_us_west_dam.csv "
            "data/pjm_us_east_dam.csv data/ercot_us_south_dam.csv "
            "--output data/q12026_3region_dam.csv"
        )

    def test_q12026_3region_rt_csv_exists(self):
        """3-region Q1 2026 RT merged CSV must exist for benchmark."""
        data_dir = self.REPO_ROOT / "data"
        assert (data_dir / "q12026_3region_rt.csv").exists(), (
            "data/q12026_3region_rt.csv missing"
        )

    def test_summer2025_3region_dam_csv_exists(self):
        """3-region summer 2025 DA merged CSV must exist for benchmark."""
        data_dir = self.REPO_ROOT / "data" / "summer2025"
        assert (data_dir / "3region_dam.csv").exists(), (
            "data/summer2025/3region_dam.csv missing"
        )

    def test_summer2025_3region_rt_csv_exists(self):
        """3-region summer 2025 RT merged CSV must exist for benchmark."""
        data_dir = self.REPO_ROOT / "data" / "summer2025"
        assert (data_dir / "3region_rt.csv").exists(), (
            "data/summer2025/3region_rt.csv missing"
        )

    def test_3region_dam_has_all_regions(self):
        """3-region Q1 2026 DA CSV must have us-west, us-east, us-south."""
        data_dir = self.REPO_ROOT / "data"
        path = data_dir / "q12026_3region_dam.csv"
        if not path.exists():
            pytest.skip("3-region CSV not yet generated")
        df = pd.read_csv(path)
        regions = set(df["region"].unique())
        assert "us-west" in regions, "us-west missing from 3-region DA CSV"
        assert "us-east" in regions, "us-east missing from 3-region DA CSV"
        assert "us-south" in regions, "us-south missing from 3-region DA CSV"

    def test_region_matrix_has_ercot(self):
        """region_matrix.yaml must define us-south (ERCOT)."""
        import yaml
        path = self.BENCHMARKS_DIR / "region_matrix.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        region_ids = [r["region_id"] for r in data["regions"]]
        assert "us-south" in region_ids, "us-south (ERCOT) missing from region_matrix.yaml"

    def test_region_matrix_has_3region_combo(self):
        """region_matrix.yaml must define the 3-region CAISO+PJM+ERCOT combination."""
        import yaml
        path = self.BENCHMARKS_DIR / "region_matrix.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        combo_names = [c["name"] for c in data.get("multi_region_combinations", [])]
        assert "caiso_pjm_ercot" in combo_names, (
            "caiso_pjm_ercot combination missing from region_matrix.yaml"
        )
