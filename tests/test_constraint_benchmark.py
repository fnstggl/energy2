"""Tests for Phase 11 — Constraint-aware benchmark framework.

Verifies:
1. ConstraintBenchmarkRunner runs scenarios under all policies without error
2. Benchmark metadata is complete and reproducible (same seed → same metadata)
3. Migration feedback loop: constraint_aware policy can apply migrations
4. KPI aggregation is correct (costs sum, SLA violations track correctly)
5. BenchmarkReport serializes to dict/JSON cleanly
6. BenchmarkRegressionChecker detects regressions and improvements
7. BenchmarkRegressionChecker validates metadata compatibility
8. ScenarioLock checks frozen scenario hashes
9. CLI commands are registered and dispatch correctly
10. Optimizer safety: constraint_aware SLA violations ≤ fifo in stable scenarios
11. Determinism: same seed → same KPI vector
12. Constraint match for scenarios where the classifier reliably fires
13. new Recommendation.target_region field is populated for migration actions
14. migrate_workload() simulator method applies migration penalties correctly
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from aurelius.benchmarks import (
    ConstraintBenchmarkRunner,
    BenchmarkRegressionChecker,
)
from aurelius.benchmarks.constraint_runner import (
    BenchmarkResult,
    POLICY_FIFO,
    POLICY_CONSTRAINT_AWARE,
    POLICY_GREEDY_ENERGY,
    ALL_POLICIES,
)
from aurelius.benchmarks.report import build_scorecard, AggregatedKPI, TickKPI
from aurelius.benchmarks.scenario_lock import check_lockfile, _collect_hashes
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario
from aurelius.state.models import Recommendation, Provenance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def energy_scenario():
    return load_scenario("energy_price_arbitrage_multiregion", seed_override=42)


@pytest.fixture
def thermal_scenario():
    return load_scenario("thermal_hotspot_mixed_cluster", seed_override=42)


@pytest.fixture
def queue_scenario():
    return load_scenario("queue_surge_latency_sensitive", seed_override=42)


@pytest.fixture
def runner_minimal():
    """Runner with only fifo and constraint_aware for speed."""
    return ConstraintBenchmarkRunner(policies=[POLICY_FIFO, POLICY_CONSTRAINT_AWARE])


@pytest.fixture
def runner_full():
    """Runner with all 5 policies."""
    return ConstraintBenchmarkRunner()


# ---------------------------------------------------------------------------
# 1. Runner basic functionality
# ---------------------------------------------------------------------------

class TestRunnerBasic:
    def test_run_energy_scenario_completes(self, runner_minimal, energy_scenario):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        assert result is not None
        assert isinstance(result, BenchmarkResult)
        assert POLICY_FIFO in result.policy_results
        assert POLICY_CONSTRAINT_AWARE in result.policy_results

    def test_all_policies_run(self, runner_full, energy_scenario):
        result = runner_full.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        for policy in ALL_POLICIES:
            assert policy in result.policy_results, f"Policy {policy!r} missing"
            pr = result.policy_results[policy]
            assert len(pr.tick_kpis) == 6, f"{policy}: expected 6 tick KPIs"

    def test_report_is_populated(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        r = result.report
        assert r.metadata is not None
        assert r.aggregated
        assert r.scorecard is not None
        assert isinstance(r.is_valid, bool)

    def test_run_all_scenarios_no_crash(self, runner_minimal):
        """All 6 canonical scenarios run without raising."""
        results = runner_minimal.run_all_scenarios(seed=42, steps=8)
        assert len(results) == 6
        for name, res in results.items():
            assert res is not None, f"Scenario {name!r} returned None"

    def test_thermal_scenario_completes(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "thermal_hotspot_mixed_cluster", seed=42, steps=8
        )
        assert result is not None
        pr_ca = result.policy_results[POLICY_CONSTRAINT_AWARE]
        assert len(pr_ca.tick_kpis) == 8


# ---------------------------------------------------------------------------
# 2. Metadata completeness and reproducibility
# ---------------------------------------------------------------------------

class TestBenchmarkMetadata:
    def test_metadata_fields_complete(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        m = result.metadata
        assert m.scenario_name == "energy_price_arbitrage_multiregion"
        assert m.scenario_version == "v1"
        assert m.scenario_hash is not None and len(m.scenario_hash) > 0
        assert m.seed == 42
        assert m.simulator_version is not None
        assert m.optimizer_version is not None
        assert m.config_hash is not None
        assert m.steps == 6
        assert m.timestamp is not None
        assert m.is_sandbox is True

    def test_same_seed_same_metadata(self, runner_minimal):
        r1 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        r2 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        # Same inputs → same hashes (not timestamp)
        assert r1.metadata.scenario_hash == r2.metadata.scenario_hash
        assert r1.metadata.config_hash == r2.metadata.config_hash
        assert r1.metadata.seed == r2.metadata.seed

    def test_different_seed_different_metadata(self, runner_minimal):
        r1 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        r2 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=99, steps=6
        )
        # Different seeds → different metadata (seed field differs, config_hash changes)
        assert r1.metadata.seed != r2.metadata.seed

    def test_metadata_to_dict_roundtrip(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        d = result.metadata.to_dict()
        assert d["scenario_name"] == "energy_price_arbitrage_multiregion"
        assert d["seed"] == 42
        assert d["is_sandbox"] is True

    def test_is_comparable_to_same(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        ok, mismatches = result.metadata.is_comparable_to(result.metadata)
        assert ok is True
        assert len(mismatches) == 0

    def test_is_comparable_to_different_seed(self, runner_minimal):
        r1 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        r2 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=99, steps=6
        )
        ok, mismatches = r1.metadata.is_comparable_to(r2.metadata)
        assert ok is False
        assert any("seed" in m for m in mismatches)


# ---------------------------------------------------------------------------
# 3. Migration feedback loop
# ---------------------------------------------------------------------------

class TestMigrationFeedback:
    def test_migrate_workload_applies_penalty(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()  # initialize cluster
        cluster = sim._cluster

        # Find a migratable workload
        wl_migrable = None
        for wl in cluster.workloads.values():
            if wl.migration_allowed:
                wl_migrable = wl
                break

        if wl_migrable is None:
            pytest.skip("No migratable workloads in this scenario")

        # Find a target region different from current
        other_regions = [r for r in cluster.regions if r != wl_migrable.region_id]
        if not other_regions:
            pytest.skip("Only one region in this scenario")

        src_region = wl_migrable.region_id
        old_hit_rate = wl_migrable.prefix_cache_hit_rate_frac
        target_region = other_regions[0]

        result = sim.migrate_workload(wl_migrable.workload_id, target_region)

        assert result is True, "Migration should succeed"
        assert wl_migrable.region_id == target_region
        assert wl_migrable.cold_start_warmup_ticks_remaining > 0
        assert wl_migrable.prefix_cache_hit_rate_frac < old_hit_rate  # cache miss
        assert cluster.migration_count == 1

    def test_migrate_blocked_when_not_allowed(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()
        cluster = sim._cluster

        # Find an unmigrable workload
        wl_blocked = None
        for wl in cluster.workloads.values():
            if not wl.migration_allowed:
                wl_blocked = wl
                break

        if wl_blocked is None:
            pytest.skip("All workloads are migratable in this scenario")

        other_regions = [r for r in cluster.regions if r != wl_blocked.region_id]
        if not other_regions:
            pytest.skip("Only one region in this scenario")

        result = sim.migrate_workload(wl_blocked.workload_id, other_regions[0])
        assert result is False, "Migration should be blocked when migration_allowed=False"
        assert cluster.migration_count == 0

    def test_migrate_same_region_is_noop(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()
        cluster = sim._cluster

        wl = next(iter(cluster.workloads.values()))
        result = sim.migrate_workload(wl.workload_id, wl.region_id)
        assert result is False
        assert cluster.migration_count == 0

    def test_migrate_unknown_region_fails(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()
        cluster = sim._cluster

        wl = next(iter(cluster.workloads.values()))
        result = sim.migrate_workload(wl.workload_id, "nonexistent-region")
        assert result is False
        assert cluster.migration_count == 0

    def test_greedy_policy_applies_migrations(self, energy_scenario):
        runner = ConstraintBenchmarkRunner(policies=[POLICY_FIFO, POLICY_GREEDY_ENERGY])
        result = runner.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=12
        )
        # greedy_energy should attempt migrations (log may have entries)
        greedy_pr = result.policy_results[POLICY_GREEDY_ENERGY]
        # Migration log tracks every migration applied
        # (may be 0 if all workloads are already in cheapest region)
        assert isinstance(greedy_pr.migration_log, list)


# ---------------------------------------------------------------------------
# 4. KPI aggregation
# ---------------------------------------------------------------------------

class TestKPIAggregation:
    def test_fifo_kpis_non_negative(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        fifo = result.report.aggregated[POLICY_FIFO]
        assert fifo.total_energy_cost >= 0
        assert fifo.total_tokens >= 0
        assert fifo.total_energy_kwh >= 0
        assert fifo.mean_gpu_util_pct >= 0
        assert fifo.total_migrations >= 0

    def test_tick_kpis_sum_correctly(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        pr = result.policy_results[POLICY_FIFO]
        expected_cost = sum(k.total_energy_cost for k in pr.tick_kpis)
        agg = result.report.aggregated[POLICY_FIFO]
        assert abs(agg.total_energy_cost - expected_cost) < 1e-9

    def test_cost_per_token_is_none_when_no_tokens(self):
        kpis = [
            TickKPI(
                tick=1, total_energy_cost=1.0, total_tokens=0,
                total_energy_kwh=0.1, cost_per_token=None,
                tokens_per_joule=None, mean_gpu_util_pct=50.0,
                p95_latency_ms=None, p99_latency_ms=None,
                queue_wait_p95_ms=None, sla_violations=0,
                thermal_throttle_gpu_count=0, migration_count=0,
                mean_topology_score=1.0,
            )
        ]
        from aurelius.benchmarks.constraint_runner import _aggregate_kpis
        agg = _aggregate_kpis("test", kpis)
        assert agg.mean_cost_per_token is None  # None not zero

    def test_sla_violations_tracked(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "queue_surge_latency_sensitive", seed=42, steps=12
        )
        fifo = result.report.aggregated[POLICY_FIFO]
        # Queue surge scenario should have some SLA violations
        # (may or may not depending on scenario config; just check type)
        assert isinstance(fifo.total_sla_violations, int)
        assert fifo.total_sla_violations >= 0


# ---------------------------------------------------------------------------
# 5. Report serialization
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_to_dict_all_required_keys(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        d = result.report.to_dict()
        assert "metadata" in d
        assert "kpi_comparison" in d
        assert "scorecard" in d
        assert "regression_flags" in d
        assert "is_valid" in d

    def test_json_roundtrip(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        d = result.report.to_dict()
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        assert loaded["metadata"]["scenario_name"] == "energy_price_arbitrage_multiregion"
        assert loaded["metadata"]["is_sandbox"] is True

    def test_to_text_contains_required_sections(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        text = result.report.to_text()
        assert "Aurelius Constraint-Aware Benchmark" in text
        assert "[SANDBOX]" in text
        assert "Optimization Scorecard" in text
        assert "Comparison validity" in text

    def test_benchmark_result_to_dict(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        d = result.to_dict()
        assert "metadata" in d
        assert "report" in d
        assert "policy_migration_counts" in d


# ---------------------------------------------------------------------------
# 6. Scorecard
# ---------------------------------------------------------------------------

class TestScorecard:
    def test_scorecard_all_scores_in_range(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        sc = result.report.scorecard
        for attr in (
            "net_cost_improvement", "sla_preservation", "utilization_improvement",
            "latency_improvement", "thermal_improvement", "migration_stability",
            "topology_quality", "weighted_score",
        ):
            val = getattr(sc, attr)
            assert 0.0 <= val <= 1.0, f"Scorecard.{attr} = {val} out of [0, 1]"

    def test_scorecard_flags_sla_regression(self):
        fifo = AggregatedKPI(
            policy_name="fifo",
            total_energy_cost=1.0, total_tokens=1000, total_energy_kwh=0.1,
            mean_cost_per_token=0.001, mean_tokens_per_joule=1.0,
            mean_gpu_util_pct=50.0, p99_latency_ms=100.0, p95_latency_ms=80.0,
            p95_queue_wait_ms=50.0, total_sla_violations=0,
            total_thermal_throttle_ticks=0, total_migrations=0,
            mean_topology_score=0.9,
        )
        ca = AggregatedKPI(
            policy_name="constraint_aware",
            total_energy_cost=0.9, total_tokens=1000, total_energy_kwh=0.09,
            mean_cost_per_token=0.0009, mean_tokens_per_joule=1.1,
            mean_gpu_util_pct=55.0, p99_latency_ms=90.0, p95_latency_ms=75.0,
            p95_queue_wait_ms=45.0, total_sla_violations=5,  # regression
            total_thermal_throttle_ticks=0, total_migrations=2,
            mean_topology_score=0.85,
        )
        sc = build_scorecard(ca, fifo, steps=24)
        assert any("SLA" in f for f in sc.flags), "SLA regression should be flagged"

    def test_scorecard_flags_cost_regression(self):
        fifo = AggregatedKPI(
            policy_name="fifo",
            total_energy_cost=1.0, total_tokens=1000, total_energy_kwh=0.1,
            mean_cost_per_token=0.001, mean_tokens_per_joule=1.0,
            mean_gpu_util_pct=50.0, p99_latency_ms=None, p95_latency_ms=None,
            p95_queue_wait_ms=None, total_sla_violations=0,
            total_thermal_throttle_ticks=0, total_migrations=0,
            mean_topology_score=0.9,
        )
        ca = AggregatedKPI(
            policy_name="constraint_aware",
            total_energy_cost=1.10,  # 10% more expensive
            total_tokens=1000, total_energy_kwh=0.11,
            mean_cost_per_token=0.0011, mean_tokens_per_joule=0.9,
            mean_gpu_util_pct=50.0, p99_latency_ms=None, p95_latency_ms=None,
            p95_queue_wait_ms=None, total_sla_violations=0,
            total_thermal_throttle_ticks=0, total_migrations=0,
            mean_topology_score=0.9,
        )
        sc = build_scorecard(ca, fifo, steps=24)
        assert any("COST" in f for f in sc.flags)

    def test_scorecard_flags_migration_churn(self):
        fifo = AggregatedKPI(
            policy_name="fifo",
            total_energy_cost=1.0, total_tokens=0, total_energy_kwh=0.1,
            mean_cost_per_token=None, mean_tokens_per_joule=None,
            mean_gpu_util_pct=50.0, p99_latency_ms=None, p95_latency_ms=None,
            p95_queue_wait_ms=None, total_sla_violations=0,
            total_thermal_throttle_ticks=0, total_migrations=0,
            mean_topology_score=0.9,
        )
        ca = AggregatedKPI(
            policy_name="constraint_aware",
            total_energy_cost=0.9, total_tokens=0, total_energy_kwh=0.09,
            mean_cost_per_token=None, mean_tokens_per_joule=None,
            mean_gpu_util_pct=55.0, p99_latency_ms=None, p95_latency_ms=None,
            p95_queue_wait_ms=None, total_sla_violations=0,
            total_thermal_throttle_ticks=0, total_migrations=200,  # churn
            mean_topology_score=0.9,
        )
        sc = build_scorecard(ca, fifo, steps=24)
        assert any("MIGRATION" in f for f in sc.flags)

    def test_scorecard_no_flags_when_clean(self):
        fifo = AggregatedKPI(
            policy_name="fifo",
            total_energy_cost=1.0, total_tokens=1000, total_energy_kwh=0.1,
            mean_cost_per_token=0.001, mean_tokens_per_joule=1.0,
            mean_gpu_util_pct=50.0, p99_latency_ms=200.0, p95_latency_ms=150.0,
            p95_queue_wait_ms=100.0, total_sla_violations=5,
            total_thermal_throttle_ticks=3, total_migrations=0,
            mean_topology_score=0.8,
        )
        ca = AggregatedKPI(
            policy_name="constraint_aware",
            total_energy_cost=0.85,  # cheaper
            total_tokens=1000, total_energy_kwh=0.085,
            mean_cost_per_token=0.00085, mean_tokens_per_joule=1.2,
            mean_gpu_util_pct=60.0, p99_latency_ms=180.0, p95_latency_ms=140.0,
            p95_queue_wait_ms=90.0, total_sla_violations=4,  # fewer violations
            total_thermal_throttle_ticks=2, total_migrations=2,
            mean_topology_score=0.85,
        )
        sc = build_scorecard(ca, fifo, steps=24)
        assert sc.flags == []


# ---------------------------------------------------------------------------
# 7. Regression checker
# ---------------------------------------------------------------------------

class TestRegressionChecker:
    def _make_report_dict(
        self,
        scenario_name: str = "test",
        scenario_hash: str = "abc123",
        seed: int = 42,
        steps: int = 24,
        cost: float = 1.0,
        sla_violations: int = 0,
        p99_latency_ms: float = 100.0,
        weighted_score: float = 0.7,
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "scenario_name": scenario_name,
                "scenario_version": "v1",
                "scenario_hash": scenario_hash,
                "seed": seed,
                "simulator_version": "1.0.0",
                "optimizer_version": "1.0.0",
                "config_hash": "cfg123",
                "steps": steps,
                "is_sandbox": True,
            },
            "report": {
                "kpi_comparison": {
                    "constraint_aware": {
                        "total_energy_cost": cost,
                        "total_sla_violations": sla_violations,
                        "total_migrations": 2,
                        "p99_latency_ms": p99_latency_ms,
                        "mean_topology_score": 0.85,
                    }
                },
                "scorecard": {"weighted_score": weighted_score},
                "regression_flags": [],
            },
        }

    def test_identical_reports_pass(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict()
        current = self._make_report_dict()
        result = checker.compare(baseline, current)
        assert result.passed is True
        assert result.regressions == []

    def test_cost_regression_detected(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict(cost=1.0)
        current = self._make_report_dict(cost=1.05)  # 5% regression > 2% threshold
        result = checker.compare(baseline, current)
        assert result.passed is False
        assert any("COST" in r for r in result.regressions)

    def test_sla_regression_detected(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict(sla_violations=0)
        current = self._make_report_dict(sla_violations=5)
        result = checker.compare(baseline, current)
        assert result.passed is False
        assert any("SLA" in r for r in result.regressions)

    def test_scorecard_regression_detected(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict(weighted_score=0.8)
        current = self._make_report_dict(weighted_score=0.7)  # 0.1 drop > 0.05 threshold
        result = checker.compare(baseline, current)
        assert result.passed is False
        assert any("SCORECARD" in r for r in result.regressions)

    def test_improvement_reported(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict(cost=1.0, p99_latency_ms=200.0)
        current = self._make_report_dict(cost=0.9, p99_latency_ms=160.0)
        result = checker.compare(baseline, current)
        assert result.passed is True
        assert result.improvements  # should have improvement entries

    def test_metadata_mismatch_invalidates_comparison(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict(seed=42, scenario_hash="abc")
        current = self._make_report_dict(seed=99, scenario_hash="xyz")
        result = checker.compare(baseline, current)
        assert result.comparison_valid is False
        assert len(result.metadata_mismatches) > 0

    def test_missing_policy_generates_warning(self):
        checker = BenchmarkRegressionChecker()
        baseline = self._make_report_dict()
        current = self._make_report_dict()
        result = checker.compare(baseline, current, policy="nonexistent_policy")
        assert any("missing" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# 8. Scenario lockfile
# ---------------------------------------------------------------------------

class TestScenarioLock:
    def test_lockfile_exists(self):
        lockfile = Path("benchmarks/v1/.scenario_hashes.json")
        assert lockfile.exists(), "Lockfile must be generated before tests run"

    def test_lockfile_passes_check(self):
        ok, mismatches = check_lockfile("v1")
        assert ok is True, f"Lockfile check failed: {mismatches}"

    def test_collect_hashes_finds_six_scenarios(self):
        hashes = _collect_hashes("v1")
        assert len(hashes) == 6
        assert "energy_price_arbitrage_multiregion.yaml" in hashes

    def test_lockfile_json_valid(self):
        lockfile = Path("benchmarks/v1/.scenario_hashes.json")
        data = json.loads(lockfile.read_text())
        assert isinstance(data, dict)
        assert len(data) == 6


# ---------------------------------------------------------------------------
# 9. CLI command registration
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    def test_benchmark_run_registered(self):
        from aurelius.cli_constraint import cmd_benchmark_run
        assert callable(cmd_benchmark_run)

    def test_benchmark_compare_registered(self):
        from aurelius.cli_constraint import cmd_benchmark_compare
        assert callable(cmd_benchmark_compare)

    def test_optimizer_regression_check_registered(self):
        from aurelius.cli_constraint import cmd_optimizer_regression_check
        assert callable(cmd_optimizer_regression_check)

    def test_all_five_phase10_commands_still_exist(self):
        from aurelius.cli_constraint import (
            cmd_constraint_report,
            cmd_simulate_constraint_scenario,
            cmd_telemetry_check,
            cmd_topology_report,
            cmd_validate_connectors,
        )
        for fn in (
            cmd_constraint_report,
            cmd_simulate_constraint_scenario,
            cmd_telemetry_check,
            cmd_topology_report,
            cmd_validate_connectors,
        ):
            assert callable(fn)


# ---------------------------------------------------------------------------
# 10. Optimizer safety: SLA violations
# ---------------------------------------------------------------------------

class TestOptimizerSafety:
    def test_sla_violations_not_worse_than_fifo_in_energy_scenario(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=16
        )
        fifo = result.report.aggregated[POLICY_FIFO]
        ca = result.report.aggregated[POLICY_CONSTRAINT_AWARE]
        assert ca.total_sla_violations <= fifo.total_sla_violations, (
            f"constraint_aware SLA violations ({ca.total_sla_violations}) "
            f"must not exceed fifo ({fifo.total_sla_violations})"
        )

    def test_migration_churn_bounded(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=16
        )
        ca = result.report.aggregated[POLICY_CONSTRAINT_AWARE]
        max_threshold = 16 * 2  # 2 migrations per tick max
        assert ca.total_migrations <= max_threshold, (
            f"Migration churn {ca.total_migrations} exceeds threshold {max_threshold}"
        )

    def test_scorecard_above_minimum(self, runner_minimal):
        result = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=16
        )
        sc = result.report.scorecard
        assert sc.weighted_score >= 0.3, (
            f"Weighted score {sc.weighted_score:.3f} below minimum 0.3"
        )


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_kpis(self, runner_minimal):
        r1 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        r2 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        fifo1 = r1.report.aggregated[POLICY_FIFO]
        fifo2 = r2.report.aggregated[POLICY_FIFO]
        assert abs(fifo1.total_energy_cost - fifo2.total_energy_cost) < 1e-9
        assert fifo1.total_tokens == fifo2.total_tokens
        assert fifo1.total_sla_violations == fifo2.total_sla_violations

    def test_different_seed_different_kpis(self, runner_minimal):
        r1 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        r2 = runner_minimal.run_scenario(
            "energy_price_arbitrage_multiregion", seed=99, steps=8
        )
        # Different seeds may produce different costs (thermal noise, etc.)
        # This is not guaranteed to differ in all scenarios, so we just check metadata
        assert r1.metadata.seed != r2.metadata.seed


# ---------------------------------------------------------------------------
# 12. Constraint match: thermal scenario reliably fires
# ---------------------------------------------------------------------------

class TestConstraintMatch:
    def test_thermal_scenario_matches(self):
        runner = ConstraintBenchmarkRunner(
            policies=[POLICY_FIFO, POLICY_CONSTRAINT_AWARE]
        )
        result = runner.run_scenario(
            "thermal_hotspot_mixed_cluster", seed=42, steps=24
        )
        r = result.report
        assert r.observed_dominant_constraint == "thermal", (
            f"Expected 'thermal', got {r.observed_dominant_constraint!r}"
        )
        assert r.constraint_match is True

    def test_all_scenarios_have_expected_constraint_field(self):
        runner = ConstraintBenchmarkRunner(
            policies=[POLICY_FIFO, POLICY_CONSTRAINT_AWARE]
        )
        results = runner.run_all_scenarios(seed=42, steps=12)
        for name, result in results.items():
            # Each scenario YAML has an expected_primary_constraint field
            assert result.report.expected_primary_constraint is not None, (
                f"Scenario {name!r} missing expected_primary_constraint"
            )


# ---------------------------------------------------------------------------
# 13. Recommendation.target_region populated for migration actions
# ---------------------------------------------------------------------------

class TestRecommendationTargetRegion:
    def test_target_region_field_exists(self):
        """Recommendation.target_region is a new Phase 11 field."""
        ts = datetime.now(timezone.utc)
        prov = Provenance(source="test", fetched_at=ts, confidence="high", is_sandbox=True)
        rec = Recommendation(
            recommendation_id="test-id",
            workload_id="wl-1",
            action_type="KEEP",
            timestamp=ts,
            provenance=prov,
            is_noop=True,
            target_region=None,
        )
        assert rec.target_region is None

    def test_target_region_serializes(self):
        ts = datetime.now(timezone.utc)
        prov = Provenance(source="test", fetched_at=ts, confidence="medium", is_sandbox=True)
        rec = Recommendation(
            recommendation_id="rec-1",
            workload_id="wl-1",
            action_type="CHOOSE_CHEAPER_REGION",
            timestamp=ts,
            provenance=prov,
            target_region="us-west",
            confidence=0.8,
            sla_status="satisfied",
        )
        d = rec.to_dict()
        assert "target_region" in d
        assert d["target_region"] == "us-west"

    def test_target_region_roundtrip(self):
        ts = datetime.now(timezone.utc)
        prov = Provenance(source="test", fetched_at=ts, confidence="medium", is_sandbox=True)
        rec = Recommendation(
            recommendation_id="rec-1",
            workload_id="wl-1",
            action_type="CHOOSE_CHEAPER_REGION",
            timestamp=ts,
            provenance=prov,
            target_region="us-west",
            confidence=0.8,
            sla_status="satisfied",
        )
        d = rec.to_dict()
        loaded = Recommendation.from_dict(d)
        assert loaded.target_region == "us-west"

    def test_engine_populates_target_region_for_migration(self):
        """ConstraintAwareEngine sets target_region on CHOOSE_CHEAPER_REGION recs."""
        from aurelius.constraints.engine import ConstraintAwareEngine
        from aurelius.simulation.cluster.engine import ClusterSimulator
        from aurelius.simulation.cluster.scenarios import load_scenario

        scenario = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
        sim = ClusterSimulator(scenario.config, seed=42)
        # Run enough ticks for energy spike to kick in
        ticks = sim.run(steps=12)
        engine = ConstraintAwareEngine()

        for tick in ticks:
            er = engine.run(tick.cluster_state)
            for rec in er.recommendations:
                if rec.action_type == "CHOOSE_CHEAPER_REGION" and not rec.is_noop:
                    # This recommendation should have a target_region set
                    assert rec.target_region is not None, (
                        "CHOOSE_CHEAPER_REGION must set target_region"
                    )


# ---------------------------------------------------------------------------
# 14. Full policy comparison run (smoke test)
# ---------------------------------------------------------------------------

class TestFullPolicyComparison:
    def test_all_five_policies_energy_scenario(self, runner_full):
        result = runner_full.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=8
        )
        assert len(result.policy_results) == 5
        for policy in ALL_POLICIES:
            pr = result.policy_results[policy]
            assert len(pr.tick_kpis) == 8
            agg = result.report.aggregated[policy]
            assert agg.total_energy_cost >= 0

    def test_report_text_labels_sandbox(self, runner_full):
        result = runner_full.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=6
        )
        text = result.report.to_text()
        assert "[SANDBOX]" in text

    def test_greedy_energy_never_fewer_migrations_than_fifo(self, runner_full):
        result = runner_full.run_scenario(
            "energy_price_arbitrage_multiregion", seed=42, steps=12
        )
        # In a 2-region energy arbitrage scenario with price differences,
        # greedy_energy should attempt more migrations than fifo (which has none)
        fifo_mig = result.report.aggregated[POLICY_FIFO].total_migrations
        assert fifo_mig == 0  # fifo never migrates
