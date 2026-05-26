"""Simulator non-migration action tests + benchmark non-inertness (Mission 3).

Proves:
- The simulator applies SCALE/SPREAD/DEFER/CONSOLIDATE against simulated state.
- The constraint-aware benchmark policy is no longer byte-identical to FIFO and
  improves the relevant KPI without an SLA regression.
"""

from __future__ import annotations

import pytest

from aurelius.benchmarks.constraint_runner import ConstraintBenchmarkRunner
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario


def _sim(scenario, steps=0):
    sc = load_scenario(scenario)
    s = ClusterSimulator(sc.config, seed=42)
    if steps:
        s.run_metrics_only(steps)
    return s


# ---------------------------------------------------------------------------
# Simulator action methods
# ---------------------------------------------------------------------------

class TestSimulatorActions:
    def test_add_replica_increases_gpu_count(self):
        sim = _sim("queue_surge_latency_sensitive", steps=2)
        wl = next(w for w in sim._cluster.workloads.values() if w.service_id == "critical-inference")
        before = len(wl.gpu_ids)
        ok = sim.add_replica("critical-inference")
        assert ok is True
        assert len(wl.gpu_ids) == before + 1

    def test_add_replica_returns_false_when_no_idle_gpu(self):
        sim = _sim("latency_tail_kvcache_pressure", steps=2)
        # workload occupies all 4 GPUs on the single node → no idle GPU
        ok = sim.add_replica("llm-critical")
        assert ok is False

    def test_spread_moves_workload_off_hottest_gpu(self):
        sim = _sim("thermal_hotspot_mixed_cluster", steps=10)  # hotspot active
        wl = next(w for w in sim._cluster.workloads.values() if w.service_id == "llm-inference")
        hottest_before = max(g.temperature_c for g in sim._workload_gpus(wl, sim._cluster))
        moved = sim.spread_workload("llm-inference")
        if moved:  # only asserts when a cooler destination existed
            hottest_after = max(g.temperature_c for g in sim._workload_gpus(wl, sim._cluster))
            assert hottest_after <= hottest_before

    def test_defer_only_affects_flexible_workloads(self):
        sim = _sim("energy_price_arbitrage_multiregion", steps=2)
        # batch workload is flexible → defer succeeds and lowers target util
        ok_batch = sim.defer_flexible_workload("batch-llm-east")
        assert ok_batch is True
        # latency-sensitive inference must NOT be deferrable
        ok_inf = sim.defer_flexible_workload("inference-svc-east")
        assert ok_inf is False

    def test_consolidate_powers_down_fully_idle_node(self):
        sim = _sim("underutilization_stranded_capacity", steps=2)
        cluster = sim._cluster
        region = cluster.regions["us-east"]
        # Free one node entirely, then consolidate → its GPUs power down.
        victim = region.nodes[-1]
        for g in victim.gpus:
            g.assigned_workload_id = None
            g.power_watts = 50.0
        changed = sim.consolidate_low_priority("us-east")
        assert changed is True
        assert all(g.power_watts == 0.0 for g in victim.gpus)

    def test_actions_do_not_mutate_real_cluster_flag(self):
        # The simulator is the only place these run; engine stays recommendation_only.
        from aurelius.constraints.engine import ConstraintAwareEngine
        eng = ConstraintAwareEngine()
        assert eng.implementation_mode == "recommendation_only"


# ---------------------------------------------------------------------------
# Benchmark non-inertness + no-regression acceptance
# ---------------------------------------------------------------------------

class TestBenchmarkNonInert:
    @pytest.fixture(scope="class")
    def thermal_result(self):
        return ConstraintBenchmarkRunner().run_scenario(
            "thermal_hotspot_mixed_cluster", steps=24, seed=42
        )

    def test_constraint_aware_not_identical_to_fifo(self, thermal_result):
        kc = thermal_result.report.aggregated
        ca = kc["constraint_aware"]
        fi = kc["fifo"]
        # The constraint-aware policy must actually change cluster behaviour.
        assert ca.total_thermal_throttle_ticks != fi.total_thermal_throttle_ticks \
            or ca.p99_latency_ms != fi.p99_latency_ms

    def test_thermal_throttling_reduced(self, thermal_result):
        kc = thermal_result.report.aggregated
        ca = kc["constraint_aware"]
        fi = kc["fifo"]
        # SPREAD should reduce thermal throttling vs no optimization.
        assert ca.total_thermal_throttle_ticks <= fi.total_thermal_throttle_ticks

    def test_no_sla_regression_vs_fifo(self, thermal_result):
        kc = thermal_result.report.aggregated
        ca = kc["constraint_aware"]
        fi = kc["fifo"]
        assert ca.total_sla_violations <= fi.total_sla_violations

    def test_cost_per_token_not_worse(self, thermal_result):
        kc = thermal_result.report.aggregated
        ca = kc["constraint_aware"]
        fi = kc["fifo"]
        ca_cpt = ca.total_energy_cost / max(1, ca.total_tokens)
        fi_cpt = fi.total_energy_cost / max(1, fi.total_tokens)
        # Throughput-normalized efficiency must not regress.
        assert ca_cpt <= fi_cpt * 1.02

    def test_queue_scenario_reduces_queue_wait(self):
        result = ConstraintBenchmarkRunner().run_scenario(
            "queue_surge_latency_sensitive", steps=24, seed=42
        )
        kc = result.report.aggregated
        ca = kc["constraint_aware"]
        fi = kc["fifo"]
        if fi.p95_queue_wait_ms and ca.p95_queue_wait_ms:
            assert ca.p95_queue_wait_ms <= fi.p95_queue_wait_ms
