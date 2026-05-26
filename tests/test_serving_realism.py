"""Validation tests for the inference-serving realism layer (Mission 5).

Covers the audited realism gaps: convex queue saturation, exploding latency
tails, the replica/batching tradeoff, Erlang-C queueing, autoscaling lag +
anti-flap, migration cost, and — the headline product property — that
*aggressive* migration can now LOSE on latency because migrations destabilize
queues. Also checks the calibration-metadata system has no hidden constants.
"""

from __future__ import annotations

import random

from aurelius.benchmarks.constraint_runner import ConstraintBenchmarkRunner
from aurelius.simulation.cluster import serving
from aurelius.simulation.cluster.calibration import SERVING_PARAMS, calibration_table
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario

# ---------------------------------------------------------------------------
# Saturation + tails
# ---------------------------------------------------------------------------

class TestSaturationAndTails:
    def test_saturation_is_convex(self):
        amps = [serving.saturation_amplifier(r) for r in (0.5, 0.7, 0.85, 0.92, 0.97)]
        assert amps[0] == 1.0 and amps[1] == 1.0  # safe band: no amplification
        # Strictly increasing and convex (each jump bigger than the last) past safe.
        assert amps[2] < amps[3] < amps[4]
        assert (amps[4] - amps[3]) > (amps[3] - amps[2])

    def test_tails_explode_and_p99_faster_than_p95(self):
        p95_lo, p99_lo = serving.tail_multipliers(0.3)
        p95_hi, p99_hi = serving.tail_multipliers(0.95)
        assert p95_hi > p95_lo and p99_hi > p99_lo          # tails grow with load
        assert (p99_hi - p99_lo) > (p95_hi - p95_lo)        # p99 grows faster than p95
        assert p99_hi > p95_hi                              # p99 always above p95

    def test_erlang_c_monotonic_in_load(self):
        w_lo = serving.erlang_c_wait_s(0.5, 1.0, 1)
        w_hi = serving.erlang_c_wait_s(0.95, 1.0, 1)
        assert 0 < w_lo < w_hi
        assert serving.erlang_c_wait_s(2.0, 1.0, 1) == float("inf")  # rho>1 → unstable


# ---------------------------------------------------------------------------
# Batching / replica tradeoff
# ---------------------------------------------------------------------------

class TestBatchingTradeoff:
    def test_more_replicas_lower_batch_efficiency(self):
        # Same offered concurrency spread over more replicas → thinner batches.
        e1 = serving.batching_efficiency(32, 1)
        e2 = serving.batching_efficiency(32, 2)
        e4 = serving.batching_efficiency(32, 4)
        assert e1 == 1.0
        assert e1 > e2 > e4 or (e1 > e2 and e4 <= e2)  # monotone non-increasing, floored

    def test_batching_efficiency_floored(self):
        assert serving.batching_efficiency(1, 64) >= 0.4  # never collapses to zero


# ---------------------------------------------------------------------------
# TTFT / TPOT decomposition
# ---------------------------------------------------------------------------

class TestLatencyDecomposition:
    def test_ttft_grows_with_each_component(self):
        base = serving.ttft_ms(10, 100, 4, 0.2)
        assert serving.ttft_ms(10, 2000, 4, 0.2) > base     # prompt length
        assert serving.ttft_ms(10, 100, 64, 0.2) > base     # active-seq contention
        assert serving.ttft_ms(10, 100, 4, 0.95) > base     # KV pressure
        assert serving.ttft_ms(5000, 100, 4, 0.2) > base    # queue wait

    def test_tpot_grows_with_active_tokens(self):
        assert serving.tpot_ms(20, 512) > serving.tpot_ms(20, 8)


# ---------------------------------------------------------------------------
# Bursty arrivals (deterministic given seed)
# ---------------------------------------------------------------------------

class TestBurstyArrivals:
    def test_burst_state_is_seed_deterministic(self):
        def run():
            rng = random.Random(7)
            st = False
            return [st := serving.step_burst_state(st, rng) for _ in range(50)]
        assert run() == run()

    def test_bursts_occur_and_end(self):
        rng = random.Random(1)
        st = False
        states = [st := serving.step_burst_state(st, rng) for _ in range(500)]
        assert any(states) and not all(states)  # bursts happen and are not permanent


# ---------------------------------------------------------------------------
# Autoscaling lag + anti-flap
# ---------------------------------------------------------------------------

class TestAutoscaling:
    def test_added_replica_triggers_warmup_lag(self):
        sc = load_scenario("queue_surge_latency_sensitive")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run_metrics_only(2)
        assert sim.add_replica("critical-inference") is True
        wl = next(w for w in sim._cluster.workloads.values()
                  if w.service_id == "critical-inference")
        # New replica is NOT instantly ready (provision + load + warmup).
        assert wl.cold_start_warmup_ticks_remaining > 0

    def test_anti_flap_cooldown_blocks_immediate_rescale(self):
        sc = load_scenario("queue_surge_latency_sensitive")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run_metrics_only(2)
        assert sim.add_replica("critical-inference") is True
        # A second scale within the stabilization window is refused.
        assert sim.add_replica("critical-inference") is False


# ---------------------------------------------------------------------------
# Aggressive migration can LOSE (headline product property)
# ---------------------------------------------------------------------------

class TestAggressiveMigrationCanLose:
    def test_greedy_energy_loses_on_latency_in_energy_scenario(self):
        result = ConstraintBenchmarkRunner().run_scenario(
            "energy_price_arbitrage_multiregion", steps=24, seed=42
        )
        agg = result.report.aggregated
        greedy = agg["greedy_energy"]
        ca = agg["constraint_aware"]
        # greedy_energy migrates aggressively for cheap energy...
        assert greedy.total_migrations > ca.total_migrations
        # ...but its migrations destabilize queues → p99 explodes far past the
        # constraint-aware policy, which protects latency.
        assert greedy.p99_latency_ms > ca.p99_latency_ms * 5

    def test_constraint_aware_no_sla_regression_vs_fifo(self):
        for scen in ("thermal_hotspot_mixed_cluster", "queue_surge_latency_sensitive",
                     "energy_price_arbitrage_multiregion"):
            result = ConstraintBenchmarkRunner().run_scenario(scen, steps=24, seed=42)
            agg = result.report.aggregated
            assert agg["constraint_aware"].total_sla_violations <= agg["fifo"].total_sla_violations


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden magic constants
# ---------------------------------------------------------------------------

class TestCalibrationMetadata:
    def test_every_param_has_full_provenance(self):
        assert SERVING_PARAMS
        for name, p in SERVING_PARAMS.items():
            assert p.source and p.source_type and p.calibration_notes, name
            assert p.confidence in ("high", "medium", "low"), name

    def test_calibration_table_serializable(self):
        table = calibration_table()
        assert table and all("confidence" in row and "value" in row for row in table)

    def test_params_are_overridable(self):
        # Any uncertain assumption must be configurable (audit requirement).
        from aurelius.simulation.cluster.calibration import serving_value
        assert serving_value("saturation_convexity", {"saturation_convexity": 1.5}) == 1.5
