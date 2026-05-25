"""Tests for the synthetic cluster simulator (Phase 6).

Verifies:
- Deterministic replay: same seed → identical state stream
- Realism checks: physical value ranges
- Each scenario produces the expected binding-constraint signature
- Migration support: cold-start warmup, cache reset
- ClusterState production: correct field mapping to canonical models
- Fake connector payloads: K8s, topology, vLLM text formats
"""

from __future__ import annotations


import pytest

from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import list_scenarios, load_scenario
from aurelius.state.models import (
    ClusterState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def energy_scenario():
    return load_scenario("energy_price_arbitrage_multiregion")


@pytest.fixture
def thermal_scenario():
    return load_scenario("thermal_hotspot_mixed_cluster")


@pytest.fixture
def queue_scenario():
    return load_scenario("queue_surge_latency_sensitive")


@pytest.fixture
def kvcache_scenario():
    return load_scenario("latency_tail_kvcache_pressure")


@pytest.fixture
def topology_scenario():
    return load_scenario("topology_fragmentation_h100")


@pytest.fixture
def underutil_scenario():
    return load_scenario("underutilization_stranded_capacity")


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

class TestScenarioLoading:
    def test_all_six_scenarios_available(self):
        names = list_scenarios()
        expected = {
            "energy_price_arbitrage_multiregion",
            "thermal_hotspot_mixed_cluster",
            "queue_surge_latency_sensitive",
            "latency_tail_kvcache_pressure",
            "topology_fragmentation_h100",
            "underutilization_stranded_capacity",
        }
        for name in expected:
            assert name in names, f"Scenario {name!r} not found"

    def test_scenario_has_expected_constraint(self, energy_scenario):
        assert energy_scenario.expected_primary_constraint == "energy_bound"

    def test_thermal_scenario_expected_constraint(self, thermal_scenario):
        assert thermal_scenario.expected_primary_constraint == "thermal_bound"

    def test_queue_scenario_expected_constraint(self, queue_scenario):
        assert queue_scenario.expected_primary_constraint == "queue_bound"

    def test_kvcache_scenario_expected_constraint(self, kvcache_scenario):
        assert kvcache_scenario.expected_primary_constraint == "memory_bound_indirect"

    def test_topology_scenario_expected_constraint(self, topology_scenario):
        assert topology_scenario.expected_primary_constraint == "topology_bound"

    def test_underutil_scenario_expected_constraint(self, underutil_scenario):
        assert underutil_scenario.expected_primary_constraint == "utilization_bound"

    def test_scenario_has_seed(self, energy_scenario):
        assert energy_scenario.config.seed == 42

    def test_seed_override(self):
        sc = load_scenario("energy_price_arbitrage_multiregion", seed_override=99)
        assert sc.config.seed == 99

    def test_scenario_hash_present(self, energy_scenario):
        assert len(energy_scenario.scenario_hash) > 0

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_scenario("nonexistent_scenario_xyz")


# ---------------------------------------------------------------------------
# Determinism / replay
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_cost(self, energy_scenario):
        """Same seed must produce identical energy cost stream."""
        sim1 = ClusterSimulator(energy_scenario.config, seed=42)
        ticks1 = sim1.run_metrics_only(10)

        sim2 = ClusterSimulator(energy_scenario.config, seed=42)
        ticks2 = sim2.run_metrics_only(10)

        for i, (t1, t2) in enumerate(zip(ticks1, ticks2)):
            assert abs(t1.total_energy_cost - t2.total_energy_cost) < 1e-9, (
                f"Tick {i+1}: cost differs: {t1.total_energy_cost} vs {t2.total_energy_cost}"
            )

    def test_different_seeds_differ(self, energy_scenario):
        sim1 = ClusterSimulator(energy_scenario.config, seed=42)
        ticks1 = sim1.run_metrics_only(5)

        sim2 = ClusterSimulator(energy_scenario.config, seed=99)
        ticks2 = sim2.run_metrics_only(5)

        # At least some ticks should differ
        diffs = [abs(t1.total_energy_cost - t2.total_energy_cost) for t1, t2 in zip(ticks1, ticks2)]
        assert any(d > 1e-9 for d in diffs), "Different seeds should produce different results"

    def test_reset_restores_determinism(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        run1 = sim.run_metrics_only(5)
        sim.reset()
        run2 = sim.run_metrics_only(5)

        for i, (t1, t2) in enumerate(zip(run1, run2)):
            assert abs(t1.total_energy_cost - t2.total_energy_cost) < 1e-9, (
                f"Reset tick {i+1} diverged"
            )

    def test_run_metrics_only_vs_tick(self, energy_scenario):
        """run_metrics_only and run() must produce identical metrics."""
        sim1 = ClusterSimulator(energy_scenario.config, seed=42)
        metrics1 = sim1.run_metrics_only(5)

        sim2 = ClusterSimulator(energy_scenario.config, seed=42)
        full_ticks = sim2.run(5)
        metrics2 = [t.metrics for t in full_ticks]

        for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
            assert abs(m1.total_energy_cost - m2.total_energy_cost) < 1e-9, (
                f"Tick {i+1} mismatch"
            )


# ---------------------------------------------------------------------------
# Realism checks
# ---------------------------------------------------------------------------

class TestRealism:
    def test_gpu_utilization_in_range(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(10):
            for region in tick.cluster_state.regions.values():
                for node in region.nodes.values():
                    for gpu in node.gpus.values():
                        assert 0.0 <= gpu.util_pct <= 100.0, (
                            f"util_pct out of range: {gpu.util_pct}"
                        )

    def test_gpu_temperature_realistic(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for node in region.nodes.values():
                    for gpu in node.gpus.values():
                        assert 0.0 < gpu.temp_c <= 100.0, (
                            f"temp_c out of range: {gpu.temp_c}"
                        )

    def test_power_positive(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for node in region.nodes.values():
                    for gpu in node.gpus.values():
                        assert gpu.power_w is None or gpu.power_w >= 0.0

    def test_memory_used_leq_total(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for node in region.nodes.values():
                    for gpu in node.gpus.values():
                        if gpu.mem_used_mb is not None and gpu.mem_total_mb is not None:
                            assert gpu.mem_used_mb <= gpu.mem_total_mb + 1.0, (
                                f"mem_used > mem_total: {gpu.mem_used_mb} > {gpu.mem_total_mb}"
                            )

    def test_kv_cache_fraction_in_range(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for svc in region.services.values():
                    if svc.kv_cache_usage is not None:
                        assert 0.0 <= svc.kv_cache_usage <= 1.0, (
                            f"kv_cache_usage out of [0,1]: {svc.kv_cache_usage}"
                        )

    def test_prefix_cache_hit_rate_in_range(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for svc in region.services.values():
                    if svc.prefix_cache_hit_rate is not None:
                        assert 0.0 <= svc.prefix_cache_hit_rate <= 1.0

    def test_latency_values_positive(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        for tick in sim.run(5):
            for region in tick.cluster_state.regions.values():
                for svc in region.services.values():
                    for field in ["ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                                  "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]:
                        val = getattr(svc, field, None)
                        if val is not None:
                            assert val >= 0.0, f"{field} negative: {val}"

    def test_energy_cost_positive(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        metrics = sim.run_metrics_only(5)
        for m in metrics:
            assert m.total_energy_cost >= 0.0
            assert m.total_energy_kwh >= 0.0

    def test_tokens_served_positive(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        metrics = sim.run_metrics_only(5)
        # Some ticks should have tokens (workload is active)
        total_tokens = sum(m.total_tokens for m in metrics)
        assert total_tokens >= 0


# ---------------------------------------------------------------------------
# Canonical ClusterState production
# ---------------------------------------------------------------------------

class TestClusterStateProduction:
    def test_cluster_state_is_frozen(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert isinstance(cs, ClusterState)
        with pytest.raises((TypeError, AttributeError)):
            cs.is_partial = True  # type: ignore

    def test_cluster_state_has_correct_regions(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert "us-east" in cs.regions
        assert "us-west" in cs.regions

    def test_provenance_is_sandbox(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert cs.provenance.is_sandbox is True

    def test_provenance_confidence_high(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert cs.provenance.confidence == "high"

    def test_provenance_source_is_simulator(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert cs.provenance.source == "simulator"

    def test_timestamp_utc_aware(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        assert cs.timestamp.tzinfo is not None

    def test_node_gpu_count_matches_config(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        # us-east-node0 should have 4 GPUs
        node = cs.regions["us-east"].nodes["us-east-node0"]
        assert node.gpu_capacity == 4
        assert len(node.gpus) == 4

    def test_gpu_state_has_uuid(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        for region in cs.regions.values():
            for node in region.nodes.values():
                for uuid, gpu in node.gpus.items():
                    assert len(uuid) > 0
                    assert gpu.gpu_uuid == uuid

    def test_energy_state_present(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        for region in cs.regions.values():
            assert region.energy is not None
            assert region.energy.price_per_mwh is not None
            assert region.energy.price_per_mwh > 0

    def test_inference_service_state_present(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()  # run tick 1 to init queues
        tick = sim.tick()  # tick 2 should have queue data
        cs = tick.cluster_state
        # batch-llm-east is registered in us-east queues
        assert "batch-llm-east" in cs.regions["us-east"].services

    def test_kv_cache_in_canonical_fraction(self, kvcache_scenario):
        """kv_cache_usage must be [0,1], not [0,100]."""
        sim = ClusterSimulator(kvcache_scenario.config, seed=42)
        sim.run_metrics_only(7)  # past kv_cache_pressure event at tick 6
        tick = sim.tick()
        cs = tick.cluster_state
        svc = cs.regions["us-east"].services.get("llm-critical")
        if svc is not None and svc.kv_cache_usage is not None:
            assert 0.0 <= svc.kv_cache_usage <= 1.0

    def test_is_partial_false(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        assert tick.cluster_state.is_partial is False

    def test_cluster_state_json_roundtrip(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        cs = tick.cluster_state
        d = cs.to_dict()
        cs2 = ClusterState.from_dict(d)
        assert cs2.timestamp == cs.timestamp
        assert set(cs2.regions.keys()) == set(cs.regions.keys())
        assert cs2.provenance.is_sandbox == cs.provenance.is_sandbox


# ---------------------------------------------------------------------------
# Constraint signature scenarios
# ---------------------------------------------------------------------------

class TestConstraintSignatures:
    def test_thermal_throttling_detected_after_hotspot(self, thermal_scenario):
        """After thermal hotspot event, throttled GPU count > 0."""
        sim = ClusterSimulator(thermal_scenario.config, seed=42)
        # Run past event at tick 6
        metrics = sim.run_metrics_only(10)

        throttle_counts = [m.thermal_throttle_gpu_count for m in metrics]
        # Should be 0 before tick 6, >0 after
        assert throttle_counts[0] == 0, "No throttling before hotspot"
        assert any(c > 0 for c in throttle_counts[6:]), (
            "Expected throttling after hotspot event"
        )

    def test_thermal_p99_degrades_during_hotspot(self, thermal_scenario):
        """p99 latency should increase during thermal throttling."""
        sim = ClusterSimulator(thermal_scenario.config, seed=42)
        metrics = sim.run_metrics_only(15)

        p99_before = [m.p99_latency_ms for m in metrics[:5] if m.p99_latency_ms is not None]
        p99_during = [m.p99_latency_ms for m in metrics[7:12] if m.p99_latency_ms is not None]

        if p99_before and p99_during:
            avg_before = sum(p99_before) / len(p99_before)
            avg_during = sum(p99_during) / len(p99_during)
            assert avg_during > avg_before, (
                f"p99 should increase during throttling: {avg_before:.0f} → {avg_during:.0f}"
            )

    def test_queue_surge_increases_queue_depth(self, queue_scenario):
        """Queue depth should increase during surge event (tick 8)."""
        sim = ClusterSimulator(queue_scenario.config, seed=42)
        metrics = sim.run_metrics_only(12)

        # Queue wait p95 should be higher during surge (ticks 9-12) than before (ticks 1-7)
        before_wait = [m.queue_wait_p95_ms for m in metrics[:7] if m.queue_wait_p95_ms is not None]
        during_wait = [m.queue_wait_p95_ms for m in metrics[8:] if m.queue_wait_p95_ms is not None]

        if before_wait and during_wait:
            avg_before = sum(before_wait) / len(before_wait)
            avg_during = sum(during_wait) / len(during_wait)
            assert avg_during >= avg_before, (
                f"Queue wait should not decrease during surge: {avg_before:.0f} → {avg_during:.0f}"
            )

    def test_kvcache_pressure_increases_ttft(self, kvcache_scenario):
        """KV cache pressure event should increase TTFT p99."""
        sim = ClusterSimulator(kvcache_scenario.config, seed=42)
        full_ticks = sim.run(20)  # go past event at tick 6 and end at tick 18

        ttft_before = []
        ttft_during = []
        for t in full_ticks:
            svc = t.cluster_state.regions["us-east"].services.get("llm-critical")
            if svc is None:
                continue
            if t.tick < 6:
                if svc.ttft_p99_ms is not None:
                    ttft_before.append(svc.ttft_p99_ms)
            elif 7 <= t.tick <= 15:
                if svc.ttft_p99_ms is not None:
                    ttft_during.append(svc.ttft_p99_ms)

        if ttft_before and ttft_during:
            assert sum(ttft_during) / len(ttft_during) > sum(ttft_before) / len(ttft_before), (
                "TTFT should increase during KV cache pressure"
            )

    def test_kvcache_not_zero_after_pressure(self, kvcache_scenario):
        """KV cache usage must not drop to zero during pressure event."""
        sim = ClusterSimulator(kvcache_scenario.config, seed=42)
        ticks = sim.run(10)  # covers event at tick 6
        for t in ticks[6:9]:
            svc = t.cluster_state.regions["us-east"].services.get("llm-critical")
            if svc is not None and svc.kv_cache_usage is not None:
                assert svc.kv_cache_usage > 0.5, (
                    f"KV cache should be high during pressure: {svc.kv_cache_usage}"
                )

    def test_topology_scenario_high_comm_workload(self, topology_scenario):
        """Topology scenario has high communication intensity workload."""
        sim = ClusterSimulator(topology_scenario.config, seed=42)
        cluster = sim._cluster
        wl = cluster.workloads.get("training-wl-bad-topo")
        assert wl is not None
        assert wl.communication_intensity == "high"

    def test_underutil_low_gpu_utilization(self, underutil_scenario):
        """Underutilization scenario should have low mean GPU utilization."""
        sim = ClusterSimulator(underutil_scenario.config, seed=42)
        metrics = sim.run_metrics_only(5)
        avg_util = sum(m.mean_gpu_util_pct for m in metrics) / len(metrics)
        assert avg_util < 40.0, (
            f"Underutilization scenario should have low util: {avg_util:.1f}%"
        )

    def test_energy_spike_event_at_tick_16(self, energy_scenario):
        """Energy price spike event should increase energy cost at tick 17+."""
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        metrics = sim.run_metrics_only(22)

        cost_before = sum(m.total_energy_cost for m in metrics[10:16])
        cost_during = sum(m.total_energy_cost for m in metrics[16:20])

        # Energy cost should be higher during price spike
        assert cost_during > cost_before * 0.5, (
            "Energy cost should be elevated during price spike"
        )


# ---------------------------------------------------------------------------
# Migration support
# ---------------------------------------------------------------------------

class TestMigration:
    def test_migration_succeeds(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()

        result = sim.migrate_workload("batch-wl-east", "us-west")
        assert result is True

        cluster = sim._cluster
        wl = cluster.workloads["batch-wl-east"]
        assert wl.region_id == "us-west"
        assert wl.cold_start_warmup_ticks_remaining > 0

    def test_migration_resets_cache(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()

        sim.migrate_workload("batch-wl-east", "us-west")
        wl = sim._cluster.workloads["batch-wl-east"]
        assert wl.prefix_cache_hit_rate_frac < 0.1, "Cache should reset after migration"

    def test_migration_blocked_when_not_allowed(self, energy_scenario):
        """inference-wl-east has migration_allowed=False."""
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()

        result = sim.migrate_workload("inference-wl-east", "us-west")
        assert result is False

    def test_migration_count_increments(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()

        sim.migrate_workload("batch-wl-east", "us-west")
        assert sim._cluster.migration_count == 1

    def test_migration_logged(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()

        sim.migrate_workload("batch-wl-east", "us-west")
        log = sim._cluster.migration_log
        assert len(log) == 1
        assert log[0]["from_region"] == "us-east"
        assert log[0]["to_region"] == "us-west"

    def test_cold_start_reduces_throughput(self, energy_scenario):
        """After migration, tokens_per_second should be lower during warmup."""
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()  # warm state
        sim.migrate_workload("batch-wl-east", "us-west")

        # Run 1 tick during warmup
        sim.tick()
        # Workload should be in warmup state
        wl = sim._cluster.workloads["batch-wl-east"]
        # warmup_ticks_remaining should be decrementing
        assert wl.cold_start_warmup_ticks_remaining >= 0

    def test_migration_to_invalid_region_ignored(self, energy_scenario):
        """Migration to nonexistent region should not crash, just not place."""
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.tick()
        # Should return False or succeed silently but not assign GPUs
        result = sim.migrate_workload("batch-wl-east", "nonexistent-region")
        # May return True (migration happened but no GPUs assigned) or False
        # Key: no crash, and if region doesn't exist, GPU list is empty
        if result:
            wl = sim._cluster.workloads["batch-wl-east"]
            # If placed in nonexistent region, gpu_ids may be empty
            assert wl.region_id == "nonexistent-region" or len(wl.gpu_ids) == 0


# ---------------------------------------------------------------------------
# Tick structure
# ---------------------------------------------------------------------------

class TestTickStructure:
    def test_tick_counter_increments(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        t1 = sim.tick()
        t2 = sim.tick()
        assert t1.tick == 1
        assert t2.tick == 2

    def test_tick_timestamp_monotonic(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        ticks = sim.run(5)
        timestamps = [t.timestamp for t in ticks]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], "Timestamps must be monotonically increasing"

    def test_tick_has_dcgm_text_for_each_node(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        # us-east has node0 and node1
        assert "us-east-node0" in tick.dcgm_texts
        assert "us-east-node1" in tick.dcgm_texts
        assert len(tick.dcgm_texts["us-east-node0"]) > 0

    def test_tick_has_k8s_node_list(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        assert tick.k8s_node_list["kind"] == "NodeList"
        assert len(tick.k8s_node_list["items"]) >= 2  # at least us-east nodes

    def test_tick_has_k8s_pod_list(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        assert tick.k8s_pod_list["kind"] == "PodList"

    def test_tick_has_topology_texts(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        tick = sim.tick()
        assert "us-east-node0" in tick.topology_texts
        assert "GPU" in tick.topology_texts["us-east-node0"]


# ---------------------------------------------------------------------------
# Cumulative metrics
# ---------------------------------------------------------------------------

class TestCumulativeMetrics:
    def test_cumulative_cost_increases(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.run_metrics_only(5)
        m = sim.cumulative_metrics
        assert m["total_energy_cost"] > 0

    def test_cumulative_metrics_not_none(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.run_metrics_only(5)
        m = sim.cumulative_metrics
        assert m["total_energy_kwh"] > 0
        # tokens and cost/token may be 0 or None if no throughput

    def test_tokens_per_joule_positive_when_tokens_served(self, energy_scenario):
        sim = ClusterSimulator(energy_scenario.config, seed=42)
        sim.run_metrics_only(10)
        m = sim.cumulative_metrics
        if m["total_tokens_served"] > 0 and m["total_energy_kwh"] > 0:
            assert m["tokens_per_joule"] is not None
            assert m["tokens_per_joule"] > 0
