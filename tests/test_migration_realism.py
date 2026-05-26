"""Validation tests for the migration / rerouting / drain / cold-start layer.

Covers the audited realism gaps the migration upgrade targets:
- Kubernetes-style drain (T_evict + T_grace + T_rebind), heavy-tailed grace;
- PodDisruptionBudget blocking (drain stalls);
- request rerouting = max(proxy, rtt, accept) + proxy saturation bottleneck;
- cache-loss penalty ΔT_prefill scaling with prompt length;
- heavy-tailed, engine-specific cold starts (compile-heavy TensorRT-LLM);
- batching degradation under churn; migration TAIL uplift (not p50-only);
- composite migration cost C_mig; migration governor veto; phased rollout;
- scale-from-zero amplification;
- emergent: naive migration LOSES, proxy bottlenecks dominate, warm pools win,
  governor restraint helps;
- calibration metadata has no hidden constants.

Pure functions are deterministic; integration scenarios use a fixed seed.
"""

from __future__ import annotations

import random

from aurelius.benchmarks.constraint_runner import ConstraintBenchmarkRunner
from aurelius.simulation.cluster import migration as mig
from aurelius.simulation.cluster.calibration import (
    ENGINE_STARTUP_PROFILES,
    MIGRATION_PARAMS,
    calibration_table,
    engine_profile_table,
    migration_value,
    resolve_engine_profile,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Kubernetes drain
# ---------------------------------------------------------------------------

class TestDrain:
    def test_drain_sums_three_stages_and_is_positive(self):
        d = mig.drain_seconds(_rng(), None)
        # ≥ evict + rebind (grace ≥ 0), and bounded above by evict + 2*grace + rebind.
        lo = migration_value("drain_evict_seconds") + migration_value("drain_rebind_seconds")
        hi = lo + 2.0 * migration_value("drain_grace_seconds")
        assert lo <= d <= hi

    def test_drain_is_seed_deterministic(self):
        assert mig.drain_seconds(_rng(7)) == mig.drain_seconds(_rng(7))

    def test_drain_is_heavy_tailed(self):
        rng = _rng(1)
        samples = [mig.drain_seconds(rng) for _ in range(400)]
        mean = sum(samples) / len(samples)
        assert max(samples) > mean * 1.3   # a right tail exists

    def test_pdb_blocks_when_zero(self):
        assert mig.pdb_blocks_migration(0) is True
        assert mig.pdb_blocks_migration(1) is False


# ---------------------------------------------------------------------------
# Rerouting + proxy
# ---------------------------------------------------------------------------

class TestReroute:
    def test_reroute_is_max_of_stages(self):
        assert mig.reroute_seconds(0.1, 0.05, 0.2) == 0.2

    def test_proxy_saturation_rises_past_capacity(self):
        cap = migration_value("proxy_capacity_rps_per_replica")
        lo = mig.proxy_saturation_factor(cap * 1 * 0.5, 1)   # under capacity
        hi = mig.proxy_saturation_factor(cap * 1 * 0.98, 1)  # near capacity
        assert lo == 1.0
        assert hi > 1.0

    def test_more_replicas_raise_proxy_capacity(self):
        offered = migration_value("proxy_capacity_rps_per_replica") * 1.5
        one = mig.proxy_saturation_factor(offered, 1)
        two = mig.proxy_saturation_factor(offered, 2)
        assert one > two   # spreading over more replicas relieves the proxy

    def test_proxy_factor_capped(self):
        assert mig.proxy_saturation_factor(1e9, 1) <= 100.0


# ---------------------------------------------------------------------------
# Cache-loss penalty
# ---------------------------------------------------------------------------

class TestCacheLoss:
    def test_scales_with_prompt_length(self):
        p1 = mig.cache_loss_penalty_ms(1000, 0.0, 0.25)
        p2 = mig.cache_loss_penalty_ms(4000, 0.0, 0.25)
        assert p2 == 4 * p1

    def test_zero_when_fully_warm_after(self):
        assert mig.cache_loss_penalty_ms(4000, 1.0, 0.25) == 0.0


# ---------------------------------------------------------------------------
# Cold start (heavy-tailed, engine-specific, bimodal)
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_decomposed_total_is_sum(self):
        cs = mig.cold_start_seconds("vllm", _rng())
        assert abs(cs.total_seconds
                   - (cs.t_node + cs.t_pull + cs.t_load + cs.t_gpu_transfer + cs.t_warmup)) < 1e-9

    def test_compile_heavy_engine_is_slower(self):
        # Average over seeds: TensorRT-LLM (compile-heavy) ≫ vLLM warmup.
        def mean_warmup(engine):
            rng = _rng(3)
            return sum(mig.cold_start_seconds(engine, rng).t_warmup for _ in range(50)) / 50
        assert mean_warmup("tensorrt-llm") > mean_warmup("vllm") * 3

    def test_cold_start_heavy_tailed(self):
        rng = _rng(2)
        totals = [mig.cold_start_seconds("vllm", rng).total_seconds for _ in range(400)]
        mean = sum(totals) / len(totals)
        assert max(totals) > mean * 1.5   # heavy upper tail, not Gaussian

    def test_first_compile_path_occurs(self):
        rng = _rng(5)
        flags = [mig.cold_start_seconds("vllm", rng).first_compile for _ in range(300)]
        assert any(flags) and not all(flags)   # bimodal

    def test_seconds_to_warmup_ticks_floor(self):
        # Multi-minute cold start is sub-tick at hourly granularity → floor 1.
        assert mig.seconds_to_warmup_ticks(200.0, 1.0) == 1
        assert mig.seconds_to_warmup_ticks(0.0, 1.0) == 0


# ---------------------------------------------------------------------------
# Batching under churn + tail uplift
# ---------------------------------------------------------------------------

class TestChurnAndTail:
    def test_batch_efficiency_degrades_with_churn(self):
        e0 = mig.batch_efficiency_under_churn(1.0, 0.0)
        e1 = mig.batch_efficiency_under_churn(1.0, 1.0)
        e3 = mig.batch_efficiency_under_churn(1.0, 3.0)
        assert e0 == 1.0 and e1 < e0 and e3 < e1
        assert e3 >= migration_value("batch_churn_floor") * 1.0 - 1e-9

    def test_tail_uplift_grows_with_drivers(self):
        lo = mig.tail_uplift(0.0, 0.0, 0.0, 0.0)
        hi = mig.tail_uplift(1.0, 1.0, 1.0, 1.0)
        assert lo == migration_value("tail_uplift_base")
        assert hi > lo
        assert hi <= migration_value("tail_uplift_max") + 1e-9


# ---------------------------------------------------------------------------
# Composite migration cost
# ---------------------------------------------------------------------------

class TestMigrationCost:
    def test_cost_terms_present_and_positive(self):
        c = mig.migration_cost(
            "vllm", prompt_tokens=2048, hit_rate_before=0.8,
            prefill_cost_per_token_ms=0.25, rng=_rng(), churn_rate=1.0,
        )
        assert c.t_transfer_ms > 0 and c.t_warmup_ms > 0 and c.t_requeue_ms > 0
        assert c.t_cacheloss_ms > 0
        assert 0 < c.t_batchloss_factor <= 1.0
        assert c.t_tail_mult >= migration_value("tail_uplift_base")
        assert c.startup_penalty_ms > 0

    def test_scale_from_zero_amplifies_tail(self):
        kw = dict(prompt_tokens=1024, hit_rate_before=0.5,
                  prefill_cost_per_token_ms=0.25)
        warm = mig.migration_cost("vllm", rng=_rng(9), from_zero=False, **kw)
        cold = mig.migration_cost("vllm", rng=_rng(9), from_zero=True, **kw)
        assert cold.t_tail_mult > warm.t_tail_mult

    def test_compile_heavy_costs_more(self):
        kw = dict(prompt_tokens=1024, hit_rate_before=0.5,
                  prefill_cost_per_token_ms=0.25)
        v = mig.migration_cost("vllm", rng=_rng(4), **kw)
        t = mig.migration_cost("tensorrt-llm", rng=_rng(4), **kw)
        assert t.t_warmup_ms > v.t_warmup_ms


# ---------------------------------------------------------------------------
# Governor + phased rollout
# ---------------------------------------------------------------------------

class TestGovernorAndRollout:
    def test_pdb_unavailable_vetoes(self):
        reason = mig.migration_veto_reason(
            queue_depth=0, locality_confidence=0.0, p95_unstable=False,
            rollout_instability=0.0, pdb_available=0, warmup_incomplete=False,
            startup_heavy=False, scale_from_zero=False,
        )
        assert reason == "pdb_unavailable"

    def test_queue_pressure_vetoes(self):
        reason = mig.migration_veto_reason(
            queue_depth=1e9, locality_confidence=0.0, p95_unstable=False,
            rollout_instability=0.0, pdb_available=2, warmup_incomplete=False,
            startup_heavy=False, scale_from_zero=False,
        )
        assert reason == "queue_pressure_high"

    def test_strong_affinity_vetoes(self):
        reason = mig.migration_veto_reason(
            queue_depth=0, locality_confidence=0.95, p95_unstable=False,
            rollout_instability=0.0, pdb_available=2, warmup_incomplete=False,
            startup_heavy=False, scale_from_zero=False,
        )
        assert reason == "cache_affinity_strong"

    def test_no_veto_when_clear(self):
        assert mig.migration_veto_reason(
            queue_depth=10, locality_confidence=0.2, p95_unstable=False,
            rollout_instability=0.0, pdb_available=2, warmup_incomplete=False,
            startup_heavy=False, scale_from_zero=False,
        ) is None

    def test_traffic_fraction_advances_only_when_stable(self):
        assert mig.next_traffic_fraction(0.1, stable=True) == 0.25
        assert mig.next_traffic_fraction(0.1, stable=False) == 0.1
        assert mig.next_traffic_fraction(0.5, stable=True) == 1.0

    def test_rollback_triggers_past_budget(self):
        assert mig.should_rollback(5000, 2000) is True    # 2.5× budget
        assert mig.should_rollback(3000, 2000) is False   # 1.5× budget


# ---------------------------------------------------------------------------
# Integration: emergent simulator behaviour
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_migration_injects_startup_penalty_and_tail(self):
        sc = load_scenario("startup_heavy_migration_trtllm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(6)
        assert sim.migrate_workload("trt-wl", "us-west") is True
        wl = sim._cluster.workloads["trt-wl"]
        m = wl.migration
        assert m.warmup.ticks_remaining > 0
        assert m.warmup.startup_penalty_ms > 0
        assert m.tail.uplift_mult > 1.0
        assert m.startup.last_cold_seconds > 0
        assert m.migration.migration_count == 1

    def test_pdb_blocks_migration_in_engine(self):
        sc = load_scenario("startup_heavy_migration_trtllm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(2)
        # Set a PDB that forbids any disruption (min_available == replicas).
        wl = sim._cluster.workloads["trt-wl"]
        assert sim.set_pdb("trt-inference", len(wl.gpu_ids)) is True
        assert sim.migrate_workload("trt-wl", "us-west") is False
        assert wl.migration.migration.veto_count >= 1
        assert wl.migration.migration.last_veto_reason == "pdb_unavailable"

    def test_governor_vetoes_under_strong_affinity(self):
        sc = load_scenario("startup_heavy_migration_trtllm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(8)  # warm the cache so locality confidence is high
        wl = sim._cluster.workloads["trt-wl"]
        # With governor on, a warm high-affinity workload should be protected.
        if wl.cache.locality.confidence >= 0.7:
            assert sim.safe_migrate_workload("trt-wl", "us-west") is False
            assert wl.region_id == "us-east"

    def test_naive_migration_loses_on_ttft(self):
        # Compile-heavy engine + abrupt energy-greedy rerouting → TTFT collapse.
        result = ConstraintBenchmarkRunner().run_scenario(
            "startup_heavy_migration_trtllm", steps=24, seed=42
        )
        agg = result.report.aggregated
        greedy = agg["greedy_energy"]
        fifo = agg["fifo"]
        assert greedy.total_reroutes > 0
        assert greedy.total_cold_starts > 0
        # Naive rerouting drowns TTFT vs the stay-put baseline.
        assert greedy.ttft_p99_ms > fifo.ttft_p99_ms * 5

    def test_proxy_bottleneck_dominates(self):
        sc = load_scenario("proxy_bottleneck_ingress")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(12)
        wl = sim._cluster.workloads["proxy-wl"]
        # Proxy saturates well past 1.0 under the high offered load.
        assert wl.migration.proxy.saturation_factor > 1.5

    def test_warm_pool_reduces_startup_penalty(self):
        def startup_after_migrate(warm: bool) -> float:
            sc = load_scenario("startup_heavy_migration_trtllm")
            sim = ClusterSimulator(sc.config, seed=42)
            sim.run(6)
            if warm:
                sim.set_warm_pool("trt-inference", size=4)
            sim.migrate_workload("trt-wl", "us-west")
            return sim._cluster.workloads["trt-wl"].migration.warmup.startup_penalty_ms
        assert startup_after_migrate(warm=True) < startup_after_migrate(warm=False)

    def test_phased_rollout_starts_small(self):
        sc = load_scenario("startup_heavy_migration_trtllm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(6)
        assert sim.migrate_workload_phased("trt-wl", "us-west") is True
        m = sim._cluster.workloads["trt-wl"].migration
        assert m.rollout.active is True
        assert m.traffic_shift.fraction == 0.1   # canary starts at 10%

    def test_scale_from_zero_amplifies(self):
        # Drain a workload to zero replicas, then scale up → scale-from-zero path.
        sc = load_scenario("startup_heavy_migration_trtllm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(4)
        wl = sim._cluster.workloads["trt-wl"]
        # Free all GPUs to simulate scale-to-zero.
        for region in sim._cluster.regions.values():
            for node in region.nodes:
                for gpu in node.gpus:
                    if gpu.gpu_id in wl.gpu_ids:
                        gpu.assigned_workload_id = None
        wl.gpu_ids = []
        assert sim.add_replica("trt-inference") is True
        assert wl.migration.coldstart.scale_from_zero is True
        assert wl.migration.tail.uplift_mult > 1.0

    def test_deterministic_under_seed(self):
        def fp(seed):
            sc = load_scenario("startup_heavy_migration_trtllm")
            sim = ClusterSimulator(sc.config, seed=seed)
            sim.run(6)
            sim.migrate_workload("trt-wl", "us-west")
            ms = sim.run_metrics_only(4)
            return [(round(m.startup_latency_s_max or 0, 4), m.reroute_count,
                     round(m.ttft_p99_ms or 0, 2)) for m in ms]
        assert fp(42) == fp(42)
        assert fp(42) != fp(7)


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden constants
# ---------------------------------------------------------------------------

class TestCalibrationMetadata:
    def test_migration_params_have_full_provenance(self):
        assert MIGRATION_PARAMS
        for name, p in MIGRATION_PARAMS.items():
            assert p.source and p.source_type and p.calibration_notes, name
            assert p.confidence in ("high", "medium", "low"), name

    def test_combined_table_includes_migration_group(self):
        groups = {row.get("group") for row in calibration_table()}
        assert "migration" in groups

    def test_engine_profiles_have_provenance(self):
        rows = engine_profile_table()
        assert rows
        for r in rows:
            assert r["total_mean_seconds"] > 0 and r["source"] and r["source_type"]
        # TensorRT-LLM is the compile-heavy outlier.
        trt = next(r for r in rows if r["name"] == "tensorrt-llm")
        assert trt["compile_heavy"] is True

    def test_migration_params_overridable(self):
        assert migration_value("drain_grace_seconds", {"drain_grace_seconds": 60}) == 60.0

    def test_all_engines_resolvable(self):
        for name in ENGINE_STARTUP_PROFILES:
            assert resolve_engine_profile(name).total_mean_seconds() > 0
