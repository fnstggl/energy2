"""Validation tests for the topology / communication / placement realism layer.

Covers the audited gaps the topology upgrade targets:
- distinct fabric regimes (NVSwitch / NVLink / PCIe / IB / cross-rack / region)
  are NOT interchangeable bandwidth pipes;
- latency-bandwidth message cost T(m) = alpha + m/B_eff with small-message
  latency amplification;
- ring / tree collective approximations + collective amplification;
- topology distance ladder → 0-1 placement quality score;
- communication throughput penalty (TP collapses off NVSwitch; batch survives);
- synchronization-stall penalty for bulk-synchronous workloads;
- communication-induced p95/p99 tail amplification (faster than the mean);
- fabric contention / NIC saturation / MoE all-to-all hotspots;
- topology telemetry confidence (missing != ideal proximity) + score discount;
- topology-aware migration veto (sync-heavy / comm-heavy jobs are pinned);
- emergent: a TP job split across racks collapses while NVSwitch co-location
  stays safe; cross-region migration of a comm-heavy job is vetoed;
- calibration metadata has no hidden constants.

Pure functions are deterministic; integration scenarios use a fixed seed.
"""

from __future__ import annotations

import random

from aurelius.simulation.cluster import topology as topo
from aurelius.simulation.cluster.calibration import (
    FABRIC_REGIMES,
    NVLINK_GENERATIONS,
    TOPOLOGY_DISTANCE_LADDER,
    TOPOLOGY_PARAMS,
    WORKLOAD_COMM_PROFILES,
    calibration_table,
    fabric_regime_table,
    nvlink_generation_table,
    resolve_comm_profile,
    resolve_fabric_regime,
    topology_value,
    workload_comm_profile_table,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _run(name: str, steps: int = 20, seed: int | None = None):
    cfg = load_scenario(name).config
    sim = ClusterSimulator(cfg, seed=seed if seed is not None else cfg.seed)
    ms = sim.run_metrics_only(steps)
    return sim, ms


# ---------------------------------------------------------------------------
# Fabric regimes are distinct (not interchangeable pipes)
# ---------------------------------------------------------------------------

class TestFabricRegimes:
    def test_distance_ladder_is_monotonic(self):
        # same GPU 0 … cross-region 6, strictly increasing where defined.
        assert TOPOLOGY_DISTANCE_LADDER["intra_gpu"] == 0
        assert TOPOLOGY_DISTANCE_LADDER["nvswitch"] == 1
        assert TOPOLOGY_DISTANCE_LADDER["pcie_root"] == 2
        assert TOPOLOGY_DISTANCE_LADDER["rack"] == 4
        assert TOPOLOGY_DISTANCE_LADDER["cross_rack"] == 5
        assert TOPOLOGY_DISTANCE_LADDER["cross_region"] == 6

    def test_bandwidth_decreases_with_distance(self):
        nvsw = FABRIC_REGIMES["nvswitch"].b_eff_gbps
        pcie = FABRIC_REGIMES["pcie_root"].b_eff_gbps
        xrack = FABRIC_REGIMES["cross_rack"].b_eff_gbps
        xregion = FABRIC_REGIMES["cross_region"].b_eff_gbps
        assert nvsw > pcie > xrack > xregion

    def test_latency_grows_off_node(self):
        # Cross-region latency is orders of magnitude above NVSwitch.
        assert (
            FABRIC_REGIMES["cross_region"].latency_us
            > 1000 * FABRIC_REGIMES["nvswitch"].latency_us
        )

    def test_nvlink_generations_distinct(self):
        a100 = NVLINK_GENERATIONS["a100"].bidir_gbps
        h100 = NVLINK_GENERATIONS["h100"].bidir_gbps
        fut = NVLINK_GENERATIONS["future"].bidir_gbps
        assert a100 < h100 < fut  # NVLink is NOT one bandwidth


# ---------------------------------------------------------------------------
# Latency-bandwidth message cost + small-message amplification
# ---------------------------------------------------------------------------

class TestMessageCost:
    def test_small_message_latency_dominates(self):
        r = FABRIC_REGIMES["cross_rack"]
        tiny = topo.message_time_ms(1024.0, r)
        big = topo.message_time_ms(64 * 1024 * 1024, r)
        # A tiny message is latency-bound (cheap in bytes but pays alpha);
        # a big message is bandwidth-bound (much larger total time).
        assert big > tiny

    def test_small_message_amplification_ramps(self):
        amp_big = topo.small_message_amplification(10 * 1024 * 1024)
        amp_small = topo.small_message_amplification(1024.0)
        assert amp_big == 1.0
        assert amp_small > 1.0

    def test_higher_bandwidth_fabric_is_faster(self):
        m = 16 * 1024 * 1024
        t_nvsw = topo.message_time_ms(m, FABRIC_REGIMES["nvswitch"])
        t_pcie = topo.message_time_ms(m, FABRIC_REGIMES["pcie_root"])
        assert t_nvsw < t_pcie

    def test_congestion_degrades_bandwidth(self):
        r = FABRIC_REGIMES["pcie_root"]
        bw_idle = topo.effective_bandwidth(r, 0.1)
        bw_busy = topo.effective_bandwidth(r, 0.95)
        assert bw_busy < bw_idle
        # never below the floor
        floor = topology_value("congestion_bw_floor")
        assert bw_busy >= r.b_eff_gbps * floor - 1e-6


# ---------------------------------------------------------------------------
# Collectives (ring / tree) + amplification
# ---------------------------------------------------------------------------

class TestCollectives:
    def test_ring_allreduce_grows_with_ranks(self):
        r = FABRIC_REGIMES["nvswitch"]
        m = 8 * 1024 * 1024
        t2 = topo.ring_allreduce_ms(m, 2, r)
        t16 = topo.ring_allreduce_ms(m, 16, r)
        assert t16 > t2 > 0

    def test_tree_latency_logarithmic(self):
        r = FABRIC_REGIMES["rack"]
        m = 1024.0  # tiny → latency-bound
        t8 = topo.tree_collective_ms(m, 8, r)
        t64 = topo.tree_collective_ms(m, 64, r)
        # 8x ranks adds only ~2x the log-latency, not 8x.
        assert t64 < 4 * t8

    def test_allreduce_amplifies_p2p(self):
        r = FABRIC_REGIMES["cross_rack"]
        amp = topo.collective_amplification("all_reduce", 8 * 1024 * 1024, 16, r)
        assert amp > 1.0

    def test_moe_hotspot_amplifies_under_congestion(self):
        a_idle = topo.moe_hotspot_amplification(0.0)
        a_busy = topo.moe_hotspot_amplification(1.0)
        assert a_idle == 1.0
        assert a_busy > 1.0

    def test_collective_jitter_is_seeded(self):
        r = FABRIC_REGIMES["rack"]
        a = topo.collective_latency_ms("all_reduce", 1 << 20, 8, r, 0.3, _rng(1))
        b = topo.collective_latency_ms("all_reduce", 1 << 20, 8, r, 0.3, _rng(1))
        assert a == b  # deterministic under fixed seed


# ---------------------------------------------------------------------------
# Placement quality score (distance ladder → 0-1)
# ---------------------------------------------------------------------------

class TestPlacementQuality:
    def test_nvswitch_best_cross_region_worst(self):
        _, q_nvsw = topo.placement_quality_score(["nvswitch"])
        _, q_rack = topo.placement_quality_score(["cross_rack"])
        _, q_region = topo.placement_quality_score(["cross_region"])
        assert q_nvsw > q_rack > q_region
        assert q_nvsw >= 0.99
        assert q_region < 0.1

    def test_worst_hop_dominates(self):
        # One cross-rack hop in an otherwise-local placement drags quality down.
        _, q = topo.placement_quality_score(["nvswitch", "nvswitch", "cross_rack"])
        _, q_all_local = topo.placement_quality_score(["nvswitch"] * 3)
        assert q < q_all_local

    def test_distance_score_traffic_weighted(self):
        dist, _ = topo.placement_quality_score(["nvswitch", "cross_region"])
        # Σ w*d with uniform weights = mean distance (1 and 6) = 3.5
        assert abs(dist - 3.5) < 1e-6


# ---------------------------------------------------------------------------
# Communication throughput penalty + synchronization + tails
# ---------------------------------------------------------------------------

class TestCommEffects:
    def test_tensor_parallel_collapses_off_nvswitch(self):
        tp = WORKLOAD_COMM_PROFILES["tensor_parallel"]
        _, q_nvsw = topo.placement_quality_score(["nvswitch"])
        _, q_xrack = topo.placement_quality_score(["cross_rack"])
        f_good = topo.comm_throughput_factor(tp, q_nvsw)
        f_bad = topo.comm_throughput_factor(tp, q_xrack)
        assert f_good > 0.95          # NVSwitch: barely penalized
        assert f_bad < 0.5            # cross-rack: throughput collapses

    def test_batch_inference_survives_bad_topology(self):
        bi = WORKLOAD_COMM_PROFILES["batch_inference"]
        _, q_xrack = topo.placement_quality_score(["cross_rack"])
        f = topo.comm_throughput_factor(bi, q_xrack)
        # Still suffers, but does NOT collapse like tensor-parallel.
        assert f > 0.75

    def test_sync_penalty_only_for_sync_heavy(self):
        tp = WORKLOAD_COMM_PROFILES["tensor_parallel"]
        bi = WORKLOAD_COMM_PROFILES["batch_inference"]
        _, q = topo.placement_quality_score(["cross_rack"])
        _, slow_tp = topo.synchronization_penalty(tp, q, 0.5, _rng(3))
        straggler_bi, slow_bi = topo.synchronization_penalty(bi, q, 0.5, _rng(3))
        assert slow_tp > 0.0
        assert slow_bi == 0.0 and straggler_bi == 0.0

    def test_comm_tail_p99_grows_faster_than_p95(self):
        tp = WORKLOAD_COMM_PROFILES["tensor_parallel"]
        _, q_good = topo.placement_quality_score(["nvswitch"])
        _, q_mid = topo.placement_quality_score(["pcie_root"])
        p95_g, p99_g = topo.comm_tail_multipliers(tp, q_good, 0.1)
        # Moderate (non-saturating) degradation so both tails are still climbing.
        p95_b, p99_b = topo.comm_tail_multipliers(tp, q_mid, 0.4)
        assert p99_b > p95_b              # p99 above p95
        assert (p99_b - p99_g) > (p95_b - p95_g)  # p99 amplifies faster

    def test_nic_saturation_incast(self):
        sat_lo, incast_lo = topo.nic_saturation(0.2, 0.3)
        sat_hi, incast_hi = topo.nic_saturation(1.0, 1.0)
        assert sat_hi > sat_lo
        assert incast_hi and not incast_lo


# ---------------------------------------------------------------------------
# Telemetry confidence + topology-aware migration veto
# ---------------------------------------------------------------------------

class TestTelemetryAndMigration:
    def test_missing_topology_lowers_confidence(self):
        assert topo.topology_telemetry_confidence(True, True, True, 0) == "high"
        assert topo.topology_telemetry_confidence(True, False, True, 0) == "medium"
        assert topo.topology_telemetry_confidence(False, False, True, 5) == "low"

    def test_low_telemetry_discounts_score(self):
        # An optimistic NVSwitch reading is NOT trusted under low telemetry.
        full = topo.telemetry_discounted_score(1.0, "high")
        low = topo.telemetry_discounted_score(1.0, "low")
        assert full == 1.0
        assert low < 1.0

    def test_migration_veto_pins_comm_heavy_jobs(self):
        tp = WORKLOAD_COMM_PROFILES["tensor_parallel"]
        bi = WORKLOAD_COMM_PROFILES["batch_inference"]
        # cross-region (distance 6) move
        assert topo.topology_migration_blocked(tp, 6, 0.05, "high") is True
        assert topo.topology_migration_blocked(bi, 6, 0.05, "high") is False

    def test_low_telemetry_more_conservative(self):
        # A cross-socket (distance 3) move into a healthy-enough destination is
        # allowed under high telemetry but vetoed under low telemetry (the veto
        # distance threshold drops from 4 to 3 — missing topology != safe).
        ar = WORKLOAD_COMM_PROFILES["all_reduce_training"]
        blocked_high = topo.topology_migration_blocked(ar, 3, 0.6, "high")
        blocked_low = topo.topology_migration_blocked(ar, 3, 0.6, "low")
        assert blocked_low and not blocked_high


# ---------------------------------------------------------------------------
# Integration scenarios (emergent behaviour)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_tensor_parallel_cross_rack_collapses(self):
        _, ms = _run("tensor_parallel_topology_collapse", steps=20)
        m = ms[-1]
        assert m.mean_topology_quality is not None and m.mean_topology_quality < 0.4
        assert m.comm_throughput_penalty_pct_mean > 30.0
        assert m.topology_risk_max > 0.4
        assert m.collective_instability_count >= 1
        assert m.cross_rack_workload_count >= 1

    def test_nvswitch_colocation_beats_cross_rack(self):
        import copy
        base = load_scenario("tensor_parallel_topology_collapse").config
        cross = copy.deepcopy(base)
        local = copy.deepcopy(base)
        local.regions[0]["nodes"] = [{
            "node_id": "tp-node0", "gpu_type": "a100-sxm4-80gb", "gpu_count": 4,
            "topology_class": "nvswitch", "rack_id": "us-east-rack0",
            "zone": "us-east-1a",
        }]
        m_cross = ClusterSimulator(cross, seed=42).run_metrics_only(16)[-1]
        m_local = ClusterSimulator(local, seed=42).run_metrics_only(16)[-1]
        # NVSwitch co-location wins on every topology axis.
        assert m_local.mean_topology_quality > m_cross.mean_topology_quality
        assert (
            m_local.comm_throughput_penalty_pct_mean
            < m_cross.comm_throughput_penalty_pct_mean
        )

    def test_moe_hotspot_congests_pcie_fabric(self):
        _, ms = _run("moe_hotspot_nic_saturation", steps=20)
        m = ms[-1]
        assert m.fabric_congestion_max is not None and m.fabric_congestion_max > 0.5
        assert m.collective_amplification_max > 1.5
        assert m.comm_throughput_penalty_pct_mean > 30.0

    def test_degraded_telemetry_discounts_and_vetoes(self):
        sim, ms = _run("degraded_topology_telemetry", steps=12)
        m = ms[-1]
        assert m.low_topology_telemetry_count >= 1
        # NVSwitch placement but discounted by low telemetry → quality < 1.
        assert m.mean_topology_quality is not None and m.mean_topology_quality < 1.0
        # Cross-region migration of the comm-heavy training job is vetoed even
        # though the destination is cheaper energy.
        ok = sim.safe_migrate_workload("blind-train-wl", "us-west")
        assert ok is False
        wl = sim._cluster.workloads["blind-train-wl"]
        assert wl.migration.migration.last_veto_reason == "topology_cross_domain"

    def test_topology_migration_veto_counted(self):
        sim, _ = _run("degraded_topology_telemetry", steps=6)
        before = sim._cluster.workloads["blind-train-wl"].topology.migration_risk.veto_count
        sim.safe_migrate_workload("blind-train-wl", "us-west")
        after = sim._cluster.workloads["blind-train-wl"].topology.migration_risk.veto_count
        assert after == before + 1

    def test_determinism_under_fixed_seed(self):
        _, ms1 = _run("moe_hotspot_nic_saturation", steps=12, seed=7)
        _, ms2 = _run("moe_hotspot_nic_saturation", steps=12, seed=7)
        assert (
            ms1[-1].comm_throughput_penalty_pct_mean
            == ms2[-1].comm_throughput_penalty_pct_mean
        )
        assert ms1[-1].fabric_congestion_max == ms2[-1].fabric_congestion_max


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden constants
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_all_topology_params_have_provenance(self):
        for name, p in TOPOLOGY_PARAMS.items():
            assert p.source, name
            assert p.source_type in (
                "measured", "benchmark_derived", "documented", "inferred", "heuristic"
            ), name
            assert p.confidence in ("high", "medium", "low"), name
            assert p.calibration_notes, name

    def test_topology_params_in_calibration_table(self):
        rows = calibration_table()
        groups = {r["group"] for r in rows}
        assert "topology" in groups
        topo_rows = [r for r in rows if r["group"] == "topology"]
        assert len(topo_rows) == len(TOPOLOGY_PARAMS)

    def test_inspection_tables_populated(self):
        assert len(fabric_regime_table()) == len(FABRIC_REGIMES)
        assert len(nvlink_generation_table()) == len(NVLINK_GENERATIONS)
        assert len(workload_comm_profile_table()) == len(WORKLOAD_COMM_PROFILES)

    def test_config_override_changes_value(self):
        assert topology_value("sync_penalty_max") != 0.123
        assert topology_value("sync_penalty_max", {"sync_penalty_max": 0.123}) == 0.123

    def test_comm_profile_resolution(self):
        # Explicit name wins; otherwise inferred from type / intensity.
        assert resolve_comm_profile("moe_expert").name == "moe_expert"
        assert resolve_comm_profile(None, "high", "inference").name == "tensor_parallel"
        assert resolve_comm_profile(None, "low", "batch_training").name == (
            "all_reduce_training"
        )
        assert resolve_comm_profile(None, "low", "inference").name == (
            "comm_light_inference"
        )

    def test_resolve_fabric_regime_default(self):
        assert resolve_fabric_regime(None).name == "nvswitch"
        assert resolve_fabric_regime("cross_rack").name == "cross_rack"
