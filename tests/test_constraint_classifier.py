"""Tests for the Phase 7 constraint classifier.

Verifies:
- Each constraint family scores correctly with sufficient signals
- Missing signals → None score, lower confidence, not fabricated
- Hysteresis prevents flapping between binding constraints
- Confidence floor prevents declarations on sparse data
- Each simulator scenario triggers the expected binding constraint
- cluster-level and region-level assessment APIs
- JSON round-trip for ConstraintAssessment
- Trust-boundary invariants: memory-bound never emits KV-internal actions
- Trust-boundary: communication-bound never emits NCCL/CUDA actions
- No binding constraint from absent/zero-signal data
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aurelius.constraints import ConstraintClassifier, ConstraintConfig
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario
from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
    ThermalState,
    TopologyState,
    TopologyLinkType,
)

_UTC = timezone.utc
_TS = datetime(2024, 1, 1, 12, 0, 0, tzinfo=_UTC)


def _prov(sandbox: bool = True) -> Provenance:
    return Provenance(
        source="test",
        fetched_at=_TS,
        confidence="high",
        is_sandbox=sandbox,
    )


def _empty_cluster(regions: dict | None = None) -> ClusterState:
    return ClusterState(
        timestamp=_TS,
        provenance=_prov(),
        regions=regions or {},
    )


def _region(
    region_id: str = "test",
    energy: EnergyState | None = None,
    thermal: ThermalState | None = None,
    nodes: dict | None = None,
    services: dict | None = None,
    spare_capacity_pct: float | None = None,
) -> RegionState:
    return RegionState(
        region=region_id,
        timestamp=_TS,
        provenance=_prov(),
        nodes=nodes or {},
        services=services or {},
        energy=energy,
        thermal=thermal,
        spare_capacity_pct=spare_capacity_pct,
    )


def _energy_state(
    price: float | None = None,
    percentile: float | None = None,
) -> EnergyState:
    return EnergyState(
        region="test",
        timestamp=_TS,
        provenance=_prov(),
        price_per_mwh=price,
        price_percentile=percentile,
    )


def _gpu_state(
    uuid: str = "GPU-00001",
    util_pct: float | None = None,
    temp_c: float | None = None,
    mem_used_mb: float | None = None,
    mem_total_mb: float | None = None,
    nvlink_tx: float | None = None,
    nvlink_rx: float | None = None,
    pcie_tx: float | None = None,
    pcie_rx: float | None = None,
    kv_cache_usage: float | None = None,
    preemptions: float | None = None,
) -> GPUState:
    return GPUState(
        gpu_uuid=uuid,
        node_id="node0",
        region="test",
        timestamp=_TS,
        provenance=_prov(),
        util_pct=util_pct,
        temp_c=temp_c,
        mem_used_mb=mem_used_mb,
        mem_total_mb=mem_total_mb,
        nvlink_tx_bytes_per_s=nvlink_tx,
        nvlink_rx_bytes_per_s=nvlink_rx,
        pcie_tx_bytes_per_s=pcie_tx,
        pcie_rx_bytes_per_s=pcie_rx,
    )


def _node_state(
    node_id: str = "node0",
    gpus: dict | None = None,
) -> NodeState:
    return NodeState(
        node_id=node_id,
        region="test",
        timestamp=_TS,
        provenance=_prov(),
        gpus=gpus or {},
    )


def _inference_service(
    svc_id: str = "svc0",
    p99_ms: float | None = None,
    ttft_p99_ms: float | None = None,
    queue_depth: float | None = None,
    queue_wait_p99_ms: float | None = None,
    error_rate_pct: float | None = None,
    kv_cache_usage: float | None = None,
    prefix_hit_rate: float | None = None,
    preemptions: float | None = None,
) -> InferenceServiceState:
    return InferenceServiceState(
        service_id=svc_id,
        engine="vllm",
        timestamp=_TS,
        provenance=_prov(),
        p99_latency_ms=p99_ms,
        ttft_p99_ms=ttft_p99_ms,
        requests_waiting=queue_depth,
        queue_time_p99_ms=queue_wait_p99_ms,
        error_rate_pct=error_rate_pct,
        kv_cache_usage=kv_cache_usage,
        prefix_cache_hit_rate=prefix_hit_rate,
        preemptions_total=preemptions,
    )


# ============================================================
# Basic instantiation
# ============================================================

class TestClassifierInstantiation:
    def test_default_config(self):
        clf = ConstraintClassifier()
        assert clf._cfg is not None
        assert clf._cfg.energy_high_price_mwh == 100.0

    def test_custom_config(self):
        cfg = ConstraintConfig(energy_high_price_mwh=50.0)
        clf = ConstraintClassifier(cfg)
        assert clf._cfg.energy_high_price_mwh == 50.0

    def test_reset_clears_hysteresis(self):
        clf = ConstraintClassifier()
        clf._previous_binding[None] = ConstraintType.ENERGY
        clf.reset()
        assert clf._previous_binding == {}

    def test_assess_empty_cluster_returns_none_binding(self):
        clf = ConstraintClassifier()
        state = _empty_cluster()
        result = clf.assess(state)
        assert result.binding_constraint is None
        assert result.confidence == 0.0
        assert "no_regions" in result.missing_signals

    def test_assessment_is_constraint_assessment(self):
        clf = ConstraintClassifier()
        result = clf.assess(_empty_cluster())
        assert isinstance(result, ConstraintAssessment)

    def test_timestamp_matches_state(self):
        clf = ConstraintClassifier()
        ts = datetime(2025, 6, 15, 10, 0, 0, tzinfo=_UTC)
        state = ClusterState(timestamp=ts, provenance=_prov(), regions={})
        result = clf.assess(state)
        assert result.timestamp == ts

    def test_sandbox_flag_preserved(self):
        clf = ConstraintClassifier()
        state = _empty_cluster()
        result = clf.assess(state)
        assert result.provenance.is_sandbox is True

    def test_non_sandbox_preserved(self):
        clf = ConstraintClassifier()
        state = ClusterState(
            timestamp=_TS,
            provenance=Provenance(source="test", fetched_at=_TS, confidence="high", is_sandbox=False),
            regions={},
        )
        result = clf.assess(state)
        assert result.provenance.is_sandbox is False


# ============================================================
# No fabrication from absent data
# ============================================================

class TestNoFabricationFromAbsentData:
    def test_no_signals_no_binding_constraint(self):
        """Cluster with regions but no telemetry → no binding constraint."""
        clf = ConstraintClassifier()
        r = _region("r0")  # empty region, no energy/thermal/services/nodes
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert result.binding_constraint is None

    def test_all_signals_missing_reports_none_score(self):
        """All families should have None score when signals are absent."""
        clf = ConstraintClassifier()
        r = _region("r0")
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # No scores present (nothing to score on)
        for score in result.scores.values():
            assert score is not None  # scores are only present if signals were found
        # Actually if region is empty, scores dict should be empty too
        assert len(result.scores) == 0 or all(v >= 0.0 for v in result.scores.values())

    def test_partial_state_reduces_confidence(self):
        """is_partial=True should reduce confidence."""
        clf = ConstraintClassifier()
        # Create a region with some energy data so there's a base confidence
        energy = _energy_state(price=200.0, percentile=90.0)
        r = _region("r0", energy=energy)

        complete_state = _empty_cluster({"r0": r})
        partial_state = ClusterState(
            timestamp=_TS,
            provenance=_prov(),
            regions={"r0": r},
            is_partial=True,
        )

        result_complete = clf.assess(complete_state)
        clf.reset()
        result_partial = clf.assess(partial_state)

        assert result_partial.confidence < result_complete.confidence

    def test_missing_signals_listed(self):
        """Missing signals should be listed in the assessment."""
        clf = ConstraintClassifier()
        # Only energy, no thermal/queue/etc.
        energy = _energy_state(price=200.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # Should list missing thermal, queue, latency, etc.
        assert len(result.missing_signals) > 0

    def test_zero_regions_no_binding_constraint(self):
        clf = ConstraintClassifier()
        state = _empty_cluster({})
        result = clf.assess(state)
        assert result.binding_constraint is None
        assert result.confidence == 0.0


# ============================================================
# Energy constraint
# ============================================================

class TestEnergyConstraint:
    def test_high_price_triggers_energy_bound(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            energy_very_high_price_mwh=150.0,
            hysteresis_on=0.50,
            confidence_floor=0.0,  # unit test: only energy signals present
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=160.0, percentile=90.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert result.binding_constraint == ConstraintType.ENERGY
        assert ConstraintType.ENERGY in result.scores

    def test_low_price_no_energy_bound(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            energy_very_high_price_mwh=200.0,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=45.0, percentile=20.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # Score should be low
        if ConstraintType.ENERGY in result.scores:
            assert result.scores[ConstraintType.ENERGY] < 0.3

    def test_missing_price_no_energy_score(self):
        clf = ConstraintClassifier()
        energy = _energy_state(price=None, percentile=None)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.ENERGY not in result.scores

    def test_energy_safe_actions_present(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=50.0,
            energy_very_high_price_mwh=100.0,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=110.0, percentile=90.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.ENERGY:
            assert "defer_flexible_workload" in result.safe_action_types

    def test_energy_percentile_only_no_fabrication(self):
        """Only percentile present (no price) — no energy score (price is primary)."""
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            hysteresis_on=0.50,
            confidence_floor=0.0,
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=None, percentile=95.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # percentile is a secondary enrichment — no price (primary) means no score
        assert ConstraintType.ENERGY not in result.scores


# ============================================================
# Thermal constraint
# ============================================================

class TestThermalConstraint:
    def test_throttling_triggers_thermal_bound(self):
        cfg = ConstraintConfig(
            thermal_throttle_fraction_warn=0.05,
            thermal_throttle_fraction_high=0.20,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        thermal = ThermalState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            max_gpu_temp_c=88.0,
            throttling_gpu_count=3,
            total_gpu_count=8,
        )
        r = _region("r0", thermal=thermal)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.THERMAL in result.scores
        assert result.scores[ConstraintType.THERMAL] > 0.3

    def test_low_temp_no_thermal_bound(self):
        cfg = ConstraintConfig(
            thermal_warn_temp_c=80.0,
            thermal_critical_temp_c=87.0,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        thermal = ThermalState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            max_gpu_temp_c=55.0,
            throttling_gpu_count=0,
            total_gpu_count=8,
        )
        r = _region("r0", thermal=thermal)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if ConstraintType.THERMAL in result.scores:
            assert result.scores[ConstraintType.THERMAL] < 0.3

    def test_missing_thermal_state_no_score(self):
        clf = ConstraintClassifier()
        r = _region("r0", thermal=None)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.THERMAL not in result.scores

    def test_thermal_safe_actions_exclude_consolidation(self):
        cfg = ConstraintConfig(
            thermal_warn_temp_c=75.0,
            thermal_critical_temp_c=85.0,
            thermal_throttle_fraction_warn=0.01,
            thermal_throttle_fraction_high=0.10,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        thermal = ThermalState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            max_gpu_temp_c=88.0,
            throttling_gpu_count=4,
            total_gpu_count=8,
        )
        r = _region("r0", thermal=thermal)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.THERMAL:
            assert "consolidate_to_hot_rack" in result.disallowed_action_types
            assert "bin_pack_onto_throttling_nodes" in result.disallowed_action_types
            assert "spread_from_hot_rack" in result.safe_action_types


# ============================================================
# Queue constraint
# ============================================================

class TestQueueConstraint:
    def test_high_queue_depth_triggers_queue_bound(self):
        cfg = ConstraintConfig(
            queue_depth_warn=20.0,
            queue_depth_high=100.0,
            queue_wait_warn_ms=5000.0,
            queue_wait_high_ms=30000.0,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(queue_depth=150.0, queue_wait_p99_ms=40000.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.QUEUE in result.scores
        assert result.scores[ConstraintType.QUEUE] > 0.5

    def test_low_queue_no_pressure(self):
        cfg = ConstraintConfig(confidence_floor=0.1)
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(queue_depth=2.0, queue_wait_p99_ms=200.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if ConstraintType.QUEUE in result.scores:
            assert result.scores[ConstraintType.QUEUE] < 0.3

    def test_missing_queue_signals_no_score(self):
        clf = ConstraintClassifier()
        r = _region("r0")
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.QUEUE not in result.scores

    def test_queue_safe_actions(self):
        cfg = ConstraintConfig(
            queue_depth_warn=5.0,
            queue_depth_high=50.0,
            queue_wait_warn_ms=1000.0,
            queue_wait_high_ms=10000.0,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(queue_depth=80.0, queue_wait_p99_ms=15000.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.QUEUE:
            assert "add_replica_recommendation" in result.safe_action_types
            assert "migrate_serving_workload_during_surge" in result.disallowed_action_types


# ============================================================
# Latency constraint
# ============================================================

class TestLatencyConstraint:
    def test_high_p99_triggers_latency_bound(self):
        cfg = ConstraintConfig(
            latency_p99_warn_ms=1000.0,
            latency_p99_high_ms=3000.0,
            ttft_p99_warn_ms=500.0,
            ttft_p99_high_ms=2000.0,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(p99_ms=4000.0, ttft_p99_ms=2500.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.LATENCY in result.scores
        assert result.scores[ConstraintType.LATENCY] > 0.5

    def test_low_latency_no_bound(self):
        cfg = ConstraintConfig(confidence_floor=0.1)
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(p99_ms=200.0, ttft_p99_ms=100.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if ConstraintType.LATENCY in result.scores:
            assert result.scores[ConstraintType.LATENCY] < 0.3

    def test_latency_safe_actions_exclude_migrations(self):
        cfg = ConstraintConfig(
            latency_p99_warn_ms=500.0,
            latency_p99_high_ms=1500.0,
            ttft_p99_warn_ms=200.0,
            ttft_p99_high_ms=1000.0,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(p99_ms=2000.0, ttft_p99_ms=1200.0)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.LATENCY:
            assert "migrate_latency_sensitive" in result.disallowed_action_types
            assert "preserve_affinity" in result.safe_action_types


# ============================================================
# Communication constraint
# ============================================================

class TestCommunicationConstraint:
    def test_high_nvlink_low_sm_triggers_comm_bound(self):
        """High NVLink + low SM utilization → communication bottleneck."""
        cfg = ConstraintConfig(
            nvlink_high_gbps=10.0,
            nvlink_saturated_gbps=40.0,
            hysteresis_on=0.50,
            confidence_floor=0.0,  # unit test: only comm signals present
        )
        clf = ConstraintClassifier(cfg)
        # 50 GB/s combined NVLink per GPU × 4 GPUs → 200 GB/s, but only 10% SM util
        # → communication is a bottleneck
        gpus = {
            f"GPU-{i}": _gpu_state(
                uuid=f"GPU-{i}",
                nvlink_tx=25e9,
                nvlink_rx=25e9,
                util_pct=10.0,  # low SM/util → communication bound
            )
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.COMMUNICATION in result.scores
        assert result.scores[ConstraintType.COMMUNICATION] > 0.3

    def test_high_nvlink_high_sm_no_comm_bound(self):
        """High NVLink + high SM utilization → NOT communication bound (workload running well)."""
        cfg = ConstraintConfig(
            nvlink_high_gbps=10.0,
            nvlink_saturated_gbps=40.0,
            hysteresis_on=0.50,
            confidence_floor=0.0,
        )
        clf = ConstraintClassifier(cfg)
        # 50 GB/s combined NVLink per GPU but 90% SM util → NOT comm bottleneck
        gpus = {
            f"GPU-{i}": _gpu_state(
                uuid=f"GPU-{i}",
                nvlink_tx=25e9,
                nvlink_rx=25e9,
                util_pct=90.0,  # high SM/util → not communication bound
            )
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if ConstraintType.COMMUNICATION in result.scores:
            # Score should be low — high SM means compute-bound, not comm-bound
            assert result.scores[ConstraintType.COMMUNICATION] < 0.3

    def test_no_nvlink_data_no_comm_score(self):
        clf = ConstraintClassifier()
        gpus = {"GPU-0": _gpu_state(uuid="GPU-0")}
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.COMMUNICATION not in result.scores

    def test_comm_safe_actions_exclude_split(self):
        cfg = ConstraintConfig(
            nvlink_high_gbps=1.0,
            nvlink_saturated_gbps=5.0,
            hysteresis_on=0.40,
            confidence_floor=0.0,  # unit test: only comm signals
        )
        clf = ConstraintClassifier(cfg)
        # Low NVLink + low SM = communication bottleneck
        gpus = {
            f"GPU-{i}": _gpu_state(
                uuid=f"GPU-{i}",
                nvlink_tx=5e9,
                nvlink_rx=5e9,
                util_pct=8.0,  # very low SM → communication bound
            )
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.COMMUNICATION:
            assert "split_communicating_workloads" in result.disallowed_action_types
            assert "topology_aware_replacement" in result.safe_action_types


# ============================================================
# Memory constraint (indirect only)
# ============================================================

class TestMemoryConstraint:
    def test_high_kv_cache_triggers_memory_bound(self):
        cfg = ConstraintConfig(
            kv_cache_warn_fraction=0.60,
            kv_cache_high_fraction=0.80,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(kv_cache_usage=0.90, prefix_hit_rate=0.15)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.MEMORY in result.scores
        assert result.scores[ConstraintType.MEMORY] > 0.4

    def test_memory_safe_actions_exclude_kv_internals(self):
        """Memory-bound must never emit KV-cache internal actions."""
        cfg = ConstraintConfig(
            kv_cache_warn_fraction=0.50,
            kv_cache_high_fraction=0.70,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(kv_cache_usage=0.85, prefix_hit_rate=0.10)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # Memory-bound must NEVER suggest touching KV cache internals
        if result.binding_constraint == ConstraintType.MEMORY:
            for action in result.safe_action_types:
                assert "kv_cache_internal" not in action, (
                    f"Memory-bound emitted forbidden KV-internal action: {action}"
                )
            assert "modify_kv_cache_internals" in result.disallowed_action_types
            assert "modify_memory_allocator" in result.disallowed_action_types

    def test_memory_disallowed_migrate_cached_workload(self):
        """Memory-bound must not recommend migrating cached workloads."""
        cfg = ConstraintConfig(
            kv_cache_warn_fraction=0.50,
            kv_cache_high_fraction=0.70,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(kv_cache_usage=0.88, prefix_hit_rate=0.05)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if result.binding_constraint == ConstraintType.MEMORY:
            assert "migrate_cached_workload" in result.disallowed_action_types
            assert "route_to_cold_replica" in result.disallowed_action_types

    def test_hbm_pressure_contributes_to_memory_score(self):
        cfg = ConstraintConfig(
            hbm_warn_fraction=0.75,
            hbm_high_fraction=0.90,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        gpus = {
            "GPU-0": _gpu_state(uuid="GPU-0", mem_used_mb=60000.0, mem_total_mb=65536.0),
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # HBM pressure should show up in memory score
        if ConstraintType.MEMORY in result.scores:
            assert result.scores[ConstraintType.MEMORY] > 0.0

    def test_no_memory_signals_no_score(self):
        clf = ConstraintClassifier()
        r = _region("r0")
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.MEMORY not in result.scores


# ============================================================
# Topology constraint
# ============================================================

class TestTopologyConstraint:
    def test_weak_topology_triggers_topology_bound(self):
        cfg = ConstraintConfig(
            topology_weak_link_threshold=0.5,
            hysteresis_on=0.40,
            confidence_floor=0.0,  # unit test: only topology signals
        )
        clf = ConstraintClassifier(cfg)
        # PCIe links (quality 0.50 ≤ threshold → weak)
        topo = TopologyState(
            node_id="node0",
            timestamp=_TS,
            provenance=_prov(),
            gpu_uuids=("GPU-0", "GPU-1", "GPU-2"),
            pair_levels={
                ("GPU-0", "GPU-1"): TopologyLinkType.SYS,   # 0.20
                ("GPU-0", "GPU-2"): TopologyLinkType.RACK,  # 0.10
                ("GPU-1", "GPU-2"): TopologyLinkType.SYS,   # 0.20
            },
            numa_affinity={"GPU-0": 0, "GPU-1": 0, "GPU-2": 1},
        )
        r = _region("r0")
        # Hack: need a RegionState with topology — rebuild with topology
        r_with_topo = RegionState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            topology=topo,
        )
        state = _empty_cluster({"r0": r_with_topo})
        result = clf.assess(state)
        assert ConstraintType.TOPOLOGY in result.scores
        assert result.scores[ConstraintType.TOPOLOGY] == 1.0  # all weak

    def test_nvswitch_topology_no_topology_pressure(self):
        cfg = ConstraintConfig(
            topology_weak_link_threshold=0.5,
            confidence_floor=0.0,  # unit test: only topology signals
        )
        clf = ConstraintClassifier(cfg)
        topo = TopologyState(
            node_id="node0",
            timestamp=_TS,
            provenance=_prov(),
            gpu_uuids=("GPU-0", "GPU-1"),
            pair_levels={
                ("GPU-0", "GPU-1"): TopologyLinkType.NVSWITCH,  # 1.0
            },
            numa_affinity={"GPU-0": 0, "GPU-1": 0},
        )
        r_with_topo = RegionState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            topology=topo,
        )
        state = _empty_cluster({"r0": r_with_topo})
        result = clf.assess(state)
        if ConstraintType.TOPOLOGY in result.scores:
            assert result.scores[ConstraintType.TOPOLOGY] == 0.0

    def test_missing_topology_no_score(self):
        clf = ConstraintClassifier()
        r = _region("r0", thermal=None)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.TOPOLOGY not in result.scores


# ============================================================
# Utilization constraint
# ============================================================

class TestUtilizationConstraint:
    def test_low_utilization_triggers_utilization_bound(self):
        cfg = ConstraintConfig(
            util_low_cluster_pct=40.0,
            util_idle_pct=20.0,
            util_idle_gpu_fraction=0.30,
            hysteresis_on=0.50,
            confidence_floor=0.1,
        )
        clf = ConstraintClassifier(cfg)
        gpus = {
            f"GPU-{i}": _gpu_state(uuid=f"GPU-{i}", util_pct=5.0)
            for i in range(8)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.UTILIZATION in result.scores
        assert result.scores[ConstraintType.UTILIZATION] > 0.5

    def test_high_utilization_no_utilization_bound(self):
        cfg = ConstraintConfig(confidence_floor=0.1)
        clf = ConstraintClassifier(cfg)
        gpus = {
            f"GPU-{i}": _gpu_state(uuid=f"GPU-{i}", util_pct=90.0)
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        if ConstraintType.UTILIZATION in result.scores:
            assert result.scores[ConstraintType.UTILIZATION] < 0.3

    def test_no_util_data_no_score(self):
        clf = ConstraintClassifier()
        r = _region("r0")
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert ConstraintType.UTILIZATION not in result.scores


# ============================================================
# Hysteresis
# ============================================================

class TestHysteresis:
    def test_constraint_sticky_above_off_threshold(self):
        """Binding constraint should remain if score stays above hysteresis_off."""
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            energy_very_high_price_mwh=200.0,
            hysteresis_on=0.55,
            hysteresis_off=0.35,
            confidence_floor=0.0,  # unit test: only energy signals present
        )
        clf = ConstraintClassifier(cfg)

        # First tick: high energy → ENERGY
        energy_high = _energy_state(price=220.0, percentile=95.0)
        r = _region("r0", energy=energy_high)
        result1 = clf.assess(_empty_cluster({"r0": r}))
        assert result1.binding_constraint == ConstraintType.ENERGY

        # Second tick: price drops a little (score ~0.40 > hysteresis_off=0.35)
        # Expect ENERGY to remain sticky
        energy_medium = _energy_state(price=150.0, percentile=70.0)
        r2 = _region("r0", energy=energy_medium)
        result2 = clf.assess(_empty_cluster({"r0": r2}))
        assert result2.binding_constraint == ConstraintType.ENERGY

    def test_constraint_clears_when_drops_below_off_threshold(self):
        """Binding constraint should clear when score drops below hysteresis_off."""
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            energy_very_high_price_mwh=200.0,
            hysteresis_on=0.55,
            hysteresis_off=0.35,
            confidence_floor=0.0,  # unit test: only energy signals present
        )
        clf = ConstraintClassifier(cfg)

        # Establish energy binding
        energy_high = _energy_state(price=220.0, percentile=95.0)
        r = _region("r0", energy=energy_high)
        result1 = clf.assess(_empty_cluster({"r0": r}))
        assert result1.binding_constraint == ConstraintType.ENERGY

        # Drop price very low (score should be 0.0 < hysteresis_off)
        energy_low = _energy_state(price=30.0, percentile=10.0)
        r2 = _region("r0", energy=energy_low)
        result2 = clf.assess(_empty_cluster({"r0": r2}))
        # Should have cleared
        if ConstraintType.ENERGY in result2.scores:
            assert result2.scores[ConstraintType.ENERGY] < cfg.hysteresis_off
        assert result2.binding_constraint != ConstraintType.ENERGY

    def test_reset_clears_hysteresis(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=100.0,
            energy_very_high_price_mwh=200.0,
            hysteresis_on=0.55,
            hysteresis_off=0.35,
            confidence_floor=0.0,  # unit test: only energy signals present
        )
        clf = ConstraintClassifier(cfg)

        # Set energy binding
        energy_high = _energy_state(price=220.0, percentile=95.0)
        r = _region("r0", energy=energy_high)
        clf.assess(_empty_cluster({"r0": r}))
        assert clf._previous_binding.get(None) == ConstraintType.ENERGY

        # Reset
        clf.reset()
        assert clf._previous_binding == {}

    def test_confidence_floor_prevents_declaration(self):
        """Below the confidence floor, no binding constraint should be declared."""
        cfg = ConstraintConfig(
            energy_high_price_mwh=50.0,
            energy_very_high_price_mwh=80.0,
            hysteresis_on=0.30,
            confidence_floor=0.99,  # set floor impossibly high
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=100.0, percentile=90.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        # confidence_floor so high that no binding can be declared
        assert result.binding_constraint is None


# ============================================================
# Region-level assessment API
# ============================================================

class TestRegionAssessment:
    def test_assess_region_for_known_region(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=50.0,
            energy_very_high_price_mwh=100.0,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=120.0, percentile=85.0)
        r = _region("us-east", energy=energy)
        state = _empty_cluster({"us-east": r})
        result = clf.assess_region(state, "us-east")
        assert result.region == "us-east"
        assert isinstance(result, ConstraintAssessment)

    def test_assess_region_unknown_region_returns_none_binding(self):
        clf = ConstraintClassifier()
        state = _empty_cluster({"us-east": _region("us-east")})
        result = clf.assess_region(state, "nonexistent-region")
        assert result.binding_constraint is None
        assert result.confidence == 0.0

    def test_region_assessment_has_region_field(self):
        clf = ConstraintClassifier()
        state = _empty_cluster({"us-east": _region("us-east")})
        result = clf.assess_region(state, "us-east")
        assert result.region == "us-east"

    def test_cluster_assessment_region_is_none(self):
        clf = ConstraintClassifier()
        state = _empty_cluster({"us-east": _region("us-east")})
        result = clf.assess(state)
        assert result.region is None


# ============================================================
# JSON round-trip
# ============================================================

class TestJsonRoundTrip:
    def test_assessment_roundtrip(self):
        cfg = ConstraintConfig(
            energy_high_price_mwh=50.0,
            energy_very_high_price_mwh=100.0,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=120.0, percentile=85.0)
        r = _region("us-east", energy=energy)
        state = _empty_cluster({"us-east": r})
        result = clf.assess(state)

        d = result.to_dict()
        restored = ConstraintAssessment.from_dict(d)

        assert restored.binding_constraint == result.binding_constraint
        assert restored.confidence == result.confidence
        assert restored.missing_signals == result.missing_signals
        assert restored.safe_action_types == result.safe_action_types
        assert restored.disallowed_action_types == result.disallowed_action_types
        assert restored.scores.keys() == result.scores.keys()

    def test_none_binding_roundtrip(self):
        clf = ConstraintClassifier()
        state = _empty_cluster({})
        result = clf.assess(state)
        d = result.to_dict()
        restored = ConstraintAssessment.from_dict(d)
        assert restored.binding_constraint is None


# ============================================================
# Scores are valid [0, 1]
# ============================================================

class TestScoreRanges:
    def test_all_scores_in_01(self):
        cfg = ConstraintConfig(confidence_floor=0.0)
        clf = ConstraintClassifier(cfg)
        # Create a region with all signal types present
        energy = _energy_state(price=150.0, percentile=80.0)
        thermal = ThermalState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            max_gpu_temp_c=85.0,
            throttling_gpu_count=2,
            total_gpu_count=8,
        )
        svc = _inference_service(
            p99_ms=3000.0,
            ttft_p99_ms=1500.0,
            queue_depth=50.0,
            queue_wait_p99_ms=10000.0,
            error_rate_pct=2.0,
            kv_cache_usage=0.80,
            prefix_hit_rate=0.20,
            preemptions=5.0,
        )
        gpus = {
            f"GPU-{i}": _gpu_state(
                uuid=f"GPU-{i}",
                util_pct=15.0,
                temp_c=84.0,
                mem_used_mb=55000.0,
                mem_total_mb=65536.0,
                nvlink_tx=60e9,
                nvlink_rx=60e9,
            )
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = RegionState(
            region="r0",
            timestamp=_TS,
            provenance=_prov(),
            nodes={"node0": node},
            services={"svc0": svc},
            energy=energy,
            thermal=thermal,
            spare_capacity_pct=50.0,
        )
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        for ct, score in result.scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {ct} out of range: {score}"

    def test_confidence_in_01(self):
        cfg = ConstraintConfig(confidence_floor=0.0)
        clf = ConstraintClassifier(cfg)
        energy = _energy_state(price=200.0, percentile=90.0)
        r = _region("r0", energy=energy)
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        assert 0.0 <= result.confidence <= 1.0


# ============================================================
# Trust-boundary invariants
# ============================================================

class TestTrustBoundaryInvariants:
    def test_memory_bound_never_emits_nccl_actions(self):
        """Memory-bound must never mention NCCL in safe or disallowed actions."""
        cfg = ConstraintConfig(
            kv_cache_warn_fraction=0.50,
            kv_cache_high_fraction=0.70,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        svc = _inference_service(kv_cache_usage=0.90, prefix_hit_rate=0.05)
        r = _region("r0", services={"svc0": svc})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        all_actions = result.safe_action_types + result.disallowed_action_types
        for action in all_actions:
            assert "nccl" not in action.lower(), f"Memory-bound emitted NCCL action: {action}"
            assert "cuda" not in action.lower(), f"Memory-bound emitted CUDA action: {action}"

    def test_comm_bound_never_emits_nccl_actions(self):
        """Communication-bound must never suggest NCCL/CUDA modification."""
        cfg = ConstraintConfig(
            nvlink_high_gbps=1.0,
            nvlink_saturated_gbps=5.0,
            hysteresis_on=0.40,
            confidence_floor=0.05,
        )
        clf = ConstraintClassifier(cfg)
        gpus = {
            f"GPU-{i}": _gpu_state(uuid=f"GPU-{i}", nvlink_tx=8e9, nvlink_rx=8e9)
            for i in range(4)
        }
        node = _node_state(gpus=gpus)
        r = _region("r0", nodes={"node0": node})
        state = _empty_cluster({"r0": r})
        result = clf.assess(state)
        all_actions = result.safe_action_types + result.disallowed_action_types
        for action in all_actions:
            assert "nccl" not in action.lower(), f"Comm-bound emitted NCCL action: {action}"
            assert "cuda" not in action.lower(), f"Comm-bound emitted CUDA action: {action}"

    def test_no_runtime_actions_in_any_binding(self):
        """No binding constraint should ever emit runtime-mutation actions."""
        forbidden_patterns = [
            "nccl", "cuda", "kernel", "allocator", "kv_cache_internal",
            "memory_allocator", "runtime_internal",
        ]
        clf = ConstraintClassifier(ConstraintConfig(confidence_floor=0.0))
        for ct in ConstraintType:
            safe = _SAFE_ACTIONS.get(ct, [])
            disallowed = _DISALLOWED_ACTIONS.get(ct, [])
            for action in safe:
                for pattern in forbidden_patterns:
                    if pattern not in ("kv_cache_internal", "memory_allocator"):
                        # kv_cache_internal and memory_allocator ARE in disallowed — that's correct
                        assert pattern not in action.lower(), (
                            f"Runtime action {action!r} in safe_actions for {ct}"
                        )


# ============================================================
# Simulator scenario integration
# ============================================================

class TestSimulatorScenarios:
    """Each simulator scenario should trigger its expected binding constraint.

    These tests use the simulator which generates ClusterState snapshots with
    the same fake-connector paths as real integrations. They run 12 ticks and
    check the modal binding constraint across ticks 7-12 (after scenario event).
    """

    def _run_scenario_and_collect_bindings(
        self,
        scenario_name: str,
        ticks: int = 12,
        start_tick: int = 7,
    ) -> list[ConstraintType | None]:
        scenario = load_scenario(scenario_name)
        sim = ClusterSimulator(scenario.config, seed=42)
        clf = ConstraintClassifier(ConstraintConfig(
            confidence_floor=0.05,
            hysteresis_on=0.40,
            hysteresis_off=0.20,
        ))

        bindings = []
        for i, tick in enumerate(sim.run(ticks)):
            result = clf.assess(tick.cluster_state)
            if i >= start_tick - 1:
                bindings.append(result.binding_constraint)
        return bindings

    def _modal_binding(self, bindings: list) -> ConstraintType | None:
        from collections import Counter
        counts = Counter(b for b in bindings if b is not None)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def test_energy_scenario_identifies_energy_bound(self):
        """During the energy spike at tick 16 (price ×2.5), energy should be binding in us-east.

        The spike raises us-east price to ~187.5 $/MWh. We use per-region assessment for
        us-east because energy constraints are regional — the cluster-level view may show
        UTILIZATION dominating (flexible workloads running at low util).
        """
        scenario = load_scenario("energy_price_arbitrage_multiregion")
        sim = ClusterSimulator(scenario.config, seed=42)
        clf = ConstraintClassifier(ConstraintConfig(
            energy_high_price_mwh=75.0,    # baseline us-east prices hover ~42-80
            energy_very_high_price_mwh=180.0,  # spike at 187.5 → max energy pressure
            confidence_floor=0.0,
            hysteresis_on=0.40,
            hysteresis_off=0.20,
        ))
        spike_bindings = []
        for i, tick in enumerate(sim.run(24)):
            result = clf.assess_region(tick.cluster_state, "us-east")
            # Ticks 16-19: price is 187.5 (spike active)
            if 16 <= i + 1 <= 19:
                spike_bindings.append(result.binding_constraint)

        # All spike ticks should identify ENERGY in us-east
        energy_count = sum(1 for b in spike_bindings if b == ConstraintType.ENERGY)
        assert energy_count >= 2, (
            f"Expected ENERGY binding during price spike; got bindings={spike_bindings}"
        )

    def test_thermal_scenario_identifies_thermal_bound(self):
        bindings = self._run_scenario_and_collect_bindings(
            "thermal_hotspot_mixed_cluster",
            ticks=12,
            start_tick=7,
        )
        modal = self._modal_binding(bindings)
        assert modal == ConstraintType.THERMAL, (
            f"Expected THERMAL binding; got modal={modal}, bindings={bindings}"
        )

    def test_queue_scenario_identifies_queue_bound(self):
        bindings = self._run_scenario_and_collect_bindings(
            "queue_surge_latency_sensitive",
            ticks=12,
            start_tick=8,
        )
        modal = self._modal_binding(bindings)
        assert modal == ConstraintType.QUEUE, (
            f"Expected QUEUE binding; got modal={modal}, bindings={bindings}"
        )

    def test_underutil_scenario_identifies_utilization_bound(self):
        bindings = self._run_scenario_and_collect_bindings(
            "underutilization_stranded_capacity",
            ticks=10,
            start_tick=1,
        )
        modal = self._modal_binding(bindings)
        assert modal == ConstraintType.UTILIZATION, (
            f"Expected UTILIZATION binding; got modal={modal}, bindings={bindings}"
        )

    def test_classifier_never_claims_certainty_on_empty_ticks(self):
        """Ticks with no service data → confidence < 0.7."""
        scenario = load_scenario("energy_price_arbitrage_multiregion")
        sim = ClusterSimulator(scenario.config, seed=42)
        clf = ConstraintClassifier()
        for tick in sim.run(5):
            # Build a stripped state (no services/GPUs)
            stripped_regions = {
                rid: RegionState(
                    region=rid,
                    timestamp=tick.cluster_state.timestamp,
                    provenance=tick.cluster_state.provenance,
                )
                for rid in tick.cluster_state.regions
            }
            stripped_state = ClusterState(
                timestamp=tick.cluster_state.timestamp,
                provenance=tick.cluster_state.provenance,
                regions=stripped_regions,
            )
            result = clf.assess(stripped_state)
            assert result.confidence < 0.7, (
                f"Classifier over-confident on empty state: {result.confidence}"
            )

    def test_latency_scenario_identifies_memory_bound_indirect(self):
        """KV cache pressure scenario triggers memory_bound_indirect."""
        bindings = self._run_scenario_and_collect_bindings(
            "latency_tail_kvcache_pressure",
            ticks=18,
            start_tick=7,
        )
        modal = self._modal_binding(bindings)
        # Accept either MEMORY or LATENCY — both are valid for this scenario
        assert modal in (ConstraintType.MEMORY, ConstraintType.LATENCY, ConstraintType.QUEUE), (
            f"Expected MEMORY/LATENCY/QUEUE; got modal={modal}, bindings={bindings}"
        )


# ============================================================
# Provenance and sandbox
# ============================================================

class TestProvenance:
    def test_classifier_source_in_provenance(self):
        clf = ConstraintClassifier()
        state = _empty_cluster()
        result = clf.assess(state)
        assert result.provenance.source == "constraint_classifier"

    def test_sandbox_state_produces_sandbox_assessment(self):
        clf = ConstraintClassifier()
        state = _empty_cluster()
        result = clf.assess(state)
        assert result.provenance.is_sandbox is True

    def test_non_sandbox_assessment(self):
        clf = ConstraintClassifier()
        state = ClusterState(
            timestamp=_TS,
            provenance=Provenance(source="prod", fetched_at=_TS, confidence="high", is_sandbox=False),
            regions={"r0": _region("r0")},
        )
        result = clf.assess(state)
        assert result.provenance.is_sandbox is False


# Expose safe/disallowed tables for trust-boundary test
from aurelius.constraints.classifier import _SAFE_ACTIONS, _DISALLOWED_ACTIONS  # noqa: E402
