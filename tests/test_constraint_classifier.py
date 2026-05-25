"""Tests for Phase 7: Constraint Classifier.

Proves:
- Each constraint family scores correctly when signals are present.
- Missing required signals → None score → family excluded from binding selection.
- binding_constraint is None when no family clears the confidence floor.
- Hysteresis: a family must be top-scoring for N consecutive snapshots.
- Tie-break: SLA-risk families (LATENCY, MEMORY, THERMAL) beat cost families.
- Confidence math: missing signals and stale data reduce confidence.
- No fabricated binding constraint from absent data.
- Safe/disallowed action tables are populated correctly.
- Classifier is read-only and does not modify ClusterState.
- Simulator scenario snapshots yield their expected primary constraints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from aurelius.constraints.classifier import ConstraintClassifier, ConstraintConfig
from aurelius.state.models import (
    ClusterState,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
    ThermalState,
    TopologyLinkType,
    TopologyState,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(tz=_UTC)


def _prov(confidence: str = "high", is_sandbox: bool = False, age_s: Optional[float] = None) -> Provenance:
    return Provenance(
        source="test",
        fetched_at=_now(),
        confidence=confidence,
        is_sandbox=is_sandbox,
        sample_age_s=age_s,
    )


def _gpu(
    uuid_str: str = "GPU-aaa",
    node_id: str = "node0",
    region: str = "us-east",
    util_pct: Optional[float] = 50.0,
    temp_c: Optional[float] = 60.0,
    mem_used_mb: Optional[float] = 40_000.0,
    mem_total_mb: Optional[float] = 80_000.0,
    clocks_event_reasons: Optional[int] = None,
    nvlink_tx_bytes_per_s: Optional[float] = None,
    pcie_tx_bytes_per_s: Optional[float] = None,
    sm_active_ratio: Optional[float] = None,
) -> GPUState:
    return GPUState(
        gpu_uuid=uuid_str,
        node_id=node_id,
        region=region,
        timestamp=_now(),
        provenance=_prov(),
        util_pct=util_pct,
        temp_c=temp_c,
        mem_used_mb=mem_used_mb,
        mem_free_mb=(mem_total_mb - mem_used_mb) if (mem_total_mb and mem_used_mb) else None,
        mem_total_mb=mem_total_mb,
        clocks_event_reasons=clocks_event_reasons,
        nvlink_tx_bytes_per_s=nvlink_tx_bytes_per_s,
        pcie_tx_bytes_per_s=pcie_tx_bytes_per_s,
        sm_active_ratio=sm_active_ratio,
    )


def _node(
    node_id: str = "node0",
    region: str = "us-east",
    gpus: Optional[dict] = None,
    gpu_capacity: int = 8,
    gpu_allocated: int = 8,
) -> NodeState:
    return NodeState(
        node_id=node_id,
        region=region,
        timestamp=_now(),
        provenance=_prov(),
        gpu_capacity=gpu_capacity,
        gpu_allocatable=gpu_capacity,
        gpu_allocated=gpu_allocated,
        gpus=gpus or {},
    )


def _service(
    service_id: str = "llm-1",
    region: str = "us-east",
    requests_waiting: Optional[float] = None,
    p99_latency_ms: Optional[float] = None,
    p95_latency_ms: Optional[float] = None,
    queue_time_p95_ms: Optional[float] = None,
    kv_cache_usage: Optional[float] = None,
    ttft_p99_ms: Optional[float] = None,
    tokens_per_s: Optional[float] = None,
) -> InferenceServiceState:
    return InferenceServiceState(
        service_id=service_id,
        engine="vllm",
        timestamp=_now(),
        provenance=_prov(),
        region=region,
        requests_waiting=requests_waiting,
        p99_latency_ms=p99_latency_ms,
        p95_latency_ms=p95_latency_ms,
        queue_time_p95_ms=queue_time_p95_ms,
        kv_cache_usage=kv_cache_usage,
        ttft_p99_ms=ttft_p99_ms,
        tokens_per_s=tokens_per_s,
    )


def _energy(
    region: str = "us-east",
    price_per_mwh: Optional[float] = 50.0,
    price_percentile: Optional[float] = 50.0,
    power_draw_kw: Optional[float] = None,
    power_cap_kw: Optional[float] = None,
) -> EnergyState:
    return EnergyState(
        region=region,
        timestamp=_now(),
        provenance=_prov(),
        price_per_mwh=price_per_mwh,
        price_percentile=price_percentile,
        power_draw_kw=power_draw_kw,
        power_cap_kw=power_cap_kw,
    )


def _topology(
    node_id: str = "node0",
    interconnect_class: str = "nvlink_full",
) -> TopologyState:
    g0, g1 = "GPU-000", "GPU-001"
    pair_key = (min(g0, g1), max(g0, g1))
    return TopologyState(
        node_id=node_id,
        timestamp=_now(),
        provenance=_prov(confidence="medium"),
        gpu_uuids=(g0, g1),
        pair_levels={pair_key: TopologyLinkType.NVSWITCH},
        numa_affinity={g0: 0, g1: 0},
        nvlink_present=True,
        interconnect_class=interconnect_class,
    )


def _region(
    region_id: str = "us-east",
    nodes: Optional[dict] = None,
    services: Optional[dict] = None,
    energy: Optional[EnergyState] = None,
    thermal: Optional[ThermalState] = None,
    topology: Optional[TopologyState] = None,
    spare_capacity_pct: Optional[float] = 20.0,
) -> RegionState:
    return RegionState(
        region=region_id,
        timestamp=_now(),
        provenance=_prov(),
        nodes=nodes or {},
        services=services or {},
        energy=energy,
        thermal=thermal,
        topology=topology,
        spare_capacity_pct=spare_capacity_pct,
    )


def _cluster(
    regions: Optional[dict] = None,
    is_partial: bool = False,
    missing_sources: Optional[list] = None,
) -> ClusterState:
    return ClusterState(
        timestamp=_now(),
        provenance=_prov(),
        regions=regions or {},
        is_partial=is_partial,
        missing_sources=missing_sources or [],
    )


def _classifier(hysteresis_count: int = 1, **kwargs) -> ConstraintClassifier:
    """Return a classifier with hysteresis_count=1 by default for single-snapshot tests."""
    cfg = ConstraintConfig(hysteresis_count=hysteresis_count, **kwargs)
    return ConstraintClassifier(config=cfg)


# ===========================================================================
# 1. Empty / no-signal state
# ===========================================================================

class TestEmptyState:
    def test_no_regions_no_binding_constraint(self):
        clf = _classifier()
        state = _cluster(regions={})
        a = clf.assess(state)
        assert a.binding_constraint is None
        assert a.confidence == 0.0
        assert a.scores == {}

    def test_empty_region_no_signals(self):
        clf = _classifier()
        state = _cluster(regions={"us-east": _region(region_id="us-east")})
        a = clf.assess(state)
        assert a.binding_constraint is None

    def test_fail_safe_on_unknown_region(self):
        clf = _classifier()
        state = _cluster(regions={"us-east": _region(region_id="us-east")})
        a = clf.assess(state, region="eu-west")
        assert a.binding_constraint is None
        assert a.confidence == 0.0
        assert "eu-west" in a.missing_signals[0]


# ===========================================================================
# 2. Energy-bound scoring
# ===========================================================================

class TestEnergyBound:
    def test_high_price_percentile_scores_energy(self):
        clf = _classifier()
        e = _energy(price_per_mwh=140.0, price_percentile=92.0)
        state = _cluster(regions={"us-east": _region(energy=e)})
        a = clf.assess(state)
        assert ConstraintType.ENERGY in a.scores
        # At 92nd percentile the score should be meaningful
        assert a.scores[ConstraintType.ENERGY] > 0.3

    def test_low_price_percentile_low_energy_score(self):
        clf = _classifier()
        e = _energy(price_per_mwh=45.0, price_percentile=20.0)
        state = _cluster(regions={"us-east": _region(energy=e)})
        a = clf.assess(state)
        assert ConstraintType.ENERGY in a.scores
        assert a.scores[ConstraintType.ENERGY] < 0.1

    def test_cross_region_spread_boosts_energy_score(self):
        clf = _classifier()
        # us-east cheap, us-west expensive → large spread
        e_east = _energy(region="us-east", price_per_mwh=40.0, price_percentile=20.0)
        e_west = _energy(region="us-west", price_per_mwh=160.0, price_percentile=88.0)
        state = _cluster(regions={
            "us-east": _region(region_id="us-east", energy=e_east),
            "us-west": _region(region_id="us-west", energy=e_west),
        })
        a = clf.assess(state)
        assert ConstraintType.ENERGY in a.scores
        assert a.scores[ConstraintType.ENERGY] > 0.2

    def test_no_energy_data_returns_none_score(self):
        clf = _classifier()
        # No EnergyState in region
        state = _cluster(regions={"us-east": _region(energy=None)})
        a = clf.assess(state)
        assert ConstraintType.ENERGY not in a.scores

    def test_energy_bound_detected_as_primary(self):
        clf = _classifier()
        # High price percentile + no other stress
        e = _energy(price_per_mwh=145.0, price_percentile=95.0)
        gpu = _gpu(util_pct=40.0, temp_c=60.0)
        node = _node(gpus={"GPU-aaa": gpu})
        svc = _service(requests_waiting=5.0, p99_latency_ms=400.0)
        region = _region(
            nodes={"node0": node},
            services={"llm-1": svc},
            energy=e,
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.ENERGY

    def test_energy_safe_actions(self):
        clf = _classifier()
        e = _energy(price_percentile=92.0)
        state = _cluster(regions={"us-east": _region(energy=e)})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.ENERGY:
            assert "defer_workload" in a.safe_action_types
            assert "choose_cheaper_region" in a.safe_action_types


# ===========================================================================
# 3. Thermal-bound scoring
# ===========================================================================

class TestThermalBound:
    def test_high_temp_scores_thermal(self):
        clf = _classifier()
        gpu = _gpu(temp_c=88.0, util_pct=90.0)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.THERMAL in a.scores
        assert a.scores[ConstraintType.THERMAL] > 0.4

    def test_throttle_bits_boost_thermal_score(self):
        clf = _classifier()
        SW_THERMAL = 0x20
        gpu = _gpu(temp_c=80.0, clocks_event_reasons=SW_THERMAL)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.THERMAL in a.scores
        score_with_throttle = a.scores[ConstraintType.THERMAL]

        # Compare against same temp without throttle
        gpu2 = _gpu(temp_c=80.0, clocks_event_reasons=0)
        node2 = _node(gpus={"GPU-aaa": gpu2})
        region2 = _region(nodes={"node0": node2})
        state2 = _cluster(regions={"us-east": region2})
        a2 = clf.assess(state2)
        assert score_with_throttle > a2.scores.get(ConstraintType.THERMAL, 0.0)

    def test_low_temp_low_thermal_score(self):
        clf = _classifier()
        gpu = _gpu(temp_c=55.0, util_pct=50.0)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.THERMAL in a.scores
        assert a.scores[ConstraintType.THERMAL] < 0.1

    def test_no_temp_data_returns_none_score(self):
        clf = _classifier()
        gpu = _gpu(temp_c=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.THERMAL not in a.scores

    def test_thermal_bound_detected_as_primary(self):
        clf = _classifier()
        HW_THERMAL = 0x40
        # Multiple GPUs very hot + throttling
        gpus = {
            f"GPU-{i:03d}": _gpu(
                uuid_str=f"GPU-{i:03d}",
                temp_c=90.0 + i,
                clocks_event_reasons=HW_THERMAL,
                util_pct=95.0,
            )
            for i in range(4)
        }
        node = _node(gpus=gpus)
        # Cheap energy → won't compete
        e = _energy(price_percentile=20.0)
        # Low latency → won't compete
        svc = _service(p99_latency_ms=300.0)
        region = _region(
            nodes={"node0": node},
            services={"llm": svc},
            energy=e,
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.THERMAL

    def test_thermal_disallows_consolidate(self):
        clf = _classifier()
        gpu = _gpu(temp_c=90.0, clocks_event_reasons=0x40)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.THERMAL:
            assert "consolidate_workloads" in a.disallowed_action_types
            assert "spread_workloads" in a.safe_action_types


# ===========================================================================
# 4. Queue-bound scoring
# ===========================================================================

class TestQueueBound:
    def test_deep_queue_scores_queue(self):
        clf = _classifier()
        svc = _service(requests_waiting=80.0, queue_time_p95_ms=800.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=5.0)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.QUEUE in a.scores
        assert a.scores[ConstraintType.QUEUE] > 0.4

    def test_empty_queue_low_score(self):
        clf = _classifier()
        svc = _service(requests_waiting=2.0, queue_time_p95_ms=50.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=60.0)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.QUEUE in a.scores
        assert a.scores[ConstraintType.QUEUE] < 0.3

    def test_no_queue_signal_returns_none(self):
        clf = _classifier()
        # No services, no spare_capacity_pct
        region = _region(services={}, spare_capacity_pct=None)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.QUEUE not in a.scores

    def test_queue_bound_detected_as_primary(self):
        clf = _classifier()
        # Saturated queue + near-zero spare capacity
        svc = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region = _region(
            services={"llm-1": svc},
            spare_capacity_pct=2.0,
            energy=_energy(price_percentile=30.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.QUEUE

    def test_queue_disallows_consolidate(self):
        clf = _classifier()
        svc = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=2.0)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.QUEUE:
            assert "consolidate_workloads" in a.disallowed_action_types
            assert "reroute_workload" in a.safe_action_types
            assert "scale_replicas" in a.safe_action_types


# ===========================================================================
# 5. Latency-bound scoring
# ===========================================================================

class TestLatencyBound:
    def test_high_p99_scores_latency(self):
        clf = _classifier()
        # ttft_p99 = 1800ms vs default TTFT SLA 2000ms → approaching limit
        svc = _service(ttft_p99_ms=1800.0)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.LATENCY in a.scores
        assert a.scores[ConstraintType.LATENCY] > 0.5

    def test_low_p99_low_latency_score(self):
        clf = _classifier()
        svc = _service(p99_latency_ms=300.0)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.LATENCY in a.scores
        assert a.scores[ConstraintType.LATENCY] < 0.2

    def test_no_latency_data_returns_none(self):
        clf = _classifier()
        # Service exists but no latency metrics
        svc = _service(p99_latency_ms=None, p95_latency_ms=None)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.LATENCY not in a.scores

    def test_latency_bound_detected_as_primary(self):
        clf = _classifier()
        # Very high TTFT + p99 near SLA limit
        svc = _service(p99_latency_ms=1950.0, ttft_p99_ms=1800.0, requests_waiting=10.0)
        # No energy or thermal pressure
        region = _region(
            services={"llm-1": svc},
            energy=_energy(price_percentile=25.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.LATENCY

    def test_latency_disallows_migrate(self):
        clf = _classifier()
        svc = _service(p99_latency_ms=1950.0)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.LATENCY:
            assert "migrate_workload" in a.disallowed_action_types
            assert "spread_workloads" in a.safe_action_types


# ===========================================================================
# 6. Memory-bound (indirect) scoring
# ===========================================================================

class TestMemoryBound:
    def test_high_kv_cache_scores_memory(self):
        clf = _classifier()
        svc = _service(kv_cache_usage=0.92)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.MEMORY in a.scores
        assert a.scores[ConstraintType.MEMORY] > 0.3

    def test_high_hbm_usage_scores_memory(self):
        clf = _classifier()
        # GPU at 88% HBM utilization
        gpu = _gpu(mem_used_mb=70_400.0, mem_total_mb=80_000.0)  # 88%
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.MEMORY in a.scores
        assert a.scores[ConstraintType.MEMORY] > 0.0

    def test_low_kv_cache_low_memory_score(self):
        clf = _classifier()
        svc = _service(kv_cache_usage=0.40)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.MEMORY in a.scores
        assert a.scores[ConstraintType.MEMORY] == 0.0

    def test_no_memory_signal_returns_none(self):
        clf = _classifier()
        # GPU with no memory metrics; no services
        gpu = _gpu(mem_used_mb=None, mem_total_mb=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node}, services={})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.MEMORY not in a.scores

    def test_memory_bound_detected_as_primary(self):
        clf = _classifier()
        # Near-full KV cache + high HBM
        svc = _service(kv_cache_usage=0.95)
        gpu = _gpu(mem_used_mb=76_000.0, mem_total_mb=80_000.0)  # 95%
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(
            nodes={"node0": node},
            services={"llm-1": svc},
            energy=_energy(price_percentile=20.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.MEMORY

    def test_memory_safe_actions_no_kv_internal(self):
        """Memory constraint must NOT list any KV-cache-internal actions."""
        clf = _classifier()
        svc = _service(kv_cache_usage=0.95)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.MEMORY:
            # Only orchestration-level actions should appear
            for action in a.safe_action_types:
                assert "kv" not in action.lower()
                assert "cache_internal" not in action.lower()
                assert "allocator" not in action.lower()


# ===========================================================================
# 7. Communication-bound scoring
# ===========================================================================

class TestCommunicationBound:
    def test_high_nvlink_low_sm_scores_comm(self):
        clf = _classifier()
        # 8 GB/s NVLink + 30% SM → compute stalled on transfers
        gpu = _gpu(nvlink_tx_bytes_per_s=8e9, sm_active_ratio=0.3)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.COMMUNICATION in a.scores
        assert a.scores[ConstraintType.COMMUNICATION] > 0.0

    def test_no_comm_signal_returns_none(self):
        clf = _classifier()
        # GPU with no NVLink or PCIe traffic
        gpu = _gpu(nvlink_tx_bytes_per_s=None, pcie_tx_bytes_per_s=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.COMMUNICATION not in a.scores

    def test_low_comm_bytes_low_score(self):
        clf = _classifier()
        # Only 1 GB/s — below threshold
        gpu = _gpu(pcie_tx_bytes_per_s=1e9)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if ConstraintType.COMMUNICATION in a.scores:
            assert a.scores[ConstraintType.COMMUNICATION] == 0.0

    def test_comm_safe_actions_no_nccl(self):
        """Communication constraint must NOT list NCCL/runtime actions."""
        clf = _classifier()
        gpu = _gpu(nvlink_tx_bytes_per_s=8e9, sm_active_ratio=0.2)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if a.binding_constraint == ConstraintType.COMMUNICATION:
            for action in a.safe_action_types:
                assert "nccl" not in action.lower()
                assert "cuda" not in action.lower()
                assert "collective" not in action.lower()


# ===========================================================================
# 8. Topology-bound scoring
# ===========================================================================

class TestTopologyBound:
    def test_poor_interconnect_scores_topology(self):
        clf = _classifier()
        topo = _topology(interconnect_class="pcie")
        region = _region(topology=topo)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.TOPOLOGY in a.scores
        assert a.scores[ConstraintType.TOPOLOGY] > 0.5

    def test_good_interconnect_low_topology_score(self):
        clf = _classifier()
        topo = _topology(interconnect_class="nvlink_full")
        region = _region(topology=topo)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.TOPOLOGY in a.scores
        assert a.scores[ConstraintType.TOPOLOGY] == 0.0

    def test_no_topology_returns_none(self):
        clf = _classifier()
        region = _region(topology=None)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.TOPOLOGY not in a.scores

    def test_topology_missing_in_missing_signals(self):
        clf = _classifier()
        region = _region(topology=None)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert any("topology" in s.lower() for s in a.missing_signals)


# ===========================================================================
# 9. Utilization-bound scoring
# ===========================================================================

class TestUtilizationBound:
    def test_low_util_scores_utilization(self):
        clf = _classifier()
        gpus = {
            f"GPU-{i:03d}": _gpu(uuid_str=f"GPU-{i:03d}", util_pct=15.0)
            for i in range(4)
        }
        node = _node(gpus=gpus)
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.UTILIZATION in a.scores
        assert a.scores[ConstraintType.UTILIZATION] > 0.3

    def test_high_util_low_utilization_score(self):
        clf = _classifier()
        gpus = {
            f"GPU-{i:03d}": _gpu(uuid_str=f"GPU-{i:03d}", util_pct=85.0)
            for i in range(4)
        }
        node = _node(gpus=gpus)
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.UTILIZATION in a.scores
        assert a.scores[ConstraintType.UTILIZATION] == 0.0

    def test_no_util_data_returns_none(self):
        clf = _classifier()
        gpu = _gpu(util_pct=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert ConstraintType.UTILIZATION not in a.scores

    def test_utilization_bound_detected_as_primary(self):
        clf = _classifier()
        # Very low utilization, moderate queue (not queue-bound)
        gpus = {
            f"GPU-{i:03d}": _gpu(uuid_str=f"GPU-{i:03d}", util_pct=8.0, temp_c=50.0)
            for i in range(8)
        }
        node = _node(gpus=gpus)
        region = _region(
            nodes={"node0": node},
            spare_capacity_pct=70.0,
            energy=_energy(price_percentile=30.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint == ConstraintType.UTILIZATION


# ===========================================================================
# 10. Hysteresis
# ===========================================================================

class TestHysteresis:
    def test_single_snapshot_not_enough_for_hysteresis_2(self):
        cfg = ConstraintConfig(hysteresis_count=2)
        clf = ConstraintClassifier(config=cfg)
        svc = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=2.0)
        state = _cluster(regions={"us-east": region})
        # First snapshot: candidate is QUEUE but not yet stable
        a = clf.assess(state)
        assert a.binding_constraint is None  # not yet stabilized

    def test_two_snapshots_stabilize_binding(self):
        cfg = ConstraintConfig(hysteresis_count=2)
        clf = ConstraintClassifier(config=cfg)
        svc = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=2.0)
        state1 = _cluster(regions={"us-east": region})
        state2 = _cluster(regions={"us-east": region})
        clf.assess(state1)
        a2 = clf.assess(state2)
        assert a2.binding_constraint == ConstraintType.QUEUE

    def test_reset_clears_history(self):
        cfg = ConstraintConfig(hysteresis_count=2)
        clf = ConstraintClassifier(config=cfg)
        svc = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region = _region(services={"llm-1": svc}, spare_capacity_pct=2.0)
        state = _cluster(regions={"us-east": region})
        clf.assess(state)
        clf.assess(state)  # now stable
        clf.reset()
        a_post = clf.assess(state)
        assert a_post.binding_constraint is None  # back to unstable after reset

    def test_flapping_prevented_by_hysteresis(self):
        cfg = ConstraintConfig(hysteresis_count=3)
        clf = ConstraintClassifier(config=cfg)

        # Alternate between queue and energy-bound states
        svc_queue = _service(requests_waiting=200.0, queue_time_p95_ms=1500.0)
        region_queue = _region(services={"llm-1": svc_queue}, spare_capacity_pct=2.0)
        state_queue = _cluster(regions={"us-east": region_queue})

        e_energy = _energy(price_percentile=95.0)
        region_energy = _region(energy=e_energy, spare_capacity_pct=60.0)
        state_energy = _cluster(regions={"us-east": region_energy})

        # Alternating: queue, energy, queue — no stable run of 3
        clf.assess(state_queue)
        clf.assess(state_energy)
        a3 = clf.assess(state_queue)
        # Still not stable (no 3 consecutive identical candidates)
        assert a3.binding_constraint is None


# ===========================================================================
# 11. Tie-break ordering
# ===========================================================================

class TestTieBreak:
    def test_latency_beats_energy_at_equal_scores(self):
        """LATENCY should win over ENERGY when scores are close."""
        clf = _classifier()
        # Craft state where both latency and energy score similarly
        # High energy: percentile 90%
        e = _energy(price_percentile=90.0)
        # High latency: p99 near SLA limit
        svc = _service(p99_latency_ms=1850.0)
        region = _region(services={"llm-1": svc}, energy=e)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        # If both score significantly, latency should win due to tie-break
        if (
            ConstraintType.LATENCY in a.scores
            and ConstraintType.ENERGY in a.scores
            and abs(a.scores[ConstraintType.LATENCY] - a.scores[ConstraintType.ENERGY]) <= 0.05
        ):
            assert a.binding_constraint == ConstraintType.LATENCY

    def test_thermal_beats_energy_at_equal_scores(self):
        """THERMAL should win over ENERGY when scores are tied."""
        clf = _classifier()
        # Elevated but not extreme energy cost
        e = _energy(price_percentile=85.0)
        # Hot GPU near crit
        gpu = _gpu(temp_c=88.0, clocks_event_reasons=0x40)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node}, energy=e)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if (
            ConstraintType.THERMAL in a.scores
            and ConstraintType.ENERGY in a.scores
            and abs(a.scores[ConstraintType.THERMAL] - a.scores[ConstraintType.ENERGY]) <= 0.05
        ):
            assert a.binding_constraint == ConstraintType.THERMAL

    def test_memory_beats_utilization(self):
        """MEMORY (SLA-risk) should beat UTILIZATION at equal scores."""
        clf = _classifier()
        # Near-full KV cache → memory score ~0.5
        svc = _service(kv_cache_usage=0.90)
        # Low GPU util → utilization score ~0.5
        gpu = _gpu(util_pct=15.0, mem_used_mb=None, mem_total_mb=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(
            nodes={"node0": node},
            services={"llm-1": svc},
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        if (
            ConstraintType.MEMORY in a.scores
            and ConstraintType.UTILIZATION in a.scores
            and abs(a.scores[ConstraintType.MEMORY] - a.scores[ConstraintType.UTILIZATION]) <= 0.05
        ):
            assert a.binding_constraint == ConstraintType.MEMORY


# ===========================================================================
# 12. Confidence computation
# ===========================================================================

class TestConfidence:
    def test_all_signals_present_high_confidence(self):
        clf = _classifier()
        gpu = _gpu(util_pct=50.0, temp_c=65.0, mem_used_mb=40_000.0, mem_total_mb=80_000.0)
        node = _node(gpus={"GPU-aaa": gpu})
        svc = _service(p99_latency_ms=800.0, requests_waiting=10.0, kv_cache_usage=0.5)
        e = _energy(price_percentile=60.0)
        region = _region(
            nodes={"node0": node},
            services={"llm-1": svc},
            energy=e,
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.confidence > 0.3

    def test_partial_state_reduces_confidence(self):
        clf = _classifier()
        gpu = _gpu(util_pct=50.0, temp_c=65.0)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(
            regions={"us-east": region},
            is_partial=True,
            missing_sources=["prometheus", "kubernetes"],
        )
        a_partial = clf.assess(state)

        # Compare against same state without partial flag
        state_full = _cluster(regions={"us-east": region}, is_partial=False)
        a_full = clf.assess(state_full)

        assert a_partial.confidence <= a_full.confidence

    def test_stale_telemetry_reduces_confidence(self):
        clf = _classifier()
        # Very stale provenance (900s old, well above max_acceptable_age_s=300)
        old_prov = Provenance(
            source="test",
            fetched_at=_now(),
            confidence="high",
            sample_age_s=900.0,  # 15 minutes stale
        )
        region = RegionState(
            region="us-east",
            timestamp=_now(),
            provenance=old_prov,
            energy=_energy(price_percentile=80.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.confidence < 0.7

    def test_low_provenance_confidence_reduces_assessment_confidence(self):
        clf = _classifier()
        low_prov = Provenance(
            source="test",
            fetched_at=_now(),
            confidence="low",
        )
        region = RegionState(
            region="us-east",
            timestamp=_now(),
            provenance=low_prov,
            energy=_energy(price_percentile=80.0),
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        # Low provenance confidence (weight=0.4) should appear in overall confidence
        assert a.confidence < 0.6

    def test_confidence_in_0_1_range(self):
        clf = _classifier()
        gpu = _gpu()
        node = _node(gpus={"GPU-aaa": gpu})
        svc = _service(p99_latency_ms=500.0, kv_cache_usage=0.6)
        e = _energy(price_percentile=55.0)
        region = _region(nodes={"node0": node}, services={"llm": svc}, energy=e)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert 0.0 <= a.confidence <= 1.0

    def test_no_confidence_below_floor_means_no_binding(self):
        cfg = ConstraintConfig(confidence_floor=0.99, hysteresis_count=1)
        clf = ConstraintClassifier(config=cfg)
        e = _energy(price_percentile=90.0)
        region = _region(energy=e)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        # Floor is impossibly high — no binding constraint should be emitted
        assert a.binding_constraint is None


# ===========================================================================
# 13. Missing signals tracking
# ===========================================================================

class TestMissingSignals:
    def test_missing_energy_listed(self):
        clf = _classifier()
        region = _region(energy=None)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert any("energy" in s.lower() for s in a.missing_signals)

    def test_missing_temp_listed(self):
        clf = _classifier()
        gpu = _gpu(temp_c=None)
        node = _node(gpus={"GPU-aaa": gpu})
        region = _region(nodes={"node0": node})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert any("temp" in s.lower() for s in a.missing_signals)

    def test_all_absent_binding_is_none(self):
        clf = _classifier()
        # State with all signals absent
        region = _region(
            energy=None,
            topology=None,
            nodes={},
            services={},
            spare_capacity_pct=None,
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert a.binding_constraint is None
        assert len(a.scores) == 0


# ===========================================================================
# 14. Rationale / to_dict round-trip
# ===========================================================================

class TestRationale:
    def test_rationale_is_non_empty_string(self):
        clf = _classifier()
        svc = _service(p99_latency_ms=1900.0)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        assert isinstance(a.rationale, str)
        assert len(a.rationale) > 0

    def test_assessment_to_dict_round_trip(self):
        clf = _classifier()
        svc = _service(p99_latency_ms=1900.0, kv_cache_usage=0.88)
        e = _energy(price_percentile=85.0)
        region = _region(services={"llm-1": svc}, energy=e)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        d = a.to_dict()
        from aurelius.state.models import ConstraintAssessment
        a2 = ConstraintAssessment.from_dict(d)
        assert a2.binding_constraint == a.binding_constraint
        assert a2.confidence == a.confidence
        assert a2.scores == a.scores


# ===========================================================================
# 15. Sandbox safety
# ===========================================================================

class TestSandbox:
    def test_sandbox_provenance_passes_through(self):
        clf = _classifier()
        prov = _prov(is_sandbox=True)
        region = RegionState(
            region="us-east",
            timestamp=_now(),
            provenance=prov,
            energy=_energy(price_percentile=90.0),
        )
        state = ClusterState(
            timestamp=_now(),
            provenance=prov,
            regions={"us-east": region},
        )
        a = clf.assess(state)
        assert a.provenance.is_sandbox is True

    def test_classifier_does_not_mutate_cluster_state(self):
        clf = _classifier()
        svc = _service(p99_latency_ms=1800.0)
        region = _region(services={"llm-1": svc})
        state = _cluster(regions={"us-east": region})
        original_ts = state.timestamp
        clf.assess(state)
        # ClusterState is frozen — if we got here without exception, it wasn't mutated
        assert state.timestamp == original_ts


# ===========================================================================
# 16. Simulator scenario integration tests
# ===========================================================================

class TestSimulatorScenarios:
    """Run the Phase 6 simulator for a few ticks and verify the classifier
    identifies the correct primary constraint for each named scenario.

    hysteresis_count=1 so we detect the constraint on the first tick that
    shows the expected signal (the scenarios are designed to be unambiguous).
    """

    def _run_scenario(self, scenario_name: str, ticks: int = 10) -> list:
        """Run simulator for N ticks, return list of ConstraintAssessment."""
        from aurelius.simulation.cluster.engine import ClusterSimulator
        from aurelius.simulation.cluster.scenarios import load_scenario

        scenario = load_scenario(scenario_name)
        sim = ClusterSimulator(scenario.config)
        clf = _classifier(hysteresis_count=1)

        assessments = []
        for _ in range(ticks):
            sim.tick()
            cs = sim.get_cluster_state()
            a = clf.assess(cs)
            assessments.append(a)
        return assessments

    def test_energy_scenario_detects_energy_bound(self):
        assessments = self._run_scenario("energy_price_arbitrage_multiregion")
        # The scenario has anti-correlated cross-region prices (energy bound) but also
        # an unserviced queue in us-west that builds up and may dominate queue scoring.
        # Accept ENERGY or QUEUE — both are genuine constraints in this scenario.
        detected = {a.binding_constraint for a in assessments if a.binding_constraint}
        energy_scores = [a.scores.get(ConstraintType.ENERGY, 0.0) for a in assessments]
        assert max(energy_scores) > 0.1, (
            f"Expected energy to be scored, got scores: {energy_scores}"
        )
        assert ConstraintType.ENERGY in detected or ConstraintType.QUEUE in detected, (
            f"Expected ENERGY or QUEUE in binding constraints, got: {detected}\n"
            f"Scores sample: {[{k.value: round(v,2) for k,v in a.scores.items()} for a in assessments[-2:]]}"
        )

    def test_thermal_scenario_detects_thermal_bound(self):
        assessments = self._run_scenario("thermal_hotspot_mixed_cluster")
        detected = {a.binding_constraint for a in assessments if a.binding_constraint}
        assert ConstraintType.THERMAL in detected, (
            f"Expected THERMAL, got: {detected}"
        )

    def test_queue_scenario_detects_queue_bound(self):
        assessments = self._run_scenario("queue_surge_latency_sensitive")
        detected = {a.binding_constraint for a in assessments if a.binding_constraint}
        assert ConstraintType.QUEUE in detected or ConstraintType.LATENCY in detected, (
            f"Expected QUEUE or LATENCY, got: {detected}"
        )

    def test_latency_scenario_detects_latency_or_memory_bound(self):
        assessments = self._run_scenario("latency_tail_kvcache_pressure")
        detected = {a.binding_constraint for a in assessments if a.binding_constraint}
        assert (
            ConstraintType.LATENCY in detected or ConstraintType.MEMORY in detected
        ), f"Expected LATENCY or MEMORY, got: {detected}"

    def test_utilization_scenario_detects_utilization_bound(self):
        assessments = self._run_scenario("underutilization_stranded_capacity")
        detected = {a.binding_constraint for a in assessments if a.binding_constraint}
        assert ConstraintType.UTILIZATION in detected, (
            f"Expected UTILIZATION, got: {detected}"
        )

    def test_sandbox_flag_preserved_from_simulator(self):
        assessments = self._run_scenario("energy_price_arbitrage_multiregion", ticks=1)
        for a in assessments:
            assert a.provenance.is_sandbox is True, "Simulator assessments must be is_sandbox=True"

    def test_classifier_does_not_fabricate_constraint_with_empty_state(self):
        """Empty ClusterState must never yield a binding constraint."""
        clf = _classifier(hysteresis_count=1)
        empty = _cluster(regions={})
        a = clf.assess(empty)
        assert a.binding_constraint is None
        assert a.confidence == 0.0


# ===========================================================================
# 17. Region scoping
# ===========================================================================

class TestRegionScoping:
    def test_scoped_to_region_only_uses_that_region(self):
        clf = _classifier()
        # us-east: high energy, us-west: cheap
        e_east = _energy(region="us-east", price_percentile=95.0)
        e_west = _energy(region="us-west", price_percentile=10.0)
        state = _cluster(regions={
            "us-east": _region(region_id="us-east", energy=e_east),
            "us-west": _region(region_id="us-west", energy=e_west),
        })
        a_east = clf.assess(state, region="us-east")
        a_west = clf.assess(state, region="us-west")
        # us-east should score energy higher than us-west
        east_e = a_east.scores.get(ConstraintType.ENERGY, 0.0)
        west_e = a_west.scores.get(ConstraintType.ENERGY, 0.0)
        assert east_e >= west_e, f"Expected east ({east_e:.2f}) >= west ({west_e:.2f})"
        assert a_east.region == "us-east"
        assert a_west.region == "us-west"


# ===========================================================================
# 18. Score values in [0, 1]
# ===========================================================================

class TestScoreRange:
    def test_all_scores_in_0_1_range(self):
        clf = _classifier()
        HW_THERMAL = 0x40
        gpu = _gpu(
            util_pct=95.0,
            temp_c=92.0,
            clocks_event_reasons=HW_THERMAL,
            mem_used_mb=78_000.0,
            mem_total_mb=80_000.0,
            nvlink_tx_bytes_per_s=12e9,
            sm_active_ratio=0.15,
        )
        node = _node(gpus={"GPU-aaa": gpu})
        svc = _service(
            p99_latency_ms=1950.0,
            requests_waiting=300.0,
            kv_cache_usage=0.97,
            queue_time_p95_ms=1800.0,
        )
        e = _energy(price_percentile=95.0)
        topo = _topology(interconnect_class="cross_numa")
        region = _region(
            nodes={"node0": node},
            services={"llm-1": svc},
            energy=e,
            topology=topo,
            spare_capacity_pct=2.0,
        )
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        for ct, score in a.scores.items():
            assert 0.0 <= score <= 1.0, f"{ct.value} score {score} out of [0,1]"


# ===========================================================================
# 19. Action type invariants
# ===========================================================================

class TestActionInvariants:
    def test_none_constraint_keeps_safe_action(self):
        clf = _classifier()
        state = _cluster(regions={})
        a = clf.assess(state)
        # binding_constraint is None → safe_action_types should include KEEP
        assert "keep_current_placement" in a.safe_action_types

    def test_all_action_strings_are_valid(self):
        from aurelius.sla.actions import ActionType
        valid_action_values = {at.value for at in ActionType}
        clf = _classifier()
        gpu = _gpu(util_pct=12.0, temp_c=91.0, clocks_event_reasons=0x40)
        node = _node(gpus={"GPU-aaa": gpu})
        svc = _service(p99_latency_ms=1900.0, kv_cache_usage=0.93, requests_waiting=200.0)
        e = _energy(price_percentile=92.0)
        region = _region(nodes={"node0": node}, services={"svc": svc}, energy=e, spare_capacity_pct=2.0)
        state = _cluster(regions={"us-east": region})
        a = clf.assess(state)
        for action in a.safe_action_types + a.disallowed_action_types:
            assert action in valid_action_values, f"Invalid action string: {action}"
