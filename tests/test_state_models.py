"""Tests for aurelius/state/models.py — Phase 1 state model validation."""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from aurelius.state.models import (
    ClusterState,
    GPUState,
    InferenceServiceState,
    NodeState,
    RegionState,
    QueueState,
    WorkloadState,
    TopologyState,
    TopologyLink,
    TopologyNode,
    EnergyState,
    ThermalState,
    MigrationEvent,
    MigrationHistory,
    ConstraintAssessment,
    Recommendation,
    Provenance,
    LinkType,
    RuntimeType,
    WorkloadType,
    ConstraintType,
    ImplementationMode,
)

UTC = timezone.utc
NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_valid_provenance(self):
        p = Provenance(source="prometheus:dcgm", collected_at=NOW, confidence=0.9)
        assert p.source == "prometheus:dcgm"
        assert p.confidence == 0.9

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(source="test", collected_at=datetime(2025, 1, 1))

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Provenance(source="test", collected_at=NOW, confidence=1.5)
        with pytest.raises(ValidationError):
            Provenance(source="test", collected_at=NOW, confidence=-0.1)

    def test_confidence_none_allowed(self):
        p = Provenance(source="test", collected_at=NOW, confidence=None)
        assert p.confidence is None

    def test_string_iso_datetime_accepted(self):
        p = Provenance(source="test", collected_at="2025-01-15T12:00:00+00:00")
        assert p.collected_at.tzinfo is not None


# ---------------------------------------------------------------------------
# GPUState
# ---------------------------------------------------------------------------

class TestGPUState:
    def test_minimal_gpu(self):
        g = GPUState(gpu_id="GPU-001")
        assert g.gpu_id == "GPU-001"
        assert g.utilization_pct is None
        assert g.power_watts is None
        assert g.temperature_c is None

    def test_all_optionals_none_by_default(self):
        g = GPUState(gpu_id="GPU-001")
        optional_fields = [
            "uuid", "node_id", "model", "sm_activity_pct", "memory_used_bytes",
            "memory_total_bytes", "memory_bandwidth_util_pct",
            "nvlink_rx_bytes_per_sec", "nvlink_tx_bytes_per_sec",
            "pcie_rx_bytes_per_sec", "pcie_tx_bytes_per_sec",
            "xid_error_count", "thermal_throttle_active",
        ]
        for f in optional_fields:
            assert getattr(g, f) is None, f"{f} should be None"

    def test_percentage_validation(self):
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", utilization_pct=101.0)
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", utilization_pct=-1.0)

    def test_valid_percentage(self):
        g = GPUState(gpu_id="g1", utilization_pct=0.0)
        assert g.utilization_pct == 0.0
        g2 = GPUState(gpu_id="g1", utilization_pct=100.0)
        assert g2.utilization_pct == 100.0

    def test_negative_power_rejected(self):
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", power_watts=-1.0)

    def test_negative_memory_rejected(self):
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", memory_used_bytes=-1)

    def test_negative_bytes_per_sec_rejected(self):
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", nvlink_rx_bytes_per_sec=-5.0)

    def test_below_absolute_zero_rejected(self):
        with pytest.raises(ValidationError):
            GPUState(gpu_id="g1", temperature_c=-300.0)

    def test_json_round_trip(self):
        g = GPUState(
            gpu_id="GPU-abc123",
            uuid="GPU-abc123",
            node_id="node-1",
            model="H100",
            utilization_pct=75.0,
            power_watts=380.0,
            temperature_c=72.0,
        )
        data = g.model_dump()
        g2 = GPUState.model_validate(data)
        assert g2.gpu_id == g.gpu_id
        assert g2.utilization_pct == g.utilization_pct


# ---------------------------------------------------------------------------
# InferenceServiceState
# ---------------------------------------------------------------------------

class TestInferenceServiceState:
    def test_minimal_service(self):
        s = InferenceServiceState(service_id="svc-1")
        assert s.service_id == "svc-1"
        assert s.runtime == RuntimeType.UNKNOWN
        assert s.ttft_p50_ms is None
        assert s.kv_cache_usage_pct is None

    def test_invalid_latency_ordering(self):
        with pytest.raises(ValidationError):
            InferenceServiceState(
                service_id="svc-1",
                ttft_p50_ms=200.0,
                ttft_p99_ms=100.0,
            )

    def test_valid_latency_ordering(self):
        s = InferenceServiceState(
            service_id="svc-1",
            ttft_p50_ms=50.0,
            ttft_p95_ms=150.0,
            ttft_p99_ms=300.0,
        )
        assert s.ttft_p99_ms == 300.0

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            InferenceServiceState(service_id="s", ttft_p50_ms=-1.0)

    def test_percentage_validation(self):
        with pytest.raises(ValidationError):
            InferenceServiceState(service_id="s", kv_cache_usage_pct=150.0)
        with pytest.raises(ValidationError):
            InferenceServiceState(service_id="s", error_rate_pct=-1.0)

    def test_all_optional_none_default(self):
        s = InferenceServiceState(service_id="s")
        assert s.requests_per_second is None
        assert s.tokens_per_second is None
        assert s.prefix_cache_hit_rate_pct is None

    def test_json_round_trip(self):
        s = InferenceServiceState(
            service_id="llm-server",
            runtime=RuntimeType.VLLM,
            requests_per_second=42.0,
            ttft_p50_ms=80.0,
            ttft_p99_ms=320.0,
            kv_cache_usage_pct=68.0,
        )
        data = s.model_dump()
        s2 = InferenceServiceState.model_validate(data)
        assert s2.service_id == s.service_id
        assert s2.runtime == RuntimeType.VLLM


# ---------------------------------------------------------------------------
# ClusterState
# ---------------------------------------------------------------------------

class TestClusterState:
    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            ClusterState(timestamp=datetime(2025, 1, 1))

    def test_utc_timestamp_accepted(self):
        cs = ClusterState(timestamp=NOW)
        assert cs.timestamp.tzinfo is not None

    def test_empty_cluster_state(self):
        cs = ClusterState(timestamp=NOW)
        assert cs.gpu_count() == 0
        assert cs.node_count() == 0
        assert cs.mean_gpu_utilization() is None

    def test_mean_gpu_utilization(self):
        cs = ClusterState(
            timestamp=NOW,
            gpus={
                "g1": GPUState(gpu_id="g1", utilization_pct=60.0),
                "g2": GPUState(gpu_id="g2", utilization_pct=80.0),
                "g3": GPUState(gpu_id="g3"),  # None — should be excluded
            },
        )
        mean = cs.mean_gpu_utilization()
        assert mean == pytest.approx(70.0)

    def test_max_gpu_temperature(self):
        cs = ClusterState(
            timestamp=NOW,
            gpus={
                "g1": GPUState(gpu_id="g1", temperature_c=70.0),
                "g2": GPUState(gpu_id="g2", temperature_c=85.0),
            },
        )
        assert cs.max_gpu_temperature() == 85.0

    def test_max_temperature_none_when_no_data(self):
        cs = ClusterState(timestamp=NOW, gpus={"g1": GPUState(gpu_id="g1")})
        assert cs.max_gpu_temperature() is None

    def test_json_round_trip_from_fixture(self):
        import json
        import pathlib
        fixture = pathlib.Path(__file__).parent / "fixtures/cluster_state/minimal_cluster.json"
        data = json.loads(fixture.read_text())
        cs = ClusterState.model_validate(data)
        assert cs.cluster_id == "test-cluster-1"
        assert "us-east-1" in cs.regions
        assert "GPU-abc123" in cs.gpus
        assert cs.gpus["GPU-abc123"].utilization_pct == 75.5
        assert cs.gpus["GPU-abc123"].temperature_c == 72.0

        serialized = cs.model_dump()
        cs2 = ClusterState.model_validate(serialized)
        assert cs2.cluster_id == cs.cluster_id


# ---------------------------------------------------------------------------
# TopologyState
# ---------------------------------------------------------------------------

class TestTopologyState:
    def test_topology_distance_same_node(self):
        topo = TopologyState(
            nodes={"g1": TopologyNode(node_id="g1")},
        )
        assert topo.topology_distance("g1", "g1") == 0

    def test_topology_distance_direct_link(self):
        topo = TopologyState(
            nodes={
                "g1": TopologyNode(node_id="g1"),
                "g2": TopologyNode(node_id="g2"),
            },
            links=[
                TopologyLink(source_id="g1", target_id="g2", link_type=LinkType.NVLINK)
            ],
        )
        assert topo.topology_distance("g1", "g2") == 1
        assert topo.topology_distance("g2", "g1") == 1

    def test_topology_distance_two_hops(self):
        topo = TopologyState(
            nodes={
                "g1": TopologyNode(node_id="g1"),
                "g2": TopologyNode(node_id="g2"),
                "g3": TopologyNode(node_id="g3"),
            },
            links=[
                TopologyLink(source_id="g1", target_id="g2", link_type=LinkType.NVLINK),
                TopologyLink(source_id="g2", target_id="g3", link_type=LinkType.PCIE),
            ],
        )
        assert topo.topology_distance("g1", "g3") == 2

    def test_topology_distance_unreachable(self):
        topo = TopologyState(
            nodes={
                "g1": TopologyNode(node_id="g1"),
                "g2": TopologyNode(node_id="g2"),
            },
            links=[],
        )
        assert topo.topology_distance("g1", "g2") is None


# ---------------------------------------------------------------------------
# MigrationHistory
# ---------------------------------------------------------------------------

class TestMigrationHistory:
    def test_migration_event_naive_rejected(self):
        with pytest.raises(ValidationError):
            MigrationEvent(
                occurred_at=datetime(2025, 1, 1),
                workload_id="w1",
            )

    def test_migrations_in_window(self):
        e1 = MigrationEvent(occurred_at=NOW - timedelta(hours=2), workload_id="w1")
        e2 = MigrationEvent(occurred_at=NOW - timedelta(hours=1), workload_id="w1")
        e3 = MigrationEvent(occurred_at=NOW, workload_id="w1")
        hist = MigrationHistory(workload_id="w1", migrations=[e1, e2, e3])
        window = hist.migrations_in_window(
            since=NOW - timedelta(hours=1, minutes=30),
            until=NOW,
        )
        assert len(window) == 2

    def test_migration_count(self):
        hist = MigrationHistory(workload_id="w1")
        assert hist.migration_count == 0
        hist.migrations.append(
            MigrationEvent(occurred_at=NOW, workload_id="w1")
        )
        assert hist.migration_count == 1


# ---------------------------------------------------------------------------
# ConstraintAssessment
# ---------------------------------------------------------------------------

class TestConstraintAssessment:
    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            ConstraintAssessment(
                assessed_at=datetime(2025, 1, 1),
                cluster_id="test",
            )

    def test_defaults(self):
        ca = ConstraintAssessment(assessed_at=NOW, cluster_id="test")
        assert ca.primary_constraint == ConstraintType.INSUFFICIENT_DATA
        assert ca.confidence == 0.0
        assert ca.missing_metrics == []


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_defaults(self):
        r = Recommendation(
            recommendation_id="rec-001",
            generated_at=NOW,
            action="no_op",
        )
        assert r.implementation_mode == ImplementationMode.RECOMMENDATION_ONLY
        assert r.confidence == 0.0

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            Recommendation(
                recommendation_id="rec-001",
                generated_at=datetime(2025, 1, 1),
                action="no_op",
            )


# ---------------------------------------------------------------------------
# RegionState
# ---------------------------------------------------------------------------

class TestRegionState:
    def test_renewable_fraction_pct_bounds(self):
        with pytest.raises(ValidationError):
            RegionState(region_id="r1", renewable_fraction_pct=110.0)
        r = RegionState(region_id="r1", renewable_fraction_pct=50.0)
        assert r.renewable_fraction_pct == 50.0

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            RegionState(region_id="r1", energy_price_per_mwh=-5.0)

    def test_all_none_allowed(self):
        r = RegionState(region_id="r1")
        assert r.energy_price_per_mwh is None
        assert r.carbon_intensity_gco2_per_kwh is None
