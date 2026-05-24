"""Tests for aurelius/state/models.py.

Proves:
- UTC-aware timestamp validation (naive datetimes raise ValueError)
- None-not-zero behavior (missing fields stay None)
- Percentage range validation (0–100 for pct fields, 0–1 for ratio fields)
- Non-negative validation for rates/bytes/durations
- JSON round-trip via to_dict() / from_dict()
- ConstraintType and TopologyLinkType enum values
- Provenance confidence weight
- GPUState thermal_throttling property
- ClusterState.all_gpus / all_services aggregation
- MigrationHistory count helpers
- Impossible values are rejected (negative latency, pct > 100, etc.)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    MigrationEvent,
    MigrationHistory,
    NodeState,
    Provenance,
    Recommendation,
    RegionState,
    ThermalState,
    TopologyLinkType,
    TopologyState,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _prov(source: str = "test", confidence: str = "high") -> Provenance:
    return Provenance(source=source, fetched_at=NOW, confidence=confidence)


def _gpu(uuid_: str = "GPU-abc", **kwargs) -> GPUState:
    defaults = dict(
        gpu_uuid=uuid_,
        node_id="node-01",
        region="us-east",
        timestamp=NOW,
        provenance=_prov(),
    )
    defaults.update(kwargs)
    return GPUState(**defaults)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_valid_provenance(self):
        p = _prov("dcgm-exporter", "high")
        assert p.confidence_weight == 1.0

    def test_medium_confidence_weight(self):
        p = _prov(confidence="medium")
        assert p.confidence_weight == 0.7

    def test_low_confidence_weight(self):
        p = _prov(confidence="low")
        assert p.confidence_weight == 0.4

    def test_naive_datetime_rejected(self):
        naive = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="UTC-aware"):
            Provenance(source="x", fetched_at=naive, confidence="high")

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            Provenance(source="x", fetched_at=NOW, confidence="very_high")

    def test_negative_sample_age_rejected(self):
        with pytest.raises(ValueError, match="sample_age_s"):
            Provenance(source="x", fetched_at=NOW, confidence="high", sample_age_s=-1.0)

    def test_json_roundtrip(self):
        p = Provenance(source="prometheus", fetched_at=NOW, confidence="medium",
                       is_sandbox=True, sample_age_s=5.2)
        d = p.to_dict()
        p2 = Provenance.from_dict(d)
        assert p2.source == "prometheus"
        assert p2.confidence == "medium"
        assert p2.is_sandbox is True
        assert abs(p2.sample_age_s - 5.2) < 1e-9
        assert p2.fetched_at.tzinfo is not None

    def test_from_dict_naive_str_gets_utc(self):
        d = {"source": "x", "fetched_at": "2024-01-01T00:00:00", "confidence": "high"}
        p = Provenance.from_dict(d)
        assert p.fetched_at.tzinfo is not None


# ---------------------------------------------------------------------------
# GPUState
# ---------------------------------------------------------------------------

class TestGPUState:
    def test_valid_gpu_state(self):
        gpu = _gpu(util_pct=75.0, mem_used_mb=40000.0, mem_total_mb=80000.0,
                   power_w=350.0, temp_c=72.0)
        assert gpu.util_pct == 75.0
        assert gpu.power_w == 350.0

    def test_naive_timestamp_rejected(self):
        naive = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="UTC-aware"):
            _gpu(timestamp=naive)

    def test_util_pct_over_100_rejected(self):
        with pytest.raises(ValueError, match="util_pct"):
            _gpu(util_pct=101.0)

    def test_util_pct_negative_rejected(self):
        with pytest.raises(ValueError, match="util_pct"):
            _gpu(util_pct=-1.0)

    def test_negative_mem_rejected(self):
        with pytest.raises(ValueError, match="mem_used_mb"):
            _gpu(mem_used_mb=-100.0)

    def test_negative_power_rejected(self):
        with pytest.raises(ValueError, match="power_w"):
            _gpu(power_w=-1.0)

    def test_negative_pcie_tx_rejected(self):
        with pytest.raises(ValueError, match="pcie_tx_bytes_per_s"):
            _gpu(pcie_tx_bytes_per_s=-1.0)

    def test_health_penalty_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="health_penalty"):
            _gpu(health_penalty=1.5)

    def test_negative_gpu_index_rejected(self):
        with pytest.raises(ValueError, match="gpu_index"):
            _gpu(gpu_index=-1)

    def test_none_is_not_zero(self):
        gpu = _gpu()
        assert gpu.util_pct is None
        assert gpu.mem_used_mb is None
        assert gpu.power_w is None
        assert gpu.nvlink_tx_bytes_per_s is None

    def test_mem_util_pct_computed(self):
        gpu = _gpu(mem_used_mb=40000.0, mem_total_mb=80000.0)
        assert abs(gpu.mem_util_pct - 50.0) < 1e-6

    def test_mem_util_pct_none_when_total_missing(self):
        gpu = _gpu(mem_used_mb=40000.0)
        assert gpu.mem_util_pct is None

    def test_thermal_throttling_true_on_hw_thermal_bit(self):
        HW_THERMAL = 0x40
        gpu = _gpu(clocks_event_reasons=HW_THERMAL)
        assert gpu.thermal_throttling is True

    def test_thermal_throttling_false_on_zero(self):
        gpu = _gpu(clocks_event_reasons=0)
        assert gpu.thermal_throttling is False

    def test_thermal_throttling_none_when_reasons_absent(self):
        gpu = _gpu()
        assert gpu.thermal_throttling is None

    def test_sm_active_ratio_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="sm_active_ratio"):
            _gpu(sm_active_ratio=1.5)

    def test_ecc_sbe_total_negative_rejected(self):
        with pytest.raises(ValueError, match="ecc_sbe_total"):
            _gpu(ecc_sbe_total=-1)

    def test_json_roundtrip_full(self):
        gpu = _gpu(
            util_pct=80.0, mem_used_mb=32000.0, mem_total_mb=80000.0,
            power_w=400.0, temp_c=75.0, health_penalty=0.1, is_schedulable=True,
            clocks_event_reasons=0, xid_last=0,
        )
        d = gpu.to_dict()
        gpu2 = GPUState.from_dict(d)
        assert gpu2.gpu_uuid == gpu.gpu_uuid
        assert gpu2.util_pct == gpu.util_pct
        assert gpu2.power_w == gpu.power_w
        assert gpu2.timestamp.tzinfo is not None
        assert gpu2.is_schedulable is True

    def test_json_roundtrip_with_none_fields(self):
        gpu = _gpu()  # all optional fields None
        d = gpu.to_dict()
        gpu2 = GPUState.from_dict(d)
        assert gpu2.util_pct is None
        assert gpu2.nvlink_tx_bytes_per_s is None


# ---------------------------------------------------------------------------
# InferenceServiceState
# ---------------------------------------------------------------------------

class TestInferenceServiceState:
    def _svc(self, **kwargs) -> InferenceServiceState:
        defaults = dict(
            service_id="svc-1",
            engine="vllm",
            timestamp=NOW,
            provenance=_prov(),
        )
        defaults.update(kwargs)
        return InferenceServiceState(**defaults)

    def test_valid_service(self):
        svc = self._svc(p99_latency_ms=250.0, requests_waiting=5.0)
        assert svc.p99_latency_ms == 250.0
        assert svc.requests_waiting == 5.0

    def test_invalid_engine_rejected(self):
        with pytest.raises(ValueError, match="engine"):
            self._svc(engine="tensorflow")

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="UTC-aware"):
            self._svc(timestamp=datetime(2024, 1, 1))

    def test_negative_latency_rejected(self):
        with pytest.raises(ValueError, match="p99_latency_ms"):
            self._svc(p99_latency_ms=-1.0)

    def test_kv_cache_usage_over_1_rejected(self):
        with pytest.raises(ValueError, match="kv_cache_usage"):
            self._svc(kv_cache_usage=1.1)

    def test_prefix_cache_hit_rate_negative_rejected(self):
        with pytest.raises(ValueError, match="prefix_cache_hit_rate"):
            self._svc(prefix_cache_hit_rate=-0.1)

    def test_error_rate_pct_over_100_rejected(self):
        with pytest.raises(ValueError, match="error_rate_pct"):
            self._svc(error_rate_pct=101.0)

    def test_negative_replicas_rejected(self):
        with pytest.raises(ValueError, match="replicas"):
            self._svc(replicas=-1)

    def test_none_not_zero(self):
        svc = self._svc()
        assert svc.p99_latency_ms is None
        assert svc.kv_cache_usage is None
        assert svc.tokens_per_s is None

    def test_all_engines_accepted(self):
        for engine in ("vllm", "triton", "ray_serve", "unknown"):
            svc = self._svc(engine=engine)
            assert svc.engine == engine

    def test_json_roundtrip(self):
        svc = self._svc(
            p50_latency_ms=100.0, p95_latency_ms=200.0, p99_latency_ms=300.0,
            kv_cache_usage=0.85, prefix_cache_hit_rate=0.65,
            requests_running=10.0, requests_waiting=3.0, error_rate_pct=0.5,
        )
        d = svc.to_dict()
        svc2 = InferenceServiceState.from_dict(d)
        assert svc2.service_id == svc.service_id
        assert svc2.p99_latency_ms == svc.p99_latency_ms
        assert svc2.kv_cache_usage == svc.kv_cache_usage
        assert svc2.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# TopologyState
# ---------------------------------------------------------------------------

class TestTopologyState:
    def _topo(self, **kwargs) -> TopologyState:
        uuids = ("GPU-0", "GPU-1")
        pair_key = TopologyState.make_pair_key(*uuids)
        defaults = dict(
            node_id="node-01",
            timestamp=NOW,
            provenance=_prov("nvml", "medium"),
            gpu_uuids=uuids,
            pair_levels={pair_key: TopologyLinkType.NV4},
            numa_affinity={"GPU-0": 0, "GPU-1": 0},
        )
        defaults.update(kwargs)
        return TopologyState(**defaults)

    def test_valid_topology(self):
        t = self._topo(nvlink_present=True, interconnect_class="nvlink_full")
        assert t.nvlink_present is True
        assert t.interconnect_class == "nvlink_full"

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="UTC-aware"):
            self._topo(timestamp=datetime(2024, 1, 1))

    def test_invalid_interconnect_class_rejected(self):
        with pytest.raises(ValueError, match="interconnect_class"):
            self._topo(interconnect_class="super_fast")

    def test_unordered_pair_key_rejected(self):
        with pytest.raises(ValueError, match="pair_levels keys must be ordered"):
            TopologyState(
                node_id="n",
                timestamp=NOW,
                provenance=_prov(),
                gpu_uuids=("GPU-0", "GPU-1"),
                pair_levels={("GPU-1", "GPU-0"): TopologyLinkType.PIX},
                numa_affinity={},
            )

    def test_make_pair_key_ordering(self):
        k = TopologyState.make_pair_key("GPU-Z", "GPU-A")
        assert k == ("GPU-A", "GPU-Z")

    def test_link_between(self):
        t = self._topo()
        link = t.link_between("GPU-0", "GPU-1")
        assert link == TopologyLinkType.NV4

    def test_link_between_absent(self):
        t = self._topo()
        link = t.link_between("GPU-0", "GPU-99")
        assert link is None

    def test_json_roundtrip(self):
        t = self._topo(nvlink_present=True, interconnect_class="nvlink_partial")
        d = t.to_dict()
        t2 = TopologyState.from_dict(d)
        assert t2.node_id == t.node_id
        assert t2.nvlink_present is True
        assert t2.interconnect_class == "nvlink_partial"
        assert t2.timestamp.tzinfo is not None
        pair_key = TopologyState.make_pair_key("GPU-0", "GPU-1")
        assert t2.pair_levels[pair_key] == TopologyLinkType.NV4


# ---------------------------------------------------------------------------
# EnergyState
# ---------------------------------------------------------------------------

class TestEnergyState:
    def _energy(self, **kwargs) -> EnergyState:
        defaults = dict(region="us-east", timestamp=NOW, provenance=_prov())
        defaults.update(kwargs)
        return EnergyState(**defaults)

    def test_valid_energy(self):
        e = self._energy(price_per_mwh=45.0, price_percentile=80.0, pue=1.3)
        assert e.price_percentile == 80.0
        assert e.pue == 1.3

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="price_per_mwh"):
            self._energy(price_per_mwh=-1.0)

    def test_price_percentile_over_100_rejected(self):
        with pytest.raises(ValueError, match="price_percentile"):
            self._energy(price_percentile=101.0)

    def test_pue_below_1_rejected(self):
        with pytest.raises(ValueError, match="pue"):
            self._energy(pue=0.9)

    def test_none_not_zero(self):
        e = self._energy()
        assert e.price_per_mwh is None
        assert e.carbon_gco2_per_kwh is None

    def test_json_roundtrip(self):
        e = self._energy(price_per_mwh=50.0, price_percentile=90.0,
                         carbon_gco2_per_kwh=250.0, pue=1.4, power_draw_kw=500.0)
        e2 = EnergyState.from_dict(e.to_dict())
        assert e2.price_per_mwh == e.price_per_mwh
        assert e2.pue == e.pue
        assert e2.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# ThermalState
# ---------------------------------------------------------------------------

class TestThermalState:
    def _thermal(self, **kwargs) -> ThermalState:
        defaults = dict(region="us-east", timestamp=NOW, provenance=_prov())
        defaults.update(kwargs)
        return ThermalState(**defaults)

    def test_valid_thermal(self):
        t = self._thermal(max_gpu_temp_c=78.0, throttling_gpu_count=2,
                          total_gpu_count=8, cooling_headroom_pct=60.0)
        assert t.throttling_fraction == 2 / 8

    def test_cooling_headroom_over_100_rejected(self):
        with pytest.raises(ValueError, match="cooling_headroom_pct"):
            self._thermal(cooling_headroom_pct=101.0)

    def test_negative_throttling_count_rejected(self):
        with pytest.raises(ValueError, match="throttling_gpu_count"):
            self._thermal(throttling_gpu_count=-1)

    def test_throttling_fraction_none_without_denominator(self):
        t = self._thermal(throttling_gpu_count=3)
        assert t.throttling_fraction is None

    def test_json_roundtrip(self):
        t = self._thermal(max_gpu_temp_c=90.0, mean_gpu_temp_c=82.0,
                          throttling_gpu_count=4, total_gpu_count=8)
        t2 = ThermalState.from_dict(t.to_dict())
        assert t2.max_gpu_temp_c == 90.0
        assert t2.throttling_gpu_count == 4


# ---------------------------------------------------------------------------
# NodeState
# ---------------------------------------------------------------------------

class TestNodeState:
    def _node(self, **kwargs) -> NodeState:
        defaults = dict(node_id="node-01", region="us-east",
                        timestamp=NOW, provenance=_prov())
        defaults.update(kwargs)
        return NodeState(**defaults)

    def test_valid_node(self):
        n = self._node(gpu_capacity=8, gpu_allocatable=8, gpu_allocated=6)
        assert n.gpu_spare == 2

    def test_negative_gpu_capacity_rejected(self):
        with pytest.raises(ValueError, match="gpu_capacity"):
            self._node(gpu_capacity=-1)

    def test_gpu_spare_none_without_both(self):
        n = self._node(gpu_allocatable=8)
        assert n.gpu_spare is None

    def test_json_roundtrip(self):
        gpu = _gpu()
        n = self._node(gpu_capacity=8, gpu_allocated=3,
                       labels={"nvidia.com/gpu.product": "A100"},
                       gpus={gpu.gpu_uuid: gpu})
        d = n.to_dict()
        n2 = NodeState.from_dict(d)
        assert n2.gpu_capacity == 8
        assert n2.labels == {"nvidia.com/gpu.product": "A100"}
        assert len(n2.gpus) == 1


# ---------------------------------------------------------------------------
# RegionState
# ---------------------------------------------------------------------------

class TestRegionState:
    def _region(self, **kwargs) -> RegionState:
        defaults = dict(region="us-east", timestamp=NOW, provenance=_prov())
        defaults.update(kwargs)
        return RegionState(**defaults)

    def test_spare_capacity_over_100_rejected(self):
        with pytest.raises(ValueError, match="spare_capacity_pct"):
            self._region(spare_capacity_pct=101.0)

    def test_total_gpu_count_aggregation(self):
        node_a = NodeState(node_id="a", region="us-east", timestamp=NOW,
                           provenance=_prov(), gpu_capacity=8)
        node_b = NodeState(node_id="b", region="us-east", timestamp=NOW,
                           provenance=_prov(), gpu_capacity=4)
        r = self._region(nodes={"a": node_a, "b": node_b})
        assert r.total_gpu_count == 12

    def test_json_roundtrip(self):
        r = self._region(spare_capacity_pct=40.0)
        r2 = RegionState.from_dict(r.to_dict())
        assert r2.region == "us-east"
        assert r2.spare_capacity_pct == 40.0


# ---------------------------------------------------------------------------
# ClusterState
# ---------------------------------------------------------------------------

class TestClusterState:
    def _cluster(self, **kwargs) -> ClusterState:
        defaults = dict(timestamp=NOW, provenance=_prov())
        defaults.update(kwargs)
        return ClusterState(**defaults)

    def test_snapshot_id_generated(self):
        c = self._cluster()
        assert len(c.snapshot_id) == 36  # UUID4

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="UTC-aware"):
            self._cluster(timestamp=datetime(2024, 1, 1))

    def test_all_gpus_aggregation(self):
        gpu = _gpu()
        node = NodeState(node_id="n1", region="us-east", timestamp=NOW,
                         provenance=_prov(), gpus={gpu.gpu_uuid: gpu})
        region = RegionState(region="us-east", timestamp=NOW, provenance=_prov(),
                             nodes={"n1": node})
        cluster = self._cluster(regions={"us-east": region})
        all_gpus = cluster.all_gpus
        assert gpu.gpu_uuid in all_gpus

    def test_all_services_aggregation(self):
        svc = InferenceServiceState(service_id="svc-1", engine="vllm",
                                    timestamp=NOW, provenance=_prov())
        region = RegionState(region="us-east", timestamp=NOW, provenance=_prov(),
                             services={"svc-1": svc})
        cluster = self._cluster(regions={"us-east": region})
        assert "svc-1" in cluster.all_services

    def test_partial_flag(self):
        c = self._cluster(is_partial=True, missing_sources=["dcgm-exporter"])
        assert c.is_partial is True
        assert "dcgm-exporter" in c.missing_sources

    def test_json_roundtrip(self):
        c = self._cluster(is_partial=False, config_hash="abc123")
        d = c.to_dict()
        c2 = ClusterState.from_dict(d)
        assert c2.config_hash == "abc123"
        assert c2.is_partial is False
        assert c2.timestamp.tzinfo is not None

    def test_json_serializable(self):
        c = self._cluster()
        json.dumps(c.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# MigrationHistory
# ---------------------------------------------------------------------------

class TestMigrationHistory:
    def test_count_last_hour(self):
        prov = _prov()
        # prov.fetched_at = NOW
        recent = MigrationEvent(
            workload_id="w1", from_region="us-east", to_region="us-west",
            timestamp=NOW - timedelta(minutes=30), reason="energy"
        )
        old = MigrationEvent(
            workload_id="w1", from_region="us-west", to_region="us-east",
            timestamp=NOW - timedelta(hours=2), reason="energy"
        )
        hist = MigrationHistory(workload_id="w1", events=(recent, old), provenance=prov)
        assert hist.count_last_hour == 1
        assert hist.count_last_24h == 2

    def test_negative_cost_hours_rejected(self):
        with pytest.raises(ValueError, match="cost_hours"):
            MigrationEvent(
                workload_id="w1", from_region="a", to_region="b",
                timestamp=NOW, reason="x", cost_hours=-0.5,
            )

    def test_json_roundtrip(self):
        event = MigrationEvent(
            workload_id="w1", from_region="us-east", to_region="us-west",
            timestamp=NOW, reason="energy", cost_hours=0.5,
        )
        hist = MigrationHistory(workload_id="w1", events=(event,), provenance=_prov())
        d = hist.to_dict()
        hist2 = MigrationHistory.from_dict(d)
        assert hist2.workload_id == "w1"
        assert len(hist2.events) == 1
        assert hist2.events[0].cost_hours == 0.5


# ---------------------------------------------------------------------------
# ConstraintAssessment
# ---------------------------------------------------------------------------

class TestConstraintAssessment:
    def _assess(self, **kwargs):
        defaults = dict(
            timestamp=NOW,
            provenance=_prov(),
            region="us-east",
            scores={ConstraintType.ENERGY: 0.8, ConstraintType.THERMAL: 0.3},
            binding_constraint=ConstraintType.ENERGY,
            confidence=0.85,
            missing_signals=[],
            rationale="High energy price in region",
        )
        defaults.update(kwargs)
        return ConstraintAssessment(**defaults)

    def test_valid_assessment(self):
        a = self._assess()
        assert a.binding_constraint == ConstraintType.ENERGY
        assert a.confidence == 0.85

    def test_confidence_over_1_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            self._assess(confidence=1.1)

    def test_score_over_1_rejected(self):
        with pytest.raises(ValueError, match="scores"):
            self._assess(scores={ConstraintType.ENERGY: 1.5})

    def test_none_binding_constraint(self):
        a = self._assess(binding_constraint=None, confidence=0.2)
        assert a.binding_constraint is None

    def test_json_roundtrip(self):
        a = self._assess()
        d = a.to_dict()
        a2 = ConstraintAssessment.from_dict(d)
        assert a2.binding_constraint == ConstraintType.ENERGY
        assert a2.confidence == 0.85
        assert a2.timestamp.tzinfo is not None

    def test_constraint_type_enum_values(self):
        for ct in ConstraintType:
            assert isinstance(ct.value, str)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def _rec(self, **kwargs):
        defaults = dict(
            recommendation_id=str(uuid.uuid4()),
            workload_id="w-1",
            action_type="migrate_workload",
            timestamp=NOW,
            provenance=_prov(),
            binding_constraint=ConstraintType.ENERGY,
            confidence=0.7,
            sla_status="satisfied",
            rationale="Energy price spike in current region",
            is_noop=False,
        )
        defaults.update(kwargs)
        return Recommendation(**defaults)

    def test_valid_recommendation(self):
        r = self._rec()
        assert r.action_type == "migrate_workload"
        assert r.sla_status == "satisfied"
        assert r.implementation_mode == "recommendation_only"

    def test_invalid_sla_status_rejected(self):
        with pytest.raises(ValueError, match="sla_status"):
            self._rec(sla_status="maybe")

    def test_invalid_implementation_mode_rejected(self):
        with pytest.raises(ValueError, match="implementation_mode"):
            self._rec(implementation_mode="auto")

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            self._rec(confidence=-0.1)

    def test_negative_migration_penalty_rejected(self):
        with pytest.raises(ValueError, match="migration_penalty"):
            self._rec(migration_penalty=-100.0)

    def test_noop_recommendation(self):
        r = self._rec(is_noop=True, action_type="keep_current_placement",
                      sla_status="satisfied")
        assert r.is_noop is True

    def test_json_roundtrip(self):
        r = self._rec(
            expected_effect={"cost_delta": -50.0, "latency_delta_ms": 10.0},
            migration_penalty=20.0,
            net_benefit=-30.0,
        )
        d = r.to_dict()
        r2 = Recommendation.from_dict(d)
        assert r2.workload_id == r.workload_id
        assert r2.net_benefit == -30.0
        assert r2.migration_penalty == 20.0
        assert r2.timestamp.tzinfo is not None

    def test_json_serializable(self):
        r = self._rec()
        json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# TopologyLinkType ordering check
# ---------------------------------------------------------------------------

class TestTopologyLinkType:
    def test_all_link_types_are_strings(self):
        for lt in TopologyLinkType:
            assert isinstance(lt.value, str)

    def test_nvswitch_and_pix_exist(self):
        assert TopologyLinkType.NVSWITCH.value == "nvswitch"
        assert TopologyLinkType.PIX.value == "pix"
