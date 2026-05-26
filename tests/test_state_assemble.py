"""Tests for the connector → ClusterState assembler (Mission 1).

Proves the missing layer identified by the audit: connector leaf objects
(GPUState/InferenceServiceState/NodeState/TopologyState/EnergyState/ThermalState)
are aggregated into the single canonical ClusterState that the classifier and
engine consume — with honest partiality, staleness, and unknown-reference
handling, and no fabricated zeros.

Includes a real-adapter integration test: simulator DCGM/vLLM Prometheus text →
FakePrometheusClient → production DCGMAdapter/VLLMAdapter → build_cluster_state →
classifier → engine.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aurelius.constraints.classifier import ConstraintClassifier, ConstraintConfig
from aurelius.constraints.engine import ConstraintAwareEngine
from aurelius.state.assemble import (
    UNASSIGNED_REGION,
    build_cluster_state,
    build_cluster_state_from_connectors,
)
from aurelius.state.models import (
    ClusterState,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    ThermalState,
    TopologyState,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def _prov(source="dcgm-exporter", sandbox=False, age=None, confidence="high"):
    return Provenance(source=source, fetched_at=NOW, confidence=confidence,
                      is_sandbox=sandbox, sample_age_s=age)


def _gpu(uuid, node, region, temp=None, util=None, age=None, sandbox=False):
    return GPUState(gpu_uuid=uuid, node_id=node, region=region, timestamp=NOW,
                    provenance=_prov(age=age, sandbox=sandbox), gpu_index=0,
                    temp_c=temp, util_pct=util)


def _svc(sid, region, p99=None, ttft=None, qwait=None):
    return InferenceServiceState(service_id=sid, engine="vllm", timestamp=NOW,
                                 provenance=_prov(source="vllm"), region=region,
                                 p99_latency_ms=p99, ttft_p99_ms=ttft,
                                 queue_time_p95_ms=qwait)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_gpus_nest_into_region_and_node(self):
        gpus = [_gpu("GPU-1", "nodeA", "us-east", temp=80, util=70),
                _gpu("GPU-2", "nodeA", "us-east", temp=82, util=72),
                _gpu("GPU-3", "nodeB", "us-west", temp=60, util=40)]
        cs = build_cluster_state(timestamp=NOW, gpu_states=gpus)
        assert set(cs.regions) == {"us-east", "us-west"}
        assert set(cs.regions["us-east"].nodes["nodeA"].gpus) == {"GPU-1", "GPU-2"}
        assert set(cs.regions["us-west"].nodes["nodeB"].gpus) == {"GPU-3"}
        assert set(cs.all_gpus) == {"GPU-1", "GPU-2", "GPU-3"}

    def test_services_grouped_by_region(self):
        svcs = [_svc("svc1", "us-east"), _svc("svc2", "us-east"), _svc("svc3", "us-west")]
        cs = build_cluster_state(timestamp=NOW, inference_services=svcs)
        assert set(cs.regions["us-east"].services) == {"svc1", "svc2"}
        assert set(cs.regions["us-west"].services) == {"svc3"}

    def test_energy_thermal_topology_attached_per_region(self):
        energy = {"us-east": EnergyState(region="us-east", timestamp=NOW,
                                         provenance=_prov(), price_per_mwh=120.0)}
        thermal = {"us-east": ThermalState(region="us-east", timestamp=NOW,
                                           provenance=_prov(), max_gpu_temp_c=88.0,
                                           throttling_gpu_count=0, total_gpu_count=4)}
        gpus = [_gpu("GPU-1", "nodeA", "us-east")]
        topo = TopologyState(node_id="nodeA", timestamp=NOW, provenance=_prov(),
                             gpu_uuids=("GPU-1",), pair_levels={}, numa_affinity={},
                             interconnect_class="nvlink_full")
        cs = build_cluster_state(timestamp=NOW, gpu_states=gpus, energy_states=energy,
                                 thermal_states=thermal, topology_state=topo)
        r = cs.regions["us-east"]
        assert r.energy.price_per_mwh == 120.0
        assert r.thermal.max_gpu_temp_c == 88.0
        assert r.topology.interconnect_class == "nvlink_full"

    def test_node_states_merge_with_gpus(self):
        node = NodeState(node_id="nodeA", region="us-east", timestamp=NOW,
                         provenance=_prov(source="k8s"), gpu_capacity=8,
                         gpu_allocatable=8, gpu_allocated=2, gpus={})
        gpus = [_gpu("GPU-1", "nodeA", "us-east")]
        cs = build_cluster_state(timestamp=NOW, node_states=[node], gpu_states=gpus)
        merged = cs.regions["us-east"].nodes["nodeA"]
        assert merged.gpu_capacity == 8  # preserved from k8s node
        assert set(merged.gpus) == {"GPU-1"}  # gpu attached


# ---------------------------------------------------------------------------
# Partiality / missing sources / no fake zeros
# ---------------------------------------------------------------------------

class TestPartiality:
    def test_missing_source_sets_partial(self):
        cs = build_cluster_state(
            timestamp=NOW, gpu_states=[_gpu("GPU-1", "nodeA", "us-east")],
            source_metadata={"kubernetes": {"present": False, "error": "no_kubeconfig"}},
        )
        assert cs.is_partial is True
        assert any("kubernetes" in s for s in cs.missing_sources)

    def test_missing_dcgm_no_crash_gpu_fields_absent(self):
        # No gpu_states at all: node exists from k8s, but has no GPUs (not fake zeros)
        node = NodeState(node_id="nodeA", region="us-east", timestamp=NOW,
                         provenance=_prov(source="k8s"), gpu_capacity=8,
                         gpu_allocatable=8, gpu_allocated=0, gpus={})
        cs = build_cluster_state(timestamp=NOW, node_states=[node],
                                 source_metadata={"dcgm": {"present": False}})
        assert cs.regions["us-east"].nodes["nodeA"].gpus == {}
        assert cs.all_gpus == {}
        assert cs.is_partial is True

    def test_no_fabricated_zeros_for_missing_energy(self):
        cs = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("GPU-1", "nodeA", "us-east")])
        # energy was never provided → must be None, never 0.0
        assert cs.regions["us-east"].energy is None

    def test_missing_k8s_lowers_confidence(self):
        cs = build_cluster_state(
            timestamp=NOW, gpu_states=[_gpu("GPU-1", "nodeA", "us-east")],
            source_metadata={"kubernetes": {"present": False}},
        )
        assert cs.provenance.confidence in ("medium", "low")


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_stale_telemetry_lowers_confidence(self):
        fresh = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", "r", age=1.0)])
        stale = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", "r", age=9999.0)])
        assert fresh.provenance.confidence == "high"
        assert stale.provenance.confidence == "low"
        assert stale.regions["r"].provenance.sample_age_s == 9999.0

    def test_stale_via_source_metadata(self):
        cs = build_cluster_state(
            timestamp=NOW, gpu_states=[_gpu("G", "n", "r")],
            source_metadata={"prometheus": {"present": True, "stale": True}},
        )
        assert cs.provenance.confidence in ("medium", "low")


# ---------------------------------------------------------------------------
# Unknown references / region resolution
# ---------------------------------------------------------------------------

class TestUnknownReferences:
    def test_service_without_region_goes_unassigned_and_recorded(self):
        cs = build_cluster_state(timestamp=NOW, inference_services=[_svc("svc1", None)])
        assert UNASSIGNED_REGION in cs.regions
        assert cs.is_partial is True
        assert any("region=None" in s or "unassigned" in s for s in cs.missing_sources)

    def test_default_region_assignment(self):
        cs = build_cluster_state(timestamp=NOW, inference_services=[_svc("svc1", None)],
                                 default_region="us-east")
        assert "svc1" in cs.regions["us-east"].services
        assert UNASSIGNED_REGION not in cs.regions

    def test_gpu_without_region_recorded(self):
        cs = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", None)])
        assert cs.is_partial is True
        assert any("region=None" in s for s in cs.missing_sources)

    def test_services_without_nodes_flagged(self):
        cs = build_cluster_state(timestamp=NOW, inference_services=[_svc("svc1", "us-east")])
        assert any("services_without_nodes" in s for s in cs.missing_sources)


# ---------------------------------------------------------------------------
# NaN / inf handling
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_nan_spare_capacity_becomes_none(self):
        cs = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", "us-east")],
                                 workload_states={"us-east": float("nan")})
        assert cs.regions["us-east"].spare_capacity_pct is None

    def test_inf_spare_capacity_becomes_none(self):
        cs = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", "us-east")],
                                 workload_states={"us-east": float("inf")})
        assert cs.regions["us-east"].spare_capacity_pct is None


# ---------------------------------------------------------------------------
# Sandbox propagation
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_sandbox_propagates_from_leaf(self):
        cs = build_cluster_state(timestamp=NOW,
                                 gpu_states=[_gpu("G", "n", "r", sandbox=True)])
        assert cs.provenance.is_sandbox is True

    def test_non_sandbox_when_all_real(self):
        cs = build_cluster_state(timestamp=NOW, gpu_states=[_gpu("G", "n", "r")])
        assert cs.provenance.is_sandbox is False


# ---------------------------------------------------------------------------
# End-to-end: assembled state drives classifier and engine
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _thermal_state(self):
        gpus = [_gpu("GPU-1", "nodeA", "us-east", temp=93, util=90),
                _gpu("GPU-2", "nodeA", "us-east", temp=94, util=92)]
        thermal = {"us-east": ThermalState(region="us-east", timestamp=NOW,
                                            provenance=_prov(), max_gpu_temp_c=94.0,
                                            throttling_gpu_count=2, total_gpu_count=2)}
        svcs = [_svc("llm-svc", "us-east", p99=1500, ttft=900)]
        return build_cluster_state(timestamp=NOW, gpu_states=gpus, thermal_states=thermal,
                                   inference_services=svcs)

    def test_assembled_state_drives_classifier(self):
        cs = self._thermal_state()
        clf = ConstraintClassifier(ConstraintConfig(hysteresis_count=1))
        a = clf.assess(cs)
        from aurelius.state.models import ConstraintType
        assert a.binding_constraint == ConstraintType.THERMAL
        assert a.scores[ConstraintType.THERMAL] > 0.5

    def test_assembled_state_drives_engine(self):
        cs = self._thermal_state()
        eng = ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))
        result = eng.run(cs)
        assert len(result.recommendations) == 1
        # recommendation_only mode preserved
        assert all(r.implementation_mode == "recommendation_only"
                   for r in result.recommendations)


# ---------------------------------------------------------------------------
# Integration: real production adapters → assembler → classifier → engine
# ---------------------------------------------------------------------------

class TestRealAdapterIntegration:
    @pytest.fixture
    def sim(self):
        from aurelius.simulation.cluster.engine import ClusterSimulator
        from aurelius.simulation.cluster.scenarios import load_scenario
        sc = load_scenario("thermal_hotspot_mixed_cluster")
        s = ClusterSimulator(sc.config, seed=42)
        s.run_metrics_only(8)
        return s

    def test_dcgm_vllm_text_to_clusterstate(self, sim):
        from aurelius.connectors.dcgm import DCGMAdapter
        from aurelius.connectors.metric_mapping import dcgm_registry, vllm_registry
        from aurelius.connectors.prometheus import FakePrometheusClient
        from aurelius.connectors.vllm import VLLMAdapter

        # DCGM path
        dcgm_text = sim.get_dcgm_prometheus_text("hot-node0")
        dclient = FakePrometheusClient(prometheus_text=dcgm_text)
        dsnap = dclient.fetch_snapshot(dcgm_registry(), source="sim-dcgm")
        gpus = DCGMAdapter(dclient).normalize_gpus(dsnap, node_id="hot-node0",
                                                   region="us-east", timestamp=NOW)
        assert gpus, "DCGM adapter should produce GPU leaf objects from sim text"

        # vLLM path (services carry no region → use default_region)
        vllm_text = sim.get_vllm_prometheus_text("llm-inference")
        vclient = FakePrometheusClient(prometheus_text=vllm_text)
        vsnap = vclient.fetch_snapshot(vllm_registry(), source="sim-vllm")
        svcs = VLLMAdapter(vclient).normalize_all_services(vsnap,
                                                           service_id_prefix="llm-inference",
                                                           timestamp=NOW)

        cs = build_cluster_state(timestamp=NOW, gpu_states=gpus, inference_services=svcs,
                                 default_region="us-east", prometheus_snapshot=dsnap)
        # Real adapter output assembled into ONE ClusterState
        assert "us-east" in cs.regions
        assert len(cs.all_gpus) == len(gpus)
        # Drives the classifier + engine without a simulator-built ClusterState
        eng = ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))
        result = eng.run(cs)
        assert isinstance(cs, ClusterState)
        assert result.assessment is not None

    def test_build_from_connectors_marks_failed_connector_missing(self):
        # A connector that raises must be captured as a missing source, not crash.
        class _Boom:
            def fetch_snapshot(self):
                raise RuntimeError("network down")

        cs = build_cluster_state_from_connectors(
            config={}, connectors={"prometheus": _Boom()}, timestamp=NOW,
        )
        assert cs.is_partial is True
        assert any("prometheus" in s for s in cs.missing_sources)

    def test_build_from_connectors_assembles_dcgm(self, sim):
        from aurelius.connectors.dcgm import DCGMAdapter
        from aurelius.connectors.metric_mapping import dcgm_registry
        from aurelius.connectors.prometheus import (
            FakePrometheusClient,
            PrometheusTelemetryConnector,
        )

        dcgm_text = sim.get_dcgm_prometheus_text("hot-node0")
        client = FakePrometheusClient(prometheus_text=dcgm_text)
        prom = PrometheusTelemetryConnector(client, dcgm_registry(), source="sim-dcgm")

        class _DCGMWrap:
            def __init__(self, c):
                self._a = DCGMAdapter(c)
            def normalize_gpus(self, snapshot):
                return self._a.normalize_gpus(snapshot, node_id="hot-node0",
                                              region="us-east", timestamp=NOW)

        cs = build_cluster_state_from_connectors(
            config={"default_region": "us-east"},
            connectors={"prometheus": prom, "dcgm": _DCGMWrap(client)},
            timestamp=NOW,
        )
        assert "us-east" in cs.regions
        assert cs.all_gpus, "from_connectors should assemble DCGM GPUs into ClusterState"
        assert cs.provenance.is_sandbox is True  # fixture-backed
