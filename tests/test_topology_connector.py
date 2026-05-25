"""Tests for Aurelius Phase 5: GPU topology collector and placement scorer.

Coverage:
- parse_nvidia_smi_topo with DGX H100 NVSwitch fixture (NV18 → NVSWITCH)
- parse_nvidia_smi_topo with PCIe dual-NUMA fixture
- NUMA affinity extraction from topo output
- parse_nvidia_smi_list inventory parsing
- build_topology_state UUID translation
- interconnect_class derivation
- NVLink/NVSwitch placement scores better than PCIe
- same-NUMA placement scores better than cross-NUMA
- communication-heavy workloads penalize bad topology more than batch
- FakeTopologyCollector uses identical parse paths
- missing topology returns conservative 0.5 score (not 0)
- NV18 token maps to NVSWITCH enum
- rank_placements orders candidates correctly
- latency-sensitive flag increases penalty weight
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from aurelius.connectors.topology import (
    FakeTopologyCollector,
    NvidiaSmiTopologyCollector,
    PlacementScore,
    PlacementWorkloadSpec,
    _derive_interconnect_class,
    _parse_link_token,
    build_topology_state,
    parse_nvidia_smi_list,
    parse_nvidia_smi_topo,
    rank_placements,
    score_placement,
)
from aurelius.state.models import Provenance, TopologyLinkType, TopologyState

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "topology"


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


@pytest.fixture
def dgx_topo_text():
    return read_fixture("dgx_h100_8gpu_nvswitch.txt")


@pytest.fixture
def dgx_inventory_text():
    return read_fixture("dgx_h100_inventory.txt")


@pytest.fixture
def pcie_topo_text():
    return read_fixture("pcie_8gpu_dual_numa.txt")


@pytest.fixture
def pcie_inventory_text():
    return read_fixture("pcie_8gpu_inventory.txt")


@pytest.fixture
def ts():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def sandbox_prov(ts):
    return Provenance(
        source="test",
        fetched_at=ts,
        confidence="medium",
        is_sandbox=True,
    )


# ---------------------------------------------------------------------------
# _parse_link_token
# ---------------------------------------------------------------------------

class TestParseLinkToken:
    def test_x_returns_none(self):
        assert _parse_link_token("X") is None
        assert _parse_link_token(" X ") is None  # with spaces: token is pre-stripped

    def test_nv18_maps_to_nvswitch(self):
        assert _parse_link_token("NV18") == TopologyLinkType.NVSWITCH

    def test_nv4_maps_to_nv4(self):
        assert _parse_link_token("NV4") == TopologyLinkType.NV4

    def test_nv3_maps_to_nv3(self):
        assert _parse_link_token("NV3") == TopologyLinkType.NV3

    def test_nv2_maps_to_nv2(self):
        assert _parse_link_token("NV2") == TopologyLinkType.NV2

    def test_nv1_maps_to_nv1(self):
        assert _parse_link_token("NV1") == TopologyLinkType.NV1

    def test_nv5_to_nvswitch(self):
        assert _parse_link_token("NV5") == TopologyLinkType.NVSWITCH

    def test_pix(self):
        assert _parse_link_token("PIX") == TopologyLinkType.PIX

    def test_pxb(self):
        assert _parse_link_token("PXB") == TopologyLinkType.PXB

    def test_phb(self):
        assert _parse_link_token("PHB") == TopologyLinkType.PHB

    def test_node(self):
        assert _parse_link_token("NODE") == TopologyLinkType.NODE

    def test_sys(self):
        assert _parse_link_token("SYS") == TopologyLinkType.SYS

    def test_unknown_returns_none(self):
        assert _parse_link_token("UNKNOWN") is None
        assert _parse_link_token("") is None


# ---------------------------------------------------------------------------
# parse_nvidia_smi_topo — DGX H100 NVSwitch
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiTopoNVSwitch:
    def test_returns_8_gpus(self, dgx_topo_text):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        assert len(gpu_ids) == 8

    def test_gpu_ids_ordered(self, dgx_topo_text):
        gpu_ids, _, _ = parse_nvidia_smi_topo(dgx_topo_text)
        assert gpu_ids == ["GPU0", "GPU1", "GPU2", "GPU3", "GPU4", "GPU5", "GPU6", "GPU7"]

    def test_all_pairs_nvswitch(self, dgx_topo_text):
        _, pair_levels, _ = parse_nvidia_smi_topo(dgx_topo_text)
        n = 8
        expected_pairs = n * (n - 1) // 2  # 28
        assert len(pair_levels) == expected_pairs
        for link in pair_levels.values():
            assert link == TopologyLinkType.NVSWITCH, f"Expected NVSWITCH, got {link}"

    def test_pair_keys_normalized_min_max(self, dgx_topo_text):
        _, pair_levels, _ = parse_nvidia_smi_topo(dgx_topo_text)
        for (a, b) in pair_levels:
            assert a <= b, f"Pair key not normalized: ({a}, {b})"

    def test_numa_affinity_two_groups(self, dgx_topo_text):
        _, _, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        # GPU0-GPU3 → NUMA 0, GPU4-GPU7 → NUMA 1
        for gpu in ["GPU0", "GPU1", "GPU2", "GPU3"]:
            assert numa_affinity.get(gpu) == 0, f"{gpu} should be NUMA 0"
        for gpu in ["GPU4", "GPU5", "GPU6", "GPU7"]:
            assert numa_affinity.get(gpu) == 1, f"{gpu} should be NUMA 1"

    def test_interconnect_class_nvlink_full(self, dgx_topo_text):
        gpu_ids, pair_levels, _ = parse_nvidia_smi_topo(dgx_topo_text)
        ic = _derive_interconnect_class(gpu_ids, pair_levels)
        assert ic == "nvlink_full"


# ---------------------------------------------------------------------------
# parse_nvidia_smi_topo — PCIe dual-NUMA
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiTopoPCIe:
    def test_returns_8_gpus(self, pcie_topo_text):
        gpu_ids, _, _ = parse_nvidia_smi_topo(pcie_topo_text)
        assert len(gpu_ids) == 8

    def test_same_numa_pairs_are_pix_or_pxb(self, pcie_topo_text):
        _, pair_levels, _ = parse_nvidia_smi_topo(pcie_topo_text)
        # GPU0-GPU1: PIX, GPU0-GPU2: PXB, etc.
        key01 = TopologyState.make_pair_key("GPU0", "GPU1")
        assert pair_levels[key01] == TopologyLinkType.PIX

        key02 = TopologyState.make_pair_key("GPU0", "GPU2")
        assert pair_levels[key02] == TopologyLinkType.PXB

    def test_cross_numa_pairs_are_sys(self, pcie_topo_text):
        _, pair_levels, _ = parse_nvidia_smi_topo(pcie_topo_text)
        key04 = TopologyState.make_pair_key("GPU0", "GPU4")
        assert pair_levels[key04] == TopologyLinkType.SYS

    def test_numa_affinity_two_groups(self, pcie_topo_text):
        _, _, numa_affinity = parse_nvidia_smi_topo(pcie_topo_text)
        for gpu in ["GPU0", "GPU1", "GPU2", "GPU3"]:
            assert numa_affinity.get(gpu) == 0
        for gpu in ["GPU4", "GPU5", "GPU6", "GPU7"]:
            assert numa_affinity.get(gpu) == 1

    def test_interconnect_class_cross_numa(self, pcie_topo_text):
        gpu_ids, pair_levels, _ = parse_nvidia_smi_topo(pcie_topo_text)
        ic = _derive_interconnect_class(gpu_ids, pair_levels)
        assert ic == "cross_numa"

    def test_no_nvlink_in_pcie_topology(self, pcie_topo_text):
        _, pair_levels, _ = parse_nvidia_smi_topo(pcie_topo_text)
        nv_types = {TopologyLinkType.NV1, TopologyLinkType.NV2, TopologyLinkType.NV3,
                    TopologyLinkType.NV4, TopologyLinkType.NVSWITCH}
        for link in pair_levels.values():
            assert link not in nv_types, f"Found unexpected NVLink type: {link}"


# ---------------------------------------------------------------------------
# parse_nvidia_smi_list
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiList:
    def test_parses_8_gpus(self, dgx_inventory_text):
        gpus = parse_nvidia_smi_list(dgx_inventory_text)
        assert len(gpus) == 8

    def test_gpu_fields_present(self, dgx_inventory_text):
        gpus = parse_nvidia_smi_list(dgx_inventory_text)
        for g in gpus:
            assert "index" in g
            assert "name" in g
            assert "uuid" in g
            assert g["uuid"].startswith("GPU-")

    def test_index_values(self, dgx_inventory_text):
        gpus = parse_nvidia_smi_list(dgx_inventory_text)
        indices = [g["index"] for g in gpus]
        assert indices == ["0", "1", "2", "3", "4", "5", "6", "7"]

    def test_model_name_preserved(self, dgx_inventory_text):
        gpus = parse_nvidia_smi_list(dgx_inventory_text)
        assert gpus[0]["name"] == "NVIDIA H100 SXM5 80GB"

    def test_empty_text_returns_empty_list(self):
        assert parse_nvidia_smi_list("") == []

    def test_malformed_lines_skipped(self):
        text = "Not a GPU line\nGPU 0: Valid GPU (UUID: GPU-abc123)\nGarbage line"
        gpus = parse_nvidia_smi_list(text)
        assert len(gpus) == 1


# ---------------------------------------------------------------------------
# build_topology_state
# ---------------------------------------------------------------------------

class TestBuildTopologyState:
    def test_nvswitch_topology_state(self, dgx_topo_text, dgx_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        inventory = parse_nvidia_smi_list(dgx_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        topo = build_topology_state(
            "dgx-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

        assert topo.node_id == "dgx-01"
        assert len(topo.gpu_uuids) == 8
        assert topo.interconnect_class == "nvlink_full"
        assert topo.nvlink_present is True

    def test_uuid_translation(self, dgx_topo_text, dgx_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        inventory = parse_nvidia_smi_list(dgx_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        topo = build_topology_state(
            "dgx-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

        # Keys in pair_levels should be UUIDs, not GPU0/GPU1
        for (a, b) in topo.pair_levels:
            assert a.startswith("GPU-"), f"Expected UUID key, got {a}"
            assert b.startswith("GPU-"), f"Expected UUID key, got {b}"

    def test_fallback_to_logical_id_without_uuid_map(self, dgx_topo_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        topo = build_topology_state(
            "node-01", gpu_ids, {}, pair_levels, numa_affinity, ts, sandbox_prov
        )
        # Without UUID map, logical IDs (GPU0, GPU1) are used as-is
        for (a, b) in topo.pair_levels:
            assert a.startswith("GPU"), f"Expected logical GPU ID, got {a}"

    def test_numa_affinity_preserved(self, dgx_topo_text, dgx_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        inventory = parse_nvidia_smi_list(dgx_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        topo = build_topology_state(
            "dgx-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

        # GPU0 → UUID → NUMA 0
        gpu0_uuid = uuid_map["GPU0"]
        assert topo.numa_affinity.get(gpu0_uuid) == 0

        gpu4_uuid = uuid_map["GPU4"]
        assert topo.numa_affinity.get(gpu4_uuid) == 1

    def test_pcie_topology_state(self, pcie_topo_text, pcie_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(pcie_topo_text)
        inventory = parse_nvidia_smi_list(pcie_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        topo = build_topology_state(
            "pcie-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

        assert topo.interconnect_class == "cross_numa"
        assert topo.nvlink_present is False


# ---------------------------------------------------------------------------
# _derive_interconnect_class
# ---------------------------------------------------------------------------

class TestDeriveInterconnectClass:
    def test_all_nvswitch_is_nvlink_full(self):
        gpu_ids = ["GPU0", "GPU1", "GPU2"]
        pair_levels = {
            ("GPU0", "GPU1"): TopologyLinkType.NVSWITCH,
            ("GPU0", "GPU2"): TopologyLinkType.NVSWITCH,
            ("GPU1", "GPU2"): TopologyLinkType.NVSWITCH,
        }
        assert _derive_interconnect_class(gpu_ids, pair_levels) == "nvlink_full"

    def test_partial_nvlink_is_nvlink_partial(self):
        gpu_ids = ["GPU0", "GPU1", "GPU2"]
        pair_levels = {
            ("GPU0", "GPU1"): TopologyLinkType.NVSWITCH,
            ("GPU0", "GPU2"): TopologyLinkType.PIX,
            ("GPU1", "GPU2"): TopologyLinkType.PIX,
        }
        assert _derive_interconnect_class(gpu_ids, pair_levels) == "nvlink_partial"

    def test_all_pcie_is_pcie(self):
        gpu_ids = ["GPU0", "GPU1"]
        pair_levels = {("GPU0", "GPU1"): TopologyLinkType.PIX}
        assert _derive_interconnect_class(gpu_ids, pair_levels) == "pcie"

    def test_cross_numa_has_sys(self):
        gpu_ids = ["GPU0", "GPU1"]
        pair_levels = {("GPU0", "GPU1"): TopologyLinkType.SYS}
        assert _derive_interconnect_class(gpu_ids, pair_levels) == "cross_numa"

    def test_empty_returns_unknown(self):
        assert _derive_interconnect_class([], {}) == "unknown"
        assert _derive_interconnect_class(["GPU0"], {}) == "unknown"


# ---------------------------------------------------------------------------
# score_placement
# ---------------------------------------------------------------------------

class TestScorePlacement:
    @pytest.fixture
    def nvswitch_topology(self, dgx_topo_text, dgx_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(dgx_topo_text)
        inventory = parse_nvidia_smi_list(dgx_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}
        return build_topology_state(
            "dgx-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

    @pytest.fixture
    def pcie_topology(self, pcie_topo_text, pcie_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(pcie_topo_text)
        inventory = parse_nvidia_smi_list(pcie_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}
        return build_topology_state(
            "pcie-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov
        )

    def test_single_gpu_is_zero_score(self, nvswitch_topology):
        workload = PlacementWorkloadSpec(gpu_count=1)
        uuids = list(nvswitch_topology.gpu_uuids[:1])
        result = score_placement(workload, uuids, nvswitch_topology)
        assert result.score == 0.0

    def test_nvswitch_better_than_pcie(self, nvswitch_topology, pcie_topology):
        workload = PlacementWorkloadSpec(gpu_count=4, communication_intensity="high")
        nvswitch_uuids = list(nvswitch_topology.gpu_uuids[:4])
        pcie_uuids = list(pcie_topology.gpu_uuids[:4])

        nvswitch_score = score_placement(workload, nvswitch_uuids, nvswitch_topology)
        pcie_score = score_placement(workload, pcie_uuids, pcie_topology)

        assert nvswitch_score.score < pcie_score.score, (
            f"NVSwitch ({nvswitch_score.score:.3f}) should beat PCIe ({pcie_score.score:.3f})"
        )

    def test_same_numa_pcie_better_than_cross_numa(self, pcie_topology):
        workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="medium")
        uuids = list(pcie_topology.gpu_uuids)
        # Same NUMA: GPU0 (idx 0) and GPU1 (idx 1) — PIX link
        same_numa = [uuids[0], uuids[1]]
        # Cross NUMA: GPU0 (idx 0) and GPU4 (idx 4) — SYS link
        cross_numa = [uuids[0], uuids[4]]

        same_score = score_placement(workload, same_numa, pcie_topology)
        cross_score = score_placement(workload, cross_numa, pcie_topology)

        assert same_score.score < cross_score.score, (
            f"Same NUMA ({same_score.score:.3f}) should beat cross NUMA ({cross_score.score:.3f})"
        )

    def test_high_comm_penalizes_more_than_low_comm(self, pcie_topology):
        uuids = list(pcie_topology.gpu_uuids)
        cross_numa_uuids = [uuids[0], uuids[4]]  # SYS link

        low_comm = PlacementWorkloadSpec(gpu_count=2, communication_intensity="low")
        high_comm = PlacementWorkloadSpec(gpu_count=2, communication_intensity="high")

        low_score = score_placement(low_comm, cross_numa_uuids, pcie_topology)
        high_score = score_placement(high_comm, cross_numa_uuids, pcie_topology)

        assert high_score.score > low_score.score, (
            f"High comm ({high_score.score:.3f}) should score worse than low comm ({low_score.score:.3f})"
        )

    def test_latency_sensitive_increases_penalty(self, pcie_topology):
        uuids = list(pcie_topology.gpu_uuids)
        cross_numa_uuids = [uuids[0], uuids[4]]

        base_workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="medium", latency_sensitive=False)
        latency_workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="medium", latency_sensitive=True)

        base_score = score_placement(base_workload, cross_numa_uuids, pcie_topology)
        latency_score = score_placement(latency_workload, cross_numa_uuids, pcie_topology)

        assert latency_score.score > base_score.score, (
            f"Latency-sensitive ({latency_score.score:.3f}) should score worse than base ({base_score.score:.3f})"
        )

    def test_no_topology_returns_conservative_midrange(self):
        workload = PlacementWorkloadSpec(gpu_count=4, communication_intensity="high")
        uuids = ["GPU-fake-0001", "GPU-fake-0002", "GPU-fake-0003", "GPU-fake-0004"]
        result = score_placement(workload, uuids, None)
        # Must not return 0.0 (fabricated certainty) or 1.0 (fabricated failure)
        assert 0.0 < result.score < 1.0, f"Expected conservative mid-range, got {result.score}"
        assert result.score == 0.5

    def test_unknown_gpu_uuids_returns_conservative(self, nvswitch_topology):
        workload = PlacementWorkloadSpec(gpu_count=2)
        fake_uuids = ["GPU-unknown-0001", "GPU-unknown-0002"]
        result = score_placement(workload, fake_uuids, nvswitch_topology)
        # GPU UUIDs not in topology → conservative 0.5
        assert result.score == 0.5

    def test_score_clamped_to_one(self, pcie_topology):
        uuids = list(pcie_topology.gpu_uuids)
        # High comm + latency sensitive + worst cross-numa link
        extreme_workload = PlacementWorkloadSpec(
            gpu_count=2, communication_intensity="high", latency_sensitive=True
        )
        cross_numa = [uuids[0], uuids[4]]
        result = score_placement(extreme_workload, cross_numa, pcie_topology)
        assert 0.0 <= result.score <= 1.0

    def test_nvswitch_all_pairs_score_zero(self, nvswitch_topology):
        workload = PlacementWorkloadSpec(gpu_count=8, communication_intensity="high", latency_sensitive=True)
        all_uuids = list(nvswitch_topology.gpu_uuids)
        result = score_placement(workload, all_uuids, nvswitch_topology)
        assert result.score == 0.0
        assert result.worst_link == TopologyLinkType.NVSWITCH

    def test_explanation_contains_relevant_info(self, pcie_topology):
        uuids = list(pcie_topology.gpu_uuids)
        workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="high")
        result = score_placement(workload, [uuids[0], uuids[4]], pcie_topology)
        assert result.explanation  # non-empty
        assert result.worst_link is not None


# ---------------------------------------------------------------------------
# rank_placements
# ---------------------------------------------------------------------------

class TestRankPlacements:
    def test_nvswitch_ranks_first_over_pcie(
        self, dgx_topo_text, dgx_inventory_text, pcie_topo_text, pcie_inventory_text, ts, sandbox_prov
    ):
        gpu_ids_nv, pl_nv, na_nv = parse_nvidia_smi_topo(dgx_topo_text)
        inv_nv = parse_nvidia_smi_list(dgx_inventory_text)
        uuid_map_nv = {f"GPU{g['index']}": g["uuid"] for g in inv_nv}
        topo_nv = build_topology_state("dgx-01", gpu_ids_nv, uuid_map_nv, pl_nv, na_nv, ts, sandbox_prov)

        gpu_ids_p, pl_p, na_p = parse_nvidia_smi_topo(pcie_topo_text)
        inv_p = parse_nvidia_smi_list(pcie_inventory_text)
        uuid_map_p = {f"GPU{g['index']}": g["uuid"] for g in inv_p}
        topo_p = build_topology_state("pcie-01", gpu_ids_p, uuid_map_p, pl_p, na_p, ts, sandbox_prov)

        # Candidate 1: NVSwitch GPUs on dgx-01 → use NVSwitch topology
        nv_candidate = list(topo_nv.gpu_uuids[:4])
        # Candidate 2: Cross-NUMA PCIe GPUs on pcie-01
        pcie_candidate = [list(topo_p.gpu_uuids)[0], list(topo_p.gpu_uuids)[4]]

        workload = PlacementWorkloadSpec(gpu_count=4, communication_intensity="high")

        # Score each against its own topology
        nv_score = score_placement(workload, nv_candidate, topo_nv)
        pcie_score = score_placement(workload, pcie_candidate, topo_p)

        assert nv_score.score < pcie_score.score

    def test_rank_returns_sorted_list(self, pcie_topo_text, pcie_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(pcie_topo_text)
        inventory = parse_nvidia_smi_list(pcie_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}
        topo = build_topology_state("pcie-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov)

        uuids = list(topo.gpu_uuids)
        workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="high")

        candidates = [
            [uuids[0], uuids[4]],  # cross-NUMA (SYS) — worst
            [uuids[0], uuids[1]],  # same NUMA (PIX) — best
            [uuids[0], uuids[2]],  # same NUMA (PXB) — medium
        ]

        ranked = rank_placements(workload, candidates, topo)
        scores = [ps.score for _, ps in ranked]
        assert scores == sorted(scores), "Ranked list must be sorted lowest-score first"

    def test_rank_best_is_same_numa(self, pcie_topo_text, pcie_inventory_text, ts, sandbox_prov):
        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(pcie_topo_text)
        inventory = parse_nvidia_smi_list(pcie_inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}
        topo = build_topology_state("pcie-01", gpu_ids, uuid_map, pair_levels, numa_affinity, ts, sandbox_prov)

        uuids = list(topo.gpu_uuids)
        workload = PlacementWorkloadSpec(gpu_count=2, communication_intensity="medium")

        candidates = [
            [uuids[0], uuids[4]],  # cross-NUMA (SYS)
            [uuids[0], uuids[1]],  # same NUMA, PIX
        ]

        ranked = rank_placements(workload, candidates, topo)
        best_candidate = ranked[0][0]
        # Best should be same NUMA (PIX), not cross-NUMA (SYS)
        assert uuids[0] in best_candidate and uuids[1] in best_candidate


# ---------------------------------------------------------------------------
# FakeTopologyCollector
# ---------------------------------------------------------------------------

class TestFakeTopologyCollector:
    def test_from_fixture_text(self, dgx_topo_text, dgx_inventory_text):
        collector = FakeTopologyCollector(
            node_id="dgx-01",
            topo_text=dgx_topo_text,
            inventory_text=dgx_inventory_text,
        )
        topo = collector.collect()
        assert topo is not None
        assert topo.node_id == "dgx-01"
        assert topo.interconnect_class == "nvlink_full"
        assert len(topo.gpu_uuids) == 8

    def test_from_prebuilt_topology_state(self, ts, sandbox_prov):
        prebuilt = TopologyState(
            node_id="test-node",
            timestamp=ts,
            provenance=sandbox_prov,
            gpu_uuids=("GPU-0001", "GPU-0002"),
            pair_levels={("GPU-0001", "GPU-0002"): TopologyLinkType.NVSWITCH},
            numa_affinity={"GPU-0001": 0, "GPU-0002": 0},
            interconnect_class="nvlink_full",
        )
        collector = FakeTopologyCollector(node_id="test-node", topology_state=prebuilt)
        topo = collector.collect()
        assert topo is prebuilt

    def test_no_text_no_state_returns_none(self):
        collector = FakeTopologyCollector(node_id="empty-node")
        topo = collector.collect()
        assert topo is None

    def test_is_sandbox_true(self, dgx_topo_text):
        collector = FakeTopologyCollector(node_id="dgx-01", topo_text=dgx_topo_text)
        topo = collector.collect()
        assert topo is not None
        assert topo.provenance.is_sandbox is True

    def test_same_parse_paths_as_real_collector(self, dgx_topo_text, dgx_inventory_text):
        # Verify parse results match what direct parse_nvidia_smi_topo returns
        fake = FakeTopologyCollector(
            node_id="dgx-01",
            topo_text=dgx_topo_text,
            inventory_text=dgx_inventory_text,
        )
        topo = fake.collect()
        gpu_ids, pair_levels, _ = parse_nvidia_smi_topo(dgx_topo_text)
        assert topo is not None
        assert len(topo.pair_levels) == len(pair_levels)

    def test_pcie_fixture(self, pcie_topo_text, pcie_inventory_text):
        collector = FakeTopologyCollector(
            node_id="pcie-01",
            topo_text=pcie_topo_text,
            inventory_text=pcie_inventory_text,
        )
        topo = collector.collect()
        assert topo is not None
        assert topo.interconnect_class == "cross_numa"
        assert topo.nvlink_present is False


# ---------------------------------------------------------------------------
# NvidiaSmiTopologyCollector (real collector — no GPU required)
# ---------------------------------------------------------------------------

class TestNvidiaSmiTopologyCollectorSafe:
    def test_returns_none_gracefully_when_nvidia_smi_missing(self, monkeypatch):
        """Collector must return None safely when nvidia-smi is not available."""
        import subprocess

        def mock_check_output(cmd, *args, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        monkeypatch.setattr(subprocess, "check_output", mock_check_output)

        collector = NvidiaSmiTopologyCollector(node_id="test-node")
        result = collector.collect()
        assert result is None

    def test_returns_none_gracefully_on_timeout(self, monkeypatch):
        import subprocess

        def mock_check_output(cmd, *args, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=10)

        monkeypatch.setattr(subprocess, "check_output", mock_check_output)

        collector = NvidiaSmiTopologyCollector(node_id="test-node")
        result = collector.collect()
        assert result is None
