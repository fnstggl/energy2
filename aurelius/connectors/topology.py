"""Topology collector and placement scorer for Aurelius.

Phase 5: GPU topology collection from nvidia-smi topo -m, nvidia-smi -L,
and optional NVML. Produces TopologyState consumed by the constraint
classifier and PlacementScorer.

Design:
- Real collector requires node-local shell access or NVML
- FakeTopologyCollector accepts text fixtures and uses the same parse paths
- PlacementScorer is pure function: topology + workload spec → score
- No NCCL, CUDA, or kernel internals are touched

Allowed:
- Parse topology text / NVML data
- Score GPU placement candidates
- Recommend topology-aware placement

Forbidden:
- Modifying NCCL, CUDA, or any runtime
- Reaching into KV cache or memory allocator
- Mutating cluster state
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..state.models import Provenance, TopologyLinkType, TopologyState

logger = logging.getLogger(__name__)

# Lazy NVML guard — nvidia-ml-py package, imports as pynvml
_NVML_AVAILABLE = False
try:
    import pynvml as _pynvml  # type: ignore  # noqa: F401
    _NVML_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# nvidia-smi topo -m parser
# ---------------------------------------------------------------------------

# Exact token → TopologyLinkType mapping from the nvidia-smi legend
_TOPO_TOKEN_MAP: dict[str, TopologyLinkType] = {
    "PIX": TopologyLinkType.PIX,
    "PXB": TopologyLinkType.PXB,
    "PHB": TopologyLinkType.PHB,
    "NODE": TopologyLinkType.NODE,
    "SYS": TopologyLinkType.SYS,
    "NV1": TopologyLinkType.NV1,
    "NV2": TopologyLinkType.NV2,
    "NV3": TopologyLinkType.NV3,
    "NV4": TopologyLinkType.NV4,
}

# NVLink bond counts that map to a specific enum value
_NV_COUNT_MAP: dict[int, TopologyLinkType] = {
    1: TopologyLinkType.NV1,
    2: TopologyLinkType.NV2,
    3: TopologyLinkType.NV3,
    4: TopologyLinkType.NV4,
}


def _parse_link_token(token: str) -> Optional[TopologyLinkType]:
    """Map one nvidia-smi topo -m cell to a TopologyLinkType.

    'X' = self-link (return None).
    NV1–NV4 → NV1–NV4. NV5+ → NVSWITCH (full NVSwitch fabric, e.g. NV18 on H100).
    """
    if token == "X":
        return None  # self-link; skip

    if token in _TOPO_TOKEN_MAP:
        return _TOPO_TOKEN_MAP[token]

    m = re.match(r"^NV(\d+)$", token)
    if m:
        n = int(m.group(1))
        return _NV_COUNT_MAP.get(n, TopologyLinkType.NVSWITCH)

    return None


def parse_nvidia_smi_topo(
    text: str,
) -> tuple[list[str], dict[tuple[str, str], TopologyLinkType], dict[str, int]]:
    """Parse nvidia-smi topo -m text output.

    The output has a header row and data rows. Example format:
        GPU0    GPU1    GPU2    NIC0    CPU Affinity    NUMA Affinity
        GPU0     X      NV18    SYS     0-47            0
        GPU1    NV18     X      SYS     0-47            0

    Returns:
        gpu_ids:       list of GPU column identifiers (GPU0, GPU1, …)
        pair_levels:   {(id_a, id_b): TopologyLinkType} with min-max key order
        numa_affinity: {gpu_id: NUMA node int}
    """
    lines = [line.rstrip() for line in text.splitlines()]
    header_idx: Optional[int] = None
    header_gpu_ids: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # Header line: tokens start with GPU0, GPU1, ...
        # It must not start with a row label GPU (data rows do too), but data
        # rows have "X" or link type as first cell after the row label.
        # Reliable heuristic: header has >= 2 consecutive GPU tokens at start.
        gpu_tokens = [t for t in tokens if re.match(r"^GPU\d+$", t)]
        if len(gpu_tokens) >= 2 and (tokens[0] == gpu_tokens[0]):
            # Could be a data row if first token is GPU0 followed by X or link
            is_data_row = tokens[1] in ("X", "SYS", "NODE", "PHB", "PXB", "PIX") or re.match(
                r"^NV\d+$", tokens[1]
            )
            if is_data_row:
                # This is a data row (row starts with GPU id, second token is a link)
                continue
            # This is a header: all GPU tokens are column labels
            header_gpu_ids = gpu_tokens
            header_idx = i
            break

    if header_idx is None or not header_gpu_ids:
        return [], {}, {}

    pair_levels: dict[tuple[str, str], TopologyLinkType] = {}
    numa_affinity: dict[str, int] = {}
    n_gpu_cols = len(header_gpu_ids)

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if not tokens or not re.match(r"^GPU\d+$", tokens[0]):
            continue  # skip NIC rows, legend lines

        row_gpu = tokens[0]
        cells = tokens[1:]

        # Parse GPU×GPU cells
        for col_idx, col_gpu in enumerate(header_gpu_ids):
            if col_idx >= len(cells):
                break
            link = _parse_link_token(cells[col_idx])
            if link is not None and row_gpu != col_gpu:
                key = TopologyState.make_pair_key(row_gpu, col_gpu)
                if key not in pair_levels:
                    pair_levels[key] = link

        # Parse NUMA Affinity: tokens after the GPU columns
        # remaining = [CPU-affinity-range, NUMA-node, ...]
        # NUMA affinity is the first pure-integer token after the GPU columns
        remaining = cells[n_gpu_cols:]
        for tok in remaining:
            if re.match(r"^\d+$", tok):
                try:
                    numa_affinity[row_gpu] = int(tok)
                except ValueError:
                    pass
                break

    return list(header_gpu_ids), pair_levels, numa_affinity


def parse_nvidia_smi_list(text: str) -> list[dict[str, str]]:
    """Parse nvidia-smi -L output into GPU info dicts.

    Example line:
        GPU 0: NVIDIA H100 SXM5 80GB (UUID: GPU-abc123-...)

    Returns list of {index, name, uuid} dicts.
    """
    gpus: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^GPU\s+(\d+)\s*:\s*(.+?)\s*\(UUID:\s*(GPU-[^\)]+)\)", stripped)
        if m:
            gpus.append({
                "index": m.group(1),
                "name": m.group(2).strip(),
                "uuid": m.group(3).strip(),
            })
    return gpus


# ---------------------------------------------------------------------------
# Interconnect classification and TopologyState builder
# ---------------------------------------------------------------------------


def _derive_interconnect_class(
    gpu_ids: list[str],
    pair_levels: dict[tuple[str, str], TopologyLinkType],
) -> Optional[str]:
    """Classify the node's overall interconnect quality.

    nvlink_full:    all GPU pairs connected via NVSwitch or NV4+
    nvlink_partial: at least one NVLink pair but not all full NVSwitch
    pcie:           all pairs via PIX/PXB/PHB (no NVLink)
    cross_numa:     worst link is NODE or SYS
    unknown:        no topology data
    """
    if not gpu_ids or not pair_levels:
        return "unknown"

    links = list(pair_levels.values())
    nv_links = {TopologyLinkType.NV1, TopologyLinkType.NV2, TopologyLinkType.NV3,
                TopologyLinkType.NV4, TopologyLinkType.NVSWITCH}
    pcie_links = {TopologyLinkType.PIX, TopologyLinkType.PXB, TopologyLinkType.PHB}
    cross_numa_links = {TopologyLinkType.NODE, TopologyLinkType.SYS}

    has_nvswitch = any(lnk == TopologyLinkType.NVSWITCH for lnk in links)
    has_nv = any(lnk in nv_links for lnk in links)
    has_cross_numa = any(lnk in cross_numa_links for lnk in links)

    if has_nvswitch:
        n = len(gpu_ids)
        expected_pairs = n * (n - 1) // 2
        nvswitch_pairs = sum(1 for lnk in links if lnk == TopologyLinkType.NVSWITCH)
        if nvswitch_pairs >= expected_pairs:
            return "nvlink_full"
        return "nvlink_partial"

    if has_nv:
        return "nvlink_partial"

    if has_cross_numa:
        return "cross_numa"

    if any(lnk in pcie_links for lnk in links):
        return "pcie"

    return "unknown"


def build_topology_state(
    node_id: str,
    gpu_ids: list[str],
    uuid_map: dict[str, str],
    pair_levels: dict[tuple[str, str], TopologyLinkType],
    numa_affinity: dict[str, int],
    ts: datetime,
    provenance: Provenance,
) -> TopologyState:
    """Build a TopologyState from parsed nvidia-smi data.

    Translates logical GPU IDs (GPU0, GPU1, …) to UUIDs via uuid_map.
    Falls back to logical IDs if UUID mapping is incomplete.
    """
    # Translate pair_levels keys from logical IDs to UUIDs
    uuid_pair_levels: dict[tuple[str, str], TopologyLinkType] = {}
    for (a, b), link in pair_levels.items():
        uuid_a = uuid_map.get(a, a)
        uuid_b = uuid_map.get(b, b)
        key = TopologyState.make_pair_key(uuid_a, uuid_b)
        uuid_pair_levels[key] = link

    gpu_uuids = tuple(uuid_map.get(gid, gid) for gid in gpu_ids)
    uuid_numa = {uuid_map.get(gid, gid): n for gid, n in numa_affinity.items()}

    interconnect_class = _derive_interconnect_class(list(gpu_uuids), uuid_pair_levels)
    has_nv = interconnect_class in ("nvlink_full", "nvlink_partial")

    return TopologyState(
        node_id=node_id,
        timestamp=ts,
        provenance=provenance,
        gpu_uuids=gpu_uuids,
        pair_levels=uuid_pair_levels,
        numa_affinity=uuid_numa,
        nvlink_present=has_nv if interconnect_class != "unknown" else None,
        interconnect_class=interconnect_class,
    )


# ---------------------------------------------------------------------------
# Placement scoring
# ---------------------------------------------------------------------------

# Base penalty per link type (0.0 = ideal, 1.0 = worst)
_LINK_PENALTY: dict[TopologyLinkType, float] = {
    TopologyLinkType.NVSWITCH: 0.00,
    TopologyLinkType.NV4:      0.05,
    TopologyLinkType.NV3:      0.10,
    TopologyLinkType.NV2:      0.15,
    TopologyLinkType.NV1:      0.20,
    TopologyLinkType.PIX:      0.35,
    TopologyLinkType.PXB:      0.45,
    TopologyLinkType.PHB:      0.55,
    TopologyLinkType.NODE:     0.70,
    TopologyLinkType.SYS:      0.85,
    TopologyLinkType.RACK:     0.92,
    TopologyLinkType.REGION:   1.00,
}

_COMM_MULTIPLIER: dict[str, float] = {"low": 0.5, "medium": 1.0, "high": 2.0}
_LATENCY_EXTRA: float = 1.5


@dataclass
class PlacementWorkloadSpec:
    """Workload characterization for placement scoring.

    Attributes:
        gpu_count:               required number of GPUs
        communication_intensity: "low" | "medium" | "high"
        latency_sensitive:       True = latency SLA applies (adds penalty weight)
    """
    gpu_count: int = 1
    communication_intensity: str = "medium"
    latency_sensitive: bool = False


@dataclass
class PlacementScore:
    """Result of scoring one placement candidate.

    score:       0.0 (ideal) … 1.0 (worst). Lower is better.
    explanation: human-readable rationale for the score.
    worst_link:  the weakest link among candidate GPU pairs (None if unknown).
    """
    score: float
    explanation: str
    worst_link: Optional[TopologyLinkType] = None


def score_placement(
    workload: PlacementWorkloadSpec,
    candidate_gpu_uuids: list[str],
    topology: Optional[TopologyState],
) -> PlacementScore:
    """Score a candidate GPU placement for a workload.

    Lower score = better placement.

    Returns 0.0 for trivial (single GPU) placements.
    Returns conservative 0.5 when topology is unavailable — not fabricated zero.
    """
    if len(candidate_gpu_uuids) <= 1 or workload.gpu_count <= 1:
        return PlacementScore(score=0.0, explanation="Single GPU; topology irrelevant")

    if topology is None:
        return PlacementScore(
            score=0.5,
            explanation="No topology data; conservative estimate",
        )

    known_links: list[TopologyLinkType] = []
    missing_count = 0

    for i, uuid_a in enumerate(candidate_gpu_uuids):
        for uuid_b in candidate_gpu_uuids[i + 1:]:
            link = topology.link_between(uuid_a, uuid_b)
            if link is not None:
                known_links.append(link)
            else:
                missing_count += 1

    if not known_links:
        return PlacementScore(
            score=0.5,
            explanation=f"No topology data for {missing_count} GPU pair(s); conservative estimate",
        )

    mean_penalty = sum(_LINK_PENALTY.get(lnk, 0.5) for lnk in known_links) / len(known_links)
    worst_link = max(known_links, key=lambda lnk: _LINK_PENALTY.get(lnk, 0.5))

    comm_mult = _COMM_MULTIPLIER.get(workload.communication_intensity, 1.0)
    latency_mult = _LATENCY_EXTRA if workload.latency_sensitive else 1.0

    score = min(1.0, mean_penalty * comm_mult * latency_mult)

    parts = [
        f"mean_penalty={mean_penalty:.3f}",
        f"comm={workload.communication_intensity}(×{comm_mult:.1f})",
    ]
    if workload.latency_sensitive:
        parts.append(f"latency_sensitive(×{latency_mult:.1f})")
    if missing_count:
        parts.append(f"missing_pairs={missing_count}")

    explanation = f"worst_link={worst_link.value}; " + "; ".join(parts)
    return PlacementScore(score=score, explanation=explanation, worst_link=worst_link)


def rank_placements(
    workload: PlacementWorkloadSpec,
    candidates: list[list[str]],
    topology: Optional[TopologyState],
) -> list[tuple[list[str], PlacementScore]]:
    """Rank placement candidates best-first (lowest score first)."""
    scored = [(c, score_placement(workload, c, topology)) for c in candidates]
    return sorted(scored, key=lambda x: x[1].score)


# ---------------------------------------------------------------------------
# Real topology collector (nvidia-smi shell)
# ---------------------------------------------------------------------------


class NvidiaSmiTopologyCollector:
    """Collect GPU topology via nvidia-smi on the local node.

    Requires node-local shell access. Returns None on any failure
    so the caller can degrade gracefully (no topology = conservative scoring).
    """

    def __init__(self, node_id: str, is_sandbox: bool = False) -> None:
        self.node_id = node_id
        self.is_sandbox = is_sandbox

    def collect(self) -> Optional[TopologyState]:
        ts = datetime.now(tz=timezone.utc)
        prov = Provenance(
            source="nvidia-smi",
            fetched_at=ts,
            confidence="medium",  # text parsing; NVML preferred but optional
            is_sandbox=self.is_sandbox,
        )

        try:
            topo_text = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode("utf-8", errors="replace")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("nvidia-smi topo -m failed on %s: %s", self.node_id, exc)
            return None

        try:
            inventory_text = subprocess.check_output(
                ["nvidia-smi", "-L"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode("utf-8", errors="replace")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            inventory_text = ""

        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(topo_text)
        if not gpu_ids:
            logger.warning("No GPUs parsed from nvidia-smi topo -m on %s", self.node_id)
            return None

        inventory = parse_nvidia_smi_list(inventory_text)
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        return build_topology_state(
            self.node_id, gpu_ids, uuid_map, pair_levels, numa_affinity, ts, prov
        )


# ---------------------------------------------------------------------------
# Fake (sandbox) topology collector
# ---------------------------------------------------------------------------


class FakeTopologyCollector:
    """Fixture-based sandbox topology collector.

    Accepts a pre-built TopologyState or raw nvidia-smi text fixtures.
    Uses the same parse/build paths as NvidiaSmiTopologyCollector.

    Usage:
        collector = FakeTopologyCollector(
            node_id="dgx-01",
            topo_text=DGX_H100_TOPO_TEXT,
            inventory_text=DGX_H100_INVENTORY_TEXT,
        )
        topology = collector.collect()
    """

    def __init__(
        self,
        node_id: str,
        topo_text: Optional[str] = None,
        inventory_text: Optional[str] = None,
        topology_state: Optional[TopologyState] = None,
    ) -> None:
        self.node_id = node_id
        self._topo_text = topo_text
        self._inventory_text = inventory_text
        self._topology_state = topology_state

    def collect(self) -> Optional[TopologyState]:
        if self._topology_state is not None:
            return self._topology_state

        if self._topo_text is None:
            return None

        ts = datetime.now(tz=timezone.utc)
        prov = Provenance(
            source="nvidia-smi-fixture",
            fetched_at=ts,
            confidence="medium",
            is_sandbox=True,
        )

        gpu_ids, pair_levels, numa_affinity = parse_nvidia_smi_topo(self._topo_text)
        if not gpu_ids:
            return None

        inventory = parse_nvidia_smi_list(self._inventory_text or "")
        uuid_map = {f"GPU{g['index']}": g["uuid"] for g in inventory}

        return build_topology_state(
            self.node_id, gpu_ids, uuid_map, pair_levels, numa_affinity, ts, prov
        )
