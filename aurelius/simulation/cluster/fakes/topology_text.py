"""Generate fake nvidia-smi topology outputs from simulator state.

Outputs match the exact format of real nvidia-smi topo -m and nvidia-smi -L
commands. The production TopologyConnector (topology.py) parses these without
modification, verifying that simulator and real topology share the same path.

Reference matrix legend (from nvidia-smi topo -m):
  NV#  = NVLink connection (#=bandwidth multiplier)
  PIX  = PCIe: same root complex
  PXB  = PCIe: traverses PCIe bridge
  PHB  = PCIe: traverses PCIe host bridge
  NODE = inter-socket (SMP interconnect)
  SYS  = different NUMA node, traverses NUMA fabric
"""

from __future__ import annotations

from ..model import SimGPU, SimNode


def generate_topo_text(node: SimNode) -> str:
    """Generate nvidia-smi topo -m format text for a node."""
    gpus = node.gpus
    if not gpus:
        return ""

    n = len(gpus)
    gpu_ids = [f"GPU{g.gpu_index}" for g in gpus]
    nic_id = "mlx5_0"

    # Build link map: (gpu_a_id, gpu_b_id) → link_type string
    link_map: dict[tuple[str, str], str] = {}
    for link in node.topology_links:
        # Find gpu_index for each side
        a_idx = _gpu_id_to_index(link.gpu_a, gpus)
        b_idx = _gpu_id_to_index(link.gpu_b, gpus)
        if a_idx is None or b_idx is None:
            continue
        key = (f"GPU{a_idx}", f"GPU{b_idx}")
        link_map[key] = link.link_type
        link_map[(f"GPU{b_idx}", f"GPU{a_idx}")] = link.link_type

    # Header row: GPU0 GPU1 ... GPUn-1 mlx5_0 CPU Affinity NUMA Affinity
    header_cols = gpu_ids + [nic_id, "CPU Affinity", "NUMA Affinity"]
    lines: list[str] = []

    # Column header line
    header = "\t".join(header_cols)
    lines.append(f"\t{header}")

    # GPU rows
    for i, gpu in enumerate(gpus):
        row_label = f"GPU{i}"
        numa_node = 0 if i < n // 2 else 1   # first half on NUMA 0, second on NUMA 1

        cols: list[str] = [row_label]
        for j, other_gpu in enumerate(gpus):
            if i == j:
                cols.append("X")
            else:
                key = (f"GPU{i}", f"GPU{j}")
                link_type = link_map.get(key, "SYS")
                cols.append(_format_link_type(link_type))

        # NIC column
        nic_link = "NODE" if n > 4 else "PIX"
        cols.append(nic_link)

        # CPU Affinity (NUMA0: 0-47, NUMA1: 48-95 for 96-core node)
        cpu_aff = "0-47" if numa_node == 0 else "48-95"
        cols.append(cpu_aff)

        # NUMA Affinity
        cols.append(str(numa_node))

        lines.append("\t".join(cols))

    # NIC row
    nic_row = [nic_id]
    for i in range(n):
        nic_link = "NODE" if n > 4 else "PIX"
        nic_row.append(nic_link)
    nic_row.append("X")  # NIC-NIC
    nic_row.append("0-47")
    nic_row.append("0")
    lines.append("\t".join(nic_row))

    lines.append("")
    lines.append("Legend:")
    lines.append("")
    lines.append("  X    = Self")
    lines.append("  SYS  = Connection traversing PCIe as well as the SMP interconnect"
                 " between NUMA nodes (e.g., QPI/UPI)")
    lines.append("  NODE = Connection traversing PCIe as well as the interconnect"
                 " between PCIe Host Bridges within a NUMA node")
    lines.append("  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge"
                 " (typically the CPU)")
    lines.append("  PXB  = Connection traversing multiple PCIe bridges"
                 " (without traversing the PCIe Host Bridge)")
    lines.append("  PIX  = Connection traversing at most a single PCIe bridge")
    lines.append("  NV#  = Connection traversing a bonded set of # NVLinks")

    return "\n".join(lines)


def generate_gpu_list_text(node: SimNode) -> str:
    """Generate nvidia-smi -L format text for a node."""
    lines: list[str] = []
    for gpu in node.gpus:
        lines.append(f"GPU {gpu.gpu_index}: {gpu.profile.model_name} (UUID: {gpu.uuid})")
    return "\n".join(lines)


def _gpu_id_to_index(gpu_id: str, gpus: list[SimGPU]) -> int | None:
    for gpu in gpus:
        if gpu.gpu_id == gpu_id:
            return gpu.gpu_index
    return None


def _format_link_type(link_type: str) -> str:
    """Format link type for nvidia-smi topo -m output."""
    if link_type == "NVSWITCH":
        return "NV18"    # NVSwitch = NV18 on H100 DGX
    if link_type == "NV4":
        return "NV4"
    if link_type == "NV2":
        return "NV2"
    if link_type == "NV1":
        return "NV1"
    # PCIe types pass through unchanged
    return link_type
