"""Explicit, mutable topology / communication / placement state models.

First-class simulator states required by the topology-realism upgrade. Mutable
(updated each tick by the engine), separate from the frozen ClusterState.
``GPUFabricState`` is attached per SimGPU; ``NodeFabricState`` per SimNode;
``CommunicationLatencyState`` / ``TopologyMigrationRiskState`` per SimWorkload.

All values are bounded proxies, NOT a network simulation. The fabric regimes,
distance ladder, and contention curves are tunable engineering heuristics (see
calibration.py), not measured per-cluster numbers. Do NOT read any value here as
production-accurate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Per-link / per-fabric sub-states
# ---------------------------------------------------------------------------

@dataclass
class NVLinkDomainState:
    """NVLink point-to-point domain bookkeeping for a GPU/node."""
    generation: str = "a100"        # a100 | h100 | future
    bidir_gbps: float = 600.0
    visible: bool = True            # partial NVLink visibility → False
    saturation: float = 0.0         # [0,1] link utilization


@dataclass
class NVSwitchState:
    """NVSwitch fully-connected fabric domain bookkeeping."""
    present: bool = False
    domain_size: int = 0            # GPUs in the NVSwitch domain
    saturation: float = 0.0         # [0,1] aggregate domain utilization


@dataclass
class PCIeFabricState:
    """PCIe fabric bookkeeping (root-complex + generation)."""
    generation: str = "gen5"        # gen4 | gen5
    root_complex_id: int = 0
    b_eff_gbps: float = 50.0
    saturation: float = 0.0
    queueing_delay_us: float = 0.0


@dataclass
class NUMAState:
    """NUMA-node bookkeeping for a GPU (which NUMA node it is attached to)."""
    numa_node: int = 0
    cross_numa_penalty_active: bool = False


@dataclass
class SocketLocalityState:
    """CPU-socket locality bookkeeping."""
    socket_id: int = 0
    cpu_staging_overhead_us: float = 0.0


@dataclass
class RackLocalityState:
    """Rack locality bookkeeping for a node."""
    rack_id: str = ""
    cross_rack_traffic_frac: float = 0.0


@dataclass
class InterconnectCongestionState:
    """Fabric contention / congestion for a node/link."""
    nvlink_congestion: float = 0.0  # [0,1]
    pcie_congestion: float = 0.0
    fabric_oversubscribed: bool = False
    bandwidth_degradation_frac: float = 0.0  # effective-BW loss this tick


@dataclass
class NICCongestionState:
    """Per-node NIC congestion (cross-node bottleneck)."""
    throughput_frac: float = 0.0    # [0,1] of NIC capacity in use
    incast_active: bool = False
    saturation: float = 0.0


@dataclass
class CrossRegionFabricState:
    """Cross-region WAN fabric bookkeeping for a node/region."""
    cross_region_active: bool = False
    rtt_ms: float = 0.0
    bandwidth_gbps: float = 0.0


@dataclass
class CommunicationPressureState:
    """Aggregate communication pressure for a node/workload."""
    pressure: float = 0.0           # [0,1]
    regime: str = "nominal"         # nominal | elevated | congested | collapse


@dataclass
class CollectiveLoadState:
    """Collective-communication load + amplification for a workload."""
    collective: str = "none"        # all_reduce | all_to_all | p2p | tree | none
    participants: int = 1           # N ranks
    amplification: float = 1.0      # collective cost / point-to-point cost
    latency_ms: float = 0.0         # modelled collective latency this tick


@dataclass
class SynchronizationPenaltyState:
    """Synchronization-stall penalty for a sync-heavy workload."""
    sync_heavy: bool = False
    straggler_frac: float = 0.0     # worst-rank lag fraction
    slowdown_frac: float = 0.0      # throughput slowdown from sync stalls


@dataclass
class TopologyRiskState:
    """Topology degradation risk for a workload's current placement."""
    risk: float = 0.0               # [0,1]
    instability: bool = False       # collective-instability risk flag


@dataclass
class PlacementAffinityState:
    """Placement affinity for a workload (how well its GPUs are co-located)."""
    best_regime: str = "nvswitch"
    worst_regime: str = "nvswitch"
    distance_score: float = 0.0     # raw traffic-weighted Σ w*d (lower=better)
    quality_score: float = 1.0      # 0-1, 1 = ideal locality


@dataclass
class FabricTelemetryConfidence:
    """Topology telemetry quality for a node/workload."""
    tier: str = "high"              # high | medium | low
    stale_ticks: int = 0
    nvlink_visible: bool = True
    pcie_visible: bool = True
    nic_visible: bool = True
    detached_devices: int = 0


@dataclass
class TopologyHealthState:
    """Topology health / fragmentation bookkeeping for a node."""
    fragmented: bool = False
    degraded_links: int = 0
    health: float = 1.0             # [0,1], 1 = healthy


@dataclass
class CommunicationLatencyState:
    """Communication latency + tail bookkeeping for a workload."""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    tail_mult: float = 1.0          # comm-induced tail amplification (p99/p50)


@dataclass
class TopologyMigrationRiskState:
    """Topology-aware migration risk + veto bookkeeping for a workload."""
    risk: float = 0.0
    veto_count: int = 0
    last_veto_reason: str = ""


# ---------------------------------------------------------------------------
# Composite per-GPU / per-node / per-workload states
# ---------------------------------------------------------------------------

@dataclass
class GPUFabricState:
    """Per-GPU fabric state (NVLink / PCIe / NUMA / socket attachment)."""
    nvlink: NVLinkDomainState = field(default_factory=NVLinkDomainState)
    nvswitch: NVSwitchState = field(default_factory=NVSwitchState)
    pcie: PCIeFabricState = field(default_factory=PCIeFabricState)
    numa: NUMAState = field(default_factory=NUMAState)
    socket: SocketLocalityState = field(default_factory=SocketLocalityState)
    nvlink_tx_frac: float = 0.0
    nvlink_rx_frac: float = 0.0
    pcie_tx_frac: float = 0.0
    pcie_rx_frac: float = 0.0


@dataclass
class NodeFabricState:
    """Per-node fabric state (rack locality, NIC, congestion, telemetry, health)."""
    rack: RackLocalityState = field(default_factory=RackLocalityState)
    congestion: InterconnectCongestionState = field(
        default_factory=InterconnectCongestionState
    )
    nic: NICCongestionState = field(default_factory=NICCongestionState)
    cross_region: CrossRegionFabricState = field(default_factory=CrossRegionFabricState)
    telemetry: FabricTelemetryConfidence = field(default_factory=FabricTelemetryConfidence)
    health: TopologyHealthState = field(default_factory=TopologyHealthState)
    topology_class: str = "nvswitch"


@dataclass
class WorkloadTopologyState:
    """Composite per-workload topology/communication state (all sub-states)."""
    affinity: PlacementAffinityState = field(default_factory=PlacementAffinityState)
    pressure: CommunicationPressureState = field(
        default_factory=CommunicationPressureState
    )
    collective: CollectiveLoadState = field(default_factory=CollectiveLoadState)
    sync: SynchronizationPenaltyState = field(
        default_factory=SynchronizationPenaltyState
    )
    risk: TopologyRiskState = field(default_factory=TopologyRiskState)
    latency: CommunicationLatencyState = field(default_factory=CommunicationLatencyState)
    migration_risk: TopologyMigrationRiskState = field(
        default_factory=TopologyMigrationRiskState
    )
    telemetry: FabricTelemetryConfidence = field(default_factory=FabricTelemetryConfidence)
    comm_profile: str = "comm_light_inference"
    # Throughput penalty applied this tick (1 - this = retained throughput).
    throughput_penalty_frac: float = 0.0
