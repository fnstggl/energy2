"""Explicit, mutable GPU-utilization / fragmentation / bin-packing state models.

First-class simulator states required by the utilization-realism upgrade.
Mutable (updated each tick by the engine), separate from the frozen
ClusterState. ``GPUUtilizationState`` is attached per SimGPU; cluster/region
fragmentation + stranded-capacity state is computed per tick; per-workload
packing/flexibility/consolidation state lives on ``WorkloadUtilizationState``.

All values are bounded proxies, NOT a scheduler/allocator simulation. The
utilization regimes, roofline ceilings, fragmentation thresholds, and
consolidation curves are tunable engineering heuristics (see calibration.py),
not measured per-cluster numbers. Do NOT read any value here as
production-accurate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Multi-dimensional per-GPU utilization sub-states
# ---------------------------------------------------------------------------

@dataclass
class SMUtilizationState:
    """Streaming-multiprocessor (compute) utilization."""
    sm_util: float = 0.0            # [0,1]


@dataclass
class MemoryBandwidthState:
    """DRAM memory-bandwidth utilization (decode is memory-bound)."""
    dram_active: float = 0.0        # [0,1] DRAM_ACTIVE
    mem_copy_util: float = 0.0      # [0,1] MEM_COPY_UTIL
    saturated: bool = False


@dataclass
class DRAMPressureState:
    """DRAM-bandwidth pressure regime."""
    pressure: float = 0.0           # [0,1]
    regime: str = "nominal"         # nominal | elevated | saturated


@dataclass
class SchedulerPressureState:
    """Scheduler/service saturation (admission + scheduling overhead)."""
    pressure: float = 0.0           # [0,1]
    active_sequences: float = 0.0
    saturated: bool = False


@dataclass
class PCIePressureState:
    """PCIe host<->device transfer pressure."""
    pressure: float = 0.0           # [0,1]
    saturated: bool = False


@dataclass
class KVHeadroomState:
    """KV-cache concurrency headroom for a GPU/workload."""
    occupancy: float = 0.0          # [0,1]
    headroom_frac: float = 1.0      # [0,1] remaining usable headroom
    admission_suppressed: bool = False


@dataclass
class MemoryHeadroomState:
    """VRAM headroom (weights + activations + KV vs reserve)."""
    used_frac: float = 0.0          # [0,1]
    reserve_frac: float = 0.05      # reserved headroom
    headroom_frac: float = 1.0      # usable remaining
    over_reserve: bool = False


@dataclass
class BatchingEfficiencyState:
    """Continuous-batching efficiency + gain."""
    occupancy: float = 0.0          # [0,1] batch occupancy
    gain: float = 1.0               # throughput multiplier vs static batching
    collapsed: bool = False         # batching collapse under churn/KV pressure


@dataclass
class ContinuousBatchingState:
    """Continuous-batching scheduling state (active-seq / variable-length)."""
    active_sequences: float = 0.0
    output_len_cv: float = 0.0      # coefficient of variation of output length
    prefill_decode_interference: float = 0.0  # [0,1]


@dataclass
class QueueAmplificationState:
    """Queue waiting-time amplification under packing density."""
    amplification: float = 1.0      # multiplier on queue wait
    unstable: bool = False          # superlinear queue growth


@dataclass
class GPUSharingState:
    """GPU sharing (MIG / fractional / time-slice) bookkeeping."""
    shared: bool = False
    mode: str = "none"              # none | mig | time_slice | fractional
    tenants: int = 1
    interference_frac: float = 0.0


@dataclass
class UtilizationTelemetryConfidence:
    """Utilization telemetry quality for a GPU/node."""
    tier: str = "high"              # high | medium | low
    stale_ticks: int = 0
    gpu_util_visible: bool = True
    dram_visible: bool = True
    scheduler_visible: bool = True


@dataclass
class GPUUtilizationState:
    """Composite multi-dimensional per-GPU utilization state.

    U_gpu = min(U_sm, U_mem, U_cpu_feed, U_sched, U_pcie). The effective GPU
    utilization is the BOTTLENECK dimension, not the compute scalar — low SM
    utilization can coexist with a saturated DRAM/scheduler/PCIe dimension.
    """
    sm: SMUtilizationState = field(default_factory=SMUtilizationState)
    mem: MemoryBandwidthState = field(default_factory=MemoryBandwidthState)
    dram: DRAMPressureState = field(default_factory=DRAMPressureState)
    scheduler: SchedulerPressureState = field(default_factory=SchedulerPressureState)
    pcie: PCIePressureState = field(default_factory=PCIePressureState)
    kv: KVHeadroomState = field(default_factory=KVHeadroomState)
    memory: MemoryHeadroomState = field(default_factory=MemoryHeadroomState)
    batching: BatchingEfficiencyState = field(default_factory=BatchingEfficiencyState)
    sharing: GPUSharingState = field(default_factory=GPUSharingState)
    telemetry: UtilizationTelemetryConfidence = field(
        default_factory=UtilizationTelemetryConfidence
    )
    # Effective bottleneck utilization + which dimension binds.
    effective_util: float = 0.0
    bottleneck: str = "sm"          # sm | mem | sched | pcie | kv
    underutilized: bool = False
    utilization_paradox: bool = False  # high DRAM_ACTIVE + low SM


# ---------------------------------------------------------------------------
# Cluster / region fragmentation + stranded-capacity states
# ---------------------------------------------------------------------------

@dataclass
class ResourceDomainState:
    """Free / schedulable capacity within one placement domain."""
    domain: str = "node"
    domain_id: str = ""
    free_gpus: int = 0
    schedulable_gpus: int = 0       # free AND usable for the demand profile


@dataclass
class FragmentationState:
    """Multidimensional / topology-aware fragmentation for a region/cluster."""
    score: float = 0.0              # 1 - schedulable_free / free
    topology_score: float = 0.0     # 1 - Σ min(free_d,demand_d)/Σ free_d
    regime: str = "nominal"         # nominal | elevated | critical
    free_gpus: int = 0
    schedulable_gpus: int = 0


@dataclass
class StrandedCapacityState:
    """Free-but-unusable GPU capacity, by reason."""
    stranded_gpus: int = 0
    topology_isolated: int = 0
    vram_isolated: int = 0
    comm_isolated: int = 0
    sla_incompatible: int = 0


@dataclass
class PackingDensityState:
    """Cluster/region packing density."""
    density: float = 0.0            # allocated / total GPUs
    allocated_gpus: int = 0
    total_gpus: int = 0


@dataclass
class SchedulabilityState:
    """Whether a representative large job can currently be placed."""
    feasible: bool = True
    largest_feasible_gpu_count: int = 0
    reject_reason: str = ""


@dataclass
class BinPackingRiskState:
    """Bin-packing risk for the region (fragmentation × density × demand)."""
    risk: float = 0.0               # [0,1]
    unsafe: bool = False


# ---------------------------------------------------------------------------
# Per-workload packing / flexibility / consolidation states
# ---------------------------------------------------------------------------

@dataclass
class WorkloadFlexibilityState:
    """Migration/consolidation freedom of a workload."""
    flexibility: str = "medium"     # low | medium | high
    multiplier: float = 0.6         # 0..1 freedom multiplier


@dataclass
class TopologyFeasibilityState:
    """Whether a workload's topology requirements can be met at a placement."""
    feasible: bool = True
    requires_locality: bool = False
    cross_node: bool = False


@dataclass
class CrossNodeShardState:
    """Cross-node sharding bookkeeping for a multi-GPU workload."""
    sharded: bool = False
    node_count: int = 1
    shard_penalty_frac: float = 0.0


@dataclass
class ConsolidationRiskState:
    """Consolidation/packing risk for a workload + its drivers."""
    risk: float = 0.0               # [0,1]
    benefit: float = 0.0            # saturating idle-capacity benefit
    unsafe: bool = False
    cross_domain: float = 0.0
    queue_pressure: float = 0.0
    thermal_pressure: float = 0.0
    kv_pressure: float = 0.0
    scheduler_pressure: float = 0.0


@dataclass
class WorkloadUtilizationState:
    """Composite per-workload utilization/packing/consolidation state."""
    workload_class: str = "standard_inference"
    flexibility: WorkloadFlexibilityState = field(
        default_factory=WorkloadFlexibilityState
    )
    topology_feasibility: TopologyFeasibilityState = field(
        default_factory=TopologyFeasibilityState
    )
    cross_node_shard: CrossNodeShardState = field(default_factory=CrossNodeShardState)
    consolidation: ConsolidationRiskState = field(default_factory=ConsolidationRiskState)
    queue_amp: QueueAmplificationState = field(default_factory=QueueAmplificationState)
    continuous_batching: ContinuousBatchingState = field(
        default_factory=ContinuousBatchingState
    )
    # Roofline token ceiling + binding term (informational).
    roofline_tokens_per_sec: float = 0.0
    roofline_bottleneck: str = "compute"
    # Effective utilization throughput factor applied this tick.
    util_throughput_factor: float = 1.0
