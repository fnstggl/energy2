"""Mutable simulation state models for the synthetic cluster simulator.

These are NOT the frozen ClusterState models from aurelius/state/models.py.
These are mutable objects the engine updates at each tick, then converts to
canonical ClusterState snapshots for connector/classifier consumption.

Design rules:
- Simulation objects are mutable (plain dataclasses, not frozen)
- Missing/unknown values → None
- All physical values are bounded to realistic ranges
- Thermal, queue, migration, cache, and communication models are proxies,
  not physics simulations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# GPU hardware profiles
# ---------------------------------------------------------------------------

@dataclass
class GPUProfile:
    """Physical characteristics of a GPU model."""
    model_name: str
    memory_total_bytes: int
    base_power_watts: float
    max_power_watts: float
    throttle_temp_c: float = 83.0
    max_temp_c: float = 100.0
    nvlink_bandwidth_gbps: float = 0.0
    pcie_bandwidth_gbps: float = 64.0
    sm_count: int = 128
    # Tokens per second capacity at 100% utilization (7B model proxy)
    tokens_per_sec_at_full_util: float = 2000.0


GPU_PROFILES: dict[str, GPUProfile] = {
    "h100-sxm5-80gb": GPUProfile(
        model_name="NVIDIA H100 SXM5 80GB",
        memory_total_bytes=80 * 1024**3,
        base_power_watts=700.0,
        max_power_watts=700.0,
        throttle_temp_c=83.0,
        max_temp_c=95.0,
        nvlink_bandwidth_gbps=900.0,
        pcie_bandwidth_gbps=128.0,
        sm_count=132,
        tokens_per_sec_at_full_util=3000.0,
    ),
    "a100-sxm4-80gb": GPUProfile(
        model_name="NVIDIA A100 SXM4 80GB",
        memory_total_bytes=80 * 1024**3,
        base_power_watts=400.0,
        max_power_watts=400.0,
        throttle_temp_c=83.0,
        max_temp_c=90.0,
        nvlink_bandwidth_gbps=600.0,
        pcie_bandwidth_gbps=64.0,
        sm_count=108,
        tokens_per_sec_at_full_util=2000.0,
    ),
    "a100-pcie-80gb": GPUProfile(
        model_name="NVIDIA A100 PCIe 80GB",
        memory_total_bytes=80 * 1024**3,
        base_power_watts=300.0,
        max_power_watts=300.0,
        throttle_temp_c=83.0,
        max_temp_c=90.0,
        nvlink_bandwidth_gbps=0.0,
        pcie_bandwidth_gbps=64.0,
        sm_count=108,
        tokens_per_sec_at_full_util=1600.0,
    ),
    "l4": GPUProfile(
        model_name="NVIDIA L4",
        memory_total_bytes=24 * 1024**3,
        base_power_watts=72.0,
        max_power_watts=72.0,
        throttle_temp_c=80.0,
        max_temp_c=90.0,
        nvlink_bandwidth_gbps=0.0,
        pcie_bandwidth_gbps=32.0,
        sm_count=58,
        tokens_per_sec_at_full_util=400.0,
    ),
}


# ---------------------------------------------------------------------------
# Simulation node topology
# ---------------------------------------------------------------------------

@dataclass
class SimTopologyLink:
    """Link between two GPUs in the same node."""
    gpu_a: str
    gpu_b: str
    link_type: str  # "NVSWITCH", "NV4", "NV2", "PIX", "PXB", "PHB", "SYS", "NODE"
    bandwidth_gbps: float


@dataclass
class SimGPU:
    """Mutable GPU simulation state updated each tick."""
    gpu_id: str       # e.g. "us-east-node0-gpu0"
    gpu_index: int    # 0-based index within node
    uuid: str
    node_id: str
    profile: GPUProfile

    # Mutable state (updated each tick)
    utilization_pct: float = 0.0
    sm_activity_pct: float = 0.0
    memory_used_bytes: int = 0
    power_watts: float = 0.0
    temperature_c: float = 35.0
    thermal_throttle_active: bool = False
    xid_error_count: int = 0
    nvlink_rx_bytes_per_sec: float = 0.0
    nvlink_tx_bytes_per_sec: float = 0.0
    pcie_rx_bytes_per_sec: float = 0.0
    pcie_tx_bytes_per_sec: float = 0.0

    # Continuous thermal/power slowdown (0 = none). Replaces the binary throttle
    # for throughput effects; thermal_throttle_active is kept as a derived bool.
    thermal_slowdown_frac: float = 0.0
    power_slowdown_frac: float = 0.0
    power_cap_watts: float = 0.0

    # First-class per-GPU thermal state (mutable). Constructed lazily by engine.
    thermal: Optional[Any] = None

    # First-class per-GPU fabric state (NVLink/PCIe/NUMA/socket). Constructed
    # lazily by the engine to avoid an import cycle.
    fabric: Optional[Any] = None

    # First-class multi-dimensional per-GPU utilization state (SM / DRAM-bandwidth
    # / scheduler / PCIe / KV bottleneck). Constructed lazily by the engine.
    utilization: Optional[Any] = None

    # Assigned workload (one workload per GPU for simplicity)
    assigned_workload_id: Optional[str] = None

    @property
    def memory_total_bytes(self) -> int:
        return self.profile.memory_total_bytes

    @property
    def memory_free_bytes(self) -> int:
        return max(0, self.profile.memory_total_bytes - self.memory_used_bytes)

    @property
    def effective_utilization_pct(self) -> float:
        """Actual effective utilization after thermal throttling."""
        if self.thermal_throttle_active and self.temperature_c > self.profile.throttle_temp_c:
            throttle_factor = max(
                0.5, 1.0 - (self.temperature_c - self.profile.throttle_temp_c) / 20.0
            )
            return self.utilization_pct * throttle_factor
        return self.utilization_pct


@dataclass
class SimNode:
    """Mutable node simulation state."""
    node_id: str
    region_id: str
    zone: str
    rack_id: str
    instance_type: str
    gpus: list[SimGPU] = field(default_factory=list)
    topology_links: list[SimTopologyLink] = field(default_factory=list)

    # Rack heat accumulation (affects GPU ambient temp)
    rack_heat_delta_c: float = 0.0
    # Event-driven extra heat (thermal_hotspot events); decays each tick.
    event_heat_c: float = 0.0

    # Cooling regime (air | liquid | hybrid | hot_aisle_containment | weak_airflow)
    cooling_regime: str = "air"
    # First-class per-rack thermal state (mutable). Constructed lazily by engine.
    rack_thermal: Optional[Any] = None

    # First-class per-node fabric state (rack/NIC/congestion/telemetry/health).
    # Constructed lazily by the engine.
    node_fabric: Optional[Any] = None

    # Labels (matches K8s topology.kubernetes.io/zone etc.)
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, str]] = field(default_factory=list)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def gpu_allocated_count(self) -> int:
        return sum(1 for g in self.gpus if g.assigned_workload_id is not None)


@dataclass
class SimQueue:
    """Request queue simulation state for an inference service."""
    queue_id: str
    service_id: str
    region_id: str

    # Arrival model: Poisson with diurnal modulation
    base_arrival_rate_per_sec: float = 1.0
    diurnal_amplitude: float = 0.4   # fraction of base (0 = flat, 1 = full swing)
    surge_active: bool = False
    surge_multiplier: float = 1.0
    in_burst: bool = False           # Markov-modulated burst state (serving realism)

    # State
    queue_depth: int = 0
    pending_jobs: int = 0

    # Computed each tick
    arrival_rate_per_sec: float = 0.0
    service_rate_per_sec: float = 0.0
    oldest_pending_age_sec: float = 0.0
    queue_wait_p95_ms: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    timeout_rate_pct: float = 0.0
    error_rate_pct: float = 0.0
    active_sequences: int = 0
    batch_size: int = 0
    kv_cache_usage_pct: Optional[float] = None
    prefix_cache_hit_rate_pct: Optional[float] = None
    # KV-cache realism telemetry (None = not visible / not yet computed)
    kv_pressure: Optional[float] = None
    kv_pressure_region: Optional[str] = None
    preemptions_total: Optional[float] = None
    cache_fragmentation_frac: Optional[float] = None
    # Migration-realism telemetry
    proxy_saturation: Optional[float] = None
    batch_efficiency: Optional[float] = None
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0


@dataclass
class SimWorkload:
    """Active workload (inference service, batch job, training run)."""
    workload_id: str
    service_id: str
    workload_type: str    # "inference", "batch_training", "fine_tuning", "embedding"
    priority_tier: str    # "critical", "latency_sensitive", "standard", "flexible", "batch"

    region_id: str
    node_ids: list[str] = field(default_factory=list)
    gpu_ids: list[str] = field(default_factory=list)
    gpu_count_required: int = 1

    # Resource demand profile
    target_util_pct: float = 60.0
    memory_required_bytes: int = 20 * 1024**3   # 20 GiB default

    # Communication profile (affects topology penalty)
    communication_intensity: str = "low"   # "low", "medium", "high"
    memory_intensity: str = "medium"
    latency_sensitive: bool = False

    # Topology / communication sensitivity profile (names an entry in
    # calibration.WORKLOAD_COMM_PROFILES). None → inferred from workload_type /
    # communication_intensity so existing scenarios keep working unchanged.
    # tensor_parallel | pipeline_parallel | all_reduce_training | moe_expert |
    # embedding | retrieval | batch_inference | comm_light_inference
    comm_profile: Optional[str] = None
    # Representative collective message size (bytes) for the comm-cost model
    # (gradient/activation/expert payload). Order-of-magnitude proxy.
    comm_message_bytes: int = 4 * 1024 * 1024  # 4 MiB default

    # Utilization / fragmentation / packing realism (drives utilization.py).
    # workload_class names an entry in calibration.WORKLOAD_CLASS_PROFILES; None →
    # inferred from workload_type + comm/memory intensity. flexibility (low|
    # medium|high) overrides the class default. sharing_policy controls GPU
    # sharing; admissible_domains restricts placement; output_len_cv drives the
    # continuous-batching gain; vram_requirement_bytes defaults to memory_required.
    workload_class: Optional[str] = None
    flexibility: Optional[str] = None
    sharing_policy: str = "exclusive"   # exclusive | mig | time_slice | fractional
    sharing_tenants: int = 1
    admissible_domains: list[str] = field(default_factory=list)
    output_len_cv: float = 0.5          # coefficient of variation of output length
    vram_requirement_bytes: Optional[int] = None

    # SLA
    sla_policy_id: Optional[str] = None
    latency_sla_p99_ms: Optional[float] = None   # None = no hard SLA
    queue_sla_p95_ms: Optional[float] = None

    # Migration state
    migration_allowed: bool = True
    last_migrated_tick: Optional[int] = None
    last_scaled_tick: Optional[int] = None       # for autoscaling cooldown / anti-flap
    cold_start_warmup_ticks_remaining: int = 0   # ticks until full throughput

    # Cache proxy (0-1 fractions)
    kv_cache_usage_frac: float = 0.3
    prefix_cache_hit_rate_frac: float = 0.5

    # KV-cache architecture + workload prefix character (drives the KV realism
    # layer). model_kv_profile names an entry in calibration.MODEL_KV_PROFILES;
    # prefix_overlap is the workload-family shared-prefix overlap in [0, 1].
    model_kv_profile: str = "llama3-8b"
    prefix_overlap: float = 0.5
    avg_seq_len_tokens: int = 1024

    # Event-forced overrides (kv_cache_pressure event). None = no override.
    kv_pressure_override: Optional[float] = None
    prefix_hit_override: Optional[float] = None

    # First-class KV-cache / prefix-affinity / locality state (mutable; updated
    # each tick). Constructed lazily by the engine to avoid an import cycle.
    cache: Optional[Any] = None

    # Migration / rerouting / drain / cold-start realism.
    engine_runtime: str = "vllm"          # vllm | tensorrt-llm | sglang | triton | ray_serve
    warm_pool_size: int = 0               # replicas kept pre-loaded/ready
    pdb_min_available: int = 0            # PodDisruptionBudget floor
    # First-class migration state (mutable). Constructed lazily by the engine.
    migration: Optional[Any] = None

    # First-class topology / communication state (mutable; updated each tick).
    # Constructed lazily by the engine.
    topology: Optional[Any] = None

    # First-class utilization / packing / consolidation state (mutable; updated
    # each tick). Constructed lazily by the engine.
    util: Optional[Any] = None

    # First-class energy / carbon / arbitrage state (shift window, churn,
    # net-savings accounting). Constructed lazily by the engine.
    energy: Optional[Any] = None
    # Objective weights: objective = alpha*cost + beta*carbon.
    alpha_cost: float = 1.0
    beta_carbon: float = 0.0

    # Computed per tick
    effective_tokens_per_second: float = 0.0
    effective_requests_per_second: float = 0.0

    # Topology quality score (0-1, 1 = best)
    topology_score: float = 1.0


@dataclass
class SimRegion:
    """Region simulation state."""
    region_id: str

    # Energy price trace ($/MWh, one per tick; wraps if shorter than run)
    energy_price_trace: list[float] = field(default_factory=list)
    carbon_intensity_trace: list[float] = field(default_factory=list)

    # Current state
    current_energy_price: float = 50.0   # $/MWh (day-ahead / planning signal)
    current_carbon_intensity: Optional[float] = None

    # Day-ahead vs real-time settlement (realtime == day-ahead when basis is
    # disabled, preserving deterministic pricing). realtime is what realized
    # consumption actually pays. Updated each tick by _update_energy.
    day_ahead_price: float = 50.0
    realtime_price: float = 50.0

    # First-class energy/carbon market state (DA/RT basis, LMP components, carbon
    # forecast, spare capacity, telemetry). Constructed lazily by the engine.
    energy_state: Optional[Any] = None

    # Spike override — set by energy_price_spike event; prevents trace from clobbering it
    price_spike_active: bool = False
    # Congestion override — set by energy congestion events for the LMP model.
    congestion_active: bool = False
    grid_stress_active: bool = False

    # Ambient temperature proxy (affects GPU cooling)
    ambient_temp_c: float = 22.0
    ambient_temp_trace: list[float] = field(default_factory=list)

    # Network distance to other regions (ms RTT)
    network_latency_to: dict[str, float] = field(default_factory=dict)

    # Capacity
    total_gpus: int = 0
    available_gpus: int = 0

    nodes: list[SimNode] = field(default_factory=list)
    queues: list[SimQueue] = field(default_factory=list)


@dataclass
class SimCluster:
    """Full mutable cluster simulation state."""
    regions: dict[str, SimRegion] = field(default_factory=dict)
    workloads: dict[str, SimWorkload] = field(default_factory=dict)
    migration_log: list[dict[str, Any]] = field(default_factory=list)

    tick: int = 0
    tick_duration_hours: float = 1.0

    # Accumulated cost metrics
    total_energy_cost: float = 0.0
    total_tokens_served: int = 0
    total_energy_kwh: float = 0.0
    sla_violations: int = 0
    migration_count: int = 0


@dataclass
class SimulatorConfig:
    """Configuration for the cluster simulator."""
    scenario_name: str = "default"
    seed: int = 42
    tick_duration_hours: float = 1.0

    # Region configs
    regions: list[dict[str, Any]] = field(default_factory=list)

    # Workload configs
    workloads: list[dict[str, Any]] = field(default_factory=list)

    # Scenario event overrides (e.g., surge at tick 10)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Baseline mode: "fifo", "current_price_only", "greedy_energy", "sla_aware", "constraint_aware"
    baseline_mode: str = "fifo"

    # SLA config path (optional)
    sla_config_path: Optional[str] = None

    # Metadata for benchmark validation
    scenario_version: str = "v1"
    simulator_version: str = "1.0.0"

    # Serving-realism overrides (calibration param overrides + enable_bursts).
    # Bursts are OFF by default so the canonical single-constraint detection
    # scenarios keep deterministic arrivals; validation scenarios opt in.
    serving_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulatorConfig":
        return cls(
            scenario_name=d.get("scenario_name", "default"),
            seed=d.get("seed", 42),
            tick_duration_hours=d.get("tick_duration_hours", 1.0),
            regions=d.get("regions", []),
            workloads=d.get("workloads", []),
            events=d.get("events", []),
            baseline_mode=d.get("baseline_mode", "fifo"),
            sla_config_path=d.get("sla_config_path"),
            scenario_version=d.get("scenario_version", "v1"),
            simulator_version=d.get("simulator_version", "1.0.0"),
            serving_config=d.get("serving_config", {}),
        )
