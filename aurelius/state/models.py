"""Normalized cluster state models for constraint-aware GPU orchestration.

Design principles:
- All optional metrics default to None, never fabricated as 0.
- All timestamps must be UTC-aware (naive datetimes are rejected).
- Percentages are validated in [0, 100].
- Byte counts, rates, and latencies must be >= 0 when present.
- JSON round-trip via Pydantic .model_dump() / .model_validate().
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LinkType(str, Enum):
    NVLINK = "NVLINK"
    NVSWITCH = "NVSWITCH"
    PCIE = "PCIe"
    PIX = "PIX"
    PHB = "PHB"
    SYS = "SYS"
    NODE = "NODE"
    RACK = "RACK"
    REGION = "REGION"
    UNKNOWN = "UNKNOWN"


class RuntimeType(str, Enum):
    VLLM = "vllm"
    TRITON = "triton"
    RAY_SERVE = "ray_serve"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class WorkloadType(str, Enum):
    INFERENCE = "inference"
    BATCH_TRAINING = "batch_training"
    FINE_TUNING = "fine_tuning"
    EMBEDDING = "embedding"
    OTHER = "other"


class CommunicationIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class MemoryIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ConstraintType(str, Enum):
    ENERGY_BOUND = "energy_bound"
    THERMAL_BOUND = "thermal_bound"
    QUEUE_BOUND = "queue_bound"
    LATENCY_BOUND = "latency_bound"
    COMMUNICATION_BOUND = "communication_bound"
    MEMORY_BOUND_INDIRECT = "memory_bound_indirect"
    TOPOLOGY_BOUND = "topology_bound"
    UTILIZATION_BOUND = "utilization_bound"
    UNKNOWN = "unknown"
    INSUFFICIENT_DATA = "insufficient_data"


class ImplementationMode(str, Enum):
    RECOMMENDATION_ONLY = "recommendation_only"
    DRY_RUN = "dry_run"
    EXECUTABLE = "executable"


# ---------------------------------------------------------------------------
# Validators (reusable)
# ---------------------------------------------------------------------------

def _require_utc(v: Optional[datetime]) -> Optional[datetime]:
    """Reject naive datetimes; return UTC-aware as-is."""
    if v is None:
        return v
    if v.tzinfo is None:
        raise ValueError(
            f"Naive datetime not allowed: {v!r}. "
            "Use datetime.now(timezone.utc) or attach tzinfo=timezone.utc."
        )
    return v


def _validate_pct(v: Optional[float]) -> Optional[float]:
    """Accept None or [0, 100]; reject outside range."""
    if v is None:
        return v
    if not (0.0 <= v <= 100.0):
        raise ValueError(f"Percentage must be in [0, 100], got {v}")
    return v


def _validate_non_neg(v: Optional[float]) -> Optional[float]:
    """Accept None or >= 0; reject negatives."""
    if v is None:
        return v
    if v < 0.0:
        raise ValueError(f"Value must be >= 0, got {v}")
    return v


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Tracks the origin and freshness of a metric or state snapshot."""

    source: str = Field(description="Connector identifier, e.g. 'prometheus:dcgm'")
    collected_at: datetime = Field(description="UTC-aware timestamp of collection")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0.0–1.0; None if entirely unknown",
    )
    staleness_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Seconds since data was last observed fresh",
    )

    @field_validator("collected_at", mode="before")
    @classmethod
    def _require_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return _require_utc(v)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# RegionState
# ---------------------------------------------------------------------------

class RegionState(BaseModel):
    """Per-region energy, capacity, and carbon state."""

    region_id: str
    energy_price_per_mwh: Optional[float] = Field(
        default=None, ge=0.0, description="Current spot price $/MWh"
    )
    day_ahead_price_per_mwh: Optional[float] = Field(default=None, ge=0.0)
    real_time_price_per_mwh: Optional[float] = Field(default=None, ge=0.0)
    carbon_intensity_gco2_per_kwh: Optional[float] = Field(default=None, ge=0.0)
    renewable_fraction_pct: Optional[float] = None
    available_capacity_gpu: Optional[int] = Field(default=None, ge=0)
    total_capacity_gpu: Optional[int] = Field(default=None, ge=0)
    weather_temp_celsius: Optional[float] = None
    pue: Optional[float] = Field(default=None, ge=1.0)
    provenance: Optional[Provenance] = None

    @field_validator("renewable_fraction_pct", mode="before")
    @classmethod
    def _pct(cls, v: Any) -> Optional[float]:
        return _validate_pct(v)


# ---------------------------------------------------------------------------
# NodeState
# ---------------------------------------------------------------------------

class NodeState(BaseModel):
    """Per-node hardware and topology state."""

    node_id: str
    region_id: Optional[str] = None
    zone: Optional[str] = None
    rack_id: Optional[str] = None
    instance_type: Optional[str] = None
    gpu_count: Optional[int] = Field(default=None, ge=0)
    cpu_count: Optional[int] = Field(default=None, ge=0)
    memory_total_bytes: Optional[int] = Field(default=None, ge=0)
    labels: dict[str, str] = Field(default_factory=dict)
    taints: list[dict[str, str]] = Field(default_factory=list)
    ready: Optional[bool] = None
    unschedulable: Optional[bool] = None
    allocatable_gpu: Optional[int] = Field(default=None, ge=0)
    allocatable_cpu_millicores: Optional[int] = Field(default=None, ge=0)
    allocatable_memory_bytes: Optional[int] = Field(default=None, ge=0)
    provenance: Optional[Provenance] = None


# ---------------------------------------------------------------------------
# GPUState
# ---------------------------------------------------------------------------

class GPUState(BaseModel):
    """Per-GPU telemetry state.

    All metric fields are Optional[float|int]. None = not available.
    Never use 0 as a placeholder for unknown data.
    """

    gpu_id: str
    uuid: Optional[str] = None
    node_id: Optional[str] = None
    model: Optional[str] = None
    index: Optional[int] = Field(default=None, ge=0)
    pci_bus_id: Optional[str] = None

    # Utilization
    utilization_pct: Optional[float] = None
    sm_activity_pct: Optional[float] = None

    # Memory
    memory_used_bytes: Optional[int] = Field(default=None, ge=0)
    memory_total_bytes: Optional[int] = Field(default=None, ge=0)
    memory_bandwidth_util_pct: Optional[float] = None

    # Power
    power_watts: Optional[float] = Field(default=None, ge=0.0)
    power_limit_watts: Optional[float] = Field(default=None, ge=0.0)

    # Thermal
    temperature_c: Optional[float] = None
    thermal_throttle_active: Optional[bool] = None
    thermal_slowdown_active: Optional[bool] = None

    # Errors
    xid_error_count: Optional[int] = Field(default=None, ge=0)

    # NVLink counters
    nvlink_rx_bytes_per_sec: Optional[float] = Field(default=None, ge=0.0)
    nvlink_tx_bytes_per_sec: Optional[float] = Field(default=None, ge=0.0)

    # PCIe counters
    pcie_rx_bytes_per_sec: Optional[float] = Field(default=None, ge=0.0)
    pcie_tx_bytes_per_sec: Optional[float] = Field(default=None, ge=0.0)

    # Workload assignment
    assigned_workload_ids: list[str] = Field(default_factory=list)

    provenance: Optional[Provenance] = None

    @field_validator("utilization_pct", "sm_activity_pct", "memory_bandwidth_util_pct", mode="before")
    @classmethod
    def _pct(cls, v: Any) -> Optional[float]:
        return _validate_pct(v)

    @field_validator("temperature_c", mode="before")
    @classmethod
    def _temp(cls, v: Any) -> Optional[float]:
        if v is None:
            return v
        v = float(v)
        if v < -273.15:
            raise ValueError(f"temperature_c below absolute zero: {v}")
        return v


# ---------------------------------------------------------------------------
# InferenceServiceState
# ---------------------------------------------------------------------------

class InferenceServiceState(BaseModel):
    """Per-inference-service telemetry (vLLM, Triton, Ray Serve, etc.)."""

    service_id: str
    runtime: RuntimeType = RuntimeType.UNKNOWN
    node_ids: list[str] = Field(default_factory=list)
    gpu_ids: list[str] = Field(default_factory=list)

    # Throughput
    requests_per_second: Optional[float] = Field(default=None, ge=0.0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0.0)

    # TTFT (time to first token)
    ttft_p50_ms: Optional[float] = Field(default=None, ge=0.0)
    ttft_p95_ms: Optional[float] = Field(default=None, ge=0.0)
    ttft_p99_ms: Optional[float] = Field(default=None, ge=0.0)

    # TPOT (time per output token)
    tpot_p50_ms: Optional[float] = Field(default=None, ge=0.0)
    tpot_p95_ms: Optional[float] = Field(default=None, ge=0.0)
    tpot_p99_ms: Optional[float] = Field(default=None, ge=0.0)

    # Overall latency
    latency_p50_ms: Optional[float] = Field(default=None, ge=0.0)
    latency_p95_ms: Optional[float] = Field(default=None, ge=0.0)
    latency_p99_ms: Optional[float] = Field(default=None, ge=0.0)

    # Queue
    queue_depth: Optional[int] = Field(default=None, ge=0)
    queue_wait_p95_ms: Optional[float] = Field(default=None, ge=0.0)

    # Batch/sequence state
    active_sequences: Optional[int] = Field(default=None, ge=0)
    batch_size: Optional[int] = Field(default=None, ge=0)

    # Error/timeout rates
    timeout_rate_pct: Optional[float] = None
    error_rate_pct: Optional[float] = None

    # Cache metrics (read-only proxy; Aurelius never manages KV cache directly)
    kv_cache_usage_pct: Optional[float] = None
    prefix_cache_hit_rate_pct: Optional[float] = None

    provenance: Optional[Provenance] = None

    @field_validator(
        "timeout_rate_pct", "error_rate_pct",
        "kv_cache_usage_pct", "prefix_cache_hit_rate_pct",
        mode="before",
    )
    @classmethod
    def _pct(cls, v: Any) -> Optional[float]:
        return _validate_pct(v)

    @model_validator(mode="after")
    def _latency_ordering(self) -> "InferenceServiceState":
        """Warn (not hard-fail) if p50 > p99 for same metric class."""
        for prefix in ("ttft", "tpot", "latency"):
            p50 = getattr(self, f"{prefix}_p50_ms")
            p99 = getattr(self, f"{prefix}_p99_ms")
            if p50 is not None and p99 is not None and p50 > p99:
                raise ValueError(
                    f"{prefix}_p50_ms ({p50}) > {prefix}_p99_ms ({p99}): "
                    "impossible latency ordering"
                )
        return self


# ---------------------------------------------------------------------------
# WorkloadState
# ---------------------------------------------------------------------------

class WorkloadState(BaseModel):
    """A running or queued workload placement description."""

    workload_id: str
    service_id: Optional[str] = None
    workload_type: WorkloadType = WorkloadType.INFERENCE
    priority_tier: int = Field(default=0, ge=0)

    current_region: Optional[str] = None
    current_node_ids: list[str] = Field(default_factory=list)
    current_gpu_ids: list[str] = Field(default_factory=list)
    gpu_count_required: int = Field(default=1, ge=1)

    flexibility_window_minutes: Optional[float] = Field(default=None, ge=0.0)
    migration_allowed: bool = True

    communication_intensity: CommunicationIntensity = CommunicationIntensity.UNKNOWN
    memory_intensity: MemoryIntensity = MemoryIntensity.UNKNOWN
    latency_sensitive: bool = False

    sla_policy_id: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# QueueState
# ---------------------------------------------------------------------------

class QueueState(BaseModel):
    """Queue metrics for a service or scheduler."""

    queue_id: str
    service_id: Optional[str] = None
    pending_jobs: Optional[int] = Field(default=None, ge=0)
    queue_depth: Optional[int] = Field(default=None, ge=0)
    oldest_pending_age_sec: Optional[float] = Field(default=None, ge=0.0)
    p95_wait_ms: Optional[float] = Field(default=None, ge=0.0)
    arrival_rate_per_sec: Optional[float] = Field(default=None, ge=0.0)
    service_rate_per_sec: Optional[float] = Field(default=None, ge=0.0)
    provenance: Optional[Provenance] = None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class TopologyNode(BaseModel):
    """A node in the topology graph (GPU, NIC, server, rack, region)."""

    node_id: str
    node_type: str = "gpu"
    parent_id: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)


class TopologyLink(BaseModel):
    """A link between two topology nodes."""

    source_id: str
    target_id: str
    link_type: LinkType = LinkType.UNKNOWN
    bandwidth_gbps: Optional[float] = Field(default=None, ge=0.0)
    latency_us: Optional[float] = Field(default=None, ge=0.0)
    bidirectional: bool = True


class TopologyState(BaseModel):
    """GPU/NIC/node/rack/region topology graph."""

    nodes: dict[str, TopologyNode] = Field(default_factory=dict)
    links: list[TopologyLink] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    def topology_distance(self, id_a: str, id_b: str) -> Optional[int]:
        """BFS hop count between two nodes. Returns None if unreachable."""
        if id_a == id_b:
            return 0
        adjacency: dict[str, list[str]] = {}
        for link in self.links:
            adjacency.setdefault(link.source_id, []).append(link.target_id)
            if link.bidirectional:
                adjacency.setdefault(link.target_id, []).append(link.source_id)

        visited = {id_a}
        frontier = [id_a]
        depth = 0
        while frontier:
            depth += 1
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency.get(node, []):
                    if neighbor == id_b:
                        return depth
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return None


# ---------------------------------------------------------------------------
# EnergyState
# ---------------------------------------------------------------------------

class EnergyState(BaseModel):
    """Region-level energy pricing and carbon state."""

    region_id: str
    current_price_per_mwh: Optional[float] = Field(default=None, ge=0.0)
    forecast_prices: list[dict[str, Any]] = Field(default_factory=list)
    carbon_intensity_gco2_per_kwh: Optional[float] = Field(default=None, ge=0.0)
    renewable_fraction_pct: Optional[float] = None
    provenance: Optional[Provenance] = None

    @field_validator("renewable_fraction_pct", mode="before")
    @classmethod
    def _pct(cls, v: Any) -> Optional[float]:
        return _validate_pct(v)


# ---------------------------------------------------------------------------
# ThermalState
# ---------------------------------------------------------------------------

class ThermalState(BaseModel):
    """Node/rack-level thermal state."""

    node_id: str
    gpu_temps_c: dict[str, float] = Field(default_factory=dict)
    rack_inlet_temp_c: Optional[float] = None
    rack_outlet_temp_c: Optional[float] = None
    throttle_events_per_min: Optional[float] = Field(default=None, ge=0.0)
    cooling_efficiency_pct: Optional[float] = None
    provenance: Optional[Provenance] = None

    @field_validator("cooling_efficiency_pct", mode="before")
    @classmethod
    def _pct(cls, v: Any) -> Optional[float]:
        return _validate_pct(v)


# ---------------------------------------------------------------------------
# MigrationHistory
# ---------------------------------------------------------------------------

class MigrationEvent(BaseModel):
    """A single workload migration event."""

    occurred_at: datetime
    workload_id: str
    from_node_id: Optional[str] = None
    to_node_id: Optional[str] = None
    from_region: Optional[str] = None
    to_region: Optional[str] = None
    reason: Optional[str] = None
    latency_penalty_ms: Optional[float] = Field(default=None, ge=0.0)
    cache_warmup_penalty_ms: Optional[float] = Field(default=None, ge=0.0)
    triggered_by: Optional[str] = None

    @field_validator("occurred_at", mode="before")
    @classmethod
    def _require_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return _require_utc(v)  # type: ignore[return-value]


class MigrationHistory(BaseModel):
    """History of migrations for a workload."""

    workload_id: str
    migrations: list[MigrationEvent] = Field(default_factory=list)

    @property
    def migration_count(self) -> int:
        return len(self.migrations)

    def migrations_in_window(self, since: datetime, until: datetime) -> list[MigrationEvent]:
        return [
            m for m in self.migrations
            if since <= m.occurred_at <= until
        ]


# ---------------------------------------------------------------------------
# ConstraintAssessment
# ---------------------------------------------------------------------------

class ConstraintAssessment(BaseModel):
    """Output of the binding constraint classifier."""

    assessed_at: datetime
    cluster_id: str
    primary_constraint: ConstraintType = ConstraintType.INSUFFICIENT_DATA
    secondary_constraints: list[ConstraintType] = Field(default_factory=list)
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-constraint pressure score 0–1",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_metrics: dict[str, Any] = Field(default_factory=dict)
    missing_metrics: list[str] = Field(default_factory=list)
    recommended_safe_action_types: list[str] = Field(default_factory=list)
    disallowed_action_types: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None

    @field_validator("assessed_at", mode="before")
    @classmethod
    def _require_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return _require_utc(v)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    """A single optimization recommendation from the constraint-aware engine."""

    recommendation_id: str
    generated_at: datetime
    action: str
    target_workload_id: Optional[str] = None
    target_service_id: Optional[str] = None
    current_state_summary: dict[str, Any] = Field(default_factory=dict)
    proposed_state_summary: dict[str, Any] = Field(default_factory=dict)
    primary_constraint_addressed: ConstraintType = ConstraintType.UNKNOWN
    expected_impact: dict[str, Any] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    sla_check_result: Optional[str] = None
    why_not_alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    implementation_mode: ImplementationMode = ImplementationMode.RECOMMENDATION_ONLY

    @field_validator("generated_at", mode="before")
    @classmethod
    def _require_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return _require_utc(v)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ClusterState
# ---------------------------------------------------------------------------

class ClusterState(BaseModel):
    """Top-level normalized cluster state snapshot.

    This is the canonical internal representation that all connectors
    (Prometheus, Kubernetes, topology, simulator) populate.
    """

    timestamp: datetime = Field(description="UTC-aware snapshot timestamp")
    cluster_id: str = Field(default="default")
    regions: dict[str, RegionState] = Field(default_factory=dict)
    nodes: dict[str, NodeState] = Field(default_factory=dict)
    gpus: dict[str, GPUState] = Field(default_factory=dict)
    services: dict[str, InferenceServiceState] = Field(default_factory=dict)
    workloads: dict[str, WorkloadState] = Field(default_factory=dict)
    queues: dict[str, QueueState] = Field(default_factory=dict)
    topology: Optional[TopologyState] = None
    energy: dict[str, EnergyState] = Field(default_factory=dict)
    thermal: dict[str, ThermalState] = Field(default_factory=dict)
    migration_history: dict[str, MigrationHistory] = Field(default_factory=dict)
    provenance: Optional[Provenance] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _require_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return _require_utc(v)  # type: ignore[return-value]

    def gpu_count(self) -> int:
        return len(self.gpus)

    def node_count(self) -> int:
        return len(self.nodes)

    def service_count(self) -> int:
        return len(self.services)

    def utilization_pcts(self) -> list[float]:
        """Return utilization_pct values for all GPUs where present."""
        return [g.utilization_pct for g in self.gpus.values() if g.utilization_pct is not None]

    def mean_gpu_utilization(self) -> Optional[float]:
        vals = self.utilization_pcts()
        if not vals:
            return None
        return sum(vals) / len(vals)

    def gpu_temperatures(self) -> list[float]:
        return [g.temperature_c for g in self.gpus.values() if g.temperature_c is not None]

    def max_gpu_temperature(self) -> Optional[float]:
        temps = self.gpu_temperatures()
        return max(temps) if temps else None
