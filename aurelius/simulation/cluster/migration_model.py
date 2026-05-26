"""Explicit, mutable migration / rerouting / drain / cold-start state models.

First-class simulator states required by the migration-realism upgrade. Mutable
(updated each tick by the engine), separate from the frozen ClusterState. A
single composite ``WorkloadMigrationState`` aggregates them and is attached to
each SimWorkload. All values are bounded proxies, not a control-plane simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DrainState:
    """Kubernetes-style drain progress for a workload's current placement."""
    draining: bool = False
    drain_seconds_total: float = 0.0     # T_evict + T_grace + T_rebind
    last_drain_seconds: float = 0.0


@dataclass
class PodEvictionState:
    """Pod eviction / restart bookkeeping."""
    cumulative_evictions: int = 0
    last_tick_evictions: int = 0


@dataclass
class PDBConstraintState:
    """PodDisruptionBudget: how many replicas may be disrupted right now."""
    available: int = 1                   # 0 → drain/migration blocked
    min_available: int = 0


@dataclass
class StartupState:
    """Cold-start decomposition of the most recent (re)start."""
    last_cold_seconds: float = 0.0
    t_node: float = 0.0
    t_pull: float = 0.0
    t_load: float = 0.0
    t_gpu_transfer: float = 0.0
    t_warmup: float = 0.0
    first_compile: bool = False


@dataclass
class ColdStartState:
    """Cold-start frequency + scale-from-zero flag."""
    cold_start_count: int = 0
    scale_from_zero: bool = False


@dataclass
class ReplicaWarmupState:
    """Reuse-driven warmup window after a (re)start; injects a TTFT penalty."""
    ticks_remaining: int = 0
    startup_penalty_ms: float = 0.0      # decays over the warmup window
    warm: bool = True


@dataclass
class WarmPoolState:
    """Warm-pool replicas: pre-loaded, ready, idle-but-costly."""
    size: int = 0                        # replicas kept warm
    occupancy: float = 0.0               # fraction currently absorbing traffic


@dataclass
class RouteShiftState:
    """Route churn bookkeeping (recent reroutes drive batching disruption)."""
    reroute_count: int = 0
    churn_rate: float = 0.0              # decayed recent-reroute intensity


@dataclass
class ProxyQueueState:
    """Node-level proxy / ingress saturation."""
    saturation_factor: float = 1.0       # ≥ 1.0; queue amplification


@dataclass
class BatchCohortState:
    """Continuous-batching cohort health under churn."""
    efficiency: float = 1.0              # η_batch ∈ (0, 1]


@dataclass
class TrafficShiftState:
    """Phased traffic-shift fraction for a rollout."""
    fraction: float = 1.0                # fraction of traffic on the new route


@dataclass
class RolloutState:
    """Phased rollout (canary/blue-green) progress + rollback bookkeeping."""
    active: bool = False
    phase: int = 0
    hold_ticks_remaining: int = 0
    instability: float = 0.0             # [0,1] drives veto + tail uplift
    rollback_count: int = 0


@dataclass
class TailInstabilityState:
    """Migration-induced p95/p99 uplift currently in effect."""
    uplift_mult: float = 1.0


@dataclass
class MigrationState:
    """Top-level migration bookkeeping + last cost breakdown."""
    migration_count: int = 0
    veto_count: int = 0
    last_veto_reason: Optional[str] = None
    last_cost_ms: float = 0.0
    overload_events: int = 0


@dataclass
class WorkloadMigrationState:
    """Composite per-workload migration/rerouting/drain/cold-start state."""
    migration: MigrationState = field(default_factory=MigrationState)
    drain: DrainState = field(default_factory=DrainState)
    eviction: PodEvictionState = field(default_factory=PodEvictionState)
    pdb: PDBConstraintState = field(default_factory=PDBConstraintState)
    startup: StartupState = field(default_factory=StartupState)
    coldstart: ColdStartState = field(default_factory=ColdStartState)
    warmup: ReplicaWarmupState = field(default_factory=ReplicaWarmupState)
    warm_pool: WarmPoolState = field(default_factory=WarmPoolState)
    route_shift: RouteShiftState = field(default_factory=RouteShiftState)
    proxy: ProxyQueueState = field(default_factory=ProxyQueueState)
    cohort: BatchCohortState = field(default_factory=BatchCohortState)
    traffic_shift: TrafficShiftState = field(default_factory=TrafficShiftState)
    rollout: RolloutState = field(default_factory=RolloutState)
    tail: TailInstabilityState = field(default_factory=TailInstabilityState)

    # serving engine runtime (drives cold-start profile)
    engine_runtime: str = "vllm"
