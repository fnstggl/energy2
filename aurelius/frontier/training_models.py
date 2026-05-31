"""Training Safe Utilization Frontier — v1 models.

A **sibling** of the serving Safe Utilization Frontier Controller
(``aurelius/frontier/``) for training / fine-tuning / GPU-batch
workloads. Where the serving controller optimizes
**rho** (request-rate utilization) subject to latency/queue/SLA
constraints, this module optimizes **GPU occupancy / packing density /
backfill aggressiveness** subject to **queue wait / starvation /
fragmentation / gang-scheduling failure / retry-waste** constraints.

Hard invariants (asserted by tests):

- Training Frontier does **NOT** import / extend the serving rho
  controller (``aurelius/frontier/controller.py``,
  ``aurelius/frontier/dynamic_controller.py``). Training candidates are
  packing / backfill / reservation / gang-scheduling descriptors, not a
  scalar rho.
- ``TrainingFrontierDecision.executable_in_real_cluster`` is ``False``
  at construction; real-cluster execution is **disabled by default**.
- All dataclasses are JSON round-trippable and stdlib-only.
- Missing telemetry stays ``None`` — never silently zero-filled
  (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1).
- No production mutation. No ML training. No new datasets ingested.

Directional simulator / public-trace evidence only — NOT production
savings (``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Categorical enums — sibling of the serving FrontierAction / SafetyStatus.
# ---------------------------------------------------------------------------

class TrainingSafetyStatus:
    """Categorical safety verdict for a single training frontier point."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


TRAINING_SAFETY_STATUSES = frozenset({
    TrainingSafetyStatus.SAFE,
    TrainingSafetyStatus.UNSAFE,
    TrainingSafetyStatus.INSUFFICIENT_TELEMETRY,
})


class TrainingFrontierAction:
    """Categorical recommendation produced by the training-frontier
    controller. NOT a copy of the serving rho actions — the action
    space is workload-shape-aware (reserve_for_large_jobs is a
    training-only verb)."""

    RECOMMEND_TRAINING_FRONTIER = "RECOMMEND_TRAINING_FRONTIER"
    KEEP_CURRENT_POLICY = "KEEP_CURRENT_POLICY"
    LOWER_PACKING_PRESSURE = "LOWER_PACKING_PRESSURE"
    RESERVE_FOR_LARGE_JOBS = "RESERVE_FOR_LARGE_JOBS"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


TRAINING_FRONTIER_ACTIONS = frozenset({
    TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER,
    TrainingFrontierAction.KEEP_CURRENT_POLICY,
    TrainingFrontierAction.LOWER_PACKING_PRESSURE,
    TrainingFrontierAction.RESERVE_FOR_LARGE_JOBS,
    TrainingFrontierAction.INSUFFICIENT_TELEMETRY,
})


# Execution modes (shared with serving but redefined locally so the
# training models do not import from the serving controller).
EXECUTION_MODE_SHADOW = "shadow"
EXECUTION_MODE_SIMULATOR = "simulator"
EXECUTION_MODE_REAL_DISABLED = "real_disabled"
EXECUTION_MODE_REAL_ENABLED = "real_enabled"
_TRAINING_EXECUTION_MODES = frozenset({
    EXECUTION_MODE_SHADOW, EXECUTION_MODE_SIMULATOR,
    EXECUTION_MODE_REAL_DISABLED, EXECUTION_MODE_REAL_ENABLED,
})

# Allowed workload-type labels for ``TrainingWorkloadProfile``.
TRAINING_WORKLOAD_TYPES = frozenset({
    "training", "fine_tuning", "gpu_batch", "mixed_training",
})

# Allowed trace_source labels.
TRAINING_TRACE_SOURCES = frozenset({
    "philly", "alibaba_gpu", "synthetic_fixture",
})

# Telemetry-confidence ordering (mirrors serving safety.py / risk.py).
_TRAINING_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


class TrainingFrontierSchemaError(ValueError):
    """Raised when a training-frontier model receives a structurally
    invalid value."""


# ---------------------------------------------------------------------------
# Per-workload profile.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingWorkloadProfile:
    """Per-workload context for the training-frontier controller.

    Identifies the workload and declares its safety budgets. Every
    numeric SLA is ``Optional`` — a missing field means "unknown", never
    zero. ``gpu_count_required_distribution`` and
    ``duration_distribution_s`` are dicts the caller may use to capture
    distributional summaries (e.g. {"p50": 1, "p95": 8, "p99": 16}).
    """

    workload_id: str
    workload_type: str
    trace_source: str
    gpu_count_required_distribution: dict = field(default_factory=dict)
    duration_distribution_s: dict = field(default_factory=dict)
    queue_wait_sla_p95_s: Optional[float] = None
    queue_wait_sla_p99_s: Optional[float] = None
    completion_time_sla_s: Optional[float] = None
    starvation_sla_pct: Optional[float] = None
    fragmentation_sla_pct: Optional[float] = None
    gang_failure_sla_pct: Optional[float] = None
    telemetry_confidence: str = "unknown"
    source: str = "unknown"

    def __post_init__(self):
        if self.workload_type not in TRAINING_WORKLOAD_TYPES:
            raise TrainingFrontierSchemaError(
                f"unknown workload_type {self.workload_type!r}; "
                f"expected one of {sorted(TRAINING_WORKLOAD_TYPES)}")
        if self.trace_source not in TRAINING_TRACE_SOURCES:
            raise TrainingFrontierSchemaError(
                f"unknown trace_source {self.trace_source!r}; "
                f"expected one of {sorted(TRAINING_TRACE_SOURCES)}")
        if self.telemetry_confidence not in _TRAINING_CONF_RANK:
            raise TrainingFrontierSchemaError(
                f"unknown telemetry_confidence "
                f"{self.telemetry_confidence!r}; "
                f"expected one of {sorted(_TRAINING_CONF_RANK)}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate descriptor.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingFrontierCandidate:
    """Descriptor for one operating point on the training frontier.

    These are the **knobs the training scheduler / packer might adjust**.
    Each field is ``Optional`` — pilot telemetry may declare a subset.
    The descriptor is paired with measured outcomes in
    :class:`TrainingFrontierPoint`; this object does NOT carry KPIs.

    Field semantics:

    - ``occupancy_target`` — desired fleet-wide GPU occupancy fraction
      (e.g. 0.85 = run hot).
    - ``packing_density_target`` — desired per-active-node packing
      density (1.0 = pack to last GPU, 0.5 = leave headroom).
    - ``backfill_aggressiveness`` — 0 = head-of-line strict FIFO,
      1 = always backfill.
    - ``large_job_reservation_fraction`` — fraction of fleet reserved
      for jobs above a size threshold (gang-scheduling reservation).
    - ``fragmentation_budget`` — tolerated fragmentation fraction.
    - ``gang_scheduling_strictness`` — 0 = relaxed, 1 = strict
      (atomic-or-nothing for multi-GPU jobs).
    - ``preemption_allowed`` — whether the scheduler may preempt
      running jobs to satisfy higher-priority demand.
    - ``checkpoint_overhead_budget`` — tolerated checkpoint-overhead
      fraction.
    - ``heterogeneity_preference`` — pack onto a single GPU type
      ("homogeneous") or spread ("heterogeneous") or "any".
    - ``price_aware_gpu_routing_enabled`` — route jobs to the cheapest
      GPU type that meets their requirements.
    """

    occupancy_target: Optional[float] = None
    packing_density_target: Optional[float] = None
    backfill_aggressiveness: Optional[float] = None
    large_job_reservation_fraction: Optional[float] = None
    fragmentation_budget: Optional[float] = None
    gang_scheduling_strictness: Optional[float] = None
    preemption_allowed: Optional[bool] = None
    checkpoint_overhead_budget: Optional[float] = None
    heterogeneity_preference: Optional[str] = None
    price_aware_gpu_routing_enabled: Optional[bool] = None
    # Free-form label naming the underlying policy / measurement source
    # so reports can trace each candidate to the policy that produced
    # its predicted metrics.
    source_policy: Optional[str] = None

    def __post_init__(self):
        for name in ("occupancy_target", "packing_density_target",
                     "backfill_aggressiveness",
                     "large_job_reservation_fraction",
                     "fragmentation_budget", "gang_scheduling_strictness",
                     "checkpoint_overhead_budget"):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 1.0 + 1e-9):
                raise TrainingFrontierSchemaError(
                    f"{name} must be in [0,1]; got {v}")
        if (self.heterogeneity_preference is not None
                and self.heterogeneity_preference not in
                ("homogeneous", "heterogeneous", "any")):
            raise TrainingFrontierSchemaError(
                f"unknown heterogeneity_preference "
                f"{self.heterogeneity_preference!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frontier point — candidate + measured / predicted outcome.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingFrontierPoint:
    """One point on the training-utilization frontier.

    ``predicted_*`` fields carry the measured (or estimator-replayed)
    outcomes; ``safety_status`` is the verdict against
    :class:`TrainingSafetyConfig`. ``predicted_*`` fields stay ``None``
    when the trace does not provide the signal (e.g. queue-wait
    metrics on the Alibaba GPU packing trace, which has no per-job
    submit timestamps for the static packing baseline).
    """

    candidate: TrainingFrontierCandidate
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_gpu_occupancy: Optional[float] = None
    predicted_packing_density: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    predicted_completed_work: Optional[float] = None
    predicted_queue_wait_p95_s: Optional[float] = None
    predicted_queue_wait_p99_s: Optional[float] = None
    predicted_job_completion_p95_s: Optional[float] = None
    predicted_job_completion_p99_s: Optional[float] = None
    predicted_starvation_rate_pct: Optional[float] = None
    predicted_fragmentation_block_rate_pct: Optional[float] = None
    predicted_gang_scheduling_failure_pct: Optional[float] = None
    predicted_backfill_success_rate_pct: Optional[float] = None
    predicted_retry_waste_gpu_hours: Optional[float] = None
    predicted_cost: Optional[float] = None
    safety_status: str = TrainingSafetyStatus.INSUFFICIENT_TELEMETRY
    safety_vetoes: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.safety_status not in TRAINING_SAFETY_STATUSES:
            raise TrainingFrontierSchemaError(
                f"unknown safety_status {self.safety_status!r}")
        # Percentage-like fields. ``backfill_success_rate_pct`` is a
        # true fraction-of-jobs in [0,100]; the *_block_rate_pct and
        # *_failure_pct counters report events-per-job and may exceed
        # 100% on traces where one job triggers multiple events (the
        # safety gate catches the breach explicitly — no silent clip).
        for name in ("predicted_starvation_rate_pct",
                     "predicted_backfill_success_rate_pct"):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 100.0 + 1e-9):
                raise TrainingFrontierSchemaError(
                    f"{name} must be in [0,100]; got {v}")
        for name in ("predicted_fragmentation_block_rate_pct",
                     "predicted_gang_scheduling_failure_pct"):
            v = getattr(self, name)
            if v is not None and v < 0.0:
                raise TrainingFrontierSchemaError(
                    f"{name} must be >= 0; got {v}")

    @property
    def is_safe(self) -> bool:
        return self.safety_status == TrainingSafetyStatus.SAFE

    @property
    def is_insufficient_telemetry(self) -> bool:
        return self.safety_status == TrainingSafetyStatus.INSUFFICIENT_TELEMETRY

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate"] = self.candidate.to_dict()
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d


# ---------------------------------------------------------------------------
# Decision.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingFrontierDecision:
    """The training-frontier controller's recommendation for one workload.

    Recommendation-only at construction (``executable_in_real_cluster
    = False``). Real-cluster execution requires the same explicit opt-in
    as the serving controller's ``execute_frontier_decision`` shim and
    a non-stub executor (the training v1 ships only a stub).
    """

    workload_id: str
    selected_candidate: Optional[TrainingFrontierCandidate]
    current_candidate: Optional[TrainingFrontierCandidate]
    selected_point: Optional[TrainingFrontierPoint]
    frontier_points: tuple
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    expected_queue_wait_delta_s: Optional[float] = None
    expected_fragmentation_delta_pct: Optional[float] = None
    expected_starvation_delta_pct: Optional[float] = None
    confidence: str = "unknown"
    execution_mode: str = EXECUTION_MODE_SHADOW
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False
    safety_vetoes: tuple = ()
    source: str = "training_frontier_v1"
    notes: tuple = ()

    def __post_init__(self):
        if self.action not in TRAINING_FRONTIER_ACTIONS:
            raise TrainingFrontierSchemaError(
                f"unknown training action {self.action!r}; "
                f"expected one of {sorted(TRAINING_FRONTIER_ACTIONS)}")
        if self.execution_mode not in _TRAINING_EXECUTION_MODES:
            raise TrainingFrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}; "
                f"expected one of {sorted(_TRAINING_EXECUTION_MODES)}")
        if self.executable_in_real_cluster:
            raise TrainingFrontierSchemaError(
                "training-frontier decisions are recommendation-only at "
                "construction; real execution requires the caller to pass "
                "allow_real_execution=True to "
                "execute_training_frontier_decision (which itself ships a "
                "no-op stub by default)")
        if self.confidence not in _TRAINING_CONF_RANK:
            raise TrainingFrontierSchemaError(
                f"unknown confidence {self.confidence!r}")

    @property
    def is_actionable(self) -> bool:
        return self.action in (
            TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER,
            TrainingFrontierAction.LOWER_PACKING_PRESSURE,
            TrainingFrontierAction.RESERVE_FOR_LARGE_JOBS,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["selected_candidate"] = (self.selected_candidate.to_dict()
                                   if self.selected_candidate is not None
                                   else None)
        d["current_candidate"] = (self.current_candidate.to_dict()
                                  if self.current_candidate is not None
                                  else None)
        d["selected_point"] = (self.selected_point.to_dict()
                               if self.selected_point is not None else None)
        d["frontier_points"] = [
            p.to_dict() if isinstance(p, TrainingFrontierPoint) else p
            for p in self.frontier_points]
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d
