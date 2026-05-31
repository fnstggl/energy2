"""Batch Inference Frontier v1 — models.

A **sibling** of the serving Safe Utilization Frontier (rho controller) and
the Training Safe Utilization Frontier. The candidate space is
multi-dimensional — ``(batch_window_seconds, batch_concurrency, target_rho,
deadline_slack_seconds)`` — and the safety floor is a **deadline-miss rate**
plus the existing serving timeout/queue gates (so a batch frontier can never
break the interactive baseline's safety floor).

Hard invariants (asserted by tests):

- Batch Inference Frontier does **NOT** import or mutate the serving rho
  controller (``aurelius/frontier/controller.py``,
  ``aurelius/frontier/dynamic_controller.py``). The candidate descriptor is
  multi-dimensional, not a scalar rho.
- ``BatchInferenceFrontierDecision.executable_in_real_cluster`` is ``False``
  at construction; real execution is **disabled by default** and the v1
  ships only a stub.
- The candidate ``deadline_slack_seconds`` is a **synthetic scenario knob** —
  Azure LLM 2024 and BurstGPT do NOT carry per-request deadlines, so the v1
  labels every Azure-2024-replay candidate as
  ``synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1"`` (or
  the equivalent BurstGPT label). The frontier never invents a real deadline
  it didn't read from the trace.
- Missing telemetry stays ``None`` — never silently zero-filled.

Simulator / public-trace evidence only — **NOT production savings**
(``docs/RESULTS.md`` §8). Real cluster execution remains disabled by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Categorical enums.
# ---------------------------------------------------------------------------

class BatchInferenceSafetyStatus:
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


BATCH_SAFETY_STATUSES = frozenset({
    BatchInferenceSafetyStatus.SAFE,
    BatchInferenceSafetyStatus.UNSAFE,
    BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY,
})


class BatchInferenceFrontierAction:
    """Categorical recommendation produced by the batch-inference-frontier
    controller. Like the training-frontier actions, these are workload-aware
    verbs — NOT a copy of the serving rho actions."""

    RECOMMEND_BATCH_FRONTIER = "RECOMMEND_BATCH_FRONTIER"
    KEEP_CURRENT_BATCH_POLICY = "KEEP_CURRENT_BATCH_POLICY"
    LOWER_BATCH_PRESSURE = "LOWER_BATCH_PRESSURE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


BATCH_FRONTIER_ACTIONS = frozenset({
    BatchInferenceFrontierAction.RECOMMEND_BATCH_FRONTIER,
    BatchInferenceFrontierAction.KEEP_CURRENT_BATCH_POLICY,
    BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE,
    BatchInferenceFrontierAction.INSUFFICIENT_TELEMETRY,
})

EXECUTION_MODE_SHADOW = "shadow"
EXECUTION_MODE_SIMULATOR = "simulator"
EXECUTION_MODE_REAL_DISABLED = "real_disabled"
EXECUTION_MODE_REAL_ENABLED = "real_enabled"
_BATCH_EXECUTION_MODES = frozenset({
    EXECUTION_MODE_SHADOW, EXECUTION_MODE_SIMULATOR,
    EXECUTION_MODE_REAL_DISABLED, EXECUTION_MODE_REAL_ENABLED,
})

# Allowed trace_source labels. Azure LLM 2024 + BurstGPT are the v1 sources;
# both are *serving* traces being re-purposed as a synthetic batch-flex
# scenario, never as native batch traces (see the synthetic_scenario_label
# requirement in the frontier candidate).
BATCH_TRACE_SOURCES = frozenset({
    "azure_llm_2024", "burstgpt", "synthetic_fixture",
})

_BATCH_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


class BatchInferenceFrontierSchemaError(ValueError):
    """Raised when a batch-inference-frontier model receives a structurally
    invalid value."""


# ---------------------------------------------------------------------------
# Per-workload profile.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchInferenceWorkloadProfile:
    """Per-workload context for the batch-inference-frontier controller.

    Identifies the workload (and the synthetic-deadline scenario backing it)
    and declares its safety budgets. Every numeric SLA is ``Optional`` — a
    missing field means "unknown", never zero.
    """

    workload_id: str
    trace_source: str
    synthetic_scenario_label: str
    # Synthetic-scenario knobs (the source trace has no deadlines; the
    # scenario adds one). Stays ``None`` only when the caller intentionally
    # explores a scenario without a deadline ceiling.
    deadline_slack_seconds_baseline: Optional[float] = None
    deadline_miss_rate_sla_pct: Optional[float] = None
    queue_wait_sla_p99_ms: Optional[float] = None
    interactive_baseline_p99_ms: Optional[float] = None
    interactive_baseline_timeout_pct: Optional[float] = None
    telemetry_confidence: str = "unknown"
    source: str = "unknown"

    def __post_init__(self):
        if self.trace_source not in BATCH_TRACE_SOURCES:
            raise BatchInferenceFrontierSchemaError(
                f"unknown trace_source {self.trace_source!r}; "
                f"expected one of {sorted(BATCH_TRACE_SOURCES)}")
        if not self.synthetic_scenario_label:
            raise BatchInferenceFrontierSchemaError(
                "synthetic_scenario_label is required — the v1 batch "
                "frontier never reads a real deadline from a serving trace")
        if self.telemetry_confidence not in _BATCH_CONF_RANK:
            raise BatchInferenceFrontierSchemaError(
                f"unknown telemetry_confidence {self.telemetry_confidence!r}")
        for f in ("deadline_miss_rate_sla_pct",
                  "interactive_baseline_timeout_pct"):
            v = getattr(self, f)
            if v is not None and not (0.0 <= v <= 100.0 + 1e-9):
                raise BatchInferenceFrontierSchemaError(
                    f"{f} must be in [0,100]; got {v}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate descriptor (multi-axis).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchInferenceFrontierCandidate:
    """Descriptor for one operating point on the batch-inference frontier.

    All four axes are ``Optional`` — the controller may sweep a subset (e.g.
    fix ``batch_window_seconds`` and vary only ``target_rho`` +
    ``deadline_slack_seconds``).
    """

    batch_window_seconds: Optional[float] = None
    batch_concurrency: Optional[int] = None
    target_rho: Optional[float] = None
    deadline_slack_seconds: Optional[float] = None
    # Free-form label naming the underlying policy / measurement source.
    source_policy: Optional[str] = None

    def __post_init__(self):
        if (self.batch_window_seconds is not None
                and self.batch_window_seconds <= 0):
            raise BatchInferenceFrontierSchemaError(
                f"batch_window_seconds must be > 0; got "
                f"{self.batch_window_seconds}")
        if (self.batch_concurrency is not None
                and self.batch_concurrency < 1):
            raise BatchInferenceFrontierSchemaError(
                f"batch_concurrency must be >= 1; got {self.batch_concurrency}")
        if self.target_rho is not None and not (0.0 < self.target_rho <= 1.0):
            raise BatchInferenceFrontierSchemaError(
                f"target_rho must be in (0,1]; got {self.target_rho}")
        if (self.deadline_slack_seconds is not None
                and self.deadline_slack_seconds < 0):
            raise BatchInferenceFrontierSchemaError(
                f"deadline_slack_seconds must be >= 0; got "
                f"{self.deadline_slack_seconds}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frontier point — candidate + measured outcome.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchInferenceFrontierPoint:
    """One point on the batch-inference frontier.

    ``predicted_goodput_per_dollar`` is the canonical KPI from
    ``docs/RESULTS.md`` §1; ``predicted_deadline_miss_rate_pct`` is the new
    batch-class safety gate.

    Honesty: ``predicted_deadline_miss_rate_pct`` is computed against the
    synthetic deadline label carried by the workload profile. The estimator
    NEVER reads a deadline from the source serving trace itself.
    """

    candidate: BatchInferenceFrontierCandidate
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_sla_safe_goodput: Optional[float] = None
    predicted_deadline_miss_rate_pct: Optional[float] = None
    predicted_timeout_rate_pct: Optional[float] = None
    predicted_queue_p95_ms: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    predicted_latency_p95_ms: Optional[float] = None
    predicted_latency_p99_ms: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    predicted_mean_utilization: Optional[float] = None
    predicted_cost_per_sla_compliant_token: Optional[float] = None
    safety_status: str = BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY
    safety_vetoes: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.safety_status not in BATCH_SAFETY_STATUSES:
            raise BatchInferenceFrontierSchemaError(
                f"unknown safety_status {self.safety_status!r}")
        for name in ("predicted_deadline_miss_rate_pct",
                     "predicted_timeout_rate_pct"):
            v = getattr(self, name)
            if v is not None and v < 0.0:
                raise BatchInferenceFrontierSchemaError(
                    f"{name} must be >= 0; got {v}")

    @property
    def is_safe(self) -> bool:
        return self.safety_status == BatchInferenceSafetyStatus.SAFE

    @property
    def is_insufficient_telemetry(self) -> bool:
        return (self.safety_status
                == BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY)

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
class BatchInferenceFrontierDecision:
    """Recommendation-only batch-frontier decision for one workload.

    Recommendation-only at construction (``executable_in_real_cluster
    = False``); real-cluster execution requires the same explicit opt-in
    pattern as the serving + training siblings.
    """

    workload_id: str
    selected_candidate: Optional[BatchInferenceFrontierCandidate]
    current_candidate: Optional[BatchInferenceFrontierCandidate]
    selected_point: Optional[BatchInferenceFrontierPoint]
    frontier_points: tuple
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_deadline_miss_delta_pct: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    confidence: str = "unknown"
    execution_mode: str = EXECUTION_MODE_SHADOW
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False
    safety_vetoes: tuple = ()
    source: str = "batch_inference_frontier_v1"
    notes: tuple = ()

    def __post_init__(self):
        if self.action not in BATCH_FRONTIER_ACTIONS:
            raise BatchInferenceFrontierSchemaError(
                f"unknown batch action {self.action!r}; "
                f"expected one of {sorted(BATCH_FRONTIER_ACTIONS)}")
        if self.execution_mode not in _BATCH_EXECUTION_MODES:
            raise BatchInferenceFrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}")
        if self.executable_in_real_cluster:
            raise BatchInferenceFrontierSchemaError(
                "batch-frontier decisions are recommendation-only at "
                "construction; real execution requires the caller to pass "
                "allow_real_execution=True to "
                "execute_batch_inference_frontier_decision (which itself "
                "ships a no-op stub by default)")
        if self.confidence not in _BATCH_CONF_RANK:
            raise BatchInferenceFrontierSchemaError(
                f"unknown confidence {self.confidence!r}")

    @property
    def is_actionable(self) -> bool:
        return self.action in (
            BatchInferenceFrontierAction.RECOMMEND_BATCH_FRONTIER,
            BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE,
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
            p.to_dict() if isinstance(p, BatchInferenceFrontierPoint) else p
            for p in self.frontier_points]
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d
