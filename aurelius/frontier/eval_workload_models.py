"""Eval Workload Frontier v1 — models.

A **sibling** of the serving Safe Utilization Frontier (rho controller),
the Training Safe Utilization Frontier, and the Batch Inference Frontier.
Eval workloads (LMSYS Chatbot Arena conversations, ShareGPT-style conversation
shapes, model-eval harnesses) are deadline-flexible — typical eval runs
tolerate hours to days of slack — so the frontier candidate space is:

  (eval_batch_window_hours, concurrency, target_rho, deadline_slack_hours,
   dedicated_fleet)

Eval workloads can run on a **dedicated fleet** or share with interactive
serving. The mixed-fleet case requires an explicit interactive-SLA veto:
the eval frontier may NEVER recommend a candidate that degrades the
interactive serving baseline. The v1 enforces this as a hard safety gate.

Hard invariants (asserted by tests):

- Eval Workload Frontier does **NOT** import or mutate the serving rho
  controller (``aurelius/frontier/controller.py``,
  ``aurelius/frontier/dynamic_controller.py``).
- ``EvalWorkloadFrontierDecision.executable_in_real_cluster`` is ``False``
  at construction; real execution is **disabled by default** and the v1
  ships only a stub.
- Deadlines are SYNTHETIC scenario knobs. The eval ingesters
  (``aurelius/traces/sharegpt_aiperf.py``,
  ``aurelius/traces/lmsys_chatbot_arena.py``) emit ``deadline_s = None``
  on every record. The frontier carries a single ``synthetic_scenario_label``
  on the workload profile so reports can audit the deadline source.
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

class EvalWorkloadSafetyStatus:
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


EVAL_SAFETY_STATUSES = frozenset({
    EvalWorkloadSafetyStatus.SAFE,
    EvalWorkloadSafetyStatus.UNSAFE,
    EvalWorkloadSafetyStatus.INSUFFICIENT_TELEMETRY,
})


class EvalWorkloadFrontierAction:
    """Categorical recommendation produced by the eval-frontier controller."""

    RECOMMEND_EVAL_FRONTIER = "RECOMMEND_EVAL_FRONTIER"
    KEEP_CURRENT_EVAL_POLICY = "KEEP_CURRENT_EVAL_POLICY"
    LOWER_EVAL_CONCURRENCY = "LOWER_EVAL_CONCURRENCY"
    ISOLATE_FROM_INTERACTIVE = "ISOLATE_FROM_INTERACTIVE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


EVAL_FRONTIER_ACTIONS = frozenset({
    EvalWorkloadFrontierAction.RECOMMEND_EVAL_FRONTIER,
    EvalWorkloadFrontierAction.KEEP_CURRENT_EVAL_POLICY,
    EvalWorkloadFrontierAction.LOWER_EVAL_CONCURRENCY,
    EvalWorkloadFrontierAction.ISOLATE_FROM_INTERACTIVE,
    EvalWorkloadFrontierAction.INSUFFICIENT_TELEMETRY,
})

EXECUTION_MODE_SHADOW = "shadow"
EXECUTION_MODE_SIMULATOR = "simulator"
EXECUTION_MODE_REAL_DISABLED = "real_disabled"
EXECUTION_MODE_REAL_ENABLED = "real_enabled"
_EVAL_EXECUTION_MODES = frozenset({
    EXECUTION_MODE_SHADOW, EXECUTION_MODE_SIMULATOR,
    EXECUTION_MODE_REAL_DISABLED, EXECUTION_MODE_REAL_ENABLED,
})

# Allowed trace_source labels for the eval frontier. The conversation-shape
# proxy traces (ShareGPT / LMSYS) are the v1 sources; "synthetic_fixture" is
# allowed for tests.
EVAL_TRACE_SOURCES = frozenset({
    "sharegpt_aiperf", "lmsys_chatbot_arena", "synthetic_fixture",
})

_EVAL_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


class EvalWorkloadFrontierSchemaError(ValueError):
    """Raised when an eval-frontier model receives a structurally invalid
    value."""


# ---------------------------------------------------------------------------
# Per-workload profile.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalWorkloadProfile:
    """Per-workload context for the eval-frontier controller.

    The fleet mode (``dedicated_fleet`` True/False) drives the
    mixed-fleet-veto safety gate. When ``dedicated_fleet=False`` the
    interactive baseline thresholds (``interactive_baseline_p99_ms`` /
    ``interactive_baseline_timeout_pct``) are REQUIRED; the safety
    classifier refuses to recommend a candidate without them in mixed mode
    (`safety_status = INSUFFICIENT_TELEMETRY`).
    """

    workload_id: str
    trace_source: str
    synthetic_scenario_label: str
    dedicated_fleet: bool = True
    deadline_slack_hours_baseline: Optional[float] = None
    deadline_miss_rate_sla_pct: Optional[float] = None
    eval_suite_completion_deadline_hours: Optional[float] = None
    interactive_baseline_p99_ms: Optional[float] = None
    interactive_baseline_timeout_pct: Optional[float] = None
    telemetry_confidence: str = "unknown"
    source: str = "unknown"

    def __post_init__(self):
        if self.trace_source not in EVAL_TRACE_SOURCES:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown trace_source {self.trace_source!r}; "
                f"expected one of {sorted(EVAL_TRACE_SOURCES)}")
        if not self.synthetic_scenario_label:
            raise EvalWorkloadFrontierSchemaError(
                "synthetic_scenario_label is required — eval workloads "
                "never carry a real deadline in the source trace")
        if self.telemetry_confidence not in _EVAL_CONF_RANK:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown telemetry_confidence {self.telemetry_confidence!r}")
        for f in ("deadline_miss_rate_sla_pct",
                  "interactive_baseline_timeout_pct"):
            v = getattr(self, f)
            if v is not None and not (0.0 <= v <= 100.0 + 1e-9):
                raise EvalWorkloadFrontierSchemaError(
                    f"{f} must be in [0,100]; got {v}")
        if (self.deadline_slack_hours_baseline is not None
                and self.deadline_slack_hours_baseline < 0):
            raise EvalWorkloadFrontierSchemaError(
                f"deadline_slack_hours_baseline must be >= 0; got "
                f"{self.deadline_slack_hours_baseline}")
        if (self.eval_suite_completion_deadline_hours is not None
                and self.eval_suite_completion_deadline_hours <= 0):
            raise EvalWorkloadFrontierSchemaError(
                f"eval_suite_completion_deadline_hours must be > 0; got "
                f"{self.eval_suite_completion_deadline_hours}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate descriptor (multi-axis).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalWorkloadFrontierCandidate:
    """Descriptor for one operating point on the eval frontier.

    All five axes are ``Optional`` — a sweep may fix a subset.
    """

    eval_batch_window_hours: Optional[float] = None
    concurrency: Optional[int] = None
    target_rho: Optional[float] = None
    deadline_slack_hours: Optional[float] = None
    dedicated_fleet: Optional[bool] = None
    source_policy: Optional[str] = None

    def __post_init__(self):
        if (self.eval_batch_window_hours is not None
                and self.eval_batch_window_hours <= 0):
            raise EvalWorkloadFrontierSchemaError(
                f"eval_batch_window_hours must be > 0; got "
                f"{self.eval_batch_window_hours}")
        if self.concurrency is not None and self.concurrency < 1:
            raise EvalWorkloadFrontierSchemaError(
                f"concurrency must be >= 1; got {self.concurrency}")
        if self.target_rho is not None and not (0.0 < self.target_rho <= 1.0):
            raise EvalWorkloadFrontierSchemaError(
                f"target_rho must be in (0,1]; got {self.target_rho}")
        if (self.deadline_slack_hours is not None
                and self.deadline_slack_hours < 0):
            raise EvalWorkloadFrontierSchemaError(
                f"deadline_slack_hours must be >= 0; got "
                f"{self.deadline_slack_hours}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frontier point — candidate + measured outcome.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalWorkloadFrontierPoint:
    """One point on the eval frontier.

    The canonical KPI is ``predicted_goodput_per_dollar``. The
    deadline-compliance signal is ``predicted_deadline_miss_rate_pct`` and
    must be reported SEPARATELY from the KPI (never folded in —
    ``docs/RESULTS.md`` §1-§2). For mixed-fleet candidates the interactive
    SLA delta is reported as ``predicted_interactive_p99_delta_ms``.
    """

    candidate: EvalWorkloadFrontierCandidate
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_sla_safe_goodput: Optional[float] = None
    predicted_deadline_miss_rate_pct: Optional[float] = None
    predicted_eval_suite_completion_hours: Optional[float] = None
    predicted_interactive_p99_delta_ms: Optional[float] = None
    predicted_interactive_timeout_delta_pct: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    predicted_latency_p99_ms: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    predicted_mean_utilization: Optional[float] = None
    safety_status: str = EvalWorkloadSafetyStatus.INSUFFICIENT_TELEMETRY
    safety_vetoes: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.safety_status not in EVAL_SAFETY_STATUSES:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown safety_status {self.safety_status!r}")
        for name in ("predicted_deadline_miss_rate_pct",):
            v = getattr(self, name)
            if v is not None and v < 0.0:
                raise EvalWorkloadFrontierSchemaError(
                    f"{name} must be >= 0; got {v}")

    @property
    def is_safe(self) -> bool:
        return self.safety_status == EvalWorkloadSafetyStatus.SAFE

    @property
    def is_insufficient_telemetry(self) -> bool:
        return (self.safety_status
                == EvalWorkloadSafetyStatus.INSUFFICIENT_TELEMETRY)

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
class EvalWorkloadFrontierDecision:
    workload_id: str
    selected_candidate: Optional[EvalWorkloadFrontierCandidate]
    current_candidate: Optional[EvalWorkloadFrontierCandidate]
    selected_point: Optional[EvalWorkloadFrontierPoint]
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
    source: str = "eval_workload_frontier_v1"
    notes: tuple = ()

    def __post_init__(self):
        if self.action not in EVAL_FRONTIER_ACTIONS:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown eval action {self.action!r}; "
                f"expected one of {sorted(EVAL_FRONTIER_ACTIONS)}")
        if self.execution_mode not in _EVAL_EXECUTION_MODES:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}")
        if self.executable_in_real_cluster:
            raise EvalWorkloadFrontierSchemaError(
                "eval-frontier decisions are recommendation-only at "
                "construction; real execution requires the caller to pass "
                "allow_real_execution=True to "
                "execute_eval_workload_frontier_decision (which itself "
                "ships a no-op stub by default)")
        if self.confidence not in _EVAL_CONF_RANK:
            raise EvalWorkloadFrontierSchemaError(
                f"unknown confidence {self.confidence!r}")

    @property
    def is_actionable(self) -> bool:
        return self.action in (
            EvalWorkloadFrontierAction.RECOMMEND_EVAL_FRONTIER,
            EvalWorkloadFrontierAction.LOWER_EVAL_CONCURRENCY,
            EvalWorkloadFrontierAction.ISOLATE_FROM_INTERACTIVE,
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
            p.to_dict() if isinstance(p, EvalWorkloadFrontierPoint) else p
            for p in self.frontier_points]
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d
