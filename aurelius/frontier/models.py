"""Data models for the Safe Utilization Frontier Controller.

The frontier is a candidate grid of utilization targets (``rho``) for one
workload. Each grid point carries predicted KPI + safety metrics; the
controller filters to safe points and selects the highest-KPI safe point.

Hard rules (``docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`` §4-§5):

- ``FrontierDecision.executable_in_real_cluster`` is ``False`` by default and
  is NEVER set true by the controller itself. Real execution requires both an
  explicit caller opt-in AND a real executor (see ``execution.py``).
- Safety statuses are categorical (SAFE / UNSAFE / INSUFFICIENT_TELEMETRY) —
  not folded into a score.
- Missing telemetry is preserved as ``None`` (not zero).

Pure / deterministic / stdlib-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


class SafetyStatus:
    """Categorical safety verdict for a single frontier point."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


SAFETY_STATUSES = frozenset({SafetyStatus.SAFE, SafetyStatus.UNSAFE,
                             SafetyStatus.INSUFFICIENT_TELEMETRY})


class FrontierAction:
    """Categorical recommendation produced by the controller."""

    RECOMMEND_RHO = "RECOMMEND_RHO"
    KEEP_RHO = "KEEP_RHO"
    LOWER_RHO = "LOWER_RHO"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


FRONTIER_ACTIONS = frozenset({FrontierAction.RECOMMEND_RHO,
                              FrontierAction.KEEP_RHO,
                              FrontierAction.LOWER_RHO,
                              FrontierAction.INSUFFICIENT_TELEMETRY})

# Execution modes (mirrored in execution.py; defined here for the
# ``FrontierDecision.execution_mode`` validator).
EXECUTION_MODE_SHADOW = "shadow"
EXECUTION_MODE_SIMULATOR = "simulator"
EXECUTION_MODE_REAL_DISABLED = "real_disabled"
EXECUTION_MODE_REAL_ENABLED = "real_enabled"
_EXECUTION_MODES = frozenset({EXECUTION_MODE_SHADOW, EXECUTION_MODE_SIMULATOR,
                              EXECUTION_MODE_REAL_DISABLED,
                              EXECUTION_MODE_REAL_ENABLED})

PRIORITY_CLASSES = frozenset({"critical", "standard", "best_effort", "batch"})


# Default candidate rho grid (mirrors the Azure 2024 frontier audit).
DEFAULT_CANDIDATE_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


class FrontierSchemaError(ValueError):
    """Raised when a frontier model receives a structurally invalid value."""


@dataclass(frozen=True)
class WorkloadFrontierProfile:
    """Per-workload context for the frontier controller.

    Identifies the workload, declares its SLA budget(s), the candidate rho
    grid the controller may consider, and the telemetry-confidence label.
    Everything other than ``workload_id`` and ``workload_type`` is optional —
    a missing field is *unknown*, never zero.

    The (``min_rho``, ``max_rho``) pair clamps the candidate grid; the
    controller MUST refuse to recommend a rho outside this band.
    """

    workload_id: str
    workload_type: str
    model_id: Optional[str] = None
    tenant_id: Optional[str] = None
    region: Optional[str] = None
    latency_sla_ms: Optional[float] = None
    timeout_sla_pct: Optional[float] = None
    queue_p99_sla_ms: Optional[float] = None
    priority_class: str = "standard"
    telemetry_confidence: str = "unknown"
    min_rho: float = 0.30
    max_rho: float = 0.95
    candidate_rhos: tuple = DEFAULT_CANDIDATE_RHOS
    source: str = "unknown"

    def __post_init__(self):
        if not (0.0 < self.min_rho < 1.0):
            raise FrontierSchemaError(
                f"min_rho must be in (0,1); got {self.min_rho}")
        if not (0.0 < self.max_rho <= 1.0):
            raise FrontierSchemaError(
                f"max_rho must be in (0,1]; got {self.max_rho}")
        if self.min_rho > self.max_rho:
            raise FrontierSchemaError(
                f"min_rho {self.min_rho} > max_rho {self.max_rho}")
        if self.priority_class not in PRIORITY_CLASSES:
            raise FrontierSchemaError(
                f"unknown priority_class {self.priority_class!r}; "
                f"expected one of {sorted(PRIORITY_CLASSES)}")
        if not self.candidate_rhos:
            raise FrontierSchemaError("candidate_rhos must be non-empty")

    def clamp_candidates(self) -> tuple:
        """Return the candidate grid clamped to ``[min_rho, max_rho]``."""
        return tuple(r for r in self.candidate_rhos
                     if self.min_rho <= r <= self.max_rho)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate_rhos"] = list(d["candidate_rhos"])
        return d


@dataclass(frozen=True)
class FrontierPoint:
    """One predicted/simulated point on the safe-utilization frontier.

    Predictions are directional — produced by an estimator (e.g. trace
    replay or recent-window simulation). ``safety_status`` is a categorical
    verdict; ``safety_vetoes`` is the set of veto codes that produced an
    UNSAFE / INSUFFICIENT_TELEMETRY verdict (empty for SAFE).
    """

    rho_target: float
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_sla_safe_goodput: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    predicted_timeout_pct: Optional[float] = None
    predicted_queue_p95_ms: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    predicted_latency_p95_ms: Optional[float] = None
    predicted_latency_p99_ms: Optional[float] = None
    predicted_scale_events: Optional[int] = None
    predicted_churn_score: Optional[float] = None
    predicted_mean_utilization: Optional[float] = None
    safety_status: str = SafetyStatus.INSUFFICIENT_TELEMETRY
    safety_vetoes: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.safety_status not in SAFETY_STATUSES:
            raise FrontierSchemaError(
                f"unknown safety_status {self.safety_status!r}; "
                f"expected one of {sorted(SAFETY_STATUSES)}")
        if not (0.0 < self.rho_target <= 1.0):
            raise FrontierSchemaError(
                f"rho_target must be in (0,1]; got {self.rho_target}")

    @property
    def is_safe(self) -> bool:
        return self.safety_status == SafetyStatus.SAFE

    @property
    def is_insufficient_telemetry(self) -> bool:
        return self.safety_status == SafetyStatus.INSUFFICIENT_TELEMETRY

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class FrontierDecision:
    """A recommendation-only frontier decision for one workload.

    Hard invariants (enforced in ``__post_init__``):

    - ``executable_in_real_cluster`` is ``False`` by default and MUST stay
      false at construction (real execution requires explicit caller opt-in
      via ``execute_frontier_decision`` — never via the model itself).
    - ``execution_mode`` is one of ``shadow`` / ``simulator`` / ``real_disabled``
      / ``real_enabled``. The controller emits ``shadow`` by default.
    """

    workload_id: str
    selected_rho: Optional[float]
    selected_point: Optional[FrontierPoint]
    frontier_points: tuple
    action: str
    reason: str
    previous_rho: Optional[float] = None
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    expected_sla_risk_delta: Optional[float] = None
    confidence: str = "unknown"
    execution_mode: str = EXECUTION_MODE_SHADOW
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False
    safety_vetoes: tuple = ()
    source: str = "frontier_controller_v1"

    def __post_init__(self):
        if self.action not in FRONTIER_ACTIONS:
            raise FrontierSchemaError(
                f"unknown frontier action {self.action!r}; "
                f"expected one of {sorted(FRONTIER_ACTIONS)}")
        if self.execution_mode not in _EXECUTION_MODES:
            raise FrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}; "
                f"expected one of {sorted(_EXECUTION_MODES)}")
        if self.executable_in_real_cluster:
            raise FrontierSchemaError(
                "frontier decisions are recommendation-only at construction; "
                "real execution requires the caller to pass "
                "allow_real_execution=True to execute_frontier_decision")

    @property
    def is_actionable(self) -> bool:
        """True when the decision recommends a concrete rho change."""
        return self.action in (FrontierAction.RECOMMEND_RHO,
                               FrontierAction.LOWER_RHO)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frontier_points"] = [p.to_dict() if isinstance(p, FrontierPoint)
                                else p for p in self.frontier_points]
        if d.get("selected_point") is not None and isinstance(self.selected_point,
                                                              FrontierPoint):
            d["selected_point"] = self.selected_point.to_dict()
        d["safety_vetoes"] = list(self.safety_vetoes)
        return d
