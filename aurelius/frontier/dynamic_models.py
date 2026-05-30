"""Dynamic Safe Frontier Estimator — telemetry-driven models (v1).

Models that represent **recent observed telemetry** for an LLM-serving
workload and the **estimate / decision** produced by the dynamic
frontier estimator. These are pure stdlib dataclasses — no ML, no
heavyweight dependencies, no future leakage.

Hard invariants (asserted by tests):

- Missing telemetry stays ``None`` — fields are *never* zero-filled.
  See ``docs/PILOT_TELEMETRY_CONTRACT.md`` §1.
- ``DynamicFrontierDecision.executable_in_real_cluster`` is ``False`` by
  default. Real-cluster execution requires the same explicit opt-in as
  the static controller (``aurelius/frontier/execution.py``).
- The dynamic estimator emits ``execution_mode = shadow`` by default;
  the caller decides whether to mutate via the existing
  ``execute_frontier_decision`` shim.
- Risk fields are documented **probability-like scores** in [0, 1],
  produced by deterministic / statistical risk estimation — **NOT
  trained ML** in v1. See ``docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md``.

Directional simulator / shadow-mode evidence only — NOT production
savings (``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

from .models import (
    EXECUTION_MODE_SHADOW,
    FrontierSchemaError,
    SafetyStatus,
    _EXECUTION_MODES,
)

# Categorical action labels for the dynamic controller. The dynamic
# controller distinguishes RAISE / KEEP / LOWER (the static controller
# also exposes RECOMMEND_RHO; a RAISE recommendation is a stricter
# subtype of RECOMMEND_RHO so the static integration shim can map both
# onto a single FrontierDecision).
DYNAMIC_ACTIONS = frozenset({"RAISE_RHO", "KEEP_RHO", "LOWER_RHO",
                              "INSUFFICIENT_TELEMETRY"})

# Telemetry-confidence ordering shared with the static controller.
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class ServingTelemetryTick:
    """One observed tick of LLM-serving telemetry.

    Every numeric field is ``Optional`` so the pilot-telemetry contract
    can declare "unknown" without forcing a zero-fill. The dynamic
    estimator/risk modules MUST treat ``None`` as missing and never
    silently default to 0.0 (asserted by tests).
    """

    timestamp_s: float
    observed_rps: Optional[float] = None
    prompt_tokens_per_s: Optional[float] = None
    output_tokens_per_s: Optional[float] = None
    total_tokens_per_s: Optional[float] = None
    active_replicas: Optional[int] = None
    gpu_hours_delta: Optional[float] = None
    mean_utilization: Optional[float] = None
    queue_p50_ms: Optional[float] = None
    queue_p95_ms: Optional[float] = None
    queue_p99_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    timeout_pct: Optional[float] = None
    sla_violation_pct: Optional[float] = None
    scale_events_delta: Optional[int] = None
    churn_delta: Optional[float] = None
    telemetry_confidence: str = "unknown"
    source: str = "unknown"

    def __post_init__(self):
        if self.telemetry_confidence not in _CONF_RANK:
            raise FrontierSchemaError(
                f"unknown telemetry_confidence "
                f"{self.telemetry_confidence!r}; "
                f"expected one of {sorted(_CONF_RANK)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ServingTelemetryTick":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DynamicFrontierCandidate:
    """Predicted outcome for one candidate rho on the dynamic frontier.

    ``predicted_*`` fields can be ``None`` when the relevant signal is
    missing from telemetry — the estimator never invents data. Risk
    fields are deterministic / statistical probability-like scores in
    [0, 1]; they are *not* trained ML in v1.
    """

    rho_target: float
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    predicted_timeout_pct: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    predicted_latency_p99_ms: Optional[float] = None
    predicted_churn_score: Optional[float] = None
    predicted_sla_risk_probability: Optional[float] = None
    predicted_queue_blowup_probability: Optional[float] = None
    safety_status: str = SafetyStatus.INSUFFICIENT_TELEMETRY
    safety_vetoes: tuple = ()
    confidence: str = "unknown"
    risk_reason_codes: tuple = ()

    def __post_init__(self):
        if not (0.0 < self.rho_target <= 1.0):
            raise FrontierSchemaError(
                f"rho_target must be in (0,1]; got {self.rho_target}")
        if self.safety_status not in (SafetyStatus.SAFE, SafetyStatus.UNSAFE,
                                       SafetyStatus.INSUFFICIENT_TELEMETRY):
            raise FrontierSchemaError(
                f"unknown safety_status {self.safety_status!r}")
        for name in ("predicted_sla_risk_probability",
                     "predicted_queue_blowup_probability"):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 1.0 + 1e-9):
                raise FrontierSchemaError(
                    f"{name} must be in [0,1] (got {v})")
        if self.confidence not in _CONF_RANK:
            raise FrontierSchemaError(
                f"unknown confidence {self.confidence!r}")

    @property
    def is_safe(self) -> bool:
        return self.safety_status == SafetyStatus.SAFE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["risk_reason_codes"] = list(self.risk_reason_codes)
        return d


@dataclass(frozen=True)
class DynamicFrontierEstimate:
    """Telemetry-derived estimate of the safe-utilization frontier.

    Carries the *evidence* — current rho estimate, candidate sweep,
    frontier slope, risk-at-current — separate from the action chosen
    by the controller (`DynamicFrontierDecision`). Round-trippable to
    JSON; immutable; ``None`` for unknown fields.
    """

    workload_id: str
    window_start_s: float
    window_end_s: float
    current_rho_estimate: Optional[float]
    estimated_safe_rho: Optional[float]
    recommended_rho: Optional[float]
    confidence: str
    frontier_slope: Optional[float]
    risk_at_current_rho: Optional[float]
    risk_at_recommended_rho: Optional[float]
    required_headroom: Optional[float]
    candidate_points: tuple
    prediction_method: str
    fallback_reason: Optional[str] = None
    notes: tuple = ()

    def __post_init__(self):
        if self.confidence not in _CONF_RANK:
            raise FrontierSchemaError(
                f"unknown confidence {self.confidence!r}")
        for name in ("risk_at_current_rho", "risk_at_recommended_rho"):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 1.0 + 1e-9):
                raise FrontierSchemaError(
                    f"{name} must be in [0,1] (got {v})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate_points"] = [c.to_dict() if isinstance(c,
                                                            DynamicFrontierCandidate)
                                  else c for c in self.candidate_points]
        d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class DynamicFrontierDecision:
    """The dynamic controller's recommendation for one workload.

    Always recommendation-only at construction
    (``executable_in_real_cluster=False``). Real-cluster execution is
    gated by the existing
    ``aurelius.frontier.execution.execute_frontier_decision`` shim and
    requires an explicit ``allow_real_execution=True`` caller opt-in.
    """

    workload_id: str
    current_rho: Optional[float]
    recommended_rho: Optional[float]
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    expected_sla_risk_delta: Optional[float] = None
    confidence: str = "unknown"
    hysteresis_applied: bool = False
    fallback_reason: Optional[str] = None
    execution_mode: str = EXECUTION_MODE_SHADOW
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False
    source: str = "dynamic_frontier_estimator_v1"
    safety_vetoes: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.action not in DYNAMIC_ACTIONS:
            raise FrontierSchemaError(
                f"unknown dynamic action {self.action!r}; "
                f"expected one of {sorted(DYNAMIC_ACTIONS)}")
        if self.execution_mode not in _EXECUTION_MODES:
            raise FrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}")
        if self.executable_in_real_cluster:
            raise FrontierSchemaError(
                "dynamic decisions are recommendation-only at "
                "construction; real execution requires the caller to "
                "pass allow_real_execution=True to "
                "execute_frontier_decision")
        if self.confidence not in _CONF_RANK:
            raise FrontierSchemaError(
                f"unknown confidence {self.confidence!r}")

    @property
    def is_actionable(self) -> bool:
        return self.action in ("RAISE_RHO", "LOWER_RHO")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d
