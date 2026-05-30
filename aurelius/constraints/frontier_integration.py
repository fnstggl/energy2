"""Frontier-controller integration shim for the ``constraint_aware`` optimizer.

Wires :mod:`aurelius.frontier` into the unchanged ``constraint_aware`` policy
*only* when the caller explicitly opts in. By default — and in every existing
benchmark, test, and production code path — this module is a no-op: the
``constraint_aware`` policy keeps its hard-coded rho 0.65 default and
byte-for-byte identical output (asserted by
``tests/test_constraint_aware_frontier_integration.py``).

When the caller passes an enabled :class:`FrontierIntegrationConfig`:

1. :func:`is_frontier_eligible` decides whether the workload is *eligible*
   for the frontier controller (LLM serving only; telemetry trust gate;
   telemetry-window minimum; explicit allow-list of workload types).
2. If eligible, :func:`select_constraint_aware_rho` calls
   ``estimate_frontier`` + ``choose_safe_utilization_target`` to pick a safe
   rho, and returns ``(selected_rho, telemetry_dict)``.
3. If ineligible OR the controller returns INSUFFICIENT_TELEMETRY / unsafe
   recommendation / errors, the function **falls back to the
   constraint_aware default rho** and records the fallback reason in
   ``telemetry_dict``.

Hard invariants (enforced by tests):

- Default ``FrontierIntegrationConfig`` is ``enabled=False`` —
  ``constraint_aware`` behaviour is unchanged.
- The adapter only supplies a rho target. It never mutates the scheduler,
  router, energy adapter, residency engine, or any production state.
- ``allow_real_execution=False`` by default. Frontier decisions remain
  recommendation-only; the existing constraint-aware SLA / queue / latency /
  cache / energy / telemetry gates downstream of the rho selection still
  run and may still reject any resulting action.
- Robust energy engine files are not touched.
- No ML model training, no new datasets, no oracle baseline.

Directional simulator/backtest evidence only — NOT production savings
(``docs/RESULTS.md`` §8). Pilot telemetry is required to calibrate the safe
rho per workload before any production claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from aurelius.frontier import (
    ANTICIPATORY,
    SHADOW_MODE,
    SIMULATOR_MODE,
    FrontierAction,
    FrontierControllerConfig,
    FrontierEstimatorConfig,
    SafetyConfig,
    SafetyStatus,
    WorkloadFrontierProfile,
    choose_safe_utilization_target,
    estimate_frontier,
)

# constraint_aware's engine-level rho default — duplicated here as a
# **read-only fallback** so the adapter can return it explicitly when the
# frontier is ineligible / unsafe / failed. The engine constant itself
# (``aurelius/traces/backtest.py:_run_policy``) is unchanged.
CONSTRAINT_AWARE_DEFAULT_RHO = 0.65

# Default LLM-serving workload type allow-list. Everything outside this set
# is ineligible by design — packing / training / batch GPU jobs do not have
# a continuous serving rho target.
DEFAULT_LLM_SERVING_WORKLOAD_TYPES = frozenset({
    "inference_standard",
    "interactive_inference",
    "llm_serving",
    "standard_interactive_inference",
    "critical_interactive_inference",
})

# Telemetry-confidence rank used by the eligibility gate; mirrors the
# controller's internal ranking.
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class FrontierIntegrationConfig:
    """Opt-in, conservative-by-default integration switch.

    Defaults are chosen so that **enabling the config without overriding any
    other field still preserves safety**: shadow-only, no real execution,
    fallback-to-default on every error.
    """

    enabled: bool = False
    # Allow-list of workload types the controller may be invoked for.
    allowed_workload_types: frozenset = field(
        default_factory=lambda: DEFAULT_LLM_SERVING_WORKLOAD_TYPES)
    # Minimum telemetry-confidence required to even call the controller.
    min_telemetry_confidence: str = "medium"
    # Candidate rho grid the controller may consider.
    candidate_rhos: tuple = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    # Pre-registered safety thresholds; failure to meet these triggers the
    # controller's safety veto (not folded into a score).
    max_timeout_pct: float = 10.0
    max_queue_p99_ms: float = 2000.0
    max_latency_p99_ms: Optional[float] = None
    # When the best safe point sits adjacent to an UNSAFE point, the
    # controller can step back to the next-lower safe rho.
    conservative_margin_enabled: bool = False
    # Minimum number of telemetry ticks required to estimate a frontier; any
    # fewer triggers fallback.
    min_telemetry_window_ticks: int = 8
    # Default: fall back to the existing constraint_aware behaviour on ANY
    # uncertainty or estimator error.
    fallback_to_existing_on_error: bool = True
    # Shadow vs simulator vs real execution mode (defaults preserve
    # recommendation-only behaviour).
    shadow_only: bool = True
    allow_simulator_execution: bool = False
    allow_real_execution: bool = False

    def __post_init__(self):
        if self.min_telemetry_confidence not in _CONF_RANK:
            raise ValueError(
                f"unknown min_telemetry_confidence "
                f"{self.min_telemetry_confidence!r}; "
                f"expected one of {sorted(_CONF_RANK)}")
        if not (0.0 < self.max_timeout_pct <= 100.0):
            raise ValueError(
                f"max_timeout_pct must be in (0,100]; got {self.max_timeout_pct}")
        if not self.candidate_rhos:
            raise ValueError("candidate_rhos must be non-empty")
        # Real-execution invariant: real execution is allowed ONLY when the
        # caller explicitly turns off shadow_only AND sets allow_real_execution.
        if self.allow_real_execution and self.shadow_only:
            raise ValueError(
                "allow_real_execution=True requires shadow_only=False "
                "(real execution is disabled by default)")


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of the workload-eligibility check.

    ``fallback_policy`` names the policy/behaviour the caller should use
    when ``eligible`` is False (always ``constraint_aware_default`` for v1).
    """

    eligible: bool
    reason: str
    missing_fields: tuple = ()
    fallback_policy: str = "constraint_aware_default"

    def to_dict(self) -> dict:
        return {"eligible": self.eligible, "reason": self.reason,
                "missing_fields": list(self.missing_fields),
                "fallback_policy": self.fallback_policy}


def _telemetry_confidence_sufficient(have: str, need: str) -> bool:
    return _CONF_RANK.get(have or "unknown", 0) >= _CONF_RANK.get(need, 0)


def is_frontier_eligible(
    service_state: dict,
    workload_metadata: dict,
    config: FrontierIntegrationConfig,
) -> EligibilityResult:
    """Decide whether the frontier controller may act on this workload.

    ``service_state`` is a plain dict of recent telemetry (the adapter does
    not depend on any production-only object), e.g.::

        {"telemetry_ticks": [...], "queue_metrics_present": True,
         "request_metrics_present": True, "telemetry_window_ticks": 12 }

    ``workload_metadata`` declares the workload's identity / SLA / type,
    e.g.::

        {"workload_id": "tenantA-vllm", "workload_type": "inference_standard",
         "telemetry_confidence": "medium", "latency_sla_ms": 1500.0,
         "priority_class": "standard"}

    Failing eligibility is **never silent**: the returned :class:`EligibilityResult`
    carries an explicit ``reason`` and the missing fields.
    """
    if not config.enabled:
        return EligibilityResult(False, "frontier_integration_disabled")

    missing: list[str] = []
    for f in ("workload_id", "workload_type"):
        if not workload_metadata.get(f):
            missing.append(f)
    if missing:
        return EligibilityResult(
            False, f"workload_metadata missing required fields {missing}",
            tuple(missing))

    wl_type = workload_metadata.get("workload_type")
    if wl_type not in config.allowed_workload_types:
        return EligibilityResult(
            False,
            f"workload_type {wl_type!r} not in allowed_workload_types "
            f"{sorted(config.allowed_workload_types)}")

    if workload_metadata.get("is_training") or wl_type in (
            "training", "fine_tuning", "offline_batch",
            "philly_training_job", "alibaba_gpu_packing_job",
            "batch_inference"):
        return EligibilityResult(
            False, f"workload_type {wl_type!r} is training/batch/packing; "
                   "frontier controller acts on serving rho only")

    have_conf = workload_metadata.get("telemetry_confidence") or "unknown"
    if not _telemetry_confidence_sufficient(
            have_conf, config.min_telemetry_confidence):
        return EligibilityResult(
            False,
            f"telemetry_confidence {have_conf!r} below required "
            f"{config.min_telemetry_confidence!r}")

    # Telemetry-window minimum: without enough ticks the estimator can only
    # produce noise; fall back to default rho.
    window_ticks = service_state.get(
        "telemetry_window_ticks",
        len(service_state.get("telemetry_ticks", []) or []))
    if window_ticks < config.min_telemetry_window_ticks:
        return EligibilityResult(
            False,
            f"telemetry_window_ticks={window_ticks} below required "
            f"{config.min_telemetry_window_ticks}")

    # Either request OR queue telemetry must exist (the controller needs
    # something to size against).
    if (not service_state.get("request_metrics_present")
            and not service_state.get("queue_metrics_present")):
        return EligibilityResult(
            False,
            "neither request nor queue telemetry is present in service_state",
            ("request_metrics_present", "queue_metrics_present"))

    # SLA / timeout budget must exist.
    if (workload_metadata.get("latency_sla_ms") is None
            and workload_metadata.get("timeout_sla_pct") is None
            and workload_metadata.get("queue_p99_sla_ms") is None):
        return EligibilityResult(
            False,
            "no SLA / timeout / queue_p99 budget declared in workload_metadata",
            ("latency_sla_ms", "timeout_sla_pct", "queue_p99_sla_ms"))

    # Degraded telemetry / failsafe: caller may explicitly mark this.
    if service_state.get("degraded_telemetry"):
        return EligibilityResult(
            False, "service_state.degraded_telemetry is set")
    if service_state.get("failsafe_active"):
        return EligibilityResult(False, "service_state.failsafe_active is set")

    return EligibilityResult(True, "ok")


@dataclass(frozen=True)
class FrontierAdapterResult:
    """Carrier for the adapter's verdict.

    ``selected_rho`` is what the constraint_aware sizer should use.
    ``used_frontier`` is True only when the controller produced an actionable
    recommendation; False (with ``fallback_reason``) in every other case.
    """

    selected_rho: float
    used_frontier: bool
    eligibility: EligibilityResult
    decision: Optional[object]
    fallback_reason: Optional[str]
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    expected_sla_risk_delta: Optional[float] = None
    confidence: str = "unknown"
    safety_vetoes: tuple = ()
    execution_mode: str = SHADOW_MODE

    def to_dict(self) -> dict:
        return {
            "selected_rho": self.selected_rho,
            "used_frontier": self.used_frontier,
            "eligibility": self.eligibility.to_dict(),
            "decision": (self.decision.to_dict()
                         if (self.decision is not None
                             and hasattr(self.decision, "to_dict"))
                         else None),
            "fallback_reason": self.fallback_reason,
            "expected_goodput_per_dollar_delta":
                self.expected_goodput_per_dollar_delta,
            "expected_gpu_hour_delta": self.expected_gpu_hour_delta,
            "expected_sla_risk_delta": self.expected_sla_risk_delta,
            "confidence": self.confidence,
            "safety_vetoes": list(self.safety_vetoes),
            "execution_mode": self.execution_mode,
        }


def _default_rho_result(eligibility: EligibilityResult, reason: str,
                        confidence: str = "unknown") -> FrontierAdapterResult:
    return FrontierAdapterResult(
        selected_rho=CONSTRAINT_AWARE_DEFAULT_RHO, used_frontier=False,
        eligibility=eligibility, decision=None, fallback_reason=reason,
        confidence=confidence, execution_mode=SHADOW_MODE)


def select_constraint_aware_rho(
    service_state: dict,
    workload_metadata: dict,
    config: FrontierIntegrationConfig,
    *,
    current_rho: float = CONSTRAINT_AWARE_DEFAULT_RHO,
    telemetry_window: Optional[Sequence] = None,
    tick_seconds: float = 60.0,
) -> FrontierAdapterResult:
    """Pick the constraint_aware target rho for this workload.

    Returns a :class:`FrontierAdapterResult` carrying either the
    frontier-selected rho (when eligible + safe) or the
    ``constraint_aware`` default rho (with an explicit fallback reason).

    The adapter never mutates state. The caller continues to run all
    existing ``constraint_aware`` SLA / queue / latency / cache / energy /
    telemetry gates downstream of the rho selection.
    """
    eligibility = is_frontier_eligible(service_state, workload_metadata, config)
    if not eligibility.eligible:
        return _default_rho_result(eligibility,
                                   f"ineligible: {eligibility.reason}")

    window = (telemetry_window
              if telemetry_window is not None
              else service_state.get("telemetry_ticks"))
    if not window:
        return _default_rho_result(
            eligibility, "telemetry_window empty",
            confidence=workload_metadata.get("telemetry_confidence", "unknown"))

    profile = WorkloadFrontierProfile(
        workload_id=workload_metadata["workload_id"],
        workload_type=workload_metadata["workload_type"],
        model_id=workload_metadata.get("model_id"),
        tenant_id=workload_metadata.get("tenant_id"),
        region=workload_metadata.get("region"),
        latency_sla_ms=workload_metadata.get("latency_sla_ms"),
        timeout_sla_pct=workload_metadata.get("timeout_sla_pct"),
        queue_p99_sla_ms=workload_metadata.get("queue_p99_sla_ms"),
        priority_class=workload_metadata.get("priority_class", "standard"),
        telemetry_confidence=workload_metadata.get("telemetry_confidence",
                                                   "unknown"),
        candidate_rhos=tuple(config.candidate_rhos),
        source="constraint_aware_frontier_integration",
    )

    safety = SafetyConfig(
        max_timeout_pct=config.max_timeout_pct,
        max_queue_p99_ms=config.max_queue_p99_ms,
        max_latency_p99_ms=config.max_latency_p99_ms,
        min_telemetry_confidence=config.min_telemetry_confidence,
    )
    est_cfg = FrontierEstimatorConfig(mode=ANTICIPATORY,
                                      tick_seconds=tick_seconds)

    try:
        points = estimate_frontier(profile, window,
                                   candidate_rhos=config.candidate_rhos,
                                   predictor_config=est_cfg,
                                   safety_config=safety)
    except Exception as exc:  # noqa: BLE001
        if not config.fallback_to_existing_on_error:
            raise
        return _default_rho_result(
            eligibility,
            f"estimator_error:{type(exc).__name__}:{exc}",
            confidence=workload_metadata.get("telemetry_confidence", "unknown"))

    ctrl_cfg = FrontierControllerConfig(
        conservative_margin=config.conservative_margin_enabled,
        min_telemetry_confidence=config.min_telemetry_confidence,
        default_execution_mode=SHADOW_MODE,
    )

    try:
        decision = choose_safe_utilization_target(
            profile, points, current_rho=current_rho,
            controller_config=ctrl_cfg)
    except Exception as exc:  # noqa: BLE001
        if not config.fallback_to_existing_on_error:
            raise
        return _default_rho_result(
            eligibility,
            f"controller_error:{type(exc).__name__}:{exc}",
            confidence=workload_metadata.get("telemetry_confidence", "unknown"))

    # Branch on the controller's action.
    if decision.action == FrontierAction.INSUFFICIENT_TELEMETRY:
        return _default_rho_result(
            eligibility,
            f"controller_insufficient_telemetry: {decision.reason}",
            confidence=decision.confidence)

    if decision.action == FrontierAction.LOWER_RHO:
        # LOWER_RHO is a safety signal: the current rho is unsafe. Fall back
        # to the constraint_aware default (the engine's hysteresis / SLA
        # trim will still cap the chosen replicas) — we do NOT promote the
        # frontier's lower recommendation past the safety boundary because
        # the integration is opt-in and conservative-by-default.
        return FrontierAdapterResult(
            selected_rho=CONSTRAINT_AWARE_DEFAULT_RHO, used_frontier=False,
            eligibility=eligibility, decision=decision,
            fallback_reason=f"controller_lower_rho: {decision.reason}",
            expected_goodput_per_dollar_delta=decision
                .expected_goodput_per_dollar_delta,
            expected_gpu_hour_delta=decision.expected_gpu_hour_delta,
            expected_sla_risk_delta=decision.expected_sla_risk_delta,
            confidence=decision.confidence,
            safety_vetoes=tuple(decision.safety_vetoes),
            execution_mode=SHADOW_MODE)

    # Final safety check: the selected point itself must be SAFE — never
    # promote an UNSAFE recommendation even if the action says RECOMMEND_RHO.
    sp = decision.selected_point
    if sp is None or sp.safety_status != SafetyStatus.SAFE:
        return _default_rho_result(
            eligibility,
            "controller_selected_unsafe_point",
            confidence=decision.confidence)

    rho = decision.selected_rho or CONSTRAINT_AWARE_DEFAULT_RHO
    if not (0.0 < rho <= 1.0):
        return _default_rho_result(
            eligibility,
            f"controller_returned_invalid_rho: {rho}",
            confidence=decision.confidence)

    return FrontierAdapterResult(
        selected_rho=float(rho), used_frontier=True,
        eligibility=eligibility, decision=decision,
        fallback_reason=None,
        expected_goodput_per_dollar_delta=decision
            .expected_goodput_per_dollar_delta,
        expected_gpu_hour_delta=decision.expected_gpu_hour_delta,
        expected_sla_risk_delta=decision.expected_sla_risk_delta,
        confidence=decision.confidence,
        safety_vetoes=tuple(decision.safety_vetoes),
        execution_mode=(SIMULATOR_MODE if config.allow_simulator_execution
                        else SHADOW_MODE),
    )


@dataclass
class FrontierIntegrationCounters:
    """Per-workload observability counters for the integration."""

    frontier_used_count: int = 0
    frontier_fallback_count: int = 0
    frontier_ineligible_count: int = 0
    frontier_low_confidence_count: int = 0
    frontier_unsafe_recommendation_count: int = 0
    frontier_lower_rho_count: int = 0
    frontier_error_count: int = 0

    def record(self, result: FrontierAdapterResult) -> None:
        if result.used_frontier:
            self.frontier_used_count += 1
            return
        self.frontier_fallback_count += 1
        reason = result.fallback_reason or ""
        if not result.eligibility.eligible:
            self.frontier_ineligible_count += 1
            return
        if "insufficient_telemetry" in reason:
            self.frontier_low_confidence_count += 1
        if "lower_rho" in reason:
            self.frontier_lower_rho_count += 1
        if "unsafe" in reason:
            self.frontier_unsafe_recommendation_count += 1
        if "error" in reason:
            self.frontier_error_count += 1

    def to_dict(self) -> dict:
        return dict(self.__dict__)
