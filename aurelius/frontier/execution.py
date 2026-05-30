"""Execution interface for frontier decisions.

Four execution modes — only ``simulator`` mutates state; ``shadow`` /
``real_disabled`` mutate nothing; ``real_enabled`` mutates only when
*every* one of these is true:

- the caller passed ``allow_real_execution=True`` (explicit opt-in),
- the caller passed a concrete ``executor`` (no real-cluster executor
  ships in this package),
- the decision's safety gates pass (no ``safety_vetoes``).

If real execution is requested without a real executor, the call returns
``ExecutionEffect(mutated=False, notes=["not_implemented_real_executor"])``.
There is no Kubernetes / router / serving-engine write path in this module.

Binding invariant (asserted by tests): ``execute_frontier_decision(..., mode=
"real_enabled")`` with the default (``executor=None``) MUST return
``mutated=False`` even when ``allow_real_execution=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import (
    EXECUTION_MODE_REAL_DISABLED,
    EXECUTION_MODE_REAL_ENABLED,
    EXECUTION_MODE_SHADOW,
    EXECUTION_MODE_SIMULATOR,
    FrontierAction,
    FrontierDecision,
)

# Re-export the mode constants under shorter aliases.
SHADOW_MODE = EXECUTION_MODE_SHADOW
SIMULATOR_MODE = EXECUTION_MODE_SIMULATOR
REAL_DISABLED = EXECUTION_MODE_REAL_DISABLED
REAL_ENABLED = EXECUTION_MODE_REAL_ENABLED

EXECUTION_MODES = frozenset({SHADOW_MODE, SIMULATOR_MODE, REAL_DISABLED, REAL_ENABLED})


class RealExecutionDisabledError(RuntimeError):
    """Raised if real-cluster execution is invoked without the explicit
    opt-in flag. Production mutation is disabled by default."""


@dataclass
class ExecutionEffect:
    """What executing the decision did (or, in non-mutating modes, would
    have recommended)."""

    workload_id: str
    action: str
    mode: str
    selected_rho: Optional[float] = None
    previous_rho: Optional[float] = None
    mutated: bool = False
    notes: list = field(default_factory=list)
    simulated_state_after: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id, "action": self.action,
            "mode": self.mode, "selected_rho": self.selected_rho,
            "previous_rho": self.previous_rho, "mutated": self.mutated,
            "notes": list(self.notes),
            "simulated_state_after": self.simulated_state_after,
        }


# A "simulated state" is just a dict the caller owns (e.g. workload_id ->
# target_rho). We update it in place when ``simulator`` mode is selected.
SimulatedState = dict


def execute_frontier_decision(decision: FrontierDecision,
                              mode: str,
                              *,
                              executor: Optional[Callable[..., Any]] = None,
                              simulated_state: Optional[SimulatedState] = None,
                              allow_real_execution: bool = False) -> ExecutionEffect:
    """Execute (or shadow-log) a frontier decision.

    - :data:`SHADOW_MODE`: log only; ``mutated=False`` always.
    - :data:`SIMULATOR_MODE`: update ``simulated_state[decision.workload_id]``
      to ``decision.selected_rho`` when the action is RECOMMEND_RHO /
      LOWER_RHO and the decision is ``executable_in_simulator``.
    - :data:`REAL_DISABLED`: log only; mutates nothing (the real write path
      *exists in the type signature* but is disabled).
    - :data:`REAL_ENABLED`: only allowed when ``allow_real_execution=True``;
      delegates to ``executor`` if provided, else returns
      ``not_implemented_real_executor`` with ``mutated=False``.
    """
    if mode not in EXECUTION_MODES:
        raise ValueError(
            f"unknown execution mode {mode!r}; expected one of {sorted(EXECUTION_MODES)}")

    effect = ExecutionEffect(
        workload_id=decision.workload_id, action=decision.action, mode=mode,
        selected_rho=decision.selected_rho, previous_rho=decision.previous_rho)

    # ---- SHADOW ----
    if mode == SHADOW_MODE:
        effect.notes.append("shadow mode: recommendation logged, no mutation")
        return effect

    # ---- SIMULATOR ----
    if mode == SIMULATOR_MODE:
        if not decision.executable_in_simulator:
            effect.notes.append(
                "decision not executable in simulator (likely insufficient telemetry)")
            return effect
        if decision.action not in (FrontierAction.RECOMMEND_RHO,
                                   FrontierAction.LOWER_RHO):
            effect.notes.append(f"action {decision.action} requires no mutation")
            return effect
        if simulated_state is None:
            effect.notes.append(
                "simulator mode requested but no simulated_state dict provided")
            return effect
        if decision.selected_rho is None:
            effect.notes.append("decision has no selected_rho; nothing to apply")
            return effect
        simulated_state[decision.workload_id] = float(decision.selected_rho)
        effect.mutated = True
        effect.simulated_state_after = dict(simulated_state)
        effect.notes.append(
            f"simulator mode: set workload {decision.workload_id} rho_target "
            f"to {decision.selected_rho}")
        return effect

    # ---- REAL_DISABLED ----
    if mode == REAL_DISABLED:
        effect.notes.append(
            "real_disabled mode: production write path exists but is disabled; "
            "no mutation")
        return effect

    # ---- REAL_ENABLED ----
    # Even with the mode set, mutation requires explicit caller opt-in AND a
    # real executor. The package ships none; this is a stub interface.
    if not allow_real_execution:
        raise RealExecutionDisabledError(
            "real-cluster execution requires allow_real_execution=True "
            "(it is disabled by default; see "
            "docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md)")
    if decision.safety_vetoes:
        effect.notes.append(
            f"real execution blocked: safety vetoes present "
            f"({', '.join(decision.safety_vetoes)})")
        return effect
    if executor is None:
        effect.notes.append("not_implemented_real_executor")
        return effect
    if decision.action not in (FrontierAction.RECOMMEND_RHO,
                               FrontierAction.LOWER_RHO):
        effect.notes.append(
            f"real_enabled: action {decision.action} requires no mutation")
        return effect
    # Real executor must be passed in by the caller (tests inject a mock).
    # The contract: executor(decision) -> Any; we mark mutated=True.
    try:
        executor_result = executor(decision)
    except Exception as exc:  # pragma: no cover - error path
        effect.notes.append(f"real_executor_error: {exc!r}")
        return effect
    effect.mutated = True
    effect.notes.append(f"real_enabled: executor returned {executor_result!r}")
    return effect
