"""Safe Utilization Frontier Controller — v1 (simulator/shadow mode).

This package implements the *Safe Utilization Frontier Controller* described in
``docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md``: it estimates a per-workload
safe-utilization frontier across a candidate rho grid, vetoes points that
breach safety gates (timeout, queue p99, latency p99, telemetry confidence,
thermal/topology/memory risk, churn), and recommends the *highest SLA-safe
goodput/$* point — not the highest utilization.

Binding boundaries (asserted by tests):

- **Recommendation-only in real/customer mode.** A
  :class:`FrontierDecision` carries ``executable_in_real_cluster=False`` by
  default. Real-cluster execution requires *both* an explicit
  ``allow_real_execution=True`` opt-in and a real executor passed in by the
  caller — the controller itself ships only a no-op stub.
- **No production mutation by default.** ``execute_frontier_decision`` in
  ``shadow`` / ``real_disabled`` modes mutates nothing. Only
  :data:`SIMULATOR_MODE` may mutate (simulated) state.
- **Safety is a veto, not a weight.** SLA / queue / latency / thermal /
  topology / memory / telemetry-confidence gates filter candidates; they are
  never folded into the KPI (``docs/RESULTS.md`` §1-§2).
- **Missing telemetry → INSUFFICIENT_TELEMETRY.** Unknown signals never get
  silently zero-filled (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1).
- **No ML training, no new datasets, no robust energy engine change.**

Directional simulator/backtest evidence only — not production savings
(``docs/RESULTS.md`` §8). Real pilot telemetry is required to calibrate the
safe rho per workload.
"""

from .controller import (
    FrontierControllerConfig,
    choose_safe_utilization_target,
)
from .estimator import (
    ANTICIPATORY,
    REACTIVE,
    FrontierEstimatorConfig,
    estimate_frontier,
    estimate_frontier_from_points,
)
from .execution import (
    EXECUTION_MODES,
    REAL_DISABLED,
    REAL_ENABLED,
    SHADOW_MODE,
    SIMULATOR_MODE,
    ExecutionEffect,
    RealExecutionDisabledError,
    execute_frontier_decision,
)
from .models import (
    FRONTIER_ACTIONS,
    SAFETY_STATUSES,
    FrontierAction,
    FrontierDecision,
    FrontierPoint,
    SafetyStatus,
    WorkloadFrontierProfile,
)
from .safety import (
    SafetyConfig,
    is_frontier_point_safe,
)
from .shadow import (
    FrontierShadowDecisionLog,
    FrontierShadowLog,
    read_shadow_log,
    write_shadow_log_entry,
)

__all__ = [
    # models
    "WorkloadFrontierProfile",
    "FrontierPoint",
    "FrontierDecision",
    "FrontierAction",
    "FRONTIER_ACTIONS",
    "SafetyStatus",
    "SAFETY_STATUSES",
    # estimator
    "FrontierEstimatorConfig",
    "estimate_frontier",
    "estimate_frontier_from_points",
    "ANTICIPATORY",
    "REACTIVE",
    # safety
    "SafetyConfig",
    "is_frontier_point_safe",
    # controller
    "FrontierControllerConfig",
    "choose_safe_utilization_target",
    # execution
    "execute_frontier_decision",
    "ExecutionEffect",
    "RealExecutionDisabledError",
    "EXECUTION_MODES",
    "SHADOW_MODE",
    "SIMULATOR_MODE",
    "REAL_DISABLED",
    "REAL_ENABLED",
    # shadow logging
    "FrontierShadowDecisionLog",
    "FrontierShadowLog",
    "write_shadow_log_entry",
    "read_shadow_log",
]
