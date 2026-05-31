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
from .dynamic_adapter import (
    dynamic_estimate_to_frontier_decision,
)
from .dynamic_calibration import (
    CalibrationPassResult,
    CalibrationReplayConfig,
    CalibrationReplayResult,
    MultiPassCalibrationConfig,
    run_dynamic_frontier_calibration_replay,
    run_multi_pass_calibration,
)
from .dynamic_confidence import (
    ConfidenceUpdateConfig,
    apply_confidence_update,
    update_confidence,
)
from .dynamic_evaluation import (
    DynamicFrontierCalibrationRecord,
    DynamicFrontierObservedOutcome,
    DynamicFrontierPrediction,
    OracleSeriesPoint,
    compute_calibration_record,
    compute_calibration_records,
    compute_frontier_calibration_summary,
    records_from_json,
    records_to_json,
)
from .dynamic_controller import (
    DynamicControllerConfig,
    choose_dynamic_rho,
)
from .dynamic_estimator import (
    DynamicEstimatorConfig,
    estimate_dynamic_frontier,
)
from .dynamic_models import (
    DYNAMIC_ACTIONS,
    DynamicFrontierCandidate,
    DynamicFrontierDecision,
    DynamicFrontierEstimate,
    ServingTelemetryTick,
)
from .dynamic_shadow import (
    DynamicFrontierOutcome,
    DynamicFrontierShadowLog,
    compare_prediction_to_observed,
    read_outcomes as read_dynamic_outcomes,
    read_shadow_log as read_dynamic_shadow_log,
    write_outcome as write_dynamic_outcome,
    write_shadow_log_entry as write_dynamic_shadow_log_entry,
)
from .dynamic_telemetry import (
    DEFAULT_REQUIRED_FIELDS,
    TelemetryWindowValidation,
    build_serving_telemetry_window,
    telemetry_tick_from_arrival_tick,
    telemetry_tick_from_inference_service_state,
    telemetry_tick_with_provenance_from_inference_service_state,
    validate_dynamic_window,
)
from .telemetry_provenance import (
    FieldOrigin,
    FieldProvenance,
    TickProvenance,
    TimeoutFallback,
    TimeoutFallbackResult,
    derive_sla_violation_pct,
    make_derived,
    make_missing,
    make_proxy,
    make_real,
    make_simulated,
    resolve_timeout_pct,
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
from .risk import (
    RiskConfig,
    RiskEstimate,
    estimate_churn_risk,
    estimate_queue_blowup_risk,
    estimate_required_headroom,
    estimate_sla_risk,
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
from .training_alibaba_gpu import (
    ALIBABA_POLICY_CANDIDATES,
    estimate_alibaba_gpu_training_frontier,
    load_alibaba_gpu_summary,
)
from .training_controller import (
    TrainingControllerConfig,
    TrainingRealExecutionDisabledError,
    choose_training_frontier_target,
    execute_training_frontier_decision,
)
from .training_models import (
    TRAINING_FRONTIER_ACTIONS,
    TRAINING_SAFETY_STATUSES,
    TRAINING_TRACE_SOURCES,
    TRAINING_WORKLOAD_TYPES,
    TrainingFrontierAction,
    TrainingFrontierCandidate,
    TrainingFrontierDecision,
    TrainingFrontierPoint,
    TrainingFrontierSchemaError,
    TrainingSafetyStatus,
    TrainingWorkloadProfile,
)
from .training_philly import (
    PHILLY_POLICY_CANDIDATES,
    estimate_philly_training_frontier,
    load_philly_summary,
)
from .training_safety import (
    ALL_TRAINING_VETOES,
    TrainingSafetyConfig,
    classify_training_frontier_point,
    is_training_frontier_point_safe,
)
from .training_shadow import (
    TrainingFrontierShadowLog,
    read_training_shadow_log,
    write_training_shadow_log_entry,
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
    # shadow logging (static)
    "FrontierShadowDecisionLog",
    "FrontierShadowLog",
    "write_shadow_log_entry",
    "read_shadow_log",
    # dynamic v1 — telemetry-driven estimator
    "ServingTelemetryTick",
    "DynamicFrontierCandidate",
    "DynamicFrontierEstimate",
    "DynamicFrontierDecision",
    "DYNAMIC_ACTIONS",
    "DEFAULT_REQUIRED_FIELDS",
    "TelemetryWindowValidation",
    "build_serving_telemetry_window",
    "telemetry_tick_from_arrival_tick",
    "telemetry_tick_from_inference_service_state",
    "telemetry_tick_with_provenance_from_inference_service_state",
    "validate_dynamic_window",
    # telemetry provenance v1
    "FieldOrigin",
    "FieldProvenance",
    "TickProvenance",
    "TimeoutFallback",
    "TimeoutFallbackResult",
    "derive_sla_violation_pct",
    "make_derived",
    "make_missing",
    "make_proxy",
    "make_real",
    "make_simulated",
    "resolve_timeout_pct",
    "DynamicEstimatorConfig",
    "estimate_dynamic_frontier",
    "DynamicControllerConfig",
    "choose_dynamic_rho",
    "RiskConfig",
    "RiskEstimate",
    "estimate_sla_risk",
    "estimate_queue_blowup_risk",
    "estimate_required_headroom",
    "estimate_churn_risk",
    "DynamicFrontierShadowLog",
    "DynamicFrontierOutcome",
    "compare_prediction_to_observed",
    "write_dynamic_shadow_log_entry",
    "read_dynamic_shadow_log",
    "write_dynamic_outcome",
    "read_dynamic_outcomes",
    "dynamic_estimate_to_frontier_decision",
    # dynamic frontier calibration + shadow evaluation v1
    "DynamicFrontierPrediction",
    "DynamicFrontierObservedOutcome",
    "DynamicFrontierCalibrationRecord",
    "OracleSeriesPoint",
    "compute_calibration_record",
    "compute_calibration_records",
    "compute_frontier_calibration_summary",
    "records_to_json",
    "records_from_json",
    "ConfidenceUpdateConfig",
    "update_confidence",
    "apply_confidence_update",
    "CalibrationReplayConfig",
    "CalibrationPassResult",
    "CalibrationReplayResult",
    "MultiPassCalibrationConfig",
    "run_dynamic_frontier_calibration_replay",
    "run_multi_pass_calibration",
    # training frontier v1 — sibling of serving frontier
    "TrainingWorkloadProfile",
    "TrainingFrontierCandidate",
    "TrainingFrontierPoint",
    "TrainingFrontierDecision",
    "TrainingFrontierAction",
    "TRAINING_FRONTIER_ACTIONS",
    "TrainingSafetyStatus",
    "TRAINING_SAFETY_STATUSES",
    "TRAINING_TRACE_SOURCES",
    "TRAINING_WORKLOAD_TYPES",
    "TrainingFrontierSchemaError",
    "TrainingSafetyConfig",
    "ALL_TRAINING_VETOES",
    "classify_training_frontier_point",
    "is_training_frontier_point_safe",
    "TrainingControllerConfig",
    "choose_training_frontier_target",
    "execute_training_frontier_decision",
    "TrainingRealExecutionDisabledError",
    "PHILLY_POLICY_CANDIDATES",
    "estimate_philly_training_frontier",
    "load_philly_summary",
    "ALIBABA_POLICY_CANDIDATES",
    "estimate_alibaba_gpu_training_frontier",
    "load_alibaba_gpu_summary",
    "TrainingFrontierShadowLog",
    "write_training_shadow_log_entry",
    "read_training_shadow_log",
]
