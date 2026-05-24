"""SLA ingestion + SLA-aware optimization correction engine for Aurelius.

Public surface:

    from aurelius.sla import (
        SLAPolicy, HardSLA, SoftSLA, PriorityTier, OptimizationAggressiveness,
        SLALoader, SLARegistry, SLAValidationError,
        OptimizationAction, ActionType,
        WorkloadState, RegionContext, HeuristicPredictor, StaticTelemetryProvider,
        evaluate_action_against_sla, SLAEvaluation,
        SLAAwareActionSelector, SLADecision,
        SLAReport,
    )

See docs/SLA.md for the full guide.
"""

from .actions import ActionType, OptimizationAction, keep_current
from .evaluator import RiskBreakdown, SLAEvaluation, evaluate_action_against_sla
from .loader import SLALoader, SLARegistry, policy_from_dict
from .report import SLAReport
from .schema import (
    TIER_DEFAULTS,
    HardSLA,
    OptimizationAggressiveness,
    PriorityTier,
    SLAPolicy,
    SLAValidationError,
    SoftSLA,
    apply_tier_defaults,
)
from .selector import ScoredAction, SLAAwareActionSelector, SLADecision
from .telemetry import (
    HeuristicPredictor,
    RegionContext,
    StaticTelemetryProvider,
    TelemetryProvider,
    WorkloadState,
)

__all__ = [
    # schema
    "HardSLA",
    "SoftSLA",
    "SLAPolicy",
    "PriorityTier",
    "OptimizationAggressiveness",
    "SLAValidationError",
    "TIER_DEFAULTS",
    "apply_tier_defaults",
    # loader
    "SLALoader",
    "SLARegistry",
    "policy_from_dict",
    # actions
    "ActionType",
    "OptimizationAction",
    "keep_current",
    # telemetry
    "WorkloadState",
    "RegionContext",
    "HeuristicPredictor",
    "StaticTelemetryProvider",
    "TelemetryProvider",
    # evaluator
    "evaluate_action_against_sla",
    "SLAEvaluation",
    "RiskBreakdown",
    # selector
    "SLAAwareActionSelector",
    "SLADecision",
    "ScoredAction",
    # report
    "SLAReport",
]
