"""Constraint classifier package for Aurelius constraint-aware orchestration.

This package is read-only over ClusterState. It does NOT touch the optimizer,
execution adapters, or any inference/runtime internals.

Public interface:
    ConstraintClassifier      — scores all 8 constraint families, emits ConstraintAssessment
    ConstraintConfig          — configurable thresholds (all marked # HEURISTIC)
    MigrationCostModel        — state-conditioned cost/risk estimator for candidate actions
    MigrationCostEstimate     — per-candidate cost/risk breakdown (with risk-factor explanation)
    RiskInputs                — state-conditioned inputs (SLA policy, workload/dest state, telemetry)
    CostModelConfig           — configurable risk weights/thresholds (all marked # HEURISTIC)
    ConstraintAwareEngine     — Phase 9: full recommendation pipeline
    EngineResult              — output of ConstraintAwareEngine.run()
    WorkloadDescriptor        — lightweight workload adapter for the engine
"""

from .classifier import ConstraintClassifier, ConstraintConfig
from .cost_model import (
    CostModelConfig,
    MigrationCostEstimate,
    MigrationCostModel,
    MigrationGovernor,
    RiskInputs,
)
from .engine import ConstraintAwareEngine, EngineResult, WorkloadDescriptor

__all__ = [
    "ConstraintClassifier",
    "ConstraintConfig",
    "CostModelConfig",
    "MigrationCostEstimate",
    "MigrationCostModel",
    "MigrationGovernor",
    "RiskInputs",
    "ConstraintAwareEngine",
    "EngineResult",
    "WorkloadDescriptor",
]
