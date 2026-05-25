"""Constraint classifier package for Aurelius constraint-aware orchestration.

This package is read-only over ClusterState. It does NOT touch the optimizer,
execution adapters, or any inference/runtime internals.

Public interface:
    ConstraintClassifier      — scores all 8 constraint families, emits ConstraintAssessment
    ConstraintConfig          — configurable thresholds (all marked # HEURISTIC)
    MigrationCostModel        — conservative heuristic cost/risk estimator for candidate actions
    MigrationCostEstimate     — per-candidate cost/risk breakdown
    ConstraintAwareEngine     — Phase 9: full recommendation pipeline
    EngineResult              — output of ConstraintAwareEngine.run()
    WorkloadDescriptor        — lightweight workload adapter for the engine
"""

from .classifier import ConstraintClassifier, ConstraintConfig
from .cost_model import MigrationCostEstimate, MigrationCostModel, MigrationGovernor
from .engine import ConstraintAwareEngine, EngineResult, WorkloadDescriptor

__all__ = [
    "ConstraintClassifier",
    "ConstraintConfig",
    "MigrationCostEstimate",
    "MigrationCostModel",
    "MigrationGovernor",
    "ConstraintAwareEngine",
    "EngineResult",
    "WorkloadDescriptor",
]
