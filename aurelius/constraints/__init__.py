"""Constraint classifier package for Aurelius constraint-aware orchestration.

This package is read-only over ClusterState. It does NOT touch the optimizer,
execution adapters, or any inference/runtime internals.

Public interface:
    ConstraintClassifier  — scores all 8 constraint families, emits ConstraintAssessment
    ConstraintConfig      — configurable thresholds (all marked # HEURISTIC)
"""

from .classifier import ConstraintClassifier, ConstraintConfig

__all__ = [
    "ConstraintClassifier",
    "ConstraintConfig",
]
