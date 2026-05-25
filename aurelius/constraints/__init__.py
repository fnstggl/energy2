"""Constraint-aware classification layer for Aurelius.

This package classifies which constraint is currently binding a cluster, based on
a normalized ClusterState snapshot. It does NOT touch any runtime systems (NCCL,
CUDA, KV cache, schedulers). It is read-only over ClusterState.

Public API:
    ConstraintClassifier — stateful classifier with hysteresis
    ConstraintConfig — heuristic threshold configuration
    ScorerResult — per-family scoring output
"""
from .classifier import ConstraintClassifier, ConstraintConfig, ScorerResult

__all__ = [
    "ConstraintClassifier",
    "ConstraintConfig",
    "ScorerResult",
]
