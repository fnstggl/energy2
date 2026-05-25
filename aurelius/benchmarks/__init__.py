"""Constraint-aware benchmark framework for Phase 11.

Provides multi-policy comparison, KPI tracking, regression detection,
and optimization scoring for the constraint-aware Aurelius system.
"""

from .constraint_runner import BenchmarkResult, ConstraintBenchmarkRunner, PolicyResult
from .regression import BenchmarkRegressionChecker
from .report import BenchmarkMetadata, BenchmarkReport, OptimizationScorecard

__all__ = [
    "ConstraintBenchmarkRunner",
    "BenchmarkResult",
    "PolicyResult",
    "BenchmarkReport",
    "BenchmarkMetadata",
    "OptimizationScorecard",
    "BenchmarkRegressionChecker",
]
