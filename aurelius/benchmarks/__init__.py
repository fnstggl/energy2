"""Constraint-aware benchmark framework for Phase 11.

Provides multi-policy comparison, KPI tracking, regression detection,
and optimization scoring for the constraint-aware Aurelius system.
"""

from .constraint_runner import BenchmarkResult, ConstraintBenchmarkRunner, PolicyResult
from .packing import (
    ClusterPackingAnalysis,
    PackingResult,
    analyze_cluster_packing,
    best_fit,
    clairvoyant_lower_bound,
    first_fit,
    first_fit_decreasing,
    greedy_bin_packing,
)
from .realism_audit import RealismAuditReport, run_realism_audit
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
    "run_realism_audit",
    "RealismAuditReport",
    "analyze_cluster_packing",
    "ClusterPackingAnalysis",
    "PackingResult",
    "first_fit",
    "best_fit",
    "first_fit_decreasing",
    "greedy_bin_packing",
    "clairvoyant_lower_bound",
]
