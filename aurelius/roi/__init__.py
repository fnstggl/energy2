"""ROI Methodology Calculator for Aurelius.

Computes projected cost savings for a customer based on:
- Their total GPU infrastructure spend
- Their workload type distribution
- Proven benchmark savings rates (leakage-free, real data)

Usage:
    from aurelius.roi import ROICalculator, ROIInput
    calc = ROICalculator()
    result = calc.calculate(ROIInput(monthly_gpu_cost_usd=500_000))
    print(result.to_text())
"""

from .calculator import (
    BENCHMARK_METADATA,
    BENCHMARK_SAVINGS_RATES,
    DEFAULT_WORKLOAD_MIX,
    MEAN_SAVINGS_P50,
    ROICalculator,
    ROIInput,
    ROIResult,
    WorkloadROIBreakdown,
)

__all__ = [
    "ROICalculator",
    "ROIInput",
    "ROIResult",
    "WorkloadROIBreakdown",
    "BENCHMARK_SAVINGS_RATES",
    "DEFAULT_WORKLOAD_MIX",
    "MEAN_SAVINGS_P50",
    "BENCHMARK_METADATA",
]
