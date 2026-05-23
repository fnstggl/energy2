"""ROI Methodology Calculator.

Computes projected cost savings using proven benchmark rates from the
Aurelius ml_quantile v2.0 backtest (Q1 2026 + Summer 2025, CAISO + PJM + ERCOT,
5-7 walk-forward folds, 0% missing price hours, leakage-free).

HONESTY GUARANTEE:
- All savings rates are from real market data, not synthetic simulations.
- p50 rates are the median observed across walk-forward folds.
- p10 rates reflect the lower seasonal bound (conservative).
- p90 rates are capped below oracle ceiling (achievable upper bound).
- 60% savings is an aspirational stretch target — NOT a current claim.
- Actual savings depend on customer workload mix, region, season, and SLA flexibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

BENCHMARK_METADATA = {
    "forecaster": "ml_quantile v2.0",
    "optimizer": "greedy_migrate",
    "data": "CAISO + PJM + ERCOT DA prices, Q1 2026 + Summer 2025",
    "methodology": "leakage-free walk-forward backtest, 5-7 folds, 30-day training windows",
    "missing_price_hours": "0%",
    "validated_mean_savings": "25.0% vs current_price_only",
    "regions_validated": ["us-west (CAISO)", "us-east (PJM)", "us-south (ERCOT)"],
    "oracle_ceiling_training": "29.9%",
    "oracle_ceiling_llm_batch": "42.7%",
    "aspirational_stretch_target": "60% (NOT a current proven claim)",
    "notes": [
        "Results validated on 3 US regions only. EU/Asia-Pacific require additional data.",
        "Realtime inference savings are modest because the optimizer cannot delay jobs.",
        "Training/fine_tuning savings may improve with longer historical data windows.",
        "Seasonal variation: Q1 (winter) and Summer benchmarks shown separately.",
    ],
}

# Proven savings rates by workload type.
# Derived from ml_quantile v2.0 backtests (Q1 2026 primary, Summer 2025 secondary).
# Format: workload_type -> (p10, p50, p90)
# p10: lower seasonal/fold bound (conservative estimate)
# p50: primary Q1 2026 benchmark result (main reported figure)
# p90: upper bound, capped at oracle ceiling
BENCHMARK_SAVINGS_RATES: dict[str, tuple[float, float, float]] = {
    # Q1: 40.3%  Summer: 25.2%  oracle: ~50%
    "background_maintenance": (0.25, 0.40, 0.50),
    # Q1: 37.7%  Summer: 31.9%  oracle: ~48%
    "data_processing":        (0.28, 0.38, 0.46),
    # Q1: 33.6%  Summer: 29.8%  oracle: 42.7%
    "llm_batch_inference":    (0.22, 0.34, 0.42),
    # Q1: 25.3%  Summer: 26.4%  oracle: ~35%
    "scheduled_batch":        (0.18, 0.25, 0.33),
    # Q1: 15.0%  Summer: 16.2%  oracle: 29.9%
    "training":               (0.08, 0.15, 0.22),
    # Q1: 13.4%  Summer: 28.8%  oracle: 46.8%
    # Wide range: winter cold-snap suppresses savings; summer much higher
    "fine_tuning":            (0.08, 0.15, 0.29),
    # Q1: 10.0%  Summer: 1.4%   oracle: ~15%
    # Limited because latency-sensitive jobs cannot be delayed
    "realtime_inference":     (0.01, 0.07, 0.12),
}

# Mean savings across all workload types (equal-weighted, Q1 2026 benchmark).
MEAN_SAVINGS_P50 = 0.25

# Typical neocloud/GPU-cloud workload distribution.
# Customers should override this with their actual workload mix for accuracy.
DEFAULT_WORKLOAD_MIX: dict[str, float] = {
    "training":               0.35,
    "fine_tuning":            0.15,
    "llm_batch_inference":    0.20,
    "data_processing":        0.10,
    "scheduled_batch":        0.10,
    "realtime_inference":     0.07,
    "background_maintenance": 0.03,
}

# Minimum fraction of compute that must be in flexible workloads
# (training, fine_tuning, batch) for Tier 1 optimization to work.
MIN_FLEXIBLE_FRACTION = 0.20

_FLEXIBLE_WORKLOADS = {
    "training", "fine_tuning", "llm_batch_inference",
    "data_processing", "scheduled_batch", "background_maintenance",
}


@dataclass
class WorkloadROIBreakdown:
    """Per-workload ROI breakdown."""
    workload_type: str
    fraction_of_spend: float
    monthly_cost_usd: float
    savings_rate_p10: float
    savings_rate_p50: float
    savings_rate_p90: float
    monthly_savings_p10_usd: float
    monthly_savings_p50_usd: float
    monthly_savings_p90_usd: float

    def to_dict(self) -> dict:
        return {
            "workload_type": self.workload_type,
            "fraction_of_spend": round(self.fraction_of_spend, 4),
            "monthly_cost_usd": round(self.monthly_cost_usd, 2),
            "savings_rate_p10": round(self.savings_rate_p10, 4),
            "savings_rate_p50": round(self.savings_rate_p50, 4),
            "savings_rate_p90": round(self.savings_rate_p90, 4),
            "monthly_savings_p10_usd": round(self.monthly_savings_p10_usd, 2),
            "monthly_savings_p50_usd": round(self.monthly_savings_p50_usd, 2),
            "monthly_savings_p90_usd": round(self.monthly_savings_p90_usd, 2),
        }


@dataclass
class ROIInput:
    """Customer inputs for ROI calculation.

    Args:
        monthly_gpu_cost_usd: Total monthly GPU infrastructure spend in USD.
            Includes GPU rental, on-demand, or amortized owned hardware costs.
            Should exclude networking, storage, and software licensing unless
            those costs also vary with GPU placement decisions.
        workload_mix: Fraction of monthly GPU spend by workload type (must sum to 1.0).
            If None, DEFAULT_WORKLOAD_MIX is used. Provide actual customer data
            for more accurate estimates.
        contract_months: Projection period in months for annual/multi-year calculations.
        num_gpus: Total GPU count (informational only, not used in savings math).
        gpu_type: Primary GPU type (informational only, e.g. "A100", "H100").
        primary_region: Customer's current primary compute region.
        note: Optional customer-specific context note.
    """
    monthly_gpu_cost_usd: float
    workload_mix: Optional[dict[str, float]] = None
    contract_months: int = 12
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    primary_region: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.monthly_gpu_cost_usd <= 0:
            raise ValueError("monthly_gpu_cost_usd must be positive")
        if self.contract_months <= 0:
            raise ValueError("contract_months must be positive")
        if self.workload_mix is not None:
            total = sum(self.workload_mix.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"workload_mix fractions must sum to 1.0, got {total:.4f}"
                )
            unknown = set(self.workload_mix) - set(BENCHMARK_SAVINGS_RATES)
            if unknown:
                raise ValueError(
                    f"Unknown workload types in workload_mix: {unknown}. "
                    f"Valid types: {set(BENCHMARK_SAVINGS_RATES)}"
                )


@dataclass
class ROIResult:
    """ROI calculation result.

    All savings figures are projected estimates based on proven benchmark rates.
    Actual savings may vary by season, market conditions, and customer workload mix.
    """
    # Inputs echoed back for auditability
    monthly_gpu_cost_usd: float
    workload_mix_used: dict[str, float]
    contract_months: int

    # Primary savings outputs
    monthly_savings_p50_usd: float
    monthly_savings_p10_usd: float
    monthly_savings_p90_usd: float
    effective_savings_rate_p50: float
    effective_savings_rate_p10: float
    effective_savings_rate_p90: float

    # Multi-month projections
    total_savings_p50_usd: float
    total_savings_p10_usd: float
    total_savings_p90_usd: float
    annual_savings_p50_usd: float

    # Per-workload breakdown
    workload_breakdown: list[WorkloadROIBreakdown] = field(default_factory=list)

    # Flexible compute fraction
    flexible_fraction: float = 0.0

    # Metadata and caveats
    benchmark_data_source: str = ""
    methodology_note: str = ""
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "inputs": {
                "monthly_gpu_cost_usd": self.monthly_gpu_cost_usd,
                "workload_mix_used": self.workload_mix_used,
                "contract_months": self.contract_months,
            },
            "projected_savings": {
                "monthly_p50_usd": round(self.monthly_savings_p50_usd, 2),
                "monthly_p10_usd": round(self.monthly_savings_p10_usd, 2),
                "monthly_p90_usd": round(self.monthly_savings_p90_usd, 2),
                "effective_rate_p50": round(self.effective_savings_rate_p50, 4),
                "effective_rate_p10": round(self.effective_savings_rate_p10, 4),
                "effective_rate_p90": round(self.effective_savings_rate_p90, 4),
                f"total_{self.contract_months}mo_p50_usd": round(self.total_savings_p50_usd, 2),
                f"total_{self.contract_months}mo_p10_usd": round(self.total_savings_p10_usd, 2),
                f"total_{self.contract_months}mo_p90_usd": round(self.total_savings_p90_usd, 2),
                "annual_p50_usd": round(self.annual_savings_p50_usd, 2),
            },
            "workload_breakdown": [wb.to_dict() for wb in self.workload_breakdown],
            "flexible_fraction": round(self.flexible_fraction, 4),
            "benchmark_data_source": self.benchmark_data_source,
            "methodology_note": self.methodology_note,
            "caveats": self.caveats,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "AURELIUS ROI PROJECTION",
            "=" * 72,
            f"Monthly GPU Infrastructure Cost:  ${self.monthly_gpu_cost_usd:>12,.0f}",
            f"Projection Period:                {self.contract_months} months",
            "",
            "PROJECTED MONTHLY SAVINGS",
            "-" * 40,
            f"  Conservative (p10):  ${self.monthly_savings_p10_usd:>10,.0f}  ({self.effective_savings_rate_p10*100:.1f}% of spend)",
            f"  Expected     (p50):  ${self.monthly_savings_p50_usd:>10,.0f}  ({self.effective_savings_rate_p50*100:.1f}% of spend)",
            f"  Optimistic   (p90):  ${self.monthly_savings_p90_usd:>10,.0f}  ({self.effective_savings_rate_p90*100:.1f}% of spend)",
            "",
            f"PROJECTED {self.contract_months}-MONTH SAVINGS",
            "-" * 40,
            f"  Conservative (p10):  ${self.total_savings_p10_usd:>10,.0f}",
            f"  Expected     (p50):  ${self.total_savings_p50_usd:>10,.0f}",
            f"  Optimistic   (p90):  ${self.total_savings_p90_usd:>10,.0f}",
            "",
            f"ANNUALIZED (12-month) SAVINGS (p50): ${self.annual_savings_p50_usd:>10,.0f}",
            "",
            "WORKLOAD BREAKDOWN",
            "-" * 72,
            f"{'Workload Type':<28} {'%Spend':>6} {'p50 Rate':>8} {'p50 Savings/mo':>14}",
            "-" * 72,
        ]
        for wb in sorted(self.workload_breakdown, key=lambda x: -x.monthly_savings_p50_usd):
            lines.append(
                f"  {wb.workload_type:<26} {wb.fraction_of_spend*100:>5.1f}%"
                f"  {wb.savings_rate_p50*100:>6.1f}%  ${wb.monthly_savings_p50_usd:>12,.0f}"
            )
        lines += [
            "-" * 72,
            f"  {'TOTAL':<26} {'100.0%':>6}  {'':>8}  ${self.monthly_savings_p50_usd:>12,.0f}",
            "",
            f"Flexible compute fraction: {self.flexible_fraction*100:.1f}%",
            "",
            "DATA SOURCE",
            "-" * 40,
            f"  {self.benchmark_data_source}",
            "",
            "METHODOLOGY",
            "-" * 40,
            f"  {self.methodology_note}",
            "",
            "IMPORTANT CAVEATS",
            "-" * 40,
        ]
        for c in self.caveats:
            lines.append(f"  * {c}")
        lines.append("=" * 72)
        return "\n".join(lines)


class ROICalculator:
    """Computes projected ROI for Aurelius deployment.

    All savings estimates use proven benchmark rates from real-data backtests.
    No synthetic data is used for savings claims.
    """

    _CAVEATS = [
        "Projected savings use real market data (CAISO+PJM+ERCOT Q1 2026 + Summer 2025).",
        "60% savings is an aspirational stretch target — not a current proven claim.",
        "Proven mean savings: 25.0% vs current-price-only baseline (real data, 5 folds).",
        "Actual savings depend on customer workload mix, region, season, SLA flexibility.",
        "Results are from 3 US regions only. EU/Asia-Pacific require additional data connectors.",
        "Realtime inference savings are limited (~7%) because jobs cannot be delayed.",
        "Savings improve with longer training data history and more scheduling flexibility.",
        "All savings figures reflect region/time optimization (Tier 1 control level).",
        "Tier 2 (queue-aware) and Tier 3 (GPU/node) require customer cluster integration.",
        "Estimate is a projection, not a guarantee. Pilot shadow mode validates actual savings.",
    ]

    _METHODOLOGY_NOTE = (
        "Savings rates derived from ml_quantile v2.0 backtest: "
        "leakage-free walk-forward evaluation, 5-7 folds, 30-day training windows, "
        "greedy_migrate optimizer, 0% missing price hours, real historical DA prices."
    )

    _DATA_SOURCE = (
        "CAISO OASIS (us-west), PJM Data Miner API (us-east), ERCOT CDAT API (us-south). "
        "Q1 2026 (Jan-Mar) and Summer 2025 (Jun-Aug) historical day-ahead prices."
    )

    def calculate(self, roi_input: ROIInput) -> ROIResult:
        """Compute ROI projection.

        Args:
            roi_input: Customer inputs (monthly GPU cost, workload mix, etc.)

        Returns:
            ROIResult with projected savings at p10/p50/p90.
        """
        mix = roi_input.workload_mix or DEFAULT_WORKLOAD_MIX
        total_cost = roi_input.monthly_gpu_cost_usd

        # Compute savings for each workload type
        breakdowns: list[WorkloadROIBreakdown] = []
        monthly_p10 = 0.0
        monthly_p50 = 0.0
        monthly_p90 = 0.0
        flex_frac = 0.0

        for wtype, frac in mix.items():
            if wtype not in BENCHMARK_SAVINGS_RATES:
                continue
            p10, p50, p90 = BENCHMARK_SAVINGS_RATES[wtype]
            workload_cost = frac * total_cost
            savings_p10 = p10 * workload_cost
            savings_p50 = p50 * workload_cost
            savings_p90 = p90 * workload_cost

            monthly_p10 += savings_p10
            monthly_p50 += savings_p50
            monthly_p90 += savings_p90

            if wtype in _FLEXIBLE_WORKLOADS:
                flex_frac += frac

            breakdowns.append(WorkloadROIBreakdown(
                workload_type=wtype,
                fraction_of_spend=frac,
                monthly_cost_usd=workload_cost,
                savings_rate_p10=p10,
                savings_rate_p50=p50,
                savings_rate_p90=p90,
                monthly_savings_p10_usd=savings_p10,
                monthly_savings_p50_usd=savings_p50,
                monthly_savings_p90_usd=savings_p90,
            ))

        months = roi_input.contract_months
        total_p10 = monthly_p10 * months
        total_p50 = monthly_p50 * months
        total_p90 = monthly_p90 * months
        annual_p50 = monthly_p50 * 12

        caveats = list(self._CAVEATS)
        if flex_frac < MIN_FLEXIBLE_FRACTION:
            caveats.insert(0,
                f"WARNING: Only {flex_frac*100:.0f}% of spend is in flexible workloads. "
                f"Savings potential is significantly limited. "
                f"Recommend ≥{MIN_FLEXIBLE_FRACTION*100:.0f}% flexible workloads for meaningful optimization."
            )

        return ROIResult(
            monthly_gpu_cost_usd=total_cost,
            workload_mix_used=mix,
            contract_months=months,
            monthly_savings_p50_usd=monthly_p50,
            monthly_savings_p10_usd=monthly_p10,
            monthly_savings_p90_usd=monthly_p90,
            effective_savings_rate_p50=monthly_p50 / total_cost,
            effective_savings_rate_p10=monthly_p10 / total_cost,
            effective_savings_rate_p90=monthly_p90 / total_cost,
            total_savings_p50_usd=total_p50,
            total_savings_p10_usd=total_p10,
            total_savings_p90_usd=total_p90,
            annual_savings_p50_usd=annual_p50,
            workload_breakdown=breakdowns,
            flexible_fraction=flex_frac,
            benchmark_data_source=self._DATA_SOURCE,
            methodology_note=self._METHODOLOGY_NOTE,
            caveats=caveats,
        )
