"""Hard carbon constraints — candidate rejection, not just soft penalties.

A soft objective penalty (``beta * carbon_cost``) can always be out-voted by a
cheap-but-dirty option. These are HARD constraints: a candidate that violates
one is REJECTED outright, with a structured, explainable reason.

Supported constraints
----------------------
* ``max_emissions_kgco2_per_job``            — absolute per-job cap
* ``max_emissions_intensity_gco2_per_kwh``   — effective intensity cap
* ``min_carbon_savings_pct_vs_baseline``     — must beat a baseline by X%
* ``max_carbon_budget_kgco2``                — remaining customer/run budget
* ``require_real_carbon_data``               — reject if MOER missing/synthetic
* ``minimum_carbon_data_coverage_pct``       — reject if coverage too low

Rejections are explainable: ``rejected_by``, ``rejection_reason``,
``carbon_constraint_value``, ``candidate_value``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .accounting import WorkloadCarbonRecord


@dataclass
class CarbonConstraints:
    """Hard carbon constraints applied to candidate placements.

    All fields default to "no constraint" (None / False) so existing cost-only
    flows are unaffected until carbon constraints are explicitly configured.
    """

    max_emissions_kgco2_per_job: Optional[float] = None
    max_emissions_intensity_gco2_per_kwh: Optional[float] = None
    min_carbon_savings_pct_vs_baseline: Optional[float] = None
    max_carbon_budget_kgco2: Optional[float] = None
    require_real_carbon_data: bool = False
    minimum_carbon_data_coverage_pct: Optional[float] = None

    @property
    def any_active(self) -> bool:
        return any([
            self.max_emissions_kgco2_per_job is not None,
            self.max_emissions_intensity_gco2_per_kwh is not None,
            self.min_carbon_savings_pct_vs_baseline is not None,
            self.max_carbon_budget_kgco2 is not None,
            self.require_real_carbon_data,
            self.minimum_carbon_data_coverage_pct is not None,
        ])


# Stable rejection codes (no embedded numbers — numbers go in the detail fields).
REJECT_MISSING_REAL_CARBON = "carbon_data_not_real"
REJECT_LOW_COVERAGE = "carbon_coverage_below_minimum"
REJECT_OVER_JOB_BUDGET = "emissions_exceed_job_budget"
REJECT_OVER_INTENSITY = "emissions_intensity_exceeds_threshold"
REJECT_INSUFFICIENT_SAVINGS = "insufficient_savings_vs_baseline"
REJECT_OVER_CARBON_BUDGET = "exceeds_remaining_carbon_budget"


@dataclass
class CarbonConstraintResult:
    """Outcome of checking one candidate against the carbon constraints."""

    status: str                          # "ok" or "rejected"
    rejected_by: Optional[str] = None    # which constraint (stable code)
    rejection_reason: Optional[str] = None
    carbon_constraint_value: Optional[float] = None
    candidate_value: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "rejected_by": self.rejected_by,
            "rejection_reason": self.rejection_reason,
            "carbon_constraint_value": self.carbon_constraint_value,
            "candidate_value": self.candidate_value,
        }


def _reject(code: str, reason: str, limit: Optional[float], value: Optional[float]) -> CarbonConstraintResult:
    return CarbonConstraintResult(
        status="rejected",
        rejected_by=code,
        rejection_reason=reason,
        carbon_constraint_value=limit,
        candidate_value=value,
    )


def check_carbon_constraints(
    record: WorkloadCarbonRecord,
    constraints: CarbonConstraints,
    *,
    baseline_emissions_kgco2: Optional[float] = None,
    remaining_budget_kgco2: Optional[float] = None,
) -> CarbonConstraintResult:
    """Check a candidate's carbon record against the hard constraints.

    Order matters: data-quality gates (real/coverage) run first, because an
    emissions number computed from missing or synthetic MOER must not be allowed
    to "pass" a numeric cap. ``baseline_emissions_kgco2`` is required to evaluate
    ``min_carbon_savings_pct_vs_baseline``; ``remaining_budget_kgco2`` (which may
    differ from the static ``max_carbon_budget_kgco2`` when a budget is consumed
    across many jobs) is required for the budget gate.
    """
    c = constraints

    # 1. require_real_carbon_data — missing/synthetic MOER rejects the candidate.
    if c.require_real_carbon_data and not record.carbon_data_is_real:
        return _reject(
            REJECT_MISSING_REAL_CARBON,
            "real carbon data required but MOER is missing or synthetic",
            1.0, 0.0,
        )

    # 2. minimum coverage.
    if c.minimum_carbon_data_coverage_pct is not None:
        if record.carbon_data_coverage_pct < c.minimum_carbon_data_coverage_pct:
            return _reject(
                REJECT_LOW_COVERAGE,
                "carbon data coverage below minimum",
                c.minimum_carbon_data_coverage_pct,
                record.carbon_data_coverage_pct,
            )

    # 3. per-job emissions budget.
    if c.max_emissions_kgco2_per_job is not None:
        if record.emissions_kgco2 > c.max_emissions_kgco2_per_job:
            return _reject(
                REJECT_OVER_JOB_BUDGET,
                "candidate emissions exceed per-job budget",
                c.max_emissions_kgco2_per_job,
                record.emissions_kgco2,
            )

    # 4. emissions intensity threshold.
    if c.max_emissions_intensity_gco2_per_kwh is not None:
        intensity = record.emissions_intensity_gco2_per_kwh
        if intensity > c.max_emissions_intensity_gco2_per_kwh:
            return _reject(
                REJECT_OVER_INTENSITY,
                "candidate emissions intensity exceeds threshold",
                c.max_emissions_intensity_gco2_per_kwh,
                intensity,
            )

    # 5. required savings vs a configured baseline.
    if c.min_carbon_savings_pct_vs_baseline is not None:
        if baseline_emissions_kgco2 is None or baseline_emissions_kgco2 <= 0:
            # Cannot prove savings without a positive baseline — fail closed.
            return _reject(
                REJECT_INSUFFICIENT_SAVINGS,
                "savings-vs-baseline required but no positive baseline emissions provided",
                c.min_carbon_savings_pct_vs_baseline,
                None,
            )
        savings_pct = 100.0 * (baseline_emissions_kgco2 - record.emissions_kgco2) / baseline_emissions_kgco2
        if savings_pct < c.min_carbon_savings_pct_vs_baseline:
            return _reject(
                REJECT_INSUFFICIENT_SAVINGS,
                "candidate fails required carbon savings vs baseline",
                c.min_carbon_savings_pct_vs_baseline,
                savings_pct,
            )

    # 6. remaining carbon budget for the customer/run.
    budget = remaining_budget_kgco2 if remaining_budget_kgco2 is not None else c.max_carbon_budget_kgco2
    if budget is not None:
        if record.emissions_kgco2 > budget:
            return _reject(
                REJECT_OVER_CARBON_BUDGET,
                "candidate emissions exceed remaining carbon budget",
                budget,
                record.emissions_kgco2,
            )

    return CarbonConstraintResult(status="ok")


__all__ = [
    "CarbonConstraints",
    "CarbonConstraintResult",
    "check_carbon_constraints",
    "REJECT_MISSING_REAL_CARBON",
    "REJECT_LOW_COVERAGE",
    "REJECT_OVER_JOB_BUDGET",
    "REJECT_OVER_INTENSITY",
    "REJECT_INSUFFICIENT_SAVINGS",
    "REJECT_OVER_CARBON_BUDGET",
]
