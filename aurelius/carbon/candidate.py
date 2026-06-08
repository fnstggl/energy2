"""Carbon-aware candidate evaluation.

Given a job and a set of candidate placements (start_time, region), compute each
candidate's forecast emissions from forecast MOER, apply the HARD carbon
constraints, and rank the feasible candidates with carbon-aware scoring. This is
the composable decision primitive that makes carbon actually change a placement
(see tests/test_carbon_scoring_candidate.py).

It is intentionally side-effect-free and independent of the legacy greedy
scheduler so it can be unit-tested in isolation and wired in opt-in without
risking the existing cost-only benchmarks.

Units: prices are $/MWh; MOER is gCO2/kWh (already normalized). Hours are real
hours; the last partial hour is pro-rated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .accounting import (
    CarbonDataKind,
    CarbonSignalType,
    MoerInterval,
    WorkloadCarbonRecord,
    build_workload_carbon_record,
)
from .constraints import CarbonConstraintResult, CarbonConstraints, check_carbon_constraints
from .scoring import (
    CandidateScoreInputs,
    ScoreWeights,
    score_candidates_carbon_priced,
    score_candidates_normalized,
)


@dataclass
class CandidatePlacement:
    """One placement option for a job."""

    region: str
    start_time: datetime
    # Per-job physicals (carried so the evaluator stays job-agnostic).
    power_kw: float
    duration_hours: float
    utilization_fraction: float = 1.0
    pue: float = 1.0
    risk: float = 0.0
    migration_penalty: float = 0.0


@dataclass
class EvaluatedCandidate:
    """A placement after carbon accounting, constraint check, and scoring."""

    placement: CandidatePlacement
    record: WorkloadCarbonRecord
    constraint_result: CarbonConstraintResult
    forecast_emissions_kgco2: float
    forecast_moer_gco2_per_kwh: float
    carbon_data_coverage_pct: float
    cost_usd: float
    score: Optional[float] = None  # filled for feasible candidates only

    @property
    def feasible(self) -> bool:
        return self.constraint_result.ok

    def to_dict(self) -> dict:
        return {
            "region": self.placement.region,
            "start_time": self.placement.start_time.isoformat(),
            "forecast_emissions_kgco2": round(self.forecast_emissions_kgco2, 6),
            "forecast_moer_gco2_per_kwh": round(self.forecast_moer_gco2_per_kwh, 4),
            "carbon_data_coverage_pct": round(self.carbon_data_coverage_pct, 3),
            "cost_usd": round(self.cost_usd, 6),
            "score": None if self.score is None else round(self.score, 6),
            "carbon_constraint_status": self.constraint_result.to_dict(),
        }


@dataclass
class CandidateEvaluation:
    """Full result of evaluating a job's candidates."""

    job_id: str
    candidates: list[EvaluatedCandidate]
    best: Optional[EvaluatedCandidate]
    rejected: list[EvaluatedCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "best": self.best.to_dict() if self.best else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "n_rejected": len(self.rejected),
        }


def _hourly_intervals(start: datetime, duration_hours: float) -> list[tuple[datetime, float]]:
    """Split [start, start+duration) into hourly (hour_key, fraction) buckets."""
    out: list[tuple[datetime, float]] = []
    remaining = duration_hours
    current = start.replace(minute=0, second=0, microsecond=0)
    # Account for a start that isn't on the hour boundary by offsetting the first
    # bucket's fraction. Aurelius schedules on hour boundaries, but be safe.
    first_offset = (start - current).total_seconds() / 3600.0
    remaining_in_first = max(0.0, 1.0 - first_offset)
    while remaining > 1e-9:
        frac = min(remaining, remaining_in_first if not out else 1.0)
        out.append((current, frac))
        remaining -= frac
        current = current + timedelta(hours=1)
    return out


def _moer_intervals_for(
    region: str,
    start: datetime,
    duration_hours: float,
    moer_data: dict[str, dict[datetime, float]],
    moer_is_real: bool,
) -> tuple[list[MoerInterval], float]:
    """Build per-hour MoerInterval list (None for missing hours — no default)."""
    region_moer = moer_data.get(region, {})
    intervals: list[MoerInterval] = []
    # One interval per hour bucket; fractional hours still count as one interval
    # but the accounting uses the fraction as interval_hours (handled by caller).
    for hour_key, _frac in _hourly_intervals(start, duration_hours):
        val = region_moer.get(hour_key)
        intervals.append(MoerInterval(value_gco2_per_kwh=val, is_real=moer_is_real))
    return intervals, 1.0


def _cost_usd(
    region: str,
    start: datetime,
    duration_hours: float,
    power_kw: float,
    utilization_fraction: float,
    pue: float,
    price_data: dict[str, dict[datetime, float]],
) -> float:
    """Energy $ over the window. Missing-price hours contribute 0 (and are not
    silently priced); this evaluator's contract is full price coverage for the
    candidate set being compared."""
    region_prices = price_data.get(region, {})
    total = 0.0
    for hour_key, frac in _hourly_intervals(start, duration_hours):
        price = region_prices.get(hour_key)
        if price is None:
            continue
        facility_kwh = power_kw * utilization_fraction * frac * pue
        total += (price / 1000.0) * facility_kwh
    return total


def evaluate_carbon_candidates(
    *,
    job_id: str,
    scheduler_name: str,
    placements: list[CandidatePlacement],
    price_data: dict[str, dict[datetime, float]],
    moer_data: dict[str, dict[datetime, float]],
    constraints: Optional[CarbonConstraints] = None,
    weights: Optional[ScoreWeights] = None,
    carbon_price_usd_per_tonne: Optional[float] = None,
    carbon_data_source: str = "watttime_co2_moer",
    carbon_signal_type: CarbonSignalType = CarbonSignalType.MARGINAL_MOER,
    carbon_data_kind: CarbonDataKind = CarbonDataKind.FORECAST,
    moer_is_real: bool = True,
    baseline_emissions_kgco2: Optional[float] = None,
) -> CandidateEvaluation:
    """Score a job's candidate placements with hard carbon constraints applied.

    Scoring mode (Phase 5): if ``carbon_price_usd_per_tonne`` is given, uses
    Option B (carbon priced to USD); otherwise Option A (normalize cost &
    emissions, weighted). Hard-constraint-violating candidates are excluded from
    selection but retained (with their rejection reason) for explainability.
    """
    constraints = constraints or CarbonConstraints()
    evaluated: list[EvaluatedCandidate] = []

    for p in placements:
        intervals, interval_hours = _moer_intervals_for(
            p.region, p.start_time, p.duration_hours, moer_data, moer_is_real
        )
        record = build_workload_carbon_record(
            job_id=job_id,
            scheduler_name=scheduler_name,
            baseline_name=scheduler_name,
            region=p.region,
            start_time_utc=p.start_time,
            end_time_utc=p.start_time + timedelta(hours=p.duration_hours),
            power_kw=p.power_kw,
            utilization_fraction=p.utilization_fraction,
            pue=p.pue,
            moer_intervals=intervals,
            interval_hours=interval_hours,
            carbon_data_source=carbon_data_source,
            carbon_signal_type=carbon_signal_type,
            carbon_data_kind=carbon_data_kind,
        )
        cost = _cost_usd(
            p.region, p.start_time, p.duration_hours, p.power_kw,
            p.utilization_fraction, p.pue, price_data,
        )
        cres = check_carbon_constraints(
            record, constraints, baseline_emissions_kgco2=baseline_emissions_kgco2
        )
        evaluated.append(EvaluatedCandidate(
            placement=p,
            record=record,
            constraint_result=cres,
            forecast_emissions_kgco2=record.emissions_kgco2,
            forecast_moer_gco2_per_kwh=record.moer_gco2_per_kwh,
            carbon_data_coverage_pct=record.carbon_data_coverage_pct,
            cost_usd=cost,
        ))

    feasible = [c for c in evaluated if c.feasible]
    rejected = [c for c in evaluated if not c.feasible]

    if feasible:
        score_inputs = [
            CandidateScoreInputs(
                cost_usd=c.cost_usd,
                emissions_kgco2=c.forecast_emissions_kgco2,
                risk=c.placement.risk,
                migration_penalty=c.placement.migration_penalty,
            )
            for c in feasible
        ]
        if carbon_price_usd_per_tonne is not None:
            scores = score_candidates_carbon_priced(score_inputs, carbon_price_usd_per_tonne)
        else:
            scores = score_candidates_normalized(score_inputs, weights)
        for c, s in zip(feasible, scores):
            c.score = s
        # Lower score is better; tie-break deterministically by region then start.
        best = min(feasible, key=lambda c: (c.score, c.placement.region, c.placement.start_time))
    else:
        best = None

    return CandidateEvaluation(
        job_id=job_id,
        candidates=evaluated,
        best=best,
        rejected=rejected,
    )


__all__ = [
    "CandidatePlacement",
    "EvaluatedCandidate",
    "CandidateEvaluation",
    "evaluate_carbon_candidates",
]
