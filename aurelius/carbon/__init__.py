"""Authoritative, constraint-aware carbon accounting for Aurelius.

This package is the single source of truth for carbon emissions, savings, and
carbon constraints. Read ``docs/CARBON_AUDIT.md`` for the audit and the contract.

Key entry points:
    accounting.emissions_kgco2 / build_workload_carbon_record  -- the formula + record
    regions.watttime_ba_map / validate_carbon_region_mapping   -- one region mapping
    constraints.check_carbon_constraints                       -- hard constraints
    candidate.evaluate_carbon_candidates                       -- carbon-aware selection
    migration.evaluate_migration_carbon                        -- carbon-aware migration
    replay.replay_schedule_carbon / compare_baselines_carbon   -- realized replay + savings
    forecast_realized.ForecastRealizedCarbon                   -- forecast vs realized
"""

from __future__ import annotations

from .accounting import (
    CARBON_CALCULATION_VERSION,
    CarbonDataKind,
    CarbonSignalType,
    MoerInterval,
    WorkloadCarbonRecord,
    aggregate_records,
    build_workload_carbon_record,
    emissions_intensity_gco2_per_kwh,
    emissions_kgco2,
    energy_kwh,
)
from .candidate import (
    CandidateEvaluation,
    CandidatePlacement,
    EvaluatedCandidate,
    evaluate_carbon_candidates,
)
from .constraints import (
    CarbonConstraintResult,
    CarbonConstraints,
    check_carbon_constraints,
)
from .forecast_realized import ForecastRealizedCarbon
from .migration import (
    MigrationCarbonAssessment,
    MigrationCarbonOverheadMode,
    evaluate_migration_carbon,
)
from .regions import (
    CARBON_AVAILABLE,
    CARBON_UNAVAILABLE,
    assert_optimizer_evaluator_consistency,
    carbon_region_status,
    validate_carbon_region_mapping,
    watttime_ba,
    watttime_ba_map,
)
from .replay import (
    ScheduleCarbonResult,
    compare_baselines_carbon,
    replay_schedule_carbon,
)
from .scoring import (
    CandidateScoreInputs,
    ScoreWeights,
    carbon_cost_usd,
    score_candidates_carbon_priced,
    score_candidates_normalized,
)

__all__ = [
    # accounting
    "CARBON_CALCULATION_VERSION",
    "CarbonDataKind",
    "CarbonSignalType",
    "MoerInterval",
    "WorkloadCarbonRecord",
    "aggregate_records",
    "build_workload_carbon_record",
    "emissions_intensity_gco2_per_kwh",
    "emissions_kgco2",
    "energy_kwh",
    # regions
    "CARBON_AVAILABLE",
    "CARBON_UNAVAILABLE",
    "assert_optimizer_evaluator_consistency",
    "carbon_region_status",
    "validate_carbon_region_mapping",
    "watttime_ba",
    "watttime_ba_map",
    # constraints
    "CarbonConstraintResult",
    "CarbonConstraints",
    "check_carbon_constraints",
    # scoring
    "CandidateScoreInputs",
    "ScoreWeights",
    "carbon_cost_usd",
    "score_candidates_carbon_priced",
    "score_candidates_normalized",
    # candidate
    "CandidateEvaluation",
    "CandidatePlacement",
    "EvaluatedCandidate",
    "evaluate_carbon_candidates",
    # migration
    "MigrationCarbonAssessment",
    "MigrationCarbonOverheadMode",
    "evaluate_migration_carbon",
    # replay
    "ScheduleCarbonResult",
    "compare_baselines_carbon",
    "replay_schedule_carbon",
    # forecast vs realized
    "ForecastRealizedCarbon",
]
