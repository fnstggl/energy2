"""Phase 6/11.3 — hard carbon constraints with explainable rejection."""

from datetime import datetime, timezone

import pytest

from aurelius.carbon.accounting import (
    CarbonDataKind,
    CarbonSignalType,
    MoerInterval,
    build_workload_carbon_record,
)
from aurelius.carbon.constraints import (
    REJECT_INSUFFICIENT_SAVINGS,
    REJECT_LOW_COVERAGE,
    REJECT_MISSING_REAL_CARBON,
    REJECT_OVER_CARBON_BUDGET,
    REJECT_OVER_INTENSITY,
    REJECT_OVER_JOB_BUDGET,
    CarbonConstraints,
    check_carbon_constraints,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _rec(moer_list, kind=CarbonDataKind.HISTORICAL, power_kw=100.0, pue=1.0):
    return build_workload_carbon_record(
        job_id="j", scheduler_name="opt", baseline_name="opt", region="us-west",
        start_time_utc=T0, end_time_utc=T0,
        power_kw=power_kw, utilization_fraction=1.0, pue=pue,
        moer_intervals=[MoerInterval(v) for v in moer_list], interval_hours=1.0,
        carbon_data_source="watttime_co2_moer",
        carbon_signal_type=CarbonSignalType.MARGINAL_MOER,
        carbon_data_kind=kind,
    )


class TestNoConstraintsPass:
    def test_default_constraints_pass(self):
        res = check_carbon_constraints(_rec([300.0, 300.0]), CarbonConstraints())
        assert res.ok
        assert res.rejected_by is None


class TestRequireRealCarbon:
    def test_missing_moer_rejected_when_real_required(self):
        rec = _rec([None, None])
        res = check_carbon_constraints(rec, CarbonConstraints(require_real_carbon_data=True))
        assert not res.ok
        assert res.rejected_by == REJECT_MISSING_REAL_CARBON

    def test_synthetic_rejected_when_real_required(self):
        rec = _rec([300.0], kind=CarbonDataKind.SYNTHETIC)
        res = check_carbon_constraints(rec, CarbonConstraints(require_real_carbon_data=True))
        assert not res.ok
        assert res.rejected_by == REJECT_MISSING_REAL_CARBON


class TestCoverage:
    def test_low_coverage_rejected(self):
        rec = _rec([300.0, None])  # 50% coverage
        res = check_carbon_constraints(
            rec, CarbonConstraints(minimum_carbon_data_coverage_pct=90.0)
        )
        assert not res.ok
        assert res.rejected_by == REJECT_LOW_COVERAGE
        assert res.candidate_value == pytest.approx(50.0)
        assert res.carbon_constraint_value == 90.0


class TestJobBudget:
    def test_over_job_budget_rejected_with_values(self):
        # 100kW * 2h * 500/1000 = 100 kg emissions; budget 50 kg.
        rec = _rec([500.0, 500.0])
        res = check_carbon_constraints(rec, CarbonConstraints(max_emissions_kgco2_per_job=50.0))
        assert not res.ok
        assert res.rejected_by == REJECT_OVER_JOB_BUDGET
        assert res.candidate_value == pytest.approx(100.0)
        assert res.carbon_constraint_value == 50.0

    def test_under_job_budget_ok(self):
        rec = _rec([100.0, 100.0])  # 20 kg
        res = check_carbon_constraints(rec, CarbonConstraints(max_emissions_kgco2_per_job=50.0))
        assert res.ok


class TestIntensity:
    def test_over_intensity_rejected(self):
        rec = _rec([600.0, 600.0])  # intensity 600 gCO2/kWh
        res = check_carbon_constraints(
            rec, CarbonConstraints(max_emissions_intensity_gco2_per_kwh=400.0)
        )
        assert not res.ok
        assert res.rejected_by == REJECT_OVER_INTENSITY
        assert res.candidate_value == pytest.approx(600.0)


class TestSavingsVsBaseline:
    def test_insufficient_savings_rejected(self):
        rec = _rec([450.0, 450.0])  # 90 kg vs baseline 100 kg => 10% savings
        res = check_carbon_constraints(
            rec, CarbonConstraints(min_carbon_savings_pct_vs_baseline=20.0),
            baseline_emissions_kgco2=100.0,
        )
        assert not res.ok
        assert res.rejected_by == REJECT_INSUFFICIENT_SAVINGS
        assert res.candidate_value == pytest.approx(10.0)

    def test_sufficient_savings_ok(self):
        rec = _rec([300.0, 300.0])  # 60 kg vs 100 => 40% savings
        res = check_carbon_constraints(
            rec, CarbonConstraints(min_carbon_savings_pct_vs_baseline=20.0),
            baseline_emissions_kgco2=100.0,
        )
        assert res.ok

    def test_missing_baseline_fails_closed(self):
        rec = _rec([300.0, 300.0])
        res = check_carbon_constraints(
            rec, CarbonConstraints(min_carbon_savings_pct_vs_baseline=20.0),
            baseline_emissions_kgco2=None,
        )
        assert not res.ok
        assert res.rejected_by == REJECT_INSUFFICIENT_SAVINGS


class TestCarbonBudget:
    def test_over_remaining_budget_rejected(self):
        rec = _rec([500.0, 500.0])  # 100 kg
        res = check_carbon_constraints(
            rec, CarbonConstraints(max_carbon_budget_kgco2=80.0)
        )
        assert not res.ok
        assert res.rejected_by == REJECT_OVER_CARBON_BUDGET

    def test_remaining_budget_override_used(self):
        rec = _rec([500.0, 500.0])  # 100 kg
        res = check_carbon_constraints(
            rec, CarbonConstraints(max_carbon_budget_kgco2=1000.0),
            remaining_budget_kgco2=50.0,
        )
        assert not res.ok
        assert res.rejected_by == REJECT_OVER_CARBON_BUDGET
        assert res.carbon_constraint_value == 50.0


class TestDataQualityCheckedFirst:
    def test_missing_data_beats_numeric_pass(self):
        # Even if emissions would pass every numeric cap, missing real data is
        # rejected first when required (a number from missing MOER is meaningless).
        rec = _rec([None])
        res = check_carbon_constraints(
            rec,
            CarbonConstraints(
                require_real_carbon_data=True,
                max_emissions_kgco2_per_job=1e9,
            ),
        )
        assert res.rejected_by == REJECT_MISSING_REAL_CARBON
