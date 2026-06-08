"""Phase 11.1/11.2/11.3 — unit consistency, conversion, missing-data handling.

Covers:
* emissions formula in kW/MW, hours/5-min intervals, g/kg
* WattTime lbs/MWh -> gCO2/kWh conversion
* missing MOER never silently becomes a default (no 400 fallback)
"""

from datetime import datetime, timezone

import pytest

from aurelius.carbon.accounting import (
    CARBON_CALCULATION_VERSION,
    CarbonDataKind,
    CarbonSignalType,
    MoerInterval,
    aggregate_records,
    build_workload_carbon_record,
    emissions_intensity_gco2_per_kwh,
    emissions_kgco2,
    energy_kwh,
)
from aurelius.ingestion.grid_apis.watttime import _LBS_PER_MWH_TO_GCO2_PER_KWH

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 11.1 Unit consistency
# ---------------------------------------------------------------------------

class TestEmissionsFormula:
    def test_canonical_example_kg(self):
        # 100 kW * 1.0 util * 2 h * 1.0 pue * 500 gCO2/kWh / 1000 = 100 kg
        assert emissions_kgco2(100.0, 1.0, 2.0, 1.0, 500.0) == pytest.approx(100.0)

    def test_kw_vs_mw_scale(self):
        # 1 MW expressed as 1000 kW must be 1000x a 1 kW run, same everything else.
        e_1kw = emissions_kgco2(1.0, 1.0, 1.0, 1.0, 400.0)
        e_1mw = emissions_kgco2(1000.0, 1.0, 1.0, 1.0, 400.0)
        assert e_1mw == pytest.approx(1000.0 * e_1kw)

    def test_pue_multiplies_emissions(self):
        base = emissions_kgco2(100.0, 1.0, 2.0, 1.0, 500.0)
        with_pue = emissions_kgco2(100.0, 1.0, 2.0, 1.4, 500.0)
        assert with_pue == pytest.approx(base * 1.4)

    def test_utilization_scales_linearly(self):
        full = emissions_kgco2(100.0, 1.0, 2.0, 1.0, 500.0)
        half = emissions_kgco2(100.0, 0.5, 2.0, 1.0, 500.0)
        assert half == pytest.approx(full * 0.5)

    def test_hours_vs_five_minute_intervals(self):
        # One 1-hour interval == twelve 5-minute (1/12 h) intervals, same MOER.
        one_hour = emissions_kgco2(100.0, 1.0, 1.0, 1.0, 600.0)
        five_min = sum(
            emissions_kgco2(100.0, 1.0, 5.0 / 60.0, 1.0, 600.0) for _ in range(12)
        )
        assert five_min == pytest.approx(one_hour, rel=1e-9)

    def test_grams_vs_kg_consistency(self):
        # emissions_kgco2 returns kg; grams = kg*1000; intensity recovers MOER.
        kg = emissions_kgco2(100.0, 1.0, 2.0, 1.0, 500.0)
        e_kwh = energy_kwh(100.0, 1.0, 2.0, 1.0)
        intensity = emissions_intensity_gco2_per_kwh(kg, e_kwh)
        assert intensity == pytest.approx(500.0)  # back out the original MOER

    def test_zero_energy_zero_intensity(self):
        assert emissions_intensity_gco2_per_kwh(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 11.2 WattTime unit conversion
# ---------------------------------------------------------------------------

class TestWattTimeConversion:
    def test_lbs_per_mwh_to_gco2_per_kwh_constant(self):
        # gCO2/kWh = lbs/MWh * 453.592 / 1000
        assert _LBS_PER_MWH_TO_GCO2_PER_KWH == pytest.approx(453.592 / 1000.0)

    def test_800_lbs_per_mwh(self):
        # 800 lbs/MWh -> 800 * 0.453592 ≈ 362.87 gCO2/kWh
        assert 800.0 * _LBS_PER_MWH_TO_GCO2_PER_KWH == pytest.approx(362.8736)


# ---------------------------------------------------------------------------
# 11.3 Missing carbon data — no silent default
# ---------------------------------------------------------------------------

class TestMissingMoer:
    def _record(self, intervals, kind=CarbonDataKind.HISTORICAL):
        return build_workload_carbon_record(
            job_id="j1", scheduler_name="opt", baseline_name="opt", region="us-west",
            start_time_utc=T0, end_time_utc=T0,
            power_kw=100.0, utilization_fraction=1.0, pue=1.0,
            moer_intervals=intervals, interval_hours=1.0,
            carbon_data_source="watttime_co2_moer",
            carbon_signal_type=CarbonSignalType.MARGINAL_MOER,
            carbon_data_kind=kind,
        )

    def test_missing_interval_excluded_not_defaulted(self):
        # Two hours: one real (500), one missing. Emissions must reflect ONLY the
        # covered hour (100 kWh * 500 / 1000 = 50 kg), never a 400 fallback.
        rec = self._record([
            MoerInterval(500.0),
            MoerInterval(None),
        ])
        assert rec.emissions_kgco2 == pytest.approx(50.0)
        assert rec.carbon_missing_intervals == 1
        assert rec.carbon_data_coverage_pct == pytest.approx(50.0)
        assert rec.carbon_data_is_complete is False

    def test_no_400_appears_anywhere(self):
        rec = self._record([MoerInterval(None), MoerInterval(None)])
        assert rec.emissions_kgco2 == 0.0          # not 2 * 400-derived value
        assert rec.moer_gco2_per_kwh == 0.0
        assert rec.carbon_data_coverage_pct == 0.0

    def test_full_coverage_is_complete(self):
        rec = self._record([MoerInterval(300.0), MoerInterval(300.0)])
        assert rec.carbon_data_coverage_pct == 100.0
        assert rec.carbon_data_is_complete is True
        assert rec.carbon_data_is_real is True
        assert rec.moer_gco2_per_kwh == pytest.approx(300.0)

    def test_synthetic_is_never_real(self):
        rec = self._record([MoerInterval(300.0)], kind=CarbonDataKind.SYNTHETIC)
        assert rec.carbon_data_is_real is False
        assert rec.carbon_data_is_historical is False

    def test_forecast_flagged_as_forecast(self):
        rec = self._record([MoerInterval(300.0)], kind=CarbonDataKind.FORECAST)
        assert rec.carbon_data_is_forecast is True
        assert rec.carbon_data_is_historical is False

    def test_record_carries_calculation_version(self):
        rec = self._record([MoerInterval(300.0)])
        assert rec.carbon_calculation_version == CARBON_CALCULATION_VERSION
        d = rec.to_dict()
        assert d["carbon_signal_type"] == "marginal_moer"


class TestAggregate:
    def test_one_synthetic_taints_realness(self):
        from aurelius.carbon.accounting import WorkloadCarbonRecord

        def mk(is_real):
            return WorkloadCarbonRecord(
                job_id="j", scheduler_name="o", baseline_name="o", region="us-west",
                start_time_utc=T0, end_time_utc=T0, duration_hours=1.0,
                power_kw=100.0, utilization_fraction=1.0, pue=1.0, energy_kwh=100.0,
                moer_gco2_per_kwh=300.0, emissions_kgco2=30.0,
                carbon_data_source="x", carbon_signal_type=CarbonSignalType.MARGINAL_MOER,
                carbon_data_is_real=is_real, carbon_data_is_forecast=False,
                carbon_data_is_historical=True, carbon_data_coverage_pct=100.0,
                carbon_missing_intervals=0,
            )
        agg = aggregate_records([mk(True), mk(False)])
        assert agg["carbon_data_is_real"] is False
        assert agg["total_emissions_kgco2"] == pytest.approx(60.0)
