"""Phase 9/11.9 — forecast and realized carbon savings stored separately."""

import pytest

from aurelius.carbon.forecast_realized import ForecastRealizedCarbon


class TestForecastVsRealizedSeparation:
    def test_stores_both_signals_separately(self):
        rec = ForecastRealizedCarbon(
            job_id="j1", region="us-west",
            forecast_moer_used_for_decision=120.0,
            realized_historical_moer=150.0,
            forecast_emissions_kgco2=24.0,
            realized_emissions_kgco2=30.0,
            baseline_name="current_price_only",
            forecast_baseline_emissions_kgco2=60.0,
            realized_baseline_emissions_kgco2=66.0,
        )
        # Forecast and realized savings are distinct numbers.
        assert rec.forecast_carbon_savings_kgco2 == pytest.approx(36.0)   # 60 - 24
        assert rec.realized_carbon_savings_kgco2 == pytest.approx(36.0)   # 66 - 30
        # They are NOT silently equal in general: emissions differ.
        assert rec.forecast_emissions_kgco2 != rec.realized_emissions_kgco2

    def test_forecast_error_pct(self):
        rec = ForecastRealizedCarbon(
            job_id="j1", region="us-west",
            forecast_moer_used_for_decision=120.0,
            realized_historical_moer=100.0,
            forecast_emissions_kgco2=24.0,
            realized_emissions_kgco2=20.0,
        )
        assert rec.forecast_error_pct == pytest.approx(20.0)            # (120-100)/100
        assert rec.emissions_forecast_error_pct == pytest.approx(20.0)  # (24-20)/20
        assert rec.carbon_data_status == "realized"

    def test_status_expected_when_no_realized(self):
        rec = ForecastRealizedCarbon(
            job_id="j1", region="us-west",
            forecast_moer_used_for_decision=120.0,
            realized_historical_moer=None,
            forecast_emissions_kgco2=24.0,
            realized_emissions_kgco2=None,
        )
        assert rec.carbon_data_status == "expected"
        assert rec.realized_carbon_savings_kgco2 is None
        assert rec.forecast_error_pct is None

    def test_status_unavailable(self):
        rec = ForecastRealizedCarbon(
            job_id="j1", region="us-west",
            forecast_moer_used_for_decision=None,
            realized_historical_moer=None,
            forecast_emissions_kgco2=None,
            realized_emissions_kgco2=None,
        )
        assert rec.carbon_data_status == "unavailable"

    def test_dict_serialization(self):
        rec = ForecastRealizedCarbon(
            job_id="j1", region="us-west",
            forecast_moer_used_for_decision=120.0,
            realized_historical_moer=150.0,
            forecast_emissions_kgco2=24.0,
            realized_emissions_kgco2=30.0,
        )
        d = rec.to_dict()
        assert d["forecast_emissions_kgco2"] == 24.0
        assert d["realized_emissions_kgco2"] == 30.0
        assert "forecast_error_pct" in d
