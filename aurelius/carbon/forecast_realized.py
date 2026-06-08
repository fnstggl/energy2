"""Forecast-vs-realized carbon tracking (Phase 9).

For every optimized workload we store BOTH the forecast MOER used to make the
decision and the realized historical MOER, plus the emissions and savings under
each, and the forecast error. Reports must distinguish:

* expected (forecast) carbon savings
* realized (historical) carbon savings
* simulated (synthetic) carbon savings
* unavailable carbon data

so a forecast number is never presented as a realized real-world saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _pct_error(forecast: float, realized: float) -> Optional[float]:
    if realized == 0:
        return None
    return 100.0 * (forecast - realized) / abs(realized)


@dataclass
class ForecastRealizedCarbon:
    """Per-workload forecast-vs-realized carbon record."""

    job_id: str
    region: str

    forecast_moer_used_for_decision: Optional[float]
    realized_historical_moer: Optional[float]

    forecast_emissions_kgco2: Optional[float]
    realized_emissions_kgco2: Optional[float]

    # Savings vs a named baseline under each signal.
    baseline_name: str = ""
    forecast_baseline_emissions_kgco2: Optional[float] = None
    realized_baseline_emissions_kgco2: Optional[float] = None

    @property
    def forecast_carbon_savings_kgco2(self) -> Optional[float]:
        if self.forecast_baseline_emissions_kgco2 is None or self.forecast_emissions_kgco2 is None:
            return None
        return self.forecast_baseline_emissions_kgco2 - self.forecast_emissions_kgco2

    @property
    def realized_carbon_savings_kgco2(self) -> Optional[float]:
        if self.realized_baseline_emissions_kgco2 is None or self.realized_emissions_kgco2 is None:
            return None
        return self.realized_baseline_emissions_kgco2 - self.realized_emissions_kgco2

    @property
    def forecast_error_pct(self) -> Optional[float]:
        """MOER forecast error: (forecast - realized)/|realized| * 100."""
        if self.forecast_moer_used_for_decision is None or self.realized_historical_moer is None:
            return None
        return _pct_error(self.forecast_moer_used_for_decision, self.realized_historical_moer)

    @property
    def emissions_forecast_error_pct(self) -> Optional[float]:
        if self.forecast_emissions_kgco2 is None or self.realized_emissions_kgco2 is None:
            return None
        return _pct_error(self.forecast_emissions_kgco2, self.realized_emissions_kgco2)

    @property
    def carbon_data_status(self) -> str:
        """One of expected / realized / unavailable for this record."""
        if self.realized_emissions_kgco2 is not None and self.realized_historical_moer is not None:
            return "realized"
        if self.forecast_emissions_kgco2 is not None:
            return "expected"
        return "unavailable"

    def to_dict(self) -> dict:
        def r(x):
            return None if x is None else round(x, 6)
        return {
            "job_id": self.job_id,
            "region": self.region,
            "baseline_name": self.baseline_name,
            "forecast_moer_used_for_decision": r(self.forecast_moer_used_for_decision),
            "realized_historical_moer": r(self.realized_historical_moer),
            "forecast_emissions_kgco2": r(self.forecast_emissions_kgco2),
            "realized_emissions_kgco2": r(self.realized_emissions_kgco2),
            "forecast_carbon_savings_kgco2": r(self.forecast_carbon_savings_kgco2),
            "realized_carbon_savings_kgco2": r(self.realized_carbon_savings_kgco2),
            "forecast_error_pct": r(self.forecast_error_pct),
            "emissions_forecast_error_pct": r(self.emissions_forecast_error_pct),
            "carbon_data_status": self.carbon_data_status,
        }


__all__ = ["ForecastRealizedCarbon"]
