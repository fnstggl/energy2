"""Realized cost evaluator for backtesting.

Computes energy cost and carbon emissions for a schedule using *actual*
historical data – never forecast data. This is the ground-truth measurement
used to score each backtesting fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from aurelius.models import Job, ScheduleDecision

logger = logging.getLogger(__name__)


@dataclass
class RealizedMetrics:
    """Ground-truth cost/carbon metrics for a schedule."""
    total_energy_cost_usd: float = 0.0
    total_carbon_gco2: float = 0.0
    jobs_evaluated: int = 0
    missing_price_hours: int = 0
    missing_carbon_hours: int = 0

    @property
    def avg_energy_cost_per_job(self) -> float:
        if self.jobs_evaluated == 0:
            return 0.0
        return self.total_energy_cost_usd / self.jobs_evaluated

    @property
    def data_coverage_pct(self) -> float:
        """Fraction of job-hours that had real (non-fallback) price data."""
        total = self.jobs_evaluated
        if total == 0:
            return 100.0
        missing = self.missing_price_hours
        # Approximate: jobs_evaluated ≈ total job-hours (imprecise but directional)
        return max(0.0, 100.0 * (1.0 - missing / max(1, total + missing)))

    def to_dict(self) -> dict:
        return {
            "total_energy_cost_usd": round(self.total_energy_cost_usd, 4),
            "total_carbon_gco2": round(self.total_carbon_gco2, 4),
            "jobs_evaluated": self.jobs_evaluated,
            "missing_price_hours": self.missing_price_hours,
            "missing_carbon_hours": self.missing_carbon_hours,
        }


def evaluate_schedule(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    price_data: dict[str, dict],
    carbon_data: dict[str, dict],
    price_fallback: float = 50.0,
    carbon_fallback: float = 400.0,
    warn_on_missing: bool = True,
) -> RealizedMetrics:
    """Compute realized energy cost and carbon from actual data.

    Args:
        schedule:       List of scheduling decisions to evaluate.
        jobs:           Corresponding job definitions (for power_kw).
        price_data:     {region: {timestamp: price_per_mwh}} – actual values.
        carbon_data:    {region: {timestamp: gco2_per_kwh}} – actual values.
        price_fallback: Price ($/MWh) to use when actual data is absent.
        carbon_fallback:Carbon (gCO2/kWh) to use when actual data is absent.

    Returns:
        RealizedMetrics with ground-truth cost and carbon.
    """
    job_by_id = {j.job_id: j for j in jobs}
    metrics = RealizedMetrics()
    _warned_missing: set[tuple[str, str]] = set()

    for decision in schedule:
        job = job_by_id.get(decision.job_id)
        if job is None:
            continue

        metrics.jobs_evaluated += 1
        power_kw = job.power_kw * decision.power_fraction

        # Walk hour by hour across the job's runtime
        current = decision.start_time.replace(minute=0, second=0, microsecond=0)
        remaining = decision.actual_runtime_hours

        while remaining > 0:
            hour_fraction = min(1.0, remaining)
            region_prices = price_data.get(decision.region, {})
            region_carbon = carbon_data.get(decision.region, {})

            price = region_prices.get(current)
            if price is None:
                price = price_fallback
                metrics.missing_price_hours += 1
                if warn_on_missing:
                    key = (decision.region, current.strftime("%Y-%m-%dT%H"))
                    if key not in _warned_missing:
                        _warned_missing.add(key)
                        logger.warning(
                            f"No actual price for region={decision.region} at {current} "
                            f"— using fallback ${price_fallback:.2f}/MWh. "
                            "Results may be unreliable if many hours are missing."
                        )

            carbon = region_carbon.get(current)
            if carbon is None:
                carbon = carbon_fallback
                metrics.missing_carbon_hours += 1

            # Energy cost: price [$/MWh] * power [kW] / 1000 * hours
            energy_kwh = power_kw * hour_fraction
            metrics.total_energy_cost_usd += (price / 1000.0) * energy_kwh

            # Carbon: gco2_per_kwh * kWh
            metrics.total_carbon_gco2 += carbon * energy_kwh

            remaining -= hour_fraction
            current += timedelta(hours=1)

    return metrics
