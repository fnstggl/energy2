"""Realized cost evaluator for backtesting.

Computes energy cost and carbon emissions for a schedule using *actual*
historical data – never forecast data. This is the ground-truth measurement
used to score each backtesting fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from aurelius.models import Job, ScheduleDecision

logger = logging.getLogger(__name__)


@dataclass
class RealizedMetrics:
    """Ground-truth cost/carbon metrics for a schedule.

    Carbon honesty: ``total_carbon_gco2`` is summed ONLY over hours that had real
    MOER data. Missing-MOER hours are counted in ``missing_carbon_hours`` and
    excluded (no silent 400 gCO2/kWh fallback), and ``carbon_data_coverage_pct``
    surfaces how complete the carbon picture is so reports can refuse to present
    a low-coverage number as a realized saving.
    """
    total_energy_cost_usd: float = 0.0
    total_carbon_gco2: float = 0.0
    jobs_evaluated: int = 0
    missing_price_hours: int = 0
    missing_carbon_hours: int = 0
    carbon_hours_total: int = 0          # job-hours where carbon was looked up
    carbon_hours_covered: int = 0        # of those, how many had real MOER

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

    @property
    def carbon_data_coverage_pct(self) -> float:
        """Fraction of job-hours that had real MOER data (100.0 if none needed)."""
        if self.carbon_hours_total == 0:
            return 0.0 if self.missing_carbon_hours else 100.0
        return 100.0 * self.carbon_hours_covered / self.carbon_hours_total

    @property
    def carbon_complete(self) -> bool:
        """True only when every evaluated job-hour had real MOER coverage."""
        return self.missing_carbon_hours == 0 and self.carbon_hours_total > 0

    def to_dict(self) -> dict:
        return {
            "total_energy_cost_usd": round(self.total_energy_cost_usd, 4),
            "total_carbon_gco2": round(self.total_carbon_gco2, 4),
            "jobs_evaluated": self.jobs_evaluated,
            "missing_price_hours": self.missing_price_hours,
            "missing_carbon_hours": self.missing_carbon_hours,
            "carbon_data_coverage_pct": round(self.carbon_data_coverage_pct, 3),
            "carbon_complete": self.carbon_complete,
        }


def evaluate_schedule(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    price_data: dict[str, dict],
    carbon_data: dict[str, dict],
    price_fallback: float = 50.0,
    carbon_fallback: Optional[float] = None,
    warn_on_missing: bool = True,
) -> RealizedMetrics:
    """Compute realized energy cost and carbon from actual data.

    Args:
        schedule:       List of scheduling decisions to evaluate.
        jobs:           Corresponding job definitions (for power_kw).
        price_data:     {region: {timestamp: price_per_mwh}} – actual values.
        carbon_data:    {region: {timestamp: gco2_per_kwh}} – actual values.
        price_fallback: Price ($/MWh) to use when actual data is absent.
        carbon_fallback:Carbon (gCO2/kWh) for missing hours. Defaults to None,
                        meaning missing MOER is NOT silently replaced — those
                        hours are counted in ``missing_carbon_hours`` and excluded
                        from ``total_carbon_gco2`` (no fabricated 400 gCO2/kWh).
                        Pass an explicit value only for clearly-labeled scenarios.

    Returns:
        RealizedMetrics with ground-truth cost and carbon. Inspect
        ``carbon_data_coverage_pct`` / ``carbon_complete`` before treating the
        carbon number as a realized saving.
    """
    job_by_id = {j.job_id: j for j in jobs}
    metrics = RealizedMetrics()
    _warned_missing: set[tuple[str, str]] = set()

    for decision in schedule:
        job = job_by_id.get(decision.job_id)
        if job is None:
            continue

        metrics.jobs_evaluated += 1

        # Iterate over segments (single synthetic segment for non-migrated decisions).
        # Each segment's [start_time, end_time) window already includes any
        # migration-overhead time at its start, so we just sum hourly prices at
        # that segment's region.
        for segment in decision.all_segments:
            power_kw = job.power_kw * segment.power_fraction
            seg_end = segment.end_time
            current = segment.start_time.replace(minute=0, second=0, microsecond=0)

            while current < seg_end:
                # Last hour may be partial
                next_hour = current + timedelta(hours=1)
                hour_fraction = min(1.0, (seg_end - current).total_seconds() / 3600.0)
                if hour_fraction <= 0:
                    break

                region_prices = price_data.get(segment.region, {})
                region_carbon = carbon_data.get(segment.region, {})

                price = region_prices.get(current)
                if price is None:
                    price = price_fallback
                    metrics.missing_price_hours += 1
                    if warn_on_missing:
                        key = (segment.region, current.strftime("%Y-%m-%dT%H"))
                        if key not in _warned_missing:
                            _warned_missing.add(key)
                            logger.warning(
                                f"No actual price for region={segment.region} at {current} "
                                f"— using fallback ${price_fallback:.2f}/MWh. "
                                "Results may be unreliable if many hours are missing."
                            )

                # Energy cost: price [$/MWh] * power [kW] / 1000 * hours
                energy_kwh = power_kw * hour_fraction
                metrics.total_energy_cost_usd += (price / 1000.0) * energy_kwh

                # Carbon: NO silent 400 fallback. Missing MOER is recorded and
                # excluded from the carbon total unless an explicit (labeled)
                # carbon_fallback was passed. Coverage tracks REAL data only.
                metrics.carbon_hours_total += 1
                carbon = region_carbon.get(current)
                if carbon is None:
                    metrics.missing_carbon_hours += 1
                    if carbon_fallback is not None:
                        metrics.total_carbon_gco2 += carbon_fallback * energy_kwh
                else:
                    metrics.carbon_hours_covered += 1
                    metrics.total_carbon_gco2 += carbon * energy_kwh

                current = next_hour

    return metrics
