"""Objective function for job scheduling optimization.

The objective combines:
1. Energy cost minimization
2. Carbon cost minimization (secondary)
3. Risk penalty for uncertainty

Objective = α * energy_cost + β * carbon_cost + γ * risk_penalty

Where:
- energy_cost = Σ(price × power × time) for all jobs
- carbon_cost = Σ(carbon_intensity × power × time) for all jobs
- risk_penalty = Σ(uncertainty_penalty) for scheduling during uncertain periods
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import logging

from ..models import Job, ScheduleDecision, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveComponents:
    """Breakdown of objective function components.

    Attributes:
        energy_cost: Total energy cost in dollars
        carbon_cost: Total carbon cost (weighted gCO2)
        risk_penalty: Total risk/uncertainty penalty
        total: Weighted sum of all components
        energy_kwh: Total energy consumed in kWh
        carbon_kg: Total carbon emissions in kg CO2
    """
    energy_cost: float
    carbon_cost: float
    risk_penalty: float
    total: float
    energy_kwh: float
    carbon_kg: float


class ObjectiveFunction:
    """Calculates the optimization objective for a schedule.

    The objective function is the core of the optimization problem.
    It quantifies how "good" a particular schedule is.

    Lower objective = better schedule.
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize the objective function.

        Args:
            config: Optimization configuration with weights
        """
        self.config = config or OptimizationConfig()

    def calculate(
        self,
        jobs: list[Job],
        schedule: list[ScheduleDecision],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> ObjectiveComponents:
        """Calculate the full objective for a schedule.

        Args:
            jobs: List of jobs being scheduled
            schedule: List of scheduling decisions
            price_data: {region: {timestamp: price_per_mwh}}
            carbon_data: {region: {timestamp: gco2_per_kwh}}
            risk_data: {region: {timestamp: risk_penalty}} (optional)

        Returns:
            ObjectiveComponents with breakdown
        """
        # Create job lookup
        job_by_id = {j.job_id: j for j in jobs}

        total_energy_cost = 0.0
        total_carbon_cost = 0.0
        total_risk_penalty = 0.0
        total_energy_kwh = 0.0
        total_carbon_kg = 0.0

        for decision in schedule:
            job = job_by_id.get(decision.job_id)
            if not job:
                logger.warning(f"Job {decision.job_id} not found")
                continue

            # Calculate energy and costs for each hour the job runs
            current_time = decision.start_time
            remaining_hours = decision.actual_runtime_hours
            effective_power = job.power_kw * decision.power_fraction

            while remaining_hours > 0:
                # How much of this hour does the job use?
                hour_fraction = min(1.0, remaining_hours)

                # Get price for this hour (floor to hour boundary)
                hour_key = current_time.replace(minute=0, second=0, microsecond=0)

                # Energy for this hour segment
                energy_kwh = effective_power * hour_fraction
                total_energy_kwh += energy_kwh

                # Price lookup
                region_prices = price_data.get(decision.region, {})
                price_per_mwh = region_prices.get(hour_key, 50.0)  # default
                energy_cost = (price_per_mwh / 1000) * energy_kwh
                total_energy_cost += energy_cost

                # Carbon lookup
                region_carbon = carbon_data.get(decision.region, {})
                gco2_per_kwh = region_carbon.get(hour_key, 400.0)  # default
                carbon_g = gco2_per_kwh * energy_kwh
                total_carbon_kg += carbon_g / 1000
                # Carbon cost is normalized (gCO2 * some factor)
                carbon_cost = carbon_g * 0.001  # scaling factor
                total_carbon_cost += carbon_cost

                # Risk penalty
                if risk_data:
                    region_risk = risk_data.get(decision.region, {})
                    risk_factor = region_risk.get(hour_key, 0.05)
                    risk_penalty = risk_factor * energy_kwh
                    total_risk_penalty += risk_penalty

                remaining_hours -= hour_fraction
                current_time += timedelta(hours=1)

        # Calculate weighted total
        total = (
            self.config.alpha * total_energy_cost +
            self.config.beta * total_carbon_cost +
            self.config.gamma * total_risk_penalty
        )

        return ObjectiveComponents(
            energy_cost=round(total_energy_cost, 2),
            carbon_cost=round(total_carbon_cost, 4),
            risk_penalty=round(total_risk_penalty, 4),
            total=round(total, 4),
            energy_kwh=round(total_energy_kwh, 2),
            carbon_kg=round(total_carbon_kg, 2),
        )

    def calculate_job_cost(
        self,
        job: Job,
        start_time: datetime,
        region: str,
        power_fraction: float,
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> ObjectiveComponents:
        """Calculate objective for a single job placement.

        Useful for evaluating individual scheduling options.

        Args:
            job: The job to evaluate
            start_time: Proposed start time
            region: Proposed region
            power_fraction: Power throttle level
            price_data: Price lookup
            carbon_data: Carbon lookup
            risk_data: Risk lookup

        Returns:
            ObjectiveComponents for this job placement
        """
        runtime = job.adjusted_runtime(power_fraction)
        decision = ScheduleDecision(
            job_id=job.job_id,
            start_time=start_time,
            region=region,
            power_fraction=power_fraction,
            actual_runtime_hours=runtime,
        )
        return self.calculate(
            [job],
            [decision],
            price_data,
            carbon_data,
            risk_data,
        )

    def compare_options(
        self,
        job: Job,
        options: list[tuple[datetime, str, float]],  # (start_time, region, power_fraction)
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> list[tuple[tuple[datetime, str, float], ObjectiveComponents]]:
        """Compare multiple scheduling options for a job.

        Args:
            job: The job to evaluate
            options: List of (start_time, region, power_fraction) tuples
            price_data: Price lookup
            carbon_data: Carbon lookup
            risk_data: Risk lookup

        Returns:
            List of (option, objective) tuples sorted by objective (best first)
        """
        results = []
        for start_time, region, power_fraction in options:
            obj = self.calculate_job_cost(
                job, start_time, region, power_fraction,
                price_data, carbon_data, risk_data
            )
            results.append(((start_time, region, power_fraction), obj))

        # Sort by total objective (lower is better)
        results.sort(key=lambda x: x[1].total)
        return results


def estimate_baseline_cost(
    jobs: list[Job],
    price_data: dict[str, dict[datetime, float]],
    carbon_data: dict[str, dict[datetime, float]],
    default_region: str = "us-west",
) -> ObjectiveComponents:
    """Estimate cost under baseline (ASAP) scheduling.

    Baseline policy:
    - Start jobs as soon as possible (earliest_start)
    - Run at full power (no throttling)
    - Use default/first region

    Args:
        jobs: List of jobs
        price_data: Price lookup
        carbon_data: Carbon lookup
        default_region: Default region to use

    Returns:
        ObjectiveComponents for baseline schedule
    """
    schedule = []
    for job in jobs:
        region = default_region if default_region in job.region_options else job.region_options[0]
        schedule.append(ScheduleDecision(
            job_id=job.job_id,
            start_time=job.earliest_start,
            region=region,
            power_fraction=1.0,
            actual_runtime_hours=job.runtime_hours,
        ))

    objective = ObjectiveFunction()
    return objective.calculate(jobs, schedule, price_data, carbon_data)
