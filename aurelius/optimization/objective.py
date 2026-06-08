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

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..models import Job, OptimizationConfig, ScheduleDecision

logger = logging.getLogger(__name__)

# DECISION-TIME ONLY fallback marginal emissions rate, used when a forecast MOER
# is missing for a (region, hour) while RANKING candidates. This is a planning
# heuristic — it is NEVER used for realized/reported carbon savings (the realized
# evaluator in backtesting/evaluator.py records missing MOER and excludes it; see
# docs/CARBON_AUDIT.md). It only nudges the soft carbon objective term (weight
# beta) when carbon data is incomplete; the realized number is what's reported.
# Kept at the historical value so existing cost benchmarks are unaffected.
DECISION_FALLBACK_MOER_GCO2_PER_KWH = 400.0


def _lookup_last_known(series: dict[datetime, float], ts: datetime) -> float:
    """Return the last value at or before `ts` in `series`.

    Used for queue state lookups where the "last known" value before a
    scheduling timestamp is the correct leakage-safe choice.

    Returns 0.0 if no entry exists at or before `ts`.
    """
    if not series:
        return 0.0
    ts_floor = ts.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    best_key: Optional[datetime] = None
    best_val: float = 0.0
    for k, v in series.items():
        k_naive = k.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        if k_naive <= ts_floor:
            if best_key is None or k_naive >= best_key:
                best_key = k_naive
                best_val = v
    return best_val


@dataclass
class ObjectiveComponents:
    """Breakdown of objective function components.

    Attributes:
        energy_cost: Total energy cost in dollars (includes PUE overhead)
        carbon_cost: Total carbon cost (weighted gCO2)
        risk_penalty: Total risk/uncertainty penalty
        sla_penalty_cost: Total SLA violation penalty cost in dollars
        data_transfer_cost: Total inter-region data transfer cost in dollars
        queue_delay_cost: Total opportunity cost of GPU-hours lost to queue wait
        gpu_health_cost: Total penalty for placing jobs on degraded/hot/throttled GPUs
        total: Weighted sum of all components
        energy_kwh: Total energy consumed in kWh (before PUE)
        carbon_kg: Total carbon emissions in kg CO2
    """
    energy_cost: float
    carbon_cost: float
    risk_penalty: float
    sla_penalty_cost: float
    data_transfer_cost: float
    queue_delay_cost: float
    gpu_health_cost: float
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
        queue_data: Optional[dict[str, dict[datetime, float]]] = None,
        gpu_health_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> ObjectiveComponents:
        """Calculate the full objective for a schedule.

        Args:
            jobs: List of jobs being scheduled
            schedule: List of scheduling decisions
            price_data: {region: {timestamp: price_per_mwh}}
            carbon_data: {region: {timestamp: gco2_per_kwh}}
            risk_data: {region: {timestamp: risk_penalty}} (optional)
            queue_data: {region: {timestamp: est_wait_hours}} (optional).
                When provided and config.queue_delay_cost_per_gpu_hour > 0,
                the estimated GPU-hours lost to queue waiting are added to the
                objective so the optimizer routes away from congested regions.
            gpu_health_data: {region: {timestamp: avg_health_penalty 0..1}} (optional).
                When provided and config.gpu_health_cost_per_hour > 0, the
                average GPU health degradation (utilization, thermal, throttle,
                ECC errors) is added as a penalty, routing jobs to healthier nodes.

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
        total_sla_penalty_cost = 0.0
        total_data_transfer_cost = 0.0
        total_queue_delay_cost = 0.0
        total_gpu_health_cost = 0.0

        for decision in schedule:
            job = job_by_id.get(decision.job_id)
            if not job:
                logger.warning(f"Job {decision.job_id} not found")
                continue

            # PUE multiplier: accounts for facility power overhead
            # PUE >= 1.0; typical data-center PUE is 1.1–1.6
            pue = getattr(job, "pue", 1.0)

            # Calculate energy and costs for each hour the job runs
            current_time = decision.start_time
            remaining_hours = decision.actual_runtime_hours
            effective_power = job.power_kw * decision.power_fraction

            while remaining_hours > 0:
                hour_fraction = min(1.0, remaining_hours)
                hour_key = current_time.replace(minute=0, second=0, microsecond=0)

                # IT energy (kWh) before PUE
                it_energy_kwh = effective_power * hour_fraction
                total_energy_kwh += it_energy_kwh

                # Total facility energy including PUE overhead
                facility_energy_kwh = it_energy_kwh * pue

                # Price lookup
                region_prices = price_data.get(decision.region, {})
                price_per_mwh = region_prices.get(hour_key, 50.0)
                energy_cost = (price_per_mwh / 1000) * facility_energy_kwh
                total_energy_cost += energy_cost

                # Carbon lookup — use facility energy for carbon accounting.
                # Missing forecast MOER falls back to the named decision-time
                # constant (NOT used for realized reporting; see its docstring).
                region_carbon = carbon_data.get(decision.region, {})
                gco2_per_kwh = region_carbon.get(hour_key, DECISION_FALLBACK_MOER_GCO2_PER_KWH)
                carbon_g = gco2_per_kwh * facility_energy_kwh
                total_carbon_kg += carbon_g / 1000
                carbon_cost = carbon_g * 0.001
                total_carbon_cost += carbon_cost

                # Risk penalty
                if risk_data:
                    region_risk = risk_data.get(decision.region, {})
                    risk_factor = region_risk.get(hour_key, 0.05)
                    total_risk_penalty += risk_factor * it_energy_kwh

                remaining_hours -= hour_fraction
                current_time += timedelta(hours=1)

            # SLA penalty: charged per hour past job deadline
            sla_penalty_per_hour = getattr(job, "sla_penalty_per_hour", 0.0)
            if sla_penalty_per_hour > 0.0:
                job_end = decision.start_time + timedelta(hours=decision.actual_runtime_hours)
                if job_end > job.deadline:
                    overrun_hours = (job_end - job.deadline).total_seconds() / 3600
                    total_sla_penalty_cost += sla_penalty_per_hour * overrun_hours

            # Data transfer cost: flat per-job charge
            data_transfer_gb = getattr(job, "data_transfer_gb", 0.0)
            if data_transfer_gb > 0.0:
                transfer_cost = data_transfer_gb * self.config.data_transfer_cost_per_gb
                total_data_transfer_cost += transfer_cost

            # Queue delay cost: opportunity cost of GPU-hours lost to queue waiting.
            # Applies at the job start hour; uses last known state ≤ start_time.
            if queue_data is not None and self.config.queue_delay_cost_per_gpu_hour > 0.0:
                gpu_count = max(1, getattr(job, "gpu_count", 1))
                region_queue = queue_data.get(decision.region, {})
                wait_h = _lookup_last_known(region_queue, decision.start_time)
                queue_cost = wait_h * self.config.queue_delay_cost_per_gpu_hour * gpu_count
                total_queue_delay_cost += queue_cost

            # GPU health cost: penalty for running on degraded/hot/throttled GPUs.
            # health_penalty 0.0=healthy, 1.0=severely degraded.
            # Applies per-GPU-hour; uses last known state ≤ start_time.
            if gpu_health_data is not None and self.config.gpu_health_cost_per_hour > 0.0:
                gpu_count = max(1, getattr(job, "gpu_count", 1))
                region_health = gpu_health_data.get(decision.region, {})
                health_penalty = _lookup_last_known(region_health, decision.start_time)
                gpu_health_cost = (
                    health_penalty
                    * self.config.gpu_health_cost_per_hour
                    * decision.actual_runtime_hours
                    * gpu_count
                )
                total_gpu_health_cost += gpu_health_cost

        # Weighted total: alpha*energy + beta*carbon + gamma*risk + delta*SLA
        #                 + data_transfer + queue + gpu_health
        total = (
            self.config.alpha * total_energy_cost
            + self.config.beta * total_carbon_cost
            + self.config.gamma * total_risk_penalty
            + self.config.delta * total_sla_penalty_cost
            + total_data_transfer_cost
            + total_queue_delay_cost
            + total_gpu_health_cost
        )

        return ObjectiveComponents(
            energy_cost=round(total_energy_cost, 2),
            carbon_cost=round(total_carbon_cost, 4),
            risk_penalty=round(total_risk_penalty, 4),
            sla_penalty_cost=round(total_sla_penalty_cost, 2),
            data_transfer_cost=round(total_data_transfer_cost, 4),
            queue_delay_cost=round(total_queue_delay_cost, 4),
            gpu_health_cost=round(total_gpu_health_cost, 4),
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
        queue_data: Optional[dict[str, dict[datetime, float]]] = None,
        gpu_health_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> ObjectiveComponents:
        """Calculate objective for a single job placement."""
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
            queue_data,
            gpu_health_data,
        )

    def compare_options(
        self,
        job: Job,
        options: list[tuple[datetime, str, float]],  # (start_time, region, power_fraction)
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
        queue_data: Optional[dict[str, dict[datetime, float]]] = None,
        gpu_health_data: Optional[dict[str, dict[datetime, float]]] = None,
    ) -> list[tuple[tuple[datetime, str, float], ObjectiveComponents]]:
        """Compare multiple scheduling options for a job."""
        results = []
        for start_time, region, power_fraction in options:
            obj = self.calculate_job_cost(
                job, start_time, region, power_fraction,
                price_data, carbon_data, risk_data, queue_data, gpu_health_data,
            )
            results.append(((start_time, region, power_fraction), obj))
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
    return objective.calculate(jobs, schedule, price_data, carbon_data, queue_data=None)
