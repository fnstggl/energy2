"""Metrics calculation for simulation results.

This module computes key performance indicators:
- Energy cost metrics
- Compute cost metrics (estimated from power/vCPU inference)
- Carbon emissions metrics
- Savings metrics (absolute, percentage)
- Efficiency metrics
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import logging

from ..models import Job, ScheduleDecision, SimulationResult

logger = logging.getLogger(__name__)


# Regional compute pricing estimates ($/vCPU-hour)
# Based on typical cloud pricing - directional estimates for pilots
REGIONAL_VCPU_PRICES = {
    "us-west": 0.042,
    "us-east": 0.040,
    "eu-west": 0.048,
    "eu-north": 0.045,
    "asia-east": 0.038,
}

# Power to vCPU ratio (kW per vCPU) - rough estimate
# Assumes ~200W per server with 32 vCPUs = 0.00625 kW/vCPU
KW_PER_VCPU = 0.00625


def estimate_vcpus_from_power(power_kw: float) -> float:
    """Estimate vCPU count from power consumption.

    This is a rough estimate for directional savings calculations.
    Pilots understand this is approximate.

    Args:
        power_kw: Power consumption in kW

    Returns:
        Estimated vCPU count
    """
    return power_kw / KW_PER_VCPU


@dataclass
class ScheduleMetrics:
    """Comprehensive metrics for a schedule.

    Attributes:
        total_energy_cost: Total energy cost in dollars
        total_compute_cost: Estimated compute cost in dollars
        total_carbon_kg: Total carbon emissions in kg CO2
        total_energy_kwh: Total energy consumed in kWh
        total_vcpu_hours: Total vCPU-hours consumed
        avg_job_delay_hours: Average delay from earliest start
        avg_power_utilization: Average power fraction used
        jobs_throttled: Number of jobs running at reduced power
        jobs_shifted: Number of jobs not starting at earliest time
        region_distribution: Jobs per region
        peak_power_kw: Maximum concurrent power usage
        makespan_hours: Time from first start to last finish
    """
    total_energy_cost: float
    total_compute_cost: float
    total_carbon_kg: float
    total_energy_kwh: float
    total_vcpu_hours: float
    avg_job_delay_hours: float
    avg_power_utilization: float
    jobs_throttled: int
    jobs_shifted: int
    region_distribution: dict[str, int]
    peak_power_kw: float
    makespan_hours: float


@dataclass
class DualBaselineComparison:
    """Comparison results against both baselines.

    Attributes:
        fifo_metrics: Metrics for FIFO baseline
        peak_blind_metrics: Metrics for peak-blind ASAP baseline
        optimized_metrics: Metrics for optimized schedule
        savings_vs_fifo: Savings compared to FIFO
        savings_vs_peak_blind: Savings compared to peak-blind
    """
    fifo_metrics: ScheduleMetrics
    peak_blind_metrics: ScheduleMetrics
    optimized_metrics: ScheduleMetrics
    savings_vs_fifo: dict
    savings_vs_peak_blind: dict


class MetricsCalculator:
    """Calculates metrics for schedule evaluation."""

    def __init__(self):
        pass

    def calculate_schedule_metrics(
        self,
        jobs: list[Job],
        schedule: list[ScheduleDecision],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
    ) -> ScheduleMetrics:
        """Calculate comprehensive metrics for a schedule.

        Args:
            jobs: List of jobs
            schedule: List of scheduling decisions
            price_data: {region: {timestamp: price_per_mwh}}
            carbon_data: {region: {timestamp: gco2_per_kwh}}

        Returns:
            ScheduleMetrics object
        """
        job_by_id = {j.job_id: j for j in jobs}

        total_energy_cost = 0.0
        total_compute_cost = 0.0
        total_carbon_kg = 0.0
        total_energy_kwh = 0.0
        total_vcpu_hours = 0.0
        total_delay_hours = 0.0
        total_power_fraction = 0.0
        jobs_throttled = 0
        jobs_shifted = 0
        region_counts: dict[str, int] = {}

        # Track power by hour for peak calculation
        power_by_hour: dict[datetime, float] = {}

        for decision in schedule:
            job = job_by_id.get(decision.job_id)
            if not job:
                continue

            # Region distribution
            region_counts[decision.region] = region_counts.get(decision.region, 0) + 1

            # Delay from earliest start
            delay = (decision.start_time - job.earliest_start).total_seconds() / 3600
            total_delay_hours += max(0, delay)
            if delay > 0.5:  # More than 30 min delay counts as shifted
                jobs_shifted += 1

            # Power utilization
            total_power_fraction += decision.power_fraction
            if decision.power_fraction < 0.99:
                jobs_throttled += 1

            # Calculate energy and compute costs per hour
            effective_power = job.power_kw * decision.power_fraction
            vcpus = estimate_vcpus_from_power(effective_power)
            vcpu_price = REGIONAL_VCPU_PRICES.get(decision.region, 0.042)

            current = decision.start_time
            remaining = decision.actual_runtime_hours

            while remaining > 0:
                hour_key = current.replace(minute=0, second=0, microsecond=0)
                hour_fraction = min(1.0, remaining)
                energy_kwh = effective_power * hour_fraction

                total_energy_kwh += energy_kwh
                total_vcpu_hours += vcpus * hour_fraction

                # Compute cost (vCPU-hours * regional price)
                total_compute_cost += vcpus * hour_fraction * vcpu_price

                # Track peak power
                power_by_hour[hour_key] = power_by_hour.get(hour_key, 0) + effective_power

                # Energy cost
                price = price_data.get(decision.region, {}).get(hour_key, 50.0)
                total_energy_cost += (price / 1000) * energy_kwh

                # Carbon
                carbon = carbon_data.get(decision.region, {}).get(hour_key, 400.0)
                total_carbon_kg += (carbon / 1000) * energy_kwh

                remaining -= hour_fraction
                current += timedelta(hours=1)

        # Calculate aggregates
        num_jobs = len(schedule)
        avg_delay = total_delay_hours / num_jobs if num_jobs > 0 else 0
        avg_power = total_power_fraction / num_jobs if num_jobs > 0 else 1.0
        peak_power = max(power_by_hour.values()) if power_by_hour else 0

        # Makespan
        if schedule:
            first_start = min(d.start_time for d in schedule)
            last_end = max(d.end_time for d in schedule)
            makespan = (last_end - first_start).total_seconds() / 3600
        else:
            makespan = 0

        return ScheduleMetrics(
            total_energy_cost=round(total_energy_cost, 2),
            total_compute_cost=round(total_compute_cost, 2),
            total_carbon_kg=round(total_carbon_kg, 2),
            total_energy_kwh=round(total_energy_kwh, 2),
            total_vcpu_hours=round(total_vcpu_hours, 2),
            avg_job_delay_hours=round(avg_delay, 2),
            avg_power_utilization=round(avg_power, 3),
            jobs_throttled=jobs_throttled,
            jobs_shifted=jobs_shifted,
            region_distribution=region_counts,
            peak_power_kw=round(peak_power, 1),
            makespan_hours=round(makespan, 2),
        )

    def calculate_savings(
        self,
        baseline_metrics: ScheduleMetrics,
        optimized_metrics: ScheduleMetrics,
    ) -> dict:
        """Calculate savings between baseline and optimized schedules.

        Args:
            baseline_metrics: Metrics for baseline schedule
            optimized_metrics: Metrics for optimized schedule

        Returns:
            Dictionary of savings metrics
        """
        # Energy cost savings
        energy_cost_delta = baseline_metrics.total_energy_cost - optimized_metrics.total_energy_cost
        energy_cost_pct = (energy_cost_delta / baseline_metrics.total_energy_cost * 100
                         if baseline_metrics.total_energy_cost > 0 else 0)

        # Compute cost savings
        compute_cost_delta = baseline_metrics.total_compute_cost - optimized_metrics.total_compute_cost
        compute_cost_pct = (compute_cost_delta / baseline_metrics.total_compute_cost * 100
                          if baseline_metrics.total_compute_cost > 0 else 0)

        # Carbon savings
        carbon_delta = baseline_metrics.total_carbon_kg - optimized_metrics.total_carbon_kg
        carbon_pct = (carbon_delta / baseline_metrics.total_carbon_kg * 100
                     if baseline_metrics.total_carbon_kg > 0 else 0)

        # Peak power reduction
        peak_delta = baseline_metrics.peak_power_kw - optimized_metrics.peak_power_kw
        peak_pct = (peak_delta / baseline_metrics.peak_power_kw * 100
                   if baseline_metrics.peak_power_kw > 0 else 0)

        return {
            "energy_cost_savings_dollars": round(energy_cost_delta, 2),
            "energy_cost_savings_pct": round(energy_cost_pct, 2),
            "compute_cost_savings_dollars": round(compute_cost_delta, 2),
            "compute_cost_savings_pct": round(compute_cost_pct, 2),
            "carbon_savings_kg": round(carbon_delta, 2),
            "carbon_savings_pct": round(carbon_pct, 2),
            "peak_power_reduction_kw": round(peak_delta, 1),
            "peak_power_reduction_pct": round(peak_pct, 2),
            "jobs_throttled": optimized_metrics.jobs_throttled,
            "jobs_shifted": optimized_metrics.jobs_shifted,
            "makespan_change_hours": round(
                optimized_metrics.makespan_hours - baseline_metrics.makespan_hours, 2
            ),
        }

    def format_dual_baseline_report(
        self,
        fifo_metrics: ScheduleMetrics,
        peak_blind_metrics: ScheduleMetrics,
        optimized_metrics: ScheduleMetrics,
        savings_vs_fifo: dict,
        savings_vs_peak_blind: dict,
    ) -> str:
        """Format a human-readable report with dual baseline comparison.

        Args:
            fifo_metrics: FIFO baseline metrics
            peak_blind_metrics: Peak-blind ASAP baseline metrics
            optimized_metrics: Optimized schedule metrics
            savings_vs_fifo: Savings compared to FIFO
            savings_vs_peak_blind: Savings compared to peak-blind

        Returns:
            Formatted string report
        """
        lines = [
            "=" * 70,
            "AURELIUS SIMULATION RESULTS",
            "=" * 70,
            "",
            "─" * 70,
            "BASELINE SCENARIOS",
            "─" * 70,
            "",
            "FIFO BASELINE (jobs run in submission order, no optimization):",
            f"  Energy Cost:      ${fifo_metrics.total_energy_cost:>12,.2f}",
            f"  Compute Cost:     ${fifo_metrics.total_compute_cost:>12,.2f}",
            f"  Carbon:           {fifo_metrics.total_carbon_kg:>13,.2f} kg CO2",
            f"  Peak Power:       {fifo_metrics.peak_power_kw:>13,.1f} kW",
            "",
            "PEAK-BLIND ASAP BASELINE (jobs run immediately, even during peaks):",
            f"  Energy Cost:      ${peak_blind_metrics.total_energy_cost:>12,.2f}",
            f"  Compute Cost:     ${peak_blind_metrics.total_compute_cost:>12,.2f}",
            f"  Carbon:           {peak_blind_metrics.total_carbon_kg:>13,.2f} kg CO2",
            f"  Peak Power:       {peak_blind_metrics.peak_power_kw:>13,.1f} kW",
            "",
            "─" * 70,
            "OPTIMIZED SCHEDULE",
            "─" * 70,
            "",
            f"  Energy Cost:      ${optimized_metrics.total_energy_cost:>12,.2f}",
            f"  Compute Cost:     ${optimized_metrics.total_compute_cost:>12,.2f}",
            f"  Carbon:           {optimized_metrics.total_carbon_kg:>13,.2f} kg CO2",
            f"  Peak Power:       {optimized_metrics.peak_power_kw:>13,.1f} kW",
            f"  Jobs Throttled:   {optimized_metrics.jobs_throttled:>13}",
            f"  Jobs Shifted:     {optimized_metrics.jobs_shifted:>13}",
            "",
            "─" * 70,
            "SAVINGS VS FIFO BASELINE",
            "─" * 70,
            f"  Energy Cost:      ${savings_vs_fifo['energy_cost_savings_dollars']:>12,.2f}  ({savings_vs_fifo['energy_cost_savings_pct']:>6.1f}%)",
            f"  Compute Cost:     ${savings_vs_fifo['compute_cost_savings_dollars']:>12,.2f}  ({savings_vs_fifo['compute_cost_savings_pct']:>6.1f}%)",
            f"  Carbon:           {savings_vs_fifo['carbon_savings_kg']:>13,.2f} kg ({savings_vs_fifo['carbon_savings_pct']:>6.1f}%)",
            "",
            "─" * 70,
            "SAVINGS VS PEAK-BLIND BASELINE",
            "─" * 70,
            f"  Energy Cost:      ${savings_vs_peak_blind['energy_cost_savings_dollars']:>12,.2f}  ({savings_vs_peak_blind['energy_cost_savings_pct']:>6.1f}%)",
            f"  Compute Cost:     ${savings_vs_peak_blind['compute_cost_savings_dollars']:>12,.2f}  ({savings_vs_peak_blind['compute_cost_savings_pct']:>6.1f}%)",
            f"  Carbon:           {savings_vs_peak_blind['carbon_savings_kg']:>13,.2f} kg ({savings_vs_peak_blind['carbon_savings_pct']:>6.1f}%)",
            "",
            "─" * 70,
            "REGION DISTRIBUTION (Optimized)",
            "─" * 70,
        ]

        for region, count in sorted(optimized_metrics.region_distribution.items()):
            lines.append(f"  {region}: {count} jobs")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def format_metrics_report(
        self,
        baseline_metrics: ScheduleMetrics,
        optimized_metrics: ScheduleMetrics,
        savings: dict,
    ) -> str:
        """Format a human-readable metrics report (legacy single baseline).

        Args:
            baseline_metrics: Baseline schedule metrics
            optimized_metrics: Optimized schedule metrics
            savings: Savings dictionary

        Returns:
            Formatted string report
        """
        lines = [
            "=" * 60,
            "AURELIUS SIMULATION RESULTS",
            "=" * 60,
            "",
            "BASELINE SCHEDULE:",
            f"  Energy Cost:    ${baseline_metrics.total_energy_cost:,.2f}",
            f"  Compute Cost:   ${baseline_metrics.total_compute_cost:,.2f}",
            f"  Carbon (kg):    {baseline_metrics.total_carbon_kg:,.2f}",
            f"  Energy (kWh):   {baseline_metrics.total_energy_kwh:,.2f}",
            f"  Peak Power:     {baseline_metrics.peak_power_kw:,.1f} kW",
            f"  Makespan:       {baseline_metrics.makespan_hours:.1f} hours",
            "",
            "OPTIMIZED SCHEDULE:",
            f"  Energy Cost:    ${optimized_metrics.total_energy_cost:,.2f}",
            f"  Compute Cost:   ${optimized_metrics.total_compute_cost:,.2f}",
            f"  Carbon (kg):    {optimized_metrics.total_carbon_kg:,.2f}",
            f"  Energy (kWh):   {optimized_metrics.total_energy_kwh:,.2f}",
            f"  Peak Power:     {optimized_metrics.peak_power_kw:,.1f} kW",
            f"  Makespan:       {optimized_metrics.makespan_hours:.1f} hours",
            f"  Jobs Throttled: {optimized_metrics.jobs_throttled}",
            f"  Jobs Shifted:   {optimized_metrics.jobs_shifted}",
            "",
            "SAVINGS:",
            f"  Energy Cost:    ${savings['energy_cost_savings_dollars']:,.2f} ({savings['energy_cost_savings_pct']:.1f}%)",
            f"  Compute Cost:   ${savings['compute_cost_savings_dollars']:,.2f} ({savings['compute_cost_savings_pct']:.1f}%)",
            f"  Carbon Saved:   {savings['carbon_savings_kg']:.2f} kg ({savings['carbon_savings_pct']:.1f}%)",
            f"  Peak Reduced:   {savings['peak_power_reduction_kw']:.1f} kW ({savings['peak_power_reduction_pct']:.1f}%)",
            "",
            "REGION DISTRIBUTION (Optimized):",
        ]

        for region, count in optimized_metrics.region_distribution.items():
            lines.append(f"  {region}: {count} jobs")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_summary_dict(
        self,
        baseline_metrics: ScheduleMetrics,
        optimized_metrics: ScheduleMetrics,
        savings: dict,
    ) -> dict:
        """Convert metrics to summary dictionary for API/JSON output.

        Args:
            baseline_metrics: Baseline schedule metrics
            optimized_metrics: Optimized schedule metrics
            savings: Savings dictionary

        Returns:
            Summary dictionary
        """
        return {
            "baseline": {
                "energy_cost": baseline_metrics.total_energy_cost,
                "compute_cost": baseline_metrics.total_compute_cost,
                "carbon_kg": baseline_metrics.total_carbon_kg,
                "energy_kwh": baseline_metrics.total_energy_kwh,
                "vcpu_hours": baseline_metrics.total_vcpu_hours,
                "peak_power_kw": baseline_metrics.peak_power_kw,
                "makespan_hours": baseline_metrics.makespan_hours,
            },
            "optimized": {
                "energy_cost": optimized_metrics.total_energy_cost,
                "compute_cost": optimized_metrics.total_compute_cost,
                "carbon_kg": optimized_metrics.total_carbon_kg,
                "energy_kwh": optimized_metrics.total_energy_kwh,
                "vcpu_hours": optimized_metrics.total_vcpu_hours,
                "peak_power_kw": optimized_metrics.peak_power_kw,
                "makespan_hours": optimized_metrics.makespan_hours,
                "jobs_throttled": optimized_metrics.jobs_throttled,
                "jobs_shifted": optimized_metrics.jobs_shifted,
                "region_distribution": optimized_metrics.region_distribution,
            },
            "savings": savings,
        }

    def to_dual_baseline_dict(
        self,
        fifo_metrics: ScheduleMetrics,
        peak_blind_metrics: ScheduleMetrics,
        optimized_metrics: ScheduleMetrics,
        savings_vs_fifo: dict,
        savings_vs_peak_blind: dict,
    ) -> dict:
        """Convert dual baseline comparison to dictionary.

        Args:
            fifo_metrics: FIFO baseline metrics
            peak_blind_metrics: Peak-blind baseline metrics
            optimized_metrics: Optimized metrics
            savings_vs_fifo: Savings vs FIFO
            savings_vs_peak_blind: Savings vs peak-blind

        Returns:
            Summary dictionary
        """
        return {
            "baselines": {
                "fifo": {
                    "energy_cost": fifo_metrics.total_energy_cost,
                    "compute_cost": fifo_metrics.total_compute_cost,
                    "carbon_kg": fifo_metrics.total_carbon_kg,
                    "vcpu_hours": fifo_metrics.total_vcpu_hours,
                    "peak_power_kw": fifo_metrics.peak_power_kw,
                },
                "peak_blind": {
                    "energy_cost": peak_blind_metrics.total_energy_cost,
                    "compute_cost": peak_blind_metrics.total_compute_cost,
                    "carbon_kg": peak_blind_metrics.total_carbon_kg,
                    "vcpu_hours": peak_blind_metrics.total_vcpu_hours,
                    "peak_power_kw": peak_blind_metrics.peak_power_kw,
                },
            },
            "optimized": {
                "energy_cost": optimized_metrics.total_energy_cost,
                "compute_cost": optimized_metrics.total_compute_cost,
                "carbon_kg": optimized_metrics.total_carbon_kg,
                "vcpu_hours": optimized_metrics.total_vcpu_hours,
                "peak_power_kw": optimized_metrics.peak_power_kw,
                "jobs_throttled": optimized_metrics.jobs_throttled,
                "jobs_shifted": optimized_metrics.jobs_shifted,
                "region_distribution": optimized_metrics.region_distribution,
            },
            "savings_vs_fifo": savings_vs_fifo,
            "savings_vs_peak_blind": savings_vs_peak_blind,
        }
