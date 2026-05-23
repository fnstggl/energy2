"""Scenario comparison for Aurelius simulations.

This module compares baseline vs optimized scheduling outcomes:
- Dual baseline comparison (FIFO and Peak-blind ASAP)
- Cost comparison (energy and compute)
- Carbon comparison
- Timeline visualization data
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..models import Job, OptimizationConfig, ScheduleDecision, SimulationResult
from ..optimization.scheduler import JobScheduler
from .metrics import MetricsCalculator, ScheduleMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing baseline vs optimized scenarios.

    Attributes:
        fifo_schedule: FIFO baseline scheduling decisions
        peak_blind_schedule: Peak-blind ASAP baseline decisions
        optimized_schedule: Optimized scheduling decisions
        fifo_metrics: Metrics for FIFO baseline
        peak_blind_metrics: Metrics for peak-blind baseline
        optimized_metrics: Metrics for optimized
        savings_vs_fifo: Savings compared to FIFO
        savings_vs_peak_blind: Savings compared to peak-blind
        timeline_data: Data for timeline visualization
    """
    fifo_schedule: list[ScheduleDecision]
    peak_blind_schedule: list[ScheduleDecision]
    optimized_schedule: list[ScheduleDecision]
    fifo_metrics: ScheduleMetrics
    peak_blind_metrics: ScheduleMetrics
    optimized_metrics: ScheduleMetrics
    savings_vs_fifo: dict
    savings_vs_peak_blind: dict
    timeline_data: Optional[dict] = None


class ScenarioComparator:
    """Compares scheduling scenarios and generates comparison reports."""

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize the comparator.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.scheduler = JobScheduler(config)
        self.metrics = MetricsCalculator()

    def create_fifo_schedule(
        self,
        jobs: list[Job],
    ) -> list[ScheduleDecision]:
        """Create FIFO baseline schedule.

        FIFO (First-In-First-Out):
        - Jobs execute strictly in submission order
        - No price awareness
        - Jobs start as soon as previous jobs complete (respecting earliest_start)
        - Uses default/first region
        - Full power, no throttling

        Args:
            jobs: List of jobs

        Returns:
            List of scheduling decisions
        """
        # Sort by submission time
        sorted_jobs = sorted(jobs, key=lambda j: j.submit_time)

        schedule = []
        # Track when each region becomes available
        region_available: dict[str, datetime] = {}

        for job in sorted_jobs:
            region = (
                self.config.default_region
                if self.config.default_region in job.region_options
                else job.region_options[0]
            )

            # Start time is max of: earliest_start, when region is available
            region_free = region_available.get(region, job.earliest_start)
            start_time = max(job.earliest_start, region_free)

            # Update region availability
            end_time = start_time + timedelta(hours=job.runtime_hours)
            region_available[region] = end_time

            schedule.append(ScheduleDecision(
                job_id=job.job_id,
                start_time=start_time,
                region=region,
                power_fraction=1.0,
                actual_runtime_hours=job.runtime_hours,
            ))

        return schedule

    def create_peak_blind_schedule(
        self,
        jobs: list[Job],
    ) -> list[ScheduleDecision]:
        """Create peak-blind ASAP baseline schedule.

        Peak-Blind ASAP:
        - Jobs start immediately at earliest_start
        - No price awareness (runs even during peak pricing)
        - Uses default/first region
        - Full power, no throttling

        This is the same as the previous baseline but explicitly named.

        Args:
            jobs: List of jobs

        Returns:
            List of scheduling decisions
        """
        schedule = []
        for job in jobs:
            region = (
                self.config.default_region
                if self.config.default_region in job.region_options
                else job.region_options[0]
            )
            schedule.append(ScheduleDecision(
                job_id=job.job_id,
                start_time=job.earliest_start,
                region=region,
                power_fraction=1.0,
                actual_runtime_hours=job.runtime_hours,
            ))
        return schedule

    def compare(
        self,
        jobs: list[Job],
        price_data: dict[str, dict[datetime, float]],
        carbon_data: dict[str, dict[datetime, float]],
        risk_data: Optional[dict[str, dict[datetime, float]]] = None,
        optimization_method: str = "greedy",
    ) -> ComparisonResult:
        """Compare dual baselines vs optimized scheduling.

        Baselines:
        1. FIFO: Jobs run in submission order, queued
        2. Peak-Blind ASAP: Jobs run immediately at earliest_start

        Optimized:
        - Time shifting within slack
        - Power throttling
        - Multi-region routing
        - Carbon and risk-aware scheduling

        Args:
            jobs: List of jobs to schedule
            price_data: {region: {timestamp: price_per_mwh}}
            carbon_data: {region: {timestamp: gco2_per_kwh}}
            risk_data: {region: {timestamp: risk_penalty}}
            optimization_method: Method for optimizer ("greedy", "local_search", "milp")

        Returns:
            ComparisonResult with full comparison
        """
        # Generate both baseline schedules
        fifo_schedule = self.create_fifo_schedule(jobs)
        peak_blind_schedule = self.create_peak_blind_schedule(jobs)

        # Generate optimized schedule
        opt_result = self.scheduler.solve(
            jobs, price_data, carbon_data, risk_data,
            method=optimization_method,
        )
        optimized_schedule = opt_result.schedule

        # Calculate metrics for all three
        fifo_metrics = self.metrics.calculate_schedule_metrics(
            jobs, fifo_schedule, price_data, carbon_data
        )
        peak_blind_metrics = self.metrics.calculate_schedule_metrics(
            jobs, peak_blind_schedule, price_data, carbon_data
        )
        optimized_metrics = self.metrics.calculate_schedule_metrics(
            jobs, optimized_schedule, price_data, carbon_data
        )

        # Calculate savings vs both baselines
        savings_vs_fifo = self.metrics.calculate_savings(fifo_metrics, optimized_metrics)
        savings_vs_peak_blind = self.metrics.calculate_savings(peak_blind_metrics, optimized_metrics)

        # Generate timeline data for visualization
        timeline_data = self._generate_timeline_data(
            jobs, peak_blind_schedule, optimized_schedule, price_data
        )

        return ComparisonResult(
            fifo_schedule=fifo_schedule,
            peak_blind_schedule=peak_blind_schedule,
            optimized_schedule=optimized_schedule,
            fifo_metrics=fifo_metrics,
            peak_blind_metrics=peak_blind_metrics,
            optimized_metrics=optimized_metrics,
            savings_vs_fifo=savings_vs_fifo,
            savings_vs_peak_blind=savings_vs_peak_blind,
            timeline_data=timeline_data,
        )

    def _generate_timeline_data(
        self,
        jobs: list[Job],
        baseline_schedule: list[ScheduleDecision],
        optimized_schedule: list[ScheduleDecision],
        price_data: dict[str, dict[datetime, float]],
    ) -> dict:
        """Generate data for timeline visualization.

        Returns data structure suitable for plotting job schedules
        and price overlays.

        Args:
            jobs: List of jobs
            baseline_schedule: Baseline decisions
            optimized_schedule: Optimized decisions
            price_data: Price data

        Returns:
            Dictionary with timeline data
        """
        job_by_id = {j.job_id: j for j in jobs}

        # Find time range
        all_times = []
        for s in baseline_schedule + optimized_schedule:
            all_times.append(s.start_time)
            all_times.append(s.end_time)

        if not all_times:
            return {}

        start = min(all_times).replace(minute=0, second=0, microsecond=0)
        end = max(all_times).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # Generate hourly timestamps
        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current.isoformat())
            current += timedelta(hours=1)

        # Get average prices per hour
        prices = []
        current = start
        while current <= end:
            hour_prices = []
            for region_prices in price_data.values():
                if current in region_prices:
                    hour_prices.append(region_prices[current])
            prices.append(sum(hour_prices) / len(hour_prices) if hour_prices else 50.0)
            current += timedelta(hours=1)

        # Format job timelines
        baseline_jobs = []
        for d in baseline_schedule:
            job = job_by_id.get(d.job_id)
            baseline_jobs.append({
                "job_id": d.job_id,
                "start": d.start_time.isoformat(),
                "end": d.end_time.isoformat(),
                "region": d.region,
                "power_kw": job.power_kw if job else 0,
            })

        optimized_jobs = []
        for d in optimized_schedule:
            job = job_by_id.get(d.job_id)
            optimized_jobs.append({
                "job_id": d.job_id,
                "start": d.start_time.isoformat(),
                "end": d.end_time.isoformat(),
                "region": d.region,
                "power_kw": (job.power_kw * d.power_fraction) if job else 0,
                "power_fraction": d.power_fraction,
            })

        return {
            "timestamps": timestamps,
            "prices": prices,
            "baseline_jobs": baseline_jobs,
            "optimized_jobs": optimized_jobs,
        }

    def to_simulation_result(
        self,
        comparison: ComparisonResult,
        run_id: Optional[str] = None,
    ) -> SimulationResult:
        """Convert comparison to SimulationResult for storage.

        Uses peak-blind as the primary baseline for backward compatibility.

        Args:
            comparison: ComparisonResult to convert
            run_id: Optional run ID (auto-generated if not provided)

        Returns:
            SimulationResult object
        """
        result = SimulationResult(
            baseline_cost=comparison.peak_blind_metrics.total_energy_cost,
            optimized_cost=comparison.optimized_metrics.total_energy_cost,
            baseline_carbon=comparison.peak_blind_metrics.total_carbon_kg,
            optimized_carbon=comparison.optimized_metrics.total_carbon_kg,
            config=self.config.to_dict(),
            baseline_schedule=comparison.peak_blind_schedule,
            optimized_schedule=comparison.optimized_schedule,
        )
        if run_id:
            result.run_id = run_id
        return result

    def generate_report(
        self,
        comparison: ComparisonResult,
    ) -> str:
        """Generate human-readable comparison report with dual baselines.

        Args:
            comparison: ComparisonResult to report

        Returns:
            Formatted string report
        """
        return self.metrics.format_dual_baseline_report(
            comparison.fifo_metrics,
            comparison.peak_blind_metrics,
            comparison.optimized_metrics,
            comparison.savings_vs_fifo,
            comparison.savings_vs_peak_blind,
        )

    def generate_json_summary(
        self,
        comparison: ComparisonResult,
        run_id: Optional[str] = None,
    ) -> dict:
        """Generate JSON-serializable summary with dual baselines.

        Args:
            comparison: ComparisonResult to summarize
            run_id: Optional run ID

        Returns:
            Dictionary suitable for JSON serialization
        """
        sim_result = self.to_simulation_result(comparison, run_id)

        return {
            "run_id": sim_result.run_id,
            "created_at": sim_result.created_at.isoformat(),
            "summary": sim_result.to_summary_dict(),
            "metrics": self.metrics.to_dual_baseline_dict(
                comparison.fifo_metrics,
                comparison.peak_blind_metrics,
                comparison.optimized_metrics,
                comparison.savings_vs_fifo,
                comparison.savings_vs_peak_blind,
            ),
            "config": self.config.to_dict(),
        }
