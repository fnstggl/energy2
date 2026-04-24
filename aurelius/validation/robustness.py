"""Robustness test harness for Aurelius optimizer.

This module validates optimizer stability by running multiple simulations
with different random seeds and aggregating results.

This is a system-level sanity check, NOT a unit test.

Usage:
    python -m aurelius.validation.robustness --runs 20
"""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from ..simulation.replay import SimulationReplay, SimulationConfig
from ..models import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics from a single simulation run."""
    seed: int

    # Energy savings
    energy_savings_vs_fifo_pct: float
    energy_savings_vs_fifo_dollars: float
    energy_savings_vs_peak_pct: float
    energy_savings_vs_peak_dollars: float

    # Carbon savings
    carbon_savings_vs_fifo_pct: float
    carbon_savings_vs_fifo_kg: float
    carbon_savings_vs_peak_pct: float
    carbon_savings_vs_peak_kg: float

    # Compute cost (expected to be negative due to throttling)
    compute_cost_vs_fifo_pct: float
    compute_cost_vs_peak_pct: float

    # Region distribution
    region_distribution: dict[str, int]

    # Job metrics
    jobs_throttled: int
    jobs_shifted: int
    total_jobs: int


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all runs."""
    # Energy savings vs FIFO
    energy_vs_fifo_pct_mean: float
    energy_vs_fifo_pct_median: float
    energy_vs_fifo_pct_min: float
    energy_vs_fifo_pct_max: float
    energy_vs_fifo_pct_stdev: float

    energy_vs_fifo_dollars_mean: float
    energy_vs_fifo_dollars_median: float
    energy_vs_fifo_dollars_min: float
    energy_vs_fifo_dollars_max: float

    # Energy savings vs Peak-blind
    energy_vs_peak_pct_mean: float
    energy_vs_peak_pct_median: float
    energy_vs_peak_pct_min: float
    energy_vs_peak_pct_max: float
    energy_vs_peak_pct_stdev: float

    energy_vs_peak_dollars_mean: float
    energy_vs_peak_dollars_median: float
    energy_vs_peak_dollars_min: float
    energy_vs_peak_dollars_max: float

    # Carbon savings vs FIFO
    carbon_vs_fifo_pct_mean: float
    carbon_vs_fifo_pct_median: float
    carbon_vs_fifo_pct_min: float
    carbon_vs_fifo_pct_max: float
    carbon_vs_fifo_pct_stdev: float

    carbon_vs_fifo_kg_mean: float
    carbon_vs_fifo_kg_median: float
    carbon_vs_fifo_kg_min: float
    carbon_vs_fifo_kg_max: float

    # Carbon savings vs Peak-blind
    carbon_vs_peak_pct_mean: float
    carbon_vs_peak_pct_median: float
    carbon_vs_peak_pct_min: float
    carbon_vs_peak_pct_max: float
    carbon_vs_peak_pct_stdev: float

    carbon_vs_peak_kg_mean: float
    carbon_vs_peak_kg_median: float
    carbon_vs_peak_kg_min: float
    carbon_vs_peak_kg_max: float


@dataclass
class RobustnessReport:
    """Complete robustness test report."""
    timestamp: str
    num_runs: int
    config: dict

    # Aggregate metrics
    aggregates: AggregateMetrics

    # Individual run metrics
    runs: list[RunMetrics]

    # Warnings
    negative_energy_savings_count: int
    negative_carbon_savings_count: int
    warnings: list[str]

    # Verdict
    is_stable: bool
    stability_score: float  # 0-100


class RobustnessTestHarness:
    """Runs multiple simulations to validate optimizer stability."""

    def __init__(
        self,
        num_jobs: int = 50,
        duration_hours: int = 72,
        regions: Optional[list[str]] = None,
        optimization_method: str = "greedy",
        price_scenario: str = "normal",
        carbon_scenario: str = "normal",
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.05,
    ):
        """Initialize the test harness.

        Args:
            num_jobs: Number of jobs per simulation
            duration_hours: Simulation duration
            regions: List of regions
            optimization_method: Optimizer method
            price_scenario: Price scenario
            carbon_scenario: Carbon scenario
            alpha: Energy cost weight
            beta: Carbon cost weight
            gamma: Risk penalty weight
        """
        self.num_jobs = num_jobs
        self.duration_hours = duration_hours
        self.regions = regions or ["us-west", "us-east", "eu-west"]
        self.optimization_method = optimization_method
        self.price_scenario = price_scenario
        self.carbon_scenario = carbon_scenario
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.replay = SimulationReplay()

    def run(self, num_runs: int = 20, base_seed: int = 1000) -> RobustnessReport:
        """Run the robustness test.

        Args:
            num_runs: Number of simulation runs
            base_seed: Starting seed (seeds will be base_seed, base_seed+1, ...)

        Returns:
            RobustnessReport with all results
        """
        logger.info(f"Starting robustness test: {num_runs} runs")

        run_metrics: list[RunMetrics] = []
        warnings: list[str] = []

        for i in range(num_runs):
            seed = base_seed + i
            logger.info(f"Run {i+1}/{num_runs} (seed={seed})")

            try:
                metrics = self._run_single(seed)
                run_metrics.append(metrics)

                # Check for negative savings
                if metrics.energy_savings_vs_fifo_pct < 0:
                    msg = f"Run {i+1} (seed={seed}): Negative energy savings vs FIFO: {metrics.energy_savings_vs_fifo_pct:.1f}%"
                    warnings.append(msg)
                    logger.warning(msg)

                if metrics.energy_savings_vs_peak_pct < 0:
                    msg = f"Run {i+1} (seed={seed}): Negative energy savings vs Peak-blind: {metrics.energy_savings_vs_peak_pct:.1f}%"
                    warnings.append(msg)
                    logger.warning(msg)

            except Exception as e:
                msg = f"Run {i+1} (seed={seed}) FAILED: {str(e)}"
                warnings.append(msg)
                logger.error(msg)

        if not run_metrics:
            raise RuntimeError("All runs failed")

        # Aggregate results
        aggregates = self._aggregate(run_metrics)

        # Count negative savings
        neg_energy_count = sum(
            1 for m in run_metrics
            if m.energy_savings_vs_fifo_pct < 0 or m.energy_savings_vs_peak_pct < 0
        )
        neg_carbon_count = sum(
            1 for m in run_metrics
            if m.carbon_savings_vs_fifo_pct < 0 or m.carbon_savings_vs_peak_pct < 0
        )

        # Calculate stability score
        # 100 = all runs positive savings, 0 = all runs negative
        positive_runs = len(run_metrics) - neg_energy_count
        stability_score = (positive_runs / len(run_metrics)) * 100

        # Verdict: stable if >90% of runs have positive savings
        is_stable = stability_score >= 90.0

        return RobustnessReport(
            timestamp=datetime.utcnow().isoformat(),
            num_runs=len(run_metrics),
            config={
                "num_jobs": self.num_jobs,
                "duration_hours": self.duration_hours,
                "regions": self.regions,
                "optimization_method": self.optimization_method,
                "price_scenario": self.price_scenario,
                "carbon_scenario": self.carbon_scenario,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            aggregates=aggregates,
            runs=run_metrics,
            negative_energy_savings_count=neg_energy_count,
            negative_carbon_savings_count=neg_carbon_count,
            warnings=warnings,
            is_stable=is_stable,
            stability_score=round(stability_score, 1),
        )

    def _run_single(self, seed: int) -> RunMetrics:
        """Run a single simulation and extract metrics."""
        opt_config = OptimizationConfig(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        sim_config = SimulationConfig(
            start_time=datetime.utcnow(),
            duration_hours=self.duration_hours,
            regions=self.regions,
            num_jobs=self.num_jobs,
            optimization_method=self.optimization_method,
            optimization_config=opt_config,
            price_scenario=self.price_scenario,
            carbon_scenario=self.carbon_scenario,
            random_seed=seed,
            save_to_db=False,
        )

        results = self.replay.run(sim_config)

        # Extract metrics from results
        metrics_data = results.get("metrics", {})
        savings_fifo = metrics_data.get("savings_vs_fifo", {})
        savings_peak = metrics_data.get("savings_vs_peak_blind", {})
        optimized = metrics_data.get("optimized", {})

        return RunMetrics(
            seed=seed,
            energy_savings_vs_fifo_pct=savings_fifo.get("energy_cost_savings_pct", 0),
            energy_savings_vs_fifo_dollars=savings_fifo.get("energy_cost_savings_dollars", 0),
            energy_savings_vs_peak_pct=savings_peak.get("energy_cost_savings_pct", 0),
            energy_savings_vs_peak_dollars=savings_peak.get("energy_cost_savings_dollars", 0),
            carbon_savings_vs_fifo_pct=savings_fifo.get("carbon_savings_pct", 0),
            carbon_savings_vs_fifo_kg=savings_fifo.get("carbon_savings_kg", 0),
            carbon_savings_vs_peak_pct=savings_peak.get("carbon_savings_pct", 0),
            carbon_savings_vs_peak_kg=savings_peak.get("carbon_savings_kg", 0),
            compute_cost_vs_fifo_pct=savings_fifo.get("compute_cost_savings_pct", 0),
            compute_cost_vs_peak_pct=savings_peak.get("compute_cost_savings_pct", 0),
            region_distribution=optimized.get("region_distribution", {}),
            jobs_throttled=optimized.get("jobs_throttled", 0),
            jobs_shifted=optimized.get("jobs_shifted", 0),
            total_jobs=results.get("summary", {}).get("jobs_scheduled", 0),
        )

    def _aggregate(self, runs: list[RunMetrics]) -> AggregateMetrics:
        """Aggregate metrics across all runs."""
        def safe_stdev(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            return statistics.stdev(values)

        # Extract lists
        energy_fifo_pct = [r.energy_savings_vs_fifo_pct for r in runs]
        energy_fifo_dollars = [r.energy_savings_vs_fifo_dollars for r in runs]
        energy_peak_pct = [r.energy_savings_vs_peak_pct for r in runs]
        energy_peak_dollars = [r.energy_savings_vs_peak_dollars for r in runs]

        carbon_fifo_pct = [r.carbon_savings_vs_fifo_pct for r in runs]
        carbon_fifo_kg = [r.carbon_savings_vs_fifo_kg for r in runs]
        carbon_peak_pct = [r.carbon_savings_vs_peak_pct for r in runs]
        carbon_peak_kg = [r.carbon_savings_vs_peak_kg for r in runs]

        return AggregateMetrics(
            # Energy vs FIFO
            energy_vs_fifo_pct_mean=round(statistics.mean(energy_fifo_pct), 2),
            energy_vs_fifo_pct_median=round(statistics.median(energy_fifo_pct), 2),
            energy_vs_fifo_pct_min=round(min(energy_fifo_pct), 2),
            energy_vs_fifo_pct_max=round(max(energy_fifo_pct), 2),
            energy_vs_fifo_pct_stdev=round(safe_stdev(energy_fifo_pct), 2),
            energy_vs_fifo_dollars_mean=round(statistics.mean(energy_fifo_dollars), 2),
            energy_vs_fifo_dollars_median=round(statistics.median(energy_fifo_dollars), 2),
            energy_vs_fifo_dollars_min=round(min(energy_fifo_dollars), 2),
            energy_vs_fifo_dollars_max=round(max(energy_fifo_dollars), 2),

            # Energy vs Peak-blind
            energy_vs_peak_pct_mean=round(statistics.mean(energy_peak_pct), 2),
            energy_vs_peak_pct_median=round(statistics.median(energy_peak_pct), 2),
            energy_vs_peak_pct_min=round(min(energy_peak_pct), 2),
            energy_vs_peak_pct_max=round(max(energy_peak_pct), 2),
            energy_vs_peak_pct_stdev=round(safe_stdev(energy_peak_pct), 2),
            energy_vs_peak_dollars_mean=round(statistics.mean(energy_peak_dollars), 2),
            energy_vs_peak_dollars_median=round(statistics.median(energy_peak_dollars), 2),
            energy_vs_peak_dollars_min=round(min(energy_peak_dollars), 2),
            energy_vs_peak_dollars_max=round(max(energy_peak_dollars), 2),

            # Carbon vs FIFO
            carbon_vs_fifo_pct_mean=round(statistics.mean(carbon_fifo_pct), 2),
            carbon_vs_fifo_pct_median=round(statistics.median(carbon_fifo_pct), 2),
            carbon_vs_fifo_pct_min=round(min(carbon_fifo_pct), 2),
            carbon_vs_fifo_pct_max=round(max(carbon_fifo_pct), 2),
            carbon_vs_fifo_pct_stdev=round(safe_stdev(carbon_fifo_pct), 2),
            carbon_vs_fifo_kg_mean=round(statistics.mean(carbon_fifo_kg), 2),
            carbon_vs_fifo_kg_median=round(statistics.median(carbon_fifo_kg), 2),
            carbon_vs_fifo_kg_min=round(min(carbon_fifo_kg), 2),
            carbon_vs_fifo_kg_max=round(max(carbon_fifo_kg), 2),

            # Carbon vs Peak-blind
            carbon_vs_peak_pct_mean=round(statistics.mean(carbon_peak_pct), 2),
            carbon_vs_peak_pct_median=round(statistics.median(carbon_peak_pct), 2),
            carbon_vs_peak_pct_min=round(min(carbon_peak_pct), 2),
            carbon_vs_peak_pct_max=round(max(carbon_peak_pct), 2),
            carbon_vs_peak_pct_stdev=round(safe_stdev(carbon_peak_pct), 2),
            carbon_vs_peak_kg_mean=round(statistics.mean(carbon_peak_kg), 2),
            carbon_vs_peak_kg_median=round(statistics.median(carbon_peak_kg), 2),
            carbon_vs_peak_kg_min=round(min(carbon_peak_kg), 2),
            carbon_vs_peak_kg_max=round(max(carbon_peak_kg), 2),
        )


def format_cli_report(report: RobustnessReport) -> str:
    """Format report as a CLI table summary."""
    agg = report.aggregates

    lines = [
        "",
        "=" * 75,
        "AURELIUS ROBUSTNESS TEST REPORT",
        "=" * 75,
        f"Timestamp:     {report.timestamp}",
        f"Runs:          {report.num_runs}",
        f"Stability:     {'STABLE' if report.is_stable else 'UNSTABLE'} ({report.stability_score}%)",
        "",
        "-" * 75,
        "CONFIGURATION",
        "-" * 75,
        f"  Jobs:        {report.config['num_jobs']}",
        f"  Duration:    {report.config['duration_hours']} hours",
        f"  Regions:     {', '.join(report.config['regions'])}",
        f"  Method:      {report.config['optimization_method']}",
        f"  Weights:     alpha={report.config['alpha']}, beta={report.config['beta']}, gamma={report.config['gamma']}",
        "",
        "-" * 75,
        "ENERGY SAVINGS VS FIFO BASELINE",
        "-" * 75,
        f"  Mean:        {agg.energy_vs_fifo_pct_mean:>6.1f}%   (${agg.energy_vs_fifo_dollars_mean:>10,.2f})",
        f"  Median:      {agg.energy_vs_fifo_pct_median:>6.1f}%   (${agg.energy_vs_fifo_dollars_median:>10,.2f})",
        f"  Min:         {agg.energy_vs_fifo_pct_min:>6.1f}%   (${agg.energy_vs_fifo_dollars_min:>10,.2f})",
        f"  Max:         {agg.energy_vs_fifo_pct_max:>6.1f}%   (${agg.energy_vs_fifo_dollars_max:>10,.2f})",
        f"  Std Dev:     {agg.energy_vs_fifo_pct_stdev:>6.1f}%",
        "",
        "-" * 75,
        "ENERGY SAVINGS VS PEAK-BLIND BASELINE",
        "-" * 75,
        f"  Mean:        {agg.energy_vs_peak_pct_mean:>6.1f}%   (${agg.energy_vs_peak_dollars_mean:>10,.2f})",
        f"  Median:      {agg.energy_vs_peak_pct_median:>6.1f}%   (${agg.energy_vs_peak_dollars_median:>10,.2f})",
        f"  Min:         {agg.energy_vs_peak_pct_min:>6.1f}%   (${agg.energy_vs_peak_dollars_min:>10,.2f})",
        f"  Max:         {agg.energy_vs_peak_pct_max:>6.1f}%   (${agg.energy_vs_peak_dollars_max:>10,.2f})",
        f"  Std Dev:     {agg.energy_vs_peak_pct_stdev:>6.1f}%",
        "",
        "-" * 75,
        "CARBON SAVINGS VS FIFO BASELINE",
        "-" * 75,
        f"  Mean:        {agg.carbon_vs_fifo_pct_mean:>6.1f}%   ({agg.carbon_vs_fifo_kg_mean:>10,.1f} kg)",
        f"  Median:      {agg.carbon_vs_fifo_pct_median:>6.1f}%   ({agg.carbon_vs_fifo_kg_median:>10,.1f} kg)",
        f"  Min:         {agg.carbon_vs_fifo_pct_min:>6.1f}%   ({agg.carbon_vs_fifo_kg_min:>10,.1f} kg)",
        f"  Max:         {agg.carbon_vs_fifo_pct_max:>6.1f}%   ({agg.carbon_vs_fifo_kg_max:>10,.1f} kg)",
        f"  Std Dev:     {agg.carbon_vs_fifo_pct_stdev:>6.1f}%",
        "",
        "-" * 75,
        "CARBON SAVINGS VS PEAK-BLIND BASELINE",
        "-" * 75,
        f"  Mean:        {agg.carbon_vs_peak_pct_mean:>6.1f}%   ({agg.carbon_vs_peak_kg_mean:>10,.1f} kg)",
        f"  Median:      {agg.carbon_vs_peak_pct_median:>6.1f}%   ({agg.carbon_vs_peak_kg_median:>10,.1f} kg)",
        f"  Min:         {agg.carbon_vs_peak_pct_min:>6.1f}%   ({agg.carbon_vs_peak_kg_min:>10,.1f} kg)",
        f"  Max:         {agg.carbon_vs_peak_pct_max:>6.1f}%   ({agg.carbon_vs_peak_kg_max:>10,.1f} kg)",
        f"  Std Dev:     {agg.carbon_vs_peak_pct_stdev:>6.1f}%",
        "",
        "-" * 75,
        "WARNINGS",
        "-" * 75,
        f"  Negative energy savings runs: {report.negative_energy_savings_count}/{report.num_runs}",
        f"  Negative carbon savings runs: {report.negative_carbon_savings_count}/{report.num_runs}",
    ]

    if report.warnings:
        lines.append("")
        lines.append("  Detailed warnings:")
        for w in report.warnings[:10]:  # Limit to first 10
            lines.append(f"    - {w}")
        if len(report.warnings) > 10:
            lines.append(f"    ... and {len(report.warnings) - 10} more")

    lines.extend([
        "",
        "=" * 75,
        f"VERDICT: {'STABLE - Ready for live execution' if report.is_stable else 'UNSTABLE - Investigate before live execution'}",
        "=" * 75,
        "",
    ])

    return "\n".join(lines)


def report_to_dict(report: RobustnessReport) -> dict:
    """Convert report to JSON-serializable dictionary."""
    return {
        "timestamp": report.timestamp,
        "num_runs": report.num_runs,
        "config": report.config,
        "is_stable": report.is_stable,
        "stability_score": report.stability_score,
        "negative_energy_savings_count": report.negative_energy_savings_count,
        "negative_carbon_savings_count": report.negative_carbon_savings_count,
        "warnings": report.warnings,
        "aggregates": asdict(report.aggregates),
        "runs": [asdict(r) for r in report.runs],
    }


def save_report_json(report: RobustnessReport, filepath: Path) -> None:
    """Save report to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
