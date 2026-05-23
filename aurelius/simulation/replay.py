"""Simulation replay and orchestration.

This module orchestrates complete simulation runs:
1. Load or generate input data
2. Run forecasting
3. Execute optimization
4. Compare scenarios
5. Store results
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..database import get_db
from ..forecasting.baseline import generate_carbon_scenario, generate_price_scenario
from ..forecasting.carbon_model import CarbonForecaster
from ..forecasting.price_model import PriceForecaster
from ..forecasting.uncertainty import UncertaintyEstimator
from ..ingestion.energy_prices import EnergyPriceIngester
from ..ingestion.job_logs import JobLogIngester
from ..models import Job, OptimizationConfig
from .compare import ScenarioComparator

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        run_id: Unique identifier (auto-generated if not provided)
        start_time: Start of simulation window
        duration_hours: Duration of simulation in hours
        regions: Regions to simulate
        num_jobs: Number of jobs (for synthetic generation)
        optimization_method: Optimization method to use
        optimization_config: Weights and constraints
        price_scenario: Price scenario for synthetic data
        carbon_scenario: Carbon scenario for synthetic data
        random_seed: Seed for reproducibility
        save_to_db: Whether to save results to database
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    duration_hours: int = 168  # 1 week
    regions: list[str] = field(default_factory=lambda: ["us-west", "us-east", "eu-west"])
    num_jobs: int = 50
    optimization_method: str = "greedy"
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    price_scenario: str = "normal"
    carbon_scenario: str = "normal"
    random_seed: Optional[int] = 42
    save_to_db: bool = True


class SimulationReplay:
    """Orchestrates simulation replays.

    A simulation replay:
    1. Generates or loads energy price data
    2. Generates or loads carbon intensity data
    3. Generates or loads job workload
    4. Fits forecasting models
    5. Runs optimization
    6. Compares baseline vs optimized
    7. Stores and returns results
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
    ):
        """Initialize the simulation replay.

        Args:
            data_dir: Directory for data files
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.price_ingester = EnergyPriceIngester(self.data_dir)
        self.job_ingester = JobLogIngester(self.data_dir)
        self.db = get_db()

    def run(
        self,
        config: Optional[SimulationConfig] = None,
        jobs: Optional[list[Job]] = None,
        price_file: Optional[Path] = None,
        carbon_file: Optional[Path] = None,
    ) -> dict:
        """Run a complete simulation.

        Args:
            config: Simulation configuration
            jobs: Pre-defined jobs (optional, otherwise generated)
            price_file: Path to price data CSV (optional)
            carbon_file: Path to carbon data CSV (optional)

        Returns:
            Complete simulation results as dictionary
        """
        config = config or SimulationConfig()
        logger.info(f"Starting simulation {config.run_id}")

        # Step 1: Load or generate energy prices
        if price_file and price_file.exists():
            prices = self.price_ingester.load_from_csv(price_file)
            logger.info(f"Loaded {len(prices)} prices from {price_file}")
        else:
            prices = generate_price_scenario(
                start_time=config.start_time,
                hours=config.duration_hours,
                regions=config.regions,
                scenario=config.price_scenario,
                seed=config.random_seed,
            )
            logger.info(f"Generated {len(prices)} synthetic prices")

        # Step 2: Load or generate carbon intensity
        if carbon_file and carbon_file.exists():
            # Would need carbon ingester, using synthetic for now
            carbon_data = generate_carbon_scenario(
                start_time=config.start_time,
                hours=config.duration_hours,
                regions=config.regions,
                scenario=config.carbon_scenario,
                seed=config.random_seed,
            )
        else:
            carbon_data = generate_carbon_scenario(
                start_time=config.start_time,
                hours=config.duration_hours,
                regions=config.regions,
                scenario=config.carbon_scenario,
                seed=config.random_seed,
            )
            logger.info(f"Generated {len(carbon_data)} synthetic carbon records")

        # Step 3: Load or generate jobs
        if jobs:
            simulation_jobs = jobs
        else:
            simulation_jobs = self.job_ingester.generate_synthetic(
                start_time=config.start_time,
                duration_hours=config.duration_hours,
                num_jobs=config.num_jobs,
                regions=config.regions,
                seed=config.random_seed,
            )
            logger.info(f"Generated {len(simulation_jobs)} synthetic jobs")

        # Validate jobs
        valid_jobs, errors = self.job_ingester.validate_jobs(simulation_jobs)
        if errors:
            logger.warning(f"Dropped {len(errors)} invalid jobs")
        simulation_jobs = valid_jobs

        # Step 4: Fit forecasting models
        price_forecaster = PriceForecaster()
        price_forecaster.fit(prices)

        carbon_forecaster = CarbonForecaster()
        carbon_forecaster.fit(carbon_data)

        # Step 5: Generate forecasts and uncertainty
        forecast_horizon = config.duration_hours
        price_forecasts = []
        carbon_forecasts = []

        for region in config.regions:
            pf = price_forecaster.predict_range(
                region, config.start_time, forecast_horizon, prices
            )
            price_forecasts.extend(pf)

            cf = carbon_forecaster.predict_range(
                region, config.start_time, forecast_horizon, carbon_data
            )
            carbon_forecasts.extend(cf)

        # Estimate uncertainty
        uncertainty_estimator = UncertaintyEstimator()
        uncertainty_estimates = uncertainty_estimator.estimate_from_forecasts(
            price_forecasts, carbon_forecasts
        )

        # Convert to lookup dicts
        price_dict = self.price_ingester.prices_to_dict(prices)
        carbon_dict: dict = {}
        for c in carbon_data:
            if c.region not in carbon_dict:
                carbon_dict[c.region] = {}
            carbon_dict[c.region][c.timestamp] = c.gco2_per_kwh

        risk_dict = uncertainty_estimator.get_risk_penalty_dict(uncertainty_estimates)

        # Step 6: Run comparison
        comparator = ScenarioComparator(config.optimization_config)
        comparison = comparator.compare(
            simulation_jobs,
            price_dict,
            carbon_dict,
            risk_dict,
            optimization_method=config.optimization_method,
        )

        # Step 7: Generate results
        result = comparator.to_simulation_result(comparison, config.run_id)

        # Generate summary
        summary = comparator.generate_json_summary(comparison, config.run_id)

        # Add simulation metadata
        summary["simulation_config"] = {
            "start_time": config.start_time.isoformat(),
            "duration_hours": config.duration_hours,
            "num_jobs": len(simulation_jobs),
            "regions": config.regions,
            "optimization_method": config.optimization_method,
            "price_scenario": config.price_scenario,
            "carbon_scenario": config.carbon_scenario,
        }

        # Add detailed schedule info (includes both baselines)
        summary["schedules"] = {
            "fifo_baseline": [
                {
                    "job_id": d.job_id,
                    "start_time": d.start_time.isoformat(),
                    "end_time": d.end_time.isoformat(),
                    "region": d.region,
                    "power_fraction": d.power_fraction,
                }
                for d in comparison.fifo_schedule
            ],
            "peak_blind_baseline": [
                {
                    "job_id": d.job_id,
                    "start_time": d.start_time.isoformat(),
                    "end_time": d.end_time.isoformat(),
                    "region": d.region,
                    "power_fraction": d.power_fraction,
                }
                for d in comparison.peak_blind_schedule
            ],
            "optimized": [
                {
                    "job_id": d.job_id,
                    "start_time": d.start_time.isoformat(),
                    "end_time": d.end_time.isoformat(),
                    "region": d.region,
                    "power_fraction": d.power_fraction,
                }
                for d in comparison.optimized_schedule
            ],
        }

        # Step 8: Save to database if configured
        if config.save_to_db and self.db.is_connected:
            saved = self.db.save_simulation(result.to_summary_dict())
            if saved:
                logger.info(f"Saved simulation {config.run_id} to database")
            else:
                logger.warning("Failed to save simulation to database")

        # Print report
        report = comparator.generate_report(comparison)
        logger.info("\n" + report)

        return summary

    def run_scenario_sweep(
        self,
        base_config: SimulationConfig,
        scenarios: list[dict],
    ) -> list[dict]:
        """Run multiple scenarios for comparison.

        Args:
            base_config: Base configuration
            scenarios: List of scenario overrides

        Returns:
            List of simulation results
        """
        results = []

        for i, scenario in enumerate(scenarios):
            config = SimulationConfig(
                run_id=f"{base_config.run_id}-scenario-{i}",
                start_time=base_config.start_time,
                duration_hours=base_config.duration_hours,
                regions=base_config.regions,
                num_jobs=base_config.num_jobs,
                optimization_method=scenario.get("method", base_config.optimization_method),
                optimization_config=OptimizationConfig(
                    alpha=scenario.get("alpha", base_config.optimization_config.alpha),
                    beta=scenario.get("beta", base_config.optimization_config.beta),
                    gamma=scenario.get("gamma", base_config.optimization_config.gamma),
                ),
                price_scenario=scenario.get("price_scenario", base_config.price_scenario),
                carbon_scenario=scenario.get("carbon_scenario", base_config.carbon_scenario),
                random_seed=base_config.random_seed,
                save_to_db=base_config.save_to_db,
            )

            logger.info(f"Running scenario {i+1}/{len(scenarios)}")
            result = self.run(config)
            result["scenario_id"] = i
            result["scenario_params"] = scenario
            results.append(result)

        return results

    def save_results_to_file(
        self,
        results: dict,
        filepath: Path,
    ) -> None:
        """Save simulation results to JSON file.

        Args:
            results: Simulation results dictionary
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")

    def load_results_from_file(
        self,
        filepath: Path,
    ) -> dict:
        """Load simulation results from JSON file.

        Args:
            filepath: Input file path

        Returns:
            Simulation results dictionary
        """
        with open(filepath, "r") as f:
            return json.load(f)
