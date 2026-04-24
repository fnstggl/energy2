"""Core data models for Aurelius.

These models represent the fundamental entities in the system:
- Jobs (Slurm-style batch jobs)
- Energy prices (hourly time series)
- Carbon intensity signals
- Simulation results
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class Job:
    """A batch compute job (Slurm-style abstraction).

    Attributes:
        job_id: Unique identifier for the job
        submit_time: When the job was submitted
        runtime_hours: Base runtime in hours (at full power)
        deadline: Latest acceptable completion time
        power_kw: Power consumption in kilowatts at full speed
        earliest_start: Earliest allowed start time
        latest_start: Latest allowed start time (derived from deadline - runtime)
        region_options: List of regions where job can run
        priority: Job priority (higher = more important)
    """
    job_id: str
    submit_time: datetime
    runtime_hours: float
    deadline: datetime
    power_kw: float
    earliest_start: datetime
    region_options: list[str]
    priority: int = 1
    latest_start: Optional[datetime] = None

    def __post_init__(self):
        if self.latest_start is None:
            from datetime import timedelta
            self.latest_start = self.deadline - timedelta(hours=self.runtime_hours)

    @property
    def slack_hours(self) -> float:
        """Available scheduling flexibility in hours."""
        return (self.latest_start - self.earliest_start).total_seconds() / 3600

    def adjusted_runtime(self, power_fraction: float) -> float:
        """Runtime when running at reduced power (throttled).

        Lower power = longer runtime (linear relationship assumed).
        """
        if power_fraction <= 0 or power_fraction > 1:
            raise ValueError("power_fraction must be in (0, 1]")
        return self.runtime_hours / power_fraction


@dataclass
class EnergyPrice:
    """Hourly energy price for a region.

    Attributes:
        timestamp: Hour (UTC) this price applies to
        region: Geographic region identifier
        price_per_mwh: Price in $/MWh
    """
    timestamp: datetime
    region: str
    price_per_mwh: float

    @property
    def price_per_kwh(self) -> float:
        return self.price_per_mwh / 1000


@dataclass
class CarbonIntensity:
    """Hourly carbon intensity for a region.

    Attributes:
        timestamp: Hour (UTC) this reading applies to
        region: Geographic region identifier
        gco2_per_kwh: Grams CO2 equivalent per kWh
    """
    timestamp: datetime
    region: str
    gco2_per_kwh: float


@dataclass
class ScheduleDecision:
    """A scheduling decision for a single job.

    Attributes:
        job_id: The job being scheduled
        start_time: Decided start time
        region: Decided region
        power_fraction: Power level (1.0 = full, 0.5 = half speed)
        actual_runtime_hours: Runtime after throttling adjustment
    """
    job_id: str
    start_time: datetime
    region: str
    power_fraction: float
    actual_runtime_hours: float

    @property
    def end_time(self) -> datetime:
        from datetime import timedelta
        return self.start_time + timedelta(hours=self.actual_runtime_hours)


@dataclass
class SimulationResult:
    """Results from a simulation run.

    Attributes:
        run_id: Unique identifier for this simulation
        baseline_cost: Total energy cost under baseline policy ($)
        optimized_cost: Total energy cost under optimized policy ($)
        baseline_carbon: Total carbon emissions under baseline (kg CO2)
        optimized_carbon: Total carbon emissions under optimized (kg CO2)
        created_at: When this simulation was run
        config: Configuration used for optimization
        baseline_schedule: List of baseline scheduling decisions
        optimized_schedule: List of optimized scheduling decisions
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    baseline_cost: float = 0.0
    optimized_cost: float = 0.0
    baseline_carbon: float = 0.0
    optimized_carbon: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: dict = field(default_factory=dict)
    baseline_schedule: list[ScheduleDecision] = field(default_factory=list)
    optimized_schedule: list[ScheduleDecision] = field(default_factory=list)

    @property
    def cost_savings_pct(self) -> float:
        """Percentage cost savings from optimization."""
        if self.baseline_cost == 0:
            return 0.0
        return ((self.baseline_cost - self.optimized_cost) / self.baseline_cost) * 100

    @property
    def carbon_savings_pct(self) -> float:
        """Percentage carbon savings from optimization."""
        if self.baseline_carbon == 0:
            return 0.0
        return ((self.baseline_carbon - self.optimized_carbon) / self.baseline_carbon) * 100

    @property
    def cost_delta(self) -> float:
        """Absolute cost difference ($)."""
        return self.baseline_cost - self.optimized_cost

    @property
    def carbon_delta(self) -> float:
        """Absolute carbon difference (kg CO2)."""
        return self.baseline_carbon - self.optimized_carbon

    def to_summary_dict(self) -> dict:
        """Convert to summary dictionary for API responses."""
        return {
            "run_id": self.run_id,
            "baseline_cost": round(self.baseline_cost, 2),
            "optimized_cost": round(self.optimized_cost, 2),
            "cost_delta": round(self.cost_delta, 2),
            "cost_savings_pct": round(self.cost_savings_pct, 2),
            "baseline_carbon_kg": round(self.baseline_carbon, 2),
            "optimized_carbon_kg": round(self.optimized_carbon, 2),
            "carbon_delta_kg": round(self.carbon_delta, 2),
            "carbon_savings_pct": round(self.carbon_savings_pct, 2),
            "jobs_scheduled": len(self.optimized_schedule),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OptimizationConfig:
    """Configuration for the optimizer.

    Attributes:
        alpha: Weight for energy cost objective
        beta: Weight for carbon cost objective
        gamma: Weight for risk/uncertainty penalty
        min_power_fraction: Minimum allowed power throttle
        max_power_fraction: Maximum power (usually 1.0)
        region_power_caps: Power cap per region in kW
        default_region: Default region for baseline
    """
    alpha: float = 1.0
    beta: float = 0.3  # Increased to make carbon tradeoffs visible
    gamma: float = 0.05
    min_power_fraction: float = 0.5
    max_power_fraction: float = 1.0
    region_power_caps: dict[str, float] = field(default_factory=lambda: {
        "us-west": 10000,
        "us-east": 10000,
        "eu-west": 8000,
    })
    # Default region for baseline - uses expensive/dirty region to show optimization value
    default_region: str = "us-east"

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "min_power_fraction": self.min_power_fraction,
            "max_power_fraction": self.max_power_fraction,
            "region_power_caps": self.region_power_caps,
            "default_region": self.default_region,
        }
