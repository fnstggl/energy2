"""Core data models for Aurelius.

These models represent the fundamental entities in the system:
- Jobs (Slurm-style batch jobs)
- Energy prices (hourly time series)
- Carbon intensity signals
- Simulation results
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

# Supported workload types
WorkloadType = Literal[
    "realtime_inference",
    "llm_batch_inference",
    "fine_tuning",
    "training",
    "data_processing",
    "scheduled_batch",
    "background_maintenance",
]

SlaClass = Literal["latency_critical", "deadline", "best_effort"]

# Default scheduling flexibility windows per workload type (hours)
WORKLOAD_DEFAULT_MAX_DELAY_HOURS: dict[str, float] = {
    "realtime_inference": 0.0,
    "llm_batch_inference": 24.0,
    "fine_tuning": 24.0,
    "training": 48.0,
    "data_processing": 48.0,
    "scheduled_batch": 24.0,
    "background_maintenance": 48.0,
}

# Default interruptibility per workload type
WORKLOAD_DEFAULT_INTERRUPTIBLE: dict[str, bool] = {
    "realtime_inference": False,
    "llm_batch_inference": True,
    "fine_tuning": False,  # overridden to True when checkpointable=True
    "training": False,      # overridden to True when checkpointable=True
    "data_processing": True,
    "scheduled_batch": True,
    "background_maintenance": True,
}

# Default SLA class per workload type
WORKLOAD_DEFAULT_SLA_CLASS: dict[str, str] = {
    "realtime_inference": "latency_critical",
    "llm_batch_inference": "deadline",
    "fine_tuning": "deadline",
    "training": "best_effort",
    "data_processing": "deadline",
    "scheduled_batch": "deadline",
    "background_maintenance": "best_effort",
}


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
        region_options: List of regions where job can run (must be non-empty)
        priority: Job priority (higher = more important)
        latest_start: Latest allowed start time (derived from deadline - runtime)
        workload_type: Category of workload for default interruptibility/SLA
        gpu_type: GPU model identifier (e.g. "a100", "h100", "v100", "t4")
        gpu_count: Number of GPUs required
        sla_penalty_per_hour: Cost incurred per hour past deadline ($/hr)
        sla_class: SLA tier — determines safety gate thresholds
        data_transfer_gb: Estimated inter-region data transfer in GB
        pue: Power Usage Effectiveness multiplier for facility overhead
        interruptible: Whether the job can be preempted mid-run
        preemptible: Whether the job can be killed and requeued
        checkpointable: Whether the job saves state and can be resumed
        max_delay_hours: Maximum allowed scheduling delay from earliest_start
        allowed_regions: If non-empty, optimizer must only select from these
        forbidden_regions: Optimizer must never select these regions
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

    # Phase 3 fields — all optional with safe defaults
    workload_type: str = "scheduled_batch"
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    sla_penalty_per_hour: float = 0.0
    sla_class: str = "best_effort"
    data_transfer_gb: float = 0.0
    pue: float = 1.0
    interruptible: bool = False
    preemptible: bool = False
    checkpointable: bool = False
    max_delay_hours: Optional[float] = None
    allowed_regions: list[str] = field(default_factory=list)
    forbidden_regions: list[str] = field(default_factory=list)

    # Mid-job region migration support. None means the job CANNOT migrate
    # (e.g. realtime_inference — latency-pinned to one region). A float value
    # is the cost in hours per migration: checkpoint-write + cross-region
    # state transfer + warmup at the destination. The job consumes energy
    # during this window (at the destination region's price) but does no
    # useful work. Typical values: training ~0.5h, fine-tuning ~0.25h,
    # batch inference ~0.1h, stateless data jobs ~0.05h.
    migration_cost_hours: Optional[float] = None

    def __post_init__(self):
        if self.latest_start is None:
            from datetime import timedelta
            self.latest_start = self.deadline - timedelta(hours=self.runtime_hours)

        # Enforce data residency: allowed_regions narrows region_options
        if self.allowed_regions:
            valid = [r for r in self.region_options if r in self.allowed_regions]
            if valid:
                self.region_options = valid
            # If no valid region exists, keep original so optimizer can mark unschedulable

        # Remove forbidden regions from options
        if self.forbidden_regions:
            filtered = [r for r in self.region_options if r not in self.forbidden_regions]
            if filtered:
                self.region_options = filtered
            # If all forbidden, keep original and let optimizer handle

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
class ScheduleSegment:
    """One contiguous (region, time-window) leg of a possibly-migrated schedule.

    A single-region (non-migrating) job has exactly one segment. A job that
    migrates K times has K+1 segments. Between consecutive segments the job
    incurs the source job's `migration_cost_hours` of paid-but-no-useful-work
    time, which is included in the segment's own duration (the destination
    segment starts later than `previous_segment.end_time` by migration_cost_hours
    of "warmup" already baked into the destination segment's start_time and
    its useful_runtime_hours computation).

    For scoring, the segment's [start_time, end_time) window is the entire
    paid window (useful work + any migration overhead at the start of this
    segment). This makes evaluator math trivial: just sum hourly prices over
    each segment's window at that segment's region.
    """
    start_time: datetime
    end_time: datetime  # exclusive
    region: str
    power_fraction: float = 1.0


@dataclass
class ScheduleDecision:
    """A scheduling decision for a single job.

    Attributes:
        job_id: The job being scheduled
        start_time: First-segment start time (for back-compat with single-segment readers)
        region: First-segment region (for back-compat)
        power_fraction: First-segment power level (1.0 = full, 0.5 = half speed)
        actual_runtime_hours: Total runtime including migration overhead
        forecast: Optional forecast snapshot used to make this decision.
            Schema: {
              "energy_cost": {"p50": float, "p90": float, "baseline": float},
              "carbon":      {"p50": float, "p90": float, "baseline": float},
            }
            None means no forecast was available at decision time.
            QuantileSafetyGate treats None as a blocked decision (fail-closed).
        segments: Optional list of ScheduleSegment for migrated jobs.
            None means single-segment (back-compat). When set, must be
            non-empty and the first segment's start_time/region must match
            the top-level fields.
    """
    job_id: str
    start_time: datetime
    region: str
    power_fraction: float
    actual_runtime_hours: float
    forecast: Optional[dict] = None
    segments: Optional[list[ScheduleSegment]] = None

    @property
    def end_time(self) -> datetime:
        from datetime import timedelta
        if self.segments:
            return self.segments[-1].end_time
        return self.start_time + timedelta(hours=self.actual_runtime_hours)

    @property
    def migration_count(self) -> int:
        """Number of region migrations (0 for single-segment schedules)."""
        if self.segments is None or len(self.segments) <= 1:
            return 0
        return len(self.segments) - 1

    @property
    def all_segments(self) -> list[ScheduleSegment]:
        """Return segments, synthesizing a single-segment list if needed.

        This lets the evaluator and other consumers iterate uniformly over
        decisions regardless of whether they were produced by a migration-aware
        optimizer or a single-segment one.
        """
        from datetime import timedelta
        if self.segments is not None:
            return self.segments
        return [ScheduleSegment(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=self.actual_runtime_hours),
            region=self.region,
            power_fraction=self.power_fraction,
        )]


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
        delta: Weight for SLA penalty cost
        min_power_fraction: Minimum allowed power throttle
        max_power_fraction: Maximum power (usually 1.0)
        region_power_caps: Power cap per region in kW
        default_region: Default region for baseline
        carbon_objective: Carbon optimization mode
            "cost_only" — ignore carbon signals
            "cost_with_carbon_weight" — minimize alpha*cost + beta*carbon
            "carbon_constrained" — enforce carbon threshold as hard constraint
        carbon_threshold_gco2_per_kwh: Hard carbon cap (carbon_constrained mode)
        data_transfer_cost_per_gb: Cost per GB of inter-region transfer ($/GB)
        sla_risk_thresholds: Max acceptable downside risk per workload_type (fraction)
    """
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.05
    delta: float = 1.0  # SLA penalty weight
    min_power_fraction: float = 0.5
    max_power_fraction: float = 1.0
    region_power_caps: dict[str, float] = field(default_factory=lambda: {
        "us-west": 10000,
        "us-east": 10000,
        "eu-west": 8000,
    })
    default_region: str = "us-east"
    carbon_objective: str = "cost_only"
    carbon_threshold_gco2_per_kwh: Optional[float] = None
    data_transfer_cost_per_gb: float = 0.01
    sla_risk_thresholds: dict[str, float] = field(default_factory=lambda: {
        "realtime_inference": 0.02,
        "llm_batch_inference": 0.05,
        "fine_tuning": 0.07,
        "training": 0.10,
        "data_processing": 0.07,
        "scheduled_batch": 0.07,
        "background_maintenance": 0.10,
    })

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "min_power_fraction": self.min_power_fraction,
            "max_power_fraction": self.max_power_fraction,
            "region_power_caps": self.region_power_caps,
            "default_region": self.default_region,
            "carbon_objective": self.carbon_objective,
            "carbon_threshold_gco2_per_kwh": self.carbon_threshold_gco2_per_kwh,
            "data_transfer_cost_per_gb": self.data_transfer_cost_per_gb,
        }
