"""GPU-typed realistic workload generation for Aurelius.

Produces Job objects with statistically distinct power, runtime, and
constraint profiles per workload_type × gpu_type combination.
All distributions are seeded for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..models import (
    Job,
)

# ---------------------------------------------------------------------------
# GPU power profiles (base power in kW at full utilisation)
# ---------------------------------------------------------------------------
_GPU_BASE_POWER_KW: dict[str, float] = {
    "h100": 700.0,
    "a100": 400.0,
    "a10g": 150.0,
    "v100": 300.0,
    "l4": 75.0,
    "t4": 70.0,
    "p100": 250.0,
    "cpu": 40.0,
}

# Fraction of base power used per gpu_count unit
_GPU_EFFICIENCY: dict[str, float] = {
    "h100": 0.95,
    "a100": 0.90,
    "a10g": 0.85,
    "v100": 0.85,
    "l4": 0.80,
    "t4": 0.80,
    "p100": 0.82,
    "cpu": 1.00,
}

# ---------------------------------------------------------------------------
# Workload type profiles
# ---------------------------------------------------------------------------

@dataclass
class _WorkloadProfile:
    runtime_hours_min: float
    runtime_hours_max: float
    gpu_count_options: list[int]
    power_kw_cpu_base: float       # power when gpu_count=0
    sla_penalty_range: tuple[float, float]
    data_transfer_gb_range: tuple[float, float]
    pue_range: tuple[float, float]
    checkpointable_prob: float
    interruptible: bool
    preemptible: bool
    sla_class: str
    max_delay_hours: float


_PROFILES: dict[str, _WorkloadProfile] = {
    "realtime_inference": _WorkloadProfile(
        runtime_hours_min=0.05,
        runtime_hours_max=0.5,
        gpu_count_options=[1, 2, 4],
        power_kw_cpu_base=20.0,
        sla_penalty_range=(500.0, 5000.0),  # very high — latency critical
        data_transfer_gb_range=(0.1, 2.0),
        pue_range=(1.1, 1.2),
        checkpointable_prob=0.0,
        interruptible=False,
        preemptible=False,
        sla_class="latency_critical",
        max_delay_hours=0.0,
    ),
    "llm_batch_inference": _WorkloadProfile(
        runtime_hours_min=1.0,
        runtime_hours_max=8.0,
        gpu_count_options=[4, 8, 16],
        power_kw_cpu_base=80.0,
        sla_penalty_range=(10.0, 100.0),
        data_transfer_gb_range=(1.0, 20.0),
        pue_range=(1.1, 1.4),
        checkpointable_prob=0.4,
        interruptible=True,
        preemptible=True,
        sla_class="deadline",
        max_delay_hours=24.0,
    ),
    "fine_tuning": _WorkloadProfile(
        runtime_hours_min=4.0,
        runtime_hours_max=48.0,
        gpu_count_options=[4, 8, 16, 32],
        power_kw_cpu_base=100.0,
        sla_penalty_range=(20.0, 200.0),
        data_transfer_gb_range=(5.0, 100.0),
        pue_range=(1.1, 1.5),
        checkpointable_prob=0.8,
        interruptible=False,  # becomes True when checkpointable
        preemptible=False,
        sla_class="deadline",
        max_delay_hours=24.0,
    ),
    "training": _WorkloadProfile(
        runtime_hours_min=24.0,
        runtime_hours_max=168.0,
        gpu_count_options=[8, 16, 32, 64],
        power_kw_cpu_base=200.0,
        sla_penalty_range=(5.0, 50.0),
        data_transfer_gb_range=(10.0, 500.0),
        pue_range=(1.2, 1.6),
        checkpointable_prob=0.7,
        interruptible=False,  # becomes True when checkpointable
        preemptible=False,
        sla_class="best_effort",
        max_delay_hours=48.0,
    ),
    "data_processing": _WorkloadProfile(
        runtime_hours_min=1.0,
        runtime_hours_max=12.0,
        gpu_count_options=[0, 2, 4],
        power_kw_cpu_base=50.0,
        sla_penalty_range=(5.0, 50.0),
        data_transfer_gb_range=(10.0, 1000.0),
        pue_range=(1.1, 1.3),
        checkpointable_prob=0.3,
        interruptible=True,
        preemptible=True,
        sla_class="deadline",
        max_delay_hours=48.0,
    ),
    "scheduled_batch": _WorkloadProfile(
        runtime_hours_min=1.0,
        runtime_hours_max=8.0,
        gpu_count_options=[0, 1, 2],
        power_kw_cpu_base=30.0,
        sla_penalty_range=(1.0, 20.0),
        data_transfer_gb_range=(0.5, 10.0),
        pue_range=(1.1, 1.3),
        checkpointable_prob=0.2,
        interruptible=True,
        preemptible=False,
        sla_class="deadline",
        max_delay_hours=24.0,
    ),
    "background_maintenance": _WorkloadProfile(
        runtime_hours_min=0.5,
        runtime_hours_max=4.0,
        gpu_count_options=[0],
        power_kw_cpu_base=15.0,
        sla_penalty_range=(0.0, 5.0),
        data_transfer_gb_range=(0.0, 5.0),
        pue_range=(1.0, 1.2),
        checkpointable_prob=0.1,
        interruptible=True,
        preemptible=True,
        sla_class="best_effort",
        max_delay_hours=48.0,
    ),
}


def _gpu_power_kw(
    rng: random.Random,
    profile: _WorkloadProfile,
    gpu_type: str,
    gpu_count: int,
) -> float:
    """Compute expected power draw for a GPU allocation."""
    if gpu_count == 0:
        base = profile.power_kw_cpu_base
        return rng.uniform(base * 0.8, base * 1.2)

    gpu_base = _GPU_BASE_POWER_KW.get(gpu_type, 300.0)
    efficiency = _GPU_EFFICIENCY.get(gpu_type, 0.85)
    # Power scales sub-linearly with count due to communication overhead
    power = gpu_base * gpu_count * efficiency
    # Add ±15% stochastic variation
    return power * rng.uniform(0.85, 1.15)


class WorkloadSimulator:
    """Generates realistic, GPU-typed synthetic job workloads.

    Each call to generate() produces a list of Job objects whose power,
    runtime, SLA, and constraint characteristics match the chosen
    workload_type × gpu_type profile.

    The same (workload_type, gpu_type, n_jobs, seed) arguments always
    produce the same output (fully reproducible).
    """

    def __init__(
        self,
        regions: Optional[list[str]] = None,
        submit_base: Optional[datetime] = None,
    ):
        """
        Args:
            regions: Region pool for job region_options. Defaults to
                ["us-west", "us-east", "eu-west"].
            submit_base: Base submit time. Defaults to 2024-01-01T00:00:00.
        """
        self.regions = regions or ["us-west", "us-east", "eu-west"]
        self.submit_base = submit_base or datetime(2024, 1, 1, 0, 0, 0)

    def generate(
        self,
        workload_type: str,
        gpu_type: str = "a100",
        n_jobs: int = 10,
        seed: int = 42,
    ) -> list[Job]:
        """Generate n_jobs of the given workload_type using the given GPU.

        Args:
            workload_type: One of the supported workload types.
            gpu_type: GPU model key (e.g. "a100", "h100", "t4", "cpu").
            n_jobs: Number of jobs to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of Job objects with realistic characteristics.

        Raises:
            ValueError: If workload_type is not supported.
        """
        if workload_type not in _PROFILES:
            raise ValueError(
                f"Unsupported workload_type={workload_type!r}. "
                f"Choose from: {sorted(_PROFILES)}"
            )
        if n_jobs <= 0:
            raise ValueError(f"n_jobs must be > 0, got {n_jobs}")

        gpu_type = gpu_type.lower()
        if gpu_type not in _GPU_BASE_POWER_KW:
            raise ValueError(
                f"Unknown gpu_type={gpu_type!r}. "
                f"Known types: {sorted(_GPU_BASE_POWER_KW)}"
            )

        rng = random.Random(seed)
        profile = _PROFILES[workload_type]
        jobs: list[Job] = []

        for i in range(n_jobs):
            job_id = f"{workload_type[:4]}-{gpu_type}-{i:04d}-{seed}"

            # Submit time: spread over a 7-day window
            submit_offset_hours = rng.uniform(0, 168)
            submit_time = self.submit_base + timedelta(hours=submit_offset_hours)

            # Runtime
            runtime_hours = rng.uniform(
                profile.runtime_hours_min, profile.runtime_hours_max
            )

            # Earliest start = submit time + up to 15 minutes queueing delay
            earliest_start = submit_time + timedelta(minutes=rng.uniform(0, 15))

            # Deadline = earliest_start + runtime + flexibility window
            delay_hours = rng.uniform(runtime_hours, runtime_hours + profile.max_delay_hours)
            deadline = earliest_start + timedelta(hours=delay_hours)

            # GPU count
            gpu_count = rng.choice(profile.gpu_count_options)

            # Power draw
            power_kw = _gpu_power_kw(rng, profile, gpu_type, gpu_count)

            # SLA penalty
            sla_penalty = rng.uniform(*profile.sla_penalty_range)

            # Data transfer
            data_transfer_gb = rng.uniform(*profile.data_transfer_gb_range)

            # PUE
            pue = rng.uniform(*profile.pue_range)

            # Checkpointable (probabilistic)
            checkpointable = rng.random() < profile.checkpointable_prob

            # interruptible: use profile default, but fine_tuning/training
            # become interruptible when checkpointable
            interruptible = profile.interruptible
            if workload_type in ("fine_tuning", "training") and checkpointable:
                interruptible = True

            # Region options: 2–3 regions from pool
            n_regions = rng.randint(1, len(self.regions))
            region_options = rng.sample(self.regions, n_regions)

            jobs.append(Job(
                job_id=job_id,
                submit_time=submit_time,
                runtime_hours=runtime_hours,
                deadline=deadline,
                power_kw=power_kw,
                earliest_start=earliest_start,
                region_options=region_options,
                priority=rng.randint(1, 5),
                workload_type=workload_type,
                gpu_type=gpu_type if gpu_count > 0 else None,
                gpu_count=gpu_count,
                sla_penalty_per_hour=sla_penalty,
                sla_class=profile.sla_class,
                data_transfer_gb=data_transfer_gb,
                pue=pue,
                interruptible=interruptible,
                preemptible=profile.preemptible,
                checkpointable=checkpointable,
                max_delay_hours=profile.max_delay_hours,
            ))

        return jobs

    def generate_mixed(
        self,
        workload_mix: dict[str, int],
        gpu_type: str = "a100",
        seed: int = 42,
    ) -> list[Job]:
        """Generate a mixed workload from multiple types.

        Args:
            workload_mix: {workload_type: n_jobs} mapping.
            gpu_type: Default GPU type for all workloads.
            seed: Base seed (each workload_type gets seed+hash offset).

        Returns:
            Combined list of Job objects sorted by submit_time.
        """
        all_jobs: list[Job] = []
        for wtype, n in workload_mix.items():
            type_seed = seed ^ hash(wtype) & 0xFFFFFF
            jobs = self.generate(wtype, gpu_type=gpu_type, n_jobs=n, seed=type_seed)
            all_jobs.extend(jobs)
        all_jobs.sort(key=lambda j: j.submit_time)
        return all_jobs
