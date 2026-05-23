"""Job log ingestion for batch compute workloads.

This module handles:
- Loading job data from CSV/JSON files
- Generating synthetic job batches for simulation
- Storing jobs in Supabase
- Job validation and normalization
"""

import csv
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import logging

from ..models import Job
from ..database import get_db

logger = logging.getLogger(__name__)


class JobLogIngester:
    """Handles batch job data ingestion and generation."""

    # Legacy size-based profiles (used when workload_mix is None / "legacy")
    JOB_PROFILES = {
        "small": {"power_kw": (10, 50), "runtime_hours": (0.5, 2)},
        "medium": {"power_kw": (50, 200), "runtime_hours": (2, 8)},
        "large": {"power_kw": (200, 500), "runtime_hours": (4, 24)},
        "xlarge": {"power_kw": (500, 1000), "runtime_hours": (12, 72)},
    }

    # Workload-type profiles — realistic datacenter mix with per-profile
    # slack and multi-region flexibility. Used when workload_mix="realistic".
    # Slack values reflect the typical deadline flexibility of each workload type:
    #   realtime_inference: ~minutes (load-balancer-level flex only)
    #   llm_batch_inference: hours (overnight batch summarization, embeddings)
    #   fine_tuning: 1-3 days (research iteration, not customer-facing)
    #   training: 2-14 days (large pre-training jobs, the biggest cost lever)
    #   data_processing: hours-days (ETL, feature gen)
    #   scheduled_batch: 8-72h (cron-style nightly/weekly jobs)
    #   background_maintenance: days (housekeeping, GC, log compaction)
    # migration_cost_hours = how much paid-but-no-useful-work time a single
    # region migration costs (checkpoint write + cross-region state transfer +
    # warmup at destination). None means the job cannot migrate at all.
    WORKLOAD_PROFILES = {
        "realtime_inference": {
            "power_kw": (5, 80),
            "runtime_hours": (0.25, 2),
            "slack_hours": (0, 2),
            "multi_region_pct": 0.10,  # latency-bound; mostly pinned
            "weight": 0.10,
            "migration_cost_hours": None,  # cannot migrate — latency SLA pinned to one region
        },
        "llm_batch_inference": {
            "power_kw": (50, 300),
            "runtime_hours": (1, 8),
            "slack_hours": (4, 24),
            "multi_region_pct": 0.90,
            "weight": 0.15,
            "migration_cost_hours": 0.10,  # ~6 min: small KV cache + framework warmup
        },
        "fine_tuning": {
            "power_kw": (100, 500),
            "runtime_hours": (4, 24),
            "slack_hours": (24, 72),
            "multi_region_pct": 0.90,
            "weight": 0.15,
            "migration_cost_hours": 0.25,  # ~15 min: optimizer state checkpoint + transfer
        },
        "training": {
            "power_kw": (200, 2000),
            "runtime_hours": (24, 168),
            "slack_hours": (48, 336),  # 2-14 days slack
            "multi_region_pct": 0.90,
            "weight": 0.15,
            "migration_cost_hours": 0.50,  # ~30 min: large model + optimizer + dataloader warmup
        },
        "data_processing": {
            "power_kw": (20, 200),
            "runtime_hours": (1, 8),
            "slack_hours": (6, 48),
            "multi_region_pct": 0.85,
            "weight": 0.20,
            "migration_cost_hours": 0.05,  # ~3 min: mostly stateless (input is files)
        },
        "scheduled_batch": {
            "power_kw": (10, 200),
            "runtime_hours": (1, 12),
            "slack_hours": (8, 72),
            "multi_region_pct": 0.80,
            "weight": 0.15,
            "migration_cost_hours": 0.10,  # ~6 min
        },
        "background_maintenance": {
            "power_kw": (5, 100),
            "runtime_hours": (0.5, 4),
            "slack_hours": (24, 168),
            "multi_region_pct": 0.95,
            "weight": 0.10,
            "migration_cost_hours": 0.05,  # ~3 min: stateless
        },
    }

    # Default regions for multi-region jobs
    DEFAULT_REGIONS = ["us-west", "us-east", "eu-west"]

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the ingester.

        Args:
            data_dir: Directory for data files (optional)
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.db = get_db()

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        """Parse an ISO timestamp, normalizing naive values to UTC-aware.

        The backtest engine and price index are UTC-aware throughout; a naive
        datetime loaded from a file silently produces degenerate (empty)
        optimizer schedules, so we attach UTC when no offset is present.
        """
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def load_from_csv(self, filepath: Path) -> list[Job]:
        """Load jobs from a CSV file.

        Expected columns: job_id, submit_time, runtime_hours, deadline,
                         power_kw, earliest_start, region_options

        Args:
            filepath: Path to CSV file

        Returns:
            List of Job objects
        """
        jobs = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                region_options = row.get("region_options", "us-west")
                if isinstance(region_options, str):
                    region_options = [r.strip() for r in region_options.split(",")]

                mch = row.get("migration_cost_hours", "")
                job = Job(
                    job_id=row["job_id"],
                    submit_time=self._parse_dt(row["submit_time"]),
                    runtime_hours=float(row["runtime_hours"]),
                    deadline=self._parse_dt(row["deadline"]),
                    power_kw=float(row["power_kw"]),
                    earliest_start=self._parse_dt(row["earliest_start"]),
                    region_options=region_options,
                    priority=int(row.get("priority", 1)),
                    workload_type=row.get("workload_type") or "scheduled_batch",
                    migration_cost_hours=float(mch) if mch not in ("", None) else None,
                )
                jobs.append(job)
        logger.info(f"Loaded {len(jobs)} jobs from {filepath}")
        return jobs

    def load_from_json(self, filepath: Path) -> list[Job]:
        """Load jobs from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List of Job objects
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        jobs = []
        for record in data:
            job = Job(
                job_id=record["job_id"],
                submit_time=self._parse_dt(record["submit_time"]),
                runtime_hours=float(record["runtime_hours"]),
                deadline=self._parse_dt(record["deadline"]),
                power_kw=float(record["power_kw"]),
                earliest_start=self._parse_dt(record["earliest_start"]),
                region_options=record.get("region_options", ["us-west"]),
                priority=record.get("priority", 1),
                workload_type=record.get("workload_type", "scheduled_batch"),
                migration_cost_hours=record.get("migration_cost_hours"),
            )
            jobs.append(job)
        logger.info(f"Loaded {len(jobs)} jobs from {filepath}")
        return jobs

    # Per-workload-type defaults used when customer CSV omits optional columns.
    # power_kw is estimated as gpu_count * 0.4 kW (A100/H100 ≈ 400W TDP).
    CUSTOMER_CSV_DEFAULTS: dict[str, dict] = {
        "training": {
            "gpu_count": 8, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 48.0, "interruptible": True,
            "checkpointable": True, "migration_cost_hours": 0.5,
        },
        "fine_tuning": {
            "gpu_count": 4, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 24.0, "interruptible": True,
            "checkpointable": True, "migration_cost_hours": 0.25,
        },
        "llm_batch_inference": {
            "gpu_count": 4, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 24.0, "interruptible": False,
            "checkpointable": True, "migration_cost_hours": 0.1,
        },
        "data_processing": {
            "gpu_count": 4, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 24.0, "interruptible": True,
            "checkpointable": True, "migration_cost_hours": 0.05,
        },
        "scheduled_batch": {
            "gpu_count": 4, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 24.0, "interruptible": True,
            "checkpointable": True, "migration_cost_hours": 0.1,
        },
        "realtime_inference": {
            "gpu_count": 2, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 0.0, "interruptible": False,
            "checkpointable": False, "migration_cost_hours": None,
        },
        "background_maintenance": {
            "gpu_count": 1, "power_kw_per_gpu": 0.4,
            "max_delay_hours": 168.0, "interruptible": True,
            "checkpointable": True, "migration_cost_hours": 0.05,
        },
    }

    def load_from_customer_csv(
        self,
        filepath: Path,
        default_regions: Optional[list[str]] = None,
    ) -> list[Job]:
        """Load jobs from a simplified customer-facing CSV.

        Required columns:
            job_id, workload_type, submit_time, duration_hours

        Optional columns (workload_type defaults applied when absent):
            gpu_count, deadline, max_delay_hours, allowed_regions,
            forbidden_regions, interruptible, checkpointable, preemptible,
            data_transfer_gb, sla_class, sla_penalty_per_hour, gpu_type,
            power_kw, migration_cost_hours, pue

        This is the recommended ingestion path for customer pilot traces.
        The legacy ``load_from_csv`` requires internal columns (power_kw,
        earliest_start, region_options, runtime_hours) — this method uses
        the public-facing customer schema and derives the remaining fields.

        Args:
            filepath: Path to the customer workload trace CSV.
            default_regions: Regions to allow when ``allowed_regions`` column
                             is absent.  Defaults to us-west, us-east, us-south.
        """
        if default_regions is None:
            default_regions = ["us-west", "us-east", "us-south"]

        import pandas as pd
        df = pd.read_csv(filepath)

        required = {"job_id", "workload_type", "submit_time", "duration_hours"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Customer CSV missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

        jobs: list[Job] = []
        for _, row in df.iterrows():
            wtype = str(row["workload_type"]).strip().lower()
            if wtype not in self.CUSTOMER_CSV_DEFAULTS:
                known = list(self.CUSTOMER_CSV_DEFAULTS)
                raise ValueError(
                    f"Unknown workload_type '{wtype}' in row {row['job_id']!r}. "
                    f"Must be one of: {known}"
                )
            defaults = self.CUSTOMER_CSV_DEFAULTS[wtype]

            submit_time = self._parse_dt(str(row["submit_time"]))
            duration_h = float(row["duration_hours"])

            # gpu_count: CSV column overrides per-workload default
            gpu_count = int(row["gpu_count"]) if "gpu_count" in df.columns and pd.notna(row.get("gpu_count")) else defaults["gpu_count"]

            # power_kw: explicit column or estimate from gpu_count
            if "power_kw" in df.columns and pd.notna(row.get("power_kw")):
                power_kw = float(row["power_kw"])
            else:
                power_kw = gpu_count * defaults["power_kw_per_gpu"]

            # max_delay_hours → deadline and earliest_start
            if "max_delay_hours" in df.columns and pd.notna(row.get("max_delay_hours")):
                max_delay_h: Optional[float] = float(row["max_delay_hours"])
            else:
                max_delay_h = defaults["max_delay_hours"]

            if "deadline" in df.columns and pd.notna(row.get("deadline")):
                deadline = self._parse_dt(str(row["deadline"]))
            elif max_delay_h is not None:
                deadline = submit_time + timedelta(hours=duration_h + max_delay_h)
            else:
                deadline = submit_time + timedelta(hours=duration_h * 3)

            earliest_start = submit_time

            # region options
            if "allowed_regions" in df.columns and pd.notna(row.get("allowed_regions")):
                allowed_str = str(row["allowed_regions"]).strip()
                # Support "|" or ";" as multi-value separators (commas are CSV delimiters)
                sep = "|" if "|" in allowed_str else (";" if ";" in allowed_str else "|")
                parts = [r.strip() for r in allowed_str.replace(";", "|").split("|") if r.strip()]
                allowed = parts if parts else list(default_regions)
            else:
                allowed = list(default_regions)

            forbidden: list[str] = []
            if "forbidden_regions" in df.columns and pd.notna(row.get("forbidden_regions")):
                forb_str = str(row["forbidden_regions"]).strip()
                forbidden = [r.strip() for r in forb_str.replace(";", "|").split("|") if r.strip()]

            # boolean fields
            def _bool(col: str, fallback: bool) -> bool:
                if col in df.columns and pd.notna(row.get(col)):
                    v = row[col]
                    if isinstance(v, bool):
                        return v
                    return str(v).strip().lower() in ("1", "true", "yes")
                return fallback

            interruptible = _bool("interruptible", bool(defaults["interruptible"]))
            checkpointable = _bool("checkpointable", bool(defaults["checkpointable"]))
            preemptible = _bool("preemptible", interruptible)

            mch = defaults["migration_cost_hours"]
            if "migration_cost_hours" in df.columns and pd.notna(row.get("migration_cost_hours")):
                mch = float(row["migration_cost_hours"])

            sla_class = str(row.get("sla_class", "best_effort")).strip() if "sla_class" in df.columns and pd.notna(row.get("sla_class")) else "best_effort"
            sla_penalty = float(row["sla_penalty_per_hour"]) if "sla_penalty_per_hour" in df.columns and pd.notna(row.get("sla_penalty_per_hour")) else 0.0
            data_gb = float(row["data_transfer_gb"]) if "data_transfer_gb" in df.columns and pd.notna(row.get("data_transfer_gb")) else 0.0
            gpu_type = str(row["gpu_type"]).strip() if "gpu_type" in df.columns and pd.notna(row.get("gpu_type")) else None
            pue = float(row["pue"]) if "pue" in df.columns and pd.notna(row.get("pue")) else 1.0

            job = Job(
                job_id=str(row["job_id"]),
                submit_time=submit_time,
                runtime_hours=duration_h,
                deadline=deadline,
                power_kw=power_kw,
                earliest_start=earliest_start,
                region_options=allowed,
                workload_type=wtype,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                interruptible=interruptible,
                preemptible=preemptible,
                checkpointable=checkpointable,
                max_delay_hours=max_delay_h,
                allowed_regions=allowed,
                forbidden_regions=forbidden,
                sla_class=sla_class,
                sla_penalty_per_hour=sla_penalty,
                data_transfer_gb=data_gb,
                pue=pue,
                migration_cost_hours=mch,
            )
            jobs.append(job)

        logger.info(f"Loaded {len(jobs)} jobs from customer CSV {filepath}")
        return jobs

    def load_from_file(self, filepath: Path) -> list[Job]:
        """Load jobs from either JSON or CSV, auto-detecting format.

        For CSV files: if the columns match the customer-facing schema
        (job_id + workload_type + submit_time + duration_hours), uses
        ``load_from_customer_csv``.  Otherwise falls back to the legacy
        ``load_from_csv`` (internal schema).

        For JSON files: uses ``load_from_json``.
        """
        p = Path(filepath)
        if p.suffix.lower() == ".json":
            return self.load_from_json(p)

        # CSV: detect which schema
        import pandas as pd
        header = pd.read_csv(p, nrows=0).columns.tolist()
        customer_required = {"job_id", "workload_type", "submit_time", "duration_hours"}
        if customer_required.issubset(set(header)):
            return self.load_from_customer_csv(p)
        return self.load_from_csv(p)

    def generate_synthetic(
        self,
        start_time: datetime,
        duration_hours: int,
        num_jobs: int,
        regions: Optional[list[str]] = None,
        profile_weights: Optional[dict[str, float]] = None,
        slack_hours_range: tuple[int, int] = (4, 24),
        high_slack_pct: float = 0.6,
        multi_region_pct: float = 0.7,
        seed: Optional[int] = None,
        workload_mix: Optional[str] = None,
        workload_filter: Optional[str] = None,
    ) -> list[Job]:
        """Generate synthetic batch jobs.

        Two mixes available:
          - "legacy" (default): size-based profiles (small/medium/large/xlarge)
            using the function-level slack/multi-region args.
          - "realistic": 7 workload-type profiles (realtime_inference, training,
            fine_tuning, etc) with per-profile slack and multi-region settings
            from WORKLOAD_PROFILES. profile_weights / slack_hours_range /
            high_slack_pct / multi_region_pct are ignored in this mode — the
            profile dict defines them.

        Args:
            workload_mix: None or "legacy" → use JOB_PROFILES.
                          "realistic" → use WORKLOAD_PROFILES.
            workload_filter: When set (realistic mix only), generate ALL jobs of
                          this single workload type — for measuring per-workload
                          savings in isolation.

        Returns:
            List of Job objects
        """
        if seed is not None:
            random.seed(seed)

        regions = regions or self.DEFAULT_REGIONS
        start_floored = start_time.replace(minute=0, second=0, microsecond=0)

        use_workload_mix = workload_mix == "realistic"
        if use_workload_mix:
            profile_dict = self.WORKLOAD_PROFILES
            if workload_filter is not None:
                if workload_filter not in profile_dict:
                    raise ValueError(
                        f"Unknown workload_filter '{workload_filter}'; "
                        f"valid: {list(profile_dict)}"
                    )
                profiles = [workload_filter]
                weights = [1.0]
            else:
                profiles = list(profile_dict.keys())
                weights = [profile_dict[p]["weight"] for p in profiles]
        else:
            profile_dict = self.JOB_PROFILES
            profile_weights = profile_weights or {
                "small": 0.4,
                "medium": 0.35,
                "large": 0.2,
                "xlarge": 0.05,
            }
            profiles = list(profile_weights.keys())
            weights = list(profile_weights.values())

        jobs = []
        for i in range(num_jobs):
            submit_offset = int(random.uniform(0, duration_hours * 0.7))
            submit_time = start_floored + timedelta(hours=submit_offset)

            profile = random.choices(profiles, weights=weights)[0]
            profile_spec = profile_dict[profile]

            power_kw = random.uniform(*profile_spec["power_kw"])
            runtime_hours = random.uniform(*profile_spec["runtime_hours"])
            earliest_start = submit_time + timedelta(hours=int(random.uniform(0, 2)))

            if use_workload_mix:
                # Per-profile slack and multi-region from WORKLOAD_PROFILES
                slack = random.uniform(*profile_spec["slack_hours"])
                profile_multi_pct = profile_spec["multi_region_pct"]
                migration_cost_hours = profile_spec.get("migration_cost_hours")
                workload_type = profile  # use profile name as workload_type
            else:
                # Legacy: global slack/multi-region settings from function args.
                # Legacy jobs do not migrate (preserves pre-migration behavior).
                if random.random() < high_slack_pct:
                    slack = random.uniform(*slack_hours_range)
                else:
                    slack = random.uniform(1, 4)
                profile_multi_pct = multi_region_pct
                migration_cost_hours = None
                workload_type = "scheduled_batch"

            deadline = earliest_start + timedelta(hours=runtime_hours + slack)

            if random.random() < profile_multi_pct:
                job_regions = regions.copy()
            else:
                job_regions = [random.choice(regions)]

            job = Job(
                job_id=f"job-{uuid.uuid4().hex[:8]}",
                submit_time=submit_time,
                runtime_hours=round(runtime_hours, 2),
                deadline=deadline,
                power_kw=round(power_kw, 1),
                earliest_start=earliest_start,
                region_options=job_regions,
                priority=random.randint(1, 5),
                workload_type=workload_type,
                migration_cost_hours=migration_cost_hours,
            )
            jobs.append(job)

        jobs.sort(key=lambda j: j.submit_time)
        logger.info(f"Generated {len(jobs)} synthetic jobs (mix={workload_mix or 'legacy'})")
        return jobs

    def save_to_csv(self, jobs: list[Job], filepath: Path) -> None:
        """Save jobs to a CSV file.

        Args:
            jobs: List of Job objects
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "job_id", "submit_time", "runtime_hours", "deadline",
                    "power_kw", "earliest_start", "region_options", "priority",
                    "workload_type", "migration_cost_hours",
                ]
            )
            writer.writeheader()
            for job in jobs:
                writer.writerow({
                    "job_id": job.job_id,
                    "submit_time": job.submit_time.isoformat(),
                    "runtime_hours": job.runtime_hours,
                    "deadline": job.deadline.isoformat(),
                    "power_kw": job.power_kw,
                    "earliest_start": job.earliest_start.isoformat(),
                    "region_options": ",".join(job.region_options),
                    "priority": job.priority,
                    "workload_type": job.workload_type,
                    "migration_cost_hours": (
                        "" if job.migration_cost_hours is None
                        else job.migration_cost_hours
                    ),
                })
        logger.info(f"Saved {len(jobs)} jobs to {filepath}")

    def save_to_json(self, jobs: list[Job], filepath: Path) -> None:
        """Save jobs to a JSON file.

        Args:
            jobs: List of Job objects
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "job_id": job.job_id,
                "submit_time": job.submit_time.isoformat(),
                "runtime_hours": job.runtime_hours,
                "deadline": job.deadline.isoformat(),
                "power_kw": job.power_kw,
                "earliest_start": job.earliest_start.isoformat(),
                "region_options": job.region_options,
                "priority": job.priority,
                "workload_type": job.workload_type,
                "migration_cost_hours": job.migration_cost_hours,
            }
            for job in jobs
        ]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(jobs)} jobs to {filepath}")

    def save_to_database(self, jobs: list[Job]) -> bool:
        """Save jobs to Supabase.

        Args:
            jobs: List of Job objects

        Returns:
            True if successful
        """
        records = [
            {
                "job_id": j.job_id,
                "submit_time": j.submit_time,
                "runtime_hours": j.runtime_hours,
                "deadline": j.deadline,
                "power_kw": j.power_kw,
                "earliest_start": j.earliest_start,
                "latest_start": j.latest_start,
                "region_options": j.region_options,
                "priority": j.priority,
            }
            for j in jobs
        ]
        return self.db.insert_jobs(records)

    def fetch_jobs(
        self,
        job_ids: Optional[list[str]] = None,
        region: Optional[str] = None,
    ) -> list[Job]:
        """Fetch jobs from the database.

        Args:
            job_ids: Filter by job IDs
            region: Filter by region availability

        Returns:
            List of Job objects
        """
        records = self.db.get_jobs(job_ids, region)
        return [
            Job(
                job_id=r["job_id"],
                submit_time=datetime.fromisoformat(r["submit_time"].replace("Z", "+00:00")),
                runtime_hours=float(r["runtime_hours"]),
                deadline=datetime.fromisoformat(r["deadline"].replace("Z", "+00:00")),
                power_kw=float(r["power_kw"]),
                earliest_start=datetime.fromisoformat(r["earliest_start"].replace("Z", "+00:00")),
                region_options=r["region_options"],
                priority=r.get("priority", 1),
            )
            for r in records
        ]

    def validate_jobs(self, jobs: list[Job]) -> tuple[list[Job], list[tuple[Job, str]]]:
        """Validate jobs and return valid jobs and errors.

        Args:
            jobs: List of Job objects to validate

        Returns:
            Tuple of (valid_jobs, list of (invalid_job, error_message))
        """
        valid = []
        errors = []

        for job in jobs:
            # Check deadline is after earliest start + runtime
            min_finish = job.earliest_start + timedelta(hours=job.runtime_hours)
            if job.deadline < min_finish:
                errors.append((job, "Deadline before minimum finish time"))
                continue

            # Check power is positive
            if job.power_kw <= 0:
                errors.append((job, "Power must be positive"))
                continue

            # Check runtime is positive
            if job.runtime_hours <= 0:
                errors.append((job, "Runtime must be positive"))
                continue

            # Check region options not empty
            if not job.region_options:
                errors.append((job, "Must have at least one region option"))
                continue

            valid.append(job)

        if errors:
            logger.warning(f"Validation: {len(errors)} invalid jobs out of {len(jobs)}")

        return valid, errors
