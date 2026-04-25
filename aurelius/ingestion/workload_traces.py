"""Workload trace ingestion for Aurelius.

Loads client-format workload CSVs and produces validated Job objects.
Applies workload_type defaults for interruptibility, SLA class, and delay window
but always lets explicit job-level fields override defaults.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..models import (
    WORKLOAD_DEFAULT_INTERRUPTIBLE,
    WORKLOAD_DEFAULT_MAX_DELAY_HOURS,
    WORKLOAD_DEFAULT_SLA_CLASS,
    Job,
)

logger = logging.getLogger(__name__)

# Required columns in the CSV
_REQUIRED_COLUMNS = {
    "job_id",
    "submit_time",
    "runtime_hours",
    "deadline",
    "power_kw",
    "earliest_start",
    "region_options",
}

# Supported GPU identifiers
_KNOWN_GPU_TYPES = {"a100", "h100", "v100", "t4", "a10g", "l4", "p100", "cpu"}


def _parse_dt(value: str, field_name: str) -> datetime:
    """Parse ISO-8601 datetime string. Raises ValueError on failure."""
    value = value.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse {field_name}={value!r}; expected ISO-8601 format (e.g. 2024-01-01T08:00:00)"
    )


def _parse_region_options(value: str) -> list[str]:
    """Parse region_options from comma-separated string or JSON array."""
    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(r).strip() for r in parsed if str(r).strip()]
        except json.JSONDecodeError:
            pass
    return [r.strip() for r in value.split(",") if r.strip()]


def _parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated or JSON list of strings."""
    if not value or value.strip() == "":
        return []
    return _parse_region_options(value)


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse boolean from string."""
    return value.strip().lower() in ("true", "1", "yes") if value.strip() else default


def _apply_workload_defaults(row: dict[str, Any], job: Job) -> None:
    """Apply workload_type-derived defaults to a Job, but only when the field
    was not explicitly provided in the CSV row."""

    wt = job.workload_type
    if wt not in WORKLOAD_DEFAULT_INTERRUPTIBLE:
        return

    # interruptible: apply default only if not in CSV
    if "interruptible" not in row or row["interruptible"].strip() == "":
        job.interruptible = WORKLOAD_DEFAULT_INTERRUPTIBLE[wt]
        # fine_tuning/training become interruptible when checkpointable
        if wt in ("fine_tuning", "training") and job.checkpointable:
            job.interruptible = True

    # preemptible: default True only for llm_batch_inference
    if "preemptible" not in row or row["preemptible"].strip() == "":
        job.preemptible = wt == "llm_batch_inference"

    # sla_class: apply default only if not in CSV
    if "sla_class" not in row or row["sla_class"].strip() == "":
        job.sla_class = WORKLOAD_DEFAULT_SLA_CLASS.get(wt, "best_effort")

    # max_delay_hours: apply default if not in CSV and no explicit deadline offset
    if "max_delay_hours" not in row or row["max_delay_hours"].strip() == "":
        default_delay = WORKLOAD_DEFAULT_MAX_DELAY_HOURS.get(wt, 24.0)
        if job.max_delay_hours is None:
            job.max_delay_hours = default_delay


def _row_to_job(row: dict[str, str], row_num: int) -> Job:
    """Convert a CSV row dict to a Job. Raises ValueError on invalid data."""

    # Required fields
    job_id = row.get("job_id", "").strip()
    if not job_id:
        raise ValueError(f"Row {row_num}: job_id is required")

    submit_time = _parse_dt(row["submit_time"], "submit_time")
    runtime_hours = float(row["runtime_hours"])
    if runtime_hours <= 0:
        raise ValueError(f"Row {row_num} ({job_id}): runtime_hours must be > 0")

    deadline = _parse_dt(row["deadline"], "deadline")
    power_kw = float(row["power_kw"])
    if power_kw <= 0:
        raise ValueError(f"Row {row_num} ({job_id}): power_kw must be > 0")

    earliest_start = _parse_dt(row["earliest_start"], "earliest_start")
    region_options = _parse_region_options(row["region_options"])
    if not region_options:
        raise ValueError(f"Row {row_num} ({job_id}): region_options must be non-empty")

    # Optional fields with defaults
    priority = int(row["priority"]) if row.get("priority", "").strip() else 1
    workload_type = row.get("workload_type", "").strip() or "scheduled_batch"
    gpu_type_raw = row.get("gpu_type", "").strip() or None
    gpu_type = gpu_type_raw.lower() if gpu_type_raw else None
    gpu_count = int(row["gpu_count"]) if row.get("gpu_count", "").strip() else 0
    sla_penalty_per_hour = float(row["sla_penalty_per_hour"]) if row.get("sla_penalty_per_hour", "").strip() else 0.0
    sla_class = row.get("sla_class", "").strip() or "best_effort"
    data_transfer_gb = float(row["data_transfer_gb"]) if row.get("data_transfer_gb", "").strip() else 0.0
    pue = float(row["pue"]) if row.get("pue", "").strip() else 1.0
    if pue < 1.0:
        raise ValueError(f"Row {row_num} ({job_id}): pue must be >= 1.0 (got {pue})")

    interruptible = _parse_bool(row.get("interruptible", ""), default=False)
    preemptible = _parse_bool(row.get("preemptible", ""), default=False)
    checkpointable = _parse_bool(row.get("checkpointable", ""), default=False)

    max_delay_str = row.get("max_delay_hours", "").strip()
    max_delay_hours = float(max_delay_str) if max_delay_str else None

    allowed_regions = _parse_str_list(row.get("allowed_regions", ""))
    forbidden_regions = _parse_str_list(row.get("forbidden_regions", ""))

    # Validate deadline is after earliest_start + runtime_hours
    if deadline < earliest_start + timedelta(hours=runtime_hours):
        raise ValueError(
            f"Row {row_num} ({job_id}): deadline {deadline} is before "
            f"earliest_start + runtime_hours ({earliest_start + timedelta(hours=runtime_hours)})"
        )

    job = Job(
        job_id=job_id,
        submit_time=submit_time,
        runtime_hours=runtime_hours,
        deadline=deadline,
        power_kw=power_kw,
        earliest_start=earliest_start,
        region_options=region_options,
        priority=priority,
        workload_type=workload_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        sla_penalty_per_hour=sla_penalty_per_hour,
        sla_class=sla_class,
        data_transfer_gb=data_transfer_gb,
        pue=pue,
        interruptible=interruptible,
        preemptible=preemptible,
        checkpointable=checkpointable,
        max_delay_hours=max_delay_hours,
        allowed_regions=allowed_regions,
        forbidden_regions=forbidden_regions,
    )

    # Apply workload_type defaults for fields not explicitly in row
    _apply_workload_defaults(row, job)

    return job


def load_workload_csv(path: str | Path) -> list[Job]:
    """Load a workload trace CSV and return a list of validated Job objects.

    Required CSV columns:
        job_id, submit_time, runtime_hours, deadline, power_kw,
        earliest_start, region_options

    Optional CSV columns (applied after workload_type defaults):
        priority, workload_type, gpu_type, gpu_count, sla_penalty_per_hour,
        sla_class, data_transfer_gb, pue, interruptible, preemptible,
        checkpointable, max_delay_hours, allowed_regions, forbidden_regions

    Datetime format: ISO-8601 (e.g. "2024-01-15T08:00:00").
    region_options: comma-separated or JSON array (e.g. "us-west,us-east").
    allowed_regions / forbidden_regions: same format.
    interruptible / preemptible / checkpointable: "true"/"false" or "1"/"0".

    Args:
        path: Path to the workload CSV file.

    Returns:
        List of Job objects. Rows with parse errors are skipped and logged.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workload CSV not found: {path}")

    jobs: list[Job] = []
    errors: list[str] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header: {path}")

        columns = set(reader.fieldnames)
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            try:
                job = _row_to_job(row, row_num)
                jobs.append(job)
            except (ValueError, KeyError, TypeError) as e:
                msg = f"Row {row_num}: {e}"
                errors.append(msg)
                logger.warning("Skipping workload row — %s", msg)

    if errors:
        logger.warning(
            "Loaded %d jobs from %s (%d rows skipped due to errors)",
            len(jobs), path, len(errors),
        )
    else:
        logger.info("Loaded %d jobs from %s", len(jobs), path)

    return jobs
