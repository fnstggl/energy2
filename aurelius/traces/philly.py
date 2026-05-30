"""Microsoft Philly trace ingester — GPU training-job scheduling backtest.

Normalizes the Philly multi-tenant GPU training trace
(https://github.com/msr-fiddle/philly-traces) into ``NormalizedGPUJob`` (jobs) +
``GPUNode`` (machine inventory) for the bin-packing / scheduling backtest in
``aurelius/traces/gpu_packing.py`` (reused from the Alibaba GPU work).

Discovered schema (verified against the official analysis notebook):

``cluster_job_log`` is a JSON **list** of job objects::

    {"status": "Pass"|"Killed"|"Failed",
     "vc": "<virtual-cluster-hash>",
     "jobid": "<hash>",
     "submitted_time": "YYYY-MM-DD HH:MM:SS",
     "user": "<user-hash>",
     "attempts": [{"start_time": "...", "end_time": "...",
                   "detail": [{"ip": "<machine>", "gpus": ["gpu0", ...]}, ...]}, ...]}

``cluster_machine_list`` is a CSV: ``machineId,number of GPUs,single GPU mem``.

Honest mapping / limits:
- GPU count per job = ``sum(len(detail.gpus))`` over the **first** attempt's
  ``detail`` (the official ``num_gpus`` definition). Whole-GPU jobs (no
  fractional sharing) → ``gpu_milli=None``.
- times are ``%Y-%m-%d %H:%M:%S`` strings parsed to UTC seconds; empty strings
  (e.g. a job that never started, or an open end_time) → ``None``.
- ``queue_wait_s = first_attempt.start − submitted_time``.
- ``is_failed = status in {Failed, Killed}`` (neither completed successfully —
  Killed is preemption/user-kill; both are excluded from goodput).
- **No explicit GPU model** in the job log; the machine list gives only per-GPU
  memory → ``gpu_type`` is inferred as a ``GPU-<mem>`` label on nodes, and
  ``None`` on jobs. **No CPU/host-memory request, no deadline** in the job log →
  ``cpu_milli`` / ``memory_mib`` / ``deadline_s`` = None. The virtual-cluster
  (`vc`) hash is available but not part of the packing decision.

Philly public data is a public research dataset, **not customer telemetry**.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from typing import Optional

from .gpu_packing import GPUNode, effective_gpu, job_work
from .schema import NormalizedGPUJob, TraceSchemaError, percentile

DATASET_NAME = "philly"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
WORKLOAD_TYPE = "training"

# Default download (a single ~1 GB git-LFS tarball; see ingest_philly.py).
TRACE_TARBALL_URL = (
    "https://media.githubusercontent.com/media/msr-fiddle/philly-traces/"
    "master/trace-data.tar.gz")
JOB_LOG_NAME = "cluster_job_log"
MACHINE_LIST_NAME = "cluster_machine_list"

# Machine-list column candidates (tolerant to header spelling).
_MACHINE_ID_KEYS = ("machineId", "machine_id", "machineID", "machine")
_GPU_COUNT_KEYS = ("number of GPUs", "number_of_gpus", "num_gpus", "gpus", "gpu")
_GPU_MEM_KEYS = ("single GPU mem", "single_gpu_mem", "gpu_mem", "mem")


def parse_time_s(raw: Optional[str]) -> Optional[float]:
    """Parse a Philly ``%Y-%m-%d %H:%M:%S`` string to UTC seconds, else None."""
    if not raw or not str(raw).strip() or str(raw).strip().lower() in ("none", "null"):
        return None
    try:
        dt = datetime.strptime(str(raw).strip(), DATE_FORMAT).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _attempt_gpu_count(attempt: dict) -> int:
    return sum(len(d.get("gpus", []) or []) for d in attempt.get("detail", []) or [])


def _attempt_placement(attempt: dict) -> tuple[Optional[str], Optional[str]]:
    details = attempt.get("detail", []) or []
    ips = [str(d.get("ip")) for d in details if d.get("ip")]
    gpus = [g for d in details for g in (d.get("gpus", []) or [])]
    return (",".join(ips) or None, ",".join(str(g) for g in gpus) or None)


def normalize_job(job: dict, index: int) -> NormalizedGPUJob:
    status = (job.get("status") or "").strip() or None
    submit = parse_time_s(job.get("submitted_time"))
    attempts = job.get("attempts") or []

    gpu_count = 0
    start = end = None
    placement_nodes = placement_gpus = None
    if attempts:
        first = attempts[0]
        gpu_count = _attempt_gpu_count(first)
        start = parse_time_s(first.get("start_time"))
        # end of the LAST attempt that has an end_time
        for att in reversed(attempts):
            e = parse_time_s(att.get("end_time"))
            if e is not None:
                end = e
                break
        placement_nodes, placement_gpus = _attempt_placement(first)

    duration = end - start if (start is not None and end is not None and end > start) else None
    queue_wait = (start - submit) if (start is not None and submit is not None
                                      and start >= submit) else None
    is_failed = status in ("Failed", "Killed")

    job_obj = NormalizedGPUJob(
        job_id=str(job.get("jobid") or f"philly-{index}"),
        submit_time_s=submit,
        start_time_s=start,
        end_time_s=end,
        duration_s=duration,
        gpu_count=gpu_count,
        gpu_type=None,                 # no GPU model in the job log
        gpu_memory_gb=None,
        status=status,
        user_or_group=(job.get("user") or None),
        workload_type=WORKLOAD_TYPE,
        priority=None,
        placement_nodes=placement_nodes,
        placement_gpus=placement_gpus,
        is_failed=is_failed,
        deadline_s=None,
        cpu_milli=None,
        memory_mib=None,
        gpu_milli=None,                # Philly is whole-GPU (no sharing)
        queue_wait_s=queue_wait,
    )
    d = job_obj.to_dict()
    d["token_equivalent_work"] = job_work(job_obj)  # gpu_seconds_work
    return NormalizedGPUJob.from_dict(d)


def load_jobs(
    path: str,
    *,
    sample_size: Optional[int] = None,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    include_failed: bool = True,
    seed: int = 0,
) -> list[NormalizedGPUJob]:
    """Load + normalize the Philly ``cluster_job_log`` JSON list."""
    import random

    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise TraceSchemaError(
            f"{DATASET_NAME}: cluster_job_log must be a JSON list of jobs, "
            f"got {type(raw).__name__}")
    if raw and not isinstance(raw[0], dict):
        raise TraceSchemaError(f"{DATASET_NAME}: job entries must be objects")
    # schema guard: required keys on the first job
    if raw:
        missing = [k for k in ("status", "jobid", "submitted_time", "attempts")
                   if k not in raw[0]]
        if missing:
            raise TraceSchemaError(
                f"{DATASET_NAME}: job log missing required field(s) {missing}")

    jobs = [normalize_job(j, i) for i, j in enumerate(raw)]
    jobs.sort(key=lambda j: (j.submit_time_s if j.submit_time_s is not None else 0.0,
                             j.job_id))

    if start_s is not None or duration_s is not None:
        base = next((j.submit_time_s for j in jobs if j.submit_time_s is not None), 0.0)
        lo = base + start_s if start_s is not None else float("-inf")
        hi = base + (start_s or 0.0) + duration_s if duration_s is not None else float("inf")
        jobs = [j for j in jobs if j.submit_time_s is not None and lo <= j.submit_time_s < hi]

    if not include_failed:
        jobs = [j for j in jobs if not j.is_failed]

    if sample_size is not None and 0 <= sample_size < len(jobs):
        rng = random.Random(seed)
        jobs = rng.sample(jobs, sample_size)
        jobs.sort(key=lambda j: (j.submit_time_s if j.submit_time_s is not None
                                 else 0.0, j.job_id))
    return jobs


def _pick(row: dict, keys) -> Optional[str]:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    # case-insensitive contains fallback
    low = {str(c).strip().lower(): c for c in row}
    for k in keys:
        if k.lower() in low:
            return row[low[k.lower()]]
    return None


def _mem_to_model(mem: Optional[str]) -> str:
    if not mem:
        return "philly-gpu"
    digits = "".join(ch for ch in str(mem) if ch.isdigit())
    return f"GPU-{digits}GB" if digits else "philly-gpu"


def load_machines(path: str) -> list[GPUNode]:
    """Load the Philly ``cluster_machine_list`` CSV → GPU node fleet."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise TraceSchemaError(f"{DATASET_NAME}: empty machine list")
        rows = list(reader)
    if not rows:
        raise TraceSchemaError(f"{DATASET_NAME}: machine list has no rows")
    if _pick(rows[0], _GPU_COUNT_KEYS) is None:
        raise TraceSchemaError(
            f"{DATASET_NAME}: machine list missing a GPU-count column "
            f"(looked for {_GPU_COUNT_KEYS}); header={reader.fieldnames}")

    nodes = []
    for i, r in enumerate(rows):
        gpu = _pick(r, _GPU_COUNT_KEYS)
        try:
            gpu_count = int(float(gpu)) if gpu is not None else 0
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count <= 0:
            continue
        mid = _pick(r, _MACHINE_ID_KEYS) or f"machine-{i}"
        model = _mem_to_model(_pick(r, _GPU_MEM_KEYS))
        nodes.append(GPUNode(node_id=str(mid), gpu_count=gpu_count,
                             gpu_model=model, cpu_milli=0, memory_mib=0))
    return nodes


def analyze_attempts(path: str) -> dict:
    """Trace-observed retry / failure / wasted-GPU-hours stats from the RAW job
    log attempt history. This is descriptive (the scheduler does not re-simulate
    failures); it lets the report state whether the trace itself is retry-heavy.
    """
    with open(path) as fh:
        raw = json.load(fh)
    total = len(raw)
    multi_attempt = 0
    total_retries = 0
    wasted_gpu_seconds = 0.0
    failed = killed = passed = 0
    for job in raw:
        attempts = job.get("attempts") or []
        if len(attempts) > 1:
            multi_attempt += 1
            total_retries += len(attempts) - 1
            # non-final attempts = wasted work (preempted/failed runs)
            for att in attempts[:-1]:
                s = parse_time_s(att.get("start_time"))
                e = parse_time_s(att.get("end_time"))
                if s is not None and e is not None and e > s:
                    wasted_gpu_seconds += _attempt_gpu_count(att) * (e - s)
        st = job.get("status")
        if st == "Failed":
            failed += 1
        elif st == "Killed":
            killed += 1
        elif st == "Pass":
            passed += 1
    return {
        "jobs": total,
        "passed": passed, "failed": failed, "killed": killed,
        "multi_attempt_jobs": multi_attempt,
        "total_retries": total_retries,
        "retry_rate_pct": round(100.0 * total_retries / total, 3) if total else 0.0,
        "wasted_gpu_hours_from_retries": round(wasted_gpu_seconds / 3600.0, 3),
    }


def summarize_jobs(jobs, nodes=None) -> dict:
    """Descriptive stats the ingest script prints (JSON-serialisable dict)."""
    if not jobs:
        return {"job_count": 0}
    subs = [j.submit_time_s for j in jobs if j.submit_time_s is not None]
    t0, t1 = (min(subs), max(subs)) if subs else (0.0, 0.0)
    durations = [j.duration_s for j in jobs if j.duration_s and j.duration_s > 0]
    waits = [j.queue_wait_s for j in jobs if j.queue_wait_s is not None]

    status_dist: dict = {}
    gpu_count_dist: dict = {}
    vc_users: set = set()
    for j in jobs:
        status_dist[j.status or "None"] = status_dist.get(j.status or "None", 0) + 1
        gpu_count_dist[j.gpu_count] = gpu_count_dist.get(j.gpu_count, 0) + 1
        if j.user_or_group:
            vc_users.add(j.user_or_group)

    gpu_jobs = [j for j in jobs if effective_gpu(j) > 0]
    out = {
        "job_count": len(jobs),
        "gpu_job_count": len(gpu_jobs),
        "no_gpu_or_unscheduled": len(jobs) - len(gpu_jobs),
        "distinct_users": len(vc_users),
        "time_start_s": t0, "time_end_s": t1, "duration_s": max(0.0, t1 - t0),
        "status_distribution": dict(sorted(status_dist.items(), key=lambda kv: str(kv[0]))),
        "gpu_count_distribution": dict(sorted(gpu_count_dist.items())),
        "failed_or_killed": sum(1 for j in jobs if j.is_failed),
        "job_duration_s_p50": percentile(durations, 50) if durations else 0.0,
        "job_duration_s_p95": percentile(durations, 95) if durations else 0.0,
        "job_duration_s_p99": percentile(durations, 99) if durations else 0.0,
        "queue_wait_s_p50": percentile(waits, 50) if waits else None,
        "queue_wait_s_p95": percentile(waits, 95) if waits else None,
        "queue_wait_s_p99": percentile(waits, 99) if waits else None,
        "total_gpu_demand": round(sum(effective_gpu(j) for j in gpu_jobs), 2),
        "gpu_utilization_samples": 0,  # not parsed in this PR (separate CSV)
        "missing_fields": [
            "gpu_type_model", "cpu_host_memory_request", "deadline",
            "per_job_gpu_utilization",
        ],
    }
    if nodes:
        models: dict = {}
        for n in nodes:
            models[n.gpu_model] = models.get(n.gpu_model, 0) + n.gpu_count
        out["fleet_node_count"] = len(nodes)
        out["fleet_gpu_count"] = sum(n.gpu_count for n in nodes)
        out["fleet_gpu_by_model"] = dict(sorted(models.items()))
        cap = out["fleet_gpu_count"] or 1
        out["gpu_demand_to_capacity_ratio"] = round(out["total_gpu_demand"] / cap, 4)
    return out
