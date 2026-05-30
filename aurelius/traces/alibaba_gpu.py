"""Alibaba cluster-trace-gpu-v2023 ingester — fragmentation / packing backtest.

Normalizes the Alibaba GPU cluster trace
(https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023)
into ``NormalizedGPUJob`` (pods) + ``GPUNode`` (fleet inventory) for the
bin-packing backtest in ``aurelius/traces/gpu_packing.py``.

Discovered schema (verified against the raw ``csv/`` files):

Pod table (``openb_pod_list_*.csv``)::

    name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,
    creation_time,deletion_time,scheduled_time

Node table (``openb_node_list_*.csv``)::

    sn,cpu_milli,memory_mib,gpu,model

Units / honest mapping:
- ``gpu_milli`` = thousandths of a GPU (sharing request); ``num_gpu`` = whole
  GPUs (0–8). ``gpu_spec`` = required GPU type or empty (empty in the default
  pod list → ``gpu_type=None``).
- times are seconds from trace start. ``duration_s = deletion − creation``.
- ``qos`` (LS/BE/Burstable) → ``priority``/``workload_type`` (no real user id).
- ``pod_phase`` (Running/Succeeded/Pending/Failed) → ``status``;
  ``is_failed = (phase == Failed)``.

What the trace does **NOT** provide (stated, not invented):
- **no GPU utilization time-series** → ``NormalizedGPUUtilizationSample`` list is
  empty for this dataset.
- **no GPU-memory column** → ``gpu_memory_gb=None`` (``gpu_milli`` is compute
  share, not memory).
- **no per-pod node placement** in the default pod list → ``placement_nodes`` /
  ``placement_gpus`` = None (placement is what the backtest computes).
- **no explicit deadline** → ``deadline_s=None``.

Alibaba public data is a public dataset, **not customer telemetry**.
"""

from __future__ import annotations

import csv
from typing import Optional

from .gpu_packing import GPUNode, effective_gpu, job_work
from .schema import NormalizedGPUJob, percentile, validate_columns

# --- Pod (job) columns -------------------------------------------------------
P_NAME = "name"
P_CPU = "cpu_milli"
P_MEM = "memory_mib"
P_NUM_GPU = "num_gpu"
P_GPU_MILLI = "gpu_milli"
P_GPU_SPEC = "gpu_spec"
P_QOS = "qos"
P_PHASE = "pod_phase"
P_CREATION = "creation_time"
P_DELETION = "deletion_time"
P_SCHEDULED = "scheduled_time"

POD_REQUIRED = (P_NAME, P_NUM_GPU, P_GPU_MILLI, P_PHASE, P_CREATION)

# --- Node columns ------------------------------------------------------------
N_SN = "sn"
N_CPU = "cpu_milli"
N_MEM = "memory_mib"
N_GPU = "gpu"
N_MODEL = "model"

NODE_REQUIRED = (N_SN, N_GPU, N_MODEL)

DATASET_NAME = "alibaba_gpu"
_BASE = ("https://github.com/alibaba/clusterdata/raw/refs/heads/master/"
         "cluster-trace-gpu-v2023/csv")
POD_URLS = {
    "default": f"{_BASE}/openb_pod_list_default.csv",
    "gpuspec33": f"{_BASE}/openb_pod_list_gpuspec33.csv",
}
NODE_URLS = {
    "gpu": f"{_BASE}/openb_node_list_gpu_node.csv",
    "all": f"{_BASE}/openb_node_list_all_node.csv",
}
DEFAULT_POD_URL = POD_URLS["default"]
DEFAULT_NODE_URL = NODE_URLS["gpu"]


def _i(v: Optional[str]) -> int:
    if v is None or str(v).strip() == "":
        return 0
    return int(float(v))


def _f_or_none(v: Optional[str]) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_pod_row(row: dict, index: int) -> NormalizedGPUJob:
    num_gpu = _i(row.get(P_NUM_GPU))
    gpu_milli = _i(row.get(P_GPU_MILLI))
    creation = _f_or_none(row.get(P_CREATION))
    deletion = _f_or_none(row.get(P_DELETION))
    scheduled = _f_or_none(row.get(P_SCHEDULED))
    phase = (row.get(P_PHASE) or "").strip() or None
    spec = (row.get(P_GPU_SPEC) or "").strip()
    qos = (row.get(P_QOS) or "").strip() or None

    start = scheduled if (scheduled is not None and scheduled > 0) else creation
    duration = None
    if deletion is not None and start is not None and deletion > start:
        duration = deletion - start
    elif deletion is not None and creation is not None and deletion > creation:
        duration = deletion - creation

    job = NormalizedGPUJob(
        job_id=row.get(P_NAME) or f"alibaba-pod-{index}",
        submit_time_s=creation,
        start_time_s=scheduled,
        end_time_s=deletion,
        duration_s=duration,
        gpu_count=num_gpu,
        gpu_type=spec or None,
        gpu_memory_gb=None,            # no GPU-memory column in the trace
        status=phase,
        user_or_group=None,            # no user/group column
        workload_type=qos,
        priority=qos,
        placement_nodes=None,          # not in default pod list
        placement_gpus=None,
        is_failed=(phase == "Failed"),
        deadline_s=None,
        cpu_milli=_i(row.get(P_CPU)),
        memory_mib=_i(row.get(P_MEM)),
        gpu_milli=gpu_milli,
    )
    return _with_work(job)


def _with_work(job: NormalizedGPUJob) -> NormalizedGPUJob:
    # attach token_equivalent_work (frozen dataclass → rebuild via dict)
    d = job.to_dict()
    d["token_equivalent_work"] = job_work(job)
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
    """Load + normalize the Alibaba pod list. Raises ``TraceSchemaError`` on a
    missing required column."""
    import random

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        validate_columns(reader.fieldnames, POD_REQUIRED, DATASET_NAME)
        jobs = [normalize_pod_row(r, i) for i, r in enumerate(reader)]

    jobs.sort(key=lambda j: (j.submit_time_s if j.submit_time_s is not None
                             else 0.0, j.job_id))

    if start_s is not None or duration_s is not None:
        lo = start_s if start_s is not None else float("-inf")
        hi = ((start_s or 0.0) + duration_s) if duration_s is not None else float("inf")
        jobs = [j for j in jobs if j.submit_time_s is not None
                and lo <= j.submit_time_s < hi]

    if not include_failed:
        jobs = [j for j in jobs if not j.is_failed]

    if sample_size is not None and 0 <= sample_size < len(jobs):
        rng = random.Random(seed)
        jobs = rng.sample(jobs, sample_size)
        jobs.sort(key=lambda j: (j.submit_time_s if j.submit_time_s is not None
                                 else 0.0, j.job_id))
    return jobs


def load_nodes(path: str) -> list[GPUNode]:
    """Load the Alibaba node inventory (GPU nodes only: gpu>0)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        validate_columns(reader.fieldnames, NODE_REQUIRED, DATASET_NAME)
        nodes = []
        for r in reader:
            gpu = _i(r.get(N_GPU))
            if gpu <= 0:
                continue  # GPU packing fleet = GPU nodes only
            nodes.append(GPUNode(
                node_id=r.get(N_SN) or f"node-{len(nodes)}",
                gpu_count=gpu,
                gpu_model=(r.get(N_MODEL) or "unknown").strip() or "unknown",
                cpu_milli=_i(r.get(N_CPU)),
                memory_mib=_i(r.get(N_MEM)),
            ))
    return nodes


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_jobs(jobs, nodes=None) -> dict:
    """Descriptive stats the ingest script prints (dict, JSON-serialisable)."""
    if not jobs:
        return {"job_count": 0}
    subs = [j.submit_time_s for j in jobs if j.submit_time_s is not None]
    t0, t1 = (min(subs), max(subs)) if subs else (0.0, 0.0)
    durations = [j.duration_s for j in jobs if j.duration_s and j.duration_s > 0]
    waits = [(j.start_time_s - j.submit_time_s) for j in jobs
             if j.start_time_s is not None and j.submit_time_s is not None
             and j.start_time_s >= j.submit_time_s]

    status_dist: dict = {}
    gpu_count_dist: dict = {}
    gpu_type_dist: dict = {}
    for j in jobs:
        status_dist[j.status or "None"] = status_dist.get(j.status or "None", 0) + 1
        gpu_count_dist[j.gpu_count] = gpu_count_dist.get(j.gpu_count, 0) + 1
        gpu_type_dist[j.gpu_type or "any"] = gpu_type_dist.get(j.gpu_type or "any", 0) + 1

    gpu_jobs = [j for j in jobs if effective_gpu(j) > 0]
    total_gpu_demand = sum(effective_gpu(j) for j in gpu_jobs)
    out = {
        "job_count": len(jobs),
        "gpu_job_count": len(gpu_jobs),
        "cpu_only_count": len(jobs) - len(gpu_jobs),
        "time_start_s": t0,
        "time_end_s": t1,
        "duration_s": max(0.0, t1 - t0),
        "status_distribution": dict(sorted(status_dist.items(), key=lambda kv: str(kv[0]))),
        "gpu_count_distribution": dict(sorted(gpu_count_dist.items())),
        "gpu_type_distribution": dict(sorted(gpu_type_dist.items())),
        "failed_jobs": sum(1 for j in jobs if j.is_failed),
        "job_duration_s_p50": percentile(durations, 50) if durations else 0.0,
        "job_duration_s_p95": percentile(durations, 95) if durations else 0.0,
        "job_duration_s_p99": percentile(durations, 99) if durations else 0.0,
        "queue_wait_s_p50": percentile(waits, 50) if waits else None,
        "queue_wait_s_p95": percentile(waits, 95) if waits else None,
        "queue_wait_s_p99": percentile(waits, 99) if waits else None,
        "total_gpu_demand_effective": round(total_gpu_demand, 3),
        "gpu_utilization_samples": 0,  # Alibaba v2023 has no utilization series
        "missing_fields": [
            "gpu_utilization_timeseries", "gpu_memory_gb", "per_pod_node_placement",
            "deadline", "user_or_group",
        ],
    }
    if nodes:
        fleet_models: dict = {}
        for n in nodes:
            fleet_models[n.gpu_model] = fleet_models.get(n.gpu_model, 0) + n.gpu_count
        out["fleet_node_count"] = len(nodes)
        out["fleet_gpu_count"] = sum(n.gpu_count for n in nodes)
        out["fleet_gpu_by_model"] = dict(sorted(fleet_models.items()))
        # fragmentation-relevant: demand vs capacity ratio
        cap = out["fleet_gpu_count"] or 1
        out["gpu_demand_to_capacity_ratio"] = round(total_gpu_demand / cap, 4)
    return out
