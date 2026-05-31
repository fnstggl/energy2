"""MIT Supercloud Dataset (Samsi et al., HPEC 2021) — read-only ingestion.

Loads anonymized Slurm scheduler logs, optional GPU / CPU / node
utilization time series, and the labelled-DNN-jobs mapping from the
MIT Supercloud Dataset published by the MIT-AI-Accelerator team.

Source:

  * Repo (notebooks + helper scripts):
    https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset
  * Raw data (≈ 1 TB compressed, NOT on GitHub):
    https://dcc.mit.edu/data
  * Paper: Samsi et al., HPEC 2021, arxiv.org/abs/2108.02037

Discovered file layout (per the README + intro notebook):

  * ``scheduler-log.csv``   — Slurm accounting log (job_id, tres_req,
                              submit / start / end timestamps, …)
  * ``labelled_jobids.csv`` — DNN-model label per job_id (3,425 jobs)
  * ``tres-mapping.txt``    — integer ↔ resource (cpu / mem / gpu:tesla
                              / gpu:volta / …) lookup
  * ``node-data.csv``       — 5-min per-node snapshots
  * ``cpu/<NN>/<job_id>.csv``    — per-job CPU time series (10 s)
  * ``cpu/<NN>/<job_id>-summary.csv`` — per-job CPU summary
  * ``gpu/<NN>/<job_id>.csv``    — per-job GPU time series
                                    (nvidia-smi, 100 ms)

Honesty rules (asserted by tests):

- Missing fields are preserved as ``None`` — never zero-filled
  (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1).
- Joins between scheduler / labels / utilization files are
  **explicit** — `compute_join_quality` reports which job IDs have
  matching label / GPU / CPU rows and which do not.
- This module is read-only; no production mutation, no robust energy
  engine change, no serving frontier change, no ML training.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from .schema import NormalizedGPUJob, TraceSchemaError, percentile

DATASET_NAME = "mit_supercloud"
PAPER_URL = "https://arxiv.org/abs/2108.02037"
DCC_DATA_URL = "https://dcc.mit.edu/data"
REPO_URL = "https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset"

# Canonical file names per the README + intro notebook. (The notebook
# uses ``scheduler-log.csv``; the README also mentions ``slurm-log.csv``
# and ``labelled-slurm-log.csv``. The loader tolerates both.)
SCHEDULER_LOG_FILES = ("scheduler-log.csv", "slurm-log.csv",
                       "labelled-slurm-log.csv")
LABELLED_JOBIDS_FILE = "labelled_jobids.csv"
TRES_MAPPING_FILE = "tres-mapping.txt"
NODE_DATA_FILE = "node-data.csv"
CPU_DIR = "cpu"
GPU_DIR = "gpu"

# TRES integer -> resource label (per the README).
DEFAULT_TRES_MAPPING: dict = {
    1: "cpu", 2: "mem", 3: "energy", 4: "node", 5: "billing",
    6: "fs", 7: "vmem", 8: "pages",
    1001: "gpu:tesla", 1002: "gpu:volta",
}

# Slurm ``state`` is published as an integer in MIT's slurm-log.csv
# (PENDING=0 / RUNNING=1 / SUSPENDED=2 / COMPLETED=3 / CANCELLED=4 /
# FAILED=5 / TIMEOUT=6 / NODE_FAIL=7 / PREEMPTED=8 / BOOT_FAIL=9 /
# DEADLINE=10 / OOM=11). The loader exposes both the integer and the
# resolved string label so reports stay readable.
SLURM_STATE_INTEGER_TO_LABEL: dict = {
    0: "PENDING", 1: "RUNNING", 2: "SUSPENDED", 3: "COMPLETED",
    4: "CANCELLED", 5: "FAILED", 6: "TIMEOUT", 7: "NODE_FAIL",
    8: "PREEMPTED", 9: "BOOT_FAIL", 10: "DEADLINE",
    11: "OUT_OF_MEMORY",
}
SLURM_FAILED_LABELS = frozenset({
    "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
    "BOOT_FAIL", "PREEMPTED", "DEADLINE",
})

# Workload-type label set is intentionally narrow — MIT Supercloud
# tags Slurm jobs with DNN model names. We expose them verbatim.
WORKLOAD_TYPE = "mit_supercloud_dnn_training"


# ---------------------------------------------------------------------------
# Dataclasses (MIT-specific carriers — converted to NormalizedGPUJob for
# Training Frontier consumption via :func:`to_normalized_gpu_job`).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedMITTrainingJob:
    """One MIT-Supercloud Slurm job.

    Every numeric / categorical field is ``Optional`` — a missing field
    means "unknown", never zero. ``token_equivalent_work`` /
    ``gpu_seconds`` are GPU-work proxies (effective_gpu × duration)
    consistent with the existing Aurelius training-frontier KPI; they
    are NOT inference tokens.
    """

    job_id: str
    submit_time_s: Optional[float]
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    duration_s: Optional[float] = None
    gpu_count_requested: Optional[int] = None
    gpu_type: Optional[str] = None
    node_count: Optional[int] = None
    nodes: Optional[str] = None
    user_or_group: Optional[str] = None
    status: Optional[str] = None
    is_failed: bool = False
    workload_label: Optional[str] = None
    model_family: Optional[str] = None
    tres_req_raw: Optional[str] = None
    token_equivalent_work: Optional[float] = None
    gpu_seconds: Optional[float] = None
    memory_requested_mib: Optional[int] = None
    memory_used_mib: Optional[float] = None
    cpu_used_pct: Optional[float] = None
    source: str = DATASET_NAME

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedMITGPUUtilizationSample:
    """One nvidia-smi sample (100 ms cadence) from the ``gpu/`` folder."""

    timestamp_s: float
    job_id: Optional[str]
    node_id: Optional[str]
    gpu_id: Optional[str]
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_used_mib: Optional[float] = None
    gpu_memory_total_mib: Optional[float] = None
    power_draw_w: Optional[float] = None
    temperature_gpu_c: Optional[float] = None
    source: str = DATASET_NAME

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedMITNodeUtilizationSample:
    """One 5-min per-node snapshot from ``node-data.csv``."""

    timestamp_s: float
    node_id: str
    system_load: Optional[float] = None
    users: Optional[int] = None
    memory_used_mib: Optional[float] = None
    memory_total_mib: Optional[float] = None
    lustre_rpcs: Optional[float] = None
    source: str = DATASET_NAME

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_optional_int(v) -> Optional[int]:
    if v is None or v == "" or str(v).strip().lower() in ("none", "null",
                                                            "nan"):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_optional_float(v) -> Optional[float]:
    if v is None or v == "" or str(v).strip().lower() in ("none", "null",
                                                            "nan"):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _pick(row: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _resolve_scheduler_path(source_dir: str) -> Optional[str]:
    for fname in SCHEDULER_LOG_FILES:
        p = os.path.join(source_dir, fname)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# TRES parser
# ---------------------------------------------------------------------------

def parse_tres_mapping(path: str) -> dict:
    """Parse ``tres-mapping.txt`` (Slurm TRES integer ↔ name table).

    The MIT README publishes the mapping as a small markdown-style
    table; the file shipped with the dataset is plain text. The parser
    tolerates both 'NN\\tname' and '| NN | name |' line formats and
    falls back to :data:`DEFAULT_TRES_MAPPING` for missing IDs.
    """
    mapping = dict(DEFAULT_TRES_MAPPING)
    if not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Allow "1001 gpu:tesla", "1001,gpu:tesla", "| 1001 | gpu:tesla |"
            cells = [c.strip() for c in re.split(r"[|\t,;]+", line)
                     if c.strip()]
            if len(cells) < 2:
                continue
            try:
                tid = int(cells[0])
            except ValueError:
                continue
            name = cells[1]
            mapping[tid] = name
    return mapping


def parse_tres_req(tres_req: Optional[str],
                   tres_mapping: Optional[dict] = None) -> dict:
    """Parse a Slurm ``tres_req`` string like ``"1=4,2=16000,1001=2"``
    into a {resource_name: amount} dict. Returns ``{}`` on missing /
    empty input — never raises."""
    if not tres_req:
        return {}
    mapping = tres_mapping or DEFAULT_TRES_MAPPING
    out: dict = {}
    for part in str(tres_req).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            tid = int(k.strip())
            amount = float(v.strip())
        except ValueError:
            continue
        name = mapping.get(tid, f"tres_{tid}")
        out[name] = amount
    return out


def gpu_count_from_tres(tres_req: Optional[str],
                         tres_mapping: Optional[dict] = None
                         ) -> tuple[int, Optional[str]]:
    """Return (gpu_count_requested, gpu_type_label) from a tres_req
    string. ``gpu_type_label`` is the first GPU-typed key seen
    (``gpu:tesla`` / ``gpu:volta`` / etc.). Returns ``(0, None)`` when
    the job did not request a GPU."""
    parsed = parse_tres_req(tres_req, tres_mapping)
    gpu_keys = [k for k in parsed if k.startswith("gpu")]
    if not gpu_keys:
        return 0, None
    count = int(sum(parsed[k] for k in gpu_keys))
    return count, gpu_keys[0]


# ---------------------------------------------------------------------------
# Scheduler / label / node loaders
# ---------------------------------------------------------------------------

def _row_get(row: dict, *keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _parse_nodelist(raw) -> Optional[str]:
    """The MIT slurm-log.csv stores ``nodelist`` as a stringified Python
    list: ``"['r9189566-n911952']"``. Strip brackets/quotes; return a
    comma-joined string of node IDs (or ``None`` when empty)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s == "[]":
        return None
    # Tolerate plain "node01,node02" and Python-list-stringified inputs.
    s = s.strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    parts = [p for p in parts if p]
    return ",".join(parts) if parts else None


def _resolve_state_label(raw) -> Optional[str]:
    """Slurm state is published as an integer in MIT's slurm-log.csv;
    map it to the canonical string label so ``is_failed`` and reports
    are readable. Strings pass through unchanged."""
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    if s.isdigit():
        return SLURM_STATE_INTEGER_TO_LABEL.get(int(s), s)
    return s.upper()


def load_scheduler_log(path: str, *,
                        tres_mapping: Optional[dict] = None,
                        labels_by_jobid: Optional[dict] = None,
                        sample_size: Optional[int] = None,
                        gpu_jobs_only: bool = False,
                        labelled_only: bool = False,
                        max_jobs: Optional[int] = None,
                        start_time_min_s: Optional[float] = None,
                        end_time_max_s: Optional[float] = None,
                        seed: int = 0,
                        ) -> list[NormalizedMITTrainingJob]:
    """Stream-parse a MIT Supercloud Slurm scheduler log.

    Memory-safe for the ~100 MB ``slurm-log.csv``: the file is read
    line-by-line via ``csv.DictReader`` and only the rows that pass
    filters (``gpu_jobs_only`` / ``labelled_only`` / time window /
    ``max_jobs``) are materialized as :class:`NormalizedMITTrainingJob`
    instances. ``sample_size`` post-filter random-samples the kept rows
    (seeded).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tres_mapping = tres_mapping or DEFAULT_TRES_MAPPING
    labels_by_jobid = labels_by_jobid or {}
    out: list[NormalizedMITTrainingJob] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise TraceSchemaError(f"{path}: missing header row")
        for row in reader:
            job_id = _row_get(row, "id_job", "job_id", "jobid", "JobID")
            if job_id is None:
                continue
            job_id = str(job_id).strip()
            tres_req = _row_get(row, "tres_req", "tres_req_str", "req_tres",
                                 "ReqTRES")
            gpu_count, gpu_type = gpu_count_from_tres(tres_req,
                                                       tres_mapping)
            if gpu_jobs_only and gpu_count == 0:
                continue
            label = labels_by_jobid.get(job_id)
            if labelled_only and not label:
                continue

            submit_time_s = _to_optional_float(
                _row_get(row, "time_submit", "submit", "submit_time",
                          "Submit"))
            start_time_s = _to_optional_float(
                _row_get(row, "time_start", "start", "start_time",
                          "Start"))
            end_time_s = _to_optional_float(
                _row_get(row, "time_end", "end", "end_time", "End"))
            if (start_time_s is None and submit_time_s is not None
                    and submit_time_s > 0):
                # Some MIT exports use ``time_eligible`` instead of
                # ``time_start``; we keep this as a non-fatal fallback.
                start_time_s = _to_optional_float(
                    _row_get(row, "time_eligible", "eligible"))
            duration_s = (
                end_time_s - start_time_s
                if (end_time_s is not None and start_time_s is not None
                    and end_time_s > start_time_s) else None)
            queue_wait_s = (
                start_time_s - submit_time_s
                if (start_time_s is not None and submit_time_s is not None
                    and start_time_s >= submit_time_s) else None)
            # Time-window filter (post-decode so we can use whichever
            # timestamp is available).
            if start_time_min_s is not None:
                anchor = submit_time_s or start_time_s
                if anchor is None or anchor < start_time_min_s:
                    continue
            if end_time_max_s is not None:
                anchor = end_time_s or start_time_s or submit_time_s
                if anchor is None or anchor > end_time_max_s:
                    continue
            status = _resolve_state_label(
                _row_get(row, "state", "status", "JobState"))
            is_failed = (status is not None
                          and status in SLURM_FAILED_LABELS)
            node_count = _to_optional_int(
                _row_get(row, "nnodes", "nodes_count", "NNodes",
                          "nodes_alloc"))
            nodes = _parse_nodelist(
                _row_get(row, "nodelist", "NodeList", "nodes"))
            user_or_group = _row_get(row, "id_user", "id_group", "user",
                                      "UserName", "User")
            memory_requested_mib = _to_optional_int(
                _row_get(row, "mem_req", "req_mem", "ReqMem"))

            parsed = parse_tres_req(tres_req, tres_mapping)
            cpu_milli = (int(parsed.get("cpu") * 1000)
                          if parsed.get("cpu") else None)
            memory_mib = (int(parsed.get("mem"))
                           if parsed.get("mem") else None)

            gpu_seconds = (gpu_count * duration_s
                           if (gpu_count and duration_s is not None)
                           else None)
            token_equivalent_work = gpu_seconds

            out.append(NormalizedMITTrainingJob(
                job_id=job_id,
                submit_time_s=submit_time_s,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                queue_wait_s=queue_wait_s,
                duration_s=duration_s,
                gpu_count_requested=(gpu_count if gpu_count >= 0 else None),
                gpu_type=gpu_type,
                node_count=node_count,
                nodes=nodes,
                user_or_group=user_or_group,
                status=status,
                is_failed=is_failed,
                workload_label=label,
                model_family=(label.split("_")[0]
                               if isinstance(label, str) else None),
                tres_req_raw=tres_req,
                token_equivalent_work=token_equivalent_work,
                gpu_seconds=gpu_seconds,
                memory_requested_mib=memory_requested_mib or memory_mib,
            ))
            if max_jobs is not None and len(out) >= max_jobs:
                break
    if sample_size is not None and 0 <= sample_size < len(out):
        rng = random.Random(seed)
        out = rng.sample(out, sample_size)
        out.sort(key=lambda j: (j.submit_time_s or 0.0, j.job_id))
    return out


def load_labelled_jobids(path: str) -> dict:
    """Read ``labelled_jobids.csv`` → {job_id: model_label} dict.

    The label column may be ``model`` / ``label`` / ``workload``
    depending on the dataset release. Returns ``{}`` if the file is
    absent."""
    if not os.path.exists(path):
        return {}
    out: dict = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            jid = _row_get(row, "id_job", "job_id", "jobid", "JobID")
            if jid is None:
                continue
            label = _row_get(row, "model", "label", "workload",
                              "model_label", "workload_label")
            if label is None:
                continue
            out[str(jid).strip()] = str(label).strip()
    return out


def load_node_data(path: str
                    ) -> list[NormalizedMITNodeUtilizationSample]:
    """Read ``node-data.csv`` (5-min per-node snapshots)."""
    if not os.path.exists(path):
        return []
    out: list[NormalizedMITNodeUtilizationSample] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        for row in reader:
            ts = _to_optional_float(_row_get(row, "timestamp",
                                              "time", "ts"))
            node_id = _row_get(row, "node", "node_id", "nodename",
                                "hostname")
            if ts is None or node_id is None:
                continue
            out.append(NormalizedMITNodeUtilizationSample(
                timestamp_s=ts, node_id=str(node_id),
                system_load=_to_optional_float(
                    _row_get(row, "system_load", "load")),
                users=_to_optional_int(_row_get(row, "users", "n_users")),
                memory_used_mib=_to_optional_float(
                    _row_get(row, "memory_used", "mem_used")),
                memory_total_mib=_to_optional_float(
                    _row_get(row, "memory_total", "mem_total")),
                lustre_rpcs=_to_optional_float(
                    _row_get(row, "lustre_rpcs", "lustre_rpc")),
            ))
    return out


def _iter_gpu_csvs(gpu_dir: str, *, max_files: Optional[int] = None
                   ) -> Iterator[str]:
    """Walk ``gpu/`` (the per-job nvidia-smi CSVs are at
    ``gpu/<NN>/<job_id>.csv``). Returns up to ``max_files`` paths."""
    if not os.path.isdir(gpu_dir):
        return
    count = 0
    for root, _dirs, files in os.walk(gpu_dir):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            yield os.path.join(root, f)
            count += 1
            if max_files is not None and count >= max_files:
                return


def gpu_filename_to_job_id(filename: str) -> str:
    """Extract id_job from a MIT Supercloud GPU CSV filename.

    Real-bucket filenames are ``<id_job>-r<id_array_job>-n<id_user>.csv``
    (e.g. ``32585007376605-r8939293-n208530.csv``). The synthetic
    fixture uses ``<id_job>.csv`` (e.g. ``1001.csv``). Both are handled:
    the leading dash-delimited token is the id_job.
    """
    base = os.path.basename(filename).rsplit(".csv", 1)[0]
    # Take the leading token before the first '-' (works for both
    # ``32585007376605-r...-n...`` and ``1001``).
    return base.split("-", 1)[0]


def load_gpu_utilization_file(path: str
                               ) -> list[NormalizedMITGPUUtilizationSample]:
    """Parse one per-job nvidia-smi CSV (100 ms cadence)."""
    if not os.path.exists(path):
        return []
    job_id = gpu_filename_to_job_id(path)
    out: list[NormalizedMITGPUUtilizationSample] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        # The MIT GPU CSV doesn't always include an absolute timestamp —
        # samples are 100 ms apart. We use the sample index as the
        # relative tick when no timestamp column is present.
        time_col = None
        for cand in ("timestamp", "time", "ts"):
            if cand in reader.fieldnames:
                time_col = cand
                break
        for i, row in enumerate(reader):
            ts_raw = (_to_optional_float(row.get(time_col))
                       if time_col else None)
            ts = ts_raw if ts_raw is not None else float(i) * 0.1
            out.append(NormalizedMITGPUUtilizationSample(
                timestamp_s=ts,
                job_id=job_id,
                node_id=_row_get(row, "node", "node_id", "hostname"),
                gpu_id=_row_get(row, "gpu", "gpu_id", "gpu_index",
                                 "index"),
                gpu_utilization_pct=_to_optional_float(
                    _row_get(row, "utilization_gpu_pct",
                              "gpu_utilization", "utilization.gpu")),
                gpu_memory_used_mib=_to_optional_float(
                    _row_get(row, "memory_used_MiB",
                              "memory_used", "memory.used")),
                gpu_memory_total_mib=_to_optional_float(
                    _row_get(row, "memory_total_MiB",
                              "memory_total", "memory.total")),
                power_draw_w=_to_optional_float(
                    _row_get(row, "power_draw_W", "power_draw",
                              "power.draw")),
                temperature_gpu_c=_to_optional_float(
                    _row_get(row, "temperature_gpu", "temperature.gpu")),
            ))
    return out


def load_gpu_utilization(gpu_dir: str, *, max_files: Optional[int] = None
                          ) -> list[NormalizedMITGPUUtilizationSample]:
    out: list[NormalizedMITGPUUtilizationSample] = []
    for path in _iter_gpu_csvs(gpu_dir, max_files=max_files):
        out += load_gpu_utilization_file(path)
    return out


# ---------------------------------------------------------------------------
# Schema discovery + join quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileClassification:
    """Per-file discovery + classification record."""

    name: str
    path: str
    status: str            # present / missing / unreadable / empty
    classification: str    # primary / scheduler_metadata / label_metadata /
                            # node_inventory / derived_aggregate / documentation
                            # / unusable
    kind: str              # request / metric / label / node / docs / unknown
    notes: str = ""


def discover(source_dir: str) -> dict:
    """Enumerate the MIT Supercloud files present in ``source_dir`` and
    classify each one. The classification is descriptive — every status
    is honest about what's actually there.

    Returns::

        {
          "source_dir": "...",
          "files": [FileClassification.to_dict(), ...],
          "summary": {
            "present_count": int, "missing_count": int,
            "primary_present": bool, "label_present": bool,
            "gpu_util_present": bool, "cpu_util_present": bool,
            "node_data_present": bool, "tres_mapping_present": bool,
          },
        }
    """
    files: list[FileClassification] = []

    def _file(name, status, classification, kind, notes=""):
        return FileClassification(
            name=name, path=os.path.join(source_dir, name),
            status=status, classification=classification, kind=kind,
            notes=notes)

    sched_path = _resolve_scheduler_path(source_dir)
    if sched_path is not None:
        files.append(_file(os.path.basename(sched_path), "present",
                           "primary", "scheduler",
                           "Slurm accounting log"))
    else:
        files.append(_file("scheduler-log.csv", "missing",
                           "primary", "scheduler",
                           f"download from {DCC_DATA_URL}"))

    for name, classification, kind, notes in [
        (LABELLED_JOBIDS_FILE, "label_metadata", "label",
         "id_job → DNN model label"),
        (TRES_MAPPING_FILE, "scheduler_metadata", "tres_mapping",
         "TRES integer ↔ resource name"),
        (NODE_DATA_FILE, "node_inventory", "node",
         "5-min per-node snapshots"),
    ]:
        p = os.path.join(source_dir, name)
        if os.path.exists(p):
            files.append(_file(name, "present", classification, kind,
                                notes))
        else:
            files.append(_file(name, "missing", classification, kind,
                                f"optional; download from {DCC_DATA_URL}"))

    gpu_dir = os.path.join(source_dir, GPU_DIR)
    if os.path.isdir(gpu_dir):
        gpu_files = list(_iter_gpu_csvs(gpu_dir, max_files=1))
        files.append(_file(GPU_DIR, "present" if gpu_files else "empty",
                           "primary", "gpu_metric",
                           f"per-job nvidia-smi CSVs (100 ms); "
                           f"{len(gpu_files) and 'samples present'}"))
    else:
        files.append(_file(GPU_DIR, "missing", "primary", "gpu_metric",
                           f"optional; download from {DCC_DATA_URL}"))

    cpu_dir = os.path.join(source_dir, CPU_DIR)
    if os.path.isdir(cpu_dir):
        files.append(_file(CPU_DIR, "present", "primary", "cpu_metric",
                           "per-job CPU CSVs (10 s)"))
    else:
        files.append(_file(CPU_DIR, "missing", "primary", "cpu_metric",
                           f"optional; download from {DCC_DATA_URL}"))

    present_count = sum(1 for f in files if f.status == "present")
    return {
        "source_dir": source_dir, "repo_url": REPO_URL,
        "dcc_data_url": DCC_DATA_URL, "paper_url": PAPER_URL,
        "files": [asdict(f) for f in files],
        "summary": {
            "present_count": present_count,
            "missing_count": len(files) - present_count,
            "primary_present": any(
                f.classification == "primary" and f.status == "present"
                for f in files),
            "label_present": any(
                f.name == LABELLED_JOBIDS_FILE and f.status == "present"
                for f in files),
            "gpu_util_present": any(
                f.name == GPU_DIR and f.status == "present" for f in files),
            "cpu_util_present": any(
                f.name == CPU_DIR and f.status == "present" for f in files),
            "node_data_present": any(
                f.name == NODE_DATA_FILE and f.status == "present"
                for f in files),
            "tres_mapping_present": any(
                f.name == TRES_MAPPING_FILE and f.status == "present"
                for f in files),
        },
    }


@dataclass(frozen=True)
class JoinResult:
    """Outcome of one join attempt (jobs ↔ labels / gpu / cpu / node)."""

    join_name: str
    join_kind: str             # exact_job_id / job_time / node_time /
                                # label_join / no_join
    left_total: int
    right_total: int
    matched_left: int
    matched_right: int
    confidence: str            # high / medium / low / none
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_join_quality(
    jobs: Sequence[NormalizedMITTrainingJob],
    *,
    labels_by_jobid: Optional[dict] = None,
    gpu_samples: Optional[Sequence[NormalizedMITGPUUtilizationSample]] = None,
    node_samples: Optional[Sequence[NormalizedMITNodeUtilizationSample]] = None,
) -> dict:
    """Build the join-quality matrix described in the mission spec.

    The join is HONEST about what's actually matchable: an exact job_id
    join (label → job) is high-confidence; node/time overlap is medium;
    no key match is none. No per-job utilization claim is made unless
    the gpu-sample file name carries the job_id (the MIT GPU CSVs do
    — they're named ``<job_id>.csv``)."""
    job_ids = {j.job_id for j in jobs}

    # Label join (exact_job_id).
    if labels_by_jobid is None:
        label_join = JoinResult(
            join_name="label_to_job", join_kind="no_join",
            left_total=0, right_total=len(jobs),
            matched_left=0, matched_right=0, confidence="none",
            notes="labelled_jobids.csv not loaded")
    else:
        matched = sum(1 for j in jobs if j.job_id in labels_by_jobid)
        label_join = JoinResult(
            join_name="label_to_job", join_kind="exact_job_id_join",
            left_total=len(labels_by_jobid), right_total=len(jobs),
            matched_left=matched, matched_right=matched,
            confidence=("high" if matched else "none"),
            notes="job_id appears in labelled_jobids.csv")

    # GPU sample join: exact_job_id, because the file name carries id.
    if gpu_samples is None:
        gpu_join = JoinResult(
            join_name="gpu_util_to_job", join_kind="no_join",
            left_total=0, right_total=len(jobs),
            matched_left=0, matched_right=0, confidence="none",
            notes="GPU utilization not loaded")
    else:
        gpu_jobs = {s.job_id for s in gpu_samples if s.job_id is not None}
        matched_jobs = gpu_jobs & job_ids
        gpu_join = JoinResult(
            join_name="gpu_util_to_job", join_kind="exact_job_id_join",
            left_total=len(gpu_jobs), right_total=len(jobs),
            matched_left=len(matched_jobs),
            matched_right=len(matched_jobs),
            confidence=("high" if matched_jobs else "none"),
            notes=("GPU sample file name == job_id (per the MIT "
                   "intro notebook); join is exact"))

    # Node sample join: only node/time overlap when nodes are recorded.
    if node_samples is None:
        node_join = JoinResult(
            join_name="node_util_to_job", join_kind="no_join",
            left_total=0, right_total=len(jobs),
            matched_left=0, matched_right=0, confidence="none",
            notes="node-data.csv not loaded")
    else:
        nodes_with_jobs = sum(
            1 for j in jobs if j.nodes is not None
            and j.start_time_s is not None and j.end_time_s is not None)
        node_set = {s.node_id for s in node_samples}
        # Heuristic match: any job whose nodelist intersects the node
        # snapshot set AND has a [start, end] window.
        candidates = [j for j in jobs
                      if j.nodes is not None
                      and any(n.strip() in node_set
                              for n in str(j.nodes).split(","))
                      and j.start_time_s is not None
                      and j.end_time_s is not None]
        node_join = JoinResult(
            join_name="node_util_to_job", join_kind="node_time_join",
            left_total=len(node_set), right_total=len(jobs),
            matched_left=len(candidates),
            matched_right=len(candidates),
            confidence=("medium" if candidates else "none"),
            notes=("node snapshot ↔ job by node-name intersection + "
                   "[start,end] window overlap; medium confidence "
                   "because snapshots are 5-min granular"))

    return {
        "joins": [j.to_dict() for j in
                  (label_join, gpu_join, node_join)],
        "n_jobs": len(jobs),
        "n_jobs_with_label": label_join.matched_right,
        "n_jobs_with_gpu_util": gpu_join.matched_right,
        "n_jobs_with_node_util_match": node_join.matched_right,
    }


# ---------------------------------------------------------------------------
# Convert MIT jobs → NormalizedGPUJob (so they feed the existing
# Aurelius training-frontier pipeline via aurelius/traces/gpu_scheduling
# and the training_philly-style estimator).
# ---------------------------------------------------------------------------

# Slurm uses INT64-near sentinels (e.g. 9223372036854775807) to mean
# "no limit" on memory / time_limit fields. Drop sentinel-like values
# so the downstream scheduling simulator's mem-gate does not reject
# every job. The threshold is generous (1 PiB) — real per-job
# memory requests never approach this.
_MEM_SENTINEL_MIB = 1 << 40  # 1 PiB in MiB


def _clamp_mem_sentinel(mem_mib: Optional[int]) -> Optional[int]:
    if mem_mib is None:
        return None
    if mem_mib < 0 or mem_mib > _MEM_SENTINEL_MIB:
        return None
    return mem_mib


def to_normalized_gpu_job(job: NormalizedMITTrainingJob
                           ) -> NormalizedGPUJob:
    """Convert a MIT-Supercloud job to the cross-dataset
    :class:`NormalizedGPUJob` contract used by the existing training
    frontier + scheduling backtest. Missing fields stay ``None``.

    ``memory_mib`` is clamped to ``None`` when the source row used a
    Slurm INT64-near sentinel for "no limit" (otherwise the scheduling
    simulator's per-node memory gate would reject every job).
    """
    return NormalizedGPUJob(
        job_id=job.job_id,
        submit_time_s=job.submit_time_s,
        start_time_s=job.start_time_s,
        end_time_s=job.end_time_s,
        duration_s=job.duration_s,
        gpu_count=int(job.gpu_count_requested or 0),
        gpu_type=job.gpu_type,
        gpu_memory_gb=None,
        status=job.status,
        user_or_group=job.user_or_group,
        workload_type=WORKLOAD_TYPE,
        priority=None,
        placement_nodes=job.nodes,
        placement_gpus=None,
        is_failed=job.is_failed,
        deadline_s=None,
        token_equivalent_work=job.token_equivalent_work,
        cpu_milli=None,
        memory_mib=_clamp_mem_sentinel(job.memory_requested_mib),
        gpu_milli=None,
        queue_wait_s=job.queue_wait_s,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_jobs(jobs: Sequence[NormalizedMITTrainingJob]) -> dict:
    """Descriptive stats — JSON-serialisable."""
    if not jobs:
        return {"job_count": 0}
    subs = [j.submit_time_s for j in jobs if j.submit_time_s is not None]
    durations = [j.duration_s for j in jobs if j.duration_s and j.duration_s > 0]
    waits = [j.queue_wait_s for j in jobs if j.queue_wait_s is not None]
    gpu_counts = [j.gpu_count_requested for j in jobs
                  if j.gpu_count_requested is not None]
    gpu_jobs = [j for j in jobs if (j.gpu_count_requested or 0) > 0]
    labelled = [j for j in jobs if j.workload_label]

    status_dist: dict = {}
    for j in jobs:
        status_dist[j.status or "Unknown"] = status_dist.get(
            j.status or "Unknown", 0) + 1
    gpu_type_dist: dict = {}
    for j in gpu_jobs:
        gpu_type_dist[j.gpu_type or "unknown"] = gpu_type_dist.get(
            j.gpu_type or "unknown", 0) + 1
    label_dist: dict = {}
    for j in labelled:
        key = str(j.model_family or j.workload_label or "unknown")
        label_dist[key] = label_dist.get(key, 0) + 1
    gpu_count_dist: dict = {}
    for j in gpu_jobs:
        k = j.gpu_count_requested or 0
        gpu_count_dist[str(k)] = gpu_count_dist.get(str(k), 0) + 1

    t0, t1 = (min(subs), max(subs)) if subs else (0.0, 0.0)
    return {
        "job_count": len(jobs),
        "gpu_job_count": len(gpu_jobs),
        "labelled_job_count": len(labelled),
        "time_start_s": t0, "time_end_s": t1,
        "duration_s_p50": (percentile(durations, 50)
                            if durations else None),
        "duration_s_p95": (percentile(durations, 95)
                            if durations else None),
        "duration_s_p99": (percentile(durations, 99)
                            if durations else None),
        "queue_wait_s_p50": (percentile(waits, 50) if waits else None),
        "queue_wait_s_p95": (percentile(waits, 95) if waits else None),
        "queue_wait_s_p99": (percentile(waits, 99) if waits else None),
        "gpu_count_distribution": dict(sorted(gpu_count_dist.items())),
        "gpu_type_distribution": dict(sorted(gpu_type_dist.items())),
        "status_distribution": dict(sorted(status_dist.items())),
        "workload_label_distribution": dict(sorted(label_dist.items())),
        "missing_fields": [
            "per_job_gpu_utilization_unless_gpu_util_loaded",
            "per_job_cpu_utilization_unless_cpu_util_loaded",
            "node_utilization_unless_node_data_loaded",
            "node_assignment_unless_present_in_scheduler_log",
        ],
    }


# ---------------------------------------------------------------------------
# All-layer convenience loader.
# ---------------------------------------------------------------------------

def load_all_layers(
    source_dir: str, *,
    include_utilization: bool = True,
    max_util_files: Optional[int] = None,
    sample_size: Optional[int] = None,
    gpu_jobs_only: bool = False,
    labelled_only: bool = False,
    max_jobs: Optional[int] = None,
    start_time_min_s: Optional[float] = None,
    end_time_max_s: Optional[float] = None,
    seed: int = 0,
) -> dict:
    """Load every present primary-telemetry layer from ``source_dir``.

    Additional bounded-real-sample parameters:

    - ``max_jobs`` short-circuits the scheduler-log read after N
      surviving rows (cheap on the ~100 MB MIT log).
    - ``start_time_min_s`` / ``end_time_max_s`` apply a time window in
      Slurm Unix seconds.
    """
    disc = discover(source_dir)
    tres = parse_tres_mapping(os.path.join(source_dir, TRES_MAPPING_FILE))
    labels = load_labelled_jobids(
        os.path.join(source_dir, LABELLED_JOBIDS_FILE))
    sched_path = _resolve_scheduler_path(source_dir)
    jobs: list[NormalizedMITTrainingJob] = []
    if sched_path is not None:
        jobs = load_scheduler_log(
            sched_path, tres_mapping=tres, labels_by_jobid=labels,
            sample_size=sample_size, gpu_jobs_only=gpu_jobs_only,
            labelled_only=labelled_only,
            max_jobs=max_jobs,
            start_time_min_s=start_time_min_s,
            end_time_max_s=end_time_max_s, seed=seed)
    node_samples = load_node_data(
        os.path.join(source_dir, NODE_DATA_FILE))
    gpu_samples = []
    if include_utilization:
        gpu_samples = load_gpu_utilization(
            os.path.join(source_dir, GPU_DIR),
            max_files=max_util_files)
    return {
        "discovery": disc,
        "tres_mapping": tres,
        "labels_by_jobid": labels,
        "jobs": jobs,
        "gpu_samples": gpu_samples,
        "node_samples": node_samples,
    }
