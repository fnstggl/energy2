#!/usr/bin/env python3
"""Focused HF telemetry-candidate audit + bounded ingestion for AcmeTrace.

Targets the highest-value HF datasets identified in the short-term mission
(``docs/HF_DATASET_REGISTRY.md`` §10 next-actions list):

1. ``Qinghao/AcmeTrace``           — Shanghai AI Lab production cluster
   traces (NSDI'24). Public, CC-BY-4.0. Contains real job-level scheduler
   traces (queue / failure / GPU / submit / end timestamps) AND
   per-server DCGM+Prometheus utilisation (15-second sampling) AND IPMI
   power telemetry. Tier 3 jobs + Tier 2 telemetry — the strongest HF
   public telemetry trace identified so far.
2. ``HuggingAGree/AcmeTrace``      — re-upload of (1). Marked duplicate
   (no ingest); discovery-only record.
3. ``osteele/llm-calibration-db``  — gated:manual. Marked
   ``gated_blocked`` (manual approval not granted; ``HF_TOKEN`` is
   not authorised). Discovery-only record.
4. ``jaytonde05/iris-prefix-cache-benchmark`` — only 20 synthetic
   prompts; no measured cache hit / latency / GPU telemetry. Marked
   ``reject_low_value`` (request_shape proxy at best; the existing
   ``jaytonde05/prefixbench`` already covers the synthetic prefix-cache
   role). Discovery-only record.

For (1) we ingest four bounded files:

  - ``data/cluster_summary.csv`` (~1.2 KB, full)            — metadata
  - ``data/job_trace/trace_kalos.csv`` (~8.6 MB, full)      — Kalos jobs
  - ``data/job_trace/trace_seren.csv`` (head ≤ 32 MB)       — Seren jobs
  - ``data/utilization/kalos/GPU_UTIL.csv`` (head ≤ 32 MB)  — DCGM util
  - ``data/utilization/ipmi/GPU_AB_Power.csv`` (head ≤ 16 MB) — IPMI

Splits:
- ``acmetrace_kalos_jobs``      → ``cluster_scheduler_trace``
- ``acmetrace_seren_jobs_head`` → ``cluster_scheduler_trace``
- ``acmetrace_kalos_gpu_util_head``  → ``telemetry_trace`` (Tier 2!)
- ``acmetrace_seren_ipmi_gpu_power_head`` → ``telemetry_trace``

Audit-only PR — does NOT modify scheduler / controllers / robust energy
engine. Raw downloads + per-config analysis_sample.jsonl are gitignored.

Layout (per-config):
    data/external/hf/Qinghao__AcmeTrace/raw/<file>          # gitignored
    data/external/hf/Qinghao__AcmeTrace/<config>/processed/
        schema_profile.json
        schema_mapping.json
        summary.json
        statistical_rollups.json
        analysis_sample.jsonl                                # gitignored
    tests/fixtures/hf/Qinghao__AcmeTrace__<config>_sample.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional  # noqa: E402

from aurelius.ingestion.operator_redistribution_policy import (  # noqa: E402
    OperatorPolicyLedger,
)
from aurelius.ingestion.redistribution_gate import (  # noqa: E402
    RedistributionGateDecision,
    decide_redistribution,
)
from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"
POLICY_PATH = DISC_DIR / "operator_redistribution_policy.json"

MAX_FIXTURE_BYTES = 16 * 1024
PER_DATASET_TIMEOUT_S = 30 * 60
PROGRESS_INTERVAL_S = 30
ROW_CAP_FOR_NORMALIZATION = 80_000  # cap analysis sample writes

# ── Redistribution-gate metadata ───────────────────────────────────────────
# Raw HF license tag + human-curated provenance for the canonical
# redistribution gate. The gate (not this script) classifies the tag
# into a ``permissive_*`` / ``unspecified_no_committed_sample`` /
# ``declared_non_permissive`` status code. Keeping the raw tag separate
# from the per-target ``license`` value here means a future HF tag
# change (e.g. the dataset owner re-licensing to apache-2.0) is a
# one-line edit; the gate handles the rest.
#
# Qinghao/AcmeTrace (NSDI'24) is the only dataset this script ingests;
# the three discovery-only records (HuggingAGree/AcmeTrace,
# osteele/llm-calibration-db, jaytonde05/iris-prefix-cache-benchmark)
# do not flow through the gate because no normalised sample is written
# for them. cc-by-4.0 is the canonical permissive HF tag — the gate
# classifies it as ``permissive_cc_by_4_0`` and permits redistribution.
DATASET_ID = "Qinghao/AcmeTrace"
LICENSE_TAG: Optional[str] = "cc-by-4.0"
LICENSE_SOURCE = (
    "HF card frontmatter license: cc-by-4.0 "
    "(NSDI'24 'Characterization of LLM Development in the Datacenter')"
)
GATE_SCOPE = "committed_normalized_sample"

logger = logging.getLogger("aurelius.hf_acmetrace")


# ── Sample-strength taxonomy ────────────────────────────────────────────────
def _statistical_sample_strength(rows: int) -> str:
    if rows >= 10_000:
        return "strong"
    if rows >= 1_000:
        return "moderate"
    if rows >= 100:
        return "weak"
    return "fixture_only"


# ── Heartbeat / timeout ─────────────────────────────────────────────────────
class _Heartbeat:
    def __init__(self, label: str):
        self.label = label
        self.start = time.monotonic()
        self.last_log = self.start
        self.phase = "init"
        self.bytes_done = 0
        self.rows_done = 0

    def update(self, *, phase=None, bytes_done=None, rows_done=None, force=False):
        if phase is not None:
            self.phase = phase
        if bytes_done is not None:
            self.bytes_done = bytes_done
        if rows_done is not None:
            self.rows_done = rows_done
        now = time.monotonic()
        if force or (now - self.last_log) >= PROGRESS_INTERVAL_S:
            elapsed = int(now - self.start)
            logger.info(
                "  [hb] %s phase=%s bytes=%d rows=%d elapsed=%ds",
                self.label, self.phase, self.bytes_done, self.rows_done, elapsed,
            )
            self.last_log = now


class _PerDatasetTimeout(Exception):
    pass


def _install_timeout(seconds: int) -> None:
    def _h(_signo, _frame):
        raise _PerDatasetTimeout(f"per-dataset timeout after {seconds}s")
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)


def _clear_timeout() -> None:
    signal.alarm(0)


# ── Redistribution gate — wire the canonical gate, do not classify here ────


def _load_ledger(policy_path: Path = POLICY_PATH) -> OperatorPolicyLedger:
    """Load the operator policy ledger from disk, or fall back to empty.

    The committed default file ships zero grants under
    ``policy_default=deny_all``; an absent file is identical in
    behaviour. We use ``empty()`` as the fallback so the script
    remains self-sufficient in a fresh checkout that may not yet have
    the committed JSON pulled — the gate still produces correct
    decisions (permit for ``cc-by-4.0`` Qinghao/AcmeTrace) instead of
    crashing.
    """

    if policy_path.exists():
        return OperatorPolicyLedger.load(policy_path)
    return OperatorPolicyLedger.empty()


def evaluate_redistribution(
    *,
    ledger: OperatorPolicyLedger,
    license_tag: Optional[str] = LICENSE_TAG,
    dataset_id: str = DATASET_ID,
    scope: str = GATE_SCOPE,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Ask the canonical gate whether the bounded normalised sample of
    Qinghao/AcmeTrace may be redistributed under the supplied license tag.

    Pure function — no I/O. Exposed so tests can drive the gate path
    without invoking the CSV download / normalisation pipeline. The
    defaults reflect the module-level constants this script ships;
    tests override them to verify the wiring (e.g. swap ``license_tag``
    to ``"cc-by-nc-4.0"`` and check that the gate denies).

    Under the default-empty ledger and the dataset's declared
    cc-by-4.0 license tag, this returns
    ``permitted=True``,
    ``license_status="permissive_cc_by_4_0"``,
    ``reason_code="permitted_declared_permissive_license"`` — the gate
    ledger is NOT consulted because the license is on the closed
    permissive allow-list.
    """

    return decide_redistribution(
        dataset_id=dataset_id,
        license_str=license_tag,
        scope=scope,
        ledger=ledger,
        now_iso=now_iso,
    )


# ── Bounded HTTP download ───────────────────────────────────────────────────
HEAD_8_MB = 8 * 1024 * 1024
HEAD_16_MB = 16 * 1024 * 1024
HEAD_32_MB = 32 * 1024 * 1024
HEAD_48_MB = 48 * 1024 * 1024


def _hf_url(dataset_id: str, repo_path: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(repo_path)}"
    )


def _bounded_download(url: str, dest: Path, *, max_bytes: int | None,
                      token: str | None, heartbeat=None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-hf-acmetrace-ingest/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{int(max_bytes - 1)}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    written = 0
    truncated = False
    status = None
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            status = resp.getcode()
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if max_bytes is not None:
                        remaining = max_bytes - written
                        if remaining <= 0:
                            truncated = True
                            break
                        if len(chunk) > remaining:
                            out.write(chunk[:remaining])
                            written += remaining
                            truncated = True
                            break
                    out.write(chunk)
                    written += len(chunk)
                    if heartbeat is not None:
                        heartbeat.update(bytes_done=written)
    except urllib.error.HTTPError as e:
        return {
            "url": url, "dest": str(dest), "status": int(e.code),
            "downloaded_bytes": 0, "truncated": False,
            "error": f"HTTPError {e.code}: {e.reason}",
            "max_bytes": max_bytes,
        }
    except urllib.error.URLError as e:
        return {
            "url": url, "dest": str(dest), "status": None,
            "downloaded_bytes": 0, "truncated": False,
            "error": f"URLError: {e.reason}",
            "max_bytes": max_bytes,
        }
    return {
        "url": url, "dest": str(dest), "status": status,
        "downloaded_bytes": written, "truncated": truncated,
        "max_bytes": max_bytes, "error": None,
    }


# ── CSV reader (with row cap, drop trailing partial) ───────────────────────
def _read_csv(path: Path, *, drop_last_partial: bool,
              max_rows: int | None = None, heartbeat=None) -> list[dict]:
    rows: list[dict] = []
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size:
                fh.seek(max(0, size - 1))
                tail = fh.read(1)
            else:
                tail = b"\n"
        file_ends_cleanly = (tail == b"\n")
    except OSError:
        file_ends_cleanly = True
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rdr = csv.DictReader(fh)
        n = 0
        for r in rdr:
            cleaned = {
                (k.rstrip("\r").strip() if k is not None else None):
                (v.rstrip("\r") if isinstance(v, str) else v)
                for k, v in r.items() if k is not None
            }
            rows.append(cleaned)
            n += 1
            if heartbeat and (n % 5000 == 0):
                heartbeat.update(rows_done=n)
            if max_rows is not None and n >= max_rows:
                break
    if drop_last_partial and not file_ends_cleanly and rows:
        rows.pop()
    return rows


# ── Schema profile + write helpers ──────────────────────────────────────────
def _classify_value(v) -> str:
    if v is None or v == "":
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    # CSV strings are str; classify common numeric-string cases as their parse type
    if isinstance(v, str):
        try:
            int(v)
            return "int_str"
        except ValueError:
            pass
        try:
            float(v)
            return "float_str"
        except ValueError:
            pass
        return "str"
    return type(v).__name__


def _profile_rows(rows: list[dict], dataset_id: str, config: str,
                  raw_file: str, file_size_bytes: int,
                  max_columns_in_profile: int | None = None) -> dict:
    """Profile rows. For very-wide utilisation CSVs (~250 server-IP columns),
    `max_columns_in_profile` keeps the JSON tractable by sampling representative
    columns; the full column list is preserved as ``raw_columns``."""
    cols: dict[str, dict] = {}
    n = len(rows)
    for r in rows:
        for k, v in r.items():
            c = cols.setdefault(k, {"present": 0, "types": set(), "examples": []})
            if v not in (None, ""):
                c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                c["examples"].append(repr(v)[:80])
    full_cols = sorted(cols.keys())
    if max_columns_in_profile is not None and len(full_cols) > max_columns_in_profile:
        # Keep first 8 + last 4 + every Nth in middle to give reviewer a sense
        sampled = full_cols[:8] + full_cols[-4:]
        stride = max(1, len(full_cols) // (max_columns_in_profile - 12))
        sampled += full_cols[8:-4:stride]
        sampled = sorted(set(sampled))[:max_columns_in_profile]
        keep_for_profile = sampled
    else:
        keep_for_profile = full_cols
    profile = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_files_inspected": [raw_file],
        "file_size_bytes": file_size_bytes,
        "inspected_row_count": n,
        "raw_columns": full_cols,
        "profiled_columns": keep_for_profile,
        "raw_column_count": len(full_cols),
        "dtypes": {k: sorted(cols[k]["types"]) for k in keep_for_profile},
        "presence_rates": {
            k: cols[k]["present"] / n if n else 0 for k in keep_for_profile
        },
        "missing_rates": {
            k: 1 - (cols[k]["present"] / n if n else 0) for k in keep_for_profile
        },
        "example_values": {k: cols[k]["examples"] for k in keep_for_profile},
    }
    return profile


def _safe_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _write_jsonl(rows: list[dict], path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    for r in rows:
        line = json.dumps(r, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        buf.write(line.encode("utf-8"))
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data), hashlib.sha256(data).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────
# Schema mappings (manually classified after inspecting the README)
# ──────────────────────────────────────────────────────────────────────────

# Kalos / Seren job traces share the canonical fields below. Kalos has
# extra mem_per_pod_GB + shared_mem_per_pod + fail_time + stop_time.
ACMETRACE_JOB_MAPPING_BASE = {
    "job_id": {
        "normalized_field": "job_id", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Unique scheduler-assigned job id.",
    },
    "user": {
        "normalized_field": "user_hash", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Hashed user id (prefix 'u').",
    },
    "node_num": {
        "normalized_field": "node_count", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Number of nodes requested for the job.",
    },
    "gpu_num": {
        "normalized_field": "gpu_count", "field_quality": "real",
        "aurelius_signal_category": "gpu_resource",
        "usable_for": ["constraint_aware_backtest", "throughput_prior"],
        "notes": "Number of GPUs requested for the job.",
    },
    "cpu_num": {
        "normalized_field": "cpu_count", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Number of CPUs requested for the job.",
    },
    "type": {
        "normalized_field": "workload_type", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only", "constraint_aware_backtest"],
        "notes": "LLM-development workload type (e.g. Other, Eval, ...).",
    },
    "state": {
        "normalized_field": "termination_state", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["constraint_aware_backtest"],
        "notes": (
            "COMPLETED / CANCELLED / FAILED / TIMEOUT / NODE_FAIL. "
            "Real timeout + failure signal."
        ),
    },
    "submit_time": {
        "normalized_field": "submit_time", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "ISO-8601 submission timestamp.",
    },
    "start_time": {
        "normalized_field": "start_time", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "ISO-8601 execution-start timestamp.",
    },
    "end_time": {
        "normalized_field": "end_time", "field_quality": "real",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "ISO-8601 termination timestamp.",
    },
    "duration": {
        "normalized_field": "duration_s", "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["throughput_prior", "constraint_aware_backtest"],
        "notes": "Derived end_time - start_time in seconds (README §1 note 2).",
    },
    "queue": {
        "normalized_field": "queue_wait_s", "field_quality": "derived",
        "aurelius_signal_category": "queue",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Derived start_time - submit_time in seconds (README §1 note 3) — REAL queue wait signal.",
    },
    "gpu_time": {
        "normalized_field": "gpu_seconds", "field_quality": "derived",
        "aurelius_signal_category": "gpu_resource",
        "usable_for": ["throughput_prior", "constraint_aware_backtest"],
        "notes": "Derived duration * gpu_num GPU-seconds (README §1 note 4).",
    },
}

ACMETRACE_KALOS_EXTRA_MAPPING = {
    "mem_per_pod_GB": {
        "normalized_field": "mem_per_pod_gb", "field_quality": "real",
        "aurelius_signal_category": "memory",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Kalos-only: requested memory per pod (GB).",
    },
    "shared_mem_per_pod": {
        "normalized_field": "shared_mem_per_pod", "field_quality": "real",
        "aurelius_signal_category": "memory",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Kalos-only: requested shared memory per pod.",
    },
    "fail_time": {
        "normalized_field": "fail_time", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Kalos-only: ISO-8601 failure timestamp.",
    },
    "stop_time": {
        "normalized_field": "stop_time", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Kalos-only: ISO-8601 user-stop timestamp.",
    },
}


# Utilisation CSVs (GPU_UTIL.csv, FB_USED.csv, IPMI power, etc.) have a
# ``Time`` column + one column per server IP. The column names are *not*
# fixed — they are the actual IPs in the dataset. We map this generically:
def _build_utilization_mapping(metric_name: str,
                               signal_category: str,
                               raw_columns: list[str]) -> dict:
    mapping = {
        "Time": {
            "normalized_field": "sample_ts", "field_quality": "real",
            "aurelius_signal_category": "request_arrival",
            "usable_for": ["dynamic_frontier_calibration"],
            "notes": "DCGM/Prometheus sample timestamp (15-second interval, README §2).",
        },
    }
    for c in raw_columns:
        if c == "Time" or not c:
            continue
        # IPs map to per-host metric streams
        mapping[c] = {
            "normalized_field": f"{metric_name}__{c.replace('.', '_')}",
            "field_quality": "real",
            "aurelius_signal_category": signal_category,
            "usable_for": ["dynamic_frontier_calibration"],
            "notes": (
                f"Per-host {metric_name} stream (server IP {c}). "
                "DCGM/Prometheus 15-second sample."
            ),
        }
    return mapping


# ──────────────────────────────────────────────────────────────────────────
# Per-target table
# ──────────────────────────────────────────────────────────────────────────

JOB_NUMERIC_FIELDS = (
    "duration_s", "queue_wait_s", "gpu_seconds", "gpu_count",
    "cpu_count", "node_count",
)

TARGETS: list[dict] = [
    {
        "dataset_id": "Qinghao/AcmeTrace",
        "config_name": "kalos_jobs",
        "raw_file": "data/job_trace/trace_kalos.csv",
        "format": "csv",
        "trace_type": "cluster_scheduler_trace",
        "stratification_keys": ["state", "type"],
        "max_download_bytes": None,  # ~8.6 MB total, full ingest
        "license": "cc-by-4.0",
        "kind": "job_trace_kalos",
        "limitations": [
            "Full ingest of trace_kalos.csv (~8.6 MB, hundreds of thousands of jobs).",
            "Real Shanghai AI Lab Kalos cluster scheduler trace (NSDI'24 'Characterization of LLM Development in the Datacenter').",
            "Job-level scheduler trace — no per-token TTFT/TPOT, no serving telemetry. Maps onto cluster_scheduler_trace (Tier 3).",
            "All durations / queues / GPU-seconds are derived columns per the README schema notes.",
            "User identifiers are hashed; no PII.",
        ],
    },
    {
        "dataset_id": "Qinghao/AcmeTrace",
        "config_name": "seren_jobs_head",
        "raw_file": "data/job_trace/trace_seren.csv",
        "format": "csv",
        "trace_type": "cluster_scheduler_trace",
        "stratification_keys": ["state", "type"],
        "max_download_bytes": HEAD_32_MB,  # full ~94 MB; head 32 MB
        "license": "cc-by-4.0",
        "kind": "job_trace_seren",
        "limitations": [
            "Bounded head-sample of trace_seren.csv (~32 MiB cap; full file is ~94 MiB).",
            "Real Shanghai AI Lab Seren cluster scheduler trace (NSDI'24).",
            "Job-level scheduler trace — no per-token TTFT/TPOT. Tier 3 cluster_scheduler_trace.",
            "Head sample preserves arrival order; subgroup_counts disclose per-state coverage so reviewers can assess statistical power before drawing tail conclusions.",
        ],
    },
    {
        "dataset_id": "Qinghao/AcmeTrace",
        "config_name": "kalos_gpu_util_head",
        "raw_file": "data/utilization/kalos/GPU_UTIL.csv",
        "format": "csv_wide_utilization",
        "trace_type": "telemetry_trace",
        "metric_name": "gpu_util_pct",
        "signal_category": "gpu_resource",
        "stratification_keys": [],
        "max_download_bytes": HEAD_32_MB,  # full ~843 MB; head 32 MB
        "license": "cc-by-4.0",
        "kind": "utilization_kalos_gpu_util",
        "limitations": [
            "Bounded head-sample of Kalos GPU_UTIL.csv (~32 MiB cap; full file is ~843 MiB).",
            "Real DCGM-collected GPU utilisation per server IP at 15-second sampling (README §2).",
            "Per-column streams are per-host GPU-utilisation percentages — one of the strongest Tier-2 public telemetry signals identified so far.",
            "Head sample is contiguous-time; statistical-power for full-cluster utilisation distribution requires the full 843 MB file (deferred).",
            "Treat as PRIOR for dynamic-frontier calibration; pilot telemetry remains the only Tier-1 calibration source.",
        ],
    },
    {
        "dataset_id": "Qinghao/AcmeTrace",
        "config_name": "seren_ipmi_gpu_power_head",
        "raw_file": "data/utilization/ipmi/GPU_AB_Power.csv",
        "format": "csv_wide_utilization",
        "trace_type": "telemetry_trace",
        "metric_name": "ipmi_gpu_power_w",
        "signal_category": "gpu_resource",
        "stratification_keys": [],
        "max_download_bytes": HEAD_16_MB,  # full ~277 MB; head 16 MB
        "license": "cc-by-4.0",
        "kind": "utilization_seren_ipmi_power",
        "limitations": [
            "Bounded head-sample of Seren GPU_AB_Power.csv (~16 MiB cap; full file is ~277 MiB).",
            "Real IPMI-collected per-server-model GPU power consumption (Watts) (README §2).",
            "Useful for energy/carbon-aware scheduling priors and dynamic-frontier energy cost calibration.",
            "Head sample is contiguous-time; not a statistical sample of the full week's diurnal pattern.",
        ],
    },
]


# Discovery-only records (no ingest):
DISCOVERY_ONLY_RECORDS = [
    {
        "dataset_id": "HuggingAGree/AcmeTrace",
        "kind": "duplicate_existing",
        "candidate_trace_type": "cluster_scheduler_trace",
        "reason": (
            "Re-upload of Qinghao/AcmeTrace (same 75 files, same SHA-equivalent "
            "content). Discovery-only; no separate ingest. The Qinghao/AcmeTrace "
            "configs are the canonical entries."
        ),
        "license_observed": "cc-by-4.0 (README + LICENSE inherited from Qinghao mirror)",
        "gated": False,
    },
    {
        "dataset_id": "osteele/llm-calibration-db",
        "kind": "gated_blocked",
        "candidate_trace_type": "telemetry_trace",
        "reason": (
            "HF gated:manual — requires manual approval from the dataset owner. "
            "HF_TOKEN is not authorised. Marked gated_blocked; revisit if/when "
            "access is granted. Schema (per HF tags) includes calibration_runs, "
            "layer_timing, memory_calibration, telemetry_samples, "
            "system_load_snapshots, inference_overhead — would qualify as Tier 4 "
            "latency_benchmark_trace + Tier 2 telemetry candidate if accessible."
        ),
        "license_observed": "mit (per HF tags)",
        "gated": True,
    },
    {
        "dataset_id": "jaytonde05/iris-prefix-cache-benchmark",
        "kind": "reject_low_value",
        "candidate_trace_type": "request_shape_trace",
        "reason": (
            "Schema is a single `prompt: string` column with 20 rows (57 KB total). "
            "No measured TTFT, no cache-hit telemetry, no GPU/queue/SLA signals — "
            "just synthetic test prompts for users to run their own benchmark. "
            "The existing jaytonde05/prefixbench config already covers the "
            "synthetic prefix-cache role with 4 jsonl files and a richer schema. "
            "Rejected as a request_shape_trace duplicate of low information density."
        ),
        "license_observed": "apache-2.0",
        "gated": False,
    },
]


# ──────────────────────────────────────────────────────────────────────────
# Per-target driver
# ──────────────────────────────────────────────────────────────────────────

def _normalize_job_row(target: dict, raw: dict, mapping: dict) -> dict:
    out: dict = {}
    for k, v in raw.items():
        m = mapping.get(k)
        if not m or not m.get("normalized_field"):
            continue
        nf = m["normalized_field"]
        # numeric CSV strings -> numeric
        if isinstance(v, str) and v != "":
            try:
                if "." in v:
                    out[nf] = float(v)
                    continue
                out[nf] = int(v)
                continue
            except ValueError:
                pass
        out[nf] = v if v != "" else None
    return out


def _normalize_util_row(raw: dict, mapping: dict) -> dict:
    """For wide utilisation rows we (a) keep ``Time`` and (b) compute per-row
    aggregates (mean/min/max/p50/p90/p95/p99) across all per-host values. This
    is far more useful than blowing up the JSONL with ~250 columns per sample."""
    ts = raw.get("Time")
    vals: list[float] = []
    for k, v in raw.items():
        if k == "Time" or not k:
            continue
        if v in (None, ""):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    out: dict = {"sample_ts": ts, "host_count": len(vals)}
    if vals:
        vals_sorted = sorted(vals)
        n = len(vals_sorted)

        def q(p: float) -> float:
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return vals_sorted[idx]
        out.update({
            "value_mean": sum(vals) / n,
            "value_min": vals_sorted[0],
            "value_max": vals_sorted[-1],
            "value_p50": q(0.5),
            "value_p90": q(0.9),
            "value_p95": q(0.95),
            "value_p99": q(0.99),
            # Pick a single representative sample column for the fixture (the
            # first non-null IP, alphabetically) so a reader can see a raw
            # data point alongside the aggregate.
        })
    return out


# ── All-signal taxonomy (aligned with telemetry_gap script) ─────────────────
ALL_SIGNALS = (
    "request_timestamps", "arrivals", "cache_reuse", "prefix_reuse",
    "kv_block_hashes", "migration_or_cache_loss_proxy",
    "autoscaling_proxy", "capacity_proxy", "routing_proxy",
    "sla_label", "timeout_label", "replica_count",
    "gpu_utilization", "cost_or_region",
    "queue_state", "latency", "ttft", "tpot",
    "model_load_event", "model_unload_event",
    "workload_shape", "customer_traffic_mix",
    "power_telemetry", "ipmi_telemetry", "dcgm_telemetry",
)


def _detect_signals_job(profile: dict) -> dict:
    cols = set(profile["raw_columns"])
    out = {s: False for s in ALL_SIGNALS}
    if any(c in cols for c in ("submit_time", "start_time", "end_time")):
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "state" in cols:
        # FAILED, TIMEOUT, NODE_FAIL in the state field provide real
        # timeout / failure labels.
        out["timeout_label"] = True
    if "queue" in cols:
        out["queue_state"] = True
    if "duration" in cols:
        out["latency"] = True
    if "gpu_num" in cols:
        out["capacity_proxy"] = True
        out["workload_shape"] = True
    if "user" in cols:
        out["customer_traffic_mix"] = True
    return out


def _detect_signals_util(profile: dict, target: dict) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    out["request_timestamps"] = True   # Time column
    out["gpu_utilization"] = True
    out["dcgm_telemetry"] = target["config_name"].startswith("kalos_gpu")
    out["ipmi_telemetry"] = "ipmi" in target["raw_file"]
    out["power_telemetry"] = "power" in target["raw_file"].lower()
    return out


def audit_one(target: dict, *, token: str | None, force_redownload: bool,
              ledger: OperatorPolicyLedger | None = None) -> dict:
    dataset_id = target["dataset_id"]
    config = target["config_name"]
    if ledger is None:
        ledger = _load_ledger()
    gate_decision = evaluate_redistribution(
        ledger=ledger,
        license_tag=target.get("license", LICENSE_TAG),
        dataset_id=dataset_id,
    )
    hb = _Heartbeat(f"{dataset_id}@{config}")
    hb.update(phase="setup", force=True)

    safe_ds = HF_DIR / dataset_id.replace("/", "__")
    raw_path = safe_ds / "raw" / Path(target["raw_file"]).name
    processed_dir = safe_ds / config / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    schema_profile_path = processed_dir / "schema_profile.json"
    schema_mapping_path = processed_dir / "schema_mapping.json"
    summary_path = processed_dir / "summary.json"
    rollups_path = processed_dir / "statistical_rollups.json"
    analysis_sample_path = processed_dir / "analysis_sample.jsonl"
    fixture_path = (
        FIXTURES_DIR / f"{dataset_id.replace('/', '__')}__{config}_sample.jsonl"
    )

    url = _hf_url(dataset_id, target["raw_file"])

    # 1. Bounded download
    hb.update(phase="download", force=True)
    if force_redownload or not raw_path.exists():
        manifest = _bounded_download(
            url, raw_path,
            max_bytes=target["max_download_bytes"],
            token=token, heartbeat=hb,
        )
        manifest["cached"] = False
    else:
        manifest = {
            "url": url, "dest": str(raw_path),
            "downloaded_bytes": raw_path.stat().st_size,
            "status": None, "truncated": None,
            "error": None, "max_bytes": target["max_download_bytes"],
            "cached": True,
        }
    if manifest.get("error"):
        return {
            "target": target, "manifest": manifest,
            "audit_status": "download_failed",
        }

    # 2. Parse bounded rows
    hb.update(phase="parse", force=True)
    fmt = target["format"]
    if fmt in ("csv", "csv_wide_utilization"):
        raw_rows = _read_csv(
            raw_path, drop_last_partial=True,
            max_rows=ROW_CAP_FOR_NORMALIZATION, heartbeat=hb,
        )
    else:
        return {"target": target, "audit_status": "unsupported_format"}

    hb.update(phase="parsed", rows_done=len(raw_rows), force=True)
    if not raw_rows:
        return {"target": target, "manifest": manifest, "audit_status": "no_rows"}

    # 3. Schema profile (for wide-util we cap profiled columns so the JSON stays
    # reviewable; the full raw_columns list is preserved separately).
    max_cols = 32 if fmt == "csv_wide_utilization" else None
    profile = _profile_rows(
        raw_rows, dataset_id, config, target["raw_file"],
        manifest["downloaded_bytes"], max_columns_in_profile=max_cols,
    )
    with open(schema_profile_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str, sort_keys=True)

    # 4. Schema mapping
    if target.get("kind") in ("job_trace_kalos",):
        mapping = dict(ACMETRACE_JOB_MAPPING_BASE)
        mapping.update(ACMETRACE_KALOS_EXTRA_MAPPING)
    elif target.get("kind") in ("job_trace_seren",):
        mapping = dict(ACMETRACE_JOB_MAPPING_BASE)
    else:
        # utilisation wide-column
        mapping = _build_utilization_mapping(
            target["metric_name"],
            target["signal_category"],
            profile["raw_columns"],
        )

    accepted = [c for c in profile["raw_columns"] if c in mapping]
    rejected = [c for c in profile["raw_columns"] if c not in mapping]
    column_records = []
    # For very-wide utilisation files, the per-host columns are programmatic
    # (one per server IP) — record one representative aggregated entry rather
    # than emitting 250+ identical column docs in JSON.
    if fmt == "csv_wide_utilization":
        column_records.append({
            "raw_column_name": "Time",
            "normalized_field": "sample_ts",
            "field_quality": mapping["Time"]["field_quality"],
            "aurelius_signal_category": mapping["Time"]["aurelius_signal_category"],
            "usable_for": mapping["Time"]["usable_for"],
            "notes": mapping["Time"]["notes"],
            "presence_rate": profile["presence_rates"].get("Time"),
            "missing_rate": profile["missing_rates"].get("Time"),
            "dtypes": profile["dtypes"].get("Time"),
        })
        # Per-host columns: collapse to a single representative entry whose
        # ``raw_column_name`` is the wildcard pattern.
        per_host_cols = [c for c in profile["raw_columns"] if c != "Time"]
        if per_host_cols:
            example = per_host_cols[0]
            column_records.append({
                "raw_column_name": "<server_ip>",
                "host_column_count": len(per_host_cols),
                "host_column_examples": per_host_cols[:5],
                "normalized_field_pattern": (
                    f"{target['metric_name']}__<server_ip_with_underscores>"
                ),
                "normalized_field": (
                    # Per-row aggregate columns that the normalized sample
                    # actually emits (value_mean / p50 / p90 / p95 / p99 / min / max)
                    "value_mean,value_min,value_max,value_p50,value_p90,value_p95,value_p99,host_count"
                ),
                "field_quality": "derived",
                "aurelius_signal_category": target["signal_category"],
                "usable_for": ["dynamic_frontier_calibration"],
                "notes": (
                    "Per-row aggregate of per-host values (DCGM/IPMI 15s sample). "
                    "Computed in the normalization step; raw per-host columns are "
                    "preserved in the gitignored raw file but not in the committed sample."
                ),
                "presence_rate": (
                    profile["presence_rates"].get(example) if example else 0.0
                ),
            })
    else:
        for c in profile["raw_columns"]:
            m = mapping.get(c, {})
            column_records.append({
                "raw_column_name": c,
                "normalized_field": m.get("normalized_field"),
                "field_quality": m.get("field_quality"),
                "aurelius_signal_category": m.get("aurelius_signal_category"),
                "usable_for": m.get("usable_for"),
                "notes": m.get("notes"),
                "presence_rate": profile["presence_rates"].get(c),
                "missing_rate": profile["missing_rates"].get(c),
                "dtypes": profile["dtypes"].get(c),
            })

    mapping_doc = {
        "dataset_id": dataset_id,
        "config_name": config,
        "accepted_columns": sorted(accepted),
        # For wide utilisation: there are NO unknown columns by construction
        # (mapping is auto-built from raw columns). For jobs: any extra
        # column should be flagged here.
        "rejected_columns": sorted(rejected) if fmt != "csv_wide_utilization" else [],
        "columns": column_records,
    }
    with open(schema_mapping_path, "w") as fh:
        json.dump(mapping_doc, fh, indent=2, default=str, sort_keys=True)

    # 5. Normalize
    hb.update(phase="normalize", force=True)
    if fmt == "csv_wide_utilization":
        normalized = [_normalize_util_row(r, mapping) for r in raw_rows]
    else:
        normalized = [_normalize_job_row(target, r, mapping) for r in raw_rows]

    normalized_schema = sorted({k for r in normalized for k in r.keys()})

    # 6. Analysis sample (gitignored)
    analysis_bytes, analysis_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], analysis_sample_path,
    )

    # 7. Fixture sample (5 deterministic rows; size-capped at 16 KiB)
    fixture_rows = [_safe_jsonable(r) for r in normalized[:5]]
    fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
    while fixture_bytes > MAX_FIXTURE_BYTES and fixture_rows:
        fixture_rows = fixture_rows[: max(1, len(fixture_rows) - 1)]
        fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
        if len(fixture_rows) == 1 and fixture_bytes > MAX_FIXTURE_BYTES:
            slim = {k: v for k, v in fixture_rows[0].items()
                    if not isinstance(v, str) or len(v) < 200}
            fixture_rows = [slim]
            fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
            break

    # 8. Signal coverage
    if fmt == "csv_wide_utilization":
        signals_detected = _detect_signals_util(profile, target)
    else:
        signals_detected = _detect_signals_job(profile)
    available_signals = sorted(s for s, v in signals_detected.items() if v)
    missing_signals = sorted(s for s, v in signals_detected.items() if not v)

    # 9. Sample strength
    strength = _statistical_sample_strength(len(normalized))

    # 10. Field-quality groupings
    real_fields = sorted([c.get("normalized_field") for c in column_records
                          if c.get("field_quality") == "real" and c.get("normalized_field")])
    derived_fields = sorted([c.get("normalized_field") for c in column_records
                             if c.get("field_quality") == "derived" and c.get("normalized_field")])
    field_quality_map = {c["normalized_field"]: c["field_quality"]
                         for c in column_records
                         if c.get("normalized_field") and c.get("field_quality")}

    # 11. Statistical rollups
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}
    for skey in target.get("stratification_keys", []):
        if skey in profile["raw_columns"]:
            counts: dict = {}
            for r in raw_rows:
                v = r.get(skey)
                if isinstance(v, (list, dict)):
                    v = "complex"
                counts[str(v)] = counts.get(str(v), 0) + 1
            rollups["subgroup_counts"][skey] = counts

    if fmt == "csv_wide_utilization":
        numeric_fields = ("value_mean", "value_p50", "value_p90", "value_p95",
                          "value_p99", "host_count")
    else:
        numeric_fields = JOB_NUMERIC_FIELDS
    for nf in numeric_fields:
        vals: list[float] = []
        for r in normalized:
            v = r.get(nf)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            vals.sort()
            n = len(vals)

            def q(p: float) -> float:
                idx = max(0, min(n - 1, int(round(p * (n - 1)))))
                return vals[idx]
            rollups["numeric_distributions"][nf] = {
                "count": n,
                "min": vals[0], "max": vals[-1],
                "mean": sum(vals) / n,
                "median": vals[n // 2],
                "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99),
            }
    with open(rollups_path, "w") as fh:
        json.dump(rollups, fh, indent=2, default=str, sort_keys=True)

    # 12. Summary (must pass promotion.gates)
    raw_schema = sorted(profile["raw_columns"])
    # For wide utilisation the "unknown_columns" gate is auto-satisfied
    # (every raw column is mapped programmatically).
    unknown_for_gate = [] if fmt == "csv_wide_utilization" else rejected

    # Signals -> string list (gates require non-empty available_signals).
    summary = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": target["license"],
        # Redistribution-gate metadata (eighth consumer of the canonical
        # gate). The gate classifies the per-target license tag —
        # cc-by-4.0 → ``permissive_cc_by_4_0`` → permit; ledger NOT
        # consulted (the closed permissive allow-list short-circuits).
        # These fields are ADDITIVE — the on-disk fixture / analysis
        # sample paths are unchanged. The script does NOT read the gate
        # verdict to decide whether to write its samples (the fixture
        # was already committed under the existing cc-by-4.0 declaration);
        # the gate fields here document the canonical permit verdict so a
        # future audit can prove the script consulted the gate rather
        # than carrying its own classifier.
        "license_redistribution_status": gate_decision.license_status,
        "license_redistribution_source": LICENSE_SOURCE,
        "redistribution_gate_reason_code": gate_decision.reason_code,
        "redistribution_gate_reason_detail": gate_decision.reason_detail,
        "redistribution_gate_permitted": gate_decision.permitted,
        "redistribution_gate_operator_grant_dataset_id": (
            gate_decision.operator_grant_dataset_id
        ),
        "redistribution_gate_scope": GATE_SCOPE,
        "gated": False,
        "canonical_trace_type": target["trace_type"],
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_bytes,
        "sample_sha256": fixture_sha,
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_bytes,
        "analysis_sample_rows": len(normalized),
        "analysis_sample_bytes": analysis_bytes,
        "analysis_sample_sha256": analysis_sha,
        "sampling_method": "head" if not target["stratification_keys"] else "head_with_stratification_keys_recorded",
        "stratification_keys": target["stratification_keys"],
        "subgroup_counts": rollups["subgroup_counts"],
        "statistical_sample_strength": strength,
        "raw_schema": raw_schema,
        "normalized_schema": normalized_schema,
        "unknown_columns": unknown_for_gate,
        "field_quality": field_quality_map,
        "available_signals": available_signals,
        "missing_signals": missing_signals,
        "real_fields": real_fields,
        "derived_fields": derived_fields,
        "proxy_fields": [],
        "synthetic_fields": [],
        "limitations": target["limitations"],
        "provenance": (
            f"{dataset_id}@{config}#{target['raw_file']}"
            f"#bytes={manifest['downloaded_bytes']}#git={(_git_sha() or '')[:7]}"
        ),
        "ingestion_timestamp_s": time.time(),
        "git_sha": _git_sha(),
        "raw_download_manifest": manifest,
        "raw_file_size_committed": False,
        "schema_profile_path": os.path.relpath(schema_profile_path, REPO_ROOT).replace(os.sep, "/"),
        "schema_mapping_path": os.path.relpath(schema_mapping_path, REPO_ROOT).replace(os.sep, "/"),
        "statistical_rollups_path": os.path.relpath(rollups_path, REPO_ROOT).replace(os.sep, "/"),
        "analysis_sample_path": os.path.relpath(analysis_sample_path, REPO_ROOT).replace(os.sep, "/"),
        "fixture_sample_path": os.path.relpath(fixture_path, REPO_ROOT).replace(os.sep, "/"),
        "summary_path_relative": os.path.relpath(summary_path, REPO_ROOT).replace(os.sep, "/"),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str, sort_keys=True)

    # 13. Promotion evaluation
    decision = promotion.evaluate_promotion(summary)
    hb.update(phase="done", force=True)
    return {
        "dataset_id": dataset_id,
        "config_name": config,
        "manifest": manifest,
        "summary": summary,
        "decision": decision,
        "audit_status": "ok",
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--only", default=None,
                   help="config_name substring filter")
    p.add_argument("--token", default=None,
                   help="HF token override (defaults to HF_TOKEN env var)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    token = args.token or os.environ.get("HF_TOKEN")

    targets = TARGETS
    if args.only:
        targets = [t for t in TARGETS if args.only in t["config_name"]]
        if not targets:
            logger.error("--only filter matched no targets")
            return 2

    # Load the operator policy ledger once; thread it through every
    # audit_one call and the audit summary writer so a single source of
    # truth records ``redistribution_gate_policy_default`` and
    # ``redistribution_gate_policy_grant_count``.
    ledger = _load_ledger()

    ingested: list[dict] = []
    failed: list[dict] = []

    for t in targets:
        logger.info("ingest %s/%s ... (timeout %ds)",
                    t["dataset_id"], t["config_name"], PER_DATASET_TIMEOUT_S)
        t_start = time.monotonic()
        try:
            _install_timeout(PER_DATASET_TIMEOUT_S)
            r = audit_one(
                t, token=token,
                force_redownload=args.force_redownload,
                ledger=ledger,
            )
        except _PerDatasetTimeout as e:
            elapsed = int(time.monotonic() - t_start)
            failed.append({
                "dataset_id": t["dataset_id"],
                "config_name": t["config_name"],
                "audit_status": "deferred_timeout",
                "elapsed_s": elapsed,
                "error": str(e),
            })
            continue
        finally:
            _clear_timeout()

        elapsed = int(time.monotonic() - t_start)
        if r.get("audit_status") != "ok":
            failed.append({
                "dataset_id": t["dataset_id"],
                "config_name": t["config_name"],
                "audit_status": r.get("audit_status"),
                "elapsed_s": elapsed,
                "manifest": r.get("manifest"),
            })
            continue
        ingested.append({
            "dataset_id": r["dataset_id"],
            "config_name": r["config_name"],
            "canonical_trace_type": r["summary"]["canonical_trace_type"],
            "license": r["summary"]["license"],
            # Eighth-consumer gate-derived fields. The audit summary
            # mirrors the same closed-set fields the per-config
            # summary.json carries so reviewers can pivot on either
            # source without re-running the gate.
            "license_redistribution_status":
                r["summary"]["license_redistribution_status"],
            "redistribution_gate_reason_code":
                r["summary"]["redistribution_gate_reason_code"],
            "redistribution_gate_permitted":
                r["summary"]["redistribution_gate_permitted"],
            "redistribution_gate_operator_grant_dataset_id":
                r["summary"]["redistribution_gate_operator_grant_dataset_id"],
            "available_signals": r["summary"]["available_signals"],
            "missing_signals": r["summary"]["missing_signals"],
            "analysis_sample_rows": r["summary"]["analysis_sample_rows"],
            "statistical_sample_strength": r["summary"]["statistical_sample_strength"],
            "promotion_state": r["decision"]["state"],
            "promotion_tags": r["decision"]["promotion_tags"],
            "elapsed_s": elapsed,
            "limitations": r["summary"]["limitations"],
        })

    rollup_path = DISC_DIR / "acmetrace_audit_summary.json"
    payload = {
        # v2 schema introduces the redistribution_gate_* top-level triple
        # mirroring the broadened_discovery_audit_summary v2 schema.
        # Per-row gate fields are also added under ``ingested``. v1
        # readers continue to function: the v1 schema is a strict
        # subset of v2 (all v1 keys are preserved).
        "doc_version": "acmetrace_audit_summary_v2",
        "stage": "hf_focused_audit_acmetrace_v2",
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "production_claim": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "redistribution_gate_scope": GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "ingested": ingested,
        "failed": failed,
        "discovery_only_records": DISCOVERY_ONLY_RECORDS,
    }
    DISC_DIR.mkdir(parents=True, exist_ok=True)
    with open(rollup_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)

    logger.info("wrote rollup %s (ingested=%d, failed=%d, discovery_only=%d)",
                rollup_path, len(ingested), len(failed),
                len(DISCOVERY_ONLY_RECORDS))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
