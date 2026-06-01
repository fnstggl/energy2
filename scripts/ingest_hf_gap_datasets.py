#!/usr/bin/env python3
"""Bounded ingestion of the top telemetry-gap datasets from the Aurelius
HF gap-closure audit (PR #125).

Audit-only. Does NOT modify scheduler / controllers / robust energy
engine. Does NOT train forecasting models. Raw downloads + large
analysis samples are gitignored — only schema_profile, schema_mapping,
summary, signal-coverage, statistical_rollups, and tiny fixtures are
committed.

Datasets ingested:
1. semianalysisai/cc-traces-weka-no-subagents-051226 (jsonl, head 80 MiB)
2. sammshen/lmcache-agentic-traces (parquet, smallest shard ≤ 400 MB)
3. lzzmm/BurstGPT (CSV, full BurstGPT_1.csv ≈ 52 MB + head of others)
4. lsliwko/google-cluster-data-2019-sorted-by-timestamp (one gz shard, 53 MB)
5. jaytonde05/prefixbench (all 4 jsonl files, total 80 MB — full)

Layout (per-config):
    data/external/hf/<safe_dataset>/raw/<file>            # gitignored
    data/external/hf/<safe_dataset>/<config>/processed/
        schema_profile.json          (committed)
        schema_mapping.json          (committed)
        summary.json                 (committed)
        statistical_rollups.json     (committed)
        analysis_sample.jsonl        (gitignored)
    tests/fixtures/hf/<safe_dataset>__<config>_sample.jsonl   (committed,
        ≤ 16 KiB, ≤ 5 rows)
"""

from __future__ import annotations

import argparse
import csv
import gzip
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

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

MAX_COMMITTED_FIXTURE_BYTES = 16 * 1024  # tighter than the 16-MiB promotion gate

# Hard runtime bounds (per dataset).
PER_DATASET_TIMEOUT_S = 30 * 60  # 30 minutes max per dataset
ROW_CAP_FOR_NORMALIZATION = 60_000  # cap analysis sample to avoid 1.4M-row stalls
PROGRESS_INTERVAL_S = 30  # log progress every 30 s

logger = logging.getLogger("aurelius.hf_gap_ingest")


class _Heartbeat:
    """Print a progress line every PROGRESS_INTERVAL_S so the operator can see
    the script is alive."""

    def __init__(self, label: str):
        self.label = label
        self.start = time.monotonic()
        self.last_log = self.start
        self.phase = "init"
        self.bytes_done = 0
        self.rows_done = 0

    def update(self, *, phase: str = None, bytes_done: int = None, rows_done: int = None,
               force: bool = False) -> None:
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

# ── Bounded-download budgets ────────────────────────────────────────────────
HEAD_80_MIB = 80 * 1024 * 1024
HEAD_64_MIB = 64 * 1024 * 1024
HEAD_FULL_55_MB = 55 * 1024 * 1024  # full BurstGPT_1.csv (52 MB on disk)
HEAD_20_MB = 20 * 1024 * 1024       # bounded BurstGPT sample (~550k rows)
HEAD_55_MB = 55 * 1024 * 1024


# ── Per-dataset target table ────────────────────────────────────────────────
TARGETS: list[dict] = [
    # 1. semianalysisai/cc-traces — Claude Code production agentic traces
    {
        "dataset_id": "semianalysisai/cc-traces-weka-no-subagents-051226",
        "config_name": "traces_head",
        "raw_file": "traces.jsonl",
        "format": "jsonl",
        "trace_type": "request_shape_trace",
        "stratification_keys": ["model"],
        "max_download_bytes": HEAD_80_MIB,
        "license": "apache-2.0",
        "limitations": [
            "Bounded head-sample of 2.77 GB traces.jsonl (80 MiB cap, ~5k rows).",
            "Real Claude Code CLI ≥ 2.1.139 production traffic (949 agent sessions, ~136.1k requests in the full file).",
            "Multi-turn agentic traces with per-request KV block hashes; bucket_hashes preserved at the raw level, hashed + sampled in the committed sample.",
            "No GPU type, no queue/scheduler state, no measured TTFT/TPOT in this trace format.",
        ],
    },
    # 2. sammshen/lmcache-agentic-traces — 787 multi-turn sessions
    {
        "dataset_id": "sammshen/lmcache-agentic-traces",
        "config_name": "train_shard4",
        "raw_file": "data/train-00004-of-00005.parquet",  # smallest shard ~398 MB
        "format": "parquet",
        "trace_type": "request_shape_trace",
        "stratification_keys": ["source"],
        "max_download_bytes": None,  # full file ~398 MB; pyarrow will read schema
        "license": "mit",
        "limitations": [
            "Single parquet shard (train-00004-of-00005, 398 MB) of the 2.3 GB 5-shard set.",
            "787 multi-turn agentic LLM sessions (24,881 total iterations); ≥5 turns/session and ≥10K context tokens.",
            "Designed for tiered KV-cache benchmarking (LMCache).",
            "No GPU type, no queue/scheduler state, no measured TTFT/TPOT.",
        ],
    },
    # 3. lzzmm/BurstGPT — Real Microsoft Azure ChatGPT trace
    {
        "dataset_id": "lzzmm/BurstGPT",
        "config_name": "burstgpt_1_full",
        "raw_file": "data/BurstGPT_1.csv",
        "format": "csv",
        "trace_type": "request_shape_trace",
        "stratification_keys": ["Model", "Log Type"],
        "max_download_bytes": HEAD_20_MB,  # bounded 20 MiB head ≈ 550k rows
        "license": "unspecified_LICENSE_file_present",
        "limitations": [
            "Bounded 20 MiB head of BurstGPT_1.csv (file is 52.3 MB; ~1.43M rows total).",
            "Normalized analysis sample capped at 60k rows (RAM + write-time bound).",
            "BurstGPT_2.csv (145 MB) and the two without_fails splits deferred.",
            "No GPU type, no scheduler state. Pure arrival timestamps + model + token counts + Conversation vs API Log Type.",
        ],
    },
    # 4. lsliwko/google-cluster-data-2019 — one instance_events shard
    {
        "dataset_id": "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
        "config_name": "instance_events_shard0",
        "raw_file": (
            "GoogleClusterData2019a/data/instance_events/"
            "instance_events-00000000-sorted.json.gz"
        ),
        "format": "jsonl_gz",
        "trace_type": "cluster_scheduler_trace",
        "stratification_keys": ["type"],
        "max_download_bytes": HEAD_55_MB,  # full ~52.7 MB
        "license": "cc_by_4_0_derived_from_google",
        "limitations": [
            "Single instance_events shard (52.7 MB gzipped, ~3M events) of the 117 GB Google Cluster 2019 mirror.",
            "Google Borg task lifecycle events (SUBMIT, SCHEDULE, EVICT, FAIL, FINISH, KILL, LOST, QUEUE, UPDATE_PENDING, UPDATE_RUNNING) — anonymized.",
            "NOT LLM serving telemetry — Borg job-level scheduling events. Treat as autoscaling / migration / fleet-inventory PROXY only.",
            "Mirror of github.com/google/cluster-data; license cc-by-4.0 per Google's release.",
        ],
    },
    # 5. jaytonde05/prefixbench — synthetic cache-eviction benchmarks
    {
        "dataset_id": "jaytonde05/prefixbench",
        "config_name": "prefixbench_all",
        "raw_file": "shared_schema_1k.jsonl",  # representative; will also fetch peers
        "format": "jsonl",
        "trace_type": "cache_residency_trace",
        "stratification_keys": [],
        "max_download_bytes": HEAD_64_MIB,
        "license": "unspecified",
        "extra_files": [
            "multiturn_agent_branching.jsonl",
            "same_document_multi_query_8k.jsonl",
            "eviction_pressure_4k.jsonl",
        ],
        "limitations": [
            "Full ingest of all 4 jsonl files (80 MB total).",
            "SYNTHETIC deterministic prompts engineered for KV-prefix-cache benchmarking; not production traffic.",
            "Useful as cache-residency replay corpus; NOT as real cache-hit telemetry.",
        ],
    },
]


# ── Signal-coverage taxonomy (matches docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md) ──
ALL_SIGNALS = (
    "request_timestamps", "arrivals", "cache_reuse", "prefix_reuse",
    "kv_block_hashes", "migration_or_cache_loss_proxy",
    "autoscaling_proxy", "capacity_proxy", "routing_proxy",
    "sla_label", "timeout_label", "replica_count",
    "gpu_utilization", "cost_or_region",
    "queue_state", "latency", "ttft", "tpot",
    "model_load_event", "model_unload_event",
    "workload_shape", "customer_traffic_mix",
)


# ──────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────


def _safe_dataset_dir(dataset_id: str) -> Path:
    return HF_DIR / dataset_id.replace("/", "__")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


def _bounded_download(url: str, dest: Path, *, max_bytes: int | None, token: str | None) -> dict:
    """HTTP-Range bounded download. If max_bytes is None, fetches the full file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-hf-gap-ingest/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{int(max_bytes - 1)}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    written = 0
    status = None
    truncated = False
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def _hf_url(dataset_id: str, raw_file: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(raw_file)}"
    )


def _read_jsonl(path: Path, *, drop_last_partial: bool, gz: bool = False,
                max_rows: int | None = None, heartbeat=None) -> list[dict]:
    rows: list[dict] = []
    opener = gzip.open if gz else open
    try:
        with opener(path, "rb") as fh:
            data = fh.read()
    except (OSError, EOFError) as e:
        logger.warning("gz read truncated (%s); falling back to partial read", e)
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            data = gzip.decompress(raw)
        except (OSError, EOFError):
            buf = io.BytesIO()
            d = gzip.GzipFile(fileobj=io.BytesIO(raw))
            try:
                while True:
                    chunk = d.read(65536)
                    if not chunk:
                        break
                    buf.write(chunk)
            except (OSError, EOFError):
                pass
            data = buf.getvalue()
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if drop_last_partial and lines:
        lines = lines[:-1]
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
            if heartbeat and (len(rows) % 5000 == 0):
                heartbeat.update(rows_done=len(rows))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _read_csv(path: Path, *, drop_last_partial: bool,
              max_rows: int | None = None, heartbeat=None) -> list[dict]:
    """Stream-read CSV with optional row cap so a 1.4M-row file doesn't stall."""
    rows: list[dict] = []
    # Determine if file ends cleanly (newline-terminated).
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
            if heartbeat and (n % 10000 == 0):
                heartbeat.update(rows_done=n)
            if max_rows is not None and n >= max_rows:
                break
    if drop_last_partial and not file_ends_cleanly and rows:
        rows.pop()
    return rows


def _read_parquet(path: Path, *, max_rows: int = 20000) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    if max_rows is not None and t.num_rows > max_rows:
        t = t.slice(0, max_rows)
    rows = t.to_pylist()
    return rows


def _hash_list(xs: list) -> str:
    """Deterministic 16-hex sha256 prefix of a list of values."""
    enc = json.dumps(xs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(enc).hexdigest()[:16]


def _sample_list(xs: list, n: int = 5) -> list:
    return xs[:n] if isinstance(xs, list) else []


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


def _statistical_sample_strength(rows: int) -> str:
    if rows >= 10_000:
        return "strong"
    if rows >= 1_000:
        return "moderate"
    if rows >= 100:
        return "weak"
    return "fixture_only"


# ──────────────────────────────────────────────────────────────────────────
# Schema profile (column inventory + dtypes + missing rates)
# ──────────────────────────────────────────────────────────────────────────


def _classify_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _profile_rows(rows: list[dict], dataset_id: str, config: str, raw_file: str, file_size_bytes: int) -> dict:
    cols: dict[str, dict] = {}
    nested: dict[str, set] = {}
    list_lens: dict[str, list[int]] = {}
    n = len(rows)
    for r in rows:
        for k, v in r.items():
            c = cols.setdefault(k, {"present": 0, "types": set(), "examples": []})
            c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                # Render compact preview
                if isinstance(v, (dict, list)):
                    try:
                        c["examples"].append(json.dumps(v, default=str)[:120])
                    except Exception:
                        c["examples"].append(repr(v)[:120])
                else:
                    c["examples"].append(repr(v)[:120])
            if isinstance(v, list):
                list_lens.setdefault(k, []).append(len(v))
            if isinstance(v, dict):
                for nk in v.keys():
                    nested.setdefault(k, set()).add(nk)
    profile = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_files_inspected": [raw_file],
        "file_size_bytes": file_size_bytes,
        "inspected_row_count": n,
        "raw_columns": sorted(cols.keys()),
        "dtypes": {k: sorted(c["types"]) for k, c in cols.items()},
        "presence_rates": {k: c["present"] / n if n else 0 for k, c in cols.items()},
        "missing_rates": {k: 1 - (c["present"] / n if n else 0) for k, c in cols.items()},
        "example_values": {k: c["examples"] for k, c in cols.items()},
        "list_length_summaries": {
            k: {
                "n": len(v),
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
                "mean": (sum(v) / len(v)) if v else 0,
            }
            for k, v in list_lens.items()
        },
        "nested_keys": {k: sorted(v) for k, v in nested.items()},
    }
    return profile


# ──────────────────────────────────────────────────────────────────────────
# Per-dataset schema mappings (raw column → normalized field + field_quality)
#
# Manually classified after inspecting the actual schemas during the
# PR #125 discovery audit. Field classifications use the canonical
# corpus FIELD_QUALITY_VALUES (real / derived / proxy / synthetic /
# missing).
# ──────────────────────────────────────────────────────────────────────────

# 1. semianalysisai/cc-traces — session-level format with one JSON object per
# agent session. Each session contains:
#   id          : session identifier
#   block_size  : KV block size in tokens
#   hash_id_scope : block-hash scoping (typically "local")
#   models      : list of models used in the session
#   requests    : list of per-request dicts with keys {t, model, in, out, hash_ids, ...}
#
# We FLATTEN every session-row into per-request rows in the parse step so
# downstream signal detection treats this as request-level cache_residency
# evidence (per-request hash_ids is the KV-block-hash signal).
CC_TRACES_MAPPING = {
    # Session-level columns (carried into each flattened request row)
    "id": {"normalized_field": "session_id", "field_quality": "real",
           "aurelius_signal_category": "session_id",
           "usable_for": ["arrival_forecast", "cache_residency_evaluation"],
           "notes": "Per-agent-session UUID. Same value for every request in the session."},
    "block_size": {"normalized_field": "block_size_tokens",
                   "field_quality": "real",
                   "aurelius_signal_category": "cache_residency",
                   "usable_for": ["cache_residency_evaluation"],
                   "notes": "Session KV block size in tokens (typically 64)."},
    "hash_id_scope": {"normalized_field": "hash_id_scope",
                      "field_quality": "real",
                      "aurelius_signal_category": "cache_residency",
                      "usable_for": ["cache_residency_evaluation"],
                      "notes": "KV block-hash scoping (e.g. 'local')."},
    "models": {"normalized_field": "session_models",
               "field_quality": "real",
               "aurelius_signal_category": "metadata_only",
               "usable_for": ["workload_shape_prior"],
               "notes": "Set of models used across the session."},
    "requests": {"normalized_field": "requests_count",
                 "field_quality": "real",
                 "aurelius_signal_category": "session_id",
                 "usable_for": ["capacity_forecast"],
                 "notes": "Session-level requests array; flattened per-request before mapping."},
    # Per-request columns (after flatten step):
    "t": {"normalized_field": "request_arrival_delta_s",
          "field_quality": "real",
          "aurelius_signal_category": "request_dispatch",
          "usable_for": ["arrival_forecast", "capacity_forecast"],
          "notes": "Per-request arrival time delta in seconds since session start."},
    "model": {"normalized_field": "model_id", "field_quality": "real",
              "aurelius_signal_category": "metadata_only",
              "usable_for": ["arrival_forecast"], "notes": "Claude model id."},
    "in": {"normalized_field": "input_tokens", "field_quality": "real",
           "aurelius_signal_category": "tokens",
           "usable_for": ["arrival_forecast", "latency_prior"],
           "notes": "Prompt token count."},
    "out": {"normalized_field": "output_tokens",
            "field_quality": "real",
            "aurelius_signal_category": "tokens",
            "usable_for": ["latency_prior"],
            "notes": "Output token count."},
    "hash_ids": {"normalized_field": "block_hashes",
                 "field_quality": "real",
                 "aurelius_signal_category": "cache_residency",
                 "usable_for": ["cache_residency_evaluation", "migration_veto_evaluation"],
                 "notes": "Per-request KV-block-hash list (cache-residency / migration-veto signal). Compressed in committed sample."},
    "turn": {"normalized_field": "turn_index",
             "field_quality": "real",
             "aurelius_signal_category": "session_id",
             "usable_for": ["cache_residency_evaluation"],
             "notes": "Sequential turn index within the session."},
    "api_time": {"normalized_field": "api_time_s",
                 "field_quality": "real",
                 "aurelius_signal_category": "latency",
                 "usable_for": ["latency_prior"],
                 "notes": "End-to-end API call duration in seconds."},
    "think_time": {"normalized_field": "think_time_s",
                   "field_quality": "real",
                   "aurelius_signal_category": "request_dispatch",
                   "usable_for": ["arrival_forecast"],
                   "notes": "User/agent think time between turns in seconds."},
    "ttft": {"normalized_field": "ttft_s",
             "field_quality": "real",
             "aurelius_signal_category": "latency",
             "usable_for": ["latency_prior"],
             "notes": "Time-to-first-token in seconds."},
    "type": {"normalized_field": "request_type",
             "field_quality": "real",
             "aurelius_signal_category": "metadata_only",
             "usable_for": ["workload_shape_prior"],
             "notes": "Request kind (e.g. message, tool_result)."},
}

# 2. sammshen/lmcache-agentic-traces (parquet schema as observed):
#   session_id    : multi-turn session UUID
#   input         : full prompt text (large string)
#   output_length : tokens generated
#   model         : model name
#   pre_gap       : think-time / inter-turn gap in seconds
LMCACHE_MAPPING = {
    "session_id": {"normalized_field": "session_id", "field_quality": "real",
                   "aurelius_signal_category": "session_id",
                   "usable_for": ["cache_residency_evaluation", "arrival_forecast"],
                   "notes": "Multi-turn agentic session identifier."},
    "model": {"normalized_field": "model_id", "field_quality": "real",
              "aurelius_signal_category": "metadata_only",
              "usable_for": ["arrival_forecast"], "notes": "Model name."},
    "input": {"normalized_field": "input_len",
              "field_quality": "real",
              "aurelius_signal_category": "tokens",
              "usable_for": ["cache_residency_evaluation"],
              "notes": "Full prompt text; only character length kept in committed sample (cache-residency proxy)."},
    "output_length": {"normalized_field": "output_tokens",
                      "field_quality": "real",
                      "aurelius_signal_category": "tokens",
                      "usable_for": ["arrival_forecast"],
                      "notes": "Generated token count."},
    "pre_gap": {"normalized_field": "pre_gap_s",
                "field_quality": "real",
                "aurelius_signal_category": "request_dispatch",
                "usable_for": ["arrival_forecast"],
                "notes": "Inter-turn gap (think-time) in seconds. Arrival pattern signal."},
}

# 3. lzzmm/BurstGPT
BURSTGPT_MAPPING = {
    "Timestamp": {"normalized_field": "request_arrival_ts_s",
                  "field_quality": "real",
                  "aurelius_signal_category": "request_dispatch",
                  "usable_for": ["arrival_forecast", "capacity_forecast"],
                  "notes": "Unix-seconds arrival timestamp from Microsoft Azure ChatGPT logs."},
    "Model": {"normalized_field": "model_id", "field_quality": "real",
              "aurelius_signal_category": "metadata_only",
              "usable_for": ["arrival_forecast"],
              "notes": "ChatGPT or GPT-4."},
    "Request tokens": {"normalized_field": "input_tokens",
                       "field_quality": "real",
                       "aurelius_signal_category": "tokens",
                       "usable_for": ["arrival_forecast"],
                       "notes": "Prompt token count."},
    "Response tokens": {"normalized_field": "output_tokens",
                        "field_quality": "real",
                        "aurelius_signal_category": "tokens",
                        "usable_for": ["arrival_forecast"],
                        "notes": "Response token count."},
    "Total tokens": {"normalized_field": "total_tokens",
                     "field_quality": "derived",
                     "aurelius_signal_category": "tokens",
                     "usable_for": ["capacity_forecast"],
                     "notes": "Sum of request + response tokens (derived)."},
    "Log Type": {"normalized_field": "log_type",
                 "field_quality": "real",
                 "aurelius_signal_category": "metadata_only",
                 "usable_for": ["workload_shape_prior"],
                 "notes": "Conversation log vs API call — customer-traffic-mix proxy."},
}

# 4. Google Cluster 2019 instance_events
GCD_MAPPING = {
    "time": {"normalized_field": "event_time_us", "field_quality": "real",
             "aurelius_signal_category": "request_dispatch",
             "usable_for": ["arrival_forecast", "capacity_forecast", "autoscaling_evaluation"],
             "notes": "Microsecond Borg trace time."},
    "type": {"normalized_field": "event_type",
             "field_quality": "real",
             "aurelius_signal_category": "scheduler_state",
             "usable_for": ["autoscaling_evaluation", "migration_veto_evaluation",
                            "model_load_unload_proxy"],
             "notes": "One of SUBMIT, SCHEDULE, EVICT, FAIL, FINISH, KILL, LOST, QUEUE, UPDATE_PENDING, UPDATE_RUNNING."},
    "collection_id": {"normalized_field": "collection_id",
                      "field_quality": "real",
                      "aurelius_signal_category": "metadata_only",
                      "usable_for": ["autoscaling_evaluation"],
                      "notes": "Borg collection (workload group) id."},
    "scheduling_class": {"normalized_field": "scheduling_class",
                         "field_quality": "real",
                         "aurelius_signal_category": "scheduler_state",
                         "usable_for": ["autoscaling_evaluation"],
                         "notes": "Borg scheduling class (0=lowest, 3=highest)."},
    "priority": {"normalized_field": "priority", "field_quality": "real",
                 "aurelius_signal_category": "scheduler_state",
                 "usable_for": ["autoscaling_evaluation"],
                 "notes": "Borg priority value."},
    "machine_id": {"normalized_field": "machine_id", "field_quality": "real",
                   "aurelius_signal_category": "placement",
                   "usable_for": ["placement_prior", "migration_veto_evaluation"],
                   "notes": "Anonymized machine cell id."},
    "instance_index": {"normalized_field": "instance_index",
                       "field_quality": "real",
                       "aurelius_signal_category": "metadata_only",
                       "usable_for": ["placement_prior"],
                       "notes": "Per-task index within a collection."},
    "alloc_collection_id": {"normalized_field": "alloc_collection_id",
                            "field_quality": "real",
                            "aurelius_signal_category": "placement",
                            "usable_for": ["placement_prior"],
                            "notes": "Alloc-set collection id (nested allocations)."},
    "alloc_instance_index": {"normalized_field": "alloc_instance_index",
                             "field_quality": "real",
                             "aurelius_signal_category": "metadata_only",
                             "usable_for": ["placement_prior"],
                             "notes": "Alloc instance index."},
    "user": {"normalized_field": "user_hash", "field_quality": "real",
             "aurelius_signal_category": "metadata_only",
             "usable_for": ["workload_shape_prior"],
             "notes": "Anonymized Borg user hash (customer traffic mix proxy)."},
    "resource_request": {"normalized_field": "resource_request_summary",
                         "field_quality": "real",
                         "aurelius_signal_category": "scheduler_state",
                         "usable_for": ["capacity_forecast"],
                         "notes": "Dict with cpus + memory; flattened to JSON in committed sample."},
    "constraint": {"normalized_field": "constraint_summary",
                   "field_quality": "real",
                   "aurelius_signal_category": "scheduler_state",
                   "usable_for": ["placement_prior"],
                   "notes": "Scheduling constraints (list); summarised."},
    "missing_type": {"normalized_field": "missing_type",
                     "field_quality": "real",
                     "aurelius_signal_category": "metadata_only",
                     "usable_for": ["not_usable"],
                     "notes": "Indicator that the event type was inferred."},
    "collection_type": {"normalized_field": "collection_type",
                        "field_quality": "real",
                        "aurelius_signal_category": "metadata_only",
                        "usable_for": ["autoscaling_evaluation"],
                        "notes": "JOB vs ALLOC_SET."},
}

# 5. jaytonde05/prefixbench
PREFIXBENCH_MAPPING = {
    "id": {"normalized_field": "prompt_id", "field_quality": "real",
           "aurelius_signal_category": "metadata_only",
           "usable_for": ["cache_residency_evaluation"], "notes": "Deterministic prompt id."},
    "prompt": {"normalized_field": "prompt_text",
               "field_quality": "synthetic",
               "aurelius_signal_category": "metadata_only",
               "usable_for": ["cache_residency_evaluation"],
               "notes": "Synthetic prompt (deterministic prefix engineering). Truncated in committed sample."},
    "max_tokens": {"normalized_field": "max_tokens",
                   "field_quality": "real",
                   "aurelius_signal_category": "tokens",
                   "usable_for": ["cache_residency_evaluation"],
                   "notes": "Generation cap (parameter)."},
    "temperature": {"normalized_field": "temperature",
                    "field_quality": "real",
                    "aurelius_signal_category": "metadata_only",
                    "usable_for": ["not_usable"], "notes": "Sampling temperature."},
    "metadata": {"normalized_field": "metadata_json",
                 "field_quality": "real",
                 "aurelius_signal_category": "metadata_only",
                 "usable_for": ["cache_residency_evaluation"],
                 "notes": "Nested dict with prefix variant + scenario; JSON-stringified."},
}


MAPPINGS = {
    ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head"): CC_TRACES_MAPPING,
    ("sammshen/lmcache-agentic-traces", "train_shard4"): LMCACHE_MAPPING,
    ("lzzmm/BurstGPT", "burstgpt_1_full"): BURSTGPT_MAPPING,
    ("lsliwko/google-cluster-data-2019-sorted-by-timestamp", "instance_events_shard0"): GCD_MAPPING,
    ("jaytonde05/prefixbench", "prefixbench_all"): PREFIXBENCH_MAPPING,
}


# ──────────────────────────────────────────────────────────────────────────
# Signal detection (per dataset, per normalized row)
# ──────────────────────────────────────────────────────────────────────────


def _detect_signals(target: dict, profile: dict, mapping: dict, rows: list[dict]) -> dict:
    """Return {signal_name: True/False} for ALL_SIGNALS."""
    cols = set(profile["raw_columns"])
    nfields = {m["normalized_field"] for m in mapping.values()}
    ds_id = target["dataset_id"]
    out: dict[str, bool] = {s: False for s in ALL_SIGNALS}

    # request_timestamps / arrivals
    if any(c in cols for c in (
        "request_timestamp", "Timestamp", "time", "request_arrival_ts_s",
        "created_at", "submit_time", "t",      # CC-traces uses 't' for arrival delta
        "pre_gap",                              # LMCache inter-turn gap
    )):
        out["request_timestamps"] = True
        out["arrivals"] = True

    # cache_reuse / prefix_reuse / kv_block_hashes
    if any(c in cols for c in (
        "block_hashes", "hash_ids", "cache_read_tokens", "cache_creation_tokens",
        "context_tokens",
    )):
        out["cache_reuse"] = True
        out["prefix_reuse"] = True
        if any(c in cols for c in ("block_hashes", "hash_ids")):
            out["kv_block_hashes"] = True
            out["migration_or_cache_loss_proxy"] = True
    # LMCache: session_id + input present → prefix-reuse signal via session affinity
    if ("session_id" in cols and "input" in cols
            and ds_id == "sammshen/lmcache-agentic-traces"):
        out["cache_reuse"] = True
        out["prefix_reuse"] = True

    # PrefixBench synthetic prefix engineering: prefix_reuse via metadata.scenario
    if "metadata" in cols and ds_id == "jaytonde05/prefixbench":
        out["cache_reuse"] = True
        out["prefix_reuse"] = True

    # autoscaling / capacity proxy
    if "type" in cols and ds_id.startswith("lsliwko/google-cluster-data"):
        out["autoscaling_proxy"] = True
        out["capacity_proxy"] = True
        # Borg SCHEDULE / EVICT events serve as model-load/unload proxy
        out["model_load_event"] = True
        out["model_unload_event"] = True
        # Migration proxy via EVICT + SCHEDULE pairs on different machine_ids
        out["migration_or_cache_loss_proxy"] = True
        out["routing_proxy"] = True

    # routing_proxy (CC-traces session affinity proxy)
    if any(c in cols for c in ("session_id", "machine_id")):
        out["routing_proxy"] = True

    # capacity proxy (BurstGPT arrival rate)
    if "Timestamp" in cols:
        out["capacity_proxy"] = True
        out["autoscaling_proxy"] = True  # arrival-driven autoscaling

    # workload_shape always present when arrivals + tokens are present
    if out["arrivals"] and any(c in cols for c in (
        "Request tokens", "prompt_tokens", "input_tokens", "in",
        "output_length", "input",
    )):
        out["workload_shape"] = True

    # customer_traffic_mix
    if any(c in cols for c in ("Log Type", "user", "source")):
        out["customer_traffic_mix"] = True

    # latency / ttft / tpot
    if any(c in cols for c in ("ttft_ms", "total_duration_ms", "ttft", "api_time")):
        out["latency"] = True
        if any(c in cols for c in ("ttft_ms", "ttft")):
            out["ttft"] = True

    return out


# ──────────────────────────────────────────────────────────────────────────
# Per-target audit driver
# ──────────────────────────────────────────────────────────────────────────


def _normalize_row(target: dict, raw: dict, mapping: dict) -> dict:
    """Project raw row → normalized row using mapping. Compress lists/dicts."""
    out: dict = {}
    for k, v in raw.items():
        m = mapping.get(k)
        if not m or not m.get("normalized_field"):
            # honest accounting; unknown columns surface in profile.rejected_columns
            continue
        nf = m["normalized_field"]
        # CC-traces hash_ids list -> compact summary
        if k == "hash_ids" and isinstance(v, list):
            out["block_hashes_count"] = len(v)
            out["block_hashes_hash"] = _hash_list(v)
            out["block_hashes_sample"] = _sample_list(v, 5)
            out[nf] = {"count": len(v), "hash": out["block_hashes_hash"]}
            continue
        # CC-traces session-level models list -> JSON
        if k == "models" and isinstance(v, list):
            out[nf] = json.dumps(v, sort_keys=True, default=str)[:200]
            continue
        # CC-traces requests array -> just emit its length (count was set above)
        if k == "requests" and isinstance(v, (list, int)):
            out[nf] = v if isinstance(v, int) else len(v)
            continue
        # Google Cluster nested resource_request
        if k == "resource_request" and isinstance(v, dict):
            try:
                out[nf] = json.dumps(v, sort_keys=True, default=str)
            except Exception:
                out[nf] = None
            continue
        if k == "constraint" and isinstance(v, list):
            try:
                out[nf] = json.dumps(v, sort_keys=True, default=str)[:500]
            except Exception:
                out[nf] = None
            continue
        if k == "metadata" and isinstance(v, dict):
            try:
                out[nf] = json.dumps(v, sort_keys=True, default=str)[:500]
            except Exception:
                out[nf] = None
            continue
        # LMCache: replace the huge 'input' prompt text with its char length
        if k == "input" and target["dataset_id"] == "sammshen/lmcache-agentic-traces":
            out[nf] = len(v) if isinstance(v, str) else 0
            continue
        # CC-traces / LMCache: drop large prompt/completion text bodies
        if k in ("prompt", "completion") and isinstance(v, str):
            out[nf + "_len"] = len(v)
            continue
        # PrefixBench: drop the long synthetic prompt body
        if k == "prompt" and target["dataset_id"] == "jaytonde05/prefixbench":
            out[nf + "_len"] = len(v) if isinstance(v, str) else 0
            continue
        # BurstGPT CSV strings -> numeric where possible
        if target["dataset_id"] == "lzzmm/BurstGPT" and isinstance(v, str):
            try:
                if "." in v:
                    out[nf] = float(v)
                else:
                    out[nf] = int(v)
                continue
            except ValueError:
                pass
        out[nf] = v
    return out


def _safe_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def audit_one(target: dict, *, token: str | None, force_redownload: bool) -> dict:
    dataset_id = target["dataset_id"]
    config = target["config_name"]
    hb = _Heartbeat(f"{dataset_id}@{config}")
    hb.update(phase="setup", force=True)
    safe_ds = _safe_dataset_dir(dataset_id)
    raw_path = safe_ds / "raw" / Path(target["raw_file"]).name
    processed_dir = safe_ds / config / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    schema_profile_path = processed_dir / "schema_profile.json"
    schema_mapping_path = processed_dir / "schema_mapping.json"
    summary_path = processed_dir / "summary.json"
    rollups_path = processed_dir / "statistical_rollups.json"
    analysis_sample_path = processed_dir / "analysis_sample.jsonl"
    fixture_path = FIXTURES_DIR / f"{dataset_id.replace('/', '__')}__{config}_sample.jsonl"

    url = _hf_url(dataset_id, target["raw_file"])

    # 1. Bounded download
    hb.update(phase="download", force=True)
    if force_redownload or not raw_path.exists():
        manifest = _bounded_download(
            url, raw_path,
            max_bytes=target["max_download_bytes"],
            token=token,
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
        return {"target": target, "manifest": manifest, "audit_status": "download_failed"}

    # 1b. Extra files (prefixbench_all)
    extra_manifests = []
    if target.get("extra_files"):
        for extra in target["extra_files"]:
            extra_url = _hf_url(dataset_id, extra)
            extra_path = safe_ds / "raw" / Path(extra).name
            if not extra_path.exists() or force_redownload:
                m = _bounded_download(
                    extra_url, extra_path,
                    max_bytes=target["max_download_bytes"], token=token,
                )
            else:
                m = {"url": extra_url, "dest": str(extra_path),
                     "downloaded_bytes": extra_path.stat().st_size,
                     "status": None, "truncated": None,
                     "error": None, "max_bytes": target["max_download_bytes"],
                     "cached": True}
            extra_manifests.append(m)

    # 2. Parse bounded rows — capped to ROW_CAP_FOR_NORMALIZATION
    hb.update(phase="parse", force=True)
    fmt = target["format"]
    if fmt == "jsonl":
        raw_rows = _read_jsonl(raw_path, drop_last_partial=True,
                               max_rows=ROW_CAP_FOR_NORMALIZATION, heartbeat=hb)
        # If there are extra files (prefixbench), parse them too — within the cap.
        for em in extra_manifests:
            if len(raw_rows) >= ROW_CAP_FOR_NORMALIZATION:
                break
            raw_rows.extend(_read_jsonl(
                Path(em["dest"]), drop_last_partial=True,
                max_rows=ROW_CAP_FOR_NORMALIZATION - len(raw_rows), heartbeat=hb,
            ))
    elif fmt == "jsonl_gz":
        raw_rows = _read_jsonl(raw_path, drop_last_partial=True, gz=True,
                               max_rows=ROW_CAP_FOR_NORMALIZATION, heartbeat=hb)
    elif fmt == "csv":
        raw_rows = _read_csv(raw_path, drop_last_partial=True,
                             max_rows=ROW_CAP_FOR_NORMALIZATION, heartbeat=hb)
    elif fmt == "parquet":
        raw_rows = _read_parquet(raw_path, max_rows=ROW_CAP_FOR_NORMALIZATION)
    else:
        return {"target": target, "manifest": manifest, "audit_status": "unsupported_format"}

    hb.update(phase="parsed", rows_done=len(raw_rows), force=True)
    if not raw_rows:
        return {"target": target, "manifest": manifest, "audit_status": "no_rows"}

    # 2b. Session-level → per-request flatten for CC-traces
    if dataset_id == "semianalysisai/cc-traces-weka-no-subagents-051226":
        flat_rows: list[dict] = []
        for sess in raw_rows:
            if not isinstance(sess, dict):
                continue
            session_id = sess.get("id")
            block_size = sess.get("block_size")
            hash_scope = sess.get("hash_id_scope")
            models = sess.get("models")
            reqs = sess.get("requests") or []
            for turn, req in enumerate(reqs):
                if not isinstance(req, dict):
                    continue
                req2 = dict(req)
                req2["id"] = session_id
                req2["block_size"] = block_size
                req2["hash_id_scope"] = hash_scope
                req2["models"] = models
                req2["requests"] = len(reqs)
                req2["turn"] = turn
                flat_rows.append(req2)
                if len(flat_rows) >= ROW_CAP_FOR_NORMALIZATION:
                    break
            if len(flat_rows) >= ROW_CAP_FOR_NORMALIZATION:
                break
            if len(flat_rows) % 5000 < 100:
                hb.update(rows_done=len(flat_rows))
        raw_rows = flat_rows
        hb.update(phase="parsed_flattened", rows_done=len(raw_rows), force=True)
        if not raw_rows:
            return {"target": target, "manifest": manifest, "audit_status": "no_rows"}

    # 3. Schema profile
    hb.update(phase="profile", force=True)
    profile = _profile_rows(raw_rows, dataset_id, config, target["raw_file"],
                            manifest["downloaded_bytes"])
    with open(schema_profile_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str, sort_keys=True)

    # 4. Schema mapping (classify every observed column)
    mapping = MAPPINGS.get((dataset_id, config), {})
    accepted = [c for c in profile["raw_columns"] if c in mapping]
    rejected = [c for c in profile["raw_columns"] if c not in mapping]
    column_records = []
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
        "rejected_columns": sorted(rejected),
        "columns": column_records,
    }
    with open(schema_mapping_path, "w") as fh:
        json.dump(mapping_doc, fh, indent=2, default=str, sort_keys=True)

    # 5. Normalize all rows
    hb.update(phase="normalize", force=True)
    normalized: list[dict] = []
    for i, r in enumerate(raw_rows):
        normalized.append(_normalize_row(target, r, mapping))
        if i % 10000 == 0:
            hb.update(rows_done=i + 1)
    hb.update(rows_done=len(normalized), force=True)
    normalized_schema = sorted({k for r in normalized for k in r.keys()})

    # 6. Analysis sample (gitignored)
    hb.update(phase="write_analysis_sample", force=True)
    analysis_bytes, analysis_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], analysis_sample_path,
    )

    # 7. Fixture sample (5 deterministic rows; size-capped at 16 KiB)
    fixture_rows = [_safe_jsonable(r) for r in normalized[:5]]
    fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
    # Sanity guard: cap fixture size
    while fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES and fixture_rows:
        fixture_rows = fixture_rows[: max(1, len(fixture_rows) - 1)]
        fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
        if len(fixture_rows) == 1 and fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES:
            # Drop bulky string fields
            slim = {k: v for k, v in fixture_rows[0].items()
                    if not isinstance(v, str) or len(v) < 200}
            fixture_rows = [slim]
            fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
            break

    # 8. Signal coverage + sample strength
    strength = _statistical_sample_strength(len(normalized))
    signals_detected = _detect_signals(target, profile, mapping, raw_rows)

    # 9. Available + missing signals (canonical names from the corpus signal taxonomy).
    available_signals: list[str] = []
    for s, present in signals_detected.items():
        if present:
            available_signals.append(s)
    available_signals = sorted(available_signals)
    missing_signals = sorted(s for s, present in signals_detected.items() if not present)

    # 10. Field-quality groupings
    real_fields = sorted([c["normalized_field"] for c in column_records
                          if c["field_quality"] == "real" and c["normalized_field"]])
    derived_fields = sorted([c["normalized_field"] for c in column_records
                             if c["field_quality"] == "derived" and c["normalized_field"]])
    proxy_fields = sorted([c["normalized_field"] for c in column_records
                           if c["field_quality"] == "proxy" and c["normalized_field"]])
    synthetic_fields = sorted([c["normalized_field"] for c in column_records
                               if c["field_quality"] == "synthetic" and c["normalized_field"]])

    field_quality = {c["normalized_field"]: c["field_quality"]
                     for c in column_records if c["normalized_field"]}

    # 11. Statistical rollups (lightweight, per-stratification-key counts +
    # numeric distribution for any duration / tokens field)
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}
    for skey in target["stratification_keys"]:
        # Use raw column name if the normalized field doesn't match (Borg "type" etc.)
        if skey in profile["raw_columns"]:
            counts: dict = {}
            for r in raw_rows:
                v = r.get(skey)
                if isinstance(v, (list, dict)):
                    v = "complex"
                counts[str(v)] = counts.get(str(v), 0) + 1
            rollups["subgroup_counts"][skey] = counts
    # Numeric distribution: pick a couple of normalized numeric fields per dataset
    NUMERIC_FIELDS = {
        ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head"):
            ("input_tokens", "output_tokens", "request_arrival_delta_s",
             "block_hashes_count", "turn_index", "requests_count"),
        ("sammshen/lmcache-agentic-traces", "train_shard4"):
            ("output_tokens", "pre_gap_s", "input_len"),
        ("lzzmm/BurstGPT", "burstgpt_1_full"):
            ("input_tokens", "output_tokens", "total_tokens"),
        ("lsliwko/google-cluster-data-2019-sorted-by-timestamp", "instance_events_shard0"):
            ("event_time_us", "priority", "scheduling_class"),
        ("jaytonde05/prefixbench", "prefixbench_all"):
            ("max_tokens", "prompt_text_len"),
    }
    for nf in NUMERIC_FIELDS.get((dataset_id, config), ()):
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
    summary = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": target["license"],
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
        "unknown_columns": rejected,
        "field_quality": field_quality,
        "available_signals": available_signals,
        "missing_signals": missing_signals,
        "real_fields": real_fields,
        "derived_fields": derived_fields,
        "proxy_fields": proxy_fields,
        "synthetic_fields": synthetic_fields,
        "limitations": target["limitations"],
        "provenance": (
            f"{dataset_id}@{config}#{target['raw_file']}"
            f"#bytes={manifest['downloaded_bytes']}#git={(_git_sha() or '')[:7]}"
        ),
        "ingestion_timestamp_s": time.time(),
        "git_sha": _git_sha(),
        "raw_download_manifest": manifest,
        "extra_files_manifest": extra_manifests,
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
    hb.update(phase="promotion", force=True)
    decision = promotion.evaluate_promotion(summary)
    hb.update(phase="done", force=True)
    return {
        "dataset_id": dataset_id,
        "config_name": config,
        "manifest": manifest,
        "extra_files_manifest": extra_manifests,
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
    p.add_argument("--only", default=None, help="dataset_id substring filter")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN env var required")
        return 2

    targets = TARGETS
    if args.only:
        targets = [t for t in TARGETS if args.only in t["dataset_id"]]
        if not targets:
            logger.error("--only filter matched no targets")
            return 2

    results = []
    for t in targets:
        logger.info("ingest %s/%s ... (timeout %ds)", t["dataset_id"],
                    t["config_name"], PER_DATASET_TIMEOUT_S)
        t_start = time.monotonic()
        try:
            _install_timeout(PER_DATASET_TIMEOUT_S)
            r = audit_one(t, token=token, force_redownload=args.force_redownload)
        except _PerDatasetTimeout as e:
            elapsed = int(time.monotonic() - t_start)
            logger.error("  DEFERRED_TIMEOUT after %ds: %s", elapsed, e)
            r = {
                "dataset_id": t["dataset_id"],
                "config_name": t["config_name"],
                "audit_status": "DEFERRED_TIMEOUT",
                "error": str(e),
                "elapsed_s": elapsed,
            }
        except Exception as e:  # noqa: BLE001
            elapsed = int(time.monotonic() - t_start)
            logger.error("  FAILED after %ds: %s", elapsed, e)
            r = {
                "dataset_id": t["dataset_id"],
                "config_name": t["config_name"],
                "audit_status": "FAILED",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": elapsed,
            }
        finally:
            _clear_timeout()
        results.append(r)
        status = r.get("audit_status")
        if status == "ok":
            s = r["summary"]
            d = r["decision"]
            logger.info(
                "  rows=%d bytes_sampled=%d strength=%s state=%s tags=%s elapsed=%ds",
                s["analysis_sample_rows"], s["analysis_sample_bytes"],
                s["statistical_sample_strength"], d["state"], d["promotion_tags"],
                int(time.monotonic() - t_start),
            )
        else:
            logger.info("  status=%s (%s)", status, r.get("error"))

    # Cross-dataset summary
    DISC_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = DISC_DIR / "telemetry_gap_ingest_summary.json"
    payload = {
        "doc_version": "telemetry_gap_ingest_summary_v1",
        "stage": "discovery_and_bounded_ingest",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "ingested_at_s": time.time(),
        "git_sha": _git_sha(),
        "ingested": [
            {
                "dataset_id": r["dataset_id"],
                "config_name": r["config_name"],
                "rows_sampled": r["summary"]["analysis_sample_rows"],
                "bytes_sampled": r["summary"]["analysis_sample_bytes"],
                "raw_bytes_downloaded": r["summary"]["raw_download_manifest"]["downloaded_bytes"],
                "strength": r["summary"]["statistical_sample_strength"],
                "trace_type": r["summary"]["canonical_trace_type"],
                "promotion_state": r["decision"]["state"],
                "promotion_tags": r["decision"]["promotion_tags"],
                "available_signals": r["summary"]["available_signals"],
                "missing_signals": r["summary"]["missing_signals"],
                "limitations": r["summary"]["limitations"],
                "license": r["summary"]["license"],
                "url": r["summary"]["source_url"],
                "summary_path": r["summary"]["summary_path_relative"],
            }
            for r in results if r.get("audit_status") == "ok"
        ],
        "failed": [
            {"dataset_id": r.get("dataset_id"),
             "config_name": r.get("config_name"),
             "audit_status": r.get("audit_status"),
             "error": r.get("error"),
             "elapsed_s": r.get("elapsed_s"),
             "manifest": r.get("manifest")}
            for r in results if r.get("audit_status") != "ok"
        ],
    }
    with open(summary_out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)
    logger.info("Wrote %s", summary_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
