#!/usr/bin/env python3
"""Bounded ingest of Lightcap/agent-runtime-telemetry-small.

A small 2026-04-21 cc-by-4.0 export of MCP-style agent-runtime tool-call
execution telemetry by Faruk Alpay (Lightcap). 8 parquet configs, all
small. The two configs ingested here:

- ``operations`` — 2,262 × 33: one row per tool call with real measured
  ``duration_ms`` + ``status`` + ``stage`` + ``error_type`` +
  ``created_at_utc`` + ``updated_at_utc`` + per-call args / kwargs / result
  fingerprints. The primary tool_runtime_trace evidence.
- ``tool_summary`` — 32 × 8: per-(tool_name, status) aggregated p50 /
  median / p95 latency + first/last seen. Pre-rolled latency priors at
  the per-tool grain, useful as a routing-quality / SLA-margin prior
  without needing to recompute over operations.

Maps onto the new ``tool_runtime_trace`` canonical type
(``aurelius/traces/hf_corpus/schemas.py::ToolRuntimeRecord``).

Trust: Tier 3 (real measured execution telemetry — job-trace shape, but
the "jobs" are MCP tool calls, not GPU jobs). NOT serving telemetry —
no model_id / no input_tokens / no GPU / no queue / no replica / no
cache state. Value to Aurelius: routing-quality + failure-rate +
tail-latency priors for agent workloads.

Audit-only. Does NOT modify scheduler / controllers / robust energy
engine. Raw downloads are gitignored under
``data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/``; only
schema_profile, schema_mapping, summary, statistical_rollups, the
tiny fixture, and the bounded normalized sample (cc-by-4.0 permits
redistribution) are committed.

Layout::

    data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/<file>   # gitignored
    data/external/hf/Lightcap__agent-runtime-telemetry-small/<config>/processed/
        schema_profile.json
        schema_mapping.json
        summary.json
        statistical_rollups.json
        normalized_sample.jsonl                                           # committed
        analysis_sample.jsonl                                             # gitignored
    tests/fixtures/hf/Lightcap__agent-runtime-telemetry-small__<config>_sample.jsonl
"""
from __future__ import annotations

import argparse
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

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")

MAX_COMMITTED_FIXTURE_BYTES = 16 * 1024
MAX_COMMITTED_NORMALIZED_BYTES = 100 * 1024 * 1024  # 100 MiB per the policy

PER_DATASET_TIMEOUT_S = 30 * 60
PROGRESS_INTERVAL_S = 30

logger = logging.getLogger("aurelius.hf_lightcap_runtime_ingest")


# ---------------------------------------------------------------------------
# Heartbeat / timeout
# ---------------------------------------------------------------------------


class _Heartbeat:
    def __init__(self, label: str):
        self.label = label
        self.start = time.monotonic()
        self.last_log = self.start
        self.phase = "init"
        self.bytes_done = 0
        self.rows_done = 0

    def update(self, *, phase: str | None = None, bytes_done: int | None = None,
               rows_done: int | None = None, force: bool = False) -> None:
        if phase is not None:
            self.phase = phase
        if bytes_done is not None:
            self.bytes_done = bytes_done
        if rows_done is not None:
            self.rows_done = rows_done
        now = time.monotonic()
        if force or (now - self.last_log) >= PROGRESS_INTERVAL_S:
            elapsed = int(now - self.start)
            logger.info("  [hb] %s phase=%s bytes=%d rows=%d elapsed=%ds",
                        self.label, self.phase, self.bytes_done,
                        self.rows_done, elapsed)
            self.last_log = now


class _PerDatasetTimeout(Exception):  # noqa: N818
    pass


def _install_timeout(seconds: int) -> None:
    def _h(_signo, _frame):
        raise _PerDatasetTimeout(f"per-dataset timeout after {seconds}s")
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)


def _clear_timeout() -> None:
    signal.alarm(0)


# ---------------------------------------------------------------------------
# Per-config target table
# ---------------------------------------------------------------------------


TARGETS: list[dict] = [
    {
        "config_name": "operations",
        "raw_file": "data/operations.parquet",
        "expected_raw_bytes": 280_000,
        "primary": True,
    },
    {
        "config_name": "tool_summary",
        "raw_file": "data/tool_summary.parquet",
        "expected_raw_bytes": 9_000,
        "primary": False,
    },
    {
        "config_name": "operation_events",
        "raw_file": "data/operation_events.parquet",
        "expected_raw_bytes": 620_000,
        "primary": False,
    },
    {
        "config_name": "audit_records",
        "raw_file": "data/audit_records.parquet",
        "expected_raw_bytes": 2_300_000,
        "primary": False,
    },
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _hf_url(dataset_id: str, raw_file: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(raw_file)}"
    )


def _bounded_download(url: str, dest: Path, *, max_bytes: int | None,
                      token: str | None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-hf-lightcap-runtime-ingest/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{int(max_bytes - 1)}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    written = 0
    status = None
    truncated = False
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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
        return {"url": url, "dest": str(dest), "status": int(e.code),
                "downloaded_bytes": 0, "truncated": False,
                "error": f"HTTPError {e.code}: {e.reason}",
                "max_bytes": max_bytes}
    except urllib.error.URLError as e:
        return {"url": url, "dest": str(dest), "status": None,
                "downloaded_bytes": 0, "truncated": False,
                "error": f"URLError: {e.reason}",
                "max_bytes": max_bytes}
    return {"url": url, "dest": str(dest), "status": status,
            "downloaded_bytes": written, "truncated": truncated,
            "max_bytes": max_bytes, "error": None}


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _iso_to_epoch_s(iso: str | None) -> float | None:
    if not iso or not isinstance(iso, str):
        return None
    try:
        from datetime import datetime
        s = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


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
        line = json.dumps(r, sort_keys=True, separators=(",", ":"),
                          default=str) + "\n"
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


# ---------------------------------------------------------------------------
# Normalisation: operations row → ToolRuntimeRecord-shaped dict
# ---------------------------------------------------------------------------


def _normalize_operations_row(raw: dict) -> dict:
    """Map one raw operations.parquet row → flat ToolRuntimeRecord-shaped row."""
    status = (raw.get("status") or "") or None
    stage = (raw.get("stage") or "") or None
    error_type = (raw.get("error_type") or "") or None
    created_iso = raw.get("created_at_utc")
    updated_iso = raw.get("updated_at_utc")
    created_s = _iso_to_epoch_s(created_iso)
    updated_s = _iso_to_epoch_s(updated_iso)
    duration_ms = raw.get("duration_ms")
    duration_s = None
    if isinstance(duration_ms, (int, float)):
        duration_s = float(duration_ms) / 1000.0
    is_error = (status == "error") if status is not None else None
    is_cancelled = (status == "cancelled") if status is not None else None
    return {
        # Identity + routing
        "operation_id": raw.get("operation_id"),
        "request_id": raw.get("request_id"),
        "tool_name": raw.get("tool_name"),
        "stage": stage,
        "status": status,
        "operation_mode": (raw.get("operation_mode") or "") or None,
        "backend_preference": (raw.get("backend_preference") or "") or None,
        # Timestamps
        "created_at_iso": created_iso,
        "updated_at_iso": updated_iso,
        "created_at_s": created_s,
        "updated_at_s": updated_s,
        "duration_ms": duration_ms,
        "duration_s": duration_s,
        # Failure / cancel labels
        "error_type": error_type,
        "error_message_preview": (raw.get("error_message_preview") or "") or None,
        "error_message_sha256": (raw.get("error_message_sha256") or "") or None,
        "is_error": is_error,
        "is_cancelled": is_cancelled,
        # Payload-size proxies
        "args_fingerprint": (raw.get("args_fingerprint") or "") or None,
        "args_count": raw.get("args_count"),
        "args_keys": (raw.get("args_keys") or "") or None,
        "kwargs_key_count": raw.get("kwargs_key_count"),
        "kwargs_keys": (raw.get("kwargs_keys") or "") or None,
        "result_summary_key_count": raw.get("result_summary_key_count"),
        "result_summary_keys": (raw.get("result_summary_keys") or "") or None,
        "result_type": (raw.get("result_type") or "") or None,
        "result_operation": (raw.get("result_operation") or "") or None,
        "result_payload_key_count": raw.get("result_payload_key_count"),
        "result_payload_keys": (raw.get("result_payload_keys") or "") or None,
        "result_payload_bytes": raw.get("result_payload_bytes"),
        "artifacts_bytes": raw.get("artifacts_bytes"),
        # Provenance flags
        "force_retrain": raw.get("force_retrain"),
        "include_control_sensitivities": raw.get("include_control_sensitivities"),
        "include_validation_protocols": raw.get("include_validation_protocols"),
        "has_input_provenance": raw.get("has_input_provenance"),
        "has_source_binding": raw.get("has_source_binding"),
        "series_rows_count": raw.get("series_rows_count"),
        "scenario_rows_count": raw.get("scenario_rows_count"),
    }


def _normalize_operation_events_rows(raw_rows: list[dict]) -> list[dict]:
    """Map all raw operation_events.parquet rows → flat ToolRuntimeRecord-shaped
    rows, with ``duration_ms`` computed per-event as ms-since the operation's
    ``started`` event (queue-wait-style prior).

    Each operation_id has 2-8 lifecycle events ordered by ``event_time_utc``;
    the per-event ``duration_ms`` lets the constraint-aware harness read off
    dispatch latency (``started -> stage(executing)``), execution latency
    (``stage(executing) -> stage(execution_completed)``), and post-processing
    latency (``stage(execution_completed) -> completed``) directly without
    re-grouping. ``field_quality`` for ``duration_ms`` is therefore
    ``derived`` (computed from real measured ``event_time_utc`` timestamps).
    """
    # First pass: per-operation_id, find the earliest event_time_utc.
    started_at_s: dict[str, float] = {}
    for r in raw_rows:
        op = r.get("operation_id")
        if not op:
            continue
        ts = _iso_to_epoch_s(r.get("event_time_utc"))
        if ts is None:
            continue
        cur = started_at_s.get(op)
        if cur is None or ts < cur:
            started_at_s[op] = ts

    out: list[dict] = []
    for r in raw_rows:
        op = r.get("operation_id")
        event_time_iso = r.get("event_time_utc")
        event_time_s = _iso_to_epoch_s(event_time_iso)
        start_s = started_at_s.get(op) if op else None
        if event_time_s is not None and start_s is not None:
            duration_ms = max(0.0, (event_time_s - start_s) * 1000.0)
            duration_s = duration_ms / 1000.0
        else:
            duration_ms = None
            duration_s = None
        status = (r.get("status") or "") or None
        stage = (r.get("stage") or "") or None
        event_type = (r.get("event_type") or "") or None
        is_error = (status == "error") if status is not None else None
        is_cancelled = (status == "cancelled") if status is not None else None
        # payload_tool is only populated on 'started' events; treat as
        # tool_name when present.
        payload_tool = (r.get("payload_tool") or "") or None
        out.append({
            "operation_id": op,
            "request_id": None,  # operation_events does not carry MCP request_id
            "tool_name": payload_tool,
            "stage": stage,
            "status": status,
            "operation_mode": None,
            "backend_preference": None,
            "created_at_iso": event_time_iso,
            "updated_at_iso": event_time_iso,
            "created_at_s": event_time_s,
            "updated_at_s": event_time_s,
            "duration_ms": duration_ms,
            "duration_s": duration_s,
            "event_id": r.get("event_id"),
            "event_type": event_type,
            "payload_bytes": r.get("payload_bytes"),
            "payload_sha256": (r.get("payload_sha256") or "") or None,
            "payload_key_count": r.get("payload_key_count"),
            "payload_keys": (r.get("payload_keys") or "") or None,
            "payload_status": (r.get("payload_status") or "") or None,
            "payload_stage": (r.get("payload_stage") or "") or None,
            "is_error": is_error,
            "is_cancelled": is_cancelled,
        })
    return out


def _normalize_audit_records_row(raw: dict) -> dict:
    """Map one raw audit_records.parquet row → flat ToolRuntimeRecord-shaped row.

    MCP-shell-layer audit records: category in {tool_requests, tool_results};
    ``duration_ms`` is real (populated only on tool_results); ``status`` and
    ``response_*`` fields are populated only on results; ``payload_*`` is the
    audit-record payload (request body for tool_requests; response body for
    tool_results).
    """
    duration_ms = raw.get("duration_ms")
    duration_s = (float(duration_ms) / 1000.0
                  if isinstance(duration_ms, (int, float)) else None)
    status = (raw.get("status") or "") or None
    created_iso = raw.get("created_at_utc")
    created_s = _iso_to_epoch_s(created_iso)
    is_error = (status == "error") if status is not None else None
    is_cancelled = (status == "cancelled") if status is not None else None
    return {
        "operation_id": None,
        "request_id": (raw.get("request_id") or "") or None,
        "tool_name": (raw.get("tool") or "") or None,
        "stage": None,
        "status": status,
        "operation_mode": None,
        "backend_preference": None,
        "created_at_iso": created_iso,
        "updated_at_iso": created_iso,
        "created_at_s": created_s,
        "updated_at_s": created_s,
        "duration_ms": duration_ms,
        "duration_s": duration_s,
        "record_id": raw.get("record_id"),
        "category": (raw.get("category") or "") or None,
        "record_name": (raw.get("record_name") or "") or None,
        "record_file": (raw.get("record_file") or "") or None,
        "record_path_scope": (raw.get("record_path_scope") or "") or None,
        "kind": (raw.get("kind") or "") or None,
        "payload_bytes": raw.get("payload_bytes"),
        "payload_sha256": (raw.get("payload_sha256") or "") or None,
        "payload_key_count": raw.get("payload_key_count"),
        "payload_keys": (raw.get("payload_keys") or "") or None,
        "response_key_count": raw.get("response_key_count"),
        "response_keys": (raw.get("response_keys") or "") or None,
        "is_error": is_error,
        "is_cancelled": is_cancelled,
    }


def _normalize_tool_summary_row(raw: dict) -> dict:
    """Map one raw tool_summary.parquet row → flat row.

    The aggregated table only has (tool_name, status, count, avg/median/p95
    durations, first/last seen). To fit the same ToolRuntimeRecord schema
    every committed row carries the same field set; aggregate-only rows
    leave per-call fields null and embed the rollup statistics as
    derived-quality duration fields ``duration_ms`` (set to median for
    schema compatibility) plus the explicit ``p95_duration_ms`` /
    ``avg_duration_ms`` reported in statistical_rollups (NOT in the
    normalized sample — those live in statistical_rollups.json).
    """
    created_iso = raw.get("first_seen_utc")
    updated_iso = raw.get("last_seen_utc")
    created_s = _iso_to_epoch_s(created_iso)
    updated_s = _iso_to_epoch_s(updated_iso)
    # We store the aggregate median as duration_ms (field_quality=derived)
    # so the row is still a valid ToolRuntimeRecord shape. The avg + p95
    # live in statistical_rollups.json, not in normalized sample rows.
    median = raw.get("median_duration_ms")
    return {
        "operation_id": None,  # aggregate row, no per-call op id
        "request_id": None,
        "tool_name": raw.get("tool_name"),
        "stage": None,
        "status": raw.get("status"),
        "operation_mode": None,
        "backend_preference": None,
        "created_at_iso": created_iso,
        "updated_at_iso": updated_iso,
        "created_at_s": created_s,
        "updated_at_s": updated_s,
        "duration_ms": median,
        "duration_s": (float(median) / 1000.0) if isinstance(median, (int, float))
                        else None,
        "error_type": None,
        "error_message_preview": None,
        "error_message_sha256": None,
        "is_error": (raw.get("status") == "error") if raw.get("status") else None,
        "is_cancelled": (raw.get("status") == "cancelled") if raw.get("status")
                        else None,
        "args_fingerprint": None,
        "args_count": None,
        "args_keys": None,
        "kwargs_key_count": None,
        "kwargs_keys": None,
        "result_summary_key_count": None,
        "result_summary_keys": None,
        "result_type": None,
        "result_operation": None,
        "result_payload_key_count": None,
        "result_payload_keys": None,
        "result_payload_bytes": None,
        "artifacts_bytes": None,
        "force_retrain": None,
        "include_control_sensitivities": None,
        "include_validation_protocols": None,
        "has_input_provenance": None,
        "has_source_binding": None,
        "series_rows_count": None,
        "scenario_rows_count": None,
    }


# ---------------------------------------------------------------------------
# Schema mapping table (raw column → normalized field + quality)
# ---------------------------------------------------------------------------


OPERATIONS_MAPPING: dict = {
    "operation_id": {
        "normalized_field": "operation_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Tool-call identifier (e.g. 'deploy-smoke-1776349977').",
    },
    "request_id": {
        "normalized_field": "request_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality"],
        "notes": "MCP request UUID. Joins to audit_records.request_id.",
    },
    "tool_name": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality", "workload_shape_only"],
        "notes": "Tool routing key (e.g. surface_affinity, workflow_run). "
                 "22 distinct tools in this export.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Terminal status: ok / error / cancelled. ~5.5% error rate "
                 "in this export.",
    },
    "stage": {
        "normalized_field": "stage", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Lifecycle stage at terminal time: completed / failed / "
                 "affinity_rejected / startup_reconciled.",
    },
    "duration_ms": {
        "normalized_field": "duration_ms", "field_quality": "real",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "Real measured end-to-end tool-call duration in ms. Closed "
                 "tool-runtime timing — NOT GPU TTFT/TPOT.",
    },
    "created_at_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO arrival timestamp; epoch seconds derived as "
                 "created_at_s.",
    },
    "updated_at_utc": {
        "normalized_field": "updated_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO completion timestamp; epoch seconds derived as "
                 "updated_at_s.",
    },
    "args_fingerprint": {
        "normalized_field": "args_fingerprint", "field_quality": "real",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "sha256 of args; same fingerprint = same args = potential "
                 "cache-reuse signal.",
    },
    "args_count": {
        "normalized_field": "args_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Positional argument count (payload-shape proxy).",
    },
    "args_keys": {
        "normalized_field": "args_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited arg key list (payload-shape proxy).",
    },
    "kwargs_key_count": {
        "normalized_field": "kwargs_key_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Keyword-argument key count (payload-shape proxy).",
    },
    "kwargs_keys": {
        "normalized_field": "kwargs_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited kwarg key list (payload-shape proxy).",
    },
    "operation_mode": {
        "normalized_field": "operation_mode", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality"],
        "notes": "Per-tool operation mode (where the tool supports modes).",
    },
    "backend_preference": {
        "normalized_field": "backend_preference", "field_quality": "real",
        "aurelius_signal_category": "routing",
        "usable_for": ["routing_quality"],
        "notes": "Routing preference recorded by the agent runtime (where set).",
    },
    "force_retrain": {
        "normalized_field": "force_retrain", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag (granite_timeseries family).",
    },
    "include_control_sensitivities": {
        "normalized_field": "include_control_sensitivities",
        "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag.",
    },
    "include_validation_protocols": {
        "normalized_field": "include_validation_protocols",
        "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag.",
    },
    "has_input_provenance": {
        "normalized_field": "has_input_provenance", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Whether the operation declared input provenance.",
    },
    "has_source_binding": {
        "normalized_field": "has_source_binding", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Whether the operation declared a source binding.",
    },
    "series_rows_count": {
        "normalized_field": "series_rows_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-call input series-row count (payload-shape proxy).",
    },
    "scenario_rows_count": {
        "normalized_field": "scenario_rows_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-call input scenario-row count (payload-shape proxy).",
    },
    "result_summary_key_count": {
        "normalized_field": "result_summary_key_count",
        "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Output result summary-key count (output-shape proxy).",
    },
    "result_summary_keys": {
        "normalized_field": "result_summary_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited result summary key list.",
    },
    "result_type": {
        "normalized_field": "result_type", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Result top-level type (object / list / ...).",
    },
    "result_operation": {
        "normalized_field": "result_operation", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Result-recorded operation label (provider-specific).",
    },
    "result_payload_key_count": {
        "normalized_field": "result_payload_key_count",
        "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Output result payload-key count.",
    },
    "result_payload_keys": {
        "normalized_field": "result_payload_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited result payload key list.",
    },
    "result_payload_bytes": {
        "normalized_field": "result_payload_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Real measured result payload byte count (output-size proxy).",
    },
    "artifacts_bytes": {
        "normalized_field": "artifacts_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Real measured side-effect artifact byte count.",
    },
    "error_type": {
        "normalized_field": "error_type", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Python exception class for status=error. 8 distinct error "
                 "types in this export (TimeoutError / RuntimeError / "
                 "ValueError / SurfaceAffinityError / ...).",
    },
    "error_message_preview": {
        "normalized_field": "error_message_preview", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Bounded error-message preview (string, redacted upstream).",
    },
    "error_message_sha256": {
        "normalized_field": "error_message_sha256", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Full-message hash for grouping; raw message not retained.",
    },
}


TOOL_SUMMARY_MAPPING: dict = {
    "tool_name": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality"],
        "notes": "Tool routing key.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Bucket status (ok / error / cancelled).",
    },
    "operation_count": {
        "normalized_field": "result_summary_key_count",
        "field_quality": "derived",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Number of operations rolled into this (tool, status) "
                 "bucket. Stored in statistical_rollups, not normalized "
                 "sample.",
    },
    "avg_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "Mean tool-call duration in ms for this (tool, status) "
                 "bucket — derived from operations rows.",
    },
    "median_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "Median tool-call duration in ms — committed as the "
                 "row-level duration_ms in tool_summary normalized sample.",
    },
    "p95_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "p95 tool-call duration in ms — recorded in "
                 "statistical_rollups.json for direct routing-quality use.",
    },
    "first_seen_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["workload_shape_only"],
        "notes": "First operation timestamp per (tool, status) bucket.",
    },
    "last_seen_utc": {
        "normalized_field": "updated_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["workload_shape_only"],
        "notes": "Last operation timestamp per (tool, status) bucket.",
    },
}


OPERATION_EVENTS_MAPPING: dict = {
    "event_id": {
        "normalized_field": "event_id", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-event sequence id (1..N). Used for stable ordering "
                 "within a single export.",
    },
    "operation_id": {
        "normalized_field": "operation_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Joins to operations.operation_id. Each operation has 2-8 "
                 "lifecycle events ordered by event_time_utc.",
    },
    "event_type": {
        "normalized_field": "event_type", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Lifecycle event type: started / stage / completed / "
                 "failed / reconciled. The started/completed pair bounds the "
                 "operation's wall-clock; stage events expose dispatch + "
                 "execution + post-processing transitions.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Per-event status: running / ok / error / cancelled. "
                 "running on transient events; ok / error / cancelled on "
                 "terminal events.",
    },
    "stage": {
        "normalized_field": "stage", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Lifecycle stage: started / executing / execution_completed "
                 "/ completed / failed / accepted / affinity_rejected / "
                 "affinity_warning / artifacts_published / startup_reconciled.",
    },
    "event_time_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO event timestamp. Per-event duration_ms is "
                 "derived as (event_time_utc - started_event_time_utc) for "
                 "the same operation_id — that's the dispatch-stage + "
                 "execution-stage latency prior.",
    },
    "payload_bytes": {
        "normalized_field": "payload_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Real measured per-event payload byte count. started "
                 "events carry the args fingerprint; completed events carry "
                 "the result summary.",
    },
    "payload_sha256": {
        "normalized_field": "payload_sha256", "field_quality": "real",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "sha256 of the per-event payload body. Same fingerprint "
                 "across operations = identical event payload (cache-reuse "
                 "proxy at the event grain).",
    },
    "payload_key_count": {
        "normalized_field": "payload_key_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-event payload key count (payload-shape proxy).",
    },
    "payload_keys": {
        "normalized_field": "payload_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited per-event payload key list.",
    },
    "payload_status": {
        "normalized_field": "payload_status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Status string carried inside the event payload (often "
                 "duplicates row-level status on terminal events; blank "
                 "on transient stage events).",
    },
    "payload_stage": {
        "normalized_field": "payload_stage", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality"],
        "notes": "Stage string carried inside the event payload (often "
                 "duplicates row-level stage; blank on transient events).",
    },
    "payload_tool": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality", "workload_shape_only"],
        "notes": "Tool routing key column in the raw schema. In this export "
                 "payload_tool is always blank — the tool name lives inside "
                 "the started-event payload body (one of payload_keys = "
                 "'public_task_state|request_id|tool_name') which is NOT "
                 "redistributed in this export. Tool resolution requires "
                 "joining to operations.tool_name via operation_id.",
    },
}


AUDIT_RECORDS_MAPPING: dict = {
    "record_id": {
        "normalized_field": "record_id", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-record sequence id (1..N).",
    },
    "category": {
        "normalized_field": "category", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Audit-record category: tool_requests (inbound MCP "
                 "request) or tool_results (outbound MCP response). The "
                 "pair shares request_id; results carry duration_ms.",
    },
    "record_name": {
        "normalized_field": "record_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Audit-record name (typically '{tool}_{request_id}').",
    },
    "record_file": {
        "normalized_field": "record_file", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Audit-record filename ('{timestamp}_{record_name}.json').",
    },
    "record_path_scope": {
        "normalized_field": "record_path_scope", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Audit-record file path scope (e.g. 'workspace').",
    },
    "tool": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality", "workload_shape_only"],
        "notes": "Tool routing key. Same set as operations.tool_name.",
    },
    "kind": {
        "normalized_field": "kind", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality"],
        "notes": "MCP message kind: mcp_tool_request or mcp_tool_result.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Terminal status (only populated on tool_results: ok / "
                 "error / accepted / running). Blank on tool_requests.",
    },
    "duration_ms": {
        "normalized_field": "duration_ms", "field_quality": "real",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "Real measured MCP-shell-layer duration in ms. Populated "
                 "only on tool_results rows (50% of audit_records). Heavy "
                 "tail: p99=2.5 s, max=900 s in this export.",
    },
    "request_id": {
        "normalized_field": "request_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "MCP request UUID. Joins to operations.request_id. Each "
                 "request_id has 2 audit records: one tool_requests + one "
                 "tool_results.",
    },
    "payload_bytes": {
        "normalized_field": "payload_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Real measured audit-record payload byte count. For "
                 "tool_requests: request-body bytes. For tool_results: "
                 "response-body bytes.",
    },
    "payload_sha256": {
        "normalized_field": "payload_sha256", "field_quality": "real",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "sha256 of the audit-record payload body. Same fingerprint "
                 "across requests = identical request body (cache-reuse "
                 "proxy at the MCP shell layer).",
    },
    "payload_key_count": {
        "normalized_field": "payload_key_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Audit-record payload key count (payload-shape proxy).",
    },
    "payload_keys": {
        "normalized_field": "payload_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited audit-record payload key list.",
    },
    "response_key_count": {
        "normalized_field": "response_key_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Audit-record response key count. Populated only on "
                 "tool_results rows.",
    },
    "response_keys": {
        "normalized_field": "response_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited response key list. Populated only on "
                 "tool_results rows.",
    },
    "created_at_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO arrival timestamp; epoch seconds derived as "
                 "created_at_s.",
    },
}


CONFIG_MAPPINGS = {
    "operations": OPERATIONS_MAPPING,
    "tool_summary": TOOL_SUMMARY_MAPPING,
    "operation_events": OPERATION_EVENTS_MAPPING,
    "audit_records": AUDIT_RECORDS_MAPPING,
}

CONFIG_NORMALIZERS = {
    "operations": _normalize_operations_row,
    "tool_summary": _normalize_tool_summary_row,
    # operation_events normalisation needs a list-level view (per-operation
    # 'started' timestamp lookup) so the entry is the batch normaliser. The
    # audit driver wraps single-row normalisers in a list-comprehension; the
    # batch entry is dispatched on a per-config branch in ``audit_one``.
    "operation_events": _normalize_operation_events_rows,
    "audit_records": _normalize_audit_records_row,
}


# ---------------------------------------------------------------------------
# Schema profile (raw + normalized)
# ---------------------------------------------------------------------------


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


def _profile_rows(raw_rows: list[dict], normalized: list[dict],
                  config: str, raw_file: str, file_size_bytes: int) -> dict:
    raw_cols: dict[str, dict] = {}
    n_raw = len(raw_rows)
    for r in raw_rows:
        for k, v in r.items():
            c = raw_cols.setdefault(k, {"present": 0, "types": set(),
                                        "examples": []})
            c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                if isinstance(v, (dict, list)):
                    try:
                        c["examples"].append(json.dumps(v, default=str)[:120])
                    except Exception:
                        c["examples"].append(repr(v)[:120])
                else:
                    c["examples"].append(repr(v)[:120])
    norm_cols: dict[str, dict] = {}
    for r in normalized:
        for k, v in r.items():
            c = norm_cols.setdefault(k, {"present": 0, "types": set(),
                                         "examples": []})
            # Treat None / empty string as "not present" for missing-rate computation.
            present = not (v is None or (isinstance(v, str) and v == ""))
            if present:
                c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                c["examples"].append(repr(v)[:120])
    n_norm = len(normalized)
    profile = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_files_inspected": [raw_file],
        "file_size_bytes": file_size_bytes,
        "inspected_row_count": n_norm,
        "raw_row_count": n_raw,
        "raw_columns": sorted(raw_cols.keys()),
        "raw_dtypes": {k: sorted(v["types"]) for k, v in raw_cols.items()},
        "raw_presence_rates": {k: (v["present"] / n_raw) if n_raw else 0
                                for k, v in raw_cols.items()},
        "raw_example_values": {k: v["examples"] for k, v in raw_cols.items()},
        "normalized_columns": sorted(norm_cols.keys()),
        "normalized_dtypes": {k: sorted(v["types"])
                               for k, v in norm_cols.items()},
        "presence_rates": {k: (v["present"] / n_norm) if n_norm else 0
                           for k, v in norm_cols.items()},
        "missing_rates": {k: 1 - ((v["present"] / n_norm) if n_norm else 0)
                          for k, v in norm_cols.items()},
        "example_values": {k: v["examples"] for k, v in norm_cols.items()},
    }
    return profile


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


ALL_SIGNALS = (
    "request_timestamps", "arrivals",
    "latency", "duration_measured",
    "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
    "tool_routing", "tool_failure_label", "tool_cancellation_label",
    "args_fingerprint_for_cache_reuse",
    "workload_shape", "customer_traffic_mix",
    "result_size_proxy", "artifacts_size_proxy",
    "model_load_event", "model_unload_event",
    "cost_or_region",
    "kv_block_hashes", "migration_or_cache_loss_proxy",
)


def _detect_signals_operations(profile: dict, normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if {"created_at_iso", "updated_at_iso", "created_at_s"} & cols:
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "duration_ms" in cols:
        out["latency"] = True
        out["duration_measured"] = True
    if "tool_name" in cols:
        out["tool_routing"] = True
        out["customer_traffic_mix"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    if "is_cancelled" in cols and any(r.get("is_cancelled") for r in normalized):
        out["tool_cancellation_label"] = True
    if "args_fingerprint" in cols:
        out["args_fingerprint_for_cache_reuse"] = True
    if any(c in cols for c in ("args_count", "kwargs_key_count",
                                "series_rows_count", "scenario_rows_count",
                                "result_payload_bytes")):
        out["workload_shape"] = True
    if "result_payload_bytes" in cols:
        out["result_size_proxy"] = True
    if "artifacts_bytes" in cols:
        out["artifacts_size_proxy"] = True
    return out


def _detect_signals_tool_summary(profile: dict, normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if {"created_at_iso", "updated_at_iso"} & cols:
        out["request_timestamps"] = True
    if "duration_ms" in cols:
        # Derived aggregate latency — recorded but tier 4-style prior only.
        out["latency"] = True
    if "tool_name" in cols:
        out["tool_routing"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    return out


def _detect_signals_operation_events(profile: dict,
                                       normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if {"created_at_iso", "created_at_s"} & cols:
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "duration_ms" in cols:
        # Derived per-event duration_ms (ms since started event) — exposes
        # dispatch / execution / completion latency stages directly.
        out["latency"] = True
        out["duration_measured"] = True
    if "tool_name" in cols and any(r.get("tool_name") for r in normalized):
        out["tool_routing"] = True
        out["customer_traffic_mix"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    if "is_cancelled" in cols and any(r.get("is_cancelled") for r in normalized):
        out["tool_cancellation_label"] = True
    if "payload_sha256" in cols:
        out["args_fingerprint_for_cache_reuse"] = True
    if "payload_bytes" in cols or "payload_key_count" in cols:
        out["workload_shape"] = True
    if "payload_bytes" in cols:
        out["result_size_proxy"] = True
    return out


def _detect_signals_audit_records(profile: dict,
                                    normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if "created_at_iso" in cols:
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "duration_ms" in cols and any(
            isinstance(r.get("duration_ms"), (int, float))
            for r in normalized):
        out["latency"] = True
        out["duration_measured"] = True
    if "tool_name" in cols and any(r.get("tool_name") for r in normalized):
        out["tool_routing"] = True
        out["customer_traffic_mix"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    if "is_cancelled" in cols and any(r.get("is_cancelled") for r in normalized):
        out["tool_cancellation_label"] = True
    if "payload_sha256" in cols:
        out["args_fingerprint_for_cache_reuse"] = True
    if any(c in cols for c in ("payload_bytes", "payload_key_count",
                                "response_key_count")):
        out["workload_shape"] = True
    if "payload_bytes" in cols:
        out["result_size_proxy"] = True
    return out


CONFIG_SIGNAL_DETECTORS = {
    "operations": _detect_signals_operations,
    "tool_summary": _detect_signals_tool_summary,
    "operation_events": _detect_signals_operation_events,
    "audit_records": _detect_signals_audit_records,
}


# ---------------------------------------------------------------------------
# Statistical rollups
# ---------------------------------------------------------------------------


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return sorted_vals[idx]


def _summarize_durations(rows: list[dict],
                          dur_field: str = "duration_ms") -> dict:
    vals = [r[dur_field] for r in rows
            if isinstance(r.get(dur_field), (int, float))]
    if not vals:
        return {"count": 0}
    vals.sort()
    n = len(vals)
    return {
        "count": n,
        "min": vals[0],
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


def _compute_rollups_operations(normalized: list[dict], raw_rows: list[dict]
                                 ) -> dict:
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}

    # Overall + per-tool + per-(tool, status) subgroup counts.
    overall_status: dict[str, int] = {}
    for r in normalized:
        s = r.get("status") or "<unset>"
        overall_status[s] = overall_status.get(s, 0) + 1
    rollups["subgroup_counts"]["status"] = overall_status

    tool_counts: dict[str, int] = {}
    for r in normalized:
        t = r.get("tool_name") or "<unset>"
        tool_counts[t] = tool_counts.get(t, 0) + 1
    rollups["subgroup_counts"]["tool_name"] = tool_counts

    stage_counts: dict[str, int] = {}
    for r in normalized:
        s = r.get("stage") or "<unset>"
        stage_counts[s] = stage_counts.get(s, 0) + 1
    rollups["subgroup_counts"]["stage"] = stage_counts

    error_counts: dict[str, int] = {}
    for r in normalized:
        e = r.get("error_type") or ""
        if e:
            error_counts[e] = error_counts.get(e, 0) + 1
    rollups["subgroup_counts"]["error_type"] = error_counts

    tool_status_counts: dict[str, int] = {}
    for r in normalized:
        key = f"{r.get('tool_name') or '<unset>'}|{r.get('status') or '<unset>'}"
        tool_status_counts[key] = tool_status_counts.get(key, 0) + 1
    rollups["subgroup_counts"]["tool_name__status"] = tool_status_counts

    # Numeric distributions (overall + per-tool + per-status).
    rollups["numeric_distributions"]["duration_ms"] = {
        "overall": _summarize_durations(normalized, "duration_ms"),
        "per_status": {
            s: _summarize_durations([r for r in normalized
                                     if (r.get("status") or "<unset>") == s])
            for s in sorted(overall_status.keys())
        },
        "per_tool": {
            t: _summarize_durations([r for r in normalized
                                     if (r.get("tool_name") or "<unset>") == t])
            for t in sorted(tool_counts.keys())
        },
    }

    for nf in ("result_payload_bytes", "artifacts_bytes",
                "args_count", "kwargs_key_count",
                "result_payload_key_count"):
        rollups["numeric_distributions"][nf] = _summarize_durations(
            normalized, nf,
        )

    # Failure rate + cancellation rate (per-tool routing-quality priors).
    per_tool_failure = {}
    for t, n in tool_counts.items():
        errors = sum(1 for r in normalized
                     if (r.get("tool_name") or "<unset>") == t
                     and r.get("is_error"))
        cancels = sum(1 for r in normalized
                      if (r.get("tool_name") or "<unset>") == t
                      and r.get("is_cancelled"))
        per_tool_failure[t] = {
            "count": n,
            "error_count": errors,
            "cancelled_count": cancels,
            "error_rate": errors / n if n else 0.0,
            "cancelled_rate": cancels / n if n else 0.0,
        }
    rollups["per_tool_failure_rates"] = per_tool_failure

    # Overall failure / cancellation rate.
    n_total = len(normalized)
    n_err = sum(1 for r in normalized if r.get("is_error"))
    n_cancel = sum(1 for r in normalized if r.get("is_cancelled"))
    rollups["overall_failure_rates"] = {
        "count": n_total,
        "error_count": n_err,
        "cancelled_count": n_cancel,
        "error_rate": n_err / n_total if n_total else 0.0,
        "cancelled_rate": n_cancel / n_total if n_total else 0.0,
    }

    # Raw column count for cross-check.
    rollups["raw_row_count"] = len(raw_rows)
    return rollups


def _compute_rollups_tool_summary(normalized: list[dict],
                                   raw_rows: list[dict]) -> dict:
    """tool_summary rows are already aggregated. Record the per-(tool, status)
    avg/median/p95 directly so the downstream consumer doesn't have to read
    the raw parquet."""
    rollups: dict = {"subgroup_counts": {}, "per_tool_status_aggregates": {}}
    tool_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for r in raw_rows:
        t = r.get("tool_name") or "<unset>"
        s = r.get("status") or "<unset>"
        tool_counts[t] = tool_counts.get(t, 0) + 1
        status_counts[s] = status_counts.get(s, 0) + 1
        key = f"{t}|{s}"
        rollups["per_tool_status_aggregates"][key] = {
            "operation_count": r.get("operation_count"),
            "avg_duration_ms": r.get("avg_duration_ms"),
            "median_duration_ms": r.get("median_duration_ms"),
            "p95_duration_ms": r.get("p95_duration_ms"),
            "first_seen_utc": r.get("first_seen_utc"),
            "last_seen_utc": r.get("last_seen_utc"),
        }
    rollups["subgroup_counts"]["tool_name"] = tool_counts
    rollups["subgroup_counts"]["status"] = status_counts
    rollups["raw_row_count"] = len(raw_rows)
    return rollups


def _compute_rollups_operation_events(normalized: list[dict],
                                        raw_rows: list[dict]) -> dict:
    """Per-stage transition timing rollups.

    The key Aurelius signal exposed by operation_events is per-stage
    duration_ms (ms since the operation's 'started' event), so the rollups
    break down duration_ms by stage and event_type. The dispatch-latency
    prior is the 'started -> stage(executing)' delta; the execution-latency
    prior is the 'stage(executing) -> stage(execution_completed)' delta.
    """
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}

    event_type_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for r in normalized:
        et = r.get("event_type") or "<unset>"
        st = r.get("stage") or "<unset>"
        sts = r.get("status") or "<unset>"
        event_type_counts[et] = event_type_counts.get(et, 0) + 1
        stage_counts[st] = stage_counts.get(st, 0) + 1
        status_counts[sts] = status_counts.get(sts, 0) + 1
    rollups["subgroup_counts"]["event_type"] = event_type_counts
    rollups["subgroup_counts"]["stage"] = stage_counts
    rollups["subgroup_counts"]["status"] = status_counts

    # Overall duration_ms (ms-since-started) distribution + per-stage +
    # per-event_type breakdowns. duration_ms for the 'started' event is
    # always 0; the value progresses through the lifecycle.
    rollups["numeric_distributions"]["duration_ms"] = {
        "overall": _summarize_durations(normalized, "duration_ms"),
        "per_event_type": {
            et: _summarize_durations([r for r in normalized
                                       if (r.get("event_type") or "<unset>") == et])
            for et in sorted(event_type_counts.keys())
        },
        "per_stage": {
            st: _summarize_durations([r for r in normalized
                                       if (r.get("stage") or "<unset>") == st])
            for st in sorted(stage_counts.keys())
        },
    }

    # Unique-operations + average events/op (lifecycle granularity proxy).
    from collections import Counter
    op_event_counts = Counter(r.get("operation_id") for r in normalized
                              if r.get("operation_id"))
    if op_event_counts:
        sizes = sorted(op_event_counts.values())
        rollups["per_operation_event_count"] = {
            "unique_operations": len(op_event_counts),
            "min_events_per_op": sizes[0],
            "max_events_per_op": sizes[-1],
            "mean_events_per_op": sum(sizes) / len(sizes),
            "p50_events_per_op": _quantile(sizes, 0.50),
            "p95_events_per_op": _quantile(sizes, 0.95),
        }

    # Per-event payload byte distribution (request shape proxy).
    rollups["numeric_distributions"]["payload_bytes"] = _summarize_durations(
        normalized, "payload_bytes",
    )

    rollups["raw_row_count"] = len(raw_rows)
    return rollups


def _compute_rollups_audit_records(normalized: list[dict],
                                     raw_rows: list[dict]) -> dict:
    """Per-(tool, category, status) audit-record duration rollups."""
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}

    category_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    for r in normalized:
        cat = r.get("category") or "<unset>"
        knd = r.get("kind") or "<unset>"
        sts = r.get("status") or "<unset>"
        t = r.get("tool_name") or "<unset>"
        category_counts[cat] = category_counts.get(cat, 0) + 1
        kind_counts[knd] = kind_counts.get(knd, 0) + 1
        status_counts[sts] = status_counts.get(sts, 0) + 1
        tool_counts[t] = tool_counts.get(t, 0) + 1
    rollups["subgroup_counts"]["category"] = category_counts
    rollups["subgroup_counts"]["kind"] = kind_counts
    rollups["subgroup_counts"]["status"] = status_counts
    rollups["subgroup_counts"]["tool_name"] = tool_counts

    # duration_ms distribution (only populated on tool_results rows).
    results_only = [r for r in normalized
                    if r.get("category") == "tool_results"]
    rollups["numeric_distributions"]["duration_ms"] = {
        "overall_tool_results": _summarize_durations(results_only,
                                                      "duration_ms"),
        "per_status": {
            s: _summarize_durations([r for r in results_only
                                     if (r.get("status") or "<unset>") == s])
            for s in sorted(set(r.get("status") or "<unset>"
                                for r in results_only))
        },
        "per_tool": {
            t: _summarize_durations([r for r in results_only
                                     if (r.get("tool_name") or "<unset>") == t])
            for t in sorted(tool_counts.keys())
            if any((r.get("tool_name") or "<unset>") == t for r in results_only)
        },
    }

    # Per-request audit-record pair counts (should be ~2: one request + one
    # result per request_id).
    from collections import Counter
    req_counts = Counter(r.get("request_id") for r in normalized
                          if r.get("request_id"))
    if req_counts:
        rollups["per_request_audit_record_count"] = {
            "unique_request_ids": len(req_counts),
            "mean_records_per_request":
                sum(req_counts.values()) / len(req_counts),
            "max_records_per_request": max(req_counts.values()),
        }

    # Per-tool failure rates (over tool_results rows where status is set).
    per_tool_failure = {}
    for t in sorted(tool_counts.keys()):
        rows_t = [r for r in results_only
                  if (r.get("tool_name") or "<unset>") == t]
        if not rows_t:
            continue
        n = len(rows_t)
        errors = sum(1 for r in rows_t if r.get("is_error"))
        per_tool_failure[t] = {
            "count": n,
            "error_count": errors,
            "error_rate": errors / n if n else 0.0,
        }
    rollups["per_tool_failure_rates"] = per_tool_failure

    # Overall failure rate over tool_results only.
    n_total = len(results_only)
    n_err = sum(1 for r in results_only if r.get("is_error"))
    rollups["overall_failure_rates"] = {
        "count": n_total,
        "error_count": n_err,
        "error_rate": n_err / n_total if n_total else 0.0,
    }

    rollups["numeric_distributions"]["payload_bytes"] = _summarize_durations(
        normalized, "payload_bytes",
    )
    rollups["numeric_distributions"]["response_key_count"] = (
        _summarize_durations(results_only, "response_key_count")
    )

    rollups["raw_row_count"] = len(raw_rows)
    return rollups


CONFIG_ROLLUPS = {
    "operations": _compute_rollups_operations,
    "tool_summary": _compute_rollups_tool_summary,
    "operation_events": _compute_rollups_operation_events,
    "audit_records": _compute_rollups_audit_records,
}


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------


def audit_one(target: dict, *, token: str | None,
              force_redownload: bool) -> dict:
    config = target["config_name"]
    hb = _Heartbeat(f"{DATASET_ID}@{config}")
    hb.update(phase="setup", force=True)

    safe_ds = HF_DIR / SAFE_DATASET
    raw_path = safe_ds / "raw" / Path(target["raw_file"]).name
    processed_dir = safe_ds / config / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    schema_profile_path = processed_dir / "schema_profile.json"
    schema_mapping_path = processed_dir / "schema_mapping.json"
    summary_path = processed_dir / "summary.json"
    rollups_path = processed_dir / "statistical_rollups.json"
    analysis_sample_path = processed_dir / "analysis_sample.jsonl"
    normalized_sample_path = processed_dir / "normalized_sample.jsonl"
    fixture_path = FIXTURES_DIR / f"{SAFE_DATASET}__{config}_sample.jsonl"

    url = _hf_url(DATASET_ID, target["raw_file"])

    hb.update(phase="download", force=True)
    if force_redownload or not raw_path.exists():
        manifest = _bounded_download(url, raw_path, max_bytes=None,
                                     token=token)
        manifest["cached"] = False
    else:
        manifest = {"url": url, "dest": str(raw_path),
                    "downloaded_bytes": raw_path.stat().st_size,
                    "status": None, "truncated": None,
                    "error": None, "max_bytes": None, "cached": True}
    if manifest.get("error"):
        return {"target": target, "manifest": manifest,
                "audit_status": "download_failed"}

    hb.update(phase="parse", force=True)
    raw_rows = _read_parquet(raw_path)
    hb.update(phase="parsed_rows", rows_done=len(raw_rows), force=True)
    if not raw_rows:
        return {"target": target, "manifest": manifest,
                "audit_status": "no_rows"}

    hb.update(phase="normalize", force=True)
    normalizer = CONFIG_NORMALIZERS[config]
    # operation_events needs a list-level pass (per-operation 'started'
    # timestamp lookup) to compute the derived per-event duration_ms.
    if config == "operation_events":
        normalized = normalizer(raw_rows)
    else:
        normalized = [normalizer(r) for r in raw_rows]
    hb.update(phase="normalized", rows_done=len(normalized), force=True)

    hb.update(phase="profile", force=True)
    profile = _profile_rows(raw_rows, normalized, config, target["raw_file"],
                             manifest["downloaded_bytes"])
    with open(schema_profile_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str, sort_keys=True)

    # Schema mapping: enumerate every raw column and label it.
    mapping_dict = CONFIG_MAPPINGS[config]
    accepted = [c for c in profile["raw_columns"] if c in mapping_dict]
    rejected = [c for c in profile["raw_columns"] if c not in mapping_dict]
    column_records = []
    for c in profile["raw_columns"]:
        m = mapping_dict.get(c, {})
        column_records.append({
            "raw_column_name": c,
            "raw_dtype": profile["raw_dtypes"].get(c),
            "presence_rate": profile["raw_presence_rates"].get(c),
            "normalized_field": m.get("normalized_field"),
            "field_quality": m.get("field_quality"),
            "aurelius_signal_category": m.get("aurelius_signal_category"),
            "usable_for": m.get("usable_for"),
            "notes": m.get("notes"),
        })
    mapping_doc = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "accepted_columns": sorted(accepted),
        "rejected_columns": sorted(rejected),
        "columns": column_records,
    }
    with open(schema_mapping_path, "w") as fh:
        json.dump(mapping_doc, fh, indent=2, default=str, sort_keys=True)

    hb.update(phase="rollups", force=True)
    rollups = CONFIG_ROLLUPS[config](normalized, raw_rows)
    with open(rollups_path, "w") as fh:
        json.dump(rollups, fh, indent=2, default=str, sort_keys=True)

    hb.update(phase="write_analysis_sample", force=True)
    analysis_bytes, analysis_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], analysis_sample_path,
    )

    hb.update(phase="write_normalized_sample", force=True)
    normalized_bytes, normalized_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], normalized_sample_path,
    )
    if normalized_bytes > MAX_COMMITTED_NORMALIZED_BYTES:
        keep = max(1, int(len(normalized) *
                          MAX_COMMITTED_NORMALIZED_BYTES / normalized_bytes))
        normalized_bytes, normalized_sha = _write_jsonl(
            [_safe_jsonable(r) for r in normalized[:keep]],
            normalized_sample_path,
        )

    hb.update(phase="write_fixture", force=True)
    fixture_rows = [_safe_jsonable(r) for r in normalized[:5]]
    fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
    while fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES and fixture_rows:
        fixture_rows = fixture_rows[: max(1, len(fixture_rows) - 1)]
        fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
        if len(fixture_rows) == 1 and fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES:
            slim = {k: v for k, v in fixture_rows[0].items()
                    if not isinstance(v, str) or len(v) < 200}
            fixture_rows = [slim]
            fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
            break

    # Signals
    signals = CONFIG_SIGNAL_DETECTORS[config](profile, normalized)
    available = sorted(k for k, v in signals.items() if v)
    missing = sorted(k for k, v in signals.items() if not v)

    derived_fields = ["created_at_s", "updated_at_s", "duration_s",
                       "is_error", "is_cancelled"]
    if config == "tool_summary":
        # In tool_summary, the duration_ms in the normalized sample is the
        # median of the aggregate bucket — that's derived, not real.
        derived_fields = derived_fields + ["duration_ms", "duration_s"]
    elif config == "operation_events":
        # Per-event duration_ms is ms-since-started (derived from real
        # measured event_time_utc timestamps).
        derived_fields = derived_fields + ["duration_ms", "duration_s"]
    proxy_fields = [
        "args_count", "kwargs_key_count", "series_rows_count",
        "scenario_rows_count", "result_payload_key_count",
        "result_payload_bytes", "artifacts_bytes",
    ]
    if config in {"operation_events", "audit_records"}:
        proxy_fields = proxy_fields + [
            "payload_bytes", "payload_key_count", "response_key_count",
        ]

    if config == "operations":
        limitations = [
            "Real measured MCP-style agent-runtime tool-call execution telemetry exported from Faruk Alpay's Lightcap runtime (operations.parquet).",
            "Closed tool-runtime end-to-end timing — duration_ms is the tool-call wall-clock from request to response, NOT GPU TTFT/TPOT, and NOT LLM serving latency.",
            "NO model_id / NO input_tokens / NO output_tokens / NO GPU type / NO queue depth / NO replica count / NO cache state / NO LLM-serving signal. Tool-runtime trace only.",
            "Single export (one runtime, ~7 days, 2,262 operations across 22 tools). Treat as routing-quality + failure-rate + tail-latency PRIOR for agent workloads — not as a serving telemetry calibration source. Pilot telemetry remains the only Tier 1 calibration source.",
            "Raw args / kwargs / result payload bodies are NOT redistributed in this export — only fingerprints (args_fingerprint = sha256), counts (args_count, kwargs_key_count), key lists (pipe-delimited), and byte totals (result_payload_bytes, artifacts_bytes). Error messages are stored as a preview + sha256 only.",
        ]
    elif config == "tool_summary":
        limitations = [
            "Pre-aggregated per-(tool_name, status) bucket summary from Faruk Alpay's Lightcap tool-runtime (tool_summary.parquet, 32 rows, 22 distinct tools).",
            "Derived aggregate latency — the normalized sample's duration_ms is the per-bucket MEDIAN computed by the upstream exporter; field_quality=derived. The exact avg / median / p95 per (tool, status) live in statistical_rollups.json::per_tool_status_aggregates.",
            "NOT a per-call trace — operations.parquet config is the per-call counterpart. Use this config as a quick per-tool latency prior; use operations for distributional analysis.",
            "NOT GPU TTFT/TPOT, NOT LLM serving telemetry — closed tool-runtime end-to-end timing only. No model_id / no input_tokens / no GPU type / no queue / no replica / no cache state.",
            "Same provenance + scope caveats as the operations config — single runtime, single ~7-day window, 22 tools.",
        ]
    elif config == "operation_events":
        limitations = [
            "Per-event lifecycle transitions from Faruk Alpay's Lightcap tool-runtime (operation_events.parquet, 9,903 events across 2,262 operations).",
            "duration_ms is DERIVED — computed as (event_time_utc - operation's earliest-event event_time_utc) for the same operation_id; field_quality=derived. The underlying event_time_utc timestamps are real, the per-event ms-since-started is a derived signal.",
            "Per-stage transition latency is exposed via duration_ms broken down by stage in statistical_rollups: dispatch (started -> stage(executing)) is ~10-15 ms; execution (stage(executing) -> stage(execution_completed)) holds the tool wall-clock; post-processing (stage(execution_completed) -> completed) is sub-millisecond.",
            "NOT GPU TTFT/TPOT, NOT LLM serving telemetry — closed tool-runtime event timing only. No model_id / no GPU type / no queue depth / no replica count / no cache state. The 'queue-wait-style' interpretation is dispatch latency at the agent-runtime level, NOT cluster scheduler queue wait.",
            "operation_id joins to operations.parquet so dispatch-stage latency can be cross-referenced with operations' end-to-end duration_ms. payload_tool is populated only on 'started' events; subsequent stage/completed events leave tool_name null and the join via operation_id is required.",
            "Single export (same provenance window + 22-tool set as operations / tool_summary). Treat as scheduler-state + dispatch-latency PRIOR for agent workloads — not as a serving telemetry calibration source.",
        ]
    else:  # audit_records
        limitations = [
            "MCP-shell-layer audit records from Faruk Alpay's Lightcap tool-runtime (audit_records.parquet, 14,053 records: 7,012 tool_requests + 7,041 tool_results).",
            "duration_ms is REAL but populated only on category='tool_results' rows (50% of audit_records). tool_requests rows have duration_ms=null because the request hasn't completed yet. Heavy tail: p95=400 ms, p99=2.5 s, max=900 s in this export.",
            "MCP-shell-layer timing is distinct from operations' runtime-layer timing — both measure tool-call latency but at different boundaries. The shell-layer duration_ms captures the request/response envelope; operations' duration_ms captures the internal execution. Joining via request_id lets the harness compare envelope-vs-execution overhead.",
            "NOT GPU TTFT/TPOT, NOT LLM serving telemetry — closed MCP-shell e2e timing only. No model_id / no GPU type / no queue depth / no replica count / no LLM-serving signal.",
            "Raw payload bodies are NOT redistributed — only fingerprints (payload_sha256), counts (payload_key_count, response_key_count), key lists (pipe-delimited payload_keys, response_keys), and byte totals (payload_bytes).",
            "Same provenance + scope caveats as the operations / tool_summary / operation_events configs — single runtime, single ~7-day window, 22 tools. Treat as MCP-shell-layer latency PRIOR + per-request cache-reuse PROXY (via payload_sha256), not as a serving telemetry calibration source.",
        ]

    strength = _statistical_sample_strength(len(normalized))

    git_sha = _git_sha()
    provenance = (
        f"{DATASET_ID}@{config}#{target['raw_file']}"
        f"#bytes={manifest['downloaded_bytes']}#git={git_sha[:7] if git_sha else 'na'}"
    )

    summary = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": "cc-by-4.0",
        "gated": False,
        "raw_schema": list(profile["raw_columns"]),
        "normalized_schema": list(profile["normalized_columns"]),
        "unknown_columns": [],  # every raw column is mapped (or explicitly rejected below)
        "rejected_columns": rejected,
        "canonical_trace_type": "tool_runtime_trace",
        "available_signals": available,
        "missing_signals": missing,
        "derived_fields": derived_fields,
        "proxy_fields": proxy_fields,
        "synthetic_fields": [],
        "provenance": provenance,
        "limitations": limitations,
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_bytes,
        "analysis_sample_rows": len(normalized),
        "analysis_sample_bytes": normalized_bytes,
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_bytes,
        "normalized_sample_rows": len(normalized),
        "normalized_sample_bytes": normalized_bytes,
        "normalized_sample_sha256": normalized_sha,
        "sample_sha256": fixture_sha,
        "sampling_method": "full_bounded",
        "stratification_keys": (
            ["operation_id", "event_type", "stage", "status"]
            if config == "operation_events"
            else ["request_id", "category", "tool_name", "status"]
            if config == "audit_records"
            else ["tool_name", "status", "stage", "error_type"]
        ),
        "statistical_sample_strength": strength,
        "ingestion_timestamp_s": time.time(),
        "summary_path_relative": f"data/external/hf/{SAFE_DATASET}/{config}/processed/summary.json",
        "schema_profile_path": f"data/external/hf/{SAFE_DATASET}/{config}/processed/schema_profile.json",
        "schema_mapping_path": f"data/external/hf/{SAFE_DATASET}/{config}/processed/schema_mapping.json",
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str, sort_keys=True)

    decision = promotion.evaluate_promotion(summary)

    return {
        "target": target,
        "manifest": manifest,
        "audit_status": "ok",
        "summary_path": str(summary_path),
        "profile_path": str(schema_profile_path),
        "mapping_path": str(schema_mapping_path),
        "rollups_path": str(rollups_path),
        "fixture_path": str(fixture_path),
        "normalized_sample_path": str(normalized_sample_path),
        "analysis_sample_path": str(analysis_sample_path),
        "decision": decision,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=None,
                        help="Limit to one or more configs (default: all).")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Re-download even if raw file exists.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.info("HF_TOKEN not set — public-only access.")
    requested = set(args.config) if args.config else None

    DISC_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    _install_timeout(PER_DATASET_TIMEOUT_S)
    try:
        per_config: list[dict] = []
        for tgt in TARGETS:
            if requested and tgt["config_name"] not in requested:
                continue
            logger.info("=== %s :: %s ===", DATASET_ID, tgt["config_name"])
            result = audit_one(tgt, token=token,
                                force_redownload=args.force_redownload)
            d = result.get("decision") or {}
            logger.info("  %s -> state=%s tags=%s",
                        tgt["config_name"], d.get("state"),
                        d.get("promotion_tags"))
            per_config.append({
                "config": tgt["config_name"],
                "audit_status": result.get("audit_status"),
                "manifest": result.get("manifest"),
                "summary_path": result.get("summary_path"),
                "decision_state": d.get("state"),
                "decision_tags": d.get("promotion_tags"),
            })

        summary_out = {
            "dataset_id": DATASET_ID,
            "wrote_at_s": time.time(),
            "configs": per_config,
        }
        out_path = (DISC_DIR /
                    "lightcap_runtime_telemetry_ingest_summary.json")
        with open(out_path, "w") as fh:
            json.dump(summary_out, fh, indent=2, default=str, sort_keys=True)
        logger.info("Wrote %s (%d configs)", out_path, len(per_config))
    finally:
        _clear_timeout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
