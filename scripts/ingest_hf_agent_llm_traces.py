#!/usr/bin/env python3
"""Bounded ingestion of Exgentic/agent-llm-traces.

A 2026 OpenTelemetry agent-trace dataset (cdla-permissive-2.0) covering
1,781 agent sessions across 6 benchmarks x 5 frameworks x 6 frontier
models (Claude / GPT / Gemini / DeepSeek / Kimi). Each session contains
a list of OpenTelemetry spans annotated with `gen_ai.*` semantic
conventions:

- per-span `start_time` / `end_time` (closed-API end-to-end span timing —
  network + provider serving, NOT GPU TTFT/TPOT),
- per-span `gen_ai.usage.input_tokens` / `output_tokens`,
- per-span `gen_ai.request.model` / `gen_ai.response.model`,
- per-span `status.code` (OTel status code, 0=UNSET, 1=OK, 2=ERROR),
- per-span `gen_ai.response.finish_reasons` (list, e.g.
  ['tool_calls', 'stop', 'length']),
- per-session `harness` / `benchmark` / `models` / `session_id`.

Trace type: `request_shape_trace` (Tier 5). Closed-API e2e timing is
recorded as `duration_ms` with `field_quality="real"` but **must not** be
used as a GPU-serving latency prior — see the limitations list. This
ingester DROPS the huge `gen_ai.input.messages` / `gen_ai.output.messages`
/ `gen_ai.tool.definitions` payload strings (median 50K chars) and keeps
only their character counts, so the committed normalized sample stays
well under the 100 MB per-file cap.

Audit-only. Does NOT modify scheduler / controllers / robust energy
engine. Does NOT train forecasting models. Raw downloads are gitignored;
only schema_profile, schema_mapping, summary, statistical_rollups, the
tiny fixture, and the bounded normalized sample (cdla-permissive-2.0
permits redistribution) are committed.

Layout:
    data/external/hf/Exgentic__agent-llm-traces/raw/<file>     # gitignored
    data/external/hf/Exgentic__agent-llm-traces/<config>/processed/
        schema_profile.json
        schema_mapping.json
        summary.json
        statistical_rollups.json
        normalized_sample.jsonl                                # committed
        analysis_sample.jsonl                                  # gitignored
    tests/fixtures/hf/Exgentic__agent-llm-traces__<config>_sample.jsonl
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

DATASET_ID = "Exgentic/agent-llm-traces"
SAFE_DATASET = DATASET_ID.replace("/", "__")

MAX_COMMITTED_FIXTURE_BYTES = 16 * 1024
MAX_COMMITTED_NORMALIZED_BYTES = 100 * 1024 * 1024  # 100 MiB per the policy

PER_DATASET_TIMEOUT_S = 30 * 60
ROW_CAP_FOR_NORMALIZATION = 60_000
PROGRESS_INTERVAL_S = 30

logger = logging.getLogger("aurelius.hf_exgentic_ingest")


# ---------------------------------------------------------------------------
# Heartbeat / timeout (mirrors scripts/ingest_hf_gap_datasets.py)
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


class _PerDatasetTimeout(Exception):
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
    # Mid-sized parquet — moderate sample strength for the SWE-bench /
    # claude_code subset (real Anthropic / Azure-hosted model traces).
    {
        "config_name": "swebench_claude_code_shard12",
        "raw_file": "data/train-00012-of-00039.parquet",
        "expected_raw_bytes": 41 * 1024 * 1024,  # 40.36 MB upstream
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
    headers = {"User-Agent": "aurelius-hf-exgentic-ingest/1.0"}
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


def _read_parquet_sessions(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


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
# Flatten OTel session → per-span request rows
# ---------------------------------------------------------------------------


def _iso_to_epoch_s(iso: str | None) -> float | None:
    """Parse RFC3339 with offset → epoch seconds. Returns None on failure."""
    if not iso or not isinstance(iso, str):
        return None
    # Python datetime can handle '2026-04-15T11:10:18.025446+00:00'.
    try:
        from datetime import datetime
        # Replace trailing 'Z' with '+00:00' if present.
        s = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _flatten_session(sess: dict) -> list[dict]:
    """Produce one normalized row per span. Drop the huge prompt/output
    payload bodies; keep only their character counts so the analysis
    sample stays small."""
    out: list[dict] = []
    session_id = sess.get("session_id")
    harness = sess.get("harness")
    benchmark = sess.get("benchmark")
    models_list = sess.get("models") or []
    session_models_json = json.dumps(
        models_list, sort_keys=True, default=str,
    )[:300] if models_list else None
    collected_at = sess.get("collected_at")
    session_max_tokens = sess.get("max_tokens")
    session_total_tokens = sess.get("total_tokens")
    spans = sess.get("spans") or []
    for i, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes") or {}
        status = span.get("status") or {}
        start_iso = span.get("start_time")
        end_iso = span.get("end_time")
        start_s = _iso_to_epoch_s(start_iso)
        end_s = _iso_to_epoch_s(end_iso)
        duration_ms = None
        if start_s is not None and end_s is not None and end_s >= start_s:
            duration_ms = (end_s - start_s) * 1000.0
        input_msgs = attrs.get("gen_ai.input.messages")
        output_msgs = attrs.get("gen_ai.output.messages")
        tool_defs = attrs.get("gen_ai.tool.definitions")
        finish_reasons = attrs.get("gen_ai.response.finish_reasons") or []
        # Status.code: OTel convention. 0=UNSET, 1=OK, 2=ERROR.
        status_code = status.get("code")
        status_message = status.get("message")
        is_error = status_code == 2 if status_code is not None else None
        # Hash the input messages as a routing/cache-residency proxy
        # (same prompt → same hash → potential prefix reuse signal).
        input_messages_hash = (
            _hash_str(input_msgs) if isinstance(input_msgs, str) and input_msgs
            else None
        )
        row = {
            # Session-level passthrough
            "session_id": session_id,
            "harness": harness,
            "benchmark": benchmark,
            "session_models_json": session_models_json,
            "session_models_count": len(models_list) if isinstance(models_list, list) else None,
            "collected_at_iso": collected_at,
            "session_max_tokens": session_max_tokens,
            "session_total_tokens": session_total_tokens,
            # Span-level core fields
            "span_id": span.get("span_id"),
            "span_index": i,
            "span_name": span.get("name"),
            "span_kind": span.get("kind"),
            "span_type": span.get("type"),
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
            "start_time_s": start_s,
            "end_time_s": end_s,
            "duration_ms": duration_ms,
            # gen_ai semantic-convention attributes
            "operation_name": attrs.get("gen_ai.operation.name"),
            "request_model": attrs.get("gen_ai.request.model"),
            "response_model": attrs.get("gen_ai.response.model"),
            "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
            "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
            "response_id": attrs.get("gen_ai.response.id"),
            "finish_reasons": list(finish_reasons) if finish_reasons else [],
            "finish_reasons_count": len(finish_reasons) if isinstance(finish_reasons, list) else 0,
            # Payload size proxies (the raw strings are DROPPED)
            "input_messages_chars": len(input_msgs) if isinstance(input_msgs, str) else 0,
            "output_messages_chars": len(output_msgs) if isinstance(output_msgs, str) else 0,
            "tool_definitions_chars": len(tool_defs) if isinstance(tool_defs, str) else 0,
            "input_messages_hash": input_messages_hash,
            # Status
            "status_code": status_code,
            "status_message": status_message,
            "is_error": is_error,
        }
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Schema mapping (raw nested field → normalized field + field_quality)
# ---------------------------------------------------------------------------


EXGENTIC_MAPPING: dict = {
    # ── Session-level (carried into every flattened span row)
    "session_id": {"normalized_field": "session_id", "field_quality": "real",
                   "aurelius_signal_category": "session_id",
                   "usable_for": ["cache_residency_evaluation", "arrival_forecast",
                                  "workload_shape_prior"],
                   "notes": "Agent session UUID. Spans in the same session share it."},
    "harness": {"normalized_field": "harness", "field_quality": "real",
                "aurelius_signal_category": "metadata_only",
                "usable_for": ["workload_shape_prior"],
                "notes": "Agent harness label (openai_solo / tool_calling / claude_code / ...)."},
    "benchmark": {"normalized_field": "benchmark", "field_quality": "real",
                  "aurelius_signal_category": "metadata_only",
                  "usable_for": ["workload_shape_prior"],
                  "notes": "Benchmark suite (swebench / appworld / tau2_airline / ...)."},
    "models": {"normalized_field": "session_models_json",
               "field_quality": "real",
               "aurelius_signal_category": "metadata_only",
               "usable_for": ["workload_shape_prior"],
               "notes": "List of models used across the session (JSON-stringified)."},
    "max_tokens": {"normalized_field": "session_max_tokens",
                   "field_quality": "real",
                   "aurelius_signal_category": "tokens",
                   "usable_for": ["workload_shape_prior"],
                   "notes": "Session-level max-tokens config (parameter)."},
    "total_tokens": {"normalized_field": "session_total_tokens",
                     "field_quality": "real",
                     "aurelius_signal_category": "tokens",
                     "usable_for": ["workload_shape_prior"],
                     "notes": "Session-level total-tokens aggregate."},
    "collected_at": {"normalized_field": "collected_at_iso",
                     "field_quality": "real",
                     "aurelius_signal_category": "metadata_only",
                     "usable_for": ["workload_shape_prior"],
                     "notes": "When the trace was collected (ISO timestamp)."},
    # ── Span-level (flattened to per-row)
    "spans": {"normalized_field": "span_index",
              "field_quality": "real",
              "aurelius_signal_category": "session_id",
              "usable_for": ["workload_shape_prior", "cache_residency_evaluation"],
              "notes": "Session-level spans list; flattened per-span before mapping."},
    "span_id": {"normalized_field": "span_id", "field_quality": "real",
                "aurelius_signal_category": "metadata_only",
                "usable_for": ["workload_shape_prior"],
                "notes": "OpenTelemetry span identifier."},
    "name": {"normalized_field": "span_name", "field_quality": "real",
             "aurelius_signal_category": "metadata_only",
             "usable_for": ["workload_shape_prior"],
             "notes": "Span name (typically 'chat <model>')."},
    "kind": {"normalized_field": "span_kind", "field_quality": "real",
             "aurelius_signal_category": "metadata_only",
             "usable_for": ["workload_shape_prior"],
             "notes": "OTel SpanKind (SPAN_KIND_CLIENT / SERVER / INTERNAL)."},
    "type": {"normalized_field": "span_type", "field_quality": "real",
             "aurelius_signal_category": "metadata_only",
             "usable_for": ["workload_shape_prior"],
             "notes": "Provider-specific span subtype."},
    "start_time": {"normalized_field": "start_time_iso",
                   "field_quality": "real",
                   "aurelius_signal_category": "request_arrival",
                   "usable_for": ["arrival_forecast", "latency_prior"],
                   "notes": "RFC3339 ISO span start; epoch seconds also exposed as start_time_s."},
    "end_time": {"normalized_field": "end_time_iso",
                 "field_quality": "real",
                 "aurelius_signal_category": "request_completion",
                 "usable_for": ["latency_prior"],
                 "notes": "RFC3339 ISO span end; epoch seconds also exposed as end_time_s."},
    "attributes": {"normalized_field": "operation_name",
                   "field_quality": "real",
                   "aurelius_signal_category": "metadata_only",
                   "usable_for": ["workload_shape_prior"],
                   "notes": "OTel attributes dict; flattened keys mapped below."},
    "resource_attributes": {"normalized_field": "span_type",
                            "field_quality": "real",
                            "aurelius_signal_category": "metadata_only",
                            "usable_for": ["workload_shape_prior"],
                            "notes": "OTel resource_attributes dict; not flattened beyond service id."},
    "status": {"normalized_field": "status_code",
               "field_quality": "real",
               "aurelius_signal_category": "failure_timeout",
               "usable_for": ["latency_prior", "cache_residency_evaluation"],
               "notes": "OTel status struct; code 2 == ERROR is the failure_timeout signal."},
    "trace_id": {"normalized_field": "session_id",
                 "field_quality": "real",
                 "aurelius_signal_category": "session_id",
                 "usable_for": ["workload_shape_prior"],
                 "notes": "OTel trace identifier; aliased onto session_id."},
}


# ---------------------------------------------------------------------------
# Schema profile (run after flattening)
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


def _profile_rows(rows: list[dict], config: str, raw_file: str, file_size_bytes: int,
                  raw_sessions: list[dict]) -> dict:
    """Profile both the raw (session-level) schema AND the flattened
    (span-level) schema. The promotion gates require both to be
    non-empty."""
    cols: dict[str, dict] = {}
    n = len(rows)
    for r in rows:
        for k, v in r.items():
            c = cols.setdefault(k, {"present": 0, "types": set(), "examples": []})
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
    # Raw (session-level) columns + nested span keys.
    raw_session_cols: dict[str, set] = {}
    raw_span_keys: set = set()
    raw_attr_keys: set = set()
    for sess in raw_sessions:
        for k in sess.keys():
            raw_session_cols.setdefault(k, set()).add(_classify_value(sess[k]))
        spans = sess.get("spans") or []
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            for sk in sp.keys():
                raw_span_keys.add(sk)
            attrs = sp.get("attributes") or {}
            for ak in attrs.keys():
                raw_attr_keys.add(ak)
    raw_cols_sorted = sorted(raw_session_cols.keys())
    profile = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_files_inspected": [raw_file],
        "file_size_bytes": file_size_bytes,
        "inspected_session_count": len(raw_sessions),
        "inspected_row_count": n,
        # Raw schema: session-level columns observed in the source parquet.
        # This is what the promotion `schema_test` gate inspects.
        "raw_columns": raw_cols_sorted,
        "raw_session_dtypes": {k: sorted(v) for k, v in raw_session_cols.items()},
        "raw_span_keys": sorted(raw_span_keys),
        "raw_attribute_keys": sorted(raw_attr_keys),
        # Flattened (span-level) schema after _flatten_session.
        "flattened_columns": sorted(cols.keys()),
        "flattened_dtypes": {k: sorted(c["types"]) for k, c in cols.items()},
        "presence_rates": {k: c["present"] / n if n else 0 for k, c in cols.items()},
        "missing_rates": {k: 1 - (c["present"] / n if n else 0) for k, c in cols.items()},
        "example_values": {k: c["examples"] for k, c in cols.items()},
    }
    return profile


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


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


def _detect_signals(profile: dict, normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["flattened_columns"])
    if {"start_time_iso", "end_time_iso", "start_time_s"} & cols:
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "duration_ms" in cols:
        # Closed-API end-to-end span duration — record as `latency` but the
        # limitations list pins this as NOT a GPU TTFT/TPOT signal.
        out["latency"] = True
    if any(c in cols for c in ("input_tokens", "output_tokens",
                                "input_messages_chars", "output_messages_chars")):
        out["workload_shape"] = True
    if "is_error" in cols:
        # OTel status.code==2 is an ERROR; treat as a (weak) failure label.
        if any(r.get("is_error") for r in normalized):
            out["timeout_label"] = True
            out["sla_label"] = True
    if "session_id" in cols:
        out["routing_proxy"] = True  # session-affinity → routing signal
    if "input_messages_hash" in cols:
        # Same hash across spans is a prefix-reuse proxy (semantic, not block-level).
        out["cache_reuse"] = True
        out["prefix_reuse"] = True
    if "harness" in cols and "benchmark" in cols:
        out["customer_traffic_mix"] = True
    return out


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------


def audit_one(target: dict, *, token: str | None, force_redownload: bool) -> dict:
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

    # 1. Bounded download (full file — bounded at the file-selection level).
    hb.update(phase="download", force=True)
    if force_redownload or not raw_path.exists():
        manifest = _bounded_download(url, raw_path, max_bytes=None, token=token)
        manifest["cached"] = False
    else:
        manifest = {"url": url, "dest": str(raw_path),
                    "downloaded_bytes": raw_path.stat().st_size,
                    "status": None, "truncated": None,
                    "error": None, "max_bytes": None, "cached": True}
    if manifest.get("error"):
        return {"target": target, "manifest": manifest,
                "audit_status": "download_failed"}

    # 2. Parse parquet → session rows.
    hb.update(phase="parse", force=True)
    raw_sessions = _read_parquet_sessions(raw_path)
    hb.update(phase="parsed_sessions", rows_done=len(raw_sessions), force=True)
    if not raw_sessions:
        return {"target": target, "manifest": manifest, "audit_status": "no_rows"}

    # 3. Flatten sessions → spans (cap at ROW_CAP_FOR_NORMALIZATION).
    hb.update(phase="flatten", force=True)
    normalized: list[dict] = []
    for sess in raw_sessions:
        rows = _flatten_session(sess)
        normalized.extend(rows)
        if len(normalized) >= ROW_CAP_FOR_NORMALIZATION:
            normalized = normalized[:ROW_CAP_FOR_NORMALIZATION]
            break
        hb.update(rows_done=len(normalized))
    hb.update(phase="flattened", rows_done=len(normalized), force=True)
    if not normalized:
        return {"target": target, "manifest": manifest, "audit_status": "no_spans"}

    # 4. Schema profile (raw + flattened).
    hb.update(phase="profile", force=True)
    profile = _profile_rows(normalized, config, target["raw_file"],
                            manifest["downloaded_bytes"], raw_sessions)
    with open(schema_profile_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str, sort_keys=True)

    # 5. Schema mapping (raw session-level column → normalized field).
    accepted = [c for c in profile["raw_columns"] if c in EXGENTIC_MAPPING]
    rejected = [c for c in profile["raw_columns"] if c not in EXGENTIC_MAPPING]
    column_records = []
    for c in profile["raw_columns"]:
        m = EXGENTIC_MAPPING.get(c, {})
        column_records.append({
            "raw_column_name": c,
            "normalized_field": m.get("normalized_field"),
            "field_quality": m.get("field_quality"),
            "aurelius_signal_category": m.get("aurelius_signal_category"),
            "usable_for": m.get("usable_for"),
            "notes": m.get("notes"),
            "presence_rate": None,  # session-level; profiled separately at flatten time
            "missing_rate": None,
            "dtypes": profile["raw_session_dtypes"].get(c),
        })
    # Per-nested-span/attribute keys: enumerate them under nested_keys so the
    # schema mapping table covers every observed key.
    nested_records = []
    for sk in profile["raw_span_keys"]:
        m = EXGENTIC_MAPPING.get(sk, {})
        nested_records.append({
            "raw_column_name": f"spans[].{sk}",
            "normalized_field": m.get("normalized_field"),
            "field_quality": m.get("field_quality"),
            "aurelius_signal_category": m.get("aurelius_signal_category"),
            "usable_for": m.get("usable_for"),
            "notes": m.get("notes"),
        })
    # gen_ai.* attribute keys.
    GEN_AI_KEYS = {
        "gen_ai.operation.name": ("operation_name", "real", "metadata_only",
                                  ["workload_shape_prior"],
                                  "gen_ai semantic-convention operation name."),
        "gen_ai.request.model": ("request_model", "real", "metadata_only",
                                 ["workload_shape_prior"],
                                 "gen_ai requested model id."),
        "gen_ai.response.model": ("response_model", "real", "metadata_only",
                                  ["workload_shape_prior"],
                                  "gen_ai responding model id."),
        "gen_ai.usage.input_tokens": ("input_tokens", "real", "tokens",
                                      ["latency_prior", "arrival_forecast"],
                                      "Real input token count."),
        "gen_ai.usage.output_tokens": ("output_tokens", "real", "tokens",
                                       ["latency_prior", "arrival_forecast"],
                                       "Real output token count."),
        "gen_ai.response.id": ("response_id", "real", "metadata_only",
                               ["workload_shape_prior"],
                               "Provider-side response id; ignored for replay."),
        "gen_ai.response.finish_reasons": ("finish_reasons", "real",
                                           "failure_timeout",
                                           ["latency_prior"],
                                           "List of finish reasons (stop / length / tool_calls)."),
        "gen_ai.input.messages": ("input_messages_chars", "derived",
                                  "tokens",
                                  ["workload_shape_prior"],
                                  "Raw payload string DROPPED; only character count retained."),
        "gen_ai.output.messages": ("output_messages_chars", "derived",
                                   "tokens",
                                   ["workload_shape_prior"],
                                   "Raw payload string DROPPED; only character count retained."),
        "gen_ai.tool.definitions": ("tool_definitions_chars", "derived",
                                    "metadata_only",
                                    ["workload_shape_prior"],
                                    "Raw payload string DROPPED; only character count retained."),
    }
    for ak in profile["raw_attribute_keys"]:
        meta = GEN_AI_KEYS.get(ak)
        if meta is None:
            nested_records.append({
                "raw_column_name": f"spans[].attributes.{ak}",
                "normalized_field": None,
                "field_quality": "unknown",
                "aurelius_signal_category": None,
                "usable_for": None,
                "notes": "Observed but not normalized in v1.",
            })
        else:
            nf, fq, cat, uf, notes = meta
            nested_records.append({
                "raw_column_name": f"spans[].attributes.{ak}",
                "normalized_field": nf,
                "field_quality": fq,
                "aurelius_signal_category": cat,
                "usable_for": uf,
                "notes": notes,
            })
    mapping_doc = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "accepted_columns": sorted(accepted),
        "rejected_columns": sorted(rejected),
        "columns": column_records,
        "nested_columns": nested_records,
    }
    with open(schema_mapping_path, "w") as fh:
        json.dump(mapping_doc, fh, indent=2, default=str, sort_keys=True)

    # 6. Analysis sample (gitignored).
    hb.update(phase="write_analysis_sample", force=True)
    analysis_bytes, analysis_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], analysis_sample_path,
    )

    # 7. Normalized sample (committed; bounded ≤100 MiB by policy).
    hb.update(phase="write_normalized_sample", force=True)
    normalized_bytes, normalized_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], normalized_sample_path,
    )
    if normalized_bytes > MAX_COMMITTED_NORMALIZED_BYTES:
        # Trim to fit the cap — shouldn't happen here (target ~1-5 MB).
        keep = max(1, int(len(normalized) * MAX_COMMITTED_NORMALIZED_BYTES /
                          normalized_bytes))
        normalized_bytes, normalized_sha = _write_jsonl(
            [_safe_jsonable(r) for r in normalized[:keep]],
            normalized_sample_path,
        )

    # 8. Fixture sample (5 rows, ≤16 KiB).
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

    # 9. Statistical rollups (per-harness/benchmark/model + numeric distribs).
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}
    for skey in ("harness", "benchmark", "request_model"):
        counts: dict[str, int] = {}
        for r in normalized:
            v = r.get(skey)
            counts[str(v)] = counts.get(str(v), 0) + 1
        rollups["subgroup_counts"][skey] = counts
    for nf in ("duration_ms", "input_tokens", "output_tokens",
               "input_messages_chars", "output_messages_chars"):
        vals: list[float] = []
        for r in normalized:
            v = r.get(nf)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            vals.sort()
            n = len(vals)

            def q(p: float, _vals=vals, _n=n) -> float:
                idx = max(0, min(_n - 1, int(round(p * (_n - 1)))))
                return _vals[idx]

            rollups["numeric_distributions"][nf] = {
                "count": n,
                "min": vals[0], "max": vals[-1],
                "mean": sum(vals) / n,
                "median": vals[n // 2],
                "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99),
            }
    with open(rollups_path, "w") as fh:
        json.dump(rollups, fh, indent=2, default=str, sort_keys=True)

    # 10. Signal coverage + sample strength.
    strength = _statistical_sample_strength(len(normalized))
    signals = _detect_signals(profile, normalized)
    available_signals = sorted(s for s, present in signals.items() if present)
    missing_signals = sorted(s for s, present in signals.items() if not present)

    # 11. Field-quality groupings (use the per-nested-key mapping + the
    # per-session column mapping). Real fields drive the available signals.
    real_fields = sorted({
        c["normalized_field"]
        for c in column_records + nested_records
        if c.get("field_quality") == "real" and c.get("normalized_field")
    })
    derived_fields = sorted({
        c["normalized_field"]
        for c in column_records + nested_records
        if c.get("field_quality") == "derived" and c.get("normalized_field")
    })
    proxy_fields = sorted({
        c["normalized_field"]
        for c in column_records + nested_records
        if c.get("field_quality") == "proxy" and c.get("normalized_field")
    })
    synthetic_fields = sorted({
        c["normalized_field"]
        for c in column_records + nested_records
        if c.get("field_quality") == "synthetic" and c.get("normalized_field")
    })
    field_quality = {}
    for c in column_records + nested_records:
        nf = c.get("normalized_field")
        fq = c.get("field_quality")
        if nf and fq and fq != "unknown":
            field_quality.setdefault(nf, fq)

    # 12. Summary
    raw_schema = sorted(profile["raw_columns"])
    normalized_schema = sorted(profile["flattened_columns"])
    summary = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": "cdla-permissive-2.0",
        "license_redistribution_status": "permissive_cdla_2",
        "license_redistribution_source": "HF card frontmatter license: cdla-permissive-2.0",
        "gated": False,
        "canonical_trace_type": "request_shape_trace",
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_bytes,
        "sample_sha256": fixture_sha,
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_bytes,
        "fixture_sample_path": os.path.relpath(fixture_path, REPO_ROOT).replace(os.sep, "/"),
        "analysis_sample_rows": len(normalized),
        "analysis_sample_bytes": analysis_bytes,
        "analysis_sample_sha256": analysis_sha,
        "analysis_sample_path": os.path.relpath(analysis_sample_path, REPO_ROOT).replace(os.sep, "/"),
        "committed_normalized_sample_rows": len(normalized),
        "committed_normalized_sample_bytes": normalized_bytes,
        "committed_normalized_sample_sha256": normalized_sha,
        "committed_normalized_sample_path": os.path.relpath(
            normalized_sample_path, REPO_ROOT).replace(os.sep, "/"),
        "committed_normalized_sample_reason_skipped": None,
        "committed_normalized_sample_materialized_at_s": time.time(),
        "committed_normalized_sample_git_sha": _git_sha(),
        "sampling_method": "head_session_then_flatten_spans",
        "stratification_keys": ["harness", "benchmark", "request_model"],
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
        "limitations": [
            f"Single parquet shard (train-00012-of-00039, {target['raw_file']}) of the "
            "39-shard 2.77 GB Exgentic agent-LLM-traces set (1,781 sessions total).",
            "OpenTelemetry span-list dataset; each row is one LLM-call span flattened from "
            "the parent agent session. duration_ms is closed-API end-to-end (network + "
            "provider serving), NOT GPU TTFT/TPOT.",
            "Models in this shard are accessed through provider APIs (azure/DeepSeek-V3.2, "
            "azure/Kimi-K2.5). Latency includes network + provider routing — do NOT use as "
            "a GPU-serving latency prior.",
            "Raw 'gen_ai.input.messages' / 'gen_ai.output.messages' / 'gen_ai.tool.definitions' "
            "payload strings (median 50K chars / max 200K) are DROPPED in the committed "
            "normalized sample; only character counts and a per-input hash remain.",
            "No GPU type, no measured TTFT/TPOT, no queue/scheduler state, no replica/autoscale "
            "signal, no cache-hit telemetry. Treat as workload-shape evidence only.",
        ],
        "provenance": (
            f"{DATASET_ID}@{config}#{target['raw_file']}"
            f"#bytes={manifest['downloaded_bytes']}#git={(_git_sha() or '')[:7]}"
        ),
        "ingestion_timestamp_s": time.time(),
        "git_sha": _git_sha(),
        "raw_download_manifest": manifest,
        "raw_committed": False,
        "schema_profile_path": os.path.relpath(schema_profile_path, REPO_ROOT).replace(os.sep, "/"),
        "schema_mapping_path": os.path.relpath(schema_mapping_path, REPO_ROOT).replace(os.sep, "/"),
        "statistical_rollups_path": os.path.relpath(rollups_path, REPO_ROOT).replace(os.sep, "/"),
        "summary_path_relative": os.path.relpath(summary_path, REPO_ROOT).replace(os.sep, "/"),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str, sort_keys=True)

    # 13. Promotion evaluation.
    hb.update(phase="promotion", force=True)
    decision = promotion.evaluate_promotion(summary)
    hb.update(phase="done", force=True)
    return {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "manifest": manifest,
        "summary": summary,
        "decision": decision,
        "audit_status": "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force-redownload", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN env var required")
        return 2

    results = []
    for t in TARGETS:
        logger.info("ingest %s/%s ... (timeout %ds)",
                    DATASET_ID, t["config_name"], PER_DATASET_TIMEOUT_S)
        t_start = time.monotonic()
        try:
            _install_timeout(PER_DATASET_TIMEOUT_S)
            r = audit_one(t, token=token, force_redownload=args.force_redownload)
        except _PerDatasetTimeout as e:
            elapsed = int(time.monotonic() - t_start)
            logger.error("  DEFERRED_TIMEOUT after %ds: %s", elapsed, e)
            r = {"dataset_id": DATASET_ID, "config_name": t["config_name"],
                 "audit_status": "DEFERRED_TIMEOUT",
                 "error": str(e), "elapsed_s": elapsed}
        except Exception as e:  # noqa: BLE001
            elapsed = int(time.monotonic() - t_start)
            logger.error("  FAILED after %ds: %s", elapsed, e)
            r = {"dataset_id": DATASET_ID, "config_name": t["config_name"],
                 "audit_status": "FAILED",
                 "error": f"{type(e).__name__}: {e}", "elapsed_s": elapsed}
        finally:
            _clear_timeout()
        results.append(r)
        if r.get("audit_status") == "ok":
            s = r["summary"]
            d = r["decision"]
            logger.info(
                "  rows=%d normalized_bytes=%d strength=%s state=%s tags=%s elapsed=%ds",
                s["analysis_sample_rows"], s["committed_normalized_sample_bytes"],
                s["statistical_sample_strength"], d["state"], d["promotion_tags"],
                int(time.monotonic() - t_start))
        else:
            logger.info("  status=%s (%s)", r.get("audit_status"), r.get("error"))

    # Cross-dataset summary
    DISC_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = DISC_DIR / "agent_llm_traces_ingest_summary.json"
    payload = {
        "doc_version": "exgentic_agent_llm_traces_ingest_summary_v1",
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
                "normalized_committed_bytes":
                    r["summary"]["committed_normalized_sample_bytes"],
                "raw_bytes_downloaded":
                    r["summary"]["raw_download_manifest"]["downloaded_bytes"],
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
