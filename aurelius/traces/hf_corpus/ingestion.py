"""Bounded ingestion + per-dataset summary writer.

The ingester is **bounded** by row + byte caps that the caller supplies:

- ``--max-rows`` defaults to 5,000. The published mission spec example
  ``agent-perf-bench/AgentPerfBench`` is small enough that the full
  ``trace_replay`` config fits, but most other datasets would balloon the
  repo if ingested in full.
- ``--max-bytes`` defaults to 16 MiB. Bytes are enforced on the
  **downloaded** parquet/jsonl/csv file size and on the **committed sample**
  size. Either cap triggers a stop.

Honesty rules:

- The bounded raw file is written to ``data/external/hf/<safe>/raw/`` and
  is **gitignored** — we never commit raw data into git.
- The bounded normalized sample is written to
  ``data/external/hf/<safe>/processed/sample.jsonl`` and committed.
- The per-dataset summary at
  ``data/external/hf/<safe>/processed/summary.json`` records: row counts,
  byte counts, raw schema, normalized schema, canonical trace type,
  available + missing + derived + proxy + synthetic field lists,
  provenance, ingestion timestamp, sample sha256, and limitations.
- Unknown columns surfaced from the parquet are **rejected** unless the
  caller passes ``--allow-unknown-columns``. This is the schema-test gate
  the promotion stage relies on.

The ingester is intentionally schema-first: ``inspect_schema(...)`` does NOT
download data; it returns the same ``HFDatasetMeta`` shape the discovery
script uses. Only after the caller approves does ``ingest_bounded(...)``
fetch the parquet file via HTTP Range.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .discovery import (
    HFAPIClient,
    USER_AGENT,
    parse_hf_metadata,
    safe_dataset_dirname,
)
from .schemas import (
    CANONICAL_TRACE_TYPE_TO_TRUST_TIER,
    CANONICAL_TRACE_TYPES,
    FIELD_QUALITY_VALUES,
)

# Map normalized-schema columns -> target signal labels. After normalization
# the schema is canonical, so signal detection becomes a simple lookup instead
# of substring matching the raw HF tags.
NORMALIZED_FIELD_TO_SIGNAL = {
    "mean_ttft_ms": "ttft", "p50_ttft_ms": "ttft", "p90_ttft_ms": "ttft",
    "p99_ttft_ms": "ttft", "ttft_p99_ms": "ttft",
    "mean_tpot_ms": "tpot", "p50_tpot_ms": "tpot", "p90_tpot_ms": "tpot",
    "p99_tpot_ms": "tpot", "tpot_p99_ms": "tpot",
    "mean_itl_ms": "itl", "p50_itl_ms": "itl",
    "p90_itl_ms": "itl", "p99_itl_ms": "itl",
    "mean_e2el_ms": "e2e_latency", "p50_e2el_ms": "e2e_latency",
    "p90_e2el_ms": "e2e_latency", "p99_e2el_ms": "e2e_latency",
    "latency_p50_ms": "latency_p50", "latency_p95_ms": "latency_p95",
    "latency_p99_ms": "latency_p99",
    "request_throughput": "request_throughput",
    "input_token_throughput": "token_throughput",
    "output_token_throughput": "token_throughput",
    "total_token_throughput": "token_throughput",
    "throughput_rps": "throughput",
    "concurrency": "concurrency",
    "batch_size": "batch_size", "sequence_length": "sequence_length",
    "prompt_tokens": "prompt_tokens", "output_tokens": "output_tokens",
    "gpu": "gpu_type", "gpu_type": "gpu_type",
    "gpu_utilization": "gpu_utilization",
    "gpu_memory_pct": "gpu_memory",
    "queue_wait_s": "queue_wait", "queue_depth": "queue_depth",
    "timeout_rate_pct": "timeout",
    "sla_violation_rate_pct": "sla",
    "cache_hit": "cache_hit", "cold_start": "cold_start",
    "prefix_id": "prefix_cache",
    "residency_state": "model_residency",
    "engine": None,  # 'vllm'/'sglang' detection handled below.
    "duration_ms": "kernel_duration",
    "duration_us": "kernel_duration",
    "kernel_name": "kernel_duration",
}


def signals_from_normalized_schema(
    normalized_schema, sample_rows=None,
) -> list[str]:
    """Return target signals implied by the normalized columns + sampled values.

    The mapping in ``NORMALIZED_FIELD_TO_SIGNAL`` covers most fields. A few
    require looking at sample values (e.g. ``engine == "vllm"`` -> ``vllm``).
    """
    out: set = set()
    for col in normalized_schema:
        sig = NORMALIZED_FIELD_TO_SIGNAL.get(col)
        if sig:
            out.add(sig)
    if sample_rows:
        for r in sample_rows[:50]:
            eng = str(r.get("engine") or "").lower()
            if "vllm" in eng:
                out.add("vllm")
            if "sglang" in eng:
                out.add("sglang")
            if "triton" in eng:
                out.add("triton")
            if "ray" in eng:
                out.add("ray_serve")
    return sorted(out)

logger = logging.getLogger(__name__)


DEFAULT_MAX_ROWS = 5_000
DEFAULT_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB
HF_FILE_RESOLVE_BASE = "https://huggingface.co/datasets"


# Column-name normalization. Datasets use inconsistent capitalisation /
# naming for the same measurement; this table is the **only** place where
# raw->normalized column renames happen so the rest of the pipeline stays
# strict-schema. Anything that isn't listed here is treated as "unknown".
RAW_TO_NORMALIZED = {
    "latency_benchmark_trace": {
        "model": "model",
        "model_family": "model_family",
        "hardware": "gpu",
        "gpu": "gpu",
        "engine": "engine",
        "profile": "profile",
        "run_id": "run_id",
        "tensor_parallelism": "tensor_parallelism",
        "concurrency": "concurrency",
        "num_requests": "num_requests",
        "duration_s": "duration_s",
        "request_throughput": "request_throughput",
        "input_token_throughput": "input_token_throughput",
        "output_token_throughput": "output_token_throughput",
        "total_token_throughput": "total_token_throughput",
        "mean_ttft_ms": "mean_ttft_ms",
        "median_ttft_ms": "p50_ttft_ms",
        "p90_ttft_ms": "p90_ttft_ms",
        "p99_ttft_ms": "p99_ttft_ms",
        "mean_tpot_ms": "mean_tpot_ms",
        "median_tpot_ms": "p50_tpot_ms",
        "p90_tpot_ms": "p90_tpot_ms",
        "p99_tpot_ms": "p99_tpot_ms",
        "mean_itl_ms": "mean_itl_ms",
        "median_itl_ms": "p50_itl_ms",
        "p90_itl_ms": "p90_itl_ms",
        "p99_itl_ms": "p99_itl_ms",
        "mean_e2el_ms": "mean_e2el_ms",
        "median_e2el_ms": "p50_e2el_ms",
        "p90_e2el_ms": "p90_e2el_ms",
        "p99_e2el_ms": "p99_e2el_ms",
    },
    "kernel_profile_trace": {
        # AgentPerfBench `kernels_labeled` config (per-kernel GEMM/attn profiles).
        "source": "source",
        "gpu": "gpu",
        "hardware": "gpu",
        "model": "model",
        "kernel_name": "kernel_name",
        "kernel_family": "kernel_family",
        "op_type": "op_type",
        "dtype": "dtype",
        "M": "m",
        "N": "n",
        "K": "k",
        "bs": "batch_size",
        "seq": "sequence_length",
        "gpu_time_duration_ms": "duration_ms",
        "duration_us": "duration_us",
        "dram_bytes_sum": "dram_bytes",
        "n_heads": "n_heads",
        "head_dim": "head_dim",
        "kv_heads": "kv_heads",
        "numel": "numel",
        "held_out": "held_out",
        "launch_block_size": "launch_block_size",
        "launch_grid_size": "launch_grid_size",
        "launch_registers_per_thread": "launch_registers_per_thread",
    },
    "request_shape_trace": {
        "request_id": "request_id",
        "timestamp_s": "timestamp_s",
        "session_id": "session_id",
        "turn_count": "turn_count",
        "prompt_tokens": "prompt_tokens",
        "output_tokens": "output_tokens",
        "model_id": "model_id",
        "model": "model_id",
    },
    "cluster_scheduler_trace": {
        "job_id": "job_id",
        "submit_time": "submit_time_s",
        "submit_time_s": "submit_time_s",
        "start_time": "start_time_s",
        "start_time_s": "start_time_s",
        "end_time": "end_time_s",
        "end_time_s": "end_time_s",
        "duration_s": "duration_s",
        "queue_wait_s": "queue_wait_s",
        "gpu_count": "gpu_count",
        "gpu_type": "gpu_type",
        "status": "status",
        "is_failed": "is_failed",
        "user_or_group": "user_or_group",
    },
    "cache_residency_trace": {
        "model_id": "model_id",
        "prefix_id": "prefix_id",
        "request_id": "request_id",
        "timestamp_s": "timestamp_s",
        "cache_hit": "cache_hit",
        "cold_start": "cold_start",
        "residency_state": "residency_state",
    },
    "telemetry_trace": {
        "timestamp_s": "timestamp_s",
        "service_id": "service_id",
        "queue_depth": "queue_depth",
        "queue_wait_s": "queue_wait_s",
        "latency_p50_ms": "latency_p50_ms",
        "latency_p95_ms": "latency_p95_ms",
        "latency_p99_ms": "latency_p99_ms",
        "ttft_p99_ms": "ttft_p99_ms",
        "tpot_p99_ms": "tpot_p99_ms",
        "throughput_rps": "throughput_rps",
        "concurrency": "concurrency",
        "replica_count": "replica_count",
        "gpu_utilization": "gpu_utilization",
        "gpu_memory_pct": "gpu_memory_pct",
        "timeout_rate_pct": "timeout_rate_pct",
        "sla_violation_rate_pct": "sla_violation_rate_pct",
    },
}


@dataclass(frozen=True)
class IngestionResult:
    dataset_id: str
    trace_type: str
    sample_rows: int
    sample_bytes: int
    sample_path: str
    summary_path: str
    sha256: str
    skipped_reason: Optional[str]
    unknown_columns: list


class IngestionBoundsExceeded(RuntimeError):
    """Raised when the bounded-download cap is breached."""


class IngestionUnknownColumns(ValueError):
    """Raised when the parquet has columns not present in RAW_TO_NORMALIZED
    for the chosen trace_type and ``allow_unknown_columns=False``."""


def _hf_file_url(dataset_id: str, path_in_repo: str) -> str:
    return (
        f"{HF_FILE_RESOLVE_BASE}/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(path_in_repo)}"
    )


def download_bounded(
    url: str, dest_path: str, *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    token: Optional[str] = None,
    timeout_s: float = 30.0,
) -> dict:
    """Bounded HTTP GET with hard ``max_bytes`` cap.

    Returns a manifest dict with ``url``, ``downloaded_bytes``,
    ``http_status``, ``dest_path``, ``truncated`` (bool). Raises
    ``IngestionBoundsExceeded`` if writing one more chunk would breach the
    cap AND the caller asked for strict bounded behaviour (we cut at the
    cap and set ``truncated=True`` instead — strict mode is
    ``max_bytes <= 0``, which we treat as input error).
    """

    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = {
        "User-Agent": USER_AGENT,
        "Range": f"bytes=0-{int(max_bytes - 1)}",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    bytes_written = 0
    truncated = False
    http_status: Optional[int] = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        http_status = resp.getcode()
        with open(dest_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                remaining = max_bytes - bytes_written
                if remaining <= 0:
                    truncated = True
                    break
                if len(chunk) > remaining:
                    out.write(chunk[:remaining])
                    bytes_written += remaining
                    truncated = True
                    break
                out.write(chunk)
                bytes_written += len(chunk)
    return {
        "url": url,
        "downloaded_bytes": bytes_written,
        "http_status": http_status,
        "dest_path": dest_path,
        "truncated": truncated,
        "max_bytes": max_bytes,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Schema-first inspection
# ---------------------------------------------------------------------------


def inspect_schema(client: HFAPIClient, dataset_id: str):
    """Return ``(HFDatasetMeta, raw_meta)``. NO data download."""
    raw = client.get(dataset_id)
    if raw is None:
        return None, None
    return parse_hf_metadata(raw), raw


# ---------------------------------------------------------------------------
# Normalisation + summary writing
# ---------------------------------------------------------------------------


def normalize_rows(
    rows: list[dict],
    trace_type: str,
    *,
    allow_unknown_columns: bool = False,
    source_dataset_id: str = "",
    provenance: str = "",
) -> tuple[list[dict], list[str], dict]:
    """Map raw rows -> normalized rows (dicts shaped for the canonical record).

    Returns ``(normalized_rows, unknown_columns, field_quality)``. Raises
    ``IngestionUnknownColumns`` when unknown columns are present and
    ``allow_unknown_columns=False`` — the schema-test gate.
    """

    if trace_type not in CANONICAL_TRACE_TYPES:
        raise ValueError(f"unknown trace_type: {trace_type}")
    if trace_type == "mixed_or_unknown_trace":
        raise ValueError("cannot normalize mixed_or_unknown_trace; classify first")
    rename = RAW_TO_NORMALIZED[trace_type]

    unknown: set = set()
    seen_normalized: set = set()
    normalized: list[dict] = []
    for r in rows:
        nrow: dict = {}
        for k, v in r.items():
            if k in rename:
                nrow[rename[k]] = v
                seen_normalized.add(rename[k])
            else:
                unknown.add(k)
        normalized.append(nrow)

    unknown_sorted = sorted(unknown)
    if unknown_sorted and not allow_unknown_columns:
        raise IngestionUnknownColumns(
            f"{source_dataset_id} -> {trace_type}: unknown columns "
            f"{unknown_sorted}; refusing normalisation. Pass "
            f"allow_unknown_columns=True only after extending RAW_TO_NORMALIZED."
        )

    field_quality = {f: "real" for f in seen_normalized}
    return normalized, unknown_sorted, field_quality


def write_summary(
    *,
    dataset_id: str,
    source_url: str,
    license_str: Optional[str],
    gated: Optional[bool],
    trace_type: str,
    sample_rows: int,
    sample_bytes: int,
    sample_sha256: str,
    raw_schema: list[str],
    normalized_schema: list[str],
    unknown_columns: list,
    field_quality: dict,
    available_signals_list: list,
    missing_signals_list: list,
    limitations: list,
    derived_fields: list,
    proxy_fields: list,
    synthetic_fields: list,
    provenance: str,
    summary_path: str,
    git_sha: Optional[str] = None,
    ingestion_timestamp_s: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    if ingestion_timestamp_s is None:
        ingestion_timestamp_s = time.time()
    summary = {
        "dataset_id": dataset_id,
        "source_url": source_url,
        "license": license_str,
        "gated": bool(gated) if gated is not None else None,
        "canonical_trace_type": trace_type,
        "trust_tier": CANONICAL_TRACE_TYPE_TO_TRUST_TIER.get(
            trace_type, "tier_6_synthetic_benchmark_data"),
        "committed_sample_rows": int(sample_rows),
        "committed_sample_bytes": int(sample_bytes),
        "sample_sha256": sample_sha256,
        "raw_schema": list(raw_schema),
        "normalized_schema": list(normalized_schema),
        "unknown_columns": list(unknown_columns),
        "field_quality": dict(field_quality),
        "available_signals": list(available_signals_list),
        "missing_signals": list(missing_signals_list),
        "derived_fields": list(derived_fields),
        "proxy_fields": list(proxy_fields),
        "synthetic_fields": list(synthetic_fields),
        "limitations": list(limitations),
        "provenance": provenance,
        "ingestion_timestamp_s": ingestion_timestamp_s,
        "git_sha": git_sha,
    }
    if extra:
        summary["extra"] = extra
    _validate_summary(summary)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def _validate_summary(s: dict) -> None:
    required = [
        "dataset_id", "source_url", "license", "gated",
        "canonical_trace_type", "trust_tier", "committed_sample_rows",
        "committed_sample_bytes", "sample_sha256", "raw_schema",
        "normalized_schema", "unknown_columns", "field_quality",
        "available_signals", "missing_signals", "derived_fields",
        "proxy_fields", "synthetic_fields", "limitations",
        "provenance", "ingestion_timestamp_s",
    ]
    missing = [k for k in required if k not in s]
    if missing:
        raise ValueError(f"summary missing required fields: {missing}")
    if s["canonical_trace_type"] not in CANONICAL_TRACE_TYPES:
        raise ValueError(
            f"canonical_trace_type='{s['canonical_trace_type']}' "
            f"not in {sorted(CANONICAL_TRACE_TYPES)}"
        )
    for k, v in s["field_quality"].items():
        if v not in FIELD_QUALITY_VALUES:
            raise ValueError(
                f"field_quality[{k}]='{v}' not in {sorted(FIELD_QUALITY_VALUES)}"
            )


# ---------------------------------------------------------------------------
# Sample writing
# ---------------------------------------------------------------------------


def write_jsonl_sample(rows: list[dict], path: str) -> tuple[int, str]:
    """Write ``rows`` as JSON-lines. Returns ``(bytes_written, sha256)``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    for r in rows:
        line = json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
        buf.write(line.encode("utf-8"))
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data), _sha256_bytes(data)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def safe_sample_paths(
    repo_root: str, dataset_id: str, config_name: Optional[str] = None,
) -> dict:
    """Build per-dataset (and optionally per-config) artefact paths.

    One HF dataset can carry several configs (``trace_replay``, ``per_layer_kernel``,
    ``kernels_labeled``, ...). Each config is ingested independently and lives in
    its own subdirectory so its summary.json + sample.jsonl don't overwrite.
    """
    safe = safe_dataset_dirname(dataset_id)
    base = os.path.join(repo_root, "data", "external", "hf", safe)
    if config_name:
        safe_cfg = safe_dataset_dirname(config_name)
        processed = os.path.join(base, config_name, "processed")
        fixture_name = f"{safe}__{safe_cfg}_sample.jsonl"
    else:
        processed = os.path.join(base, "processed")
        fixture_name = f"{safe}_sample.jsonl"
    return {
        "raw_dir": os.path.join(base, "raw"),
        "processed_dir": processed,
        "sample_path": os.path.join(processed, "sample.jsonl"),
        "summary_path": os.path.join(processed, "summary.json"),
        "fixture_path": os.path.join(
            repo_root, "tests", "fixtures", "hf", fixture_name),
    }


def ingest_from_records(
    *,
    repo_root: str,
    dataset_id: str,
    source_url: str,
    license_str: Optional[str],
    gated: Optional[bool],
    raw_records: list[dict],
    trace_type: str,
    provenance: str,
    available_signals_list: list,
    missing_signals_list: list,
    limitations: list,
    derived_fields: Optional[list] = None,
    proxy_fields: Optional[list] = None,
    synthetic_fields: Optional[list] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_unknown_columns: bool = False,
    git_sha: Optional[str] = None,
    write_fixture: bool = True,
    config_name: Optional[str] = None,
) -> IngestionResult:
    """Ingest already-loaded raw records (used by tests + parquet driver).

    The parquet path is intentionally split out: ``raw_records`` is the only
    interface the schema-test / normalisation code uses. The CLI script
    loads parquet (or any other format) and passes the parsed rows here.
    """

    if max_rows <= 0:
        raise ValueError(f"max_rows must be > 0, got {max_rows}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")

    bounded_records = raw_records[: int(max_rows)]
    raw_schema = sorted({k for r in bounded_records for k in r.keys()})

    normalized, unknown_cols, field_quality = normalize_rows(
        bounded_records, trace_type,
        allow_unknown_columns=allow_unknown_columns,
        source_dataset_id=dataset_id, provenance=provenance,
    )
    normalized_schema = sorted({k for r in normalized for k in r.keys()})

    paths = safe_sample_paths(repo_root, dataset_id, config_name)
    sample_bytes, sample_sha = write_jsonl_sample(normalized, paths["sample_path"])
    # Halve until the JSONL fits the byte cap. Stops at 1 row; if a single row
    # still overflows we surface the failure rather than silently truncating
    # mid-record.
    while sample_bytes > max_bytes and len(normalized) > 1:
        half = max(1, len(normalized) // 2)
        normalized = normalized[:half]
        sample_bytes, sample_sha = write_jsonl_sample(normalized, paths["sample_path"])
    if sample_bytes > max_bytes:
        raise IngestionBoundsExceeded(
            f"sample after truncation still {sample_bytes} bytes > "
            f"max_bytes={max_bytes} ({len(normalized)} row(s); pass a larger "
            f"--max-bytes or pre-trim columns)"
        )

    if write_fixture:
        # Tiny deterministic fixture (5 rows) for unit tests.
        fixture_rows = normalized[:5]
        write_jsonl_sample(fixture_rows, paths["fixture_path"])

    # Merge caller-supplied signals with signals implied by the actual
    # ingested normalized schema + sampled engine values. This catches per-
    # config signals (e.g. kernel_duration) that the dataset-wide HF metadata
    # alone may miss.
    inferred = set(signals_from_normalized_schema(normalized_schema, normalized))
    merged_signals = sorted(set(available_signals_list or []) | inferred)
    from .discovery import TARGET_SIGNALS
    merged_missing = [s for s in TARGET_SIGNALS if s not in set(merged_signals)]

    summary = write_summary(
        dataset_id=dataset_id,
        source_url=source_url,
        license_str=license_str,
        gated=gated,
        trace_type=trace_type,
        sample_rows=len(normalized),
        sample_bytes=sample_bytes,
        sample_sha256=sample_sha,
        raw_schema=raw_schema,
        normalized_schema=normalized_schema,
        unknown_columns=unknown_cols,
        field_quality=field_quality,
        available_signals_list=merged_signals,
        missing_signals_list=merged_missing,
        limitations=limitations,
        derived_fields=derived_fields or [],
        proxy_fields=proxy_fields or [],
        synthetic_fields=synthetic_fields or [],
        provenance=provenance,
        summary_path=paths["summary_path"],
        git_sha=git_sha,
    )
    return IngestionResult(
        dataset_id=dataset_id,
        trace_type=trace_type,
        sample_rows=summary["committed_sample_rows"],
        sample_bytes=summary["committed_sample_bytes"],
        sample_path=paths["sample_path"],
        summary_path=paths["summary_path"],
        sha256=sample_sha,
        skipped_reason=None,
        unknown_columns=unknown_cols,
    )


# ---------------------------------------------------------------------------
# Parquet loader (optional — only used by CLI; tests use JSON fixtures)
# ---------------------------------------------------------------------------


def try_load_parquet_rows(path: str, *, max_rows: int) -> Optional[list[dict]]:
    """Best-effort parquet -> list[dict]. Returns None if pyarrow missing.

    Avoids forcing a pyarrow runtime dependency on test paths.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None
    table = pq.read_table(path)
    if table.num_rows > max_rows:
        table = table.slice(0, int(max_rows))
    return table.to_pylist()


def try_load_json_rows(path: str, *, max_rows: int) -> list[dict]:
    """Load a json or jsonl file as ``list[dict]`` (bounded)."""
    with open(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected JSON list")
            return [r for r in data[:max_rows] if isinstance(r, dict)]
        rows: list[dict] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= max_rows:
                break
        return rows
