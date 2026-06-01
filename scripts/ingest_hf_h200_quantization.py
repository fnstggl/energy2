#!/usr/bin/env python3
"""Round 6 broadened HF discovery — bounded ingest of ssakethch/h200-quantization-benchmarks.

This script extends the federated benchmark corpus with one new Tier-4
``latency_benchmark_trace`` dataset that closes the H200 hardware gap:

* ``ssakethch/h200-quantization-benchmarks`` — NVIDIA H200 SXM (141 GB
  HBM3e) MIG-partitioned vLLM serving benchmark. 275 rows of measured
  TTFT (mean/median/p99), TPOT (mean/median/p99), ITL (mean/median/p99),
  req/output/total token throughput, successful/failed counts, duration,
  input/output tokens, across 40 quantized + non-quantized instruction-
  tuned LLMs × 5 quantizations (AWQ, GPTQ, FP8, BF16, NVFP4) × 5 request
  rates (1, 2, 4, 8, 16).

The round-6 audit also records 8 round-5 candidates as discovery-only
rejection / deferral records:

* ``sairamn/gcp-cloud-billing-cost`` — SYNTHETIC GCP billing data
* ``ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1`` — SYNTHETIC ML eval
* ``ClarusC64/datacenter-power-load-coherence-risk-v0.1`` — SYNTHETIC ML eval
* ``Phipper/pe-energy-infrastructure-training-data`` — finance-LLM SFT, no infra
* ``uohna/llm_inference_energy_combined.parquet`` — empty repository
* ``metrum-ai/llm-perf-dashboard`` — 404 deleted
* ``crozai/vllm-benchmark-coding`` — ShareGPT-derived workload input, duplicate
* ``intellistream/sage-agent-benchmark`` — agent capability eval, no infra

NO production claims. NO scheduler / controller / robust-energy-engine changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

FIXTURE_MAX_BYTES = 16 * 1024
COMMITTED_NORMALIZED_MAX_BYTES = 100 * 1024
MAX_RAW_DOWNLOAD_BYTES = 2 * 1024 * 1024  # 2 MiB cap (file is ~41 KiB)

DATASET_ID = "ssakethch/h200-quantization-benchmarks"
SAFE_NAME = "ssakethch__h200-quantization-benchmarks"
CONFIG = "throughput"
RAW_FILE = "data/throughput.csv"
LICENSE: Optional[str] = None  # README front-matter has no `license:` field
GATED = False

logger = logging.getLogger("aurelius.hf_h200_quantization_ingest")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def _hf_url(dataset_id: str, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/{path}"
    )


def _download(url: str, dest: Path, *, max_bytes: int) -> dict:
    """Streaming download with a hard byte-cap. Returns a manifest dict."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-discovery/round6"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    written = 0
    truncated = False
    err = None
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            with open(dest, "wb") as fh:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    if written + len(chunk) > max_bytes:
                        chunk = chunk[: max_bytes - written]
                        fh.write(chunk)
                        written += len(chunk)
                        truncated = True
                        break
                    fh.write(chunk)
                    written += len(chunk)
    except urllib.error.HTTPError as e:
        err = f"HTTPError {e.code}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {
        "url": url,
        "dest": str(dest.relative_to(REPO_ROOT)) if dest.exists() else None,
        "downloaded_bytes": written,
        "truncated": truncated,
        "elapsed_s": round(time.time() - t0, 3),
        "error": err,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return ""


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    vs = sorted(vals)
    n = len(vs)

    def _q(p: float) -> float:
        if n == 1:
            return float(vs[0])
        k = max(0.0, min(p * (n - 1), n - 1))
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return float(vs[lo] + (vs[hi] - vs[lo]) * (k - lo))

    return {
        "count": n,
        "min": float(vs[0]),
        "max": float(vs[-1]),
        "mean": float(statistics.fmean(vs)),
        "p50": _q(0.50),
        "p90": _q(0.90),
        "p95": _q(0.95),
        "p99": _q(0.99),
    }


def _norm_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "null", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_int(v: Any) -> Optional[int]:
    f = _norm_num(v)
    if f is None:
        return None
    return int(f)


def _norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "null", "none"}:
        return None
    return s


def _model_family(model: Optional[str]) -> Optional[str]:
    """Bucket models into a family for stratification (Llama-3.1, Qwen3, ...)."""
    if not model:
        return None
    m = model.lower()
    if "llama-3.1" in m:
        return "Llama-3.1"
    if "llama-3.2" in m:
        return "Llama-3.2"
    if "qwen3" in m:
        return "Qwen3"
    if "qwen2.5" in m or "qwen-2.5" in m:
        return "Qwen2.5"
    if "deepseek-r1-distill-qwen" in m or "deepseek_r1_distill_qwen" in m:
        return "DeepSeek-R1-Distill-Qwen"
    if "gemma" in m:
        return "Gemma"
    return "Other"


# ---------------------------------------------------------------------------
# Row normaliser
# ---------------------------------------------------------------------------


def _normalize_row(raw: dict, idx: int, git_sha: str) -> dict:
    """Project one raw throughput.csv row into the canonical schema."""
    model = _norm_str(raw.get("model"))
    return {
        "source_dataset_id": DATASET_ID,
        "trace_type": "latency_benchmark_trace",
        "provenance": (
            f"{DATASET_ID}@{CONFIG}#{RAW_FILE}#row={idx}"
            f"#git={git_sha[:7]}"
        ),
        "row_index": idx,
        "model_id": model,
        "model_family": _model_family(model),
        "quantization": _norm_str(raw.get("quant")),
        "request_rate": _norm_num(raw.get("request_rate")),  # concurrent rps
        "concurrency": _norm_num(raw.get("request_rate")),  # alias
        "num_successful": _norm_int(raw.get("successful_reqs")),
        "num_failed": _norm_int(raw.get("failed_reqs")),
        "duration_s": _norm_num(raw.get("duration_s")),
        "input_tokens": _norm_int(raw.get("input_tokens")),
        "output_tokens": _norm_int(raw.get("output_tokens")),
        "request_throughput": _norm_num(raw.get("req_throughput")),
        "output_token_throughput": _norm_num(raw.get("output_tok_throughput")),
        "peak_output_token_throughput": _norm_num(
            raw.get("peak_output_tok_throughput")),
        "total_token_throughput": _norm_num(raw.get("total_tok_throughput")),
        "mean_ttft_ms": _norm_num(raw.get("mean_ttft_ms")),
        "median_ttft_ms": _norm_num(raw.get("median_ttft_ms")),
        "p99_ttft_ms": _norm_num(raw.get("p99_ttft_ms")),
        "mean_tpot_ms": _norm_num(raw.get("mean_tpot_ms")),
        "median_tpot_ms": _norm_num(raw.get("median_tpot_ms")),
        "p99_tpot_ms": _norm_num(raw.get("p99_tpot_ms")),
        "mean_itl_ms": _norm_num(raw.get("mean_itl_ms")),
        "median_itl_ms": _norm_num(raw.get("median_itl_ms")),
        "p99_itl_ms": _norm_num(raw.get("p99_itl_ms")),
        # Constant metadata shared by every row.
        "gpu_type": "NVIDIA H200 SXM",
        "gpu_memory_gb": 141,
        "gpu_partition": "MIG",
        "engine": "vllm",
    }


# ---------------------------------------------------------------------------
# Field quality + signal vocabulary
# ---------------------------------------------------------------------------


def _field_quality() -> dict:
    return {
        # Measured columns from upstream throughput.csv.
        "model_id": "real",
        "quantization": "real",
        "request_rate": "real",
        "concurrency": "derived",  # alias of request_rate
        "num_successful": "real",
        "num_failed": "real",
        "duration_s": "real",
        "input_tokens": "real",
        "output_tokens": "real",
        "request_throughput": "real",
        "output_token_throughput": "real",
        "peak_output_token_throughput": "real",
        "total_token_throughput": "real",
        "mean_ttft_ms": "real",
        "median_ttft_ms": "real",
        "p99_ttft_ms": "real",
        "mean_tpot_ms": "real",
        "median_tpot_ms": "real",
        "p99_tpot_ms": "real",
        "mean_itl_ms": "real",
        "median_itl_ms": "real",
        "p99_itl_ms": "real",
        # Derived / constant fields.
        "model_family": "derived",
        "gpu_type": "derived",  # constant per dataset card
        "gpu_memory_gb": "derived",
        "gpu_partition": "derived",
        "engine": "derived",
        # Missing for Aurelius decision-making.
        "queue_wait": "missing",
        "queue_depth": "missing",
        "gpu_utilization": "missing",
        "memory_pressure": "missing",
        "kv_cache_size": "missing",
        "cache_hit": "missing",
        "batch_size": "missing",
        "engine_version": "missing",
        "timeout_label": "missing",
        "sla_label": "missing",
        "autoscaling": "missing",
        "replica_count": "missing",
        "carbon_intensity": "missing",
        "energy_per_request": "missing",
        "cost_per_token": "missing",
        "cost_per_request": "missing",
    }


def _build_columns_map(raw_columns: list[str]) -> list[dict]:
    """Map raw throughput.csv columns to Aurelius signal categories."""
    mapping: dict[str, tuple[str, str, str, list[str], str]] = {
        "model": (
            "model_id", "real", "metadata_only",
            ["latency_prior", "throughput_prior"],
            "Upstream model identifier (HF org/name).",
        ),
        "quant": (
            "quantization", "real", "metadata_only",
            ["latency_prior", "throughput_prior"],
            "Quantisation regime: awq / gptq / fp8 / bf16 / nvfp4.",
        ),
        "request_rate": (
            "request_rate", "real", "request_arrival",
            ["latency_prior", "throughput_prior",
             "constraint_aware_backtest"],
            "Configured request rate (requests/s) for the vLLM benchmark "
            "harness. NOT a real arrival trace.",
        ),
        "successful_reqs": (
            "num_successful", "real", "request_completion",
            ["throughput_prior"],
            "Successful request count for this (model, quant, rate) cell.",
        ),
        "failed_reqs": (
            "num_failed", "real", "failure_timeout",
            ["latency_prior"],
            "Failed request count. >0 at rr=8 only for "
            "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic.",
        ),
        "duration_s": (
            "duration_s", "real", "request_completion",
            ["throughput_prior"],
            "Wall time for the run, seconds.",
        ),
        "input_tokens": (
            "input_tokens", "real", "tokens",
            ["latency_prior", "throughput_prior"],
            "Total input tokens across all requests in this cell.",
        ),
        "output_tokens": (
            "output_tokens", "real", "tokens",
            ["throughput_prior"],
            "Total generated tokens across all requests in this cell.",
        ),
        "req_throughput": (
            "request_throughput", "real", "throughput",
            ["throughput_prior"],
            "Request throughput (req/s) over the wall-clock duration.",
        ),
        "output_tok_throughput": (
            "output_token_throughput", "real", "throughput",
            ["throughput_prior"],
            "Output token throughput (tok/s) — generation phase.",
        ),
        "peak_output_tok_throughput": (
            "peak_output_token_throughput", "real", "throughput",
            ["throughput_prior"],
            "Peak observed output-token throughput in this run.",
        ),
        "total_tok_throughput": (
            "total_token_throughput", "real", "throughput",
            ["throughput_prior"],
            "Total (input+output) token throughput (tok/s).",
        ),
        "mean_ttft_ms": (
            "mean_ttft_ms", "real", "latency", ["latency_prior"],
            "Mean time-to-first-token, milliseconds.",
        ),
        "median_ttft_ms": (
            "median_ttft_ms", "real", "latency", ["latency_prior"],
            "Median time-to-first-token, milliseconds.",
        ),
        "p99_ttft_ms": (
            "p99_ttft_ms", "real", "latency", ["latency_prior"],
            "p99 time-to-first-token, milliseconds.",
        ),
        "mean_tpot_ms": (
            "mean_tpot_ms", "real", "latency", ["latency_prior"],
            "Mean time-per-output-token, milliseconds.",
        ),
        "median_tpot_ms": (
            "median_tpot_ms", "real", "latency", ["latency_prior"],
            "Median time-per-output-token, milliseconds.",
        ),
        "p99_tpot_ms": (
            "p99_tpot_ms", "real", "latency", ["latency_prior"],
            "p99 time-per-output-token, milliseconds.",
        ),
        "mean_itl_ms": (
            "mean_itl_ms", "real", "latency", ["latency_prior"],
            "Mean inter-token latency, milliseconds.",
        ),
        "median_itl_ms": (
            "median_itl_ms", "real", "latency", ["latency_prior"],
            "Median inter-token latency, milliseconds.",
        ),
        "p99_itl_ms": (
            "p99_itl_ms", "real", "latency", ["latency_prior"],
            "p99 inter-token latency, milliseconds.",
        ),
    }
    out: list[dict] = []
    for col in raw_columns:
        norm = mapping.get(col)
        if norm is None:
            out.append({
                "raw_column_name": col,
                "normalized_field": col,
                "dtypes": ["str"],
                "field_quality": "real",
                "aurelius_signal_category": "metadata_only",
                "usable_for": [],
                "units": None,
                "notes": "Unmapped column — recorded as metadata_only.",
            })
            continue
        nf, fq, sig, usable, notes = norm
        out.append({
            "raw_column_name": col,
            "normalized_field": nf,
            "dtypes": ["str", "float", "int"],
            "field_quality": fq,
            "aurelius_signal_category": sig,
            "usable_for": usable,
            "units": "ms" if "ms" in nf else (
                "s" if nf.endswith("_s") else (
                    "tok/s" if "throughput" in nf and "request" not in nf else (
                        "req/s" if nf in {"request_throughput", "request_rate"}
                        else None
                    ))),
            "notes": notes,
        })
    return out


# ---------------------------------------------------------------------------
# Discovery-only audit records (8 round-5 candidates not ingested)
# ---------------------------------------------------------------------------


ROUND6_DISCOVERY_ONLY: list[dict] = [
    {
        "dataset_id": "sairamn/gcp-cloud-billing-cost",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_synthetic_economics",
        "reason": (
            "Auditing data.csv shows clearly SYNTHETIC GCP billing data: "
            "sequential resource_NNN IDs, random-looking CPU/memory "
            "utilisation paired with arbitrary services (e.g. 'Artifact "
            "Registry' with 78 GB and 93% CPU util), no README content, "
            "no provenance, no measurement methodology. Rejected per the "
            "binding directive 'DO NOT treat synthetic cost fields as real "
            "economics'. The data does not close the operational × economic "
            "join gap."
        ),
        "bucket": "D_synthetic_economics",
    },
    {
        "dataset_id":
            "ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_synthetic_eval_task",
        "reason": (
            "Synthetic text-classification 'coherence-risk' eval task "
            "(n<1K rows, scorer.py + train.csv + tester.csv). Not an "
            "infrastructure dataset — it is an LLM-output evaluation "
            "fixture. No measured serving / scheduling / cost / carbon "
            "telemetry. Rejected as irrelevant to the Aurelius constraint-"
            "aware objective."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id":
            "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_synthetic_eval_task",
        "reason": (
            "Same author / template as the AI-load coherence-risk dataset. "
            "Synthetic text-classification eval fixture (n<1K, scorer.py "
            "+ train.csv + tester.csv). Not real datacenter power-load "
            "telemetry. Rejected as irrelevant."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id": "Phipper/pe-energy-infrastructure-training-data",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "reject_irrelevant_domain",
        "reason": (
            "Private-equity energy-infrastructure finance LLM training "
            "fixtures (dpo_pairs.jsonl + opus_reasoning.jsonl + "
            "sft_conversations.jsonl, 1K<n<10K). Domain = finance / "
            "private-equity reasoning, NOT serving / scheduling / "
            "datacenter telemetry. Tagged 'energy-infrastructure' but "
            "refers to PE asset-class language, not Aurelius signals. "
            "Rejected."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id": "uohna/llm_inference_energy_combined.parquet",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_empty_repository",
        "reason": (
            "Repository contains only .gitattributes — no actual data file "
            "is published despite the .parquet repo name. usedStorage=0. "
            "Cannot be ingested."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id": "metrum-ai/llm-perf-dashboard",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_repository_not_found",
        "reason": (
            "HF API returns 404 ('Repository not found'). The sister "
            "dataset metrum-ai/llm-perfdata is already ingested in the "
            "corpus (multi_source_curated_v1 config). No further "
            "data-source available from Metrum AI."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id": "crozai/vllm-benchmark-coding",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "ShareGPT-derived coding-focused conversation fixtures used as "
            "INPUT to vLLM's benchmark_serving.py (not the benchmark "
            "RESULTS). Single column 'conversations' (role/value pairs). "
            "Duplicate of the existing sharegpt_aiperf workload-shape "
            "ingester. No measured TTFT / TPOT / throughput / GPU "
            "telemetry. License unspecified. Rejected as duplicate."
        ),
        "bucket": "F_irrelevant",
    },
    {
        "dataset_id": "intellistream/sage-agent-benchmark",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_capability_benchmark_no_infra",
        "reason": (
            "SAGE agent capability benchmark (tool selection, task "
            "planning, timing judgement, ~11K QA samples). Evaluates "
            "agent accuracy / correctness, NOT serving / scheduling / "
            "infrastructure cost. No measured latency / queue / GPU / "
            "energy. Rejected — capability-only benchmark adds nothing "
            "to the Aurelius constraint-aware decision engine."
        ),
        "bucket": "F_irrelevant",
    },
]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest(*, output_root: Path = HF_DIR) -> dict:
    t0 = time.time()
    git_sha = _git_sha()

    base = output_root / SAFE_NAME
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = base / CONFIG / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download raw CSV (gitignored).
    raw_local = raw_dir / Path(RAW_FILE).name
    manifest = _download(
        _hf_url(DATASET_ID, RAW_FILE), raw_local,
        max_bytes=MAX_RAW_DOWNLOAD_BYTES,
    )
    if manifest["error"]:
        raise RuntimeError(
            f"{DATASET_ID}: download failed for {RAW_FILE}: "
            f"{manifest['error']}"
        )
    logger.info(
        "%s :: downloaded %d bytes in %.2fs",
        DATASET_ID, manifest["downloaded_bytes"], manifest["elapsed_s"],
    )

    # 2. Parse CSV → list of dicts.
    with open(raw_local) as fh:
        reader = csv.DictReader(fh)
        raw_rows = list(reader)
        raw_columns = reader.fieldnames or []
    n_rows = len(raw_rows)
    logger.info("%s :: parsed %d rows × %d cols", DATASET_ID, n_rows,
                len(raw_columns))

    # 3. Normalise rows.
    normalized_rows = [
        _normalize_row(r, idx, git_sha) for idx, r in enumerate(raw_rows)
    ]

    # 4. Analysis sample (gitignored — full normalised set).
    analysis_path = proc_dir / "analysis_sample.jsonl"
    buf = io.BytesIO()
    for row in normalized_rows:
        buf.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    analysis_path.write_bytes(buf.getvalue())
    analysis_sha = _sha256(analysis_path)
    analysis_size = analysis_path.stat().st_size

    # 5. Fixture — 5 rows, deterministic, stratified to cover the diverse
    #    quant × concurrency cells. Always pick the first-occurring row
    #    matching each strata key so the output is fully reproducible.
    fixture_path = FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_target_keys = [
        ("bf16", 1.0),
        ("fp8", 4.0),
        ("awq", 8.0),
        ("gptq", 16.0),
        ("nvfp4", 4.0),
    ]
    fixture_indexes: list[int] = []
    fixture_rows: list[dict] = []
    used = set()
    for q, rate in fixture_target_keys:
        for idx, row in enumerate(normalized_rows):
            if idx in used:
                continue
            if row.get("quantization") == q and row.get("request_rate") == rate:
                fixture_rows.append(row)
                fixture_indexes.append(idx)
                used.add(idx)
                break
    fixture_buf = io.BytesIO()
    fixture_committed = 0
    for row in fixture_rows:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if fixture_buf.tell() + len(line) > FIXTURE_MAX_BYTES:
            break
        fixture_buf.write(line)
        fixture_committed += 1
    fixture_path.write_bytes(fixture_buf.getvalue())
    fixture_sha = _sha256(fixture_path)
    fixture_size = fixture_path.stat().st_size

    # 6. License is UNKNOWN → no committed normalised analysis sample.
    committed_norm_path = None
    committed_norm_rows = 0
    committed_norm_bytes = 0
    committed_norm_sha = None
    committed_norm_reason_skipped = (
        "license_unspecified_no_redistribution_promise"
    )

    # 7. Schema profile.
    dtypes: dict[str, list[str]] = {}
    example_values: dict[str, list[str]] = {}
    missing_rates: dict[str, float] = {}
    for col in raw_columns:
        seen: set[str] = set()
        examples: list[str] = []
        n_missing = 0
        for r in raw_rows:
            v = r.get(col)
            if v is None or v == "" or (isinstance(v, str) and
                                        v.lower() in {"nan", "null"}):
                n_missing += 1
                continue
            # CSV values are strings; try float to capture numeric semantics.
            try:
                float(v)
                seen.add("float")
            except ValueError:
                seen.add("str")
            if len(examples) < 3:
                examples.append(repr(v)[:120])
        if not seen:
            seen.add("str")
        dtypes[col] = sorted(seen)
        example_values[col] = examples
        missing_rates[col] = round(n_missing / max(1, n_rows), 4)

    schema_profile = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "source_files": [RAW_FILE],
        "inspected_row_count": n_rows,
        "raw_columns": list(raw_columns),
        "nested_keys": [],
        "dtypes": dtypes,
        "example_values": example_values,
        "missing_rates": missing_rates,
        "unknown_columns": [],
        "rejected_columns": [],
        "accepted_columns": list(raw_columns),
        "list_length_summaries": {},
        "file_size_bytes": raw_local.stat().st_size,
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # 8. Schema mapping.
    columns_map = _build_columns_map(list(raw_columns))
    schema_mapping = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "accepted_columns": list(raw_columns),
        "rejected_columns": [],
        "unknown_columns": [],
        "columns": columns_map,
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # 9. Statistical rollups — overall + per (quantization, request_rate).
    overall: dict[str, dict] = {}
    for k in [
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "request_throughput", "output_token_throughput",
        "total_token_throughput", "peak_output_token_throughput",
        "duration_s", "input_tokens", "output_tokens",
        "num_successful", "num_failed",
    ]:
        vals = [r[k] for r in normalized_rows if r.get(k) is not None]
        if vals:
            overall[k] = _percentiles(vals)

    by_strata: dict[str, dict] = {}
    subgroup_counts: dict[str, int] = {}
    groups: dict[tuple, list[dict]] = {}
    for row in normalized_rows:
        key = (row.get("quantization"), row.get("request_rate"))
        groups.setdefault(key, []).append(row)
    for key, grp in groups.items():
        label = f"quant={key[0] or 'unknown'}|request_rate={key[1] or 'unknown'}"
        subgroup_counts[label] = len(grp)
        if len(grp) >= 5:
            stats_for_group: dict[str, dict] = {}
            for k in [
                "mean_ttft_ms", "p99_ttft_ms",
                "mean_tpot_ms", "p99_tpot_ms",
                "mean_itl_ms", "p99_itl_ms",
                "output_token_throughput", "request_throughput",
            ]:
                vals = [r[k] for r in grp if r.get(k) is not None]
                if vals:
                    stats_for_group[k] = _percentiles(vals)
            if stats_for_group:
                by_strata[label] = stats_for_group

    rollups = {
        "overall": overall,
        "by_strata": by_strata,
        "subgroup_counts": subgroup_counts,
        "stratification_keys": ["quantization", "request_rate"],
        "insufficient_sample_groups": [
            label for label, n in subgroup_counts.items() if n < 5
        ],
    }
    (proc_dir / "statistical_rollups.json").write_text(
        json.dumps(rollups, indent=2, sort_keys=True)
    )

    # 10. Sample strength — 275 rows with ≥5 cells per (quant, request_rate)
    #     cross-tab makes this STRONG by the corpus convention.
    has_strata = bool(by_strata)
    strength = "strong" if (n_rows >= 200 and has_strata) else (
        "moderate" if n_rows >= 25 else "weak"
    )

    normalized_schema = sorted({
        k for r in normalized_rows for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    available_signals = [
        "ttft", "tpot", "itl", "throughput",
        "concurrency", "input_tokens", "output_tokens",
        "gpu_type", "model_id", "engine",
        "request_arrival",  # request_rate as arrival proxy
        "failure_label",    # num_failed > 0 for one cell
        "request_completion",  # num_successful + duration_s
    ]
    missing_signals = [
        "e2e_latency",
        "queue_wait", "queue_depth",
        "gpu_utilization", "memory_pressure",
        "batch_size", "engine_version",
        "kv_cache_size", "cache_hit",
        "kernel_duration",
        "timeout_label", "sla_label",
        "autoscaling", "replica_count",
        "carbon_intensity", "energy_per_request",
        "cost_per_token", "cost_per_request",
    ]

    field_quality = _field_quality()

    limitations = [
        "ssakethch/h200-quantization-benchmarks: benchmark results for "
        "40 quantized + non-quantized instruction-tuned LLMs on NVIDIA "
        "H200 SXM (141 GB HBM3e) MIG-partitioned instances via vLLM. "
        "Trust Tier 4 (latency_benchmark_trace) — NOT pilot telemetry.",
        "LICENSE = unspecified (no `license:` field in the HF dataset "
        "card YAML front-matter). Recorded license=None. Bounded "
        "normalised sample is NOT committed for this dataset "
        "(license_redistribution_status=unspecified). 5-row schema-test "
        "fixture is committed as fair-use schema evidence only.",
        "request_rate is the CONFIGURED concurrency level (1, 2, 4, 8, "
        "16) for the vLLM benchmark_serving.py harness, NOT a real "
        "arrival trace. Treat as a closed-loop throughput sweep, not a "
        "scheduler arrival process.",
        "GPU coverage = NVIDIA H200 SXM only. This dataset is the "
        "corpus' FIRST measured-source H200 entry (metrum-ai/llm-perfdata "
        "carries 10 multi-source-curated H200 rows but those are not a "
        "single-campaign measurement). Generalisation OUTSIDE H200 SXM "
        "MIG is unsafe.",
        "Engine = vLLM only. Engine version is NOT recorded. "
        "Generalisation to SGLang / TGI / Triton / TensorRT-LLM is "
        "unsafe.",
        "Single failure observation: rr=8 with "
        "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic shows "
        "num_failed=1. dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4 "
        "at rr=16 shows a ~34 s mean_ttft_ms — saturation regime, not a "
        "stable measurement. Aurelius MUST treat the rr=16 highest-"
        "concurrency cells as backpressure-saturated outliers when "
        "fitting latency priors.",
        "NO queue / wait / scheduler-state / KV-cache / GPU-utilisation "
        "telemetry. Aurelius queue-risk, batch-frontier, KV-cache, and "
        "energy-cost modules MUST NOT consume this dataset.",
        "NO cost / energy / carbon-intensity / billing fields. "
        "Goodput/$ denominator MUST remain operator-policy-supplied + "
        "public-pricing-prior + ElectricityMaps / ENTSO-E carbon "
        "intensity (already integrated). This dataset does NOT close "
        "the operational × economic join gap.",
        "model_family is a derived bucket (Llama-3.1 / Llama-3.2 / "
        "Qwen3 / Qwen2.5 / DeepSeek-R1-Distill-Qwen / Gemma / Other) — "
        "do not treat as upstream-author-supplied metadata.",
        "gpu_type / gpu_memory_gb / gpu_partition / engine are CONSTANT "
        "per dataset card and labelled field_quality=derived. They are "
        "not measured per row.",
        "NVFP4 coverage is SMALL — only 5 rows (1 model × 5 request "
        "rates). Subgroup p95/p99 claims for quant=nvfp4 are statistically "
        "weak even though overall strength=strong.",
        "BENCHMARK-ONLY — Tier 4. Not pilot telemetry. Pilot telemetry "
        "remains the only Tier 1 calibration source.",
    ]

    summary_obj = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": LICENSE,
        "gated": GATED,
        "canonical_trace_type": "latency_benchmark_trace",
        "raw_committed": False,
        "raw_file_size_committed": False,
        "raw_columns": list(raw_columns),
        "raw_schema": list(raw_columns),
        "normalized_schema": normalized_schema,
        "available_signals": available_signals,
        "missing_signals": missing_signals,
        "derived_fields": [k for k, v in field_quality.items()
                           if v == "derived"],
        "proxy_fields": [k for k, v in field_quality.items()
                         if v == "proxy"],
        "synthetic_fields": [k for k, v in field_quality.items()
                             if v == "synthetic"],
        "real_fields": [k for k, v in field_quality.items() if v == "real"],
        "field_quality": field_quality,
        "limitations": limitations,
        "stratification_keys": ["quantization", "request_rate"],
        "sampling_method": "full_bounded",
        "fixture_sample_rows": fixture_committed,
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "fixture_row_indexes": fixture_indexes,
        "fixture_sample_strata_keys": ["quantization", "request_rate"],
        "analysis_sample_rows": n_rows,
        "analysis_sample_bytes": analysis_size,
        "analysis_sample_sha256": analysis_sha,
        "analysis_sample_path": str(analysis_path.relative_to(REPO_ROOT)),
        "committed_normalized_sample_rows": committed_norm_rows,
        "committed_normalized_sample_bytes": committed_norm_bytes,
        "committed_normalized_sample_path": committed_norm_path,
        "committed_normalized_sample_sha256": committed_norm_sha,
        "committed_normalized_sample_reason_skipped": committed_norm_reason_skipped,
        "committed_sample_rows": fixture_committed,
        "committed_sample_bytes": fixture_size,
        "sample_sha256": fixture_sha,
        "subgroup_counts": subgroup_counts,
        "statistical_sample_strength": strength,
        "schema_profile_path": str(
            (proc_dir / "schema_profile.json").relative_to(REPO_ROOT)
        ),
        "schema_mapping_path": str(
            (proc_dir / "schema_mapping.json").relative_to(REPO_ROOT)
        ),
        "statistical_rollups_path": str(
            (proc_dir / "statistical_rollups.json").relative_to(REPO_ROOT)
        ),
        "summary_path_relative": str(
            (proc_dir / "summary.json").relative_to(REPO_ROOT)
        ),
        "provenance": f"{DATASET_ID}@{CONFIG}#{RAW_FILE}#git={git_sha[:7]}",
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": [{"file": RAW_FILE, **manifest}],
        "elapsed_s": round(time.time() - t0, 3),
        "unknown_columns": [],
    }
    (proc_dir / "summary.json").write_text(
        json.dumps(summary_obj, indent=2, sort_keys=True)
    )

    decision = promotion.evaluate_promotion(summary_obj)
    entry = promotion.build_registry_entry(summary_obj, decision)
    return {
        "summary": summary_obj,
        "decision": decision,
        "registry_entry": entry,
        "raw_manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Registry + candidates updates
# ---------------------------------------------------------------------------


def _merge_into_registry(new_entries: list[dict]) -> Path:
    registry_path = DISC_DIR / "canonical_corpus_registry.json"
    existing = promotion.load_canonical_registry(str(registry_path))
    entries = list((existing or {}).get("entries", []))

    new_keys = {(e["dataset_id"], e["config_name"]) for e in new_entries}
    entries = [
        e for e in entries
        if (e["dataset_id"], e["config_name"]) not in new_keys
    ]
    entries.extend(new_entries)
    entries.sort(key=lambda e: (e["dataset_id"], e["config_name"] or ""))
    promotion.write_canonical_registry(entries, str(registry_path))
    return registry_path


def _update_candidate_registry(path: Path, ingested_summary: dict) -> None:
    if not path.exists():
        return
    d = json.loads(path.read_text())
    cands = d.get("candidates", [])

    target_id = ingested_summary["dataset_id"]
    ingested_keywords = [
        "round6::economic_priority_pass",
        "round6::h200_mig_vllm_quantization_benchmark",
        "latency_benchmark::ttft_tpot_itl_p99",
        "gpu_coverage::h200_sxm_141gb_hbm3e",
        "engine_coverage::vllm",
        "quantization_coverage::awq_gptq_fp8_bf16_nvfp4",
    ]
    found = False
    for c in cands:
        if c.get("dataset_id") == target_id:
            found = True
            c["recommended_action"] = "ingest_now_bounded"
            c["candidate_trace_type"] = "latency_benchmark_trace"
            c["trust_level"] = "tier_4_latency_benchmark_traces"
            c["gated_status"] = "public"
            c["license"] = "unspecified"
            c["matched_keywords"] = sorted(set(
                c.get("matched_keywords", []) + ingested_keywords
            ))
            c["overall_priority_score"] = 4.0
            c["ingestion_feasibility_score"] = 5
            c["frontier_value_score"] = 4
            c["schema_quality_score"] = 5
            c["production_similarity_score"] = 2
            c["aurelius_use_case"] = (
                "First measured-source NVIDIA H200 SXM (141 GB HBM3e) "
                "MIG-partitioned vLLM serving latency benchmark in the "
                "federated corpus. 275 rows of real mean/median/p99 "
                "TTFT/TPOT/ITL + per-cell throughput across 40 models × "
                "5 quantizations × 5 request rates. Closes the H200 "
                "single-source gap and adds NVFP4 quantization coverage."
            )
            c["not_recommended_uses"] = [
                "Real arrival / queue scheduling (closed-loop benchmark)",
                "Dynamic frontier calibration (Tier-4 benchmark)",
                "Goodput/$ denominator calibration (no measured cost/energy)",
                "Generalisation outside H200 SXM MIG + vLLM",
                "Subgroup percentile claims for quant=nvfp4 (only 5 rows)",
                "Treating rr=16 highest-concurrency cells as stable "
                "non-saturation measurements",
            ]
            break

    if not found:
        cands.append({
            "dataset_id": target_id,
            "dataset_url": f"https://huggingface.co/datasets/{target_id}",
            "gated_status": "public",
            "license": "unspecified",
            "estimated_size": ["n<1K"],
            "available_splits": ["train"],
            "schema_available": True,
            "matched_keywords": ingested_keywords,
            "candidate_trace_type": "latency_benchmark_trace",
            "trust_level": "tier_4_latency_benchmark_traces",
            "available_signals": ingested_summary["available_signals"],
            "missing_signals": ingested_summary["missing_signals"],
            "aurelius_use_case": (
                "First measured-source NVIDIA H200 SXM (141 GB HBM3e) "
                "MIG vLLM serving latency benchmark in the federated "
                "corpus."
            ),
            "not_recommended_uses": [],
            "ingestion_feasibility_score": 5,
            "frontier_value_score": 4,
            "schema_quality_score": 5,
            "production_similarity_score": 2,
            "overall_priority_score": 4.0,
            "recommended_action": "ingest_now_bounded",
            "feature_names": ingested_summary["raw_columns"],
            "classification_evidence": {
                "latency_benchmark_trace": [
                    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                    "mean_tpot_ms", "p99_tpot_ms",
                    "mean_itl_ms", "p99_itl_ms",
                    "request_rate",
                    "output_tok_throughput", "total_tok_throughput",
                ],
            },
            "downloads": 38,
            "likes": 1,
            "last_modified": None,
            "discovery_timestamp_s": time.time(),
            "configs": [CONFIG],
        })

    # Round-6 discovery-only records.
    existing_ids = {c.get("dataset_id") for c in cands}
    for rec in ROUND6_DISCOVERY_ONLY:
        if rec["dataset_id"] in existing_ids:
            # Update with round-6 audit note.
            for c in cands:
                if c.get("dataset_id") == rec["dataset_id"]:
                    c["recommended_action"] = rec["kind"]
                    c["round6_audit_reason"] = rec["reason"]
                    c["round6_audit_bucket"] = rec["bucket"]
            continue
        cands.append({
            "dataset_id": rec["dataset_id"],
            "dataset_url": (
                f"https://huggingface.co/datasets/{rec['dataset_id']}"
            ),
            "gated_status": "gated_auto" if rec.get("gated") else "public",
            "license": rec.get("license_observed"),
            "estimated_size": [],
            "available_splits": [],
            "schema_available": False,
            "matched_keywords": [f"round6::{rec['kind']}"],
            "candidate_trace_type": rec["candidate_trace_type"],
            "trust_level": "tier_6_synthetic_benchmark_data",
            "available_signals": [],
            "missing_signals": [],
            "aurelius_use_case": "Round-6 discovery audit — see reason.",
            "not_recommended_uses": [],
            "ingestion_feasibility_score": 1,
            "frontier_value_score": 1,
            "schema_quality_score": 1,
            "production_similarity_score": 1,
            "overall_priority_score": 1.0,
            "recommended_action": rec["kind"],
            "feature_names": [],
            "classification_evidence": {},
            "downloads": 0,
            "likes": 0,
            "last_modified": None,
            "discovery_timestamp_s": time.time(),
            "configs": [],
            "round6_audit_reason": rec["reason"],
            "round6_audit_bucket": rec["bucket"],
        })

    d["candidates"] = cands
    d["candidate_count"] = len(cands)
    path.write_text(json.dumps(d, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Round 6 audit summary
# ---------------------------------------------------------------------------


def _write_round6_audit_summary(ingest_out: dict) -> Path:
    out = DISC_DIR / "round6_broadened_discovery_audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = ingest_out["summary"]
    decision = ingest_out["decision"]
    payload = {
        "doc_version": "round6_broadened_discovery_audit_summary_v1",
        "audited_at_s": time.time(),
        "scope": (
            "Round 6 broadened HF discovery — bounded ingest of "
            "ssakethch/h200-quantization-benchmarks (Tier-4 "
            "latency_benchmark_trace; first measured-source H200 SXM "
            "vLLM benchmark in the federated corpus) plus discovery-only "
            "rejection records for the 8 remaining round-5 candidates."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "ingested": [{
            "dataset_id": summary["dataset_id"],
            "config_name": summary["config_name"],
            "license": summary["license"],
            "gated": summary["gated"],
            "canonical_trace_type": summary["canonical_trace_type"],
            "analysis_sample_rows": summary["analysis_sample_rows"],
            "fixture_sample_rows": summary["fixture_sample_rows"],
            "committed_normalized_sample_rows":
                summary["committed_normalized_sample_rows"],
            "committed_normalized_sample_bytes":
                summary["committed_normalized_sample_bytes"],
            "committed_normalized_sample_reason_skipped":
                summary["committed_normalized_sample_reason_skipped"],
            "statistical_sample_strength":
                summary["statistical_sample_strength"],
            "available_signals": summary["available_signals"],
            "missing_signals": summary["missing_signals"],
            "promotion_state": decision["state"],
            "promotion_tags": decision["promotion_tags"],
            "promotion_reasons": decision["reasons"],
            "limitations": summary["limitations"],
            "subgroup_counts": summary["subgroup_counts"],
        }],
        "failed": [],
        "discovery_only_records": ROUND6_DISCOVERY_ONLY,
        "economic_priority_summary": {
            "datasets_with_operational_and_economic_signals": [],
            "datasets_with_economic_only_signals": [],
            "join_keys_available_for_economic_overlays": [
                "gpu_type", "model_id", "engine", "quantization",
                "request_rate",
            ],
            "scorer_coefficients_calibratable_from_round6": [],
            "scorer_coefficients_operator_policy_only_after_round6": [
                "gpu_hour_price_usd",
                "kwh_per_request",
                "carbon_g_per_kwh",
                "spot_interruption_probability",
                "egress_cost_per_gb",
                "regional_price_usd_per_mwh",
            ],
            "negative_result_finding": (
                "Round 6 inspected the 9 round-5 discovery-only "
                "candidates. ONE qualified for bounded ingest "
                "(ssakethch/h200-quantization-benchmarks — H200 latency "
                "benchmark with measured TTFT/TPOT/ITL p50/p99). EIGHT "
                "were rejected as synthetic, irrelevant, deleted, "
                "duplicate, or capability-only. The single ingested "
                "dataset carries NO economic columns. Aurelius' goodput/$ "
                "denominator therefore remains operator-policy-supplied "
                "+ public-pricing-prior + ElectricityMaps/ENTSO-E carbon "
                "intensity. This confirms the round-5 finding that the "
                "public HF dataset space does NOT currently close the "
                "operational × economic join gap."
            ),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    out = ingest()
    logger.info(
        "%s@%s :: rows=%d strength=%s promotion=%s tags=%s",
        DATASET_ID, CONFIG,
        out["summary"]["analysis_sample_rows"],
        out["summary"]["statistical_sample_strength"],
        out["decision"]["state"],
        out["decision"]["promotion_tags"],
    )

    if args.dry_run:
        return 0

    registry_path = _merge_into_registry([out["registry_entry"]])
    logger.info("Updated registry at %s", registry_path)

    candidates_path = DISC_DIR / "hf_dataset_candidates.json"
    _update_candidate_registry(candidates_path, out["summary"])
    logger.info("Updated candidate registry at %s", candidates_path)

    audit_path = _write_round6_audit_summary(out)
    logger.info("Wrote audit summary %s", audit_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
