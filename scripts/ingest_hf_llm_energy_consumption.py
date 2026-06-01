#!/usr/bin/env python3
"""Round-4 broadened HF discovery — bounded ingest of
``ejhusom/llm-inference-energy-consumption``.

This script extends the federated benchmark corpus with one Tier-4
``latency_benchmark_trace`` dataset that fills a gap none of the existing
HF-ingested datasets cover: **measured per-request energy consumption
across consumer/laptop AND workstation hardware tiers with an Ollama
serving stack**.

Why it matters for Aurelius:

* Every existing Tier-4 / energy-bearing dataset in the corpus
  (AgentPerfBench, Odyn, Memoriant, Intellistream, optimum-benchmark,
  ssong1/llmperf-bedrock) measures GPU-direct serving on **server-class
  hardware only** (A100 / A10 / T4 / DGX Spark / 32vCPU Sapphire-Rapids /
  bedrock-closed-API). Aurelius placement / routing / deferral decisions
  for edge / consumer-tier workloads have **NO public prior at all**.
* `ejhusom/llm-inference-energy-consumption` (Husom et al. 2024,
  arxiv:2407.16893 "The Price of Prompting") captures the same
  workload (Alpaca + Code-Feedback prompts) on:
  * laptop1 (CPU only, gemma 2b / codellama 7b)
  * laptop2 (with GPU, gemma 2b / 7b / llama3 8b / codellama 7b)
  * workstation (with GPU, gemma 2b / 7b, codellama 7b / 70b)
  * server (CPU only, llama3 70b)
* This lets the Aurelius energy term reason about cross-tier placement
  for the first time. It is also the first dataset in the corpus that
  uses the Ollama serving stack (all others are vLLM / Triton / Ray
  Serve / llama.cpp / Bedrock).

Trust: **Tier 4** (`latency_benchmark_trace`). Not pilot telemetry.

License: **CC-BY-SA-4.0** (redistribution-friendly with attribution +
share-alike). Committed normalised samples retain attribution + license
in the per-config summary.json.

Configs ingested (4 of 16 available):

* `alpaca_gemma_7b_laptop2`               (laptop2, gemma:7b, alpaca)
* `alpaca_gemma_7b_workstation`           (workstation, gemma:7b, alpaca)
* `codefeedback_codellama_7b_workstation` (workstation, codellama:7b, codefeedback)
* `codefeedback_codellama_70b_workstation`(workstation, codellama:70b, codefeedback)

These four cover three independent comparisons:
* Cross-tier (laptop2 vs workstation, both running gemma:7b alpaca)
* Cross-workload (alpaca vs codefeedback, both on workstation/7b)
* Cross-model-size (codellama:7b vs codellama:70b on workstation)

Outputs (per config):

    data/external/hf/ejhusom__llm-inference-energy-consumption/raw/<file>
        # gitignored
    data/external/hf/ejhusom__llm-inference-energy-consumption/<config>/processed/
        schema_profile.json                 (committed)
        schema_mapping.json                 (committed)
        summary.json                        (committed)
        statistical_rollups.json            (committed)
        committed_normalized_sample.jsonl   (committed, ≤ 100 KiB / config)
        analysis_sample.jsonl               (gitignored)
    tests/fixtures/hf/ejhusom__llm-inference-energy-consumption__<config>_sample.jsonl
                                            (committed, ≤ 16 KiB, 5 rows)

It also writes a Round-4 discovery audit summary at
``data/external/hf_discovery/round4_broadened_discovery_audit_summary.json``
recording the ingested rows + the negative-result discovery records.

NO production claims. NO scheduler / controller / robust-energy-engine
changes.
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
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

# Per-config bounds. Each CSV ≤ 30 MiB; we trim linguistic columns when
# building the normalised sample so committed samples stay << 100 KiB / cfg.
FIXTURE_MAX_BYTES = 16 * 1024
COMMITTED_NORMALIZED_MAX_BYTES_PER_CONFIG = 100 * 1024  # 100 KiB
RAW_FILE_MAX_BYTES = 30 * 1024 * 1024  # 30 MiB per raw CSV
PER_DATASET_TIMEOUT_S = 30 * 60

DATASET_ID = "ejhusom/llm-inference-energy-consumption"
SAFE_NAME = "ejhusom__llm-inference-energy-consumption"
LICENSE = "cc-by-sa-4.0"
GATED = False

logger = logging.getLogger("aurelius.hf_llm_energy_ingest")


# ---------------------------------------------------------------------------
# Run manifest (4 representative configs)
# ---------------------------------------------------------------------------


RUNS = [
    {
        "config_name": "alpaca_gemma_7b_laptop2",
        "csv_path": "data/alpaca_gemma_7b_laptop2.csv",
        "hardware_tier": "laptop",
        "hardware_id": "laptop2",
        "model_name": "gemma:7b",
        "model_family": "gemma",
        "model_size_b": 7.0,
        "prompt_dataset": "alpaca",
    },
    {
        "config_name": "alpaca_gemma_7b_workstation",
        "csv_path": "data/alpaca_gemma_7b_workstation.csv",
        "hardware_tier": "workstation",
        "hardware_id": "workstation",
        "model_name": "gemma:7b",
        "model_family": "gemma",
        "model_size_b": 7.0,
        "prompt_dataset": "alpaca",
    },
    {
        "config_name": "codefeedback_codellama_7b_workstation",
        "csv_path": "data/codefeedback_codellama_7b_workstation.csv",
        "hardware_tier": "workstation",
        "hardware_id": "workstation",
        "model_name": "codellama:7b",
        "model_family": "codellama",
        "model_size_b": 7.0,
        "prompt_dataset": "codefeedback",
    },
    {
        "config_name": "codefeedback_codellama_70b_workstation",
        "csv_path": "data/codefeedback_codellama_70b_workstation.csv",
        "hardware_tier": "workstation",
        "hardware_id": "workstation",
        "model_name": "codellama:70b",
        "model_family": "codellama",
        "model_size_b": 70.0,
        "prompt_dataset": "codefeedback",
    },
]


# Raw CSV columns that get propagated into the normalised sample. We
# DROP the verbose `prompt` + `response` text columns and ALL linguistic
# features (sentiment, readability, etc.) — they are not Aurelius signals
# and would blow up the committed sample size 10×.
NORMALIZED_COLUMNS = {
    # Identity
    "index": ("index", "int"),
    "model_name": ("model_name", "str"),
    "type": ("prompt_type_raw", "str"),
    # Timestamps (real)
    "created_at": ("created_at", "str"),
    "start_time": ("start_time", "str"),
    "end_time": ("end_time", "str"),
    "clock_duration": ("clock_duration_str", "str"),
    # Durations (real, in nanoseconds per Ollama API)
    "total_duration": ("total_duration_ns", "float"),
    "load_duration": ("load_duration_ns", "float"),
    "prompt_duration": ("prompt_duration_ns", "float"),
    "response_duration": ("response_duration_ns", "float"),
    # Tokens (real)
    "prompt_token_length": ("prompt_token_length", "int"),
    "response_token_length": ("response_token_length", "int"),
    # Energy (real, kWh)
    "energy_consumption_monitoring": ("energy_kwh_monitoring", "float"),
    "energy_consumption_llm_cpu": ("energy_kwh_llm_cpu", "float"),
    "energy_consumption_llm_gpu": ("energy_kwh_llm_gpu", "float"),
    "energy_consumption_llm_total": ("energy_kwh_llm_total", "float"),
    "energy_consumption_llm": ("energy_kwh_llm", "float"),
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def _hf_url(dataset_id: str, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"
        f"{urllib.parse.quote(path)}"
    )


def _download(url: str, dest: Path, *, max_bytes: int) -> dict:
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "url": url, "dest": str(dest),
            "downloaded_bytes": dest.stat().st_size,
            "cached": True, "truncated": False, "error": None,
        }
    req = urllib.request.Request(url, headers=headers)
    truncated = False
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            with open(dest, "wb") as fh:
                while True:
                    chunk = r.read(128 * 1024)
                    if not chunk:
                        break
                    remaining = max_bytes - written
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(chunk) > remaining:
                        fh.write(chunk[:remaining])
                        written += remaining
                        truncated = True
                        break
                    fh.write(chunk)
                    written += len(chunk)
        return {
            "url": url, "dest": str(dest),
            "downloaded_bytes": written,
            "cached": False, "truncated": truncated, "error": None,
        }
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, ConnectionError) as e:
        return {
            "url": url, "dest": str(dest),
            "downloaded_bytes": written,
            "cached": False, "truncated": False, "error": str(e),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)

    def _q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = (len(s) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "count": len(s),
        "min": s[0],
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "p90": _q(0.90),
        "p95": _q(0.95),
        "p99": _q(0.99),
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


# ---------------------------------------------------------------------------
# Ingest one config
# ---------------------------------------------------------------------------


def _parse_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Parse a CSV with potentially huge text fields. Returns (header, rows)."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        header = list(rdr.fieldnames or [])
        rows = list(rdr)
    return header, rows


def _coerce(value, kind: str):
    if value is None or value == "":
        return None
    if kind == "int":
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value)


def _normalize_row(
    raw_row: dict, run: dict, git_sha: str, idx: int,
) -> dict:
    out: dict = {
        "source_dataset_id": DATASET_ID,
        "trace_type": "latency_benchmark_trace",
        "provenance": (
            f"{DATASET_ID}@{run['config_name']}#"
            f"{Path(run['csv_path']).name}#"
            f"git={git_sha[:7]}"
        ),
        "config_name": run["config_name"],
        "hardware_tier": run["hardware_tier"],
        "hardware_id": run["hardware_id"],
        "model_family": run["model_family"],
        "model_size_b": run["model_size_b"],
        "prompt_dataset": run["prompt_dataset"],
        "engine": "ollama",
        "row_index_in_config": idx,
    }
    for raw_col, (norm_field, kind) in NORMALIZED_COLUMNS.items():
        out[norm_field] = _coerce(raw_row.get(raw_col), kind)
    # Derived signals (label as 'derived'): TTFT and TPOT proxies.
    # Ollama exposes `prompt_duration` (time to compute prompt -> first
    # token) and `response_duration` (time generating all output tokens).
    # TTFT proxy = prompt_duration (ns).
    pd_ns = out.get("prompt_duration_ns")
    rd_ns = out.get("response_duration_ns")
    rtl = out.get("response_token_length")
    out["ttft_proxy_ms"] = (pd_ns / 1.0e6) if pd_ns is not None else None
    out["e2e_latency_ms"] = (
        (out.get("total_duration_ns") / 1.0e6)
        if out.get("total_duration_ns") is not None else None
    )
    if rd_ns is not None and rtl is not None and rtl > 1:
        # TPOT proxy (ms / token) = response_duration / response_tokens
        out["tpot_proxy_ms_per_token"] = (rd_ns / 1.0e6) / rtl
        out["request_output_throughput_tps"] = (rtl / (rd_ns / 1.0e9)) if rd_ns > 0 else None
    else:
        out["tpot_proxy_ms_per_token"] = None
        out["request_output_throughput_tps"] = None
    return out


def _build_columns_map() -> list[dict]:
    return [
        {
            "raw_column_name": "model_name",
            "normalized_field": "model_name",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["workload_shape_only", "latency_prior"],
            "notes": "Ollama model id (e.g. gemma:7b, codellama:70b).",
        },
        {
            "raw_column_name": "type",
            "normalized_field": "prompt_type_raw",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["workload_shape_only"],
            "notes": (
                "Ollama `done_reason` (always 'unknown' in this dataset "
                "snapshot — Ollama did not yet emit a labelled reason). "
                "Kept for forward-compat."
            ),
        },
        {
            "raw_column_name": "created_at",
            "normalized_field": "created_at",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": "ISO-8601 timestamp with timezone",
            "aurelius_signal_category": "request_completion",
            "usable_for": ["workload_shape_only"],
            "notes": "Ollama-reported response completion timestamp.",
        },
        {
            "raw_column_name": "start_time",
            "normalized_field": "start_time",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": "ISO-8601 timestamp with timezone",
            "aurelius_signal_category": "request_arrival",
            "usable_for": ["workload_shape_only"],
            "notes": "Client-side request start (wall clock).",
        },
        {
            "raw_column_name": "end_time",
            "normalized_field": "end_time",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": "ISO-8601 timestamp with timezone",
            "aurelius_signal_category": "request_completion",
            "usable_for": ["workload_shape_only"],
            "notes": "Client-side request end (wall clock).",
        },
        {
            "raw_column_name": "clock_duration",
            "normalized_field": "clock_duration_str",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": "human-formatted duration",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior"],
            "notes": "Wall-clock duration string. Use total_duration_ns instead.",
        },
        {
            "raw_column_name": "total_duration",
            "normalized_field": "total_duration_ns",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "nanoseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": (
                "Ollama-reported end-to-end request duration in ns "
                "(includes model load + prompt eval + generation)."
            ),
        },
        {
            "raw_column_name": "load_duration",
            "normalized_field": "load_duration_ns",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "nanoseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior", "cache_residency_evaluation"],
            "notes": (
                "Time to load the model into memory. Effectively zero "
                "when the model is already resident (residency / cold-"
                "start proxy)."
            ),
        },
        {
            "raw_column_name": "prompt_duration",
            "normalized_field": "prompt_duration_ns",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "nanoseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior"],
            "notes": (
                "Ollama prompt-eval duration; proxies TTFT (time to "
                "first token) under a single-stream serving setup."
            ),
        },
        {
            "raw_column_name": "response_duration",
            "normalized_field": "response_duration_ns",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "nanoseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": (
                "Ollama generation duration; divided by output_token_count "
                "gives TPOT (ms / token) under single-stream."
            ),
        },
        {
            "raw_column_name": "prompt_token_length",
            "normalized_field": "prompt_token_length",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "latency_prior"],
            "notes": "Input-token count (Ollama tokenizer).",
        },
        {
            "raw_column_name": "response_token_length",
            "normalized_field": "response_token_length",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "throughput_prior"],
            "notes": "Output-token count (Ollama tokenizer).",
        },
        {
            "raw_column_name": "energy_consumption_monitoring",
            "normalized_field": "energy_kwh_monitoring",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "kWh",
            "aurelius_signal_category": "cost_energy_carbon",
            "usable_for": ["latency_prior"],
            "notes": (
                "Energy used by the monitoring/measurement framework "
                "itself (CodeCarbon overhead). Subtract from total when "
                "building energy priors."
            ),
        },
        {
            "raw_column_name": "energy_consumption_llm_cpu",
            "normalized_field": "energy_kwh_llm_cpu",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "kWh",
            "aurelius_signal_category": "cost_energy_carbon",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": "CPU-side energy per request (real, CodeCarbon).",
        },
        {
            "raw_column_name": "energy_consumption_llm_gpu",
            "normalized_field": "energy_kwh_llm_gpu",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "kWh",
            "aurelius_signal_category": "cost_energy_carbon",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": (
                "GPU-side energy per request (real, CodeCarbon). "
                "Zero when the run was CPU-only (laptop1 / server)."
            ),
        },
        {
            "raw_column_name": "energy_consumption_llm_total",
            "normalized_field": "energy_kwh_llm_total",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "kWh",
            "aurelius_signal_category": "cost_energy_carbon",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": "CPU+GPU per-request energy (real).",
        },
        {
            "raw_column_name": "energy_consumption_llm",
            "normalized_field": "energy_kwh_llm",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "kWh",
            "aurelius_signal_category": "cost_energy_carbon",
            "usable_for": ["latency_prior"],
            "notes": (
                "Alternative per-request LLM energy column. Usually "
                "equals energy_consumption_llm_total."
            ),
        },
        {
            "raw_column_name": "index",
            "normalized_field": "index",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["workload_shape_only"],
            "notes": "Position of the request in the source experiment log.",
        },
    ]


# Columns in the source CSV that we deliberately drop from the
# normalised sample (text content + 60-odd linguistic features). Kept as
# `rejected_columns` for traceability so the schema profile records WHY.
DROPPED_COLUMNS = [
    "",  # leading unnamed
    "Unnamed: 0",
    "Unnamed: 0.1",
    "prompt",
    "response",
    # Linguistic features
    "word_count", "sentence_count", "avg_word_length", "word_diversity",
    "unique_word_count", "avg_sentence_length", "punctuation_count",
    "stop_word_count", "long_word_count", "named_entity_count",
    "noun_count", "verb_count", "adj_count", "adverb_count",
    "pronoun_count", "prop_adverbs", "prop_pronouns",
    "sentiment_polarity", "sentiment_subjectivity",
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    "smog_index", "automated_readability_index", "coleman_liau_index",
    "linsear_write_formula", "dale_chall_readability_score",
    "text_standard", "spache_readability", "mcalpine_eflaw",
    "reading_time", "fernandez_huerta", "szigriszt_pazos",
    "gutierrez_polini", "crawford", "osman", "gulpease_index",
    "wiener_sachtextformel", "syllable_count", "lexicon_count",
    "char_count", "letter_count", "polysyllabcount", "monosyllabcount",
    "question_marks", "exclamation_marks", "sentence_embedding_variance",
    "personal_pronouns", "named_entities", "adjectives", "adverbs",
    "length_x_complexity", "questions_about_entities",
    "desc_complexity_ratio", "word_count_squared",
    "avg_sentence_length_cubed", "lexical_diversity",
]


def ingest_config(
    *, run: dict, output_root: Path, git_sha: str,
) -> dict:
    t0 = time.time()
    base = output_root / SAFE_NAME
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = base / run["config_name"] / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_local = raw_dir / Path(run["csv_path"]).name
    raw_manifest = _download(
        _hf_url(DATASET_ID, run["csv_path"]),
        raw_local, max_bytes=RAW_FILE_MAX_BYTES,
    )
    if raw_manifest["error"]:
        raise RuntimeError(
            f"download failed for {run['config_name']}: "
            f"{raw_manifest['error']}"
        )

    logger.info(
        "ingest config=%s file_bytes=%d truncated=%s",
        run["config_name"], raw_manifest["downloaded_bytes"],
        raw_manifest["truncated"],
    )

    raw_header, raw_rows = _parse_csv(raw_local)

    # Normalize rows
    normalized: list[dict] = []
    for i, raw_row in enumerate(raw_rows):
        normalized.append(_normalize_row(raw_row, run, git_sha, i))

    if not normalized:
        raise RuntimeError(
            f"No rows parsed from {raw_local} for {run['config_name']}"
        )

    # Write analysis sample (gitignored, full normalized rows)
    analysis_path = proc_dir / "analysis_sample.jsonl"
    with open(analysis_path, "wb") as fh:
        for row in normalized:
            fh.write((json.dumps(row, sort_keys=True) + "\n").encode())
    analysis_sha = _sha256(analysis_path)
    analysis_size = analysis_path.stat().st_size

    # Committed normalized sample (≤ 100 KiB per config). License is
    # cc-by-sa-4.0 → redistribution-safe with attribution.
    committed_path = proc_dir / "committed_normalized_sample.jsonl"
    committed_buf = io.BytesIO()
    committed_rows_count = 0
    for row in normalized:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if (committed_buf.tell() + len(line)
                > COMMITTED_NORMALIZED_MAX_BYTES_PER_CONFIG):
            break
        committed_buf.write(line)
        committed_rows_count += 1
    committed_path.write_bytes(committed_buf.getvalue())
    committed_size = committed_path.stat().st_size
    committed_sha = _sha256(committed_path)

    # Fixture (5 rows, ≤ 16 KiB)
    fixture_path = (
        FIXTURES_DIR
        / f"{SAFE_NAME}__{run['config_name']}_sample.jsonl"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_buf = io.BytesIO()
    fixture_rows = 0
    for row in normalized:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if fixture_buf.tell() + len(line) > FIXTURE_MAX_BYTES:
            break
        if fixture_rows >= 5:
            break
        fixture_buf.write(line)
        fixture_rows += 1
    fixture_path.write_bytes(fixture_buf.getvalue())
    fixture_size = fixture_path.stat().st_size
    fixture_sha = _sha256(fixture_path)

    # Schema profile
    raw_columns = list(raw_header)
    accepted_columns = list(NORMALIZED_COLUMNS.keys())
    # rejected_columns: those in raw_columns we deliberately dropped.
    rejected_columns = [c for c in raw_columns if c in DROPPED_COLUMNS]
    # unknown_columns: those in raw_columns NOT in either accepted or
    # rejected — should be empty.
    unknown_columns = [
        c for c in raw_columns
        if c not in NORMALIZED_COLUMNS and c not in DROPPED_COLUMNS
    ]

    # Missing rates for accepted columns
    missing_rates: dict = {}
    example_values: dict = {}
    dtypes_map: dict = {}
    for raw_col in accepted_columns:
        n_missing = 0
        examples: list[str] = []
        seen_dtypes: set[str] = set()
        for r in raw_rows:
            v = r.get(raw_col)
            if v is None or v == "":
                n_missing += 1
                continue
            seen_dtypes.add(type(v).__name__)
            if len(examples) < 3:
                examples.append(repr(v)[:120])
        missing_rates[raw_col] = (
            n_missing / max(1, len(raw_rows))
        )
        example_values[raw_col] = examples
        dtypes_map[raw_col] = sorted(seen_dtypes or {"str"})

    schema_profile = {
        "dataset_id": DATASET_ID,
        "config_name": run["config_name"],
        "source_files": [run["csv_path"]],
        "inspected_row_count": len(raw_rows),
        "raw_columns": raw_columns,
        "nested_keys": [],
        "dtypes": dtypes_map,
        "example_values": example_values,
        "missing_rates": missing_rates,
        "unknown_columns": unknown_columns,
        "rejected_columns": rejected_columns,
        "accepted_columns": accepted_columns,
        "list_length_summaries": {},
        "file_size_bytes": raw_local.stat().st_size,
        "raw_file_truncated": bool(raw_manifest["truncated"]),
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # Schema mapping
    schema_mapping = {
        "dataset_id": DATASET_ID,
        "config_name": run["config_name"],
        "accepted_columns": accepted_columns,
        "rejected_columns": rejected_columns,
        "unknown_columns": unknown_columns,
        "columns": _build_columns_map(),
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # Statistical rollups
    rollup_keys = [
        ("e2e_latency_ms", "e2e_latency_ms"),
        ("ttft_proxy_ms", "ttft_proxy_ms"),
        ("tpot_proxy_ms_per_token", "tpot_proxy_ms_per_token"),
        ("request_output_throughput_tps", "request_output_throughput_tps"),
        ("energy_kwh_llm_total", "energy_kwh_llm_total"),
        ("energy_kwh_llm_cpu", "energy_kwh_llm_cpu"),
        ("energy_kwh_llm_gpu", "energy_kwh_llm_gpu"),
        ("prompt_token_length", "prompt_token_length"),
        ("response_token_length", "response_token_length"),
    ]
    overall = {}
    for k_out, k_in in rollup_keys:
        vals = [
            r.get(k_in) for r in normalized
            if r.get(k_in) is not None
        ]
        if vals:
            overall[k_out] = _percentiles(vals)

    # GPU energy is zero on CPU-only tiers; mark that explicitly.
    nz_gpu = sum(
        1 for r in normalized
        if r.get("energy_kwh_llm_gpu") and r["energy_kwh_llm_gpu"] > 0
    )
    has_gpu_energy = nz_gpu > (len(normalized) // 2)

    rollups = {
        "overall": overall,
        "subgroup_counts": {
            f"hardware_tier={run['hardware_tier']}": len(normalized),
            f"model={run['model_name']}": len(normalized),
            f"prompt_dataset={run['prompt_dataset']}": len(normalized),
        },
        "rows_with_nonzero_gpu_energy": nz_gpu,
        "fraction_rows_with_gpu_energy": nz_gpu / max(1, len(normalized)),
        "has_gpu_energy_signal": has_gpu_energy,
    }
    (proc_dir / "statistical_rollups.json").write_text(
        json.dumps(rollups, indent=2, sort_keys=True)
    )

    # Sample strength: per-config 161 / 3109 / 5099 / 8735 rows.
    # >= 5000 rows = strong, 1000-5000 = moderate, 100-1000 = weak,
    # < 100 = fixture_only.
    if len(normalized) >= 5000:
        strength = "strong"
    elif len(normalized) >= 1000:
        strength = "moderate"
    elif len(normalized) >= 100:
        strength = "weak"
    else:
        strength = "fixture_only"

    # Available + missing signals
    available_signals = [
        "e2e_latency",
        "ttft_proxy",
        "tpot_proxy",
        "throughput_derived",
        "input_tokens",
        "output_tokens",
        "energy_kwh_cpu",
        "energy_kwh_total",
        "model_id",
        "engine",
        "hardware_tier",
        "model_load_event",  # via load_duration
    ]
    if has_gpu_energy:
        available_signals.append("energy_kwh_gpu")
    missing_signals = [
        "ttft_measured",  # we only have ollama prompt_duration as TTFT proxy
        "itl",
        "queue_state",
        "queue_wait",
        "queue_depth",
        "memory_pressure",
        "gpu_utilization",
        "gpu_type",  # not in schema; hardware_tier only
        "batch_size",
        "concurrency",  # single-stream
        "timeout_label",
        "sla_label",
        "autoscaling",
        "replica_count",
        "kv_cache_size",
        "cache_hit",
        "kernel_duration",
        "carbon_intensity",  # need to combine with regional grid
    ]
    if not has_gpu_energy:
        missing_signals.append("energy_kwh_gpu")

    field_quality = {
        "e2e_latency_ms": "derived",  # from total_duration_ns / 1e6
        "ttft_proxy_ms": "derived",   # from prompt_duration_ns
        "tpot_proxy_ms_per_token": "derived",
        "request_output_throughput_tps": "derived",
        "total_duration_ns": "real",
        "prompt_duration_ns": "real",
        "response_duration_ns": "real",
        "load_duration_ns": "real",
        "prompt_token_length": "real",
        "response_token_length": "real",
        "energy_kwh_llm_cpu": "real",
        "energy_kwh_llm_gpu": "real",
        "energy_kwh_llm_total": "real",
        "energy_kwh_llm": "real",
        "energy_kwh_monitoring": "real",
        "model_name": "real",
        "created_at": "real",
        "start_time": "real",
        "end_time": "real",
        "clock_duration_str": "real",
        "prompt_type_raw": "real",
        "hardware_tier": "real",
        "hardware_id": "real",
        "model_family": "real",
        "model_size_b": "real",
        "prompt_dataset": "real",
        "engine": "real",
        "row_index_in_config": "real",
        "index": "real",
        # Missing for Aurelius decision-making
        "gpu_type": "missing",
        "gpu_utilization": "missing",
        "queue_wait": "missing",
        "queue_depth": "missing",
        "batch_size": "missing",
        "kv_cache_size": "missing",
        "ttft_measured": "missing",
        "itl": "missing",
        "concurrency": "missing",
    }

    limitations = [
        f"{DATASET_ID}: cc-by-sa-4.0 benchmark output from the SINTEF "
        f"Digital + Singapore Management University study "
        "(arxiv:2407.16893). Trust Tier 4 (latency_benchmark_trace).",
        "ENGINE: Ollama HTTP API serving stack — NOT vLLM / SGLang / "
        "Triton / Ray Serve. Ollama internally uses llama.cpp; durations "
        "reflect llama.cpp prompt-eval + generation timing under the "
        "Ollama wrapper. Do NOT generalise to vLLM / TGI throughput "
        "without independent validation.",
        "TTFT IS A PROXY: `prompt_duration` measures Ollama prompt "
        "evaluation, which approximates TTFT under single-stream serving "
        "but is NOT a measured first-token wall-clock timestamp. "
        "Schema_mapping labels ttft_proxy_ms as DERIVED.",
        "CONCURRENCY = 1 across all runs — Ollama API was driven by a "
        "synchronous client. NO queue / contention / batching signal. "
        "The Aurelius queue-risk and batch-frontier modules MUST NOT "
        "consume this dataset.",
        f"HARDWARE TIER: {run['hardware_tier']} ({run['hardware_id']}) "
        f"— a single device per config. Do NOT extrapolate absolute "
        f"latency / energy numbers to other devices in the same tier "
        f"without independent validation.",
        "ENERGY MEASUREMENT: CodeCarbon (per the upstream paper). "
        "Energy values are real but depend on the regional grid "
        "intensity for any carbon-cost conversion — combine with "
        "ElectricityMaps / regional CO2 g/kWh data before reporting "
        "carbon cost.",
        "BENCHMARK ONLY — Tier 4. Pilot telemetry remains the only "
        "Tier 1 calibration source.",
        "type='unknown' across all rows — Ollama did not emit a "
        "structured done_reason in this snapshot. NO failure / SLA / "
        "timeout label available.",
        "Single model per config — do NOT cross-merge "
        "(model, hardware_tier) rows from different runs without "
        "matching keys.",
    ]

    if not has_gpu_energy:
        limitations.append(
            f"CPU-ONLY RUN: energy_consumption_llm_gpu = 0 for "
            f"{run['config_name']} (the {run['hardware_id']} device did "
            "not have a GPU). energy_kwh_llm_total == energy_kwh_llm_cpu "
            "for this config."
        )

    normalized_schema = sorted({
        k for r in normalized for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    summary_obj = {
        "dataset_id": DATASET_ID,
        "config_name": run["config_name"],
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": LICENSE,
        "license_redistribution_status": (
            "cc-by-sa-4.0 — attribution + share-alike required when "
            "redistributing committed normalised sample. "
            "Citation: Husom et al. 2024, 'The Price of Prompting: "
            "Profiling Energy Use in Large Language Models Inference', "
            "arxiv:2407.16893."
        ),
        "gated": GATED,
        "canonical_trace_type": "latency_benchmark_trace",
        "raw_committed": False,
        "raw_file_size_committed": False,
        "raw_columns": raw_columns,
        "raw_schema": raw_columns,
        "normalized_schema": normalized_schema,
        "available_signals": available_signals,
        "missing_signals": missing_signals,
        "derived_fields": [
            k for k, v in field_quality.items() if v == "derived"
        ],
        "proxy_fields": [
            k for k, v in field_quality.items() if v == "proxy"
        ],
        "synthetic_fields": [
            k for k, v in field_quality.items() if v == "synthetic"
        ],
        "real_fields": [k for k, v in field_quality.items() if v == "real"],
        "field_quality": field_quality,
        "limitations": limitations,
        "stratification_keys": [
            "hardware_tier", "model_name", "prompt_dataset",
        ],
        "sampling_method": (
            "full_bounded" if not raw_manifest["truncated"] else "head"
        ),
        "fixture_sample_rows": fixture_rows,
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "analysis_sample_rows": len(normalized),
        "analysis_sample_bytes": analysis_size,
        "analysis_sample_sha256": analysis_sha,
        "analysis_sample_path": str(analysis_path.relative_to(REPO_ROOT)),
        "committed_normalized_sample_rows": committed_rows_count,
        "committed_normalized_sample_bytes": committed_size,
        "committed_normalized_sample_path": str(
            committed_path.relative_to(REPO_ROOT)),
        "committed_normalized_sample_sha256": committed_sha,
        "committed_normalized_sample_reason_skipped": None,
        "committed_sample_rows": fixture_rows,
        "committed_sample_bytes": fixture_size,
        "sample_sha256": fixture_sha,
        "subgroup_counts": {
            f"hardware_tier={run['hardware_tier']}": len(normalized),
            f"model={run['model_name']}": len(normalized),
            f"prompt_dataset={run['prompt_dataset']}": len(normalized),
        },
        "statistical_sample_strength": strength,
        "unknown_columns": unknown_columns,
        "rejected_columns": rejected_columns,
        "accepted_columns": accepted_columns,
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
        "provenance": (
            f"{DATASET_ID}@{run['config_name']}#"
            f"{Path(run['csv_path']).name}#git={git_sha[:7]}"
        ),
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": [{
            "file": run["csv_path"], **raw_manifest,
        }],
        "elapsed_s": int(time.time() - t0),
        "hardware_tier": run["hardware_tier"],
        "hardware_id": run["hardware_id"],
        "model_name": run["model_name"],
        "model_family": run["model_family"],
        "model_size_b": run["model_size_b"],
        "prompt_dataset": run["prompt_dataset"],
        "engine": "ollama",
        "has_gpu_energy_signal": has_gpu_energy,
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
    }


def ingest(*, output_root: Path = HF_DIR) -> dict:
    git_sha = _git_sha()
    results = []
    for run in RUNS:
        logger.info("ingest config %s", run["config_name"])
        results.append(ingest_config(
            run=run, output_root=output_root, git_sha=git_sha,
        ))
    return {"results": results}


# ---------------------------------------------------------------------------
# Round-4 discovery records — negative-result audits
# ---------------------------------------------------------------------------


ROUND4_DISCOVERY_RECORDS = [
    {
        "dataset_id": "ohdoking/energy_consumption_by_model_and_gpu",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "license_unspecified_low_priority",
        "reason": (
            "Energy-by-model-and-GPU benchmark CSV. License "
            "unspecified on the dataset card; without redistribution "
            "clarity, committing a normalised sample is unsafe. "
            "Defer pending license clarification; "
            "ejhusom/llm-inference-energy-consumption (cc-by-sa-4.0) "
            "covers the same role with safe redistribution."
        ),
    },
    {
        "dataset_id": "adityaupasani/llm-inference-energy-consumption",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Near-identical fork of ejhusom/llm-inference-energy-"
            "consumption. License declared on neither card. "
            "Duplicate of the canonical SINTEF dataset already "
            "ingested in Round 4."
        ),
    },
    {
        "dataset_id": "Nayan10767/llm-inference-energy-consumption",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Near-identical fork of ejhusom/llm-inference-energy-"
            "consumption. License unspecified. Duplicate."
        ),
    },
    {
        "dataset_id": "vgyhj/llm-inference-energy-consumption",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Near-identical fork of ejhusom/llm-inference-energy-"
            "consumption. License unspecified. Duplicate."
        ),
    },
    {
        "dataset_id": "nishant-k/speculative-decoding-benchmark-results",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_low_information_density",
        "reason": (
            "Speculative-decoding benchmark output. Each file is a "
            "HumanEval task → completion → pass/fail trace. NO "
            "measured latency / throughput / energy / GPU / queue "
            "signal — code-completion correctness only. Out of scope "
            "for the Aurelius constraint-aware engine."
        ),
    },
    {
        "dataset_id": "inference-optimization/speculators_benchmarks_tool_call",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_low_information_density",
        "reason": (
            "BFCL v4 tool-call evaluation tasks (function-call test "
            "cases). Workload-shape only — NO measured infrastructure "
            "signal. The existing sharegpt_aiperf ingester covers "
            "request-shape priors with comparable density."
        ),
    },
    {
        "dataset_id": "kshitijthakkar/large-moe-inference-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": "apache-2.0",
        "gated": True,
        "kind": "gated_blocked",
        "reason": (
            "HF gated:manual (companion to "
            "kshitijthakkar/moe-inference-benchmark). 38 rows, "
            "MoE-specific (model_id, prompt, tokens_generated, "
            "time_seconds, tokens_per_second, total_params, "
            "active_params). HF_TOKEN is NOT authorised. Re-confirmed "
            "gated_blocked in Round 4."
        ),
    },
]


def write_round4_audit_summary(ingest_out: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_version": "round4_broadened_discovery_audit_summary_v1",
        "scope": (
            "Round 4 broadened HF discovery — bounded ingest of "
            "ejhusom/llm-inference-energy-consumption (Tier-4 "
            "cross-hardware-tier × workload × model-size energy + "
            "timing priors, 4 configs) + 7 negative-result discovery "
            "records covering ohdoking energy (license unspecified), "
            "3 ejhusom forks (duplicate), nishant-k speculative "
            "decoding (low information density — code-completion "
            "only), inference-optimization tool-call (BFCL tasks, "
            "no infra signal), kshitijthakkar large-moe (gated)."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "ingested": [
            {
                "dataset_id": r["summary"]["dataset_id"],
                "config_name": r["summary"]["config_name"],
                "canonical_trace_type": r["summary"][
                    "canonical_trace_type"],
                "license": r["summary"]["license"],
                "gated": r["summary"]["gated"],
                "analysis_sample_rows": r["summary"][
                    "analysis_sample_rows"],
                "fixture_sample_rows": r["summary"][
                    "fixture_sample_rows"],
                "committed_normalized_sample_rows": r["summary"][
                    "committed_normalized_sample_rows"],
                "committed_normalized_sample_bytes": r["summary"][
                    "committed_normalized_sample_bytes"],
                "available_signals": r["summary"]["available_signals"],
                "missing_signals": r["summary"]["missing_signals"],
                "statistical_sample_strength": r["summary"][
                    "statistical_sample_strength"],
                "promotion_state": r["decision"]["state"],
                "promotion_tags": r["decision"]["promotion_tags"],
                "promotion_reasons": r["decision"]["reasons"],
                "has_gpu_energy_signal": r["summary"][
                    "has_gpu_energy_signal"],
            }
            for r in ingest_out["results"]
        ],
        "discovery_only_records": ROUND4_DISCOVERY_RECORDS,
        "failed": [],
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dest


# ---------------------------------------------------------------------------
# Registry update
# ---------------------------------------------------------------------------


def update_canonical_registry(registry_path: Path,
                              entries: list[dict]) -> dict:
    if registry_path.exists():
        d = json.loads(registry_path.read_text())
    else:
        d = {
            "doc_version": "hf_corpus_canonical_registry_v1",
            "entries": [],
        }
    existing = d.get("entries", [])
    # Drop any existing entries for this (dataset_id, config_name); add new ones.
    new_keys = {(e["dataset_id"], e.get("config_name")) for e in entries}
    filtered = [
        e for e in existing
        if (e.get("dataset_id"), e.get("config_name")) not in new_keys
    ]
    filtered.extend(entries)
    d["entries"] = filtered
    d["entry_count"] = len(filtered)
    d["written_at_s"] = time.time()
    d["stage"] = d.get("stage") or "research_data_engine_pr"
    d["production_claim"] = False
    d["modifies_controllers_or_defaults"] = False
    d["modifies_robust_energy_engine"] = False
    d["uses_oracle_as_headline"] = False
    d["trust_hierarchy_note"] = d.get(
        "trust_hierarchy_note",
        "Pilot telemetry is the only Tier 1 calibration source; "
        "all HF datasets are Tier 2-6 at best.")
    registry_path.write_text(json.dumps(d, indent=2, sort_keys=True))
    return d


def update_candidate_registry(candidates_path: Path,
                              ingested_entries: list[dict]) -> dict:
    if not candidates_path.exists():
        return {}
    d = json.loads(candidates_path.read_text())
    cands = d.get("candidates", [])

    target_id = DATASET_ID
    found_idx = None
    for i, c in enumerate(cands):
        if c.get("dataset_id") == target_id:
            found_idx = i
            break

    ingested_configs = [e["config_name"] for e in ingested_entries]
    avail = list({
        s
        for e in ingested_entries
        for s in e.get("available_signals", [])
    })
    miss = list({
        s
        for e in ingested_entries
        for s in e.get("missing_signals", [])
    })

    new_entry = {
        "dataset_id": target_id,
        "dataset_url": f"https://huggingface.co/datasets/{target_id}",
        "gated_status": "public",
        "license": LICENSE,
        "estimated_size": ["10K<n<100K"],
        "available_splits": [],
        "schema_available": True,
        "matched_keywords": [
            "round4::llm_energy_consumption",
            "energy::cpu_gpu_total_kwh",
            "engine::ollama",
            "hardware_tier::laptop_workstation_server",
            "cross_hardware_tier_priors",
        ],
        "candidate_trace_type": "latency_benchmark_trace",
        "trust_level": "tier_4_latency_benchmark_traces",
        "available_signals": avail,
        "missing_signals": miss,
        "aurelius_use_case": (
            "Cross-hardware-tier (laptop / workstation / server) "
            "energy + timing priors for Aurelius placement / routing / "
            "deferral decisions. First public dataset in the federated "
            "corpus that covers consumer/laptop tier."
        ),
        "not_recommended_uses": [
            "Real-arrival scheduling (concurrency = 1; no queue)",
            "GPU-direct vLLM/SGLang latency calibration (Ollama engine)",
            "Cross-tier extrapolation outside (model, hardware) cells",
            "Dynamic frontier calibration (Tier 4 benchmark only)",
        ],
        "ingestion_feasibility_score": 5,
        "frontier_value_score": 4,
        "schema_quality_score": 5,
        "production_similarity_score": 2,
        "overall_priority_score": 4.5,
        "recommended_action": "ingest_now_bounded",
        "feature_names": [
            "model_name", "type", "created_at", "start_time", "end_time",
            "clock_duration", "total_duration", "load_duration",
            "prompt_duration", "response_duration", "prompt_token_length",
            "response_token_length", "energy_consumption_monitoring",
            "energy_consumption_llm_cpu", "energy_consumption_llm_gpu",
            "energy_consumption_llm_total", "energy_consumption_llm",
            "index",
        ],
        "classification_evidence": {
            "latency_benchmark_trace": [
                "total_duration", "prompt_duration", "response_duration",
                "energy_consumption_llm_total", "prompt_token_length",
                "response_token_length",
            ],
        },
        "downloads": 70,
        "likes": 1,
        "last_modified": "2024-08-15T18:36:58.000Z",
        "discovery_timestamp_s": time.time(),
        "configs_ingested": ingested_configs,
        "configs": [],
    }
    if found_idx is None:
        cands.append(new_entry)
    else:
        cands[found_idx] = {**cands[found_idx], **new_entry}

    # Add Round-4 discovery-only / negative-result records.
    existing_ids = {c.get("dataset_id") for c in cands}
    for rec in ROUND4_DISCOVERY_RECORDS:
        if rec["dataset_id"] in existing_ids:
            continue
        cands.append({
            "dataset_id": rec["dataset_id"],
            "dataset_url": (
                f"https://huggingface.co/datasets/{rec['dataset_id']}"
            ),
            "gated_status": "gated_manual" if rec.get("gated") else "public",
            "license": rec.get("license_observed"),
            "estimated_size": [],
            "available_splits": [],
            "schema_available": True,
            "matched_keywords": [f"round4::{rec['kind']}"],
            "candidate_trace_type": rec["candidate_trace_type"],
            "trust_level": "tier_4_latency_benchmark_traces",
            "available_signals": [],
            "missing_signals": [],
            "aurelius_use_case": "Round-4 discovery audit — see reason.",
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
            "round4_audit_reason": rec["reason"],
        })

    d["candidates"] = cands
    d["candidate_count"] = len(cands)
    candidates_path.write_text(json.dumps(d, indent=2, sort_keys=True))
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Round-4 broadened HF ingest — "
            "ejhusom/llm-inference-energy-consumption"
        )
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--skip-registry-update", action="store_true",
        help="Run the ingest only; do not modify the corpus registries.",
    )
    p.add_argument(
        "--output-root", default=str(HF_DIR),
        help="Override HF_DIR — used by tests.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )

    out = ingest(output_root=Path(args.output_root))

    if not args.skip_registry_update:
        registry_path = DISC_DIR / "canonical_corpus_registry.json"
        candidates_path = DISC_DIR / "hf_dataset_candidates.json"
        audit_path = (
            DISC_DIR / "round4_broadened_discovery_audit_summary.json"
        )

        entries = [r["registry_entry"] for r in out["results"]]
        update_canonical_registry(registry_path, entries)
        update_candidate_registry(candidates_path, entries)
        write_round4_audit_summary(out, audit_path)

    for r in out["results"]:
        decision = r["decision"]
        print(json.dumps({
            "dataset_id": r["summary"]["dataset_id"],
            "config_name": r["summary"]["config_name"],
            "promotion_state": decision["state"],
            "promotion_tags": decision["promotion_tags"],
            "analysis_sample_rows": r["summary"]["analysis_sample_rows"],
            "committed_normalized_sample_rows": r["summary"][
                "committed_normalized_sample_rows"],
            "committed_normalized_sample_bytes": r["summary"][
                "committed_normalized_sample_bytes"],
            "fixture_sample_rows": r["summary"]["fixture_sample_rows"],
            "license": r["summary"]["license"],
            "gated": r["summary"]["gated"],
            "trace_type": r["summary"]["canonical_trace_type"],
            "statistical_sample_strength": r["summary"][
                "statistical_sample_strength"],
            "has_gpu_energy_signal": r["summary"][
                "has_gpu_energy_signal"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
