#!/usr/bin/env python3
"""Round-5 broadened HF discovery — bounded ingest of ``metrum-ai/llm-perfdata``.

This script extends the federated benchmark corpus with a Tier-4
``latency_benchmark_trace`` dataset that fills a gap the existing
HF-ingested datasets do not cover: **measured TTFT / TPOT / throughput
across NVIDIA H100, H200, B200, AMD MI300X / MI355X, Intel Gaudi 3, and
the SGLang / vLLM-ROCm serving engines** (none of which appear in
AgentPerfBench / Odyn / Memoriant / Intellistream / optimum-benchmark /
ssong1-llmperf-bedrock / ejhusom).

Why it matters for Aurelius:

* The Aurelius placement / routing / deferral engine previously had NO
  public prior for H100, H200, B200, AMD MI300X / MI355X, or Intel
  Gaudi 3 — the existing latency benchmarks are A100 / A10 / T4 / DGX
  Spark / 32vCPU-C7i / Bedrock only.
* This is also the first SGLang and vLLM-ROCm coverage in the corpus.
* metrum-ai is a **multi-source curated ledger**: each row's
  ``Source_URL`` points to the upstream public benchmark report it was
  copied from. Trust is "curator copied numbers from public benchmark
  reports" — strictly weaker than a single-campaign measurement.
* Statistical_sample_strength is therefore **weak** (80 rows × 24
  models × 9 GPUs × 5 engines → ~1-2 rows per (model, gpu, engine)
  cell), so promotion gates restrict this to
  ``promoted_for_training_priors`` only.

This is NOT pilot telemetry; it is a public multi-source curated
ledger and stays strictly Tier-4 (latency_benchmark_trace).

Outputs (single config: ``multi_source_curated_v1``):

    data/external/hf/metrum-ai__llm-perfdata/raw/<file>             # gitignored
    data/external/hf/metrum-ai__llm-perfdata/multi_source_curated_v1/processed/
        schema_profile.json               (committed)
        schema_mapping.json               (committed)
        summary.json                      (committed)
        statistical_rollups.json          (committed)
        committed_normalized_sample.jsonl (committed, all 80 rows, ~25 KiB)
        analysis_sample.jsonl             (gitignored)
    tests/fixtures/hf/metrum-ai__llm-perfdata__multi_source_curated_v1_sample.jsonl
                                          (committed, ≤ 16 KiB, 5 rows)

It also writes a discovery audit summary at
``data/external/hf_discovery/round5_broadened_discovery_audit_summary.json``
that records the negative-result candidates from this round.

NO production claims. NO scheduler / controller / robust-energy-engine
changes.
"""

from __future__ import annotations

import argparse
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

FIXTURE_MAX_BYTES = 16 * 1024
COMMITTED_NORMALIZED_MAX_BYTES = 128 * 1024  # 128 KiB — full 80 rows fit
PER_DATASET_TIMEOUT_S = 30 * 60

logger = logging.getLogger("ingest_hf_metrum_llmperfdata")

DATASET_ID = "metrum-ai/llm-perfdata"
SAFE_NAME = "metrum-ai__llm-perfdata"
CONFIG = "multi_source_curated_v1"
LICENSE = "mit"
GATED = False
RAW_FILE = "data/train-00000-of-00001.parquet"


# Round-5 discovery-only negative-result records (datasets searched and
# rejected during the same HF discovery pass that surfaced metrum-ai).
ROUND5_DISCOVERY_RECORDS: list[dict] = [
    {
        "dataset_id": "sairamn/gcp-cloud-billing-cost",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "kind": "reject_synthetic_economics",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "GCP cloud-billing CSV (18.9 MB, 100K-1M rows). Schema looks "
            "economic-relevant on paper (Resource ID, Service Name, "
            "Usage Quantity, Region/Zone, CPU/Memory Utilization, "
            "Cost per Quantity ($), Total Cost (INR)) but inspection of "
            "the first 20 rows shows clearly SYNTHETIC values: resource "
            "IDs are uniform 'resource_NNN', cost columns are rounded "
            "round numbers, network-data columns are absurd "
            "(4.4e+11 bytes per ~4-day window), and INR-conversion "
            "rounds match the synthetic-pricing pattern. The dataset "
            "card carries no provenance or upstream-source attribution. "
            "Synthetic chargeback/cost data — Tier-6; rejected to "
            "enforce the anti-dataset-spam rule for economic signals "
            "(Round-5 economic-priority gate)."
        ),
    },
    {
        "dataset_id": "ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "kind": "reject_synthetic_ai_safety_eval",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "ClarusC64 'coherence-risk' series — text-classification "
            "eval that detects when claimed carbon-aware scheduling "
            "decisions diverge from claimed emissions outcomes. The "
            "schema is observable_a (workload_shiftability) vs "
            "observable_b (carbon_intensity_forecasts) → "
            "ground_truth_label. NO measured infrastructure signal — "
            "this is an AI-safety / claim-coherence evaluation, not "
            "telemetry. n<1K. Rejected as out-of-scope synthetic eval."
        ),
    },
    {
        "dataset_id": "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "kind": "reject_synthetic_ai_safety_eval",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "Same ClarusC64 'coherence-risk' family — datacenter "
            "power-load claim coherence vs. true outcome. AI-safety "
            "eval format, n<1K, no measured telemetry. Rejected as "
            "out-of-scope synthetic eval (duplicate of the "
            "carbon-aware-scheduling rejection rationale)."
        ),
    },
    {
        "dataset_id": "Phipper/pe-energy-infrastructure-training-data",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "kind": "reject_out_of_scope",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "Private-equity / energy-infrastructure finance training "
            "data for fine-tuning a PE-focused AI assistant (Thurin). "
            "297 DPO pairs + 804 SFT conversations + 2,308 "
            "Opus-distilled reasoning traces across PE deal analysis / "
            "financial modeling / regulatory / strategy categories. "
            "Despite the 'energy-infrastructure' tag, this is finance "
            "domain text — NO measured infrastructure or telemetry "
            "signal. Out of scope for Aurelius."
        ),
    },
    {
        "dataset_id": "uohna/llm_inference_energy_combined.parquet",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "kind": "reject_empty_dataset",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Empty dataset — only `.gitattributes` is committed in the "
            "repository tree; no actual parquet files. Despite the "
            "promising name 'llm_inference_energy_combined.parquet', "
            "there is no data to ingest. 2 downloads, 0 likes."
        ),
    },
    {
        "dataset_id": "Lightcap/agent-runtime-telemetry-small",
        "candidate_trace_type": "request_shape_trace",
        "kind": "defer_high_value_different_trace_class",
        "license_observed": "cc-by-4.0",
        "gated": False,
        "reason": (
            "REAL MCP-style agent-runtime tool-call telemetry exported "
            "from local SQLite stores (Faruk Alpay). 8 parquet configs: "
            "operations (2,262 × 33, real duration_ms + status + "
            "error_type + tool_name + UTC timestamps), operation_events "
            "(9,903 lifecycle events), audit_records (14,053 audit "
            "rows), tool_summary (32 tools with avg/median/p95 "
            "duration_ms). cc-by-4.0 (redistributable with "
            "attribution). HIGH information density for tool-call / "
            "agent-orchestration RELIABILITY priors — but the canonical "
            "Aurelius trace types are LLM-serving-focused; tool-call "
            "telemetry has NO model_id / NO input_tokens / NO GPU / NO "
            "queue / NO concurrency / NO cache fields. Deferred to a "
            "follow-on run that adds a (new) `tool_runtime_trace` "
            "canonical type OR maps tool-call durations into the "
            "existing request_shape_trace as a routing-quality / "
            "failure-rate prior. Not a duplicate of Exgentic — Exgentic "
            "captures LLM-CALL spans with model + tokens; Lightcap "
            "captures TOOL-CALL operations with no LLM-specific "
            "fields. See HF_DATASET_REGISTRY §10 for the follow-up "
            "action."
        ),
    },
    {
        "dataset_id": "metrum-ai/llm-perf-dashboard",
        "candidate_trace_type": "latency_benchmark_trace",
        "kind": "defer_pending_inspection",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Companion dashboard dataset to metrum-ai/llm-perfdata "
            "(same author). Schema not yet inspected because the "
            "dashboard format is markdown / static-site oriented "
            "rather than tabular. Deferred to a follow-on run if the "
            "dashboard exports additional measurement rows beyond what "
            "llm-perfdata already covers."
        ),
    },
    {
        "dataset_id": "ssakethch/h200-quantization-benchmarks",
        "candidate_trace_type": "latency_benchmark_trace",
        "kind": "defer_pending_full_schema_probe",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Benchmark results for 40 quantized + non-quantized "
            "instruction-tuned LLMs on NVIDIA H200 MIG (Multi-Instance "
            "GPU). Potentially high-value (first H200-MIG coverage in "
            "the corpus + first 40-model breadth quantization "
            "comparison), but the dataset card declares no SPDX "
            "license. Defer until license clarified — without "
            "redistribution clarity, committing a normalised sample "
            "would violate the corpus license-and-gating gate. "
            "Re-audit if author adds a license."
        ),
    },
    {
        "dataset_id": "crozai/vllm-benchmark-coding",
        "candidate_trace_type": "request_shape_trace",
        "kind": "duplicate_existing",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Coding-workload prompt fixtures designed for "
            "vllm/benchmark_serving.py — workload-shape only (prompt "
            "strings + token counts). NO measured TTFT / TPOT / "
            "throughput. The existing `aurelius/traces/sharegpt_aiperf.py` "
            "and `hlarcher/inference-benchmarker` (already rejected) "
            "cover this role at higher density."
        ),
    },
    {
        "dataset_id": "intellistream/sage-agent-benchmark",
        "candidate_trace_type": "request_shape_trace",
        "kind": "reject_eval_only_no_telemetry",
        "license_observed": None,
        "gated": False,
        "reason": (
            "AgentBench-style evaluation for tool-selection / "
            "task-planning / response-generation accuracy. NO measured "
            "latency / GPU / queue / cache signal — agent-capability "
            "scoring only. Out of scope for the constraint-aware "
            "engine, which needs measured-routing telemetry not router "
            "training labels (same rationale as the rejected "
            "abdallah1008 semantic-router benchmark)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or None


def _hf_url(dataset_id: str, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/{path}"
    )


def _download(url: str, dest: Path, *, max_bytes: int) -> dict:
    """Streaming download with a hard byte-cap. Returns a manifest dict."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-discovery/round5"}
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    n = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, "wb") as fh:
                while True:
                    if n >= max_bytes:
                        err = f"max_bytes={max_bytes} exceeded"
                        break
                    chunk = resp.read(min(64 * 1024, max_bytes - n))
                    if not chunk:
                        break
                    fh.write(chunk)
                    n += len(chunk)
    except urllib.error.HTTPError as e:
        err = f"HTTPError {e.code}: {e.reason}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "url": url,
        "dest": str(dest.relative_to(REPO_ROOT)) if dest.exists() else None,
        "downloaded_bytes": n,
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
    except Exception:
        return ""


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    vals_sorted = sorted(vals)

    def _q(p: float) -> float:
        if len(vals_sorted) == 1:
            return float(vals_sorted[0])
        k = max(0.0, min(p * (len(vals_sorted) - 1), len(vals_sorted) - 1))
        lo, hi = int(k), min(int(k) + 1, len(vals_sorted) - 1)
        return float(vals_sorted[lo] + (vals_sorted[hi] - vals_sorted[lo]) * (k - lo))

    return {
        "count": len(vals_sorted),
        "min": float(min(vals_sorted)),
        "max": float(max(vals_sorted)),
        "mean": float(statistics.fmean(vals_sorted)),
        "p25": _q(0.25), "p50": _q(0.50), "p75": _q(0.75),
        "p90": _q(0.90), "p95": _q(0.95), "p99": _q(0.99),
    }


def _norm_str(v) -> Optional[str]:
    """Normalise pandas NA / NaN string / empty string → None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _norm_num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _normalize_row(raw: dict, run_index: int, git_sha: str) -> dict:
    """Project one raw metrum-ai parquet row to a canonical record."""
    return {
        "source_dataset_id": DATASET_ID,
        "trace_type": "latency_benchmark_trace",
        "provenance": (
            f"{DATASET_ID}@{CONFIG}#row={run_index}"
            f"#src={_norm_str(raw.get('Source_URL')) or 'unspecified'}"
            f"#git={git_sha[:7]}"
        ),
        "row_index": run_index,
        "model_id": _norm_str(raw.get("Model")),
        "model_size": _norm_str(raw.get("Size")),
        "precision": _norm_str(raw.get("Precision")),
        "quantization": _norm_str(raw.get("Quantization")),
        "gpu_type": _norm_str(raw.get("GPU_Type")),
        "num_gpus": _norm_num(raw.get("Num_GPUs")),
        "engine": _norm_str(raw.get("Serving_Engine")),
        "concurrency": _norm_num(raw.get("Concurrency")),
        "tokens_per_sec": _norm_num(raw.get("Tokens_per_sec")),
        "ttft_ms": _norm_num(raw.get("TTFT_ms")),
        "tpot_ms": _norm_num(raw.get("TPOT_ms")),
        "prompt_tokens": _norm_num(raw.get("Prompt_Tokens")),
        "output_tokens": _norm_num(raw.get("Output_Tokens")),
        "context_window": _norm_num(raw.get("Context_Window")),
        "source_url": _norm_str(raw.get("Source_URL")),
        "source_notes": _norm_str(raw.get("Source_Notes")),
    }


def ingest(*, output_root: Path = HF_DIR, force: bool = False) -> dict:
    t0 = time.time()
    git_sha = _git_sha()

    base = output_root / SAFE_NAME
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = base / CONFIG / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download raw parquet (gitignored).
    raw_local = raw_dir / Path(RAW_FILE).name
    manifest = _download(_hf_url(DATASET_ID, RAW_FILE), raw_local,
                         max_bytes=2 * 1024 * 1024)  # 2 MiB cap; file is ~11 KiB
    if manifest["error"]:
        raise RuntimeError(
            f"metrum-ai/llm-perfdata: failed to download {RAW_FILE}: "
            f"{manifest['error']}"
        )
    raw_manifests = [{"file": RAW_FILE, **manifest}]

    # 2. Read parquet → list of dicts.
    import pandas as pd  # noqa: PLC0415 — runtime dep
    df = pd.read_parquet(raw_local)
    raw_columns = list(df.columns)
    n_rows = len(df)
    logger.info("metrum-ai parquet: %d rows × %d cols", n_rows, len(raw_columns))

    # 3. Normalise rows.
    normalized_rows = [
        _normalize_row(row.to_dict(), i, git_sha)
        for i, row in df.iterrows()
    ]

    # 4. Write analysis_sample.jsonl (gitignored — all 80 rows).
    analysis_path = proc_dir / "analysis_sample.jsonl"
    analysis_buf = io.BytesIO()
    for row in normalized_rows:
        analysis_buf.write((json.dumps(row, sort_keys=True) + "\n").encode())
    analysis_path.write_bytes(analysis_buf.getvalue())
    analysis_sha = _sha256(analysis_path)
    analysis_size = analysis_path.stat().st_size

    # 5. Fixture (committed) — 5 rows, ≤ 16 KiB.
    fixture_path = FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_buf = io.BytesIO()
    fixture_rows = 0
    # Pick 5 rows that cover the most diverse strata.
    # We want H100, H200, B200, AMD, and one other (Gaudi) ideally.
    seen_gpu_engine: set = set()
    sample_indexes: list[int] = []
    for i, row in enumerate(normalized_rows):
        if fixture_rows >= 5:
            break
        key = (row.get("gpu_type"), row.get("engine"))
        if key in seen_gpu_engine:
            continue
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if fixture_buf.tell() + len(line) > FIXTURE_MAX_BYTES:
            continue
        fixture_buf.write(line)
        seen_gpu_engine.add(key)
        sample_indexes.append(i)
        fixture_rows += 1
    fixture_path.write_bytes(fixture_buf.getvalue())
    fixture_size = fixture_path.stat().st_size
    fixture_sha = _sha256(fixture_path)

    # 6. Committed normalised sample (license=MIT → redistributable; all 80
    # rows fit easily under 128 KiB).
    committed_path = proc_dir / "committed_normalized_sample.jsonl"
    committed_buf = io.BytesIO()
    committed_rows_count = 0
    for row in normalized_rows:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if committed_buf.tell() + len(line) > COMMITTED_NORMALIZED_MAX_BYTES:
            break
        committed_buf.write(line)
        committed_rows_count += 1
    committed_path.write_bytes(committed_buf.getvalue())
    committed_size = committed_path.stat().st_size
    committed_sha = _sha256(committed_path)

    # 7. Schema profile.
    dtypes: dict[str, list[str]] = {}
    example_values: dict[str, list[str]] = {}
    missing_rates: dict[str, float] = {}
    for col in raw_columns:
        seen = set()
        examples = []
        n_missing = 0
        for _, row in df.iterrows():
            v = row.get(col)
            if v is None or (isinstance(v, float) and v != v):  # NaN
                n_missing += 1
                continue
            seen.add(type(v).__name__)
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
        "raw_columns": sorted(raw_columns),
        "nested_keys": [],
        "dtypes": dtypes,
        "example_values": example_values,
        "missing_rates": missing_rates,
        "unknown_columns": [],
        "rejected_columns": [],
        "accepted_columns": sorted(raw_columns),
        "list_length_summaries": {},
        "file_size_bytes": raw_local.stat().st_size,
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # 8. Schema mapping.
    columns_map = _build_columns_map()
    schema_mapping = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "accepted_columns": sorted(raw_columns),
        "rejected_columns": [],
        "unknown_columns": [],
        "columns": columns_map,
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # 9. Statistical rollups — overall + by stratum.
    overall: dict[str, dict] = {}
    for k in ["ttft_ms", "tpot_ms", "tokens_per_sec", "concurrency",
              "num_gpus", "prompt_tokens", "output_tokens"]:
        vals = [r[k] for r in normalized_rows if r.get(k) is not None]
        if vals:
            overall[k] = _percentiles(vals)

    by_strata: dict[str, dict] = {}
    subgroup_counts: dict[str, int] = {}
    # Stratify by (gpu_type, engine, precision)
    strata_groups: dict[tuple, list[dict]] = {}
    for row in normalized_rows:
        key = (row.get("gpu_type"), row.get("engine"), row.get("precision"))
        strata_groups.setdefault(key, []).append(row)
    for key, grp in strata_groups.items():
        label = (
            f"gpu={key[0] or 'unknown'}|engine={key[1] or 'unknown'}"
            f"|precision={key[2] or 'unknown'}"
        )
        subgroup_counts[label] = len(grp)
        if len(grp) >= 3:
            grp_stats: dict[str, dict] = {}
            for k in ["ttft_ms", "tpot_ms", "tokens_per_sec"]:
                vals = [r[k] for r in grp if r.get(k) is not None]
                if vals:
                    grp_stats[k] = _percentiles(vals)
            if grp_stats:
                by_strata[label] = grp_stats

    rollups = {
        "overall": overall,
        "by_strata": by_strata,
        "subgroup_counts": subgroup_counts,
        "stratification_keys": ["gpu_type", "engine", "precision"],
        "insufficient_sample_groups": [
            label for label, n in subgroup_counts.items() if n < 5
        ],
    }
    (proc_dir / "statistical_rollups.json").write_text(
        json.dumps(rollups, indent=2, sort_keys=True)
    )

    # 10. Sample strength — 80 rows × 24 models × 9 GPUs × 5 engines means
    # the densest cell is (NVIDIA A100, vLLM, FP16) with 8 rows. This is
    # WEAK by the corpus convention (PROMOTION_TAG_MIN_SAMPLE_STRENGTH:
    # `weak` qualifies only training_priors).
    strength = "weak"

    normalized_schema = sorted({
        k for r in normalized_rows for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    available_signals = [
        "ttft", "tpot", "throughput", "concurrency",
        "input_tokens", "output_tokens",
        "gpu_type", "num_gpus", "model_id", "engine",
        "precision", "quantization", "context_window",
    ]
    missing_signals = [
        "itl", "e2e_latency",
        "queue_state", "queue_wait", "queue_depth",
        "memory_pressure", "gpu_utilization",
        "batch_size",
        "timeout_label", "sla_label", "failure_label",
        "autoscaling", "replica_count",
        "kv_cache_size", "cache_hit", "kernel_duration",
        "carbon_intensity", "energy_per_request",
        "cost_per_token", "cost_per_request",
    ]

    field_quality = _field_quality()

    limitations = [
        "metrum-ai/llm-perfdata: MIT-licensed multi-source curated "
        "ledger maintained by Metrum AI. 80 rows of "
        "(Model, Size, Precision, GPU_Type, Num_GPUs, Serving_Engine, "
        "Concurrency, Tokens_per_sec, TTFT_ms, TPOT_ms, Prompt_Tokens, "
        "Output_Tokens, Context_Window, Quantization, Source_URL, "
        "Source_Notes). Trust Tier 4 (latency_benchmark_trace).",
        "MULTI-SOURCE CURATED — every row's Source_URL points to a "
        "DIFFERENT upstream public benchmark report; this is NOT a "
        "single-campaign measurement. Trust model: 'the curator copied "
        "numbers from public benchmark reports'. Aurelius MUST honour "
        "Source_URL as the primary attribution and treat absolute "
        "numbers as a CROSS-SOURCE-AVERAGED prior, not a calibrated "
        "single measurement.",
        "SPARSE FIELD COVERAGE — TTFT_ms 17/80 non-null, TPOT_ms "
        "10/80, Tokens_per_sec 38/80, Prompt_Tokens 4/80, "
        "Output_Tokens 4/80, Context_Window 8/80. Many rows carry "
        "only Tokens_per_sec or only a Source_Notes blurb. The "
        "field_quality map labels every column 'real' (because the "
        "values originate from real upstream measurements), but "
        "downstream consumers MUST check presence-rate before "
        "computing per-stratum statistics.",
        "STATISTICAL_SAMPLE_STRENGTH = weak — densest cell "
        "(NVIDIA A100, vLLM, FP16) has 8 rows. Promotion is gated to "
        "promoted_for_training_priors ONLY; promoted_for_performance_priors "
        "and promoted_for_constraint_aware_evaluation require moderate "
        "strength which this dataset does not meet.",
        "GPU coverage is BROAD but cells are small: NVIDIA H100 (28 rows), "
        "Intel Gaudi 3 (11), NVIDIA H200 (10), NVIDIA A100 (8), "
        "NVIDIA B200 (5), AMD MI300X (5), NVIDIA L40S (5), "
        "AMD MI355X (2), NVIDIA RTX 4090 (2). The corpus had NO public "
        "H100/H200/B200/Gaudi-3/MI300X/MI355X/L40S/RTX-4090 priors "
        "before this ingest, so even weak coverage is a net gain.",
        "Engine coverage: vLLM (43 rows), SGLang (25), vLLM-ROCm (6), "
        "vLLM-v0 (3), vLLM-v1 (3). First SGLang and vLLM-ROCm entries "
        "in the corpus.",
        "NO measured queue / concurrency contention / batching / "
        "memory-pressure / GPU-utilization / energy / carbon / cost "
        "fields. Aurelius queue-risk, batch-frontier, energy-cost, and "
        "carbon-cost modules MUST NOT consume this dataset.",
        "TTFT_ms / TPOT_ms semantics are NOT formally defined in the "
        "dataset card — they are whatever each upstream benchmark "
        "report called them. ITL is NOT recorded; TPOT may collapse "
        "ITL × output_tokens depending on the upstream source.",
        "Source_Notes is free-text describing each row's upstream "
        "context (e.g. 'MLPerf-style server aggregate; engine vLLM. "
        "[1] indicates 8xB200 hits ~160k tok/s.'). Notes are not "
        "structured — do NOT parse them programmatically.",
        "NO timestamps — rows do not carry a measurement-date column. "
        "Treat the whole dataset as a static cross-source snapshot "
        "(last_modified = 2026-04-01 upstream).",
        "BENCHMARK-ONLY — Tier 4. Not pilot telemetry. Pilot telemetry "
        "remains the only Tier 1 calibration source.",
        "Dataset card explicitly disclaims: 'numbers depend on external "
        "documentation, there may be gaps, inconsistencies, or "
        "occasional inaccuracies' and 'expect the dataset to drift out "
        "of date as serving stacks and software releases evolve'. "
        "Aurelius MUST refresh / re-validate when re-using.",
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
        "raw_columns": sorted(raw_columns),
        "raw_schema": sorted(raw_columns),
        "normalized_schema": normalized_schema,
        "available_signals": available_signals,
        "missing_signals": missing_signals,
        "derived_fields": [k for k, v in field_quality.items() if v == "derived"],
        "proxy_fields":   [k for k, v in field_quality.items() if v == "proxy"],
        "synthetic_fields": [k for k, v in field_quality.items() if v == "synthetic"],
        "real_fields":    [k for k, v in field_quality.items() if v == "real"],
        "field_quality": field_quality,
        "limitations": limitations,
        "stratification_keys": ["gpu_type", "engine", "precision"],
        "sampling_method": "full_bounded",
        "fixture_sample_rows": fixture_rows,
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "fixture_sample_strata_keys": ["gpu_type", "engine"],
        "fixture_row_indexes": sample_indexes,
        "analysis_sample_rows": n_rows,
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
        "provenance": (
            f"{DATASET_ID}@{CONFIG}#{RAW_FILE}#git={git_sha[:7]}"
        ),
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": raw_manifests,
        "elapsed_s": int(time.time() - t0),
        "unknown_columns": [],
    }
    (proc_dir / "summary.json").write_text(
        json.dumps(summary_obj, indent=2, sort_keys=True)
    )

    decision = promotion.evaluate_promotion(summary_obj)
    entry = promotion.build_registry_entry(summary_obj, decision)
    return {"summary": summary_obj, "decision": decision,
            "registry_entry": entry, "raw_manifests": raw_manifests}


def _field_quality() -> dict:
    return {
        # Curated-but-real per-row fields from metrum-ai parquet.
        "model_id": "real",
        "model_size": "real",
        "precision": "real",
        "quantization": "real",
        "gpu_type": "real",
        "num_gpus": "real",
        "engine": "real",
        "concurrency": "real",
        "tokens_per_sec": "real",
        "ttft_ms": "real",
        "tpot_ms": "real",
        "prompt_tokens": "real",
        "output_tokens": "real",
        "context_window": "real",
        "source_url": "real",
        "source_notes": "real",
        # Missing for Aurelius decision-making
        "itl_ms": "missing",
        "e2e_latency_ms": "missing",
        "queue_wait": "missing",
        "queue_depth": "missing",
        "gpu_utilization": "missing",
        "memory_pressure": "missing",
        "batch_size": "missing",
        "engine_version": "missing",
        "kv_cache_size": "missing",
        "cache_hit": "missing",
        "kernel_duration": "missing",
        "energy_per_request": "missing",
        "carbon_intensity": "missing",
        "cost_per_token": "missing",
        "cost_per_request": "missing",
        "timeout_label": "missing",
        "sla_label": "missing",
        "failure_label": "missing",
        "autoscaling": "missing",
        "replica_count": "missing",
    }


def _build_columns_map() -> list[dict]:
    """Map metrum-ai raw parquet columns → Aurelius signal categories."""
    return [
        {
            "raw_column_name": "Model",
            "normalized_field": "model_id",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": "Published model identifier as named by the upstream source.",
        },
        {
            "raw_column_name": "Size",
            "normalized_field": "model_size",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": "params shorthand (e.g. '7B', '70B')",
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": "Parameter scale shorthand.",
        },
        {
            "raw_column_name": "Precision",
            "normalized_field": "precision",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": "Numeric precision used during serving (FP16, BF16, FP8, INT4).",
        },
        {
            "raw_column_name": "GPU_Type",
            "normalized_field": "gpu_type",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "gpu_resource",
            "usable_for": ["latency_prior", "throughput_prior",
                           "constraint_aware_evaluation"],
            "notes": (
                "Accelerator family (NVIDIA A100/H100/H200/B200/L40S/RTX-4090, "
                "AMD MI300X/MI355X, Intel Gaudi 3). First H100/H200/B200/"
                "Gaudi-3/MI300X/MI355X/L40S/RTX-4090 coverage in the corpus."
            ),
        },
        {
            "raw_column_name": "Num_GPUs",
            "normalized_field": "num_gpus",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "count",
            "aurelius_signal_category": "gpu_resource",
            "usable_for": ["constraint_aware_evaluation"],
            "notes": "Integer count of GPUs participating in the benchmark.",
        },
        {
            "raw_column_name": "Serving_Engine",
            "normalized_field": "engine",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": (
                "vLLM / SGLang / vLLM-ROCm / vLLM-v0 / vLLM-v1. First SGLang "
                "and vLLM-ROCm coverage in the federated corpus."
            ),
        },
        {
            "raw_column_name": "Concurrency",
            "normalized_field": "concurrency",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "concurrent requests",
            "aurelius_signal_category": "throughput",
            "usable_for": ["throughput_prior"],
            "notes": (
                "Concurrent request count for the benchmark. NOT queue-state "
                "telemetry — Aurelius queue-risk module MUST NOT consume."
            ),
        },
        {
            "raw_column_name": "Tokens_per_sec",
            "normalized_field": "tokens_per_sec",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "tokens / second",
            "aurelius_signal_category": "throughput",
            "usable_for": ["throughput_prior"],
            "notes": "Aggregate output throughput as reported by the upstream source.",
        },
        {
            "raw_column_name": "TTFT_ms",
            "normalized_field": "ttft_ms",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "milliseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior"],
            "notes": (
                "Time to first token (whatever the upstream source meant by "
                "TTFT). Present in 17/80 rows. Cross-source semantics "
                "differ — do NOT compute single-source-quality percentiles."
            ),
        },
        {
            "raw_column_name": "TPOT_ms",
            "normalized_field": "tpot_ms",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "milliseconds",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior"],
            "notes": (
                "Tail-period-of-token generation as reported by the upstream "
                "source. May collapse ITL × output_tokens depending on the "
                "source convention. Present in 10/80 rows."
            ),
        },
        {
            "raw_column_name": "Prompt_Tokens",
            "normalized_field": "prompt_tokens",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "latency_prior"],
            "notes": "Input prompt token count. Present in 4/80 rows.",
        },
        {
            "raw_column_name": "Output_Tokens",
            "normalized_field": "output_tokens",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "throughput_prior"],
            "notes": "Generated output token count. Present in 4/80 rows.",
        },
        {
            "raw_column_name": "Context_Window",
            "normalized_field": "context_window",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "tokens",
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["workload_shape_only"],
            "notes": "Maximum supported context tokens for the configuration.",
        },
        {
            "raw_column_name": "Quantization",
            "normalized_field": "quantization",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["latency_prior", "throughput_prior"],
            "notes": "Applied quantization strategy (when distinct from Precision).",
        },
        {
            "raw_column_name": "Source_URL",
            "normalized_field": "source_url",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["not_usable"],
            "notes": (
                "Public URL of the upstream benchmark report each row was "
                "copied from. PRIMARY attribution and trust-tracing key — "
                "every row's TTFT/TPOT/throughput inherits the upstream "
                "source's methodology."
            ),
        },
        {
            "raw_column_name": "Source_Notes",
            "normalized_field": "source_notes",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["not_usable"],
            "notes": (
                "Free-text hardware-topology / measurement-context blurbs "
                "from the curator. Do NOT parse programmatically."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Audit + registry writers
# ---------------------------------------------------------------------------


def write_round5_audit_summary(ingest_out: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_version": "round5_broadened_discovery_audit_summary_v1",
        "scope": (
            "Round 5 broadened HF discovery (economic-priority pass) — "
            "bounded ingest of metrum-ai/llm-perfdata (Tier-4 multi-source "
            "curated TTFT/TPOT/throughput ledger across NVIDIA H100/H200/"
            "B200/L40S/RTX-4090, AMD MI300X/MI355X, Intel Gaudi 3, on "
            "vLLM/SGLang/vLLM-ROCm/vLLM-v0/vLLM-v1; weak strength → "
            "training_priors only) + 10 negative-result discovery records "
            "covering sairamn synthetic GCP billing, two ClarusC64 "
            "AI-safety coherence-risk evals, Phipper PE finance training "
            "data, uohna empty parquet, Lightcap tool-call telemetry "
            "(different trace class — deferred), metrum-ai dashboard "
            "(deferred), ssakethch H200 quantization (no license), crozai "
            "vLLM coding prompts (duplicate), intellistream sage agent "
            "benchmark (eval-only). Searched ~80 economic + operational + "
            "infrastructure HF queries; this round confirms the dataset "
            "space for the Aurelius constraint-aware engine is saturated "
            "for measured economic signals — the corpus still has NO "
            "dataset that joins (operational metric × measured "
            "GPU-hour-cost or measured kWh-per-request × verifiable "
            "provenance)."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "ingested": [
            {
                "dataset_id": ingest_out["summary"]["dataset_id"],
                "config_name": ingest_out["summary"]["config_name"],
                "canonical_trace_type": ingest_out["summary"][
                    "canonical_trace_type"],
                "license": ingest_out["summary"]["license"],
                "gated": ingest_out["summary"]["gated"],
                "analysis_sample_rows": ingest_out["summary"][
                    "analysis_sample_rows"],
                "fixture_sample_rows": ingest_out["summary"][
                    "fixture_sample_rows"],
                "committed_normalized_sample_rows": ingest_out["summary"][
                    "committed_normalized_sample_rows"],
                "committed_normalized_sample_bytes": ingest_out["summary"][
                    "committed_normalized_sample_bytes"],
                "available_signals": ingest_out["summary"]["available_signals"],
                "missing_signals": ingest_out["summary"]["missing_signals"],
                "limitations": ingest_out["summary"]["limitations"],
                "statistical_sample_strength": ingest_out["summary"][
                    "statistical_sample_strength"],
                "promotion_state": ingest_out["decision"]["state"],
                "promotion_tags": ingest_out["decision"]["promotion_tags"],
                "promotion_reasons": ingest_out["decision"]["reasons"],
            },
        ],
        "discovery_only_records": ROUND5_DISCOVERY_RECORDS,
        "failed": [],
        "economic_priority_summary": {
            "datasets_with_operational_and_economic_signals": [],
            "datasets_with_economic_only_signals": [],
            "join_keys_available_for_economic_overlays": [
                "model_id",
                "gpu_type",
                "engine",
                "precision",
                "concurrency",
            ],
            "scorer_coefficients_calibratable_from_round5": [],
            "scorer_coefficients_operator_policy_only_after_round5": [
                "gpu_hour_price_usd",
                "kwh_per_request",
                "carbon_g_per_kwh",
                "spot_interruption_probability",
                "egress_cost_per_gb",
                "regional_price_usd_per_mwh",
            ],
            "negative_result_finding": (
                "After ~80 targeted economic + operational searches, NO "
                "HF dataset was found that provides measured "
                "per-request economic signals (GPU-hour cost, kWh per "
                "request joined with TTFT/TPOT, spot price, or "
                "billing-truth chargeback) together with operational "
                "telemetry from the same campaign. metrum-ai contributes "
                "operational priors but carries zero economic columns. "
                "Aurelius' goodput/$ denominator remains "
                "operator-policy-supplied + public-pricing-prior + "
                "regional grid carbon intensity from ElectricityMaps / "
                "ENTSO-E (already integrated). This is a useful "
                "negative result: the public HF dataset space does NOT "
                "currently close the (operational × economic) join gap."
            ),
        },
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dest


def update_canonical_registry(registry_path: Path, entry: dict) -> dict:
    if registry_path.exists():
        d = json.loads(registry_path.read_text())
    else:
        d = {
            "doc_version": "hf_corpus_canonical_registry_v1",
            "entries": [],
        }
    existing = d.get("entries", [])
    filtered = [
        e for e in existing
        if not (e.get("dataset_id") == entry["dataset_id"]
                and e.get("config_name") == entry["config_name"])
    ]
    filtered.append(entry)
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
                              ingested_entry: dict) -> dict:
    if not candidates_path.exists():
        return {}
    d = json.loads(candidates_path.read_text())
    cands = d.get("candidates", [])

    target_id = ingested_entry["dataset_id"]
    found = False
    for c in cands:
        if c.get("dataset_id") == target_id:
            found = True
            c["recommended_action"] = "ingest_now_bounded"
            c["candidate_trace_type"] = "latency_benchmark_trace"
            c["trust_level"] = "tier_4_latency_benchmark_traces"
            c["gated_status"] = "public"
            c["license"] = "mit"
            c["matched_keywords"] = list(set(c.get("matched_keywords", []) + [
                "round5::economic_priority_pass",
                "round5::multi_source_curated_ledger",
                "latency_benchmark::ttft_tpot_throughput",
                "gpu_coverage::h100_h200_b200_gaudi3_mi300x_mi355x_l40s",
                "engine_coverage::sglang_vllm_rocm",
            ]))
            c["overall_priority_score"] = 3.0
            c["ingestion_feasibility_score"] = 5
            c["frontier_value_score"] = 3
            c["schema_quality_score"] = 4
            c["production_similarity_score"] = 2
            c["aurelius_use_case"] = (
                "Cross-hardware-tier latency / throughput priors covering "
                "H100, H200, B200, AMD MI300X / MI355X, Intel Gaudi 3, "
                "NVIDIA L40S / RTX-4090 — first appearance of these GPUs "
                "in the Aurelius federated corpus. Multi-source curated "
                "ledger; weak strength → training priors only."
            )
            c["not_recommended_uses"] = [
                "GPU-direct per-request latency calibration",
                "Queue / concurrency / batching risk calibration (no queue signal)",
                "Dynamic frontier calibration (Tier-4 benchmark)",
                "Goodput/$ denominator calibration (no measured cost/energy/billing)",
                "Carbon-cost calibration (no measured kWh per request)",
                "Single-source p95/p99 percentile claims",
            ]
            break
    if not found:
        cands.append({
            "dataset_id": target_id,
            "dataset_url": f"https://huggingface.co/datasets/{target_id}",
            "gated_status": "public",
            "license": "mit",
            "estimated_size": ["10K<n<100K"],
            "available_splits": ["train"],
            "schema_available": True,
            "matched_keywords": [
                "round5::economic_priority_pass",
                "round5::multi_source_curated_ledger",
                "latency_benchmark::ttft_tpot_throughput",
                "gpu_coverage::h100_h200_b200_gaudi3_mi300x_mi355x_l40s",
                "engine_coverage::sglang_vllm_rocm",
            ],
            "candidate_trace_type": "latency_benchmark_trace",
            "trust_level": "tier_4_latency_benchmark_traces",
            "available_signals": [
                "ttft", "tpot", "throughput", "concurrency",
                "input_tokens", "output_tokens", "gpu_type", "num_gpus",
                "model_id", "engine", "precision", "quantization",
                "context_window",
            ],
            "missing_signals": [
                "itl", "e2e_latency", "queue_state", "queue_wait",
                "queue_depth", "memory_pressure", "gpu_utilization",
                "batch_size", "timeout_label", "sla_label",
                "failure_label", "autoscaling", "replica_count",
                "kv_cache_size", "cache_hit", "kernel_duration",
                "carbon_intensity", "energy_per_request",
                "cost_per_token", "cost_per_request",
            ],
            "aurelius_use_case": (
                "Cross-hardware-tier latency / throughput priors covering "
                "GPUs and engines NOT present anywhere else in the corpus "
                "(H100, H200, B200, AMD MI300X / MI355X, Intel Gaudi 3, "
                "NVIDIA L40S / RTX-4090; SGLang, vLLM-ROCm, vLLM-v0/v1)."
            ),
            "not_recommended_uses": [
                "GPU-direct per-request latency calibration",
                "Queue / concurrency / batching risk calibration",
                "Dynamic frontier calibration",
                "Goodput/$ denominator calibration",
                "Carbon-cost calibration",
                "Single-source p95/p99 percentile claims",
            ],
            "ingestion_feasibility_score": 5,
            "frontier_value_score": 3,
            "schema_quality_score": 4,
            "production_similarity_score": 2,
            "overall_priority_score": 3.0,
            "recommended_action": "ingest_now_bounded",
            "feature_names": [
                "Model", "Size", "Precision", "GPU_Type", "Num_GPUs",
                "Serving_Engine", "Concurrency", "Tokens_per_sec",
                "TTFT_ms", "TPOT_ms", "Prompt_Tokens", "Output_Tokens",
                "Context_Window", "Quantization", "Source_URL",
                "Source_Notes",
            ],
            "classification_evidence": {
                "latency_benchmark_trace": [
                    "TTFT_ms", "TPOT_ms", "Tokens_per_sec", "Concurrency",
                    "GPU_Type", "Serving_Engine",
                ],
            },
            "downloads": 32,
            "likes": 0,
            "last_modified": "2026-04-01T00:00:00.000Z",
            "discovery_timestamp_s": time.time(),
            "configs": [CONFIG],
        })

    existing_ids = {c.get("dataset_id") for c in cands}
    for rec in ROUND5_DISCOVERY_RECORDS:
        if rec["dataset_id"] in existing_ids:
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
            "schema_available": True,
            "matched_keywords": [f"round5::{rec['kind']}"],
            "candidate_trace_type": rec["candidate_trace_type"],
            "trust_level": "tier_6_synthetic_benchmark_data",
            "available_signals": [],
            "missing_signals": [],
            "aurelius_use_case": "Round-5 discovery audit — see reason.",
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
            "round5_audit_reason": rec["reason"],
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
        description="Round-5 broadened HF ingest — metrum-ai/llm-perfdata"
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--skip-registry-update", action="store_true",
        help="Run the ingest only; do not modify the corpus registries.")
    p.add_argument(
        "--skip-audit", action="store_true",
        help="Skip writing the round-5 audit summary.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    out = ingest()
    print(f"\n[ingest] dataset={out['summary']['dataset_id']} "
          f"config={out['summary']['config_name']}")
    print(f"[ingest] analysis_rows={out['summary']['analysis_sample_rows']} "
          f"fixture_rows={out['summary']['fixture_sample_rows']} "
          f"committed_normalized_rows="
          f"{out['summary']['committed_normalized_sample_rows']}")
    print(f"[ingest] promotion_state={out['decision']['state']} "
          f"promotion_tags={out['decision']['promotion_tags']}")
    if out['decision']['reasons']:
        print(f"[ingest] promotion_reasons={out['decision']['reasons']}")

    if not args.skip_registry_update:
        registry_path = DISC_DIR / "canonical_corpus_registry.json"
        update_canonical_registry(registry_path, out["registry_entry"])
        candidates_path = DISC_DIR / "hf_dataset_candidates.json"
        update_candidate_registry(candidates_path, out["registry_entry"])
        print(f"[registry] updated {registry_path}")
        print(f"[candidates] updated {candidates_path}")

    if not args.skip_audit:
        audit_path = DISC_DIR / "round5_broadened_discovery_audit_summary.json"
        write_round5_audit_summary(out, audit_path)
        print(f"[audit] wrote {audit_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
