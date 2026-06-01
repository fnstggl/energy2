#!/usr/bin/env python3
"""Refresh ``hf_dataset_candidates.json`` with the deep-audit findings for
the 3 datasets ingested in PR ``claude/determined-pascal-w98qa``
(``ingest_hf_latency_benchmarks.py``).

The HF API discovery pass (``scripts/discover_hf_aurelius_datasets.py``)
only sees top-level tags and the dataset card frontmatter — it missed the
``results/`` directory inside ``odyn-network/odyn-benchmarks``, so the
existing candidate row mis-classifies it as ``reject_low_value`` with a
``frontier_value_score=1``. The deep audit found measured TTFT/TPOT/
e2e/throughput. This script overwrites that row and appends fresh rows
for ``memoriant/dgx-spark-kv-cache-benchmark`` and
``intellistream/vllm-hust-benchmark-results``.

Audit-only; produces no side effects beyond ``hf_dataset_candidates.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = (
    REPO_ROOT / "data" / "external" / "hf_discovery" / "hf_dataset_candidates.json"
)


REPLACEMENT_ODYN = {
    "dataset_id": "odyn-network/odyn-benchmarks",
    "dataset_url": "https://huggingface.co/datasets/odyn-network/odyn-benchmarks",
    "candidate_trace_type": "latency_benchmark_trace",
    "trust_level": "tier_4_latency_benchmark_traces",
    "gated_status": "public",
    "license": "apache-2.0",
    "downloads": 26,
    "likes": 0,
    "last_modified": "2026-03-23T23:02:44.000Z",
    "estimated_size": ["1K<n<10K"],
    "configs": [],
    "available_splits": [],
    "feature_names": [],
    "schema_available": True,
    "matched_keywords": [
        "vllm", "ray_serve", "ttft", "tpot", "e2e_latency",
        "request_throughput", "token_throughput", "concurrency",
        "batch_size",
    ],
    "available_signals": [
        "ttft", "tpot", "e2e_latency", "throughput",
        "concurrency", "engine", "gpu_type", "model_id",
        "failure_label",
    ],
    "missing_signals": [
        "itl", "input_tokens", "output_tokens", "kv_cache_size",
        "memory_pressure", "timeout_label", "p50_latency",
        "p90_latency", "p99_latency",
    ],
    "aurelius_use_case": (
        "Performance-surface priors (TTFT_avg/p95, TPOT_avg/p95, e2e_avg/p95, "
        "throughput by model × hardware × concurrency × profile); "
        "concurrency-saturation priors via `failed` counts at concurrencies "
        "≥ 192; profile-aware request-shape priors for the eval/batch frontier."
    ),
    "not_recommended_uses": [
        "Real arrival/queue scheduling — benchmark, no arrival trace",
        "Production latency calibration — vLLM benchmark, not pilot",
        "Cross-deployment generalisation outside (model, gpu, engine) tuple",
    ],
    "ingestion_feasibility_score": 5,
    "frontier_value_score": 4,
    "schema_quality_score": 4,
    "production_similarity_score": 3,
    "overall_priority_score": 4.0,
    "recommended_action": "ingested_2026_06_01",
    "classification_evidence": {
        "latency_benchmark_trace": [
            "vllm", "ttft_avg", "ttft_p95", "tpot_avg", "tpot_p95",
            "e2e_avg", "e2e_p95", "throughput_tok_s", "throughput_req_s",
            "concurrency", "batch_size",
        ],
    },
    "deep_audit_note": (
        "Deep audit found measured TTFT_avg/p95, TPOT_avg/p95, e2e_avg/p95, "
        "throughput in `results/<model>_results/{chat,batch}_benchmarks.csv` — "
        "the top-level API listing only sees the prompt profile CSVs."
    ),
    "discovery_timestamp_s": time.time(),
}

NEW_MEMORIANT = {
    "dataset_id": "memoriant/dgx-spark-kv-cache-benchmark",
    "dataset_url": "https://huggingface.co/datasets/memoriant/dgx-spark-kv-cache-benchmark",
    "candidate_trace_type": "latency_benchmark_trace",
    "trust_level": "tier_4_latency_benchmark_traces",
    "gated_status": "public",
    "license": "apache-2.0",
    "downloads": 44,
    "likes": 0,
    "last_modified": "2026-04-01T00:00:00.000Z",
    "estimated_size": ["n<1K"],
    "configs": [],
    "available_splits": [],
    "feature_names": [
        "context_tokens", "cache_type", "kv_buffer_mib",
        "gpu_mem_mib", "prompt_tps", "gen_tps", "notes",
    ],
    "schema_available": True,
    "matched_keywords": [
        "kv-cache", "quantization", "llama.cpp", "nvidia",
        "dgx-spark", "gb10", "benchmarking", "inference",
    ],
    "available_signals": [
        "throughput", "kv_cache_size", "memory_pressure",
        "gpu_type", "model_id", "engine",
    ],
    "missing_signals": [
        "ttft", "tpot", "itl", "e2e_latency", "concurrency",
        "batch_size", "failure_label", "timeout_label",
    ],
    "aurelius_use_case": (
        "KV-cache memory-pressure priors (216 / 408 / 768 MiB per 110K "
        "context for q4_0 / q8_0 / f16); cache-quantization throughput "
        "trade-off priors (gen_tps degrades ~37% at 110K context under q4_0); "
        "GB10 Grace Blackwell unified-memory residency priors."
    ),
    "not_recommended_uses": [
        "Latency frontier source on its own (no TTFT/TPOT)",
        "Generalisation beyond GB10 (single GPU class)",
    ],
    "ingestion_feasibility_score": 5,
    "frontier_value_score": 3,
    "schema_quality_score": 5,
    "production_similarity_score": 2,
    "overall_priority_score": 3.5,
    "recommended_action": "ingested_2026_06_01",
    "classification_evidence": {
        "latency_benchmark_trace": [
            "kv-cache", "benchmarking", "llama.cpp", "gen_tps",
            "kv_buffer_mib",
        ],
    },
    "discovery_timestamp_s": time.time(),
}

NEW_INTELLISTREAM = {
    "dataset_id": "intellistream/vllm-hust-benchmark-results",
    "dataset_url": "https://huggingface.co/datasets/intellistream/vllm-hust-benchmark-results",
    "candidate_trace_type": "latency_benchmark_trace",
    "trust_level": "tier_4_latency_benchmark_traces",
    "gated_status": "public",
    "license": None,
    "downloads": 519,
    "likes": 0,
    "last_modified": "2026-06-01T01:24:55.000Z",
    "estimated_size": [],
    "configs": [],
    "available_splits": [],
    "feature_names": [
        "entry_id", "engine", "engine_version", "config_type",
        "hardware", "model", "workload", "metrics", "constraints",
    ],
    "schema_available": True,
    "matched_keywords": [
        "vllm", "hust", "leaderboard", "ttft_ms", "tbt_ms",
        "throughput_tps", "huawei", "910b3", "ascend",
    ],
    "available_signals": [
        "ttft", "tpot", "throughput", "concurrency",
        "batch_size", "input_tokens", "output_tokens",
        "gpu_type", "model_id", "engine", "engine_version",
        "memory_pressure", "failure_label",
    ],
    "missing_signals": [
        "itl", "e2e_latency", "kv_cache_size", "timeout_label",
        "p50_latency", "p90_latency", "p95_latency", "p99_latency",
    ],
    "aurelius_use_case": (
        "Performance-surface priors for Ascend-class hardware (TTFT_ms + "
        "TBT_ms + throughput at this granularity for Huawei 910B3); "
        "engine-version comparison priors (vLLM vs vLLM-HUST forks)."
    ),
    "not_recommended_uses": [
        "Cross-vendor generalisation outside Ascend-class",
        "Production latency calibration (Tier 4 benchmark)",
        "Memory-pressure analysis when `peak_mem_mb` is zero",
        "Treating `error_rate == 0` as truth (upper bound only)",
    ],
    "ingestion_feasibility_score": 5,
    "frontier_value_score": 3,
    "schema_quality_score": 5,
    "production_similarity_score": 3,
    "overall_priority_score": 3.0,
    "recommended_action": "ingested_2026_06_01",
    "classification_evidence": {
        "latency_benchmark_trace": [
            "ttft_ms", "tbt_ms", "throughput_tps", "vllm",
            "concurrent_requests", "input_length", "output_length",
        ],
    },
    "deep_audit_note": (
        "License is NOT declared on the HF card; treated as license=None "
        "with no committed normalised sample under the conservative "
        "redistribution policy."
    ),
    "discovery_timestamp_s": time.time(),
}


def main() -> int:
    with open(CANDIDATES_PATH) as fh:
        doc = json.load(fh)

    candidates = list(doc.get("candidates", []))
    new_ids = {
        "odyn-network/odyn-benchmarks",
        "memoriant/dgx-spark-kv-cache-benchmark",
        "intellistream/vllm-hust-benchmark-results",
    }
    # Drop any pre-existing rows for these ids and append the fresh ones.
    candidates = [c for c in candidates if c.get("dataset_id") not in new_ids]
    candidates.extend([REPLACEMENT_ODYN, NEW_MEMORIANT, NEW_INTELLISTREAM])
    candidates.sort(key=lambda c: c.get("dataset_id") or "")

    doc["candidates"] = candidates
    doc["updated_at_s"] = time.time()
    doc["updated_at_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )

    with open(CANDIDATES_PATH, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print(f"Updated {CANDIDATES_PATH} — {len(candidates)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
