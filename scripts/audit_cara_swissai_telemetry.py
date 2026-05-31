#!/usr/bin/env python3
"""Focused HF telemetry-candidate audit for CARA + SwissAI.

This is a FOCUSED audit, not broad discovery. It:

1. Inspects HF metadata for ``asdwb/cara_latency_prediction`` and
   ``eth-easl/swissai-serving-trace`` (HF_TOKEN-honoured, never logged).
2. Downloads BOUNDED HTTP-Range chunks per file into the gitignored
   ``data/external/hf/<safe>/raw/`` directory.
3. Profiles every observed raw column + nested key into
   ``schema_profile.json`` + ``schema_mapping.json``.
4. Normalises bounded rows + writes per-config ``summary.json`` with
   ``statistical_sample_strength`` recorded honestly.
5. Computes per-config + stratified statistics (numeric distributions,
   per-subgroup p95/p99 with insufficient-sample flagging).
6. Writes federated registry entries through
   ``aurelius.traces.hf_corpus.promotion.evaluate_promotion`` so the
   sample-strength gate is enforced.

The script NEVER:
- modifies the production scheduler / robust energy engine / controllers
- claims production savings
- ingests > the per-file ``max_bytes`` budget (default 10 MiB per file)
- commits raw downloaded bytes (gitignored)
- prints / logs / echoes HF_TOKEN
- silently drops unknown columns (refusal lives in
  ``aurelius.traces.hf_corpus.ingestion.normalize_rows``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import (  # noqa: E402
    discovery, ingestion, promotion, schema_profile,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

# Each entry in TARGETS describes one (dataset, config, file) audit unit.
# ``trace_type`` is the manually-assigned canonical type after schema review.
# ``stratification_keys`` are normalized field names used for stratified
# subgroup analysis. ``raw_file`` is the HF repo-relative path.
TARGETS = [
    {
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "test_flat",
        "raw_file": "test.jsonl",
        "split": "test",
        "trace_type": "telemetry_trace",
        "is_nested": False,
        "stratification_keys": ["instance_type"],
        "latency_field": "actual_e2e_latency_s",
        "max_download_bytes": 10 * 1024 * 1024,  # 10 MiB head
        "limitations": [
            "Bounded head-sample of 49.1 MB test.jsonl (10 MiB cap).",
            "vLLM scheduler state + measured latency; CloudLab research "
            "cluster, NOT a production pilot.",
            "All requests served Qwen2.5 (3B/7B/14B/72B) on A100/V100/A30/P100.",
            "num_waiting is typically 0 due to vLLM continuous batching.",
            "actual_output_tokens recovered via round((e2e-ttft)/tpot)+1.",
        ],
    },
    {
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "test_queue_details",
        "raw_file": "test_queue_details.jsonl",
        "split": "test",
        "trace_type": "telemetry_trace",
        "is_nested": True,
        "stratification_keys": ["instance_type"],
        "latency_field": None,  # queue details file has no top-level latency
        "max_download_bytes": 10 * 1024 * 1024,
        "limitations": [
            "Bounded head-sample of 95.3 MB test_queue_details.jsonl (10 MiB).",
            "Per-request running_requests[] and waiting_requests[] arrays are "
            "preserved at the raw level; the normalised committed sample uses "
            "the flattened schedule_state.* counters only.",
            "Per-request actual_output_tokens enriched post-collection "
            "(99.96% match rate per README).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "trace",
        "raw_file": "trace.jsonl",
        "split": None,
        "trace_type": "request_shape_trace",
        "is_nested": False,
        "stratification_keys": ["model_id", "status"],
        "latency_field": None,
        "max_download_bytes": 10 * 1024 * 1024,  # 10 MiB head of 7 GB file
        "limitations": [
            "Bounded head-sample of 7.0 GB trace.jsonl (10 MiB head).",
            "reported_token_input / reported_token_output frequently -1 "
            "('unavailable'); treated as missing in statistics.",
            "model_parameters is heterogeneous; we extract a stable subset "
            "(temperature, max_tokens, top_p, seed) and JSON-stringify the "
            "rest as model_parameters_json.",
            "No GPU/hardware identity, no queue/scheduler state, no measured "
            "TTFT/TPOT. Latency = finished_at - created_at only.",
            "License: 'other' on HF card — researchers must verify ToS before "
            "redistributing raw rows. Only summary statistics are committed.",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen3_32b_buckets",
        "raw_file": "qwen3-32b-buckets.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": ["model_id", "status"],
        "latency_field": None,
        "max_download_bytes": 10 * 1024 * 1024,
        "limitations": [
            "Bounded head-sample of 4.6 GB qwen3-32b-buckets.jsonl (10 MiB).",
            "Token-bucket IDs are model+preprocessing-specific (Qwen/Qwen3-32B, "
            "16-token buckets, right padding); NOT vocabulary token IDs.",
            "bucket_ids list compressed in committed sample (hash + 5-id sample).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen3_32b_bucket_reuse",
        "raw_file": "qwen3-32b-bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        "max_download_bytes": 10 * 1024 * 1024,
        "limitations": [
            "Bounded head-sample of 3.7 GB qwen3-32b-bucket-reuse.jsonl (10 MiB).",
            "Pre-computed per-request reuse_percentage; cache hit rate is "
            "(reused_buckets/total_buckets), not a wall-clock cache hit.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Analysis-tier TARGETS (50-100 MiB; gitignored raw + analysis_sample;
# committed only the summary, schema profile/mapping, statistical_rollups,
# and a 5-row fixture).
#
# Stratification + schema mappings reuse the focused-tier mappings: the CARA
# train/test splits share an identical schema per the dataset README, and the
# SwissAI per-model bucket-reuse files all share the qwen3-32b schema.
# ---------------------------------------------------------------------------

_ANALYSIS_TIER_BYTES = 80 * 1024 * 1024  # 80 MiB head per analysis-tier file

ANALYSIS_TIER_TARGETS = [
    {
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "train_flat",
        "raw_file": "train.jsonl",
        "split": "train",
        "trace_type": "telemetry_trace",
        "is_nested": False,
        "stratification_keys": ["instance_type"],
        "latency_field": "actual_e2e_latency_s",
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 392.3 MB train.jsonl (80 MiB cap; ~75k rows).",
            "Train split identical schema to test split (CARA README, sweep 2).",
            "vLLM scheduler state + measured latency; CloudLab research cluster, "
            "NOT a production pilot.",
        ],
    },
    {
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "train_queue_details",
        "raw_file": "train_queue_details.jsonl",
        "split": "train",
        "trace_type": "telemetry_trace",
        "is_nested": True,
        "stratification_keys": ["instance_type"],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 812.4 MB train_queue_details.jsonl "
            "(80 MiB cap; ~36k rows).",
            "Identical schema to test_queue_details (per CARA README, sweep 2).",
            "Per-request running_requests[] + waiting_requests[] arrays preserved "
            "at the raw level; the normalised committed sample uses flattened "
            "schedule_state.* counters only.",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "trace_analysis",
        "raw_file": "trace.jsonl",
        "split": None,
        "trace_type": "request_shape_trace",
        "is_nested": False,
        "stratification_keys": ["model_id", "status"],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 7.0 GB trace.jsonl (80 MiB cap; ~190k rows).",
            "Identical schema to the 10 MiB focused audit (trace config).",
            "reported_token_input / reported_token_output frequently -1 "
            "('unavailable'); treated as missing in statistics.",
            "License: 'other' — only summary statistics committed.",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen3_32b_buckets_analysis",
        "raw_file": "qwen3-32b-buckets.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": ["model_id", "status"],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 4.6 GB qwen3-32b-buckets.jsonl "
            "(80 MiB cap; ~60k rows).",
            "Same schema as the 10 MiB focused audit (qwen3_32b_buckets).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen3_32b_bucket_reuse_analysis",
        "raw_file": "qwen3-32b-bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 3.7 GB qwen3-32b-bucket-reuse.jsonl "
            "(80 MiB cap; ~250k rows).",
            "Same schema as the 10 MiB focused audit (qwen3_32b_bucket_reuse).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "apertus_70b_bucket_reuse",
        "raw_file": "apertus-70b-bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        # apertus-70b file is 40 MB total — entire file fits in the budget.
        "max_download_bytes": 50 * 1024 * 1024,
        "limitations": [
            "Whole-file sample of 40 MB apertus-70b-bucket-reuse.jsonl "
            "(50 MiB cap covers the entire file).",
            "Same row schema as qwen3-32b-bucket-reuse; Apertus-70B served.",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen380b_instruct_bucket_reuse",
        "raw_file": "qwen380b_instruct_bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 307 MB qwen380b-instruct-bucket-reuse.jsonl "
            "(80 MiB cap).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "qwen380b_thinking_bucket_reuse",
        "raw_file": "qwen380b_thinking_bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 385 MB qwen380b-thinking-bucket-reuse.jsonl "
            "(80 MiB cap).",
        ],
    },
    {
        "dataset_id": "eth-easl/swissai-serving-trace",
        "config_name": "llama3_70b_bucket_reuse",
        "raw_file": "llama3-70b_bucket-reuse.jsonl",
        "split": None,
        "trace_type": "cache_residency_trace",
        "is_nested": False,
        "stratification_keys": [],
        "latency_field": None,
        "max_download_bytes": _ANALYSIS_TIER_BYTES,
        "limitations": [
            "Bounded head-sample of 4.9 GB llama3-70b_bucket-reuse.jsonl "
            "(80 MiB cap; ~125k rows).",
        ],
    },
]


# ---------------------------------------------------------------------------
# Manual schema_mapping per (dataset, config). Keyed by raw_column_name or
# flattened nested key (e.g. "model_parameters.temperature"). Every observed
# column the profiler finds will be classified. Unmapped columns are recorded
# as ``rejected_columns`` by ``schema_profile.build_schema_mapping``.
# ---------------------------------------------------------------------------

# Common pieces.
_FQ_REAL = "real"
_FQ_DERIVED = "derived"
_FQ_PROXY = "proxy"
_FQ_MISSING = "missing"

CARA_FLAT_MAPPING = {
    "request_id": {
        "normalized_field": "request_id", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["constraint_aware_backtest", "dynamic_frontier_calibration"],
        "notes": "UUID per CARA request.",
    },
    "instance_id": {
        "normalized_field": "instance_id", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "CloudLab hostname + port; identifies serving instance.",
    },
    "instance_type": {
        "normalized_field": "instance_type", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "gpu_resource",
        "usable_for": ["constraint_aware_backtest", "latency_prior",
                       "throughput_prior"],
        "notes": "model+gpu, e.g. 'qwen2.5-3b_p100'. Stratification key.",
    },
    "num_prompt_tokens": {
        "normalized_field": "num_prompt_tokens", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "tokens",
        "usable_for": ["latency_prior", "throughput_prior",
                       "constraint_aware_backtest"],
        "notes": "Input prompt length.",
    },
    "num_predicted_output_tokens": {
        "normalized_field": "num_predicted_output_tokens",
        "field_quality": _FQ_REAL, "units": "tokens",
        "aurelius_signal_category": "tokens",
        "usable_for": ["latency_prior"],
        "notes": "Max-tokens parameter; 1024 for all rows in CARA sweep 2.",
    },
    "actual_output_tokens": {
        "normalized_field": "actual_output_tokens", "field_quality": _FQ_DERIVED,
        "units": "tokens", "aurelius_signal_category": "tokens",
        "usable_for": ["latency_prior", "throughput_prior"],
        "notes": "Recovered via round((e2e-ttft)/tpot)+1 (per README); derived.",
    },
    "actual_e2e_latency": {
        "normalized_field": "actual_e2e_latency_s", "field_quality": _FQ_REAL,
        "units": "seconds", "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "constraint_aware_backtest",
                       "dynamic_frontier_calibration"],
        "notes": "Client-measured end-to-end latency.",
    },
    "actual_ttft": {
        "normalized_field": "actual_ttft_s", "field_quality": _FQ_REAL,
        "units": "seconds", "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "Time-to-first-token, client-measured.",
    },
    "actual_tpot": {
        "normalized_field": "actual_tpot_s", "field_quality": _FQ_REAL,
        "units": "seconds_per_token", "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "throughput_prior"],
        "notes": "Mean inter-token latency.",
    },
    "prediction_timestamp": {
        "normalized_field": "prediction_timestamp_s", "field_quality": _FQ_REAL,
        "units": "seconds_unix", "aurelius_signal_category": "request_dispatch",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Unix time of scheduling decision.",
    },
    "completion_timestamp": {
        "normalized_field": "completion_timestamp_s", "field_quality": _FQ_REAL,
        "units": "seconds_unix", "aurelius_signal_category": "request_completion",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Unix time of completion.",
    },
    "prediction_latency_ms": {
        "normalized_field": "prediction_latency_ms", "field_quality": _FQ_REAL,
        "units": "milliseconds", "aurelius_signal_category": "metadata_only",
        "usable_for": ["not_usable"],
        "notes": "Predictor inference overhead — not the workload latency.",
    },
    "probe_latency_ms": {
        "normalized_field": "probe_latency_ms", "field_quality": _FQ_REAL,
        "units": "milliseconds", "aurelius_signal_category": "metadata_only",
        "usable_for": ["not_usable"],
        "notes": "vLLM /instance_stats fetch overhead — out-of-band.",
    },
    "num_running": {
        "normalized_field": "num_running", "field_quality": _FQ_REAL,
        "units": "requests", "aurelius_signal_category": "queue",
        "usable_for": ["dynamic_frontier_calibration", "latency_prior",
                       "constraint_aware_backtest"],
        "notes": "Concurrent active requests on the instance at decision time.",
    },
    "num_waiting": {
        "normalized_field": "num_waiting", "field_quality": _FQ_REAL,
        "units": "requests", "aurelius_signal_category": "queue",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Queued requests; usually 0 (vLLM continuous batching).",
    },
    "num_active_decode_seqs": {
        "normalized_field": "num_active_decode_seqs", "field_quality": _FQ_REAL,
        "units": "sequences", "aurelius_signal_category": "queue",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Sequences in decode phase.",
    },
    "decode_ctx_p50": {
        "normalized_field": "decode_ctx_p50", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "queue",
        "usable_for": ["latency_prior"],
        "notes": "Decode-context length p50 (0 on some instances).",
    },
    "decode_ctx_p95": {
        "normalized_field": "decode_ctx_p95", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "queue",
        "usable_for": ["latency_prior"],
        "notes": "Decode-context length p95.",
    },
    "decode_ctx_max": {
        "normalized_field": "decode_ctx_max", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "queue",
        "usable_for": ["latency_prior"],
        "notes": "Decode-context length max.",
    },
    "pending_prefill_tokens": {
        "normalized_field": "pending_prefill_tokens", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "queue",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Pending prefill work; often 0.",
    },
    "pending_decode_tokens": {
        "normalized_field": "pending_decode_tokens", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "queue",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Pending decode work; 0 on some instances.",
    },
    "kv_cache_utilization": {
        "normalized_field": "kv_cache_utilization", "field_quality": _FQ_REAL,
        "units": "fraction", "aurelius_signal_category": "cache_residency",
        "usable_for": ["dynamic_frontier_calibration",
                       "cache_residency_evaluation"],
        "notes": "KV-cache utilisation ∈ [0,1].",
    },
    "kv_free_blocks": {
        "normalized_field": "kv_free_blocks", "field_quality": _FQ_REAL,
        "units": "blocks", "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Free KV-cache blocks.",
    },
    "token_budget_per_iter": {
        "normalized_field": "token_budget_per_iter", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "scheduler_state",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "vLLM scheduler iter budget.",
    },
    "prefill_chunk_size": {
        "normalized_field": "prefill_chunk_size", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "scheduler_state",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Chunked-prefill size.",
    },
    "max_num_seqs": {
        "normalized_field": "max_num_seqs", "field_quality": _FQ_REAL,
        "units": "sequences", "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Max concurrent sequences allowed.",
    },
    "num_preempted": {
        "normalized_field": "num_preempted", "field_quality": _FQ_REAL,
        "units": "requests", "aurelius_signal_category": "failure_timeout",
        "usable_for": ["dynamic_frontier_calibration"],
        "notes": "Cumulative preemption count.",
    },
    "ema_decode_tok_per_s": {
        "normalized_field": "ema_decode_tok_per_s", "field_quality": _FQ_REAL,
        "units": "tokens_per_second", "aurelius_signal_category": "throughput",
        "usable_for": ["throughput_prior"],
        "notes": "Exponential moving avg decode tok/s.",
    },
    "ema_prefill_tok_per_s": {
        "normalized_field": "ema_prefill_tok_per_s", "field_quality": _FQ_REAL,
        "units": "tokens_per_second", "aurelius_signal_category": "throughput",
        "usable_for": ["throughput_prior"],
        "notes": "EMA prefill tok/s.",
    },
    "ema_decode_iter_ms": {
        "normalized_field": "ema_decode_iter_ms", "field_quality": _FQ_REAL,
        "units": "milliseconds", "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "EMA per-iteration decode latency.",
    },
    "kv_evictions_per_s": {
        "normalized_field": "kv_evictions_per_s", "field_quality": _FQ_REAL,
        "units": "events_per_second", "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "KV-cache eviction rate.",
    },
    "running_requests_count": {
        "normalized_field": "running_requests_count", "field_quality": _FQ_DERIVED,
        "units": "requests", "aurelius_signal_category": "queue",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Derived from running_requests[] list (flat files only).",
    },
    "waiting_requests_count": {
        "normalized_field": "waiting_requests_count", "field_quality": _FQ_DERIVED,
        "units": "requests", "aurelius_signal_category": "queue",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Derived from waiting_requests[] list (flat files only).",
    },
}

# Queue-details file: same top-level + nested schedule_state.* keys + lists.
CARA_QUEUE_DETAILS_MAPPING = dict(CARA_FLAT_MAPPING)
# Top-level structural fields.
CARA_QUEUE_DETAILS_MAPPING.update({
    "schedule_state": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "scheduler_state",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Container dict; flattened into schedule_state.* keys.",
    },
})
# Nested schedule_state.* keys mirror flat schedule fields. Build them
# automatically from the flat mapping so the audit cannot drift.
_SCHEDULE_STATE_FIELDS = [
    "num_running", "num_waiting", "num_active_decode_seqs", "decode_ctx_p50",
    "decode_ctx_p95", "decode_ctx_max", "pending_prefill_tokens",
    "pending_decode_tokens", "kv_cache_utilization", "kv_free_blocks",
    "token_budget_per_iter", "prefill_chunk_size", "max_num_seqs",
    "num_preempted", "ema_decode_tok_per_s", "ema_prefill_tok_per_s",
    "ema_decode_iter_ms", "kv_evictions_per_s",
]
for _f in _SCHEDULE_STATE_FIELDS:
    CARA_QUEUE_DETAILS_MAPPING[f"schedule_state.{_f}"] = dict(
        CARA_FLAT_MAPPING[_f],
        notes=f"Nested schedule_state mirror of {_f}; same semantics.",
    )
CARA_QUEUE_DETAILS_MAPPING.update({
    "schedule_state.running_requests": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": "list[dict]", "aurelius_signal_category": "queue",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Per-running-request list; not flattened into normalised sample.",
    },
    "schedule_state.waiting_requests": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": "list[dict]", "aurelius_signal_category": "queue",
        "usable_for": ["constraint_aware_backtest"],
        "notes": "Per-waiting-request list; not flattened.",
    },
})


SWISS_TRACE_MAPPING = {
    "id": {
        "normalized_field": "request_id", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Composite request id; anonymised.",
    },
    "status": {
        "normalized_field": "status", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "failure_timeout",
        "usable_for": ["workload_shape_only", "constraint_aware_backtest"],
        "notes": "Request status: DEFAULT|ERROR|... ",
    },
    "created_at": {
        "normalized_field": "created_at_iso", "field_quality": _FQ_REAL,
        "units": "iso8601_string", "aurelius_signal_category": "request_arrival",
        "usable_for": ["workload_shape_only", "constraint_aware_backtest"],
        "notes": "Request arrival ISO-8601.",
    },
    "finished_at": {
        "normalized_field": "finished_at_iso", "field_quality": _FQ_REAL,
        "units": "iso8601_string",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Request completion ISO-8601; latency = finished - created.",
    },
    "model": {
        "normalized_field": "model_id", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Served model identifier (Qwen/Qwen3-32B etc).",
    },
    "reported_token_input": {
        "normalized_field": "prompt_tokens", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Frequently -1 (missing). Treated as missing in stats.",
    },
    "reported_token_output": {
        "normalized_field": "output_tokens", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Frequently -1 (missing).",
    },
    "model_parameters": {
        "normalized_field": "model_parameters_json", "field_quality": _FQ_REAL,
        "units": "json_string", "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Heterogeneous; JSON-stringified into model_parameters_json.",
    },
    "model_parameters.temperature": {
        "normalized_field": "temperature", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"], "notes": "Sampling temp.",
    },
    "model_parameters.max_tokens": {
        "normalized_field": "max_tokens_param", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"], "notes": "max_tokens param.",
    },
    "model_parameters.top_p": {
        "normalized_field": "top_p", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"], "notes": "nucleus-sampling top_p.",
    },
    "model_parameters.frequency_penalty": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Recorded but not extracted (heterogeneous types).",
    },
    "model_parameters.presence_penalty": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Recorded but not extracted (heterogeneous types).",
    },
    "model_parameters.seed": {
        "normalized_field": "seed", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"], "notes": "Sampling seed.",
    },
    "model_parameters.n": {
        "normalized_field": None, "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"], "notes": "Optional n param.",
    },
}


SWISS_BUCKETS_MAPPING = dict(SWISS_TRACE_MAPPING)
SWISS_BUCKETS_MAPPING.update({
    "token_count": {
        "normalized_field": "token_count", "field_quality": _FQ_REAL,
        "units": "tokens", "aurelius_signal_category": "tokens",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Qwen3-32B-tokenized input length.",
    },
    "bucket_ids": {
        "normalized_field": "bucket_ids_hash", "field_quality": _FQ_DERIVED,
        "units": "sha256_hex_16",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "List of 16-token bucket ids; hashed + sampled into 2 fields.",
    },
})


SWISS_BUCKET_REUSE_MAPPING = {
    "id": {
        "normalized_field": "request_id", "field_quality": _FQ_REAL,
        "units": None, "aurelius_signal_category": "metadata_only",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Composite request id.",
    },
    "created_at": {
        "normalized_field": "created_at_iso", "field_quality": _FQ_REAL,
        "units": "iso8601_string",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Request creation timestamp.",
    },
    "bucket_ids": {
        "normalized_field": "bucket_ids_hash", "field_quality": _FQ_DERIVED,
        "units": "sha256_hex_16",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "List of bucket ids; hashed for committed sample.",
    },
    "total_buckets": {
        "normalized_field": "bucket_count", "field_quality": _FQ_REAL,
        "units": "buckets", "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Total token buckets for this request.",
    },
    "reused_buckets": {
        "normalized_field": "reused_bucket_count", "field_quality": _FQ_REAL,
        "units": "buckets", "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Buckets reused from previous requests.",
    },
    "reuse_percentage": {
        "normalized_field": "reuse_percentage", "field_quality": _FQ_REAL,
        "units": "fraction", "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "Pre-computed = reused/total.",
    },
}


MAPPINGS = {
    ("asdwb/cara_latency_prediction", "test_flat"): CARA_FLAT_MAPPING,
    ("asdwb/cara_latency_prediction", "test_queue_details"):
        CARA_QUEUE_DETAILS_MAPPING,
    ("eth-easl/swissai-serving-trace", "trace"): SWISS_TRACE_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen3_32b_buckets"): SWISS_BUCKETS_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen3_32b_bucket_reuse"):
        SWISS_BUCKET_REUSE_MAPPING,
    # Analysis-tier configs reuse the same per-(dataset, schema-family) mappings.
    ("asdwb/cara_latency_prediction", "train_flat"): CARA_FLAT_MAPPING,
    ("asdwb/cara_latency_prediction", "train_queue_details"):
        CARA_QUEUE_DETAILS_MAPPING,
    ("eth-easl/swissai-serving-trace", "trace_analysis"): SWISS_TRACE_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen3_32b_buckets_analysis"):
        SWISS_BUCKETS_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen3_32b_bucket_reuse_analysis"):
        SWISS_BUCKET_REUSE_MAPPING,
    ("eth-easl/swissai-serving-trace", "apertus_70b_bucket_reuse"):
        SWISS_BUCKET_REUSE_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen380b_instruct_bucket_reuse"):
        SWISS_BUCKET_REUSE_MAPPING,
    ("eth-easl/swissai-serving-trace", "qwen380b_thinking_bucket_reuse"):
        SWISS_BUCKET_REUSE_MAPPING,
    ("eth-easl/swissai-serving-trace", "llama3_70b_bucket_reuse"):
        SWISS_BUCKET_REUSE_MAPPING,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch == "/":
            out.append("__")
        else:
            out.append("_")
    return "".join(out).lower()


def _safe_dataset_dir(dataset_id: str) -> str:
    safe = _safe_name(dataset_id)
    return os.path.join(REPO_ROOT, "data", "external", "hf", safe)


def _bounded_download(url: str, dest: str, *, max_bytes: int, token: str | None) -> dict:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {
        "User-Agent": "aurelius-cara-swissai-audit/1.0",
        "Range": f"bytes=0-{int(max_bytes - 1)}",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    written = 0
    status = None
    truncated = False
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.getcode()
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
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
            "url": url, "dest": dest, "status": int(e.code),
            "downloaded_bytes": 0, "truncated": False, "error": "HTTPError",
            "max_bytes": max_bytes,
        }
    except urllib.error.URLError as e:
        return {
            "url": url, "dest": dest, "status": None,
            "downloaded_bytes": 0, "truncated": False,
            "error": f"URLError:{e.reason}", "max_bytes": max_bytes,
        }
    return {
        "url": url, "dest": dest, "status": status,
        "downloaded_bytes": written, "truncated": truncated,
        "max_bytes": max_bytes, "error": None,
    }


def _read_jsonl_bounded(path: str, *, drop_last_partial: bool = True) -> list[dict]:
    """Parse the bounded raw chunk. Drops the last partial line if truncated."""
    rows: list[dict] = []
    with open(path, "rb") as fh:
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if drop_last_partial and lines:
        lines = lines[:-1]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _flatten_for_normalize(row: dict, target: dict) -> dict:
    """Prepare a raw row for ingestion.normalize_rows according to target.

    - For CARA queue_details: flatten ``schedule_state.*`` keys into top-level.
      Lists (``running_requests``, ``waiting_requests``) are dropped from the
      normalised row but their counts are recorded as
      ``running_requests_count`` / ``waiting_requests_count``.
    - For SwissAI trace + buckets: extract a subset of ``model_parameters.*``
      and JSON-stringify the rest as ``model_parameters_json``.
    - For SwissAI buckets: replace ``bucket_ids`` list with
      ``bucket_ids_hash`` + ``bucket_ids_sample``.
    """
    out = {}
    is_nested = target["is_nested"]
    for k, v in row.items():
        if k == "schedule_state" and isinstance(v, dict):
            for nk, nv in v.items():
                if nk == "running_requests":
                    out["running_requests_count"] = (
                        len(nv) if isinstance(nv, list) else None
                    )
                elif nk == "waiting_requests":
                    out["waiting_requests_count"] = (
                        len(nv) if isinstance(nv, list) else None
                    )
                else:
                    out[nk] = nv
        elif k == "model_parameters" and isinstance(v, dict):
            # Extract stable subset, normalise "null" string to None, list to None.
            for sub in ("temperature", "max_tokens", "top_p", "seed"):
                if sub in v:
                    val = v[sub]
                    if isinstance(val, str) and val.lower() == "null":
                        val = None
                    if sub == "max_tokens":
                        out["max_tokens_param"] = val if isinstance(
                            val, (int, float)) else None
                    elif sub in ("temperature", "top_p"):
                        out[sub] = val if isinstance(val, (int, float)) else None
                    elif sub == "seed":
                        out["seed"] = int(val) if isinstance(val, int) else None
            try:
                out["model_parameters_json"] = json.dumps(v, sort_keys=True)
            except (TypeError, ValueError):
                out["model_parameters_json"] = None
        elif k == "bucket_ids" and isinstance(v, list):
            out["bucket_ids_hash"] = schema_profile.hash_bucket_ids(v)
            out["bucket_ids_sample"] = schema_profile.sample_bucket_ids(v)
        else:
            out[k] = v
    return out


def _statistical_sample_strength(analysis_row_count: int) -> str:
    if analysis_row_count >= 10_000:
        return "strong"
    if analysis_row_count >= 1_000:
        return "moderate"
    if analysis_row_count >= 100:
        return "weak"
    return "fixture_only"


def _write_jsonl(rows: list[dict], path: str) -> tuple[int, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    for r in rows:
        line = json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
        buf.write(line.encode("utf-8"))
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data), hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Per-target audit driver
# ---------------------------------------------------------------------------


def audit_one(target: dict, *, token: str | None, force_redownload: bool) -> dict:
    dataset_id = target["dataset_id"]
    config = target["config_name"]
    safe_ds = _safe_dataset_dir(dataset_id)
    raw_path = os.path.join(safe_ds, "raw", target["raw_file"])
    processed_dir = os.path.join(safe_ds, config, "processed")
    schema_profile_path = os.path.join(processed_dir, "schema_profile.json")
    schema_mapping_path = os.path.join(processed_dir, "schema_mapping.json")
    summary_path = os.path.join(processed_dir, "summary.json")
    analysis_sample_path = os.path.join(processed_dir, "analysis_sample.jsonl")
    fixture_path = os.path.join(
        REPO_ROOT, "tests", "fixtures", "hf",
        f"{_safe_name(dataset_id)}__{_safe_name(config)}_sample.jsonl",
    )

    url = (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(target['raw_file'])}"
    )

    # 1. Bounded download (or reuse cached).
    if not force_redownload and os.path.exists(raw_path):
        manifest = {
            "url": url, "dest": raw_path,
            "downloaded_bytes": os.path.getsize(raw_path),
            "status": None, "truncated": True, "error": None,
            "max_bytes": target["max_download_bytes"],
            "cached": True,
        }
    else:
        manifest = _bounded_download(
            url, raw_path, max_bytes=target["max_download_bytes"], token=token
        )
        manifest["cached"] = False
    if manifest.get("error"):
        return {
            "target": target, "manifest": manifest,
            "audit_status": "download_failed",
        }

    # 2. Parse bounded rows.
    raw_rows = _read_jsonl_bounded(raw_path)

    # 3. Schema profile + mapping.
    profile = schema_profile.profile_rows(
        raw_rows, dataset_id=dataset_id, config_name=config,
        split=target.get("split"),
        source_files_inspected=[target["raw_file"]],
        file_size_bytes=manifest["downloaded_bytes"],
    )
    schema_profile.write_schema_profile(profile, schema_profile_path)

    column_mapping = MAPPINGS[(dataset_id, config)]
    mapping = schema_profile.build_schema_mapping(
        profile, column_mapping, dataset_id=dataset_id, config_name=config,
    )
    schema_profile.write_schema_mapping(mapping, schema_mapping_path)

    # 4. Normalize.
    pre_normalize = [_flatten_for_normalize(r, target) for r in raw_rows]
    try:
        normalized, unknown_cols, field_quality = ingestion.normalize_rows(
            pre_normalize, target["trace_type"],
            allow_unknown_columns=False,
            source_dataset_id=dataset_id, provenance=f"{dataset_id}@{config}",
        )
    except ingestion.IngestionUnknownColumns as e:
        return {
            "target": target, "manifest": manifest, "profile": profile,
            "mapping": mapping, "audit_status": "unknown_columns",
            "error": str(e),
        }
    normalized_schema = sorted({k for r in normalized for k in r.keys()})

    # 5. Analysis sample (uncommitted summary statistics; gitignored sample).
    #    Stratified per per-config keys if provided, otherwise head.
    per_stratum_cap = max(1, len(normalized))
    if target["stratification_keys"]:
        kept_idx, subgroup_counts = schema_profile.stratify_indices(
            normalized,
            stratification_keys=[
                k for k in target["stratification_keys"]
                if any(k in r for r in normalized)
            ],
            per_stratum_cap=per_stratum_cap,
        )
        sampling_method = "stratified" if kept_idx else "head"
    else:
        kept_idx = list(range(len(normalized)))
        subgroup_counts = {"__all__": len(normalized)}
        sampling_method = "head"
    analysis_rows = [normalized[i] for i in kept_idx]
    analysis_bytes, analysis_sha = _write_jsonl(analysis_rows, analysis_sample_path)

    # 6. Fixture sample (5 deterministic rows).
    fixture_rows = analysis_rows[:5]
    fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)

    # 7. Sample-strength label.
    strength = _statistical_sample_strength(len(analysis_rows))

    # 8. Numeric distribution summary (if a latency field exists).
    distribution = {}
    per_subgroup = {}
    if target.get("latency_field"):
        distribution[target["latency_field"]] = (
            schema_profile.compute_numeric_summary(
                analysis_rows, field=target["latency_field"]
            )
        )
        per_subgroup[target["latency_field"]] = (
            schema_profile.per_subgroup_latency_summary(
                analysis_rows,
                field=target["latency_field"],
                stratification_keys=target["stratification_keys"],
            )
        )

    # 9. Available + missing signals (derived from the actual normalized schema).
    inferred = set(
        ingestion.signals_from_normalized_schema(normalized_schema, normalized)
    )
    # Add explicit signals for status string / cache hit / sla.
    sample_statuses = {str(r.get("status") or "") for r in normalized}
    if any(s for s in sample_statuses):
        inferred.add("failure")
    if target["trace_type"] == "cache_residency_trace":
        inferred.update({"cache_hit", "prefix_cache"})
    if target["trace_type"] == "telemetry_trace":
        # CARA has measured TTFT + e2e + scheduler state + KV residency.
        if "actual_ttft_s" in normalized_schema:
            inferred.add("ttft")
        if "actual_e2e_latency_s" in normalized_schema:
            inferred.add("e2e_latency")
        if "num_running" in normalized_schema:
            inferred.add("concurrency")
        if "num_waiting" in normalized_schema or "pending_prefill_tokens" in normalized_schema:
            inferred.add("queue_depth")
        if "kv_cache_utilization" in normalized_schema:
            inferred.add("cache_hit")
    available_signals = sorted(inferred)
    missing_signals = [
        s for s in discovery.TARGET_SIGNALS if s not in set(available_signals)
    ]

    # 10. Field-quality grouping.
    real_fields = sorted([k for k, v in field_quality.items() if v == "real"])
    derived_fields = sorted([
        k for k, v in column_mapping.items()
        if v.get("field_quality") == "derived" and v.get("normalized_field")
    ])
    proxy_fields = sorted([
        k for k, v in column_mapping.items()
        if v.get("field_quality") == "proxy" and v.get("normalized_field")
    ])
    synthetic_fields: list[str] = []

    # 11. Write summary.
    raw_schema = sorted({k for r in raw_rows for k in r.keys()})
    summary = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_url": (
            f"https://huggingface.co/datasets/{dataset_id}"
        ),
        "license": _LICENSE_PER_DATASET[dataset_id],
        "gated": False,  # both datasets are public on HF (verified at audit time)
        "canonical_trace_type": target["trace_type"],
        "trust_tier": _TRUST_TIER_PER_TARGET[(dataset_id, config)],
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_bytes,
        "sample_sha256": fixture_sha,
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_bytes,
        "analysis_sample_rows": len(analysis_rows),
        "analysis_sample_bytes": analysis_bytes,
        "analysis_sample_sha256": analysis_sha,
        "sampling_method": sampling_method,
        "stratification_keys": target["stratification_keys"],
        "subgroup_counts": subgroup_counts,
        "statistical_sample_strength": strength,
        "raw_schema": raw_schema,
        "normalized_schema": normalized_schema,
        "unknown_columns": unknown_cols,
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
            f"#bytes={manifest['downloaded_bytes']}"
            f"#git={(_git_sha() or '')[:7]}"
        ),
        "ingestion_timestamp_s": time.time(),
        "git_sha": _git_sha(),
        "raw_download_manifest": manifest,
        "raw_file_size_committed": False,  # raw is gitignored
        "schema_profile_path": os.path.relpath(
            schema_profile_path, REPO_ROOT).replace(os.sep, "/"),
        "schema_mapping_path": os.path.relpath(
            schema_mapping_path, REPO_ROOT).replace(os.sep, "/"),
        "analysis_sample_path": os.path.relpath(
            analysis_sample_path, REPO_ROOT).replace(os.sep, "/"),
        "summary_path_relative": os.path.relpath(
            summary_path, REPO_ROOT).replace(os.sep, "/"),
        "distribution_summary": distribution,
        "per_subgroup_summary": per_subgroup,
        "rejected_columns_count": len(mapping.get("rejected_columns") or []),
    }
    # 11b. Statistical rollups (committed; the raw analysis_sample.jsonl is
    # gitignored, so rollups carry the per-config descriptive evidence).
    rollups = _compute_statistical_rollups(
        analysis_rows, trace_type=target["trace_type"], target=target,
    )
    rollups_path = os.path.join(processed_dir, "statistical_rollups.json")
    with open(rollups_path, "w") as fh:
        json.dump(rollups, fh, indent=2, sort_keys=True)
    summary["statistical_rollups_path"] = os.path.relpath(
        rollups_path, REPO_ROOT).replace(os.sep, "/")
    summary["rollups_subgroup_count"] = len(
        rollups.get("per_instance_type_latency", {}) or {}
    )
    summary["rollups_insufficient_groups"] = list(
        rollups.get("insufficient_sample_groups", []) or []
    )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # 12. Promotion decision.
    decision = promotion.evaluate_promotion(summary)
    entry = promotion.build_registry_entry(summary, decision)

    return {
        "target": target, "manifest": manifest, "profile": profile,
        "mapping": mapping, "summary": summary, "decision": decision,
        "entry": entry, "audit_status": "completed",
        "rollups": rollups,
    }


# ---------------------------------------------------------------------------
# Statistical rollups (committed) — descriptive per-subgroup distributions
# that summarise the larger gitignored analysis sample.
# ---------------------------------------------------------------------------


_PROMPT_TOKEN_BINS = [(0, 50), (50, 200), (200, 800), (800, 3200), (3200, 1_000_000)]
_QUEUE_DEPTH_BINS = [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1_000_000)]
_KV_UTIL_BINS = [(0.0, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 0.9), (0.9, 1.01)]


def _bin_label(v, bins):
    if v is None:
        return "missing"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in bins:
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f">={bins[-1][1]}"


def _compute_statistical_rollups(rows, *, trace_type, target):
    """Per-trace-type descriptive rollups.

    Numeric distributions use ``schema_profile.compute_numeric_summary``
    (which treats -1 as a missing sentinel and computes p50/p95/p99).
    Per-subgroup p99 is flagged INSUFFICIENT_SAMPLE_P99 below 100 rows.
    """
    from aurelius.traces.hf_corpus import schema_profile as _sp

    out = {
        "trace_type": trace_type,
        "config_name": target["config_name"],
        "row_count": len(rows),
        "per_instance_type_latency": {},
        "per_prompt_bin_latency": {},
        "per_queue_depth_bin_latency": {},
        "per_kv_util_bin_latency": {},
        "queue_vs_latency_correlation_proxy": None,
        "kv_util_vs_latency_correlation_proxy": None,
        "reuse_percentage_distribution": None,
        "prompt_tokens_distribution": None,
        "output_tokens_distribution": None,
        "ttft_overall": None,
        "tpot_overall": None,
        "e2e_overall": None,
        "insufficient_sample_groups": [],
    }
    if not rows:
        return out

    if trace_type == "telemetry_trace":
        # Overall latency distributions.
        for field, key in (
            ("actual_ttft_s", "ttft_overall"),
            ("actual_tpot_s", "tpot_overall"),
            ("actual_e2e_latency_s", "e2e_overall"),
        ):
            if any(field in r for r in rows):
                out[key] = _sp.compute_numeric_summary(rows, field=field)

        # Per-(instance_type) p50/p95/p99 for e2e + ttft + tpot.
        for field_label, field in (
            ("e2e", "actual_e2e_latency_s"),
            ("ttft", "actual_ttft_s"),
            ("tpot", "actual_tpot_s"),
        ):
            if not any(field in r for r in rows):
                continue
            grp = _sp.per_subgroup_latency_summary(
                rows, field=field,
                stratification_keys=["instance_type"],
            )
            out["per_instance_type_latency"][field_label] = grp["subgroups"]
            out["insufficient_sample_groups"].extend(
                f"{field_label}@{g}" for g in grp["insufficient_sample_groups"]
            )

        # Prompt-token-bin × e2e latency.
        if any("num_prompt_tokens" in r and "actual_e2e_latency_s" in r for r in rows):
            binned = {}
            for r in rows:
                if "actual_e2e_latency_s" not in r:
                    continue
                lab = _bin_label(r.get("num_prompt_tokens"), _PROMPT_TOKEN_BINS)
                binned.setdefault(lab, []).append(r)
            for lab, group in binned.items():
                s = _sp.compute_numeric_summary(group, field="actual_e2e_latency_s")
                if s["count"] < 100:
                    s["p99"] = None
                    s["flags"] = ["INSUFFICIENT_SAMPLE_P99"]
                    out["insufficient_sample_groups"].append(
                        f"e2e@prompt_bin={lab}"
                    )
                out["per_prompt_bin_latency"][lab] = s

        # Queue-depth bin × e2e latency.
        if any("num_running" in r and "actual_e2e_latency_s" in r for r in rows):
            binned = {}
            for r in rows:
                if "actual_e2e_latency_s" not in r:
                    continue
                lab = _bin_label(r.get("num_running"), _QUEUE_DEPTH_BINS)
                binned.setdefault(lab, []).append(r)
            for lab, group in binned.items():
                s = _sp.compute_numeric_summary(group, field="actual_e2e_latency_s")
                if s["count"] < 100:
                    s["p99"] = None
                    s["flags"] = ["INSUFFICIENT_SAMPLE_P99"]
                    out["insufficient_sample_groups"].append(
                        f"e2e@queue_depth_bin={lab}"
                    )
                out["per_queue_depth_bin_latency"][lab] = s

            # Correlation proxy: per-bin mean e2e.
            out["queue_vs_latency_correlation_proxy"] = {
                lab: s.get("mean")
                for lab, s in out["per_queue_depth_bin_latency"].items()
                if isinstance(s, dict)
            }

        # KV-cache-util bin × e2e latency.
        if any("kv_cache_utilization" in r and "actual_e2e_latency_s" in r for r in rows):
            binned = {}
            for r in rows:
                if "actual_e2e_latency_s" not in r:
                    continue
                lab = _bin_label(r.get("kv_cache_utilization"), _KV_UTIL_BINS)
                binned.setdefault(lab, []).append(r)
            for lab, group in binned.items():
                s = _sp.compute_numeric_summary(group, field="actual_e2e_latency_s")
                if s["count"] < 100:
                    s["p99"] = None
                    s["flags"] = ["INSUFFICIENT_SAMPLE_P99"]
                    out["insufficient_sample_groups"].append(
                        f"e2e@kv_util_bin={lab}"
                    )
                out["per_kv_util_bin_latency"][lab] = s
            out["kv_util_vs_latency_correlation_proxy"] = {
                lab: s.get("mean")
                for lab, s in out["per_kv_util_bin_latency"].items()
                if isinstance(s, dict)
            }

    elif trace_type == "cache_residency_trace":
        if any("reuse_percentage" in r for r in rows):
            out["reuse_percentage_distribution"] = _sp.compute_numeric_summary(
                rows, field="reuse_percentage")
        if any("bucket_count" in r for r in rows):
            out["bucket_count_distribution"] = _sp.compute_numeric_summary(
                rows, field="bucket_count")
        if any("token_count" in r for r in rows):
            out["token_count_distribution"] = _sp.compute_numeric_summary(
                rows, field="token_count")

    elif trace_type == "request_shape_trace":
        if any("prompt_tokens" in r for r in rows):
            out["prompt_tokens_distribution"] = _sp.compute_numeric_summary(
                rows, field="prompt_tokens")
        if any("output_tokens" in r for r in rows):
            out["output_tokens_distribution"] = _sp.compute_numeric_summary(
                rows, field="output_tokens")

    return out


_LICENSE_PER_DATASET = {
    "asdwb/cara_latency_prediction": "apache-2.0",
    "eth-easl/swissai-serving-trace": "other",
}

# Trust tier per (dataset, config). Manually assigned after schema review.
_TRUST_TIER_PER_TARGET = {
    ("asdwb/cara_latency_prediction", "test_flat"):
        "tier_2_public_telemetry_traces",
    ("asdwb/cara_latency_prediction", "test_queue_details"):
        "tier_2_public_telemetry_traces",
    ("eth-easl/swissai-serving-trace", "trace"):
        "tier_5_request_shape_traces",
    ("eth-easl/swissai-serving-trace", "qwen3_32b_buckets"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "qwen3_32b_bucket_reuse"):
        "tier_4_latency_benchmark_traces",
    # Analysis-tier (50-100 MiB samples).
    ("asdwb/cara_latency_prediction", "train_flat"):
        "tier_2_public_telemetry_traces",
    ("asdwb/cara_latency_prediction", "train_queue_details"):
        "tier_2_public_telemetry_traces",
    ("eth-easl/swissai-serving-trace", "trace_analysis"):
        "tier_5_request_shape_traces",
    ("eth-easl/swissai-serving-trace", "qwen3_32b_buckets_analysis"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "qwen3_32b_bucket_reuse_analysis"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "apertus_70b_bucket_reuse"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "qwen380b_instruct_bucket_reuse"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "qwen380b_thinking_bucket_reuse"):
        "tier_4_latency_benchmark_traces",
    ("eth-easl/swissai-serving-trace", "llama3_70b_bucket_reuse"):
        "tier_4_latency_benchmark_traces",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CARA + SwissAI HF telemetry audit.")
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument(
        "--write-registry", action="store_true", default=True,
        help="Update canonical_corpus_registry.json (default true).",
    )
    p.add_argument(
        "--registry-path", default=os.path.join(
            REPO_ROOT, "data", "external", "hf_discovery",
            "canonical_corpus_registry.json"
        )
    )
    p.add_argument(
        "--candidates-path", default=os.path.join(
            REPO_ROOT, "data", "external", "hf_discovery",
            "hf_dataset_candidates.json"
        )
    )
    p.add_argument(
        "--target-set",
        choices=("focused", "analysis_tier", "all"),
        default="focused",
        help="Which TARGETS list to ingest: focused (default, 10 MiB audit "
             "samples), analysis_tier (50-100 MiB samples per file for "
             "forecasting robustness), or all (both).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    token = os.environ.get("HF_TOKEN")

    if args.target_set == "focused":
        target_list = TARGETS
    elif args.target_set == "analysis_tier":
        target_list = ANALYSIS_TIER_TARGETS
    else:
        target_list = TARGETS + ANALYSIS_TIER_TARGETS

    results = []
    for tgt in target_list:
        print(f"\n=== {tgt['dataset_id']} / {tgt['config_name']} ===")
        r = audit_one(tgt, token=token, force_redownload=args.force_redownload)
        results.append(r)
        if r["audit_status"] == "completed":
            d = r["decision"]
            s = r["summary"]
            print(f"  trace_type        = {s['canonical_trace_type']}")
            print(f"  trust_tier        = {s['trust_tier']}")
            print(f"  fixture_rows      = {s['fixture_sample_rows']}")
            print(f"  analysis_rows     = {s['analysis_sample_rows']}")
            print(f"  sample_strength   = {s['statistical_sample_strength']}")
            print(f"  rejected_columns  = {s['rejected_columns_count']}")
            print(f"  promotion_state   = {d['state']}")
            print(f"  promotion_tags    = {d['promotion_tags']}")
            if d.get("reasons"):
                for reason in d["reasons"]:
                    print(f"    note: {reason}")
        else:
            print(f"  audit_status      = {r['audit_status']}")
            if "error" in r:
                print(f"  error             = {r['error']}")

    if args.write_registry:
        registry = promotion.load_canonical_registry(args.registry_path) or {
            "entries": []
        }
        existing = registry.get("entries") or []
        completed = [r for r in results if r["audit_status"] == "completed"]
        new_keys = {
            (r["entry"]["dataset_id"], r["entry"].get("config_name"))
            for r in completed
        }
        kept = [
            e for e in existing
            if (e.get("dataset_id"), e.get("config_name")) not in new_keys
        ]
        entries = kept + [r["entry"] for r in completed]
        promotion.write_canonical_registry(entries, args.registry_path)
        print(
            f"\n[audit] canonical registry updated: {args.registry_path} "
            f"(+{len(completed)} CARA/SwissAI entries; "
            f"{len(kept)} existing kept)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
