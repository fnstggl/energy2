#!/usr/bin/env python3
"""Broadened HF discovery — bounded ingest of 3 new Tier-4 latency benchmarks.

This script extends the federated benchmark corpus with three latency-benchmark
datasets that were flagged INGEST_LATER / MONITOR in the gap-closure audit
(PR #125) and that have since been confirmed to contain measured TTFT / TPOT /
throughput / concurrency signals during the focused-audit follow-on:

1. ``odyn-network/odyn-benchmarks`` — vLLM + Ray Serve benchmark across 4
   prompt profiles × 2 models (Qwen2.5-7B-Instruct on DGX Spark Blackwell,
   facebook/opt-125m on RTX 3090) × multiple concurrency levels. Apache-2.0.
2. ``memoriant/dgx-spark-kv-cache-benchmark`` — corrected v3 KV cache
   quantization benchmark on NVIDIA DGX Spark (GB10, Grace Blackwell unified
   memory) for llama.cpp. Apache-2.0. Tiny (<1 KiB).
3. ``intellistream/vllm-hust-benchmark-results`` — vLLM-HUST benchmark
   leaderboard with measured TTFT_ms / TBT_ms / throughput_tps across
   Huawei 910B3 hardware and several Qwen / DeepSeek models. No declared
   license (treated as ``None``).

All three are Tier-4 ``latency_benchmark_trace`` candidates. None are pilot
telemetry — they remain research-class priors only. The script writes:

    data/external/hf/<safe_dataset>/raw/<file>              # gitignored
    data/external/hf/<safe_dataset>/<config>/processed/
        schema_profile.json               (committed)
        schema_mapping.json               (committed)
        summary.json                      (committed)
        statistical_rollups.json          (committed)
        analysis_sample.jsonl             (gitignored ≥)
        committed_normalized_sample.jsonl (committed if license permits, ≤ 100 KiB)
    tests/fixtures/hf/<safe_dataset>__<config>_sample.jsonl (committed, ≤ 16 KiB)

It also writes a discovery audit summary at
``data/external/hf_discovery/broadened_discovery_audit_summary.json`` that
includes rejected / blocked datasets from the same pool.

Redistribution gate wiring
--------------------------

Sixth consumer of the canonical
:func:`aurelius.ingestion.redistribution_gate.decide_redistribution`
gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``, and
``scripts/ingest_hf_llm_energy_consumption.py``).

This script differs from the first five gate consumers because it
covers **three datasets with two different license tags in one
script** (mixed Apache-2.0 / ``None``):

* ``odyn-network/odyn-benchmarks`` — ``apache-2.0`` → gate permits →
  committed normalised sample.
* ``memoriant/dgx-spark-kv-cache-benchmark`` — ``apache-2.0`` → gate
  permits → committed normalised sample.
* ``intellistream/vllm-hust-benchmark-results`` — ``None`` → gate
  denies under the default ``deny_all`` / zero-grants ledger → no
  committed normalised sample (the pre-existing
  ``"license_unspecified_no_redistribution_promise"`` reason string is
  preserved verbatim on the ``committed_normalized_sample_reason_skipped``
  field for downstream-test back-compat; the canonical gate verdict
  lives in the additive ``redistribution_gate_*`` fields).

The on-disk committed normalised samples are byte-for-byte unchanged
under the gate-wired path; only the additive ``redistribution_gate_*``
metadata gains presence in the per-config summary.json and the
broadened-discovery audit summary.

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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

FIXTURE_MAX_BYTES = 16 * 1024
COMMITTED_NORMALIZED_MAX_BYTES = 100 * 1024  # 100 KiB per file
PROMOTION_BOUNDED_GUARD = 16 * 1024 * 1024
PER_DATASET_TIMEOUT_S = 30 * 60

# ---------------------------------------------------------------------------
# Per-dataset license metadata — single source of truth for the gate
# ---------------------------------------------------------------------------
# Each dataset's raw HF license tag lives at module level so a future tag
# change (e.g. an upstream owner declaring ``license: mit``) is a one-line
# edit; the canonical
# :func:`aurelius.ingestion.redistribution_gate.decide_redistribution` gate
# derives both the ``permissive_*`` / ``unspecified_no_committed_sample`` /
# ``declared_non_permissive`` status code AND the redistribute-yes/no
# commit decision. This script never classifies the tag itself.

ODYN_DATASET_ID = "odyn-network/odyn-benchmarks"
ODYN_LICENSE_TAG: Optional[str] = "apache-2.0"
ODYN_LICENSE_SOURCE = "HF card frontmatter license: apache-2.0"

MEMORIANT_DATASET_ID = "memoriant/dgx-spark-kv-cache-benchmark"
MEMORIANT_LICENSE_TAG: Optional[str] = "apache-2.0"
MEMORIANT_LICENSE_SOURCE = "HF card frontmatter license: apache-2.0"

INTELLISTREAM_DATASET_ID = "intellistream/vllm-hust-benchmark-results"
INTELLISTREAM_LICENSE_TAG: Optional[str] = None  # no `license:` in frontmatter
INTELLISTREAM_LICENSE_SOURCE = (
    "HF card frontmatter has no `license:` field; recorded as unspecified"
)

GATE_SCOPE = "committed_normalized_sample"

# Pre-existing script-level reason string for skipping the committed
# normalised sample under ``license = None``. Kept VERBATIM — downstream
# tests (e.g. ``test_intellistream_has_no_committed_normalized_sample``
# in ``tests/test_hf_latency_benchmarks_ingest.py``) pin this exact
# string. The canonical gate verdict is recorded additively in the new
# ``redistribution_gate_*`` fields.
COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON = (
    "license_unspecified_no_redistribution_promise"
)

logger = logging.getLogger("aurelius.hf_latency_benchmarks_ingest")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def _download(url: str, dest: Path, *, max_bytes: int) -> dict:
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "url": url, "dest": str(dest), "downloaded_bytes": dest.stat().st_size,
            "cached": True, "truncated": None, "error": None,
        }
    req = urllib.request.Request(url, headers=headers)
    truncated = False
    written = 0
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
        return {
            "url": url, "dest": str(dest), "downloaded_bytes": 0,
            "cached": False, "truncated": None, "error": f"HTTP {e.code}",
        }
    return {
        "url": url, "dest": str(dest), "downloaded_bytes": written,
        "cached": False, "truncated": truncated, "error": None,
    }


def _hf_url(dataset_id: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{urllib.parse.quote(path)}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bytes_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Redistribution gate — wire the canonical gate, do not classify here
# ---------------------------------------------------------------------------


def _load_ledger(policy_path: Path = POLICY_PATH) -> OperatorPolicyLedger:
    """Load the operator policy ledger from disk, or fall back to empty.

    The committed default file ships zero grants under
    ``policy_default=deny_all``; an absent file is identical in
    behaviour. We use ``empty()`` as the fallback so the script
    remains self-sufficient in a fresh checkout that may not yet have
    the committed JSON pulled — the gate still produces correct
    decisions (deny for ``license = None`` intellistream, permit for
    ``apache-2.0`` odyn/memoriant) instead of crashing.
    """

    if policy_path.exists():
        return OperatorPolicyLedger.load(policy_path)
    return OperatorPolicyLedger.empty()


def evaluate_redistribution(
    *,
    ledger: OperatorPolicyLedger,
    dataset_id: str,
    license_tag: Optional[str],
    scope: str = GATE_SCOPE,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Ask the canonical gate whether the committed normalised sample of
    a given dataset may be redistributed under the supplied license tag.

    Pure function — no I/O. Exposed so tests can drive the gate path
    without invoking the CSV/JSON download / normalisation pipeline.
    This script covers three datasets with two distinct license tags
    so ``dataset_id`` and ``license_tag`` are required arguments (no
    sensible default exists at the module level — each ingest function
    supplies its own pair).

    Under the default-empty ledger:

    * ``license_tag = "apache-2.0"`` → permitted=True, license_status=
      ``"permissive_apache_2_0"``, reason_code=
      ``"permitted_declared_permissive_license"`` (odyn + memoriant).
    * ``license_tag = None`` → permitted=False, license_status=
      ``"unspecified_no_committed_sample"``, reason_code=
      ``"no_grant_recorded"`` (intellistream).

    Both verdicts match the v1 commit behaviour byte-for-byte (odyn +
    memoriant samples committed, intellistream skipped).
    """

    return decide_redistribution(
        dataset_id=dataset_id,
        license_str=license_tag,
        scope=scope,
        ledger=ledger,
        now_iso=now_iso,
    )


# ---------------------------------------------------------------------------
# Field-quality + signals (Aurelius signal vocabulary)
# ---------------------------------------------------------------------------


AURELIUS_SIGNALS = [
    "ttft", "tpot", "itl", "e2e_latency", "throughput",
    "concurrency", "batch_size", "input_tokens", "output_tokens",
    "gpu_type", "model_id", "engine", "engine_version",
    "failure_label", "timeout_label", "memory_pressure",
    "kv_cache_size",
]


# ---------------------------------------------------------------------------
# Odyn ingest
# ---------------------------------------------------------------------------


def _ingest_odyn(safe_root: Path,
                 ledger: Optional[OperatorPolicyLedger] = None) -> list[dict]:
    """odyn-network/odyn-benchmarks → 2 configs:

    * ``qwen_chat_streaming`` — Qwen2.5-7B-Instruct on DGX Spark Blackwell,
      streaming chat across 4 profiles × 8 concurrency levels.
    * ``facebook_chat_streaming`` — facebook/opt-125m on RTX 3090, streaming
      chat across 4 profiles × 6 concurrency levels.

    Both come from ``results/<model>_results/chat_benchmarks.csv`` which
    contains measured TTFT_avg / TTFT_p95 / TPOT_avg / TPOT_p95 / e2e_avg /
    e2e_p95 / throughput_tok_s / throughput_req_s / failure counts.
    """
    if ledger is None:
        ledger = _load_ledger()
    dataset_id = ODYN_DATASET_ID
    license = ODYN_LICENSE_TAG
    gate_decision = evaluate_redistribution(
        ledger=ledger,
        dataset_id=dataset_id,
        license_tag=license,
    )
    base = HF_DIR / "odyn-network__odyn-benchmarks"
    base_raw = base / "raw"
    base_raw.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    git_sha = _git_sha()

    files = {
        "qwen_chat_streaming": "results/qwen_results/chat_benchmarks.csv",
        "facebook_chat_streaming": "results/facebook_results/chat_benchmarks.csv",
        "qwen_batch": "results/qwen_results/batch_benchmarks.csv",
        "facebook_batch": "results/facebook_results/batch_benchmarks.csv",
    }

    # The (model, hardware, engine) metadata that ships with each file.
    config_meta = {
        "qwen_chat_streaming": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "model_family": "Qwen2.5",
            "gpu": "DGX Spark (Blackwell)",
            "engine": "vllm",
        },
        "facebook_chat_streaming": {
            "model": "facebook/opt-125m",
            "model_family": "OPT",
            "gpu": "RTX 3090",
            "engine": "vllm",
        },
        "qwen_batch": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "model_family": "Qwen2.5",
            "gpu": "DGX Spark (Blackwell)",
            "engine": "vllm",
        },
        "facebook_batch": {
            "model": "facebook/opt-125m",
            "model_family": "OPT",
            "gpu": "RTX 3090",
            "engine": "vllm",
        },
    }

    for cfg, remote in files.items():
        local = base_raw / remote.replace("/", "__")
        manifest = _download(_hf_url(dataset_id, remote), local,
                             max_bytes=1024 * 1024)
        if manifest["error"]:
            logger.warning("odyn: %s — %s", remote, manifest["error"])
            continue

        with open(local, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            logger.warning("odyn: %s — empty", remote)
            continue

        is_chat = cfg.endswith("_chat_streaming")
        meta = config_meta[cfg]

        # Normalise into BenchmarkLatencyRecord-shaped jsonl. For chat
        # streaming rows: keep both streaming + non_streaming. For batch:
        # there is no TTFT — only e2e per-batch.
        normalized_rows = []
        for r in rows:
            try:
                if is_chat:
                    rec = {
                        "source_dataset_id": dataset_id,
                        "trace_type": "latency_benchmark_trace",
                        "provenance": (
                            f"{dataset_id}@{cfg}#{remote}#git={git_sha[:7]}"
                        ),
                        "profile": r.get("profile"),
                        "mode": r.get("mode"),
                        "model": meta["model"],
                        "model_family": meta["model_family"],
                        "gpu": meta["gpu"],
                        "engine": meta["engine"],
                        "concurrency": int(r["concurrency"]),
                        "total_requests": int(r["total_requests"]),
                        "successful": int(r["successful"]),
                        "failed": int(r["failed"]),
                        "wall_time_s": float(r["wall_time_s"]),
                        "mean_e2el_ms": float(r["e2e_avg"]),
                        "p95_e2el_ms": float(r["e2e_p95"]),
                        "mean_tpot_ms": float(r["tpot_avg"]),
                        "p95_tpot_ms": float(r["tpot_p95"]),
                        "throughput_tok_s": float(r["throughput_tok_s"]),
                        "throughput_req_s": float(r["throughput_req_s"]),
                        "mean_ttft_ms": (
                            float(r["ttft_avg"]) if r.get("ttft_avg") else None
                        ),
                        "p95_ttft_ms": (
                            float(r["ttft_p95"]) if r.get("ttft_p95") else None
                        ),
                    }
                else:
                    rec = {
                        "source_dataset_id": dataset_id,
                        "trace_type": "latency_benchmark_trace",
                        "provenance": (
                            f"{dataset_id}@{cfg}#{remote}#git={git_sha[:7]}"
                        ),
                        "profile": r.get("profile"),
                        "model": meta["model"],
                        "model_family": meta["model_family"],
                        "gpu": meta["gpu"],
                        "engine": meta["engine"],
                        "batch_size": (
                            int(r["batch_size"]) if r.get("batch_size") else None
                        ),
                        "num_prompts": int(r["num_prompts"]),
                        "num_results": int(r["num_results"]),
                        "submit_ms": float(r["submit_ms"]),
                        "total_ms": float(r["total_ms"]),
                        "avg_per_prompt_ms": float(r["avg_per_prompt_ms"]),
                        "throughput_prompts_s": float(r["throughput_prompts_s"]),
                    }
                normalized_rows.append(rec)
            except (KeyError, ValueError) as e:
                logger.warning("odyn: skipped malformed row in %s: %s", remote, e)
                continue

        out.append(_finalize_config(
            dataset_id=dataset_id,
            safe_name="odyn-network__odyn-benchmarks",
            config=cfg,
            source_file_relative=remote,
            raw_local_path=local,
            raw_columns=list(rows[0].keys()),
            normalized_rows=normalized_rows,
            raw_rows=rows,
            license=license,
            license_source=ODYN_LICENSE_SOURCE,
            gate_decision=gate_decision,
            gated=False,
            field_quality=_odyn_field_quality(is_chat),
            available_signals=_odyn_available_signals(is_chat),
            missing_signals=_odyn_missing_signals(is_chat),
            limitations=_odyn_limitations(cfg, meta),
            stratification_keys=(
                ["profile", "concurrency", "mode"] if is_chat
                else ["profile", "batch_size"]
            ),
            git_sha=git_sha,
        ))
    return out


def _odyn_field_quality(is_chat: bool) -> dict:
    if is_chat:
        return {
            "model": "real",
            "model_family": "real",
            "gpu": "real",
            "engine": "real",
            "profile": "real",
            "concurrency": "real",
            "num_requests": "real",
            "mean_ttft_ms": "real",
            "p90_ttft_ms": "missing",
            "p99_ttft_ms": "missing",
            "p50_ttft_ms": "missing",
            "mean_tpot_ms": "real",
            "p90_tpot_ms": "missing",
            "p99_tpot_ms": "missing",
            "p50_tpot_ms": "missing",
            "mean_e2el_ms": "real",
            "p90_e2el_ms": "missing",
            "p99_e2el_ms": "missing",
            "p50_e2el_ms": "missing",
            "request_throughput": "real",
            "total_token_throughput": "real",
            "duration_s": "real",
            "input_token_throughput": "missing",
            "output_token_throughput": "missing",
            "mean_itl_ms": "missing",
            "p50_itl_ms": "missing",
            "p90_itl_ms": "missing",
            "p99_itl_ms": "missing",
            "run_id": "missing",
            "tensor_parallelism": "missing",
        }
    return {
        "model": "real",
        "model_family": "real",
        "gpu": "real",
        "engine": "real",
        "profile": "real",
        "concurrency": "missing",
        "num_requests": "real",
        "mean_ttft_ms": "missing",
        "p50_ttft_ms": "missing",
        "p90_ttft_ms": "missing",
        "p99_ttft_ms": "missing",
        "mean_tpot_ms": "missing",
        "p50_tpot_ms": "missing",
        "p90_tpot_ms": "missing",
        "p99_tpot_ms": "missing",
        "mean_e2el_ms": "derived",  # avg_per_prompt_ms is per-prompt e2e
        "p50_e2el_ms": "missing",
        "p90_e2el_ms": "missing",
        "p99_e2el_ms": "missing",
        "request_throughput": "real",
        "total_token_throughput": "missing",
        "duration_s": "derived",  # total_ms/1000
        "input_token_throughput": "missing",
        "output_token_throughput": "missing",
        "mean_itl_ms": "missing",
        "p50_itl_ms": "missing",
        "p90_itl_ms": "missing",
        "p99_itl_ms": "missing",
        "run_id": "missing",
        "tensor_parallelism": "missing",
    }


def _odyn_available_signals(is_chat: bool) -> list[str]:
    if is_chat:
        return [
            "ttft", "tpot", "e2e_latency", "throughput",
            "concurrency", "engine", "gpu_type", "model_id",
            "failure_label",
        ]
    return [
        "e2e_latency", "throughput", "batch_size", "engine",
        "gpu_type", "model_id",
    ]


def _odyn_missing_signals(is_chat: bool) -> list[str]:
    if is_chat:
        return [
            "itl", "input_tokens", "output_tokens", "kv_cache_size",
            "memory_pressure", "timeout_label",
        ]
    return [
        "ttft", "tpot", "itl", "concurrency", "input_tokens",
        "output_tokens", "kv_cache_size", "memory_pressure",
        "timeout_label", "failure_label",
    ]


def _odyn_limitations(cfg: str, meta: dict) -> list[str]:
    return [
        "Odyn Network public benchmark (vLLM + Ray Serve based, Apache-2.0).",
        f"Single model × single hardware target ({meta['model']} on {meta['gpu']}); "
        "do NOT generalise these latency numbers to other deployments.",
        "TTFT / TPOT / e2e are reported as avg + p95 only — no p50 / p90 / p99.",
        "Failures (`failed`) reported at the highest concurrency levels — these "
        "encode timeout/SLA backpressure; treat as failure-rate prior, not as a "
        "real timeout label.",
        "Prompt distributions A/B/C/D are partly drawn from ShareGPT (rows 251-500 "
        "of each profile per the README); the first 250 are original Odyn traffic.",
        "Benchmark-only — not production / pilot telemetry. Tier 4.",
    ]


# ---------------------------------------------------------------------------
# Memoriant ingest
# ---------------------------------------------------------------------------


def _ingest_memoriant(safe_root: Path,
                      ledger: Optional[OperatorPolicyLedger] = None
                      ) -> list[dict]:
    """memoriant/dgx-spark-kv-cache-benchmark → 1 config:

    * ``v3_corrected`` — corrected KV cache quantization benchmark on NVIDIA
      DGX Spark GB10 (Grace Blackwell unified memory, llama.cpp).
    """
    if ledger is None:
        ledger = _load_ledger()
    dataset_id = MEMORIANT_DATASET_ID
    license = MEMORIANT_LICENSE_TAG
    gate_decision = evaluate_redistribution(
        ledger=ledger,
        dataset_id=dataset_id,
        license_tag=license,
    )
    base = HF_DIR / "memoriant__dgx-spark-kv-cache-benchmark"
    base_raw = base / "raw"
    base_raw.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    out: list[dict] = []

    remote = "data/benchmark_results_v3_complete.csv"
    local = base_raw / "benchmark_results_v3_complete.csv"
    manifest = _download(_hf_url(dataset_id, remote), local,
                         max_bytes=64 * 1024)
    if manifest["error"]:
        logger.error("memoriant: %s", manifest["error"])
        return out
    with open(local, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out

    normalized_rows = []
    for r in rows:
        try:
            normalized_rows.append({
                "source_dataset_id": dataset_id,
                "trace_type": "latency_benchmark_trace",
                "provenance": (
                    f"{dataset_id}@v3_corrected#{remote}#git={git_sha[:7]}"
                ),
                "model": "llama.cpp/llama-2-7b",  # README says llama.cpp 7B reference
                "model_family": "llama",
                "gpu": "DGX Spark GB10 (Grace Blackwell)",
                "engine": "llama.cpp",
                "context_tokens": int(r["context_tokens"]),
                "cache_type": r["cache_type"],
                "kv_buffer_mib": int(r["kv_buffer_mib"]),
                "gpu_mem_mib": int(r["gpu_mem_mib"]),
                "prompt_tps": float(r["prompt_tps"]) if r.get("prompt_tps") else None,
                "gen_tps": float(r["gen_tps"]) if r.get("gen_tps") else None,
                "notes": r.get("notes") or "",
            })
        except (KeyError, ValueError) as e:
            logger.warning("memoriant: skipped malformed row: %s", e)
            continue

    out.append(_finalize_config(
        dataset_id=dataset_id,
        safe_name="memoriant__dgx-spark-kv-cache-benchmark",
        config="v3_corrected",
        source_file_relative=remote,
        raw_local_path=local,
        raw_columns=list(rows[0].keys()),
        normalized_rows=normalized_rows,
        raw_rows=rows,
        license=license,
        license_source=MEMORIANT_LICENSE_SOURCE,
        gate_decision=gate_decision,
        gated=False,
        field_quality={
            "model": "real",
            "model_family": "real",
            "gpu": "real",
            "engine": "real",
            "profile": "missing",
            "concurrency": "missing",
            "num_requests": "missing",
            "mean_ttft_ms": "missing",
            "p50_ttft_ms": "missing",
            "p90_ttft_ms": "missing",
            "p99_ttft_ms": "missing",
            "mean_tpot_ms": "missing",
            "p50_tpot_ms": "missing",
            "p90_tpot_ms": "missing",
            "p99_tpot_ms": "missing",
            "mean_e2el_ms": "missing",
            "p50_e2el_ms": "missing",
            "p90_e2el_ms": "missing",
            "p99_e2el_ms": "missing",
            "request_throughput": "missing",
            "total_token_throughput": "real",  # gen_tps = generation tokens/sec
            "input_token_throughput": "real",  # prompt_tps
            "output_token_throughput": "missing",
            "mean_itl_ms": "missing",
            "p50_itl_ms": "missing",
            "p90_itl_ms": "missing",
            "p99_itl_ms": "missing",
            "run_id": "missing",
            "tensor_parallelism": "missing",
            "duration_s": "missing",
        },
        available_signals=[
            "throughput", "kv_cache_size", "memory_pressure",
            "gpu_type", "model_id", "engine",
        ],
        missing_signals=[
            "ttft", "tpot", "itl", "e2e_latency", "concurrency",
            "batch_size", "failure_label", "timeout_label",
        ],
        limitations=[
            "Tiny corrected benchmark (18 rows total) — KV cache quantization "
            "ablation on a single GPU (DGX Spark GB10).",
            "v3 fixes the v1 memory-measurement error (RSS was wrong on unified "
            "memory; v3 uses nvidia-smi + llama.cpp internals per the CORRECTION "
            "notice).",
            "prompt_tps + gen_tps are real measured tokens/sec, but no TTFT / "
            "TPOT / e2e is reported — generation latency must be derived from "
            "gen_tps + output_token_count.",
            "Single model context-length sweep (0 → 110k tokens) under 3 cache "
            "quantizations (f16 / q8_0 / q4_0); not a model / GPU / engine sweep.",
            "Useful for KV-cache memory-pressure priors and the cache-quantization "
            "throughput trade-off; NOT a latency frontier source on its own.",
            "Benchmark-only — Tier 4. Author: Nathan Maine / Memoriant Inc.",
        ],
        stratification_keys=["cache_type", "context_tokens"],
        git_sha=git_sha,
    ))
    return out


# ---------------------------------------------------------------------------
# Intellistream vLLM-HUST leaderboard ingest
# ---------------------------------------------------------------------------


def _ingest_intellistream(safe_root: Path,
                          ledger: Optional[OperatorPolicyLedger] = None
                          ) -> list[dict]:
    """intellistream/vllm-hust-benchmark-results → 2 configs:

    * ``single_gpu`` — single-GPU leaderboard (≈ 200 KiB JSON).
    * ``multi_gpu`` — multi-GPU leaderboard (≈ 10 KiB JSON).

    Each entry records measured TTFT_ms, TBT_ms (time-between-tokens =
    TPOT), throughput_tps, peak_mem_mb, error_rate across Huawei 910B3
    hardware running vLLM 0.11.0 / vLLM-HUST forks.
    """
    if ledger is None:
        ledger = _load_ledger()
    dataset_id = INTELLISTREAM_DATASET_ID
    license = INTELLISTREAM_LICENSE_TAG
    gate_decision = evaluate_redistribution(
        ledger=ledger,
        dataset_id=dataset_id,
        license_tag=license,
    )
    base = HF_DIR / "intellistream__vllm-hust-benchmark-results"
    base_raw = base / "raw"
    base_raw.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    out: list[dict] = []

    files = [
        ("single_gpu", "leaderboard_single.json"),
        ("multi_gpu", "leaderboard_multi.json"),
    ]

    for cfg, remote in files:
        local = base_raw / remote
        manifest = _download(_hf_url(dataset_id, remote), local,
                             max_bytes=512 * 1024)
        if manifest["error"]:
            logger.warning("intellistream: %s — %s", remote, manifest["error"])
            continue
        with open(local) as fh:
            entries = json.load(fh)
        if not isinstance(entries, list) or not entries:
            continue

        normalized_rows = []
        for e in entries:
            try:
                metrics = e.get("metrics") or {}
                hw = e.get("hardware") or {}
                model = e.get("model") or {}
                wl = e.get("workload") or {}
                normalized_rows.append({
                    "source_dataset_id": dataset_id,
                    "trace_type": "latency_benchmark_trace",
                    "provenance": (
                        f"{dataset_id}@{cfg}#{remote}#"
                        f"entry={e.get('entry_id','')[:8]}#git={git_sha[:7]}"
                    ),
                    "entry_id": e.get("entry_id"),
                    "engine": e.get("engine"),
                    "engine_version": e.get("engine_version"),
                    "config_type": e.get("config_type"),
                    "model": model.get("name") or model.get("canonical_id"),
                    "model_family": model.get("short_name"),
                    "model_parameters": model.get("parameters"),
                    "model_precision": model.get("precision"),
                    "model_quantization": model.get("quantization"),
                    "gpu": (
                        f"{hw.get('vendor','')} {hw.get('chip_model','')}".strip()
                    ),
                    "gpu_count": hw.get("chip_count"),
                    "gpu_memory_per_chip_gb": hw.get("memory_per_chip_gb"),
                    "gpu_total_memory_gb": hw.get("total_memory_gb"),
                    "workload_name": wl.get("name"),
                    "workload_dataset": wl.get("dataset"),
                    "input_length": wl.get("input_length"),
                    "output_length": wl.get("output_length"),
                    "batch_size": wl.get("batch_size"),
                    "concurrent_requests": wl.get("concurrent_requests"),
                    "mean_ttft_ms": metrics.get("ttft_ms"),
                    "mean_tpot_ms": metrics.get("tbt_ms"),  # tbt == TPOT
                    "throughput_tps": metrics.get("throughput_tps"),
                    "peak_mem_mb": metrics.get("peak_mem_mb"),
                    "error_rate": metrics.get("error_rate"),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("intellistream: skipped entry: %s", exc)
                continue

        raw_top_keys = sorted(set().union(*(e.keys() for e in entries)))

        out.append(_finalize_config(
            dataset_id=dataset_id,
            safe_name="intellistream__vllm-hust-benchmark-results",
            config=cfg,
            source_file_relative=remote,
            raw_local_path=local,
            raw_columns=raw_top_keys,
            normalized_rows=normalized_rows,
            raw_rows=entries,
            license=license,
            license_source=INTELLISTREAM_LICENSE_SOURCE,
            gate_decision=gate_decision,
            gated=False,
            field_quality={
                "model": "real",
                "model_family": "real",
                "gpu": "real",
                "engine": "real",
                "profile": "missing",
                "concurrency": "real",  # concurrent_requests when set
                "num_requests": "missing",
                "mean_ttft_ms": "real",
                "p50_ttft_ms": "missing",
                "p90_ttft_ms": "missing",
                "p99_ttft_ms": "missing",
                "mean_tpot_ms": "real",  # tbt_ms
                "p50_tpot_ms": "missing",
                "p90_tpot_ms": "missing",
                "p99_tpot_ms": "missing",
                "mean_e2el_ms": "missing",
                "p50_e2el_ms": "missing",
                "p90_e2el_ms": "missing",
                "p99_e2el_ms": "missing",
                "request_throughput": "missing",
                "total_token_throughput": "real",
                "input_token_throughput": "missing",
                "output_token_throughput": "missing",
                "mean_itl_ms": "missing",
                "p50_itl_ms": "missing",
                "p90_itl_ms": "missing",
                "p99_itl_ms": "missing",
                "run_id": "real",  # entry_id
                "tensor_parallelism": "missing",
                "duration_s": "missing",
            },
            available_signals=[
                "ttft", "tpot", "throughput", "concurrency",
                "batch_size", "input_tokens", "output_tokens",
                "gpu_type", "model_id", "engine", "engine_version",
                "memory_pressure", "failure_label",
            ],
            missing_signals=[
                "itl", "e2e_latency", "kv_cache_size",
                "timeout_label",
            ],
            limitations=[
                "vLLM-HUST benchmark leaderboard (intellistream). Submissions-driven "
                "leaderboard with per-entry hardware × model × workload × engine "
                "× engine_version metadata.",
                "Single point measurement per entry — TTFT_ms / TBT_ms (=TPOT) are "
                "scalar means with NO p50/p90/p95/p99 breakdown.",
                "Huawei 910B3 (Ascend-class) dominates the leaderboard — generalises "
                "POORLY to NVIDIA / AMD / TPU. Treat as Ascend-specific prior.",
                "peak_mem_mb is 0 for many entries (not measured upstream).",
                "error_rate is reported but always 0 for the present snapshot — "
                "treat as upper-bound only.",
                "NO declared license on the HF card frontmatter — recorded "
                "license=None. Bounded normalised sample is NOT committed for "
                "this dataset (license_redistribution_status=unspecified).",
                "Benchmark-only — Tier 4. Not pilot telemetry.",
            ],
            stratification_keys=[
                "engine", "model_family", "workload_name",
            ],
            git_sha=git_sha,
        ))
    return out


# ---------------------------------------------------------------------------
# Finalize one config — write all artefacts, run promotion gates
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    vs = sorted(v for v in values if v is not None)
    if not vs:
        return {}
    n = len(vs)

    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round((n - 1) * p))))
        return vs[idx]

    return {
        "count": n,
        "min": vs[0],
        "p50": _q(0.50),
        "p90": _q(0.90),
        "p95": _q(0.95),
        "p99": _q(0.99),
        "max": vs[-1],
        "mean": sum(vs) / n,
    }


def _statistical_rollups(rows: list[dict],
                         stratification_keys: list[str]) -> dict:
    """Compute strong-enough rollups for the latency benchmark configs."""
    overall: dict[str, dict] = {}
    for k in [
        "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms",
        "mean_e2el_ms", "p95_e2el_ms", "throughput_tok_s",
        "throughput_req_s", "throughput_tps", "throughput_prompts_s",
        "prompt_tps", "gen_tps", "peak_mem_mb", "error_rate",
        "kv_buffer_mib", "gpu_mem_mib",
    ]:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if vals:
            overall[k] = _percentiles(vals)

    by_strata: dict[str, dict] = {}
    if stratification_keys:
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            key = tuple(r.get(k) for k in stratification_keys)
            groups.setdefault(key, []).append(r)
        subgroup_counts = {}
        for grp_key, grp_rows in groups.items():
            label = "|".join(
                f"{stratification_keys[i]}={grp_key[i]}"
                for i in range(len(grp_key))
            )
            subgroup_counts[label] = len(grp_rows)
            # Only compute rollups for groups large enough to be meaningful.
            if len(grp_rows) >= 5:
                grp_stats: dict[str, dict] = {}
                for k in ["mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms",
                          "throughput_tok_s", "throughput_tps"]:
                    vals = [r.get(k) for r in grp_rows if r.get(k) is not None]
                    if vals:
                        grp_stats[k] = _percentiles(vals)
                if grp_stats:
                    by_strata[label] = grp_stats
        return {
            "overall": overall, "by_strata": by_strata,
            "subgroup_counts": subgroup_counts,
        }
    return {"overall": overall, "by_strata": {}, "subgroup_counts": {}}


def _sample_strength(rows: int, has_strata_coverage: bool) -> str:
    if rows == 0:
        return "fixture_only"
    if rows < 25:
        return "weak" if not has_strata_coverage else "moderate"
    if rows < 200:
        return "moderate"
    return "strong"


def _finalize_config(*, dataset_id: str, safe_name: str, config: str,
                     source_file_relative: str, raw_local_path: Path,
                     raw_columns: list[str], normalized_rows: list[dict],
                     license: Optional[str],
                     license_source: str,
                     gate_decision: RedistributionGateDecision,
                     gated: bool,
                     field_quality: dict, available_signals: list[str],
                     missing_signals: list[str], limitations: list[str],
                     stratification_keys: list[str], git_sha: str,
                     raw_rows: Optional[list[dict]] = None) -> dict:
    proc_dir = HF_DIR / safe_name / config / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = proc_dir / "analysis_sample.jsonl"
    analysis_bytes = io.BytesIO()
    for row in normalized_rows:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        analysis_bytes.write(line)
    analysis_path.write_bytes(analysis_bytes.getvalue())
    analysis_sha = _sha256(analysis_path)
    analysis_size = analysis_path.stat().st_size

    fixture_path = FIXTURES_DIR / f"{safe_name}__{config}_sample.jsonl"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_rows: list[dict] = []
    fixture_bytes = io.BytesIO()
    for row in normalized_rows:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if fixture_bytes.tell() + len(line) > FIXTURE_MAX_BYTES:
            break
        if len(fixture_rows) >= 5:
            break
        fixture_bytes.write(line)
        fixture_rows.append(row)
    fixture_path.write_bytes(fixture_bytes.getvalue())
    fixture_sha = _sha256(fixture_path)
    fixture_size = fixture_path.stat().st_size

    # The canonical redistribution gate (not this script) decides whether
    # the committed normalised sample may be redistributed. Under the
    # default ``deny_all`` / zero-grants ledger the verdicts match the
    # v1 behaviour byte-for-byte: ``apache-2.0`` (odyn + memoriant)
    # permits → sample committed; ``None`` (intellistream) denies →
    # sample skipped. The pre-existing
    # ``"license_unspecified_no_redistribution_promise"`` reason string
    # is preserved verbatim on ``committed_normalized_sample_reason_skipped``
    # so downstream tests stay green; the canonical gate verdict is
    # recorded additively in the new ``redistribution_gate_*`` fields
    # below.
    committed_norm_path = None
    committed_norm_rows = 0
    committed_norm_bytes = 0
    committed_norm_sha = None
    committed_norm_reason_skipped: Optional[str] = None
    if gate_decision.permitted:
        committed_path = proc_dir / "committed_normalized_sample.jsonl"
        buf = io.BytesIO()
        for row in normalized_rows:
            line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            if buf.tell() + len(line) > COMMITTED_NORMALIZED_MAX_BYTES:
                break
            buf.write(line)
            committed_norm_rows += 1
        committed_path.write_bytes(buf.getvalue())
        committed_norm_path = str(committed_path.relative_to(REPO_ROOT))
        committed_norm_bytes = committed_path.stat().st_size
        committed_norm_sha = _sha256(committed_path)
    else:
        committed_norm_reason_skipped = COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON

    # Schema profile — dtypes / examples are computed from RAW rows (when
    # provided) so that columns dropped during normalisation (e.g. raw
    # ``e2e_avg`` is mapped to normalised ``mean_e2el_ms``) still have a
    # populated example list.
    dtypes: dict[str, list[str]] = {}
    example_values: dict[str, list[str]] = {}
    probe_rows = raw_rows if raw_rows is not None else normalized_rows
    for col in raw_columns:
        seen_dtypes: set[str] = set()
        examples: list[str] = []
        for r in probe_rows[:20]:
            v = r.get(col)
            if v is None or v == "":
                continue
            seen_dtypes.add(type(v).__name__)
            if len(examples) < 3:
                examples.append(repr(v)[:120])
        if not seen_dtypes:
            seen_dtypes.add("str")
        dtypes[col] = sorted(seen_dtypes)
        example_values[col] = examples

    schema_profile = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_file": source_file_relative,
        "inspected_row_count": len(normalized_rows),
        "raw_columns": raw_columns,
        "nested_keys": [],
        "dtypes": dtypes,
        "example_values": example_values,
        "missing_rates": {},
        "unknown_columns": [],
        "rejected_columns": [],
        "accepted_columns": raw_columns,
        "list_length_summaries": {},
        "file_size_bytes": raw_local_path.stat().st_size if raw_local_path.exists() else 0,
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # Schema mapping
    columns_map = _build_columns_map(raw_columns, normalized_rows, field_quality)
    schema_mapping = {
        "dataset_id": dataset_id,
        "config_name": config,
        "accepted_columns": raw_columns,
        "rejected_columns": [],
        "unknown_columns": [],
        "columns": columns_map,
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # Statistical rollups
    rollups = _statistical_rollups(normalized_rows, stratification_keys)
    rollups_path = proc_dir / "statistical_rollups.json"
    rollups_path.write_text(json.dumps(rollups, indent=2, sort_keys=True))

    has_strata = bool(rollups["by_strata"])
    strength = _sample_strength(len(normalized_rows), has_strata)

    normalized_schema = sorted({
        k for r in normalized_rows for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    summary = {
        "dataset_id": dataset_id,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": license,
        "license_redistribution_status": gate_decision.license_status,
        "license_redistribution_source": license_source,
        "redistribution_gate_reason_code": gate_decision.reason_code,
        "redistribution_gate_reason_detail": gate_decision.reason_detail,
        "redistribution_gate_permitted": gate_decision.permitted,
        "redistribution_gate_operator_grant_dataset_id": (
            gate_decision.operator_grant_dataset_id
        ),
        "redistribution_gate_scope": GATE_SCOPE,
        "gated": gated,
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
        "stratification_keys": stratification_keys,
        "sampling_method": "full_bounded",
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "analysis_sample_rows": len(normalized_rows),
        "analysis_sample_bytes": analysis_size,
        "analysis_sample_sha256": analysis_sha,
        "analysis_sample_path": str(analysis_path.relative_to(REPO_ROOT)),
        "committed_normalized_sample_rows": committed_norm_rows,
        "committed_normalized_sample_bytes": committed_norm_bytes,
        "committed_normalized_sample_path": committed_norm_path,
        "committed_normalized_sample_sha256": committed_norm_sha,
        "committed_normalized_sample_reason_skipped": committed_norm_reason_skipped,
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_size,
        "sample_sha256": fixture_sha,
        "subgroup_counts": rollups["subgroup_counts"],
        "statistical_sample_strength": strength,
        "schema_profile_path": str(
            (proc_dir / "schema_profile.json").relative_to(REPO_ROOT)
        ),
        "schema_mapping_path": str(
            (proc_dir / "schema_mapping.json").relative_to(REPO_ROOT)
        ),
        "statistical_rollups_path": str(rollups_path.relative_to(REPO_ROOT)),
        "summary_path_relative": str(
            (proc_dir / "summary.json").relative_to(REPO_ROOT)
        ),
        "provenance": f"{dataset_id}@{config}#{source_file_relative}#git={git_sha[:7]}",
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": {
            "url": _hf_url(dataset_id, source_file_relative),
            "dest": str(raw_local_path),
            "downloaded_bytes": (
                raw_local_path.stat().st_size if raw_local_path.exists() else 0
            ),
        },
    }
    (proc_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    decision = promotion.evaluate_promotion(summary)
    entry = promotion.build_registry_entry(summary, decision)
    return {"summary": summary, "decision": decision, "registry_entry": entry}


def _build_columns_map(raw_columns: list[str], rows: list[dict],
                       field_quality_norm: dict) -> list[dict]:
    """Build the schema_mapping ``columns`` list for the raw columns."""
    # Map each raw column to the most likely normalized field. The Aurelius
    # signal category drives the evaluation harness routing.
    column_to_norm = {
        # Odyn chat
        "profile": ("profile", "real", "metadata_only", []),
        "mode": ("mode", "real", "metadata_only", []),
        "concurrency": ("concurrency", "real", "request_arrival",
                        ["constraint_aware_backtest", "latency_prior"]),
        "total_requests": ("num_requests", "real", "request_arrival",
                           ["latency_prior", "throughput_prior"]),
        "successful": ("num_successful", "real", "request_completion",
                       ["latency_prior"]),
        "failed": ("num_failed", "real", "failure_timeout",
                   ["latency_prior"]),
        "wall_time_s": ("duration_s", "real", "request_completion",
                        ["throughput_prior"]),
        "e2e_avg": ("mean_e2el_ms", "real", "latency", ["latency_prior"]),
        "e2e_p95": ("p95_e2el_ms", "real", "latency", ["latency_prior"]),
        "tpot_avg": ("mean_tpot_ms", "real", "latency", ["latency_prior"]),
        "tpot_p95": ("p95_tpot_ms", "real", "latency", ["latency_prior"]),
        "ttft_avg": ("mean_ttft_ms", "real", "latency", ["latency_prior"]),
        "ttft_p95": ("p95_ttft_ms", "real", "latency", ["latency_prior"]),
        "throughput_tok_s": ("total_token_throughput", "real", "throughput",
                             ["throughput_prior"]),
        "throughput_req_s": ("request_throughput", "real", "throughput",
                             ["throughput_prior"]),
        # Odyn batch
        "batch_size": ("batch_size", "real", "request_arrival",
                       ["latency_prior", "constraint_aware_backtest"]),
        "num_prompts": ("num_requests", "real", "request_arrival",
                        ["throughput_prior"]),
        "num_results": ("num_completed", "real", "request_completion",
                        ["throughput_prior"]),
        "submit_ms": ("submit_ms", "real", "request_arrival",
                      ["latency_prior"]),
        "total_ms": ("total_ms", "real", "latency", ["latency_prior"]),
        "avg_per_prompt_ms": ("mean_e2el_ms", "derived", "latency",
                              ["latency_prior"]),
        "throughput_prompts_s": ("request_throughput", "real", "throughput",
                                 ["throughput_prior"]),
        # Memoriant
        "context_tokens": ("context_tokens", "real", "tokens",
                           ["latency_prior", "cache_residency_evaluation"]),
        "cache_type": ("cache_type", "real", "cache_residency",
                       ["cache_residency_evaluation"]),
        "kv_buffer_mib": ("kv_buffer_mib", "real", "memory",
                          ["cache_residency_evaluation"]),
        "gpu_mem_mib": ("gpu_mem_mib", "real", "memory",
                        ["cache_residency_evaluation"]),
        "prompt_tps": ("input_token_throughput", "real", "throughput",
                       ["throughput_prior"]),
        "gen_tps": ("total_token_throughput", "real", "throughput",
                    ["throughput_prior"]),
        "notes": ("notes", "real", "metadata_only", []),
        # Intellistream
        "entry_id": ("run_id", "real", "metadata_only", []),
        "engine": ("engine", "real", "metadata_only",
                   ["latency_prior", "throughput_prior"]),
        "engine_version": ("engine_version", "real", "metadata_only", []),
        "config_type": ("config_type", "real", "metadata_only", []),
        "hardware": ("gpu_spec_json", "real", "gpu_resource",
                     ["latency_prior"]),
        "model": ("model_spec_json", "real", "metadata_only",
                  ["latency_prior"]),
        "workload": ("workload_spec_json", "real", "metadata_only",
                     ["latency_prior"]),
        "metrics": ("metrics_json", "real", "latency",
                    ["latency_prior", "throughput_prior"]),
        "constraints": ("constraints_json", "real", "metadata_only", []),
    }

    columns = []
    for col in raw_columns:
        norm = column_to_norm.get(col)
        if norm is None:
            # Default: treat as real metadata
            columns.append({
                "raw_column_name": col,
                "normalized_field": col,
                "dtypes": ["unknown"],
                "field_quality": "real",
                "aurelius_signal_category": "metadata_only",
                "usable_for": [],
                "presence_rate": 1.0,
                "missing_rate": 0.0,
                "notes": f"raw {col} (no explicit mapping)",
            })
            continue
        nf, fq, sig, usable = norm
        # Determine missing rate from the normalized rows
        miss = 0
        for r in rows[:100]:
            if r.get(nf) is None and r.get(col) is None:
                miss += 1
        denom = min(100, len(rows)) or 1
        columns.append({
            "raw_column_name": col,
            "normalized_field": nf,
            "dtypes": ["str", "int", "float", "dict", "NoneType"],
            "field_quality": fq,
            "aurelius_signal_category": sig,
            "usable_for": usable,
            "presence_rate": 1.0 - miss / denom,
            "missing_rate": miss / denom,
            "notes": f"raw {col} → {nf}",
        })
    return columns


# ---------------------------------------------------------------------------
# Audit summary (discovery + rejection records)
# ---------------------------------------------------------------------------


REJECTED_OR_BLOCKED = [
    {
        "dataset_id": "tarekmasryo/llm-system-ops-production-telemetry-sft-data",
        "candidate_trace_type": "telemetry_trace",
        "license_observed": "cc-by-4.0",
        "gated": False,
        "kind": "reject_low_value",
        "reason": (
            "Self-declared SYNTHETIC dataset (README: 'Synthetic data (safe "
            "for teaching, prototyping, and portfolio notebooks). Not real "
            "user data. cost_usd and token fields are synthetic estimates "
            "(not billing truth).'). Tagged as 'production-telemetry' but "
            "is in fact tabular synthetic data for SFT — not Aurelius-grade "
            "Tier-2 telemetry. Rejected to enforce the anti-dataset-spam "
            "rule and avoid pretending benchmark/synthetic data is "
            "production telemetry."
        ),
    },
    {
        "dataset_id": "spiritbuun/turboquant-tcq-kv-cache",
        "candidate_trace_type": "kernel_profile_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "reject_not_a_dataset",
        "reason": (
            "Repository contains quantization codebooks (.bin / .pt artefacts), "
            "not a benchmark dataset. No measured latency / throughput / cache "
            "telemetry. Not ingestible into the federated benchmark corpus."
        ),
    },
    {
        "dataset_id": "hlarcher/inference-benchmarker",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Hosts ShareGPT-derived prompt fixtures used to drive the "
            "huggingface/inference-benchmarker tool. The fixtures are "
            "workload-shape only (no measured TTFT / TPOT / queue / GPU), "
            "and the existing `aurelius/traces/sharegpt_aiperf.py` ingester "
            "already covers the ShareGPT request-shape role. Rejected as "
            "duplicate-shape, low information density."
        ),
    },
    {
        "dataset_id": "Boxoffice1280/Neurips2026_evaluating_accuracy_KV-cache_reuse_techniques",
        "candidate_trace_type": "cache_residency_trace",
        "license_observed": "cc-by-nc-nd-4.0",
        "gated": False,
        "kind": "license_restricted_no_redistribution",
        "reason": (
            "License is cc-by-nc-nd-4.0 — Non-Commercial + No-Derivatives. "
            "Aurelius normalised samples are derivatives, so committing any "
            "normalised excerpt would violate the No-Derivatives clause. "
            "Marked license_restricted; HF metadata reference retained but "
            "no normalised sample ingested."
        ),
    },
    {
        "dataset_id": "Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days-SVLSGPU_EvalData",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Combines BurstGPT + LMSYSChat prompt traces — the BurstGPT "
            "shape role is already covered by `lzzmm/BurstGPT/burstgpt_1_full` "
            "(promoted_for_training_priors); LMSYSChat is request-shape only. "
            "No declared license (license=None) — duplicate of an existing "
            "ingestion path."
        ),
    },
    {
        "dataset_id": "MCP-1st-Birthday/smoltrace-cloud-cost-tasks",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_synthetic_agent_eval",
        "reason": (
            "Synthetic MCP agent-evaluation task set (smoltrace). No measured "
            "infrastructure signals (latency / queue / GPU / cache). The "
            "'cloud-cost-tasks' name suggests cost intent but the content is "
            "agent-tool-use tasks, not cost telemetry. Tier 6 — rejected to "
            "preserve information density."
        ),
    },
    {
        "dataset_id": "rbgo/llm-inference-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "license_unspecified_low_priority",
        "reason": (
            "Single CSV with inference benchmark numbers but no declared "
            "license. Without license clarity, committing a normalised "
            "sample is unsafe. Deferred — revisit if licence clarified "
            "upstream. Lower priority than odyn-network / memoriant "
            "(both Apache-2.0) which fill the same Tier-4 role."
        ),
    },
    {
        "dataset_id": "project-vajra/dev-staging-h100-dgx",
        "candidate_trace_type": "kernel_profile_trace",
        "license_observed": None,
        "gated": False,
        "kind": "license_unspecified_low_priority",
        "reason": (
            "NCCL all_reduce / send_recv CSV traces (compressed .xz). "
            "Potentially useful as inter-GPU communication priors, but no "
            "declared license. Deferred — revisit if licence clarified or "
            "if Aurelius adds a multi-GPU placement / collective evaluator."
        ),
    },
]


def _write_audit_summary(
    ingested: list[dict],
    *,
    ledger: Optional[OperatorPolicyLedger] = None,
) -> Path:
    """Write/merge the broadened-discovery audit summary.

    The audit file is a running log shared with later rounds (e.g. the
    optimum-benchmark round-2 ingest also appends here). We MERGE
    rather than overwrite: previous ``ingested`` entries from other
    scripts (keyed on ``(dataset_id, config_name)``) are preserved
    byte-for-byte; this script's seven latency-benchmark entries are
    written through with the new ``redistribution_gate_*`` fields the
    sixth gate consumer attaches.

    ``doc_version`` advances to ``broadened_discovery_audit_summary_v2``
    when this script runs, signalling to downstream consumers that
    every latency-benchmark ingested row now carries the canonical
    gate's verdict. Optimum-benchmark entries (and any future
    pre-existing entries) remain at their original shape until those
    scripts are wired through the gate themselves in follow-up PRs.
    """

    if ledger is None:
        ledger = _load_ledger()

    out_path = DISC_DIR / "broadened_discovery_audit_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_payload = None
    if out_path.exists():
        try:
            prev_payload = json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001
            prev_payload = None

    prev_ingested = (prev_payload or {}).get("ingested") or []
    prev_disc_only = (prev_payload or {}).get("discovery_only_records") or []
    prev_failed = (prev_payload or {}).get("failed") or []

    new_ingested = [
        {
            "dataset_id": x["summary"]["dataset_id"],
            "config_name": x["summary"]["config_name"],
            "canonical_trace_type": x["summary"]["canonical_trace_type"],
            "license": x["summary"]["license"],
            "license_redistribution_status": x["summary"][
                "license_redistribution_status"],
            "redistribution_gate_reason_code": x["summary"][
                "redistribution_gate_reason_code"],
            "redistribution_gate_permitted": x["summary"][
                "redistribution_gate_permitted"],
            "redistribution_gate_operator_grant_dataset_id": x["summary"][
                "redistribution_gate_operator_grant_dataset_id"],
            "gated": x["summary"]["gated"],
            "analysis_sample_rows": x["summary"]["analysis_sample_rows"],
            "fixture_sample_rows": x["summary"]["fixture_sample_rows"],
            "committed_normalized_sample_rows": x["summary"][
                "committed_normalized_sample_rows"],
            "committed_normalized_sample_bytes": x["summary"][
                "committed_normalized_sample_bytes"],
            "available_signals": x["summary"]["available_signals"],
            "missing_signals": x["summary"]["missing_signals"],
            "limitations": x["summary"]["limitations"],
            "statistical_sample_strength": x["summary"][
                "statistical_sample_strength"],
            "promotion_state": x["decision"]["state"],
            "promotion_tags": x["decision"]["promotion_tags"],
            "promotion_reasons": x["decision"]["reasons"],
        }
        for x in ingested
    ]

    # Merge previous + new, deduplicating on (dataset_id, config_name).
    new_keys = {(e["dataset_id"], e["config_name"]) for e in new_ingested}
    merged_ingested = [
        e for e in prev_ingested
        if (e.get("dataset_id"), e.get("config_name")) not in new_keys
    ] + new_ingested

    # Discovery-only records: replace this script's authoritative set,
    # preserve any records owned by other scripts.
    rejected_ids = {r["dataset_id"] for r in REJECTED_OR_BLOCKED}
    merged_disc = [
        r for r in prev_disc_only if r.get("dataset_id") not in rejected_ids
    ] + REJECTED_OR_BLOCKED

    payload = {
        "doc_version": "broadened_discovery_audit_summary_v2",
        "scope": (
            "Broadened HF discovery follow-on to PR #133 — bounded ingest of "
            "3 new Tier-4 latency_benchmark_trace candidates "
            "(odyn-network/odyn-benchmarks, memoriant/dgx-spark-kv-cache-benchmark, "
            "intellistream/vllm-hust-benchmark-results) plus 8 rejection / "
            "deferral records from the same INGEST_LATER / MONITOR pool. "
            "v2 wires the per-dataset summary writer through the canonical "
            "redistribution gate (sixth consumer); optimum-benchmark rows "
            "remain at v1 shape until that script is gate-wired separately."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "redistribution_gate_scope": GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "ingested": merged_ingested,
        "discovery_only_records": merged_disc,
        "failed": prev_failed,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


# ---------------------------------------------------------------------------
# Registry merge
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["odyn", "memoriant",
                                            "intellistream", "all"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # Load the operator policy ledger once; thread it through every
    # ingest function and the audit summary writer so a single source
    # of truth records ``redistribution_gate_policy_default`` and
    # ``redistribution_gate_policy_grant_count``.
    ledger = _load_ledger()

    safe_root = HF_DIR
    ingested: list[dict] = []

    if args.only in ("odyn", "all"):
        ingested.extend(_ingest_odyn(safe_root, ledger=ledger))
    if args.only in ("memoriant", "all"):
        ingested.extend(_ingest_memoriant(safe_root, ledger=ledger))
    if args.only in ("intellistream", "all"):
        ingested.extend(_ingest_intellistream(safe_root, ledger=ledger))

    logger.info("Ingested %d configs", len(ingested))

    if args.dry_run:
        return 0

    new_entries = [x["registry_entry"] for x in ingested]
    if new_entries:
        path = _merge_into_registry(new_entries)
        logger.info("Updated registry at %s", path)

    summary_path = _write_audit_summary(ingested, ledger=ledger)
    logger.info("Wrote audit summary %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
