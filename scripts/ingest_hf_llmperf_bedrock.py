#!/usr/bin/env python3
"""Round-3 broadened HF discovery — bounded ingest of ``ssong1/llmperf-bedrock``.

This script extends the federated benchmark corpus with one Tier-4
``latency_benchmark_trace`` dataset that fills a gap none of the existing
HF-ingested datasets cover: **measured per-request TTFT / ITL / e2e-latency
against a closed managed-API LLM provider (AWS Bedrock)**.

Why it matters for Aurelius:

* The existing latency-benchmark datasets (AgentPerfBench, Odyn, Memoriant,
  Intellistream, optimum-benchmark) all measure GPU-direct serving — i.e.
  vLLM / Triton / Ray Serve on a known GPU.
* For Aurelius' constraint-aware routing + deferral decisions, the API
  /managed-provider latency surface is a distinct prior — it includes
  client→provider network + provider scheduling + closed-source batching.
* ``ssong1/llmperf-bedrock`` is Apache-2.0, produced by Ray's standard
  LLMPerf benchmark (``token_benchmark_ray.py``), and has 350 individual
  measured requests across 4 (input_tokens, output_tokens, concurrency,
  region) runs.

This is NOT pilot telemetry; it is a public closed-API benchmark and stays
strictly Tier-4 (latency_benchmark_trace).

Outputs (per config; 1 config: ``bedrock_claude_instant_v1``):

    data/external/hf/ssong1__llmperf-bedrock/raw/<file>              # gitignored
    data/external/hf/ssong1__llmperf-bedrock/bedrock_claude_instant_v1/processed/
        schema_profile.json               (committed)
        schema_mapping.json               (committed)
        summary.json                      (committed)
        statistical_rollups.json          (committed)
        committed_normalized_sample.jsonl (committed, all 350 rows, ~85 KiB)
        analysis_sample.jsonl             (gitignored)
    tests/fixtures/hf/ssong1__llmperf-bedrock__bedrock_claude_instant_v1_sample.jsonl
                                          (committed, ≤ 16 KiB, 5 rows)

It also writes a discovery audit summary at
``data/external/hf_discovery/round3_broadened_discovery_audit_summary.json``
that records the negative-result candidates (``DistServe/2025-05-06T14-…``
license unspecified, ``hlarcher/inference-benchmarker`` ShareGPT-shape only,
``deepanjalimishra99/datacenter-traces`` DynamoRIO microarchitecture only,
``intellistream/sage-control-plane-llm-workloads`` insufficient sample,
``Nathan-Maine/dgx-spark-kv-cache-benchmark`` duplicate of memoriant,
``ssong1/llmperf-bedrock`` — ingested).

NO production claims. NO scheduler / controller / robust-energy-engine changes.
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
COMMITTED_NORMALIZED_MAX_BYTES = 512 * 1024  # 512 KiB — full 350 rows fit
PER_DATASET_TIMEOUT_S = 30 * 60

DATASET_ID = "ssong1/llmperf-bedrock"
SAFE_NAME = "ssong1__llmperf-bedrock"
CONFIG = "bedrock_claude_instant_v1"
LICENSE = "apache-2.0"
GATED = False

# ── Redistribution-gate metadata ───────────────────────────────────────────
# Raw HF license tag + human-curated provenance for the canonical
# redistribution gate. The gate (not this script) classifies the tag
# into a ``permissive_*`` / ``unspecified_no_committed_sample`` /
# ``declared_non_permissive`` status code. Keeping the raw tag separate
# from any inline ``"license"`` value here means a future HF tag change
# is a one-line edit; the gate handles the rest.
#
# ssong1/llmperf-bedrock is the only dataset this script ingests; the
# eight Round-3 discovery-only records do not flow through the gate
# because no normalised sample is written for them. apache-2.0 is the
# canonical permissive HF tag — the gate classifies it as
# ``permissive_apache_2_0`` and permits redistribution.
LICENSE_TAG: Optional[str] = "apache-2.0"
LICENSE_SOURCE = (
    "HF card frontmatter license: apache-2.0 "
    "(ssong1 / Ray LLMPerf token_benchmark_ray.py output against "
    "AWS Bedrock anthropic.claude-instant-v1)"
)
GATE_SCOPE = "committed_normalized_sample"

logger = logging.getLogger("aurelius.hf_llmperf_bedrock_ingest")


# ── Redistribution gate — wire the canonical gate, do not classify here ────


def _load_ledger(policy_path: Path = POLICY_PATH) -> OperatorPolicyLedger:
    """Load the operator policy ledger from disk, or fall back to empty.

    The committed default file ships zero grants under
    ``policy_default=deny_all``; an absent file is identical in
    behaviour. We use ``empty()`` as the fallback so the script
    remains self-sufficient in a fresh checkout that may not yet have
    the committed JSON pulled — the gate still produces correct
    decisions (permit for ``apache-2.0`` ssong1/llmperf-bedrock)
    instead of crashing.
    """

    if policy_path.exists():
        return OperatorPolicyLedger.load(policy_path)
    return OperatorPolicyLedger.empty()


def evaluate_redistribution(
    *,
    ledger: OperatorPolicyLedger,
    license_tag: Optional[str] = LICENSE_TAG,
    dataset_id: str = DATASET_ID,
    scope: str = GATE_SCOPE,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Ask the canonical gate whether the bounded normalised sample of
    ssong1/llmperf-bedrock may be redistributed under the supplied
    license tag.

    Pure function — no I/O. Exposed so tests can drive the gate path
    without invoking the JSON download / normalisation pipeline. The
    defaults reflect the module-level constants this script ships;
    tests override them to verify the wiring (e.g. swap ``license_tag``
    to ``None`` and check that the gate denies).

    Under the default-empty ledger and the dataset's declared
    apache-2.0 license tag, this returns
    ``permitted=True``,
    ``license_status="permissive_apache_2_0"``,
    ``reason_code="permitted_declared_permissive_license"`` — the gate
    ledger is NOT consulted because the license is on the closed
    permissive allow-list.
    """

    return decide_redistribution(
        dataset_id=dataset_id,
        license_str=license_tag,
        scope=scope,
        ledger=ledger,
        now_iso=now_iso,
    )


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


RUNS = [
    {
        "run_id": "bedrock_claude_instant_v1_in1024_out1024_concur1",
        "summary_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_1024_0_1024_100_summary.json"
        ),
        "individual_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_1024_0_1024_100_individual_responses.json"
        ),
        "mean_input_tokens": 1024,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 1024,
        "stddev_output_tokens": 100,
        "num_concurrent_requests": 1,
        "region": "default",
        "prompt_kind": "ray_default",
    },
    {
        "run_id": "bedrock_claude_instant_v1_in1024stddev100_out1024_concur1",
        "summary_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_1024_100_1024_0_summary.json"
        ),
        "individual_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_1024_100_1024_0_individual_responses.json"
        ),
        "mean_input_tokens": 1024,
        "stddev_input_tokens": 100,
        "mean_output_tokens": 1024,
        "stddev_output_tokens": 0,
        "num_concurrent_requests": 1,
        "region": "default",
        "prompt_kind": "ray_default",
    },
    {
        "run_id": "bedrock_claude_instant_v1_in252_out1024_concur1_dm",
        "summary_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_252_0_1024_0_summary (dm prompt).json"
        ),
        "individual_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_252_0_1024_0_individual_responses (dm prompt).json"
        ),
        "mean_input_tokens": 252,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 1024,
        "stddev_output_tokens": 0,
        "num_concurrent_requests": 1,
        "region": "default",
        "prompt_kind": "dm_prompt",
    },
    {
        "run_id": "bedrock_claude_instant_v1_in252_out1024_concur1_dm_apne1",
        "summary_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_252_0_1024_0_summary_apne1 (dm prompt).json"
        ),
        "individual_file": (
            "raw_data/bedrock-anthropic-claude-instant-v1_252_0_1024_0_individual_responses_ap_ne1 (dm prompt).json"
        ),
        "mean_input_tokens": 252,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 1024,
        "stddev_output_tokens": 0,
        "num_concurrent_requests": 1,
        "region": "ap-northeast-1",
        "prompt_kind": "dm_prompt",
    },
]


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
    except urllib.error.URLError as e:
        return {
            "url": url, "dest": str(dest), "downloaded_bytes": 0,
            "cached": False, "truncated": None, "error": f"URL {e.reason}",
        }
    return {
        "url": url, "dest": str(dest), "downloaded_bytes": written,
        "cached": False, "truncated": truncated, "error": None,
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
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _percentiles(vals: list[float]) -> dict:
    s = sorted(float(v) for v in vals if v is not None)
    if not s:
        return {}
    n = len(s)

    def _q(p: float) -> float:
        if n == 1:
            return s[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "n": n,
        "mean": statistics.fmean(s),
        "min": s[0], "max": s[-1],
        "p25": _q(0.25), "p50": _q(0.50), "p75": _q(0.75),
        "p90": _q(0.90), "p95": _q(0.95), "p99": _q(0.99),
    }


def ingest(
    *,
    output_root: Path = HF_DIR,
    force: bool = False,
    ledger: OperatorPolicyLedger | None = None,
) -> dict:
    t0 = time.time()
    git_sha = _git_sha()
    if ledger is None:
        ledger = _load_ledger()
    gate_decision = evaluate_redistribution(
        ledger=ledger,
        license_tag=LICENSE_TAG,
        dataset_id=DATASET_ID,
    )

    base = output_root / SAFE_NAME
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = base / CONFIG / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_manifests = []
    normalized_rows: list[dict] = []
    per_run_summaries: list[dict] = []
    raw_schema_observed: set[str] = set()
    raw_summary_schema_observed: set[str] = set()

    for run in RUNS:
        logger.info("ingest run %s", run["run_id"])
        sum_local = raw_dir / Path(run["summary_file"]).name
        ind_local = raw_dir / Path(run["individual_file"]).name

        sm = _download(_hf_url(DATASET_ID, run["summary_file"]),
                       sum_local, max_bytes=128 * 1024)
        im = _download(_hf_url(DATASET_ID, run["individual_file"]),
                       ind_local, max_bytes=512 * 1024)
        raw_manifests.append({"file": run["summary_file"], **sm})
        raw_manifests.append({"file": run["individual_file"], **im})

        if sm["error"] or im["error"]:
            logger.warning("ssong1/llmperf-bedrock %s: %s / %s",
                           run["run_id"], sm["error"], im["error"])
            continue

        with open(sum_local) as fh:
            raw_summary = json.load(fh)
        with open(ind_local) as fh:
            raw_indv = json.load(fh)

        raw_summary_schema_observed.update(raw_summary.keys())
        if raw_indv:
            raw_schema_observed.update(raw_indv[0].keys())

        per_run_summaries.append({
            "run_id": run["run_id"],
            "model": raw_summary.get("model"),
            "provider": "aws_bedrock",
            "model_family": "anthropic_claude_instant_v1",
            "engine": "bedrock_managed_api",
            "client": "ray_llmperf",
            "client_script": "token_benchmark_ray.py",
            "client_origin": "on_premise_kubernetes_bastion",
            "region": run["region"],
            "prompt_kind": run["prompt_kind"],
            "mean_input_tokens": int(raw_summary["mean_input_tokens"]),
            "stddev_input_tokens": int(raw_summary["stddev_input_tokens"]),
            "mean_output_tokens": int(raw_summary["mean_output_tokens"]),
            "stddev_output_tokens": int(raw_summary["stddev_output_tokens"]),
            "num_concurrent_requests": int(
                raw_summary["num_concurrent_requests"]),
            "num_requests_started": int(
                raw_summary.get("results_num_requests_started", 0)),
            "num_completed_requests": int(
                raw_summary.get("results_num_completed_requests", 0)),
            "num_completed_requests_per_min": float(
                raw_summary.get("results_num_completed_requests_per_min", 0.0)),
            "error_rate": float(raw_summary.get("results_error_rate", 0.0)),
            "number_errors": int(raw_summary.get("results_number_errors", 0)),
            "ttft_s_p25": raw_summary.get("results_ttft_s_quantiles_p25"),
            "ttft_s_p50": raw_summary.get("results_ttft_s_quantiles_p50"),
            "ttft_s_p75": raw_summary.get("results_ttft_s_quantiles_p75"),
            "ttft_s_p90": raw_summary.get("results_ttft_s_quantiles_p90"),
            "ttft_s_p95": raw_summary.get("results_ttft_s_quantiles_p95"),
            "ttft_s_p99": raw_summary.get("results_ttft_s_quantiles_p99"),
            "ttft_s_mean": raw_summary.get("results_ttft_s_mean"),
            "ttft_s_min": raw_summary.get("results_ttft_s_min"),
            "ttft_s_max": raw_summary.get("results_ttft_s_max"),
            "ttft_s_stddev": raw_summary.get("results_ttft_s_stddev"),
            "itl_s_p25": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p25"),
            "itl_s_p50": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p50"),
            "itl_s_p75": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p75"),
            "itl_s_p90": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p90"),
            "itl_s_p95": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p95"),
            "itl_s_p99": raw_summary.get(
                "results_inter_token_latency_s_quantiles_p99"),
            "itl_s_mean": raw_summary.get(
                "results_inter_token_latency_s_mean"),
            "e2e_latency_s_p25": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p25"),
            "e2e_latency_s_p50": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p50"),
            "e2e_latency_s_p75": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p75"),
            "e2e_latency_s_p90": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p90"),
            "e2e_latency_s_p95": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p95"),
            "e2e_latency_s_p99": raw_summary.get(
                "results_end_to_end_latency_s_quantiles_p99"),
            "e2e_latency_s_mean": raw_summary.get(
                "results_end_to_end_latency_s_mean"),
            "output_throughput_tps_p50": raw_summary.get(
                "results_request_output_throughput_token_per_s_quantiles_p50"),
            "output_throughput_tps_p95": raw_summary.get(
                "results_request_output_throughput_token_per_s_quantiles_p95"),
            "mean_output_throughput_tps": raw_summary.get(
                "results_mean_output_throughput_token_per_s"),
            "timestamp_unix": raw_summary.get("timestamp"),
            "llmperf_version": raw_summary.get("version"),
        })

        for i, rec in enumerate(raw_indv):
            normalized_rows.append({
                "source_dataset_id": DATASET_ID,
                "trace_type": "latency_benchmark_trace",
                "provenance": (
                    f"{DATASET_ID}@{CONFIG}#"
                    f"{Path(run['individual_file']).name}#"
                    f"git={git_sha[:7]}"
                ),
                "run_id": run["run_id"],
                "request_index_in_run": i,
                "model": raw_summary.get("model"),
                "model_family": "anthropic_claude_instant_v1",
                "provider": "aws_bedrock",
                "engine": "bedrock_managed_api",
                "client": "ray_llmperf",
                "region": run["region"],
                "prompt_kind": run["prompt_kind"],
                "mean_input_tokens_target": int(
                    raw_summary["mean_input_tokens"]),
                "mean_output_tokens_target": int(
                    raw_summary["mean_output_tokens"]),
                "num_concurrent_requests": int(
                    raw_summary["num_concurrent_requests"]),
                "number_input_tokens": int(rec["number_input_tokens"]),
                "number_output_tokens": int(rec["number_output_tokens"]),
                "number_total_tokens": int(rec["number_total_tokens"]),
                "ttft_s": float(rec["ttft_s"]),
                "ttft_ms": float(rec["ttft_s"]) * 1000.0,
                "inter_token_latency_s": float(rec["inter_token_latency_s"]),
                "itl_ms": float(rec["inter_token_latency_s"]) * 1000.0,
                "end_to_end_latency_s": float(rec["end_to_end_latency_s"]),
                "e2e_latency_ms": float(rec["end_to_end_latency_s"]) * 1000.0,
                "request_output_throughput_tps": float(
                    rec["request_output_throughput_token_per_s"]),
                "error_code": rec.get("error_code"),
                "error_msg": rec.get("error_msg") or None,
                "failure": bool(rec.get("error_code")),
            })

    if not normalized_rows:
        raise RuntimeError(
            "No rows ingested — all 4 LLMPerf runs failed to download"
        )

    # Write analysis_sample.jsonl (gitignored)
    analysis_path = proc_dir / "analysis_sample.jsonl"
    analysis_buf = io.BytesIO()
    for row in normalized_rows:
        analysis_buf.write((json.dumps(row, sort_keys=True) + "\n").encode())
    analysis_path.write_bytes(analysis_buf.getvalue())
    analysis_sha = _sha256(analysis_path)
    analysis_size = analysis_path.stat().st_size

    # Fixture (committed) — 5 rows max, ≤ 16 KiB
    fixture_path = (
        FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_buf = io.BytesIO()
    fixture_rows = 0
    for row in normalized_rows:
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

    # Committed normalized sample (committed since license=apache-2.0)
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

    # Schema profile
    individual_raw_schema = sorted(raw_schema_observed)
    summary_raw_schema = sorted(raw_summary_schema_observed)
    dtypes: dict[str, list[str]] = {}
    example_values: dict[str, list[str]] = {}
    for col in individual_raw_schema:
        seen_dtypes: set[str] = set()
        examples: list[str] = []
        # The first run's raw rows
        for run in RUNS[:1]:
            ind_local = raw_dir / Path(run["individual_file"]).name
            if not ind_local.exists():
                continue
            with open(ind_local) as fh:
                rows = json.load(fh)
            for r in rows[:20]:
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
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "source_files": [r["summary_file"] for r in RUNS]
                        + [r["individual_file"] for r in RUNS],
        "inspected_row_count": len(normalized_rows),
        "raw_columns": individual_raw_schema,
        "raw_summary_columns": summary_raw_schema,
        "nested_keys": [],
        "dtypes": dtypes,
        "example_values": example_values,
        "missing_rates": {
            col: 0.0 for col in individual_raw_schema
        },
        "unknown_columns": [],
        "rejected_columns": [],
        "accepted_columns": individual_raw_schema,
        "list_length_summaries": {},
        "file_size_bytes": sum(
            (raw_dir / Path(r["individual_file"]).name).stat().st_size
            for r in RUNS
            if (raw_dir / Path(r["individual_file"]).name).exists()
        ),
        "per_run_summary_schema": summary_raw_schema,
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # Schema mapping
    columns_map = _build_columns_map()
    schema_mapping = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "accepted_columns": individual_raw_schema,
        "rejected_columns": [],
        "unknown_columns": [],
        "columns": columns_map,
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # Statistical rollups — per-run + overall
    overall: dict[str, dict] = {}
    for k, key in [
        ("ttft_ms", "ttft_ms"),
        ("itl_ms", "itl_ms"),
        ("e2e_latency_ms", "e2e_latency_ms"),
        ("request_output_throughput_tps", "request_output_throughput_tps"),
        ("number_input_tokens", "number_input_tokens"),
        ("number_output_tokens", "number_output_tokens"),
    ]:
        vals = [r.get(key) for r in normalized_rows if r.get(key) is not None]
        if vals:
            overall[k] = _percentiles(vals)

    by_strata: dict[str, dict] = {}
    subgroup_counts: dict[str, int] = {}
    runs_groups: dict[str, list[dict]] = {}
    for row in normalized_rows:
        runs_groups.setdefault(row["run_id"], []).append(row)
    for run_id, grp_rows in runs_groups.items():
        subgroup_counts[f"run_id={run_id}"] = len(grp_rows)
        if len(grp_rows) >= 5:
            grp_stats: dict[str, dict] = {}
            for k in ["ttft_ms", "itl_ms", "e2e_latency_ms",
                      "request_output_throughput_tps"]:
                vals = [r.get(k) for r in grp_rows if r.get(k) is not None]
                if vals:
                    grp_stats[k] = _percentiles(vals)
            if grp_stats:
                by_strata[f"run_id={run_id}"] = grp_stats

    rollups = {
        "overall": overall,
        "by_strata": by_strata,
        "subgroup_counts": subgroup_counts,
        "per_run_aggregated": per_run_summaries,
    }
    (proc_dir / "statistical_rollups.json").write_text(
        json.dumps(rollups, indent=2, sort_keys=True)
    )

    # Sample strength: 350 rows × 4 stratified runs → moderate (≥ 200 rows
    # + ≥ 4 strata, each with ≥ 50 rows).
    strength = "moderate"

    # Build summary
    normalized_schema = sorted({
        k for r in normalized_rows for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    available_signals = [
        "ttft", "itl", "e2e_latency", "throughput",
        "concurrency", "input_tokens", "output_tokens",
        "model_id", "engine", "failure_label",
    ]
    missing_signals = [
        "tpot", "queue_state", "queue_wait", "kv_cache_size",
        "memory_pressure", "gpu_type", "gpu_utilization",
        "batch_size", "engine_version", "timeout_label",
        "autoscaling", "replica_count", "kernel_duration",
        "cache_hit",
    ]

    field_quality = _field_quality()

    limitations = [
        "ssong1/llmperf-bedrock: Apache-2.0 LLMPerf benchmark output against "
        "AWS Bedrock (closed managed-API).",
        "CLOSED-API timing — TTFT/ITL/e2e include Bedrock provider scheduling "
        "+ closed-source batching + AWS network. NOT a GPU-direct serving "
        "latency prior. Treat as managed-API SLO prior only.",
        "Single model (bedrock/anthropic.claude-instant-v1, retired 2024-07). "
        "Do NOT extrapolate to newer Claude models or other Bedrock-hosted "
        "providers (e.g. AI21 / Titan / Llama-via-Bedrock).",
        "Only 4 (input_len, output_len, concurrency, region) cells × ~50-100 "
        "requests each (350 individual requests total). Stratification is "
        "by run_id; cross-cell extrapolation is weak.",
        "Concurrency is fixed at 1 across all runs — NO queue/contention "
        "signal. The Aurelius queue-risk and batch-frontier modules MUST NOT "
        "consume this dataset.",
        "Client ran from an on-premise Kubernetes bastion (per dataset card) "
        "— TTFT includes the bastion→AWS network RTT, which biases TTFT "
        "absolute values upward by 1+ network hops.",
        "Results dated 2024-01-19 — Bedrock has since changed its serving "
        "stack; treat absolute latency numbers as historical priors.",
        "Per-request `error_rate=0.0` across all 4 runs — NO failure data "
        "to calibrate timeout-risk from this dataset.",
        "Benchmark-only — Tier 4. Not pilot telemetry.",
    ]

    summary_obj = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": LICENSE,
        # Redistribution-gate metadata (tenth consumer of the canonical
        # gate). The gate classifies the recorded license tag —
        # apache-2.0 → ``permissive_apache_2_0`` → permit; ledger NOT
        # consulted (the closed permissive allow-list short-circuits).
        # These fields are ADDITIVE — the on-disk fixture / committed
        # normalized sample / analysis sample paths are unchanged. The
        # script does NOT read the gate verdict to decide whether to
        # write its samples (the committed normalised sample was
        # already committed under the existing apache-2.0 declaration);
        # the gate fields here document the canonical permit verdict so
        # a future audit can prove the script consulted the gate rather
        # than carrying its own classifier.
        "license_redistribution_status": gate_decision.license_status,
        "license_redistribution_source": LICENSE_SOURCE,
        "redistribution_gate_reason_code": gate_decision.reason_code,
        "redistribution_gate_reason_detail": gate_decision.reason_detail,
        "redistribution_gate_permitted": gate_decision.permitted,
        "redistribution_gate_operator_grant_dataset_id": (
            gate_decision.operator_grant_dataset_id
        ),
        "redistribution_gate_scope": GATE_SCOPE,
        "gated": GATED,
        "canonical_trace_type": "latency_benchmark_trace",
        "raw_committed": False,
        "raw_file_size_committed": False,
        "raw_columns": individual_raw_schema,
        "raw_schema": individual_raw_schema,
        "raw_summary_schema": summary_raw_schema,
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
        "stratification_keys": ["run_id"],
        "sampling_method": "full_bounded",
        "fixture_sample_rows": fixture_rows,
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "analysis_sample_rows": len(normalized_rows),
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
            f"{DATASET_ID}@{CONFIG}#raw_data/*_individual_responses.json"
            f"#git={git_sha[:7]}"
        ),
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": raw_manifests,
        "elapsed_s": int(time.time() - t0),
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
        # Per-request real fields from LLMPerf individual_responses.json
        "ttft_ms": "real",
        "itl_ms": "real",
        "e2e_latency_ms": "real",
        "request_output_throughput_tps": "real",
        "number_input_tokens": "real",
        "number_output_tokens": "real",
        "number_total_tokens": "real",
        "error_code": "real",
        "error_msg": "real",
        "failure": "derived",
        # Run-level metadata (real, but propagated)
        "run_id": "real",
        "request_index_in_run": "real",
        "model": "real",
        "model_family": "real",
        "provider": "real",
        "engine": "real",
        "client": "real",
        "region": "real",
        "prompt_kind": "real",
        "mean_input_tokens_target": "real",
        "mean_output_tokens_target": "real",
        "num_concurrent_requests": "real",
        # Derived
        "ttft_s": "real",
        "inter_token_latency_s": "real",
        "end_to_end_latency_s": "real",
        # Missing for Aurelius decision-making
        "gpu_type": "missing",
        "gpu_utilization": "missing",
        "queue_wait": "missing",
        "queue_depth": "missing",
        "batch_size": "missing",
        "engine_version": "missing",
        "kv_cache_size": "missing",
        "tpot_ms": "missing",
        "p50_ttft_ms": "missing",  # per-request only; aggregates in rollups
        "p95_ttft_ms": "missing",
        "p99_ttft_ms": "missing",
    }


def _build_columns_map() -> list[dict]:
    """Map LLMPerf raw individual_responses columns to Aurelius signals."""
    return [
        {
            "raw_column_name": "ttft_s",
            "normalized_field": "ttft_ms",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "seconds (×1000 → ms in normalised)",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": (
                "Time to first token measured by Ray LLMPerf client. Closed-"
                "API: includes Bedrock provider + AWS network."
            ),
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "inter_token_latency_s",
            "normalized_field": "itl_ms",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "seconds (×1000 → ms in normalised)",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior"],
            "notes": (
                "Mean per-request ITL (= (e2e - ttft) / (output_tokens - 1)). "
                "LLMPerf computes this as the actual streaming ITL of the "
                "individual request."
            ),
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "end_to_end_latency_s",
            "normalized_field": "e2e_latency_ms",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "seconds (×1000 → ms in normalised)",
            "aurelius_signal_category": "latency",
            "usable_for": ["latency_prior", "constraint_aware_evaluation"],
            "notes": "End-to-end request latency (closed-API).",
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "request_output_throughput_token_per_s",
            "normalized_field": "request_output_throughput_tps",
            "field_quality": "real",
            "dtypes": ["float"],
            "units": "tokens / second",
            "aurelius_signal_category": "throughput",
            "usable_for": ["throughput_prior"],
            "notes": "Per-request output-token throughput.",
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "number_input_tokens",
            "normalized_field": "number_input_tokens",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "latency_prior"],
            "notes": "Actual input-token count of the request.",
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "number_output_tokens",
            "normalized_field": "number_output_tokens",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only", "throughput_prior"],
            "notes": "Actual output-token count of the request.",
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "number_total_tokens",
            "normalized_field": "number_total_tokens",
            "field_quality": "real",
            "dtypes": ["int"],
            "units": "tokens",
            "aurelius_signal_category": "tokens",
            "usable_for": ["workload_shape_only"],
            "notes": "input + output tokens.",
            "presence_rate": 1.0,
            "missing_rate": 0.0,
        },
        {
            "raw_column_name": "error_code",
            "normalized_field": "error_code",
            "field_quality": "real",
            "dtypes": ["NoneType"],
            "units": None,
            "aurelius_signal_category": "failure_timeout",
            "usable_for": ["latency_prior"],
            "notes": (
                "Always null in this dataset (error_rate=0.0). Schema "
                "supports failure labels, but this snapshot has no failures."
            ),
            "presence_rate": 0.0,
            "missing_rate": 1.0,
        },
        {
            "raw_column_name": "error_msg",
            "normalized_field": "error_msg",
            "field_quality": "real",
            "dtypes": ["str"],
            "units": None,
            "aurelius_signal_category": "failure_timeout",
            "usable_for": ["latency_prior"],
            "notes": "Empty string when no error. No failures in this snapshot.",
            "presence_rate": 0.0,
            "missing_rate": 1.0,
        },
    ]


# ---------------------------------------------------------------------------
# Discovery-only audit records (Round 3)
# ---------------------------------------------------------------------------


ROUND3_DISCOVERY_RECORDS = [
    {
        "dataset_id": "DistServe/2025-05-06T14-automatic-profiling",
        "candidate_trace_type": "kernel_profile_trace",
        "license_observed": None,
        "gated": False,
        "kind": "license_unspecified_low_priority",
        "reason": (
            "vLLM CUDA-kernel-level profiling output for "
            "DeepSeek-R1-Distill-Llama-8B on H100, swept across "
            "batch_size × prompt_length (43 JSON files). Each file "
            "contains per-kernel cuda_time_us + per-layer breakdown "
            "(LlamaDecoderLayer / RMSNorm / VocabParallelEmbedding) + "
            "full vLLM engine_args context. HIGH research value as a "
            "kernel_profile_trace prior — but the dataset card has NO "
            "declared license and the repo has no LICENSE file. "
            "Without license clarity, committing a normalised sample "
            "would violate the corpus license-and-gating-recorded gate. "
            "Deferred; revisit if DistServe org adds an SPDX license."
        ),
    },
    {
        "dataset_id": "DistServe/test-amd-ci-profiler",
        "candidate_trace_type": "kernel_profile_trace",
        "license_observed": None,
        "gated": False,
        "kind": "license_unspecified_low_priority",
        "reason": (
            "Companion DistServe AMD CI profiler output. Same license "
            "issue as the 2025-05-06 profiling dump — no SPDX license "
            "on the dataset card. Deferred pending license clarity."
        ),
    },
    {
        "dataset_id": "DistServe/test-sample",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_out_of_scope",
        "reason": (
            "DistServe test sample — modality:imagefolder + n<1K. Not "
            "an Aurelius-relevant trace; image data. Rejected as "
            "out-of-scope."
        ),
    },
    {
        "dataset_id": "deepanjalimishra99/datacenter-traces",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "kind": "reject_out_of_scope",
        "reason": (
            "DynamoRIO drmemtrace lz4-compressed binary memory traces "
            "from SPECrate/lectern + multiple workloads (bc, blender, "
            "etc.). Microarchitectural memory-trace data, NOT LLM "
            "serving / scheduler / GPU telemetry. Useful for hardware "
            "architecture simulation but out of scope for Aurelius' "
            "constraint-aware LLM-serving decisions. Rejected as "
            "out-of-scope."
        ),
    },
    {
        "dataset_id": "intellistream/sage-control-plane-llm-workloads",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "insufficient_sample_no_license",
        "reason": (
            "3 rows of workload-configuration metadata "
            "(workload_id, request_count, rate_per_second, "
            "arrival_pattern, model_distribution, priority_distribution, "
            "prompt_len_range, output_len_range, slo_deadlines). "
            "Aurelius-relevant fields present, but 3 rows is below "
            "the fixture-only threshold AND no declared license. "
            "Rejected as insufficient sample + license unspecified."
        ),
    },
    {
        "dataset_id": "Nathan-Maine/dgx-spark-kv-cache-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Already audited in PR #134; same KV-cache quantization "
            "benchmark as memoriant/dgx-spark-kv-cache-benchmark "
            "(promoted_for_training_priors). Re-noting in Round 3 "
            "for completeness."
        ),
    },
    {
        "dataset_id": "hlarcher/inference-benchmarker",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Already audited in PR #134 — ShareGPT-derived prompt "
            "fixtures used to drive huggingface/inference-benchmarker. "
            "Workload-shape only; existing sharegpt_aiperf ingester "
            "covers this role. Re-confirmed in Round 3."
        ),
    },
    {
        "dataset_id": "kshitijthakkar/moe-inference-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": "apache-2.0",
        "gated": True,
        "kind": "gated_blocked",
        "reason": (
            "HF gated:manual. Features list (model_id, prompt, "
            "tokens_generated, time_seconds, tokens_per_second, "
            "total_params, active_params, error) is exactly the "
            "MoE-specific latency-benchmark schema we want as a "
            "first MoE-specific prior. HF_TOKEN is NOT authorised. "
            "Confirmed gated_blocked in Round 3. Revisit if manual "
            "approval is granted."
        ),
    },
]


def write_round3_audit_summary(
    ingest_out: dict,
    dest: Path,
    *,
    ledger: OperatorPolicyLedger | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ledger is None:
        ledger = _load_ledger()
    s = ingest_out["summary"]
    payload = {
        # v2 schema introduces the redistribution_gate_* top-level
        # triple mirroring the broadened_discovery_audit_summary v2 and
        # acmetrace_audit_summary_v2 schemas. Per-row gate fields are
        # also added under ``ingested``. v1 readers continue to
        # function: the v1 schema is a strict subset of v2 (all v1
        # keys are preserved).
        "doc_version": "round3_broadened_discovery_audit_summary_v2",
        "scope": (
            "Round 3 broadened HF discovery — bounded ingest of "
            "ssong1/llmperf-bedrock (Tier-4 closed-managed-API LLMPerf "
            "TTFT/ITL/e2e priors) + 8 negative-result discovery records "
            "covering DistServe profiling (license unspecified), "
            "DynamoRIO drmemtrace (out of scope), "
            "intellistream sage-control-plane (insufficient sample), "
            "Nathan-Maine + hlarcher (already audited duplicates), "
            "kshitijthakkar MoE benchmarks (gated_blocked)."
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
        "ingested": [
            {
                "dataset_id": s["dataset_id"],
                "config_name": s["config_name"],
                "canonical_trace_type": s["canonical_trace_type"],
                "license": s["license"],
                # Tenth-consumer gate-derived fields. The audit summary
                # mirrors the same closed-set fields the per-config
                # summary.json carries so reviewers can pivot on either
                # source without re-running the gate.
                "license_redistribution_status":
                    s["license_redistribution_status"],
                "redistribution_gate_reason_code":
                    s["redistribution_gate_reason_code"],
                "redistribution_gate_permitted":
                    s["redistribution_gate_permitted"],
                "redistribution_gate_operator_grant_dataset_id":
                    s["redistribution_gate_operator_grant_dataset_id"],
                "gated": s["gated"],
                "analysis_sample_rows": s["analysis_sample_rows"],
                "fixture_sample_rows": s["fixture_sample_rows"],
                "committed_normalized_sample_rows":
                    s["committed_normalized_sample_rows"],
                "committed_normalized_sample_bytes":
                    s["committed_normalized_sample_bytes"],
                "available_signals": s["available_signals"],
                "missing_signals": s["missing_signals"],
                "limitations": s["limitations"],
                "statistical_sample_strength":
                    s["statistical_sample_strength"],
                "promotion_state": ingest_out["decision"]["state"],
                "promotion_tags": ingest_out["decision"]["promotion_tags"],
                "promotion_reasons": ingest_out["decision"]["reasons"],
            },
        ],
        "discovery_only_records": ROUND3_DISCOVERY_RECORDS,
        "failed": [],
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dest


# ---------------------------------------------------------------------------
# Registry update
# ---------------------------------------------------------------------------


def update_canonical_registry(registry_path: Path, entry: dict) -> dict:
    if registry_path.exists():
        d = json.loads(registry_path.read_text())
    else:
        d = {
            "doc_version": "hf_corpus_canonical_registry_v1",
            "entries": [],
        }
    existing = d.get("entries", [])
    # Replace any prior entry for the same (dataset_id, config_name)
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

    # Build the ssong1/llmperf-bedrock candidate if not already present;
    # otherwise update.
    target_id = ingested_entry["dataset_id"]
    found = False
    for c in cands:
        if c.get("dataset_id") == target_id:
            found = True
            c["recommended_action"] = "ingest_now_bounded"
            c["candidate_trace_type"] = "latency_benchmark_trace"
            c["trust_level"] = "tier_4_latency_benchmark_traces"
            c["gated_status"] = "public"
            c["license"] = "apache-2.0"
            c["matched_keywords"] = list(set(c.get("matched_keywords", []) + [
                "round3::llmperf_bedrock",
                "latency_benchmark::ttft_itl_e2e",
                "closed_api::aws_bedrock",
            ]))
            c["overall_priority_score"] = 4.0
            c["ingestion_feasibility_score"] = 5
            c["frontier_value_score"] = 3
            c["schema_quality_score"] = 4
            c["production_similarity_score"] = 2
            c["aurelius_use_case"] = (
                "Managed-API TTFT/ITL/e2e priors — Aurelius routing + "
                "deferral when a managed-provider (Bedrock-class) option "
                "exists. NOT a GPU-direct serving prior."
            )
            c["not_recommended_uses"] = [
                "GPU-direct serving latency calibration",
                "Queue / concurrency risk calibration (concurrency = 1)",
                "Dynamic frontier calibration",
                "Pilot-grade SLA truth",
            ]
            break
    if not found:
        cands.append({
            "dataset_id": target_id,
            "dataset_url": f"https://huggingface.co/datasets/{target_id}",
            "gated_status": "public",
            "license": "apache-2.0",
            "estimated_size": ["10K<n<100K"],
            "available_splits": [],
            "schema_available": True,
            "matched_keywords": [
                "round3::llmperf_bedrock",
                "latency_benchmark::ttft_itl_e2e",
                "closed_api::aws_bedrock",
            ],
            "candidate_trace_type": "latency_benchmark_trace",
            "trust_level": "tier_4_latency_benchmark_traces",
            "available_signals": [
                "ttft", "itl", "e2e_latency", "throughput",
                "concurrency", "input_tokens", "output_tokens",
            ],
            "missing_signals": [
                "tpot", "queue_state", "queue_wait", "kv_cache_size",
                "memory_pressure", "gpu_type", "gpu_utilization",
                "batch_size", "engine_version", "timeout_label",
                "autoscaling", "replica_count", "kernel_duration",
                "cache_hit",
            ],
            "aurelius_use_case": (
                "Managed-API TTFT/ITL/e2e priors for routing + deferral."
            ),
            "not_recommended_uses": [
                "GPU-direct serving latency calibration",
                "Queue / concurrency risk calibration (concurrency = 1)",
                "Dynamic frontier calibration",
                "Pilot-grade SLA truth",
            ],
            "ingestion_feasibility_score": 5,
            "frontier_value_score": 3,
            "schema_quality_score": 4,
            "production_similarity_score": 2,
            "overall_priority_score": 4.0,
            "recommended_action": "ingest_now_bounded",
            "feature_names": [
                "end_to_end_latency_s", "error_code", "error_msg",
                "inter_token_latency_s", "number_input_tokens",
                "number_output_tokens", "number_total_tokens",
                "request_output_throughput_token_per_s", "ttft_s",
            ],
            "classification_evidence": {
                "latency_benchmark_trace": [
                    "ttft_s", "inter_token_latency_s",
                    "end_to_end_latency_s",
                    "request_output_throughput_token_per_s",
                ],
            },
            "downloads": 4,
            "likes": 1,
            "last_modified": "2024-01-24T00:50:26.000Z",
            "discovery_timestamp_s": time.time(),
            "configs": [],
        })

    # Add discovery-only records (negative results) if not already present.
    existing_ids = {c.get("dataset_id") for c in cands}
    for rec in ROUND3_DISCOVERY_RECORDS:
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
            "matched_keywords": [f"round3::{rec['kind']}"],
            "candidate_trace_type": rec["candidate_trace_type"],
            "trust_level": "tier_4_latency_benchmark_traces",
            "available_signals": [],
            "missing_signals": [],
            "aurelius_use_case": "Round-3 discovery audit — see reason.",
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
            "round3_audit_reason": rec["reason"],
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
        description="Round-3 broadened HF ingest — ssong1/llmperf-bedrock"
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--skip-registry-update", action="store_true",
        help="Run the ingest only; do not modify the corpus registries.")
    p.add_argument(
        "--output-root", default=str(HF_DIR),
        help="Override HF_DIR — used by tests.")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load the operator policy ledger once; thread it through the
    # ingest call and the audit summary writer so a single source of
    # truth records ``redistribution_gate_policy_default`` and
    # ``redistribution_gate_policy_grant_count``.
    ledger = _load_ledger()

    out = ingest(output_root=Path(args.output_root), ledger=ledger)

    if not args.skip_registry_update:
        registry_path = DISC_DIR / "canonical_corpus_registry.json"
        candidates_path = DISC_DIR / "hf_dataset_candidates.json"
        audit_path = DISC_DIR / "round3_broadened_discovery_audit_summary.json"

        update_canonical_registry(registry_path, out["registry_entry"])
        update_candidate_registry(candidates_path, out["registry_entry"])
        write_round3_audit_summary(out, audit_path, ledger=ledger)

    decision = out["decision"]
    print(json.dumps({
        "dataset_id": out["summary"]["dataset_id"],
        "config_name": out["summary"]["config_name"],
        "promotion_state": decision["state"],
        "promotion_tags": decision["promotion_tags"],
        "analysis_sample_rows": out["summary"]["analysis_sample_rows"],
        "committed_normalized_sample_rows": out["summary"][
            "committed_normalized_sample_rows"],
        "committed_normalized_sample_bytes": out["summary"][
            "committed_normalized_sample_bytes"],
        "fixture_sample_rows": out["summary"]["fixture_sample_rows"],
        "license": out["summary"]["license"],
        "gated": out["summary"]["gated"],
        "trace_type": out["summary"]["canonical_trace_type"],
        "statistical_sample_strength": out["summary"][
            "statistical_sample_strength"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
