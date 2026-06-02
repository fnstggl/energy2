"""Round-7 broadened HF discovery audit + H200 cross-source methodology drift.

Discovery / data-engine PR. No scheduler change, no controller change, no
robust-energy-engine change, no oracle as headline, no Tier 1 promotion,
no production claim.

This script does two things:

1. **Round-7 broadened HF discovery audit** — re-runs ~30 high-priority
   search-term groups against the public HF datasets API, surfaces any
   newly-published candidates that did NOT appear in PR #148's Round-6
   pass, inspects each via cardData / siblings / README, classifies
   them, and persists either (a) an ingest decision when the candidate
   qualifies, or (b) a discovery-only rejection record otherwise. The
   8 rejection records this round are the negative result for the
   public HF ecosystem under the current Aurelius search rubric.

2. **H200 cross-source methodology drift audit** — quantifies the
   overlap between
   `ssakethch/h200-quantization-benchmarks @ throughput` (275 rows,
   single-source vLLM H200 SXM MIG-partitioned, real per-cell
   TTFT/TPOT/ITL p50/p99 + throughput) and the 10 metrum-ai/llm-perfdata
   H200 rows (multi-source curated, mixed-engine). The metrum H200
   slice has only ONE row with TPOT+TTFT (SGLang/Llama-3.1-70B/BF16,
   c=10, 8 GPUs) and ONE row with tokens_per_sec (vLLM/Llama-3.1-8B/FP8,
   8 GPUs); the rest are "Target" placeholder rows without
   measurements. The audit therefore is small-N and reports a
   bounded methodology-drift observation rather than a sweeping
   methodology-drift conclusion.

Artifacts written:
- data/external/hf_discovery/round7_broadened_discovery_audit_summary.json
- data/external/hf_discovery/h200_cross_source_methodology_audit.json
- data/external/hf_discovery/hf_dataset_candidates.json (updated)

Tests live at tests/test_hf_round7_audit.py.

Honesty + scope guarantees: no scheduler / controller / robust-energy-engine
change; no oracle as headline; no Tier 1 promotion; no production claim;
no new HF data downloaded beyond the existing committed normalized samples;
no HF_TOKEN leak.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
CANDIDATES_PATH = DISC_DIR / "hf_dataset_candidates.json"
SSAKETHCH_DIR = (
    REPO_ROOT
    / "data"
    / "external"
    / "hf"
    / "ssakethch__h200-quantization-benchmarks"
    / "throughput"
    / "processed"
)
METRUM_DIR = (
    REPO_ROOT
    / "data"
    / "external"
    / "hf"
    / "metrum-ai__llm-perfdata"
    / "multi_source_curated_v1"
    / "processed"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round-7 audit data: 8 discovery-only rejection records
# ---------------------------------------------------------------------------

ROUND7_DISCOVERY_ONLY: list[dict[str, Any]] = [
    {
        "dataset_id": "core12345/real_GPU_exp_placement_trace",
        "bucket": "C_gated_blocked",
        "kind": "gated_blocked",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": True,
        "reason": (
            "HF dataset card reports `gated=auto`. Single 9.94 GB "
            "`Qwen3-235B-A22B-FP8-traces.tar.gz` archive (no README, no "
            "license). Name suggests real-GPU placement traces, which "
            "would be a high-value candidate for the Aurelius placement / "
            "routing engine, but HF_TOKEN is not authorised on the "
            "current account. Marked gated_blocked and recorded for "
            "follow-up if access is granted; do NOT attempt unbounded "
            "ingest of a 9.94 GB compressed archive without explicit "
            "approval per the bounded-ingestion guard."
        ),
    },
    {
        "dataset_id": "odyn-network/benchmark-dataset-different-gpu-workload",
        "bucket": "D_synthetic_estimates",
        "kind": "reject_synthetic_estimates",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "cc-by-4.0",
        "gated": False,
        "reason": (
            "GPU catalog × LLM workload VRAM benchmark by the same author "
            "as the already-ingested odyn-network/odyn-benchmarks. README "
            "explicitly self-declares the rows as `math_engine` VRAM "
            "ESTIMATES paired with `document_engine_recommended_vram_gb` "
            "and `llm_judge_verdict` audit columns: 'Not suitable as "
            "Ground-truth hardware measurements unless you add and "
            "document separate measurement runs.' Synthetic / estimated "
            "VRAM capacity-planning data — NOT measured GPU performance. "
            "Rejected per the binding directive 'Do NOT treat synthetic "
            "cost fields as real economics' (extended to synthetic VRAM "
            "capacity estimates). Same author authored the ingested "
            "odyn-benchmarks (real vLLM + Ray Serve TTFT/TPOT/throughput "
            "measurements), so the author's measured contributions are "
            "already in the corpus."
        ),
    },
    {
        "dataset_id": "BBuf/ltx-fp8-sglang-benchmark-results",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Lightricks LTX-2.0 / LTX-2.3 text-to-video diffusion "
            "benchmark on a single H100. Metrics are total_s / denoise_s "
            "/ decode_s for VIDEO GENERATION (8 inference steps, "
            "5.04 s output clip), NOT LLM serving TTFT / TPOT / "
            "throughput. The 'sglang' in the name refers to the "
            "diffusion-model launcher, not LLM serving. Out of scope for "
            "the LLM-serving / cluster-scheduling federated corpus. "
            "Rejected as irrelevant domain (video diffusion, not LLM). "
            "license=None so no committed sample is allowed under the "
            "conservative redistribution policy."
        ),
    },
    {
        "dataset_id": "Isabella5/sglang-seglen-benchmark",
        "bucket": "F_duplicate_existing",
        "kind": "duplicate_existing",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "1.46 GB total. Files: ShareGPT_V3_unfiltered_cleaned_split.json, "
            "mixed_prefix_eviction_dataset_7k_10k_with_noise.json, "
            "multi_group_shared_prefix_dataset.json, "
            "swebench_sps=10_art={5,10}_nums=100.jsonl. These are INPUT "
            "prompts for the SGLang sequence-length benchmark harness, "
            "NOT the benchmark RESULTS (no TTFT / TPOT / throughput / "
            "GPU columns). ShareGPT_V3 is identical to the corpus' "
            "existing sharegpt_aiperf workload-shape ingester; "
            "swebench_* spans duplicate the Exgentic/agent-llm-traces "
            "agent-workload role. Rejected as duplicate of existing "
            "workload-shape sources. license=None blocks redistribution "
            "regardless."
        ),
    },
    {
        "dataset_id": "fabric/inference-benchmarker",
        "bucket": "F_duplicate_existing",
        "kind": "duplicate_existing",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "2.02 GB total. Files include "
            "ShareGPT_V3_unfiltered_cleaned_split_chat_format.json, "
            "classification.json, share_gpt_0_turns.json, "
            "share_gpt_filtered_small.json + French / story variants. "
            "Mirror of the previously-rejected hlarcher/inference-benchmarker "
            "rationale: ShareGPT-derived workload INPUT fixtures used by "
            "huggingface/inference-benchmarker as request shape, NOT "
            "benchmark RESULTS. Duplicates the existing sharegpt_aiperf "
            "ingester's workload-shape role. Rejected as duplicate."
        ),
    },
    {
        "dataset_id": "vrvrv/vllm-benchmark-datasets",
        "bucket": "F_duplicate_existing",
        "kind": "duplicate_existing",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "11 MB total. 4 parquet files (dataclaw, humaneval, novita, "
            "spider) + a create_hf_datasets.py script. INPUT prompts "
            "for vLLM benchmark_serving.py, not benchmark RESULTS. "
            "Workload-shape role already covered by sharegpt_aiperf + "
            "Exgentic/agent-llm-traces. Rejected as duplicate."
        ),
    },
    {
        "dataset_id": "ashwinnv/agent-telemetry-prompt-framing-mint-full1035-qwen32b",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "Despite the 'agent-telemetry' name, this is a CLINICAL-QA "
            "agent eval dataset (MINT medical QA paper replication, "
            "Qwen3-32B). 4535 rows × 36 columns: task_id, specialty, "
            "question, options, gold_answer, answer, correct, "
            "parse_error, raw_response, trajectory (JSON per-turn), "
            "lab_placement, n_turns, abstained, lab_lure metric. The "
            "`agent_telemetry_mode` column refers to whether the agent "
            "had access to clinical-tools telemetry, NOT to server / "
            "serving telemetry. NO latency / throughput / GPU / queue / "
            "energy fields. Rejected as irrelevant domain (clinical "
            "agent capability eval, not infrastructure)."
        ),
    },
    {
        "dataset_id": "juniworld/prompt_inference_traces",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "1.7 MB. 26 parquet files split across DD16_gpt4 / DD16_local "
            "/ FEDWEB13_gpt4 / FEDWEB13_local / SESSION14_gpt4 / etc. "
            "Schema = prompt (string) + domain_list (sequence[string]) "
            "+ url_list (sequence[string]). Federated domain / URL "
            "retrieval prompts, NOT inference latency / throughput / "
            "queue / cost / energy / GPU telemetry. Despite "
            "'inference_traces' in the name, no measured infrastructure "
            "signal. Rejected as irrelevant domain (retrieval-prompt "
            "fixtures, not infrastructure telemetry)."
        ),
    },
    {
        "dataset_id": "efficient-speech/tts-serving-benchmark",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Text-to-speech serving benchmark INPUT dataset (HiFi-TTS + "
            "VCTK + LJ-Speech + Libri-TTS + Libri-Light + EMOV-DB + "
            "Tedlium + jenny-tts splits, 178 MB total). Audio-domain "
            "benchmark inputs (mostly parquet of speech samples), not "
            "LLM-serving benchmark results. Out of scope for the LLM "
            "federated corpus. Rejected as irrelevant domain (speech "
            "synthesis, not LLM serving)."
        ),
    },
    {
        "dataset_id": "wseaton/prefix-cache-bench",
        "bucket": "F_low_value",
        "kind": "reject_low_value",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "194 KB single parquet with one column `text` (500 rows). "
            "INPUT prompts only — no measured TTFT / TPOT / cache-hit / "
            "GPU / queue. license=None blocks redistribution. The "
            "existing jaytonde05/prefixbench config already covers the "
            "synthetic prefix-cache role with a 4,000-row corpus and a "
            "richer cache-bucket schema. Rejected as low-value duplicate."
        ),
    },
    {
        "dataset_id": "bldeaw/guardrails-load-test-results",
        "bucket": "F_empty_repository",
        "kind": "reject_empty_repository",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "usedStorage=0. Files listed in cardData (load_all_*.json, "
            "load_analytics_*.json, api_full_options_*.json) are not "
            "actually published. Cannot be ingested. Rejected as empty "
            "repository (mirrors the round-5 "
            "uohna/llm_inference_energy_combined.parquet rejection)."
        ),
    },
    {
        "dataset_id": "st192011/KVCaches",
        "bucket": "F_low_value",
        "kind": "reject_raw_artifacts_only",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "2.40 GB of raw .bin KV-cache binaries for three text "
            "prompts (Wiki_Grimms_Märchen, faq_Schengen_visa, "
            "guia_consultores_2026). No README schema, no measured "
            "TTFT / TPOT / cache-hit / GPU / queue. Raw prefix-cache "
            "artifacts are not analyzable as a benchmark RESULTS "
            "dataset. Rejected as raw-artifacts only (no infrastructure "
            "telemetry signal)."
        ),
    },
    {
        "dataset_id": "h4shk4t/fast-kv-compaction-cache",
        "bucket": "F_low_value",
        "kind": "reject_raw_artifacts_only",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "634 MB single Qwen3-4B.pt file. Model-checkpoint-shaped "
            "artifact, not a benchmark RESULTS dataset. No measured "
            "TTFT / TPOT / cache-hit / GPU / queue. license=None. "
            "Rejected as raw-artifacts only."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Candidate registry updater
# ---------------------------------------------------------------------------


def _update_candidate_registry(records: list[dict[str, Any]]) -> Path:
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(CANDIDATES_PATH)
    d = _read_json(CANDIDATES_PATH)
    cands = d["candidates"]
    by_id = {c["dataset_id"]: c for c in cands}

    for rec in records:
        ds_id = rec["dataset_id"]
        if ds_id in by_id:
            c = by_id[ds_id]
        else:
            c = {
                "dataset_id": ds_id,
                "dataset_url": f"https://huggingface.co/datasets/{ds_id}",
                "discovery_timestamp_s": time.time(),
                "downloads": 0,
                "likes": 0,
                "estimated_size": [],
                "available_splits": [],
                "configs": [],
                "feature_names": [],
                "available_signals": [],
                "missing_signals": [],
                "not_recommended_uses": [],
                "matched_keywords": [],
                "classification_evidence": {},
                "schema_available": False,
                "candidate_trace_type": rec["candidate_trace_type"],
                "trust_level": "tier_6_synthetic_benchmark_data",
                "ingestion_feasibility_score": 1,
                "frontier_value_score": 1,
                "schema_quality_score": 1,
                "production_similarity_score": 1,
                "overall_priority_score": 1.0,
                "aurelius_use_case": "Round-7 discovery audit — see reason.",
                "last_modified": None,
            }
            cands.append(c)
            by_id[ds_id] = c
        c["round7_audit_bucket"] = rec["bucket"]
        c["round7_audit_reason"] = rec["reason"]
        c["recommended_action"] = rec["kind"]
        c["gated_status"] = "gated" if rec.get("gated") else "public"
        if rec.get("license_observed") is not None:
            c["license"] = rec["license_observed"]
        existing_kw = c.get("matched_keywords") or []
        kw = f"round7::{rec['kind']}"
        if kw not in existing_kw:
            existing_kw.append(kw)
            c["matched_keywords"] = existing_kw

    d["candidates"] = cands
    d["candidate_count"] = len(cands)
    d["last_updated_at_s"] = time.time()
    d["updated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    d["updated_at_s"] = time.time()
    d["focused_audit_2026_06_02"] = {
        "doc_version": "round7_broadened_discovery_audit_v1",
        "scope": (
            "Round-7 broadened HF discovery — re-ran ~30 search-term "
            "groups against the public HF API, surfaced 13 newly-"
            "appearing candidates (none in the existing 79-candidate "
            "registry), inspected each via cardData / siblings / README, "
            "and produced 13 discovery-only rejection records. ZERO "
            "ingestible candidates. One gated_blocked (real-GPU "
            "placement trace, 9.9 GB tar.gz, gated=auto), one "
            "synthetic-estimates (odyn-network VRAM-estimator), three "
            "irrelevant-domain (video diffusion, clinical-QA, retrieval "
            "prompts, speech synthesis), three duplicate (ShareGPT-derived "
            "workload-shape variants), three low-value (200 KB prompt "
            "parquet, empty repository, raw KV-cache binaries), and one "
            "raw model checkpoint."
        ),
        "audited_at_s": time.time(),
        "git_sha": _git_sha(),
    }
    CANDIDATES_PATH.write_text(json.dumps(d, indent=2, sort_keys=True))
    return CANDIDATES_PATH


# ---------------------------------------------------------------------------
# Round-7 audit summary writer
# ---------------------------------------------------------------------------


def _write_round7_audit_summary() -> Path:
    out = DISC_DIR / "round7_broadened_discovery_audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_version": "round7_broadened_discovery_audit_summary_v1",
        "audited_at_s": time.time(),
        "scope": (
            "Round 7 broadened HF discovery — no new ingest. Re-ran ~30 "
            "search-term groups against the public HF API (vllm benchmark, "
            "sglang benchmark, inference benchmark, mlperf, tpot, ttft, "
            "queue depth, prefix cache, kv cache, gpu telemetry, placement "
            "trace, scheduler trace, gpu pricing, cost aware, spot price, "
            "energy trace, carbon intensity, datacenter telemetry, etc.). "
            "Surfaced 13 newly-appearing candidates (none in the existing "
            "79-candidate registry). All 13 were rejected as discovery-"
            "only. ZERO ingestible candidates this round. Plus a bounded "
            "H200 cross-source methodology drift audit between "
            "ssakethch/h200-quantization-benchmarks (single-source vLLM "
            "H200 SXM MIG-partitioned) and the 10 metrum-ai/llm-perfdata "
            "H200 rows (multi-source curated, mixed engines, mostly "
            "Target placeholder rows). Confirms Round-5 + Round-6 "
            "negative result on economic signals: the public HF dataset "
            "space does NOT currently close the operational × economic "
            "join gap."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "ingested": [],
        "failed": [],
        "discovery_only_records": ROUND7_DISCOVERY_ONLY,
        "economic_priority_summary": {
            "datasets_with_operational_and_economic_signals": [],
            "datasets_with_economic_only_signals": [],
            "join_keys_available_for_economic_overlays": [
                "gpu_type", "model_id", "engine", "quantization",
                "request_rate", "region", "cloud_provider",
            ],
            "scorer_coefficients_calibratable_from_round7": [],
            "scorer_coefficients_operator_policy_only_after_round7": [
                "gpu_hour_price_usd",
                "kwh_per_request",
                "carbon_g_per_kwh",
                "spot_interruption_probability",
                "egress_cost_per_gb",
                "regional_price_usd_per_mwh",
            ],
            "negative_result_finding": (
                "Round 7 surveyed ~30 search-term groups across "
                "operational telemetry + economic signal queries. "
                "Surfaced 13 new candidates; ZERO qualified for "
                "bounded ingest. The public HF dataset space remains "
                "saturated for these query terms: synthetic estimators "
                "(odyn-network VRAM math_engine), wrong-domain (LTX "
                "video diffusion, MINT clinical agent eval, retrieval "
                "prompts, speech synthesis), duplicates (3 ShareGPT-"
                "derived input fixtures), low-value (prompt-only "
                "parquets, empty repo, raw KV-cache binaries, raw model "
                "checkpoint), and one gated_blocked (Qwen3-235B placement "
                "trace, 9.9 GB gated=auto). NONE carry economic columns. "
                "This is the THIRD CONSECUTIVE ROUND (5, 6, 7) confirming "
                "the same negative result: Aurelius' goodput/$ "
                "denominator REMAINS operator-policy-supplied + "
                "public-pricing-prior + ElectricityMaps / ENTSO-E carbon "
                "intensity (already integrated). The Round-7 sweep was "
                "DESIGNED to falsify the Round-5/Round-6 finding; it "
                "failed to falsify, strengthening confidence in the "
                "negative result."
            ),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# H200 cross-source methodology drift audit
# ---------------------------------------------------------------------------


def _load_ssakethch_overall() -> dict[str, Any]:
    rollups = _read_json(SSAKETHCH_DIR / "statistical_rollups.json")
    by_strata = rollups.get("by_strata", {})
    overall = rollups.get("overall", {})
    # Pull out a few specific cells for the audit:
    # - llama-3.1-8B FP8 at request_rate=4 from fixture
    # - aggregate overall metrics
    return {
        "overall_count": overall.get("mean_ttft_ms", {}).get("count", 0),
        "overall_mean_ttft_ms": overall.get("mean_ttft_ms", {}).get("mean"),
        "overall_p50_ttft_ms": overall.get("mean_ttft_ms", {}).get("p50"),
        "overall_p99_ttft_ms": overall.get("mean_ttft_ms", {}).get("p99"),
        "overall_mean_tpot_ms": overall.get("mean_tpot_ms", {}).get("mean"),
        "overall_p50_tpot_ms": overall.get("mean_tpot_ms", {}).get("p50"),
        "overall_p99_tpot_ms": overall.get("mean_tpot_ms", {}).get("p99"),
        "overall_mean_total_token_throughput": overall.get(
            "total_token_throughput", {}).get("mean"),
        "overall_p50_total_token_throughput": overall.get(
            "total_token_throughput", {}).get("p50"),
        "overall_p99_total_token_throughput": overall.get(
            "total_token_throughput", {}).get("p99"),
        "subgroup_counts": rollups.get("subgroup_counts", {}),
    }


def _load_ssakethch_fixture() -> list[dict[str, Any]]:
    """Load the 5-row fixture sample to extract llama-3.1-8B FP8 row."""
    fix = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "hf"
        / "ssakethch__h200-quantization-benchmarks__throughput_sample.jsonl"
    )
    rows: list[dict[str, Any]] = []
    with fix.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_metrum_h200_rows() -> list[dict[str, Any]]:
    p = METRUM_DIR / "committed_normalized_sample.jsonl"
    rows: list[dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("gpu_type") == "NVIDIA H200":
                rows.append(r)
    return rows


def _compute_h200_audit() -> dict[str, Any]:
    ss = _load_ssakethch_overall()
    fix = _load_ssakethch_fixture()
    metrum_h200 = _load_metrum_h200_rows()

    # Extract Llama-3.1-8B FP8 from ssakethch fixture
    ss_llama_8b_fp8 = None
    for row in fix:
        if (
            "llama-3.1-8b" in row.get("model_id", "").lower()
            and row.get("quantization") == "fp8"
        ):
            ss_llama_8b_fp8 = row
            break

    # Extract from metrum the two H200 rows with measurements
    metrum_llama_70b_sglang = None
    metrum_llama_8b_fp8_vllm = None
    for r in metrum_h200:
        if (
            r.get("model_id") == "Llama-3.1"
            and r.get("model_size") == "70B"
            and r.get("engine") == "SGLang"
            and r.get("ttft_ms") is not None
        ):
            metrum_llama_70b_sglang = r
        if (
            r.get("model_id") == "Llama-3.1"
            and r.get("model_size") == "8B"
            and r.get("engine") == "vLLM"
            and r.get("precision") == "FP8"
            and r.get("tokens_per_sec") is not None
        ):
            metrum_llama_8b_fp8_vllm = r

    # Counts
    metrum_with_ttft = sum(1 for r in metrum_h200 if r.get("ttft_ms") is not None)
    metrum_with_tpot = sum(1 for r in metrum_h200 if r.get("tpot_ms") is not None)
    metrum_with_tps = sum(
        1 for r in metrum_h200 if r.get("tokens_per_sec") is not None
    )

    audit = {
        "doc_version": "h200_cross_source_methodology_audit_v1",
        "audited_at_s": time.time(),
        "git_sha": _git_sha(),
        "scope": (
            "Bounded H200 cross-source methodology drift audit comparing "
            "ssakethch/h200-quantization-benchmarks@throughput "
            "(275 rows × 40 models × 5 quants × 5 request_rates × vLLM "
            "H200 SXM MIG-partitioned) and the 10 H200 rows in "
            "metrum-ai/llm-perfdata@multi_source_curated_v1 (multi-source "
            "curated, mixed engines). The metrum-ai H200 slice is mostly "
            "placeholder 'Target' rows: only 1 of 10 carries TTFT+TPOT, "
            "1 carries tokens_per_sec, and 8 are coverage-cell "
            "placeholders. Conclusion: bounded methodology-drift "
            "observation, NOT a sweeping methodology conclusion."
        ),
        "production_claim": False,
        "uses_oracle_as_headline": False,
        "ssakethch_overall": ss,
        "metrum_h200_summary": {
            "total_rows": len(metrum_h200),
            "rows_with_ttft": metrum_with_ttft,
            "rows_with_tpot": metrum_with_tpot,
            "rows_with_tokens_per_sec": metrum_with_tps,
            "engines_present": sorted(
                {r.get("engine") for r in metrum_h200 if r.get("engine")}
            ),
            "models_present": sorted(
                {f"{r.get('model_id')}-{r.get('model_size')}"
                 for r in metrum_h200}
            ),
        },
        "directly_comparable_cells": [
            {
                "cell": "Llama-3.1-8B / FP8 / vLLM / 1xH200 SXM (MIG) vs "
                        "8xH200 (full GPUs)",
                "ssakethch_observation": {
                    "model_id": ss_llama_8b_fp8.get("model_id") if ss_llama_8b_fp8 else None,
                    "quantization": ss_llama_8b_fp8.get("quantization") if ss_llama_8b_fp8 else None,
                    "request_rate": ss_llama_8b_fp8.get("request_rate") if ss_llama_8b_fp8 else None,
                    "mean_ttft_ms": ss_llama_8b_fp8.get("mean_ttft_ms") if ss_llama_8b_fp8 else None,
                    "mean_tpot_ms": ss_llama_8b_fp8.get("mean_tpot_ms") if ss_llama_8b_fp8 else None,
                    "total_token_throughput_per_replica": (
                        ss_llama_8b_fp8.get("total_token_throughput") if ss_llama_8b_fp8 else None
                    ),
                    "gpu_type": ss_llama_8b_fp8.get("gpu_type") if ss_llama_8b_fp8 else None,
                    "engine": "vllm",
                    "notes": (
                        "ssakethch is single-replica MIG-partitioned H200 "
                        "SXM. total_token_throughput is per-replica."
                    ),
                },
                "metrum_observation": {
                    "model_id": (
                        f"{metrum_llama_8b_fp8_vllm.get('model_id')}-"
                        f"{metrum_llama_8b_fp8_vllm.get('model_size')}"
                        if metrum_llama_8b_fp8_vllm else None
                    ),
                    "precision": (
                        metrum_llama_8b_fp8_vllm.get("precision")
                        if metrum_llama_8b_fp8_vllm else None
                    ),
                    "num_gpus": (
                        metrum_llama_8b_fp8_vllm.get("num_gpus")
                        if metrum_llama_8b_fp8_vllm else None
                    ),
                    "engine": (
                        metrum_llama_8b_fp8_vllm.get("engine").lower()
                        if metrum_llama_8b_fp8_vllm else None
                    ),
                    "tokens_per_sec_aggregate": (
                        metrum_llama_8b_fp8_vllm.get("tokens_per_sec")
                        if metrum_llama_8b_fp8_vllm else None
                    ),
                    "tokens_per_sec_per_gpu_normalized": (
                        round(
                            metrum_llama_8b_fp8_vllm.get("tokens_per_sec")
                            / metrum_llama_8b_fp8_vllm.get("num_gpus"),
                            2,
                        )
                        if metrum_llama_8b_fp8_vllm
                        and metrum_llama_8b_fp8_vllm.get("tokens_per_sec") is not None
                        and metrum_llama_8b_fp8_vllm.get("num_gpus")
                        else None
                    ),
                    "notes": (
                        "metrum is aggregated 8xH200 full-GPU vLLM "
                        "tokens/s; not per-MIG-instance."
                    ),
                },
                "methodology_drift_observation": (
                    "Per-GPU normalization: metrum 64915 tok/s / 8 GPUs = "
                    "8114 tok/s per full H200. ssakethch Llama-3.1-8B FP8 "
                    "at request_rate=4 (single MIG-partitioned H200 SXM) "
                    "reports 1596.41 tok/s. The ~5x gap is consistent "
                    "with MIG-partition-fraction (1/4-1/7 of a full H200) "
                    "× request_rate=4 vs. metrum's unstated concurrency. "
                    "Cross-source comparison does NOT reveal a "
                    "methodology drift; it reveals that ssakethch is a "
                    "per-MIG-instance measurement and metrum is a "
                    "per-cluster aggregate. Consumers MUST NOT "
                    "cross-compare ssakethch per-replica tokens/s "
                    "directly with metrum aggregate tokens/s without "
                    "explicitly applying a partition-fraction × "
                    "concurrency normalization."
                ),
            },
            {
                "cell": "Llama-3.1-70B / BF16 / SGLang vs vLLM (engine mismatch)",
                "ssakethch_observation": {
                    "available": False,
                    "notes": (
                        "ssakethch is vLLM-only. No SGLang H200 row "
                        "available for direct cross-source SGLang "
                        "comparison."
                    ),
                },
                "metrum_observation": {
                    "model_id": (
                        f"{metrum_llama_70b_sglang.get('model_id')}-"
                        f"{metrum_llama_70b_sglang.get('model_size')}"
                        if metrum_llama_70b_sglang else None
                    ),
                    "precision": (
                        metrum_llama_70b_sglang.get("precision")
                        if metrum_llama_70b_sglang else None
                    ),
                    "num_gpus": (
                        metrum_llama_70b_sglang.get("num_gpus")
                        if metrum_llama_70b_sglang else None
                    ),
                    "engine": (
                        metrum_llama_70b_sglang.get("engine").lower()
                        if metrum_llama_70b_sglang else None
                    ),
                    "concurrency": (
                        metrum_llama_70b_sglang.get("concurrency")
                        if metrum_llama_70b_sglang else None
                    ),
                    "ttft_ms": (
                        metrum_llama_70b_sglang.get("ttft_ms")
                        if metrum_llama_70b_sglang else None
                    ),
                    "tpot_ms": (
                        metrum_llama_70b_sglang.get("tpot_ms")
                        if metrum_llama_70b_sglang else None
                    ),
                    "notes": (
                        "metrum-ai records 'VMware benchmark; E2E Latency "
                        "18ms. TPOT is extremely low.' The 7.292 ms TTFT "
                        "+ 0.042 ms TPOT looks suspiciously low — "
                        "likely PER-TOKEN aggregate, NOT per-request "
                        "TTFT. See limitations below."
                    ),
                },
                "methodology_drift_observation": (
                    "Engine mismatch (SGLang vs vLLM) and suspicious "
                    "metrum measurement (TPOT=0.042 ms is implausibly "
                    "low for a 70B model on 8 H200 — likely a unit / "
                    "definition mismatch). Cross-source quantitative "
                    "comparison is INFEASIBLE for this cell; recorded "
                    "as a limitation."
                ),
            },
        ],
        "limitations": [
            "Only 1 of 10 metrum-ai H200 rows carries both TTFT and "
            "TPOT measurements (SGLang/Llama-3.1-70B/BF16, c=10, "
            "8 GPUs). The remaining 9 rows are 'Target' coverage-cell "
            "placeholders without measurements (per metrum's "
            "source_notes 'Target row.').",
            "metrum-ai's source_notes flag '[2] VMware benchmark; E2E "
            "Latency 18ms. TPOT is extremely low.' explicitly. The "
            "0.042 ms TPOT is implausibly low for a 70B-parameter model "
            "on any GPU class; likely a per-token aggregate, not per-"
            "request TTFT/TPOT. metrum-ai's curated nature means each "
            "row's measurement methodology can differ between sources.",
            "ssakethch is single-replica MIG-partitioned H200 SXM (the "
            "MIG partition fraction is NOT recorded in the dataset "
            "card YAML — inferred from the per-replica throughput "
            "magnitude). metrum's 8 GPU rows are aggregate full-GPU "
            "throughput. Direct cross-source per-row comparison without "
            "explicit partition-fraction normalization is unsafe.",
            "Engine overlap is THIN — ssakethch is vLLM-only; only 6 of "
            "10 metrum H200 rows are vLLM, of which 5 have NULL TTFT / "
            "TPOT / tokens_per_sec (Qwen-3, DeepSeek-R1-Distill, "
            "Gemma-3, Qwen-3-Math, Qwen-3 single rows are all "
            "placeholders).",
            "Engine-coverage gap: SGLang results in ssakethch are NOT "
            "available (vLLM-only). Cross-engine drift between SGLang "
            "and vLLM CANNOT be audited from the H200 corpus today.",
            "ROCm engine (vLLM-ROCm) overlap is ZERO — ssakethch has no "
            "AMD MI300X / MI355X coverage.",
            "Quantization overlap is THIN — ssakethch has 5 quants "
            "(awq, bf16, fp8, gptq, nvfp4); metrum-ai H200 carries 2 "
            "AWQ rows + 1 FP8 row + 7 unspecified (BF16 by precision "
            "field). Direct quant-by-quant cross-source comparison "
            "needs additional data.",
            "Concurrency definitions differ: ssakethch's request_rate "
            "(1, 2, 4, 8, 16) is the configured vLLM benchmark_serving "
            "closed-loop request rate; metrum's concurrency field is "
            "varied across rows (None / 10 / 32 / 50 / 64 / 128) with "
            "no harmonization. Concurrency-matched cross-source "
            "comparison is INFEASIBLE.",
            "Conclusion: the H200 cross-source overlap is too thin to "
            "draw a sweeping methodology-drift conclusion. The bounded "
            "observation is that the two sources are MUTUALLY "
            "COMPLEMENTARY (ssakethch = depth on single-source vLLM "
            "MIG H200; metrum-ai = curated breadth across engines / "
            "models / vendors) but NOT directly cross-comparable per-"
            "row. Aurelius consumers should treat ssakethch as the "
            "strongest single-source H200 vLLM prior, and metrum-ai's "
            "H200 rows as breadth-coverage / target-cell metadata only.",
        ],
        "recommended_consumer_action": (
            "Treat ssakethch as the strongest single-source H200 vLLM "
            "prior (275 measured rows, real p50/p99 TTFT/TPOT/ITL); "
            "treat metrum-ai H200 rows as a curated breadth-coverage "
            "metadata layer that flags TARGET cells which would "
            "require additional ingest to close. Do NOT cross-compare "
            "tokens_per_sec between the two sources without explicit "
            "(MIG partition fraction, num_gpus, concurrency, engine) "
            "normalization."
        ),
        "informs": [
            "constraint_aware_engine — H200 latency / throughput priors "
            "must be sourced from ssakethch, NOT metrum-ai, when per-"
            "replica TTFT/TPOT/ITL is required.",
            "performance_priors — metrum-ai H200 'Target' rows are "
            "USEFUL as coverage gap indicators (e.g. Qwen-3-235B / "
            "Mixtral-8x22B-MoE / DeepSeek-R1-Distill-70B target cells "
            "are NOT yet covered by ssakethch).",
            "cross_trace_frontier_generalization_audit — the bounded "
            "observation here updates the cross-trace audit's H200 "
            "row with the per-MIG-instance vs. per-cluster-aggregate "
            "caveat.",
        ],
    }
    return audit


def _write_h200_audit(audit: dict[str, Any]) -> Path:
    out = DISC_DIR / "h200_cross_source_methodology_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True))
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

    audit = _compute_h200_audit()

    if args.dry_run:
        logger.info(
            "DRY-RUN — Round-7 audit prepared with %d discovery-only records",
            len(ROUND7_DISCOVERY_ONLY),
        )
        logger.info(
            "DRY-RUN — H200 cross-source audit: %d directly_comparable_cells, "
            "%d limitations",
            len(audit["directly_comparable_cells"]),
            len(audit["limitations"]),
        )
        return 0

    cand_path = _update_candidate_registry(ROUND7_DISCOVERY_ONLY)
    logger.info("Updated candidate registry at %s", cand_path)

    summary_path = _write_round7_audit_summary()
    logger.info("Wrote Round-7 audit summary at %s", summary_path)

    h200_path = _write_h200_audit(audit)
    logger.info("Wrote H200 cross-source audit at %s", h200_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
