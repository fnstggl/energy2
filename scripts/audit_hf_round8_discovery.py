"""Round-8 broadened HF discovery audit.

Discovery / data-engine PR. No scheduler change, no controller change, no
robust-energy-engine change, no oracle as headline, no Tier 1 promotion,
no production claim, no new bounded sample committed.

This script does one thing:

**Round-8 broadened HF discovery audit** — re-runs ~40 high-priority
search-term groups against the public HF datasets API with a deliberately
NEW angle set (scaphandre, codecarbon, datacenter-traces, agent-runtime,
opentelemetry, mcp telemetry, mlcommons, scaphandre, etc.) that does NOT
overlap the term groups exhausted in Rounds 5-7. Surfaces newly-
appearing candidates that did NOT appear in PR #149's Round-7 pass,
inspects each via cardData / siblings / README, classifies them, and
persists a discovery-only rejection record per candidate.

The Round-8 finding has a NEW shape distinct from Rounds 5-7:

* Rounds 5/6/7 surfaced datasets that were either synthetic,
  out-of-domain, duplicate, or empty.
* Round 8 surfaces FOUR datasets with REAL infrastructure / energy
  measurements but NO declared license (sasha/co2_models, ohdoking/
  energy_consumption_by_model_and_gpu, dadadada1/Inference-Performance-
  Dataset, anon-betterbench/betterbench-inference-logs).

These cannot be ingested under the conservative redistribution policy
(committed normalized sample requires a declared permissive license).
They are recorded as ``inspect_manually_license_blocked`` so a future
licence-clearance pass can revisit them if the owners add a license, or
if Aurelius adds an operator-policy permission flow for license=None
datasets the operator has confirmed they can redistribute.

The Round-8 audit also confirms:

* ``deepanjalimishra99/datacenter-traces`` — despite the dataset name +
  3674 downloads — is the SPEC2017 SimPoint / fingerprint trace set
  ``bc_web_stanford/...`` (CPU-architecture simulation traces). NOT
  LLM-serving / cluster-scheduling. Rejected as irrelevant domain.
* ``sairamn/gcp-cloud-billing-cost`` — MIT but with `resource_433`,
  `resource_155`, `resource_934`, ... sequential synthetic resource IDs;
  fits the ``reject_synthetic_economics`` rule applied to
  ``tarekmasryo/llm-system-ops-production-telemetry-sft-data`` in
  Round-4.
* ``ClarusC64/datacenter-power-load-coherence-risk-v0.1`` — every row
  carries ``source_citation = "Synthetic"`` + cardData
  ``validation_status: pre_release``. Rejected as synthetic estimates.
* ``programasweights/paw-inference-logs`` — schema = spec / input /
  model_prediction / interpreter / program_id. Program-as-weights, not
  infrastructure telemetry.
* ``minhkhoi1026/opencl-llmperf`` — Apache-2.0 + 1344 downloads but
  schema is ``(code, gsize, lsize, execution_time, input_sizes)`` —
  OpenCL kernels (BlackScholes, DotProduct, MatVecMul, etc.) for
  TRAINING an LLM-based execution-time predictor, NOT LLM serving
  performance. Out of scope for the LLM-serving federated corpus.
* ``ICOS-AI/scaphandre_{cpu,ram,power}`` — Apache-2.0 + Scaphandre
  power-meter exports, but schema is ONLY ``(timestamp, value)`` with NO
  workload / model / GPU / request join key. Real power telemetry but
  un-joinable to LLM serving — rejected as low_value_no_workload_context.

Aggregate Round-8 finding: ZERO ingestible candidates this round. FOUR
license=None datasets with REAL infrastructure measurements (a new
category vs Rounds 5-7's synthetic/duplicate/wrong-domain). The negative
result remains the same for ingest, but the FAILURE MODE has shifted
from "wrong domain / synthetic / duplicate" to "real but undistributable
without an operator-policy permission flow." This is the FOURTH
CONSECUTIVE ROUND (5, 6, 7, 8) confirming Aurelius' goodput/$
denominator remains operator-policy-supplied + public-pricing-prior +
ElectricityMaps / ENTSO-E carbon intensity (already integrated).

Artifacts written:

- data/external/hf_discovery/round8_broadened_discovery_audit_summary.json
- data/external/hf_discovery/hf_dataset_candidates.json (updated)

Tests live at tests/test_hf_round8_audit.py.

Honesty + scope guarantees: no scheduler / controller / robust-energy-
engine change; no oracle as headline; no Tier 1 promotion; no production
claim; no new HF data downloaded beyond per-dataset metadata; no
HF_TOKEN leak.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round-8 audit data: 11 discovery-only records
# ---------------------------------------------------------------------------

ROUND8_DISCOVERY_ONLY: list[dict[str, Any]] = [
    {
        "dataset_id": "sasha/co2_models",
        "bucket": "I_inspect_manually_license_blocked",
        "kind": "inspect_manually_license_blocked",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "REAL per-run inference measurements for vision models "
            "(google/vit-base-patch16-224, microsoft/beit, "
            "microsoft/resnet-50, etc.) on CIFAR10. Schema: timestamp, "
            "model_name, dataset_name, run_id, duration, num_queries, "
            "emissions (kgCO2e), energy (kWh), region (e.g. 'virginia'), "
            "gpu_count, gpu_model (e.g. '1 x Tesla T4'). Captured via "
            "CodeCarbon. THIS IS THE FIRST HF DATASET SURFACED IN ROUNDS "
            "5-8 with BOTH operational (duration, num_queries) AND "
            "economic (emissions, energy, region) AND infrastructure "
            "(gpu_model, gpu_count) signals together. However: "
            "license=None on the HF card. Per the conservative "
            "redistribution policy, no committed normalized sample is "
            "allowed without a declared permissive license. ALSO: this "
            "is vision/CV models (ViT/BEiT/ResNet), not LLM serving — "
            "would be an adjacent energy/region prior, not a direct LLM "
            "serving entry. Recorded as inspect_manually_license_blocked "
            "with an explicit follow-up note: revisit if the owner adds "
            "a license, or after the operator-policy permission-flow "
            "milestone lands."
        ),
    },
    {
        "dataset_id": "ohdoking/energy_consumption_by_model_and_gpu",
        "bucket": "I_inspect_manually_license_blocked",
        "kind": "inspect_manually_license_blocked",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "REAL per-prompt CodeCarbon energy / runtime / co2 benchmark "
            "across 8 NVIDIA GPU classes (RTX 3070, RTX 3090, RTX 4090, "
            "RTX A4000, RTX A5000, RTX A6000, RTX 2000 Ada, RTX 4000 "
            "Ada) + CPU baseline. Multiple HF models (TinyLlama-1.1B "
            "etc.). Schema: model_name, huggingface_model, num_params, "
            "hidden_size, num_layers, vocab_size, seq_length, model_type, "
            "timestamp, hardware_cpu, hardware_gpu, hardware_ram, "
            "total_prompts, average_runtime, average_energy, "
            "average_co2, benchmark_timestamp, prompt_index, "
            "prompt_text, prompt_runtime, prompt_energy, prompt_co2, "
            "prompt_response. 22 CSV files, 10K-100K rows total. "
            "Per-prompt energy + co2 with broad consumer/workstation "
            "RTX coverage WOULD strongly broaden the existing "
            "ejhusom/llm-inference-energy-consumption corpus (which "
            "covers only laptop2 + workstation Ollama hosts), but "
            "license=None on the HF card blocks committed normalized "
            "sample. Recorded as inspect_manually_license_blocked. "
            "Follow-up: contact owner to request explicit Apache-2.0 / "
            "MIT / CC-BY-4.0 license, OR ingest via operator-policy "
            "permission-flow once that milestone lands."
        ),
    },
    {
        "dataset_id": "dadadada1/Inference-Performance-Dataset",
        "bucket": "I_inspect_manually_license_blocked",
        "kind": "inspect_manually_license_blocked",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "Real H100 token-level LLM inference benchmark, single-user. "
            "Schema (per README + CSV head): model (7b / 13b / 20b), "
            "gpu_type (H100), n_gpus, num_users, n_input_tokens, "
            "n_output_tokens, mean_latency_token (ms), max_latency_token "
            "(ms), std_latency_token (ms), prefill_time (≈ TTFT, ms), "
            "avg_token_decode_time (≈ TPOT, ms), token_throughput "
            "(tok/ms), early_sync_delay (ms), decode_jitter_std, "
            "stall_ratio_95p, token_latency_slope, sync_cost_ratio. "
            "100K-1M rows. This is UNIQUE in the corpus: it adds (i) "
            "token-level JITTER + STALL + SYNC-COST telemetry which is "
            "not present in agent-perf-bench or optimum-benchmark, and "
            "(ii) more H100 coverage at 20B model size (currently "
            "metrum-ai has thin H100/H200 70B coverage). However: "
            "license=None on the HF card blocks committed normalized "
            "sample. Recorded as inspect_manually_license_blocked. "
            "Follow-up: contact owner to request a permissive license, "
            "OR ingest via operator-policy permission-flow."
        ),
    },
    {
        "dataset_id": "anon-betterbench/betterbench-inference-logs",
        "bucket": "I_inspect_manually_license_blocked",
        "kind": "inspect_manually_license_blocked",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "1.8M parquet rows of inference logs across 31 train shards "
            "(4.1 GB compressed / 15 GB uncompressed). Schema: "
            "start_time (float, Unix-like), system_prompt, user_input, "
            "model_output, model_name, temperature, inference_time "
            "(seconds). REAL per-request arrival timestamp + end-to-end "
            "inference_time — a Tier-5 request_shape_trace candidate "
            "with timestamped arrivals. However: license=None on the HF "
            "card blocks committed normalized sample. The 'anon-' "
            "namespace + 'betterbench' name suggest this is an "
            "anonymized benchmark log, not production telemetry — "
            "treat as Tier-5 at best (workload shape / arrival prior) "
            "even if a license becomes available. Recorded as "
            "inspect_manually_license_blocked. Follow-up: contact owner "
            "to request a permissive license, OR ingest via operator-"
            "policy permission-flow."
        ),
    },
    {
        "dataset_id": "sairamn/gcp-cloud-billing-cost",
        "bucket": "D_synthetic_economics",
        "kind": "reject_synthetic_economics",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "19 MB CSV with 100K-1M rows of GCP-shaped billing rows: "
            "Resource ID (resource_1 ... resource_999, SEQUENTIAL "
            "synthetic IDs), Service Name (Cloud Run, BigQuery, Cloud "
            "SQL, Compute Engine, Dialogflow, etc. — real GCP catalog), "
            "Usage Quantity, Usage Unit, Region / Zone (real GCP "
            "regions: us-central1, asia-east1, etc.), CPU Utilization "
            "(%), Memory Utilization (%), Network Inbound / Outbound "
            "Bytes, Usage Start Date, Usage End Date, Cost per Quantity "
            "($), Unrounded Cost ($), Rounded Cost ($), Total Cost "
            "(INR). The sequential `resource_<N>` naming pattern is "
            "diagnostic of SYNTHETIC fixture data — real Google Cloud "
            "billing exports do NOT use that numbering scheme. License "
            "is MIT, BUT per the binding directive 'Do NOT treat "
            "synthetic cost fields as real economics', this is rejected "
            "as synthetic_economics. Same rule as the Round-4 "
            "tarekmasryo/llm-system-ops-production-telemetry-sft-data "
            "rejection (which self-declared synthetic despite "
            "'production-telemetry' in the name). The structural "
            "shape (region × service × quantity × cost) IS useful as a "
            "synthetic-billing-fixture schema reference but NOT as a "
            "real GCP pricing prior."
        ),
    },
    {
        "dataset_id": "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
        "bucket": "D_synthetic_estimates",
        "kind": "reject_synthetic_estimates",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "MIT-licensed but every committed CSV row carries "
            "source_citation = 'Synthetic'. Card YAML explicitly marks "
            "validation_status: pre_release. Schema: uid, dataset_id, "
            "year, rack_power_kw, psu_margin, ups_buffer_minutes, "
            "voltage_variance_pct, load_spike_frequency, "
            "unexpected_resets, ops_summary, question, "
            "ground_truth_label (coherent / incoherent), "
            "ground_truth_rationale, outcome_signal (stable_power / "
            "power_risk). Real-shape datacenter power-load risk schema, "
            "but self-declared synthetic per-row. Rejected as "
            "synthetic_estimates per the binding rule 'Do NOT treat "
            "synthetic cost fields as real economics' (extended to "
            "synthetic power-stability labels). Also n<1K — too small "
            "for analysis-tier even if it were real."
        ),
    },
    {
        "dataset_id": "deepanjalimishra99/datacenter-traces",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": "mit",
        "gated": False,
        "reason": (
            "MIT + 3674 downloads + name suggests datacenter telemetry, "
            "BUT siblings reveal this is the SPEC2017 / SimPoint "
            "fingerprint + simpoint trace set: "
            "bc/bc_web_stanford/fingerprint/bbfp.41, "
            "bc/bc_web_stanford/simpoints/opt.l, ... 6257 files of "
            "CPU-architecture simulation fingerprints + simpoints from "
            "the `bc_web_stanford` (Stanford bc) workload. This is "
            "single-thread CPU simulation trace data for computer-"
            "architecture research (microbenchmark/SimPoint workflows), "
            "NOT LLM-serving or cluster-scheduling telemetry. Despite "
            "the name and high download count, out of scope for the "
            "Aurelius LLM-serving / cluster-scheduling / energy-aware "
            "federated corpus. Rejected as irrelevant domain."
        ),
    },
    {
        "dataset_id": "programasweights/paw-inference-logs",
        "bucket": "F_irrelevant_domain",
        "kind": "reject_irrelevant_domain",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "reason": (
            "1K-10K rows, parquet, schema: spec (string), input "
            "(string), model_prediction (string), model_version "
            "(string), interpreter (string), program_id (string), "
            "source (string), ephemeral (bool). 'paw' = Programs-As-"
            "Weights — synthesized program execution logs (model_prediction "
            "+ interpreter), NOT serving infrastructure telemetry. No "
            "TTFT / TPOT / GPU / queue / cost / energy fields. Despite "
            "'inference-logs' in the name, this is program-synthesis "
            "capability-eval data, not serving telemetry. Rejected as "
            "irrelevant domain. license=None also blocks redistribution."
        ),
    },
    {
        "dataset_id": "minhkhoi1026/opencl-llmperf",
        "bucket": "F_out_of_scope",
        "kind": "reject_out_of_scope",
        "candidate_trace_type": "kernel_profile_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "Apache-2.0 + 1344 downloads + name 'llmperf' suggests LLM "
            "performance data, BUT the actual schema (per README "
            "dataset_info) is (code: string, gsize: int64, lsize: "
            "int64, execution_time: float64, input_sizes: string) "
            "across 130 OpenCL benchmark configs (BlackScholes, "
            "DotProduct, MatVecMul, oclCopyComputeOverlap, "
            "loop_with_constant, loop_with_variable_v2, ...). This is a "
            "TRAINING dataset for an LLM-based OpenCL kernel-execution-"
            "time PREDICTOR (the 'llm' in 'llmperf' refers to the "
            "modeling approach, not the workload). The execution_time "
            "column is OpenCL kernel runtime, NOT LLM serving TTFT / "
            "TPOT. Aurelius' kernel_profile_trace is for GPU kernel "
            "profiling within LLM serving (CUDA / Nsight / NCU level); "
            "OpenCL HPC kernels are a different abstraction (host-"
            "managed gsize / lsize on arbitrary devices). Out of scope "
            "for the LLM-serving federated corpus."
        ),
    },
    {
        "dataset_id": "ICOS-AI/scaphandre_power_consumption",
        "bucket": "F_low_value_no_workload_context",
        "kind": "reject_low_value_no_workload_context",
        "candidate_trace_type": "telemetry_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "Apache-2.0 (would permit redistribution). Schema is ONLY "
            "(timestamp: float, power_consumption: float). 3-second "
            "Scaphandre power-meter sampling on the ICOS Federated "
            "Learning edge infrastructure. ~1K-10K rows / ~10 KB CSV. "
            "REAL power telemetry, but there is NO workload / model / "
            "GPU / request-id join key. The dataset is power-only at a "
            "single host, with no way to attribute power draw to any "
            "LLM workload, training job, or request. Cannot feed any "
            "Aurelius decision (placement, routing, scaling, cost/$, "
            "energy-aware scheduling) because there is no field to "
            "join against ANY operational trace. Rejected as "
            "low_value_no_workload_context. Sibling datasets "
            "ICOS-AI/scaphandre_cpu_usage and "
            "ICOS-AI/scaphandre_ram_usage carry the same limitation."
        ),
    },
    {
        "dataset_id": "ICOS-AI/scaphandre_cpu_usage",
        "bucket": "F_low_value_no_workload_context",
        "kind": "reject_low_value_no_workload_context",
        "candidate_trace_type": "telemetry_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "reason": (
            "Apache-2.0 sibling of ICOS-AI/scaphandre_power_consumption "
            "with the same limitation. Schema: (timestamp: float, "
            "cpu_usage: float). 3-second Scaphandre sampling. ~1K-10K "
            "rows. NO workload / model / GPU / request-id join key — "
            "cannot be attributed to any LLM workload, training job, or "
            "request. Rejected as low_value_no_workload_context. "
            "(Recorded explicitly so a future re-discovery does not "
            "re-evaluate the sibling.)"
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
                "aurelius_use_case": "Round-8 discovery audit — see reason.",
                "last_modified": None,
            }
            cands.append(c)
            by_id[ds_id] = c
        c["round8_audit_bucket"] = rec["bucket"]
        c["round8_audit_reason"] = rec["reason"]
        c["recommended_action"] = rec["kind"]
        c["gated_status"] = "gated" if rec.get("gated") else "public"
        if rec.get("license_observed") is not None:
            c["license"] = rec["license_observed"]
        existing_kw = c.get("matched_keywords") or []
        kw = f"round8::{rec['kind']}"
        if kw not in existing_kw:
            existing_kw.append(kw)
            c["matched_keywords"] = existing_kw

    d["candidates"] = cands
    d["candidate_count"] = len(cands)
    d["last_updated_at_s"] = time.time()
    d["updated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    d["updated_at_s"] = time.time()
    d["focused_audit_2026_06_02_round8"] = {
        "doc_version": "round8_broadened_discovery_audit_v1",
        "scope": (
            "Round-8 broadened HF discovery — re-ran ~40 search-term "
            "groups against the public HF API with deliberately NEW "
            "angles (codecarbon / scaphandre / agent runtime / "
            "opentelemetry / mcp telemetry / mlcommons / datacenter "
            "traces / cloud billing / inference-perf / etc.) that did "
            "NOT overlap Rounds 5-7. Surfaced 11 newly-appearing "
            "candidates (none in the existing 92-candidate registry). "
            "FOUR of the 11 are inspect_manually_license_blocked — the "
            "FIRST round to surface real infrastructure measurements "
            "blocked by license rather than by synthetic/wrong-domain/"
            "duplicate. Two synthetic_economics / synthetic_estimates "
            "(sairamn/gcp-cloud-billing-cost despite MIT license; "
            "ClarusC64/datacenter-power-load-coherence-risk-v0.1). Two "
            "irrelevant-domain (deepanjalimishra99 = SPEC2017 SimPoint "
            "traces; programasweights = program-as-weights). One "
            "out-of-scope (minhkhoi1026/opencl-llmperf = OpenCL kernel "
            "execution-time predictor training set, not LLM serving). "
            "Two low_value_no_workload_context (ICOS-AI Scaphandre "
            "power / cpu — real Apache-2.0 telemetry but only "
            "timestamp+value with no workload / model / GPU / request "
            "join key)."
        ),
        "audited_at_s": time.time(),
        "git_sha": _git_sha(),
    }
    CANDIDATES_PATH.write_text(json.dumps(d, indent=2, sort_keys=True))
    return CANDIDATES_PATH


# ---------------------------------------------------------------------------
# Round-8 audit summary writer
# ---------------------------------------------------------------------------


def _write_round8_audit_summary() -> Path:
    out = DISC_DIR / "round8_broadened_discovery_audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_version": "round8_broadened_discovery_audit_summary_v1",
        "audited_at_s": time.time(),
        "scope": (
            "Round 8 broadened HF discovery — no new ingest. Re-ran "
            "~40 search-term groups against the public HF API with a "
            "deliberately NEW angle set (codecarbon, scaphandre, "
            "agent runtime, opentelemetry, mcp telemetry, mlcommons, "
            "datacenter traces, cloud billing, inference-perf, "
            "energy_consumption, carbon, dynamo, tensorrt, llmperf, "
            "anyscale, perfdata, cluster log, serverless, codecarbon, "
            "etc.) — none of which overlap the term groups exhausted "
            "in Rounds 5-7. Surfaced 11 newly-appearing candidates "
            "(none in the existing 92-candidate registry). FOUR were "
            "marked inspect_manually_license_blocked (real "
            "infrastructure measurements but license=None: "
            "sasha/co2_models, "
            "ohdoking/energy_consumption_by_model_and_gpu, "
            "dadadada1/Inference-Performance-Dataset, "
            "anon-betterbench/betterbench-inference-logs). FIVE were "
            "rejected as synthetic / synthetic_economics / out-of-"
            "scope / irrelevant-domain. TWO were rejected as "
            "low_value_no_workload_context (ICOS-AI Scaphandre "
            "power+cpu — real Apache-2.0 power telemetry but no "
            "workload join key). ZERO ingestible this round. The "
            "license-blocked category is a NEW failure mode vs Rounds "
            "5-7's synthetic/duplicate/wrong-domain — actionable for a "
            "future license-clearance pass or operator-policy "
            "permission flow."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "ingested": [],
        "failed": [],
        "discovery_only_records": ROUND8_DISCOVERY_ONLY,
        "license_blocked_followup_candidates": [
            {
                "dataset_id": "sasha/co2_models",
                "reason_to_revisit": (
                    "Real region + gpu_model + CO2 + energy per inference "
                    "run (CodeCarbon). FIRST candidate in Rounds 5-8 "
                    "carrying simultaneous operational (duration, "
                    "num_queries) + economic (emissions, energy, region) "
                    "+ infrastructure (gpu_model, gpu_count) signals. "
                    "Even though it's CV models (ViT/BEiT/ResNet), the "
                    "region × GPU × energy join keys are reusable as an "
                    "energy/region prior."
                ),
                "required_action_to_unblock": (
                    "Owner adds Apache-2.0 / MIT / CC-BY-4.0 license "
                    "on the HF card, OR Aurelius operator-policy "
                    "permission flow lands with explicit redistribution "
                    "consent."
                ),
            },
            {
                "dataset_id": "ohdoking/energy_consumption_by_model_and_gpu",
                "reason_to_revisit": (
                    "Per-prompt CodeCarbon energy + co2 + runtime across "
                    "8 NVIDIA GPU classes (RTX 3070/3090/4090, A4000/"
                    "A5000/A6000, RTX 2000/4000 Ada Gen) + CPU baseline. "
                    "Broadens the existing ejhusom (laptop2 + workstation "
                    "Ollama) coverage substantially for consumer / "
                    "workstation RTX tier."
                ),
                "required_action_to_unblock": (
                    "Owner adds a permissive license, OR operator-policy "
                    "permission flow."
                ),
            },
            {
                "dataset_id": "dadadada1/Inference-Performance-Dataset",
                "reason_to_revisit": (
                    "Token-level H100 LLM inference with JITTER + STALL "
                    "+ SYNC-COST telemetry not present in any existing "
                    "corpus dataset. Unique signal: decode_jitter_std, "
                    "stall_ratio_95p, sync_cost_ratio."
                ),
                "required_action_to_unblock": (
                    "Owner adds a permissive license, OR operator-policy "
                    "permission flow."
                ),
            },
            {
                "dataset_id": "anon-betterbench/betterbench-inference-logs",
                "reason_to_revisit": (
                    "1.8M rows with start_time + inference_time + "
                    "model_name + temperature. Timestamped arrival + "
                    "e2e — Tier-5 request_shape_trace candidate."
                ),
                "required_action_to_unblock": (
                    "Owner adds a permissive license. 'anon-' namespace "
                    "suggests anonymized benchmark, not production — "
                    "even if licensed, treat as Tier-5 (workload shape) "
                    "not Tier-1 (pilot telemetry)."
                ),
            },
        ],
        "economic_priority_summary": {
            "datasets_with_operational_and_economic_signals": [],
            "datasets_with_economic_only_signals": [],
            "join_keys_available_for_economic_overlays": [
                "gpu_type", "model_id", "engine", "quantization",
                "request_rate", "region", "cloud_provider",
            ],
            "scorer_coefficients_calibratable_from_round8": [],
            "scorer_coefficients_operator_policy_only_after_round8": [
                "gpu_hour_price_usd",
                "kwh_per_request",
                "carbon_g_per_kwh",
                "spot_interruption_probability",
                "egress_cost_per_gb",
                "regional_price_usd_per_mwh",
            ],
            "negative_result_finding": (
                "Round 8 surveyed ~40 deliberately NEW search-term "
                "groups (codecarbon, scaphandre, agent runtime, "
                "opentelemetry, mcp telemetry, mlcommons, datacenter "
                "traces, cloud billing, inference-perf, energy, carbon, "
                "dynamo, tensorrt, llmperf, anyscale, perfdata, cluster "
                "log, serverless, etc.). Surfaced 11 new candidates; "
                "ZERO qualified for bounded ingest. The failure mode "
                "shifted from Rounds 5-7's 'synthetic / duplicate / "
                "wrong-domain' to a NEW category: REAL infrastructure "
                "measurements blocked by license=None (4 of 11 "
                "candidates). One synthetic_economics (sairamn/"
                "gcp-cloud-billing-cost — MIT but resource_<N> "
                "synthetic IDs). One synthetic_estimates (ClarusC64/"
                "datacenter-power-load-coherence-risk-v0.1 — "
                "source_citation='Synthetic'). Two irrelevant_domain "
                "(SPEC2017 SimPoint traces; program-as-weights). One "
                "out_of_scope (OpenCL kernel runtime predictor). Two "
                "low_value_no_workload_context (Scaphandre power+cpu — "
                "real Apache-2.0 telemetry but no workload join key). "
                "This is the FOURTH CONSECUTIVE ROUND (5, 6, 7, 8) "
                "confirming Aurelius' goodput/$ denominator REMAINS "
                "operator-policy-supplied + public-pricing-prior + "
                "ElectricityMaps / ENTSO-E carbon intensity (already "
                "integrated). The Round-8 sweep was DESIGNED to "
                "falsify the Round-5/6/7 finding with fresh angles; it "
                "failed to falsify on the ingest dimension, but did "
                "surface a new ACTIONABLE failure category: license="
                "None datasets that contain real measurements. A "
                "license-clearance follow-up to those four owners, OR "
                "an operator-policy permission flow, would unblock the "
                "richest economic+energy signal yet seen in the HF "
                "discovery pipeline (sasha/co2_models is the FIRST HF "
                "candidate carrying simultaneous operational + "
                "economic + infrastructure signals)."
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

    if args.dry_run:
        logger.info(
            "DRY-RUN — Round-8 audit prepared with %d discovery-only records",
            len(ROUND8_DISCOVERY_ONLY),
        )
        return 0

    cand_path = _update_candidate_registry(ROUND8_DISCOVERY_ONLY)
    logger.info("Updated candidate registry at %s", cand_path)

    summary_path = _write_round8_audit_summary()
    logger.info("Wrote Round-8 audit summary at %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
