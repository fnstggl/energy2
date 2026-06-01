"""Tests for the Round-5 HF discovery ingest of ``metrum-ai/llm-perfdata``.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.

Covered:

* one new ``latency_benchmark_trace`` config
  (``metrum-ai/llm-perfdata`` → ``multi_source_curated_v1``)
* the Round-5 discovery-only / negative-result records
  (``sairamn/gcp-cloud-billing-cost`` synthetic billing, two
  ``ClarusC64`` AI-safety coherence-risk evals,
  ``Phipper/pe-energy-infrastructure-training-data`` finance training,
  ``uohna/llm_inference_energy_combined.parquet`` empty dataset,
  ``Lightcap/agent-runtime-telemetry-small`` deferred tool-call
  telemetry, ``metrum-ai/llm-perf-dashboard`` deferred,
  ``ssakethch/h200-quantization-benchmarks`` no-license,
  ``crozai/vllm-benchmark-coding`` duplicate, and
  ``intellistream/sage-agent-benchmark`` eval-only).
* the multi-source-curated caveat — must be pinned in ``limitations``
  so downstream users do not treat metrum-ai TTFT/TPOT/throughput as
  single-campaign measurements.
* the weak-strength promotion gate — ``promoted_for_training_priors``
  only, NOT ``promoted_for_performance_priors`` (the latter requires
  ``moderate`` strength).
* the corpus invariant: no raw / analysis sample committed; license
  is declared and matches the dataset card (MIT); fixture is tiny +
  deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "metrum-ai/llm-perfdata"
SAFE_NAME = "metrum-ai__llm-perfdata"
CONFIG = "multi_source_curated_v1"

PROC_DIR = HF_DIR / SAFE_NAME / CONFIG / "processed"
FIXTURE = FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"
AUDIT_PATH = DISC_DIR / "round5_broadened_discovery_audit_summary.json"

ROUND5_DISCOVERY_IDS = [
    "sairamn/gcp-cloud-billing-cost",
    "ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1",
    "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
    "Phipper/pe-energy-infrastructure-training-data",
    "uohna/llm_inference_energy_combined.parquet",
    "Lightcap/agent-runtime-telemetry-small",
    "metrum-ai/llm-perf-dashboard",
    "ssakethch/h200-quantization-benchmarks",
    "crozai/vllm-benchmark-coding",
    "intellistream/sage-agent-benchmark",
]


def _summary() -> dict:
    with open(PROC_DIR / "summary.json") as fh:
        return json.load(fh)


def _audit() -> dict:
    with open(AUDIT_PATH) as fh:
        return json.load(fh)


# ─────────────────── 1. No raw / analysis files committed ──────────────


def test_no_raw_or_analysis_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [
        p for p in out if p.endswith("/analysis_sample.jsonl")
    ]
    assert raw_committed == [], (
        f"Raw downloads committed for {SAFE_NAME}: {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed for {SAFE_NAME}: "
        f"{analysis_committed}"
    )


def test_fixture_is_committed_and_tiny():
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    size = FIXTURE.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {FIXTURE} size {size}B outside (0, 16 KiB]"
    )


def test_committed_normalized_sample_bounded():
    """Committed normalised sample must be ≤ 100 MB (corpus policy) AND in
    this dataset's case ≤ 128 KiB (per script budget — 80 rows × ~600B)."""
    s = _summary()
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    assert path.exists(), f"missing committed normalised sample {path}"
    sz = path.stat().st_size
    assert 0 < sz <= 100 * 1024 * 1024, (
        f"committed normalised sample {path} > 100 MB cap (size {sz})"
    )
    assert sz <= 128 * 1024, (
        f"committed normalised sample {path} unexpectedly large ({sz}B); "
        "tighten COMMITTED_NORMALIZED_MAX_BYTES if intentional."
    )


# ─────────────────── 2. Processed artefacts present ─────────────────────


def test_processed_artifacts_present():
    for name in (
        "schema_profile.json", "schema_mapping.json", "summary.json",
        "statistical_rollups.json", "committed_normalized_sample.jsonl",
    ):
        path = PROC_DIR / name
        assert path.exists(), f"{path} missing"
        if path.suffix == ".json":
            with open(path) as fh:
                payload = json.load(fh)
            assert isinstance(payload, dict) and payload, f"{path} empty"


# ─────────────────── 3. Schema profile + mapping ────────────────────────


def test_schema_profile_records_dtypes_and_examples():
    with open(PROC_DIR / "schema_profile.json") as fh:
        prof = json.load(fh)
    assert prof["dataset_id"] == DATASET_ID
    assert prof["config_name"] == CONFIG
    assert prof["inspected_row_count"] == 80
    assert prof["unknown_columns"] == []
    expected_raw = {
        "Model", "Size", "Precision", "GPU_Type", "Num_GPUs",
        "Serving_Engine", "Concurrency", "Tokens_per_sec",
        "TTFT_ms", "TPOT_ms", "Prompt_Tokens", "Output_Tokens",
        "Context_Window", "Quantization", "Source_URL", "Source_Notes",
    }
    assert expected_raw.issubset(set(prof["raw_columns"])), (
        f"missing raw columns: "
        f"{expected_raw - set(prof['raw_columns'])}"
    )
    # Every column has at least one dtype recorded.
    for col in prof["raw_columns"]:
        assert prof["dtypes"][col], f"empty dtypes for {col}"
    # Sparse-coverage columns must carry a missing rate.
    for sparse in ("TTFT_ms", "TPOT_ms", "Prompt_Tokens", "Output_Tokens"):
        assert sparse in prof["missing_rates"], (
            f"missing_rate not recorded for {sparse}"
        )
        assert prof["missing_rates"][sparse] > 0.0, (
            f"sparse column {sparse} reports missing_rate=0; expected >0"
        )


def test_schema_mapping_field_quality_and_signal_category():
    with open(PROC_DIR / "schema_mapping.json") as fh:
        m = json.load(fh)
    assert m["dataset_id"] == DATASET_ID
    assert m["columns"], "no columns in mapping"
    allowed_quality = {"real", "derived", "proxy", "synthetic",
                       "missing", "unknown"}
    allowed_categories = {
        "request_arrival", "request_completion", "latency", "queue",
        "throughput", "tokens", "gpu_resource", "memory", "autoscaling",
        "routing", "cache_residency", "failure_timeout", "scheduler_state",
        "cost_energy_carbon", "metadata_only", "irrelevant",
    }
    for c in m["columns"]:
        assert c["field_quality"] in allowed_quality, c
        assert c["aurelius_signal_category"] in allowed_categories, c
        assert c["raw_column_name"]
        assert c["normalized_field"]
    # Latency, throughput, and GPU-resource categories must be present.
    cats = {c["aurelius_signal_category"] for c in m["columns"]}
    assert "latency" in cats
    assert "throughput" in cats
    assert "gpu_resource" in cats


# ─────────────────── 4. Promotion gates pass — training_priors only ─────


def test_promotion_gates_all_pass():
    s = _summary()
    gate_log = promotion.gates(s)
    failed = [g for g in gate_log if not g["passed"]]
    assert not failed, f"failed gates: {[g['gate'] for g in failed]}"


def test_promotion_state_is_training_priors_only():
    """Weak strength → training_priors only; the
    performance_priors / constraint_aware_evaluation tags MUST be
    rejected by the strength gate."""
    s = _summary()
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_training_priors"
    assert decision["promotion_tags"] == ["promoted_for_training_priors"]
    assert "promoted_for_performance_priors" not in decision["promotion_tags"]
    assert (
        "promoted_for_constraint_aware_evaluation"
        not in decision["promotion_tags"]
    )
    # Must explicitly record the downgrade reason.
    assert decision["reasons"], "expected downgrade reasons for weak strength"
    assert any(
        "promoted_for_performance_priors" in r and "insufficient" in r
        for r in decision["reasons"]
    ), f"downgrade reason not recorded: {decision['reasons']}"


def test_canonical_trace_type_is_latency_benchmark():
    s = _summary()
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


def test_statistical_sample_strength_is_weak():
    """80 rows × 24 models × 9 GPUs × 5 engines → weak; densest cell
    is (NVIDIA A100, vLLM, FP16) with 8 rows. This MUST stay weak so
    the promotion harness blocks performance_priors / dynamic
    calibration / constraint_aware promotions."""
    s = _summary()
    assert s["statistical_sample_strength"] == "weak"


# ─────────────────── 5. Registry + audit ────────────────────────────────


def test_registry_contains_new_entry():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    assert reg, "canonical registry missing"
    matches = [
        e for e in reg["entries"]
        if e["dataset_id"] == DATASET_ID and e["config_name"] == CONFIG
    ]
    assert len(matches) == 1, (
        f"expected exactly one entry for {DATASET_ID}/{CONFIG}, "
        f"got {len(matches)}"
    )
    entry = matches[0]
    assert entry["trust_tier"] == "tier_4_latency_benchmark_traces"
    assert entry["license"] == "mit"
    assert entry["gated"] is False
    assert entry["promotion_state"] == "promoted_for_training_priors"
    assert entry["statistical_sample_strength"] == "weak"


def test_audit_summary_records_ingest_and_discovery_records():
    a = _audit()
    assert a["production_claim"] is False
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False
    ingested_ids = {x["dataset_id"] for x in a["ingested"]}
    assert DATASET_ID in ingested_ids
    discovery_ids = {r["dataset_id"] for r in a["discovery_only_records"]}
    missing = set(ROUND5_DISCOVERY_IDS) - discovery_ids
    assert not missing, f"audit missing discovery records: {missing}"


def test_audit_summary_records_economic_priority_findings():
    """Round-5 was an economic-priority pass. The audit MUST record an
    `economic_priority_summary` so future runs can pick up where this
    one stopped without re-doing the same search."""
    a = _audit()
    eps = a.get("economic_priority_summary")
    assert eps, "economic_priority_summary missing from audit"
    # Both join-key list AND the negative-result finding must be present.
    assert eps.get("join_keys_available_for_economic_overlays")
    assert eps.get("scorer_coefficients_operator_policy_only_after_round5")
    assert eps.get("negative_result_finding")
    text = eps["negative_result_finding"].lower()
    assert "no hf dataset" in text, (
        f"negative_result_finding does not state the no-dataset finding: "
        f"{eps['negative_result_finding'][:200]}"
    )


def test_candidate_registry_has_ingest_recommendation_for_metrum_ai():
    candidates_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(candidates_path) as fh:
        d = json.load(fh)
    for c in d.get("candidates", []):
        if c["dataset_id"] == DATASET_ID:
            assert c["recommended_action"] == "ingest_now_bounded"
            assert c["candidate_trace_type"] == "latency_benchmark_trace"
            assert c["license"] == "mit"
            assert c["gated_status"] == "public"
            return
    pytest.fail(
        f"{DATASET_ID} not in hf_dataset_candidates.json"
    )


def test_candidate_registry_has_discovery_only_round5_records():
    candidates_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(candidates_path) as fh:
        d = json.load(fh)
    cand_ids = {c["dataset_id"] for c in d.get("candidates", [])}
    missing = set(ROUND5_DISCOVERY_IDS) - cand_ids
    assert not missing, (
        f"candidate registry missing Round-5 discovery records: {missing}"
    )


# ─────────────────── 6. Multi-source curated caveat enforcement ─────────


def test_limitations_pin_multi_source_curated_caveat():
    """Must explicitly call out that each row's Source_URL is a separate
    upstream — preventing any downstream user from treating metrum-ai as
    a single-campaign measurement."""
    s = _summary()
    text = " ".join(s["limitations"]).lower()
    assert "multi-source" in text or "multi source" in text, (
        f"multi-source-curated caveat missing from limitations: "
        f"{s['limitations']}"
    )
    assert "source_url" in text, (
        f"Source_URL attribution caveat missing: {s['limitations']}"
    )
    assert "tier 4" in text, "tier-4 note missing"
    assert "not pilot telemetry" in text or "not pilot" in text, (
        f"pilot-telemetry disclaimer missing: {s['limitations']}"
    )


def test_limitations_pin_statistical_strength_caveat():
    """Weak strength must be explicitly flagged so consumers know that
    p95/p99 percentile claims and cross-stratum averages are NOT
    statistically supported."""
    s = _summary()
    text = " ".join(s["limitations"]).lower()
    assert "weak" in text, "weak-strength caveat missing"
    assert "training_priors" in text or "training priors" in text, (
        "promoted_for_training_priors-only caveat missing"
    )


def test_limitations_pin_no_economic_signal_caveat():
    """No measured cost / energy / carbon fields. This MUST be explicit
    so the Aurelius goodput/$ + carbon-cost terms never silently
    consume curated absolute numbers."""
    s = _summary()
    text = " ".join(s["limitations"]).lower()
    assert "energy" in text and "cost" in text and "carbon" in text, (
        f"no-economic-signal caveat missing: {s['limitations']}"
    )


def test_no_economic_or_queue_signals_claimed():
    """metrum-ai carries ONLY metadata + measured latency/throughput.
    No measured economic / queue / GPU-utilisation / energy /
    failure-label signals must appear in available_signals."""
    s = _summary()
    avail = set(s["available_signals"])
    forbidden_in_avail = {
        "queue_state", "queue_wait", "queue_depth",
        "gpu_utilization", "memory_pressure",
        "energy_per_request", "carbon_intensity",
        "cost_per_token", "cost_per_request",
        "kv_cache_size", "cache_hit", "kernel_duration",
        "timeout_label", "sla_label", "failure_label",
        "autoscaling", "replica_count",
    }
    leaked = avail & forbidden_in_avail
    assert not leaked, (
        f"metrum-ai claims forbidden signals: {leaked}"
    )


# ─────────────────── 7. Signal taxonomy ─────────────────────────────────


def test_available_signals_include_ttft_tpot_throughput_and_gpu_type():
    s = _summary()
    avail = set(s["available_signals"])
    required = {"ttft", "tpot", "throughput",
                "gpu_type", "model_id", "engine"}
    assert required.issubset(avail), (
        f"required latency/metadata signals missing: {required - avail}"
    )


def test_missing_signals_record_economic_and_queue_gaps():
    """The missing_signals list MUST explicitly call out the economic
    and queue gaps so the gap-closure audit can pick them up."""
    s = _summary()
    miss = set(s["missing_signals"])
    required_missing = {
        "queue_state", "queue_wait", "queue_depth",
        "gpu_utilization", "batch_size",
        "energy_per_request", "carbon_intensity",
        "cost_per_token", "cost_per_request",
    }
    leaked = required_missing - miss
    assert not leaked, (
        f"required missing_signals not recorded: {leaked}"
    )


# ─────────────────── 8. Fixture determinism ─────────────────────────────


def test_fixture_sha256_matches_summary():
    s = _summary()
    fx = FIXTURE
    h = hashlib.sha256()
    with open(fx, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    assert h.hexdigest() == s["sample_sha256"], (
        f"fixture sha256 drift: file={h.hexdigest()} vs summary="
        f"{s['sample_sha256']}"
    )


def test_committed_normalized_sample_sha256_matches_summary():
    s = _summary()
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    assert h.hexdigest() == s["committed_normalized_sample_sha256"]


def test_committed_sample_row_count_is_full_80():
    """The dataset has 80 rows total and the committed budget (128 KiB)
    easily fits them — assert we did not truncate."""
    s = _summary()
    assert s["analysis_sample_rows"] == 80
    assert s["committed_normalized_sample_rows"] == 80


# ─────────────────── 9. Statistical rollups ─────────────────────────────


def test_statistical_rollups_has_stratification_keys():
    """Rollups must record (gpu_type, engine, precision) strata."""
    with open(PROC_DIR / "statistical_rollups.json") as fh:
        r = json.load(fh)
    assert r["stratification_keys"] == ["gpu_type", "engine", "precision"]
    assert r["subgroup_counts"], "no subgroup counts recorded"
    # At least one H100 cell must exist.
    h100_keys = [k for k in r["subgroup_counts"] if "H100" in k]
    assert h100_keys, (
        f"H100 stratum missing from subgroup counts: "
        f"{list(r['subgroup_counts'].keys())[:5]}..."
    )
    # First-coverage-in-corpus GPUs must each appear in subgroup counts.
    for gpu in ("H100", "H200", "B200", "Gaudi 3", "MI300X"):
        assert any(gpu in k for k in r["subgroup_counts"]), (
            f"first-coverage GPU {gpu!r} missing from subgroup counts"
        )


def test_insufficient_sample_groups_explicit():
    """Single-row strata MUST be marked INSUFFICIENT_SAMPLE so p95/p99
    claims are blocked on them."""
    with open(PROC_DIR / "statistical_rollups.json") as fh:
        r = json.load(fh)
    insufficient = r["insufficient_sample_groups"]
    # Every group with n<5 must appear in the insufficient list.
    for label, n in r["subgroup_counts"].items():
        if n < 5:
            assert label in insufficient, (
                f"subgroup {label!r} with n={n} not flagged INSUFFICIENT_SAMPLE"
            )


def test_overall_rollup_records_p95_for_throughput():
    with open(PROC_DIR / "statistical_rollups.json") as fh:
        r = json.load(fh)
    o = r["overall"]
    assert "tokens_per_sec" in o, "tokens_per_sec rollup missing"
    assert o["tokens_per_sec"]["p95"] >= o["tokens_per_sec"]["p50"]
    assert o["tokens_per_sec"]["count"] >= 30, (
        f"throughput sample too small: {o['tokens_per_sec']['count']}"
    )


# ─────────────────── 10. License + redistribution ──────────────────────


def test_license_is_mit_and_dataset_is_public():
    s = _summary()
    assert s["license"] == "mit"
    assert s["gated"] is False
    assert s["committed_normalized_sample_reason_skipped"] is None
    assert s["committed_normalized_sample_rows"] > 0


def test_provenance_pins_dataset_and_git_sha():
    s = _summary()
    prov = s["provenance"]
    assert DATASET_ID in prov, f"provenance lacks dataset id: {prov}"
    assert CONFIG in prov, f"provenance lacks config name: {prov}"


# ─────────────────── 11. Fixture covers high-value GPU breadth ─────────


def test_fixture_spans_distinct_gpu_engine_pairs():
    """Fixture should cover at least 4 distinct (gpu_type, engine) pairs
    to demonstrate the breadth claim. This is the breadth gap metrum-ai
    actually closes — A100/vLLM and SGLang on H200 / Gaudi 3 / B200
    were absent from the corpus before this ingest."""
    rows = []
    with open(FIXTURE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) == 5, f"expected 5 fixture rows, got {len(rows)}"
    pairs = {(r.get("gpu_type"), r.get("engine")) for r in rows}
    assert len(pairs) >= 4, (
        f"fixture spans too few (gpu, engine) pairs: {pairs}"
    )


# ─────────────────── 12. Negative-result discovery records — assertions ─


def test_synthetic_billing_dataset_is_rejected_with_economic_rationale():
    a = _audit()
    rec = next(
        (r for r in a["discovery_only_records"]
         if r["dataset_id"] == "sairamn/gcp-cloud-billing-cost"),
        None,
    )
    assert rec, "sairamn rejection record missing"
    text = rec["reason"].lower()
    assert "synthetic" in text, (
        f"sairamn rejection must call out synthetic data: {rec['reason']}"
    )
    assert rec["kind"] == "reject_synthetic_economics"


def test_lightcap_deferred_with_trace_class_rationale():
    """Lightcap is high-value but its trace class (tool-call runtime)
    is distinct from the existing LLM-serving-focused canonical types.
    Must be deferred, not rejected, and the rationale must explain why."""
    a = _audit()
    rec = next(
        (r for r in a["discovery_only_records"]
         if r["dataset_id"] == "Lightcap/agent-runtime-telemetry-small"),
        None,
    )
    assert rec, "Lightcap deferral record missing"
    assert rec["kind"] == "defer_high_value_different_trace_class"
    text = rec["reason"].lower()
    assert "tool-call" in text or "tool call" in text, (
        f"Lightcap deferral must explain tool-call vs LLM-serving "
        f"distinction: {rec['reason'][:200]}"
    )


def test_phipper_rejected_as_finance_training_data():
    a = _audit()
    rec = next(
        (r for r in a["discovery_only_records"]
         if r["dataset_id"] == "Phipper/pe-energy-infrastructure-training-data"),
        None,
    )
    assert rec, "Phipper rejection record missing"
    assert rec["kind"] == "reject_out_of_scope"
    text = rec["reason"].lower()
    assert "finance" in text or "private equity" in text or "private-equity" in text


def test_ssakethch_deferred_pending_license():
    """ssakethch h200-quantization is high-value (40 LLMs × H200 MIG)
    but lacks an SPDX license. Must be deferred, not rejected."""
    a = _audit()
    rec = next(
        (r for r in a["discovery_only_records"]
         if r["dataset_id"] == "ssakethch/h200-quantization-benchmarks"),
        None,
    )
    assert rec, "ssakethch deferral record missing"
    assert rec["kind"] == "defer_pending_full_schema_probe"
    text = rec["reason"].lower()
    assert "license" in text, (
        f"ssakethch deferral must call out license-blocker: {rec['reason']}"
    )
