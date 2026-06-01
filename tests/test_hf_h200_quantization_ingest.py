"""Tests for the round-6 HF discovery — ssakethch/h200-quantization-benchmarks.

Covers:

* Bounded-ingest artefacts (schema profile, schema mapping, summary,
  statistical rollups, fixture) for the new Tier-4 latency_benchmark_trace
  config ``ssakethch/h200-quantization-benchmarks@throughput``.
* Promotion gates pass + state is the expected
  ``promoted_for_performance_priors``.
* Round-6 discovery-only rejection / deferral records for the 8 remaining
  round-5 candidates.
* No raw / analysis-sample data committed.
* No HF token leaked, no committed normalised sample under
  license=unspecified.

Audit-only — tests read committed artefacts; they do NOT hit the HF API.
"""
from __future__ import annotations

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

DATASET_ID = "ssakethch/h200-quantization-benchmarks"
CONFIG = "throughput"
SAFE_NAME = "ssakethch__h200-quantization-benchmarks"
PROC_DIR = HF_DIR / SAFE_NAME / CONFIG / "processed"
FIXTURE_PATH = FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"

ROUND6_DISCOVERY_ONLY_IDS = [
    "sairamn/gcp-cloud-billing-cost",
    "ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1",
    "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
    "Phipper/pe-energy-infrastructure-training-data",
    "uohna/llm_inference_energy_combined.parquet",
    "metrum-ai/llm-perf-dashboard",
    "crozai/vllm-benchmark-coding",
    "intellistream/sage-agent-benchmark",
]


def _summary() -> dict:
    return json.loads((PROC_DIR / "summary.json").read_text())


# ───────────────────────── 1. No raw / analysis data committed ─────────────


def test_no_raw_files_tracked_by_git() -> None:
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw = [p for p in out if "/raw/" in p]
    analysis = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw == [], f"Raw downloads committed: {raw}"
    assert analysis == [], (
        f"analysis_sample.jsonl committed: {analysis}"
    )


def test_no_committed_normalized_sample_under_unspecified_license() -> None:
    """License is unspecified → no normalised sample is committed."""
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    committed = [p for p in out
                 if p.endswith("/committed_normalized_sample.jsonl")]
    assert committed == [], (
        "committed_normalized_sample.jsonl present despite "
        f"license=unspecified: {committed}"
    )

    summary = _summary()
    assert summary["license"] is None
    assert summary["committed_normalized_sample_rows"] == 0
    assert summary["committed_normalized_sample_bytes"] == 0
    assert summary["committed_normalized_sample_path"] is None
    assert summary["committed_normalized_sample_reason_skipped"] == (
        "license_unspecified_no_redistribution_promise"
    )


def test_fixture_present_and_under_16_kib() -> None:
    assert FIXTURE_PATH.exists(), f"missing fixture: {FIXTURE_PATH}"
    size = FIXTURE_PATH.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {FIXTURE_PATH} size {size}B outside (0, 16 KiB]"
    )


def test_no_hf_token_in_committed_artifacts() -> None:
    """Catch accidental token leaks in any committed JSON / fixture."""
    paths = list(PROC_DIR.glob("*.json")) + [FIXTURE_PATH]
    for p in paths:
        content = p.read_text()
        assert "hf_" not in content.lower() or "hf_corpus" in content, (
            f"possible HF_TOKEN leaked into {p}"
        )
        assert "Bearer " not in content, f"Bearer token in {p}"


# ───────────────────────── 2. Processed artefact presence ─────────────────


@pytest.mark.parametrize("name", [
    "schema_profile.json",
    "schema_mapping.json",
    "summary.json",
    "statistical_rollups.json",
])
def test_processed_artifacts_present(name: str) -> None:
    p = PROC_DIR / name
    assert p.exists(), f"{p} missing"
    payload = json.loads(p.read_text())
    assert isinstance(payload, dict) and payload, f"{p} is empty"


# ───────────────────────── 3. Schema profile + mapping ────────────────────


def test_schema_profile_records_dtypes_and_examples() -> None:
    prof = json.loads((PROC_DIR / "schema_profile.json").read_text())
    assert prof["dataset_id"] == DATASET_ID
    assert prof["config_name"] == CONFIG
    assert prof["inspected_row_count"] == 275
    assert prof["unknown_columns"] == []
    expected_columns = {
        "model", "quant", "request_rate", "successful_reqs", "failed_reqs",
        "duration_s", "input_tokens", "output_tokens",
        "req_throughput", "output_tok_throughput",
        "peak_output_tok_throughput", "total_tok_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    }
    assert set(prof["raw_columns"]) == expected_columns
    for col in prof["raw_columns"]:
        assert col in prof["dtypes"]
        assert prof["dtypes"][col]


def test_schema_mapping_field_quality_and_signal_categories() -> None:
    m = json.loads((PROC_DIR / "schema_mapping.json").read_text())
    assert m["dataset_id"] == DATASET_ID
    assert m["columns"], "no columns in mapping"
    assert m["unknown_columns"] == []
    allowed_quality = {"real", "derived", "proxy", "synthetic",
                       "missing", "unknown"}
    allowed_signal = {
        "request_arrival", "request_completion", "latency", "queue",
        "throughput", "tokens", "gpu_resource", "memory", "autoscaling",
        "routing", "cache_residency", "failure_timeout", "scheduler_state",
        "cost_energy_carbon", "metadata_only", "irrelevant",
    }
    for c in m["columns"]:
        assert c["field_quality"] in allowed_quality, c
        assert c["aurelius_signal_category"] in allowed_signal, c
        assert c["normalized_field"]
        assert c["raw_column_name"]


def test_latency_columns_mapped_to_latency_signal() -> None:
    m = json.loads((PROC_DIR / "schema_mapping.json").read_text())
    by_raw = {c["raw_column_name"]: c for c in m["columns"]}
    for col in [
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    ]:
        assert col in by_raw, f"{col} missing in schema_mapping"
        assert by_raw[col]["aurelius_signal_category"] == "latency"
        assert by_raw[col]["field_quality"] == "real"


def test_throughput_columns_mapped_to_throughput_signal() -> None:
    m = json.loads((PROC_DIR / "schema_mapping.json").read_text())
    by_raw = {c["raw_column_name"]: c for c in m["columns"]}
    for col in ["req_throughput", "output_tok_throughput",
                "peak_output_tok_throughput", "total_tok_throughput"]:
        assert by_raw[col]["aurelius_signal_category"] == "throughput"


# ───────────────────────── 4. Promotion gates pass ────────────────────────


def test_promotion_gates_all_pass() -> None:
    s = _summary()
    gate_log = promotion.gates(s)
    failed = [g for g in gate_log if not g["passed"]]
    assert not failed, (
        f"gates failed: {[g['gate'] for g in failed]} (log={gate_log})"
    )


def test_promotion_state_is_performance_priors() -> None:
    s = _summary()
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_performance_priors"
    assert "promoted_for_performance_priors" in decision["promotion_tags"]
    assert "promoted_for_constraint_aware_evaluation" in decision[
        "promotion_tags"]
    assert "promoted_for_training_priors" in decision["promotion_tags"]


def test_canonical_trace_type_is_latency_benchmark() -> None:
    s = _summary()
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


def test_statistical_sample_strength_is_strong() -> None:
    s = _summary()
    # 275 rows with ≥12 rows per (quant, request_rate) cell except nvfp4 → strong.
    assert s["statistical_sample_strength"] == "strong"


def test_fixture_vs_analysis_separation_recorded() -> None:
    s = _summary()
    assert s["fixture_sample_rows"] == 5
    assert s["analysis_sample_rows"] == 275
    assert s["fixture_sample_rows"] < s["analysis_sample_rows"]


# ───────────────────────── 5. Signals are explicit ─────────────────────────


def test_available_signals_include_ttft_tpot_itl_throughput() -> None:
    s = _summary()
    for sig in ("ttft", "tpot", "itl", "throughput", "concurrency"):
        assert sig in s["available_signals"], (
            f"missing {sig} in available_signals"
        )


def test_missing_signals_include_queue_and_economics() -> None:
    s = _summary()
    for sig in (
        "queue_wait", "queue_depth", "gpu_utilization",
        "memory_pressure", "cost_per_request", "cost_per_token",
        "energy_per_request", "carbon_intensity",
    ):
        assert sig in s["missing_signals"], (
            f"missing {sig} in missing_signals — Aurelius rules require "
            "explicit absence labelling"
        )


def test_limitations_disclose_tier4_and_no_pilot_telemetry() -> None:
    s = _summary()
    blob = " ".join(s["limitations"]).lower()
    assert "tier 4" in blob or "tier-4" in blob
    assert "not pilot telemetry" in blob or "pilot telemetry" in blob
    assert "h200" in blob


# ───────────────────────── 6. Fixture determinism ─────────────────────────


def test_fixture_jsonl_is_well_formed_and_5_rows() -> None:
    lines = FIXTURE_PATH.read_text().splitlines()
    assert lines, "fixture empty"
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 5
    for r in rows:
        assert r["source_dataset_id"] == DATASET_ID
        assert r["trace_type"] == "latency_benchmark_trace"
        assert r["gpu_type"] == "NVIDIA H200 SXM"
        assert r["engine"] == "vllm"


def test_fixture_covers_all_five_quantizations() -> None:
    lines = FIXTURE_PATH.read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    quants = {r["quantization"] for r in rows}
    assert quants == {"bf16", "fp8", "awq", "gptq", "nvfp4"}


def test_fixture_sample_sha256_matches_summary() -> None:
    import hashlib
    h = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    s = _summary()
    assert s["sample_sha256"] == h


# ───────────────────────── 7. Statistical rollups ─────────────────────────


def test_rollups_overall_includes_all_latency_percentiles() -> None:
    r = json.loads((PROC_DIR / "statistical_rollups.json").read_text())
    for k in [
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "request_throughput", "output_token_throughput",
        "total_token_throughput",
    ]:
        assert k in r["overall"], f"missing overall stat: {k}"
        assert r["overall"][k]["count"] == 275


def test_rollups_subgroup_counts_match_quant_request_rate_grid() -> None:
    r = json.loads((PROC_DIR / "statistical_rollups.json").read_text())
    counts = r["subgroup_counts"]
    # 5 quants × 5 request rates = 25 cells.
    assert len(counts) == 25
    # AWQ / BF16 / FP8 have 14 per cell. GPTQ has 12. NVFP4 has 1.
    for q in ("awq", "bf16", "fp8"):
        for rate in (1.0, 2.0, 4.0, 8.0, 16.0):
            assert counts[f"quant={q}|request_rate={rate}"] == 14
    for rate in (1.0, 2.0, 4.0, 8.0, 16.0):
        assert counts[f"quant=gptq|request_rate={rate}"] == 12
        assert counts[f"quant=nvfp4|request_rate={rate}"] == 1


def test_insufficient_sample_groups_flag_nvfp4_cells() -> None:
    r = json.loads((PROC_DIR / "statistical_rollups.json").read_text())
    insufficient = set(r["insufficient_sample_groups"])
    # All 5 nvfp4 cells (n=1) should be flagged.
    for rate in (1.0, 2.0, 4.0, 8.0, 16.0):
        assert f"quant=nvfp4|request_rate={rate}" in insufficient, (
            f"insufficient_sample_groups did not flag nvfp4 rr={rate}"
        )


# ───────────────────────── 8. Registries updated ──────────────────────────


def test_canonical_registry_contains_new_entry() -> None:
    reg = json.loads(
        (DISC_DIR / "canonical_corpus_registry.json").read_text()
    )
    by_key = {(e["dataset_id"], e["config_name"]): e for e in reg["entries"]}
    assert (DATASET_ID, CONFIG) in by_key, (
        f"registry missing {DATASET_ID}@{CONFIG}"
    )
    entry = by_key[(DATASET_ID, CONFIG)]
    assert entry["canonical_trace_type"] == "latency_benchmark_trace"
    assert entry["trust_tier"] == "tier_4_latency_benchmark_traces"
    assert entry["promotion_state"] == "promoted_for_performance_priors"
    assert entry["statistical_sample_strength"] == "strong"
    assert entry["license"] is None  # unspecified upstream
    assert entry["gated"] is False
    assert entry["analysis_sample_rows"] == 275
    assert entry["fixture_sample_rows"] == 5


def test_candidate_registry_records_round6_discovery_only_ids() -> None:
    cands_path = DISC_DIR / "hf_dataset_candidates.json"
    cands = json.loads(cands_path.read_text())["candidates"]
    by_id = {c["dataset_id"]: c for c in cands}
    for ds in ROUND6_DISCOVERY_ONLY_IDS:
        assert ds in by_id, f"discovery-only id missing: {ds}"
        c = by_id[ds]
        assert c.get("round6_audit_reason"), (
            f"{ds} missing round6_audit_reason"
        )
        assert c.get("round6_audit_bucket"), (
            f"{ds} missing round6_audit_bucket"
        )
        assert c["recommended_action"].startswith(
            ("reject_", "duplicate_")
        ), (
            f"{ds} recommended_action expected reject_*/duplicate_*, "
            f"got {c['recommended_action']}"
        )


def test_candidate_registry_records_ingested_dataset() -> None:
    cands = json.loads(
        (DISC_DIR / "hf_dataset_candidates.json").read_text()
    )["candidates"]
    by_id = {c["dataset_id"]: c for c in cands}
    assert DATASET_ID in by_id
    c = by_id[DATASET_ID]
    assert c["recommended_action"] == "ingest_now_bounded"
    assert c["candidate_trace_type"] == "latency_benchmark_trace"
    assert c["trust_level"] == "tier_4_latency_benchmark_traces"


# ───────────────────────── 9. Round-6 audit summary ───────────────────────


def test_round6_audit_summary_exists_and_lists_ingested_dataset() -> None:
    path = DISC_DIR / "round6_broadened_discovery_audit_summary.json"
    assert path.exists()
    d = json.loads(path.read_text())
    assert d["doc_version"] == "round6_broadened_discovery_audit_summary_v1"
    assert d["production_claim"] is False
    assert d["modifies_robust_energy_engine"] is False
    assert d["uses_oracle_as_headline"] is False
    assert len(d["ingested"]) == 1
    assert d["ingested"][0]["dataset_id"] == DATASET_ID
    assert d["ingested"][0]["promotion_state"] == (
        "promoted_for_performance_priors"
    )


def test_round6_audit_summary_records_all_eight_discovery_only() -> None:
    d = json.loads(
        (DISC_DIR / "round6_broadened_discovery_audit_summary.json").read_text()
    )
    ids = {r["dataset_id"] for r in d["discovery_only_records"]}
    assert ids == set(ROUND6_DISCOVERY_ONLY_IDS), (
        f"discovery_only_records id-set mismatch: "
        f"missing={set(ROUND6_DISCOVERY_ONLY_IDS) - ids}, "
        f"extra={ids - set(ROUND6_DISCOVERY_ONLY_IDS)}"
    )


def test_round6_audit_summary_economic_priority_notes_negative_finding() -> None:
    d = json.loads(
        (DISC_DIR / "round6_broadened_discovery_audit_summary.json").read_text()
    )
    eps = d["economic_priority_summary"]
    # No new operational+economic dataset found this round.
    assert eps["datasets_with_operational_and_economic_signals"] == []
    assert eps["scorer_coefficients_calibratable_from_round6"] == []
    assert "negative_result_finding" in eps
    blob = eps["negative_result_finding"].lower()
    assert "economic" in blob or "goodput" in blob
    assert "h200" in blob


# ───────────────────────── 10. Anti-spam guardrails ───────────────────────


def test_no_unknown_columns_in_summary() -> None:
    s = _summary()
    assert s["unknown_columns"] == []


def test_no_oracle_headline_or_production_claim() -> None:
    s = _summary()
    blob = " ".join(s["limitations"]).lower()
    assert "production telemetry" in blob or "not pilot telemetry" in blob
    assert "tier 4" in blob or "tier-4" in blob
