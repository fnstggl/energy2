"""Tests for the optimum-benchmark/llm-perf-leaderboard ingest.

Covers the 10 Tier-4 ``latency_benchmark_trace`` configs ingested by
``scripts/ingest_hf_optimum_benchmark.py``:

* 4 unquantized configs: 1xA100, 1xA10, 1xT4 (pytorch-cuda),
  32vCPU-C7i (pytorch-cpu)
* 5 quantized configs: bnb-1xA100, gptq-1xA100, awq-1xA10, bnb-1xT4,
  torchao-1xA10
* 1 alternate-backend config: openvino-cpu-unquantized-32vCPU-C7i

Plus a discovery-only round-2 rejection record set.

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

DATASET_ID = "optimum-benchmark/llm-perf-leaderboard"
SAFE_NAME = "optimum-benchmark__llm-perf-leaderboard"

NEW_CONFIGS = [
    "pytorch_cuda_unquantized_1xA100",
    "pytorch_cuda_unquantized_1xA10",
    "pytorch_cuda_unquantized_1xT4",
    "pytorch_cuda_bnb_1xA100",
    "pytorch_cuda_gptq_1xA100",
    "pytorch_cuda_awq_1xA10",
    "pytorch_cuda_bnb_1xT4",
    "pytorch_cuda_torchao_1xA10",
    "pytorch_cpu_unquantized_32vCPU_C7i",
]

DISCOVERY_ONLY_IDS_ROUND2 = {
    "Exgentic/agent-llm-traces",
    "wseaton/prefix-cache-bench",
    "aintech/vdf_prefix-cache",
    "kshitijthakkar/moe-inference-benchmark",
    "kshitijthakkar/large-moe-inference-benchmark",
    "JohnGavin/llmtelemetry-metrics",
    "abdallah1008/semantic-router-benchmark-data",
    "Nathan-Maine/dgx-spark-kv-cache-benchmark",
    "fabric/inference-benchmarker",
    "optimum-benchmark/llm-perf-leaderboard@openvino_cpu_unquantized_32vCPU_C7i",
}


def _processed_dir(config: str) -> Path:
    return HF_DIR / SAFE_NAME / config / "processed"


def _fixture_path(config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_NAME}__{config}_sample.jsonl"


def _summary(config: str) -> dict:
    with open(_processed_dir(config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis data committed ─────────────


def test_no_raw_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    assert raw_committed == [], (
        f"Raw downloads committed for {SAFE_NAME}: {raw_committed}"
    )


def test_no_analysis_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    analysis_committed = [
        p for p in out if p.endswith("/analysis_sample.jsonl")
    ]
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed for {SAFE_NAME}: "
        f"{analysis_committed}"
    )


def test_no_committed_normalized_sample_files():
    """License unspecified — no committed_normalized_sample.jsonl allowed."""
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    cns_committed = [
        p for p in out
        if p.endswith("/committed_normalized_sample.jsonl")
    ]
    assert cns_committed == [], (
        f"committed_normalized_sample.jsonl present despite license=None: "
        f"{cns_committed}"
    )


def test_fixture_files_are_committed_and_tiny():
    for config in NEW_CONFIGS:
        fx = _fixture_path(config)
        assert fx.exists(), f"missing fixture: {fx}"
        size = fx.stat().st_size
        assert 0 < size <= 16 * 1024, (
            f"fixture {fx} size {size}B outside (0, 16 KiB]"
        )


def test_fixtures_are_committed_to_git():
    """Fixtures must be tracked (not ignored)."""
    for config in NEW_CONFIGS:
        fx = _fixture_path(config)
        out = subprocess.run(
            ["git", "check-ignore", str(fx.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT, capture_output=True,
        )
        # check-ignore returns 1 when the file is NOT ignored (what we want).
        assert out.returncode == 1, (
            f"fixture {fx} is gitignored — should be committed"
        )


# ───────────────────────── 2. Per-config artefact presence ─────────────────


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_processed_artifacts_present(config: str) -> None:
    pd = _processed_dir(config)
    for name in (
        "schema_profile.json", "schema_mapping.json",
        "summary.json", "statistical_rollups.json",
    ):
        path = pd / name
        assert path.exists(), f"{path} missing for {config}"
        with open(path) as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict) and payload, f"{path} is empty"


# ───────────────────────── 3. Schema profile + mapping ─────────────────────


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_schema_profile_records_dtypes_and_examples(config: str) -> None:
    pd = _processed_dir(config)
    with open(pd / "schema_profile.json") as fh:
        prof = json.load(fh)
    assert prof["dataset_id"] == DATASET_ID
    assert prof["config_name"] == config
    assert prof["inspected_row_count"] > 0
    # CUDA pytorch CSVs have ~149 columns; CPU/openvino/onnxruntime have
    # ~60-80 columns (no CUDA-specific environment fields). Assert
    # raw_columns is non-trivially large for both shapes.
    assert len(prof["raw_columns"]) >= 60, (
        f"raw_columns suspiciously small for {config}: "
        f"{len(prof['raw_columns'])}"
    )
    assert prof["unknown_columns"] == [], (
        f"unknown_columns leaked: {prof['unknown_columns']}"
    )
    for col in prof["raw_columns"]:
        assert col in prof["dtypes"], f"missing dtype for {col}"
        assert prof["dtypes"][col], f"empty dtypes list for {col}"


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_schema_mapping_has_columns_and_field_quality(config: str) -> None:
    pd = _processed_dir(config)
    with open(pd / "schema_mapping.json") as fh:
        m = json.load(fh)
    assert m["dataset_id"] == DATASET_ID
    assert m["columns"], "no columns in mapping"
    assert m["unknown_columns"] == []
    allowed_quality = {"real", "derived", "proxy", "synthetic",
                       "missing", "unknown"}
    allowed_signal_categories = {
        "request_arrival", "request_completion", "latency", "queue",
        "throughput", "tokens", "gpu_resource", "memory", "autoscaling",
        "routing", "cache_residency", "failure_timeout", "scheduler_state",
        "cost_energy_carbon", "metadata_only", "irrelevant",
    }
    for c in m["columns"]:
        assert c["field_quality"] in allowed_quality, (
            f"bad field_quality: {c}"
        )
        assert c["aurelius_signal_category"] in allowed_signal_categories, (
            f"bad aurelius_signal_category: {c}"
        )
        assert "normalized_field" in c and c["normalized_field"]
        assert "raw_column_name" in c and c["raw_column_name"]


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_schema_mapping_classifies_energy_and_latency(config: str) -> None:
    """The signal taxonomy must correctly route energy / latency columns."""
    pd = _processed_dir(config)
    with open(pd / "schema_mapping.json") as fh:
        m = json.load(fh)
    by_raw = {c["raw_column_name"]: c for c in m["columns"]}
    # Latency columns
    for raw_col in [
        "report.prefill.latency.mean",
        "report.prefill.latency.p95",
        "report.prefill.latency.p99",
        "report.decode.latency.mean",
    ]:
        if raw_col in by_raw:
            c = by_raw[raw_col]
            assert c["aurelius_signal_category"] == "latency", (
                f"{raw_col} should be latency category, got "
                f"{c['aurelius_signal_category']}"
            )
            assert c["field_quality"] == "real"
    # Energy columns
    for raw_col in [
        "report.prefill.energy.total",
        "report.prefill.energy.gpu",
        "report.decode.energy.total",
        "report.decode.energy.gpu",
    ]:
        if raw_col in by_raw:
            c = by_raw[raw_col]
            assert c["aurelius_signal_category"] == "cost_energy_carbon", (
                f"{raw_col} should be cost_energy_carbon category, got "
                f"{c['aurelius_signal_category']}"
            )
            assert c["field_quality"] == "real"
    # Memory columns
    for raw_col in [
        "report.prefill.memory.max_global_vram",
        "report.decode.memory.max_global_vram",
    ]:
        if raw_col in by_raw:
            c = by_raw[raw_col]
            assert c["aurelius_signal_category"] == "memory", (
                f"{raw_col} should be memory category, got "
                f"{c['aurelius_signal_category']}"
            )


# ───────────────────────── 4. Promotion gates pass ─────────────────────────


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_promotion_gates_all_pass(config: str) -> None:
    s = _summary(config)
    gate_log = promotion.gates(s)
    failed = [g for g in gate_log if not g["passed"]]
    assert not failed, (
        f"{config} failed gates: {[g['gate'] for g in failed]}"
    )


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_canonical_trace_type_is_latency_benchmark(config: str) -> None:
    s = _summary(config)
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_license_recorded_as_none(config: str) -> None:
    """The HF card has no declared license — must be recorded as None."""
    s = _summary(config)
    assert s["license"] is None, (
        f"license must be None (unspecified) for {config}, got {s['license']}"
    )
    assert s["gated"] is False
    # And no committed normalized sample.
    assert s["committed_normalized_sample_rows"] == 0
    assert s["committed_normalized_sample_bytes"] == 0
    assert s["committed_normalized_sample_path"] is None
    assert s["committed_normalized_sample_reason_skipped"] == (
        "license_unspecified_no_redistribution_promise"
    )


# ───────────────────────── 5. Registry contains new entries ────────────────


def test_registry_contains_all_new_configs():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    assert reg, "canonical registry missing"
    seen = {
        (e["dataset_id"], e["config_name"]) for e in reg["entries"]
    }
    for config in NEW_CONFIGS:
        assert (DATASET_ID, config) in seen, (
            f"registry missing {DATASET_ID}/{config}"
        )


def test_registry_entries_are_promoted_not_rejected():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    for config in NEW_CONFIGS:
        entry = next(
            e for e in reg["entries"]
            if e["dataset_id"] == DATASET_ID and e["config_name"] == config
        )
        assert entry["promotion_state"] in {
            "promoted_for_performance_priors",
            "promoted_for_training_priors",
            "promoted_for_constraint_aware_evaluation",
        }, (
            f"unexpected promotion state for {config}: "
            f"{entry['promotion_state']}"
        )
        assert entry["trust_tier"] == "tier_4_latency_benchmark_traces"


def test_strong_configs_promoted_for_performance_priors():
    """Configs with ≥200 rows must reach promoted_for_performance_priors."""
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    strong_configs = {
        "pytorch_cuda_unquantized_1xA10",   # 1344 rows
        "pytorch_cuda_unquantized_1xT4",    # 1265 rows
        "pytorch_cuda_bnb_1xA100",          # 401 rows
        "pytorch_cuda_gptq_1xA100",         # 314 rows
        "pytorch_cuda_awq_1xA10",           # 1569 rows
        "pytorch_cuda_bnb_1xT4",            # 775 rows
        "pytorch_cpu_unquantized_32vCPU_C7i",  # 1128 rows
    }
    for config in strong_configs:
        entry = next(
            e for e in reg["entries"]
            if e["dataset_id"] == DATASET_ID and e["config_name"] == config
        )
        assert entry["promotion_state"] == "promoted_for_performance_priors", (
            f"strong config {config} must promote to performance_priors, "
            f"got {entry['promotion_state']}"
        )
        assert "promoted_for_performance_priors" in entry["promotion_tags"]


# ───────────────────────── 6. Sample policy + signal coverage ──────────────


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_analysis_sample_size_recorded(config: str) -> None:
    s = _summary(config)
    assert s["fixture_sample_rows"] > 0
    assert s["analysis_sample_rows"] >= s["fixture_sample_rows"]
    assert s["statistical_sample_strength"] in {
        "weak", "moderate", "strong"
    }


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_available_signals_cover_aurelius_objective_terms(
    config: str,
) -> None:
    """The optimum dataset directly informs latency / energy / memory / "
    throughput priors — those must appear in available_signals."""
    s = _summary(config)
    sig = set(s["available_signals"])
    must_have = {"ttft", "tpot", "throughput",
                 "energy_per_request", "memory_pressure"}
    missing = must_have - sig
    assert not missing, (
        f"{config} missing critical signals: {missing} "
        f"(have {sorted(sig)})"
    )


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_missing_signals_explicit(config: str) -> None:
    """Concurrency / queue / cache / routing must be in missing_signals —
    the optimum benchmark is single-stream + no cache."""
    s = _summary(config)
    miss = set(s["missing_signals"])
    must_be_missing = {"concurrency", "queue_state", "cache_hit",
                       "routing", "sla_label"}
    not_listed = must_be_missing - miss
    assert not not_listed, (
        f"{config} should list {not_listed} as missing"
    )


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_provenance_includes_dataset_and_config(config: str) -> None:
    s = _summary(config)
    prov = s["provenance"]
    assert DATASET_ID in prov
    assert config in prov
    assert "git=" in prov


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_fixture_rows_have_real_measurements(config: str) -> None:
    """Every fixture row must carry at least mean_ttft_ms OR mean_tpot_ms
    as a finite measurement — otherwise the dataset isn't useful as a
    latency prior."""
    fx = _fixture_path(config)
    with open(fx) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows, f"empty fixture: {fx}"
    measured_rows = [
        r for r in rows
        if r.get("mean_ttft_ms") is not None
        or r.get("mean_tpot_ms") is not None
    ]
    assert measured_rows, (
        f"fixture {fx} has zero rows with measured TTFT or TPOT"
    )
    # Trace type + provenance must be carried per-row.
    for r in rows:
        assert r["source_dataset_id"] == DATASET_ID
        assert r["trace_type"] == "latency_benchmark_trace"
        assert "provenance" in r


# ───────────────────────── 7. Statistical rollups ──────────────────────────


@pytest.mark.parametrize("config", NEW_CONFIGS)
def test_statistical_rollups_present(config: str) -> None:
    pd = _processed_dir(config)
    with open(pd / "statistical_rollups.json") as fh:
        rollups = json.load(fh)
    assert "overall" in rollups
    assert "by_strata" in rollups
    assert "subgroup_counts" in rollups
    # Overall must include at least mean_ttft_ms and mean_tpot_ms.
    overall = rollups["overall"]
    assert any(k.endswith("_ttft_ms") or k.endswith("_tpot_ms")
               for k in overall), (
        f"{config} rollups missing TTFT/TPOT: {list(overall.keys())}"
    )


# ───────────────────────── 8. Audit summary contents ───────────────────────


def _audit() -> dict:
    with open(DISC_DIR / "broadened_discovery_audit_summary.json") as fh:
        return json.load(fh)


def test_audit_summary_records_optimum_configs():
    a = _audit()
    seen = {(x["dataset_id"], x["config_name"]) for x in a["ingested"]}
    for config in NEW_CONFIGS:
        assert (DATASET_ID, config) in seen, (
            f"audit ingested set missing {DATASET_ID}/{config}"
        )


def test_audit_summary_records_round2_discovery_only_ids():
    a = _audit()
    seen = {r["dataset_id"] for r in a["discovery_only_records"]}
    missing = DISCOVERY_ONLY_IDS_ROUND2 - seen
    assert not missing, f"discovery-only set missing round-2 entries: {missing}"


def test_audit_summary_no_production_claim():
    a = _audit()
    assert a["production_claim"] is False
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False
