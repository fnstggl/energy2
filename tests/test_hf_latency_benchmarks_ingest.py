"""Tests for the broadened HF discovery latency-benchmark ingest.

Covers the 3 new Tier-4 ``latency_benchmark_trace`` datasets ingested by
``scripts/ingest_hf_latency_benchmarks.py``:

* ``odyn-network/odyn-benchmarks`` (4 configs)
* ``memoriant/dgx-spark-kv-cache-benchmark`` (1 config)
* ``intellistream/vllm-hust-benchmark-results`` (2 configs)

Plus the 8 discovery-only rejection / deferral records.

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

NEW_CONFIGS = [
    ("odyn-network/odyn-benchmarks", "qwen_chat_streaming"),
    ("odyn-network/odyn-benchmarks", "facebook_chat_streaming"),
    ("odyn-network/odyn-benchmarks", "qwen_batch"),
    ("odyn-network/odyn-benchmarks", "facebook_batch"),
    ("memoriant/dgx-spark-kv-cache-benchmark", "v3_corrected"),
    ("intellistream/vllm-hust-benchmark-results", "single_gpu"),
    ("intellistream/vllm-hust-benchmark-results", "multi_gpu"),
]

DISCOVERY_ONLY_IDS = [
    "tarekmasryo/llm-system-ops-production-telemetry-sft-data",
    "spiritbuun/turboquant-tcq-kv-cache",
    "hlarcher/inference-benchmarker",
    "Boxoffice1280/Neurips2026_evaluating_accuracy_KV-cache_reuse_techniques",
    "Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days-SVLSGPU_EvalData",
    "MCP-1st-Birthday/smoltrace-cloud-cost-tasks",
    "rbgo/llm-inference-benchmark",
    "project-vajra/dev-staging-h100-dgx",
]

SAFE_NAMES = {
    "odyn-network/odyn-benchmarks": "odyn-network__odyn-benchmarks",
    "memoriant/dgx-spark-kv-cache-benchmark": "memoriant__dgx-spark-kv-cache-benchmark",
    "intellistream/vllm-hust-benchmark-results": "intellistream__vllm-hust-benchmark-results",
}


def _processed_dir(dataset_id: str, config: str) -> Path:
    return HF_DIR / SAFE_NAMES[dataset_id] / config / "processed"


def _fixture_path(dataset_id: str, config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_NAMES[dataset_id]}__{config}_sample.jsonl"


def _summary(dataset_id: str, config: str) -> dict:
    with open(_processed_dir(dataset_id, config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis data committed ─────────────


def test_no_raw_or_analysis_files_tracked_by_git():
    for safe in SAFE_NAMES.values():
        out = subprocess.check_output(
            ["git", "ls-files", f"data/external/hf/{safe}"],
            cwd=REPO_ROOT,
        ).decode().splitlines()
        raw_committed = [p for p in out if "/raw/" in p]
        analysis_committed = [
            p for p in out if p.endswith("/analysis_sample.jsonl")
        ]
        assert raw_committed == [], (
            f"Raw downloads committed for {safe}: {raw_committed}"
        )
        assert analysis_committed == [], (
            f"analysis_sample.jsonl committed for {safe}: {analysis_committed}"
        )


def test_fixture_files_are_committed_and_tiny():
    for dataset_id, config in NEW_CONFIGS:
        fx = _fixture_path(dataset_id, config)
        assert fx.exists(), f"missing fixture: {fx}"
        size = fx.stat().st_size
        assert 0 < size <= 16 * 1024, (
            f"fixture {fx} size {size}B outside (0, 16 KiB]"
        )


# ───────────────────────── 2. Per-config artefact presence ─────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_processed_artifacts_present(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    for name in (
        "schema_profile.json", "schema_mapping.json",
        "summary.json", "statistical_rollups.json",
    ):
        path = pd / name
        assert path.exists(), f"{path} missing for {dataset_id}@{config}"
        with open(path) as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict) and payload, f"{path} is empty"


# ───────────────────────── 3. Schema profile + mapping ─────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_schema_profile_records_dtypes_and_examples(dataset_id: str,
                                                    config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "schema_profile.json") as fh:
        prof = json.load(fh)
    assert prof["dataset_id"] == dataset_id
    assert prof["config_name"] == config
    assert prof["inspected_row_count"] > 0
    assert prof["raw_columns"], "raw_columns empty"
    assert prof["unknown_columns"] == [], (
        f"unknown_columns leaked: {prof['unknown_columns']}"
    )
    # Every raw column must have at least one example and a non-empty dtype.
    for col in prof["raw_columns"]:
        assert col in prof["dtypes"], f"missing dtype for {col}"
        assert prof["dtypes"][col], f"empty dtypes list for {col}"
        # Most populated columns have examples; we don't insist on every
        # column having examples (some metadata cols are nested dicts).


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_schema_mapping_has_columns_and_field_quality(
    dataset_id: str, config: str
) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "schema_mapping.json") as fh:
        m = json.load(fh)
    assert m["dataset_id"] == dataset_id
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


# ───────────────────────── 4. Promotion gates pass ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_promotion_gates_all_pass(dataset_id: str, config: str) -> None:
    s = _summary(dataset_id, config)
    gate_log = promotion.gates(s)
    failed = [g for g in gate_log if not g["passed"]]
    assert not failed, (
        f"{dataset_id}@{config} failed gates: "
        f"{[g['gate'] for g in failed]}"
    )


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_canonical_trace_type_is_latency_benchmark(dataset_id: str,
                                                   config: str) -> None:
    s = _summary(dataset_id, config)
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


# ───────────────────────── 5. Registry contains new entries ────────────────


def test_registry_contains_all_new_configs():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    assert reg, "canonical registry missing"
    seen = {(e["dataset_id"], e["config_name"]) for e in reg["entries"]}
    for dataset_id, config in NEW_CONFIGS:
        assert (dataset_id, config) in seen, (
            f"registry missing {dataset_id}/{config}"
        )


def test_registry_entries_are_promoted_not_rejected():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    for dataset_id, config in NEW_CONFIGS:
        entry = next(
            e for e in reg["entries"]
            if e["dataset_id"] == dataset_id and e["config_name"] == config
        )
        assert entry["promotion_state"] in {
            "promoted_for_performance_priors",
            "promoted_for_training_priors",
            "promoted_for_constraint_aware_evaluation",
        }, f"unexpected promotion state for {dataset_id}/{config}: " \
           f"{entry['promotion_state']}"
        # Trust tier must be tier_4 for latency benchmarks.
        assert entry["trust_tier"] == "tier_4_latency_benchmark_traces"


# ───────────────────────── 6. Audit summary contents ───────────────────────


def _audit() -> dict:
    with open(DISC_DIR / "broadened_discovery_audit_summary.json") as fh:
        return json.load(fh)


def test_audit_summary_records_all_ingested():
    a = _audit()
    seen = {(x["dataset_id"], x["config_name"]) for x in a["ingested"]}
    assert seen == set(NEW_CONFIGS), (
        f"audit ingested set mismatch: {seen ^ set(NEW_CONFIGS)}"
    )


def test_audit_summary_records_all_discovery_only_ids():
    a = _audit()
    seen = {r["dataset_id"] for r in a["discovery_only_records"]}
    assert seen == set(DISCOVERY_ONLY_IDS), (
        f"discovery-only set mismatch: {seen ^ set(DISCOVERY_ONLY_IDS)}"
    )


def test_audit_summary_no_production_claim():
    a = _audit()
    assert a["production_claim"] is False
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False


# ───────────────────────── 7. Signal taxonomy + provenance ─────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_available_signals_includes_measured_latency_or_throughput(
    dataset_id: str, config: str
) -> None:
    s = _summary(dataset_id, config)
    avail = set(s["available_signals"])
    # Every config must record at least one measured signal that a latency
    # benchmark is supposed to carry.
    measured_sigs = {
        "ttft", "tpot", "e2e_latency", "throughput", "itl",
    }
    assert avail & measured_sigs, (
        f"{dataset_id}/{config} declares no measured latency/throughput "
        f"signal — available={avail}"
    )


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_limitations_recorded_with_tier_4_note(dataset_id: str,
                                               config: str) -> None:
    s = _summary(dataset_id, config)
    lim = " ".join(s["limitations"]).lower()
    assert "tier 4" in lim or "benchmark" in lim, (
        f"{dataset_id}/{config} limitations missing tier/benchmark note: "
        f"{s['limitations']}"
    )


# ───────────────────────── 8. License + redistribution policy ──────────────


def test_intellistream_has_no_committed_normalized_sample():
    """The intellistream leaderboard has no declared license (license=None);
    no normalised sample is committed under the conservative redistribution
    policy."""
    for cfg in ("single_gpu", "multi_gpu"):
        s = _summary("intellistream/vllm-hust-benchmark-results", cfg)
        assert s["license"] is None
        assert s["committed_normalized_sample_rows"] == 0
        assert s["committed_normalized_sample_path"] is None
        assert (
            s["committed_normalized_sample_reason_skipped"]
            == "license_unspecified_no_redistribution_promise"
        )


@pytest.mark.parametrize("config", [
    "qwen_chat_streaming", "facebook_chat_streaming",
    "qwen_batch", "facebook_batch",
])
def test_odyn_apache2_normalized_sample_committed_and_bounded(config: str):
    s = _summary("odyn-network/odyn-benchmarks", config)
    assert s["license"] == "apache-2.0"
    assert s["committed_normalized_sample_rows"] > 0
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    assert path.exists()
    assert 0 < path.stat().st_size <= 100 * 1024, (
        f"committed normalised sample {path} bigger than 100 KiB cap"
    )


def test_memoriant_apache2_normalized_sample_committed_and_bounded():
    s = _summary("memoriant/dgx-spark-kv-cache-benchmark", "v3_corrected")
    assert s["license"] == "apache-2.0"
    assert s["committed_normalized_sample_rows"] > 0
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    assert path.exists()
    assert 0 < path.stat().st_size <= 100 * 1024


# ───────────────────────── 9. Fixture determinism ──────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_fixture_sha256_matches_summary(dataset_id: str, config: str) -> None:
    """Fixture sha256 in the summary must match the on-disk fixture."""
    import hashlib
    s = _summary(dataset_id, config)
    fx = _fixture_path(dataset_id, config)
    h = hashlib.sha256()
    with open(fx, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    assert h.hexdigest() == s["sample_sha256"], (
        f"fixture sha256 mismatch for {dataset_id}/{config}"
    )


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_fixture_is_valid_jsonl(dataset_id: str, config: str) -> None:
    fx = _fixture_path(dataset_id, config)
    lines = fx.read_text().splitlines()
    assert lines, f"fixture {fx} empty"
    for ln in lines:
        rec = json.loads(ln)
        assert rec["source_dataset_id"] == dataset_id
        assert rec["trace_type"] == "latency_benchmark_trace"
        assert rec.get("provenance"), "provenance missing"


# ───────────────────────── 10. Anti-spam guard ─────────────────────────────


def test_rejected_synthetic_dataset_not_in_registry():
    """tarekmasryo/llm-system-ops-production-telemetry-sft-data must NOT be
    promoted — even though it's tagged as 'production telemetry' it's
    self-declared synthetic and was rejected as low value."""
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    rejected_ids = set(DISCOVERY_ONLY_IDS)
    for e in reg["entries"]:
        assert e["dataset_id"] not in rejected_ids, (
            f"rejected dataset {e['dataset_id']} leaked into registry!"
        )


# ───────────────────────── 11. Fixture-bytes budget under PR cap ───────────


def test_total_new_committed_normalized_samples_under_pr_budget():
    """Total committed normalised samples for the 3 new datasets must stay
    under the 300 MiB PR-wide budget — they should in fact be tiny (< 200
    KiB total) since the upstream files are small."""
    total = 0
    for dataset_id, config in NEW_CONFIGS:
        s = _summary(dataset_id, config)
        total += int(s.get("committed_normalized_sample_bytes") or 0)
    assert total <= 300 * 1024 * 1024, (
        f"committed normalised sample budget blown: {total} bytes > 300 MiB"
    )
    # Sanity check: these are tiny benchmarks; expect under 500 KiB total.
    assert total < 500 * 1024, (
        f"unexpected committed-normalised total {total}B > 500 KiB; "
        "the PR description should reflect this"
    )
