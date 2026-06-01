"""Tests for the HF gap-closure ingest of the top-5 datasets.

Covers:
- no raw / large analysis_sample data committed,
- per-dataset schema_profile + schema_mapping + summary + rollups exist,
- schema coverage (mapping accepts every observed column for promotion),
- signal-coverage table records both available + missing signals,
- all 5 entries pass the canonical-corpus promotion gates,
- the canonical registry knows about them,
- per-dataset fixture file exists and is tiny.

Audit-only: tests read committed artifacts; they do NOT hit the HF API.
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

NEW_DATASETS = [
    ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head"),
    ("sammshen/lmcache-agentic-traces", "train_shard4"),
    ("lzzmm/BurstGPT", "burstgpt_1_full"),
    ("lsliwko/google-cluster-data-2019-sorted-by-timestamp", "instance_events_shard0"),
    ("jaytonde05/prefixbench", "prefixbench_all"),
]


def _processed_dir(dataset_id: str, config: str) -> Path:
    return HF_DIR / dataset_id.replace("/", "__") / config / "processed"


def _fixture_path(dataset_id: str, config: str) -> Path:
    return FIXTURES_DIR / f"{dataset_id.replace('/', '__')}__{config}_sample.jsonl"


# ───────────────────────── 1. No raw data committed ─────────────────────────


def test_no_raw_files_tracked_by_git():
    """git ls-files must never list anything under data/external/hf/*/raw/ or
    any processed/analysis_sample.jsonl."""
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed (gitignore broken): {analysis_committed}"
    )


def test_fixture_files_are_committed_and_tiny():
    """Each new dataset has a committed fixture sample under 16 KiB."""
    for dataset_id, config in NEW_DATASETS:
        fixture = _fixture_path(dataset_id, config)
        assert fixture.exists(), f"missing fixture: {fixture}"
        size = fixture.stat().st_size
        assert 0 < size <= 16 * 1024, (
            f"fixture {fixture} size {size}B is outside the (0, 16 KiB] range"
        )


# ───────────────────────── 2. Per-dataset artefacts ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_DATASETS)
def test_processed_artifacts_present(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    for name in ("schema_profile.json", "schema_mapping.json",
                 "summary.json", "statistical_rollups.json"):
        path = pd / name
        assert path.exists(), f"{path} missing for {dataset_id}@{config}"
        with open(path) as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict) and payload, f"{path} is empty"


@pytest.mark.parametrize("dataset_id,config", NEW_DATASETS)
def test_schema_mapping_classifies_every_known_column(dataset_id: str, config: str) -> None:
    """For every column the mapping is expected to recognise, it must record
    a field_quality value in the canonical set."""
    pd = _processed_dir(dataset_id, config)
    with open(pd / "schema_mapping.json") as fh:
        mapping = json.load(fh)
    assert "columns" in mapping and mapping["columns"], "no columns in mapping"
    # Accepted columns must all have a normalized_field + field_quality.
    accepted_records = [c for c in mapping["columns"]
                        if c["raw_column_name"] in mapping["accepted_columns"]]
    assert accepted_records, f"no accepted columns for {dataset_id}@{config}"
    for c in accepted_records:
        assert c["normalized_field"], f"missing normalized_field for {c}"
        assert c["field_quality"] in {
            "real", "derived", "proxy", "synthetic", "missing", "unknown",
        }, f"invalid field_quality: {c}"


# ───────────────────────── 3. Promotion gates ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_DATASETS)
def test_summary_passes_all_promotion_gates(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "summary.json") as fh:
        summary = json.load(fh)
    gates = promotion.gates(summary)
    failed = [g for g in gates if not g["passed"]]
    assert not failed, (
        f"{dataset_id}@{config}: gates failed: "
        f"{[(g['gate'], g['detail']) for g in failed]}"
    )


@pytest.mark.parametrize("dataset_id,config", NEW_DATASETS)
def test_summary_promoted_to_at_least_training_priors(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "summary.json") as fh:
        summary = json.load(fh)
    decision = promotion.evaluate_promotion(summary)
    assert decision["state"] in (
        "promoted_for_training_priors",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_backtest",
        "promoted_for_cache_residency_evaluation",
        "promoted_for_performance_priors",
        "promoted_for_dynamic_calibration",
    ), (
        f"{dataset_id}@{config}: unexpected state {decision['state']}; "
        f"reasons={decision.get('reasons')}"
    )
    assert decision["promotion_tags"], (
        f"{dataset_id}@{config}: no promotion_tags assigned"
    )


# ───────────────────────── 4. Signal-coverage table ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_DATASETS)
def test_signal_coverage_recorded(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "summary.json") as fh:
        summary = json.load(fh)
    avail = summary["available_signals"]
    missing = summary["missing_signals"]
    assert isinstance(avail, list) and avail, "available_signals is empty"
    assert isinstance(missing, list), "missing_signals must be a list"
    # No overlap between available and missing.
    assert not (set(avail) & set(missing)), (
        "available_signals and missing_signals overlap: "
        f"{set(avail) & set(missing)}"
    )


# ───────────────────────── 5. Per-dataset signal expectations ─────────────


def _summary(dataset_id: str, config: str) -> dict:
    with open(_processed_dir(dataset_id, config) / "summary.json") as fh:
        return json.load(fh)


def test_cc_traces_has_kv_block_hashes_and_migration_proxy():
    s = _summary("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head")
    for sig in ("kv_block_hashes", "migration_or_cache_loss_proxy",
                "cache_reuse", "prefix_reuse", "ttft"):
        assert sig in s["available_signals"], (
            f"cc-traces missing expected signal {sig}: {s['available_signals']}"
        )


def test_lmcache_has_cache_residency_signals():
    s = _summary("sammshen/lmcache-agentic-traces", "train_shard4")
    for sig in ("cache_reuse", "prefix_reuse", "routing_proxy", "arrivals"):
        assert sig in s["available_signals"], (
            f"lmcache missing expected signal {sig}: {s['available_signals']}"
        )


def test_burstgpt_has_arrival_and_capacity_signals():
    s = _summary("lzzmm/BurstGPT", "burstgpt_1_full")
    for sig in ("arrivals", "request_timestamps", "workload_shape",
                "capacity_proxy", "autoscaling_proxy", "customer_traffic_mix"):
        assert sig in s["available_signals"], (
            f"burstgpt missing expected signal {sig}: {s['available_signals']}"
        )


def test_google_cluster_has_autoscaling_migration_load_unload_proxies():
    s = _summary(
        "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
        "instance_events_shard0",
    )
    for sig in ("autoscaling_proxy", "capacity_proxy",
                "migration_or_cache_loss_proxy", "model_load_event",
                "model_unload_event", "routing_proxy"):
        assert sig in s["available_signals"], (
            f"google-cluster missing expected signal {sig}: {s['available_signals']}"
        )


def test_prefixbench_has_cache_residency_signals():
    s = _summary("jaytonde05/prefixbench", "prefixbench_all")
    for sig in ("cache_reuse", "prefix_reuse"):
        assert sig in s["available_signals"], (
            f"prefixbench missing expected signal {sig}: {s['available_signals']}"
        )


# ───────────────────────── 6. Cross-dataset summary ─────────────────────────


def test_cross_dataset_ingest_summary_consistent():
    path = DISC_DIR / "telemetry_gap_ingest_summary.json"
    assert path.exists(), f"missing {path}"
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["modifies_robust_energy_engine"] is False
    assert payload["modifies_controllers_or_defaults"] is False
    assert payload["production_claim"] is False
    ingested_keys = {
        (e["dataset_id"], e["config_name"]) for e in payload["ingested"]
    }
    assert set(NEW_DATASETS) <= ingested_keys, (
        f"missing entries in ingest summary: {set(NEW_DATASETS) - ingested_keys}"
    )


def test_canonical_registry_includes_new_entries():
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    assert reg_path.exists()
    with open(reg_path) as fh:
        reg = json.load(fh)
    keys = {(e["dataset_id"], e.get("config_name")) for e in reg["entries"]}
    assert set(NEW_DATASETS) <= keys, (
        f"canonical registry missing: {set(NEW_DATASETS) - keys}"
    )


# ───────────────────────── 7. Trust-tier honesty ─────────────────────────


def test_no_new_dataset_marked_tier_1():
    """Tier 1 is reserved for real pilot telemetry. None of the public HF
    datasets may be promoted to Tier 1."""
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    with open(reg_path) as fh:
        reg = json.load(fh)
    for e in reg["entries"]:
        if (e["dataset_id"], e.get("config_name")) in NEW_DATASETS:
            assert e["trust_tier"] != "tier_1_real_pilot_telemetry", (
                f"{e['dataset_id']}@{e['config_name']} is marked Tier 1 "
                "but is HF-public, not pilot"
            )
