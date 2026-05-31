"""Tests for the compatibility-routed corpus evaluation harness."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces.hf_corpus import evaluation, ingestion, promotion  # noqa: E402


def _registry_entry(**overrides):
    base = {
        "dataset_id": "test/latency-bench",
        "config_name": None,
        "source_url": "https://huggingface.co/datasets/test/latency-bench",
        "canonical_trace_type": "latency_benchmark_trace",
        "trust_tier": "tier_4_latency_benchmark_traces",
        "license": "apache-2.0",
        "gated": False,
        "promotion_state": "promoted_for_performance_priors",
        "promotion_tags": ["promoted_for_performance_priors"],
        "promotion_reasons": [],
        "available_signals": ["ttft", "tpot"],
        "missing_signals": [],
        "derived_fields": [], "proxy_fields": [], "synthetic_fields": [],
        "limitations": ["bounded"],
        "ingestion_timestamp_s": 1.0,
        "promotion_evaluated_at_s": 2.0,
        "committed_sample_rows": 2,
        "committed_sample_bytes": 1000,
        "sample_sha256": "x" * 64,
        "provenance": "test",
        "summary_path_relative": "data/external/hf/test__latency-bench/processed/summary.json",
    }
    base.update(overrides)
    return base


def _write_sample(repo_root, entry, rows):
    paths = ingestion.safe_sample_paths(
        repo_root, entry["dataset_id"], entry.get("config_name"))
    os.makedirs(os.path.dirname(paths["sample_path"]), exist_ok=True)
    with open(paths["sample_path"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return paths["sample_path"]


# --- 1. Routing ----------------------------------------------------------


def test_route_latency_benchmark_dispatch():
    entry = _registry_entry()
    r = evaluation.route_dataset(entry)
    assert r["evaluator_id"] == "latency_benchmark_prior_smoke_v1"
    assert r["primary_baseline"] == "sla_aware_serving_frontier_static"
    assert r["skip_reason"] is None


def test_route_skips_when_required_signal_missing():
    entry = _registry_entry(available_signals=["concurrency"])  # no ttft/tpot/e2e
    r = evaluation.route_dataset(entry)
    assert r["skip_reason"] is not None
    assert "requires" in r["skip_reason"]


def test_route_skips_unknown_trace_type():
    entry = _registry_entry(canonical_trace_type="not_a_real_type")
    r = evaluation.route_dataset(entry)
    assert r["evaluator_id"] is None
    assert r["skip_reason"] is not None


# --- 2. Evaluator functions per trace type --------------------------------


def test_latency_benchmark_evaluator_computes_distribution(tmp_path):
    entry = _registry_entry()
    rows = [
        {"p99_ttft_ms": 100.0, "p99_tpot_ms": 20.0, "p99_e2el_ms": 500.0,
         "mean_ttft_ms": 80.0, "mean_tpot_ms": 18.0, "mean_e2el_ms": 480.0},
        {"p99_ttft_ms": 200.0, "p99_tpot_ms": 30.0, "p99_e2el_ms": 800.0,
         "mean_ttft_ms": 150.0, "mean_tpot_ms": 25.0, "mean_e2el_ms": 600.0},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["skip_reason"] is None
    assert result["evaluator_id"] == "latency_benchmark_prior_smoke_v1"
    assert "p99_ttft_ms" in result["result"]
    assert result["result"]["p99_ttft_ms"]["count"] == 2
    assert "performance_priors" in result["informs"]
    assert result["result_quality"] == "prior_only"


def test_kernel_profile_evaluator(tmp_path):
    entry = _registry_entry(
        canonical_trace_type="kernel_profile_trace",
        promotion_state="promoted_for_performance_priors",
        available_signals=["kernel_duration"],
    )
    rows = [
        {"duration_ms": 1.5}, {"duration_ms": 2.0}, {"duration_ms": 0.8},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["skip_reason"] is None
    assert result["result"]["duration_ms"]["count"] == 3


def test_cluster_scheduler_evaluator(tmp_path):
    entry = _registry_entry(
        canonical_trace_type="cluster_scheduler_trace",
        promotion_state="promoted_for_backtest",
        available_signals=["queue_wait"],
    )
    rows = [
        {"queue_wait_s": 12.0, "duration_s": 100.0},
        {"queue_wait_s": 4.0, "duration_s": 60.0},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["result"]["queue_wait_s"]["count"] == 2


def test_telemetry_evaluator(tmp_path):
    entry = _registry_entry(
        canonical_trace_type="telemetry_trace",
        promotion_state="promoted_for_dynamic_calibration",
        available_signals=["queue_wait", "gpu_utilization"],
        trust_tier="tier_2_public_telemetry_traces",
    )
    rows = [
        {"queue_wait_s": 0.5, "gpu_utilization": 0.7, "timeout_rate_pct": 0.1},
        {"queue_wait_s": 1.5, "gpu_utilization": 0.8, "timeout_rate_pct": 0.2},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["result_quality"] == "prior_only"  # benchmark, not pilot
    assert "dynamic_frontier" in result["informs"]


def test_cache_residency_evaluator(tmp_path):
    entry = _registry_entry(
        canonical_trace_type="cache_residency_trace",
        promotion_state="promoted_for_cache_residency_evaluation",
        available_signals=["cache_hit", "cold_start"],
    )
    rows = [
        {"cache_hit": True, "cold_start": False},
        {"cache_hit": False, "cold_start": True},
        {"cache_hit": True, "cold_start": False},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["result"]["cache_hit_rate"] == 2 / 3
    assert result["result"]["cold_start_rate"] == 1 / 3


def test_request_shape_evaluator(tmp_path):
    entry = _registry_entry(
        canonical_trace_type="request_shape_trace",
        promotion_state="promoted_for_training_priors",
        available_signals=["prompt_tokens"],
    )
    rows = [
        {"prompt_tokens": 100, "output_tokens": 20},
        {"prompt_tokens": 200, "output_tokens": 40},
    ]
    _write_sample(str(tmp_path), entry, rows)
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["result"]["prompt_tokens"]["count"] == 2


# --- 3. Missing sample file is explicitly skipped ------------------------


def test_missing_sample_file_skipped_with_explicit_reason(tmp_path):
    entry = _registry_entry()
    result = evaluation.evaluate_one(entry, str(tmp_path))
    assert result["skip_reason"] is not None
    assert "sample file" in result["skip_reason"].lower()


# --- 4. select_eligible filters to promoted states only -------------------


def test_select_eligible_filters_non_promoted_entries():
    registry = {"entries": [
        _registry_entry(promotion_state="candidate"),
        _registry_entry(dataset_id="t/2", promotion_state="rejected"),
        _registry_entry(dataset_id="t/3",
                        promotion_state="promoted_for_performance_priors"),
        _registry_entry(dataset_id="t/4",
                        promotion_state="gated_blocked"),
    ]}
    elig = evaluation.select_eligible(registry)
    assert len(elig) == 1
    assert elig[0]["dataset_id"] == "t/3"


# --- 5. Aggregation discipline + production-claim invariants -------------


def test_corpus_evaluation_payload_invariants(tmp_path):
    entry = _registry_entry()
    rows = [{"p99_ttft_ms": 100.0, "mean_ttft_ms": 50.0}]
    _write_sample(str(tmp_path), entry, rows)
    payload = evaluation.run_corpus_evaluation(
        {"entries": [entry]}, str(tmp_path)
    )
    assert payload["production_claim"] is False
    assert payload["uses_oracle_as_headline"] is False
    assert payload["treats_benchmark_as_production_telemetry"] is False
    assert "aggregation_rule" in payload
    # Per-dataset rows include the comparison-against-oracle flag.
    for r in payload["per_dataset_results"]:
        assert r["comparison_against_oracle_is_headline"] is False
        assert r["is_production_telemetry_substitute"] is False


def test_evaluation_summary_writer_writes_json(tmp_path):
    entry = _registry_entry()
    rows = [{"p99_ttft_ms": 100.0, "mean_ttft_ms": 50.0}]
    _write_sample(str(tmp_path), entry, rows)
    payload = evaluation.run_corpus_evaluation(
        {"entries": [entry]}, str(tmp_path)
    )
    out = tmp_path / "summary.json"
    evaluation.write_evaluation_summary(payload, str(out))
    loaded = json.load(open(out))
    assert loaded["n_eligible"] == 1


# --- 6. Datasets cluster by trace type, not aggregated across -------------


def test_datasets_by_trace_type_grouping_preserved(tmp_path):
    e1 = _registry_entry(dataset_id="t/lat-1")
    e2 = _registry_entry(dataset_id="t/lat-2")
    e3 = _registry_entry(
        dataset_id="t/sched-1",
        canonical_trace_type="cluster_scheduler_trace",
        promotion_state="promoted_for_backtest",
        available_signals=["queue_wait"],
    )
    for e in (e1, e2, e3):
        rows = [
            {"p99_ttft_ms": 100.0} if e["canonical_trace_type"]
            == "latency_benchmark_trace"
            else {"queue_wait_s": 1.0, "duration_s": 10.0}
        ]
        _write_sample(str(tmp_path), e, rows)
    payload = evaluation.run_corpus_evaluation(
        {"entries": [e1, e2, e3]}, str(tmp_path)
    )
    by_tt = payload["datasets_by_trace_type"]
    assert sorted(by_tt["latency_benchmark_trace"]) == ["t/lat-1", "t/lat-2"]
    assert by_tt["cluster_scheduler_trace"] == ["t/sched-1"]
