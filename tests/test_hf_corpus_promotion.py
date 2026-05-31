"""Tests for the canonical-corpus promotion gates + registry writer."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces.hf_corpus import promotion  # noqa: E402


def _good_summary(**overrides) -> dict:
    base = {
        "dataset_id": "test/latency-bench",
        "source_url": "https://huggingface.co/datasets/test/latency-bench",
        "license": "apache-2.0",
        "gated": False,
        "canonical_trace_type": "latency_benchmark_trace",
        "committed_sample_rows": 100,
        "committed_sample_bytes": 10_000,
        "sample_sha256": "f" * 64,
        "raw_schema": ["model", "mean_ttft_ms"],
        "normalized_schema": ["model", "mean_ttft_ms"],
        "unknown_columns": [],
        "field_quality": {"model": "real", "mean_ttft_ms": "real"},
        "available_signals": ["ttft", "tpot"],
        "missing_signals": ["queue_wait"],
        "derived_fields": [],
        "proxy_fields": [],
        "synthetic_fields": [],
        "limitations": ["bounded test fixture"],
        "provenance": "test/latency-bench@unit-test#v1",
        "ingestion_timestamp_s": 1_750_000_000.0,
        "git_sha": "deadbeef",
        "config_name": None,
    }
    base.update(overrides)
    return base


# --- 1. Gates: positive path ---------------------------------------------


def test_all_gates_pass_for_a_good_summary():
    gates = promotion.gates(_good_summary())
    assert all(g["passed"] for g in gates), gates


def test_promotion_decision_promoted_for_performance_priors():
    decision = promotion.evaluate_promotion(_good_summary())
    assert decision["state"] == "promoted_for_performance_priors"
    assert "promoted_for_performance_priors" in decision["promotion_tags"]


# --- 2. Gates: each individual gate fails on a tweaked summary -----------


def test_schema_gate_fails_on_unknown_columns():
    s = _good_summary(unknown_columns=["mystery"])
    gates = promotion.gates(s)
    schema = [g for g in gates if g["gate"] == "schema_test"][0]
    assert not schema["passed"]


def test_schema_gate_fails_on_empty_schemas():
    s = _good_summary(raw_schema=[], normalized_schema=[])
    gates = promotion.gates(s)
    schema = [g for g in gates if g["gate"] == "schema_test"][0]
    assert not schema["passed"]


def test_fixture_gate_fails_on_zero_rows():
    s = _good_summary(committed_sample_rows=0)
    gates = promotion.gates(s)
    fixture = [g for g in gates if g["gate"] == "fixture_test"][0]
    assert not fixture["passed"]


def test_bounded_size_gate_fails_on_oversized_sample():
    s = _good_summary(committed_sample_bytes=promotion.MAX_COMMITTED_SAMPLE_BYTES + 1)
    gates = promotion.gates(s)
    bg = [g for g in gates if g["gate"] == "bounded_size_guard"][0]
    assert not bg["passed"]


def test_license_gate_fails_when_no_gated_field():
    s = _good_summary()
    del s["gated"]
    gates = promotion.gates(s)
    lic = [g for g in gates if g["gate"] == "license_and_gating_recorded"][0]
    assert not lic["passed"]


def test_trace_type_gate_fails_on_mixed_or_unknown():
    s = _good_summary(canonical_trace_type="mixed_or_unknown_trace")
    gates = promotion.gates(s)
    tt = [g for g in gates if g["gate"] == "canonical_trace_type_assigned"][0]
    assert not tt["passed"]


def test_signals_gate_fails_on_missing_lists():
    s = _good_summary(available_signals=None)
    gates = promotion.gates(s)
    sg = [g for g in gates if g["gate"] == "signals_explicit"][0]
    assert not sg["passed"]


def test_limitations_gate_fails_when_empty():
    s = _good_summary(limitations=[])
    gates = promotion.gates(s)
    lg = [g for g in gates if g["gate"] == "limitations_recorded"][0]
    assert not lg["passed"]


# --- 3. Gated dataset short-circuits to gated_blocked --------------------


def test_gated_dataset_is_blocked_regardless_of_gates():
    decision = promotion.evaluate_promotion(_good_summary(gated=True))
    assert decision["state"] == "gated_blocked"


# --- 4. Rejection on any gate failure ------------------------------------


def test_rejection_emits_specific_reasons():
    decision = promotion.evaluate_promotion(_good_summary(
        committed_sample_rows=0, limitations=[]
    ))
    assert decision["state"] == "rejected"
    reasons = " ".join(decision["reasons"])
    assert "fixture_test" in reasons
    assert "limitations_recorded" in reasons


# --- 5. Mixed_or_unknown_trace cannot be promoted ------------------------


def test_mixed_trace_type_passes_signals_but_no_promotion():
    s = _good_summary(canonical_trace_type="mixed_or_unknown_trace")
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "rejected"


# --- 6. Per-trace-type promotion mappings --------------------------------


def test_telemetry_trace_promoted_to_dynamic_calibration():
    s = _good_summary(canonical_trace_type="telemetry_trace",
                      available_signals=["queue_wait", "gpu_utilization"])
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_dynamic_calibration"
    assert set(decision["promotion_tags"]).issuperset({
        "promoted_for_dynamic_calibration",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_backtest",
    })


def test_cluster_scheduler_trace_promoted_for_backtest():
    s = _good_summary(canonical_trace_type="cluster_scheduler_trace",
                      available_signals=["queue_wait"])
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_backtest"


def test_request_shape_trace_promoted_for_training_priors_only():
    s = _good_summary(canonical_trace_type="request_shape_trace",
                      available_signals=["prompt_tokens"])
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_training_priors"
    assert decision["promotion_tags"] == ["promoted_for_training_priors"]


# --- 7. Registry writer round-trip ---------------------------------------


def test_registry_write_and_load_roundtrip(tmp_path):
    s = _good_summary()
    decision = promotion.evaluate_promotion(s)
    entry = promotion.build_registry_entry(s, decision)
    out = tmp_path / "registry.json"
    payload = promotion.write_canonical_registry([entry], str(out))
    assert payload["entry_count"] == 1
    loaded = promotion.load_canonical_registry(str(out))
    assert loaded["entries"][0]["dataset_id"] == s["dataset_id"]
    assert loaded["entries"][0]["promotion_state"] == decision["state"]


def test_registry_records_trust_tier():
    s = _good_summary(canonical_trace_type="telemetry_trace",
                      available_signals=["queue_wait"])
    decision = promotion.evaluate_promotion(s)
    entry = promotion.build_registry_entry(s, decision)
    assert entry["trust_tier"] == "tier_2_public_telemetry_traces"


def test_registry_carries_provenance_and_signals():
    s = _good_summary()
    decision = promotion.evaluate_promotion(s)
    entry = promotion.build_registry_entry(s, decision)
    for k in (
        "available_signals", "missing_signals", "derived_fields",
        "proxy_fields", "synthetic_fields", "limitations", "provenance",
    ):
        assert k in entry


# --- 8. No secrets in summaries ------------------------------------------


def test_summary_never_contains_token_keys():
    s = _good_summary()
    payload = json.dumps(s).lower()
    assert "hf_token" not in payload
    assert "authorization" not in payload


# --- 9. Promotion states are exhaustive ----------------------------------


def test_promotion_states_enumerated():
    expected = {
        "candidate", "validated_bounded", "promoted_for_backtest",
        "promoted_for_training_priors",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_dynamic_calibration",
        "promoted_for_performance_priors",
        "promoted_for_cache_residency_evaluation",
        "rejected", "gated_blocked",
    }
    assert expected == set(promotion.PROMOTION_STATES)
