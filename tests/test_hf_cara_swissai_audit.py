"""Tests for the CARA + SwissAI telemetry-candidate audit.

The audit script ``scripts/audit_cara_swissai_telemetry.py`` is run live
against the HF API in the audit PR; these tests cover the building blocks
(``schema_profile.py``, the extended schemas, the sample-strength gates,
the per-config column mappings) so future re-runs cannot silently
regress the audit's promises.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces.hf_corpus import (  # noqa: E402
    ingestion, promotion, schema_profile, schemas,
)


# ---------- 1. schema_profile module --------------------------------------


def test_profile_flat_rows_records_all_top_level_keys():
    rows = [
        {"a": 1, "b": "x", "c": None},
        {"a": 2, "b": "y", "d": 3.5},
        {"a": -1, "b": "z", "d": 4.0, "c": True},
    ]
    p = schema_profile.profile_rows(
        rows, dataset_id="t/x", config_name="cfg", split="t",
        source_files_inspected=["x.jsonl"], file_size_bytes=100,
    )
    assert p["inspected_row_count"] == 3
    assert set(p["raw_columns"]) == {"a", "b", "c", "d"}
    assert p["presence_rates"]["a"] == 1.0
    assert p["presence_rates"]["c"] == pytest.approx(2 / 3)
    assert p["presence_rates"]["d"] == pytest.approx(2 / 3)
    # -1 numeric sentinel counted as missing.
    assert p["missing_rates"]["a"] == pytest.approx(1 / 3)


def test_profile_flattens_one_level_of_nested_dict():
    rows = [
        {"id": "r1", "schedule_state": {"num_running": 3, "num_waiting": 0}},
        {"id": "r2", "schedule_state": {"num_running": 5, "num_waiting": 1}},
    ]
    p = schema_profile.profile_rows(
        rows, dataset_id="t/x", config_name="c", split=None,
        source_files_inspected=["x"], file_size_bytes=None,
    )
    assert "schedule_state.num_running" in p["nested_keys"]
    assert "schedule_state.num_waiting" in p["nested_keys"]
    assert p["presence_rates"]["schedule_state.num_running"] == 1.0


def test_profile_records_list_length_summary():
    rows = [
        {"id": "r1", "bucket_ids": [1, 2, 3]},
        {"id": "r2", "bucket_ids": [4, 5, 6, 7, 8]},
        {"id": "r3", "bucket_ids": [9]},
    ]
    p = schema_profile.profile_rows(
        rows, dataset_id="t/x", config_name="c", split=None,
        source_files_inspected=[], file_size_bytes=None,
    )
    s = p["list_length_summaries"]["bucket_ids"]
    assert s["min_len"] == 1 and s["max_len"] == 5
    assert s["samples"] == 3


def test_build_schema_mapping_distinguishes_accepted_and_rejected():
    profile = {
        "raw_columns": ["a", "b", "c"], "nested_keys": [], "dtypes": {},
        "presence_rates": {"a": 1.0, "b": 1.0, "c": 1.0},
        "missing_rates": {"a": 0.0, "b": 0.0, "c": 0.0},
        "example_values": {},
    }
    column_mapping = {
        "a": {
            "normalized_field": "a", "field_quality": "real", "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["workload_shape_only"], "notes": "ok",
        },
        # 'b' absent from mapping -> rejected.
    }
    m = schema_profile.build_schema_mapping(
        profile, column_mapping, dataset_id="t/x", config_name="c",
    )
    assert "a" in m["accepted_columns"]
    assert sorted(m["rejected_columns"]) == ["b", "c"]


def test_compute_numeric_summary_ignores_sentinel_minus_one():
    rows = [
        {"x": 1.0}, {"x": 2.0}, {"x": -1},
        {"x": 3.0}, {"x": None}, {"x": 4.0},
    ]
    s = schema_profile.compute_numeric_summary(rows, field="x")
    assert s["count"] == 4
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["missing"] == 2


def test_per_subgroup_summary_flags_insufficient_p99_and_p95():
    rows = (
        [{"group": "A", "lat": v} for v in [0.1, 0.2, 0.3]]
        + [{"group": "B", "lat": v} for v in [float(i) for i in range(120)]]
    )
    out = schema_profile.per_subgroup_latency_summary(
        rows, field="lat", stratification_keys=["group"],
    )
    a = out["subgroups"]["A"]
    b = out["subgroups"]["B"]
    assert "INSUFFICIENT_SAMPLE_P95" in a["flags"]
    assert a["p95"] is None and a["p99"] is None
    assert not b["flags"]
    assert b["p99"] is not None


def test_stratify_indices_caps_each_stratum():
    rows = [{"k": "a"} for _ in range(10)] + [{"k": "b"} for _ in range(3)]
    kept, counts = schema_profile.stratify_indices(rows, ["k"], per_stratum_cap=4)
    assert counts["a"] == 4
    assert counts["b"] == 3
    assert len(kept) == 7


def test_hash_bucket_ids_deterministic():
    assert schema_profile.hash_bucket_ids([1, 2, 3]) == \
        schema_profile.hash_bucket_ids([1, 2, 3])
    assert schema_profile.hash_bucket_ids([1, 2, 3]) != \
        schema_profile.hash_bucket_ids([3, 2, 1])
    assert schema_profile.hash_bucket_ids([]) == ""


# ---------- 2. Extended schemas accept CARA + SwissAI fields --------------


def test_telemetry_record_accepts_cara_scheduler_state_fields():
    rec = schemas.TelemetryRecord(
        source_dataset_id="asdwb/cara_latency_prediction",
        trace_type="telemetry_trace",
        provenance="cara@test_flat",
        field_quality={
            "instance_id": "real",
            "instance_type": "real",
            "num_running": "real",
            "num_waiting": "real",
            "kv_cache_utilization": "real",
            "actual_e2e_latency_s": "real",
            "actual_ttft_s": "real",
            "actual_tpot_s": "real",
        },
        limitations=("cloudlab",),
        instance_id="c240g5", instance_type="qwen2.5-3b_p100",
        num_running=11, num_waiting=0,
        kv_cache_utilization=0.0,
        actual_e2e_latency_s=23.7, actual_ttft_s=0.13, actual_tpot_s=0.094,
    )
    assert rec.num_running == 11
    assert rec.kv_cache_utilization == 0.0


def test_cache_residency_record_accepts_swissai_bucket_fields():
    rec = schemas.CacheResidencyRecord(
        source_dataset_id="eth-easl/swissai-serving-trace",
        trace_type="cache_residency_trace",
        provenance="swissai@qwen3_32b_bucket_reuse",
        field_quality={
            "bucket_count": "real",
            "reused_bucket_count": "real",
            "reuse_percentage": "real",
            "bucket_ids_hash": "derived",
        },
        limitations=("synthetic-stable bucket ids",),
        bucket_count=7, reused_bucket_count=0, reuse_percentage=0.0,
        bucket_ids_hash="abc123",
    )
    assert rec.bucket_count == 7
    assert rec.bucket_ids_hash == "abc123"


def test_request_shape_record_accepts_swissai_iso_timestamps():
    rec = schemas.RequestShapeRecord(
        source_dataset_id="eth-easl/swissai-serving-trace",
        trace_type="request_shape_trace",
        provenance="swissai@trace",
        field_quality={
            "request_id": "real", "created_at_iso": "real",
            "finished_at_iso": "real", "status": "real",
            "model_id": "real", "temperature": "real", "seed": "real",
        },
        limitations=("anonymised ids",),
        request_id="abc", created_at_iso="2025-08-21T14:34:25.590Z",
        finished_at_iso="2025-08-21T14:34:26.945Z",
        status="ERROR", model_id="Qwen/Qwen3-32B",
        temperature=0.7, seed=42,
    )
    assert rec.status == "ERROR"
    assert rec.finished_at_iso.startswith("2025-08-21")


# ---------- 3. Ingestion mappings cover CARA + SwissAI -------------------


def test_telemetry_mapping_includes_cara_scheduler_state_columns():
    m = ingestion.RAW_TO_NORMALIZED["telemetry_trace"]
    for col in (
        "num_running", "num_waiting", "kv_cache_utilization", "kv_free_blocks",
        "ema_decode_tok_per_s", "actual_e2e_latency", "actual_ttft",
        "actual_tpot", "instance_id", "instance_type",
    ):
        assert col in m, f"telemetry_trace mapping missing CARA column {col}"


def test_cache_residency_mapping_includes_swissai_bucket_columns():
    m = ingestion.RAW_TO_NORMALIZED["cache_residency_trace"]
    for col in (
        "total_buckets", "reused_buckets", "reuse_percentage", "token_count",
        "bucket_ids_hash", "bucket_ids_sample",
    ):
        assert col in m, f"cache_residency_trace mapping missing SwissAI column {col}"


def test_request_shape_mapping_includes_swissai_columns():
    m = ingestion.RAW_TO_NORMALIZED["request_shape_trace"]
    for col in (
        "id", "created_at", "finished_at", "model", "status",
        "reported_token_input", "reported_token_output",
    ):
        assert col in m, f"request_shape_trace mapping missing SwissAI column {col}"


def test_normalize_renames_cara_telemetry_columns():
    raw = [{
        "request_id": "u1", "instance_id": "host:port",
        "instance_type": "qwen2.5-3b_p100",
        "num_prompt_tokens": 47, "num_predicted_output_tokens": 1024,
        "actual_output_tokens": 250, "actual_e2e_latency": 23.7,
        "actual_ttft": 0.13, "actual_tpot": 0.094,
        "num_running": 11, "num_waiting": 0,
        "kv_cache_utilization": 0.0, "kv_free_blocks": 6118,
        "ema_decode_tok_per_s": 10.0, "ema_prefill_tok_per_s": 289.0,
        "ema_decode_iter_ms": 95.99, "kv_evictions_per_s": 0.0,
        "prediction_timestamp": 1773889091.81,
        "completion_timestamp": 1773889115.86,
        "prediction_latency_ms": 22.2, "probe_latency_ms": 20.9,
        "num_active_decode_seqs": 0, "decode_ctx_p50": 0.0,
        "decode_ctx_p95": 0.0, "decode_ctx_max": 0.0,
        "pending_prefill_tokens": 0, "pending_decode_tokens": 0,
        "token_budget_per_iter": 0, "prefill_chunk_size": 0,
        "max_num_seqs": 0, "num_preempted": 0,
        "running_requests_count": 11, "waiting_requests_count": 0,
    }]
    normalized, unknown, fq = ingestion.normalize_rows(
        raw, "telemetry_trace", source_dataset_id="cara/test",
    )
    assert not unknown
    assert normalized[0]["actual_e2e_latency_s"] == 23.7
    assert normalized[0]["actual_ttft_s"] == 0.13
    assert normalized[0]["completion_timestamp_s"] == 1773889115.86


def test_normalize_renames_swissai_request_shape_columns():
    raw = [{
        "id": "abc:def:ghi", "status": "ERROR",
        "created_at": "2025-08-21T14:34:25.590Z",
        "finished_at": "2025-08-21T14:34:26.945Z",
        "model": "Qwen/Qwen3-32B",
        "reported_token_input": -1, "reported_token_output": -1,
        "model_parameters_json": "{}",
        "temperature": 0.7, "max_tokens_param": None, "top_p": 1, "seed": 42,
    }]
    normalized, unknown, _ = ingestion.normalize_rows(
        raw, "request_shape_trace", source_dataset_id="swissai/trace",
    )
    assert not unknown
    assert normalized[0]["request_id"] == "abc:def:ghi"
    assert normalized[0]["created_at_iso"] == "2025-08-21T14:34:25.590Z"
    assert normalized[0]["finished_at_iso"] == "2025-08-21T14:34:26.945Z"
    assert normalized[0]["model_id"] == "Qwen/Qwen3-32B"


# ---------- 4. Promotion-gate sample-strength enforcement ----------------


def _good_telemetry_summary(strength="strong", **overrides):
    base = {
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "test_flat",
        "source_url": "https://huggingface.co/datasets/asdwb/cara_latency_prediction",
        "license": "apache-2.0", "gated": False,
        "canonical_trace_type": "telemetry_trace",
        "committed_sample_rows": 5, "committed_sample_bytes": 1024,
        "sample_sha256": "f" * 64,
        "raw_schema": ["actual_e2e_latency", "num_running"],
        "normalized_schema": ["actual_e2e_latency_s", "num_running"],
        "unknown_columns": [],
        "field_quality": {
            "actual_e2e_latency_s": "real", "num_running": "real",
        },
        "available_signals": ["e2e_latency", "queue_depth", "cache_hit"],
        "missing_signals": ["replica_count"],
        "derived_fields": [], "proxy_fields": [], "synthetic_fields": [],
        "limitations": ["bounded test fixture"],
        "provenance": "test@v1", "ingestion_timestamp_s": 1.0,
        "statistical_sample_strength": strength,
        "fixture_sample_rows": 5,
        "analysis_sample_rows": (
            10_000 if strength == "strong"
            else (1_000 if strength == "moderate"
                  else (100 if strength == "weak" else 5))
        ),
    }
    base.update(overrides)
    return base


def test_strong_sample_telemetry_promoted_for_dynamic_calibration():
    s = _good_telemetry_summary(strength="strong")
    s["analysis_sample_rows"] = 50_000
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_dynamic_calibration"
    assert "promoted_for_dynamic_calibration" in d["promotion_tags"]


def test_moderate_sample_telemetry_downgrades_dynamic_calibration():
    s = _good_telemetry_summary(strength="moderate")
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_constraint_aware_evaluation"
    assert "promoted_for_dynamic_calibration" not in d["promotion_tags"]


def test_weak_sample_telemetry_only_training_priors_after_downgrade():
    # telemetry_trace allowed list is dynamic_cal + constraint_aware + backtest.
    # Weak strength satisfies none of those (min=moderate), so it falls back
    # to promoted_for_schema_only.
    s = _good_telemetry_summary(strength="weak")
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_schema_only"


def test_fixture_only_sample_blocks_performance_priors():
    s = _good_telemetry_summary(strength="fixture_only")
    s["analysis_sample_rows"] = 5
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_schema_only"
    assert all(
        t not in d["promotion_tags"]
        for t in (
            "promoted_for_performance_priors",
            "promoted_for_dynamic_calibration",
            "promoted_for_backtest",
            "promoted_for_constraint_aware_evaluation",
        )
    )


def test_strong_request_shape_sample_promoted_for_training_priors():
    s = _good_telemetry_summary(
        strength="strong",
        canonical_trace_type="request_shape_trace",
        available_signals=["prompt_tokens", "output_tokens"],
    )
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_training_priors"


def test_auth_blocked_short_circuit():
    s = _good_telemetry_summary(strength="strong")
    s["auth_status"] = "auth_blocked"
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "auth_blocked"


def test_missing_sample_strength_falls_back_to_schema_only():
    s = _good_telemetry_summary()
    del s["statistical_sample_strength"]
    # Without the strength label, gate still passes (we record fixture/analysis
    # rows), but the sample-strength filter cannot match anything beyond
    # promoted_for_schema_only.
    d = promotion.evaluate_promotion(s)
    assert d["state"] == "promoted_for_schema_only"


# ---------- 5. PROMOTION_TAG_MIN_SAMPLE_STRENGTH coverage ---------------


def test_promotion_min_sample_strength_table_covers_all_promoted_states():
    promoted_states = {
        s for s in promotion.PROMOTION_STATES
        if s.startswith("promoted_for_")
    }
    for s in promoted_states:
        assert s in promotion.PROMOTION_TAG_MIN_SAMPLE_STRENGTH, (
            f"sample-strength table missing entry for {s}"
        )


def test_sample_strength_ordering_monotone():
    o = promotion._SAMPLE_STRENGTH_ORDER
    assert o["fixture_only"] < o["weak"] < o["moderate"] < o["strong"]


# ---------- 6. Audit artefact paths follow the documented layout --------


def test_audit_artefact_paths_match_spec(tmp_path):
    """The summary written by the audit script must point at
    schema_profile.json + schema_mapping.json next to it."""
    summary_path = tmp_path / "data" / "external" / "hf" / "asdwb__cara_latency_prediction" / "test_flat" / "processed" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "dataset_id": "asdwb/cara_latency_prediction",
        "config_name": "test_flat",
        "schema_profile_path": (
            "data/external/hf/asdwb__cara_latency_prediction/test_flat/"
            "processed/schema_profile.json"
        ),
        "schema_mapping_path": (
            "data/external/hf/asdwb__cara_latency_prediction/test_flat/"
            "processed/schema_mapping.json"
        ),
    }))
    loaded = json.loads(summary_path.read_text())
    assert loaded["schema_profile_path"].endswith("schema_profile.json")
    assert loaded["schema_mapping_path"].endswith("schema_mapping.json")


# ---------- 7. The committed audit summaries are well-formed ------------


REPO_DATA = os.path.join(REPO_ROOT, "data", "external", "hf")


@pytest.mark.parametrize("safe_id,config,trace_type,strength,trust_tier", [
    ("asdwb__cara_latency_prediction", "test_flat", "telemetry_trace",
     "moderate", "tier_2_public_telemetry_traces"),
    ("asdwb__cara_latency_prediction", "test_queue_details", "telemetry_trace",
     "moderate", "tier_2_public_telemetry_traces"),
    ("eth-easl__swissai-serving-trace", "trace", "request_shape_trace",
     "strong", "tier_5_request_shape_traces"),
    ("eth-easl__swissai-serving-trace", "qwen3_32b_buckets",
     "cache_residency_trace", "strong", "tier_4_latency_benchmark_traces"),
    ("eth-easl__swissai-serving-trace", "qwen3_32b_bucket_reuse",
     "cache_residency_trace", "strong", "tier_4_latency_benchmark_traces"),
])
def test_audit_summary_invariants(
    safe_id, config, trace_type, strength, trust_tier,
):
    path = os.path.join(REPO_DATA, safe_id, config, "processed", "summary.json")
    if not os.path.exists(path):
        pytest.skip(f"audit not run for {safe_id}/{config}; run "
                    "scripts/audit_cara_swissai_telemetry.py first")
    with open(path) as fh:
        s = json.load(fh)
    assert s["canonical_trace_type"] == trace_type
    assert s["trust_tier"] == trust_tier
    assert s["statistical_sample_strength"] == strength
    assert s["analysis_sample_rows"] >= 1_000  # all five reach analysis-tier
    assert s["fixture_sample_rows"] >= 1
    assert s["committed_sample_bytes"] <= promotion.MAX_COMMITTED_SAMPLE_BYTES
    assert s["unknown_columns"] == []  # strict-schema invariant
    assert isinstance(s["limitations"], list) and s["limitations"]
    # Schema profile + mapping artefact paths exist as files.
    for k in ("schema_profile_path", "schema_mapping_path"):
        assert os.path.exists(os.path.join(REPO_ROOT, s[k])), \
            f"{safe_id}/{config} -> missing {k} artefact"


@pytest.mark.parametrize("safe_id,config", [
    ("asdwb__cara_latency_prediction", "test_flat"),
    ("asdwb__cara_latency_prediction", "test_queue_details"),
    ("eth-easl__swissai-serving-trace", "trace"),
    ("eth-easl__swissai-serving-trace", "qwen3_32b_buckets"),
    ("eth-easl__swissai-serving-trace", "qwen3_32b_bucket_reuse"),
])
def test_schema_profile_artefact_covers_every_observed_column(safe_id, config):
    profile_path = os.path.join(
        REPO_DATA, safe_id, config, "processed", "schema_profile.json")
    mapping_path = os.path.join(
        REPO_DATA, safe_id, config, "processed", "schema_mapping.json")
    if not os.path.exists(profile_path):
        pytest.skip("audit not yet run")
    with open(profile_path) as fh:
        profile = json.load(fh)
    with open(mapping_path) as fh:
        mapping = json.load(fh)
    observed = set(profile["raw_columns"]) | set(profile["nested_keys"])
    accepted = set(mapping["accepted_columns"])
    rejected = set(mapping["rejected_columns"])
    assert accepted.union(rejected) == observed, (
        f"{safe_id}/{config}: accepted+rejected do not cover all observed "
        f"columns; missing="
        f"{observed - accepted - rejected}"
    )
    assert not rejected, (
        f"{safe_id}/{config}: rejected_columns={rejected}; "
        "every observed column must be classified in MAPPINGS"
    )


# ---------- 8. Audit results are recorded in the canonical registry -----


def test_canonical_registry_includes_cara_and_swissai_entries():
    path = os.path.join(
        REPO_ROOT, "data", "external", "hf_discovery",
        "canonical_corpus_registry.json",
    )
    if not os.path.exists(path):
        pytest.skip("registry not present")
    with open(path) as fh:
        registry = json.load(fh)
    ds_ids = {e["dataset_id"] for e in registry["entries"]}
    assert "asdwb/cara_latency_prediction" in ds_ids
    assert "eth-easl/swissai-serving-trace" in ds_ids
    # Every CARA / SwissAI entry must carry the new sample-policy fields.
    for e in registry["entries"]:
        if e["dataset_id"] in (
            "asdwb/cara_latency_prediction", "eth-easl/swissai-serving-trace",
        ):
            assert "statistical_sample_strength" in e
            assert e["sampling_method"] in ("head", "stratified", "full_bounded")


def test_raw_files_are_gitignored_not_committed():
    """The audit must NEVER commit raw 10 MiB chunks under data/external/hf/<safe>/raw/."""
    cara_raw = os.path.join(
        REPO_ROOT, "data", "external", "hf", "asdwb__cara_latency_prediction",
        "raw", "test.jsonl",
    )
    swiss_raw = os.path.join(
        REPO_ROOT, "data", "external", "hf", "eth-easl__swissai-serving-trace",
        "raw", "trace.jsonl",
    )
    import subprocess
    for raw in (cara_raw, swiss_raw):
        if not os.path.exists(raw):
            continue  # the audit hasn't been run on this checkout
        r = subprocess.run(
            ["git", "check-ignore", raw], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        # check-ignore returns 0 iff the file is ignored.
        assert r.returncode == 0, (
            f"raw audit download not gitignored: {raw}"
        )
