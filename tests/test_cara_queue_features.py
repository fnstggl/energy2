"""Tests for the CARA queue-wait feature pipeline + target honesty."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cara_queue_features import (  # noqa: E402
    MEASURED_QUEUE_WAIT_AVAILABLE,
    QUEUE_LEAKAGE_TARGET_FIELDS,
    QUEUE_TARGET_NAMES,
    QueueLeakageError,
    build_queue_feature_matrix,
    build_queue_feature_spec,
    derive_queue_wait_s,
    extract_queue_target,
    queue_pressure_score,
    target_field_quality,
)


def _row(**ov):
    base = {
        "request_id": "abc", "instance_type": "qwen2.5-3b_p100",
        "num_prompt_tokens": 47, "num_predicted_output_tokens": 1024,
        "actual_output_tokens": 250, "actual_e2e_latency_s": 23.7,
        "actual_ttft_s": 0.13, "actual_tpot_s": 0.094,
        "num_running": 11, "num_waiting": 0, "num_active_decode_seqs": 0,
        "decode_ctx_p50": 0.0, "decode_ctx_p95": 0.0, "decode_ctx_max": 0.0,
        "pending_prefill_tokens": 0, "pending_decode_tokens": 0,
        "kv_cache_utilization": 0.0, "kv_free_blocks": 6118,
        "token_budget_per_iter": 0, "prefill_chunk_size": 0,
        "max_num_seqs": 0, "num_preempted": 0,
        "ema_decode_tok_per_s": 10.4, "ema_prefill_tok_per_s": 289.0,
        "ema_decode_iter_ms": 96.0, "kv_evictions_per_s": 0.0,
        "running_requests_count": 11, "waiting_requests_count": 0,
        "prediction_timestamp_s": 1000.0,
        "completion_timestamp_s": 1024.0,
    }
    base.update(ov)
    return base


# ---------- 1. Target is explicit + honest -------------------------------


def test_measured_queue_wait_not_available():
    assert MEASURED_QUEUE_WAIT_AVAILABLE is False


def test_target_names_explicit():
    assert QUEUE_TARGET_NAMES == ("derived_queue_wait_s", "queue_pressure_score")


def test_target_field_quality_labels_are_honest():
    assert target_field_quality("derived_queue_wait_s") == "derived"
    assert target_field_quality("queue_pressure_score") == "synthetic"
    assert target_field_quality("measured_queue_wait_s") == "missing"


def test_extract_target_refuses_measured_queue_wait():
    with pytest.raises(QueueLeakageError):
        extract_queue_target([_row()], "measured_queue_wait_s")


def test_extract_target_rejects_unknown_target():
    with pytest.raises(ValueError):
        extract_queue_target([_row()], "not_a_target")


# ---------- 2. Derived queue wait computation ----------------------------


def test_derive_queue_wait_is_dispatch_gap():
    # (1024 - 1000) - 23.7 = 0.3
    r = _row(prediction_timestamp_s=1000.0, completion_timestamp_s=1024.0,
             actual_e2e_latency_s=23.7)
    assert derive_queue_wait_s(r) == pytest.approx(0.3)


def test_derive_queue_wait_clamps_negative_to_zero():
    r = _row(prediction_timestamp_s=1000.0, completion_timestamp_s=1010.0,
             actual_e2e_latency_s=15.0)  # 10 - 15 = -5 -> 0
    assert derive_queue_wait_s(r) == 0.0


def test_derive_queue_wait_none_on_missing_field():
    r = _row()
    del r["completion_timestamp_s"]
    assert derive_queue_wait_s(r) is None


def test_extract_derived_target_array():
    rows = [_row(prediction_timestamp_s=0.0, completion_timestamp_s=10.0,
                 actual_e2e_latency_s=9.0)]
    y = extract_queue_target(rows, "derived_queue_wait_s")
    assert y.shape == (1,)
    assert y[0] == pytest.approx(1.0)


# ---------- 3. Queue pressure score is synthetic -------------------------


def test_queue_pressure_score_combines_state():
    r = _row(num_running=5, num_waiting=2, pending_prefill_tokens=512,
             pending_decode_tokens=0)
    # 5 + 4*2 + 512/512 + 0 = 14.0
    assert queue_pressure_score(r) == pytest.approx(14.0)


# ---------- 4. Leakage exclusions ----------------------------------------


def test_queue_leakage_set_includes_latency_and_targets():
    for f in ("actual_e2e_latency_s", "actual_ttft_s", "completion_timestamp_s",
              "derived_queue_wait_s", "queue_pressure_score",
              "prediction_timestamp_s"):
        assert f in QUEUE_LEAKAGE_TARGET_FIELDS


def test_queue_feature_spec_excludes_leakage():
    rows = [_row()] * 5
    spec = build_queue_feature_spec(rows)
    for c in spec.numeric_columns:
        assert c not in QUEUE_LEAKAGE_TARGET_FIELDS, (
            f"queue feature spec emitted leakage column {c}"
        )


def test_queue_feature_matrix_has_no_leakage_names():
    rows = [_row(instance_type=it) for it in (
        "qwen2.5-3b_p100", "qwen2.5-7b_a30", "qwen2.5-72b_a100")]
    spec = build_queue_feature_spec(rows)
    X, names, groups = build_queue_feature_matrix(rows, spec)
    assert X.shape[0] == 3
    for n in names:
        assert n not in QUEUE_LEAKAGE_TARGET_FIELDS
    # Group arrays for subgroup audit must be present.
    for g in ("instance_type", "gpu_type", "model_size",
              "prompt_token_bin", "queue_depth_bin", "kv_util_bin"):
        assert g in groups
