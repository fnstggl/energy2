"""Tests for the cache / prefix-reuse forecaster feature pipeline.

The pipeline must:
- block target-as-feature leakage at spec-build time,
- block post-decision fields (ttft, e2e latency, cache_hit) as features,
- compute rolling features in chronological order (no future leak),
- parse SwissAI ISO timestamps in both common formats,
- bin numeric features using pre-registered constants,
- preserve field-quality categorisation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cache_prefix_features import (  # noqa: E402
    BUCKET_SIZE_BINS,
    HIGH_REUSE_THRESHOLD,
    INPUT_TOKEN_BINS,
    LEAKAGE_TARGET_FIELDS,
    PREDICT_TIME_NUMERIC_FEATURES,
    TARGETS,
    LeakageError,
    _parse_swissai_iso,
    add_rolling_features,
    assert_no_leakage,
    bin_bucket_size,
    bin_input_tokens,
    build_feature_matrix,
    build_feature_spec,
    derive_high_reuse,
    derive_intra_session_reuse_from_cc_traces,
    enrich_row,
    extract_reuse_percentage,
    holdout_by_group,
    holdout_by_session,
    hour_of_day_label,
    random_holdout,
    time_holdout,
)

# ---------- 1. Leakage rules --------------------------------------------


def test_leakage_target_fields_includes_reuse_percentage_and_actual_latency():
    assert "reuse_percentage" in LEAKAGE_TARGET_FIELDS
    assert "reused_buckets" in LEAKAGE_TARGET_FIELDS
    assert "reused_bucket_count" in LEAKAGE_TARGET_FIELDS
    assert "actual_e2e_latency_s" in LEAKAGE_TARGET_FIELDS
    assert "actual_ttft_s" in LEAKAGE_TARGET_FIELDS
    assert "ttft_s" in LEAKAGE_TARGET_FIELDS
    assert "api_time_s" in LEAKAGE_TARGET_FIELDS
    assert "cache_hit" in LEAKAGE_TARGET_FIELDS
    assert "completion_timestamp_s" in LEAKAGE_TARGET_FIELDS


def test_leakage_blocker_raises_when_feature_in_blocklist():
    with pytest.raises(LeakageError):
        assert_no_leakage(("bucket_count", "reuse_percentage"))
    with pytest.raises(LeakageError):
        assert_no_leakage(("ttft_s",))


def test_predict_time_features_are_leakage_free():
    # Mission spec: pipeline outputs must be leakage-free.
    assert_no_leakage(PREDICT_TIME_NUMERIC_FEATURES)
    for name in PREDICT_TIME_NUMERIC_FEATURES:
        assert name not in LEAKAGE_TARGET_FIELDS


def test_build_feature_spec_refuses_leakage_columns(monkeypatch):
    # If someone tries to extend the numeric feature list with a leakage
    # field, spec construction must refuse.
    import aurelius.forecasting.cache_prefix_features as mod
    monkeypatch.setattr(
        mod, "PREDICT_TIME_NUMERIC_FEATURES",
        mod.PREDICT_TIME_NUMERIC_FEATURES + ("reuse_percentage",),
    )
    with pytest.raises(LeakageError):
        mod.build_feature_spec([{"model_id": "x"}])


# ---------- 2. Target derivation -----------------------------------------


def test_derive_high_reuse_uses_50pct_threshold():
    y_pct = np.array([0.0, 49.9, 50.0, 75.0, 100.0, np.nan])
    high = derive_high_reuse(y_pct)
    assert high[0] == 0.0
    assert high[1] == 0.0
    assert high[2] == 1.0
    assert high[3] == 1.0
    assert high[4] == 1.0
    assert np.isnan(high[5])
    assert HIGH_REUSE_THRESHOLD == 50.0


def test_extract_reuse_percentage_returns_nan_for_missing():
    rows = [{"reuse_percentage": 75.0}, {"reuse_percentage": None}, {}]
    y = extract_reuse_percentage(rows)
    assert y[0] == 75.0
    assert np.isnan(y[1])
    assert np.isnan(y[2])


def test_derive_intra_session_reuse_marks_repeated_hash_within_session():
    rows = [
        # Two sessions: A has a repeated hash on turn 1 -> reuse;
        # B's count grows without resetting -> reuse on turn 2.
        {"session_id": "A", "turn_index": 0, "block_hashes_hash": "h1",
         "block_hashes_count": 5},
        {"session_id": "A", "turn_index": 1, "block_hashes_hash": "h1",
         "block_hashes_count": 5},
        {"session_id": "B", "turn_index": 0, "block_hashes_hash": "h2",
         "block_hashes_count": 10},
        {"session_id": "B", "turn_index": 1, "block_hashes_hash": "h3",
         "block_hashes_count": 12},
    ]
    y = derive_intra_session_reuse_from_cc_traces(rows)
    assert y[0] == 0.0     # first turn — no prior
    assert y[1] == 1.0     # same hash as prior
    assert y[2] == 0.0     # first turn of B
    assert y[3] == 1.0     # count grew without resetting


# ---------- 3. Timestamp parsing ----------------------------------------


def test_parse_swissai_iso_handles_both_formats():
    assert _parse_swissai_iso("2025-05-23 15:05:19.910") is not None
    assert _parse_swissai_iso("2025-10-10T16:17:11.338Z") is not None
    assert _parse_swissai_iso("2024-12-09 21:20:07.756") is not None
    assert _parse_swissai_iso(None) is None
    assert _parse_swissai_iso("not-a-date") is None


def test_hour_of_day_label_returns_missing_for_none():
    assert hour_of_day_label(None) == "missing"
    # 2025-05-23 15:00:00 UTC -> hour 15.
    ts = _parse_swissai_iso("2025-05-23 15:30:00.000")
    assert hour_of_day_label(ts) == "hour=15"


# ---------- 4. Binning ---------------------------------------------------


def test_bucket_size_bin_uses_pre_registered_boundaries():
    assert bin_bucket_size(0) == "[0,2)"
    assert bin_bucket_size(1) == "[0,2)"
    assert bin_bucket_size(2) == "[2,16)"
    assert bin_bucket_size(500) == "[256,1024)"
    assert bin_bucket_size(None) == "missing"
    assert bin_bucket_size(20_000_000) == ">=10000000"


def test_input_token_bin_handles_missing():
    assert bin_input_tokens(None) == "missing"
    assert bin_input_tokens(100) == "[0,256)"
    assert bin_input_tokens(5000) == "[4096,16384)"


def test_bin_boundaries_are_constants_not_fitted():
    # Mission spec: bin boundaries are NEVER fitted from data.
    assert BUCKET_SIZE_BINS[0] == (0, 2)
    assert INPUT_TOKEN_BINS[0] == (0, 256)


# ---------- 5. Rolling features (chronological) --------------------------


def test_rolling_features_use_only_prior_rows():
    # Construct two rows with the same bucket_ids_hash; row 1's
    # rolling_per_hash_seen_count must be 0 (no prior rows), row 2 must
    # be 1 (row 1 is now prior).
    rows = [
        {"model_id": "m", "bucket_ids_hash": "H",
         "created_at_iso": "2025-05-23 10:00:00.000",
         "reuse_percentage": 100.0, "bucket_count": 1, "session_id": "s"},
        {"model_id": "m", "bucket_ids_hash": "H",
         "created_at_iso": "2025-05-23 11:00:00.000",
         "reuse_percentage": 100.0, "bucket_count": 1, "session_id": "s"},
        {"model_id": "m", "bucket_ids_hash": "H",
         "created_at_iso": "2025-05-23 12:00:00.000",
         "reuse_percentage": 100.0, "bucket_count": 1, "session_id": "s"},
    ]
    enriched = add_rolling_features(rows, source="swissai")
    # Output retains original list order; first-by-time has 0 prior.
    seen = [r["rolling_per_hash_seen_count"] for r in enriched]
    assert seen == [0.0, 1.0, 2.0]


def test_rolling_per_model_reuse_is_chronological_and_excludes_future():
    rows = [
        {"model_id": "X", "created_at_iso": "2025-05-23 10:00:00.000",
         "reuse_percentage": 100.0, "bucket_count": 1, "session_id": "s1"},
        {"model_id": "X", "created_at_iso": "2025-05-23 11:00:00.000",
         "reuse_percentage": 0.0, "bucket_count": 1, "session_id": "s2"},
        {"model_id": "X", "created_at_iso": "2025-05-23 12:00:00.000",
         "reuse_percentage": 100.0, "bucket_count": 1, "session_id": "s3"},
    ]
    enriched = add_rolling_features(rows, source="swissai")
    # First row: no prior, NaN.
    assert np.isnan(enriched[0]["rolling_per_model_reuse_pct"])
    # Second row: prior is row 0 (100.0). Rolling mean = 100.0.
    assert enriched[1]["rolling_per_model_reuse_pct"] == pytest.approx(100.0)
    # Third row: priors are rows 0 (100.0) and 1 (0.0). Mean = 50.0.
    assert enriched[2]["rolling_per_model_reuse_pct"] == pytest.approx(50.0)


def test_session_turns_so_far_grows_within_session():
    rows = [
        {"session_id": "S", "created_at_iso": "2025-05-23 10:00:00.000"},
        {"session_id": "S", "created_at_iso": "2025-05-23 10:01:00.000"},
        {"session_id": "T", "created_at_iso": "2025-05-23 10:02:00.000"},
        {"session_id": "S", "created_at_iso": "2025-05-23 10:03:00.000"},
    ]
    enriched = add_rolling_features(rows, source="cc_traces")
    turns = [r["session_turns_so_far"] for r in enriched]
    assert turns == [0.0, 1.0, 0.0, 2.0]


# ---------- 6. Feature-matrix shape + group keys ------------------------


def test_build_feature_matrix_produces_expected_shape_and_groups():
    rows = [
        {"model_id": "m1", "bucket_count": 5, "input_tokens": 100,
         "reuse_percentage": 100.0,
         "created_at_iso": "2025-05-23 10:00:00.000", "session_id": "s1"},
        {"model_id": "m2", "bucket_count": 100, "input_tokens": 2000,
         "reuse_percentage": 0.0,
         "created_at_iso": "2025-05-23 11:00:00.000", "session_id": "s2"},
    ]
    enriched = add_rolling_features(rows, source="swissai")
    spec = build_feature_spec(enriched)
    X, names, gk = build_feature_matrix(enriched, spec)
    assert X.shape[0] == 2
    assert len(names) == X.shape[1]
    assert "model_id" in gk
    assert "bucket_size_bin" in gk
    assert gk["model_id"][0] == "m1"


def test_build_feature_spec_encodes_unseen_levels_with_sentinel():
    train = [
        {"model_id": "A", "request_type": "s",
         "created_at_iso": "2025-05-23 10:00:00.000"},
    ]
    enriched = add_rolling_features(train, source="swissai")
    spec = build_feature_spec(enriched)
    holdout = [
        {"model_id": "B", "request_type": "n",
         "created_at_iso": "2025-05-23 12:00:00.000"},
    ]
    enriched_h = add_rolling_features(holdout, source="swissai")
    Xh, names_h, _ = build_feature_matrix(enriched_h, spec)
    # Sentinel column must be active for each categorical.
    sentinel_idx = [i for i, n in enumerate(names_h) if "=__UNSEEN__" in n]
    assert len(sentinel_idx) >= 1
    assert Xh[0, sentinel_idx].sum() > 0


# ---------- 7. Holdout helpers ------------------------------------------


def test_random_holdout_is_deterministic():
    tr1, te1 = random_holdout(100, seed=42)
    tr2, te2 = random_holdout(100, seed=42)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(te1, te2)
    assert len(set(tr1.tolist()) & set(te1.tolist())) == 0


def test_time_holdout_returns_last_chronologically():
    timestamps = np.array([10.0, 5.0, 100.0, 50.0, 20.0])
    tr, te = time_holdout(timestamps, holdout_frac=0.2)
    # The chronologically-last (100.0) is at index 2.
    assert 2 in te.tolist()


def test_holdout_by_session_returns_disjoint_session_sets():
    sids = np.array(["A", "A", "B", "B", "C", "C", "D", "D"], dtype=object)
    tr, te = holdout_by_session(sids, holdout_frac=0.5, seed=7)
    train_sids = set(sids[tr].tolist())
    test_sids = set(sids[te].tolist())
    assert train_sids.isdisjoint(test_sids)
    assert len(train_sids) + len(test_sids) == 4


def test_holdout_by_session_handles_none_ids():
    sids = np.array([None, None, "X", "X", "Y", "Y"], dtype=object)
    tr, te = holdout_by_session(sids, holdout_frac=0.5, seed=1)
    assert tr.size + te.size == 6
    # No raise on None.


def test_holdout_by_group_holds_out_named_groups():
    groups = np.array(["m1", "m1", "m2", "m2", "m3", "m3"], dtype=object)
    tr, te = holdout_by_group(groups, ("m2",))
    assert set(groups[te].tolist()) == {"m2"}
    assert "m2" not in set(groups[tr].tolist())


# ---------- 8. Field-quality preservation -------------------------------


def test_enrich_row_marks_decision_timestamp_from_iso():
    r = {"created_at_iso": "2025-05-23 15:05:19.910",
         "bucket_count": 4, "input_tokens": 1000, "model_id": "m"}
    enriched = enrich_row(r, source="swissai")
    assert enriched["__decision_timestamp_s"] is not None
    assert enriched["bucket_size_bin"] == "[2,16)"
    assert enriched["input_token_bin"] == "[256,1024)"
    assert enriched["hour_of_day"] == "hour=15"


def test_field_quality_targets_present():
    assert "reuse_percentage" in TARGETS
    assert "high_reuse" in TARGETS
    assert "intra_session_reuse" in TARGETS
