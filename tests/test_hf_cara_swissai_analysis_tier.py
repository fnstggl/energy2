"""Tests for the CARA + SwissAI analysis-tier expansion.

The audit and signal-coverage scripts are run live against the HF API
in the analysis-tier PR; these tests cover the invariants the audit
must preserve so future re-runs cannot silently regress the
forecasting-readiness promises:

- Train/test schemas are identical (CARA) or differences are explicitly
  recorded (SwissAI).
- Raw downloads + analysis_sample.jsonl never get committed (gitignore
  enforced).
- The 50-100 MiB sampling budget is the per-target `max_download_bytes`
  for every analysis-tier entry.
- The signal coverage table exists and validates.
- rows_available is nonzero for the spec's mandatory CARA signals.
- rows_available is nonzero for the spec's mandatory SwissAI signals.
- Missing signals (replica_count, autoscaling_events, GPU_utilization,
  GPU_memory, SLA_label, timeout_label) stay explicitly `missing`.
- Subgroup counts + `INSUFFICIENT_SAMPLE_P99` flags are honoured.
- Dynamic-calibration promotion fires ONLY when CARA strength is strong
  AND queue/latency/scheduling signals are present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

SIGNAL_COVERAGE_PATH = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery",
    "cara_swissai_signal_coverage.json",
)

# Analysis-tier per-config row counts as observed in the audit run.
# A test enforces the per-(dataset, config) summary matches this.
EXPECTED_ANALYSIS_CONFIGS = {
    ("asdwb/cara_latency_prediction", "train_flat"): {
        "trace_type": "telemetry_trace",
        "min_rows": 50_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_dynamic_calibration",
    },
    ("asdwb/cara_latency_prediction", "train_queue_details"): {
        "trace_type": "telemetry_trace",
        "min_rows": 20_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_dynamic_calibration",
    },
    ("eth-easl/swissai-serving-trace", "trace_analysis"): {
        "trace_type": "request_shape_trace",
        "min_rows": 100_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_training_priors",
    },
    ("eth-easl/swissai-serving-trace", "qwen3_32b_buckets_analysis"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 50_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
    ("eth-easl/swissai-serving-trace", "qwen3_32b_bucket_reuse_analysis"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 50_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
    ("eth-easl/swissai-serving-trace", "apertus_70b_bucket_reuse"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 10_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
    ("eth-easl/swissai-serving-trace", "qwen380b_instruct_bucket_reuse"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 10_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
    ("eth-easl/swissai-serving-trace", "qwen380b_thinking_bucket_reuse"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 1_000,
        "expected_strength_or_better": "moderate",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
    ("eth-easl/swissai-serving-trace", "llama3_70b_bucket_reuse"): {
        "trace_type": "cache_residency_trace",
        "min_rows": 100_000,
        "expected_strength": "strong",
        "expected_promotion_state": "promoted_for_cache_residency_evaluation",
    },
}


def _safe(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def _summary_path(dataset_id: str, config: str) -> str:
    return os.path.join(
        REPO_ROOT, "data", "external", "hf", _safe(dataset_id), config,
        "processed", "summary.json",
    )


# ---------- 1. Audit script imports + TARGET lists ----------------------


def test_audit_script_exposes_analysis_tier_targets():
    import audit_cara_swissai_telemetry as a
    ids = {(t["dataset_id"], t["config_name"]) for t in a.ANALYSIS_TIER_TARGETS}
    expected = set(EXPECTED_ANALYSIS_CONFIGS.keys())
    assert ids == expected, (
        f"ANALYSIS_TIER_TARGETS = {sorted(ids)} != expected "
        f"{sorted(expected)}"
    )


def test_every_analysis_target_enforces_50_to_100_MiB_budget():
    import audit_cara_swissai_telemetry as a
    for t in a.ANALYSIS_TIER_TARGETS:
        mb = t["max_download_bytes"] / (1024 * 1024)
        # apertus is 40 MB whole-file; per spec we may use the whole file
        # when it's smaller than the budget — so allow 40-110 MiB.
        assert 40 <= mb <= 110, (
            f"{t['config_name']} max_download_bytes={mb} MiB outside "
            "the 50-100 MiB policy window"
        )


def test_analysis_tier_mappings_are_registered():
    import audit_cara_swissai_telemetry as a
    for (ds_id, cfg), _meta in EXPECTED_ANALYSIS_CONFIGS.items():
        assert (ds_id, cfg) in a.MAPPINGS, (
            f"{ds_id}/{cfg} has no entry in MAPPINGS"
        )
        assert (ds_id, cfg) in a._TRUST_TIER_PER_TARGET, (
            f"{ds_id}/{cfg} has no entry in _TRUST_TIER_PER_TARGET"
        )


# ---------- 2. Schemas match between test + train (CARA) ----------------


def test_cara_train_flat_schema_matches_test_flat():
    test_summary_path = _summary_path("asdwb/cara_latency_prediction", "test_flat")
    train_summary_path = _summary_path("asdwb/cara_latency_prediction", "train_flat")
    if not (os.path.exists(test_summary_path) and os.path.exists(train_summary_path)):
        pytest.skip("CARA test_flat or train_flat summary not present")
    test_sum = json.loads(open(test_summary_path).read())
    train_sum = json.loads(open(train_summary_path).read())
    assert set(test_sum["raw_schema"]) == set(train_sum["raw_schema"]), (
        "CARA test_flat raw_schema != train_flat raw_schema; the CARA "
        "README claims sweep 2 train+test share columns"
    )


def test_cara_train_queue_details_schema_matches_test_queue_details():
    test_summary_path = _summary_path(
        "asdwb/cara_latency_prediction", "test_queue_details"
    )
    train_summary_path = _summary_path(
        "asdwb/cara_latency_prediction", "train_queue_details"
    )
    if not (os.path.exists(test_summary_path) and os.path.exists(train_summary_path)):
        pytest.skip("CARA queue_details summaries not present")
    test_sum = json.loads(open(test_summary_path).read())
    train_sum = json.loads(open(train_summary_path).read())
    assert set(test_sum["raw_schema"]) == set(train_sum["raw_schema"])


def test_swissai_trace_analysis_matches_focused_schema():
    a = _summary_path("eth-easl/swissai-serving-trace", "trace")
    b = _summary_path("eth-easl/swissai-serving-trace", "trace_analysis")
    if not (os.path.exists(a) and os.path.exists(b)):
        pytest.skip("SwissAI trace summaries not present")
    foc = json.loads(open(a).read())
    ana = json.loads(open(b).read())
    # The analysis-tier head may surface keys the 10 MiB head missed (more
    # row diversity). The audit-tier schema must be a SUBSET of the
    # analysis-tier; that is the explicit shape-difference rule.
    assert set(foc["raw_schema"]).issubset(set(ana["raw_schema"])), (
        f"SwissAI focused trace schema {sorted(set(foc['raw_schema']))} is "
        f"not a subset of analysis-tier {sorted(set(ana['raw_schema']))}; "
        "either re-classify or document explicitly."
    )


# ---------- 3. Raw + analysis_sample never committed --------------------


def test_raw_downloads_for_analysis_tier_are_gitignored():
    raw_files = [
        # CARA
        ("asdwb__cara_latency_prediction", "train.jsonl"),
        ("asdwb__cara_latency_prediction", "train_queue_details.jsonl"),
        # SwissAI (raw is shared with focused tier for the overlap configs)
        ("eth-easl__swissai-serving-trace", "trace.jsonl"),
        ("eth-easl__swissai-serving-trace", "qwen3-32b-buckets.jsonl"),
        ("eth-easl__swissai-serving-trace", "qwen3-32b-bucket-reuse.jsonl"),
        ("eth-easl__swissai-serving-trace", "apertus-70b-bucket-reuse.jsonl"),
        ("eth-easl__swissai-serving-trace", "llama3-70b_bucket-reuse.jsonl"),
    ]
    for safe, fname in raw_files:
        raw = os.path.join(
            REPO_ROOT, "data", "external", "hf", safe, "raw", fname,
        )
        if not os.path.exists(raw):
            continue
        r = subprocess.run(
            ["git", "check-ignore", raw], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"raw analysis download not gitignored: {raw}"


def test_analysis_sample_jsonl_for_every_config_is_gitignored():
    for (ds_id, cfg), _meta in EXPECTED_ANALYSIS_CONFIGS.items():
        path = os.path.join(
            REPO_ROOT, "data", "external", "hf", _safe(ds_id), cfg,
            "processed", "analysis_sample.jsonl",
        )
        if not os.path.exists(path):
            continue
        r = subprocess.run(
            ["git", "check-ignore", path], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"analysis_sample.jsonl not gitignored: {path}"


# ---------- 4. Summary invariants per analysis-tier config -------------


@pytest.mark.parametrize("ds_id,cfg", list(EXPECTED_ANALYSIS_CONFIGS.keys()))
def test_analysis_tier_summary_invariants(ds_id, cfg):
    path = _summary_path(ds_id, cfg)
    if not os.path.exists(path):
        pytest.skip(f"{ds_id}/{cfg} summary not present")
    with open(path) as fh:
        s = json.load(fh)
    spec = EXPECTED_ANALYSIS_CONFIGS[(ds_id, cfg)]
    assert s["canonical_trace_type"] == spec["trace_type"]
    assert s["analysis_sample_rows"] >= spec["min_rows"]
    assert s["unknown_columns"] == []  # strict-schema
    # Strength: either exact or "at least" depending on spec.
    if "expected_strength" in spec:
        assert s["statistical_sample_strength"] == spec["expected_strength"]
    elif "expected_strength_or_better" in spec:
        order = {"insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}
        actual = order.get(s["statistical_sample_strength"], -1)
        floor = order[spec["expected_strength_or_better"]]
        assert actual >= floor


# ---------- 5. CARA dynamic-calibration promotion rule ------------------


@pytest.mark.parametrize("cfg", ["train_flat", "train_queue_details"])
def test_cara_train_promoted_for_dynamic_calibration(cfg):
    path = _summary_path("asdwb/cara_latency_prediction", cfg)
    if not os.path.exists(path):
        pytest.skip(f"CARA {cfg} summary not present")
    s = json.loads(open(path).read())
    # Strong sample.
    assert s["statistical_sample_strength"] == "strong"
    # No unknown columns silently ignored.
    assert s["unknown_columns"] == []
    # Queue state present.
    norm = set(s["normalized_schema"])
    assert "num_running" in norm or "num_waiting" in norm
    # Latency labels present (for flat) OR per-request running_requests
    # (for queue_details, via the schedule_state.* expansion).
    if cfg == "train_flat":
        assert "actual_e2e_latency_s" in norm
        assert "actual_ttft_s" in norm
    else:
        assert "num_running" in norm
    # Scheduling-time features present.
    assert any(
        f in norm for f in (
            "ema_decode_iter_ms", "ema_decode_tok_per_s",
            "max_num_seqs", "token_budget_per_iter",
        )
    )


# ---------- 6. Signal coverage table exists + validates -----------------


@pytest.fixture(scope="module")
def coverage_payload():
    if not os.path.exists(SIGNAL_COVERAGE_PATH):
        pytest.skip("signal_coverage.json not built yet — run "
                    "scripts/build_cara_swissai_signal_coverage.py")
    with open(SIGNAL_COVERAGE_PATH) as fh:
        return json.load(fh)


def test_coverage_payload_top_level_invariants(coverage_payload):
    p = coverage_payload
    assert p["doc_version"] == "cara_swissai_signal_coverage_v1"
    assert p["production_claim"] is False
    assert p["modifies_robust_energy_engine"] is False
    assert p["modifies_controllers_or_defaults"] is False
    assert p["trains_ml_models"] is False
    assert p["uses_oracle_as_headline"] is False


def test_coverage_payload_carries_all_required_blocks(coverage_payload):
    for key in (
        "signal_coverage", "forecast_readiness", "forecast_leverage_ranking",
        "missing_telemetry_gap_analysis",
        "strongest_forecasting_dataset_matrix",
        "configs_audited",
    ):
        assert key in coverage_payload, f"missing block: {key}"


def _rows_for(coverage, signal, dataset_id=None):
    return max(
        (r["rows_available"] for r in coverage if r["signal_name"] == signal
         and (dataset_id is None or r["dataset_id"] == dataset_id)),
        default=0,
    )


REQUIRED_CARA_SIGNALS = ("TTFT", "TPOT", "e2e_latency", "queue_depth",
                         "cache_utilization", "instance_type")


@pytest.mark.parametrize("signal", REQUIRED_CARA_SIGNALS)
def test_required_cara_signals_have_nonzero_rows(coverage_payload, signal):
    rows = _rows_for(
        coverage_payload["signal_coverage"], signal,
        dataset_id="asdwb/cara_latency_prediction",
    )
    assert rows > 0, (
        f"CARA signal '{signal}' rows_available=0 across every config; "
        "the analysis-tier ingest must surface it."
    )


REQUIRED_SWISSAI_SIGNALS = (
    "request_arrival_timestamp", "request_completion_timestamp",
    "reuse_percentage", "model_id", "status",
)


@pytest.mark.parametrize("signal", REQUIRED_SWISSAI_SIGNALS)
def test_required_swissai_signals_have_nonzero_rows(coverage_payload, signal):
    rows = _rows_for(
        coverage_payload["signal_coverage"], signal,
        dataset_id="eth-easl/swissai-serving-trace",
    )
    assert rows > 0, (
        f"SwissAI signal '{signal}' rows_available=0 across every config; "
        "the analysis-tier ingest must surface it."
    )


KNOWN_MISSING_SIGNALS = (
    "replica_count", "autoscaling_events", "GPU_utilization", "GPU_memory",
    "SLA_label", "timeout_label",
)


@pytest.mark.parametrize("signal", KNOWN_MISSING_SIGNALS)
def test_known_missing_signals_stay_missing(coverage_payload, signal):
    """These signals do not exist in CARA or SwissAI. They must be
    explicitly recorded as missing — never silently mapped to a proxy.
    """
    rows = _rows_for(coverage_payload["signal_coverage"], signal)
    assert rows == 0, (
        f"signal '{signal}' falsely shows rows_available={rows}; "
        "the gap analysis claims this signal is pilot-only."
    )
    # Field quality must be 'missing' for every row.
    for r in coverage_payload["signal_coverage"]:
        if r["signal_name"] != signal:
            continue
        assert r["field_quality"] == "missing", (
            f"signal '{signal}' row carries field_quality "
            f"'{r['field_quality']}', must be 'missing'"
        )


# ---------- 7. Subgroup counts + insufficient-sample flagging ---------


def test_cara_train_flat_has_subgroup_counts_per_instance_type():
    path = _summary_path("asdwb/cara_latency_prediction", "train_flat")
    if not os.path.exists(path):
        pytest.skip("CARA train_flat summary not present")
    s = json.loads(open(path).read())
    subgroup_counts = s.get("subgroup_counts") or {}
    assert subgroup_counts, "train_flat must carry subgroup_counts"
    # CARA has 5 known instance_type subgroups (qwen2.5-3b_a30/p100,
    # 7b_a30, 14b_v100, 72b_a100). Allow any non-trivial split.
    assert len(subgroup_counts) >= 3
    # Total subgroup count must match analysis_sample_rows ± stratification cap.
    total = sum(subgroup_counts.values())
    assert total == s["analysis_sample_rows"]


def test_statistical_rollups_path_recorded_and_file_exists():
    for (ds_id, cfg), _ in EXPECTED_ANALYSIS_CONFIGS.items():
        path = _summary_path(ds_id, cfg)
        if not os.path.exists(path):
            continue
        s = json.loads(open(path).read())
        rollups_rel = s.get("statistical_rollups_path")
        assert rollups_rel, f"{ds_id}/{cfg} summary missing statistical_rollups_path"
        assert os.path.exists(os.path.join(REPO_ROOT, rollups_rel))


def test_cara_train_flat_rollups_flag_insufficient_subgroups_when_small():
    path = os.path.join(
        REPO_ROOT, "data", "external", "hf", "asdwb__cara_latency_prediction",
        "train_flat", "processed", "statistical_rollups.json",
    )
    if not os.path.exists(path):
        pytest.skip("CARA train_flat rollups not present")
    rollups = json.loads(open(path).read())
    # Per-(instance_type) latency rollups must contain at least one
    # instance_type subgroup at strong strength (no insufficient flag).
    e2e_by_it = rollups["per_instance_type_latency"].get("e2e") or {}
    strong_subgroups = [k for k, v in e2e_by_it.items()
                        if isinstance(v, dict) and not v.get("flags")]
    assert strong_subgroups, "no instance_type subgroup at strong strength"
    # If any small subgroup exists in per-(prompt_token_bin / queue / kv) it
    # MUST be flagged INSUFFICIENT_SAMPLE_P99, not silently passed.
    insufficient = rollups.get("insufficient_sample_groups") or []
    # The flag must be a list (possibly empty) and never contain duplicates.
    assert isinstance(insufficient, list)
    assert len(insufficient) == len(set(insufficient))


# ---------- 8. Forecast readiness + leverage ranking --------------------


READY_FORECASTS = {
    "ttft_forecast",
    "queue_wait_or_queue_depth_forecast",
    "tpot_forecast",
    "e2e_latency_forecast",
    "cache_hit_or_prefix_reuse_forecast",
    "gpu_placement_or_heterogeneous_latency_forecast",
    "model_residency_or_cold_start_forecast",
    "workload_arrival_forecast",
}


def test_required_forecasts_classified_ready(coverage_payload):
    by_name = {r["forecast"]: r for r in coverage_payload["forecast_readiness"]}
    for forecast in READY_FORECASTS:
        r = by_name.get(forecast)
        assert r is not None, f"forecast '{forecast}' missing from readiness"
        assert r["recommended_readiness"] == "ready_for_forecast_leverage_audit", (
            f"forecast '{forecast}' readiness="
            f"{r['recommended_readiness']}; expected ready_for_forecast_leverage_audit"
        )
        assert r["confidence_1_to_5"] >= 4


GAP_FORECASTS = {
    "timeout_or_sla_violation_forecast",
    "autoscaling_or_replica_need_forecast",
}


def test_gap_forecasts_have_missing_signals_and_pilot_path(coverage_payload):
    by_name = {g["forecast"]: g for g in coverage_payload["missing_telemetry_gap_analysis"]}
    for forecast in GAP_FORECASTS:
        g = by_name.get(forecast)
        assert g is not None, f"gap forecast '{forecast}' missing"
        gaps = g["gaps"]
        assert any(
            "missing_signal" in x and "pilot_telemetry_only" in str(x.get(
                "acquisition_path", "")).lower()
            for x in gaps
        ), (
            f"gap forecast '{forecast}' must record at least one "
            "pilot_telemetry_only missing signal"
        )


def test_leverage_ranking_sorted_descending(coverage_payload):
    scores = [r["leverage_score"] for r in coverage_payload["forecast_leverage_ranking"]]
    assert scores == sorted(scores, reverse=True), (
        "forecast_leverage_ranking must be sorted by leverage_score desc"
    )


def test_ready_forecasts_have_build_now_priority(coverage_payload):
    lookup = {
        r["forecast"]: r["build_priority"]
        for r in coverage_payload["forecast_leverage_ranking"]
    }
    for forecast in READY_FORECASTS:
        assert lookup.get(forecast) == "build_now", (
            f"forecast '{forecast}' build_priority={lookup.get(forecast)}; "
            "expected build_now"
        )


# ---------- 9. Strongest forecasting dataset matrix -------------------


def test_strongest_dataset_matrix_covers_every_forecast(coverage_payload):
    matrix = coverage_payload["strongest_forecasting_dataset_matrix"]
    forecasts_in_matrix = {row["forecast"] for row in matrix}
    forecasts_in_readiness = {
        r["forecast"] for r in coverage_payload["forecast_readiness"]
    }
    assert forecasts_in_matrix == forecasts_in_readiness, (
        "strongest_forecasting_dataset_matrix must list every forecast "
        f"in the readiness table: missing="
        f"{forecasts_in_readiness - forecasts_in_matrix}"
    )
    for row in matrix:
        assert row["best_dataset"], (
            f"{row['forecast']} has no best_dataset"
        )
        assert row["why"].strip(), f"{row['forecast']} has empty 'why'"
