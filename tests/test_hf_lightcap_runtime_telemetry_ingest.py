"""Tests for the Lightcap/agent-runtime-telemetry-small bounded ingest.

Covers:
- No raw / analysis_sample data committed.
- Per-config schema_profile + schema_mapping + summary + rollups present.
- Schema mapping classifies every accepted column, no unknown columns.
- Summary passes every promotion gate.
- ``operations`` config is promoted to backtest + constraint_aware_eval +
  training_priors (moderate strength, 2,262 rows).
- ``tool_summary`` config is promoted_for_schema_only (32 rows,
  fixture-only).
- ``tool_runtime_trace`` canonical type + ``ToolRuntimeRecord`` validated.
- Trust tier is ``tier_3_cluster_scheduler_traces`` (NOT Tier 1, NOT Tier 2).
- License is ``cc-by-4.0`` and gated=False.
- Signal coverage: routing / failure_timeout / cache-residency-proxy
  present; GPU / queue / replica / model / TTFT / TPOT absent.
- Limitations pin the closed-runtime-timing + no-LLM-serving caveats.
- Per-config normalized samples are committed under the 100-MiB cap.
- Per-config fixtures are committed and tiny.
- Canonical corpus registry knows about both configs.
- Candidates JSON records the focused_audit_2026_06_01c block.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces.hf_corpus import (
    promotion,  # noqa: E402
    schemas,  # noqa: E402
)

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")
CONFIGS = ["operations", "tool_summary", "operation_events", "audit_records"]
PRIMARY_CONFIG = "operations"
# Per-call grain configs (one row per tool call or lifecycle event) — all
# but ``tool_summary`` (the aggregate-only pre-rolled bucket summary).
GRAIN_CONFIGS = ["operations", "operation_events", "audit_records"]


def _processed_dir(config: str) -> Path:
    return HF_DIR / SAFE_DATASET / config / "processed"


def _fixture_path(config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_DATASET}__{config}_sample.jsonl"


def _summary(config: str) -> dict:
    with open(_processed_dir(config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis data committed ─────────────


def test_no_raw_files_tracked_by_git() -> None:
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_DATASET}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed (gitignore broken): "
        f"{analysis_committed}"
    )


# ───────────────────────── 2. Fixture files ────────────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_fixture_files_are_committed_and_tiny(config: str) -> None:
    fixture = _fixture_path(config)
    assert fixture.exists(), f"missing fixture: {fixture}"
    size = fixture.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {fixture} size {size} outside 1-16KiB band"
    )
    # Deterministic JSONL: every line must parse as JSON.
    with open(fixture) as fh:
        lines = fh.readlines()
    assert len(lines) >= 1
    for ln in lines:
        json.loads(ln)


# ───────────────────────── 3. Processed artefacts ──────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_present_and_well_formed(config: str) -> None:
    p = _processed_dir(config) / "schema_profile.json"
    assert p.exists(), f"missing schema_profile: {p}"
    with open(p) as fh:
        profile = json.load(fh)
    assert profile["dataset_id"] == DATASET_ID
    assert profile["config_name"] == config
    assert profile["inspected_row_count"] > 0
    assert isinstance(profile["raw_columns"], list)
    assert isinstance(profile["normalized_columns"], list)
    assert profile["raw_columns"]
    assert profile["normalized_columns"]


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_mapping_classifies_every_accepted_column(config: str) -> None:
    p = _processed_dir(config) / "schema_mapping.json"
    assert p.exists(), f"missing schema_mapping: {p}"
    with open(p) as fh:
        mapping = json.load(fh)
    assert mapping["dataset_id"] == DATASET_ID
    assert mapping["config_name"] == config
    assert mapping["accepted_columns"], (
        f"no accepted columns recorded for {config}"
    )
    assert mapping["rejected_columns"] == [], (
        f"{config} has rejected columns; tighten the mapping table: "
        f"{mapping['rejected_columns']}"
    )
    # Every column record must have non-null normalized_field, quality,
    # and aurelius_signal_category for accepted columns.
    accepted_set = set(mapping["accepted_columns"])
    for col in mapping["columns"]:
        if col["raw_column_name"] not in accepted_set:
            continue
        assert col["normalized_field"], (
            f"{config}: accepted column {col['raw_column_name']} has no "
            f"normalized_field"
        )
        assert col["field_quality"] in schemas.FIELD_QUALITY_VALUES, (
            f"{config}: bad field_quality {col['field_quality']} for "
            f"{col['raw_column_name']}"
        )
        assert col["aurelius_signal_category"], (
            f"{config}: accepted column {col['raw_column_name']} has no "
            f"aurelius_signal_category"
        )


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_present(config: str) -> None:
    s = _summary(config)
    assert s["dataset_id"] == DATASET_ID
    assert s["config_name"] == config
    assert s["canonical_trace_type"] == "tool_runtime_trace"
    assert s["license"] == "cc-by-4.0"
    assert s["gated"] is False
    assert s["source_url"] == f"https://huggingface.co/datasets/{DATASET_ID}"


@pytest.mark.parametrize("config", CONFIGS)
def test_statistical_rollups_present(config: str) -> None:
    p = _processed_dir(config) / "statistical_rollups.json"
    assert p.exists(), f"missing statistical_rollups: {p}"
    with open(p) as fh:
        rollups = json.load(fh)
    assert "subgroup_counts" in rollups
    if config == "operations":
        # Operations rollups must include duration_ms distribution +
        # per-tool failure rates.
        assert "numeric_distributions" in rollups
        assert "duration_ms" in rollups["numeric_distributions"]
        assert "per_tool_failure_rates" in rollups
        assert "overall_failure_rates" in rollups
        overall = rollups["overall_failure_rates"]
        # Sanity: error rate is in (0, 1) and matches subgroup counts.
        assert 0 < overall["error_rate"] < 0.5
        assert overall["count"] >= 1000
    elif config == "operation_events":
        # Per-event rollups must expose per-stage + per-event_type
        # duration_ms breakdowns + the unique-operation count.
        assert "numeric_distributions" in rollups
        nd = rollups["numeric_distributions"]
        assert "duration_ms" in nd
        assert "per_stage" in nd["duration_ms"]
        assert "per_event_type" in nd["duration_ms"]
        assert "per_operation_event_count" in rollups
        op_ct = rollups["per_operation_event_count"]
        # Each operation in this export has 2-8 lifecycle events. The
        # rollup MUST record at least the 2,262 operations from operations.
        assert op_ct["unique_operations"] >= 1000
        assert op_ct["min_events_per_op"] >= 2
        assert op_ct["max_events_per_op"] <= 16
    elif config == "audit_records":
        # Audit-record rollups must expose duration_ms over tool_results
        # rows + per-request audit-pair counts.
        assert "numeric_distributions" in rollups
        nd = rollups["numeric_distributions"]
        assert "duration_ms" in nd
        assert "overall_tool_results" in nd["duration_ms"]
        results_durations = nd["duration_ms"]["overall_tool_results"]
        # Strong-strength sample: count must be ≥ 1000 and p95 should be
        # in a sensible heavy-tailed band.
        assert results_durations["count"] >= 1000
        assert results_durations["p95"] > results_durations["p50"]
        assert "per_request_audit_record_count" in rollups
        rcounts = rollups["per_request_audit_record_count"]
        # Audit records come in request/result pairs → mean ≈ 2.
        assert 1.5 <= rcounts["mean_records_per_request"] <= 2.5
        assert "overall_failure_rates" in rollups
        # Error rate is bounded between 0 and 0.5.
        assert 0 <= rollups["overall_failure_rates"]["error_rate"] < 0.5


# ───────────────────────── 4. Promotion gates ─────────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_passes_every_promotion_gate(config: str) -> None:
    s = _summary(config)
    gates = promotion.gates(s)
    failed = [g for g in gates if not g["passed"]]
    assert failed == [], (
        f"{config}: {len(failed)} promotion gate(s) failed: {failed}"
    )


def test_operations_config_is_promoted_for_backtest_and_more() -> None:
    s = _summary("operations")
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_backtest"
    tags = set(decision["promotion_tags"])
    assert {
        "promoted_for_backtest",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    }.issubset(tags), (
        f"operations missing expected promotion tags; got {sorted(tags)}"
    )


def test_tool_summary_config_is_promoted_for_schema_only() -> None:
    s = _summary("tool_summary")
    decision = promotion.evaluate_promotion(s)
    # 32 aggregated rows → fixture_only strength → schema-only promotion.
    assert decision["state"] == "promoted_for_schema_only", (
        f"unexpected tool_summary state: {decision['state']}"
    )


def test_operation_events_config_is_promoted_for_backtest_and_more() -> None:
    s = _summary("operation_events")
    decision = promotion.evaluate_promotion(s)
    # 9,903 lifecycle events → moderate strength → backtest + constraint +
    # training_priors.
    assert decision["state"] == "promoted_for_backtest"
    tags = set(decision["promotion_tags"])
    assert {
        "promoted_for_backtest",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    }.issubset(tags), (
        f"operation_events missing expected promotion tags; got {sorted(tags)}"
    )
    assert s["statistical_sample_strength"] == "moderate", (
        f"operation_events sample strength != moderate: "
        f"{s['statistical_sample_strength']}"
    )


def test_audit_records_config_is_promoted_for_backtest_and_more() -> None:
    s = _summary("audit_records")
    decision = promotion.evaluate_promotion(s)
    # 14,053 audit records → strong strength → backtest + constraint +
    # training_priors. (tool_runtime_trace does NOT allow
    # promoted_for_dynamic_calibration — no queue / replica / GPU-util.)
    assert decision["state"] == "promoted_for_backtest"
    tags = set(decision["promotion_tags"])
    assert {
        "promoted_for_backtest",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    }.issubset(tags), (
        f"audit_records missing expected promotion tags; got {sorted(tags)}"
    )
    assert "promoted_for_dynamic_calibration" not in tags, (
        "tool_runtime_trace must NOT promote to dynamic_calibration even at "
        "strong sample strength (no queue / replica / GPU-util signal)."
    )
    assert s["statistical_sample_strength"] == "strong", (
        f"audit_records sample strength != strong: "
        f"{s['statistical_sample_strength']}"
    )


# ───────────────────────── 5. Canonical type + trust tier ─────────────────


def test_tool_runtime_trace_is_a_canonical_type() -> None:
    assert "tool_runtime_trace" in schemas.CANONICAL_TRACE_TYPES


def test_tool_runtime_record_class_is_registered() -> None:
    cls = schemas.TRACE_TYPE_TO_RECORD_CLASS["tool_runtime_trace"]
    assert cls is schemas.ToolRuntimeRecord
    # And its payload fields are non-empty and registered.
    fields = schemas.TRACE_TYPE_TO_PAYLOAD_FIELDS["tool_runtime_trace"]
    assert "operation_id" in fields
    assert "tool_name" in fields
    assert "duration_ms" in fields
    assert "status" in fields
    assert "error_type" in fields


def test_tool_runtime_payload_fields_cover_operation_events_dimensions() -> None:
    """operation_events config introduces 8 new payload fields
    (event_id / event_type / payload_bytes / payload_sha256 /
    payload_key_count / payload_keys / payload_status / payload_stage)
    so the canonical schema must enumerate them all."""
    fields = schemas.TRACE_TYPE_TO_PAYLOAD_FIELDS["tool_runtime_trace"]
    for f in ("event_id", "event_type", "payload_bytes", "payload_sha256",
              "payload_key_count", "payload_keys", "payload_status",
              "payload_stage"):
        assert f in fields, (
            f"TOOL_RUNTIME_PAYLOAD_FIELDS missing operation_events "
            f"dimension '{f}'"
        )


def test_tool_runtime_payload_fields_cover_audit_records_dimensions() -> None:
    """audit_records config introduces 8 new payload fields
    (record_id / category / record_name / record_file /
    record_path_scope / kind / response_key_count / response_keys)
    so the canonical schema must enumerate them all."""
    fields = schemas.TRACE_TYPE_TO_PAYLOAD_FIELDS["tool_runtime_trace"]
    for f in ("record_id", "category", "record_name", "record_file",
              "record_path_scope", "kind", "response_key_count",
              "response_keys"):
        assert f in fields, (
            f"TOOL_RUNTIME_PAYLOAD_FIELDS missing audit_records "
            f"dimension '{f}'"
        )


def test_tool_runtime_record_accepts_event_dimensions() -> None:
    """The dataclass must accept event_id / event_type / payload_bytes etc.
    so per-event normalised rows can be constructed without raising."""
    r = schemas.ToolRuntimeRecord(
        source_dataset_id=DATASET_ID,
        trace_type="tool_runtime_trace",
        provenance="p",
        field_quality={
            "operation_id": "real",
            "event_id": "real",
            "event_type": "real",
            "stage": "real",
            "status": "real",
            "duration_ms": "derived",
            "payload_bytes": "real",
            "payload_sha256": "real",
        },
        operation_id="op-1",
        event_id=1,
        event_type="started",
        stage="started",
        status="running",
        duration_ms=0.0,
        payload_bytes=109,
        payload_sha256="da92" + "0" * 60,
    )
    assert r.event_type == "started"
    assert r.payload_bytes == 109


def test_tool_runtime_record_accepts_audit_record_dimensions() -> None:
    """The dataclass must accept record_id / category / kind etc."""
    r = schemas.ToolRuntimeRecord(
        source_dataset_id=DATASET_ID,
        trace_type="tool_runtime_trace",
        provenance="p",
        field_quality={
            "record_id": "real",
            "category": "real",
            "kind": "real",
            "request_id": "real",
            "tool_name": "real",
            "duration_ms": "real",
            "response_key_count": "real",
        },
        record_id=42,
        category="tool_results",
        kind="mcp_tool_result",
        request_id="abc",
        tool_name="surface_affinity",
        duration_ms=4.7,
        response_key_count=3,
    )
    assert r.category == "tool_results"
    assert r.response_key_count == 3


def test_tool_runtime_record_validates_field_quality() -> None:
    # Smoke-test the dataclass validator: it must reject unknown fields
    # in field_quality and bad trace_type.
    with pytest.raises(schemas.HFCorpusSchemaError):
        schemas.ToolRuntimeRecord(
            source_dataset_id=DATASET_ID,
            trace_type="tool_runtime_trace",
            provenance="p",
            field_quality={"NOT_A_FIELD": "real"},
            operation_id="x",
        )
    with pytest.raises(schemas.HFCorpusSchemaError):
        schemas.ToolRuntimeRecord(
            source_dataset_id=DATASET_ID,
            trace_type="request_shape_trace",  # wrong type
            provenance="p",
            field_quality={"operation_id": "real"},
            operation_id="x",
        )
    # And a clean record builds.
    r = schemas.ToolRuntimeRecord(
        source_dataset_id=DATASET_ID,
        trace_type="tool_runtime_trace",
        provenance="p",
        field_quality={
            "operation_id": "real",
            "tool_name": "real",
            "duration_ms": "real",
            "status": "real",
        },
        operation_id="op-1",
        tool_name="surface_affinity",
        duration_ms=12.3,
        status="ok",
    )
    assert r.duration_ms == 12.3


def test_trust_tier_for_tool_runtime_trace_is_tier3() -> None:
    assert (schemas.CANONICAL_TRACE_TYPE_TO_TRUST_TIER["tool_runtime_trace"]
            == "tier_3_cluster_scheduler_traces")


@pytest.mark.parametrize("config", CONFIGS)
def test_registry_trust_tier_is_tier3_not_tier1(config: str) -> None:
    with open(DISC_DIR / "canonical_corpus_registry.json") as fh:
        reg = json.load(fh)
    entries = [e for e in reg["entries"]
               if e["dataset_id"] == DATASET_ID
               and e["config_name"] == config]
    assert len(entries) == 1, (
        f"expected exactly one canonical entry for {DATASET_ID}@{config}"
    )
    e = entries[0]
    assert e["trust_tier"] == "tier_3_cluster_scheduler_traces"
    # Pilot telemetry remains the only Tier-1 source. This dataset must
    # NEVER claim Tier 1 or Tier 2.
    assert e["trust_tier"] != "tier_1_real_pilot_telemetry"
    assert e["trust_tier"] != "tier_2_public_telemetry_traces"


# ───────────────────────── 6. Signal coverage ─────────────────────────────


def test_operations_signals_are_explicit_and_disjoint() -> None:
    s = _summary("operations")
    avail = set(s["available_signals"])
    miss = set(s["missing_signals"])
    assert avail.isdisjoint(miss), (
        f"available and missing signals overlap: {avail & miss}"
    )
    # The operations config MUST advertise these tool-runtime signals.
    expected_present = {
        "arrivals",
        "request_timestamps",
        "latency",
        "duration_measured",
        "tool_routing",
        "tool_failure_label",
        "tool_cancellation_label",
        "args_fingerprint_for_cache_reuse",
        "workload_shape",
    }
    missing_from_avail = expected_present - avail
    assert not missing_from_avail, (
        f"operations missing expected signals: {missing_from_avail}"
    )


def test_operations_does_not_claim_gpu_serving_signals() -> None:
    """No model_id / no input/output_tokens / no GPU type / no queue /
    no replica / no TTFT / no TPOT — Lightcap is tool-runtime telemetry,
    not LLM serving telemetry. These MUST live in missing_signals."""
    s = _summary("operations")
    miss = set(s["missing_signals"])
    forbidden_in_avail = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
        "model_load_event", "model_unload_event",
    }
    avail = set(s["available_signals"])
    leak = forbidden_in_avail & avail
    assert leak == set(), (
        f"operations falsely advertises serving signals it does NOT measure: "
        f"{leak}"
    )
    # And the absences are explicit.
    expected_missing = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
    }
    not_recorded = expected_missing - miss
    assert not_recorded == set(), (
        f"operations did NOT record absences for: {not_recorded}"
    )


@pytest.mark.parametrize("config", GRAIN_CONFIGS)
def test_grain_configs_do_not_claim_gpu_serving_signals(config: str) -> None:
    """Same anti-overclaim guard as operations, applied uniformly to every
    per-call grain config (operations / operation_events / audit_records).
    None of them have GPU / queue / replica / model signals; promoting
    those would silently inflate the corpus' telemetry surface area."""
    s = _summary(config)
    miss = set(s["missing_signals"])
    avail = set(s["available_signals"])
    forbidden_in_avail = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
        "model_load_event", "model_unload_event",
    }
    leak = forbidden_in_avail & avail
    assert leak == set(), (
        f"{config} falsely advertises serving signals it does NOT measure: "
        f"{leak}"
    )
    expected_missing = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
    }
    not_recorded = expected_missing - miss
    assert not_recorded == set(), (
        f"{config} did NOT record absences for: {not_recorded}"
    )


def test_operation_events_carries_per_event_dispatch_latency_signal() -> None:
    """operation_events MUST advertise the per-event latency signal
    (derived from real event_time_utc timestamps) so the constraint-aware
    harness can pick it up as a dispatch / execution / completion-stage
    prior."""
    s = _summary("operation_events")
    avail = set(s["available_signals"])
    # Per-event ms-since-started is the dispatch / execution / completion
    # latency exposure. Real timestamps + arrivals + workload_shape must
    # also be present. Cache-fingerprint via payload_sha256.
    expected_present = {
        "arrivals", "request_timestamps", "latency", "duration_measured",
        "tool_failure_label", "args_fingerprint_for_cache_reuse",
        "workload_shape",
    }
    missing_from_avail = expected_present - avail
    assert not missing_from_avail, (
        f"operation_events missing expected signals: {missing_from_avail}"
    )
    # And duration_ms must be labeled as a DERIVED field (it's computed
    # from real event_time_utc timestamps, not a raw measurement).
    derived = set(s["derived_fields"])
    assert "duration_ms" in derived, (
        "operation_events duration_ms must be labeled DERIVED — it is "
        "ms-since-started, not a raw measurement."
    )


def test_audit_records_carries_real_mcp_shell_layer_duration() -> None:
    """audit_records MUST advertise REAL duration_ms (MCP-shell-layer
    timing on tool_results rows). This is a Tier-3 real measurement, not
    a derived signal."""
    s = _summary("audit_records")
    avail = set(s["available_signals"])
    expected_present = {
        "arrivals", "request_timestamps", "latency", "duration_measured",
        "tool_routing", "tool_failure_label",
        "args_fingerprint_for_cache_reuse", "workload_shape",
    }
    missing_from_avail = expected_present - avail
    assert not missing_from_avail, (
        f"audit_records missing expected signals: {missing_from_avail}"
    )
    # duration_ms is REAL on audit_records (NOT in derived_fields).
    derived = set(s["derived_fields"])
    assert "duration_ms" not in derived, (
        "audit_records duration_ms is REAL (only populated on tool_results "
        "rows); it must NOT be labeled DERIVED."
    )


# ───────────────────────── 7. Limitations pinning ─────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_limitations_pin_no_llm_serving_signal(config: str) -> None:
    s = _summary(config)
    lims = s["limitations"]
    assert isinstance(lims, list) and lims, (
        f"{config}: limitations must be non-empty"
    )
    joined = " ".join(lims)
    # Must call out the closed-runtime / not-LLM-serving caveat somewhere.
    assert (
        "NOT GPU" in joined or "NOT LLM" in joined
        or "tool-runtime" in joined.lower() or "tool runtime" in joined.lower()
    ), (
        f"{config}: limitations do not pin the not-LLM-serving caveat: "
        f"{lims}"
    )


# ───────────────────────── 8. Bounded normalized sample ───────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_normalized_sample_is_committed_and_bounded(config: str) -> None:
    p = _processed_dir(config) / "normalized_sample.jsonl"
    assert p.exists(), f"missing normalized_sample: {p}"
    size = p.stat().st_size
    assert size > 0
    # 100 MiB cap from the policy.
    assert size <= 100 * 1024 * 1024, (
        f"{config}: normalized_sample.jsonl {size} bytes exceeds 100 MiB cap"
    )
    # The sha256 in the summary must match the actual file (or at least
    # be a 64-hex sha256 string).
    s = _summary(config)
    expected_sha = s.get("normalized_sample_sha256")
    assert isinstance(expected_sha, str) and len(expected_sha) == 64
    with open(p, "rb") as fh:
        actual_sha = hashlib.sha256(fh.read()).hexdigest()
    assert actual_sha == expected_sha, (
        f"{config}: normalized_sample.jsonl sha256 mismatch: "
        f"recorded={expected_sha} actual={actual_sha}"
    )


# ───────────────────────── 9. Registry consistency ────────────────────────


def test_canonical_registry_includes_every_config() -> None:
    with open(DISC_DIR / "canonical_corpus_registry.json") as fh:
        reg = json.load(fh)
    ids = {(e["dataset_id"], e.get("config_name")) for e in reg["entries"]}
    for cfg in CONFIGS:
        assert (DATASET_ID, cfg) in ids, (
            f"canonical registry missing {DATASET_ID}@{cfg}"
        )


def test_candidates_json_records_focused_audit_2026_06_01c() -> None:
    with open(DISC_DIR / "hf_dataset_candidates.json") as fh:
        cands = json.load(fh)
    assert "focused_audit_2026_06_01c" in cands
    block = cands["focused_audit_2026_06_01c"]
    assert DATASET_ID in block["datasets"]
    assert block["new_canonical_type"] == "tool_runtime_trace"
    assert block["trust_tier"] == "tier_3_cluster_scheduler_traces"
    assert block["production_claim"] is False
    assert block["modifies_robust_energy_engine"] is False
    # And the candidate row itself reflects the ingest decision. The
    # follow-on 2026-06-01d run bumps audit_round to the follow-up name
    # while preserving the inaugural audit_note_2026_06_01c block.
    candidates = cands["candidates"]
    light = [c for c in candidates if c.get("dataset_id") == DATASET_ID]
    assert len(light) == 1
    assert light[0]["recommended_action"] == "ingest_now_bounded"
    assert light[0]["audit_round"] in {
        "focused_audit_2026_06_01c",
        "focused_audit_2026_06_01d",
    }
    assert "audit_note_2026_06_01c" in light[0], (
        "inaugural 2026-06-01c audit note must be preserved across "
        "follow-up runs"
    )


def test_candidates_json_records_focused_audit_2026_06_01d_followup() -> None:
    """Follow-up block: Lightcap operation_events + audit_records ingested
    as the same tool_runtime_trace canonical type. Must be present alongside
    the inaugural 2026-06-01c block (not replacing it)."""
    with open(DISC_DIR / "hf_dataset_candidates.json") as fh:
        cands = json.load(fh)
    assert "focused_audit_2026_06_01d" in cands, (
        "follow-up 2026-06-01d audit block missing — "
        "scripts/register_hf_lightcap_runtime_telemetry.py did not run"
    )
    block = cands["focused_audit_2026_06_01d"]
    assert DATASET_ID in block["datasets"]
    assert block["canonical_type"] == "tool_runtime_trace"
    assert block["trust_tier"] == "tier_3_cluster_scheduler_traces"
    assert block["license"] == "cc-by-4.0"
    assert block["production_claim"] is False
    assert block["modifies_robust_energy_engine"] is False
    assert block["modifies_controllers_or_defaults"] is False
    assert block["headline_promotion_state"] == "promoted_for_backtest"
    # Two configs ingested.
    configs_ingested = " ".join(block["configs_ingested"])
    assert "operation_events" in configs_ingested
    assert "audit_records" in configs_ingested
    # And the canonical join keys are recorded so future cross-config
    # consumers don't have to re-derive them.
    join_keys = block["join_keys"]
    assert join_keys["operation_events__operations"] == "operation_id"
    assert join_keys["audit_records__operations"] == "request_id"
    # And the candidate row's per-run audit note records the follow-up.
    candidates = cands["candidates"]
    light = [c for c in candidates if c.get("dataset_id") == DATASET_ID][0]
    assert "audit_note_2026_06_01d" in light, (
        "follow-up audit_note_2026_06_01d missing from candidate row"
    )


# ───────────────────────── 10. Promotion rules wiring ─────────────────────


def test_promotion_rules_include_tool_runtime_trace() -> None:
    allowed = promotion.TRACE_TYPE_TO_ALLOWED_PROMOTIONS["tool_runtime_trace"]
    assert "promoted_for_backtest" in allowed
    assert "promoted_for_constraint_aware_evaluation" in allowed
    assert "promoted_for_training_priors" in allowed
    # NOT dynamic_calibration — no queue / replica / GPU-util signal.
    assert "promoted_for_dynamic_calibration" not in allowed
