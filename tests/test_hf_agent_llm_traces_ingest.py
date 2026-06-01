"""Tests for the Exgentic/agent-llm-traces bounded ingest.

Covers:
- no raw / large analysis_sample data committed,
- schema_profile + schema_mapping + summary + statistical_rollups present,
- schema mapping classifies every accepted column,
- summary passes every promotion gate,
- summary is promoted to at least training_priors,
- signal-coverage records both available + missing signals,
- expected request-shape signals are present and do NOT include
  GPU-serving signals (no ttft/tpot/queue/replica),
- canonical corpus registry knows about the new entry,
- candidates JSON records the follow-on audit,
- per-config normalized sample is committed (cdla-permissive-2.0) and
  bounded under the 100 MiB policy cap,
- per-config fixture is committed and tiny.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.
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

DATASET_ID = "Exgentic/agent-llm-traces"
SAFE_DATASET = DATASET_ID.replace("/", "__")
CONFIGS = ["swebench_claude_code_shard12"]


def _processed_dir(config: str) -> Path:
    return HF_DIR / SAFE_DATASET / config / "processed"


def _fixture_path(config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_DATASET}__{config}_sample.jsonl"


def _summary(config: str) -> dict:
    with open(_processed_dir(config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw data committed ─────────────────────────


def test_no_raw_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files",
         f"data/external/hf/{SAFE_DATASET}"], cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed (gitignore broken): {analysis_committed}"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_fixture_files_are_committed_and_tiny(config: str) -> None:
    fixture = _fixture_path(config)
    assert fixture.exists(), f"missing fixture: {fixture}"
    size = fixture.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {fixture} size {size}B outside (0, 16 KiB]"
    )


# ───────────────────────── 2. Per-dataset artefacts ─────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_processed_artifacts_present(config: str) -> None:
    pd = _processed_dir(config)
    for name in ("schema_profile.json", "schema_mapping.json",
                 "summary.json", "statistical_rollups.json"):
        path = pd / name
        assert path.exists(), f"{path} missing for {config}"
        with open(path) as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict) and payload, f"{path} is empty"


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_mapping_classifies_every_accepted_column(config: str) -> None:
    with open(_processed_dir(config) / "schema_mapping.json") as fh:
        mapping = json.load(fh)
    assert mapping.get("columns"), "no columns in mapping"
    accepted_records = [c for c in mapping["columns"]
                        if c["raw_column_name"] in mapping["accepted_columns"]]
    assert accepted_records, f"no accepted columns for {config}"
    for c in accepted_records:
        assert c["normalized_field"], f"missing normalized_field for {c}"
        assert c["field_quality"] in {
            "real", "derived", "proxy", "synthetic", "missing", "unknown",
        }, f"invalid field_quality: {c}"
    # The nested-keys table must enumerate every observed gen_ai.* attribute.
    nested = mapping.get("nested_columns") or []
    nested_paths = {c["raw_column_name"] for c in nested}
    expected_nested = {
        "spans[].attributes.gen_ai.usage.input_tokens",
        "spans[].attributes.gen_ai.usage.output_tokens",
        "spans[].attributes.gen_ai.request.model",
        "spans[].attributes.gen_ai.response.model",
        "spans[].attributes.gen_ai.operation.name",
    }
    assert expected_nested <= nested_paths, (
        f"missing nested keys: {expected_nested - nested_paths}"
    )


# ───────────────────────── 3. Promotion gates ─────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_passes_all_promotion_gates(config: str) -> None:
    s = _summary(config)
    gates = promotion.gates(s)
    failed = [g for g in gates if not g["passed"]]
    assert not failed, (
        f"{config}: gates failed: "
        f"{[(g['gate'], g['detail']) for g in failed]}"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_promoted_to_training_priors(config: str) -> None:
    s = _summary(config)
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] in (
        "promoted_for_training_priors",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_backtest",
    ), (
        f"{config}: unexpected state {decision['state']}; "
        f"reasons={decision.get('reasons')}"
    )
    assert decision["promotion_tags"], f"{config}: no promotion_tags"


# ───────────────────────── 4. Signal-coverage ─────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_signal_coverage_recorded(config: str) -> None:
    s = _summary(config)
    avail = s["available_signals"]
    missing = s["missing_signals"]
    assert isinstance(avail, list) and avail
    assert isinstance(missing, list)
    assert not (set(avail) & set(missing)), (
        f"available/missing overlap: {set(avail) & set(missing)}"
    )


def test_expected_request_shape_signals_present() -> None:
    s = _summary("swebench_claude_code_shard12")
    for sig in ("arrivals", "request_timestamps", "workload_shape",
                "routing_proxy", "customer_traffic_mix"):
        assert sig in s["available_signals"], (
            f"missing expected signal {sig}: {s['available_signals']}"
        )


def test_no_gpu_serving_signals_claimed() -> None:
    """The duration_ms field is closed-API end-to-end; the ingester must
    NOT claim ttft / tpot / queue_state / gpu_utilization / replica_count
    signals just because timing exists."""
    s = _summary("swebench_claude_code_shard12")
    forbidden = ("ttft", "tpot", "queue_state", "gpu_utilization",
                 "replica_count")
    for sig in forbidden:
        assert sig in s["missing_signals"], (
            f"forbidden signal {sig} claimed as available: "
            f"{s['available_signals']}"
        )


def test_limitations_pin_closed_api_caveat() -> None:
    """The limitations list must explicitly call out that duration_ms is
    closed-API end-to-end (not GPU TTFT/TPOT) — this is what stops
    downstream consumers from treating it as a serving-latency prior."""
    s = _summary("swebench_claude_code_shard12")
    lims = " ".join(s.get("limitations") or []).lower()
    assert "closed-api" in lims or "closed api" in lims, (
        f"limitations missing closed-API caveat: {s.get('limitations')}"
    )
    assert "ttft" in lims or "tpot" in lims or "gpu" in lims, (
        f"limitations missing GPU-not-measured caveat: {s.get('limitations')}"
    )


# ───────────────────────── 5. Field-quality honesty ─────────────────────────


def test_input_messages_chars_is_derived_not_real() -> None:
    """The raw gen_ai.input.messages payload string is dropped; only the
    char-count proxy remains. The mapping must record that as derived,
    NOT as a real token count."""
    with open(_processed_dir("swebench_claude_code_shard12")
              / "schema_mapping.json") as fh:
        mapping = json.load(fh)
    by_name = {c["raw_column_name"]: c for c in mapping.get("nested_columns") or []}
    im = by_name.get("spans[].attributes.gen_ai.input.messages")
    assert im is not None, "input.messages not enumerated in nested mapping"
    assert im["field_quality"] == "derived", (
        f"input.messages field_quality must be 'derived' (chars-only proxy), "
        f"got {im['field_quality']}"
    )


# ───────────────────────── 6. Normalized sample policy ─────────────────────────


def test_normalized_sample_committed_and_bounded() -> None:
    s = _summary("swebench_claude_code_shard12")
    rel = s.get("committed_normalized_sample_path")
    assert rel, "summary missing committed_normalized_sample_path"
    path = REPO_ROOT / rel
    assert path.exists(), f"normalized sample missing: {path}"
    size = path.stat().st_size
    # Policy: ≤100 MiB per committed normalized sample.
    assert 0 < size <= 100 * 1024 * 1024, (
        f"normalized sample {size}B exceeds 100 MiB cap"
    )
    # Sha must match.
    import hashlib
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    assert sha == s["committed_normalized_sample_sha256"], (
        "committed_normalized_sample_sha256 mismatch — file was modified"
    )


def test_normalized_sample_drops_raw_message_bodies() -> None:
    """Verify the committed sample does NOT contain the huge raw
    gen_ai.input.messages / gen_ai.output.messages strings — only their
    character counts. Median raw size is 50K chars; if the body was
    accidentally kept, sample size would explode."""
    s = _summary("swebench_claude_code_shard12")
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert rows
    # Forbidden raw payload keys must not appear.
    forbidden = ("gen_ai.input.messages", "gen_ai.output.messages",
                 "gen_ai.tool.definitions", "input_messages", "output_messages")
    for r in rows[:50]:
        for k in forbidden:
            assert k not in r, (
                f"raw payload key {k} leaked into committed sample row: {r}"
            )
    # Required proxy keys must be present and bounded.
    required = ("input_messages_chars", "output_messages_chars",
                "tool_definitions_chars", "duration_ms", "input_tokens",
                "output_tokens", "request_model", "status_code")
    for r in rows[:50]:
        for k in required:
            assert k in r, f"required key {k} missing from row: {r.keys()}"


# ───────────────────────── 7. Cross-dataset summary + registry ─────────────────────────


def test_cross_dataset_ingest_summary_consistent() -> None:
    path = DISC_DIR / "agent_llm_traces_ingest_summary.json"
    assert path.exists(), f"missing {path}"
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["modifies_robust_energy_engine"] is False
    assert payload["modifies_controllers_or_defaults"] is False
    assert payload["production_claim"] is False
    ingested = {(e["dataset_id"], e["config_name"]) for e in payload["ingested"]}
    assert (DATASET_ID, CONFIGS[0]) in ingested, (
        f"missing entry in ingest summary: {ingested}"
    )


def test_canonical_registry_includes_new_entry() -> None:
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    assert reg_path.exists()
    with open(reg_path) as fh:
        reg = json.load(fh)
    keys = {(e["dataset_id"], e.get("config_name")) for e in reg["entries"]}
    assert (DATASET_ID, CONFIGS[0]) in keys, (
        f"canonical registry missing: {(DATASET_ID, CONFIGS[0])}"
    )


def test_candidates_records_followup_audit() -> None:
    cand_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(cand_path) as fh:
        doc = json.load(fh)
    ids = {c["dataset_id"] for c in doc.get("candidates", [])}
    assert DATASET_ID in ids, f"candidates missing {DATASET_ID}"
    audit = doc.get("focused_audit_2026_06_01b")
    assert audit and DATASET_ID in (audit.get("outcomes") or {}), (
        "candidates JSON missing focused_audit_2026_06_01b for Exgentic"
    )


# ───────────────────────── 8. Trust-tier honesty ─────────────────────────


def test_dataset_not_marked_tier_1() -> None:
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    with open(reg_path) as fh:
        reg = json.load(fh)
    for e in reg["entries"]:
        if (e["dataset_id"], e.get("config_name")) == (DATASET_ID, CONFIGS[0]):
            assert e["trust_tier"] != "tier_1_real_pilot_telemetry", (
                "Exgentic agent-llm-traces is HF-public, not pilot Tier 1"
            )
            assert e["trust_tier"] == "tier_5_request_shape_traces", (
                f"expected tier_5_request_shape_traces, got {e['trust_tier']}"
            )


def test_dataset_license_recorded() -> None:
    s = _summary("swebench_claude_code_shard12")
    assert s.get("license") == "cdla-permissive-2.0", (
        f"expected cdla-permissive-2.0, got {s.get('license')}"
    )
    assert s.get("gated") is False
    assert s.get("license_redistribution_status") == "permissive_cdla_2"


# ───────────────────────── 9. Statistical rollups ─────────────────────────


def test_statistical_rollups_have_subgroup_and_duration_distribs() -> None:
    with open(_processed_dir("swebench_claude_code_shard12")
              / "statistical_rollups.json") as fh:
        rollups = json.load(fh)
    sg = rollups.get("subgroup_counts") or {}
    nd = rollups.get("numeric_distributions") or {}
    assert "request_model" in sg and sg["request_model"], "no request_model subgroup"
    assert "duration_ms" in nd, "no duration_ms distribution"
    assert "input_tokens" in nd, "no input_tokens distribution"
    assert nd["duration_ms"]["count"] > 0
    # p99 should be larger than p50 (sanity).
    assert nd["duration_ms"]["p99"] >= nd["duration_ms"]["p50"]


def test_sample_strength_reflects_row_count() -> None:
    s = _summary("swebench_claude_code_shard12")
    # 2,294 rows → moderate per the canonical thresholds (≥1000, <10000).
    rows = s["analysis_sample_rows"]
    strength = s["statistical_sample_strength"]
    if rows >= 10000:
        assert strength == "strong"
    elif rows >= 1000:
        assert strength == "moderate"
    elif rows >= 100:
        assert strength == "weak"
    else:
        assert strength == "fixture_only"
