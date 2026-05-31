"""Tests for the HF dataset discovery + scoring + classification pipeline.

All tests run hermetically against ``tests/fixtures/hf_api/`` cached JSON
responses — no network calls.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces.hf_corpus import discovery  # noqa: E402
from aurelius.traces.hf_corpus.schemas import (  # noqa: E402
    CANONICAL_TRACE_TYPES,
    CANONICAL_TRACE_TYPE_TO_TRUST_TIER,
)

FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures", "hf_api")


# ---------- 1. Schema-level invariants -----------------------------------


def test_canonical_trace_types_includes_all_documented():
    expected = {
        "request_shape_trace", "latency_benchmark_trace", "kernel_profile_trace",
        "cluster_scheduler_trace", "cache_residency_trace", "telemetry_trace",
        "mixed_or_unknown_trace",
    }
    assert expected == set(CANONICAL_TRACE_TYPES)


def test_trust_tiers_cover_every_canonical_trace_type():
    for tt in CANONICAL_TRACE_TYPES:
        assert tt in CANONICAL_TRACE_TYPE_TO_TRUST_TIER, (
            f"trace_type {tt} has no trust tier mapping"
        )


def test_target_signals_are_unique():
    assert len(discovery.TARGET_SIGNALS) == len(set(discovery.TARGET_SIGNALS))


# ---------- 2. Metadata parsing ------------------------------------------


def _load_detail(safe_id):
    path = os.path.join(FIXTURES, "datasets", safe_id + ".json")
    with open(path) as fh:
        return json.load(fh)


def test_parse_hf_metadata_agent_perf_bench():
    raw = _load_detail("agent-perf-bench__agentperfbench")
    meta = discovery.parse_hf_metadata(raw)
    assert meta.dataset_id == "agent-perf-bench/AgentPerfBench"
    assert meta.dataset_url == (
        "https://huggingface.co/datasets/agent-perf-bench/AgentPerfBench"
    )
    assert meta.gated is False
    assert meta.license == "apache-2.0"
    assert "trace_replay" in meta.configs
    assert any("ttft" in f.lower() for f in meta.feature_names)
    assert any("tpot" in f.lower() for f in meta.feature_names)


def test_parse_hf_metadata_lmsys_gated_auto_string():
    raw = _load_detail("lmsys__chatbot_arena_conversations")
    meta = discovery.parse_hf_metadata(raw)
    # LMSYS returns gated: "auto" (string), which must normalize to True.
    assert meta.gated is True


def test_parse_hf_metadata_rejects_missing_id():
    with pytest.raises(ValueError):
        discovery.parse_hf_metadata({})


# ---------- 3. Classifier --------------------------------------------------


def test_classify_agent_perf_bench_is_latency_benchmark():
    raw = _load_detail("agent-perf-bench__agentperfbench")
    meta = discovery.parse_hf_metadata(raw)
    cls = discovery.classify_dataset(meta)
    assert cls["trace_type"] == "latency_benchmark_trace"
    assert "latency_benchmark_trace" in cls["evidence"]


def test_classify_lmsys_chatbot_is_request_shape():
    raw = _load_detail("lmsys__chatbot_arena_conversations")
    meta = discovery.parse_hf_metadata(raw)
    cls = discovery.classify_dataset(meta)
    assert cls["trace_type"] == "request_shape_trace"


def test_classify_empty_metadata_is_mixed_unknown():
    meta = discovery.HFDatasetMeta(
        dataset_id="foo/bar", dataset_url="https://hf.co/datasets/foo/bar",
        gated=None, private=None, license=None, description=None,
        tags=(), downloads=None, likes=None, size_categories=(),
        configs=(), splits=(), feature_names=(), siblings=(),
        last_modified=None,
    )
    cls = discovery.classify_dataset(meta)
    assert cls["trace_type"] == "mixed_or_unknown_trace"
    assert cls["evidence"] == {}


# ---------- 4. Scoring -----------------------------------------------------


def test_score_agent_perf_bench_high_value():
    raw = _load_detail("agent-perf-bench__agentperfbench")
    meta = discovery.parse_hf_metadata(raw)
    cls = discovery.classify_dataset(meta)
    s = discovery.score_dataset(meta, cls, matched_keywords=["seed"])
    assert s["overall_priority_score"] >= 4.0, s
    assert s["recommended_action"] == "ingest_now_bounded"
    # Has measured TTFT + TPOT + e2e -> should score 5 for frontier value.
    assert s["frontier_value_score"] >= 4
    assert s["schema_quality_score"] == 5
    # Available signals include TTFT, TPOT, etc.
    assert {"ttft", "tpot"}.issubset(set(s["available_signals"]))


def test_score_lmsys_gated_marks_gated_blocked():
    raw = _load_detail("lmsys__chatbot_arena_conversations")
    meta = discovery.parse_hf_metadata(raw)
    cls = discovery.classify_dataset(meta)
    s = discovery.score_dataset(meta, cls, matched_keywords=["seed"])
    assert s["recommended_action"] == "gated_blocked"


def test_score_empty_features_marks_unknown_schema():
    meta = discovery.HFDatasetMeta(
        dataset_id="foo/empty", dataset_url="https://hf.co/datasets/foo/empty",
        gated=False, private=False, license="apache-2.0", description=None,
        tags=(), downloads=10, likes=0, size_categories=(),
        configs=(), splits=(), feature_names=(), siblings=(),
        last_modified=None,
    )
    cls = discovery.classify_dataset(meta)
    s = discovery.score_dataset(meta, cls, matched_keywords=[])
    assert s["recommended_action"] == "unknown_schema"


def test_score_request_shape_capped_at_3():
    raw = _load_detail("anon8231489123__sharegpt_vicuna_unfiltered")
    meta = discovery.parse_hf_metadata(raw)
    cls = discovery.classify_dataset(meta)
    s = discovery.score_dataset(meta, cls, matched_keywords=["seed"])
    assert cls["trace_type"] == "request_shape_trace"
    # Anti-spam: conversation-only datasets capped at 3 for frontier value.
    assert s["frontier_value_score"] <= 3


# ---------- 5. End-to-end discovery with OfflineHFClient -----------------


def test_discovery_runs_offline_and_writes_sorted_candidates(tmp_path):
    client = discovery.OfflineHFClient(FIXTURES)
    candidates = discovery.discover(
        client,
        query_groups=discovery.DEFAULT_QUERY_GROUPS,
        extra_seed_ids=["agent-perf-bench/AgentPerfBench"],
        now=1_750_000_000.0,
    )
    assert candidates, "discovery returned no candidates from fixtures"

    # Sorted by overall_priority_score descending, then dataset_id ascending.
    for i in range(1, len(candidates)):
        prev = (-candidates[i - 1]["overall_priority_score"],
                candidates[i - 1]["dataset_id"])
        cur = (-candidates[i]["overall_priority_score"],
               candidates[i]["dataset_id"])
        assert prev <= cur

    # AgentPerfBench must be the top candidate.
    assert candidates[0]["dataset_id"] == "agent-perf-bench/AgentPerfBench"

    # Every candidate carries the required scoring fields.
    required = {
        "dataset_id", "dataset_url", "gated_status", "license",
        "estimated_size", "available_splits", "schema_available",
        "matched_keywords", "candidate_trace_type", "available_signals",
        "missing_signals", "trust_level", "ingestion_feasibility_score",
        "frontier_value_score", "schema_quality_score",
        "production_similarity_score", "overall_priority_score",
        "recommended_action", "aurelius_use_case", "not_recommended_uses",
        "discovery_timestamp_s",
    }
    for c in candidates:
        missing = required - set(c.keys())
        assert not missing, f"candidate missing fields {missing}: {c['dataset_id']}"


def test_discovery_classification_evidence_present():
    client = discovery.OfflineHFClient(FIXTURES)
    candidates = discovery.discover(client, now=1_750_000_000.0)
    top = candidates[0]
    assert top["candidate_trace_type"] in CANONICAL_TRACE_TYPES
    assert isinstance(top["classification_evidence"], dict)


def test_discovery_handles_missing_metadata_gracefully(tmp_path):
    # Build a fixtures dir with only a search hit and no detail for a dataset.
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / "ttft_tpot_latency_benchmark.json").write_text(
        json.dumps([{"id": "ghost/nonexistent"}])
    )
    client = discovery.OfflineHFClient(str(tmp_path))
    candidates = discovery.discover(
        client, query_groups={"latency_benchmark": ["TTFT TPOT latency benchmark"]},
        now=1_750_000_000.0,
    )
    # Ghost dataset has no detail file; falls back to the bare search hit;
    # mixed_or_unknown_trace + unknown_schema action.
    assert len(candidates) == 1
    assert candidates[0]["dataset_id"] == "ghost/nonexistent"
    assert candidates[0]["candidate_trace_type"] == "mixed_or_unknown_trace"


# ---------- 6. Token never appears in logs / output ----------------------


def test_offline_client_never_uses_token():
    # OfflineHFClient is what tests use; assert no env vars leak into output.
    client = discovery.OfflineHFClient(FIXTURES)
    res = client.search("AgentPerfBench")
    assert all("token" not in json.dumps(r).lower() for r in res)


# ---------- 7. Use-case routing matches spec rules -----------------------


def test_aurelius_use_case_table_covers_all_types():
    for tt in CANONICAL_TRACE_TYPES:
        uc = discovery.aurelius_use_case(tt)
        assert "use" in uc and "not_recommended" in uc
        assert isinstance(uc["not_recommended"], list)


def test_aurelius_use_case_request_shape_blocks_latency_calibration():
    uc = discovery.aurelius_use_case("request_shape_trace")
    blocks = " ".join(uc["not_recommended"]).lower()
    assert "latency" in blocks
