"""Tests for the ShareGPT eval-conversation-shape ingester.

Schema-first invariants the tests pin:

1.  Fixture parses; record + turn keys are EXACTLY ``{id, conversations}``
    and ``{from, value}``.
2.  Unknown record / turn keys are REJECTED with ``EvalWorkloadSchemaError``.
3.  Truncated JSON parses only complete records; never raises on the trailing
    truncation.
4.  Normalization populates the proxy fields with the documented chars/4
    formula AND labels ``token_count_source = "char_div_4_proxy"``.
5.  Missing-field honesty: ``timestamp_s``, ``model_id``, ``language``,
    ``prompt_tokens_real``, ``response_tokens_real``, ``deadline_s`` are
    ALWAYS ``None`` for ShareGPT.
6.  Failure flag is True iff ``response_chars == 0``.
7.  Summary records the (no-timestamps, no-real-tokens, no-model-id,
    no-language) honesty flags.
8.  The committed processed summary (when present) has ``provenance``
    matching the constant and includes every relevant key.
9.  The processed summary file size is within the user-spec 100 MB cap.
10. The committed bounded-download manifest (when present) records the
    requested + downloaded byte counts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces import sharegpt_aiperf  # noqa: E402
from aurelius.traces.eval_schema import (  # noqa: E402
    CHARS_PER_TOKEN_PROXY,
    EvalWorkloadSchemaError,
    chars_to_token_estimate,
    role_sequence_signature,
)

FIXTURE = (REPO_ROOT / "tests" / "fixtures" / "sharegpt_aiperf_sample"
           / "sg_head_fixture.json")
PROCESSED = (REPO_ROOT / "data" / "external" / "sharegpt_aiperf"
             / "processed" / "sharegpt_aiperf_ingest_summary.json")
MANIFEST = (REPO_ROOT / "data" / "external" / "sharegpt_aiperf" / "raw"
            / "bounded_download_manifest.json")
USER_SPEC_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


# ---- 1. Fixture parses with exact schema ----

def test_fixture_parses():
    recs = sharegpt_aiperf.load_json_path(str(FIXTURE))
    assert len(recs) == 3
    assert {r.request_id for r in recs} == {"fixture-A", "fixture-B",
                                            "fixture-C"}
    for r in recs:
        assert r.turn_count >= 1
        assert r.provenance == sharegpt_aiperf.PROVENANCE


def test_record_keys_strictly_id_and_conversations():
    assert sharegpt_aiperf.RECORD_KEYS == frozenset({"id", "conversations"})


def test_turn_keys_strictly_from_and_value():
    assert sharegpt_aiperf.TURN_KEYS == frozenset({"from", "value"})


# ---- 2. Unknown keys rejected ----

def test_unknown_record_key_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"id": "x", "conversations": [{"from": "human", "value": "hi"}],
         "extra_field": "nope"}
    ]))
    with pytest.raises(EvalWorkloadSchemaError) as ei:
        sharegpt_aiperf.load_json_path(str(bad))
    assert "unknown keys" in str(ei.value)


def test_unknown_turn_key_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"id": "x", "conversations": [
            {"from": "human", "value": "hi", "metadata": "nope"}
        ]}
    ]))
    with pytest.raises(EvalWorkloadSchemaError) as ei:
        sharegpt_aiperf.load_json_path(str(bad))
    assert "turn has unknown keys" in str(ei.value)


def test_missing_required_record_key_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(EvalWorkloadSchemaError):
        sharegpt_aiperf.load_json_path(str(bad))


def test_empty_conversations_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x", "conversations": []}]))
    with pytest.raises(EvalWorkloadSchemaError):
        sharegpt_aiperf.load_json_path(str(bad))


# ---- 3. Truncated JSON: complete records only, no exception ----

def test_truncated_partial_array_parses_completes_only(tmp_path):
    bad = tmp_path / "trunc.json"
    full = json.dumps([
        {"id": "A", "conversations": [{"from": "human", "value": "hi"}]},
        {"id": "B", "conversations": [{"from": "human", "value": "yo"}]},
    ])
    # Slice off the last char to truncate the closing `]`.
    bad.write_text(full[:-1])
    recs = sharegpt_aiperf.load_json_path(str(bad))
    assert len(recs) == 2


def test_truncated_inside_record_yields_only_complete(tmp_path):
    bad = tmp_path / "trunc2.json"
    full = json.dumps([
        {"id": "A", "conversations": [{"from": "human", "value": "hi"}]},
        {"id": "B", "conversations": [{"from": "human", "value": "yo"}]},
    ])
    # Truncate inside the second record's value.
    cut = full.index("yo") - 1
    bad.write_text(full[:cut])
    recs = sharegpt_aiperf.load_json_path(str(bad))
    # The first record is complete, the second is cut.
    assert len(recs) == 1
    assert recs[0].request_id == "A"


# ---- 4. Token-estimate proxy is exactly char/4 ----

def test_token_estimate_uses_chars_over_four():
    assert CHARS_PER_TOKEN_PROXY == 4.0
    assert chars_to_token_estimate(0) == 0
    assert chars_to_token_estimate(4) == 1
    assert chars_to_token_estimate(40) == 10
    # Use 43/4 = 10.75 -> 11 (Python's banker rounding makes 10.5 -> 10).
    assert chars_to_token_estimate(43) == 11
    # All ShareGPT records report char/4 proxy as the token-count source.
    recs = sharegpt_aiperf.load_json_path(str(FIXTURE))
    for r in recs:
        assert r.token_count_source == "char_div_4_proxy"


def test_token_estimate_matches_char_count(tmp_path):
    one = tmp_path / "one.json"
    one.write_text(json.dumps([
        {"id": "X", "conversations": [
            {"from": "human", "value": "a" * 80},
            {"from": "gpt", "value": "b" * 200},
        ]}
    ]))
    recs = sharegpt_aiperf.load_json_path(str(one))
    assert recs[0].prompt_chars == 80
    assert recs[0].response_chars == 200
    assert recs[0].prompt_tokens_est == 20
    assert recs[0].response_tokens_est == 50


# ---- 5. Missing-field honesty ----

def test_missing_fields_always_none():
    recs = sharegpt_aiperf.load_json_path(str(FIXTURE))
    for r in recs:
        assert r.timestamp_s is None
        assert r.model_id is None
        assert r.language is None
        assert r.prompt_tokens_real is None
        assert r.response_tokens_real is None
        assert r.deadline_s is None
        assert r.e2e_latency_s is None


# ---- 6. Failure flag = response_chars == 0 ----

def test_failure_flag_when_no_response(tmp_path):
    one = tmp_path / "one.json"
    one.write_text(json.dumps([
        {"id": "fail", "conversations": [
            {"from": "human", "value": "hi"}
        ]},
        {"id": "ok", "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]}
    ]))
    recs = sharegpt_aiperf.load_json_path(str(one))
    fail_rec = [r for r in recs if r.request_id == "fail"][0]
    ok_rec = [r for r in recs if r.request_id == "ok"][0]
    assert fail_rec.is_failure is True
    assert ok_rec.is_failure is False


# ---- 7. Summary records honesty flags ----

def test_summary_records_no_timestamps_no_real_tokens():
    recs = sharegpt_aiperf.load_json_path(str(FIXTURE))
    s = sharegpt_aiperf.summarize(recs)
    assert s.has_timestamps is False
    assert s.has_real_tokens is False
    assert s.has_model_id is False
    assert s.has_language is False
    assert s.token_count_source_distribution == {"char_div_4_proxy": 3}
    assert s.row_count == 3


# ---- 8. Committed processed summary, when present, matches schema ----

REQUIRED_PROCESSED_KEYS = {
    "dataset", "provenance", "source_url", "source_repo_url",
    "aiperf_docs_url", "bounded_download", "filters", "summary", "records",
}


@pytest.mark.skipif(not PROCESSED.exists(),
                    reason="processed summary not committed yet")
def test_committed_processed_summary_shape():
    with open(PROCESSED) as fh:
        payload = json.load(fh)
    missing = REQUIRED_PROCESSED_KEYS - set(payload.keys())
    assert not missing, f"processed summary missing keys: {missing}"
    assert payload["dataset"] == sharegpt_aiperf.DATASET_NAME
    assert payload["provenance"] == sharegpt_aiperf.PROVENANCE
    assert payload["source_url"] == sharegpt_aiperf.DEFAULT_SOURCE_URL
    # Records list does not store raw text.
    for r in payload["records"][:5]:
        assert "raw_text" not in r
        assert r["token_count_source"] == "char_div_4_proxy"


# ---- 9. Processed summary file size ≤ 100 MB ----

@pytest.mark.skipif(not PROCESSED.exists(),
                    reason="processed summary not committed yet")
def test_processed_summary_within_user_spec_bound():
    sz = PROCESSED.stat().st_size
    assert sz <= USER_SPEC_MAX_BYTES, (
        f"committed processed summary is {sz:,} bytes — exceeds user-spec "
        f"100 MB bound")


# ---- 10. Bounded-download manifest, when present ----

@pytest.mark.skipif(not MANIFEST.exists(),
                    reason="bounded-download manifest not committed yet")
def test_bounded_download_manifest_shape():
    with open(MANIFEST) as fh:
        m = json.load(fh)
    assert "url" in m and "requested_bytes" in m
    assert "downloaded_bytes" in m and "dest_path" in m
    assert m["downloaded_bytes"] <= m["requested_bytes"] + 1


# ---- 11. Role-sequence signature helper ----

def test_role_sequence_signature_simple():
    assert role_sequence_signature(["human", "gpt"]) == "h-g"
    assert role_sequence_signature(["system", "user", "assistant"]) == "s-h-g"
    assert role_sequence_signature(["chatgpt", "model"]) == "g-g"
    assert role_sequence_signature(["weird-role"]) == "x"
