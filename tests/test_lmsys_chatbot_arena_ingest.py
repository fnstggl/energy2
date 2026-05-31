"""Tests for the (gated) LMSYS Chatbot Arena ingester.

The dataset is gated on HuggingFace; auto-download requires accepting the
LMSYS terms of use and supplying ``HF_TOKEN``. The tests pin the gated
behavior so the script can never silently bypass the gate:

1.  ``download_gated`` raises ``LMSYSGatedAccessError`` when no token is
    available (env var unset + arg None).
2.  The gated banner is non-empty and includes the URL + the env-var name.
3.  Schema field set matches what the published dataset card declares.
4.  ``normalize_record`` maps one row + one side onto ``EvalWorkloadRequest``
    with the expected proxy fields.
5.  Unknown turn keys are rejected.
6.  ``side`` is restricted to ``"a"`` / ``"b"``.
7.  The script's CLI exit code is non-zero when no token is set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces import lmsys_chatbot_arena  # noqa: E402
from aurelius.traces.eval_schema import EvalWorkloadSchemaError  # noqa: E402
from aurelius.traces.lmsys_chatbot_arena import LMSYSGatedAccessError  # noqa: E402


# ---- 1. Gated refusal ----

def test_download_gated_raises_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dest = tmp_path / "parquet.bin"
    with pytest.raises(LMSYSGatedAccessError):
        lmsys_chatbot_arena.download_gated(dest_path=str(dest))


def test_download_gated_raises_explicit_none_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dest = tmp_path / "parquet.bin"
    with pytest.raises(LMSYSGatedAccessError):
        lmsys_chatbot_arena.download_gated(dest_path=str(dest), hf_token=None)


# ---- 2. Gated banner ----

def test_gated_banner_mentions_url_and_env_var():
    banner = lmsys_chatbot_arena.LMSYS_GATED_BANNER
    assert "chatbot_arena_conversations" in banner
    assert "HF_TOKEN" in banner
    assert "huggingface.co" in banner


# ---- 3. Schema fields ----

def test_schema_field_set_pinned():
    # These are the canonical card fields the ingester accepts. Adding /
    # removing a field upstream should require an explicit change here.
    assert "question_id" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "model_a" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "model_b" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "conversation_a" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "conversation_b" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "turn" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "tstamp" in lmsys_chatbot_arena.LMSYS_FIELDS
    assert "language" in lmsys_chatbot_arena.LMSYS_FIELDS


# ---- 4. normalize_record happy path ----

def _make_lmsys_row():
    return {
        "question_id": "q123",
        "model_a": "llama-2-7b-chat",
        "model_b": "gpt-3.5-turbo",
        "conversation_a": [
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am model A."},
        ],
        "conversation_b": [
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am model B."},
        ],
        "turn": 1,
        "language": "English",
        "tstamp": 1701000000.0,
        "winner": "model_a",
        "judge": "human",
    }


def test_normalize_record_side_a():
    rec = _make_lmsys_row()
    req = lmsys_chatbot_arena.normalize_record(rec, side="a")
    assert req.request_id == "q123-a"
    assert req.turn_count == 2
    assert req.role_sequence_signature == "h-g"
    assert req.model_id == "llama-2-7b-chat"
    assert req.timestamp_s == 1701000000.0
    assert req.language == "English"
    assert req.token_count_source == "char_div_4_proxy"
    assert req.prompt_chars == len("Hello, who are you?")
    assert req.response_chars == len("I am model A.")
    assert req.is_failure is False
    assert req.deadline_s is None


def test_normalize_record_side_b():
    rec = _make_lmsys_row()
    req = lmsys_chatbot_arena.normalize_record(rec, side="b")
    assert req.request_id == "q123-b"
    assert req.model_id == "gpt-3.5-turbo"


# ---- 5. Unknown turn key rejected ----

def test_normalize_record_unknown_turn_key():
    rec = _make_lmsys_row()
    rec["conversation_a"][0]["extra_key"] = "nope"
    with pytest.raises(EvalWorkloadSchemaError):
        lmsys_chatbot_arena.normalize_record(rec, side="a")


# ---- 6. Side restricted ----

def test_side_restricted_to_a_or_b():
    rec = _make_lmsys_row()
    with pytest.raises(EvalWorkloadSchemaError):
        lmsys_chatbot_arena.normalize_record(rec, side="c")


# ---- 7. CLI exits non-zero without token ----

def test_cli_no_token_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    script = REPO_ROOT / "scripts" / "ingest_lmsys_chatbot_arena.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--raw-path", str(tmp_path / "x.parquet"),
         "--processed-path", str(tmp_path / "x.json"),
         "--manifest-path", str(tmp_path / "m.json")],
        capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "BLOCKED_GATED_DATASET" in result.stderr
