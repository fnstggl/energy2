"""LMSYS Chatbot Arena conversations ingester (gated — stub + helper).

The dataset is at
https://huggingface.co/datasets/lmsys/chatbot_arena_conversations and is
**gated** (`gated: auto` on the HuggingFace API) — auto-download requires the
user to accept the LMSYS terms-of-use and supply a HuggingFace API token. The
ingester therefore cannot auto-download in CI; it provides:

1. The published dataset schema (transcribed from the README card so tests can
   pin it without touching the dataset).
2. A ``normalize_record`` adapter that maps an LMSYS row onto
   ``EvalWorkloadRequest`` (so the eval frontier works against LMSYS the
   moment a user supplies a local file).
3. A ``download_gated`` helper that requires ``HF_TOKEN`` and explicitly
   refuses to proceed without it.

This is the standard pattern used by ``ingest_philly.py`` for the git-LFS
gated Philly tarball (see ``docs/PUBLIC_TRACE_BACKTESTS.md`` §3d).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

from .eval_schema import (
    EvalWorkloadRequest,
    EvalWorkloadSchemaError,
    chars_to_token_estimate,
    role_sequence_signature,
)

DATASET_NAME = "lmsys_chatbot_arena"
PROVENANCE = "lmsys_chatbot_arena_conversations_v1"

DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/"
    "resolve/main/data/train-00000-of-00001-cced8514c7ed782a.parquet"
)
SOURCE_REPO_URL = (
    "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations"
)


# Schema (from the HF dataset card; transcribed so tests can pin it without
# downloading the dataset). Fields the ingester cares about:
LMSYS_FIELDS = frozenset({
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "judge",
    "conversation_a",
    "conversation_b",
    "turn",
    "anony",
    "language",
    "tstamp",
    "openai_moderation",
    "toxic_chat_tag",
})
# Of those, the eval-frontier-relevant subset (we IGNORE moderation /
# toxic_chat_tag for the conversation-shape backtest).
LMSYS_FIELDS_RELEVANT = (
    "question_id",
    "model_a", "model_b",
    "conversation_a", "conversation_b",
    "turn", "language", "tstamp",
)
LMSYS_TURN_KEYS = frozenset({"content", "role"})

LMSYS_GATED_BANNER = (
    "LMSYS Chatbot Arena conversations is a GATED HuggingFace dataset.\n"
    "Download requires:\n"
    "  1. Visit https://huggingface.co/datasets/lmsys/chatbot_arena_conversations\n"
    "     and accept the dataset terms-of-use.\n"
    "  2. Create an access token at https://huggingface.co/settings/tokens\n"
    "  3. Export HF_TOKEN=<your-token>\n"
    "Then re-run this ingester."
)


class LMSYSGatedAccessError(RuntimeError):
    """Raised when the LMSYS dataset cannot be auto-downloaded due to its
    gated terms-of-use. Tests rely on this name to assert that the gated
    path is NOT silently bypassed."""


def normalize_record(rec: dict, *, provenance: str = PROVENANCE,
                     side: str = "a") -> EvalWorkloadRequest:
    """Map one LMSYS row + one conversation side onto ``EvalWorkloadRequest``.

    LMSYS rows carry *two* conversations (model_a + model_b). The eval
    frontier uses one side per request; the caller picks ``side="a"`` or
    ``"b"``. Both sides share the question_id; we append the side to the
    request_id so the (a, b) pair stays distinguishable downstream.
    """
    if side not in ("a", "b"):
        raise EvalWorkloadSchemaError(
            f"side must be 'a' or 'b'; got {side!r}")
    qid_key = "question_id"
    if qid_key not in rec:
        raise EvalWorkloadSchemaError(
            f"lmsys record missing required key {qid_key!r}")
    conv_key = f"conversation_{side}"
    model_key = f"model_{side}"
    if conv_key not in rec or model_key not in rec:
        raise EvalWorkloadSchemaError(
            f"lmsys record missing required side keys "
            f"{[conv_key, model_key]}")
    convs = rec[conv_key]
    if not isinstance(convs, list) or not convs:
        raise EvalWorkloadSchemaError(
            f"lmsys {conv_key!r} must be a non-empty list")

    roles: list = []
    prompt_chars = 0
    response_chars = 0
    for t in convs:
        if not isinstance(t, dict):
            raise EvalWorkloadSchemaError(
                f"lmsys turn not a dict: {type(t).__name__}")
        extra = set(t.keys()) - LMSYS_TURN_KEYS
        if extra:
            raise EvalWorkloadSchemaError(
                f"lmsys turn has unknown keys {sorted(extra)}; "
                f"expected exactly {sorted(LMSYS_TURN_KEYS)}")
        role = (t.get("role") or "").strip().lower()
        val = t.get("content") or ""
        roles.append(role)
        nchars = len(val) if isinstance(val, str) else 0
        if role in ("human", "user", "system"):
            prompt_chars += nchars
        elif role in ("assistant", "model", "gpt", "chatgpt"):
            response_chars += nchars

    tstamp = rec.get("tstamp")
    lang = rec.get("language")
    model_id = rec.get(model_key)

    return EvalWorkloadRequest(
        request_id=f"{rec[qid_key]}-{side}",
        turn_count=len(convs),
        role_sequence_signature=role_sequence_signature(roles),
        token_count_source="char_div_4_proxy",
        provenance=provenance,
        timestamp_s=(float(tstamp) if isinstance(tstamp, (int, float))
                     else None),
        model_id=(str(model_id) if model_id is not None else None),
        language=(str(lang) if lang is not None else None),
        prompt_tokens_real=None,
        response_tokens_real=None,
        prompt_tokens_est=chars_to_token_estimate(prompt_chars),
        response_tokens_est=chars_to_token_estimate(response_chars),
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        e2e_latency_s=None,
        is_failure=(response_chars == 0),
        deadline_s=None,
    )


def download_gated(*, url: str = DEFAULT_SOURCE_URL, dest_path: str,
                   hf_token: Optional[str] = None,
                   max_bytes: Optional[int] = None) -> dict:
    """Download the gated LMSYS parquet using ``HF_TOKEN``.

    Refuses to proceed without a token (``LMSYSGatedAccessError``). Optional
    ``max_bytes`` HTTP-Range cap mirrors the ShareGPT bounded-ingest pattern,
    though parquet does not parse partial slices cleanly — the cap is mostly
    a safety budget.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise LMSYSGatedAccessError(LMSYS_GATED_BANNER)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    if max_bytes is not None and max_bytes > 0:
        headers["Range"] = f"bytes=0-{int(max_bytes) - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 documented public URL
        status = resp.getcode()
        data = resp.read(max_bytes) if max_bytes else resp.read()
    with open(dest_path, "wb") as fh:
        fh.write(data)
    return {
        "url": url,
        "requested_bytes": (int(max_bytes) if max_bytes else None),
        "downloaded_bytes": len(data),
        "http_status": int(status),
        "dest_path": dest_path,
    }
