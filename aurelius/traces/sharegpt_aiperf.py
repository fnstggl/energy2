"""ShareGPT eval-conversation-shape ingester (bounded only).

ShareGPT is the canonical *eval / benchmark conversation-shape* dataset cited
by the AIPerf docs (https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-share-gpt-dataset)
and used by many open LLM evaluation harnesses (vLLM benchmark, FastChat,
Vicuna training). The public mirror this ingester reads is
``RyokoAI/ShareGPT52K`` on HuggingFace (the "old" 52K subset is the smallest
and the canonical benchmark shape). The dataset is **not** customer telemetry
and is **not** a measured serving trace — it has no arrival timestamps and no
measured latency. It is **conversation-shape proxy** for eval-class workloads
only.

Discovered schema (verified against ``sg_52k.json``)::

    [
      {
        "id": "<short-string>",
        "conversations": [
          {"from": "human|gpt|chatgpt|system", "value": "<text>"},
          ...
        ]
      },
      ...
    ]

Strictly: a JSON array of records, each with exactly two top-level keys
(``id``, ``conversations``), and each turn with exactly two keys (``from``,
``value``). The ingester refuses to silently accept unknown keys.

Honesty / scope:

- No arrival timestamps in the source → ``timestamp_s = None`` per record.
- No model id (the responding model is the unknown teacher) → ``model_id = None``.
- No measured latency / TTFT → ``e2e_latency_s = None``. **No TTFT claim.**
- No real tokenizer call → ``token_count_source = "char_div_4_proxy"``
  (``chars_to_token_estimate`` from ``eval_schema.py``). Reports MUST label
  the result as a proxy.
- The bounded download is a fixed-size HTTP-Range slice (default 50 MB); the
  parser lenient-parses as many *complete* top-level records as fit, then
  stops. The remainder is **not** silently discarded — the ingest summary
  records the byte slice and the number of records actually parsed.
- The raw source file is ``data/external/sharegpt_aiperf/raw/sg_52k_head.json``
  and is **gitignored**; only the processed summary + a small fixture are
  committed.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Iterable, Optional

from .eval_schema import (
    EvalWorkloadRequest,
    EvalWorkloadSchemaError,
    chars_to_token_estimate,
    role_sequence_signature,
    summarize_eval_requests,
)


DATASET_NAME = "sharegpt_aiperf"
PROVENANCE = "sharegpt_52k_head_sample_v1"

# Public mirror used by AIPerf / vLLM / FastChat benchmark scripts. The
# ``old/sg_52k.json`` subset is the smallest variant and the one referenced as
# the canonical benchmark shape.
DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/"
    "old/sg_52k.json"
)
SOURCE_REPO_URL = "https://huggingface.co/datasets/RyokoAI/ShareGPT52K"
AIPERF_DOCS_URL = (
    "https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/"
    "profile-with-share-gpt-dataset"
)

# Top-level + per-turn keys we recognise. Any unknown key in a record/turn
# triggers ``EvalWorkloadSchemaError`` — the ingester refuses to silently
# normalize an unrecognised structure.
RECORD_KEYS = frozenset({"id", "conversations"})
TURN_KEYS = frozenset({"from", "value"})

# Default bounded download cap: 50 MB. The committed processed summary is
# small (≤ ~200 KB); the raw head sample is gitignored. The user spec caps
# the committed processed sample at 100 MB — leaving generous headroom.
DEFAULT_BOUNDED_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_records_from_partial_array(raw: str) -> Iterable[dict]:
    """Yield complete top-level records from a possibly-truncated JSON array.

    The bounded download may have cut the JSON mid-record; this parser walks
    the byte stream stateful-y and yields every COMPLETE ``{...}`` it finds at
    array depth 1, stopping cleanly when truncation hits a partial record.
    String escapes are honored so a ``{`` inside a string does not corrupt the
    depth count.
    """
    i = 0
    n = len(raw)
    # Skip leading whitespace + opening bracket.
    while i < n and raw[i] in " \t\n\r":
        i += 1
    if i >= n or raw[i] != "[":
        return
    i += 1
    while i < n:
        # Skip whitespace + commas between records.
        while i < n and raw[i] in " \t\n\r,":
            i += 1
        if i >= n or raw[i] == "]":
            return
        if raw[i] != "{":
            return
        # Walk one record honouring string escapes.
        depth = 0
        in_str = False
        esc = False
        j = i
        record_complete = False
        while j < n:
            c = raw[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(raw[i:j + 1])
                        except (json.JSONDecodeError, ValueError):
                            return
                        i = j + 1
                        record_complete = True
                        break
            j += 1
        if not record_complete:
            # Truncation hit a partial record — stop honestly.
            return


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_record(rec: dict, *, provenance: str = PROVENANCE
                     ) -> EvalWorkloadRequest:
    """Map one ShareGPT record onto :class:`EvalWorkloadRequest`.

    Raises ``EvalWorkloadSchemaError`` on unknown keys or missing required
    fields (``id``, ``conversations``).
    """
    if not isinstance(rec, dict):
        raise EvalWorkloadSchemaError(
            f"sharegpt record not a dict: {type(rec).__name__}")
    extra = set(rec.keys()) - RECORD_KEYS
    if extra:
        raise EvalWorkloadSchemaError(
            f"sharegpt record has unknown keys {sorted(extra)}; "
            f"expected exactly {sorted(RECORD_KEYS)}")
    if "id" not in rec or "conversations" not in rec:
        missing = RECORD_KEYS - set(rec.keys())
        raise EvalWorkloadSchemaError(
            f"sharegpt record missing required keys {sorted(missing)}")
    convs = rec["conversations"]
    if not isinstance(convs, list) or not convs:
        raise EvalWorkloadSchemaError(
            "sharegpt 'conversations' must be a non-empty list")

    roles: list = []
    prompt_chars = 0
    response_chars = 0
    for t in convs:
        if not isinstance(t, dict):
            raise EvalWorkloadSchemaError(
                f"sharegpt turn not a dict: {type(t).__name__}")
        extra_t = set(t.keys()) - TURN_KEYS
        if extra_t:
            raise EvalWorkloadSchemaError(
                f"sharegpt turn has unknown keys {sorted(extra_t)}; "
                f"expected exactly {sorted(TURN_KEYS)}")
        frm = (t.get("from") or "").strip().lower()
        val = t.get("value") or ""
        roles.append(frm)
        nchars = len(val) if isinstance(val, str) else 0
        # Map role -> prompt/response bucket. Honest mapping: human/system
        # contribute to prompt_chars; gpt/chatgpt/assistant contribute to
        # response_chars; anything else does not double-count.
        if frm in ("human", "user", "system"):
            prompt_chars += nchars
        elif frm in ("gpt", "chatgpt", "assistant", "model"):
            response_chars += nchars

    return EvalWorkloadRequest(
        request_id=str(rec["id"]),
        turn_count=len(convs),
        role_sequence_signature=role_sequence_signature(roles),
        token_count_source="char_div_4_proxy",
        provenance=provenance,
        timestamp_s=None,
        model_id=None,
        language=None,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_json_path(
    path: str,
    *,
    max_records: Optional[int] = None,
    provenance: str = PROVENANCE,
) -> list[EvalWorkloadRequest]:
    """Load + normalize a ShareGPT JSON file (possibly bounded / partial).

    The parser is tolerant of a head-bounded JSON file (e.g. produced by
    ``download_bounded``) — it stops at the first incomplete record.
    """
    with open(path, "r") as fh:
        raw = fh.read()
    out: list[EvalWorkloadRequest] = []
    for rec in _parse_records_from_partial_array(raw):
        try:
            req = normalize_record(rec, provenance=provenance)
        except EvalWorkloadSchemaError:
            raise
        out.append(req)
        if max_records is not None and len(out) >= max_records:
            break
    return out


def download_bounded(
    *,
    url: str = DEFAULT_SOURCE_URL,
    dest_path: str,
    max_bytes: int = DEFAULT_BOUNDED_BYTES,
) -> dict:
    """Bounded HTTP-Range download of the ShareGPT JSON head.

    Writes at most ``max_bytes`` bytes to ``dest_path`` (overwrite). Uses the
    HTTP Range header to ask the server for only the head slice. Returns a
    manifest dict (url, requested byte range, downloaded bytes, status).

    Raises ``OSError`` on network / IO errors.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{max(0, int(max_bytes) - 1)}"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 documented public URL
        status = resp.getcode()
        data = resp.read(max_bytes)
    with open(dest_path, "wb") as fh:
        fh.write(data)
    return {
        "url": url,
        "requested_bytes": int(max_bytes),
        "downloaded_bytes": len(data),
        "http_status": int(status),
        "dest_path": dest_path,
    }


def summarize(requests: list, *, dataset: str = DATASET_NAME,
              provenance: str = PROVENANCE):
    """Descriptive stats over a normalized request list."""
    return summarize_eval_requests(requests, dataset=dataset,
                                   provenance=provenance)
