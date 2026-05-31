"""Eval-class workload request schema (sibling of NormalizedLLMRequest).

Eval-class public traces (LMSYS Chatbot Arena, ShareGPT, MT-Bench, etc.) carry
*conversation shape* — turn counts, role sequences, prompt/response text — but
typically do **not** carry the per-request arrival timestamps, model id,
measured latency, or session/cache key that
``aurelius/traces/schema.py::NormalizedLLMRequest`` requires. Forcing these
records into ``NormalizedLLMRequest`` would mean inventing fields
(`timestamp_s`, `model`, `cache_affinity_key`) — exactly the pattern the
ingestion-honesty rule in ``docs/PUBLIC_TRACE_BACKTESTS.md`` prohibits.

This module defines a narrow, separate ``EvalWorkloadRequest`` contract:
mandatory turn / role shape, optional arrival / model / latency fields, and
**explicitly labelled token-estimate proxies** computed from text-character
counts. The Eval Workload Frontier v1 consumes this record; the existing
serving / training / residency machinery is untouched.

Honesty rules:

- Token counts that come from a **measurement** are ``prompt_tokens_real`` /
  ``response_tokens_real``. Token counts that are estimated from character
  counts are ``prompt_tokens_est`` / ``response_tokens_est`` and are flagged
  by ``token_count_source = "char_div_4_proxy" | "real_token_count" | "none"``.
- Missing fields stay ``None``; we never zero-fill.
- The record carries an explicit ``provenance`` label (e.g.
  ``"sharegpt_52k_head_sample_v1"``) so reports can trace every field back to
  the source.
- ``deadline_s`` is optional and stays ``None`` for the raw trace.
  Synthetic-deadline scenarios MUST tag the deadline source via
  ``synthetic_scenario_label``; the v1 ingesters never emit a non-None deadline
  themselves (the frontier benchmark provides the synthetic deadline at
  evaluation time, not the trace).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


# Allowed token-count source labels. The ingester records WHICH source the
# numeric fields came from so the frontier can refuse to draw a conclusion
# from a proxy when a real signal is required.
TOKEN_COUNT_SOURCES = frozenset({
    "real_token_count",
    "char_div_4_proxy",
    "none",
})


# Char-to-token proxy ratio. 4 chars/token is the standard rule-of-thumb for
# English text (OpenAI tokenizer rough average); it is **NOT** a real tokenizer
# call and reports must label it as a proxy.
CHARS_PER_TOKEN_PROXY = 4.0


class EvalWorkloadSchemaError(ValueError):
    """Raised when an eval-class trace is missing required fields or has bad
    values."""


@dataclass(frozen=True)
class EvalWorkloadRequest:
    """One normalized eval-class request — the cross-eval-dataset contract.

    Required (every source provides them):

    - ``request_id``: stable per-record id.
    - ``turn_count``: ``len(conversation)`` (>= 1).
    - ``role_sequence_signature``: compact string capturing the alternation
      pattern, e.g. ``"h-g-h-g"`` for a 4-turn user/assistant exchange.
    - ``token_count_source``: which source the prompt/response numbers came
      from (real tokenizer vs char/4 proxy vs none).
    - ``provenance``: free-form label naming the source variant (e.g.
      ``"sharegpt_52k_head_sample_v1"``).

    Optional (kept ``None`` when the source omits them):

    - ``timestamp_s``: arrival time in absolute seconds. ShareGPT has none.
    - ``model_id``: responding model id. ShareGPT has none; LMSYS has it.
    - ``language``: language label when present.
    - ``prompt_tokens_real`` / ``response_tokens_real``: real measured tokens.
    - ``prompt_tokens_est`` / ``response_tokens_est``: char/4 proxy.
    - ``prompt_chars`` / ``response_chars``: raw char counts (no text stored).
    - ``e2e_latency_s``: end-to-end latency (NOT TTFT). Most eval traces
      lack this; we never claim TTFT.
    - ``is_failure``: only set True when the source explicitly marks a
      failure / no-response row.
    - ``deadline_s``: ALWAYS ``None`` for raw eval-trace records; the
      Eval Workload Frontier v1 synthesises deadlines at evaluation time
      from a labelled scenario, not from the trace.
    """

    request_id: str
    turn_count: int
    role_sequence_signature: str
    token_count_source: str
    provenance: str
    timestamp_s: Optional[float] = None
    model_id: Optional[str] = None
    language: Optional[str] = None
    prompt_tokens_real: Optional[int] = None
    response_tokens_real: Optional[int] = None
    prompt_tokens_est: Optional[int] = None
    response_tokens_est: Optional[int] = None
    prompt_chars: Optional[int] = None
    response_chars: Optional[int] = None
    e2e_latency_s: Optional[float] = None
    is_failure: bool = False
    deadline_s: Optional[float] = None

    def __post_init__(self):
        if self.turn_count < 1:
            raise EvalWorkloadSchemaError(
                f"turn_count must be >= 1; got {self.turn_count}")
        if self.token_count_source not in TOKEN_COUNT_SOURCES:
            raise EvalWorkloadSchemaError(
                f"unknown token_count_source {self.token_count_source!r}; "
                f"expected one of {sorted(TOKEN_COUNT_SOURCES)}")
        if not self.provenance:
            raise EvalWorkloadSchemaError("provenance label is required")
        for f in ("prompt_tokens_real", "response_tokens_real",
                  "prompt_tokens_est", "response_tokens_est",
                  "prompt_chars", "response_chars"):
            v = getattr(self, f)
            if v is not None and v < 0:
                raise EvalWorkloadSchemaError(f"{f} must be >= 0; got {v}")
        # The estimator-proxy fields and the real-token fields are
        # mutually-additive: at most one labelled source per record. We do
        # NOT forbid both being populated (a future ingester may carry real
        # tokens AND a proxy for cross-check), but we DO require
        # ``token_count_source`` to honestly name the headline source.

    @property
    def effective_prompt_tokens(self) -> Optional[int]:
        """Best-effort prompt-token count, preferring real over proxy."""
        if self.prompt_tokens_real is not None:
            return self.prompt_tokens_real
        return self.prompt_tokens_est

    @property
    def effective_response_tokens(self) -> Optional[int]:
        """Best-effort response-token count, preferring real over proxy."""
        if self.response_tokens_real is not None:
            return self.response_tokens_real
        return self.response_tokens_est

    @property
    def effective_total_tokens(self) -> Optional[int]:
        p = self.effective_prompt_tokens
        r = self.effective_response_tokens
        if p is None and r is None:
            return None
        return (p or 0) + (r or 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalWorkloadRequest":
        def _opt_int(k):
            v = d.get(k)
            return None if v in (None, "") else int(v)

        def _opt_float(k):
            v = d.get(k)
            return None if v in (None, "") else float(v)

        return cls(
            request_id=str(d["request_id"]),
            turn_count=int(d["turn_count"]),
            role_sequence_signature=str(d["role_sequence_signature"]),
            token_count_source=str(d["token_count_source"]),
            provenance=str(d["provenance"]),
            timestamp_s=_opt_float("timestamp_s"),
            model_id=(None if d.get("model_id") in (None, "")
                      else str(d["model_id"])),
            language=(None if d.get("language") in (None, "")
                      else str(d["language"])),
            prompt_tokens_real=_opt_int("prompt_tokens_real"),
            response_tokens_real=_opt_int("response_tokens_real"),
            prompt_tokens_est=_opt_int("prompt_tokens_est"),
            response_tokens_est=_opt_int("response_tokens_est"),
            prompt_chars=_opt_int("prompt_chars"),
            response_chars=_opt_int("response_chars"),
            e2e_latency_s=_opt_float("e2e_latency_s"),
            is_failure=bool(d.get("is_failure", False)),
            deadline_s=_opt_float("deadline_s"),
        )


def chars_to_token_estimate(n_chars: int) -> int:
    """Char-to-token proxy used by every eval ingester. Labelled as proxy."""
    if n_chars <= 0:
        return 0
    return max(1, int(round(n_chars / CHARS_PER_TOKEN_PROXY)))


def role_sequence_signature(roles: list) -> str:
    """Compact role alternation signature.

    Maps standard role labels onto single letters and joins with ``-``:
      - ``human`` / ``user`` -> ``h``
      - ``gpt`` / ``assistant`` / ``chatgpt`` / ``model`` -> ``g``
      - ``system`` -> ``s``
      - anything else -> ``x``
    """
    out = []
    for r in roles:
        s = (str(r) or "").strip().lower()
        if s in ("human", "user"):
            out.append("h")
        elif s in ("gpt", "assistant", "chatgpt", "model"):
            out.append("g")
        elif s == "system":
            out.append("s")
        else:
            out.append("x")
    return "-".join(out)


@dataclass(frozen=True)
class EvalWorkloadSummary:
    """Descriptive stats over a list of ``EvalWorkloadRequest`` records.

    Pure / deterministic / stdlib-only — same shape as
    ``schema.TraceSummary``.
    """

    dataset: str
    provenance: str
    row_count: int
    has_timestamps: bool
    has_real_tokens: bool
    has_model_id: bool
    has_language: bool
    turn_count_p50: float
    turn_count_p95: float
    turn_count_p99: float
    prompt_tokens_eff_p50: float
    prompt_tokens_eff_p95: float
    prompt_tokens_eff_p99: float
    response_tokens_eff_p50: float
    response_tokens_eff_p95: float
    response_tokens_eff_p99: float
    role_sequence_top: dict
    model_distribution: dict
    language_distribution: dict
    token_count_source_distribution: dict
    failure_rate_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_eval_requests(
    requests, *, dataset: str, provenance: str
) -> EvalWorkloadSummary:
    """Build the descriptive summary the ingest script prints + commits."""
    from .schema import percentile  # reuse the existing nearest-rank pctile

    if not requests:
        return EvalWorkloadSummary(
            dataset=dataset, provenance=provenance, row_count=0,
            has_timestamps=False, has_real_tokens=False, has_model_id=False,
            has_language=False,
            turn_count_p50=0.0, turn_count_p95=0.0, turn_count_p99=0.0,
            prompt_tokens_eff_p50=0.0, prompt_tokens_eff_p95=0.0,
            prompt_tokens_eff_p99=0.0,
            response_tokens_eff_p50=0.0, response_tokens_eff_p95=0.0,
            response_tokens_eff_p99=0.0,
            role_sequence_top={}, model_distribution={},
            language_distribution={}, token_count_source_distribution={},
            failure_rate_pct=0.0,
        )

    turns = [r.turn_count for r in requests]
    prompts = [r.effective_prompt_tokens for r in requests
               if r.effective_prompt_tokens is not None]
    responses = [r.effective_response_tokens for r in requests
                 if r.effective_response_tokens is not None]

    role_top: dict = {}
    for r in requests:
        role_top[r.role_sequence_signature] = (
            role_top.get(r.role_sequence_signature, 0) + 1)
    # keep the 8 most common signatures
    role_top_sorted = dict(sorted(role_top.items(), key=lambda kv: -kv[1])[:8])

    model_dist: dict = {}
    for r in requests:
        if r.model_id is not None:
            model_dist[r.model_id] = model_dist.get(r.model_id, 0) + 1

    lang_dist: dict = {}
    for r in requests:
        if r.language is not None:
            lang_dist[r.language] = lang_dist.get(r.language, 0) + 1

    src_dist: dict = {}
    for r in requests:
        src_dist[r.token_count_source] = (
            src_dist.get(r.token_count_source, 0) + 1)

    failures = sum(1 for r in requests if r.is_failure)
    failure_rate = 100.0 * failures / len(requests)

    return EvalWorkloadSummary(
        dataset=dataset,
        provenance=provenance,
        row_count=len(requests),
        has_timestamps=any(r.timestamp_s is not None for r in requests),
        has_real_tokens=any(r.prompt_tokens_real is not None
                            or r.response_tokens_real is not None
                            for r in requests),
        has_model_id=any(r.model_id is not None for r in requests),
        has_language=any(r.language is not None for r in requests),
        turn_count_p50=percentile(turns, 50),
        turn_count_p95=percentile(turns, 95),
        turn_count_p99=percentile(turns, 99),
        prompt_tokens_eff_p50=percentile(prompts, 50) if prompts else 0.0,
        prompt_tokens_eff_p95=percentile(prompts, 95) if prompts else 0.0,
        prompt_tokens_eff_p99=percentile(prompts, 99) if prompts else 0.0,
        response_tokens_eff_p50=(
            percentile(responses, 50) if responses else 0.0),
        response_tokens_eff_p95=(
            percentile(responses, 95) if responses else 0.0),
        response_tokens_eff_p99=(
            percentile(responses, 99) if responses else 0.0),
        role_sequence_top=role_top_sorted,
        model_distribution=dict(sorted(model_dist.items())),
        language_distribution=dict(sorted(lang_dist.items())),
        token_count_source_distribution=dict(sorted(src_dist.items())),
        failure_rate_pct=failure_rate,
    )
