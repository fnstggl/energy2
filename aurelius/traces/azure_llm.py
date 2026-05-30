"""Azure LLM inference-trace ingester — CANONICAL_TRACE_BACKTEST_AZURE_LLM_V1.

The Azure public LLM inference traces (https://github.com/Azure/AzurePublicDataset)
record real production LLM-serving **token demand + arrival timing**. This module
normalizes them into the cross-dataset ``NormalizedLLMRequest`` contract in
``aurelius/traces/schema.py`` (the same one BurstGPT uses).

Discovered schema (verified against the raw 2023 + 2024 files)::

    TIMESTAMP,ContextTokens,GeneratedTokens

i.e. **exactly three columns**. ``TIMESTAMP`` is an absolute high-precision
datetime (e.g. ``2023-11-16 18:15:46.6805900`` — 7 fractional digits, .NET
ticks); ``ContextTokens`` is the input/prompt token count; ``GeneratedTokens``
is the output token count.

What Azure does **NOT** provide (and how we degrade honestly):

- **No model / service id.** ``model`` is set to a single ``"azure-llm"`` label.
- **No request / session / conversation id, no prefix info.** ``session_id`` and
  ``cache_affinity_key`` are ``None`` → the replay applies **no** cache-affinity
  benefit and the backtest **omits** ``cache_affinity_baseline``. Real cache
  affinity is unavailable for this trace.
- **No latency / TTFT / elapsed column.** ``elapsed_s`` is ``None``. This is a
  **token-demand and arrival replay, NOT a measured-latency replay**, and no
  TTFT is measured from Azure.
- **No explicit failure column.** Following the framework convention, a row is a
  failure only if ``GeneratedTokens == 0`` (none observed in the 2023 files).

The dataset ships two workload variants in separate files — ``conv``
(conversation) and ``code`` (coding assistant) — which are the only logical
workload signal; we record the variant as ``log_type``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Iterable, Optional

from .schema import (
    NormalizedLLMRequest,
    summarize_trace,
    validate_columns,
)

# --- Raw column names (exact strings in the Azure CSV header) ----------------
COL_TIMESTAMP = "TIMESTAMP"
COL_CONTEXT = "ContextTokens"
COL_GENERATED = "GeneratedTokens"

REQUIRED_COLUMNS = (COL_TIMESTAMP, COL_CONTEXT, COL_GENERATED)

DATASET_NAME = "azure_llm"
DEFAULT_MODEL = "azure-llm"

# Known public file URLs (raw, on GitHub) for the 2023 release. The 2024
# "_1week" variants live on Azure blob storage; pass --source-url to use them.
SOURCE_URLS = {
    "conv": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
    "code": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv",
}
DEFAULT_SOURCE_URL = SOURCE_URLS["conv"]


def variant_from_path(path: str) -> str:
    """Infer the workload variant (conv/code) from a filename; else 'unknown'."""
    low = path.lower()
    if "conv" in low:
        return "conv"
    if "code" in low:
        return "code"
    return "unknown"


def parse_timestamp_s(raw: str) -> float:
    """Parse an Azure TIMESTAMP into absolute POSIX seconds (UTC, sub-second).

    Handles the 7-fractional-digit .NET form ``YYYY-MM-DD HH:MM:SS.fffffff``
    that ``datetime.strptime`` cannot (it caps at 6). Naive timestamps are
    treated as UTC; only relative spacing matters downstream.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty Azure TIMESTAMP")
    if "." in raw:
        head, frac = raw.split(".", 1)
        frac_digits = "".join(ch for ch in frac if ch.isdigit())
        frac_seconds = (int(frac_digits) / (10 ** len(frac_digits))) if frac_digits else 0.0
    else:
        head, frac_seconds = raw, 0.0
    dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.timestamp() + frac_seconds


def _to_int(value: Optional[str]) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(float(value))


class AzureLLMSource:
    """Normalizes Azure LLM inference rows into ``NormalizedLLMRequest``."""

    name = DATASET_NAME
    required_columns = REQUIRED_COLUMNS
    default_source_url = DEFAULT_SOURCE_URL

    def __init__(self, *, variant: str = "unknown", model: str = DEFAULT_MODEL):
        self.variant = variant
        self.model = model

    def normalize_row(self, row: dict, index: int) -> NormalizedLLMRequest:
        prompt_tokens = max(0, _to_int(row.get(COL_CONTEXT)))
        output_tokens = max(0, _to_int(row.get(COL_GENERATED)))
        timestamp_s = parse_timestamp_s(row.get(COL_TIMESTAMP))
        # Azure has no failure column; only a zero-output row is a failure.
        is_failure = output_tokens == 0
        return NormalizedLLMRequest(
            request_id=f"azure-llm-{index}",
            timestamp_s=timestamp_s,
            session_id=None,          # Azure has no session/conversation id
            model=self.model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,  # not in file; derived
            elapsed_s=None,           # no latency in Azure → token-demand replay
            log_type=self.variant,    # conv/code workload variant
            is_failure=is_failure,
            cache_affinity_key=None,  # no session/prefix info → no cache proxy
        )

    def normalize(self, rows: Iterable[dict]) -> list[NormalizedLLMRequest]:
        return [self.normalize_row(row, i) for i, row in enumerate(rows)]


def load_csv(
    path: str,
    *,
    variant: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    sample_size: Optional[int] = None,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    include_failures: bool = False,
    scale_rps: float = 1.0,
    seed: int = 0,
) -> list[NormalizedLLMRequest]:
    """Load + normalize an Azure LLM CSV with the ingestion filters applied.

    Filters mirror the BurstGPT loader exactly (time window → failures →
    seeded sample → ``scale_rps`` time-warp). ``start_s``/``duration_s`` are
    *relative to the first request* (the Azure absolute epoch is opaque).
    Raises ``TraceSchemaError`` if required columns are missing.
    """
    import random

    variant = variant or variant_from_path(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        validate_columns(reader.fieldnames, REQUIRED_COLUMNS, DATASET_NAME)
        source = AzureLLMSource(variant=variant, model=model)
        requests = source.normalize(reader)

    requests.sort(key=lambda r: (r.timestamp_s, r.request_id))

    # time window relative to first request
    if (start_s is not None or duration_s is not None) and requests:
        base = requests[0].timestamp_s
        lo = base + start_s if start_s is not None else float("-inf")
        hi = base + (start_s or 0.0) + duration_s if duration_s is not None else float("inf")
        requests = [r for r in requests if lo <= r.timestamp_s < hi]

    if not include_failures:
        requests = [r for r in requests if not r.is_failure]

    requests.sort(key=lambda r: (r.timestamp_s, r.request_id))

    if sample_size is not None and 0 <= sample_size < len(requests):
        rng = random.Random(seed)
        requests = rng.sample(requests, sample_size)
        requests.sort(key=lambda r: (r.timestamp_s, r.request_id))

    if scale_rps and scale_rps > 0 and scale_rps != 1.0 and requests:
        t0 = requests[0].timestamp_s
        requests = [
            NormalizedLLMRequest(
                request_id=r.request_id,
                timestamp_s=t0 + (r.timestamp_s - t0) / scale_rps,
                session_id=r.session_id, model=r.model,
                prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens,
                total_tokens=r.total_tokens, elapsed_s=r.elapsed_s,
                log_type=r.log_type, is_failure=r.is_failure,
                cache_affinity_key=r.cache_affinity_key,
            )
            for r in requests
        ]

    return requests


def summarize(requests, *, bin_seconds: float = 60.0):
    """Convenience wrapper around the dataset-agnostic ``summarize_trace``."""
    return summarize_trace(requests, dataset=DATASET_NAME, bin_seconds=bin_seconds)
