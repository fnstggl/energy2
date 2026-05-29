"""BurstGPT public-trace ingester — CANONICAL_TRACE_BACKTEST_BURSTGPT_V1.

BurstGPT (https://github.com/HPMLL/BurstGPT) is a real LLM-serving workload
trace (request/response token counts + arrival timestamps for ChatGPT/GPT-4
traffic). This module normalizes it into the cross-dataset
``NormalizedLLMRequest`` contract in ``aurelius/traces/schema.py``.

Discovered schema (the published ``BurstGPT_1.csv``, verified against the raw
file)::

    Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type

Important honesty notes:

- The published ``BurstGPT_1.csv`` carries **no Session ID column** and **no
  Elapsed-time column**, even though the project README documents them for the
  fuller schema. This ingester therefore:
    * maps a ``Session ID`` column onto ``session_id`` / ``cache_affinity_key``
      *when present*, and
    * when it is absent (BurstGPT_1.csv), leaves ``session_id = None`` and sets
      ``cache_affinity_key`` to a **model-level locality proxy** (``model:<m>``).
      This is an honest proxy for prefix locality — it is **not** a real KV
      cache hit rate and must never be reported as one.
- BurstGPT's ``Elapsed time`` (when present) is the *end-to-end final response
  time*, **not** TTFT. We never claim TTFT is measured from BurstGPT.
- Failures: a request is a failure when ``Response tokens == 0`` (the project's
  own failure convention) OR — only when an Elapsed-time column exists — when
  that elapsed value is missing/invalid. When there is no Elapsed column at all
  (BurstGPT_1.csv) the elapsed signal cannot mark failures, so only the
  zero-response rule applies.
"""

from __future__ import annotations

import csv
import random
from typing import Iterable, Optional

from .schema import (
    NormalizedLLMRequest,
    summarize_trace,
    validate_columns,
)

# --- Raw column names (exact strings in the BurstGPT CSV header) -------------
COL_TIMESTAMP = "Timestamp"
COL_MODEL = "Model"
COL_REQUEST_TOKENS = "Request tokens"
COL_RESPONSE_TOKENS = "Response tokens"
COL_TOTAL_TOKENS = "Total tokens"
COL_LOG_TYPE = "Log Type"
# Optional columns (documented by the BurstGPT README; absent from BurstGPT_1.csv).
COL_SESSION_ID = "Session ID"
COL_ELAPSED = "Elapsed time"

# Always-present columns — the schema guard checks exactly these.
REQUIRED_COLUMNS = (
    COL_TIMESTAMP,
    COL_MODEL,
    COL_REQUEST_TOKENS,
    COL_RESPONSE_TOKENS,
    COL_TOTAL_TOKENS,
    COL_LOG_TYPE,
)
OPTIONAL_COLUMNS = (COL_SESSION_ID, COL_ELAPSED)

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
)
DATASET_NAME = "burstgpt"


def _to_int(value: Optional[str]) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))  # tolerate "18.0"-style ints


def _to_float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BurstGPTSource:
    """Normalizes BurstGPT raw CSV rows into ``NormalizedLLMRequest``."""

    name = DATASET_NAME
    required_columns = REQUIRED_COLUMNS
    default_source_url = DEFAULT_SOURCE_URL

    def __init__(self, *, has_session: bool = False, has_elapsed: bool = False):
        # Whether the source file actually carries the optional columns; set by
        # ``load_csv`` after inspecting the real header.
        self.has_session = has_session
        self.has_elapsed = has_elapsed

    def _cache_affinity_key(self, session_id: Optional[str], model: str) -> str:
        if session_id is not None and session_id != "":
            return session_id
        # Honest model-level locality proxy when no session id exists.
        return f"model:{model}"

    def normalize_row(self, row: dict, index: int) -> NormalizedLLMRequest:
        model = (row.get(COL_MODEL) or "unknown").strip()
        prompt_tokens = max(0, _to_int(row.get(COL_REQUEST_TOKENS)))
        output_tokens = max(0, _to_int(row.get(COL_RESPONSE_TOKENS)))
        total_tokens = max(0, _to_int(row.get(COL_TOTAL_TOKENS)))
        log_type = (row.get(COL_LOG_TYPE) or "").strip()
        timestamp_s = float(row.get(COL_TIMESTAMP) or 0.0)

        session_id: Optional[str] = None
        if self.has_session:
            raw_sid = (row.get(COL_SESSION_ID) or "").strip()
            session_id = raw_sid or None

        elapsed_s: Optional[float] = None
        elapsed_invalid = False
        if self.has_elapsed:
            elapsed_s = _to_float_or_none(row.get(COL_ELAPSED))
            # Only treat elapsed as a failure signal when the column exists.
            elapsed_invalid = elapsed_s is None or elapsed_s <= 0.0

        is_failure = (output_tokens == 0) or elapsed_invalid

        return NormalizedLLMRequest(
            request_id=f"burstgpt-{index}",
            timestamp_s=timestamp_s,
            session_id=session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            elapsed_s=elapsed_s,
            log_type=log_type,
            is_failure=is_failure,
            cache_affinity_key=self._cache_affinity_key(session_id, model),
        )

    def normalize(self, rows: Iterable[dict]) -> list[NormalizedLLMRequest]:
        return [self.normalize_row(row, i) for i, row in enumerate(rows)]


def load_csv(
    path: str,
    *,
    sample_size: Optional[int] = None,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    include_failures: bool = False,
    scale_rps: float = 1.0,
    seed: int = 0,
) -> list[NormalizedLLMRequest]:
    """Load + normalize a BurstGPT CSV with the ingestion filters applied.

    Filter order (deterministic):
      1. parse + normalize all rows,
      2. time window ``[start_s, start_s + duration_s)`` (absolute trace seconds),
      3. drop failures unless ``include_failures``,
      4. uniform random ``sample_size`` (seeded, then re-sorted by time),
      5. ``scale_rps`` time-warp (compresses/dilates timestamps about the window
         start so arrival density scales by the factor; >1 = denser/busier).

    Raises ``TraceSchemaError`` if required columns are missing.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        validate_columns(header, REQUIRED_COLUMNS, DATASET_NAME)
        header_set = set(header or [])
        source = BurstGPTSource(
            has_session=COL_SESSION_ID in header_set,
            has_elapsed=COL_ELAPSED in header_set,
        )
        requests = source.normalize(reader)

    # 2. time window
    if start_s is not None or duration_s is not None:
        lo = start_s if start_s is not None else float("-inf")
        hi = (start_s or 0.0) + duration_s if duration_s is not None else float("inf")
        requests = [r for r in requests if lo <= r.timestamp_s < hi]

    # 3. failures
    if not include_failures:
        requests = [r for r in requests if not r.is_failure]

    # keep deterministic time order before sampling
    requests.sort(key=lambda r: (r.timestamp_s, r.request_id))

    # 4. sampling
    if sample_size is not None and 0 <= sample_size < len(requests):
        rng = random.Random(seed)
        requests = rng.sample(requests, sample_size)
        requests.sort(key=lambda r: (r.timestamp_s, r.request_id))

    # 5. scale_rps time-warp
    if scale_rps and scale_rps > 0 and scale_rps != 1.0 and requests:
        t0 = requests[0].timestamp_s
        warped = []
        for r in requests:
            new_ts = t0 + (r.timestamp_s - t0) / scale_rps
            warped.append(
                NormalizedLLMRequest(
                    request_id=r.request_id,
                    timestamp_s=new_ts,
                    session_id=r.session_id,
                    model=r.model,
                    prompt_tokens=r.prompt_tokens,
                    output_tokens=r.output_tokens,
                    total_tokens=r.total_tokens,
                    elapsed_s=r.elapsed_s,
                    log_type=r.log_type,
                    is_failure=r.is_failure,
                    cache_affinity_key=r.cache_affinity_key,
                )
            )
        requests = warped

    return requests


def summarize(requests, *, bin_seconds: float = 60.0):
    """Convenience wrapper around the dataset-agnostic ``summarize_trace``."""
    return summarize_trace(requests, dataset=DATASET_NAME, bin_seconds=bin_seconds)
