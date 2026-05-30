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

# Azure LLM Inference Dataset **2024** — the week-long (May 10–19 2024) sample of
# multiple Azure LLM inference services used by DynamoLLM (HPCA 2025). Same
# 3-column schema as 2023 (TIMESTAMP, ContextTokens, GeneratedTokens) — verified
# against AzureLLMInferenceDataset2024.md — but two longer ``_1week`` variants on
# Azure blob storage. Licensed CC-BY; cite Stojkovic et al., HPCA 2025
# (arxiv 2408.00741). The files are ~0.7–1.1 GB each (tens of millions of rows),
# so the 2024 path is STREAMING (see ``stream_week_aggregate``).
DATASET_NAME_2024 = "azure_llm_2024"
SOURCE_URLS_2024 = {
    "code": ("https://azurepublicdatasettraces.blob.core.windows.net/"
             "azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv"),
    "conv": ("https://azurepublicdatasettraces.blob.core.windows.net/"
             "azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv"),
}
AZURE_2024_CITATION = (
    "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy "
    "Efficiency, HPCA 2025, Stojkovic et al. (arxiv.org/abs/2408.00741); "
    "dataset CC-BY (github.com/Azure/AzurePublicDataset)")


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

    Handles the 2023 7-fractional-digit .NET form ``YYYY-MM-DD HH:MM:SS.fffffff``
    (which ``datetime.strptime`` caps at 6) AND the 2024 form with a trailing
    timezone offset ``...SS.ffffff+00:00`` / ``Z``. Naive timestamps are treated
    as UTC; only relative spacing matters downstream.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty Azure TIMESTAMP")
    # strip a trailing timezone offset (Azure 2024 carries +00:00 = UTC)
    if raw.endswith("Z"):
        raw = raw[:-1]
    else:
        # an offset sign appears AFTER the time (date hyphens are within head)
        for sign in ("+", "-"):
            pos = raw.find(sign, 11)   # search past the 'YYYY-MM-DD ' prefix
            if pos != -1:
                raw = raw[:pos]
                break
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


# ===========================================================================
# Azure LLM 2024 week-long STREAMING aggregation (memory-bounded)
# ===========================================================================
#
# The 2024 ``_1week`` files are far too large (~0.7–1.1 GB, tens of millions of
# rows) to materialize as a list of NormalizedLLMRequest. We stream each CSV in
# one pass, binning into fixed arrival ticks + per-minute RPS, and accumulating
# token *histograms* (exact percentiles, bounded memory). The same honesty
# rules hold: no model/service id, no session/cache key, no latency/TTFT.

import datetime as _dt  # noqa: E402
import math as _math  # noqa: E402
import os  # noqa: E402

from .replay import ArrivalTick  # noqa: E402


def iter_raw_rows(path: str):
    """Yield ``(timestamp_s, context_tokens, generated_tokens)`` from an Azure
    LLM CSV, one row at a time (streaming; never materializes the file)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        validate_columns(reader.fieldnames, REQUIRED_COLUMNS, DATASET_NAME)
        for row in reader:
            ts = parse_timestamp_s(row.get(COL_TIMESTAMP))
            yield (ts, max(0, _to_int(row.get(COL_CONTEXT))),
                   max(0, _to_int(row.get(COL_GENERATED))))


def _fast_iter_rows(path: str):
    """Fast streaming row reader for the week-long files (~tens of millions of
    rows). Uses ``csv.reader`` + a minute-prefix epoch cache so ``strptime`` is
    called once per minute, not once per row. Yields ``(epoch_s, ctx, gen)``."""
    minute_cache: dict = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        validate_columns(header, REQUIRED_COLUMNS, DATASET_NAME)
        i_ts = header.index(COL_TIMESTAMP)
        i_c = header.index(COL_CONTEXT)
        i_g = header.index(COL_GENERATED)
        for row in reader:
            if len(row) <= i_g:
                continue
            ts_raw = row[i_ts]
            mkey = ts_raw[:16]                       # "YYYY-MM-DD HH:MM"
            base = minute_cache.get(mkey)
            if base is None:
                base = datetime.strptime(mkey, "%Y-%m-%d %H:%M").replace(
                    tzinfo=timezone.utc).timestamp()
                minute_cache[mkey] = base
            # "SS.ffffff+00:00" (2024) or "SS.fffffff" (2023). Strip any tz
            # offset (Azure 2024 carries +00:00 = UTC, matching the minute cache).
            sec_part = ts_raw[17:]
            if sec_part.endswith("Z"):
                sec_part = sec_part[:-1]
            elif "+" in sec_part:
                sec_part = sec_part.split("+", 1)[0]
            elif "-" in sec_part:                     # negative tz offset
                sec_part = sec_part.split("-", 1)[0]
            if "." in sec_part:
                s, frac = sec_part.split(".", 1)
                sec = int(s) + int(frac) / (10 ** len(frac))
            else:
                sec = float(sec_part) if sec_part else 0.0
            yield (base + sec, _to_int(row[i_c]), _to_int(row[i_g]))


def _first_timestamp(path: str) -> Optional[float]:
    for ts, _p, _o in _fast_iter_rows(path):
        return ts
    return None


def _hist_percentile(hist: dict, pct: float) -> Optional[float]:
    """Exact nearest-rank percentile from an integer-value→count histogram."""
    total = sum(hist.values())
    if total == 0:
        return None
    rank = max(1, _math.ceil((pct / 100.0) * total))
    cum = 0
    for val in sorted(hist):
        cum += hist[val]
        if cum >= rank:
            return float(val)
    return float(max(hist))


def _series_stats(values: list) -> dict:
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    ordered = sorted(values)

    def pct(q):
        return float(ordered[max(0, min(n - 1, _math.ceil(q / 100.0 * n) - 1))])
    var = sum((v - mean) ** 2 for v in values) / n
    std = _math.sqrt(var)
    return {
        "mean": round(mean, 6), "p95": round(pct(95), 6), "p99": round(pct(99), 6),
        "max": round(max(values), 6),
        "peak_over_mean": round((max(values) / mean), 4) if mean else None,
        "p99_over_mean": round((pct(99) / mean), 4) if mean else None,
        "coefficient_of_variation": round((std / mean), 4) if mean else None,
    }


def stream_week_aggregate(paths: dict, *, tick_seconds: float = 60.0,
                          bin_seconds: float = 60.0,
                          count_failures_as_arrivals: bool = True) -> dict:
    """Single-pass streaming aggregation of the Azure 2024 week-long file(s).

    ``paths`` maps a workload variant (``conv``/``code``) → CSV path. Returns a
    dict with ``arrival_ticks`` (list[ArrivalTick], combined multi-service
    demand keyed to one absolute timeline) and ``summary`` (row count, time
    range, token percentiles, per-minute RPS + burstiness, day/night and
    weekday/weekend cycles, variant distribution, missing fields).

    Memory is O(n_ticks + distinct_token_values), independent of row count.
    """
    paths = {v: p for v, p in paths.items() if p and os.path.exists(p)}
    if not paths:
        raise FileNotFoundError("no Azure 2024 CSV paths exist")

    firsts = [t for t in (_first_timestamp(p) for p in paths.values()) if t is not None]
    if not firsts:
        raise ValueError("no rows in Azure 2024 files")
    t0 = min(firsts)

    per_tick: dict = {}
    per_minute: dict = {}
    prompt_hist: dict = {}
    output_hist: dict = {}
    total_hist: dict = {}
    variant_counts: dict = {}
    n = 0
    failures = 0
    t_max = t0
    out_of_order = 0
    prev_ts = None

    for variant, path in paths.items():
        prev_ts = None
        for ts, prompt, output in _fast_iter_rows(path):
            n += 1
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            if ts > t_max:
                t_max = ts
            if prev_ts is not None and ts < prev_ts:
                out_of_order += 1
            prev_ts = ts
            idx = int((ts - t0) / tick_seconds)
            if idx < 0:
                idx = 0
            d = per_tick.get(idx)
            if d is None:
                d = {"count": 0, "prompt_sum": 0, "output_sum": 0, "served": 0,
                     "served_prompt": 0, "served_output": 0, "failures": 0,
                     "variant": {}}
                per_tick[idx] = d
            d["count"] += 1
            d["prompt_sum"] += prompt
            d["output_sum"] += output
            d["variant"][variant] = d["variant"].get(variant, 0) + 1
            is_fail = output == 0
            if is_fail:
                d["failures"] += 1
                failures += 1
            else:
                d["served"] += 1
                d["served_prompt"] += prompt
                d["served_output"] += output
            midx = int((ts - t0) / bin_seconds)
            per_minute[midx] = per_minute.get(midx, 0) + 1
            prompt_hist[prompt] = prompt_hist.get(prompt, 0) + 1
            output_hist[output] = output_hist.get(output, 0) + 1
            tot = prompt + output
            total_hist[tot] = total_hist.get(tot, 0) + 1

    n_ticks = int((t_max - t0) / tick_seconds) + 1
    ticks: list = []
    for i in range(n_ticks):
        start = t0 + i * tick_seconds
        d = per_tick.get(i)
        if d is None:
            ticks.append(ArrivalTick(
                tick_index=i, start_s=start, end_s=start + tick_seconds,
                duration_s=tick_seconds, request_count=0, arrival_rate_rps=0.0,
                prompt_tokens_mean=0.0, output_tokens_mean=0.0,
                total_prompt_tokens=0, total_output_tokens=0, failures=0,
                distinct_cache_keys=0, reuse_fraction=0.0,
                model_mix={}, log_type_mix={}))
            continue
        served = d["served"] or d["count"]
        prompt_mean = (d["served_prompt"] or d["prompt_sum"]) / served
        output_mean = (d["served_output"] or d["output_sum"]) / served
        arrivals = d["count"] if count_failures_as_arrivals else d["served"]
        ticks.append(ArrivalTick(
            tick_index=i, start_s=start, end_s=start + tick_seconds,
            duration_s=tick_seconds, request_count=d["count"],
            arrival_rate_rps=arrivals / tick_seconds,
            prompt_tokens_mean=prompt_mean, output_tokens_mean=output_mean,
            total_prompt_tokens=d["prompt_sum"], total_output_tokens=d["output_sum"],
            failures=d["failures"], distinct_cache_keys=0, reuse_fraction=0.0,
            model_mix={DEFAULT_MODEL: d["count"]},   # no model id → single label
            log_type_mix=dict(sorted(d["variant"].items()))))

    summary = _build_week_summary(
        t0=t0, t_max=t_max, n=n, failures=failures, out_of_order=out_of_order,
        per_minute=per_minute, bin_seconds=bin_seconds,
        prompt_hist=prompt_hist, output_hist=output_hist, total_hist=total_hist,
        variant_counts=variant_counts, n_ticks=n_ticks, tick_seconds=tick_seconds)
    return {"arrival_ticks": ticks, "summary": summary, "t0": t0}


def _build_week_summary(*, t0, t_max, n, failures, out_of_order, per_minute,
                        bin_seconds, prompt_hist, output_hist, total_hist,
                        variant_counts, n_ticks, tick_seconds) -> dict:
    duration_s = t_max - t0
    n_minutes = int(duration_s / bin_seconds) + 1
    rps_per_min = [per_minute.get(m, 0) / bin_seconds for m in range(n_minutes)]

    # day/night + weekday/weekend cycles (wall-clock UTC from absolute t0)
    day_counts, night_counts = [], []
    weekday_counts, weekend_counts = [], []
    for m in range(n_minutes):
        wall = _dt.datetime.fromtimestamp(t0 + m * bin_seconds, tz=_dt.timezone.utc)
        c = per_minute.get(m, 0)
        (day_counts if 8 <= wall.hour < 20 else night_counts).append(c)
        (weekend_counts if wall.weekday() >= 5 else weekday_counts).append(c)

    def _avg_rps(counts):
        return round((sum(counts) / len(counts) / bin_seconds), 6) if counts else None

    return {
        "dataset": DATASET_NAME_2024,
        "row_count": n,
        "failure_count": failures,
        "failure_rate_pct": round(failures / n * 100.0, 6) if n else 0.0,
        "out_of_order_rows": out_of_order,
        "time_start_utc": _dt.datetime.fromtimestamp(
            t0, tz=_dt.timezone.utc).isoformat(),
        "time_end_utc": _dt.datetime.fromtimestamp(
            t_max, tz=_dt.timezone.utc).isoformat(),
        "duration_s": round(duration_s, 2),
        "duration_hours": round(duration_s / 3600.0, 3),
        "duration_days": round(duration_s / 86400.0, 3),
        "n_ticks": n_ticks,
        "tick_seconds": tick_seconds,
        "variant_distribution": dict(sorted(variant_counts.items())),
        "prompt_tokens": {"p50": _hist_percentile(prompt_hist, 50),
                          "p95": _hist_percentile(prompt_hist, 95),
                          "p99": _hist_percentile(prompt_hist, 99),
                          "max": float(max(prompt_hist)) if prompt_hist else None},
        "output_tokens": {"p50": _hist_percentile(output_hist, 50),
                         "p95": _hist_percentile(output_hist, 95),
                         "p99": _hist_percentile(output_hist, 99),
                         "max": float(max(output_hist)) if output_hist else None},
        "total_tokens": {"p50": _hist_percentile(total_hist, 50),
                        "p95": _hist_percentile(total_hist, 95),
                        "p99": _hist_percentile(total_hist, 99),
                        "max": float(max(total_hist)) if total_hist else None},
        "rps_per_minute": _series_stats(rps_per_min),
        "day_mean_rps": _avg_rps(day_counts),
        "night_mean_rps": _avg_rps(night_counts),
        "weekday_mean_rps": _avg_rps(weekday_counts),
        "weekend_mean_rps": _avg_rps(weekend_counts),
        "missing_fields": ["model/service id", "session/conversation id",
                          "cache/prefix key", "latency/TTFT/elapsed",
                          "explicit failure flag (derived: GeneratedTokens==0)"],
        "citation": AZURE_2024_CITATION,
    }
