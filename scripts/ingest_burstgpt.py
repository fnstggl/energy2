#!/usr/bin/env python3
"""Ingest the BurstGPT public LLM-serving trace into Aurelius's normalized form.

Downloads ``BurstGPT_1.csv`` if missing, validates the schema, normalizes rows
into ``NormalizedLLMRequest`` (``aurelius/traces/schema.py``), writes a processed
trace, and prints descriptive stats.

Honest scope (see docs/PUBLIC_TRACE_BACKTESTS.md):
- BurstGPT is a public serving trace, NOT customer telemetry.
- The published BurstGPT_1.csv has no Session ID and no Elapsed-time column; the
  cache-affinity key is a model-level prefix-locality PROXY, not a KV hit rate.
- BurstGPT elapsed time (when a file provides it) is end-to-end response time,
  NOT TTFT.

Examples
--------
    python scripts/ingest_burstgpt.py                       # full file
    python scripts/ingest_burstgpt.py --sample-size 50000 --seed 7
    python scripts/ingest_burstgpt.py --start-s 0 --duration-s 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

# Allow running as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import burstgpt  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

DEFAULT_RAW = "data/external/burstgpt/raw/BurstGPT_1.csv"
DEFAULT_PROCESSED = "data/external/burstgpt/processed/burstgpt_normalized.json"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[ingest] downloading {url}\n         -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (documented public URL)
    print(f"[ingest] downloaded {os.path.getsize(dest):,} bytes")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest BurstGPT trace.")
    p.add_argument("--source-url", default=burstgpt.DEFAULT_SOURCE_URL)
    p.add_argument("--raw-path", default=DEFAULT_RAW)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--scale-rps", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download", action="store_true",
                   help="fail instead of downloading if the raw file is missing")
    args = p.parse_args(argv)

    if not os.path.exists(args.raw_path):
        if args.no_download:
            print(f"[ingest] raw file missing and --no-download set: {args.raw_path}",
                  file=sys.stderr)
            return 2
        _download(args.source_url, args.raw_path)

    try:
        requests = burstgpt.load_csv(
            args.raw_path,
            sample_size=args.sample_size,
            start_s=args.start_s,
            duration_s=args.duration_s,
            include_failures=args.include_failures,
            scale_rps=args.scale_rps,
            seed=args.seed,
        )
    except TraceSchemaError as e:
        print(f"[ingest] SCHEMA ERROR: {e}", file=sys.stderr)
        return 3

    if not requests:
        print("[ingest] no rows after filtering", file=sys.stderr)
        return 4

    summary = burstgpt.summarize(requests)

    # Persist processed trace (normalized records) for the backtest to reuse.
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    with open(args.processed_path, "w") as fh:
        json.dump(
            {
                "dataset": "burstgpt",
                "source_url": args.source_url,
                "raw_path": args.raw_path,
                "filters": {
                    "sample_size": args.sample_size,
                    "start_s": args.start_s,
                    "duration_s": args.duration_s,
                    "include_failures": args.include_failures,
                    "scale_rps": args.scale_rps,
                    "seed": args.seed,
                },
                "summary": summary.to_dict(),
                "requests": [r.to_dict() for r in requests],
            },
            fh,
        )

    _print_summary(summary, args.processed_path)
    return 0


def _print_summary(s, processed_path: str) -> None:
    print("\n=== BurstGPT ingestion summary ===")
    print(f"rows ingested        : {s.row_count:,}")
    print(f"time range (s)       : {s.time_start_s:.1f} -> {s.time_end_s:.1f}")
    print(f"duration (s)         : {s.duration_s:.1f}  ({s.duration_s/3600.0:.2f} h)")
    print(f"has Session ID col   : {s.has_session_ids}")
    print(f"has Elapsed-time col : {s.has_elapsed}  (NOTE: elapsed is end-to-end, NOT TTFT)")
    print("\nmodel distribution   :")
    for k, v in s.model_distribution.items():
        print(f"    {k:<24} {v:,}")
    print("log-type distribution:")
    for k, v in s.log_type_distribution.items():
        print(f"    {k:<24} {v:,}")
    print(f"\nfailure rate (%)     : {s.failure_rate_pct:.4f}")
    print("token percentiles    :  p50 / p95 / p99")
    print(f"    prompt_tokens     {s.prompt_tokens_p50:.0f} / {s.prompt_tokens_p95:.0f} / {s.prompt_tokens_p99:.0f}")
    print(f"    output_tokens     {s.output_tokens_p50:.0f} / {s.output_tokens_p95:.0f} / {s.output_tokens_p99:.0f}")
    print(f"    total_tokens      {s.total_tokens_p50:.0f} / {s.total_tokens_p95:.0f} / {s.total_tokens_p99:.0f}")
    print("\nRPS by minute        :")
    print(f"    mean / p95 / max  {s.rps_mean_per_min:.4f} / {s.rps_p95_per_min:.4f} / {s.rps_max_per_min:.4f}")
    print("\nsession / cache-affinity proxy (NOT a measured KV hit rate):")
    print(f"    distinct keys     {s.distinct_cache_keys:,}")
    print(f"    reuse rate (%)    {s.cache_key_reuse_rate_pct:.4f}")
    print(f"    reqs / key (mean) {s.mean_requests_per_cache_key:.4f}")
    print(f"\nprocessed trace -> {processed_path}")


if __name__ == "__main__":
    raise SystemExit(main())
