#!/usr/bin/env python3
"""Ingest an Azure public LLM inference trace into Aurelius's normalized form.

Downloads the trace if a URL is available and the raw file is missing,
validates the 3-column schema, normalizes rows into ``NormalizedLLMRequest``
(``aurelius/traces/schema.py``), writes a processed trace, and prints stats.

Honest scope (see docs/AZURE_LLM_BACKTEST_RESULTS.md):
- Azure public LLM inference traces are a public dataset, NOT customer telemetry.
- Schema is ``TIMESTAMP,ContextTokens,GeneratedTokens`` only. There is **no**
  model id, request id, session id, prefix info, or latency/TTFT/elapsed column.
- ``session_id`` / ``cache_affinity_key`` are None (no real cache affinity), and
  ``elapsed_s`` is None → this is a **token-demand and arrival replay, not a
  measured-latency replay**.

Examples
--------
    python scripts/ingest_azure_llm.py                       # conv 2023 (downloads)
    python scripts/ingest_azure_llm.py --workload code
    python scripts/ingest_azure_llm.py --source-path some_local_azure.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

DEFAULT_PROCESSED = "data/external/azure_llm/processed/azure_llm_normalized.json"


def _raw_path_for(workload: str) -> str:
    return f"data/external/azure_llm/raw/AzureLLMInferenceTrace_{workload}.csv"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[ingest] downloading {url}\n         -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (documented public URL)
    print(f"[ingest] downloaded {os.path.getsize(dest):,} bytes")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest an Azure LLM inference trace.")
    p.add_argument("--workload", choices=["conv", "code"], default="conv",
                   help="Azure LLM workload variant (selects default URL/paths)")
    p.add_argument("--source-url", default=None,
                   help="override download URL (default: 2023 file for --workload)")
    p.add_argument("--source-path", default=None,
                   help="use a local CSV instead of downloading")
    p.add_argument("--raw-path", default=None)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--scale-rps", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    if args.source_path:
        path = args.source_path
    else:
        path = args.raw_path or _raw_path_for(args.workload)
        if not os.path.exists(path):
            if args.no_download:
                print(f"[ingest] raw file missing and --no-download set: {path}",
                      file=sys.stderr)
                return 2
            url = args.source_url or azure_llm.SOURCE_URLS[args.workload]
            _download(url, path)

    try:
        requests = azure_llm.load_csv(
            path,
            variant=args.workload,
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

    summary = azure_llm.summarize(requests)

    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    with open(args.processed_path, "w") as fh:
        json.dump(
            {
                "dataset": "azure_llm",
                "workload": args.workload,
                "source": args.source_path or path,
                "filters": {
                    "sample_size": args.sample_size, "start_s": args.start_s,
                    "duration_s": args.duration_s,
                    "include_failures": args.include_failures,
                    "scale_rps": args.scale_rps, "seed": args.seed,
                },
                "summary": summary.to_dict(),
                "requests": [r.to_dict() for r in requests],
            },
            fh,
        )

    _print_summary(summary, args.workload, args.processed_path)
    return 0


def _print_summary(s, workload: str, processed_path: str) -> None:
    print("\n=== Azure LLM ingestion summary ===")
    print(f"workload variant     : {workload}")
    print(f"rows ingested        : {s.row_count:,}")
    print(f"time range (s)       : {s.time_start_s:.3f} -> {s.time_end_s:.3f}")
    print(f"duration (s)         : {s.duration_s:.1f}  ({s.duration_s/3600.0:.3f} h)")
    print("\n--- available fields ---")
    print("  timestamps         : yes (absolute, sub-second)")
    print("  prompt/input tokens: yes (ContextTokens)")
    print("  output tokens      : yes (GeneratedTokens)")
    print("--- MISSING fields (stated explicitly) ---")
    print("  model/service id   : NO  -> model set to 'azure-llm'")
    print("  request/session id : NO  -> session_id = None")
    print(f"  cache/prefix info  : NO  -> cache_affinity_key = None "
          f"(has_cache_affinity={s.has_cache_affinity})")
    print("  latency/TTFT/elapsed: NO -> elapsed_s = None "
          "(token-demand replay, NOT measured-latency replay)")
    print("  failure column     : NO  -> failure only if GeneratedTokens == 0")
    print("\nmodel/service distribution:")
    for k, v in s.model_distribution.items():
        print(f"    {k:<24} {v:,}")
    print("workload (log_type) distribution:")
    for k, v in s.log_type_distribution.items():
        print(f"    {k:<24} {v:,}")
    print(f"\nfailure rate (%)     : {s.failure_rate_pct:.4f}")
    print("token percentiles    :  p50 / p95 / p99")
    print(f"    prompt/input      {s.prompt_tokens_p50:.0f} / {s.prompt_tokens_p95:.0f} / {s.prompt_tokens_p99:.0f}")
    print(f"    output            {s.output_tokens_p50:.0f} / {s.output_tokens_p95:.0f} / {s.output_tokens_p99:.0f}")
    print(f"    total             {s.total_tokens_p50:.0f} / {s.total_tokens_p95:.0f} / {s.total_tokens_p99:.0f}")
    print("\nrequest-rate (per-minute bins):  mean / p95 / max")
    print(f"    rps               {s.rps_mean_per_min:.4f} / {s.rps_p95_per_min:.4f} / {s.rps_max_per_min:.4f}")
    print(f"\nprocessed trace -> {processed_path}")


if __name__ == "__main__":
    raise SystemExit(main())
