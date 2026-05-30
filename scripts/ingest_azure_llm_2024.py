#!/usr/bin/env python3
"""Ingest the Azure LLM Inference Dataset **2024** (week-long, multi-service).

Downloads the two week-long ``_1week`` variants (code + conv) from Azure blob
storage if missing, streams them (they are ~0.7–1.1 GB / tens of millions of
rows), validates the discovered schema, and prints the ingestion summary
(Task 2). Writes a processed summary JSON (NOT the raw rows — far too large).

Discovered schema (verified against the ACTUAL 2024 files, NOT assumed from
2023): ``TIMESTAMP,ContextTokens,GeneratedTokens`` — the 2024 TIMESTAMP carries
a ``+00:00`` UTC offset + 6 fractional digits (distinct from the 2023 .NET
7-digit form). No latency/TTFT, no model/service id, no session/cache key.

Dataset is CC-BY; cite DynamoLLM (HPCA 2025), Stojkovic et al.
(arxiv 2408.00741). Token-demand + arrival replay, NOT a measured-latency
replay. Directional simulator evidence — not production savings.

Examples
--------
    python scripts/ingest_azure_llm_2024.py                 # downloads both files
    python scripts/ingest_azure_llm_2024.py --no-download --source-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import azure_llm as az  # noqa: E402

RAW_DIR = "data/external/azure_llm_2024/raw"
PROCESSED_DIR = "data/external/azure_llm_2024/processed"
FILES = {
    "code": "AzureLLMInferenceTrace_code_1week.csv",
    "conv": "AzureLLMInferenceTrace_conv_1week.csv",
}


def _download(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for variant, fname in FILES.items():
        dest = os.path.join(raw_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[ingest] {fname}: present ({os.path.getsize(dest):,} bytes)")
            continue
        url = az.SOURCE_URLS_2024[variant]
        print(f"[ingest] downloading {fname} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310 (documented public dataset)
            print(f"[ingest] {fname}: {os.path.getsize(dest):,} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] WARN could not fetch {fname}: {e}", file=sys.stderr)


def _resolve(raw_dir: str) -> dict:
    return {v: os.path.join(raw_dir, f) for v, f in FILES.items()
            if os.path.exists(os.path.join(raw_dir, f))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest Azure LLM 2024 week-long trace.")
    p.add_argument("--source-dir", default=None)
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--processed-dir", default=PROCESSED_DIR)
    p.add_argument("--tick-seconds", type=float, default=60.0)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    raw_dir = args.source_dir or args.raw_dir
    if not args.source_dir and not args.no_download:
        _download(raw_dir)
    paths = _resolve(raw_dir)
    if not paths:
        print(f"[ingest] no Azure 2024 files under {raw_dir}; expected one of "
              f"{list(FILES.values())}", file=sys.stderr)
        print("[ingest] download URLs:\n  " +
              "\n  ".join(az.SOURCE_URLS_2024.values()), file=sys.stderr)
        return 2

    agg = az.stream_week_aggregate(paths, tick_seconds=args.tick_seconds)
    s = agg["summary"]

    os.makedirs(args.processed_dir, exist_ok=True)
    out = os.path.join(args.processed_dir, "azure_llm_2024_ingestion_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"files_used": list(paths.values()), "summary": s}, fh, indent=2,
                  sort_keys=True, default=str)

    print("\n=== Azure LLM 2024 ingestion summary ===")
    print(f"files used        : {list(paths.values())}")
    print("schema            : TIMESTAMP,ContextTokens,GeneratedTokens (verified)")
    print(f"row count         : {s['row_count']:,}")
    print(f"time range (UTC)  : {s['time_start_utc']} -> {s['time_end_utc']}")
    print(f"duration          : {s['duration_days']} days ({s['duration_hours']} h), "
          f"{s['n_ticks']} ticks @ {s['tick_seconds']}s")
    print(f"variant dist      : {s['variant_distribution']}")
    print(f"failures(0-output): {s['failure_count']} ({s['failure_rate_pct']}%)  "
          f"out_of_order: {s['out_of_order_rows']}")
    print(f"prompt p50/p95/p99: {s['prompt_tokens']['p50']} / {s['prompt_tokens']['p95']} "
          f"/ {s['prompt_tokens']['p99']}  (max {s['prompt_tokens']['max']})")
    print(f"output p50/p95/p99: {s['output_tokens']['p50']} / {s['output_tokens']['p95']} "
          f"/ {s['output_tokens']['p99']}  (max {s['output_tokens']['max']})")
    print(f"total  p50/p95/p99: {s['total_tokens']['p50']} / {s['total_tokens']['p95']} "
          f"/ {s['total_tokens']['p99']}")
    r = s["rps_per_minute"]
    print(f"RPS/min mean/p95/p99/max: {r['mean']} / {r['p95']} / {r['p99']} / {r['max']}")
    print(f"burstiness        : peak/mean {r['peak_over_mean']}  p99/mean "
          f"{r['p99_over_mean']}  CV {r['coefficient_of_variation']}")
    print(f"day/night mean RPS: {s['day_mean_rps']} / {s['night_mean_rps']}")
    print(f"weekday/weekend   : {s['weekday_mean_rps']} / {s['weekend_mean_rps']}")
    print(f"missing fields    : {s['missing_fields']}")
    print(f"citation          : {s['citation']}")
    print(f"\nprocessed summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
