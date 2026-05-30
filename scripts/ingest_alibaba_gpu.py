#!/usr/bin/env python3
"""Ingest the Alibaba cluster-trace-gpu-v2023 trace into Aurelius's normalized form.

Downloads the pod + node CSVs if missing (or prints exact download instructions),
validates the schema, normalizes pods → ``NormalizedGPUJob`` and the node list →
``GPUNode`` fleet inventory, saves a processed trace, and prints summary stats.

Honest scope (see docs/ALIBABA_GPU_BACKTEST_RESULTS.md):
- Alibaba public GPU cluster data is a public dataset, NOT customer telemetry.
- v2023 has NO GPU utilization time-series, NO GPU-memory column, NO per-pod node
  placement in the default pod list, NO deadline/user columns. These are None /
  empty and stated explicitly — not invented.

Examples
--------
    python scripts/ingest_alibaba_gpu.py                  # downloads default pod + gpu nodes
    python scripts/ingest_alibaba_gpu.py --include-failed
    python scripts/ingest_alibaba_gpu.py --source-dir /path/to/cluster-trace-gpu-v2023/csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import alibaba_gpu as az  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

RAW_DIR = "data/external/alibaba_gpu/raw"
DEFAULT_PROCESSED = "data/external/alibaba_gpu/processed/alibaba_gpu_normalized.json"
POD_FILE = "openb_pod_list_default.csv"
NODE_FILE = "openb_node_list_gpu_node.csv"


def _download(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"[ingest] downloading {url}\n         -> {dest}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (documented public URL)
        print(f"[ingest] downloaded {os.path.getsize(dest):,} bytes")
        return True
    except Exception as e:  # network may be unavailable in some environments
        print(f"[ingest] download failed ({e}).", file=sys.stderr)
        return False


def _instructions() -> None:
    print(
        "\n[ingest] Could not download automatically. Manual download:\n"
        "  Repo: https://github.com/alibaba/clusterdata\n"
        "  Path: cluster-trace-gpu-v2023/csv/\n"
        f"    - {POD_FILE}   (pod/job list)\n"
        f"    - {NODE_FILE}  (GPU node inventory)\n"
        f"  Place them under {RAW_DIR}/ and re-run with --source-dir {RAW_DIR}.\n",
        file=sys.stderr,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest Alibaba GPU v2023 trace.")
    p.add_argument("--source-url", default=None, help="override pod CSV URL")
    p.add_argument("--source-dir", default=None,
                   help="local dir containing the CSVs (skips download)")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--pod-file", default=POD_FILE)
    p.add_argument("--node-file", default=NODE_FILE)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-failed", default="true",
                   choices=["true", "false"])
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    raw_dir = args.source_dir or args.raw_dir
    pod_path = os.path.join(raw_dir, args.pod_file)
    node_path = os.path.join(raw_dir, args.node_file)

    if not args.source_dir:
        if not os.path.exists(pod_path):
            if args.no_download or not _download(
                    args.source_url or az.DEFAULT_POD_URL, pod_path):
                _instructions()
                return 2
        if not os.path.exists(node_path):
            if args.no_download or not _download(az.DEFAULT_NODE_URL, node_path):
                _instructions()
                return 2

    if not os.path.exists(pod_path) or not os.path.exists(node_path):
        print(f"[ingest] missing CSVs in {raw_dir}", file=sys.stderr)
        _instructions()
        return 2

    include_failed = args.include_failed == "true"
    try:
        jobs = az.load_jobs(pod_path, sample_size=args.sample_size,
                            start_s=args.start_s, duration_s=args.duration_s,
                            include_failed=include_failed, seed=args.seed)
        nodes = az.load_nodes(node_path)
    except TraceSchemaError as e:
        print(f"[ingest] SCHEMA ERROR: {e}", file=sys.stderr)
        return 3

    if not jobs:
        print("[ingest] no jobs after filtering", file=sys.stderr)
        return 4

    summary = az.summarize_jobs(jobs, nodes)
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    with open(args.processed_path, "w") as fh:
        json.dump({
            "dataset": "alibaba_gpu",
            "pod_source": pod_path, "node_source": node_path,
            "filters": {"sample_size": args.sample_size, "start_s": args.start_s,
                        "duration_s": args.duration_s,
                        "include_failed": include_failed, "seed": args.seed},
            "summary": summary,
            "jobs": [j.to_dict() for j in jobs],
        }, fh)

    _print_summary(summary, args.processed_path)
    return 0


def _print_summary(s: dict, processed_path: str) -> None:
    print("\n=== Alibaba GPU v2023 ingestion summary ===")
    print(f"jobs ingested        : {s['job_count']:,}  "
          f"(GPU jobs {s['gpu_job_count']:,}, CPU-only {s['cpu_only_count']:,})")
    print(f"failed jobs          : {s['failed_jobs']:,}")
    print(f"time range / duration: {s['duration_s']:.0f}s "
          f"({s['duration_s']/86400.0:.1f} days)")
    print("\nstatus distribution  :")
    for k, v in s["status_distribution"].items():
        print(f"    {k:<12} {v:,}")
    print("num_gpu distribution :")
    for k, v in s["gpu_count_distribution"].items():
        print(f"    num_gpu={k} -> {v:,}")
    print("gpu_type distribution:")
    for k, v in s["gpu_type_distribution"].items():
        print(f"    {k:<10} {v:,}")
    print(f"\njob duration s p50/p95/p99 : {s['job_duration_s_p50']:.0f} / "
          f"{s['job_duration_s_p95']:.0f} / {s['job_duration_s_p99']:.0f}")
    print(f"queue wait s p50/p95/p99   : {s['queue_wait_s_p50']} / "
          f"{s['queue_wait_s_p95']} / {s['queue_wait_s_p99']}  (trace-observed)")
    if "fleet_gpu_count" in s:
        print(f"\nfleet: {s['fleet_node_count']:,} GPU nodes, "
              f"{s['fleet_gpu_count']:,} GPUs  by model {s['fleet_gpu_by_model']}")
        print(f"GPU demand/capacity ratio  : {s['gpu_demand_to_capacity_ratio']}")
    print(f"\nGPU utilization samples    : {s['gpu_utilization_samples']} "
          f"(none in v2023)")
    print(f"MISSING fields (explicit)  : {', '.join(s['missing_fields'])}")
    print(f"\nprocessed trace -> {processed_path}")


if __name__ == "__main__":
    raise SystemExit(main())
