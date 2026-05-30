#!/usr/bin/env python3
"""Ingest the Microsoft Philly trace into Aurelius's normalized form.

The Philly trace ships as a single ~1 GB git-LFS tarball (``trace-data.tar.gz``,
~6.6 GB extracted) — too large to bundle, so this script prints exact download
instructions and parses a local extracted copy if present; unit tests + the
canonical demonstration run on ``tests/fixtures/philly_sample/``.

Honest scope (see docs/PHILLY_BACKTEST_RESULTS.md):
- Philly public data is a research dataset, NOT customer telemetry.
- The cluster_job_log has NO GPU model, NO CPU/host-memory request, NO deadline;
  these are None. GPU type is inferred only as a GPU-<mem> label from the
  machine list. goodput_unit = gpu_seconds_work (NOT inference tokens).

Examples
--------
    python scripts/ingest_philly.py                       # fixture (no full download)
    python scripts/ingest_philly.py --source-dir data/external/philly/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import philly  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

RAW_DIR = "data/external/philly/raw"
DEFAULT_PROCESSED = "data/external/philly/processed/philly_normalized.json"
FIX_DIR = "tests/fixtures/philly_sample"


def _instructions() -> None:
    print(
        "\n[ingest] Philly trace is a ~1 GB git-LFS tarball (no per-file HTTP "
        "download). To use the FULL trace:\n"
        "  1. git lfs install\n"
        "  2. git clone https://github.com/msr-fiddle/philly-traces\n"
        "     (or download trace-data.tar.gz via the repo's LFS media URL)\n"
        f"  3. tar -xzf trace-data.tar.gz  -> yields cluster_job_log, "
        "cluster_machine_list, ...\n"
        f"  4. place them under {RAW_DIR}/ and re-run with --source-dir {RAW_DIR}\n"
        "Falling back to the committed fixture for this run.\n",
        file=sys.stderr,
    )


def _resolve(args) -> tuple[str, str, str]:
    src = args.source_dir or RAW_DIR
    job_log = os.path.join(src, philly.JOB_LOG_NAME)
    machines = os.path.join(src, philly.MACHINE_LIST_NAME)
    if os.path.exists(job_log) and os.path.exists(machines):
        return job_log, machines, f"raw:{src}"
    # tolerate a .json / .csv suffix
    for jl in (job_log + ".json", os.path.join(src, "cluster_job_log.json")):
        for ml in (machines + ".csv", os.path.join(src, "cluster_machine_list.csv")):
            if os.path.exists(jl) and os.path.exists(ml):
                return jl, ml, f"raw:{src}"
    _instructions()
    return (os.path.join(FIX_DIR, "cluster_job_log.json"),
            os.path.join(FIX_DIR, "cluster_machine_list.csv"), f"fixture:{FIX_DIR}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest the Microsoft Philly trace.")
    p.add_argument("--source-url", default=philly.TRACE_TARBALL_URL,
                   help="(informational) LFS tarball URL")
    p.add_argument("--source-dir", default=None,
                   help="dir with extracted cluster_job_log + cluster_machine_list")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-failed", default="true", choices=["true", "false"])
    args = p.parse_args(argv)

    job_log, machines, source = _resolve(args)
    include_failed = args.include_failed == "true"
    try:
        jobs = philly.load_jobs(job_log, sample_size=args.sample_size,
                                start_s=args.start_s, duration_s=args.duration_s,
                                include_failed=include_failed, seed=args.seed)
        nodes = philly.load_machines(machines)
        attempts = philly.analyze_attempts(job_log)
    except TraceSchemaError as e:
        print(f"[ingest] SCHEMA ERROR: {e}", file=sys.stderr)
        return 3
    if not jobs:
        print("[ingest] no jobs after filtering", file=sys.stderr)
        return 4

    summary = philly.summarize_jobs(jobs, nodes)
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    with open(args.processed_path, "w") as fh:
        json.dump({
            "dataset": "philly", "source": source,
            "filters": {"sample_size": args.sample_size, "start_s": args.start_s,
                        "duration_s": args.duration_s,
                        "include_failed": include_failed, "seed": args.seed},
            "summary": summary, "attempt_analysis": attempts,
            "jobs": [j.to_dict() for j in jobs],
        }, fh)

    _print_summary(summary, attempts, source, args.processed_path)
    return 0


def _print_summary(s, attempts, source, processed_path) -> None:
    print("\n=== Philly ingestion summary ===")
    print(f"source               : {source}")
    print(f"jobs ingested        : {s['job_count']:,}  "
          f"(GPU jobs {s['gpu_job_count']:,}, distinct users {s['distinct_users']})")
    print(f"time range / duration: {s['duration_s']:.0f}s "
          f"({s['duration_s']/86400.0:.2f} days)")
    print("\nstatus distribution  :")
    for k, v in s["status_distribution"].items():
        print(f"    {k:<10} {v:,}")
    print("gpu_count distribution:")
    for k, v in s["gpu_count_distribution"].items():
        print(f"    gpu={k} -> {v:,}")
    print(f"\njob duration s p50/p95/p99 : {s['job_duration_s_p50']:.0f} / "
          f"{s['job_duration_s_p95']:.0f} / {s['job_duration_s_p99']:.0f}")
    print(f"queue wait s p50/p95/p99   : {s['queue_wait_s_p50']} / "
          f"{s['queue_wait_s_p95']} / {s['queue_wait_s_p99']}  (trace-observed)")
    print("\n--- retry / failure (trace-observed attempt history) ---")
    print(f"  pass/failed/killed : {attempts['passed']} / {attempts['failed']} "
          f"/ {attempts['killed']}")
    print(f"  multi-attempt jobs : {attempts['multi_attempt_jobs']}  "
          f"retries={attempts['total_retries']} "
          f"(rate {attempts['retry_rate_pct']}%)")
    print(f"  wasted GPU-hours   : {attempts['wasted_gpu_hours_from_retries']}")
    if "fleet_gpu_count" in s:
        print(f"\nfleet: {s['fleet_node_count']} machines, {s['fleet_gpu_count']} "
              f"GPUs by model {s['fleet_gpu_by_model']}")
        print(f"GPU demand/capacity ratio  : {s['gpu_demand_to_capacity_ratio']}")
    print(f"\nMISSING fields (explicit)  : {', '.join(s['missing_fields'])}")
    print(f"GPU utilization samples    : {s['gpu_utilization_samples']} "
          f"(cluster_gpu_util CSV not parsed in this PR)")
    print(f"\nprocessed trace -> {processed_path}")


if __name__ == "__main__":
    raise SystemExit(main())
