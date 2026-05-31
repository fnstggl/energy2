#!/usr/bin/env python3
"""Ingest the MIT Supercloud Dataset (Samsi et al., HPEC 2021).

Loads the **already-extracted** MIT Supercloud trace files from a
local directory (the ~1 TB compressed archive itself is NOT in the
GitHub repo and is downloaded separately from https://dcc.mit.edu/data
— this script prints the exact download instructions when raw files
are missing).

What it does:

1. Discovers which MIT Supercloud files are present at ``--source-dir``
   (scheduler-log.csv / labelled_jobids.csv / tres-mapping.txt /
   node-data.csv / gpu/<NN>/ / cpu/<NN>/).
2. Parses the TRES integer ↔ resource mapping (``tres-mapping.txt``).
3. Loads the Slurm scheduler log and joins the labelled-DNN mapping.
4. Optionally loads per-job GPU utilization CSVs (capped by
   ``--max-util-files`` so the ingest stays fast on a sample).
5. Loads the 5-min node-data snapshots.
6. Computes the **join quality matrix** (jobs ↔ labels, jobs ↔ GPU
   utilization, jobs ↔ node snapshots) — no fake joins.
7. Emits a small JSON summary to
   ``data/external/mit_supercloud/processed/`` (committed; large
   normalized JSON is gitignored).

Honesty rules:

- Missing telemetry is preserved as ``None`` — never zero-filled.
- Per-job utilization is reported only when the file-name join holds
  (the MIT GPU CSVs are named ``<job_id>.csv``, so the join is exact).
- Node ↔ job overlap is reported as **medium** confidence (5-min
  granular snapshots).
- No production mutation. No ML training. No new datasets pulled
  beyond MIT Supercloud. No robust energy engine change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import mit_supercloud as mit  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(REPO_ROOT, "data", "external",
                            "mit_supercloud", "raw")
DEFAULT_PROCESSED = os.path.join(REPO_ROOT, "data", "external",
                                  "mit_supercloud", "processed")
DEFAULT_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                                "mit_supercloud_sample")


DOWNLOAD_INSTRUCTIONS = textwrap.dedent(f"""
================================================================================
MIT Supercloud raw archives ARE NOT in the GitHub repo.
================================================================================

The MIT-AI-Accelerator/MIT-Supercloud-Dataset GitHub repo
({mit.REPO_URL}) ships only notebooks + helper scripts. The actual
~1 TB compressed dataset (scheduler-log.csv / labelled_jobids.csv /
node-data.csv / gpu/ / cpu/ time series) is hosted at:

    {mit.DCC_DATA_URL}

To run this ingestion against the real trace:

  1. Visit {mit.DCC_DATA_URL} and follow the download instructions
     for the MIT Supercloud Dataset (registration / agreement may be
     required).
  2. Extract the archives so the following files exist under
     ``--source-dir`` (default: {DEFAULT_RAW}):
       * scheduler-log.csv         (Slurm accounting; ~460,497 jobs)
       * labelled_jobids.csv       (3,425 labelled DNN jobs)
       * tres-mapping.txt          (TRES integer ↔ resource table)
       * node-data.csv             (5-min per-node snapshots)
       * gpu/<NN>/<job_id>.csv     (nvidia-smi 100-ms time series)
       * cpu/<NN>/<job_id>.csv     (CPU 10-s time series)
  3. Re-run this script with ``--source-dir <extracted_path>``.

For local CI / unit tests, this script falls back to the small
synthetic fixture at ``{DEFAULT_FIXTURE}`` (NOT a copy of the real
dataset — see that directory's README.md).

Citation: Samsi et al., HPEC 2021, {mit.PAPER_URL}
================================================================================
""").strip()


def _print(s, **kw):
    print(s, file=sys.stdout, flush=True, **kw)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-url", default=mit.REPO_URL)
    p.add_argument("--raw-dir", "--source-dir", dest="source_dir",
                   default=None,
                   help=(f"directory containing the extracted MIT Supercloud "
                         f"files; default {DEFAULT_RAW} when present, else "
                         f"the small synthetic fixture at {DEFAULT_FIXTURE}"))
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
    p.add_argument("--sample-size", type=int, default=None,
                   help="random-sample N jobs from the scheduler log "
                        "(seeded); default: keep all")
    p.add_argument("--gpu-jobs-only", action="store_true",
                   help="drop CPU-only jobs (keep jobs that requested "
                        "any GPU via tres_req)")
    p.add_argument("--labelled-only", action="store_true",
                   help="keep only jobs that appear in "
                        "labelled_jobids.csv")
    p.add_argument("--include-utilization", default="true",
                   choices=("true", "false"),
                   help="load per-job GPU utilization CSVs from gpu/")
    p.add_argument("--max-util-files", type=int, default=200,
                   help="cap on per-job GPU CSV files loaded; keeps the "
                        "ingest fast on a sample (default 200)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-only-instructions", action="store_true",
                   help="just print the download instructions and exit")
    p.add_argument("--summary-json", default=None,
                   help="output JSON path; default writes under "
                        "--processed-dir")
    args = p.parse_args(argv)

    if args.print_only_instructions:
        _print(DOWNLOAD_INSTRUCTIONS)
        return 0

    source_dir = args.source_dir
    using_fixture = False
    if source_dir is None:
        if os.path.isdir(DEFAULT_RAW) and any(
                os.path.exists(os.path.join(DEFAULT_RAW, f))
                for f in mit.SCHEDULER_LOG_FILES):
            source_dir = DEFAULT_RAW
        else:
            source_dir = DEFAULT_FIXTURE
            using_fixture = True
            _print(DOWNLOAD_INSTRUCTIONS)
            _print("")
            _print(f"[ingest] raw files not present at {DEFAULT_RAW}; "
                   f"falling back to synthetic fixture at {source_dir}")

    if not os.path.isdir(source_dir):
        _print(DOWNLOAD_INSTRUCTIONS, **{})
        _print(f"\n[ingest] --source-dir does not exist: {source_dir}",
               **{})
        return 4

    include_util = (args.include_utilization == "true")
    layers = mit.load_all_layers(
        source_dir,
        include_utilization=include_util,
        max_util_files=args.max_util_files,
        sample_size=args.sample_size,
        gpu_jobs_only=args.gpu_jobs_only,
        labelled_only=args.labelled_only,
        seed=args.seed)

    summary = mit.summarize_jobs(layers["jobs"])
    joins = mit.compute_join_quality(
        layers["jobs"], labels_by_jobid=layers["labels_by_jobid"],
        gpu_samples=layers["gpu_samples"] if include_util else None,
        node_samples=layers["node_samples"])

    payload = {
        "source": ("fixture" if using_fixture else "raw"),
        "source_dir": os.path.relpath(source_dir, REPO_ROOT),
        "repo_url": args.repo_url,
        "discovery": layers["discovery"],
        "trace_summary": summary,
        "join_quality": joins,
        "filters": {
            "sample_size": args.sample_size,
            "gpu_jobs_only": args.gpu_jobs_only,
            "labelled_only": args.labelled_only,
            "include_utilization": include_util,
            "max_util_files": args.max_util_files,
            "seed": args.seed,
        },
    }
    os.makedirs(args.processed_dir, exist_ok=True)
    summary_path = (args.summary_json
                    or os.path.join(args.processed_dir,
                                     "mit_supercloud_ingest_summary.json"))
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)),
                 exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)

    _print(f"[ingest] source: {source_dir}")
    _print(f"[ingest] jobs={summary['job_count']:,} "
           f"gpu_jobs={summary['gpu_job_count']:,} "
           f"labelled={summary['labelled_job_count']:,}")
    for j in joins["joins"]:
        _print(f"[ingest]   join {j['join_name']:18s} kind="
               f"{j['join_kind']:22s} matched={j['matched_right']:>6} / "
               f"{j['right_total']:<6} confidence={j['confidence']}")
    _print(f"[ingest] queue wait p95 / p99 (s): "
           f"{summary['queue_wait_s_p95']} / {summary['queue_wait_s_p99']}")
    _print(f"[ingest] duration   p95 / p99 (s): "
           f"{summary['duration_s_p95']} / {summary['duration_s_p99']}")
    _print(f"[ingest] gpu_count distribution: "
           f"{summary['gpu_count_distribution']}")
    _print(f"[ingest] workload labels: "
           f"{summary['workload_label_distribution']}")
    _print(f"[ingest] summary JSON -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
