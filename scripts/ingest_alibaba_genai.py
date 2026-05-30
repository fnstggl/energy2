#!/usr/bin/env python3
"""Ingest the Alibaba cluster-trace-v2026-GenAI (GenTD26) multi-layer trace.

Downloads the (small) layer files if missing, discovers + classifies every file,
loads ALL primary telemetry layers (application / middleware / scheduler /
infrastructure), classifies cross-layer linkage quality from the actual data
(no faked joins), and writes a processed normalized trace + summary.

Honest scope (see docs/ALIBABA_GENAI_BACKTEST_RESULTS.md):
- Alibaba public data is a public dataset, NOT customer telemetry.
- The application layer (lora_request_trace.csv) and the metric layers use
  incompatible anonymized time bases and the request layer has no container_ip,
  so application <-> metric layers are NO_JOIN. The metric layers join to each
  other by container_ip. No request->GPU causality is claimed.

Examples
--------
    python scripts/ingest_alibaba_genai.py                 # downloads all layers
    python scripts/ingest_alibaba_genai.py --source-dir data/external/alibaba_genai/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import alibaba_genai as ag  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

RAW_DIR = "data/external/alibaba_genai/raw"
PROCESSED_DIR = "data/external/alibaba_genai/processed"
# files to fetch: the request CSV (plain) + the metric tarballs
_TARBALLS = ["qps", "queue_size_raw_anon", "queue_rt_raw_anon",
             "pipeline_inference_data_anon", "pipeline_update_latency_anon",
             "model_predict_data_anon", "basemodel_update_latency_anon",
             "controlnet_latency_data_anon", "lora_update_latency_anon",
             "pod_gpu_duty_cycle_anon", "pod_gpu_memory_used_bytes_anon",
             "pod_memory_util_anon"]


def _download_all(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    req = os.path.join(raw_dir, ag.REQUEST_FILE)
    if not os.path.exists(req):
        print(f"[ingest] downloading {ag.REQUEST_FILE}")
        urllib.request.urlretrieve(f"{ag._BASE_URL}/{ag.REQUEST_FILE}", req)  # noqa: S310
    for name in _TARBALLS:
        csv_path = os.path.join(raw_dir, f"{name}.csv")
        if os.path.exists(csv_path):
            continue
        tgz = os.path.join(raw_dir, f"{name}.tar.gz")
        try:
            print(f"[ingest] downloading {name}.tar.gz")
            urllib.request.urlretrieve(f"{ag._BASE_URL}/{name}.tar.gz", tgz)  # noqa: S310
            with tarfile.open(tgz) as tf:
                tf.extractall(raw_dir)  # noqa: S202 (documented public dataset)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] WARN could not fetch {name}: {e}", file=sys.stderr)


def _instructions(raw_dir: str) -> None:
    print(
        f"\n[ingest] Manual download (all files are small, < 5 MB compressed):\n"
        f"  Repo: https://github.com/alibaba/clusterdata/tree/master/"
        f"cluster-trace-v2026-GenAI\n"
        f"  Fetch {ag.REQUEST_FILE} + the *.tar.gz metric files, extract the\n"
        f"  CSVs into {raw_dir}/ and re-run with --source-dir {raw_dir}.\n",
        file=sys.stderr)


def _linkage_matrix(layers) -> dict:
    groups = {
        "application": ("app", layers["requests"]),
        "middleware": ("metric", layers["gateway"]),
        "scheduler": ("metric", layers["pipeline"]),
        "infrastructure": ("metric", layers["infra"]),
    }
    names = list(groups)
    matrix = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            if a == b:
                matrix[a][b] = "self"
                continue
            app_pair = "application" in (a, b)
            matrix[a][b] = ag.classify_linkage(
                a, groups[a][1], b, groups[b][1], app_request_layer=app_pair)
    return matrix


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest Alibaba GenAI 2026 trace.")
    p.add_argument("--source-dir", default=None)
    p.add_argument("--source-url", default=ag._BASE_URL, help="(informational)")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--processed-dir", default=PROCESSED_DIR)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-failures", default="false", choices=["true", "false"])
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    raw_dir = args.source_dir or args.raw_dir
    if not args.source_dir and not args.no_download:
        if not os.path.exists(os.path.join(raw_dir, ag.REQUEST_FILE)):
            _download_all(raw_dir)
    if not os.path.exists(os.path.join(raw_dir, ag.REQUEST_FILE)):
        _instructions(raw_dir)
        return 2

    try:
        layers = ag.load_all_layers(raw_dir, request_kwargs=dict(
            sample_size=args.sample_size, start_s=args.start_s,
            duration_s=args.duration_s,
            include_failures=(args.include_failures == "true"), seed=args.seed))
    except TraceSchemaError as e:
        print(f"[ingest] SCHEMA ERROR: {e}", file=sys.stderr)
        return 3

    disc = layers["discovery"]
    summary = ag.summarize(layers["requests"], layers["gateway"],
                           layers["pipeline"], layers["infra"])
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    matrix = _linkage_matrix(layers)

    os.makedirs(args.processed_dir, exist_ok=True)
    out_path = os.path.join(args.processed_dir, "alibaba_genai_normalized.json")
    with open(out_path, "w") as fh:
        json.dump({
            "dataset": "alibaba_genai", "source": raw_dir,
            "discovery": {k: v for k, v in disc.items() if k != "files"},
            "file_classification": disc["files"],
            "linkage_matrix": matrix,
            "cold_start_calibration_s": cold,
            "layer_summary": summary,
            # persist the request layer (drives the replay); metric layers are
            # large telemetry — summarised, not dumped row-by-row.
            "requests": [r.to_dict() for r in layers["requests"]],
        }, fh)

    _print(disc, summary, cold, matrix, out_path)
    return 0


def _print(disc, summary, cold, matrix, out_path) -> None:
    print("\n=== Alibaba GenAI 2026 ingestion summary ===")
    print(f"layers present : {disc['layers_present']}")
    print(f"primary files  : {len(disc['primary_present'])}  "
          f"empty: {disc['empty']}  skipped(non-telemetry): {len(disc['skipped'])}")
    print("\n--- file classification ---")
    for name, e in disc["files"].items():
        print(f"  {name:<38} {e['classification']:<13} {e['layer']:<15} "
              f"{e['status']}" + (f" rows={e['rows']:,}" if "rows" in e else ""))
    app = summary["application"]
    if app:
        print(f"\napplication: {app['request_count']:,} requests, "
              f"{app['distinct_models']} models, failed {app['failed']}, "
              f"lora_frac {app['lora_request_frac']}")
        print(f"  request_type: {app['request_type_distribution']}")
        print(f"  e2e_latency_s p50/p95/p99: {app['e2e_latency_s_p50']} / "
              f"{app['e2e_latency_s_p95']} / {app['e2e_latency_s_p99']}")
    mw = summary["middleware"]
    if mw:
        print(f"middleware: {mw['samples']:,} samples; queue_depth p50/p95 "
              f"{mw['queue_depth_p50']}/{mw['queue_depth_p95']}; waiting_time_s "
              f"p95/p99 {mw['waiting_time_s_p95']}/{mw['waiting_time_s_p99']}")
    sch = summary["scheduler"]
    if sch:
        print(f"scheduler: {sch['events']:,} events; stage p50 (s): "
              f"{ {k: round(v,1) for k,v in sch['stage_duration_s_p50'].items() if v} }")
    inf = summary["infrastructure"]
    if inf:
        print(f"infrastructure: {inf['samples']:,} samples; gpu_util% p50/p95 "
              f"{inf['gpu_util_pct_p50']}/{inf['gpu_util_pct_p95']}; "
              f"container_mem_frac p95 {inf['container_mem_frac_p95']}")
    print(f"\ncold-start calibration (s, pipeline-layer medians): "
          f"{ {k: round(v,1) for k,v in cold.items()} }")
    print("\n--- cross-layer linkage matrix (computed from data; no faked joins) ---")
    names = list(matrix)
    print("  " + " " * 16 + "".join(f"{n[:12]:<14}" for n in names))
    for a in names:
        print(f"  {a:<16}" + "".join(f"{matrix[a][b]:<14}" for b in names))
    print(f"\nprocessed trace -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
