#!/usr/bin/env python3
"""Bounded fetch of analysis-tier operational + economic sources for the
Economic Overlay Layer.

This script ONLY downloads into ``data/external/hf/<safe>/raw/`` (gitignored)
and ``data/external/economic_overlay/raw/`` (gitignored). It commits nothing.
JSONL / CSV files are line-delimited so an HTTP-Range *head* download of the
first ``--max-bytes`` yields complete leading rows.

Credentials (PJM_API_KEY / ERCOT_* / WATTTIME_* / HF_TOKEN) are read from the
environment only. No value is ever printed or written to a committed file.

Idempotent: skips a target if the raw file already exists with >= the
requested byte budget (within 1 MiB).
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_RAW = REPO_ROOT / "data" / "external" / "hf"
ECON_RAW = REPO_ROOT / "data" / "external" / "economic_overlay" / "raw"

MiB = 1024 * 1024

# (dataset_id, repo_path, safe_dir, out_name, max_bytes)
HF_TARGETS = [
    ("asdwb/cara_latency_prediction", "train.jsonl",
     "asdwb__cara_latency_prediction", "train_flat.head.jsonl", 280 * MiB),
    ("asdwb/cara_latency_prediction", "train_queue_details.jsonl",
     "asdwb__cara_latency_prediction", "train_queue_details.head.jsonl", 170 * MiB),
    ("eth-easl/swissai-serving-trace", "llama3-70b_bucket-reuse.jsonl",
     "eth-easl__swissai-serving-trace", "llama3_70b_bucket_reuse.head.jsonl", 150 * MiB),
    ("eth-easl/swissai-serving-trace", "qwen3-32b-bucket-reuse.jsonl",
     "eth-easl__swissai-serving-trace", "qwen3_32b_bucket_reuse.head.jsonl", 130 * MiB),
    ("Qinghao/AcmeTrace", "data/job_trace/trace_kalos.csv",
     "Qinghao__AcmeTrace", "trace_kalos.csv", 40 * MiB),
    ("Qinghao/AcmeTrace", "data/utilization/ipmi/GPU_AB_Power.csv",
     "Qinghao__AcmeTrace", "GPU_AB_Power.head.csv", 120 * MiB),
    ("Qinghao/AcmeTrace", "data/utilization/kalos/GPU_UTIL.csv",
     "Qinghao__AcmeTrace", "GPU_UTIL.head.csv", 120 * MiB),
    # Optimum CUDA configs — small, full download.
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-unquantized-1xA100.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_unquantized_1xA100.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-unquantized-1xA10.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_unquantized_1xA10.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-unquantized-1xT4.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_unquantized_1xT4.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-awq-1xA10.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_awq_1xA10.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-bnb-1xA100.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_bnb_1xA100.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cuda-gptq-1xA100.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cuda_gptq_1xA100.csv", 40 * MiB),
    ("optimum-benchmark/llm-perf-leaderboard",
     "data/perf-df-pytorch-cpu-unquantized-32vCPU-C7i.csv",
     "optimum-benchmark__llm-perf-leaderboard", "cpu_unquantized_32vCPU_C7i.csv", 40 * MiB),
]


def _hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def fetch_range(url: str, out: Path, max_bytes: int, headers: dict) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size >= max_bytes - MiB:
        return out.stat().st_size
    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes=0-{max_bytes - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
            data = r.read()
            # Trim to last complete line so partial trailing rows are dropped.
            nl = data.rfind(b"\n")
            if nl != -1:
                data = data[: nl + 1]
            f.write(data)
        return out.stat().st_size
    except Exception as e:  # noqa: BLE001
        print(f"  FETCH-FAIL {url.split('/resolve/')[0].split('datasets/')[-1]} "
              f"{type(e).__name__}", file=sys.stderr)
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply every per-file byte budget (for quick tests).")
    args = p.parse_args(argv)
    headers = _hf_headers()
    total = 0
    for ds, repo_path, safe, out_name, max_bytes in HF_TARGETS:
        out = HF_RAW / safe / "raw" / out_name
        url = f"https://huggingface.co/datasets/{ds}/resolve/main/{repo_path}"
        n = fetch_range(url, out, int(max_bytes * args.scale), headers)
        total += n
        print(f"{n / MiB:8.1f} MiB  {safe}/{out_name}")
    print(f"\nTOTAL fetched: {total / MiB:.1f} MiB (all gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
