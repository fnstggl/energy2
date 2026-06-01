#!/usr/bin/env python3
"""Regenerate (bounded) SwissAI bucket-reuse analysis samples.

The committed repo only ships 5-row fixtures for each SwissAI
bucket_reuse config. This script re-downloads a bounded head per config
(default 10 MiB each), flattens to per-request rows, and writes the
analysis_sample.jsonl into the gitignored processed/ directory so the
cache-prefix forecaster can train on the same rows that
``docs/HF_DATASET_REGISTRY.md`` describes.

This script does NOT modify schemas or summary.json — it ONLY writes the
gitignored analysis_sample.jsonl. It is safe to re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    # (config_name, raw_file_name)
    ("qwen3_32b_bucket_reuse", "qwen3-32b-bucket-reuse.jsonl"),
    ("qwen380b_instruct_bucket_reuse", "qwen380b_instruct_bucket-reuse.jsonl"),
    ("qwen380b_thinking_bucket_reuse", "qwen380b_thinking_bucket-reuse.jsonl"),
    ("llama3_70b_bucket_reuse", "llama3-70b_bucket-reuse.jsonl"),
    ("apertus_70b_bucket_reuse", "apertus-70b-bucket-reuse.jsonl"),
]

DATASET_DIR = REPO_ROOT / "data" / "external" / "hf" / (
    "eth-easl__swissai-serving-trace"
)
RAW_DIR = DATASET_DIR / "raw"


def _bounded_download(url: str, dest: Path, *, max_bytes: int) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "aurelius-swissai-regen/1.0",
        "Range": f"bytes=0-{int(max_bytes - 1)}",
    }
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    written = 0
    err = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    remaining = max_bytes - written
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        out.write(chunk[:remaining])
                        written += remaining
                        break
                    out.write(chunk)
                    written += len(chunk)
    except urllib.error.HTTPError as e:
        err = f"HTTPError {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        err = f"URLError: {e.reason}"
    except Exception as e:  # pragma: no cover
        err = f"{type(e).__name__}: {e}"
    return {
        "url": url, "dest": str(dest), "downloaded_bytes": written,
        "max_bytes": max_bytes, "error": err,
    }


def _normalize_row(raw: dict, *, model_id: str) -> dict:
    """Match the canonical schema in
    data/external/hf/eth-easl__swissai-serving-trace/<config>/processed/
    summary.json."""
    rid = raw.get("id") or ""
    reused = raw.get("reused_buckets") or 0
    total = raw.get("total_buckets") or 0
    reuse_pct = raw.get("reuse_percentage")
    bucket_ids = raw.get("bucket_ids") or []
    if not isinstance(bucket_ids, list):
        bucket_ids = []
    bh = hashlib.blake2b(digest_size=8)
    for b in bucket_ids:
        bh.update(str(b).encode("utf-8"))
        bh.update(b",")
    bucket_ids_hash = bh.hexdigest()
    if bucket_ids:
        head = bucket_ids[:5]
        tail_count = max(0, len(bucket_ids) - 5)
        sample = ",".join(str(x) for x in head)
        if tail_count > 0:
            sample = f"{sample},...(+{tail_count})"
    else:
        sample = ""
    return {
        "request_id": rid,
        "created_at_iso": raw.get("created_at"),
        "reuse_percentage": float(reuse_pct) if reuse_pct is not None else None,
        "reused_bucket_count": int(reused),
        "bucket_count": int(total),
        "bucket_ids_hash": bucket_ids_hash,
        "bucket_ids_sample": sample,
        "model_id": model_id,
    }


def _safe_jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and v != v:
            continue
        out[k] = v
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-bytes-per-config", type=int,
                   default=10 * 1024 * 1024)
    p.add_argument("--configs", nargs="*", default=None)
    args = p.parse_args(argv)

    chosen = {c[0] for c in CONFIGS}
    if args.configs:
        chosen = set(args.configs)

    out_summary: list[dict] = []
    for config_name, raw_file in CONFIGS:
        if config_name not in chosen:
            continue
        model_id = config_name.rsplit("_bucket_reuse", 1)[0]
        url = (
            "https://huggingface.co/datasets/eth-easl/"
            f"swissai-serving-trace/resolve/main/{raw_file}"
        )
        raw_dest = RAW_DIR / raw_file
        t0 = time.monotonic()
        if raw_dest.exists() and raw_dest.stat().st_size >= args.max_bytes_per_config:
            manifest = {"reused_existing": True, "downloaded_bytes":
                        raw_dest.stat().st_size, "error": None,
                        "url": url, "dest": str(raw_dest)}
        else:
            print(f"[swissai] download {config_name} ...")
            manifest = _bounded_download(url, raw_dest,
                                         max_bytes=args.max_bytes_per_config)
        if manifest.get("error"):
            print(f"  error: {manifest['error']}")
            out_summary.append({
                "config_name": config_name, "manifest": manifest,
                "rows_written": 0,
            })
            continue
        # Normalize + write analysis_sample.jsonl (gitignored)
        out_dir = DATASET_DIR / config_name / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "analysis_sample.jsonl"
        n = 0
        with open(raw_dest, "rb") as fh:
            data = fh.read()
        last = data.rfind(b"\n")
        if last >= 0:
            data = data[:last]
        with open(out_path, "w") as out:
            for raw_line in data.splitlines():
                if not raw_line.strip():
                    continue
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                row = _normalize_row(raw, model_id=model_id)
                out.write(json.dumps(_safe_jsonable(row), sort_keys=True))
                out.write("\n")
                n += 1
        elapsed = time.monotonic() - t0
        print(f"[swissai] {config_name}: rows={n} bytes_dl="
              f"{manifest['downloaded_bytes']} elapsed={elapsed:.1f}s")
        out_summary.append({
            "config_name": config_name, "manifest": manifest,
            "rows_written": n, "analysis_sample_path": str(out_path),
        })
    print("[swissai] summary:")
    for r in out_summary:
        print(f"  {r['config_name']}: rows={r['rows_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
