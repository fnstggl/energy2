#!/usr/bin/env python3
"""Bounded-ingest the ShareGPT eval-conversation-shape dataset.

ShareGPT is an EVAL-CLASS / BENCHMARK CONVERSATION-SHAPE proxy (see
``docs/PUBLIC_TRACE_BACKTESTS.md`` registry row). It is NOT customer
telemetry and NOT a measured serving trace. The dataset is **bounded
downloaded** via HTTP Range — by default 50 MB — exactly mirroring the
MIT-Supercloud bounded-real-sample precedent (``docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md``).

Outputs:
- ``data/external/sharegpt_aiperf/raw/sg_52k_head.json`` (gitignored)
- ``data/external/sharegpt_aiperf/raw/bounded_download_manifest.json``
- ``data/external/sharegpt_aiperf/processed/sharegpt_aiperf_ingest_summary.json``

No raw text is persisted in the processed summary — only per-record turn
counts + character-count-derived token-estimate proxies. The processed
file is small (≤ ~200 KB), well within the user-spec 100 MB cap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import sharegpt_aiperf  # noqa: E402
from aurelius.traces.eval_schema import EvalWorkloadSchemaError  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(
    REPO_ROOT, "data", "external", "sharegpt_aiperf", "raw",
    "sg_52k_head.json")
DEFAULT_PROCESSED = os.path.join(
    REPO_ROOT, "data", "external", "sharegpt_aiperf", "processed",
    "sharegpt_aiperf_ingest_summary.json")
DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "data", "external", "sharegpt_aiperf", "raw",
    "bounded_download_manifest.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bounded-ingest ShareGPT.")
    p.add_argument("--source-url",
                   default=sharegpt_aiperf.DEFAULT_SOURCE_URL)
    p.add_argument("--raw-path", default=DEFAULT_RAW)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    p.add_argument(
        "--max-bytes", type=int,
        default=sharegpt_aiperf.DEFAULT_BOUNDED_BYTES,
        help="HTTP-Range download cap (bytes). Default: 50 MB.")
    p.add_argument(
        "--max-records", type=int, default=None,
        help="Stop parsing after this many normalized records.")
    p.add_argument(
        "--no-download", action="store_true",
        help="Use --raw-path as-is (fail if missing) instead of downloading.")
    args = p.parse_args(argv)

    if args.max_bytes <= 0:
        print(f"[sharegpt] --max-bytes must be > 0 (got {args.max_bytes})",
              file=sys.stderr)
        return 2

    # 1. Bounded download (or reuse the local cached head).
    manifest: dict
    if not os.path.exists(args.raw_path) and not args.no_download:
        try:
            manifest = sharegpt_aiperf.download_bounded(
                url=args.source_url, dest_path=args.raw_path,
                max_bytes=args.max_bytes)
        except OSError as e:
            print(f"[sharegpt] DOWNLOAD ERROR: {e}", file=sys.stderr)
            return 3
        os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
        with open(args.manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[sharegpt] bounded download wrote {manifest['downloaded_bytes']:,} bytes")
    else:
        if not os.path.exists(args.raw_path):
            print(f"[sharegpt] raw file missing and --no-download set: "
                  f"{args.raw_path}", file=sys.stderr)
            return 4
        manifest = {
            "url": args.source_url,
            "requested_bytes": args.max_bytes,
            "downloaded_bytes": os.path.getsize(args.raw_path),
            "http_status": None,
            "dest_path": args.raw_path,
        }

    # 2. Parse + normalize.
    try:
        records = sharegpt_aiperf.load_json_path(
            args.raw_path, max_records=args.max_records)
    except EvalWorkloadSchemaError as e:
        print(f"[sharegpt] SCHEMA ERROR: {e}", file=sys.stderr)
        return 5

    if not records:
        print("[sharegpt] no records parsed", file=sys.stderr)
        return 6

    summary = sharegpt_aiperf.summarize(records)

    # 3. Persist processed summary. Token estimates + turn counts only — no
    # raw text. The processed file is committed; the raw head sample is
    # gitignored.
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    payload = {
        "dataset": sharegpt_aiperf.DATASET_NAME,
        "provenance": sharegpt_aiperf.PROVENANCE,
        "source_url": args.source_url,
        "source_repo_url": sharegpt_aiperf.SOURCE_REPO_URL,
        "aiperf_docs_url": sharegpt_aiperf.AIPERF_DOCS_URL,
        "bounded_download": manifest,
        "filters": {
            "max_bytes": args.max_bytes,
            "max_records": args.max_records,
        },
        "summary": summary.to_dict(),
        # Per-record (lightweight) records list — turn counts + token
        # estimates + provenance. NO raw text included.
        "records": [r.to_dict() for r in records],
    }
    with open(args.processed_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    _print_summary(summary, args.processed_path,
                   manifest_path=args.manifest_path,
                   n_records=len(records))
    return 0


def _print_summary(s, processed_path: str, *, manifest_path: str,
                   n_records: int) -> None:
    print("\n=== ShareGPT eval-conversation-shape ingestion summary ===")
    print(f"records parsed       : {n_records:,}")
    print(f"provenance           : {s.provenance}")
    print(f"has_timestamps       : {s.has_timestamps}  (ShareGPT has NONE)")
    print(f"has_real_tokens      : {s.has_real_tokens}  "
          f"(ShareGPT has NONE; char/4 proxy used)")
    print(f"has_model_id         : {s.has_model_id}    (ShareGPT has NONE)")
    print(f"has_language         : {s.has_language}    (ShareGPT has NONE)")
    print(f"failure_rate_pct     : {s.failure_rate_pct:.4f}")
    print("turn_count           : "
          f"p50={s.turn_count_p50:.0f} / p95={s.turn_count_p95:.0f} / "
          f"p99={s.turn_count_p99:.0f}")
    print("prompt_tokens_eff    : "
          f"p50={s.prompt_tokens_eff_p50:.0f} / "
          f"p95={s.prompt_tokens_eff_p95:.0f} / "
          f"p99={s.prompt_tokens_eff_p99:.0f}")
    print("response_tokens_eff  : "
          f"p50={s.response_tokens_eff_p50:.0f} / "
          f"p95={s.response_tokens_eff_p95:.0f} / "
          f"p99={s.response_tokens_eff_p99:.0f}")
    print(f"token_count_source   : {s.token_count_source_distribution}")
    print(f"role_sequence_top    : {s.role_sequence_top}")
    print(f"\nmanifest -> {manifest_path}")
    print(f"processed -> {processed_path}")


if __name__ == "__main__":
    raise SystemExit(main())
