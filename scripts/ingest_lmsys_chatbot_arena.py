#!/usr/bin/env python3
"""Ingest the (gated) LMSYS Chatbot Arena conversations dataset.

The dataset is **gated** on HuggingFace; auto-download requires:
  1. Accepting the LMSYS terms-of-use at
     https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
  2. Supplying a HuggingFace token via ``HF_TOKEN`` (or ``--hf-token``).

The script will REFUSE to proceed without a token rather than silently
downloading nothing — the gated-access path must be explicit. This mirrors
the gated-resource pattern used by ``scripts/ingest_philly.py``.

If no token is supplied, the script prints the gated-access banner from
``aurelius.traces.lmsys_chatbot_arena`` and exits with a non-zero status. No
partial data is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import lmsys_chatbot_arena  # noqa: E402
from aurelius.traces.lmsys_chatbot_arena import (  # noqa: E402
    LMSYSGatedAccessError,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(
    REPO_ROOT, "data", "external", "lmsys_chatbot_arena", "raw",
    "chatbot_arena_conversations.parquet")
DEFAULT_PROCESSED = os.path.join(
    REPO_ROOT, "data", "external", "lmsys_chatbot_arena", "processed",
    "lmsys_chatbot_arena_ingest_summary.json")
DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "data", "external", "lmsys_chatbot_arena", "raw",
    "bounded_download_manifest.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Ingest LMSYS Chatbot Arena (gated).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=lmsys_chatbot_arena.LMSYS_GATED_BANNER,
    )
    p.add_argument("--source-url",
                   default=lmsys_chatbot_arena.DEFAULT_SOURCE_URL)
    p.add_argument("--raw-path", default=DEFAULT_RAW)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace API token. If omitted, falls back to "
                        "HF_TOKEN env var; if neither is set, the gated "
                        "download is refused.")
    p.add_argument("--max-bytes", type=int, default=None,
                   help="HTTP-Range download cap (parquet does not "
                        "decompress partial slices cleanly; mostly a budget).")
    args = p.parse_args(argv)

    try:
        manifest = lmsys_chatbot_arena.download_gated(
            url=args.source_url, dest_path=args.raw_path,
            hf_token=args.hf_token, max_bytes=args.max_bytes)
    except LMSYSGatedAccessError as e:
        print(str(e), file=sys.stderr)
        print("\n[lmsys] BLOCKED_GATED_DATASET — refusing to proceed without "
              "HF_TOKEN.", file=sys.stderr)
        return 7

    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    with open(args.manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[lmsys] gated download wrote "
          f"{manifest['downloaded_bytes']:,} bytes")

    # Parquet parsing requires pyarrow/pandas; the v1 ingester stops at the
    # download manifest and leaves the parquet -> EvalWorkloadRequest step to
    # the caller (which can use ``lmsys_chatbot_arena.normalize_record``
    # directly with a pyarrow-loaded row dict). We persist a minimal
    # processed summary recording the gated download success.
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)
    payload = {
        "dataset": lmsys_chatbot_arena.DATASET_NAME,
        "provenance": lmsys_chatbot_arena.PROVENANCE,
        "source_url": args.source_url,
        "source_repo_url": lmsys_chatbot_arena.SOURCE_REPO_URL,
        "gated": True,
        "gated_access_required": True,
        "schema_fields": sorted(lmsys_chatbot_arena.LMSYS_FIELDS),
        "schema_fields_relevant": list(
            lmsys_chatbot_arena.LMSYS_FIELDS_RELEVANT),
        "bounded_download": manifest,
        "note": ("v1 ingester stops at the download manifest; parquet -> "
                 "EvalWorkloadRequest normalization runs in caller code "
                 "via lmsys_chatbot_arena.normalize_record."),
    }
    with open(args.processed_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[lmsys] processed manifest -> {args.processed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
