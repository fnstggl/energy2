#!/usr/bin/env python3
"""Flatten the existing 3000 MiB CC-traces raw file into a per-request
analysis_sample.jsonl (gitignored) so the cache/prefix forecaster can
train without re-downloading or re-flattening.

The committed normalized_sample.jsonl is small (capped at 8 MiB / 5000
rows). This script materialises the FULL flattened set into the
gitignored analysis_sample.jsonl in the same directory.

No raw prompt/completion text is written (CC-traces does not contain
it). Block-id hash + count only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_cc_traces_3000mib import (  # noqa: E402
    _flatten_session,
    _read_jsonl_sessions,
    _safe_jsonable,
)

RAW_PATH = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "raw" / "traces_3000mib.jsonl"
OUT_PATH = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_3000mib" / "processed" / "analysis_sample.jsonl"


def main() -> int:
    if not RAW_PATH.exists():
        print(f"raw file missing: {RAW_PATH}", file=sys.stderr)
        return 2
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT_PATH, "w") as fh:
        for sess in _read_jsonl_sessions(RAW_PATH):
            for row in _flatten_session(sess, max_rows=None):
                fh.write(json.dumps(_safe_jsonable(row), sort_keys=True) + "\n")
                n += 1
    print(f"wrote {OUT_PATH} rows={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
