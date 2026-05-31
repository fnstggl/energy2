#!/usr/bin/env python3
"""Capture HF API responses to tests/fixtures/hf_api/ for hermetic CI.

Run once per discovery-spec change. The fixtures are tiny JSON files that
let the test suite + ``discover_hf_aurelius_datasets.py --fixtures-dir``
run offline without flaking on the live HF API.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import discovery

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO_ROOT, "tests", "fixtures", "hf_api")


SEED_DATASET_IDS = [
    "agent-perf-bench/AgentPerfBench",
    "jaytonde05/prefixbench",
    "lmsys/chatbot_arena_conversations",
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "semianalysisai/cc-traces-weka-no-subagents-051226",
]


def main() -> int:
    client = discovery.HFAPIClient()
    os.makedirs(os.path.join(OUT_ROOT, "search"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "datasets"), exist_ok=True)

    # Search queries.
    for group, queries in discovery.DEFAULT_QUERY_GROUPS.items():
        for q in queries:
            res = client.search(q, limit=10)
            path = os.path.join(OUT_ROOT, "search",
                                discovery._safe_name(q) + ".json")
            # Keep only the safe metadata fields we use.
            slim = [
                {
                    "id": r.get("id"),
                    "gated": r.get("gated"),
                    "private": r.get("private"),
                    "tags": (r.get("tags") or [])[:25],
                    "downloads": r.get("downloads"),
                    "likes": r.get("likes"),
                    "description": (r.get("description") or "")[:400],
                    "lastModified": r.get("lastModified"),
                }
                for r in res if isinstance(r, dict) and "id" in r
            ]
            with open(path, "w") as fh:
                json.dump(slim, fh, indent=2, sort_keys=True)
            print(f"search: {q!r:50s} -> {len(slim)} hits -> {path}")

    # Dataset detail.
    for ds_id in SEED_DATASET_IDS:
        meta = client.get(ds_id)
        path = os.path.join(OUT_ROOT, "datasets",
                            discovery._safe_name(ds_id) + ".json")
        if meta is None:
            with open(path, "w") as fh:
                json.dump({"id": ds_id, "_unavailable": True}, fh, indent=2,
                          sort_keys=True)
            print(f"detail: {ds_id} UNAVAILABLE")
            continue
        # Keep relevant fields only; drop big siblings list trailing entries.
        slim = {
            "id": meta.get("id"),
            "gated": meta.get("gated"),
            "private": meta.get("private"),
            "tags": (meta.get("tags") or [])[:30],
            "description": (meta.get("description") or "")[:600],
            "downloads": meta.get("downloads"),
            "likes": meta.get("likes"),
            "lastModified": meta.get("lastModified"),
            "cardData": meta.get("cardData"),
            "siblings": meta.get("siblings"),
        }
        with open(path, "w") as fh:
            json.dump(slim, fh, indent=2, sort_keys=True)
        print(f"detail: {ds_id} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
