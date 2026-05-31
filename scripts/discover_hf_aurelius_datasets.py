#!/usr/bin/env python3
"""Discover high-value Hugging Face datasets for Aurelius.

Pipeline:

1. Query the HF API for the keyword groups in
   ``aurelius.traces.hf_corpus.discovery.DEFAULT_QUERY_GROUPS`` plus the
   ``--extra-seed-id`` list. ``HF_TOKEN`` is honoured when present.
2. For every candidate: classify into a canonical trace type, score for
   Aurelius value, decide a recommended action.
3. Write the candidate registry to
   ``data/external/hf_discovery/hf_dataset_candidates.json``.

The discovery stage NEVER downloads data — only metadata. Ingestion is a
separate, opt-in stage (``scripts/ingest_hf_aurelius_dataset.py``).

Offline / hermetic CI mode: pass ``--fixtures-dir`` to use cached HF API
JSON responses instead of the live API. This is what tests run on.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import discovery  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery", "hf_dataset_candidates.json")


# Known-priority seed list from the mission spec. These are evaluated even
# if the keyword searches miss them.
SEED_DATASET_IDS = [
    "agent-perf-bench/AgentPerfBench",
    "odyn-network/odyn-benchmarks",
    "jaytonde05/prefixbench",
    "lmsys/chatbot_arena_conversations",
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "semianalysisai/cc-traces-weka-no-subagents-051226",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="HF dataset discovery for Aurelius.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--fixtures-dir", default=None,
                   help="Use OfflineHFClient + this fixtures dir instead "
                        "of the live HF API.")
    p.add_argument("--max-results-per-query", type=int,
                   default=discovery.DEFAULT_MAX_RESULTS_PER_QUERY)
    p.add_argument("--include-seeds", action="store_true", default=True,
                   help="(default) also probe the known-priority seed list.")
    p.add_argument("--no-seeds", dest="include_seeds", action="store_false")
    p.add_argument("--timeout-s", type=float, default=15.0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.fixtures_dir:
        client = discovery.OfflineHFClient(args.fixtures_dir)
        client_kind = f"offline:{args.fixtures_dir}"
    else:
        token = os.environ.get("HF_TOKEN")
        client = discovery.HFAPIClient(token=token, timeout_s=args.timeout_s)
        client_kind = "live_hf_api" + (" (HF_TOKEN_set)" if token else " (anon)")

    extra_seed_ids = SEED_DATASET_IDS if args.include_seeds else []

    t0 = time.time()
    candidates = discovery.discover(
        client,
        query_groups=discovery.DEFAULT_QUERY_GROUPS,
        extra_seed_ids=extra_seed_ids,
        max_results_per_query=args.max_results_per_query,
    )
    elapsed_s = time.time() - t0

    payload = {
        "doc_version": "hf_dataset_candidates_v1",
        "stage": "hf_dataset_discovery_v1",
        "production_claim": False,
        "client": client_kind,
        "query_groups_used": list(discovery.DEFAULT_QUERY_GROUPS.keys()),
        "seed_ids_used": list(extra_seed_ids),
        "discovery_elapsed_s": round(elapsed_s, 3),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    top = candidates[:20]
    print(f"[discovery] wrote {len(candidates)} candidates -> {args.output}")
    print(f"[discovery] top {len(top)} by overall_priority_score:")
    for c in top:
        print(f"  - {c['dataset_id']:60s}  score={c['overall_priority_score']:.3f}  "
              f"type={c['candidate_trace_type']:30s}  action={c['recommended_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
