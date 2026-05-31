#!/usr/bin/env python3
"""Run the compatibility-routed corpus evaluation harness.

Reads ``data/external/hf_discovery/canonical_corpus_registry.json``,
routes every promoted entry to its trace-type-appropriate smoke
evaluator, runs bounded evaluations only, and writes the structured
result to ``data/external/hf_discovery/hf_corpus_evaluation_summary.json``.

Rules (from the mission spec):

- No oracle as headline.
- No aggregation across incompatible trace types.
- Bounded sample reads only (no full-trace replay).
- Skips with explicit reasons when required signals are missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import evaluation, promotion  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery", "canonical_corpus_registry.json")
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery", "hf_corpus_evaluation_summary.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="HF corpus evaluation harness.")
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    registry = promotion.load_canonical_registry(args.registry)
    if registry is None:
        print(f"[eval] registry missing: {args.registry}", file=sys.stderr)
        return 2

    payload = evaluation.run_corpus_evaluation(
        registry, REPO_ROOT, max_rows=args.max_rows
    )
    evaluation.write_evaluation_summary(payload, args.output)

    print(f"[eval] eligible={payload['n_eligible']}  "
          f"evaluated={payload['n_evaluated']}  "
          f"skipped={payload['n_skipped']}")
    for r in payload["per_dataset_results"]:
        if r.get("skip_reason"):
            print(f"  SKIP {r['dataset_id']:60s}  reason={r['skip_reason']}")
        else:
            print(f"   OK  {r['dataset_id']:60s}  evaluator={r['evaluator_id']:40s}  "
                  f"kpi={r['kpi']}")
    print(f"[eval] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
