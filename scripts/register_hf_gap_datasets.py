#!/usr/bin/env python3
"""Append the 5 newly-ingested gap datasets to the canonical corpus registry.

Reads each per-config summary.json under data/external/hf/<safe>/<config>/processed/
and writes the union (existing entries + new entries) back to
data/external/hf_discovery/canonical_corpus_registry.json via
``aurelius.traces.hf_corpus.promotion``.

Audit-only. No production claim. Does not modify scheduler / controllers /
robust energy engine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
REGISTRY_PATH = REPO_ROOT / "data" / "external" / "hf_discovery" / "canonical_corpus_registry.json"

NEW_ENTRIES = [
    ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head"),
    ("sammshen/lmcache-agentic-traces", "train_shard4"),
    ("lzzmm/BurstGPT", "burstgpt_1_full"),
    ("lsliwko/google-cluster-data-2019-sorted-by-timestamp", "instance_events_shard0"),
    ("jaytonde05/prefixbench", "prefixbench_all"),
]


def main() -> int:
    if not REGISTRY_PATH.exists():
        print(f"registry not found at {REGISTRY_PATH}", file=sys.stderr)
        return 1
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)

    existing = {(e["dataset_id"], e.get("config_name")): e for e in reg["entries"]}

    appended = 0
    for dataset_id, config in NEW_ENTRIES:
        safe = dataset_id.replace("/", "__")
        summary_path = HF_DIR / safe / config / "processed" / "summary.json"
        if not summary_path.exists():
            print(f"missing summary: {summary_path}", file=sys.stderr)
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)
        decision = promotion.evaluate_promotion(summary)
        entry = promotion.build_registry_entry(summary, decision)
        existing[(dataset_id, config)] = entry
        appended += 1
        print(f"  registered {dataset_id}@{config} "
              f"state={decision['state']} tags={decision['promotion_tags']}")

    entries = sorted(existing.values(),
                     key=lambda e: (e["dataset_id"], e.get("config_name") or ""))
    payload = {
        "doc_version": reg.get("doc_version", "hf_corpus_canonical_registry_v1"),
        "stage": reg.get("stage", "federated_benchmark_corpus_v1"),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "trust_hierarchy_note": reg.get(
            "trust_hierarchy_note",
            "Tier 1 (real pilot telemetry) remains the only production "
            "calibration source. Promotion here is research-class only.",
        ),
        "written_at_s": time.time(),
        "entry_count": len(entries),
        "entries": entries,
    }
    with open(REGISTRY_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nWrote {REGISTRY_PATH} ({len(entries)} entries, +{appended} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
