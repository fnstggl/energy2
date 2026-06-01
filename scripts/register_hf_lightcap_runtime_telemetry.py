#!/usr/bin/env python3
"""Register the Lightcap/agent-runtime-telemetry-small bounded ingest into
the canonical corpus registry + update the candidates JSON entry to
reflect that the Round-5 ``defer_high_value_different_trace_class`` block
has been cleared by the introduction of the ``tool_runtime_trace``
canonical type.

Reads the per-config summary.json under
``data/external/hf/Lightcap__agent-runtime-telemetry-small/<config>/processed/``
and re-writes
``data/external/hf_discovery/canonical_corpus_registry.json`` via
``aurelius.traces.hf_corpus.promotion``. Also stamps a
``focused_audit_2026_06_01c`` block on the candidates JSON.

Audit-only. No production claim. No scheduler / controller / robust
energy engine modified.
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
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
REGISTRY_PATH = DISC_DIR / "canonical_corpus_registry.json"
CANDIDATES_PATH = DISC_DIR / "hf_dataset_candidates.json"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")

NEW_ENTRIES = [
    (DATASET_ID, "operations"),
    (DATASET_ID, "tool_summary"),
    (DATASET_ID, "operation_events"),
    (DATASET_ID, "audit_records"),
]

AUDIT_BLOCK_KEY = "focused_audit_2026_06_01c"
FOLLOWUP_BLOCK_KEY = "focused_audit_2026_06_01d"


def _register_canonical_corpus() -> int:
    if not REGISTRY_PATH.exists():
        print(f"registry not found at {REGISTRY_PATH}", file=sys.stderr)
        return 1
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)

    existing = {(e["dataset_id"], e.get("config_name")): e
                for e in reg["entries"]}

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
        "doc_version": reg.get("doc_version",
                                "hf_corpus_canonical_registry_v1"),
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


def _update_candidates() -> int:
    if not CANDIDATES_PATH.exists():
        print(f"candidates not found at {CANDIDATES_PATH}", file=sys.stderr)
        return 1
    with open(CANDIDATES_PATH) as fh:
        cands = json.load(fh)

    audit_note = (
        "Round-5 defer cleared: Lightcap ingested 2026-06-01 as the inaugural "
        "tool_runtime_trace canonical type (new). operations config (2,262 "
        "rows × 33 cols) -> promoted_for_backtest + "
        "promoted_for_constraint_aware_evaluation + "
        "promoted_for_training_priors. tool_summary config (32 aggregated "
        "rows) -> promoted_for_schema_only. Trust tier: Tier 3 "
        "(tier_3_cluster_scheduler_traces — real measured execution "
        "telemetry, job-trace shape, but the 'jobs' are MCP tool calls, "
        "not GPU jobs). Routing-quality + failure-rate + tail-latency "
        "priors for agent workloads. NO GPU / NO model / NO LLM-serving "
        "signal — closed tool-runtime e2e timing only."
    )
    followup_note = (
        "Follow-up 2026-06-01d: Lightcap operation_events + audit_records "
        "configs ingested into the same tool_runtime_trace canonical type. "
        "operation_events (9,903 lifecycle events × 13 cols, moderate "
        "strength) -> promoted_for_backtest + constraint_aware_evaluation "
        "+ training_priors. Per-event duration_ms = ms-since-the-operation's "
        "started event = real dispatch-latency / execution-latency / "
        "completion-stage priors (started→executing p50=19 ms; "
        "affinity_warning p50=806 ms; full lifecycle p50=125 ms / p95=19 s). "
        "audit_records (14,053 rows × 17 cols, strong strength) -> "
        "promoted_for_backtest + constraint_aware_evaluation + "
        "training_priors. MCP-shell-layer duration_ms is REAL on "
        "tool_results rows (7,041 / 14,053; p50=4.7 ms / p95=400 ms / "
        "p99=2.5 s / max=900 s; error_rate=8.6%). NO GPU / NO model / NO "
        "LLM-serving signal — same closed-runtime-timing scope as the "
        "operations config. Joins: operation_id (operation_events <-> "
        "operations); request_id (audit_records <-> operations)."
    )

    candidates = cands.get("candidates") or []
    updated = 0
    for c in candidates:
        if c.get("dataset_id") == DATASET_ID:
            c["recommended_action"] = "ingest_now_bounded"
            c["audit_round"] = "focused_audit_2026_06_01d"
            c["audit_decision"] = "ingested_as_tool_runtime_trace_followup"
            c["audit_note_2026_06_01c"] = audit_note
            c["audit_note_2026_06_01d"] = followup_note
            c["canonical_trace_type"] = "tool_runtime_trace"
            c["trust_level"] = "tier_3_cluster_scheduler_traces"
            updated += 1

    cands["candidates"] = candidates
    cands["last_updated_at_s"] = time.time()
    # Preserve the original 2026_06_01c audit block (do not overwrite if
    # already present); the follow-up registers its own 2026_06_01d block.
    if AUDIT_BLOCK_KEY not in cands:
        cands[AUDIT_BLOCK_KEY] = {
            "ran_at_s": time.time(),
            "scope": (
                "Inaugural tool_runtime_trace canonical-type ingest "
                "(Lightcap/agent-runtime-telemetry-small). Clears Round-5 "
                "'defer_high_value_different_trace_class' block by adding the "
                "new canonical type to aurelius/traces/hf_corpus/schemas.py + "
                "promotion.py."
            ),
            "datasets": [DATASET_ID],
            "configs_ingested": [
                f"{DATASET_ID}@operations (2,262 rows, moderate strength)",
                f"{DATASET_ID}@tool_summary (32 aggregated rows, fixture_only "
                "strength → promoted_for_schema_only)",
            ],
            "new_canonical_type": "tool_runtime_trace",
            "trust_tier": "tier_3_cluster_scheduler_traces",
            "license": "cc-by-4.0",
            "headline_promotion_state": "promoted_for_backtest",
            "headline_promotion_tags": [
                "promoted_for_backtest",
                "promoted_for_constraint_aware_evaluation",
                "promoted_for_training_priors",
            ],
            "production_claim": False,
            "modifies_robust_energy_engine": False,
            "modifies_controllers_or_defaults": False,
        }
    cands[FOLLOWUP_BLOCK_KEY] = {
        "ran_at_s": time.time(),
        "scope": (
            "Lightcap follow-up: ingested the two remaining configs "
            "(operation_events + audit_records) into the existing "
            "tool_runtime_trace canonical type. Unlocks per-stage "
            "lifecycle-transition timing (dispatch / execution / "
            "completion priors at the agent-runtime layer) and "
            "MCP-shell-layer request/response duration priors with "
            "strong-strength sample (14,053 rows; p99=2.5 s, max=900 s). "
            "Closes the documented §10 next-task 'Lightcap follow-up "
            "(next-run priority)' from registry doc / PR #143."
        ),
        "datasets": [DATASET_ID],
        "configs_ingested": [
            f"{DATASET_ID}@operation_events (9,903 events × 2,262 "
            "operations, moderate strength)",
            f"{DATASET_ID}@audit_records (14,053 audit records: 7,012 "
            "requests + 7,041 results, strong strength)",
        ],
        "canonical_type": "tool_runtime_trace",
        "trust_tier": "tier_3_cluster_scheduler_traces",
        "license": "cc-by-4.0",
        "headline_promotion_state": "promoted_for_backtest",
        "headline_promotion_tags": [
            "promoted_for_backtest",
            "promoted_for_constraint_aware_evaluation",
            "promoted_for_training_priors",
        ],
        "join_keys": {
            "operation_events__operations": "operation_id",
            "audit_records__operations": "request_id",
        },
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
    }
    with open(CANDIDATES_PATH, "w") as fh:
        json.dump(cands, fh, indent=2, sort_keys=True)
    print(f"Updated {CANDIDATES_PATH} ({updated} entries touched, "
          f"+1 follow-up audit block: {FOLLOWUP_BLOCK_KEY})")
    return 0


def main() -> int:
    rc = _register_canonical_corpus()
    if rc != 0:
        return rc
    return _update_candidates()


if __name__ == "__main__":
    sys.exit(main())
