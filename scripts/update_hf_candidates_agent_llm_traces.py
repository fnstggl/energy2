#!/usr/bin/env python3
"""Update data/external/hf_discovery/hf_dataset_candidates.json with the
Exgentic/agent-llm-traces follow-on entry (was 'defer_high_value_large_size'
in PR #135; now bounded-ingested as a single shard).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATES_PATH = REPO_ROOT / "data" / "external" / "hf_discovery" / "hf_dataset_candidates.json"


CANDIDATE = {
    "dataset_id": "Exgentic/agent-llm-traces",
    "dataset_url": "https://huggingface.co/datasets/Exgentic/agent-llm-traces",
    "gated_status": "public",
    "license": "cdla-permissive-2.0",
    "estimated_size": ["1K<n<10K"],
    "available_splits": [{"split": "train", "config": "default",
                          "num_examples": 1781}],
    "configs": ["default"],
    "feature_names": ["harness", "benchmark", "models", "max_tokens",
                      "total_tokens", "session_id", "spans", "collected_at"],
    "candidate_trace_type": "request_shape_trace",
    "classification_evidence": {
        "request_shape_trace": [
            "OpenTelemetry spans with gen_ai.usage.input_tokens / output_tokens",
            "session_id + spans + start_time/end_time → workload shape",
        ],
    },
    "matched_keywords": [
        "agent_workload::OpenTelemetry",
        "request_shape::gen_ai_tokens",
        "latency::closed_api_duration",
    ],
    "available_signals": [
        "arrivals", "request_timestamps", "workload_shape",
        "customer_traffic_mix", "routing_proxy",
        "cache_reuse", "prefix_reuse",
        "latency",   # closed-API end-to-end, NOT GPU TTFT/TPOT
        "sla_label", "timeout_label",  # OTel status.code == ERROR
    ],
    "missing_signals": [
        "ttft", "tpot", "itl", "queue_state",
        "gpu_utilization", "gpu_memory", "memory_pressure",
        "replica_count", "autoscaling_proxy", "capacity_proxy",
        "kv_block_hashes", "model_load_event", "model_unload_event",
    ],
    "aurelius_use_case": (
        "Per-LLM-call duration / tokens / status / model priors for "
        "AGENT workloads (Claude Code / OpenAI tool_calling / Gemini / "
        "DeepSeek / Kimi). Workload-shape evidence only; the duration_ms "
        "is closed-API end-to-end (network + provider serving), NOT a "
        "GPU TTFT/TPOT signal."
    ),
    "not_recommended_uses": [
        "GPU-serving latency calibration (durations include provider routing + network)",
        "TTFT / TPOT inference (no per-token timing)",
        "Production dynamic-frontier calibration",
        "Cross-vendor latency generalisation (azure-hosted models in this shard)",
    ],
    "trust_level": "tier_5_request_shape_traces",
    "ingestion_feasibility_score": 5,   # parquet, public, permissive license
    "frontier_value_score": 3,          # agent task duration / token gap fill
    "schema_quality_score": 5,          # OTel semantic conventions
    "production_similarity_score": 2,   # closed-API e2e, not pilot telemetry
    "overall_priority_score": 3.5,
    "recommended_action": "ingest_now_bounded",
    "schema_available": True,
    "downloads": 2117,
    "likes": 17,
    "last_modified": "2026-05-14T18:50:50.000Z",
    "discovery_timestamp_s": time.time(),
}


def main() -> int:
    if not CANDIDATES_PATH.exists():
        print(f"candidates not found at {CANDIDATES_PATH}", file=sys.stderr)
        return 1
    with open(CANDIDATES_PATH) as fh:
        doc = json.load(fh)
    cands = doc.get("candidates", [])
    by_id = {c["dataset_id"]: i for i, c in enumerate(cands)}
    if CANDIDATE["dataset_id"] in by_id:
        cands[by_id[CANDIDATE["dataset_id"]]] = CANDIDATE
        action = "updated"
    else:
        cands.append(CANDIDATE)
        action = "appended"
    cands.sort(key=lambda c: c["dataset_id"])
    doc["candidates"] = cands
    doc["candidate_count"] = len(cands)
    doc["last_updated_at_s"] = time.time()
    doc["updated_at_s"] = time.time()
    # Stamp the follow-on audit under a new dated key.
    doc["focused_audit_2026_06_01b"] = {
        "audit_label": "exgentic_agent_llm_traces_followup_ingest",
        "audit_date": "2026-06-01",
        "outcomes": {
            "Exgentic/agent-llm-traces": {
                "trace_type": "request_shape_trace",
                "trust_tier": "tier_5_request_shape_traces",
                "license": "cdla-permissive-2.0",
                "action": "ingest_now_bounded",
                "config_ingested": "swebench_claude_code_shard12",
                "rows_sampled": 2294,
                "strength": "moderate",
                "promotion_state": "promoted_for_training_priors",
                "promotion_tags": ["promoted_for_training_priors"],
                "informs": ["workload_modelling"],
                "notes": (
                    "Bounded ingest of one mid-sized parquet "
                    "(train-00012-of-00039, 41 MB raw, gitignored). "
                    "Flattens OTel session-spans to per-span request rows "
                    "with duration_ms + tokens + model + status + finish "
                    "reasons + payload-size proxies. duration_ms is "
                    "closed-API e2e — NOT a GPU TTFT/TPOT signal. "
                    "Other shards remain available for follow-on ingest."
                ),
            },
        },
        "audit_timestamp_s": time.time(),
    }
    with open(CANDIDATES_PATH, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print(f"  {action} {CANDIDATE['dataset_id']} candidate "
          f"(total {len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
