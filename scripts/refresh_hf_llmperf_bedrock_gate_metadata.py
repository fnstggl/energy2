#!/usr/bin/env python3
"""Refresh the gate-derived metadata on the committed llmperf-bedrock JSONs.

This is an in-place maintenance helper that updates the already-committed
``data/external/hf/ssong1__llmperf-bedrock/bedrock_claude_instant_v1/
processed/summary.json`` and the rollup
``data/external/hf_discovery/round3_broadened_discovery_audit_summary.json``
so they carry the new redistribution-gate fields the tenth gate-consumer
wiring of ``scripts/ingest_hf_llmperf_bedrock.py`` introduces. We avoid
re-downloading the LLMPerf raw JSONs (already cached under
``data/external/hf/ssong1__llmperf-bedrock/raw/``, gitignored) — the gate
decision is a pure function of the recorded license tag, so we can
compute the verdict from the existing committed JSONs alone.

Pure-Python; no third-party deps; no HF API call; no HF_TOKEN read.

Invariants this script preserves:

* Every other field in summary.json is byte-for-byte unchanged.
* Field ordering is preserved by writing with ``sort_keys=True`` (the
  v1 writer used the same convention).
* The fixture file on disk is NOT touched.
* The committed normalised sample bytes are NOT touched.
* The discovery-only records are re-emitted from the canonical
  ``ROUND3_DISCOVERY_RECORDS`` table in the ingest script.

Run once after wiring the gate; the next live ingest writes the same
fields directly through the tenth-consumer code path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"

LLMPERF_BEDROCK_ROOT = HF_DIR / "ssong1__llmperf-bedrock"
CONFIG = "bedrock_claude_instant_v1"


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_hf_llmperf_bedrock_refresh",
        REPO_ROOT / "scripts" / "ingest_hf_llmperf_bedrock.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_hf_llmperf_bedrock_refresh"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ingest = _load_ingest_module()
    ledger = ingest._load_ledger()

    summary_path = (
        LLMPERF_BEDROCK_ROOT / CONFIG / "processed" / "summary.json"
    )
    if not summary_path.exists():
        print(f"skip {CONFIG}: no summary.json on disk")
        return 1

    s = json.loads(summary_path.read_text())

    decision = ingest.evaluate_redistribution(
        ledger=ledger,
        license_tag=ingest.LICENSE_TAG,
        dataset_id=ingest.DATASET_ID,
    )

    s["license_redistribution_status"] = decision.license_status
    s["license_redistribution_source"] = ingest.LICENSE_SOURCE
    s["redistribution_gate_reason_code"] = decision.reason_code
    s["redistribution_gate_reason_detail"] = decision.reason_detail
    s["redistribution_gate_permitted"] = decision.permitted
    s["redistribution_gate_operator_grant_dataset_id"] = (
        decision.operator_grant_dataset_id
    )
    s["redistribution_gate_scope"] = ingest.GATE_SCOPE

    summary_path.write_text(
        json.dumps(s, indent=2, default=str, sort_keys=True)
    )
    print(f"wrote {summary_path}")

    # Refresh the audit summary in place. Preserve the existing
    # ``ingested`` row's pre-existing keys (promotion_state /
    # promotion_tags / promotion_reasons / etc.) so the refresh stays
    # additive.
    audit_path = (
        DISC_DIR / "round3_broadened_discovery_audit_summary.json"
    )
    prev = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    prev_ingested = prev.get("ingested", [])
    updated_ingested: list[dict] = []
    matched_target = False
    for row in prev_ingested:
        if (row.get("dataset_id") == ingest.DATASET_ID
                and row.get("config_name") == CONFIG):
            matched_target = True
            updated_row = dict(row)
            updated_row["license_redistribution_status"] = (
                decision.license_status
            )
            updated_row["redistribution_gate_reason_code"] = (
                decision.reason_code
            )
            updated_row["redistribution_gate_permitted"] = (
                decision.permitted
            )
            updated_row["redistribution_gate_operator_grant_dataset_id"] = (
                decision.operator_grant_dataset_id
            )
            updated_ingested.append(updated_row)
        else:
            updated_ingested.append(row)
    if not matched_target:
        # First-run path (or audit summary regenerated from scratch):
        # build a fresh row from the committed summary.json.
        updated_ingested.append({
            "dataset_id": s["dataset_id"],
            "config_name": s["config_name"],
            "canonical_trace_type": s["canonical_trace_type"],
            "license": s["license"],
            "license_redistribution_status": decision.license_status,
            "redistribution_gate_reason_code": decision.reason_code,
            "redistribution_gate_permitted": decision.permitted,
            "redistribution_gate_operator_grant_dataset_id":
                decision.operator_grant_dataset_id,
            "gated": s["gated"],
            "analysis_sample_rows": s["analysis_sample_rows"],
            "fixture_sample_rows": s["fixture_sample_rows"],
            "committed_normalized_sample_rows":
                s["committed_normalized_sample_rows"],
            "committed_normalized_sample_bytes":
                s["committed_normalized_sample_bytes"],
            "available_signals": s["available_signals"],
            "missing_signals": s["missing_signals"],
            "limitations": s["limitations"],
            "statistical_sample_strength":
                s["statistical_sample_strength"],
        })

    payload = {
        "doc_version": "round3_broadened_discovery_audit_summary_v2",
        "scope": prev.get(
            "scope",
            "Round 3 broadened HF discovery — bounded ingest of "
            "ssong1/llmperf-bedrock (Tier-4 closed-managed-API LLMPerf "
            "TTFT/ITL/e2e priors) + 8 negative-result discovery records "
            "covering DistServe profiling (license unspecified), "
            "DynamoRIO drmemtrace (out of scope), "
            "intellistream sage-control-plane (insufficient sample), "
            "Nathan-Maine + hlarcher (already audited duplicates), "
            "kshitijthakkar MoE benchmarks (gated_blocked).",
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "redistribution_gate_scope": ingest.GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "ingested": updated_ingested,
        "discovery_only_records": ingest.ROUND3_DISCOVERY_RECORDS,
        "failed": prev.get("failed", []),
    }
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {audit_path} ({len(updated_ingested)} ingested rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
