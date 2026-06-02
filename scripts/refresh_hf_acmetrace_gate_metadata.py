#!/usr/bin/env python3
"""Refresh the gate-derived metadata on every committed AcmeTrace JSON.

This is an in-place maintenance helper that updates the four already-
committed ``data/external/hf/Qinghao__AcmeTrace/<config>/processed/
summary.json`` files and the rollup
``data/external/hf_discovery/acmetrace_audit_summary.json`` so they
carry the new redistribution-gate fields the eighth gate-consumer
wiring of ``scripts/ingest_hf_acmetrace.py`` introduces. We avoid
re-downloading the AcmeTrace dataset (hundreds of MB) — the gate
decision is a pure function of the recorded license tag, so we can
compute the verdict from the existing committed JSONs alone.

Pure-Python; no third-party deps; no HF API call; no HF_TOKEN read.

Invariants this script preserves:

* Every other field in summary.json is byte-for-byte unchanged.
* Field ordering is preserved by writing with ``sort_keys=True`` (the
  v1 writer used the same convention).
* The fixture files on disk are NOT touched.
* The audit summary's ``ingested`` row order matches the per-config
  ingest order embedded in ``TARGETS`` in the main ingest script.
* The ``discovery_only_records`` are re-emitted from the canonical
  ``DISCOVERY_ONLY_RECORDS`` table in the ingest script.

Run once after wiring the gate; the next live ingest writes the same
fields directly through the eighth-consumer code path.
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

ACMETRACE_ROOT = HF_DIR / "Qinghao__AcmeTrace"


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_hf_acmetrace_refresh",
        REPO_ROOT / "scripts" / "ingest_hf_acmetrace.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_hf_acmetrace_refresh"] = mod
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

    updated_configs: list[dict] = []
    for target in ingest.TARGETS:
        config = target["config_name"]
        processed = ACMETRACE_ROOT / config / "processed"
        summary_path = processed / "summary.json"
        if not summary_path.exists():
            print(f"skip {config}: no summary.json on disk")
            continue
        s = json.loads(summary_path.read_text())

        decision = ingest.evaluate_redistribution(
            ledger=ledger,
            license_tag=target.get("license", ingest.LICENSE_TAG),
            dataset_id=target["dataset_id"],
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
            json.dumps(s, indent=2, default=str, sort_keys=True) + "\n"
        )

        updated_configs.append({
            "dataset_id": target["dataset_id"],
            "config_name": config,
            "license": s["license"],
            "license_redistribution_status": decision.license_status,
            "redistribution_gate_reason_code": decision.reason_code,
            "redistribution_gate_permitted": decision.permitted,
            "redistribution_gate_operator_grant_dataset_id":
                decision.operator_grant_dataset_id,
            "canonical_trace_type": s.get("canonical_trace_type"),
            "available_signals": s.get("available_signals"),
            "missing_signals": s.get("missing_signals"),
            "analysis_sample_rows": s.get("analysis_sample_rows"),
            "statistical_sample_strength": s.get("statistical_sample_strength"),
            "limitations": s.get("limitations"),
        })

    audit_path = DISC_DIR / "acmetrace_audit_summary.json"
    prev = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    # Merge: for every config we updated, REPLACE the existing
    # ingested row in-place; preserve the rest of the v1 row keys (like
    # ``promotion_state`` / ``promotion_tags`` / ``elapsed_s``) so the
    # refresh stays additive.
    prev_ingested = prev.get("ingested", [])
    new_by_key = {
        (e["dataset_id"], e["config_name"]): e for e in updated_configs
    }
    merged_ingested: list[dict] = []
    for row in prev_ingested:
        key = (row.get("dataset_id"), row.get("config_name"))
        if key in new_by_key:
            updated = dict(row)
            updated.update(new_by_key[key])
            merged_ingested.append(updated)
            del new_by_key[key]
        else:
            merged_ingested.append(row)
    # Append any new rows (none expected; all four configs already exist).
    merged_ingested.extend(new_by_key.values())

    payload = {
        "doc_version": "acmetrace_audit_summary_v2",
        "stage": "hf_focused_audit_acmetrace_v2",
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "production_claim": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "redistribution_gate_scope": ingest.GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "ingested": merged_ingested,
        "failed": prev.get("failed", []),
        "discovery_only_records": ingest.DISCOVERY_ONLY_RECORDS,
    }
    audit_path.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )
    print(f"wrote {audit_path} ({len(merged_ingested)} ingested rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
