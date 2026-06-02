#!/usr/bin/env python3
"""Materialize bounded normalized analysis samples for the PR #129 datasets.

PR #129 ingested raw → normalized but only committed tiny fixtures. This
script promotes the gitignored ``processed/analysis_sample.jsonl`` to a
committed ``processed/normalized_sample.jsonl`` for datasets whose
license permits redistribution.

Redistribution decision (wired through the canonical gate)
----------------------------------------------------------

This script is now the second consumer of
:func:`aurelius.ingestion.redistribution_gate.decide_redistribution`
(after ``scripts/audit_hf_redistribution_gate.py``). The hard-coded
``license_redistribution_status`` / ``commit_sample`` fields that used to
live in this script's TARGETS table have been removed; the script now
records only the raw HF license tag plus a human-curated provenance
string (``license_source``), and asks the gate for both the canonical
status label and the commit decision.

Behaviour under the default policy (committed
``operator_redistribution_policy.json`` ships zero grants):

- Permissive declared licenses (apache-2.0, mit, cc-by-4.0, cdla-2, …)
  → permitted; ``license_redistribution_status`` is the canonical
  ``permissive_*`` label from the gate (unchanged from before).
- ``license = None`` → denied with ``no_grant_recorded`` (unchanged
  from before, but now this is the gate denying via the ledger rather
  than a hard-coded ``commit_sample=False``).

The pre-existing backwards-compat test
(``test_classify_license_agrees_with_commit_script_targets`` in
``tests/test_hf_redistribution_gate.py``) pins the gate's verdicts on
every license tag this script ships, so wiring the gate in cannot
change the four already-committed normalised samples' status.

If an operator records a grant entry in
``operator_redistribution_policy.json`` for a ``license = None`` row
listed in TARGETS (e.g. prefixbench), the gate flips that row to
``permitted_operator_grant`` and this script will commit a normalised
sample on the next run — without any code change. That is the entire
point of wiring the gate in.

Policy (binding):
- raw downloads remain gitignored
- each committed normalized sample ≤ 50 MB
- sum of all committed normalized samples in this run ≤ 150 MB
- redistribution decision comes from ``decide_redistribution`` only;
  this script does NOT classify licenses itself

Summary additions per dataset:
- committed_normalized_sample_path / _bytes / _rows / _sha256
- license_redistribution_status (canonical ``permissive_*`` or
  ``unspecified_no_committed_sample`` etc., produced by the gate)
- license_redistribution_source (human-curated provenance — where the
  license tag came from)
- redistribution_gate_reason_code (gate's closed-set verdict code so
  downstream tooling can pivot on a stable token)
- raw_committed=false
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.ingestion.operator_redistribution_policy import (  # noqa: E402
    OperatorPolicyLedger,
)
from aurelius.ingestion.redistribution_gate import (  # noqa: E402
    RedistributionGateDecision,
    decide_redistribution,
)

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
POLICY_PATH = DISC_DIR / "operator_redistribution_policy.json"

MAX_PER_SAMPLE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_TOTAL_COMMITTED_BYTES = 150 * 1024 * 1024  # 150 MB

GATE_SCOPE = "committed_normalized_sample"

# Per-dataset RAW HF license tag + human-curated provenance.
#
# The script does NOT classify the tag here — it asks the gate. The tag
# is the same string that appears on the HF dataset card frontmatter
# (or, where the HF card was silent, the string read from the dataset's
# LICENSE file fetched directly from the repo and recorded in
# ``license_source``). ``None`` means the HF card has no ``license:``
# key and no LICENSE file was found → the gate denies under the default
# policy, but an operator grant could opt this dataset in without any
# code change.
TARGETS = [
    {
        "dataset_id": "lzzmm/BurstGPT",
        "config_name": "burstgpt_1_full",
        "license_tag": "cc-by-4.0",
        "license_source": (
            "LICENSE file at https://huggingface.co/datasets/lzzmm/BurstGPT/blob/main/LICENSE "
            "= 'Attribution 4.0 International' (CC-BY-4.0)"
        ),
    },
    {
        "dataset_id": "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
        "config_name": "instance_events_shard0",
        "license_tag": "cc-by-4.0",
        "license_source": (
            "Mirror of github.com/google/cluster-data, released by Google "
            "under CC-BY-4.0; HF redistribution preserves the same terms"
        ),
    },
    {
        "dataset_id": "sammshen/lmcache-agentic-traces",
        "config_name": "train_shard4",
        "license_tag": "mit",
        "license_source": "HF card frontmatter license: mit",
    },
    {
        "dataset_id": "semianalysisai/cc-traces-weka-no-subagents-051226",
        "config_name": "traces_head",
        "license_tag": "apache-2.0",
        "license_source": "HF card frontmatter license: apache-2.0",
    },
    {
        "dataset_id": "jaytonde05/prefixbench",
        "config_name": "prefixbench_all",
        "license_tag": None,
        "license_source": (
            "HF card frontmatter has no `license:` key; README provides no "
            "redistribution statement. Conservative: the gate denies under "
            "the default deny_all policy with reason_code "
            "`no_grant_recorded`. An operator grant entry in "
            "`operator_redistribution_policy.json` would unblock without a "
            "code change."
        ),
    },
]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _processed_dir(dataset_id: str, config: str) -> Path:
    return HF_DIR / dataset_id.replace("/", "__") / config / "processed"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def _load_ledger(policy_path: Path) -> OperatorPolicyLedger:
    """Load the operator policy ledger from disk, or fall back to empty.

    The default behaviour when the policy file is missing is the same
    as the committed default file: deny-all, zero grants. We use
    ``empty()`` so the script remains self-sufficient (e.g. in a fresh
    checkout without the committed JSON, the script still produces
    correct deny decisions instead of crashing).
    """

    if policy_path.exists():
        return OperatorPolicyLedger.load(policy_path)
    return OperatorPolicyLedger.empty()


def evaluate_target(
    target: dict,
    *,
    ledger: OperatorPolicyLedger,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Ask the gate whether this target's normalised sample may be committed.

    Pure function — no I/O. Exposed so tests can drive the gate path
    without invoking the filesystem-mutating ``materialize``.
    """

    return decide_redistribution(
        dataset_id=target["dataset_id"],
        license_str=target.get("license_tag"),
        scope=GATE_SCOPE,
        ledger=ledger,
        now_iso=now_iso,
    )


def materialize(
    target: dict,
    total_committed_so_far: int,
    *,
    ledger: OperatorPolicyLedger,
    now_iso: Optional[str] = None,
) -> tuple[dict, int]:
    pd = _processed_dir(target["dataset_id"], target["config_name"])
    summary_path = pd / "summary.json"
    if not summary_path.exists():
        return (
            {"target": target, "audit_status": "summary_missing"},
            total_committed_so_far,
        )
    with open(summary_path) as fh:
        summary = json.load(fh)

    decision = evaluate_target(target, ledger=ledger, now_iso=now_iso)

    analysis = pd / "analysis_sample.jsonl"
    committed_path = pd / "normalized_sample.jsonl"
    result = {
        "dataset_id": target["dataset_id"],
        "config_name": target["config_name"],
        "license_tag": target.get("license_tag"),
        "license_redistribution_status": decision.license_status,
        "license_source": target["license_source"],
        "redistribution_gate_reason_code": decision.reason_code,
        "redistribution_gate_permitted": decision.permitted,
        "redistribution_gate_operator_grant_dataset_id": (
            decision.operator_grant_dataset_id
        ),
        "commit_decision": "skip",
    }

    # Always update summary metadata, regardless of commit decision. The
    # gate's verdict is the single source of truth for the redistribution
    # status string and the reason code.
    summary["license_redistribution_status"] = decision.license_status
    summary["license_redistribution_source"] = target["license_source"]
    summary["redistribution_gate_reason_code"] = decision.reason_code
    summary["redistribution_gate_reason_detail"] = decision.reason_detail
    summary["redistribution_gate_permitted"] = decision.permitted
    summary["redistribution_gate_operator_grant_dataset_id"] = (
        decision.operator_grant_dataset_id
    )
    summary["raw_committed"] = False

    if not decision.permitted:
        summary["committed_normalized_sample_path"] = None
        summary["committed_normalized_sample_bytes"] = 0
        summary["committed_normalized_sample_rows"] = 0
        summary["committed_normalized_sample_sha256"] = None
        summary["committed_normalized_sample_reason_skipped"] = (
            f"redistribution_gate denied: reason_code="
            f"{decision.reason_code!r}; "
            f"license_redistribution_status="
            f"{decision.license_status!r}"
        )
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        result["commit_decision"] = (
            f"SKIPPED (gate denied: {decision.reason_code})"
        )
        return result, total_committed_so_far

    if not analysis.exists():
        # Fast path: in a fresh checkout the gitignored
        # ``analysis_sample.jsonl`` is missing, but the committed
        # ``normalized_sample.jsonl`` is git-tracked and present. If
        # the existing committed sample's bytes match the sha256
        # recorded in summary.json, we are idempotent — the gate's
        # verdict for this target has not changed (the
        # backwards-compat test pins that), and there is nothing to
        # re-do. We still refresh the gate-derived fields in the
        # summary so a re-run picks up the new reason_code metadata.
        existing_sha = summary.get("committed_normalized_sample_sha256")
        existing_path = summary.get("committed_normalized_sample_path")
        existing_bytes = summary.get("committed_normalized_sample_bytes") or 0
        existing_rows = summary.get("committed_normalized_sample_rows") or 0
        if (
            committed_path.exists()
            and existing_sha
            and existing_path
            and _sha256_file(committed_path) == existing_sha
        ):
            with open(summary_path, "w") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
            result["commit_decision"] = "COMMITTED"
            result["committed_path"] = existing_path
            result["committed_bytes"] = int(existing_bytes)
            result["committed_rows"] = int(existing_rows)
            result["committed_sha256"] = existing_sha
            return result, total_committed_so_far + int(existing_bytes)

        result["commit_decision"] = (
            "SKIPPED (analysis_sample missing — re-run "
            "scripts/ingest_hf_gap_datasets.py)"
        )
        return result, total_committed_so_far

    sz = analysis.stat().st_size
    if sz > MAX_PER_SAMPLE_BYTES:
        result["commit_decision"] = (
            f"SKIPPED (analysis_sample is {sz:,} bytes, exceeds 50 MB per-sample cap)"
        )
        return result, total_committed_so_far
    if total_committed_so_far + sz > MAX_TOTAL_COMMITTED_BYTES:
        result["commit_decision"] = (
            f"SKIPPED (would exceed 150 MB total committed cap: "
            f"running {total_committed_so_far + sz:,} > 150 MB)"
        )
        return result, total_committed_so_far

    # Copy analysis -> committed. They're identical bytes; we keep both
    # because gitignore matches analysis_sample.jsonl, not normalized_sample.jsonl.
    committed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis, "rb") as src, open(committed_path, "wb") as dst:
        while True:
            chunk = src.read(1 << 16)
            if not chunk:
                break
            dst.write(chunk)

    committed_bytes = committed_path.stat().st_size
    committed_rows = _count_rows(committed_path)
    committed_sha = _sha256_file(committed_path)

    summary["committed_normalized_sample_path"] = os.path.relpath(
        committed_path, REPO_ROOT,
    ).replace(os.sep, "/")
    summary["committed_normalized_sample_bytes"] = committed_bytes
    summary["committed_normalized_sample_rows"] = committed_rows
    summary["committed_normalized_sample_sha256"] = committed_sha
    summary["committed_normalized_sample_materialized_at_s"] = time.time()
    summary["committed_normalized_sample_git_sha"] = _git_sha()
    summary["committed_normalized_sample_reason_skipped"] = None
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    result["commit_decision"] = "COMMITTED"
    result["committed_path"] = summary["committed_normalized_sample_path"]
    result["committed_bytes"] = committed_bytes
    result["committed_rows"] = committed_rows
    result["committed_sha256"] = committed_sha
    return result, total_committed_so_far + committed_bytes


def main() -> int:
    ledger = _load_ledger(POLICY_PATH)
    results = []
    total = 0
    for t in TARGETS:
        r, total = materialize(t, total, ledger=ledger)
        results.append(r)
        print(f"  {t['dataset_id']}@{t['config_name']}: {r['commit_decision']}")
        if "committed_bytes" in r:
            print(f"    bytes={r['committed_bytes']:,} rows={r['committed_rows']:,} "
                  f"sha256={r['committed_sha256'][:16]}…")
    rollup_path = DISC_DIR / "telemetry_gap_normalized_sample_commit_summary.json"
    payload = {
        "doc_version": "telemetry_gap_normalized_sample_commit_v2",
        "stage": "phase_a_normalized_sample_commit",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "max_per_sample_bytes": MAX_PER_SAMPLE_BYTES,
        "max_total_committed_bytes": MAX_TOTAL_COMMITTED_BYTES,
        "total_committed_bytes": total,
        "materialized_at_s": time.time(),
        "git_sha": _git_sha(),
        "redistribution_gate_scope": GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "datasets": results,
    }
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rollup_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    print(f"\nTotal committed: {total:,} bytes (cap 150 MB)")
    print(f"Wrote {rollup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
