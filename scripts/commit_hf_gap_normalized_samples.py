#!/usr/bin/env python3
"""Materialize bounded normalized analysis samples for the PR #129 datasets.

PR #129 ingested raw → normalized but only committed tiny fixtures. This
script promotes the gitignored ``processed/analysis_sample.jsonl`` to a
committed ``processed/normalized_sample.jsonl`` for datasets whose
license permits redistribution.

Policy (binding):
- raw downloads remain gitignored
- each committed normalized sample ≤ 50 MB
- sum of all committed normalized samples in this run ≤ 150 MB
- ``license_redistribution_status`` must be one of:
    ``permissive_apache_2_0`` / ``permissive_mit`` / ``permissive_cc_by_4_0``
- samples with ``license_redistribution_status == "unspecified"`` are
  recorded but NOT copied to a committed file.

Summary additions per dataset:
- committed_normalized_sample_path / _bytes / _rows / _sha256
- license_redistribution_status
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

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"

MAX_PER_SAMPLE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_TOTAL_COMMITTED_BYTES = 150 * 1024 * 1024  # 150 MB

# Per-dataset license verdict + commit decision. Verified against each
# dataset's HF card (frontmatter `license:` field) or, where the HF card
# was ambiguous (BurstGPT), against the LICENSE file fetched directly
# from the dataset repo.
TARGETS = [
    {
        "dataset_id": "lzzmm/BurstGPT",
        "config_name": "burstgpt_1_full",
        "license_redistribution_status": "permissive_cc_by_4_0",
        "license_source": (
            "LICENSE file at https://huggingface.co/datasets/lzzmm/BurstGPT/blob/main/LICENSE "
            "= 'Attribution 4.0 International' (CC-BY-4.0)"
        ),
        "commit_sample": True,
    },
    {
        "dataset_id": "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
        "config_name": "instance_events_shard0",
        "license_redistribution_status": "permissive_cc_by_4_0",
        "license_source": (
            "Mirror of github.com/google/cluster-data, released by Google "
            "under CC-BY-4.0; HF redistribution preserves the same terms"
        ),
        "commit_sample": True,
    },
    {
        "dataset_id": "sammshen/lmcache-agentic-traces",
        "config_name": "train_shard4",
        "license_redistribution_status": "permissive_mit",
        "license_source": "HF card frontmatter license: mit",
        "commit_sample": True,
    },
    {
        "dataset_id": "semianalysisai/cc-traces-weka-no-subagents-051226",
        "config_name": "traces_head",
        "license_redistribution_status": "permissive_apache_2_0",
        "license_source": "HF card frontmatter license: apache-2.0",
        "commit_sample": True,
    },
    {
        "dataset_id": "jaytonde05/prefixbench",
        "config_name": "prefixbench_all",
        "license_redistribution_status": "unspecified_no_committed_sample",
        "license_source": (
            "HF card frontmatter has no `license:` key; README provides no "
            "redistribution statement. Conservative: do NOT commit a "
            "normalized sample."
        ),
        "commit_sample": False,
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


def materialize(target: dict, total_committed_so_far: int) -> tuple[dict, int]:
    pd = _processed_dir(target["dataset_id"], target["config_name"])
    summary_path = pd / "summary.json"
    if not summary_path.exists():
        return {"target": target, "audit_status": "summary_missing"}, total_committed_so_far
    with open(summary_path) as fh:
        summary = json.load(fh)

    analysis = pd / "analysis_sample.jsonl"
    committed_path = pd / "normalized_sample.jsonl"
    result = {
        "dataset_id": target["dataset_id"],
        "config_name": target["config_name"],
        "license_redistribution_status": target["license_redistribution_status"],
        "license_source": target["license_source"],
        "commit_decision": "skip",
    }

    # Always update summary metadata, regardless of commit decision.
    summary["license_redistribution_status"] = target["license_redistribution_status"]
    summary["license_redistribution_source"] = target["license_source"]
    summary["raw_committed"] = False

    if not target["commit_sample"]:
        summary["committed_normalized_sample_path"] = None
        summary["committed_normalized_sample_bytes"] = 0
        summary["committed_normalized_sample_rows"] = 0
        summary["committed_normalized_sample_sha256"] = None
        summary["committed_normalized_sample_reason_skipped"] = (
            f"license_redistribution_status="
            f"{target['license_redistribution_status']}; "
            "no committed normalized sample"
        )
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        result["commit_decision"] = "SKIPPED (license)"
        return result, total_committed_so_far

    if not analysis.exists():
        result["commit_decision"] = "SKIPPED (analysis_sample missing — re-run scripts/ingest_hf_gap_datasets.py)"
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
    results = []
    total = 0
    for t in TARGETS:
        r, total = materialize(t, total)
        results.append(r)
        print(f"  {t['dataset_id']}@{t['config_name']}: {r['commit_decision']}")
        if "committed_bytes" in r:
            print(f"    bytes={r['committed_bytes']:,} rows={r['committed_rows']:,} "
                  f"sha256={r['committed_sha256'][:16]}…")
    rollup_path = DISC_DIR / "telemetry_gap_normalized_sample_commit_summary.json"
    payload = {
        "doc_version": "telemetry_gap_normalized_sample_commit_v1",
        "stage": "phase_a_normalized_sample_commit",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "max_per_sample_bytes": MAX_PER_SAMPLE_BYTES,
        "max_total_committed_bytes": MAX_TOTAL_COMMITTED_BYTES,
        "total_committed_bytes": total,
        "materialized_at_s": time.time(),
        "git_sha": _git_sha(),
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
