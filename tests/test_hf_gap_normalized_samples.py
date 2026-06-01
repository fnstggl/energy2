"""Tests for the committed normalized analysis samples from PR #129 (PHASE A).

Covers:
- no raw / analysis_sample data committed (gitignore still enforced)
- every committed normalized sample is ≤ 50 MB
- sum of committed normalized samples ≤ 150 MB
- each committed sample has checksum / provenance / field_quality recorded
  in the matching summary.json
- each sample is loadable JSONL and every row's schema matches the
  normalized_schema field of its summary
- license_redistribution_status is one of the permissive labels OR the
  dataset is explicitly skipped with reason

Audit-only: tests read committed artifacts only. They do NOT hit the
HF API and do NOT download any data.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"

# Datasets PR #129 ingested. The 4 with permissive licenses commit a
# normalized sample. The 5th (prefixbench) has no license → skipped.
ALL_DATASETS = [
    ("lzzmm/BurstGPT", "burstgpt_1_full", True),
    ("lsliwko/google-cluster-data-2019-sorted-by-timestamp",
     "instance_events_shard0", True),
    ("sammshen/lmcache-agentic-traces", "train_shard4", True),
    ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head", True),
    ("jaytonde05/prefixbench", "prefixbench_all", False),  # license: unspecified
]

COMMITTED_DATASETS = [(d, c) for d, c, commit in ALL_DATASETS if commit]
SKIPPED_DATASETS = [(d, c) for d, c, commit in ALL_DATASETS if not commit]

MAX_PER_SAMPLE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_COMMITTED_BYTES = 150 * 1024 * 1024

PERMISSIVE_LICENSES = {
    "permissive_apache_2_0",
    "permissive_mit",
    "permissive_cc_by_4_0",
}


def _processed_dir(dataset_id: str, config: str) -> Path:
    return HF_DIR / dataset_id.replace("/", "__") / config / "processed"


def _load_summary(dataset_id: str, config: str) -> dict:
    with open(_processed_dir(dataset_id, config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw data committed ──────────────────────


def test_no_raw_files_tracked_by_git():
    """``git ls-files`` must not list any raw download path or any
    ``analysis_sample.jsonl`` (which is the unbounded sibling of the
    committed normalized sample)."""
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed (gitignore broken): {analysis_committed}"
    )


# ───────────────────────── 2. Per-sample bounds ──────────────────────────


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS)
def test_committed_normalized_sample_exists_and_under_50_mb(
    dataset_id: str, config: str,
) -> None:
    pd = _processed_dir(dataset_id, config)
    s = _load_summary(dataset_id, config)
    rel = s.get("committed_normalized_sample_path")
    assert rel, f"{dataset_id}@{config}: summary missing committed_normalized_sample_path"
    path = REPO_ROOT / rel
    assert path.exists(), f"committed sample {path} does not exist"
    sz = path.stat().st_size
    assert sz == s["committed_normalized_sample_bytes"], (
        f"size on disk ({sz}) != summary committed_normalized_sample_bytes "
        f"({s['committed_normalized_sample_bytes']})"
    )
    assert 0 < sz <= MAX_PER_SAMPLE_BYTES, (
        f"{rel} size {sz}B is outside (0, 50 MB] bound"
    )


def test_total_committed_normalized_bytes_under_150_mb():
    total = 0
    for dataset_id, config in COMMITTED_DATASETS:
        s = _load_summary(dataset_id, config)
        total += int(s.get("committed_normalized_sample_bytes") or 0)
    assert 0 < total <= MAX_TOTAL_COMMITTED_BYTES, (
        f"total committed bytes {total:,} exceeds 150 MB cap "
        f"or is empty"
    )


# ───────────────────────── 3. Checksum / provenance ─────────────────────


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS)
def test_committed_sample_sha256_matches_disk(dataset_id: str, config: str) -> None:
    s = _load_summary(dataset_id, config)
    rel = s["committed_normalized_sample_path"]
    path = REPO_ROOT / rel
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    on_disk = h.hexdigest()
    assert on_disk == s["committed_normalized_sample_sha256"], (
        f"{rel}: sha256 mismatch (disk={on_disk[:12]}… "
        f"summary={s['committed_normalized_sample_sha256'][:12]}…)"
    )


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS + SKIPPED_DATASETS)
def test_summary_records_provenance_and_field_quality(
    dataset_id: str, config: str,
) -> None:
    s = _load_summary(dataset_id, config)
    assert s.get("provenance"), f"{dataset_id}@{config}: missing provenance"
    fq = s.get("field_quality")
    assert isinstance(fq, dict) and fq, (
        f"{dataset_id}@{config}: missing or empty field_quality"
    )
    assert isinstance(s.get("limitations"), list) and s["limitations"], (
        f"{dataset_id}@{config}: missing limitations"
    )
    assert s.get("raw_committed") is False, (
        f"{dataset_id}@{config}: raw_committed must be explicitly False"
    )
    assert s.get("license_redistribution_status"), (
        f"{dataset_id}@{config}: missing license_redistribution_status"
    )


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS)
def test_committed_dataset_uses_permissive_license_label(
    dataset_id: str, config: str,
) -> None:
    s = _load_summary(dataset_id, config)
    assert s["license_redistribution_status"] in PERMISSIVE_LICENSES, (
        f"{dataset_id}@{config}: committed but license_redistribution_status "
        f"{s['license_redistribution_status']!r} is not in "
        f"{PERMISSIVE_LICENSES}"
    )


@pytest.mark.parametrize("dataset_id,config", SKIPPED_DATASETS)
def test_skipped_dataset_records_skip_reason(dataset_id: str, config: str) -> None:
    s = _load_summary(dataset_id, config)
    assert s.get("committed_normalized_sample_path") in (None, ""), (
        f"{dataset_id}@{config}: license is non-permissive but a committed "
        f"path is recorded"
    )
    assert s["license_redistribution_status"] not in PERMISSIVE_LICENSES, (
        f"{dataset_id}@{config}: marked as skipped but label is permissive"
    )
    assert s.get("committed_normalized_sample_reason_skipped"), (
        f"{dataset_id}@{config}: skip reason must be recorded"
    )


# ───────────────────────── 4. Loadable + schema-valid ───────────────────


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS)
def test_committed_sample_is_valid_jsonl(dataset_id: str, config: str) -> None:
    s = _load_summary(dataset_id, config)
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    n = 0
    keys_union: set = set()
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"{path}:{i + 1}: invalid JSON: {e}"
                )
            assert isinstance(obj, dict), f"{path}:{i + 1}: row is not a dict"
            keys_union |= set(obj.keys())
            n += 1
    assert n == s["committed_normalized_sample_rows"], (
        f"row count mismatch: disk={n} summary={s['committed_normalized_sample_rows']}"
    )
    # Every key seen in the data must appear in summary.normalized_schema.
    expected = set(s.get("normalized_schema") or [])
    unexpected = keys_union - expected
    assert not unexpected, (
        f"{dataset_id}@{config}: columns present in committed sample but "
        f"missing from normalized_schema: {sorted(unexpected)}"
    )


# ───────────────────────── 5. No raw / no prompt-text leakage ──────────


# Forbidden raw-text column names. A normalized sample must NOT contain
# any of these — they are upstream raw fields that would carry prompt /
# completion content if not normalized.
RAW_TEXT_FIELDS = frozenset({
    "prompt", "completion", "input", "output",
    "message", "messages", "content",
    # CC-traces raw nested keys
    "requests", "hash_ids",
})


@pytest.mark.parametrize("dataset_id,config", COMMITTED_DATASETS)
def test_committed_sample_contains_no_raw_text_columns(
    dataset_id: str, config: str,
) -> None:
    s = _load_summary(dataset_id, config)
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    forbidden_hits: dict[str, list[int]] = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in obj.keys():
                if k in RAW_TEXT_FIELDS:
                    forbidden_hits.setdefault(k, []).append(i + 1)
            if i > 1000:  # spot-check is enough; schema test covered the rest
                break
    assert not forbidden_hits, (
        f"{dataset_id}@{config}: committed sample contains raw-text columns "
        f"that should have been normalized away: {forbidden_hits}"
    )


# ───────────────────────── 6. Cross-dataset rollup ──────────────────────


def test_commit_rollup_exists_and_matches_summaries():
    rollup_path = DISC_DIR / "telemetry_gap_normalized_sample_commit_summary.json"
    assert rollup_path.exists(), f"missing {rollup_path}"
    with open(rollup_path) as fh:
        rollup = json.load(fh)
    assert rollup["modifies_robust_energy_engine"] is False
    assert rollup["modifies_controllers_or_defaults"] is False
    assert rollup["production_claim"] is False
    assert rollup["max_per_sample_bytes"] == MAX_PER_SAMPLE_BYTES
    assert rollup["max_total_committed_bytes"] == MAX_TOTAL_COMMITTED_BYTES
    # Cross-check total against summaries.
    committed_sum = 0
    for dataset_id, config in COMMITTED_DATASETS:
        s = _load_summary(dataset_id, config)
        committed_sum += int(s["committed_normalized_sample_bytes"])
    assert rollup["total_committed_bytes"] == committed_sum
    # Skipped datasets must be in the rollup with skip decisions.
    decisions = {(e["dataset_id"], e["config_name"]): e["commit_decision"]
                 for e in rollup["datasets"]}
    for d, c in COMMITTED_DATASETS:
        assert decisions.get((d, c)) == "COMMITTED", (
            f"{d}@{c}: rollup says {decisions.get((d, c))!r}, expected COMMITTED"
        )
    for d, c in SKIPPED_DATASETS:
        verdict = decisions.get((d, c)) or ""
        assert verdict.startswith("SKIPPED"), (
            f"{d}@{c}: rollup says {verdict!r}, expected SKIPPED*"
        )
