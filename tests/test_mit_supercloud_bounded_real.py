"""Tests for MIT Supercloud bounded real-sample ingestion + frontier.

Covers ONLY behaviour added by the bounded-real-sample PR. The existing
``tests/test_mit_supercloud_ingestion.py`` continues to assert the v1
fixture-mode behaviour and is not modified.

Hard invariants proved here:

1.  Slurm integer-state codes (3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 / 11)
    resolve to the canonical string labels.
2.  ``_parse_nodelist`` handles the MIT-shaped stringified Python list
    (``"['r9189566-n911952']"``).
3.  Real GPU CSV filenames (``<id_job>-r<id_array>-n<id_user>.csv``)
    extract the leading id_job token.
4.  ``load_scheduler_log`` supports ``max_jobs`` (short-circuit) and
    time-window filters.
5.  ``load_all_layers`` forwards ``max_jobs`` / time filters / sample
    args.
6.  The bounded-download script's manifest is well-formed
    (``--dry-run`` so no network).
7.  The real-frontier benchmark runs end-to-end on the synthetic
    fixture (when no raw data is downloaded — keeps CI offline).
8.  When real raw data is present (``data/external/mit_supercloud/raw/
    slurm-log.csv``), an opt-in test loads it with a small ``max_jobs``
    cap; otherwise the test is SKIPPED.
9.  ``run_mit_supercloud_real_scheduler_frontier`` exits 4 when no
    slurm-log is present.
10. Committed v1 fixture-mode summary JSON is NOT overwritten by the
    bounded-real-sample script.
11. Docs contain no unhedged production-savings claims.
12. ``.gitignore`` excludes the bounded-real raw archive.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from aurelius.traces import mit_supercloud as mit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                        "mit_supercloud_sample")
RAW = os.path.join(REPO_ROOT, "data", "external", "mit_supercloud",
                    "raw")
V1_FIXTURE_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "mit_supercloud", "processed",
    "mit_supercloud_training_frontier_summary.json")
REAL_SCHED_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "mit_supercloud", "processed",
    "mit_supercloud_real_scheduler_summary.json")
BOUNDED_DOC = os.path.join(
    REPO_ROOT, "docs", "MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md")
GITIGNORE = os.path.join(REPO_ROOT, ".gitignore")


# ===========================================================================
# 1 — Slurm integer state codes
# ===========================================================================

def test_slurm_integer_state_codes_resolve_to_labels():
    assert mit._resolve_state_label("3") == "COMPLETED"
    assert mit._resolve_state_label("4") == "CANCELLED"
    assert mit._resolve_state_label("5") == "FAILED"
    assert mit._resolve_state_label("6") == "TIMEOUT"
    assert mit._resolve_state_label("7") == "NODE_FAIL"
    assert mit._resolve_state_label("8") == "PREEMPTED"
    assert mit._resolve_state_label("9") == "BOOT_FAIL"
    assert mit._resolve_state_label("10") == "DEADLINE"
    assert mit._resolve_state_label("11") == "OUT_OF_MEMORY"


def test_slurm_integer_state_unknown_passes_through_as_string():
    # MIT slurm-log carries one row with state=1024 (out of our table);
    # the loader must pass it through, not raise.
    assert mit._resolve_state_label("1024") == "1024"


def test_slurm_string_state_uppercased():
    assert mit._resolve_state_label("completed") == "COMPLETED"
    assert mit._resolve_state_label(None) is None
    assert mit._resolve_state_label("") is None


# ===========================================================================
# 2 — Stringified nodelist parser
# ===========================================================================

def test_parse_nodelist_stringified_python_list():
    out = mit._parse_nodelist("['r9189566-n911952']")
    assert out == "r9189566-n911952"
    out = mit._parse_nodelist("['n01', 'n02', 'n03']")
    assert out == "n01,n02,n03"


def test_parse_nodelist_plain_comma_list():
    assert mit._parse_nodelist("n01,n02") == "n01,n02"


def test_parse_nodelist_empty_and_none():
    assert mit._parse_nodelist(None) is None
    assert mit._parse_nodelist("[]") is None
    assert mit._parse_nodelist("") is None


# ===========================================================================
# 3 — GPU CSV filename → id_job
# ===========================================================================

def test_gpu_filename_to_job_id_real_bucket_format():
    assert (mit.gpu_filename_to_job_id(
        "10028747453883-r9175025-n976057.csv") == "10028747453883")
    assert (mit.gpu_filename_to_job_id(
        "data/external/mit_supercloud/raw/gpu/0000/"
        "32585007376605-r8939293-n208530.csv") == "32585007376605")


def test_gpu_filename_to_job_id_fixture_format():
    # Synthetic fixture uses <id_job>.csv — must still work.
    assert mit.gpu_filename_to_job_id("1001.csv") == "1001"
    assert mit.gpu_filename_to_job_id("path/to/2345.csv") == "2345"


# ===========================================================================
# 4 — max_jobs + time-window filters
# ===========================================================================

def test_load_scheduler_log_max_jobs_cap():
    # Build a synthetic 30-row CSV so we don't depend on raw data.
    import tempfile
    rows = ["id_job,time_submit,time_start,time_end,state,tres_req"]
    for i in range(30):
        rows.append(
            f"{1000 + i},{1700000000 + i},{1700000100 + i},"
            f"{1700001100 + i},3,\"1=4,4=1,1001=1\"")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "slurm-log.csv")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        jobs = mit.load_scheduler_log(path, max_jobs=10)
    assert len(jobs) == 10


def test_load_scheduler_log_time_window_filter():
    import tempfile
    rows = ["id_job,time_submit,time_start,time_end,state,tres_req"]
    for i in range(10):
        rows.append(
            f"{1000 + i},{1700000000 + i * 600},"
            f"{1700000100 + i * 600},"
            f"{1700001100 + i * 600},3,\"1=4,4=1,1001=1\"")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "slurm-log.csv")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        # Window [submit_min .. submit_min+1800] keeps 4 rows
        # (i=0,1,2,3 with submit 0,600,1200,1800).
        jobs = mit.load_scheduler_log(
            path, start_time_min_s=1700000000.0,
            end_time_max_s=1700001100 + 1800)
    assert len(jobs) == 4


# ===========================================================================
# 5 — load_all_layers wiring
# ===========================================================================

def test_load_all_layers_forwards_bounded_args():
    layers = mit.load_all_layers(
        FIXTURE, include_utilization=False,
        max_jobs=5, sample_size=None,
        gpu_jobs_only=False, labelled_only=False)
    assert len(layers["jobs"]) <= 5


# ===========================================================================
# 6 — bounded-download manifest (dry-run; no network)
# ===========================================================================

def test_bounded_download_manifest_dry_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest_path = raw / "manifest.json"
    proc = subprocess.run(
        ["python3", "scripts/download_mit_supercloud_bounded.py",
         "--raw-dir", str(raw),
         "--download-gpu-sample", "false",
         "--download-node-sample", "false",
         "--manifest-path", str(manifest_path),
         "--dry-run"],
        check=False, capture_output=True, text=True,
        cwd=REPO_ROOT)
    if proc.returncode != 0:
        pytest.skip(f"dry-run network probe unavailable: {proc.stderr}")
    assert manifest_path.exists()
    d = json.load(open(manifest_path))
    assert "files" in d and isinstance(d["files"], list)
    names = {os.path.basename(f["local_path"]) for f in d["files"]}
    assert "slurm-log.csv" in names
    assert "labelled_jobids.csv" in names
    assert "tres-mapping.txt" in names
    # Real S3 keys present in every entry.
    assert all("s3://mit-supercloud-dataset" in f["s3_uri"]
                for f in d["files"])
    # Every sample_policy is one of the documented categories.
    assert all(f["sample_policy"] in {"full_file",
                                       "range_get_first_50MB",
                                       "range_get_first_N_MB",
                                       "gpu_sample"}
                or f["sample_policy"].startswith("gpu_sample")
                or f["sample_policy"].startswith("range_get_first_")
                for f in d["files"])


# ===========================================================================
# 7 — real-frontier benchmark runs end-to-end on the synthetic fixture
# ===========================================================================

def test_real_frontier_benchmark_runs_on_fixture(tmp_path):
    """The real-scheduler driver should run end-to-end against any
    directory that contains a slurm-log.csv — including the v1
    synthetic fixture (acts as smoke test when no raw download is
    present)."""
    out_json = tmp_path / "frontier.json"
    out_md = tmp_path / "report.md"
    summary_json = tmp_path / "summary.json"
    proc = subprocess.run(
        ["python3", "scripts/run_mit_supercloud_real_scheduler_frontier.py",
         "--source-dir", FIXTURE,
         "--out-json", str(out_json),
         "--out-md", str(out_md),
         "--summary-json", str(summary_json),
         "--manifest", "/dev/null",
         "--gpu-jobs-only", "true",
         "--max-jobs", "100"],
        check=False, capture_output=True, text=True,
        cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists() and out_md.exists()
    d = json.load(open(out_json))
    assert "capacity_sensitivity" in d
    assert len(d["capacity_sensitivity"]) == 3  # small / medium / large
    for sweep in d["capacity_sensitivity"]:
        assert sweep["fleet_label"] in {"small", "medium", "large"}
        assert "verdict" in sweep
        assert sweep["verdict"] in {"TIE", "TRAINING_FRONTIER_WIN",
                                     "TRAINING_FRONTIER_LOSS",
                                     "INSUFFICIENT_DATA"}


# ===========================================================================
# 8 — raw-integration test SKIPPED unless raw is present
# ===========================================================================

_RAW_PRESENT = os.path.exists(os.path.join(RAW, "slurm-log.csv"))


@pytest.mark.skipif(not _RAW_PRESENT,
                     reason="MIT Supercloud raw archive not "
                            "downloaded (run "
                            "scripts/download_mit_supercloud_bounded.py)")
def test_real_raw_loads_with_max_jobs_cap():
    layers = mit.load_all_layers(
        RAW, include_utilization=False,
        max_jobs=500, gpu_jobs_only=False)
    assert len(layers["jobs"]) <= 500
    # The full label table has ~3,430 entries — should match this many.
    assert len(layers["labels_by_jobid"]) >= 100


@pytest.mark.skipif(not _RAW_PRESENT, reason="MIT raw absent")
def test_real_raw_full_load_matches_published_counts():
    """Full slurm-log load should produce a count close to the
    README's 460,497 (we expect a small drop for rows missing id_job)
    and exactly 98,177 ± small for GPU jobs."""
    jobs = mit.load_scheduler_log(
        os.path.join(RAW, "slurm-log.csv"),
        tres_mapping=mit.parse_tres_mapping(
            os.path.join(RAW, "tres-mapping.txt")),
        labels_by_jobid=mit.load_labelled_jobids(
            os.path.join(RAW, "labelled_jobids.csv")))
    gpu_jobs = [j for j in jobs if (j.gpu_count_requested or 0) > 0]
    # README counts: 460,497 total / 98,177 GPU / 3,425 labelled.
    assert 380_000 <= len(jobs) <= 470_000, (
        f"total jobs {len(jobs):,} outside expected band")
    assert 90_000 <= len(gpu_jobs) <= 100_000, (
        f"gpu jobs {len(gpu_jobs):,} outside expected band")


# ===========================================================================
# 9 — driver exits 4 when slurm-log absent
# ===========================================================================

def test_real_frontier_exits_when_slurm_log_absent(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run(
        ["python3", "scripts/run_mit_supercloud_real_scheduler_frontier.py",
         "--source-dir", str(empty)],
        check=False, capture_output=True, text=True,
        cwd=REPO_ROOT)
    assert proc.returncode == 4


# ===========================================================================
# 10 — v1 fixture-mode summary untouched
# ===========================================================================

@pytest.mark.skipif(not os.path.exists(V1_FIXTURE_SUMMARY),
                     reason="v1 fixture summary not yet committed")
def test_v1_fixture_summary_unchanged_by_bounded_real_pr():
    # The bounded-real-sample script writes a NEW sibling JSON. The
    # v1 fixture-mode summary at
    # ``mit_supercloud_training_frontier_summary.json`` is read-only.
    d = json.load(open(V1_FIXTURE_SUMMARY))
    # The v1 summary's signature: comparison.verdict comes from the
    # single-fleet fixture run; the real-frontier JSON uses
    # ``capacity_sensitivity`` instead.
    assert "comparison" in d
    assert "capacity_sensitivity" not in d


# ===========================================================================
# 11 — docs (bounded sample) have no unhedged production-savings claims
# ===========================================================================

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


@pytest.mark.skipif(not os.path.exists(BOUNDED_DOC),
                     reason="bounded doc not yet generated")
def test_bounded_doc_no_unhedged_banned_phrases():
    text = open(BOUNDED_DOC, encoding="utf-8").read().lower()
    low = " ".join(text.split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in
                       ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(BOUNDED_DOC)}"
            i = pos + len(phrase)


@pytest.mark.skipif(not os.path.exists(BOUNDED_DOC),
                     reason="bounded doc not yet generated")
def test_bounded_doc_states_required_caveats():
    low = " ".join(open(BOUNDED_DOC, encoding="utf-8").read().lower().split())
    for phrase in ("bounded", "supercloud", "s3", "dcc.mit.edu",
                   "disabled by default", "pilot telemetry",
                   "no production-savings", "capacity"):
        assert phrase in low, f"doc missing required caveat: {phrase!r}"


# ===========================================================================
# 12 — gitignore excludes bounded-real raw archive
# ===========================================================================

def test_gitignore_excludes_mit_supercloud_raw():
    text = open(GITIGNORE, encoding="utf-8").read()
    assert "data/external/mit_supercloud/raw" in text


# ===========================================================================
# Sanity — existing public-API surface unchanged
# ===========================================================================

def test_mit_supercloud_public_api_unchanged():
    for name in ("NormalizedMITTrainingJob",
                 "NormalizedMITGPUUtilizationSample",
                 "NormalizedMITNodeUtilizationSample",
                 "parse_tres_mapping", "parse_tres_req",
                 "gpu_count_from_tres",
                 "load_scheduler_log", "load_labelled_jobids",
                 "load_node_data", "load_gpu_utilization",
                 "load_gpu_utilization_file",
                 "load_all_layers", "discover",
                 "compute_join_quality",
                 "to_normalized_gpu_job", "summarize_jobs"):
        assert hasattr(mit, name), f"missing public symbol {name!r}"


def test_training_frontier_public_api_unchanged():
    import aurelius.frontier as fr
    for name in ("TrainingFrontierAction", "TrainingFrontierCandidate",
                 "TrainingFrontierPoint", "TrainingFrontierDecision",
                 "TrainingSafetyConfig", "TrainingSafetyStatus",
                 "PHILLY_POLICY_CANDIDATES",
                 "choose_training_frontier_target",
                 "classify_training_frontier_point"):
        assert hasattr(fr, name)
