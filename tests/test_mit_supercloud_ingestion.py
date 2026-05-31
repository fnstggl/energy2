"""Tests for the MIT Supercloud (Samsi et al., HPEC 2021) ingestion +
Training Frontier validation.

Hard invariants proved here:

1.  ``scheduler-log.csv`` fixture parses; gpu/cpu/status/queue-wait/
    duration are computed honestly.
2.  ``labelled_jobids.csv`` joins to jobs by exact ``id_job``.
3.  GPU job detection works (tres_req → gpu_count + gpu_type).
4.  Queue wait = start − submit (None when either is missing).
5.  Duration   = end − start   (None when either is missing).
6.  ``token_equivalent_work`` = ``gpu_seconds`` = gpu_count × duration.
7.  ``FAILED`` / ``CANCELLED`` / ``TIMEOUT`` flag ``is_failed=True``.
8.  ``tres-mapping.txt`` parses (CSV/TSV/markdown-table tolerant).
9.  Per-job GPU utilization CSVs parse.
10. ``node-data.csv`` parses.
11. Unknown fields stay ``None`` — never silently zero-filled.
12. ``compute_join_quality`` produces a structured matrix with
    confidence labels.
13. Per-job GPU utilization is NOT claimed without a valid file-name
    join.
14. Normalized MIT jobs round-trip into ``NormalizedGPUJob``.
15. Training Frontier v1 runs end-to-end on the fixture.
16. Raw-integration test is SKIPPED when no raw archive is present.
17. Docs contain no unhedged production-savings claims.
18. Existing Training Frontier tests still pass (proxied — public API
    unchanged).
19. Existing Philly + Alibaba GPU tests still pass (proxied — those
    trace modules are imported but not modified).
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from aurelius.frontier import (
    TrainingFrontierAction,
    TrainingFrontierCandidate,
    TrainingFrontierPoint,
    TrainingSafetyConfig,
    TrainingSafetyStatus,
    choose_training_frontier_target,
)
from aurelius.traces import mit_supercloud as mit
from aurelius.traces.schema import NormalizedGPUJob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                        "mit_supercloud_sample")
RAW = os.path.join(REPO_ROOT, "data", "external", "mit_supercloud",
                    "raw")
RESULTS_DOC = os.path.join(REPO_ROOT, "docs",
                            "MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md")
RESULTS_JSON = os.path.join(
    REPO_ROOT, "data", "external", "mit_supercloud", "processed",
    "mit_supercloud_training_frontier_summary.json")


# ===========================================================================
# 1 — scheduler log parses
# ===========================================================================

def test_scheduler_log_parses():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    jobs = layers["jobs"]
    assert jobs, "expected at least one MIT job in fixture"
    j = next(j for j in jobs if j.job_id == "1001")
    assert j.submit_time_s == 1700000000.0
    assert j.start_time_s == 1700000060.0
    assert j.end_time_s == 1700001260.0
    assert j.queue_wait_s == 60.0
    assert j.duration_s == 1200.0
    assert j.gpu_count_requested == 2  # 1001=2
    assert j.gpu_type == "gpu:tesla"


# ===========================================================================
# 2 — label join
# ===========================================================================

def test_label_join_exact_job_id():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    labels = layers["labels_by_jobid"]
    assert labels.get("1001") == "VGG"
    assert labels.get("1003") == "ResNet"
    assert labels.get("1008") == "U-Net"
    # Unlabelled jobs stay unlabelled.
    assert "1002" not in labels
    # And the merge is applied during scheduler-log loading.
    j = next(j for j in layers["jobs"] if j.job_id == "1001")
    assert j.workload_label == "VGG"
    assert j.model_family == "VGG"


# ===========================================================================
# 3 — GPU job detection (tres_req → gpu_count + type)
# ===========================================================================

def test_gpu_job_detection():
    n2_tesla = mit.gpu_count_from_tres("1=4,2=16000,4=1,1001=1")
    assert n2_tesla == (1, "gpu:tesla")
    n4_volta = mit.gpu_count_from_tres("1=16,2=64000,4=2,1002=4")
    assert n4_volta == (4, "gpu:volta")
    cpu_only = mit.gpu_count_from_tres("1=4,2=8000,4=1")
    assert cpu_only == (0, None)
    none_in = mit.gpu_count_from_tres(None)
    assert none_in == (0, None)


# ===========================================================================
# 4-6 — queue wait, duration, gpu_seconds
# ===========================================================================

def test_queue_wait_and_duration_are_computed_correctly():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    j = next(j for j in layers["jobs"] if j.job_id == "1003")
    # submit=1700002000, start=1700002100, end=1700005700
    assert j.queue_wait_s == 100.0
    assert j.duration_s == 3600.0


def test_gpu_seconds_and_token_equivalent_work_match():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    j = next(j for j in layers["jobs"] if j.job_id == "1008")
    # 8 GPUs × 7200s
    assert j.gpu_seconds == 8 * 7200.0
    assert j.token_equivalent_work == j.gpu_seconds


# ===========================================================================
# 7 — failure flags
# ===========================================================================

def test_failed_and_cancelled_set_is_failed():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    failed = next(j for j in layers["jobs"] if j.job_id == "1004")
    assert failed.is_failed
    cancelled = next(j for j in layers["jobs"] if j.job_id == "1007")
    assert cancelled.is_failed
    completed = next(j for j in layers["jobs"] if j.job_id == "1001")
    assert not completed.is_failed


# ===========================================================================
# 8 — TRES mapping parse
# ===========================================================================

def test_tres_mapping_parses_tsv_format():
    m = mit.parse_tres_mapping(os.path.join(FIXTURE, "tres-mapping.txt"))
    assert m[1] == "cpu" and m[2] == "mem"
    assert m[1001] == "gpu:tesla"
    assert m[1002] == "gpu:volta"


def test_tres_mapping_default_when_file_absent(tmp_path):
    m = mit.parse_tres_mapping(str(tmp_path / "missing.txt"))
    # Default ships the README mapping verbatim.
    assert m[1001] == "gpu:tesla"
    assert m[1002] == "gpu:volta"


def test_parse_tres_req_returns_resource_dict():
    parsed = mit.parse_tres_req("1=8,2=32000,4=1,1001=2")
    assert parsed["cpu"] == 8.0
    assert parsed["mem"] == 32000.0
    assert parsed["gpu:tesla"] == 2.0
    # Empty / None input returns {}.
    assert mit.parse_tres_req(None) == {}
    assert mit.parse_tres_req("") == {}


# ===========================================================================
# 9 — GPU utilization parse
# ===========================================================================

def test_gpu_utilization_csv_parses():
    samples = mit.load_gpu_utilization_file(
        os.path.join(FIXTURE, "gpu", "00", "1001.csv"))
    assert samples
    # File name is "1001.csv" → job_id=1001 (the exact-name join).
    assert all(s.job_id == "1001" for s in samples)
    # The first row carries non-None utilization + memory metrics.
    first = samples[0]
    assert first.gpu_utilization_pct is not None
    assert first.gpu_memory_used_mib is not None


# ===========================================================================
# 10 — node-data parse
# ===========================================================================

def test_node_data_csv_parses():
    samples = mit.load_node_data(os.path.join(FIXTURE, "node-data.csv"))
    assert samples
    n01 = [s for s in samples if s.node_id == "node01"]
    assert n01
    assert n01[0].system_load == 12.5
    assert n01[0].users == 2
    assert n01[0].memory_total_mib == 128000.0


# ===========================================================================
# 11 — None preserved for unknowns
# ===========================================================================

def test_unknown_fields_remain_none_not_zero():
    j = mit.NormalizedMITTrainingJob(job_id="x", submit_time_s=None)
    for f in ("start_time_s", "end_time_s", "queue_wait_s", "duration_s",
              "gpu_count_requested", "gpu_type", "node_count", "nodes",
              "user_or_group", "workload_label", "model_family",
              "tres_req_raw", "token_equivalent_work", "gpu_seconds",
              "memory_requested_mib"):
        assert getattr(j, f) is None, \
            f"{f} should default to None; got {getattr(j, f)!r}"


def test_gpu_sample_unknown_fields_remain_none():
    s = mit.NormalizedMITGPUUtilizationSample(
        timestamp_s=0.0, job_id=None, node_id=None, gpu_id=None)
    assert s.gpu_utilization_pct is None
    assert s.gpu_memory_used_mib is None
    assert s.gpu_memory_total_mib is None
    assert s.power_draw_w is None
    assert s.temperature_gpu_c is None


# ===========================================================================
# 12 — join-quality matrix
# ===========================================================================

def test_join_quality_matrix_classifies_each_join():
    layers = mit.load_all_layers(FIXTURE, include_utilization=True,
                                  max_util_files=10)
    joins = mit.compute_join_quality(
        layers["jobs"], labels_by_jobid=layers["labels_by_jobid"],
        gpu_samples=layers["gpu_samples"],
        node_samples=layers["node_samples"])
    by_name = {j["join_name"]: j for j in joins["joins"]}
    assert by_name["label_to_job"]["join_kind"] == "exact_job_id_join"
    assert by_name["label_to_job"]["confidence"] == "high"
    assert by_name["gpu_util_to_job"]["join_kind"] == "exact_job_id_join"
    assert by_name["gpu_util_to_job"]["matched_right"] >= 1
    assert by_name["node_util_to_job"]["join_kind"] == "node_time_join"
    assert by_name["node_util_to_job"]["confidence"] in ("medium", "none")


# ===========================================================================
# 13 — per-job utilization NOT claimed without a valid join
# ===========================================================================

def test_gpu_join_is_none_when_utilization_not_loaded():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    joins = mit.compute_join_quality(
        layers["jobs"], labels_by_jobid=layers["labels_by_jobid"],
        gpu_samples=None, node_samples=None)
    by_name = {j["join_name"]: j for j in joins["joins"]}
    assert by_name["gpu_util_to_job"]["join_kind"] == "no_join"
    assert by_name["gpu_util_to_job"]["confidence"] == "none"
    assert by_name["gpu_util_to_job"]["matched_right"] == 0


# ===========================================================================
# 14 — NormalizedMITTrainingJob → NormalizedGPUJob round-trip
# ===========================================================================

def test_mit_to_normalized_gpu_job_round_trip():
    layers = mit.load_all_layers(FIXTURE, include_utilization=False)
    mit_job = next(j for j in layers["jobs"] if j.job_id == "1001")
    gpu_job = mit.to_normalized_gpu_job(mit_job)
    assert isinstance(gpu_job, NormalizedGPUJob)
    assert gpu_job.job_id == "1001"
    assert gpu_job.gpu_count == 2
    assert gpu_job.duration_s == 1200.0
    assert gpu_job.queue_wait_s == 60.0
    assert gpu_job.workload_type == mit.WORKLOAD_TYPE


# ===========================================================================
# 15 — Training Frontier v1 runs on the fixture
# ===========================================================================

def test_training_frontier_runs_on_mit_fixture():
    """The benchmark script's end-to-end path: discover → ingest →
    schedule → frontier-point → controller."""
    from scripts import run_mit_supercloud_training_frontier as rmit
    layers = mit.load_all_layers(FIXTURE, gpu_jobs_only=True)
    gpu_jobs = [mit.to_normalized_gpu_job(j) for j in layers["jobs"]]
    fleet = rmit._synth_fleet(gpu_jobs, gpus_per_node=8,
                               node_overhead_factor=5.0)
    from aurelius.traces import gpu_scheduling as gs
    bt = gs.run_backtest(gpu_jobs, fleet)
    safety = TrainingSafetyConfig(
        max_gang_scheduling_failure_pct=None)  # MIT does not measure
    points = [rmit._point_from_sched_policy(
        name, sched, n_scheduled=bt.n_scheduled, safety_config=safety)
        for name, sched in bt.policy_results.items()]
    assert points, "expected at least one frontier point"
    dec = choose_training_frontier_target(points)
    assert dec.action in {TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER,
                          TrainingFrontierAction.KEEP_CURRENT_POLICY,
                          TrainingFrontierAction.LOWER_PACKING_PRESSURE,
                          TrainingFrontierAction.RESERVE_FOR_LARGE_JOBS,
                          TrainingFrontierAction.INSUFFICIENT_TELEMETRY}


def test_committed_benchmark_summary_exists_and_is_well_formed():
    assert os.path.exists(RESULTS_JSON)
    d = json.load(open(RESULTS_JSON))
    assert "discovery" in d and "trace_summary" in d
    assert "join_quality" in d and "frontier_rows" in d
    assert "comparison" in d and "alpha_finding" in d
    assert d["config"]["real_execution_disabled_by_default"] is True
    assert d["config"]["execution_mode_default"] == "shadow"


# ===========================================================================
# 16 — raw-integration test is skipped if archive absent
# ===========================================================================

@pytest.mark.skipif(
    not (os.path.isdir(RAW) and any(
        os.path.exists(os.path.join(RAW, f))
        for f in mit.SCHEDULER_LOG_FILES)),
    reason="MIT Supercloud raw archive not present (download from "
            "https://dcc.mit.edu/data)")
def test_raw_ingestion_when_archive_present():
    layers = mit.load_all_layers(RAW, include_utilization=False,
                                  sample_size=200)
    assert layers["jobs"]


# ===========================================================================
# 17 — docs check
# ===========================================================================

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def test_results_doc_no_unhedged_banned_phrases():
    assert os.path.exists(RESULTS_DOC)
    text = open(RESULTS_DOC, encoding="utf-8").read().lower()
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
                f"unhedged '{phrase}' in {os.path.basename(RESULTS_DOC)}"
            i = pos + len(phrase)


def test_results_doc_states_required_caveats():
    text = open(RESULTS_DOC, encoding="utf-8").read().lower()
    low = " ".join(text.split())
    for phrase in ("synthetic", "supercloud", "dcc.mit.edu",
                   "real-cluster execution",
                   "disabled by default", "pilot telemetry"):
        assert phrase in low, f"doc missing required caveat: {phrase!r}"


# ===========================================================================
# 18-19 — existing public APIs still importable
# ===========================================================================

def test_training_frontier_public_api_unchanged():
    import aurelius.frontier as fr
    for required in ("TrainingFrontierAction",
                     "TrainingFrontierCandidate",
                     "TrainingFrontierPoint",
                     "TrainingFrontierDecision",
                     "TrainingSafetyConfig",
                     "TrainingSafetyStatus",
                     "TRAINING_FRONTIER_ACTIONS",
                     "ALL_TRAINING_VETOES",
                     "classify_training_frontier_point",
                     "is_training_frontier_point_safe",
                     "choose_training_frontier_target",
                     "execute_training_frontier_decision",
                     "estimate_philly_training_frontier",
                     "estimate_alibaba_gpu_training_frontier",
                     "PHILLY_POLICY_CANDIDATES",
                     "ALIBABA_POLICY_CANDIDATES",
                     "TrainingWorkloadProfile"):
        assert hasattr(fr, required)


def test_philly_and_alibaba_trace_modules_importable():
    import aurelius.traces.philly  # noqa: F401
    import aurelius.traces.alibaba_gpu  # noqa: F401
    import aurelius.traces.gpu_scheduling  # noqa: F401
    import aurelius.traces.gpu_packing  # noqa: F401


# ===========================================================================
# Bonus — ingestion CLI runs and emits a JSON summary
# ===========================================================================

def test_ingest_cli_emits_summary_json(tmp_path):
    out = tmp_path / "summary.json"
    proc = subprocess.run(
        ["python3", "scripts/ingest_mit_supercloud.py",
         "--source-dir", FIXTURE,
         "--summary-json", str(out),
         "--include-utilization", "true",
         "--max-util-files", "10"],
        check=False, capture_output=True, text=True,
        cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    d = json.load(open(out))
    assert d["trace_summary"]["job_count"] > 0
    assert "joins" in d["join_quality"]


def test_ingest_cli_print_only_instructions_exits_zero(tmp_path):
    proc = subprocess.run(
        ["python3", "scripts/ingest_mit_supercloud.py",
         "--print-only-instructions"],
        check=False, capture_output=True, text=True,
        cwd=REPO_ROOT)
    assert proc.returncode == 0
    assert "dcc.mit.edu" in proc.stdout
    assert "NOT" in proc.stdout  # download disclaimer surfaced
