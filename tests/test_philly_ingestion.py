"""Tests for Microsoft Philly ingestion + the temporal scheduling backtest.

Unit tests use ONLY ``tests/fixtures/philly_sample/`` and never touch the
network or the full (~6.6 GB) trace. The full-trace backtest is integration-only
and skipped when the raw files are absent.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces import gpu_scheduling as gs  # noqa: E402
from aurelius.traces import philly  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "philly_sample")
FIX_JOBS = os.path.join(FIX, "cluster_job_log.json")
FIX_MACH = os.path.join(FIX, "cluster_machine_list.csv")
RAW_JOBS = os.path.join(REPO_ROOT, "data", "external", "philly", "raw",
                        "cluster_job_log")
RAW_MACH = os.path.join(REPO_ROOT, "data", "external", "philly", "raw",
                        "cluster_machine_list")
RESULTS_MD = os.path.join(REPO_ROOT, "docs", "PHILLY_BACKTEST_RESULTS.md")
PUBLIC_DOC = os.path.join(REPO_ROOT, "docs", "PUBLIC_TRACE_BACKTESTS.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


# --- 1. Fixture parses ------------------------------------------------------

def test_sample_fixture_parses():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    nodes = philly.load_machines(FIX_MACH)
    assert len(jobs) > 0 and len(nodes) > 0
    assert all(j.workload_type == "training" for j in jobs)


# --- 2. JSON job-log schema validates ---------------------------------------

def test_schema_rejects_non_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(TraceSchemaError):
        philly.load_jobs(str(bad))


def test_schema_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"jobid": "x"}]))  # missing status/submitted_time/attempts
    with pytest.raises(TraceSchemaError):
        philly.load_jobs(str(bad))


# --- 3. Times parse correctly -----------------------------------------------

def test_time_parsing():
    t0 = philly.parse_time_s("2017-10-07 01:00:00")
    t1 = philly.parse_time_s("2017-10-07 01:00:30")
    assert t1 - t0 == 30.0
    assert philly.parse_time_s("") is None
    assert philly.parse_time_s(None) is None


def test_submit_start_end_duration():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    j0 = next(j for j in jobs if j.job_id == "job-000")
    # job-000: submit 01:00:00, attempt start +30s, dur 1800
    assert j0.submit_time_s is not None
    assert j0.start_time_s - j0.submit_time_s == 30.0
    assert j0.duration_s == 1800.0
    assert j0.queue_wait_s == 30.0


# --- 4. Attempts map to placement nodes/GPUs --------------------------------

def test_attempts_map_to_placement():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    j0 = next(j for j in jobs if j.job_id == "job-000")
    assert j0.placement_nodes is not None    # ip from attempt detail
    assert j0.placement_gpus is not None     # gpu ids from attempt detail
    # multi-attempt job: duration spans first start -> last end
    j13 = next(j for j in jobs if j.job_id == "job-013")
    assert j13.duration_s and j13.duration_s > 0


def test_empty_attempts_handled():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    j = next((x for x in jobs if x.job_id == "job-noattempt"), None)
    assert j is not None
    assert j.gpu_count == 0
    assert j.start_time_s is None and j.duration_s is None


# --- 5. GPU count maps correctly --------------------------------------------

def test_gpu_count_from_attempt_detail():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    by = {j.job_id: j for j in jobs}
    assert by["job-000"].gpu_count == 1
    assert by["job-004"].gpu_count == 8        # spec idx 4 = 8-GPU
    assert by["job-031"].gpu_count == 16       # the 16-GPU job
    assert by["job-000"].gpu_milli is None     # whole-GPU (no sharing)


# --- 6. Failed/killed handled honestly --------------------------------------

def test_failed_killed_include_exclude():
    incl = philly.load_jobs(FIX_JOBS, include_failed=True)
    excl = philly.load_jobs(FIX_JOBS, include_failed=False)
    assert sum(1 for j in incl if j.is_failed) > 0
    assert all(not j.is_failed for j in excl)
    assert len(excl) < len(incl)
    # is_failed covers BOTH Failed and Killed
    statuses = {j.status for j in incl if j.is_failed}
    assert statuses <= {"Failed", "Killed"}
    assert "Killed" in statuses and "Failed" in statuses


def test_attempt_analysis():
    a = philly.analyze_attempts(FIX_JOBS)
    assert a["passed"] == 28
    assert a["failed"] == 3 and a["killed"] == 2
    assert a["multi_attempt_jobs"] >= 1
    assert a["total_retries"] >= 1


# --- 7. Machine inventory parses --------------------------------------------

def test_machine_inventory_parses():
    nodes = philly.load_machines(FIX_MACH)
    assert sum(n.gpu_count for n in nodes) == 28
    assert len({n.gpu_model for n in nodes}) > 1   # GPU-<mem> labels
    assert all(n.gpu_count > 0 for n in nodes)


def test_machine_list_missing_gpu_col(tmp_path):
    bad = tmp_path / "m.csv"
    bad.write_text("machineId,foo\nm1,3\n")
    with pytest.raises(TraceSchemaError):
        philly.load_machines(str(bad))


# --- 8. Normalized jobs generate simulator arrivals -------------------------

def test_jobs_generate_scheduler_arrivals():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    nodes = philly.load_machines(FIX_MACH)
    res = gs.run_scheduling(jobs, nodes, "best_fit")
    assert res.completed_jobs > 0
    assert res.goodput_gpu_seconds > 0
    assert res.makespan_s > 0


# --- 9. Baselines run deterministically -------------------------------------

def test_baselines_deterministic():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    nodes = philly.load_machines(FIX_MACH)
    for policy in gs.SCHEDULING_POLICIES:
        r1 = gs.run_scheduling(jobs, nodes, policy)
        r2 = gs.run_scheduling(jobs, nodes, policy)
        assert r1.summary() == r2.summary(), f"{policy} not deterministic"


# --- 10. FIFO is not the headline -------------------------------------------

def test_fifo_not_headline():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    nodes = philly.load_machines(FIX_MACH)
    result = gs.run_backtest(jobs, nodes)
    assert result.outcome.headline != "fifo"
    assert result.outcome.headline in gs.HEADLINE_CANDIDATES
    assert result.to_summary_dict()["headline_is_scheduling_baseline"] is True
    # backfill policies must beat naive head-of-line FIFO on goodput/$
    fifo = result.policy_results["fifo"]
    head = result.policy_results[result.outcome.headline]
    assert (head.goodput_per_dollar or 0) > (fifo.goodput_per_dollar or 0)


# --- 11. Backtest deterministic under fixed seed ----------------------------

def test_backtest_deterministic_seed():
    j1 = philly.load_jobs(FIX_JOBS, sample_size=20, seed=5, include_failed=True)
    j2 = philly.load_jobs(FIX_JOBS, sample_size=20, seed=5, include_failed=True)
    assert [j.to_dict() for j in j1] == [j.to_dict() for j in j2]
    nodes = philly.load_machines(FIX_MACH)
    b1 = gs.run_backtest(j1, nodes)
    b2 = gs.run_backtest(j2, nodes)
    assert b1.to_summary_dict() == b2.to_summary_dict()


def test_scheduler_reports_pressure_diagnostics():
    jobs = philly.load_jobs(FIX_JOBS, include_failed=True)
    nodes = philly.load_machines(FIX_MACH)
    r = gs.run_scheduling(jobs, nodes, "constraint_aware")
    d = r.summary()
    for key in ("queue_wait_s_p95", "queue_wait_s_p99", "utilization_mean_pct",
                "fragmentation_block_events", "backfill_placements",
                "wait_by_size_class", "starvation_events", "mean_slowdown"):
        assert key in d
    # FIFO (head-of-line) must do no backfill; a backfill policy must do some
    rf = gs.run_scheduling(jobs, nodes, "fifo")
    assert rf.backfill_placements == 0
    assert r.backfill_placements > 0


# --- 12. Full trace test skipped if raw file missing ------------------------

@pytest.mark.skipif(not (os.path.exists(RAW_JOBS) and os.path.exists(RAW_MACH)),
                    reason="raw Philly trace not present (integration only)")
def test_full_trace_integration():
    jobs = philly.load_jobs(RAW_JOBS, sample_size=2000, seed=1, include_failed=True)
    nodes = philly.load_machines(RAW_MACH)
    result = gs.run_backtest(jobs, nodes)
    ca = result.policy_results["constraint_aware"]
    assert ca.goodput_per_dollar is not None


# --- 13 & 15. Docs honesty --------------------------------------------------

def _no_unhedged(text):
    low = text.lower()
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 24):pos]
            assert any(n in pre for n in ("not ", "no ", "never", "n't")), \
                f"unhedged '{phrase}' near ...{text[max(0,pos-24):pos+len(phrase)+8]}..."
            i = pos + len(phrase)


def test_docs_state_limitations():
    text = open(RESULTS_MD).read()
    assert "not customer telemetry" in text.lower()
    assert "gpu_seconds_work" in text
    for missing in ("no real gpu model", "deadline"):
        assert missing in text.lower()
    assert "NOT" in text and "fifo" in text.lower()  # headline-not-fifo stated


def test_docs_no_production_savings():
    for path in (RESULTS_MD, PUBLIC_DOC):
        _no_unhedged(open(path).read())


# --- 14. Other ingesters still work -----------------------------------------

def test_other_ingesters_unaffected():
    from aurelius.traces import alibaba_gpu, azure_llm, burstgpt
    assert burstgpt.load_csv(
        os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv"),
        include_failures=True)
    assert azure_llm.load_csv(
        os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_sample.csv"),
        variant="conv", include_failures=True)
    aj = alibaba_gpu.load_jobs(
        os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_gpu",
                     "openb_pod_list_sample.csv"), include_failed=True)
    assert aj
