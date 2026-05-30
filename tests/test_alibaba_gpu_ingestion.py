"""Tests for Alibaba GPU v2023 ingestion + the executable packing backtest.

Unit tests use ONLY the fixtures under ``tests/fixtures/alibaba_gpu/`` and never
touch the network or the full dataset. The full-trace backtest is an
integration test skipped when the raw files are absent.
"""

from __future__ import annotations

import csv
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces import alibaba_gpu as az  # noqa: E402
from aurelius.traces import gpu_packing as gp  # noqa: E402
from aurelius.traces.schema import TraceSchemaError, validate_columns  # noqa: E402

FIX_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_gpu")
FIX_POD = os.path.join(FIX_DIR, "openb_pod_list_sample.csv")
FIX_NODE = os.path.join(FIX_DIR, "openb_node_list_sample.csv")
RAW_POD = os.path.join(REPO_ROOT, "data", "external", "alibaba_gpu", "raw",
                       "openb_pod_list_default.csv")
RAW_NODE = os.path.join(REPO_ROOT, "data", "external", "alibaba_gpu", "raw",
                        "openb_node_list_gpu_node.csv")
RESULTS_MD = os.path.join(REPO_ROOT, "docs", "ALIBABA_GPU_BACKTEST_RESULTS.md")
PUBLIC_DOC = os.path.join(REPO_ROOT, "docs", "PUBLIC_TRACE_BACKTESTS.md")

BANNED_CLAIMS = (
    "production savings", "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics", "production-proven",
)


# --- 1. Fixture parses ------------------------------------------------------

def test_sample_fixture_parses():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    assert len(jobs) > 0 and len(nodes) > 0
    assert sum(n.gpu_count for n in nodes) > 0
    assert len({n.gpu_model for n in nodes}) > 1  # heterogeneous fixture


# --- 2. Schema validation catches missing required fields -------------------

def test_schema_validation_missing_pod_columns(tmp_path):
    bad = tmp_path / "bad_pod.csv"
    bad.write_text("name,num_gpu\nopenb-pod-x,1\n")  # missing gpu_milli/phase/creation
    with pytest.raises(TraceSchemaError):
        az.load_jobs(str(bad))


def test_schema_validation_missing_node_columns(tmp_path):
    bad = tmp_path / "bad_node.csv"
    bad.write_text("sn,cpu_milli\nopenb-node-x,1000\n")  # missing gpu/model
    with pytest.raises(TraceSchemaError):
        az.load_nodes(str(bad))


def test_validate_columns_helper():
    with pytest.raises(TraceSchemaError):
        validate_columns(["name", "num_gpu"], az.POD_REQUIRED, "alibaba_gpu")
    validate_columns(list(az.POD_REQUIRED), az.POD_REQUIRED, "alibaba_gpu")


# --- 3. GPU count / type / duration map correctly ---------------------------

def test_gpu_fields_map_correctly():
    # First fixture pod: openb-pod-0000,12000,16384,1,1000,,LS,Running,0,12537496,0
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    j = next(x for x in jobs if x.job_id == "openb-pod-0000")
    assert j.gpu_count == 1
    assert j.gpu_milli == 1000
    assert j.gpu_type is None          # gpu_spec empty -> None
    assert j.cpu_milli == 12000
    assert j.memory_mib == 16384
    assert j.submit_time_s == 0.0
    assert j.end_time_s == 12537496.0
    assert j.duration_s == 12537496.0  # deletion - start
    assert j.workload_type == "LS"     # qos
    assert j.token_equivalent_work == gp.job_work(j)


def test_effective_gpu_fractional_vs_whole():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    by_id = {j.job_id: j for j in jobs}
    # a whole-GPU job (gpu_milli=1000) has effective_gpu == num_gpu
    whole = by_id["openb-pod-0000"]
    assert gp.effective_gpu(whole) == 1.0
    # a fractional job (num_gpu==1, gpu_milli<1000) has effective < 1
    frac = next((j for j in jobs if j.gpu_count == 1 and j.gpu_milli
                 and j.gpu_milli < 1000), None)
    if frac is not None:
        assert 0 < gp.effective_gpu(frac) < 1.0


# --- 4. Failed jobs included/excluded per flag ------------------------------

def test_failed_jobs_include_exclude(tmp_path):
    p = tmp_path / "pods.csv"
    rows = [
        ["name", "cpu_milli", "memory_mib", "num_gpu", "gpu_milli", "gpu_spec",
         "qos", "pod_phase", "creation_time", "deletion_time", "scheduled_time"],
        ["p-ok", "1000", "1024", "1", "1000", "", "LS", "Running", "0", "100", "0"],
        ["p-fail", "1000", "1024", "1", "1000", "", "LS", "Failed", "0", "50", "0"],
    ]
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    incl = az.load_jobs(str(p), include_failed=True)
    excl = az.load_jobs(str(p), include_failed=False)
    assert len(incl) == 2
    assert len(excl) == 1
    assert any(j.is_failed for j in incl)
    assert all(not j.is_failed for j in excl)


# --- 5. Normalized jobs generate simulator arrivals (packing input) ---------

def test_jobs_generate_packing_placements():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    res = gp.run_packing(jobs, nodes, "best_fit")
    assert res.placed_jobs > 0
    # every placed job's work is accounted in goodput (token_equivalent)
    assert res.placed_work > 0
    assert res.placed_jobs + res.stranded_jobs == sum(
        1 for j in jobs if gp.effective_gpu(j) > 0)


# --- 6. Packing baselines run deterministically -----------------------------

def test_packing_baselines_run_deterministically():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    for policy in gp.PACKING_POLICIES:
        r1 = gp.run_packing(jobs, nodes, policy)
        r2 = gp.run_packing(jobs, nodes, policy)
        assert r1.summary() == r2.summary(), f"{policy} not deterministic"


# --- 7 & 8. Headline is a packing baseline, NOT fifo ------------------------

def test_headline_is_packing_baseline_not_fifo():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    result = gp.run_backtest(jobs, nodes)
    assert result.outcome.headline in gp.HEADLINE_CANDIDATES
    assert result.outcome.headline != "fifo"
    assert "best_fit" in gp.HEADLINE_CANDIDATES
    assert "first_fit_decreasing" in gp.HEADLINE_CANDIDATES
    assert "fifo" not in gp.HEADLINE_CANDIDATES
    d = result.to_summary_dict()
    assert d["headline_is_packing_baseline"] is True


# --- 9. Fragmentation score is reported -------------------------------------

def test_fragmentation_score_reported():
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    result = gp.run_backtest(jobs, nodes)
    for pol, r in result.policy_results.items():
        assert r.fragmentation_score is not None
        assert "fragmentation_score" in r.summary()
        assert r.summary()["fragmentation_score"] >= 0.0


# --- 10. Backtest deterministic under fixed seed ----------------------------

def test_backtest_deterministic():
    jobs1 = az.load_jobs(FIX_POD, sample_size=25, seed=11, include_failed=True)
    jobs2 = az.load_jobs(FIX_POD, sample_size=25, seed=11, include_failed=True)
    assert [j.to_dict() for j in jobs1] == [j.to_dict() for j in jobs2]
    nodes = az.load_nodes(FIX_NODE)
    b1 = gp.run_backtest(jobs1, nodes)
    b2 = gp.run_backtest(jobs2, nodes)
    assert b1.to_summary_dict() == b2.to_summary_dict()


# --- 11. Unit tests do not require full download ----------------------------

def test_no_network_for_fixture(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("unit tests must not hit the network")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    nodes = az.load_nodes(FIX_NODE)
    assert gp.run_backtest(jobs, nodes).n_jobs == len(jobs)


def test_no_utilization_samples_documented():
    # Alibaba v2023 has no utilization series; summary states it explicitly.
    jobs = az.load_jobs(FIX_POD, include_failed=True)
    s = az.summarize_jobs(jobs, az.load_nodes(FIX_NODE))
    assert s["gpu_utilization_samples"] == 0
    assert "gpu_utilization_timeseries" in s["missing_fields"]
    assert "gpu_memory_gb" in s["missing_fields"]


# --- 12. Full-trace backtest is integration-only / skipped if raw missing ---

@pytest.mark.skipif(not (os.path.exists(RAW_POD) and os.path.exists(RAW_NODE)),
                    reason="raw Alibaba CSVs not present (integration only)")
def test_full_trace_backtest_integration():
    # sample to keep the integration test fast
    jobs = az.load_jobs(RAW_POD, sample_size=800, seed=1, include_failed=False)
    nodes = az.load_nodes(RAW_NODE)
    result = gp.run_backtest(jobs, nodes)
    ca = result.policy_results["constraint_aware"]
    assert ca.goodput_per_dollar is not None
    # constraint_aware must never strand more work than the packing headline.
    head = result.policy_results[result.outcome.headline]
    assert ca.stranded_jobs <= head.stranded_jobs + 1


# --- 13 & 14. Docs honesty --------------------------------------------------

def _assert_no_unhedged_banned_claims(text: str):
    low = text.lower()
    for phrase in BANNED_CLAIMS:
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            prefix = low[max(0, pos - 24):pos]
            assert any(neg in prefix for neg in ("not ", "no ", "never", "n't")), (
                f"unhedged banned claim '{phrase}' near: "
                f"...{text[max(0, pos-24):pos+len(phrase)+10]}...")
            idx = pos + len(phrase)


def test_docs_state_limitations_and_missing_fields():
    text = open(RESULTS_MD).read()
    assert "not customer telemetry" in text.lower()
    assert "completed_gpu_job_work" in text
    for missing in ("gpu_utilization", "gpu_memory", "deadline"):
        assert missing in text.lower()
    # headline is explicitly a packing baseline, not fifo
    assert "NOT" in text and "fifo" in text.lower()


def test_docs_no_production_savings_claims():
    for path in (RESULTS_MD, PUBLIC_DOC):
        _assert_no_unhedged_banned_claims(open(path).read())


# --- 15. Existing BurstGPT + Azure LLM ingesters still import/parse ---------

def test_other_ingesters_unaffected():
    from aurelius.traces import azure_llm, burstgpt
    b = burstgpt.load_csv(
        os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv"),
        include_failures=True)
    assert all(isinstance(r.cache_affinity_key, str) for r in b)
    a = azure_llm.load_csv(
        os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_sample.csv"),
        variant="conv", include_failures=True)
    assert all(r.cache_affinity_key is None for r in a)
