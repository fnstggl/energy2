"""Tests for Alibaba GenAI 2026 multi-layer ingestion + serving backtest.

Unit tests use ONLY ``tests/fixtures/alibaba_genai_sample/`` and never touch the
network or the full dataset. The full-trace backtest is integration-only and
skipped when the raw files are absent.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces import alibaba_genai as ag  # noqa: E402
from aurelius.traces import genai_backtest as gb  # noqa: E402
from aurelius.traces.schema import TraceSchemaError  # noqa: E402

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")
REQ = os.path.join(FIX, "lora_request_trace.csv")
RAW = os.path.join(REPO_ROOT, "data", "external", "alibaba_genai", "raw")
RAW_REQ = os.path.join(RAW, ag.REQUEST_FILE)
RESULTS_MD = os.path.join(REPO_ROOT, "docs", "ALIBABA_GENAI_BACKTEST_RESULTS.md")
PUBLIC_DOC = os.path.join(REPO_ROOT, "docs", "PUBLIC_TRACE_BACKTESTS.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


# --- 1. Fixture parses ------------------------------------------------------

def test_sample_fixture_parses():
    reqs = ag.load_requests(REQ, include_failures=True)
    assert len(reqs) > 0
    assert all(r.timestamp_s is not None for r in reqs)
    assert all(r.service_id for r in reqs)


# --- 2. File discovery identifies layers ------------------------------------

def test_discovery_identifies_layers():
    disc = ag.discover(FIX)
    assert set(disc["layers_present"]) >= {"application", "middleware",
                                           "scheduler", "infrastructure"}
    assert "lora_request_trace.csv" in disc["primary_present"]
    # classification present for every registry file
    for name, e in disc["files"].items():
        assert e["classification"] in ("primary", "derived", "metadata", "documentation")


# --- 3. Missing/empty layer handled gracefully ------------------------------

def test_empty_layer_handled():
    disc = ag.discover(FIX)
    # the fixture ships an empty pipeline_update_latency file (header only)
    assert "pipeline_update_latency_anon.csv" in disc["empty"]
    # load_all_layers must not crash with an empty + (some) missing files
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    assert layers["requests"] and layers["gateway"] and layers["infra"]


def test_missing_dir_is_graceful(tmp_path):
    disc = ag.discover(str(tmp_path))   # empty dir → everything missing
    assert disc["primary_present"] == []
    assert len(disc["missing"]) > 0


# --- 4. Request fields normalize correctly ----------------------------------

def test_request_fields_normalize():
    reqs = ag.load_requests(REQ, include_failures=True)
    r0 = reqs[0]
    # first fixture row: 2024-11-15 16:57:50,TXT_2_IMG,SUCCEED,32.0,G0000,63.0,...,M0000,0
    assert r0.request_type == "TXT_2_IMG"
    assert r0.e2e_latency_s == 32.0
    assert r0.service_id == "M0000"
    assert r0.prompt_or_input_size == 63.0
    assert r0.group_id == "G0000"
    assert r0.is_failed is False


def test_request_schema_validation(tmp_path):
    bad = tmp_path / "r.csv"
    bad.write_text("gmt_create,foo\n2024-11-15 16:57:50,1\n")  # missing exec/status
    with pytest.raises(TraceSchemaError):
        ag.load_requests(str(bad))


# --- 5. Gateway fields normalize --------------------------------------------

def test_gateway_normalize():
    g = ag.load_gateway(os.path.join(FIX, "queue_rt_raw_anon.csv"), "waiting_time_s")
    assert g
    # queue_rt values are ms in the file -> normalized to seconds
    assert all(s.waiting_time_s is None or s.waiting_time_s >= 0 for s in g)
    qd = ag.load_gateway(os.path.join(FIX, "queue_size_raw_anon.csv"), "queue_depth")
    assert qd and all(s.queue_depth is not None for s in qd)


# --- 6. Scheduler/pipeline events normalize ---------------------------------

def test_pipeline_normalize():
    ev = ag.load_pipeline(os.path.join(FIX, "basemodel_update_latency_anon.csv"),
                          "basemodel_load")
    assert ev
    assert all(e.stage == "basemodel_load" for e in ev)
    # ms -> seconds
    assert all(e.duration_s is None or e.duration_s > 0 for e in ev)
    assert all(e.container_id for e in ev)  # pipeline files have container_ip


# --- 7. Infra GPU util/memory normalize -------------------------------------

def test_infra_normalize():
    util = ag.load_infra(os.path.join(FIX, "pod_gpu_duty_cycle_anon.csv"),
                         "gpu_utilization")
    mem = ag.load_infra(os.path.join(FIX, "pod_gpu_memory_used_bytes_anon.csv"),
                        "gpu_memory_used")
    assert util and mem
    assert all(s.gpu_utilization is not None for s in util)
    assert all(s.gpu_memory_used is not None for s in mem)
    assert all(s.container_id for s in util)


# --- 8 & 9. Cross-layer linkage classification (no faked joins) -------------

def test_linkage_app_is_no_join():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    # application <-> metric layers MUST be no_join (different time base, no key)
    lk = ag.classify_linkage("application", layers["requests"],
                             "infrastructure", layers["infra"], app_request_layer=True)
    assert lk == "no_join"


def test_linkage_metrics_container_join():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    lk = ag.classify_linkage("infrastructure", layers["infra"],
                             "scheduler", layers["pipeline"], app_request_layer=False)
    assert lk in ("container_join", "time_join")  # share container_ip + time base


def test_missing_cross_layer_keys_no_crash():
    # queue_size has empty container_ip -> classification must not crash
    qd = ag.load_gateway(os.path.join(FIX, "queue_size_raw_anon.csv"), "queue_depth")
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    lk = ag.classify_linkage("middleware", qd, "infrastructure", layers["infra"],
                             app_request_layer=False)
    assert lk in ("no_join", "time_join", "container_join")


# --- 10. Normalized trace generates simulator arrivals/state ----------------

def test_backtest_generates_arrivals():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    res = gb.run_backtest(layers["requests"], tick_seconds=3600.0, cold_start_s=cold)
    assert res.n_requests == len(layers["requests"])
    assert set(res.policy_results) == set(gb.POLICIES)
    for r in res.policy_results.values():
        assert r.kpi.total_infrastructure_cost > 0


# --- 11. Backtest deterministic under fixed seed ----------------------------

def test_backtest_deterministic():
    r1 = ag.load_requests(REQ, sample_size=40, seed=3, include_failures=True)
    r2 = ag.load_requests(REQ, sample_size=40, seed=3, include_failures=True)
    assert [r.to_dict() for r in r1] == [r.to_dict() for r in r2]
    b1 = gb.run_backtest(r1, tick_seconds=3600.0)
    b2 = gb.run_backtest(r2, tick_seconds=3600.0)
    assert b1.to_summary_dict() == b2.to_summary_dict()


def test_cold_start_calibration_from_pipeline():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    assert cold.get("basemodel_load", 0) > 0   # calibrated from real medians
    # affinity must reduce modelled cold-start vs the baselines
    res = gb.run_backtest(layers["requests"], tick_seconds=3600.0, cold_start_s=cold)
    ca = res.policy_results["constraint_aware"]
    base = res.policy_results["sla_aware"]
    assert ca.mean_cold_start_s <= base.mean_cold_start_s


# --- 12. No full-dataset download required ----------------------------------

def test_no_network_for_fixture(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("unit tests must not hit the network")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=True))
    assert gb.run_backtest(layers["requests"]).n_requests == len(layers["requests"])


# --- 13. Full trace test skipped if raw missing -----------------------------

@pytest.mark.skipif(not os.path.exists(RAW_REQ),
                    reason="raw GenAI trace not present (integration only)")
def test_full_trace_integration():
    layers = ag.load_all_layers(RAW, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    res = gb.run_backtest(layers["requests"], tick_seconds=3600.0, cold_start_s=cold)
    ca = res.policy_results["constraint_aware"]
    assert ca.kpi.sla_safe_goodput_per_infra_dollar is not None
    # affinity must not produce MORE cold-start than the headline
    assert ca.mean_cold_start_s <= res.policy_results["sla_aware"].mean_cold_start_s


# --- 14 & 15. Docs honesty --------------------------------------------------

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


def test_docs_state_limitations_and_no_join():
    text = open(RESULTS_MD).read()
    assert "not customer telemetry" in text.lower()
    assert "no_join" in text
    assert "completed_requests" in text
    assert "request" in text.lower() and "gpu" in text.lower()


def test_docs_no_production_savings():
    for path in (RESULTS_MD, PUBLIC_DOC):
        _no_unhedged(open(path).read())


# --- 16. Existing ingesters still importable/parse --------------------------

def test_other_ingesters_unaffected():
    from aurelius.traces import alibaba_gpu, azure_llm, burstgpt, philly
    fx = os.path.join(REPO_ROOT, "tests", "fixtures")
    assert burstgpt.load_csv(os.path.join(fx, "burstgpt_sample.csv"), include_failures=True)
    assert azure_llm.load_csv(os.path.join(fx, "azure_llm_sample.csv"),
                              variant="conv", include_failures=True)
    assert alibaba_gpu.load_jobs(os.path.join(fx, "alibaba_gpu",
                                              "openb_pod_list_sample.csv"),
                                 include_failed=True)
    assert philly.load_jobs(os.path.join(fx, "philly_sample", "cluster_job_log.json"),
                            include_failed=True)
