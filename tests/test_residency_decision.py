"""Tests for the Model Residency Decision Engine v1.

Covers the 15 required behaviors: resident routing, model+adapter routing,
KPI-aware cold preference, SLA-unsafe cold rejection, honest missing-data
handling, INSUFFICIENT_TELEMETRY, memory-headroom blocking, the no-substitution
rule, prewarm accept/reject economics, eviction only under memory pressure,
real-mode immutability, deterministic simulator mutation, the preserved GenAI
backtest, and the no-production-savings-claim docs gate.
"""

from __future__ import annotations

import os

import pytest

from aurelius.residency import sim
from aurelius.residency.decision import (
    SafetyContext,
    choose_residency_decision,
    score_residency_candidate,
)
from aurelius.residency.models import (
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
    ResidencyAction,
    ResidencyDecision,
    ResidencySchemaError,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _loc(gpu, *, models=(), adapters=(), used_gb=0.0, total_gb=80.0, queue_s=0.1,
         conf="high", thermal=None, region="r"):
    return ModelLocationState(
        region=region, node_id="n", gpu_id=gpu, container_id=f"pod-{gpu}",
        loaded_model_ids=list(models), loaded_adapter_ids=list(adapters),
        gpu_memory_used=used_gb * 1e9, gpu_memory_total=total_gb * 1e9,
        estimated_queue_wait_s=queue_s, thermal_risk=thermal,
        telemetry_confidence=conf)


def _profile(model="m", *, p50=22.0, p95=30.0, mem=16.0, a50=4.0, a95=6.0,
             adapter=None):
    return ModelLoadProfile(model_id=model, cold_load_p50_s=p50, cold_load_p95_s=p95,
                            adapter_id=adapter, adapter_load_p50_s=a50,
                            adapter_load_p95_s=a95, memory_required_gb=mem,
                            source="cal", confidence="high")


def _ctx(**kw):
    base = dict(gpu_hour_price=3.0, default_latency_sla_ms=120000.0,
                service_time_proxy_s=2.0, min_telemetry_confidence="low")
    base.update(kw)
    return SafetyContext(**base)


def _req(model="m", **kw):
    base = dict(request_id="r1", timestamp=1.0, workload_id="w", model_id=model,
                priority_class="standard")
    base.update(kw)
    return ModelResidencyRequest(**base)


# 1 -------------------------------------------------------------------------
def test_routes_to_location_where_model_resident():
    warm = _loc("g0", models=["m"], used_gb=16, queue_s=0.1)
    cold = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    d = choose_residency_decision(_req(), [warm, cold], {"m": _profile()},
                                  _ctx(), _ctx())
    assert d.action == ResidencyAction.ROUTE_TO_RESIDENT_MODEL
    assert d.target_location == warm.location_key
    assert d.executable_in_real_cluster is False


# 2 -------------------------------------------------------------------------
def test_routes_to_location_where_model_and_adapter_resident():
    both = _loc("g0", models=["m"], adapters=["a"], used_gb=18, queue_s=0.1)
    base_only = _loc("g1", models=["m"], adapters=[], used_gb=16, queue_s=0.05)
    prof = {("m", "a"): _profile(adapter="a")}
    d = choose_residency_decision(_req(adapter_id="a"), [both, base_only], prof,
                                  _ctx(), _ctx())
    assert d.action == ResidencyAction.ROUTE_TO_RESIDENT_MODEL
    assert d.target_location == both.location_key


# 3 -------------------------------------------------------------------------
def test_high_queue_resident_can_make_cold_node_preferable():
    prof = {"m": _profile(p50=22.0, p95=30.0)}
    # low-queue resident → prefer the warm replica (affinity)
    warm_lo = _loc("g0", models=["m"], used_gb=16, queue_s=0.1)
    cold = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    d_warm = choose_residency_decision(_req(), [warm_lo, cold], prof, _ctx(), _ctx())
    assert d_warm.target_location == warm_lo.location_key

    # resident replica queue so high it is SLA-unsafe; cold replica is SLA-safe →
    # the engine must prefer the cold lower-queue node (KPI improves).
    warm_hi = _loc("g0", models=["m"], used_gb=16, queue_s=400.0)
    d_cold = choose_residency_decision(
        _req(current_route=warm_hi.location_key), [warm_hi, cold], prof,
        _ctx(default_latency_sla_ms=60000.0), _ctx(default_latency_sla_ms=60000.0))
    assert d_cold.target_location == cold.location_key
    assert d_cold.action == ResidencyAction.PREWARM_MODEL


# 4 -------------------------------------------------------------------------
def test_cold_route_rejected_when_cold_start_violates_sla():
    cold = _loc("g0", models=[], used_gb=0, queue_s=0.05)
    # SLA 10s but cold load is 30s → cannot meet SLA anywhere
    d = choose_residency_decision(_req(latency_sla_ms=10000.0), [cold],
                                  {"m": _profile(p50=30.0, p95=30.0)}, _ctx(), _ctx())
    assert d.action == ResidencyAction.REJECT_UNSAFE_ROUTE


# 5 -------------------------------------------------------------------------
def test_missing_load_latency_does_not_become_zero():
    cold = _loc("g0", models=[], used_gb=0, queue_s=0.05)
    prof = ModelLoadProfile(model_id="m", cold_load_p50_s=None, cold_load_p95_s=None,
                            memory_required_gb=16.0, source="cal", confidence="low")
    score = score_residency_candidate(_req(), cold, prof, _ctx(), _ctx())
    # unknown load latency makes the candidate not safely scorable (not free)
    assert score.feasible is False
    assert "missing_load_latency" in score.safety_vetoes
    assert score.expected_latency_s is None


# 6 -------------------------------------------------------------------------
def test_missing_telemetry_returns_insufficient_telemetry():
    unknown = _loc("g0", models=["m"], used_gb=16, conf="unknown")
    d = choose_residency_decision(_req(), [unknown], {"m": _profile()},
                                  _ctx(min_telemetry_confidence="low"), _ctx())
    assert d.action == ResidencyAction.INSUFFICIENT_TELEMETRY


def test_no_profile_for_cold_model_is_insufficient_telemetry():
    cold = _loc("g0", models=[], used_gb=0)
    d = choose_residency_decision(_req(), [cold], {}, _ctx(), _ctx())
    assert d.action == ResidencyAction.INSUFFICIENT_TELEMETRY


# 7 -------------------------------------------------------------------------
def test_memory_headroom_blocks_route():
    full = _loc("g0", models=["other"], used_gb=78, total_gb=80, queue_s=0.05)
    score = score_residency_candidate(_req(), full, _profile(mem=16.0), _ctx(), _ctx())
    assert "insufficient_memory_headroom" in score.safety_vetoes
    assert score.feasible is False


# 8 -------------------------------------------------------------------------
def test_no_substitution_decision_never_changes_requested_model():
    warm_other = _loc("g0", models=["DIFFERENT"], used_gb=16, queue_s=0.05)
    cold_req = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    req = _req(model="requested", adapter_id="reqadapter")
    prof = {("requested", "reqadapter"): _profile(model="requested", adapter="reqadapter")}
    d = choose_residency_decision(req, [warm_other, cold_req], prof, _ctx(), _ctx())
    # the decision carries no field that could express a substitute model, and it
    # never targets a location *because* a different model is resident there.
    assert "substitute" not in d.to_dict()
    # a residency hit is only credited for the REQUESTED model
    sc = score_residency_candidate(req, warm_other, prof[("requested", "reqadapter")],
                                   _ctx(), _ctx())
    assert sc.model_resident is False  # "DIFFERENT" resident ≠ requested model


# 9 -------------------------------------------------------------------------
def test_prewarm_recommended_when_saved_exceeds_warm_pool_cost():
    prof = {"m": _profile(p50=22.0, p95=30.0)}
    warm_hi = _loc("g0", models=["m"], used_gb=16, queue_s=40.0)
    cold = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    # cold beats the very-busy warm replica on KPI; many expected hits → prewarm
    ctx = _ctx(default_latency_sla_ms=120000.0, prewarm_expected_hits=200.0,
               warm_pool_hold_hours=1.0)
    d = choose_residency_decision(_req(current_route=warm_hi.location_key),
                                  [warm_hi, cold], prof, ctx, ctx)
    assert d.action == ResidencyAction.PREWARM_MODEL
    assert d.target_location == cold.location_key
    assert d.expected_cold_start_saved_s and d.expected_cold_start_saved_s > 0


# 10 ------------------------------------------------------------------------
def test_prewarm_rejected_when_cost_exceeds_benefit_falls_back_to_affinity():
    prof = {"m": _profile(p50=22.0, p95=30.0)}
    warm_hi = _loc("g0", models=["m"], used_gb=16, queue_s=40.0)
    cold = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    # cold beats warm on KPI, but few expected hits + expensive hold → no prewarm;
    # fall back to the warm resident replica (affinity), never PREWARM.
    ctx = _ctx(default_latency_sla_ms=120000.0, prewarm_expected_hits=0.1,
               warm_pool_hold_hours=100.0)
    d = choose_residency_decision(_req(current_route=warm_hi.location_key),
                                  [warm_hi, cold], prof, ctx, ctx)
    assert d.action in (ResidencyAction.ROUTE_TO_RESIDENT_MODEL,
                        ResidencyAction.PRESERVE_AFFINITY,
                        ResidencyAction.KEEP_CURRENT_ROUTE)
    assert d.target_location == warm_hi.location_key


# 11 ------------------------------------------------------------------------
def test_eviction_candidate_only_under_memory_pressure():
    prof = {"m": _profile(mem=16.0, p50=22.0, p95=30.0)}
    # memory pressure: only location is full of a different model → EVICT
    full = _loc("g0", models=["other"], used_gb=78, total_gb=80, queue_s=0.05)
    d_pressure = choose_residency_decision(_req(), [full], prof,
                                           _ctx(prewarm_expected_hits=200.0),
                                           _ctx(prewarm_expected_hits=200.0))
    assert d_pressure.action == ResidencyAction.EVICT_CANDIDATE

    # no pressure: same model set but ample free memory → NOT eviction
    roomy = _loc("g0", models=["other"], used_gb=16, total_gb=80, queue_s=0.05)
    d_ok = choose_residency_decision(_req(), [roomy], prof,
                                     _ctx(prewarm_expected_hits=200.0),
                                     _ctx(prewarm_expected_hits=200.0))
    assert d_ok.action != ResidencyAction.EVICT_CANDIDATE


# 12 ------------------------------------------------------------------------
def test_real_mode_never_mutates_state():
    loc = _loc("g0", models=[], used_gb=0)
    locs = {loc.location_key: loc}
    req = _req(model="newmodel")
    dec = ResidencyDecision(request_id="r", action=ResidencyAction.PREWARM_MODEL,
                            reason="x", target_location=loc.location_key)
    before_models = list(loc.loaded_model_ids)
    before_mem = loc.gpu_memory_used
    eff = sim.apply_residency_decision(dec, locs, mode=sim.REAL_MODE, request=req,
                                       load_profile=_profile("newmodel"))
    assert eff.mutated is False
    assert loc.loaded_model_ids == before_models
    assert loc.gpu_memory_used == before_mem


def test_decision_cannot_be_marked_real_executable():
    with pytest.raises(ResidencySchemaError):
        ResidencyDecision(request_id="r", action=ResidencyAction.PREWARM_MODEL,
                          reason="x", executable_in_real_cluster=True)


# 13 ------------------------------------------------------------------------
def test_simulator_mode_mutates_state_deterministically():
    def fresh():
        loc = _loc("g0", models=[], used_gb=0)
        return loc, {loc.location_key: loc}
    req = _req(model="newmodel")
    dec = ResidencyDecision(request_id="r", action=ResidencyAction.PREWARM_MODEL,
                            reason="x", target_location="r/n/g0/pod-g0")
    loc1, locs1 = fresh()
    e1 = sim.apply_residency_decision(dec, locs1, mode=sim.SIMULATOR_MODE,
                                      request=req, load_profile=_profile("newmodel"))
    loc2, locs2 = fresh()
    e2 = sim.apply_residency_decision(dec, locs2, mode=sim.SIMULATOR_MODE,
                                      request=req, load_profile=_profile("newmodel"))
    assert e1.mutated and e2.mutated
    assert loc1.loaded_model_ids == loc2.loaded_model_ids == ["newmodel"]
    assert loc1.gpu_memory_used == loc2.gpu_memory_used == 16.0 * 1e9


def test_simulator_evict_then_admit():
    loc = _loc("g0", models=["victim"], used_gb=78, total_gb=80)
    locs = {loc.location_key: loc}
    req = _req(model="newmodel")
    dec = ResidencyDecision(request_id="r", action=ResidencyAction.EVICT_CANDIDATE,
                            reason="x", target_location=loc.location_key)
    eff = sim.apply_residency_decision(dec, locs, mode=sim.SIMULATOR_MODE,
                                       request=req, load_profile=_profile("newmodel"),
                                       memory_required_gb=16.0)
    assert eff.evicted_model_id == "victim"
    assert "victim" not in loc.loaded_model_ids
    assert "newmodel" in loc.loaded_model_ids


# 14 ------------------------------------------------------------------------
def test_genai_ablation_and_residency_backtest_still_run():
    from aurelius.residency import backtest as rb
    from aurelius.traces import alibaba_genai as ag
    from aurelius.traces import genai_ablation as abl
    fix = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")
    layers = ag.load_all_layers(fix, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    # the EXISTING ablation still produces its configs (preserved, unchanged)
    abl_res = abl.run_ablation(layers["requests"], tick_seconds=3600.0, cold_start_s=cold)
    for name in ("fifo", "sla_aware", "constraint_aware", "constraint_aware_no_affinity"):
        assert name in abl_res
    # the NEW per-request residency backtest runs all policies
    res = rb.run_residency_backtest(layers["requests"], cold_start_s=cold, n_gpus=4)
    assert set(res.policy_results) == set(rb.POLICIES)
    # residency routing beats residency-blind FIFO on the model hit-rate
    fifo_hit = res.policy_results["fifo_round_robin"].model_residency_hit_rate
    eng_hit = res.policy_results["residency_engine"].model_residency_hit_rate
    assert eng_hit >= fifo_hit


def test_residency_backtest_is_deterministic():
    from aurelius.residency import backtest as rb
    from aurelius.traces import alibaba_genai as ag
    fix = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")
    layers = ag.load_all_layers(fix, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    r1 = rb.run_residency_backtest(layers["requests"], cold_start_s=cold, n_gpus=4)
    r2 = rb.run_residency_backtest(layers["requests"], cold_start_s=cold, n_gpus=4)
    for p in rb.POLICIES:
        assert r1.policy_results[p].summary() == r2.policy_results[p].summary()


# 15 ------------------------------------------------------------------------
def test_decision_engine_docs_have_no_unhedged_production_claims():
    banned = ("production savings", "guaranteed savings",
              "enterprise-ready autonomous optimization",
              "hyperscaler-validated economics", "production-proven")
    docs = ["docs/MODEL_RESIDENCY_DECISION_ENGINE.md",
            "docs/MODEL_RESIDENCY_DECISION_ENGINE_RESULTS.md"]
    for rel in docs:
        path = os.path.join(REPO_ROOT, rel)
        assert os.path.exists(path), f"missing doc: {rel}"
        low = " ".join(open(path, encoding="utf-8").read().lower().split())
        for phrase in banned:
            i = 0
            while True:
                pos = low.find(phrase, i)
                if pos == -1:
                    break
                pre = low[max(0, pos - 30):pos]
                assert any(n in pre for n in ("not ", "no ", "never ", "n't ",
                                              "without ")), \
                    f"unhedged '{phrase}' in {rel}"
                i = pos + len(phrase)
