"""Tests for the Batch Inference Frontier v1 (opt-in, shadow only).

Hard invariants:

1.  Batch Inference Frontier does NOT import the serving rho controller
    modules.
2.  Models reject unknown enum values / out-of-range fields.
3.  Empty arrival-tick window → all points are INSUFFICIENT_TELEMETRY.
4.  Highest-goodput safe point is the recommendation (when current candidate
    is None).
5.  Synthetic-scenario label REQUIRED on the workload profile (the v1 never
    reads a real deadline from the source trace).
6.  Current candidate UNSAFE → LOWER_BATCH_PRESSURE.
7.  Deadline-miss rate above the cap → UNSAFE.
8.  Interactive-baseline veto: ``queue_p99 > interactive_baseline_p99_ms``
    → UNSAFE.
9.  ``BatchInferenceFrontierDecision.executable_in_real_cluster`` is False
    at construction; True raises.
10. ``execute_batch_inference_frontier_decision`` returns shadow-only by
    default.
11. Azure 2024 deadline-slack sanity check: the fixture replay shows a
    positive deadline-slack-vs-rho slope (Phase A acceptance gate).
12. Reusing the existing Azure 2024 fixture does NOT mutate any committed
    serving-frontier artifact.
13. Serving rho frontier and existing serving backtest still work.
14. JSON round-trip for decisions.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.frontier import (  # noqa: E402
    batch_inference_estimator as bie,
    batch_inference_models as bim,
    batch_inference_safety as bis,
)
from aurelius.frontier.batch_inference_controller import (  # noqa: E402
    BatchInferenceFrontierControllerConfig,
    choose_batch_inference_frontier_target,
    execute_batch_inference_frontier_decision,
)
from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402
from aurelius.traces.schema import time_rescale  # noqa: E402

AZURE_FIXTURE = (REPO_ROOT / "tests" / "fixtures"
                 / "azure_llm_2024_sample.csv")


def _profile(**kw):
    defaults = dict(
        workload_id="azure_batch_test",
        trace_source="azure_llm_2024",
        synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1",
        deadline_slack_seconds_baseline=600.0,
        deadline_miss_rate_sla_pct=2.0,
        queue_wait_sla_p99_ms=2000.0,
        telemetry_confidence="medium",
    )
    defaults.update(kw)
    return bim.BatchInferenceWorkloadProfile(**defaults)


def _busy_azure_ticks(scale: float = 100.0):
    reqs = azure_llm.load_csv(str(AZURE_FIXTURE))
    busy = time_rescale(reqs, factor=scale)
    return requests_to_arrival_ticks(busy, tick_seconds=60.0)


# ---- 1. No serving rho controller imports ----

def test_does_not_import_serving_rho_controller_modules():
    for mod in (bim, bis, bie):
        src = inspect.getsource(mod)
        for forbidden in (
            "from aurelius.frontier.controller import",
            "from aurelius.frontier.dynamic_controller import",
            "from aurelius.frontier.estimator import",
            "from aurelius.frontier.dynamic_estimator import",
            "from .controller import",
            "from .dynamic_controller import",
            "from .estimator import",
            "from .dynamic_estimator import",
        ):
            assert forbidden not in src, (
                f"{mod.__name__} should not import the serving rho controller "
                f"(found `{forbidden}`)")


# ---- 2. Validation ----

def test_unknown_trace_source_rejected():
    with pytest.raises(bim.BatchInferenceFrontierSchemaError):
        bim.BatchInferenceWorkloadProfile(
            workload_id="x", trace_source="atari",
            synthetic_scenario_label="x")


def test_missing_synthetic_scenario_label_rejected():
    with pytest.raises(bim.BatchInferenceFrontierSchemaError):
        bim.BatchInferenceWorkloadProfile(
            workload_id="x", trace_source="azure_llm_2024",
            synthetic_scenario_label="")


def test_candidate_target_rho_range():
    with pytest.raises(bim.BatchInferenceFrontierSchemaError):
        bim.BatchInferenceFrontierCandidate(target_rho=1.5)


def test_candidate_deadline_slack_non_negative():
    with pytest.raises(bim.BatchInferenceFrontierSchemaError):
        bim.BatchInferenceFrontierCandidate(deadline_slack_seconds=-1.0)


# ---- 3. Empty window ----

def test_empty_window_returns_insufficient_telemetry():
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=60.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), [], cands)
    assert len(points) == 1
    assert points[0].is_insufficient_telemetry


# ---- 4. Recommendation = highest-goodput safe point ----

def test_recommends_highest_safe_goodput_point():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [
        bim.BatchInferenceFrontierCandidate(
            target_rho=R, deadline_slack_seconds=60.0,
            source_policy=f"rho{R}")
        for R in (0.55, 0.65, 0.75, 0.85)
    ]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    safe = [p for p in points if p.is_safe]
    assert len(safe) >= 1
    decision = choose_batch_inference_frontier_target(_profile(), points)
    assert decision.action == bim.BatchInferenceFrontierAction.RECOMMEND_BATCH_FRONTIER
    # Selected goodput/$ is the best among safe.
    best = max((p.predicted_goodput_per_dollar or 0.0) for p in safe)
    assert (decision.selected_point.predicted_goodput_per_dollar
            >= best - 1e-9)


# ---- 5. Synthetic-scenario label required ----

def test_profile_requires_synthetic_scenario_label():
    # Already covered by validation test; also assert the label round-trips.
    p = _profile(synthetic_scenario_label="batch_overnight_v1")
    d = p.to_dict()
    assert d["synthetic_scenario_label"] == "batch_overnight_v1"


# ---- 6. Current unsafe → LOWER_BATCH_PRESSURE ----

def test_current_unsafe_lowers_pressure():
    ticks = _busy_azure_ticks(scale=100.0)
    # current candidate with slack=0 -> 100% deadline miss
    cur = bim.BatchInferenceFrontierCandidate(
        target_rho=0.85, deadline_slack_seconds=0.0)
    safe = bim.BatchInferenceFrontierCandidate(
        target_rho=0.55, deadline_slack_seconds=60.0)
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, [cur, safe])
    cur_pt = [p for p in points
              if p.candidate.deadline_slack_seconds == 0.0][0]
    assert not cur_pt.is_safe
    decision = choose_batch_inference_frontier_target(
        _profile(), points, current_candidate=cur)
    assert decision.action == bim.BatchInferenceFrontierAction.LOWER_BATCH_PRESSURE


# ---- 7. Deadline miss > cap → UNSAFE ----

def test_deadline_miss_above_cap_unsafe():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.85, deadline_slack_seconds=0.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    p = points[0]
    assert not p.is_safe
    assert any("deadline_miss" in v for v in p.safety_vetoes)


# ---- 8. Interactive baseline veto ----

def test_interactive_baseline_veto():
    ticks = _busy_azure_ticks(scale=100.0)
    profile = _profile(
        interactive_baseline_p99_ms=10.0,  # absurdly tight floor
        interactive_baseline_timeout_pct=0.5,
    )
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0)]
    points = bie.estimate_batch_inference_frontier(profile, ticks, cands)
    p = points[0]
    # Either the queue or timeout regression veto must fire.
    assert not p.is_safe
    assert any("regresses_vs_interactive_baseline" in v
               for v in p.safety_vetoes)


# ---- 9. Decision real-cluster safety ----

def test_decision_executable_in_real_cluster_false_by_default():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    decision = choose_batch_inference_frontier_target(_profile(), points)
    assert decision.executable_in_real_cluster is False


def test_constructing_decision_with_real_execution_raises():
    with pytest.raises(bim.BatchInferenceFrontierSchemaError):
        bim.BatchInferenceFrontierDecision(
            workload_id="x", selected_candidate=None,
            current_candidate=None, selected_point=None,
            frontier_points=tuple(),
            action=bim.BatchInferenceFrontierAction.RECOMMEND_BATCH_FRONTIER,
            reason="r", executable_in_real_cluster=True)


# ---- 10. Execute shim shadow-only ----

def test_execute_shim_shadow_only_by_default():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    decision = choose_batch_inference_frontier_target(_profile(), points)
    res = execute_batch_inference_frontier_decision(decision)
    assert res["mode"] == "shadow"
    assert res["executed"] is False


def test_execute_shim_real_disabled_without_executor():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    decision = choose_batch_inference_frontier_target(_profile(), points)
    res = execute_batch_inference_frontier_decision(
        decision, allow_real_execution=True, executor=None)
    assert res["mode"] == "real_disabled"
    assert res["executed"] is False


# ---- 11. Azure 2024 deadline-slack sanity check (Phase A acceptance) ----

def test_azure_2024_phase_a_sanity_deadline_slack_slope():
    ticks = _busy_azure_ticks(scale=100.0)
    rho_grid = (0.45, 0.55, 0.65, 0.75, 0.85)
    slack_grid = (0.0, 60.0)  # 0s should be UNSAFE, 60s SAFE
    cands = [
        bim.BatchInferenceFrontierCandidate(
            target_rho=R, deadline_slack_seconds=s)
        for R in rho_grid for s in slack_grid
    ]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    # At slack=0 every candidate must be UNSAFE (the sanity check).
    slack0 = [p for p in points
              if p.candidate.deadline_slack_seconds == 0.0]
    assert slack0, "no zero-slack candidates evaluated"
    assert not any(p.is_safe for p in slack0), (
        "at slack=0 some candidates were safe — sanity check broken")
    # At slack=60s at least one candidate must be SAFE.
    slack60 = [p for p in points
               if p.candidate.deadline_slack_seconds == 60.0]
    assert any(p.is_safe for p in slack60), (
        "no candidate safe at slack=60s — deadline slack lever inactive")
    # Goodput/$ at rho=0.55 must be >= goodput/$ at rho=0.45 (rho slope).
    rho45 = [p for p in slack60 if p.candidate.target_rho == 0.45][0]
    rho55 = [p for p in slack60 if p.candidate.target_rho == 0.55][0]
    assert (rho55.predicted_goodput_per_dollar
            >= rho45.predicted_goodput_per_dollar), (
        "rho=0.55 did not beat rho=0.45 on goodput/$ — no positive rho slope")


# ---- 12. Serving frontier still works (regression) ----

def test_existing_serving_frontier_modules_still_import():
    from aurelius.frontier import (  # noqa: F401
        controller, dynamic_controller, dynamic_estimator, estimator,
        models, safety,
    )


def test_serving_rho_default_unchanged():
    # The serving constraint_aware default rho is hard-coded to 0.65 in
    # aurelius/traces/backtest.py::_run_policy. Make sure that byte-for-byte
    # default has not been disturbed.
    from aurelius.traces import backtest as bt
    src = inspect.getsource(bt._run_policy)
    assert "ca_target_rho = 0.65" in src


# ---- 14. JSON round-trip ----

def test_batch_decision_to_dict_serializable():
    ticks = _busy_azure_ticks(scale=100.0)
    cands = [bim.BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0)]
    points = bie.estimate_batch_inference_frontier(
        _profile(), ticks, cands)
    decision = choose_batch_inference_frontier_target(_profile(), points)
    json.dumps(decision.to_dict())
