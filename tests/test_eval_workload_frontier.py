"""Tests for the Eval Workload Frontier v1 (opt-in, shadow only).

Hard invariants:

1.  Eval Workload Frontier does NOT import the serving rho controller
    modules (controller.py / dynamic_controller.py / estimator.py /
    dynamic_estimator.py).
2.  Models reject unknown enum values / out-of-range fields.
3.  Empty eval-request set → all points are INSUFFICIENT_TELEMETRY.
4.  When candidates are SAFE, the controller picks the highest-goodput/$
    safe point.
5.  When the current candidate is UNSAFE, the controller emits
    LOWER_EVAL_CONCURRENCY.
6.  Mixed-fleet veto: with ``dedicated_fleet=False`` AND interactive
    baselines missing → INSUFFICIENT_TELEMETRY for that candidate; the
    controller's recommendation moves to ISOLATE_FROM_INTERACTIVE when all
    unsafe candidates carry the mixed-fleet veto.
7.  Mixed-fleet veto with baselines present AND predicted deltas above
    tolerance → UNSAFE.
8.  Deadline-miss-rate above the configured cap → UNSAFE.
9.  ``EvalWorkloadFrontierDecision.executable_in_real_cluster`` is False at
    construction; constructing with True raises.
10. ``execute_eval_workload_frontier_decision`` returns shadow-only by
    default; non-zero opt-in requires both flag + executor.
11. Synthetic-scenario label required on the profile.
12. JSON round-trip works for models.
13. The default profile (telemetry_confidence="low") still produces SAFE
    points when the candidate is in range — the gate is opt-in.
14. The serving rho frontier modules still import cleanly (no breakage).
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
    batch_inference_estimator as bie,  # noqa: F401  (importable check)
)
from aurelius.frontier import (  # noqa: E402
    eval_workload_estimator as ewe,
    eval_workload_models as ewm,
    eval_workload_safety as ews,
)
from aurelius.frontier.eval_workload_controller import (  # noqa: E402
    EvalWorkloadFrontierControllerConfig,
    choose_eval_workload_frontier_target,
    execute_eval_workload_frontier_decision,
)
from aurelius.traces.eval_schema import EvalWorkloadRequest  # noqa: E402


def _make_eval_requests(n: int = 100, tokens: int = 400):
    """Synthetic eval-request set with constant token shape."""
    return [
        EvalWorkloadRequest(
            request_id=f"r-{i}", turn_count=2, role_sequence_signature="h-g",
            token_count_source="char_div_4_proxy",
            provenance="synthetic_eval_fixture_v1",
            prompt_tokens_est=tokens // 2,
            response_tokens_est=tokens // 2,
            prompt_chars=tokens * 2,
            response_chars=tokens * 2,
        )
        for i in range(n)
    ]


def _profile(dedicated: bool = True, **kw):
    defaults = dict(
        workload_id="eval_test",
        trace_source="synthetic_fixture",
        synthetic_scenario_label="synthetic_eval_overnight_v1",
        dedicated_fleet=dedicated,
        deadline_slack_hours_baseline=4.0,
        deadline_miss_rate_sla_pct=1.0,
        eval_suite_completion_deadline_hours=24.0,
        telemetry_confidence="medium",
    )
    defaults.update(kw)
    return ewm.EvalWorkloadProfile(**defaults)


# ---- 1. No import of serving rho controller ----

def test_does_not_import_serving_rho_controller_modules():
    for mod in (ewm, ews, ewe):
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


# ---- 2. Enum + range validation ----

def test_unknown_trace_source_rejected():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadProfile(
            workload_id="x", trace_source="atari_2600",
            synthetic_scenario_label="synthetic_v1")


def test_missing_synthetic_scenario_label_rejected():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadProfile(
            workload_id="x", trace_source="synthetic_fixture",
            synthetic_scenario_label="")


def test_candidate_rho_range_validated():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadFrontierCandidate(target_rho=1.5)
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadFrontierCandidate(target_rho=0.0)


def test_candidate_concurrency_must_be_at_least_one():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadFrontierCandidate(concurrency=0)


def test_candidate_deadline_slack_must_be_non_negative():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadFrontierCandidate(deadline_slack_hours=-0.5)


# ---- 3. Empty request set ----

def test_empty_request_set_returns_all_insufficient_telemetry():
    cands = [ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=4.0,
        dedicated_fleet=True)]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), [], cands)
    assert len(points) == 1
    assert points[0].is_insufficient_telemetry


# ---- 4. Highest-goodput safe point selected ----

def test_selects_highest_goodput_safe_point():
    cands = [
        ewm.EvalWorkloadFrontierCandidate(
            target_rho=0.55, concurrency=4, deadline_slack_hours=24.0,
            dedicated_fleet=True),
        ewm.EvalWorkloadFrontierCandidate(
            target_rho=0.85, concurrency=4, deadline_slack_hours=24.0,
            dedicated_fleet=True),
    ]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), cands)
    # Higher rho should beat lower rho on goodput/$ when both are SAFE.
    safe = [p for p in points if p.is_safe]
    assert len(safe) == 2
    high_rho = [p for p in safe if p.candidate.target_rho == 0.85][0]
    low_rho = [p for p in safe if p.candidate.target_rho == 0.55][0]
    assert (high_rho.predicted_goodput_per_dollar
            >= low_rho.predicted_goodput_per_dollar)
    decision = choose_eval_workload_frontier_target(_profile(), points)
    assert decision.action == ewm.EvalWorkloadFrontierAction.RECOMMEND_EVAL_FRONTIER
    assert decision.selected_candidate.target_rho == 0.85


# ---- 5. Current candidate UNSAFE -> LOWER_EVAL_CONCURRENCY ----

def test_unsafe_current_triggers_lower():
    # Current candidate with deadline_slack=0 forces all requests to miss
    # the deadline -> UNSAFE.
    cur = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.85, concurrency=8, deadline_slack_hours=0.0,
        dedicated_fleet=True)
    safe_lower = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.85, concurrency=2, deadline_slack_hours=24.0,
        dedicated_fleet=True)
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), [cur, safe_lower])
    cur_pt = [p for p in points if p.candidate.concurrency == 8][0]
    assert not cur_pt.is_safe
    decision = choose_eval_workload_frontier_target(
        _profile(), points, current_candidate=cur)
    assert decision.action == ewm.EvalWorkloadFrontierAction.LOWER_EVAL_CONCURRENCY


# ---- 6. Mixed-fleet veto: baselines missing -> INSUFFICIENT_TELEMETRY ----

def test_mixed_fleet_without_baselines_is_insufficient_telemetry():
    cand = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=4.0,
        dedicated_fleet=False)
    points = ewe.estimate_eval_workload_frontier(
        _profile(dedicated=False), _make_eval_requests(), [cand])
    p = points[0]
    assert p.is_insufficient_telemetry
    assert any("baseline" in v for v in p.safety_vetoes)


# ---- 7. Mixed-fleet with baselines + high rho -> UNSAFE ----

def test_mixed_fleet_high_rho_unsafe_with_baselines():
    profile = _profile(
        dedicated=False,
        interactive_baseline_p99_ms=2000.0,
        interactive_baseline_timeout_pct=2.0,
    )
    cand = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.95, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=False)
    points = ewe.estimate_eval_workload_frontier(
        profile, _make_eval_requests(), [cand])
    p = points[0]
    # With R=0.95 the structural model predicts a large interactive p99 delta
    # which exceeds the zero-tolerance gate.
    assert not p.is_safe
    assert any("interactive" in v and "regresses" in v
               for v in p.safety_vetoes)


def test_mixed_fleet_veto_recommends_isolate_when_only_unsafe():
    profile = _profile(
        dedicated=False,
        interactive_baseline_p99_ms=2000.0,
        interactive_baseline_timeout_pct=2.0,
    )
    # one unsafe shared-fleet candidate + one safe dedicated-fleet candidate
    cand_unsafe = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.95, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=False)
    cand_safe = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.85, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=True)
    points = ewe.estimate_eval_workload_frontier(
        profile, _make_eval_requests(), [cand_unsafe, cand_safe])
    decision = choose_eval_workload_frontier_target(profile, points)
    assert decision.action == ewm.EvalWorkloadFrontierAction.ISOLATE_FROM_INTERACTIVE
    assert decision.selected_candidate.dedicated_fleet is True


# ---- 8. Deadline-miss-rate above cap -> UNSAFE ----

def test_deadline_miss_above_cap_unsafe():
    profile = _profile()
    # tight deadline -> high miss rate
    cand = ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.55, concurrency=1, deadline_slack_hours=0.0001,
        dedicated_fleet=True)
    points = ewe.estimate_eval_workload_frontier(
        profile, _make_eval_requests(), [cand])
    p = points[0]
    assert not p.is_safe
    assert any("deadline_miss" in v for v in p.safety_vetoes)


# ---- 9. Decision construction safety ----

def test_decision_executable_in_real_cluster_is_false_by_default():
    cands = [ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=True)]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), cands)
    decision = choose_eval_workload_frontier_target(_profile(), points)
    assert decision.executable_in_real_cluster is False


def test_constructing_decision_with_real_execution_raises():
    with pytest.raises(ewm.EvalWorkloadFrontierSchemaError):
        ewm.EvalWorkloadFrontierDecision(
            workload_id="x", selected_candidate=None,
            current_candidate=None, selected_point=None,
            frontier_points=tuple(),
            action=ewm.EvalWorkloadFrontierAction.RECOMMEND_EVAL_FRONTIER,
            reason="r", executable_in_real_cluster=True)


# ---- 10. execute shim is shadow-only by default ----

def test_execute_shim_shadow_only_by_default():
    cands = [ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=True)]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), cands)
    decision = choose_eval_workload_frontier_target(_profile(), points)
    res = execute_eval_workload_frontier_decision(decision)
    assert res["mode"] == "shadow"
    assert res["executed"] is False


def test_execute_shim_real_disabled_without_executor():
    cands = [ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=True)]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), cands)
    decision = choose_eval_workload_frontier_target(_profile(), points)
    res = execute_eval_workload_frontier_decision(
        decision, allow_real_execution=True, executor=None)
    assert res["mode"] == "real_disabled"
    assert res["executed"] is False


# ---- 12. JSON round-trip ----

def test_eval_request_round_trip():
    r = EvalWorkloadRequest(
        request_id="X", turn_count=2, role_sequence_signature="h-g",
        token_count_source="char_div_4_proxy",
        provenance="synthetic_eval_fixture_v1",
        prompt_tokens_est=20, response_tokens_est=40,
        prompt_chars=80, response_chars=160)
    rt = EvalWorkloadRequest.from_dict(r.to_dict())
    assert rt == r


def test_frontier_decision_to_dict_serializable():
    cands = [ewm.EvalWorkloadFrontierCandidate(
        target_rho=0.75, concurrency=4, deadline_slack_hours=24.0,
        dedicated_fleet=True)]
    points = ewe.estimate_eval_workload_frontier(
        _profile(), _make_eval_requests(), cands)
    decision = choose_eval_workload_frontier_target(_profile(), points)
    payload = decision.to_dict()
    # Must be JSON-serializable.
    json.dumps(payload)


# ---- 14. Serving rho frontier still imports + works ----

def test_serving_rho_frontier_still_imports():
    # Confirm existing serving frontier modules still importable.
    from aurelius.frontier import (  # noqa: F401
        controller, dynamic_controller, dynamic_estimator, estimator,
        models, safety,
    )
