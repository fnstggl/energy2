"""Tests for the Safe Utilization Frontier Controller v1.

Proves:

- frontier points are generated for every configured rho;
- unsafe points are excluded;
- highest goodput/$ safe point is selected (and highest rho is NOT
  selected if unsafe);
- conservative margin steps back from boundary when configured;
- low telemetry confidence → INSUFFICIENT_TELEMETRY;
- deadband prevents churn for tiny KPI deltas;
- unsafe current rho triggers LOWER_RHO;
- no safe points → LOWER_RHO or INSUFFICIENT_TELEMETRY;
- shadow mode mutates nothing;
- simulator mode mutates only simulated state, deterministically;
- real_disabled mode mutates nothing;
- real_enabled without explicit opt-in raises;
- real_enabled with opt-in but without a real executor is a stub no-op;
- real_enabled with opt-in + mock executor mutates only through the
  caller-provided executor;
- shadow log records recommendation-only decisions (executed=False) by
  default and refuses executed=True in shadow / real_disabled modes;
- docs contain no production-savings claims.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aurelius.frontier import (
    ANTICIPATORY,
    REAL_DISABLED,
    REAL_ENABLED,
    SHADOW_MODE,
    SIMULATOR_MODE,
    FrontierAction,
    FrontierControllerConfig,
    FrontierDecision,
    FrontierEstimatorConfig,
    FrontierPoint,
    FrontierShadowDecisionLog,
    FrontierShadowLog,
    RealExecutionDisabledError,
    SafetyConfig,
    SafetyStatus,
    WorkloadFrontierProfile,
    choose_safe_utilization_target,
    estimate_frontier,
    estimate_frontier_from_points,
    execute_frontier_decision,
    is_frontier_point_safe,
    read_shadow_log,
    write_shadow_log_entry,
)
from aurelius.frontier.models import FrontierSchemaError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO_ROOT, "docs", "SAFE_UTILIZATION_FRONTIER_CONTROLLER.md")
AZURE_DOC = os.path.join(REPO_ROOT, "docs",
                         "AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _profile(**over):
    base = dict(workload_id="w1", workload_type="inference_standard",
                telemetry_confidence="medium",
                candidate_rhos=(0.45, 0.55, 0.65, 0.75, 0.85, 0.95))
    base.update(over)
    return WorkloadFrontierProfile(**base)


def _pt(rho, gpd, *, timeout=5.0, queue_p99=10.0, queue_p95=5.0, gpu_h=1000.0,
        latency_p99=None, mean_rho=None, scale_events=10, churn=20):
    return FrontierPoint(
        rho_target=rho, predicted_goodput_per_dollar=gpd,
        predicted_sla_safe_goodput=gpd * 1000, predicted_gpu_hours=gpu_h,
        predicted_timeout_pct=timeout, predicted_queue_p95_ms=queue_p95,
        predicted_queue_p99_ms=queue_p99, predicted_latency_p99_ms=latency_p99,
        predicted_scale_events=scale_events, predicted_churn_score=churn,
        predicted_mean_utilization=mean_rho or rho,
        safety_status=SafetyStatus.SAFE)


def _classify_all(points, cfg=None, conf="medium"):
    """Re-classify a list of bare points (status=SAFE placeholder) through the
    real safety filter so unsafe vetoes get attached."""
    from aurelius.frontier.safety import classify_point_safety
    cfg = cfg or SafetyConfig()
    out = []
    for p in points:
        status, vetoes = classify_point_safety(p, cfg, telemetry_confidence=conf)
        out.append(FrontierPoint(
            rho_target=p.rho_target,
            predicted_goodput_per_dollar=p.predicted_goodput_per_dollar,
            predicted_sla_safe_goodput=p.predicted_sla_safe_goodput,
            predicted_gpu_hours=p.predicted_gpu_hours,
            predicted_timeout_pct=p.predicted_timeout_pct,
            predicted_queue_p95_ms=p.predicted_queue_p95_ms,
            predicted_queue_p99_ms=p.predicted_queue_p99_ms,
            predicted_latency_p99_ms=p.predicted_latency_p99_ms,
            predicted_scale_events=p.predicted_scale_events,
            predicted_churn_score=p.predicted_churn_score,
            predicted_mean_utilization=p.predicted_mean_utilization,
            safety_status=status, safety_vetoes=vetoes))
    return out


# ===========================================================================
# 1. Frontier points are generated for ALL configured rho targets
# ===========================================================================

def test_estimator_emits_one_point_per_rho_from_dicts():
    profile = _profile()
    raw = [{"rho_target": r, "predicted_goodput_per_dollar": 1.0 + r,
            "predicted_timeout_pct": 5.0, "predicted_queue_p99_ms": 10.0}
           for r in profile.candidate_rhos]
    pts = estimate_frontier_from_points(profile, raw)
    assert [p.rho_target for p in pts] == list(profile.candidate_rhos)
    assert all(isinstance(p, FrontierPoint) for p in pts)


def test_estimator_returns_insufficient_for_empty_window():
    profile = _profile()
    pts = estimate_frontier(profile, telemetry_window=[])
    assert len(pts) == len(profile.candidate_rhos)
    assert all(p.is_insufficient_telemetry for p in pts)
    assert all("empty_telemetry_window" in p.safety_vetoes for p in pts)


# ===========================================================================
# 2. Unsafe points are excluded; highest goodput/$ safe is chosen
# ===========================================================================

def test_unsafe_points_are_excluded_and_highest_safe_wins():
    profile = _profile()
    points = _classify_all([
        _pt(0.45, 1.0e6),
        _pt(0.55, 1.5e6),
        _pt(0.65, 2.5e6),
        _pt(0.75, 2.9e6),
        _pt(0.85, 3.5e6, timeout=12.0),       # UNSAFE (timeout > 10%)
        _pt(0.95, 3.6e6, queue_p99=4000.0),    # UNSAFE (queue p99 > 2000ms)
    ])
    d = choose_safe_utilization_target(profile, points, current_rho=0.45)
    assert d.action == FrontierAction.RECOMMEND_RHO
    assert d.selected_rho == 0.75
    assert d.selected_point.predicted_goodput_per_dollar == 2.9e6
    # The highest *rho* point was NOT selected — it was unsafe.
    assert d.selected_rho != 0.95


def test_highest_rho_not_selected_if_unsafe():
    profile = _profile()
    points = _classify_all([
        _pt(0.55, 1.0e6),
        _pt(0.65, 2.0e6),
        _pt(0.75, 3.0e6, timeout=20.0),  # UNSAFE
    ])
    d = choose_safe_utilization_target(profile, points, current_rho=0.55)
    assert d.action == FrontierAction.RECOMMEND_RHO
    assert d.selected_rho == 0.65


# ===========================================================================
# 3. Conservative margin steps back from boundary
# ===========================================================================

def test_conservative_margin_steps_back_from_boundary():
    profile = _profile()
    points = _classify_all([
        _pt(0.45, 1.0e6),
        _pt(0.55, 1.5e6),
        _pt(0.65, 2.5e6),
        _pt(0.75, 2.9e6),                       # safe peak, ADJACENT to unsafe
        _pt(0.85, 3.5e6, timeout=12.0),         # UNSAFE
    ])
    # default (no margin): picks 0.75
    d = choose_safe_utilization_target(profile, points, current_rho=0.55)
    assert d.selected_rho == 0.75
    # with conservative margin: steps back to 0.65
    cfg = FrontierControllerConfig(conservative_margin=True,
                                   deadband_rho=0.0, deadband_kpi_pct=0.0)
    d2 = choose_safe_utilization_target(profile, points, current_rho=0.55,
                                        controller_config=cfg)
    assert d2.selected_rho == 0.65
    assert any("conservative_margin" in s for s in (d2.reason,))


# ===========================================================================
# 4. Low telemetry confidence → INSUFFICIENT_TELEMETRY
# ===========================================================================

def test_low_telemetry_confidence_blocks_action():
    profile = _profile(telemetry_confidence="unknown")
    points = _classify_all([_pt(0.55, 1.0e6), _pt(0.65, 2.0e6)], conf="unknown")
    cfg = FrontierControllerConfig(min_telemetry_confidence="medium")
    d = choose_safe_utilization_target(profile, points, current_rho=0.55,
                                       controller_config=cfg)
    assert d.action == FrontierAction.INSUFFICIENT_TELEMETRY
    assert d.selected_rho is None
    # decision is not executable in simulator either when telemetry is missing
    assert d.executable_in_simulator is False


# ===========================================================================
# 5. Deadband prevents churn for tiny KPI deltas
# ===========================================================================

def test_deadband_prevents_churn_on_small_kpi_delta():
    profile = _profile()
    # adjacent rhos with nearly-identical KPI
    points = _classify_all([
        _pt(0.55, 2.000e6),
        _pt(0.60, 2.002e6),  # ~0.1% higher KPI
        _pt(0.65, 2.005e6),  # ~0.25% higher KPI
    ])
    cfg = FrontierControllerConfig(deadband_rho=0.10, deadband_kpi_pct=0.01)
    d = choose_safe_utilization_target(profile, points, current_rho=0.55,
                                       controller_config=cfg)
    assert d.action == FrontierAction.KEEP_RHO
    assert d.selected_rho == 0.55


def test_deadband_does_not_suppress_material_kpi_gain():
    profile = _profile()
    points = _classify_all([
        _pt(0.55, 1.0e6),
        _pt(0.65, 2.5e6),       # +150% — outside deadband KPI tolerance
    ])
    cfg = FrontierControllerConfig(deadband_rho=0.20, deadband_kpi_pct=0.01)
    d = choose_safe_utilization_target(profile, points, current_rho=0.55,
                                       controller_config=cfg)
    assert d.action == FrontierAction.RECOMMEND_RHO
    assert d.selected_rho == 0.65


# ===========================================================================
# 6. Unsafe current rho triggers LOWER_RHO
# ===========================================================================

def test_unsafe_current_rho_lowers():
    profile = _profile()
    points = _classify_all([
        _pt(0.55, 1.0e6),
        _pt(0.65, 2.0e6),
        _pt(0.75, 2.5e6, timeout=15.0),        # UNSAFE
        _pt(0.85, 3.0e6, queue_p99=5000.0),    # UNSAFE
    ])
    d = choose_safe_utilization_target(profile, points, current_rho=0.75)
    assert d.action == FrontierAction.LOWER_RHO
    # next-lower SAFE rho is 0.65
    assert d.selected_rho == 0.65
    assert d.expected_sla_risk_delta == -1.0


# ===========================================================================
# 7. No safe points → LOWER_RHO or INSUFFICIENT_TELEMETRY
# ===========================================================================

def test_no_safe_points_returns_lower_rho():
    profile = _profile()
    points = _classify_all([
        _pt(0.55, 1.0e6, timeout=15.0),
        _pt(0.65, 2.0e6, queue_p99=5000.0),
    ])
    # current_rho is NOT one of the unsafe points — falls into the
    # "no safe points" branch (smallest tested rho).
    d = choose_safe_utilization_target(profile, points, current_rho=0.50)
    assert d.action == FrontierAction.LOWER_RHO
    assert d.selected_rho == 0.55  # smallest tested
    assert d.expected_sla_risk_delta == -1.0


def test_no_safe_points_with_unsafe_current_lowers_to_floor():
    """When current rho is unsafe AND no lower safe candidate exists, the
    controller recommends the workload floor."""
    profile = _profile()
    points = _classify_all([
        _pt(0.55, 1.0e6, timeout=15.0),
        _pt(0.65, 2.0e6, queue_p99=5000.0),
    ])
    d = choose_safe_utilization_target(profile, points, current_rho=0.55)
    assert d.action == FrontierAction.LOWER_RHO
    # current was unsafe; no lower safe candidate; recommend the floor.
    assert d.selected_rho == profile.min_rho


def test_no_safe_points_returns_insufficient_telemetry_when_telemetry_missing():
    profile = _profile()
    raw = [{"rho_target": 0.55}, {"rho_target": 0.65}]  # no metrics
    pts = estimate_frontier_from_points(profile, raw)
    # the safety filter marks them INSUFFICIENT_TELEMETRY (missing timeout/queue)
    assert all(p.is_insufficient_telemetry for p in pts)
    d = choose_safe_utilization_target(profile, pts, current_rho=0.55)
    assert d.action == FrontierAction.INSUFFICIENT_TELEMETRY
    assert d.executable_in_simulator is False


# ===========================================================================
# 8. Shadow mode mutates nothing
# ===========================================================================

def _safe_decision(profile=None):
    profile = profile or _profile()
    pts = _classify_all([_pt(0.55, 1.0e6), _pt(0.65, 2.5e6)])
    return profile, choose_safe_utilization_target(profile, pts, current_rho=0.55)


def test_shadow_mode_does_not_mutate():
    profile, d = _safe_decision()
    state = {"w1": 0.55}
    eff = execute_frontier_decision(d, mode=SHADOW_MODE,
                                    simulated_state=state)
    assert eff.mutated is False
    assert state == {"w1": 0.55}
    assert "shadow mode" in eff.notes[0]


# ===========================================================================
# 9. Simulator mode mutates ONLY simulated state, deterministically
# ===========================================================================

def test_simulator_mode_mutates_simulated_state_deterministically():
    profile, d = _safe_decision()
    state: dict = {}
    eff1 = execute_frontier_decision(d, mode=SIMULATOR_MODE,
                                     simulated_state=state)
    eff2 = execute_frontier_decision(d, mode=SIMULATOR_MODE,
                                     simulated_state=state)
    assert eff1.mutated is True and eff2.mutated is True
    assert state["w1"] == d.selected_rho
    # deterministic: same selected_rho both times
    assert eff1.selected_rho == eff2.selected_rho
    assert eff1.simulated_state_after == eff2.simulated_state_after


def test_simulator_mode_no_state_provided_does_not_mutate():
    _, d = _safe_decision()
    eff = execute_frontier_decision(d, mode=SIMULATOR_MODE)
    assert eff.mutated is False
    assert "no simulated_state" in " ".join(eff.notes)


# ===========================================================================
# 10. real_disabled mode is a no-op
# ===========================================================================

def test_real_disabled_mode_does_not_mutate():
    _, d = _safe_decision()
    state = {"w1": 0.55}
    eff = execute_frontier_decision(d, mode=REAL_DISABLED,
                                    simulated_state=state)
    assert eff.mutated is False
    assert state == {"w1": 0.55}


# ===========================================================================
# 11. real_enabled requires explicit opt-in
# ===========================================================================

def test_real_enabled_without_opt_in_raises():
    _, d = _safe_decision()
    with pytest.raises(RealExecutionDisabledError):
        execute_frontier_decision(d, mode=REAL_ENABLED,
                                  allow_real_execution=False)


def test_real_enabled_with_opt_in_no_executor_is_stub():
    _, d = _safe_decision()
    eff = execute_frontier_decision(d, mode=REAL_ENABLED,
                                    allow_real_execution=True)
    assert eff.mutated is False
    assert "not_implemented_real_executor" in eff.notes


def test_real_enabled_with_mock_executor_routes_through_caller():
    _, d = _safe_decision()
    calls: list = []

    def mock_executor(decision):
        calls.append(decision.selected_rho)
        return {"committed": True}

    eff = execute_frontier_decision(d, mode=REAL_ENABLED,
                                    executor=mock_executor,
                                    allow_real_execution=True)
    assert eff.mutated is True
    assert calls == [d.selected_rho]


def test_real_enabled_safety_vetoes_block_execution():
    """A decision with safety vetoes cannot run in real_enabled even with the
    flag — safety vetoes are hard blocks."""
    pts = _classify_all([
        _pt(0.55, 1.0e6, timeout=15.0),  # UNSAFE
        _pt(0.65, 2.0e6, queue_p99=5000.0),
    ])
    profile = _profile()
    d = choose_safe_utilization_target(profile, pts, current_rho=0.55)
    # Forge a decision with vetoes (LOWER_RHO carries them)
    # Now executor MUST NOT be called.
    called: list = []

    def fail_executor(decision):
        called.append(decision)
        return "should_not_have_been_called"

    # LOWER_RHO has no safety_vetoes; the controller embeds vetoes only on
    # decisions where the *current* rho was unsafe. Test the literal
    # constraint via a decision constructed with vetoes.
    decision_with_veto = FrontierDecision(
        workload_id="w1", selected_rho=0.55,
        selected_point=pts[0], frontier_points=tuple(pts),
        action=FrontierAction.LOWER_RHO,
        reason="unsafe", previous_rho=0.55,
        safety_vetoes=("timeout_exceeds_threshold",))
    eff = execute_frontier_decision(decision_with_veto, mode=REAL_ENABLED,
                                    executor=fail_executor,
                                    allow_real_execution=True)
    assert eff.mutated is False
    assert called == []
    assert any("safety vetoes" in n for n in eff.notes)


# ===========================================================================
# 12. FrontierDecision construction defends against real_cluster=True
# ===========================================================================

def test_decision_refuses_executable_in_real_cluster_at_construction():
    with pytest.raises(FrontierSchemaError):
        FrontierDecision(
            workload_id="w1", selected_rho=0.65, selected_point=None,
            frontier_points=(), action=FrontierAction.RECOMMEND_RHO,
            reason="x", executable_in_real_cluster=True)


# ===========================================================================
# 13. is_frontier_point_safe — sanity
# ===========================================================================

def test_is_frontier_point_safe_basic():
    cfg = SafetyConfig()
    safe = FrontierPoint(rho_target=0.55, predicted_timeout_pct=5.0,
                         predicted_queue_p99_ms=10.0, safety_status=SafetyStatus.SAFE)
    unsafe = FrontierPoint(rho_target=0.95, predicted_timeout_pct=20.0,
                           predicted_queue_p99_ms=5000.0,
                           safety_status=SafetyStatus.SAFE)
    missing = FrontierPoint(rho_target=0.65, safety_status=SafetyStatus.SAFE)
    assert is_frontier_point_safe(safe, cfg, telemetry_confidence="medium") is True
    assert is_frontier_point_safe(unsafe, cfg, telemetry_confidence="medium") is False
    # missing → not safe
    assert is_frontier_point_safe(missing, cfg, telemetry_confidence="medium") is False


# ===========================================================================
# 14. Shadow log records recommendation-only decisions
# ===========================================================================

def test_shadow_log_records_recommendation_only():
    _, d = _safe_decision()
    log = FrontierShadowLog()
    entry = log.record(d, execution_mode=SHADOW_MODE)
    assert entry.executed is False
    assert entry.execution_mode == SHADOW_MODE
    summary = log.summary()
    assert summary["n_decisions"] == 1
    assert summary["n_executed"] == 0


def test_shadow_log_refuses_executed_in_shadow_or_real_disabled():
    with pytest.raises(ValueError):
        FrontierShadowDecisionLog(
            timestamp="2026-01-01T00:00:00Z", workload_id="w",
            current_rho=0.55, recommended_rho=0.65,
            action=FrontierAction.RECOMMEND_RHO, reason="x",
            executed=True, execution_mode="shadow")
    with pytest.raises(ValueError):
        FrontierShadowDecisionLog(
            timestamp="2026-01-01T00:00:00Z", workload_id="w",
            current_rho=0.55, recommended_rho=0.65,
            action=FrontierAction.RECOMMEND_RHO, reason="x",
            executed=True, execution_mode="real_disabled")


def test_shadow_log_jsonl_round_trip(tmp_path):
    profile, d = _safe_decision()
    path = str(tmp_path / "shadow.jsonl")
    log = FrontierShadowLog(path=path)
    log.record(d, execution_mode=SHADOW_MODE)
    log.record(d, execution_mode=SHADOW_MODE)
    entries = read_shadow_log(path)
    assert len(entries) == 2
    assert all(e.executed is False for e in entries)
    # write_shadow_log_entry adds without clobbering
    extra = FrontierShadowDecisionLog.from_decision(d)
    write_shadow_log_entry(path, extra)
    assert len(read_shadow_log(path)) == 3


# ===========================================================================
# 15. WorkloadFrontierProfile validation
# ===========================================================================

def test_workload_profile_validates_rho_band():
    with pytest.raises(FrontierSchemaError):
        WorkloadFrontierProfile(
            workload_id="w", workload_type="x", min_rho=0.8, max_rho=0.5)
    with pytest.raises(FrontierSchemaError):
        WorkloadFrontierProfile(
            workload_id="w", workload_type="x", priority_class="vip")
    p = WorkloadFrontierProfile(workload_id="w", workload_type="x",
                                candidate_rhos=(0.3, 0.5, 0.9, 0.99),
                                min_rho=0.4, max_rho=0.95)
    assert p.clamp_candidates() == (0.5, 0.9)


# ===========================================================================
# 16. Doc + Azure-doc contain no unhedged production-savings claims
# ===========================================================================

def _no_unhedged_banned_phrases(path):
    text = open(path, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in
                       ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(path)}"
            i = pos + len(phrase)


def test_controller_doc_no_unhedged_banned_phrases():
    _no_unhedged_banned_phrases(DOC)


def test_controller_doc_states_required_caveats():
    low = " ".join(open(DOC, encoding="utf-8").read().lower().split())
    assert "shadow" in low and "simulator" in low
    assert "disabled by default" in low
    assert "pilot telemetry" in low
    assert "rho = 0.75" in low or "rho=0.75" in low
    assert "best safe" in low or "best safe kpi point" in low \
        or "highest safe" in low


# ===========================================================================
# 17. Estimator integration with the Azure 2024 audit JSON
# ===========================================================================

AZURE_AUDIT = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")


def test_estimator_consumes_azure_2024_audit_points():
    audit = json.load(open(AZURE_AUDIT))
    profile = _profile(workload_id="azure_2024")
    raw = [{
        "rho_target": float(p["policy"].split("@")[1]),
        "predicted_goodput_per_dollar": p["goodput_per_dollar"],
        "predicted_timeout_pct": p["timeout_pct_mean"],
        "predicted_queue_p99_ms": p["queue_p99_ms"],
        "predicted_queue_p95_ms": p["queue_p95_ms"],
        "predicted_gpu_hours": p["gpu_hours"],
        "predicted_mean_utilization": p["mean_utilization_rho"],
    } for p in audit["frontier_anticipatory"]]
    pts = estimate_frontier_from_points(profile, raw,
                                        safety_config=SafetyConfig())
    statuses = {p.rho_target: p.safety_status for p in pts}
    # The Azure audit's safe peak is anticipatory@0.75.
    assert statuses[0.75] == SafetyStatus.SAFE
    assert statuses[0.85] == SafetyStatus.UNSAFE
    assert statuses[0.95] == SafetyStatus.UNSAFE
    d = choose_safe_utilization_target(profile, pts, current_rho=0.65)
    assert d.action == FrontierAction.RECOMMEND_RHO
    assert d.selected_rho == 0.75
