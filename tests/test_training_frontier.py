"""Tests for Training Safe Utilization Frontier v1.

Hard invariants proved here:

1.  Training frontier does NOT import the serving rho controller's
    `choose_safe_utilization_target` / `choose_dynamic_rho`.
2.  Training candidates carry packing / backfill / reservation /
    fragmentation / gang-scheduling fields.
3.  Unsafe queue wait is vetoed.
4.  Unsafe starvation is vetoed.
5.  Unsafe fragmentation is vetoed.
6.  Unsafe gang-scheduling failure is vetoed (when the gate is enabled).
7.  Highest-goodput UNSAFE point is NOT selected.
8.  Highest-goodput SAFE point IS selected.
9.  Deadband preserves current policy on tiny KPI delta.
10. Insufficient telemetry returns INSUFFICIENT_TELEMETRY.
11. Philly estimator reports queue / starvation / backfill metrics.
12. Alibaba estimator reports packing / fragmentation /
    heterogeneity-aware fields.
13. Alibaba estimator does NOT invent queue wait when absent.
14. Training frontier runs against the committed Philly summary.
15. Training frontier runs against the committed Alibaba GPU summary.
16. Shadow logs round-trip JSONL.
17. Real execution is disabled by construction.
18. Docs contain no unhedged production-savings claims.
19. Existing Philly + Alibaba GPU ingestion / backtest tests still pass
    (proxied via importing those modules without modification).
20. Serving frontier tests still pass (proxied — the serving package
    public API is verified unchanged).
"""

from __future__ import annotations

import json
import os
import re

import pytest

from aurelius.frontier import (
    ALL_TRAINING_VETOES,
    PHILLY_POLICY_CANDIDATES,
    TrainingControllerConfig,
    TrainingFrontierAction,
    TrainingFrontierCandidate,
    TrainingFrontierDecision,
    TrainingFrontierPoint,
    TrainingFrontierShadowLog,
    TrainingRealExecutionDisabledError,
    TrainingSafetyConfig,
    TrainingSafetyStatus,
    TrainingWorkloadProfile,
    choose_training_frontier_target,
    classify_training_frontier_point,
    estimate_alibaba_gpu_training_frontier,
    estimate_philly_training_frontier,
    execute_training_frontier_decision,
    is_training_frontier_point_safe,
    load_alibaba_gpu_summary,
    load_philly_summary,
    read_training_shadow_log,
    write_training_shadow_log_entry,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DOC = os.path.join(REPO_ROOT, "docs",
                             "TRAINING_SAFE_UTILIZATION_FRONTIER.md")
TRAINING_RESULTS_DOC = os.path.join(
    REPO_ROOT, "docs", "TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md")
TRAINING_RESULTS_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier", "training_frontier_summary.json")
PHILLY_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "philly", "processed",
    "philly_backtest_summary.json")
ALIBABA_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "alibaba_gpu", "processed",
    "alibaba_gpu_backtest_summary.json")


# ---------------------------------------------------------------------------
# Helpers — synthesize safe / unsafe / insufficient points
# ---------------------------------------------------------------------------

def _safe_point(policy: str, gpd: float = 1.0,
                packing_density: float = 0.7,
                queue_wait_p99_s: float = 1000.0,
                starvation_pct: float = 0.0,
                frag_pct: float = 5.0,
                completed_work: float = 100.0) -> TrainingFrontierPoint:
    cand = TrainingFrontierCandidate(
        packing_density_target=packing_density,
        backfill_aggressiveness=1.0,
        large_job_reservation_fraction=0.0,
        gang_scheduling_strictness=0.5,
        source_policy=policy)
    p = TrainingFrontierPoint(
        candidate=cand,
        predicted_goodput_per_dollar=gpd,
        predicted_gpu_occupancy=packing_density,
        predicted_packing_density=packing_density,
        predicted_gpu_hours=10.0,
        predicted_completed_work=completed_work,
        predicted_queue_wait_p95_s=queue_wait_p99_s * 0.7,
        predicted_queue_wait_p99_s=queue_wait_p99_s,
        predicted_starvation_rate_pct=starvation_pct,
        predicted_fragmentation_block_rate_pct=frag_pct,
        predicted_gang_scheduling_failure_pct=0.0,
        predicted_backfill_success_rate_pct=50.0,
        predicted_cost=100.0,
        safety_status=TrainingSafetyStatus.SAFE,
    )
    # Re-classify to populate safety status against default config.
    status, vetoes = classify_training_frontier_point(
        p, TrainingSafetyConfig(max_gang_scheduling_failure_pct=None),
        telemetry_confidence="medium")
    return TrainingFrontierPoint(
        candidate=p.candidate,
        predicted_goodput_per_dollar=p.predicted_goodput_per_dollar,
        predicted_gpu_occupancy=p.predicted_gpu_occupancy,
        predicted_packing_density=p.predicted_packing_density,
        predicted_gpu_hours=p.predicted_gpu_hours,
        predicted_completed_work=p.predicted_completed_work,
        predicted_queue_wait_p95_s=p.predicted_queue_wait_p95_s,
        predicted_queue_wait_p99_s=p.predicted_queue_wait_p99_s,
        predicted_starvation_rate_pct=p.predicted_starvation_rate_pct,
        predicted_fragmentation_block_rate_pct=
            p.predicted_fragmentation_block_rate_pct,
        predicted_gang_scheduling_failure_pct=
            p.predicted_gang_scheduling_failure_pct,
        predicted_backfill_success_rate_pct=
            p.predicted_backfill_success_rate_pct,
        predicted_cost=p.predicted_cost,
        safety_status=status, safety_vetoes=tuple(vetoes),
    )


# ===========================================================================
# 1 — Training frontier does NOT import serving rho controller
# ===========================================================================

def test_training_modules_do_not_import_serving_controllers():
    """Read the source files of all training modules and assert no
    imports of the serving rho controllers."""
    training_files = [
        os.path.join(REPO_ROOT, "aurelius", "frontier", "training_models.py"),
        os.path.join(REPO_ROOT, "aurelius", "frontier", "training_safety.py"),
        os.path.join(REPO_ROOT, "aurelius", "frontier",
                     "training_philly.py"),
        os.path.join(REPO_ROOT, "aurelius", "frontier",
                     "training_alibaba_gpu.py"),
        os.path.join(REPO_ROOT, "aurelius", "frontier",
                     "training_controller.py"),
        os.path.join(REPO_ROOT, "aurelius", "frontier",
                     "training_shadow.py"),
    ]
    forbidden = (
        "from .controller import",
        "from .dynamic_controller import",
        "choose_safe_utilization_target",
        "choose_dynamic_rho",
        "from .dynamic_estimator import",
        "from .estimator import",
    )
    for path in training_files:
        text = open(path, encoding="utf-8").read()
        for token in forbidden:
            assert token not in text, \
                (f"{os.path.basename(path)} should NOT reference "
                 f"{token!r} (training frontier is a sibling, not a "
                 "subclass)")


def test_training_decision_is_distinct_class_from_serving():
    from aurelius.frontier import FrontierDecision
    assert TrainingFrontierDecision is not FrontierDecision
    # Field surface is different — training carries packing/backfill
    # fields the serving FrontierDecision does not.
    training_fields = set(TrainingFrontierDecision.__dataclass_fields__)
    assert "expected_fragmentation_delta_pct" in training_fields
    assert "expected_starvation_delta_pct" in training_fields


# ===========================================================================
# 2 — Candidate carries training-specific knobs
# ===========================================================================

def test_candidate_fields_include_training_knobs():
    fields = set(TrainingFrontierCandidate.__dataclass_fields__)
    for required in ("occupancy_target", "packing_density_target",
                     "backfill_aggressiveness",
                     "large_job_reservation_fraction",
                     "fragmentation_budget",
                     "gang_scheduling_strictness", "preemption_allowed",
                     "checkpoint_overhead_budget",
                     "heterogeneity_preference",
                     "price_aware_gpu_routing_enabled"):
        assert required in fields, f"candidate missing field {required!r}"


def test_candidate_validates_ranges_and_enum():
    with pytest.raises(Exception):
        TrainingFrontierCandidate(occupancy_target=1.5)
    with pytest.raises(Exception):
        TrainingFrontierCandidate(heterogeneity_preference="bogus")


# ===========================================================================
# 3-6 — Safety vetoes
# ===========================================================================

def test_unsafe_queue_wait_p99_vetoed():
    p = _safe_point("p", queue_wait_p99_s=200000.0)  # > default p99 12h
    status, vetoes = classify_training_frontier_point(
        p, TrainingSafetyConfig(), telemetry_confidence="medium")
    assert status == TrainingSafetyStatus.UNSAFE
    assert "queue_wait_p99_exceeded" in vetoes


def test_unsafe_starvation_vetoed():
    p = _safe_point("p", starvation_pct=20.0)  # > default 5%
    status, vetoes = classify_training_frontier_point(
        p, TrainingSafetyConfig(), telemetry_confidence="medium")
    assert status == TrainingSafetyStatus.UNSAFE
    assert "starvation_rate_exceeded" in vetoes


def test_unsafe_fragmentation_vetoed():
    p = _safe_point("p", frag_pct=80.0)  # > default 25%
    status, vetoes = classify_training_frontier_point(
        p, TrainingSafetyConfig(), telemetry_confidence="medium")
    assert status == TrainingSafetyStatus.UNSAFE
    assert "fragmentation_budget_exceeded" in vetoes


def test_unsafe_gang_scheduling_failure_vetoed_when_gate_enabled():
    cand = TrainingFrontierCandidate(packing_density_target=0.7,
                                      source_policy="p")
    point = TrainingFrontierPoint(
        candidate=cand,
        predicted_goodput_per_dollar=1.0,
        predicted_queue_wait_p99_s=1000.0,
        predicted_starvation_rate_pct=0.0,
        predicted_fragmentation_block_rate_pct=5.0,
        predicted_gang_scheduling_failure_pct=25.0,  # > default 10%
        predicted_completed_work=10.0,
    )
    status, vetoes = classify_training_frontier_point(
        point, TrainingSafetyConfig(), telemetry_confidence="medium")
    assert status == TrainingSafetyStatus.UNSAFE
    assert "gang_scheduling_failure_exceeded" in vetoes


# ===========================================================================
# 7-8 — Controller selection
# ===========================================================================

def test_controller_does_not_select_highest_kpi_unsafe_point():
    safe = _safe_point("safe", gpd=1.0)
    unsafe = TrainingFrontierPoint(
        candidate=TrainingFrontierCandidate(packing_density_target=0.95,
                                             source_policy="dense"),
        predicted_goodput_per_dollar=999.0,  # huge KPI but unsafe
        predicted_queue_wait_p99_s=200000.0,
        predicted_starvation_rate_pct=0.0,
        predicted_fragmentation_block_rate_pct=5.0,
        predicted_gang_scheduling_failure_pct=0.0,
        predicted_completed_work=100.0,
        safety_status=TrainingSafetyStatus.UNSAFE,
        safety_vetoes=("queue_wait_p99_exceeded",),
    )
    dec = choose_training_frontier_target(
        [unsafe, safe], workload_id="t")
    assert dec.selected_candidate.source_policy == "safe"


def test_controller_selects_highest_safe_goodput():
    low = _safe_point("low", gpd=1.0)
    high = _safe_point("high", gpd=5.0, packing_density=0.75)
    dec = choose_training_frontier_target([low, high], workload_id="t")
    assert dec.selected_candidate.source_policy == "high"
    assert dec.action == TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER


# ===========================================================================
# 9 — Deadband
# ===========================================================================

def test_deadband_preserves_current_policy_on_small_delta():
    cur = _safe_point("cur", gpd=10.0, packing_density=0.70)
    next_ = _safe_point("next", gpd=10.05, packing_density=0.71)  # +0.5%
    dec = choose_training_frontier_target(
        [cur, next_], current_candidate=cur.candidate,
        config=TrainingControllerConfig(deadband_kpi_pct=0.01,
                                         deadband_packing_density=0.05),
        workload_id="t")
    assert dec.action == TrainingFrontierAction.KEEP_CURRENT_POLICY
    assert dec.selected_candidate.source_policy == "cur"


# ===========================================================================
# 10 — INSUFFICIENT_TELEMETRY
# ===========================================================================

def test_insufficient_telemetry_returns_insufficient_action():
    insuff = TrainingFrontierPoint(
        candidate=TrainingFrontierCandidate(source_policy="x"),
        predicted_goodput_per_dollar=None,
        predicted_completed_work=None,
        safety_status=TrainingSafetyStatus.INSUFFICIENT_TELEMETRY,
        safety_vetoes=("low_telemetry_confidence",),
    )
    dec = choose_training_frontier_target([insuff], workload_id="t")
    assert dec.action == TrainingFrontierAction.INSUFFICIENT_TELEMETRY


def test_low_telemetry_confidence_triggers_insufficient():
    safe = _safe_point("safe", gpd=1.0)
    dec = choose_training_frontier_target(
        [safe], workload_id="t",
        config=TrainingControllerConfig(min_telemetry_confidence="high"),
        telemetry_confidence="low")
    assert dec.action == TrainingFrontierAction.INSUFFICIENT_TELEMETRY


# ===========================================================================
# 11 — Philly estimator reports queue / starvation / backfill
# ===========================================================================

def test_philly_estimator_reports_queue_starvation_backfill():
    ph = load_philly_summary()
    points = estimate_philly_training_frontier(ph)
    assert points, "expected at least one Philly point"
    bf = next(p for p in points
              if p.candidate.source_policy == "best_fit")
    assert bf.predicted_queue_wait_p99_s is not None
    assert bf.predicted_starvation_rate_pct is not None
    assert bf.predicted_backfill_success_rate_pct is not None
    assert bf.predicted_fragmentation_block_rate_pct is not None
    # Retry waste is sourced from attempt_analysis
    assert bf.predicted_retry_waste_gpu_hours is not None


# ===========================================================================
# 12-13 — Alibaba estimator: packing yes, queue NOT INVENTED
# ===========================================================================

def test_alibaba_estimator_reports_packing_and_fragmentation():
    ag = load_alibaba_gpu_summary()
    points = estimate_alibaba_gpu_training_frontier(ag)
    assert points
    ca = next(p for p in points
              if p.candidate.source_policy == "constraint_aware")
    assert ca.predicted_gpu_occupancy is not None
    assert ca.predicted_packing_density is not None
    assert ca.predicted_fragmentation_block_rate_pct is not None
    # Candidate carries heterogeneity + price-aware fields
    assert ca.candidate.heterogeneity_preference is not None
    assert ca.candidate.price_aware_gpu_routing_enabled is not None


def test_alibaba_estimator_does_not_invent_queue_wait():
    ag = load_alibaba_gpu_summary()
    points = estimate_alibaba_gpu_training_frontier(ag)
    for p in points:
        assert p.predicted_queue_wait_p95_s is None, \
            "Alibaba packing summary lacks queue wait — must NOT invent"
        assert p.predicted_queue_wait_p99_s is None
        assert p.predicted_gang_scheduling_failure_pct is None
        assert p.predicted_retry_waste_gpu_hours is None


# ===========================================================================
# 14-15 — Training frontier runs against committed summaries
# ===========================================================================

def test_training_frontier_runs_against_philly_summary():
    assert os.path.exists(PHILLY_SUMMARY)
    ph = load_philly_summary()
    points = estimate_philly_training_frontier(ph)
    assert points
    dec = choose_training_frontier_target(
        points,
        current_candidate=PHILLY_POLICY_CANDIDATES["constraint_aware"],
        workload_id="philly")
    assert dec.action in (
        TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER,
        TrainingFrontierAction.KEEP_CURRENT_POLICY,
        TrainingFrontierAction.LOWER_PACKING_PRESSURE,
        TrainingFrontierAction.RESERVE_FOR_LARGE_JOBS,
    )


def test_training_frontier_runs_against_alibaba_summary():
    assert os.path.exists(ALIBABA_SUMMARY)
    ag = load_alibaba_gpu_summary()
    points = estimate_alibaba_gpu_training_frontier(ag)
    assert points
    dec = choose_training_frontier_target(points, workload_id="alibaba")
    assert dec.selected_candidate is not None


# ===========================================================================
# 16 — Shadow log round-trip
# ===========================================================================

def test_shadow_log_jsonl_round_trip(tmp_path):
    path = str(tmp_path / "training_shadow.jsonl")
    dec = TrainingFrontierDecision(
        workload_id="w",
        selected_candidate=TrainingFrontierCandidate(
            packing_density_target=0.7, source_policy="best_fit"),
        current_candidate=TrainingFrontierCandidate(
            source_policy="fifo"),
        selected_point=None,
        frontier_points=(),
        action=TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER,
        reason="t",
        confidence="medium")
    entry = TrainingFrontierShadowLog.from_decision(dec, timestamp_s=1.0)
    write_training_shadow_log_entry(path, entry)
    write_training_shadow_log_entry(path, entry)
    read = read_training_shadow_log(path)
    assert len(read) == 2
    assert read[0].workload_id == "w"
    assert read[0].action == TrainingFrontierAction.RECOMMEND_TRAINING_FRONTIER


# ===========================================================================
# 17 — Real execution disabled by construction
# ===========================================================================

def test_training_decision_default_recommendation_only():
    dec = TrainingFrontierDecision(
        workload_id="w",
        selected_candidate=TrainingFrontierCandidate(source_policy="x"),
        current_candidate=None,
        selected_point=None,
        frontier_points=(),
        action=TrainingFrontierAction.KEEP_CURRENT_POLICY,
        reason="t", confidence="medium")
    assert dec.executable_in_real_cluster is False
    assert dec.execution_mode == "shadow"


def test_training_real_execution_requires_explicit_opt_in():
    dec = TrainingFrontierDecision(
        workload_id="w",
        selected_candidate=TrainingFrontierCandidate(source_policy="x"),
        current_candidate=None,
        selected_point=None,
        frontier_points=(),
        action=TrainingFrontierAction.KEEP_CURRENT_POLICY,
        reason="t", confidence="medium")
    with pytest.raises(TrainingRealExecutionDisabledError):
        execute_training_frontier_decision(dec, mode="real_enabled")


def test_training_real_execution_returns_stub_without_executor():
    dec = TrainingFrontierDecision(
        workload_id="w",
        selected_candidate=TrainingFrontierCandidate(source_policy="x"),
        current_candidate=None,
        selected_point=None,
        frontier_points=(),
        action=TrainingFrontierAction.KEEP_CURRENT_POLICY,
        reason="t", confidence="medium")
    out = execute_training_frontier_decision(
        dec, mode="real_enabled", allow_real_execution=True)
    assert out["mutated"] is False
    assert "not_implemented_real_executor" in out["notes"]


def test_training_shadow_mode_mutates_nothing():
    dec = TrainingFrontierDecision(
        workload_id="w",
        selected_candidate=TrainingFrontierCandidate(source_policy="x"),
        current_candidate=None,
        selected_point=None,
        frontier_points=(),
        action=TrainingFrontierAction.KEEP_CURRENT_POLICY,
        reason="t", confidence="medium")
    out = execute_training_frontier_decision(dec, mode="shadow")
    assert out["mutated"] is False


# ===========================================================================
# 18 — Docs check
# ===========================================================================

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


@pytest.mark.parametrize("doc_path",
                          [TRAINING_DOC, TRAINING_RESULTS_DOC])
def test_docs_have_no_unhedged_production_savings_claims(doc_path):
    assert os.path.exists(doc_path), f"missing doc {doc_path}"
    text = open(doc_path, encoding="utf-8").read().lower()
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
                f"unhedged '{phrase}' in {os.path.basename(doc_path)}"
            i = pos + len(phrase)


def test_design_doc_states_required_caveats():
    text = open(TRAINING_DOC, encoding="utf-8").read().lower()
    low = " ".join(text.split())
    for phrase in ("sibling", "training", "philly", "alibaba",
                   "mit supercloud", "opt-in", "disabled by default",
                   "shadow", "pilot telemetry"):
        assert phrase in low, f"design doc missing caveat: {phrase!r}"


# ===========================================================================
# 19 — Existing Philly / Alibaba GPU tests still pass (proxy)
# ===========================================================================

def test_philly_and_alibaba_trace_modules_importable():
    """Importing the canonical trace modules must not raise; the
    training frontier does not touch them."""
    import aurelius.traces.philly  # noqa: F401
    import aurelius.traces.alibaba_gpu  # noqa: F401
    import aurelius.traces.gpu_scheduling  # noqa: F401
    import aurelius.traces.gpu_packing  # noqa: F401


# ===========================================================================
# 20 — Serving frontier still works (proxy)
# ===========================================================================

def test_serving_frontier_public_api_unchanged():
    import aurelius.frontier as fr
    for required in ("FrontierAction", "FrontierDecision",
                     "FrontierPoint", "WorkloadFrontierProfile",
                     "SafetyConfig", "SafetyStatus",
                     "FrontierControllerConfig",
                     "choose_safe_utilization_target",
                     "estimate_frontier", "estimate_frontier_from_points",
                     "execute_frontier_decision",
                     "ServingTelemetryTick",
                     "DynamicFrontierDecision",
                     "estimate_dynamic_frontier",
                     "choose_dynamic_rho"):
        assert hasattr(fr, required), \
            f"serving frontier missing public symbol {required!r}"


def test_training_frontier_public_api_present():
    import aurelius.frontier as fr
    for required in ("TrainingFrontierAction",
                     "TrainingFrontierCandidate",
                     "TrainingFrontierPoint",
                     "TrainingFrontierDecision",
                     "TrainingFrontierShadowLog",
                     "TrainingSafetyConfig", "TrainingSafetyStatus",
                     "TRAINING_FRONTIER_ACTIONS",
                     "ALL_TRAINING_VETOES",
                     "classify_training_frontier_point",
                     "is_training_frontier_point_safe",
                     "choose_training_frontier_target",
                     "execute_training_frontier_decision",
                     "TrainingRealExecutionDisabledError",
                     "estimate_philly_training_frontier",
                     "estimate_alibaba_gpu_training_frontier",
                     "PHILLY_POLICY_CANDIDATES",
                     "ALIBABA_POLICY_CANDIDATES",
                     "TrainingWorkloadProfile",
                     "read_training_shadow_log",
                     "write_training_shadow_log_entry",
                     "load_philly_summary",
                     "load_alibaba_gpu_summary"):
        assert hasattr(fr, required), \
            f"training frontier missing public symbol {required!r}"


# ===========================================================================
# Benchmark JSON exists + verdicts are valid
# ===========================================================================

def test_training_frontier_summary_exists_and_is_well_formed():
    assert os.path.exists(TRAINING_RESULTS_JSON), \
        f"missing {TRAINING_RESULTS_JSON}"
    d = json.load(open(TRAINING_RESULTS_JSON))
    assert "config" in d and "per_trace" in d and "synthesis" in d
    valid_verdicts = {"TIE", "TRAINING_FRONTIER_WIN",
                       "TRAINING_FRONTIER_LOSS", "INSUFFICIENT_DATA"}
    for r in d["per_trace"]:
        if not r["applicable"]:
            continue
        assert r["verdict"] in valid_verdicts, \
            f"unknown verdict {r['verdict']!r}"


def test_committed_philly_and_alibaba_summaries_unchanged_by_audit(tmp_path):
    """Re-running the benchmark must NOT modify the committed Philly /
    Alibaba GPU backtest summaries."""
    import hashlib

    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    before_ph = _sha(PHILLY_SUMMARY)
    before_ag = _sha(ALIBABA_SUMMARY)
    # Re-invoke main without committing output to the standard paths
    from scripts import run_training_frontier as rt
    rt.main(["--out-json", str(tmp_path / "x.json"),
             "--out-md", str(tmp_path / "x.md")])
    assert _sha(PHILLY_SUMMARY) == before_ph
    assert _sha(ALIBABA_SUMMARY) == before_ag
