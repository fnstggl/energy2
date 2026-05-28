"""Tests for the per-workload-type benchmark reporting layer.

Covers:
  - scenario classification (intent, primary workload type, telemetry-failsafe)
  - headline-baseline selection (workload-relevant, NOT FIFO)
  - outcome analysis (alpha vs safety classifier, multi-cause loss reasons)
  - cross-scenario aggregation (median+mean, telemetry-failsafe separation)
  - markdown invariants (forbidden phrases, required columns)
  - end-to-end via ConstraintBenchmarkRunner
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurelius.benchmarks import (
    ConstraintBenchmarkRunner,
    CrossScenarioReport,
    OutcomeAnalysis,
    PerScenarioRow,
    ScenarioMetadata,
    analyze_outcome,
    classify_scenario,
    select_headline_baseline,
)
from aurelius.benchmarks.per_workload import (
    LOSS_REASON_CODES,
    OUTCOMES,
    WORKLOAD_TYPE_ALIASES,
    WORKLOAD_TYPES,
    _infer_primary_workload_type,
)
from aurelius.simulation.cluster.scenarios import _BUILTIN_SCENARIOS

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# classify_scenario
# ---------------------------------------------------------------------------

def test_classify_scenario_non_none_for_every_builtin():
    for name, raw in _BUILTIN_SCENARIOS.items():
        m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
        assert isinstance(m, ScenarioMetadata)
        assert m.scenario_name == name
        assert m.primary_workload_type
        assert m.optimization_intent
        assert m.relevant_baselines


def test_energy_scenario_classifies_as_energy_arbitrage_with_price_baselines():
    raw = _BUILTIN_SCENARIOS["energy_price_arbitrage_multiregion"]
    m = classify_scenario(
        "energy_price_arbitrage_multiregion",
        raw.get("expected_primary_constraint"), raw,
    )
    assert m.optimization_intent == "energy_arbitrage"
    assert "current_price_only" in m.relevant_baselines
    assert "greedy_energy" in m.relevant_baselines
    assert m.is_telemetry_failsafe is False


def test_thermal_scenario_classifies_as_thermal_spread():
    raw = _BUILTIN_SCENARIOS["thermal_hotspot_mixed_cluster"]
    m = classify_scenario(
        "thermal_hotspot_mixed_cluster",
        raw.get("expected_primary_constraint"), raw,
    )
    assert m.optimization_intent == "thermal_spread"
    assert m.is_telemetry_failsafe is False


@pytest.mark.parametrize("name", [
    "degraded_topology_telemetry",
    "partial_utilization_telemetry",
    "low_confidence_energy_telemetry",
])
def test_telemetry_scenarios_classify_as_failsafe(name):
    raw = _BUILTIN_SCENARIOS[name]
    m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
    assert m.primary_workload_type == "telemetry_fail_safe"
    assert m.optimization_intent == "safety_keep"
    assert m.is_telemetry_failsafe is True


@pytest.mark.parametrize("name", [
    "fragmentation_stranded_capacity",
    "underutilization_stranded_capacity",
])
def test_utilization_scenarios_classify_with_packing_baselines(name):
    raw = _BUILTIN_SCENARIOS[name]
    m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
    assert m.optimization_intent == "fragmentation_packing"
    # At least one packing baseline is in relevant_baselines.
    assert any(p in m.relevant_baselines
               for p in ("first_fit", "best_fit", "first_fit_decreasing"))


def test_explicit_raw_override_wins():
    raw = {
        "primary_workload_type": "batch_training",
        "optimization_intent": "energy_arbitrage",
        "relevant_baselines": ["fifo", "current_price_only"],
        "headline_baseline": "greedy_energy",
        "goodput_unit": "token_equivalent",
        "sla_slo_type": "throughput_only",
        "workloads": [{"priority_tier": "latency_sensitive",
                       "workload_type": "inference"}],
    }
    m = classify_scenario("synthetic_override", "energy_bound", raw)
    assert m.primary_workload_type == "batch_training"
    assert m.headline_baseline_override == "greedy_energy"
    assert m.goodput_unit == "token_equivalent"


def test_workload_type_tie_is_mixed():
    # 1 batch + 1 inference → no strict plurality.
    workloads = [
        {"priority_tier": "batch", "workload_type": "batch_training"},
        {"priority_tier": "standard", "workload_type": "inference"},
    ]
    assert _infer_primary_workload_type(workloads) == "mixed"


def test_strict_plurality_batch_wins_3_of_5():
    workloads = (
        [{"priority_tier": "batch", "workload_type": "batch_training"}] * 3
        + [{"priority_tier": "standard", "workload_type": "inference"}] * 2
    )
    assert _infer_primary_workload_type(workloads) == "batch_training"


def test_classification_golden_fixture():
    """Classification of every builtin scenario must match the golden fixture."""
    golden = json.loads(
        (FIXTURES / "scenario_classification.json").read_text()
    )
    actual = {}
    for name, raw in _BUILTIN_SCENARIOS.items():
        m = classify_scenario(
            name, raw.get("expected_primary_constraint"), raw,
        )
        actual[name] = m.primary_workload_type
    assert actual == golden, (
        "Classification drift vs fixtures/scenario_classification.json. "
        "If intentional, regenerate the fixture."
    )


# ---------------------------------------------------------------------------
# select_headline_baseline
# ---------------------------------------------------------------------------

class _FakeKPI:
    """Lightweight KPI stand-in used in unit tests."""

    def __init__(self, *, goodput=100.0, p99=200.0, sla=0, thermal=0,
                 scale_up_recommended=0, scale_up_applied=0,
                 total_migrations=0, total_net_savings=None,
                 blocked_scale_for_low_value_queue_relief=0):
        self.sla_safe_goodput_per_infra_dollar = goodput
        self.p99_latency_ms = p99
        self.total_sla_violations = sla
        self.total_thermal_throttle_ticks = thermal
        self.scale_up_recommended = scale_up_recommended
        self.scale_up_applied = scale_up_applied
        self.total_migrations = total_migrations
        self.total_net_savings = total_net_savings
        self.blocked_scale_for_low_value_queue_relief = (
            blocked_scale_for_low_value_queue_relief
        )


def _make_metadata(**over):
    base = dict(
        scenario_name="synthetic",
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
        headline_baseline_override=None,
        goodput_unit="tokens",
        sla_slo_type="p99_latency",
        is_telemetry_failsafe=False,
    )
    base.update(over)
    return ScenarioMetadata(**base)


def test_select_headline_batch_picks_max_safe():
    md = _make_metadata()
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200, sla=2),
        # current_price_only has highest goodput but p99 is 5x FIFO → unsafe
        "current_price_only": _FakeKPI(goodput=300, p99=1100, sla=2),
        "greedy_energy": _FakeKPI(goodput=150, p99=210, sla=2),
        "sla_aware": _FakeKPI(goodput=120, p99=190, sla=2),
    }
    name, rationale = select_headline_baseline(md, policy)
    # current_price_only disqualified (p99 > 1.5x FIFO), greedy_energy wins
    assert name == "greedy_energy"
    assert "strongest_safe_relevant_baseline" in rationale


def test_select_headline_inference_critical_falls_back_to_fifo_when_sla_aware_violates():
    md = _make_metadata(primary_workload_type="inference_critical")
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200, sla=2),
        # sla_aware has MORE SLA violations than fifo → not safe.
        "sla_aware": _FakeKPI(goodput=200, p99=200, sla=5),
    }
    name, _ = select_headline_baseline(md, policy)
    assert name == "fifo"


def test_select_headline_telemetry_failsafe_picks_fifo():
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    policy = {
        "fifo": _FakeKPI(goodput=100),
        "sla_aware": _FakeKPI(goodput=200),
    }
    name, rationale = select_headline_baseline(md, policy)
    assert name == "fifo"
    assert "telemetry_failsafe" in rationale


def test_select_headline_explicit_override_wins():
    md = _make_metadata(headline_baseline_override="greedy_energy")
    policy = {
        "fifo": _FakeKPI(),
        "greedy_energy": _FakeKPI(),
    }
    name, rationale = select_headline_baseline(md, policy)
    assert name == "greedy_energy"
    assert rationale == "explicit_override"


def test_select_headline_deterministic_across_two_calls():
    md = _make_metadata()
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200),
        "current_price_only": _FakeKPI(goodput=200, p99=200),
        "greedy_energy": _FakeKPI(goodput=150, p99=200),
        "sla_aware": _FakeKPI(goodput=120, p99=200),
    }
    a = select_headline_baseline(md, policy)
    b = select_headline_baseline(md, policy)
    assert a == b


# ---------------------------------------------------------------------------
# analyze_outcome
# ---------------------------------------------------------------------------

def test_analyze_outcome_alpha_win_above_1pct():
    md = _make_metadata()
    ca = _FakeKPI(goodput=120)
    hd = _FakeKPI(goodput=100)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "ALPHA_WIN"
    assert out.margin_pct > 1.0


def test_analyze_outcome_safety_win_in_tie_band_with_better_p99():
    md = _make_metadata()
    # Within 1% on goodput BUT 2x better p99.
    ca = _FakeKPI(goodput=100.5, p99=100)
    hd = _FakeKPI(goodput=100.0, p99=300)
    others = {"fifo": hd, "constraint_aware": ca,
              "sla_aware": _FakeKPI(goodput=99, p99=250)}
    out = analyze_outcome(md, ca, hd, others)
    assert out.outcome == "SAFETY_WIN"
    assert any("p99" in e for e in out.safety_evidence)


def test_analyze_outcome_tie_within_1pct_no_safety_evidence():
    md = _make_metadata()
    ca = _FakeKPI(goodput=100.5, p99=200)
    hd = _FakeKPI(goodput=100.0, p99=200)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "TIE"


def test_analyze_outcome_loss_below_minus_1pct():
    md = _make_metadata()
    ca = _FakeKPI(goodput=80, scale_up_recommended=0, total_migrations=0,
                  total_net_savings=-10.0)
    hd = _FakeKPI(goodput=100, scale_up_applied=2)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        # Headline is an energy baseline → missing_forecast_lookahead is the
        # meaningful loss reason (ISSUE 7 gate).
        headline_name="current_price_only",
    )
    assert out.outcome == "LOSS"
    # Energy arbitrage vs an energy baseline with non-positive net_savings → flagged
    assert "missing_forecast_lookahead" in out.loss_reasons


def test_analyze_outcome_keep_correct_for_telemetry_failsafe():
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep", is_telemetry_failsafe=True,
    )
    ca = _FakeKPI(goodput=100, sla=0)
    hd = _FakeKPI(goodput=100, sla=0)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "KEEP_CORRECT"


def test_analyze_outcome_multi_cause_loss_reasons():
    """Telemetry-failsafe scenario that ALSO loses on alpha and blocked an
    aggressive scale-up — both reasons should appear."""
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    ca = _FakeKPI(goodput=80,
                  blocked_scale_for_low_value_queue_relief=5,
                  scale_up_applied=0)
    hd = _FakeKPI(goodput=100, scale_up_applied=3)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "LOSS"
    assert "telemetry_fail_safe" in out.loss_reasons
    assert "over_conservative_gate" in out.loss_reasons


# ---------------------------------------------------------------------------
# CrossScenarioReport
# ---------------------------------------------------------------------------

class _FakeBenchmarkReport:
    def __init__(self, scenario_metadata, aggregated, headline_name,
                 headline_rationale, outcome):
        self.scenario_metadata = scenario_metadata
        self.aggregated = aggregated
        self.headline_baseline_name = headline_name
        self.headline_baseline_rationale = headline_rationale
        self.outcome = outcome
        self.expected_primary_constraint = None


class _FakeBenchmarkResult:
    def __init__(self, report):
        self.report = report


def _build_fake_results(scenarios):
    """scenarios: list of (name, metadata, ca_goodput, fifo_goodput,
    headline_name, outcome_str). Returns {name: FakeResult}."""
    out = {}
    for entry in scenarios:
        name, md, ca_g, fifo_g, hd_name, outcome_str = entry
        ca = _FakeKPI(goodput=ca_g, p99=200, sla=0)
        fifo = _FakeKPI(goodput=fifo_g, p99=200, sla=0)
        agg = {"fifo": fifo, "current_price_only": fifo, "greedy_energy": fifo,
               "sla_aware": fifo, "constraint_aware": ca}
        oc = OutcomeAnalysis(
            outcome=outcome_str, margin_pct=0.0,
            safety_evidence=(), loss_reasons=(), notes="",
        )
        out[name] = _FakeBenchmarkResult(
            _FakeBenchmarkReport(md, agg, hd_name, "explicit_override", oc),
        )
    return out


def test_cross_scenario_groups_by_workload_type():
    md_inf = _make_metadata(primary_workload_type="inference_critical")
    md_bat = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("inf1", md_inf, 100, 90, "sla_aware", "ALPHA_WIN"),
        ("inf2", md_inf, 110, 100, "sla_aware", "ALPHA_WIN"),
        ("bat1", md_bat, 200, 180, "current_price_only", "ALPHA_WIN"),
    ])
    cross = CrossScenarioReport.from_results(results)
    md = cross.to_markdown()
    # Per-workload section shows both workload types.
    assert "inference_critical" in md
    assert "batch_training" in md


def test_cross_scenario_separates_telemetry_failsafe_rows():
    md_inf = _make_metadata(primary_workload_type="inference_critical")
    md_telem = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    results = _build_fake_results([
        ("inf1", md_inf, 100, 90, "sla_aware", "ALPHA_WIN"),
        ("tel1", md_telem, 100, 100, "fifo", "KEEP_CORRECT"),
    ])
    cross = CrossScenarioReport.from_results(results)
    telem = cross.telemetry_failsafe_rows
    econ = cross.economic_rows
    assert len(telem) == 1
    assert len(econ) == 1
    assert telem[0].scenario_name == "tel1"


def test_cross_scenario_markdown_has_all_section_headers():
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    assert "## A. Overall policy" in md
    assert "## B. Per-workload-type" in md
    assert "## C. Per-scenario outcome" in md
    assert "## D. Baseline strength" in md


def test_cross_scenario_markdown_contains_goodput_unit_in_every_row():
    md_b = _make_metadata(primary_workload_type="batch_training",
                          goodput_unit="token_equivalent")
    md_i = _make_metadata(primary_workload_type="inference_critical",
                          goodput_unit="tokens")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("i1", md_i, 100, 90, "sla_aware", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    # Each per-scenario row in section C carries goodput_unit.
    assert "token_equivalent" in md
    assert "tokens" in md


@pytest.mark.parametrize("forbidden", [
    "production savings", "guaranteed savings", "production-proven",
    "hyperscaler-validated", "enterprise-ready autonomous",
])
def test_cross_scenario_markdown_does_not_contain_forbidden_phrases(forbidden):
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    assert forbidden.lower() not in md.lower()


def test_cross_scenario_energy_row_includes_both_baselines():
    md_e = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
    )
    results = _build_fake_results([
        ("energy1", md_e, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    cross = CrossScenarioReport.from_results(results)
    md = cross.to_markdown()
    # Section D should reference both current_price_only and greedy_energy.
    assert "current_price_only" in md
    assert "greedy_energy" in md


def test_per_workload_table_shows_mean_and_median():
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("b2", md_b, 200, 180, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    # Both 'Mean goodput/$' and 'Median goodput/$' columns must be present.
    assert "Mean goodput/$" in md
    assert "Median goodput/$" in md


# ---------------------------------------------------------------------------
# End-to-end via ConstraintBenchmarkRunner
# ---------------------------------------------------------------------------

def test_e2e_energy_scenario_reports_workload_relevant_baseline():
    r = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=8, seed=42,
    )
    assert r.report.headline_baseline_name in {
        "current_price_only", "greedy_energy", "sla_aware", "fifo",
    }
    assert r.report.outcome.outcome in OUTCOMES
    assert r.report.scenario_metadata.is_telemetry_failsafe is False


def test_e2e_thermal_scenario_is_not_a_loss():
    """thermal_hotspot is the canonical CA win — should be ALPHA_WIN or SAFETY_WIN."""
    r = ConstraintBenchmarkRunner().run_scenario(
        "thermal_hotspot_mixed_cluster", steps=24, seed=42,
    )
    assert r.report.outcome.outcome in {"ALPHA_WIN", "SAFETY_WIN", "TIE"}, (
        f"Thermal expected to be a CA win, got {r.report.outcome.outcome} "
        f"(margin {r.report.outcome.margin_pct:+.2f}%)"
    )


def test_e2e_telemetry_failsafe_yields_keep_correct():
    r = ConstraintBenchmarkRunner().run_scenario(
        "degraded_topology_telemetry", steps=8, seed=42,
    )
    assert r.report.outcome.outcome == "KEEP_CORRECT"
    assert r.report.scenario_metadata.is_telemetry_failsafe is True


def test_per_scenario_row_holds_metadata_and_outcome():
    """PerScenarioRow construction surface."""
    md = _make_metadata()
    row = PerScenarioRow(
        scenario_name="s",
        metadata=md,
        headline_baseline_name="fifo",
        headline_baseline_rationale="r",
        outcome=OutcomeAnalysis(
            outcome="TIE", margin_pct=0.0,
            safety_evidence=(), loss_reasons=(), notes="",
        ),
    )
    assert row.scenario_name == "s"
    assert row.metadata is md


# ---------------------------------------------------------------------------
# ISSUE 2 — select_headline FIFO-in-loop rationale fix
# ---------------------------------------------------------------------------

def test_select_headline_returns_disqualified_rationale_when_all_candidates_unsafe():
    """When every non-fifo candidate trips the safety filter, the rationale
    must be 'headline_baseline_disqualified_for_safety' — NOT
    'strongest_safe_relevant_baseline:fifo' (FIFO can't be its own safe
    candidate).
    """
    md = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
    )
    fifo = _FakeKPI(goodput=100, p99=200, sla=2)
    # current_price_only and greedy_energy both p99 ~5-6x fifo + sla > fifo
    cpo = _FakeKPI(goodput=300, p99=1100, sla=5)
    ge = _FakeKPI(goodput=250, p99=1200, sla=6)
    # sla_aware also tripping safety (more sla violations than fifo)
    sa = _FakeKPI(goodput=110, p99=1000, sla=5)
    name, rationale = select_headline_baseline(
        md,
        {"fifo": fifo, "current_price_only": cpo, "greedy_energy": ge,
         "sla_aware": sa},
    )
    assert name == "fifo"
    assert "disqualified_for_safety" in rationale


# ---------------------------------------------------------------------------
# ISSUE 3 — optimization_intent priority over workload_type
# ---------------------------------------------------------------------------

def test_select_headline_fragmentation_packing_wins_over_workload_type():
    """An inference_standard workload with fragmentation_packing intent must
    select a packing primitive — not sla_aware via the inference rule.
    """
    md = _make_metadata(
        primary_workload_type="inference_standard",
        optimization_intent="fragmentation_packing",
        relevant_baselines=("fifo", "first_fit", "best_fit",
                            "first_fit_decreasing"),
    )
    pol = {
        "fifo": _FakeKPI(goodput=100),
        "first_fit": _FakeKPI(goodput=200),
        "best_fit": _FakeKPI(goodput=150),
    }
    name, _ = select_headline_baseline(md, pol)
    assert name in ("first_fit", "best_fit")


def test_select_headline_energy_arbitrage_wins_over_workload_type():
    """An inference_standard workload with energy_arbitrage intent must
    select the strongest safe energy/sla baseline — not sla_aware just
    because the workload is interactive.
    """
    md = _make_metadata(
        primary_workload_type="inference_standard",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
    )
    fifo = _FakeKPI(goodput=100, p99=200, sla=0)
    cpo = _FakeKPI(goodput=150, p99=210, sla=0)   # safe
    sla = _FakeKPI(goodput=110, p99=210, sla=0)   # safe but weaker
    name, _ = select_headline_baseline(
        md,
        {"fifo": fifo, "current_price_only": cpo, "sla_aware": sla},
    )
    assert name == "current_price_only"


# ---------------------------------------------------------------------------
# ISSUE 4 — packing scenarios with no packing baseline → honest disclaimer
# ---------------------------------------------------------------------------

def test_select_headline_packing_when_no_baseline_data_returns_honest_disclaimer():
    """Packing scenario where the runner didn't compute first_fit/best_fit/...
    must surface the gap explicitly, not hide it behind a 'strongest=FIFO' claim.
    """
    md = _make_metadata(
        primary_workload_type="inference_standard",
        optimization_intent="fragmentation_packing",
        relevant_baselines=("fifo", "first_fit", "best_fit",
                            "first_fit_decreasing"),
    )
    pol = {"fifo": _FakeKPI(goodput=100)}   # no packing baselines computed
    name, rationale = select_headline_baseline(md, pol)
    assert name == "fifo"
    assert (
        "no_packing_baseline_computed" in rationale
        or "headline_baseline_disqualified" in rationale
    )


# ---------------------------------------------------------------------------
# ISSUE 5 — three new loss reason codes
# ---------------------------------------------------------------------------

def test_loss_reason_wrong_workload_classification_fires_for_interactive():
    """Engine treated an interactive workload as batch — the
    low-value-queue-relief gate fired despite the workload being interactive.
    """
    md = _make_metadata(
        primary_workload_type="inference_critical",
        optimization_intent="queue_relief",
        relevant_baselines=("fifo", "sla_aware"),
    )
    ca = _FakeKPI(
        goodput=80, p99=200, sla=0,
        blocked_scale_for_low_value_queue_relief=3,
        total_migrations=0,
    )
    hd = _FakeKPI(goodput=100, p99=200, sla=0, scale_up_applied=2)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        headline_name="fifo",
    )
    assert out.outcome == "LOSS"
    assert "wrong_workload_classification" in out.loss_reasons


def test_loss_reason_under_modeled_action_effect_fires_when_topology_damaged():
    """CA dropped topology score sharply vs the headline → under-modeled action."""
    md = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="topology_fit",
        relevant_baselines=("fifo", "sla_aware"),
    )
    ca = _FakeKPI(goodput=80, p99=200, sla=0, total_migrations=3)
    ca.mean_topology_score = 0.3   # bad
    hd = _FakeKPI(goodput=100, p99=200, sla=0)
    hd.mean_topology_score = 0.9   # healthy
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        headline_name="fifo",
    )
    assert out.outcome == "LOSS"
    assert "under_modeled_action_effect" in out.loss_reasons


def test_loss_reason_scenario_not_applicable_fires_for_inert_mixed_loss():
    """Mixed workload with no SLA-risk constraint active → scenario does not
    actually exercise any CA action; the loss is not a real signal.
    """
    md = _make_metadata(
        primary_workload_type="mixed",
        optimization_intent="safety_keep",
        relevant_baselines=("fifo", "sla_aware"),
    )
    # No relevant_actions, no blocked_low, no migrations, no SLA risk.
    ca = _FakeKPI(goodput=80, p99=100, sla=0, total_migrations=0)
    hd = _FakeKPI(goodput=100, p99=100, sla=0)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        headline_name="fifo",
    )
    assert out.outcome == "LOSS"
    assert "scenario_not_applicable" in out.loss_reasons


def test_loss_reason_codes_includes_three_new_codes():
    """ISSUE 5: the LOSS_REASON_CODES tuple must include all three new codes."""
    for code in ("wrong_workload_classification",
                 "under_modeled_action_effect", "scenario_not_applicable"):
        assert code in LOSS_REASON_CODES


# ---------------------------------------------------------------------------
# ISSUE 6 — alpha/safety counters on CrossScenarioReport
# ---------------------------------------------------------------------------

def test_alpha_safety_counters_compute_correctly_for_synthetic_report():
    """Synthetic 5-row report: 1 ALPHA_WIN, 1 SAFETY_WIN, 1 KEEP_CORRECT,
    1 LOSS, 1 TIE. Verify each named counter.
    """
    md_b = _make_metadata(primary_workload_type="batch_training")
    md_telem = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    results = _build_fake_results([
        ("s1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("s2", md_b, 100, 90, "current_price_only", "SAFETY_WIN"),
        ("s3", md_telem, 100, 100, "fifo", "KEEP_CORRECT"),
        ("s4", md_b, 80, 100, "current_price_only", "LOSS"),
        ("s5", md_b, 100, 100, "current_price_only", "TIE"),
    ])
    cross = CrossScenarioReport.from_results(results)
    assert cross.alpha_win_count == 1
    assert cross.safety_win_count == 1
    assert cross.correct_keep_count == 1
    assert cross.economic_loss_count == 1


def test_alpha_safety_counters_rendered_in_markdown():
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("s1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    assert "alpha_wins" in md
    assert "safety_wins" in md
    assert "correct_keeps" in md
    assert "economic_losses" in md
    assert "SLA_regressions" in md
    assert "catastrophic_baseline_avoidances" in md


def test_sla_regression_count_increments_when_ca_violations_exceed_fifo():
    """A row where CA has MORE SLA violations than FIFO should bump the
    SLA_regression counter, regardless of the ALPHA outcome label.
    """
    md = _make_metadata(primary_workload_type="batch_training")
    row = PerScenarioRow(
        scenario_name="r1", metadata=md,
        headline_baseline_name="fifo", headline_baseline_rationale="r",
        outcome=OutcomeAnalysis(
            outcome="ALPHA_WIN", margin_pct=5.0,
            safety_evidence=(), loss_reasons=(), notes="",
        ),
        sla_violations={"fifo": 1, "constraint_aware": 5},
        p99_latency_ms={"fifo": 200, "constraint_aware": 200},
    )
    cross = CrossScenarioReport(rows=[row])
    assert cross.sla_regression_count == 1


def test_catastrophic_baseline_avoidance_count_increments_when_aggressive_blows_up():
    """If greedy_energy's p99 is >= 2x CA's, the row counts toward
    catastrophic_baseline_avoidance.
    """
    md = _make_metadata(primary_workload_type="batch_training")
    row = PerScenarioRow(
        scenario_name="r1", metadata=md,
        headline_baseline_name="current_price_only", headline_baseline_rationale="r",
        outcome=OutcomeAnalysis(
            outcome="SAFETY_WIN", margin_pct=0.0,
            safety_evidence=("p99 better",), loss_reasons=(), notes="",
        ),
        p99_latency_ms={"constraint_aware": 200, "greedy_energy": 600,
                        "current_price_only": 250},
        sla_violations={},
    )
    cross = CrossScenarioReport(rows=[row])
    assert cross.catastrophic_baseline_avoidance_count == 1


# ---------------------------------------------------------------------------
# ISSUE 7 — missing_forecast_lookahead gated by headline being energy baseline
# ---------------------------------------------------------------------------

def test_missing_forecast_lookahead_not_emitted_when_headline_is_sla_aware():
    """An energy_arbitrage scenario where the headline is sla_aware (because
    the energy baselines tripped safety) must NOT report
    missing_forecast_lookahead — the comparison isn't a forecast test.
    """
    md = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
    )
    ca = _FakeKPI(goodput=80, total_net_savings=-10.0, total_migrations=0)
    hd = _FakeKPI(goodput=100)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca, "sla_aware": hd},
        headline_name="sla_aware",
    )
    assert out.outcome == "LOSS"
    assert "missing_forecast_lookahead" not in out.loss_reasons


# ---------------------------------------------------------------------------
# ISSUE 8 — workload-type vocabulary aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias", [
    "critical_interactive_inference",
    "standard_interactive_inference",
    "embeddings_offline",
    "training",
    "communication_heavy",
    "mixed_cluster",
    "telemetry_degraded",
])
def test_workload_type_alias_is_in_vocabulary(alias):
    assert alias in WORKLOAD_TYPES


def test_workload_type_aliases_mapping_is_consistent():
    """Every key + value in WORKLOAD_TYPE_ALIASES must be in WORKLOAD_TYPES."""
    for impl, spec in WORKLOAD_TYPE_ALIASES.items():
        assert impl in WORKLOAD_TYPES
        assert spec in WORKLOAD_TYPES


# ---------------------------------------------------------------------------
# ISSUE 9 — explicit-override beats telemetry-name heuristic
# ---------------------------------------------------------------------------

def test_explicit_workload_type_beats_telemetry_name_heuristic():
    """Scenario name contains "telemetry" but the raw dict explicitly sets
    primary_workload_type='batch_training' — the explicit override must win.
    """
    raw = {
        "primary_workload_type": "batch_training",
        "workloads": [{"priority_tier": "batch",
                       "workload_type": "batch_training"}],
    }
    m = classify_scenario("synthetic_telemetry_named", "energy_bound", raw)
    assert m.primary_workload_type == "batch_training"
    # Without explicit is_telemetry_failsafe in raw, the heuristic should
    # NOT promote this to True.
    assert m.is_telemetry_failsafe is False


# ---------------------------------------------------------------------------
# ISSUE 10 — parametrized SAFETY_WIN + content assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ca_goodput", [99.5, 100.0, 100.5])
def test_safety_win_across_tie_band_with_strong_p99(ca_goodput):
    """SAFETY_WIN should fire for ANY alpha in the tie band {-0.5%, 0, +0.5%}
    so long as p99 is materially better (here 100ms vs 300ms = 3x).
    """
    md = _make_metadata()
    ca = _FakeKPI(goodput=ca_goodput, p99=100)
    hd = _FakeKPI(goodput=100.0, p99=300)
    others = {"fifo": hd, "constraint_aware": ca,
              "sla_aware": _FakeKPI(goodput=99, p99=250)}
    out = analyze_outcome(md, ca, hd, others)
    assert out.outcome == "SAFETY_WIN", (
        f"goodput={ca_goodput} should be SAFETY_WIN, got {out.outcome}"
    )


def test_loss_reason_missing_candidate_action_fires_when_relevant_action_absent():
    """thermal scenario: ca did NO migrations and headline did → missing_candidate_action."""
    md = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="thermal_spread",
        relevant_baselines=("fifo", "sla_aware"),
    )
    ca = _FakeKPI(goodput=80, total_migrations=0)
    hd = _FakeKPI(goodput=100, total_migrations=5)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        headline_name="sla_aware",
    )
    assert out.outcome == "LOSS"
    assert "missing_candidate_action" in out.loss_reasons


def test_loss_reason_simulator_limitation_fires_for_packing_scenarios():
    """fragmentation_packing intent always tags simulator_limitation on LOSS
    (the simulator has no arbitrary-placement primitive in CA actions).
    """
    md = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="fragmentation_packing",
        relevant_baselines=("fifo", "first_fit", "best_fit"),
    )
    ca = _FakeKPI(goodput=80, total_migrations=0)
    hd = _FakeKPI(goodput=100)
    out = analyze_outcome(
        md, ca, hd, {"fifo": hd, "constraint_aware": ca},
        headline_name="first_fit",
    )
    assert out.outcome == "LOSS"
    assert "simulator_limitation" in out.loss_reasons


def test_each_row_metadata_goodput_unit_appears_in_section_c():
    """Per-row goodput_unit must be present in the corresponding section-C row,
    not just somewhere in the document.
    """
    md_b = _make_metadata(primary_workload_type="batch_training",
                          goodput_unit="token_equivalent")
    md_i = _make_metadata(primary_workload_type="inference_critical",
                          goodput_unit="tokens")
    results = _build_fake_results([
        ("b1_row", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("i1_row", md_i, 100, 90, "sla_aware", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    # Find the section-C row for each scenario and confirm the unit string
    # appears in that row's body (between the row's two newlines).
    for sname, unit in (("b1_row", "token_equivalent"), ("i1_row", "tokens")):
        match = [ln for ln in md.splitlines()
                 if ln.startswith(f"| {sname} ")]
        assert match, f"no section-C row for {sname}"
        assert unit in match[0], (
            f"row {sname} missing unit {unit!r} in: {match[0]}"
        )
