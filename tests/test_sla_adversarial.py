"""Adversarial audit tests for the SLA-aware optimization system.

These tests are intentionally adversarial: they try to make the optimizer cheat
(pick a cheap SLA-violating placement), claim false safety on missing telemetry,
or block actions it should allow. They also drive the REAL JobScheduler decision
path end-to-end (not just isolated helpers) and include regression tests for two
bugs found during the audit:

  Bug A: greedy fallback picked region_options[0] (possibly a forbidden cheap
         region) when every placement violated a hard SLA, instead of the
         least-change current region.
  Bug B: the MILP method ignored SLA entirely and emitted SLA-violating
         schedules; it now falls back to the SLA-aware greedy solver.
"""

from datetime import datetime, timedelta, timezone

import pytest

from aurelius.models import Job, OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler
from aurelius.sla import (
    ActionType,
    HeuristicPredictor,
    OptimizationAction,
    OptimizationAggressiveness,
    PriorityTier,
    RegionContext,
    SLAAwareActionSelector,
    SLALoader,
    SLAPolicy,
    SLAValidationError,
    WorkloadState,
    evaluate_action_against_sla,
)
from aurelius.sla.schema import HardSLA, SoftSLA, apply_tier_defaults

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _h(n):
    return timedelta(hours=n)


class _WL:
    def __init__(self, job_id="wl", workload_type="realtime_inference"):
        self.job_id = job_id
        self.workload_type = workload_type


def _std(**hard):
    return apply_tier_defaults(SLAPolicy(name="t", tier=PriorityTier.STANDARD, hard=HardSLA(**hard)))


# ===========================================================================
# Section 2: adversarial correctness of block / allow / rank
# ===========================================================================
class TestAdversarialBlocking:
    def setup_method(self):
        self.sel = SLAAwareActionSelector()

    def test_disabled_sla_is_identical_to_no_policy(self):
        # SLA disabled must preserve old behavior exactly (same chosen action).
        cur = WorkloadState(region="us-east")
        acts = [
            OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18),
            OptimizationAction(ActionType.MIGRATE, "us-west", expected_savings_pct=4),
        ]
        pol = _std(allowed_regions=["us-east"])  # would block ercot if enabled
        pol.enabled = False
        d_disabled = self.sel.select(_WL(), acts, cur, pol)
        d_none = self.sel.select(_WL(), acts, cur, None)
        assert d_disabled.chosen_action.target_region == d_none.chosen_action.target_region == "ercot"
        assert not d_disabled.was_corrected

    def test_high_savings_never_overrides_hard_violation(self):
        # A catastrophic SLA violation must block even with absurd savings.
        pol = _std(forbidden_regions=["ercot"])
        cur = WorkloadState(region="us-east")
        rc = {"ercot": RegionContext(region="ercot", spare_capacity_pct=90)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=99.0)]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.is_noop
        assert "forbidden" in " ".join(d.blocked_reasons)

    def test_soft_violation_does_not_block_safe_action(self):
        # Non-preferred but otherwise-safe region must still be allowed.
        pol = apply_tier_defaults(
            SLAPolicy(name="t", tier=PriorityTier.STANDARD,
                      hard=HardSLA(max_queue_wait_ms=10_000_000, max_p99_latency_ms=10_000_000),
                      soft=SoftSLA(preferred_regions=["us-east"], preferred_latency_headroom_pct=0.0))
        )
        cur = WorkloadState(region="us-east", p99_latency_ms=500, queue_wait_ms=10)
        rc = {"us-west": RegionContext(region="us-west", spare_capacity_pct=80,
                                       baseline_p99_latency_ms=500)}
        acts = [OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-west", expected_savings_pct=30)]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.target_region == "us-west"  # soft penalty didn't block
        # but the soft penalty was recorded
        sa = next(s for s in d.scored_actions if s.action is acts[0])
        assert sa.evaluation.soft_penalty_score > 0

    def test_noop_when_all_savers_violate(self):
        pol = _std(migration_allowed=False)
        cur = WorkloadState(region="us-east")
        rc = {r: RegionContext(region=r, spare_capacity_pct=80) for r in ("a", "b", "c")}
        acts = [OptimizationAction(ActionType.MIGRATE, r, expected_savings_pct=s)
                for r, s in (("a", 20), ("b", 15), ("c", 10))]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.is_noop
        assert d.chosen_savings_pct == 0.0

    def test_unknown_telemetry_does_not_claim_safety(self):
        # Constrained metric with no telemetry: reported unknown, allowed (no
        # fabricated pass), and flagged so we never CLAIM the SLA is met.
        pol = _std(max_p99_latency_ms=3000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHANGE_PLACEMENT, "x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
        )
        assert ev.allowed
        assert "p99_latency_ms" in ev.unknown_metrics

    def test_block_on_unknown_fails_closed(self):
        pol = _std(max_p99_latency_ms=3000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHANGE_PLACEMENT, "x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
            block_on_unknown=True,
        )
        assert not ev.allowed

    def test_critical_materially_more_conservative_than_batch(self):
        # Same modest-risk migration; critical blocks (no migration), batch allows.
        cur = WorkloadState(region="us-east", p99_latency_ms=300, availability_pct=99.99,
                            capacity_buffer_pct=50, queue_wait_ms=10)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=350,
                                     spare_capacity_pct=50)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18)]
        crit = apply_tier_defaults(SLAPolicy(name="c", tier=PriorityTier.CRITICAL))
        batch = apply_tier_defaults(SLAPolicy(name="b", tier=PriorityTier.BATCH))
        d_c = self.sel.select(_WL(), acts, cur, crit, region_contexts=rc)
        d_b = self.sel.select(_WL("j", "training"), acts, cur, batch, region_contexts=rc)
        assert d_c.chosen_action.is_noop
        assert d_b.chosen_action.target_region == "ercot"

    def test_migration_blocked_even_with_huge_savings(self):
        pol = _std(migration_allowed=False)
        cur = WorkloadState(region="us-east")
        rc = {"ercot": RegionContext(region="ercot", spare_capacity_pct=90)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=95)]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.is_noop

    def test_max_migrations_per_hour_enforced(self):
        pol = _std(migration_allowed=True, max_migrations_per_hour=1)
        cur = WorkloadState(region="us-east", migration_count_last_hour=1)
        rc = {"us-west": RegionContext(region="us-west", spare_capacity_pct=80)}
        acts = [OptimizationAction(ActionType.MIGRATE, "us-west", expected_savings_pct=20)]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.is_noop
        assert "max_migrations_per_hour" in " ".join(d.blocked_reasons)

    def test_no_migration_window_enforced(self):
        pol = _std(migration_allowed=True,
                   no_migration_windows=[["2026-01-01T00:00:00", "2026-12-31T23:59:59"]])
        cur = WorkloadState(region="us-east")
        rc = {"us-west": RegionContext(region="us-west", spare_capacity_pct=80)}
        acts = [OptimizationAction(ActionType.MIGRATE, "us-west", expected_savings_pct=20)]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc,
                            now=datetime(2026, 6, 1, tzinfo=UTC))
        assert d.chosen_action.is_noop

    def test_data_residency_overrides_cheaper_region(self):
        pol = _std(data_residency_region="eu-west", allowed_regions=["eu-west", "us-east"])
        cur = WorkloadState(region="eu-west")
        rc = {"us-east": RegionContext(region="us-east", spare_capacity_pct=90),
              "eu-west": RegionContext(region="eu-west", spare_capacity_pct=90)}
        acts = [
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-east", expected_savings_pct=40),
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "eu-west", expected_savings_pct=2),
        ]
        d = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert d.chosen_action.target_region == "eu-west"

    @pytest.mark.parametrize("metric,limit_kw,bad_val", [
        ("p95_latency_ms", {"max_p95_latency_ms": 200}, {"p95_latency_ms": 5000}),
        ("p99_latency_ms", {"max_p99_latency_ms": 500}, {"p99_latency_ms": 9000}),
        ("queue_wait_ms", {"max_queue_wait_ms": 100}, {"queue_wait_ms": 9000}),
        ("capacity_buffer_pct", {"required_capacity_buffer_pct": 30}, {"capacity_buffer_pct": 1}),
    ])
    def test_each_constraint_independently_blocks(self, metric, limit_kw, bad_val):
        # Use precomputed predicted state so only the one metric is bad.
        pol = _std(**limit_kw)
        cur = WorkloadState(region="us-east")
        act = OptimizationAction(ActionType.CHANGE_PLACEMENT, "us-east", expected_savings_pct=10)
        pred = WorkloadState(region="us-east", **bad_val)
        d = self.sel.select(_WL(), [act], cur, pol,
                            predicted_states={id(act): pred})
        assert d.chosen_action.is_noop, f"{metric} should have blocked"

    def test_preferred_region_only_wins_within_tradeoff(self):
        rc = {"us-east": RegionContext(region="us-east", spare_capacity_pct=80),
              "us-west": RegionContext(region="us-west", spare_capacity_pct=80)}
        cur = WorkloadState(region="us-east", p99_latency_ms=500, queue_wait_ms=10)
        acts = [
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-west", expected_savings_pct=10),
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-east", expected_savings_pct=7),
        ]

        def pol(tradeoff):
            return apply_tier_defaults(SLAPolicy(
                name="p", tier=PriorityTier.STANDARD,
                hard=HardSLA(max_queue_wait_ms=10_000_000, max_p99_latency_ms=10_000_000),
                soft=SoftSLA(preferred_regions=["us-east"],
                             max_acceptable_savings_tradeoff_pct=tradeoff,
                             preferred_latency_headroom_pct=0.0),
            ))

        within = self.sel.select(_WL(), acts, cur, pol(5), region_contexts=rc)
        exceeds = self.sel.select(_WL(), acts, cur, pol(2), region_contexts=rc)
        assert within.chosen_action.target_region == "us-east"   # gap 3 <= 5
        assert exceeds.chosen_action.target_region == "us-west"  # gap 3 > 2


# ===========================================================================
# Section 3: end-to-end JobScheduler integration scenarios
# ===========================================================================
def _price(regions, hours=48, cheap=None):
    cheap = cheap or {"ercot": 10.0, "us-west": 40.0, "us-east": 80.0, "eu-west": 60.0}
    return {r: {T0 + _h(h): cheap.get(r, 50.0) for h in range(hours)} for r in regions}


def _carbon(regions, hours=48):
    return {r: {T0 + _h(h): 300.0 for h in range(hours)} for r in regions}


def _job(jid="inf", wtype="realtime_inference", regions=("us-east", "ercot"), runtime=4):
    return Job(job_id=jid, submit_time=T0, runtime_hours=runtime, deadline=T0 + _h(24),
               power_kw=100.0, earliest_start=T0, region_options=list(regions),
               priority=5, workload_type=wtype)


class TestEndToEndOptimizer:
    def test_scenario_a_energy_only_picks_cheapest(self):
        regions = ["us-east", "ercot"]
        sched = JobScheduler(OptimizationConfig(default_region="us-east"))
        res = sched.solve([_job(regions=regions)], _price(regions), _carbon(regions),
                          method="greedy")
        assert res.schedule[0].region == "ercot"  # cheapest, no SLA

    def test_scenario_b_sla_rejects_cheapest(self):
        regions = ["us-east", "ercot"]
        reg = SLALoader.load_text("""
        policies:
          - name: inf
            tier: latency_sensitive
            applies_to_workloads: [inf]
            hard:
              allowed_regions: [us-east]
              max_p99_latency_ms: 800
        """, fmt="yaml")
        rc = {r: RegionContext(region=r, baseline_p99_latency_ms=400, spare_capacity_pct=60)
              for r in regions}
        states = {"inf": WorkloadState(region="us-east", p99_latency_ms=400,
                                       availability_pct=99.99, capacity_buffer_pct=50,
                                       queue_wait_ms=10)}
        sched = JobScheduler(OptimizationConfig(default_region="us-east"),
                             sla_registry=reg, region_contexts=rc, current_states=states)
        res = sched.solve([_job(regions=regions)], _price(regions), _carbon(regions),
                          method="greedy")
        assert res.schedule[0].region == "us-east"  # cheapest ercot rejected

        # And the standalone evaluator confirms WHY, with risk + explanation.
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18.4),
            _WL("inf"), states["inf"],
            HeuristicPredictor().predict(
                OptimizationAction(ActionType.MIGRATE, "ercot"), states["inf"], rc["ercot"]),
            reg.resolve(workload_id="inf"),
        )
        assert not ev.allowed
        assert ev.risk_score > 0
        assert "ercot" in ev.explanation

    def test_scenario_c_batch_allows_what_critical_blocks(self):
        regions = ["us-east", "ercot"]
        rc = {r: RegionContext(region=r, baseline_p99_latency_ms=400, spare_capacity_pct=60)
              for r in regions}

        crit = SLALoader.load_text("""
        policies:
          - name: c
            tier: critical
            applies_to_workloads: [inf]
        """, fmt="yaml")
        batch = SLALoader.load_text("""
        policies:
          - name: b
            tier: batch
            applies_to_workload_types: [training]
        """, fmt="yaml")

        states_c = {"inf": WorkloadState(region="us-east", p99_latency_ms=300,
                                         availability_pct=99.99, capacity_buffer_pct=50, queue_wait_ms=5)}
        sched_c = JobScheduler(OptimizationConfig(default_region="us-east"),
                               sla_registry=crit, region_contexts=rc, current_states=states_c)
        res_c = sched_c.solve([_job("inf", "realtime_inference", regions)],
                              _price(regions), _carbon(regions), method="greedy")

        states_b = {"trn": WorkloadState(region="us-east", p99_latency_ms=2000,
                                          availability_pct=99.5, capacity_buffer_pct=20, queue_wait_ms=1000)}
        sched_b = JobScheduler(OptimizationConfig(default_region="us-east"),
                               sla_registry=batch, region_contexts=rc, current_states=states_b)
        res_b = sched_b.solve([_job("trn", "training", regions)],
                              _price(regions), _carbon(regions), method="greedy")

        assert res_c.schedule[0].region == "us-east"  # critical blocks migration
        assert res_b.schedule[0].region == "ercot"    # batch allows cheap move

    def test_no_registry_matches_baseline_exactly(self):
        # SLA-disabled scheduler must produce the identical schedule object shape.
        regions = ["us-east", "ercot"]
        job = _job(regions=regions)
        a = JobScheduler(OptimizationConfig(default_region="us-east")).solve(
            [job], _price(regions), _carbon(regions), method="greedy")
        reg = SLALoader.load_text(
            "enabled: false\npolicies: [{name: x, applies_to_workloads: [inf], hard: {allowed_regions: [us-east]}}]",
            fmt="yaml")
        b = JobScheduler(OptimizationConfig(default_region="us-east"), sla_registry=reg).solve(
            [job], _price(regions), _carbon(regions), method="greedy")
        assert a.schedule[0].region == b.schedule[0].region == "ercot"


# ===========================================================================
# Regression tests for bugs found during the audit
# ===========================================================================
class TestAuditRegressions:
    def test_bug_a_all_violate_fallback_prefers_current_region(self):
        # region_options ordered with the forbidden/cheap region FIRST. Before
        # the fix, the fallback returned region_options[0] (ercot). It must now
        # keep the current region (us-east).
        regions = ["ercot", "us-east"]
        reg = SLALoader.load_text("""
        policies:
          - name: inf
            tier: latency_sensitive
            applies_to_workloads: [inf]
            hard: {allowed_regions: [eu-west]}
        """, fmt="yaml")
        sched = JobScheduler(
            OptimizationConfig(default_region="us-east"), sla_registry=reg,
            current_states={"inf": WorkloadState(region="us-east")},
            region_contexts={r: RegionContext(region=r, spare_capacity_pct=60) for r in regions},
        )
        res = sched.solve([_job(regions=regions)], _price(regions), _carbon(regions),
                          method="greedy")
        assert res.schedule[0].region == "us-east"

    def test_bug_b_milp_does_not_emit_sla_violation(self):
        regions = ["ercot", "us-east"]
        reg = SLALoader.load_text("""
        policies:
          - name: inf
            tier: latency_sensitive
            applies_to_workloads: [inf]
            hard: {allowed_regions: [us-east]}
        """, fmt="yaml")
        sched = JobScheduler(
            OptimizationConfig(default_region="us-east"), sla_registry=reg,
            current_states={"inf": WorkloadState(region="us-east", p99_latency_ms=100,
                                                 availability_pct=99.99, capacity_buffer_pct=50,
                                                 queue_wait_ms=5)},
            region_contexts={r: RegionContext(region=r, spare_capacity_pct=60,
                                              baseline_p99_latency_ms=100) for r in regions},
        )
        res = sched.solve([_job(regions=regions)], _price(regions), _carbon(regions),
                          method="milp")
        # MILP must not route to forbidden ercot.
        assert res.schedule[0].region == "us-east"

    def test_milp_without_sla_still_uses_milp(self):
        # No SLA => MILP path is unaffected (still optimizes to cheapest).
        regions = ["us-east", "ercot"]
        sched = JobScheduler(OptimizationConfig(default_region="us-east"))
        res = sched.solve([_job(regions=regions)], _price(regions), _carbon(regions),
                          method="milp")
        assert res.schedule[0].region == "ercot"


# ===========================================================================
# Section 5: edge cases / malformed configs
# ===========================================================================
class TestEdgeCases:
    def test_malformed_yaml_raises(self):
        with pytest.raises(Exception):
            SLALoader.load_text("policies: [ {name: x, : }", fmt="yaml")

    def test_malformed_json_raises(self):
        with pytest.raises(Exception):
            SLALoader.load_text('{"policies": [', fmt="json")

    def test_missing_workload_policy_returns_none(self):
        reg = SLALoader.load_text(
            "policies: [{name: x, applies_to_workloads: [other]}]", fmt="yaml")
        assert reg.resolve(workload_id="nope") is None

    def test_invalid_tier_raises(self):
        with pytest.raises(SLAValidationError):
            SLALoader.load_text("policies: [{name: x, tier: superduper}]", fmt="yaml")

    def test_same_region_allowed_and_forbidden_raises(self):
        with pytest.raises(SLAValidationError):
            SLALoader.load_text(
                "policies: [{name: x, hard: {allowed_regions: [a], forbidden_regions: [a]}}]",
                fmt="yaml")

    def test_negative_latency_raises(self):
        with pytest.raises(SLAValidationError):
            SLALoader.load_text(
                "policies: [{name: x, hard: {max_p99_latency_ms: -5}}]", fmt="yaml")

    def test_empty_allowed_regions_blocks_everything(self):
        # allowed_regions: [] is an explicit empty allow-list -> no region allowed.
        pol = _std(allowed_regions=[])
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, "us-east"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="us-east"), pol,
        )
        assert not ev.allowed

    def test_impossible_sla_p99_below_current(self):
        # Hard p99 limit below current p99: keeping put would itself be a
        # violation, but since the evaluator scores the PREDICTED state of each
        # candidate, the no-op (predicted == current) is correctly blocked too,
        # so the selector must surface that nothing is safe.
        pol = _std(max_p99_latency_ms=100)
        cur = WorkloadState(region="us-east", p99_latency_ms=2000, queue_wait_ms=5,
                            availability_pct=99.99, capacity_buffer_pct=50)
        sel = SLAAwareActionSelector()
        # Only a no-op candidate; predicted == current p99=2000 > 100.
        d = sel.select(_WL(), [OptimizationAction(ActionType.KEEP, "us-east")], cur, pol)
        sa = d.scored_actions[0]
        assert not sa.sla_safe  # even keeping put violates the impossible SLA

    def test_zero_savings_lower_risk_beats_risky_saver_via_score(self):
        # An action with high savings but high risk can lose to a 0-savings
        # safe action once risk penalties apply (conservative tier amplifies).
        pol = apply_tier_defaults(SLAPolicy(
            name="p", tier=PriorityTier.STANDARD,
            hard=HardSLA(max_p99_latency_ms=10_000_000, max_queue_wait_ms=10_000_000),
            soft=SoftSLA(optimization_aggressiveness=OptimizationAggressiveness.CONSERVATIVE,
                         preferred_regions=["us-east"], preferred_latency_headroom_pct=0.0),
        ))
        cur = WorkloadState(region="us-east", p99_latency_ms=1000, queue_wait_ms=10)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=1000,
                                     spare_capacity_pct=5)}  # low capacity -> risk
        sel = SLAAwareActionSelector()
        # Tiny savings migration vs staying put.
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=1.0)]
        d = sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        # Migration penalty + soft + capacity risk should outweigh 1% savings.
        assert d.chosen_action.is_noop

    def test_missing_predicted_metrics_uses_heuristic_predictor(self):
        # When no predicted_states are supplied, the selector must invoke the
        # predictor rather than assuming the current state is unchanged.
        pol = _std(max_queue_wait_ms=1000)
        cur = WorkloadState(region="us-east", queue_wait_ms=500)
        rc = {"ercot": RegionContext(region="ercot", spare_capacity_pct=60)}
        sel = SLAAwareActionSelector()
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=20)]
        d = sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        # Cold-start queue inflation pushes predicted queue over 1000 -> blocked.
        assert d.chosen_action.is_noop
