"""Tests for the SLA ingestion + correction engine.

Covers schema/tier defaults, the loader/validator, the evaluator for each SLA
type, and the SLA-aware selector (with vs without SLA, aggressiveness, tiers,
preferred-region tradeoff, thermal avoidance).
"""

from datetime import datetime

import pytest

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
    policy_from_dict,
)
from aurelius.sla.schema import HardSLA, SoftSLA, apply_tier_defaults


class _WL:
    """Minimal workload stub."""

    def __init__(self, job_id="wl", workload_type="realtime_inference"):
        self.job_id = job_id
        self.workload_type = workload_type


# ---------------------------------------------------------------------------
# Schema + tier defaults
# ---------------------------------------------------------------------------
class TestSchemaAndTiers:
    def test_critical_is_safest(self):
        p = apply_tier_defaults(SLAPolicy(name="c", tier=PriorityTier.CRITICAL))
        assert p.hard.migration_allowed is False
        assert p.hard.max_migrations_per_hour == 0
        assert p.aggressiveness == OptimizationAggressiveness.CONSERVATIVE
        assert p.hard.required_capacity_buffer_pct >= 20

    def test_batch_is_most_cost_optimized(self):
        p = apply_tier_defaults(SLAPolicy(name="b", tier=PriorityTier.BATCH))
        assert p.hard.migration_allowed is True
        assert p.hard.max_migrations_per_hour >= 10
        assert p.aggressiveness == OptimizationAggressiveness.AGGRESSIVE
        assert p.soft.max_acceptable_savings_tradeoff_pct >= 20

    def test_explicit_field_overrides_tier_default(self):
        # Critical tier default migration_allowed=False, but explicit True wins.
        p = apply_tier_defaults(
            SLAPolicy(name="c", tier=PriorityTier.CRITICAL, hard=HardSLA(migration_allowed=True))
        )
        assert p.hard.migration_allowed is True
        # Unspecified fields still come from the tier.
        assert p.hard.max_p99_latency_ms == 500.0


# ---------------------------------------------------------------------------
# Loader / validation
# ---------------------------------------------------------------------------
class TestLoader:
    def test_load_yaml(self):
        text = """
        policies:
          - name: p1
            tier: standard
            applies_to_workloads: [svc-a]
            hard:
              allowed_regions: [us-east]
              max_p99_latency_ms: 2000
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        assert len(reg) == 1
        pol = reg.resolve(workload_id="svc-a")
        assert pol.hard.allowed_regions == ["us-east"]

    def test_load_json(self):
        text = '{"policies":[{"name":"p","tier":"batch","applies_to_workload_types":["training"]}]}'
        reg = SLALoader.load_text(text, fmt="json")
        assert reg.resolve(workload_type="training").tier == PriorityTier.BATCH

    def test_invalid_tier_raises(self):
        with pytest.raises(SLAValidationError) as e:
            policy_from_dict({"name": "x", "tier": "ultra"})
        assert any("tier" in m for m in e.value.errors)

    def test_unknown_field_raises(self):
        with pytest.raises(SLAValidationError) as e:
            policy_from_dict({"name": "x", "hard": {"max_p99_latency_ms": 10, "bogus": 1}})
        assert any("bogus" in m for m in e.value.errors)

    def test_contradictory_regions_raise(self):
        with pytest.raises(SLAValidationError):
            policy_from_dict(
                {"name": "x", "hard": {"allowed_regions": ["a"], "forbidden_regions": ["a"]}}
            )

    def test_p99_below_p95_raises(self):
        with pytest.raises(SLAValidationError):
            policy_from_dict(
                {"name": "x", "hard": {"max_p95_latency_ms": 500, "max_p99_latency_ms": 100}}
            )

    def test_out_of_range_pct_raises(self):
        with pytest.raises(SLAValidationError):
            policy_from_dict({"name": "x", "hard": {"min_availability_pct": 150}})

    def test_resolution_precedence_workload_over_type(self):
        text = """
        policies:
          - name: by_type
            applies_to_workload_types: [training]
            tier: batch
          - name: by_id
            applies_to_workloads: [job-7]
            tier: critical
        """
        reg = SLALoader.load_text(text, fmt="yaml")
        # job-7 matches both id and type; id wins.
        pol = reg.resolve(workload_id="job-7", workload_type="training")
        assert pol.name == "by_id"


# ---------------------------------------------------------------------------
# Evaluator — one test per SLA type
# ---------------------------------------------------------------------------
class TestEvaluator:
    def _policy(self, **hard):
        return apply_tier_defaults(SLAPolicy(name="t", tier=PriorityTier.STANDARD, hard=HardSLA(**hard)))

    def test_no_policy_allows(self):
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), WorkloadState(), WorkloadState(), None,
        )
        assert ev.allowed

    def test_disabled_policy_allows(self):
        pol = self._policy(allowed_regions=["us-east"])
        pol.enabled = False
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="ercot"),
            _WL(), WorkloadState(region="us-east"), WorkloadState(region="ercot"), pol,
        )
        assert ev.allowed

    def test_forbidden_region_blocks(self):
        pol = self._policy(forbidden_regions=["ercot"])
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, target_region="ercot"),
            _WL(), WorkloadState(region="us-east"), WorkloadState(region="ercot"), pol,
        )
        assert not ev.allowed
        assert any("forbidden" in v for v in ev.violated_hard_constraints)

    def test_allowed_regions_blocks_others(self):
        pol = self._policy(allowed_regions=["us-east", "us-west"])
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="ercot"),
            _WL(), WorkloadState(region="us-east"), WorkloadState(region="ercot"), pol,
        )
        assert not ev.allowed

    def test_data_residency_blocks(self):
        pol = self._policy(data_residency_region="eu-west", allowed_regions=["eu-west"])
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="us-east"),
            _WL(), WorkloadState(region="eu-west"), WorkloadState(region="us-east"), pol,
        )
        assert not ev.allowed
        assert any("residency" in v for v in ev.violated_hard_constraints)

    def test_p99_latency_blocks(self):
        pol = self._policy(max_p99_latency_ms=3000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHANGE_PLACEMENT, target_region="x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x", p99_latency_ms=6200), pol,
        )
        assert not ev.allowed
        assert any("p99" in v for v in ev.violated_hard_constraints)

    def test_queue_wait_blocks(self):
        pol = self._policy(max_queue_wait_ms=1000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CONSOLIDATE, target_region="a"),
            _WL(), WorkloadState(region="a"),
            WorkloadState(region="a", queue_wait_ms=5000), pol,
        )
        assert not ev.allowed
        assert any("queue" in v for v in ev.violated_hard_constraints)

    def test_availability_blocks(self):
        pol = self._policy(min_availability_pct=99.9)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), WorkloadState(region="a"),
            WorkloadState(region="x", availability_pct=99.0), pol,
        )
        assert not ev.allowed

    def test_capacity_buffer_blocks(self):
        pol = self._policy(required_capacity_buffer_pct=20)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), WorkloadState(region="a"),
            WorkloadState(region="x", capacity_buffer_pct=5), pol,
        )
        assert not ev.allowed
        assert any("capacity" in v for v in ev.violated_hard_constraints)

    def test_migration_not_allowed_blocks(self):
        pol = self._policy(migration_allowed=False)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
        )
        assert not ev.allowed
        assert any("migration_allowed" in v for v in ev.violated_hard_constraints)

    def test_max_migrations_per_hour_blocks(self):
        pol = self._policy(migration_allowed=True, max_migrations_per_hour=1)
        cur = WorkloadState(region="a", migration_count_last_hour=1)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), cur, WorkloadState(region="x"), pol,
        )
        assert not ev.allowed
        assert any("max_migrations_per_hour" in v for v in ev.violated_hard_constraints)

    def test_no_migration_window_blocks(self):
        pol = self._policy(
            migration_allowed=True,
            no_migration_windows=[["2026-01-01T00:00:00", "2026-12-31T23:59:59"]],
        )
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
            now=datetime(2026, 5, 24, 12, 0),
        )
        assert not ev.allowed
        assert any("no_migration_window" in v for v in ev.violated_hard_constraints)

    def test_soft_violation_does_not_block_but_penalizes(self):
        pol = apply_tier_defaults(
            SLAPolicy(
                name="t", tier=PriorityTier.STANDARD,
                soft=SoftSLA(preferred_regions=["us-east"]),
            )
        )
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, target_region="ercot"),
            _WL(), WorkloadState(region="us-east"), WorkloadState(region="ercot"), pol,
        )
        assert ev.allowed
        assert ev.soft_penalty_score > 0
        assert ev.soft_violations

    def test_corrected_action_on_block(self):
        pol = self._policy(forbidden_regions=["ercot"])
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.MIGRATE, target_region="ercot"),
            _WL(), WorkloadState(region="us-east"), WorkloadState(region="ercot"), pol,
        )
        assert ev.corrected_action is not None
        assert ev.corrected_action.action_type == ActionType.KEEP

    def test_unknown_metric_not_blocking_by_default(self):
        pol = self._policy(max_p99_latency_ms=3000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHANGE_PLACEMENT, target_region="x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
        )
        assert ev.allowed
        assert "p99_latency_ms" in ev.unknown_metrics

    def test_unknown_metric_blocks_when_fail_closed(self):
        pol = self._policy(max_p99_latency_ms=3000)
        ev = evaluate_action_against_sla(
            OptimizationAction(ActionType.CHANGE_PLACEMENT, target_region="x"),
            _WL(), WorkloadState(region="a"), WorkloadState(region="x"), pol,
            block_on_unknown=True,
        )
        assert not ev.allowed


# ---------------------------------------------------------------------------
# Selector — with vs without SLA, aggressiveness, tiers, tradeoff, thermal
# ---------------------------------------------------------------------------
class TestSelector:
    def setup_method(self):
        self.sel = SLAAwareActionSelector()

    def _std_policy(self, **hard):
        return apply_tier_defaults(
            SLAPolicy(name="p", tier=PriorityTier.STANDARD, hard=HardSLA(**hard))
        )

    def test_unconstrained_picks_cheapest(self):
        # No SLA: highest savings wins.
        cur = WorkloadState(region="us-east")
        acts = [
            OptimizationAction(ActionType.MIGRATE, target_region="ercot", expected_savings_pct=18.4),
            OptimizationAction(ActionType.MIGRATE, target_region="us-west", expected_savings_pct=5.0),
        ]
        dec = self.sel.select(_WL(), acts, cur, None)
        assert dec.chosen_action.target_region == "ercot"
        assert not dec.was_corrected

    def test_energy_vs_latency_rejects_cheapest(self):
        # Cheapest region violates p99 SLA -> rejected.
        pol = self._std_policy(max_p99_latency_ms=3000)
        cur = WorkloadState(region="us-east", p99_latency_ms=1500)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=6000, energy_price=10)}
        acts = [OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "ercot", expected_savings_pct=18.4)]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.is_noop
        assert dec.was_corrected
        assert dec.savings_sacrificed_pct == pytest.approx(18.4)

    def test_allowed_regions_not_selected(self):
        pol = self._std_policy(allowed_regions=["us-east", "us-west"])
        cur = WorkloadState(region="us-east")
        acts = [
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "ercot", expected_savings_pct=20),
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-west", expected_savings_pct=4),
        ]
        rc = {
            "ercot": RegionContext(region="ercot", spare_capacity_pct=80),
            "us-west": RegionContext(region="us-west", spare_capacity_pct=80),
        }
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.target_region != "ercot"

    def test_migration_forbidden_chooses_noop(self):
        pol = self._std_policy(migration_allowed=False)
        cur = WorkloadState(region="us-east")
        acts = [OptimizationAction(ActionType.MIGRATE, "us-west", expected_savings_pct=12)]
        rc = {"us-west": RegionContext(region="us-west", spare_capacity_pct=80)}
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.is_noop

    def test_consolidation_rejected_on_queue(self):
        pol = self._std_policy(max_queue_wait_ms=1000)
        cur = WorkloadState(region="us-east", queue_wait_ms=800, gpu_utilization_pct=40)
        acts = [OptimizationAction(ActionType.CONSOLIDATE, "us-east", expected_savings_pct=15)]
        dec = self.sel.select(_WL(), acts, cur, pol)
        assert dec.chosen_action.is_noop

    def test_capacity_buffer_rejects_move(self):
        pol = self._std_policy(required_capacity_buffer_pct=20)
        cur = WorkloadState(region="us-east", capacity_buffer_pct=40)
        rc = {"ercot": RegionContext(region="ercot", spare_capacity_pct=5)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18)]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.is_noop

    def _tradeoff_policy(self, tradeoff):
        # Generous HARD limits so both migrations are SLA-safe; this isolates
        # the SOFT preferred-region tradeoff logic.
        return apply_tier_defaults(
            SLAPolicy(
                name="p", tier=PriorityTier.STANDARD,
                hard=HardSLA(max_queue_wait_ms=1_000_000, max_p99_latency_ms=1_000_000),
                soft=SoftSLA(
                    preferred_regions=["us-east"],
                    max_acceptable_savings_tradeoff_pct=tradeoff,
                    preferred_latency_headroom_pct=0.0,
                ),
            )
        )

    def test_preferred_region_tradeoff_within_threshold(self):
        # Preferred region slightly more expensive, within tradeoff -> chosen.
        pol = self._tradeoff_policy(5)
        cur = WorkloadState(region="us-east", p99_latency_ms=1000, queue_wait_ms=100)
        rc = {
            "us-east": RegionContext(region="us-east", spare_capacity_pct=80),
            "us-west": RegionContext(region="us-west", spare_capacity_pct=80),
        }
        acts = [
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-west", expected_savings_pct=10),
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-east", expected_savings_pct=7),
        ]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.target_region == "us-east"

    def test_preferred_region_tradeoff_exceeds_threshold(self):
        pol = self._tradeoff_policy(2)
        cur = WorkloadState(region="us-east", p99_latency_ms=1000, queue_wait_ms=100)
        rc = {
            "us-east": RegionContext(region="us-east", spare_capacity_pct=80),
            "us-west": RegionContext(region="us-west", spare_capacity_pct=80),
        }
        acts = [
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-west", expected_savings_pct=10),
            OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "us-east", expected_savings_pct=7),
        ]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        # Gap 3% > 2% threshold -> cheaper non-preferred wins.
        assert dec.chosen_action.target_region == "us-west"

    def test_aggressive_allows_more_than_conservative(self):
        # A migration with modest risk but no hard violation: aggressive keeps
        # the move, conservative may reject it due to amplified risk penalty.
        cur = WorkloadState(region="us-east", p99_latency_ms=1000, queue_wait_ms=200,
                            availability_pct=99.99, capacity_buffer_pct=50)
        rc = {"us-west": RegionContext(region="us-west", baseline_p99_latency_ms=1500,
                                       spare_capacity_pct=60, baseline_queue_wait_ms=300)}
        acts = [OptimizationAction(ActionType.MIGRATE, "us-west", expected_savings_pct=3.0)]

        cons = apply_tier_defaults(SLAPolicy(
            name="c", tier=PriorityTier.STANDARD,
            soft=SoftSLA(optimization_aggressiveness=OptimizationAggressiveness.CONSERVATIVE),
            hard=HardSLA(max_p99_latency_ms=10000, max_queue_wait_ms=60000),
        ))
        aggr = apply_tier_defaults(SLAPolicy(
            name="a", tier=PriorityTier.STANDARD,
            soft=SoftSLA(optimization_aggressiveness=OptimizationAggressiveness.AGGRESSIVE),
            hard=HardSLA(max_p99_latency_ms=10000, max_queue_wait_ms=60000),
        ))
        dec_c = self.sel.select(_WL(), acts, cur, cons, region_contexts=rc)
        dec_a = self.sel.select(_WL(), acts, cur, aggr, region_contexts=rc)
        # Aggressive should be at least as willing to move as conservative.
        moved_c = not dec_c.chosen_action.is_noop
        moved_a = not dec_a.chosen_action.is_noop
        assert moved_a or (not moved_a and not moved_c)
        # The aggressive risk penalty must be <= conservative for the same move.
        ev_c = next(s for s in dec_c.scored_actions if s.action is acts[0])
        ev_a = next(s for s in dec_a.scored_actions if s.action is acts[0])
        assert ev_a.evaluation.risk_score <= ev_c.evaluation.risk_score

    def test_critical_tier_avoids_risky_migration(self):
        pol = apply_tier_defaults(SLAPolicy(name="crit", tier=PriorityTier.CRITICAL))
        cur = WorkloadState(region="us-east", p99_latency_ms=300, availability_pct=99.99,
                            capacity_buffer_pct=50)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=400,
                                     spare_capacity_pct=50, energy_price=10)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18)]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        # Critical => migration_allowed False => keep.
        assert dec.chosen_action.is_noop

    def test_batch_tier_allows_aggressive_cost_optimization(self):
        pol = apply_tier_defaults(
            SLAPolicy(name="batch", tier=PriorityTier.BATCH,
                      applies_to_workload_types=["training"])
        )
        cur = WorkloadState(region="us-east", p99_latency_ms=2000, availability_pct=99.5,
                            capacity_buffer_pct=10, queue_wait_ms=1000)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=3000,
                                     spare_capacity_pct=40, baseline_queue_wait_ms=2000,
                                     energy_price=10)}
        acts = [OptimizationAction(ActionType.MIGRATE, "ercot", expected_savings_pct=18.4)]
        dec = self.sel.select(_WL("j", "training"), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.target_region == "ercot"
        assert not dec.was_corrected

    def test_thermal_stressed_region_avoided(self):
        # Cheaper region is throttling/thermally stressed -> inflated p99 -> blocked.
        pol = self._std_policy(max_p99_latency_ms=3000)
        cur = WorkloadState(region="us-east", p99_latency_ms=1500)
        rc = {"ercot": RegionContext(region="ercot", baseline_p99_latency_ms=2500,
                                     throttling=True, thermally_stressed=True)}
        acts = [OptimizationAction(ActionType.CHOOSE_CHEAPER_REGION, "ercot", expected_savings_pct=18)]
        dec = self.sel.select(_WL(), acts, cur, pol, region_contexts=rc)
        assert dec.chosen_action.is_noop


class TestPredictor:
    def test_migration_increases_p99_and_queue(self):
        pred = HeuristicPredictor()
        cur = WorkloadState(region="a", p99_latency_ms=1000, queue_wait_ms=100)
        out = pred.predict(
            OptimizationAction(ActionType.MIGRATE, target_region="b"), cur,
            RegionContext(region="b", spare_capacity_pct=50),
        )
        assert out.queue_wait_ms > cur.queue_wait_ms
        assert out.migration_count_last_hour == 1

    def test_consolidation_raises_utilization(self):
        pred = HeuristicPredictor()
        cur = WorkloadState(region="a", gpu_utilization_pct=40, queue_wait_ms=100, p99_latency_ms=1000)
        out = pred.predict(OptimizationAction(ActionType.CONSOLIDATE, target_region="a"), cur)
        assert out.gpu_utilization_pct > cur.gpu_utilization_pct
        assert out.queue_wait_ms > cur.queue_wait_ms
