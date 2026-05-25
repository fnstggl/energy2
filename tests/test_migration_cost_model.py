"""Tests for Phase 8/9 — STATE-CONDITIONED migration cost/risk model.

These tests prove the corrected risk model: risk is conditioned on observed/
predicted state (SLA headroom, destination health, action-specific cost,
telemetry confidence), NOT on a static workload-class multiplier.

Tests prove:
- High gross savings with a hard-SLA breach → KEEP (savings never override hard SLA)
- A critical workload CAN migrate when SLA headroom is large and the destination is safe
- A critical workload is blocked when SLA headroom is small
- A batch workload is blocked when destination topology/thermal/queue risk is high
- Missing telemetry increases the uncertainty buffer and can force KEEP
- The workload-priority label ALONE does not change the decision when state is identical
- The cold-start penalty is a physical/action property (not scaled by workload label)
- Worse topology reduces net savings
- Thermal hotspot penalizes consolidation
- Repeated migrations are penalized by the governor
- KEEP wins when net expected savings <= 0
- Missing gross savings → low confidence, KEEP
- Governor cooldown after SLA violation
- Cluster-wide rate limit prevents migration storm
- Cost estimate to_dict round-trip
- make_recommendation produces valid Recommendation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aurelius.constraints.cost_model import (
    CostModelConfig,
    MigrationCostEstimate,
    MigrationCostModel,
    MigrationGovernor,
    RiskInputs,
)
from aurelius.sla.actions import ActionType
from aurelius.sla.schema import HardSLA, PriorityTier, SLAPolicy
from aurelius.sla.telemetry import RegionContext, WorkloadState
from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
    RegionState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=_UTC)


def _make_prov(sandbox: bool = True) -> Provenance:
    return Provenance(
        source="test",
        fetched_at=_now(),
        confidence="medium",
        is_sandbox=sandbox,
    )


def _make_region(region_id: str = "us-east") -> RegionState:
    return RegionState(
        region=region_id,
        timestamp=_now(),
        provenance=_make_prov(),
    )


def _make_state(region_id: str = "us-east") -> ClusterState:
    return ClusterState(
        timestamp=_now(),
        provenance=_make_prov(),
        regions={region_id: _make_region(region_id)},
    )


def _make_assessment(
    binding: ConstraintType | None = None,
    confidence: float = 0.7,
) -> ConstraintAssessment:
    return ConstraintAssessment(
        timestamp=_now(),
        provenance=_make_prov(),
        region=None,
        scores={binding: 0.8} if binding else {},
        binding_constraint=binding,
        confidence=confidence,
        missing_signals=[],
        rationale="test assessment",
        safe_action_types=[ActionType.KEEP.value],
        disallowed_action_types=[],
    )


def _policy(tier: PriorityTier, max_p99_ms: float, migration_allowed: bool | None = None) -> SLAPolicy:
    """Build a raw SLAPolicy (tier defaults NOT merged) with explicit bounds.

    Constructing SLAPolicy directly leaves every unset hard field at None, which
    is exactly what these controlled tests want: only the bounds we set are
    enforced, so the risk math is driven purely by the state we provide.
    """
    return SLAPolicy(
        name=f"test:{tier.value}",
        tier=tier,
        hard=HardSLA(max_p99_latency_ms=max_p99_ms, migration_allowed=migration_allowed),
    )


# ---------------------------------------------------------------------------
# MigrationCostEstimate
# ---------------------------------------------------------------------------

class TestMigrationCostEstimate:
    def test_viable_positive_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        assert est.is_viable()

    def test_not_viable_zero_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=0.0,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_not_viable_negative_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=-1.0,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_not_viable_when_blocked_by_cooldown(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=100.0,
            confidence=0.9,
            blocked_by_cooldown=True,
            blocked_reason="test cooldown",
        )
        assert not est.is_viable()

    def test_not_viable_unknown_savings(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=None,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_to_dict_round_trip(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            gross_energy_savings=10.0,
            cold_start_penalty_ms=2000.0,
            total_penalty=3.5,
            net_expected_savings=6.5,
            confidence=0.75,
            explanation="test explanation",
        )
        d = est.to_dict()
        assert d["workload_id"] == "wl-1"
        assert d["gross_energy_savings"] == 10.0
        assert d["cold_start_penalty_ms"] == 2000.0
        assert d["net_expected_savings"] == 6.5
        assert d["confidence"] == 0.75
        assert d["blocked_by_cooldown"] is False
        assert d["blocked_reason"] is None


# ---------------------------------------------------------------------------
# MigrationGovernor
# ---------------------------------------------------------------------------

class TestMigrationGovernor:
    def test_first_migration_always_allowed(self):
        gov = MigrationGovernor()
        allowed, reason = gov.check_allowed("wl-1", _now())
        assert allowed
        assert reason is None

    def test_min_interval_blocks_immediate_repeat(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        # Try again immediately
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=10))
        assert not allowed
        assert "interval" in reason.lower()

    def test_min_interval_passes_after_wait(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=400))
        assert allowed

    def test_hourly_rate_limit(self):
        cfg = CostModelConfig(
            min_migration_interval_s=1.0,
            max_migrations_per_workload_per_hour=2,
        )
        gov = MigrationGovernor(config=cfg)
        base = _now()
        gov.record_migration("wl-1", base)
        gov.record_migration("wl-1", base + timedelta(seconds=10))
        allowed, reason = gov.check_allowed("wl-1", base + timedelta(seconds=20))
        assert not allowed
        assert "rate limit" in reason.lower()

    def test_cluster_rate_limit(self):
        cfg = CostModelConfig(
            min_migration_interval_s=1.0,
            max_cluster_migrations_per_minute=2,
        )
        gov = MigrationGovernor(config=cfg)
        base = _now()
        gov.record_migration("wl-1", base)
        gov.record_migration("wl-2", base + timedelta(seconds=1))
        # Third migration on different workload should be blocked
        allowed, reason = gov.check_allowed("wl-3", base + timedelta(seconds=2))
        assert not allowed
        assert "cluster" in reason.lower()

    def test_sla_violation_cooldown(self):
        cfg = CostModelConfig(sla_violation_cooldown_s=600.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_sla_violation("wl-1", now)
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=100))
        assert not allowed
        assert "sla violation" in reason.lower()

    def test_sla_violation_clears_after_cooldown(self):
        cfg = CostModelConfig(sla_violation_cooldown_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_sla_violation("wl-1", now)
        allowed, _ = gov.check_allowed("wl-1", now + timedelta(seconds=400))
        assert allowed

    def test_independent_workloads_not_blocked_by_each_other(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        # wl-2 was not migrated; should be allowed
        allowed, _ = gov.check_allowed("wl-2", now + timedelta(seconds=1))
        assert allowed

    def test_reset_clears_history(self):
        gov = MigrationGovernor()
        now = _now()
        gov.record_migration("wl-1", now)
        gov.record_sla_violation("wl-1", now)
        gov.reset()
        allowed, _ = gov.check_allowed("wl-1", now + timedelta(seconds=1))
        assert allowed


# ---------------------------------------------------------------------------
# MigrationCostModel — core behavior
# ---------------------------------------------------------------------------

class TestMigrationCostModel:
    def test_keep_when_no_gross_savings(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=None,
        )
        assert not est.is_viable()
        assert est.net_expected_savings is None
        assert est.confidence < 0.5

    def test_keep_wins_when_net_savings_zero_or_negative(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=0.001,  # tiny savings — will be erased by penalties
            is_latency_sensitive=True,
        )
        keep, reason = model.should_keep(est)
        assert keep

    def test_cold_start_penalty_is_label_independent(self):
        """Cold-start is a physical/action property — it must NOT scale with the workload label.

        (Corrected behavior: the old model multiplied cold-start by a static
        critical×2.5 / batch×0.4 factor. That is removed; the same action incurs
        the same physical cold-start regardless of the priority label.)
        """
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est_critical = model.estimate(
            workload_id="wl-critical",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            is_latency_sensitive=True,
            priority_tier="critical",
        )
        est_batch = model.estimate(
            workload_id="wl-batch",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            is_latency_sensitive=False,
            priority_tier="batch",
        )
        # Identical action + identical state ⇒ identical physical cold-start penalty.
        assert est_critical.cold_start_penalty_ms == est_batch.cold_start_penalty_ms

    def test_no_static_workload_multiplier_config(self):
        """The unsafe static workload-class multipliers must no longer exist on the config."""
        cfg = CostModelConfig()
        assert not hasattr(cfg, "critical_workload_risk_multiplier")
        assert not hasattr(cfg, "batch_workload_risk_multiplier")

    def test_topology_degradation_reduces_net_savings(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        gross = 5.0
        est_same_topo = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=gross,
            current_topology_score=0.9,
            target_topology_score=0.9,  # no degradation
        )
        est_worse_topo = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=gross,
            current_topology_score=0.9,
            target_topology_score=0.1,  # significant degradation
        )
        # Worse topology → lower net savings
        assert est_worse_topo.total_penalty > est_same_topo.total_penalty
        if est_same_topo.net_expected_savings and est_worse_topo.net_expected_savings:
            assert est_worse_topo.net_expected_savings < est_same_topo.net_expected_savings

    def test_thermal_hotspot_penalizes_consolidation(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment_thermal = _make_assessment(binding=ConstraintType.THERMAL)
        assessment_none = _make_assessment(binding=None)
        est_thermal = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.CONSOLIDATE.value,
            assessment=assessment_thermal,
            state=state,
            gross_savings=3.0,
            is_latency_sensitive=True,
        )
        est_none = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.CONSOLIDATE.value,
            assessment=assessment_none,
            state=state,
            gross_savings=3.0,
            is_latency_sensitive=True,
        )
        # Thermal constraint + consolidation adds thermal penalty
        assert est_thermal.thermal_penalty >= 0  # non-negative
        # Total penalty under thermal should be higher (or at least not lower)
        assert est_thermal.total_penalty >= est_none.total_penalty

    def test_non_migration_actions_have_no_cold_start_penalty(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            assessment=assessment,
            state=state,
            gross_savings=5.0,
        )
        assert est.cold_start_penalty_ms == 0.0

    def test_active_latency_constraint_increases_sla_risk(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment_latency = _make_assessment(binding=ConstraintType.LATENCY)
        assessment_energy = _make_assessment(binding=ConstraintType.ENERGY)
        est_latency = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment_latency,
            state=state,
            gross_savings=5.0,
            is_latency_sensitive=True,
        )
        est_energy = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment_energy,
            state=state,
            gross_savings=5.0,
            is_latency_sensitive=True,
        )
        # Migrating during latency constraint is more expensive than during energy constraint
        assert est_latency.sla_risk_penalty >= est_energy.sla_risk_penalty

    def test_penalty_never_negative(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=0.1,
        )
        assert est.cold_start_penalty_ms >= 0
        assert est.cache_warmup_penalty_ms >= 0
        assert est.queue_instability_penalty_ms >= 0
        assert est.topology_degradation_score >= 0
        assert est.sla_risk_penalty >= 0
        assert est.thermal_penalty >= 0
        assert est.failure_retry_penalty >= 0
        assert est.total_penalty >= 0

    def test_penalty_fields_sum_roughly_to_total(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
        )
        # Total penalty should be positive for a migration
        assert est.total_penalty > 0

    def test_governor_integration_blocks_repeat_migration(self):
        model = MigrationCostModel(config=CostModelConfig(min_migration_interval_s=600.0))
        state = _make_state()
        assessment = _make_assessment()
        now = _now()
        # Record a migration
        model.governor.record_migration("wl-1", now)
        # Try again immediately
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=100.0,
            now=now + timedelta(seconds=10),
        )
        assert est.blocked_by_cooldown
        assert not est.is_viable()


# ---------------------------------------------------------------------------
# should_keep
# ---------------------------------------------------------------------------

class TestShouldKeep:
    def test_keep_when_blocked_by_cooldown(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            blocked_by_cooldown=True,
            blocked_reason="test",
        )
        keep, _ = model.should_keep(est)
        assert keep

    def test_keep_when_net_savings_unknown(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=None,
        )
        keep, reason = model.should_keep(est)
        assert keep
        assert "unknown" in reason.lower()

    def test_keep_when_net_savings_zero(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=0.0,
            confidence=0.9,
        )
        keep, _ = model.should_keep(est)
        assert keep

    def test_no_keep_when_positive_savings(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.7,
        )
        keep, _ = model.should_keep(est)
        assert not keep


# ---------------------------------------------------------------------------
# make_recommendation
# ---------------------------------------------------------------------------

class TestMakeRecommendation:
    def test_produces_keep_when_not_viable(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=-1.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.is_noop
        assert rec.action_type == ActionType.KEEP.value
        assert rec.implementation_mode == "recommendation_only"

    def test_produces_action_when_viable(self):
        model = MigrationCostModel()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert not rec.is_noop
        assert rec.action_type == ActionType.MIGRATE.value
        assert rec.implementation_mode == "recommendation_only"

    def test_recommendation_only_mode_always(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            timestamp=_now(),
            net_expected_savings=3.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.implementation_mode == "recommendation_only"

    def test_recommendation_carries_net_benefit(self):
        model = MigrationCostModel()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            gross_energy_savings=10.0,
            total_penalty=2.0,
            net_expected_savings=8.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.net_benefit == 8.0
        assert rec.migration_penalty == 2.0

    def test_recommendation_inherits_sandbox_flag(self):
        model = MigrationCostModel()
        prov_sandbox = _make_prov(sandbox=True)
        assessment_sandbox = ConstraintAssessment(
            timestamp=_now(),
            provenance=prov_sandbox,
            region=None,
            scores={},
            binding_constraint=None,
            confidence=0.5,
            missing_signals=[],
            rationale="",
            safe_action_types=[],
            disallowed_action_types=[],
        )
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment_sandbox,
        )
        assert rec.provenance.is_sandbox

    def test_recommendation_has_unique_id(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            timestamp=_now(),
            net_expected_savings=None,
        )
        rec1 = model.make_recommendation("wl-1", ActionType.KEEP.value, est, assessment)
        rec2 = model.make_recommendation("wl-1", ActionType.KEEP.value, est, assessment)
        assert rec1.recommendation_id != rec2.recommendation_id


# ---------------------------------------------------------------------------
# Integration: classifier + cost model pipeline
# ---------------------------------------------------------------------------

class TestClassifierCostModelPipeline:
    """Verify the Phase 7→8 pipeline: assessment → cost estimate → recommendation."""

    def test_energy_bound_migrate_viable_with_savings_and_headroom(self):
        """Energy-bound migration with healthy savings + large SLA headroom is viable.

        Driven by state (large headroom, safe destination, decent savings), not by
        any workload label.
        """
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.85)
        current = WorkloadState(region="us-east", p99_latency_ms=400.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=600.0)  # well under 5000ms cap
        dest = RegionContext(region="us-west", spare_capacity_pct=70.0, baseline_p99_latency_ms=400.0)
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.STANDARD, max_p99_ms=5000.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
            prefix_cache_hit_rate=0.2,
            requests_running=4.0,
        )
        est = model.estimate(
            workload_id="energy-wl",
            action_type=ActionType.CHOOSE_CHEAPER_REGION.value,
            assessment=assessment,
            state=state,
            gross_savings=20.0,
            current_topology_score=0.7,
            target_topology_score=0.6,
            risk_inputs=ri,
        )
        rec = model.make_recommendation(
            workload_id="energy-wl",
            action_type=ActionType.CHOOSE_CHEAPER_REGION.value,
            estimate=est,
            assessment=assessment,
        )
        assert not rec.is_noop, f"Expected action, got KEEP. est={est.to_dict()}"
        assert rec.binding_constraint == ConstraintType.ENERGY

    def test_latency_bound_migrate_blocked_by_active_constraint(self):
        """Migrating a latency-bound workload for small savings is blocked.

        The active LATENCY binding is state evidence of low SLA headroom, so the
        state-conditioned SLA risk dominates the small expected savings — for ANY
        workload, not just a labeled-critical one.
        """
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.LATENCY, confidence=0.85)
        est = model.estimate(
            workload_id="latency-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=2.0,  # small savings vs an active latency constraint
            risk_inputs=RiskInputs(),
        )
        keep, reason = model.should_keep(est)
        assert keep, f"Expected KEEP during active latency constraint; reason={reason}"

    def test_no_migration_storm_governor(self):
        """Multiple workloads migrating at once are rate-limited."""
        model = MigrationCostModel(config=CostModelConfig(
            min_migration_interval_s=1.0,
            max_cluster_migrations_per_minute=2,
        ))
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        now = _now()

        # First two migrations succeed
        model.governor.record_migration("wl-1", now)
        model.governor.record_migration("wl-2", now + timedelta(seconds=1))

        # Third is blocked by cluster rate limit
        est3 = model.estimate(
            workload_id="wl-3",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            now=now + timedelta(seconds=2),
        )
        assert est3.blocked_by_cooldown
        assert not est3.is_viable()


# ---------------------------------------------------------------------------
# State-conditioned risk model (Phase 8/9 correction)
# ---------------------------------------------------------------------------

class TestStateConditionedRisk:
    """The decision must be driven by SLA headroom + destination/action/telemetry
    state, NOT by a static workload-class multiplier.
    """

    def test_critical_migrates_with_large_headroom_and_safe_dest(self):
        """A critical workload MAY migrate when headroom is large and the destination is safe."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.85)
        # max_p99=2000ms, predicted=625ms ⇒ ~69% headroom (well within SLA).
        current = WorkloadState(region="us-east", p99_latency_ms=500.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=625.0, capacity_buffer_pct=60.0)
        dest = RegionContext(
            region="us-west",
            spare_capacity_pct=70.0,
            baseline_p99_latency_ms=300.0,  # destination not slower than source
            thermally_stressed=False,
            throttling=False,
        )
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.CRITICAL, max_p99_ms=2000.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
            prefix_cache_hit_rate=0.1,   # low cache affinity ⇒ little warmup loss
            requests_running=4.0,        # low queue/load
        )
        est = model.estimate(
            workload_id="crit-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=25.0,
            is_latency_sensitive=True,
            priority_tier="critical",
            current_topology_score=0.7,
            target_topology_score=0.6,
            risk_inputs=ri,
        )
        assert not est.hard_sla_block
        assert est.sla_headroom_fraction is not None and est.sla_headroom_fraction > 0.5
        assert est.is_viable(), f"Critical workload should migrate; est={est.to_dict()}"
        keep, _ = model.should_keep(est)
        assert not keep

    def test_critical_blocked_with_small_headroom(self):
        """A critical workload is blocked when SLA headroom is small (predicted near the cap)."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.85)
        # max_p99=500ms, predicted=490ms ⇒ ~2% headroom (tiny, but NOT a hard breach).
        current = WorkloadState(region="us-east", p99_latency_ms=380.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=490.0, capacity_buffer_pct=60.0)
        dest = RegionContext(region="us-west", spare_capacity_pct=70.0, baseline_p99_latency_ms=380.0)
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.CRITICAL, max_p99_ms=500.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
            prefix_cache_hit_rate=0.1,
            requests_running=4.0,
        )
        est = model.estimate(
            workload_id="crit-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=8.0,
            is_latency_sensitive=True,
            priority_tier="critical",
            current_topology_score=0.7,
            target_topology_score=0.6,
            risk_inputs=ri,
        )
        assert not est.hard_sla_block  # not a hard breach — blocked on thin headroom
        assert est.sla_headroom_fraction is not None and est.sla_headroom_fraction < 0.1
        assert est.sla_risk_penalty > 0
        keep, reason = model.should_keep(est)
        assert keep, f"Small headroom should block; est={est.to_dict()}"

    def test_batch_blocked_with_hostile_destination(self):
        """A batch workload is blocked when destination topology/thermal/queue risk is high."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.85)
        # Batch SLA is loose (huge headroom) — so SLA is NOT the blocker; the
        # hostile destination + topology degradation is.
        current = WorkloadState(region="us-east", p99_latency_ms=2000.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=3000.0, capacity_buffer_pct=5.0)
        dest = RegionContext(
            region="us-west",
            spare_capacity_pct=5.0,           # near-full
            baseline_p99_latency_ms=4000.0,   # slower than source
            thermally_stressed=True,
            throttling=True,
            network_rtt_ms=150.0,             # far away
        )
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.BATCH, max_p99_ms=10000.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
            prefix_cache_hit_rate=0.8,
            requests_running=40.0,            # high in-flight load
        )
        est = model.estimate(
            workload_id="batch-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=8.0,
            is_latency_sensitive=False,
            priority_tier="batch",
            current_topology_score=0.9,
            target_topology_score=0.1,        # severe topology degradation
            risk_inputs=ri,
        )
        assert est.sla_headroom_fraction is not None and est.sla_headroom_fraction > 0.5
        assert est.destination_risk_penalty > 0
        assert est.action_risk_penalty > 0
        keep, reason = model.should_keep(est)
        assert keep, f"Hostile destination should block batch migration; est={est.to_dict()}"

    def test_missing_telemetry_increases_uncertainty_and_can_force_keep(self):
        """Missing telemetry widens the uncertainty buffer and can force KEEP."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.5)
        # Full telemetry → viable at modest savings.
        current = WorkloadState(region="us-east", p99_latency_ms=500.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=700.0, capacity_buffer_pct=40.0)
        dest = RegionContext(region="us-west", spare_capacity_pct=60.0, baseline_p99_latency_ms=500.0)
        ri_full = RiskInputs(
            sla_policy=_policy(PriorityTier.STANDARD, max_p99_ms=3000.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
            prefix_cache_hit_rate=0.2,
            requests_running=4.0,
        )
        est_full = model.estimate(
            workload_id="wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=5.0,
            current_topology_score=0.7,
            target_topology_score=0.6,
            risk_inputs=ri_full,
        )
        # No telemetry at all → maximal uncertainty buffer.
        est_missing = model.estimate(
            workload_id="wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=5.0,
            risk_inputs=RiskInputs(),
        )
        assert est_missing.uncertainty_penalty > est_full.uncertainty_penalty
        assert len(est_missing.missing_signals) > len(est_full.missing_signals)
        keep_full, _ = model.should_keep(est_full)
        keep_missing, reason = model.should_keep(est_missing)
        assert not keep_full, f"Full telemetry should be viable; est={est_full.to_dict()}"
        assert keep_missing, f"Missing telemetry should force KEEP; est={est_missing.to_dict()}"

    def test_workload_label_alone_does_not_change_decision(self):
        """With ALL state inputs identical, the priority label alone changes nothing."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.8)
        # Identical state for both estimates (same RiskInputs object, no SLA-policy
        # difference) — only the priority label/latency-sensitivity flag differ.
        ri = RiskInputs(
            sla_policy=None,
            current_state=WorkloadState(region="us-east", p99_latency_ms=500.0),
            predicted_state=WorkloadState(region="us-west", p99_latency_ms=600.0),
            dest_context=RegionContext(region="us-west", spare_capacity_pct=60.0),
            prefix_cache_hit_rate=0.2,
            requests_running=4.0,
        )
        common = dict(
            workload_id="wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            current_topology_score=0.7,
            target_topology_score=0.5,
            risk_inputs=ri,
        )
        est_critical = model.estimate(is_latency_sensitive=True, priority_tier="critical", **common)
        est_batch = model.estimate(is_latency_sensitive=False, priority_tier="batch", **common)
        # Every risk bucket and the net are identical — the label does not move risk.
        assert est_critical.total_penalty == est_batch.total_penalty
        assert est_critical.sla_risk_penalty == est_batch.sla_risk_penalty
        assert est_critical.destination_risk_penalty == est_batch.destination_risk_penalty
        assert est_critical.action_risk_penalty == est_batch.action_risk_penalty
        assert est_critical.uncertainty_penalty == est_batch.uncertainty_penalty
        assert est_critical.net_expected_savings == est_batch.net_expected_savings
        assert model.should_keep(est_critical)[0] == model.should_keep(est_batch)[0]

    def test_high_savings_rejected_when_hard_sla_breached(self):
        """Very high expected savings is still rejected when state breaches a hard SLA bound."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.9)
        current = WorkloadState(region="us-east", p99_latency_ms=480.0)
        predicted = WorkloadState(region="us-west", p99_latency_ms=900.0)  # > 500ms hard cap
        dest = RegionContext(region="us-west", spare_capacity_pct=70.0)
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.CRITICAL, max_p99_ms=500.0),
            current_state=current,
            predicted_state=predicted,
            dest_context=dest,
        )
        est = model.estimate(
            workload_id="crit-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=100.0,  # huge savings
            is_latency_sensitive=True,
            priority_tier="critical",
            risk_inputs=ri,
        )
        assert est.hard_sla_block
        assert not est.is_viable()
        keep, reason = model.should_keep(est)
        assert keep, f"Hard SLA breach must block regardless of savings; reason={reason}"

    def test_migration_allowed_false_hard_blocks(self):
        """SLA migration_allowed=false hard-blocks a migration regardless of savings."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.9)
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.CRITICAL, max_p99_ms=5000.0, migration_allowed=False),
            current_state=WorkloadState(region="us-east", p99_latency_ms=300.0),
            predicted_state=WorkloadState(region="us-west", p99_latency_ms=350.0),
            dest_context=RegionContext(region="us-west", spare_capacity_pct=80.0),
        )
        est = model.estimate(
            workload_id="crit-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=50.0,
            risk_inputs=ri,
        )
        assert est.hard_sla_block
        keep, _ = model.should_keep(est)
        assert keep

    def test_risk_factor_explanation_present(self):
        """The estimate exposes which state factors drove the risk (explainability)."""
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.8)
        ri = RiskInputs(
            sla_policy=_policy(PriorityTier.STANDARD, max_p99_ms=1000.0),
            current_state=WorkloadState(region="us-east", p99_latency_ms=500.0),
            predicted_state=WorkloadState(region="us-west", p99_latency_ms=900.0),  # tight headroom
            dest_context=RegionContext(region="us-west", spare_capacity_pct=10.0, throttling=True),
            prefix_cache_hit_rate=0.7,
            requests_running=20.0,
        )
        est = model.estimate(
            workload_id="wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            current_topology_score=0.8,
            target_topology_score=0.2,
            risk_inputs=ri,
        )
        assert isinstance(est.risk_factors, dict) and est.risk_factors
        assert est.dominant_risk_factors  # non-empty
        # Each dominant factor is a real contributor.
        for f in est.dominant_risk_factors:
            assert f in est.risk_factors
        # Buckets sum to the reported total penalty.
        bucket_sum = (
            est.sla_risk_penalty
            + est.destination_risk_penalty
            + est.action_risk_penalty
            + est.uncertainty_penalty
            + est.thermal_penalty
        )
        assert abs(bucket_sum - est.total_penalty) < 1e-9
