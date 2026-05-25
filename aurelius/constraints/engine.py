"""Constraint-aware recommendation engine — Phase 9.

Pipeline for each ClusterState snapshot:
1. ConstraintClassifier.assess(state) → ConstraintAssessment
2. Per-service: generate candidate OptimizationActions for the binding constraint
3. SLAAwareActionSelector.select(candidates) → SLADecision  (SLA gate)
4. MigrationCostModel.estimate(best_action) → MigrationCostEstimate  (cost gate)
5. Emit Recommendation (always recommendation_only mode by default)

Design rules:
- Read-only over ClusterState. Never mutates cluster state or runtime internals.
- All recommendations are in recommendation_only mode unless explicitly changed.
- Missing SLA policy → SLA gate passes with no penalty (preserves pre-SLA behavior).
- Low-confidence assessment → all services get KEEP (fail-safe).
- Empty ClusterState → empty recommendation list (no crash).
- is_sandbox=True from ClusterState provenance passes through to all outputs.
- Disallowed actions from ConstraintAssessment are rejected before SLA gate.
- Cost model gate: actions with non-positive net savings become KEEP.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..sla.actions import ActionType, OptimizationAction
from ..sla.loader import SLARegistry
from ..sla.selector import SLAAwareActionSelector
from ..sla.telemetry import RegionContext, WorkloadState
from ..state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    InferenceServiceState,
    Provenance,
    Recommendation,
)
from .classifier import ConstraintClassifier, ConstraintConfig
from .cost_model import CostModelConfig, MigrationCostModel, RiskInputs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workload descriptor (adapter for SLAAwareActionSelector)
# ---------------------------------------------------------------------------

@dataclass
class WorkloadDescriptor:
    """Minimal workload descriptor that satisfies SLAAwareActionSelector's interface.

    The selector uses getattr(workload, "job_id", ...) to extract the workload id.
    """
    job_id: str
    workload_type: str = "realtime_inference"


# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Output of one ConstraintAwareEngine.run() cycle."""
    assessment: ConstraintAssessment
    recommendations: list[Recommendation]
    # Actions evaluated and rejected before becoming recommendations (observability)
    rejected: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def noop_count(self) -> int:
        return sum(1 for r in self.recommendations if r.is_noop)

    @property
    def actionable_count(self) -> int:
        return len(self.recommendations) - self.noop_count

    def to_dict(self) -> dict:
        return {
            "assessment": self.assessment.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "rejected": list(self.rejected),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "noop_count": self.noop_count,
            "actionable_count": self.actionable_count,
        }


# ---------------------------------------------------------------------------
# Per-constraint candidate action generators
# ---------------------------------------------------------------------------

def _cheaper_regions(
    state: ClusterState, current_region: str
) -> list[tuple[str, float]]:
    """Return (region_id, price) pairs for regions cheaper than current, sorted ascending."""
    current_price: Optional[float] = None
    reg = state.regions.get(current_region)
    if reg and reg.energy:
        current_price = reg.energy.price_per_mwh
    if current_price is None:
        return []

    cheaper: list[tuple[str, float]] = []
    for region_id, region in state.regions.items():
        if region_id == current_region:
            continue
        if region.energy and region.energy.price_per_mwh is not None:
            if region.energy.price_per_mwh < current_price:
                cheaper.append((region_id, region.energy.price_per_mwh))
    return sorted(cheaper, key=lambda x: x[1])


def _gen_energy(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """ENERGY-bound: shift to cheaper region; offer DEFER for flexible workloads."""
    candidates: list[OptimizationAction] = []
    current_region = service.region
    if current_region is None:
        return candidates

    current_price: Optional[float] = None
    reg = state.regions.get(current_region)
    if reg and reg.energy:
        current_price = reg.energy.price_per_mwh

    for target_region, target_price in _cheaper_regions(state, current_region):
        if current_price and current_price > 0:
            savings_pct = (current_price - target_price) / current_price * 100.0
        else:
            savings_pct = 5.0  # HEURISTIC: assume 5% savings when price unknown
        candidates.append(OptimizationAction(
            action_type=ActionType.CHOOSE_CHEAPER_REGION,
            target_region=target_region,
            expected_savings_pct=savings_pct,
            description=(
                f"Move {service.service_id} from {current_region} "
                f"to cheaper {target_region}"
            ),
        ))

    # DEFER: off-peak scheduling for any service (batch-friendly)
    candidates.append(OptimizationAction(
        action_type=ActionType.DEFER,
        target_region=current_region,
        expected_savings_pct=3.0,  # HEURISTIC
        description=f"Defer {service.service_id} to off-peak pricing window",
    ))
    return candidates


def _gen_thermal(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """THERMAL-bound: spread load; reroute to cooler regions. Never CONSOLIDATE."""
    current_region = service.region or "unknown"
    candidates: list[OptimizationAction] = [
        OptimizationAction(
            action_type=ActionType.SPREAD,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=(
                f"Spread {service.service_id} to reduce thermal density "
                f"in {current_region}"
            ),
        )
    ]

    # Offer REROUTE to cooler regions if any exist
    current_max_temp: Optional[float] = None
    cur_reg = state.regions.get(current_region)
    if cur_reg and cur_reg.thermal:
        current_max_temp = cur_reg.thermal.max_gpu_temp_c

    for region_id, region in state.regions.items():
        if region_id == current_region:
            continue
        if region.thermal and region.thermal.max_gpu_temp_c is not None:
            if current_max_temp is None or region.thermal.max_gpu_temp_c < current_max_temp - 5.0:
                candidates.append(OptimizationAction(
                    action_type=ActionType.REROUTE,
                    target_region=region_id,
                    expected_savings_pct=2.0,  # HEURISTIC: thermal relief avoids throttle cost
                    description=(
                        f"Reroute {service.service_id} to cooler region {region_id}"
                    ),
                ))
    return candidates


def _gen_queue(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """QUEUE-bound: add replicas; spread to reduce per-service queue depth."""
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.SCALE_REPLICAS,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=f"Add replicas for {service.service_id} to absorb queue surge",
            metadata={"target_replicas_delta": 1},
        ),
        OptimizationAction(
            action_type=ActionType.SPREAD,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=f"Spread {service.service_id} to reduce queue concentration",
        ),
    ]


def _gen_latency(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """LATENCY-bound: scale replicas to reduce tail latency. Never migrate.

    MIGRATE is in the disallowed list for LATENCY — the SLA/classifier gate
    enforces this. The generator does not emit MIGRATE candidates here.
    """
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.SCALE_REPLICAS,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=(
                f"Add replicas for {service.service_id} to reduce p99 latency tail"
            ),
            metadata={"target_replicas_delta": 1},
        ),
    ]


def _gen_communication(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """COMMUNICATION-bound: change placement to NVLink/NVSwitch domain."""
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.CHANGE_PLACEMENT,
            target_region=current_region,
            expected_savings_pct=5.0,  # HEURISTIC: ~5% throughput gain from better topology
            description=(
                f"Move {service.service_id} to NVLink/NVSwitch GPU group "
                f"to reduce communication stall"
            ),
        ),
        OptimizationAction(
            action_type=ActionType.SPREAD,
            target_region=current_region,
            expected_savings_pct=2.0,
            description=(
                f"Spread {service.service_id} to reduce communication contention"
            ),
        ),
    ]


def _gen_memory(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """MEMORY-bound (indirect): spread + scale to reduce per-replica KV/HBM pressure."""
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.SPREAD,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=(
                f"Spread {service.service_id} to reduce memory/KV cache pressure"
            ),
        ),
        OptimizationAction(
            action_type=ActionType.SCALE_REPLICAS,
            target_region=current_region,
            expected_savings_pct=0.0,
            description=(
                f"Add replicas for {service.service_id} to reduce per-replica KV pressure"
            ),
            metadata={"target_replicas_delta": 1},
        ),
    ]


def _gen_topology(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """TOPOLOGY-bound: change placement to NVSwitch domain; consolidate onto fast links."""
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.CHANGE_PLACEMENT,
            target_region=current_region,
            expected_savings_pct=8.0,  # HEURISTIC: ~8% effective throughput from NVSwitch
            description=(
                f"Move {service.service_id} to NVSwitch GPU group "
                f"for optimal topology"
            ),
        ),
        OptimizationAction(
            action_type=ActionType.CONSOLIDATE,
            target_region=current_region,
            expected_savings_pct=5.0,
            description=(
                f"Consolidate {service.service_id} onto NVLink-connected GPUs"
            ),
        ),
    ]


def _gen_utilization(
    service: InferenceServiceState,
    state: ClusterState,
    assessment: ConstraintAssessment,
) -> list[OptimizationAction]:
    """UTILIZATION-bound (underutilization): consolidate to reduce fragmentation."""
    current_region = service.region or "unknown"
    return [
        OptimizationAction(
            action_type=ActionType.CONSOLIDATE,
            target_region=current_region,
            expected_savings_pct=10.0,  # HEURISTIC: consolidating idle GPUs reduces cost
            description=(
                f"Consolidate {service.service_id} to reduce GPU fragmentation"
            ),
        ),
    ]


_CANDIDATE_GENERATORS = {
    ConstraintType.ENERGY: _gen_energy,
    ConstraintType.THERMAL: _gen_thermal,
    ConstraintType.QUEUE: _gen_queue,
    ConstraintType.LATENCY: _gen_latency,
    ConstraintType.COMMUNICATION: _gen_communication,
    ConstraintType.MEMORY: _gen_memory,
    ConstraintType.TOPOLOGY: _gen_topology,
    ConstraintType.UTILIZATION: _gen_utilization,
    ConstraintType.NONE: lambda *_: [],
}


# ---------------------------------------------------------------------------
# State adapters
# ---------------------------------------------------------------------------

def _service_to_sla_workload_state(
    service: InferenceServiceState,
    state: ClusterState,
    migration_count_last_hour: int = 0,
) -> WorkloadState:
    """Map InferenceServiceState → sla.WorkloadState for the SLA evaluator."""
    current_region = service.region
    energy_price: Optional[float] = None
    carbon_intensity: Optional[float] = None
    energy_price_percentile: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    capacity_buffer_pct: Optional[float] = None

    if current_region:
        reg = state.regions.get(current_region)
        if reg:
            if reg.energy:
                energy_price = reg.energy.price_per_mwh
                carbon_intensity = reg.energy.carbon_gco2_per_kwh
                energy_price_percentile = reg.energy.price_percentile
            capacity_buffer_pct = reg.spare_capacity_pct
            util_vals = [
                gpu.util_pct
                for node in reg.nodes.values()
                for gpu in node.gpus.values()
                if gpu.util_pct is not None
            ]
            if util_vals:
                gpu_utilization_pct = sum(util_vals) / len(util_vals)

    return WorkloadState(
        region=current_region,
        timestamp=service.timestamp,
        p95_latency_ms=service.p95_latency_ms,
        p99_latency_ms=service.p99_latency_ms,
        queue_wait_ms=service.queue_time_p95_ms,
        gpu_utilization_pct=gpu_utilization_pct,
        error_rate_pct=service.error_rate_pct,
        capacity_buffer_pct=capacity_buffer_pct,
        migration_count_last_hour=migration_count_last_hour,
        energy_price=energy_price,
        carbon_intensity=carbon_intensity,
        energy_price_percentile=energy_price_percentile,
    )


def _build_region_contexts(state: ClusterState) -> dict[str, RegionContext]:
    """Build RegionContext map from ClusterState for the HeuristicPredictor."""
    contexts: dict[str, RegionContext] = {}
    for region_id, region in state.regions.items():
        thermally_stressed = False
        throttling = False
        if region.thermal:
            th = region.thermal
            if th.max_gpu_temp_c is not None and th.max_gpu_temp_c > 83.0:  # HEURISTIC
                thermally_stressed = True
            if th.throttling_fraction is not None and th.throttling_fraction > 0.0:
                throttling = True

        baseline_p99: Optional[float] = None
        baseline_q: Optional[float] = None
        for svc in region.services.values():
            if svc.p99_latency_ms is not None:
                baseline_p99 = (
                    svc.p99_latency_ms if baseline_p99 is None
                    else max(baseline_p99, svc.p99_latency_ms)
                )
            if svc.queue_time_p95_ms is not None:
                baseline_q = (
                    svc.queue_time_p95_ms if baseline_q is None
                    else max(baseline_q, svc.queue_time_p95_ms)
                )

        contexts[region_id] = RegionContext(
            region=region_id,
            spare_capacity_pct=region.spare_capacity_pct or 50.0,
            baseline_p99_latency_ms=baseline_p99,
            baseline_queue_wait_ms=baseline_q,
            thermally_stressed=thermally_stressed,
            throttling=throttling,
            energy_price=region.energy.price_per_mwh if region.energy else None,
            carbon_intensity=(
                region.energy.carbon_gco2_per_kwh if region.energy else None
            ),
            energy_price_percentile=(
                region.energy.price_percentile if region.energy else None
            ),
        )
    return contexts


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ConstraintAwareEngine:
    """Constraint-aware recommendation engine.

    Generates per-service Recommendations from a ClusterState snapshot by:
    1. Classifying the binding constraint
    2. Generating constraint-appropriate candidate actions
    3. Filtering through the SLA gate and migration cost model
    4. Emitting Recommendation objects in recommendation_only mode

    Usage::

        engine = ConstraintAwareEngine()
        result = engine.run(state, sla_registry=registry)
        for rec in result.recommendations:
            if not rec.is_noop:
                print(rec.action_type, rec.rationale)
    """

    def __init__(
        self,
        classifier: Optional[ConstraintClassifier] = None,
        cost_model: Optional[MigrationCostModel] = None,
        classifier_config: Optional[ConstraintConfig] = None,
        cost_config: Optional[CostModelConfig] = None,
        implementation_mode: str = "recommendation_only",
    ) -> None:
        self.classifier = classifier or ConstraintClassifier(
            config=classifier_config or ConstraintConfig()
        )
        self.cost_model = cost_model or MigrationCostModel(
            config=cost_config or CostModelConfig()
        )
        if implementation_mode not in Recommendation._VALID_IMPL_MODES:
            raise ValueError(
                f"implementation_mode must be one of "
                f"{sorted(Recommendation._VALID_IMPL_MODES)}, got {implementation_mode!r}"
            )
        self.implementation_mode = implementation_mode
        self._selector = SLAAwareActionSelector()

    def run(
        self,
        state: ClusterState,
        sla_registry: Optional[SLARegistry] = None,
    ) -> EngineResult:
        """Run one recommendation cycle over a ClusterState snapshot.

        Always returns in recommendation_only mode. Never mutates the cluster.
        """
        t0 = time.monotonic()
        rejected: list[dict] = []

        # 1. Classify binding constraint
        assessment = self.classifier.assess(state)

        # 2. Low-confidence fallback: KEEP everything (fail-safe)
        if assessment.confidence < self.classifier.config.confidence_floor:
            recommendations = self._keep_all(state, assessment)
            return EngineResult(
                assessment=assessment,
                recommendations=recommendations,
                rejected=rejected,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 3. Build region contexts once for the SLA predictor
        region_contexts = _build_region_contexts(state)

        # 4. Per-service recommendations
        recommendations: list[Recommendation] = []
        for service in state.all_services.values():
            rec, rej = self._recommend_service(
                service=service,
                state=state,
                assessment=assessment,
                sla_registry=sla_registry,
                region_contexts=region_contexts,
            )
            recommendations.append(rec)
            rejected.extend(rej)

        return EngineResult(
            assessment=assessment,
            recommendations=recommendations,
            rejected=rejected,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Per-service recommendation
    # ------------------------------------------------------------------

    def _recommend_service(
        self,
        service: InferenceServiceState,
        state: ClusterState,
        assessment: ConstraintAssessment,
        sla_registry: Optional[SLARegistry],
        region_contexts: dict[str, RegionContext],
    ) -> tuple[Recommendation, list[dict]]:
        """Generate one Recommendation for a single service."""
        binding = assessment.binding_constraint
        workload_id = service.service_id
        rejected: list[dict] = []

        # Resolve SLA policy
        sla_policy = None
        if sla_registry is not None:
            sla_policy = sla_registry.resolve(
                workload_id=workload_id,
                workload_type="realtime_inference",
            )

        # Determine workload risk tier from SLA policy
        priority_tier = sla_policy.tier.value if sla_policy else "standard"
        is_latency_sensitive = (
            binding == ConstraintType.LATENCY
            or priority_tier in ("critical", "latency_sensitive")
        )

        # SLA telemetry adapter
        current_ws = _service_to_sla_workload_state(service, state)

        # Generate candidates for the binding constraint
        generator = _CANDIDATE_GENERATORS.get(binding or ConstraintType.NONE)
        raw_candidates = generator(service, state, assessment) if generator else []

        # Filter candidates disallowed by the classifier for this constraint
        disallowed = set(assessment.disallowed_action_types)
        allowed_candidates: list[OptimizationAction] = []
        for action in raw_candidates:
            if action.action_type.value in disallowed:
                rejected.append({
                    "service_id": workload_id,
                    "action": action.action_type.value,
                    "target_region": action.target_region,
                    "reject_reason": (
                        f"disallowed_by_classifier for "
                        f"{binding.value if binding else 'none'}"
                    ),
                })
                logger.debug(
                    "Rejected %s for %s: disallowed by classifier (%s)",
                    action.action_type.value, workload_id,
                    binding.value if binding else "none",
                )
            else:
                allowed_candidates.append(action)

        # SLA gate: selector picks the best SLA-safe action
        wl_descriptor = WorkloadDescriptor(job_id=workload_id)
        sla_decision = self._selector.select(
            workload=wl_descriptor,
            candidate_actions=allowed_candidates,
            current_state=current_ws,
            sla_policy=sla_policy,
            region_contexts=region_contexts,
            now=state.timestamp,
        )

        # Record SLA-blocked actions for observability
        for scored in sla_decision.scored_actions:
            if not scored.sla_safe and not scored.action.is_noop:
                rejected.append({
                    "service_id": workload_id,
                    "action": scored.action.action_type.value,
                    "target_region": scored.action.target_region,
                    "reject_reason": "sla_gate: " + "; ".join(
                        scored.evaluation.violated_hard_constraints
                    ),
                })

        chosen = sla_decision.chosen_action
        sla_status = (
            "satisfied" if not sla_decision.was_corrected
            else "corrected" if not sla_decision.blocked_reasons
            else "blocked"
        )

        # Cost model gate — applied to the SLA-chosen action
        # Non-migration non-noop actions (SPREAD, SCALE, CONSOLIDATE) use gross_savings=None
        # since they represent operational improvement without direct monetary savings signal.
        gross_savings: Optional[float] = None
        if not chosen.is_noop and chosen.expected_savings_pct > 0:
            gross_savings = chosen.expected_savings_pct  # use pct as abstract cost unit

        # Topology scores: proxy from region topology availability
        current_topo_score: Optional[float] = None
        target_topo_score: Optional[float] = None
        if service.region and chosen.target_region and chosen.target_region != service.region:
            # Cross-region migration implies topology degradation
            current_topo_score = 0.7  # HEURISTIC: decent within-region topology
            target_topo_score = 0.3   # HEURISTIC: cross-region = worse topology

        # Build state-conditioned risk inputs for the chosen action. The cost
        # model derives risk from these observed/predicted states rather than from
        # any static workload-class multiplier.
        dest_ctx = (
            region_contexts.get(chosen.target_region)
            if chosen.target_region and chosen.target_region != service.region
            else None
        )
        predicted_ws = self._selector.predictor.predict(chosen, current_ws, dest_ctx)
        risk_inputs = RiskInputs(
            sla_policy=sla_policy,
            current_state=current_ws,
            predicted_state=predicted_ws,
            dest_context=dest_ctx,
            prefix_cache_hit_rate=service.prefix_cache_hit_rate,
            kv_cache_usage=service.kv_cache_usage,
            requests_running=service.requests_running,
            requests_waiting=service.requests_waiting,
            sample_age_s=service.provenance.sample_age_s,
        )

        cost_estimate = self.cost_model.estimate(
            workload_id=workload_id,
            action_type=chosen.action_type.value,
            assessment=assessment,
            state=state,
            gross_savings=gross_savings,
            is_latency_sensitive=is_latency_sensitive,
            priority_tier=priority_tier,
            current_topology_score=current_topo_score,
            target_topology_score=target_topo_score,
            now=state.timestamp,
            risk_inputs=risk_inputs,
        )

        # Cost model gate: veto the chosen action if net savings ≤ 0
        keep, keep_reason = self.cost_model.should_keep(cost_estimate)
        if keep and not chosen.is_noop:
            rejected.append({
                "service_id": workload_id,
                "action": chosen.action_type.value,
                "target_region": chosen.target_region,
                "reject_reason": f"cost_model: {keep_reason}",
            })
            logger.debug(
                "Cost model KEEP for %s (%s): %s",
                workload_id, chosen.action_type.value, keep_reason,
            )
            final_action_type = ActionType.KEEP.value
            is_noop = True
            final_sla_status = "unknown"
            net_benefit = None
            rationale = (
                f"KEEP — cost model: {keep_reason}. "
                f"Binding constraint: {binding.value if binding else 'none'} "
                f"(confidence={assessment.confidence:.2f}). "
                f"{cost_estimate.explanation}"
            )
        else:
            final_action_type = chosen.action_type.value
            is_noop = chosen.is_noop
            final_sla_status = sla_status if not is_noop else "unknown"
            net_benefit = cost_estimate.net_expected_savings if not is_noop else None
            rationale = (
                f"Binding constraint: {binding.value if binding else 'none'} "
                f"(confidence={assessment.confidence:.2f}). "
                f"SLA: {sla_status}. "
                f"Action: {chosen.description}. "
                f"{cost_estimate.explanation}"
            )

        # Provenance inherits sandbox flag from ClusterState
        is_sandbox = state.provenance.is_sandbox
        conf_level = (
            "high" if assessment.confidence >= 0.7
            else "medium" if assessment.confidence >= 0.4
            else "low"
        )
        prov = Provenance(
            source="constraint-engine",
            fetched_at=state.timestamp,
            confidence=conf_level,
            is_sandbox=is_sandbox,
        )

        # Effective confidence: min of classifier and cost model
        effective_confidence = (
            min(assessment.confidence, cost_estimate.confidence)
            if not is_noop
            else assessment.confidence
        )

        # SLA evaluation snapshot for first scored action (if any)
        sla_eval_dict: Optional[dict] = None
        if sla_decision.scored_actions:
            chosen_scored = next(
                (s for s in sla_decision.scored_actions if s.action is chosen),
                sla_decision.scored_actions[0],
            )
            sla_eval_dict = chosen_scored.evaluation.to_dict()

        # Resolve target_region from the chosen action (None for non-migration actions)
        final_target_region: Optional[str] = None
        if not is_noop and chosen.target_region and chosen.target_region != service.region:
            final_target_region = chosen.target_region

        recommendation = Recommendation(
            recommendation_id=str(uuid.uuid4()),
            workload_id=workload_id,
            action_type=final_action_type,
            timestamp=state.timestamp,
            provenance=prov,
            binding_constraint=binding,
            expected_effect={
                "gross_savings_pct": gross_savings or 0.0,
                "total_penalty": cost_estimate.total_penalty,
            },
            confidence=min(1.0, max(0.0, effective_confidence)),
            sla_status=final_sla_status,
            sla_evaluation=sla_eval_dict,
            migration_penalty=cost_estimate.total_penalty,
            net_benefit=net_benefit,
            rationale=rationale,
            is_noop=is_noop,
            implementation_mode=self.implementation_mode,
            target_region=final_target_region,
        )

        return recommendation, rejected

    # ------------------------------------------------------------------
    # Low-confidence fallback
    # ------------------------------------------------------------------

    def _keep_all(
        self,
        state: ClusterState,
        assessment: ConstraintAssessment,
    ) -> list[Recommendation]:
        """Emit KEEP for every service when classifier confidence is too low."""
        prov = Provenance(
            source="constraint-engine",
            fetched_at=state.timestamp,
            confidence="low",
            is_sandbox=state.provenance.is_sandbox,
        )
        recommendations: list[Recommendation] = []
        for service in state.all_services.values():
            recommendations.append(Recommendation(
                recommendation_id=str(uuid.uuid4()),
                workload_id=service.service_id,
                action_type=ActionType.KEEP.value,
                timestamp=state.timestamp,
                provenance=prov,
                binding_constraint=assessment.binding_constraint,
                confidence=assessment.confidence,
                sla_status="unknown",
                rationale=(
                    f"KEEP — classifier confidence {assessment.confidence:.2f} "
                    f"below floor {self.classifier.config.confidence_floor:.2f}. "
                    f"Missing signals: {assessment.missing_signals}."
                ),
                is_noop=True,
                implementation_mode=self.implementation_mode,
            ))
        return recommendations
