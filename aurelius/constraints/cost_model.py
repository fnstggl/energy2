"""Migration cost/risk model for Aurelius constraint-aware orchestration.

Phase 8 (corrected): before any recommendation is emitted, estimate the total
cost impact including hidden operational penalties. Risk is estimated from
**first principles on observed/predicted state** — NOT from a static
workload-class multiplier.

Design rules
------------
- Risk is STATE-CONDITIONED. There is no ``risk = base_risk * workload_multiplier``.
  The five risk families below are each derived from telemetry / policy / action:
    1. SLA headroom        — predicted metrics vs hard SLA bounds (+ active binding)
    2. Workload/runtime     — request rate, active sequences, cache affinity, churn
    3. Destination state    — spare capacity, hot/throttling, dest latency/queue, topology, distance
    4. Action-specific      — cold-start, cache warmup, lost batching, topology degradation, rollback
    5. Telemetry confidence — missing/stale metrics, sandbox provenance, classifier confidence
- Workload priority (``priority_tier``/``is_latency_sensitive``) is INFORMATIONAL.
  It never multiplies risk on its own. Conservatism for critical workloads enters
  ONLY through their explicit SLA policy (tighter bounds → less headroom → more
  state-conditioned risk) and through the telemetry-uncertainty buffer.
- A recommendation with net_expected_savings <= 0 must produce a KEEP (no-op).
- Hard SLA constraints (predicted breach of a hard bound, or migration_allowed=false)
  ALWAYS block, regardless of expected savings.
- Missing telemetry increases the uncertainty buffer and can force KEEP.
- The model never touches KV cache internals, NCCL, or memory allocators.
- MigrationGovernor prevents trigger-happy optimization through cooldown,
  per-workload history, and cluster-level migration-rate limits.

This module is read-only over ClusterState. It produces cost estimates and
decisions but does NOT execute migrations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..sla.actions import MIGRATION_ACTIONS, ActionType
from ..sla.schema import SLAPolicy
from ..sla.telemetry import RegionContext, WorkloadState
from ..state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
    Recommendation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _combine_or(values: list[float]) -> float:
    """Saturating OR-combination of independent risk fractions in [0, 1].

    Returns 1 - prod(1 - v). Higher individual risks compound but never exceed 1.
    """
    prod = 1.0
    for v in values:
        prod *= (1.0 - _clamp01(v))
    return 1.0 - prod


# ---------------------------------------------------------------------------
# Cost model configuration
# ---------------------------------------------------------------------------

@dataclass
class CostModelConfig:
    """Configurable parameters for the state-conditioned migration cost/risk estimator.

    All values are HEURISTIC — derived from engineering judgment, not from
    calibrated production telemetry. Tune against real outcomes before
    using in production savings claims.

    NOTE: There are intentionally NO ``critical_workload_risk_multiplier`` /
    ``batch_workload_risk_multiplier`` knobs. Static label-conditioned
    multipliers were removed in the Phase 8/9 risk-model correction because they
    are unsafe (they alone could permit or block an action irrespective of
    actual SLA headroom and destination health). Conservatism now flows from the
    SLA policy, measured headroom, and the telemetry-uncertainty buffer.
    """

    # --- Informational physical penalty magnitudes (reported, not label-scaled) ---
    cold_start_p99_penalty_ms: float = 2000.0     # HEURISTIC: p99 inflation during warmup
    cache_warmup_hit_rate_loss: float = 0.40      # HEURISTIC: fraction of prefix-cache hit rate lost during warmup
    queue_instability_penalty_ms: float = 300.0   # HEURISTIC: extra queue wait during migration
    topology_degradation_fraction: float = 0.30   # HEURISTIC: $ weight of topology degradation

    # --- State-conditioned risk weights (savings-equivalent units) ---
    sla_headroom_risk_weight: float = 8.0         # HEURISTIC: dominant — SLA headroom consumption
    destination_risk_weight: float = 5.0          # HEURISTIC: hot/full/distant destination
    action_risk_weight: float = 4.0               # HEURISTIC: cold-start / cache / churn / topology
    uncertainty_risk_weight: float = 4.0          # HEURISTIC: missing/stale/sandbox/low-confidence telemetry
    thermal_risk_weight: float = 3.0              # HEURISTIC: consolidation into a thermal-bound region

    # --- Headroom / destination / churn thresholds ---
    sla_headroom_safe_fraction: float = 0.5       # HEURISTIC: >50% headroom ⇒ negligible SLA risk
    dest_low_spare_capacity_pct: float = 20.0     # HEURISTIC: spare below this ⇒ destination capacity risk
    cold_start_saturation_ms: float = 5000.0      # HEURISTIC: cold-start cost saturates here
    batching_ref_active_seqs: float = 32.0        # HEURISTIC: active sequences for full batching-loss risk
    churn_saturation_count: int = 3               # HEURISTIC: migrations/hour at which churn risk saturates
    network_distance_ref_ms: float = 200.0        # HEURISTIC: cross-region RTT for full network risk
    max_acceptable_age_s: float = 300.0           # HEURISTIC: telemetry older than this is stale

    # --- Active binding constraint as a state proxy for SLA headroom ---
    # When explicit headroom telemetry is absent, an actively-binding SLA-risk
    # constraint is itself evidence that the workload is near its SLA. These are
    # state signals (classifier output), NOT workload labels.
    binding_sla_risk_proxy: dict = field(default_factory=lambda: {
        ConstraintType.LATENCY: 0.90,        # HEURISTIC: latency-bound now ⇒ little tail headroom
        ConstraintType.QUEUE: 0.60,          # HEURISTIC
        ConstraintType.MEMORY: 0.50,         # HEURISTIC: KV/HBM pressure
        ConstraintType.THERMAL: 0.40,        # HEURISTIC: throttle → latency spike
        ConstraintType.COMMUNICATION: 0.30,  # HEURISTIC
        ConstraintType.TOPOLOGY: 0.10,       # HEURISTIC
        ConstraintType.ENERGY: 0.0,          # cost constraint, not SLA-risk
        ConstraintType.UTILIZATION: 0.0,
        ConstraintType.NONE: 0.0,
    }, repr=False)

    # --- Decision thresholds ---
    min_net_savings_threshold: float = 0.0        # HEURISTIC: 0 = break-even; tune upward for conservatism
    min_savings_improvement_fraction: float = 0.0  # HEURISTIC
    min_confidence_to_act: float = 0.15           # HEURISTIC: below this ⇒ KEEP

    # --- Hysteresis / rate limiting ---
    min_migration_interval_s: float = 300.0       # HEURISTIC: 5 min
    max_migrations_per_workload_per_hour: int = 2  # HEURISTIC
    max_cluster_migrations_per_minute: int = 3    # HEURISTIC
    sla_violation_cooldown_s: float = 600.0       # HEURISTIC: 10 min


# ---------------------------------------------------------------------------
# State-conditioned risk inputs
# ---------------------------------------------------------------------------

@dataclass
class RiskInputs:
    """State-conditioned inputs for first-principles migration-risk estimation.

    Every field is Optional. A ``None`` field is treated as MISSING — it widens
    the telemetry-uncertainty buffer — and is NEVER silently coerced to a safe
    zero. Risk is conditioned on these observed/predicted states rather than on
    a static workload-class multiplier.

    The model reuses existing canonical types so the engine does not duplicate
    data carriers:
      * ``sla_policy``       — hard SLA bounds for headroom math (and migration policy)
      * ``current_state``    — observed runtime state of the workload (source)
      * ``predicted_state``  — predicted post-action state (e.g. HeuristicPredictor)
      * ``dest_context``     — destination region health for a migration
    """

    sla_policy: Optional[SLAPolicy] = None
    current_state: Optional[WorkloadState] = None
    predicted_state: Optional[WorkloadState] = None
    dest_context: Optional[RegionContext] = None

    # Workload/runtime extras not carried by WorkloadState
    prefix_cache_hit_rate: Optional[float] = None   # 0..1 — cache affinity at source
    kv_cache_usage: Optional[float] = None           # 0..1 — KV/HBM pressure at source
    requests_running: Optional[float] = None         # active sequences (batching disruption)
    requests_waiting: Optional[float] = None         # queue depth

    # Destination memory pressure (RegionContext does not carry it)
    dest_memory_pressure: Optional[float] = None     # 0..1

    # Telemetry staleness for the source workload snapshot
    sample_age_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Per-action cost/risk estimate
# ---------------------------------------------------------------------------

@dataclass
class MigrationCostEstimate:
    """Per-candidate action cost/risk breakdown.

    All monetary fields are in the same units as the gross-savings input
    (savings-equivalent). ``net_expected_savings > 0`` means the action is
    estimated to save money/resources after all state-conditioned penalties.
    ``net_expected_savings <= 0`` (or ``hard_sla_block``) means the action should
    be a no-op (KEEP).
    """
    workload_id: str
    action_type: str
    timestamp: datetime

    # Raw savings before penalties (from energy/cost model)
    gross_energy_savings: Optional[float] = None       # savings-equivalent
    gross_compute_savings: Optional[float] = None      # cost units

    # Informational physical penalties (reported; folded into risk buckets below)
    cold_start_penalty_ms: float = 0.0
    cache_warmup_penalty_ms: float = 0.0
    queue_instability_penalty_ms: float = 0.0
    lost_batching_efficiency_pct: float = 0.0
    network_transfer_cost: float = 0.0
    topology_degradation_score: float = 0.0    # [0, 1]; higher = worse
    failure_retry_penalty: float = 0.0          # rollback/failure probability [0, 1]

    # State-conditioned risk buckets (savings-equivalent; these sum to total_penalty)
    sla_risk_penalty: float = 0.0               # SLA headroom family
    destination_risk_penalty: float = 0.0       # destination-state family
    action_risk_penalty: float = 0.0            # action-specific family
    uncertainty_penalty: float = 0.0            # telemetry-confidence family
    thermal_penalty: float = 0.0                # consolidation-into-thermal family

    # Composite
    total_penalty: float = 0.0
    net_expected_savings: Optional[float] = None
    confidence: float = 0.0

    # SLA / governance gates
    hard_sla_block: bool = False
    blocked_by_cooldown: bool = False
    blocked_reason: Optional[str] = None

    # Explainability: which state factors drove the risk
    risk_factors: dict[str, float] = field(default_factory=dict)
    dominant_risk_factors: list[str] = field(default_factory=list)
    sla_headroom_fraction: Optional[float] = None   # worst SLA headroom across dims
    missing_signals: list[str] = field(default_factory=list)

    explanation: str = ""

    def is_viable(self) -> bool:
        """True if the action passes the cost/benefit threshold and no hard gate fired."""
        if self.blocked_by_cooldown or self.hard_sla_block:
            return False
        if self.net_expected_savings is None:
            return False
        return self.net_expected_savings > 0

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
            "gross_energy_savings": self.gross_energy_savings,
            "gross_compute_savings": self.gross_compute_savings,
            "cold_start_penalty_ms": self.cold_start_penalty_ms,
            "cache_warmup_penalty_ms": self.cache_warmup_penalty_ms,
            "queue_instability_penalty_ms": self.queue_instability_penalty_ms,
            "lost_batching_efficiency_pct": self.lost_batching_efficiency_pct,
            "network_transfer_cost": self.network_transfer_cost,
            "topology_degradation_score": self.topology_degradation_score,
            "failure_retry_penalty": self.failure_retry_penalty,
            "sla_risk_penalty": self.sla_risk_penalty,
            "destination_risk_penalty": self.destination_risk_penalty,
            "action_risk_penalty": self.action_risk_penalty,
            "uncertainty_penalty": self.uncertainty_penalty,
            "thermal_penalty": self.thermal_penalty,
            "total_penalty": self.total_penalty,
            "net_expected_savings": self.net_expected_savings,
            "confidence": self.confidence,
            "hard_sla_block": self.hard_sla_block,
            "blocked_by_cooldown": self.blocked_by_cooldown,
            "blocked_reason": self.blocked_reason,
            "risk_factors": dict(self.risk_factors),
            "dominant_risk_factors": list(self.dominant_risk_factors),
            "sla_headroom_fraction": self.sla_headroom_fraction,
            "missing_signals": list(self.missing_signals),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Migration governor (hysteresis / rate limiting)
# ---------------------------------------------------------------------------

@dataclass
class MigrationGovernor:
    """Prevents trigger-happy migration optimization.

    Tracks per-workload migration history and cluster-wide migration rate.
    All governance checks are in-memory; production deployments should
    persist migration history to the Postgres store (Phase 11/12).
    """

    config: CostModelConfig = field(default_factory=CostModelConfig)

    # workload_id → list of migration timestamps (most recent last)
    _workload_history: dict[str, list[datetime]] = field(default_factory=dict, repr=False)
    # cluster-wide migration timestamps
    _cluster_history: list[datetime] = field(default_factory=list, repr=False)
    # workload_id → timestamp of last SLA violation
    _sla_violation_times: dict[str, datetime] = field(default_factory=dict, repr=False)

    def record_migration(self, workload_id: str, ts: datetime) -> None:
        """Record that a migration was executed for the given workload."""
        if workload_id not in self._workload_history:
            self._workload_history[workload_id] = []
        self._workload_history[workload_id].append(ts)
        self._cluster_history.append(ts)

    def record_sla_violation(self, workload_id: str, ts: datetime) -> None:
        """Record a SLA violation so the governor enforces cooldown."""
        self._sla_violation_times[workload_id] = ts

    def check_allowed(self, workload_id: str, now: datetime) -> tuple[bool, Optional[str]]:
        """Return (allowed, reason_if_blocked).

        Checks:
        1. Per-workload minimum interval since last migration
        2. Per-workload rate limit (max N migrations per hour)
        3. Cluster-wide migration rate limit
        4. SLA violation cooldown for this workload
        """
        cfg = self.config

        # SLA violation cooldown
        if workload_id in self._sla_violation_times:
            last_violation = self._sla_violation_times[workload_id]
            age_s = (now - last_violation).total_seconds()
            if age_s < cfg.sla_violation_cooldown_s:
                remaining = cfg.sla_violation_cooldown_s - age_s
                return False, f"SLA violation cooldown: {remaining:.0f}s remaining"

        # Per-workload minimum interval
        wl_history = self._workload_history.get(workload_id, [])
        if wl_history:
            last_migration = wl_history[-1]
            age_s = (now - last_migration).total_seconds()
            if age_s < cfg.min_migration_interval_s:
                remaining = cfg.min_migration_interval_s - age_s
                return False, f"Minimum migration interval: {remaining:.0f}s remaining"

        # Per-workload hourly rate
        one_hour_ago = now - timedelta(hours=1)
        recent_wl = [t for t in wl_history if t > one_hour_ago]
        if len(recent_wl) >= cfg.max_migrations_per_workload_per_hour:
            return False, (
                f"Workload migration rate limit: {len(recent_wl)} migrations in last hour "
                f"(max {cfg.max_migrations_per_workload_per_hour})"
            )

        # Cluster-wide rate limit (per minute)
        one_minute_ago = now - timedelta(minutes=1)
        recent_cluster = [t for t in self._cluster_history if t > one_minute_ago]
        if len(recent_cluster) >= cfg.max_cluster_migrations_per_minute:
            return False, (
                f"Cluster migration rate limit: {len(recent_cluster)} migrations in last minute "
                f"(max {cfg.max_cluster_migrations_per_minute})"
            )

        return True, None

    def reset(self) -> None:
        """Clear all history (e.g. for a new simulation run)."""
        self._workload_history.clear()
        self._cluster_history.clear()
        self._sla_violation_times.clear()


# ---------------------------------------------------------------------------
# Migration cost model
# ---------------------------------------------------------------------------

class MigrationCostModel:
    """Estimates total cost impact of a candidate action from observed state.

    Conservative, state-conditioned heuristics are used when no trained predictor
    is available. The model never trains on or returns real customer data.

    Usage::

        model = MigrationCostModel(config=CostModelConfig())
        estimate = model.estimate(
            workload_id="llm-service-1",
            action_type=ActionType.MIGRATE.value,
            assessment=constraint_assessment,
            state=cluster_state,
            gross_savings=5.0,
            risk_inputs=RiskInputs(sla_policy=policy, current_state=ws,
                                   predicted_state=pred, dest_context=dest),
        )
        if estimate.is_viable():
            ...  # emit recommendation
    """

    def __init__(
        self,
        config: Optional[CostModelConfig] = None,
        governor: Optional[MigrationGovernor] = None,
    ) -> None:
        self.config = config or CostModelConfig()
        self.governor = governor or MigrationGovernor(config=self.config)

    # ------------------------------------------------------------------
    # Risk-family computations (each returns savings-equivalent penalty +
    # a dict of named factor contributions for the explanation)
    # ------------------------------------------------------------------

    def _sla_headroom_risk(
        self,
        is_migration: bool,
        is_noop: bool,
        assessment: ConstraintAssessment,
        ri: RiskInputs,
    ) -> tuple[float, float, Optional[float], bool, dict[str, float], list[str]]:
        """SLA headroom family.

        Returns (penalty, raw_risk_frac, worst_headroom, hard_block, factors, missing).
        Combines explicit headroom (predicted metrics vs hard SLA bounds) with the
        active-binding-constraint proxy (an actively-binding SLA-risk constraint is
        state evidence of low headroom). Uses the WORSE of the two.
        """
        cfg = self.config
        factors: dict[str, float] = {}
        missing: list[str] = []
        if is_noop:
            return 0.0, 0.0, None, False, factors, missing

        # ---- Explicit headroom from predicted/current metrics vs hard bounds ----
        worst_headroom: Optional[float] = None
        hard_block = False
        hard = ri.sla_policy.hard if (ri.sla_policy and ri.sla_policy.enabled) else None
        # Prefer predicted post-action state; fall back to current observed state.
        st = ri.predicted_state or ri.current_state

        if hard is not None and st is not None:
            checks: list[tuple[str, Optional[float], Optional[float]]] = [
                ("p95", st.p95_latency_ms, hard.max_p95_latency_ms),
                ("p99", st.p99_latency_ms, hard.max_p99_latency_ms),
                ("queue", st.queue_wait_ms, hard.max_queue_wait_ms),
                ("error_rate", st.error_rate_pct, hard.max_error_rate_pct),
            ]
            for name, value, limit in checks:
                if limit is None:
                    continue
                if value is None:
                    missing.append(f"sla.{name}")
                    continue
                if limit <= 0:
                    continue
                headroom = 1.0 - value / limit  # 1=far below, 0=at limit, <0=over
                worst_headroom = headroom if worst_headroom is None else min(worst_headroom, headroom)
                if headroom < 0:
                    hard_block = True
            # Availability floor (min-type)
            if hard.min_availability_pct is not None:
                if st.availability_pct is None:
                    missing.append("sla.availability")
                else:
                    gap = 100.0 - hard.min_availability_pct
                    headroom = 1.0 if gap <= 0 else (st.availability_pct - hard.min_availability_pct) / gap
                    worst_headroom = headroom if worst_headroom is None else min(worst_headroom, headroom)
                    if st.availability_pct < hard.min_availability_pct:
                        hard_block = True
            # Capacity buffer floor (min-type) — evaluated at destination/predicted
            if hard.required_capacity_buffer_pct is not None:
                buf = st.capacity_buffer_pct
                if buf is None:
                    missing.append("sla.capacity_buffer")
                elif hard.required_capacity_buffer_pct > 0:
                    headroom = (buf - hard.required_capacity_buffer_pct) / hard.required_capacity_buffer_pct
                    worst_headroom = headroom if worst_headroom is None else min(worst_headroom, headroom)
                    if buf < hard.required_capacity_buffer_pct:
                        hard_block = True
        else:
            if ri.sla_policy is None:
                missing.append("sla.policy")
            if st is None:
                missing.append("sla.workload_state")

        explicit_risk = 0.0
        if worst_headroom is not None:
            safe = cfg.sla_headroom_safe_fraction
            if worst_headroom >= safe:
                explicit_risk = 0.0
            elif worst_headroom <= 0.0:
                explicit_risk = 1.0
            else:
                explicit_risk = (safe - worst_headroom) / safe
            factors["sla_headroom"] = explicit_risk

        # ---- Active-binding proxy (state evidence of low headroom) ----
        # Applies to MIGRATION actions only: disrupting a workload that is
        # actively bound by an SLA-risk constraint (latency/queue/memory/thermal)
        # is risky. In-place remediations (SCALE/SPREAD/CONSOLIDATE) are NOT
        # penalized here — they are the intended fix for such a constraint, and
        # their own SLA impact is captured by the explicit headroom check above.
        binding = assessment.binding_constraint
        proxy_risk = 0.0
        if is_migration and binding is not None:
            base = cfg.binding_sla_risk_proxy.get(binding, 0.0)
            # Scale by the binding score (intensity), defaulting to full weight.
            score = assessment.scores.get(binding, 1.0)
            proxy_risk = _clamp01(base * score)
            if proxy_risk > 0:
                factors["sla_active_binding"] = proxy_risk

        risk_frac = max(explicit_risk, proxy_risk)
        penalty = cfg.sla_headroom_risk_weight * risk_frac
        return penalty, risk_frac, worst_headroom, hard_block, factors, missing

    def _destination_risk(
        self,
        is_migration: bool,
        ri: RiskInputs,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Destination-state family. Only meaningful for migrations."""
        cfg = self.config
        factors: dict[str, float] = {}
        missing: list[str] = []
        if not is_migration:
            return 0.0, factors, missing

        dest = ri.dest_context
        if dest is None:
            missing.append("dest.context")
            return 0.0, factors, missing

        sub: list[float] = []

        # Spare capacity at destination
        if dest.spare_capacity_pct is None:
            missing.append("dest.spare_capacity")
        else:
            low = cfg.dest_low_spare_capacity_pct
            cap_risk = _clamp01((low - dest.spare_capacity_pct) / low) if low > 0 else 0.0
            if cap_risk > 0:
                factors["dest_low_capacity"] = cap_risk
            sub.append(cap_risk)

        # Thermal stress / throttling at destination
        thermal_risk = 1.0 if (dest.thermally_stressed or dest.throttling) else 0.0
        if thermal_risk > 0:
            factors["dest_thermal"] = thermal_risk
        sub.append(thermal_risk)

        # Destination already slower than source (predicted floor vs current)
        cur = ri.current_state
        if dest.baseline_p99_latency_ms is not None and cur is not None and cur.p99_latency_ms:
            if dest.baseline_p99_latency_ms > cur.p99_latency_ms:
                lat_risk = _clamp01(
                    (dest.baseline_p99_latency_ms - cur.p99_latency_ms) / cur.p99_latency_ms
                )
                if lat_risk > 0:
                    factors["dest_higher_latency"] = lat_risk
                sub.append(lat_risk)

        # Network distance (added serving RTT)
        if dest.network_rtt_ms and cfg.network_distance_ref_ms > 0:
            net_risk = _clamp01(dest.network_rtt_ms / cfg.network_distance_ref_ms)
            if net_risk > 0:
                factors["dest_network_distance"] = net_risk
            sub.append(net_risk)

        # Destination memory pressure (if exposed)
        if ri.dest_memory_pressure is not None:
            mem_risk = _clamp01(ri.dest_memory_pressure)
            if mem_risk > 0:
                factors["dest_memory_pressure"] = mem_risk
            sub.append(mem_risk)

        risk_frac = _combine_or(sub)
        return cfg.destination_risk_weight * risk_frac, factors, missing

    def _action_risk(
        self,
        is_migration: bool,
        is_noop: bool,
        action_type: str,
        cur_topo: Optional[float],
        tgt_topo: Optional[float],
        ri: RiskInputs,
        dest_risk_frac: float,
    ) -> tuple[float, float, dict[str, float], list[str]]:
        """Action-specific family. Returns (penalty, rollback_prob, factors, missing)."""
        cfg = self.config
        factors: dict[str, float] = {}
        missing: list[str] = []
        if is_noop:
            return 0.0, 0.0, factors, missing

        sub: list[float] = []

        # Topology degradation (applies to migration AND consolidation placement)
        topo_deg = 0.0
        if cur_topo is not None and tgt_topo is not None:
            topo_deg = max(0.0, cur_topo - tgt_topo)
            if topo_deg > 0:
                factors["topology_degradation"] = topo_deg
            sub.append(topo_deg)
        elif is_migration:
            missing.append("action.topology_score")

        if is_migration:
            # Cold-start cost (normalized)
            cs_frac = _clamp01(cfg.cold_start_p99_penalty_ms / cfg.cold_start_saturation_ms)
            if cs_frac > 0:
                factors["cold_start"] = cs_frac
            sub.append(cs_frac)

            # Cache warmup — scaled by cache affinity (more hits ⇒ more to lose).
            if ri.prefix_cache_hit_rate is None:
                affinity = 0.5  # conservative default; also recorded as missing telemetry
                missing.append("action.prefix_cache_hit_rate")
            else:
                affinity = _clamp01(ri.prefix_cache_hit_rate)
            cache_frac = _clamp01(affinity * cfg.cache_warmup_hit_rate_loss)
            if cache_frac > 0:
                factors["cache_warmup"] = cache_frac
            sub.append(cache_frac)

            # Lost batching efficiency — scaled by active in-flight sequences.
            if ri.requests_running is None:
                missing.append("action.requests_running")
                batch_frac = 0.15  # conservative default
            else:
                batch_frac = _clamp01(ri.requests_running / cfg.batching_ref_active_seqs)
            if batch_frac > 0:
                factors["lost_batching"] = batch_frac
            sub.append(batch_frac)

            # Recent migration churn
            churn_count = ri.current_state.migration_count_last_hour if ri.current_state else 0
            churn_frac = _clamp01(churn_count / cfg.churn_saturation_count) if cfg.churn_saturation_count else 0.0
            if churn_frac > 0:
                factors["migration_churn"] = churn_frac
            sub.append(churn_frac)

        # Rollback / failure probability rises with destination risk, churn, topology loss.
        churn_for_rollback = factors.get("migration_churn", 0.0)
        rollback = _clamp01(
            (0.10 if is_migration else 0.0)
            + 0.30 * dest_risk_frac
            + 0.20 * churn_for_rollback
            + 0.20 * topo_deg
        )
        if rollback > 0:
            factors["rollback_probability"] = rollback
        sub.append(rollback)

        risk_frac = _combine_or(sub)
        return cfg.action_risk_weight * risk_frac, rollback, factors, missing

    def _uncertainty_risk(
        self,
        assessment: ConstraintAssessment,
        ri: RiskInputs,
        is_sandbox: bool,
        prior_missing: list[str],
    ) -> tuple[float, dict[str, float]]:
        """Telemetry-confidence family. Missing/stale/sandbox/low-confidence widen the buffer."""
        cfg = self.config
        factors: dict[str, float] = {}

        # Missing telemetry: count the distinct missing signals gathered so far.
        missing_unique = sorted(set(prior_missing))
        # Normalize against a rough expectation of key signals for a confident call.
        expected = 6.0  # HEURISTIC: ~6 key signals expected when fully observed
        missing_frac = _clamp01(len(missing_unique) / expected)
        if missing_frac > 0:
            factors["missing_telemetry"] = missing_frac

        # Stale telemetry (WorkloadState carries no age; rely on RiskInputs.sample_age_s)
        stale_frac = 0.0
        age = ri.sample_age_s
        if age is not None and cfg.max_acceptable_age_s > 0:
            stale_frac = _clamp01((age - cfg.max_acceptable_age_s) / cfg.max_acceptable_age_s)
            if stale_frac > 0:
                factors["stale_telemetry"] = stale_frac

        # Low classifier confidence
        conf_frac = _clamp01(1.0 - assessment.confidence)
        if conf_frac > 0:
            factors["low_classifier_confidence"] = conf_frac

        # Sandbox provenance (mild — sandbox excludes economic claims, not recommendations)
        sandbox_frac = 0.10 if is_sandbox else 0.0
        if sandbox_frac > 0:
            factors["sandbox_provenance"] = sandbox_frac

        combined = _clamp01(
            0.50 * missing_frac
            + 0.30 * conf_frac
            + 0.20 * stale_frac
            + sandbox_frac
        )
        return cfg.uncertainty_risk_weight * combined, factors

    # ------------------------------------------------------------------
    # Main estimate
    # ------------------------------------------------------------------

    def estimate(
        self,
        workload_id: str,
        action_type: str,
        assessment: ConstraintAssessment,
        state: ClusterState,
        gross_savings: Optional[float] = None,
        is_latency_sensitive: bool = False,
        priority_tier: str = "standard",
        current_topology_score: Optional[float] = None,
        target_topology_score: Optional[float] = None,
        now: Optional[datetime] = None,
        risk_inputs: Optional[RiskInputs] = None,
    ) -> MigrationCostEstimate:
        """Produce a state-conditioned cost/risk estimate for a candidate action.

        Parameters
        ----------
        workload_id: Workload being considered for the action.
        action_type: The action type string (ActionType.value).
        assessment: Current ConstraintAssessment from the classifier.
        state: Current ClusterState snapshot.
        gross_savings: Gross expected savings before penalties (savings-equivalent).
            None means savings are unknown; estimate will have low confidence.
        is_latency_sensitive: INFORMATIONAL ONLY. Recorded for observability; it does
            not multiply risk. Conservatism comes from the SLA policy + headroom.
        priority_tier: INFORMATIONAL ONLY (same as above). Never a risk multiplier.
        current_topology_score: Topology score for current placement (0-1, higher=better).
        target_topology_score: Topology score for proposed placement (0-1).
        now: Override current time (useful for testing).
        risk_inputs: State-conditioned inputs (SLA policy, current/predicted workload
            state, destination context, cache affinity, etc.). Missing fields widen
            the uncertainty buffer.
        """
        cfg = self.config
        ts = state.timestamp
        if now is None:
            now = datetime.now(tz=timezone.utc)
        ri = risk_inputs or RiskInputs()

        is_migration = action_type in {a.value for a in MIGRATION_ACTIONS}
        is_noop = action_type == ActionType.KEEP.value
        is_sandbox = assessment.provenance.is_sandbox

        # 1. Governance check (cooldown, rate limits) — applies to migrations only.
        if is_migration:
            allowed, block_reason = self.governor.check_allowed(workload_id, now)
            if not allowed:
                return MigrationCostEstimate(
                    workload_id=workload_id,
                    action_type=action_type,
                    timestamp=ts,
                    gross_energy_savings=gross_savings,
                    net_expected_savings=None,
                    confidence=0.0,
                    blocked_by_cooldown=True,
                    blocked_reason=block_reason,
                    explanation=f"Blocked by migration governor: {block_reason}",
                )

        # 2. Hard migration-policy gate (SLA migration_allowed=false ⇒ always block).
        hard_block = False
        hard_block_reason: Optional[str] = None
        if is_migration and ri.sla_policy and ri.sla_policy.enabled:
            if ri.sla_policy.hard.migration_allowed is False:
                hard_block = True
                hard_block_reason = "SLA policy forbids migration (migration_allowed=false)"

        # 3. State-conditioned risk families
        sla_pen, sla_frac, worst_headroom, sla_hard, sla_factors, sla_missing = self._sla_headroom_risk(
            is_migration, is_noop, assessment, ri
        )
        if sla_hard:
            hard_block = True
            hard_block_reason = hard_block_reason or "predicted state breaches a hard SLA bound"

        dest_pen, dest_factors, dest_missing = self._destination_risk(is_migration, ri)
        dest_risk_frac = dest_pen / cfg.destination_risk_weight if cfg.destination_risk_weight else 0.0

        action_pen, rollback_prob, action_factors, action_missing = self._action_risk(
            is_migration, is_noop, action_type,
            current_topology_score, target_topology_score, ri, dest_risk_frac,
        )

        # Thermal family: consolidating into a thermal-bound region is especially harmful.
        thermal_pen = 0.0
        thermal_factors: dict[str, float] = {}
        binding = assessment.binding_constraint
        if action_type == ActionType.CONSOLIDATE.value and binding == ConstraintType.THERMAL:
            thermal_score = assessment.scores.get(ConstraintType.THERMAL, 1.0)
            thermal_pen = cfg.thermal_risk_weight * _clamp01(thermal_score)
            thermal_factors["consolidate_into_thermal"] = _clamp01(thermal_score)

        # 4. Telemetry-uncertainty buffer (depends on what was missing above).
        prior_missing = sla_missing + dest_missing + action_missing + list(assessment.missing_signals)
        unc_pen, unc_factors = self._uncertainty_risk(assessment, ri, is_sandbox, prior_missing)

        # 5. Aggregate total penalty (savings-equivalent).
        total_penalty = sla_pen + dest_pen + action_pen + unc_pen + thermal_pen

        # 6. Informational physical penalty fields (reported; not double-counted in total).
        cold_start_ms = cfg.cold_start_p99_penalty_ms if is_migration else 0.0
        affinity = ri.prefix_cache_hit_rate if ri.prefix_cache_hit_rate is not None else 0.5
        cache_warmup_ms = (
            cfg.cache_warmup_hit_rate_loss * cfg.cold_start_p99_penalty_ms * _clamp01(affinity)
            if is_migration else 0.0
        )
        queue_ms = cfg.queue_instability_penalty_ms if is_migration else 0.0
        topo_deg_score = 0.0
        if current_topology_score is not None and target_topology_score is not None:
            topo_deg_score = max(0.0, current_topology_score - target_topology_score)
        lost_batching_pct = action_factors.get("lost_batching", 0.0) * 100.0
        network_cost = topo_deg_score * (gross_savings or 0.0) * cfg.topology_degradation_fraction

        # 7. Net expected savings + confidence.
        gross = gross_savings if gross_savings is not None else 0.0
        if gross_savings is None:
            net_value: Optional[float] = None
            conf = 0.2  # HEURISTIC: low confidence when gross savings unknown
        else:
            net_value = gross - total_penalty
            # Confidence reflects classifier confidence AND telemetry completeness.
            unc_frac = unc_pen / cfg.uncertainty_risk_weight if cfg.uncertainty_risk_weight else 0.0
            conf = _clamp01(assessment.confidence * (1.0 - 0.5 * unc_frac))

        # 8. Assemble risk-factor explanation.
        risk_factors: dict[str, float] = {}
        for d in (sla_factors, dest_factors, action_factors, unc_factors, thermal_factors):
            risk_factors.update(d)
        # Dominant factors: top 3 by contribution.
        dominant = [
            name for name, contrib in sorted(risk_factors.items(), key=lambda kv: -kv[1])[:3]
            if contrib > 0.0
        ]
        missing_signals = sorted(set(sla_missing + dest_missing + action_missing))

        explanation = self._build_explanation(
            workload_id=workload_id,
            action_type=action_type,
            gross=gross if gross_savings is not None else None,
            total_penalty=total_penalty,
            net=net_value,
            sla_pen=sla_pen, dest_pen=dest_pen, action_pen=action_pen,
            unc_pen=unc_pen, thermal_pen=thermal_pen,
            worst_headroom=worst_headroom,
            dominant=dominant,
            hard_block=hard_block, hard_block_reason=hard_block_reason,
            missing_signals=missing_signals,
            priority_tier=priority_tier, is_latency_sensitive=is_latency_sensitive,
            binding=binding,
        )

        return MigrationCostEstimate(
            workload_id=workload_id,
            action_type=action_type,
            timestamp=ts,
            gross_energy_savings=gross_savings,
            gross_compute_savings=None,
            cold_start_penalty_ms=cold_start_ms,
            cache_warmup_penalty_ms=cache_warmup_ms,
            queue_instability_penalty_ms=queue_ms,
            lost_batching_efficiency_pct=lost_batching_pct,
            network_transfer_cost=network_cost,
            topology_degradation_score=topo_deg_score,
            failure_retry_penalty=rollback_prob,
            sla_risk_penalty=sla_pen,
            destination_risk_penalty=dest_pen,
            action_risk_penalty=action_pen,
            uncertainty_penalty=unc_pen,
            thermal_penalty=thermal_pen,
            total_penalty=total_penalty,
            net_expected_savings=net_value,
            confidence=_clamp01(conf),
            hard_sla_block=hard_block,
            blocked_by_cooldown=False,
            blocked_reason=hard_block_reason,
            risk_factors=risk_factors,
            dominant_risk_factors=dominant,
            sla_headroom_fraction=worst_headroom,
            missing_signals=missing_signals,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(
        workload_id: str,
        action_type: str,
        gross: Optional[float],
        total_penalty: float,
        net: Optional[float],
        sla_pen: float,
        dest_pen: float,
        action_pen: float,
        unc_pen: float,
        thermal_pen: float,
        worst_headroom: Optional[float],
        dominant: list[str],
        hard_block: bool,
        hard_block_reason: Optional[str],
        missing_signals: list[str],
        priority_tier: str,
        is_latency_sensitive: bool,
        binding: Optional[ConstraintType],
    ) -> str:
        parts: list[str] = [f"Action: {action_type} for workload {workload_id}."]
        if gross is not None:
            parts.append(
                f"Gross={gross:.3f}, penalty={total_penalty:.3f} "
                f"(sla={sla_pen:.2f} dest={dest_pen:.2f} action={action_pen:.2f} "
                f"uncertainty={unc_pen:.2f} thermal={thermal_pen:.2f}), net={net:.3f}."
            )
        if binding is not None:
            parts.append(f"Active binding constraint: {binding.value}.")
        if worst_headroom is not None:
            parts.append(f"Worst SLA headroom fraction: {worst_headroom:.2f}.")
        if dominant:
            parts.append("Dominant risk factors: " + ", ".join(dominant) + ".")
        if missing_signals:
            shown = missing_signals[:4]
            extra = len(missing_signals) - len(shown)
            parts.append(
                "Missing telemetry (↑uncertainty): "
                + ", ".join(shown) + (f" +{extra} more" if extra else "") + "."
            )
        if hard_block:
            parts.append(f"HARD BLOCK: {hard_block_reason}.")
        elif net is not None and net <= 0:
            parts.append("Net savings non-positive — KEEP recommended.")
        # Workload priority is recorded but is NOT a risk multiplier.
        parts.append(
            f"Workload tier={priority_tier} (latency_sensitive={is_latency_sensitive}); "
            f"informational only — risk is state-conditioned, not label-multiplied."
        )
        return " ".join(parts)

    def should_keep(
        self,
        estimate: MigrationCostEstimate,
        cfg: Optional[CostModelConfig] = None,
    ) -> tuple[bool, str]:
        """Return (should_keep, reason) — True when the action should be a no-op.

        A KEEP is recommended when:
        - a hard SLA constraint is breached (always blocks, regardless of savings)
        - blocked_by_cooldown / governor
        - net_expected_savings is None (unknown gross savings)
        - net_expected_savings <= min_net_savings_threshold
        - confidence is very low
        """
        if cfg is None:
            cfg = self.config

        if estimate.hard_sla_block:
            return True, estimate.blocked_reason or "hard SLA constraint breached"

        if estimate.blocked_by_cooldown:
            return True, estimate.blocked_reason or "migration governor blocked"

        if estimate.net_expected_savings is None:
            return True, "gross savings unknown; cannot estimate net benefit"

        if estimate.net_expected_savings <= cfg.min_net_savings_threshold:
            return True, (
                f"net savings {estimate.net_expected_savings:.3f} ≤ threshold "
                f"{cfg.min_net_savings_threshold:.3f} after state-conditioned penalties"
            )

        if estimate.confidence < cfg.min_confidence_to_act:
            return True, f"confidence {estimate.confidence:.3f} too low to recommend action"

        return False, "action passes state-conditioned cost/benefit threshold"

    def make_recommendation(
        self,
        workload_id: str,
        action_type: str,
        estimate: MigrationCostEstimate,
        assessment: ConstraintAssessment,
        provenance: Optional[Provenance] = None,
        recommendation_id: Optional[str] = None,
    ) -> Recommendation:
        """Convert a cost estimate into a Recommendation.

        Always emits in recommendation_only mode.
        """
        import uuid
        keep, keep_reason = self.should_keep(estimate)
        if keep:
            final_action = ActionType.KEEP.value
            sla_status = "blocked" if estimate.hard_sla_block else "unknown"
            is_noop = True
            rationale = f"KEEP: {keep_reason}. {estimate.explanation}"
        else:
            final_action = action_type
            sla_status = "satisfied"
            is_noop = False
            rationale = estimate.explanation

        prov = provenance or Provenance(
            source="migration-cost-model",
            fetched_at=estimate.timestamp,
            confidence=(
                "high" if estimate.confidence >= 0.7
                else "medium" if estimate.confidence >= 0.4
                else "low"
            ),
            is_sandbox=assessment.provenance.is_sandbox,
        )

        return Recommendation(
            recommendation_id=recommendation_id or str(uuid.uuid4()),
            workload_id=workload_id,
            action_type=final_action,
            timestamp=estimate.timestamp,
            provenance=prov,
            binding_constraint=assessment.binding_constraint,
            expected_effect={
                "gross_savings": estimate.gross_energy_savings or 0.0,
                "total_penalty": estimate.total_penalty,
                "sla_risk_penalty": estimate.sla_risk_penalty,
                "destination_risk_penalty": estimate.destination_risk_penalty,
                "action_risk_penalty": estimate.action_risk_penalty,
                "uncertainty_penalty": estimate.uncertainty_penalty,
            },
            confidence=estimate.confidence,
            sla_status=sla_status,
            migration_penalty=estimate.total_penalty,
            net_benefit=estimate.net_expected_savings,
            rationale=rationale,
            is_noop=is_noop,
            implementation_mode="recommendation_only",
        )
