"""Energy-arbitrage adapter — route the EXISTING energy engine's recommendations
through the constraint-aware SLA / KPI / risk gates.

ARCHITECTURE (binding — see the PR description and docs/ENERGY_SYSTEM_MAP.md):

    Existing energy engine
      (aurelius.optimization.scheduler.JobScheduler  +
       aurelius.backtesting.*  +  aurelius.forecasting.*)
            |
            v
      energy recommendation X        (region shift / defer / keep, per job)
            |
            v
      EnergyArbitrageAdapter          <-- THIS MODULE (thin wrapper, no energy logic)
            |
            v
      constraint-aware gates          (eligibility / destination / SLA / KPI)
            |
            v
      ACCEPT / REJECT / DEFER / MODIFY

The energy engine decides *what energy move should be attempted*. The adapter +
constraint-aware gates decide *whether that move is safe and KPI-positive to
execute*. The adapter is the canonical execution/safety layer for energy moves;
the energy engine remains the canonical energy decision-maker.

GUARDRAILS this module enforces on itself:
- It NEVER re-derives an energy decision. It consumes ``JobScheduler.solve()``
  output verbatim (via its public API) and only classifies/diffs it.
- It NEVER mutates the energy engine, its constants, or its forecasters.
- It NEVER mutates a real cluster — it emits decisions, it does not execute them.
- It produces the SAME energy recommendation the engine produced; it can only
  ACCEPT / REJECT / DEFER / MODIFY-EXECUTION-OF it, never substitute a
  fundamentally different energy target.

Primary KPI is the canonical one from docs/RESULTS.md:
``SLA-safe goodput per infrastructure dollar``. The adapter never folds a
revenue/value weight into it; SLA is a *filter on the numerator* (a job that
misses its deadline contributes zero goodput), never a subtraction term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from ..models import Job, OptimizationConfig, ScheduleDecision
from ..sla.actions import ActionType

# ---------------------------------------------------------------------------
# Workload-class policy (constraint-aware policy, NOT energy logic)
# ---------------------------------------------------------------------------

# Job.workload_type values that are latency-pinned interactive serving. These
# are blocked from energy region-shifts by default (§D.1) — they are not
# shiftable without breaking an interactive SLO.
_CRITICAL_INTERACTIVE_TYPES: frozenset[str] = frozenset({"realtime_inference"})

# Job.workload_type values that are flexible / batch / offline and therefore
# *eligible* for an energy shift when migration is allowed and the flexibility
# window / deadline permits it (§D.1).
_FLEXIBLE_TYPES: frozenset[str] = frozenset({
    "llm_batch_inference", "fine_tuning", "training",
    "data_processing", "scheduled_batch", "background_maintenance",
})


class EnergyCandidateAction(str, Enum):
    """Semantic label for the energy engine's recommendation for one job.

    These map onto the constraint-aware action vocabulary
    (``aurelius.sla.actions.ActionType``) via :meth:`to_sla_action_type`.
    """
    KEEP = "keep_current_placement"
    CHOOSE_CHEAPER_REGION = "choose_cheaper_region"
    SHIFT_BATCH_TO_CHEAPER_REGION = "shift_batch_to_cheaper_region"
    DEFER_FLEXIBLE_WORKLOAD = "defer_flexible_workload"

    def to_sla_action_type(self) -> ActionType:
        if self is EnergyCandidateAction.DEFER_FLEXIBLE_WORKLOAD:
            return ActionType.DEFER
        if self is EnergyCandidateAction.KEEP:
            return ActionType.KEEP
        # Both region-shift variants are CHOOSE_CHEAPER_REGION in the
        # constraint vocabulary (the simulator/engine treat them identically;
        # the batch variant is a reporting refinement only).
        return ActionType.CHOOSE_CHEAPER_REGION


class GateDecision(str, Enum):
    """Outcome of routing an energy candidate through the constraint gates."""
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    MODIFY = "modify"


# ---------------------------------------------------------------------------
# Destination safety context (read-only — supplied by the caller's telemetry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DestinationContext:
    """Read-only safety telemetry for a candidate destination region.

    The adapter does NOT fetch this — the caller supplies it from whatever
    telemetry source it trusts (simulator, live connectors, or a fixed
    canonical-backtest assumption). All fields optional; ``None`` means
    "unknown", which the destination-safety gate treats as unsafe (fail-closed)
    for the dimensions that require positive evidence.
    """
    region: str
    spare_capacity_pct: Optional[float] = None   # [0, 100]; low => full
    is_hot: bool = False                          # thermal hotspot at destination
    queue_p95_ms: Optional[float] = None          # destination queue pressure
    telemetry_confidence: str = "high"            # high | medium | low
    is_stale: bool = False                         # stale / missing telemetry
    topology_fit_ok: bool = True                   # destination fabric suits workload
    is_cold: bool = False                          # destination has no warm pool
    # True when the destination already holds a warm prefix/KV cache for this
    # workload (e.g. a replicated home), so the move does NOT pay a cold-route
    # penalty. Default False = cold cache at the destination.
    preserves_affinity: bool = False

    # HEURISTIC floors (constraint-aware policy). A destination with spare
    # capacity below this floor is "full"; queue p95 above this is "hot queue".
    CRITICAL_SPARE_PCT: float = 5.0
    HOT_QUEUE_P95_MS: float = 2000.0


# ---------------------------------------------------------------------------
# The energy engine's recommendation for one job (consumed verbatim)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExistingEnergyCandidate:
    """One energy recommendation produced by the EXISTING energy engine.

    Built by diffing the engine's optimized schedule against its own ASAP
    baseline (both from the engine's public API). The adapter never invents
    these fields — they come straight from the scheduler's output and the
    price data the scheduler was given.
    """
    job_id: str
    workload_type: str
    action: EnergyCandidateAction
    current_region: str            # baseline (ASAP) region
    recommended_region: str        # region the energy engine chose
    # Forecast / savings (from the energy engine + the price data it used).
    gross_savings_usd: float           # DA-planned baseline cost - DA-planned candidate cost
    gross_savings_pct: float
    da_price_current_mwh: Optional[float]
    da_price_target_mwh: Optional[float]
    rt_price_target_mwh: Optional[float]
    da_rt_basis_risk_usd: float        # expected (RT - DA) settlement risk on the candidate
    forecast_confidence: float         # [0, 1]; engine/forecaster confidence
    # Window / flexibility (from the Job).
    window_start: datetime           # energy engine's chosen start
    baseline_start: datetime         # ASAP baseline start (earliest_start)
    runtime_hours: float
    slack_hours: float
    deadline: datetime
    migration_allowed: bool
    migration_cost_hours: float
    latency_sensitive: bool
    gpu_count: int
    power_kw: float
    # Cache / topology sensitivity (constraint-aware risk signals).
    cache_hit_rate: Optional[float] = None   # high => region move destroys affinity
    cache_sensitive: bool = False
    topology_heavy: bool = False
    # Provenance of this candidate placement within the engine's ranked
    # alternatives (e.g. "engine_optimized", "current_price_only", "home").
    # Used by evaluate_best() to map the accepted alternative back to its
    # concrete placement. Never alters gating — purely a label.
    source: str = ""

    @property
    def is_region_move(self) -> bool:
        return self.recommended_region != self.current_region

    @property
    def migration_energy_usd(self) -> float:
        """Paid-but-no-useful-work energy during the migration warmup window.

        At the destination region's price (DA), for ``migration_cost_hours`` of
        power draw. This is the cost the engine's gross-savings figure does not
        include; the KPI gate must net it out.
        """
        if not self.is_region_move or self.migration_cost_hours <= 0:
            return 0.0
        price = self.da_price_target_mwh or self.da_price_current_mwh or 0.0
        return (price / 1000.0) * self.power_kw * self.migration_cost_hours

    @property
    def net_savings_usd(self) -> float:
        """Gross savings minus migration energy minus DA/RT basis risk."""
        return self.gross_savings_usd - self.migration_energy_usd - self.da_rt_basis_risk_usd


# ---------------------------------------------------------------------------
# The adapter's verdict on one candidate
# ---------------------------------------------------------------------------

@dataclass
class ConstraintAwareEnergyCandidate:
    """An energy candidate after routing through the constraint-aware gates.

    ``reasons`` holds STABLE reason CODES (no embedded numbers) so they can be
    histogrammed deterministically; ``reason_details`` holds the human-readable
    detail strings (which may carry numbers) for the per-candidate explanation.
    """
    candidate: ExistingEnergyCandidate
    decision: GateDecision
    reasons: list[str] = field(default_factory=list)
    reason_details: list[str] = field(default_factory=list)
    # KPI accounting (canonical KPI: SLA-safe goodput per infra dollar).
    baseline_goodput_per_dollar: float = 0.0
    candidate_goodput_per_dollar: float = 0.0
    prevents_deadline_miss: bool = False
    # Fraction of goodput value retained after a (possibly cold) move (Part C).
    cache_loss_factor: float = 1.0

    @property
    def accepted(self) -> bool:
        return self.decision in (GateDecision.ACCEPT, GateDecision.MODIFY)

    @property
    def kpi_delta(self) -> float:
        return self.candidate_goodput_per_dollar - self.baseline_goodput_per_dollar

    @property
    def applied_region(self) -> str:
        """Region that should actually be used after the gate verdict.

        ACCEPT/MODIFY -> the energy engine's recommended region.
        REJECT/DEFER  -> the safe current (baseline) region (move not executed).
        """
        if self.accepted:
            return self.candidate.recommended_region
        return self.candidate.current_region

    def explanation(self) -> dict:
        """Per-candidate explanation required by §D.5 + docs/RESULTS.md."""
        c = self.candidate
        return {
            "energy_candidate_id": c.job_id,
            "workload_type": c.workload_type,
            "energy_action": c.action.value,
            "sla_action_type": c.action.to_sla_action_type().value,
            "current_region": c.current_region,
            "recommended_region": c.recommended_region,
            "applied_region": self.applied_region,
            "gross_forecasted_energy_savings_usd": round(c.gross_savings_usd, 4),
            "gross_savings_pct": round(c.gross_savings_pct, 3),
            "migration_energy_usd": round(c.migration_energy_usd, 4),
            "da_rt_basis_risk_usd": round(c.da_rt_basis_risk_usd, 4),
            "net_savings_usd": round(c.net_savings_usd, 4),
            "forecast_confidence": round(c.forecast_confidence, 3),
            "window_start": c.window_start.isoformat(),
            "runtime_hours": c.runtime_hours,
            "slack_hours": round(c.slack_hours, 3),
            "migration_allowed": c.migration_allowed,
            "latency_sensitive": c.latency_sensitive,
            "cache_hit_rate": c.cache_hit_rate,
            "cache_sensitive": c.cache_sensitive,
            "estimated_cache_loss_pct": round((1.0 - self.cache_loss_factor) * 100.0, 2),
            "topology_heavy": c.topology_heavy,
            "baseline_goodput_per_dollar": round(self.baseline_goodput_per_dollar, 6),
            "candidate_goodput_per_dollar": round(self.candidate_goodput_per_dollar, 6),
            "kpi_delta": round(self.kpi_delta, 6),
            "prevents_deadline_miss": self.prevents_deadline_miss,
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "reason_details": list(self.reason_details),
        }

    @property
    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else "unspecified"


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class EnergyArbitrageAdapter:
    """Thin wrapper that consumes the existing energy engine and gates its output.

    ``recommend()`` calls the energy engine's PUBLIC API
    (``JobScheduler.solve`` + ``JobScheduler.create_baseline_schedule``) and
    diffs the two schedules into :class:`ExistingEnergyCandidate` objects.
    ``evaluate()`` routes each candidate through the constraint-aware gates.
    """

    # HEURISTIC GPU-hour cost used only as the denominator's infra term when the
    # caller does not pass one. Public-list directional figure, NOT a production
    # procurement rate (docs/RESULTS.md §8 production-claim gate).
    DEFAULT_GPU_HOUR_USD: float = 2.0
    # HEURISTIC per-migration network cost (data egress + control plane).
    DEFAULT_MIGRATION_NETWORK_USD: float = 0.5
    # HEURISTIC DA/RT basis-risk buffer (fraction of candidate energy cost) used
    # when no realized RT price is supplied. Reflects that a DA-only planner is
    # exposed to RT settlement risk.
    DEFAULT_BASIS_RISK_FRACTION: float = 0.10

    # Cache-loss model (Part C). When a workload is moved to a COLD destination
    # (one that does not preserve its prefix/KV cache), the share of served
    # goodput that depended on cache hits is degraded by the cold-route penalty
    # (recompute + TTFT). CACHE_WARMUP_HIT_RATE_LOSS mirrors the existing
    # cost-model realism constant (CostModelConfig.cache_warmup_hit_rate_loss =
    # 0.40) rather than introducing a new synthetic weight: a fraction
    # ``hit_rate * 0.40`` of goodput value is lost on a cold route.
    CACHE_WARMUP_HIT_RATE_LOSS: float = 0.40
    # At/above this prefix-cache hit rate a cache-sensitive / latency-sensitive
    # workload is preserved on its home route regardless of headline savings.
    CACHE_PRESERVE_HIT_RATE: float = 0.70
    # Below this hit rate the workload has low cache dependency — moving is safe.
    CACHE_LOW_DEPENDENCY_HIT_RATE: float = 0.30

    def __init__(
        self,
        scheduler: Optional[object] = None,
        config: Optional[OptimizationConfig] = None,
        gpu_hour_usd: Optional[float] = None,
        migration_network_usd: Optional[float] = None,
    ) -> None:
        self._scheduler = scheduler  # injected energy engine; lazily built if None
        self._config = config
        self.gpu_hour_usd = (
            gpu_hour_usd if gpu_hour_usd is not None else self.DEFAULT_GPU_HOUR_USD
        )
        self.migration_network_usd = (
            migration_network_usd if migration_network_usd is not None
            else self.DEFAULT_MIGRATION_NETWORK_USD
        )

    # ------------------------------------------------------------------
    # Calling the existing energy engine (public API only)
    # ------------------------------------------------------------------

    def recommend(
        self,
        jobs: list[Job],
        da_price_data: dict[str, dict[datetime, float]],
        carbon_data: Optional[dict[str, dict[datetime, float]]] = None,
        rt_price_data: Optional[dict[str, dict[datetime, float]]] = None,
        method: str = "greedy",
        forecast_confidence: float = 0.7,
    ) -> list[ExistingEnergyCandidate]:
        """Run the existing energy engine and return its per-job recommendations.

        Uses ONLY the engine's public API. The returned candidates are the
        engine's decisions verbatim — the adapter does not alter them here.
        """
        scheduler = self._scheduler
        if scheduler is None:
            # Lazy import keeps this module dependency-light and avoids any
            # import-time coupling to the energy core.
            from ..optimization.scheduler import JobScheduler
            scheduler = JobScheduler(self._config or OptimizationConfig())
        carbon_data = carbon_data or {r: {} for r in da_price_data}

        baseline = scheduler.create_baseline_schedule(jobs)
        result = scheduler.solve(jobs, da_price_data, carbon_data, method=method)
        return self.candidates_from_schedules(
            jobs=jobs,
            baseline_schedule=baseline,
            optimized_schedule=result.schedule,
            da_price_data=da_price_data,
            rt_price_data=rt_price_data,
            forecast_confidence=forecast_confidence,
        )

    def candidates_from_schedules(
        self,
        jobs: list[Job],
        baseline_schedule: list[ScheduleDecision],
        optimized_schedule: list[ScheduleDecision],
        da_price_data: dict[str, dict[datetime, float]],
        rt_price_data: Optional[dict[str, dict[datetime, float]]] = None,
        forecast_confidence: float = 0.7,
        source: str = "engine_optimized",
    ) -> list[ExistingEnergyCandidate]:
        """Diff baseline vs optimized schedules into energy candidates (pure)."""
        job_by_id = {j.job_id: j for j in jobs}
        base_by_id = {d.job_id: d for d in baseline_schedule}
        out: list[ExistingEnergyCandidate] = []

        for opt in optimized_schedule:
            job = job_by_id.get(opt.job_id)
            base = base_by_id.get(opt.job_id)
            if job is None or base is None:
                continue

            cur_region = base.region
            rec_region = opt.region
            da_cur = _price_at(da_price_data, cur_region, base.start_time)
            da_tgt = _price_at(da_price_data, rec_region, opt.start_time)

            base_cost = _da_cost(da_price_data, base, job)
            opt_cost = _da_cost(da_price_data, opt, job)
            gross = base_cost - opt_cost
            gross_pct = (gross / base_cost * 100.0) if base_cost > 0 else 0.0

            rt_tgt = _price_at(rt_price_data, rec_region, opt.start_time) if rt_price_data else None
            basis_risk = self._basis_risk_usd(opt, job, da_tgt, rt_tgt, opt_cost)

            if rec_region != cur_region:
                action = (
                    EnergyCandidateAction.SHIFT_BATCH_TO_CHEAPER_REGION
                    if job.workload_type in _FLEXIBLE_TYPES
                    else EnergyCandidateAction.CHOOSE_CHEAPER_REGION
                )
            elif opt.start_time > base.start_time + timedelta(minutes=1):
                action = EnergyCandidateAction.DEFER_FLEXIBLE_WORKLOAD
            else:
                action = EnergyCandidateAction.KEEP

            out.append(ExistingEnergyCandidate(
                job_id=job.job_id,
                workload_type=job.workload_type,
                action=action,
                current_region=cur_region,
                recommended_region=rec_region,
                gross_savings_usd=gross,
                gross_savings_pct=gross_pct,
                da_price_current_mwh=da_cur,
                da_price_target_mwh=da_tgt,
                rt_price_target_mwh=rt_tgt,
                da_rt_basis_risk_usd=basis_risk,
                forecast_confidence=forecast_confidence,
                window_start=opt.start_time,
                baseline_start=base.start_time,
                runtime_hours=job.runtime_hours,
                slack_hours=job.slack_hours,
                deadline=job.deadline,
                migration_allowed=job.migration_cost_hours is not None,
                migration_cost_hours=job.migration_cost_hours or 0.0,
                latency_sensitive=(job.workload_type in _CRITICAL_INTERACTIVE_TYPES),
                gpu_count=job.gpu_count,
                power_kw=job.power_kw,
                source=source,
            ))
        return out

    def _basis_risk_usd(self, decision, job, da_tgt, rt_tgt, opt_cost) -> float:
        """Expected DA/RT settlement basis risk on the candidate window."""
        if rt_tgt is not None and da_tgt is not None:
            # Realized basis on the destination start hour, applied to the
            # whole window's energy. Only the adverse (RT > DA) side is a risk.
            basis_mwh = max(0.0, rt_tgt - da_tgt)
            energy_mwh = job.power_kw / 1000.0 * job.runtime_hours
            return basis_mwh * energy_mwh
        # No realized RT data: charge a conservative buffer on the candidate cost.
        return self.DEFAULT_BASIS_RISK_FRACTION * max(0.0, opt_cost)

    # ------------------------------------------------------------------
    # Constraint-aware gates (§D)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: ExistingEnergyCandidate,
        destination_context: Optional[DestinationContext] = None,
    ) -> ConstraintAwareEnergyCandidate:
        """Route one energy candidate through the constraint-aware gates."""
        reasons: list[str] = []
        details: list[str] = []

        # KEEP candidates pass through unchanged (no move to gate).
        if candidate.action is EnergyCandidateAction.KEEP or not candidate.is_region_move:
            gp = self._goodput_per_dollar(
                candidate, candidate.baseline_start, candidate.current_region, moved=False
            )
            return ConstraintAwareEnergyCandidate(
                candidate=candidate, decision=GateDecision.ACCEPT,
                reasons=["keep_no_move"], reason_details=["keep_no_move"],
                baseline_goodput_per_dollar=gp, candidate_goodput_per_dollar=gp,
            )

        # Destination preserves the warm cache only when telemetry says so.
        preserves_affinity = bool(
            destination_context is not None and destination_context.preserves_affinity
        )
        hit = candidate.cache_hit_rate or 0.0
        cache_loss_factor = self._cache_loss_factor(candidate, preserves_affinity)

        baseline_gp = self._goodput_per_dollar(
            candidate, candidate.baseline_start, candidate.current_region,
            moved=False, cache_loss_factor=1.0,
        )
        candidate_gp = self._goodput_per_dollar(
            candidate, candidate.window_start, candidate.recommended_region,
            moved=True, cache_loss_factor=cache_loss_factor,
        )
        prevents_miss = self._move_prevents_deadline_miss(candidate)

        def verdict(decision: GateDecision) -> ConstraintAwareEnergyCandidate:
            return ConstraintAwareEnergyCandidate(
                candidate=candidate, decision=decision,
                reasons=reasons, reason_details=details,
                baseline_goodput_per_dollar=baseline_gp,
                candidate_goodput_per_dollar=candidate_gp,
                prevents_deadline_miss=prevents_miss,
                cache_loss_factor=cache_loss_factor,
            )

        def fail(code: str, detail: str) -> ConstraintAwareEnergyCandidate:
            reasons.append(code)
            details.append(detail)
            return verdict(GateDecision.REJECT)

        # Gate 1 — workload eligibility (§D.1).
        ok, code, detail = self._eligible(candidate)
        if not ok:
            return fail(code, detail)

        # Gate 2 — destination safety (§D.2).
        ok, code, detail = self._destination_safe(candidate, destination_context)
        if not ok:
            return fail(code, detail)

        # Gate 3 — SLA / deadline safety (§D.3).
        ok, code, detail = self._sla_safe(candidate)
        if not ok:
            return fail(code, detail)

        # Gate 3b — cache-affinity safety (Part C). Preserve a high-hit-rate
        # cache-/latency-sensitive workload on its home route regardless of
        # headline savings (cold-route TTFT/recompute dominates).
        if (candidate.cache_sensitive or candidate.latency_sensitive) \
                and hit >= self.CACHE_PRESERVE_HIT_RATE and not preserves_affinity:
            return fail(
                "preserve_affinity_high_cache_hit_rate",
                f"prefix_cache_hit_rate={hit:.2f}≥{self.CACHE_PRESERVE_HIT_RATE:.2f}; "
                f"est_goodput_loss={(1.0 - cache_loss_factor) * 100:.0f}% on cold route",
            )

        # Gate 4 — KPI safety (§D.4): accept only if cache-loss-adjusted SLA-safe
        # goodput/$ improves OR the move prevents a hard deadline miss.
        if candidate_gp > baseline_gp:
            if preserves_affinity:
                reasons.append("accept_energy_move_cache_safe")
                details.append(
                    f"destination preserves cache affinity; goodput/$ "
                    f"{baseline_gp:.4f}->{candidate_gp:.4f}"
                )
            elif hit < self.CACHE_LOW_DEPENDENCY_HIT_RATE:
                reasons.append("accept_energy_move_low_cache_dependency")
                details.append(
                    f"hit_rate={hit:.2f}<{self.CACHE_LOW_DEPENDENCY_HIT_RATE:.2f}; "
                    f"goodput/$ {baseline_gp:.4f}->{candidate_gp:.4f}"
                )
            else:
                reasons.append("kpi_positive")
                details.append(
                    f"cache-loss-adjusted goodput/$ {baseline_gp:.4f}->{candidate_gp:.4f}"
                )
            return verdict(GateDecision.ACCEPT)
        if prevents_miss:
            reasons.append("accepted_prevents_deadline_miss")
            details.append("move buys SLA-compliance the no-move placement misses")
            return verdict(GateDecision.MODIFY)
        # KPI is non-positive. Attribute to cache loss when that is the cause.
        if hit >= self.CACHE_LOW_DEPENDENCY_HIT_RATE and not preserves_affinity \
                and cache_loss_factor < 1.0:
            return fail(
                "reject_energy_move_cache_loss_exceeds_savings",
                f"hit_rate={hit:.2f}; cache-loss-adjusted goodput/$ "
                f"{baseline_gp:.4f}->{candidate_gp:.4f}; gross_savings="
                f"{candidate.gross_savings_usd:.2f} eroded by cold-route penalty",
            )
        reasons.append("kpi_non_positive")
        details.append(
            f"goodput/$ {baseline_gp:.4f}->{candidate_gp:.4f}; "
            f"net_savings={candidate.net_savings_usd:.4f}"
        )
        return verdict(GateDecision.REJECT)

    def evaluate_all(
        self,
        candidates: list[ExistingEnergyCandidate],
        destination_contexts: Optional[dict[str, DestinationContext]] = None,
    ) -> list[ConstraintAwareEnergyCandidate]:
        contexts = destination_contexts or {}
        return [
            self.evaluate(c, contexts.get(c.recommended_region))
            for c in candidates
        ]

    def evaluate_best(
        self,
        ranked_candidates: list[ExistingEnergyCandidate],
        destination_contexts: Optional[dict[str, DestinationContext]] = None,
    ) -> ConstraintAwareEnergyCandidate:
        """Search the energy engine's RANKED alternatives for the next-best safe
        one (Part D), instead of rejecting the top pick straight to home.

        ``ranked_candidates`` is the engine's own ranking for ONE workload, in
        priority order (e.g. engine-optimized placement, then the
        current_price_only placement, then home). The adapter does NOT generate
        these — the caller derives them from the energy engine's schedule/cost
        context. The first SLA-safe + KPI-positive alternative is accepted; if
        none are, the last (home / no-move) verdict is returned as the safe
        fallback. This preserves the engine's ranking and never forks energy
        logic.
        """
        contexts = destination_contexts or {}
        last: Optional[ConstraintAwareEnergyCandidate] = None
        rejected_chain: list[str] = []
        for c in ranked_candidates:
            v = self.evaluate(c, contexts.get(c.recommended_region))
            if v.decision in (GateDecision.ACCEPT, GateDecision.MODIFY):
                # Record which earlier alternatives were skipped and why, so the
                # explanation shows the search that found this safe alternative.
                if rejected_chain:
                    v.reasons.append("accepted_after_search")
                    v.reason_details.append(
                        "next-best safe alternative; skipped: " + "; ".join(rejected_chain)
                    )
                return v
            rejected_chain.append(f"{c.source or c.recommended_region}={v.primary_reason}")
            last = v
        # All alternatives rejected — return the last (home/no-move) verdict.
        return last if last is not None else ConstraintAwareEnergyCandidate(
            candidate=ranked_candidates[-1] if ranked_candidates else None,  # type: ignore[arg-type]
            decision=GateDecision.REJECT, reasons=["no_alternatives"],
        )

    # -- individual gates -------------------------------------------------

    def _eligible(self, c: ExistingEnergyCandidate) -> tuple[bool, str, str]:
        """Workload eligibility (§D.1). Returns (ok, code, detail)."""
        if c.latency_sensitive or c.workload_type in _CRITICAL_INTERACTIVE_TYPES:
            return (False, "ineligible_critical_interactive_inference",
                    f"latency-pinned workload_type={c.workload_type}")
        # NOTE: cache-affinity is handled by the dedicated cache gate in
        # evaluate() (Part C) so it can emit the precise preserve/reject reason
        # codes and model cache-loss-vs-savings — not a blanket eligibility block.
        if c.topology_heavy:
            return (False, "ineligible_topology_heavy_workload",
                    "topology-heavy workload pinned to its fabric")
        if c.is_region_move and not c.migration_allowed:
            return (False, "ineligible_migration_not_allowed",
                    "migration_allowed=False")
        if c.workload_type in _FLEXIBLE_TYPES:
            if c.slack_hours <= 0:
                return (False, "ineligible_no_flexibility_window",
                        f"slack_hours={c.slack_hours:.2f}")
            return (True, "eligible_flexible_batch",
                    f"flexible {c.workload_type}, slack={c.slack_hours:.1f}h")
        if c.migration_allowed and c.slack_hours > 0:
            return (True, "eligible_unknown_class_with_flexibility", "")
        return (False, "ineligible_unknown_class_default_block",
                f"workload_type={c.workload_type}")

    def _destination_safe(
        self, c: ExistingEnergyCandidate, ctx: Optional[DestinationContext]
    ) -> tuple[bool, str, str]:
        """Destination safety (§D.2). Returns (ok, code, detail)."""
        if ctx is None:
            return (False, "destination_unsafe_missing_telemetry",
                    f"no telemetry for {c.recommended_region}")
        if ctx.is_stale or ctx.telemetry_confidence == "low":
            return (False, "destination_unsafe_stale_or_low_confidence_telemetry",
                    f"stale={ctx.is_stale} conf={ctx.telemetry_confidence}")
        if ctx.is_hot:
            return (False, "destination_unsafe_thermal_hot",
                    f"{ctx.region} thermal hotspot")
        if ctx.spare_capacity_pct is not None and ctx.spare_capacity_pct < ctx.CRITICAL_SPARE_PCT:
            return (False, "destination_unsafe_full",
                    f"spare={ctx.spare_capacity_pct:.0f}%<{ctx.CRITICAL_SPARE_PCT:.0f}%")
        if ctx.spare_capacity_pct is None:
            return (False, "destination_unsafe_capacity_unknown", "")
        if ctx.queue_p95_ms is not None and ctx.queue_p95_ms > ctx.HOT_QUEUE_P95_MS:
            return (False, "destination_unsafe_high_queue",
                    f"queue_p95={ctx.queue_p95_ms:.0f}ms")
        if not ctx.topology_fit_ok:
            return (False, "destination_unsafe_bad_topology_for_workload", "")
        if ctx.is_cold and c.migration_energy_usd >= max(0.0, c.gross_savings_usd):
            return (False, "destination_unsafe_cold_warmup_exceeds_benefit",
                    f"warmup={c.migration_energy_usd:.2f}>=gross={c.gross_savings_usd:.2f}")
        return (True, "destination_safe", "")

    def _sla_safe(self, c: ExistingEnergyCandidate) -> tuple[bool, str, str]:
        """SLA / deadline safety (§D.3). Returns (ok, code, detail).

        The energy engine already plans within the deadline; this gate guards the
        EXTRA migration-warmup overhead the engine's region choice introduces and
        blocks any latency-sensitive move outright.
        """
        if c.latency_sensitive:
            return (False, "sla_unsafe_latency_sensitive_move",
                    "latency-sensitive workloads are not energy-shiftable")
        if c.migration_cost_hours > c.slack_hours:
            return (False, "sla_unsafe_deadline",
                    f"migration_cost {c.migration_cost_hours:.2f}h > "
                    f"slack {c.slack_hours:.2f}h")
        # Engine-chosen completion + warmup must not exceed the deadline.
        completion = c.window_start + timedelta(
            hours=c.runtime_hours + c.migration_cost_hours
        )
        if completion > c.deadline:
            return (False, "sla_unsafe_deadline_with_warmup",
                    "engine start + runtime + warmup exceeds deadline")
        return (True, "sla_safe", "")

    # -- KPI accounting ---------------------------------------------------

    def _cache_loss_factor(
        self, c: ExistingEnergyCandidate, preserves_affinity: bool
    ) -> float:
        """Fraction of goodput value retained after a (possibly cold) move.

        A move to a destination that does not preserve the warm cache degrades
        the share of goodput that depended on cache hits by the cold-route
        penalty (recompute + TTFT). Returns 1.0 when no degradation applies.
        Uses the existing realism constant (no new synthetic weight).
        """
        if preserves_affinity or not c.is_region_move:
            return 1.0
        hit = c.cache_hit_rate or 0.0
        return max(0.0, 1.0 - hit * self.CACHE_WARMUP_HIT_RATE_LOSS)

    def _goodput_per_dollar(
        self,
        c: ExistingEnergyCandidate,
        start_time: datetime,
        region: str,
        moved: bool,
        cache_loss_factor: float = 1.0,
    ) -> float:
        """SLA-safe goodput per infrastructure dollar for a placement.

        Numerator: ``token_equivalent`` goodput = gpu_count * runtime_hours
        (job-progress proxy, labelled token_equivalent per docs/RESULTS.md §5),
        scaled by ``cache_loss_factor`` (≤ 1.0) when a cold-route move degrades
        cache-dependent goodput. Zero if the placement misses the deadline.

        Denominator: energy_cost + gpu_infra_cost + network_cost. The migration
        warmup energy + DA/RT basis risk are charged to the moved placement's
        cost (denominator), NOT subtracted from goodput.
        """
        # Deadline filter (numerator gate). The energy engine plans within the
        # deadline; the migration warmup is charged as a cost, not a time-buster.
        completion = start_time + timedelta(hours=c.runtime_hours)
        if completion > c.deadline:
            return 0.0
        goodput = max(0.0, c.gpu_count * c.runtime_hours)
        if goodput <= 0:
            goodput = max(0.0, c.runtime_hours)  # fall back to job-progress hours
        goodput *= max(0.0, min(1.0, cache_loss_factor))  # cold-route cache loss

        price = (c.da_price_target_mwh if moved else c.da_price_current_mwh)
        price = price if price is not None else 50.0
        energy_cost = (price / 1000.0) * c.power_kw * c.runtime_hours
        if moved:
            energy_cost += c.migration_energy_usd + c.da_rt_basis_risk_usd
        gpu_infra_cost = self.gpu_hour_usd * c.gpu_count * c.runtime_hours
        network_cost = self.migration_network_usd if moved else 0.0
        denom = energy_cost + gpu_infra_cost + network_cost
        if denom <= 0:
            return 0.0
        return goodput / denom

    def _move_prevents_deadline_miss(self, c: ExistingEnergyCandidate) -> bool:
        """True iff staying in the current region misses the deadline but moving
        meets it (the move buys SLA-compliance, a safety win)."""
        stay_completion = c.window_start + timedelta(hours=c.runtime_hours)
        move_completion = c.window_start + timedelta(
            hours=c.runtime_hours + c.migration_cost_hours
        )
        return stay_completion > c.deadline and move_completion <= c.deadline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_at(
    price_data: Optional[dict[str, dict[datetime, float]]],
    region: str,
    when: datetime,
) -> Optional[float]:
    if not price_data:
        return None
    region_prices = price_data.get(region, {})
    anchor = when.replace(minute=0, second=0, microsecond=0)
    return region_prices.get(anchor)


def _da_cost(
    price_data: dict[str, dict[datetime, float]],
    decision: ScheduleDecision,
    job: Job,
) -> float:
    """DA-planned energy cost of a schedule decision (sum hourly over window)."""
    total = 0.0
    for segment in decision.all_segments:
        power_kw = job.power_kw * segment.power_fraction
        current = segment.start_time.replace(minute=0, second=0, microsecond=0)
        end = segment.end_time
        region_prices = price_data.get(segment.region, {})
        while current < end:
            hour_fraction = min(1.0, (end - current).total_seconds() / 3600.0)
            if hour_fraction <= 0:
                break
            price = region_prices.get(current, 50.0)
            total += (price / 1000.0) * power_kw * hour_fraction
            current += timedelta(hours=1)
    return total
