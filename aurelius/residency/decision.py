"""Model Residency Decision Engine v1 — scoring + decision logic.

Given a request for ``(model_id[, adapter_id])`` and a set of candidate serving
locations, this engine recommends **where** to place/route the request (or
whether to prewarm / evict / reject), optimizing the ``docs/RESULTS.md`` primary
KPI — **SLA-safe goodput per infrastructure dollar**.

Binding rules (``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §4, §5):

- **Recommendation-only in real/customer mode.** This module computes
  :class:`ResidencyDecision` objects with ``executable_in_real_cluster=False``.
  It performs **no** mutation — not of real clusters, routers, or serving
  engines. Simulator mutation lives in ``aurelius/residency/sim.py``.
- **No substitution.** A decision NEVER serves a different model/adapter than
  requested — it only changes *where/when* the requested model is loaded/routed.
- **Missing ≠ zero / free.** An unknown load latency is never treated as 0; an
  unknown memory headroom never permits a route; unknown telemetry lowers
  confidence and can force ``INSUFFICIENT_TELEMETRY``.
- **Safety is a veto, not a weight.** Memory headroom, SLA, thermal, topology,
  region, and telemetry-confidence gates veto candidates; they are never folded
  into the KPI as weighted terms (``docs/RESULTS.md`` §1-§2).

Pure / deterministic / stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import (
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
    ResidencyAction,
    ResidencyDecision,
)

# Confidence ordering for gating.
_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}

# Vetoes that make a candidate INFEASIBLE (hard safety) vs ones that signal
# MISSING telemetry (the candidate is not safely scorable).
_HARD_UNSAFE_VETOES = frozenset({
    "region_not_allowed", "thermal_risk", "low_topology_score",
    "insufficient_memory_headroom", "queue_wait_exceeds_max",
})
_MISSING_VETOES = frozenset({
    "missing_load_latency", "memory_telemetry_missing",
    "queue_telemetry_missing", "low_telemetry_confidence",
})


@dataclass
class SafetyContext:
    """Safety gates + scoring priors for a decision (never weights in the KPI).

    All thresholds are explicit and conservative. ``service_time_proxy_s`` is the
    decode/serve time the engine assumes when the request carries no token
    counts; it is a *latency proxy*, documented as such.
    """

    gpu_hour_price: float = 3.0
    # Hard safety vetoes
    min_memory_headroom_gb: float = 0.0       # require model fits in free memory
    max_thermal_risk: Optional[float] = 0.85  # veto above this (None = ignore)
    min_topology_score: Optional[float] = None
    min_telemetry_confidence: str = "low"     # below this → location not trusted
    max_queue_wait_s: Optional[float] = None  # hard queue veto (None = SLA-only)
    allowed_regions: Optional[tuple] = None    # overrides request.allowed_regions
    # Latency model
    default_latency_sla_ms: float = 30000.0   # used if request has none
    service_time_proxy_s: float = 2.0         # decode/serve proxy when no tokens
    seconds_per_token: float = 0.0            # optional token→latency proxy
    # Warm-pool / prewarm economics (recommendation thresholds, not KPI weights)
    warm_pool_hold_hours: float = 1.0         # GPU-hours a prewarm reserves
    prewarm_expected_hits: float = 5.0        # expected matching requests in window
    # Churn / eviction
    eviction_cost_s: float = 0.0              # modelled churn cost of an evict
    # Affinity hysteresis: only abandon a warm/resident replica for a cold one
    # when goodput/$ improves by MORE than this fraction (avoid churn on noise).
    kpi_tie_band: float = 0.01

    def sla_s(self, request: ModelResidencyRequest) -> float:
        if request.latency_sla_ms is not None:
            return request.latency_sla_ms / 1000.0
        if request.deadline_s is not None:
            return request.deadline_s
        return self.default_latency_sla_ms / 1000.0


@dataclass
class CandidateScore:
    """Score for routing ``request`` to one candidate location."""

    location_key: str
    feasible: bool
    expected_latency_s: Optional[float]
    expected_cost: Optional[float]
    sla_met: Optional[bool]
    goodput_per_dollar: Optional[float]     # (1 if sla_met else 0) / cost
    model_resident: bool
    adapter_resident: bool
    is_cold: bool
    safety_vetoes: tuple = ()
    confidence: str = "unknown"
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "location_key": self.location_key,
            "feasible": self.feasible,
            "expected_latency_s": self.expected_latency_s,
            "expected_cost": self.expected_cost,
            "sla_met": self.sla_met,
            "goodput_per_dollar": self.goodput_per_dollar,
            "model_resident": self.model_resident,
            "adapter_resident": self.adapter_resident,
            "is_cold": self.is_cold,
            "safety_vetoes": list(self.safety_vetoes),
            "confidence": self.confidence,
            "components": dict(self.components),
        }


def _service_time_s(request: ModelResidencyRequest, ctx: SafetyContext) -> float:
    """Decode/serve time proxy (latency only — never a KPI weight)."""
    if ctx.seconds_per_token > 0 and request.output_tokens:
        return max(0.01, ctx.seconds_per_token * request.output_tokens)
    return ctx.service_time_proxy_s


def score_residency_candidate(request: ModelResidencyRequest,
                              candidate_location: ModelLocationState,
                              load_profile: Optional[ModelLoadProfile],
                              cost_config: SafetyContext,
                              safety_context: SafetyContext) -> CandidateScore:
    """Estimate expected latency / cost / goodput-per-$ for one candidate.

    ``cost_config`` and ``safety_context`` are both :class:`SafetyContext` (the
    cost basis lives on the same context); they are accepted separately to match
    the spec signature. ``safety_context`` is authoritative for gates.
    """
    ctx = safety_context
    loc = candidate_location
    vetoes: list[str] = []
    components: dict = {}

    model_resident = loc.has_model(request.model_id)
    adapter_needed = request.adapter_id is not None
    adapter_resident = (not adapter_needed) or loc.has_adapter(request.adapter_id)
    is_cold = (not model_resident) or (adapter_needed and not adapter_resident)

    # --- region gate ---
    allowed = ctx.allowed_regions or request.allowed_regions
    if allowed is not None and loc.region not in allowed:
        vetoes.append("region_not_allowed")

    # --- telemetry-confidence gate ---
    if _CONF_RANK.get(loc.telemetry_confidence, 0) < _CONF_RANK.get(
            ctx.min_telemetry_confidence, 0):
        vetoes.append("low_telemetry_confidence")

    # --- load penalties (missing latency must NOT become 0) ---
    safety_critical = request.is_safety_critical
    model_penalty = 0.0
    adapter_penalty = 0.0
    missing_latency = False
    if not model_resident:
        p = load_profile.model_load_penalty_s(
            safety_critical=safety_critical) if load_profile else None
        if p is None:
            missing_latency = True
        else:
            model_penalty = p
    if adapter_needed and not adapter_resident:
        p = load_profile.adapter_load_penalty_s(
            safety_critical=safety_critical) if load_profile else None
        if p is None:
            missing_latency = True
        else:
            adapter_penalty = p
    if missing_latency:
        vetoes.append("missing_load_latency")

    # --- memory headroom gate (cold loads only; missing memory ≠ unlimited) ---
    if is_cold and load_profile is not None and load_profile.memory_required_gb:
        free = loc.memory_free_gb
        if free is None:
            vetoes.append("memory_telemetry_missing")
        elif free - load_profile.memory_required_gb < ctx.min_memory_headroom_gb:
            vetoes.append("insufficient_memory_headroom")

    # --- thermal / topology gates ---
    if (ctx.max_thermal_risk is not None and loc.thermal_risk is not None
            and loc.thermal_risk > ctx.max_thermal_risk):
        vetoes.append("thermal_risk")
    if (ctx.min_topology_score is not None and loc.topology_score is not None
            and loc.topology_score < ctx.min_topology_score):
        vetoes.append("low_topology_score")

    # --- queue wait ---
    queue_wait = loc.estimated_queue_wait_s
    if queue_wait is None:
        # derive a coarse proxy from queue_depth × service time if available
        if loc.queue_depth is not None:
            queue_wait = loc.queue_depth * _service_time_s(request, ctx)
        else:
            vetoes.append("queue_telemetry_missing")
            queue_wait = None
    if (queue_wait is not None and ctx.max_queue_wait_s is not None
            and queue_wait > ctx.max_queue_wait_s):
        vetoes.append("queue_wait_exceeds_max")

    service_s = _service_time_s(request, ctx)
    components.update(model_load_penalty_s=model_penalty,
                      adapter_load_penalty_s=adapter_penalty,
                      queue_wait_s=queue_wait, service_time_s=service_s)

    # If a hard input is missing, the candidate is not safely scorable.
    if any(v in _MISSING_VETOES for v in vetoes):
        return CandidateScore(
            location_key=loc.location_key, feasible=False,
            expected_latency_s=None, expected_cost=None, sla_met=None,
            goodput_per_dollar=None, model_resident=model_resident,
            adapter_resident=adapter_resident, is_cold=is_cold,
            safety_vetoes=tuple(vetoes), confidence=loc.telemetry_confidence,
            components=components)

    expected_latency = queue_wait + model_penalty + adapter_penalty + service_s

    # --- cost: incremental gpu cost + memory-pressure + (prewarm/evict handled
    #     by the engine) ---
    incremental_gpu_cost = (expected_latency / 3600.0) * ctx.gpu_hour_price
    memory_pressure_cost = 0.0
    if (load_profile is not None and load_profile.memory_required_gb
            and loc.memory_free_gb is not None and loc.gpu_memory_total):
        used_after = (loc.gpu_memory_used or 0.0) + load_profile.memory_required_gb * 1e9
        pressure = max(0.0, used_after / loc.gpu_memory_total - 0.9)
        memory_pressure_cost = pressure * incremental_gpu_cost
    expected_cost = incremental_gpu_cost + memory_pressure_cost
    components.update(incremental_gpu_cost=round(incremental_gpu_cost, 6),
                      memory_pressure_cost=round(memory_pressure_cost, 6))

    sla = ctx.sla_s(request)
    sla_met = expected_latency <= sla
    # hard safety vetoes (region/thermal/topology/memory/queue-max) make the
    # candidate infeasible even if latency is fine.
    feasible = not any(v in _HARD_UNSAFE_VETOES for v in vetoes)
    gpd = ((1.0 if sla_met else 0.0) / expected_cost) if expected_cost > 0 else None

    return CandidateScore(
        location_key=loc.location_key, feasible=feasible,
        expected_latency_s=expected_latency, expected_cost=expected_cost,
        sla_met=sla_met, goodput_per_dollar=gpd, model_resident=model_resident,
        adapter_resident=adapter_resident, is_cold=is_cold,
        safety_vetoes=tuple(vetoes), confidence=loc.telemetry_confidence,
        components=components)


def _conf_from(*labels: str) -> str:
    """Weakest-link confidence of the inputs."""
    rank = min((_CONF_RANK.get(x, 0) for x in labels), default=0)
    for name, r in _CONF_RANK.items():
        if r == rank:
            return name
    return "unknown"


def _is_resident(s: CandidateScore) -> bool:
    return not s.is_cold


def _select_best(sla_ok: list, current: Optional[CandidateScore],
                 tie_band: float) -> CandidateScore:
    """Pick the max-goodput/$ SLA-safe candidate, preferring affinity within a
    KPI tie band (only abandon a resident replica when goodput/$ improves by
    more than ``tie_band``)."""
    best_gpd = max((s.goodput_per_dollar or 0.0) for s in sla_ok)
    threshold = best_gpd * (1.0 - tie_band)
    near = [s for s in sla_ok if (s.goodput_per_dollar or 0.0) >= threshold]

    # within the band, prefer: current route (if present & resident) →
    # resident → lowest latency → stable key.
    def _tiebreak(s: CandidateScore):
        is_current = current is not None and s.location_key == current.location_key
        return (
            1 if (is_current and _is_resident(s)) else 0,
            1 if _is_resident(s) else 0,
            -(s.expected_latency_s if s.expected_latency_s is not None else 1e9),
            s.goodput_per_dollar or 0.0,
        )

    # If the current route is itself within the band, keep it (no churn).
    if current is not None:
        for s in near:
            if s.location_key == current.location_key:
                return s
    return max(near, key=_tiebreak)


def choose_residency_decision(request: ModelResidencyRequest,
                              locations: list,
                              load_profiles: dict,
                              cost_config: SafetyContext,
                              safety_context: SafetyContext) -> ResidencyDecision:
    """Recommend a residency action for ``request`` over ``locations``.

    ``load_profiles`` maps ``model_id`` and ``(model_id, adapter_id)`` →
    :class:`ModelLoadProfile`. Returns a recommendation-only
    :class:`ResidencyDecision` (``executable_in_real_cluster=False``).
    """
    ctx = safety_context
    profile = (load_profiles.get((request.model_id, request.adapter_id))
               or load_profiles.get(request.model_id))
    current_key = request.current_route

    # --- 1. telemetry gate ---
    if not locations:
        return ResidencyDecision(
            request_id=request.request_id,
            action=ResidencyAction.INSUFFICIENT_TELEMETRY,
            reason="no candidate locations supplied", confidence="unknown")
    trusted = [loc for loc in locations
               if _CONF_RANK.get(loc.telemetry_confidence, 0)
               >= _CONF_RANK.get(ctx.min_telemetry_confidence, 0)]
    if not trusted:
        return ResidencyDecision(
            request_id=request.request_id,
            action=ResidencyAction.INSUFFICIENT_TELEMETRY,
            current_location=current_key,
            reason="no location meets the minimum telemetry-confidence gate "
                   f"({ctx.min_telemetry_confidence})", confidence="low")

    # --- 2. score all candidates ---
    loc_by_key = {loc.location_key: loc for loc in locations}
    scored = {loc.location_key: score_residency_candidate(
        request, loc, profile, cost_config, ctx) for loc in locations}
    feasible = [s for s in scored.values() if s.feasible]

    # collect vetoes for transparency
    all_vetoes = sorted({v for s in scored.values() for v in s.safety_vetoes})

    if not feasible:
        # (i) eviction opportunity: a candidate blocked ONLY by memory headroom
        #     (no other hard veto, no missing telemetry) whose location holds an
        #     evictable resident model → recommend EVICT_CANDIDATE.
        for s in scored.values():
            vetoset = set(s.safety_vetoes)
            if ("insufficient_memory_headroom" in vetoset
                    and not (vetoset & (_HARD_UNSAFE_VETOES
                                        - {"insufficient_memory_headroom"}))
                    and not (vetoset & _MISSING_VETOES)):
                loc = loc_by_key[s.location_key]
                if any(m != request.model_id for m in loc.loaded_model_ids):
                    return ResidencyDecision(
                        request_id=request.request_id,
                        action=ResidencyAction.EVICT_CANDIDATE, reason=(
                            "memory pressure blocks every route; recommend "
                            "evicting a low-value resident model to admit the "
                            "requested one (advisory; respect anti-thrash cooldown)"),
                        current_location=current_key,
                        target_location=s.location_key,
                        expected_cost_delta=round(ctx.eviction_cost_s / 3600.0
                                                  * ctx.gpu_hour_price, 6),
                        safety_vetoes=tuple(all_vetoes), confidence="low")
        # (ii) distinguish missing-telemetry from genuinely-unsafe
        if any(v in _MISSING_VETOES for v in all_vetoes) and profile is None:
            action = ResidencyAction.INSUFFICIENT_TELEMETRY
            reason = "no feasible candidate and required load/telemetry missing"
        else:
            action = ResidencyAction.REJECT_UNSAFE_ROUTE
            reason = ("no candidate passes hard safety gates: "
                      + ", ".join(all_vetoes))
        return ResidencyDecision(
            request_id=request.request_id, action=action, reason=reason,
            current_location=current_key, safety_vetoes=tuple(all_vetoes),
            confidence="low")

    # --- 3. SLA-safe candidates first; among them, MAX goodput/$ is the
    #     objective. Affinity (resident) is only a tie-breaker WITHIN a small
    #     KPI band, so a cold node wins only when goodput/$ MATERIALLY improves
    #     (docs/RESULTS.md §1; spec: "do not route to lower queue if it causes a
    #     larger cold-start penalty unless KPI improves").
    sla_ok = [s for s in feasible if s.sla_met]
    current = scored.get(current_key) if current_key else None

    # If NO candidate can meet SLA at all → unsafe.
    if not sla_ok:
        least_bad = min(feasible, key=lambda s: (s.expected_latency_s
                                                 if s.expected_latency_s is not None
                                                 else float("inf")))
        return ResidencyDecision(
            request_id=request.request_id,
            action=ResidencyAction.REJECT_UNSAFE_ROUTE,
            reason="no candidate meets the latency SLA even when feasible",
            current_location=current_key, target_location=least_bad.location_key,
            safety_vetoes=tuple(all_vetoes),
            expected_latency_delta_s=least_bad.expected_latency_s,
            confidence=_conf_from(least_bad.confidence,
                                  profile.confidence if profile else "unknown"))

    best = _select_best(sla_ok, current, ctx.kpi_tie_band)
    conf = _conf_from(best.confidence, profile.confidence if profile else "unknown")
    cold_saved = None
    if best.is_cold and profile is not None:
        cold_saved = (profile.model_load_penalty_s(
            safety_critical=request.is_safety_critical) or 0.0)

    # deltas vs current route
    cost_delta = None
    gpd_delta = None
    lat_delta = None
    if current is not None and current.expected_cost is not None:
        if best.expected_cost is not None:
            cost_delta = round(best.expected_cost - current.expected_cost, 6)
        if (best.goodput_per_dollar is not None
                and current.goodput_per_dollar is not None):
            gpd_delta = round(best.goodput_per_dollar - current.goodput_per_dollar, 6)
        if (best.expected_latency_s is not None
                and current.expected_latency_s is not None):
            lat_delta = round(best.expected_latency_s - current.expected_latency_s, 4)

    # --- 4. classify the action ---
    resident_sla_ok = [s for s in sla_ok if not s.is_cold]
    best_resident = not best.is_cold

    deltas = dict(expected_cold_start_saved_s=cold_saved,
                  expected_queue_delta_s=lat_delta,
                  expected_latency_delta_s=lat_delta,
                  expected_cost_delta=cost_delta,
                  expected_goodput_per_dollar_delta=gpd_delta,
                  safety_vetoes=tuple(all_vetoes), confidence=conf)

    # (b) best candidate is warm (model + adapter resident).
    if best_resident:
        # current route is itself the chosen warm best → keep / preserve affinity
        if current is not None and current.location_key == best.location_key:
            tempting = [s for s in sla_ok if s.is_cold
                        and s.goodput_per_dollar is not None
                        and current.goodput_per_dollar is not None
                        and s.goodput_per_dollar > current.goodput_per_dollar]
            if tempting:
                return ResidencyDecision(
                    request_id=request.request_id,
                    action=ResidencyAction.PRESERVE_AFFINITY, target_location=best.location_key,
                    current_location=current_key, reason=(
                        "preserve affinity: requested model is resident here; a "
                        "cold replica scored higher per-request but moving would "
                        "pay a cold start and churn the warm pool"), **deltas)
            return ResidencyDecision(
                request_id=request.request_id,
                action=ResidencyAction.KEEP_CURRENT_ROUTE, target_location=best.location_key,
                current_location=current_key,
                reason="current route already optimal (model resident, SLA-safe)",
                **deltas)
        # route to a (different) warm replica
        return ResidencyDecision(
            request_id=request.request_id,
            action=ResidencyAction.ROUTE_TO_RESIDENT_MODEL,
            target_location=best.location_key, current_location=current_key,
            reason=(f"route to location with requested model"
                    f"{'+adapter' if request.adapter_id else ''} resident "
                    "(highest SLA-safe goodput/$)"), **deltas)

    # (c) best SLA-safe candidate is COLD.
    return _cold_best_decision(request, best, scored, current, resident_sla_ok,
                               profile, ctx, all_vetoes, conf, cold_saved, lat_delta)


def _cold_best_decision(request, best, scored, current, resident_sla_ok, profile,
                        ctx, all_vetoes, conf, cold_saved, lat_delta):
    """The top-KPI SLA-safe candidate is cold. Decide prewarm / evict / route.

    - If a WARM SLA-safe replica also exists, default to AFFINITY (route there);
      only PREWARM the cold target when warm-pool economics justify keeping it
      warm for sustained demand (don't churn for a marginal one-request gain).
    - If NO warm SLA-safe replica exists, the cold target is REQUIRED to serve
      the request safely → PREWARM_MODEL/ADAPTER (or EVICT_CANDIDATE under memory
      pressure).
    """
    base_resident = best.model_resident
    saved_s = cold_saved or 0.0
    saved_dollars = (saved_s / 3600.0) * ctx.gpu_hour_price * ctx.prewarm_expected_hits
    warm_pool_cost = ctx.warm_pool_hold_hours * ctx.gpu_hour_price
    prewarm_justified = saved_dollars > warm_pool_cost

    def _prewarm_decision(reason):
        action = (ResidencyAction.PREWARM_ADAPTER
                  if base_resident and request.adapter_id
                  else ResidencyAction.PREWARM_MODEL)
        return ResidencyDecision(
            request_id=request.request_id, action=action, reason=reason,
            current_location=request.current_route, target_location=best.location_key,
            expected_cold_start_saved_s=saved_s,
            expected_cost_delta=round(warm_pool_cost, 6),
            expected_latency_delta_s=lat_delta, safety_vetoes=tuple(all_vetoes),
            confidence=conf)

    # No warm SLA-safe replica → the (feasible) cold target is required to serve
    # the request within SLA. (Memory-blocked clusters are handled earlier as
    # EVICT_CANDIDATE; if we got here the cold target has memory headroom.)
    if not resident_sla_ok:
        return _prewarm_decision(
            "no warm SLA-safe replica for the requested model; a cold load is "
            f"required to serve within SLA (≈{round(saved_s, 2)}s load)"
            + (" — and warm-pool economics also justify keeping it warm"
               if prewarm_justified else ""))

    # A warm SLA-safe replica exists. Only prewarm the cold target if sustained
    # demand justifies it; otherwise keep affinity (route to the warm replica).
    if prewarm_justified:
        return _prewarm_decision(
            f"prewarm beneficial: expected saved ≈ ${round(saved_dollars, 4)} > "
            f"warm-pool cost ${round(warm_pool_cost, 4)} over "
            f"{ctx.prewarm_expected_hits} expected hits")

    # affinity fallback: route to the best WARM replica instead of churning cold.
    warm_best = _select_best(resident_sla_ok, current, ctx.kpi_tie_band)
    if current is not None and current.location_key == warm_best.location_key:
        return ResidencyDecision(
            request_id=request.request_id, action=ResidencyAction.PRESERVE_AFFINITY,
            target_location=warm_best.location_key, current_location=request.current_route,
            reason=("a cold replica scored higher per-request but prewarm is not "
                    "justified; preserve affinity on the warm resident replica"),
            expected_cold_start_saved_s=cold_saved, safety_vetoes=tuple(all_vetoes),
            confidence=conf)
    return ResidencyDecision(
        request_id=request.request_id, action=ResidencyAction.ROUTE_TO_RESIDENT_MODEL,
        target_location=warm_best.location_key, current_location=request.current_route,
        reason=("cold replica scored higher per-request but prewarm not justified; "
                "route to the best warm resident replica (affinity)"),
        expected_cold_start_saved_s=cold_saved, safety_vetoes=tuple(all_vetoes),
        confidence=conf)
