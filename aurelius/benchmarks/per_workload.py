"""Per-workload-type benchmark reporting layer.

The previous report compared constraint_aware to FIFO across 26 scenarios using
only economic alpha. This module adds:
  - per-scenario workload-type / optimization-intent classification
  - per-scenario selection of the *strongest relevant baseline* (not FIFO)
  - alpha-vs-safety outcome classifier (KEEP_CORRECT, SAFETY_WIN, ALPHA_WIN,
    TIE, LOSS) with multi-cause loss reasons
  - aggregator that emits the four required tables (A overall policy, B
    per-workload-type, C per-scenario, D baseline strength)

FIFO is now a sanity-only baseline. This module is the buyer-facing benchmark
truth surface — pure functions, no I/O, no global state.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# Allowed string values (validated, but stored as plain str for serialization).
# The mission spec (PR #87) requires a specific vocabulary
# (critical_interactive_inference, standard_interactive_inference,
# embeddings_offline, training, communication_heavy, mixed_cluster,
# telemetry_degraded). The implementation predates that spec and uses shorter
# names; rather than rename and break the golden classification fixture, we
# accept BOTH forms and document the mapping in WORKLOAD_TYPE_ALIASES below.
WORKLOAD_TYPES = frozenset({
    # Implementation names (back-compat with existing tests + golden fixture).
    "inference_critical", "inference_standard", "batch_inference",
    "batch_training", "fine_tuning", "embedding_offline",
    "telemetry_fail_safe", "mixed", "best_effort",
    # Mission-spec aliases (PR #87 spec) — both forms are valid.
    "critical_interactive_inference", "standard_interactive_inference",
    "embeddings_offline", "training", "communication_heavy",
    "mixed_cluster", "telemetry_degraded",
})

# Map implementation-name → mission-spec-alias. Both forms are valid in
# WORKLOAD_TYPES; this dict documents the equivalence for migrations.
WORKLOAD_TYPE_ALIASES = {
    "inference_critical": "critical_interactive_inference",
    "inference_standard": "standard_interactive_inference",
    "embedding_offline": "embeddings_offline",
    "batch_training": "training",
    "telemetry_fail_safe": "telemetry_degraded",
    "mixed": "mixed_cluster",
}
OPTIMIZATION_INTENTS = frozenset({
    "energy_arbitrage", "thermal_spread", "queue_relief",
    "memory_pressure_relief", "topology_fit", "fragmentation_packing",
    "safety_keep", "completion_time",
})
BASELINE_NAMES = frozenset({
    "fifo", "current_price_only", "greedy_energy", "sla_aware",
    "first_fit", "best_fit", "first_fit_decreasing", "clairvoyant_lower_bound",
})
SLO_TYPES = frozenset({
    "p99_latency", "queue_wait_p95", "deadline", "throughput_only", "none",
})

OUTCOMES = ("KEEP_CORRECT", "SAFETY_WIN", "ALPHA_WIN", "TIE", "LOSS")
LOSS_REASON_CODES = (
    "missing_candidate_action", "over_conservative_gate",
    "simulator_limitation", "telemetry_fail_safe",
    "missing_forecast_lookahead",
    # Mission-spec additions (PR #87): three more reason codes so the buyer
    # report can attribute LOSSes to under-modeled simulator effects, wrong
    # workload classification, or scenarios that simply aren't a fair test of
    # any CA action.
    "wrong_workload_classification",
    "under_modeled_action_effect",
    "scenario_not_applicable",
)

# Material threshold for ALPHA / TIE band (1%).
_MATERIAL = 0.01

# Safety filter epsilon — a candidate baseline must not have MORE SLA
# violations than FIFO to be considered "safe" (strict). p99 must be at
# most 1.5x FIFO's p99 (the safety filter — critic 5).
_SAFETY_P99_MULTIPLE = 1.5


@dataclass(frozen=True)
class ScenarioMetadata:
    """Per-scenario classification used to pick the relevant baseline."""
    scenario_name: str
    primary_workload_type: str
    optimization_intent: str
    relevant_baselines: tuple
    headline_baseline_override: Optional[str]
    goodput_unit: str   # "tokens", "token_equivalent", "telemetry_correct_keeps"
    sla_slo_type: str
    is_telemetry_failsafe: bool


@dataclass(frozen=True)
class OutcomeAnalysis:
    """Result of analyze_outcome — alpha-vs-safety classification."""
    outcome: str
    margin_pct: float        # signed (ca - baseline) / baseline * 100
    safety_evidence: tuple
    loss_reasons: tuple   # may be multiple; first is primary
    notes: str


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# expected_primary_constraint → (intent, relevant_baselines, slo)
_INTENT_MAP = {
    "energy_bound": (
        "energy_arbitrage",
        ("fifo", "current_price_only", "greedy_energy", "sla_aware"),
        "p99_latency",
    ),
    "thermal_bound": (
        "thermal_spread", ("fifo", "sla_aware"), "p99_latency",
    ),
    "queue_bound": (
        "queue_relief", ("fifo", "sla_aware"), "queue_wait_p95",
    ),
    "latency_bound": (
        "queue_relief", ("fifo", "sla_aware"), "p99_latency",
    ),
    "memory_bound_indirect": (
        "memory_pressure_relief", ("fifo", "sla_aware"), "p99_latency",
    ),
    "topology_bound": (
        "topology_fit", ("fifo", "sla_aware"), "p99_latency",
    ),
    "utilization_bound": (
        "fragmentation_packing",
        ("fifo", "first_fit", "best_fit", "first_fit_decreasing"),
        "throughput_only",
    ),
    "communication_bound": (
        "topology_fit", ("fifo", "sla_aware"), "p99_latency",
    ),
}


def classify_scenario(name: str,
                      expected_primary_constraint: Optional[str],
                      raw: Mapping[str, Any]) -> ScenarioMetadata:
    """Build ScenarioMetadata from explicit raw dict keys with rule-based
    inference as fallback. Never raises.

    Args:
        name: scenario name.
        expected_primary_constraint: e.g. 'energy_bound', 'thermal_bound', …
        raw: the scenario dict (from YAML or builtin).
    """
    # Explicit-override path runs FIRST: explicit raw-dict keys must beat
    # any name-based heuristic (otherwise a scenario named with "telemetry"
    # plus an explicit primary_workload_type=batch_training would silently
    # lose its override). Critic ISSUE 9.
    explicit_primary = raw.get("primary_workload_type")
    explicit_intent = raw.get("optimization_intent")
    explicit_baselines = raw.get("relevant_baselines")
    explicit_slo = raw.get("sla_slo_type")
    explicit_goodput_unit = raw.get("goodput_unit")
    explicit_is_failsafe = raw.get("is_telemetry_failsafe")

    has_explicit = any(
        v is not None for v in (
            explicit_primary, explicit_intent, explicit_baselines,
            explicit_slo, explicit_goodput_unit, explicit_is_failsafe,
        )
    )

    if has_explicit:
        key = (expected_primary_constraint or "").strip()
        intent, baselines, slo = _INTENT_MAP.get(
            key, ("queue_relief", ("fifo", "sla_aware"), "p99_latency"),
        )
        primary_wt = explicit_primary or _infer_primary_workload_type(
            raw.get("workloads", [])
        )
        return ScenarioMetadata(
            scenario_name=name,
            primary_workload_type=primary_wt,
            optimization_intent=explicit_intent or intent,
            relevant_baselines=(
                tuple(explicit_baselines) if explicit_baselines else baselines
            ),
            headline_baseline_override=raw.get("headline_baseline"),
            goodput_unit=explicit_goodput_unit or _default_goodput_unit(primary_wt),
            sla_slo_type=explicit_slo or slo,
            is_telemetry_failsafe=bool(explicit_is_failsafe)
                if explicit_is_failsafe is not None else False,
        )

    # Telemetry-failsafe detection by name (only when no explicit overrides).
    lname = name.lower()
    is_failsafe = any(
        t in lname for t in ("telemetry", "_partial", "degraded", "low_confidence")
    )
    if is_failsafe:
        return ScenarioMetadata(
            scenario_name=name,
            primary_workload_type="telemetry_fail_safe",
            optimization_intent="safety_keep",
            relevant_baselines=("fifo", "sla_aware"),
            headline_baseline_override=raw.get("headline_baseline"),
            goodput_unit=raw.get("goodput_unit", "telemetry_correct_keeps"),
            sla_slo_type=raw.get("sla_slo_type", "none"),
            is_telemetry_failsafe=True,
        )

    key = (expected_primary_constraint or "").strip()
    intent, baselines, slo = _INTENT_MAP.get(
        key, ("queue_relief", ("fifo", "sla_aware"), "p99_latency"),
    )

    primary_wt = _infer_primary_workload_type(raw.get("workloads", []))
    return ScenarioMetadata(
        scenario_name=name,
        primary_workload_type=primary_wt,
        optimization_intent=intent,
        relevant_baselines=baselines,
        headline_baseline_override=raw.get("headline_baseline"),
        goodput_unit=_default_goodput_unit(primary_wt),
        sla_slo_type=slo,
        is_telemetry_failsafe=False,
    )


def _infer_primary_workload_type(workloads_list) -> str:
    """Strict-plurality (>50% by count) of workload-class; else 'mixed'."""
    if not workloads_list:
        return "mixed"
    counts: dict = {}
    for w in workloads_list:
        wt = (w.get("workload_type") or "").lower()
        tier = (w.get("priority_tier") or "").lower()
        cls = _workload_type_from_fields(
            wt, tier, bool(w.get("latency_sensitive", False)),
        )
        counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return "mixed"
    total = sum(counts.values())
    top, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top if top_count * 2 > total else "mixed"  # strict plurality


def _workload_type_from_fields(wtype: str, tier: str,
                               latency_sensitive: bool) -> str:
    """Mirrors engine._workload_class but keyed off raw scenario dict fields.

    Output values use the spec workload-type vocabulary
    (inference_critical / inference_standard / batch_inference /
    batch_training / fine_tuning / embedding_offline).
    """
    if tier == "critical":
        return "inference_critical"
    if tier == "best_effort":
        return "batch_inference"
    if tier == "batch" and ("training" in wtype or "fine_tuning" in wtype
                            or wtype == "batch_training"):
        return "batch_training"
    if tier == "batch":
        return "batch_inference"
    if "fine_tuning" in wtype:
        return "fine_tuning"
    if "training" in wtype:
        return "batch_training"
    if "embedding" in wtype:
        return "embedding_offline"
    if tier == "latency_sensitive" or latency_sensitive:
        return "inference_standard"
    if tier == "flexible":
        if "training" in wtype or wtype == "batch_training":
            return "batch_training"
        return "inference_standard"
    if tier == "standard":
        return "inference_standard"
    return "mixed"


def _default_goodput_unit(workload_type: str) -> str:
    if workload_type == "telemetry_fail_safe":
        return "telemetry_correct_keeps"
    if workload_type in ("batch_training", "fine_tuning"):
        return "token_equivalent"
    return "tokens"


# ---------------------------------------------------------------------------
# Workload-class reuse — public alias around the engine helper.
# ---------------------------------------------------------------------------

def workload_class_from_iss(service) -> str:
    """Stable public wrapper for the engine's _workload_class.

    External callers should use this instead of the underscored engine helper.
    """
    from aurelius.constraints.engine import _workload_class
    return _workload_class(service)


# ---------------------------------------------------------------------------
# Baseline selection
# ---------------------------------------------------------------------------

def _kpi_field(kpi: Any, name: str, default=None):
    """Safe getattr for objects or dicts."""
    if kpi is None:
        return default
    if isinstance(kpi, Mapping):
        return kpi.get(name, default)
    return getattr(kpi, name, default)


def _goodput_per_dollar(kpi: Any) -> Optional[float]:
    v = _kpi_field(kpi, "sla_safe_goodput_per_infra_dollar", None)
    return v


def _p99(kpi: Any) -> Optional[float]:
    return _kpi_field(kpi, "p99_latency_ms", None)


def _sla(kpi: Any) -> int:
    v = _kpi_field(kpi, "total_sla_violations", 0)
    return int(v) if v is not None else 0


def _thermal(kpi: Any) -> int:
    v = _kpi_field(kpi, "total_thermal_throttle_ticks", 0)
    return int(v) if v is not None else 0


def _is_candidate_safe(cand: Any, fifo: Any) -> bool:
    """Strict safety filter — a candidate is safe iff its SLA violations are
    <= FIFO's AND its p99 is <= 1.5x FIFO's.

    p99=None on either side is treated as 'not measurable, accept'.
    """
    if cand is None or fifo is None:
        return False
    if _sla(cand) > _sla(fifo):
        return False
    cp = _p99(cand)
    fp = _p99(fifo)
    if cp is not None and fp is not None and fp > 0:
        if cp > fp * _SAFETY_P99_MULTIPLE:
            return False
    return True


def select_headline_baseline(metadata: ScenarioMetadata,
                             policy_results: Mapping[str, Any]) -> tuple:
    """Pure. Returns (baseline_name, rationale).

    Rule cascade (in order — ISSUE 3: optimization_intent must take priority
    over workload_type, because intent describes the actual control problem
    while workload_type describes the traffic shape):
      1. metadata.headline_baseline_override → return that name with
         rationale 'explicit_override' (only if present in policy_results).
      2. metadata.is_telemetry_failsafe → 'fifo' (correctness measured by KEEP).
      3. optimization_intent == 'fragmentation_packing' → packing primitive
         from policy_results, or honest disclaimer when none were computed
         (ISSUE 4).
      4. optimization_intent == 'energy_arbitrage' → strongest safe of
         (current_price_only, greedy_energy, sla_aware) via the safety filter.
      5. inference_critical / inference_standard: prefer 'sla_aware' if safe;
         else 'fifo'.
      6. batch_inference / batch_training / embedding_offline / fine_tuning:
         pick the strongest of the relevant_baselines by
         sla_safe_goodput_per_infra_dollar, EXCLUDING any candidate whose
         sla_violations > fifo's OR p99 > 1.5x fifo's. If all candidates
         fail safety, return
         ('fifo', 'headline_baseline_disqualified_for_safety') (ISSUE 2).
      7. Fallback: strongest safe candidate else fifo.
    """
    if metadata is None:
        return ("fifo", "no_metadata")

    fifo = policy_results.get("fifo")

    # Rule 1 — explicit override
    override = metadata.headline_baseline_override
    if override:
        if override in policy_results:
            return (override, "explicit_override")
        # Override missing from results → fall through with note appended.

    # Rule 2 — telemetry failsafe
    if metadata.is_telemetry_failsafe:
        return ("fifo", "telemetry_failsafe_correctness")

    # Rule 3 — fragmentation_packing intent. ISSUE 4: if no packing primitive
    # was computed in this run, return an honest disclaimer instead of
    # silently claiming "FIFO was strongest" — packing scenarios can't be
    # judged against the standard 5 policies.
    if metadata.optimization_intent == "fragmentation_packing":
        packing_names = ("first_fit", "best_fit", "first_fit_decreasing")
        best = None
        best_val = -math.inf
        any_present = False
        for n in packing_names:
            r = policy_results.get(n)
            if r is None:
                continue
            any_present = True
            v = _goodput_per_dollar(r)
            if v is not None and v > best_val:
                best, best_val = n, v
        if best is not None:
            return (best, "best_packing_baseline_by_goodput_per_dollar")
        if not any_present:
            # The runner did not compute packing primitives for this scenario.
            # Surface the gap explicitly instead of hiding it behind FIFO.
            return ("fifo", "no_packing_baseline_computed_for_this_run")
        # Packing primitives were present but had no goodput value — fall
        # through to general fallback (same as before).

    # Rule 4 — energy_arbitrage intent: pick strongest safe energy/sla
    # candidate from the standard set. Same safety filter as Rule 6.
    if metadata.optimization_intent == "energy_arbitrage":
        candidates = ("current_price_only", "greedy_energy", "sla_aware")
        best_e = None
        best_e_val = -math.inf
        any_disqualified_e = False
        for n in candidates:
            cand = policy_results.get(n)
            if cand is None:
                continue
            v = _goodput_per_dollar(cand)
            if v is None:
                continue
            if not _is_candidate_safe(cand, fifo):
                any_disqualified_e = True
                continue
            if v > best_e_val:
                best_e, best_e_val = n, v
        if best_e is not None:
            return (best_e, f"strongest_safe_relevant_baseline:{best_e}")
        if any_disqualified_e:
            return ("fifo", "headline_baseline_disqualified_for_safety")
        # Fall through if no candidate had a goodput value.

    # Rule 5 — interactive workloads prefer SLA-aware if safe
    if metadata.primary_workload_type in (
        "inference_critical", "inference_standard",
    ):
        sla = policy_results.get("sla_aware")
        if sla is not None and _is_candidate_safe(sla, fifo):
            return ("sla_aware", "interactive_workload_prefers_sla_aware")
        return ("fifo", "sla_aware_failed_safety_falls_back_to_fifo")

    # Rule 6 / 7 — strongest safe non-oracle candidate from relevant_baselines.
    # ISSUE 2: skip "fifo" (and "clairvoyant_lower_bound") in the safe-candidate
    # loop so we can distinguish "no non-fifo candidate cleared safety" from
    # "fifo was the strongest" — and emit a correct rationale either way.
    best = None
    best_val = -math.inf
    any_non_fifo_present = False
    any_non_fifo_disqualified = False
    for name in metadata.relevant_baselines:
        if name in ("fifo", "clairvoyant_lower_bound"):
            # fifo cannot be its own "best safe candidate"; clairvoyant is
            # an oracle and not deployable. Both are excluded from this loop.
            continue
        cand = policy_results.get(name)
        if cand is None:
            continue
        v = _goodput_per_dollar(cand)
        if v is None:
            continue
        any_non_fifo_present = True
        if not _is_candidate_safe(cand, fifo):
            any_non_fifo_disqualified = True
            continue
        if v > best_val:
            best, best_val = name, v
    if best is not None:
        return (best, f"strongest_safe_relevant_baseline:{best}")
    if any_non_fifo_disqualified:
        return ("fifo", "headline_baseline_disqualified_for_safety")
    if not any_non_fifo_present:
        return ("fifo", "no_relevant_baseline_with_goodput_value")
    return ("fifo", "no_relevant_baseline_with_goodput_value")


# ---------------------------------------------------------------------------
# Outcome analysis (alpha vs safety)
# ---------------------------------------------------------------------------

def _engine_blocked_count(scorecard_flags) -> int:
    """Count how many scorecard.flags indicate over-conservative gating."""
    # The runner doesn't directly surface blocked counts via scorecard.flags;
    # this helper is conservative — it returns 0 if the flags don't include
    # known blocked markers. The real check uses ca_kpi.blocked_* fields.
    return 0


def analyze_outcome(metadata: ScenarioMetadata,
                    ca_kpi: Any,
                    headline_kpi: Any,
                    all_baseline_kpis: Mapping[str, Any],
                    *, scorecard_flags=(),
                    headline_name: Optional[str] = None) -> OutcomeAnalysis:
    """Combines outcome classification and loss-reason detection.

    See OUTCOMES, LOSS_REASON_CODES. Ordering (per critic 5):
      1. telemetry_failsafe + KEEP rate matches FIFO + no SLA regression
         → KEEP_CORRECT.
      2. SAFETY_WIN: |margin_pct| <= 1% AND materially-better safety
         (p99 / sla / thermal <= 0.5 × strongest_baseline).
      3. ALPHA_WIN: margin_pct > 1%.
      4. TIE: |margin_pct| <= 1%.
      5. LOSS: margin_pct < -1%.
    """
    if metadata is None or ca_kpi is None or headline_kpi is None:
        return OutcomeAnalysis(
            outcome="TIE", margin_pct=0.0, safety_evidence=(),
            loss_reasons=(), notes="missing kpi inputs",
        )

    ca_goodput = _goodput_per_dollar(ca_kpi)
    hd_goodput = _goodput_per_dollar(headline_kpi)
    if ca_goodput is None or hd_goodput is None or hd_goodput <= 0:
        margin_pct = 0.0
    else:
        margin_pct = (ca_goodput - hd_goodput) / hd_goodput * 100.0

    # Strongest baseline values across ALL baselines (used by SAFETY_WIN).
    def _strongest(getter, default):
        vals = []
        for name, k in all_baseline_kpis.items():
            if name == "constraint_aware":
                continue
            v = getter(k)
            if v is not None:
                vals.append(v)
        return min(vals) if vals else default

    strongest_p99 = _strongest(_p99, None)
    strongest_sla = _strongest(_sla, None)
    strongest_thr = _strongest(_thermal, None)

    safety_evidence: list = []
    ca_p99 = _p99(ca_kpi)
    ca_sla_v = _sla(ca_kpi)
    ca_thr = _thermal(ca_kpi)
    if (ca_p99 is not None and strongest_p99 is not None
            and strongest_p99 > 0 and ca_p99 <= 0.5 * strongest_p99):
        safety_evidence.append(
            f"p99 {ca_p99:.0f}ms <= 0.5 x strongest baseline {strongest_p99:.0f}ms"
        )
    if (strongest_sla is not None
            and ca_sla_v <= 0.5 * strongest_sla and strongest_sla > 0):
        safety_evidence.append(
            f"sla_violations {ca_sla_v} <= 0.5 x strongest {strongest_sla}"
        )
    if (strongest_thr is not None
            and ca_thr <= 0.5 * strongest_thr and strongest_thr > 0):
        safety_evidence.append(
            f"thermal_throttle_ticks {ca_thr} <= 0.5 x strongest {strongest_thr}"
        )

    # 1) KEEP_CORRECT for telemetry-failsafe scenarios.
    if metadata.is_telemetry_failsafe:
        fifo = all_baseline_kpis.get("fifo")
        fifo_sla = _sla(fifo) if fifo is not None else 0
        within_tie = abs(margin_pct) <= _MATERIAL * 100.0
        if ca_sla_v <= fifo_sla and within_tie:
            return OutcomeAnalysis(
                outcome="KEEP_CORRECT", margin_pct=margin_pct,
                safety_evidence=tuple(safety_evidence),
                loss_reasons=(),
                notes="telemetry-failsafe scenario; KEEP matched FIFO "
                      "and no SLA regression",
            )
        # Falls through if KEEP_CORRECT condition is not met.

    # 2) SAFETY_WIN
    within_tie = abs(margin_pct) <= _MATERIAL * 100.0
    if within_tie and safety_evidence:
        return OutcomeAnalysis(
            outcome="SAFETY_WIN", margin_pct=margin_pct,
            safety_evidence=tuple(safety_evidence),
            loss_reasons=(),
            notes="alpha tied, but safety materially better",
        )

    # 3) ALPHA_WIN
    if margin_pct > _MATERIAL * 100.0:
        return OutcomeAnalysis(
            outcome="ALPHA_WIN", margin_pct=margin_pct,
            safety_evidence=tuple(safety_evidence),
            loss_reasons=(),
            notes=f"constraint_aware beat headline by {margin_pct:+.2f}%",
        )

    # 4) TIE
    if within_tie:
        return OutcomeAnalysis(
            outcome="TIE", margin_pct=margin_pct,
            safety_evidence=tuple(safety_evidence),
            loss_reasons=(),
            notes="within tie band, no material safety edge",
        )

    # 5) LOSS — populate loss_reasons (multi-cause)
    reasons: list = []

    # telemetry_fail_safe
    if metadata.is_telemetry_failsafe:
        reasons.append("telemetry_fail_safe")

    # missing_candidate_action — CA produced no non-noop recommendations of
    # the action type relevant to the intent.
    intent_to_action = {
        "energy_arbitrage": ("migrate",),
        "thermal_spread": ("migrate",),
        "queue_relief": ("scale_replicas",),
        "memory_pressure_relief": ("scale_replicas", "migrate"),
        "topology_fit": ("migrate",),
        "fragmentation_packing": ("consolidate", "migrate"),
        "safety_keep": (),
        "completion_time": ("scale_replicas",),
    }
    relevant_actions = intent_to_action.get(metadata.optimization_intent, ())
    if relevant_actions:
        scale_recommended = int(_kpi_field(ca_kpi, "scale_up_recommended", 0) or 0)
        scale_applied = int(_kpi_field(ca_kpi, "scale_up_applied", 0) or 0)
        migrations = int(_kpi_field(ca_kpi, "total_migrations", 0) or 0)
        any_applied = (
            (("scale_replicas" in relevant_actions)
                and (scale_recommended + scale_applied) > 0)
            or (("migrate" in relevant_actions) and migrations > 0)
            or (("consolidate" in relevant_actions) and migrations > 0)
        )
        if not any_applied:
            reasons.append("missing_candidate_action")

    # over_conservative_gate
    blk_low = int(
        _kpi_field(ca_kpi, "blocked_scale_for_low_value_queue_relief", 0) or 0
    )
    hd_scale_app = int(_kpi_field(headline_kpi, "scale_up_applied", 0) or 0)
    if blk_low > 0 and hd_scale_app > 0:
        reasons.append("over_conservative_gate")

    # wrong_workload_classification (ISSUE 5): the engine treated an
    # interactive workload as if it were batch — gated scale-up actions
    # using the low-value-queue-relief check fired despite the workload
    # actually being interactive (critical/standard inference).
    if (blk_low > 0 and metadata.primary_workload_type in (
            "inference_critical", "inference_standard")):
        reasons.append("wrong_workload_classification")

    # under_modeled_action_effect (ISSUE 5): CA degraded topology score
    # materially below the headline, indicating an under-modeled effect
    # of the action set the simulator actually executes.
    ca_topo = _kpi_field(ca_kpi, "mean_topology_score", None)
    hd_topo = _kpi_field(headline_kpi, "mean_topology_score", None)
    if (ca_topo is not None and hd_topo is not None
            and ca_topo <= 0.5 and hd_topo >= 0.8):
        reasons.append("under_modeled_action_effect")

    # simulator_limitation: packing scenarios where headline is a packing primitive
    if metadata.optimization_intent == "fragmentation_packing":
        # Headline names below are packing primitives
        # (the simulator has no arbitrary-placement primitive in CA actions).
        reasons.append("simulator_limitation")

    # missing_forecast_lookahead — ISSUE 7: only meaningful when the headline
    # is itself an energy baseline. Otherwise the comparison isn't a
    # forecast/lookahead test and we'd be misattributing the loss.
    if metadata.optimization_intent == "energy_arbitrage":
        net_savings = _kpi_field(ca_kpi, "total_net_savings", None)
        if (net_savings is not None and net_savings <= 0
                and headline_name in ("current_price_only", "greedy_energy")):
            reasons.append("missing_forecast_lookahead")

    # scenario_not_applicable (ISSUE 5): catch-all for mixed workloads where
    # none of the SLA-risk constraints were materially active — the scenario
    # simply doesn't exercise any CA action and the loss isn't a real signal.
    # Checked BEFORE the missing_candidate_action fallback so it can fire as
    # the primary reason.
    if not reasons and metadata.primary_workload_type == "mixed":
        ca_p99v = _p99(ca_kpi)
        ca_qw = _kpi_field(ca_kpi, "p95_queue_wait_ms", None)
        ca_sla_count = _sla(ca_kpi)
        no_sla_risk = (
            (ca_p99v is None or ca_p99v < 500)
            and (ca_qw is None or ca_qw < 500)
            and ca_sla_count == 0
        )
        if no_sla_risk:
            reasons.append("scenario_not_applicable")

    if not reasons:
        reasons.append("missing_candidate_action")

    primary_notes = {
        "missing_candidate_action":
            "constraint_aware emitted no relevant action type",
        "over_conservative_gate":
            "constraint_aware blocked actions the headline applied",
        "simulator_limitation":
            "fragmentation_packing intent vs packing-primitive headline; "
            "simulator lacks an arbitrary-placement primitive",
        "telemetry_fail_safe":
            "telemetry-failsafe correctness path overrode alpha",
        "missing_forecast_lookahead":
            "energy_arbitrage with no positive net_savings (no DA/RT lookahead)",
        "wrong_workload_classification":
            "engine treated an interactive workload as batch; "
            "low-value queue-relief gate fired inappropriately",
        "under_modeled_action_effect":
            "CA degraded topology score materially vs the headline; "
            "the action set has an under-modeled effect on topology",
        "scenario_not_applicable":
            "mixed workload with no active SLA-risk constraint; "
            "scenario does not exercise any CA action",
    }
    primary = reasons[0]
    return OutcomeAnalysis(
        outcome="LOSS", margin_pct=margin_pct,
        safety_evidence=tuple(safety_evidence),
        loss_reasons=tuple(reasons),
        notes=primary_notes.get(primary, "loss with unclassified cause"),
    )


# ---------------------------------------------------------------------------
# Per-scenario and cross-scenario aggregation
# ---------------------------------------------------------------------------

@dataclass
class PerScenarioRow:
    """A single-scenario summary row used in the cross-scenario report."""
    scenario_name: str
    metadata: ScenarioMetadata
    headline_baseline_name: str
    headline_baseline_rationale: str
    outcome: OutcomeAnalysis
    policy_goodput_per_dollar: dict = field(default_factory=dict)
    raw_cost: dict = field(default_factory=dict)
    gpu_infra_cost: dict = field(default_factory=dict)
    energy_cost: dict = field(default_factory=dict)
    p99_latency_ms: dict = field(default_factory=dict)
    queue_p95_ms: dict = field(default_factory=dict)
    sla_violations: dict = field(default_factory=dict)
    sla_compliant_goodput: dict = field(default_factory=dict)
    total_infrastructure_cost: dict = field(default_factory=dict)


_POLICY_LABELS = (
    ("fifo", "FIFO"),
    ("current_price_only", "current_price_only"),
    ("greedy_energy", "greedy_energy"),
    ("sla_aware", "SLA-aware"),
    ("constraint_aware", "constraint_aware"),
)


@dataclass
class CrossScenarioReport:
    """Renders the four tables (A overall, B per-workload, C per-scenario,
    D baseline-strength) used in the buyer-facing markdown.
    """
    rows: list = field(default_factory=list)

    @classmethod
    def from_results(cls, benchmark_results: Mapping[str, Any]) -> "CrossScenarioReport":
        """Build rows from {scenario_name: BenchmarkResult}."""
        rows: list = []
        for name, br in benchmark_results.items():
            report = getattr(br, "report", None)
            if report is None:
                continue
            agg = getattr(report, "aggregated", None) or {}
            if not agg:
                continue
            metadata = getattr(report, "scenario_metadata", None)
            if metadata is None:
                # Try via the scenario object on the result, or build a
                # best-effort one from expected_primary_constraint.
                expected = getattr(report, "expected_primary_constraint", None)
                metadata = classify_scenario(name, expected, {})
            headline_name = getattr(report, "headline_baseline_name", None)
            headline_rationale = getattr(
                report, "headline_baseline_rationale", "",
            )
            outcome = getattr(report, "outcome", None)
            if headline_name is None:
                headline_name, headline_rationale = select_headline_baseline(
                    metadata, agg,
                )
            if outcome is None:
                hd = agg.get(headline_name, agg.get("fifo"))
                outcome = analyze_outcome(
                    metadata, agg.get("constraint_aware"), hd, agg,
                    headline_name=headline_name,
                )

            row = PerScenarioRow(
                scenario_name=name,
                metadata=metadata,
                headline_baseline_name=headline_name,
                headline_baseline_rationale=headline_rationale,
                outcome=outcome,
                policy_goodput_per_dollar={
                    p: _goodput_per_dollar(agg.get(p)) for p, _ in _POLICY_LABELS
                },
                raw_cost={
                    p: _kpi_field(agg.get(p), "total_energy_cost", None)
                    for p, _ in _POLICY_LABELS
                },
                gpu_infra_cost={
                    p: _kpi_field(agg.get(p), "gpu_infra_cost", None)
                    for p, _ in _POLICY_LABELS
                },
                energy_cost={
                    p: _kpi_field(agg.get(p), "energy_cost", None)
                    for p, _ in _POLICY_LABELS
                },
                p99_latency_ms={
                    p: _kpi_field(agg.get(p), "p99_latency_ms", None)
                    for p, _ in _POLICY_LABELS
                },
                queue_p95_ms={
                    p: _kpi_field(agg.get(p), "p95_queue_wait_ms", None)
                    for p, _ in _POLICY_LABELS
                },
                sla_violations={
                    p: _kpi_field(agg.get(p), "total_sla_violations", None)
                    for p, _ in _POLICY_LABELS
                },
                sla_compliant_goodput={
                    p: _kpi_field(agg.get(p), "sla_compliant_goodput", None)
                    for p, _ in _POLICY_LABELS
                },
                total_infrastructure_cost={
                    p: _kpi_field(agg.get(p), "total_infrastructure_cost", None)
                    for p, _ in _POLICY_LABELS
                },
            )
            rows.append(row)
        return cls(rows=rows)

    @property
    def telemetry_failsafe_rows(self) -> list:
        return [r for r in self.rows if r.metadata.is_telemetry_failsafe]

    @property
    def economic_rows(self) -> list:
        return [r for r in self.rows if not r.metadata.is_telemetry_failsafe]

    # ------------------------------------------------------------------ #
    # Alpha/safety counters — ISSUE 6.                                    #
    # ------------------------------------------------------------------ #
    @property
    def alpha_win_count(self) -> int:
        """Scenarios where constraint_aware beat the headline by >1%."""
        return sum(1 for r in self.rows if r.outcome.outcome == "ALPHA_WIN")

    @property
    def safety_win_count(self) -> int:
        """Scenarios in the tie band where CA had materially better safety."""
        return sum(1 for r in self.rows if r.outcome.outcome == "SAFETY_WIN")

    @property
    def correct_keep_count(self) -> int:
        """Telemetry-failsafe scenarios where CA correctly held with FIFO."""
        return sum(1 for r in self.rows if r.outcome.outcome == "KEEP_CORRECT")

    @property
    def economic_loss_count(self) -> int:
        """Scenarios where CA lost the alpha race by >1%."""
        return sum(1 for r in self.rows if r.outcome.outcome == "LOSS")

    @property
    def sla_regression_count(self) -> int:
        """Scenarios where CA produced more SLA violations than FIFO."""
        n = 0
        for r in self.rows:
            ca = r.sla_violations.get("constraint_aware")
            fifo = r.sla_violations.get("fifo")
            if ca is not None and fifo is not None and ca > fifo:
                n += 1
        return n

    @property
    def catastrophic_baseline_avoidance_count(self) -> int:
        """Scenarios where an aggressive baseline (greedy_energy or
        current_price_only) had p99 >= 2x CA's, regardless of outcome label.
        Counts the buyer-relevant 'CA avoided a baseline blow-up' result.
        """
        n = 0
        for r in self.rows:
            ca_p99 = r.p99_latency_ms.get("constraint_aware") or 0
            if ca_p99 == 0:
                continue
            for b in ("greedy_energy", "current_price_only"):
                bp = r.p99_latency_ms.get(b) or 0
                if bp >= 2 * ca_p99:
                    n += 1
                    break
        return n

    def to_markdown(self) -> str:
        """Renders the four required sections.

        - A. Overall policy table (mean+median goodput/$, wins, losses).
        - B. Per-workload-type table (mean and median goodput/$ per policy).
        - C. Per-scenario table including outcome + loss_reasons.
        - D. Baseline strength per scenario.
        Telemetry-failsafe scenarios get their own subsection so they don't
        distort economic averages.
        """
        lines: list = []
        # Section A: Overall policy
        lines.append("## A. Overall policy comparison")
        lines.append("")
        lines.append(
            "Median is the headline aggregate (robust to scenario heterogeneity); "
            "mean is shown as a secondary number. Telemetry-failsafe scenarios "
            "are excluded from these economic aggregates and reported separately "
            "(see end of section A).")
        lines.append("")
        lines.append(
            "| Policy | Mean goodput/$ | Median goodput/$ | "
            "CA ALPHA_WIN when this=headline | CA SAFETY_WIN when this=headline "
            "| SLA regressions vs FIFO |"
        )
        lines.append("|---|---|---|---|---|---|")
        for policy, label in _POLICY_LABELS:
            vals = [r.policy_goodput_per_dollar.get(policy)
                    for r in self.economic_rows]
            vals = [v for v in vals if v is not None]
            mean_v = (f"{statistics.mean(vals):,.0f}" if vals else "—")
            med_v = (f"{statistics.median(vals):,.0f}" if vals else "—")
            wins = sum(
                1 for r in self.economic_rows
                if r.headline_baseline_name == policy
                and r.outcome.outcome == "ALPHA_WIN"
            )
            safe = sum(
                1 for r in self.economic_rows
                if r.headline_baseline_name == policy
                and r.outcome.outcome == "SAFETY_WIN"
            )
            sla_reg = sum(
                1 for r in self.economic_rows
                if (r.sla_violations.get(policy) or 0) >
                   (r.sla_violations.get("fifo") or 0)
            )
            lines.append(
                f"| {label} | {mean_v} | {med_v} | {wins} | {safe} | {sla_reg} |"
            )
        lines.append("")
        # Alpha/safety counters block (ISSUE 6).
        lines.append(
            f"**Alpha/safety counters:** "
            f"alpha_wins={self.alpha_win_count} · "
            f"safety_wins={self.safety_win_count} · "
            f"correct_keeps={self.correct_keep_count} · "
            f"economic_losses={self.economic_loss_count} · "
            f"SLA_regressions={self.sla_regression_count} · "
            f"catastrophic_baseline_avoidances="
            f"{self.catastrophic_baseline_avoidance_count}"
        )
        lines.append("")
        if self.telemetry_failsafe_rows:
            lines.append(
                "Telemetry-failsafe scenarios (KEEP-correctness, not alpha): "
                + ", ".join(r.scenario_name
                            for r in self.telemetry_failsafe_rows)
            )
            lines.append("")

        # Section B: Per workload type
        lines.append("## B. Per-workload-type comparison")
        lines.append("")
        wt_buckets: dict = {}
        for r in self.economic_rows:
            wt_buckets.setdefault(r.metadata.primary_workload_type, []).append(r)
        lines.append("| Workload type | Scenarios | Policy | Mean goodput/$ | "
                     "Median goodput/$ |")
        lines.append("|---|---|---|---|---|")
        for wt in sorted(wt_buckets.keys()):
            bucket = wt_buckets[wt]
            for policy, label in _POLICY_LABELS:
                vals = [r.policy_goodput_per_dollar.get(policy) for r in bucket]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                mean_v = f"{statistics.mean(vals):,.0f}"
                med_v = f"{statistics.median(vals):,.0f}"
                lines.append(
                    f"| {wt} | {len(bucket)} | {label} | {mean_v} | {med_v} |"
                )
        lines.append("")

        # Section C: per-scenario
        lines.append("## C. Per-scenario outcome")
        lines.append("")
        lines.append("Headline-baseline column is the *workload-relevant strong "
                     "baseline*, not FIFO. Outcome compares constraint_aware "
                     "against that headline.")
        lines.append("")
        lines.append("| scenario | workload type | intent | goodput_unit | "
                     "headline baseline | rationale | outcome | margin % | "
                     "loss reasons | notes |")
        lines.append("|" + "---|" * 10)
        for r in self.rows:
            lr = ", ".join(r.outcome.loss_reasons) if r.outcome.loss_reasons else "—"
            lines.append(
                f"| {r.scenario_name} | {r.metadata.primary_workload_type} "
                f"| {r.metadata.optimization_intent} | {r.metadata.goodput_unit} "
                f"| {r.headline_baseline_name} | {r.headline_baseline_rationale} "
                f"| {r.outcome.outcome} | {r.outcome.margin_pct:+.2f} | {lr} "
                f"| {r.outcome.notes} |"
            )
        lines.append("")

        # Section D: baseline strength per scenario
        lines.append("## D. Baseline strength per scenario")
        lines.append("")
        lines.append("Per-policy goodput/$ for every scenario, so reviewers can "
                     "see which baseline was strongest and whether the headline "
                     "selection was reasonable.")
        lines.append("")
        lines.append("| scenario | FIFO | current_price_only | greedy_energy | "
                     "SLA-aware | constraint_aware |")
        lines.append("|---|---|---|---|---|---|")
        for r in self.rows:
            def _fmt(p):
                v = r.policy_goodput_per_dollar.get(p)
                return f"{v:,.0f}" if v is not None else "—"
            lines.append(
                f"| {r.scenario_name} | {_fmt('fifo')} | "
                f"{_fmt('current_price_only')} | {_fmt('greedy_energy')} | "
                f"{_fmt('sla_aware')} | {_fmt('constraint_aware')} |"
            )
        lines.append("")

        return "\n".join(lines)
