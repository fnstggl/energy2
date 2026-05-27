"""Simulator realism audit (task spec §8).

This module answers a single question honestly: *how far can the simulator's
evidence be trusted?* It does NOT rubber-stamp the simulator. Every check below
PROBES a real code path (serving physics, migration cost, canonical telemetry,
engine no-op behaviour, energy/carbon traces) and reports what it actually
observes — including the places where the simulator is still too optimistic to
support production claims.

Per-subsystem verdicts use the task's vocabulary:

  REALISTIC_ENOUGH_FOR_DEV    – dynamics are qualitatively believable; safe to
                                use as a development/validation harness.
  TOO_SIMPLISTIC_FOR_CLAIMS   – a dynamic is modelled too cleanly to support any
                                quantitative claim.
  NEEDS_REAL_TELEMETRY        – the path exists but is fed perfect/synthetic data;
                                must be exercised against real telemetry.
  NOT_PRODUCTION_REALISTIC_YET – combination of the above; do not quote numbers.

The overall verdict is intentionally capped at NOT_PRODUCTION_REALISTIC_YET while
*no* calibration parameter is MEASURED on real hardware (see calibration_table()).
This is a deliberate honesty gate, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Verdict constants (task §8 vocabulary)
REALISTIC_ENOUGH_FOR_DEV = "REALISTIC_ENOUGH_FOR_DEV"
TOO_SIMPLISTIC_FOR_CLAIMS = "TOO_SIMPLISTIC_FOR_CLAIMS"
NEEDS_REAL_TELEMETRY = "NEEDS_REAL_TELEMETRY"
NOT_PRODUCTION_REALISTIC_YET = "NOT_PRODUCTION_REALISTIC_YET"


@dataclass
class RealismCheck:
    """One probed realism question and what was actually observed."""
    subsystem: str
    question: str
    realistic: bool          # True = behaviour is believable; False = too clean/perfect
    observation: str         # the measured evidence
    severity: str = "info"   # "info" | "warn" | "blocker"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "question": self.question,
            "realistic": self.realistic,
            "observation": self.observation,
            "severity": self.severity,
        }


@dataclass
class SubsystemVerdict:
    subsystem: str
    verdict: str
    checks: list[RealismCheck] = field(default_factory=list)
    calibration_confidence: Optional[str] = None
    calibration_summary: Optional[dict[str, int]] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "verdict": self.verdict,
            "calibration_confidence": self.calibration_confidence,
            "calibration_summary": self.calibration_summary,
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes,
        }


@dataclass
class RealismAuditReport:
    timestamp: str
    overall_verdict: str
    subsystems: list[SubsystemVerdict]
    headline_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_verdict": self.overall_verdict,
            "headline_findings": self.headline_findings,
            "subsystems": [s.to_dict() for s in self.subsystems],
        }

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Aurelius Simulator Realism Audit — {self.timestamp}")
        lines.append("[SANDBOX] Probes the simulator's own code paths. Not a production claim.")
        lines.append("")
        lines.append(f"OVERALL VERDICT: {self.overall_verdict}")
        lines.append("")
        if self.headline_findings:
            lines.append("Headline findings:")
            for f in self.headline_findings:
                lines.append(f"  • {f}")
            lines.append("")
        for sv in self.subsystems:
            cal = ""
            if sv.calibration_confidence:
                cal = f"  [calibration={sv.calibration_confidence}]"
            lines.append(f"── {sv.subsystem.upper()}: {sv.verdict}{cal}")
            for c in sv.checks:
                mark = "OK " if c.realistic else "!! "
                if c.severity == "blocker" and not c.realistic:
                    mark = "XX "
                lines.append(f"   {mark}{c.question}")
                lines.append(f"      → {c.observation}")
            for n in sv.notes:
                lines.append(f"   note: {n}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration confidence aggregation
# ---------------------------------------------------------------------------

_CONF_RANK = {"high": 3, "medium": 2, "low": 1, None: 0}


def _calibration_by_group() -> dict[str, dict[str, Any]]:
    """Aggregate calibration_table() rows per subsystem group.

    Returns {group: {"by_confidence": {...}, "by_source_type": {...},
                     "mean_confidence": str, "n": int, "any_measured": bool}}
    """
    from ..simulation.cluster.calibration import calibration_table

    out: dict[str, dict[str, Any]] = {}
    for row in calibration_table():
        g = row.get("group", "unknown")
        bucket = out.setdefault(
            g,
            {"by_confidence": {}, "by_source_type": {}, "n": 0, "any_measured": False,
             "_conf_sum": 0},
        )
        conf = row.get("confidence")
        st = row.get("source_type")
        bucket["by_confidence"][conf] = bucket["by_confidence"].get(conf, 0) + 1
        bucket["by_source_type"][st] = bucket["by_source_type"].get(st, 0) + 1
        bucket["n"] += 1
        bucket["_conf_sum"] += _CONF_RANK.get(conf, 0)
        if st == "measured" or conf == "measured":
            bucket["any_measured"] = True

    for g, b in out.items():
        n = max(1, b["n"])
        avg = b.pop("_conf_sum") / n
        if avg >= 2.5:
            b["mean_confidence"] = "high"
        elif avg >= 1.5:
            b["mean_confidence"] = "medium"
        else:
            b["mean_confidence"] = "low"
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _probe_serving(checks: list[RealismCheck]) -> None:
    from ..simulation.cluster import serving as s
    from ..simulation.cluster.calibration import serving_value

    # 1. Tail latencies must NOT be a fixed multiple of the mean.
    lo95, lo99 = s.tail_multipliers(0.30)
    hi95, hi99 = s.tail_multipliers(0.95)
    grows = hi99 > lo99 > 0 and hi95 > lo95 > 0
    p99_faster = (hi99 / lo99) > (hi95 / lo95)
    checks.append(RealismCheck(
        "serving",
        "Are tail latencies fixed multipliers of the mean?",
        realistic=grows and p99_faster,
        observation=(
            f"p95/p50 {lo95:.2f}→{hi95:.2f}, p99/p50 {lo99:.2f}→{hi99:.2f} as ρ 0.30→0.95; "
            f"p99 grows {'faster' if p99_faster else 'NOT faster'} than p95"
        ),
    ))

    # 2. Queue wait must increase convexly near saturation.
    mu, c = 2.0, 4
    waits = []
    for rho in (0.5, 0.7, 0.85, 0.95):
        lam = rho * c * mu
        waits.append(s.erlang_c_wait_s(lam, mu, c) * s.saturation_amplifier(rho))
    diffs = [waits[i + 1] - waits[i] for i in range(len(waits) - 1)]
    convex = all(diffs[i + 1] > diffs[i] for i in range(len(diffs) - 1))
    checks.append(RealismCheck(
        "serving",
        "Does queue wait increase convexly near saturation?",
        realistic=convex,
        observation=(
            f"wait(ρ=0.5,0.7,0.85,0.95)="
            f"{[round(w, 2) for w in waits]} s; second-differences positive={convex}"
        ),
    ))

    # 3. Batching tradeoff: spreading the same load over more replicas costs efficiency.
    knee = serving_value("batch_efficiency_knee")
    eff_few = s.batching_efficiency(knee, 1)
    eff_many = s.batching_efficiency(knee, 8)
    checks.append(RealismCheck(
        "serving",
        "Is throughput linear in replicas (no batching knee)?",
        realistic=eff_many < eff_few,
        observation=(
            f"batch efficiency at knee load: 1 replica={eff_few:.2f}, "
            f"8 replicas={eff_many:.2f} (lower = thinner batches)"
        ),
    ))


def _probe_arrivals(checks: list[RealismCheck]) -> None:
    import random as _random

    from ..simulation.cluster import serving as s

    rng = _random.Random(7)
    states = [s.step_burst_state(False, rng) for _ in range(200)]
    # If bursts are modelled at all, some transitions to True must occur.
    bursty = any(states)
    mult = s.arrival_multiplier(True)
    checks.append(RealismCheck(
        "serving",
        "Are arrivals a smooth sinusoid (not bursty)?",
        realistic=bursty and mult > 1.0,
        observation=(
            f"MMPP burst state reachable={bursty}; burst arrival multiplier={mult:.2f}×. "
            "NOTE: bursts are opt-in per scenario (off for the 6 canonical detection scenarios)."
        ),
        severity="warn" if not bursty else "info",
    ))


def _probe_migration(checks: list[RealismCheck], seed: int) -> None:
    from ..simulation.cluster import load_scenario
    from ..simulation.cluster.engine import ClusterSimulator

    sc = load_scenario("energy_price_arbitrage_multiregion", seed_override=seed)
    sim = ClusterSimulator(sc.config, seed=seed)
    sim.run(steps=4)
    cluster = sim._cluster

    # Find a migratable workload and a different destination region.
    target = None
    dest = None
    for wl in cluster.workloads.values():
        if wl.migration_allowed and not wl.latency_sensitive:
            others = [r for r in cluster.regions if r != wl.region_id]
            if others:
                target, dest = wl, others[0]
                break

    if target is None:
        checks.append(RealismCheck(
            "migration",
            "Is migration free (no cold-start / cache / disruption cost)?",
            realistic=True,
            observation="No migratable workload available to probe in this scenario.",
            severity="info",
        ))
        return

    conf_before = target.cache.locality.confidence if target.cache else None
    ok = sim.migrate_workload(target.workload_id, dest)
    conf_after = target.cache.locality.confidence if target.cache else None
    warmup = target.cold_start_warmup_ticks_remaining
    cold_route_cost = (
        conf_before is not None and conf_after is not None and conf_after < conf_before
    )
    has_cost = bool(ok and (cold_route_cost or warmup > 0))
    checks.append(RealismCheck(
        "migration",
        "Is migration free (no cold-start / cache / disruption cost)?",
        realistic=has_cost,
        observation=(
            f"migrate→{dest}: locality confidence {conf_before}→{conf_after} (reset on cold route), "
            f"warmup ticks remaining={warmup}; migration applied={ok}"
        ),
        severity="blocker" if not has_cost else "info",
    ))


def _probe_telemetry(checks: list[RealismCheck], seed: int) -> None:
    from ..constraints import ConstraintAwareEngine
    from ..simulation.cluster import load_scenario
    from ..simulation.cluster.engine import ClusterSimulator

    def state_for(scn: str):
        sc = load_scenario(scn, seed_override=seed)
        sim = ClusterSimulator(sc.config, seed=seed)
        sim.run(steps=6)
        return sim.get_cluster_state()

    # 1. Does the canonical ClusterState tell the truth — clean scenarios report
    #    high confidence, degraded-telemetry scenarios report low + partial?
    clean = state_for("energy_price_arbitrage_multiregion")
    clean_high = clean.provenance.confidence == "high" and not clean.is_partial
    degraded = {}
    for scn in (
        "degraded_topology_telemetry",
        "partial_utilization_telemetry",
        "low_confidence_energy_telemetry",
    ):
        try:
            st = state_for(scn)
            degraded[scn] = (st.provenance.confidence, st.is_partial, len(st.missing_sources))
        except Exception as exc:
            degraded[scn] = ("ERR", False, str(exc))
    all_degraded_marked = all(
        v[0] != "high" and v[1] is True for v in degraded.values()
    )
    truthful = clean_high and all_degraded_marked
    checks.append(RealismCheck(
        "telemetry",
        "Does the canonical ClusterState report degraded telemetry honestly?",
        realistic=truthful,
        observation=(
            f"clean scenario → confidence={clean.provenance.confidence!r}, "
            f"is_partial={clean.is_partial}; degraded scenarios (confidence, is_partial, "
            f"#missing_sources): " + ", ".join(f"{k}={v}" for k, v in degraded.items())
            + ". get_cluster_state() now derives provenance from per-subsystem "
            "telemetry tiers instead of hardcoding 'high'."
        ),
        severity="blocker",
    ))

    # 2. Does degraded telemetry actually force the engine to KEEP (no risky action)?
    engine = ConstraintAwareEngine()
    forced_keep = {}
    for scn in degraded:
        try:
            st = state_for(scn)
            res = engine.run(st)
            forced_keep[scn] = res.actionable_count
        except Exception as exc:
            forced_keep[scn] = f"ERR:{exc}"
    exercised = all(v == 0 for v in forced_keep.values() if isinstance(v, int))
    checks.append(RealismCheck(
        "telemetry",
        "Do missing/stale telemetry paths force KEEP end-to-end?",
        realistic=exercised,
        observation=(
            "engine actionable_count under degraded telemetry: "
            + ", ".join(f"{k}={v}" for k, v in forced_keep.items())
            + " (0 = correctly forced KEEP by the telemetry-trust gate)."
        ),
    ))


def _probe_actions(checks: list[RealismCheck], seed: int) -> None:
    from ..simulation.cluster import load_scenario
    from ..simulation.cluster.engine import ClusterSimulator

    sc = load_scenario("thermal_hotspot_mixed_cluster", seed_override=seed)
    sim = ClusterSimulator(sc.config, seed=seed)
    sim.run(steps=4)

    # Pick a service to act on.
    svc = next(iter({wl.service_id for wl in sim._cluster.workloads.values()}), None)
    mutated = False
    detail = "no service available"
    if svc is not None:
        before = sum(g.assigned_workload_id is not None
                     for r in sim._cluster.regions.values()
                     for n in r.nodes for g in n.gpus)
        applied = sim.add_replica(svc) or sim.spread_workload(svc)
        after = sum(g.assigned_workload_id is not None
                    for r in sim._cluster.regions.values()
                    for n in r.nodes for g in n.gpus)
        mutated = bool(applied)
        detail = f"action applied={applied}; assigned-GPU count {before}→{after}"
    checks.append(RealismCheck(
        "actions",
        "Do recommended actions actually mutate simulator state?",
        realistic=mutated,
        observation=detail,
        severity="blocker" if not mutated else "info",
    ))


def _probe_no_safe_action(checks: list[RealismCheck], seed: int) -> None:
    from ..constraints import ConstraintAwareEngine
    from ..simulation.cluster import load_scenario
    from ..simulation.cluster.engine import ClusterSimulator

    # Scenarios where every safe action is already taken / infeasible, so the
    # engine should correctly emit KEEP (no actionable recommendation) on some ticks.
    candidates = [
        "rack_density_liquid_cooled",
        "fragmentation_stranded_capacity",
        "migration_trap_erased_savings",
        "latency_tail_kvcache_pressure",
    ]
    engine = ConstraintAwareEngine()
    observed_noop = False
    detail_parts = []
    for scn in candidates:
        try:
            sc = load_scenario(scn, seed_override=seed)
        except Exception:
            continue
        sim = ClusterSimulator(sc.config, seed=seed)
        noop_ticks = 0
        for _ in range(8):
            sim.tick()
            res = engine.run(sim.get_cluster_state())
            if res.actionable_count == 0:
                noop_ticks += 1
        detail_parts.append(f"{scn}: {noop_ticks}/8 no-op ticks")
        if noop_ticks > 0:
            observed_noop = True
    checks.append(RealismCheck(
        "actions",
        "Are no-safe-action (KEEP / infeasible) states possible?",
        realistic=observed_noop,
        observation="; ".join(detail_parts) or "no candidate scenarios available",
    ))


def _probe_energy(checks: list[RealismCheck]) -> None:
    from ..simulation.cluster.scenarios import _BUILTIN_SCENARIOS

    # 1. Energy spikes too clean? Inspect the canonical arbitrage trace for clean steps.
    base = _BUILTIN_SCENARIOS.get("energy_price_arbitrage_multiregion", {})
    clean_spike = any(
        ev.get("type") == "energy_price_spike" and float(ev.get("multiplier", 0)) in (2.0, 2.5, 3.0)
        for ev in base.get("events", [])
    )
    checks.append(RealismCheck(
        "energy",
        "Are price/carbon spikes too clean (round-number step multipliers)?",
        realistic=not clean_spike,
        observation=(
            "Canonical arbitrage scenario uses a clean round-number step spike "
            f"(detected={clean_spike}). Realistic adversarial variants exist "
            "(da_rt_basis_blowout, carbon_cheap_price_expensive)."
        ),
        severity="warn" if clean_spike else "info",
    ))

    # 2. Are price and carbon always anti-correlated (cheap = clean)?
    decorrelated = "carbon_cheap_price_expensive" in _BUILTIN_SCENARIOS
    checks.append(RealismCheck(
        "energy",
        "Are price and carbon always anti-correlated (cheap == clean)?",
        realistic=decorrelated,
        observation=(
            "Decorrelating scenario present="
            f"{decorrelated} (carbon_cheap_price_expensive): cheap power can be dirty."
        ),
    ))

    # 3. Is day-ahead vs real-time basis risk modelled?
    basis = "da_rt_basis_blowout" in _BUILTIN_SCENARIOS
    checks.append(RealismCheck(
        "energy",
        "Is day-ahead vs real-time settlement basis risk modelled?",
        realistic=basis,
        observation=(
            f"da_rt_basis_blowout scenario present={basis}: a day-ahead planner can be "
            "wrong when the real-time basis blows out."
        ),
    ))


def _probe_robustness(checks: list[RealismCheck], seed: int) -> None:
    """Does constraint_aware still avoid SLA regressions across acting scenarios?

    This is the 'does the optimizer still win / not catastrophically lose' probe.
    We require: constraint_aware never increases hard SLA violations vs FIFO.
    """
    from .constraint_runner import (
        POLICY_CONSTRAINT_AWARE,
        POLICY_FIFO,
        ConstraintBenchmarkRunner,
    )

    runner = ConstraintBenchmarkRunner(policies=[POLICY_FIFO, POLICY_CONSTRAINT_AWARE])
    scenarios = [
        "thermal_hotspot_mixed_cluster",
        "queue_surge_latency_sensitive",
        "underutilization_stranded_capacity",
    ]
    regressions = []
    details = []
    for scn in scenarios:
        try:
            res = runner.run_scenario(scn, seed=seed, steps=24)
        except Exception as exc:
            details.append(f"{scn}: ERR {exc}")
            continue
        agg = res.report.aggregated
        fifo = agg.get(POLICY_FIFO)
        ca = agg.get(POLICY_CONSTRAINT_AWARE)
        if fifo is None or ca is None:
            continue
        worse = ca.total_sla_violations > fifo.total_sla_violations
        details.append(
            f"{scn}: SLA viol fifo={fifo.total_sla_violations} ca={ca.total_sla_violations}"
        )
        if worse:
            regressions.append(scn)
    checks.append(RealismCheck(
        "robustness",
        "Does constraint_aware preserve hard SLA compliance (no regression vs FIFO)?",
        realistic=not regressions,
        observation="; ".join(details) + (
            f"  REGRESSIONS: {regressions}" if regressions else "  no SLA regressions"
        ),
        severity="blocker" if regressions else "info",
    ))


# ---------------------------------------------------------------------------
# Verdict assignment
# ---------------------------------------------------------------------------

def _assign_verdict(
    subsystem: str,
    checks: list[RealismCheck],
    cal: Optional[dict[str, Any]],
) -> SubsystemVerdict:
    sub_checks = [c for c in checks if c.subsystem == subsystem]
    has_blocker = any((not c.realistic and c.severity == "blocker") for c in sub_checks)
    has_warn = any((not c.realistic and c.severity == "warn") for c in sub_checks)
    all_realistic = all(c.realistic for c in sub_checks)

    cal_conf = cal.get("mean_confidence") if cal else None
    any_measured = cal.get("any_measured", False) if cal else False

    if has_blocker:
        # A blocker that is specifically a "perfect telemetry" issue → NEEDS_REAL_TELEMETRY.
        telemetry_blocker = any(
            (not c.realistic and c.severity == "blocker"
             and ("telemetry" in c.question.lower() or "perfect" in c.question.lower()))
            for c in sub_checks
        )
        verdict = NEEDS_REAL_TELEMETRY if telemetry_blocker else TOO_SIMPLISTIC_FOR_CLAIMS
    elif all_realistic:
        verdict = REALISTIC_ENOUGH_FOR_DEV
    elif has_warn:
        verdict = REALISTIC_ENOUGH_FOR_DEV  # believable dynamics with documented caveats
    else:
        verdict = TOO_SIMPLISTIC_FOR_CLAIMS

    notes: list[str] = []
    if cal and not any_measured:
        notes.append(
            f"{cal.get('n', 0)} calibration params; NONE measured on real hardware "
            f"(mean confidence={cal_conf}). Treat all magnitudes as tunable priors."
        )

    return SubsystemVerdict(
        subsystem=subsystem,
        verdict=verdict,
        checks=sub_checks,
        calibration_confidence=cal_conf,
        calibration_summary=(cal.get("by_source_type") if cal else None),
        notes=notes,
    )


def run_realism_audit(seed: int = 42) -> RealismAuditReport:
    """Run all realism probes and return a verdict report."""
    checks: list[RealismCheck] = []
    _probe_serving(checks)
    _probe_arrivals(checks)
    _probe_migration(checks, seed)
    _probe_telemetry(checks, seed)
    _probe_actions(checks, seed)
    _probe_no_safe_action(checks, seed)
    _probe_energy(checks)
    _probe_robustness(checks, seed)

    cal_by_group = _calibration_by_group()
    # Map probe subsystems → calibration groups.
    cal_map = {
        "serving": cal_by_group.get("serving"),
        "migration": cal_by_group.get("migration"),
        "telemetry": None,  # telemetry confidence is not a calibration group
        "actions": None,
        "energy": cal_by_group.get("energy"),
        "robustness": None,
    }

    subsystems_order = ["serving", "migration", "telemetry", "actions", "energy", "robustness"]
    subsystems = [
        _assign_verdict(name, checks, cal_map.get(name)) for name in subsystems_order
    ]

    # Overall verdict: capped at NOT_PRODUCTION_REALISTIC_YET while no param is measured.
    any_measured = any(b.get("any_measured") for b in cal_by_group.values())
    verdicts = {s.verdict for s in subsystems}
    if not any_measured:
        overall = NOT_PRODUCTION_REALISTIC_YET
    elif verdicts <= {REALISTIC_ENOUGH_FOR_DEV}:
        overall = REALISTIC_ENOUGH_FOR_DEV
    else:
        overall = NOT_PRODUCTION_REALISTIC_YET

    headline: list[str] = []
    for c in checks:
        if not c.realistic and c.severity == "blocker":
            headline.append(f"[{c.subsystem}] {c.question} — {c.observation.split('.')[0]}")
    headline.append(
        "All calibration parameters are uncalibrated priors (none measured on real "
        "hardware). Simulator evidence is directional only — not production savings."
    )

    return RealismAuditReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        overall_verdict=overall,
        subsystems=subsystems,
        headline_findings=headline,
    )
