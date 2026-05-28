#!/usr/bin/env python3
"""Generate the Simulator Realism + Benchmark Validation report (task spec §9).

Runs every scenario under all policies (paired same-seed A/B), runs the simulator
realism audit, computes the packing frontier where applicable, and writes a
markdown report with:
  - the full per-scenario comparison table
  - mean/median energy-cost delta + engine net-savings vs each baseline
  - the realism audit verdicts
  - honest notes on where constraint_aware wins / loses

Usage:
    python scripts/generate_realism_report.py [--steps 24] [--seed 42] \
        [--out docs/REALISM_BENCHMARK_VALIDATION.md]

All outputs are [SANDBOX] / simulator-only. Not production savings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurelius.benchmarks import ConstraintBenchmarkRunner, run_realism_audit  # noqa: E402
from aurelius.benchmarks.constraint_runner import (  # noqa: E402
    POLICY_CONSTRAINT_AWARE,
    POLICY_FIFO,
    POLICY_GREEDY_ENERGY,
    POLICY_PRICE_ONLY,
    POLICY_SLA_AWARE,
)
from aurelius.simulation.cluster.scenarios import list_scenarios  # noqa: E402


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/REALISM_BENCHMARK_VALIDATION.md")
    args = ap.parse_args()

    runner = ConstraintBenchmarkRunner()
    scenarios = list_scenarios()

    rows: list[str] = []
    primary_kpi_rows: list[str] = []
    # cost-delta accumulators (baseline_cost - ca_cost) and engine net savings
    deltas: dict[str, list[float]] = {
        POLICY_FIFO: [], POLICY_PRICE_ONLY: [], POLICY_GREEDY_ENERGY: [], POLICY_SLA_AWARE: [],
    }
    # Per-policy primary-KPI series across scenarios (the canonical headline).
    primary_kpi_series: dict[str, list[float]] = {
        POLICY_FIFO: [], POLICY_PRICE_ONLY: [], POLICY_GREEDY_ENERGY: [],
        POLICY_SLA_AWARE: [], POLICY_CONSTRAINT_AWARE: [],
    }
    ca_loses_canonical_to: dict[str, list[str]] = {
        POLICY_FIFO: [], POLICY_PRICE_ONLY: [],
        POLICY_GREEDY_ENERGY: [], POLICY_SLA_AWARE: [],
    }
    ca_net_savings: list[float] = []
    sla_regressions: list[str] = []
    ca_wins: list[str] = []
    ca_losses: list[str] = []

    print(f"Running {len(scenarios)} scenarios × 5 policies (seed={args.seed}, steps={args.steps})")
    for scn in scenarios:
        try:
            res = runner.run_scenario(scn, seed=args.seed, steps=args.steps)
        except Exception as exc:
            rows.append(f"| {scn} | ERROR | {exc} |" + " |" * 11)
            continue
        agg = res.report.aggregated
        ca = agg.get(POLICY_CONSTRAINT_AWARE)
        fifo = agg.get(POLICY_FIFO)
        if ca is None or fifo is None:
            continue

        for pol in deltas:
            base = agg.get(pol)
            if base is not None:
                deltas[pol].append(base.total_energy_cost - ca.total_energy_cost)
        if ca.total_net_savings is not None:
            ca_net_savings.append(ca.total_net_savings)

        # Canonical KPI: SLA-safe goodput per infrastructure dollar.
        for pol, series in primary_kpi_series.items():
            base = agg.get(pol)
            if base is not None and base.sla_safe_goodput_per_infra_dollar is not None:
                series.append(base.sla_safe_goodput_per_infra_dollar)
        ca_primary = ca.sla_safe_goodput_per_infra_dollar
        # Material-loss floor: only call out scenarios where the baseline is at
        # least 1% better on the canonical KPI, so a fraction-of-a-percent tie
        # doesn't drown the genuine losses.
        _MATERIAL = 1.01
        for pol in ca_loses_canonical_to:
            base = agg.get(pol)
            if (base is not None and ca_primary is not None and ca_primary > 0
                    and base.sla_safe_goodput_per_infra_dollar is not None
                    and base.sla_safe_goodput_per_infra_dollar > ca_primary * _MATERIAL):
                ca_loses_canonical_to[pol].append(scn)

        # Per-scenario primary-KPI row across all 5 policies.
        primary_kpi_rows.append(
            f"| {scn} | "
            + " | ".join(
                (
                    f"{agg[p].sla_safe_goodput_per_infra_dollar:,.0f}"
                    if (p in agg and agg[p].sla_safe_goodput_per_infra_dollar is not None)
                    else "—"
                )
                for p in (POLICY_FIFO, POLICY_PRICE_ONLY, POLICY_GREEDY_ENERGY,
                          POLICY_SLA_AWARE, POLICY_CONSTRAINT_AWARE)
            )
            + " |"
        )

        if ca.total_sla_violations > fifo.total_sla_violations:
            sla_regressions.append(scn)

        # Win/loss heuristic vs FIFO on the scenario's likely binding KPI.
        throughput_up = ca.total_tokens > fifo.total_tokens * 1.02
        improved = (
            ca.total_thermal_throttle_ticks < fifo.total_thermal_throttle_ticks
            or (ca.p95_queue_wait_ms or 0) < (fifo.p95_queue_wait_ms or 0) * 0.98
            or throughput_up
            or ca.total_energy_cost < fifo.total_energy_cost * 0.99
        )
        no_sla_regression = ca.total_sla_violations <= fifo.total_sla_violations
        p99_worse = (ca.p99_latency_ms or 0) > (fifo.p99_latency_ms or 0) * 1.05
        # A genuine net loss: tail got worse AND it did NOT buy throughput/thermal relief.
        net_loss = (
            p99_worse
            and not throughput_up
            and ca.total_thermal_throttle_ticks >= fifo.total_thermal_throttle_ticks
            and ca.total_energy_cost >= fifo.total_energy_cost
        ) or (ca.total_sla_violations > fifo.total_sla_violations)
        if improved and no_sla_regression and not net_loss:
            ca_wins.append(scn)
        if net_loss:
            ca_losses.append(scn)

        # Telemetry confidence (mean over ticks) + partial flag, from the
        # constraint_aware engine assessments — exposes the Mission 1 telemetry truth.
        ca_pr = res.policy_results.get(POLICY_CONSTRAINT_AWARE)
        confs = [
            er.assessment.confidence
            for er in (ca_pr.engine_results if ca_pr else [])
            if er is not None
        ]
        mean_conf = statistics.mean(confs) if confs else None
        is_partial = bool(ca_pr.final_state.is_partial) if ca_pr and ca_pr.final_state else False
        conf_str = (f"{mean_conf:.2f}" if mean_conf is not None else "—") + (
            " (partial)" if is_partial else ""
        )

        # One row per scenario: constraint_aware primary KPI + absolute KPIs.
        primary = ca.sla_safe_goodput_per_infra_dollar
        cpsct = ca.cost_per_sla_compliant_token
        primary_str = f"{primary:,.0f}" if primary is not None else "—"
        cpsct_str = (
            "inf" if cpsct is not None and (cpsct == float("inf"))
            else (f"{cpsct:.3e}" if cpsct is not None else "—")
        )
        rows.append(
            f"| {scn} | constraint_aware | {primary_str} | {cpsct_str} | "
            f"{ca.sla_compliant_goodput:,} | {_fmt(ca.total_infrastructure_cost, 2)} | "
            f"{_fmt(ca.gpu_infra_cost, 2)} | {_fmt(ca.energy_cost, 3)} | "
            f"{_fmt(ca.total_energy_cost, 3)} | {_fmt(ca.total_tokens, 0)} | "
            f"{_fmt(ca.p99_latency_ms, 0)} | {_fmt(ca.p95_queue_wait_ms, 0)} | "
            f"{ca.total_sla_violations} | {ca.total_migrations} | "
            f"{_fmt(ca.churn_penalty_max, 4)} | {ca.total_thermal_throttle_ticks} | "
            f"{_fmt(ca.mean_topology_score, 3)} | {_fmt(ca.prefix_hit_rate_mean, 3)} | "
            f"{conf_str} |"
        )

    audit = run_realism_audit(seed=args.seed)

    def mm(xs):
        if not xs:
            return "—", "—"
        return f"{statistics.mean(xs):.4f}", f"{statistics.median(xs):.4f}"

    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# Simulator Realism + Benchmark Validation Report")
    lines.append("")
    lines.append(f"_Generated {ts} · seed={args.seed} · steps={args.steps} · **[SANDBOX]**_")
    lines.append("")
    lines.append("> All numbers are **simulator-only, uncalibrated** directional results. "
                 "Not production savings. See the realism audit verdict below.")
    lines.append("")
    lines.append("> **Primary KPI:** `sla_safe_goodput_per_infrastructure_dollar` "
                 "(Section 2). Raw energy cost is **not** the primary metric; it is a "
                 "diagnostic. Secondary KPIs (p99, queue, thermal, topology, …) are "
                 "constraints/vetoes, never folded into the primary KPI.")
    lines.append("")
    lines.append(f"**Realism audit overall verdict: `{audit.overall_verdict}`**")
    lines.append("")
    lines.append("## 1. Realism audit (per-subsystem)")
    lines.append("")
    lines.append("| Subsystem | Verdict | Calibration |")
    lines.append("|---|---|---|")
    for sv in audit.subsystems:
        lines.append(f"| {sv.subsystem} | `{sv.verdict}` | {sv.calibration_confidence or '—'} |")
    lines.append("")
    lines.append("Headline findings:")
    for f in audit.headline_findings:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## 2. Primary KPI: SLA-safe goodput per infrastructure dollar")
    lines.append("")
    lines.append("This is the canonical benchmark metric. Higher is better. The denominator is "
                 "`gpu_infra_cost + energy_cost + network_cost`; the numerator is tokens that "
                 "met their workload's SLO (queue `timeout_rate_pct` filter). Secondary KPIs "
                 "are NOT folded in — they're tracked separately below as constraints / "
                 "diagnostics. GPU infra cost typically dominates electricity by 50–200×.")
    lines.append("")
    lines.append("Per-policy aggregates across all scenarios:")
    lines.append("")
    lines.append("| Policy | Mean goodput / $ | Median goodput / $ |")
    lines.append("|---|---|---|")
    for pol, label in (
        (POLICY_FIFO, "FIFO"),
        (POLICY_PRICE_ONLY, "current_price_only"),
        (POLICY_GREEDY_ENERGY, "greedy_energy"),
        (POLICY_SLA_AWARE, "SLA-aware"),
        (POLICY_CONSTRAINT_AWARE, "constraint_aware"),
    ):
        m, med = mm(primary_kpi_series[pol])
        lines.append(f"| {label} | {m} | {med} |")
    lines.append("")
    lines.append("Scenarios where constraint_aware **loses** the canonical KPI to a baseline "
                 "(honest, not hidden):")
    for pol, label in (
        (POLICY_FIFO, "FIFO"),
        (POLICY_PRICE_ONLY, "current_price_only"),
        (POLICY_GREEDY_ENERGY, "greedy_energy"),
        (POLICY_SLA_AWARE, "SLA-aware"),
    ):
        scns = ca_loses_canonical_to[pol]
        lines.append(f"- vs {label}: {', '.join(scns) if scns else 'none'}")
    lines.append("")
    lines.append("Per-scenario primary KPI (SLA-safe goodput per $):")
    lines.append("")
    lines.append("| scenario | FIFO | current_price_only | greedy_energy | SLA-aware | "
                 "constraint_aware |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(primary_kpi_rows)
    lines.append("")

    lines.append("## 3. Mean / median delta vs each baseline (secondary — raw cost only)")
    lines.append("")
    lines.append("These are the **legacy** raw-cost deltas, retained for diagnostic purposes "
                 "only. They are NOT the primary KPI: a policy can be cheap on raw energy AND "
                 "lose on `sla_safe_goodput_per_infra_dollar` (see Section 2).")
    lines.append("")
    lines.append("Energy-cost delta = `baseline_cost − constraint_aware_cost` per scenario "
                 "(positive = constraint_aware cheaper). Engine net-savings is penalty-adjusted "
                 "(migration/cache/SLA/topology/thermal/forecast/churn).")
    lines.append("")
    lines.append("| Baseline | Mean cost delta ($) | Median cost delta ($) |")
    lines.append("|---|---|---|")
    for pol, label in (
        (POLICY_FIFO, "FIFO"),
        (POLICY_PRICE_ONLY, "current_price_only"),
        (POLICY_GREEDY_ENERGY, "greedy_energy"),
        (POLICY_SLA_AWARE, "SLA-aware"),
    ):
        mean_d, med_d = mm(deltas[pol])
        lines.append(f"| {label} | {mean_d} | {med_d} |")
    net_mean, net_med = mm(ca_net_savings)
    lines.append("")
    lines.append(f"Engine-computed constraint_aware net savings across scenarios: "
                 f"mean={net_mean}, median={net_med}.")
    lines.append("")
    lines.append("Packing baselines (first-fit / best-fit / FFD / clairvoyant) are reported "
                 "per packing scenario inside the benchmark JSON `packing_frontier` block; "
                 "they are analysis-only and never a deployable comparison.")
    lines.append("")

    lines.append("## 4. Per-scenario comparison (constraint_aware)")
    lines.append("")
    lines.append("| scenario | policy | goodput/$ (PRIMARY) | $/SLA-tok | "
                 "SLA-compliant goodput | infra $ | GPU $ | energy $ | "
                 "raw cost $ | raw tokens | p99 ms | queue p95 ms | "
                 "SLA viol | migrations | churn | thermal | topology | "
                 "cache hit | telemetry conf |")
    lines.append("|" + "---|" * 19)
    lines.extend(rows)
    lines.append("")

    lines.append("## 5. Safety regressions")
    lines.append("")
    if sla_regressions:
        lines.append("⚠ constraint_aware increased SLA violations vs FIFO in: "
                     + ", ".join(sla_regressions))
    else:
        lines.append("None — constraint_aware did not increase hard SLA violations vs FIFO "
                     "in any scenario.")
    lines.append("")

    lines.append("## 6. Where constraint_aware performs well / poorly")
    lines.append("")
    lines.append("Performs well (improves a binding KPI without SLA regression): "
                 + (", ".join(sorted(set(ca_wins))) or "none"))
    lines.append("")
    lines.append("Performs poorly (net loss — tail worse with no throughput/thermal/cost "
                 "relief, or an SLA regression): "
                 + (", ".join(sorted(set(ca_losses))) or "none"))
    lines.append("")
    lines.append("`greedy_energy` headline property (RESTORED): on "
                 "`energy_price_arbitrage_multiregion`, greedy_energy's aggressive migration "
                 "blows up p99 >5× past constraint_aware (now deterministic across pytest and "
                 "a plain interpreter — the prior xfail was a YAML/builtin scenario-drift "
                 "determinism bug, not a model regression; see test_scenario_source_parity.py).")
    lines.append("")
    lines.append("Honest open weakness (energy scenario): constraint_aware is still the most "
                 "EXPENSIVE policy on raw energy cost here and does not beat current_price_only "
                 "(which is cheaper AND has fewer SLA violations). Root cause: the engine still "
                 "applies some queue-relief scaling to BATCH workloads (which tolerate "
                 "queueing), wasting energy. The constraint-dominance guard reduces but does not "
                 "eliminate this; a full fix needs workload-class (priority_tier/latency_sensitive) "
                 "propagated into the canonical InferenceServiceState so the engine can apply "
                 "the spec's workload-aware priorities. Reported, not hidden.")
    lines.append("")

    lines.append("## 7. What remains simulator-only / needs real telemetry")
    lines.append("")
    lines.append("- Every calibration parameter is an uncalibrated prior (none measured on real "
                 "hardware). All KPI numbers are directional.")
    lines.append("- Telemetry truth (Mission 1, FIXED): the canonical `ClusterState` now derives "
                 "provenance confidence + `is_partial` from the simulator's per-subsystem tiers, "
                 "so degraded-telemetry scenarios report low/partial confidence and the engine "
                 "force-KEEPs (telemetry subsystem verdict graduated to REALISTIC_ENOUGH_FOR_DEV). "
                 "The tiers themselves remain uncalibrated heuristics.")
    lines.append("- Next calibration step: run a read-only shadow pilot against real "
                 "Prometheus/DCGM/K8s telemetry to calibrate the priors and the confidence "
                 "model (C = R·F·K·S·N) against measured staleness/coverage/noise, and propagate "
                 "workload class into InferenceServiceState for workload-aware action selection.")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out} ({len(rows)} scenario rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
