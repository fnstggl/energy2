#!/usr/bin/env python3
"""Azure LLM 2024 — Dynamic Safe Frontier Estimator benchmark.

Validates the Dynamic Safe Frontier Estimator
(:mod:`aurelius.frontier.dynamic_estimator`) on the Azure 2024 trace
using **offline streaming replay**: each rolling-window decision sees
only the telemetry from times t' ≤ t. No future leakage.

Compares, at each rolling step:

  * ``constraint_aware_static`` — fixed rho 0.65 (engine default).
  * ``static_frontier_controller`` — fixed rho 0.75 (the committed
    Azure 2024 frontier winner; see
    ``docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md``).
  * ``sla_aware`` / ``utilization_aware`` — fixed rho 0.50 / 0.85.
  * ``dynamic_frontier_estimator`` — rho recommended each window from
    the dynamic estimator (no future leakage).
  * ``oracle_realized_optimal`` — analysis-only post-hoc upper bound
    (which rho would have been chosen had the optimizer seen the
    realized outcomes). NEVER used as the headline baseline.

Outputs (new files only; committed artifacts are NOT overwritten):

  * docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md
  * data/external/azure_llm_2024/processed/
    azure_2024_dynamic_frontier_summary.json

Honesty / non-goals: simulator / shadow-mode evidence only — NOT
production savings (``docs/RESULTS.md`` §8). Real-cluster execution is
disabled by default. The committed Azure 2024 audit / backtest /
controller / integration / full-trace JSON are **read-only**.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.frontier import (  # noqa: E402
    DynamicControllerConfig,
    DynamicEstimatorConfig,
    RiskConfig,
    SafetyConfig,
    ServingTelemetryTick,
    WorkloadFrontierProfile,
    choose_dynamic_rho,
    estimate_dynamic_frontier,
    telemetry_tick_from_arrival_tick,
)
from aurelius.traces import azure_llm as az  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_dynamic_frontier_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs", "AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md")
RAW_DIR = os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "raw")
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                       "azure_llm_2024_sample.csv")

# Pre-registered constants — never tuned per trace.
TICK_S = 60.0
PRIMARY_SCALE = 10.0  # matches the committed Azure 2024 audit busy-tier
CANDIDATE_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
STATIC_CA_RHO = 0.65
STATIC_FC_RHO = 0.75
SLA_AWARE_RHO = 0.50
UTIL_AWARE_RHO = 0.85
WINDOW_TICKS = (30, 60, 180)  # rolling window sizes in minutes


# ---------------------------------------------------------------------------
# Helpers — scale ticks, build sizers, evaluate per-tick.
# ---------------------------------------------------------------------------

def _resolve_paths(raw_dir):
    return {v: os.path.join(raw_dir, f) for v, f in
            (("code", "AzureLLMInferenceTrace_code_1week.csv"),
             ("conv", "AzureLLMInferenceTrace_conv_1week.csv"))
            if os.path.exists(os.path.join(raw_dir, f))}


def _scale_ticks(ticks, f: float):
    if f == 1.0:
        return list(ticks)
    from dataclasses import replace
    return [replace(t, request_count=int(round(t.request_count * f)),
                    arrival_rate_rps=t.arrival_rate_rps * f,
                    total_prompt_tokens=int(round(t.total_prompt_tokens * f)),
                    total_output_tokens=int(round(t.total_output_tokens * f)),
                    model_mix={k: int(round(v * f))
                               for k, v in t.model_mix.items()})
            for t in ticks]


def _eval_tick(tick, target_rho: float, *, tick_hours: float) -> dict:
    """Evaluate one ArrivalTick at a chosen rho via the unchanged engine.

    Returns a dict carrying the *realized* per-tick metrics the dynamic
    estimator would observe: timeout %, queue p99, latency p99, replicas,
    GPU-hours, mean rho.
    """
    r = bt._size_for_target(tick.arrival_rate_rps,
                            max(1.0, tick.output_tokens_mean),
                            bt._tick_throughput_tokps(tick), target_rho)
    ev = bt.evaluate_tick(tick, r, prefill_savings=0.0,
                          tick_hours=tick_hours)
    return {
        "tick_index": tick.tick_index, "start_s": tick.start_s,
        "request_count": tick.request_count,
        "rho_target": target_rho,
        "replicas": ev.replicas, "rho": ev.rho,
        "timeout_pct": ev.timeout_rate_pct,
        "queue_p95_ms": ev.queue_wait_p95_ms,
        "queue_p99_ms": ev.queue_wait_p99_ms,
        "latency_p99_ms": ev.latency_p99_ms,
        "gpu_hours": sum(ev.gpu_hours_by_type.values()),
        "energy_cost": ev.energy_cost,
        "tokens_offered": ev.tokens_offered,
    }


def _kpi_from_evals(evals: list) -> dict:
    """Aggregate per-tick evals to the canonical KPI fields.

    Goodput/$ uses sla_compliant_tokens / (gpu_hours * gpu_hour_price +
    energy_cost). Same shape as the engine's compute_economic_kpi.
    """
    from aurelius.benchmarks.economics import (
        InfrastructureCostConfig,
        compute_economic_kpi,
    )
    cfg = InfrastructureCostConfig()
    tokens_per_tick = [e["tokens_offered"] for e in evals]
    timeout_pct_per_tick = [e["timeout_pct"] for e in evals]
    energy_per_tick = [e["energy_cost"] for e in evals]
    gpu_hours_per_tick = [{"a100-40gb": e["gpu_hours"]} for e in evals]
    kpi = compute_economic_kpi(
        tokens_per_tick=tokens_per_tick,
        timeout_rate_pct_per_tick=timeout_pct_per_tick,
        energy_cost_per_tick=energy_per_tick,
        active_gpu_hours_by_type_per_tick=gpu_hours_per_tick,
        migration_count=0, config=cfg)
    active = [e for e in evals if e["request_count"] > 0]
    aw = sum(e["request_count"] for e in active) or 1
    queue_p95_w = sum(e["queue_p95_ms"] * e["request_count"]
                       for e in active) / aw
    queue_p99_w = sum(e["queue_p99_ms"] * e["request_count"]
                       for e in active) / aw
    timeout_w = sum(e["timeout_pct"] * e["request_count"]
                     for e in active) / aw
    rho_w = sum(e["rho"] * e["request_count"] for e in active) / aw
    return {
        "goodput_per_dollar": float(kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "sla_compliant_goodput": int(kpi.sla_compliant_goodput),
        "gpu_hours": float(kpi.active_gpu_hours),
        "infra_cost": float(kpi.total_infrastructure_cost),
        "timeout_pct_mean": float(timeout_w),
        "queue_p95_ms": float(queue_p95_w),
        "queue_p99_ms": float(queue_p99_w),
        "mean_utilization_rho": float(rho_w),
        "n_active_ticks": len(active),
        "scale_events": sum(1 for i in range(1, len(evals))
                            if evals[i]["replicas"] != evals[i - 1]["replicas"]),
    }


# ---------------------------------------------------------------------------
# Streaming replay driver.
# ---------------------------------------------------------------------------

def _build_telemetry_tick(arrival_tick, eval_result) -> ServingTelemetryTick:
    """Build a ServingTelemetryTick from one realized eval — these are
    the observations the dynamic estimator would have at tick t."""
    return telemetry_tick_from_arrival_tick(
        arrival_tick,
        active_replicas=eval_result["replicas"],
        queue_p95_ms=eval_result["queue_p95_ms"],
        queue_p99_ms=eval_result["queue_p99_ms"],
        latency_p99_ms=eval_result["latency_p99_ms"],
        timeout_pct=eval_result["timeout_pct"],
        sla_violation_pct=(eval_result["timeout_pct"]
                           if eval_result["timeout_pct"] > 0 else 0.0),
        mean_utilization=eval_result["rho"],
        telemetry_confidence="medium",
        source="azure_2024_streaming_replay")


@dataclass
class DynamicReplayResult:
    """Per-window summary of the streaming-replay benchmark."""

    window_ticks: int
    evals: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    rho_history: list = field(default_factory=list)
    convergence_tick: Optional[int] = None
    n_actions: dict = field(default_factory=dict)
    kpi: dict = field(default_factory=dict)


def _run_static_policy(ticks, *, tick_hours: float,
                       target_rho: float) -> dict:
    evals = [_eval_tick(t, target_rho, tick_hours=tick_hours) for t in ticks]
    return _kpi_from_evals(evals)


def _run_dynamic_replay(ticks, *, tick_hours: float, window_ticks: int,
                        estimator_cfg: DynamicEstimatorConfig,
                        controller_cfg: DynamicControllerConfig,
                        safety_cfg: SafetyConfig,
                        risk_cfg: RiskConfig,
                        candidate_rhos=CANDIDATE_RHOS,
                        ) -> DynamicReplayResult:
    """Offline streaming replay: at each tick, the dynamic estimator
    sees only the most recent ``window_ticks`` realized observations and
    recommends the rho to use *for the next tick*."""
    profile = WorkloadFrontierProfile(
        workload_id="azure_llm_2024_week",
        workload_type="inference_standard",
        telemetry_confidence="medium",
        priority_class="standard",
        candidate_rhos=tuple(candidate_rhos),
        source="azure_2024_streaming_replay")

    res = DynamicReplayResult(window_ticks=window_ticks)
    history: list[ServingTelemetryTick] = []
    current_rho = STATIC_CA_RHO  # bootstrap with the engine default
    prev_action: Optional[str] = None
    converged_at: Optional[int] = None
    last_rec_rho = current_rho

    for i, t in enumerate(ticks):
        # Decide rho for this tick using the recent window (no leakage)
        if len(history) >= estimator_cfg.min_window_ticks:
            est = estimate_dynamic_frontier(
                workload_profile=profile,
                telemetry_window=history[-window_ticks:],
                current_rho=current_rho,
                candidate_rhos=candidate_rhos,
                config=estimator_cfg,
                safety_config=safety_cfg,
                risk_config=risk_cfg)
            dec = choose_dynamic_rho(est, current_rho=current_rho,
                                     config=controller_cfg,
                                     previous_action=prev_action)
            new_rho = (dec.recommended_rho
                       if dec.recommended_rho is not None
                       else current_rho)
            res.decisions.append({
                "tick_index": i, "current_rho": current_rho,
                "recommended_rho": new_rho, "action": dec.action,
                "confidence": dec.confidence,
                "reason": dec.reason[:120],
                "fallback_reason": dec.fallback_reason,
                "risk_at_current": est.risk_at_current_rho,
                "risk_at_recommended": est.risk_at_recommended_rho,
                "estimated_safe_rho": est.estimated_safe_rho,
                "frontier_slope": est.frontier_slope,
            })
            prev_action = dec.action
            current_rho = new_rho
            # Convergence: when the recommendation stays in the same
            # ±0.05 band for 10 consecutive ticks.
            if converged_at is None:
                if abs(new_rho - last_rec_rho) <= 0.05:
                    if i - (res.convergence_tick or 0) >= 10:
                        converged_at = i
            last_rec_rho = new_rho

        # Apply the chosen rho to the actual tick (realized outcome).
        ev = _eval_tick(t, current_rho, tick_hours=tick_hours)
        res.evals.append(ev)
        res.rho_history.append(current_rho)
        history.append(_build_telemetry_tick(t, ev))

    # Action counts
    action_counts: dict = {"RAISE_RHO": 0, "KEEP_RHO": 0,
                            "LOWER_RHO": 0, "INSUFFICIENT_TELEMETRY": 0}
    for d in res.decisions:
        action_counts[d["action"]] = action_counts.get(d["action"], 0) + 1
    res.n_actions = action_counts
    res.convergence_tick = converged_at
    res.kpi = _kpi_from_evals(res.evals)
    return res


# ---------------------------------------------------------------------------
# Oracle (analysis-only) — picks the realized best safe rho per window.
# ---------------------------------------------------------------------------

def _oracle_realized_optimal(ticks, *, tick_hours: float,
                              candidate_rhos=CANDIDATE_RHOS) -> dict:
    """Analysis-only oracle: the realized best safe rho over the whole
    trace. NOT used as the headline baseline — it sees the future."""
    best_gpd = float("-inf")
    best_rho = None
    by_rho = {}
    for rho in candidate_rhos:
        kpi = _run_static_policy(ticks, tick_hours=tick_hours, target_rho=rho)
        by_rho[rho] = kpi
        safe = (kpi["timeout_pct_mean"] <= SAFE_TIMEOUT_PCT
                and kpi["queue_p99_ms"] <= SAFE_QUEUE_P99_MS)
        if safe and kpi["goodput_per_dollar"] > best_gpd:
            best_gpd = kpi["goodput_per_dollar"]
            best_rho = rho
    return {"best_safe_rho": best_rho,
            "best_safe_goodput_per_dollar": best_gpd
                if best_gpd > float("-inf") else None,
            "per_rho_kpi": by_rho}


# ---------------------------------------------------------------------------
# Cross-comparison vs static baselines.
# ---------------------------------------------------------------------------

def _comparison_row(label: str, kpi: dict, rho_label) -> dict:
    return {
        "label": label, "rho": rho_label,
        "goodput_per_dollar": kpi["goodput_per_dollar"],
        "sla_compliant_goodput": kpi["sla_compliant_goodput"],
        "gpu_hours": kpi["gpu_hours"],
        "infra_cost": kpi["infra_cost"],
        "timeout_pct_mean": kpi["timeout_pct_mean"],
        "queue_p99_ms": kpi["queue_p99_ms"],
        "mean_utilization_rho": kpi["mean_utilization_rho"],
        "scale_events": kpi.get("scale_events", 0),
        "safe": (kpi["timeout_pct_mean"] <= SAFE_TIMEOUT_PCT
                 and kpi["queue_p99_ms"] <= SAFE_QUEUE_P99_MS),
    }


def _comparison(ticks, *, tick_hours: float,
                replay_results: dict, oracle: dict) -> dict:
    rows = [
        _comparison_row("sla_aware",
                        _run_static_policy(ticks, tick_hours=tick_hours,
                                            target_rho=SLA_AWARE_RHO),
                        SLA_AWARE_RHO),
        _comparison_row("constraint_aware_static",
                        _run_static_policy(ticks, tick_hours=tick_hours,
                                            target_rho=STATIC_CA_RHO),
                        STATIC_CA_RHO),
        _comparison_row("static_frontier_controller",
                        _run_static_policy(ticks, tick_hours=tick_hours,
                                            target_rho=STATIC_FC_RHO),
                        STATIC_FC_RHO),
        _comparison_row("utilization_aware",
                        _run_static_policy(ticks, tick_hours=tick_hours,
                                            target_rho=UTIL_AWARE_RHO),
                        UTIL_AWARE_RHO),
    ]
    for w in WINDOW_TICKS:
        if w not in replay_results:
            continue
        rep = replay_results[w]
        rows.append(_comparison_row(
            f"dynamic_frontier_estimator_w{w}m",
            rep.kpi,
            f"dynamic({rep.kpi['mean_utilization_rho']:.3f} mean)"))
    # Oracle (analysis-only)
    if oracle.get("best_safe_rho") is not None:
        best = oracle["per_rho_kpi"][oracle["best_safe_rho"]]
        rows.append({**_comparison_row(
            "oracle_realized_optimal_ANALYSIS_ONLY", best,
            oracle["best_safe_rho"]),
            "note": "analysis-only — sees full trace; not a real-time baseline"})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Synthesis.
# ---------------------------------------------------------------------------

def _synthesize(comparison: dict, replay_results: dict,
                 oracle: dict) -> dict:
    rows = comparison["rows"]
    ca_row = next(r for r in rows if r["label"] == "constraint_aware_static")
    fc_row = next(r for r in rows if r["label"] == "static_frontier_controller")
    or_row = next((r for r in rows
                   if r["label"] == "oracle_realized_optimal_ANALYSIS_ONLY"),
                  None)

    deltas = []
    for w in WINDOW_TICKS:
        if w not in replay_results:
            continue
        rep_row = next((r for r in rows
                        if r["label"] == f"dynamic_frontier_estimator_w{w}m"),
                       None)
        if rep_row is None:
            continue
        dyn = rep_row["goodput_per_dollar"]
        ca = ca_row["goodput_per_dollar"]
        fc = fc_row["goodput_per_dollar"]
        or_gpd = or_row["goodput_per_dollar"] if or_row else None
        delta_ca_pct = ((dyn - ca) / ca * 100.0) if ca else 0.0
        delta_fc_pct = ((dyn - fc) / fc * 100.0) if fc else 0.0
        opt_gap_pct = (((or_gpd - dyn) / or_gpd * 100.0)
                       if or_gpd else None)
        alpha_retained = ((dyn - ca) / (or_gpd - ca) * 100.0
                           if (or_gpd and (or_gpd - ca) > 0) else None)
        deltas.append({
            "window_minutes": w,
            "dynamic_goodput_per_dollar": dyn,
            "vs_constraint_aware_static_pct": delta_ca_pct,
            "vs_static_frontier_controller_pct": delta_fc_pct,
            "vs_oracle_pct": (-opt_gap_pct
                              if opt_gap_pct is not None else None),
            "optimality_gap_pct": opt_gap_pct,
            "alpha_retained_vs_oracle_pct": alpha_retained,
            "safe": rep_row["safe"],
            "action_distribution": replay_results[w].n_actions,
            "convergence_tick": replay_results[w].convergence_tick,
            "rho_mean": rep_row["mean_utilization_rho"],
        })

    # Verdict
    if not deltas:
        verdict = "NO_DYNAMIC_RESULT"
    elif any(d["vs_constraint_aware_static_pct"] >= 1.0 and d["safe"]
              for d in deltas):
        verdict = "DYNAMIC_BEATS_STATIC_CA"
    elif all(abs(d["vs_constraint_aware_static_pct"]) <= 1.0 and d["safe"]
              for d in deltas):
        verdict = "DYNAMIC_TIES_STATIC_CA"
    elif any(not d["safe"] for d in deltas):
        verdict = "DYNAMIC_UNSAFE"
    else:
        verdict = "DYNAMIC_BEHIND_STATIC_CA"

    # Frontier-recovery answer
    if or_row and deltas:
        # Use the longest window as the reference for the frontier-recovery
        # question — that's the configuration the estimator has the most
        # context for.
        ref = deltas[-1]
        recovery_pct = ref.get("alpha_retained_vs_oracle_pct")
        if recovery_pct is None:
            recovery = "INSUFFICIENT_DATA"
        elif recovery_pct >= 80.0:
            recovery = (f"YES — dynamic estimator recovered "
                        f"{recovery_pct:.1f}% of the alpha between "
                        "constraint_aware and the oracle.")
        elif recovery_pct >= 30.0:
            recovery = (f"PARTIAL — dynamic estimator recovered "
                        f"{recovery_pct:.1f}% of the alpha between "
                        "constraint_aware and the oracle.")
        else:
            recovery = (f"NO — dynamic estimator only recovered "
                        f"{recovery_pct:.1f}% of the alpha (≤30%); the "
                        "static frontier controller is the safer bet on "
                        "this trace.")
    else:
        recovery = "INSUFFICIENT_DATA"

    return {
        "deltas": deltas, "verdict": verdict,
        "frontier_recovery": recovery,
        "constraint_aware_static_goodput_per_dollar":
            ca_row["goodput_per_dollar"],
        "static_frontier_controller_goodput_per_dollar":
            fc_row["goodput_per_dollar"],
        "oracle_goodput_per_dollar":
            or_row["goodput_per_dollar"] if or_row else None,
    }


# ---------------------------------------------------------------------------
# Markdown.
# ---------------------------------------------------------------------------

def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Azure LLM 2024 — Dynamic Safe Frontier Estimator Results\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Streaming replay of "
      "the Azure 2024 trace with the Dynamic Safe Frontier Estimator "
      "(`aurelius/frontier/dynamic_estimator.py`); each per-tick "
      "decision sees only the telemetry from t' ≤ t (no future "
      "leakage). The robust energy engine is **unchanged**; the static "
      "frontier controller and committed Azure 2024 artifacts are "
      "**read-only**. Real-cluster execution is disabled by default.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, "
      "`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, "
      "`docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, "
      "`docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, "
      "`docs/PILOT_TELEMETRY_CONTRACT.md`.\n")

    cfg = payload["config"]
    A("## 1. Configuration\n")
    A(f"- **Tick seconds:** {cfg['tick_seconds']}")
    A(f"- **Primary scale:** {cfg['primary_scale']}×")
    A(f"- **Candidate rho grid:** `{cfg['candidate_rhos']}`")
    A(f"- **Safety thresholds (pre-registered):** timeout ≤ "
      f"{cfg['safety_timeout_pct']}% AND queue p99 ≤ "
      f"{cfg['safety_queue_p99_ms']} ms")
    A(f"- **Rolling windows:** {cfg['window_minutes']} min")
    A(f"- **No future leakage** — each decision sees t' ≤ t only.\n")

    src = payload["source"]
    A("## 2. Source\n")
    A(f"- **Trace:** `{src['path']}` ({src['n_ticks']:,} ticks @ "
      f"{src['tick_seconds']:.0f}s; "
      f"{src['time_span_seconds']:,.0f} s total)\n")

    A("## 3. Streaming-replay results\n")
    A("| label | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | "
      "mean rho | scale ev | safe |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in payload["comparison"]["rows"]:
        A(f"| `{r['label']}` | {_f(r['rho'])} | "
          f"{_f(r['goodput_per_dollar'])} | "
          f"{_f(r['timeout_pct_mean'])} | "
          f"{_f(r['queue_p99_ms'])} | "
          f"{_f(r['gpu_hours'])} | "
          f"{_f(r['mean_utilization_rho'], nd=4)} | "
          f"{_f(r['scale_events'])} | "
          f"{'✅' if r['safe'] else '❌'} |")
    A("")

    syn = payload["synthesis"]
    A("## 4. Cross-window deltas\n")
    A("| window | dynamic goodput/$ | Δ vs CA static | Δ vs FC static | "
      "Δ vs oracle | optimality gap | alpha retained vs oracle | "
      "convergence tick | safe |")
    A("|---|---|---|---|---|---|---|---|---|")
    for d in syn["deltas"]:
        vs_oracle = (f"{d['vs_oracle_pct']:+.3f}%"
                     if d['vs_oracle_pct'] is not None else "—")
        opt_gap = (f"{d['optimality_gap_pct']:+.3f}%"
                   if d['optimality_gap_pct'] is not None else "—")
        alpha = (f"{d['alpha_retained_vs_oracle_pct']:+.3f}%"
                 if d['alpha_retained_vs_oracle_pct'] is not None else "—")
        A(f"| {d['window_minutes']} min | "
          f"{_f(d['dynamic_goodput_per_dollar'])} | "
          f"{d['vs_constraint_aware_static_pct']:+.3f}% | "
          f"{d['vs_static_frontier_controller_pct']:+.3f}% | "
          f"{vs_oracle} | {opt_gap} | {alpha} | "
          f"{d.get('convergence_tick', '—')} | "
          f"{'✅' if d['safe'] else '❌'} |")
    A("")
    A(f"**Verdict:** **`{syn['verdict']}`**\n")
    A(f"**Frontier recovery:** {syn['frontier_recovery']}\n")

    A("## 5. Action distribution by window\n")
    A("| window | RAISE | KEEP | LOWER | INSUFFICIENT |")
    A("|---|---|---|---|---|")
    for w in payload["config"]["window_minutes"]:
        if w not in payload["replay_action_counts"]:
            continue
        a = payload["replay_action_counts"][w]
        A(f"| {w} min | {a.get('RAISE_RHO', 0)} | {a.get('KEEP_RHO', 0)} | "
          f"{a.get('LOWER_RHO', 0)} | "
          f"{a.get('INSUFFICIENT_TELEMETRY', 0)} |")
    A("")

    A("## 6. Honesty / scope\n")
    A("- The Dynamic Safe Frontier Estimator is **opt-in** and "
      "**disabled by default**. The static frontier controller remains "
      "the committed default; this benchmark is a measurement.")
    A("- **No production mutation.** Decisions are recommendation-only "
      "(`executable_in_real_cluster=False`).")
    A("- **No future leakage.** Each per-tick decision sees only the "
      "telemetry from t' ≤ t in its rolling window.")
    A("- **No ML training in v1.** Risk scores are deterministic / "
      "statistical heuristics (EWMA, slopes, CV, Erlang-C tails). See "
      "`docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`.")
    A("- The oracle row is **analysis-only**. It sees the entire trace "
      "ahead of time and is not a real-time baseline — it is a ceiling "
      "for the recovery question.")
    A("- This is **directional simulator / shadow-mode evidence** — NOT "
      "production savings. Pilot telemetry is required to calibrate the "
      "safe rho per workload before any production claim.")
    A("- The robust energy engine, the static frontier controller, the "
      "committed Azure 2024 audit / backtest / controller / integration "
      "/ full-trace JSON are **NOT modified** by this benchmark.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--tick-seconds", type=float, default=TICK_S)
    p.add_argument("--primary-scale", type=float, default=None,
                   help=(f"load multiplier; default {PRIMARY_SCALE} for the "
                         f"full week-long raw trace, 100.0 for the "
                         "single-day fixture (the frontier needs enough "
                         "per-tick load for replicas > 1)"))
    p.add_argument("--windows", nargs="*", type=int, default=list(WINDOW_TICKS),
                   help="rolling-window sizes in MINUTES")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="cap tick count for faster runs (default: all)")
    args = p.parse_args(argv)

    paths = _resolve_paths(args.raw_dir)
    if paths:
        agg = az.stream_week_aggregate(paths, tick_seconds=args.tick_seconds)
        ticks = agg["arrival_ticks"]
        src_path = f"{args.raw_dir}/AzureLLMInferenceTrace_*_1week.csv"
        scale_default = PRIMARY_SCALE
    else:
        agg = az.stream_week_aggregate({"conv": args.fixture},
                                        tick_seconds=args.tick_seconds)
        ticks = agg["arrival_ticks"]
        src_path = args.fixture
        # Fixture is ~1 day vs full week — bump scale so per-tick replica
        # counts > 1, otherwise every rho ties at MIN_REPLICAS.
        scale_default = 100.0

    primary_scale = (args.primary_scale if args.primary_scale is not None
                     else scale_default)
    if args.max_ticks:
        ticks = ticks[: args.max_ticks]
    ticks = _scale_ticks(ticks, primary_scale)
    tick_hours = args.tick_seconds / 3600.0

    safety_cfg = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                               max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                               min_telemetry_confidence="low")
    risk_cfg = RiskConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                           max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                           min_telemetry_confidence="low")
    estimator_cfg = DynamicEstimatorConfig(
        min_telemetry_confidence="low",
        conservative_margin_enabled=True)
    controller_cfg = DynamicControllerConfig()

    replay_results: dict = {}
    for w in args.windows:
        rep = _run_dynamic_replay(ticks, tick_hours=tick_hours,
                                  window_ticks=w,
                                  estimator_cfg=estimator_cfg,
                                  controller_cfg=controller_cfg,
                                  safety_cfg=safety_cfg, risk_cfg=risk_cfg)
        replay_results[w] = rep
        print(f"[dynamic] window={w} min: rho_mean="
              f"{rep.kpi['mean_utilization_rho']:.3f} "
              f"gpd/$={rep.kpi['goodput_per_dollar']:,.2f} "
              f"actions={rep.n_actions} converged_at={rep.convergence_tick}",
              flush=True)

    oracle = _oracle_realized_optimal(ticks, tick_hours=tick_hours)
    comparison = _comparison(ticks, tick_hours=tick_hours,
                              replay_results=replay_results, oracle=oracle)
    synthesis = _synthesize(comparison, replay_results, oracle)

    payload = {
        "config": {
            "tick_seconds": args.tick_seconds,
            "primary_scale": primary_scale,
            "candidate_rhos": list(CANDIDATE_RHOS),
            "safety_timeout_pct": SAFE_TIMEOUT_PCT,
            "safety_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "window_minutes": list(args.windows),
            "no_future_leakage": True,
            "execution_mode_default": "shadow",
            "real_execution_disabled_by_default": True,
        },
        "source": {
            "path": src_path,
            "n_ticks": len(ticks),
            "tick_seconds": args.tick_seconds,
            "time_span_seconds": (ticks[-1].start_s - ticks[0].start_s
                                   if ticks else 0.0),
            "scale": args.primary_scale,
        },
        "replay_action_counts": {w: rep.n_actions
                                  for w, rep in replay_results.items()},
        "replay_rho_history": {w: rep.rho_history
                                for w, rep in replay_results.items()},
        "replay_decisions": {w: rep.decisions[:50]
                              for w, rep in replay_results.items()},
        "oracle": oracle,
        "comparison": comparison,
        "synthesis": synthesis,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)
    print(f"[dynamic] verdict: {synthesis['verdict']}")
    print(f"[dynamic] recovery: {synthesis['frontier_recovery']}")
    print(f"[dynamic] JSON -> {args.out_json}")
    print(f"[dynamic] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
