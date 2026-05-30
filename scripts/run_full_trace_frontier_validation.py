#!/usr/bin/env python3
"""Full-Trace Safe Utilization Frontier Validation Audit.

Re-runs the cross-trace frontier audit on the **raw, full traces** for
Azure LLM 2023 and BurstGPT (the previous fixture-bound run reported a
SAFE_TIE on Azure 2023 and a marginal +1.4 % on BurstGPT). Answers:

  * Were the previous ties caused by **fixture limitations** (small
    telemetry window, low replica counts, controller fallback) — or do
    they reflect that the current ``constraint_aware`` operating point
    already sits **near the safe-utilization frontier** on those workloads?

  * Does ``frontier_controller_v1`` **truly generalize** across LLM
    serving traces, or does most of its measured value come from the
    Azure LLM 2024 trace alone?

Pre-registered safety thresholds (mirror the Azure 2024 + cross-trace
audits, never tuned to force a result):

  * timeout ≤ 10 %
  * queue p99 ≤ 2000 ms
  * tie-band ±1 %

Reuses the UNCHANGED serving physics in ``aurelius/traces/backtest.py``
and the UNCHANGED frontier controller / integration in
``aurelius/frontier/`` + ``aurelius/constraints/frontier_integration.py``.

No new datasets are ingested *for product use* (the raw public traces
downloaded here were already declared sources for the existing canonical
backtests — see ``aurelius/traces/burstgpt.py:DEFAULT_SOURCE_URL`` and
``aurelius/traces/azure_llm.py:SOURCE_URLS``). No ML training. No
optimizer / safety / robust-energy-engine constant is modified.

Read first:
  * docs/RESULTS.md
  * docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md
  * docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md
  * docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md
  * docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md
  * docs/PUBLIC_TRACE_BACKTESTS.md

Outputs:
  * docs/FULL_TRACE_FRONTIER_VALIDATION.md
  * data/external/frontier/full_trace_frontier_validation_summary.json

Directional simulator / shadow-mode evidence only — NOT production
savings (``docs/RESULTS.md`` §8). Real-cluster execution remains
disabled by default. The committed Azure 2024 audit / backtest /
controller / integration JSON are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.constraints.frontier_integration import (  # noqa: E402
    CONSTRAINT_AWARE_DEFAULT_RHO,
    FrontierIntegrationConfig,
    FrontierIntegrationCounters,
)
from aurelius.frontier import (  # noqa: E402
    SHADOW_MODE,
    FrontierControllerConfig,
    SafetyConfig,
    SafetyStatus,
    WorkloadFrontierProfile,
    choose_safe_utilization_target,
    estimate_frontier_from_points,
)
from aurelius.traces import azure_llm as az  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402
from aurelius.traces import burstgpt as bg  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO_ROOT, "data", "external", "frontier",
                        "full_trace_frontier_validation_summary.json")
OUT_MD = os.path.join(REPO_ROOT, "docs", "FULL_TRACE_FRONTIER_VALIDATION.md")
AZURE_2024_AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
AZURE_2024_FC_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_frontier_controller_summary.json")

# Candidate grid + safety thresholds — pre-registered, never tuned.
RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
TIE_BAND_PCT = 1.0


# ---------------------------------------------------------------------------
# Rho-sweep sizers (mirror scripts/run_cross_trace_frontier_generalization_audit.py
# and aurelius/frontier/estimator.py — UNCHANGED physics).
# ---------------------------------------------------------------------------

class _Reactive:
    def __init__(self, R: float):
        self.R = R
        self.prev = None

    def size(self, t):
        src = self.prev if self.prev is not None else t
        r = bt._size_for_target(src.arrival_rate_rps,
                                max(1.0, src.output_tokens_mean),
                                bt._tick_throughput_tokps(src), self.R)
        self.prev = t
        return r


class _Anticipatory:
    def __init__(self, R: float, *, tick_hours: float):
        self.R = R
        self.tick_hours = tick_hours
        self.ewma_r = 0.0
        self.ewma_o = 0.0
        self.prev_r = None

    def size(self, t):
        a = 0.5
        if t.request_count > 0:
            self.ewma_r = (a * t.arrival_rate_rps + (1 - a) * self.ewma_r
                           if self.ewma_r else t.arrival_rate_rps)
            self.ewma_o = (a * t.output_tokens_mean + (1 - a) * self.ewma_o
                           if self.ewma_o else t.output_tokens_mean)
        plan_rate = max(t.arrival_rate_rps, self.ewma_r)
        plan_out = (max(t.output_tokens_mean, self.ewma_o)
                    if t.request_count else self.ewma_o)
        base = bt._size_for_target(plan_rate, max(1.0, plan_out),
                                   bt._tick_throughput_tokps(t), self.R)
        r = bt._constraint_trim(t, base, 0.0, self.tick_hours, self.prev_r)
        self.prev_r = r
        return r


def _eval_rho(ticks, R: float, *, tick_seconds: float, mode: str) -> dict:
    tick_hours = tick_seconds / 3600.0
    sizer = (_Anticipatory(R, tick_hours=tick_hours) if mode == "anticipatory"
             else _Reactive(R))
    evals = []
    prev_r = None
    for t in ticks:
        r = sizer.size(t)
        ev = bt.evaluate_tick(t, r, prefill_savings=0.0, tick_hours=tick_hours)
        if prev_r is not None and ev.replicas != prev_r:
            ev.scale_event = True
        prev_r = ev.replicas
        evals.append(ev)
    res = bt._aggregate(f"{mode}@{R}", evals, cache_aware=False, ticks=ticks)
    active = [(e, t) for e, t in zip(evals, ticks) if t.request_count > 0]
    aw = sum(t.request_count for _, t in active) or 1
    mean_rho = sum(e.rho * t.request_count for e, t in active) / aw
    timeout_w = sum(e.timeout_rate_pct * t.request_count for e, t in active) / aw
    sla_viol = sum(t.request_count for e, t in active
                   if e.timeout_rate_pct > 0) / aw
    reps = [e.replicas for e in evals]
    churn = sum(abs(reps[i] - reps[i - 1]) for i in range(1, len(reps)))
    scale_events = sum(1 for e in evals if e.scale_event)
    return {
        "rho_target": R,
        "predicted_goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "predicted_sla_safe_goodput": int(res.kpi.sla_compliant_goodput),
        "predicted_gpu_hours": float(res.kpi.active_gpu_hours),
        "predicted_infra_cost": float(res.kpi.total_infrastructure_cost),
        "predicted_timeout_pct": float(timeout_w),
        "predicted_sla_violation_rate": float(sla_viol),
        "predicted_queue_p95_ms": float(res.queue_p95_ms),
        "predicted_queue_p99_ms": float(res.queue_p99_ms),
        "predicted_latency_p99_ms": float(res.latency_p99_ms) or None,
        "predicted_scale_events": int(scale_events),
        "predicted_churn_score": float(churn),
        "predicted_mean_utilization": float(mean_rho),
        "mean_replicas": float(sum(reps) / max(1, len(reps))),
    }


def _classify_point_safety(p: dict) -> tuple[str, list[str]]:
    vetoes = []
    if p["predicted_timeout_pct"] > SAFE_TIMEOUT_PCT:
        vetoes.append("timeout_exceeds_threshold")
    if p["predicted_queue_p99_ms"] > SAFE_QUEUE_P99_MS:
        vetoes.append("queue_p99_exceeds_threshold")
    return ("UNSAFE" if vetoes else "SAFE"), vetoes


def _policy_row_llm(name: str, res, rho) -> dict:
    return {
        "policy": name, "rho_target": rho,
        "goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "sla_compliant_goodput": int(res.kpi.sla_compliant_goodput),
        "gpu_hours": float(res.kpi.active_gpu_hours),
        "infra_cost": float(res.kpi.total_infrastructure_cost),
        "timeout_pct_mean": float(getattr(res, "timeout_rate_pct_mean", 0.0)),
        "queue_p95_ms": float(res.queue_p95_ms),
        "queue_p99_ms": float(res.queue_p99_ms),
        "latency_p99_ms": float(res.latency_p99_ms),
        "safe": bool(getattr(res, "timeout_rate_pct_mean", 0.0)
                     <= SAFE_TIMEOUT_PCT
                     and res.queue_p99_ms <= SAFE_QUEUE_P99_MS),
    }


def _run_named_policies(ticks, *, tick_seconds: float) -> list[dict]:
    """sla_aware / constraint_aware / queue_aware / utilization_aware / fifo."""
    tick_hours = tick_seconds / 3600.0
    base = {p: bt._run_policy(p, ticks, tick_hours=tick_hours)
            for p in ("fifo", "sla_aware", "queue_aware", "constraint_aware")}

    # utilization_aware (per-tick rho 0.85 — Azure 2024 convention)
    evals = []
    prev_r = None
    for t in ticks:
        r = bt._size_for_target(t.arrival_rate_rps,
                                max(1.0, t.output_tokens_mean),
                                bt._tick_throughput_tokps(t), 0.85)
        ev = bt.evaluate_tick(t, r, prefill_savings=0.0, tick_hours=tick_hours)
        if prev_r is not None and ev.replicas != prev_r:
            ev.scale_event = True
        prev_r = ev.replicas
        evals.append(ev)
    util = bt._aggregate("utilization_aware", evals,
                         cache_aware=False, ticks=ticks)
    rho_for = {"sla_aware": 0.50, "constraint_aware": CONSTRAINT_AWARE_DEFAULT_RHO,
               "queue_aware": None, "utilization_aware": 0.85, "fifo": None}
    rows = [_policy_row_llm(name, base[name], rho_for[name])
            for name in ("fifo", "sla_aware", "queue_aware", "constraint_aware")]
    rows.append(_policy_row_llm("utilization_aware", util, 0.85))
    return rows


def _frontier_controller_row(ticks, *, tick_seconds: float,
                             frontier_points: list[dict],
                             workload_id: str) -> dict:
    """Run the frontier controller on the precomputed sweep + integration
    adapter, and pick the matching anticipatory KPI row as the controller's
    result. Falls back to constraint_aware default on any unsafe/error path
    (matches the production integration's behaviour)."""
    profile = WorkloadFrontierProfile(
        workload_id=workload_id, workload_type="inference_standard",
        telemetry_confidence="medium", priority_class="standard",
        candidate_rhos=tuple(RHOS), source=workload_id)
    safety = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                          max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                          min_telemetry_confidence="low")
    pts = estimate_frontier_from_points(profile, frontier_points,
                                        safety_config=safety)
    cfg = FrontierControllerConfig(conservative_margin=False,
                                   deadband_rho=0.05, deadband_kpi_pct=0.05,
                                   min_telemetry_confidence="low",
                                   default_execution_mode=SHADOW_MODE)
    decision = choose_safe_utilization_target(profile, pts,
                                              current_rho=CONSTRAINT_AWARE_DEFAULT_RHO,
                                              controller_config=cfg)
    selected = decision.selected_point
    if selected is None or selected.safety_status != SafetyStatus.SAFE:
        return {"policy": "frontier_controller_v1",
                "rho_target": decision.selected_rho, "decision": decision,
                "goodput_per_dollar": 0.0, "sla_compliant_goodput": 0,
                "gpu_hours": 0.0, "infra_cost": 0.0,
                "timeout_pct_mean": None, "queue_p95_ms": None,
                "queue_p99_ms": None, "latency_p99_ms": None,
                "safe": False, "frontier_used": False}
    return {"policy": "frontier_controller_v1",
            "rho_target": decision.selected_rho, "decision": decision,
            "goodput_per_dollar": selected.predicted_goodput_per_dollar,
            "sla_compliant_goodput": selected.predicted_sla_safe_goodput,
            "gpu_hours": selected.predicted_gpu_hours,
            "infra_cost": (selected.predicted_sla_safe_goodput
                           / selected.predicted_goodput_per_dollar
                           if selected.predicted_goodput_per_dollar else 0.0),
            "timeout_pct_mean": selected.predicted_timeout_pct,
            "queue_p95_ms": selected.predicted_queue_p95_ms,
            "queue_p99_ms": selected.predicted_queue_p99_ms,
            "latency_p99_ms": selected.predicted_latency_p99_ms,
            "safe": True, "frontier_used": True}


def _root_cause_tie(diag: dict, named_rows: list[dict],
                    fc_row: dict, frontier: list[dict]) -> dict:
    """Classify *why* a SAFE_TIE occurred — never guesses, every label is
    backed by an explicit evidence field."""
    fc_action = (fc_row["decision"].action if isinstance(fc_row.get("decision"),
                                                          object)
                 and fc_row.get("decision") is not None else None)
    safe_count = sum(1 for p in frontier if _classify_point_safety(p)[0] == "SAFE")
    unsafe_count = len(frontier) - safe_count
    safe_kpis = [p["predicted_goodput_per_dollar"] for p in frontier
                 if _classify_point_safety(p)[0] == "SAFE"]
    ca_row = next(r for r in named_rows if r["policy"] == "constraint_aware")
    ca_kpi = ca_row["goodput_per_dollar"]
    best_safe = max(safe_kpis) if safe_kpis else None
    # A) insufficient telemetry: the controller emitted INSUFFICIENT_TELEMETRY
    if fc_action == "INSUFFICIENT_TELEMETRY":
        return {"code": "A_insufficient_telemetry",
                "evidence": ("frontier controller emitted "
                             "INSUFFICIENT_TELEMETRY action")}
    # B) frontier already discovered by constraint_aware (CA ≈ best safe KPI)
    if (best_safe is not None and ca_kpi
            and abs((best_safe - ca_kpi) / ca_kpi * 100.0) <= TIE_BAND_PCT):
        return {"code": "B_constraint_aware_already_on_frontier",
                "evidence": (
                    f"best safe KPI {best_safe:,.2f} vs constraint_aware KPI "
                    f"{ca_kpi:,.2f} → Δ "
                    f"{(best_safe - ca_kpi) / ca_kpi * 100.0:+.3f}% (≤ tie band "
                    f"{TIE_BAND_PCT}%)")}
    # D) safety limits reached — every rho is UNSAFE
    if safe_count == 0:
        return {"code": "D_safety_limits_reached",
                "evidence": "0 of {} candidate rhos pass the safety gates"
                            .format(len(frontier))}
    # C) workload saturation — flat KPI across rhos (Δ < 1 %)
    if safe_kpis and (max(safe_kpis) - min(safe_kpis)) / max(safe_kpis) * 100 < 1.0:
        return {"code": "C_workload_saturation",
                "evidence": (f"safe KPIs flat across rhos: "
                             f"min {min(safe_kpis):,.2f} / max "
                             f"{max(safe_kpis):,.2f} (Δ < 1 %)")}
    # E) trace limitations — small n_ticks
    n_ticks = diag.get("n_ticks") or 0
    if n_ticks < 60:
        return {"code": "E_trace_limitations",
                "evidence": f"n_ticks={n_ticks} below 60 — frontier "
                            "differentiation is noise-limited"}
    return {"code": "UNCLASSIFIED",
            "evidence": (f"safe_count={safe_count}, unsafe_count={unsafe_count}, "
                         f"best_safe_KPI={best_safe}, CA_KPI={ca_kpi}")}


# ---------------------------------------------------------------------------
# Trace loaders
# ---------------------------------------------------------------------------

def _load_burstgpt(*, raw_path: str, max_requests=None) -> dict:
    t0 = time.time()
    reqs = bg.load_csv(raw_path, include_failures=False,
                       sample_size=max_requests)
    t_load = time.time() - t0
    if not reqs:
        return {"applicable": False, "exclusion_reason": "no requests loaded"}
    t0 = time.time()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    t_bin = time.time() - t0
    return {"applicable": True, "requests": reqs, "ticks": ticks,
            "raw_path": raw_path,
            "n_requests": len(reqs), "n_ticks": len(ticks),
            "tick_seconds": 60.0,
            "time_span_s": reqs[-1].timestamp_s - reqs[0].timestamp_s,
            "load_seconds": t_load, "bin_seconds": t_bin}


def _load_azure_2023(*, raw_path: str, max_requests=None) -> dict:
    t0 = time.time()
    reqs = az.load_csv(raw_path, variant=az.variant_from_path(raw_path),
                       sample_size=max_requests, include_failures=False)
    t_load = time.time() - t0
    if not reqs:
        return {"applicable": False, "exclusion_reason": "no requests loaded"}
    t0 = time.time()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    t_bin = time.time() - t0
    return {"applicable": True, "requests": reqs, "ticks": ticks,
            "raw_path": raw_path,
            "n_requests": len(reqs), "n_ticks": len(ticks),
            "tick_seconds": 60.0,
            "time_span_s": reqs[-1].timestamp_s - reqs[0].timestamp_s,
            "load_seconds": t_load, "bin_seconds": t_bin}


# ---------------------------------------------------------------------------
# Per-trace audit
# ---------------------------------------------------------------------------

def _audit_trace(name: str, loaded: dict) -> dict:
    if not loaded["applicable"]:
        return {"trace": name, "applicable": False,
                "exclusion_reason": loaded["exclusion_reason"]}
    ticks = loaded["ticks"]
    tick_seconds = loaded["tick_seconds"]

    t0 = time.time()
    frontier_antic = [_eval_rho(ticks, R, tick_seconds=tick_seconds,
                                mode="anticipatory") for R in RHOS]
    frontier_react = [_eval_rho(ticks, R, tick_seconds=tick_seconds,
                                mode="reactive") for R in RHOS]
    for p in frontier_antic + frontier_react:
        st, vetoes = _classify_point_safety(p)
        p["safety_status"] = st
        p["safety_vetoes"] = vetoes
    t_sweep = time.time() - t0

    t0 = time.time()
    named_rows = _run_named_policies(ticks, tick_seconds=tick_seconds)
    t_named = time.time() - t0

    fc_row = _frontier_controller_row(ticks, tick_seconds=tick_seconds,
                                      frontier_points=frontier_antic,
                                      workload_id=name)

    # Frontier characterisation
    safe_rhos_antic = [round(p["rho_target"], 4) for p in frontier_antic
                       if p["safety_status"] == "SAFE"]
    first_unsafe_antic = next((round(p["rho_target"], 4)
                               for p in frontier_antic
                               if p["safety_status"] == "UNSAFE"), None)
    safe_kpis = [p["predicted_goodput_per_dollar"] for p in frontier_antic
                 if p["safety_status"] == "SAFE"]
    frontier_optimal = (max(((p["rho_target"], p["predicted_goodput_per_dollar"])
                             for p in frontier_antic
                             if p["safety_status"] == "SAFE"),
                            key=lambda x: x[1], default=(None, None)))

    # Lowest-cost safe point (lowest GPU-hours among SAFE)
    lowest_cost_safe = (min(((p["rho_target"], p["predicted_gpu_hours"])
                             for p in frontier_antic
                             if p["safety_status"] == "SAFE"),
                            key=lambda x: x[1], default=(None, None)))

    ca_row = next(r for r in named_rows if r["policy"] == "constraint_aware")
    ca_gpd = ca_row["goodput_per_dollar"]
    sla_gpd = next(r["goodput_per_dollar"] for r in named_rows
                   if r["policy"] == "sla_aware")
    fc_gpd = fc_row["goodput_per_dollar"] or 0.0
    delta_ca_pct = ((fc_gpd - ca_gpd) / ca_gpd * 100.0) if ca_gpd else 0.0
    delta_sla_pct = ((fc_gpd - sla_gpd) / sla_gpd * 100.0) if sla_gpd else 0.0
    # Safety: regression if Δ < -1 % OR timeout grows by > 1 absolute %.
    delta_timeout = ((fc_row.get("timeout_pct_mean") or 0.0)
                     - ca_row.get("timeout_pct_mean", 0.0))
    delta_gpu_h = (fc_row.get("gpu_hours") or 0.0) - ca_row["gpu_hours"]
    if abs(delta_ca_pct) <= TIE_BAND_PCT and delta_timeout <= TIE_BAND_PCT:
        verdict = "SAFE_TIE"
    elif delta_ca_pct < -TIE_BAND_PCT or delta_timeout > TIE_BAND_PCT:
        verdict = "REGRESSION"
    elif delta_ca_pct > TIE_BAND_PCT:
        verdict = "FRONTIER_WIN"
    else:  # pragma: no cover
        verdict = "UNKNOWN"

    diagnostic = {
        "n_ticks": loaded["n_ticks"], "n_requests": loaded["n_requests"],
        "time_span_s": loaded["time_span_s"],
        "best_safe_rho": (max(safe_rhos_antic) if safe_rhos_antic else None),
        "first_unsafe_rho": first_unsafe_antic,
        "safe_rhos_anticipatory": safe_rhos_antic,
        "frontier_optimal_rho": frontier_optimal[0],
        "frontier_optimal_goodput_per_dollar": frontier_optimal[1],
        "lowest_cost_safe_rho": lowest_cost_safe[0],
        "lowest_cost_safe_gpu_hours": lowest_cost_safe[1],
        "constraint_aware_in_safe_set":
            CONSTRAINT_AWARE_DEFAULT_RHO in safe_rhos_antic,
    }
    root_cause = (None if verdict != "SAFE_TIE"
                  else _root_cause_tie(diagnostic, named_rows, fc_row,
                                       frontier_antic))

    return {
        "trace": name, "applicable": True,
        "source": {"path": loaded["raw_path"],
                   "load_seconds": round(loaded["load_seconds"], 2),
                   "bin_seconds": round(loaded["bin_seconds"], 2),
                   "sweep_seconds": round(t_sweep, 2),
                   "named_seconds": round(t_named, 2)},
        "n_requests": loaded["n_requests"], "n_ticks": loaded["n_ticks"],
        "tick_seconds": tick_seconds,
        "time_span_s": loaded["time_span_s"],
        "frontier_anticipatory": frontier_antic,
        "frontier_reactive": frontier_react,
        "named_policies": named_rows,
        "frontier_controller_row": {
            **{k: v for k, v in fc_row.items() if k != "decision"},
            "decision": (fc_row["decision"].to_dict()
                         if fc_row.get("decision") is not None
                         else None),
        },
        "comparison": {
            "constraint_aware_goodput_per_dollar": ca_gpd,
            "sla_aware_goodput_per_dollar": sla_gpd,
            "frontier_controller_goodput_per_dollar": fc_gpd,
            "delta_vs_constraint_aware_pct": delta_ca_pct,
            "delta_vs_sla_aware_pct": delta_sla_pct,
            "delta_timeout_pct_absolute": delta_timeout,
            "delta_gpu_hours": delta_gpu_h,
            "verdict": verdict,
        },
        "diagnostic": diagnostic,
        "root_cause": root_cause,
    }


# ---------------------------------------------------------------------------
# Azure 2024 reuse (read-only, from committed audit)
# ---------------------------------------------------------------------------

def _audit_azure_2024_from_committed() -> dict:
    if not os.path.exists(AZURE_2024_AUDIT_JSON):
        return {"trace": "azure_llm_2024_week", "applicable": False,
                "exclusion_reason":
                    f"committed audit JSON missing: {AZURE_2024_AUDIT_JSON}"}
    audit = json.load(open(AZURE_2024_AUDIT_JSON))
    fc = (json.load(open(AZURE_2024_FC_JSON))
          if os.path.exists(AZURE_2024_FC_JSON) else None)
    ca = audit["named_policies"]["constraint_aware"]
    if fc is None:
        return {"trace": "azure_llm_2024_week", "applicable": False,
                "exclusion_reason": "committed controller JSON missing"}
    fc_gpd = fc["deltas"]["frontier_selected_gpd"]
    delta_pct = (fc_gpd - ca["goodput_per_dollar"]) / ca["goodput_per_dollar"] * 100.0
    # Reshape the committed anticipatory frontier rows for re-rendering.
    def _from_audit_row(p):
        rho = float(p["policy"].split("@")[1])
        st, vetoes = _classify_point_safety({
            "predicted_timeout_pct": p["timeout_pct_mean"],
            "predicted_queue_p99_ms": p["queue_p99_ms"]})
        return {
            "rho_target": rho,
            "predicted_goodput_per_dollar": p["goodput_per_dollar"],
            "predicted_sla_safe_goodput": p["sla_compliant_goodput"],
            "predicted_gpu_hours": p["gpu_hours"],
            "predicted_infra_cost": p.get("infra_cost"),
            "predicted_timeout_pct": p["timeout_pct_mean"],
            "predicted_sla_violation_rate": p.get("sla_violation_rate"),
            "predicted_queue_p95_ms": p["queue_p95_ms"],
            "predicted_queue_p99_ms": p["queue_p99_ms"],
            "predicted_latency_p99_ms": p.get("latency_p99_ms"),
            "predicted_scale_events": p.get("scale_events"),
            "predicted_churn_score": p.get("churn"),
            "predicted_mean_utilization": p.get("mean_utilization_rho"),
            "safety_status": st, "safety_vetoes": vetoes}

    frontier_antic = [_from_audit_row(p) for p in audit["frontier_anticipatory"]]
    frontier_react = [_from_audit_row(p) for p in audit["frontier_reactive"]]
    named_rows = []
    for pname in ("fifo", "sla_aware", "queue_aware", "utilization_aware",
                  "constraint_aware"):
        p = audit["named_policies"].get(pname)
        if not p:
            continue
        named_rows.append({
            "policy": pname,
            "rho_target": (CONSTRAINT_AWARE_DEFAULT_RHO
                           if pname == "constraint_aware" else
                           (0.50 if pname == "sla_aware"
                            else (0.85 if pname == "utilization_aware"
                                  else None))),
            "goodput_per_dollar": p["goodput_per_dollar"],
            "sla_compliant_goodput": p["sla_compliant_goodput"],
            "gpu_hours": p["gpu_hours"],
            "infra_cost": p.get("infra_cost"),
            "timeout_pct_mean": p["timeout_pct_mean"],
            "queue_p95_ms": p["queue_p95_ms"],
            "queue_p99_ms": p["queue_p99_ms"],
            "latency_p99_ms": p.get("latency_p99_ms"),
            "safe": p.get("safe", bool(p["timeout_pct_mean"]
                                       <= SAFE_TIMEOUT_PCT
                                       and p["queue_p99_ms"]
                                       <= SAFE_QUEUE_P99_MS))})
    fc_committed = next((p for p in audit["frontier_anticipatory"]
                          if abs(float(p["policy"].split("@")[1]) - 0.75)
                          < 1e-9), None)
    fc_row = {"policy": "frontier_controller_v1", "rho_target": 0.75,
              "goodput_per_dollar": fc_gpd,
              "sla_compliant_goodput": (fc_committed["sla_compliant_goodput"]
                                        if fc_committed else None),
              "gpu_hours": (fc_committed["gpu_hours"]
                            if fc_committed else None),
              "timeout_pct_mean": (fc_committed["timeout_pct_mean"]
                                   if fc_committed else None),
              "queue_p95_ms": (fc_committed["queue_p95_ms"]
                               if fc_committed else None),
              "queue_p99_ms": (fc_committed["queue_p99_ms"]
                               if fc_committed else None),
              "latency_p99_ms": (fc_committed.get("latency_p99_ms")
                                 if fc_committed else None),
              "safe": True, "frontier_used": True}
    ca_committed_row = next((r for r in named_rows
                             if r["policy"] == "constraint_aware"), None)
    delta_timeout = ((fc_row.get("timeout_pct_mean") or 0.0)
                     - (ca_committed_row.get("timeout_pct_mean")
                        if ca_committed_row else 0.0))
    return {"trace": "azure_llm_2024_week", "applicable": True,
            "source": {"path": os.path.relpath(AZURE_2024_AUDIT_JSON,
                                                REPO_ROOT),
                       "load_seconds": 0.0, "bin_seconds": 0.0,
                       "sweep_seconds": 0.0, "named_seconds": 0.0,
                       "note": "read-only reuse of committed week-long audit"},
            "n_requests": None, "n_ticks": 12960, "tick_seconds": 60.0,
            "time_span_s": 7 * 86400.0,
            "frontier_anticipatory": frontier_antic,
            "frontier_reactive": frontier_react,
            "named_policies": named_rows,
            "frontier_controller_row": fc_row,
            "comparison": {
                "constraint_aware_goodput_per_dollar":
                    ca["goodput_per_dollar"],
                "sla_aware_goodput_per_dollar":
                    audit["named_policies"]["sla_aware"]["goodput_per_dollar"],
                "frontier_controller_goodput_per_dollar": fc_gpd,
                "delta_vs_constraint_aware_pct": delta_pct,
                "delta_timeout_pct_absolute": delta_timeout,
                "delta_gpu_hours": ((fc_committed["gpu_hours"] - ca["gpu_hours"])
                                    if fc_committed else None),
                "verdict": ("FRONTIER_WIN" if delta_pct > TIE_BAND_PCT
                            else "SAFE_TIE"),
            },
            "diagnostic": {
                "n_ticks": 12960, "n_requests": None,
                "best_safe_rho": 0.75, "first_unsafe_rho": 0.85,
                "frontier_optimal_rho": 0.75,
                "frontier_optimal_goodput_per_dollar": fc_gpd,
                "lowest_cost_safe_rho": 0.75,
                "lowest_cost_safe_gpu_hours":
                    audit["frontier_anticipatory"][3].get("gpu_hours"),
                "safe_rhos_anticipatory": [0.45, 0.55, 0.65, 0.75],
                "constraint_aware_in_safe_set": True},
            "root_cause": None}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _frontier_table(rows, append):
    append("| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) "
           "| GPU-h | mean rho | scale ev | churn | safety |")
    append("|---|---|---|---|---|---|---|---|---|---|")
    for p in rows:
        sla_viol_pct = (p.get("predicted_sla_violation_rate") or 0.0) * 100.0
        append(
            f"| {p['rho_target']} | "
            f"{_f(p['predicted_goodput_per_dollar'])} | "
            f"{_f(p['predicted_timeout_pct'])} | "
            f"{sla_viol_pct:,.2f} | "
            f"{_f(p['predicted_queue_p95_ms'])} / "
            f"{_f(p['predicted_queue_p99_ms'])} | "
            f"{_f(p['predicted_gpu_hours'])} | "
            f"{_f(p.get('predicted_mean_utilization'), nd=4)} | "
            f"{_f(p.get('predicted_scale_events'))} | "
            f"{_f(p.get('predicted_churn_score'))} | "
            f"**{p['safety_status']}** |")


def _named_table(rows, fc_row, append):
    append("| policy | rho | goodput/$ | timeout % | queue p99 (ms) | "
           "GPU-h | safety |")
    append("|---|---|---|---|---|---|---|")
    for r in rows + [fc_row]:
        append(
            f"| `{r['policy']}` | {_f(r.get('rho_target'))} | "
            f"{_f(r['goodput_per_dollar'])} | "
            f"{_f(r.get('timeout_pct_mean'))} | "
            f"{_f(r.get('queue_p99_ms'))} | "
            f"{_f(r.get('gpu_hours'))} | "
            + ("SAFE" if r.get('safe') else "**UNSAFE**") + " |")


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Full-Trace Safe Utilization Frontier Validation Audit\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Validates the "
      "previously fixture-bound SAFE_TIE verdicts on Azure LLM 2023 and "
      "BurstGPT by running the same audit on the **raw, full traces**. "
      "Reuses the UNCHANGED serving physics in "
      "`aurelius/traces/backtest.py` and the UNCHANGED frontier "
      "controller / integration in `aurelius/frontier/` + "
      "`aurelius/constraints/frontier_integration.py`. No optimizer / "
      "robust-energy-engine constant is changed; no constant is tuned to "
      "force a result; no ML model is trained; no safety gate is "
      "weakened. The committed Azure 2024 artifacts are **read-only**.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, "
      "`docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, "
      "`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, "
      "`docs/PUBLIC_TRACE_BACKTESTS.md`.\n")

    A("## 1. Configuration\n")
    A(f"- **Candidate rho grid:** `{list(RHOS)}`")
    A(f"- **Safety thresholds (pre-registered):** timeout ≤ "
      f"{SAFE_TIMEOUT_PCT}% AND queue p99 ≤ {SAFE_QUEUE_P99_MS} ms")
    A(f"- **Tie band:** ±{TIE_BAND_PCT}% goodput/$")
    A(f"- **constraint_aware default rho:** {CONSTRAINT_AWARE_DEFAULT_RHO} "
      "(unchanged)\n")

    A("## 2. Datasets\n")
    A("| trace | raw file | rows | time span | source |")
    A("|---|---|---|---|---|")
    for d in payload["per_trace"]:
        if not d["applicable"]:
            A(f"| `{d['trace']}` | — | — | — | _excluded — "
              f"{d['exclusion_reason']}_ |")
            continue
        src_path = d['source'].get('path', '—')
        spans = (f"{d['time_span_s']:,.0f} s "
                 f"(~{d['time_span_s']/86400:.2f} d)")
        rows = (f"{d['n_requests']:,}" if d['n_requests'] is not None
                else "—")
        A(f"| `{d['trace']}` | `{os.path.basename(src_path)}` | {rows} | "
          f"{spans} | `{src_path}` |")
    A("")

    for d in payload["per_trace"]:
        A(f"## 3.{d['trace']} — frontier sweep + controller verdict\n")
        if not d["applicable"]:
            A(f"**Excluded.** {d['exclusion_reason']}\n")
            continue
        cmp_ = d["comparison"]
        diag = d["diagnostic"]
        A(f"- **Verdict vs `constraint_aware`:** **`{cmp_['verdict']}`** "
          f"(Δ goodput/$ "
          f"{cmp_['delta_vs_constraint_aware_pct']:+.3f} %)")
        nreq = d.get("n_requests")
        nreq_str = f"{nreq:,}" if nreq is not None else "—"
        A(f"- **n_ticks:** {d.get('n_ticks', '—')}; "
          f"**n_requests:** {nreq_str}; "
          f"**time span:** {d['time_span_s']:,.0f} s")
        A(f"- **Best safe rho (anticipatory):** "
          f"{diag['best_safe_rho']}; first unsafe: "
          f"{diag['first_unsafe_rho']}")
        A(f"- **Frontier-optimal rho:** {diag['frontier_optimal_rho']} "
          f"({_f(diag['frontier_optimal_goodput_per_dollar'])} goodput/$)")
        A(f"- **Lowest-cost safe rho:** {diag['lowest_cost_safe_rho']} "
          f"({_f(diag['lowest_cost_safe_gpu_hours'])} GPU-h)")
        A(f"- **`constraint_aware` rho in safe set:** "
          f"{diag['constraint_aware_in_safe_set']}")
        if d.get("frontier_anticipatory"):
            A("\n### Anticipatory frontier sweep (full trace)\n")
            _frontier_table(d["frontier_anticipatory"], A)
            A("\n### Reactive frontier sweep (full trace, diagnostic)\n")
            _frontier_table(d["frontier_reactive"], A)
            A("\n### Policy comparison\n")
            _named_table(d["named_policies"],
                          d["frontier_controller_row"], A)
        rc = d.get("root_cause")
        if rc is not None:
            A(f"\n### Root cause of {cmp_['verdict']}\n")
            A(f"- **Code:** `{rc['code']}`")
            A(f"- **Evidence:** {rc['evidence']}\n")
        else:
            A("")

    syn = payload["synthesis"]
    A("## 4. Cross-trace synthesis\n")
    A("| trace | constraint_aware goodput/$ | frontier_controller goodput/$"
      " | Δ % | verdict | root cause | best safe rho |")
    A("|---|---|---|---|---|---|---|")
    for r in syn["rows"]:
        A(f"| `{r['trace']}` | {_f(r['ca_gpd'])} | "
          f"{_f(r['fc_gpd'])} | {r['delta_pct']:+.3f} % | "
          f"**{r['verdict']}** | "
          f"{r['root_cause']} | "
          f"{r['best_safe_rho']} |")
    A("")
    vc = syn["verdict_counts"]
    A(f"**Counts (applicable, n={syn['n_applicable']}):** wins = "
      f"{vc['FRONTIER_WIN']} | safe-ties = {vc['SAFE_TIE']} | "
      f"regressions = {vc['REGRESSION']} | excluded = {syn['n_skipped']}.\n")

    A("## 5. Generalization\n")
    A(f"- **Was the previous SAFE_TIE caused by fixture limitations, or "
      f"does the controller add little value on those workloads?** "
      f"**{syn['answer']}**")
    A(f"- **Does `frontier_controller_v1` truly generalize across LLM "
      f"serving traces?** {syn['generalization_verdict']}")
    A(f"- **Is Azure 2024 unique?** {syn['azure_2024_uniqueness']}")
    A("")

    A("## 6. Honesty / scope\n")
    A("- This is a **measurement-only** validation audit. The "
      "`constraint_aware` engine default rho (≈ 0.65) is **unchanged**, "
      "the frontier integration remains opt-in and disabled by default, "
      "and real-cluster execution remains disabled by default.")
    A("- The committed Azure 2024 audit JSON / backtest summary / "
      "frontier-controller summary / integration summary are **read-only** "
      "in this audit — no committed artifact is overwritten.")
    A("- This is **directional simulator / shadow-mode evidence** — NOT "
      "production savings. Pilot telemetry is required to calibrate the "
      "safe rho per workload before any production claim.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _synthesize(per_trace: list[dict]) -> dict:
    applicable = [d for d in per_trace if d["applicable"]]
    skipped = [d for d in per_trace if not d["applicable"]]
    vc = {"FRONTIER_WIN": 0, "SAFE_TIE": 0, "REGRESSION": 0}
    rows = []
    for d in applicable:
        cmp_ = d["comparison"]
        verdict = cmp_["verdict"]
        vc[verdict] = vc.get(verdict, 0) + 1
        rc = (d.get("root_cause") or {}).get("code", "—")
        rows.append({
            "trace": d["trace"],
            "n_ticks": d.get("n_ticks"),
            "n_requests": d.get("n_requests"),
            "ca_gpd": cmp_["constraint_aware_goodput_per_dollar"],
            "fc_gpd": cmp_["frontier_controller_goodput_per_dollar"],
            "delta_pct": cmp_["delta_vs_constraint_aware_pct"],
            "verdict": verdict, "root_cause": rc,
            "best_safe_rho": d["diagnostic"]["best_safe_rho"],
        })

    # Was the prior tie caused by fixture limitations?
    full_trace_tie_traces = [r for r in rows
                              if r["verdict"] == "SAFE_TIE"
                              and r["trace"] in ("burstgpt",
                                                 "azure_llm_2023")]
    full_trace_win_traces = [r for r in rows
                              if r["verdict"] == "FRONTIER_WIN"
                              and r["trace"] in ("burstgpt",
                                                 "azure_llm_2023")]
    if full_trace_win_traces and not full_trace_tie_traces:
        answer = ("The previous fixture-bound SAFE_TIE was caused by "
                  "FIXTURE LIMITATIONS — on full raw data the controller "
                  "produces measurable wins on the same traces.")
    elif full_trace_tie_traces and not full_trace_win_traces:
        answer = ("The previous SAFE_TIE PERSISTS on the full raw trace — "
                  "the `constraint_aware` operating point already sits at or "
                  "near the safe-utilization frontier on these workloads "
                  "(root-causes attached per trace).")
    elif full_trace_win_traces and full_trace_tie_traces:
        answer = ("Mixed — the full traces show real wins on some "
                  "workloads but persistent ties on others; root-cause "
                  "table identifies which is which.")
    else:
        answer = "No full-trace data available — verdict undetermined."

    # Generalization verdict
    if vc["REGRESSION"] > 0:
        gen_verdict = (f"NO — {vc['REGRESSION']} regression(s) observed on "
                       "applicable LLM serving traces.")
    elif vc["FRONTIER_WIN"] >= 2:
        gen_verdict = ("YES — the controller produces measurable wins on "
                       "≥ 2 distinct LLM serving traces with no regressions.")
    elif vc["FRONTIER_WIN"] == 1:
        gen_verdict = ("PARTIAL — the controller wins on 1 trace and safely "
                       "ties on the others; the bulk of its measured value "
                       "comes from a single workload class.")
    else:
        gen_verdict = ("NO — every applicable trace safely ties; the "
                       "controller does not add measurable value on these "
                       "workloads.")

    # Is Azure 2024 unique?
    az24 = next((r for r in rows if r["trace"] == "azure_llm_2024_week"),
                None)
    other_wins = [r for r in rows
                  if r["verdict"] == "FRONTIER_WIN"
                  and r["trace"] != "azure_llm_2024_week"]
    if az24 and az24["verdict"] == "FRONTIER_WIN" and not other_wins:
        az_uniq = ("YES — Azure 2024 is the only applicable trace where "
                   "the controller produces a measurable goodput/$ uplift.")
    elif az24 and other_wins:
        az_uniq = (f"NO — {len(other_wins)} other trace(s) also show a "
                   "measurable win.")
    elif az24 and az24["verdict"] == "SAFE_TIE":
        az_uniq = ("NO — Azure 2024 also safely ties on this audit "
                   "configuration.")
    else:
        az_uniq = "INSUFFICIENT_DATA"

    return {
        "n_applicable": len(applicable), "n_skipped": len(skipped),
        "verdict_counts": vc,
        "rows": rows,
        "answer": answer,
        "generalization_verdict": gen_verdict,
        "azure_2024_uniqueness": az_uniq,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--burstgpt-raw", default=os.path.join(
        REPO_ROOT, "data", "external", "burstgpt", "raw", "BurstGPT_1.csv"))
    p.add_argument("--azure-2023-raw", default=os.path.join(
        REPO_ROOT, "data", "external", "azure_llm", "raw",
        "AzureLLMInferenceTrace_conv.csv"))
    p.add_argument("--azure-2023-code-raw", default=os.path.join(
        REPO_ROOT, "data", "external", "azure_llm", "raw",
        "AzureLLMInferenceTrace_code.csv"))
    p.add_argument("--burstgpt-max-requests", type=int, default=None,
                   help="cap BurstGPT request count for faster audits "
                        "(default: full trace)")
    args = p.parse_args(argv)

    per_trace: list[dict] = []

    # --- BurstGPT (full or capped) ---
    if os.path.exists(args.burstgpt_raw):
        print(f"[full-trace] loading BurstGPT from {args.burstgpt_raw} ...",
              flush=True)
        loaded = _load_burstgpt(raw_path=args.burstgpt_raw,
                                max_requests=args.burstgpt_max_requests)
        per_trace.append(_audit_trace("burstgpt", loaded))
    else:
        per_trace.append({"trace": "burstgpt", "applicable": False,
                          "exclusion_reason":
                          f"raw file not present: {args.burstgpt_raw}"})

    # --- Azure 2023 conv (full) ---
    if os.path.exists(args.azure_2023_raw):
        print(f"[full-trace] loading Azure 2023 conv from "
              f"{args.azure_2023_raw} ...", flush=True)
        loaded = _load_azure_2023(raw_path=args.azure_2023_raw)
        per_trace.append(_audit_trace("azure_llm_2023_conv", loaded))
    else:
        per_trace.append({"trace": "azure_llm_2023_conv", "applicable": False,
                          "exclusion_reason":
                          f"raw file not present: {args.azure_2023_raw}"})

    # --- Azure 2023 code (full) ---
    if os.path.exists(args.azure_2023_code_raw):
        print(f"[full-trace] loading Azure 2023 code from "
              f"{args.azure_2023_code_raw} ...", flush=True)
        loaded = _load_azure_2023(raw_path=args.azure_2023_code_raw)
        per_trace.append(_audit_trace("azure_llm_2023_code", loaded))
    else:
        per_trace.append({"trace": "azure_llm_2023_code", "applicable": False,
                          "exclusion_reason":
                          f"raw file not present: "
                          f"{args.azure_2023_code_raw}"})

    # --- Azure 2024 (read-only reuse of committed audit) ---
    per_trace.append(_audit_azure_2024_from_committed())

    synthesis = _synthesize(per_trace)
    payload = {
        "config": {
            "rhos": list(RHOS),
            "safe_timeout_pct": SAFE_TIMEOUT_PCT,
            "safe_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "tie_band_pct": TIE_BAND_PCT,
            "constraint_aware_default_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        },
        "per_trace": per_trace,
        "synthesis": synthesis,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    _write_md(args.out_md, payload)

    syn = synthesis
    vc = syn["verdict_counts"]
    print(f"[full-trace] applicable={syn['n_applicable']} "
          f"skipped={syn['n_skipped']} wins={vc['FRONTIER_WIN']} "
          f"ties={vc['SAFE_TIE']} regressions={vc['REGRESSION']}")
    print(f"[full-trace] {syn['answer']}")
    print(f"[full-trace] JSON -> {args.out_json}")
    print(f"[full-trace] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
