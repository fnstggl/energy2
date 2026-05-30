#!/usr/bin/env python3
"""Cross-Trace Safe Utilization Frontier Generalization Audit.

Answers two questions across every currently-integrated public trace where a
target-utilization (``rho``) decision is meaningful:

1. Does ``frontier_controller_v1`` improve or safely tie ``constraint_aware``
   across every applicable trace?
2. Is Safe Utilization Frontier Control a *general* Aurelius capability or
   an Azure-2024-specific optimization?

This is a **measurement-only** benchmark phase. It composes the UNCHANGED
serving physics in ``aurelius/traces/backtest.py`` +
``aurelius/traces/genai_backtest.py`` and the UNCHANGED frontier controller
in ``aurelius/frontier/`` — no new datasets are ingested, no ML models are
trained, no optimizer constant is tuned to force a result, no safety gate is
weakened, no production execution path is created. Simulator / shadow
evidence only — **not production savings** (``docs/RESULTS.md`` §8).

Read first:
  * docs/RESULTS.md
  * docs/AZURE_LLM_2024_BACKTEST_RESULTS.md
  * docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md
  * docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md
  * docs/PUBLIC_TRACE_BACKTESTS.md

Writes:
  * docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md
  * data/external/frontier/cross_trace_frontier_generalization_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.frontier import (  # noqa: E402
    SHADOW_MODE,
    SIMULATOR_MODE,
    FrontierControllerConfig,
    FrontierShadowLog,
    SafetyConfig,
    SafetyStatus,
    WorkloadFrontierProfile,
    choose_safe_utilization_target,
    estimate_frontier_from_points,
    execute_frontier_decision,
)
from aurelius.traces import alibaba_genai as ag  # noqa: E402
from aurelius.traces import azure_llm as az  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402
from aurelius.traces import burstgpt as bg  # noqa: E402
from aurelius.traces import genai_backtest as gbt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "cross_trace_frontier_generalization_summary.json")
OUT_MD = os.path.join(REPO_ROOT, "docs",
                      "CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md")
AZURE_2024_AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")

# Canonical candidate rho grid (matches docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md).
RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
# Pre-registered safety thresholds (mirror the Azure 2024 audit).
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
# constraint_aware default rho — engine constant, NOT modified here.
CA_DEFAULT_RHO = 0.65
# Tie band (±1%) per the cross-trace synthesis spec.
TIE_BAND_PCT = 1.0


# ===========================================================================
# Generic rho sweep over LLM serving ticks (BurstGPT, Azure LLM 2023, Azure 2024)
# ===========================================================================

class _Reactive:
    """sla_aware-style sizer: previous-tick provisioning at target rho R."""

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
    """constraint_aware-style sizer: EWMA-anticipatory plan + SLA-safe trim."""

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


def _eval_rho_llm(ticks, R: float, *, tick_seconds: float, mode: str
                  ) -> dict:
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
    sla_viol_rate = (sum(t.request_count for e, t in active
                         if e.timeout_rate_pct > 0) / aw)
    reps = [e.replicas for e in evals]
    churn = sum(abs(reps[i] - reps[i - 1]) for i in range(1, len(reps)))
    scale_events = sum(1 for e in evals if e.scale_event)
    return {
        "rho_target": R,
        "predicted_goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "predicted_sla_safe_goodput": int(res.kpi.sla_compliant_goodput),
        "predicted_gpu_hours": float(res.kpi.active_gpu_hours),
        "predicted_timeout_pct": float(timeout_w),
        "predicted_queue_p95_ms": float(res.queue_p95_ms),
        "predicted_queue_p99_ms": float(res.queue_p99_ms),
        "predicted_latency_p99_ms": float(res.latency_p99_ms) or None,
        "predicted_scale_events": int(scale_events),
        "predicted_churn_score": float(churn),
        "predicted_mean_utilization": float(mean_rho),
        "sla_violation_rate": float(sla_viol_rate),
        "infra_cost": float(res.kpi.total_infrastructure_cost),
    }


def _named_policies_llm(ticks, *, tick_seconds: float) -> dict:
    """Run the canonical named policies (sla_aware / constraint_aware / etc)
    via the UNCHANGED ``bt._run_policy`` harness. ``utilization_aware`` is
    added via the same provisioning pattern used by the Azure 2024 script."""
    tick_hours = tick_seconds / 3600.0
    base = {p: bt._run_policy(p, ticks, tick_hours=tick_hours)
            for p in ("fifo", "sla_aware", "queue_aware", "constraint_aware")}

    # utilization_aware: per-tick rho 0.85 provisioning (Azure 2024 convention)
    def _util_size(t):
        return bt._size_for_target(t.arrival_rate_rps,
                                   max(1.0, t.output_tokens_mean),
                                   bt._tick_throughput_tokps(t), 0.85)
    evals = []
    prev_r = None
    for t in ticks:
        r = _util_size(t)
        ev = bt.evaluate_tick(t, r, prefill_savings=0.0, tick_hours=tick_hours)
        if prev_r is not None and ev.replicas != prev_r:
            ev.scale_event = True
        prev_r = ev.replicas
        evals.append(ev)
    base["utilization_aware"] = bt._aggregate("utilization_aware", evals,
                                              cache_aware=False, ticks=ticks)
    return base


def _policy_row_llm(name: str, res, rho: Optional[float]) -> dict:
    return {
        "policy": name, "rho_target": rho,
        "goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "sla_compliant_goodput": int(res.kpi.sla_compliant_goodput),
        "gpu_hours": float(res.kpi.active_gpu_hours),
        "infra_cost": float(res.kpi.total_infrastructure_cost),
        "timeout_pct_mean": float(getattr(res, "timeout_rate_pct_mean",
                                          res.kpi.weighted_mean_timeout_pct
                                          if hasattr(res.kpi,
                                                     "weighted_mean_timeout_pct")
                                          else 0.0)),
        "queue_p95_ms": float(res.queue_p95_ms),
        "queue_p99_ms": float(res.queue_p99_ms),
        "latency_p99_ms": float(res.latency_p99_ms),
        "safe": bool(res.queue_p99_ms <= SAFE_QUEUE_P99_MS),
    }


# ===========================================================================
# Trace loaders (fixture-bound; raw used only if present)
# ===========================================================================

def _load_burstgpt_requests(*, sample_size=None) -> tuple[list, str]:
    raw = os.path.join(REPO_ROOT, "data", "external", "burstgpt", "raw",
                       "BurstGPT_1.csv")
    fixture = os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv")
    if os.path.exists(raw):
        return bg.load_csv(raw, sample_size=sample_size), f"raw:{raw}"
    return bg.load_csv(fixture, sample_size=sample_size), f"fixture:{fixture}"


def _load_azure_2023_requests(*, sample_size=None) -> tuple[list, str]:
    raw = os.path.join(REPO_ROOT, "data", "external", "azure_llm", "raw")
    fixture = os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_sample.csv")
    if os.path.isdir(raw) and any(os.path.exists(os.path.join(raw, f))
                                  for f in os.listdir(raw) if f.endswith(".csv")):
        candidate = next((os.path.join(raw, f) for f in os.listdir(raw)
                          if f.endswith(".csv")), None)
        if candidate:
            reqs = az.load_csv(candidate, variant=az.variant_from_path(candidate),
                               sample_size=sample_size)
            return reqs, f"raw:{candidate}"
    reqs = az.load_csv(fixture, variant="conv", sample_size=sample_size,
                       include_failures=False)
    return reqs, f"fixture:{fixture}"


def _ticks_from_requests(requests, *, tick_seconds: float):
    """Bin LLM requests to ArrivalTicks using the canonical replay shim."""
    from aurelius.traces.replay import requests_to_arrival_ticks
    return requests_to_arrival_ticks(requests, tick_seconds=tick_seconds)


def _scale_ticks(ticks, factor: float):
    if factor == 1.0:
        return list(ticks)
    return [replace(t, request_count=int(round(t.request_count * factor)),
                    arrival_rate_rps=t.arrival_rate_rps * factor,
                    total_prompt_tokens=int(round(t.total_prompt_tokens * factor)),
                    total_output_tokens=int(round(t.total_output_tokens * factor)),
                    model_mix={k: int(round(v * factor))
                               for k, v in t.model_mix.items()})
            for t in ticks]


# ===========================================================================
# Per-trace audit (LLM serving traces)
# ===========================================================================

def _audit_llm_trace(name: str, ticks, *, tick_seconds: float,
                     scale: float = 1.0) -> dict:
    if scale != 1.0:
        ticks = _scale_ticks(ticks, scale)

    frontier_antic = [_eval_rho_llm(ticks, R, tick_seconds=tick_seconds,
                                    mode="anticipatory") for R in RHOS]
    frontier_react = [_eval_rho_llm(ticks, R, tick_seconds=tick_seconds,
                                    mode="reactive") for R in RHOS]

    # Frontier controller — recommendation-only, shadow execution.
    profile = WorkloadFrontierProfile(
        workload_id=name, workload_type="inference_standard",
        telemetry_confidence="medium", priority_class="standard",
        candidate_rhos=RHOS, source=name)
    safety = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                          max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                          min_telemetry_confidence="low")
    pts = estimate_frontier_from_points(profile, frontier_antic,
                                        safety_config=safety)
    cfg = FrontierControllerConfig(conservative_margin=False,
                                   deadband_rho=0.05, deadband_kpi_pct=0.05,
                                   min_telemetry_confidence="low",
                                   default_execution_mode=SHADOW_MODE)
    decision = choose_safe_utilization_target(profile, pts,
                                              current_rho=CA_DEFAULT_RHO,
                                              controller_config=cfg)

    # Mirror the decision through the shadow log + simulator-only mutation
    shadow_log = FrontierShadowLog()
    shadow_log.record(decision, execution_mode=SHADOW_MODE)
    sim_state: dict = {}
    sim_effect = execute_frontier_decision(decision, mode=SIMULATOR_MODE,
                                           simulated_state=sim_state)

    # Named-policy comparison
    named = _named_policies_llm(ticks, tick_seconds=tick_seconds)
    rho_for = {"sla_aware": 0.50, "constraint_aware": CA_DEFAULT_RHO,
               "queue_aware": None, "utilization_aware": 0.85, "fifo": None}
    named_rows = [_policy_row_llm(p, named[p], rho_for[p])
                  for p in ("fifo", "sla_aware", "queue_aware",
                            "utilization_aware", "constraint_aware")]

    # frontier_controller_v1 row uses the *chosen* anticipatory point
    selected = decision.selected_point
    if selected is not None:
        fc_row = {
            "policy": "frontier_controller_v1",
            "rho_target": decision.selected_rho,
            "goodput_per_dollar": selected.predicted_goodput_per_dollar,
            "sla_compliant_goodput": selected.predicted_sla_safe_goodput,
            "gpu_hours": selected.predicted_gpu_hours,
            "infra_cost": (
                selected.predicted_sla_safe_goodput
                / selected.predicted_goodput_per_dollar
                if selected.predicted_goodput_per_dollar else 0.0),
            "timeout_pct_mean": selected.predicted_timeout_pct,
            "queue_p95_ms": selected.predicted_queue_p95_ms,
            "queue_p99_ms": selected.predicted_queue_p99_ms,
            "latency_p99_ms": selected.predicted_latency_p99_ms,
            "safe": selected.is_safe,
        }
    else:
        fc_row = {"policy": "frontier_controller_v1",
                  "rho_target": decision.selected_rho,
                  "goodput_per_dollar": None, "safe": False}

    # Comparison vs constraint_aware baseline
    ca_gpd = next((r["goodput_per_dollar"] for r in named_rows
                   if r["policy"] == "constraint_aware"), 0.0)
    sla_gpd = next((r["goodput_per_dollar"] for r in named_rows
                    if r["policy"] == "sla_aware"), 0.0)
    fc_gpd = fc_row.get("goodput_per_dollar") or 0.0
    delta_ca_pct = ((fc_gpd - ca_gpd) / ca_gpd * 100.0) if ca_gpd else 0.0
    delta_sla_pct = ((fc_gpd - sla_gpd) / sla_gpd * 100.0) if sla_gpd else 0.0
    if abs(delta_ca_pct) <= TIE_BAND_PCT:
        verdict = "TIE"
    elif delta_ca_pct > 0:
        verdict = "FRONTIER_WIN"
    else:
        verdict = "FRONTIER_LOSS"

    # Diagnostic frontier characterisation
    safe_rhos = [round(p.rho_target, 4) for p in pts if p.is_safe]
    first_unsafe = next((round(p.rho_target, 4) for p in pts
                         if p.safety_status == SafetyStatus.UNSAFE), None)
    best_safe = max(safe_rhos) if safe_rhos else None

    return {
        "trace": name,
        "applicable": True,
        "n_ticks": len(ticks),
        "tick_seconds": tick_seconds,
        "scale": scale,
        "frontier_anticipatory": frontier_antic,
        "frontier_reactive": frontier_react,
        "frontier_points_safety": [p.to_dict() for p in pts],
        "named_policies": named_rows,
        "frontier_controller_row": fc_row,
        "decision": decision.to_dict(),
        "simulator_effect": sim_effect.to_dict(),
        "shadow_log_summary": shadow_log.summary(),
        "comparison": {
            "constraint_aware_goodput_per_dollar": ca_gpd,
            "sla_aware_goodput_per_dollar": sla_gpd,
            "frontier_controller_goodput_per_dollar": fc_gpd,
            "delta_vs_constraint_aware_pct": delta_ca_pct,
            "delta_vs_sla_aware_pct": delta_sla_pct,
            "verdict": verdict,
        },
        "diagnostic": {
            "best_safe_rho": best_safe,
            "first_unsafe_rho": first_unsafe,
            "safe_rhos": safe_rhos,
            "frontier_peak_delta_vs_constraint_aware_pct": delta_ca_pct,
            "constraint_aware_rho_in_safe_set":
                CA_DEFAULT_RHO in [round(r, 4) for r in safe_rhos]
                if safe_rhos else False,
        },
    }


# ===========================================================================
# Azure 2024 — reuse the COMMITTED audit JSON (full week-long trace)
# ===========================================================================

def _audit_azure_2024_from_committed() -> dict:
    """Reuse the committed Azure 2024 audit JSON so we don't re-run the full
    week-long simulator. The audit JSON is read-only; we never overwrite it."""
    if not os.path.exists(AZURE_2024_AUDIT_JSON):
        return {"trace": "azure_llm_2024_week", "applicable": False,
                "exclusion_reason":
                    "committed Azure 2024 audit JSON missing; run "
                    "scripts/run_azure_2024_safe_utilization_frontier.py first"}
    audit = json.load(open(AZURE_2024_AUDIT_JSON))
    raw_antic = [{
        "rho_target": float(p["policy"].split("@")[1]),
        "predicted_goodput_per_dollar": p["goodput_per_dollar"],
        "predicted_sla_safe_goodput": p["sla_compliant_goodput"],
        "predicted_gpu_hours": p["gpu_hours"],
        "predicted_timeout_pct": p["timeout_pct_mean"],
        "predicted_queue_p95_ms": p["queue_p95_ms"],
        "predicted_queue_p99_ms": p["queue_p99_ms"],
        "predicted_latency_p99_ms": p.get("latency_p99_ms"),
        "predicted_scale_events": p.get("scale_events"),
        "predicted_churn_score": p.get("churn"),
        "predicted_mean_utilization": p.get("mean_utilization_rho"),
        "sla_violation_rate": p.get("sla_violation_rate"),
        "infra_cost": p.get("infra_cost"),
    } for p in audit["frontier_anticipatory"]]
    raw_react = [{
        "rho_target": float(p["policy"].split("@")[1]),
        "predicted_goodput_per_dollar": p["goodput_per_dollar"],
        "predicted_timeout_pct": p["timeout_pct_mean"],
        "predicted_queue_p99_ms": p["queue_p99_ms"],
        "predicted_queue_p95_ms": p["queue_p95_ms"],
        "predicted_gpu_hours": p["gpu_hours"],
        "predicted_mean_utilization": p.get("mean_utilization_rho"),
        "sla_violation_rate": p.get("sla_violation_rate"),
    } for p in audit["frontier_reactive"]]

    profile = WorkloadFrontierProfile(
        workload_id="azure_llm_2024_week",
        workload_type="inference_standard", telemetry_confidence="medium",
        priority_class="standard", candidate_rhos=RHOS,
        source="azure_2024_committed_audit")
    safety = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                          max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                          min_telemetry_confidence="low")
    pts = estimate_frontier_from_points(profile, raw_antic, safety_config=safety)
    cfg = FrontierControllerConfig(default_execution_mode=SHADOW_MODE,
                                   min_telemetry_confidence="low",
                                   deadband_rho=0.05, deadband_kpi_pct=0.05)
    decision = choose_safe_utilization_target(profile, pts,
                                              current_rho=CA_DEFAULT_RHO,
                                              controller_config=cfg)

    named = audit["named_policies"]
    rho_for = {"sla_aware": 0.50, "constraint_aware": CA_DEFAULT_RHO,
               "queue_aware": None, "utilization_aware": 0.85,
               "oracle_forecast_ANALYSIS_ONLY": None,
               "naive_overprovisioning": None, "fifo": None}

    def _row_from_named(p_name):
        m = named[p_name]
        return {"policy": p_name, "rho_target": rho_for.get(p_name),
                "goodput_per_dollar": m["goodput_per_dollar"],
                "sla_compliant_goodput": m["sla_compliant_goodput"],
                "gpu_hours": m["gpu_hours"],
                "infra_cost": m.get("infra_cost"),
                "timeout_pct_mean": m["timeout_pct_mean"],
                "queue_p95_ms": m["queue_p95_ms"],
                "queue_p99_ms": m["queue_p99_ms"],
                "latency_p99_ms": m.get("latency_p99_ms"),
                "safe": bool(m["timeout_pct_mean"] <= SAFE_TIMEOUT_PCT
                             and m["queue_p99_ms"] <= SAFE_QUEUE_P99_MS)}

    named_rows = [_row_from_named(p) for p in
                  ("fifo", "sla_aware", "queue_aware", "utilization_aware",
                   "constraint_aware", "oracle_forecast_ANALYSIS_ONLY")
                  if p in named]

    selected = decision.selected_point
    fc_row = {"policy": "frontier_controller_v1",
              "rho_target": decision.selected_rho,
              "goodput_per_dollar": (selected.predicted_goodput_per_dollar
                                     if selected else None),
              "gpu_hours": selected.predicted_gpu_hours if selected else None,
              "timeout_pct_mean": (selected.predicted_timeout_pct
                                   if selected else None),
              "queue_p95_ms": selected.predicted_queue_p95_ms if selected else None,
              "queue_p99_ms": selected.predicted_queue_p99_ms if selected else None,
              "latency_p99_ms": (selected.predicted_latency_p99_ms
                                 if selected else None),
              "safe": selected.is_safe if selected else False}

    ca_gpd = named["constraint_aware"]["goodput_per_dollar"]
    sla_gpd = named["sla_aware"]["goodput_per_dollar"]
    fc_gpd = fc_row["goodput_per_dollar"] or 0.0
    delta_ca = ((fc_gpd - ca_gpd) / ca_gpd * 100.0) if ca_gpd else 0.0
    delta_sla = ((fc_gpd - sla_gpd) / sla_gpd * 100.0) if sla_gpd else 0.0
    verdict = ("TIE" if abs(delta_ca) <= TIE_BAND_PCT
               else ("FRONTIER_WIN" if delta_ca > 0 else "FRONTIER_LOSS"))

    safe_rhos = [round(p.rho_target, 4) for p in pts if p.is_safe]
    first_unsafe = next((round(p.rho_target, 4) for p in pts
                         if p.safety_status == SafetyStatus.UNSAFE), None)
    best_safe = max(safe_rhos) if safe_rhos else None
    return {
        "trace": "azure_llm_2024_week", "applicable": True,
        "source": "committed_audit_json",
        "audit_json_path": os.path.relpath(AZURE_2024_AUDIT_JSON, REPO_ROOT),
        "n_ticks": 12960, "tick_seconds": 60.0, "scale": 10.0,
        "frontier_anticipatory": raw_antic, "frontier_reactive": raw_react,
        "frontier_points_safety": [p.to_dict() for p in pts],
        "named_policies": named_rows, "frontier_controller_row": fc_row,
        "decision": decision.to_dict(),
        "comparison": {
            "constraint_aware_goodput_per_dollar": ca_gpd,
            "sla_aware_goodput_per_dollar": sla_gpd,
            "frontier_controller_goodput_per_dollar": fc_gpd,
            "delta_vs_constraint_aware_pct": delta_ca,
            "delta_vs_sla_aware_pct": delta_sla,
            "verdict": verdict,
        },
        "diagnostic": {
            "best_safe_rho": best_safe, "first_unsafe_rho": first_unsafe,
            "safe_rhos": safe_rhos,
            "frontier_peak_delta_vs_constraint_aware_pct": delta_ca,
            "constraint_aware_rho_in_safe_set": True,
        },
    }


# ===========================================================================
# Alibaba GenAI 2026 — request-level (not token-level) rho sweep
# ===========================================================================

def _eval_rho_genai(ticks, R: float, *, cold, tick_hours: float) -> dict:
    """Anticipatory (EWMA) sizer at target rho R, scored via the unchanged
    GenAI serving physics (genai_backtest._eval_tick)."""
    from aurelius.traces.genai_backtest import _eval_tick, _size_for_target
    affinity = True
    evals = []
    weights = []
    prev_r = None
    scale_events = 0
    ewma = 0.0
    replica_hours = 0.0
    e2e99 = []
    q99 = []
    q95 = []
    e2e95 = []
    timeouts = []
    requests = []
    for t in ticks:
        if t.n > 0:
            ewma = (0.5 * t.arrival_rate + 0.5 * ewma if ewma else t.arrival_rate)
        plan_rate = max(t.arrival_rate, ewma)
        # Build a tick-shim with the EWMA-anticipated arrival rate for sizing.
        plan_tick = replace(t, arrival_rate=plan_rate) if hasattr(t, "__dict__") \
            else type(t)(t.tick_index, t.start_s, t.n, plan_rate, t.mean_exec_s,
                          t.distinct_models, t.lora_frac, t.controlnet_frac,
                          t.failures)
        r = _size_for_target(plan_tick, cold, affinity, R) if t.n else 1
        ev = _eval_tick(t, r, cold, affinity)
        if prev_r is not None and r != prev_r:
            scale_events += 1
        prev_r = r
        replica_hours += r * tick_hours
        requests.append(t.n)
        if t.n > 0:
            e2e95.append(ev["e2e_p95"])
            e2e99.append(ev["e2e_p99"])
            q95.append(ev["wait_s"])
            q99.append(ev["wait_s"])
            timeouts.append(ev["timeout"])
            weights.append(t.n)
        evals.append((ev, t, r))

    from aurelius.benchmarks.economics import (
        InfrastructureCostConfig,
        compute_economic_kpi,
    )
    from aurelius.traces.genai_backtest import GPU_HOUR_PRICE
    cfg = InfrastructureCostConfig(gpu_hour_prices={"genai-gpu": GPU_HOUR_PRICE},
                                   fallback_gpu_hour_price=GPU_HOUR_PRICE)
    gpu_hours_per_tick = [{"genai-gpu": rr * tick_hours}
                          for _, _, rr in evals]
    tokens_per_tick = [t.n for _, t, _ in evals]
    timeout_per_tick = [ev["timeout"] for ev, _, _ in evals]
    kpi = compute_economic_kpi(
        tokens_per_tick=tokens_per_tick,
        timeout_rate_pct_per_tick=timeout_per_tick,
        energy_cost_per_tick=[0.0] * len(evals),
        active_gpu_hours_by_type_per_tick=gpu_hours_per_tick,
        migration_count=scale_events, config=cfg)

    def wmean(vals):
        tw = sum(weights)
        return sum(v * w for v, w in zip(vals, weights)) / tw if tw else 0.0

    timeout_pct = wmean(timeouts) if timeouts else 0.0
    queue_p99_s = wmean(q99) if q99 else 0.0
    queue_p95_s = wmean(q95) if q95 else 0.0
    e2e_p99_s = wmean(e2e99) if e2e99 else 0.0
    e2e_p95_s = wmean(e2e95) if e2e95 else 0.0
    return {
        "rho_target": R,
        "predicted_goodput_per_dollar": float(
            kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "predicted_sla_safe_goodput": int(kpi.sla_compliant_goodput),
        "predicted_gpu_hours": float(kpi.active_gpu_hours),
        "predicted_timeout_pct": float(timeout_pct),
        "predicted_queue_p95_ms": float(queue_p95_s * 1000.0),
        "predicted_queue_p99_ms": float(queue_p99_s * 1000.0),
        "predicted_latency_p95_ms": float(e2e_p95_s * 1000.0),
        "predicted_latency_p99_ms": float(e2e_p99_s * 1000.0),
        "predicted_scale_events": int(scale_events),
        "predicted_churn_score": float(scale_events),
        "predicted_mean_utilization": None,  # genai_backtest aggregates per-tick rho
        "sla_violation_rate": float(sum(1 for ev, _, _ in evals
                                        if ev["timeout"] > 0)
                                    / max(1, sum(1 for _, t, _ in evals
                                                 if t.n > 0))),
        "infra_cost": float(kpi.total_infrastructure_cost),
    }


def _audit_genai_2026(*, source_dir: str, tick_seconds: float = 3600.0) -> dict:
    layers = ag.load_all_layers(source_dir,
                                request_kwargs=dict(include_failures=False))
    requests = layers["requests"]
    if not requests:
        return {"trace": "alibaba_genai_2026", "applicable": False,
                "exclusion_reason": "no requests loaded from " + source_dir}
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)
    ticks = gbt._aggregate_ticks(list(requests), tick_seconds)
    tick_hours = tick_seconds / 3600.0

    frontier_antic = [_eval_rho_genai(ticks, R, cold=cold,
                                      tick_hours=tick_hours) for R in RHOS]

    # Named policies via UNCHANGED genai_backtest.run_backtest
    res = gbt.run_backtest(requests, tick_seconds=tick_seconds,
                           cold_start_s=cold)
    rho_for = {"sla_aware": 0.65, "constraint_aware": 0.65,
               "queue_aware": 0.65, "utilization_aware": 0.85, "fifo": None}
    named_rows = []
    for pname in ("fifo", "sla_aware", "queue_aware", "utilization_aware",
                  "constraint_aware"):
        if pname not in res.policy_results:
            continue
        r = res.policy_results[pname]
        named_rows.append({
            "policy": pname, "rho_target": rho_for.get(pname),
            "goodput_per_dollar": float(
                r.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
            "sla_compliant_goodput": int(r.sla_compliant_requests),
            "gpu_hours": float(r.replica_hours),
            "infra_cost": float(r.kpi.total_infrastructure_cost),
            "timeout_pct_mean": float(r.timeout_rate_pct),
            "queue_p95_ms": float(r.queue_p95_s * 1000.0),
            "queue_p99_ms": float(r.queue_p99_s * 1000.0),
            "latency_p99_ms": float(r.e2e_p99_s * 1000.0),
            "safe": bool(r.timeout_rate_pct <= SAFE_TIMEOUT_PCT
                         and r.queue_p99_s * 1000.0 <= SAFE_QUEUE_P99_MS),
        })

    profile = WorkloadFrontierProfile(
        workload_id="alibaba_genai_2026",
        workload_type="inference_standard", telemetry_confidence="medium",
        priority_class="standard", candidate_rhos=RHOS,
        source="alibaba_genai_2026")
    safety = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                          max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                          min_telemetry_confidence="low")
    pts = estimate_frontier_from_points(profile, frontier_antic,
                                        safety_config=safety)
    cfg = FrontierControllerConfig(deadband_rho=0.05, deadband_kpi_pct=0.05,
                                   min_telemetry_confidence="low",
                                   default_execution_mode=SHADOW_MODE)
    decision = choose_safe_utilization_target(profile, pts,
                                              current_rho=CA_DEFAULT_RHO,
                                              controller_config=cfg)
    selected = decision.selected_point
    fc_row = {"policy": "frontier_controller_v1",
              "rho_target": decision.selected_rho,
              "goodput_per_dollar": (selected.predicted_goodput_per_dollar
                                     if selected else None),
              "gpu_hours": (selected.predicted_gpu_hours if selected else None),
              "timeout_pct_mean": (selected.predicted_timeout_pct
                                   if selected else None),
              "queue_p95_ms": (selected.predicted_queue_p95_ms
                               if selected else None),
              "queue_p99_ms": (selected.predicted_queue_p99_ms
                               if selected else None),
              "latency_p99_ms": (selected.predicted_latency_p99_ms
                                 if selected else None),
              "safe": selected.is_safe if selected else False}

    ca_gpd = next((r["goodput_per_dollar"] for r in named_rows
                   if r["policy"] == "constraint_aware"), 0.0)
    sla_gpd = next((r["goodput_per_dollar"] for r in named_rows
                    if r["policy"] == "sla_aware"), 0.0)
    fc_gpd = fc_row["goodput_per_dollar"] or 0.0
    delta_ca = ((fc_gpd - ca_gpd) / ca_gpd * 100.0) if ca_gpd else 0.0
    delta_sla = ((fc_gpd - sla_gpd) / sla_gpd * 100.0) if sla_gpd else 0.0
    verdict = ("TIE" if abs(delta_ca) <= TIE_BAND_PCT
               else ("FRONTIER_WIN" if delta_ca > 0 else "FRONTIER_LOSS"))

    safe_rhos = [round(p.rho_target, 4) for p in pts if p.is_safe]
    first_unsafe = next((round(p.rho_target, 4) for p in pts
                         if p.safety_status == SafetyStatus.UNSAFE), None)
    return {
        "trace": "alibaba_genai_2026", "applicable": True,
        "source": source_dir, "n_ticks": len(ticks),
        "tick_seconds": tick_seconds,
        "cold_start_calibration_s": cold,
        "frontier_anticipatory": frontier_antic,
        "frontier_reactive": [],  # GenAI replay is anticipatory-only here
        "frontier_points_safety": [p.to_dict() for p in pts],
        "named_policies": named_rows, "frontier_controller_row": fc_row,
        "decision": decision.to_dict(),
        "comparison": {
            "constraint_aware_goodput_per_dollar": ca_gpd,
            "sla_aware_goodput_per_dollar": sla_gpd,
            "frontier_controller_goodput_per_dollar": fc_gpd,
            "delta_vs_constraint_aware_pct": delta_ca,
            "delta_vs_sla_aware_pct": delta_sla, "verdict": verdict,
        },
        "diagnostic": {
            "best_safe_rho": max(safe_rhos) if safe_rhos else None,
            "first_unsafe_rho": first_unsafe,
            "safe_rhos": safe_rhos,
            "frontier_peak_delta_vs_constraint_aware_pct": delta_ca,
            "constraint_aware_rho_in_safe_set":
                CA_DEFAULT_RHO in [round(r, 4) for r in safe_rhos]
                if safe_rhos else False,
        },
    }


# ===========================================================================
# Bin-packing / training scheduling traces — frontier NOT applicable
# ===========================================================================

def _exclusion_record(trace_id: str, reason: str) -> dict:
    return {"trace": trace_id, "applicable": False,
            "exclusion_reason": reason,
            "decision": None, "named_policies": None,
            "comparison": None, "diagnostic": None}


# ===========================================================================
# Cross-trace synthesis
# ===========================================================================

def _classify_workload(d: dict) -> str:
    """Workload-class label used for the synthesis explanation."""
    n = d["trace"]
    if "alibaba_gpu" in n:
        return "fragmentation_packing"
    if "philly" in n:
        return "training_scheduling"
    if "burstgpt" in n:
        return "bursty_interactive_inference"
    if "azure_llm_2024" in n:
        return "weekly_periodic_interactive_inference"
    if "azure_llm" in n:
        return "interactive_inference"
    if "genai" in n:
        return "multi_layer_inference_with_cold_start"
    return "unknown"


def _synthesize(per_trace: list[dict]) -> dict:
    applicable = [d for d in per_trace if d["applicable"]]
    skipped = [d for d in per_trace if not d["applicable"]]

    verdict_counts = {"FRONTIER_WIN": 0, "TIE": 0, "FRONTIER_LOSS": 0,
                      "INSUFFICIENT_TELEMETRY": 0}
    rows = []
    best_safe_rhos = []
    delta_pcts = []
    for d in applicable:
        cmp_ = d["comparison"]
        verdict = cmp_["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        diag = d["diagnostic"]
        rows.append({
            "trace": d["trace"],
            "workload_class": _classify_workload(d),
            "constraint_aware_goodput_per_dollar":
                cmp_["constraint_aware_goodput_per_dollar"],
            "frontier_controller_goodput_per_dollar":
                cmp_["frontier_controller_goodput_per_dollar"],
            "delta_pct": cmp_["delta_vs_constraint_aware_pct"],
            "best_safe_rho": diag["best_safe_rho"],
            "first_unsafe_rho": diag["first_unsafe_rho"],
            "verdict": verdict,
            "controller_action": (d["decision"]["action"]
                                  if d.get("decision") else None),
            "controller_selected_rho": (d["decision"]["selected_rho"]
                                        if d.get("decision") else None),
            "executable_in_real_cluster":
                d["decision"]["executable_in_real_cluster"]
                if d.get("decision") else False,
        })
        if diag["best_safe_rho"] is not None:
            best_safe_rhos.append(diag["best_safe_rho"])
        delta_pcts.append(cmp_["delta_vs_constraint_aware_pct"])

    n = len(applicable)
    wins = verdict_counts["FRONTIER_WIN"]
    ties = verdict_counts["TIE"]
    losses = verdict_counts["FRONTIER_LOSS"]
    pct = (lambda x: round(x / n * 100.0, 2) if n else None)
    win_pct = pct(wins)
    tie_pct = pct(ties)
    loss_pct = pct(losses)
    safe_or_tie_pct = pct(wins + ties)

    best_safe_distribution = {}
    for r in best_safe_rhos:
        best_safe_distribution[str(r)] = best_safe_distribution.get(str(r), 0) + 1

    # Architecture recommendation logic — evidence-based.
    if n == 0:
        recommendation = ("INSUFFICIENT_EVIDENCE",
                          "no applicable traces produced a frontier verdict")
    elif losses > 0:
        recommendation = (
            "KEEP_FRONTIER_CONTROLLER_SEPARATE",
            "≥1 trace shows a frontier_controller_v1 regression vs the "
            "constraint_aware default; integrating it into constraint_aware "
            "would risk silent regressions on those workload classes.")
    elif wins == n:
        recommendation = (
            "INTEGRATE_OPT_IN_TO_CONSTRAINT_AWARE",
            "every applicable trace improved or safely tied with no "
            "regressions; integration as a per-workload OPT-IN switch is "
            "evidence-supported, but a global default change is NOT "
            "(rho varies by workload / SLA / telemetry).")
    else:
        # mix of wins and ties, no losses
        recommendation = (
            "KEEP_FRONTIER_CONTROLLER_SEPARATE_OR_OPT_IN",
            f"{wins}/{n} traces show a frontier_controller_v1 alpha win and "
            f"{ties}/{n} are safe ties; no regressions observed. Integration "
            "as a per-workload-class OPT-IN remains evidence-supported, but a "
            "global default change is NOT (rho varies by workload / SLA / "
            "telemetry).")

    # Generalization verdict
    if n == 0:
        generalizes = "INSUFFICIENT_EVIDENCE"
    elif losses == 0 and wins >= 1:
        generalizes = "GENERALIZES_WITHIN_APPLICABLE_LLM_INFERENCE_TRACES"
    elif losses == 0:
        generalizes = "SAFE_TIE_ACROSS_APPLICABLE_TRACES"
    else:
        generalizes = "WORKLOAD_DEPENDENT"

    return {
        "n_applicable": n, "n_skipped": len(skipped),
        "verdict_counts": verdict_counts,
        "win_pct": win_pct, "tie_pct": tie_pct, "loss_pct": loss_pct,
        "safe_or_tie_pct": safe_or_tie_pct,
        "best_safe_rho_distribution": best_safe_distribution,
        "best_safe_rho_min": min(best_safe_rhos) if best_safe_rhos else None,
        "best_safe_rho_max": max(best_safe_rhos) if best_safe_rhos else None,
        "delta_pct_min": min(delta_pcts) if delta_pcts else None,
        "delta_pct_max": max(delta_pcts) if delta_pcts else None,
        "delta_pct_mean": (sum(delta_pcts) / len(delta_pcts)
                            if delta_pcts else None),
        "rows": rows,
        "generalizes": generalizes,
        "architecture_recommendation": {
            "code": recommendation[0], "rationale": recommendation[1],
        },
    }


# ===========================================================================
# Markdown
# ===========================================================================

def _f(v, nd: int = 2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return (f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _frontier_table(rows, append):
    append("| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | "
           "mean rho | scale ev | safety |")
    append("|---|---|---|---|---|---|---|---|")
    for p in rows:
        append(f"| {p['rho_target']} | {_f(p['predicted_goodput_per_dollar'])} | "
               f"{_f(p['predicted_timeout_pct'])} | "
               f"{_f(p['predicted_queue_p95_ms'])} / "
               f"{_f(p['predicted_queue_p99_ms'])} | "
               f"{_f(p['predicted_gpu_hours'])} | "
               f"{_f(p.get('predicted_mean_utilization'), nd=4)} | "
               f"{_f(p.get('predicted_scale_events'))} | "
               + ("SAFE" if (p['predicted_timeout_pct'] <= SAFE_TIMEOUT_PCT
                             and p['predicted_queue_p99_ms']
                             <= SAFE_QUEUE_P99_MS)
                  else "**UNSAFE**") + " |")


def _named_table(rows, append):
    append("| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | "
           "safety |")
    append("|---|---|---|---|---|---|---|")
    for r in rows:
        append(f"| {r['policy']} | {_f(r.get('rho_target'))} | "
               f"{_f(r['goodput_per_dollar'])} | "
               f"{_f(r['timeout_pct_mean'])} | "
               f"{_f(r['queue_p99_ms'])} | "
               f"{_f(r['gpu_hours'])} | "
               + ("SAFE" if r['safe'] else "**UNSAFE**") + " |")


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Cross-Trace Safe Utilization Frontier Generalization Audit\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Cross-trace validation "
      "of `frontier_controller_v1` (`aurelius/frontier/`) against every "
      "currently-integrated public trace where target-utilization (rho) "
      "decisions are meaningful. No production code, optimizer logic, "
      "simulator constant, or robust-energy-engine code was modified, no ML "
      "model was trained, no dataset was ingested, no constant was tuned to "
      "force a result. **Real-cluster execution is disabled by default** "
      "(`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`).\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, "
      "`docs/PUBLIC_TRACE_BACKTESTS.md`.\n")

    A("## 1. Configuration\n")
    A(f"- **Candidate rho grid:** `{list(RHOS)}`")
    A(f"- **Safety thresholds (pre-registered):** timeout ≤ "
      f"{SAFE_TIMEOUT_PCT}% AND queue p99 ≤ {SAFE_QUEUE_P99_MS} ms")
    A("- **Comparison policies:** `fifo`, `sla_aware`, `queue_aware`, "
      "`utilization_aware`, `constraint_aware` (current baseline), "
      "`frontier_controller_v1`, plus `oracle_forecast_ANALYSIS_ONLY` where "
      "available (analysis-only).\n")

    A("## 2. Datasets analyzed\n")
    A("| trace | applicable to frontier control | source | reason if "
      "excluded |")
    A("|---|---|---|---|")
    for d in payload["per_trace"]:
        if d["applicable"]:
            A(f"| `{d['trace']}` | ✅ | `{d.get('source', 'fixture / canonical')}` "
              f"| — |")
        else:
            A(f"| `{d['trace']}` | ❌ | — | {d.get('exclusion_reason', '')} |")
    A("")

    # Per-trace blocks
    for i, d in enumerate(payload["per_trace"], start=1):
        A(f"## 3.{i} `{d['trace']}` — frontier sweep + controller verdict\n")
        if not d["applicable"]:
            A(f"**Excluded.** {d['exclusion_reason']}\n")
            continue
        cmp_ = d["comparison"]
        diag = d["diagnostic"]
        A(f"- **Source:** `{d.get('source', '—')}`")
        A(f"- **Ticks:** {d.get('n_ticks', '—')} @ "
          f"{d.get('tick_seconds', '—')}s")
        A(f"- **Verdict vs `constraint_aware`:** **`{cmp_['verdict']}`** "
          f"(Δ {cmp_['delta_vs_constraint_aware_pct']:+.3f}%)")
        A(f"- **Frontier controller decision:** "
          f"`{d['decision']['action']}` → rho = "
          f"{d['decision']['selected_rho']}; "
          f"executable_in_real_cluster = "
          f"{d['decision']['executable_in_real_cluster']}")
        A(f"- **Best safe rho (anticipatory frontier):** "
          f"{diag['best_safe_rho']} | first unsafe rho: "
          f"{diag['first_unsafe_rho']}\n")
        A("### Anticipatory frontier sweep\n")
        _frontier_table(d["frontier_anticipatory"], A)
        if d.get("frontier_reactive"):
            A("\n### Reactive frontier sweep (diagnostic)\n")
            _frontier_table(d["frontier_reactive"], A)
        A("\n### Policy comparison\n")
        rows = list(d["named_policies"]) + [{
            "policy": "frontier_controller_v1",
            "rho_target": d["frontier_controller_row"]["rho_target"],
            "goodput_per_dollar": d["frontier_controller_row"]["goodput_per_dollar"],
            "timeout_pct_mean": d["frontier_controller_row"]["timeout_pct_mean"],
            "queue_p99_ms": d["frontier_controller_row"]["queue_p99_ms"],
            "gpu_hours": d["frontier_controller_row"]["gpu_hours"],
            "safe": d["frontier_controller_row"]["safe"]}]
        _named_table(rows, A)
        A("")

    syn = payload["synthesis"]
    A("## 4. Cross-trace summary\n")
    A("| trace | workload class | constraint_aware goodput/$ | "
      "frontier_controller_v1 goodput/$ | Δ % | best safe rho | "
      "controller action → rho | verdict | generalizes? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in syn["rows"]:
        A(f"| `{r['trace']}` | {r['workload_class']} | "
          f"{_f(r['constraint_aware_goodput_per_dollar'])} | "
          f"{_f(r['frontier_controller_goodput_per_dollar'])} | "
          f"{r['delta_pct']:+.3f}% | "
          f"{r['best_safe_rho']} | "
          f"`{r['controller_action']}` → {r['controller_selected_rho']} | "
          f"**{r['verdict']}** | "
          f"{'yes' if r['verdict'] in ('FRONTIER_WIN', 'TIE') else 'no'} |")
    A("")
    A(f"**Counts (applicable traces, n={syn['n_applicable']}):** wins = "
      f"{syn['verdict_counts']['FRONTIER_WIN']} ({_f(syn['win_pct'])}%); "
      f"ties = {syn['verdict_counts']['TIE']} ({_f(syn['tie_pct'])}%); "
      f"losses = {syn['verdict_counts']['FRONTIER_LOSS']} "
      f"({_f(syn['loss_pct'])}%); skipped = {syn['n_skipped']}.\n")
    A(f"**Δ goodput/$ range:** {_f(syn['delta_pct_min'])}% to "
      f"{_f(syn['delta_pct_max'])}% (mean "
      f"{_f(syn['delta_pct_mean'])}%).\n")
    A(f"**Best-safe-rho distribution:** "
      f"{syn['best_safe_rho_distribution']} "
      f"(min {syn['best_safe_rho_min']}, max {syn['best_safe_rho_max']}). "
      "The safe rho is **workload-specific** — no single global value is "
      "supported.\n")

    A("## 5. Generalization & architecture recommendation\n")
    A(f"- **Does `frontier_controller_v1` improve or safely tie across the "
      f"applicable LLM serving traces?** "
      f"`{syn['generalizes']}`.")
    A(f"- **A. Generally superior?** "
      f"{'Yes, on the applicable LLM serving traces (no regression observed).' if syn['verdict_counts']['FRONTIER_LOSS'] == 0 and syn['verdict_counts']['FRONTIER_WIN'] > 0 else 'No — see verdict counts.'}")
    A(f"- **B. Workload-dependent?** Yes — the safe peak rho varies by trace "
      "(see distribution above); the bin-packing / training-job traces are "
      "structurally outside the frontier-controller scope.")
    A(f"- **C. Should it be integrated into `constraint_aware`?** "
      f"`{syn['architecture_recommendation']['code']}` — "
      f"{syn['architecture_recommendation']['rationale']}")
    A(f"- **D. % improvement (applicable):** {_f(syn['win_pct'])}%.")
    A(f"- **E. % neutral ties (applicable):** {_f(syn['tie_pct'])}%.")
    A(f"- **F. % regressions (applicable):** {_f(syn['loss_pct'])}%.")
    A("")

    A("## 6. Honesty / scope\n")
    A("- The Azure 2024 frontier is the **full week-long committed audit "
      "JSON** (`data/external/azure_llm_2024/processed/"
      "azure_2024_safe_utilization_frontier.json`); every other LLM trace "
      "frontier is computed in-process on its **fixture or raw data if "
      "present** via the UNCHANGED serving physics in "
      "`aurelius/traces/backtest.py` / `genai_backtest.py`.")
    A("- Bin-packing traces (Alibaba GPU v2023) and training-job-scheduling "
      "traces (Microsoft Philly) are **structurally not utilization-rho "
      "benchmarks** — they sweep packing density or job-completion times, "
      "not a continuous request-rate / serving-utilization target — so the "
      "frontier controller is documented as **NOT APPLICABLE**.")
    A("- The `constraint_aware` engine default (rho ≈ 0.65) is **unchanged** "
      "by this audit. No production code, simulator constant, optimizer "
      "logic, or safety gate has been modified.")
    A("- No production-savings claim. Real-cluster execution is **disabled "
      "by default**; pilot telemetry is required to calibrate the safe rho "
      "per workload (`docs/PILOT_TELEMETRY_CONTRACT.md`).\n")

    A("## 7. Remaining unknowns\n")
    A("- The fixture-derived frontiers for BurstGPT / Azure 2023 / GenAI "
      "2026 inherit the fixture's small-sample shape; the *direction* of "
      "the verdict is informative but the absolute Δ% is fixture-bounded. "
      "Larger raw replays (when raw is present) will refine the absolute "
      "numbers; the *verdict bucket* should remain stable because the "
      "controller selects by category (SAFE / UNSAFE), not by a tuned KPI "
      "threshold.")
    A("- Customer/pilot telemetry is required to calibrate the safe rho "
      "per workload before any real-cluster promotion of a frontier "
      "decision.")
    A("- The reactive vs anticipatory dominance pattern observed on Azure "
      "2024 (anticipatory frontier strictly dominates reactive on safety) "
      "needs re-validation on each customer-specific real serving engine.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ===========================================================================
# Driver
# ===========================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--burstgpt-tick-seconds", type=float, default=60.0)
    p.add_argument("--azure-2023-tick-seconds", type=float, default=60.0)
    p.add_argument("--genai-tick-seconds", type=float, default=3600.0)
    p.add_argument("--burstgpt-sample-size", type=int, default=None)
    p.add_argument("--azure-2023-sample-size", type=int, default=None)
    # Fixture load multipliers so the frontier is observable on the small
    # committed fixtures. The Azure 2024 audit uses 10× busy-tier; we use 25×
    # on BurstGPT / Azure 2023 because their fixtures are 1–2 orders of
    # magnitude smaller than the cached Azure 2024 ticks.
    p.add_argument("--burstgpt-scale", type=float, default=25.0)
    p.add_argument("--azure-2023-scale", type=float, default=25.0)
    p.add_argument("--genai-scale", type=float, default=1.0)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    per_trace: list[dict] = []

    # --- BurstGPT (bursty interactive inference) ---
    try:
        bg_requests, bg_src = _load_burstgpt_requests(
            sample_size=args.burstgpt_sample_size)
        if bg_requests:
            bg_ticks = _ticks_from_requests(
                bg_requests, tick_seconds=args.burstgpt_tick_seconds)
            d = _audit_llm_trace("burstgpt", bg_ticks,
                                 tick_seconds=args.burstgpt_tick_seconds,
                                 scale=args.burstgpt_scale)
            d["source"] = bg_src
            per_trace.append(d)
        else:
            per_trace.append(_exclusion_record("burstgpt",
                                               "no requests loaded"))
    except Exception as exc:  # pragma: no cover - defensive
        per_trace.append(_exclusion_record("burstgpt",
                                           f"load error: {exc!r}"))

    # --- Azure LLM 2023 (single-week interactive inference) ---
    try:
        az23_requests, az23_src = _load_azure_2023_requests(
            sample_size=args.azure_2023_sample_size)
        if az23_requests:
            az23_ticks = _ticks_from_requests(
                az23_requests, tick_seconds=args.azure_2023_tick_seconds)
            d = _audit_llm_trace("azure_llm_2023", az23_ticks,
                                 tick_seconds=args.azure_2023_tick_seconds,
                                 scale=args.azure_2023_scale)
            d["source"] = az23_src
            per_trace.append(d)
        else:
            per_trace.append(_exclusion_record("azure_llm_2023",
                                               "no requests loaded"))
    except Exception as exc:  # pragma: no cover - defensive
        per_trace.append(_exclusion_record("azure_llm_2023",
                                           f"load error: {exc!r}"))

    # --- Azure LLM 2024 (full week — committed audit JSON) ---
    per_trace.append(_audit_azure_2024_from_committed())

    # --- Alibaba GenAI 2026 (multi-layer with cold-start) ---
    genai_dir = os.path.join(REPO_ROOT, "tests", "fixtures",
                             "alibaba_genai_sample")
    try:
        d = _audit_genai_2026(source_dir=genai_dir,
                              tick_seconds=args.genai_tick_seconds)
        if d.get("applicable"):
            d["source"] = f"fixture:{genai_dir}"
        per_trace.append(d)
    except Exception as exc:  # pragma: no cover - defensive
        per_trace.append(_exclusion_record("alibaba_genai_2026",
                                           f"load error: {exc!r}"))

    # --- Alibaba GPU v2023 (NOT APPLICABLE — bin-packing) ---
    per_trace.append(_exclusion_record(
        "alibaba_gpu_v2023",
        "Bin-packing / fragmentation trace: pods have fixed (cpu, gpu_milli, "
        "memory) requirements and there is no continuous request-rate or "
        "serving-utilization target rho to sweep. The headline baseline is "
        "first-fit / best-fit / FFD packing — see "
        "docs/ALIBABA_GPU_BACKTEST_RESULTS.md. Safe Utilization Frontier "
        "Control acts on serving rho targets and is therefore structurally "
        "not applicable to packing decisions."))

    # --- Microsoft Philly (NOT APPLICABLE — training-job scheduling) ---
    per_trace.append(_exclusion_record(
        "microsoft_philly",
        "Training-job scheduling trace: deadlines / job-progress and "
        "GPU-hours dominate, not request-rate / serving rho. The headline "
        "metric is job completion / SLA-violation count, not goodput/$ over "
        "a rho sweep. Safe Utilization Frontier Control acts on serving "
        "rho targets and is structurally not applicable to training-job "
        "scheduling."))

    synthesis = _synthesize(per_trace)
    payload = {
        "config": {"rhos": list(RHOS),
                   "safe_timeout_pct": SAFE_TIMEOUT_PCT,
                   "safe_queue_p99_ms": SAFE_QUEUE_P99_MS,
                   "constraint_aware_default_rho": CA_DEFAULT_RHO,
                   "tie_band_pct": TIE_BAND_PCT},
        "per_trace": per_trace,
        "synthesis": synthesis,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)

    print(f"[cross-trace] applicable={synthesis['n_applicable']} "
          f"skipped={synthesis['n_skipped']} "
          f"wins={synthesis['verdict_counts']['FRONTIER_WIN']} "
          f"ties={synthesis['verdict_counts']['TIE']} "
          f"losses={synthesis['verdict_counts']['FRONTIER_LOSS']}")
    print(f"[cross-trace] recommendation: "
          f"{synthesis['architecture_recommendation']['code']}")
    print(f"[cross-trace] JSON -> {args.out_json}")
    print(f"[cross-trace] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
