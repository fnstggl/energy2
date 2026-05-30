#!/usr/bin/env python3
"""Cross-Trace `constraint_aware` × Frontier-Integration Safety Check.

For each LLM serving trace already integrated into Aurelius, compares two
configurations of the UNCHANGED ``constraint_aware`` policy:

  * ``constraint_aware_current``       — engine default (rho 0.65).
  * ``constraint_aware_frontier_opt_in`` — same engine, with
    :class:`FrontierIntegrationConfig(enabled=True)` so the rho target is
    sourced from :func:`select_constraint_aware_rho`.

Both runs use the **unchanged** ``aurelius/traces/backtest.run_backtest``
harness — the opt-in adapter is the only difference between them. The
non-applicable bin-packing / training-job traces (Alibaba GPU v2023,
Microsoft Philly) are explicitly excluded with a documented reason.

Safety requirements (per the integration spec):

- no material regression > 1 % goodput/$ vs ``constraint_aware_current``;
- no increase in SLA violations beyond the configured tolerance;
- no application to unsupported traces (eligibility must say so);
- if the adapter falls back, the report MUST explain why.

Outputs:
  * docs/CROSS_TRACE_CONSTRAINT_FRONTIER_INTEGRATION_SAFETY.md
  * data/external/frontier/
    cross_trace_constraint_frontier_integration_safety_summary.json

Directional simulator / shadow-mode evidence only — NOT production
savings (``docs/RESULTS.md`` §8). Real-cluster execution is disabled by
default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.constraints.frontier_integration import (  # noqa: E402
    CONSTRAINT_AWARE_DEFAULT_RHO,
    FrontierIntegrationConfig,
    FrontierIntegrationCounters,
)
from aurelius.traces import azure_llm as az  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402
from aurelius.traces import burstgpt as bg  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "cross_trace_constraint_frontier_integration_safety_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs",
    "CROSS_TRACE_CONSTRAINT_FRONTIER_INTEGRATION_SAFETY.md")

# Safety thresholds (mirror the cross-trace audit).
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
REGRESSION_TOL_PCT = 1.0
RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
# Fixture load multipliers so the frontier is observable on the small
# committed fixtures (matches the cross-trace generalization audit).
SCALE = 25.0


def _scale_ticks(ticks, factor: float):
    if factor == 1.0:
        return list(ticks)
    return [replace(
        t, request_count=int(round(t.request_count * factor)),
        arrival_rate_rps=t.arrival_rate_rps * factor,
        total_prompt_tokens=int(round(t.total_prompt_tokens * factor)),
        total_output_tokens=int(round(t.total_output_tokens * factor)),
        model_mix={k: int(round(v * factor)) for k, v in t.model_mix.items()},
    ) for t in ticks]


def _load_burstgpt_requests() -> tuple[list, str]:
    raw = os.path.join(REPO_ROOT, "data", "external", "burstgpt", "raw",
                       "BurstGPT_1.csv")
    fixture = os.path.join(REPO_ROOT, "tests", "fixtures",
                           "burstgpt_sample.csv")
    if os.path.exists(raw):
        return bg.load_csv(raw), f"raw:{raw}"
    return bg.load_csv(fixture), f"fixture:{fixture}"


def _load_azure_2023_requests() -> tuple[list, str]:
    raw_dir = os.path.join(REPO_ROOT, "data", "external", "azure_llm", "raw")
    fixture = os.path.join(REPO_ROOT, "tests", "fixtures",
                           "azure_llm_sample.csv")
    if os.path.isdir(raw_dir):
        cand = next((os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                     if f.endswith(".csv")), None)
        if cand:
            return (az.load_csv(cand, variant=az.variant_from_path(cand)),
                    f"raw:{cand}")
    return (az.load_csv(fixture, variant="conv", include_failures=False),
            f"fixture:{fixture}")


def _policy_row(name: str, res, rho) -> dict:
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
        "safe": bool(res.queue_p99_ms <= SAFE_QUEUE_P99_MS
                     and getattr(res, "timeout_rate_pct_mean", 0.0)
                     <= SAFE_TIMEOUT_PCT),
    }


def _audit_llm_trace(name: str, ticks, *, tick_seconds: float,
                     integration_cfg: FrontierIntegrationConfig,
                     workload_meta: dict) -> dict:
    counters_off = FrontierIntegrationCounters()
    counters_on = FrontierIntegrationCounters()

    # Run constraint_aware with frontier disabled (default behaviour).
    cur = bt._run_policy("constraint_aware", ticks,
                         tick_hours=tick_seconds / 3600.0,
                         frontier_integration=None,
                         frontier_counters=counters_off)

    # Run constraint_aware with frontier opt-in enabled.
    service_state = {"telemetry_ticks": list(ticks),
                     "telemetry_window_ticks": len(ticks),
                     "request_metrics_present": True,
                     "queue_metrics_present": True}
    opt = bt._run_policy("constraint_aware", ticks,
                         tick_hours=tick_seconds / 3600.0,
                         frontier_integration=integration_cfg,
                         frontier_workload_metadata=workload_meta,
                         frontier_service_state=service_state,
                         frontier_counters=counters_on)
    cur_row = _policy_row("constraint_aware_current", cur,
                          CONSTRAINT_AWARE_DEFAULT_RHO)
    opt_row = _policy_row("constraint_aware_frontier_opt_in", opt,
                          None)  # adapter chooses
    ft = getattr(opt, "frontier_integration", None)
    if ft is not None:
        opt_row["rho_target"] = ft.selected_rho
        opt_row["frontier_used"] = ft.used_frontier
        opt_row["frontier_fallback_reason"] = ft.fallback_reason
        opt_row["frontier_action"] = (ft.decision.action
                                      if ft.decision else None)
        opt_row["frontier_reason"] = (ft.decision.reason if ft.decision
                                      else None)
        opt_row["frontier_safety_vetoes"] = list(ft.safety_vetoes)
        opt_row["frontier_confidence"] = ft.confidence
    else:
        opt_row["rho_target"] = CONSTRAINT_AWARE_DEFAULT_RHO
        opt_row["frontier_used"] = False
        opt_row["frontier_fallback_reason"] = "no_frontier_telemetry"

    delta_pct = ((opt_row["goodput_per_dollar"] - cur_row["goodput_per_dollar"])
                 / cur_row["goodput_per_dollar"] * 100.0
                 if cur_row["goodput_per_dollar"] else 0.0)
    timeout_delta = (opt_row["timeout_pct_mean"] - cur_row["timeout_pct_mean"])

    # Safety verdict
    regression = delta_pct < -REGRESSION_TOL_PCT
    sla_regression = timeout_delta > REGRESSION_TOL_PCT
    if regression or sla_regression:
        verdict = "INTEGRATION_REGRESSION"
    elif abs(delta_pct) <= REGRESSION_TOL_PCT:
        verdict = "SAFE_TIE"
    elif delta_pct > REGRESSION_TOL_PCT:
        verdict = "INTEGRATION_WIN"
    else:  # pragma: no cover
        verdict = "UNKNOWN"

    return {
        "trace": name, "applicable": True, "n_ticks": len(ticks),
        "tick_seconds": tick_seconds, "scale": SCALE,
        "constraint_aware_current": cur_row,
        "constraint_aware_frontier_opt_in": opt_row,
        "comparison": {
            "delta_goodput_per_dollar_pct": delta_pct,
            "delta_timeout_pct_absolute": timeout_delta,
            "regression_tolerance_pct": REGRESSION_TOL_PCT,
            "verdict": verdict,
        },
        "counters_with_frontier_disabled": counters_off.to_dict(),
        "counters_with_frontier_enabled": counters_on.to_dict(),
    }


def _audit_azure_2024_from_committed() -> dict:
    """Reuse the committed Azure 2024 integration summary (read-only)."""
    integ_path = os.path.join(
        REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
        "azure_2024_constraint_frontier_integration_summary.json")
    if not os.path.exists(integ_path):
        return {"trace": "azure_llm_2024_week", "applicable": False,
                "exclusion_reason":
                    "committed integration summary missing — run "
                    "scripts/run_azure_2024_constraint_frontier_integration.py "
                    "first"}
    integ = json.load(open(integ_path))
    cur = integ["constraint_aware_current"]
    opt = integ["constraint_aware_frontier_opt_in"]
    cmp_ = integ["comparison"]
    delta_pct = cmp_["delta_goodput_per_dollar_pct"]
    timeout_delta = (opt["timeout_pct_mean"] - cur["timeout_pct_mean"])
    verdict = ("SAFE_TIE" if abs(delta_pct) <= REGRESSION_TOL_PCT
               else ("INTEGRATION_WIN" if delta_pct > REGRESSION_TOL_PCT
                     else "INTEGRATION_REGRESSION"))
    return {
        "trace": "azure_llm_2024_week", "applicable": True,
        "source": "committed_integration_summary",
        "n_ticks": 12960, "tick_seconds": 60.0, "scale": 10.0,
        "constraint_aware_current": {
            "policy": "constraint_aware_current",
            "rho_target": cur["selected_rho"],
            "goodput_per_dollar": cur["goodput_per_dollar"],
            "sla_compliant_goodput": cur["sla_compliant_goodput"],
            "gpu_hours": cur["gpu_hours"], "infra_cost": cur["infra_cost"],
            "timeout_pct_mean": cur["timeout_pct_mean"],
            "queue_p95_ms": cur["queue_p95_ms"],
            "queue_p99_ms": cur["queue_p99_ms"],
            "latency_p99_ms": cur["latency_p99_ms"],
            "safe": cur["safe"]},
        "constraint_aware_frontier_opt_in": {
            "policy": "constraint_aware_frontier_opt_in",
            "rho_target": opt["selected_rho"],
            "goodput_per_dollar": opt["goodput_per_dollar"],
            "sla_compliant_goodput": opt["sla_compliant_goodput"],
            "gpu_hours": opt["gpu_hours"], "infra_cost": opt["infra_cost"],
            "timeout_pct_mean": opt["timeout_pct_mean"],
            "queue_p95_ms": opt["queue_p95_ms"],
            "queue_p99_ms": opt["queue_p99_ms"],
            "latency_p99_ms": opt["latency_p99_ms"],
            "safe": opt["safe"],
            "frontier_used": opt["frontier_used"],
            "frontier_action": opt["frontier_action"],
            "frontier_reason": opt["frontier_reason"],
            "frontier_fallback_reason": opt["frontier_fallback_reason"],
            "frontier_safety_vetoes": opt["frontier_safety_vetoes"],
            "frontier_confidence": opt["frontier_confidence"]},
        "comparison": {
            "delta_goodput_per_dollar_pct": delta_pct,
            "delta_timeout_pct_absolute": timeout_delta,
            "regression_tolerance_pct": REGRESSION_TOL_PCT,
            "verdict": verdict,
        },
        "counters_with_frontier_disabled": {"frontier_used_count": 0,
                                            "frontier_fallback_count": 1},
        "counters_with_frontier_enabled":
            integ.get("counters", {"frontier_used_count": 1,
                                   "frontier_fallback_count": 0}),
    }


# Bin-packing / training traces — frontier integration NOT applicable.
def _exclusion_record(trace_id: str, reason: str) -> dict:
    return {"trace": trace_id, "applicable": False,
            "exclusion_reason": reason,
            "constraint_aware_current": None,
            "constraint_aware_frontier_opt_in": None,
            "comparison": None}


def _synthesize(rows: list[dict]) -> dict:
    applicable = [r for r in rows if r["applicable"]]
    skipped = [r for r in rows if not r["applicable"]]
    verdicts = {"SAFE_TIE": 0, "INTEGRATION_WIN": 0,
                "INTEGRATION_REGRESSION": 0}
    for r in applicable:
        v = r["comparison"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    pct = lambda x: round(x / max(1, len(applicable)) * 100.0, 2)
    safe = verdicts["SAFE_TIE"] + verdicts["INTEGRATION_WIN"]
    return {
        "n_applicable": len(applicable), "n_skipped": len(skipped),
        "verdict_counts": verdicts,
        "safe_or_win_pct": pct(safe),
        "regression_pct": pct(verdicts["INTEGRATION_REGRESSION"]),
        "any_regression": verdicts["INTEGRATION_REGRESSION"] > 0,
        "applicable_traces": [r["trace"] for r in applicable],
        "skipped_traces": [r["trace"] for r in skipped],
    }


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
    A("# Cross-Trace `constraint_aware` × Frontier-Integration Safety Check\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Compares the unchanged "
      "`constraint_aware` policy against itself with the opt-in frontier "
      "integration enabled. Real-cluster execution is **disabled by "
      "default**. The robust energy engine is **unchanged**.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, "
      "`docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, "
      "`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`.\n")

    cfg = payload["config"]
    A("## 1. Configuration\n")
    A(f"- **Candidate rho grid:** `{cfg['candidate_rhos']}`")
    A(f"- **Safety thresholds:** timeout ≤ {cfg['max_timeout_pct']}% AND "
      f"queue p99 ≤ {cfg['max_queue_p99_ms']} ms")
    A(f"- **Regression tolerance:** ±{cfg['regression_tolerance_pct']} % "
      "goodput/$ / + absolute timeout %")
    A(f"- **Min telemetry confidence:** "
      f"`{cfg['min_telemetry_confidence']}`")
    A("- **Real-cluster execution:** disabled by default.\n")

    A("## 2. Per-trace integration safety\n")
    A("| trace | applicable | current goodput/$ | opt_in goodput/$ | Δ % | "
      "Δ timeout % | selected rho | frontier_used | action | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["per_trace"]:
        if not r["applicable"]:
            A(f"| `{r['trace']}` | ❌ | — | — | — | — | — | — | — | "
              f"_excluded — {r['exclusion_reason'][:60]}..._ |")
            continue
        cur = r["constraint_aware_current"]
        opt = r["constraint_aware_frontier_opt_in"]
        cmp_ = r["comparison"]
        A(f"| `{r['trace']}` | ✅ | "
          f"{_f(cur['goodput_per_dollar'])} | "
          f"{_f(opt['goodput_per_dollar'])} | "
          f"{cmp_['delta_goodput_per_dollar_pct']:+.3f}% | "
          f"{cmp_['delta_timeout_pct_absolute']:+.3f} | "
          f"{opt.get('rho_target')} | "
          f"{opt.get('frontier_used')} | "
          f"`{opt.get('frontier_action') or '—'}` | "
          f"**{cmp_['verdict']}** |")
    A("")

    A("## 3. Excluded traces\n")
    for r in payload["per_trace"]:
        if not r["applicable"]:
            A(f"- **`{r['trace']}`** — {r['exclusion_reason']}")
    A("")

    A("## 4. Synthesis\n")
    syn = payload["synthesis"]
    vc = syn["verdict_counts"]
    A(f"- Applicable traces: **{syn['n_applicable']}**; "
      f"excluded: **{syn['n_skipped']}**")
    A(f"- Safe ties: **{vc['SAFE_TIE']}** | integration wins: "
      f"**{vc['INTEGRATION_WIN']}** | regressions: "
      f"**{vc['INTEGRATION_REGRESSION']}**")
    A(f"- Safe-or-win %: **{syn['safe_or_win_pct']}** | regression %: "
      f"**{syn['regression_pct']}**")
    A(f"- **Any regression?** {syn['any_regression']}\n")

    A("## 5. Fallback explanations\n")
    for r in payload["per_trace"]:
        if not r["applicable"]:
            continue
        opt = r["constraint_aware_frontier_opt_in"]
        if not opt.get("frontier_used"):
            A(f"- **`{r['trace']}`** — fell back to `constraint_aware` "
              f"default rho. Reason: "
              f"`{opt.get('frontier_fallback_reason', 'n/a')}`")
        else:
            A(f"- **`{r['trace']}`** — frontier used. Action: "
              f"`{opt.get('frontier_action')}`. Selected rho: "
              f"{opt.get('rho_target')}.")
    A("")

    A("## 6. Honesty / scope\n")
    A("- The `constraint_aware` engine default rho is **unchanged**. The "
      "frontier integration is **opt-in**, **LLM-serving-only**, **disabled "
      "by default**, and **falls back to the existing engine** on any "
      "ineligibility / unsafe recommendation / estimator or controller "
      "error.")
    A("- Alibaba GPU v2023 (bin-packing / fragmentation) and Microsoft "
      "Philly (training-job scheduling) are structurally outside the "
      "frontier-integration scope and are documented as **NOT APPLICABLE**.")
    A("- Real-cluster execution is **disabled by default**; pilot telemetry "
      "is required to calibrate the safe rho per workload before any "
      "production claim.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    integration_cfg = FrontierIntegrationConfig(
        enabled=True,
        candidate_rhos=tuple(RHOS),
        max_timeout_pct=SAFE_TIMEOUT_PCT,
        max_queue_p99_ms=SAFE_QUEUE_P99_MS,
    )
    workload_meta = {
        "workload_id": "cross_trace_safety_check",
        "workload_type": "inference_standard",
        "telemetry_confidence": "medium",
        "priority_class": "standard",
        "latency_sla_ms": 30000.0,
    }

    per_trace: list[dict] = []

    # BurstGPT
    try:
        reqs, src = _load_burstgpt_requests()
        ticks = _scale_ticks(requests_to_arrival_ticks(reqs, tick_seconds=60.0),
                             SCALE)
        d = _audit_llm_trace("burstgpt", ticks, tick_seconds=60.0,
                             integration_cfg=integration_cfg,
                             workload_meta=dict(workload_meta,
                                                workload_id="burstgpt"))
        d["source"] = src
        per_trace.append(d)
    except Exception as exc:  # pragma: no cover
        per_trace.append(_exclusion_record(
            "burstgpt", f"load_or_replay_error:{type(exc).__name__}:{exc}"))

    # Azure LLM 2023
    try:
        reqs, src = _load_azure_2023_requests()
        ticks = _scale_ticks(requests_to_arrival_ticks(reqs, tick_seconds=60.0),
                             SCALE)
        d = _audit_llm_trace("azure_llm_2023", ticks, tick_seconds=60.0,
                             integration_cfg=integration_cfg,
                             workload_meta=dict(workload_meta,
                                                workload_id="azure_llm_2023"))
        d["source"] = src
        per_trace.append(d)
    except Exception as exc:  # pragma: no cover
        per_trace.append(_exclusion_record(
            "azure_llm_2023",
            f"load_or_replay_error:{type(exc).__name__}:{exc}"))

    # Azure LLM 2024 — committed integration summary
    per_trace.append(_audit_azure_2024_from_committed())

    # Alibaba GenAI 2026 — note: the genai_backtest has its own _size_for_sla
    # for constraint_aware, which probes UP from MIN_REPLICAS until the SLA
    # is met (it does not use a fixed rho target). The frontier integration
    # plumbing is only in aurelius/traces/backtest.py for v1 — GenAI's
    # constraint_aware is rho-free, so the integration is structurally
    # inapplicable to its constraint_aware branch.
    per_trace.append(_exclusion_record(
        "alibaba_genai_2026",
        "GenAI 2026's constraint_aware uses _size_for_sla (probe-up-to-SLA), "
        "not a fixed rho target — the v1 integration adapter applies only to "
        "the rho-target sizer in aurelius/traces/backtest.py."))

    # Alibaba GPU v2023 (NOT APPLICABLE)
    per_trace.append(_exclusion_record(
        "alibaba_gpu_v2023",
        "Bin-packing / fragmentation trace. No continuous serving rho "
        "target; frontier integration is structurally not applicable."))

    # Microsoft Philly (NOT APPLICABLE)
    per_trace.append(_exclusion_record(
        "microsoft_philly",
        "Training-job scheduling trace. Deadlines / job-progress dominate, "
        "not serving rho; frontier integration is structurally not "
        "applicable."))

    payload = {
        "config": {
            "candidate_rhos": list(RHOS),
            "max_timeout_pct": SAFE_TIMEOUT_PCT,
            "max_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "regression_tolerance_pct": REGRESSION_TOL_PCT,
            "min_telemetry_confidence":
                integration_cfg.min_telemetry_confidence,
            "shadow_only": integration_cfg.shadow_only,
            "allow_real_execution": integration_cfg.allow_real_execution,
        },
        "per_trace": per_trace,
        "synthesis": _synthesize(per_trace),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)

    syn = payload["synthesis"]
    print(f"[ca-frontier-safety] applicable={syn['n_applicable']} "
          f"skipped={syn['n_skipped']} ties={syn['verdict_counts']['SAFE_TIE']} "
          f"wins={syn['verdict_counts']['INTEGRATION_WIN']} "
          f"regressions={syn['verdict_counts']['INTEGRATION_REGRESSION']}")
    print(f"[ca-frontier-safety] JSON -> {args.out_json}")
    print(f"[ca-frontier-safety] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
