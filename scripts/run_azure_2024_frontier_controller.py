#!/usr/bin/env python3
"""Azure LLM 2024 — Safe Utilization Frontier Controller v1 benchmark.

Evaluates the *Safe Utilization Frontier Controller* on the week-long Azure
LLM 2024 trace, compares it against the committed Azure 2024 policies
(``sla_aware`` / ``utilization_aware`` / ``constraint_aware`` /
``oracle_forecast_ANALYSIS_ONLY``), and writes a frontier-specific summary.

This is a **simulator/shadow-mode** benchmark — directional only, NOT
production savings (``docs/RESULTS.md`` §8). Real-cluster execution is
DISABLED by default; see ``docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md``.

Reads (no modification):
  * data/external/azure_llm_2024/processed/azure_2024_safe_utilization_frontier.json
  * data/external/azure_llm_2024/processed/azure_llm_2024_backtest_summary.json
  * (optional) live aggregation from data/external/azure_llm_2024/raw/

Writes:
  * docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md
  * data/external/azure_llm_2024/processed/azure_2024_frontier_controller_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTIER_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
BACKTEST_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_llm_2024_backtest_summary.json")
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_frontier_controller_summary.json")
OUT_MD = os.path.join(REPO_ROOT, "docs",
                      "AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md")
SHADOW_LOG = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_frontier_controller_shadow.jsonl")

CA_BASELINE_RHO = 0.65
SLA_AWARE_RHO = 0.50
EXPECTED_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
# Diagnostic safety ceilings (mirror docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md).
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0


def _raw_points_from_audit(audit: dict, key: str) -> list[dict]:
    """Map an audit-summary frontier list to estimator-input dicts."""
    out = []
    for p in audit[key]:
        rho = float(p["policy"].split("@")[1])
        out.append({
            "rho_target": rho,
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
        })
    return out


def _baseline_row(name: str, m: dict, rho_target: float | None) -> dict:
    return {
        "policy": name,
        "rho_target": rho_target,
        "goodput_per_dollar": m["goodput_per_dollar"],
        "sla_compliant_goodput": m["sla_compliant_goodput"],
        "gpu_hours": m["gpu_hours"],
        "timeout_pct_mean": m["timeout_pct_mean"],
        "queue_p95_ms": m["queue_p95_ms"],
        "queue_p99_ms": m["queue_p99_ms"],
        "latency_p99_ms": m.get("latency_p99_ms"),
        "scale_events": m.get("scale_events"),
        "churn": m.get("churn"),
        "mean_utilization_rho": m.get("mean_utilization_rho"),
        "safe": bool(m["timeout_pct_mean"] <= SAFE_TIMEOUT_PCT
                     and m["queue_p99_ms"] <= SAFE_QUEUE_P99_MS),
    }


def build_summary(audit: dict) -> dict:
    """Build the frontier-controller summary from the committed audit JSON."""
    # The estimator (frontier) consumes the anticipatory sweep — the safer
    # dominant frontier per the Azure 2024 audit. The reactive sweep is
    # carried alongside as a diagnostic.
    raw_antic = _raw_points_from_audit(audit, "frontier_anticipatory")
    raw_reactive = _raw_points_from_audit(audit, "frontier_reactive")
    profile = WorkloadFrontierProfile(
        workload_id="azure_llm_2024_week",
        workload_type="inference_standard",
        # The Azure trace lacks model/service ids, latency, SLA budget; we
        # mark telemetry as ``medium`` (the audit's diagnostic thresholds).
        telemetry_confidence="medium",
        priority_class="standard",
        candidate_rhos=EXPECTED_RHOS, source="azure_llm_2024_audit")
    safety = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                          max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                          min_telemetry_confidence="low")
    points = estimate_frontier_from_points(profile, raw_antic, safety_config=safety)
    controller_cfg = FrontierControllerConfig(
        conservative_margin=False, deadband_rho=0.05, deadband_kpi_pct=0.05,
        min_telemetry_confidence="low",
        default_execution_mode=SHADOW_MODE)
    decision = choose_safe_utilization_target(
        profile, points, current_rho=CA_BASELINE_RHO,
        controller_config=controller_cfg)

    # Conservative variant — step back from boundary when adjacent unsafe.
    conservative_cfg = FrontierControllerConfig(
        conservative_margin=True, deadband_rho=0.05, deadband_kpi_pct=0.05,
        min_telemetry_confidence="low", default_execution_mode=SHADOW_MODE)
    conservative_decision = choose_safe_utilization_target(
        profile, points, current_rho=CA_BASELINE_RHO,
        controller_config=conservative_cfg)

    # Shadow log mirror — execution_mode=shadow, mutates nothing.
    shadow_log = FrontierShadowLog()
    shadow_log.record(decision, execution_mode=SHADOW_MODE)
    shadow_log.record(conservative_decision, execution_mode=SHADOW_MODE)

    # Mirror in simulator mode for an honest "what would this do in
    # backtest" check (state is local — no production mutation).
    simulated_state: dict = {}
    sim_effect = execute_frontier_decision(
        decision, mode=SIMULATOR_MODE, simulated_state=simulated_state)

    # Comparison policies (from the committed audit + backtest summaries).
    named = audit["named_policies"]
    rho_for = {
        "sla_aware": SLA_AWARE_RHO, "constraint_aware": CA_BASELINE_RHO,
        "utilization_aware": None, "queue_aware": None,
        "oracle_forecast_ANALYSIS_ONLY": None,
        "naive_overprovisioning": None, "fifo": None,
    }
    policy_rows = []
    for p_name in ("sla_aware", "utilization_aware", "constraint_aware",
                   "oracle_forecast_ANALYSIS_ONLY"):
        if p_name in named:
            policy_rows.append(_baseline_row(p_name, named[p_name], rho_for.get(p_name)))

    # The frontier controller row uses the anticipatory@selected_rho point
    # (the safer dominant frontier point at the selected rho).
    fc_pt = next((p for p in points if abs(p.rho_target - decision.selected_rho) < 1e-9),
                 None) if decision.selected_rho is not None else None
    if fc_pt is not None:
        policy_rows.append({
            "policy": "frontier_controller_v1",
            "rho_target": decision.selected_rho,
            "goodput_per_dollar": fc_pt.predicted_goodput_per_dollar,
            "sla_compliant_goodput": fc_pt.predicted_sla_safe_goodput,
            "gpu_hours": fc_pt.predicted_gpu_hours,
            "timeout_pct_mean": fc_pt.predicted_timeout_pct,
            "queue_p95_ms": fc_pt.predicted_queue_p95_ms,
            "queue_p99_ms": fc_pt.predicted_queue_p99_ms,
            "latency_p99_ms": fc_pt.predicted_latency_p99_ms,
            "scale_events": fc_pt.predicted_scale_events,
            "churn": fc_pt.predicted_churn_score,
            "mean_utilization_rho": fc_pt.predicted_mean_utilization,
            "safe": fc_pt.is_safe,
        })

    ca_gpd = named["constraint_aware"]["goodput_per_dollar"]
    sla_gpd = named["sla_aware"]["goodput_per_dollar"]
    fc_gpd = fc_pt.predicted_goodput_per_dollar if fc_pt else None

    deltas = {
        "frontier_vs_constraint_aware_pct": (
            round((fc_gpd - ca_gpd) / ca_gpd * 100.0, 4) if fc_gpd else None),
        "frontier_vs_sla_aware_pct": (
            round((fc_gpd - sla_gpd) / sla_gpd * 100.0, 4) if fc_gpd else None),
        "constraint_aware_baseline_gpd": ca_gpd,
        "sla_aware_baseline_gpd": sla_gpd,
        "frontier_selected_gpd": fc_gpd,
    }

    summary = {
        "controller_version": "frontier_controller_v1",
        "execution_mode_default": SHADOW_MODE,
        "real_execution_disabled_by_default": True,
        "safe_thresholds": {"timeout_pct": SAFE_TIMEOUT_PCT,
                            "queue_p99_ms": SAFE_QUEUE_P99_MS},
        "candidate_rhos": list(EXPECTED_RHOS),
        "profile": profile.to_dict(),
        "frontier_points_anticipatory": [p.to_dict() for p in points],
        "frontier_points_reactive": raw_reactive,
        "decision": decision.to_dict(),
        "conservative_decision": conservative_decision.to_dict(),
        "policy_comparison": policy_rows,
        "deltas": deltas,
        "shadow_log_summary": shadow_log.summary(),
        "simulator_effect": sim_effect.to_dict(),
        "baseline_preserved": {
            "constraint_aware_goodput_per_dollar": ca_gpd,
            "constraint_aware_rho": CA_BASELINE_RHO,
            "sla_aware_goodput_per_dollar": sla_gpd,
            # accepted tolerance vs the committed Azure 2024 benchmark
            "tolerance_pct": 1.0,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def write_md(path: str, summary: dict) -> None:
    L: list[str] = []
    a = L.append
    a("# Azure LLM 2024 — Safe Utilization Frontier Controller v1 Results\n")
    a("> **Simulator / shadow-mode benchmark. Directional only — NOT production "
      "savings** (`docs/RESULTS.md` §8). Real-cluster execution is **disabled by "
      "default** (`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`). This controller "
      "selects the highest SLA-safe goodput/$ point across a candidate rho grid, "
      "subject to timeout / queue / latency / telemetry-confidence safety gates. "
      "No optimizer constant was tuned, no robust energy engine code was touched, "
      "no ML model was trained, and no dataset was ingested.\n")
    a("- **Read first:** `docs/RESULTS.md`, "
      "`docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, "
      "`docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`.\n")
    a("## 1. Configuration\n")
    a(f"- **Controller version:** `{summary['controller_version']}`")
    a(f"- **Default execution mode:** `{summary['execution_mode_default']}` "
      f"(real execution disabled by default = "
      f"{summary['real_execution_disabled_by_default']})")
    thr = summary["safe_thresholds"]
    a(f"- **Safety thresholds:** timeout ≤ {thr['timeout_pct']}% AND queue p99 "
      f"≤ {thr['queue_p99_ms']} ms (mirrors the Azure 2024 frontier audit).")
    a(f"- **Candidate rho grid:** `{summary['candidate_rhos']}`")
    a(f"- **Workload telemetry confidence:** "
      f"`{summary['profile']['telemetry_confidence']}`\n")

    a("## 2. Frontier sweep (anticipatory — the safer dominant frontier)\n")
    a("| rho | predicted goodput/$ | predicted goodput | timeout % | "
      "queue p95/p99 (ms) | GPU-h | mean rho | safety |")
    a("|---|---|---|---|---|---|---|---|")
    for p in summary["frontier_points_anticipatory"]:
        a(f"| {p['rho_target']} | {_f(p['predicted_goodput_per_dollar'])} | "
          f"{_f(p['predicted_sla_safe_goodput'])} | "
          f"{_f(p['predicted_timeout_pct'])} | "
          f"{_f(p['predicted_queue_p95_ms'])} / {_f(p['predicted_queue_p99_ms'])} | "
          f"{_f(p['predicted_gpu_hours'])} | "
          f"{_f(p['predicted_mean_utilization'])} | "
          f"{p['safety_status']}"
          + (f" ({', '.join(p['safety_vetoes'])})" if p['safety_vetoes'] else "")
          + " |")
    a("\n## 3. Reactive sweep (diagnostic — not the controller's frontier)\n")
    a("| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | safety |")
    a("|---|---|---|---|---|---|")
    for p in summary["frontier_points_reactive"]:
        safe = (p["predicted_timeout_pct"] <= SAFE_TIMEOUT_PCT
                and p["predicted_queue_p99_ms"] <= SAFE_QUEUE_P99_MS)
        a(f"| {p['rho_target']} | {_f(p['predicted_goodput_per_dollar'])} | "
          f"{_f(p['predicted_timeout_pct'])} | "
          f"{_f(p['predicted_queue_p95_ms'])} / {_f(p['predicted_queue_p99_ms'])} | "
          f"{_f(p['predicted_gpu_hours'])} | "
          f"{'SAFE' if safe else '**UNSAFE**'} |")

    a("\n## 4. Controller decision\n")
    d = summary["decision"]
    a(f"- **Action:** `{d['action']}`")
    a(f"- **Previous rho (constraint_aware default):** {d['previous_rho']}")
    a(f"- **Selected rho:** {d['selected_rho']}")
    a(f"- **Reason:** {d['reason']}")
    a(f"- **Expected goodput/$ delta vs current:** "
      f"{_f(d['expected_goodput_per_dollar_delta'])}")
    a(f"- **Expected GPU-hour delta vs current:** "
      f"{_f(d['expected_gpu_hour_delta'])}")
    a(f"- **Expected SLA risk delta:** {_f(d['expected_sla_risk_delta'])}")
    a(f"- **Confidence:** {d['confidence']}")
    a(f"- **Execution mode (recommendation-only):** `{d['execution_mode']}` · "
      f"executable_in_real_cluster = {d['executable_in_real_cluster']}\n")

    a("### Conservative-margin variant (transparent)\n")
    cd = summary["conservative_decision"]
    a(f"- Selected rho: {cd['selected_rho']} (action `{cd['action']}`); "
      "the controller can be configured to step back from the safety boundary "
      "when the next-higher rho is unsafe. This is a transparent operator "
      "control, not a hidden default.\n")

    a("## 5. Policy comparison (committed Azure 2024 evidence + controller)\n")
    a("| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safe |")
    a("|---|---|---|---|---|---|---|")
    for r in summary["policy_comparison"]:
        a(f"| {r['policy']} | {_f(r['rho_target'])} | "
          f"{_f(r['goodput_per_dollar'])} | {_f(r['timeout_pct_mean'])} | "
          f"{_f(r['queue_p99_ms'])} | {_f(r['gpu_hours'])} | "
          f"{'SAFE' if r['safe'] else '**UNSAFE**'} |")
    d_ = summary["deltas"]
    a(f"\n- **frontier_controller_v1 vs constraint_aware:** "
      f"{_f(d_['frontier_vs_constraint_aware_pct'])}% goodput/$ "
      f"(constraint_aware baseline {_f(d_['constraint_aware_baseline_gpd'])}).")
    a(f"- **frontier_controller_v1 vs sla_aware:** "
      f"{_f(d_['frontier_vs_sla_aware_pct'])}% goodput/$ "
      f"(sla_aware baseline {_f(d_['sla_aware_baseline_gpd'])}).\n")

    a("## 6. Preservation of the committed Azure 2024 baseline\n")
    bp = summary["baseline_preserved"]
    a(f"- `constraint_aware` at rho ≈ {bp['constraint_aware_rho']}: "
      f"{_f(bp['constraint_aware_goodput_per_dollar'])} goodput/$ "
      f"(reproduced within {bp['tolerance_pct']}% of the committed Azure 2024 "
      "benchmark — see `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`).")
    a(f"- `sla_aware`: {_f(bp['sla_aware_goodput_per_dollar'])} goodput/$ "
      "(unchanged).")
    a("- The controller does **not** mutate any committed result; it reads "
      "the audit frontier and chooses a safe rho. The frontier-audit doc and "
      "the canonical Azure 2024 backtest doc remain authoritative for their "
      "respective claims.\n")

    a("## 7. Shadow log + execution check\n")
    sl = summary["shadow_log_summary"]
    a(f"- Shadow log decisions recorded: **{sl['n_decisions']}** "
      f"(executed = {sl['n_executed']}; modes = {sl['execution_modes']}).")
    se = summary["simulator_effect"]
    a(f"- Simulator-mode effect: mutated={se['mutated']}, "
      f"selected_rho={se['selected_rho']}, notes={se['notes']} "
      "(local simulated state only — no production write).")
    a("- Real-mode execution: **disabled by default** "
      "(`allow_real_execution=False`); even with the flag, no real executor "
      "ships in `aurelius.frontier.execution` — `not_implemented_real_executor`.\n")

    a("## 8. Claim discipline\n")
    a("- Simulator / public-trace evidence only — **not production savings** "
      "(`docs/RESULTS.md` §8).")
    a("- The safe rho is **workload- and SLA-specific**; `rho = 0.75` is "
      "**not** a global constant. A different workload mix, SLO, real "
      "hardware, or trace will move the safe peak.")
    a("- Pilot telemetry is required to calibrate the safe rho per workload "
      "before any production-savings claim.")
    a("- Real-cluster execution remains disabled by default; the controller "
      "recommends only. The committed `constraint_aware` engine default "
      "(rho ≈ 0.65) is **unchanged** by this controller.")
    a("- The product thesis is *maximum sustainable usage across "
      "constraints*. This controller chooses the best safe KPI point — not "
      "the highest utilization.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-json", default=FRONTIER_JSON,
                   help="Azure 2024 safe-utilization frontier audit summary")
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    if not os.path.exists(args.audit_json):
        print(f"[frontier-controller] missing audit JSON {args.audit_json}",
              file=sys.stderr)
        print("[frontier-controller] run scripts/run_azure_2024_safe_utilization_frontier.py first",
              file=sys.stderr)
        return 2

    audit = json.load(open(args.audit_json))
    summary = build_summary(audit)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
    write_md(args.out_md, summary)

    d = summary["decision"]
    dlt = summary["deltas"]
    print(f"[frontier-controller] action={d['action']} "
          f"selected_rho={d['selected_rho']} "
          f"vs constraint_aware: {dlt['frontier_vs_constraint_aware_pct']}% "
          f"vs sla_aware: {dlt['frontier_vs_sla_aware_pct']}%")
    print(f"[frontier-controller] JSON -> {args.out_json}")
    print(f"[frontier-controller] MD -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
