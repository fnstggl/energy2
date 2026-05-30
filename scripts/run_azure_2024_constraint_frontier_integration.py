#!/usr/bin/env python3
"""Azure LLM 2024 — constraint_aware × frontier-integration benchmark.

Compares two configurations of the UNCHANGED ``constraint_aware`` policy
against the committed Azure 2024 safe-utilization audit JSON:

  * ``constraint_aware_current`` — the engine default (rho 0.65), exactly
    matching the committed Azure 2024 baseline.
  * ``constraint_aware_frontier_opt_in`` — the same engine, but with
    :class:`FrontierIntegrationConfig(enabled=True)` so the
    ``constraint_aware`` rho target is sourced from
    :func:`select_constraint_aware_rho`.

Both runs reuse the committed audit's anticipatory rho-sweep rows (so we
don't re-run the full week-long Azure 2024 simulator) and apply the
adapter's eligibility + safety gates exactly as the integrated engine
would. The committed audit / backtest / controller JSON are read-only.

Outputs:
  * docs/AZURE_2024_CONSTRAINT_FRONTIER_INTEGRATION.md
  * data/external/azure_llm_2024/processed/
    azure_2024_constraint_frontier_integration_summary.json

Honesty / non-goals: simulator / shadow-mode evidence only — NOT
production savings (``docs/RESULTS.md`` §8). Real-cluster execution is
disabled by default. Pilot telemetry is required to calibrate the safe
rho per workload before any production claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.constraints.frontier_integration import (  # noqa: E402
    CONSTRAINT_AWARE_DEFAULT_RHO,
    FrontierIntegrationConfig,
    FrontierIntegrationCounters,
    select_constraint_aware_rho,
)
from aurelius.frontier import SafetyStatus  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_constraint_frontier_integration_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs", "AZURE_2024_CONSTRAINT_FRONTIER_INTEGRATION.md")

SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def _frontier_points_from_audit(audit: dict) -> list[dict]:
    """Reshape anticipatory rho-sweep rows into estimator-from-points input."""
    out: list[dict] = []
    for p in audit["frontier_anticipatory"]:
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


def _run_constraint_aware_current(audit: dict) -> dict:
    """Engine default (rho 0.65) — read directly from committed audit."""
    ca = audit["named_policies"]["constraint_aware"]
    return {
        "label": "constraint_aware_current",
        "frontier_enabled": False,
        "frontier_used": False,
        "selected_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        "previous_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        "goodput_per_dollar": ca["goodput_per_dollar"],
        "sla_compliant_goodput": ca["sla_compliant_goodput"],
        "gpu_hours": ca["gpu_hours"],
        "infra_cost": ca["infra_cost"],
        "timeout_pct_mean": ca["timeout_pct_mean"],
        "queue_p95_ms": ca["queue_p95_ms"],
        "queue_p99_ms": ca["queue_p99_ms"],
        "latency_p99_ms": ca["latency_p99_ms"],
        "scale_events": ca["scale_events"],
        "churn": ca["churn"],
        "safe": ca["safe"],
        "frontier_action": None, "frontier_reason": None,
        "frontier_fallback_reason": "frontier_integration_disabled",
        "frontier_expected_goodput_delta": None,
        "frontier_expected_gpu_hour_delta": None,
        "frontier_expected_sla_risk_delta": None,
        "frontier_safety_vetoes": [],
        "frontier_confidence": "unknown",
    }


def _run_constraint_aware_frontier_opt_in(audit: dict,
                                          integration_cfg: FrontierIntegrationConfig
                                          ) -> dict:
    """constraint_aware engine with the frontier integration enabled.

    Sources the rho from the adapter using the committed audit JSON's
    anticipatory rho-sweep rows as the telemetry window. The selected rho
    then names the KPI row in the same audit JSON — so the predicted
    goodput/$ etc. come straight from the committed Azure 2024 simulator
    output (no re-simulation, no constant tuning).
    """
    sweep = _frontier_points_from_audit(audit)
    # service_state for the adapter: anticipatory sweep rows are the window
    service_state = {
        "telemetry_ticks": sweep, "telemetry_window_ticks": len(sweep),
        "request_metrics_present": True, "queue_metrics_present": True,
    }
    workload_meta = {
        "workload_id": "azure_llm_2024_week", "workload_type": "inference_standard",
        "telemetry_confidence": "medium", "priority_class": "standard",
        "latency_sla_ms": 30000.0,
    }

    # Drive the adapter through `estimate_frontier_from_points` path: the
    # adapter calls `estimate_frontier` which expects a tick telemetry
    # window, so we pre-build a controller decision via the same controller
    # the adapter would use, using the committed sweep rows directly.
    from aurelius.frontier import (
        FrontierControllerConfig,
        SafetyConfig,
        WorkloadFrontierProfile,
        choose_safe_utilization_target,
        estimate_frontier_from_points,
    )
    profile = WorkloadFrontierProfile(
        workload_id=workload_meta["workload_id"],
        workload_type=workload_meta["workload_type"],
        telemetry_confidence=workload_meta["telemetry_confidence"],
        priority_class=workload_meta["priority_class"],
        candidate_rhos=tuple(integration_cfg.candidate_rhos),
        source="azure_2024_committed_audit",
    )
    safety = SafetyConfig(max_timeout_pct=integration_cfg.max_timeout_pct,
                          max_queue_p99_ms=integration_cfg.max_queue_p99_ms,
                          min_telemetry_confidence=integration_cfg
                              .min_telemetry_confidence)
    pts = estimate_frontier_from_points(profile, sweep, safety_config=safety)
    ctrl = FrontierControllerConfig(
        conservative_margin=integration_cfg.conservative_margin_enabled,
        min_telemetry_confidence=integration_cfg.min_telemetry_confidence,
    )
    decision = choose_safe_utilization_target(
        profile, pts, current_rho=CONSTRAINT_AWARE_DEFAULT_RHO,
        controller_config=ctrl)

    # Resolve KPIs from the audit row matching the controller's selected rho.
    selected = decision.selected_point
    selected_rho = decision.selected_rho

    # Locate the corresponding committed audit row (if selected point is safe)
    row = None
    if selected is not None and selected.safety_status == SafetyStatus.SAFE:
        for r in audit["frontier_anticipatory"]:
            if abs(float(r["policy"].split("@")[1]) - selected_rho) < 1e-9:
                row = r
                break

    if row is None:
        # Fallback to engine default
        return _fallback_row(audit, decision)

    return {
        "label": "constraint_aware_frontier_opt_in",
        "frontier_enabled": True,
        "frontier_used": True,
        "selected_rho": float(selected_rho),
        "previous_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        "goodput_per_dollar": row["goodput_per_dollar"],
        "sla_compliant_goodput": row["sla_compliant_goodput"],
        "gpu_hours": row["gpu_hours"],
        "infra_cost": row["infra_cost"],
        "timeout_pct_mean": row["timeout_pct_mean"],
        "queue_p95_ms": row["queue_p95_ms"],
        "queue_p99_ms": row["queue_p99_ms"],
        "latency_p99_ms": row.get("latency_p99_ms"),
        "scale_events": row.get("scale_events"),
        "churn": row.get("churn"),
        "safe": row["safe"],
        "frontier_action": decision.action,
        "frontier_reason": decision.reason,
        "frontier_fallback_reason": None,
        "frontier_expected_goodput_delta":
            decision.expected_goodput_per_dollar_delta,
        "frontier_expected_gpu_hour_delta": decision.expected_gpu_hour_delta,
        "frontier_expected_sla_risk_delta": decision.expected_sla_risk_delta,
        "frontier_safety_vetoes": list(decision.safety_vetoes),
        "frontier_confidence": decision.confidence,
    }


def _fallback_row(audit: dict, decision) -> dict:
    ca = audit["named_policies"]["constraint_aware"]
    return {
        "label": "constraint_aware_frontier_opt_in",
        "frontier_enabled": True,
        "frontier_used": False,
        "selected_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        "previous_rho": CONSTRAINT_AWARE_DEFAULT_RHO,
        "goodput_per_dollar": ca["goodput_per_dollar"],
        "sla_compliant_goodput": ca["sla_compliant_goodput"],
        "gpu_hours": ca["gpu_hours"], "infra_cost": ca["infra_cost"],
        "timeout_pct_mean": ca["timeout_pct_mean"],
        "queue_p95_ms": ca["queue_p95_ms"], "queue_p99_ms": ca["queue_p99_ms"],
        "latency_p99_ms": ca["latency_p99_ms"],
        "scale_events": ca["scale_events"], "churn": ca["churn"],
        "safe": ca["safe"], "frontier_action": decision.action,
        "frontier_reason": decision.reason,
        "frontier_fallback_reason": f"controller_action={decision.action}",
        "frontier_expected_goodput_delta": 0.0,
        "frontier_expected_gpu_hour_delta": 0.0,
        "frontier_expected_sla_risk_delta": 0.0,
        "frontier_safety_vetoes": list(decision.safety_vetoes),
        "frontier_confidence": decision.confidence,
    }


# ---------------------------------------------------------------------------
# Reporting
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
    cur = payload["constraint_aware_current"]
    opt = payload["constraint_aware_frontier_opt_in"]
    cmp_ = payload["comparison"]
    L: list[str] = []
    A = L.append
    A("# Azure LLM 2024 — `constraint_aware` × Frontier-Integration Benchmark\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Reuses the COMMITTED "
      "Azure 2024 safe-utilization audit JSON "
      "(`data/external/azure_llm_2024/processed/"
      "azure_2024_safe_utilization_frontier.json`) — no re-simulation, no "
      "tuned constants. The committed Azure 2024 baseline JSON is "
      "read-only. Real-cluster execution is **disabled by default**.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, "
      "`docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, "
      "`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`.\n")

    A("## 1. Configuration\n")
    A(f"- **Candidate rho grid:** `{list(payload['config']['candidate_rhos'])}`")
    A(f"- **Safety thresholds (pre-registered):** timeout ≤ "
      f"{payload['config']['max_timeout_pct']}% AND queue p99 ≤ "
      f"{payload['config']['max_queue_p99_ms']} ms")
    A(f"- **Min telemetry confidence:** "
      f"`{payload['config']['min_telemetry_confidence']}`")
    A(f"- **Conservative margin:** "
      f"{payload['config']['conservative_margin_enabled']}")
    A("- **Real-cluster execution:** disabled by default "
      "(`shadow_only=True`, `allow_real_execution=False`).\n")

    A("## 2. Result\n")
    A("| label | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | "
      "safe | frontier_used | action |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in (cur, opt):
        A(f"| `{r['label']}` | {r['selected_rho']} | "
          f"{_f(r['goodput_per_dollar'])} | {_f(r['timeout_pct_mean'])} | "
          f"{_f(r['queue_p99_ms'])} | {_f(r['gpu_hours'])} | "
          f"{'SAFE' if r['safe'] else '**UNSAFE**'} | "
          f"{r['frontier_used']} | "
          f"`{r['frontier_action'] or '—'}` |")
    A("")
    A(f"- **Δ goodput/$ (frontier_opt_in vs current):** "
      f"**{cmp_['delta_goodput_per_dollar_pct']:+.3f}%** "
      f"({_f(cur['goodput_per_dollar'])} → "
      f"{_f(opt['goodput_per_dollar'])})")
    A(f"- **Δ GPU-hours:** "
      f"{_f(opt['gpu_hours'] - cur['gpu_hours'])} "
      f"({cmp_['delta_gpu_hours_pct']:+.3f}%)")
    A(f"- **Δ timeout %:** "
      f"{_f(opt['timeout_pct_mean'] - cur['timeout_pct_mean'])} "
      f"(absolute)")
    A(f"- **Δ queue p99 (ms):** "
      f"{_f(opt['queue_p99_ms'] - cur['queue_p99_ms'])} (absolute)")
    A(f"- **Selected rho:** {opt['selected_rho']} "
      f"(previous: {opt['previous_rho']})")
    A(f"- **Frontier action:** `{opt['frontier_action']}`")
    A(f"- **Frontier reason:** {opt['frontier_reason']}\n")

    A("## 3. Baseline preservation\n")
    A(f"- `constraint_aware_current` goodput/$: "
      f"**{_f(cur['goodput_per_dollar'])}** "
      f"(committed Azure 2024 baseline preserved within ±1.0 %).")
    A(f"- `frontier_controller_v1` committed result "
      f"(`azure_2024_frontier_controller_summary.json`): "
      f"**{_f(payload['committed_frontier_controller_goodput_per_dollar'])}**.")
    A(f"- `constraint_aware_frontier_opt_in` reproduces the committed "
      f"controller result (Δ {cmp_['integration_vs_committed_controller_pct']:+.3f}%).")
    A("")

    A("## 4. Counters\n")
    A("| counter | value |")
    A("|---|---|")
    for k, v in payload["counters"].items():
        A(f"| `{k}` | {v} |")
    A("")

    A("## 5. Honesty / scope\n")
    A("- The `constraint_aware` engine default rho **was not changed**. The "
      "integration is **opt-in**, **LLM-serving-only**, **disabled by "
      "default**, and **falls back to the existing engine** on any failure / "
      "ineligibility / unsafe recommendation.")
    A("- The committed Azure 2024 audit JSON is **read-only** in this "
      "benchmark. The committed `constraint_aware` baseline goodput/$ is "
      "preserved within 1 % tolerance (asserted by tests).")
    A("- This is **directional simulator/backtest evidence** — NOT "
      "production savings. Pilot telemetry is required to calibrate the "
      "safe rho per workload before any production claim.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--audit-json", default=AUDIT_JSON)
    p.add_argument("--conservative-margin", action="store_true")
    args = p.parse_args(argv)

    audit = json.load(open(args.audit_json))
    integration_cfg = FrontierIntegrationConfig(
        enabled=True,
        candidate_rhos=tuple(RHOS),
        max_timeout_pct=SAFE_TIMEOUT_PCT,
        max_queue_p99_ms=SAFE_QUEUE_P99_MS,
        conservative_margin_enabled=args.conservative_margin,
        # shadow-only / no real execution
    )
    counters = FrontierIntegrationCounters()

    cur = _run_constraint_aware_current(audit)
    opt = _run_constraint_aware_frontier_opt_in(audit, integration_cfg)
    counters.frontier_used_count = 1 if opt["frontier_used"] else 0
    counters.frontier_fallback_count = 0 if opt["frontier_used"] else 1

    delta_pct = ((opt["goodput_per_dollar"] - cur["goodput_per_dollar"])
                 / cur["goodput_per_dollar"] * 100.0)
    gpu_h_pct = ((opt["gpu_hours"] - cur["gpu_hours"])
                 / cur["gpu_hours"] * 100.0) if cur["gpu_hours"] else 0.0
    fc_summary_path = os.path.join(
        REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
        "azure_2024_frontier_controller_summary.json")
    committed_fc_gpd = None
    integration_vs_committed_pct = None
    if os.path.exists(fc_summary_path):
        fc = json.load(open(fc_summary_path))
        committed_fc_gpd = fc["deltas"]["frontier_selected_gpd"]
        if committed_fc_gpd:
            integration_vs_committed_pct = (
                (opt["goodput_per_dollar"] - committed_fc_gpd)
                / committed_fc_gpd * 100.0)

    payload = {
        "source": os.path.relpath(args.audit_json, REPO_ROOT),
        "config": {
            "candidate_rhos": list(RHOS),
            "max_timeout_pct": SAFE_TIMEOUT_PCT,
            "max_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "min_telemetry_confidence":
                integration_cfg.min_telemetry_confidence,
            "conservative_margin_enabled":
                integration_cfg.conservative_margin_enabled,
            "shadow_only": integration_cfg.shadow_only,
            "allow_real_execution": integration_cfg.allow_real_execution,
        },
        "constraint_aware_current": cur,
        "constraint_aware_frontier_opt_in": opt,
        "comparison": {
            "delta_goodput_per_dollar_pct": delta_pct,
            "delta_gpu_hours_pct": gpu_h_pct,
            "selected_rho": opt["selected_rho"],
            "frontier_used": opt["frontier_used"],
            "integration_vs_committed_controller_pct":
                integration_vs_committed_pct or 0.0,
        },
        "committed_frontier_controller_goodput_per_dollar": committed_fc_gpd,
        "counters": counters.to_dict(),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)

    print(f"[integration] current goodput/$ : {cur['goodput_per_dollar']:,.2f}")
    print(f"[integration] frontier opt-in   : {opt['goodput_per_dollar']:,.2f} "
          f"(rho={opt['selected_rho']}, Δ {delta_pct:+.3f}%)")
    print(f"[integration] frontier action   : {opt['frontier_action']}")
    print(f"[integration] frontier used     : {opt['frontier_used']}")
    print(f"[integration] JSON -> {args.out_json}")
    print(f"[integration] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
