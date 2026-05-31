#!/usr/bin/env python3
"""Training Safe Utilization Frontier — public-trace benchmark.

Runs the Training Frontier estimator against both currently-integrated
training/packing traces:

  * Philly  (queue / starvation / backfill / multi-GPU gang pressure)
  * Alibaba GPU v2023 (packing / fragmentation / heterogeneous fleet)

Compares ``training_frontier_v1`` against the canonical packing /
scheduling baselines (fifo, first_fit, best_fit, FFD, greedy_packing,
topology_aware/utilization_aware where reported, constraint_aware) and
reports whether training_frontier beats, ties, or loses to the
trace's existing ``constraint_aware`` baseline.

Reads only **committed** Philly + Alibaba GPU backtest summary JSONs —
no new datasets, no ML training, no robust-energy-engine change. The
committed Azure 2024 / serving frontier artifacts are NOT touched.

Outputs (NEW files only — committed Philly / Alibaba JSON unchanged):
  * docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md
  * data/external/frontier/training_frontier_summary.json

Directional simulator / public-trace evidence only — NOT production
savings (``docs/RESULTS.md`` §8). Real-cluster execution is disabled
by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.frontier import (  # noqa: E402
    PHILLY_POLICY_CANDIDATES,
    TrainingControllerConfig,
    TrainingFrontierAction,
    TrainingSafetyConfig,
    TrainingSafetyStatus,
    TrainingWorkloadProfile,
    choose_training_frontier_target,
    estimate_alibaba_gpu_training_frontier,
    estimate_philly_training_frontier,
    load_alibaba_gpu_summary,
    load_philly_summary,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "training_frontier_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs", "TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md")
PHILLY_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "philly", "processed",
    "philly_backtest_summary.json")
ALIBABA_SUMMARY = os.path.join(
    REPO_ROOT, "data", "external", "alibaba_gpu", "processed",
    "alibaba_gpu_backtest_summary.json")

# Tie band used for verdict classification (±1 %).
TIE_BAND_PCT = 1.0


def _point_row(p, *, source: str) -> dict:
    return {
        "source": source,
        "policy": p.candidate.source_policy,
        "candidate": p.candidate.to_dict(),
        "predicted_goodput_per_dollar": p.predicted_goodput_per_dollar,
        "predicted_gpu_occupancy": p.predicted_gpu_occupancy,
        "predicted_packing_density": p.predicted_packing_density,
        "predicted_gpu_hours": p.predicted_gpu_hours,
        "predicted_completed_work": p.predicted_completed_work,
        "predicted_queue_wait_p95_s": p.predicted_queue_wait_p95_s,
        "predicted_queue_wait_p99_s": p.predicted_queue_wait_p99_s,
        "predicted_starvation_rate_pct": p.predicted_starvation_rate_pct,
        "predicted_fragmentation_block_rate_pct":
            p.predicted_fragmentation_block_rate_pct,
        "predicted_gang_scheduling_failure_pct":
            p.predicted_gang_scheduling_failure_pct,
        "predicted_backfill_success_rate_pct":
            p.predicted_backfill_success_rate_pct,
        "predicted_retry_waste_gpu_hours": p.predicted_retry_waste_gpu_hours,
        "predicted_cost": p.predicted_cost,
        "safety_status": p.safety_status,
        "safety_vetoes": list(p.safety_vetoes),
        "notes": list(p.notes),
    }


def _verdict(selected_gpd: Optional[float], ca_gpd: Optional[float]) -> str:
    if selected_gpd is None or ca_gpd is None or ca_gpd == 0:
        return "INSUFFICIENT_DATA"
    delta = (selected_gpd - ca_gpd) / ca_gpd * 100.0
    if abs(delta) <= TIE_BAND_PCT:
        return "TIE"
    if delta > TIE_BAND_PCT:
        return "TRAINING_FRONTIER_WIN"
    return "TRAINING_FRONTIER_LOSS"


def _run_trace(name: str, points, *, current_policy: str,
               workload_id: str) -> dict:
    """Score one trace's frontier + controller decision."""
    if not points:
        return {"trace": name, "applicable": False,
                "exclusion_reason": "no points produced"}
    current_candidate = PHILLY_POLICY_CANDIDATES.get(current_policy)
    if current_candidate is None:
        # Fall back to any candidate naming this policy in the points.
        for p in points:
            if p.candidate.source_policy == current_policy:
                current_candidate = p.candidate
                break
    dec = choose_training_frontier_target(
        points,
        current_candidate=current_candidate,
        config=TrainingControllerConfig(),
        workload_id=workload_id,
        telemetry_confidence="medium")
    ca_point = next((p for p in points
                     if p.candidate.source_policy == current_policy), None)
    ca_gpd = (ca_point.predicted_goodput_per_dollar
              if ca_point is not None else None)
    selected_gpd = (dec.selected_point.predicted_goodput_per_dollar
                    if dec.selected_point is not None else None)
    return {
        "trace": name, "applicable": True,
        "workload_id": workload_id,
        "current_policy": current_policy,
        "current_goodput_per_dollar": ca_gpd,
        "selected_policy": (
            dec.selected_candidate.source_policy
            if dec.selected_candidate is not None else None),
        "selected_goodput_per_dollar": selected_gpd,
        "delta_vs_constraint_aware_pct": (
            (selected_gpd - ca_gpd) / ca_gpd * 100.0
            if ca_gpd and selected_gpd is not None else None),
        "verdict": _verdict(selected_gpd, ca_gpd),
        "action": dec.action,
        "reason": dec.reason,
        "expected_queue_wait_delta_s": dec.expected_queue_wait_delta_s,
        "expected_fragmentation_delta_pct":
            dec.expected_fragmentation_delta_pct,
        "expected_starvation_delta_pct":
            dec.expected_starvation_delta_pct,
        "expected_gpu_hour_delta": dec.expected_gpu_hour_delta,
        "decision": dec.to_dict(),
        "frontier_points": [_point_row(p, source=name) for p in points],
        "safe_points_count": sum(
            1 for p in points if p.is_safe),
        "unsafe_points_count": sum(
            1 for p in points
            if p.safety_status == TrainingSafetyStatus.UNSAFE),
        "insufficient_telemetry_points_count": sum(
            1 for p in points
            if p.safety_status == TrainingSafetyStatus.INSUFFICIENT_TELEMETRY),
    }


def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _frontier_table(rows, append, *, source: str):
    append(f"### {source} frontier sweep\n")
    append("| policy | goodput/$ | occupancy | queue p99 (s) | "
           "starv % | frag block % | backfill % | safety |")
    append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        append(
            f"| `{r['policy']}` | "
            f"{_f(r['predicted_goodput_per_dollar'])} | "
            f"{_f(r['predicted_gpu_occupancy'], nd=4)} | "
            f"{_f(r['predicted_queue_wait_p99_s'])} | "
            f"{_f(r['predicted_starvation_rate_pct'])} | "
            f"{_f(r['predicted_fragmentation_block_rate_pct'])} | "
            f"{_f(r['predicted_backfill_success_rate_pct'])} | "
            f"**{r['safety_status']}** |")
    append("")


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Training Safe Utilization Frontier — v1 Results\n")
    A("> **Simulator / public-trace benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Training Frontier "
      "v1 reads only the COMMITTED Philly + Alibaba GPU v2023 backtest "
      "summaries. The serving Safe Utilization Frontier Controller, "
      "the robust energy engine, and every committed benchmark artifact "
      "are **unchanged**. Real-cluster execution is **disabled by "
      "default**.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/PUBLIC_TRACE_BACKTESTS.md`, "
      "`docs/PHILLY_BACKTEST_RESULTS.md`, "
      "`docs/ALIBABA_GPU_BACKTEST_RESULTS.md`, "
      "`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, "
      "`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, "
      "`docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`.\n")

    cfg = payload["config"]
    A("## 1. Configuration\n")
    A(f"- **Trace sources:** Philly + Alibaba GPU v2023")
    A(f"- **Tie band:** ±{cfg['tie_band_pct']} % goodput/$")
    A(f"- **Default safety thresholds:**")
    A(f"  - `max_queue_wait_p95_s`: {cfg['max_queue_wait_p95_s']}")
    A(f"  - `max_queue_wait_p99_s`: {cfg['max_queue_wait_p99_s']}")
    A(f"  - `max_starvation_rate_pct`: {cfg['max_starvation_rate_pct']}")
    A(f"  - `max_fragmentation_block_rate_pct`: "
      f"{cfg['max_fragmentation_block_rate_pct']}")
    A(f"  - `max_gang_scheduling_failure_pct`: "
      f"{cfg['max_gang_scheduling_failure_pct']} "
      "*(disabled for Philly — see §3 missing-signals note)*")
    A(f"- **Real-cluster execution:** disabled by default.\n")

    A("## 2. Per-trace summary\n")
    A("| trace | current_policy | current goodput/$ | training_frontier_v1 | "
      "Δ vs current | verdict | action | safe / unsafe / insufficient |")
    A("|---|---|---|---|---|---|---|---|")
    for r in payload["per_trace"]:
        if not r["applicable"]:
            A(f"| `{r['trace']}` | — | — | — | — | _excluded_ | — | — |")
            continue
        d = r['delta_vs_constraint_aware_pct']
        delta_str = f"{d:+.3f}%" if d is not None else "—"
        A(f"| `{r['trace']}` | `{r['current_policy']}` | "
          f"{_f(r['current_goodput_per_dollar'])} | "
          f"{_f(r['selected_goodput_per_dollar'])} → "
          f"`{r['selected_policy']}` | "
          f"{delta_str} | **{r['verdict']}** | `{r['action']}` | "
          f"{r['safe_points_count']} / {r['unsafe_points_count']} / "
          f"{r['insufficient_telemetry_points_count']} |")
    A("")

    A("## 3. Per-trace frontier sweeps + missing-signals notes\n")
    for r in payload["per_trace"]:
        if not r["applicable"]:
            continue
        _frontier_table(r["frontier_points"], A, source=r["trace"])
        A(f"**Controller decision:** `{r['action']}` → policy "
          f"`{r['selected_policy']}` "
          f"({_f(r['selected_goodput_per_dollar'])} goodput/$)")
        A(f"**Reason:** {r['reason']}\n")
        # Missing-signals note per trace
        if r["trace"] == "philly":
            A("**Missing signals on Philly (not invented):**")
            A("- per-policy gang-scheduling failure: NOT cleanly "
              "labelled; the gate is **disabled by default**. "
              "`failed_or_killed_run` includes non-gang causes.")
            A("- per-job completion p95 / p99: not reported by the "
              "committed summary (only `mean_completion_s`).")
            A("- GPU model price heterogeneity: Philly has no GPU "
              "model column.\n")
        elif r["trace"] == "alibaba_gpu":
            A("**Missing signals on Alibaba GPU (not invented):**")
            A("- per-job queue wait p95 / p99: NOT reported by the "
              "static packing baseline (no consistent submit / start "
              "times). Queue gates are **disabled by default**.")
            A("- starvation rate: not directly measured; "
              "`stranded_jobs / n_gpu_jobs` is reported as a "
              "fragmentation-pressure proxy and explicitly NOT "
              "labelled as starvation in the per-policy notes.")
            A("- gang-scheduling failure: NOT measured; gate "
              "**disabled by default**.")
            A("- retry / wasted GPU-hours: NOT measured; gate "
              "**disabled by default**.\n")

    A("## 4. What metrics were unavailable (consolidated)\n")
    A("| signal | Philly | Alibaba GPU |")
    A("|---|---|---|")
    A("| queue wait p95 / p99 | ✅ measured | ❌ not measured |")
    A("| starvation rate | ✅ measured | ⚠ approximated via "
      "stranded fraction (labelled in notes) |")
    A("| fragmentation block | ✅ measured "
      "(`failed_placement_rate_pct`) | ✅ measured "
      "(`fragmentation_score`) |")
    A("| backfill success | ✅ measured | ❌ not measured |")
    A("| gang-scheduling failure | ❌ not cleanly labelled | "
      "❌ not measured |")
    A("| retry / waste GPU-hours | ✅ committed "
      "`attempt_analysis.wasted_gpu_hours_from_retries` | "
      "❌ not measured |")
    A("| per-job p95 / p99 completion | ❌ not in summary | "
      "❌ not reported |")
    A("| GPU model price heterogeneity | ❌ no GPU type column | "
      "✅ measured |\n")

    A("## 5. Honesty / scope\n")
    A("- Training Frontier v1 is the **sibling** of the serving Safe "
      "Utilization Frontier Controller — it does NOT optimize request "
      "latency, does NOT use the serving rho controller, and does "
      "NOT replace any existing scheduling / packing baseline.")
    A("- Training Frontier v1 is **opt-in**, **shadow / simulator** "
      "only, and **does not mutate** real infrastructure.")
    A("- No new datasets ingested. MIT Supercloud is the next "
      "validation step (out of scope for this PR).")
    A("- Public-trace evidence only — **NOT production savings** "
      "(`docs/RESULTS.md` §8). Pilot telemetry is required to "
      "calibrate per-tenant safety thresholds.")
    A("- The committed Philly / Alibaba GPU backtest summaries are "
      "**read-only** in this benchmark; the serving frontier code is "
      "**unchanged**.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--philly-summary", default=PHILLY_SUMMARY)
    p.add_argument("--alibaba-summary", default=ALIBABA_SUMMARY)
    p.add_argument("--philly-current-policy", default="constraint_aware")
    p.add_argument("--alibaba-current-policy", default="constraint_aware")
    args = p.parse_args(argv)

    per_trace: list[dict] = []

    # Philly
    if os.path.exists(args.philly_summary):
        ph = json.load(open(args.philly_summary))
        ph_points = estimate_philly_training_frontier(
            ph, telemetry_confidence="medium")
        per_trace.append(_run_trace(
            "philly", ph_points,
            current_policy=args.philly_current_policy,
            workload_id="philly_training_workload"))
    else:
        per_trace.append({
            "trace": "philly", "applicable": False,
            "exclusion_reason":
                f"Philly summary not present: {args.philly_summary}"})

    # Alibaba GPU
    if os.path.exists(args.alibaba_summary):
        ag = json.load(open(args.alibaba_summary))
        ag_points = estimate_alibaba_gpu_training_frontier(
            ag, telemetry_confidence="medium")
        per_trace.append(_run_trace(
            "alibaba_gpu", ag_points,
            current_policy=args.alibaba_current_policy,
            workload_id="alibaba_gpu_packing_workload"))
    else:
        per_trace.append({
            "trace": "alibaba_gpu", "applicable": False,
            "exclusion_reason":
                f"Alibaba summary not present: {args.alibaba_summary}"})

    safety_defaults = TrainingSafetyConfig()
    config = {
        "tie_band_pct": TIE_BAND_PCT,
        "max_queue_wait_p95_s": safety_defaults.max_queue_wait_p95_s,
        "max_queue_wait_p99_s": safety_defaults.max_queue_wait_p99_s,
        "max_starvation_rate_pct": safety_defaults.max_starvation_rate_pct,
        "max_fragmentation_block_rate_pct":
            safety_defaults.max_fragmentation_block_rate_pct,
        "max_gang_scheduling_failure_pct":
            safety_defaults.max_gang_scheduling_failure_pct,
        "min_telemetry_confidence":
            safety_defaults.min_telemetry_confidence,
        "real_execution_disabled_by_default": True,
        "execution_mode_default": "shadow",
    }
    applicable = [t for t in per_trace if t["applicable"]]
    verdict_counts = {"TRAINING_FRONTIER_WIN": 0, "TIE": 0,
                       "TRAINING_FRONTIER_LOSS": 0,
                       "INSUFFICIENT_DATA": 0}
    for t in applicable:
        verdict_counts[t["verdict"]] = verdict_counts.get(t["verdict"], 0) + 1

    payload = {
        "config": config,
        "per_trace": per_trace,
        "synthesis": {
            "n_applicable": len(applicable),
            "n_excluded": len(per_trace) - len(applicable),
            "verdict_counts": verdict_counts,
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)

    print(f"[training-frontier] applicable={len(applicable)} "
          f"excluded={len(per_trace) - len(applicable)}")
    for t in per_trace:
        if not t["applicable"]:
            continue
        delta = t.get('delta_vs_constraint_aware_pct')
        delta_str = f"{delta:+.3f}%" if delta is not None else "—"
        print(f"  {t['trace']:14s} verdict={t['verdict']:25s} "
              f"action={t['action']:30s} selected={t['selected_policy']:24s} "
              f"Δ={delta_str}")
    print(f"[training-frontier] JSON -> {args.out_json}")
    print(f"[training-frontier] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
