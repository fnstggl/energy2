#!/usr/bin/env python3
"""MIT Supercloud — Training Safe Utilization Frontier benchmark.

Validates Training Frontier v1 (``aurelius/frontier/training_*``) on
MIT Supercloud (Samsi et al., HPEC 2021) — answering the open question
from ``docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md``:

  *Training Frontier v1 currently TIES constraint_aware on Philly +
   Alibaba GPU. Is that because constraint_aware already captures the
   safe training frontier, or because those traces lack the richer
   scheduler + GPU-utilization telemetry needed to expose Training
   Frontier alpha?*

This script:

1. Loads the MIT Supercloud scheduler log + labelled-job mapping (raw
   if present at ``--source-dir``, else the small synthetic fixture
   in ``tests/fixtures/mit_supercloud_sample/``).
2. Converts MIT jobs to the cross-dataset ``NormalizedGPUJob`` contract
   (gpu_count from tres_req, queue_wait from start − submit, etc.).
3. Builds a synthetic ``GPUNode`` fleet sized to the trace's peak GPU
   demand (MIT publishes per-node *utilization* but not per-node
   capacity; the synthetic fleet keeps the comparison **relative**).
4. Runs the unchanged ``gpu_scheduling.run_backtest`` for the full
   policy grid (fifo / first_fit / best_fit / FFD / greedy_packing /
   topology_aware / utilization_aware / constraint_aware).
5. Maps each policy's measured outcome to a ``TrainingFrontierPoint``
   and runs ``choose_training_frontier_target`` to pick the safe peak.
6. Reports beats / safely-ties / loses verdict vs ``constraint_aware``.

Outputs (NEW files only; committed Philly / Alibaba / MIT raw artifacts
are READ-ONLY):

  * docs/MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md
  * data/external/mit_supercloud/processed/
    mit_supercloud_training_frontier_summary.json

Directional public-trace / simulator evidence only — NOT production
savings (``docs/RESULTS.md`` §8). Real-cluster execution is disabled
by default. No serving-frontier code is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.frontier import (  # noqa: E402
    PHILLY_POLICY_CANDIDATES,
    TrainingControllerConfig,
    TrainingFrontierAction,
    TrainingFrontierCandidate,
    TrainingFrontierPoint,
    TrainingSafetyConfig,
    TrainingSafetyStatus,
    choose_training_frontier_target,
    classify_training_frontier_point,
)
from aurelius.traces import gpu_scheduling as gs  # noqa: E402
from aurelius.traces import mit_supercloud as mit  # noqa: E402
from aurelius.traces.gpu_packing import GPUNode  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(REPO_ROOT, "data", "external",
                            "mit_supercloud", "raw")
DEFAULT_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                                "mit_supercloud_sample")
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "mit_supercloud", "processed",
    "mit_supercloud_training_frontier_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs", "MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md")

TIE_BAND_PCT = 1.0


# ---------------------------------------------------------------------------
# Fleet synthesis (MIT publishes node utilization but not per-node
# capacity; we size the synthetic fleet to the trace's peak GPU demand).
# ---------------------------------------------------------------------------

def _synth_fleet(jobs, *, gpus_per_node: int, node_overhead_factor: float
                  ) -> list:
    if not jobs:
        return []
    peak = max(int(j.gpu_count or 0) for j in jobs) or 1
    total_demand = sum(int(j.gpu_count or 0) * (j.duration_s or 0.0)
                       for j in jobs)
    # Sum of GPU-seconds demanded — choose a fleet that can deliver the
    # demand over the trace window with the configured overhead factor.
    subs = [j.submit_time_s for j in jobs if j.submit_time_s is not None]
    ends = [j.end_time_s for j in jobs if j.end_time_s is not None]
    if subs and ends:
        window_s = max(ends) - min(subs)
    else:
        window_s = (max((j.duration_s or 0.0) for j in jobs)
                    if jobs else 1.0)
    window_s = max(60.0, window_s)
    avg_demand_gpu = total_demand / window_s if window_s else 1.0
    fleet_gpu = max(
        peak,
        int(math.ceil(avg_demand_gpu * node_overhead_factor)))
    n_nodes = max(1, int(math.ceil(fleet_gpu / max(1, gpus_per_node))))
    return [GPUNode(
        node_id=f"mit-synth-node-{i:03d}",
        gpu_count=gpus_per_node,
        gpu_model="V100",
        cpu_milli=40_000, memory_mib=384_000) for i in range(n_nodes)]


# ---------------------------------------------------------------------------
# Map a Philly-style SchedResult into a TrainingFrontierPoint, in the
# same shape the Philly estimator produces.
# ---------------------------------------------------------------------------

def _point_from_sched_policy(name: str, sched, *, n_scheduled: int,
                              safety_config: TrainingSafetyConfig
                              ) -> TrainingFrontierPoint:
    cand = PHILLY_POLICY_CANDIDATES.get(
        name, TrainingFrontierCandidate(source_policy=name))
    util_pct = sched.utilization_mean_pct
    occupancy = util_pct / 100.0 if util_pct is not None else None

    starvation_rate_pct = (
        100.0 * float(sched.starvation_events) / max(1, n_scheduled)
        if sched.starvation_events is not None and n_scheduled else None)
    backfill_success_rate_pct = (
        100.0 * float(sched.backfill_placements) / max(1, n_scheduled)
        if sched.backfill_placements is not None and n_scheduled else None)
    # Use ``fragmentation_loss_pct`` (events / attempts; bounded
    # [0, 100]) as the safety-gate signal rather than
    # ``failed_placement_rate_pct`` (unplaceable / placed; rate-like,
    # can exceed 100 on tight fleets).
    failed_placement_rate_pct = sched.fragmentation_loss_pct

    point = TrainingFrontierPoint(
        candidate=cand,
        predicted_goodput_per_dollar=sched.goodput_per_dollar,
        predicted_gpu_occupancy=occupancy,
        predicted_packing_density=occupancy,
        predicted_gpu_hours=sched.gpu_hours_used,
        predicted_completed_work=sched.goodput_gpu_seconds,
        predicted_queue_wait_p95_s=sched.queue_wait_p95,
        predicted_queue_wait_p99_s=sched.queue_wait_p99,
        predicted_job_completion_p95_s=None,
        predicted_job_completion_p99_s=None,
        predicted_starvation_rate_pct=starvation_rate_pct,
        predicted_fragmentation_block_rate_pct=failed_placement_rate_pct,
        predicted_gang_scheduling_failure_pct=None,  # not cleanly labelled
        predicted_backfill_success_rate_pct=backfill_success_rate_pct,
        predicted_retry_waste_gpu_hours=None,  # MIT scheduler-log has no
                                                # attempt history
        predicted_cost=sched.infra_cost,
        notes=tuple(filter(None, (
            f"source_policy={name}",
            "queue_wait_p95/p99: measured (Slurm submit/start)",
            "fragmentation: measured via failed_placement_rate_pct",
            "gang_failure: NOT cleanly labelled — gate disabled",
            "retry_waste: MIT scheduler-log lacks attempt history",
        ))),
    )
    status, vetoes = classify_training_frontier_point(
        point, safety_config, telemetry_confidence="medium")
    return TrainingFrontierPoint(
        candidate=point.candidate,
        predicted_goodput_per_dollar=point.predicted_goodput_per_dollar,
        predicted_gpu_occupancy=point.predicted_gpu_occupancy,
        predicted_packing_density=point.predicted_packing_density,
        predicted_gpu_hours=point.predicted_gpu_hours,
        predicted_completed_work=point.predicted_completed_work,
        predicted_queue_wait_p95_s=point.predicted_queue_wait_p95_s,
        predicted_queue_wait_p99_s=point.predicted_queue_wait_p99_s,
        predicted_starvation_rate_pct=point.predicted_starvation_rate_pct,
        predicted_fragmentation_block_rate_pct=
            point.predicted_fragmentation_block_rate_pct,
        predicted_gang_scheduling_failure_pct=
            point.predicted_gang_scheduling_failure_pct,
        predicted_backfill_success_rate_pct=
            point.predicted_backfill_success_rate_pct,
        predicted_retry_waste_gpu_hours=
            point.predicted_retry_waste_gpu_hours,
        predicted_cost=point.predicted_cost,
        safety_status=status, safety_vetoes=tuple(vetoes),
        notes=point.notes,
    )


def _verdict(selected, baseline):
    if selected is None or baseline is None or baseline == 0:
        return "INSUFFICIENT_DATA"
    delta = (selected - baseline) / baseline * 100.0
    if abs(delta) <= TIE_BAND_PCT:
        return "TIE"
    if delta > TIE_BAND_PCT:
        return "TRAINING_FRONTIER_WIN"
    return "TRAINING_FRONTIER_LOSS"


def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _frontier_md(rows, append):
    append("| policy | goodput/$ | occupancy | queue p99 (s) | "
           "starv % | frag block % | backfill % | safety |")
    append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        append(
            f"| `{r['policy']}` | "
            f"{_f(r['goodput_per_dollar'])} | "
            f"{_f(r['gpu_occupancy'], nd=4)} | "
            f"{_f(r['queue_wait_p99_s'])} | "
            f"{_f(r['starvation_rate_pct'])} | "
            f"{_f(r['fragmentation_block_rate_pct'])} | "
            f"{_f(r['backfill_success_rate_pct'])} | "
            f"**{r['safety_status']}** |")


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# MIT Supercloud — Training Safe Utilization Frontier Results\n")
    A("> **Simulator / public-trace benchmark. Directional only — NOT "
      "production savings** (`docs/RESULTS.md` §8). Validates Training "
      "Frontier v1 (`aurelius/frontier/training_*`) on the MIT "
      "Supercloud Dataset (Samsi et al., HPEC 2021). The serving Safe "
      "Utilization Frontier Controller, the robust energy engine, the "
      "committed Azure 2024 / Philly / Alibaba GPU benchmark artifacts "
      "are all **unchanged**. Real-cluster execution is **disabled by "
      "default**. The MIT Supercloud raw archive (~1 TB) is NOT "
      "committed — see `scripts/ingest_mit_supercloud.py` for download "
      "instructions.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/PUBLIC_TRACE_BACKTESTS.md`, "
      "`docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`, "
      "`docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md`.\n")

    src = payload["source"]
    A("## 1. Source + discovery\n")
    A(f"- **Source dir:** `{src['source_dir']}` "
      f"({'real raw' if src['is_raw'] else 'synthetic fixture'})")
    A(f"- **Repo:** {src['repo_url']}")
    A(f"- **Raw archive home:** {src['dcc_data_url']}")
    A(f"- **Paper:** {src['paper_url']}")
    A(f"- **n_jobs:** {src['n_jobs']:,}  "
      f"**n_gpu_jobs:** {src['n_gpu_jobs']:,}  "
      f"**n_labelled:** {src['n_labelled']:,}")
    A("")
    A("### Discovered files\n")
    A("| file | status | classification | kind |")
    A("|---|---|---|---|")
    for f in payload["discovery"]["files"]:
        A(f"| `{f['name']}` | {f['status']} | {f['classification']} | "
          f"{f['kind']} |")
    A("")
    A("### Join quality matrix\n")
    A("| join | kind | matched / right | confidence | notes |")
    A("|---|---|---|---|---|")
    for j in payload["join_quality"]["joins"]:
        A(f"| `{j['join_name']}` | `{j['join_kind']}` | "
          f"{j['matched_right']} / {j['right_total']} | "
          f"`{j['confidence']}` | {j['notes']} |")
    A("")

    A("## 2. Trace summary\n")
    ts = payload["trace_summary"]
    A(f"- queue wait p50/p95/p99 (s): "
      f"{_f(ts.get('queue_wait_s_p50'))} / "
      f"{_f(ts.get('queue_wait_s_p95'))} / "
      f"{_f(ts.get('queue_wait_s_p99'))}")
    A(f"- duration   p50/p95/p99 (s): "
      f"{_f(ts.get('duration_s_p50'))} / "
      f"{_f(ts.get('duration_s_p95'))} / "
      f"{_f(ts.get('duration_s_p99'))}")
    A(f"- gpu_count distribution: `{ts.get('gpu_count_distribution')}`")
    A(f"- gpu_type distribution:  `{ts.get('gpu_type_distribution')}`")
    A(f"- status distribution:    `{ts.get('status_distribution')}`")
    A(f"- workload labels:        "
      f"`{ts.get('workload_label_distribution')}`")
    A("")

    fl = payload["fleet"]
    A("## 3. Synthetic fleet (sized to trace demand)\n")
    A(f"- **n_nodes:** {fl['n_nodes']}  "
      f"**gpus_per_node:** {fl['gpus_per_node']}  "
      f"**total_gpus:** {fl['total_gpus']}")
    A(f"- **node_overhead_factor:** {fl['node_overhead_factor']:.2f} "
      "(fleet sized to peak job × overhead — MIT does not publish "
      "per-node capacity)\n")

    A("## 4. Training-frontier sweep (one row per policy)\n")
    _frontier_md(payload["frontier_rows"], A)
    A("")

    A("## 5. Controller verdict\n")
    cmp_ = payload["comparison"]
    A(f"- **current policy (baseline):** `{cmp_['current_policy']}` → "
      f"goodput/$ {_f(cmp_['current_goodput_per_dollar'])}")
    A(f"- **training_frontier_v1 selected:** "
      f"`{cmp_['selected_policy']}` → "
      f"goodput/$ {_f(cmp_['selected_goodput_per_dollar'])}")
    delta = cmp_['delta_vs_current_pct']
    delta_str = f"{delta:+.3f} %" if delta is not None else "—"
    A(f"- **Δ vs current:** {delta_str}")
    A(f"- **action:** `{cmp_['action']}`")
    A(f"- **verdict:** **`{cmp_['verdict']}`**")
    A(f"- **reason:** {cmp_['reason']}\n")

    A("## 6. Does MIT Supercloud reveal new Training Frontier alpha?\n")
    A(f"- **Verdict:** {payload['alpha_finding']['verdict']}")
    A(f"- **Evidence:** {payload['alpha_finding']['evidence']}\n")

    A("## 7. Metrics that were UNAVAILABLE and NOT INVENTED\n")
    for line in payload["unavailable_metrics"]:
        A(f"- {line}")
    A("")

    A("## 8. Honesty / scope\n")
    A("- The MIT Supercloud raw dataset (~1 TB compressed) is NOT "
      "committed to this repo. The full benchmark requires running "
      "the script with the published archive (see "
      "`scripts/ingest_mit_supercloud.py`).")
    A("- The synthetic fleet sizes nodes from the trace's peak demand "
      "and an overhead factor; MIT does NOT publish per-node "
      "capacity. Absolute KPIs are therefore relative across policies, "
      "not production-comparable.")
    A("- No new datasets ingested beyond MIT Supercloud.")
    A("- No serving-frontier code changed.")
    A("- No robust-energy-engine change.")
    A("- No ML training.")
    A("- No production-savings claim. Pilot telemetry is required to "
      "calibrate per-tenant safety thresholds.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default=None,
                   help="MIT Supercloud extracted directory; defaults "
                        "to data/external/mit_supercloud/raw when "
                        "present, else the synthetic fixture")
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--gpu-jobs-only", action="store_true", default=True)
    p.add_argument("--labelled-only", action="store_true")
    p.add_argument("--gpus-per-node", type=int, default=8,
                   help="synthetic fleet: GPUs per node (MIT does not "
                        "publish per-node capacity; default 8 mirrors "
                        "MIT Supercloud's V100 8-GPU node SKU)")
    p.add_argument("--node-overhead-factor", type=float, default=5.0,
                   help="synthetic fleet sizing safety factor over "
                        "the trace's average GPU demand (default 5×; "
                        "high enough that gang-packed multi-GPU jobs "
                        "have a credible chance to land on one node)")
    # Safety knobs (transparent; the defaults mirror the Training
    # Frontier v1 defaults, never tuned per-trace).
    p.add_argument("--max-queue-wait-p95-s", type=float, default=6 * 3600.0)
    p.add_argument("--max-queue-wait-p99-s", type=float, default=12 * 3600.0)
    p.add_argument("--max-starvation-rate-pct", type=float, default=5.0)
    p.add_argument("--max-fragmentation-block-rate-pct", type=float,
                   default=25.0)
    p.add_argument("--max-gang-scheduling-failure-pct",
                   type=lambda s: None if s.lower() == "none" else float(s),
                   default=None,
                   help="MIT scheduler-log does NOT cleanly label gang "
                        "failures; default is None (gate disabled)")
    p.add_argument("--min-completed-work-ratio", type=float, default=0.50)
    p.add_argument("--min-telemetry-confidence", default="low")
    p.add_argument("--current-policy", default="constraint_aware")
    args = p.parse_args(argv)

    source_dir = args.source_dir
    is_raw = False
    if source_dir is None:
        if os.path.isdir(DEFAULT_RAW) and any(
                os.path.exists(os.path.join(DEFAULT_RAW, f))
                for f in mit.SCHEDULER_LOG_FILES):
            source_dir = DEFAULT_RAW
            is_raw = True
        else:
            source_dir = DEFAULT_FIXTURE

    layers = mit.load_all_layers(
        source_dir, include_utilization=True, max_util_files=50,
        sample_size=args.sample_size,
        gpu_jobs_only=args.gpu_jobs_only,
        labelled_only=args.labelled_only)
    mit_jobs = layers["jobs"]
    summary = mit.summarize_jobs(mit_jobs)
    joins = mit.compute_join_quality(
        mit_jobs, labels_by_jobid=layers["labels_by_jobid"],
        gpu_samples=layers["gpu_samples"],
        node_samples=layers["node_samples"])

    # MIT → NormalizedGPUJob → scheduling backtest
    gpu_jobs = [mit.to_normalized_gpu_job(j) for j in mit_jobs]
    fleet = _synth_fleet(gpu_jobs, gpus_per_node=args.gpus_per_node,
                          node_overhead_factor=args.node_overhead_factor)
    bt_result = gs.run_backtest(gpu_jobs, fleet)
    n_scheduled = bt_result.n_scheduled

    safety = TrainingSafetyConfig(
        max_queue_wait_p95_s=args.max_queue_wait_p95_s,
        max_queue_wait_p99_s=args.max_queue_wait_p99_s,
        max_starvation_rate_pct=args.max_starvation_rate_pct,
        max_fragmentation_block_rate_pct=
            args.max_fragmentation_block_rate_pct,
        max_gang_scheduling_failure_pct=
            args.max_gang_scheduling_failure_pct,
        min_completed_work_ratio=args.min_completed_work_ratio,
        min_telemetry_confidence=args.min_telemetry_confidence,
    )
    frontier_points = [
        _point_from_sched_policy(name, sched, n_scheduled=n_scheduled,
                                  safety_config=safety)
        for name, sched in bt_result.policy_results.items()]

    current_candidate = PHILLY_POLICY_CANDIDATES.get(args.current_policy)
    decision = choose_training_frontier_target(
        frontier_points,
        current_candidate=current_candidate,
        config=TrainingControllerConfig(
            min_telemetry_confidence=args.min_telemetry_confidence),
        workload_id="mit_supercloud_training_workload",
        telemetry_confidence="medium")

    current_point = next(
        (p for p in frontier_points
         if p.candidate.source_policy == args.current_policy), None)
    cur_gpd = (current_point.predicted_goodput_per_dollar
                if current_point is not None else None)
    sel_gpd = (decision.selected_point.predicted_goodput_per_dollar
                if decision.selected_point is not None else None)
    verdict = _verdict(sel_gpd, cur_gpd)

    frontier_rows = []
    for p in frontier_points:
        frontier_rows.append({
            "policy": p.candidate.source_policy,
            "goodput_per_dollar": p.predicted_goodput_per_dollar,
            "gpu_occupancy": p.predicted_gpu_occupancy,
            "gpu_hours": p.predicted_gpu_hours,
            "queue_wait_p95_s": p.predicted_queue_wait_p95_s,
            "queue_wait_p99_s": p.predicted_queue_wait_p99_s,
            "starvation_rate_pct": p.predicted_starvation_rate_pct,
            "fragmentation_block_rate_pct":
                p.predicted_fragmentation_block_rate_pct,
            "backfill_success_rate_pct":
                p.predicted_backfill_success_rate_pct,
            "cost": p.predicted_cost,
            "safety_status": p.safety_status,
            "safety_vetoes": list(p.safety_vetoes),
        })

    # Alpha-finding heuristic: did the controller pick something other
    # than the current policy with a measurable KPI delta?
    safe_count = sum(1 for p in frontier_points if p.is_safe)
    if verdict == "TRAINING_FRONTIER_WIN":
        alpha = {"verdict": "YES — MIT Supercloud reveals new "
                              "Training Frontier alpha.",
                  "evidence": (f"controller picked "
                               f"`{decision.selected_candidate.source_policy}` "
                               f"over the current `{args.current_policy}` "
                               f"with Δ = "
                               f"{((sel_gpd - cur_gpd) / cur_gpd * 100):.3f}% "
                               "goodput/$")}
    elif verdict == "TRAINING_FRONTIER_LOSS":
        alpha = {"verdict": "NO — MIT Supercloud regression observed.",
                  "evidence": (f"selected policy underperformed "
                               f"`{args.current_policy}`; verdict "
                               "treated as evidence the current "
                               "baseline is well-tuned for this trace")}
    elif verdict == "TIE":
        alpha = {"verdict": ("NO — MIT Supercloud safely ties "
                              "constraint_aware. Result is consistent "
                              "with Philly + Alibaba GPU: "
                              "constraint_aware is already on or near "
                              "the safe training frontier."),
                  "evidence": (f"selected policy `"
                               f"{decision.selected_candidate.source_policy}` "
                               "matches the baseline KPI within "
                               f"±{TIE_BAND_PCT}% goodput/$; "
                               f"{safe_count} of "
                               f"{len(frontier_points)} candidates SAFE")}
    else:
        alpha = {"verdict": "INSUFFICIENT_DATA",
                  "evidence": "controller emitted INSUFFICIENT_TELEMETRY"}

    unavailable = [
        "Per-job gang-scheduling failure — MIT scheduler-log does not "
        "cleanly distinguish gang failures from other failure causes "
        "(gate disabled by default).",
        "Per-job retry/wasted-GPU-hours — MIT scheduler-log lacks "
        "attempt history (Philly has it; Alibaba GPU does not).",
        "Per-node capacity — MIT publishes `node-data.csv` "
        "utilization but not per-node capacity; fleet is sized "
        "synthetically to the trace's peak demand.",
        "Per-job utilization integration into KPI — GPU CSVs match "
        "job_id exactly, but the KPI uses requested GPU-seconds, not "
        "realized utilization, to stay comparable across traces.",
    ]

    payload = {
        "source": {
            "source_dir": os.path.relpath(source_dir, REPO_ROOT),
            "is_raw": is_raw,
            "repo_url": mit.REPO_URL,
            "dcc_data_url": mit.DCC_DATA_URL,
            "paper_url": mit.PAPER_URL,
            "n_jobs": summary["job_count"],
            "n_gpu_jobs": summary["gpu_job_count"],
            "n_labelled": summary["labelled_job_count"],
        },
        "discovery": layers["discovery"],
        "trace_summary": summary,
        "join_quality": joins,
        "fleet": {
            "n_nodes": len(fleet),
            "gpus_per_node": args.gpus_per_node,
            "total_gpus": sum(n.gpu_count for n in fleet),
            "node_overhead_factor": args.node_overhead_factor,
        },
        "frontier_rows": frontier_rows,
        "decision": decision.to_dict(),
        "comparison": {
            "current_policy": args.current_policy,
            "current_goodput_per_dollar": cur_gpd,
            "selected_policy": (decision.selected_candidate.source_policy
                                 if decision.selected_candidate is not None
                                 else None),
            "selected_goodput_per_dollar": sel_gpd,
            "delta_vs_current_pct": (
                (sel_gpd - cur_gpd) / cur_gpd * 100.0
                if (cur_gpd and sel_gpd is not None) else None),
            "verdict": verdict,
            "action": decision.action,
            "reason": decision.reason,
        },
        "alpha_finding": alpha,
        "unavailable_metrics": unavailable,
        "config": {
            "tie_band_pct": TIE_BAND_PCT,
            "max_queue_wait_p95_s": safety.max_queue_wait_p95_s,
            "max_queue_wait_p99_s": safety.max_queue_wait_p99_s,
            "max_starvation_rate_pct": safety.max_starvation_rate_pct,
            "max_fragmentation_block_rate_pct":
                safety.max_fragmentation_block_rate_pct,
            "max_gang_scheduling_failure_pct":
                safety.max_gang_scheduling_failure_pct,
            "min_completed_work_ratio": safety.min_completed_work_ratio,
            "min_telemetry_confidence": safety.min_telemetry_confidence,
            "real_execution_disabled_by_default": True,
            "execution_mode_default": "shadow",
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    _write_md(args.out_md, payload)

    delta_str = (f"{payload['comparison']['delta_vs_current_pct']:+.3f}%"
                 if payload['comparison']['delta_vs_current_pct'] is not None
                 else "—")
    print(f"[mit-training-frontier] source: {source_dir} "
          f"({'raw' if is_raw else 'fixture'})")
    print(f"[mit-training-frontier] jobs={summary['job_count']:,} "
          f"gpu_jobs={summary['gpu_job_count']:,} "
          f"labelled={summary['labelled_job_count']:,}")
    print(f"[mit-training-frontier] fleet: {len(fleet)} nodes × "
          f"{args.gpus_per_node} GPUs = "
          f"{sum(n.gpu_count for n in fleet)} total GPUs")
    sel_policy_str = (payload['comparison']['selected_policy']
                       if payload['comparison']['selected_policy']
                       is not None else "—")
    print(f"[mit-training-frontier] verdict: "
          f"{payload['comparison']['verdict']:25s} "
          f"action={payload['comparison']['action']:30s} "
          f"selected={sel_policy_str:24s} "
          f"Δ={delta_str}")
    print(f"[mit-training-frontier] alpha: "
          f"{alpha['verdict'][:80]}")
    print(f"[mit-training-frontier] JSON -> {args.out_json}")
    print(f"[mit-training-frontier] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
