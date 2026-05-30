#!/usr/bin/env python3
"""Alibaba GPU fragmentation/packing backtest
(CANONICAL_TRACE_BACKTEST_ALIBABA_GPU_V2023_FRAGMENTATION_V1).

Loads the normalized Alibaba GPU trace (pods → jobs, nodes → fleet), runs the
executable packing baselines + constraint_aware in
``aurelius/traces/gpu_packing.py``, scores the canonical KPI (docs/RESULTS.md §1
— SLA-safe goodput per infrastructure dollar; goodput_unit =
``completed_gpu_job_work`` / token_equivalent), and writes:

  * docs/ALIBABA_GPU_BACKTEST_RESULTS.md
  * data/external/alibaba_gpu/processed/alibaba_gpu_backtest_summary.json

Static fractional bin-packing benchmark — directional simulator result, NOT
production savings. The headline baseline for fragmentation is the strongest
PACKING baseline (best_fit / FFD / greedy), never FIFO.

Examples
--------
    python scripts/run_alibaba_gpu_backtest.py            # raw if present, else fixture
    python scripts/run_alibaba_gpu_backtest.py --node-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import alibaba_gpu as az  # noqa: E402
from aurelius.traces import gpu_packing as gp  # noqa: E402
from aurelius.traces.schema import NormalizedGPUJob  # noqa: E402

RAW_POD = "data/external/alibaba_gpu/raw/openb_pod_list_default.csv"
RAW_NODE = "data/external/alibaba_gpu/raw/openb_node_list_gpu_node.csv"
FIX_POD = "tests/fixtures/alibaba_gpu/openb_pod_list_sample.csv"
FIX_NODE = "tests/fixtures/alibaba_gpu/openb_node_list_sample.csv"
SUMMARY_JSON = "data/external/alibaba_gpu/processed/alibaba_gpu_backtest_summary.json"
RESULTS_MD = "docs/ALIBABA_GPU_BACKTEST_RESULTS.md"
SWEEP_FRACTIONS = (1.0, 0.7, 0.5, 0.35)


def _subsample_nodes(nodes, fraction, seed):
    if fraction >= 1.0:
        return list(nodes)
    k = max(1, int(round(len(nodes) * fraction)))
    chosen = random.Random(seed).sample(list(nodes), k)
    return sorted(chosen, key=lambda n: n.node_id)


def _load(args):
    if args.processed:
        with open(args.processed) as fh:
            payload = json.load(fh)
        jobs = [NormalizedGPUJob.from_dict(d) for d in payload["jobs"]]
        nodes = az.load_nodes(args.node_csv or (RAW_NODE if os.path.exists(RAW_NODE)
                                                else FIX_NODE))
        return jobs, nodes, f"processed:{args.processed}"
    pod = args.pod_csv or (RAW_POD if os.path.exists(RAW_POD) else FIX_POD)
    node = args.node_csv or (RAW_NODE if os.path.exists(RAW_NODE) else FIX_NODE)
    jobs = az.load_jobs(pod, sample_size=args.sample_size, start_s=args.start_s,
                        duration_s=args.duration_s,
                        include_failed=args.include_failed, seed=args.seed)
    nodes = az.load_nodes(node)
    return jobs, nodes, f"pod:{pod} node:{node}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Alibaba GPU packing backtest.")
    p.add_argument("--processed", default=None)
    p.add_argument("--pod-csv", default=None)
    p.add_argument("--node-csv", default=None)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--include-failed", action="store_true")
    p.add_argument("--node-fraction", type=float, default=1.0,
                   help="use a deterministic subset of the fleet (contention)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-json", default=SUMMARY_JSON)
    p.add_argument("--results-md", default=RESULTS_MD)
    p.add_argument("--no-sweep", action="store_true")
    args = p.parse_args(argv)

    jobs, nodes, source = _load(args)
    if not jobs or not nodes:
        print("[backtest] no jobs/nodes", file=sys.stderr)
        return 4

    fleet = _subsample_nodes(nodes, args.node_fraction, args.seed)
    summary = az.summarize_jobs(jobs, fleet)
    result = gp.run_backtest(jobs, fleet)

    sweep = []
    if not args.no_sweep:
        for frac in SWEEP_FRACTIONS:
            sub = _subsample_nodes(nodes, frac, args.seed)
            res = gp.run_backtest(jobs, sub)
            sweep.append({
                "node_fraction": frac,
                "fleet_gpu_count": sum(n.gpu_count for n in sub),
                "goodput_per_dollar": {
                    pol: r.goodput_per_dollar for pol, r in res.policy_results.items()},
                "stranded_jobs": {
                    pol: r.stranded_jobs for pol, r in res.policy_results.items()},
                "ca_vs_headline_pct": round(res.outcome.margin_pct, 2),
                "headline": res.outcome.headline,
                "ca_outcome": res.outcome.outcome,
            })

    payload = {
        "source": source,
        "node_fraction": args.node_fraction,
        "filters": {"sample_size": args.sample_size, "start_s": args.start_s,
                    "duration_s": args.duration_s,
                    "include_failed": args.include_failed, "seed": args.seed},
        "trace_summary": summary,
        "backtest": result.to_summary_dict(),
        "load_sensitivity_sweep": sweep,
        "cross_trace_note": (
            "BurstGPT (bursty LLM serving) + Azure LLM (smooth serving) are "
            "serving-replay backtests; this is a GPU bin-packing/fragmentation "
            "backtest — a different workload class, not a like-for-like number."),
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    _write_markdown(args.results_md, source, summary, result, sweep)

    o = result.outcome
    print(f"[backtest] source   : {source}")
    print(f"[backtest] jobs={result.n_jobs} gpu_jobs={result.n_gpu_jobs} "
          f"nodes={result.n_nodes} fleet_gpus={result.fleet_gpu_count}")
    print(f"[backtest] headline : {o.headline} (packing baseline, NOT fifo)")
    print(f"[backtest] CA outcome: {o.outcome} (margin {o.margin_pct:+.2f}% vs "
          f"{o.headline}; beats_fifo={o.beats_fifo})")
    for pol, r in result.policy_results.items():
        v = r.goodput_per_dollar
        print(f"    {pol:<22} gpd={('%.2f' % v) if v is not None else 'n/a':>10} "
              f"placed={r.placed_jobs} stranded={r.stranded_jobs} "
              f"active_nodes={r.active_nodes} frag={r.fragmentation_score:.3f}")
    print(f"[backtest] summary  -> {args.summary_json}")
    print(f"[backtest] report   -> {args.results_md}")
    return 0


def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def _write_markdown(path, source, summary, result, sweep) -> None:
    s = summary
    o = result.outcome
    pr = result.policy_results
    L = []
    A = L.append
    A("# Alibaba GPU Backtest Results — "
      "CANONICAL_TRACE_BACKTEST_ALIBABA_GPU_V2023_FRAGMENTATION_V1")
    A("")
    A("> **Simulator benchmark result — directional only, NOT production "
      "savings.** Live customer-telemetry calibration is required before any "
      "external savings number (`docs/RESULTS.md` §8).")
    A(">")
    A("> Read `docs/RESULTS.md` and `docs/PUBLIC_TRACE_BACKTESTS.md` first.")
    A("")
    A("## Provenance")
    A("")
    A(f"- **Source:** `{source}`")
    A("- **Dataset:** Alibaba `cluster-trace-gpu-v2023` "
      "(https://github.com/alibaba/clusterdata) — pod list `openb_pod_list_default.csv` "
      "+ GPU node inventory `openb_node_list_gpu_node.csv`.")
    A("- Alibaba public data is a **public dataset, not customer telemetry**.")
    A("")
    A("## Available vs missing fields (honest)")
    A("")
    A("Pod schema: `name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,"
      "pod_phase,creation_time,deletion_time,scheduled_time`. "
      "Node schema: `sn,cpu_milli,memory_mib,gpu,model`.")
    A("")
    A(f"- **Missing (stated):** {', '.join(s.get('missing_fields', []))}.")
    A("- `gpu_milli` = thousandths of a GPU (sharing); `num_gpu` = whole GPUs. "
      "`gpu_spec` empty in the default pod list → `gpu_type=None`.")
    A("- **No GPU utilization time-series** in this dataset → "
      "`NormalizedGPUUtilizationSample` list is empty (0 samples).")
    A("- **No GPU-memory column** → `gpu_memory_gb=None`. **No per-pod node "
      "placement** in the default pod list → placement is what the backtest "
      "computes. **No deadline / user** columns.")
    A("")
    A("## Trace summary")
    A("")
    A(f"- Jobs: **{s['job_count']:,}** ({s['gpu_job_count']:,} GPU jobs, "
      f"{s['cpu_only_count']:,} CPU-only)  ·  failed: {s['failed_jobs']:,}")
    A(f"- Time range: {s['duration_s']:.0f}s ({s['duration_s']/86400.0:.1f} days)")
    A(f"- Status distribution: {s['status_distribution']}")
    A(f"- num_gpu distribution: {s['gpu_count_distribution']}")
    A(f"- Job duration s p50/p95/p99: {s['job_duration_s_p50']:.0f} / "
      f"{s['job_duration_s_p95']:.0f} / {s['job_duration_s_p99']:.0f}")
    qp = s.get("queue_wait_s_p95")
    A(f"- Queue wait s p50/p95/p99 (trace-observed): "
      f"{s.get('queue_wait_s_p50')} / {qp} / {s.get('queue_wait_s_p99')}")
    A(f"- Fleet: **{s.get('fleet_node_count', 0):,} GPU nodes**, "
      f"**{s.get('fleet_gpu_count', 0):,} GPUs**, by model "
      f"{s.get('fleet_gpu_by_model', {})}")
    A(f"- GPU demand / capacity ratio: **{s.get('gpu_demand_to_capacity_ratio')}**")
    A("- GPU utilization samples: **0** (no utilization series in v2023).")
    A("")
    A("## Primary KPI — SLA-safe goodput per infrastructure dollar")
    A("")
    A("Per `docs/RESULTS.md` §1. **goodput_unit = `completed_gpu_job_work` "
      "(token_equivalent = effective_GPU × duration)** — explicitly NOT inference "
      "output tokens. A job that cannot be placed is **stranded** (the "
      "SLA-violation analogue). Infra cost bills every **active** node (≥1 placed "
      "job) for the trace window at a documented per-GPU-type price, so "
      "fragmentation/spreading (more under-filled active nodes) costs more per "
      "unit work. Same packing physics, fleet, prices and window for all "
      "policies — only the placement decision differs.")
    A("")
    A(f"**Headline baseline = `{o.headline}` (a real PACKING baseline, NOT "
      f"fifo).** FIFO is the do-nothing sanity baseline only (`docs/RESULTS.md` §3).")
    A("")
    A("| policy | goodput/$ | placed | stranded | active nodes | GPU util % | "
      "fragmentation | infra $ |")
    A("|---|---|---|---|---|---|---|---|")
    for pol in pr:
        r = pr[pol]
        tag = ""
        if pol == "constraint_aware":
            tag = " **(CA)**"
        elif pol == o.headline:
            tag = " *(headline)*"
        A(f"| {pol}{tag} | {_fmt(r.goodput_per_dollar)} | {r.placed_jobs} | "
          f"{r.stranded_jobs} | {r.active_nodes} | {_fmt(r.gpu_utilization_pct,1)} | "
          f"{_fmt(r.fragmentation_score,4)} | {_fmt(r.infra_cost,0)} |")
    A("")
    A("## Packing baselines are EXECUTABLE (not analysis-only)")
    A("")
    A("`first_fit`, `best_fit`, `first_fit_decreasing` (FFD) and `greedy_packing` "
      "are run as real placement algorithms over the normalized trace/fleet — "
      "closing the prior analysis-only gap. `fifo` is a naive round-robin spread "
      "(no consolidation) and is the sanity baseline only.")
    A("")
    A("## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)")
    A("")
    A(f"- **Outcome:** `{o.outcome}`  ·  margin vs `{o.headline}`: "
      f"**{o.margin_pct:+.2f}%** on goodput/$")
    if o.safety_evidence:
        A(f"- **Safety evidence:** {', '.join(o.safety_evidence)}")
    if o.loss_reasons:
        A(f"- **Loss reasons:** {', '.join(o.loss_reasons)}")
    A(f"- **Sanity vs FIFO:** constraint_aware "
      f"{'beats' if o.beats_fifo else 'DOES NOT beat'} naive FIFO "
      f"({o.fifo_margin_pct:+.2f}%).")
    if o.notes:
        A(f"- Notes: {o.notes}")
    A("")
    A("### What improved / what did not")
    A("")
    ca = pr["constraint_aware"]
    head = pr.get(o.headline)
    if head is not None:
        A(f"- vs `{o.headline}` (strongest packing baseline): goodput/$ "
          f"{_fmt(ca.goodput_per_dollar)} vs {_fmt(head.goodput_per_dollar)} "
          f"({o.margin_pct:+.2f}%). constraint_aware adds **heterogeneous "
          f"GPU-type price-awareness** (route to the cheapest adequate GPU) + "
          f"big-job reservation on top of best-fit consolidation — infra $ "
          f"{_fmt(ca.infra_cost,0)} vs {_fmt(head.infra_cost,0)}.")
        A(f"- vs naive `fifo` spread: {o.fifo_margin_pct:+.2f}% — consolidation "
          f"avoids powering ~{pr['fifo'].active_nodes - ca.active_nodes} extra "
          f"under-filled nodes.")
        A("- Where it can lose / tie: when the fleet is homogeneous (no cheaper "
          "GPU type to exploit) or under-subscribed, best-fit already packs "
          "near-optimally and constraint_aware's edge shrinks to a tie — "
          "reported honestly.")
    A("")
    if sweep:
        A("## Fleet-contention sensitivity (deterministic node subsets)")
        A("")
        A("Replays the same job set onto progressively smaller fleets so "
          "fragmentation/stranding pressure rises transparently (no single "
          "cherry-picked fleet). `node_fraction=1.0` is the full fleet.")
        A("")
        A("| node × | fleet GPUs | fifo gpd | best_fit gpd | constraint_aware gpd "
          "| CA vs headline | CA stranded | fifo stranded |")
        A("|---|---|---|---|---|---|---|---|")
        for row in sweep:
            g = row["goodput_per_dollar"]
            st = row["stranded_jobs"]
            A(f"| {row['node_fraction']:g}× | {row['fleet_gpu_count']:,} | "
              f"{_fmt(g.get('fifo'),2)} | {_fmt(g.get('best_fit'),2)} | "
              f"{_fmt(g.get('constraint_aware'),2)} | "
              f"{row['ca_vs_headline_pct']:+.2f}% | "
              f"{st.get('constraint_aware')} | {st.get('fifo')} |")
        A("")
    A("## Honest limits")
    A("")
    A("- **Static** fractional bin-packing (openb-style); no temporal "
      "migration/churn (churn = 0). Goodput is `completed_gpu_job_work` "
      "(token_equivalent), NOT inference tokens. Durations are partly censored "
      "at the trace window.")
    A("- GPU-hour prices per model are documented public priors (±50%), "
      "identical across policies. Override before any external claim "
      "(`docs/RESULTS.md` §8).")
    A("- Alibaba public data is **not customer telemetry**. No GPU utilization, "
      "GPU-memory, deadline, or user columns exist in v2023.")
    A("- **Not production-real savings.** Directional simulator result only.")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
