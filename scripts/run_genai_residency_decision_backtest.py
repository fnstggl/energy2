#!/usr/bin/env python3
"""GenAI 2026 residency Decision-Engine backtest (per-request routing).

Runs the Model Residency Decision Engine (``aurelius/residency/decision.py``)
against residency-blind and naive-prewarm baselines on the Alibaba GenAI 2026
trace, in **simulator mode** (mutates only simulated state). Reports per-request
residency hit/miss, cold-start distribution, prewarm / route-to-resident /
eviction counts, warm-pool GPU-hours, SLA violations, and SLA-safe goodput/$.

Also references the EXISTING tick-based ablation summary (preserved, unchanged)
so the before/after is explicit. Directional simulator/backtest evidence —
**not production savings** (``docs/RESULTS.md`` §8).

Writes:
  * docs/MODEL_RESIDENCY_DECISION_ENGINE_RESULTS.md
  * data/external/alibaba_genai/processed/genai_residency_decision_summary.json

Examples
--------
    python scripts/run_genai_residency_decision_backtest.py            # fixture
    python scripts/run_genai_residency_decision_backtest.py --source-dir data/external/alibaba_genai/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.residency import backtest as rb  # noqa: E402
from aurelius.traces import alibaba_genai as ag  # noqa: E402

RAW_DIR = "data/external/alibaba_genai/raw"
FIX_DIR = "tests/fixtures/alibaba_genai_sample"
SUMMARY_JSON = "data/external/alibaba_genai/processed/genai_residency_decision_summary.json"
ABLATION_JSON = "data/external/alibaba_genai/processed/alibaba_genai_ablation_summary.json"
RESULTS_MD = "docs/MODEL_RESIDENCY_DECISION_ENGINE_RESULTS.md"


def _load_ablation_reference() -> dict:
    """Existing tick-based ablation numbers (the 'before' / full-trace economics)."""
    if not os.path.exists(ABLATION_JSON):
        return {}
    try:
        d = json.load(open(ABLATION_JSON))
    except (OSError, json.JSONDecodeError):
        return {}
    cfgs = d.get("configs", {})

    def g(name):
        return cfgs.get(name, {})
    return {
        "source": d.get("source"),
        "constraint_aware": g("constraint_aware"),
        "constraint_aware_no_affinity": g("constraint_aware_no_affinity"),
        "sla_aware": g("sla_aware"),
        "fifo": g("fifo"),
        "attribution": d.get("attribution", {}),
    }


def _fmt(v):
    return "—" if v is None else (f"{v}")


def _write_md(path, src, cold, result, ablation_ref):
    L = []
    a = L.append
    a("# Model Residency Decision Engine — GenAI 2026 Backtest\n")
    a("> **Directional simulator / backtest result — not production savings** "
      "(`docs/RESULTS.md` §8). The Model Residency Decision Engine is "
      "**recommendation-only in real/customer mode**; this backtest runs it in "
      "**simulator mode** (mutates only simulated state — no real cluster, "
      "router, or serving engine is touched). The engine **never** changes which "
      "model/adapter is requested; it only recommends placement / routing / "
      "prewarm / evict. Residency metrics are diagnostics — the primary KPI is "
      "unchanged: SLA-safe goodput per infrastructure dollar.\n")
    a(f"\n- **Source:** `{src}`")
    a(f"- **Requests:** {result.n_requests} · **simulated GPUs:** {result.n_gpus}")
    a(f"- **Cold-start calibration (s):** {cold}\n")

    a("## Per-request policy comparison (this engine vs baselines)\n")
    a("| policy | goodput/$ | model hit-rate | adapter hit-rate | cold starts | "
      "cold p50/p95/p99 (s) | route→resident | prewarm | evictions | warm-pool GPU-h | SLA viol | e2e p99 (s) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, r in result.policy_results.items():
        s = r.summary()
        cold_dist = (f"{_fmt(s['cold_start_p50_s'])}/{_fmt(s['cold_start_p95_s'])}/"
                     f"{_fmt(s['cold_start_p99_s'])}")
        a(f"| {p} | {_fmt(s['sla_safe_goodput_per_infra_dollar'])} | "
          f"{_fmt(s['model_residency_hit_rate'])} | {_fmt(s['adapter_residency_hit_rate'])} | "
          f"{s['cold_start_count']} | {cold_dist} | {s['route_to_resident_count']} | "
          f"{s['prewarm_count']} | {s['eviction_count']} | {s['warm_pool_gpu_hours']} | "
          f"{s['sla_violations']} | {_fmt(s['e2e_latency_s_p99'])} |")

    a("\n### Reading these numbers honestly\n")
    a("- This is a **small-sample** per-request replay (the committed fixture). "
      "On it, residency routing **cuts cold starts and raises the residency "
      "hit-rate** (residency-blind FIFO/least-queue vs affinity/engine); where "
      "the SLA is met by every policy, goodput/$ is similar because the shared "
      "fixed GPU pool gives the same cost denominator.")
    a("- The **economic** payoff of residency at scale (where cold starts cause "
      "SLA violations) is carried by the **full-trace tick-based ablation** "
      "below — preserved unchanged.")
    a("- `sla_aware_naive_prewarm` eliminates cold starts but pays a warm-pool "
      "cost for every distinct model held warm beyond pool capacity (zero here "
      "only because all models fit); at the trace's ~80-model scale this is "
      "expensive.\n")

    if ablation_ref:
        a("## Before/after — existing full-trace tick-based ablation (preserved)\n")
        a("> Source: committed `alibaba_genai_ablation_summary.json` "
          "(full 26,392-request trace). Unchanged by this PR.\n")
        a("| config (full trace) | goodput/$ | mean cold-start (s) | e2e p99 (s) | replica GPU-hrs |")
        a("|---|---|---|---|---|")
        for label, key in (("constraint_aware (with affinity/prewarm)", "constraint_aware"),
                           ("constraint_aware (no affinity)", "constraint_aware_no_affinity"),
                           ("sla_aware (headline)", "sla_aware"),
                           ("fifo", "fifo")):
            row = ablation_ref.get(key, {})
            a(f"| {label} | {_fmt(row.get('sla_safe_goodput_per_infra_dollar'))} | "
              f"{_fmt(row.get('mean_cold_start_s'))} | {_fmt(row.get('e2e_latency_s_p99'))} | "
              f"{_fmt(row.get('replica_gpu_hours'))} |")
        attr = ablation_ref.get("attribution", {})
        shap = attr.get("shapley_attribution_of_ca_vs_sla_gain", {})
        if shap:
            a(f"\n- Existing attribution: **affinity/prewarm ≈ "
              f"{shap.get('affinity_share_pct')}%** of the "
              f"+{attr.get('constraint_aware_vs_sla_aware_gain_pct')}% "
              "constraint_aware-vs-sla_aware gain; anticipatory sizing ≈ "
              f"{shap.get('sizing_share_pct')}%. The decision engine operationalises "
              "the affinity/prewarm lever as explicit per-request routing.\n")

    a("## Method / honesty\n")
    a("- The decision engine optimises SLA-safe goodput/$ subject to hard safety "
      "vetoes (memory headroom, SLA, thermal, topology, region, telemetry "
      "confidence). It never substitutes the requested model/adapter.")
    a("- Simulator mode mutates only simulated `ModelLocationState`. Real/customer "
      "mode is recommendation-only (`executable_in_real_cluster=False`).")
    a("- All policies share one fixed simulated GPU pool (same cost denominator) "
      "except `sla_aware_naive_prewarm`, which is charged for replicas held warm "
      "beyond pool capacity. Cold-start magnitudes are the trace's pipeline-layer "
      "calibration, not a per-request causal join.")
    a("- **No production-savings claim.** `docs/RESULTS.md` §8 production-claim "
      "gate is not met.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GenAI residency decision-engine backtest.")
    p.add_argument("--source-dir", default=None)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--summary-json", default=SUMMARY_JSON)
    p.add_argument("--results-md", default=RESULTS_MD)
    args = p.parse_args(argv)

    src = args.source_dir or (RAW_DIR if os.path.exists(
        os.path.join(RAW_DIR, ag.REQUEST_FILE)) else FIX_DIR)
    layers = ag.load_all_layers(src, request_kwargs=dict(include_failures=False))
    if not layers["requests"]:
        print("[residency-bt] no requests", file=sys.stderr)
        return 4
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    cold = ag.calibrate_cold_start(by_stage)

    result = rb.run_residency_backtest(
        layers["requests"], cold_start_s=cold, n_gpus=args.n_gpus)
    ablation_ref = _load_ablation_reference()

    payload = result.to_summary_dict()
    payload["source"] = f"raw:{src}" if src == RAW_DIR else f"dir:{src}"
    payload["ablation_reference_full_trace"] = ablation_ref
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    _write_md(args.results_md, src, cold, result, ablation_ref)

    print(f"[residency-bt] source: {src}  requests: {result.n_requests}")
    for pol, r in result.policy_results.items():
        s = r.summary()
        print(f"[residency-bt] {pol:24s} gpd={s['sla_safe_goodput_per_infra_dollar']} "
              f"hit={s['model_residency_hit_rate']} cold={s['cold_start_count']} "
              f"prewarm={s['prewarm_count']} route_res={s['route_to_resident_count']}")
    print(f"[residency-bt] summary -> {args.summary_json}")
    print(f"[residency-bt] report  -> {args.results_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
