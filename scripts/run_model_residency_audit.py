#!/usr/bin/env python3
"""Model-residency / cold-start readiness audit — measurement only.

Reads the ALREADY-COMMITTED public-trace backtest summaries (GenAI 2026 backtest
+ ablation, BurstGPT) and emits a machine-readable audit summary. It adds **no
optimizer behavior, no constants, no new datasets** — it only re-reads existing
results to quantify how much of the public-trace alpha depends on
model-affinity/prewarm and which datasets can *measure* vs only *simulate* it.

Output: data/external/alibaba_genai/processed/model_residency_audit_summary.json

Directional / simulator-trace result only — NOT production savings
(docs/RESULTS.md §8).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GENAI_BT = "data/external/alibaba_genai/processed/alibaba_genai_backtest_summary.json"
GENAI_ABL = "data/external/alibaba_genai/processed/alibaba_genai_ablation_summary.json"
BURST_BT = "data/external/burstgpt/processed/burstgpt_backtest_summary.json"
OUT = "data/external/alibaba_genai/processed/model_residency_audit_summary.json"

# Readiness ladder (most → least mature). The audit picks the honest rung.
READINESS_LADDER = [
    "SPEC_ONLY", "SIMULATOR_APPROXIMATION", "TRACE_BACKTESTED_APPROXIMATION",
    "SHADOW_PILOT_READY_READ_ONLY", "PRODUCTION_READY",
]


def _load(p):
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def _genai_with_without(abl) -> dict:
    """With-vs-without-affinity rows from the ablation (existing numbers)."""
    if not abl:
        return {}
    cfgs = abl.get("configs", {})

    def row(name):
        r = cfgs.get(name, {})
        return {
            "goodput_per_dollar": r.get("sla_safe_goodput_per_infra_dollar"),
            "sla_compliant_requests": r.get("sla_compliant_requests"),
            "e2e_latency_s_p99": r.get("e2e_latency_s_p99"),
            "mean_cold_start_s": r.get("mean_cold_start_s"),
            "replica_gpu_hours": r.get("replica_gpu_hours"),
        }

    ca = row("constraint_aware")
    ca_no = row("constraint_aware_no_affinity")
    avoided = None
    if ca["mean_cold_start_s"] is not None and ca_no["mean_cold_start_s"] is not None:
        avoided = round(ca_no["mean_cold_start_s"] - ca["mean_cold_start_s"], 3)
    return {
        "with_affinity_prewarm": ca,
        "without_affinity_prewarm": ca_no,
        "fifo": row("fifo"),
        "fifo_plus_affinity": row("fifo_plus_affinity"),
        "sla_aware_headline": row("sla_aware"),
        "mean_cold_start_avoided_s": avoided,
        "warm_pool_gpu_hours_note": (
            "NOT separately metered; affinity reduces required replica-hours "
            f"({ca['replica_gpu_hours']} vs {ca_no['replica_gpu_hours']}) rather "
            "than charging an explicit warm-pool line item"),
        "cold_start_p50_p95_p99_note": (
            "NOT per-request measured; only a calibrated pipeline-layer "
            "distribution (basemodel/LoRA/ControlNet load medians) exists"),
        "residency_hit_rate_note": (
            "NOT measured; the affinity model uses switch_rate≈distinct_models/N "
            "as a proxy (app↔infra layers are no_join in GenTD26)"),
    }


def build_audit() -> dict:
    genai_bt = _load(GENAI_BT)
    abl = _load(GENAI_ABL)
    burst = _load(BURST_BT)

    attribution = (abl or {}).get("attribution", {})
    shap = attribution.get("shapley_attribution_of_ca_vs_sla_gain", {})
    cold_cal = ((genai_bt or {}).get("backtest", {}).get("cold_start_calibration_s")
                or (abl or {}).get("cold_start_calibration_s") or {})

    # BurstGPT cache-affinity proxy effect (model-level proxy)
    burst_pol = (burst or {}).get("backtest", {}).get("policies", {})
    burst_fifo = burst_pol.get("fifo", {}).get("sla_safe_goodput_per_infra_dollar")
    burst_cache = burst_pol.get("cache_affinity_baseline", {}).get(
        "sla_safe_goodput_per_infra_dollar")
    burst_lift = None
    if burst_fifo and burst_cache:
        burst_lift = round((burst_cache - burst_fifo) / burst_fifo * 100.0, 3)

    dataset_coverage = [
        {"dataset": "alibaba_genai_2026", "model_id": "yes",
         "adapter_id_lora": "yes (num_lora count, not per-adapter id)",
         "real_e2e_latency": "yes (exec_time_seconds)",
         "cold_start_latency": "CALIBRATED distribution (pipeline layer)",
         "per_request_residency_hit": "NO (app↔infra no_join)",
         "can_measure_vs_simulate": "cold-start CALIBRATED + affinity SIMULATED; "
                                    "residency NOT measured per-request"},
        {"dataset": "burstgpt", "model_id": "yes",
         "adapter_id_lora": "no", "real_e2e_latency": "elapsed (not TTFT)",
         "cold_start_latency": "no",
         "per_request_residency_hit": "model-level proxy only (no session id)",
         "can_measure_vs_simulate": f"affinity PROXY only; economic effect "
                                    f"~{burst_lift}% (negligible, model-level key)"},
        {"dataset": "azure_llm", "model_id": "no", "adapter_id_lora": "no",
         "real_e2e_latency": "no", "cold_start_latency": "no",
         "per_request_residency_hit": "no",
         "can_measure_vs_simulate": "NOT APPLICABLE (no model/session/cache fields)"},
        {"dataset": "live_vllm_connector", "model_id": "service_id only",
         "adapter_id_lora": "no", "real_e2e_latency": "yes (TTFT/TPOT/e2e)",
         "cold_start_latency": "NO (not exposed by vLLM /metrics)",
         "per_request_residency_hit": "NO (no model_loaded_before_request)",
         "can_measure_vs_simulate": "prefix_cache_hit_rate + kv_cache_usage are "
                                    "REAL; model residency / cold-start are NOT"},
    ]

    return {
        "kpi": "sla_safe_goodput_per_infrastructure_dollar",
        "directional_only_not_production_savings": True,
        "genai_2026_affinity_dependence": {
            "constraint_aware_vs_sla_aware_gain_pct":
                attribution.get("constraint_aware_vs_sla_aware_gain_pct"),
            "affinity_prewarm_share_pct": shap.get("affinity_share_pct"),
            "anticipatory_sizing_share_pct": shap.get("sizing_share_pct"),
            "interaction_share_pct": shap.get("interaction_share_pct"),
            "with_vs_without": _genai_with_without(abl),
            "cold_start_calibration_s": cold_cal,
        },
        "burstgpt_affinity_proxy": {
            "fifo_goodput_per_dollar": burst_fifo,
            "cache_affinity_baseline_goodput_per_dollar": burst_cache,
            "cache_proxy_lift_pct": burst_lift,
            "note": "model-level cache_affinity_key proxy (no session id); "
                    "negligible economic effect, NOT a measured KV hit rate",
        },
        "azure_llm": "NOT_APPLICABLE (no model/session/cache fields)",
        "dataset_coverage": dataset_coverage,
        "readiness_verdict": "TRACE_BACKTESTED_APPROXIMATION",
        "readiness_ladder": READINESS_LADDER,
        "biggest_missing_pieces": [
            "real model_loaded_before_request / model-load timestamps from a "
            "serving engine (vLLM/Triton/SGLang) — entirely absent",
            "adapter_id/lora_id residency tracking — absent",
            "per-request residency hit-rate + cold-start p50/p95/p99 — only "
            "simulated/calibrated, never measured",
            "explicit warm-pool GPU-hour cost line item — implicit only",
            "explicit no-substitution gate + test — implicitly safe (no action "
            "substitutes a model) but not asserted",
            "shadow-mode residency recommendation logging + counterfactual — the "
            "energy/scheduling shadow runner exists but emits no residency log",
        ],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Model-residency readiness audit.")
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    audit = build_audit()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True)
    g = audit["genai_2026_affinity_dependence"]
    print(f"[audit] verdict: {audit['readiness_verdict']}")
    print(f"[audit] GenAI CA vs sla_aware: "
          f"{g['constraint_aware_vs_sla_aware_gain_pct']}% "
          f"(affinity/prewarm share {g['affinity_prewarm_share_pct']}%)")
    ww = g["with_vs_without"]
    print(f"[audit] with affinity:    gpd={ww['with_affinity_prewarm']['goodput_per_dollar']} "
          f"cold={ww['with_affinity_prewarm']['mean_cold_start_s']}s")
    print(f"[audit] without affinity: gpd={ww['without_affinity_prewarm']['goodput_per_dollar']} "
          f"cold={ww['without_affinity_prewarm']['mean_cold_start_s']}s")
    print(f"[audit] mean cold-start avoided: {ww['mean_cold_start_avoided_s']}s (modelled)")
    print(f"[audit] summary -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
