#!/usr/bin/env python3
"""Phase 3 -- Frontier Signal Strength audit.

Reads the Phase-1/2 processed outputs and the Phase-4 frontier ML results and
emits data/external/frontier_ingest_v1/signal_strength.json: per source, the
row counts, unique entities, coverage, target-label availability, the
measured/proxy/simulated breakdown, and honest suitable_for flags.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "data/external/frontier_signals"
FRONT = ROOT / "data/external/forecasting/economic_ml_alpha_frontier_v1"
OUT = ROOT / "data/external/frontier_ingest_v1/signal_strength.json"


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def main():
    mc = _load(SIG / "mooncake/processed/summary.json")
    mc_roll = _load(SIG / "mooncake/processed/statistical_rollups.json")
    hw = _load(SIG / "huawei_faas_2025/processed/summary.json")
    al = _load(SIG / "alibaba_gpu_v2025/processed/summary.json")
    al_roll = _load(SIG / "alibaba_gpu_v2025/processed/statistical_rollups.json")
    eval_ = _load(FRONT / "economic_alpha_eval.json")
    verdict = (eval_.get("answer") or {}).get("status")

    sources = {
        "mooncake": {
            "rows_normalized": mc.get("normalized_rows"),
            "unique_sessions": None,
            "unique_models": None,
            "unique_prefix_groups": mc.get("unique_prefix_groups"),
            "timestamp_coverage": 1.0,
            "subgroup_coverage": {"per_trace": mc_roll.get("per_trace", {})},
            "target_labels_available": {
                "cache_reuse_pct": "derived_proxy", "high_reuse": "derived_proxy",
                "cache_hit": "not_measured", "cache_hit_proxy": "derived"},
            "measured_proxy_simulated_breakdown": {
                "measured_anonymized": ["timestamp_ms", "input_length", "output_length", "hash_ids_count"],
                "derived_proxy": ["cache_reuse_pct", "high_reuse", "cache_hit_proxy"],
                "simulated": [], "not_measured": ["cache_hit", "model_id", "session_id"]},
            "suitable_for": {
                "cache_reuse_training": "yes_proxy_only",
                "cache_reuse_cross_dataset_validation": "limited -- derived proxy, NOT identical to "
                    "SwissAI measured reuse_percentage; transfer to SwissAI underperforms SwissAI's "
                    "own baseline",
                "cold_start_simulator_prior": "no",
                "cold_start_ml_training": "no",
                "autoscaling_proxy_training": "no",
                "queue_risk_training": "no",
                "diagnostic_only": "applies to cross-dataset claim"},
        },
        "huawei_faas_2025": {
            "rows_normalized": hw.get("normalized_rows"),
            "trigger_runtime_rows": hw.get("trigger_runtime_rows"),
            "unique_functions": hw.get("unique_functions"),
            "days_covered": hw.get("days_covered"),
            "timestamp_coverage": 1.0,
            "subgroup_coverage": {"functions": hw.get("unique_functions"), "clusters": "1-4"},
            "target_labels_available": {
                "cold_start_latency_s": "measured_faas", "pod_allocation_s": "measured_faas",
                "deploy_code_s": "measured_faas", "deploy_dependency_s": "measured_faas",
                "scheduling_s": "measured_faas", "gpu_model_load_s": "ABSENT (FaaS != GPU)"},
            "measured_proxy_simulated_breakdown": {
                "measured_faas": ["cold_start_latency_s", "pod_allocation_s", "deploy_code_s",
                                  "deploy_dependency_s", "scheduling_s"],
                "prior_proxy_for_gpu": ["all cold-start fields used as GPU prior"],
                "simulated": []},
            "suitable_for": {
                "cache_reuse_training": "no",
                "cache_reuse_cross_dataset_validation": "no",
                "cold_start_simulator_prior": "yes -- calibrates cold-start COST STRUCTURE (FaaS)",
                "cold_start_ml_training": "no -- FaaS, not GPU model-load; not promoted to GPU ML",
                "autoscaling_proxy_training": "weak (num_pods proxy not ingested in this bounded slice)",
                "queue_risk_training": "no",
                "diagnostic_only": "GPU cold-start remains blocked_by_missing_labels"},
        },
        "alibaba_gpu_v2025": {
            "rows_normalized": al.get("normalized_rows"),
            "unique_jobs_instances": al.get("normalized_rows"),
            "n_apps": al.get("n_apps"),
            "n_gpu_instances": al.get("n_gpu_instances"),
            "timestamp_coverage": al.get("scheduler_delay_coverage"),
            "subgroup_coverage": {"apps": al.get("n_apps"), "roles": al_roll.get("role_distribution", {})},
            "target_labels_available": {
                "scheduler_delay_s": "derived_proxy", "queue_wait_s": "ABSENT",
                "utilization": "ABSENT", "failure_or_timeout_state": "ABSENT",
                "autoscaling_event": "ABSENT (inferred from per-app create/delete only)"},
            "measured_proxy_simulated_breakdown": {
                "measured": ["creation_time_s", "scheduled_time_s", "deletion_time_s",
                             "gpu_count(allocation)"],
                "derived_proxy": ["scheduler_delay_s", "instance_lifetime_s", "autoscaling_count_proxy"],
                "absent": ["queue_wait_s", "utilization", "failure_or_timeout_state",
                           "measured_autoscaling_event"]},
            "suitable_for": {
                "cache_reuse_training": "no",
                "cache_reuse_cross_dataset_validation": "no",
                "cold_start_simulator_prior": "no",
                "cold_start_ml_training": "no",
                "autoscaling_proxy_training": "proxy_only -- instance-lifecycle, NOT measured serving "
                    "autoscaling events",
                "queue_risk_training": "proxy_only -- scheduler_delay tail, NOT per-request serving "
                    "queue-wait; weaker than existing AcmeTrace/CARA queue evidence",
                "diagnostic_only": "applies; no measured serving autoscaling/queue"},
        },
    }

    out = {
        "doc_version": "frontier_ingest_v1",
        "phase": "3_signal_strength",
        "cache_reuse_cross_dataset_verdict": verdict,
        "sources": sources,
        "measured_vs_proxy_summary": {
            "mooncake": "reuse label = derived proxy; timestamps/tokens/blocks = measured (anonymized)",
            "huawei_faas_2025": "cold-start cost = measured FaaS (CPU pod); GPU usage = prior/proxy only",
            "alibaba_gpu_v2025": "lifecycle/allocation = measured; queue/autoscaling = derived proxy; "
                                 "utilization/failure = absent",
        },
        "no_production_behavior_change": True, "production_claim": False,
        "public_data_is_not_pilot_telemetry": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    for s, v in sources.items():
        print(f"  {s}: rows={v.get('rows_normalized')}")


if __name__ == "__main__":
    main()
