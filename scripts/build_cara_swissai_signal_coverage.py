#!/usr/bin/env python3
"""Aggregate per-config summaries into the CARA + SwissAI signal coverage table.

Reads every ``data/external/hf/<safe>/<config>/processed/summary.json`` for the
two audited datasets, plus the per-config ``schema_profile.json`` +
``statistical_rollups.json``, and emits:

- ``data/external/hf_discovery/cara_swissai_signal_coverage.json`` —
  signal × (dataset, config) presence + sample-strength table required by
  the mission spec.
- a ``forecast_readiness`` block — per-forecast data-readiness score
  driven by which signals reach which sample strength.
- a ``missing_telemetry_gap_analysis`` block — every forecast not
  classified ``ready_for_forecast_leverage_audit`` is paired with the
  exact missing signals and the acquisition path (existing HF dataset /
  public non-HF trace / pilot telemetry only).
- a ``strongest_forecasting_dataset_matrix`` block — best (dataset,
  config) for each forecast plus rationale.

Nothing here writes any production-claim or runs a controller. The
output is research-class evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


DATASETS = [
    ("asdwb__cara_latency_prediction", "asdwb/cara_latency_prediction"),
    ("eth-easl__swissai-serving-trace", "eth-easl/swissai-serving-trace"),
]


# Signal -> definition. ``normalized_fields`` are the columns whose presence
# in ``schema_profile.json`` proves the signal exists; ``proxy_fields`` are
# weaker derivable evidence. ``usable_for`` is the canonical
# forecast/calibration use-case list.
SIGNALS = {
    "TTFT": {
        "normalized_fields": ["actual_ttft_s"],
        "proxy_fields": ["mean_ttft_ms", "p99_ttft_ms"],
        "usable_for": ["latency_forecast", "placement_forecast",
                       "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "TPOT": {
        "normalized_fields": ["actual_tpot_s"],
        "proxy_fields": ["mean_tpot_ms", "p99_tpot_ms"],
        "usable_for": ["latency_forecast", "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "e2e_latency": {
        "normalized_fields": ["actual_e2e_latency_s"],
        "proxy_fields": ["mean_e2el_ms", "p99_e2el_ms"],
        "usable_for": ["latency_forecast", "placement_forecast",
                       "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "queue_depth": {
        "normalized_fields": ["num_running", "num_waiting"],
        "proxy_fields": ["queue_depth", "pending_prefill_tokens"],
        "usable_for": ["queue_forecast", "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "queue_wait": {
        "normalized_fields": ["queue_wait_s"],
        "proxy_fields": [],
        "usable_for": ["queue_forecast", "dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "scheduling_state": {
        "normalized_fields": [
            "num_running", "num_active_decode_seqs", "pending_prefill_tokens",
            "pending_decode_tokens", "decode_ctx_p50", "decode_ctx_p95",
            "decode_ctx_max", "token_budget_per_iter", "prefill_chunk_size",
            "max_num_seqs", "num_preempted", "ema_decode_iter_ms",
        ],
        "proxy_fields": [],
        "usable_for": ["queue_forecast", "latency_forecast",
                       "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "cache_utilization": {
        "normalized_fields": ["kv_cache_utilization", "kv_free_blocks",
                              "kv_evictions_per_s"],
        "proxy_fields": [],
        "usable_for": ["cache_reuse_forecast", "dynamic_frontier_calibration"],
        "field_quality": "real",
    },
    "prefix_reuse": {
        "normalized_fields": ["bucket_count", "reused_bucket_count",
                              "bucket_ids_hash"],
        "proxy_fields": [],
        "usable_for": ["cache_reuse_forecast"],
        "field_quality": "real",
    },
    "reuse_percentage": {
        "normalized_fields": ["reuse_percentage"],
        "proxy_fields": [],
        "usable_for": ["cache_reuse_forecast"],
        "field_quality": "real",
    },
    "instance_type": {
        "normalized_fields": ["instance_type", "instance_id"],
        "proxy_fields": ["gpu", "gpu_type"],
        "usable_for": ["placement_forecast", "latency_forecast"],
        "field_quality": "real",
    },
    "request_arrival_timestamp": {
        "normalized_fields": ["prediction_timestamp_s", "created_at_iso"],
        "proxy_fields": ["timestamp_s"],
        "usable_for": ["queue_forecast", "workload_shape"],
        "field_quality": "real",
    },
    "request_completion_timestamp": {
        "normalized_fields": ["completion_timestamp_s", "finished_at_iso"],
        "proxy_fields": [],
        "usable_for": ["queue_forecast", "latency_forecast", "workload_shape"],
        "field_quality": "real",
    },
    "status": {
        "normalized_fields": ["status"],
        "proxy_fields": ["is_failed"],
        "usable_for": ["workload_shape"],
        "field_quality": "real",
    },
    "prompt_tokens": {
        "normalized_fields": ["num_prompt_tokens", "prompt_tokens"],
        "proxy_fields": [],
        "usable_for": ["latency_forecast", "workload_shape"],
        "field_quality": "real",
    },
    "output_tokens": {
        "normalized_fields": ["actual_output_tokens", "output_tokens"],
        "proxy_fields": ["num_predicted_output_tokens"],
        "usable_for": ["latency_forecast", "workload_shape"],
        "field_quality": "real",
    },
    "model_id": {
        "normalized_fields": ["model_id", "model"],
        "proxy_fields": [],
        "usable_for": ["placement_forecast", "workload_shape"],
        "field_quality": "real",
    },
    "replica_count": {
        "normalized_fields": ["replica_count"],
        "proxy_fields": [],
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "autoscaling_events": {
        "normalized_fields": [],
        "proxy_fields": [],
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "GPU_utilization": {
        "normalized_fields": ["gpu_utilization"],
        "proxy_fields": ["kv_cache_utilization"],  # weak proxy
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "GPU_memory": {
        "normalized_fields": ["gpu_memory_pct"],
        "proxy_fields": ["kv_free_blocks"],  # weak proxy
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "SLA_label": {
        "normalized_fields": ["sla_violation_rate_pct"],
        "proxy_fields": [],
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
    "timeout_label": {
        "normalized_fields": ["timeout_rate_pct"],
        "proxy_fields": ["num_preempted"],  # weak proxy
        "usable_for": ["dynamic_frontier_calibration"],
        "field_quality": "missing",
    },
}


def _load_summaries() -> list[dict]:
    """Walk data/external/hf/<safe>/.../processed/summary.json and return
    summaries + paths to schema_profile + statistical_rollups."""
    out = []
    for safe, ds_id in DATASETS:
        ds_root = REPO_ROOT / "data" / "external" / "hf" / safe
        if not ds_root.exists():
            continue
        for summary_path in ds_root.rglob("processed/summary.json"):
            with summary_path.open() as fh:
                summary = json.load(fh)
            profile_path = summary_path.parent / "schema_profile.json"
            rollups_path = summary_path.parent / "statistical_rollups.json"
            profile = None
            rollups = None
            if profile_path.exists():
                with profile_path.open() as fh:
                    profile = json.load(fh)
            if rollups_path.exists():
                with rollups_path.open() as fh:
                    rollups = json.load(fh)
            out.append({
                "dataset_id": summary.get("dataset_id"),
                "safe_dataset": safe,
                "config_name": summary.get("config_name"),
                "trace_type": summary.get("canonical_trace_type"),
                "trust_tier": summary.get("trust_tier"),
                "analysis_rows": int(summary.get("analysis_sample_rows") or 0),
                "fixture_rows": int(summary.get("fixture_sample_rows") or 0),
                "sample_strength": summary.get("statistical_sample_strength"),
                "promotion_state": (
                    summary.get("promotion_state")
                    or summary.get("promotion_tags") or "candidate"
                ),
                "normalized_schema": list(summary.get("normalized_schema") or []),
                "raw_schema": list(summary.get("raw_schema") or []),
                "missing_signals_summary": list(
                    summary.get("missing_signals") or []),
                "available_signals_summary": list(
                    summary.get("available_signals") or []),
                "rollups_insufficient_groups": list(
                    summary.get("rollups_insufficient_groups") or []),
                "summary_path": summary_path,
                "profile": profile,
                "rollups": rollups,
            })
    return out


def _signal_rows_for(profile, normalized_fields, raw_schema, normalized_schema):
    """Return (rows_available, availability_pct, field_names_present)."""
    if profile is None:
        return 0, 0.0, []
    inspected = int(profile.get("inspected_row_count") or 0)
    presence = profile.get("presence_rates") or {}
    missing_rates = profile.get("missing_rates") or {}
    raw_to_normalized = {}
    # Try direct normalized column names + inverse-map back through raw columns.
    field_present = []
    best_rows_available = 0
    best_availability = 0.0
    for nf in normalized_fields:
        # Either the normalized column appears in normalized_schema OR a raw
        # column whose mapping yields nf appears in raw_schema.
        if nf in normalized_schema:
            field_present.append(nf)
            best_rows_available = max(best_rows_available, inspected)
            best_availability = max(best_availability, 1.0)
        # Also check raw fields that present in profile.presence_rates.
    # Now check the raw profile too — raw column names appear directly in
    # the profile's presence_rates / missing_rates.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from audit_cara_swissai_telemetry import MAPPINGS  # noqa
    return best_rows_available, best_availability, field_present


def _row_count_for_signal(summary, signal_def) -> tuple[int, list[str]]:
    """Return (rows_with_signal_present, normalized_fields_present)."""
    profile = summary.get("profile") or {}
    normalized_schema = set(summary.get("normalized_schema") or [])
    inspected = int(profile.get("inspected_row_count") or 0)
    presence_rates = profile.get("presence_rates") or {}
    missing_rates = profile.get("missing_rates") or {}

    fields_present: list[str] = []
    rows_available = 0
    for nf in signal_def["normalized_fields"]:
        if nf in normalized_schema:
            fields_present.append(nf)
            # When normalized schema includes nf, the analysis sample
            # carries it for every row — use the analysis_rows count as
            # the proxy.
            rows_available = max(rows_available, summary["analysis_rows"])

    # If the normalized field never appears, fall back to inspecting raw
    # field presence rates (e.g. if reverse-mapped column is in raw schema).
    if rows_available == 0:
        proxy_present_in_raw = []
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from audit_cara_swissai_telemetry import MAPPINGS  # noqa
        ds_id = summary["dataset_id"]
        cfg = summary["config_name"]
        col_map = MAPPINGS.get((ds_id, cfg)) or {}
        for raw_col, m in col_map.items():
            if m.get("normalized_field") in signal_def["normalized_fields"]:
                if raw_col in presence_rates:
                    p = presence_rates[raw_col]
                    rows = int(round(p * inspected * (1 - missing_rates.get(raw_col, 0.0))))
                    rows_available = max(rows_available, rows)
                    proxy_present_in_raw.append(raw_col)
        fields_present.extend(proxy_present_in_raw)

    return rows_available, fields_present


def _sample_strength_for_count(n: int) -> str:
    if n >= 10_000:
        return "strong"
    if n >= 1_000:
        return "moderate"
    if n >= 100:
        return "weak"
    if n > 0:
        return "insufficient"
    return "insufficient"


def build_signal_coverage(summaries: list[dict]) -> list[dict]:
    out = []
    for sig_name, sig_def in SIGNALS.items():
        for s in summaries:
            rows_available, fields_present = _row_count_for_signal(s, sig_def)
            inspected = s["analysis_rows"]
            rows_missing = max(0, inspected - rows_available)
            availability_pct = (
                round(100.0 * rows_available / inspected, 3) if inspected else 0.0
            )
            # Field quality: real if rows_available > 0 and any field
            # matches, else missing (or proxy if only proxy_fields match).
            if rows_available > 0:
                fq = sig_def["field_quality"]
            else:
                fq = "missing"
            usable_for = sig_def["usable_for"] if rows_available > 0 else ["not_usable"]
            # Sample strength for THIS signal in THIS config.
            sample_strength = _sample_strength_for_count(rows_available)

            note = ""
            if rows_available == 0:
                note = (
                    f"Signal not present in {s['dataset_id']} / {s['config_name']} — "
                    f"normalized_fields={sig_def['normalized_fields']} not in "
                    "normalized_schema."
                )
            elif sample_strength != "strong":
                note = (
                    f"Subgroup sample {sample_strength}; statistical analysis "
                    "limited for p99."
                )
            out.append({
                "signal_name": sig_name,
                "dataset_id": s["dataset_id"],
                "config_name": s["config_name"],
                "trace_type": s["trace_type"],
                "trust_tier": s["trust_tier"],
                "analysis_rows": inspected,
                "rows_available": rows_available,
                "rows_missing": rows_missing,
                "availability_pct": availability_pct,
                "field_names": fields_present,
                "field_quality": fq,
                "usable_for": usable_for,
                "sample_strength": sample_strength,
                "notes": note,
            })
    return out


# ---------------------------------------------------------------------------
# Forecast readiness — per-forecast composite signal coverage
# ---------------------------------------------------------------------------


FORECAST_REQUIREMENTS = {
    "queue_wait_or_queue_depth_forecast": {
        "required_signals": ["queue_depth", "scheduling_state",
                             "request_arrival_timestamp",
                             "request_completion_timestamp"],
        "optional_signals": ["queue_wait", "instance_type"],
        "decision_frequency": "per-request",
        "horizon": "1-60s",
        "strongest_baseline": "dynamic_safe_frontier_estimator_v1 (Erlang-C tail risk)",
    },
    "ttft_forecast": {
        "required_signals": ["TTFT", "instance_type", "prompt_tokens",
                             "scheduling_state"],
        "optional_signals": ["queue_depth", "cache_utilization"],
        "decision_frequency": "per-request",
        "horizon": "request-time",
        "strongest_baseline": "constant_per_instance_type_p99",
    },
    "tpot_forecast": {
        "required_signals": ["TPOT", "instance_type", "scheduling_state"],
        "optional_signals": ["queue_depth", "cache_utilization", "output_tokens"],
        "decision_frequency": "per-request",
        "horizon": "request-time",
        "strongest_baseline": "constant_per_instance_type_tpot",
    },
    "e2e_latency_forecast": {
        "required_signals": ["e2e_latency", "instance_type", "prompt_tokens"],
        "optional_signals": ["scheduling_state", "queue_depth", "cache_utilization"],
        "decision_frequency": "per-request",
        "horizon": "request-time",
        "strongest_baseline": "constant_per_instance_type_p99",
    },
    "timeout_or_sla_violation_forecast": {
        "required_signals": ["SLA_label", "timeout_label", "e2e_latency",
                             "scheduling_state"],
        "optional_signals": ["queue_depth"],
        "decision_frequency": "per-request",
        "horizon": "1-60s",
        "strongest_baseline": "deterministic_erlang_c_sla_risk",
    },
    "cache_hit_or_prefix_reuse_forecast": {
        "required_signals": ["prefix_reuse", "reuse_percentage", "model_id"],
        "optional_signals": ["cache_utilization", "request_arrival_timestamp"],
        "decision_frequency": "per-request",
        "horizon": "request-time + 60-300s",
        "strongest_baseline": "residency_aware_routing",
    },
    "model_residency_or_cold_start_forecast": {
        "required_signals": ["cache_utilization", "model_id"],
        "optional_signals": ["instance_type", "scheduling_state"],
        "decision_frequency": "10s-1000s/h",
        "horizon": "300-1800s",
        "strongest_baseline": "static_load_profile_priors",
    },
    "gpu_placement_or_heterogeneous_latency_forecast": {
        "required_signals": ["e2e_latency", "TTFT", "instance_type",
                             "prompt_tokens"],
        "optional_signals": ["scheduling_state", "queue_depth"],
        "decision_frequency": "per-request",
        "horizon": "request-time",
        "strongest_baseline": "round_robin_placement",
    },
    "autoscaling_or_replica_need_forecast": {
        "required_signals": ["replica_count", "autoscaling_events",
                             "request_arrival_timestamp"],
        "optional_signals": ["queue_depth", "e2e_latency"],
        "decision_frequency": "per-minute",
        "horizon": "1-60m",
        "strongest_baseline": "static_replica_target",
    },
    "workload_arrival_forecast": {
        "required_signals": ["request_arrival_timestamp", "model_id"],
        "optional_signals": ["prompt_tokens", "output_tokens"],
        "decision_frequency": "per-minute",
        "horizon": "1-60m",
        "strongest_baseline": "diurnal_replay_prior",
    },
}


def _aggregate_signal_rows_across_configs(coverage, signal_name):
    """Return ``{(dataset_id, config_name): rows_available, sample_strength}``
    for the highest-strength config carrying the signal."""
    best = {}
    for row in coverage:
        if row["signal_name"] != signal_name:
            continue
        key = (row["dataset_id"], row["config_name"])
        if row["rows_available"] > best.get(key, {}).get("rows_available", -1):
            best[key] = {
                "rows_available": row["rows_available"],
                "sample_strength": row["sample_strength"],
                "field_quality": row["field_quality"],
            }
    return best


def build_forecast_readiness(coverage: list[dict]) -> list[dict]:
    out = []
    for forecast_name, req in FORECAST_REQUIREMENTS.items():
        present_signals = []
        missing_signals = []
        max_rows_per_signal = {}
        strength_per_signal = {}
        for sig in req["required_signals"]:
            agg = _aggregate_signal_rows_across_configs(coverage, sig)
            best = max(
                ((d["rows_available"], d["sample_strength"]) for d in agg.values()),
                default=(0, "insufficient"),
            )
            rows, strength = best
            max_rows_per_signal[sig] = rows
            strength_per_signal[sig] = strength
            if rows > 0:
                present_signals.append(sig)
            else:
                missing_signals.append(sig)

        present_optional = []
        for sig in req["optional_signals"]:
            agg = _aggregate_signal_rows_across_configs(coverage, sig)
            best_rows = max((d["rows_available"] for d in agg.values()), default=0)
            if best_rows > 0:
                present_optional.append(sig)

        # Overall data strength = weakest required signal.
        weakest_strength = "strong"
        order = {"insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}
        for sig in req["required_signals"]:
            s = strength_per_signal.get(sig, "insufficient")
            if order[s] < order[weakest_strength]:
                weakest_strength = s

        if not missing_signals and weakest_strength == "strong":
            readiness = "ready_for_forecast_leverage_audit"
            confidence = 5
        elif not missing_signals and weakest_strength == "moderate":
            readiness = "needs_more_ingest"
            confidence = 4
        elif missing_signals and any(
            strength_per_signal.get(s, "insufficient") in ("moderate", "strong")
            for s in req["required_signals"]
            if s not in missing_signals
        ):
            readiness = "priors_only"
            confidence = 3
        elif missing_signals:
            readiness = "not_supported"
            confidence = 2
        else:
            readiness = "needs_more_ingest"
            confidence = 3

        out.append({
            "forecast": forecast_name,
            "required_signals": req["required_signals"],
            "optional_signals": req["optional_signals"],
            "present_signals": present_signals,
            "missing_critical_signals": missing_signals,
            "present_optional_signals": present_optional,
            "decision_frequency": req["decision_frequency"],
            "horizon": req["horizon"],
            "strongest_baseline": req["strongest_baseline"],
            "rows_available_per_required_signal": max_rows_per_signal,
            "sample_strength_per_required_signal": strength_per_signal,
            "weakest_required_strength": weakest_strength,
            "recommended_readiness": readiness,
            "confidence_1_to_5": confidence,
        })
    return out


# ---------------------------------------------------------------------------
# Forecast leverage quantification — leverage = alpha × frequency × strength.
# ---------------------------------------------------------------------------


LEVERAGE = {
    "ttft_forecast": {
        "affected_modules": [
            "aurelius/frontier/dynamic_estimator.py",
            "aurelius/residency/decision.py",
            "aurelius/optimization/scheduler.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar", "p99_latency"],
        "expected_alpha_range": "high (1.5-3x p99 improvement plausible vs "
                                "constant-per-(model,GPU) baseline)",
        "strongest_existing_baseline": "constant_per_instance_type_p99",
        "can_plausibly_beat_existing": "yes (CARA train shows 9x p99 spread)",
        "frequency_score": 5,
    },
    "tpot_forecast": {
        "affected_modules": [
            "aurelius/frontier/dynamic_estimator.py",
            "aurelius/frontier/risk.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar", "tpot_p99"],
        "expected_alpha_range": "medium-high",
        "strongest_existing_baseline": "constant_per_instance_type_tpot",
        "can_plausibly_beat_existing": "yes (CARA train moderate-strong)",
        "frequency_score": 5,
    },
    "queue_wait_or_queue_depth_forecast": {
        "affected_modules": [
            "aurelius/frontier/risk.py",
            "aurelius/frontier/dynamic_controller.py",
        ],
        "affected_KPIs": ["queue_p95_ms", "queue_p99_ms",
                          "sla_safe_goodput_per_infra_dollar"],
        "expected_alpha_range": "high",
        "strongest_existing_baseline": "dynamic_safe_frontier_estimator_v1",
        "can_plausibly_beat_existing": "yes (CARA queue_details has per-request scheduler state)",
        "frequency_score": 5,
    },
    "e2e_latency_forecast": {
        "affected_modules": [
            "aurelius/frontier/dynamic_estimator.py",
            "aurelius/optimization/scheduler.py",
            "aurelius/residency/decision.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar", "p99_latency"],
        "expected_alpha_range": "high",
        "strongest_existing_baseline": "constant_per_instance_type_p99",
        "can_plausibly_beat_existing": "yes",
        "frequency_score": 5,
    },
    "cache_hit_or_prefix_reuse_forecast": {
        "affected_modules": [
            "aurelius/residency/decision.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar",
                          "cache_savings_applied"],
        "expected_alpha_range": "medium-high",
        "strongest_existing_baseline": "residency_aware_routing",
        "can_plausibly_beat_existing": "yes (SwissAI bucket_reuse + CARA KV)",
        "frequency_score": 4,
    },
    "gpu_placement_or_heterogeneous_latency_forecast": {
        "affected_modules": [
            "aurelius/optimization/scheduler.py",
            "aurelius/residency/decision.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar", "p99_latency"],
        "expected_alpha_range": "high",
        "strongest_existing_baseline": "round_robin_placement",
        "can_plausibly_beat_existing": "yes (trivial alpha at 9x p99 spread)",
        "frequency_score": 5,
    },
    "autoscaling_or_replica_need_forecast": {
        "affected_modules": ["aurelius/frontier/dynamic_controller.py"],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar",
                          "replica_count"],
        "expected_alpha_range": "medium",
        "strongest_existing_baseline": "static_replica_target",
        "can_plausibly_beat_existing": "unknown (missing replica_count labels)",
        "frequency_score": 3,
    },
    "model_residency_or_cold_start_forecast": {
        "affected_modules": ["aurelius/residency/decision.py"],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar"],
        "expected_alpha_range": "medium",
        "strongest_existing_baseline": "static_load_profile_priors",
        "can_plausibly_beat_existing": "unknown (no measured cold-start labels)",
        "frequency_score": 3,
    },
    "timeout_or_sla_violation_forecast": {
        "affected_modules": [
            "aurelius/frontier/risk.py",
        ],
        "affected_KPIs": ["sla_violation_rate_pct", "timeout_rate_pct"],
        "expected_alpha_range": "medium (deterministic baseline already strong)",
        "strongest_existing_baseline": "deterministic_erlang_c_sla_risk",
        "can_plausibly_beat_existing": "unknown (no measured SLA labels)",
        "frequency_score": 5,
    },
    "workload_arrival_forecast": {
        "affected_modules": [
            "aurelius/frontier/dynamic_controller.py",
            "aurelius/forecasting/baseline.py",
        ],
        "affected_KPIs": ["sla_safe_goodput_per_infra_dollar"],
        "expected_alpha_range": "low-medium",
        "strongest_existing_baseline": "diurnal_replay_prior",
        "can_plausibly_beat_existing": "partial (SwissAI arrivals add evidence)",
        "frequency_score": 3,
    },
}


_ALPHA_NUMERIC = {"low": 1, "low-medium": 2, "medium": 3, "medium-high": 4,
                  "high": 5}
_STRENGTH_NUMERIC = {"insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}


def _alpha_score(alpha_range: str) -> int:
    head = alpha_range.split(" ", 1)[0]
    return _ALPHA_NUMERIC.get(head, 3)


def build_leverage_table(readiness_table: list[dict]) -> list[dict]:
    readiness_by_forecast = {r["forecast"]: r for r in readiness_table}
    out = []
    for forecast, lev in LEVERAGE.items():
        r = readiness_by_forecast.get(forecast)
        if r is None:
            continue
        data_strength = r["weakest_required_strength"]
        alpha_s = _alpha_score(lev["expected_alpha_range"])
        freq_s = lev["frequency_score"]
        ds_s = _STRENGTH_NUMERIC.get(data_strength, 0)
        composite = alpha_s * freq_s * (ds_s + 1)
        out.append({
            "forecast": forecast,
            "decision_frequency": r["decision_frequency"],
            "data_strength": data_strength,
            "expected_alpha_range": lev["expected_alpha_range"],
            "affected_modules": lev["affected_modules"],
            "affected_KPIs": lev["affected_KPIs"],
            "strongest_existing_baseline": lev["strongest_existing_baseline"],
            "can_plausibly_beat_existing": lev["can_plausibly_beat_existing"],
            "frequency_score": freq_s,
            "alpha_score": alpha_s,
            "data_strength_score": ds_s,
            "leverage_score": composite,
            "build_priority": (
                "build_now" if r["recommended_readiness"]
                == "ready_for_forecast_leverage_audit"
                else (
                    "build_after_data_expansion"
                    if r["recommended_readiness"]
                    in ("needs_more_ingest", "priors_only")
                    else "blocked"
                )
            ),
        })
    out.sort(key=lambda e: -e["leverage_score"])
    for i, r in enumerate(out):
        r["leverage_rank"] = i + 1
    return out


# ---------------------------------------------------------------------------
# Missing-telemetry gap analysis
# ---------------------------------------------------------------------------


ACQUISITION_PATH = {
    "queue_wait": "pilot_telemetry_only (or vLLM /metrics export)",
    "replica_count": "pilot_telemetry_only",
    "autoscaling_events": "pilot_telemetry_only",
    "GPU_utilization": "DCGM_export_from_pilot",
    "GPU_memory": "DCGM_export_from_pilot",
    "SLA_label": "pilot_telemetry_only",
    "timeout_label": "pilot_telemetry_only",
    "TPOT": "CARA_train_already_in_corpus",
    "TTFT": "CARA_train_already_in_corpus",
    "e2e_latency": "CARA_train_already_in_corpus",
    "instance_type": "CARA_train_already_in_corpus",
    "prompt_tokens": "CARA_train + SwissAI_trace_already_in_corpus",
    "scheduling_state": "CARA_train_queue_details_already_in_corpus",
    "model_id": "SwissAI_trace_already_in_corpus",
}


def build_gap_analysis(readiness_table: list[dict]) -> list[dict]:
    out = []
    for r in readiness_table:
        if r["recommended_readiness"] == "ready_for_forecast_leverage_audit":
            continue
        gaps = []
        for sig in r["missing_critical_signals"]:
            gaps.append({
                "missing_signal": sig,
                "acquisition_path": ACQUISITION_PATH.get(
                    sig, "unknown / requires further audit"),
            })
        # Also flag required signals that exist but are weak.
        weak_signals = [
            sig for sig, strength in
            r["sample_strength_per_required_signal"].items()
            if strength in ("insufficient", "weak")
            and sig not in r["missing_critical_signals"]
        ]
        for sig in weak_signals:
            gaps.append({
                "weak_signal": sig,
                "acquisition_path": ACQUISITION_PATH.get(
                    sig, "extend bounded ingest"),
            })
        out.append({
            "forecast": r["forecast"],
            "recommended_readiness": r["recommended_readiness"],
            "gaps": gaps,
        })
    return out


# ---------------------------------------------------------------------------
# Strongest forecasting dataset matrix
# ---------------------------------------------------------------------------


DATASET_RATIONALE = {
    "ttft_forecast": {
        "best": "CARA train_flat",
        "why": "Per-request measured actual_ttft_s + scheduler state at "
               "decision time (76k rows in 80 MiB head).",
    },
    "tpot_forecast": {
        "best": "CARA train_flat",
        "why": "Per-request actual_tpot_s + EMA decode iter latency in same "
               "row as scheduler state.",
    },
    "queue_wait_or_queue_depth_forecast": {
        "best": "CARA train_queue_details",
        "why": "Nested schedule_state.* with running_requests[] + "
               "waiting_requests[] lists (38k rows).",
    },
    "e2e_latency_forecast": {
        "best": "CARA train_flat",
        "why": "Per-request actual_e2e_latency_s at 76k rows + per-(instance_type) "
               "9x p99 spread.",
    },
    "cache_hit_or_prefix_reuse_forecast": {
        "best": "SwissAI llama3_70b_bucket_reuse + qwen3_32b_bucket_reuse_analysis",
        "why": "Per-request reuse_percentage + bucket_ids hash; 147k + 153k rows.",
    },
    "gpu_placement_or_heterogeneous_latency_forecast": {
        "best": "CARA train_flat + AgentPerfBench trace_replay",
        "why": "5 instance_type subgroups in CARA + 14 GPU configurations in "
               "AgentPerfBench (priors).",
    },
    "autoscaling_or_replica_need_forecast": {
        "best": "Azure LLM 2024 (arrivals) — replica labels missing everywhere in HF corpus",
        "why": "Arrival forecasting feasible from Azure; replica-count labels "
               "block the end-to-end loop (pilot data required).",
    },
    "model_residency_or_cold_start_forecast": {
        "best": "CARA train_queue_details (kv_evictions_per_s as proxy)",
        "why": "Eviction rate is the closest signal; real cold-start labels "
               "require pilot data.",
    },
    "timeout_or_sla_violation_forecast": {
        "best": "CARA train_flat (num_preempted proxy) + SwissAI status=ERROR proxy",
        "why": "No measured SLA budget labels; proxies only.",
    },
    "workload_arrival_forecast": {
        "best": "SwissAI trace_analysis + CARA train_flat",
        "why": "ISO timestamps from SwissAI (~16M requests) + Unix timestamps "
               "from CARA (76k requests in 80 MiB head).",
    },
}


def build_dataset_matrix(readiness_table: list[dict],
                         leverage_table: list[dict]) -> list[dict]:
    readiness_by = {r["forecast"]: r for r in readiness_table}
    out = []
    for forecast, info in DATASET_RATIONALE.items():
        r = readiness_by.get(forecast)
        out.append({
            "forecast": forecast,
            "best_dataset": info["best"],
            "why": info["why"],
            "data_strength": (r["weakest_required_strength"] if r else "unknown"),
            "recommended_readiness": (
                r["recommended_readiness"] if r else "unknown"),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", default=str(REPO_ROOT / "data" / "external" / "hf_discovery"
                                / "cara_swissai_signal_coverage.json"),
    )
    args = p.parse_args(argv)

    summaries = _load_summaries()
    if not summaries:
        print("[coverage] no summaries found under data/external/hf/<safe>/...",
              file=sys.stderr)
        return 2

    coverage = build_signal_coverage(summaries)
    readiness = build_forecast_readiness(coverage)
    leverage = build_leverage_table(readiness)
    gap_analysis = build_gap_analysis(readiness)
    dataset_matrix = build_dataset_matrix(readiness, leverage)

    payload = {
        "doc_version": "cara_swissai_signal_coverage_v1",
        "stage": "analysis_tier_audit",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "trains_ml_models": False,
        "uses_oracle_as_headline": False,
        "summary_count": len(summaries),
        "configs_audited": [
            {
                "dataset_id": s["dataset_id"], "config_name": s["config_name"],
                "trace_type": s["trace_type"], "trust_tier": s["trust_tier"],
                "analysis_rows": s["analysis_rows"],
                "sample_strength": s["sample_strength"],
            }
            for s in summaries
        ],
        "signal_coverage": coverage,
        "forecast_readiness": readiness,
        "forecast_leverage_ranking": leverage,
        "missing_telemetry_gap_analysis": gap_analysis,
        "strongest_forecasting_dataset_matrix": dataset_matrix,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"[coverage] wrote {args.output}")
    print(f"[coverage] signal × config rows: {len(coverage)}")
    print(f"[coverage] forecast readiness rows: {len(readiness)}")
    print(f"[coverage] leverage ranking rows: {len(leverage)}")
    print(f"[coverage] gap analysis forecasts: {len(gap_analysis)}")
    print()
    print("Top leverage ranking:")
    for r in leverage[:5]:
        print(f"  #{r['leverage_rank']}  {r['forecast']:50s}  "
              f"score={r['leverage_score']}  data={r['data_strength']:10s}  "
              f"build={r['build_priority']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
