"""Compatibility-routed evaluation harness for the federated corpus.

This module reads ``canonical_corpus_registry.json`` and routes each promoted
dataset to a **bounded smoke evaluation** based on its canonical trace type
and the signals actually present.

Critical rules from the mission spec:

- Do not force every dataset through the same evaluator.
- Skip incompatible datasets with explicit reasons.
- Bounded evaluations only — no full backtests, no controller execution.
- Do not aggregate KPIs across incompatible trace types.
- Never use oracle as a headline.
- Never treat benchmark data as production telemetry.

The harness produces ``hf_corpus_evaluation_summary.json`` — a structured
log of: dataset, trace_type, evaluator used, baseline used, KPI, result,
whether the result is ``measured`` / ``proxy`` / ``synthetic`` /
``derived`` / ``prior_only``, whether the result informs the constraint-
aware engine or a frontier module (or both), and skip reasons.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import Optional

from .schemas import CANONICAL_TRACE_TYPES

logger = logging.getLogger(__name__)


# Mapping of trace_type -> (evaluator_id, baseline_id, kpi). These are the
# v1 SMOKE evaluators: small functions that compute one bounded statistic
# per dataset. They are NOT controllers. They do NOT execute the scheduler.
# A future PR may wire each evaluator into a real backtest fixture once
# the trace_type's promotion is widely validated.

EVALUATOR_REGISTRY = {
    "latency_benchmark_trace": {
        "evaluator_id": "latency_benchmark_prior_smoke_v1",
        "primary_baseline": "sla_aware_serving_frontier_static",
        "kpi": "p99_ttft_ms_ratio_vs_baseline_prior",
        "informs": ["performance_priors", "constraint_aware_engine"],
        "result_quality": "prior_only",
    },
    "kernel_profile_trace": {
        "evaluator_id": "kernel_profile_prior_smoke_v1",
        "primary_baseline": "static_kernel_cost_prior",
        "kpi": "kernel_duration_ms_p50_distribution",
        "informs": ["performance_priors"],
        "result_quality": "prior_only",
    },
    "cluster_scheduler_trace": {
        "evaluator_id": "cluster_scheduler_prior_smoke_v1",
        "primary_baseline": "sla_aware_packing",
        "kpi": "queue_wait_seconds_p95_distribution",
        "informs": ["constraint_aware_engine", "training_frontier"],
        "result_quality": "prior_only",
    },
    "cache_residency_trace": {
        "evaluator_id": "cache_residency_prior_smoke_v1",
        "primary_baseline": "residency_aware_routing",
        "kpi": "cache_hit_rate_distribution",
        "informs": ["cache_residency_evaluation", "constraint_aware_engine"],
        "result_quality": "prior_only",
    },
    "telemetry_trace": {
        "evaluator_id": "telemetry_calibration_smoke_v1",
        "primary_baseline": "dynamic_safe_frontier_estimator_v1",
        "kpi": "queue_wait_seconds_p99_distribution",
        "informs": ["dynamic_frontier", "constraint_aware_engine"],
        "result_quality": "prior_only",
    },
    "request_shape_trace": {
        "evaluator_id": "request_shape_prior_smoke_v1",
        "primary_baseline": "diurnal_arrival_replay_prior",
        "kpi": "prompt_tokens_distribution_summary",
        "informs": ["workload_modelling"],
        "result_quality": "prior_only",
    },
}


# Signal-requirement table. An evaluator only runs when AT LEAST ONE of
# these signals is present in the dataset's ``available_signals`` list.
# Missing required signals -> skip with explicit reason.
EVALUATOR_REQUIRED_SIGNALS = {
    "latency_benchmark_trace": ["ttft", "tpot", "e2e_latency"],
    "kernel_profile_trace": ["kernel_duration"],
    "cluster_scheduler_trace": ["queue_wait"],
    "cache_residency_trace": ["cache_hit", "prefix_cache", "cold_start"],
    "telemetry_trace": ["queue_wait", "queue_depth", "sla", "gpu_utilization"],
    "request_shape_trace": ["prompt_tokens", "output_tokens"],
}


def _percentile(values, p):
    if not values:
        return None
    vs = sorted(values)
    if p <= 0:
        return float(vs[0])
    if p >= 100:
        return float(vs[-1])
    idx = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
    return float(vs[idx])


def _load_sample(sample_path: str, max_rows: int = 2000) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(sample_path):
        return rows
    with open(sample_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
            if len(rows) >= max_rows:
                break
    return rows


def _summarize_distribution(values) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p90": float(_percentile(values, 90)),
        "p95": float(_percentile(values, 95)),
        "p99": float(_percentile(values, 99)),
    }


def _eval_latency_benchmark(rows: list[dict]) -> dict:
    fields = ("p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms", "mean_ttft_ms",
              "mean_tpot_ms", "mean_e2el_ms")
    out: dict = {}
    for f in fields:
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        if vals:
            out[f] = _summarize_distribution(vals)
    return out


def _eval_kernel_profile(rows: list[dict]) -> dict:
    vals = [r["duration_ms"] for r in rows if isinstance(r.get("duration_ms"), (int, float))]
    return {"duration_ms": _summarize_distribution(vals)}


def _eval_cluster_scheduler(rows: list[dict]) -> dict:
    qw = [r["queue_wait_s"] for r in rows if isinstance(r.get("queue_wait_s"), (int, float))]
    dur = [r["duration_s"] for r in rows if isinstance(r.get("duration_s"), (int, float))]
    return {
        "queue_wait_s": _summarize_distribution(qw),
        "duration_s": _summarize_distribution(dur),
    }


def _eval_cache_residency(rows: list[dict]) -> dict:
    hits = [bool(r["cache_hit"]) for r in rows if "cache_hit" in r]
    hit_rate = (sum(1 for h in hits if h) / len(hits)) if hits else None
    cold = [bool(r["cold_start"]) for r in rows if "cold_start" in r]
    cold_rate = (sum(1 for c in cold if c) / len(cold)) if cold else None
    return {
        "cache_hit_rate": hit_rate,
        "cold_start_rate": cold_rate,
        "samples": len(rows),
    }


def _eval_telemetry(rows: list[dict]) -> dict:
    qw = [r["queue_wait_s"] for r in rows if isinstance(r.get("queue_wait_s"), (int, float))]
    util = [r["gpu_utilization"] for r in rows
            if isinstance(r.get("gpu_utilization"), (int, float))]
    timeout = [r["timeout_rate_pct"] for r in rows
               if isinstance(r.get("timeout_rate_pct"), (int, float))]
    return {
        "queue_wait_s": _summarize_distribution(qw),
        "gpu_utilization": _summarize_distribution(util),
        "timeout_rate_pct": _summarize_distribution(timeout),
    }


def _eval_request_shape(rows: list[dict]) -> dict:
    pt = [r["prompt_tokens"] for r in rows
          if isinstance(r.get("prompt_tokens"), (int, float))]
    ot = [r["output_tokens"] for r in rows
          if isinstance(r.get("output_tokens"), (int, float))]
    return {
        "prompt_tokens": _summarize_distribution(pt),
        "output_tokens": _summarize_distribution(ot),
    }


_EVALUATOR_FUNCS = {
    "latency_benchmark_trace": _eval_latency_benchmark,
    "kernel_profile_trace": _eval_kernel_profile,
    "cluster_scheduler_trace": _eval_cluster_scheduler,
    "cache_residency_trace": _eval_cache_residency,
    "telemetry_trace": _eval_telemetry,
    "request_shape_trace": _eval_request_shape,
}


PROMOTED_STATES = frozenset({
    "promoted_for_backtest",
    "promoted_for_training_priors",
    "promoted_for_constraint_aware_evaluation",
    "promoted_for_dynamic_calibration",
    "promoted_for_performance_priors",
    "promoted_for_cache_residency_evaluation",
})


def select_eligible(registry: dict) -> list[dict]:
    out: list[dict] = []
    for entry in registry.get("entries") or []:
        if entry.get("promotion_state") in PROMOTED_STATES:
            out.append(entry)
    return out


def route_dataset(entry: dict) -> dict:
    """Return ``{evaluator_id, baseline, kpi, skip_reason}`` for one entry."""
    tt = entry.get("canonical_trace_type")
    if tt not in EVALUATOR_REGISTRY:
        return {
            "evaluator_id": None,
            "skip_reason": f"trace_type '{tt}' has no registered evaluator",
        }
    required = EVALUATOR_REQUIRED_SIGNALS.get(tt, [])
    available = set(entry.get("available_signals") or [])
    if required and not any(s in available for s in required):
        return {
            "evaluator_id": EVALUATOR_REGISTRY[tt]["evaluator_id"],
            "skip_reason": (
                f"trace_type '{tt}' requires one of {required}; "
                f"available={sorted(available)}"
            ),
        }
    reg = EVALUATOR_REGISTRY[tt]
    return {
        "evaluator_id": reg["evaluator_id"],
        "primary_baseline": reg["primary_baseline"],
        "kpi": reg["kpi"],
        "informs": list(reg["informs"]),
        "result_quality": reg["result_quality"],
        "skip_reason": None,
    }


def evaluate_one(
    entry: dict, repo_root: str, *, max_rows: int = 2000,
) -> dict:
    routing = route_dataset(entry)
    base = {
        "dataset_id": entry.get("dataset_id"),
        "canonical_trace_type": entry.get("canonical_trace_type"),
        "trust_tier": entry.get("trust_tier"),
        "promotion_state": entry.get("promotion_state"),
        "promotion_tags": list(entry.get("promotion_tags") or []),
        "evaluator_id": routing.get("evaluator_id"),
        "primary_baseline": routing.get("primary_baseline"),
        "kpi": routing.get("kpi"),
        "informs": routing.get("informs") or [],
        "result_quality": routing.get("result_quality"),
        "skip_reason": routing.get("skip_reason"),
        "result": None,
        "comparison_against_oracle_is_headline": False,
        "is_production_telemetry_substitute": False,
        "evaluated_at_s": time.time(),
    }
    if routing.get("skip_reason"):
        return base

    tt = entry.get("canonical_trace_type")
    func = _EVALUATOR_FUNCS.get(tt)
    if func is None:
        base["skip_reason"] = f"no evaluator function for {tt}"
        return base

    from .ingestion import safe_sample_paths

    sample_path = safe_sample_paths(
        repo_root, entry["dataset_id"], entry.get("config_name")
    )["sample_path"]
    rows = _load_sample(sample_path, max_rows=max_rows)
    if not rows:
        base["skip_reason"] = f"sample file empty or missing: {sample_path}"
        return base
    base["result"] = func(rows)
    base["sample_rows_evaluated"] = len(rows)
    return base


def run_corpus_evaluation(
    registry: dict,
    repo_root: str,
    *,
    max_rows: int = 2000,
) -> dict:
    eligible = select_eligible(registry)
    results: list[dict] = []
    by_trace_type: dict = {}
    for entry in eligible:
        r = evaluate_one(entry, repo_root, max_rows=max_rows)
        results.append(r)
        tt = r["canonical_trace_type"]
        by_trace_type.setdefault(tt, []).append(r["dataset_id"])
    return {
        "doc_version": "hf_corpus_evaluation_summary_v1",
        "stage": "federated_benchmark_corpus_evaluation_v1",
        "production_claim": False,
        "uses_oracle_as_headline": False,
        "treats_benchmark_as_production_telemetry": False,
        "aggregation_rule": (
            "Results are NOT aggregated across trace_types. Aggregation is "
            "valid only within the same trace_type, evaluator, and KPI."
        ),
        "evaluated_at_s": time.time(),
        "n_eligible": len(eligible),
        "n_evaluated": sum(1 for r in results if r.get("result") is not None),
        "n_skipped": sum(1 for r in results if r.get("skip_reason")),
        "datasets_by_trace_type": {k: sorted(v) for k, v in by_trace_type.items()},
        "per_dataset_results": results,
    }


def write_evaluation_summary(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
