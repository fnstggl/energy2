#!/usr/bin/env python3
"""Offline shadow eval: does the TTFT p50 prior change placement rankings?

Method (shadow-only, no production execution):

1. Fit ``TTFTShadowPrior`` on the CARA train_flat **train** holdout.
2. For each request in the CARA train_flat **test** holdout, build 5
   candidate ``ModelLocationState`` objects (one per CARA instance_type)
   sharing the request's measured queue_depth / kv_cache_utilization.
3. Score every (request, candidate) with the existing
   ``score_residency_candidate`` under two policies:

   - **baseline** — the default SafetyContext (static
     ``service_time_proxy_s = 2.0``).
   - **with_prior** — the same context with ``service_time_proxy_s``
     refined by ``refine_service_time_proxy_s`` (clamped by ``max(static,
     predicted_ttft_p50)``). The clamp ensures the prior can only widen
     the latency estimate.

4. Compare:

   - top-1 placement change rate
   - candidate ranking change rate (Spearman-style rank inequality)
   - projected goodput/$ delta (per-request and aggregate)
   - SLA-met delta
   - safety regressions (with_prior chooses a candidate that was
     infeasible under baseline)
   - subgroup-level regressions by chosen instance_type

5. Apply the binding promotion rule (mission spec):
   - < 2% projected goodput/$ improvement vs baseline -> ``diagnostic_only``
   - 2 - 5% -> ``promising_needs_validation``
   - >= 5% + no safety regression -> ``shadow_ready_for_integration_review``

Writes ``data/external/forecasting/placement_prior_audit/ttft_shadow_prior_eval.json``.

Never enables real execution. Never modifies default scheduler / scorer
behaviour. p95/p99 ML tails are NOT used here — only the p50 shadow prior
(per the mission's "do not use TTFT p95/p99 ML tails for control" rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    random_holdout,
)
from aurelius.forecasting.ttft_shadow_prior import (  # noqa: E402
    TTFTShadowPrior,
    _derive_gpu_type,
    _derive_model_size,
    refine_service_time_proxy_s,
    save_prior,
)
from aurelius.residency.decision import (  # noqa: E402
    SafetyContext,
    score_residency_candidate,
)
from aurelius.residency.models import (  # noqa: E402
    ModelLocationState,
    ModelResidencyRequest,
)

CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)
OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "placement_prior_audit" / "ttft_shadow_prior_eval.json"
)
PRIOR_TABLE_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "placement_prior_audit" / "ttft_shadow_prior_table.json"
)


# Five CARA candidate instance_types (model+gpu combinations).
CANDIDATE_INSTANCE_TYPES = (
    "qwen2.5-3b_a30", "qwen2.5-3b_p100", "qwen2.5-7b_a30",
    "qwen2.5-14b_v100", "qwen2.5-72b_a100",
)


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_request(row: dict) -> ModelResidencyRequest:
    return ModelResidencyRequest(
        request_id=str(row.get("request_id") or "synthetic"),
        timestamp=float(row.get("prediction_timestamp_s") or 0.0),
        workload_id="cara_train_flat",
        model_id=str(row.get("instance_type")).split("_", 1)[0]
        if row.get("instance_type") else "qwen2.5-unknown",
        priority_class="standard",
        prompt_tokens=int(row.get("num_prompt_tokens") or 0),
        output_tokens=int(row.get("num_predicted_output_tokens") or 0),
        latency_sla_ms=30_000.0,
        region="us-west",
    )


def _build_candidate(
    row: dict, instance_type: str,
) -> ModelLocationState:
    """Build a synthetic ModelLocationState whose location_key encodes the
    candidate instance_type. The queue_depth is the request's recorded
    num_running; the model is treated as resident (warm placement) on the
    candidate."""
    model_size = _derive_model_size(instance_type) or "unknown"
    gpu_type = _derive_gpu_type(instance_type) or "unknown"
    model_id = "qwen2.5-" + (model_size or "unknown")
    return ModelLocationState(
        region="us-west", node_id=f"node_{gpu_type}",
        gpu_id=instance_type, container_id="c0",
        loaded_model_ids=[model_id, str(row.get("instance_type"))
                          .split("_", 1)[0]],
        loaded_adapter_ids=[],
        gpu_memory_used=0.0, gpu_memory_total=None,  # avoid memory_pressure_cost
        gpu_utilization=None,
        queue_depth=int(row.get("num_running") or 0),
        estimated_queue_wait_s=None,
        thermal_risk=0.0, topology_score=1.0,
        telemetry_confidence="high", last_updated_s=0.0,
    )


def _ranking_change(baseline_scores, prior_scores, *, tie_tol: float = 1e-6) -> dict:
    """Return ranking-change metrics for one request.

    baseline_scores / prior_scores are lists of (instance_type, score)
    sorted ascending by score (lower = better). The score is
    predicted expected_latency_s. Tracks tie-breaking honestly: when the
    baseline assigns multiple candidates the same score (within
    ``tie_tol``), we record that the baseline is **indifferent** among
    them and the prior provides discriminating information whether or
    not the prior's top-1 happens to land on the same alphabetical
    first."""
    base_order = [it for it, _ in baseline_scores]
    prior_order = [it for it, _ in prior_scores]
    base_top_score = baseline_scores[0][1]
    base_top_tied_set = sorted(
        it for it, s in baseline_scores if abs(s - base_top_score) <= tie_tol
    )
    baseline_indifferent_at_top = len(base_top_tied_set) > 1
    prior_top_score = prior_scores[0][1]
    prior_top_tied_set = sorted(
        it for it, s in prior_scores if abs(s - prior_top_score) <= tie_tol
    )
    top1_changed = base_order[0] != prior_order[0]
    # Prior gives information ANY time it either changes top-1 OR breaks
    # a baseline tie (the baseline considered multiple candidates equal;
    # the prior strictly preferred a smaller subset).
    prior_breaks_tie = (
        baseline_indifferent_at_top and len(prior_top_tied_set) < len(base_top_tied_set)
    )
    rank_changed = any(b != p for b, p in zip(base_order, prior_order))
    # Latency-estimate change: any candidate whose latency moved by > 1e-4 s.
    base_dict = dict(baseline_scores)
    prior_dict = dict(prior_scores)
    latency_estimate_changed = any(
        abs(base_dict[it] - prior_dict[it]) > 1e-4 for it in base_dict
    )
    return {
        "top1_baseline": base_order[0],
        "top1_with_prior": prior_order[0],
        "top1_changed": top1_changed,
        "rank_changed": rank_changed,
        "baseline_indifferent_at_top": baseline_indifferent_at_top,
        "baseline_top_tied_set": base_top_tied_set,
        "prior_top_tied_set": prior_top_tied_set,
        "prior_breaks_tie": prior_breaks_tie,
        "latency_estimate_changed": latency_estimate_changed,
        "baseline_order": base_order,
        "prior_order": prior_order,
    }


def _score_request(row: dict, base_ctx: SafetyContext,
                   prior: "TTFTShadowPrior | None",
                   *, use_max_clamp: bool = True):
    request = _build_request(row)
    base_scores = []
    prior_scores = []
    base_gpd, prior_gpd = None, None
    base_top, prior_top = None, None
    refinement_logs = []
    for it in CANDIDATE_INSTANCE_TYPES:
        cand = _build_candidate(row, it)
        # Baseline scoring.
        base = score_residency_candidate(
            request=request, candidate_location=cand,
            load_profile=None, cost_config=base_ctx, safety_context=base_ctx,
        )
        base_scores.append((it, base.expected_latency_s
                            if base.expected_latency_s is not None
                            else float("inf")))
        if base.feasible and (base_top is None or
                              base.expected_latency_s
                              < base_top.expected_latency_s):
            base_top = base
        # Prior-refined scoring (apply_to_scorer=True to measure ranking
        # change WHAT-IF the integration were live; the decision artefact
        # itself is shadow-only).
        if prior is not None:
            refined_ctx, log = refine_service_time_proxy_s(
                base_ctx,
                model_size=_derive_model_size(it),
                gpu_type=_derive_gpu_type(it),
                prompt_tokens=row.get("num_prompt_tokens"),
                prior=prior, apply_to_scorer=True,
            )
            if not use_max_clamp:
                # Diagnostic variant: drop the max(static, predicted) clamp
                # so the predicted TTFT prior is used directly even when
                # it is BELOW the static service-time proxy. This is
                # *not* the binding shadow integration shape; it is an
                # offline what-if comparison.
                pred = log.get("predicted_ttft_p50_s")
                if pred is not None and not np.isnan(pred):
                    import dataclasses as _dc
                    refined_ctx = _dc.replace(
                        base_ctx, service_time_proxy_s=float(pred))
            refinement_logs.append({"candidate": it, **log})
        else:
            refined_ctx = base_ctx
        pr = score_residency_candidate(
            request=request, candidate_location=cand,
            load_profile=None, cost_config=refined_ctx,
            safety_context=refined_ctx,
        )
        prior_scores.append((it, pr.expected_latency_s
                             if pr.expected_latency_s is not None
                             else float("inf")))
        if pr.feasible and (prior_top is None or
                            pr.expected_latency_s
                            < prior_top.expected_latency_s):
            prior_top = pr

    base_scores.sort(key=lambda x: x[1])
    prior_scores.sort(key=lambda x: x[1])
    ranking = _ranking_change(base_scores, prior_scores)

    base_gpd = base_top.goodput_per_dollar if base_top else None
    prior_gpd = prior_top.goodput_per_dollar if prior_top else None
    base_sla = bool(base_top.sla_met) if base_top else False
    prior_sla = bool(prior_top.sla_met) if prior_top else False
    base_lat = base_top.expected_latency_s if base_top else None
    prior_lat = prior_top.expected_latency_s if prior_top else None
    safety_regression = bool(
        (base_top is not None and not base_top.feasible) and
        (prior_top is not None and prior_top.feasible is False)
    )
    # Stronger safety check: prior_top is infeasible while baseline_top was
    # feasible.
    safety_regression = bool(
        (base_top is not None and base_top.feasible) and
        (prior_top is None or not prior_top.feasible)
    )
    return {
        "request_id": request.request_id,
        "actual_ttft_s": row.get("actual_ttft_s"),
        "prompt_tokens": row.get("num_prompt_tokens"),
        "num_running": row.get("num_running"),
        "instance_actual": row.get("instance_type"),
        "ranking": ranking,
        "baseline_top_instance": base_top.location_key if base_top else None,
        "prior_top_instance": prior_top.location_key if prior_top else None,
        "baseline_expected_latency_s": base_lat,
        "prior_expected_latency_s": prior_lat,
        "baseline_goodput_per_dollar": base_gpd,
        "prior_goodput_per_dollar": prior_gpd,
        "baseline_sla_met": base_sla,
        "prior_sla_met": prior_sla,
        "safety_regression": safety_regression,
        "prior_refinement_logs": refinement_logs,
    }


def _classify_promotion(goodput_delta_pct: float,
                        safety_regression_count: int,
                        top1_change_rate: float) -> tuple[str, str]:
    if safety_regression_count > 0:
        return ("diagnostic_only",
                f"{safety_regression_count} safety regressions observed")
    if top1_change_rate == 0.0:
        return ("diagnostic_only",
                "no top-1 placement change; prior does not affect rankings")
    if goodput_delta_pct >= 5.0:
        return ("shadow_ready_for_integration_review",
                f"+{goodput_delta_pct:.2f}% goodput/$ with zero safety regressions")
    if goodput_delta_pct >= 2.0:
        return ("promising_needs_validation",
                f"+{goodput_delta_pct:.2f}% goodput/$ (2-5% band)")
    return ("diagnostic_only",
            f"+{goodput_delta_pct:.2f}% goodput/$ below 2% threshold")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=2000,
                   help="Cap on eval requests (default 2000 for fast shadow run)")
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--prior-table-path", default=str(PRIOR_TABLE_PATH))
    args = p.parse_args(argv)

    if not CARA_TRAIN_FLAT.exists():
        print(f"[ttft-prior-eval] CARA train_flat missing: {CARA_TRAIN_FLAT}",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(CARA_TRAIN_FLAT)
    print(f"[ttft-prior-eval] loaded {len(rows)} rows")

    # Random holdout: fit prior on train, evaluate on test.
    train_idx, test_idx = random_holdout(len(rows))
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx][: args.limit_rows]
    print(f"[ttft-prior-eval] train={len(train_rows)} test_eval={len(test_rows)}")

    prior = TTFTShadowPrior().fit_from_rows(train_rows)
    save_prior(prior, args.prior_table_path)
    print(f"[ttft-prior-eval] fit prior: {prior.fit_row_count} rows; "
          f"global_p50={prior.global_p50:.4f}s; "
          f"per-GPU medians={prior.by_gpu}")

    # Default SafetyContext — never mutated.
    base_ctx = SafetyContext(
        gpu_hour_price=3.0, service_time_proxy_s=2.0, seconds_per_token=0.0,
        max_thermal_risk=0.85,
    )

    per_request: list[dict] = []
    safety_regressions = 0
    top1_changes = 0
    ranking_changes = 0
    base_gpd_sum = 0.0
    prior_gpd_sum = 0.0
    base_sla_count = 0
    prior_sla_count = 0
    base_lat_sum = 0.0
    prior_lat_sum = 0.0
    by_instance_actual: dict = {}
    # Diagnostic variant (without the MAX clamp). Pure offline analysis;
    # NOT the binding shadow integration shape — the clamped version stays
    # the contract.
    diagnostic_top1_changes = 0
    diagnostic_safety_regressions = 0
    diagnostic_tie_breaks = 0
    diagnostic_latency_changes = 0
    diagnostic_ranking_changes = 0
    binding_tie_breaks = 0
    binding_latency_changes = 0
    for row in test_rows:
        res = _score_request(row, base_ctx, prior)
        res_diag = _score_request(row, base_ctx, prior, use_max_clamp=False)
        if res_diag["ranking"]["top1_changed"]:
            diagnostic_top1_changes += 1
        if res_diag["ranking"]["rank_changed"]:
            diagnostic_ranking_changes += 1
        if res_diag["ranking"]["prior_breaks_tie"]:
            diagnostic_tie_breaks += 1
        if res_diag["ranking"]["latency_estimate_changed"]:
            diagnostic_latency_changes += 1
        if res_diag["safety_regression"]:
            diagnostic_safety_regressions += 1
        if res["ranking"]["prior_breaks_tie"]:
            binding_tie_breaks += 1
        if res["ranking"]["latency_estimate_changed"]:
            binding_latency_changes += 1
        per_request.append(res)
        if res["safety_regression"]:
            safety_regressions += 1
        if res["ranking"]["top1_changed"]:
            top1_changes += 1
        if res["ranking"]["rank_changed"]:
            ranking_changes += 1
        if res["baseline_goodput_per_dollar"] is not None:
            base_gpd_sum += float(res["baseline_goodput_per_dollar"])
        if res["prior_goodput_per_dollar"] is not None:
            prior_gpd_sum += float(res["prior_goodput_per_dollar"])
        if res["baseline_sla_met"]:
            base_sla_count += 1
        if res["prior_sla_met"]:
            prior_sla_count += 1
        if res["baseline_expected_latency_s"] is not None:
            base_lat_sum += float(res["baseline_expected_latency_s"])
        if res["prior_expected_latency_s"] is not None:
            prior_lat_sum += float(res["prior_expected_latency_s"])
        ia = row.get("instance_type") or "unknown"
        sub = by_instance_actual.setdefault(ia, {"n": 0, "top1_changes": 0})
        sub["n"] += 1
        if res["ranking"]["top1_changed"]:
            sub["top1_changes"] += 1

    n = max(1, len(per_request))
    goodput_delta_pct = (
        100.0 * (prior_gpd_sum - base_gpd_sum) / base_gpd_sum
        if base_gpd_sum > 0 else 0.0
    )
    sla_delta_pct = 100.0 * (prior_sla_count - base_sla_count) / n
    latency_delta_pct = (
        100.0 * (prior_lat_sum - base_lat_sum) / base_lat_sum
        if base_lat_sum > 0 else 0.0
    )
    top1_change_rate = top1_changes / n
    ranking_change_rate = ranking_changes / n

    final_status, reason = _classify_promotion(
        goodput_delta_pct, safety_regressions, top1_change_rate,
    )

    payload = {
        "doc_version": "ttft_shadow_prior_eval_v1",
        "audit_only": True,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "production_claim": False,
        "shadow_only": True,
        "ttft_p95_p99_used_for_control": False,
        "evaluated_at_s": time.time(),
        "dataset": "asdwb/cara_latency_prediction",
        "config": "train_flat",
        "train_rows": len(train_rows),
        "test_eval_rows": n,
        "candidate_instance_types": list(CANDIDATE_INSTANCE_TYPES),
        "prior_metadata": {
            "model_version": prior.model_version,
            "fit_row_count": prior.fit_row_count,
            "global_p50_s": prior.global_p50,
            "by_gpu": prior.by_gpu,
        },
        "baseline_policy": "score_residency_candidate(default SafetyContext)",
        "prior_policy": "score_residency_candidate(refined service_time_proxy_s "
                        "= max(static, predicted_ttft_p50))",
        "metrics": {
            "top1_placement_change_rate": top1_change_rate,
            "ranking_change_rate": ranking_change_rate,
            "tie_break_rate": binding_tie_breaks / n,
            "latency_estimate_change_rate": binding_latency_changes / n,
            "projected_goodput_per_dollar_delta_pct": goodput_delta_pct,
            "projected_sla_met_delta_pct": sla_delta_pct,
            "projected_expected_latency_delta_pct": latency_delta_pct,
            "safety_regression_count": safety_regressions,
            "subgroup_top1_change_by_actual_instance": by_instance_actual,
        },
        "diagnostic_without_max_clamp": {
            "note": (
                "ANALYSIS ONLY — not the binding shadow shape. Drops the "
                "max(static, predicted) safety floor so the predicted TTFT "
                "p50 is used directly even when below the static service-time "
                "proxy. Reports what fraction of rankings would change under "
                "a more aggressive integration that ignores the safety floor."
            ),
            "top1_change_rate": diagnostic_top1_changes / n,
            "ranking_change_rate": diagnostic_ranking_changes / n,
            "tie_break_rate": diagnostic_tie_breaks / n,
            "latency_estimate_change_rate": diagnostic_latency_changes / n,
            "safety_regression_count": diagnostic_safety_regressions,
            "binding_integration_shape": "clamped (max(static, predicted))",
        },
        "promotion_rule": {
            "diagnostic_only_max_goodput_delta_pct": 2.0,
            "promising_band_pct": [2.0, 5.0],
            "shadow_ready_for_integration_review_min_pct": 5.0,
            "applies_only_if_safety_regression_count_is_zero": True,
        },
        "final_status": final_status,
        "reason": reason,
        "first_50_per_request": per_request[:50],
    }

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[ttft-prior-eval] wrote {args.out_path}")
    print(f"[ttft-prior-eval] top1_changes={top1_changes}/{n} "
          f"({top1_change_rate:.4f}); ranking_changes={ranking_changes}/{n}")
    print(f"[ttft-prior-eval] goodput/$ delta={goodput_delta_pct:+.2f}%; "
          f"sla delta={sla_delta_pct:+.2f}%; "
          f"latency delta={latency_delta_pct:+.2f}%; "
          f"safety_regressions={safety_regressions}")
    print(f"[ttft-prior-eval] final_status={final_status}  reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
