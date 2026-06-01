#!/usr/bin/env python3
"""Cache / Prefix-Reuse Forecaster v1 — train + evaluate (shadow only).

Reads SwissAI bucket-reuse files (primary training source), CC-traces
flattened-request rows (training source if Phase 0 promoted it), LMCache
agentic rows (cross-dataset proxy), and PrefixBench fixtures (synthetic
generalisation check). Trains baselines + ML candidates, runs four
holdouts (random / time / by-model / by-session), computes predictive
metrics + a shadow economic proxy, and writes
``data/external/forecasting/cache_prefix_reuse_v1/summary.json``.

This is the PHASE C driver. No scheduler / residency controller is
modified. No production behaviour changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurelius.forecasting.cache_prefix_features import (  # noqa: E402
    HIGH_REUSE_THRESHOLD,
    LEAKAGE_TARGET_FIELDS,
    add_rolling_features,
    build_feature_matrix,
    build_feature_spec,
    derive_high_reuse,
    derive_intra_session_reuse_from_cc_traces,
    extract_reuse_percentage,
    holdout_by_group,
    holdout_by_session,
    random_holdout,
    time_holdout,
)
from aurelius.forecasting.cache_prefix_forecaster import (  # noqa: E402
    GlobalReuseRateBaseline,
    HistGradientBoostingReuseClassifier,
    HistGradientBoostingReuseRegressor,
    LogisticReuseClassifier,
    PerGroupReuseRateBaseline,
    PerSessionHistoryBaseline,
    RandomForestReuseClassifier,
    RecencyFrequencyBaseline,
    auprc,
    auroc,
    brier_score,
    calibration_error,
    classify_economic_status,
    mae,
    rmse,
)

logger = logging.getLogger(__name__)


SWISSAI_DATASET_DIR = REPO_ROOT / "data" / "external" / "hf" / (
    "eth-easl__swissai-serving-trace"
)
SWISSAI_BUCKET_REUSE_CONFIGS = [
    "qwen3_32b_bucket_reuse",
    "qwen380b_instruct_bucket_reuse",
    "qwen380b_thinking_bucket_reuse",
    "llama3_70b_bucket_reuse",
    "apertus_70b_bucket_reuse",
]

CC_TRACES_ANALYSIS = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_3000mib" / "processed" / "analysis_sample.jsonl"
CC_TRACES_NORMALIZED = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_3000mib" / "processed" / "normalized_sample.jsonl"
CC_TRACES_HEAD_NORMALIZED = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_head" / "processed" / "normalized_sample.jsonl"

LMCACHE_NORMALIZED = REPO_ROOT / "data" / "external" / "hf" / (
    "sammshen__lmcache-agentic-traces"
) / "train_shard4" / "processed" / "normalized_sample.jsonl"

PREFIXBENCH_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "hf" / (
    "jaytonde05__prefixbench__prefixbench_all_sample.jsonl"
)
PREFIXBENCH_ANALYSIS = REPO_ROOT / "data" / "external" / "hf" / (
    "jaytonde05__prefixbench"
) / "prefixbench_all" / "processed" / "analysis_sample.jsonl"

OUT_DIR = REPO_ROOT / "data" / "external" / "forecasting" / "cache_prefix_reuse_v1"
SUMMARY_PATH = OUT_DIR / "summary.json"
AUDIT_PATH = OUT_DIR / "data_readiness_audit.json"
PHASE0_PATH = OUT_DIR / "cc_traces_strength_expansion.json"


def _load_jsonl(path: Path, *, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
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


# ---------------------------------------------------------------------------
# Phase A — data readiness audit
# ---------------------------------------------------------------------------


def _load_swissai_bucket_reuse() -> list[dict]:
    out: list[dict] = []
    for cfg in SWISSAI_BUCKET_REUSE_CONFIGS:
        p = SWISSAI_DATASET_DIR / cfg / "processed" / "analysis_sample.jsonl"
        rows = _load_jsonl(p)
        for r in rows:
            if r.get("model_id") is None:
                r["model_id"] = cfg.rsplit("_bucket_reuse", 1)[0]
            r["__source"] = "swissai_bucket_reuse"
            r["__config"] = cfg
        out.extend(rows)
    return out


def _load_cc_traces() -> tuple[list[dict], str]:
    """Prefer the 3000 MiB analysis sample. Fall back to the 80 MiB
    committed normalized sample."""
    if CC_TRACES_ANALYSIS.exists():
        rows = _load_jsonl(CC_TRACES_ANALYSIS)
        if rows:
            for r in rows:
                r["__source"] = "cc_traces_3000mib"
            return rows, "3000_mib_expanded"
    if CC_TRACES_NORMALIZED.exists():
        rows = _load_jsonl(CC_TRACES_NORMALIZED)
        if rows:
            for r in rows:
                r["__source"] = "cc_traces_3000mib_normalized_only"
            return rows, "3000_mib_normalized_only"
    rows = _load_jsonl(CC_TRACES_HEAD_NORMALIZED)
    for r in rows:
        r["__source"] = "cc_traces_80mib_committed"
    return rows, "80_mib_committed"


def _load_lmcache() -> list[dict]:
    rows = _load_jsonl(LMCACHE_NORMALIZED)
    for r in rows:
        r["__source"] = "lmcache_agentic"
    return rows


def _load_prefixbench() -> list[dict]:
    rows = _load_jsonl(PREFIXBENCH_ANALYSIS) or _load_jsonl(PREFIXBENCH_FIXTURE)
    for r in rows:
        r["__source"] = "prefixbench"
        # Extract prefix_group from metadata_json.
        meta = r.get("metadata_json")
        if isinstance(meta, str):
            try:
                d = json.loads(meta)
                r["prefix_group"] = d.get("prefix_group")
                r["scenario"] = d.get("scenario")
                r["expected_shared_prefix_tokens"] = d.get(
                    "expected_shared_prefix_tokens")
                r["request_order"] = d.get("request_order")
            except json.JSONDecodeError:
                pass
    return rows


def _field_quality_for_targets(swissai_rows: list, cc_rows: list,
                                lmcache_rows: list, prefixbench_rows: list,
                                phase0: dict) -> dict:
    out: dict = {}
    out["reuse_percentage"] = {
        "swissai_bucket_reuse": "real" if swissai_rows else "missing",
        "cc_traces": "missing",  # CC-traces has no labelled reuse percentage
        "lmcache": "missing",
        "prefixbench": "missing",
    }
    out["high_reuse"] = {
        "swissai_bucket_reuse": "derived" if swissai_rows else "missing",
        "cc_traces": "missing",
        "lmcache": "missing",
        "prefixbench": "missing",
    }
    out["intra_session_reuse"] = {
        "swissai_bucket_reuse": "missing",
        "cc_traces": ("derived"
                      if phase0.get("decision") == "use_for_training"
                      else "proxy"),
        "lmcache": "proxy",   # session_id continuity only
        "prefixbench": "synthetic",
    }
    return out


def write_data_readiness_audit(*, swissai_rows: list, cc_rows: list,
                                cc_source: str, lmcache_rows: list,
                                prefixbench_rows: list, phase0: dict) -> dict:
    swissai_by_config: dict = {}
    for r in swissai_rows:
        cfg = r.get("__config") or "unknown"
        swissai_by_config[cfg] = swissai_by_config.get(cfg, 0) + 1
    audit = {
        "doc_version": "cache_prefix_reuse_data_readiness_audit_v1",
        "shadow_only": True,
        "production_claim": False,
        "datasets": {
            "swissai_bucket_reuse": {
                "rows": len(swissai_rows),
                "configs": swissai_by_config,
                "license_redistribution_status": "other_swissai_research_license",
                "available_cache_fields": [
                    "reuse_percentage", "reused_bucket_count", "bucket_count",
                    "bucket_ids_hash", "bucket_ids_sample",
                ],
                "missing_cache_fields": [
                    "actual_ttft_s", "actual_e2e_latency_s", "cache_hit",
                    "queue_state", "model_residency_state",
                ],
                "target_candidates": ["reuse_percentage", "high_reuse"],
                "feature_candidates": [
                    "model_id", "bucket_count", "bucket_ids_hash",
                    "rolling_per_hash_seen_count",
                    "rolling_per_model_reuse_pct",
                    "rolling_per_session_mean_block_count",
                    "hour_of_day", "input_token_bin (when available)",
                ],
                "field_quality": {
                    "reuse_percentage": "real (label only — leakage as feature)",
                    "reused_bucket_count": "real (label only — leakage as feature)",
                    "bucket_count": "real (decision-time observable)",
                    "bucket_ids_hash": "real (decision-time observable)",
                    "created_at_iso": "real",
                    "model_id": "real",
                },
                "suitable_for": ["training", "validation", "cross_model_holdout"],
                "raw_committed": False,
                "analysis_sample_committed": False,
                "fixture_committed": True,
            },
            "cc_traces": {
                "rows": len(cc_rows),
                "source": cc_source,
                "license_redistribution_status": "permissive_apache_2_0",
                "available_cache_fields": [
                    "block_hashes_hash", "block_hashes_count",
                    "block_size_tokens", "session_id", "turn_index",
                    "request_arrival_delta_s", "think_time_s", "api_time_s",
                    "ttft_s", "request_type",
                ],
                "missing_cache_fields": [
                    "reuse_percentage (no label)", "cache_hit (no label)",
                    "queue_state", "kv_evictions_per_s", "gpu_type",
                ],
                "target_candidates": ["intra_session_reuse (derived proxy)"],
                "feature_candidates": [
                    "session_id", "turn_index", "session_turns_so_far",
                    "input_tokens", "block_hashes_count (prior turn)",
                    "rolling_per_hash_seen_count", "request_type",
                    "model_id", "hour_of_day",
                ],
                "field_quality": {
                    "block_hashes_hash": "real",
                    "block_hashes_count": "real",
                    "session_id": "real",
                    "turn_index": "real",
                    "input_tokens": "real",
                    "ttft_s": "real (label only at decision-time — leakage)",
                    "api_time_s": "real (label only at decision-time — leakage)",
                    "intra_session_reuse": "derived",
                },
                "phase0_decision": phase0.get("decision"),
                "phase0_decision_reason": phase0.get("decision_reason"),
                "suitable_for": (
                    ["training", "validation", "diagnostic"]
                    if phase0.get("decision") == "use_for_training"
                    else ["diagnostic_only"]),
                "raw_committed": False,
                "analysis_sample_committed": False,
                "fixture_committed": True,
            },
            "lmcache": {
                "rows": len(lmcache_rows),
                "license_redistribution_status": "permissive_mit",
                "available_cache_fields": [
                    "session_id", "pre_gap_s", "model_id", "output_tokens",
                ],
                "missing_cache_fields": [
                    "reuse_percentage", "block_hashes",
                    "request_arrival_delta_s", "ttft_s",
                ],
                "target_candidates": [
                    "intra_session_reuse (very weak proxy — session continuity only)",
                ],
                "feature_candidates": [
                    "session_id", "session_turns_so_far", "model_id",
                    "pre_gap_s",
                ],
                "field_quality": {
                    "session_id": "real",
                    "pre_gap_s": "real",
                    "output_tokens": "real (label only at decision-time)",
                    "model_id": "real",
                },
                "suitable_for": ["cross_dataset_validation_proxy", "diagnostic"],
                "raw_committed": False,
                "analysis_sample_committed": False,
                "fixture_committed": True,
                "normalized_sample_committed": True,
            },
            "prefixbench": {
                "rows": len(prefixbench_rows),
                "license_redistribution_status": "unspecified_no_committed_sample",
                "available_cache_fields": [
                    "prefix_group", "scenario", "expected_shared_prefix_tokens",
                    "request_order",
                ],
                "missing_cache_fields": [
                    "real cache_hit", "queue_state", "ttft_s", "session_id",
                ],
                "target_candidates": [
                    "synthetic_prefix_reuse (prefix_group identity)",
                ],
                "feature_candidates": [
                    "prefix_group", "scenario", "request_order",
                    "prompt_text_len",
                ],
                "field_quality": {
                    "prefix_group": "real (synthetic-trace identifier)",
                    "scenario": "real",
                    "prompt_text_len": "real",
                    "max_tokens": "real",
                    "prompt_text": "synthetic",
                },
                "suitable_for": ["priors_only", "synthetic_generalization_check"],
                "raw_committed": False,
                "analysis_sample_committed": False,
                "fixture_committed": True,
            },
        },
        "strict_leakage_rules": {
            "reuse_percentage_cannot_be_feature_when_predicting_reuse": True,
            "future_requests_cannot_predict_current_request": True,
            "post_decision_fields_excluded": sorted(LEAKAGE_TARGET_FIELDS),
            "target_derived_bucket_overlap_excluded_unless_observable": True,
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit


# ---------------------------------------------------------------------------
# Phase B + C — train + evaluate
# ---------------------------------------------------------------------------


SUBGROUP_KEYS = ("model_id", "bucket_size_bin", "input_token_bin",
                 "request_type", "hour_of_day")


def _eval_classification(*, y_true: np.ndarray, y_score: np.ndarray,
                          y_baseline: np.ndarray) -> dict:
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "expected_calibration_error": calibration_error(y_true, y_score),
        "baseline_auroc": auroc(y_true, y_baseline),
        "baseline_brier": brier_score(y_true, y_baseline),
    }


def _subgroup_audit(*, y_true, y_score, y_baseline, groups: dict,
                    min_rows: int = 100,
                    regression_threshold_pct: float = -5.0) -> dict:
    audit: dict = {}
    has_reg = False
    for key in SUBGROUP_KEYS:
        g = groups.get(key)
        if g is None:
            continue
        per: dict = {}
        for val in sorted(set(g.tolist())):
            mask = g == val
            n = int(mask.sum())
            if n == 0:
                continue
            yt = y_true[mask]
            ys = y_score[mask]
            yb = y_baseline[mask]
            base_b = brier_score(yt, yb)
            ml_b = brier_score(yt, ys)
            imp = (100.0 * (base_b - ml_b) / base_b
                   if np.isfinite(base_b) and base_b > 0 else 0.0)
            status = "PASS"
            if n < min_rows:
                status = "INSUFFICIENT_SAMPLE"
            elif imp < regression_threshold_pct:
                status = "REGRESSION"
                has_reg = True
            per[str(val)] = {
                "row_count": n,
                "baseline_brier": base_b,
                "model_brier": ml_b,
                "improvement_pct": imp,
                "status": status,
            }
        audit[key] = per
    return {"by_subgroup": audit, "has_subgroup_regression": has_reg}


def _shadow_economic_proxy(*, y_true: np.ndarray, y_score: np.ndarray,
                            y_baseline: np.ndarray,
                            decision_threshold: float = 0.5) -> dict:
    """Translate cache-reuse predictions into a shadow economic proxy.

    We model two cache-aware routing decisions:

    (a) Prefill-savings proxy: a high-reuse request routed to a
        cache-warm replica avoids the cost of recomputing its prefill.
        Per-request, the expected savings under a perfect-information
        oracle equal ``reuse_percentage``. Our shadow proxy assumes the
        decision threshold ``0.5`` routes the request to a warm replica
        when the predicted probability ``y_score`` exceeds it. Under
        that decision, expected prefill savings ~ y_true * routed.

    (b) Migration-veto FP/FN proxy: vetoing migration when the model
        predicts high reuse but it isn't (FP) costs a missed-migration
        opportunity; allowing migration when it IS high-reuse (FN)
        loses cache state. We report both rates.

    Returns a dict with the proxy metrics. NOT a production claim — see
    ``docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`` §4.
    """
    routed = (y_score >= decision_threshold).astype(np.float64)
    routed_base = (y_baseline >= decision_threshold).astype(np.float64)

    # Prefill-savings proxy: y_true in [0, 1] for binary classification;
    # the expected savings of routing this request to a warm replica are
    # proportional to y_true (1 if high-reuse, 0 otherwise).
    savings_model = float((y_true * routed).sum())
    savings_baseline = float((y_true * routed_base).sum())
    # Normalise to a "% improvement" relative to the strongest realistic
    # baseline. Guard against zero-division.
    if savings_baseline > 0:
        pct = 100.0 * (savings_model - savings_baseline) / savings_baseline
    elif savings_model > 0:
        pct = float("inf")
    else:
        pct = 0.0

    # Migration-veto proxy:
    fp = float(((routed > 0.5) & (y_true <= 0.5)).sum())
    fn = float(((routed <= 0.5) & (y_true > 0.5)).sum())
    tp = float(((routed > 0.5) & (y_true > 0.5)).sum())
    tn = float(((routed <= 0.5) & (y_true <= 0.5)).sum())
    base_fp = float(((routed_base > 0.5) & (y_true <= 0.5)).sum())
    base_fn = float(((routed_base <= 0.5) & (y_true > 0.5)).sum())

    return {
        "decision_threshold": decision_threshold,
        "prefill_savings_proxy_model": savings_model,
        "prefill_savings_proxy_baseline": savings_baseline,
        "prefill_savings_improvement_pct_vs_baseline": pct,
        "migration_veto_fp_model": fp,
        "migration_veto_fn_model": fn,
        "migration_veto_tp_model": tp,
        "migration_veto_tn_model": tn,
        "migration_veto_fp_baseline": base_fp,
        "migration_veto_fn_baseline": base_fn,
        "n_test": int(len(y_true)),
        "result_quality": "shadow_proxy",
    }


def _eval_one_holdout(*, holdout_name: str, train_idx: np.ndarray,
                       test_idx: np.ndarray, X: np.ndarray,
                       y_high: np.ndarray, y_pct: np.ndarray,
                       group_keys: dict, rows: list, rolling_seen: np.ndarray,
                       rolling_session_history: np.ndarray) -> dict:
    Xtr, Xte = X[train_idx], X[test_idx]
    yh_tr, yh_te = y_high[train_idx], y_high[test_idx]
    yp_tr, yp_te = y_pct[train_idx], y_pct[test_idx]
    # Mask NaN labels
    mh = ~np.isnan(yh_tr)
    mp = ~np.isnan(yp_tr)
    if mh.sum() == 0 or yh_te.size == 0:
        return {"holdout": holdout_name, "skipped": "no_high_reuse_labels"}

    Xtr_m = Xtr[mh]
    yh_tr_m = yh_tr[mh]
    rolling_seen_tr_m = rolling_seen[train_idx][mh]
    rolling_session_history_tr_m = rolling_session_history[train_idx][mh]
    mh_te = ~np.isnan(yh_te)
    if mh_te.sum() == 0:
        return {"holdout": holdout_name, "skipped": "no_test_labels"}

    # Baselines
    global_b = GlobalReuseRateBaseline().fit(Xtr_m, yh_tr_m)
    per_model_b = PerGroupReuseRateBaseline().fit(
        Xtr_m, yh_tr_m,
        group_keys_train=group_keys["model_id"][train_idx][mh])
    session_b = PerSessionHistoryBaseline().fit(
        Xtr_m, yh_tr_m,
        session_keys_train=group_keys["session_id"][train_idx][mh])
    rf_b = RecencyFrequencyBaseline(min_seen=1).fit(
        Xtr_m, yh_tr_m, rolling_seen_count_train=rolling_seen_tr_m)

    p_global = global_b.predict(Xte)
    p_per_model = per_model_b.predict(
        Xte, group_keys_predict=group_keys["model_id"][test_idx])
    p_session = session_b.predict(
        Xte, session_keys_predict=group_keys["session_id"][test_idx],
        rolling_session_history=rolling_session_history[test_idx])
    p_rf = rf_b.predict(Xte, rolling_seen_count_predict=rolling_seen[test_idx])

    baselines = {
        "global_reuse_rate": p_global,
        "per_model_reuse_rate": p_per_model,
        "per_session_history": p_session,
        "recency_frequency_seen": p_rf,
    }

    # Strongest baseline = max-Brier among the baselines on training, but
    # we evaluate ALL on test for the report. Choose ``per_model`` by
    # default as our strongest realistic baseline (residency-aware proxy).
    strongest_name = "per_model_reuse_rate"

    # ML candidates
    ml_metrics: dict = {}
    try:
        logit = LogisticReuseClassifier().fit(Xtr_m, yh_tr_m)
        p_logit = logit.predict(Xte)
        ml_metrics["logistic"] = p_logit
    except Exception as e:  # pragma: no cover
        ml_metrics["logistic_error"] = str(e)
    try:
        hgb = HistGradientBoostingReuseClassifier().fit(Xtr_m, yh_tr_m)
        p_hgb = hgb.predict(Xte)
        ml_metrics["hist_gradient_boosting"] = p_hgb
    except Exception as e:  # pragma: no cover
        ml_metrics["hist_gradient_boosting_error"] = str(e)
    if Xtr_m.shape[0] <= 50_000:
        try:
            rf = RandomForestReuseClassifier().fit(Xtr_m, yh_tr_m)
            p_rf_ml = rf.predict(Xte)
            ml_metrics["random_forest"] = p_rf_ml
        except Exception as e:  # pragma: no cover
            ml_metrics["random_forest_error"] = str(e)

    test_groups = {k: group_keys[k][test_idx] for k in group_keys.keys()}
    yh_te_clean = np.where(np.isnan(yh_te), 0.0, yh_te)

    per_baseline_metrics: dict = {}
    for name, p in baselines.items():
        per_baseline_metrics[name] = _eval_classification(
            y_true=yh_te_clean, y_score=p, y_baseline=baselines[strongest_name])

    per_ml_metrics: dict = {}
    economic_results: dict = {}
    best_econ_pct = float("-inf")
    best_model_name = None
    for name, p in ml_metrics.items():
        if not isinstance(p, np.ndarray):
            continue
        per_ml_metrics[name] = _eval_classification(
            y_true=yh_te_clean, y_score=p,
            y_baseline=baselines[strongest_name])
        econ = _shadow_economic_proxy(
            y_true=yh_te_clean, y_score=p,
            y_baseline=baselines[strongest_name])
        economic_results[name] = econ
        if econ["prefill_savings_improvement_pct_vs_baseline"] > best_econ_pct:
            best_econ_pct = econ["prefill_savings_improvement_pct_vs_baseline"]
            best_model_name = name

    # Subgroup audit for the best ML model.
    if best_model_name is not None:
        sub = _subgroup_audit(
            y_true=yh_te_clean, y_score=ml_metrics[best_model_name],
            y_baseline=baselines[strongest_name], groups=test_groups)
    else:
        sub = {"by_subgroup": {}, "has_subgroup_regression": False}

    # Regression target: reuse_percentage.
    reg_metrics: dict = {}
    if mp.sum() >= 100:
        Xtr_p = Xtr[mp]
        yp_tr_p = yp_tr[mp]
        try:
            hgb_reg = HistGradientBoostingReuseRegressor().fit(Xtr_p, yp_tr_p)
            mp_te = ~np.isnan(yp_te)
            if mp_te.sum() > 0:
                p_reg = hgb_reg.predict(Xte[mp_te])
                reg_metrics = {
                    "hgb_regressor_mae": mae(yp_te[mp_te], p_reg),
                    "hgb_regressor_rmse": rmse(yp_te[mp_te], p_reg),
                    "baseline_global_mean_mae": mae(
                        yp_te[mp_te],
                        np.full_like(p_reg, float(np.nanmean(yp_tr_p)))),
                }
        except Exception as e:  # pragma: no cover
            reg_metrics = {"error": str(e)}

    return {
        "holdout": holdout_name,
        "n_train": int(train_idx.size),
        "n_train_high_reuse_labelled": int(mh.sum()),
        "n_test": int(test_idx.size),
        "n_test_high_reuse_labelled": int(mh_te.sum()),
        "strongest_baseline": strongest_name,
        "baseline_metrics": per_baseline_metrics,
        "ml_metrics": per_ml_metrics,
        "economic_proxy_by_model": economic_results,
        "best_ml_model": best_model_name,
        "best_economic_improvement_pct": (
            None if best_model_name is None else best_econ_pct),
        "regression_metrics": reg_metrics,
        "subgroup_audit": sub,
    }


def _run_swissai(swissai_rows: list) -> dict:
    rows = add_rolling_features(swissai_rows, source="swissai_bucket_reuse")
    spec = build_feature_spec(rows)
    X, names, group_keys = build_feature_matrix(rows, spec)
    y_pct = extract_reuse_percentage(rows)
    y_high = derive_high_reuse(y_pct, threshold=HIGH_REUSE_THRESHOLD)

    timestamps = np.array(
        [r.get("__decision_timestamp_s") if r.get("__decision_timestamp_s") is not None
         else float("nan") for r in rows], dtype=np.float64)
    rolling_seen = np.array(
        [r.get("rolling_per_hash_seen_count", 0.0) for r in rows],
        dtype=np.float64)
    rolling_session_history = np.array(
        [r.get("rolling_per_model_reuse_pct", float("nan")) for r in rows],
        dtype=np.float64)

    holdouts: dict = {}
    r_tr, r_te = random_holdout(X.shape[0])
    holdouts["random_holdout"] = (r_tr, r_te)
    if np.isfinite(timestamps).sum() >= 0.95 * len(rows):
        t_tr, t_te = time_holdout(timestamps)
        holdouts["time_holdout"] = (t_tr, t_te)
    uniq_models = sorted(set(group_keys["model_id"].tolist()))
    if len(uniq_models) >= 3:
        hold_model = uniq_models[-1]
        m_tr, m_te = holdout_by_group(group_keys["model_id"], (hold_model,))
        holdouts[f"holdout_by_model_{hold_model}"] = (m_tr, m_te)
    # session-holdout — SwissAI bucket_reuse doesn't have session_id but
    # bucket_ids_hash can stand in.
    uniq_hashes = group_keys.get("session_id")
    if uniq_hashes is not None and any(x is not None for x in uniq_hashes.tolist()):
        s_tr, s_te = holdout_by_session(uniq_hashes)
        if s_te.size > 0:
            holdouts["holdout_by_session"] = (s_tr, s_te)

    per_holdout = []
    for name, (tr, te) in holdouts.items():
        logger.info("[swissai] eval holdout=%s ntrain=%d ntest=%d",
                    name, tr.size, te.size)
        cell = _eval_one_holdout(
            holdout_name=name, train_idx=tr, test_idx=te, X=X,
            y_high=y_high, y_pct=y_pct, group_keys=group_keys, rows=rows,
            rolling_seen=rolling_seen,
            rolling_session_history=rolling_session_history)
        per_holdout.append(cell)
    return {
        "dataset": "swissai_bucket_reuse",
        "row_count": int(X.shape[0]),
        "feature_count": len(names),
        "feature_names": names,
        "high_reuse_threshold_pct": HIGH_REUSE_THRESHOLD,
        "leakage_features_excluded": sorted(LEAKAGE_TARGET_FIELDS),
        "per_holdout": per_holdout,
    }


def _run_cc_traces(cc_rows: list, source: str) -> dict:
    if not cc_rows:
        return {"dataset": "cc_traces", "row_count": 0, "skipped": "no_rows"}
    rows = add_rolling_features(cc_rows, source="cc_traces")
    # CC-traces does NOT have a reuse_percentage label; the target is the
    # derived intra-session reuse proxy.
    y_isr = derive_intra_session_reuse_from_cc_traces(rows)
    # Reuse the same feature pipeline.
    spec = build_feature_spec(rows)
    X, names, group_keys = build_feature_matrix(rows, spec)
    rolling_seen = np.array(
        [r.get("rolling_per_hash_seen_count", 0.0) for r in rows],
        dtype=np.float64)
    rolling_session_history = np.array(
        [r.get("rolling_per_model_reuse_pct", float("nan")) for r in rows],
        dtype=np.float64)
    timestamps = np.array(
        [r.get("__decision_timestamp_s") if r.get("__decision_timestamp_s") is not None
         else float("nan") for r in rows], dtype=np.float64)

    holdouts: dict = {}
    r_tr, r_te = random_holdout(X.shape[0])
    holdouts["random_holdout"] = (r_tr, r_te)
    # CC-traces uses request_arrival_delta_s relative to session start;
    # absolute timestamps are not available, so a time holdout is not
    # straightforward. Use session-holdout instead.
    s_tr, s_te = holdout_by_session(group_keys["session_id"])
    if s_te.size > 0:
        holdouts["holdout_by_session"] = (s_tr, s_te)

    per_holdout = []
    y_pct_placeholder = np.full(len(rows), float("nan"))
    for name, (tr, te) in holdouts.items():
        cell = _eval_one_holdout(
            holdout_name=name, train_idx=tr, test_idx=te, X=X,
            y_high=y_isr, y_pct=y_pct_placeholder, group_keys=group_keys,
            rows=rows, rolling_seen=rolling_seen,
            rolling_session_history=rolling_session_history)
        per_holdout.append(cell)
    return {
        "dataset": "cc_traces",
        "row_count": int(X.shape[0]),
        "source": source,
        "target": "intra_session_reuse (derived proxy)",
        "feature_count": len(names),
        "leakage_features_excluded": sorted(LEAKAGE_TARGET_FIELDS),
        "per_holdout": per_holdout,
        "result_quality": "shadow_proxy_derived_label",
        "headline_eligible": False,
    }


def _run_lmcache_proxy(lmcache_rows: list) -> dict:
    if not lmcache_rows:
        return {"dataset": "lmcache", "row_count": 0, "skipped": "no_rows"}
    rows = add_rolling_features(lmcache_rows, source="lmcache_agentic")
    # LMCache has no labelled reuse — we can only report the rolling
    # session history as a structural prior (not a trained model).
    per_session_counts: dict = {}
    for r in rows:
        s = r.get("session_id")
        per_session_counts[s] = per_session_counts.get(s, 0) + 1
    return {
        "dataset": "lmcache",
        "row_count": len(rows),
        "session_count": len(per_session_counts),
        "median_session_size": (
            float(np.median(list(per_session_counts.values())))
            if per_session_counts else 0.0),
        "available_signal": "session continuity + pre_gap_s",
        "result_quality": "structural_prior_only",
        "headline_eligible": False,
        "skipped_training": (
            "lmcache has no reuse-percentage / cache-hit label; used as "
            "cross-dataset structural prior only"),
    }


def _run_prefixbench(prefixbench_rows: list) -> dict:
    if not prefixbench_rows:
        return {"dataset": "prefixbench", "row_count": 0, "skipped": "no_rows"}
    # Synthetic prefix_group: per-prefix-group mean prompt_text_len.
    by_group: dict = {}
    for r in prefixbench_rows:
        g = r.get("prefix_group")
        by_group.setdefault(g, []).append(r)
    return {
        "dataset": "prefixbench",
        "row_count": len(prefixbench_rows),
        "prefix_group_count": len(by_group),
        "result_quality": "synthetic",
        "headline_eligible": False,
        "skipped_training": (
            "prefixbench is synthetic; used as priors / generalization check"
            " only, never as headline training source"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _scorer_supports_cache_value() -> bool:
    """Return whether ``aurelius.residency.decision.score_residency_candidate``
    can express cache reuse / prefill savings / migration cache-loss
    value today. As of PR #131 the answer is FALSE — see
    ``docs/PLACEMENT_PRIOR_AUDIT.md`` §scoring_inputs.
    """
    try:
        from aurelius.residency.decision import score_residency_candidate  # noqa: F401
    except Exception:
        return False
    # The audit reports cache_reuse + prefill_savings as missing inputs
    # to the scorer. Until a future scorer-side PR adds these hooks the
    # cache forecaster cannot drive the production economic KPI.
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-swissai-rows", type=int, default=None)
    p.add_argument("--limit-cc-rows", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0 result (already committed).
    if PHASE0_PATH.exists():
        phase0 = json.loads(PHASE0_PATH.read_text())
    else:
        phase0 = {"decision": "not_run",
                  "decision_reason": "Phase 0 expansion summary not found"}

    # Load datasets.
    swissai_rows = _load_swissai_bucket_reuse()
    if args.limit_swissai_rows:
        swissai_rows = swissai_rows[: args.limit_swissai_rows]
    cc_rows, cc_source = _load_cc_traces()
    if args.limit_cc_rows:
        cc_rows = cc_rows[: args.limit_cc_rows]
    lmcache_rows = _load_lmcache()
    prefixbench_rows = _load_prefixbench()

    logger.info("dataset rows: swissai=%d cc_traces=%d (%s) lmcache=%d prefixbench=%d",
                len(swissai_rows), len(cc_rows), cc_source,
                len(lmcache_rows), len(prefixbench_rows))

    # Phase A — readiness audit.
    audit = write_data_readiness_audit(
        swissai_rows=swissai_rows, cc_rows=cc_rows, cc_source=cc_source,
        lmcache_rows=lmcache_rows, prefixbench_rows=prefixbench_rows,
        phase0=phase0,
    )

    # Phase B + C — train + evaluate.
    summary: dict = {
        "doc_version": "cache_prefix_reuse_forecaster_v1",
        "production_claim": False,
        "shadow_only": True,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "evaluated_at_s": time.time(),
        "datasets_used": {
            "swissai_bucket_reuse_rows": len(swissai_rows),
            "cc_traces_rows": len(cc_rows),
            "cc_traces_source": cc_source,
            "lmcache_rows": len(lmcache_rows),
            "prefixbench_rows": len(prefixbench_rows),
        },
        "phase0_cc_traces_decision": phase0.get("decision"),
        "phase0_cc_traces_decision_reason": phase0.get("decision_reason"),
        "data_readiness_audit_path": str(AUDIT_PATH.relative_to(REPO_ROOT)),
        "leakage_features_excluded": sorted(LEAKAGE_TARGET_FIELDS),
        "baselines_used": [
            "global_reuse_rate", "per_model_reuse_rate",
            "per_session_history", "recency_frequency_seen",
            "prefix_group (PrefixBench only)",
        ],
        "ml_candidates_used": [
            "logistic_reuse_classifier",
            "hist_gradient_boosting_reuse_classifier",
            "random_forest_reuse_classifier (capped at 50k rows)",
            "hist_gradient_boosting_reuse_regressor",
        ],
    }

    if swissai_rows:
        summary["swissai_results"] = _run_swissai(swissai_rows)
    else:
        summary["swissai_results"] = {"skipped": "no swissai rows present"}

    summary["cc_traces_results"] = _run_cc_traces(cc_rows, cc_source)
    summary["lmcache_results"] = _run_lmcache_proxy(lmcache_rows)
    summary["prefixbench_results"] = _run_prefixbench(prefixbench_rows)

    # Cross-dataset summary: report time_holdout + by_model_holdout +
    # random_holdout as parallel signals. The "binding" economic metric
    # is the time_holdout result (most realistic generalisation test);
    # if absent, fall back to the worst by_model holdout.
    swissai_per_holdout = summary.get("swissai_results", {}).get(
        "per_holdout", [])
    holdout_econ: dict = {}
    for cell in swissai_per_holdout:
        b = cell.get("best_economic_improvement_pct")
        if b is not None:
            holdout_econ[cell["holdout"]] = b
    summary["swissai_economic_improvement_by_holdout_pct"] = holdout_econ

    time_econ = holdout_econ.get("time_holdout")
    by_model_keys = [k for k in holdout_econ if k.startswith("holdout_by_model_")]
    worst_by_model = (min(holdout_econ[k] for k in by_model_keys)
                      if by_model_keys else None)
    random_econ = holdout_econ.get("random_holdout")
    if time_econ is not None:
        binding_econ = time_econ
        binding_holdout = "time_holdout"
    elif worst_by_model is not None:
        binding_econ = worst_by_model
        binding_holdout = "worst_by_model_holdout"
    elif random_econ is not None:
        binding_econ = random_econ
        binding_holdout = "random_holdout"
    else:
        binding_econ = 0.0
        binding_holdout = "no_holdout_passed"

    summary["binding_swissai_economic_improvement_pct"] = binding_econ
    summary["binding_swissai_holdout"] = binding_holdout
    summary["worst_by_model_economic_improvement_pct"] = worst_by_model
    summary["random_holdout_economic_improvement_pct"] = random_econ
    summary["time_holdout_economic_improvement_pct"] = time_econ
    # Back-compat key for older callers / tests that read "best_*".
    summary["best_swissai_economic_improvement_pct"] = binding_econ
    summary["best_swissai_holdout"] = binding_holdout
    summary["best_swissai_subgroup_failures"] = [
        c["holdout"] for c in swissai_per_holdout
        if c.get("subgroup_audit", {}).get("has_subgroup_regression", False)
    ]
    # Locate the cell used for binding classification.
    best_econ_cell = next(
        (c for c in swissai_per_holdout if c.get("holdout") == binding_holdout),
        None)
    if best_econ_cell is None and swissai_per_holdout:
        best_econ_cell = swissai_per_holdout[0]
    best_econ = binding_econ

    # Cross-dataset note (CC-traces shadow proxy).
    best_cc_econ = float("-inf")
    best_cc_cell = None
    for cell in summary.get("cc_traces_results", {}).get("per_holdout", []):
        b = cell.get("best_economic_improvement_pct")
        if b is not None and b > best_cc_econ:
            best_cc_econ = b
            best_cc_cell = cell
    summary["best_cc_traces_economic_improvement_pct"] = (
        best_cc_econ if best_cc_cell is not None else None)
    summary["cc_traces_reported_separately_because"] = (
        "CC-traces uses KV-block-hash-derived intra_session_reuse labels, "
        "not reuse_percentage. Per mission spec, headline reporting only "
        "from SwissAI / PrefixBench / LMCache labels."
    )

    # Promotion classification.
    scorer_ok = _scorer_supports_cache_value()
    has_calibration_failure = False
    has_subgroup_regression = False
    if best_econ_cell is not None:
        has_subgroup_regression = bool(
            best_econ_cell.get("subgroup_audit", {}).get(
                "has_subgroup_regression", False))
        # Calibration failure when the best ML model's ECE > 0.15.
        best_ml = best_econ_cell.get("best_ml_model")
        if best_ml:
            ml = best_econ_cell["ml_metrics"].get(best_ml, {})
            ece = ml.get("expected_calibration_error")
            if ece is not None and not (ece != ece) and ece > 0.15:
                has_calibration_failure = True

    status, reason = classify_economic_status(
        best_economic_improvement_pct=(best_econ if best_econ_cell is not None
                                       else 0.0),
        has_subgroup_regression=has_subgroup_regression,
        has_calibration_failure=has_calibration_failure,
        leakage_free=True,
        scorer_supports_cache_value=scorer_ok,
    )
    summary["final_status"] = status
    summary["final_status_reason"] = reason
    summary["scorer_supports_cache_value_today"] = scorer_ok
    summary["scorer_limitation_note"] = (
        "aurelius.residency.decision.score_residency_candidate does not "
        "today consume cache-reuse / prefill-savings / migration-cache-loss "
        "value (see docs/PLACEMENT_PRIOR_AUDIT.md::scoring_inputs). A future "
        "scorer-side PR is required before any shadow-ready cache forecast "
        "can drive a residency / routing / migration-veto decision."
    )
    summary["shadow_integration_justified"] = (
        status == "shadow_ready_for_integration_review")
    summary["pilot_only_remaining_items"] = [
        "real measured cache_hit per request (no HF dataset provides this)",
        "measured kv_evictions per second per replica (CARA proxy only)",
        "measured cold-start latency per model (no HF dataset provides this)",
        "scorer-side PR to expose cache_value / prefill_savings hooks",
    ]

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                       default=str))
    print(f"[cache-prefix] wrote {SUMMARY_PATH}")
    print(f"[cache-prefix] final_status={status}")
    print(f"[cache-prefix] reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
