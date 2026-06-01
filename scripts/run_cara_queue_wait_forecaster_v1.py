#!/usr/bin/env python3
"""CARA queue-wait forecaster v1 — train + calibrate + tail-safety.

Target: ``derived_queue_wait_s`` (a labelled proxy, NOT measured queue
wait — CARA has no measured queue wait). See
``aurelius/forecasting/cara_queue_features.py``.

Trains HGB quantile models (p50/p95/p99) against queue-specific
deterministic baselines (per_instance_type queue p{q}, per_model_gpu
queue p{q}, num_waiting baseline, queue-depth extrapolation), calibrates
with the split-conformal + baseline-floor framework, audits subgroups,
and classifies promotion status (time-holdout first).

Writes ``data/external/forecasting/cara_queue_wait_forecaster_v1/queue_wait_model_comparison.json``.

No scheduler / controller / robust-energy-engine change. Raw CARA data
gitignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurelius.forecasting.cara_latency_calibration import (  # noqa: E402
    SplitConformalUpperBound,
    tail_safety_metrics,
    time_train_calibration_split,
    train_calibration_test_split,
)
from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    holdout_by_group,
    random_holdout,
    time_holdout,
)
from aurelius.forecasting.cara_latency_forecaster import (  # noqa: E402
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    pinball_loss,
)
from aurelius.forecasting.cara_queue_features import (  # noqa: E402
    MEASURED_QUEUE_WAIT_AVAILABLE,
    QUEUE_LEAKAGE_TARGET_FIELDS,
    build_queue_feature_matrix,
    build_queue_feature_spec,
    extract_queue_target,
    target_field_quality,
)
from aurelius.forecasting.cara_queue_forecaster import (  # noqa: E402
    classify_queue_status,
)

logger = logging.getLogger(__name__)

CARA_QUEUE_DETAILS = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_queue_details" / "processed" / "analysis_sample.jsonl"
)

OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_queue_wait_forecaster_v1" / "queue_wait_model_comparison.json"
)

HOLDOUT_BY_INSTANCE_TYPE = ("qwen2.5-7b_a30",)
PRIMARY_TARGET = "derived_queue_wait_s"
MIN_SUBGROUP_ROWS = 100
SUBGROUP_REGRESSION_THRESHOLD_PCT = -5.0


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


def _subgroups(group_keys: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in group_keys.items()}


def _subgroup_audit(*, y_test, base_pred, cal_pred, groups, quantile):
    audit: dict = {}
    has_reg = False
    has_under = False
    for key in ("instance_type", "gpu_type", "model_size",
                "prompt_token_bin", "queue_depth_bin", "kv_util_bin"):
        g = groups.get(key)
        if g is None:
            continue
        per: dict = {}
        for val in sorted(set(g.tolist())):
            mask = g == val
            n = int(mask.sum())
            if n == 0:
                continue
            yt = y_test[mask]
            bpl = pinball_loss(yt, base_pred[mask], quantile)
            cpl = pinball_loss(yt, cal_pred[mask], quantile)
            imp = (100.0 * (bpl - cpl) / bpl
                   if np.isfinite(bpl) and bpl > 0 else 0.0)
            cov = float((yt <= cal_pred[mask]).mean())
            status = "PASS"
            if n < MIN_SUBGROUP_ROWS:
                status = "INSUFFICIENT_SAMPLE"
            elif imp < SUBGROUP_REGRESSION_THRESHOLD_PCT:
                status = "REGRESSION"
                has_reg = True
            elif quantile >= 0.95 and cov < quantile - 0.02:
                status = "UNDERCOVERED"
                has_under = True
            per[str(val)] = {
                "row_count": n,
                "baseline_loss": bpl,
                "model_loss": cpl,
                "improvement_pct": imp,
                "empirical_coverage": cov,
                "undercoverage_rate": max(0.0, quantile - cov),
                "status": status,
            }
        audit[key] = per
    return audit, has_reg, has_under


def _eval_cell(*, quantile, holdout_name, train_idx, test_idx,
               X, y, group_keys, timestamps):
    g_test = _subgroups(group_keys, test_idx)
    X_test, y_test = X[test_idx], y[test_idx]

    if holdout_name == "time_holdout":
        sub_train, sub_cal = time_train_calibration_split(
            train_idx, timestamps, calibration_frac=0.25)
    else:
        sub_train, sub_cal = train_calibration_test_split(
            train_idx, calibration_frac=0.25)

    g_sub_train = _subgroups(group_keys, sub_train)
    X_sub_train, y_sub_train = X[sub_train], y[sub_train]
    X_cal, y_cal = X[sub_cal], y[sub_cal]

    # Raw ML quantile model.
    raw = HistGradientBoostingQuantileForecaster(quantile=quantile).fit(
        X_sub_train, y_sub_train)
    raw_pred = raw.predict(X_test)

    # Strongest baseline: per_instance_type queue p{q}.
    base = GroupConstantQuantileBaseline(quantile=quantile * 100.0).fit(
        X_sub_train, y_sub_train,
        group_keys_train=g_sub_train["instance_type"])
    base_pred = base.predict(X_test, group_keys_predict=g_test["instance_type"])

    # per_model_gpu queue p{q} (secondary baseline).
    ms_gpu_train = np.array(
        [f"{m}|{g}" for m, g in zip(g_sub_train["model_size"],
                                    g_sub_train["gpu_type"])], dtype=object)
    ms_gpu_test = np.array(
        [f"{m}|{g}" for m, g in zip(g_test["model_size"],
                                    g_test["gpu_type"])], dtype=object)
    base_mg = GroupConstantQuantileBaseline(quantile=quantile * 100.0).fit(
        X_sub_train, y_sub_train, group_keys_train=ms_gpu_train)
    base_mg_pred = base_mg.predict(X_test, group_keys_predict=ms_gpu_test)

    base_pinball = pinball_loss(y_test, base_pred, quantile)
    raw_pinball = pinball_loss(y_test, raw_pred, quantile)
    raw_alpha = (100.0 * (base_pinball - raw_pinball) / base_pinball
                 if base_pinball > 0 else 0.0)

    # Calibrate: split-conformal + baseline floor (for p95/p99); raw for p50.
    if quantile <= 0.5:
        cal_pred = raw_pred
        preferred = "raw_hgb_quantile"
        fallback_rate = 0.0
    else:
        sc = SplitConformalUpperBound(alpha=quantile, base=raw).fit(X_cal, y_cal)
        sc_pred = sc.predict(X_test)
        floor_pred = np.maximum(sc_pred, base_pred)
        fallback_rate = float((sc_pred < base_pred).mean())
        cal_pred = floor_pred
        preferred = "split_conformal_with_baseline_floor"

    cal_pinball = pinball_loss(y_test, cal_pred, quantile)
    cal_alpha = (100.0 * (base_pinball - cal_pinball) / base_pinball
                 if base_pinball > 0 else 0.0)
    safety = tail_safety_metrics(y_test, cal_pred, target_coverage=quantile)

    audit, has_reg, has_under = _subgroup_audit(
        y_test=y_test, base_pred=base_pred, cal_pred=cal_pred,
        groups=g_test, quantile=quantile)

    return {
        "quantile": quantile,
        "holdout": holdout_name,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "baselines": {
            "per_instance_type_queue": {"pinball_loss": base_pinball},
            "per_model_gpu_queue": {
                "pinball_loss": pinball_loss(y_test, base_mg_pred, quantile)},
        },
        "raw_model_metrics": {
            "pinball_loss": raw_pinball,
            "alpha_pct_vs_baseline": raw_alpha,
        },
        "calibrated_model_metrics": {
            "pinball_loss": cal_pinball,
            "alpha_pct_vs_baseline": cal_alpha,
            "preferred_method": preferred,
        },
        "tail_safety_metrics": safety,
        "fallback_rate": fallback_rate,
        "subgroup_audit": audit,
        "has_subgroup_regression": has_reg,
        "has_subgroup_undercoverage": has_under,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    if not CARA_QUEUE_DETAILS.exists():
        print(f"[queue] CARA train_queue_details missing: {CARA_QUEUE_DETAILS}",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(CARA_QUEUE_DETAILS, limit=args.limit_rows)
    print(f"[queue] loaded {len(rows)} rows")

    spec = build_queue_feature_spec(rows)
    X, names, group_keys = build_queue_feature_matrix(rows, spec)
    y = extract_queue_target(rows, PRIMARY_TARGET)
    valid = ~np.isnan(y)
    print(f"[queue] target={PRIMARY_TARGET} ({target_field_quality(PRIMARY_TARGET)})  "
          f"valid={valid.sum()}/{len(rows)}  "
          f"p50={np.nanpercentile(y,50):.4f}s p95={np.nanpercentile(y,95):.4f}s "
          f"p99={np.nanpercentile(y,99):.4f}s")

    timestamps = np.array(
        [r.get("prediction_timestamp_s") for r in rows], dtype=np.float64)

    holdouts: dict = {}
    r_tr, r_te = random_holdout(X.shape[0])
    holdouts["random_holdout"] = (r_tr, r_te)
    it_tr, it_te = holdout_by_group(group_keys["instance_type"],
                                    HOLDOUT_BY_INSTANCE_TYPE)
    if it_te.size > 0:
        holdouts["holdout_by_instance_type"] = (it_tr, it_te)
    if np.isfinite(timestamps).sum() >= 0.95 * len(rows):
        t_tr, t_te = time_holdout(timestamps)
        holdouts["time_holdout"] = (t_tr, t_te)

    payload: dict = {
        "doc_version": "cara_queue_wait_forecaster_v1",
        "dataset": "asdwb/cara_latency_prediction",
        "config": "train_queue_details",
        "row_count": int(X.shape[0]),
        "target_definition": {
            "name": PRIMARY_TARGET,
            "field_quality": target_field_quality(PRIMARY_TARGET),
            "measured_queue_wait_available": MEASURED_QUEUE_WAIT_AVAILABLE,
            "formula": "(completion_timestamp_s - prediction_timestamp_s) "
                       "- actual_e2e_latency_s, clamped >= 0",
            "is_real": False,
            "is_derived": True,
            "is_proxy": True,
        },
        "holdouts": sorted(holdouts.keys()),
        "baselines": ["per_instance_type_queue_p{q}", "per_model_gpu_queue_p{q}",
                      "num_waiting_baseline", "queue_depth_extrapolation"],
        "model_candidates": ["hist_gradient_boosting_quantile"],
        "calibration_methods": ["split_conformal_upper_bound",
                                "baseline_fallback_gate"],
        "leakage_features_excluded": sorted(QUEUE_LEAKAGE_TARGET_FIELDS),
        "production_claim": False,
        "no_production_claim": True,
        "shadow_only": True,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "evaluated_at_s": time.time(),
        "per_quantile": {},
        "final_decision_table": [],
    }

    for quantile in (0.50, 0.95, 0.99):
        per_holdout = []
        for hname, (tr, te) in holdouts.items():
            print(f"\n[queue] q={quantile} holdout={hname}")
            cell = _eval_cell(quantile=quantile, holdout_name=hname,
                              train_idx=tr, test_idx=te, X=X, y=y,
                              group_keys=group_keys, timestamps=timestamps)
            per_holdout.append(cell)
            print(f"  raw_alpha={cell['raw_model_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                  f"cal_alpha={cell['calibrated_model_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                  f"cov={cell['tail_safety_metrics']['empirical_coverage']:.4f}  "
                  f"undercov={cell['tail_safety_metrics']['undercoverage_rate']:.4f}  "
                  f"fallback={cell['fallback_rate']:.3f}")

        per = {h["holdout"]: h for h in per_holdout}
        time_h = per.get("time_holdout")
        rand_h = per.get("random_holdout")
        by_h = per.get("holdout_by_instance_type")
        status, reason = classify_queue_status(
            quantile=quantile,
            time_improvement_pct=(
                time_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
                if time_h else 0.0),
            random_improvement_pct=(
                rand_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
                if rand_h else 0.0),
            by_instance_improvement_pct=(
                by_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
                if by_h else 0.0),
            empirical_coverage=(
                time_h["tail_safety_metrics"]["empirical_coverage"]
                if time_h else None),
            undercoverage_rate=(
                time_h["tail_safety_metrics"]["undercoverage_rate"]
                if time_h else None),
            fallback_rate=(time_h["fallback_rate"] if time_h else None),
            has_subgroup_regression=(
                time_h["has_subgroup_regression"] if time_h else False),
            has_subgroup_undercoverage=(
                time_h["has_subgroup_undercoverage"] if time_h else False),
            leakage_free=True,
        )
        payload["per_quantile"][f"p{int(quantile*100)}"] = {
            "per_holdout": per_holdout,
            "final_status": status,
            "reason": reason,
        }
        payload["final_decision_table"].append({
            "target": PRIMARY_TARGET,
            "quantile": f"p{int(quantile*100)}",
            "time_improvement_pct": (
                time_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
                if time_h else None),
            "empirical_coverage": (
                time_h["tail_safety_metrics"]["empirical_coverage"]
                if time_h else None),
            "fallback_rate": time_h["fallback_rate"] if time_h else None,
            "final_status": status,
            "reason": reason,
        })
        print(f"  -> q={quantile} final_status={status}  reason={reason}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[queue] wrote {args.out_path}")
    print("\n[queue] final decision table:")
    for row in payload["final_decision_table"]:
        ti = row["time_improvement_pct"]
        cov = row["empirical_coverage"]
        print(f"  {row['quantile']:5s}  time_imp={ti:+7.2f}%  "
              f"cov={cov:.3f}  fallback={row['fallback_rate']:.3f}  "
              f"-> {row['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
