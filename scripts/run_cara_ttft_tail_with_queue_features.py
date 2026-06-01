#!/usr/bin/env python3
"""Experiment: do queue-forecast features improve TTFT p95/p99 tail safety?

This is a NEW experiment, not a replacement for the prior forecaster.

Method (leakage-safe):

1. For each TTFT holdout (random / by_instance / time):
   - Train a queue-wait forecaster (HGB quantile p50/p95/p99) on the
     TTFT **train** fold only.
   - Cross-fit (2-fold) within the TTFT train fold to produce
     **out-of-fold** queue predictions for the TTFT train rows — so the
     TTFT model never trains on in-sample (optimistic) queue features.
   - Predict queue p50/p95/p99 + uncertainty for the TTFT **test** fold
     using a queue model that never saw the TTFT test labels.
2. Append the queue forecasts as extra TTFT features:
   ``predicted_queue_p50``, ``predicted_queue_p95``,
   ``predicted_queue_p99``, ``queue_pressure_score``,
   ``queue_forecast_uncertainty`` (= p99 - p50).
3. Retrain TTFT p95/p99 with the augmented feature set.
4. Calibrate (split-conformal + baseline floor) + tail-safety + subgroup.
5. Compare against the per_instance_type baseline AND the prior
   calibrated TTFT p95/p99 (from calibration_tail_safety_summary.json).

Promotion gates are identical to PR #127's tail gates. A model is only
promoted if it actually clears them on the time-holdout. Negative
results are reported honestly.

Writes ``data/external/forecasting/cara_latency_forecaster_v1/ttft_tail_with_queue_features_summary.json``.
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
    classify_tail_status,
    tail_safety_metrics,
    time_train_calibration_split,
    train_calibration_test_split,
)
from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    build_feature_matrix,
    build_feature_spec,
    extract_target,
    holdout_by_group,
    random_holdout,
    time_holdout,
)
from aurelius.forecasting.cara_latency_forecaster import (  # noqa: E402
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    pinball_loss,
)
from aurelius.forecasting.cara_queue_features import extract_queue_target  # noqa: E402

logger = logging.getLogger(__name__)

CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)
PRIOR_CALIB = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1" / "calibration_tail_safety_summary.json"
)
OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1"
    / "ttft_tail_with_queue_features_summary.json"
)

TARGET = "actual_ttft_s"
HOLDOUT_BY_INSTANCE_TYPE = ("qwen2.5-7b_a30",)
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


def _queue_features_for_split(
    X_base, y_queue, train_local, test_local, *, seed=20260601,
):
    """Return (queue_feats_train, queue_feats_test).

    Cross-fits within ``train_local`` (2 folds) so the queue predictions
    for the train rows are out-of-fold. The test rows get predictions
    from a queue model trained on ALL of train_local.

    Each returned matrix has 5 columns:
    [q_p50, q_p95, q_p99, queue_uncertainty(=p99-p50), pressure_proxy(=p95)].
    """
    rng = np.random.default_rng(seed)
    n_train = train_local.size
    perm = rng.permutation(n_train)
    fold_a = train_local[perm[: n_train // 2]]
    fold_b = train_local[perm[n_train // 2:]]

    train_feats = np.zeros((n_train, 5), dtype=np.float64)
    pos = {idx: i for i, idx in enumerate(train_local)}

    for fit_fold, pred_fold in ((fold_a, fold_b), (fold_b, fold_a)):
        preds = {}
        for q in (0.50, 0.95, 0.99):
            m = HistGradientBoostingQuantileForecaster(quantile=q).fit(
                X_base[fit_fold], y_queue[fit_fold])
            preds[q] = m.predict(X_base[pred_fold])
        for j, idx in enumerate(pred_fold):
            row = pos[idx]
            train_feats[row, 0] = preds[0.50][j]
            train_feats[row, 1] = preds[0.95][j]
            train_feats[row, 2] = preds[0.99][j]
            train_feats[row, 3] = preds[0.99][j] - preds[0.50][j]
            train_feats[row, 4] = preds[0.95][j]

    # Test predictions from a queue model trained on all of train_local.
    test_feats = np.zeros((test_local.size, 5), dtype=np.float64)
    test_preds = {}
    for q in (0.50, 0.95, 0.99):
        m = HistGradientBoostingQuantileForecaster(quantile=q).fit(
            X_base[train_local], y_queue[train_local])
        test_preds[q] = m.predict(X_base[test_local])
    test_feats[:, 0] = test_preds[0.50]
    test_feats[:, 1] = test_preds[0.95]
    test_feats[:, 2] = test_preds[0.99]
    test_feats[:, 3] = test_preds[0.99] - test_preds[0.50]
    test_feats[:, 4] = test_preds[0.95]
    return train_feats, test_feats


def _subgroups(group_keys, idx):
    return {k: v[idx] for k, v in group_keys.items()}


def _subgroup_audit(*, y_test, base_pred, cal_pred, groups, quantile):
    audit, has_reg, has_under = {}, False, False
    for key in ("instance_type", "gpu_type", "model_size",
                "prompt_token_bin", "queue_depth_bin", "kv_util_bin"):
        g = groups.get(key)
        if g is None:
            continue
        per = {}
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
            elif cov < quantile - 0.02:
                status = "UNDERCOVERED"
                has_under = True
            per[str(val)] = {
                "row_count": n, "baseline_loss": bpl, "model_loss": cpl,
                "improvement_pct": imp, "empirical_coverage": cov,
                "status": status,
            }
        audit[key] = per
    return audit, has_reg, has_under


def _eval_cell(*, quantile, holdout_name, train_idx, test_idx,
               X, X_aug_builder, y, y_queue, group_keys, timestamps):
    """Evaluate one TTFT-with-queue (quantile, holdout) cell."""
    g_test = _subgroups(group_keys, test_idx)
    y_test = y[test_idx]

    if holdout_name == "time_holdout":
        sub_train, sub_cal = time_train_calibration_split(
            train_idx, timestamps, calibration_frac=0.25)
    else:
        sub_train, sub_cal = train_calibration_test_split(
            train_idx, calibration_frac=0.25)

    g_sub_train = _subgroups(group_keys, sub_train)

    # Queue features for (sub_train, sub_cal, test). Each split's queue
    # predictions come from models that never saw that split's labels.
    qf_sub_train, qf_cal = _queue_features_for_split(
        X, y_queue, sub_train, sub_cal)
    _, qf_test = _queue_features_for_split(X, y_queue, sub_train, test_idx)

    X_sub_train = np.hstack([X[sub_train], qf_sub_train])
    X_cal = np.hstack([X[sub_cal], qf_cal])
    X_test = np.hstack([X[test_idx], qf_test])
    y_sub_train = y[sub_train]
    y_cal = y[sub_cal]

    raw = HistGradientBoostingQuantileForecaster(quantile=quantile).fit(
        X_sub_train, y_sub_train)
    raw_pred = raw.predict(X_test)

    base = GroupConstantQuantileBaseline(quantile=quantile * 100.0).fit(
        X[sub_train], y_sub_train,
        group_keys_train=g_sub_train["instance_type"])
    base_pred = base.predict(X[test_idx],
                             group_keys_predict=g_test["instance_type"])

    base_pinball = pinball_loss(y_test, base_pred, quantile)
    raw_pinball = pinball_loss(y_test, raw_pred, quantile)
    raw_alpha = (100.0 * (base_pinball - raw_pinball) / base_pinball
                 if base_pinball > 0 else 0.0)

    sc = SplitConformalUpperBound(alpha=quantile, base=raw).fit(X_cal, y_cal)
    sc_pred = sc.predict(X_test)
    cal_pred = np.maximum(sc_pred, base_pred)
    fallback_rate = float((sc_pred < base_pred).mean())
    cal_pinball = pinball_loss(y_test, cal_pred, quantile)
    cal_alpha = (100.0 * (base_pinball - cal_pinball) / base_pinball
                 if base_pinball > 0 else 0.0)
    safety = tail_safety_metrics(y_test, cal_pred, target_coverage=quantile)
    audit, has_reg, has_under = _subgroup_audit(
        y_test=y_test, base_pred=base_pred, cal_pred=cal_pred,
        groups=g_test, quantile=quantile)

    return {
        "quantile": quantile, "holdout": holdout_name,
        "n_train": int(train_idx.size), "n_test": int(test_idx.size),
        "raw_with_queue_metrics": {
            "pinball_loss": raw_pinball, "alpha_pct_vs_baseline": raw_alpha},
        "calibrated_with_queue_metrics": {
            "pinball_loss": cal_pinball, "alpha_pct_vs_baseline": cal_alpha},
        "tail_safety_metrics": safety,
        "fallback_rate": fallback_rate,
        "subgroup_audit": audit,
        "has_subgroup_regression": has_reg,
        "has_subgroup_undercoverage": has_under,
    }


def _prior_calibrated_alpha(prior, quantile_label):
    """Pull the prior calibrated time-holdout alpha for actual_ttft_s."""
    try:
        cell = prior["per_target"]["actual_ttft_s"]["per_quantile"][
            quantile_label]
        for h in cell["per_holdout"]:
            if h["holdout"] == "time_holdout":
                return {
                    "calibrated_alpha_pct": h["calibrated_model_metrics"][
                        "alpha_pct_vs_baseline"],
                    "empirical_coverage": h["tail_safety_metrics"][
                        "empirical_coverage"],
                    "final_status": cell["decision"]["final_status"],
                }
    except (KeyError, TypeError):
        pass
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    if not CARA_TRAIN_FLAT.exists():
        print(f"[ttft+q] CARA train_flat missing: {CARA_TRAIN_FLAT}",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(CARA_TRAIN_FLAT, limit=args.limit_rows)
    print(f"[ttft+q] loaded {len(rows)} rows")

    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X, names, group_keys = build_feature_matrix(rows, spec)
    y = extract_target(rows, TARGET)
    y_queue = extract_queue_target(rows, "derived_queue_wait_s")
    timestamps = np.array(
        [r.get("prediction_timestamp_s") for r in rows], dtype=np.float64)

    prior = json.loads(PRIOR_CALIB.read_text()) if PRIOR_CALIB.exists() else {}

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
        "doc_version": "cara_ttft_tail_with_queue_features_v1",
        "dataset": "asdwb/cara_latency_prediction",
        "config": "train_flat",
        "row_count": int(X.shape[0]),
        "target": TARGET,
        "queue_feature_names": [
            "predicted_queue_p50", "predicted_queue_p95",
            "predicted_queue_p99", "queue_forecast_uncertainty",
            "queue_pressure_score"],
        "queue_features_are_out_of_fold": True,
        "queue_target_is_derived_proxy": True,
        "holdouts": sorted(holdouts.keys()),
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

    for quantile in (0.95, 0.99):
        q_label = f"p{int(quantile*100)}"
        per_holdout = []
        for hname, (tr, te) in holdouts.items():
            print(f"\n[ttft+q] q={quantile} holdout={hname}")
            cell = _eval_cell(
                quantile=quantile, holdout_name=hname, train_idx=tr,
                test_idx=te, X=X, X_aug_builder=None, y=y, y_queue=y_queue,
                group_keys=group_keys, timestamps=timestamps)
            per_holdout.append(cell)
            print(f"  raw+q_alpha={cell['raw_with_queue_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                  f"cal+q_alpha={cell['calibrated_with_queue_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                  f"cov={cell['tail_safety_metrics']['empirical_coverage']:.4f}  "
                  f"fallback={cell['fallback_rate']:.3f}")

        per = {h["holdout"]: h for h in per_holdout}
        time_h = per.get("time_holdout")
        rand_h = per.get("random_holdout")
        by_h = per.get("holdout_by_instance_type")
        fallback_required = bool(
            time_h and time_h["fallback_rate"] > 0.25)
        status, reason = classify_tail_status(
            target_family="ttft", quantile=quantile,
            time_holdout_improvement_pct=(
                time_h["calibrated_with_queue_metrics"]["alpha_pct_vs_baseline"]
                if time_h else 0.0),
            random_holdout_improvement_pct=(
                rand_h["calibrated_with_queue_metrics"]["alpha_pct_vs_baseline"]
                if rand_h else 0.0),
            by_instance_holdout_improvement_pct=(
                by_h["calibrated_with_queue_metrics"]["alpha_pct_vs_baseline"]
                if by_h else 0.0),
            empirical_coverage=(
                time_h["tail_safety_metrics"]["empirical_coverage"]
                if time_h else None),
            undercoverage_rate=(
                time_h["tail_safety_metrics"]["undercoverage_rate"]
                if time_h else None),
            has_subgroup_regression=(
                time_h["has_subgroup_regression"] if time_h else False),
            has_subgroup_undercoverage=(
                time_h["has_subgroup_undercoverage"] if time_h else False),
            fallback_required_on_time=fallback_required,
            leakage_free=True, no_test_label_calibration=True)

        prior_cell = _prior_calibrated_alpha(prior, q_label)
        new_time_alpha = (
            time_h["calibrated_with_queue_metrics"]["alpha_pct_vs_baseline"]
            if time_h else None)
        new_cov = (time_h["tail_safety_metrics"]["empirical_coverage"]
                   if time_h else None)
        payload["per_quantile"][q_label] = {
            "per_holdout": per_holdout,
            "new_status_with_queue_features": status,
            "reason": reason,
            "prior_calibrated_without_queue": prior_cell,
        }
        payload["final_decision_table"].append({
            "target": TARGET, "quantile": q_label,
            "prior_status": (prior_cell["final_status"]
                             if prior_cell else "unknown"),
            "prior_time_alpha_pct": (prior_cell["calibrated_alpha_pct"]
                                     if prior_cell else None),
            "new_time_alpha_pct": new_time_alpha,
            "time_alpha_delta_pct": (
                (new_time_alpha - prior_cell["calibrated_alpha_pct"])
                if (prior_cell and new_time_alpha is not None) else None),
            "prior_coverage": (prior_cell["empirical_coverage"]
                               if prior_cell else None),
            "new_coverage": new_cov,
            "new_fallback_rate": (time_h["fallback_rate"] if time_h else None),
            "new_status": status,
            "final_decision": status,
            "reason": reason,
        })
        print(f"  -> {q_label} new_status={status}  reason={reason}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[ttft+q] wrote {args.out_path}")
    print("\n[ttft+q] final decision table:")
    for row in payload["final_decision_table"]:
        print(f"  {row['quantile']:5s}  prior={row['prior_status']:20s}  "
              f"prior_α={_fmt(row['prior_time_alpha_pct'])}  "
              f"new_α={_fmt(row['new_time_alpha_pct'])}  "
              f"Δ={_fmt(row['time_alpha_delta_pct'])}  -> {row['new_status']}")
    return 0


def _fmt(v):
    return f"{v:+7.2f}%" if isinstance(v, (int, float)) else "   n/a "


if __name__ == "__main__":
    raise SystemExit(main())
