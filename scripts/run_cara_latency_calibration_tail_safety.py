#!/usr/bin/env python3
"""CARA latency forecaster v1 — calibration + tail-safety driver.

Implements PHASES A–E of the mission spec:

A. Re-run the v1 evaluation; reproduce the prior pinball-loss numbers.
B. For each (target × quantile × holdout) train a calibration-aware
   variant: ``ConservativeMultiplierCalibration``, ``QuantileResidualCalibration``,
   ``SplitConformalUpperBound``, and a ``BaselineFallbackGate`` floor.
C. Compute tail-safety metrics on test (empirical coverage,
   undercoverage, conservatism, residual quantiles).
D. Subgroup safety audit by instance_type / gpu_type / model_size /
   prompt_token_bin / queue_depth_bin / kv_util_bin.
E. Time-holdout-first promotion classifier.

Writes ``data/external/forecasting/cara_latency_forecaster_v1/calibration_tail_safety_summary.json``.

The script never modifies the scheduler / robust energy engine /
controllers / production defaults. All training is bounded
(HistGradientBoostingRegressor with ``max_iter=300``). Raw CARA data
stays gitignored.
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
    FINAL_STATUS_VALUES,
    PROMOTION_THRESHOLDS,
    BaselineFallbackGate,
    ConservativeMultiplierCalibration,
    QuantileResidualCalibration,
    SplitConformalUpperBound,
    classify_tail_status,
    tail_safety_metrics,
    time_train_calibration_split,
    train_calibration_test_split,
)
from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    LEAKAGE_TARGET_FIELDS,
    TARGETS,
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

logger = logging.getLogger(__name__)


CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)

OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1"
    / "calibration_tail_safety_summary.json"
)

HOLDOUT_BY_INSTANCE_TYPE = ("qwen2.5-7b_a30",)


# Subgroup row-count threshold below which p95/p99 metrics are
# INSUFFICIENT_SAMPLE (mission spec PHASE D).
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


# ---------------------------------------------------------------------------
# Per-quantile baseline (per_instance_type p{q}) — the strongest realistic
# baseline per the mission spec.
# ---------------------------------------------------------------------------


def _fit_baseline(X_train, y_train, group_train_instance_type, quantile):
    return GroupConstantQuantileBaseline(quantile=quantile * 100.0).fit(
        X_train, y_train, group_keys_train=group_train_instance_type,
    )


def _baseline_predict(b, X, group_keys_predict):
    return b.predict(X, group_keys_predict=group_keys_predict)


# ---------------------------------------------------------------------------
# Per-(target, quantile, holdout) evaluation
# ---------------------------------------------------------------------------


def _build_subgroups(group_keys: dict, indices: np.ndarray) -> dict:
    return {k: v[indices] for k, v in group_keys.items()}


def _per_subgroup_audit(
    *, y_test, baseline_pred, calibrated_pred,
    group_arrays: dict, target_coverage: float,
) -> tuple[dict, bool, bool]:
    """Per-subgroup safety audit. Returns
    ``(audit_dict, has_regression, has_undercoverage)``."""
    audit: dict = {}
    has_regression = False
    has_undercoverage = False
    for key in ("instance_type", "gpu_type", "model_size",
                "prompt_token_bin", "queue_depth_bin", "kv_util_bin"):
        groups = group_arrays.get(key)
        if groups is None:
            continue
        per_subgroup: dict = {}
        for g in sorted(set(groups.tolist())):
            mask = groups == g
            n = int(mask.sum())
            if n == 0:
                continue
            yt = y_test[mask]
            base_pl = pinball_loss(yt, baseline_pred[mask], target_coverage)
            cal_pl = pinball_loss(yt, calibrated_pred[mask], target_coverage)
            if not (np.isfinite(base_pl) and base_pl > 0):
                improvement = 0.0
            else:
                improvement = 100.0 * (base_pl - cal_pl) / base_pl
            cov = float((yt <= calibrated_pred[mask]).mean()) if n else float("nan")
            status = "PASS"
            if n < MIN_SUBGROUP_ROWS:
                status = "INSUFFICIENT_SAMPLE"
            elif improvement < SUBGROUP_REGRESSION_THRESHOLD_PCT:
                status = "REGRESSION"
                has_regression = True
            elif cov < target_coverage - 0.02:
                status = "UNDERCOVERED"
                has_undercoverage = True
            per_subgroup[str(g)] = {
                "row_count": n,
                "baseline_pinball_loss": base_pl,
                "calibrated_pinball_loss": cal_pl,
                "improvement_pct": improvement,
                "empirical_coverage": cov,
                "status": status,
            }
        audit[key] = per_subgroup
    return audit, has_regression, has_undercoverage


def _eval_one_holdout(
    *, target, quantile, holdout_name, train_idx, test_idx,
    X, y, group_keys, timestamps, raw_alpha_summary: dict,
) -> dict:
    """Train + calibrate + evaluate one (target, quantile, holdout) cell."""
    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    g_train_full = _build_subgroups(group_keys, train_idx)
    X_test = X[test_idx]
    y_test = y[test_idx]
    g_test = _build_subgroups(group_keys, test_idx)

    # Calibration split: time-holdout uses a *temporal* calibration tail,
    # other holdouts use a random calibration block.
    if holdout_name == "time_holdout":
        sub_train, sub_cal = time_train_calibration_split(
            train_idx, timestamps,
            calibration_frac=0.25,
        )
    else:
        sub_train, sub_cal = train_calibration_test_split(
            train_idx, calibration_frac=0.25,
        )

    X_sub_train = X[sub_train]
    y_sub_train = y[sub_train]
    g_sub_train = _build_subgroups(group_keys, sub_train)
    X_cal = X[sub_cal]
    y_cal = y[sub_cal]

    # Raw HGB quantile model — trained on sub_train only (calibration split
    # carved off).
    raw = HistGradientBoostingQuantileForecaster(quantile=quantile).fit(
        X_sub_train, y_sub_train,
    )
    raw_pred_test = raw.predict(X_test)
    raw_pred_cal = raw.predict(X_cal)

    # Strongest baseline (per_instance_type_p{q}) — fit on sub_train so the
    # calibration block isn't leaked into the baseline.
    baseline = _fit_baseline(
        X_sub_train, y_sub_train,
        g_sub_train["instance_type"], quantile,
    )
    base_pred_test = _baseline_predict(baseline, X_test,
                                       g_test["instance_type"])

    raw_pinball_test = pinball_loss(y_test, raw_pred_test, quantile)
    base_pinball_test = pinball_loss(y_test, base_pred_test, quantile)
    raw_alpha_pct = (
        100.0 * (base_pinball_test - raw_pinball_test) / base_pinball_test
        if np.isfinite(base_pinball_test) and base_pinball_test > 0 else 0.0
    )

    # --- Calibration variants ----------------------------------------------
    calibrators: dict = {}
    calibrated_preds: dict = {}

    cm = ConservativeMultiplierCalibration(
        target_quantile=quantile, base=raw,
    ).fit(X_cal, y_cal)
    calibrators["conservative_multiplier"] = cm
    calibrated_preds["conservative_multiplier"] = cm.predict(X_test)

    qr = QuantileResidualCalibration(
        target_quantile=quantile, base=raw,
    ).fit(X_cal, y_cal)
    calibrators["quantile_residual"] = qr
    calibrated_preds["quantile_residual"] = qr.predict(X_test)

    sc = SplitConformalUpperBound(alpha=quantile, base=raw).fit(X_cal, y_cal)
    calibrators["split_conformal_upper_bound"] = sc
    calibrated_preds["split_conformal_upper_bound"] = sc.predict(X_test)

    # Baseline-floor wrapper (post-calibration).
    floor_pred, floor_used = BaselineFallbackGate(
        policy="floor_at_baseline",
        ml=type("Wrapped", (), {"predict": lambda _self, X: sc.predict(X)})(),
        baseline=type("Wrapped", (), {
            "predict": lambda _self, X: _baseline_predict(
                baseline, X, g_test["instance_type"],
            ),
        })(),
    ).predict_with_fallback(X_test)
    calibrated_preds["split_conformal_with_baseline_floor"] = floor_pred
    fallback_fired_rate = float(np.asarray(floor_used).mean())

    # Pick the "preferred calibrated model" — split-conformal-with-floor by
    # default for p95/p99; conservative_multiplier for p50.
    if quantile <= 0.5:
        preferred = "conservative_multiplier"
    else:
        preferred = "split_conformal_with_baseline_floor"
    preferred_pred = calibrated_preds[preferred]
    preferred_pinball = pinball_loss(y_test, preferred_pred, quantile)
    preferred_alpha_pct = (
        100.0 * (base_pinball_test - preferred_pinball) / base_pinball_test
        if np.isfinite(base_pinball_test) and base_pinball_test > 0 else 0.0
    )

    safety = tail_safety_metrics(
        y_test, preferred_pred, target_coverage=quantile,
    )

    sub_audit, has_subgroup_regression, has_subgroup_undercoverage = (
        _per_subgroup_audit(
            y_test=y_test, baseline_pred=base_pred_test,
            calibrated_pred=preferred_pred,
            group_arrays=g_test, target_coverage=quantile,
        )
    )

    # Each calibration variant's pinball loss + coverage on test.
    variant_metrics = {}
    for name, pred in calibrated_preds.items():
        variant_metrics[name] = {
            "pinball_loss": pinball_loss(y_test, pred, quantile),
            "empirical_coverage": float((y_test <= pred).mean()),
            "alpha_pct_vs_baseline": (
                100.0 * (base_pinball_test - pinball_loss(y_test, pred,
                                                          quantile))
                / base_pinball_test
                if np.isfinite(base_pinball_test) and base_pinball_test > 0
                else 0.0
            ),
        }

    return {
        "target": target,
        "target_family": _target_family(target),
        "quantile": quantile,
        "holdout": holdout_name,
        "n_train_full": int(train_idx.size),
        "n_sub_train": int(sub_train.size),
        "n_calibration": int(sub_cal.size),
        "n_test": int(test_idx.size),
        "raw_model_metrics": {
            "pinball_loss": raw_pinball_test,
            "alpha_pct_vs_baseline": raw_alpha_pct,
        },
        "strongest_baseline_metrics": {
            "name": f"per_instance_type_p{int(quantile * 100)}",
            "pinball_loss": base_pinball_test,
        },
        "calibration_variants": variant_metrics,
        "preferred_calibration_method": preferred,
        "calibrated_model_metrics": {
            "pinball_loss": preferred_pinball,
            "alpha_pct_vs_baseline": preferred_alpha_pct,
        },
        "calibration_diagnostics": {
            name: c.diagnostics() for name, c in calibrators.items()
        },
        "tail_safety_metrics": safety,
        "subgroup_audit": sub_audit,
        "has_subgroup_regression": has_subgroup_regression,
        "has_subgroup_undercoverage": has_subgroup_undercoverage,
        "fallback_used_rate_on_test": fallback_fired_rate,
        "leakage_features_excluded": sorted(LEAKAGE_TARGET_FIELDS),
        "no_test_label_calibration": True,
    }


def _target_family(target: str) -> str:
    return "ttft" if "ttft" in target else "e2e"


def _classify_per_target_quantile(
    *, target_family, quantile, per_holdout_results,
) -> dict:
    """Apply Phase E gating + classify_tail_status."""
    per = {h["holdout"]: h for h in per_holdout_results}
    time_h = per.get("time_holdout")
    rand_h = per.get("random_holdout")
    by_h = per.get("holdout_by_instance_type")

    if time_h is None:
        return {
            "final_status": "diagnostic_only",
            "reason": "time_holdout missing",
        }

    fallback_required_on_time = bool(
        time_h["fallback_used_rate_on_test"] > 0.25
    )
    final, reason = classify_tail_status(
        target_family=target_family,
        quantile=quantile,
        time_holdout_improvement_pct=time_h["calibrated_model_metrics"][
            "alpha_pct_vs_baseline"],
        random_holdout_improvement_pct=(
            rand_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
            if rand_h else 0.0
        ),
        by_instance_holdout_improvement_pct=(
            by_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"]
            if by_h else 0.0
        ),
        empirical_coverage=time_h["tail_safety_metrics"]["empirical_coverage"],
        undercoverage_rate=time_h["tail_safety_metrics"]["undercoverage_rate"],
        has_subgroup_regression=time_h["has_subgroup_regression"],
        has_subgroup_undercoverage=time_h["has_subgroup_undercoverage"],
        fallback_required_on_time=fallback_required_on_time,
        leakage_free=True,
        no_test_label_calibration=True,
    )
    return {
        "final_status": final,
        "reason": reason,
        "time_holdout_pinball_improvement_pct":
            time_h["calibrated_model_metrics"]["alpha_pct_vs_baseline"],
        "time_holdout_empirical_coverage":
            time_h["tail_safety_metrics"]["empirical_coverage"],
        "time_holdout_undercoverage_rate":
            time_h["tail_safety_metrics"]["undercoverage_rate"],
        "fallback_required_on_time": fallback_required_on_time,
        "subgroup_regression_on_time": time_h["has_subgroup_regression"],
        "subgroup_undercoverage_on_time":
            time_h["has_subgroup_undercoverage"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    if not CARA_TRAIN_FLAT.exists():
        print(f"[calib] CARA train_flat missing: {CARA_TRAIN_FLAT}",
              file=sys.stderr)
        return 2

    print(f"[calib] loading {CARA_TRAIN_FLAT}")
    rows = _load_jsonl(CARA_TRAIN_FLAT, limit=args.limit_rows)
    print(f"[calib] loaded {len(rows)} rows")

    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X, _, group_keys = build_feature_matrix(rows, spec)

    timestamps = np.array(
        [r.get("prediction_timestamp_s") for r in rows], dtype=np.float64,
    )

    # Three holdouts.
    holdouts: dict = {}
    r_train, r_test = random_holdout(X.shape[0])
    holdouts["random_holdout"] = (r_train, r_test)
    it_train, it_test = holdout_by_group(
        group_keys["instance_type"], HOLDOUT_BY_INSTANCE_TYPE,
    )
    if it_test.size > 0:
        holdouts["holdout_by_instance_type"] = (it_train, it_test)
    if np.isfinite(timestamps).sum() >= 0.95 * len(rows):
        t_train, t_test = time_holdout(timestamps)
        holdouts["time_holdout"] = (t_train, t_test)

    print(f"[calib] holdouts ready: {list(holdouts.keys())}")

    payload: dict = {
        "doc_version": "cara_latency_calibration_tail_safety_v1",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "shadow_only": True,
        "no_production_claim": True,
        "dataset": "asdwb/cara_latency_prediction",
        "config": "train_flat",
        "row_count": int(X.shape[0]),
        "holdouts_used": sorted(holdouts.keys()),
        "leakage_features_excluded": sorted(LEAKAGE_TARGET_FIELDS),
        "promotion_thresholds": {
            f"{k[0]}_p{int(k[1] * 100)}": v
            for k, v in PROMOTION_THRESHOLDS.items()
        },
        "evaluated_at_s": time.time(),
        "per_target": {},
    }

    raw_alpha_summary: dict = {}
    for target in TARGETS:
        y = extract_target(rows, target)
        per_target: dict = {"per_quantile": {}}
        for quantile in (0.50, 0.95, 0.99):
            per_holdout: list[dict] = []
            for holdout_name, (train_idx, test_idx) in holdouts.items():
                print(f"\n[calib] {target}  q={quantile}  holdout={holdout_name}")
                row = _eval_one_holdout(
                    target=target, quantile=quantile,
                    holdout_name=holdout_name, train_idx=train_idx,
                    test_idx=test_idx, X=X, y=y, group_keys=group_keys,
                    timestamps=timestamps, raw_alpha_summary=raw_alpha_summary,
                )
                per_holdout.append(row)
                print(f"  raw alpha={row['raw_model_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                      f"calibrated alpha={row['calibrated_model_metrics']['alpha_pct_vs_baseline']:+.2f}%  "
                      f"coverage={row['tail_safety_metrics']['empirical_coverage']:.4f}  "
                      f"undercov={row['tail_safety_metrics']['undercoverage_rate']:.4f}  "
                      f"fallback={row['fallback_used_rate_on_test']:.3f}")
            decision = _classify_per_target_quantile(
                target_family=_target_family(target), quantile=quantile,
                per_holdout_results=per_holdout,
            )
            per_target["per_quantile"][f"p{int(quantile * 100)}"] = {
                "per_holdout": per_holdout,
                "decision": decision,
            }
            print(f"  -> {target} p{int(quantile*100)} final_status="
                  f"{decision['final_status']}  reason={decision['reason']}")
        payload["per_target"][target] = per_target

    # Final decision table.
    table: list[dict] = []
    for target, pt in payload["per_target"].items():
        for q_label, cell in pt["per_quantile"].items():
            decision = cell["decision"]
            time_holdout_payload = next(
                (h for h in cell["per_holdout"]
                 if h["holdout"] == "time_holdout"),
                None,
            )
            raw_alpha_time = (
                time_holdout_payload["raw_model_metrics"]["alpha_pct_vs_baseline"]
                if time_holdout_payload else None
            )
            cal_alpha_time = (
                time_holdout_payload["calibrated_model_metrics"][
                    "alpha_pct_vs_baseline"]
                if time_holdout_payload else None
            )
            coverage_time = (
                time_holdout_payload["tail_safety_metrics"]["empirical_coverage"]
                if time_holdout_payload else None
            )
            fallback_used = (
                time_holdout_payload["fallback_used_rate_on_test"]
                if time_holdout_payload else None
            )
            raw_status = (
                "improvement_>=5pct" if raw_alpha_time is not None and
                raw_alpha_time >= 5.0 else
                ("regression" if raw_alpha_time is not None and
                 raw_alpha_time < 0 else "parity")
            )
            table.append({
                "target": target,
                "quantile": q_label,
                "raw_status_time_holdout": raw_status,
                "raw_alpha_pct_time_holdout": raw_alpha_time,
                "calibrated_alpha_pct_time_holdout": cal_alpha_time,
                "time_holdout_empirical_coverage": coverage_time,
                "fallback_used_rate_on_time": fallback_used,
                "final_status": decision["final_status"],
                "reason": decision["reason"],
            })

    payload["final_decision_table"] = table
    payload["final_status_values_allowed"] = sorted(FINAL_STATUS_VALUES)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[calib] wrote {args.out_path}")

    print("\n[calib] final decision table:")
    for row in table:
        print(f"  {row['target']:25s} {row['quantile']:5s} "
              f"raw_alpha_time={row['raw_alpha_pct_time_holdout']:+7.2f}%  "
              f"cal_alpha_time={row['calibrated_alpha_pct_time_holdout']:+7.2f}%  "
              f"cov={row['time_holdout_empirical_coverage']:.3f}  "
              f"final={row['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
