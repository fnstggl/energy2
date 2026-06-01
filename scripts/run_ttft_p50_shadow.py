#!/usr/bin/env python3
"""Produce the TTFT p50 shadow-evaluation summary.

Trains the ``shadow_ready`` TTFT p50 forecaster on the CARA train_flat
train split, wraps it in the shadow-only predictor
(``aurelius/forecasting/ttft_shadow.py``), runs it over each holdout's
test split, and writes a summary proving:

- shadow mode took NO control action,
- predictions are logged only,
- TTFT p50 is the only model marked shadow_ready.

Writes ``data/external/forecasting/cara_latency_forecaster_v1/ttft_p50_shadow_summary.json``.

No scheduler / controller change. Raw CARA data gitignored.
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
)
from aurelius.forecasting.ttft_shadow import (  # noqa: E402
    ShadowConfig,
    TTFTp50ShadowPredictor,
    summarize_shadow_batch,
)

CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)
OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1" / "ttft_p50_shadow_summary.json"
)
TARGET = "actual_ttft_s"
HOLDOUT_BY_INSTANCE_TYPE = ("qwen2.5-7b_a30",)


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--disabled", action="store_true",
                   help="Run with shadow DISABLED to prove zero rows emitted.")
    args = p.parse_args(argv)

    if not CARA_TRAIN_FLAT.exists():
        print(f"[shadow] CARA train_flat missing: {CARA_TRAIN_FLAT}",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(CARA_TRAIN_FLAT, limit=args.limit_rows)
    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X, _, group_keys = build_feature_matrix(rows, spec)
    y = extract_target(rows, TARGET)
    request_ids = [r.get("request_id") for r in rows]
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

    config = ShadowConfig(enabled=not args.disabled)

    payload: dict = {
        "doc_version": "ttft_p50_shadow_summary_v1",
        "dataset": "asdwb/cara_latency_prediction",
        "config": "train_flat",
        "row_count": int(X.shape[0]),
        "shadow_enabled": config.enabled,
        "shadow_only": True,
        "executable_in_real_cluster": False,
        "no_control_action_taken": True,
        "ttft_p50_status": "shadow_ready",
        "other_models_shadow_ready": [],
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "production_claim": False,
        "no_production_claim": True,
        "evaluated_at_s": time.time(),
        "per_holdout": {},
    }

    for hname, (tr, te) in holdouts.items():
        # Train TTFT p50 + baseline on the train split only.
        g_tr = {k: v[tr] for k, v in group_keys.items()}
        g_te = {k: v[te] for k, v in group_keys.items()}
        ml = HistGradientBoostingQuantileForecaster(quantile=0.50).fit(
            X[tr], y[tr])
        base = GroupConstantQuantileBaseline(quantile=50.0).fit(
            X[tr], y[tr], group_keys_train=g_tr["instance_type"])

        def _baseline_predict(Xq, instance_types, _base=base):
            return _base.predict(
                Xq, group_keys_predict=np.asarray(instance_types, dtype=object))

        predictor = TTFTp50ShadowPredictor(
            ml_model=ml, baseline_predict=_baseline_predict, config=config)

        it_test = g_te["instance_type"]
        rids_test = [request_ids[i] for i in te]
        preds = predictor.predict_shadow(
            X[te], instance_types=it_test, request_ids=rids_test)
        summary = summarize_shadow_batch(
            preds, y_true=y[te], holdout_name=hname, enabled=config.enabled)
        payload["per_holdout"][hname] = summary
        cov = summary.get("prediction_coverage", 0.0)
        imp = summary.get("pinball_improvement_pct")
        print(f"[shadow] {hname:28s} rows={summary['rows_evaluated']:6d}  "
              f"coverage={cov:.3f}  "
              f"pinball_improvement={imp:+.2f}%" if imp is not None
              else f"[shadow] {hname}: rows={summary['rows_evaluated']}")

    # Aggregate time-holdout performance (the binding gate).
    th = payload["per_holdout"].get("time_holdout", {})
    payload["time_holdout_pinball_improvement_pct"] = th.get(
        "pinball_improvement_pct")
    payload["shadow_ready"] = True  # TTFT p50 is shadow_ready per PR #127.

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[shadow] wrote {args.out_path}")
    print(f"[shadow] shadow_enabled={config.enabled}  "
          f"no_control_action_taken={payload['no_control_action_taken']}  "
          f"ttft_p50_status={payload['ttft_p50_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
