"""Honest candidate-vs-active model promotion for the continuous learning loop.

This replaces the previous unsound logic (which compared a freshly-trained
in-engine model against a stale scalar). Here we:

  1. Split data into train (< eval_start) and a leakage-free holdout window.
  2. Train a candidate forecaster on the train split.
  3. Load the current ACTIVE model from the registry + artifact store.
  4. Evaluate BOTH models on the SAME holdout window (forecast accuracy).
  5. Promote the candidate only if it genuinely beats the active model.
  6. Persist the artifact, registry row, and an append-only promotion decision.

Everything is scoped by (model_type, scope=customer_id, pilot_id) so pilots
learn independently and never contaminate each other.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from aurelius.ml.forecast_evaluator import ForecastEvaluator, compare_models
from aurelius.models import EnergyPrice

logger = logging.getLogger(__name__)


def dataset_hash(price_df: pd.DataFrame) -> str:
    """Deterministic sha256 (first 16 hex) over (timestamp, region, price) rows.

    Two datasets with identical price content produce the same hash, enabling
    exact reproduction of which data a model was trained on.
    """
    if price_df.empty:
        return "empty"
    df = price_df[["timestamp", "region", "price_per_mwh"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "region"]).reset_index(drop=True)
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_forecaster(forecaster: Any, path: str) -> None:
    joblib.dump(forecaster, path)


def load_forecaster(path: str) -> Any:
    return joblib.load(path)


def _df_to_prices(df: pd.DataFrame) -> list[EnergyPrice]:
    out = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        out.append(EnergyPrice(
            timestamp=ts.to_pydatetime(),
            region=str(row["region"]),
            price_per_mwh=float(row["price_per_mwh"]),
        ))
    return out


def evaluate_on_holdout(forecaster, train_df, holdout_df, context_hours=336):
    """Evaluate a fitted forecaster on a holdout window (forecast accuracy).

    Returns an EvaluationResult (MAE/MAPE/RMSE/coverage), or None if no overlap.
    """
    holdout_actuals = _df_to_prices(holdout_df)
    if not holdout_actuals:
        return None
    eval_start = pd.to_datetime(holdout_df["timestamp"], utc=True).min()
    context_start = eval_start - timedelta(hours=context_hours)
    ctx_mask = pd.to_datetime(train_df["timestamp"], utc=True) >= context_start
    context = _df_to_prices(train_df[ctx_mask])
    evaluator = ForecastEvaluator()
    return evaluator.evaluate_from_model(
        forecaster, holdout_actuals, recent_context=context or None
    )


def run_model_update(
    price_df: pd.DataFrame,
    regions: list[str],
    forecaster_cls,
    forecaster_config,
    store,                       # TimeSeriesStore (may be disabled / None)
    artifact_store,              # ArtifactStore (always available)
    eval_days: int = 7,
    min_train_rows: int = 200,
    primary_metric: str = "mae",
    min_improvement_pct: float = 1.0,
    scope: str = "global",
    pilot_id: str = "unknown",
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Train a candidate, compare it to the active model on a held-out window,
    and promote only if it genuinely wins. Returns a summary dict.

    When `store` is disabled (no DATABASE_URL) there is no persistent registry,
    so the candidate is evaluated and the decision is computed, but nothing is
    persisted and no active model can be loaded (candidate is reported as the
    de-facto model). This keeps local/dev runs fully functional.
    """
    summary: dict = {"status": "ok", "promoted": False, "reason": "", "scope": scope,
                     "pilot_id": pilot_id, "primary_metric": primary_metric}

    df = price_df[price_df["region"].isin(regions)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if len(df) < min_train_rows:
        return {"status": "skipped", "reason": "insufficient_data", "rows": len(df)}

    max_ts = df["timestamp"].max()
    eval_start = max_ts - timedelta(days=eval_days)
    train_df = df[df["timestamp"] < eval_start]
    holdout_df = df[df["timestamp"] >= eval_start]
    if len(train_df) < min_train_rows or holdout_df.empty:
        return {"status": "skipped", "reason": "insufficient_split",
                "train_rows": len(train_df), "holdout_rows": len(holdout_df)}

    # --- Train candidate on the train split (leakage-free vs holdout) ---
    candidate = forecaster_cls(forecaster_config) if forecaster_config is not None else forecaster_cls()
    candidate.fit(_df_to_prices(train_df))
    cand_eval = evaluate_on_holdout(candidate, train_df, holdout_df)
    if cand_eval is None:
        return {"status": "error", "reason": "candidate_eval_failed"}

    ds_hash = dataset_hash(train_df)
    summary["candidate_metrics"] = cand_eval.to_dict()
    summary["training_dataset_hash"] = ds_hash
    summary["training_rows"] = len(train_df)

    store_enabled = store is not None and getattr(store, "enabled", False)

    # --- Load and evaluate the current active model on the SAME holdout ---
    active_row = store.get_active_model("price", scope, pilot_id) if store_enabled else None
    active_eval = None
    if active_row and active_row.get("artifact_uri"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp_path = tmp.name
            artifact_store.get(active_row["artifact_uri"], tmp_path)
            active_model = load_forecaster(tmp_path)
            active_eval = evaluate_on_holdout(active_model, train_df, holdout_df)
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:  # corrupt/missing artifact must not crash the loop
            logger.warning("Active model load/eval failed (%s); treating as no active model", exc)
            active_eval = None

    # --- Decide ---
    if active_eval is None:
        promote = True
        reason = "no_active_model"
        cand_value = getattr(cand_eval, primary_metric)
        active_value = None
    else:
        cmp = compare_models(cand_eval, active_eval, primary_metric=primary_metric,
                             min_improvement_pct=min_improvement_pct)
        promote = cmp.promote
        reason = cmp.reason
        cand_value = cmp.candidate_value
        active_value = cmp.current_value

    summary["promoted"] = bool(promote)
    summary["reason"] = reason
    summary["candidate_value"] = None if cand_value is None else float(cand_value)
    summary["active_value"] = None if active_value is None else float(active_value)
    promote = bool(promote)

    if dry_run:
        summary["note"] = "dry_run: no artifact saved, no registry change"
        return summary

    # --- Persist the candidate artifact + registry row + decision ---
    model_id = uuid.uuid4().hex[:16]
    version = "v_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    with tempfile.TemporaryDirectory() as td:
        local = str(Path(td) / "model.joblib")
        save_forecaster(candidate, local)
        key = f"models/price/{scope}/{pilot_id}/{model_id}/model.joblib"
        artifact_uri = artifact_store.put(key, local)
    summary["model_id"] = model_id
    summary["artifact_uri"] = artifact_uri

    if store_enabled:
        store.register_model(
            model_id=model_id, version=version, artifact_uri=artifact_uri,
            model_type="price", scope=scope, pilot_id=pilot_id, status="candidate",
            training_dataset_hash=ds_hash, training_rows=len(train_df),
            eval_metrics=cand_eval.to_dict(),
            parent_model_id=active_row["model_id"] if active_row else None,
            run_id=run_id,
        )
        if promote:
            store.promote_model(model_id)
        store.record_promotion_decision(
            decision="promote" if promote else "reject",
            model_type="price", scope=scope, pilot_id=pilot_id,
            model_id=model_id, run_id=run_id, primary_metric=primary_metric,
            candidate_value=cand_value, active_value=active_value, reason=reason,
        )
    else:
        summary["note"] = "store_disabled: artifact saved, no registry persistence"

    return summary
