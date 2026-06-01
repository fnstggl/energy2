"""Shadow-only TTFT p50 forecaster integration.

This module wires the ``shadow_ready`` TTFT p50 forecaster
(``docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md``) into a **shadow
evaluation path only**. It NEVER changes a routing, placement,
autoscaling, queue-admission, or frontier-rho decision. It only
*computes and logs* a prediction next to a deterministic baseline so the
two can be compared offline.

Binding contract:

- ``shadow_only = True`` on every emitted record.
- ``executable_in_real_cluster = False`` on every emitted record.
- ``no_control_action_taken = True`` on every batch summary.
- The shadow path is **enabled by default** (active shadow evaluation),
  but may be disabled via ``ShadowConfig(enabled=False)``. When disabled
  the predictor emits nothing.

The shadow predictor wraps an already-fit base TTFT-p50 model plus the
strongest deterministic baseline (per-instance-type p50). It produces,
per request:

- ``ttft_p50_prediction_s`` — the ML forecaster's p50 prediction.
- ``baseline_ttft_p50_prediction_s`` — the per-instance-type p50.
- ``prediction_delta_s`` — ML minus baseline.
- ``model_version`` / ``feature_version`` — provenance tags.

Nothing in this module imports the scheduler, the frontier controller,
or any executor. The class cannot mutate cluster state because it has no
handle to any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Provenance tags — bumped when the model / feature pipeline changes.
MODEL_VERSION = "cara_ttft_p50_hgb_quantile_v1"
FEATURE_VERSION = "cara_latency_features_predicted_only_v1"


@dataclass(frozen=True)
class ShadowConfig:
    """Shadow-evaluation configuration.

    ``enabled`` defaults to ``True`` — active shadow evaluation is the
    default behaviour. Set ``enabled=False`` to disable (the predictor
    then emits no records and the summary reports zero rows evaluated).
    """

    enabled: bool = True
    # Hard invariant — shadow mode can never be flipped to executable.
    executable_in_real_cluster: bool = False


@dataclass(frozen=True)
class ShadowPrediction:
    """One shadow TTFT-p50 prediction record. Recommendation-only."""

    request_id: Optional[str]
    instance_type: Optional[str]
    ttft_p50_prediction_s: float
    baseline_ttft_p50_prediction_s: float
    prediction_delta_s: float
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    shadow_only: bool = True
    executable_in_real_cluster: bool = False

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "instance_type": self.instance_type,
            "ttft_p50_prediction_s": self.ttft_p50_prediction_s,
            "baseline_ttft_p50_prediction_s": self.baseline_ttft_p50_prediction_s,
            "prediction_delta_s": self.prediction_delta_s,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "shadow_only": self.shadow_only,
            "executable_in_real_cluster": self.executable_in_real_cluster,
        }


class TTFTp50ShadowPredictor:
    """Shadow-only TTFT-p50 predictor.

    Wraps a fitted ML model (``ml_model.predict(X) -> np.ndarray``) and a
    baseline callable (``baseline_predict(X, instance_types) -> np.ndarray``).
    The predictor produces shadow records ONLY; it has no method that can
    change a control decision.
    """

    def __init__(
        self,
        ml_model,
        baseline_predict,
        config: Optional[ShadowConfig] = None,
    ):
        self.ml_model = ml_model
        self.baseline_predict = baseline_predict
        self.config = config or ShadowConfig()
        # Defensive: the predictor must never be constructed in an
        # executable configuration.
        if self.config.executable_in_real_cluster:
            raise ValueError(
                "TTFTp50ShadowPredictor cannot be executable_in_real_cluster; "
                "shadow mode is recommendation-only"
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def predict_shadow(
        self, X, *, instance_types, request_ids=None,
    ) -> list[ShadowPrediction]:
        """Return shadow predictions. Empty list when disabled.

        This is the ONLY public prediction method. It returns records;
        it does not — and cannot — take any control action.
        """
        if not self.config.enabled:
            return []
        ml_pred = np.asarray(self.ml_model.predict(X), dtype=np.float64)
        base_pred = np.asarray(
            self.baseline_predict(X, instance_types), dtype=np.float64,
        )
        n = ml_pred.shape[0]
        out: list[ShadowPrediction] = []
        for i in range(n):
            rid = (request_ids[i] if request_ids is not None
                   and i < len(request_ids) else None)
            it = (instance_types[i] if i < len(instance_types) else None)
            out.append(ShadowPrediction(
                request_id=rid,
                instance_type=(None if it is None else str(it)),
                ttft_p50_prediction_s=float(ml_pred[i]),
                baseline_ttft_p50_prediction_s=float(base_pred[i]),
                prediction_delta_s=float(ml_pred[i] - base_pred[i]),
            ))
        return out


def summarize_shadow_batch(
    predictions: list[ShadowPrediction],
    y_true: Optional[np.ndarray] = None,
    *,
    holdout_name: str = "unspecified",
    enabled: bool = True,
) -> dict:
    """Aggregate a batch of shadow predictions into a summary dict.

    When ``y_true`` is supplied, also reports prediction-quality metrics
    (these are evaluation-only and never feed a control loop).
    """
    n = len(predictions)
    summary = {
        "holdout": holdout_name,
        "shadow_enabled": enabled,
        "shadow_only": True,
        "executable_in_real_cluster": False,
        "no_control_action_taken": True,
        "rows_evaluated": n,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
    }
    if n == 0:
        summary["prediction_coverage"] = 0.0
        return summary

    ml = np.array([p.ttft_p50_prediction_s for p in predictions], float)
    base = np.array([p.baseline_ttft_p50_prediction_s for p in predictions], float)
    deltas = np.array([p.prediction_delta_s for p in predictions], float)
    summary["prediction_coverage"] = float(np.isfinite(ml).mean())
    summary["mean_prediction_delta_s"] = float(np.nanmean(deltas))
    summary["median_prediction_delta_s"] = float(np.nanmedian(deltas))
    summary["mean_ml_prediction_s"] = float(np.nanmean(ml))
    summary["mean_baseline_prediction_s"] = float(np.nanmean(base))

    if y_true is not None:
        yt = np.asarray(y_true, dtype=np.float64)
        mask = ~(np.isnan(yt) | np.isnan(ml))
        if mask.any():
            ml_mae = float(np.abs(ml[mask] - yt[mask]).mean())
            base_mae = float(np.abs(base[mask] - yt[mask]).mean())
            # p50 pinball comparison.
            summary["ml_p50_pinball"] = _pinball(yt[mask], ml[mask], 0.5)
            summary["baseline_p50_pinball"] = _pinball(yt[mask], base[mask], 0.5)
            summary["ml_mae"] = ml_mae
            summary["baseline_mae"] = base_mae
            b = summary["baseline_p50_pinball"]
            m = summary["ml_p50_pinball"]
            summary["pinball_improvement_pct"] = (
                100.0 * (b - m) / b if b > 0 else 0.0
            )
    return summary


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.maximum(quantile * diff, (quantile - 1.0) * diff).mean())
