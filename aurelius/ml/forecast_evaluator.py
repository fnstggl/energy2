"""Leakage-free forecast evaluation for price and carbon quantile models.

Computes calibration-aware accuracy metrics on a strict temporal holdout.

Metrics computed:
  - MAPE  : mean absolute percentage error on p50 vs actuals
  - RMSE  : root mean squared error on p50 vs actuals
  - MAE   : mean absolute error on p50 vs actuals
  - p50_bias : mean signed error (p50 - actual); positive = systematic over-forecast
  - p90_coverage : empirical fraction of actuals <= p90 prediction
  - calibration_error : |p90_coverage - 0.90|  (should be near 0 for a well-calibrated model)
  - downside_risk : mean(max(0, actual - p50)) / mean(actual)  — unexpected upside fraction
  - savings_lift  : % reduction in mean absolute error vs naive (mean of training set)

All metrics are computed solely on held-out data that was NOT used for training.
The caller is responsible for the temporal split; this module never touches training data.

Usage:
    from aurelius.ml.forecast_evaluator import ForecastEvaluator, EvaluationResult

    evaluator = ForecastEvaluator()
    result = evaluator.evaluate(
        actuals=actuals_list,        # list of (timestamp, region, actual_value)
        p50_forecasts=p50_list,      # list of (timestamp, region, p50)
        p90_forecasts=p90_list,      # list of (timestamp, region, p90)
        training_mean=train_mean,    # float, mean of training target (for savings_lift)
    )
    print(result.to_dict())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ForecastPoint:
    """A single (timestamp, region) forecast/actual observation."""
    timestamp: datetime
    region: str
    value: float


@dataclass
class EvaluationResult:
    """Holdout evaluation metrics for a single forecaster.

    All metrics are computed on actuals only (no training data contamination).
    """
    n_samples: int
    mape: float            # mean absolute % error on p50; lower is better
    rmse: float            # root mean squared error on p50; lower is better
    mae: float             # mean absolute error on p50; lower is better
    p50_bias: float        # mean(p50 - actual); near 0 is ideal
    p90_coverage: float    # fraction of actuals <= p90; should be ~0.90
    calibration_error: float  # |p90_coverage - 0.90|; near 0 is ideal
    downside_risk: float   # mean unexpected upside as fraction of mean actual
    savings_lift: float    # % MAE reduction vs naive baseline (training mean)
    regions: list[str] = field(default_factory=list)
    per_region: dict[str, dict] = field(default_factory=dict)  # metrics per region

    @property
    def is_well_calibrated(self) -> bool:
        """True if p90 calibration error < 5 percentage points."""
        return self.calibration_error < 0.05

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mape": round(self.mape, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "p50_bias": round(self.p50_bias, 4),
            "p90_coverage": round(self.p90_coverage, 4),
            "calibration_error": round(self.calibration_error, 4),
            "downside_risk": round(self.downside_risk, 4),
            "savings_lift_pct": round(self.savings_lift * 100, 2),
            "is_well_calibrated": self.is_well_calibrated,
            "regions": self.regions,
            "per_region": self.per_region,
        }

    def is_better_than(self, other: "EvaluationResult", primary_metric: str = "mape") -> bool:
        """Return True if this result beats *other* on the primary metric.

        Lower is better for MAPE, RMSE, MAE, calibration_error, downside_risk.
        Higher is better for savings_lift, p90_coverage (closer to 0.90).
        """
        if primary_metric == "mape":
            return self.mape < other.mape
        elif primary_metric == "rmse":
            return self.rmse < other.rmse
        elif primary_metric == "mae":
            return self.mae < other.mae
        elif primary_metric == "calibration_error":
            return self.calibration_error < other.calibration_error
        elif primary_metric == "savings_lift":
            return self.savings_lift > other.savings_lift
        else:
            raise ValueError(f"Unknown primary_metric: {primary_metric!r}")


class ForecastEvaluator:
    """Compute calibrated holdout metrics for quantile forecasters.

    This class is stateless — it does not store training data or models.
    Call `evaluate()` with holdout actuals and forecast arrays.

    Invariant enforced: all actuals must be from timestamps AFTER the training
    window ends. The caller must enforce this; this class does not re-check.
    However, it logs a warning if actuals list is suspiciously small (<24 hours).
    """

    def evaluate(
        self,
        actuals: list[ForecastPoint],
        p50_forecasts: list[ForecastPoint],
        p90_forecasts: list[ForecastPoint],
        training_mean: Optional[float] = None,
    ) -> EvaluationResult:
        """Evaluate quantile forecasts against held-out actuals.

        Args:
            actuals: Actual observed values on the holdout set.
            p50_forecasts: p50 (median) predictions aligned by (timestamp, region).
            p90_forecasts: p90 predictions aligned by (timestamp, region).
            training_mean: Mean target value from the TRAINING set (not holdout).
                           Used for savings_lift. If None, uses holdout mean as proxy
                           (less accurate but still meaningful).

        Returns:
            EvaluationResult with all metrics.

        Raises:
            ValueError: If actuals or forecasts lists are empty, or if they
                        cannot be aligned (no matching timestamps).
        """
        if not actuals:
            raise ValueError("actuals list is empty — cannot evaluate on zero samples")
        if not p50_forecasts:
            raise ValueError("p50_forecasts list is empty")
        if not p90_forecasts:
            raise ValueError("p90_forecasts list is empty")

        if len(actuals) < 24:
            logger.warning(
                f"ForecastEvaluator: only {len(actuals)} holdout samples — "
                "metrics may be unreliable (< 24 hours)"
            )

        # Build lookup dicts: (timestamp, region) -> value
        actual_lookup = {(p.timestamp, p.region): p.value for p in actuals}
        p50_lookup = {(p.timestamp, p.region): p.value for p in p50_forecasts}
        p90_lookup = {(p.timestamp, p.region): p.value for p in p90_forecasts}

        # Find intersection keys
        keys = set(actual_lookup) & set(p50_lookup) & set(p90_lookup)
        if not keys:
            raise ValueError(
                "No matching (timestamp, region) keys between actuals and forecasts. "
                "Cannot compute metrics. Check that timestamps are aligned."
            )

        missed = len(actual_lookup) - len(keys)
        if missed > 0:
            logger.warning(
                f"ForecastEvaluator: {missed}/{len(actual_lookup)} actual points "
                "had no matching forecast — excluded from evaluation"
            )

        keys_sorted = sorted(keys)
        actual_vals = [actual_lookup[k] for k in keys_sorted]
        p50_vals = [p50_lookup[k] for k in keys_sorted]
        p90_vals = [p90_lookup[k] for k in keys_sorted]

        n = len(actual_vals)
        regions_present = sorted({k[1] for k in keys_sorted})

        # --- global metrics ---
        mape = _compute_mape(actual_vals, p50_vals)
        rmse = _compute_rmse(actual_vals, p50_vals)
        mae = _compute_mae(actual_vals, p50_vals)
        p50_bias = _compute_p50_bias(actual_vals, p50_vals)
        p90_coverage = _compute_p90_coverage(actual_vals, p90_vals)
        calibration_error = abs(p90_coverage - 0.90)
        downside_risk = _compute_downside_risk(actual_vals, p50_vals)

        # Savings lift: % MAE reduction vs naive predictor (training mean)
        naive_mean = training_mean if training_mean is not None else sum(actual_vals) / len(actual_vals)
        naive_mae = sum(abs(a - naive_mean) for a in actual_vals) / len(actual_vals)
        if naive_mae > 0:
            savings_lift = (naive_mae - mae) / naive_mae
        else:
            savings_lift = 0.0

        # --- per-region metrics ---
        per_region: dict[str, dict] = {}
        for region in regions_present:
            rkeys = [k for k in keys_sorted if k[1] == region]
            if not rkeys:
                continue
            ra = [actual_lookup[k] for k in rkeys]
            rp50 = [p50_lookup[k] for k in rkeys]
            rp90 = [p90_lookup[k] for k in rkeys]
            per_region[region] = {
                "n": len(rkeys),
                "mape": round(_compute_mape(ra, rp50), 4),
                "rmse": round(_compute_rmse(ra, rp50), 4),
                "mae": round(_compute_mae(ra, rp50), 4),
                "p50_bias": round(_compute_p50_bias(ra, rp50), 4),
                "p90_coverage": round(_compute_p90_coverage(ra, rp90), 4),
                "calibration_error": round(abs(_compute_p90_coverage(ra, rp90) - 0.90), 4),
            }

        return EvaluationResult(
            n_samples=n,
            mape=mape,
            rmse=rmse,
            mae=mae,
            p50_bias=p50_bias,
            p90_coverage=p90_coverage,
            calibration_error=calibration_error,
            downside_risk=downside_risk,
            savings_lift=savings_lift,
            regions=regions_present,
            per_region=per_region,
        )

    def evaluate_from_model(
        self,
        forecaster,
        holdout_actuals: list,  # list[EnergyPrice] or list[CarbonIntensity]
        training_mean: Optional[float] = None,
        recent_context: Optional[list] = None,
    ) -> EvaluationResult:
        """Evaluate a fitted forecaster against holdout actual observations.

        This is a convenience wrapper that calls forecaster.predict() for each
        unique (region, timestamp) in *holdout_actuals*, then computes metrics.

        The forecaster must already be fitted on training data that ends
        STRICTLY before the earliest timestamp in *holdout_actuals*.

        Args:
            forecaster: A fitted PriceQuantileForecaster or CarbonQuantileForecaster.
            holdout_actuals: Actual observations (EnergyPrice or CarbonIntensity).
            training_mean: Mean target value from training set (for savings_lift).
            recent_context: Recent observations to pass to predict() for lag features.
                            These must all be from the training window, not holdout.

        Returns:
            EvaluationResult with all metrics.
        """
        if not holdout_actuals:
            raise ValueError("holdout_actuals is empty")

        # Detect field name: EnergyPrice uses price_per_mwh, CarbonIntensity uses gco2_per_kwh
        sample = holdout_actuals[0]
        if hasattr(sample, "price_per_mwh"):
            value_attr = "price_per_mwh"
        elif hasattr(sample, "gco2_per_kwh"):
            value_attr = "gco2_per_kwh"
        else:
            raise ValueError(
                "holdout_actuals items must have .price_per_mwh or .gco2_per_kwh attribute"
            )

        # Group timestamps by region
        by_region: dict[str, list] = {}
        for obs in holdout_actuals:
            by_region.setdefault(obs.region, []).append(obs)

        actuals: list[ForecastPoint] = []
        p50_forecasts: list[ForecastPoint] = []
        p90_forecasts: list[ForecastPoint] = []

        for region, obs_list in by_region.items():
            obs_list.sort(key=lambda x: x.timestamp)
            timestamps = [o.timestamp for o in obs_list]

            forecasts = forecaster.predict(region, timestamps, recent_context)
            forecast_lookup = {f.timestamp: f for f in forecasts}

            for obs in obs_list:
                fc = forecast_lookup.get(obs.timestamp)
                if fc is None:
                    logger.warning(
                        f"No forecast for ({obs.timestamp}, {region}) — skipping"
                    )
                    continue
                actual_val = getattr(obs, value_attr)
                actuals.append(ForecastPoint(obs.timestamp, region, actual_val))
                p50_forecasts.append(ForecastPoint(obs.timestamp, region, fc.p50))
                p90_forecasts.append(ForecastPoint(obs.timestamp, region, fc.p90))

        return self.evaluate(actuals, p50_forecasts, p90_forecasts, training_mean)


# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------

def _compute_mape(actuals: list[float], predictions: list[float]) -> float:
    """Mean absolute percentage error. Skips zero-actual rows to avoid division by zero."""
    n = len(actuals)
    assert n == len(predictions), "actuals and predictions must have same length"
    errors = []
    for a, p in zip(actuals, predictions):
        if abs(a) < 1e-9:
            continue  # skip near-zero actuals; % error is undefined
        errors.append(abs((a - p) / a))
    if not errors:
        return float("nan")
    return sum(errors) / len(errors)


def _compute_rmse(actuals: list[float], predictions: list[float]) -> float:
    """Root mean squared error."""
    n = len(actuals)
    assert n == len(predictions)
    sse = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    return math.sqrt(sse / n)


def _compute_mae(actuals: list[float], predictions: list[float]) -> float:
    """Mean absolute error."""
    n = len(actuals)
    assert n == len(predictions)
    return sum(abs(a - p) for a, p in zip(actuals, predictions)) / n


def _compute_p50_bias(actuals: list[float], p50_predictions: list[float]) -> float:
    """Mean signed error (p50 - actual). Positive = systematic over-forecast."""
    n = len(actuals)
    assert n == len(p50_predictions)
    return sum(p - a for a, p in zip(actuals, p50_predictions)) / n


def _compute_p90_coverage(actuals: list[float], p90_predictions: list[float]) -> float:
    """Empirical fraction of actuals that fall at or below the p90 prediction.

    A well-calibrated p90 should cover ~90% of actuals.
    """
    n = len(actuals)
    assert n == len(p90_predictions)
    covered = sum(1 for a, p in zip(actuals, p90_predictions) if a <= p)
    return covered / n


def _compute_downside_risk(actuals: list[float], p50_predictions: list[float]) -> float:
    """Mean unexpected upside (actual > p50) as a fraction of mean actual.

    Captures how often and by how much actuals exceed the median forecast.
    Higher values indicate greater downside (cost/carbon) exposure.
    """
    n = len(actuals)
    assert n == len(p50_predictions)
    mean_actual = sum(actuals) / n
    if mean_actual < 1e-9:
        return 0.0
    exceedances = [max(0.0, a - p) for a, p in zip(actuals, p50_predictions)]
    return sum(exceedances) / (n * mean_actual)


# ---------------------------------------------------------------------------
# Model comparison helper
# ---------------------------------------------------------------------------

@dataclass
class ModelComparisonResult:
    """Result of comparing a candidate model vs the current active model."""
    candidate_better: bool
    primary_metric: str
    candidate_value: float
    current_value: float
    improvement_pct: float  # positive = candidate is better by this %
    promote: bool           # whether to promote candidate to active
    reason: str


def compare_models(
    candidate: EvaluationResult,
    current: EvaluationResult,
    primary_metric: str = "mape",
    min_improvement_pct: float = 1.0,
    max_calibration_regression_pts: float = 0.05,
) -> ModelComparisonResult:
    """Determine whether to promote the candidate model.

    Promotion criteria (all must pass):
    1. Candidate beats current on primary metric by at least min_improvement_pct.
    2. Candidate calibration error does not regress by more than
       max_calibration_regression_pts absolute percentage points.

    Args:
        candidate: Evaluation result for the newly trained candidate model.
        current: Evaluation result for the currently active model.
        primary_metric: Metric to optimize ("mape", "rmse", "mae").
        min_improvement_pct: Minimum % improvement required for promotion.
        max_calibration_regression_pts: Max allowed calibration error increase.

    Returns:
        ModelComparisonResult with promote=True if candidate should be promoted.
    """
    _valid_metrics = ("mape", "rmse", "mae", "calibration_error", "downside_risk", "savings_lift")
    if primary_metric not in _valid_metrics:
        raise ValueError(
            f"Unknown primary_metric: {primary_metric!r}. Valid: {_valid_metrics}"
        )

    cand_val = getattr(candidate, primary_metric)
    curr_val = getattr(current, primary_metric)

    # For lower-is-better metrics: improvement = (curr - cand) / curr
    if primary_metric in ("mape", "rmse", "mae", "calibration_error", "downside_risk"):
        if curr_val > 0:
            improvement_pct = (curr_val - cand_val) / curr_val * 100
        else:
            improvement_pct = 0.0
        candidate_better = improvement_pct >= min_improvement_pct
    elif primary_metric == "savings_lift":
        improvement_pct = (cand_val - curr_val) * 100
        candidate_better = improvement_pct >= min_improvement_pct
    else:
        raise ValueError(f"Unknown primary_metric: {primary_metric!r}")

    # Calibration regression check
    cal_regression = candidate.calibration_error - current.calibration_error
    calibration_ok = cal_regression <= max_calibration_regression_pts

    promote = candidate_better and calibration_ok

    if promote:
        reason = (
            f"candidate improves {primary_metric} by {improvement_pct:.2f}% "
            f"({curr_val:.4f} → {cand_val:.4f}); calibration_error change: "
            f"{cal_regression:+.4f} (within limit)"
        )
    elif not candidate_better:
        reason = (
            f"candidate does not improve {primary_metric} by min {min_improvement_pct}% "
            f"(improvement={improvement_pct:.2f}%)"
        )
    else:
        reason = (
            f"calibration regression {cal_regression:+.4f} exceeds limit "
            f"{max_calibration_regression_pts:.4f}"
        )

    return ModelComparisonResult(
        candidate_better=candidate_better,
        primary_metric=primary_metric,
        candidate_value=cand_val,
        current_value=curr_val,
        improvement_pct=improvement_pct,
        promote=promote,
        reason=reason,
    )
