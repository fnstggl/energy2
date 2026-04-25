"""Quantile calibration for price and carbon forecasters.

After training, a quantile forecaster's empirical coverage may drift from
the nominal level due to distribution shift or model bias.  This module
provides calibrate_quantile(), which finds a scale factor that brings
empirical coverage within ±2 percentage points of the desired target.

Usage:
    from aurelius.forecasting.calibration import calibrate_quantile
    scale = calibrate_quantile(forecaster, holdout_prices, target_quantile=0.90)
    # Apply at predict time: adjusted_p90 = raw_p90 * scale
"""

from __future__ import annotations

import logging
from typing import Union

from aurelius.models import EnergyPrice, CarbonIntensity

logger = logging.getLogger(__name__)

# Tolerance: stop binary search when |empirical_coverage - target| < TOL
_COVERAGE_TOL = 0.005       # 0.5 pp — well within the ±2 pp requirement
_SCALE_LO = 0.50            # Lower bound for scale factor search
_SCALE_HI = 3.00            # Upper bound (3× p90 is very conservative)
_MAX_ITER = 50              # Binary search iterations


def _empirical_coverage(actuals: list[float], p90s: list[float], scale: float) -> float:
    """Fraction of actuals ≤ scaled p90 prediction."""
    if not actuals:
        return 0.0
    covered = sum(a <= (p * scale) for a, p in zip(actuals, p90s))
    return covered / len(actuals)


def calibrate_quantile(
    model,
    holdout_data: Union[list[EnergyPrice], list[CarbonIntensity]],
    target_quantile: float = 0.90,
) -> float:
    """Find a multiplicative scale factor that achieves target empirical coverage.

    The calibration is performed by binary-searching for a scale ``s`` such that:

        fraction(actual ≤ p90_raw × s) ≈ target_quantile   (within ±2 pp)

    The model is called with ``predict()`` to get raw p90 forecasts.  No future
    data is used: predictions are compared to actuals in the holdout set after
    the fact (the holdout must itself have been withheld during model training).

    Args:
        model: A fitted ``PriceQuantileForecaster`` or ``CarbonQuantileForecaster``
               with a ``predict(region, timestamps, recent)`` method that returns
               objects with ``.p90`` attributes.
        holdout_data: List of ``EnergyPrice`` or ``CarbonIntensity`` records that
               were NOT used during model training (temporal holdout).  Must be
               non-empty.
        target_quantile: Desired empirical coverage level (default 0.90 = 90th
               percentile).  Must be in (0, 1).

    Returns:
        scale_factor: A positive float.  Multiply raw p90 predictions by this
            value to achieve empirical coverage ≈ target_quantile ± 2 pp.

    Raises:
        ValueError: If holdout_data is empty or target_quantile is out of range.
        RuntimeError: If the model is not fitted.
    """
    if not holdout_data:
        raise ValueError("calibrate_quantile: holdout_data is empty")
    if not (0.0 < target_quantile < 1.0):
        raise ValueError(f"calibrate_quantile: target_quantile must be in (0,1), got {target_quantile}")
    if hasattr(model, "is_fitted") and not model.is_fitted:
        raise RuntimeError("calibrate_quantile: model is not fitted")

    # Detect value attribute (EnergyPrice vs CarbonIntensity)
    first = holdout_data[0]
    val_attr = "price_per_mwh" if hasattr(first, "price_per_mwh") else "gco2_per_kwh"

    # Group by region and build (actual, p90_raw) pairs leakage-free.
    # For each region, sort by timestamp, predict without context (no lag leakage),
    # then pair with actuals.
    actuals: list[float] = []
    p90s_raw: list[float] = []

    by_region: dict[str, list] = {}
    for rec in holdout_data:
        by_region.setdefault(rec.region, []).append(rec)

    for region, recs in by_region.items():
        recs_sorted = sorted(recs, key=lambda r: r.timestamp)
        timestamps = [r.timestamp for r in recs_sorted]
        region_actuals = [getattr(r, val_attr) for r in recs_sorted]

        # Predict without recent context to avoid any leakage.
        # Using no context means we lose lag features but remain leakage-free.
        # Try both common kwarg names (recent_prices / recent_data / positional).
        try:
            forecasts = model.predict(region, timestamps, recent_prices=None)
        except TypeError:
            try:
                forecasts = model.predict(region, timestamps, recent_data=None)
            except TypeError:
                forecasts = model.predict(region, timestamps)

        for f, actual in zip(forecasts, region_actuals):
            actuals.append(float(actual))
            p90s_raw.append(float(f.p90))

    if not actuals:
        logger.warning("calibrate_quantile: no (actual, p90) pairs collected; returning scale=1.0")
        return 1.0

    # Check if scale=1.0 already meets target
    cov_at_1 = _empirical_coverage(actuals, p90s_raw, 1.0)
    logger.debug(f"calibrate_quantile: coverage at scale=1.0 is {cov_at_1:.4f}, target={target_quantile:.4f}")

    if abs(cov_at_1 - target_quantile) <= _COVERAGE_TOL:
        logger.info(f"calibrate_quantile: scale=1.0 already achieves coverage {cov_at_1:.4f}")
        return 1.0

    # Binary search for scale in [_SCALE_LO, _SCALE_HI]
    lo, hi = _SCALE_LO, _SCALE_HI

    # Sanity check: does increasing scale increase coverage?
    cov_lo = _empirical_coverage(actuals, p90s_raw, lo)
    cov_hi = _empirical_coverage(actuals, p90s_raw, hi)

    if cov_hi < target_quantile:
        logger.warning(
            f"calibrate_quantile: even scale={hi} gives coverage {cov_hi:.4f} < target "
            f"{target_quantile:.4f}. Model p90s may be severely underestimated. Returning {hi}."
        )
        return float(hi)

    if cov_lo > target_quantile:
        logger.warning(
            f"calibrate_quantile: even scale={lo} gives coverage {cov_lo:.4f} > target "
            f"{target_quantile:.4f}. Model p90s may be overestimated. Returning {lo}."
        )
        return float(lo)

    scale = 1.0
    for _ in range(_MAX_ITER):
        scale = (lo + hi) / 2.0
        cov = _empirical_coverage(actuals, p90s_raw, scale)
        if abs(cov - target_quantile) <= _COVERAGE_TOL:
            break
        if cov < target_quantile:
            lo = scale
        else:
            hi = scale

    final_cov = _empirical_coverage(actuals, p90s_raw, scale)
    logger.info(
        f"calibrate_quantile: scale={scale:.4f}, "
        f"empirical_coverage={final_cov:.4f}, "
        f"target={target_quantile:.4f}, "
        f"n={len(actuals)}"
    )
    return float(scale)
