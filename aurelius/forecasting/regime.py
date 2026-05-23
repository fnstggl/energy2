"""Regime detection and forecast correction for post-spike price recovery.

When a region experiences a sharp price spike (e.g., ERCOT winter cold snap)
followed by price normalization, an ML forecaster trained on that spike period
will systematically overestimate future prices. This creates a costly error:
the optimizer routes jobs away from the cheap recovering region.

This module detects such "recovery regimes" and applies a statistically
grounded, leakage-safe correction to reduce forecast bias during recovery.

Key design invariants:
- Uses ONLY training-window data (never eval-window actuals)
- Conservative activation threshold: recent_mean < 40% of training_mean
- Absolute price gate: recent_mean must be < 30 $/MWh (avoids correcting regions
  at "normal" price levels, even if they are low relative to a spike-inflated mean)
- Asymmetric: reduces overpredictions ONLY (never inflates underpredictions)
- Exponential decay: near-term hours corrected more than long-horizon
- Per-region: each region evaluated independently
- Magnitude bounded: correction never exceeds 50% of the prediction excess
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeInfo:
    """Result of regime analysis for a single region.

    Attributes:
        region: The grid region analyzed.
        is_recovering: True if the region is in a post-spike recovery state.
        recovery_ratio: recent_mean / training_mean. Values < threshold = recovery.
        recent_mean: Mean price over the last recent_hours of training context.
        training_mean: Mean price over the full training window.
        correction_magnitude: Fraction of excess to correct (0 = no correction).
    """
    region: str
    is_recovering: bool
    recovery_ratio: float
    recent_mean: float
    training_mean: float
    correction_magnitude: float


class RegimeDetector:
    """Detects post-spike price recovery regimes and applies forecast corrections.

    Detection logic:
        recovery_ratio = recent_mean / training_mean
        is_recovering = recovery_ratio < recovery_ratio_threshold

    If a region is recovering:
        For each future prediction at horizon h:
            decay = exp(-h * log(2) / decay_halflife_hours)
            effective_correction = correction_magnitude * decay
            excess = max(0, predicted - recent_mean)
            corrected = max(recent_mean * 0.8, predicted - effective_correction * excess)

    This ensures:
        - Near-term predictions are pulled strongly toward recent observed prices
        - Long-horizon predictions decay back to the ML model's forecast
        - Predictions never fall below 80% of recent_mean (avoids over-correction)
        - Stable regions (ratio >= threshold) are untouched

    Args:
        recovery_ratio_threshold: Ratio below which recovery is detected (default 0.40).
            This means: recent prices must be < 40% of the full training mean.
            Conservative enough to avoid false positives from normal diurnal variation
            (overnight min/24h-mean ratio is typically 0.5-0.8, safely above threshold).
        max_recent_mean_for_correction: Absolute price ceiling ($/MWh, default 30.0).
            Even if the ratio test passes, correction is suppressed when recent_mean
            exceeds this value. Prevents correcting regions whose prices are in the
            normal operating range (e.g., PJM at $32-55/MWh post-spike) — a region
            at "normal" wholesale prices doesn't need a recovery correction even if
            its training mean was spike-inflated.
        recent_hours: Hours of training tail to use for recent_mean (default 24).
        max_correction_fraction: Maximum correction as fraction of excess (default 0.50).
        decay_halflife_hours: Hours at which correction decays to 50% (default 72).
            Training jobs (~96-200h): correction meaningful for first ~72-144h.
            Fine-tuning (~8-24h): correction applies to the full window.
    """

    def __init__(
        self,
        recovery_ratio_threshold: float = 0.40,
        max_recent_mean_for_correction: float = 30.0,
        recent_hours: int = 24,
        max_correction_fraction: float = 0.50,
        decay_halflife_hours: float = 72.0,
    ) -> None:
        if not (0.0 < recovery_ratio_threshold < 1.0):
            raise ValueError(f"recovery_ratio_threshold must be in (0, 1), got {recovery_ratio_threshold}")
        if max_recent_mean_for_correction <= 0.0:
            raise ValueError(f"max_recent_mean_for_correction must be > 0, got {max_recent_mean_for_correction}")
        if recent_hours < 1:
            raise ValueError(f"recent_hours must be >= 1, got {recent_hours}")
        if not (0.0 < max_correction_fraction <= 1.0):
            raise ValueError(f"max_correction_fraction must be in (0, 1], got {max_correction_fraction}")
        if decay_halflife_hours <= 0:
            raise ValueError(f"decay_halflife_hours must be > 0, got {decay_halflife_hours}")

        self.recovery_ratio_threshold = recovery_ratio_threshold
        self.max_recent_mean_for_correction = max_recent_mean_for_correction
        self.recent_hours = recent_hours
        self.max_correction_fraction = max_correction_fraction
        self.decay_halflife_hours = decay_halflife_hours

    def detect(
        self,
        region: str,
        training_prices: list[float],
        recent_context_prices: list[float],
    ) -> RegimeInfo:
        """Detect recovery regime for a single region.

        Args:
            region: Region identifier for logging.
            training_prices: All prices from the training window for this region.
            recent_context_prices: The most recent prices (up to recent_hours used).

        Returns:
            RegimeInfo with detection result and correction parameters.
        """
        if not training_prices or not recent_context_prices:
            return RegimeInfo(
                region=region,
                is_recovering=False,
                recovery_ratio=1.0,
                recent_mean=0.0,
                training_mean=0.0,
                correction_magnitude=0.0,
            )

        training_mean = float(np.mean(training_prices))
        recent = recent_context_prices[-self.recent_hours:]
        recent_mean = float(np.mean(recent)) if recent else 0.0

        if training_mean <= 0.0 or recent_mean <= 0.0:
            return RegimeInfo(
                region=region,
                is_recovering=False,
                recovery_ratio=1.0,
                recent_mean=recent_mean,
                training_mean=training_mean,
                correction_magnitude=0.0,
            )

        ratio = recent_mean / training_mean

        # Two-gate activation: ratio test AND absolute price ceiling.
        # The ceiling prevents correcting regions at "normal" price levels (e.g.,
        # PJM post-spike at $32–55/MWh) when the spike-inflated training mean
        # makes the ratio look like recovery but prices aren't genuinely cheap.
        is_recovering = (
            ratio < self.recovery_ratio_threshold
            and recent_mean <= self.max_recent_mean_for_correction
        )

        if is_recovering:
            # Deeper recovery (lower ratio) → larger correction
            depth = 1.0 - ratio  # range (0.6, 1.0] for recovery
            magnitude = min(self.max_correction_fraction, depth * 0.65)
        else:
            magnitude = 0.0

        return RegimeInfo(
            region=region,
            is_recovering=is_recovering,
            recovery_ratio=ratio,
            recent_mean=recent_mean,
            training_mean=training_mean,
            correction_magnitude=magnitude,
        )

    def correct_predictions(
        self,
        predicted_prices: dict,
        regime: RegimeInfo,
    ) -> dict:
        """Apply decaying correction to a region's predicted prices.

        Args:
            predicted_prices: Dict of {timestamp: price} for the forecast horizon.
            regime: RegimeInfo from detect().

        Returns:
            Corrected dict. Unchanged if regime.is_recovering is False.
        """
        if not regime.is_recovering or regime.correction_magnitude <= 0.0:
            return predicted_prices

        sorted_ts = sorted(predicted_prices.keys())
        corrected: dict = {}

        for i, ts in enumerate(sorted_ts):
            # Exponential decay: magnitude halves every decay_halflife_hours
            decay = np.exp(-i * np.log(2) / self.decay_halflife_hours)
            effective = regime.correction_magnitude * decay

            predicted = predicted_prices[ts]
            excess = predicted - regime.recent_mean

            if excess > 0.0:
                # Reduce the excess above recent_mean
                reduction = effective * excess
                # Floor: never correct below 80% of recent_mean (avoids over-correction)
                floor = regime.recent_mean * 0.8
                corrected[ts] = max(floor, predicted - reduction)
            else:
                # Already at or below recent_mean — leave unchanged
                corrected[ts] = predicted

        return corrected

    def apply_corrections_to_forecast(
        self,
        forecast_price_data: dict,
        train_price_data: dict,
        recent_context: list,
    ) -> dict:
        """Apply regime corrections to all regions in the forecast.

        Args:
            forecast_price_data: {region: {timestamp: price}} — ML model output.
            train_price_data: {region: {timestamp: price}} — full training window.
            recent_context: list[EnergyPrice] — recent training records (last context_hours
                per region; used to compute recent_mean per region).

        Returns:
            Corrected forecast dict with same structure as forecast_price_data.
        """
        # Build per-region recent prices from context
        context_by_region: dict[str, list[float]] = {}
        for record in recent_context:
            region = record.region
            context_by_region.setdefault(region, []).append(record.price_per_mwh)

        corrected_forecast: dict = {}
        n_corrected = 0

        for region, ts_price_map in forecast_price_data.items():
            training_prices = list(train_price_data.get(region, {}).values())
            recent_prices = context_by_region.get(region, [])

            regime = self.detect(region, training_prices, recent_prices)

            if regime.is_recovering:
                n_corrected += 1
                logger.info(
                    "Regime correction applied: region=%s recent_mean=%.1f "
                    "training_mean=%.1f ratio=%.3f magnitude=%.3f",
                    region, regime.recent_mean, regime.training_mean,
                    regime.recovery_ratio, regime.correction_magnitude,
                )
                corrected_forecast[region] = self.correct_predictions(ts_price_map, regime)
            else:
                corrected_forecast[region] = ts_price_map

        if n_corrected > 0:
            logger.info(
                "RegimeDetector: corrected %d/%d regions",
                n_corrected, len(forecast_price_data),
            )

        return corrected_forecast


def compute_region_regime_summary(
    train_price_data: dict,
    recent_context: list,
    detector: Optional[RegimeDetector] = None,
) -> dict[str, RegimeInfo]:
    """Compute regime info for all regions (diagnostic helper).

    Returns {region: RegimeInfo} for all regions in train_price_data.
    """
    if detector is None:
        detector = RegimeDetector()

    context_by_region: dict[str, list[float]] = {}
    for record in recent_context:
        context_by_region.setdefault(record.region, []).append(record.price_per_mwh)

    result = {}
    for region, ts_map in train_price_data.items():
        training_prices = list(ts_map.values())
        recent_prices = context_by_region.get(region, [])
        result[region] = detector.detect(region, training_prices, recent_prices)

    return result
