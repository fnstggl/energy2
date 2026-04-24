"""Energy price forecasting model.

This module provides:
- LightGBM quantile regressors for p50 and p90 forecasts
- Simple interpretable forecasting using rolling averages and seasonality
- Fallback to baseline when LightGBM unavailable

Produces p50 and p90 forecasts. Does NOT apply safety logic beyond p90 >= p50.

IMPORTANT:
- Offline batch training ONLY
- Fixed random seeds for determinism
- No learning during execution

v1.1 enables minimal short-horizon lag features (1h, 6h) for accuracy
while preserving predict-time safety and deterministic fallback.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from ..models import EnergyPrice
from .quantile_model import (
    DEFAULT_SEED,
    MIN_RECENT_HOURS,
    ModelMetadata,
    QuantileForecast,
    QUANTILE_P50,
    QUANTILE_P90,
    build_feature_matrix,
    build_feature_matrix_for_predict,
    check_recent_data_sufficient,
    predict_with_fallback,
    set_deterministic_seed,
    time_series_cv_split,
    train_lightgbm_quantile,
    validate_quantiles,
)

logger = logging.getLogger(__name__)


@dataclass
class PriceForecast:
    """A price forecast with uncertainty bounds.

    Attributes:
        timestamp: The forecasted hour
        region: Geographic region
        mean: Mean predicted price ($/MWh) - used as p50
        std: Standard deviation of prediction
        lower_bound: Lower confidence bound (mean - 2*std)
        upper_bound: Upper confidence bound (mean + 2*std)
    """
    timestamp: datetime
    region: str
    mean: float
    std: float

    @property
    def lower_bound(self) -> float:
        return max(0, self.mean - 2 * self.std)

    @property
    def upper_bound(self) -> float:
        return self.mean + 2 * self.std


@dataclass
class PriceQuantileForecast:
    """A price forecast with quantile predictions.

    Attributes:
        timestamp: The forecasted hour
        region: Geographic region
        p50: Median predicted price ($/MWh)
        p90: 90th percentile predicted price ($/MWh)
        model_type: Type of model used
        features_version: Version of feature set
    """
    timestamp: datetime
    region: str
    p50: float
    p90: float
    model_type: str = "ridge+lightgbm_quantile"
    features_version: str = "v1"


@dataclass
class PriceModelConfig:
    """Configuration for price forecasting model.

    Attributes:
        seed: Random seed for reproducibility
        n_estimators: Number of LightGBM trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate for boosting
        use_baseline_fallback: Whether to fallback to baseline if LightGBM unavailable
    """
    seed: int = DEFAULT_SEED
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    use_baseline_fallback: bool = True


class PriceQuantileForecaster:
    """LightGBM quantile forecaster for energy prices.

    Produces p50 (median) and p90 (upper bound) forecasts.
    Falls back to baseline if LightGBM unavailable.

    Training discipline:
    - Offline batch training only
    - Fixed random seeds
    - Time-based cross-validation
    - Deterministic outputs
    """

    def __init__(self, config: Optional[PriceModelConfig] = None):
        """Initialize the price quantile forecaster.

        Args:
            config: Model configuration
        """
        self.config = config or PriceModelConfig()
        self._model_p50: Any = None
        self._model_p90: Any = None
        self._fitted = False
        self._known_regions: list[str] = []
        self._metadata: Optional[ModelMetadata] = None
        self._baseline_mean: float = 50.0  # Default baseline
        # v1.1: Enable minimal lags for improved accuracy with safe fallback
        # Price model uses lag_1h, lag_6h, rolling_mean_6h
        # Predict-time: falls back to temporal+region if recent data unavailable
        self._use_lags = True
        self._use_rolling = True
        self._lag_hours = [1, 6]  # lag_1h, lag_6h
        self._rolling_hours = [6]  # rolling_mean_6h only

    def fit(self, prices: list[EnergyPrice]) -> "PriceQuantileForecaster":
        """Fit the quantile models on historical price data.

        Args:
            prices: List of historical EnergyPrice objects

        Returns:
            Self for chaining
        """
        set_deterministic_seed(self.config.seed)

        # Extract training data
        timestamps = [p.timestamp for p in prices]
        regions = [p.region for p in prices]
        values = np.array([p.price_per_mwh for p in prices])

        self._known_regions = sorted(set(regions))
        self._baseline_mean = float(np.mean(values)) if len(values) > 0 else 50.0

        # v1.1: Build feature matrix with minimal lags (lag_1h, lag_6h, rolling_mean_6h)
        X = build_feature_matrix(
            timestamps,
            regions,
            values,
            include_lags=self._use_lags,
            include_rolling=self._use_rolling,
            known_regions=self._known_regions,
            lag_hours=self._lag_hours,
            rolling_hours=self._rolling_hours,
        )
        X_np = X.values

        # Train p50 model
        logger.info("Training p50 (median) price model...")
        self._model_p50 = train_lightgbm_quantile(
            X_np, values, QUANTILE_P50,
            seed=self.config.seed,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
        )

        # Train p90 model
        logger.info("Training p90 (upper bound) price model...")
        self._model_p90 = train_lightgbm_quantile(
            X_np, values, QUANTILE_P90,
            seed=self.config.seed,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
        )

        self._fitted = True

        # Determine model type based on what was trained
        if self._model_p50 is not None and self._model_p90 is not None:
            model_type = "ridge+lightgbm_quantile"
        else:
            model_type = "baseline_fallback"

        self._metadata = ModelMetadata(
            model_type=model_type,
            trained_at=datetime.utcnow(),
            features_version="v1.1",  # Updated for minimal lags
            training_samples=len(prices),
            regions=self._known_regions,
            seed=self.config.seed,
        )

        logger.info(
            f"Fitted price quantile model ({model_type}) on "
            f"{len(prices)} samples, {len(self._known_regions)} regions"
        )

        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_prices: Optional[list[EnergyPrice]] = None,
    ) -> list[PriceQuantileForecast]:
        """Generate quantile price forecasts.

        v1.1: Uses lag features if sufficient recent data available (≥6 hours),
        otherwise falls back silently to temporal+region features.

        Args:
            region: Region to forecast
            timestamps: List of future timestamps to predict
            recent_prices: Recent price data for feature computation (≥6 hours for lags)

        Returns:
            List of PriceQuantileForecast objects
        """
        if not self._fitted:
            logger.warning("Model not fitted, using baseline fallback")
            return self._baseline_predict(timestamps, region)

        # v1.1: Extract recent values for this region
        recent_values = None
        if recent_prices:
            region_prices = [p.price_per_mwh for p in recent_prices if p.region == region]
            if len(region_prices) >= MIN_RECENT_HOURS:
                recent_values = np.array(region_prices[-48:])  # Use up to 48 hours

        regions = [region] * len(timestamps)

        # v1.1: Check if sufficient recent data for lag features
        use_lags = check_recent_data_sufficient(recent_values)

        if use_lags:
            # Build feature matrix with lag features from recent data
            X, _ = build_feature_matrix_for_predict(
                timestamps,
                regions,
                recent_values,
                known_regions=self._known_regions,
                lag_hours=self._lag_hours,
                rolling_hours=self._rolling_hours,
            )
        else:
            # Fallback: temporal+region only, fill lag columns with 0
            logger.debug("Price model: using temporal+region only (insufficient recent data)")
            X = build_feature_matrix(
                timestamps,
                regions,
                values=None,
                include_lags=True,  # Keep columns for model compatibility
                include_rolling=True,
                known_regions=self._known_regions,
                lag_hours=self._lag_hours,
                rolling_hours=self._rolling_hours,
            )

        X_np = X.values

        # Get baseline for fallback
        baseline = np.full(len(timestamps), self._baseline_mean)

        # Predict with fallback
        p50_preds, p90_preds = predict_with_fallback(
            self._model_p50,
            self._model_p90,
            X_np,
            baseline,
        )

        # Build forecast objects
        model_type = self._metadata.model_type if self._metadata else "unknown"
        forecasts = []
        for i, ts in enumerate(timestamps):
            forecasts.append(PriceQuantileForecast(
                timestamp=ts,
                region=region,
                p50=round(p50_preds[i], 2),
                p90=round(p90_preds[i], 2),
                model_type=model_type,
                features_version="v1.1",
            ))

        return forecasts

    def predict_range(
        self,
        region: str,
        start_time: datetime,
        hours: int,
        recent_prices: Optional[list[EnergyPrice]] = None,
    ) -> list[PriceQuantileForecast]:
        """Generate quantile forecasts for a time range.

        Args:
            region: Region to forecast
            start_time: Start of forecast window
            hours: Number of hours to forecast
            recent_prices: Recent price data

        Returns:
            List of PriceQuantileForecast objects
        """
        timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
        return self.predict(region, timestamps, recent_prices)

    def _baseline_predict(
        self,
        timestamps: list[datetime],
        region: str,
    ) -> list[PriceQuantileForecast]:
        """Baseline prediction when model not trained.

        Args:
            timestamps: Timestamps to predict
            region: Region identifier

        Returns:
            List of baseline forecasts
        """
        forecasts = []
        for ts in timestamps:
            # Simple baseline: use mean with 30% uplift for p90
            p50 = self._baseline_mean
            p90 = self._baseline_mean * 1.3

            forecasts.append(PriceQuantileForecast(
                timestamp=ts,
                region=region,
                p50=round(p50, 2),
                p90=round(p90, 2),
                model_type="baseline_fallback",
                features_version="v1",
            ))

        return forecasts

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def forecasts_to_dict(
        self,
        forecasts: list[PriceQuantileForecast],
    ) -> dict[str, dict[datetime, dict[str, float]]]:
        """Convert forecasts to lookup dict.

        Returns:
            Dict of {region: {timestamp: {"p50": float, "p90": float}}}
        """
        result: dict[str, dict[datetime, dict[str, float]]] = {}
        for f in forecasts:
            if f.region not in result:
                result[f.region] = {}
            result[f.region][f.timestamp] = {"p50": f.p50, "p90": f.p90}
        return result


class PriceForecaster:
    """Forecasts energy prices using interpretable methods.

    The model combines:
    1. Historical rolling average as baseline
    2. Hour-of-day seasonal adjustment
    3. Day-of-week adjustment
    4. Simple linear trend (optional)

    Forecast equation:
        price = rolling_avg * hourly_factor * dow_factor + trend

    This is the legacy forecaster. For quantile forecasts, use PriceQuantileForecaster.
    """

    def __init__(
        self,
        lookback_hours: int = 168,  # 1 week
        rolling_window: int = 24,
    ):
        """Initialize the forecaster.

        Args:
            lookback_hours: Hours of history to consider
            rolling_window: Window size for rolling average
        """
        self.lookback_hours = lookback_hours
        self.rolling_window = rolling_window

        # Learned parameters per region
        self._hourly_factors: dict[str, dict[int, float]] = {}
        self._dow_factors: dict[str, dict[int, float]] = {}
        self._base_levels: dict[str, float] = {}
        self._volatilities: dict[str, float] = {}

    def fit(self, prices: list[EnergyPrice]) -> "PriceForecaster":
        """Fit the model on historical price data.

        Args:
            prices: List of historical EnergyPrice objects

        Returns:
            Self for chaining
        """
        # Group by region
        by_region: dict[str, list[EnergyPrice]] = {}
        for p in prices:
            if p.region not in by_region:
                by_region[p.region] = []
            by_region[p.region].append(p)

        for region, region_prices in by_region.items():
            # Sort by timestamp
            region_prices.sort(key=lambda x: x.timestamp)

            # Calculate base level (overall mean)
            values = [p.price_per_mwh for p in region_prices]
            base_level = sum(values) / len(values) if values else 50.0
            self._base_levels[region] = base_level

            # Calculate volatility (std dev of relative changes)
            if len(values) > 1:
                relative_changes = [
                    (values[i] - values[i-1]) / values[i-1]
                    for i in range(1, len(values))
                    if values[i-1] > 0
                ]
                if relative_changes:
                    mean_change = sum(relative_changes) / len(relative_changes)
                    variance = sum((c - mean_change) ** 2 for c in relative_changes) / len(relative_changes)
                    self._volatilities[region] = math.sqrt(variance) * base_level
                else:
                    self._volatilities[region] = base_level * 0.1
            else:
                self._volatilities[region] = base_level * 0.1

            # Calculate hour-of-day factors
            hourly_sums: dict[int, float] = {h: 0.0 for h in range(24)}
            hourly_counts: dict[int, int] = {h: 0 for h in range(24)}

            for p in region_prices:
                hour = p.timestamp.hour
                hourly_sums[hour] += p.price_per_mwh
                hourly_counts[hour] += 1

            self._hourly_factors[region] = {}
            for hour in range(24):
                if hourly_counts[hour] > 0:
                    avg = hourly_sums[hour] / hourly_counts[hour]
                    self._hourly_factors[region][hour] = avg / base_level if base_level > 0 else 1.0
                else:
                    self._hourly_factors[region][hour] = 1.0

            # Calculate day-of-week factors
            dow_sums: dict[int, float] = {d: 0.0 for d in range(7)}
            dow_counts: dict[int, int] = {d: 0 for d in range(7)}

            for p in region_prices:
                dow = p.timestamp.weekday()
                dow_sums[dow] += p.price_per_mwh
                dow_counts[dow] += 1

            self._dow_factors[region] = {}
            for dow in range(7):
                if dow_counts[dow] > 0:
                    avg = dow_sums[dow] / dow_counts[dow]
                    self._dow_factors[region][dow] = avg / base_level if base_level > 0 else 1.0
                else:
                    self._dow_factors[region][dow] = 1.0

        logger.info(f"Fitted price model for {len(by_region)} regions")
        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_prices: Optional[list[EnergyPrice]] = None,
    ) -> list[PriceForecast]:
        """Generate price forecasts.

        Args:
            region: Region to forecast
            timestamps: List of future timestamps to predict
            recent_prices: Recent price data for rolling average adjustment

        Returns:
            List of PriceForecast objects
        """
        forecasts = []

        # Get base level and factors
        base_level = self._base_levels.get(region, 50.0)
        hourly_factors = self._hourly_factors.get(region, {h: 1.0 for h in range(24)})
        dow_factors = self._dow_factors.get(region, {d: 1.0 for d in range(7)})
        volatility = self._volatilities.get(region, base_level * 0.1)

        # Calculate rolling average from recent prices if available
        if recent_prices:
            recent_values = [
                p.price_per_mwh for p in recent_prices
                if p.region == region
            ][-self.rolling_window:]
            if recent_values:
                rolling_avg = sum(recent_values) / len(recent_values)
            else:
                rolling_avg = base_level
        else:
            rolling_avg = base_level

        for ts in timestamps:
            hour = ts.hour
            dow = ts.weekday()

            # Combined prediction
            hourly_factor = hourly_factors.get(hour, 1.0)
            dow_factor = dow_factors.get(dow, 1.0)

            # Blend rolling average with seasonal factors
            mean_price = rolling_avg * hourly_factor * dow_factor

            # Uncertainty increases with forecast horizon
            hours_ahead = max(1, int((ts - datetime.utcnow()).total_seconds() / 3600))
            uncertainty_factor = 1 + 0.02 * math.sqrt(hours_ahead)

            forecasts.append(PriceForecast(
                timestamp=ts,
                region=region,
                mean=round(mean_price, 2),
                std=round(volatility * uncertainty_factor, 2),
            ))

        return forecasts

    def predict_range(
        self,
        region: str,
        start_time: datetime,
        hours: int,
        recent_prices: Optional[list[EnergyPrice]] = None,
    ) -> list[PriceForecast]:
        """Generate forecasts for a time range.

        Args:
            region: Region to forecast
            start_time: Start of forecast window
            hours: Number of hours to forecast
            recent_prices: Recent price data

        Returns:
            List of PriceForecast objects
        """
        timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
        return self.predict(region, timestamps, recent_prices)

    def forecasts_to_dict(
        self,
        forecasts: list[PriceForecast],
    ) -> dict[str, dict[datetime, tuple[float, float]]]:
        """Convert forecasts to lookup dict.

        Returns:
            Dict of {region: {timestamp: (mean, std)}}
        """
        result: dict[str, dict[datetime, tuple[float, float]]] = {}
        for f in forecasts:
            if f.region not in result:
                result[f.region] = {}
            result[f.region][f.timestamp] = (f.mean, f.std)
        return result


# ============================================================================
# INLINE VALIDATION
# ============================================================================
# Run with: python -c "from aurelius.forecasting.price_model import _run_validation; _run_validation()"

def _run_validation():
    """Validate price quantile forecaster."""
    print("=" * 60)
    print("Price Quantile Forecaster Validation")
    print("=" * 60)

    # Generate synthetic training data
    print("\nGenerating training data...")
    from .baseline import generate_price_scenario
    prices = generate_price_scenario(
        start_time=datetime(2025, 1, 1),
        hours=168 * 4,  # 4 weeks
        regions=["us-west", "us-east"],
        scenario="normal",
        seed=42,
    )
    print(f"  Generated {len(prices)} price records")

    # Test 1: Fit quantile forecaster
    print("\nTest 1: FIT QUANTILE FORECASTER")
    print("-" * 40)
    config = PriceModelConfig(seed=42, n_estimators=50)
    forecaster = PriceQuantileForecaster(config)
    forecaster.fit(prices)
    print(f"  Fitted: {forecaster.is_fitted}")
    assert forecaster.is_fitted

    # Test 2: Predict quantiles
    print("\nTest 2: PREDICT QUANTILES")
    print("-" * 40)
    pred_ts = [datetime(2025, 2, 1, h) for h in range(24)]
    recent = [p for p in prices if p.region == "us-west"][-48:]
    forecasts = forecaster.predict("us-west", pred_ts, recent)
    print(f"  Generated {len(forecasts)} forecasts")
    for f in forecasts[:3]:
        print(f"    {f.timestamp.hour}:00 - p50={f.p50:.2f}, p90={f.p90:.2f}")

    # Test 3: Validate p90 >= p50
    print("\nTest 3: VALIDATE p90 >= p50")
    print("-" * 40)
    all_valid = True
    for f in forecasts:
        if f.p90 < f.p50:
            print(f"  FAIL: p90 ({f.p90}) < p50 ({f.p50}) at {f.timestamp}")
            all_valid = False
    if all_valid:
        print("  All forecasts satisfy p90 >= p50: PASS")
    assert all_valid

    # Test 4: Determinism
    print("\nTest 4: DETERMINISM")
    print("-" * 40)
    forecaster1 = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=50))
    forecaster1.fit(prices)
    preds1 = forecaster1.predict("us-west", pred_ts[:5], recent)

    forecaster2 = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=50))
    forecaster2.fit(prices)
    preds2 = forecaster2.predict("us-west", pred_ts[:5], recent)

    for i in range(len(preds1)):
        assert preds1[i].p50 == preds2[i].p50, "p50 should be identical"
        assert preds1[i].p90 == preds2[i].p90, "p90 should be identical"
    print("  Same seed produces identical predictions: PASS")

    # Test 5: Regional variation
    print("\nTest 5: REGIONAL VARIATION")
    print("-" * 40)
    west_preds = forecaster.predict("us-west", pred_ts[:5], recent)
    east_recent = [p for p in prices if p.region == "us-east"][-48:]
    east_preds = forecaster.predict("us-east", pred_ts[:5], east_recent)

    west_avg = np.mean([f.p50 for f in west_preds])
    east_avg = np.mean([f.p50 for f in east_preds])
    print(f"  us-west avg p50: {west_avg:.2f}")
    print(f"  us-east avg p50: {east_avg:.2f}")
    # us-east should be more expensive in our scenario
    if east_avg > west_avg:
        print("  Regional price difference detected: PASS")
    else:
        print("  Warning: Expected us-east to be more expensive")

    # Test 6: Metadata (v1.1)
    print("\nTest 6: METADATA (v1.1)")
    print("-" * 40)
    meta = forecaster.metadata
    assert meta is not None
    print(f"  Model type: {meta.model_type}")
    print(f"  Trained at: {meta.trained_at}")
    print(f"  Features version: {meta.features_version}")
    print(f"  Training samples: {meta.training_samples}")
    print(f"  Regions: {meta.regions}")
    assert meta.features_version == "v1.1", "Should be v1.1"

    # Test 7: Baseline fallback
    print("\nTest 7: BASELINE FALLBACK")
    print("-" * 40)
    unfitted = PriceQuantileForecaster()
    fallback_preds = unfitted.predict("unknown-region", pred_ts[:3])
    print(f"  Unfitted model returns fallback: {len(fallback_preds)} forecasts")
    for f in fallback_preds:
        assert f.model_type == "baseline_fallback"
        assert f.p90 >= f.p50
    print("  Baseline fallback works: PASS")

    # Test 8: Predictions with/without recent data (v1.1)
    print("\nTest 8: PREDICTIONS WITH/WITHOUT RECENT DATA (v1.1)")
    print("-" * 40)
    # With sufficient recent data
    preds_with = forecaster.predict("us-west", pred_ts[:5], recent)
    print(f"  With recent data: {len(preds_with)} forecasts, p50={preds_with[0].p50:.2f}")
    assert all(f.p90 >= f.p50 for f in preds_with)

    # Without recent data (fallback)
    preds_without = forecaster.predict("us-west", pred_ts[:5], None)
    print(f"  Without recent data: {len(preds_without)} forecasts, p50={preds_without[0].p50:.2f}")
    assert all(f.p90 >= f.p50 for f in preds_without)

    # With insufficient recent data
    insufficient = [p for p in prices if p.region == "us-west"][:3]  # Only 3 hours
    preds_insuff = forecaster.predict("us-west", pred_ts[:5], insufficient)
    print(f"  With insufficient data: {len(preds_insuff)} forecasts, p50={preds_insuff[0].p50:.2f}")
    assert all(f.p90 >= f.p50 for f in preds_insuff)
    print("  All fallback scenarios work: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
