"""Baseline forecasting methods.

This module provides:
1. Simple forecasting approaches as fallbacks
2. Deterministic baseline regression (Ridge/ElasticNet) for point estimates
3. Scenario generators for simulation

The baseline regression model outputs POINT ESTIMATES ONLY (used as p50 anchor).
No quantile logic here - that's handled by price_model/carbon_model.

IMPORTANT:
- Offline batch training ONLY
- Fixed random seeds for determinism
- No learning during execution

v1.1 enables minimal short-horizon lag features (1h, 6h) for accuracy
while preserving predict-time safety and deterministic fallback.
"""

import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ..models import CarbonIntensity, EnergyPrice
from .carbon_model import CarbonForecast
from .price_model import PriceForecast
from .quantile_model import (
    DEFAULT_SEED,
    ModelMetadata,
    build_feature_matrix,
    build_feature_matrix_for_predict,
    check_recent_data_sufficient,
    set_deterministic_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineRegressionConfig:
    """Configuration for baseline regression model.

    Attributes:
        model_type: "ridge" or "elasticnet"
        alpha: Regularization strength
        l1_ratio: ElasticNet mixing parameter (ignored for ridge)
        seed: Random seed for reproducibility
    """
    model_type: str = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5  # Only for ElasticNet
    seed: int = DEFAULT_SEED


class BaselineRegressor:
    """Deterministic baseline regression for point estimates.

    Uses Ridge or ElasticNet regression to produce deterministic
    point estimates. These serve as:
    - p50 anchor for quantile models
    - Fallback when LightGBM unavailable
    - Sanity baseline for comparison

    Output: Point estimates ONLY. No quantiles.
    """

    def __init__(self, config: Optional[BaselineRegressionConfig] = None):
        """Initialize the baseline regressor.

        Args:
            config: Configuration for the model
        """
        self.config = config or BaselineRegressionConfig()
        self._model = None
        self._fitted = False
        self._known_regions: list[str] = []
        self._metadata: Optional[ModelMetadata] = None
        # v1.1: Enable minimal lags for baseline (simpler than price/carbon models)
        # Baseline uses lag_1h and rolling_mean_6h only (no lag_6h)
        # Predict-time: falls back to temporal+region if recent data unavailable
        self._use_lags = True
        self._use_rolling = True
        self._lag_hours = [1]  # baseline: lag_1h only
        self._rolling_hours = [6]  # baseline: rolling_mean_6h only
        self._n_lag_features = 1 + 1  # lag_1h + rolling_mean_6h

    def fit(
        self,
        timestamps: list[datetime],
        regions: list[str],
        values: np.ndarray,
    ) -> "BaselineRegressor":
        """Fit the baseline regression model.

        Args:
            timestamps: Training timestamps
            regions: Training regions
            values: Training target values

        Returns:
            Self for chaining
        """
        from sklearn.linear_model import ElasticNet, Ridge

        set_deterministic_seed(self.config.seed)

        # Store known regions for consistent encoding
        self._known_regions = sorted(set(regions))

        # v1.1: Build feature matrix with minimal lags (lag_1h, rolling_mean_6h)
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

        # Convert to numpy
        X_np = X.values
        y_np = np.array(values)

        # Train model
        if self.config.model_type == "elasticnet":
            self._model = ElasticNet(
                alpha=self.config.alpha,
                l1_ratio=self.config.l1_ratio,
                random_state=self.config.seed,
                max_iter=1000,
            )
        else:
            self._model = Ridge(
                alpha=self.config.alpha,
                random_state=self.config.seed,
            )

        self._model.fit(X_np, y_np)
        self._fitted = True

        self._metadata = ModelMetadata(
            model_type=self.config.model_type,
            trained_at=datetime.utcnow(),
            features_version="v1.1",  # Updated for minimal lags
            training_samples=len(timestamps),
            regions=self._known_regions,
            seed=self.config.seed,
        )

        logger.info(
            f"Fitted baseline {self.config.model_type} model on "
            f"{len(timestamps)} samples, {len(self._known_regions)} regions"
        )

        return self

    def predict(
        self,
        timestamps: list[datetime],
        regions: list[str],
        recent_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict point estimates.

        v1.1: Uses lag features if sufficient recent data available,
        otherwise falls back silently to temporal+region features with
        zeros for lag columns.

        Args:
            timestamps: Prediction timestamps
            regions: Prediction regions
            recent_values: Recent historical values for lagged features (≥6 hours)

        Returns:
            Array of point estimates
        """
        if not self._fitted or self._model is None:
            logger.warning("Model not fitted, returning default values")
            return np.full(len(timestamps), 50.0)

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
            logger.debug("Baseline: using temporal+region only (insufficient recent data)")
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
        predictions = self._model.predict(X_np)

        # Ensure non-negative
        predictions = np.maximum(predictions, 0.0)

        return predictions

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class BaselineForecaster:
    """Simple baseline forecasting methods.

    Methods:
    1. Naive: Use last observed value
    2. Mean: Use historical mean
    3. Seasonal Naive: Use value from same hour yesterday/last week
    4. Simple Moving Average: Average of recent observations

    These are simple fallbacks when sophisticated models aren't available.
    """

    def __init__(self):
        """Initialize the baseline forecaster."""
        pass

    def naive_forecast(
        self,
        last_value: float,
        timestamps: list[datetime],
        region: str,
        std_estimate: float = 5.0,
        is_carbon: bool = False,
    ) -> list:
        """Naive forecast: repeat last observed value.

        Args:
            last_value: The most recent observed value
            timestamps: Timestamps to forecast
            region: Region identifier
            std_estimate: Estimated standard deviation
            is_carbon: If True, return CarbonForecast; else PriceForecast

        Returns:
            List of forecast objects
        """
        forecasts = []
        for ts in timestamps:
            if is_carbon:
                forecasts.append(CarbonForecast(
                    timestamp=ts,
                    region=region,
                    mean=last_value,
                    std=std_estimate,
                ))
            else:
                forecasts.append(PriceForecast(
                    timestamp=ts,
                    region=region,
                    mean=last_value,
                    std=std_estimate,
                ))
        return forecasts

    def mean_forecast(
        self,
        historical_values: list[float],
        timestamps: list[datetime],
        region: str,
        is_carbon: bool = False,
    ) -> list:
        """Mean forecast: use historical average.

        Args:
            historical_values: Past observations
            timestamps: Timestamps to forecast
            region: Region identifier
            is_carbon: If True, return CarbonForecast; else PriceForecast

        Returns:
            List of forecast objects
        """
        if not historical_values:
            mean = 50.0 if not is_carbon else 400.0
            std = 10.0 if not is_carbon else 80.0
        else:
            mean = sum(historical_values) / len(historical_values)
            if len(historical_values) > 1:
                variance = sum((v - mean) ** 2 for v in historical_values) / len(historical_values)
                std = math.sqrt(variance)
            else:
                std = mean * 0.1

        forecasts = []
        for ts in timestamps:
            if is_carbon:
                forecasts.append(CarbonForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))
            else:
                forecasts.append(PriceForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))
        return forecasts

    def seasonal_naive_forecast(
        self,
        historical_data: list,
        timestamps: list[datetime],
        region: str,
        period_hours: int = 24,
        is_carbon: bool = False,
    ) -> list:
        """Seasonal naive: use value from same time in previous period.

        Args:
            historical_data: Past observations (EnergyPrice or CarbonIntensity)
            timestamps: Timestamps to forecast
            region: Region identifier
            period_hours: Seasonal period (24 = daily, 168 = weekly)
            is_carbon: If True, return CarbonForecast; else PriceForecast

        Returns:
            List of forecast objects
        """
        # Index by hour-of-day for lookup
        by_hour: dict[int, list[float]] = {h: [] for h in range(24)}

        for d in historical_data:
            if hasattr(d, 'region') and d.region != region:
                continue
            hour = d.timestamp.hour
            value = d.gco2_per_kwh if is_carbon else d.price_per_mwh
            by_hour[hour].append(value)

        # Calculate mean and std per hour
        hour_stats: dict[int, tuple[float, float]] = {}
        for hour, values in by_hour.items():
            if values:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    std = math.sqrt(variance)
                else:
                    std = mean * 0.1
                hour_stats[hour] = (mean, std)
            else:
                default = 400.0 if is_carbon else 50.0
                hour_stats[hour] = (default, default * 0.1)

        forecasts = []
        for ts in timestamps:
            hour = ts.hour
            mean, std = hour_stats.get(hour, (50.0, 5.0))

            if is_carbon:
                forecasts.append(CarbonForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))
            else:
                forecasts.append(PriceForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))

        return forecasts

    def moving_average_forecast(
        self,
        historical_values: list[float],
        timestamps: list[datetime],
        region: str,
        window: int = 24,
        is_carbon: bool = False,
    ) -> list:
        """Simple moving average forecast.

        Args:
            historical_values: Past observations (most recent last)
            timestamps: Timestamps to forecast
            region: Region identifier
            window: Moving average window size
            is_carbon: If True, return CarbonForecast; else PriceForecast

        Returns:
            List of forecast objects
        """
        if not historical_values:
            mean = 50.0 if not is_carbon else 400.0
            std = 10.0 if not is_carbon else 80.0
        else:
            recent = historical_values[-window:]
            mean = sum(recent) / len(recent)
            if len(recent) > 1:
                variance = sum((v - mean) ** 2 for v in recent) / len(recent)
                std = math.sqrt(variance)
            else:
                std = mean * 0.1

        forecasts = []
        for ts in timestamps:
            if is_carbon:
                forecasts.append(CarbonForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))
            else:
                forecasts.append(PriceForecast(
                    timestamp=ts,
                    region=region,
                    mean=round(mean, 2),
                    std=round(std, 2),
                ))

        return forecasts


def generate_price_scenario(
    start_time: datetime,
    hours: int,
    regions: list[str],
    scenario: str = "normal",
    seed: Optional[int] = None,
) -> list[EnergyPrice]:
    """Generate a price scenario for testing.

    Scenarios:
    - normal: Typical price patterns
    - volatile: High volatility with spikes
    - low: Generally low prices
    - high: Generally high prices
    - peak_valley: Strong peak/off-peak difference

    Args:
        start_time: Start of scenario
        hours: Duration in hours
        regions: List of regions
        scenario: Scenario type
        seed: Random seed

    Returns:
        List of EnergyPrice objects
    """
    if seed is not None:
        random.seed(seed)

    # Strong regional divergence for arbitrage opportunities
    # Base prices vary significantly by region
    base_prices = {
        "us-west": 35,   # Cheap - high renewables
        "us-east": 65,   # Expensive - legacy grid
        "eu-west": 55,   # Moderate
    }

    # Regional volatility factors (some regions more volatile than others)
    regional_volatility = {
        "us-west": 0.25,   # High volatility (solar/wind intermittency)
        "us-east": 0.12,   # Low volatility (stable fossil)
        "eu-west": 0.20,   # Moderate
    }

    # Regional peak multipliers (different peak pricing patterns)
    regional_peak_mult = {
        "us-west": 2.5,    # Very high peak premium
        "us-east": 1.8,    # Moderate peak premium
        "eu-west": 2.2,    # High peak premium
    }

    scenario_mods = {
        "normal": {"multiplier": 1.0, "volatility_mult": 1.0},
        "volatile": {"multiplier": 1.0, "volatility_mult": 2.0},
        "low": {"multiplier": 0.6, "volatility_mult": 0.8},
        "high": {"multiplier": 1.5, "volatility_mult": 1.2},
        "peak_valley": {"multiplier": 1.0, "volatility_mult": 0.8},  # Clear peaks
    }

    mod = scenario_mods.get(scenario, scenario_mods["normal"])
    prices = []

    # Floor start_time to hour boundary for consistent lookups
    start_floored = start_time.replace(minute=0, second=0, microsecond=0)

    for h in range(hours):
        ts = start_floored + timedelta(hours=h)
        hour = ts.hour
        day_of_week = ts.weekday()
        is_peak = 9 <= hour <= 20
        is_weekend = day_of_week >= 5

        for region in regions:
            base = base_prices.get(region, 50) * mod["multiplier"]
            vol = regional_volatility.get(region, 0.15) * mod["volatility_mult"]

            # Apply peak pricing with regional variation
            if is_peak and not is_weekend:
                peak_mult = regional_peak_mult.get(region, 2.0)
                if scenario == "peak_valley":
                    base *= peak_mult  # Full peak effect
                else:
                    base *= 1 + (peak_mult - 1) * 0.5  # Partial peak effect
            elif not is_peak:
                # Off-peak discount
                base *= 0.6 if scenario == "peak_valley" else 0.8

            # Weekend discount
            if is_weekend:
                base *= 0.75

            noise = 1 + random.gauss(0, vol)
            price = max(5, base * noise)

            prices.append(EnergyPrice(
                timestamp=ts,
                region=region,
                price_per_mwh=round(price, 2),
            ))

    return prices


def generate_carbon_scenario(
    start_time: datetime,
    hours: int,
    regions: list[str],
    scenario: str = "normal",
    seed: Optional[int] = None,
) -> list[CarbonIntensity]:
    """Generate a carbon intensity scenario for testing.

    Scenarios:
    - normal: Typical patterns
    - clean: Low carbon (high renewables)
    - dirty: High carbon (fossil heavy)
    - variable: High variability (intermittent renewables)

    Args:
        start_time: Start of scenario
        hours: Duration in hours
        regions: List of regions
        scenario: Scenario type
        seed: Random seed

    Returns:
        List of CarbonIntensity objects
    """
    if seed is not None:
        random.seed(seed)

    # Strong regional divergence in carbon intensity
    # Reflects different grid mixes
    base_intensity = {
        "us-west": 200,   # Very clean - high solar/wind
        "us-east": 550,   # Dirty - coal/gas heavy
        "eu-west": 280,   # Moderate - mixed with nuclear
    }

    # Regional carbon volatility (renewable variability)
    regional_carbon_volatility = {
        "us-west": 0.35,   # High - solar/wind intermittency
        "us-east": 0.10,   # Low - stable fossil baseline
        "eu-west": 0.25,   # Moderate
    }

    # Solar impact by region (how much solar affects carbon)
    regional_solar_factor = {
        "us-west": 0.5,   # 50% reduction during solar hours
        "us-east": 0.9,   # Only 10% reduction (less solar)
        "eu-west": 0.65,  # 35% reduction
    }

    scenario_mods = {
        "normal": {"multiplier": 1.0, "volatility_mult": 1.0},
        "clean": {"multiplier": 0.5, "volatility_mult": 1.2},
        "dirty": {"multiplier": 1.5, "volatility_mult": 0.8},
        "variable": {"multiplier": 1.0, "volatility_mult": 2.5},
    }

    mod = scenario_mods.get(scenario, scenario_mods["normal"])
    data = []

    # Floor start_time to hour boundary for consistent lookups
    start_floored = start_time.replace(minute=0, second=0, microsecond=0)

    for h in range(hours):
        ts = start_floored + timedelta(hours=h)
        hour = ts.hour

        for region in regions:
            base = base_intensity.get(region, 400) * mod["multiplier"]
            vol = regional_carbon_volatility.get(region, 0.15) * mod["volatility_mult"]

            # Solar hours have lower carbon (varies by region)
            if 10 <= hour <= 16:
                solar_factor = regional_solar_factor.get(region, 0.8)
                base *= solar_factor
            elif 6 <= hour < 10 or 16 < hour <= 20:
                # Partial solar effect at edges of day
                partial_factor = (regional_solar_factor.get(region, 0.8) + 1.0) / 2
                base *= partial_factor

            noise = 1 + random.gauss(0, vol)
            intensity = max(50, base * noise)

            data.append(CarbonIntensity(
                timestamp=ts,
                region=region,
                gco2_per_kwh=round(intensity, 1),
            ))

    return data


# ============================================================================
# INLINE VALIDATION
# ============================================================================
# Run with: python -c "from aurelius.forecasting.baseline import _run_validation; _run_validation()"

def _run_validation():
    """Validate baseline regression."""
    print("=" * 60)
    print("Baseline Regression Validation")
    print("=" * 60)

    # Test 1: Ridge regression
    print("\nTest 1: RIDGE REGRESSION")
    print("-" * 40)
    config = BaselineRegressionConfig(model_type="ridge", alpha=1.0, seed=42)
    regressor = BaselineRegressor(config)

    # Generate training data
    timestamps = [datetime(2025, 1, 1, h % 24) + timedelta(days=h // 24) for h in range(168)]
    regions = ["us-west"] * 168
    values = np.array([50 + 10 * np.sin(2 * np.pi * h / 24) + np.random.randn() * 2 for h in range(168)])

    regressor.fit(timestamps, regions, values)
    print(f"  Fitted: {regressor.is_fitted}")
    assert regressor.is_fitted

    # Predict (without recent values - uses only temporal/region features)
    pred_ts = [datetime(2025, 1, 8, h) for h in range(24)]
    pred_regions = ["us-west"] * 24
    preds = regressor.predict(pred_ts, pred_regions)
    print(f"  Predictions: {preds[:5].round(2)}...")
    assert len(preds) == 24
    assert all(p >= 0 for p in preds)

    # Test 2: ElasticNet regression
    print("\nTest 2: ELASTICNET REGRESSION")
    print("-" * 40)
    config = BaselineRegressionConfig(model_type="elasticnet", alpha=0.5, l1_ratio=0.5, seed=42)
    regressor = BaselineRegressor(config)
    regressor.fit(timestamps, regions, values)
    print(f"  Fitted: {regressor.is_fitted}")
    assert regressor.is_fitted

    preds = regressor.predict(pred_ts, pred_regions)
    print(f"  Predictions: {preds[:5].round(2)}...")

    # Test 3: Determinism
    print("\nTest 3: DETERMINISM")
    print("-" * 40)
    config = BaselineRegressionConfig(model_type="ridge", seed=42)
    reg1 = BaselineRegressor(config)
    reg1.fit(timestamps, regions, values)
    preds1 = reg1.predict(pred_ts, pred_regions)

    reg2 = BaselineRegressor(config)
    reg2.fit(timestamps, regions, values)
    preds2 = reg2.predict(pred_ts, pred_regions)

    assert np.allclose(preds1, preds2), "Same seed should produce identical predictions"
    print("  Same seed produces identical predictions: PASS")

    # Test 4: Metadata (v1.1)
    print("\nTest 4: METADATA (v1.1)")
    print("-" * 40)
    meta = regressor.metadata
    assert meta is not None
    print(f"  Model type: {meta.model_type}")
    print(f"  Trained at: {meta.trained_at}")
    print(f"  Features version: {meta.features_version}")
    print(f"  Training samples: {meta.training_samples}")
    assert meta.features_version == "v1.1", "Should be v1.1"

    # Test 5: Prediction with recent data (v1.1)
    print("\nTest 5: PREDICTION WITH RECENT DATA (v1.1)")
    print("-" * 40)
    config = BaselineRegressionConfig(model_type="ridge", seed=42)
    regressor = BaselineRegressor(config)
    regressor.fit(timestamps, regions, values)

    # With sufficient recent data (>=6 hours)
    recent_sufficient = np.array([50.0, 52.0, 48.0, 55.0, 53.0, 51.0, 49.0, 54.0])
    preds_with = regressor.predict(pred_ts, pred_regions, recent_values=recent_sufficient)
    print(f"  With 8 hours recent data: {preds_with[:3].round(2)}...")
    assert len(preds_with) == 24
    assert all(p >= 0 for p in preds_with)

    # Test 6: Prediction without recent data (v1.1 fallback)
    print("\nTest 6: PREDICTION WITHOUT RECENT DATA (v1.1)")
    print("-" * 40)
    # With insufficient recent data (<6 hours)
    recent_insufficient = np.array([50.0, 52.0])
    preds_without = regressor.predict(pred_ts, pred_regions, recent_values=recent_insufficient)
    print(f"  With 2 hours recent data (fallback): {preds_without[:3].round(2)}...")
    assert len(preds_without) == 24
    assert all(p >= 0 for p in preds_without)

    # With None recent data
    preds_none = regressor.predict(pred_ts, pred_regions, recent_values=None)
    print(f"  With None recent data (fallback): {preds_none[:3].round(2)}...")
    assert len(preds_none) == 24
    assert all(p >= 0 for p in preds_none)
    print("  Fallback predictions work: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
