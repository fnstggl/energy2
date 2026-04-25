"""Carbon intensity forecasting model.

This module provides:
- LightGBM quantile regressors for p50 and p90 forecasts
- Simple interpretable forecasting using rolling averages and seasonality
- Fallback to baseline when LightGBM unavailable

Produces p50 and p90 forecasts. Does NOT apply safety logic beyond p90 >= p50.
Mirrors price_model behavior symmetrically.

IMPORTANT:
- Offline batch training ONLY
- Fixed random seeds for determinism
- No learning during execution

v1.1 enables minimal short-horizon lag features (1h, 6h) for accuracy
while preserving predict-time safety and deterministic fallback.
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ..models import CarbonIntensity
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
class CarbonForecast:
    """A carbon intensity forecast with uncertainty bounds.

    Attributes:
        timestamp: The forecasted hour
        region: Geographic region
        mean: Mean predicted carbon intensity (gCO2/kWh)
        std: Standard deviation of prediction
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
class CarbonQuantileForecast:
    """A carbon intensity forecast with quantile predictions.

    Attributes:
        timestamp: The forecasted hour
        region: Geographic region
        p50: Median predicted carbon intensity (gCO2/kWh)
        p90: 90th percentile predicted carbon intensity (gCO2/kWh)
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
class CarbonModelConfig:
    """Configuration for carbon forecasting model.

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


class CarbonQuantileForecaster:
    """LightGBM quantile forecaster for carbon intensity.

    Produces p50 (median) and p90 (upper bound) forecasts.
    Falls back to baseline if LightGBM unavailable.

    Training discipline:
    - Offline batch training only
    - Fixed random seeds
    - Time-based cross-validation
    - Deterministic outputs
    """

    # Default carbon intensity by region (gCO2/kWh)
    REGION_DEFAULTS = {
        "us-west": 350,   # More renewables
        "us-east": 450,   # More fossil
        "eu-west": 300,   # Good renewable mix
        "eu-north": 150,  # High hydro/nuclear
        "asia-east": 550, # Coal heavy
    }

    _DEFAULT_CORRECTIONS_REL = Path("data/ml_artifacts/forecast_corrections_v1.json")

    def __init__(
        self,
        config: Optional[CarbonModelConfig] = None,
        corrections_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the carbon quantile forecaster.

        Args:
            config: Model configuration
            corrections_path: Path to forecast_corrections_v1.json artifact.
                Defaults to <package>/data/ml_artifacts/forecast_corrections_v1.json.
                Set to False to disable bias correction loading.
        """
        self.config = config or CarbonModelConfig()
        self._model_p50: Any = None
        self._model_p90: Any = None
        self._fitted = False
        self._known_regions: list[str] = []
        self._metadata: Optional[ModelMetadata] = None
        self._baseline_mean: float = 400.0  # Default baseline
        # v1.1: Enable minimal lags for improved accuracy with safe fallback
        # Carbon model mirrors price_model: lag_1h, lag_6h, rolling_mean_6h
        # Predict-time: falls back to temporal+region if recent data unavailable
        self._use_lags = True
        self._use_rolling = True
        self._lag_hours = [1, 6]  # lag_1h, lag_6h
        self._rolling_hours = [6]  # rolling_mean_6h only
        # Bias-correction lookup: {region: {hour: bias}}
        self._p50_bias: dict[str, dict[int, float]] = {}
        self._corrections_loaded: bool = False
        if corrections_path is not False:
            self._load_corrections(corrections_path)

    def _load_corrections(self, path: Optional[Union[str, Path]]) -> None:
        """Load forecast bias corrections from artifact file if available."""
        if path is None:
            pkg_root = Path(__file__).parent.parent
            path = pkg_root / self._DEFAULT_CORRECTIONS_REL
        path = Path(path)
        if not path.exists():
            logger.debug(f"CarbonQuantileForecaster: no corrections artifact at {path}; skipping")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            bias: dict[str, dict[int, float]] = {}
            for bucket in data.get("buckets", []):
                region = bucket.get("region")
                hour = bucket.get("hour_utc")
                carbon = bucket.get("carbon", {})
                if region and hour is not None and "mean_error" in carbon:
                    bias.setdefault(region, {})[int(hour)] = float(carbon["mean_error"])
            self._p50_bias = bias
            self._corrections_loaded = True
            logger.info(
                f"CarbonQuantileForecaster: loaded bias corrections from {path} "
                f"({sum(len(v) for v in bias.values())} region-hour entries)"
            )
        except Exception as exc:
            logger.warning(f"CarbonQuantileForecaster: failed to load corrections from {path}: {exc}")

    def fit(self, carbon_data: list[CarbonIntensity]) -> "CarbonQuantileForecaster":
        """Fit the quantile models on historical carbon data.

        Args:
            carbon_data: List of historical CarbonIntensity objects

        Returns:
            Self for chaining
        """
        set_deterministic_seed(self.config.seed)

        # Extract training data
        timestamps = [c.timestamp for c in carbon_data]
        regions = [c.region for c in carbon_data]
        values = np.array([c.gco2_per_kwh for c in carbon_data])

        self._known_regions = sorted(set(regions))
        self._baseline_mean = float(np.mean(values)) if len(values) > 0 else 400.0

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
        logger.info("Training p50 (median) carbon model...")
        self._model_p50 = train_lightgbm_quantile(
            X_np, values, QUANTILE_P50,
            seed=self.config.seed,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
        )

        # Train p90 model
        logger.info("Training p90 (upper bound) carbon model...")
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
            training_samples=len(carbon_data),
            regions=self._known_regions,
            seed=self.config.seed,
        )

        logger.info(
            f"Fitted carbon quantile model ({model_type}) on "
            f"{len(carbon_data)} samples, {len(self._known_regions)} regions"
        )

        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_data: Optional[list[CarbonIntensity]] = None,
    ) -> list[CarbonQuantileForecast]:
        """Generate quantile carbon intensity forecasts.

        v1.1: Uses lag features if sufficient recent data available (≥6 hours),
        otherwise falls back silently to temporal+region features.

        Args:
            region: Region to forecast
            timestamps: List of future timestamps to predict
            recent_data: Recent carbon data for feature computation (≥6 hours for lags)

        Returns:
            List of CarbonQuantileForecast objects
        """
        if not self._fitted:
            logger.warning("Model not fitted, using baseline fallback")
            return self._baseline_predict(timestamps, region)

        # v1.1: Extract recent values for this region
        recent_values = None
        if recent_data:
            region_data = [c.gco2_per_kwh for c in recent_data if c.region == region]
            if len(region_data) >= MIN_RECENT_HOURS:
                recent_values = np.array(region_data[-48:])  # Use up to 48 hours

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
            logger.debug("Carbon model: using temporal+region only (insufficient recent data)")
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

        # Apply bias correction to p50 if artifact was loaded
        region_bias = self._p50_bias.get(region, {})

        # Build forecast objects
        model_type = self._metadata.model_type if self._metadata else "unknown"
        forecasts = []
        for i, ts in enumerate(timestamps):
            p50_raw = p50_preds[i]
            if region_bias:
                hour_bias = region_bias.get(ts.hour, 0.0)
                p50_corrected = p50_raw - hour_bias
            else:
                p50_corrected = p50_raw
            p90 = max(p90_preds[i], p50_corrected)
            forecasts.append(CarbonQuantileForecast(
                timestamp=ts,
                region=region,
                p50=round(p50_corrected, 1),
                p90=round(p90, 1),
                model_type=model_type,
                features_version="v1.1",
            ))

        return forecasts

    def predict_range(
        self,
        region: str,
        start_time: datetime,
        hours: int,
        recent_data: Optional[list[CarbonIntensity]] = None,
    ) -> list[CarbonQuantileForecast]:
        """Generate quantile forecasts for a time range.

        Args:
            region: Region to forecast
            start_time: Start of forecast window
            hours: Number of hours to forecast
            recent_data: Recent carbon data

        Returns:
            List of CarbonQuantileForecast objects
        """
        timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
        return self.predict(region, timestamps, recent_data)

    def _baseline_predict(
        self,
        timestamps: list[datetime],
        region: str,
    ) -> list[CarbonQuantileForecast]:
        """Baseline prediction when model not trained.

        Args:
            timestamps: Timestamps to predict
            region: Region identifier

        Returns:
            List of baseline forecasts
        """
        # Use regional default if available
        default = self.REGION_DEFAULTS.get(region, self._baseline_mean)

        forecasts = []
        for ts in timestamps:
            # Simple baseline: use mean with 30% uplift for p90
            p50 = default
            p90 = default * 1.3

            forecasts.append(CarbonQuantileForecast(
                timestamp=ts,
                region=region,
                p50=round(p50, 1),
                p90=round(p90, 1),
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
        forecasts: list[CarbonQuantileForecast],
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to disk using joblib."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"CarbonQuantileForecaster saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CarbonQuantileForecaster":
        """Load a persisted model from disk."""
        import joblib
        path = Path(path)
        obj = joblib.load(path)
        logger.info(f"CarbonQuantileForecaster loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Holdout validation
    # ------------------------------------------------------------------

    def validate_coverage(
        self,
        holdout_data: list[CarbonIntensity],
    ) -> dict:
        """Compute empirical p90 coverage on a holdout set.

        Args:
            holdout_data: CarbonIntensity records withheld from training.

        Returns:
            dict with empirical_p90_coverage, n_samples, meets_88pct_threshold.
        """
        if not self._fitted:
            raise RuntimeError("validate_coverage: model must be fitted first")
        if not holdout_data:
            raise ValueError("validate_coverage: holdout_data is empty")

        by_region: dict[str, list[CarbonIntensity]] = {}
        for c in holdout_data:
            by_region.setdefault(c.region, []).append(c)

        covered_count = 0
        total_count = 0

        for region, recs in by_region.items():
            recs_sorted = sorted(recs, key=lambda r: r.timestamp)
            timestamps = [r.timestamp for r in recs_sorted]
            actuals = [r.gco2_per_kwh for r in recs_sorted]
            forecasts = self.predict(region, timestamps, recent_data=None)
            for f, actual in zip(forecasts, actuals):
                total_count += 1
                if actual <= f.p90:
                    covered_count += 1

        coverage = covered_count / total_count if total_count > 0 else 0.0
        return {
            "empirical_p90_coverage": round(coverage, 4),
            "n_samples": total_count,
            "meets_88pct_threshold": coverage >= 0.88,
        }


class CarbonForecaster:
    """Forecasts carbon intensity using interpretable methods.

    Carbon intensity varies with:
    - Hour of day (solar/wind availability)
    - Day of week (industrial demand)
    - Region (grid mix)

    The model is similar to PriceForecaster but tuned for carbon signals.
    This is the legacy forecaster. For quantile forecasts, use CarbonQuantileForecaster.
    """

    # Default carbon intensity by region (gCO2/kWh)
    REGION_DEFAULTS = {
        "us-west": 350,   # More renewables
        "us-east": 450,   # More fossil
        "eu-west": 300,   # Good renewable mix
        "eu-north": 150,  # High hydro/nuclear
        "asia-east": 550, # Coal heavy
    }

    def __init__(
        self,
        lookback_hours: int = 168,
        rolling_window: int = 24,
    ):
        """Initialize the forecaster.

        Args:
            lookback_hours: Hours of history to consider
            rolling_window: Window size for rolling average
        """
        self.lookback_hours = lookback_hours
        self.rolling_window = rolling_window

        self._hourly_factors: dict[str, dict[int, float]] = {}
        self._dow_factors: dict[str, dict[int, float]] = {}
        self._base_levels: dict[str, float] = {}
        self._volatilities: dict[str, float] = {}

    def fit(self, carbon_data: list[CarbonIntensity]) -> "CarbonForecaster":
        """Fit the model on historical carbon intensity data.

        Args:
            carbon_data: List of historical CarbonIntensity objects

        Returns:
            Self for chaining
        """
        by_region: dict[str, list[CarbonIntensity]] = {}
        for c in carbon_data:
            if c.region not in by_region:
                by_region[c.region] = []
            by_region[c.region].append(c)

        for region, region_data in by_region.items():
            region_data.sort(key=lambda x: x.timestamp)

            values = [c.gco2_per_kwh for c in region_data]
            base_level = sum(values) / len(values) if values else self.REGION_DEFAULTS.get(region, 400)
            self._base_levels[region] = base_level

            # Calculate volatility
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                self._volatilities[region] = math.sqrt(variance)
            else:
                self._volatilities[region] = base_level * 0.15

            # Hour-of-day factors (carbon often lower during sunny hours due to solar)
            hourly_sums: dict[int, float] = {h: 0.0 for h in range(24)}
            hourly_counts: dict[int, int] = {h: 0 for h in range(24)}

            for c in region_data:
                hour = c.timestamp.hour
                hourly_sums[hour] += c.gco2_per_kwh
                hourly_counts[hour] += 1

            self._hourly_factors[region] = {}
            for hour in range(24):
                if hourly_counts[hour] > 0:
                    avg = hourly_sums[hour] / hourly_counts[hour]
                    self._hourly_factors[region][hour] = avg / base_level if base_level > 0 else 1.0
                else:
                    self._hourly_factors[region][hour] = 1.0

            # Day-of-week factors
            dow_sums: dict[int, float] = {d: 0.0 for d in range(7)}
            dow_counts: dict[int, int] = {d: 0 for d in range(7)}

            for c in region_data:
                dow = c.timestamp.weekday()
                dow_sums[dow] += c.gco2_per_kwh
                dow_counts[dow] += 1

            self._dow_factors[region] = {}
            for dow in range(7):
                if dow_counts[dow] > 0:
                    avg = dow_sums[dow] / dow_counts[dow]
                    self._dow_factors[region][dow] = avg / base_level if base_level > 0 else 1.0
                else:
                    self._dow_factors[region][dow] = 1.0

        logger.info(f"Fitted carbon model for {len(by_region)} regions")
        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_data: Optional[list[CarbonIntensity]] = None,
    ) -> list[CarbonForecast]:
        """Generate carbon intensity forecasts.

        Args:
            region: Region to forecast
            timestamps: List of future timestamps
            recent_data: Recent carbon data for adjustment

        Returns:
            List of CarbonForecast objects
        """
        forecasts = []

        base_level = self._base_levels.get(region, self.REGION_DEFAULTS.get(region, 400))
        hourly_factors = self._hourly_factors.get(region, {h: 1.0 for h in range(24)})
        dow_factors = self._dow_factors.get(region, {d: 1.0 for d in range(7)})
        volatility = self._volatilities.get(region, base_level * 0.15)

        # Rolling average from recent data
        if recent_data:
            recent_values = [
                c.gco2_per_kwh for c in recent_data
                if c.region == region
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

            hourly_factor = hourly_factors.get(hour, 1.0)
            dow_factor = dow_factors.get(dow, 1.0)

            mean_intensity = rolling_avg * hourly_factor * dow_factor

            # Uncertainty increases with horizon
            hours_ahead = max(1, int((ts - datetime.utcnow()).total_seconds() / 3600))
            uncertainty_factor = 1 + 0.03 * math.sqrt(hours_ahead)

            forecasts.append(CarbonForecast(
                timestamp=ts,
                region=region,
                mean=round(mean_intensity, 1),
                std=round(volatility * uncertainty_factor, 1),
            ))

        return forecasts

    def predict_range(
        self,
        region: str,
        start_time: datetime,
        hours: int,
        recent_data: Optional[list[CarbonIntensity]] = None,
    ) -> list[CarbonForecast]:
        """Generate forecasts for a time range.

        Args:
            region: Region to forecast
            start_time: Start of forecast window
            hours: Number of hours to forecast
            recent_data: Recent carbon data

        Returns:
            List of CarbonForecast objects
        """
        timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
        return self.predict(region, timestamps, recent_data)

    def generate_synthetic_history(
        self,
        start_time: datetime,
        hours: int,
        regions: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> list[CarbonIntensity]:
        """Generate synthetic historical carbon data.

        Useful for simulation when real data is unavailable.

        Args:
            start_time: Start of history
            hours: Number of hours
            regions: Regions to generate
            seed: Random seed

        Returns:
            List of CarbonIntensity objects
        """
        import random
        if seed is not None:
            random.seed(seed)

        regions = regions or list(self.REGION_DEFAULTS.keys())
        data = []

        for hour_offset in range(hours):
            ts = start_time + timedelta(hours=hour_offset)
            hour = ts.hour
            dow = ts.weekday()

            for region in regions:
                base = self.REGION_DEFAULTS.get(region, 400)

                # Solar effect (lower during midday)
                if 10 <= hour <= 16:
                    solar_factor = 0.8
                else:
                    solar_factor = 1.0

                # Wind effect (some random variation)
                wind_factor = 0.9 + random.random() * 0.2

                # Weekend slightly lower
                dow_factor = 0.95 if dow >= 5 else 1.0

                intensity = base * solar_factor * wind_factor * dow_factor
                intensity *= (0.9 + random.random() * 0.2)  # noise

                data.append(CarbonIntensity(
                    timestamp=ts,
                    region=region,
                    gco2_per_kwh=round(max(50, intensity), 1),
                ))

        logger.info(f"Generated {len(data)} synthetic carbon records")
        return data

    def forecasts_to_dict(
        self,
        forecasts: list[CarbonForecast],
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
# Run with: python -c "from aurelius.forecasting.carbon_model import _run_validation; _run_validation()"

def _run_validation():
    """Validate carbon quantile forecaster."""
    print("=" * 60)
    print("Carbon Quantile Forecaster Validation")
    print("=" * 60)

    # Generate synthetic training data
    print("\nGenerating training data...")
    from .baseline import generate_carbon_scenario
    carbon_data = generate_carbon_scenario(
        start_time=datetime(2025, 1, 1),
        hours=168 * 4,  # 4 weeks
        regions=["us-west", "us-east"],
        scenario="normal",
        seed=42,
    )
    print(f"  Generated {len(carbon_data)} carbon records")

    # Test 1: Fit quantile forecaster
    print("\nTest 1: FIT QUANTILE FORECASTER")
    print("-" * 40)
    config = CarbonModelConfig(seed=42, n_estimators=50)
    forecaster = CarbonQuantileForecaster(config)
    forecaster.fit(carbon_data)
    print(f"  Fitted: {forecaster.is_fitted}")
    assert forecaster.is_fitted

    # Test 2: Predict quantiles
    print("\nTest 2: PREDICT QUANTILES")
    print("-" * 40)
    pred_ts = [datetime(2025, 2, 1, h) for h in range(24)]
    recent = [c for c in carbon_data if c.region == "us-west"][-48:]
    forecasts = forecaster.predict("us-west", pred_ts, recent)
    print(f"  Generated {len(forecasts)} forecasts")
    for f in forecasts[:3]:
        print(f"    {f.timestamp.hour}:00 - p50={f.p50:.1f}, p90={f.p90:.1f}")

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
    forecaster1 = CarbonQuantileForecaster(CarbonModelConfig(seed=42, n_estimators=50))
    forecaster1.fit(carbon_data)
    preds1 = forecaster1.predict("us-west", pred_ts[:5], recent)

    forecaster2 = CarbonQuantileForecaster(CarbonModelConfig(seed=42, n_estimators=50))
    forecaster2.fit(carbon_data)
    preds2 = forecaster2.predict("us-west", pred_ts[:5], recent)

    for i in range(len(preds1)):
        assert preds1[i].p50 == preds2[i].p50, "p50 should be identical"
        assert preds1[i].p90 == preds2[i].p90, "p90 should be identical"
    print("  Same seed produces identical predictions: PASS")

    # Test 5: Regional variation
    print("\nTest 5: REGIONAL VARIATION")
    print("-" * 40)
    west_preds = forecaster.predict("us-west", pred_ts[:5], recent)
    east_recent = [c for c in carbon_data if c.region == "us-east"][-48:]
    east_preds = forecaster.predict("us-east", pred_ts[:5], east_recent)

    west_avg = np.mean([f.p50 for f in west_preds])
    east_avg = np.mean([f.p50 for f in east_preds])
    print(f"  us-west avg p50: {west_avg:.1f}")
    print(f"  us-east avg p50: {east_avg:.1f}")
    # us-east should be dirtier in our scenario
    if east_avg > west_avg:
        print("  Regional carbon difference detected: PASS")
    else:
        print("  Warning: Expected us-east to be dirtier")

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
    unfitted = CarbonQuantileForecaster()
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
    print(f"  With recent data: {len(preds_with)} forecasts, p50={preds_with[0].p50:.1f}")
    assert all(f.p90 >= f.p50 for f in preds_with)

    # Without recent data (fallback)
    preds_without = forecaster.predict("us-west", pred_ts[:5], None)
    print(f"  Without recent data: {len(preds_without)} forecasts, p50={preds_without[0].p50:.1f}")
    assert all(f.p90 >= f.p50 for f in preds_without)

    # With insufficient recent data
    insufficient = [c for c in carbon_data if c.region == "us-west"][:3]  # Only 3 hours
    preds_insuff = forecaster.predict("us-west", pred_ts[:5], insufficient)
    print(f"  With insufficient data: {len(preds_insuff)} forecasts, p50={preds_insuff[0].p50:.1f}")
    assert all(f.p90 >= f.p50 for f in preds_insuff)
    print("  All fallback scenarios work: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
