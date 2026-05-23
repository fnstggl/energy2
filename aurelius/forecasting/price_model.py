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

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ..models import EnergyPrice
from .quantile_model import (
    DEFAULT_SEED,
    MIN_RECENT_HOURS,
    QUANTILE_P50,
    QUANTILE_P90,
    ModelMetadata,
    build_feature_matrix,
    build_feature_matrix_for_predict,
    build_weather_lookup,
    check_recent_data_sufficient,
    predict_with_fallback,
    set_deterministic_seed,
    train_lightgbm_quantile,
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
        include_volatility_features: Enable volatility regime features (spike detection,
            rolling_std, price_momentum). Recommended for high-volatility grids like ERCOT
            winter. Adds 6 features; requires slightly more training data but significantly
            improves detection of price-spike regimes.
        num_leaves: LightGBM num_leaves (overrides max_depth when set to > 0).
            Default 0 means use max_depth.
        include_weather_features: Enable weather features (temperature, HDD, CDD, wind
            speed) when a weather DataFrame is supplied at fit/predict time. When no
            weather data is available the model degrades gracefully to price-only mode.
            Primarily improves ERCOT winter (cold-snap price spike prediction) and
            summer heat-wave demand spikes. Default True — features are only active when
            weather_df is actually provided.
        min_child_samples: Minimum samples per LightGBM leaf node. Higher values
            increase regularization and reduce overfitting, especially important when
            adding weather features to the training set. Default 20 (LightGBM default).
        reg_lambda: L2 regularization coefficient for LightGBM. Non-zero values
            shrink leaf weights toward zero, reducing overfit on small training sets.
            Default 0.0 (disabled).
        include_rank_features: Enable v5.0 price rank/percentile features
            (rolling_mean_168h, price_range_position_168h, below_p10_168h,
            price_vs_mean_168h). These encode "is the current price cheap relative
            to recent history?" which is the core routing signal for multi-region
            optimization. Default True.
    """
    seed: int = DEFAULT_SEED
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    use_baseline_fallback: bool = True
    include_volatility_features: bool = True
    num_leaves: int = 63
    include_weather_features: bool = True
    min_child_samples: int = 20
    reg_lambda: float = 0.0
    include_rank_features: bool = False


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

    # Default location relative to the package root for bias-correction artifacts.
    _DEFAULT_CORRECTIONS_REL = Path("data/ml_artifacts/forecast_corrections_v1.json")

    def __init__(
        self,
        config: Optional[PriceModelConfig] = None,
        corrections_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the price quantile forecaster.

        Args:
            config: Model configuration
            corrections_path: Path to forecast_corrections_v1.json artifact.
                Defaults to <package>/data/ml_artifacts/forecast_corrections_v1.json.
                Set to False to disable bias correction loading.
        """
        self.config = config or PriceModelConfig()
        self._model_p50: Any = None
        self._model_p90: Any = None
        self._fitted = False
        self._known_regions: list[str] = []
        self._metadata: Optional[ModelMetadata] = None
        self._baseline_mean: float = 50.0  # Default baseline
        # v1.2: Full seasonal lag set for accurate long-horizon prediction.
        # Electricity prices have strong daily (24h) and weekly (168h) cycles —
        # the previous lag_1h/lag_6h-only set had no long-horizon signal and
        # collapsed to global mean for predictions >6h out. Adding lag_24h and
        # lag_168h lets the model learn "today at 4pm looks like yesterday at
        # 4pm" and "next Monday at 9am looks like this Monday at 9am".
        # Rolling_24h captures the current price-regime (e.g. cold-snap week).
        # Requires recent_context of >=168h per region at predict time; the
        # backtesting engine handles this via per-region context slicing.
        self._use_lags = True
        self._use_rolling = True
        # v2.0: Volatility regime features for price spike detection.
        self._use_volatility = self.config.include_volatility_features
        # v3.0: Weather features for temperature/demand-driven price signals.
        self._use_weather = self.config.include_weather_features
        # v5.0: Price rank features — encode cheap/expensive regime for routing.
        # Active when config.include_rank_features is True (default).
        self._use_rank = self.config.include_rank_features
        # v5.0: add lag_336h (2-week bi-weekly lag) when rank features are enabled.
        # Rank features require the 168h trailing window, and lag_336h provides a
        # second weekly reference point (same-hour 2 weeks ago).
        # When rank features are off, use the v2.0 lag set for backward compat.
        if self._use_rank:
            self._lag_hours = [1, 6, 24, 168, 336]
        else:
            self._lag_hours = [1, 6, 24, 168]
        self._rolling_hours = [6, 24]
        # Stored weather lookup built during fit() — used to ensure training-time
        # weather features are available for the predict() call path when the
        # caller does not supply a separate predict-time weather_df.
        self._train_weather_lookup: dict = {}
        # Bias-correction lookup: {region: {hour: bias}} loaded from artifact
        self._p50_bias: dict[str, dict[int, float]] = {}
        self._corrections_loaded: bool = False
        if corrections_path is not False:
            self._load_corrections(corrections_path)

    def _load_corrections(self, path: Optional[Union[str, Path]]) -> None:
        """Load forecast bias corrections from artifact file if available."""
        if path is None:
            # Resolve relative to package root
            pkg_root = Path(__file__).parent.parent
            path = pkg_root / self._DEFAULT_CORRECTIONS_REL
        path = Path(path)
        if not path.exists():
            logger.debug(f"PriceQuantileForecaster: no corrections artifact at {path}; skipping")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            bias: dict[str, dict[int, float]] = {}
            for bucket in data.get("buckets", []):
                region = bucket.get("region")
                hour = bucket.get("hour_utc")
                if region is None or hour is None:
                    continue
                # Primary schema: energy_cost_p50_bias (from train_forecast_corrections)
                bias_val = bucket.get("energy_cost_p50_bias")
                # Legacy schema: energy_cost.mean_error
                if bias_val is None:
                    ec = bucket.get("energy_cost", {})
                    bias_val = ec.get("mean_error") if isinstance(ec, dict) else None
                if bias_val is not None:
                    bias.setdefault(region, {})[int(hour)] = float(bias_val)
            self._p50_bias = bias
            self._corrections_loaded = True
            logger.info(
                f"PriceQuantileForecaster: loaded bias corrections from {path} "
                f"({sum(len(v) for v in bias.values())} region-hour entries)"
            )
        except Exception as exc:
            logger.warning(f"PriceQuantileForecaster: failed to load corrections from {path}: {exc}")

    def fit(
        self,
        prices: list[EnergyPrice],
        weather_df: Optional[Any] = None,
    ) -> "PriceQuantileForecaster":
        """Fit the quantile models on historical price data.

        Args:
            prices:     List of historical EnergyPrice objects.
            weather_df: Optional canonical weather DataFrame (columns: timestamp,
                        region, temperature_c, hdd_f, cdd_f, wind_speed_ms,
                        temp_rolling_24h_c, temp_delta_24h_c).  When provided and
                        include_weather_features is True, weather features are joined
                        into the training matrix.  None → price-only mode (no change
                        in behaviour vs. v2.0).

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

        # Build weather lookup once and cache for predict-time reuse
        self._train_weather_lookup = {}
        effective_weather_lookup: dict = {}
        if self._use_weather and weather_df is not None and not (
            hasattr(weather_df, "empty") and weather_df.empty
        ):
            effective_weather_lookup = build_weather_lookup(weather_df)
            self._train_weather_lookup = effective_weather_lookup
            logger.info(
                f"Weather features enabled: {len(effective_weather_lookup)} hourly entries, "
                f"regions={list(weather_df['region'].unique()) if 'region' in weather_df else '?'}"
            )

        X = build_feature_matrix(
            timestamps,
            regions,
            values,
            include_lags=self._use_lags,
            include_rolling=self._use_rolling,
            include_volatility=self._use_volatility,
            include_rank_features=self._use_rank,
            known_regions=self._known_regions,
            lag_hours=self._lag_hours,
            rolling_hours=self._rolling_hours,
            weather_lookup=effective_weather_lookup,
        )
        X_np = X.values

        num_leaves = getattr(self.config, "num_leaves", 0)
        min_child_samples = getattr(self.config, "min_child_samples", 20)
        reg_lambda = getattr(self.config, "reg_lambda", 0.0)

        logger.info("Training p50 (median) price model...")
        self._model_p50 = train_lightgbm_quantile(
            X_np, values, QUANTILE_P50,
            seed=self.config.seed,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
        )

        logger.info("Training p90 (upper bound) price model...")
        self._model_p90 = train_lightgbm_quantile(
            X_np, values, QUANTILE_P90,
            seed=self.config.seed,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
        )

        self._fitted = True

        if self._model_p50 is not None and self._model_p90 is not None:
            vol_suffix = "+volatility" if self._use_volatility else ""
            wx_suffix = "+weather" if effective_weather_lookup else ""
            rank_suffix = "+rank" if self._use_rank else ""
            model_type = f"lightgbm_quantile{vol_suffix}{wx_suffix}{rank_suffix}"
        else:
            model_type = "baseline_fallback"

        if effective_weather_lookup:
            features_version = "v3.0"
        elif self._use_rank:
            features_version = "v5.0"
        elif self._use_volatility:
            features_version = "v2.0"
        else:
            features_version = "v1.2"

        self._metadata = ModelMetadata(
            model_type=model_type,
            trained_at=datetime.utcnow(),
            features_version=features_version,
            training_samples=len(prices),
            regions=self._known_regions,
            seed=self.config.seed,
        )

        logger.info(
            f"Fitted price quantile model ({model_type}, {features_version}) on "
            f"{len(prices)} samples, {len(self._known_regions)} regions"
        )

        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_prices: Optional[list[EnergyPrice]] = None,
        weather_df: Optional[Any] = None,
    ) -> list[PriceQuantileForecast]:
        """Generate quantile price forecasts.

        v1.1: Uses lag features if sufficient recent data available (≥6 hours),
        otherwise falls back silently to temporal+region features.

        v3.0: Optionally accepts weather_df for weather feature augmentation.
        If weather_df is None, falls back to the training-time weather lookup
        (if available), otherwise runs price-only (no crash).

        Args:
            region:        Region to forecast.
            timestamps:    List of future timestamps to predict.
            recent_prices: Recent price data for feature computation (≥6 hours for lags).
            weather_df:    Optional canonical weather DataFrame covering the prediction
                           timestamps (columns: timestamp, region, temperature_c, …).
                           When None, the model falls back to training-time weather data
                           if available, then to price-only mode.  Supplying eval-window
                           historical actuals is correct for backtesting (they serve as
                           a proxy for high-accuracy weather forecasts).

        Returns:
            List of PriceQuantileForecast objects
        """
        if not self._fitted:
            logger.warning("Model not fitted, using baseline fallback")
            return self._baseline_predict(timestamps, region)

        # Resolve weather lookup for the prediction window
        predict_weather_lookup: dict = {}
        if self._use_weather:
            if weather_df is not None and not (
                hasattr(weather_df, "empty") and weather_df.empty
            ):
                predict_weather_lookup = build_weather_lookup(weather_df)
            elif self._train_weather_lookup:
                # Fallback: use training-time weather lookup (covers context window)
                predict_weather_lookup = self._train_weather_lookup

        # Extract recent values for this region.
        # Use up to max_lag+24h of context to support lag_336h (v5.0) and rank features.
        # The caller (BacktestEngine) provides context_hours=336 records per region.
        max_lag = max(self._lag_hours) if self._lag_hours else 168
        context_window = max_lag + 24  # safety margin beyond longest lag
        recent_values = None
        if recent_prices:
            region_prices = [p.price_per_mwh for p in recent_prices if p.region == region]
            if len(region_prices) >= MIN_RECENT_HOURS:
                recent_values = np.array(region_prices[-context_window:])

        regions = [region] * len(timestamps)

        # Check if sufficient recent data for lag features
        use_lags = check_recent_data_sufficient(recent_values)

        if use_lags:
            X, _ = build_feature_matrix_for_predict(
                timestamps,
                regions,
                recent_values,
                known_regions=self._known_regions,
                lag_hours=self._lag_hours,
                rolling_hours=self._rolling_hours,
                include_volatility=self._use_volatility,
                include_rank_features=self._use_rank,
                weather_lookup=predict_weather_lookup,
            )
        else:
            logger.debug("Price model: using temporal+region only (insufficient recent data)")
            X = build_feature_matrix(
                timestamps,
                regions,
                values=None,
                include_lags=True,
                include_rolling=True,
                include_volatility=self._use_volatility,
                include_rank_features=self._use_rank,
                known_regions=self._known_regions,
                lag_hours=self._lag_hours,
                rolling_hours=self._rolling_hours,
                weather_lookup=predict_weather_lookup,
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
            # Subtract empirical mean error (mean_error = predicted - actual)
            if region_bias:
                hour_bias = region_bias.get(ts.hour, 0.0)
                p50_corrected = p50_raw - hour_bias
            else:
                p50_corrected = p50_raw
            # Ensure p90 >= corrected p50
            p90 = max(p90_preds[i], p50_corrected)
            fv = self._metadata.features_version if self._metadata else "v2.0"
            forecasts.append(PriceQuantileForecast(
                timestamp=ts,
                region=region,
                p50=round(p50_corrected, 2),
                p90=round(p90, 2),
                model_type=model_type,
                features_version=fv,
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to disk using joblib.

        Args:
            path: Destination file path (e.g. models/price_v1.joblib).
        """
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"PriceQuantileForecaster saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PriceQuantileForecaster":
        """Load a persisted model from disk.

        Args:
            path: Path previously written by save().

        Returns:
            Loaded PriceQuantileForecaster instance.
        """
        import joblib
        path = Path(path)
        obj = joblib.load(path)
        logger.info(f"PriceQuantileForecaster loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Holdout validation
    # ------------------------------------------------------------------

    def validate_coverage(
        self,
        holdout_data: list[EnergyPrice],
    ) -> dict:
        """Compute empirical p90 coverage on a holdout set.

        Predictions are made per-region using no lag context (to guarantee
        zero leakage from the holdout into lag features).  This gives a
        conservative lower-bound on coverage; live coverage with proper
        context will be equal or better.

        Args:
            holdout_data: EnergyPrice records withheld from training.

        Returns:
            dict with keys:
                empirical_p90_coverage (float): fraction of actuals ≤ p90
                n_samples (int)
                meets_88pct_threshold (bool): True if coverage ≥ 0.88
        """
        if not self._fitted:
            raise RuntimeError("validate_coverage: model must be fitted first")
        if not holdout_data:
            raise ValueError("validate_coverage: holdout_data is empty")

        by_region: dict[str, list[EnergyPrice]] = {}
        for p in holdout_data:
            by_region.setdefault(p.region, []).append(p)

        covered_count = 0
        total_count = 0

        for region, recs in by_region.items():
            recs_sorted = sorted(recs, key=lambda r: r.timestamp)
            timestamps = [r.timestamp for r in recs_sorted]
            actuals = [r.price_per_mwh for r in recs_sorted]
            # Predict without context — leakage-free by construction
            forecasts = self.predict(region, timestamps, recent_prices=None)
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


# ===========================================================================
# Per-Region Forecaster (v4.0)
# ===========================================================================

@dataclass
class PerRegionForecasterConfig:
    """Configuration for per-region forecaster.

    Trains one PriceQuantileForecaster per region, eliminating cross-region
    feature stealing that occurs in joint multi-region models.

    Attributes:
        base_config: Base model config applied to all regions unless
            overridden by region_configs.
        weather_regions: Regions that receive weather features during training
            and prediction. Regions NOT in this list use price-only models.
            Default: ["us-south"] — ERCOT gets weather (cold-snap/heat-wave
            detection); CAISO/PJM run price-only (no feature stealing).
        region_configs: Optional per-region overrides. Keys are region strings;
            values are PriceModelConfig instances that override base_config for
            that region only. Allows different num_leaves/n_estimators per grid.
    """
    base_config: PriceModelConfig = field(default_factory=PriceModelConfig)
    weather_regions: list = field(default_factory=lambda: ["us-south"])
    region_configs: dict = field(default_factory=dict)  # {region: PriceModelConfig}


class PerRegionForecaster:
    """Per-region LightGBM quantile forecaster.

    Trains one PriceQuantileForecaster per region instead of a joint model,
    eliminating cross-region feature stealing.

    Key benefits over joint PriceQuantileForecaster:
    - Each region gets its own model capacity (no shared features).
    - Weather features applied selectively: ERCOT gets temperature/HDD/CDD;
      CAISO/PJM use price-only (avoids joint-model degradation from
      January cold-snap features polluting non-cold-snap regions).
    - Per-region hyperparameter tuning supported via region_configs.

    Interface is identical to PriceQuantileForecaster (fit / predict /
    metadata), so it is a drop-in replacement in BacktestEngine.

    Training discipline:
    - Offline batch training only (fit() must be called before predict()).
    - Fixed random seeds for determinism.
    - Leakage-safe: fit() receives only training records.
    """

    def __init__(
        self,
        config: Optional[PerRegionForecasterConfig] = None,
        corrections_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if config is None:
            config = PerRegionForecasterConfig()
        # Accept a bare PriceModelConfig for backward compatibility
        if isinstance(config, PriceModelConfig):
            config = PerRegionForecasterConfig(base_config=config)
        self.config: PerRegionForecasterConfig = config
        self._corrections_path = corrections_path
        # One forecaster per region, populated at fit() time.
        self._region_forecasters: dict[str, PriceQuantileForecaster] = {}
        self._fitted = False
        self._known_regions: list[str] = []
        self._train_weather_lookup: dict = {}

    def _make_forecaster(self, region: str) -> PriceQuantileForecaster:
        """Build a PriceQuantileForecaster for a specific region."""
        # Use per-region config override if available, else base config
        region_cfg = self.config.region_configs.get(region, self.config.base_config)
        fc = PriceQuantileForecaster(
            config=region_cfg,
            corrections_path=self._corrections_path if self._corrections_path is not False else False,
        )
        return fc

    def fit(
        self,
        prices: list[EnergyPrice],
        weather_df: Optional[Any] = None,
    ) -> "PerRegionForecaster":
        """Fit one PriceQuantileForecaster per region.

        Args:
            prices:     List of historical EnergyPrice objects (all regions).
            weather_df: Optional canonical weather DataFrame. Weather features
                        are applied ONLY to regions listed in
                        config.weather_regions; other regions are trained
                        price-only to avoid cross-region feature pollution.

        Returns:
            Self for chaining.
        """

        # Group price records by region
        by_region: dict[str, list[EnergyPrice]] = {}
        for p in prices:
            by_region.setdefault(p.region, []).append(p)

        self._known_regions = sorted(by_region)
        self._region_forecasters = {}

        # Build weather lookup once for fast slicing
        weather_lookup: dict = {}
        if weather_df is not None and not (
            hasattr(weather_df, "empty") and weather_df.empty
        ):
            from .quantile_model import build_weather_lookup as _bwl
            weather_lookup = _bwl(weather_df)
            self._train_weather_lookup = weather_lookup

        for region, region_prices in by_region.items():
            fc = self._make_forecaster(region)
            # Supply weather_df only for designated regions
            region_weather: Optional[Any] = None
            if weather_df is not None and region in self.config.weather_regions:
                if not (hasattr(weather_df, "empty") and weather_df.empty):
                    mask = weather_df["region"] == region
                    rw = weather_df[mask]
                    region_weather = rw if not rw.empty else None

            fc.fit(region_prices, weather_df=region_weather)
            self._region_forecasters[region] = fc
            logger.info(
                f"PerRegionForecaster: fitted region={region}, "
                f"n={len(region_prices)}, weather={'on' if region_weather is not None else 'off'}"
            )

        self._fitted = True
        return self

    def predict(
        self,
        region: str,
        timestamps: list[datetime],
        recent_prices: Optional[list[EnergyPrice]] = None,
        weather_df: Optional[Any] = None,
    ) -> list[PriceQuantileForecast]:
        """Predict prices for a specific region using its dedicated model.

        Args:
            region:        Target region.
            timestamps:    Future timestamps to forecast.
            recent_prices: Recent price data for lag features.
            weather_df:    Optional weather DataFrame for regions in
                           config.weather_regions. Ignored for price-only regions.

        Returns:
            List of PriceQuantileForecast objects.
        """
        if not self._fitted:
            logger.warning("PerRegionForecaster not fitted; returning flat fallback")
            return self._flat_fallback(region, timestamps)

        fc = self._region_forecasters.get(region)
        if fc is None:
            logger.warning(
                f"PerRegionForecaster: no model for region={region}; returning flat fallback"
            )
            return self._flat_fallback(region, timestamps)

        # Apply weather only for designated regions
        effective_weather: Optional[Any] = None
        if weather_df is not None and region in self.config.weather_regions:
            if not (hasattr(weather_df, "empty") and weather_df.empty):
                effective_weather = weather_df

        return fc.predict(region, timestamps, recent_prices, weather_df=effective_weather)

    def predict_range(
        self,
        region: str,
        start_time: datetime,
        hours: int,
        recent_prices: Optional[list[EnergyPrice]] = None,
    ) -> list[PriceQuantileForecast]:
        timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
        return self.predict(region, timestamps, recent_prices)

    def _flat_fallback(
        self, region: str, timestamps: list[datetime]
    ) -> list[PriceQuantileForecast]:
        return [
            PriceQuantileForecast(
                timestamp=ts,
                region=region,
                p50=50.0,
                p90=65.0,
                model_type="per_region_fallback",
                features_version="v4.0",
            )
            for ts in timestamps
        ]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        if not self._fitted or not self._region_forecasters:
            return None
        # Return metadata from the first sub-model as a representative sample
        first = next(iter(self._region_forecasters.values()))
        return first.metadata

    def forecasts_to_dict(
        self,
        forecasts: list[PriceQuantileForecast],
    ) -> dict[str, dict[datetime, dict[str, float]]]:
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
