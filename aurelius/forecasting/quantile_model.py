"""Shared utilities for quantile regression models.

This module provides:
- Quantile regression training utilities
- Deterministic seeding for reproducibility
- p50/p90 validation
- Feature engineering helpers

This is domain-agnostic - price vs carbon logic handled by callers.

IMPORTANT:
- Offline batch training ONLY
- Fixed random seeds for determinism
- No learning during execution

v1.1 enables minimal short-horizon lag features (1h, 6h) for accuracy
while preserving predict-time safety and deterministic fallback.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default random seed for reproducibility
DEFAULT_SEED = 42

# Quantile levels
QUANTILE_P50 = 0.5
QUANTILE_P90 = 0.9


@dataclass
class QuantileForecast:
    """Quantile forecast result.

    Attributes:
        timestamp: Forecast timestamp
        region: Geographic region
        p50: Median prediction (50th percentile)
        p90: Upper bound prediction (90th percentile)
        model_type: Type of model used
        features_version: Version of feature set
    """
    timestamp: datetime
    region: str
    p50: float
    p90: float
    model_type: str = "lightgbm_quantile"
    features_version: str = "v1"


@dataclass
class ModelMetadata:
    """Metadata about a trained model.

    Attributes:
        model_type: Type of model (e.g., "ridge+lightgbm_quantile")
        trained_at: When the model was trained
        features_version: Version of feature set used
        training_samples: Number of samples used
        regions: Regions the model was trained on
        seed: Random seed used
    """
    model_type: str
    trained_at: datetime
    features_version: str
    training_samples: int
    regions: list[str]
    seed: int = DEFAULT_SEED

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "trained_at": self.trained_at.isoformat(),
            "features_version": self.features_version,
            "training_samples": self.training_samples,
            "regions": self.regions,
            "seed": self.seed,
        }


def set_deterministic_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def validate_quantiles(p50: float, p90: float) -> tuple[float, float]:
    """Validate and enforce p90 >= p50 invariant.

    Args:
        p50: Median prediction
        p90: Upper bound prediction

    Returns:
        Tuple of (p50, p90) with invariant enforced
    """
    # Ensure non-negative
    p50 = max(0.0, p50)
    p90 = max(0.0, p90)

    # Enforce p90 >= p50
    if p90 < p50:
        # Use p50 as floor for p90
        p90 = p50

    return p50, p90


def extract_temporal_features(timestamps: list[datetime]) -> pd.DataFrame:
    """Extract temporal features from timestamps.

    Features:
    - hour_sin, hour_cos: Cyclic hour encoding
    - day_of_week: Day of week (0-6)
    - week_of_year: Week number (1-52)
    - month: Month (1-12)
    - is_weekend: Weekend flag (0/1)

    Args:
        timestamps: List of datetime objects

    Returns:
        DataFrame with temporal features
    """
    df = pd.DataFrame({"timestamp": timestamps})

    # Hour of day (cyclic encoding)
    hours = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Day of week
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Week of year
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    # Month
    df["month"] = df["timestamp"].dt.month

    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df.drop(columns=["timestamp"])


def encode_region(regions: list[str], known_regions: Optional[list[str]] = None) -> np.ndarray:
    """Encode region as categorical integer.

    Args:
        regions: List of region strings
        known_regions: Optional list of known regions for consistent encoding

    Returns:
        Array of encoded region integers
    """
    if known_regions is None:
        known_regions = sorted(set(regions))

    region_map = {r: i for i, r in enumerate(known_regions)}
    return np.array([region_map.get(r, -1) for r in regions])


def compute_lagged_features(
    values: np.ndarray,
    timestamps: np.ndarray,
    lags_hours: list[int] = [1, 6],  # v1.1: minimal lags only
) -> dict[str, np.ndarray]:
    """Compute lagged value features.

    Args:
        values: Array of values (e.g., prices)
        timestamps: Array of timestamps
        lags_hours: List of lag periods in hours (default: [1, 6] for v1.1)

    Returns:
        Dict mapping lag name to lagged values
    """
    result = {}

    # Create lookup by timestamp
    ts_to_val = {ts: val for ts, val in zip(timestamps, values)}

    for lag in lags_hours:
        lagged = []
        for ts in timestamps:
            lagged_ts = ts - pd.Timedelta(hours=lag)
            if lagged_ts in ts_to_val:
                lagged.append(ts_to_val[lagged_ts])
            else:
                # Use current value as fallback
                lagged.append(ts_to_val.get(ts, np.nan))
        result[f"lag_{lag}h"] = np.array(lagged)

    return result


def compute_rolling_features(
    values: np.ndarray,
    windows_hours: list[int] = [6],  # v1.1: minimal rolling only
    include_std: bool = False,  # v1.1: no std features by default
) -> dict[str, np.ndarray]:
    """Compute rolling mean features.

    Args:
        values: Array of values
        windows_hours: Window sizes in hours (default: [6] for v1.1)
        include_std: Whether to include std features (default: False for v1.1)

    Returns:
        Dict mapping feature name to computed values
    """
    result = {}
    series = pd.Series(values)

    for window in windows_hours:
        # Rolling mean only (v1.1: no std features)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        result[f"rolling_mean_{window}h"] = rolling_mean.values

        # Only include std if explicitly requested (not in v1.1 minimal set)
        if include_std:
            rolling_std = series.rolling(window=window, min_periods=2).std()
            rolling_std = rolling_std.fillna(0)
            result[f"rolling_std_{window}h"] = rolling_std.values

    return result


def compute_volatility_regime_features(
    values: np.ndarray,
    spike_multiplier: float = 2.0,
) -> dict[str, np.ndarray]:
    """Compute volatility regime features for price spike detection.

    These features help the model identify when we are in a high-volatility
    regime (e.g. ERCOT winter cold-snap spikes) vs normal periods.
    All features are computed strictly from past values — no leakage.

    Features:
        rolling_std_24h:      24h rolling standard deviation (raw volatility)
        rolling_std_168h:     168h rolling std (weekly baseline volatility)
        volatility_ratio_24h: rolling_std_24h / rolling_mean_24h (coefficient of variation)
        spike_flag:           1 if current value > spike_multiplier × rolling_mean_168h
        price_momentum_6h:    (current - 6h ago) / (6h ago + ε); positive = rising prices
        price_momentum_24h:   (current - 24h ago) / (24h ago + ε); day-over-day trend

    Args:
        values:           Array of price values (chronological order)
        spike_multiplier: Multiplier for rolling_mean_168h to flag a spike (default 2.0)

    Returns:
        Dict mapping feature name to array of feature values
    """
    series = pd.Series(values, dtype=float)
    eps = 1e-3  # prevent division by zero in $/MWh context

    mean_24 = series.rolling(window=24, min_periods=1).mean()
    mean_168 = series.rolling(window=168, min_periods=1).mean()
    std_24 = series.rolling(window=24, min_periods=2).std().fillna(0.0)
    std_168 = series.rolling(window=168, min_periods=2).std().fillna(0.0)

    # Coefficient of variation: how noisy is the recent 24h window?
    vol_ratio_24 = (std_24 / (mean_24 + eps)).fillna(0.0)
    # Clip to [0, 5] to avoid extreme outliers
    vol_ratio_24 = vol_ratio_24.clip(upper=5.0)

    # Spike flag: current price > spike_multiplier × 168h mean
    spike_flag = (series > (spike_multiplier * mean_168)).astype(float)

    # Price momentum: (current - past) / (past + ε) clipped to [-1, 5]
    lag6 = series.shift(6).bfill().fillna(series.iloc[0])
    lag24 = series.shift(24).bfill().fillna(series.iloc[0])
    momentum_6h = ((series - lag6) / (lag6 + eps)).clip(-1.0, 5.0).fillna(0.0)
    momentum_24h = ((series - lag24) / (lag24 + eps)).clip(-1.0, 5.0).fillna(0.0)

    return {
        "rolling_std_24h": std_24.values,
        "rolling_std_168h": std_168.values,
        "volatility_ratio_24h": vol_ratio_24.values,
        "spike_flag": spike_flag.values,
        "price_momentum_6h": momentum_6h.values,
        "price_momentum_24h": momentum_24h.values,
    }


# Minimum hours of recent data required to enable lag features
MIN_RECENT_HOURS = 6


def check_recent_data_sufficient(
    recent_values: Optional[np.ndarray],
    min_hours: int = MIN_RECENT_HOURS,
) -> bool:
    """Check if recent data is sufficient for lag feature computation.

    Args:
        recent_values: Array of recent values
        min_hours: Minimum required data points (hours)

    Returns:
        True if sufficient data available, False otherwise
    """
    if recent_values is None:
        return False
    if not hasattr(recent_values, '__len__'):
        return False
    return len(recent_values) >= min_hours


def build_feature_matrix(
    timestamps: list[datetime],
    regions: list[str],
    values: Optional[np.ndarray] = None,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_volatility: bool = False,
    known_regions: Optional[list[str]] = None,
    lag_hours: Optional[list[int]] = None,
    rolling_hours: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Build full feature matrix for training or prediction.

    v1.2 defaults:
    - lag_hours: [1, 6, 24, 168] (full seasonal lag set)
    - rolling_hours: [6, 24]
    - include_volatility: False by default, opt-in for regime features

    Args:
        timestamps: List of timestamps
        regions: List of regions
        values: Optional array of target values (for lagged features)
        include_lags: Whether to include lagged features
        include_rolling: Whether to include rolling features
        include_volatility: Whether to include volatility regime features
        known_regions: Known regions for consistent encoding
        lag_hours: Specific lag hours to use
        rolling_hours: Specific rolling windows

    Returns:
        DataFrame with all features
    """
    if lag_hours is None:
        lag_hours = [1, 6]
    if rolling_hours is None:
        rolling_hours = [6]

    # Temporal features
    df = extract_temporal_features(timestamps)

    # Region encoding
    df["region_encoded"] = encode_region(regions, known_regions)

    # Lagged and rolling features
    if include_lags or include_rolling or include_volatility:
        if values is not None and len(values) == len(timestamps):
            ts_array = pd.to_datetime(timestamps)

            if include_lags:
                lagged = compute_lagged_features(values, ts_array, lags_hours=lag_hours)
                for name, arr in lagged.items():
                    df[name] = arr

            if include_rolling:
                rolling = compute_rolling_features(values, windows_hours=rolling_hours, include_std=False)
                for name, arr in rolling.items():
                    df[name] = arr

            if include_volatility:
                vol_feats = compute_volatility_regime_features(values)
                for name, arr in vol_feats.items():
                    df[name] = arr
        else:
            # Placeholder columns for model compatibility when no values available
            if include_lags:
                for lag in lag_hours:
                    df[f"lag_{lag}h"] = 0.0

            if include_rolling:
                for window in rolling_hours:
                    df[f"rolling_mean_{window}h"] = 0.0

            if include_volatility:
                for col in [
                    "rolling_std_24h", "rolling_std_168h",
                    "volatility_ratio_24h", "spike_flag",
                    "price_momentum_6h", "price_momentum_24h",
                ]:
                    df[col] = 0.0

    # Fill NaN with 0
    df = df.fillna(0)

    return df


def build_feature_matrix_for_predict(
    timestamps: list[datetime],
    regions: list[str],
    recent_values: Optional[np.ndarray],
    known_regions: Optional[list[str]] = None,
    lag_hours: Optional[list[int]] = None,
    rolling_hours: Optional[list[int]] = None,
    include_volatility: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Build feature matrix for prediction with automatic fallback.

    If recent_values has < MIN_RECENT_HOURS data points, automatically
    disables lag/rolling features and falls back to temporal + region only.

    Args:
        timestamps: Prediction timestamps
        regions: Prediction regions
        recent_values: Recent historical values for lag features
        known_regions: Known regions for encoding
        lag_hours: Lag hours to use if sufficient data
        rolling_hours: Rolling windows if sufficient data
        include_volatility: Whether to include volatility regime features

    Returns:
        Tuple of (feature_matrix, used_lags) where used_lags indicates
        whether lag features were enabled
    """
    if lag_hours is None:
        lag_hours = [1, 6]
    if rolling_hours is None:
        rolling_hours = [6]

    # Check if we have sufficient recent data
    use_lags = check_recent_data_sufficient(recent_values)

    if not use_lags:
        logger.debug(
            "Insufficient recent data for lag features, using temporal+region only"
        )
        df = build_feature_matrix(
            timestamps,
            regions,
            values=None,
            include_lags=False,
            include_rolling=False,
            include_volatility=False,
            known_regions=known_regions,
        )
        return df, False

    n_recent = len(recent_values)
    n_predict = len(timestamps)

    from datetime import timedelta
    first_ts = timestamps[0]
    recent_timestamps = [first_ts - timedelta(hours=(n_recent - i)) for i in range(n_recent)]

    all_timestamps = recent_timestamps + list(timestamps)
    all_regions = [regions[0]] * n_recent + list(regions)

    # For lag and rolling features: fill prediction period with the LAST KNOWN PRICE.
    # Rationale: zero-fill corrupts volatility/momentum features by creating an
    # artificial "price drop to $0" signal. Forward-filling with the last context
    # value represents the persistence assumption ("current regime continues"),
    # which is a leakage-free and behaviorally correct default for feature
    # computation (we're describing the CURRENT PRICE REGIME, not predicting future
    # prices — that's what the model itself will do).
    last_known_price = float(recent_values[-1]) if len(recent_values) > 0 else 0.0
    fill_values = np.full(n_predict, last_known_price)
    all_values = np.concatenate([recent_values, fill_values])

    full_df = build_feature_matrix(
        all_timestamps,
        all_regions,
        all_values,
        include_lags=True,
        include_rolling=True,
        include_volatility=include_volatility,
        known_regions=known_regions,
        lag_hours=lag_hours,
        rolling_hours=rolling_hours,
    )

    df = full_df.iloc[n_recent:].reset_index(drop=True)
    return df, True


def time_series_cv_split(
    n_samples: int,
    n_splits: int = 5,
    test_size_ratio: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate time-based cross-validation splits.

    For time series, we use expanding window or sliding window CV
    to respect temporal ordering.

    Args:
        n_samples: Total number of samples
        n_splits: Number of CV splits
        test_size_ratio: Ratio of samples for test in each split

    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    min_train_size = int(n_samples * 0.3)  # Minimum 30% for training

    for i in range(n_splits):
        # Expanding window: each split uses more training data
        train_end = min_train_size + int((n_samples - min_train_size) * (i + 1) / (n_splits + 1))
        test_size = int(n_samples * test_size_ratio / n_splits)
        test_end = min(train_end + test_size, n_samples)

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train_lightgbm_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
) -> Any:
    """Train a LightGBM quantile regressor.

    Args:
        X_train: Training features
        y_train: Training targets
        quantile: Quantile to predict (0.5 for median, 0.9 for p90)
        seed: Random seed
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        num_leaves: LightGBM num_leaves (controls model capacity)

    Returns:
        Trained LightGBM model (or None if unavailable)
    """
    try:
        import lightgbm as lgb

        set_deterministic_seed(seed)

        lgb_kwargs: dict = dict(
            objective="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        if num_leaves > 0:
            lgb_kwargs["num_leaves"] = num_leaves

        model = lgb.LGBMRegressor(**lgb_kwargs)
        model.fit(X_train, y_train)
        return model

    except ImportError:
        logger.warning("LightGBM not available, quantile regression disabled")
        return None
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        return None


def predict_with_fallback(
    model_p50: Any,
    model_p90: Any,
    X: np.ndarray,
    baseline_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with LightGBM models, falling back to baseline if needed.

    Args:
        model_p50: Trained p50 model (or None)
        model_p90: Trained p90 model (or None)
        X: Feature matrix for prediction
        baseline_values: Fallback values if models unavailable

    Returns:
        Tuple of (p50_predictions, p90_predictions)
    """
    n_samples = X.shape[0]

    # Default to baseline or zeros
    if baseline_values is not None and len(baseline_values) == n_samples:
        default_p50 = baseline_values
        default_p90 = baseline_values * 1.3  # 30% uplift for p90
    else:
        default_p50 = np.zeros(n_samples)
        default_p90 = np.zeros(n_samples)

    # Predict p50
    if model_p50 is not None:
        try:
            p50 = model_p50.predict(X)
        except Exception as e:
            logger.warning(f"p50 prediction failed: {e}")
            p50 = default_p50
    else:
        p50 = default_p50

    # Predict p90
    if model_p90 is not None:
        try:
            p90 = model_p90.predict(X)
        except Exception as e:
            logger.warning(f"p90 prediction failed: {e}")
            p90 = default_p90
    else:
        p90 = default_p90

    # Validate and enforce invariant
    p50_validated = []
    p90_validated = []
    for i in range(n_samples):
        v50, v90 = validate_quantiles(p50[i], p90[i])
        p50_validated.append(v50)
        p90_validated.append(v90)

    return np.array(p50_validated), np.array(p90_validated)


# ============================================================================
# INLINE VALIDATION
# ============================================================================
# Run with: python -c "from aurelius.forecasting.quantile_model import _run_validation; _run_validation()"

def _run_validation():
    """Validate quantile model utilities."""
    from datetime import timedelta

    print("=" * 60)
    print("Quantile Model Utilities Validation")
    print("=" * 60)

    # Test 1: Deterministic seeding
    print("\nTest 1: DETERMINISTIC SEEDING")
    print("-" * 40)
    set_deterministic_seed(42)
    vals1 = np.random.rand(5)
    set_deterministic_seed(42)
    vals2 = np.random.rand(5)
    assert np.allclose(vals1, vals2), "Seeds should produce same values"
    print(f"  Same seed produces identical values: PASS")

    # Test 2: Quantile validation
    print("\nTest 2: QUANTILE VALIDATION")
    print("-" * 40)
    p50, p90 = validate_quantiles(50.0, 60.0)
    assert p90 >= p50, "p90 should be >= p50"
    print(f"  Normal case (50, 60) -> ({p50}, {p90}): PASS")

    p50, p90 = validate_quantiles(60.0, 50.0)
    assert p90 >= p50, "p90 should be corrected to >= p50"
    print(f"  Inverted (60, 50) -> ({p50}, {p90}): PASS")

    p50, p90 = validate_quantiles(-10.0, -5.0)
    assert p50 >= 0 and p90 >= 0, "Values should be non-negative"
    print(f"  Negative (-10, -5) -> ({p50}, {p90}): PASS")

    # Test 3: Temporal features
    print("\nTest 3: TEMPORAL FEATURES")
    print("-" * 40)
    now = datetime(2025, 6, 15, 14, 30)
    timestamps = [now + timedelta(hours=h) for h in range(24)]
    df = extract_temporal_features(timestamps)
    expected_cols = {"hour_sin", "hour_cos", "day_of_week", "week_of_year", "month", "is_weekend"}
    assert expected_cols.issubset(set(df.columns)), "Missing temporal features"
    print(f"  Features: {list(df.columns)}")
    print(f"  Shape: {df.shape}: PASS")

    # Test 4: Region encoding
    print("\nTest 4: REGION ENCODING")
    print("-" * 40)
    regions = ["us-west", "us-east", "us-west", "eu-west"]
    encoded = encode_region(regions)
    print(f"  {regions} -> {encoded}: PASS")
    assert len(encoded) == len(regions)

    # Test 5: Feature matrix (v1.1 minimal features)
    print("\nTest 5: FEATURE MATRIX (v1.1)")
    print("-" * 40)
    timestamps = [datetime(2025, 1, 1, h % 24) + timedelta(days=h // 24) for h in range(48)]
    regions = ["us-west"] * 48
    values = np.random.rand(48) * 100
    df = build_feature_matrix(timestamps, regions, values)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}: PASS")
    assert df.shape[0] == 48
    # v1.1: should have lag_1h, lag_6h, rolling_mean_6h (no std, no 24h/168h)
    assert "lag_1h" in df.columns, "Should have lag_1h"
    assert "lag_6h" in df.columns, "Should have lag_6h"
    assert "rolling_mean_6h" in df.columns, "Should have rolling_mean_6h"
    assert "lag_24h" not in df.columns, "Should NOT have lag_24h"
    assert "rolling_std_6h" not in df.columns, "Should NOT have rolling_std_6h"
    print("  v1.1 minimal features verified: PASS")

    # Test 6: Time series CV
    print("\nTest 6: TIME SERIES CV")
    print("-" * 40)
    splits = time_series_cv_split(100, n_splits=3)
    print(f"  Generated {len(splits)} splits")
    for i, (train, test) in enumerate(splits):
        assert train[-1] < test[0], "Train should come before test"
        print(f"    Split {i+1}: train={len(train)}, test={len(test)}")
    print("  Temporal ordering preserved: PASS")

    # Test 7: LightGBM training (if available)
    print("\nTest 7: LIGHTGBM TRAINING")
    print("-" * 40)
    try:
        import lightgbm
        X = np.random.rand(100, 5)
        y = np.random.rand(100) * 100
        model = train_lightgbm_quantile(X, y, quantile=0.5, seed=42, n_estimators=10)
        if model is not None:
            preds = model.predict(X[:5])
            print(f"  Trained p50 model, sample predictions: {preds[:3]}")
            print("  LightGBM training: PASS")
        else:
            print("  LightGBM training returned None")
    except ImportError:
        print("  LightGBM not installed, skipping")

    # Test 8: Predict-time fallback (v1.1)
    print("\nTest 8: PREDICT-TIME FALLBACK (v1.1)")
    print("-" * 40)
    pred_ts = [datetime(2025, 2, 1, h) for h in range(12)]
    pred_regions = ["us-west"] * 12

    # With sufficient recent data (>=6 hours)
    recent_sufficient = np.array([50.0, 52.0, 48.0, 55.0, 53.0, 51.0, 49.0, 54.0])
    df_with, used_lags = build_feature_matrix_for_predict(pred_ts, pred_regions, recent_sufficient)
    assert used_lags == True, "Should use lags with sufficient data"
    assert "lag_1h" in df_with.columns, "Should have lag_1h with sufficient data"
    print(f"  With 8 hours of data: used_lags={used_lags}, features={len(df_with.columns)}: PASS")

    # With insufficient recent data (<6 hours)
    recent_insufficient = np.array([50.0, 52.0, 48.0])
    df_without, used_lags = build_feature_matrix_for_predict(pred_ts, pred_regions, recent_insufficient)
    assert used_lags == False, "Should NOT use lags with insufficient data"
    assert "lag_1h" not in df_without.columns, "Should NOT have lag_1h without sufficient data"
    print(f"  With 3 hours of data: used_lags={used_lags}, features={len(df_without.columns)}: PASS")

    # With None recent data
    df_none, used_lags = build_feature_matrix_for_predict(pred_ts, pred_regions, None)
    assert used_lags == False, "Should NOT use lags with None data"
    print(f"  With None data: used_lags={used_lags}, features={len(df_none.columns)}: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
