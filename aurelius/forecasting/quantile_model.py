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
from dataclasses import dataclass
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


# v5.0 price context feature column names
PRICE_RANK_FEATURE_COLS = [
    "price_momentum_168h",
    "price_vs_lag168_abs",
]


def _compute_per_region_lag_168h(
    values: np.ndarray,
    timestamps: "np.ndarray",  # pd.DatetimeIndex or array of Timestamps
    regions: list[str],
    lag_h: int = 168,
) -> np.ndarray:
    """Compute lag_Nh per-region, avoiding cross-region timestamp contamination.

    compute_lagged_features builds a single ts→val dict for all regions combined.
    When multiple regions share the same timestamp (joint training), only one
    region's value survives in the dict per timestamp, causing contamination
    (e.g. ERCOT's cold-snap $2000 price overwriting CAISO's lag lookup).

    This function builds per-region dicts so each region's lag is computed
    against its own historical prices only.

    Args:
        values:     Price values aligned with timestamps and regions.
        timestamps: Timestamp array (pd.Timestamps).
        regions:    Region strings aligned with values.
        lag_h:      Lag in hours.

    Returns:
        Array of lag values (same length as values), per-region correct.
    """
    n = len(values)
    lag_arr = np.empty(n, dtype=float)
    regions_arr = np.array(regions)
    delta = pd.Timedelta(hours=lag_h)

    for rgn in sorted(set(regions)):
        mask = regions_arr == rgn
        idx = np.where(mask)[0]
        rgn_ts = timestamps[mask]
        rgn_vals = values[mask]
        rgn_lookup: dict = {ts: val for ts, val in zip(rgn_ts, rgn_vals)}
        for i in idx:
            lagged_ts = timestamps[i] - delta
            # Fall back to current value if 168h context not available
            lag_arr[i] = rgn_lookup.get(lagged_ts, values[i])

    return lag_arr


def compute_price_rank_features(
    values: np.ndarray,
    lag_168h_values: Optional[np.ndarray] = None,
    eps: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Compute v5.0 price context features targeting the cold-snap recovery bottleneck.

    These features encode "how does the current price compare to last week's price
    for this region?" — the core signal for detecting post-cold-snap recovery periods
    when a region transitions from very expensive to very cheap within ~7 days.

    Features are derived directly from the existing time-based lag_168h values
    (computed correctly by compute_lagged_features using timestamp lookups, not
    row-based shifts). This guarantees correctness for multi-region joint models.

    Features:
        price_momentum_168h: (price - lag_168h) / (|lag_168h| + ε), clipped [-1, 5]
            Encodes "how much has this region's price changed vs last week?"
            Value -0.95 means "95% cheaper than last week" (cold-snap recovery).
            Value +3.0 means "4× more expensive than last week" (spike onset).
            Correct at predict time: lag_168h reaches into context for k<168h,
            then degrades gracefully to 0 (neutral) for k≥168h.

        price_vs_lag168_abs: price / (lag_168h + ε), clipped [0, 10]
            Absolute ratio (complementary to momentum_168h).
            Model can learn "ratio < 0.2 means much cheaper than last week."

    Args:
        values:          Chronologically ordered price array ($/MWh)
        lag_168h_values: Pre-computed time-based 168h lag values (same length as values).
                         Must come from compute_lagged_features() for correctness.
                         If None, returns zero arrays (fallback mode).
        eps:             Denominator floor for division

    Returns:
        Dict mapping feature name → array (same length as input)
    """
    n = len(values)
    if lag_168h_values is None or len(lag_168h_values) != n:
        return {
            "price_momentum_168h": np.zeros(n),
            "price_vs_lag168_abs": np.ones(n),
        }

    s = np.array(values, dtype=float)
    lag = np.array(lag_168h_values, dtype=float)
    lag_abs = np.abs(lag)

    # Momentum: (current - lag168) / |lag168|, clipped to [-1, 5]
    momentum = ((s - lag) / (lag_abs + eps)).clip(-1.0, 5.0)
    # Replace NaN/inf from zero-lag (if context had no 168h data)
    momentum = np.where(np.isfinite(momentum), momentum, 0.0)

    # Absolute ratio: current / lag168, clipped to [0, 10]
    ratio = (s / (lag + eps)).clip(0.0, 10.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    return {
        "price_momentum_168h": momentum,
        "price_vs_lag168_abs": ratio,
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


# Weather feature columns used in build_feature_matrix when weather_df is provided.
# These must match the canonical weather CSV schema from fetch_weather_data.py.
WEATHER_FEATURE_COLS = [
    "temperature_c",       # dry-bulb temperature (°C) — primary demand driver
    "hdd_f",               # heating degree-hours (°F base 65): max(0, 65°F - T)
    "cdd_f",               # cooling degree-hours (°F base 65): max(0, T - 65°F)
    "wind_speed_ms",       # surface wind speed m/s — key for ERCOT wind generation
    "temp_rolling_24h_c",  # 24h trailing mean temp — cold-snap / heat-wave regime flag
    "temp_delta_24h_c",    # T minus T_24h_ago — rapid warming/cooling detection
]


def build_weather_lookup(
    weather_df: pd.DataFrame,
) -> dict[tuple, dict[str, float]]:
    """Build a fast (timestamp, region) → {col: value} lookup from a weather DataFrame.

    Timestamps are normalised to UTC-aware, floored to the hour, so they match
    the price-record timestamps used throughout the backtest engine.

    Args:
        weather_df: Canonical weather DataFrame with columns: timestamp, region,
                    temperature_c, hdd_f, cdd_f, wind_speed_ms, temp_rolling_24h_c,
                    temp_delta_24h_c.  Extra columns are silently ignored.

    Returns:
        Dict keyed by (floor_hour_ts_utc, region) → {col: value}.
        Empty dict if weather_df is None or empty.
    """
    if weather_df is None or weather_df.empty:
        return {}

    lookup: dict[tuple, dict[str, float]] = {}
    ts_series = pd.to_datetime(weather_df["timestamp"], utc=True).dt.floor("h")

    available_cols = [c for c in WEATHER_FEATURE_COLS if c in weather_df.columns]

    for i, row in weather_df.reset_index(drop=True).iterrows():
        ts = ts_series.iloc[i]
        region = str(row["region"])
        key = (ts, region)
        vals = {}
        for col in available_cols:
            v = row.get(col, float("nan"))
            vals[col] = float(v) if not (v != v) else 0.0  # NaN → 0
        lookup[key] = vals

    return lookup


def add_weather_features(
    df: pd.DataFrame,
    timestamps: list[datetime],
    regions: list[str],
    weather_lookup: dict[tuple, dict[str, float]],
) -> pd.DataFrame:
    """Join weather features into a feature matrix row-by-row.

    For each (timestamp, region) row, look up weather features from the prebuilt
    lookup dict.  Missing entries → 0.0 (graceful degradation when weather data
    has gaps, or when this is a holdout region with no weather file).

    Args:
        df:             Feature matrix to augment (one row per sample).
        timestamps:     Timestamps aligned with df rows.
        regions:        Regions aligned with df rows.
        weather_lookup: Prebuilt lookup from build_weather_lookup().

    Returns:
        df with weather columns appended.  Never raises; missing → 0.0.
    """
    if not weather_lookup:
        return df

    # Determine which columns are present in the lookup
    sample_vals = next(iter(weather_lookup.values()), {})
    cols = list(sample_vals.keys())
    if not cols:
        return df

    weather_rows = {col: [] for col in cols}
    ts_utc = [
        (pd.Timestamp(t).tz_convert("UTC") if t.tzinfo is not None
         else pd.Timestamp(t, tz="UTC")).floor("h")
        for t in timestamps
    ]

    for ts, region in zip(ts_utc, regions):
        entry = weather_lookup.get((ts, region), {})
        for col in cols:
            weather_rows[col].append(entry.get(col, 0.0))

    for col in cols:
        df[col] = weather_rows[col]

    return df


def build_feature_matrix(
    timestamps: list[datetime],
    regions: list[str],
    values: Optional[np.ndarray] = None,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_volatility: bool = False,
    include_rank_features: bool = False,
    known_regions: Optional[list[str]] = None,
    lag_hours: Optional[list[int]] = None,
    rolling_hours: Optional[list[int]] = None,
    weather_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Build full feature matrix for training or prediction.

    v1.2 defaults:
    - lag_hours: [1, 6, 24, 168] (full seasonal lag set)
    - rolling_hours: [6, 24]
    - include_volatility: False by default, opt-in for regime features

    v3.0 addition:
    - weather_lookup: optional prebuilt dict from build_weather_lookup(); when
      provided, weather features (temperature, HDD, CDD, wind) are appended.

    v5.0 addition:
    - include_rank_features: adds price rank/percentile features that encode
      "is the current price cheap relative to recent history?" — the core signal
      for multi-region routing decisions (rolling_mean_168h, range_position_168h,
      below_p10_168h, price_vs_mean_168h).

    Args:
        timestamps: List of timestamps
        regions: List of regions
        values: Optional array of target values (for lagged features)
        include_lags: Whether to include lagged features
        include_rolling: Whether to include rolling features
        include_volatility: Whether to include volatility regime features
        include_rank_features: Whether to include v5.0 price rank features
        known_regions: Known regions for consistent encoding
        lag_hours: Specific lag hours to use
        rolling_hours: Specific rolling windows
        weather_lookup: Prebuilt weather lookup from build_weather_lookup() (optional)

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
    if include_lags or include_rolling or include_volatility or include_rank_features:
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

            if include_rank_features:
                # Use per-region lag_168h to prevent cross-region contamination.
                # The global df["lag_168h"] shares one ts→val dict across all regions
                # in joint training, causing ERCOT's cold-snap $2000 to overwrite
                # lag lookups for CAISO/PJM. Recompute per-region instead.
                lag_168h_arr = _compute_per_region_lag_168h(
                    values, ts_array, regions, lag_h=168
                )
                rank_feats = compute_price_rank_features(values, lag_168h_values=lag_168h_arr)
                for name, arr in rank_feats.items():
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

            if include_rank_features:
                df["price_momentum_168h"] = 0.0   # neutral: no change vs last week
                df["price_vs_lag168_abs"] = 1.0   # neutral: ratio 1 (same as last week)

    # Weather features (v3.0)
    if weather_lookup:
        df = add_weather_features(df, timestamps, regions, weather_lookup)

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
    include_rank_features: bool = False,
    weather_lookup: Optional[dict] = None,
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
        include_rank_features: Whether to include v5.0 price rank features
        weather_lookup: Prebuilt weather lookup from build_weather_lookup() (optional).
            When provided, weather features are joined for each (timestamp, region).
            For the prediction horizon, entries from the weather_lookup covering those
            timestamps are used directly (historical actuals serve as proxy for weather
            forecasts at backtesting time; production use would substitute weather API
            forecast values instead).

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
            include_rank_features=False,
            known_regions=known_regions,
            weather_lookup=weather_lookup,
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
        include_rank_features=include_rank_features,
        known_regions=known_regions,
        lag_hours=lag_hours,
        rolling_hours=rolling_hours,
        weather_lookup=weather_lookup,
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
    min_child_samples: int = 20,
    reg_lambda: float = 0.0,
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
        min_child_samples: Minimum samples required in a leaf node (regularization)
        reg_lambda: L2 regularization term (0.0 = disabled)

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
            min_child_samples=min_child_samples,
        )
        if num_leaves > 0:
            lgb_kwargs["num_leaves"] = num_leaves
        if reg_lambda > 0.0:
            lgb_kwargs["reg_lambda"] = reg_lambda

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
    print("  Same seed produces identical values: PASS")

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
        import lightgbm as _lgbm  # noqa: F401
        x_train = np.random.rand(100, 5)
        y = np.random.rand(100) * 100
        model = train_lightgbm_quantile(x_train, y, quantile=0.5, seed=42, n_estimators=10)
        if model is not None:
            preds = model.predict(x_train[:5])
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
    assert used_lags is True, "Should use lags with sufficient data"
    assert "lag_1h" in df_with.columns, "Should have lag_1h with sufficient data"
    print(f"  With 8 hours of data: used_lags={used_lags}, features={len(df_with.columns)}: PASS")

    # With insufficient recent data (<6 hours)
    recent_insufficient = np.array([50.0, 52.0, 48.0])
    df_without, used_lags = build_feature_matrix_for_predict(pred_ts, pred_regions, recent_insufficient)
    assert used_lags is False, "Should NOT use lags with insufficient data"
    assert "lag_1h" not in df_without.columns, "Should NOT have lag_1h without sufficient data"
    print(f"  With 3 hours of data: used_lags={used_lags}, features={len(df_without.columns)}: PASS")

    # With None recent data
    df_none, used_lags = build_feature_matrix_for_predict(pred_ts, pred_regions, None)
    assert used_lags is False, "Should NOT use lags with None data"
    print(f"  With None data: used_lags={used_lags}, features={len(df_none.columns)}: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
