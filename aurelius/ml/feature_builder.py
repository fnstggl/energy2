"""Leakage-free feature builder for electricity price and carbon intensity forecasting.

LEAKAGE INVARIANT (enforced throughout):
    For every training row at index i with timestamp t_i, every feature value
    must be derived solely from observations at timestamps t < t_i.

    Specifically:
      - lag_Nh feature at row i  = value observed at t_i - N hours
      - rolling_mean_Wh at row i = mean of values in (t_i - W hours, t_i)  (exclusive of t_i)
      - All calendar features are derived from t_i itself (no future info)

    For prediction rows at timestamp t_pred:
      - Lag features require a `recent_context` window ending strictly before t_pred.
      - If context is insufficient for a lag (e.g., lag_168h but context only has 48h),
        the feature is filled with the context mean — never a future value.

Features produced (v2):
    Calendar (always present):
        hour_sin, hour_cos       – cyclic hour-of-day encoding
        dow_sin, dow_cos         – cyclic day-of-week encoding
        month_sin, month_cos     – cyclic month encoding
        week_of_year             – 1–52
        is_weekend               – 1 if Sat/Sun
        is_peak_hour             – 1 if 7 <= hour <= 22 (US peak window heuristic)

    Lag features (require historical values):
        lag_1h                   – value 1 hour prior
        lag_6h                   – value 6 hours prior
        lag_24h                  – same hour yesterday (strong daily cycle)
        lag_168h                 – same hour last week (strong weekly cycle)

    Rolling mean features (require historical values):
        rolling_mean_6h          – trailing 6-hour mean
        rolling_mean_24h         – trailing 24-hour mean (daily average)
        rolling_mean_168h        – trailing 7-day mean (weekly average)

Usage — training:
    from aurelius.ml.feature_builder import FeatureBuilder, assert_no_feature_leakage

    builder = FeatureBuilder()
    X, y = builder.build_training(price_records)   # list[EnergyPrice]
    assert_no_feature_leakage(X, [r.timestamp for r in price_records])

Usage — prediction:
    X_pred = builder.build_prediction(
        timestamps=eval_timestamps,
        region="us-west",
        recent_context=last_48h_prices,    # list[EnergyPrice], all BEFORE eval_timestamps
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard lag hours for electricity forecasting
DEFAULT_LAG_HOURS: list[int] = [1, 6, 24, 168]
# Standard rolling windows
DEFAULT_ROLLING_HOURS: list[int] = [6, 24, 168]

# Minimum recent-context hours needed to unlock lag features
_MIN_CONTEXT_HOURS = 6


@dataclass
class FeatureConfig:
    """Configuration for the feature builder."""
    lag_hours: list[int] = None       # type: ignore[assignment]
    rolling_hours: list[int] = None   # type: ignore[assignment]
    include_peak_hour: bool = True

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = list(DEFAULT_LAG_HOURS)
        if self.rolling_hours is None:
            self.rolling_hours = list(DEFAULT_ROLLING_HOURS)


class FeatureBuilder:
    """Builds leakage-free feature matrices for electricity time-series models.

    Must be fitted on training data before calling build_prediction().
    Fitting records the known regions and value statistics needed for
    consistent encoding between train and predict.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._known_regions: list[str] = []
        self._region_map: dict[str, int] = {}
        self._context_mean: float = 0.0   # fallback for missing lag values
        self._fitted = False

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def build_training(
        self,
        records: list,      # list[EnergyPrice] or list[CarbonIntensity]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Build feature matrix and target array from historical records.

        Records must be sorted by timestamp (ascending). The builder will
        sort them internally if not already sorted.

        LEAKAGE INVARIANT:
            All lag and rolling features at row i use only values from
            rows j < i (strictly earlier timestamps). Rolling windows
            are exclusive of the current row (right-open intervals).

        Args:
            records: Historical price or carbon records.

        Returns:
            (X, y) where X is a DataFrame and y is a 1-D float array.

        Raises:
            ValueError: If records is empty.
        """
        if not records:
            raise ValueError("records list is empty — cannot build training features")

        records_sorted = sorted(records, key=lambda r: r.timestamp)

        # Detect value attribute
        val_attr = _detect_value_attr(records_sorted[0])

        timestamps = [r.timestamp for r in records_sorted]
        regions = [r.region for r in records_sorted]
        values = np.array([getattr(r, val_attr) for r in records_sorted], dtype=float)

        # Fit region encoding
        self._known_regions = sorted(set(regions))
        self._region_map = {r: i for i, r in enumerate(self._known_regions)}
        self._context_mean = float(np.mean(values))
        self._fitted = True

        X = self._build_matrix(timestamps, regions, values, is_training=True)
        return X, values

    # ------------------------------------------------------------------
    # Prediction path
    # ------------------------------------------------------------------

    def build_prediction(
        self,
        timestamps: list[datetime],
        region: str,
        recent_context: Optional[list] = None,    # list[EnergyPrice / CarbonIntensity]
    ) -> pd.DataFrame:
        """Build prediction feature matrix.

        LEAKAGE INVARIANT:
            `recent_context` must contain only observations with timestamps
            strictly BEFORE min(timestamps). The caller is responsible for
            enforcing this constraint.

        When recent_context is insufficient (< 6 records), lag and rolling
        features are filled with `_context_mean` (the training mean) — never
        with future values.

        Args:
            timestamps: Future timestamps to predict.
            region: Target region.
            recent_context: Recent historical observations for lag features.
                            All timestamps must be < min(predict_timestamps).

        Returns:
            Feature DataFrame with the same columns as build_training().
        """
        if not timestamps:
            raise ValueError("timestamps list is empty")

        if not self._fitted:
            logger.warning("FeatureBuilder not fitted; using empty region encoding")
            self._known_regions = [region]
            self._region_map = {region: 0}
            self._context_mean = 50.0

        # Build combined timeline: recent_context + predict timestamps
        # This allows proper lag computation for early predict rows
        context_values: Optional[np.ndarray] = None
        context_timestamps: list[datetime] = []

        if recent_context and len(recent_context) >= _MIN_CONTEXT_HOURS:
            val_attr = _detect_value_attr(recent_context[0])
            ctx_sorted = sorted(recent_context, key=lambda r: r.timestamp)

            # --- LEAKAGE GUARD ---
            # Ensure all context timestamps are strictly before prediction window
            min_pred_ts = _to_utc(min(timestamps))
            ctx_after_pred = [r for r in ctx_sorted if _to_utc(r.timestamp) >= min_pred_ts]
            if ctx_after_pred:
                logger.warning(
                    f"FeatureBuilder: {len(ctx_after_pred)} context records have timestamps "
                    f">= min prediction timestamp {min_pred_ts.isoformat()} — "
                    "dropping them to prevent leakage"
                )
                ctx_sorted = [r for r in ctx_sorted if _to_utc(r.timestamp) < min_pred_ts]

            if len(ctx_sorted) >= _MIN_CONTEXT_HOURS:
                context_values = np.array([getattr(r, val_attr) for r in ctx_sorted], dtype=float)
                context_timestamps = [r.timestamp for r in ctx_sorted]
            else:
                logger.debug("FeatureBuilder: insufficient clean context after leakage filter")

        regions = [region] * len(timestamps)

        if context_values is not None and len(context_values) > 0:
            # Build matrix over combined timeline, then slice to predict portion
            all_timestamps = context_timestamps + list(timestamps)
            all_regions = [region] * len(context_timestamps) + regions
            all_values_placeholder = np.concatenate([
                context_values,
                np.full(len(timestamps), self._context_mean)
            ])
            full_X = self._build_matrix(
                all_timestamps, all_regions, all_values_placeholder, is_training=True
            )
            n_ctx = len(context_timestamps)
            return full_X.iloc[n_ctx:].reset_index(drop=True)
        else:
            # No context: fill lag/rolling features with context_mean
            return self._build_matrix(timestamps, regions, None, is_training=False)

    # ------------------------------------------------------------------
    # Internal matrix builder
    # ------------------------------------------------------------------

    def _build_matrix(
        self,
        timestamps: list[datetime],
        regions: list[str],
        values: Optional[np.ndarray],
        is_training: bool,
    ) -> pd.DataFrame:
        df = _calendar_features(timestamps)
        df["region_enc"] = np.array(
            [self._region_map.get(r, -1) for r in regions], dtype=float
        )

        if values is not None and len(values) == len(timestamps):
            ts_series = pd.to_datetime(timestamps)

            for lag in self.config.lag_hours:
                df[f"lag_{lag}h"] = _compute_lag(values, ts_series, lag, self._context_mean)

            for window in self.config.rolling_hours:
                df[f"roll_{window}h"] = _compute_rolling_mean(
                    values, window, self._context_mean
                )
        else:
            # No values available — fill with fallback (never future data)
            fill = self._context_mean if self._fitted else 0.0
            for lag in self.config.lag_hours:
                df[f"lag_{lag}h"] = fill
            for window in self.config.rolling_hours:
                df[f"roll_{window}h"] = fill

        return df.fillna(self._context_mean if self._fitted else 0.0)

    @property
    def feature_names(self) -> list[str]:
        """Return the expected feature column names after fitting."""
        cal = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "week_of_year",
            "is_weekend", "is_peak_hour", "region_enc",
        ]
        lags = [f"lag_{h}h" for h in self.config.lag_hours]
        rolls = [f"roll_{w}h" for w in self.config.rolling_hours]
        return cal + lags + rolls


# ------------------------------------------------------------------
# Leakage audit
# ------------------------------------------------------------------

def assert_no_feature_leakage(
    X: pd.DataFrame,
    timestamps: list[datetime],
    values: np.ndarray,
    lag_hours: Optional[list[int]] = None,
) -> None:
    """Assert that lag features in X are consistent with strict past-only lookups.

    For each lag_Nh column and each row i, verifies that:
        X["lag_Nh"].iloc[i] == values[j]  where j is the row with timestamp t_i - N hours

    If the lag timestamp is not in the data, the fallback (context mean) is
    acceptable and the check is skipped for that row.

    Args:
        X: Feature DataFrame from FeatureBuilder.build_training().
        timestamps: Corresponding timestamps (same order as X rows).
        values: Target values array (same order as X rows).
        lag_hours: Which lags to check. Defaults to DEFAULT_LAG_HOURS.

    Raises:
        AssertionError: If any lag feature contains a future value.
    """
    if lag_hours is None:
        lag_hours = DEFAULT_LAG_HOURS

    ts_to_val: dict[datetime, float] = {}
    for ts, v in zip(timestamps, values):
        ts_norm = _to_utc(ts).replace(minute=0, second=0, microsecond=0)
        ts_to_val[ts_norm] = float(v)

    for lag in lag_hours:
        col = f"lag_{lag}h"
        if col not in X.columns:
            continue

        for i, (ts, feat_val) in enumerate(zip(timestamps, X[col].values)):
            ts_norm = _to_utc(ts).replace(minute=0, second=0, microsecond=0)
            lag_ts = ts_norm - timedelta(hours=lag)

            # The feature value must never equal the CURRENT or FUTURE value
            # It must equal the past value at lag_ts (if that timestamp exists)
            if lag_ts in ts_to_val:
                expected = ts_to_val[lag_ts]
                actual = float(feat_val)
                assert abs(actual - expected) < 1e-6 or math.isnan(actual), (
                    f"LEAKAGE in {col} at row {i} (ts={ts_norm.isoformat()}): "
                    f"feature value {actual:.4f} != expected past value {expected:.4f}"
                )

            # Critical: the feature value must NOT equal the value at ts_norm (current)
            current_val = ts_to_val.get(ts_norm)
            if current_val is not None and lag > 0:
                feat = float(feat_val)
                # If feat == current_val AND it doesn't match the lag_ts value,
                # that's suspicious (might be a leakage from filling current value)
                past_val = ts_to_val.get(lag_ts)
                if past_val is not None and abs(feat - current_val) < 1e-6 and abs(feat - past_val) > 1.0:
                    raise AssertionError(
                        f"POTENTIAL LEAKAGE in {col} at row {i}: "
                        f"feature={feat:.4f} equals current value {current_val:.4f} "
                        f"but expected past value {past_val:.4f}"
                    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _detect_value_attr(record) -> str:
    if hasattr(record, "price_per_mwh"):
        return "price_per_mwh"
    elif hasattr(record, "gco2_per_kwh"):
        return "gco2_per_kwh"
    raise ValueError(
        f"Record has neither .price_per_mwh nor .gco2_per_kwh: {type(record)}"
    )


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _calendar_features(timestamps: list[datetime]) -> pd.DataFrame:
    """Calendar features with cyclic encoding. Never uses future data."""
    rows = []
    for ts in timestamps:
        h = ts.hour
        dow = ts.weekday()
        month = ts.month
        woy = ts.isocalendar()[1]
        rows.append({
            "hour_sin":    math.sin(2 * math.pi * h / 24),
            "hour_cos":    math.cos(2 * math.pi * h / 24),
            "dow_sin":     math.sin(2 * math.pi * dow / 7),
            "dow_cos":     math.cos(2 * math.pi * dow / 7),
            "month_sin":   math.sin(2 * math.pi * (month - 1) / 12),
            "month_cos":   math.cos(2 * math.pi * (month - 1) / 12),
            "week_of_year": float(woy),
            "is_weekend":   1.0 if dow >= 5 else 0.0,
            "is_peak_hour": 1.0 if 7 <= h <= 22 else 0.0,
        })
    return pd.DataFrame(rows)


def _compute_lag(
    values: np.ndarray,
    ts_series: pd.DatetimeIndex,
    lag: int,
    fallback: float,
) -> np.ndarray:
    """Compute lag_Nh feature. Falls back to `fallback` when past value unavailable.

    INVARIANT: output[i] uses only values[j] where ts_series[j] < ts_series[i].
    """
    # Build lookup: normalised ts → value
    ts_lookup: dict = {}
    for i, (ts, v) in enumerate(zip(ts_series, values)):
        norm = pd.Timestamp(ts).replace(minute=0, second=0, microsecond=0, nanosecond=0)
        ts_lookup[norm] = float(v)

    result = np.full(len(values), fallback, dtype=float)
    for i, ts in enumerate(ts_series):
        norm = pd.Timestamp(ts).replace(minute=0, second=0, microsecond=0, nanosecond=0)
        lag_ts = norm - pd.Timedelta(hours=lag)
        if lag_ts in ts_lookup:
            result[i] = ts_lookup[lag_ts]
        # else: keep fallback

    return result


def _compute_rolling_mean(
    values: np.ndarray,
    window: int,
    fallback: float,
) -> np.ndarray:
    """Trailing rolling mean. Uses min_periods=1 so output is never NaN.

    INVARIANT: output[i] = mean(values[max(0,i-window) : i])  (exclusive of values[i]).
    """
    n = len(values)
    result = np.full(n, fallback, dtype=float)
    for i in range(n):
        start = max(0, i - window)
        if i > 0:
            # Exclusive of values[i] — look only at strictly past values
            result[i] = float(np.mean(values[start:i]))
        else:
            result[i] = fallback
    return result
