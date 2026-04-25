"""Forecasting feature builder — pandas DataFrame interface.

Exposes build_features(), a functional entry point that wraps the
FeatureBuilder class in aurelius.ml.feature_builder and returns a
leakage-free feature matrix from a time-series DataFrame.

LEAKAGE INVARIANT (enforced throughout):
    For every training row at index i with timestamp t_i, every feature
    value is derived solely from observations at timestamps t < t_i.

    Lag features at row i  = value at t_i − N hours (prior row only).
    Rolling mean at row i  = mean of (t_i − W hours, t_i) — exclusive of t_i.
    Calendar features       = derived from t_i alone (zero future info).
    Weather features        = joined on timestamp; missing values → 0.0.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from aurelius.ml.feature_builder import (
    FeatureBuilder,
    FeatureConfig,
    assert_no_feature_leakage,
)

logger = logging.getLogger(__name__)

_VALUE_COLS = ("price_per_mwh", "gco2_per_kwh", "value")
_TS_COL = "timestamp"
_REGION_COL = "region"


def _resolve_value_col(df: pd.DataFrame) -> str:
    for col in _VALUE_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        f"build_features: DataFrame must contain one of {_VALUE_COLS}. "
        f"Got columns: {list(df.columns)}"
    )


class _MockRecord:
    """Minimal record shim so FeatureBuilder can ingest raw DataFrame rows."""
    __slots__ = ("timestamp", "region", "price_per_mwh")

    def __init__(self, ts, region: str, val: float) -> None:
        self.timestamp = ts
        self.region = region
        self.price_per_mwh = float(val)


def build_features(
    df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    lag_hours: Optional[list[int]] = None,
    rolling_hours: Optional[list[int]] = None,
    validate_leakage: bool = True,
) -> pd.DataFrame:
    """Build a leakage-free feature matrix from a time-series DataFrame.

    Args:
        df: DataFrame containing at minimum:
              - ``timestamp``: datetime-like column (UTC recommended)
              - ``region``: string region identifier
              - one value column: ``price_per_mwh``, ``gco2_per_kwh``, or ``value``
            Rows need not be sorted; the function sorts internally.
        weather_df: Optional weather DataFrame with columns:
              - ``timestamp``: must be joinable to df timestamps
              - ``solar_cf``: solar capacity factor [0, 1] (optional)
              - ``wind_cf``:  wind  capacity factor [0, 1] (optional)
            Missing values after the join are filled with 0.0.
        lag_hours: Lag hours to include (default: [1, 6, 24, 168]).
        rolling_hours: Rolling mean windows (default: [6, 24, 168]).
        validate_leakage: If True, run assert_no_feature_leakage after
            building the matrix (raises on any detected violation).

    Returns:
        pd.DataFrame with one row per input row, columns:
            Calendar:  hour_sin, hour_cos, dow_sin, dow_cos, month_sin,
                       month_cos, week_of_year, is_weekend, is_peak_hour,
                       region_enc
            Lags:      lag_1h, lag_6h, lag_24h, lag_168h  (or configured)
            Rolling:   roll_6h, roll_24h, roll_168h        (or configured)
            Weather:   solar_cf, wind_cf                   (if weather_df supplied)

    Raises:
        ValueError: If df is empty or missing required columns.
        aurelius.validation.leakage_audit.DataLeakageError: If a future
            value is detected and validate_leakage=True.
    """
    if df is None or len(df) == 0:
        raise ValueError("build_features: input DataFrame is empty")

    missing = {_TS_COL, _REGION_COL} - set(df.columns)
    if missing:
        raise ValueError(f"build_features: missing required columns: {missing}")

    val_col = _resolve_value_col(df)

    df_sorted = df.sort_values(_TS_COL).reset_index(drop=True)

    config = FeatureConfig(
        lag_hours=lag_hours or [1, 6, 24, 168],
        rolling_hours=rolling_hours or [6, 24, 168],
    )
    builder = FeatureBuilder(config=config)

    records = [
        _MockRecord(row[_TS_COL], row[_REGION_COL], row[val_col])
        for _, row in df_sorted.iterrows()
    ]
    X, y = builder.build_training(records)

    # Attach weather features if provided
    if weather_df is not None and len(weather_df) > 0:
        weather_feature_cols = [c for c in ("solar_cf", "wind_cf") if c in weather_df.columns]
        if weather_feature_cols:
            w = weather_df[[_TS_COL] + weather_feature_cols].copy()
            w[_TS_COL] = pd.to_datetime(w[_TS_COL], utc=True).dt.tz_localize(None)
            ts_key = pd.to_datetime(df_sorted[_TS_COL]).dt.tz_localize(None)
            merged = pd.merge(
                pd.DataFrame({_TS_COL: ts_key}),
                w.rename(columns={_TS_COL: _TS_COL}),
                on=_TS_COL,
                how="left",
            )
            for col in weather_feature_cols:
                X[col] = merged[col].fillna(0.0).values
        else:
            logger.debug("build_features: weather_df has no solar_cf or wind_cf columns; skipping")

    if validate_leakage:
        timestamps = list(df_sorted[_TS_COL])
        assert_no_feature_leakage(X, timestamps, y)

    return X
