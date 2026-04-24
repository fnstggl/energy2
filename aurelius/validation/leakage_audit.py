"""Temporal data-leakage detection for backtesting.

Usage:
    from aurelius.validation.leakage_audit import DataLeakageError, assert_no_leakage

    assert_no_leakage(train_df, eval_df)  # raises DataLeakageError on overlap
"""

import pandas as pd


class DataLeakageError(Exception):
    """Raised when training data temporally overlaps evaluation data.

    Hard invariant: max(train_timestamp) < min(eval_timestamp)
    """


def assert_no_leakage(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> None:
    """Assert that no training timestamp is >= any evaluation timestamp.

    Args:
        train_df: Training split DataFrame (must contain *ts_col*).
        eval_df:  Evaluation split DataFrame (must contain *ts_col*).
        ts_col:   Name of the timestamp column.

    Raises:
        DataLeakageError: If max(train) >= min(eval).
        ValueError: If either DataFrame is missing *ts_col* or is empty.
    """
    if ts_col not in train_df.columns:
        raise ValueError(f"train_df missing column '{ts_col}'")
    if ts_col not in eval_df.columns:
        raise ValueError(f"eval_df missing column '{ts_col}'")
    if train_df.empty:
        raise ValueError("train_df is empty")
    if eval_df.empty:
        raise ValueError("eval_df is empty")

    train_max = pd.Timestamp(train_df[ts_col].max())
    eval_min = pd.Timestamp(eval_df[ts_col].min())

    if train_max >= eval_min:
        raise DataLeakageError(
            f"Data leakage detected: max(train_timestamp)={train_max} "
            f">= min(eval_timestamp)={eval_min}. "
            "Training data must end strictly before evaluation data begins."
        )
