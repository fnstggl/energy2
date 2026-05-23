"""Temporal train/eval splitter with hard leakage guard.

Hard invariant: max(train_timestamp) < min(eval_timestamp).
Any violation raises DataLeakageError immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from aurelius.validation.leakage_audit import assert_no_leakage


@dataclass
class TemporalSplit:
    """One train/eval window pair."""
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp    # exclusive upper bound of training window
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp     # exclusive upper bound of eval window
    fold_index: int


class TemporalSplitter:
    """Walk-forward temporal splitter that enforces strict leakage isolation.

    The splitter produces (train, eval) window pairs by stepping through
    historical data. The training window always ends *before* the evaluation
    window begins – validated by `assert_no_leakage` on every fold.

    Args:
        train_days: Length of the training window in days.
        eval_days:  Length of the evaluation window in days.
        step_days:  How many days to advance between folds (default = eval_days).
        ts_col:     Name of the timestamp column in the input DataFrame.
    """

    def __init__(
        self,
        train_days: int = 30,
        eval_days: int = 7,
        step_days: int = 0,
        ts_col: str = "timestamp",
    ) -> None:
        if train_days <= 0:
            raise ValueError("train_days must be > 0")
        if eval_days <= 0:
            raise ValueError("eval_days must be > 0")
        self.train_days = train_days
        self.eval_days = eval_days
        self.step_days = step_days if step_days > 0 else eval_days
        self.ts_col = ts_col

    def split(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[TemporalSplit]:
        """Generate non-leaking train/eval splits from *df*.

        Args:
            df:    DataFrame with a timestamp column.
            start: Earliest timestamp to use (defaults to min in df).
            end:   Latest timestamp to use exclusive (defaults to max+1h in df).

        Returns:
            Ordered list of TemporalSplit objects.

        Raises:
            DataLeakageError: If any produced split contains leaking data
                              (should never happen; this is a safety net).
            ValueError: If the DataFrame is empty or missing the timestamp column.
        """
        if self.ts_col not in df.columns:
            raise ValueError(f"DataFrame missing column '{self.ts_col}'")
        if df.empty:
            raise ValueError("DataFrame is empty; cannot split")

        timestamps = pd.to_datetime(df[self.ts_col])
        if start is None:
            start = timestamps.min()
        if end is None:
            end = timestamps.max() + pd.Timedelta(hours=1)

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        splits: list[TemporalSplit] = []
        fold_index = 0

        train_delta = timedelta(days=self.train_days)
        eval_delta = timedelta(days=self.eval_days)
        step_delta = timedelta(days=self.step_days)

        # Expanding or rolling window: train window starts at `start`, eval window
        # slides forward in steps.
        eval_start = start + train_delta

        while eval_start + eval_delta <= end:
            train_start = start
            train_end = eval_start           # exclusive
            eval_end = eval_start + eval_delta

            # Slice the dataframe
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            eval_mask = (timestamps >= eval_start) & (timestamps < eval_end)

            train_df = df[train_mask].copy()
            eval_df = df[eval_mask].copy()

            if train_df.empty or eval_df.empty:
                eval_start += step_delta
                fold_index += 1
                continue

            # Hard invariant check – raises DataLeakageError on any overlap
            assert_no_leakage(train_df, eval_df, ts_col=self.ts_col)

            splits.append(TemporalSplit(
                train_df=train_df,
                eval_df=eval_df,
                train_start=train_start,
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
                fold_index=fold_index,
            ))

            eval_start += step_delta
            fold_index += 1

        return splits
