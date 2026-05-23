"""Tests for TemporalSplitter – verifies no-leakage guarantee."""

import pandas as pd
import pytest

from aurelius.backtesting.splitter import TemporalSplitter

# Use small windows so tests need only a few days of data
TRAIN_DAYS = 3
EVAL_DAYS = 1
STEP_DAYS = 1
# 7 days of hourly data → enough for 3-day train + 1-day eval + 3 step folds
DATA_HOURS = (TRAIN_DAYS + EVAL_DAYS * 4) * 24


def _price_df(hours=DATA_HOURS, start="2024-01-01"):
    base = pd.Timestamp(start, tz="UTC")
    timestamps = [base + pd.Timedelta(hours=h) for h in range(hours)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "region": "us-west",
        "price_per_mwh": 50.0 + (pd.Series(range(hours)) % 24),
    })


class TestTemporalSplitter:
    def test_basic_split_produces_folds(self):
        df = _price_df()
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, step_days=STEP_DAYS)
        splits = splitter.split(df)
        assert len(splits) >= 1

    def test_all_splits_no_leakage(self):
        df = _price_df()
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, step_days=STEP_DAYS)
        splits = splitter.split(df)
        for split in splits:
            train_max = split.train_df["timestamp"].max()
            eval_min = split.eval_df["timestamp"].min()
            assert train_max < eval_min, (
                f"Fold {split.fold_index}: max(train)={train_max} >= min(eval)={eval_min}"
            )

    def test_fold_indices_increment(self):
        df = _price_df()
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, step_days=STEP_DAYS)
        splits = splitter.split(df)
        indices = [s.fold_index for s in splits]
        assert indices == sorted(indices)

    def test_train_end_equals_eval_start(self):
        df = _price_df()
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS, step_days=STEP_DAYS)
        splits = splitter.split(df)
        for split in splits:
            assert split.train_end == split.eval_start

    def test_empty_df_raises(self):
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
                           "region": [], "price_per_mwh": []})
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS)
        with pytest.raises(ValueError, match="empty"):
            splitter.split(df)

    def test_missing_ts_col_raises(self):
        df = pd.DataFrame({"ts": ["2024-01-01"], "price": [50]})
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS)
        with pytest.raises(ValueError, match="missing column"):
            splitter.split(df)

    def test_insufficient_data_returns_no_splits(self):
        # 10 hours is less than even 1 day; splitter needs train_days + eval_days
        df = _price_df(hours=10)
        splitter = TemporalSplitter(train_days=TRAIN_DAYS, eval_days=EVAL_DAYS)
        splits = splitter.split(df)
        assert splits == []

    def test_invalid_train_days_raises(self):
        with pytest.raises(ValueError):
            TemporalSplitter(train_days=0, eval_days=EVAL_DAYS)

    def test_invalid_eval_days_raises(self):
        with pytest.raises(ValueError):
            TemporalSplitter(train_days=TRAIN_DAYS, eval_days=0)
