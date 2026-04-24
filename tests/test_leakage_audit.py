"""Tests for DataLeakageError and assert_no_leakage."""

import pandas as pd
import pytest

from aurelius.validation.leakage_audit import DataLeakageError, assert_no_leakage


def _df(timestamps):
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True), "value": 1.0})


class TestAssertNoLeakage:
    def test_clean_split(self):
        train = _df(["2024-01-01", "2024-01-02", "2024-01-03"])
        eval_ = _df(["2024-01-04", "2024-01-05"])
        assert_no_leakage(train, eval_)  # must not raise

    def test_overlap_raises(self):
        train = _df(["2024-01-01", "2024-01-04"])  # max = Jan 4
        eval_ = _df(["2024-01-04", "2024-01-05"])  # min = Jan 4 → overlap
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)

    def test_train_after_eval_raises(self):
        train = _df(["2024-01-10"])
        eval_ = _df(["2024-01-05"])
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)

    def test_empty_train_raises(self):
        train = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "value": []})
        eval_ = _df(["2024-01-05"])
        with pytest.raises(ValueError, match="empty"):
            assert_no_leakage(train, eval_)

    def test_empty_eval_raises(self):
        train = _df(["2024-01-01"])
        eval_ = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "value": []})
        with pytest.raises(ValueError, match="empty"):
            assert_no_leakage(train, eval_)

    def test_missing_ts_col_raises(self):
        train = pd.DataFrame({"ts": ["2024-01-01"], "value": [1]})
        eval_ = _df(["2024-01-05"])
        with pytest.raises(ValueError, match="missing column"):
            assert_no_leakage(train, eval_)

    def test_single_row_clean(self):
        train = _df(["2024-01-01T23:00:00"])
        eval_ = _df(["2024-01-02T00:00:00"])
        assert_no_leakage(train, eval_)  # must not raise

    def test_single_row_overlap(self):
        ts = "2024-06-15T12:00:00"
        train = _df([ts])
        eval_ = _df([ts])
        with pytest.raises(DataLeakageError):
            assert_no_leakage(train, eval_)
