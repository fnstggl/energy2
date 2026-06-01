"""Model-layer tests for Economic ML Alpha v1.

Proves: deterministic baselines exist and behave; ML models compare against
the strongest deterministic baseline; the fallback wrapper tracks fallback
rate; metrics are sane; no production module is touched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting import economic_ml_forecaster as M  # noqa: E402,N812


def test_global_mean_baseline_predicts_mean():
    b = M.GlobalMeanBaseline().fit(np.zeros((4, 1)), np.array([1, 2, 3, 4.0]))
    out = b.predict(np.zeros((2, 1)))
    assert np.allclose(out, 2.5)


def test_group_median_baseline_uses_group():
    X = np.array([[0.], [0.], [1.], [1.]])
    y = np.array([1., 3., 10., 20.])
    b = M.GroupMedianBaseline(group_col_idx=0).fit(X, y)
    pred = b.predict(np.array([[0.], [1.]]))
    assert pred[0] == 2.0   # median of group 0
    assert pred[1] == 15.0  # median of group 1


def test_group_median_falls_back_to_global_for_unseen_group():
    X = np.array([[0.], [0.]])
    y = np.array([2., 4.])
    b = M.GroupMedianBaseline(group_col_idx=0).fit(X, y)
    pred = b.predict(np.array([[99.]]))
    assert pred[0] == 3.0  # global median


@pytest.mark.skipif(not M._SKLEARN, reason="sklearn unavailable")
def test_hgb_regressor_fits_and_beats_mean_on_learnable_signal():
    rng = np.random.RandomState(0)
    X = rng.rand(500, 3)
    y = 5 * X[:, 0] + 2 * X[:, 1]  # learnable
    hgb = M.HGBRegressor().fit(X, y)
    mae_hgb = np.mean(np.abs(hgb.predict(X) - y))
    base = M.GlobalMeanBaseline().fit(X, y)
    mae_base = np.mean(np.abs(base.predict(X) - y))
    assert mae_hgb < mae_base


def test_fallback_wrapper_tracks_fallback_rate():
    class _NaNML:
        def fit(self, X, y):
            return self

        def predict(self, X):
            out = np.full(X.shape[0], np.nan)
            out[0] = 1.0
            return out

    X = np.zeros((4, 1))
    y = np.array([1., 2., 3., 4.])
    fw = M.FallbackRegressor(ml=_NaNML(), baseline=M.GlobalMeanBaseline())
    fw.fit(X, y)
    pred = fw.predict(X)
    assert fw.fallback_rate == 0.75   # 3 of 4 rows fell back
    assert pred[1] == 2.5             # baseline mean fill
    assert pred[0] == 1.0             # ml value kept


def test_regression_metrics_sane():
    m = M.regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["spearman"] == pytest.approx(1.0)


def test_classification_metrics_auroc_perfect_separation():
    m = M.classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == pytest.approx(1.0)


def test_classification_metrics_single_class_returns_none():
    m = M.classification_metrics([1, 1, 1], [0.5, 0.6, 0.7])
    assert m["auroc"] is None


def test_forecaster_module_does_not_touch_production():
    src = (REPO_ROOT / "aurelius" / "forecasting"
           / "economic_ml_forecaster.py").read_text()
    for forbidden in ("optimization.scheduler", "constraint_shadow_scorer",
                      "residency.decision", "frontier.controller",
                      "executable_in_real_cluster"):
        assert forbidden not in src
