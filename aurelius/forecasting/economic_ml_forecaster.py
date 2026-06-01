"""Economic ML Alpha v1 — modular forecasters + deterministic baselines.

Trains SEPARATE small models per target (never one giant black box). Every
model group is benchmarked against the strongest *deterministic* baseline for
that target; ML is only credited when it beats the baseline on a binding
holdout. All models expose a uniform ``fit`` / ``predict`` interface and a
``fallback`` to the baseline when inputs are insufficient.

No production module is imported or modified. No invented constants. The
deterministic economic-overlay formulas remain the ground truth for cost
targets (see ``docs/ECONOMIC_OVERLAY_LAYER_V1.md``); ML over a derived cost
target is diagnostic-only and labelled as such by the driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # sklearn is optional at import time; the driver checks availability.
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Deterministic regression baselines.
# ---------------------------------------------------------------------------


class GlobalMeanBaseline:
    """Predict the training mean (the honest no-information baseline)."""

    def __init__(self):
        self._mu = 0.0

    def fit(self, X, y):
        self._mu = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mu, dtype=float)


class GlobalMedianBaseline:
    def __init__(self):
        self._med = 0.0

    def fit(self, X, y):
        self._med = float(np.median(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._med, dtype=float)


class GroupMedianBaseline:
    """Predict the median of the target within a group column (e.g. the
    (model, gpu) cell). This is the strongest realistic non-ML baseline for
    latency/cost: an operator who knows per-cell historical medians."""

    def __init__(self, group_col_idx: int):
        self.group_col_idx = group_col_idx
        self._by_group: dict = {}
        self._global = 0.0

    def fit(self, X, y):
        self._global = float(np.median(y)) if len(y) else 0.0
        groups = X[:, self.group_col_idx]
        for g in np.unique(groups):
            self._by_group[g] = float(np.median(y[groups == g]))
        return self

    def predict(self, X):
        groups = X[:, self.group_col_idx]
        return np.array([self._by_group.get(g, self._global) for g in groups],
                        dtype=float)


# ---------------------------------------------------------------------------
# ML regressors (thin sklearn wrappers with NaN-safe inputs).
# ---------------------------------------------------------------------------


def _nan_to_med(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if m.any():
            fill = np.nanmedian(col) if not np.all(m) else 0.0
            col[m] = 0.0 if np.isnan(fill) else fill
            X[:, j] = col
    return X


class HGBRegressor:
    def __init__(self, **kw):
        self.kw = dict(max_iter=200, max_depth=6, learning_rate=0.08,
                       random_state=0)
        self.kw.update(kw)
        self._m = None

    def fit(self, X, y):
        self._m = HistGradientBoostingRegressor(**self.kw)
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)


class RFRegressor:
    def __init__(self, **kw):
        self.kw = dict(n_estimators=120, max_depth=14, random_state=0,
                       n_jobs=-1)
        self.kw.update(kw)
        self._m = None

    def fit(self, X, y):
        self._m = RandomForestRegressor(**self.kw)
        self._m.fit(_nan_to_med(X), y)
        return self

    def predict(self, X):
        return self._m.predict(_nan_to_med(X))


class LinearReg:
    def __init__(self):
        self._m = None

    def fit(self, X, y):
        self._m = LinearRegression()
        self._m.fit(_nan_to_med(X), y)
        return self

    def predict(self, X):
        return self._m.predict(_nan_to_med(X))


# ---------------------------------------------------------------------------
# Classification (cache high-reuse / risk targets).
# ---------------------------------------------------------------------------


class GlobalRateBaseline:
    """Predict the training positive-rate as a constant probability."""

    def __init__(self):
        self._p = 0.0

    def fit(self, X, y):
        self._p = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X):
        return np.full(X.shape[0], self._p, dtype=float)


class HGBClassifierProba:
    def __init__(self, **kw):
        self.kw = dict(max_iter=200, max_depth=6, learning_rate=0.08,
                       random_state=0)
        self.kw.update(kw)
        self._m = None

    def fit(self, X, y):
        self._m = HistGradientBoostingClassifier(**self.kw)
        self._m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(X)[:, 1]


class LogisticProba:
    def __init__(self):
        self._m = None
        self._const = None

    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            self._const = float(np.mean(y)) if len(y) else 0.0
            return self
        self._m = LogisticRegression(max_iter=400)
        self._m.fit(_nan_to_med(X), y)
        return self

    def predict_proba(self, X):
        if self._m is None:
            return np.full(X.shape[0], self._const or 0.0, dtype=float)
        return self._m.predict_proba(_nan_to_med(X))[:, 1]


# ---------------------------------------------------------------------------
# Fallback wrapper.
# ---------------------------------------------------------------------------


@dataclass
class FallbackRegressor:
    """Wraps an ML model + a deterministic baseline. ``predict`` uses the ML
    model but falls back to the baseline for rows the ML cannot score (here:
    any all-NaN feature row). Tracks the fallback rate."""
    ml: object
    baseline: object
    fallback_rate: float = 0.0

    def fit(self, X, y):
        self.baseline.fit(X, y)
        self.ml.fit(X, y)
        return self

    def predict(self, X):
        pred = np.asarray(self.ml.predict(X), dtype=float)
        base = np.asarray(self.baseline.predict(X), dtype=float)
        bad = ~np.isfinite(pred)
        self.fallback_rate = float(np.mean(bad)) if len(pred) else 0.0
        pred[bad] = base[bad]
        return pred


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sp = _spearman(y_true, y_pred)
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "spearman": sp}


def classification_metrics(y_true, p) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"n": int(len(y_true)), "auroc": None, "brier": None,
                "ece": None}
    brier = float(np.mean((p - y_true) ** 2))
    return {"n": int(len(y_true)), "auroc": _auroc(y_true, p),
            "brier": brier, "ece": _ece(y_true, p)}


def _spearman(a, b) -> Optional[float]:
    if len(a) < 3:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return None
    return float((ra * rb).sum() / denom)


def _rankdata(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _auroc(y, p) -> float:
    order = np.argsort(p)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(p)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                               else p <= edges[i + 1])
        if m.any():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(ece)
