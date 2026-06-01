"""Cache / Prefix-Reuse Forecaster v1 — baselines + ML candidates.

Predicts whether a request is likely to benefit from cache-aware
routing, residency, or migration veto. Targets:

- ``high_reuse`` (binary): ``reuse_percentage >= 50%`` per SwissAI
  bucket-reuse files. Primary classification target.
- ``reuse_percentage`` (continuous, 0-100): regression target.
- ``intra_session_reuse`` (binary): CC-traces-derived proxy.

No model in this module is wired into any controller. The forecaster is
research-class until the shadow-integration gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class GlobalReuseRateBaseline:
    """Predicts the training-set mean reuse rate for every row.

    For classification targets (high_reuse): predicts the base rate.
    For regression targets (reuse_percentage): predicts the mean.
    """
    _value: float = field(default=float("nan"), init=False)

    def fit(self, X, y) -> "GlobalReuseRateBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        mask = ~np.isnan(y_arr)
        self._value = float(y_arr[mask].mean()) if mask.any() else float("nan")
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self._value, dtype=np.float64)

    def predict_proba(self, X) -> np.ndarray:
        p = np.clip(self._value, 1e-9, 1.0 - 1e-9)
        n = np.asarray(X).shape[0]
        out = np.zeros((n, 2), dtype=np.float64)
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out


@dataclass
class PerGroupReuseRateBaseline:
    """Per-group mean reuse predictor (e.g. per-model)."""
    _per_group: dict = field(default_factory=dict, init=False)
    _fallback: float = field(default=float("nan"), init=False)

    def fit(self, X, y, *, group_keys_train) -> "PerGroupReuseRateBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        groups = np.asarray(
            [("__none__" if g is None else g) for g in group_keys_train],
            dtype=object)
        mask = ~np.isnan(y_arr)
        self._fallback = (float(y_arr[mask].mean())
                         if mask.any() else float("nan"))
        self._per_group = {}
        for g in np.unique(groups):
            m = (groups == g) & mask
            if m.sum() >= 5:
                self._per_group[g] = float(y_arr[m].mean())
        return self

    def predict(self, X, *, group_keys_predict) -> np.ndarray:
        groups = np.asarray(
            [("__none__" if g is None else g) for g in group_keys_predict],
            dtype=object)
        out = np.full(len(groups), self._fallback, dtype=np.float64)
        for i, g in enumerate(groups):
            v = self._per_group.get(g)
            if v is not None:
                out[i] = v
        return out

    def predict_proba(self, X, *, group_keys_predict) -> np.ndarray:
        p = self.predict(X, group_keys_predict=group_keys_predict)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        out = np.zeros((len(p), 2), dtype=np.float64)
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out


@dataclass
class PerSessionHistoryBaseline:
    """Within-session running mean baseline: predicts current row's
    reuse-rate as the mean of the prior rows in the same session.

    Falls back to the global mean for the first request of a session
    or for sessions unseen at train time.
    """
    _per_session: dict = field(default_factory=dict, init=False)
    _fallback: float = field(default=float("nan"), init=False)

    def fit(self, X, y, *, session_keys_train) -> "PerSessionHistoryBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        sids = np.asarray(
            [("__none__" if s is None else s) for s in session_keys_train],
            dtype=object)
        mask = ~np.isnan(y_arr)
        self._fallback = (float(y_arr[mask].mean())
                         if mask.any() else float("nan"))
        sums: dict = {}
        ns: dict = {}
        for sid in np.unique(sids):
            m = (sids == sid) & mask
            if m.any():
                sums[sid] = float(y_arr[m].sum())
                ns[sid] = int(m.sum())
        self._per_session = {sid: sums[sid] / ns[sid] for sid in sums}
        return self

    def predict(self, X, *, session_keys_predict,
                rolling_session_history: Optional[np.ndarray] = None) -> np.ndarray:
        """If ``rolling_session_history`` is supplied (per-row running mean
        from the chronological feature pipeline), use it. Otherwise fall
        back to the global per-session mean from training."""
        sids = np.asarray(session_keys_predict, dtype=object)
        out = np.full(len(sids), self._fallback, dtype=np.float64)
        if rolling_session_history is not None:
            rsh = np.asarray(rolling_session_history, dtype=np.float64)
            valid = ~np.isnan(rsh)
            out[valid] = rsh[valid]
            invalid = np.where(~valid)[0]
            for i in invalid:
                v = self._per_session.get(sids[i])
                if v is not None:
                    out[i] = v
        else:
            for i, sid in enumerate(sids):
                v = self._per_session.get(sid)
                if v is not None:
                    out[i] = v
        return out

    def predict_proba(self, X, *, session_keys_predict,
                      rolling_session_history=None) -> np.ndarray:
        p = self.predict(X, session_keys_predict=session_keys_predict,
                         rolling_session_history=rolling_session_history)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        out = np.zeros((len(p), 2), dtype=np.float64)
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out


@dataclass
class RecencyFrequencyBaseline:
    """Predicts high reuse if the request's bucket-hash has been seen
    >= ``min_seen`` times in the rolling prior window.

    Uses ``rolling_per_hash_seen_count`` as input.
    """
    min_seen: int = 1
    _train_high_rate_given_seen: float = field(default=float("nan"), init=False)
    _train_high_rate_given_unseen: float = field(default=float("nan"), init=False)

    def fit(self, X, y, *, rolling_seen_count_train) -> "RecencyFrequencyBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        seen = np.asarray(rolling_seen_count_train, dtype=np.float64) >= self.min_seen
        mask = ~np.isnan(y_arr)
        mh = mask & seen
        mu = mask & (~seen)
        self._train_high_rate_given_seen = (
            float(y_arr[mh].mean()) if mh.any() else float("nan"))
        self._train_high_rate_given_unseen = (
            float(y_arr[mu].mean()) if mu.any() else float("nan"))
        if np.isnan(self._train_high_rate_given_seen):
            self._train_high_rate_given_seen = (
                float(y_arr[mask].mean()) if mask.any() else 0.5)
        if np.isnan(self._train_high_rate_given_unseen):
            self._train_high_rate_given_unseen = (
                float(y_arr[mask].mean()) if mask.any() else 0.5)
        return self

    def predict(self, X, *, rolling_seen_count_predict) -> np.ndarray:
        seen = np.asarray(rolling_seen_count_predict, dtype=np.float64) >= self.min_seen
        out = np.where(seen, self._train_high_rate_given_seen,
                       self._train_high_rate_given_unseen)
        return out.astype(np.float64)

    def predict_proba(self, X, *, rolling_seen_count_predict) -> np.ndarray:
        p = self.predict(X, rolling_seen_count_predict=rolling_seen_count_predict)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        out = np.zeros((len(p), 2), dtype=np.float64)
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out


@dataclass
class PrefixGroupBaseline:
    """Per prefix_group mean reuse predictor (PrefixBench-style).

    Falls back to the global mean for unseen prefix groups.
    """
    _per_group: dict = field(default_factory=dict, init=False)
    _fallback: float = field(default=float("nan"), init=False)

    def fit(self, X, y, *, prefix_groups_train) -> "PrefixGroupBaseline":
        return PerGroupReuseRateBaseline.fit(
            self, X, y, group_keys_train=prefix_groups_train,
        )

    def predict(self, X, *, prefix_groups_predict) -> np.ndarray:
        return PerGroupReuseRateBaseline.predict(
            self, X, group_keys_predict=prefix_groups_predict,
        )


# ---------------------------------------------------------------------------
# ML candidates
# ---------------------------------------------------------------------------


@dataclass
class LogisticReuseClassifier:
    """Logistic regression for high_reuse binary classification."""
    max_iter: int = 1000
    C: float = 1.0
    _model: object = field(default=None, init=False)

    def fit(self, X, y) -> "LogisticReuseClassifier":
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for LogisticReuseClassifier")
        self._model = LogisticRegression(
            max_iter=self.max_iter, C=self.C, solver="lbfgs")
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)


@dataclass
class HistGradientBoostingReuseClassifier:
    """sklearn HGB classifier for high_reuse."""
    max_depth: int = 6
    max_iter: int = 200
    learning_rate: float = 0.07
    random_state: int = 19690720
    _model: object = field(default=None, init=False)

    def fit(self, X, y) -> "HistGradientBoostingReuseClassifier":
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")
        self._model = HistGradientBoostingClassifier(
            max_depth=self.max_depth, max_iter=self.max_iter,
            learning_rate=self.learning_rate, random_state=self.random_state)
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)


@dataclass
class RandomForestReuseClassifier:
    n_estimators: int = 100
    max_depth: Optional[int] = 12
    random_state: int = 19690720
    _model: object = field(default=None, init=False)

    def fit(self, X, y) -> "RandomForestReuseClassifier":
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=1)
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)


@dataclass
class HistGradientBoostingReuseRegressor:
    """HGB regressor for continuous reuse_percentage."""
    max_depth: int = 6
    max_iter: int = 200
    learning_rate: float = 0.07
    random_state: int = 19690720
    _model: object = field(default=None, init=False)

    def fit(self, X, y) -> "HistGradientBoostingReuseRegressor":
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")
        self._model = HistGradientBoostingRegressor(
            max_depth=self.max_depth, max_iter=self.max_iter,
            learning_rate=self.learning_rate, random_state=self.random_state)
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


@dataclass
class FallbackToBaselineWrapper:
    """Predict with the ML model where ``ml_predict_proba(X)`` is at
    least ``min_confidence`` away from 0.5; otherwise fall back to the
    baseline. This protects against low-signal subgroups.
    """
    ml_model: object
    baseline_proba: object  # callable(X) -> proba
    min_confidence: float = 0.6

    def predict_proba(self, X) -> np.ndarray:
        p_ml = self.ml_model.predict_proba(X)
        p_base = self.baseline_proba(X)
        conf = np.maximum(p_ml[:, 1], 1.0 - p_ml[:, 1])
        use_ml = conf >= self.min_confidence
        out = np.where(use_ml[:, None], p_ml, p_base)
        return out

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Deterministic AUROC via Mann-Whitney U.

    Returns NaN if either class is empty.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-sum
    order = np.argsort(y_score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_pos = ranks[y_true > 0.5].sum()
    n_pos = pos.size
    n_neg = neg.size
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Deterministic average precision (AUPRC)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted > 0.5)
    fp = np.cumsum(y_true_sorted <= 0.5)
    n_pos = (y_true > 0.5).sum()
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / n_pos
    # Average precision = sum over thresholds of (recall_i - recall_{i-1}) * precision_i
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[mask] - y_prob[mask]) ** 2))


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *,
                      n_bins: int = 10) -> float:
    """Expected calibration error (ECE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    n = y_true.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        err += (m.sum() / n) * abs(acc - conf)
    return float(err)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Promotion thresholds
# ---------------------------------------------------------------------------


# Per mission-spec PHASE C promotion rule (economic-improvement based):
#   <2%   -> diagnostic_only
#   2-5%  -> promising_needs_validation
#   >5%   -> shadow_ready_for_integration_review
# These are economic-improvement thresholds vs the strongest realistic
# baseline (NOT predictive AUROC). Classifier callers should compute
# the economic delta separately and call ``classify_economic_status``.

ECONOMIC_PROMOTION_THRESHOLDS = {
    "diagnostic_only_max_pct": 2.0,
    "promising_max_pct": 5.0,
}


FINAL_STATUS_VALUES = frozenset({
    "shadow_ready_for_integration_review",
    "promising_needs_validation",
    "diagnostic_only",
    "rejected_regression",
    "blocked_by_scorer_limitations",
    "needs_more_data",
})


def classify_economic_status(
    *,
    best_economic_improvement_pct: float,
    has_subgroup_regression: bool,
    has_calibration_failure: bool,
    leakage_free: bool,
    scorer_supports_cache_value: bool,
) -> tuple[str, str]:
    """Classify the forecaster's promotion status using the mission-spec
    economic-improvement thresholds.

    Returns ``(status, reason)``.
    """
    if not leakage_free:
        return ("rejected_regression", "leakage feature detected in pipeline")
    if not scorer_supports_cache_value:
        # Mission spec PHASE C: if the existing scorer cannot express
        # cache reuse / prefill savings / migration-cache-loss value,
        # mark integration as blocked_by_scorer_limitations.
        if best_economic_improvement_pct >= ECONOMIC_PROMOTION_THRESHOLDS[
                "promising_max_pct"]:
            return ("blocked_by_scorer_limitations",
                    f"shadow proxy improvement {best_economic_improvement_pct:.2f}% "
                    "exceeds the >5% bar but the residency/routing scorer "
                    "cannot express cache reuse / prefill savings yet — "
                    "future scorer-side PR required before integration")
        if best_economic_improvement_pct >= ECONOMIC_PROMOTION_THRESHOLDS[
                "diagnostic_only_max_pct"]:
            return ("blocked_by_scorer_limitations",
                    f"shadow proxy improvement {best_economic_improvement_pct:.2f}% "
                    "is in the 2-5% promising band but the scorer cannot "
                    "express cache value yet — future scorer-side PR required")
        return ("diagnostic_only",
                f"shadow proxy improvement {best_economic_improvement_pct:.2f}% "
                "< 2% and scorer does not yet express cache value")
    if best_economic_improvement_pct < ECONOMIC_PROMOTION_THRESHOLDS[
            "diagnostic_only_max_pct"]:
        return ("diagnostic_only",
                f"best economic improvement {best_economic_improvement_pct:.2f}% "
                "< 2% vs strongest realistic baseline")
    if best_economic_improvement_pct < ECONOMIC_PROMOTION_THRESHOLDS[
            "promising_max_pct"]:
        return ("promising_needs_validation",
                f"best economic improvement {best_economic_improvement_pct:.2f}% "
                "in 2-5% band; awaits broader validation before shadow promotion")
    if has_calibration_failure:
        return ("diagnostic_only",
                "calibration failure (high ECE / Brier score) prevents promotion")
    if has_subgroup_regression:
        return ("rejected_regression",
                "high-volume subgroup regression detected")
    return ("shadow_ready_for_integration_review",
            f"best economic improvement {best_economic_improvement_pct:.2f}% "
            "> 5%; safe to begin shadow-integration review")
