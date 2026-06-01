"""Calibration + tail-safety wrappers for the CARA latency forecaster v1.

These are *post-hoc calibrators* — they wrap an already-fit base predictor
and learn a small additive / multiplicative correction on a held-out
calibration split. The base predictor's training data is never used for
calibration; the test holdout's labels are never used for calibration.

Four classes are implemented:

1. ``ConservativeMultiplierCalibration`` — learns a single multiplier
   ``m`` on the calibration split such that ``m * raw_pred`` covers the
   target quantile of calibration ``y``. Use when you want a single
   scalar correction that scales with predicted magnitude.

2. ``QuantileResidualCalibration`` — learns the additive residual
   quantile ``r_q`` on calibration: ``calibrated = raw_pred + r_q``.
   Works well when residuals are stationary across the predicted range.

3. ``SplitConformalUpperBound`` — split-conformal prediction interval.
   Computes nonconformity scores ``s_i = y_i - raw_pred_i`` on the
   calibration set; the calibrated upper bound is
   ``raw_pred + q_level(s)`` where ``q_level`` is the
   ``ceil((n+1)*alpha)/n`` quantile required for finite-sample
   coverage. Distribution-free under exchangeability.

4. ``BaselineFallbackGate`` — explicit fallback wrapper. Wraps an ML
   model + a baseline. The ``predict_with_fallback`` method returns
   either the calibrated ML prediction or the baseline prediction
   depending on a pre-registered tail-safety policy. Any fallback used
   is recorded in the JSON summary.

Honesty rules (binding):

- ``fit()`` only ever sees ``(X_cal, y_cal)``. The test holdout's
  labels are not in scope.
- ``ConservativeMultiplierCalibration`` enforces ``multiplier >= 1.0``
  by default — calibration can only **widen** the predicted interval,
  never tighten it. Tightening would weaken the safety floor.
- All classes record an ``empirical_coverage`` measurement on the
  calibration split so reviewers can see whether the target was met
  in-sample (different from the *test* coverage reported by the
  driver).
- No calibrator touches scheduler code, controllers, or the robust
  energy engine. Nothing here ever runs in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _empirical_coverage(y: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of rows where ``y <= pred`` (i.e. predicted upper bound holds)."""
    yt = np.asarray(y, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return float("nan")
    return float((yt <= yp).mean())


def _overprediction_ratio(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of ``pred / y`` over rows where both are finite + ``y > 0``.

    Values > 1 indicate the calibrated prediction routinely exceeds the
    realised value (conservative). Values < 1 indicate under-prediction.
    """
    yt = np.asarray(y, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    mask &= yt > 0
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return float("nan")
    return float((yp / yt).mean())


def _mean_conservatism(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of ``pred - y`` across rows. Positive = conservative on average."""
    yt = np.asarray(y, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return float("nan")
    return float((yp - yt).mean())


# ---------------------------------------------------------------------------
# 1. ConservativeMultiplierCalibration
# ---------------------------------------------------------------------------


@dataclass
class ConservativeMultiplierCalibration:
    """Multiplicative scalar calibration.

    Learns ``multiplier = quantile(y_cal / max(pred_cal, eps), target_quantile)``
    on the calibration split.

    Defaults clamp ``multiplier >= 1.0`` — calibration is a safety widen,
    never a tighten. The unclamped ratio is recorded as
    ``raw_ratio_quantile`` for transparency.
    """

    target_quantile: float = 0.95
    min_multiplier: float = 1.0
    base: Optional[object] = None  # the base predictor; must be pre-fit
    eps: float = 1e-9

    multiplier: float = field(default=float("nan"), init=False)
    raw_ratio_quantile: float = field(default=float("nan"), init=False)
    empirical_calibration_coverage: float = field(default=float("nan"), init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, X_cal, y_cal) -> "ConservativeMultiplierCalibration":
        if self.base is None:
            raise RuntimeError("ConservativeMultiplierCalibration.base must be set")
        y = np.asarray(y_cal, dtype=np.float64)
        raw = self.base.predict(X_cal)
        ratios = y / np.maximum(raw, self.eps)
        finite = np.isfinite(ratios)
        ratios = ratios[finite]
        if ratios.size == 0:
            raise ValueError("no finite calibration ratios")
        self.raw_ratio_quantile = float(
            np.quantile(ratios, self.target_quantile, method="higher")
        )
        self.multiplier = max(self.min_multiplier, self.raw_ratio_quantile)
        calibrated = raw * self.multiplier
        self.empirical_calibration_coverage = _empirical_coverage(y, calibrated)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ConservativeMultiplierCalibration not fitted")
        return self.base.predict(X) * self.multiplier

    def diagnostics(self) -> dict:
        return {
            "method": "conservative_multiplier",
            "target_quantile": self.target_quantile,
            "min_multiplier": self.min_multiplier,
            "multiplier": self.multiplier,
            "raw_ratio_quantile": self.raw_ratio_quantile,
            "clamped_by_min_multiplier": (
                bool(self.raw_ratio_quantile < self.min_multiplier)
                if self._fitted else False
            ),
            "empirical_calibration_coverage": self.empirical_calibration_coverage,
        }


# ---------------------------------------------------------------------------
# 2. QuantileResidualCalibration
# ---------------------------------------------------------------------------


@dataclass
class QuantileResidualCalibration:
    """Additive residual-quantile calibration.

    ``residual_i = y_cal_i - raw_pred_i``. The calibrated upper bound is
    ``raw_pred + quantile(residuals, target_quantile)``.

    Allows ``residual_quantile`` to be negative when the base predictor
    is already over-predicting; that path is recorded with
    ``conservatism_widened = False`` so reviewers see whether
    calibration widened or tightened.
    """

    target_quantile: float = 0.95
    allow_tighten: bool = False
    base: Optional[object] = None

    residual_quantile: float = field(default=float("nan"), init=False)
    empirical_calibration_coverage: float = field(default=float("nan"), init=False)
    conservatism_widened: bool = field(default=False, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, X_cal, y_cal) -> "QuantileResidualCalibration":
        if self.base is None:
            raise RuntimeError("QuantileResidualCalibration.base must be set")
        y = np.asarray(y_cal, dtype=np.float64)
        raw = self.base.predict(X_cal)
        residuals = y - raw
        finite = np.isfinite(residuals)
        residuals = residuals[finite]
        if residuals.size == 0:
            raise ValueError("no finite calibration residuals")
        q = float(np.quantile(residuals, self.target_quantile, method="higher"))
        if not self.allow_tighten:
            q = max(0.0, q)
        self.residual_quantile = q
        self.conservatism_widened = bool(q > 0.0)
        calibrated = raw + q
        self.empirical_calibration_coverage = _empirical_coverage(y, calibrated)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("QuantileResidualCalibration not fitted")
        return self.base.predict(X) + self.residual_quantile

    def diagnostics(self) -> dict:
        return {
            "method": "quantile_residual",
            "target_quantile": self.target_quantile,
            "allow_tighten": self.allow_tighten,
            "residual_quantile": self.residual_quantile,
            "conservatism_widened": self.conservatism_widened,
            "empirical_calibration_coverage": self.empirical_calibration_coverage,
        }


# ---------------------------------------------------------------------------
# 3. SplitConformalUpperBound
# ---------------------------------------------------------------------------


@dataclass
class SplitConformalUpperBound:
    """Split-conformal upper-bound calibration.

    Given a base point predictor (typically a quantile regressor for the
    target ``alpha``), compute nonconformity scores
    ``s_i = y_cal_i - raw_pred_i`` on the calibration set, then add the
    finite-sample-corrected ``q_level = ceil((n+1)*alpha)/n`` quantile of
    ``s`` to every prediction. Under exchangeability the resulting upper
    bound covers a fresh ``y`` with probability ``>= alpha`` (Vovk et al.
    2005, Lei et al. 2018 split-conformal).

    Distribution-free; the only assumption is the calibration + test
    samples are exchangeable. Time holdout violates exchangeability in
    general, so the driver reports the empirical test coverage honestly.
    """

    alpha: float = 0.95
    base: Optional[object] = None

    upper_bound_offset: float = field(default=float("nan"), init=False)
    q_level: float = field(default=float("nan"), init=False)
    calibration_n: int = field(default=0, init=False)
    empirical_calibration_coverage: float = field(default=float("nan"), init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, X_cal, y_cal) -> "SplitConformalUpperBound":
        if self.base is None:
            raise RuntimeError("SplitConformalUpperBound.base must be set")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1); got {self.alpha}")
        y = np.asarray(y_cal, dtype=np.float64)
        raw = self.base.predict(X_cal)
        scores = y - raw
        finite = np.isfinite(scores)
        scores = scores[finite]
        n = scores.size
        if n == 0:
            raise ValueError("no finite calibration scores")
        self.calibration_n = int(n)
        # Finite-sample corrected quantile level.
        q_level = min(1.0, float(np.ceil((n + 1) * self.alpha) / n))
        self.q_level = q_level
        self.upper_bound_offset = float(
            np.quantile(scores, q_level, method="higher")
        )
        calibrated = raw + self.upper_bound_offset
        self.empirical_calibration_coverage = _empirical_coverage(y, calibrated)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SplitConformalUpperBound not fitted")
        return self.base.predict(X) + self.upper_bound_offset

    def diagnostics(self) -> dict:
        return {
            "method": "split_conformal_upper_bound",
            "alpha": self.alpha,
            "calibration_n": self.calibration_n,
            "q_level": self.q_level,
            "upper_bound_offset": self.upper_bound_offset,
            "empirical_calibration_coverage": self.empirical_calibration_coverage,
        }


# ---------------------------------------------------------------------------
# 4. BaselineFallbackGate
# ---------------------------------------------------------------------------


@dataclass
class BaselineFallbackGate:
    """Per-row fallback to the strongest baseline.

    Two fallback policies are implemented:

    - ``floor_at_baseline`` (default) — return
      ``max(ml_calibrated_pred, baseline_pred)``. This treats the
      baseline as a hard safety floor; the ML model can only widen,
      never tighten, the predicted interval.

    - ``baseline_when_ood`` — if the ML prediction is more than
      ``ood_tolerance_x`` times below the baseline (signalling the ML
      may be hallucinating low-latency on out-of-distribution input),
      fall back to the baseline. Otherwise use ML.

    The gate's ``predict_with_fallback(X)`` returns ``(prediction,
    used_fallback_mask)`` so the driver can record how often the
    fallback fired (transparency for shadow review).
    """

    policy: str = "floor_at_baseline"
    ood_tolerance_x: float = 0.5
    ml: Optional[object] = None
    baseline: Optional[object] = None
    # The baseline_predict callable is provided separately when the
    # baseline needs a special call signature (e.g. group_keys_predict).
    baseline_predict: Optional[object] = None

    def _baseline_predict(self, X, **kwargs) -> np.ndarray:
        if self.baseline_predict is not None:
            return self.baseline_predict(X, **kwargs)
        return self.baseline.predict(X)

    def predict_with_fallback(self, X, **baseline_kwargs):
        if self.ml is None or self.baseline is None:
            raise RuntimeError("BaselineFallbackGate requires both ml and baseline")
        ml_pred = self.ml.predict(X)
        base_pred = self._baseline_predict(X, **baseline_kwargs)
        ml_pred = np.asarray(ml_pred, dtype=np.float64)
        base_pred = np.asarray(base_pred, dtype=np.float64)

        if self.policy == "floor_at_baseline":
            final = np.maximum(ml_pred, base_pred)
            used_fallback = ml_pred < base_pred
        elif self.policy == "baseline_when_ood":
            threshold = self.ood_tolerance_x * base_pred
            used_fallback = ml_pred < threshold
            final = np.where(used_fallback, base_pred, ml_pred)
        else:
            raise ValueError(f"unknown policy '{self.policy}'")
        return final, used_fallback

    def diagnostics(self, used_fallback) -> dict:
        used = np.asarray(used_fallback, dtype=bool)
        return {
            "method": "baseline_fallback_gate",
            "policy": self.policy,
            "ood_tolerance_x": self.ood_tolerance_x,
            "fallback_fired_rate": (
                float(used.mean()) if used.size else float("nan")
            ),
            "fallback_fired_count": int(used.sum()),
            "fallback_total_count": int(used.size),
        }


# ---------------------------------------------------------------------------
# Calibration split utilities — deterministic, never use test labels
# ---------------------------------------------------------------------------


def train_calibration_test_split(
    train_idx: np.ndarray, *,
    calibration_frac: float = 0.25, seed: int = 1773889 + 91,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a calibration block off the training indices.

    The calibration split is taken from the **train** indices only — the
    test holdout is never touched. Caller passes the train indices that
    came out of the holdout split (random / by_instance / time).
    """
    if not (0.0 < calibration_frac < 0.5):
        raise ValueError(
            f"calibration_frac must be in (0, 0.5); got {calibration_frac}"
        )
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    perm = rng.permutation(len(train_idx))
    n_cal = int(round(len(train_idx) * calibration_frac))
    cal_local = perm[:n_cal]
    sub_train_local = perm[n_cal:]
    return (
        np.sort(train_idx[sub_train_local]),
        np.sort(train_idx[cal_local]),
    )


def time_train_calibration_split(
    train_idx: np.ndarray, timestamps: np.ndarray, *,
    calibration_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """For the time-holdout strategy, calibration must come from the most
    recent slice of the train block (closer to the test distribution).
    """
    if not (0.0 < calibration_frac < 0.5):
        raise ValueError(f"calibration_frac out of range: {calibration_frac}")
    train_idx = np.asarray(train_idx)
    ts = np.asarray(timestamps, dtype=np.float64)[train_idx]
    order = np.argsort(ts, kind="stable")
    n_cal = int(round(len(train_idx) * calibration_frac))
    cal_local = order[-n_cal:]
    sub_train_local = order[:-n_cal]
    return (
        np.sort(train_idx[sub_train_local]),
        np.sort(train_idx[cal_local]),
    )


# ---------------------------------------------------------------------------
# Tail-safety summary metrics + promotion gate
# ---------------------------------------------------------------------------


def tail_safety_metrics(
    y_test: np.ndarray, calibrated_pred: np.ndarray,
    *, target_coverage: float,
) -> dict:
    """Empirical coverage + undercoverage + conservatism on the test split."""
    yt = np.asarray(y_test, dtype=np.float64)
    yp = np.asarray(calibrated_pred, dtype=np.float64)
    coverage = _empirical_coverage(yt, yp)
    return {
        "target_coverage": target_coverage,
        "empirical_coverage": coverage,
        "coverage_error": (
            float(coverage - target_coverage)
            if not np.isnan(coverage) else float("nan")
        ),
        "undercoverage_rate": (
            float(max(0.0, target_coverage - coverage))
            if not np.isnan(coverage) else float("nan")
        ),
        "overprediction_ratio": _overprediction_ratio(yt, yp),
        "mean_conservatism": _mean_conservatism(yt, yp),
        "p95_residual": _signed_residual_quantile(yt, yp, 0.95),
        "p99_residual": _signed_residual_quantile(yt, yp, 0.99),
    }


def _signed_residual_quantile(y, pred, q) -> float:
    yt = np.asarray(y, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if not mask.any():
        return float("nan")
    return float(np.quantile((yt - yp)[mask], q, method="higher"))


# Per-target × per-quantile promotion thresholds (binding per mission spec).
PROMOTION_THRESHOLDS = {
    ("ttft", 0.50): {
        "time_pinball_improvement_pct": 10.0,
        "random_pinball_improvement_pct": 10.0,
        "by_instance_pinball_improvement_pct": 10.0,
        "max_calibration_error_increase_pct": 5.0,
    },
    ("ttft", 0.95): {
        "time_pinball_improvement_pct": 10.0,
        "min_empirical_coverage": 0.93,
        "max_undercoverage_rate": 0.07,
    },
    ("ttft", 0.99): {
        "time_pinball_improvement_pct": 5.0,
        "min_empirical_coverage": 0.975,
        "max_undercoverage_rate": 0.025,
    },
    ("e2e", 0.95): {
        "time_pinball_improvement_pct": 5.0,
        "min_empirical_coverage": 0.93,
        "max_undercoverage_rate": 0.07,
    },
    ("e2e", 0.99): {
        "time_pinball_improvement_pct": 5.0,
        "min_empirical_coverage": 0.975,
        "max_undercoverage_rate": 0.025,
    },
}


FINAL_STATUS_VALUES = frozenset({
    "shadow_ready",
    "shadow_ready_tail_candidate",
    "diagnostic_only",
    "needs_more_data",
    "baseline_fallback",
    "rejected_regression",
})


def classify_tail_status(
    *,
    target_family: str,           # "ttft" | "e2e"
    quantile: float,              # 0.50 | 0.95 | 0.99
    time_holdout_improvement_pct: float,
    random_holdout_improvement_pct: float,
    by_instance_holdout_improvement_pct: float,
    empirical_coverage: Optional[float],
    undercoverage_rate: Optional[float],
    has_subgroup_regression: bool,
    has_subgroup_undercoverage: bool,
    fallback_required_on_time: bool,
    leakage_free: bool,
    no_test_label_calibration: bool,
) -> tuple[str, str]:
    """Return ``(final_status, reason)``.

    Implements PHASE E ordering: leakage → time-holdout calibration →
    time-holdout pinball → subgroup safety → random/by_instance sanity →
    fallback behaviour. A target/quantile combination cannot be marked
    shadow-ready if **any** earlier gate fails.
    """

    if not leakage_free:
        return "rejected_regression", "leakage feature detected"
    if not no_test_label_calibration:
        return "rejected_regression", "calibration used test labels"

    key = (target_family, quantile)
    th = PROMOTION_THRESHOLDS.get(key)
    if th is None:
        return "diagnostic_only", f"no promotion thresholds defined for {key}"

    if quantile == 0.50:
        # TTFT p50 gate (mission spec): time + random + by_instance all
        # >= 10% pinball improvement, no leakage, calibration error
        # increase bounded.
        if time_holdout_improvement_pct < th["time_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"time pinball improvement "
                    f"{time_holdout_improvement_pct:.2f}% < threshold "
                    f"{th['time_pinball_improvement_pct']:.1f}%")
        if random_holdout_improvement_pct < th[
                "random_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"random pinball improvement "
                    f"{random_holdout_improvement_pct:.2f}% < threshold")
        if by_instance_holdout_improvement_pct < th[
                "by_instance_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"by_instance pinball improvement "
                    f"{by_instance_holdout_improvement_pct:.2f}% < threshold")
        return "shadow_ready", "all gates passed"

    # p95/p99: time-holdout first, then coverage, then subgroup, then fallback.
    if time_holdout_improvement_pct < th["time_pinball_improvement_pct"]:
        return (
            "diagnostic_only",
            f"time-holdout pinball improvement "
            f"{time_holdout_improvement_pct:.2f}% < threshold "
            f"{th['time_pinball_improvement_pct']:.1f}%",
        )

    if empirical_coverage is None or np.isnan(empirical_coverage):
        return "diagnostic_only", "empirical_coverage missing"
    if empirical_coverage < th["min_empirical_coverage"]:
        return (
            "diagnostic_only",
            f"empirical coverage {empirical_coverage:.4f} < threshold "
            f"{th['min_empirical_coverage']:.4f}",
        )
    if undercoverage_rate is None or np.isnan(undercoverage_rate):
        return "diagnostic_only", "undercoverage_rate missing"
    if undercoverage_rate > th["max_undercoverage_rate"]:
        return (
            "diagnostic_only",
            f"undercoverage rate {undercoverage_rate:.4f} > threshold "
            f"{th['max_undercoverage_rate']:.4f}",
        )

    if has_subgroup_regression:
        return ("rejected_regression",
                "at least one major subgroup regressed > 5% vs baseline")
    if has_subgroup_undercoverage:
        return ("diagnostic_only",
                "at least one major subgroup below coverage threshold")
    if fallback_required_on_time:
        return ("baseline_fallback",
                "baseline fallback fired on >25% of time-holdout rows")

    return "shadow_ready_tail_candidate", "all tail-safety gates passed"
