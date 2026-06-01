"""CARA queue-wait forecaster v1 — baselines + ML for derived_queue_wait_s.

Reuses the latency forecaster's ML quantile model
(``HistGradientBoostingQuantileForecaster``) and the calibration /
tail-safety framework. Adds queue-specific deterministic baselines.

No model here is wired into any controller. The target is the
*derived* queue-wait proxy (``cara_queue_features.derive_queue_wait_s``),
never a measured queue wait — CARA has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse the ML model + metric helpers from the latency forecaster.
from .cara_latency_forecaster import (  # noqa: F401
    GlobalConstantP95Baseline,
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    pinball_loss,
    quantile_metrics,
)


def _percentile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q, method="nearest"))


@dataclass
class NumWaitingBaseline:
    """Predicts queue wait proportional to the decision-time ``num_waiting``.

    ``pred = base_service_time_s * num_waiting`` (a crude Little's-law-ish
    proxy). When ``num_waiting`` is ~always 0 (CARA), this degenerates to
    predicting ~0 — which is exactly the honest baseline a naive operator
    would use. ``base_service_time_s`` is learned as the calibration-split
    median ``derived_queue_wait_s`` among rows with ``num_waiting >= 1``,
    falling back to the global median.
    """

    num_waiting_col_index: int = 0  # index into the raw feature vector
    base_service_time_s: float = field(default=float("nan"), init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, X, y, *, num_waiting_train) -> "NumWaitingBaseline":
        y = np.asarray(y, dtype=np.float64)
        nw = np.asarray(num_waiting_train, dtype=np.float64)
        mask = nw >= 1
        if mask.sum() >= 5:
            self.base_service_time_s = float(
                np.nanmedian(y[mask] / np.maximum(nw[mask], 1.0))
            )
        else:
            self.base_service_time_s = float(np.nanmedian(y))
        self._fitted = True
        return self

    def predict(self, X, *, num_waiting_predict) -> np.ndarray:
        nw = np.asarray(num_waiting_predict, dtype=np.float64)
        return self.base_service_time_s * nw


@dataclass
class QueueDepthExtrapolationBaseline:
    """``pred = per_running_service_time_s * num_running``.

    Treats each currently-running request as adding a fixed service-time
    increment to the dispatch delay. ``per_running_service_time_s`` is
    learned as the calibration-split slope of ``derived_queue_wait_s`` on
    ``num_running`` (non-negative-clamped).
    """

    per_running_service_time_s: float = field(default=float("nan"), init=False)
    intercept_s: float = field(default=0.0, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, X, y, *, num_running_train) -> "QueueDepthExtrapolationBaseline":
        y = np.asarray(y, dtype=np.float64)
        nr = np.asarray(num_running_train, dtype=np.float64)
        finite = ~(np.isnan(y) | np.isnan(nr))
        y, nr = y[finite], nr[finite]
        if nr.size >= 5 and np.ptp(nr) > 0:
            slope, intercept = np.polyfit(nr, y, 1)
            self.per_running_service_time_s = max(0.0, float(slope))
            self.intercept_s = max(0.0, float(intercept))
        else:
            self.per_running_service_time_s = 0.0
            self.intercept_s = float(np.nanmedian(y)) if y.size else 0.0
        self._fitted = True
        return self

    def predict(self, X, *, num_running_predict) -> np.ndarray:
        nr = np.asarray(num_running_predict, dtype=np.float64)
        return np.maximum(
            0.0, self.intercept_s + self.per_running_service_time_s * nr,
        )


# Queue-forecaster promotion thresholds (mission spec PHASE C).
QUEUE_PROMOTION_THRESHOLDS = {
    0.50: {
        "time_pinball_improvement_pct": 10.0,
        "random_pinball_improvement_pct": 10.0,
        "by_instance_pinball_improvement_pct": 10.0,
    },
    0.95: {
        "time_pinball_improvement_pct": 10.0,
        "min_empirical_coverage": 0.93,
        "max_undercoverage_rate": 0.07,
    },
    0.99: {
        "time_pinball_improvement_pct": 5.0,
        "min_empirical_coverage": 0.975,
        "max_undercoverage_rate": 0.025,
        "max_fallback_rate": 0.25,
    },
}


QUEUE_FINAL_STATUS_VALUES = frozenset({
    "shadow_ready",
    "shadow_ready_tail_candidate",
    "diagnostic_only",
    "baseline_fallback",
    "needs_more_data",
    "rejected_regression",
})


def classify_queue_status(
    *,
    quantile: float,
    time_improvement_pct: float,
    random_improvement_pct: float,
    by_instance_improvement_pct: float,
    empirical_coverage: Optional[float],
    undercoverage_rate: Optional[float],
    fallback_rate: Optional[float],
    has_subgroup_regression: bool,
    has_subgroup_undercoverage: bool,
    leakage_free: bool,
) -> tuple[str, str]:
    """Queue-forecaster promotion classifier (time-holdout first)."""
    if not leakage_free:
        return "rejected_regression", "leakage feature detected"

    th = QUEUE_PROMOTION_THRESHOLDS.get(quantile)
    if th is None:
        return "diagnostic_only", f"no threshold for quantile {quantile}"

    if quantile == 0.50:
        if time_improvement_pct < th["time_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"time improvement {time_improvement_pct:.2f}% < "
                    f"{th['time_pinball_improvement_pct']:.1f}%")
        if random_improvement_pct < th["random_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"random improvement {random_improvement_pct:.2f}% < threshold")
        if by_instance_improvement_pct < th["by_instance_pinball_improvement_pct"]:
            return ("diagnostic_only",
                    f"by_instance improvement "
                    f"{by_instance_improvement_pct:.2f}% < threshold")
        return "shadow_ready", "all p50 gates passed"

    # p95 / p99 tail gates.
    if time_improvement_pct < th["time_pinball_improvement_pct"]:
        return ("diagnostic_only",
                f"time improvement {time_improvement_pct:.2f}% < "
                f"{th['time_pinball_improvement_pct']:.1f}%")
    if empirical_coverage is None or np.isnan(empirical_coverage):
        return "diagnostic_only", "empirical_coverage missing"
    if empirical_coverage < th["min_empirical_coverage"]:
        return ("diagnostic_only",
                f"coverage {empirical_coverage:.4f} < "
                f"{th['min_empirical_coverage']:.4f}")
    if undercoverage_rate is None or np.isnan(undercoverage_rate):
        return "diagnostic_only", "undercoverage_rate missing"
    if undercoverage_rate > th["max_undercoverage_rate"]:
        return ("diagnostic_only",
                f"undercoverage {undercoverage_rate:.4f} > "
                f"{th['max_undercoverage_rate']:.4f}")
    if "max_fallback_rate" in th and fallback_rate is not None and \
            fallback_rate > th["max_fallback_rate"]:
        return ("baseline_fallback",
                f"fallback rate {fallback_rate:.3f} > "
                f"{th['max_fallback_rate']:.2f}")
    if has_subgroup_regression:
        return "rejected_regression", "high-volume subgroup regression"
    if has_subgroup_undercoverage:
        return "diagnostic_only", "high-volume subgroup undercoverage"
    return "shadow_ready_tail_candidate", "all tail gates passed"
