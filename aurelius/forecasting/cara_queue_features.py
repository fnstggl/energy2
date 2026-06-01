"""Feature engineering for the CARA queue-wait forecaster v1.

CARA carries **no literal measured queue-wait label**. The dataset's
``num_waiting`` field is ~always 0 (vLLM continuous batching absorbs
load, per the CARA README), so a "waiting queue length" target is
uninformative. This module therefore defines explicit, honestly-labelled
targets:

- ``derived_queue_wait_s`` — the dispatch/queue delay derived from
  timestamps: ``(completion_timestamp_s - prediction_timestamp_s)
  - actual_e2e_latency_s``. This is the gap between total
  scheduling-to-completion wall-clock and the client-measured serving
  latency: the time the request spent NOT being actively served. It is
  empirically non-negative on CARA train_queue_details. This is a
  **derived proxy**, NOT a measured queue wait — reports must label it
  as such.

- ``queue_pressure_score`` — a deterministic decision-time score from
  scheduler state (``num_running``, ``num_waiting``, pending tokens).
  This is **synthetic** (a hand-built score), and only retained as a
  diagnostic — it is trivially predictable from the same features, so it
  is not a primary forecast target.

The honest target hierarchy (mission spec):

1. ``measured_queue_wait_s`` — NOT AVAILABLE in CARA. Never emitted.
2. ``derived_queue_wait_s`` — the primary forecast target here.
3. ``queue_pressure_score`` — synthetic diagnostic only.

Leakage rules are identical to ``cara_latency_features``: the timestamp
+ e2e fields used to *construct* the target are never used as features.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .cara_latency_features import (
    LEAKAGE_TARGET_FIELDS as _LATENCY_LEAKAGE,
)
from .cara_latency_features import (
    PREDICT_TIME_CATEGORICAL_FEATURES,
    PREDICT_TIME_NUMERIC_FEATURES,
    FeatureSpec,
    build_feature_matrix,
    build_feature_spec,
)

# Queue forecaster reuses the same decision-time feature set as the
# latency forecaster (the scheduler state at decision time is exactly
# what predicts queue pressure).
QUEUE_PREDICT_TIME_NUMERIC_FEATURES = tuple(PREDICT_TIME_NUMERIC_FEATURES)
QUEUE_PREDICT_TIME_CATEGORICAL_FEATURES = tuple(PREDICT_TIME_CATEGORICAL_FEATURES)


# Explicit target names. ``measured_queue_wait_s`` is listed so callers
# can assert it is NEVER produced (CARA has no measured queue wait).
QUEUE_TARGET_NAMES = (
    "derived_queue_wait_s",
    "queue_pressure_score",
)
MEASURED_QUEUE_WAIT_AVAILABLE = False


# Leakage fields used to *construct* the derived target — must never be
# used as features. Superset of the latency leakage set.
QUEUE_LEAKAGE_TARGET_FIELDS = frozenset(
    set(_LATENCY_LEAKAGE) | {
        "derived_queue_wait_s", "queue_pressure_score",
        "prediction_timestamp_s",  # used in target construction; allowed
                                    # ONLY via hour_of_day derivation, never raw
    }
)


class QueueLeakageError(ValueError):
    """Raised when a queue feature pipeline would emit a leakage column."""


def derive_queue_wait_s(row: dict) -> Optional[float]:
    """Compute the derived queue-wait proxy for one row.

    ``(completion_timestamp_s - prediction_timestamp_s) - actual_e2e_latency_s``.

    Returns ``None`` when any required field is missing. The result is
    clamped at 0 (negative values, if any, are floating-point noise — on
    CARA train_queue_details the raw value is always >= 0).
    """
    comp = row.get("completion_timestamp_s")
    pred = row.get("prediction_timestamp_s")
    e2e = row.get("actual_e2e_latency_s")
    if comp is None or pred is None or e2e is None:
        return None
    try:
        v = (float(comp) - float(pred)) - float(e2e)
    except (TypeError, ValueError):
        return None
    return max(0.0, v)


def queue_pressure_score(row: dict) -> float:
    """A deterministic synthetic decision-time queue-pressure score.

    ``num_running + 4 * num_waiting + pending_prefill_tokens / 512 +
    pending_decode_tokens / 512``. This is a hand-built score, labelled
    SYNTHETIC; it is a diagnostic only, never a measured signal.
    """
    def _f(k):
        v = row.get(k)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    return (
        _f("num_running")
        + 4.0 * _f("num_waiting")
        + _f("pending_prefill_tokens") / 512.0
        + _f("pending_decode_tokens") / 512.0
    )


def extract_queue_target(rows: list, target: str) -> np.ndarray:
    """Extract a queue target array. Raises on ``measured_queue_wait_s``."""
    if target == "measured_queue_wait_s":
        raise QueueLeakageError(
            "measured_queue_wait_s does not exist in CARA; use "
            "derived_queue_wait_s (a labelled proxy) instead"
        )
    if target not in QUEUE_TARGET_NAMES:
        raise ValueError(
            f"target must be one of {QUEUE_TARGET_NAMES}, got {target!r}"
        )
    if target == "derived_queue_wait_s":
        vals = [derive_queue_wait_s(r) for r in rows]
    else:
        vals = [queue_pressure_score(r) for r in rows]
    return np.array(
        [np.nan if v is None else float(v) for v in vals],
        dtype=np.float64,
    )


def target_field_quality(target: str) -> str:
    """Return the honest field-quality label for a queue target."""
    return {
        "derived_queue_wait_s": "derived",
        "queue_pressure_score": "synthetic",
        "measured_queue_wait_s": "missing",
    }.get(target, "unknown")


def build_queue_feature_spec(rows: list) -> FeatureSpec:
    """Build the queue feature spec (predicted_only mode — no leakage)."""
    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    # Defensive leakage check: none of the numeric feature columns may be
    # a queue-leakage field.
    bad = [c for c in spec.numeric_columns
           if c in QUEUE_LEAKAGE_TARGET_FIELDS]
    if bad:
        raise QueueLeakageError(
            f"queue feature spec emitted leakage columns {bad}"
        )
    return spec


def build_queue_feature_matrix(rows: list, spec: FeatureSpec):
    """Build (X, names, group_keys) for the queue forecaster."""
    X, names, groups = build_feature_matrix(rows, spec)
    bad = [n for n in names if n in QUEUE_LEAKAGE_TARGET_FIELDS]
    if bad:
        raise QueueLeakageError(
            f"queue feature matrix emitted leakage columns {bad}"
        )
    return X, names, groups
