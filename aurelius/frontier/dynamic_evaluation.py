"""Dynamic Serving Frontier — calibration / shadow-evaluation metrics.

This module turns paired (prediction, observed outcome) tuples into
calibration records and aggregates them into a calibration summary.

Hard rules:

- **No production-savings claims.** All metrics are derived from
  simulator / shadow-mode rollouts. The static frontier controller,
  ``constraint_aware`` defaults, and the robust energy engine are
  unchanged.
- **No future leakage.** Oracle-alpha capture uses the *post-hoc*
  realized best safe rho per window. This is **analysis-only** — the
  estimator under evaluation never sees it.
- **Honest reporting.** Negative oracle-alpha capture (the estimator did
  worse than the static baseline) is reported, never hidden. Zero
  denominators are handled explicitly.
- **Safety vetoes are categorical** (false-safe / false-unsafe /
  conservative-miss). They are never folded into the headline KPI.
- Pure stdlib. JSON round-trippable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Models — JSON round-trippable dataclasses.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicFrontierPrediction:
    """A single recommendation emitted by the dynamic estimator.

    Carries every quantity the calibration loop needs to compare against
    the realized outcome at the next decision window. Missing fields stay
    ``None`` — never zero-filled (mirrors ``ServingTelemetryTick``).
    """

    timestamp_s: float
    workload_id: str
    current_rho: Optional[float]
    recommended_rho: Optional[float]
    action: str
    predicted_goodput_per_dollar: Optional[float] = None
    predicted_goodput_delta: Optional[float] = None
    predicted_timeout_pct: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    predicted_latency_p99_ms: Optional[float] = None
    predicted_sla_risk_probability: Optional[float] = None
    predicted_queue_blowup_probability: Optional[float] = None
    predicted_gpu_hours: Optional[float] = None
    confidence: float = 0.5
    risk_reason_codes: tuple = ()
    source: str = "dynamic_frontier_estimator_v1"
    # Optional per-field provenance for the telemetry the prediction was
    # built from. JSON-serializable dict produced by
    # ``TickProvenance.to_dict()`` — kept as a plain dict so the
    # evaluation module does not depend on provenance internals.
    tick_provenance: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_reason_codes"] = list(self.risk_reason_codes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicFrontierPrediction":
        rrc = d.get("risk_reason_codes") or ()
        tp = d.get("tick_provenance")
        payload = {k: d.get(k) for k in cls.__dataclass_fields__
                   if k not in ("risk_reason_codes", "tick_provenance")}
        return cls(**payload, risk_reason_codes=tuple(rrc),
                   tick_provenance=tp)


@dataclass(frozen=True)
class DynamicFrontierObservedOutcome:
    """The observed outcome for the window following a prediction.

    ``was_safe`` is the post-hoc verdict (True iff every realized signal
    stayed within the configured safety thresholds for the window).
    """

    timestamp_s: float
    workload_id: str
    applied_rho: Optional[float]
    observed_goodput_per_dollar: Optional[float] = None
    observed_timeout_pct: Optional[float] = None
    observed_queue_p99_ms: Optional[float] = None
    observed_latency_p99_ms: Optional[float] = None
    observed_sla_violation_pct: Optional[float] = None
    observed_gpu_hours: Optional[float] = None
    observed_churn: Optional[float] = None
    was_safe: Optional[bool] = None
    source: str = "dynamic_frontier_estimator_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicFrontierObservedOutcome":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DynamicFrontierCalibrationRecord:
    """One row of the calibration ledger.

    Carries the prediction, the realized outcome, every per-metric error,
    the categorical safety verdict (false-safe / false-unsafe /
    conservative-miss), oracle-alpha values for this decision window, and
    the resulting confidence update with a reason code.
    """

    prediction: DynamicFrontierPrediction
    outcome: DynamicFrontierObservedOutcome
    prediction_error_goodput: Optional[float]
    prediction_error_timeout: Optional[float]
    prediction_error_queue_p99: Optional[float]
    risk_calibration_error: Optional[float]
    safety_correct: Optional[bool]
    false_safe: bool
    false_unsafe: bool
    conservative_miss: bool
    oracle_alpha_available: Optional[float]
    oracle_alpha_captured: Optional[float]
    oracle_alpha_capture_pct: Optional[float]
    confidence_before: float
    confidence_after: float
    confidence_update_reason: str

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction.to_dict(),
            "outcome": self.outcome.to_dict(),
            "prediction_error_goodput": self.prediction_error_goodput,
            "prediction_error_timeout": self.prediction_error_timeout,
            "prediction_error_queue_p99": self.prediction_error_queue_p99,
            "risk_calibration_error": self.risk_calibration_error,
            "safety_correct": self.safety_correct,
            "false_safe": self.false_safe,
            "false_unsafe": self.false_unsafe,
            "conservative_miss": self.conservative_miss,
            "oracle_alpha_available": self.oracle_alpha_available,
            "oracle_alpha_captured": self.oracle_alpha_captured,
            "oracle_alpha_capture_pct": self.oracle_alpha_capture_pct,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "confidence_update_reason": self.confidence_update_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicFrontierCalibrationRecord":
        return cls(
            prediction=DynamicFrontierPrediction.from_dict(d["prediction"]),
            outcome=DynamicFrontierObservedOutcome.from_dict(d["outcome"]),
            prediction_error_goodput=d.get("prediction_error_goodput"),
            prediction_error_timeout=d.get("prediction_error_timeout"),
            prediction_error_queue_p99=d.get("prediction_error_queue_p99"),
            risk_calibration_error=d.get("risk_calibration_error"),
            safety_correct=d.get("safety_correct"),
            false_safe=bool(d.get("false_safe", False)),
            false_unsafe=bool(d.get("false_unsafe", False)),
            conservative_miss=bool(d.get("conservative_miss", False)),
            oracle_alpha_available=d.get("oracle_alpha_available"),
            oracle_alpha_captured=d.get("oracle_alpha_captured"),
            oracle_alpha_capture_pct=d.get("oracle_alpha_capture_pct"),
            confidence_before=float(d.get("confidence_before", 0.5)),
            confidence_after=float(d.get("confidence_after", 0.5)),
            confidence_update_reason=str(d.get("confidence_update_reason",
                                               "unknown")),
        )


# ---------------------------------------------------------------------------
# Oracle-series — analysis-only realized best safe rho per window.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OracleSeriesPoint:
    """One window of the post-hoc oracle.

    ``best_safe_rho`` is the realized highest safe rho for the window;
    ``oracle_goodput_per_dollar`` is its realized goodput/$;
    ``baseline_goodput_per_dollar`` is the realized goodput/$ that the
    static ``constraint_aware`` baseline would have produced for the same
    window (also realized — not predicted). Both are analysis-only and
    must NOT be visible to the dynamic estimator at decision time.
    """

    timestamp_s: float
    workload_id: str
    best_safe_rho: Optional[float]
    oracle_goodput_per_dollar: Optional[float]
    baseline_goodput_per_dollar: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(num: Optional[float], denom: Optional[float],
              ) -> Optional[float]:
    if num is None or denom is None:
        return None
    if abs(denom) < 1e-12:
        return None
    return num / denom


def _oracle_capture(
    *,
    actual_goodput: Optional[float],
    oracle_goodput: Optional[float],
    baseline_goodput: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (oracle_alpha_available, oracle_alpha_captured,
    oracle_alpha_capture_pct). Honest with zero/negative denominators.

    - ``oracle_alpha_available = oracle - baseline``
    - ``oracle_alpha_captured = actual - baseline``
    - ``oracle_alpha_capture_pct = captured / available``

    When ``available <= 0`` we return ``None`` for the percentage (no
    alpha to capture). Negative capture (the estimator was worse than the
    baseline) is reported as a negative number — never clipped to 0.
    """
    if (actual_goodput is None or oracle_goodput is None
            or baseline_goodput is None):
        return None, None, None
    available = oracle_goodput - baseline_goodput
    captured = actual_goodput - baseline_goodput
    if available <= 0:
        # No alpha to capture (oracle no better than baseline). Pct is
        # None so it can be filtered out of the average.
        return available, captured, None
    pct = captured / available
    return available, captured, pct


# ---------------------------------------------------------------------------
# Public API — per-record + summary metrics.
# ---------------------------------------------------------------------------

def compute_calibration_record(
    prediction: DynamicFrontierPrediction,
    outcome: DynamicFrontierObservedOutcome,
    *,
    oracle: Optional[OracleSeriesPoint] = None,
    confidence_before: Optional[float] = None,
    confidence_after: Optional[float] = None,
    confidence_update_reason: str = "no_update",
    safety_timeout_pct: float = 10.0,
    safety_queue_p99_ms: float = 2000.0,
) -> DynamicFrontierCalibrationRecord:
    """Pair one prediction with its realized outcome.

    Categorical verdicts:

    - ``false_safe`` — prediction said SAFE (no LOWER), realized was
      unsafe.
    - ``false_unsafe`` — prediction said LOWER (or kept rho low), but
      realized at the recommended rho was actually safe.
    - ``conservative_miss`` — prediction kept rho low / lowered rho, but
      the oracle shows a higher safe rho was available.
    """
    err_goodput = _safe_div(
        (outcome.observed_goodput_per_dollar or 0.0)
        - (prediction.predicted_goodput_per_dollar or 0.0)
        if (outcome.observed_goodput_per_dollar is not None
            and prediction.predicted_goodput_per_dollar is not None) else None,
        1.0)
    err_timeout = (
        (outcome.observed_timeout_pct - prediction.predicted_timeout_pct)
        if (outcome.observed_timeout_pct is not None
            and prediction.predicted_timeout_pct is not None) else None)
    err_queue = (
        (outcome.observed_queue_p99_ms - prediction.predicted_queue_p99_ms)
        if (outcome.observed_queue_p99_ms is not None
            and prediction.predicted_queue_p99_ms is not None) else None)

    # Risk calibration error: predicted SLA-risk probability vs realized
    # 0/1 unsafe label. (Brier-style per-point term.)
    risk_err = None
    if (prediction.predicted_sla_risk_probability is not None
            and outcome.was_safe is not None):
        realized_unsafe = 0.0 if outcome.was_safe else 1.0
        risk_err = (prediction.predicted_sla_risk_probability
                    - realized_unsafe)

    # Predicted-safe label: action != LOWER and risk < 0.75 (the
    # pre-registered unsafe threshold). LOWER and INSUFFICIENT do not
    # count as "predicted safe at recommended rho".
    predicted_safe = None
    if prediction.action in ("RAISE_RHO", "KEEP_RHO"):
        if (prediction.predicted_sla_risk_probability is not None
                or prediction.predicted_queue_blowup_probability is not None):
            worst = max(
                prediction.predicted_sla_risk_probability or 0.0,
                prediction.predicted_queue_blowup_probability or 0.0)
            predicted_safe = (worst < 0.75)
        else:
            # No risk score → treat as "predicted safe" since the
            # controller emitted a non-LOWER action.
            predicted_safe = True
    elif prediction.action == "LOWER_RHO":
        predicted_safe = False
    else:
        predicted_safe = None  # INSUFFICIENT_TELEMETRY → not predicted

    realized_safe = outcome.was_safe
    safety_correct = (None if (predicted_safe is None or realized_safe is None)
                      else (predicted_safe == realized_safe))

    false_safe = bool(predicted_safe is True and realized_safe is False)
    false_unsafe = bool(predicted_safe is False and realized_safe is True)

    # Conservative-miss: predicted LOWER (or kept rho strictly below
    # current) and oracle shows a higher safe rho was available.
    conservative_miss = False
    if oracle is not None and oracle.best_safe_rho is not None:
        rec = prediction.recommended_rho
        cur = prediction.current_rho
        kept_low = ((rec is not None and rec < oracle.best_safe_rho - 1e-9)
                    or (prediction.action == "LOWER_RHO"))
        if kept_low:
            conservative_miss = True

    # Oracle-alpha bookkeeping (analysis-only).
    avail, captured, pct = _oracle_capture(
        actual_goodput=outcome.observed_goodput_per_dollar,
        oracle_goodput=(oracle.oracle_goodput_per_dollar
                        if oracle is not None else None),
        baseline_goodput=(oracle.baseline_goodput_per_dollar
                          if oracle is not None else None))

    conf_before = (confidence_before
                   if confidence_before is not None else prediction.confidence)
    conf_after = (confidence_after
                  if confidence_after is not None else conf_before)
    return DynamicFrontierCalibrationRecord(
        prediction=prediction,
        outcome=outcome,
        prediction_error_goodput=err_goodput,
        prediction_error_timeout=err_timeout,
        prediction_error_queue_p99=err_queue,
        risk_calibration_error=risk_err,
        safety_correct=safety_correct,
        false_safe=false_safe,
        false_unsafe=false_unsafe,
        conservative_miss=conservative_miss,
        oracle_alpha_available=avail,
        oracle_alpha_captured=captured,
        oracle_alpha_capture_pct=pct,
        confidence_before=float(conf_before),
        confidence_after=float(conf_after),
        confidence_update_reason=confidence_update_reason,
    )


def compute_calibration_records(
    predictions: Sequence[DynamicFrontierPrediction],
    outcomes: Sequence[DynamicFrontierObservedOutcome],
    *,
    oracle_series: Optional[Sequence[OracleSeriesPoint]] = None,
    safety_timeout_pct: float = 10.0,
    safety_queue_p99_ms: float = 2000.0,
) -> list[DynamicFrontierCalibrationRecord]:
    """Pair predictions with outcomes by index (1:1) and the oracle
    series (when provided). The caller is responsible for the alignment.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"predictions ({len(predictions)}) and outcomes "
            f"({len(outcomes)}) must be the same length")
    if oracle_series is not None and len(oracle_series) != len(predictions):
        raise ValueError(
            f"oracle_series ({len(oracle_series)}) must match predictions "
            f"({len(predictions)})")
    out: list[DynamicFrontierCalibrationRecord] = []
    for i, p in enumerate(predictions):
        oracle = oracle_series[i] if oracle_series is not None else None
        out.append(compute_calibration_record(
            p, outcomes[i], oracle=oracle,
            safety_timeout_pct=safety_timeout_pct,
            safety_queue_p99_ms=safety_queue_p99_ms))
    return out


# ---------------------------------------------------------------------------
# Summary aggregation.
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def _mae(values: Iterable[Optional[float]]) -> Optional[float]:
    vs = [abs(v) for v in values if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def _brier(records: Sequence[DynamicFrontierCalibrationRecord]
           ) -> Optional[float]:
    terms = []
    for r in records:
        if (r.prediction.predicted_sla_risk_probability is not None
                and r.outcome.was_safe is not None):
            realized = 0.0 if r.outcome.was_safe else 1.0
            p = r.prediction.predicted_sla_risk_probability
            terms.append((p - realized) ** 2)
    if not terms:
        return None
    return sum(terms) / len(terms)


def compute_frontier_calibration_summary(
    records: Sequence[DynamicFrontierCalibrationRecord],
) -> dict:
    """Aggregate a sequence of calibration records into a summary dict.

    The summary is JSON-serializable. Negative oracle-alpha capture and
    zero-denominator cases are preserved honestly.
    """
    n = len(records)
    if n == 0:
        return {
            "n_records": 0,
            "mae_goodput_per_dollar": None,
            "mae_timeout_pct": None,
            "mae_queue_p99_ms": None,
            "risk_brier_score": None,
            "safe_recommendation_count": 0,
            "unsafe_recommendation_count": 0,
            "false_safe_count": 0,
            "false_unsafe_count": 0,
            "conservative_miss_count": 0,
            "false_safe_rate": None,
            "false_unsafe_rate": None,
            "conservative_miss_rate": None,
            "safety_correct_rate": None,
            "oracle_alpha_available_total": None,
            "oracle_alpha_captured_total": None,
            "oracle_alpha_capture_pct_mean": None,
            "oracle_alpha_capture_pct_overall": None,
            "average_confidence_before": None,
            "average_confidence_after": None,
            "confidence_calibration_trend": None,
            "overestimation_rate": None,
            "underestimation_rate": None,
            "action_distribution": {},
            "average_recommended_rho": None,
            "rho_distribution": {},
        }

    actions: dict[str, int] = {}
    rho_counts: dict[str, int] = {}
    for r in records:
        a = r.prediction.action
        actions[a] = actions.get(a, 0) + 1
        rec = r.prediction.recommended_rho
        if rec is not None:
            key = f"{round(rec, 2):.2f}"
            rho_counts[key] = rho_counts.get(key, 0) + 1

    # Brier + MAEs
    mae_g = _mae(r.prediction_error_goodput for r in records)
    mae_t = _mae(r.prediction_error_timeout for r in records)
    mae_q = _mae(r.prediction_error_queue_p99 for r in records)
    brier = _brier(records)

    # Safety counts
    false_safe = sum(1 for r in records if r.false_safe)
    false_unsafe = sum(1 for r in records if r.false_unsafe)
    cons_miss = sum(1 for r in records if r.conservative_miss)
    safe_count = sum(
        1 for r in records
        if r.prediction.action in ("RAISE_RHO", "KEEP_RHO")
        and not r.false_safe)
    unsafe_count = sum(
        1 for r in records
        if r.prediction.action == "LOWER_RHO"
        or r.false_safe)
    safety_correct_vals = [r.safety_correct for r in records
                            if r.safety_correct is not None]
    safety_correct_rate = (
        sum(1 for v in safety_correct_vals if v) / len(safety_correct_vals)
        if safety_correct_vals else None)

    # Oracle-alpha
    avails = [r.oracle_alpha_available for r in records
              if r.oracle_alpha_available is not None]
    captured = [r.oracle_alpha_captured for r in records
                if r.oracle_alpha_captured is not None]
    pcts = [r.oracle_alpha_capture_pct for r in records
            if r.oracle_alpha_capture_pct is not None]
    avail_total = sum(avails) if avails else None
    captured_total = sum(captured) if captured else None
    pct_mean = (sum(pcts) / len(pcts)) if pcts else None
    overall_pct = (captured_total / avail_total
                   if (captured_total is not None and avail_total
                       and avail_total > 0) else None)

    # Confidence
    avg_conf_before = _mean(r.confidence_before for r in records)
    avg_conf_after = _mean(r.confidence_after for r in records)
    conf_trend = (avg_conf_after - avg_conf_before
                  if (avg_conf_before is not None
                      and avg_conf_after is not None) else None)

    # Over / underestimation rates — based on signed prediction errors.
    # For goodput/$: predicted > observed ⇒ overestimation; for timeout
    # and queue p99 the *risk-side* metric is the predicted minus
    # observed; predicted > observed ⇒ overestimated risk. We pool the
    # three signals with equal weight.
    over_terms = 0
    under_terms = 0
    n_terms = 0
    for r in records:
        # goodput/$: under-estimating goodput when observed > predicted.
        if r.prediction_error_goodput is not None:
            n_terms += 1
            if r.prediction_error_goodput > 0:
                under_terms += 1  # observed > predicted ⇒ under-predicted
            elif r.prediction_error_goodput < 0:
                over_terms += 1
        # timeout & queue: predicted > observed ⇒ over-estimated risk.
        for v in (r.prediction_error_timeout, r.prediction_error_queue_p99):
            if v is None:
                continue
            n_terms += 1
            # observed - predicted; positive ⇒ observed exceeded predicted
            # ⇒ risk was underestimated.
            if v > 0:
                under_terms += 1
            elif v < 0:
                over_terms += 1
    overestimation_rate = (over_terms / n_terms) if n_terms else None
    underestimation_rate = (under_terms / n_terms) if n_terms else None

    # Average recommended rho
    rec_rhos = [r.prediction.recommended_rho for r in records
                if r.prediction.recommended_rho is not None]
    avg_rec_rho = (sum(rec_rhos) / len(rec_rhos)) if rec_rhos else None

    # Telemetry-provenance roll-up — for honest shadow reports.
    # Counts each (record, field) pair classified by origin so the
    # report shows e.g. "90 % of queue_p99_ms inputs were PROXY".
    origin_counts: dict[str, int] = {}
    field_origin_counts: dict[str, dict[str, int]] = {}
    timeout_fallback_counts: dict[str, int] = {}
    records_with_provenance = 0
    for r in records:
        tp = getattr(r.prediction, "tick_provenance", None)
        if not tp:
            continue
        records_with_provenance += 1
        for entry in tp.get("entries", ()):
            origin = entry.get("origin")
            field_name = entry.get("field")
            if not origin or not field_name:
                continue
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            per_field = field_origin_counts.setdefault(field_name, {})
            per_field[origin] = per_field.get(origin, 0) + 1
            if field_name == "timeout_pct":
                fb = entry.get("fallback_level")
                if fb:
                    timeout_fallback_counts[fb] = (
                        timeout_fallback_counts.get(fb, 0) + 1)

    return {
        "n_records": n,
        "mae_goodput_per_dollar": mae_g,
        "mae_timeout_pct": mae_t,
        "mae_queue_p99_ms": mae_q,
        "risk_brier_score": brier,
        "safe_recommendation_count": safe_count,
        "unsafe_recommendation_count": unsafe_count,
        "false_safe_count": false_safe,
        "false_unsafe_count": false_unsafe,
        "conservative_miss_count": cons_miss,
        "false_safe_rate": false_safe / n,
        "false_unsafe_rate": false_unsafe / n,
        "conservative_miss_rate": cons_miss / n,
        "safety_correct_rate": safety_correct_rate,
        "oracle_alpha_available_total": avail_total,
        "oracle_alpha_captured_total": captured_total,
        "oracle_alpha_capture_pct_mean": pct_mean,
        "oracle_alpha_capture_pct_overall": overall_pct,
        "average_confidence_before": avg_conf_before,
        "average_confidence_after": avg_conf_after,
        "confidence_calibration_trend": conf_trend,
        "overestimation_rate": overestimation_rate,
        "underestimation_rate": underestimation_rate,
        "action_distribution": dict(sorted(actions.items())),
        "average_recommended_rho": avg_rec_rho,
        "rho_distribution": dict(sorted(rho_counts.items())),
        # Telemetry-provenance audit — None when no record carried a
        # provenance payload (older callers or pure simulator runs).
        "telemetry_provenance": (
            None if records_with_provenance == 0 else {
                "records_with_provenance": records_with_provenance,
                "origin_counts": dict(sorted(origin_counts.items())),
                "per_field_origin_counts": {
                    k: dict(sorted(v.items()))
                    for k, v in sorted(field_origin_counts.items())
                },
                "timeout_fallback_counts":
                    dict(sorted(timeout_fallback_counts.items())),
            }),
    }


# ---------------------------------------------------------------------------
# JSON I/O helpers.
# ---------------------------------------------------------------------------

def records_to_json(
    records: Sequence[DynamicFrontierCalibrationRecord],
) -> str:
    return json.dumps([r.to_dict() for r in records], sort_keys=True,
                       default=str)


def records_from_json(s: str) -> list[DynamicFrontierCalibrationRecord]:
    return [DynamicFrontierCalibrationRecord.from_dict(d)
            for d in json.loads(s)]
