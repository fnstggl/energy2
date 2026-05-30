"""Dynamic-frontier shadow logging + prediction-vs-observed comparison.

JSONL-based shadow log (one decision per line) and a compact
:class:`DynamicFrontierOutcome` model that records what was actually
observed after the recommendation was emitted. The dynamic estimator
is opt-in and runs in shadow mode by default — these logs are the
substrate for any future pilot calibration.

No production mutation. The log writer / reader are pure stdlib.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from .dynamic_models import DynamicFrontierDecision


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class DynamicFrontierShadowLog:
    """One shadow-log entry for a dynamic frontier decision.

    ``executed`` is ``False`` by default — even when the decision is
    routed through the existing ``execute_frontier_decision`` shim, the
    log records the recommendation before any simulator / real mutation
    is applied.
    """

    timestamp_s: float
    workload_id: str
    current_rho: Optional[float]
    recommended_rho: Optional[float]
    action: str
    predicted_goodput_per_dollar_delta: Optional[float]
    predicted_sla_risk_delta: Optional[float]
    predicted_queue_p99_ms: Optional[float]
    confidence: str
    executed: bool = False
    source: str = "dynamic_frontier_estimator_v1"
    timestamp_iso: str = ""
    notes: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        if not d.get("timestamp_iso"):
            d["timestamp_iso"] = _now_utc_iso()
        return d

    @classmethod
    def from_decision(cls, decision: DynamicFrontierDecision, *,
                       timestamp_s: float,
                       predicted_queue_p99_ms: Optional[float] = None,
                       executed: bool = False,
                       notes: Iterable[str] = ()) -> "DynamicFrontierShadowLog":
        return cls(
            timestamp_s=timestamp_s,
            workload_id=decision.workload_id,
            current_rho=decision.current_rho,
            recommended_rho=decision.recommended_rho,
            action=decision.action,
            predicted_goodput_per_dollar_delta=
                decision.expected_goodput_per_dollar_delta,
            predicted_sla_risk_delta=decision.expected_sla_risk_delta,
            predicted_queue_p99_ms=predicted_queue_p99_ms,
            confidence=decision.confidence,
            executed=executed,
            source=decision.source,
            timestamp_iso=_now_utc_iso(),
            notes=tuple(notes) or tuple(decision.notes or ()),
        )


@dataclass(frozen=True)
class DynamicFrontierOutcome:
    """One realized outcome for a previously logged dynamic decision.

    Every observed-* field is optional — pilot telemetry may report a
    subset. ``was_safe`` is the post-hoc verdict (True iff the realized
    outcome stayed within the configured safety thresholds).
    """

    timestamp_s: float
    workload_id: str
    recommended_rho: Optional[float]
    observed_rho: Optional[float]
    observed_goodput_per_dollar: Optional[float] = None
    observed_timeout_pct: Optional[float] = None
    observed_queue_p99_ms: Optional[float] = None
    observed_latency_p99_ms: Optional[float] = None
    observed_sla_violation_pct: Optional[float] = None
    was_safe: Optional[bool] = None
    rho_error: Optional[float] = None
    predicted_goodput_per_dollar: Optional[float] = None
    goodput_per_dollar_error: Optional[float] = None
    predicted_queue_p99_ms: Optional[float] = None
    queue_p99_error_ms: Optional[float] = None
    predicted_timeout_pct: Optional[float] = None
    timeout_pct_error: Optional[float] = None
    source: str = "dynamic_frontier_estimator_v1"
    notes: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


def write_shadow_log_entry(path: str,
                            entry: DynamicFrontierShadowLog) -> None:
    """Append one JSON-encoded entry to the shadow log."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict(), sort_keys=True,
                            default=str) + "\n")


def read_shadow_log(path: str) -> list[DynamicFrontierShadowLog]:
    """Read every entry from a JSONL shadow log."""
    out: list[DynamicFrontierShadowLog] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Drop unknown keys; recreate via dataclass kwargs.
            allowed = set(DynamicFrontierShadowLog.__dataclass_fields__)
            payload = {k: v for k, v in data.items() if k in allowed}
            if "notes" in payload and isinstance(payload["notes"], list):
                payload["notes"] = tuple(payload["notes"])
            out.append(DynamicFrontierShadowLog(**payload))
    return out


def write_outcome(path: str, outcome: DynamicFrontierOutcome) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(outcome.to_dict(), sort_keys=True,
                            default=str) + "\n")


def read_outcomes(path: str) -> list[DynamicFrontierOutcome]:
    out: list[DynamicFrontierOutcome] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            allowed = set(DynamicFrontierOutcome.__dataclass_fields__)
            payload = {k: v for k, v in data.items() if k in allowed}
            if "notes" in payload and isinstance(payload["notes"], list):
                payload["notes"] = tuple(payload["notes"])
            out.append(DynamicFrontierOutcome(**payload))
    return out


def compare_prediction_to_observed(
    log: DynamicFrontierShadowLog,
    observed_rho: Optional[float],
    *,
    observed_goodput_per_dollar: Optional[float] = None,
    observed_timeout_pct: Optional[float] = None,
    observed_queue_p99_ms: Optional[float] = None,
    observed_latency_p99_ms: Optional[float] = None,
    observed_sla_violation_pct: Optional[float] = None,
    safety_timeout_pct: float = 10.0,
    safety_queue_p99_ms: float = 2000.0,
) -> DynamicFrontierOutcome:
    """Combine a shadow log entry with the realized outcome metrics.

    The returned :class:`DynamicFrontierOutcome` carries:

    - ``rho_error`` — observed_rho − recommended_rho.
    - ``goodput_per_dollar_error`` — observed − predicted.
    - ``queue_p99_error_ms`` — observed − predicted.
    - ``timeout_pct_error`` — observed − predicted.
    - ``was_safe`` — observed safety verdict post-hoc.
    """
    rho_err = None
    if (observed_rho is not None and log.recommended_rho is not None):
        rho_err = observed_rho - log.recommended_rho
    queue_err = None
    if (observed_queue_p99_ms is not None
            and log.predicted_queue_p99_ms is not None):
        queue_err = observed_queue_p99_ms - log.predicted_queue_p99_ms
    timeout_err = None
    # predicted_timeout from the shadow log is implicit (we may add it
    # later); for now we leave it None and let the caller supply it.
    # was_safe
    was_safe = None
    if (observed_timeout_pct is not None
            or observed_queue_p99_ms is not None):
        if ((observed_timeout_pct is not None
                and observed_timeout_pct > safety_timeout_pct)
                or (observed_queue_p99_ms is not None
                    and observed_queue_p99_ms > safety_queue_p99_ms)):
            was_safe = False
        else:
            was_safe = True
    return DynamicFrontierOutcome(
        timestamp_s=log.timestamp_s,
        workload_id=log.workload_id,
        recommended_rho=log.recommended_rho,
        observed_rho=observed_rho,
        observed_goodput_per_dollar=observed_goodput_per_dollar,
        observed_timeout_pct=observed_timeout_pct,
        observed_queue_p99_ms=observed_queue_p99_ms,
        observed_latency_p99_ms=observed_latency_p99_ms,
        observed_sla_violation_pct=observed_sla_violation_pct,
        was_safe=was_safe,
        rho_error=rho_err,
        predicted_queue_p99_ms=log.predicted_queue_p99_ms,
        queue_p99_error_ms=queue_err,
        timeout_pct_error=timeout_err,
        source=log.source,
        notes=log.notes,
    )
