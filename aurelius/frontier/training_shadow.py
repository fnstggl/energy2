"""Training-frontier shadow log (JSONL, recommendation-only).

Sibling of ``aurelius/frontier/shadow.py`` for training-workload
decisions. Records one ``TrainingFrontierShadowLog`` per recommendation;
``executed`` defaults to ``False`` (the training-frontier controller is
recommendation-only — real-cluster execution is disabled by default).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from .training_models import (
    TrainingFrontierCandidate,
    TrainingFrontierDecision,
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class TrainingFrontierShadowLog:
    """One shadow-log entry for a training-frontier decision."""

    timestamp_s: float
    workload_id: str
    current_candidate: Optional[dict]
    recommended_candidate: Optional[dict]
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float]
    expected_gpu_hour_delta: Optional[float]
    expected_queue_wait_delta_s: Optional[float]
    expected_fragmentation_delta_pct: Optional[float]
    expected_starvation_delta_pct: Optional[float]
    confidence: str
    safety_vetoes: tuple
    executed: bool = False
    source: str = "training_frontier_v1"
    timestamp_iso: str = ""
    notes: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        if not d.get("timestamp_iso"):
            d["timestamp_iso"] = _now_utc_iso()
        return d

    @classmethod
    def from_decision(cls, decision: TrainingFrontierDecision, *,
                       timestamp_s: float,
                       executed: bool = False,
                       notes: Iterable[str] = ()) -> "TrainingFrontierShadowLog":
        return cls(
            timestamp_s=timestamp_s,
            workload_id=decision.workload_id,
            current_candidate=(decision.current_candidate.to_dict()
                                if decision.current_candidate is not None
                                else None),
            recommended_candidate=(decision.selected_candidate.to_dict()
                                    if decision.selected_candidate is not None
                                    else None),
            action=decision.action,
            reason=decision.reason,
            expected_goodput_per_dollar_delta=
                decision.expected_goodput_per_dollar_delta,
            expected_gpu_hour_delta=decision.expected_gpu_hour_delta,
            expected_queue_wait_delta_s=decision.expected_queue_wait_delta_s,
            expected_fragmentation_delta_pct=
                decision.expected_fragmentation_delta_pct,
            expected_starvation_delta_pct=
                decision.expected_starvation_delta_pct,
            confidence=decision.confidence,
            safety_vetoes=tuple(decision.safety_vetoes),
            executed=executed,
            source=decision.source,
            timestamp_iso=_now_utc_iso(),
            notes=tuple(notes) or tuple(decision.notes or ()))


def write_training_shadow_log_entry(
    path: str, entry: TrainingFrontierShadowLog) -> None:
    """Append one JSON-encoded entry to the shadow log."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict(), sort_keys=True,
                            default=str) + "\n")


def read_training_shadow_log(path: str) -> list[TrainingFrontierShadowLog]:
    """Read every entry from a JSONL shadow log."""
    out: list[TrainingFrontierShadowLog] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            allowed = set(TrainingFrontierShadowLog.__dataclass_fields__)
            payload = {k: v for k, v in data.items() if k in allowed}
            for key in ("safety_vetoes", "notes"):
                if key in payload and isinstance(payload[key], list):
                    payload[key] = tuple(payload[key])
            out.append(TrainingFrontierShadowLog(**payload))
    return out
