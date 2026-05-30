"""Shadow logging for frontier decisions (recommendation-only persistence).

A ``FrontierShadowDecisionLog`` is the append-only audit record of one
recommendation. It captures expected KPI / GPU-hour / SLA-risk deltas, the
controller's reason, and whether the decision was executed (only ever true
in simulator mode). No production mutation happens here — this module is
storage only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from .models import EXECUTION_MODE_SHADOW, FrontierAction


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FrontierShadowDecisionLog:
    """One persisted frontier recommendation (append-only)."""

    timestamp: str
    workload_id: str
    current_rho: Optional[float]
    recommended_rho: Optional[float]
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_gpu_hour_delta: Optional[float] = None
    expected_sla_risk_delta: Optional[float] = None
    confidence: str = "unknown"
    safety_vetoes: list = field(default_factory=list)
    executed: bool = False
    execution_mode: str = EXECUTION_MODE_SHADOW
    source: str = "frontier_controller_v1"

    def __post_init__(self):
        if self.action not in (FrontierAction.RECOMMEND_RHO,
                               FrontierAction.KEEP_RHO,
                               FrontierAction.LOWER_RHO,
                               FrontierAction.INSUFFICIENT_TELEMETRY):
            raise ValueError(
                f"unknown frontier action {self.action!r}")
        # In shadow / real_disabled execution modes, executed must remain False.
        if self.execution_mode in ("shadow", "real_disabled") and self.executed:
            raise ValueError(
                f"execution_mode {self.execution_mode!r} disallows executed=True")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_decision(cls, decision, *, execution_mode: str = EXECUTION_MODE_SHADOW,
                      executed: bool = False) -> "FrontierShadowDecisionLog":
        return cls(
            timestamp=_now_utc_iso(),
            workload_id=decision.workload_id,
            current_rho=decision.previous_rho,
            recommended_rho=decision.selected_rho,
            action=decision.action, reason=decision.reason,
            expected_goodput_per_dollar_delta=decision.expected_goodput_per_dollar_delta,
            expected_gpu_hour_delta=decision.expected_gpu_hour_delta,
            expected_sla_risk_delta=decision.expected_sla_risk_delta,
            confidence=decision.confidence,
            safety_vetoes=list(decision.safety_vetoes),
            executed=executed, execution_mode=execution_mode,
            source=decision.source)

    @classmethod
    def from_dict(cls, d: dict) -> "FrontierShadowDecisionLog":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def from_json(cls, line: str) -> "FrontierShadowDecisionLog":
        return cls.from_dict(json.loads(line))


def write_shadow_log_entry(path: str, entry: FrontierShadowDecisionLog) -> None:
    """Append ``entry`` to ``path`` as a single JSONL row."""
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")


def read_shadow_log(path: str) -> list[FrontierShadowDecisionLog]:
    """Read every JSONL row from ``path`` into a list of log entries."""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(FrontierShadowDecisionLog.from_json(line))
    return out


class FrontierShadowLog:
    """In-memory append-only log; optionally mirrored to a JSONL file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.entries: list[FrontierShadowDecisionLog] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def record(self, decision_or_entry, *,
               execution_mode: str = EXECUTION_MODE_SHADOW,
               executed: bool = False) -> FrontierShadowDecisionLog:
        if isinstance(decision_or_entry, FrontierShadowDecisionLog):
            entry = decision_or_entry
        else:
            entry = FrontierShadowDecisionLog.from_decision(
                decision_or_entry, execution_mode=execution_mode,
                executed=executed)
        self.entries.append(entry)
        if self.path:
            write_shadow_log_entry(self.path, entry)
        return entry

    def record_all(self, decisions: Iterable, *,
                   execution_mode: str = EXECUTION_MODE_SHADOW,
                   executed: bool = False) -> list[FrontierShadowDecisionLog]:
        return [self.record(d, execution_mode=execution_mode, executed=executed)
                for d in decisions]

    def summary(self) -> dict:
        counts = {a: 0 for a in (
            FrontierAction.RECOMMEND_RHO, FrontierAction.KEEP_RHO,
            FrontierAction.LOWER_RHO, FrontierAction.INSUFFICIENT_TELEMETRY)}
        for e in self.entries:
            counts[e.action] = counts.get(e.action, 0) + 1
        return {
            "n_decisions": len(self.entries),
            "action_counts": counts,
            "n_executed": sum(1 for e in self.entries if e.executed),
            "execution_modes": sorted({e.execution_mode for e in self.entries}),
        }
