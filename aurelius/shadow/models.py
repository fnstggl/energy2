"""DecisionRecord — the core shadow-mode data model.

One DecisionRecord is created per scheduled job in a shadow run. It captures
both the optimizer's prediction (at decision time) and, later, the realized
outcome (after the job window has passed).

Serialization: JSON-serializable dict (ISO timestamps, float/None for all numbers).
Storage: JSONL file, one record per line (append-safe, easy to stream).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DecisionRecord:
    """One scheduling decision from a shadow run.

    Predicted fields are filled at decision time by LiveShadowRunner.
    Realized fields (realized_*) start as None and are filled by
    RealizedSavingsCalculator after the scheduled time window has passed.

    The key comparison:
        predicted_savings_pct — what the optimizer expected to save (vs CPO baseline)
        realized_savings_pct  — what was actually saved (vs CPO baseline)

    If |predicted - realized| is small, the forecaster is accurate.
    If realized > predicted, the optimizer was conservative (good).
    If realized < predicted, the optimizer over-estimated savings (needs calibration).
    """

    # --- Identifiers ---
    run_id: str
    job_id: str
    workload_type: str
    decision_time: datetime

    # --- Optimizer decision ---
    scheduled_region: str
    scheduled_start: datetime
    scheduled_end: datetime
    scheduled_runtime_h: float

    # --- Forecast snapshot at decision time ---
    forecast_da_price_p50: float   # $/MWh predicted at scheduled slot
    forecast_da_price_p90: float   # p90 uncertainty bound

    # --- Predicted costs ---
    predicted_energy_cost: float   # optimizer's estimated cost ($)

    # --- Baseline (current_price_only) decision ---
    baseline_region: str
    baseline_start: datetime
    baseline_energy_cost: float    # CPO baseline estimated cost ($)

    # --- Predicted savings vs CPO ---
    predicted_savings_pct: float   # (1 - opt/base) * 100

    # --- Job metadata (for grouping / filtering in reports) ---
    power_kw: float
    gpu_count: int
    sla_class: str = "best_effort"

    # --- Audit metadata ---
    forecaster_version: str = "ml_quantile"
    optimizer_version: str = "greedy_migrate"
    data_source: str = "csv"

    # --- Realized fields (None until RealizedSavingsCalculator fills them) ---
    realized_rt_price: Optional[float] = None   # $/MWh actual RT at scheduled_start
    realized_energy_cost: Optional[float] = None  # power * realized_rt / 1000 * h
    realized_baseline_rt_price: Optional[float] = None
    realized_baseline_cost: Optional[float] = None
    realized_savings_pct: Optional[float] = None
    sla_met: Optional[bool] = None
    realization_note: Optional[str] = None     # e.g. "missing_rt_price"

    @property
    def is_realized(self) -> bool:
        return self.realized_savings_pct is not None

    @property
    def savings_delta(self) -> Optional[float]:
        """realized - predicted (positive = optimizer was conservative = good)."""
        if not self.is_realized:
            return None
        return self.realized_savings_pct - self.predicted_savings_pct

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("decision_time", "scheduled_start", "scheduled_end", "baseline_start"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionRecord":
        dt_fields = ("decision_time", "scheduled_start", "scheduled_end", "baseline_start")
        for key in dt_fields:
            if key in d and isinstance(d[key], str):
                ts = datetime.fromisoformat(d[key])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                d[key] = ts
        return cls(**d)

    @classmethod
    def from_json(cls, line: str) -> "DecisionRecord":
        return cls.from_dict(json.loads(line))


def make_run_id() -> str:
    return str(uuid.uuid4())[:8]
