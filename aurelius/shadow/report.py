"""ShadowReport — aggregated comparison of predicted vs realized shadow savings.

Produces a human-readable and machine-readable summary showing:
- How many jobs were decided (and realized)
- Mean predicted savings vs mean realized savings
- Per-workload-type breakdown
- Forecast accuracy (predicted DA price vs actual RT price)
- Confidence that the optimizer beats the CPO baseline

Typical usage:
    report = ShadowReport.from_records(realized_records)
    print(report.to_text())
    paths = report.save(Path("reports/shadow/"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import DecisionRecord

logger = logging.getLogger(__name__)


@dataclass
class WorkloadBreakdown:
    workload_type: str
    n_jobs: int
    n_realized: int
    mean_predicted_savings_pct: float
    mean_realized_savings_pct: Optional[float]
    mean_forecast_error_pct: Optional[float]   # |predicted_da - realized_rt| / realized_rt

    def to_dict(self) -> dict:
        return {
            "workload_type": self.workload_type,
            "n_jobs": self.n_jobs,
            "n_realized": self.n_realized,
            "mean_predicted_savings_pct": round(self.mean_predicted_savings_pct, 2),
            "mean_realized_savings_pct": (
                round(self.mean_realized_savings_pct, 2)
                if self.mean_realized_savings_pct is not None else None
            ),
            "mean_forecast_error_pct": (
                round(self.mean_forecast_error_pct, 2)
                if self.mean_forecast_error_pct is not None else None
            ),
        }


@dataclass
class ShadowReport:
    run_id: str
    decision_time: Optional[datetime]
    generated_at: datetime
    n_jobs: int
    n_realized: int
    n_pending: int

    # Aggregate savings
    mean_predicted_savings_pct: float
    mean_realized_savings_pct: Optional[float]  # None if no realized records
    median_predicted_savings_pct: float
    median_realized_savings_pct: Optional[float]

    # Forecast accuracy: predicted DA price vs actual RT price
    mean_forecast_error_mae: Optional[float]   # $/MWh absolute error
    mean_forecast_error_pct: Optional[float]   # % relative error

    # Delta: was prediction conservative or optimistic?
    # positive = optimizer was conservative (realized > predicted = good)
    # negative = optimizer over-estimated savings
    mean_savings_delta_pp: Optional[float]

    # Per-workload breakdown
    by_workload: list[WorkloadBreakdown] = field(default_factory=list)

    # Methodology note
    methodology_note: str = (
        "Predicted savings computed against current_price_only baseline "
        "using forecast DA prices at decision time. "
        "Realized savings computed against same baseline using actual RT settlement prices."
    )

    data_source_note: str = ""

    @classmethod
    def from_records(
        cls,
        records: list[DecisionRecord],
        data_source_note: str = "",
    ) -> "ShadowReport":
        """Build a ShadowReport from a list of DecisionRecords."""
        if not records:
            return cls(
                run_id="(no records)",
                decision_time=None,
                generated_at=datetime.now(timezone.utc),
                n_jobs=0,
                n_realized=0,
                n_pending=0,
                mean_predicted_savings_pct=0.0,
                mean_realized_savings_pct=None,
                median_predicted_savings_pct=0.0,
                median_realized_savings_pct=None,
                mean_forecast_error_mae=None,
                mean_forecast_error_pct=None,
                mean_savings_delta_pp=None,
                data_source_note=data_source_note,
            )

        realized = [r for r in records if r.is_realized]
        pending = [r for r in records if not r.is_realized]

        run_id = records[0].run_id
        decision_time = records[0].decision_time

        # Aggregate predicted savings
        pred_savings = [r.predicted_savings_pct for r in records]
        mean_pred = sum(pred_savings) / len(pred_savings)
        median_pred = _median(pred_savings)

        # Aggregate realized savings
        mean_real: Optional[float] = None
        median_real: Optional[float] = None
        mean_delta: Optional[float] = None
        mae: Optional[float] = None
        mape: Optional[float] = None

        if realized:
            real_savings = [r.realized_savings_pct for r in realized]
            mean_real = sum(real_savings) / len(real_savings)
            median_real = _median(real_savings)
            deltas = [r.savings_delta for r in realized if r.savings_delta is not None]
            mean_delta = sum(deltas) / len(deltas) if deltas else None

            # Forecast accuracy: compare forecast_da_price_p50 vs realized_rt_price
            errors = []
            rel_errors = []
            for r in realized:
                if r.realized_rt_price is not None and r.realized_rt_price > 0:
                    err = abs(r.forecast_da_price_p50 - r.realized_rt_price)
                    errors.append(err)
                    rel_errors.append(err / r.realized_rt_price * 100.0)
            mae = sum(errors) / len(errors) if errors else None
            mape = sum(rel_errors) / len(rel_errors) if rel_errors else None

        by_workload = _build_workload_breakdown(records)

        return cls(
            run_id=run_id,
            decision_time=decision_time,
            generated_at=datetime.now(timezone.utc),
            n_jobs=len(records),
            n_realized=len(realized),
            n_pending=len(pending),
            mean_predicted_savings_pct=mean_pred,
            mean_realized_savings_pct=mean_real,
            median_predicted_savings_pct=median_pred,
            median_realized_savings_pct=median_real,
            mean_forecast_error_mae=mae,
            mean_forecast_error_pct=mape,
            mean_savings_delta_pp=mean_delta,
            by_workload=by_workload,
            data_source_note=data_source_note,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "n_jobs": self.n_jobs,
                "n_realized": self.n_realized,
                "n_pending": self.n_pending,
                "mean_predicted_savings_pct": round(self.mean_predicted_savings_pct, 2),
                "mean_realized_savings_pct": (
                    round(self.mean_realized_savings_pct, 2)
                    if self.mean_realized_savings_pct is not None else None
                ),
                "median_predicted_savings_pct": round(self.median_predicted_savings_pct, 2),
                "median_realized_savings_pct": (
                    round(self.median_realized_savings_pct, 2)
                    if self.median_realized_savings_pct is not None else None
                ),
                "mean_savings_delta_pp": (
                    round(self.mean_savings_delta_pp, 2)
                    if self.mean_savings_delta_pp is not None else None
                ),
                "forecast_accuracy": {
                    "mae_usd_per_mwh": (
                        round(self.mean_forecast_error_mae, 2)
                        if self.mean_forecast_error_mae is not None else None
                    ),
                    "mape_pct": (
                        round(self.mean_forecast_error_pct, 2)
                        if self.mean_forecast_error_pct is not None else None
                    ),
                },
            },
            "by_workload": [b.to_dict() for b in self.by_workload],
            "methodology_note": self.methodology_note,
            "data_source_note": self.data_source_note,
        }

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "AURELIUS SHADOW MODE REPORT",
            "=" * 72,
            f"Run ID:         {self.run_id}",
            f"Decision time:  {self.decision_time.isoformat() if self.decision_time else 'N/A'}",
            f"Generated:      {self.generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "",
            "-" * 72,
            "SUMMARY",
            "-" * 72,
            f"Jobs decided:   {self.n_jobs}",
            f"Jobs realized:  {self.n_realized}",
            f"Jobs pending:   {self.n_pending}",
            "",
        ]

        preds = f"{self.mean_predicted_savings_pct:.1f}%"
        lines.append(f"Mean predicted savings vs CPO:  {preds:>10}")

        if self.mean_realized_savings_pct is not None:
            real = f"{self.mean_realized_savings_pct:.1f}%"
            lines.append(f"Mean realized  savings vs CPO:  {real:>10}")
        else:
            lines.append(f"Mean realized  savings vs CPO:  PENDING (no realized records yet)")

        if self.mean_savings_delta_pp is not None:
            delta_sign = "+" if self.mean_savings_delta_pp >= 0 else ""
            lines.append(
                f"  Optimizer delta (realized-predicted):  "
                f"{delta_sign}{self.mean_savings_delta_pp:.1f}pp  "
                f"({'conservative — good' if self.mean_savings_delta_pp >= 0 else 'optimistic — review forecast'})"
            )

        if self.mean_forecast_error_mae is not None:
            lines.append(
                f"Forecast accuracy (DA vs RT):  "
                f"MAE={self.mean_forecast_error_mae:.2f} $/MWh  "
                f"MAPE={self.mean_forecast_error_pct:.1f}%"
            )

        if self.by_workload:
            lines += [
                "",
                "-" * 72,
                "BY WORKLOAD TYPE",
                "-" * 72,
                f"{'Workload':<28} {'Pred%':>8} {'Real%':>8} {'N':>5} {'NReal':>6}",
                "-" * 72,
            ]
            for b in sorted(self.by_workload, key=lambda x: -x.mean_predicted_savings_pct):
                real_str = (
                    f"{b.mean_realized_savings_pct:.1f}%"
                    if b.mean_realized_savings_pct is not None else "pending"
                )
                lines.append(
                    f"{b.workload_type:<28} "
                    f"{b.mean_predicted_savings_pct:>7.1f}%  "
                    f"{real_str:>8}  "
                    f"{b.n_jobs:>4}  "
                    f"{b.n_realized:>5}"
                )

        lines += [
            "",
            "-" * 72,
            "METHODOLOGY",
            "-" * 72,
            self.methodology_note,
        ]
        if self.data_source_note:
            lines += ["", f"Data: {self.data_source_note}"]

        lines += [
            "",
            "IMPORTANT: Predicted savings are from historical DA backtesting.",
            "Realized savings require actual RT settlement data for comparison.",
            "Only realized results constitute proven pilot-grade savings evidence.",
            "=" * 72,
        ]
        return "\n".join(lines)

    def save(self, output_dir: Path) -> dict:
        """Save JSON and TXT report files.

        Returns: {"json": Path, "txt": Path}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = self.generated_at.strftime("%Y%m%dT%H%M%SZ")
        json_path = output_dir / f"shadow_report_{ts}.json"
        txt_path = output_dir / f"shadow_report_{ts}.txt"

        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        txt_path.write_text(self.to_text(), encoding="utf-8")

        logger.info(f"ShadowReport saved: {json_path}, {txt_path}")
        return {"json": json_path, "txt": txt_path}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _build_workload_breakdown(records: list[DecisionRecord]) -> list[WorkloadBreakdown]:
    groups: dict[str, list[DecisionRecord]] = {}
    for r in records:
        groups.setdefault(r.workload_type, []).append(r)

    result = []
    for wt, group in sorted(groups.items()):
        realized = [r for r in group if r.is_realized]
        pred_savings = [r.predicted_savings_pct for r in group]
        mean_pred = sum(pred_savings) / len(pred_savings)

        mean_real: Optional[float] = None
        if realized:
            real_savings = [r.realized_savings_pct for r in realized if r.realized_savings_pct is not None]
            mean_real = sum(real_savings) / len(real_savings) if real_savings else None

        # Forecast error for realized records
        errors = []
        for r in realized:
            if r.realized_rt_price is not None and r.realized_rt_price > 0:
                errors.append(
                    abs(r.forecast_da_price_p50 - r.realized_rt_price)
                    / r.realized_rt_price * 100.0
                )
        mean_err = sum(errors) / len(errors) if errors else None

        result.append(WorkloadBreakdown(
            workload_type=wt,
            n_jobs=len(group),
            n_realized=len(realized),
            mean_predicted_savings_pct=mean_pred,
            mean_realized_savings_pct=mean_real,
            mean_forecast_error_pct=mean_err,
        ))
    return result
