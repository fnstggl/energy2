"""Forecast drift detector for Aurelius.

Monitors recent forecast error against a stored baseline MAPE.
Flags when a model needs retraining before errors compound into bad decisions.

Design principles:
- Read-only: never modifies models, artifacts, or decisions
- Failure-tolerant: always returns a DriftReport; never raises
- Conservative: insufficient data → drift_detected=False (don't cry wolf)
- Threshold: recent_mape > baseline_mape × threshold_multiplier (default 2×)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a drift detection check.

    Attributes:
        model_name: Label identifying which model was checked.
        checked_at: UTC timestamp when check ran.
        n_recent_records: Total records passed in (not all may be valid).
        n_valid_records: Records with computable MAPE values.
        recent_mape: MAPE computed from recent valid records (NaN if insufficient).
        baseline_mape: Reference MAPE from training / prior holdout evaluation.
        drift_ratio: recent_mape / baseline_mape (NaN if either is invalid).
        drift_detected: True if drift_ratio > threshold_multiplier.
        alert_message: Human-readable explanation (None if no drift).
    """

    model_name: str
    checked_at: datetime
    n_recent_records: int
    n_valid_records: int
    recent_mape: float
    baseline_mape: float
    drift_ratio: float
    drift_detected: bool
    alert_message: Optional[str] = None

    def to_dict(self) -> dict:
        def _fmt(v: float) -> Optional[float]:
            return round(v, 6) if not math.isnan(v) else None

        return {
            "model_name": self.model_name,
            "checked_at": self.checked_at.isoformat(),
            "n_recent_records": self.n_recent_records,
            "n_valid_records": self.n_valid_records,
            "recent_mape": _fmt(self.recent_mape),
            "baseline_mape": _fmt(self.baseline_mape),
            "drift_ratio": _fmt(self.drift_ratio),
            "drift_detected": self.drift_detected,
            "alert_message": self.alert_message,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _nan_report(
    model_name: str,
    baseline_mape: float,
    n_records: int,
    message: str,
) -> DriftReport:
    return DriftReport(
        model_name=model_name,
        checked_at=datetime.utcnow(),
        n_recent_records=n_records,
        n_valid_records=0,
        recent_mape=float("nan"),
        baseline_mape=baseline_mape,
        drift_ratio=float("nan"),
        drift_detected=False,
        alert_message=message,
    )


class DriftDetector:
    """Detects forecast drift by comparing recent p50 errors against a baseline MAPE.

    Uses PostExecutionRecord JSONL data (fields: ``energy_cost_p50_error``,
    ``forecast_energy_cost_p50``) to compute a recent empirical MAPE.

    Error formula:
        actual        = forecast_energy_cost_p50 + energy_cost_p50_error
                        (because error = actual - p50_forecast)
        ape_i         = |error_i| / |actual_i|   (skips rows where actual == 0)
        recent_mape   = mean(ape_i)

    Drift rule:
        drift_detected = recent_mape > baseline_mape × threshold_multiplier
        Default threshold_multiplier = 2.0  (2× baseline error → alert)

    Conservative under low data:
        If fewer than ``min_records`` valid rows are found, ``drift_detected``
        is set to False and the report documents the insufficient-data reason.

    Usage::

        detector = DriftDetector(threshold_multiplier=2.0, min_records=10)
        report = detector.check(records, baseline_mape=0.08, model_name="price")
        if report.drift_detected:
            print(report.alert_message)
    """

    def __init__(
        self,
        threshold_multiplier: float = 2.0,
        min_records: int = 10,
    ) -> None:
        """
        Args:
            threshold_multiplier: Alert when recent_mape > baseline × this value.
            min_records: Minimum valid records for a reliable MAPE estimate.
        """
        if threshold_multiplier <= 0:
            raise ValueError(
                f"threshold_multiplier must be > 0, got {threshold_multiplier}"
            )
        if min_records < 1:
            raise ValueError(f"min_records must be >= 1, got {min_records}")
        self.threshold_multiplier = threshold_multiplier
        self.min_records = min_records

    # ------------------------------------------------------------------
    # Core check API
    # ------------------------------------------------------------------

    def check(
        self,
        recent_records: list[dict],
        baseline_mape: float,
        model_name: str = "price_model",
    ) -> DriftReport:
        """Check for drift in a list of PostExecutionRecord dicts.

        Args:
            recent_records: PostExecutionRecord dicts (from JSONL).
                Each dict must contain ``energy_cost_p50_error`` and
                ``forecast_energy_cost_p50`` for the row to contribute
                to the MAPE calculation.  Rows with missing or None
                values are silently skipped.
            baseline_mape: Reference MAPE from training or holdout.
                Must be > 0 for drift_ratio to be meaningful.
            model_name: Label for reporting.

        Returns:
            DriftReport — never raises.
        """
        try:
            return self._check_internal(recent_records, baseline_mape, model_name)
        except Exception as exc:
            logger.error("DriftDetector.check unexpected error: %s", exc)
            return _nan_report(
                model_name,
                baseline_mape,
                len(recent_records) if recent_records else 0,
                f"Unexpected error during drift check: {exc}",
            )

    def check_from_jsonl(
        self,
        jsonl_path: "str | Path",
        baseline_mape: float,
        model_name: str = "price_model",
        last_n: int = 500,
    ) -> DriftReport:
        """Load PostExecutionRecord JSONL and run drift check.

        Args:
            jsonl_path: Path to the PostExecutionRecord JSONL file.
            baseline_mape: Reference MAPE from training or holdout.
            model_name: Label for reporting.
            last_n: Use only the last N records (most recent first-in-file).

        Returns:
            DriftReport — never raises.
        """
        path = Path(jsonl_path)
        if not path.exists():
            return _nan_report(
                model_name,
                baseline_mape,
                0,
                f"JSONL file not found: {path}",
            )
        try:
            records: list[dict] = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            logger.error("DriftDetector: failed to read %s: %s", path, exc)
            return _nan_report(
                model_name,
                baseline_mape,
                0,
                f"Failed to read JSONL: {exc}",
            )

        recent = records[-last_n:] if len(records) > last_n else records
        return self.check(recent, baseline_mape, model_name)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _check_internal(
        self,
        recent_records: list[dict],
        baseline_mape: float,
        model_name: str,
    ) -> DriftReport:
        checked_at = datetime.utcnow()
        n_total = len(recent_records) if recent_records else 0

        if not recent_records:
            return DriftReport(
                model_name=model_name,
                checked_at=checked_at,
                n_recent_records=0,
                n_valid_records=0,
                recent_mape=float("nan"),
                baseline_mape=baseline_mape,
                drift_ratio=float("nan"),
                drift_detected=False,
                alert_message="No recent records provided for drift check.",
            )

        # Compute per-record APE
        ape_values: list[float] = []
        for rec in recent_records:
            p50_error = rec.get("energy_cost_p50_error")
            p50_forecast = rec.get("forecast_energy_cost_p50")
            if p50_error is None or p50_forecast is None:
                continue
            try:
                p50_error = float(p50_error)
                p50_forecast = float(p50_forecast)
            except (TypeError, ValueError):
                continue
            # actual = forecast + error  (error = actual - forecast)
            actual = p50_forecast + p50_error
            if actual == 0.0:
                continue
            ape_values.append(abs(p50_error) / abs(actual))

        n_valid = len(ape_values)

        if n_valid < self.min_records:
            return DriftReport(
                model_name=model_name,
                checked_at=checked_at,
                n_recent_records=n_total,
                n_valid_records=n_valid,
                recent_mape=float("nan"),
                baseline_mape=baseline_mape,
                drift_ratio=float("nan"),
                drift_detected=False,
                alert_message=(
                    f"Insufficient valid records: {n_valid} < {self.min_records}. "
                    "Drift check deferred."
                ),
            )

        recent_mape = float(sum(ape_values) / n_valid)

        # Validate baseline
        if baseline_mape <= 0.0 or math.isnan(baseline_mape) or math.isinf(baseline_mape):
            return DriftReport(
                model_name=model_name,
                checked_at=checked_at,
                n_recent_records=n_total,
                n_valid_records=n_valid,
                recent_mape=recent_mape,
                baseline_mape=baseline_mape,
                drift_ratio=float("nan"),
                drift_detected=False,
                alert_message=f"Invalid baseline_mape={baseline_mape}; drift ratio not computable.",
            )

        drift_ratio = recent_mape / baseline_mape
        drift_detected = drift_ratio > self.threshold_multiplier

        alert_message: Optional[str] = None
        if drift_detected:
            alert_message = (
                f"DRIFT DETECTED in {model_name}: "
                f"recent_mape={recent_mape:.4f} is {drift_ratio:.2f}× "
                f"baseline_mape={baseline_mape:.4f} "
                f"(threshold={self.threshold_multiplier}×, n={n_valid}). "
                "Retraining recommended."
            )
            logger.warning(alert_message)
        else:
            logger.info(
                "Drift check OK for %s: recent_mape=%.4f, "
                "baseline_mape=%.4f, ratio=%.2f (threshold=%.1f×, n=%d)",
                model_name,
                recent_mape,
                baseline_mape,
                drift_ratio,
                self.threshold_multiplier,
                n_valid,
            )

        return DriftReport(
            model_name=model_name,
            checked_at=checked_at,
            n_recent_records=n_total,
            n_valid_records=n_valid,
            recent_mape=recent_mape,
            baseline_mape=baseline_mape,
            drift_ratio=drift_ratio,
            drift_detected=drift_detected,
            alert_message=alert_message,
        )
