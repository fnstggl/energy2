"""RealizedSavingsCalculator — fill in actual RT prices for pending shadow decisions.

After the scheduled job windows have passed (typically 7-14 days), load the
actual real-time prices and compare them against the optimizer's predictions.

Leakage invariant: RT prices are strictly post-hoc data. They must NEVER be
available in the LiveShadowRunner. This class is the only place RT prices touch
DecisionRecords, and it must only be run AFTER the job's scheduled window has passed.
"""

from __future__ import annotations

import logging
from datetime import timedelta, timezone
from typing import Optional

import pandas as pd

from .models import DecisionRecord

logger = logging.getLogger(__name__)


class RealizedSavingsCalculator:
    """Fill realized_ fields on DecisionRecords using actual RT settlement prices.

    Usage:
        realizer = RealizedSavingsCalculator(rt_price_df)
        realized_records = realizer.realize(pending_records)
        # realized_records now have realized_savings_pct filled in
    """

    def __init__(self, rt_price_df: pd.DataFrame) -> None:
        """
        Args:
            rt_price_df: Canonical RT settlement price DataFrame
                (columns: timestamp, region, price_per_mwh).
                These are the prices the customer ACTUALLY paid.
        """
        self._rt_price_data = self._load_price_df(rt_price_df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def realize(
        self,
        records: list[DecisionRecord],
        skip_realized: bool = True,
    ) -> list[DecisionRecord]:
        """Fill realized_ fields from actual RT prices.

        For each pending DecisionRecord:
        1. Look up actual RT price at scheduled_region, scheduled_start.
        2. Compute realized_energy_cost = power * rt_price / 1000 * runtime_h.
        3. Look up actual RT price at baseline_region, baseline_start.
        4. Compute realized_baseline_cost = power * baseline_rt / 1000 * runtime_h.
        5. Compute realized_savings_pct = (1 - opt/base) * 100.

        Args:
            records:        List of DecisionRecord.
            skip_realized:  If True, skip records that already have realized_ data.

        Returns:
            Same list with realized_ fields populated where RT data exists.
            Records with missing RT data get realization_note="missing_rt_price".
        """
        updated = 0
        skipped_already = 0
        skipped_no_data = 0

        for record in records:
            if skip_realized and record.is_realized:
                skipped_already += 1
                continue

            self._realize_one(record)

            if record.is_realized:
                updated += 1
            else:
                skipped_no_data += 1

        logger.info(
            f"RealizedSavingsCalculator: {updated} realized, "
            f"{skipped_already} already done, {skipped_no_data} missing RT data"
        )
        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _realize_one(self, record: DecisionRecord) -> None:
        # Compute actual cost for optimizer decision window
        opt_rt = self._compute_windowed_rt_cost(
            region=record.scheduled_region,
            start=record.scheduled_start,
            runtime_h=record.scheduled_runtime_h,
            power_kw=record.power_kw,
        )
        if opt_rt is None:
            record.realization_note = "missing_rt_price"
            return

        # Compute actual cost for baseline decision window
        base_runtime_h = record.scheduled_runtime_h  # same job, same runtime
        base_rt = self._compute_windowed_rt_cost(
            region=record.baseline_region,
            start=record.baseline_start,
            runtime_h=base_runtime_h,
            power_kw=record.power_kw,
        )
        if base_rt is None:
            record.realization_note = "missing_baseline_rt_price"
            return

        record.realized_energy_cost = opt_rt["cost"]
        record.realized_rt_price = opt_rt["avg_price"]
        record.realized_baseline_cost = base_rt["cost"]
        record.realized_baseline_rt_price = base_rt["avg_price"]

        if base_rt["cost"] > 0:
            record.realized_savings_pct = (1.0 - opt_rt["cost"] / base_rt["cost"]) * 100.0
        else:
            record.realized_savings_pct = 0.0

        record.realization_note = "realized"

    def _compute_windowed_rt_cost(
        self,
        region: str,
        start,
        runtime_h: float,
        power_kw: float,
    ) -> Optional[dict]:
        """Compute total RT cost over the job window.

        Returns {"cost": float, "avg_price": float} or None if no RT data.
        """
        region_prices = self._rt_price_data.get(region, {})
        if not region_prices:
            return None

        current = start.replace(minute=0, second=0, microsecond=0)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        end = current + timedelta(hours=runtime_h)

        total_cost = 0.0
        total_price_hours = 0.0
        total_hours = 0.0
        missing = 0

        while current < end:
            hour_fraction = min(1.0, (end - current).total_seconds() / 3600.0)
            if hour_fraction <= 0:
                break

            price = region_prices.get(current)
            if price is None:
                missing += 1
                current += timedelta(hours=1)
                continue

            energy_kwh = power_kw * hour_fraction
            total_cost += (price / 1000.0) * energy_kwh
            total_price_hours += price * hour_fraction
            total_hours += hour_fraction
            current += timedelta(hours=1)

        if total_hours == 0:
            return None

        # Allow up to 50% missing hours (rough real-world tolerance)
        if missing > total_hours:
            return None

        avg_price = total_price_hours / total_hours if total_hours > 0 else 0.0
        return {"cost": total_cost, "avg_price": avg_price}

    @staticmethod
    def _load_price_df(df: pd.DataFrame) -> dict:
        """Convert canonical price DataFrame to {region: {timestamp: price}}."""
        if df is None or df.empty:
            return {}
        result: dict = {}
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for _, row in df.iterrows():
            region = str(row["region"])
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            ts = ts.replace(minute=0, second=0, microsecond=0)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            result.setdefault(region, {})[ts] = float(row["price_per_mwh"])
        return result
