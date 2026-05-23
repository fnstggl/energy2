"""Queue-state ingestion for queue-aware optimization.

Provides per-region GPU availability and queue congestion signals that allow
the optimizer to route jobs away from congested clusters even when energy
prices are similar.

CSV schema (one row per region × timestamp):
    timestamp,region,cluster_id,gpu_type,available_gpus,queue_depth_jobs,est_wait_hours

queue_data lookup format used internally (mirrors price_data / carbon_data):
    {region: {timestamp_floor_hour: est_wait_hours}}

When multiple clusters exist in the same region the provider aggregates by
taking the weighted mean wait time (weighted by queue_depth_jobs).

Leakage safety: get_wait_hours() uses a "last known at or before T" lookup
so that the optimizer never sees future congestion state during backtesting.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from aurelius.models import QueueState

logger = logging.getLogger(__name__)

# Canonical CSV column names
_COLS = {
    "timestamp", "region", "cluster_id", "gpu_type",
    "available_gpus", "queue_depth_jobs", "est_wait_hours",
}


@dataclass
class QueueProvider:
    """Loads, stores, and looks up GPU cluster queue state.

    Usage::

        provider = QueueProvider.from_csv("data/queue_state.csv")
        wait_h = provider.get_wait_hours("us-west", timestamp=dt)
        queue_data = provider.to_dict_lookup()   # plug into optimizer
    """

    # Internal storage: list[QueueState] sorted by timestamp ascending
    _records: list[QueueState] = field(default_factory=list)

    # Aggregated lookup: {region: sorted list of (timestamp, est_wait_hours)}
    # Built lazily on first call to get_wait_hours / to_dict_lookup.
    _lookup: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)
    _lookup_built: bool = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(cls, path: str) -> "QueueProvider":
        """Load queue state from a CSV file.

        The CSV must contain at minimum: timestamp, region, est_wait_hours.
        Missing optional columns (cluster_id, gpu_type, available_gpus,
        queue_depth_jobs) are filled with sensible defaults.
        """
        df = pd.read_csv(path)
        missing = {"timestamp", "region", "est_wait_hours"} - set(df.columns)
        if missing:
            raise ValueError(f"Queue CSV missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.floor("h")

        if "cluster_id" not in df.columns:
            df["cluster_id"] = "default"
        if "gpu_type" not in df.columns:
            df["gpu_type"] = None
        if "available_gpus" not in df.columns:
            df["available_gpus"] = 0
        if "queue_depth_jobs" not in df.columns:
            df["queue_depth_jobs"] = 0

        records = []
        for _, row in df.iterrows():
            gpu_type = row.get("gpu_type")
            if pd.isna(gpu_type):
                gpu_type = None
            records.append(QueueState(
                timestamp=row["timestamp"].to_pydatetime().replace(tzinfo=timezone.utc),
                region=str(row["region"]),
                cluster_id=str(row["cluster_id"]),
                gpu_type=gpu_type,
                available_gpus=int(row["available_gpus"]),
                queue_depth_jobs=int(row["queue_depth_jobs"]),
                est_wait_hours=float(row["est_wait_hours"]),
            ))

        provider = cls(_records=sorted(records, key=lambda r: r.timestamp))
        return provider

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "QueueProvider":
        """Construct from a DataFrame with the canonical queue schema."""
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp = f.name
        try:
            return cls.from_csv(tmp)
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def _build_lookup(self) -> None:
        """Build the region → [(timestamp, est_wait_hours)] index."""
        from collections import defaultdict

        # Aggregate: for each (region, timestamp) sum queue_depth and compute
        # weighted-mean wait time across clusters.
        agg: dict[tuple[str, datetime], list[tuple[float, int]]] = defaultdict(list)
        for r in self._records:
            agg[(r.region, r.timestamp)].append((r.est_wait_hours, max(1, r.queue_depth_jobs)))

        region_series: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for (region, ts), entries in agg.items():
            total_weight = sum(w for _, w in entries)
            if total_weight == 0:
                wait = entries[0][0]
            else:
                wait = sum(wh * w for wh, w in entries) / total_weight
            region_series[region].append((ts, wait))

        self._lookup = {
            region: sorted(series, key=lambda x: x[0])
            for region, series in region_series.items()
        }
        self._lookup_built = True

    def get_wait_hours(
        self,
        region: str,
        timestamp: datetime,
        gpu_type: Optional[str] = None,
    ) -> float:
        """Return estimated wait hours for region at timestamp.

        Uses the last known state at or before `timestamp` (leakage-safe:
        the optimizer never sees future congestion during backtesting).

        Returns 0.0 if no queue data exists for this region.
        """
        if not self._lookup_built:
            self._build_lookup()

        series = self._lookup.get(region)
        if not series:
            return 0.0

        # Normalise to UTC-aware hour
        if timestamp.tzinfo is None:
            ts = timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = timestamp

        ts_floor = ts.replace(minute=0, second=0, microsecond=0)

        # Binary search for last entry ≤ ts_floor
        lo, hi = 0, len(series) - 1
        result = 0.0
        while lo <= hi:
            mid = (lo + hi) // 2
            if series[mid][0] <= ts_floor:
                result = series[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1

        return result

    def to_dict_lookup(self) -> dict[str, dict[datetime, float]]:
        """Return {region: {timestamp: est_wait_hours}} mirror of price_data format.

        This format plugs directly into the objective function and scheduler.
        All timestamps are UTC-hour floored datetimes.
        """
        if not self._lookup_built:
            self._build_lookup()

        return {
            region: {ts: wait for ts, wait in series}
            for region, series in self._lookup.items()
        }

    @property
    def regions(self) -> list[str]:
        """List of regions with queue data."""
        if not self._lookup_built:
            self._build_lookup()
        return list(self._lookup.keys())

    @property
    def n_records(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------ #
    # Fixture generation
    # ------------------------------------------------------------------ #

    @classmethod
    def generate_fixture(
        cls,
        regions: list[str],
        start: datetime,
        end: datetime,
        gpu_types: Optional[list[str]] = None,
        seed: int = 42,
        base_wait_hours: dict[str, float] | None = None,
    ) -> "QueueProvider":
        """Generate realistic synthetic queue state for testing and demos.

        Models typical GPU cluster congestion patterns:
        - Morning peak (8–12 UTC): queue builds as jobs submitted overnight start
        - Business hours (12–20 UTC): highest congestion
        - Off-peak (20–8 UTC): low congestion

        Args:
            regions: List of region identifiers.
            start: Start datetime (UTC).
            end: End datetime (UTC, exclusive).
            gpu_types: GPU types per region. If None uses ["a100"].
            seed: Random seed for reproducibility.
            base_wait_hours: Per-region baseline wait hours. Defaults vary
                by region position in the list (first region gets lowest base).
        """
        rng = random.Random(seed)
        gpu_types = gpu_types or ["a100"]

        if base_wait_hours is None:
            # Assign different baseline congestion levels per region
            bases = [0.5, 1.5, 2.5, 3.5]
            base_wait_hours = {r: bases[i % len(bases)] for i, r in enumerate(regions)}

        records = []
        current = start.replace(minute=0, second=0, microsecond=0)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        while current < end:
            hour_utc = current.hour
            # Congestion multiplier: high during business hours (12-20 UTC)
            if 12 <= hour_utc < 20:
                multiplier = 2.5 + rng.uniform(-0.3, 0.3)
            elif 8 <= hour_utc < 12:
                multiplier = 1.5 + rng.uniform(-0.2, 0.2)
            else:
                multiplier = 0.4 + rng.uniform(-0.1, 0.1)

            for region in regions:
                base = base_wait_hours[region]
                est_wait = max(0.0, base * multiplier + rng.gauss(0, 0.1 * base))
                queue_depth = max(0, int(est_wait * 10 + rng.gauss(0, 2)))
                avail_gpus = max(0, int(80 - queue_depth * 3 + rng.gauss(0, 5)))

                for gpu_type in gpu_types:
                    records.append(QueueState(
                        timestamp=current,
                        region=region,
                        cluster_id=f"{region}-cluster-1",
                        gpu_type=gpu_type,
                        available_gpus=avail_gpus,
                        queue_depth_jobs=queue_depth,
                        est_wait_hours=round(est_wait, 2),
                    ))

            current += timedelta(hours=1)

        provider = cls(_records=records)
        return provider

    def to_dataframe(self) -> pd.DataFrame:
        """Export records as a DataFrame with the canonical queue CSV schema."""
        if not self._records:
            return pd.DataFrame(columns=list(_COLS))
        rows = [
            {
                "timestamp": r.timestamp.isoformat(),
                "region": r.region,
                "cluster_id": r.cluster_id,
                "gpu_type": r.gpu_type,
                "available_gpus": r.available_gpus,
                "queue_depth_jobs": r.queue_depth_jobs,
                "est_wait_hours": r.est_wait_hours,
            }
            for r in self._records
        ]
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> None:
        """Save queue state to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"Saved {self.n_records} queue records to {path}")
