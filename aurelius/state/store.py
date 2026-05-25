"""ClusterStateStore — leakage-safe time-indexed state storage.

The store holds an ordered series of ClusterState snapshots and provides
lookups that are guaranteed not to return states from after a requested time.
This prevents any future-data leakage in backtesting or simulation.
"""

from __future__ import annotations

import bisect
from datetime import datetime
from typing import Optional

from .models import ClusterState


class ClusterStateStore:
    """An ordered, leakage-safe store of ClusterState snapshots.

    Snapshots are kept sorted by timestamp. All lookups guarantee that
    only states with timestamp <= query_time are returned.

    Usage::

        store = ClusterStateStore()
        store.add(state1)
        store.add(state2)
        latest = store.latest_at_or_before(query_time)
    """

    def __init__(self) -> None:
        self._states: list[ClusterState] = []
        self._timestamps: list[datetime] = []

    def add(self, state: ClusterState) -> None:
        """Insert a ClusterState snapshot in sorted order by timestamp."""
        ts = state.timestamp
        idx = bisect.bisect_right(self._timestamps, ts)
        self._timestamps.insert(idx, ts)
        self._states.insert(idx, state)

    def latest_at_or_before(self, query_time: datetime) -> Optional[ClusterState]:
        """Return the most recent state with timestamp <= query_time.

        Returns None if no state exists at or before query_time.
        This guarantee prevents future-data leakage.
        """
        if not self._timestamps:
            return None
        idx = bisect.bisect_right(self._timestamps, query_time)
        if idx == 0:
            return None
        return self._states[idx - 1]

    def all_at_or_before(self, query_time: datetime) -> list[ClusterState]:
        """Return all states with timestamp <= query_time, oldest first."""
        if not self._timestamps:
            return []
        idx = bisect.bisect_right(self._timestamps, query_time)
        return list(self._states[:idx])

    def states_in_window(self, start: datetime, end: datetime) -> list[ClusterState]:
        """Return states where start <= timestamp <= end, oldest first."""
        lo = bisect.bisect_left(self._timestamps, start)
        hi = bisect.bisect_right(self._timestamps, end)
        return list(self._states[lo:hi])

    def latest(self) -> Optional[ClusterState]:
        """Return the most recent state regardless of time."""
        if not self._states:
            return None
        return self._states[-1]

    def oldest(self) -> Optional[ClusterState]:
        """Return the oldest state."""
        if not self._states:
            return None
        return self._states[0]

    def __len__(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        """Remove all stored states."""
        self._states.clear()
        self._timestamps.clear()

    def snapshot_count(self) -> int:
        return len(self._states)

    def time_range(self) -> Optional[tuple[datetime, datetime]]:
        """Return (oldest_ts, newest_ts) or None if empty."""
        if not self._timestamps:
            return None
        return self._timestamps[0], self._timestamps[-1]
