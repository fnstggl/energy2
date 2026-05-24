"""Append-only in-memory ClusterState snapshot store.

Design rules:
- Append-only: snapshots are never mutated or removed during a session.
- Leakage-safe lookup: last_known_at_or_before(t) returns only snapshots
  whose timestamp <= t, preventing future-data leakage in replay/backtest.
- Thread-safe for read-heavy workloads (list append is GIL-protected in CPython;
  for write-concurrent use add explicit locking).
- Hashable metadata: each snapshot carries a sha256-based snapshot_id for
  benchmark reproducibility comparisons.
"""

from __future__ import annotations

import bisect
from datetime import datetime
from typing import Optional

from aurelius.state.models import ClusterState


class StateStore:
    """Append-only in-memory store for ClusterState snapshots.

    Snapshots are kept sorted by timestamp for efficient leakage-safe lookup.

    Usage::

        store = StateStore()
        store.append(snapshot)
        latest = store.last_known_at_or_before(query_time)
    """

    def __init__(self) -> None:
        self._snapshots: list[ClusterState] = []
        # Parallel list of timestamps for bisect-based O(log n) lookup.
        self._timestamps: list[datetime] = []

    def append(self, snapshot: ClusterState) -> None:
        """Add a new snapshot, maintaining timestamp order.

        Snapshots are inserted in sorted order so that multiple sources
        writing out-of-order snapshots still result in a consistent timeline.
        Duplicate timestamps are allowed (multiple connectors may snapshot at
        the same wall-clock second).
        """
        idx = bisect.bisect_right(self._timestamps, snapshot.timestamp)
        self._snapshots.insert(idx, snapshot)
        self._timestamps.insert(idx, snapshot.timestamp)

    def last_known_at_or_before(self, t: datetime) -> Optional[ClusterState]:
        """Return the most recent snapshot with timestamp <= t.

        Returns None if no snapshot exists at or before t.

        This is the leakage-safe lookup pattern used throughout Aurelius:
        the optimizer/classifier must never see a snapshot whose timestamp
        is after the decision point.
        """
        if not self._timestamps:
            return None
        idx = bisect.bisect_right(self._timestamps, t)
        if idx == 0:
            return None
        return self._snapshots[idx - 1]

    def snapshots_in_range(
        self,
        start: datetime,
        end: datetime,
        *,
        inclusive_end: bool = True,
    ) -> list[ClusterState]:
        """Return all snapshots with timestamp in [start, end].

        Args:
            start: earliest timestamp (inclusive)
            end: latest timestamp (inclusive by default)
            inclusive_end: if False, exclude snapshots at exactly end
        """
        lo = bisect.bisect_left(self._timestamps, start)
        if inclusive_end:
            hi = bisect.bisect_right(self._timestamps, end)
        else:
            hi = bisect.bisect_left(self._timestamps, end)
        return list(self._snapshots[lo:hi])

    def latest(self) -> Optional[ClusterState]:
        """Return the most recent snapshot, or None if empty."""
        return self._snapshots[-1] if self._snapshots else None

    def earliest(self) -> Optional[ClusterState]:
        """Return the oldest snapshot, or None if empty."""
        return self._snapshots[0] if self._snapshots else None

    def __len__(self) -> int:
        return len(self._snapshots)

    def __bool__(self) -> bool:
        return bool(self._snapshots)

    @property
    def snapshot_ids(self) -> list[str]:
        """Snapshot IDs in timestamp order, for benchmark metadata."""
        return [s.snapshot_id for s in self._snapshots]

    def clear(self) -> None:
        """Remove all snapshots (e.g. between test cases)."""
        self._snapshots.clear()
        self._timestamps.clear()
