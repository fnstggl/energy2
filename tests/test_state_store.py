"""Tests for aurelius/state/store.py.

Proves:
- Append-only behavior
- Leakage-safe last_known_at_or_before() lookup
- Returns None when no snapshot exists at or before query time
- Snapshots inserted out of order are stored in timestamp order
- snapshots_in_range() works correctly
- latest() and earliest() work correctly
- len() and bool() work correctly
- clear() empties the store
- snapshot_ids returns ordered list
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aurelius.state.models import ClusterState, Provenance
from aurelius.state.store import StateStore

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _prov() -> Provenance:
    return Provenance(source="test", fetched_at=T0, confidence="high")


def _snapshot(ts: datetime, *, partial: bool = False) -> ClusterState:
    return ClusterState(timestamp=ts, provenance=_prov(), is_partial=partial)


# ---------------------------------------------------------------------------

class TestStateStore:
    def test_empty_store_returns_none(self):
        store = StateStore()
        result = store.last_known_at_or_before(T0)
        assert result is None

    def test_empty_len_and_bool(self):
        store = StateStore()
        assert len(store) == 0
        assert not store

    def test_single_snapshot_returned(self):
        store = StateStore()
        s = _snapshot(T0)
        store.append(s)
        assert store.last_known_at_or_before(T0) is s
        assert store.last_known_at_or_before(T0 + timedelta(hours=1)) is s

    def test_snapshot_not_returned_before_its_time(self):
        store = StateStore()
        store.append(_snapshot(T0))
        result = store.last_known_at_or_before(T0 - timedelta(seconds=1))
        assert result is None

    def test_leakage_safe_lookup(self):
        """The returned snapshot must have timestamp <= query time (core invariant)."""
        store = StateStore()
        s1 = _snapshot(T0)
        s2 = _snapshot(T0 + timedelta(minutes=5))
        s3 = _snapshot(T0 + timedelta(minutes=10))
        store.append(s1)
        store.append(s2)
        store.append(s3)

        # Query at T0+7min: must return s2 (T0+5), not s3 (T0+10)
        result = store.last_known_at_or_before(T0 + timedelta(minutes=7))
        assert result is s2
        assert result.timestamp <= T0 + timedelta(minutes=7)

    def test_exact_timestamp_match_returned(self):
        store = StateStore()
        s = _snapshot(T0 + timedelta(minutes=5))
        store.append(s)
        result = store.last_known_at_or_before(T0 + timedelta(minutes=5))
        assert result is s

    def test_out_of_order_insert_sorted_correctly(self):
        """Snapshots appended out of chronological order must be sorted."""
        store = StateStore()
        s_late = _snapshot(T0 + timedelta(hours=2))
        s_early = _snapshot(T0)
        s_mid = _snapshot(T0 + timedelta(hours=1))

        store.append(s_late)
        store.append(s_early)
        store.append(s_mid)

        # earliest/latest must reflect sorted order
        assert store.earliest() is s_early
        assert store.latest() is s_late

        # Lookup at T0+90min must return s_mid (T0+1h), not s_early
        result = store.last_known_at_or_before(T0 + timedelta(minutes=90))
        assert result is s_mid

    def test_snapshots_in_range_inclusive(self):
        store = StateStore()
        s0 = _snapshot(T0)
        s1 = _snapshot(T0 + timedelta(hours=1))
        s2 = _snapshot(T0 + timedelta(hours=2))
        s3 = _snapshot(T0 + timedelta(hours=3))
        for s in (s0, s1, s2, s3):
            store.append(s)

        result = store.snapshots_in_range(
            T0 + timedelta(hours=1),
            T0 + timedelta(hours=2),
        )
        assert len(result) == 2
        assert s1 in result
        assert s2 in result
        assert s0 not in result
        assert s3 not in result

    def test_snapshots_in_range_exclusive_end(self):
        store = StateStore()
        s1 = _snapshot(T0)
        s2 = _snapshot(T0 + timedelta(hours=1))
        store.append(s1)
        store.append(s2)

        result = store.snapshots_in_range(T0, T0 + timedelta(hours=1), inclusive_end=False)
        assert result == [s1]

    def test_multiple_snapshots_same_timestamp(self):
        """Duplicate timestamps are allowed (multiple connectors may race)."""
        store = StateStore()
        s1 = _snapshot(T0, partial=False)
        s2 = _snapshot(T0, partial=True)
        store.append(s1)
        store.append(s2)
        assert len(store) == 2
        # last_known_at_or_before returns one of the two (the later-inserted one)
        result = store.last_known_at_or_before(T0)
        assert result is not None

    def test_latest_returns_last(self):
        store = StateStore()
        s1 = _snapshot(T0)
        s2 = _snapshot(T0 + timedelta(hours=1))
        store.append(s1)
        store.append(s2)
        assert store.latest() is s2

    def test_earliest_returns_first(self):
        store = StateStore()
        s1 = _snapshot(T0)
        s2 = _snapshot(T0 + timedelta(hours=1))
        store.append(s1)
        store.append(s2)
        assert store.earliest() is s1

    def test_latest_none_on_empty(self):
        store = StateStore()
        assert store.latest() is None

    def test_earliest_none_on_empty(self):
        store = StateStore()
        assert store.earliest() is None

    def test_len_tracks_appends(self):
        store = StateStore()
        for i in range(5):
            store.append(_snapshot(T0 + timedelta(hours=i)))
        assert len(store) == 5

    def test_bool_true_after_append(self):
        store = StateStore()
        store.append(_snapshot(T0))
        assert bool(store)

    def test_clear_empties_store(self):
        store = StateStore()
        store.append(_snapshot(T0))
        store.clear()
        assert len(store) == 0
        assert not store
        assert store.last_known_at_or_before(T0) is None

    def test_snapshot_ids_ordered(self):
        store = StateStore()
        s1 = _snapshot(T0)
        s2 = _snapshot(T0 + timedelta(hours=1))
        store.append(s2)  # insert later-timestamp first
        store.append(s1)
        ids = store.snapshot_ids
        # Must be in timestamp order after sorting
        assert ids[0] == s1.snapshot_id
        assert ids[1] == s2.snapshot_id

    def test_leakage_boundary_exact(self):
        """Snapshot at exactly query_time is included (<=, not <)."""
        store = StateStore()
        query_time = T0 + timedelta(hours=3)
        s_at_query = _snapshot(query_time)
        s_after = _snapshot(query_time + timedelta(seconds=1))
        store.append(s_at_query)
        store.append(s_after)

        result = store.last_known_at_or_before(query_time)
        assert result is s_at_query

    def test_future_snapshot_never_leaks(self):
        """A snapshot in the future relative to query_time must never be returned."""
        store = StateStore()
        future = T0 + timedelta(hours=100)
        store.append(_snapshot(future))
        result = store.last_known_at_or_before(T0)
        assert result is None
