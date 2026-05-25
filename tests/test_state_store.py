"""Tests for aurelius/state/store.py — leakage-safe ClusterStateStore."""

import pytest
from datetime import datetime, timezone, timedelta

from aurelius.state.models import ClusterState
from aurelius.state.store import ClusterStateStore

UTC = timezone.utc
T0 = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)
T2 = T0 + timedelta(minutes=10)
T3 = T0 + timedelta(minutes=15)
T4 = T0 + timedelta(minutes=20)


def make_state(ts: datetime, cluster_id: str = "test") -> ClusterState:
    return ClusterState(timestamp=ts, cluster_id=cluster_id)


class TestClusterStateStore:
    def test_empty_store_returns_none(self):
        store = ClusterStateStore()
        assert store.latest_at_or_before(T0) is None
        assert store.latest() is None
        assert store.oldest() is None

    def test_single_state_lookup_exact(self):
        store = ClusterStateStore()
        s = make_state(T1)
        store.add(s)
        result = store.latest_at_or_before(T1)
        assert result is s

    def test_single_state_lookup_after(self):
        store = ClusterStateStore()
        s = make_state(T1)
        store.add(s)
        result = store.latest_at_or_before(T2)
        assert result is s

    def test_single_state_lookup_before_returns_none(self):
        store = ClusterStateStore()
        s = make_state(T2)
        store.add(s)
        result = store.latest_at_or_before(T1)
        assert result is None

    def test_leakage_guarantee_no_future_state(self):
        store = ClusterStateStore()
        past = make_state(T1)
        future = make_state(T3)
        store.add(past)
        store.add(future)

        # Querying at T2 must NOT return the T3 state
        result = store.latest_at_or_before(T2)
        assert result is past
        assert result.timestamp == T1

    def test_multiple_states_latest_wins(self):
        store = ClusterStateStore()
        s1 = make_state(T1)
        s2 = make_state(T2)
        s3 = make_state(T3)
        store.add(s3)
        store.add(s1)
        store.add(s2)

        result = store.latest_at_or_before(T3)
        assert result is s3

        result2 = store.latest_at_or_before(T2)
        assert result2 is s2

    def test_out_of_order_insertion(self):
        store = ClusterStateStore()
        s1 = make_state(T1)
        s2 = make_state(T2)
        s3 = make_state(T3)
        # Insert out of order
        store.add(s3)
        store.add(s1)
        store.add(s2)
        assert len(store) == 3
        # Store should still be sorted
        assert store.oldest().timestamp == T1
        assert store.latest().timestamp == T3

    def test_all_at_or_before(self):
        store = ClusterStateStore()
        for ts in [T1, T2, T3, T4]:
            store.add(make_state(ts))
        result = store.all_at_or_before(T2)
        assert len(result) == 2
        assert result[0].timestamp == T1
        assert result[1].timestamp == T2

    def test_states_in_window(self):
        store = ClusterStateStore()
        for ts in [T0, T1, T2, T3, T4]:
            store.add(make_state(ts))
        result = store.states_in_window(T1, T3)
        assert len(result) == 3
        assert result[0].timestamp == T1
        assert result[-1].timestamp == T3

    def test_len_and_clear(self):
        store = ClusterStateStore()
        store.add(make_state(T1))
        store.add(make_state(T2))
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert store.latest() is None

    def test_time_range(self):
        store = ClusterStateStore()
        assert store.time_range() is None
        store.add(make_state(T1))
        store.add(make_state(T3))
        lo, hi = store.time_range()
        assert lo == T1
        assert hi == T3

    def test_snapshot_count(self):
        store = ClusterStateStore()
        assert store.snapshot_count() == 0
        store.add(make_state(T1))
        assert store.snapshot_count() == 1

    def test_exact_boundary_inclusive(self):
        store = ClusterStateStore()
        s = make_state(T2)
        store.add(s)
        assert store.latest_at_or_before(T2) is s
        assert store.latest_at_or_before(T2 - timedelta(seconds=1)) is None

    def test_duplicate_timestamps(self):
        store = ClusterStateStore()
        s1 = make_state(T1, cluster_id="a")
        s2 = make_state(T1, cluster_id="b")
        store.add(s1)
        store.add(s2)
        assert len(store) == 2
        result = store.latest_at_or_before(T1)
        assert result is not None

    def test_add_preserves_order_after_many_inserts(self):
        store = ClusterStateStore()
        timestamps = [T0 + timedelta(minutes=i) for i in range(20)]
        import random
        shuffled = list(timestamps)
        random.shuffle(shuffled)
        for ts in shuffled:
            store.add(make_state(ts))
        assert len(store) == 20
        ts_list = [s.timestamp for s in store.all_at_or_before(T0 + timedelta(hours=1))]
        assert ts_list == sorted(ts_list)
