"""Aurelius database persistence layer.

Provides two database backends:
1. TimeSeriesStore — SQLAlchemy-backed time-series storage for energy prices,
   carbon intensity, and benchmark results. Uses Postgres (production) or
   SQLite (dev/test). Falls back to a no-op mode when DATABASE_URL is absent.

2. SupabaseClient / get_db — legacy Supabase integration (backward compat).

Quickstart (TimeSeriesStore):
    store = TimeSeriesStore()                     # reads DATABASE_URL from env
    store = TimeSeriesStore("sqlite:///:memory:") # for testing
    store = TimeSeriesStore("postgresql://u:p@host/aurelius")

    n = store.upsert_prices(price_df)
    df = store.get_prices("us-west", start, end)
"""

from .store import TimeSeriesStore
from .supabase_client import SupabaseClient, get_db

__all__ = ["TimeSeriesStore", "SupabaseClient", "get_db"]
