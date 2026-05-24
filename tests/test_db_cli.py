"""Tests for the 'aurelius db' CLI subcommand group.

Tests use SQLite in-memory / tmp-file stores so no Postgres is required.
All operations that depend on DATABASE_URL are exercised via explicit SQLite
URLs to ensure the commands work regardless of environment configuration.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from aurelius.cli import cmd_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(db_command, **kwargs):
    """Build a Namespace that mimics parsed CLI args for 'db' subcommands."""
    base = Namespace(db_command=db_command, prices_command=None)
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def _sqlite_url(tmp_path):
    return f"sqlite:///{tmp_path}/test.db"


# ---------------------------------------------------------------------------
# db status — disabled (no DATABASE_URL)
# ---------------------------------------------------------------------------

class TestDbStatusDisabled:
    def test_prints_disabled_when_no_url(self, capsys):
        with patch.dict("os.environ", {"DATABASE_URL": ""}):
            cmd_db(_make_args("status"))
        out = capsys.readouterr().out
        assert "DISABLED" in out

    def test_status_disabled_no_error(self, capsys):
        with patch.dict("os.environ", {"DATABASE_URL": ""}):
            cmd_db(_make_args("status"))
        out = capsys.readouterr().out
        assert "DATABASE_URL" in out


# ---------------------------------------------------------------------------
# db status — enabled (SQLite)
# ---------------------------------------------------------------------------

class TestDbStatusEnabled:
    def test_status_ok_with_sqlite(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(_make_args("status"))
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "sqlite" in captured.out
        assert "energy_prices" in captured.out

    def test_status_shows_zero_counts_on_empty_db(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(_make_args("status"))
        out = capsys.readouterr().out
        # All tables should show 0
        for line in out.splitlines():
            if ":" in line and any(c.isdigit() for c in line):
                # Row count lines end with ": N"
                count_str = line.strip().split(":")[-1].strip().replace(",", "")
                if count_str.isdigit():
                    assert int(count_str) == 0

    def test_status_shows_nonzero_after_upsert(self, tmp_path, capsys):
        from aurelius.database.store import TimeSeriesStore
        url = _sqlite_url(tmp_path)
        store = TimeSeriesStore(url)
        df = pd.DataFrame({
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "region": ["us-west"],
            "price_per_mwh": [42.0],
            "currency": ["USD"],
            "source": ["test"],
        })
        store.upsert_prices(df)
        store.close()

        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(_make_args("status"))
        out = capsys.readouterr().out
        assert "energy_prices" in out
        assert "1" in out  # at least the count shows up


# ---------------------------------------------------------------------------
# db migrate
# ---------------------------------------------------------------------------

class TestDbMigrate:
    def test_migrate_ok_with_sqlite(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(_make_args("migrate"))
        out = capsys.readouterr().out
        assert "ok" in out
        assert "sqlite" in out
        assert "9" in out  # 9 tables

    def test_migrate_skipped_when_no_url(self, capsys):
        with patch.dict("os.environ", {"DATABASE_URL": ""}):
            cmd_db(_make_args("migrate"))
        out = capsys.readouterr().out
        assert "skipped" in out

    def test_migrate_idempotent(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(_make_args("migrate"))
            cmd_db(_make_args("migrate"))
        # Running twice should not raise and should both succeed
        out = capsys.readouterr().out
        assert out.count("ok") >= 1


# ---------------------------------------------------------------------------
# db prices show
# ---------------------------------------------------------------------------

class TestDbPricesShow:
    def test_show_empty_no_region(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        args = _make_args("prices", prices_command="show", region=None,
                          limit=20, start=None, end=None)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(args)
        out = capsys.readouterr().out
        # No region → show summary with count
        assert "total price rows" in out
        assert "--region" in out

    def test_show_empty_with_region(self, tmp_path, capsys):
        url = _sqlite_url(tmp_path)
        args = _make_args("prices", prices_command="show", region="us-west",
                          limit=20, start=None, end=None)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(args)
        out = capsys.readouterr().out
        assert "no rows found" in out
        assert "us-west" in out

    def test_show_rows_with_region(self, tmp_path, capsys):
        from aurelius.database.store import TimeSeriesStore
        url = _sqlite_url(tmp_path)
        store = TimeSeriesStore(url)
        df = pd.DataFrame({
            "timestamp": [datetime(2026, 1, 1, h, tzinfo=timezone.utc) for h in range(5)],
            "region": ["us-west"] * 5,
            "price_per_mwh": [40.0 + h for h in range(5)],
            "currency": ["USD"] * 5,
            "source": ["test"] * 5,
        })
        store.upsert_prices(df)
        store.close()

        args = _make_args("prices", prices_command="show", region="us-west",
                          limit=20, start=None, end=None)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(args)
        out = capsys.readouterr().out
        assert "5" in out  # 5 rows found
        assert "us-west" in out

    def test_show_limit_respected(self, tmp_path, capsys):
        from aurelius.database.store import TimeSeriesStore
        url = _sqlite_url(tmp_path)
        store = TimeSeriesStore(url)
        df = pd.DataFrame({
            "timestamp": [datetime(2026, 1, 1, h, tzinfo=timezone.utc) for h in range(24)],
            "region": ["us-west"] * 24,
            "price_per_mwh": [30.0 + h for h in range(24)],
            "currency": ["USD"] * 24,
            "source": ["test"] * 24,
        })
        store.upsert_prices(df)
        store.close()

        args = _make_args("prices", prices_command="show", region="us-west",
                          limit=5, start=None, end=None)
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(args)
        out = capsys.readouterr().out
        assert "24" in out  # total count
        assert "showing last 5" in out

    def test_show_disabled(self, capsys):
        args = _make_args("prices", prices_command="show", region="us-west",
                          limit=20, start=None, end=None)
        with patch.dict("os.environ", {"DATABASE_URL": ""}):
            cmd_db(args)
        out = capsys.readouterr().out
        assert "disabled" in out

    def test_show_unknown_prices_command_exits(self, tmp_path):
        url = _sqlite_url(tmp_path)
        args = _make_args("prices", prices_command="unknown")
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_db(args)
        assert exc_info.value.code == 1

    def test_show_start_end_filter(self, tmp_path, capsys):
        from aurelius.database.store import TimeSeriesStore
        url = _sqlite_url(tmp_path)
        store = TimeSeriesStore(url)
        df = pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 1, tzinfo=timezone.utc),
                datetime(2026, 3, 1, tzinfo=timezone.utc),
            ],
            "region": ["us-west"] * 3,
            "price_per_mwh": [30.0, 31.0, 32.0],
            "currency": ["USD"] * 3,
            "source": ["test"] * 3,
        })
        store.upsert_prices(df)
        store.close()

        args = _make_args("prices", prices_command="show", region="us-west",
                          limit=20, start="2026-01-15", end="2026-02-15")
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            cmd_db(args)
        out = capsys.readouterr().out
        # Only Feb row should match
        assert "1" in out


# ---------------------------------------------------------------------------
# db — unknown subcommand exits
# ---------------------------------------------------------------------------

class TestDbUnknownSubcommand:
    def test_unknown_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_db(_make_args("unknown_cmd"))
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Usage" in out
