"""Idempotent schema migration runner for Aurelius persistence.

Brings a Postgres (production) or SQLite (dev/test) database up to the current
schema. Safe to run repeatedly and on every deploy.

Two layers are applied, in order:

1. **ORM metadata (source of truth).** ``TimeSeriesStore`` defines every managed
   table via SQLAlchemy ``MetaData``. ``MetaData.create_all()`` creates any
   missing table / index and is a no-op for tables that already exist. This
   covers all persistence tables: energy_prices, carbon_intensity,
   benchmark_runs, decision_events, realized_outcomes, telemetry_snapshots,
   model_registry, promotion_decisions, learning_runs.

2. **SQL migration files (Postgres-only extras).** The ``*.sql`` files in
   ``migrations/`` add things the ORM cannot express portably — TimescaleDB
   hypertables and extra covering indexes. They are written ``IF NOT EXISTS`` /
   wrapped in ``DO`` blocks so they degrade gracefully on plain Postgres and are
   skipped entirely on SQLite (their ``TIMESTAMPTZ`` / ``DO $$`` syntax is
   Postgres-specific).

Usage:
    DATABASE_URL=postgresql://u:p@host/db python -m aurelius.database.migrate
    DATABASE_URL=sqlite:///./aurelius.db   python -m aurelius.database.migrate

Exit codes: 0 on success (or cleanly skipped when DATABASE_URL is unset), 1 on
a connection / SQL error.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

from .store import _META

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _apply_sql_files(engine) -> list[str]:
    """Apply the Postgres-only ``*.sql`` migration files in lexical order.

    Skipped for non-Postgres dialects (the files use Postgres-specific syntax;
    the ORM ``create_all`` already covers the portable schema). Each file is
    executed in its own transaction so one failure does not abort the rest.
    """
    if engine.dialect.name != "postgresql":
        logger.info("Dialect %s: skipping Postgres-only .sql files.", engine.dialect.name)
        return []

    applied: list[str] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text()
        with engine.begin() as conn:
            conn.execute(text(sql))
        applied.append(path.name)
        logger.info("Applied SQL migration: %s", path.name)
    return applied


def run_migrations(url: str | None = None) -> dict:
    """Create / update all persistence tables. Returns a summary dict.

    Args:
        url: SQLAlchemy URL. Defaults to the ``DATABASE_URL`` env var.

    Returns:
        {"status": "ok"|"skipped"|"error", ...}. ``skipped`` when no URL is set.
    """
    url = url if url is not None else os.environ.get("DATABASE_URL", "")
    if not url:
        logger.warning("DATABASE_URL not set — nothing to migrate (no-op).")
        return {"status": "skipped", "reason": "no_database_url"}

    is_sqlite = url.startswith("sqlite")
    engine = create_engine(url, pool_pre_ping=not is_sqlite)
    try:
        _META.create_all(engine)
        tables = sorted(_META.tables.keys())
        logger.info("ORM create_all complete (%d tables): %s", len(tables), ", ".join(tables))
        sql_applied = _apply_sql_files(engine)
        return {
            "status": "ok",
            "dialect": engine.dialect.name,
            "tables": tables,
            "sql_files_applied": sql_applied,
        }
    finally:
        engine.dispose()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = run_migrations()
    except Exception as exc:  # connection / SQL failure
        logger.error("Migration failed: %s", exc)
        return 1
    if result["status"] == "skipped":
        # No DB configured — not an error (no-op deploys / local dev).
        print("migrate: skipped (DATABASE_URL not set)")
        return 0
    print(
        f"migrate: ok ({result['dialect']}) — {len(result['tables'])} tables ensured; "
        f"sql files: {result['sql_files_applied'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
