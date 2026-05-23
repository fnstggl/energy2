"""SQLAlchemy-backed time-series persistence for Aurelius.

Supports Postgres (production) and SQLite (tests / single-node pilots).
Falls back to a no-op mode when DATABASE_URL is not set — all write methods
return 0 / empty results without crashing.

Tables managed by this module (created on first connection):
  energy_prices        — hourly DA/RT price rows per region + source
  carbon_intensity     — hourly marginal emissions per region + source
  benchmark_runs       — archived benchmark result rows (one row per workload cell)
  decision_events      — optimizer scheduling decisions (append-only, scoped by
                         customer_id + pilot_id + run_id; reproducible via
                         data_source_hash)
  realized_outcomes    — realized RT prices/costs/savings per decision (the
                         predicted-vs-realized feedback the learning loop reads)
  telemetry_snapshots  — queue / GPU-DCGM telemetry snapshots (generic payload)

Environment variables:
  DATABASE_URL       — SQLAlchemy-compatible URL (required for any DB use)
                       Postgres example: postgresql://user:pass@host/aurelius
                       SQLite example:   sqlite:///./aurelius.db
                       Test:             sqlite:///:memory:
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — dialect-agnostic (DateTime(timezone=True) works on both SQLite
# and Postgres; SQLite stores it as TEXT/NUMERIC transparently)
# ---------------------------------------------------------------------------

_META = MetaData()

_ENERGY_PRICES = Table(
    "energy_prices",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("region", String(64), nullable=False),
    Column("price_per_mwh", Float, nullable=False),
    Column("currency", String(8), nullable=False, default="USD"),
    Column("source", String(64), nullable=False),
    Column("source_granularity", String(32), nullable=False, default="hourly"),
    Column("fetched_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("timestamp", "region", "source", name="uq_energy_prices_ts_region_source"),
)

_CARBON_INTENSITY = Table(
    "carbon_intensity",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("region", String(64), nullable=False),
    Column("gco2_per_kwh", Float, nullable=False),
    Column("source", String(64), nullable=False),
    Column("source_granularity", String(32), nullable=False, default="hourly"),
    Column("fetched_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("timestamp", "region", "source", name="uq_carbon_intensity_ts_region_source"),
)

_BENCHMARK_RUNS = Table(
    "benchmark_runs",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("run_at", DateTime(timezone=True), nullable=False),
    Column("forecaster", String(64), nullable=False),
    Column("region_combo", String(128), nullable=False),
    Column("workload", String(64), nullable=False),
    Column("savings_vs_cpo", Float, nullable=False),
    Column("folds", Integer, nullable=False),
    Column("miss_pct", Float, nullable=False, default=0.0),
    Column("meta_json", Text, nullable=True),
)

# ---------------------------------------------------------------------------
# Data-moat event tables (append-only). These capture the structured
# operational record that future models / offline policy learning depend on.
# Every row is scoped by customer_id + pilot_id + run_id so pilots are
# isolated and a historical decision can be reproduced exactly.
# ---------------------------------------------------------------------------

# One optimizer scheduling decision (the prediction made at decision time).
_DECISION_EVENTS = Table(
    "decision_events",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String(64), nullable=False, default="unknown"),
    Column("pilot_id", String(64), nullable=False, default="unknown"),
    Column("run_id", String(64), nullable=False),
    Column("job_id", String(128), nullable=False),
    Column("workload_type", String(64), nullable=False),
    Column("decision_time", DateTime(timezone=True), nullable=False),
    Column("scheduled_region", String(64), nullable=False),
    Column("scheduled_start", DateTime(timezone=True), nullable=True),
    Column("scheduled_runtime_h", Float, nullable=True),
    Column("forecast_da_price_p50", Float, nullable=True),
    Column("forecast_da_price_p90", Float, nullable=True),
    Column("predicted_energy_cost", Float, nullable=True),
    Column("baseline_region", String(64), nullable=True),
    Column("baseline_energy_cost", Float, nullable=True),
    Column("predicted_savings_pct", Float, nullable=True),
    Column("sla_class", String(32), nullable=True),
    Column("gate_status", String(16), nullable=True),       # passed / filtered / null
    Column("gate_reason", Text, nullable=True),             # safety-gate reason code
    Column("forecaster_version", String(64), nullable=True),
    Column("optimizer_version", String(64), nullable=True),
    Column("data_source", String(64), nullable=True),
    Column("data_source_hash", String(64), nullable=True),  # reproducibility hash
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("meta_json", Text, nullable=True),
    UniqueConstraint("run_id", "job_id", name="uq_decision_events_run_job"),
)

# The realized outcome for a decision (filled in after the job window passes).
_REALIZED_OUTCOMES = Table(
    "realized_outcomes",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String(64), nullable=False, default="unknown"),
    Column("pilot_id", String(64), nullable=False, default="unknown"),
    Column("run_id", String(64), nullable=False),
    Column("job_id", String(128), nullable=False),
    Column("workload_type", String(64), nullable=True),
    Column("predicted_savings_pct", Float, nullable=True),
    Column("realized_rt_price", Float, nullable=True),
    Column("realized_energy_cost", Float, nullable=True),
    Column("realized_baseline_cost", Float, nullable=True),
    Column("realized_savings_pct", Float, nullable=True),
    Column("sla_met", Integer, nullable=True),              # 1/0/null
    Column("realization_note", Text, nullable=True),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "job_id", name="uq_realized_outcomes_run_job"),
)

# Generic telemetry snapshot table (queue depth, GPU/DCGM health, etc.).
# kind discriminates the payload; payload_json carries the kind-specific fields.
_TELEMETRY_SNAPSHOTS = Table(
    "telemetry_snapshots",
    _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String(64), nullable=False, default="unknown"),
    Column("pilot_id", String(64), nullable=False, default="unknown"),
    Column("kind", String(32), nullable=False),             # 'queue' | 'gpu_dcgm'
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("region", String(64), nullable=True),
    Column("node_id", String(128), nullable=False, default=""),
    Column("source", String(64), nullable=False, default="unknown"),
    Column("payload_json", Text, nullable=True),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "kind", "source", "region", "node_id", "timestamp",
        name="uq_telemetry_kind_src_region_node_ts",
    ),
)


# ---------------------------------------------------------------------------
# Dialect-agnostic upsert helper
# ---------------------------------------------------------------------------

def _upsert_ignore(engine: Engine, table: Table, rows: list[dict]) -> int:
    """Insert rows into table, silently ignoring duplicates.

    Uses INSERT OR IGNORE for SQLite and INSERT ... ON CONFLICT DO NOTHING
    for Postgres. For other dialects falls back to individual try/except.

    Returns the number of rows actually inserted.
    """
    if not rows:
        return 0

    dialect = engine.dialect.name
    inserted = 0

    with engine.begin() as conn:
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(table).on_conflict_do_nothing()
            result = conn.execute(stmt, rows)
            inserted = result.rowcount if result.rowcount >= 0 else len(rows)

        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(table).prefix_with("OR IGNORE")
            result = conn.execute(stmt, rows)
            inserted = result.rowcount if result.rowcount >= 0 else len(rows)

        else:
            # Generic fallback: insert one-by-one, skip IntegrityError
            from sqlalchemy import insert as generic_insert
            from sqlalchemy.exc import IntegrityError

            for row in rows:
                try:
                    conn.execute(generic_insert(table), [row])
                    inserted += 1
                except IntegrityError:
                    conn.rollback()

    return inserted


# ---------------------------------------------------------------------------
# TimeSeriesStore
# ---------------------------------------------------------------------------

class TimeSeriesStore:
    """Persistent time-series store for Aurelius prices, carbon, and benchmarks.

    Connects to Postgres (production) or SQLite (tests / file-based pilots).
    When DATABASE_URL is not set and no url is provided, operates in no-op
    mode: all writes are silently skipped, all reads return empty DataFrames.

    Thread-safe via SQLAlchemy connection pool (Postgres) or serialised
    access (SQLite).

    Args:
        url: SQLAlchemy database URL. Defaults to DATABASE_URL env var.

    Example:
        store = TimeSeriesStore()                        # from env
        store = TimeSeriesStore("sqlite:///:memory:")   # tests
        store = TimeSeriesStore("postgresql://u:p@h/db")
    """

    def __init__(self, url: Optional[str] = None) -> None:
        self._url = url if url is not None else os.environ.get("DATABASE_URL", "")
        self._engine: Optional[Engine] = None
        self._enabled = False

        if not self._url:
            logger.debug(
                "TimeSeriesStore: DATABASE_URL not set. Operating in no-op mode. "
                "Set DATABASE_URL to enable Postgres/SQLite persistence."
            )
            return

        try:
            # SQLite doesn't support connection-pool settings
            is_sqlite = self._url.startswith("sqlite")
            engine_kwargs: dict = {"pool_pre_ping": not is_sqlite}
            if not is_sqlite:
                engine_kwargs["pool_size"] = 5
                engine_kwargs["max_overflow"] = 10

            self._engine = create_engine(self._url, **engine_kwargs)
            # Create tables on first connection (idempotent)
            _META.create_all(self._engine)
            self._enabled = True
            logger.info(
                "TimeSeriesStore connected. Dialect: %s",
                self._engine.dialect.name,
            )
        except Exception as exc:
            logger.warning(
                "TimeSeriesStore: connection failed (%s). Operating in no-op mode.", exc
            )
            self._engine = None
            self._enabled = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when connected to a real database."""
        return self._enabled

    @property
    def dialect(self) -> str:
        """SQLAlchemy dialect name ('postgresql', 'sqlite', etc.) or 'disabled'."""
        if self._engine is None:
            return "disabled"
        return self._engine.dialect.name

    # ------------------------------------------------------------------
    # Energy prices
    # ------------------------------------------------------------------

    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Bulk-upsert price rows from a canonical price DataFrame.

        Expected columns (from aurelius.ingestion.grid_apis.base.normalize_price_df):
          timestamp, region, price_per_mwh, [currency], [source], [source_granularity]

        Duplicate rows (same timestamp × region × source) are silently skipped.

        Returns:
            Number of rows actually inserted (0 when disabled or df is empty).
        """
        if not self._enabled or df.empty:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "region": str(row["region"]),
                    "price_per_mwh": float(row["price_per_mwh"]),
                    "currency": str(row.get("currency", "USD")),
                    "source": str(row.get("source", "unknown")),
                    "source_granularity": str(row.get("source_granularity", "hourly")),
                    "fetched_at": now,
                }
            )

        assert self._engine is not None
        return _upsert_ignore(self._engine, _ENERGY_PRICES, rows)

    def get_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return hourly price rows for region in [start, end] (inclusive).

        Args:
            region: Region identifier (e.g. 'us-west').
            start:  UTC start datetime (inclusive).
            end:    UTC end datetime (inclusive).
            source: Optional filter by source name.

        Returns:
            DataFrame with columns [timestamp, region, price_per_mwh, currency, source].
            Empty DataFrame when disabled or no rows found.
        """
        if not self._enabled:
            return _empty_price_df()

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        assert self._engine is not None
        t = _ENERGY_PRICES
        cond = and_(
            t.c.region == region,
            t.c.timestamp >= start_utc,
            t.c.timestamp <= end_utc,
        )
        if source is not None:
            cond = and_(cond, t.c.source == source)

        stmt = (
            select(t.c.timestamp, t.c.region, t.c.price_per_mwh, t.c.currency, t.c.source)
            .where(cond)
            .order_by(t.c.timestamp)
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()

        if not rows:
            return _empty_price_df()

        out = pd.DataFrame(rows, columns=["timestamp", "region", "price_per_mwh", "currency", "source"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out

    # ------------------------------------------------------------------
    # Carbon intensity
    # ------------------------------------------------------------------

    def upsert_carbon(self, df: pd.DataFrame) -> int:
        """Bulk-upsert carbon intensity rows.

        Expected columns: timestamp, region, gco2_per_kwh, [source], [source_granularity]

        Returns:
            Number of rows inserted (0 when disabled or df is empty).
        """
        if not self._enabled or df.empty:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "region": str(row["region"]),
                    "gco2_per_kwh": float(row["gco2_per_kwh"]),
                    "source": str(row.get("source", "unknown")),
                    "source_granularity": str(row.get("source_granularity", "hourly")),
                    "fetched_at": now,
                }
            )

        assert self._engine is not None
        return _upsert_ignore(self._engine, _CARBON_INTENSITY, rows)

    def get_carbon(
        self,
        region: str,
        start: datetime,
        end: datetime,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return hourly carbon intensity rows for region in [start, end].

        Returns:
            DataFrame with columns [timestamp, region, gco2_per_kwh, source].
            Empty DataFrame when disabled or no rows found.
        """
        if not self._enabled:
            return _empty_carbon_df()

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        assert self._engine is not None
        t = _CARBON_INTENSITY
        cond = and_(
            t.c.region == region,
            t.c.timestamp >= start_utc,
            t.c.timestamp <= end_utc,
        )
        if source is not None:
            cond = and_(cond, t.c.source == source)

        stmt = (
            select(t.c.timestamp, t.c.region, t.c.gco2_per_kwh, t.c.source)
            .where(cond)
            .order_by(t.c.timestamp)
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()

        if not rows:
            return _empty_carbon_df()

        out = pd.DataFrame(rows, columns=["timestamp", "region", "gco2_per_kwh", "source"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out

    # ------------------------------------------------------------------
    # Benchmark run archival
    # ------------------------------------------------------------------

    def save_benchmark_run(
        self,
        run_id: str,
        forecaster: str,
        region_combo: str,
        workload: str,
        savings_vs_cpo: float,
        folds: int,
        miss_pct: float = 0.0,
        meta: Optional[dict] = None,
    ) -> None:
        """Archive a single benchmark result cell.

        Overwrites existing row for the same (run_id, region_combo, workload).

        Args:
            run_id:         Benchmark run identifier (e.g. '20260523T200730Z').
            forecaster:     Forecaster name (e.g. 'ml_quantile_recovery').
            region_combo:   Region combo (e.g. 'caiso_pjm_ercot_da_rt').
            workload:       Workload type (e.g. 'training').
            savings_vs_cpo: Savings vs current_price_only as a percent (e.g. 25.5).
            folds:          Number of evaluation folds.
            miss_pct:       Fraction of missing price hours.
            meta:           Optional extra metadata (stored as JSON).
        """
        if not self._enabled:
            return

        now = datetime.now(timezone.utc)
        row = {
            "run_id": run_id,
            "run_at": now,
            "forecaster": forecaster,
            "region_combo": region_combo,
            "workload": workload,
            "savings_vs_cpo": float(savings_vs_cpo),
            "folds": int(folds),
            "miss_pct": float(miss_pct),
            "meta_json": json.dumps(meta) if meta else None,
        }

        assert self._engine is not None
        # Delete existing row for this (run_id, region_combo, workload) then insert
        with self._engine.begin() as conn:
            from sqlalchemy import delete

            conn.execute(
                delete(_BENCHMARK_RUNS).where(
                    and_(
                        _BENCHMARK_RUNS.c.run_id == run_id,
                        _BENCHMARK_RUNS.c.region_combo == region_combo,
                        _BENCHMARK_RUNS.c.workload == workload,
                    )
                )
            )
            conn.execute(_BENCHMARK_RUNS.insert(), [row])

    def get_benchmark_history(
        self,
        region_combo: Optional[str] = None,
        workload: Optional[str] = None,
        forecaster: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent benchmark results, newest first.

        Args:
            region_combo: Optional filter by region combo.
            workload:     Optional filter by workload type.
            forecaster:   Optional filter by forecaster name.
            limit:        Maximum rows to return.

        Returns:
            List of dicts with keys: run_id, run_at, forecaster, region_combo,
            workload, savings_vs_cpo, folds, miss_pct, meta.
        """
        if not self._enabled:
            return []

        assert self._engine is not None
        t = _BENCHMARK_RUNS
        conds = []
        if region_combo is not None:
            conds.append(t.c.region_combo == region_combo)
        if workload is not None:
            conds.append(t.c.workload == workload)
        if forecaster is not None:
            conds.append(t.c.forecaster == forecaster)

        stmt = (
            select(
                t.c.run_id,
                t.c.run_at,
                t.c.forecaster,
                t.c.region_combo,
                t.c.workload,
                t.c.savings_vs_cpo,
                t.c.folds,
                t.c.miss_pct,
                t.c.meta_json,
            )
            .order_by(t.c.run_at.desc())
            .limit(limit)
        )
        if conds:
            from sqlalchemy import and_ as sa_and

            stmt = stmt.where(sa_and(*conds))

        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "run_id": r.run_id,
                    "run_at": r.run_at,
                    "forecaster": r.forecaster,
                    "region_combo": r.region_combo,
                    "workload": r.workload,
                    "savings_vs_cpo": r.savings_vs_cpo,
                    "folds": r.folds,
                    "miss_pct": r.miss_pct,
                    "meta": json.loads(r.meta_json) if r.meta_json else None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Decision events (data-moat: optimizer decisions)
    # ------------------------------------------------------------------

    def record_decisions(
        self,
        records: list,
        customer_id: str = "unknown",
        pilot_id: str = "unknown",
        data_source_hash: Optional[str] = None,
    ) -> int:
        """Append optimizer scheduling decisions to durable storage.

        Args:
            records: list of shadow DecisionRecord (or dicts / objects exposing
                     the same fields). Duplicate (run_id, job_id) rows are
                     silently skipped (append-only, idempotent on replays).
            customer_id: Pilot customer identifier (isolates pilots).
            pilot_id:    Pilot/engagement identifier.
            data_source_hash: Optional hash of the input price/data files for
                     exact reproduction of a historical decision.

        Returns:
            Number of rows actually inserted (0 when disabled or empty).
        """
        if not self._enabled or not records:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for rec in records:
            d = _as_dict(rec)
            rows.append(
                {
                    "customer_id": customer_id,
                    "pilot_id": pilot_id,
                    "run_id": str(d.get("run_id", "")),
                    "job_id": str(d.get("job_id", "")),
                    "workload_type": str(d.get("workload_type", "unknown")),
                    "decision_time": _coerce_ts(d.get("decision_time")) or now,
                    "scheduled_region": str(d.get("scheduled_region", "")),
                    "scheduled_start": _coerce_ts(d.get("scheduled_start")),
                    "scheduled_runtime_h": _opt_float(d.get("scheduled_runtime_h")),
                    "forecast_da_price_p50": _opt_float(d.get("forecast_da_price_p50")),
                    "forecast_da_price_p90": _opt_float(d.get("forecast_da_price_p90")),
                    "predicted_energy_cost": _opt_float(d.get("predicted_energy_cost")),
                    "baseline_region": _opt_str(d.get("baseline_region")),
                    "baseline_energy_cost": _opt_float(d.get("baseline_energy_cost")),
                    "predicted_savings_pct": _opt_float(d.get("predicted_savings_pct")),
                    "sla_class": _opt_str(d.get("sla_class")),
                    "gate_status": _opt_str(d.get("gate_status")),
                    "gate_reason": _opt_str(d.get("gate_reason")),
                    "forecaster_version": _opt_str(d.get("forecaster_version")),
                    "optimizer_version": _opt_str(d.get("optimizer_version")),
                    "data_source": _opt_str(d.get("data_source")),
                    "data_source_hash": data_source_hash,
                    "recorded_at": now,
                    "meta_json": None,
                }
            )

        assert self._engine is not None
        return _upsert_ignore(self._engine, _DECISION_EVENTS, rows)

    def get_decisions(
        self,
        run_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        pilot_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return recorded decisions (newest first), optionally filtered."""
        if not self._enabled:
            return []
        assert self._engine is not None
        t = _DECISION_EVENTS
        conds = []
        if run_id is not None:
            conds.append(t.c.run_id == run_id)
        if customer_id is not None:
            conds.append(t.c.customer_id == customer_id)
        if pilot_id is not None:
            conds.append(t.c.pilot_id == pilot_id)
        stmt = select(t).order_by(t.c.recorded_at.desc()).limit(limit)
        if conds:
            stmt = stmt.where(and_(*conds))
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Realized outcomes (data-moat: predicted vs realized feedback)
    # ------------------------------------------------------------------

    def record_realized_outcomes(
        self,
        records: list,
        customer_id: str = "unknown",
        pilot_id: str = "unknown",
    ) -> int:
        """Append realized outcomes for decisions whose windows have passed.

        Only records with a non-None realized_savings_pct are persisted.
        Duplicate (run_id, job_id) rows are skipped (append-only).

        Returns:
            Number of rows inserted (0 when disabled or none realized).
        """
        if not self._enabled or not records:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for rec in records:
            d = _as_dict(rec)
            if d.get("realized_savings_pct") is None:
                continue
            sla_met = d.get("sla_met")
            rows.append(
                {
                    "customer_id": customer_id,
                    "pilot_id": pilot_id,
                    "run_id": str(d.get("run_id", "")),
                    "job_id": str(d.get("job_id", "")),
                    "workload_type": _opt_str(d.get("workload_type")),
                    "predicted_savings_pct": _opt_float(d.get("predicted_savings_pct")),
                    "realized_rt_price": _opt_float(d.get("realized_rt_price")),
                    "realized_energy_cost": _opt_float(d.get("realized_energy_cost")),
                    "realized_baseline_cost": _opt_float(d.get("realized_baseline_cost")),
                    "realized_savings_pct": _opt_float(d.get("realized_savings_pct")),
                    "sla_met": None if sla_met is None else int(bool(sla_met)),
                    "realization_note": _opt_str(d.get("realization_note")),
                    "recorded_at": now,
                }
            )

        if not rows:
            return 0
        assert self._engine is not None
        return _upsert_ignore(self._engine, _REALIZED_OUTCOMES, rows)

    def get_realized_outcomes(
        self,
        customer_id: Optional[str] = None,
        pilot_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5000,
    ) -> list[dict]:
        """Return realized outcomes (newest first), optionally filtered.

        The daily learning loop reads from here to track realized savings and
        forecast error over time rather than re-deriving them from JSONL files.
        """
        if not self._enabled:
            return []
        assert self._engine is not None
        t = _REALIZED_OUTCOMES
        conds = []
        if customer_id is not None:
            conds.append(t.c.customer_id == customer_id)
        if pilot_id is not None:
            conds.append(t.c.pilot_id == pilot_id)
        if run_id is not None:
            conds.append(t.c.run_id == run_id)
        stmt = select(t).order_by(t.c.recorded_at.desc()).limit(limit)
        if conds:
            stmt = stmt.where(and_(*conds))
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Telemetry snapshots (data-moat: queue + GPU/DCGM)
    # ------------------------------------------------------------------

    def record_telemetry(
        self,
        kind: str,
        df: pd.DataFrame,
        source: str,
        customer_id: str = "unknown",
        pilot_id: str = "unknown",
    ) -> int:
        """Append telemetry snapshots (queue depth, GPU/DCGM health, etc.).

        Args:
            kind:   'queue' or 'gpu_dcgm' (free-form discriminator).
            df:     DataFrame with at least a 'timestamp' column. Optional
                    'region' and 'node_id' columns are promoted to columns;
                    all remaining columns are stored in payload_json.
            source: Telemetry source label (e.g. 'prometheus_dcgm').

        Returns:
            Number of rows inserted (0 when disabled or empty).
        """
        if not self._enabled or df is None or df.empty:
            return 0

        now = datetime.now(timezone.utc)
        promoted = {"timestamp", "region", "node_id"}
        rows = []
        for _, row in df.iterrows():
            ts = _coerce_ts(row.get("timestamp"))
            if ts is None:
                continue
            payload = {
                k: (None if pd.isna(v) else v)
                for k, v in row.items()
                if k not in promoted
            }
            rows.append(
                {
                    "customer_id": customer_id,
                    "pilot_id": pilot_id,
                    "kind": str(kind),
                    "timestamp": ts,
                    "region": _opt_str(row.get("region")),
                    "node_id": str(row.get("node_id", "") or ""),
                    "source": str(source),
                    "payload_json": json.dumps(payload, default=str),
                    "recorded_at": now,
                }
            )

        assert self._engine is not None
        return _upsert_ignore(self._engine, _TELEMETRY_SNAPSHOTS, rows)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def row_counts(self) -> dict[str, int]:
        """Return row counts for each managed table (for health checks)."""
        if not self._enabled:
            return {"enabled": False}

        assert self._engine is not None
        counts: dict[str, int] = {}
        with self._engine.connect() as conn:
            for tbl in [
                _ENERGY_PRICES,
                _CARBON_INTENSITY,
                _BENCHMARK_RUNS,
                _DECISION_EVENTS,
                _REALIZED_OUTCOMES,
                _TELEMETRY_SNAPSHOTS,
            ]:
                stmt = select(func.count()).select_from(tbl)
                counts[tbl.name] = conn.execute(stmt).scalar() or 0
        return counts

    def close(self) -> None:
        """Dispose the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._enabled = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_dict(rec) -> dict:
    """Normalize a record (dataclass with to_dict, plain object, or dict) to a dict."""
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    return {k: v for k, v in vars(rec).items()}


def _coerce_ts(value) -> Optional[datetime]:
    """Coerce a value (datetime / ISO string / pandas Timestamp) to a UTC datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        try:
            value = pd.Timestamp(value).to_pydatetime()
        except (ValueError, TypeError):
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _opt_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _empty_price_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "region", "price_per_mwh", "currency", "source"])


def _empty_carbon_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh", "source"])
