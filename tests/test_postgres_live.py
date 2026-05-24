"""Optional live-Postgres tests for TimeSeriesStore.

These exercise the SAME store methods as ``test_database_store.py`` but against a
real Postgres server, so the Postgres dialect path (``ON CONFLICT DO NOTHING``
upserts, ``TIMESTAMPTZ`` handling, the SQL migration files, TimescaleDB DO
blocks) is verified end-to-end — not just the SQLite path.

They are SKIPPED unless a Postgres URL is provided, so the default test run
(SQLite only) is unaffected and CI stays hermetic. To run them:

    # dedicated test DB (preferred)
    TEST_DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/aurelius_test \
        python -m pytest tests/test_postgres_live.py -v

    # or reuse DATABASE_URL if it points at Postgres
    DATABASE_URL=postgresql://... python -m pytest tests/test_postgres_live.py -v

The test DB is left clean: every test runs inside its own unique run_id /
source / customer namespace and asserts on counts scoped to that namespace, so
runs never collide and the suite is re-runnable without a reset.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from aurelius.database import TimeSeriesStore
from aurelius.database.migrate import run_migrations


def _pg_url() -> str | None:
    """Return a Postgres URL from TEST_DATABASE_URL or DATABASE_URL, else None."""
    for var in ("TEST_DATABASE_URL", "DATABASE_URL"):
        url = os.environ.get(var, "")
        if url and ("postgresql" in url or url.startswith("postgres://")):
            return url
    return None


pytestmark = pytest.mark.skipif(
    _pg_url() is None,
    reason="No Postgres URL (set TEST_DATABASE_URL or a postgres DATABASE_URL).",
)


@pytest.fixture(scope="module")
def pg_store():
    url = _pg_url()
    assert url is not None
    store = TimeSeriesStore(url)
    assert store.enabled, "could not connect to Postgres"
    assert store.dialect == "postgresql"
    yield store
    store.close()


@pytest.fixture
def ns():
    """A unique namespace so re-runs against a shared DB never collide."""
    tag = uuid.uuid4().hex[:12]
    return {
        "run_id": f"pgtest-{tag}",
        "source": f"pgtest_{tag}",
        "customer_id": f"cust_{tag}",
        "pilot_id": f"pilot_{tag}",
    }


def _decision(run_id, job_id, source, realized=False):
    from aurelius.shadow.models import DecisionRecord

    dt = datetime(2026, 1, 13, tzinfo=timezone.utc)
    rec = DecisionRecord(
        run_id=run_id, job_id=job_id, workload_type="training", decision_time=dt,
        scheduled_region="us-west", scheduled_start=dt, scheduled_end=dt + timedelta(hours=4),
        scheduled_runtime_h=4.0, forecast_da_price_p50=30.0, forecast_da_price_p90=36.0,
        predicted_energy_cost=10.0, baseline_region="us-east", baseline_start=dt,
        baseline_energy_cost=15.0, predicted_savings_pct=33.3, power_kw=100.0, gpu_count=8,
    )
    rec.data_source = source
    if realized:
        rec.realized_rt_price = 32.0
        rec.realized_energy_cost = 11.0
        rec.realized_baseline_cost = 15.0
        rec.realized_savings_pct = 26.7
        rec.sla_met = True
    return rec


def test_migrations_run_against_postgres(pg_store):
    """The migration runner brings a real Postgres up to schema, idempotently."""
    url = _pg_url()
    result = run_migrations(url)
    assert result["status"] == "ok"
    assert result["dialect"] == "postgresql"
    assert "decision_events" in result["tables"]
    assert "realized_outcomes" in result["tables"]
    # Re-running is a no-op (idempotent).
    again = run_migrations(url)
    assert again["status"] == "ok"


def test_price_upsert_idempotent_on_postgres(pg_store, ns):
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [t0 + timedelta(hours=i) for i in range(5)],
        "region": ["us-west"] * 5,
        "price_per_mwh": [30.0 + i for i in range(5)],
        "currency": ["USD"] * 5,
        "source": [ns["source"]] * 5,
    })
    assert pg_store.upsert_prices(df) == 5
    # Re-upsert: ON CONFLICT DO NOTHING → zero new rows.
    assert pg_store.upsert_prices(df) == 0
    got = pg_store.get_prices("us-west", t0, t0 + timedelta(hours=4), source=ns["source"])
    assert len(got) == 5


def test_shadow_decision_write_and_dedup_on_postgres(pg_store, ns):
    rec = _decision(ns["run_id"], "j1", ns["source"])
    assert pg_store.record_decisions([rec], ns["customer_id"], ns["pilot_id"], "hash123") == 1
    # Replay the same run/job → deduped.
    assert pg_store.record_decisions([rec], ns["customer_id"], ns["pilot_id"]) == 0
    rows = pg_store.get_decisions(run_id=ns["run_id"])
    assert len(rows) == 1
    assert rows[0]["data_source_hash"] == "hash123"
    assert rows[0]["customer_id"] == ns["customer_id"]


def test_realized_outcome_write_read_dedup_on_postgres(pg_store, ns):
    rec = _decision(ns["run_id"], "j1", ns["source"], realized=True)
    assert pg_store.record_realized_outcomes([rec], ns["customer_id"], ns["pilot_id"]) == 1
    assert pg_store.record_realized_outcomes([rec], ns["customer_id"], ns["pilot_id"]) == 0
    # Read back the realized outcome the learning loop consumes.
    rows = pg_store.get_realized_outcomes(customer_id=ns["customer_id"])
    assert len(rows) == 1
    assert abs(rows[0]["realized_savings_pct"] - 26.7) < 0.01
    assert rows[0]["sla_met"] == 1


def test_benchmark_and_learning_run_on_postgres(pg_store, ns):
    pg_store.save_benchmark_run(
        run_id=ns["run_id"], forecaster="ml_quantile", region_combo="caiso_pjm_ercot",
        workload="training", savings_vs_cpo=15.0, folds=5,
    )
    hist = pg_store.get_benchmark_history(region_combo="caiso_pjm_ercot")
    assert any(h["run_id"] == ns["run_id"] for h in hist)

    assert pg_store.start_learning_run(ns["run_id"], scope=ns["customer_id"], pilot_id=ns["pilot_id"])
    assert pg_store.finish_learning_run(ns["run_id"], state="completed", summary={"ok": 1})


def test_model_registry_promote_rollback_on_postgres(pg_store, ns):
    scope = ns["customer_id"]
    m1, m2 = f"{ns['run_id']}-m1", f"{ns['run_id']}-m2"
    pg_store.register_model(m1, "v1", f"file:///{m1}", scope=scope, pilot_id=ns["pilot_id"])
    assert pg_store.promote_model(m1)
    pg_store.register_model(m2, "v2", f"file:///{m2}", scope=scope, pilot_id=ns["pilot_id"])
    assert pg_store.promote_model(m2)
    assert pg_store.get_active_model("price", scope, ns["pilot_id"])["model_id"] == m2
    restored = pg_store.rollback_active("price", scope, ns["pilot_id"])
    assert restored["model_id"] == m1
