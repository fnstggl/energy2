# Aurelius Data Moat / Continuous Learning Storage Architecture

**Continuous-learning status: PARTIAL (not COMPLETE, not BLOCKED).**
The full persistence surface is live on production Postgres and the loop reads
realized outcomes back from Postgres, but model **promotion** is still driven by
held-out *forecast accuracy*, not by *realized customer savings* (gap **G1′**).
Per the completion bar, we therefore do **not** claim "data moat complete":
realized outcomes are read back and surfaced, but they do not yet drive the
promotion/comparison metric. See §6 and the gap table in §3a.

**Status:** Storage + model-registry + outcome-driven promotion implemented
(Postgres/SQLite via SQLAlchemy; artifacts via a pluggable object store).
Production is Postgres-backed on Railway (DATABASE_URL → the managed Postgres
service); migrations run on every deploy via `python -m aurelius.database.migrate`.
**Honest scope / positioning:** Aurelius is now *append-only operational
learning infrastructure with outcome tracking, a model registry, rollback, and
shadow-mode evaluation*. It is NOT a "fully autonomous learning system" or a
"complete data moat" — telemetry-driven learning and full feature
co-persistence remain open (see gaps below). Do not market beyond the
infrastructure that exists.

### Update log
- **2026-05-24 (Railway Postgres productionization):** Made DATABASE_URL-backed
  Postgres the production source of truth. Added a root `railway.json`
  (Dockerfile builder — fixes the Railpack "could not determine how to build"
  failure caused by the Python manifests living under `aurelius/` rather than
  repo root), an idempotent migration runner (`aurelius/database/migrate.py`,
  run on every deploy before the API boots), and an optional live-Postgres test
  suite (`tests/test_postgres_live.py`, skipped unless a Postgres
  `TEST_DATABASE_URL`/`DATABASE_URL` is set). Verified the full
  shadow-run → shadow-realize → daily-learning-loop flow end-to-end against a
  real Postgres server (decisions + realized outcomes written, realized
  outcomes read back by the loop, replays deduped). Closes G5 (a migration
  runner now exists; still not Alembic-versioned).
- **2026-05-23 (productionization):** Added the DB-backed model registry
  (`model_registry`), append-only `promotion_decisions`, learning-run lifecycle
  (`learning_runs`), a pluggable artifact store (`aurelius/storage/`), a
  single-host run lock, safety-gate execution + persistence in the shadow
  decision path, and an **honest candidate-vs-active promotion** that loads the
  persisted active model and compares it on a leakage-free holdout. Closes
  former gaps G1 (partly), G2, G4, and the locking/rollback items.

---

## 1. Purpose

Aurelius must not merely optimize once. To build a defensible advantage it must
record structured operational data from every shadow run, pilot, and (future)
production deployment, so that later models and offline policy learning can be
trained and evaluated against *real outcomes*.

This document describes the storage layer that captures that data.

---

## 2. Chosen architecture

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Structured time-series + decision/outcome events | **Postgres** (prod), **SQLite** (dev/single-node pilot) via a single **SQLAlchemy** abstraction (`aurelius/database/store.py`) | Deployment-portable: the same code runs on Supabase Postgres, AWS RDS, Neon, Timescale Cloud, or local Postgres/SQLite. No Supabase lock-in. |
| Large artifacts / model files | **Pluggable artifact store** (`aurelius/storage/artifacts.py`): local filesystem, or any S3-compatible store (AWS S3, R2, MinIO, Supabase Storage) selected by `ARTIFACT_STORE_URI`. Postgres stores only the `artifact_uri`. | Model binaries never live in Postgres or git. `file://` for local/dev, `s3://` (+ `AWS_ENDPOINT_URL`) for production. boto3 is required only for `s3://`. |
| Schemas / migrations / fixtures / sample traces / docs | **git repo** | Only non-sensitive scaffolding. Real customer/pilot data is NEVER committed. |

The store is **append-only** for event tables and **no-op safe**: when
`DATABASE_URL` is unset, every write returns 0 and every read returns empty —
nothing crashes, JSONL/CSV remain the source of record for that run.

```
DATABASE_URL=postgresql://user:pass@host/aurelius   # production
DATABASE_URL=sqlite:///./aurelius.db                # single-node pilot
DATABASE_URL=sqlite:///:memory:                     # tests
(unset)                                             # no-op (JSONL/CSV only)
```

---

## 3. Tables

Managed by `aurelius/database/store.py` (`TimeSeriesStore`), created idempotently
via SQLAlchemy `MetaData.create_all()` on first connection.

### Pre-existing (time-series)
| Table | Key | Notes |
|-------|-----|-------|
| `energy_prices` | (timestamp, region, source) unique | DA/RT prices |
| `carbon_intensity` | (timestamp, region, source) unique | marginal emissions |
| `benchmark_runs` | (run_id, region_combo, workload) | archived benchmark cells |

### New event tables (this work — the data moat backbone)
| Table | Unique key | Captures |
|-------|-----------|----------|
| `decision_events` | (run_id, job_id) | One optimizer scheduling decision: scheduled region/start/runtime, forecast p50/p90, predicted vs baseline cost, predicted savings, SLA class, **gate_status/gate_reason**, forecaster/optimizer version, **data_source_hash**, **customer_id/pilot_id** |
| `realized_outcomes` | (run_id, job_id) | Realized RT price/cost, realized savings, SLA met, linked to the decision by (run_id, job_id) — the predicted-vs-realized feedback |
| `telemetry_snapshots` | (kind, source, region, node_id, timestamp) | Generic queue / GPU-DCGM snapshots; `kind` discriminates, `payload_json` carries kind-specific fields |

### Model registry + learning lifecycle (this work)
| Table | Key | Captures |
|-------|-----|----------|
| `model_registry` | model_id unique | One trained model version: model_type, **scope (customer_id)**, pilot_id, version, status (`candidate`→`active`→`archived`/`rolled_back`), **artifact_uri**, **training_dataset_hash**, training_rows, eval_metrics, **parent_model_id** (lineage), run_id, promoted_at |
| `promotion_decisions` | append-only | Every promote / reject / rollback decision with primary_metric + candidate vs active values + reason |
| `learning_runs` | run_id unique | Run lifecycle (UUID, state started/completed/failed, summary) for idempotency + audit |

Every event/model row is scoped by `customer_id` + `pilot_id` (+ `run_id`), so
pilots are isolated and a historical decision can be reproduced exactly (given
the `training_dataset_hash` / `data_source_hash` of the inputs).

---

## 3a. Gap table vs the target 16-table surface

The mission listed 16 target tables as a *minimum persistence surface, not a
rigid schema*. Several targets are deliberately consolidated into broader,
event-style tables rather than duplicated as thin per-concept tables — this is
the simpler, safer schema and avoids writers that would otherwise be empty.
Audited 2026-05-24; **no working tables were renamed or duplicated.**

| # | Target entity | Existing table? | Store method? | Written by shadow/loop? | Read back for learning? | Gap | Action |
|---|---------------|-----------------|---------------|-------------------------|-------------------------|-----|--------|
| 1 | energy_prices | ✅ `energy_prices` | `upsert_prices`/`get_prices` | loop (`_persist_prices_to_db`) | ✅ (forecaster training data) | none | keep |
| 2 | carbon_intensity | ✅ `carbon_intensity` | `upsert_carbon`/`get_carbon` | available (no loop writer wired) | optional | minor | keep; document |
| 3 | weather_observations | ❌ | — | — | — | real gap | **document** — used at compute time from CSV + `data_source_hash`; per-decision feature co-persistence is open gap (see §6). Not adding an empty table. |
| 4 | queue_snapshots | ✅ via `telemetry_snapshots` (kind=`queue`) | `record_telemetry` | interface only | — | no live writer | keep consolidated |
| 5 | gpu_telemetry_snapshots | ✅ via `telemetry_snapshots` (kind=`gpu_dcgm`) | `record_telemetry` | interface only | — | no live writer | keep consolidated |
| 6 | workload_traces | ❌ | — | — | — | real gap | **document** — ingested from customer CSV at run time; `data_source_hash` on each decision links back to the trace. Not adding an empty table. |
| 7 | forecast_snapshots | ⚠ embedded in `decision_events` (p50/p90) | within `record_decisions` | shadow run | partial | no standalone table | keep embedded |
| 8 | optimizer_decisions | ✅ `decision_events` | `record_decisions`/`get_decisions` | shadow run | ✅ (offline pairs) | none | keep |
| 9 | baseline_decisions | ✅ `decision_events.baseline_*` | within `record_decisions` | shadow run | n/a | none | keep |
| 10 | safety_gate_decisions | ✅ `decision_events.gate_status/gate_reason` | within `record_decisions` | shadow run (fail-closed) | n/a | none | keep |
| 11 | realized_outcomes | ✅ `realized_outcomes` | `record_realized_outcomes`/`get_realized_outcomes` | shadow realize | ✅ (loop Step 5) | none | keep |
| 12 | benchmark_runs | ✅ `benchmark_runs` | `save_benchmark_run`/`get_benchmark_history` | loop + benchmark runner | ✅ | none | keep |
| 13 | model_versions | ✅ `model_registry` | `register_model`/`promote_model`/… | loop (model update) | ✅ (`get_active_model`) | none | keep |
| 14 | model_promotion_events | ✅ `promotion_decisions` | `record_promotion_decision`/`get_promotion_decisions` | loop + rollback | ✅ | none | keep |
| 15 | learning_loop_runs | ✅ `learning_runs` | `start_learning_run`/`finish_learning_run` | loop | n/a | none | keep |
| 16 | raw_artifact_index | ⚠ artifact store + `model_registry.artifact_uri` | `register_model(artifact_uri=…)` | loop | ✅ | no separate index | **document** — large binaries live in the pluggable artifact store (`ARTIFACT_STORE_URI`); the DB holds the `artifact_uri`. A separate raw index is unnecessary; object storage is the right home for blobs. |

**Net:** every customer/pilot-scoped table carries `customer_id`, `pilot_id`,
`run_id` (where applicable), `source`, a timestamp, and `data_source_hash`
(where applicable). The only genuine *missing* persistence is per-decision
weather/queue/GPU **feature co-persistence** and **workload-trace archival** —
both deliberately deferred to source CSV + `data_source_hash` (documented gap,
not silently dropped).

---

## 4. Required-entity coverage matrix

The brief listed 17 entities. Honest current state:

| # | Entity | Stored? | Where |
|---|--------|---------|-------|
| 1 | Raw market data pulls | Partial | `energy_prices` (normalized only; raw API dumps not archived) |
| 2 | Normalized energy prices | ✅ | `energy_prices` |
| 3 | Carbon intensity | ✅ | `carbon_intensity` |
| 4 | Weather/cooling | ❌ | used at compute time from CSV; not persisted |
| 5 | Queue snapshots | ✅ (interface) | `telemetry_snapshots` kind=`queue` (no live writer wired yet) |
| 6 | GPU/DCGM telemetry | ✅ (interface) | `telemetry_snapshots` kind=`gpu_dcgm` (no live writer wired yet) |
| 7 | Workload/job traces | ❌ | ingested from CSV at run time; not persisted as an entity |
| 8 | Forecast snapshots | Partial | p50/p90 stored *per decision* in `decision_events`; no standalone forecast table |
| 9 | Optimizer decisions | ✅ | `decision_events` (also JSONL) |
| 10 | Safety-gate decisions + reason codes | ✅ | The `QuantileSafetyGate` now runs in the shadow decision path (fail-closed) and `gate_status`/`gate_reason` are persisted on every `decision_events` row |
| 10b | Model versions + promotion lineage | ✅ | `model_registry` (status, artifact_uri, dataset hash, parent_model_id) + append-only `promotion_decisions` |
| 11 | Baseline decisions | ✅ | `decision_events.baseline_region/baseline_energy_cost` |
| 12 | Realized prices/costs | ✅ | `realized_outcomes` |
| 13 | Realized savings/losses | ✅ | `realized_outcomes` |
| 14 | SLA/deadline outcomes | Partial | `realized_outcomes.sla_met` (populated only if realizer sets it) |
| 15 | Migration/checkpoint outcomes | ❌ | not modeled |
| 16 | Model/optimizer version, data source hash | ✅ | `decision_events.forecaster_version/optimizer_version/data_source_hash` |
| 17 | Customer/pilot/run identifiers | ✅ | `customer_id/pilot_id/run_id` on all event tables |

### Per-entity audit dimensions
- **Append-only / auditable:** decision/outcome/telemetry tables are insert-only with a unique key (replays are idempotent, no silent overwrite). `benchmark_runs` is overwrite-on-cell (legacy).
- **Customer separation:** `customer_id` + `pilot_id` on every event row.
- **Future model training:** decision + realized tables give (features-at-decision, realized outcome) pairs. Sufficient to *start* offline evaluation; weather/queue/GPU features are not yet co-persisted per decision, so full feature reconstruction needs the source CSVs + `data_source_hash`.
- **Reproduce a historical decision exactly:** possible iff the input price files referenced by `data_source_hash` are retained. The hash is stored; the files themselves are the operator's responsibility (recommend object storage — Gap G4).

---

## 5. Data flow (implemented)

```
shadow run  ──► QuantileSafetyGate (decision path, fail-closed)
            ──► DecisionRecord[] (gate_status/gate_reason) ──► JSONL (always)
                                 └─► decision_events (if DATABASE_URL set)
                                       customer_id, pilot_id, data_source_hash

(7–14 days later)
shadow realize ──► realized DecisionRecord[] ──► JSONL (always)
                                            └──► realized_outcomes (if DB set)

daily_learning_loop.py (single-host locked; run UUID + lifecycle):
  train candidate (train split)
  load ACTIVE model from registry + artifact store
  evaluate BOTH on leakage-free holdout (ForecastEvaluator)
  promote candidate iff it wins  ──► artifact_store.put(model.joblib)
                                 ──► model_registry (candidate→active, archive old)
                                 ──► promotion_decisions (promote/reject)
  read_realized_outcomes_summary() ──► realized savings + |forecast error| feedback
  rollback (--rollback) ──► restore previous active, log decision
```

CLI:
```bash
# Persist pilot-scoped decisions (+ safety-gate status) to the store
DATABASE_URL=postgresql://... ARTIFACT_STORE_URI=s3://bucket/aurelius \
python -m aurelius.cli shadow run \
  --price-file data/q12026_3region_dam.csv --regions us-west,us-east,us-south \
  --jobs-file customer_trace.csv --forecaster ml_quantile \
  --customer-id acme --pilot-id pilot-q1 --output-dir reports/shadow/

DATABASE_URL=postgresql://... python -m aurelius.cli shadow realize \
  --decisions-file reports/shadow/decisions_*.jsonl \
  --rt-price-file rt_settlement.csv --customer-id acme --pilot-id pilot-q1

# Daily model update (train → compare vs active → promote if better)
DATABASE_URL=postgresql://... ARTIFACT_STORE_URI=file://./data/artifacts \
python scripts/daily_learning_loop.py --customer-id acme --pilot-id pilot-q1

# Roll the active model back to the previous version
DATABASE_URL=postgresql://... python scripts/daily_learning_loop.py \
  --rollback --customer-id acme --pilot-id pilot-q1
```

---

## 5a. Production deployment (Railway) + migrations

**Builder.** A root `railway.json` pins the Docker builder to `docker/Dockerfile`.
Without it, Railway's Railpack auto-detector fails with *"could not determine how
to build the app"* because the Python manifests (`requirements.txt`,
`pyproject.toml`) live under `aurelius/`, not at repo root. The Dockerfile
installs `sqlalchemy` + `psycopg2-binary`, so the Postgres path is available in
the deployed image.

**DATABASE_URL.** The app reads `os.environ["DATABASE_URL"]`. On Railway it must
be set on the `energy2` service (e.g. a reference to the managed Postgres
service, `${{Postgres-w5kk.DATABASE_URL}}`, or the service's internal connection
URL `postgresql://…@postgres-*.railway.internal:5432/railway`). The
`*.railway.internal` host is only resolvable **inside** Railway's private
network — it is intentionally unreachable from laptops / CI, so production DB
verification must run from within the Railway environment.

**Migrations on deploy.** `railway.json`'s `startCommand` runs
`python -m aurelius.database.migrate` before launching uvicorn. The runner:
1. applies `MetaData.create_all()` (the ORM schema — the source of truth for all
   9 persistence tables), then
2. applies the Postgres-only `migrations/*.sql` (TimescaleDB hypertables + extra
   covering indexes; skipped on SQLite, and the hypertable `DO` blocks no-op on
   plain Postgres without the TimescaleDB extension).

It is idempotent (safe to re-run every deploy) and a no-op when `DATABASE_URL`
is unset. Run it anywhere:

```bash
DATABASE_URL=postgresql://… python -m aurelius.database.migrate   # production / Railway
DATABASE_URL=sqlite:///./aurelius.db python -m aurelius.database.migrate  # local
```

> **TimescaleDB note:** because `create_all()` runs first and creates the price
> tables with an `id`-only primary key, enabling TimescaleDB hypertables on a
> fresh DB requires running the `*.sql` files *before* `create_all` (so the
> composite `(id, timestamp)` PK exists). Railway's managed Postgres is plain
> Postgres (no TimescaleDB), so the hypertable `DO` blocks are skipped and this
> ordering is moot in production today.

**End-to-end verification (run against a real Postgres):**
```bash
DATABASE_URL=postgresql://…  # a reachable Postgres (prod-from-inside-Railway, or a local PG)
python -m aurelius.database.migrate
python -m aurelius.cli shadow run    --price-file <da.csv> --num-jobs 30 \
    --customer-id acme --pilot-id p1 --output-dir reports/shadow/
python -m aurelius.cli shadow realize --decisions-file reports/shadow/decisions_*.jsonl \
    --rt-price-file <rt.csv> --customer-id acme --pilot-id p1
python scripts/daily_learning_loop.py --no-fetch --customer-id acme --pilot-id p1
python -c "from aurelius.database import TimeSeriesStore as T; s=T(); print(s.row_counts()); s.close()"
TEST_DATABASE_URL=postgresql://… python -m pytest tests/test_postgres_live.py -v
```

---

## 6. What is real vs aspirational

**Real today:**
- Append-only, customer-isolated persistence of decisions, realized outcomes,
  and telemetry on portable Postgres/SQLite.
- Shadow mode writes every decision (with safety-gate status + reason) and,
  later, every realized outcome.
- **Model registry with real candidate-vs-active promotion:** the loop trains a
  candidate on a train split, **loads the persisted active model**, evaluates
  BOTH on a leakage-free holdout window (`ForecastEvaluator`), and promotes only
  if the candidate genuinely wins. No stale-scalar comparison.
- **Rollback:** registry status transitions + `rollback_active()` + `--rollback`
  restore the previous active model; promote/reject/rollback are all logged.
- **Artifact store abstraction** (local / S3-compatible) keeps binaries out of
  Postgres and git; the DB holds only the `artifact_uri`.
- **Run locking + lifecycle:** a single-host file lock prevents overlapping
  cron/Railway runs; `learning_runs` records a UUID + state for an audit trail.
- Reproducibility metadata (versions + `training_dataset_hash`/`data_source_hash`).

**NOT real yet (do not over-claim):**
- **G1′ — Promotion uses held-out *forecast accuracy* (MAE), not yet *realized
  customer savings*.** Realized outcomes are collected and read back, but the
  promotion criterion is leakage-free holdout accuracy, not the customer's
  realized $ savings. Feeding realized savings into the promotion metric is the
  remaining step for a fully outcome-driven moat.
- **G3 — Telemetry writers not wired.** Queue/GPU snapshot tables exist + are
  tested, but no live ingestion path writes to them yet.
- **G5 (partly closed) — migration runner exists, not yet Alembic-versioned.**
  `python -m aurelius.database.migrate` now applies the ORM schema + Postgres
  `*.sql` extras idempotently and runs on every Railway deploy. A *versioned*
  migration tool (Alembic) with up/down revisions and a documented retention/
  audit policy is still required for enterprise procurement.
- **Multi-host locking.** The run lock is single-host (fcntl); multi-host
  deployments should add a Postgres advisory lock.
- **Decision-time feature co-persistence.** Weather/queue/GPU features used at
  decision time are not yet stored per decision (source CSV + hash needed for
  exact replay).

Correct positioning: *append-only operational learning infrastructure with a
model registry, rollback, outcome tracking, and shadow-mode evaluation.* A fully
outcome-driven moat requires closing **G1′** (realized customer savings drive
promotion).

---

## 7. Recommendations (priority order)

1. **Close G1′:** incorporate realized customer savings (from
   `realized_outcomes`) into the promotion metric once enough pilot outcomes
   exist, keeping holdout accuracy as a guardrail; auto-rollback if a promoted
   model later underperforms on realized data.
2. **Wire telemetry ingestion** (queue/DCGM) to `telemetry_snapshots`.
3. **Add Alembic migrations** and a documented retention/audit policy.
4. **Add a Postgres advisory lock** for multi-host deployments.
5. **Co-persist decision-time features** for exact offline reconstruction.
