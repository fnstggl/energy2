# Aurelius Data Moat / Continuous Learning Storage Architecture

**Status:** Storage layer implemented (Postgres/SQLite via SQLAlchemy).
**Honest scope:** This is the *data-collection backbone* for a compounding
advantage. It is NOT yet a closed self-improving loop — see "What is real vs
aspirational" below. Do not market this as "data moat complete."

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
| Large raw artifacts / model files | **Object storage (S3 / R2 / GCS / Supabase Storage)** | *Recommended, not yet implemented.* Model `.pkl` files and large raw API dumps should not live in Postgres or the git repo. Today model artifacts are written to `data/models/` on local disk (see Gap G4). |
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

Every event row is scoped by `customer_id` + `pilot_id` + `run_id`, so pilots
are isolated and a historical decision can be reproduced exactly (given the
`data_source_hash` of the input price files).

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
| 10 | Safety-gate decisions + reason codes | Schema-ready | `decision_events.gate_status/gate_reason` columns exist; **the gate is not wired into the shadow/backtest decision path, so these are null today** |
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
shadow run  ──► DecisionRecord[] ──► JSONL (always)
                                 └─► decision_events (if DATABASE_URL set)
                                       customer_id, pilot_id, data_source_hash

(7–14 days later)
shadow realize ──► realized DecisionRecord[] ──► JSONL (always)
                                            └──► realized_outcomes (if DB set)

daily_learning_loop.py ──► read_realized_outcomes_summary()
                            reads realized_outcomes back from the store:
                            mean realized savings, mean |forecast error| (pp)
```

CLI:
```bash
# Persist decisions to a pilot-scoped store
DATABASE_URL=postgresql://... python -m aurelius.cli shadow run \
  --price-file data/q12026_3region_dam.csv --regions us-west,us-east,us-south \
  --jobs-file customer_trace.csv --forecaster ml_quantile \
  --customer-id acme --pilot-id pilot-q1 --output-dir reports/shadow/

DATABASE_URL=postgresql://... python -m aurelius.cli shadow realize \
  --decisions-file reports/shadow/decisions_*.jsonl \
  --rt-price-file rt_settlement.csv --customer-id acme --pilot-id pilot-q1
```

---

## 6. What is real vs aspirational

**Real today:**
- Append-only, customer-isolated persistence of decisions, realized outcomes,
  and telemetry, on portable Postgres/SQLite.
- Shadow mode writes every decision and (later) every realized outcome.
- The daily loop reads realized outcomes back from the store.
- Reproducibility metadata (versions + data_source_hash) on every decision.

**NOT real yet (do not claim "data moat complete"):**
- **G1 — Outcomes do not yet drive model selection.** The daily loop *reads* the
  realized-outcome summary but does not use it to accept/reject candidate
  models. Promotion still compares a freshly-trained in-engine forecaster's
  backtest savings against a stored number — the saved model artifacts are never
  loaded for inference, and realized customer outcomes are not the promotion
  criterion. This is the single biggest gap before "compounding advantage."
- **G2 — Safety-gate decisions are not recorded** because the gate is not wired
  into the shadow/backtest decision path (columns exist, values are null).
- **G3 — Telemetry writers not wired.** Queue/GPU snapshot tables exist and are
  tested, but no live ingestion path writes to them yet.
- **G4 — No object storage.** Model `.pkl` artifacts and raw API dumps live on
  local disk / not archived. Add S3/R2/GCS for artifacts before production.
- **G5 — No retention policy / Alembic migrations.** Tables are created via
  `create_all()`; there is no schema-versioning or documented retention/audit
  policy yet (enterprise procurement will require both).

A real compounding moat requires closing **G1** at minimum: realized,
customer-specific outcomes must feed back into model evaluation and selection.

---

## 7. Recommendations (priority order)

1. **Close G1:** make the daily loop evaluate the saved candidate model against
   realized outcomes (and/or a held-out leakage-free window) and promote on that
   basis, with rollback if a promoted model later underperforms on realized data.
2. **Wire the safety gate** into the decision path and persist gate_status/reason.
3. **Add object storage** for model artifacts + raw pulls; store the URI in the DB.
4. **Add Alembic migrations** and a documented retention/audit policy.
5. **Co-persist decision features** (weather/queue/GPU at decision time) to enable
   exact offline feature reconstruction without re-reading source CSVs.
