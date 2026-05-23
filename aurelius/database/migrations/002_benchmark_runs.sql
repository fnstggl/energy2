-- Aurelius benchmark run archival table.
--
-- One row per (run_id, region_combo, workload) benchmark cell.
-- Allows regression tracking across runs and forecaster comparisons.
--
-- Usage:
--   psql $DATABASE_URL -f 002_benchmark_runs.sql

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              BIGSERIAL,
    run_id          TEXT            NOT NULL,
    run_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    forecaster      TEXT            NOT NULL,
    region_combo    TEXT            NOT NULL,
    workload        TEXT            NOT NULL,
    savings_vs_cpo  DOUBLE PRECISION NOT NULL,
    folds           INTEGER         NOT NULL,
    miss_pct        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    meta_json       TEXT,
    PRIMARY KEY (id, run_at)
);

-- Unique per run + workload cell (overwrite on re-run)
CREATE UNIQUE INDEX IF NOT EXISTS uq_benchmark_run_cell
    ON benchmark_runs (run_id, region_combo, workload);

-- Fast lookup for regression comparison
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_combo_workload_ts
    ON benchmark_runs (region_combo, workload, run_at DESC);

-- Optional TimescaleDB hypertable
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) THEN
        PERFORM create_hypertable(
            'benchmark_runs', 'run_at',
            if_not_exists => TRUE,
            migrate_data  => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for benchmark_runs.';
    ELSE
        RAISE NOTICE 'TimescaleDB not present; using plain Postgres benchmark_runs table.';
    END IF;
END
$$;
