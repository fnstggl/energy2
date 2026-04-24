-- Aurelius time-series tables for energy prices and carbon intensity.
--
-- Usage:
--   Standard Postgres:   psql $DATABASE_URL -f 001_timeseries.sql
--   TimescaleDB (optional): set TIMESCALEDB=1 in the session before running.
--
-- The TimescaleDB hypertable calls are wrapped in a DO block so the migration
-- degrades gracefully on plain Postgres (the extension simply won't be found).

-- ---------------------------------------------------------------------------
-- energy_prices
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS energy_prices (
    id                  BIGSERIAL,
    timestamp           TIMESTAMPTZ     NOT NULL,
    region              TEXT            NOT NULL,
    price_per_mwh       DOUBLE PRECISION NOT NULL,
    currency            TEXT            NOT NULL DEFAULT 'USD',
    source              TEXT            NOT NULL,
    source_granularity  TEXT            NOT NULL DEFAULT 'hourly',
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Prevent duplicate rows for the same (timestamp, region, source) tuple.
CREATE UNIQUE INDEX IF NOT EXISTS uq_energy_prices_ts_region_source
    ON energy_prices (timestamp, region, source);

-- Fast lookups by region + time range (the most common query pattern).
CREATE INDEX IF NOT EXISTS idx_energy_prices_region_ts
    ON energy_prices (region, timestamp DESC);

-- ---------------------------------------------------------------------------
-- carbon_intensity
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS carbon_intensity (
    id                  BIGSERIAL,
    timestamp           TIMESTAMPTZ     NOT NULL,
    region              TEXT            NOT NULL,
    gco2_per_kwh        DOUBLE PRECISION NOT NULL,
    source              TEXT            NOT NULL,
    source_granularity  TEXT            NOT NULL DEFAULT 'hourly',
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_carbon_intensity_ts_region_source
    ON carbon_intensity (timestamp, region, source);

CREATE INDEX IF NOT EXISTS idx_carbon_intensity_region_ts
    ON carbon_intensity (region, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Optional: convert to TimescaleDB hypertables
-- This block is a no-op on plain Postgres.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) THEN
        PERFORM create_hypertable(
            'energy_prices', 'timestamp',
            if_not_exists => TRUE,
            migrate_data  => TRUE
        );
        PERFORM create_hypertable(
            'carbon_intensity', 'timestamp',
            if_not_exists => TRUE,
            migrate_data  => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertables created.';
    ELSE
        RAISE NOTICE 'TimescaleDB not present; using plain Postgres tables.';
    END IF;
END
$$;
