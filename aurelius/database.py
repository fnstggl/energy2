"""Supabase database integration for Aurelius.

This module handles all database operations including:
- Connection management
- CRUD operations for energy prices, carbon intensity, jobs, and simulations
- Schema documentation

No authentication layer - uses anon key only.
"""

import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for Supabase database operations.

    Uses environment variables:
    - SUPABASE_URL: The Supabase project URL
    - SUPABASE_ANON_KEY: The anonymous/public API key

    Tables expected:
    - energy_prices: timestamp, region, price_per_mwh
    - carbon_intensity: timestamp, region, gco2_per_kwh
    - jobs: job_id, submit_time, runtime_hours, deadline, power_kw, region_options
    - simulations: run_id, baseline_cost, optimized_cost, baseline_carbon,
                   optimized_carbon, savings_pct, created_at
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """Initialize Supabase client.

        Args:
            url: Supabase URL (defaults to SUPABASE_URL env var)
            key: Supabase anon key (defaults to SUPABASE_ANON_KEY env var)
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_ANON_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load the Supabase client."""
        if self._client is None:
            if not self.url or not self.key:
                logger.warning(
                    "Supabase credentials not configured. "
                    "Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables."
                )
                return None
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except ImportError:
                logger.warning("supabase-py not installed. Database features disabled.")
                return None
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}")
                return None
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if database is available."""
        return self.client is not None

    # -------------------------------------------------------------------------
    # Energy Prices
    # -------------------------------------------------------------------------

    def insert_energy_prices(self, prices: list[dict]) -> bool:
        """Insert energy price records.

        Args:
            prices: List of dicts with keys: timestamp, region, price_per_mwh

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        try:
            for price in prices:
                if isinstance(price.get("timestamp"), datetime):
                    price["timestamp"] = price["timestamp"].isoformat()
            self.client.table("energy_prices").insert(prices).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to insert energy prices: {e}")
            return False

    def get_energy_prices(
        self,
        region: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict]:
        """Retrieve energy price records.

        Args:
            region: Filter by region (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)

        Returns:
            List of price records
        """
        if not self.is_connected:
            return []
        try:
            query = self.client.table("energy_prices").select("*")
            if region:
                query = query.eq("region", region)
            if start_time:
                query = query.gte("timestamp", start_time.isoformat())
            if end_time:
                query = query.lte("timestamp", end_time.isoformat())
            result = query.order("timestamp").execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get energy prices: {e}")
            return []

    # -------------------------------------------------------------------------
    # Carbon Intensity
    # -------------------------------------------------------------------------

    def insert_carbon_intensity(self, records: list[dict]) -> bool:
        """Insert carbon intensity records.

        Args:
            records: List of dicts with keys: timestamp, region, gco2_per_kwh

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        try:
            for record in records:
                if isinstance(record.get("timestamp"), datetime):
                    record["timestamp"] = record["timestamp"].isoformat()
            self.client.table("carbon_intensity").insert(records).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to insert carbon intensity: {e}")
            return False

    def get_carbon_intensity(
        self,
        region: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict]:
        """Retrieve carbon intensity records."""
        if not self.is_connected:
            return []
        try:
            query = self.client.table("carbon_intensity").select("*")
            if region:
                query = query.eq("region", region)
            if start_time:
                query = query.gte("timestamp", start_time.isoformat())
            if end_time:
                query = query.lte("timestamp", end_time.isoformat())
            result = query.order("timestamp").execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get carbon intensity: {e}")
            return []

    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------

    def insert_jobs(self, jobs: list[dict]) -> bool:
        """Insert job records.

        Args:
            jobs: List of job dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        try:
            for job in jobs:
                for key in ["submit_time", "deadline", "earliest_start", "latest_start"]:
                    if isinstance(job.get(key), datetime):
                        job[key] = job[key].isoformat()
            self.client.table("jobs").insert(jobs).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to insert jobs: {e}")
            return False

    def get_jobs(
        self,
        job_ids: Optional[list[str]] = None,
        region: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve job records."""
        if not self.is_connected:
            return []
        try:
            query = self.client.table("jobs").select("*")
            if job_ids:
                query = query.in_("job_id", job_ids)
            if region:
                query = query.contains("region_options", [region])
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get jobs: {e}")
            return []

    # -------------------------------------------------------------------------
    # Simulations
    # -------------------------------------------------------------------------

    def save_simulation(self, result: dict) -> bool:
        """Save a simulation result.

        Args:
            result: Simulation result dictionary

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        try:
            record = {
                "run_id": result.get("run_id"),
                "baseline_cost": result.get("baseline_cost"),
                "optimized_cost": result.get("optimized_cost"),
                "baseline_carbon": result.get("baseline_carbon_kg"),
                "optimized_carbon": result.get("optimized_carbon_kg"),
                "savings_pct": result.get("cost_savings_pct"),
                "created_at": result.get("created_at"),
            }
            self.client.table("simulations").insert(record).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save simulation: {e}")
            return False

    def get_simulations(
        self,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve simulation records."""
        if not self.is_connected:
            return []
        try:
            query = self.client.table("simulations").select("*")
            if run_id:
                query = query.eq("run_id", run_id)
            result = query.order("created_at", desc=True).limit(limit).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get simulations: {e}")
            return []


# Schema documentation for reference
SCHEMA_SQL = """
-- Aurelius v1 Database Schema
-- Run these statements in your Supabase SQL editor

-- Energy prices table
CREATE TABLE IF NOT EXISTS energy_prices (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    region TEXT NOT NULL,
    price_per_mwh DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, region)
);

CREATE INDEX idx_energy_prices_region_time ON energy_prices(region, timestamp);

-- Carbon intensity table
CREATE TABLE IF NOT EXISTS carbon_intensity (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    region TEXT NOT NULL,
    gco2_per_kwh DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, region)
);

CREATE INDEX idx_carbon_intensity_region_time ON carbon_intensity(region, timestamp);

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    submit_time TIMESTAMPTZ NOT NULL,
    runtime_hours DECIMAL(10, 2) NOT NULL,
    deadline TIMESTAMPTZ NOT NULL,
    power_kw DECIMAL(10, 2) NOT NULL,
    region_options TEXT[] NOT NULL,
    earliest_start TIMESTAMPTZ,
    latest_start TIMESTAMPTZ,
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_submit_time ON jobs(submit_time);

-- Simulations table
CREATE TABLE IF NOT EXISTS simulations (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    baseline_cost DECIMAL(12, 2) NOT NULL,
    optimized_cost DECIMAL(12, 2) NOT NULL,
    baseline_carbon DECIMAL(12, 2) NOT NULL,
    optimized_carbon DECIMAL(12, 2) NOT NULL,
    savings_pct DECIMAL(5, 2),
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_simulations_created_at ON simulations(created_at DESC);
"""


def print_schema():
    """Print the database schema SQL for manual setup."""
    print(SCHEMA_SQL)


# Singleton instance
_db: Optional[SupabaseClient] = None


def get_db() -> SupabaseClient:
    """Get the global database client instance."""
    global _db
    if _db is None:
        _db = SupabaseClient()
    return _db
