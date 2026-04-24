"""Abstract provider interfaces for grid energy and carbon data.

All providers normalize output to canonical DataFrame schemas so the rest
of the system never has to know which data source it's talking to.

Price schema columns:  PRICE_COLUMNS
Carbon schema columns: CARBON_COLUMNS
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column sets (order is enforced by normalizers)
# ---------------------------------------------------------------------------
PRICE_COLUMNS = [
    "timestamp",          # pd.Timestamp UTC-aware
    "region",             # str identifier (e.g. "us-west")
    "price_per_mwh",      # float $/MWh
    "currency",           # str ISO-4217 (e.g. "USD", "EUR")
    "source",             # str provider name
    "source_granularity", # str e.g. "hourly", "5min"
    "fetched_at",         # pd.Timestamp UTC-aware — when we pulled this row
]

CARBON_COLUMNS = [
    "timestamp",
    "region",
    "gco2_per_kwh",       # float grams CO2-eq per kWh
    "source",
    "source_granularity",
    "fetched_at",
]


class ProviderConfigError(Exception):
    """Raised when a required API key or configuration value is missing."""


def empty_price_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical price schema."""
    return pd.DataFrame(columns=PRICE_COLUMNS)


def empty_carbon_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical carbon schema."""
    return pd.DataFrame(columns=CARBON_COLUMNS)


def _ensure_utc(ts) -> pd.Timestamp:
    """Coerce any timestamp-like value to a UTC-aware pd.Timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def normalize_price_df(df: pd.DataFrame, source: str, currency: str = "USD",
                        granularity: str = "hourly") -> pd.DataFrame:
    """Enforce canonical schema on a raw price DataFrame.

    Required input columns: timestamp, region, price_per_mwh
    All other canonical columns are filled-in if missing.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].apply(_ensure_utc)
    df["currency"] = currency
    df["source"] = source
    df["source_granularity"] = granularity
    df["fetched_at"] = _ensure_utc(datetime.now(timezone.utc))
    df["price_per_mwh"] = pd.to_numeric(df["price_per_mwh"], errors="coerce")
    df = df.dropna(subset=["price_per_mwh"])
    return df[PRICE_COLUMNS].reset_index(drop=True)


def normalize_carbon_df(df: pd.DataFrame, source: str,
                         granularity: str = "hourly") -> pd.DataFrame:
    """Enforce canonical schema on a raw carbon DataFrame.

    Required input columns: timestamp, region, gco2_per_kwh
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].apply(_ensure_utc)
    df["source"] = source
    df["source_granularity"] = granularity
    df["fetched_at"] = _ensure_utc(datetime.now(timezone.utc))
    df["gco2_per_kwh"] = pd.to_numeric(df["gco2_per_kwh"], errors="coerce")
    df = df.dropna(subset=["gco2_per_kwh"])
    return df[CARBON_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class PriceProvider(ABC):
    """Abstract interface for electricity price data providers."""

    @abstractmethod
    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly electricity price data.

        Args:
            region: Provider-specific region identifier.
            start: Inclusive start datetime (timezone-naive → assumed UTC).
            end:   Exclusive end datetime.

        Returns:
            DataFrame with columns matching PRICE_COLUMNS.
            Returns empty_price_df() on failure — never raises on missing data.

        Raises:
            ProviderConfigError: If API key or required config is absent.
        """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable provider name, used in the 'source' column."""


class CarbonProvider(ABC):
    """Abstract interface for carbon intensity data providers."""

    @abstractmethod
    def fetch_carbon(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly carbon intensity data.

        Returns:
            DataFrame with columns matching CARBON_COLUMNS.
            Returns empty_carbon_df() on failure — never raises on missing data.

        Raises:
            ProviderConfigError: If API key or required config is absent.
        """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable provider name."""
