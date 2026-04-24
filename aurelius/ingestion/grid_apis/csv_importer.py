"""CSV fallback importer for price and carbon data.

Accepts CSVs in any of these formats and normalises to the canonical schema:

Price CSV (required cols):
  timestamp, region, price_per_mwh
  Optional: currency, source, source_granularity

Carbon CSV (required cols):
  timestamp, region, gco2_per_kwh
  Optional: source, source_granularity

Timestamps may be ISO-8601 strings, Unix seconds, or any format pandas
can parse. Timezone-naive timestamps are assumed UTC.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .base import (
    PriceProvider,
    CarbonProvider,
    ProviderConfigError,
    PRICE_COLUMNS,
    CARBON_COLUMNS,
    empty_price_df,
    empty_carbon_df,
    normalize_price_df,
    normalize_carbon_df,
)

logger = logging.getLogger(__name__)

_REQUIRED_PRICE_COLS = {"timestamp", "region", "price_per_mwh"}
_REQUIRED_CARBON_COLS = {"timestamp", "region", "gco2_per_kwh"}


def _load_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file, returning None on failure."""
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {path}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load CSV {path}: {exc}")
        return None


class CSVPriceImporter(PriceProvider):
    """Load electricity price data from a CSV file.

    The CSV must have at minimum: timestamp, region, price_per_mwh.
    Any additional canonical columns present in the file are respected.

    Args:
        path: Path to the CSV file.
        default_currency: ISO-4217 currency code (default "USD").
        default_granularity: Time resolution string (default "hourly").
    """

    def __init__(
        self,
        path: Union[str, Path],
        default_currency: str = "USD",
        default_granularity: str = "hourly",
    ) -> None:
        self._path = Path(path)
        self._currency = default_currency
        self._granularity = default_granularity

    @property
    def source_name(self) -> str:
        return f"csv:{self._path.name}"

    def fetch_prices(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return price rows for *region* within [start, end).

        Filters by region after loading; filters by time range.
        Returns empty_price_df() on any error.
        """
        raw = _load_csv(self._path)
        if raw is None:
            return empty_price_df()

        missing = _REQUIRED_PRICE_COLS - set(raw.columns)
        if missing:
            logger.error(f"CSV {self._path} missing required columns: {missing}")
            return empty_price_df()

        df = raw[raw["region"] == region].copy()
        if df.empty:
            logger.warning(f"No rows for region '{region}' in {self._path}")
            return empty_price_df()

        try:
            df = normalize_price_df(
                df,
                source=self.source_name,
                currency=df.get("currency", self._currency).iloc[0]
                if "currency" in df.columns else self._currency,
                granularity=self._granularity,
            )
        except Exception as exc:
            logger.error(f"Normalisation error for {self._path}: {exc}")
            return empty_price_df()

        start_ts = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
        return df.reset_index(drop=True)

    def load_all(self) -> pd.DataFrame:
        """Load entire file, normalised but unfiltered by region/time."""
        raw = _load_csv(self._path)
        if raw is None:
            return empty_price_df()
        missing = _REQUIRED_PRICE_COLS - set(raw.columns)
        if missing:
            logger.error(f"CSV {self._path} missing required columns: {missing}")
            return empty_price_df()
        try:
            return normalize_price_df(
                raw,
                source=self.source_name,
                currency=self._currency,
                granularity=self._granularity,
            )
        except Exception as exc:
            logger.error(f"Normalisation error: {exc}")
            return empty_price_df()


class CSVCarbonImporter(CarbonProvider):
    """Load carbon intensity data from a CSV file.

    Required columns: timestamp, region, gco2_per_kwh
    """

    def __init__(
        self,
        path: Union[str, Path],
        default_granularity: str = "hourly",
    ) -> None:
        self._path = Path(path)
        self._granularity = default_granularity

    @property
    def source_name(self) -> str:
        return f"csv:{self._path.name}"

    def fetch_carbon(
        self,
        region: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return carbon rows for *region* within [start, end)."""
        raw = _load_csv(self._path)
        if raw is None:
            return empty_carbon_df()

        missing = _REQUIRED_CARBON_COLS - set(raw.columns)
        if missing:
            logger.error(f"CSV {self._path} missing required columns: {missing}")
            return empty_carbon_df()

        df = raw[raw["region"] == region].copy()
        if df.empty:
            logger.warning(f"No rows for region '{region}' in {self._path}")
            return empty_carbon_df()

        try:
            df = normalize_carbon_df(df, source=self.source_name, granularity=self._granularity)
        except Exception as exc:
            logger.error(f"Normalisation error for {self._path}: {exc}")
            return empty_carbon_df()

        start_ts = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
        return df.reset_index(drop=True)

    def load_all(self) -> pd.DataFrame:
        """Load entire file, normalised but unfiltered."""
        raw = _load_csv(self._path)
        if raw is None:
            return empty_carbon_df()
        missing = _REQUIRED_CARBON_COLS - set(raw.columns)
        if missing:
            logger.error(f"CSV {self._path} missing required columns: {missing}")
            return empty_carbon_df()
        try:
            return normalize_carbon_df(raw, source=self.source_name, granularity=self._granularity)
        except Exception as exc:
            logger.error(f"Normalisation error: {exc}")
            return empty_carbon_df()
