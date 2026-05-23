"""Provenance-aware market-data provider abstraction for Aurelius.

This module sits *above* ``grid_apis/base.py``. Where ``base.py`` defines the
canonical pandas-DataFrame schemas used throughout the optimizer/backtester,
this module adds the per-observation provenance the rest of the task requires:

* which provider produced the value,
* whether it is a true source-of-truth reading or an aggregated/estimated one,
* whether it came from a sandbox / randomized endpoint (which must never be
  used for savings or benchmark claims),
* the native granularity and when it was fetched.

The two layers are deliberately complementary, not duplicative:

    base.py            -> bulk numeric series as DataFrames (hot path: backtest)
    market_data_provider.py -> typed points + capability discovery + provenance

Helpers are provided to convert typed points back into the canonical
DataFrame schemas (``points_to_price_df`` / ``points_to_carbon_df``) so a
provenance-aware provider can still feed the existing DataFrame pipeline.

Nothing here imports or copies any Electricity Maps code. The Electricity Maps
contrib repository (AGPL-3.0) was used only as a *reference* for official data
sources and zone naming — see docs/ELECTRICITYMAPS_CONTRIB_AUDIT.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

# NOTE: base.py is imported lazily inside the conversion helpers below to avoid
# a circular import (grid_apis/__init__ imports the Electricity Maps adapter,
# which imports this module).


# ---------------------------------------------------------------------------
# Provenance vocabulary
# ---------------------------------------------------------------------------

class Provenance:
    """How trustworthy is a single observation, for downstream gating.

    SOURCE_OF_TRUTH : settled/published value direct from the ISO/TSO operator.
    AGGREGATED      : re-published by an aggregator (e.g. Electricity Maps) that
                      itself sourced it from the operator. Usually fine, but the
                      operator is the authority of record.
    ESTIMATED       : modelled / interpolated / forecast — not a measured value.
    SANDBOX         : randomized or synthetic value from a sandbox/demo endpoint.
                      MUST NOT be used for savings or benchmark claims.
    """

    SOURCE_OF_TRUTH = "source_of_truth"
    AGGREGATED = "aggregated"
    ESTIMATED = "estimated"
    SANDBOX = "sandbox"

    ALL = frozenset({SOURCE_OF_TRUTH, AGGREGATED, ESTIMATED, SANDBOX})

    # Provenance values that are NOT admissible as evidence for economic /
    # benchmark claims about real-world savings.
    NON_BENCHMARK = frozenset({SANDBOX})


class MarketType:
    """Wholesale market product types Aurelius understands."""

    DAY_AHEAD_LMP = "day_ahead_lmp"
    REAL_TIME_LMP = "real_time_lmp"
    DAY_AHEAD_PRICE = "day_ahead_price"   # zonal/bidding-zone price (not nodal LMP)
    REAL_TIME_PRICE = "real_time_price"
    HUB_PRICE = "hub_price"

    ALL = frozenset({
        DAY_AHEAD_LMP, REAL_TIME_LMP, DAY_AHEAD_PRICE, REAL_TIME_PRICE, HUB_PRICE,
    })


class Signal:
    """Data signals a provider may expose."""

    PRICE = "price"
    CARBON = "carbon"

    ALL = frozenset({PRICE, CARBON})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Typed observations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketPricePoint:
    """A single wholesale electricity price observation with provenance."""

    timestamp: datetime          # UTC-aware
    region: str                  # canonical Aurelius region (e.g. "us-west")
    price_per_mwh: float
    currency: str = "USD"
    market_type: str = MarketType.DAY_AHEAD_LMP
    source: str = ""             # operator / authority of record (e.g. "CAISO")
    provider: str = ""           # adapter that fetched it (e.g. "electricitymaps")
    source_granularity: str = "hourly"
    provenance: str = Provenance.SOURCE_OF_TRUTH
    is_sandbox: bool = False
    is_estimated: bool = False
    fetched_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.provenance not in Provenance.ALL:
            raise ValueError(f"Unknown provenance: {self.provenance!r}")
        if self.market_type not in MarketType.ALL:
            raise ValueError(f"Unknown market_type: {self.market_type!r}")
        # Sandbox and estimated provenance must agree with the boolean flags so
        # downstream gates can rely on either signal.
        if self.is_sandbox and self.provenance != Provenance.SANDBOX:
            object.__setattr__(self, "provenance", Provenance.SANDBOX)
        if self.provenance == Provenance.SANDBOX and not self.is_sandbox:
            object.__setattr__(self, "is_sandbox", True)
        if self.provenance == Provenance.ESTIMATED:
            object.__setattr__(self, "is_estimated", True)

    @property
    def benchmark_admissible(self) -> bool:
        """True only if this value may back a savings/benchmark claim."""
        return not self.is_sandbox and self.provenance not in Provenance.NON_BENCHMARK


@dataclass(frozen=True)
class CarbonPoint:
    """A single carbon-intensity observation with provenance."""

    timestamp: datetime          # UTC-aware
    region: str
    gco2_per_kwh: float
    source: str = ""
    provider: str = ""
    source_granularity: str = "hourly"
    provenance: str = Provenance.AGGREGATED
    is_sandbox: bool = False
    is_estimated: bool = False
    fetched_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.provenance not in Provenance.ALL:
            raise ValueError(f"Unknown provenance: {self.provenance!r}")
        if self.is_sandbox and self.provenance != Provenance.SANDBOX:
            object.__setattr__(self, "provenance", Provenance.SANDBOX)
        if self.provenance == Provenance.SANDBOX and not self.is_sandbox:
            object.__setattr__(self, "is_sandbox", True)
        if self.provenance == Provenance.ESTIMATED:
            object.__setattr__(self, "is_estimated", True)

    @property
    def benchmark_admissible(self) -> bool:
        return not self.is_sandbox and self.provenance not in Provenance.NON_BENCHMARK


@dataclass(frozen=True)
class ProviderCapability:
    """Declares what one provider can do for one signal.

    Used for discovery/routing so Aurelius can pick a source-of-truth provider
    first and fall back to an aggregator only when necessary.
    """

    provider: str
    signal: str                       # Signal.PRICE | Signal.CARBON
    regions: tuple[str, ...]          # canonical Aurelius regions supported
    granularity: str                  # e.g. "hourly", "5min"
    history_supported: bool
    forecast_supported: bool
    sandbox_supported: bool
    production_supported: bool
    auth_required: bool
    market_types: tuple[str, ...] = ()  # for price signals

    def __post_init__(self) -> None:
        if self.signal not in Signal.ALL:
            raise ValueError(f"Unknown signal: {self.signal!r}")

    def supports_region(self, region: str) -> bool:
        return region in self.regions


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class BenchmarkDataError(Exception):
    """Raised when sandbox/estimated data is used where real data is required."""


class MarketDataProvider(ABC):
    """Provenance-aware market-data provider.

    Concrete providers return typed points carrying provenance. Use
    ``points_to_price_df`` / ``points_to_carbon_df`` to feed the canonical
    DataFrame pipeline when needed.
    """

    @abstractmethod
    def get_capabilities(self) -> list[ProviderCapability]:
        """Return what this provider can do, per signal."""

    @abstractmethod
    def fetch_price_series(
        self,
        region: str,
        start_dt: datetime,
        end_dt: datetime,
        market_type: str = MarketType.DAY_AHEAD_LMP,
    ) -> list[MarketPricePoint]:
        """Fetch price points for [start_dt, end_dt). Empty list if unsupported."""

    @abstractmethod
    def fetch_carbon_series(
        self,
        region: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[CarbonPoint]:
        """Fetch carbon points for [start_dt, end_dt). Empty list if unsupported."""

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Return True if credentials are present and accepted (no network calls
        required for absent credentials — return False instead of raising)."""

    @abstractmethod
    def is_sandbox(self) -> bool:
        """True if this provider is currently operating against sandbox data."""


# ---------------------------------------------------------------------------
# Benchmark-safety gate
# ---------------------------------------------------------------------------

def assert_benchmark_admissible(points) -> None:
    """Raise BenchmarkDataError if any point is sandbox/non-admissible.

    Call this from savings/benchmark code paths before treating values as real.
    """
    for p in points:
        if not getattr(p, "benchmark_admissible", True):
            raise BenchmarkDataError(
                f"Refusing to use {p.provenance} data from provider "
                f"'{p.provider}' for a benchmark/savings claim "
                f"(region={p.region}, ts={p.timestamp}). "
                "Sandbox/randomized data is for connector/schema tests only; "
                "production claims require real unrandomized historical data."
            )


def filter_benchmark_admissible(points) -> list:
    """Return only the points admissible for benchmark/savings claims."""
    return [p for p in points if getattr(p, "benchmark_admissible", True)]


# ---------------------------------------------------------------------------
# Conversion to the canonical DataFrame schema
# ---------------------------------------------------------------------------

def points_to_price_df(points: list[MarketPricePoint]) -> pd.DataFrame:
    """Convert price points to the canonical PRICE_COLUMNS DataFrame.

    Provenance columns (provider/provenance/market_type/is_sandbox/is_estimated)
    are appended after the canonical columns so existing consumers that select
    PRICE_COLUMNS keep working unchanged.
    """
    from .grid_apis.base import PRICE_COLUMNS, empty_price_df
    if not points:
        return empty_price_df()
    rows = []
    for p in points:
        rows.append({
            "timestamp": pd.Timestamp(p.timestamp).tz_convert("UTC")
            if pd.Timestamp(p.timestamp).tzinfo else pd.Timestamp(p.timestamp).tz_localize("UTC"),
            "region": p.region,
            "price_per_mwh": float(p.price_per_mwh),
            "currency": p.currency,
            "source": p.source or p.provider,
            "source_granularity": p.source_granularity,
            "fetched_at": pd.Timestamp(p.fetched_at),
            "provider": p.provider,
            "market_type": p.market_type,
            "provenance": p.provenance,
            "is_sandbox": p.is_sandbox,
            "is_estimated": p.is_estimated,
        })
    df = pd.DataFrame(rows)
    extra = ["provider", "market_type", "provenance", "is_sandbox", "is_estimated"]
    return df[PRICE_COLUMNS + extra]


def points_to_carbon_df(points: list[CarbonPoint]) -> pd.DataFrame:
    """Convert carbon points to the canonical CARBON_COLUMNS DataFrame (+provenance)."""
    from .grid_apis.base import CARBON_COLUMNS, empty_carbon_df
    if not points:
        return empty_carbon_df()
    rows = []
    for p in points:
        rows.append({
            "timestamp": pd.Timestamp(p.timestamp).tz_convert("UTC")
            if pd.Timestamp(p.timestamp).tzinfo else pd.Timestamp(p.timestamp).tz_localize("UTC"),
            "region": p.region,
            "gco2_per_kwh": float(p.gco2_per_kwh),
            "source": p.source or p.provider,
            "source_granularity": p.source_granularity,
            "fetched_at": pd.Timestamp(p.fetched_at),
            "provider": p.provider,
            "provenance": p.provenance,
            "is_sandbox": p.is_sandbox,
            "is_estimated": p.is_estimated,
        })
    df = pd.DataFrame(rows)
    extra = ["provider", "provenance", "is_sandbox", "is_estimated"]
    return df[CARBON_COLUMNS + extra]


__all__ = [
    "Provenance",
    "MarketType",
    "Signal",
    "MarketPricePoint",
    "CarbonPoint",
    "ProviderCapability",
    "MarketDataProvider",
    "BenchmarkDataError",
    "assert_benchmark_admissible",
    "filter_benchmark_admissible",
    "points_to_price_df",
    "points_to_carbon_df",
]
