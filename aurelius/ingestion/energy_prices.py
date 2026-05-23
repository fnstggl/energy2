"""Energy price data ingestion.

This module handles:
- Loading energy price data from CSV/JSON files
- Generating synthetic price data for simulation
- Storing prices in Supabase
- Fetching prices for optimization
"""

import csv
import json
import logging
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..database import get_db
from ..models import EnergyPrice

logger = logging.getLogger(__name__)


class EnergyPriceIngester:
    """Handles energy price data ingestion and generation."""

    # Typical base prices by region ($/MWh)
    REGION_BASE_PRICES = {
        "us-west": 45.0,
        "us-east": 55.0,
        "eu-west": 65.0,
        "eu-north": 50.0,
        "asia-east": 70.0,
    }

    # Peak hours (0-23)
    PEAK_HOURS = {9, 10, 11, 12, 13, 14, 17, 18, 19, 20}

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the ingester.

        Args:
            data_dir: Directory for data files (optional)
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.db = get_db()

    def load_from_csv(self, filepath: Path) -> list[EnergyPrice]:
        """Load energy prices from a CSV file.

        Expected columns: timestamp, region, price_per_mwh

        Args:
            filepath: Path to CSV file

        Returns:
            List of EnergyPrice objects
        """
        prices = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = EnergyPrice(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    region=row["region"],
                    price_per_mwh=float(row["price_per_mwh"]),
                )
                prices.append(price)
        logger.info(f"Loaded {len(prices)} price records from {filepath}")
        return prices

    def load_from_json(self, filepath: Path) -> list[EnergyPrice]:
        """Load energy prices from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List of EnergyPrice objects
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        prices = []
        for record in data:
            price = EnergyPrice(
                timestamp=datetime.fromisoformat(record["timestamp"]),
                region=record["region"],
                price_per_mwh=float(record["price_per_mwh"]),
            )
            prices.append(price)
        logger.info(f"Loaded {len(prices)} price records from {filepath}")
        return prices

    def generate_synthetic(
        self,
        start_time: datetime,
        hours: int,
        regions: Optional[list[str]] = None,
        volatility: float = 0.15,
        seed: Optional[int] = None,
    ) -> list[EnergyPrice]:
        """Generate synthetic energy price data.

        Creates realistic price patterns with:
        - Base regional prices
        - Hour-of-day seasonality (peak vs off-peak)
        - Day-of-week effects (weekends cheaper)
        - Random noise

        Args:
            start_time: Start of the time series
            hours: Number of hours to generate
            regions: List of regions (defaults to all known regions)
            volatility: Price volatility factor (0-1)
            seed: Random seed for reproducibility

        Returns:
            List of EnergyPrice objects
        """
        if seed is not None:
            random.seed(seed)

        regions = regions or list(self.REGION_BASE_PRICES.keys())
        prices = []

        for hour_offset in range(hours):
            timestamp = start_time + timedelta(hours=hour_offset)
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()

            for region in regions:
                base_price = self.REGION_BASE_PRICES.get(region, 50.0)

                # Hour-of-day effect (peak hours more expensive)
                if hour_of_day in self.PEAK_HOURS:
                    hourly_factor = 1.3 + 0.1 * math.sin(hour_of_day * math.pi / 12)
                else:
                    hourly_factor = 0.7 + 0.1 * math.sin(hour_of_day * math.pi / 12)

                # Weekend discount
                weekend_factor = 0.85 if day_of_week >= 5 else 1.0

                # Random noise
                noise = 1 + random.gauss(0, volatility)

                price = max(5.0, base_price * hourly_factor * weekend_factor * noise)

                prices.append(EnergyPrice(
                    timestamp=timestamp,
                    region=region,
                    price_per_mwh=round(price, 2),
                ))

        logger.info(
            f"Generated {len(prices)} synthetic prices "
            f"({hours} hours × {len(regions)} regions)"
        )
        return prices

    def save_to_csv(self, prices: list[EnergyPrice], filepath: Path) -> None:
        """Save energy prices to a CSV file.

        Args:
            prices: List of EnergyPrice objects
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "region", "price_per_mwh"])
            writer.writeheader()
            for price in prices:
                writer.writerow({
                    "timestamp": price.timestamp.isoformat(),
                    "region": price.region,
                    "price_per_mwh": price.price_per_mwh,
                })
        logger.info(f"Saved {len(prices)} prices to {filepath}")

    def save_to_database(self, prices: list[EnergyPrice]) -> bool:
        """Save energy prices to Supabase.

        Args:
            prices: List of EnergyPrice objects

        Returns:
            True if successful
        """
        records = [
            {
                "timestamp": p.timestamp,
                "region": p.region,
                "price_per_mwh": p.price_per_mwh,
            }
            for p in prices
        ]
        return self.db.insert_energy_prices(records)

    def fetch_prices(
        self,
        region: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[EnergyPrice]:
        """Fetch energy prices from the database.

        Args:
            region: Filter by region
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of EnergyPrice objects
        """
        records = self.db.get_energy_prices(region, start_time, end_time)
        return [
            EnergyPrice(
                timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
                region=r["region"],
                price_per_mwh=float(r["price_per_mwh"]),
            )
            for r in records
        ]

    def prices_to_dict(
        self,
        prices: list[EnergyPrice],
    ) -> dict[str, dict[datetime, float]]:
        """Convert price list to nested dict for fast lookup.

        Returns:
            Dict of {region: {timestamp: price_per_mwh}}
        """
        result: dict[str, dict[datetime, float]] = {}
        for price in prices:
            if price.region not in result:
                result[price.region] = {}
            result[price.region][price.timestamp] = price.price_per_mwh
        return result
