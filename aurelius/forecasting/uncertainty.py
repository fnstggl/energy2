"""Forecast packaging for Aurelius.

This module provides:
- Normalizing and packaging probabilistic forecast outputs
- Enforcing invariants (p90 >= p50)
- Fallback handling when quantiles unavailable
- Legacy forecast conversion

IMPORTANT:
- This module is for OUTPUT PACKAGING ONLY
- MUST NOT train models (training is in price_model/carbon_model)
- MUST NOT alter forecasts beyond invariant enforcement
- MUST NOT introduce safety gating or decision logic
- MUST NOT influence optimizer behavior
- Forecasting is advisory data only
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .carbon_model import CarbonForecast, CarbonQuantileForecast
from .price_model import PriceForecast, PriceQuantileForecast
from .quantile_model import validate_quantiles

logger = logging.getLogger(__name__)


@dataclass
class DecisionForecast:
    """Forecast attachment for a ScheduleDecision.

    This is the output contract format that can be attached to decisions
    without modifying the optimizer interfaces.

    Output Contract:
        decision.forecast = {
            "energy_cost": {"p50": float, "p90": float},
            "carbon": {"p50": float, "p90": float},
            "model_meta": {"model_type": str, "trained_at": str, "features_version": str}
        }

    Attributes:
        energy_cost: Price forecast with p50 and p90
        carbon: Carbon forecast with p50 and p90
        model_meta: Metadata about the models used
    """
    energy_cost: dict  # {"p50": float, "p90": float}
    carbon: dict       # {"p50": float, "p90": float}
    model_meta: dict   # {"model_type": str, "trained_at": str, "features_version": str}

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "energy_cost": self.energy_cost,
            "carbon": self.carbon,
            "model_meta": self.model_meta,
        }

    @classmethod
    def from_quantile_forecasts(
        cls,
        price_forecast: PriceQuantileForecast,
        carbon_forecast: CarbonQuantileForecast,
    ) -> "DecisionForecast":
        """Create from quantile forecast objects.

        Args:
            price_forecast: Price quantile forecast
            carbon_forecast: Carbon quantile forecast

        Returns:
            DecisionForecast with validated quantiles
        """
        # Validate and enforce invariants
        price_p50, price_p90 = validate_quantiles(
            price_forecast.p50, price_forecast.p90
        )
        carbon_p50, carbon_p90 = validate_quantiles(
            carbon_forecast.p50, carbon_forecast.p90
        )

        return cls(
            energy_cost={
                "p50": round(price_p50, 2),
                "p90": round(price_p90, 2),
            },
            carbon={
                "p50": round(carbon_p50, 1),
                "p90": round(carbon_p90, 1),
            },
            model_meta={
                "model_type": price_forecast.model_type,
                "trained_at": datetime.utcnow().isoformat(),
                "features_version": price_forecast.features_version,
            },
        )

    @classmethod
    def from_legacy_forecasts(
        cls,
        price_forecast: PriceForecast,
        carbon_forecast: CarbonForecast,
    ) -> "DecisionForecast":
        """Create from legacy mean/std forecast objects.

        Converts to quantiles using mean as p50 and mean + 1.28*std as p90.
        (1.28 std gives approximately 90th percentile for normal distribution)

        Args:
            price_forecast: Legacy price forecast with mean/std
            carbon_forecast: Legacy carbon forecast with mean/std

        Returns:
            DecisionForecast with derived quantiles
        """
        # Derive p90 from mean + 1.28 std (approx 90th percentile)
        price_p50 = price_forecast.mean
        price_p90 = price_forecast.mean + 1.28 * price_forecast.std

        carbon_p50 = carbon_forecast.mean
        carbon_p90 = carbon_forecast.mean + 1.28 * carbon_forecast.std

        # Validate invariants
        price_p50, price_p90 = validate_quantiles(price_p50, price_p90)
        carbon_p50, carbon_p90 = validate_quantiles(carbon_p50, carbon_p90)

        return cls(
            energy_cost={
                "p50": round(price_p50, 2),
                "p90": round(price_p90, 2),
            },
            carbon={
                "p50": round(carbon_p50, 1),
                "p90": round(carbon_p90, 1),
            },
            model_meta={
                "model_type": "legacy_mean_std",
                "trained_at": datetime.utcnow().isoformat(),
                "features_version": "v1",
            },
        )

    @classmethod
    def fallback(
        cls,
        default_price_p50: float = 50.0,
        default_carbon_p50: float = 400.0,
        p90_uplift: float = 1.3,
    ) -> "DecisionForecast":
        """Create a fallback forecast when models unavailable.

        Args:
            default_price_p50: Default price p50 value
            default_carbon_p50: Default carbon p50 value
            p90_uplift: Multiplier for p90 relative to p50

        Returns:
            DecisionForecast with fallback values
        """
        return cls(
            energy_cost={
                "p50": round(default_price_p50, 2),
                "p90": round(default_price_p50 * p90_uplift, 2),
            },
            carbon={
                "p50": round(default_carbon_p50, 1),
                "p90": round(default_carbon_p50 * p90_uplift, 1),
            },
            model_meta={
                "model_type": "fallback",
                "trained_at": datetime.utcnow().isoformat(),
                "features_version": "v1",
            },
        )


class ForecastPackager:
    """Packages forecasts for attachment to ScheduleDecision.

    This class handles:
    - Combining price and carbon forecasts
    - Enforcing p90 >= p50 invariant
    - Providing fallback when quantiles unavailable
    - Formatting for the output contract

    IMPORTANT:
    - Does NOT train models
    - Does NOT alter forecasts beyond invariant enforcement
    - Does NOT introduce safety gating
    - Does NOT influence optimizer behavior
    - Advisory output packaging ONLY
    """

    def __init__(
        self,
        default_price_p50: float = 50.0,
        default_carbon_p50: float = 400.0,
        p90_uplift: float = 1.3,
    ):
        """Initialize the forecast packager.

        Args:
            default_price_p50: Default price for fallback
            default_carbon_p50: Default carbon for fallback
            p90_uplift: Multiplier for p90 when using fallback
        """
        self.default_price_p50 = default_price_p50
        self.default_carbon_p50 = default_carbon_p50
        self.p90_uplift = p90_uplift

    def package(
        self,
        price_forecast: Optional[PriceQuantileForecast] = None,
        carbon_forecast: Optional[CarbonQuantileForecast] = None,
    ) -> DecisionForecast:
        """Package forecasts into decision attachment format.

        Handles missing forecasts with fallback values.

        Args:
            price_forecast: Optional price quantile forecast
            carbon_forecast: Optional carbon quantile forecast

        Returns:
            DecisionForecast ready for attachment
        """
        # If both provided, use them
        if price_forecast is not None and carbon_forecast is not None:
            return DecisionForecast.from_quantile_forecasts(
                price_forecast, carbon_forecast
            )

        # Build manually with fallbacks
        if price_forecast is not None:
            price_p50, price_p90 = validate_quantiles(
                price_forecast.p50, price_forecast.p90
            )
            price_dict = {"p50": round(price_p50, 2), "p90": round(price_p90, 2)}
            model_type = price_forecast.model_type
        else:
            price_dict = {
                "p50": round(self.default_price_p50, 2),
                "p90": round(self.default_price_p50 * self.p90_uplift, 2),
            }
            model_type = "partial_fallback"

        if carbon_forecast is not None:
            carbon_p50, carbon_p90 = validate_quantiles(
                carbon_forecast.p50, carbon_forecast.p90
            )
            carbon_dict = {"p50": round(carbon_p50, 1), "p90": round(carbon_p90, 1)}
        else:
            carbon_dict = {
                "p50": round(self.default_carbon_p50, 1),
                "p90": round(self.default_carbon_p50 * self.p90_uplift, 1),
            }
            model_type = "partial_fallback"

        return DecisionForecast(
            energy_cost=price_dict,
            carbon=carbon_dict,
            model_meta={
                "model_type": model_type,
                "trained_at": datetime.utcnow().isoformat(),
                "features_version": "v1",
            },
        )

    def package_legacy(
        self,
        price_forecast: Optional[PriceForecast] = None,
        carbon_forecast: Optional[CarbonForecast] = None,
    ) -> DecisionForecast:
        """Package legacy mean/std forecasts.

        Args:
            price_forecast: Optional legacy price forecast
            carbon_forecast: Optional legacy carbon forecast

        Returns:
            DecisionForecast with derived quantiles
        """
        if price_forecast is not None and carbon_forecast is not None:
            return DecisionForecast.from_legacy_forecasts(
                price_forecast, carbon_forecast
            )

        return DecisionForecast.fallback(
            self.default_price_p50,
            self.default_carbon_p50,
            self.p90_uplift,
        )


# ============================================================================
# INLINE VALIDATION
# ============================================================================
# Run with: python -c "from aurelius.forecasting.uncertainty import _run_validation; _run_validation()"
#
# NOTE: _run_validation() confirms correctness of packaging logic in isolation.
# Backward-compatibility alias – replay.py imports UncertaintyEstimator
UncertaintyEstimator = ForecastPackager

# Integration checks must be run separately to verify:
# - Forecasting enabled -> optimizer decisions unchanged
# - Dry-run execution adapters unchanged (except logging forecasts)

def _run_validation():
    """Validate forecast packaging (no optimizer influence)."""
    print("=" * 60)
    print("Forecast Packaging Validation")
    print("=" * 60)

    # Test 1: DecisionForecast from quantile forecasts
    print("\nTest 1: DECISION FORECAST FROM QUANTILES")
    print("-" * 40)
    price_f = PriceQuantileForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        p50=45.0,
        p90=60.0,
    )
    carbon_f = CarbonQuantileForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        p50=200.0,
        p90=280.0,
    )
    forecast = DecisionForecast.from_quantile_forecasts(price_f, carbon_f)
    print(f"  Energy cost: {forecast.energy_cost}")
    print(f"  Carbon: {forecast.carbon}")
    print(f"  Model meta: {forecast.model_meta}")
    assert forecast.energy_cost["p90"] >= forecast.energy_cost["p50"]
    assert forecast.carbon["p90"] >= forecast.carbon["p50"]
    print("  Invariants satisfied: PASS")

    # Test 2: Invariant enforcement
    print("\nTest 2: INVARIANT ENFORCEMENT (p90 >= p50)")
    print("-" * 40)
    bad_price = PriceQuantileForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        p50=60.0,
        p90=50.0,  # Invalid: p90 < p50
    )
    bad_carbon = CarbonQuantileForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        p50=300.0,
        p90=250.0,  # Invalid: p90 < p50
    )
    fixed = DecisionForecast.from_quantile_forecasts(bad_price, bad_carbon)
    print(f"  Input: price p50=60, p90=50 -> Output: {fixed.energy_cost}")
    print(f"  Input: carbon p50=300, p90=250 -> Output: {fixed.carbon}")
    assert fixed.energy_cost["p90"] >= fixed.energy_cost["p50"]
    assert fixed.carbon["p90"] >= fixed.carbon["p50"]
    print("  Invariants corrected: PASS")

    # Test 3: Fallback forecast
    print("\nTest 3: FALLBACK FORECAST")
    print("-" * 40)
    fallback = DecisionForecast.fallback()
    print(f"  Fallback energy: {fallback.energy_cost}")
    print(f"  Fallback carbon: {fallback.carbon}")
    assert fallback.model_meta["model_type"] == "fallback"
    print("  Fallback works: PASS")

    # Test 4: Legacy forecast conversion
    print("\nTest 4: LEGACY FORECAST CONVERSION")
    print("-" * 40)
    legacy_price = PriceForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        mean=50.0,
        std=10.0,
    )
    legacy_carbon = CarbonForecast(
        timestamp=datetime(2025, 1, 1, 12),
        region="us-west",
        mean=400.0,
        std=50.0,
    )
    converted = DecisionForecast.from_legacy_forecasts(legacy_price, legacy_carbon)
    print(f"  Legacy mean=50, std=10 -> {converted.energy_cost}")
    print(f"  Expected p90 ≈ 50 + 1.28*10 = 62.8")
    assert converted.energy_cost["p90"] >= converted.energy_cost["p50"]
    print("  Legacy conversion works: PASS")

    # Test 5: ForecastPackager
    print("\nTest 5: FORECAST PACKAGER")
    print("-" * 40)
    packager = ForecastPackager()
    packaged = packager.package(price_f, carbon_f)
    print(f"  Packaged: {packaged.to_dict()}")
    assert packaged.energy_cost["p90"] >= packaged.energy_cost["p50"]
    print("  Packaging works: PASS")

    # Test 6: Partial fallback
    print("\nTest 6: PARTIAL FALLBACK")
    print("-" * 40)
    partial = packager.package(price_f, None)  # Only price provided
    print(f"  Partial (price only): {partial.to_dict()}")
    assert partial.model_meta["model_type"] == "partial_fallback"
    assert partial.carbon["p90"] >= partial.carbon["p50"]
    print("  Partial fallback works: PASS")

    # Test 7: Output contract field names
    print("\nTest 7: OUTPUT CONTRACT FIELD NAMES")
    print("-" * 40)
    d = packaged.to_dict()
    assert "energy_cost" in d, "Must use 'energy_cost' not 'cost'"
    assert "carbon" in d, "Must have 'carbon' key"
    assert "p50" in d["energy_cost"], "energy_cost must have p50"
    assert "p90" in d["energy_cost"], "energy_cost must have p90"
    assert "p50" in d["carbon"], "carbon must have p50"
    assert "p90" in d["carbon"], "carbon must have p90"
    print("  Output contract keys correct: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
