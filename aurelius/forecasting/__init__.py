"""Forecasting modules for Aurelius.

This package provides probabilistic forecasting for:
- Energy prices (p50/p90 quantiles)
- Carbon intensity (p50/p90 quantiles)

Key components:
- PriceQuantileForecaster: LightGBM quantile regression for prices
- CarbonQuantileForecaster: LightGBM quantile regression for carbon
- BaselineRegressor: Ridge/ElasticNet for deterministic point estimates
- ForecastPackager: Packages forecasts for decision attachment

Usage:
    from aurelius.forecasting import (
        PriceQuantileForecaster,
        CarbonQuantileForecaster,
        ForecastPackager,
        DecisionForecast,
    )

    # Train models
    price_model = PriceQuantileForecaster()
    price_model.fit(historical_prices)

    carbon_model = CarbonQuantileForecaster()
    carbon_model.fit(historical_carbon)

    # Generate forecasts
    price_forecasts = price_model.predict(region, timestamps, recent_prices)
    carbon_forecasts = carbon_model.predict(region, timestamps, recent_carbon)

    # Package for decision attachment
    packager = ForecastPackager()
    decision.forecast = packager.package(price_forecast, carbon_forecast).to_dict()

Output Contract:
    decision.forecast = {
        "energy_cost": {"p50": float, "p90": float},
        "carbon": {"p50": float, "p90": float},
        "model_meta": {
            "model_type": "ridge+lightgbm_quantile",
            "trained_at": "<ISO8601>",
            "features_version": "v1"
        }
    }

IMPORTANT:
- p90 >= p50 ALWAYS (enforced automatically)
- Fallback to baseline when LightGBM unavailable
- Offline batch training only
- Fixed random seeds for determinism
- Forecasting is ADVISORY DATA ONLY - does NOT influence optimizer behavior

VALIDATION vs INTEGRATION:
- Each module's _run_validation() confirms correctness in ISOLATION
- Integration checks must be run separately to verify:
  - Forecasting enabled -> optimizer decisions UNCHANGED
  - Dry-run execution adapters UNCHANGED (except logging forecasts)
"""

# Legacy forecasters (still available for backwards compatibility)
# Simple baseline forecasters
# Baseline regression
from .baseline import (
    BaselineForecaster,
    BaselineRegressionConfig,
    BaselineRegressor,
    generate_carbon_scenario,
    generate_price_scenario,
)
from .carbon_model import (
    CarbonForecast,
    CarbonForecaster,
    CarbonModelConfig,
    CarbonQuantileForecast,
    CarbonQuantileForecaster,
)

# Quantile forecasting
from .price_model import (
    PriceForecast,
    PriceForecaster,
    PriceModelConfig,
    PriceQuantileForecast,
    PriceQuantileForecaster,
)

# Quantile model utilities
from .quantile_model import (
    DEFAULT_SEED,
    QUANTILE_P50,
    QUANTILE_P90,
    ModelMetadata,
    QuantileForecast,
    build_feature_matrix,
    set_deterministic_seed,
    train_lightgbm_quantile,
    validate_quantiles,
)

# Forecast packaging (advisory output only)
from .uncertainty import DecisionForecast, ForecastPackager

__all__ = [
    # Legacy forecasters
    "PriceForecaster",
    "PriceForecast",
    "CarbonForecaster",
    "CarbonForecast",
    "BaselineForecaster",
    # Scenario generators
    "generate_price_scenario",
    "generate_carbon_scenario",
    # Quantile forecasters
    "PriceQuantileForecaster",
    "PriceQuantileForecast",
    "PriceModelConfig",
    "CarbonQuantileForecaster",
    "CarbonQuantileForecast",
    "CarbonModelConfig",
    # Baseline regression
    "BaselineRegressor",
    "BaselineRegressionConfig",
    # Forecast packaging
    "DecisionForecast",
    "ForecastPackager",
    # Quantile model utilities
    "QuantileForecast",
    "ModelMetadata",
    "validate_quantiles",
    "set_deterministic_seed",
    "build_feature_matrix",
    "train_lightgbm_quantile",
    "DEFAULT_SEED",
    "QUANTILE_P50",
    "QUANTILE_P90",
]
