# API-NEEDED: Open-Meteo Weather API

## Provider
Open-Meteo (open-meteo.com)

## Why needed
Weather data is used for Roadmap Phase 3 (Weather & Cooling Intelligence):
- Predict facility cooling load from temperature and humidity
- Estimate PUE (Power Usage Effectiveness) penalty in hot weather
- Detect heat-wave risk periods (ERCOT Texas is especially vulnerable)
- Forecast future grid stress from weather patterns
- Generate weather-to-PUE feature pipeline for the optimizer

Weather is NOT a substitute for GPU telemetry (DCGM).
DCGM measures current GPU state.
Weather predicts future cooling/grid conditions.

## Env var
None required — Open-Meteo is a free, open API with no authentication.

## .env.example entry
```
# Open-Meteo: no API key required for the free tier
# OPEN_METEO_BASE_URL defaults to https://api.open-meteo.com/v1/
# Override only if using a self-hosted instance
# OPEN_METEO_BASE_URL=https://api.open-meteo.com/v1/
```

## Docs URL
https://open-meteo.com/en/docs

## Free tier
- Completely free, no API key, no rate limits (for non-commercial use)
- For commercial use: contact Open-Meteo for a commercial license
- Historical weather: https://open-meteo.com/en/docs/historical-weather-api
- Forecast: https://open-meteo.com/en/docs

## Variables to request for Aurelius
- temperature_2m (°C) — air temperature at 2m above ground
- relativehumidity_2m (%) — humidity
- windspeed_10m (km/h) — wind cooling effect
- apparent_temperature (°C) — feels-like temperature (cooling load proxy)
- precipitation (mm) — extreme weather signal

## DC locations to monitor
| Region    | Location          | Latitude  | Longitude  |
|-----------|-------------------|-----------|------------|
| us-west   | Sacramento, CA    | 38.55     | -121.47    |
| us-east   | Ashburn, VA       | 39.04     | -77.49     |
| us-south  | Dallas, TX        | 32.78     | -96.80     |

## Planned implementation
- `aurelius/ingestion/weather.py` (not yet implemented)
- Used by Phase 3 PUE/cooling optimizer

## Is live test required for CI?
No. Weather integration is Roadmap Phase 3. When implemented, live
tests will be optional (free API, no credentials needed).
