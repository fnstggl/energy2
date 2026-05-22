# API-NEEDED: Electricity Maps API

## Provider
Electricity Maps (electricitymaps.com)

## Why needed
Electricity Maps provides real-time and historical carbon intensity data
(gCO2/kWh) for electricity grids worldwide. Aurelius uses this for:
- carbon-weighted optimization (minimize carbon cost alongside energy cost)
- carbon intensity forecasting
- carbon constraint enforcement (hard cap on gCO2/kWh)

Without carbon data, Aurelius optimizes on energy price only. Carbon signal
is optional but enables carbon-aware SLA classes and carbon reporting.

## Env var
```
ELECTRICITYMAPS_API_KEY=<your-api-key>
```

## .env.example entry
```
# Electricity Maps API key
# Register at https://api.electricitymap.org/
# Free tier: limited requests/month. Paid tier for production.
ELECTRICITYMAPS_API_KEY=
```

## Docs URL
https://docs.electricitymap.org/

## Registration
1. Go to https://api.electricitymap.org/
2. Create a free account (limited requests/month)
3. Or contact sales for a commercial plan
4. Retrieve your API key from the dashboard

## Free tier limits
- 1,000 requests/month on free tier
- For production/continuous use, a paid plan is required
- Historical data may require paid tier

## Coverage
- ~70 countries / 200+ regions
- Real-time carbon intensity
- Historical data (varies by region)
- Forecasts (paid tier)

## Used by
- `aurelius/ingestion/grid_apis/electricitymaps.py`
- CLI: `aurelius backtest --carbon-provider electricitymaps`

## Live test
```bash
ELECTRICITYMAPS_API_KEY=<key> python -m pytest tests/live/ -k electricitymaps -v
```

## Is live test required for CI?
No. Carbon data is optional in all backtests (--carbon-provider none).
Live tests skipped when key absent.

## Alternative
WattTime API (see API-NEEDED/WATTTIME.md) — MOER signal, US-focused.
