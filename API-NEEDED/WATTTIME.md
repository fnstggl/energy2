# API-NEEDED: WattTime API

## Provider
WattTime (watttime.org)

## Why needed
WattTime provides MOER (Marginal Operating Emissions Rate) data — the carbon
intensity of the electricity being consumed at the margin right now. Unlike
average carbon intensity, MOER measures whether your consumption causes more
coal or more renewables to be dispatched. This is the most accurate signal
for carbon-aware optimization.

Aurelius uses WattTime as an alternative/complement to Electricity Maps for:
- US-region carbon signals (CAISO/PJM/ERCOT)
- Marginal emissions forecasting
- Real-time carbon dispatch signals

## Env vars
```
WATTTIME_USERNAME=<your-username>
WATTTIME_PASSWORD=<your-password>
```

## .env.example entry
```
# WattTime API credentials
# Register at https://www.watttime.org/api-documentation/
# Free tier: basic access. Paid tier for forecasts and historical data.
WATTTIME_USERNAME=
WATTTIME_PASSWORD=
```

## Docs URL
https://www.watttime.org/api-documentation/

## Registration
1. Go to https://www.watttime.org/api-documentation/
2. Register a free account via the API registration endpoint
3. Use WATTTIME_USERNAME / WATTTIME_PASSWORD for all API calls
   (Bearer token fetched automatically per session)

## Free tier coverage
- Real-time MOER for US balancing authorities (CAISO, PJM, ERCOT, etc.)
- Historical MOER is paid-tier only
- Forecasts are paid-tier only

## Used by
- `aurelius/ingestion/grid_apis/watttime.py` — `WattTimeCarbonProvider`
- CLI: `aurelius backtest --carbon-provider watttime`

## Live test
```bash
WATTTIME_USERNAME=<user> WATTTIME_PASSWORD=<pass> \
  python -m pytest tests/live/ -k watttime -v
```

## Is live test required for CI?
No. Live tests skipped when credentials absent.

## Alternative
Electricity Maps API (see API-NEEDED/ELECTRICITYMAPS.md) — broader global coverage.
