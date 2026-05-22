# API-NEEDED: PJM Data Miner 2

## Provider
PJM Interconnection — PJM Data Miner 2 API

## Why needed
PJM covers the US-East region (roughly 65 million people, largest US grid).
Aurelius uses PJM day-ahead LMP prices to:
- plan job schedules against US-East grid pricing
- compute realized cost for RT-exposed customers (PJM real-time prices)
- provide the primary US-East region benchmark datapoint

Without PJM live ingest, US-East data must be refreshed manually from CSV files.

## Env var
```
PJM_API_KEY=<your-key>
```

## .env.example entry
```
# PJM Data Miner 2 API key
# Register at https://dataminer2.pjm.com/
# Free for registered users (no cost tier limits for historical data)
PJM_API_KEY=
```

## Docs URL
https://dataminer2.pjm.com/

## Registration
1. Go to https://dataminer2.pjm.com/
2. Create a free account
3. Generate an API subscription key
4. Set PJM_API_KEY in your .env file

## Pricing
Free for registered users. Rate limits apply to live/streaming endpoints.
Historical data (day-ahead LMP) is unlimited.

## Used by
- `aurelius/ingestion/grid_apis/pjm.py` — `PJMPriceProvider` (DA) and `PJMRealtimePriceProvider` (RT)
- CLI: `aurelius backtest --price-provider pjm`
- CLI: `aurelius backtest --price-provider pjm-rt`

## Live test
```bash
PJM_API_KEY=<key> python -m pytest tests/live/ -k pjm -v
```

## Is live test required for CI?
No. Live tests are skipped when PJM_API_KEY is not set.
Contract tests against mock data always run.
