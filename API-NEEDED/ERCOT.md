# API-NEEDED: ERCOT API (Texas Grid)

## Provider
ERCOT (Electric Reliability Council of Texas) — public API via api.ercot.com

## Why needed
ERCOT covers ~90% of Texas electricity. US-South is a critical benchmark region
because ERCOT prices are highly volatile (extreme weather events, grid stress)
and often anti-correlated with CAISO/PJM — making multi-region migration
especially valuable.

Without live ERCOT credentials, US-South data must be refreshed manually.

## Env vars
```
ERCOT_SUBSCRIPTION_KEY=<your-subscription-key>
ERCOT_USERNAME=<your-username>
ERCOT_PASSWORD=<your-password>
```

## .env.example entry
```
# ERCOT API credentials
# Register at https://www.ercot.com/services/api
# Free for market participants and researchers
ERCOT_SUBSCRIPTION_KEY=
ERCOT_USERNAME=
ERCOT_PASSWORD=
```

## Docs URL
https://developer.ercot.com/

## Registration
1. Go to https://developer.ercot.com/
2. Create a free developer account
3. Subscribe to the "Public" or "Market" plan
4. Retrieve your subscription key from the developer portal

## What data is available
- Day-ahead settlement point prices (SPP) — hourly
- Real-time settlement point prices — 15-minute intervals
- Houston Hub (main pricing point used by Aurelius)

## Pricing
Free for developer access. Rate limits on live endpoints.

## Used by
- `aurelius/ingestion/grid_apis/ercot.py` — `ERCOTPriceProvider` and `ERCOTRealtimePriceProvider`
- CLI: `aurelius backtest --price-provider ercot`
- CLI: `aurelius backtest --price-provider ercot-rt`

## Live test
```bash
ERCOT_SUBSCRIPTION_KEY=<key> ERCOT_USERNAME=<user> ERCOT_PASSWORD=<pass> \
  python -m pytest tests/live/ -k ercot -v
```

## Is live test required for CI?
No. Live tests are skipped when ERCOT credentials are absent.
Contract tests against real downloaded CSV data always run.
