# API-NEEDED: ENTSO-E Transparency Platform

## Provider
ENTSO-E (European Network of Transmission System Operators for Electricity)

## Why needed
ENTSO-E covers EU electricity markets (Germany, France, Spain, Netherlands, etc.).
EU expansion is Roadmap Phase 2. Without ENTSO-E, Aurelius cannot benchmark or
optimize EU-region jobs.

ENTSO-E provides:
- Day-ahead prices per bidding zone (hourly)
- Actual total load
- Generation by fuel type
- Cross-border flows

Aurelius uses day-ahead prices ONLY — not load or generation, which are not prices.

## Env var
```
ENTSOE_API_KEY=<your-security-token>
```

## .env.example entry
```
# ENTSO-E Transparency Platform security token
# Register at https://transparency.entsoe.eu/
# Free for registered users
ENTSOE_API_KEY=
```

## Docs URL
https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

## Registration
1. Go to https://transparency.entsoe.eu/
2. Register a free account
3. Email transparency@entsoe.eu requesting Web API access
4. Receive your security token by email (usually within 1-3 business days)

## Key: bidding zone codes used by Aurelius
| Region ID   | ENTSO-E EIC code | Description          |
|-------------|------------------|----------------------|
| eu-de       | 10Y1001A1001A83F | Germany (DE-LU)      |
| eu-fr       | 10YFR-RTE------C | France               |
| eu-nl       | 10YNL----------L | Netherlands          |
| eu-es       | 10YES-REE------0 | Spain                |

## Pricing
Free for registered users. Rate limits: 400 requests/minute.

## Used by
- `aurelius/ingestion/grid_apis/entsoe.py` — `ENTSOEPriceProvider`
- CLI: `aurelius backtest --price-provider entsoe`

## Live test
```bash
ENTSOE_API_KEY=<token> python -m pytest tests/live/ -k entsoe -v
```

## Is live test required for CI?
No. Live tests skipped when ENTSOE_API_KEY absent.
