# Developer Guide

Engineering reference for running Aurelius locally: setup, the command-line
interface, tests, benchmarks, and repository layout. Aurelius is implemented in
Python (3.11+).

## Setup

```bash
git clone https://github.com/fnstggl/energy2.git
cd energy2

# Runtime dependencies
pip install -r aurelius/requirements.txt

# Development / test dependencies (pytest, httpx, etc.)
pip install -r aurelius/requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in credentials as needed. CAISO requires
no key. PJM and ERCOT require API keys for live fetches; the repository ships
with real historical data so most workflows run without any credentials.

## Command-line interface

The CLI is invoked as `python -m aurelius.cli <command>`.

| Command | Purpose |
|---------|---------|
| `backtest` | Leakage-free walk-forward backtest on real or supplied data |
| `shadow run` / `realize` / `report` | Record live decisions, realize against settlement, compare |
| `roi` | Customer ROI projection from monthly spend and workload mix |
| `simulate` | Synthetic-scenario simulation (development/demo only) |
| `generate-data` | Generate synthetic data files |
| `robustness-test` | Stress the optimizer across scenarios |
| `show-schema` | Print the persistence schema |

### Backtest

```bash
python -m aurelius.cli backtest \
  --price-file data/q12026_3region_dam.csv \
  --rt-price-file data/q12026_3region_rt.csv \
  --regions us-west,us-east,us-south \
  --jobs-file customer_workload_trace.csv \
  --forecaster ml_quantile \
  --train-days 30 \
  --eval-days 7
```

`--jobs-file` accepts a customer CSV (auto-detected) or a JSON trace. The
minimum CSV schema is `job_id,workload_type,submit_time,duration_hours`; a
sample is at `data/fixtures/sample_customer_workload_trace.csv`. Omitting
`--jobs-file` generates synthetic jobs for testing.

### Shadow mode

```bash
# 1. Record decisions (no workloads executed)
python -m aurelius.cli shadow run \
  --price-file <da_prices.csv> --regions us-west,us-east,us-south \
  --jobs-file <trace.csv> --forecaster ml_quantile \
  --output-dir reports/shadow/

# 2. After the settlement window closes, realize against RT prices
python -m aurelius.cli shadow realize \
  --decisions-file reports/shadow/decisions_<ts>.jsonl \
  --rt-price-file <rt_prices.csv> \
  --output-file reports/shadow/realized_<ts>.jsonl

# 3. Compare predicted vs. realized
python -m aurelius.cli shadow report \
  --decisions-file reports/shadow/realized_<ts>.jsonl \
  --output-dir reports/shadow/
```

### ROI projection

```bash
python -m aurelius.cli roi --monthly-cost 500000
python -m aurelius.cli roi --monthly-cost 1000000 \
  --workload-mix '{"training":0.5,"llm_batch_inference":0.3,"realtime_inference":0.2}' \
  --contract-months 24 --output reports/roi_projection.json
```

## Running tests

```bash
python -m pytest tests/ --tb=short        # full suite (excludes live API tests)
python -m pytest tests/test_leakage_audit.py -v
python -m pytest tests/test_safety_gate.py -v
python -m pytest tests/test_shadow_mode.py -v
```

Live-market tests under `tests/live/` require real credentials and are skipped
by default.

## Running benchmarks

```bash
# Smoke test (CI; < 1 min, bundled data)
python benchmarks/run_benchmark.py --quick

# Validated configuration
python benchmarks/run_benchmark.py \
  --region-combo caiso_pjm_ercot_da_rt \
  --forecaster ml_quantile --train-days 30 --num-jobs 100

# Upper-bound diagnostic
python benchmarks/run_benchmark.py \
  --region-combo caiso_pjm_ercot_da_rt --forecaster ml_quantile --oracle

# Regression comparison against an archived run
python benchmarks/compare_against_previous.py \
  --current benchmarks/results/<new>.json \
  --previous benchmarks/results/<archived>.json
```

Optional decision signals are supplied with `--carbon-file`, `--queue-file`
(with `--queue-delay-cost`), and `--gpu-file` (with `--gpu-health-cost`). Queue
and GPU fixtures in `data/` are synthetic and excluded from savings claims.

Fresh market data can be fetched with `scripts/fetch_caiso_pjm_prices.py`,
`scripts/fetch_watttime_carbon.py`, and `scripts/fetch_weather_data.py`
(credentials apply per provider).

## REST API

```bash
export AURELIUS_API_KEY=$(openssl rand -hex 32)
uvicorn aurelius.api.app:app --host 0.0.0.0 --port 8000
curl -H "X-API-Key: $AURELIUS_API_KEY" http://localhost:8000/health
```

## Docker

```bash
docker build -f docker/Dockerfile -t aurelius:latest .
docker run --env-file .env -p 8000:8000 aurelius:latest
```

## Continuous integration

GitHub Actions runs lint (ruff), type-check (mypy, non-blocking), the test suite
with coverage, a benchmark smoke test, and a Docker build with a health-check.

## Repository structure

```
aurelius/
  ingestion/      Market, workload, carbon, queue, GPU data → canonical schema
  forecasting/    Quantile price models, calibration, regime, baselines
  optimization/   Scheduler, objective, constraints
  backtesting/    Walk-forward engine, splitter, baselines, evaluator
  safety/         Quantile safety gate (fail-closed)
  shadow/         Live-decision recorder, realizer, report
  roi/            ROI calculator
  execution/      Scheduler adapters (K8s, Slurm, AWS Batch); dry-run default
  reporting/      Savings and HTML reports
  monitoring/     Drift detection
  validation/     Leakage audit, robustness
  ml/             Offline training, model store, artifacts
  api/            FastAPI service
benchmarks/       Harness, matrices, archived results
data/             Real market data + synthetic fixtures
scripts/          Data fetch, dataset build, learning loop
tests/            Unit, integration, and (skipped) live-market tests
```

## Engineering notes

- Determinism: forecasting and optimization are seeded; benchmarks pin window,
  baselines, folds, and seed, and archive outputs.
- Optional-signal weights default to zero, so a price-only deployment is the
  default behavior and added signals are strictly opt-in.
- Persistence is local JSONL by default; a database backend (Postgres/
  TimescaleDB) is configurable but not required for single-node pilots.
