# Aurelius — Local Production-Like Run

This document describes how to run Aurelius in a local environment that mirrors
production behaviour as closely as possible.

---

## Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerised run)
- pip

---

## 1. Clone and install

```bash
git clone https://github.com/fnstggl/energy2.git
cd energy2

# Install runtime dependencies
pip install -r aurelius/requirements.txt

# Install dev/test dependencies (adds pytest, httpx, etc.)
pip install -r aurelius/requirements-dev.txt
```

---

## 2. Configure environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with real values (PJM, ERCOT, WattTime credentials)
# At minimum for US-region optimization, set:
#   PJM_API_KEY=...
#   ERCOT_API_KEY=...
#   WATTTIME_USERNAME=...
#   WATTTIME_PASSWORD=...
#   AURELIUS_API_KEY=...   (generate: openssl rand -hex 32)
```

CAISO requires no credentials (public API).
WattTime free plan covers CAISO (us-west) only; PJM/ERCOT require a paid plan.

---

## 3. Verify installation

```bash
# All tests must pass before any other step
python -m pytest tests/ --tb=short -q

# Expected: 917 passed, 0 failed, 5 skipped
```

---

## 4. Benchmark smoke test

```bash
# Quick smoke test (< 60 seconds, uses pre-existing data files)
python benchmarks/run_benchmark.py --quick

# Full benchmark (5-10 minutes)
python benchmarks/run_benchmark.py \
  --forecaster ml_quantile \
  --train-days 30 \
  --num-jobs 100

# Output: benchmarks/results/benchmark_<timestamp>.json
#         benchmarks/results/summary_<timestamp>.txt
```

---

## 5. Fetch fresh energy price data

```bash
# Fetch CAISO day-ahead prices (public, no key)
python scripts/fetch_caiso_prices.py \
  --start 2026-01-01 --end 2026-03-31 \
  --output data/caiso_recent_dam.csv

# Fetch PJM day-ahead prices (requires PJM_API_KEY)
python scripts/fetch_pjm_prices.py \
  --start 2026-01-01 --end 2026-03-31 \
  --output data/pjm_recent_dam.csv

# Fetch ERCOT day-ahead prices (requires ERCOT_API_KEY)
python scripts/fetch_ercot_prices.py \
  --start 2026-01-01 --end 2026-03-31 \
  --output data/ercot_recent_dam.csv

# Fetch WattTime carbon data for CAISO (requires WATTTIME_USERNAME/PASSWORD)
python scripts/fetch_watttime_carbon.py \
  --start 2026-01-01 --end 2026-03-31 \
  --output data/watttime_carbon_recent.csv
```

---

## 6. Run a walk-forward backtest

```bash
# Single-region backtest (us-west, CAISO)
python -m aurelius.cli backtest \
  --price-file data/caiso_recent_dam.csv \
  --regions us-west \
  --method greedy_migrate \
  --forecaster ml_quantile \
  --train-days 30 \
  --eval-days 7

# Multi-region backtest (3 US regions)
python -m aurelius.cli backtest \
  --price-file data/q12026_3region_dam.csv \
  --rt-price-file data/q12026_3region_rt.csv \
  --regions us-west,us-east,us-south \
  --method greedy_migrate \
  --forecaster ml_quantile \
  --train-days 30 \
  --eval-days 7 \
  --start 2026-01-01 \
  --end 2026-03-10

# With queue-aware routing (requires queue CSV)
python -m aurelius.cli backtest \
  --price-file data/q12026_3region_dam.csv \
  --rt-price-file data/q12026_3region_rt.csv \
  --regions us-west,us-east,us-south \
  --method greedy_migrate \
  --forecaster ml_quantile \
  --queue-file data/queue_q12026_3region.csv \
  --queue-delay-cost 2.0

# With GPU health-aware routing (requires GPU telemetry CSV)
python -m aurelius.cli backtest \
  --price-file data/q12026_3region_dam.csv \
  --rt-price-file data/q12026_3region_rt.csv \
  --regions us-west,us-east,us-south \
  --method greedy_migrate \
  --forecaster ml_quantile \
  --gpu-file data/gpu_q12026_3region.csv \
  --gpu-health-cost 5.0
```

---

## 7. Run oracle diagnostics

Oracle diagnostics compute the theoretical maximum savings if the forecaster
had perfect future knowledge. Use this to diagnose whether the bottleneck is
forecasting quality or structural (region set, workload flexibility, migration cost).

```bash
python benchmarks/run_benchmark.py \
  --region-combo caiso_pjm_ercot_da_rt \
  --forecaster ml_quantile \
  --oracle \
  --num-jobs 100
```

Interpretation:
- If oracle ceiling ≈ ml_quantile actual: forecasting is near-optimal, look elsewhere
- If oracle ceiling >> ml_quantile actual: forecasting is the bottleneck

---

## 8. Start the FastAPI REST service

```bash
# Set auth key in environment
export AURELIUS_API_KEY=your_generated_secret

# Start server
uvicorn aurelius.api.app:app --host 0.0.0.0 --port 8000 --reload

# Verify health
curl -s http://localhost:8000/health

# Run a simulation
curl -s -H "X-API-Key: $AURELIUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jobs": 20, "hours": 48, "method": "greedy_migrate"}' \
  http://localhost:8000/simulate | python -m json.tool
```

---

## 9. Run via Docker

```bash
# Build image
docker build -f docker/Dockerfile -t aurelius:latest .

# Run with env file
docker run \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/benchmarks/results:/app/benchmarks/results \
  aurelius:latest

# Run benchmark inside container
docker run --env-file .env aurelius:latest \
  python benchmarks/run_benchmark.py --quick
```

---

## 10. Run CI pipeline locally

```bash
# Lint (same as GitHub Actions)
ruff check aurelius/ scripts/ tests/

# Type check
mypy aurelius/ --ignore-missing-imports

# Full test suite
python -m pytest tests/ --tb=short

# Benchmark smoke test
python benchmarks/run_benchmark.py --quick
```

---

## 11. Benchmark regression comparison

After making any code change that affects the optimizer or forecaster, compare
results against the last saved benchmark to detect regressions.

```bash
python benchmarks/compare_against_previous.py \
  --current benchmarks/results/benchmark_<new>.json \
  --previous benchmarks/results/benchmark_ml_quantile_v2_3region_q12026_20260522.json
```

This script reports:
- Savings delta per workload vs current_price_only
- Missing price hour changes
- SLA violation delta
- Regression warnings if savings drop unexpectedly

---

## 12. Pilot shadow test workflow

For an enterprise pilot (Tier 1, region/time optimization):

### Step 1 — Customer provides workload trace
The customer exports their last 30-90 days of GPU job submissions as CSV.
Minimum required columns: `job_id, workload_type, submit_time, duration_hours, gpu_count`

### Step 2 — Backtest on customer's trace
```bash
# Supply a customer workload trace CSV (or JSON) via --jobs-file
# See data/fixtures/sample_customer_workload_trace.csv for the column schema.
# Minimum required columns: job_id, workload_type, submit_time, duration_hours
python -m aurelius.cli backtest \
  --price-file data/q12026_3region_dam.csv \
  --rt-price-file data/q12026_3region_rt.csv \
  --regions us-west,us-east,us-south \
  --jobs-file customer_workload_trace.csv \
  --forecaster ml_quantile \
  --train-days 30 \
  --eval-days 7
```

### Step 3 — Review results
```bash
# Latest summary
ls -t benchmarks/results/summary_*.txt | head -1 | xargs cat
```

### Step 4 — Compute ROI estimate
Savings % × customer GPU spend/month = estimated monthly savings.
E.g., 25% × $500k/month = $125k/month.

---

## Environment variables reference

| Variable               | Required for          | Notes                                   |
|------------------------|-----------------------|-----------------------------------------|
| `PJM_API_KEY`          | PJM price data        | Free registration at developer.pjm.com |
| `ERCOT_API_KEY`        | ERCOT price data      | Requires ERCOT CDAT registration        |
| `ERCOT_USERNAME`       | ERCOT price data      |                                         |
| `ERCOT_PASSWORD`       | ERCOT price data      |                                         |
| `WATTTIME_USERNAME`    | Carbon signal         | Free plan: CAISO only                  |
| `WATTTIME_PASSWORD`    | Carbon signal         |                                         |
| `ENTSOE_API_KEY`       | EU prices             | Not yet available                      |
| `ELECTRICITYMAPS_API_KEY` | Carbon (EU/global) | Sandbox only without paid plan         |
| `AURELIUS_API_KEY`     | REST API auth         | Generate: `openssl rand -hex 32`        |
| `PROMETHEUS_URL`       | GPU telemetry (Tier 3)| Customer's Prometheus endpoint          |
| `DCGM_EXPORTER_URL`    | GPU telemetry (Tier 3)| Direct dcgm-exporter /metrics URL       |

---

## Known limitations

1. **Live shadow mode** not yet implemented.
   Current workaround: use historical backtest as shadow proxy.

2. **WattTime carbon data** available for CAISO (us-west) only on free plan.
   PJM/ERCOT carbon requires a paid WattTime subscription.

3. **EU expansion** requires ENTSO-E API token (not yet obtained).

4. **Database persistence** is JSONL-only.
   For multi-instance or long-running pilots, a Postgres/TimescaleDB setup is recommended.

5. **GPU-level placement** (Tier 3) requires customer to operate DCGM + Prometheus.
   Without those, GPU health signals are fixture-only.
