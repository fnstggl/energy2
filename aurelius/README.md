# Aurelius v1

**Predictive Control Layer for Energy-Constrained Batch Compute Systems**

Aurelius is a comprehensive simulator + optimizer that proves the economic value of foresight in scheduling batch compute workloads. It answers a fundamental question: *How much money and carbon could we save if we scheduled jobs intelligently based on predicted energy prices?*

---

## What Aurelius Does

Given:
- Variable future energy prices
- Carbon intensity signals
- Batch compute jobs with deadlines and flexibility
- Power constraints and throttling options
- Multiple simulated regions

Aurelius decides:
- **When** jobs should run (time shifting)
- **Where** they should run (region routing)
- **How fast** they should run (power throttling)

To minimize:
- Total energy cost (primary objective)
- Carbon emissions (secondary objective)
- Risk from forecast uncertainty

While respecting:
- Job deadlines
- Power caps
- Regional capacity constraints

---

## What Aurelius Does NOT Do

- ❌ No real-time control or live execution
- ❌ No Kubernetes integration
- ❌ No Slurm adapters
- ❌ No user dashboards or UI
- ❌ No authentication or user management
- ❌ No deep learning or black-box models

This is a **shadow-mode simulator** for proving value before production deployment.

---

## Quick Start

### Installation

```bash
# Clone the repository
cd aurelius

# Install dependencies
pip install -r requirements.txt

# Optional: Install with development dependencies
pip install -r requirements-dev.txt
```

### Run a Simulation (CLI)

```bash
# Basic simulation with defaults (50 jobs, 1 week, greedy optimizer)
python -m aurelius.cli simulate

# Custom simulation
python -m aurelius.cli simulate \
    --jobs 100 \
    --hours 72 \
    --method local_search \
    --price-scenario volatile

# Save results to file
python -m aurelius.cli simulate --output results.json
```

### Run a Simulation (API)

```bash
# Start the API server
uvicorn aurelius.api.app:app --host 0.0.0.0 --port 8000

# Run a simulation via API
curl -X POST http://localhost:8000/simulate \
    -H "Content-Type: application/json" \
    -d '{"num_jobs": 50, "optimization_method": "greedy"}'
```

### Run a Simulation (Python)

```python
from aurelius.simulation.replay import SimulationReplay, SimulationConfig
from aurelius.models import OptimizationConfig

# Configure optimization weights
opt_config = OptimizationConfig(
    alpha=1.0,   # energy cost weight
    beta=0.1,    # carbon cost weight
    gamma=0.05,  # uncertainty risk penalty
)

# Configure simulation
sim_config = SimulationConfig(
    num_jobs=50,
    duration_hours=168,
    optimization_method="local_search",
    optimization_config=opt_config,
)

# Run simulation
replay = SimulationReplay()
results = replay.run(sim_config)

# Print results
print(f"Cost savings: {results['summary']['cost_savings_pct']:.1f}%")
print(f"Carbon savings: {results['summary']['carbon_savings_pct']:.1f}%")
```

---

## Architecture

```
aurelius/
├── ingestion/           # Data loading and generation
│   ├── energy_prices.py # Energy price ingestion
│   └── job_logs.py      # Job workload ingestion
├── forecasting/         # Simple, explainable forecasters
│   ├── price_model.py   # Price forecasting
│   ├── carbon_model.py  # Carbon intensity forecasting
│   ├── uncertainty.py   # Uncertainty estimation
│   └── baseline.py      # Baseline forecasting methods
├── optimization/        # Core scheduling optimizer
│   ├── scheduler.py     # Main optimization solver
│   ├── constraints.py   # Constraint definitions
│   └── objective.py     # Objective function
├── simulation/          # Scenario simulation and comparison
│   ├── replay.py        # Simulation orchestration
│   ├── compare.py       # Baseline vs optimized comparison
│   └── metrics.py       # Performance metrics
├── api/                 # REST API
│   └── app.py           # FastAPI application
├── models.py            # Core data models
├── database.py          # Supabase integration
└── cli.py               # Command-line interface
```

---

## Optimization Model

### Decision Variables

For each job j:
- `start_time[j]` - When to start the job
- `region[j]` - Which region to run in
- `power_fraction[j]` - Power level (0.5 to 1.0)

### Objective Function

```
Minimize: α * energy_cost + β * carbon_cost + γ * risk_penalty
```

Where:
- `energy_cost = Σ(price × power × time)` for all jobs
- `carbon_cost = Σ(carbon_intensity × power × time)` for all jobs
- `risk_penalty` increases when scheduling during high-uncertainty periods

Default weights:
- `α = 1.0` (energy cost)
- `β = 0.1` (carbon cost)
- `γ = 0.05` (risk penalty)

### Constraints

1. **Time window**: `earliest_start ≤ start_time ≤ latest_start`
2. **Deadline**: `start_time + adjusted_runtime ≤ deadline`
3. **Power cap**: `Σ power[jobs in region at time t] ≤ region_power_cap`
4. **Power range**: `min_power_fraction ≤ power_fraction ≤ 1.0`

### Throttling Model

When a job runs at reduced power:
```
adjusted_runtime = base_runtime / power_fraction
```

Example: A 4-hour job at 50% power runs for 8 hours but uses half the power each hour.

---

## Forecasting

Aurelius uses simple, explainable forecasting methods:

### Price Forecasting
- Hour-of-day seasonality factors
- Day-of-week adjustments
- Rolling average baseline
- Uncertainty grows with forecast horizon

### Carbon Forecasting
- Similar seasonal patterns
- Accounts for solar hours (lower carbon midday)
- Regional baseline differences

### Uncertainty Estimation
- Coefficient of variation (std/mean)
- Combined price + carbon uncertainty
- Risk penalty formula: `penalty = base * (1 + uncertainty)^2`

---

## Interpreting Results

### Sample Output

```
============================================================
AURELIUS SIMULATION RESULTS
============================================================

BASELINE SCHEDULE:
  Energy Cost:    $4,523.45
  Carbon (kg):    892.34
  Peak Power:     2,450.0 kW
  Makespan:       72.0 hours

OPTIMIZED SCHEDULE:
  Energy Cost:    $3,891.23
  Carbon (kg):    756.12
  Peak Power:     1,890.0 kW
  Jobs Throttled: 12
  Jobs Shifted:   28

SAVINGS:
  Cost Savings:   $632.22 (14.0%)
  Carbon Saved:   136.22 kg (15.3%)
  Peak Reduced:   560.0 kW (22.9%)
============================================================
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `cost_savings_pct` | Percentage reduction in energy cost |
| `carbon_savings_pct` | Percentage reduction in CO2 emissions |
| `jobs_throttled` | Jobs running at reduced power |
| `jobs_shifted` | Jobs delayed from earliest start |
| `peak_power_reduction` | Reduction in maximum concurrent power |

---

## Database Schema

Aurelius uses Supabase for persistence. Tables:

### `energy_prices`
| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMPTZ | Hour (UTC) |
| region | TEXT | Geographic region |
| price_per_mwh | DECIMAL | Price in $/MWh |

### `carbon_intensity`
| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMPTZ | Hour (UTC) |
| region | TEXT | Geographic region |
| gco2_per_kwh | DECIMAL | Grams CO2 per kWh |

### `jobs`
| Column | Type | Description |
|--------|------|-------------|
| job_id | TEXT | Unique identifier |
| submit_time | TIMESTAMPTZ | When submitted |
| runtime_hours | DECIMAL | Base runtime |
| deadline | TIMESTAMPTZ | Must finish by |
| power_kw | DECIMAL | Power consumption |
| region_options | TEXT[] | Allowed regions |

### `simulations`
| Column | Type | Description |
|--------|------|-------------|
| run_id | TEXT | Unique run ID |
| baseline_cost | DECIMAL | Baseline energy cost |
| optimized_cost | DECIMAL | Optimized energy cost |
| savings_pct | DECIMAL | Cost savings percentage |
| created_at | TIMESTAMPTZ | When run |

To create the schema:
```bash
python -m aurelius.cli show-schema
```

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |

### Optimization Config

```python
OptimizationConfig(
    alpha=1.0,              # Energy cost weight
    beta=0.1,               # Carbon cost weight
    gamma=0.05,             # Risk penalty weight
    min_power_fraction=0.5, # Minimum throttle (50%)
    max_power_fraction=1.0, # Maximum power
    region_power_caps={     # Power caps per region
        "us-west": 10000,
        "us-east": 10000,
        "eu-west": 8000,
    },
    default_region="us-west",
)
```

---

## Price Scenarios

For synthetic data generation:

| Scenario | Description |
|----------|-------------|
| `normal` | Typical price patterns |
| `volatile` | High volatility with spikes |
| `low` | Generally low prices |
| `high` | Generally high prices |
| `peak_valley` | Strong peak/off-peak difference |

Carbon scenarios: `normal`, `clean`, `dirty`, `variable`

---

## API Reference

### POST /simulate

Run a simulation.

**Request:**
```json
{
    "num_jobs": 50,
    "duration_hours": 168,
    "regions": ["us-west", "us-east", "eu-west"],
    "optimization_method": "greedy",
    "alpha": 1.0,
    "beta": 0.1,
    "gamma": 0.05,
    "price_scenario": "normal",
    "random_seed": 42
}
```

**Response:**
```json
{
    "run_id": "abc123",
    "baseline_cost": 4523.45,
    "optimized_cost": 3891.23,
    "cost_savings_pct": 14.0,
    "baseline_carbon_kg": 892.34,
    "optimized_carbon_kg": 756.12,
    "carbon_savings_pct": 15.3,
    "jobs_scheduled": 50
}
```

### GET /simulations

List past simulations.

### GET /simulations/{run_id}

Get specific simulation details.

### GET /health

Health check endpoint.

---

## For Pilot Partners

### Expected Results

Typical cost savings range from 10-30% depending on:
- Price volatility (higher volatility = more savings opportunity)
- Job flexibility (more slack = more optimization room)
- Multi-region availability (more regions = more routing options)

### How to Validate

1. **Run with your price data**: Load real price data from your region
2. **Model your workload**: Create job profiles matching your batch workload
3. **Compare scenarios**: Test different optimization strategies
4. **Export results**: Save JSON results for analysis

### Limitations

- This is a simulator, not live control
- Assumes perfect execution of scheduled jobs
- Does not account for queue wait times
- Forecasts are simplified models

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- All data models in `models.py`
- Database access in `database.py`
- Each module is self-contained with clear interfaces

### Adding New Features

1. Forecasting models: Add to `forecasting/`
2. Optimization methods: Add to `optimization/scheduler.py`
3. Constraints: Add to `optimization/constraints.py`
4. Metrics: Add to `simulation/metrics.py`

---

## License

MIT License - See LICENSE file

---

## Contact

For questions about Aurelius or pilot partnerships, please open an issue.
