# Energy System Map — the existing robust energy arbitrage / forecasting engine

> **Read `docs/RESULTS.md` first.** This document maps the *existing* energy
> engine so future changes can preserve it. The engine has already been
> optimized and is the **source of truth** for energy decisions. This PR adds a
> thin adapter around it; it does **not** modify the core (see
> "Core energy code — do not modify" below).

## 1. What the energy system is

The energy system is the **Job-trace optimizer**: it takes a list of compute
`Job`s plus per-region energy price (and carbon) time series and produces a
`ScheduleDecision` per job (region, start time, power level, optional mid-job
migration segments). It performs:

- **regional arbitrage** (place a job in the cheapest eligible region),
- **temporal shifting** (defer flexible jobs into cheaper hours within slack),
- **power throttling** (run slower at lower price when allowed),
- **DA/RT basis** handling (plan on day-ahead price, settle on real-time),
- **carbon-aware** placement (optional carbon weight / constraint),
- **walk-forward backtesting** with leakage-free train/eval folds and
  probabilistic price/carbon **forecasting**.

It is distinct from the **cluster simulator / constraint-aware engine**
(`aurelius/constraints/`, `aurelius/simulation/cluster/`), which operates on
live `ClusterState` snapshots of inference services. This PR connects the two:
the energy engine's recommendations are routed through the constraint-aware
SLA/KPI/risk gates via `aurelius/constraints/energy_adapter.py`.

## 2. Files / modules

| Module | Role |
|---|---|
| `aurelius/optimization/scheduler.py` | **`JobScheduler`** — the optimizer. `solve()` (greedy / local_search / milp / *_migrate / *_migrate_dp), `create_baseline_schedule()`, SLA-aware correction. |
| `aurelius/optimization/objective.py` | `ObjectiveFunction` — energy + carbon + risk + SLA + queue objective and `ObjectiveComponents`. |
| `aurelius/optimization/constraints.py` | `ConstraintBuilder` — feasible-start ranges, deadline / power / region constraints. |
| `aurelius/backtesting/engine.py` | **`BacktestEngine`** — leakage-free walk-forward backtest; builds seasonal-naive / ML / oracle forecasts; runs the optimizer + baselines per fold; measures forecast quality. |
| `aurelius/backtesting/baselines.py` | Deterministic baseline policies: `fifo`, `peak_blind_asap`, `latency_first`, `closest_region`, `fixed_primary_region`, **`current_price_only`**, `round_robin`. |
| `aurelius/backtesting/evaluator.py` | `evaluate_schedule()` — realized cost/carbon scored on **actual** (settlement) prices. |
| `aurelius/backtesting/splitter.py` | `TemporalSplitter` — train/eval fold windows (no leakage). |
| `aurelius/forecasting/*` | Price / carbon forecasters: `PriceQuantileForecaster`, `CarbonQuantileForecaster`, `BaselineRegressor`, regime / spread-risk / uncertainty packaging. Forecasting is advisory input to the optimizer. |
| `aurelius/models.py` | Core data models: `Job`, `EnergyPrice`, `CarbonIntensity`, `ScheduleDecision`, `ScheduleSegment`, `OptimizationConfig`, `SimulationResult`. |

## 3. Public entry points

```python
from aurelius.optimization.scheduler import JobScheduler
from aurelius.models import OptimizationConfig

scheduler = JobScheduler(OptimizationConfig())
result = scheduler.solve(jobs, price_data, carbon_data, method="greedy")
#   -> SchedulerResult(schedule=list[ScheduleDecision], objective, violations, ...)
baseline = scheduler.create_baseline_schedule(jobs)  # ASAP / default-region
```

```python
from aurelius.backtesting.engine import BacktestEngine
engine = BacktestEngine(method="greedy", train_days=30, eval_days=7,
                        price_forecaster_cls=None)
rounds = engine.run(jobs, price_df, carbon_df, settle_price_df=rt_df)
#   -> list[BacktestRound] (optimizer vs baseline realized metrics per fold)
```

The CLI wrapper is `aurelius/cli.py::cmd_backtest` (`--price-provider`,
`--num-jobs`, `--method`, `--forecaster`, …).

## 4. Input schema

- **`jobs`**: `list[Job]` (see `aurelius/models.py::Job`). Key fields:
  `job_id, submit_time, runtime_hours, deadline, power_kw, earliest_start,
  region_options, gpu_count, workload_type, migration_cost_hours` (None ⇒ cannot
  migrate), `allowed_regions / forbidden_regions` (data residency).
- **`price_data`**: `{region: {hour(datetime, UTC): price_per_mwh(float)}}`.
- **`carbon_data`**: `{region: {hour: gco2_per_kwh}}` (may be empty).
- Optional: `risk_data`, `queue_data`, `gpu_health_data` keyed the same way.

## 5. Output schema

`ScheduleDecision(job_id, start_time, region, power_fraction,
actual_runtime_hours, forecast?, segments?)`; `end_time`, `migration_count`,
`all_segments` are derived. `evaluate_schedule()` returns `RealizedMetrics`
(`total_energy_cost_usd, total_carbon_gco2, missing_price_hours, …`).

## 6. Supported regions / data sources

| ISO | Internal region | DA file | RT file |
|---|---|---|---|
| CAISO | `us-west` | `data/caiso_us_west_dam.csv` | `data/caiso_us_west_rt.csv` |
| PJM | `us-east` | `data/pjm_us_east_dam.csv` | `data/pjm_us_east_rt.csv` |
| ERCOT | `us-south` | `data/ercot_us_south_dam.csv` | `data/ercot_us_south_rt.csv` |

CSV schema: `timestamp, region, price_per_mwh, currency, source,
source_granularity, fetched_at`. (ENTSO-E / EU regions exist in the ingestion
layer but are **out of scope** here.)

## 7. Existing tests / backtests

- `tests/test_backtesting.py`, `tests/test_e2e_backtest.py` — BacktestEngine,
  leakage proof, settlement-price scoring.
- `tests/test_baselines.py` — baseline policies.
- `tests/test_caiso_pjm.py`, `tests/test_ercot.py` — ISO ingestion + provider
  validation.
- `tests/test_scheduler.py`, `tests/test_migration*.py`, `tests/test_sla_*` —
  optimizer behavior, migration, SLA correction.
- `tests/backtesting/test_ml_forecaster_backtest.py`, `tests/test_forecaster_v5.py`,
  `tests/test_per_region_forecaster.py` — forecasting.

### Known benchmark commands

```bash
python -m aurelius.cli backtest --price-provider caiso --num-jobs 20 --method greedy
python scripts/run_canonical_backtests.py          # NEW canonical 1000-job suite
```

## 8. Core energy code — DO NOT MODIFY

The following are the **source of truth**. This integration PR must not change
their algorithms or constants. Any future change here is, by definition, a
change to the energy engine and must be justified explicitly; the guard test
`tests/test_energy_core_preservation.py` pins their output:

- `aurelius/optimization/scheduler.py` (`JobScheduler` + solvers)
- `aurelius/optimization/objective.py` (`ObjectiveFunction`)
- `aurelius/optimization/constraints.py` (`ConstraintBuilder`)
- `aurelius/backtesting/engine.py` (`BacktestEngine`)
- `aurelius/backtesting/baselines.py` (`current_price_only`, `fifo`, …)
- `aurelius/backtesting/evaluator.py` (`evaluate_schedule`)
- `aurelius/forecasting/*` (price / carbon forecasters)
- `aurelius/models.py` (energy data models)

## 9. How this PR integrates without touching the core

```
JobScheduler.solve()  ──►  EnergyArbitrageAdapter  ──►  constraint-aware gates
 (UNCHANGED energy engine)   (thin wrapper, no energy   (eligibility / destination /
                              logic; reads schedules)    SLA / KPI) ─► ACCEPT/REJECT/
                                                                       DEFER/MODIFY
```

- Adapter: `aurelius/constraints/energy_adapter.py`
  (`EnergyArbitrageAdapter`, `ExistingEnergyCandidate`,
  `ConstraintAwareEnergyCandidate`, `DestinationContext`).
- Canonical frozen benchmark: `aurelius/benchmarks/canonical_backtests.py`
  (+ `scripts/run_canonical_backtests.py`, golden snapshot under
  `aurelius/benchmarks/golden/`). See `docs/BACKTESTS.md`.

The adapter consumes the engine's recommendation **verbatim** and only decides
whether it is safe / KPI-positive to execute — it never produces a different
energy recommendation.
