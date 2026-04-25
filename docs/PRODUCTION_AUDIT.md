# Aurelius Production Audit

## Executive Summary

The simulator core works and is honest. Greedy/local-search/MILP optimization, dual-baseline comparison (FIFO + peak-blind), energy and carbon cost reporting, and robustness testing all execute correctly against synthetic data.

Zero real-world data exists anywhere. Every energy price, carbon intensity, and job trace is procedurally generated. No API calls to EIA, ENTSO-E, WattTime, ElectricityMaps, or any other grid/carbon source have been written.

Future-data leakage is intentional and pervasive. The simulation trains forecasting models on the entire historical window and then optimizes against the same window. Reported savings figures are best-case upper bounds, not realistic estimates.

There is no backtesting engine. No walk-forward validation, no train/test split, no holdout period. The word "backtesting" does not appear in any code path.

The execution adapters (AWS Batch, Kubernetes, Slurm) are architecturally sound but entirely untested against real infrastructure. They default to dry-run, have a kill switch, and require a signed policy bundle for live mode — but resource estimates (kW→vCPU) are rough guesses and the safety gate is silently bypassed because ScheduleDecision has no forecast field.

An offline learning-loop skeleton exists (execution/post_execution.py → ml/dataset.py → ml/trainers.py → ml/artifacts.py), but it has never been wired end-to-end and has a cold-start problem: it requires live execution data to produce anything useful.

No production deployment infrastructure exists. No Dockerfile, no docker-compose, no Kubernetes manifests, no CI/CD pipeline, no .env.example, no secrets management.

No GPU modeling, no workload-type differentiation, no PUE, no data-transfer costs, no latency modeling. All jobs are generic power-and-runtime bags. Inference, batch inference, fine-tuning, and training are indistinguishable.

The ML training pipeline (ml/trainers.py) uses only descriptive statistics (bucketed means, percentiles, empirical coverage rates) — no gradient-based models for the offline artifacts. The LightGBM quantile models in forecasting/ train on the simulation price/carbon series, not on realized execution outcomes.

The most dangerous single bug: the safety gate never fires. QuantileSafetyGate.filter() reads decision.forecast, but ScheduleDecision has no forecast attribute; the gate silently passes every decision as "missing forecast → treated as passing" (safety/quantile_gate.py:155–162).

## Repo Map

```text
aurelius/
├── models.py               Core dataclasses (Job, EnergyPrice, CarbonIntensity,
│                           ScheduleDecision, SimulationResult, OptimizationConfig)
├── database.py             Optional Supabase client; silently fails if not configured
├── cli.py                  CLI: simulate, generate-data, robustness-test, show-schema
├── api/app.py              FastAPI: POST /simulate, GET /health (only); /simulations 404s
│
├── ingestion/
│   ├── energy_prices.py    STUB — synthetic prices with hardcoded base rates + noise
│   └── job_logs.py         STUB — synthetic jobs with hardcoded profiles
│
├── forecasting/
│   ├── price_model.py      WORKING — LightGBM quantile (p50,p90) offline, synthetic data
│   ├── carbon_model.py     WORKING — identical to price_model for carbon
│   ├── baseline.py         WORKING — Ridge/ElasticNet baseline + scenario multipliers
│   ├── quantile_model.py   WORKING — shared training utilities, time-series CV split
│   └── uncertainty.py      STUB — coefficient-of-variation formula only
│
├── optimization/
│   ├── scheduler.py        WORKING — greedy, local search, MILP (PuLP optional)
│   ├── objective.py        WORKING — α·cost + β·carbon + γ·risk
│   └── constraints.py      PARTIAL — constraints defined; power-cap unenforced in solver
│
├── simulation/
│   ├── replay.py           WORKING — orchestrates full simulation; intentional leakage
│   ├── compare.py          WORKING — FIFO and peak-blind ASAP baselines
│   └── metrics.py          WORKING — energy/carbon/compute/delay metrics
│
├── execution/
│   ├── base.py             WORKING — abstract Executor, ExecutionConfig, ExecutionResult
│   ├── aws_batch.py        WORKING-NOT-PROD — dry-run default, kill switch, boto3 lazy-load
│   ├── kubernetes.py       WORKING-NOT-PROD — dry-run default, kill switch, K8s lazy-load
│   ├── slurm.py            WORKING-NOT-PROD — dry-run default, kill switch, sbatch wrapper
│   ├── policy.py           WORKING-NOT-PROD — signed policy bundle gate; HMAC secret in code
│   ├── constraints.py      WORKING — batch_optimized / latency_safe constraint evaluator
│   └── post_execution.py   WORKING — PostExecutionRecorder writes JSONL; never called by opt.
│
├── ml/
│   ├── artifacts.py        WORKING — versioned JSON artifact writer/reader
│   ├── dataset.py          WORKING — loads PostExecutionRecord JSONL, extracts TrainingRecord
│   ├── train_offline.py    WORKING — CLI: trains 5 artifact types; requires JSONL input data
│   └── trainers.py         WORKING — bucketed stats only (no gradient ML for artifacts)
│
├── safety/
│   └── quantile_gate.py    BROKEN — gate always passes; ScheduleDecision.forecast missing
│
├── data/
│   └── persistence.py      WORKING — JSONLWriter append-only; no DB dependency
│
└── validation/
    └── robustness.py       WORKING — multi-seed stability harness


    Capability Assessment

What Aurelius can actually do today:

Generate synthetic energy prices and jobs, run a greedy/local-search/MILP optimizer against them, and report % cost and carbon savings vs. two baselines. CLI command works.

Train LightGBM quantile regression models on synthetic price/carbon series and use those forecasts to weight the optimizer objective.

Accept a simulation request via REST API, execute it, return a JSON result. /simulate works; /simulations/{id} does not.

Run N simulations with different seeds and report savings stability.

In dry-run mode: accept scheduling decisions, check guardrails (kill switch, delay, power reduction), log what would be submitted to AWS Batch / Kubernetes / Slurm.

Record post-execution data to JSONL (if called), and train offline ML artifacts from that JSONL (if enough data exists).

What it cannot do today:

Ingest any real energy or carbon data.

Backtest against historical reality.

Model GPU queues, utilization, PUE, data-transfer costs, or workload types.

Generate a credible savings estimate that a buyer could independently verify.

Actually submit a job anywhere (live mode is gated behind a signed policy bundle that doesn’t exist).

Activate the safety gate in any real scenario (it silently passes everything).

Run the learning loop end-to-end (cold-start; requires live execution data).

Gap Analysis


ea

Current Status

Evidence / Files

Production Risk

Required Fix

Real energy price ingestion

E — MISSING

ingestion/energy_prices.py: 100% synthetic via generate_synthetic()

CRITICAL — any savings claim is fiction

Integrate EIA API v2 (US), ENTSO-E Transparency (EU), real CSV import path

Real carbon intensity ingestion

E — MISSING

ingestion/energy_prices.py and ingestion/job_logs.py: no carbon API

CRITICAL

Integrate ElectricityMaps or WattTime

Historical time-series storage schema

D — STUB

database.py: expects 4 Supabase tables that may not exist; no migration scripts

HIGH — data silently lost

Create TimescaleDB or Postgres schema with proper migrations

Forecasting on real data

C — PARTIAL

forecasting/price_model.py: LightGBM model trains but only on synthetic data; good architecture

HIGH — model accuracy unknown

Wire ingestion → training; add rolling retrain cron

Backtesting engine

E — MISSING

No file anywhere contains walk-forward logic

CRITICAL — claimed savings unverifiable

Build backtesting/ module with horizon-aware train/test split

Future-data leakage prevention

E — MISSING

simulation/replay.py: trains on full history, then optimizes same period

CRITICAL — inflates savings

Enforce t_train < t_eval invariant throughout

Baseline policies (latency-first, closest-region, round-robin, current-price-only)

C — PARTIAL

simulation/compare.py: only FIFO and peak-blind ASAP; no latency-first, round-robin, or current-price-only

MEDIUM — comparison set too narrow

Add 4 additional baselines

Safety gate

B-broken

safety/quantile_gate.py: gate code is correct, but ScheduleDecision.forecast doesn’t exist → always passes

HIGH — risk control silently disabled

Add forecast: Optional[dict] to ScheduleDecision; wire optimizer to populate it

Power-cap constraint enforcement

C — PARTIAL

optimization/constraints.py:check_schedule_constraints() exists; scheduler.py greedy/local-search don’t call it pre-assignment

MEDIUM

Add regional capacity check inside _solve_greedy() assignment loop

Workload-type modeling

E — MISSING

models.py:Job: no workload_type field

HIGH

Add workload_type: Literal[“inference”,“batch”,“fine_tune”,“training”]; differentiate runtime/power profiles

GPU modeling

E — MISSING

No GPU-hours, GPU type, queue, or utilization anywhere

HIGH

Model GPU-count, memory, queue depth, interruptibility

PUE modeling

E — MISSING

No PUE factor applied anywhere

MEDIUM

Add per-datacenter PUE to OptimizationConfig

Data-transfer cost modeling

E — MISSING

Not in objective function or constraints

MEDIUM

Add network egress cost per region pair

SLA penalty modeling

E — MISSING

Deadlines exist but no penalty for violations

MEDIUM

Add sla_penalty_per_hour to Job; add it to objective

Learning loop end-to-end

C — PARTIAL

All pieces exist; none wired together; requires live execution data to bootstrap

HIGH

Wire optimizer → PostExecutionRecorder → periodic ml/train_offline.py; load artifacts into forecasters

Reporting with confidence intervals

C — PARTIAL

simulation/metrics.py and validation/robustness.py exist; no CIs in API output

MEDIUM

Bootstrap CI over N seeds in reporting layer

REST API completeness

C — PARTIAL

/simulate and /health work; /simulations and /simulations/{id} return 404

MEDIUM

Implement list/get simulation endpoints

Docker / deployment

E — MISSING

No Dockerfile, no docker-compose, no K8s manifests

HIGH

Containerize; add compose for local dev

CI/CD pipeline

E — MISSING

No .github/workflows, no gitlab-ci.yml

MEDIUM

Add test + lint pipeline

Database migrations

E — MISSING

No alembic, no flyway, no migration files

HIGH

Schema migrations required before any persistent data

Policy HMAC secret exposure

B — WORKING NOT PROD

execution/policy.py:_HMAC_V0_SECRET embedded in source code

HIGH — if code leaks, policy gating bypassed

Move to HSM or proper key management; implement Ed25519 signing properly

vCPU/resource estimation

D — STUB

execution/aws_batch.py:315: assume 100kW base hardcoded comment; KW_PER_VCPU = 0.00625 unvalidated

MEDIUM

Add per-job resource specification in Job model

Real unit tests

E — MISSING

No tests/ directory, no pytest infrastructure, inline-only

MEDIUM

pytest suite with CI

Shadow mode / production pilot path

C — PARTIAL

Dry-run execution layer exists; no real data feed to shadow against

HIGH

Backtesting Audit

Verdict: MISSING. What exists is a forward-only simulator with intentional leakage.

What passes for “backtesting” today

simulation/replay.py loads synthetic price/carbon data, trains forecasting models on it, then runs the optimizer over the same data range. The README calls this “shadow-mode simulation.”

The leakage problem in detail

replay.py feeds the entire price/carbon series to PriceQuantileForecaster and CarbonQuantileForecaster at simulation start. The forecasting models then predict prices at time t with knowledge of prices at t+1, t+2, …, t+N because they were trained on the full series. When the optimizer asks “what will the price be at hour 14 tomorrow?” the model has already seen hour 14 in its training set. This is survivorship bias masquerading as forecasting.

What a real backtesting engine requires

Walk-forward split: At each evaluation step t, train on [t_start, t) only. Predict [t, t+H).

Realized vs. predicted tracking: Store (predicted_price_t, actual_price_t) pairs.

Optimizer decisions using only available information: No future prices, no future carbon values.

Baseline policy decisions made under identical information constraints.

Aggregate: savings = Σ(baseline_cost_t - optimized_cost_t) where both costs are computed at realized prices, not predicted ones.

What’s needed

backtesting/engine.py: BacktestEngine.run(price_series, carbon_series, jobs, horizon_hours, train_window_hours)

backtesting/splitter.py: strict temporal split enforcer

backtesting/evaluator.py: per-step realized vs. predicted cost/carbon

Integration test: assert train_end_timestamp < eval_start_timestamp for every training call

ML Forecasting Audit

What exists

forecasting/price_model.py — PriceQuantileForecaster

Algorithm: LightGBM quantile regression (p50,p90) with Ridge regression as baseline anchor

Features: hour-of-day, day-of-week, region one-hot, lag_1h, lag_6h, rolling_mean_6h

Training: single fit at simulation start on full synthetic history

Persistence: models stored as in-memory objects; no serialization to disk

Reproducibility: fixed seed (42 inside quantile_model.py)

Status: WORKING-NOT-PRODUCTION — architecture is reasonable; trained on fake data; not persisted

forecasting/carbon_model.py

Identical architecture to price model. Carbon base values hardcoded: us-west: 350, us-east: 450, eu-west: 380 gCO2/kWh.

forecasting/baseline.py

Ridge or ElasticNet on same features

Also provides named scenarios: normal, high_renewable, peak_demand (static multipliers — not ML)

Status: WORKING-NOT-PRODUCTION

forecasting/uncertainty.py

estimate_uncertainty(): coefficient of variation = std/mean from forecast series

apply_risk_penalty(): base_cost × (1 + uncertainty)²

No calibration, no prediction intervals, no held-out coverage tracking

Status: STUB/MOCK

ml/trainers.py — offline artifacts

5 trainers: forecast_corrections, error_models, uncertainty_rules, savings_model, risk_priors

All use bucketed descriptive statistics (means, percentiles, coverage rates) — no gradient-based ML

Input: PostExecutionRecord JSONL from live execution

These artifacts are NOT loaded anywhere in the current codebase; they’re produced but never consumed by price_model.py or the optimizer

Status: WORKING infrastructure; disconnected from everything else

What’s missing

Model serialization/persistence to disk (joblib/pickle) for the LightGBM models

Model versioning and rollback

Calibration of quantile forecasts (are p90 bounds actually at 90th percentile coverage?)

Feature engineering from real grid data (LBMP zones, settlement intervals, REC prices)

Actual artifact loading in the forecasting pipeline

Any weather features (wind, solar irradiance — key drivers of renewable carbon intensity)

Multi-step forecast horizon validation (how accuracy degrades at +1h, +6h, +24h)

Production Testing Audit

What exists

Dry-run execution layer (execution/base.py, aws_batch.py, kubernetes.py, slurm.py): Well-designed. Default mode is dry_run. Kill switch via env var. Logs every action. Policy gate blocks live mode without signed bundle.

Robustness harness (validation/robustness.py): Runs N seeds, reports savings mean/std/min/max, flags instability.

Constraint evaluator (execution/constraints.py): Enforces latency_safe and batch_optimized profiles at execution time.

What’s missing

No real data feed to shadow against. Dry-run execution has the right structure but there is nothing to shadow — the inputs are synthetic.

No shadow-mode wiring to real workload traces. A real pilot requires: (1) client workloads ingested, (2) price/carbon data from real APIs, (3) optimizer running alongside existing scheduler, (4) decisions logged without executing, (5) outcome tracking.

No integration tests. The execution adapters have inline _run_tests() functions but they use mock data and never touch real AWS/K8s/Slurm endpoints.

No connection between simulation output and execution layer. replay.py produces ScheduleDecision objects but never passes them to any Executor.

Required to get to a real pilot

Real price/carbon ingestion (Phase 1 blocker)

Client workload trace ingestion (CSV/API format to be defined)

Wire simulation/replay.py → execution/aws_batch.py (or K8s/Slurm) in dry-run mode

Wire post_execution.py to record every dry-run outcome

Compare dry-run Aurelius decisions against what the client’s scheduler actually did

Learning Loop Audit

Architecture as designed (correct but unbuilt)

Live execution
    ↓
PostExecutionRecorder (execution/post_execution.py)
    → data/post_execution/post_execution_records.jsonl
    ↓
ml/train_offline.py  (runs periodically)
    → trains forecast_corrections, error_models, uncertainty_rules, savings_model, risk_priors
    → writes ml_artifacts/
    ↓
Forecasters load artifacts  ← THIS STEP DOES NOT EXIST
    ↓
Improved forecasts → better optimizer decisions
    → more data → better artifacts → compounding data moat

What’s actually wired: nothing

PostExecutionRecorder.record() exists but is never called anywhere in the optimizer or simulation flow.

The 5 ml/trainers.py functions produce correct artifacts given input data.

No code in price_model.py or carbon_model.py loads these artifacts.

The ml/train_offline.py CLI works but requires PostExecutionRecord JSONL that can only exist after live execution.

Cold-start problem

The learning loop requires realized energy prices and carbon intensities from actual job execution windows. These can only come from:

Real grid APIs (currently MISSING), or

Proxy data attached to dry-run records

There is no synthetic warm-start path — you cannot bootstrap the learning loop from simulation data because the realized_* fields in PostExecutionRecord represent actual grid measurements, not forecasts.

What “data moat” would require

Routine ingestion of real grid prices/carbon (creates proprietary time-series dataset)

Per-client workload trace accumulation (creates proprietary demand dataset)

Per-region, per-hour forecast error tracking across many clients (creates proprietary calibration dataset)

Client-specific constraint profiling (creates proprietary constraint database)

None of these are accruing today.

Security + Reliability Concerns

Security

HMAC secret hardcoded in source (execution/policy.py:48): _HMAC_V0_SECRET = b”v0_hmac_not_secure_against_key_exposure_aurelius_policy_secret” — the comment says it, and it’s true. Any code leak breaks policy gating entirely. Fix: implement Ed25519 properly (_ED25519_PUBLIC_KEY_B64 exists but the signing infrastructure does not).

Supabase anon key in environment (database.py): Uses SUPABASE_ANON_KEY — fine for read-only, but any writes go through an unauthenticated client. Simulation results are written to a public-key endpoint.

No API authentication: api/app.py has no auth middleware. Anyone who can reach the endpoint can trigger arbitrary simulations.

No rate limiting: Simulation endpoint has no throttling; CPU-intensive MILP runs can be triggered freely.

Reliability

Regional power-cap constraint silently violated: optimization/constraints.py:check_schedule_constraints() exists but is called only post-hoc in greedy/local-search. The MILP solver also ignores it. A schedule claiming to respect a 10MW cap may actually assign 15MW.

Silent DB failures: database.py catches all exceptions and returns False/[]. Simulation results appear to persist but may not.

MILP fallback is silent: If PuLP returns an infeasible or incomplete solution, scheduler.py silently falls back to greedy. The caller never knows MILP failed.

AWS resource estimation hardcoded: aws_batch.py:315: vcpus = estimate_vcpus_from_power(100.0, decision.power_fraction) — assumes 100kW base for every job. This is wrong for the overwhelming majority of real workloads.

Data Quality

No input validation on synthetic generation parameters: ingestion/energy_prices.py:generate_synthetic() accepts arbitrary start_date, end_date, regions with no checks. Zero-length windows produce silent empty DataFrames that propagate through the optimizer.

No schema enforcement on Supabase inserts: Records are inserted as plain dicts; no Pydantic validation before DB writes. Schema drift between code and DB will cause silent failures.
