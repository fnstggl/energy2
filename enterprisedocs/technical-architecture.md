# Technical Architecture

This document describes the engineering structure of Aurelius for platform and
ML infrastructure teams: the components, the data flow between them, the
boundaries of what the system controls, and its behavior under failure.

## Overview

Aurelius is a decision pipeline. Market and workload data enter through
ingestion, a forecaster produces calibrated price expectations, an optimization
engine selects placement and timing under constraints, a safety gate vets each
decision, and the result is either reported (replay, shadow) or handed to an
execution adapter (controlled execution). Every stage is deterministic given
its inputs and a fixed seed, and every stage degrades to the customer's baseline
behavior rather than to an unsafe state.

```
            ┌────────────┐   ┌─────────────┐   ┌───────────────┐   ┌────────────┐
 markets ──▶│ Ingestion  │──▶│ Forecasting │──▶│ Optimization  │──▶│ Safety gate│──┐
 workload   └────────────┘   └─────────────┘   └───────────────┘   └────────────┘  │
 signals          │                                    ▲                            │
                  │          carbon / queue / GPU ─────┘                            │
                  ▼                                                                 ▼
            (canonical schema)                                    ┌──────────────────────────┐
                                                                  │ Reporting  │ Execution     │
                                                                  │ (replay,   │ adapters       │
                                                                  │  shadow)   │ (dry-run def.) │
                                                                  └──────────────────────────┘
```

## Components

### Data ingestion

Ingestion normalizes every external signal into a canonical, timestamped,
per-region schema. Wholesale prices come from direct ISO/TSO connectors — CAISO
OASIS, PJM Data Miner, ERCOT CDAT — each returning hourly day-ahead and
real-time settlement prices. A provenance layer tags every record with its
source and market type and enforces a benchmark-admissibility gate: sandbox or
synthetic data is structurally barred from any savings or benchmark path.

Workload traces are ingested from customer CSV exports. A four-column minimum
(`job_id`, `workload_type`, `submit_time`, `duration_hours`) is sufficient to
begin; optional columns (GPU count, deadline, allowed/forbidden regions, SLA
class, interruptible/checkpointable flags) refine the optimization. Per-workload
defaults are applied for any omitted field.

Optional signals — carbon intensity (WattTime marginal emissions), queue state,
and GPU telemetry (DCGM via Prometheus) — enter through the same canonical
schema and are joined to price data by region and UTC hour.

### Forecasting

The forecaster produces calibrated quantile predictions (p50/p90) of future
prices per region and horizon. The validated production model is a gradient-
boosted quantile model (LightGBM) with lagged price, time-of-day and
day-of-week seasonality, and volatility-regime features designed to detect
price spikes. Calibration of the prediction interval is treated as a
first-class metric alongside point accuracy, because the safety gate depends on
the interval being trustworthy.

A deterministic seasonal-naive forecaster is retained as a fallback and as a
benchmark reference. Several experimental variants (weather-aware, per-region,
regime-correction) exist in the codebase; none currently outperforms the
production model as the default, and they remain opt-in. See
[benchmark-methodology.md](benchmark-methodology.md).

### Optimization engine

The optimizer assigns each job a region and start window to minimize a weighted
objective:

```
energy_cost + β·carbon_cost + γ·risk_penalty + δ·SLA_penalty
            + queue_delay_cost + gpu_health_cost + data_transfer_cost
```

subject to hard constraints: each job's earliest-start/deadline window, its
allowed and forbidden regions, per-region power capacity, and minimum power
fraction. The default solver is a greedy migration-aware heuristic; a local-
search refinement and an exact MILP formulation are also available for smaller
problem sizes. The risk term penalizes scheduling into high-uncertainty periods,
so a wide forecast interval naturally biases the optimizer toward the safer,
baseline-like choice.

Optional signal weights default to zero, which means a deployment that supplies
only prices behaves identically to a price-only optimizer; carbon, queue, and
GPU-health terms activate only when the corresponding data and cost weight are
provided.

### Policy and safety gate

Before any decision is emitted, the safety gate evaluates the forecast interval
behind it against a workload-specific downside threshold (most conservative for
real-time inference, most permissive for training). If the p90 cost projection
exceeds the baseline by more than the threshold, or if no valid forecast
interval is available, the gate blocks the optimized decision and the system
falls back to the `current_price_only` choice. The gate is fail-closed: absence
of evidence is treated as a reason to defer to the baseline, not to proceed.

Hard constraints — deadlines, forbidden regions, SLA class — are enforced by the
optimizer and re-checked at the gate; a decision that would violate them is
never produced.

### Shadow mode

Shadow mode is the mechanism for customer-specific validation. A single-pass
runner makes the same decisions the optimizer would make live, training only on
price data with a timestamp strictly before the decision time, and records one
decision per job: the chosen region and start, the forecast, the predicted cost,
and the baseline cost. Real-time settlement prices are never visible at decision
time. After the settlement window closes, a separate realizer fills in the
actual realized cost from settlement data, and a report compares predicted to
realized savings per job and per workload class, with forecast-accuracy metrics.

This separation — decide, then realize from independent data — is what makes the
shadow result credible rather than self-graded.

### Reporting

Benchmark and shadow runs emit machine-readable JSON and human-readable
summaries. Reports include savings versus each baseline, carbon impact, SLA
violation counts, migration counts, downside events, and per-fold variance.
Benchmark artifacts are archived for regression comparison so that a code change
that degrades savings is detectable.

### Scheduler integration

Execution adapters translate decisions into actions for Kubernetes (batch Jobs
with region node-selectors), Slurm (`sbatch` with partition/constraint
mapping), AWS Batch (region-specific job queues), and a CSV replay interface.
All adapters default to dry-run, log every attempted action for audit, support a
global kill switch, and require a signed policy bundle to enter live mode. These
adapters are implemented and unit-tested against mocks; they have not yet been
validated against live production infrastructure, and resource-mapping
heuristics are deployment-specific and should be reviewed before controlled
execution.

## Data flow and control boundaries

Aurelius reads market data and a workload description, and writes decisions. In
replay and shadow modes it writes only to its own report files; it never
contacts the customer's scheduler. In controlled execution it submits to the
scheduler through an adapter, but only for workloads explicitly in scope and
only under an active policy. The system never holds workload data or model
weights, and the only credentials it needs are read-only market-data API keys
plus, for controlled execution, whatever the chosen scheduler adapter requires.

The control boundary is deliberate: Aurelius owns the *decision*, the customer's
platform owns the *execution*. This keeps the blast radius of any Aurelius
fault contained to a suboptimal-but-valid placement, never an unsafe action.

## Failure modes and fallback behavior

| Condition | Behavior |
|-----------|----------|
| Forecast unavailable or invalid | Safety gate blocks; fall back to `current_price_only` |
| Forecast interval too wide / high downside | Gate blocks optimized decision; baseline used |
| Market data feed missing for a region | Region excluded from routing; remaining regions used |
| Optional signal (carbon/queue/GPU) absent | Corresponding objective term is zero; price-only behavior |
| Hard constraint cannot be satisfied | Job placed at its constrained baseline; flagged in report |
| Execution adapter error (live mode) | Logged; kill switch available; no retry into unsafe state |

The common thread is that every failure resolves to the customer's existing
behavior or a strictly safe subset of it. There is no failure path in which
Aurelius produces a placement worse than the baseline it is measured against,
because the baseline itself is always available as the fallback.

## Reproducibility

Forecasting and optimization are deterministic under a fixed random seed.
Benchmarks pin the data window, baseline set, fold structure, and seed, and
archive their outputs. Any reported figure can be regenerated from committed
data and the published commands. See [benchmark-methodology.md](benchmark-methodology.md).
