# Benchmark Methodology

This document is the technical validation reference. It describes how Aurelius
is benchmarked, the controls that keep the results honest, the baselines it is
measured against, and the limitations of the current validation. It is the
appropriate level of detail for a technical evaluator who wants to judge whether
the reported savings are credible and reproducible.

## What is measured

The benchmark measures the energy-cost reduction Aurelius achieves on a workload
relative to a baseline scheduling policy, using real wholesale electricity
prices. The headline metric is percentage cost reduction versus
`current_price_only`, reported per workload class and as a mean across classes.
Secondary metrics — carbon impact, SLA violations, migration count, downside
events, and per-fold variance — are reported alongside.

## Data

| Market | Region | Source | Signal |
|--------|--------|--------|--------|
| CAISO | us-west | OASIS (public) | Day-ahead + real-time LMP |
| PJM | us-east | Data Miner API | Day-ahead + real-time LMP |
| ERCOT | us-south | CDAT API | Day-ahead + real-time LMP |

Validation uses two real historical windows — Q1 2026 (January–March, higher
winter volatility) and Summer 2025 (June–August, more stable) — with 0% missing
price hours in the reported configurations. A longer combined window
(mid-2025 through Q1 2026) is also available and was used for seasonal
robustness analysis. All economic claims use real, unrandomized data from the
source ISO; synthetic and sandbox data are structurally barred from any savings
path by an admissibility check in the ingestion layer.

## Walk-forward validation

Evaluation is leakage-free walk-forward, not in-sample. For each fold the
forecaster is trained only on price data with a timestamp strictly before the
fold's evaluation window, then evaluated on the held-out window. The validated
configuration uses 30-day training windows and 5 walk-forward folds. This
structure ensures the forecaster never sees information that would not have been
available at decision time.

The same temporal split is applied to every optional signal: carbon, queue
state, and GPU telemetry are all restricted to data preceding the evaluation
window. Weather, where used experimentally, is treated as exogenous (its future
values are legitimately knowable) but its model is still trained only on the
pre-evaluation split.

## Leakage prevention

Leakage prevention is treated as a correctness property and is independently
tested. The controls are:

- Forecaster training data is filtered to `timestamp < eval_start` per fold.
- Real-time settlement prices are never visible to the forecaster or optimizer
  at decision time; in shadow mode they are introduced only by a separate
  realization step after the window closes.
- Optional signals (carbon, queue, GPU) use a "last known before T" lookup, so a
  future reading cannot influence a past decision.
- Sandbox/synthetic provenance raises an error if it reaches a benchmark path.

A documented example of why this matters: during development, a predict-time
feature was inadvertently zero-filling the future window, which manufactured an
artificial "prices collapse to zero" signal and inflated apparent savings. The
defect was found in adversarial review, the inflated result was discarded and
never archived, and the fix (forward-filling from the last known price) produced
the honest figure that is reported today. The benchmark harness is designed so
that this class of error surfaces rather than ships.

## Baselines

| Baseline | Definition | Role |
|----------|------------|------|
| `current_price_only` | Always route to the cheapest region at submission, using live prices | Primary — strong, realistic |
| Upper-bound diagnostic | Optimizer with perfect future price knowledge | Diagnostic ceiling, not a baseline |
| Naive single-region / FIFO | No optimization | Reference only; not used for headline claims |

The headline metric is always versus `current_price_only`. It is the strongest
realistic baseline — it has perfect current-price information and represents
what a sophisticated manual operator achieves — so improvements over it are
attributable to forecasting, cross-region routing, and rescheduling rather than
to a weak comparison.

## Results summary

Mean cost reduction versus `current_price_only`, validated configuration
(ml_quantile forecaster, migration-aware solver, 30-day windows, 5 folds, real
data, 0% missing hours):

| Window | Mean reduction |
|--------|----------------|
| Q1 2026 (CAISO/PJM/ERCOT) | 25.0% |
| Summer 2025 (CAISO/PJM/ERCOT) | 22.8% |

Per-workload p50 reductions are listed in
[roi-methodology.md](roi-methodology.md). Savings persist across both seasonal
windows, which is the relevant robustness check.

## Upper-bound diagnostic

The upper-bound diagnostic quantifies how much of the available opportunity the
optimizer captures. Where the gap to the diagnostic is small (LLM batch
inference, ~9 points), the optimizer is near the structural limit and further
forecasting effort has little headroom. Where the gap is large (fine-tuning,
~33 points on winter data), forecasting — specifically anticipating winter price
spikes — is the binding constraint. The same workloads show single-digit gaps on
stable summer data, confirming that the optimizer is near-optimal when the
market is predictable and that the residual gap is a forecasting problem, not an
optimization one.

## Reproducibility

All results are reproducible from committed data and a fixed seed. Forecasting
and optimization are deterministic given the seed; benchmark runs pin the data
window, baseline set, fold count, and training-window length, and archive both a
machine-readable result and a human-readable summary. A regression-comparison
step diffs a new run against an archived one and flags unexpected savings drops,
missing-hour changes, or SLA-violation changes. Commands are in
[developer-guide.md](developer-guide.md).

## Limitations

The current validation is bounded, and the bounds are stated rather than
implied:

- Coverage is three U.S. markets. European (ENTSO-E) and Asia-Pacific markets
  are not yet validated.
- Two seasonal windows of roughly 90 days each are validated; persistence over a
  full year is not yet demonstrated.
- Benchmark workloads are generated from per-class profiles for the published
  figures. Customer-specific results require the customer's own trace, run in
  offline replay and confirmed in shadow mode.
- Carbon-aware results are limited to CAISO marginal-emissions coverage on the
  available data plan.
- Queue-aware (Tier 2) and GPU-health (Tier 3) signals have been validated only
  against synthetic fixtures, which are explicitly excluded from savings claims;
  their economic contribution is established per deployment from customer data.
- Experimental forecaster variants (weather-aware, per-region, regime-
  correction) did not consistently beat the production model in the windows
  tested and are not the default. The production model remains the validated
  configuration.
