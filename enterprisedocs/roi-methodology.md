# ROI Methodology

This document explains how Aurelius estimates cost reduction for a specific
customer, and — equally important — what those estimates do and do not mean. It
is written to be defensible in front of finance and procurement review. The
guiding principle is that savings are workload-dependent and pilot-validated;
the historical benchmark sets a prior, and shadow mode confirms it on the
customer's own footprint.

## Three distinct figures

Conflating these is the most common source of overstated ROI. They are kept
separate throughout.

| Figure | Source | Use |
|--------|--------|-----|
| Historical validation | Leakage-free replay on real CAISO/PJM/ERCOT data | Establishes a credible prior |
| Customer-specific projection | Historical rates applied to the customer's workload mix and spend | Pre-pilot estimate |
| Shadow-mode validation | The customer's live decisions vs. realized settlement prices | The figure to commit on |

The historical validation is observed, not promised. The projection is an
estimate conditioned on the customer's mix. Only the shadow-mode result reflects
the customer's actual environment, and it is the figure a commitment should rest
on.

## Historical validation (the prior)

In leakage-free walk-forward replay on real day-ahead prices from CAISO, PJM,
and ERCOT, the observed mean cost reduction versus a strong `current_price_only`
baseline was 25.0% on Q1 2026 data and 22.8% on Summer 2025 data, across seven
workload classes. The baseline is deliberately strong: it routes every job to
the cheapest region at submission time using live prices, which is what a
capable manual operator could achieve. Reporting against a weak baseline would
inflate the figures; Aurelius does not.

Observed reduction by workload class (Q1 2026, p50):

| Workload class | p50 reduction | Driver |
|----------------|---------------|--------|
| Background maintenance | ~40% | Fully flexible, freely reschedulable |
| Data processing | ~38% | High flexibility, short duration |
| LLM batch inference | ~34% | Batch, ~24h delay tolerance |
| Scheduled batch | ~25% | Moderate flexibility |
| Training | ~15% | Long, often non-interruptible |
| Fine-tuning | ~13% | Moderate; sensitive to volatility |
| Real-time inference | ~10% | Cannot be delayed |

Savings track flexibility. A fleet weighted toward flexible batch and
maintenance work sits at the upper end; a fleet weighted toward latency-hard
inference sits at the lower end.

## Customer-specific projection

The projection applies the per-workload rates to the customer's own workload mix
and controllable spend:

```
projected_monthly_saving = monthly_controllable_GPU_spend
                         × Σ (workload_fraction_i × savings_rate_i)
```

A confidence range (p10/p50/p90) accompanies the projection. The bounds reflect
seasonal variation (winter volatility versus stable summer markets) and fold
variance observed across the walk-forward evaluation; the upper bound is capped
at the upper-bound diagnostic (the savings achievable with perfect foresight)
so the projection cannot exceed what is structurally possible.

Controllable spend excludes storage, networking, licensing, and workloads with
no scheduling flexibility. If more than roughly half the fleet is latency-hard
real-time inference, the projection is correspondingly modest, and the
methodology flags this rather than masking it.

A calculator is provided to produce this projection from a monthly spend figure
and a workload mix; see [developer-guide.md](developer-guide.md) for the
command.

## Shadow-mode validation (the figure to commit on)

Before commitment, Aurelius runs in shadow mode on the customer's actual
workload trace and real market data for a representative window. It records the
decisions it would make, then — after settlement prices are available — compares
predicted savings to realized savings per job and per workload class. This is
the most credible economic evidence available pre-deployment, because it is
measured on the customer's environment and graded against independent settlement
data rather than the model's own forecast. The pilot decision should rest on
this result.

## Why the baseline matters

All figures are reported against `current_price_only`. This baseline has perfect
knowledge of the current price and always picks the cheapest available region at
submission. Aurelius improves on it by forecasting where prices are heading,
routing across regional spreads, and rescheduling eligible jobs when a better
price emerges. Beating a baseline this strong is the relevant test;
out-performing a naive single-region baseline would not be informative.

## Upper-bound diagnostic

To separate "the optimizer is weak" from "the market offered little," each
configuration is also run against an upper-bound diagnostic — the savings an
optimizer with perfect future price knowledge would achieve. A small gap means
the optimizer is near the structural limit; a large gap means forecasting is the
constraint.

| Workload | Validated (p50) | Upper-bound diagnostic | Gap |
|----------|-----------------|------------------------|-----|
| LLM batch inference | ~34% | ~43% | ~9 pts |
| Training | ~15% | ~30% | ~15 pts |
| Fine-tuning | ~13% | ~47% | ~33 pts |

For batch inference the optimizer is near-optimal. For training and fine-tuning
the larger gaps are driven by winter price volatility (notably ERCOT cold-snap
spikes) that the forecaster only partly anticipates; on stable summer data these
gaps narrow to single digits. This is disclosed because it bounds the realistic
upside and explains the variation between seasons.

## What is not claimed

- No savings figure is guaranteed. All historical figures are observed in
  replay; customer outcomes are confirmed in shadow mode.
- Real-time-heavy fleets see limited savings, by construction, because such work
  cannot be moved.
- European and Asia-Pacific markets are not yet validated.
- Tier 2 (queue) and Tier 3 (GPU-level) savings are not included in the figures
  above; where they apply they are additive to Tier 1 and require customer
  integration to quantify.

## Frequently raised questions

**Can a specific percentage be guaranteed?** No. The benchmark mean is an
observation on historical data. Actual savings depend on workload mix, regions,
season, and flexibility, which is precisely why shadow mode is run before
commitment.

**Why is training lower than batch?** Training jobs run for many hours and are
frequently non-interruptible, so only their start time and region can be
optimized. Maintenance and batch jobs are short and freely reschedulable, giving
the optimizer more room.

**Does this work across clouds?** Yes, wherever job placement can be redirected
across regions or time. Cloud regions are mapped to the wholesale market in
their geography. Integration is via dry-run scheduler adapters or a CSV replay
interface.

**What happens on a missed price spike?** The safety gate blocks decisions whose
forecast interval projects a downside beyond a workload-specific threshold, and
the system falls back to baseline placement. This caps downside at the cost of
some upside.
