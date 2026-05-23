# Pilot Guide

This guide describes how a first Aurelius pilot is run: what the customer
provides, the non-invasive phases the pilot moves through, how success is
measured, and how data is handled. A pilot is designed to produce
customer-specific economic evidence before any change to production scheduling.

## Scope of a first pilot

A first pilot targets Tier 1 optimization — choosing the region and start window
for flexible workloads. This is the validated capability and requires no
privileged access to customer infrastructure. Tier 2 (queue-aware) and Tier 3
(GPU/node-level) optimization can be layered on later if the customer exposes
the necessary scheduler and telemetry data; they are not part of the initial
pilot's success criteria.

## Prerequisites

The pilot can begin with a single data export and two short confirmations.

| Prerequisite | Form | Purpose |
|--------------|------|---------|
| Workload trace | CSV, 30–90 days of recent jobs | Establishes the real workload mix |
| Compute regions | List mapped to markets | Selects the wholesale price feeds |
| SLA / flexibility per workload type | Delay tolerance, hard vs. flexible | Bounds the optimizer's headroom |
| Current monthly GPU spend | USD figure | Sets the magnitude for the projection |

The workload trace requires only four columns to start — job identifier,
workload type, submission time, and duration. Additional columns (GPU count,
deadline, allowed regions, interruptible/checkpointable flags, SLA class)
improve precision and can be supplied incrementally.

### Optional data (improves accuracy or unlocks higher tiers)

| Optional input | Enables |
|----------------|---------|
| Historical energy invoices | Cross-check of price-data alignment |
| Queue depth / wait-time export | Tier 2 queue-aware placement |
| DCGM telemetry via Prometheus | Tier 3 GPU/node-level placement |
| Marginal-emissions data license | Carbon-aware optimization beyond CAISO |

## Deployment modes

A pilot proceeds through three modes of increasing engagement. Most of the
evidence is produced in the first two; the third is optional and scoped.

### Offline replay

Aurelius re-runs the customer's historical workload trace against the
corresponding historical market prices and reports the cost reduction the
optimizer would have achieved versus the baseline. This is a read-only analysis
that touches no live system and produces an initial, workload-specific
projection within the first phase.

### Shadow mode

Aurelius records the decisions it would make against live market data — for each
job, the chosen region and start window, the forecast, and the predicted
savings — without executing any workload. After the relevant settlement window
closes, realized settlement prices are loaded and predicted savings are compared
to realized savings, per job and per workload class. This is the primary source
of credible, customer-specific economic evidence, because it uses the customer's
actual workload and their actual markets, and grades predictions against
independent settlement data.

### Controlled execution

Optionally, and only after shadow-mode results support it, Aurelius acts on
decisions for a limited set of flexible workloads through a scheduler adapter.
This mode is opt-in, defaults to dry-run, requires a signed policy bundle to go
live, supports an immediate kill switch, and logs every action for audit.
Latency-hard workloads are excluded.

## Phases

The pilot is organized as a sequence of phases gated by evidence, not by a fixed
calendar.

| Phase | Activity | Gate to next phase |
|-------|----------|--------------------|
| Onboarding | Ingest trace; confirm regions and SLA constraints | Trace validates; regions mapped |
| Offline replay | Produce historical-replay projection | Projection reviewed with customer |
| Shadow validation | Record live decisions; realize against settlement | Realized savings meet customer's threshold |
| Controlled execution (optional) | Act on flexible workloads under policy | Operational sign-off |

Each phase is reversible and produces an artifact the customer can review
independently before agreeing to proceed.

## Success metrics

A pilot is evaluated on evidence that is meaningful to infrastructure and
finance reviewers:

- Realized cost reduction versus the `current_price_only` baseline, on the
  customer's own workload, measured in shadow mode against settlement prices.
- Agreement between predicted and realized savings (forecast accuracy), which
  indicates how dependable the projection is going forward.
- Zero SLA violations and zero deadline misses introduced by optimized
  placement.
- Downside events — cases where an optimized decision realized worse than
  baseline — which the safety gate is designed to bound.

The historical benchmark sets expectations; the shadow-mode result is the figure
a pilot is judged on.

## Security and data handling

The pilot's minimum footprint is a workload trace and read-only market-data
access. Aurelius does not take custody of workloads, model weights, or data;
in replay and shadow modes it does not connect to the customer's scheduler at
all. The workload trace contains job metadata (type, timing, resource shape),
not payloads. Market-data credentials are read-only. No secrets are stored in
the repository, and controlled execution requires explicit, signed
authorization. Full detail is in
[security-and-deployment.md](security-and-deployment.md).

## Known limitations entering a pilot

These are stated plainly so they can be planned around:

- Validated savings cover U.S. markets (CAISO, PJM, ERCOT). European markets
  require an ENTSO-E connection that is implemented but not yet validated.
- Carbon-aware optimization currently has marginal-emissions coverage for CAISO
  only on the available data plan.
- Tier 2 and Tier 3 have been exercised against synthetic fixtures; live
  validation depends on customer-supplied queue and telemetry data.
- Controlled-execution adapters are unit-tested but not yet validated against
  live production infrastructure; first controlled execution should be scoped
  narrowly and reviewed.
