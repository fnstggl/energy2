# Enterprise Overview

## The problem

GPU compute is the dominant variable cost in AI infrastructure, and the energy
that powers it is priced in markets that move hour to hour and differ region to
region. Wholesale electricity prices routinely vary by a factor of two or more
across a day and across interconnects, yet most schedulers place jobs as they
arrive, in a default region, without regard to when or where that work would be
cheapest to run. A large share of GPU workloads — model training, fine-tuning,
batch inference, data processing, maintenance — carries real scheduling slack
that is left unused.

The result is a recurring, structural overspend that is invisible on any single
invoice and difficult to address manually at fleet scale.

## The approach

Aurelius is an orchestration layer that turns workload flexibility into cost
reduction. For each schedulable job it forecasts near-term wholesale price
movements across the relevant markets and selects the region and start window
that minimizes energy cost, subject to the job's deadline, SLA class, and the
capacity of each region. The same work runs; it simply runs at a better time
and place.

Three design choices make this safe to adopt in production:

Decisions are advisory and reversible. Aurelius emits placement and timing
decisions; the customer's existing scheduler executes them. There is no custody
of workloads and no privileged execution path required to begin.

Behavior degrades to the status quo. When a forecast is missing or its
confidence interval projects a cost above a workload-specific threshold, the
decision falls back deterministically to the customer's current
lowest-current-price behavior. The optimizer cannot push a job into a worse
outcome than the baseline it is measured against.

Savings are evidenced before they are claimed. Shadow mode records the
decisions Aurelius would have made against live market data, then compares
predicted savings to realized settlement prices once the window closes — on the
customer's own workload, in their own markets.

## Deployment model

Aurelius runs alongside existing infrastructure rather than in front of it. The
minimum integration is a workload trace (a CSV export of recent jobs) and the
identification of which wholesale market each compute region maps to. From
there the system can run entirely offline.

| Mode | What it does | Customer exposure |
|------|--------------|-------------------|
| Offline replay | Re-runs historical workload against historical prices | None — read-only analysis |
| Shadow mode | Records live decisions; compares to realized prices | None — no workloads executed |
| Controlled execution | Acts on decisions for selected workloads | Policy-gated, dry-run default, kill switch |

Most pilots remain in offline replay and shadow mode. Controlled execution is
opt-in, scoped to flexible workloads, and gated by a signed policy bundle.

## Where savings come from

Savings derive from three mechanisms, in order of contribution:

Time-shifting moves flexible jobs into lower-priced hours within their allowed
delay window. This is the largest and most reliable source.

Region routing places jobs in the lowest-cost market among those a job is
permitted to run in, exploiting price spreads between interconnects.

Migration and checkpoint-aware rescheduling relocates eligible long-running
jobs when a materially better price emerges. This contributes least and applies
only to checkpointable workloads.

Because all three depend on flexibility, savings scale with the share of the
fleet that can tolerate delay or relocation. Fully flexible maintenance and
batch workloads show the largest reductions; latency-hard real-time inference
shows little, because it cannot be moved.

## What is validated today

Tier 1 region/time placement is validated in leakage-free historical replay on
real day-ahead prices from the three largest U.S. wholesale markets — CAISO,
PJM, and ERCOT. Against a strong `current_price_only` baseline (which always
routes to the cheapest region at submission time using live prices), the
observed mean reduction across seven workload classes was 25.0% on Q1 2026 data
and 22.8% on Summer 2025 data.

| Workload class | Observed reduction (p50) |
|----------------|--------------------------|
| Background maintenance | ~40% |
| Data processing | ~38% |
| LLM batch inference | ~34% |
| Scheduled batch | ~25% |
| Training | ~15% |
| Fine-tuning | ~13% |
| Real-time inference | ~10% |

These are historical-replay observations, not guarantees. Customer-specific
savings depend on workload mix, region set, season, and scheduling flexibility,
and are confirmed in shadow mode.

## What requires customer integration

Carbon-aware optimization is available where marginal-emissions data is
licensed (currently CAISO on the available data plan). Queue-aware placement
(Tier 2) requires the customer to export queue depth and wait-time from their
scheduler. GPU/node-level placement (Tier 3) requires DCGM telemetry via
Prometheus and a scheduler integration capable of node-level selection.
European markets require an ENTSO-E data connection, which is implemented but
not yet validated. None of these are prerequisites for a Tier 1 pilot.

## Pilot process

A first pilot is structured to produce customer-specific economic evidence
before any operational change. The customer supplies a recent workload trace
and confirms region and SLA constraints; Aurelius runs offline replay to
produce an initial projection, then shadow mode to validate that projection
against realized prices. The decision to proceed to controlled execution rests
on the shadow-mode result, not on the historical benchmark. See
[pilot-guide.md](pilot-guide.md).
