# Aurelius

Aurelius is an infrastructure orchestration layer that reduces GPU operating
cost through SLA-aware workload placement, time-shifting, and market-aware
scheduling. It decides *when* and *where* flexible compute should run so that
the same work lands in lower-cost hours and regions, without changing the work
itself.

Aurelius is a decision layer, not an execution platform. It produces placement
and timing decisions that an existing scheduler (Kubernetes, Slurm, Ray, AWS
Batch, or a CSV-driven pipeline) carries out. It does not hold customer
workloads, does not require privileged access to run them, and operates in a
read-only or dry-run posture by default.

## What it does

Given a workload trace and the relevant wholesale electricity markets, Aurelius
forecasts near-term price movements and selects, for each flexible job, the
region and start window that minimizes energy cost while respecting deadlines,
SLA class, and capacity constraints. Carbon intensity, queue depth, and GPU
health can be added as additional decision signals where the customer exposes
them.

The system is built around three properties that matter for production use: a
deterministic fallback to the customer's current scheduling behavior whenever a
forecast is unavailable or low-confidence; a leakage-free evaluation pipeline so
that reported savings reflect information actually available at decision time;
and a shadow mode that validates customer-specific savings against realized
market prices before any decision is acted on.

## Who it is for

Neoclouds, GPU cloud and inference providers, enterprise GPU fleets, HPC
clusters, and data center operators — organizations that run schedulable compute
and can redirect job placement across regions or time. The primary requirement
is workload flexibility, not a specific cloud or scheduler.

## Optimization tiers

| Tier | Decision | Status | Requires |
|------|----------|--------|----------|
| 1 | Which region, which hour | Validated in historical replay | Workload trace + market region |
| 2 | Which cluster / queue | Implemented; pending customer data | Queue depth / wait-time export |
| 3 | Which node / GPU | Implemented; fixture-tested | DCGM/Prometheus + scheduler integration |

Tier 1 is the basis for a first pilot and is the only tier with validated
savings. Tiers 2 and 3 add decision signals on top of Tier 1 and are evaluated
per deployment against customer-supplied data.

## Validation status

Tier 1 placement has been validated in leakage-free historical replay on real
day-ahead prices from CAISO, PJM, and ERCOT. Across seven workload classes the
mean cost reduction versus a strong `current_price_only` baseline was 25.0% on
Q1 2026 data and 22.8% on Summer 2025 data. Savings are workload-dependent:
flexible batch and maintenance workloads see the largest reduction; latency-hard
real-time inference sees little. Savings for any specific customer are expected
to be confirmed through shadow-mode evaluation before contract commitment.

See [benchmark-methodology.md](benchmark-methodology.md) for the full
validation detail and [roi-methodology.md](roi-methodology.md) for how these
figures translate into a customer projection.

## Pilot flow

A pilot proceeds in three non-invasive phases: offline replay against the
customer's historical workload trace; shadow-mode evaluation that records live
decisions and compares predicted savings to realized prices after the
settlement window closes; and, optionally, controlled execution of a limited
set of flexible workloads behind a policy gate. No phase requires Aurelius to
take custody of production workloads.

See [pilot-guide.md](pilot-guide.md) for prerequisites and phase detail.

## Documentation

| Document | Audience |
|----------|----------|
| [enterprise-overview.md](enterprise-overview.md) | Buyers, infrastructure leadership |
| [technical-architecture.md](technical-architecture.md) | Platform and ML infrastructure engineers |
| [pilot-guide.md](pilot-guide.md) | Pilot evaluators and onboarding |
| [roi-methodology.md](roi-methodology.md) | Finance, procurement, infrastructure leads |
| [benchmark-methodology.md](benchmark-methodology.md) | Technical evaluators |
| [security-and-deployment.md](security-and-deployment.md) | Security and procurement reviewers |
| [developer-guide.md](developer-guide.md) | Engineers running the system locally |
| [deck-outline.md](deck-outline.md) | Internal — pilot conversation structure |
