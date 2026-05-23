# Security and Deployment

This document is intended for security and procurement reviewers. It describes
what data Aurelius requires, what it deliberately does not require, how it
operates within a customer environment, and the controls that bound its
behavior. The central point is that Aurelius is a decision layer with a small,
read-only footprint by default; it does not take custody of workloads or data.

## Data required

A Tier 1 deployment requires two inputs:

| Input | Contents | Sensitivity |
|-------|----------|-------------|
| Workload trace | Job metadata: type, submit time, duration, resource shape | Operational metadata, not payloads |
| Market-data access | Read-only wholesale price feeds for the relevant regions | Public or read-only API keys |

The workload trace describes the *shape* of jobs — what kind, how long, how
flexible — not their contents. No model weights, datasets, customer data, or job
payloads are required at any tier.

## Data not required

Aurelius does not require, request, or store: training data or datasets; model
weights or artifacts; source code of customer workloads; production credentials
for compute clusters (except for opt-in controlled execution, below); or any
personally identifiable information. In replay and shadow modes it does not
connect to the customer's scheduler at all.

## Operating posture

| Mode | Network access | Side effects |
|------|----------------|--------------|
| Offline replay | None beyond local data | Writes report files only |
| Shadow mode | Read-only market data | Writes decision/report files only |
| Controlled execution | Market data + scheduler adapter | Submits scoped jobs under signed policy |

The default and recommended posture for a pilot is read-only: offline replay and
shadow mode make no changes to any live system.

## IAM and least privilege

Market-data connectors use read-only API keys (CAISO requires none). For
controlled execution, the scheduler adapter requires only the permission to
submit and label jobs in the regions and queues explicitly in scope — for
example, a Kubernetes service account scoped to a namespace, or a Slurm account
limited to designated partitions. Aurelius requests no cluster-admin
privileges, no access to other tenants' workloads, and no standing credentials
beyond what the in-scope adapter needs. Grant the narrowest role that lets the
adapter place the agreed set of flexible workloads.

## Controlled execution controls

Controlled execution is opt-in and layered with controls so that the worst case
is a deferred or skipped action, never an unsafe one:

- Dry-run by default. Adapters log the actions they would take without taking
  them until live mode is explicitly enabled.
- Signed policy bundle. Live mode requires a valid signed policy; the runtime
  can enforce a policy but cannot mint one, so authorization is held by the
  customer.
- Kill switch. A single environment flag aborts all execution immediately.
- Scope limiting. Only workloads explicitly designated flexible are eligible;
  latency-hard workloads are excluded.
- Safety gate. Every decision is vetted against a workload-specific downside
  threshold and falls back to baseline placement if the forecast is missing or
  low-confidence (fail-closed).

The execution adapters are unit-tested against mocks but not yet validated
against live production infrastructure; a first controlled execution should be
scoped narrowly and reviewed jointly.

## Audit trail

Every decision and every execution attempt is recorded. Shadow-mode decision
records (append-only JSONL) capture, per job, the decision time, chosen region
and start, the forecast, predicted cost, and baseline cost; the realization
step later appends the realized outcome. Execution adapters log each attempted
action with its policy context. Benchmark runs archive machine- and
human-readable artifacts. These records support reconstruction of why any
decision was made and what resulted.

## Secrets handling

Credentials are supplied through environment variables and are never committed.
The repository ships only an `.env.example` with placeholder values; `.gitignore`
excludes `.env`. Provider tokens are redacted in logs and object representations.
A secret-scanning step and the no-secrets policy are part of the development
process. The REST API requires an API key (`AURELIUS_API_KEY`) supplied via an
`X-API-Key` header.

## Deployment options

| Option | Description | Suited to |
|--------|-------------|-----------|
| Local Python | Direct CLI / library use | First pilot, offline replay, shadow mode |
| Docker container | Packaged service incl. REST API | Reproducible single-node deployment |
| REST service | FastAPI with API-key auth | Programmatic integration |

Persistence is local append-only JSONL by default, which is sufficient for
single-node pilots and keeps the deployment self-contained. A database backend
(Postgres/TimescaleDB) is configurable for multi-instance deployments and
longer-running audit retention; it is not required for a pilot.

## Compliance posture

Aurelius is pre-SOC 2. A formal compliance program is on the roadmap and is not
yet in place; this is stated plainly for procurement planning. The architecture
is designed to make such a program tractable: a small read-only data footprint,
no custody of customer data or workloads, environment-variable secret handling,
least-privilege execution, and a complete decision and execution audit trail. A
pilot can be conducted entirely in read-only modes (offline replay and shadow),
which avoids granting any execution privilege while the compliance posture
matures.

## Summary for reviewers

- Default footprint: a workload trace plus read-only market-data access.
- No custody of workloads, data, or model weights at any tier.
- Pilots run read-only; execution is opt-in, scoped, signed, and reversible.
- Full audit trail of decisions and execution attempts.
- Secrets via environment only; none committed; tokens redacted in logs.
- SOC 2 not yet in place; architecture minimizes the data and privilege surface.
