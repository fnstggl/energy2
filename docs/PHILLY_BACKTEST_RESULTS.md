# Philly Backtest Results — CANONICAL_TRACE_BACKTEST_PHILLY_TRAINING_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` and `docs/PUBLIC_TRACE_BACKTESTS.md` first.

## Provenance

- **Source:** `fixture:tests/fixtures/philly_sample`
- **Dataset:** Microsoft Philly traces (https://github.com/msr-fiddle/philly-traces) — `cluster_job_log` (JSON) + `cluster_machine_list` (CSV).
- Philly public data is a research dataset, **not customer telemetry**.
- ⚠️ **This run used the committed fixture**, not the full ~6.6 GB trace (a ~1 GB git-LFS tarball; see `scripts/ingest_philly.py` for download steps). Numbers are a fixture-scale demonstration; the full-trace backtest is integration-only.

## Discovered schema + missing fields (honest)

`cluster_job_log` = JSON list of `{status, vc, jobid, submitted_time, user, attempts[{start_time, end_time, detail[{ip, gpus[]}]}]}`; times are `%Y-%m-%d %H:%M:%S`. GPU count = `sum(len(detail.gpus))` of the first attempt. `cluster_machine_list` = `machineId, number of GPUs, single GPU mem`.

- **Missing (stated):** gpu_type_model, cpu_host_memory_request, deadline, per_job_gpu_utilization. GPU type is inferred only as a `GPU-<mem>` label; **no real GPU model / price**, so constraint_aware's heterogeneous-pricing lever is inactive here. **No CPU/host-mem request, no deadline.**
- `is_failed` = status ∈ {Failed, Killed}. **goodput_unit = `gpu_seconds_work`** (effective_GPU × duration) — NOT inference tokens.

## Trace summary

- Jobs: **33** (32 GPU jobs, 5 users)  ·  scheduled: **32**
- Status: {'Failed': 3, 'Killed': 2, 'Pass': 28}
- num_gpu distribution: {0: 1, 1: 17, 2: 6, 4: 4, 8: 4, 16: 1}
- Job duration s p50/p95/p99: 1800 / 3600 / 3600
- Trace-observed queue wait s p50/p95/p99: 30.0 / 30.0 / 30.0
- Fleet: **6 machines / 28 GPUs** by model {'GPU-12GB': 4, 'GPU-16GB': 16, 'GPU-24GB': 8}; demand/capacity **3.3214**
- **Retry/failure (trace-observed):** pass/failed/killed 28/3/2; multi-attempt 3, retries 3 (rate 9.091%), wasted GPU-hours 0.058.

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. SLA-safe = a job (not Failed/Killed) that starts within its queue-wait budget (max(1h, 2× runtime)). Cost bills every node ever powered for the makespan at the documented per-GPU price. Same fleet/prices/jobs across policies — only the scheduling decision differs. **Headline = `best_fit` (a real scheduling baseline, NOT fifo).**

| policy | goodput/$ | completed | GPU-hrs | infra $ | qw p95 (s) | qw p99 (s) | mean compl (s) | util % | frag blocks | backfill |
|---|---|---|---|---|---|---|---|---|---|---|
| fifo | 840.00 | 13 | 49.5 | 126 | 7,280 | 7,300 | 6,279 | 58.9 | 41 | 0 |
| first_fit | 1,230.35 | 25 | 49.5 | 113 | 5,740 | 6,760 | 3,295 | 65.8 | 21 | 17 |
| best_fit *(headline)* | 1,362.98 | 25 | 49.5 | 102 | 5,740 | 5,820 | 3,455 | 72.9 | 19 | 18 |
| first_fit_decreasing | 1,230.35 | 25 | 49.5 | 113 | 5,740 | 6,760 | 3,295 | 65.8 | 21 | 17 |
| greedy_packing | 1,362.98 | 25 | 49.5 | 102 | 5,740 | 5,820 | 3,455 | 72.9 | 19 | 18 |
| topology_aware | 1,362.98 | 25 | 49.5 | 102 | 5,740 | 5,820 | 3,455 | 72.9 | 19 | 18 |
| utilization_aware | 1,230.35 | 25 | 49.5 | 113 | 5,740 | 6,760 | 3,295 | 65.8 | 21 | 17 |
| constraint_aware **(CA)** | 1,362.98 | 25 | 49.5 | 102 | 5,740 | 5,820 | 3,455 | 72.9 | 19 | 18 |

## Scheduler-pressure analysis (the Philly point)

### Queueing
- constraint_aware queue wait p50/p95/p99 = 1,340 / 5,740 / 5,820 s; queue-collapse events 0; starvation events 0.
- vs naive FIFO (head-of-line, no backfill): p95 7,280 s, completed 13 — constraint_aware reduces queue latency.

### Cluster saturation / utilization
- constraint_aware mean util 72.9% (p95 100.0%); FIFO mean 58.9%.

### Fragmentation (jobs blocked despite sufficient aggregate GPUs)
- fifo: 41 block-events (48.81% of scheduling attempts)
- first_fit: 21 block-events (5.51% of scheduling attempts)
- best_fit: 19 block-events (4.40% of scheduling attempts)
- first_fit_decreasing: 21 block-events (5.51% of scheduling attempts)
- greedy_packing: 19 block-events (4.40% of scheduling attempts)
- topology_aware: 19 block-events (4.40% of scheduling attempts)
- utilization_aware: 21 block-events (5.51% of scheduling attempts)
- constraint_aware: 19 block-events (4.40% of scheduling attempts)

### Large vs small job fairness (mean queue wait s by GPU-count class)

| policy | 1 GPU | 2-4 | 5-8 | 9+ |
|---|---|---|---|---|
| fifo | 3,954 | 4,376 | 4,670 | n/a |
| first_fit | 499 | 1,390 | 3,695 | n/a |
| best_fit | 856 | 1,372 | 3,460 | n/a |
| first_fit_decreasing | 499 | 1,390 | 3,695 | n/a |
| greedy_packing | 856 | 1,372 | 3,460 | n/a |
| topology_aware | 856 | 1,372 | 3,460 | n/a |
| utilization_aware | 499 | 1,390 | 3,695 | n/a |
| constraint_aware | 856 | 1,372 | 3,460 | n/a |

### Backfill
- constraint_aware backfill placements: 18 (small jobs run while a larger earlier-submitted job waits). FIFO performs **no** backfill (strict head-of-line) → 0.

### Retry / failure behaviour
- Trace-observed: 3 retries across 3 multi-attempt jobs, 0.058 wasted GPU-hours. The scheduler does not re-simulate failures; constraint_aware reduces the queueing/fragmentation that drives preemption-retries (directional, not a re-simulation).

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)

- **Outcome:** `TIE`  ·  margin vs `best_fit`: **+0.00%** on goodput/$
- **Sanity vs FIFO:** constraint_aware beats naive FIFO (+62.26%).

### What improved / what did not

- **Big win vs naive FIFO** (+62.3% goodput/$): FIFO's strict head-of-line blocking lets one large queued job stall the whole cluster; constraint_aware (backfill + consolidation + big-node reservation) keeps GPUs busy and cuts queue wait across every job-size class.
- **vs `best_fit` (strongest scheduling baseline):** +0.00% — on Philly the GPU **type/price is unknown**, so constraint_aware's heterogeneous-pricing lever (which won on Alibaba) is inactive; it **TIE** the strongest packing/scheduling baseline here. Honest: the Philly value is a throughput/fairness **safety** win over naive scheduling, not pricing alpha over an already-good packer.

## Honest limits

- Temporal scheduler over the trace's job durations; failures/retries are trace-observed, **not** re-simulated. GPU prices are documented priors (±50%), identical across policies; Philly has no real GPU model, so the fleet is effectively single-price here.
- The `cluster_gpu_util` / cpu / mem CSVs are not parsed in this PR (0 utilization samples). No CPU/host-mem request or deadline in the job log. Philly public data is **not customer telemetry**.
- **Not production-real savings.** Directional simulator result only.

