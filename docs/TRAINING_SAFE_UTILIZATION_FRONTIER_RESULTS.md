# Training Safe Utilization Frontier — v1 Results

> **Simulator / public-trace benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Training Frontier v1 reads only the COMMITTED Philly + Alibaba GPU v2023 backtest summaries. The serving Safe Utilization Frontier Controller, the robust energy engine, and every committed benchmark artifact are **unchanged**. Real-cluster execution is **disabled by default**.

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/PHILLY_BACKTEST_RESULTS.md`, `docs/ALIBABA_GPU_BACKTEST_RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`.

## 1. Configuration

- **Trace sources:** Philly + Alibaba GPU v2023
- **Tie band:** ±1.0 % goodput/$
- **Default safety thresholds:**
  - `max_queue_wait_p95_s`: 21600.0
  - `max_queue_wait_p99_s`: 43200.0
  - `max_starvation_rate_pct`: 5.0
  - `max_fragmentation_block_rate_pct`: 25.0
  - `max_gang_scheduling_failure_pct`: 10.0 *(disabled for Philly — see §3 missing-signals note)*
- **Real-cluster execution:** disabled by default.

## 2. Per-trace summary

| trace | current_policy | current goodput/$ | training_frontier_v1 | Δ vs current | verdict | action | safe / unsafe / insufficient |
|---|---|---|---|---|---|---|---|
| `philly` | `constraint_aware` | 1,362.98 | 1,362.98 → `constraint_aware` | +0.000% | **TIE** | `KEEP_CURRENT_POLICY` | 8 / 0 / 0 |
| `alibaba_gpu` | `constraint_aware` | 8.24 | 8.24 → `constraint_aware` | +0.000% | **TIE** | `KEEP_CURRENT_POLICY` | 5 / 1 / 0 |

## 3. Per-trace frontier sweeps + missing-signals notes

### philly frontier sweep

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `best_fit` | 1,362.98 | 0.729100 | 5,820.00 | 0.0000 | 3.23 | 56.25 | **SAFE** |
| `constraint_aware` | 1,362.98 | 0.729100 | 5,820.00 | 0.0000 | 3.23 | 56.25 | **SAFE** |
| `fifo` | 840.00 | 0.588700 | 7,300.00 | 0.0000 | 3.23 | 0.0000 | **SAFE** |
| `first_fit` | 1,230.35 | 0.658200 | 6,760.00 | 0.0000 | 3.23 | 53.12 | **SAFE** |
| `first_fit_decreasing` | 1,230.35 | 0.658200 | 6,760.00 | 0.0000 | 3.23 | 53.12 | **SAFE** |
| `greedy_packing` | 1,362.98 | 0.729100 | 5,820.00 | 0.0000 | 3.23 | 56.25 | **SAFE** |
| `topology_aware` | 1,362.98 | 0.729100 | 5,820.00 | 0.0000 | 3.23 | 56.25 | **SAFE** |
| `utilization_aware` | 1,230.35 | 0.658200 | 6,760.00 | 0.0000 | 3.23 | 53.12 | **SAFE** |

**Controller decision:** `KEEP_CURRENT_POLICY` → policy `constraint_aware` (1,362.98 goodput/$)
**Reason:** current candidate 'constraint_aware' within KPI deadband (0.0000 ≤ 0.01) and packing-density deadband (0.0000 ≤ 0.05)

**Missing signals on Philly (not invented):**
- per-policy gang-scheduling failure: NOT cleanly labelled; the gate is **disabled by default**. `failed_or_killed_run` includes non-gang causes.
- per-job completion p95 / p99: not reported by the committed summary (only `mean_completion_s`).
- GPU model price heterogeneity: Philly has no GPU model column.

### alibaba_gpu frontier sweep

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `best_fit` | 7.77 | 0.897100 | — | 0.0000 | 9.70 | — | **SAFE** |
| `constraint_aware` | 8.24 | 0.897860 | — | 0.0000 | 9.98 | — | **SAFE** |
| `fifo` | 5.27 | 0.665460 | — | 0.1925 | 28.63 | — | **UNSAFE** |
| `first_fit` | 7.54 | 0.903640 | — | 0.0000 | 9.11 | — | **SAFE** |
| `first_fit_decreasing` | 7.48 | 0.896710 | — | 0.0000 | 8.35 | — | **SAFE** |
| `greedy_packing` | 7.55 | 0.874720 | — | 0.0000 | 8.96 | — | **SAFE** |

**Controller decision:** `KEEP_CURRENT_POLICY` → policy `constraint_aware` (8.24 goodput/$)
**Reason:** current candidate 'constraint_aware' within KPI deadband (0.0000 ≤ 0.01) and packing-density deadband (0.0000 ≤ 0.05)

**Missing signals on Alibaba GPU (not invented):**
- per-job queue wait p95 / p99: NOT reported by the static packing baseline (no consistent submit / start times). Queue gates are **disabled by default**.
- starvation rate: not directly measured; `stranded_jobs / n_gpu_jobs` is reported as a fragmentation-pressure proxy and explicitly NOT labelled as starvation in the per-policy notes.
- gang-scheduling failure: NOT measured; gate **disabled by default**.
- retry / wasted GPU-hours: NOT measured; gate **disabled by default**.

## 4. What metrics were unavailable (consolidated)

| signal | Philly | Alibaba GPU |
|---|---|---|
| queue wait p95 / p99 | ✅ measured | ❌ not measured |
| starvation rate | ✅ measured | ⚠ approximated via stranded fraction (labelled in notes) |
| fragmentation block | ✅ measured (`failed_placement_rate_pct`) | ✅ measured (`fragmentation_score`) |
| backfill success | ✅ measured | ❌ not measured |
| gang-scheduling failure | ❌ not cleanly labelled | ❌ not measured |
| retry / waste GPU-hours | ✅ committed `attempt_analysis.wasted_gpu_hours_from_retries` | ❌ not measured |
| per-job p95 / p99 completion | ❌ not in summary | ❌ not reported |
| GPU model price heterogeneity | ❌ no GPU type column | ✅ measured |

## 5. Honesty / scope

- Training Frontier v1 is the **sibling** of the serving Safe Utilization Frontier Controller — it does NOT optimize request latency, does NOT use the serving rho controller, and does NOT replace any existing scheduling / packing baseline.
- Training Frontier v1 is **opt-in**, **shadow / simulator** only, and **does not mutate** real infrastructure.
- No new datasets ingested. MIT Supercloud is the next validation step (out of scope for this PR).
- Public-trace evidence only — **NOT production savings** (`docs/RESULTS.md` §8). Pilot telemetry is required to calibrate per-tenant safety thresholds.
- The committed Philly / Alibaba GPU backtest summaries are **read-only** in this benchmark; the serving frontier code is **unchanged**.

