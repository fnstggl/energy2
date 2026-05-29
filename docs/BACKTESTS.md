# Canonical Energy Backtests — frozen CAISO/PJM/ERCOT 1000-job benchmark

> Read `docs/RESULTS.md` (reporting standard) and `docs/ENERGY_SYSTEM_MAP.md`
> (energy engine map) first. This document defines the **frozen** canonical
> energy backtest so every future optimizer / forecasting / adapter PR can be
> compared apples-to-apples against a fixed reference.
>
> **Simulator/benchmark result — directional only, NOT a production-savings
> claim.** Live customer-telemetry calibration is required before any external
> savings number (`docs/RESULTS.md` §8).

## 1. What it is

A single, fully-deterministic energy backtest implemented in
`aurelius/benchmarks/canonical_backtests.py` and runnable via
`scripts/run_canonical_backtests.py`. It runs the **existing robust energy
engine** (`aurelius.optimization.scheduler.JobScheduler`) and the
**constraint-aware energy adapter**
(`aurelius.constraints.energy_adapter.EnergyArbitrageAdapter`) over a frozen
1000-job CAISO/PJM/ERCOT workload trace and emits a golden JSON summary.

```bash
python scripts/run_canonical_backtests.py            # human-readable table
python scripts/run_canonical_backtests.py --json     # golden JSON
python scripts/run_canonical_backtests.py --write-golden   # regenerate snapshot
```

Golden snapshot: `aurelius/benchmarks/golden/canonical_energy_backtest.json`.
Pinned by `tests/test_canonical_energy_backtest.py`.

## 2. Fixed workload trace

| Property | Value |
|---|---|
| Job count | **1000** (`CANONICAL_JOB_COUNT`) |
| Seed | **20260201** (`CANONICAL_SEED`) |
| Job ids | stable `job-00000` … (no uuid) |
| Region eligibility | all 3 ISOs, except a fixed per-class data-residency pin fraction |

Fixed job mix (`CANONICAL_WORKLOAD_MIX`), by fraction:

| workload_type | fraction | migration_cost_hours | latency-pinned |
|---|---|---|---|
| `llm_batch_inference` | 0.35 | 0.10 | no |
| `data_processing` | 0.15 | 0.05 | no |
| `scheduled_batch` | 0.15 | 0.10 | no |
| `fine_tuning` | 0.10 | 0.25 | no |
| `training` | 0.10 | 0.50 | no |
| `realtime_inference` | 0.15 | None (cannot migrate) | **yes** |

Runtime, slack (flexibility window), power, gpu_count and region-pin are drawn
from fixed per-class choice sets with the canonical seed, so the trace is
byte-identical on every machine. Deadlines = `earliest_start + runtime + slack`.

## 3. Fixed energy data windows

| ISO | region | DA file | RT file |
|---|---|---|---|
| CAISO | us-west | `data/caiso_us_west_dam.csv` | `data/caiso_us_west_rt.csv` |
| PJM | us-east | `data/pjm_us_east_dam.csv` | `data/pjm_us_east_rt.csv` |
| ERCOT | us-south | `data/ercot_us_south_dam.csv` | `data/ercot_us_south_rt.csv` |

Window: **2026-02-01T00:00Z → 2026-02-27T00:00Z** (fully inside the DA + RT
ranges of all three ISOs). DA = planning price; RT = realized settlement price
the schedule is actually charged. Prices are loaded with the stdlib `csv`
module (no PyYAML / DB / ingestion dependency).

## 4. Fixed baselines / policies

| Policy | What it is |
|---|---|
| `fifo` | "No optimization" — each job ASAP in the default region (`create_baseline_schedule`). Sanity baseline. |
| `current_price_only` | `current_price_only_policy` — cheapest region at earliest_start, no temporal shift. |
| `greedy_energy` | Aggressive comparison baseline — cheapest (region, hour) in the feasible window on DA price, **no** migration/basis/SLA awareness. |
| `robust_energy_standalone` | The **existing energy engine** standalone (`JobScheduler.solve(method="greedy")`). The energy source of truth. |
| `sla_aware` | Energy engine + deadline/latency safety only (reverts latency-pinned + deadline-unsafe moves to home). |
| `constraint_aware_with_energy_adapter` | Energy engine routed through the full constraint-aware gates (eligibility / destination / SLA / KPI). |

## 5. Fixed metrics

Per policy (scored on **realized RT** prices):

- `realized_energy_cost_usd`, `da_planned_cost_usd`, `da_rt_basis_usd`
  (realized − planned; >0 = adverse basis),
- `gpu_infra_cost_usd`, `network_cost_usd`, `total_infra_cost_usd`,
- `migrations`, `deadline_misses` (SLA violations),
- `sla_compliant_goodput` (`token_equivalent` = gpu_count × runtime for jobs
  meeting the deadline; deadline miss ⇒ 0),
- **`sla_safe_goodput_per_infra_dollar`** — the primary KPI (`docs/RESULTS.md`
  §1),
- `cost_per_sla_compliant_job`,
- `gross_energy_savings_vs_fifo_usd`, `net_energy_savings_vs_fifo_usd`.

Deadline scoring is **warmup-aware**: a job relocated to a non-home region pays
`migration_cost_hours` of warmup that the warmup-blind energy engine does not
budget for. This is what surfaces the constraint-aware wrapper's safety value.

Adapter diagnostics (on the wrapped policy): `candidates_generated`,
`candidates_accepted`, `candidates_rejected`, `candidates_deferred`,
`rejection_reasons` (stable reason codes, histogrammed).

## 6. Fixed reporting

The golden summary reports the standalone energy result, the constraint-aware
wrapped result, and the **delta** between them
(`standalone_vs_wrapped_delta`): energy cost, deadline misses, goodput/$,
candidates generated / accepted / rejected / deferred, and the rejection-reason
histogram (SLA/risk vetoes). Goodput unit is `token_equivalent`; the primary KPI
is SLA-safe goodput per infrastructure dollar.

### Reference result (committed golden snapshot)

The frozen run shows the **alpha-vs-safety split** `docs/RESULTS.md` §6
requires: the standalone energy engine takes the lowest energy cost but, being
warmup-blind, misses 143 deadlines; the constraint-aware wrapper eliminates
**all** deadline misses and reverts 137 unsafe critical-interactive moves, at a
near-zero goodput/$ cost — a **SAFETY_WIN**, not an alpha win. The energy engine
remains the canonical energy decision-maker; the wrapper is the safety layer.

## 7. Determinism / stability guarantees

The benchmark must NOT silently change when:

- PyYAML / optional dependencies change — prices load via stdlib `csv`; the
  trace uses only `random.Random(seed)`.
- random seeds differ — the seed is frozen (`CANONICAL_SEED`).
- local environment differs — no wall-clock, no network, no DB; only committed
  CSVs.
- file / dict / set ordering differs — jobs are sorted by `(submit_time,
  job_id)`; regions/policies are processed in fixed order; the golden dict is
  fully rounded and sort-keyed.

`tests/test_canonical_energy_backtest.py` asserts: job count = 1000, regions =
CAISO/PJM/ERCOT, the fixed window is inside all ISO data, two runs are
byte-identical, the live run matches the committed golden snapshot, the
standalone / `current_price_only` / `greedy_energy` results are frozen, and the
constraint-aware wrapper never regresses SLA vs FIFO.

## 8. Changing the benchmark

Regenerating the golden snapshot is a **deliberate** act. If a future change
legitimately moves the numbers (new optimizer behavior, new adapter gate, etc.):

1. `python scripts/run_canonical_backtests.py --write-golden`
2. Commit the updated `canonical_energy_backtest.json`.
3. Explain the delta in the PR body — which policy changed, why, and whether it
   is an alpha or a safety change.

A change that alters `robust_energy_standalone`,
`current_price_only`, or `greedy_energy` without an explanation is, by
definition, a change to the energy core or a frozen baseline and must be
reviewed as such (`tests/test_energy_core_preservation.py` guards the engine).
