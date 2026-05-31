# Training Safe Utilization Frontier — v1

> **Sibling of the serving Safe Utilization Frontier Controller.** Opt-in, shadow / simulator only, real-cluster execution **disabled by default**. Simulator / public-trace evidence only — **NOT production savings** (`docs/RESULTS.md` §8).

Training Frontier v1 turns the serving-side "maximum safe utilization" thesis into a training-shaped sibling:

> *How densely can we pack, backfill, and occupy GPUs for training / fine-tuning workloads before queue starvation, fragmentation, gang-scheduling failure, or completion-time risk becomes unsafe?*

It is **not** the serving rho controller. The training frontier candidate space is multi-dimensional (packing density, backfill aggressiveness, large-job reservation, fragmentation budget, gang-scheduling strictness, heterogeneity preference, price-aware GPU routing) — not a scalar rho.

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/PHILLY_BACKTEST_RESULTS.md`, `docs/ALIBABA_GPU_BACKTEST_RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`.

## 1. Scope (binding)

- **Sibling, not subclass.** The training-frontier modules (`aurelius/frontier/training_*.py`) do NOT import or extend the serving rho controller (`controller.py`, `dynamic_controller.py`, `estimator.py`, `dynamic_estimator.py`). The action space is workload-shape-aware: `RECOMMEND_TRAINING_FRONTIER` / `KEEP_CURRENT_POLICY` / `LOWER_PACKING_PRESSURE` / `RESERVE_FOR_LARGE_JOBS` / `INSUFFICIENT_TELEMETRY`.
- **Opt-in.** No code path enables Training Frontier by default.
- **No new datasets.** Reads only the **committed** Philly + Alibaba GPU v2023 backtest summaries already in `data/external/`. MIT Supercloud is the **next** validation dataset and is intentionally out of scope for this PR.
- **No ML training.** Frontier candidates are descriptors over the **existing measured policies** in those committed summaries. The training estimator never re-runs the schedulers / packers and never invents missing telemetry.
- **No production mutation.** `TrainingFrontierDecision.executable_in_real_cluster=False` at construction; the v1 real-execution path is a deliberate stub. Real execution requires the same explicit opt-in pattern used by the serving controller (`execute_training_frontier_decision(..., allow_real_execution=True)` plus a non-stub executor).
- **No robust-energy-engine change.** This module only adds new files.
- **Honest missing-signal reporting.** Where a trace does not measure a signal (Alibaba's queue wait, retry waste, gang-failure breakdown; Philly's GPU type, completion p99), the field stays `None` and the corresponding safety gate is either disabled by default or surfaces as `INSUFFICIENT_TELEMETRY`.

## 2. Architecture

```
TrainingWorkloadProfile  ──┐
                            ▼
committed Philly summary ──►  estimate_philly_training_frontier
committed Alibaba summary ──►  estimate_alibaba_gpu_training_frontier
                            │
                            ▼
                    list[TrainingFrontierPoint]
                    (one per measured policy)
                            │
                            ▼
              choose_training_frontier_target
              (KPI-max safe; deadband; large-job
               reservation; lower-pressure on UNSAFE)
                            │
                            ▼
                  TrainingFrontierDecision
                  (RECOMMEND / KEEP / LOWER /
                   RESERVE / INSUFFICIENT)
                            │
                            ▼
         execute_training_frontier_decision
             (shadow / simulator only by
              default; real exec disabled)
```

## 3. Modules

| file | purpose |
|---|---|
| `aurelius/frontier/training_models.py` | `TrainingWorkloadProfile`, `TrainingFrontierCandidate`, `TrainingFrontierPoint`, `TrainingFrontierDecision` + categorical enums |
| `aurelius/frontier/training_safety.py` | `TrainingSafetyConfig` + `classify_training_frontier_point` (queue / starvation / fragmentation / gang / retry vetoes — never weighted into a score) |
| `aurelius/frontier/training_philly.py` | Philly estimator — reads `data/external/philly/processed/philly_backtest_summary.json` + `attempt_analysis` (retry waste) |
| `aurelius/frontier/training_alibaba_gpu.py` | Alibaba GPU estimator — reads `data/external/alibaba_gpu/processed/alibaba_gpu_backtest_summary.json` |
| `aurelius/frontier/training_controller.py` | `choose_training_frontier_target` + recommendation-only `execute_training_frontier_decision` (no-op stub for real mode) |
| `aurelius/frontier/training_shadow.py` | `TrainingFrontierShadowLog` JSONL writer / reader |

## 4. Frontier candidates per trace

Each committed policy in the source backtest is mapped 1:1 to a `TrainingFrontierCandidate` descriptor that **labels what the policy emphasizes**. Descriptors are interpretive (occupancy / packing / backfill / reservation / gang-strictness / heterogeneity / price-aware routing) but they do **not** modify the measured KPI in any way — every KPI on every point is sourced directly from the committed simulator output.

| trace | covered candidate dimensions |
|---|---|
| **Philly** | packing_density_target, backfill_aggressiveness, large_job_reservation_fraction, gang_scheduling_strictness |
| **Alibaba GPU v2023** | occupancy_target, packing_density_target, fragmentation_budget, heterogeneity_preference, price_aware_gpu_routing_enabled |

## 5. Safety vetoes (transparent, never weighted)

`TrainingSafetyConfig` exposes every gate explicitly. The pre-registered defaults are:

| gate | default | rationale |
|---|---|---|
| `max_queue_wait_p95_s` | 6 h | matches Philly's `SLA_WAIT_ABS_S` baseline grace |
| `max_queue_wait_p99_s` | 12 h | 2× p95 budget |
| `max_starvation_rate_pct` | 5 % | "starvation" rate proxy (Philly: `starvation_events` per scheduled) |
| `max_fragmentation_block_rate_pct` | 25 % | Philly: `failed_placement_rate_pct`; Alibaba: `fragmentation_score` × 100 |
| `max_gang_scheduling_failure_pct` | 10 % | **disabled by default for Philly** (no clean gang-failure label) and Alibaba (not measured) |
| `max_retry_waste_gpu_hours` | `None` | gate disabled; populated when the trace reports it |
| `min_completed_work_ratio` | 0.50 | guards against "win by under-running" |
| `min_telemetry_confidence` | `low` | floor; controllers may demand higher |

Every veto produces a structured reason code; the full set is exported as `ALL_TRAINING_VETOES`.

## 6. What's measured vs not (honestly)

| signal | Philly | Alibaba GPU v2023 |
|---|---|---|
| queue wait p95 / p99 | ✅ measured | ❌ not measured (queue gate disabled) |
| starvation rate | ✅ measured | ⚠ stranded-fraction proxy (labelled in notes) |
| fragmentation block rate | ✅ measured (`failed_placement_rate_pct`) | ✅ measured (`fragmentation_score`) |
| backfill success rate | ✅ measured | ❌ not measured |
| gang-scheduling failure | ❌ not cleanly labelled (gate disabled) | ❌ not measured (gate disabled) |
| retry / wasted GPU-hours | ✅ from `attempt_analysis` | ❌ not measured |
| per-job p95 / p99 completion | ❌ not in summary | ❌ not reported |
| GPU model price heterogeneity | ❌ no GPU type column | ✅ measured |

## 7. Controller rules (in order)

1. Telemetry-confidence floor → `INSUFFICIENT_TELEMETRY`.
2. Current policy `UNSAFE` (with `prefer_lower_pressure_on_current_unsafe=True`) → `LOWER_PACKING_PRESSURE`.
3. No safe candidates → `LOWER_PACKING_PRESSURE` toward the lowest-pressure option.
4. Best safe candidate selected by max predicted goodput/$.
5. Optional conservative-margin: step back when the best safe is adjacent to an UNSAFE point.
6. If the best safe candidate shows large-job starvation AND a candidate with `large_job_reservation_fraction > 0` exists → `RESERVE_FOR_LARGE_JOBS`.
7. KPI / packing-density deadband → `KEEP_CURRENT_POLICY`.
8. Otherwise → `RECOMMEND_TRAINING_FRONTIER`.

## 8. Public-trace v1 results (summary)

See `docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md` for the full per-trace tables. The Philly / Alibaba GPU `constraint_aware` baselines are already on the safe frontier under the default safety gates, so the v1 controller emits `KEEP_CURRENT_POLICY` (verdict = `TIE`) when `constraint_aware` is the current policy. When the current policy is the FIFO sanity baseline, the controller correctly upgrades to the safe peak (verdict = `TRAINING_FRONTIER_WIN`).

## 9. Hard non-goals

- ❌ MIT Supercloud ingestion (next-step validation; out of scope here).
- ❌ Apply the serving rho controller to training workloads.
- ❌ Make Training Frontier the `constraint_aware` default.
- ❌ Real-cluster execution by default.
- ❌ Weaken existing safety gates.
- ❌ Modify the robust energy engine.
- ❌ Train ML models.
- ❌ No claims of production savings.
- ❌ Hide losses or invent missing signals.

## 10. Remaining gaps before MIT Supercloud ingestion

- **Pilot telemetry calibration.** Per-tenant queue / starvation / fragmentation thresholds need pilot calibration. The pre-registered defaults are conservative starting points, not customer SLAs.
- **Real per-policy gang-scheduling failure** is not cleanly measured by either Philly or Alibaba; MIT Supercloud reports per-job allocation patterns that can refine this.
- **Per-job completion p95 / p99** is not in the committed Philly summary; adding it requires re-running the scheduling simulator (deliberately out of scope here — Training Frontier v1 reads only committed JSONs).
- **Heterogeneous GPU price routing** is a candidate descriptor but Alibaba's static packing baseline does not currently expose the per-policy price-routing decision granularly. MIT Supercloud has GPU-type telemetry that can resolve this.
- **Real-executor implementation** remains a deliberate stub — promoting it requires the same binding-boundary work documented in `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` §"Real-mode execution boundary", applied to the training-side action space.
