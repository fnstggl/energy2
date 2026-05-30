# Model Residency Decision Engine — GenAI 2026 Backtest

> **Directional simulator / backtest result — not production savings** (`docs/RESULTS.md` §8). The Model Residency Decision Engine is **recommendation-only in real/customer mode**; this backtest runs it in **simulator mode** (mutates only simulated state — no real cluster, router, or serving engine is touched). The engine **never** changes which model/adapter is requested; it only recommends placement / routing / prewarm / evict. Residency metrics are diagnostics — the primary KPI is unchanged: SLA-safe goodput per infrastructure dollar.


- **Source:** `tests/fixtures/alibaba_genai_sample`
- **Requests:** 60 · **simulated GPUs:** 4
- **Cold-start calibration (s):** {'pipeline_inference': 14.7025, 'basemodel_load': 2.782, 'controlnet_load': 4.396, 'lora_load': 3.696}

## Per-request policy comparison (this engine vs baselines)

| policy | goodput/$ | model hit-rate | adapter hit-rate | cold starts | cold p50/p95/p99 (s) | route→resident | prewarm | evictions | warm-pool GPU-h | SLA viol | e2e p99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fifo_round_robin | 0.8734 | 0.7 | 0.7143 | 18 | 2.782/6.478/6.478 | 0 | 0 | 2 | 0.0 | 0 | 74.782 |
| sla_aware_least_queue | 0.8734 | 0.7 | 0.7143 | 18 | 2.782/6.478/6.478 | 0 | 0 | 2 | 0.0 | 0 | 74.782 |
| sla_aware_naive_prewarm | 0.8734 | 1.0 | 1.0 | 0 | —/—/— | 0 | 0 | 0 | 0.0 | 0 | 72.0 |
| affinity_only | 0.8734 | 0.9 | 0.9286 | 6 | 2.782/6.478/6.478 | 0 | 0 | 0 | 0.0 | 0 | 79.0 |
| residency_engine | 0.8734 | 0.9 | 0.9286 | 6 | 2.782/6.478/6.478 | 54 | 6 | 0 | 0.0 | 0 | 79.0 |

### Reading these numbers honestly

- This is a **small-sample** per-request replay (the committed fixture). On it, residency routing **cuts cold starts and raises the residency hit-rate** (residency-blind FIFO/least-queue vs affinity/engine); where the SLA is met by every policy, goodput/$ is similar because the shared fixed GPU pool gives the same cost denominator.
- The **economic** payoff of residency at scale (where cold starts cause SLA violations) is carried by the **full-trace tick-based ablation** below — preserved unchanged.
- `sla_aware_naive_prewarm` eliminates cold starts but pays a warm-pool cost for every distinct model held warm beyond pool capacity (zero here only because all models fit); at the trace's ~80-model scale this is expensive.

## Before/after — existing full-trace tick-based ablation (preserved)

> Source: committed `alibaba_genai_ablation_summary.json` (full 26,392-request trace). Unchanged by this PR.

| config (full trace) | goodput/$ | mean cold-start (s) | e2e p99 (s) | replica GPU-hrs |
|---|---|---|---|---|
| constraint_aware (with affinity/prewarm) | 9.8404 | 2.87 | 53.39 | 894.0 |
| constraint_aware (no affinity) | 7.0548 | 23.55 | 66.43 | 1247.0 |
| sla_aware (headline) | 5.1938 | 23.55 | 1219.35 | 1142.0 |
| fifo | 1.7676 | 23.55 | 53.46 | 4977.0 |

- Existing attribution: **affinity/prewarm ≈ 62.1%** of the +89.46% constraint_aware-vs-sla_aware gain; anticipatory sizing ≈ 37.9%. The decision engine operationalises the affinity/prewarm lever as explicit per-request routing.

## Method / honesty

- The decision engine optimises SLA-safe goodput/$ subject to hard safety vetoes (memory headroom, SLA, thermal, topology, region, telemetry confidence). It never substitutes the requested model/adapter.
- Simulator mode mutates only simulated `ModelLocationState`. Real/customer mode is recommendation-only (`executable_in_real_cluster=False`).
- All policies share one fixed simulated GPU pool (same cost denominator) except `sla_aware_naive_prewarm`, which is charged for replicas held warm beyond pool capacity. Cold-start magnitudes are the trace's pipeline-layer calibration, not a per-request causal join.
- **No production-savings claim.** `docs/RESULTS.md` §8 production-claim gate is not met.

