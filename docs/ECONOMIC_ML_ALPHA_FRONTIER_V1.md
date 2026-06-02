# Economic ML Alpha — Frontier Refresh v1

> **Research / shadow refresh.** No production scheduler / scorer / residency /
> frontier / overlay module is modified. No real execution. No production-
> savings claim. No oracle / FIFO headline. No invented constants. Public /
> artifact data is never pilot telemetry. FaaS cold-start is never promoted to
> GPU model-load ML; autoscaling proxies are never reported as measured serving
> autoscaling.
>
> **Read first:** `docs/FRONTIER_SIGNAL_BOUNDED_INGEST_V1.md` (the ingest),
> `docs/ECONOMIC_ML_ALPHA_V1.md` (the baseline audit),
> `docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`.
>
> **Evidence:** `data/external/forecasting/economic_ml_alpha_frontier_v1/
> {summary, target_catalog, trained_models, economic_alpha_eval}.json`.

## 1. Question

ML Alpha v1 found `cache_reuse_pct` **shadow-ready** but flagged it
**single-dataset (SwissAI only)** with *no cross-dataset generalization
evidence*. This refresh tests that caveat with the newly ingested **Mooncake**
trace (a second, production-derived reuse source) and records honest status for
the **Huawei** cold-start prior and the **Alibaba v2025** autoscaling/queue proxy.

The binding KPI is unchanged
(`sla_safe_goodput_per_dollar = (output_tokens if sla_met else 0) /
(gpu_cost + energy_cost + migration_cost + cold_start_cost − cache_value)`);
cache reuse feeds the `cache_value` term.

## 2. Label compatibility (the crux)

| | SwissAI `bucket_reuse` | Mooncake |
|---|---|---|
| Reuse label | **MEASURED** `reuse_percentage` = reused_buckets/total_buckets | **DERIVED** global-prefix-cache simulation from `hash_ids` |
| Block list | `bucket_ids` (full list, summarized) | `hash_ids` (full list) |
| Group key | `model_id` (5 families) | `trace_name` (4 traces) |
| Provenance | author-measured | reconstructed by us |

The two use the **same definition** (reused blocks / total blocks) but
**different provenance**. Only a **harmonized `high_reuse` proxy** (≥ 50%) over
a **shared minimal feature space** (`block_count`, decision-time
`rolling_group_reuse_mean`, `rolling_block_count_mean`) is cross-comparable.
**A literal cross-dataset validation of the MEASURED SwissAI label is
impossible** — Mooncake has no measured reuse label.

## 3. Experiment matrix (AUROC for `high_reuse`)

Strongest realistic baseline = per-group base rate. Real data: 67,190 SwissAI
rows (5 families, bounded 10 MiB/config) + 63,240 Mooncake rows.

| Experiment | label | baseline AUROC | best ML AUROC | vs **its own** baseline |
|---|---|---:|---:|---:|
| SwissAI-only (time holdout) | measured | **0.715** | 0.760 | **+6.3%** |
| Mooncake-only (sequence holdout) | derived proxy | 0.500¹ | 0.874 | +74.7%¹ |
| SwissAI → Mooncake (transfer) | mixed | 0.500² | 0.637 | — |
| Mooncake → SwissAI (transfer) | mixed | 0.500² | 0.607 | — |

¹ **Artifact caveat.** Mooncake's per-trace baseline is near-degenerate
(AUROC ≈ 0.5) and the derived reuse proxy is autocorrelated, so the
decision-time rolling-reuse feature predicts it strongly. The +74.7% is **not**
evidence of measured alpha.

² **Degenerate-baseline caveat.** In cross-dataset transfer the per-group
baseline cannot use the source dataset's groups, so it collapses to ~0.5.
Comparing transfer to that 0.5 inflates the gain — **the honest test is vs the
TARGET dataset's OWN baseline**, below.

## 4. The rigorous cross-dataset test

Does a model trained on A predict B **better than B's own strongest baseline**?

| Direction | transfer AUROC | target's own baseline | verdict |
|---|---:|---:|---|
| SwissAI → Mooncake | 0.637 | 0.500 (Mooncake) | beats (+27.4%), but target baseline is degenerate |
| **Mooncake → SwissAI** | **0.607** | **0.715 (SwissAI)** | **FAILS (−15.1%)** |

Reverse transfer (the meaningful direction, since SwissAI carries the measured
label and a *strong* per-model baseline) **underperforms SwissAI's own
baseline**. Reuse structure is *partially* universal (transfer AUROC 0.61–0.64
> chance), but not enough to beat the target's own baseline in both directions.

## 5. Verdict — does `cache_reuse_pct` remain shadow-ready beyond SwissAI?

**No.** Status moves to **`single_dataset_promising_only`**.

| | ML Alpha v1 | Frontier Refresh v1 |
|---|---|---|
| `cache_reuse_pct` | `shadow_ready_for_integration_review` *(single-dataset caveat, untested)* | **`single_dataset_promising_only`** *(caveat now TESTED and confirmed)* |

The v1 caveat ("needs a second dataset") is now **evidenced**: the second source
(Mooncake) is a **derived proxy**, and cross-dataset transfer does **not** beat
the target's own baseline both ways. `becomes_more_production_plausible:
false`. SwissAI-only still beats its strong baseline (+6.3%), so the within-
dataset signal is real — it just does not generalize on a derived-proxy second
source. A **measured** second source (pilot per-request cache_hit) is still
required before shadow-ready.

## 6. Huawei cold-start — `simulator_prior_calibrated` (NOT GPU ML)

`cold_start_prior` in `trained_models.json`. 239,405 measured FaaS cold-start
events give a calibration of the **cost structure**: scheduling 51.1%,
deploy_code 23.4%, deploy_dependency 15.5%, pod_allocation 10.1%; latency p50
2.33 s / p90 9.34 s / p99 42.1 s. This calibrates the **shape** of the
Economic ML Alpha v1 cold-start simulator-prior sweep **only**.

**GPU cold-start ML remains `blocked_by_missing_labels`.** `deploy_code` /
`deploy_dependency` are code/dependency download, **not** GPU model-weight load;
the source measures **no** GPU model-load. Promotion to GPU cold-start ML is
**not** permitted (`is_gpu_model_load: false`, `calibration_only: true`).

## 7. Alibaba v2025 — autoscaling/queue **proxy**

`autoscaling_queue_proxy` in `trained_models.json`. A proxy tail-risk model
(will an instance see scheduler_delay > p90?) trained on
`[gpu_count, cpu, memory, is_gpu, app_instance_count]`. This is an **instance-
level scheduler-delay proxy**, NOT per-request serving queue-wait and NOT a
measured autoscaling event. The existing **AcmeTrace** job-level `queue_wait` +
**CARA** queue features (ML Alpha v1) remain the stronger queue evidence; Alibaba
v2025 adds only a proxy. `has_measured_serving_autoscaling: false`.

## 8. Carried-from-v1 targets (unchanged — no new data)

Mooncake/Huawei/Alibaba supply no latency/memory/energy/cost labels, so:
`ttft_s` = promising_needs_validation, `tpot_s`/`e2e_latency_s`/`energy_kwh` =
diagnostic_only, `peak_vram_gb` = shadow_ready (single-dataset, unchanged),
`estimated_gpu_cost_usd` = diagnostic_only_deterministic_formula.

## 9. What can / cannot be claimed externally

**Can:** Mooncake is a second *independent* reuse dataset (proxy-grade) whose
structure is *consistent with* SwissAI's; Huawei calibrates cold-start cost
**structure** for the simulator prior; Alibaba adds an instance-level
scheduler-delay proxy. **Cannot:** cross-dataset *measured* validation of
`cache_reuse_pct` (Mooncake label is derived; reverse transfer fails); GPU
model-load cold-start forecasting from Huawei (FaaS ≠ GPU); measured serving
autoscaling forecasting from Alibaba (proxy only); **any production savings**.

## 10. Remains pilot-only

Server-class GPU `model_load_duration_s`; measured serving autoscaling events;
per-request migration / cache-loss seconds; real per-request measured
`cache_hit`. Consistent with ML Alpha v1 and Frontier Discovery v2.
