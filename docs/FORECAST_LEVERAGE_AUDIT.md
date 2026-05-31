# Forecast Leverage Audit — what to forecast next + what NOT to

> **Audit-only PR.** No ML model is trained. No scheduler is modified. No
> production claim is made. This document maps every major Aurelius
> decision to the forecast (or prediction) that controls it, ranks
> forecasting engines by `value × decision frequency`, and decides which
> ones to build now vs defer.
>
> **Read first:**
> - `docs/RESULTS.md` (canonical reporting standard)
> - `docs/BENCHMARK_BASELINE_AUDIT.md`
> - `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` (Tier 2 telemetry evidence)
> - `docs/HF_DATASET_REGISTRY.md`
> - `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`
> - `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`
> - `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`

## 1. Scope (binding)

- **Forecast = anything that predicts a future-value or unobserved
  quantity used in a decision**: ML quantile regressor, deterministic
  Erlang-C tail, static prior, lookup table.
- **A forecast is only `build_now` when**: (a) it controls a real
  Aurelius decision, (b) at least one in-repo dataset can evaluate it,
  (c) a strongest-realistic baseline is identifiable, and (d) a success
  metric is pre-registered. The four-clause gate is tested by
  `tests/test_forecast_leverage_audit.py::test_build_now_forecasts_meet_gate`.
- **Trust hierarchy unchanged.** HF benchmark/telemetry data is Tier 2-5
  (`docs/HF_DATASET_REGISTRY.md`). No HF dataset is treated as a
  production pilot. Live customer telemetry is the only Tier 1 source
  and remains the binding production-claim gate.
- **No oracle as headline.** Per `docs/RESULTS.md` §3, all comparisons
  in this document name the strongest realistic production-feasible
  baseline (`sla_aware`, `current_price_only`, `residency_aware`,
  `dynamic_safe_frontier_estimator_v1`, etc.). Oracle is analysis-only.

## 2. Decision inventory (Aurelius decisions × forecasts)

The full machine-readable inventory is in
`data/external/frontier/forecast_leverage_audit.json` under
`decision_inventory`. The 16 decisions, summarized:

| # | decision | current module | forecast needed | freq | priority |
|---|---|---|---|---|---|
| 1 | Placement — region/zone (energy + carbon) | `aurelius/optimization/scheduler.py` | energy_price p10/p50/p90 + carbon by region, 1-72h | per-job | already_sufficient |
| 2 | Placement — heterogeneous GPU | _new_ (extends scheduler + residency) | per-(model, GPU, prompt-bin) p50/p95/p99 latency | per-request | **build_now** |
| 3 | Routing — request-level | `aurelius/frontier/dynamic_estimator.py` + `aurelius/residency/decision.py` | TTFT + queue_wait at each candidate | per-request | **build_now** |
| 4 | Queue admission | `aurelius/frontier/risk.py::estimate_queue_blowup_risk` | queue_wait p95/p99 over next N requests | per-request | **build_now** |
| 5 | Cache / prefix-aware routing | `aurelius/residency/decision.py` | prefix_reuse probability + KV persistence | per-request | **build_now** |
| 6 | Model residency (load/evict/prewarm) | `aurelius/residency/decision.py::choose_residency_decision` | cold-start latency + prewarm hit probability | 10s-1000s/hour | build_after_data_expansion |
| 7 | Autoscaling — replica count | `aurelius/frontier/dynamic_controller.py` | arrival_rate + required replicas, 1-60m | per-minute | build_after_data_expansion |
| 8 | Batching — window + max-size | `aurelius/frontier/batch_inference_controller.py` + `eval_workload_controller.py` | deadline-miss probability vs batch window | per batch boundary | build_after_data_expansion |
| 9 | Deferral — training backfill | `aurelius/frontier/training_controller.py` + `aurelius/optimization/scheduler.py` | queue-wait + completion time per heterogeneous slot | per-arrival | build_after_data_expansion |
| 10 | Migration veto | `aurelius/optimization/scheduler.py::_sla_forbids_migration` + residency cache-veto | cache-loss + cold-start cost per migration | per-candidate | build_after_data_expansion |
| 11 | SLA risk gating | `aurelius/frontier/risk.py::estimate_sla_risk` | P(SLA violation) at candidate rho | per-decision | build_after_data_expansion |
| 12 | Timeout risk gating | `aurelius/frontier/risk.py` (timeout pct in KPI) | P(timeout) per request | per-request | build_after_data_expansion |
| 13 | GPU packing — training | `aurelius/traces/gpu_packing.py` + `aurelius/frontier/training_alibaba_gpu.py` | per-(GPU, request shape) duration | per-job arrival | diagnostic_only |
| 14 | Energy shifting | `aurelius/optimization/scheduler.py` + `aurelius/forecasting/price_model.py` | energy_price p10/p50/p90 per region 1-72h | per-job + per-hour | already_sufficient |
| 15 | Carbon shifting | `aurelius/optimization/scheduler.py` + `aurelius/forecasting/carbon_model.py` | carbon intensity 1-72h | per-job + per-hour | already_sufficient |
| 16 | GPU thermal / resource pressure | _absent_ | GPU temperature + power | per-minute | not_enough_data |

## 3. Forecast engine ranking — top 12 (`value × decision frequency`)

Ranked by composite of expected alpha × decision frequency × evidence
quality. Full per-engine fields (`controls_decisions`, `target_variable`,
`horizon`, `datasets_supporting`, `primary_baseline`, `success_metric`,
`expected_alpha_label`, `expected_alpha_rationale`, `build_status`,
`blocking_gates`) are in
`data/external/frontier/forecast_leverage_audit.json` under
`forecast_engine_rankings`.

| rank | engine | controls | datasets | baseline | success metric | build status |
|---|---|---|---|---|---|---|
| 1 | **TTFT forecast** — per-(model, GPU, queue_state) p50/p95/p99 | placement (het GPU), routing, queue admission | CARA test_flat + test_queue_details | constant_per_instance_type_p99 | predicted_p99_TTFT_within_2x; routing_p99_TTFT ≤ baseline at equal goodput | **build_now** |
| 2 | **Queue-wait forecast** — p95/p99 over next N requests | queue admission, autoscaling, SLA gating | CARA queue_details + Azure LLM 2024 + Alibaba GenAI 2026 | current_queue_depth_extrapolation | predicted_p95 within 2×; admission_p99 ≤ sla_aware | **build_now** |
| 3 | **E2E latency forecast** — placement + admission | placement (het GPU), queue admission, SLA gating | CARA both configs + AgentPerfBench | constant_per_instance_type_p99 | predicted_p99 within 2×; placement_p99 ≤ sla_aware | **build_now** |
| 4 | **Heterogeneous placement scorer** — composite of TTFT + e2e | placement (het GPU) | CARA + AgentPerfBench | round_robin_placement | placement_p99 ≤ sla_aware at equal cost; goodput/$ ≥ sla_aware | **build_now** |
| 5 | **Cache / prefix-reuse forecast** | cache routing, model residency, migration veto | SwissAI qwen3_32b_buckets + qwen3_32b_bucket_reuse + CARA kv_cache_utilization | residency_aware_routing | predicted_reuse within 2×; cache_hit_rate ≥ baseline | **build_now** |
| 6 | TPOT forecast — per-output-token latency | queue admission, SLA gating, routing | CARA (moderate today; train.jsonl unlocks strong) | constant_per_instance_type_tpot | predicted_p99 within 2× | build_after_data_expansion |
| 7 | SLA / timeout violation forecast | SLA + timeout gating | CARA proxy + SwissAI status=ERROR proxy | deterministic_erlang_c_sla_risk | predicted within 2× (in pilot) | build_after_data_expansion |
| 8 | Autoscaling / replica-need forecast | autoscaling | Azure LLM 2024 + BurstGPT + Alibaba GenAI 2026 (no replica labels) | static_replica_target | predicted_arrivals within 2× per window; replica choice ≥ sla_aware | build_after_data_expansion |
| 9 | Cold-start residency forecast | model residency, migration veto | CARA kv_evictions (proxy); no labels | static_load_profile_priors | predicted_load_latency within 2× | build_after_data_expansion |
| 10 | Energy price forecast (CAISO/PJM/ERCOT) | energy shifting, region placement | CAISO + PJM + ERCOT | current_price_only | calibration coverage + savings vs current_price_only | already_sufficient |
| 11 | Carbon intensity forecast | carbon shifting, region placement | ElectricityMaps | current_price_only | diurnal carbon within 2×; carbon shifted ≥ baseline | already_sufficient |
| 12 | GPU thermal / resource pressure forecast | thermal pressure | — | static_thermal_veto | predicted temp within 2°C | not_enough_data |

## 4. Build classification

### 4.1 `build_now` (5 engines)

1. **TTFT forecast** — direct CARA evidence (9× p99 spread across GPU
   types for the same Qwen2.5-3B model); controls per-request routing
   + heterogeneous placement + queue admission.
2. **Queue-wait forecast** — CARA queue_details carries
   `num_running`, `num_waiting`, per-request `running_requests[]` +
   `waiting_requests[]` at scheduling time, which is exactly the input a
   queue-tail model needs.
3. **E2E latency forecast** — same CARA evidence as #1, with
   `actual_e2e_latency_s` as the target. Drives placement + admission +
   SLA gating.
4. **Heterogeneous placement scorer** — a composite that combines (1)
   and (3) into a per-(request, candidate instance) score. Trivial
   alpha when the underlying TTFT spread is 9×.
5. **Cache / prefix-reuse forecast** — SwissAI bucket-reuse files
   carry `reuse_percentage = reused_buckets / total_buckets`, 16,593
   strong-strength rows in the qwen3-32b subset alone; the format
   directly matches what Aurelius' residency engine consumes.

### 4.2 `build_after_data_expansion` (4 engines)

6. **TPOT forecast** — CARA test_flat is `moderate` strength
   (~9,605 rows); per-subgroup `INSUFFICIENT_SAMPLE_P99` flags fire
   below 100 rows/subgroup. Re-running the CARA audit against
   `train.jsonl` (392 MB · 359k rows) unlocks `strong` strength.
7. **SLA / timeout violation forecast** — needs explicit SLA labels.
   CARA has `num_preempted` and per-tick scheduler state but no
   per-request SLA budget label. Build after pilot telemetry lands
   with measured SLA outcomes.
8. **Autoscaling / replica-need forecast** — arrival rate is forecast-able
   from Azure 2024 + BurstGPT today, but the loop (arrival → replica
   count → latency) requires measured replica-count events. Block on
   pilot autoscaler telemetry OR a HF dataset that adds replica labels.
9. **Cold-start residency forecast** — no measured cold-start latency
   in the HF corpus. Build after pilot prewarm/eviction events.

### 4.3 `diagnostic_only` (1 engine)

10. **GPU packing — training** — the
    `docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md` audit shows
    `constraint_aware_packing` already at parity with the strongest
    realistic baseline on Alibaba GPU / MIT Supercloud / Philly. Spend
    cycles elsewhere; revisit only if a new dataset surfaces a duration
    surface gap.

### 4.4 `not_enough_data` (1 engine)

11. **GPU thermal / resource pressure forecast** — no DCGM export in
    the HF corpus; no pilot thermal data. Cannot evaluate; cannot
    promote.

### 4.5 `already_sufficient` (2 engines)

12. **Energy price forecast** — LightGBM quantile model in
    `aurelius/forecasting/price_model.py`; +11% vs `current_price_only`
    in the frozen CAISO/PJM/ERCOT backtest at 0 deadline misses.
13. **Carbon intensity forecast** — `aurelius/forecasting/carbon_model.py`
    already deployed; the marginal alpha is in sub-zone granularity.

## 5. Recommended build order (binding for the next 4 PRs)

> Each recommendation is a *forecasting research engine*, not a
> controller. Promotion of a forecaster's output into any controller
> requires (a) shadow evaluation against the primary baseline, (b)
> calibration validation, (c) safety-floor preservation
> (`docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md` §5), and (d) a
> documented promotion gate in `docs/RESULTS.md`.

| order | PR title | forecasts built | data needed | safety floor preserved |
|---|---|---|---|---|
| 1 | `forecaster_v1_per_(model,gpu)_latency_priors` | TTFT + e2e latency + heterogeneous placement scorer | CARA test_flat + test_queue_details + AgentPerfBench (already-committed) | yes — falls back to `sla_aware` + `current_price_only` |
| 2 | `forecaster_v1_queue_wait` | queue-wait p95/p99 | CARA queue_details | yes — falls back to `dynamic_safe_frontier_estimator_v1` Erlang-C tail |
| 3 | `forecaster_v1_cache_prefix_reuse` | reuse probability per bucket-id pattern | SwissAI qwen3_32b_buckets + qwen3_32b_bucket_reuse + CARA KV signals | yes — falls back to `residency_aware_routing` |
| 4 | `cara_train_strong_strength_ingest` (data PR) | unlocks TPOT, full SLA-label work, dynamic_calibration | CARA train.jsonl + train_queue_details.jsonl (bounded ingest, 50-100 MiB budget) | n/a (data engine) |

## 6. What NOT to build (yet, with reasons)

- **TPOT forecaster** — wait for the CARA train ingest; the test_flat
  moderate sample triggers `INSUFFICIENT_SAMPLE_P99` in too many
  subgroups for safe deployment.
- **SLA / timeout ML forecaster** — the deterministic Erlang-C tail is
  already a strong baseline; without measured SLA labels the ML version
  can't beat it under hold-out.
- **Autoscaling ML forecaster** — arrival forecasting alone is
  insufficient without measured replica-count events; building it
  before the data lands is premature.
- **Cold-start ML forecaster** — no cold-start labels in any HF
  dataset; build after pilot data.
- **Thermal pressure forecaster** — no DCGM telemetry available;
  cannot evaluate.
- **A new GPU-packing forecaster** — current parity with strongest
  realistic baseline means alpha is small.

## 7. Open questions (carried to next audit)

1. Should TTFT + queue_wait + e2e latency forecasters share a feature
   pipeline (joint training) or remain independent?
2. What is the binding telemetry-coverage threshold before a forecaster
   is allowed to override the deterministic Erlang-C baseline?
3. How do we adversarially audit a forecaster's safety: hold-out by
   GPU type vs by time vs by model?
4. Should the `dynamic_safe_frontier_estimator_v1` (deterministic
   Erlang-C) remain as the safety floor under every ML forecaster?

## 8. Non-goals

- Training any ML model in this PR.
- Changing scheduler behaviour.
- Modifying the robust energy engine.
- Promoting any forecaster to production.
- Quoting any external savings number (per the `docs/RESULTS.md` §8
  production-claim gate).
