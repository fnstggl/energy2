# Alibaba GenAI 2026 Backtest — CANONICAL_TRACE_BACKTEST_ALIBABA_GENAI_2026_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` and `docs/PUBLIC_TRACE_BACKTESTS.md` first.

## Provenance
- **Source:** `raw:data/external/alibaba_genai/raw`
- **Dataset:** Alibaba `cluster-trace-v2026-GenAI` (GenTD26), a top-down stable-diffusion serving trace. Public dataset, **not customer telemetry**.

## Schema report — files discovered / used / skipped

- **Layers present:** ['application', 'infrastructure', 'middleware', 'scheduler']
- **Primary telemetry files used:** 13
- **Empty files:** none
- **Skipped (non-telemetry / derived):** none

| file | classification | layer | status |
|---|---|---|---|
| lora_request_trace.csv | primary | application | present |
| qps.csv | primary | middleware | present |
| queue_size_raw_anon.csv | primary | middleware | present |
| queue_rt_raw_anon.csv | primary | middleware | present |
| pipeline_inference_data_anon.csv | primary | scheduler | present |
| pipeline_update_latency_anon.csv | primary | scheduler | present |
| model_predict_data_anon.csv | primary | scheduler | present |
| basemodel_update_latency_anon.csv | primary | scheduler | present |
| controlnet_latency_data_anon.csv | primary | scheduler | present |
| lora_update_latency_anon.csv | primary | scheduler | present |
| pod_gpu_duty_cycle_anon.csv | primary | infrastructure | present |
| pod_gpu_memory_used_bytes_anon.csv | primary | infrastructure | present |
| pod_memory_util_anon.csv | primary | infrastructure | present |
| data_trace_processed.csv | derived | mixed | missing |
| README.md | documentation | n/a | missing |
| MLoRA-Pipeline.png | documentation | n/a | missing |
| lora_request_processing.ipynb | documentation | n/a | missing |

## Cross-layer linkage matrix (computed from data — no faked joins)

Linkage quality ∈ {`exact_join`, `container_join`, `time_join`, `no_join`}. **The application (request) layer is `no_join` to every metric layer**: it uses a different anonymized time base (2024 vs the 2022 metric epoch) and has no `container_ip`. The metric layers join to each other by `container_ip`. **No request→GPU causality is claimed.**

| layer | application | middleware | scheduler | infrastructure |
|---|---|---|---|---|
| **application** | self | no_join | no_join | no_join |
| **middleware** | no_join | self | container_join | container_join |
| **scheduler** | no_join | container_join | self | container_join |
| **infrastructure** | no_join | container_join | container_join | self |

Consequence for the backtest: the request replay is built from the **application layer only**; the pipeline cold-start latencies are used as a **distribution calibration** (medians), not a per-request join; the middleware/infra layers are summarised + container-joined for calibration.

## Trace summary by layer

- **application:** 26,392 requests, 79 models, lora_frac 0.1652; e2e_latency_s p50/p95/p99 23.0/70.0/106.0; types {'IMG_2_IMG': 2152, 'INPAINTING': 180, 'TXT_2_IMG': 24060}
- **middleware:** 49,539 samples; gateway waiting_time_s p95/p99 0.451/0.628; queue_depth p95 0.3333333333333333
- **scheduler/pipeline:** 140,888 events; stage p50 (s) {'pipeline_inference': 15.1, 'pipeline_update': 8.5, 'model_predict': 35.0, 'basemodel_load': 22.7, 'controlnet_load': 3.9, 'lora_load': 4.4}
- **infrastructure:** 533,720 samples; GPU util% p50/p95 0.0/31.733333333333334; container mem frac p95 0.8773185809453329
- **cold-start calibration (s, pipeline medians):** {'basemodel_load': 22.7, 'lora_load': 4.4, 'controlnet_load': 3.9}

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. **goodput_unit = `completed_requests`** (no output-token field exists). Service demand = measured `e2e_latency_s` per request; the model cold-start adder is calibrated from the pipeline layer. Same serving physics (`serving.py`), calibration and cost basis across all policies — only provisioning/routing differs. Headline = **sla_aware** (interactive inference, `docs/RESULTS.md` §3 rule 5).

| policy | goodput/$ | SLA-compliant req | completed | infra $ | GPU-hrs | e2e p95 (s) | e2e p99 (s) | timeout % | mean cold-start (s) | affinity |
|---|---|---|---|---|---|---|---|---|---|---|
| fifo | 1.77 | 26,392 | 26,392 | 14,931 | 4,977 | 53 | 53 | 0.00 | 23.6 | no |
| sla_aware *(headline)* | 5.19 | 17,794 | 26,392 | 3,426 | 1,142 | 614 | 1,219 | 6.34 | 23.6 | no |
| queue_aware | 5.25 | 15,815 | 26,392 | 3,015 | 1,005 | 793 | 1,597 | 8.90 | 23.6 | no |
| utilization_aware | 6.83 | 18,045 | 26,392 | 2,643 | 881 | 239 | 406 | 8.41 | 23.6 | no |
| constraint_aware **(CA)** | 9.84 | 26,392 | 26,392 | 2,682 | 894 | 45 | 53 | 0.00 | 2.9 | yes |

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)
- **Outcome:** `ALPHA_WIN` · margin vs `sla_aware`: **+89.46%** on goodput/$
- **Safety evidence:** e2e_p99<=0.5x_queue_aware, e2e_p99<=0.5x_utilization_aware

## Aurelius-specific findings

1. **Proxy/gateway awareness:** marginal here — gateway waiting time is ~0.451s p95 (tiny vs the 23s base-model cold-start). The gateway is **not** the bottleneck.
2. **Queue-aware / prewarm / reserve:** **helps decisively.** `constraint_aware` prewarm + model-affinity cuts mean cold-start to 2.9s (vs 23.6s for the baselines), the dominant latency term.
3. **Scheduler/pipeline awareness:** **most impactful addressable lever** — pipeline cold-start (basemodel/LoRA/ControlNet load) is the largest p99 term an optimizer can act on (intrinsic request-size variance is larger but not schedulable); affinity routing that respects warm pools is the key.
4. **GPU utilization / memory pressure:** GPUs are mostly idle (util p50 0.0%, p95 31.733333333333334%); `utilization_aware` scales replicas down (cheapest GPU-hours) but pays in SLA without affinity. Memory frac p95 0.8773185809453329 bounds how many models can stay warm per container.
5. **constraint_aware vs sla_aware/queue_aware:** beats the headline (`+89.5%`); it also dominates queue_aware/utilization_aware on SLA-safe goodput here.
6. **Economic alpha or only safety?** **Both:** lower infra $ (2,682 vs 3,426) AND lower e2e p99 (53s vs 1,219s).
7. **Losses / limitations:** the application↔infra layers are `no_join` (incompatible time bases, no shared key), so queue_aware/utilization_aware use the **simulated** queue/util, not the real telemetry (which cannot be aligned per-request). The cold-start model is a pipeline-layer **calibration**, not a measured per-request join — a simulator limitation, stated honestly.
8. **Which layer is most predictive of p99?** Largest single term is **request_exec_variance_s** (contributions (s): {'scheduler_pipeline_cold_start_s': 22.72, 'gateway_queue_wait_s': 0.628, 'request_exec_variance_s': 83.0}). The biggest term — intrinsic request execution-time variance — is **not addressable** by orchestration. Among the **addressable** layers the dominant one is **scheduler_pipeline_cold_start_s** (scheduler/pipeline cold-start ≫ gateway queue), which is exactly the lever `constraint_aware` pulls via affinity/prewarm.

## Honest limits
- Request-level serving replay over proxy physics; metric layers used for calibration only (no per-request request→GPU join exists). GPU price + cold-start magnitudes are documented priors / measured medians, identical across policies. The baselines load-balance **without** model-affinity; `constraint_aware`'s win is specifically the affinity/prewarm lever — a real gap, honestly the point of the dataset.
- **Not production-real savings.** Directional simulator result only.

