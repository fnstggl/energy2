# Hugging Face Dataset Registry — Federated Benchmark Corpus

> **Discovery / data-engine PR.** This document is the authoritative
> registry for every Hugging Face dataset Aurelius has evaluated. Read it
> before re-discovering a dataset — discovery volume is **not** success;
> improved decision quality is.
>
> **Read first:**
> - `docs/RESULTS.md` (canonical KPI + claim rules)
> - `docs/PUBLIC_TRACE_BACKTESTS.md` (dataset roles + ingester contract)
> - `docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md`
> - `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`
> - `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`
> - `docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md`
>
> Simulator / benchmark results are **directional only**, never
> production savings. Pilot telemetry remains the only Tier 1 calibration
> source. See §1 (trust hierarchy) and §6 (prohibited uses).

## 1. Trust hierarchy (binding)

Datasets are NOT equally valuable. The Aurelius objective function
optimizes goodput per dollar, SLA compliance, latency / queue / timeout
risk, placement, routing, residency, and energy / carbon cost under
operational constraints. Discovery prioritises information that improves
constraint-aware decision quality. Lower trust = lower priority.

| Tier | Class | Examples | Aurelius uses |
|---|---|---|---|
| 1 | Real pilot telemetry | production vLLM / Triton / Ray Serve / K8s / Prometheus / DCGM | dynamic frontier calibration, scheduler calibration, production validation |
| 2 | Public telemetry traces | public Prometheus / DCGM / serving exports | telemetry + risk calibration |
| 3 | Cluster scheduler traces | Alibaba GPU, MIT Supercloud, Philly, Azure Functions | scheduling / placement / queue modelling / resource allocation |
| 4 | Latency benchmark traces | AgentPerfBench, Odyn, prefixbench | latency / throughput / concurrency priors, kernel cost priors, cache hit priors |
| 5 | Request shape traces | LMSYS, ShareGPT | workload shape / prompt distributions / replay |
| 6 | Synthetic benchmark data | — | lowest priority |

**Tier 1 calibration sources are not produced by this PR.** Every HF
dataset on this page sits at Tier 2-6 at best.

## 2. Canonical trace types

The federated corpus keeps datasets separate and typed. Each dataset
maps onto exactly one of:

- `request_shape_trace` — turn / prompt / output shape; e.g. ShareGPT,
  LMSYS conversations.
- `latency_benchmark_trace` — measured TTFT / TPOT / ITL / e2e latency,
  throughput, concurrency, model / GPU / engine; e.g. AgentPerfBench
  trace_replay / synthetic_distributional / mse_validation.
- `kernel_profile_trace` — GEMM / Nsight / NCU per-kernel profiles;
  e.g. AgentPerfBench kernels_labeled / per_layer_kernel.
- `cluster_scheduler_trace` — jobs + queues + resources; e.g. Alibaba
  GPU, MIT Supercloud, Philly, Azure Functions.
- `cache_residency_trace` — prefix-cache / cache-hit / cold-start /
  residency; e.g. prefixbench.
- `telemetry_trace` — real serving / scheduler telemetry (Tier 2). The
  highest-trust HF class.
- `tool_runtime_trace` — real measured MCP / agent-runtime tool-call
  execution telemetry (operation_id + tool_name + duration_ms + status +
  error_type + timestamps), one row per tool call. Job-trace shape (akin
  to `cluster_scheduler_trace`), but the "jobs" are tool calls inside an
  agent runtime, not GPU jobs. e.g. Lightcap/agent-runtime-telemetry-small.
- `mixed_or_unknown_trace` — uncertain; cannot be promoted until
  manually classified.

## 3. Federated corpus design

A federated corpus means **datasets stay separate** — there is no merged
super-dataset. Cross-dataset queries must select compatible trace types
+ signals, never silently merge incompatible records. Every committed
record carries:

- `source_dataset_id`
- `trace_type`
- `provenance` (free-form label, e.g.
  `agent-perf-bench/AgentPerfBench@trace_replay#summary_v1`)
- `field_quality` — per-field map onto `{real, derived, proxy, synthetic,
  missing, unknown}`. Reports must label proxy / synthetic / derived
  values; the evaluation harness refuses to use proxies as if they were
  measured.
- `limitations` — explicit list of what the source does NOT measure.

See `aurelius/traces/hf_corpus/schemas.py` for the canonical record
shapes (`BenchmarkLatencyRecord`, `RequestShapeRecord`,
`KernelProfileRecord`, `ClusterSchedulerRecord`, `CacheResidencyRecord`,
`TelemetryRecord`).

## 4. Pipeline

```
scripts/discover_hf_aurelius_datasets.py
  -> data/external/hf_discovery/hf_dataset_candidates.json

scripts/ingest_hf_aurelius_dataset.py --dataset-id ... --from-hf-file ...
  -> data/external/hf/<safe>/<config>/processed/sample.jsonl
  -> data/external/hf/<safe>/<config>/processed/summary.json
  -> data/external/hf_discovery/canonical_corpus_registry.json (with promotion gates)
  -> tests/fixtures/hf/<safe>__<config>_sample.jsonl (5-row deterministic fixture)

scripts/ingest_hf_acmetrace.py
  -> Qinghao/AcmeTrace 4 configs (kalos_jobs, seren_jobs_head,
     kalos_gpu_util_head, seren_ipmi_gpu_power_head).

scripts/ingest_hf_gap_datasets.py
  -> 5 telemetry-gap datasets (cc-traces, lmcache-agentic-traces,
     BurstGPT, google-cluster-data-2019, prefixbench).

scripts/ingest_hf_latency_benchmarks.py
  -> 3 broadened-discovery latency benchmarks (odyn-network/odyn-benchmarks,
     memoriant/dgx-spark-kv-cache-benchmark,
     intellistream/vllm-hust-benchmark-results).

scripts/ingest_hf_optimum_benchmark.py
  -> 9 optimum-benchmark/llm-perf-leaderboard configs covering A100 / A10 /
     T4 / 32vCPU-C7i × pytorch-cuda / pytorch-cpu × unquantized / awq / bnb
     / gptq / torchao. Real measured prefill (TTFT) + decode (TPOT) latency
     at p50/p90/p95/p99, per-request GPU/CPU/RAM energy (kWh) and peak
     VRAM/RAM memory.

scripts/ingest_hf_llm_energy_consumption.py
  -> 4 ejhusom/llm-inference-energy-consumption configs (Round 4) covering
     cross-hardware-tier (laptop2 vs workstation) × workload (alpaca vs
     codefeedback) × model-size (gemma:7b, codellama:7b, codellama:70b).
     Real per-request Ollama timing (total / load / prompt / response
     duration ns) + per-request CodeCarbon energy (kWh split CPU / GPU /
     total). First Ollama-engine entry + first consumer/laptop-tier prior
     in the federated corpus. License: cc-by-sa-4.0.

scripts/ingest_hf_metrum_llmperfdata.py
  -> 1 metrum-ai/llm-perfdata config (Round 5; multi_source_curated_v1)
     covering 80 rows × 24 models × 9 GPUs × 5 engines × 3 precisions —
     a multi-source curated TTFT / TPOT / throughput ledger that fills the
     corpus' previously-empty H100 / H200 / B200 / L40S / RTX-4090 /
     AMD MI300X / MI355X / Intel Gaudi 3 cells and the SGLang / vLLM-ROCm
     engine cells. Statistical strength = weak (densest cell n=8) →
     promoted_for_training_priors only. License: MIT.

scripts/ingest_hf_lightcap_runtime_telemetry.py
  -> Lightcap/agent-runtime-telemetry-small 4 configs covering the
     inaugural tool_runtime_trace canonical type: operations
     (2,262 rows / per-call execution telemetry, moderate strength,
     promoted_for_backtest), tool_summary (32 aggregated rows,
     fixture_only, promoted_for_schema_only), operation_events
     (9,903 lifecycle events × 2,262 operations, moderate strength,
     promoted_for_backtest; per-event duration_ms = ms-since-started
     exposes dispatch / execution / completion-stage priors directly),
     audit_records (14,053 MCP audit rows × 7,013 request_ids, strong
     strength, promoted_for_backtest; MCP-shell-layer duration_ms is
     REAL on tool_results rows with p99=2.5 s, max=900 s). cc-by-4.0.

scripts/run_hf_corpus_evaluations.py
  -> data/external/hf_discovery/hf_corpus_evaluation_summary.json
```

Discovery never downloads data — only metadata via the public HF API
(`https://huggingface.co/api/datasets`). HF_TOKEN is honoured when set;
gated datasets without access are marked `gated_blocked` and skipped.

Ingestion is **bounded** by `--max-rows` and `--max-bytes`. Raw downloads
live under `data/external/hf/<safe>/raw/` and are gitignored. Only the
small normalised sample + summary JSON are committed.

## 5. Promotion gates

A dataset is promotable into the canonical corpus only when **every**
gate passes (see `aurelius/traces/hf_corpus/promotion.py`):

1. `schema_test` — non-empty raw + normalized schemas; no unknown
   columns.
2. `fixture_test` — at least one row committed and a sample sha256.
3. `bounded_size_guard` — sample ≤ 16 MiB.
4. `license_and_gating_recorded` — license + gated status present.
5. `canonical_trace_type_assigned` — not `mixed_or_unknown_trace`.
6. `signals_explicit` — available + missing signals lists present.
7. `limitations_recorded` — non-empty limitations list.
8. `at_least_one_aurelius_use_case` — trace_type maps to ≥ 1 promotion
   tag.

Gated short-circuit: `gated=True` → `gated_blocked` regardless of any
other gate.

Promotion states (see `PROMOTION_STATES`):

- `candidate`, `validated_bounded`, `rejected`, `gated_blocked`
- `promoted_for_backtest`, `promoted_for_training_priors`,
  `promoted_for_constraint_aware_evaluation`,
  `promoted_for_dynamic_calibration`,
  `promoted_for_performance_priors`,
  `promoted_for_cache_residency_evaluation`

Promotion to `promoted_for_training_priors` does **not** mean production
truth. Tier 1 pilot telemetry remains the binding production calibration
source.

## 6. Evaluation harness (Phase C)

`scripts/run_hf_corpus_evaluations.py` routes each promoted dataset to
the bounded smoke evaluator that matches its canonical trace type and
signals:

| trace_type | evaluator_id | primary_baseline | informs |
|---|---|---|---|
| `latency_benchmark_trace` | `latency_benchmark_prior_smoke_v1` | `sla_aware_serving_frontier_static` | performance_priors, constraint_aware_engine |
| `kernel_profile_trace` | `kernel_profile_prior_smoke_v1` | `static_kernel_cost_prior` | performance_priors |
| `cluster_scheduler_trace` | `cluster_scheduler_prior_smoke_v1` | `sla_aware_packing` | constraint_aware_engine, training_frontier |
| `cache_residency_trace` | `cache_residency_prior_smoke_v1` | `residency_aware_routing` | cache_residency_evaluation, constraint_aware_engine |
| `telemetry_trace` | `telemetry_calibration_smoke_v1` | `dynamic_safe_frontier_estimator_v1` | dynamic_frontier, constraint_aware_engine |
| `request_shape_trace` | `request_shape_prior_smoke_v1` | `diurnal_arrival_replay_prior` | workload_modelling |

Rules (binding):

- Skip incompatible datasets with explicit reasons (no required signals
  present → skip).
- Bounded evaluations only — no full backtests, no controller execution.
- **No aggregation across incompatible trace types.** Aggregation is
  valid only within the same trace_type + evaluator + KPI.
- **No oracle as headline.** Oracle baselines are analysis-only.
- **Never treat benchmark data as production telemetry.**
- Every result carries a `result_quality ∈ {prior_only, derived, proxy,
  synthetic}` label so reports cannot accidentally claim a measured
  result.

## 7. Current registry — datasets evaluated

> Updated by re-running `scripts/discover_hf_aurelius_datasets.py` +
> `scripts/ingest_hf_aurelius_dataset.py`. Datasets that have been
> evaluated should NOT be re-discovered — the candidate JSON +
> canonical registry are the authoritative memory.

### 7.1 Datasets ingested + promoted

| dataset_id | config | trace_type | trust tier | promotion state | fixture rows | analysis rows | sample strength | ingestion date |
|---|---|---|---|---|---|---|---|---|
| `agent-perf-bench/AgentPerfBench` | `trace_replay` | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 100 | n/a | n/a | 2026-05-31 |
| `agent-perf-bench/AgentPerfBench` | `kernels_labeled` | `kernel_profile_trace` | Tier 4 | `promoted_for_performance_priors` (+ `promoted_for_training_priors`) | 100 | n/a | n/a | 2026-05-31 |
| `asdwb/cara_latency_prediction` | `test_flat` | `telemetry_trace` | **Tier 2** | `promoted_for_constraint_aware_evaluation` (+ `promoted_for_backtest`; `dynamic_calibration` downgraded — needs `strong` strength) | 5 | 9,605 | moderate | 2026-05-31 |
| `asdwb/cara_latency_prediction` | `test_queue_details` | `telemetry_trace` | **Tier 2** | `promoted_for_constraint_aware_evaluation` (+ `promoted_for_backtest`; `dynamic_calibration` downgraded) | 5 | 4,876 | moderate | 2026-05-31 |
| `asdwb/cara_latency_prediction` | **`train_flat`** (analysis-tier) | `telemetry_trace` | **Tier 2** | **`promoted_for_dynamic_calibration`** (+ `constraint_aware_evaluation`, `backtest`) | 5 | **76,825** | strong | 2026-05-31 |
| `asdwb/cara_latency_prediction` | **`train_queue_details`** (analysis-tier) | `telemetry_trace` | **Tier 2** | **`promoted_for_dynamic_calibration`** (+ `constraint_aware_evaluation`, `backtest`) | 5 | 38,509 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | `trace` | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | 25,409 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`trace_analysis`** | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | **202,215** | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | `qwen3_32b_buckets` | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `promoted_for_training_priors`) | 5 | 19,130 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`qwen3_32b_buckets_analysis`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | **103,507** | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | `qwen3_32b_bucket_reuse` | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `promoted_for_training_priors`) | 5 | 16,593 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`qwen3_32b_bucket_reuse_analysis`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | **147,440** | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`apertus_70b_bucket_reuse`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | 49,434 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`qwen380b_instruct_bucket_reuse`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | 45,887 | strong | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`qwen380b_thinking_bucket_reuse`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | 7,399 | moderate | 2026-05-31 |
| `eth-easl/swissai-serving-trace` | **`llama3_70b_bucket_reuse`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `training_priors`) | 5 | **153,275** | strong | 2026-05-31 |
| `semianalysisai/cc-traces-weka-no-subagents-051226` | **`traces_head`** | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | 761 | weak | 2026-06-01 |
| `sammshen/lmcache-agentic-traces` | **`train_shard4`** | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | 4,976 | moderate | 2026-06-01 |
| `lzzmm/BurstGPT` | **`burstgpt_1_full`** | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | 59,999 | strong | 2026-06-01 |
| `lsliwko/google-cluster-data-2019-sorted-by-timestamp` | **`instance_events_shard0`** | `cluster_scheduler_trace` | **Tier 3** | `promoted_for_backtest` (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | 60,000 | strong | 2026-06-01 |
| `jaytonde05/prefixbench` | **`prefixbench_all`** | `cache_residency_trace` | Tier 4 | `promoted_for_cache_residency_evaluation` (+ `promoted_for_training_priors`) | 5 | 4,000 | moderate | 2026-06-01 |
| `Qinghao/AcmeTrace` | **`kalos_jobs`** | `cluster_scheduler_trace` | **Tier 3** | `promoted_for_backtest` (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **62,413** | strong | 2026-06-01 |
| `Qinghao/AcmeTrace` | **`seren_jobs_head`** | `cluster_scheduler_trace` | **Tier 3** | `promoted_for_backtest` (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **79,999** | strong | 2026-06-01 |
| `Qinghao/AcmeTrace` | **`kalos_gpu_util_head`** | `telemetry_trace` | **Tier 2** | `promoted_for_constraint_aware_evaluation` (+ `promoted_for_backtest`; `dynamic_calibration` downgraded — needs `strong` strength) | 5 | 6,680 | moderate | 2026-06-01 |
| `Qinghao/AcmeTrace` | **`seren_ipmi_gpu_power_head`** | `telemetry_trace` | **Tier 2** | **`promoted_for_dynamic_calibration`** (+ `constraint_aware_evaluation`, `backtest`) | 5 | **79,999** | strong | 2026-06-01 |
| `odyn-network/odyn-benchmarks` | **`qwen_chat_streaming`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 64 | moderate | 2026-06-01 |
| `odyn-network/odyn-benchmarks` | **`facebook_chat_streaming`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 48 | moderate | 2026-06-01 |
| `odyn-network/odyn-benchmarks` | **`qwen_batch`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 28 | moderate | 2026-06-01 |
| `odyn-network/odyn-benchmarks` | **`facebook_batch`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 4 | 4 | weak | 2026-06-01 |
| `memoriant/dgx-spark-kv-cache-benchmark` | **`v3_corrected`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 5 | 18 | weak | 2026-06-01 |
| `intellistream/vllm-hust-benchmark-results` | **`single_gpu`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 42 | moderate | 2026-06-01 |
| `intellistream/vllm-hust-benchmark-results` | **`multi_gpu`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 3 | 3 | weak | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_unquantized_1xA100`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 190 | moderate | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_unquantized_1xA10`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **1,344** | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_unquantized_1xT4`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **1,265** | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_bnb_1xA100`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 401 | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_gptq_1xA100`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 314 | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_awq_1xA10`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **1,569** | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_bnb_1xT4`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | 775 | strong | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cuda_torchao_1xA10`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 5 | 15 | weak | 2026-06-01 |
| `optimum-benchmark/llm-perf-leaderboard` | **`pytorch_cpu_unquantized_32vCPU_C7i`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **1,128** | strong | 2026-06-01 |
| `Exgentic/agent-llm-traces` | **`swebench_claude_code_shard12`** | `request_shape_trace` | Tier 5 | `promoted_for_training_priors` | 5 | **2,294** | moderate | 2026-06-01 |
| `ssong1/llmperf-bedrock` | **`bedrock_claude_instant_v1`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **350** | moderate | 2026-06-01 |
| `ejhusom/llm-inference-energy-consumption` | **`alpaca_gemma_7b_laptop2`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **5,099** | strong | 2026-06-01 |
| `ejhusom/llm-inference-energy-consumption` | **`alpaca_gemma_7b_workstation`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **8,735** | strong | 2026-06-01 |
| `ejhusom/llm-inference-energy-consumption` | **`codefeedback_codellama_7b_workstation`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `constraint_aware_evaluation`, `training_priors`) | 5 | **3,109** | moderate | 2026-06-01 |
| `ejhusom/llm-inference-energy-consumption` | **`codefeedback_codellama_70b_workstation`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 5 | 161 | weak | 2026-06-01 |
| `metrum-ai/llm-perfdata` | **`multi_source_curated_v1`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_training_priors` | 5 | 80 | weak | 2026-06-01 |
| `Lightcap/agent-runtime-telemetry-small` | **`operations`** | **`tool_runtime_trace`** | **Tier 3** | **`promoted_for_backtest`** (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **2,262** | moderate | 2026-06-01 |
| `Lightcap/agent-runtime-telemetry-small` | **`tool_summary`** | **`tool_runtime_trace`** | **Tier 3** | `promoted_for_schema_only` | 5 | 32 | fixture_only | 2026-06-01 |
| `Lightcap/agent-runtime-telemetry-small` | **`operation_events`** | **`tool_runtime_trace`** | **Tier 3** | **`promoted_for_backtest`** (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **9,903** | moderate | 2026-06-01 |
| `Lightcap/agent-runtime-telemetry-small` | **`audit_records`** | **`tool_runtime_trace`** | **Tier 3** | **`promoted_for_backtest`** (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **14,053** | strong | 2026-06-01 |
| `ssakethch/h200-quantization-benchmarks` | **`throughput`** | `latency_benchmark_trace` | Tier 4 | `promoted_for_performance_priors` (+ `promoted_for_constraint_aware_evaluation`, `promoted_for_training_priors`) | 5 | **275** | strong | 2026-06-01 |

> **CARA** is the first Tier 2 (public telemetry trace) entry in the
> federated corpus. CARA **train_flat** + **train_queue_details** are
> the first Tier 2 entries promoted to `promoted_for_dynamic_calibration`
> after the analysis-tier expansion (76,825 + 38,509 strong-strength
> rows). See `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` §2.1-§2.6 for
> the analysis-tier expansion, signal coverage table, forecast
> readiness table, forecast leverage quantification, missing-telemetry
> gap analysis, and strongest-forecasting-dataset matrix.

> **Telemetry-gap ingest 2026-06-01.** Five datasets from the
> `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md` top-10 ingest-now list have
> now been bounded-ingested:
>
> - `semianalysisai/cc-traces-weka-no-subagents-051226` — Real Claude
>   Code production agentic traces with per-request KV block hashes
>   (`kv_block_hashes` + `migration_or_cache_loss_proxy` signals;
>   weak strength = 7 sessions / 761 requests from the 80 MiB head).
> - `sammshen/lmcache-agentic-traces` — 787 multi-turn agentic sessions
>   with `pre_gap` (think-time) + `session_id` for routing/cache
>   forecasting (moderate strength = 4,976 rows from one parquet shard).
> - `lzzmm/BurstGPT` — Real Microsoft Azure ChatGPT/GPT-4 arrival trace
>   (strong strength = 59,999 rows from the bounded 20 MiB head).
> - `lsliwko/google-cluster-data-2019-sorted-by-timestamp` — Google
>   Borg 2019 instance lifecycle events
>   (`autoscaling_proxy` + `migration_or_cache_loss_proxy` +
>   `model_load_event` + `model_unload_event` proxies; strong strength
>   = 60,000 rows from one ~53 MB gzipped shard). **Cluster_scheduler_trace
>   Tier 3** — the first Tier-3 cluster trace ingested via the HF pipeline.
> - `jaytonde05/prefixbench` — Synthetic prefix-cache benchmark prompts
>   (moderate strength = 4,000 rows; full 80 MB corpus across 4 jsonl
>   files).
>
> Ingest summary: `data/external/hf_discovery/telemetry_gap_ingest_summary.json`.
> Ingest script: `scripts/ingest_hf_gap_datasets.py`. Registry update
> script: `scripts/register_hf_gap_datasets.py`. Tests:
> `tests/test_hf_gap_ingest.py` (35 tests, all green).

> **Normalized-sample commit follow-up 2026-06-01 (b).** PR #129
> deliberately committed only summaries + 5-row fixtures. A follow-up
> commits a bounded **normalized** analysis sample per dataset for the
> four datasets with verified permissive licenses (Apache-2.0 / MIT /
> CC-BY-4.0): BurstGPT 8.0 MB / Google Cluster 21.2 MB / LMCache 0.7 MB
> / CC-traces 0.4 MB → 30.4 MB total (cap 150 MB). `jaytonde05/prefixbench`
> is SKIPPED — its HF card has no `license:` field, so the policy
> requires non-redistribution. Each committed sample lives at
> `data/external/hf/<safe>/<config>/processed/normalized_sample.jsonl`
> (a distinct filename so the existing gitignore rule for
> `analysis_sample.jsonl` is untouched). Per-dataset `summary.json`
> records `committed_normalized_sample_{path,bytes,rows,sha256}` +
> `license_redistribution_status` + `raw_committed=false`. Rollup at
> `data/external/hf_discovery/telemetry_gap_normalized_sample_commit_summary.json`;
> tests at `tests/test_hf_gap_normalized_samples.py` (29 tests, all green).

> **AcmeTrace focused audit 2026-06-01.** Four short-term-mission datasets
> from §10 (`Qinghao/AcmeTrace`, `HuggingAGree/AcmeTrace`,
> `osteele/llm-calibration-db`, `jaytonde05/iris-prefix-cache-benchmark`)
> were focus-audited. Outcome:
>
> - **`Qinghao/AcmeTrace`** — bounded-ingested into four configs
>   (`kalos_jobs` full, `seren_jobs_head` 32 MiB cap, `kalos_gpu_util_head`
>   32 MiB cap, `seren_ipmi_gpu_power_head` 16 MiB cap). Real Shanghai AI
>   Lab Kalos + Seren cluster scheduler trace (NSDI'24
>   *Characterization of LLM Development in the Datacenter*). Job-level
>   trace carries **real** `queue_wait` (derived per README) and **real**
>   `state ∈ {COMPLETED, CANCELLED, FAILED, TIMEOUT, NODE_FAIL}` failure /
>   timeout labels — the first HF Tier-3 cluster_scheduler_trace promoted
>   to `promoted_for_backtest` via this pipeline. DCGM-collected per-host
>   GPU utilisation (15-second sampling) gives a Tier-2 telemetry signal;
>   IPMI per-host GPU power telemetry (79,999 rows strong-strength) is
>   the first non-CARA HF dataset promoted to
>   **`promoted_for_dynamic_calibration`**. License: CC-BY-4.0.
> - **`HuggingAGree/AcmeTrace`** — re-upload of (1) with the same 75
>   files. Marked `duplicate_existing` (discovery-only; no separate
>   ingest tree).
> - **`osteele/llm-calibration-db`** — HF `gated:manual` (requires
>   manual approval from the dataset owner). Marked `gated_blocked`;
>   would qualify as Tier-4 latency_benchmark_trace + Tier-2 telemetry
>   candidate (calibration_runs / layer_timing / memory_calibration /
>   telemetry_samples / system_load_snapshots / inference_overhead per
>   the HF card) once access is granted. Revisit when authorised.
> - **`jaytonde05/iris-prefix-cache-benchmark`** — 20 synthetic prompts
>   only (57 KB total), single `prompt: string` column. No measured
>   TTFT, cache-hit, GPU, queue, or SLA signals. Marked
>   `reject_low_value` — the existing `jaytonde05/prefixbench` config
>   already covers the synthetic prefix-cache role with a richer schema.
>
> Audit script: `scripts/ingest_hf_acmetrace.py`. Registry update
> script: `scripts/register_hf_acmetrace.py`. Candidates-update script:
> `scripts/update_hf_candidates_acmetrace.py`. Rollup at
> `data/external/hf_discovery/acmetrace_audit_summary.json`. Tests:
> `tests/test_hf_acmetrace_ingest.py` (46 tests, all green).

> **Broadened-discovery latency-benchmark ingest 2026-06-01.** Eleven
> candidates from the INGEST_LATER / MONITOR groups of
> `data/external/hf_discovery/aurelius_gap_closure_audit.json` were
> follow-on-audited. Three were bounded-ingested as Tier-4
> `latency_benchmark_trace`; eight were classified discovery-only.
>
> - **`odyn-network/odyn-benchmarks`** — vLLM + Ray Serve benchmark with
>   measured TTFT_avg/p95, TPOT_avg/p95, e2e_avg/p95, throughput_tok_s,
>   throughput_req_s, and failure counts across 4 prompt profiles
>   (A short→long, B long→short, C long→long, D short→short) × 2 models
>   (Qwen2.5-7B on DGX Spark Blackwell, OPT-125m on RTX 3090) × 6-8
>   concurrency levels. Apache-2.0. Split into 4 configs
>   (`qwen_chat_streaming`, `facebook_chat_streaming`, `qwen_batch`,
>   `facebook_batch`).
> - **`memoriant/dgx-spark-kv-cache-benchmark`** — corrected v3 KV-cache
>   quantization benchmark (llama.cpp f16 / q8_0 / q4_0 on DGX Spark
>   GB10 Grace Blackwell unified memory). 18 rows, real
>   `kv_buffer_mib` + `gpu_mem_mib` + `prompt_tps` + `gen_tps` per
>   `(cache_type, context_tokens)` cell from 0 to 110K tokens.
>   Apache-2.0.
> - **`intellistream/vllm-hust-benchmark-results`** — vLLM-HUST
>   submissions-driven leaderboard with measured TTFT_ms, TBT_ms
>   (=TPOT), throughput_tps, peak_mem_mb, error_rate across Huawei
>   910B3 hardware × multiple Qwen / DeepSeek models × workloads
>   (prefix-repetition-online, sonnet-throughput). No declared license
>   → `license=None`, no committed normalised sample under the
>   conservative redistribution policy.
>
> Eight rejection / deferral records (audited but not ingested):
> `tarekmasryo/llm-system-ops-production-telemetry-sft-data`
> (self-declared SYNTHETIC despite the "production-telemetry" name —
> rejected to enforce the anti-dataset-spam rule);
> `spiritbuun/turboquant-tcq-kv-cache` (codebooks, not a dataset);
> `hlarcher/inference-benchmarker` (ShareGPT-derived workload fixtures
> only — duplicate of the existing `sharegpt_aiperf` ingester);
> `Boxoffice1280/Neurips2026_evaluating_accuracy_KV-cache_reuse_techniques`
> (cc-by-nc-nd-4.0 — No-Derivatives clause blocks committing normalised
> samples); `Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days-SVLSGPU_EvalData`
> (duplicate of `lzzmm/BurstGPT`, license=None);
> `MCP-1st-Birthday/smoltrace-cloud-cost-tasks` (synthetic MCP agent-eval
> tasks, no infrastructure signals); `rbgo/llm-inference-benchmark`
> (license=None — deferred); `project-vajra/dev-staging-h100-dgx`
> (license=None — NCCL collective traces deferred).
>
> Audit script: `scripts/ingest_hf_latency_benchmarks.py`. Rollup at
> `data/external/hf_discovery/broadened_discovery_audit_summary.json`.
> Tests: `tests/test_hf_latency_benchmarks_ingest.py` (78 tests, all green).

#### Odyn Network — Tier-4 vLLM + Ray Serve latency benchmark

- **Provenance.** [`odyn-network/odyn-benchmarks`](https://huggingface.co/datasets/odyn-network/odyn-benchmarks)
  — inference benchmark results from the Odyn Network distributed,
  OpenAI-compatible serving platform (Apache-2.0, built on vLLM + Ray
  Serve + FastAPI). 4 prompt profiles A/B/C/D (short/long × short/long
  input/output tokens) × 2 model + hardware combinations
  (Qwen2.5-7B-Instruct on DGX Spark Blackwell at concurrencies 4-250;
  facebook/opt-125m on RTX 3090 at concurrencies 1-32).
- **Available signals (`*_chat_streaming`):** TTFT_avg + TTFT_p95
  (streaming only), TPOT_avg + TPOT_p95, e2e_avg + e2e_p95,
  throughput_tok_s, throughput_req_s, concurrency,
  successful/failed counts (failure_label proxy for SLA backpressure
  at the highest concurrencies), wall_time_s, engine=vllm.
- **Available signals (`*_batch`):** batch_size, num_prompts,
  total_ms, avg_per_prompt_ms (e2e-derived), throughput_prompts_s.
- **Missing signals:** ITL, p50/p90/p99 (only avg + p95 reported),
  KV-cache instrumentation, GPU utilisation telemetry, real
  arrival/queue trace, real timeout label, autoscaling / replica
  signals.
- **Recommended Aurelius uses:**
  - Performance-surface priors (TTFT_avg + TPOT_avg + e2e_avg + p95
    by model × hardware × concurrency × profile).
  - Concurrency-saturation priors — `failed` counts at concurrencies
    ≥ 192 calibrate the failure-rate prior under high backpressure on
    a single replica.
  - Profile-aware request-shape priors — A/B/C/D cover the full
    short/long input/output quadrant for the eval / batch frontier.
- **Prohibited uses:**
  - Real arrival / queue scheduling (benchmark, no arrival trace).
  - Production latency calibration (vLLM benchmark, not pilot).
  - Cross-deployment generalisation — single model × single GPU per
    config; only use within the same `(model, gpu, engine)` tuple.
- **Bounded ingest layout:** all 4 raw CSVs
  (`results/qwen_results/chat_benchmarks.csv`,
  `results/qwen_results/batch_benchmarks.csv`,
  `results/facebook_results/chat_benchmarks.csv`,
  `results/facebook_results/batch_benchmarks.csv`; total ~11 KB raw)
  live under `data/external/hf/odyn-network__odyn-benchmarks/raw/`
  and are **gitignored**. Per-config schema_profile + schema_mapping +
  summary + statistical_rollups + 5-row fixture ARE committed.
  Apache-2.0 license permits redistribution → bounded normalised
  sample committed per config (44 + 31 + 14 + 2 ≈ 91 KiB total,
  100 KiB/file cap; well under the 300 MiB PR budget).

#### Memoriant DGX Spark — Tier-4 KV-cache quantization benchmark

- **Provenance.** [`memoriant/dgx-spark-kv-cache-benchmark`](https://huggingface.co/datasets/memoriant/dgx-spark-kv-cache-benchmark)
  — corrected v3 KV-cache quantization benchmark (Apache-2.0) by
  Nathan Maine / Memoriant Inc. Hardware: NVIDIA DGX Spark (GB10
  Grace Blackwell unified memory architecture, 128 GB unified RAM,
  compute 12.1). Engine: llama.cpp. Configurations: f16 / q8_0 / q4_0
  KV cache quantisation, 6 context-length steps from 0 to 110,019
  tokens.
- **Available signals:** `kv_buffer_mib` (KV cache memory pressure,
  real per `cache_type × context_tokens` cell), `gpu_mem_mib`
  (`nvidia-smi`-measured total GPU memory — replaces the v1 wrong
  RSS-on-unified-memory measurement per the CORRECTION-NOTICE),
  `prompt_tps`, `gen_tps` (real prompt-processing and
  generation-tokens-per-second). engine=llama.cpp, model_family=llama.
- **Missing signals:** TTFT / TPOT / ITL / e2e (only throughput
  reported), concurrency (single request at a time), batch_size,
  failure / timeout labels.
- **Recommended Aurelius uses:**
  - KV-cache memory-pressure priors (216 / 408 / 768 MiB per 110K
    context for q4_0 / q8_0 / f16 — a 72% memory saving with q4_0).
  - Cache-quantization throughput trade-off priors (gen_tps degrades
    from ~45 to ~24 at 110K context under q4_0 — a 37% gen-speed
    hit at long context).
  - GB10 Grace Blackwell unified-memory residency priors.
- **Prohibited uses:**
  - Latency frontier source on its own (no TTFT / TPOT — must be
    combined with a TTFT-aware dataset).
  - Generalisation beyond GB10 (single GPU class).
- **Bounded ingest layout:** 1 raw CSV (846 B) at
  `data/external/hf/memoriant__dgx-spark-kv-cache-benchmark/raw/`
  (gitignored). Schema_profile + schema_mapping + summary +
  statistical_rollups + 5-row fixture + 18-row Apache-2.0 committed
  normalised sample (9 KiB) ARE committed.

#### Intellistream vLLM-HUST — Tier-4 leaderboard (Huawei Ascend)

- **Provenance.** [`intellistream/vllm-hust-benchmark-results`](https://huggingface.co/datasets/intellistream/vllm-hust-benchmark-results)
  — submissions-driven leaderboard for the vLLM-HUST community
  benchmark. Last updated 2026-06-01. Currently dominated by Huawei
  910B3 (Ascend-class, 64 GB/chip) entries, with vLLM 0.11.0 and
  vLLM-HUST 0.20.1rc1.dev314+ as the engines.
- **Available signals:** TTFT_ms (mean), TBT_ms (=TPOT mean),
  throughput_tps, peak_mem_mb, error_rate, concurrent_requests,
  input_length, output_length, batch_size, model (parameters,
  precision, quantization), hardware (vendor, chip_model, chip_count,
  memory), workload (name, dataset), engine + engine_version,
  constraints (scenario_source, scope).
- **Missing signals:** p50 / p90 / p95 / p99 (only scalar means
  reported), ITL, e2e_latency, KV-cache instrumentation, timeout
  label, real arrival / queue trace, autoscaling, replica counts.
- **Recommended Aurelius uses:**
  - Performance-surface priors for Ascend-class hardware (the only
    public leaderboard exposing TTFT + TBT + throughput at this
    granularity for Huawei 910B3).
  - Engine-version comparison priors (vLLM vs vLLM-HUST forks under
    matched hardware + model + workload).
- **Prohibited uses:**
  - Cross-vendor generalisation (Ascend-class only; do NOT apply to
    NVIDIA / AMD / TPU without independent validation).
  - Production latency calibration (Tier 4 benchmark).
  - Memory-pressure analysis when `peak_mem_mb` is zero (a large
    fraction of entries do not report it).
  - Treating `error_rate == 0` as truth — current snapshot reports 0
    for all entries; treat as an upper-bound only.
- **License:** no declared license on the HF card frontmatter →
  `license=None`. The conservative redistribution policy applies — no
  committed normalised sample is shipped for this dataset (raw
  download is gitignored; only schema_profile + schema_mapping +
  summary + statistical_rollups + 5-row fixture are committed).

#### AcmeTrace — Tier-3 cluster jobs + Tier-2 GPU/IPMI telemetry

- **Provenance.** Shanghai AI Lab Acme traces (Kalos + Seren clusters)
  from the NSDI'24 paper "Characterization of Large Language Model
  Development in the Datacenter" — the most important publicly-released
  GPU-cluster trace with both job-level queue/failure data AND
  per-host DCGM+IPMI telemetry.
- **Available signals (`kalos_jobs` / `seren_jobs_head`):** arrivals,
  request_timestamps, queue_state (real queue_wait — derived from
  start_time-submit_time per README §1 note 3), timeout_label
  (FAILED/TIMEOUT/NODE_FAIL states), capacity_proxy (gpu_num + node_num
  + cpu_num), customer_traffic_mix (hashed user), workload_shape, latency
  (derived end_time-start_time).
- **Available signals (`kalos_gpu_util_head` / `seren_ipmi_gpu_power_head`):**
  request_timestamps (15-second DCGM/IPMI sample interval),
  gpu_utilization, dcgm_telemetry (Kalos), ipmi_telemetry +
  power_telemetry (Seren GPU_AB_Power.csv).
- **Missing signals:** TTFT, TPOT, ITL (no per-token timing — this is
  job-level not request-level); cache_reuse / prefix_reuse /
  kv_block_hashes (no KV cache instrumentation in the released trace);
  sla_label, model_load_event, model_unload_event, replica_count,
  cost_or_region.
- **Recommended Aurelius uses:**
  - **Constraint-aware scheduler backtests** — queue-wait and gpu-time
    distributions per workload type for SLA-aware vs FIFO packing
    comparison (cluster_scheduler_trace, Tier 3).
  - **Performance-surface priors** — GPU utilisation distributions per
    host at 15-second resolution feed the static utilisation frontier
    prior (telemetry_trace, Tier 2).
  - **Energy / carbon-aware scheduling priors** — IPMI per-host GPU
    power consumption (W) is a direct input to the energy-cost objective
    in the dynamic frontier estimator (telemetry_trace, Tier 2; first
    non-CARA HF dataset promoted to `promoted_for_dynamic_calibration`).
  - **Cluster failure-mode priors** — termination_state distribution
    (FAILED / TIMEOUT / NODE_FAIL) calibrates the failure_timeout risk
    prior in constraint_aware_engine.
- **Prohibited uses:**
  - LLM serving TTFT/TPOT calibration (no per-token timing; use CARA
    `train_flat`+`train_queue_details` instead).
  - Cache-hit / prefix-cache calibration (not measured here; use
    SwissAI `qwen3_32b_bucket_reuse_analysis` instead).
  - Production-truth SLA calibration (still benchmark/research-class
    — Tier 1 pilot telemetry remains the only production calibration
    source).
- **Bounded ingest layout:** all 4 raw downloads
  (`trace_kalos.csv`, `trace_seren.csv` head 32 MiB,
  `GPU_UTIL.csv` head 32 MiB, `GPU_AB_Power.csv` head 16 MiB) live
  under `data/external/hf/Qinghao__AcmeTrace/raw/` and are
  **gitignored**. Per-config processed summaries + schema profiles +
  schema mappings + statistical rollups + 5-row fixtures ARE committed
  (~141 KB total). Per-config `analysis_sample.jsonl` (66 MB total
  across 4 files) is gitignored — regenerable from the bounded raw
  download via `scripts/ingest_hf_acmetrace.py`.

#### AgentPerfBench / trace_replay

- **Available signals:** ttft, tpot, itl, e2e_latency,
  request_throughput, token_throughput, concurrency, batch_size,
  sequence_length, gpu_type, vllm, sglang.
- **Missing signals:** real queue_wait, queue_depth, timeout, sla,
  failure, gpu_utilization, autoscaling, replica_count, prefix_cache,
  cache_hit.
- **Recommended Aurelius uses:**
  - Performance-surface priors (TTFT / TPOT / e2e at p50 / p90 / p99 by
    model × hardware × concurrency).
  - Throughput / latency risk priors for the static serving frontier.
  - Batch-size + concurrency priors for the eval / batch frontier and
    the constraint-aware engine.
- **Prohibited uses:**
  - Real-arrival scheduling (no arrival timestamps in trace_replay
    summary).
  - Production latency calibration (this is a benchmark, not pilot
    telemetry).
  - Real queue-wait calibration (queue is not measured here).

#### AgentPerfBench / kernels_labeled

- **Available signals:** kernel_duration, gpu_type, batch_size,
  sequence_length, prompt_tokens, output_tokens.
- **Missing signals:** queue_wait, timeout, sla, latency_p99,
  cache_hit, autoscaling, replica_count.
- **Recommended Aurelius uses:**
  - Low-level GPU performance priors (per-kernel duration distribution).
  - Model cost estimation (M × N × K × dtype × duration).
  - Kernel / memory bottleneck priors for the static frontier.
- **Prohibited uses:**
  - Request-level scheduler backtests.
  - SLA / queue calibration.

#### Optimum-benchmark — Tier-4 cross-hardware × quantization × model

- **Provenance.** [`optimum-benchmark/llm-perf-leaderboard`](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard)
  — HuggingFace's official `optimum-benchmark` performance leaderboard
  data. The upstream tool ([huggingface/optimum-benchmark](https://github.com/huggingface/optimum-benchmark))
  is Apache-2.0 and uses `codecarbon` for energy measurement. The HF
  dataset card itself has NO declared license — recorded as `license=None`
  and treated under the conservative
  `license_unspecified_no_redistribution_promise` policy (no committed
  normalised sample; raw downloads gitignored).
- **Configs ingested (9 of 16 available; 1 dropped as failure-only).**
  The leaderboard ships one CSV per (hardware, backend, quantization)
  combo. The 9 configs cover:
  - **NVIDIA A100-SXM4-80GB:** `pytorch_cuda_unquantized_1xA100` (190 rows),
    `pytorch_cuda_bnb_1xA100` (401), `pytorch_cuda_gptq_1xA100` (314).
  - **NVIDIA A10G:** `pytorch_cuda_unquantized_1xA10` (1,344),
    `pytorch_cuda_awq_1xA10` (1,569),
    `pytorch_cuda_torchao_1xA10` (15 — weak).
  - **NVIDIA Tesla T4:** `pytorch_cuda_unquantized_1xT4` (1,265),
    `pytorch_cuda_bnb_1xT4` (775).
  - **32vCPU Sapphire-Rapids (AWS C7i):**
    `pytorch_cpu_unquantized_32vCPU_C7i` (1,128).
  - **Excluded:** `openvino_cpu_unquantized_32vCPU_C7i` — every row is an
    isolated-process crash with zero measured latency columns (recorded
    in the audit summary as `reject_failure_only_no_measurements`).
- **Available signals:** TTFT (= prefill latency mean / p50 / p90 / p95 /
  p99 / count), TPOT (= decode latency mean / p50 / p90 / p95 / p99 /
  count), prefill + decode throughput (tokens/s), per-request energy
  (kWh — CPU + RAM + GPU + total, separately), peak memory (max_global_vram
  / max_allocated / max_reserved / max_ram MB), model, model_family,
  GPU type, backend (pytorch / openvino / onnxruntime), quantization
  scheme (unquantized / awq / bnb / gptq / torchao), dtype, batch_size,
  sequence_length, new_tokens, error_type / error_message.
- **Missing signals:** concurrency (all runs are single-stream with
  batch_size=1), real queue / arrival / dispatch trace, cache hit /
  prefix reuse, model residency / cold start, autoscaling / replica
  count, SLA / timeout label (failures are crashes, not SLA violations),
  routing state, carbon intensity (energy is in kWh; consumers must
  combine with regional CO2 g/kWh to derive carbon).
- **Recommended Aurelius uses:**
  - **Quantization-aware placement priors** — measured latency × memory ×
    energy delta for AWQ / BNB / GPTQ / TorchAO vs unquantized across
    A100 / A10 / T4. The constraint-aware placement engine previously
    had no cross-quantization performance surface — this is the
    strongest public one (≥314 rows per quantization × hardware combo,
    36-93 distinct models per combo).
  - **Energy-aware scheduling priors** — real per-request GPU/CPU/RAM
    energy (kWh) for the energy / carbon cost terms in the Aurelius
    objective function. The first public dataset in the federated
    corpus with measured per-request energy at this granularity.
  - **Cross-hardware throughput priors** — decode + prefill throughput
    (tokens/s) across A100 / A10 / T4 / Sapphire-Rapids vCPU lets the
    routing/residency engine reason about which GPU class a model fits.
  - **OOM / memory-pressure priors** — peak `max_global_vram` and
    `max_allocated` per (model, quantization, hardware) feeds the
    constraint-aware placement engine's memory headroom check; rows
    with null measurements (OOM on the listed GPU) are themselves a
    valuable failure prior.
- **Prohibited uses:**
  - **Real-arrival scheduling.** All runs are single-stream batch_size=1;
    NO arrival process, NO queue, NO concurrent request mix.
  - **Production latency calibration.** Tier 4 benchmark — pilot
    telemetry remains the only Tier 1 calibration source.
  - **Cross-quantization-method generalisation outside the matrix.**
    The CSV is one quantization method × one hardware; rolling-up
    "AWQ vs unquantized" requires explicit matching by (model, hardware).
  - **Single-config p95/p99 claims with < 10 measurements.** The TPOT
    p95/p99 within one row reflects only `report.decode.latency.count`
    iterations (typically 10-100) — use the across-row rollups in
    `statistical_rollups.json` for cross-model p95/p99.
- **Bounded ingest layout:** raw CSVs (~73 MiB total across 9 files)
  live under `data/external/hf/optimum-benchmark__llm-perf-leaderboard/raw/`
  and are gitignored. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`, `statistical_rollups.json`,
  and 5-row fixture (≤ 16 KiB) ARE committed. Per-config
  `analysis_sample.jsonl` (~5 MiB total) is gitignored — regenerable
  from the bounded raw download via
  `scripts/ingest_hf_optimum_benchmark.py`.

> **Exgentic agent-LLM-traces follow-on ingest 2026-06-01 (c).** The
> "next-action" candidate from PR #135 has now been bounded-ingested as
> Tier-5 `request_shape_trace`. This fills the agent-workload per-call
> duration / token-usage gap that the federated corpus previously covered
> only via `sammshen/lmcache-agentic-traces` (single LMCache shard). One
> mid-sized parquet (`train-00012-of-00039`, 41 MB raw, gitignored) gives
> 46 SWE-bench × claude_code agent sessions across two azure-hosted
> models (DeepSeek-V3.2 + Kimi-K2.5), flattened to **2,294 spans /
> request rows** at moderate strength. cdla-permissive-2.0 license permits
> redistribution → a 2.3 MiB normalised sample is committed under the
> 100 MiB-per-file / 300 MiB-per-PR policy.
>
> Ingest script: `scripts/ingest_hf_agent_llm_traces.py`. Registry
> updater: `scripts/register_hf_agent_llm_traces.py`. Candidate updater:
> `scripts/update_hf_candidates_agent_llm_traces.py`. Cross-dataset
> rollup: `data/external/hf_discovery/agent_llm_traces_ingest_summary.json`.
> Tests: `tests/test_hf_agent_llm_traces_ingest.py` (20 tests, all green).

#### Exgentic agent-LLM-traces — Tier-5 OpenTelemetry agent spans

- **Provenance.** [`Exgentic/agent-llm-traces`](https://huggingface.co/datasets/Exgentic/agent-llm-traces)
  — 1,781 OpenTelemetry agent traces collected May 2026 across 6
  benchmarks × 5 frameworks × 6 frontier models (Claude / GPT / Gemini /
  DeepSeek / Kimi). License: cdla-permissive-2.0 (Community Data License
  Agreement, permissive variant 2.0 — explicitly redistribution-friendly
  for derivative datasets). Total upstream size 2.77 GB across 39
  parquet files (range 0.27 MiB → 137 MiB per file).
- **Configs ingested.** `swebench_claude_code_shard12` — one mid-sized
  parquet (`data/train-00012-of-00039.parquet`, 41.03 MiB) containing 46
  SWE-bench agent sessions running through the `claude_code` harness on
  `azure/DeepSeek-V3.2` (799 spans) + `azure/Kimi-K2.5` (1,495 spans).
  Flattened to **2,294 span / request rows** (moderate strength).
- **Available signals:** per-span `start_time` + `end_time` (RFC3339
  ISO) → epoch seconds + `duration_ms` (closed-API e2e, see prohibited
  uses), real `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`,
  `gen_ai.request.model` + `gen_ai.response.model`,
  `gen_ai.response.finish_reasons` (stop / length / tool_calls),
  `status.code` (OTel: 0=UNSET, 1=OK, 2=ERROR) → `is_error` /
  `sla_label` / `timeout_label`, payload-size proxies
  (`input_messages_chars` / `output_messages_chars` /
  `tool_definitions_chars` — derived from the dropped raw strings),
  `input_messages_hash` (16-hex sha256 prefix of the input messages JSON,
  a session-affinity / prefix-reuse proxy). Per-session: `harness`,
  `benchmark`, `models`, `session_id`, `max_tokens`, `total_tokens`.
- **Missing signals:** `ttft`, `tpot`, `itl`, real `queue_state`,
  `kv_block_hashes`, `gpu_utilization` / `gpu_memory` / `memory_pressure`,
  `replica_count`, `autoscaling_proxy`, `capacity_proxy`,
  `model_load_event` / `model_unload_event`, `cost_or_region`.
- **Recommended Aurelius uses:**
  - Agent-workload **request-shape priors** — per-LLM-call
    `(input_tokens, output_tokens, finish_reason, request_model)` joint
    distributions for the eval/batch frontier replay corpus. Strongest
    public source for SWE-bench-style multi-turn agent traffic shape
    that the existing `lmcache-agentic-traces` ingester does not cover
    (different harnesses, different models, different benchmarks).
  - Multi-model agent **routing-proxy priors** — `session_id` +
    `input_messages_hash` let consumers measure how often the same
    semantic prefix recurs within a session. Useful as a soft
    prefix-reuse signal in agent workloads (not as a measured cache-hit
    rate — there is no KV-cache instrumentation here).
  - Agent-failure-mode priors — `status_code == 2` (OTel ERROR) and
    `finish_reasons == ['length']` flag are the only failure / timeout
    signals available; combined gives a (weak) SLA-violation prior.
- **Prohibited uses:**
  - **GPU-serving latency calibration.** `duration_ms` is closed-API
    end-to-end span timing (network + provider routing + provider
    serving), NOT a measured GPU TTFT/TPOT. Treating it as a serving
    latency surface would conflate provider routing with GPU-execution
    time. Use `optimum-benchmark/llm-perf-leaderboard` or
    `agent-perf-bench/AgentPerfBench` for that instead.
  - **Cross-vendor latency generalisation.** This shard's models are
    azure-hosted (DeepSeek + Kimi); duration_ms includes azure's
    network and provider-routing latency. Do NOT generalise to
    on-premise / different-region setups.
  - **Production dynamic-frontier calibration.** Tier-5
    request_shape_trace — pilot telemetry remains the only Tier-1
    calibration source.
  - **Treating `is_error == True` as a production SLA-violation rate.**
    OTel ERROR can be provider timeouts / 5xx / tool-call failures;
    without provider SLA contracts this is a (weak) failure proxy only.
- **Bounded ingest layout:** the 41 MiB parquet lives under
  `data/external/hf/Exgentic__agent-llm-traces/raw/` and is
  **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, 5-row fixture (5.0 KiB), and 2.3 MiB
  `normalized_sample.jsonl` (cdla-permissive-2.0 → redistribution-safe)
  ARE committed. Per-config `analysis_sample.jsonl` is gitignored —
  regenerable via `scripts/ingest_hf_agent_llm_traces.py`.

#### ssong1/llmperf-bedrock — Tier-4 closed-managed-API LLMPerf priors (Round 3)

- **Provenance.** [`ssong1/llmperf-bedrock`](https://huggingface.co/datasets/ssong1/llmperf-bedrock)
  — Apache-2.0 Ray LLMPerf (`token_benchmark_ray.py`) benchmark output
  against AWS Bedrock's `anthropic.claude-instant-v1` endpoint, January
  2024. Client ran from an on-premise Kubernetes bastion (per dataset
  card). 4 (input_tokens, output_tokens, concurrency, region) cells ×
  ~50-100 requests per cell = **350 individual requests** committed +
  4 per-run quantile summaries.
- **Why it fills a gap.** Every existing Tier-4 latency benchmark in
  the corpus (AgentPerfBench, Odyn, Memoriant, Intellistream,
  optimum-benchmark) measures **GPU-direct serving** — vLLM / Triton /
  Ray Serve on a known accelerator. The Aurelius **routing** + **deferral**
  objective also needs a **closed managed-API** latency prior (Bedrock,
  Anthropic API, OpenAI API, AzureOpenAI) where the GPU is opaque and
  TTFT/ITL include provider scheduling + closed-source batching + AWS
  network. `ssong1/llmperf-bedrock` is the first such dataset in the
  federated corpus.
- **Available signals (per-request, real):** `ttft_ms`, `itl_ms`
  (= LLMPerf `inter_token_latency_s` × 1000), `e2e_latency_ms`,
  `request_output_throughput_tps`, `number_input_tokens`,
  `number_output_tokens`, `number_total_tokens`, `error_code` +
  `error_msg` + `failure` (derived from `error_code != null`).
- **Per-run roll-ups** (from upstream `*_summary.json`): TTFT / ITL /
  e2e quantiles `p25 / p50 / p75 / p90 / p95 / p99`, output-throughput
  quantiles, `num_completed_requests`, `num_completed_requests_per_min`,
  `error_rate`. Stored under `statistical_rollups.json#per_run_aggregated`.
- **Missing signals:** `tpot`, `queue_state`, `queue_wait`, `kv_cache_size`,
  `memory_pressure`, `gpu_type`, `gpu_utilization`, `batch_size`,
  `engine_version`, `timeout_label`, `autoscaling`, `replica_count`,
  `kernel_duration`, `cache_hit`. `concurrency` is fixed at 1 across
  all four runs in this snapshot — there is **no contention / queue
  signal**.
- **Recommended Aurelius uses:**
  - **Managed-API TTFT / ITL / e2e routing priors** — per `(model,
    region, prompt_kind, input_len, output_len)` cell the corpus now
    has a real measured quantile distribution that the routing /
    deferral / placement engine can consume as a closed-API SLO prior.
  - Constraint-aware evaluation cross-reference — compare GPU-direct
    AgentPerfBench / optimum-benchmark numbers against this Bedrock
    snapshot to measure the "managed-API tax" for the same prompt size.
  - **Region-pricing-vs-latency study** — the `apne1` (ap-northeast-1)
    run has p50 TTFT 0.584 s vs. `default` region 1.219 s for the
    identical model + prompt size; a real-data prior for the
    regional-latency-vs-egress-cost tradeoff in the Aurelius
    residency/routing engine.
- **Prohibited uses:**
  - **GPU-direct serving latency calibration.** TTFT here includes
    Bedrock provider scheduling + AWS network from an on-premise
    bastion. Treating it as GPU-execution TTFT would inflate every
    GPU-direct prior by 1+ network hops.
  - **Queue / concurrency risk calibration.** All 4 runs use
    `num_concurrent_requests = 1`. The Aurelius queue-risk module
    (`aurelius/forecasting/cara_queue_features.py`) MUST NOT consume
    this dataset.
  - **Cross-model latency generalisation.** Bedrock has retired
    `claude-instant-v1` (2024-07). Do not extrapolate to Claude 3 /
    3.5 / 4.x, other Bedrock-hosted providers (AI21, Titan,
    Llama-via-Bedrock), or other managed APIs (Anthropic direct,
    OpenAI, AzureOpenAI, Vertex).
  - **Production dynamic-frontier calibration.** Tier-4 benchmark —
    pilot telemetry remains the only Tier-1 calibration source.
  - **Treating `error_rate = 0.0` as a production SLA guarantee.**
    No failures occurred in this 350-request snapshot; the dataset
    has zero timeout signal.
- **Bounded ingest layout:** the 8 raw `*_summary.json` +
  `*_individual_responses.json` files (~150 KiB total) live under
  `data/external/hf/ssong1__llmperf-bedrock/raw/` and are **gitignored**.
  Committed: per-config `summary.json`, `schema_profile.json`,
  `schema_mapping.json`, `statistical_rollups.json`, 5-row fixture
  (5.4 KiB) and 350-row `committed_normalized_sample.jsonl` (373 KiB —
  Apache-2.0 → redistribution-safe).
- **Round-3 discovery audit summary.**
  `data/external/hf_discovery/round3_broadened_discovery_audit_summary.json`
  records the ingest + 8 negative-result candidates from the same
  Round-3 broadened search (`DistServe/2025-05-06T14-…` and
  `DistServe/test-amd-ci-profiler` — `license_unspecified_low_priority`
  for vLLM CUDA-kernel profiling JSONs whose dataset cards declare no
  SPDX license; `DistServe/test-sample` — out-of-scope image data;
  `deepanjalimishra99/datacenter-traces` — out-of-scope DynamoRIO
  drmemtrace; `intellistream/sage-control-plane-llm-workloads` —
  `insufficient_sample_no_license` (3 rows of workload config); the
  three previously audited duplicates / gated cases re-noted for
  completeness).
- **Tests:** `tests/test_hf_llmperf_bedrock_ingest.py` (24 tests, all
  green) — including a hard assert that no GPU / queue / batch_size
  signals leak into `available_signals` (this is API-only data) and
  that the closed-API caveat + `concurrency=1` caveat are pinned in
  the `limitations` list.

> **Round-4 broadened discovery 2026-06-01.** One new candidate from
> Round-4 keyword-expansion has been bounded-ingested as Tier-4
> `latency_benchmark_trace`. 7 negative-result records (4 license-
> unspecified or duplicate forks of the canonical SINTEF dataset; 2
> low-information-density code/tool benchmarks; 1 gated companion
> dataset re-confirmed) are recorded for completeness.
>
> Ingested: **`ejhusom/llm-inference-energy-consumption`** — the
> SINTEF Digital + Singapore Management University LLM inference
> energy datasheet (Husom et al. 2024, arxiv:2407.16893
> *The Price of Prompting*). 4 of 16 CSV files were bounded-ingested
> covering 3 independent comparison axes:
>
> - Cross-hardware-tier: `alpaca_gemma_7b_laptop2` (5,099 rows, strong)
>   vs `alpaca_gemma_7b_workstation` (8,735 rows, strong) — first
>   public laptop-tier prior in the Aurelius federated corpus.
> - Cross-workload: `alpaca_gemma_7b_workstation` (instruction) vs
>   `codefeedback_codellama_7b_workstation` (code, 3,109 rows
>   moderate) — same workstation tier, different prompt distributions.
> - Cross-model-size: `codefeedback_codellama_7b_workstation` vs
>   `codefeedback_codellama_70b_workstation` (161 rows, weak; only
>   promoted_for_training_priors) — code workload on the same
>   workstation under a 10× model size jump.
>
> License: **cc-by-sa-4.0** (committed normalised sample is
> redistribution-safe under attribution + share-alike). Engine:
> **Ollama** — first Ollama engine entry in the corpus (existing
> entries are vLLM / SGLang / Triton / Ray Serve / llama.cpp /
> Bedrock).
>
> Ingest script: `scripts/ingest_hf_llm_energy_consumption.py`.
> Rollup at `data/external/hf_discovery/round4_broadened_discovery_audit_summary.json`.
> Tests: `tests/test_hf_llm_energy_consumption_ingest.py` (122 tests,
> all green).

#### ejhusom/llm-inference-energy-consumption — Tier-4 cross-hardware-tier energy + timing (Round 4)

- **Provenance.** [`ejhusom/llm-inference-energy-consumption`](https://huggingface.co/datasets/ejhusom/llm-inference-energy-consumption)
  — cc-by-sa-4.0 datasheet (Husom et al. 2024, arxiv:2407.16893
  *The Price of Prompting: Profiling Energy Use in Large Language
  Model Inference*) by the SINTEF Digital Trustworthy Green IoT
  Software group + Singapore Management University. The full dataset
  spans 16 CSV files = 4 (Alpaca + Code-Feedback) × 5 model
  configurations × 4 hardware classes (laptop1 / laptop2 / workstation
  / server). Engine: **Ollama** (HTTP API; Ollama internally uses
  llama.cpp). Energy measurement: **CodeCarbon** (per-request, kWh).
- **Why it fills a gap.** Every existing Tier-4 latency benchmark in
  the Aurelius federated corpus (AgentPerfBench, Odyn, Memoriant,
  Intellistream, optimum-benchmark, ssong1/llmperf-bedrock) measures
  server-class hardware only — A100 / A10 / T4 / DGX Spark / 32vCPU
  Sapphire-Rapids / Bedrock-closed-API. The Aurelius placement /
  routing / deferral engine previously had **NO public prior** for
  consumer / laptop tier hardware. This is also the first Ollama-engine
  entry in the corpus.
- **Configs ingested (4 of 16 available).** Picked to cover three
  independent comparison axes simultaneously:
  - **`alpaca_gemma_7b_laptop2`** (5,099 rows, strong) —
    laptop2 + gemma:7b + alpaca instruction prompts.
  - **`alpaca_gemma_7b_workstation`** (8,735 rows, strong) —
    workstation + gemma:7b + alpaca instruction prompts. Matches the
    laptop2 model + workload for direct cross-tier comparison.
  - **`codefeedback_codellama_7b_workstation`** (3,109 rows,
    moderate) — workstation + codellama:7b + Code-Feedback prompts.
    Cross-workload pair with the alpaca / gemma:7b workstation config.
  - **`codefeedback_codellama_70b_workstation`** (161 rows, weak) —
    workstation + codellama:70b + Code-Feedback. Cross-model-size pair
    with codellama:7b; capped at training_priors only due to weak
    sample strength.
- **Available signals (all configs, per-request real values).**
  Ollama-reported `total_duration_ns`, `load_duration_ns`,
  `prompt_duration_ns`, `response_duration_ns`; `prompt_token_length`
  + `response_token_length`; per-request CodeCarbon energy in kWh
  split into `energy_kwh_llm_cpu` / `energy_kwh_llm_gpu` /
  `energy_kwh_llm_total` (CPU+GPU) plus a separate
  `energy_kwh_monitoring` (framework overhead). Wall-clock
  `start_time`, `end_time`, `created_at`.
- **Derived signals (labelled DERIVED in field_quality).**
  `e2e_latency_ms = total_duration_ns / 1e6`,
  `ttft_proxy_ms = prompt_duration_ns / 1e6`,
  `tpot_proxy_ms_per_token = response_duration_ns / (1e6 ×
  response_token_length)`,
  `request_output_throughput_tps = response_token_length /
  (response_duration_ns / 1e9)`.
- **Missing signals.** `ttft_measured` (only an Ollama prompt_eval
  PROXY is available — schema_mapping labels it `derived`); `itl` (no
  per-token streaming timestamps); `queue_state` / `queue_wait` /
  `queue_depth` (concurrency = 1, no contention); `memory_pressure` /
  `gpu_utilization` / `gpu_type` (CodeCarbon reports kWh aggregates,
  not raw nvidia-smi); `batch_size` / `concurrency` (single-stream);
  `timeout_label` / `sla_label` (type='unknown' across all rows);
  `autoscaling` / `replica_count` (single host); `kv_cache_size` /
  `cache_hit` (not instrumented); `kernel_duration` (high-level
  serving, not kernel-level); `carbon_intensity` (energy is in kWh —
  combine with regional g CO2/kWh before reporting carbon).
- **Recommended Aurelius uses.**
  - **Cross-hardware-tier placement priors** — first public dataset
    that lets the placement engine reason about laptop vs workstation
    energy + latency for the SAME (model, workload) pair. For example
    gemma:7b alpaca: laptop2 p50 e2e ~2.0 s vs workstation p50 e2e
    ~35.6 s — a counter-intuitive finding (the labelled "workstation"
    is a slower or older GPU than this laptop's GPU) that the
    placement engine MUST encode rather than naively prefer the
    "workstation" tier.
  - **Cross-workload energy priors** — codefeedback prompts on
    codellama:7b workstation consume ~1.85× the energy per request
    of alpaca prompts on gemma:7b workstation under the SAME
    hardware. The energy term in the constraint-aware engine can now
    differentiate code-completion vs instruction workloads.
  - **Cross-model-size energy priors** — codellama:70b
    energy/request ~ 0.0044 kWh vs codellama:7b ~ 0.00018 kWh on the
    same workstation (24× more energy / request). Lets the deferral
    engine reason about quality vs energy for the same code workload.
  - **Cold-start / model residency proxy** — `load_duration_ns` is
    effectively zero when the model is already resident and non-zero
    on the first request after a model switch; this is the first
    HF-corpus signal that captures residency in `latency_benchmark_trace`
    rather than `cache_residency_trace`.
- **Prohibited uses.**
  - **TTFT calibration as if measured.** `ttft_proxy_ms` is Ollama's
    `prompt_duration` — an approximation of TTFT under single-stream
    serving but NOT a measured first-token wall-clock timestamp.
    The field_quality map marks it `derived`; consumers MUST honour
    that.
  - **Concurrency / queue calibration.** All runs are concurrency = 1.
    The Aurelius queue-risk module and batch-frontier MUST NOT
    consume this dataset.
  - **vLLM / SGLang generalisation.** Ollama wraps llama.cpp under an
    HTTP API; do NOT generalise absolute throughput / latency
    numbers to vLLM / SGLang / TGI without independent validation.
  - **Cross-device generalisation within a tier.** "laptop2" is one
    specific machine; "workstation" is one specific machine. Do NOT
    treat absolute numbers as representative of "all laptops" or
    "all workstations". Use the (model, hardware_tier) cell as the
    join key.
  - **Carbon claims without grid data.** Energy is in kWh —
    combining with regional CO2 g/kWh is mandatory before reporting
    carbon cost. The Aurelius carbon term should pair this with
    ElectricityMaps / WattTime regional intensity.
  - **Production dynamic-frontier calibration.** Tier 4 benchmark —
    pilot telemetry remains the only Tier 1 calibration source.
- **Bounded ingest layout.** Four raw CSVs (~30 MB total: 7.5 + 13.1
  + 8.5 + 0.4 MB) live under
  `data/external/hf/ejhusom__llm-inference-energy-consumption/raw/`
  and are **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, 5-row fixture (≤ 7 KiB), and the
  committed cc-by-sa-4.0 `committed_normalized_sample.jsonl` (≤ 100
  KiB per config; ~ 400 KiB combined) ARE committed. Per-config
  `analysis_sample.jsonl` (multi-MB across the 17,104 rows) is
  gitignored — regenerable via
  `scripts/ingest_hf_llm_energy_consumption.py`.

#### `metrum-ai/llm-perfdata` — Round-5 broadened-discovery multi-source curated GPU-coverage breadth prior

- **What it is.** MIT-licensed multi-source curated ledger maintained
  by Metrum AI. One config (`multi_source_curated_v1`) ships 80 rows
  covering 24 models × 9 GPU types × 5 serving engines × 3 precisions,
  each row carrying TTFT / TPOT / Tokens-per-sec / Concurrency
  measurements that the curator copied from a public upstream
  benchmark report (every row's `Source_URL` is a separate upstream).
- **Why it fills a gap.** Before Round 5 the Aurelius federated
  corpus had NO public prior for **NVIDIA H100, H200, B200, L40S,
  RTX-4090, AMD MI300X, MI355X, or Intel Gaudi 3** — the existing
  Tier-4 latency benchmarks are A100 / A10 / T4 / DGX-Spark /
  32vCPU-C7i / Bedrock-managed-API only. This is also the first
  **SGLang** and **vLLM-ROCm** (and vLLM v0 / v1) coverage in the
  corpus. The placement / routing engine previously could not even
  enumerate the H200 / B200 / Gaudi-3 cells as legal destinations
  with a measurable prior.
- **Coverage by GPU (n per row).** NVIDIA H100 (28), Intel Gaudi 3
  (11), NVIDIA H200 (10), NVIDIA A100 (8), NVIDIA B200 (5),
  AMD MI300X (5), NVIDIA L40S (5), AMD MI355X (2), NVIDIA RTX 4090
  (2), unknown (4). By engine: vLLM (43), SGLang (25), vLLM-ROCm (6),
  vLLM-v0 (3), vLLM-v1 (3).
- **Available signals (with real per-row values, but SPARSE field
  coverage).** `model_id` / `model_size` / `precision` / `gpu_type`
  / `num_gpus` / `engine` (16/16 columns populated 76-80/80); but
  `ttft_ms` 17/80, `tpot_ms` 10/80, `tokens_per_sec` 38/80,
  `prompt_tokens` 4/80, `output_tokens` 4/80, `context_window` 8/80.
  Many rows carry only Tokens_per_sec or only a free-text
  Source_Notes blurb describing aggregate behaviour.
- **Statistical sample strength = weak.** Densest (gpu, engine,
  precision) cell is (NVIDIA A100, vLLM, FP16) with 8 rows — below
  the `moderate` threshold required for
  `promoted_for_performance_priors` and
  `promoted_for_constraint_aware_evaluation`. Promotion is therefore
  gated to **`promoted_for_training_priors` ONLY**. p95 / p99
  percentile claims per cell are explicitly blocked — the rollups
  record an `insufficient_sample_groups` list for every stratum with
  fewer than 5 rows.
- **Missing signals.** `itl`, `e2e_latency`, `queue_state` /
  `queue_wait` / `queue_depth`, `gpu_utilization` / `memory_pressure`,
  `batch_size`, `timeout_label` / `sla_label` / `failure_label`,
  `autoscaling` / `replica_count`, `kv_cache_size` / `cache_hit` /
  `kernel_duration`, **`energy_per_request` / `carbon_intensity` /
  `cost_per_token` / `cost_per_request`** (the entire economic
  signal-set is absent — see the Round-5 negative-result finding
  below).
- **Recommended Aurelius uses.**
  - **Cross-hardware-tier placement breadth.** First public prior
    that lets the placement engine reason about H100 / H200 / B200 /
    L40S / RTX-4090 / MI300X / MI355X / Gaudi-3 cells as legal
    destinations. Use as a **breadth-only** prior — absolute
    numbers cross-source-averaged.
  - **Cross-engine routing breadth.** First SGLang and vLLM-ROCm
    coverage; lets the routing engine include these engines as
    legal destinations and back off using metrum-ai numbers when
    no direct measurement exists.
  - **Cross-cell training prior input.** Use the
    `committed_normalized_sample.jsonl` (all 80 rows) as a training
    prior for any (gpu, engine) → throughput estimator that uses
    the broader corpus's denser cells as the calibration source and
    metrum-ai only as a coverage anchor.
- **Prohibited uses.**
  - **Single-source p95 / p99 percentile claims.** Every row's
    Source_URL is a different upstream report; percentiles inside
    one (gpu, engine) cell would average heterogeneous methodologies.
  - **TTFT / TPOT per-request calibration.** TTFT_ms and TPOT_ms
    semantics are NOT formally defined in the dataset card — they
    are whatever each upstream benchmark report called them. ITL is
    not recorded; TPOT may collapse ITL × output_tokens depending
    on the upstream source.
  - **Queue-risk / batching / concurrency-contention calibration.**
    No queue state, no measured contention; the Aurelius queue-risk
    module and batch-frontier MUST NOT consume this dataset.
  - **Goodput / $ and carbon-cost calibration.** No measured
    cost / energy / carbon fields. The Aurelius goodput/$
    denominator MUST NOT consume metrum-ai absolute numbers — they
    are operational priors only.
  - **Pilot-grade SLA truth.** Tier 4 multi-source curated ledger,
    not pilot telemetry.
- **Round-5 negative-result finding (economic priority).** Round 5
  ran ~80 targeted HF queries for economic + operational signal
  combinations (`gpu pricing`, `gpu hourly price`, `cloud gpu
  pricing`, `spot gpu price`, `cloud billing`, `chargeback`, `cost
  per token`, `cost per request`, `inference cost`, `gpu energy`,
  `energy per request`, `kwh inference`, `carbon intensity
  datacenter`, `electricity price workload`, `price aware
  scheduling`, `cost aware scheduling`, `spot instance trace`,
  `region cost latency`, …). NO public HF dataset was found that
  joins (operational TTFT / TPOT / throughput / queue × measured
  GPU-hour-cost or measured kWh-per-request × verifiable
  provenance). The high-attention dataset shapes that surfaced
  were either: (a) synthetic billing data (`sairamn/gcp-cloud-
  billing-cost` — clearly generated values, `Total Cost (INR)`
  pattern), (b) AI-safety "claim-coherence" eval data
  (`ClarusC64/*-coherence-risk-v0.1` family — text classification,
  not telemetry), or (c) finance training data
  (`Phipper/pe-energy-infrastructure-training-data` — PE deal
  reasoning, not infrastructure). The Aurelius goodput/$
  denominator therefore remains **operator-policy + public-pricing
  prior + regional grid carbon intensity** (from
  ElectricityMaps / ENTSO-E, already integrated). This is a
  **useful negative result**: the public HF dataset space does NOT
  currently close the (operational × economic) join gap; the
  economic side has to come from operator chargeback policy +
  cloud price catalogues + grid carbon intensity, not from public
  HF datasets. The Round-5 audit summary
  (`data/external/hf_discovery/round5_broadened_discovery_audit_summary.json`)
  records this finding under `economic_priority_summary`.
- **Bounded ingest layout.** One raw parquet (~11 KiB) lives under
  `data/external/hf/metrum-ai__llm-perfdata/raw/` and is
  **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, 5-row fixture (≤ 4 KiB) covering
  4 distinct (gpu, engine) pairs, and the committed MIT
  `committed_normalized_sample.jsonl` (full 80 rows, ~ 47 KiB) ARE
  committed. `analysis_sample.jsonl` (same 80 rows, ~ 47 KiB) is
  gitignored — regenerable via
  `scripts/ingest_hf_metrum_llmperfdata.py`.

#### `ssakethch/h200-quantization-benchmarks` — Round-6 first measured-source H200 SXM (Tier-4 latency_benchmark_trace)

- **What it is.** vLLM serving benchmark on NVIDIA H200 SXM (141 GB
  HBM3e, MIG-partitioned) covering 40 instruction-tuned LLMs × 5
  quantizations (AWQ, GPTQ, FP8, BF16, NVFP4) × 5 request rates
  (1, 2, 4, 8, 16). One config (`throughput`) ships 275 rows of real
  measured `mean_ttft_ms` / `median_ttft_ms` / `p99_ttft_ms` +
  matching TPOT and ITL percentiles + per-cell req / output / total
  token throughput + successful / failed counts + run duration +
  input / output tokens. License is **unspecified** upstream (no
  `license:` field in the HF card YAML front-matter).
- **Why it fills a gap.** The corpus previously had NO
  **single-source** measured H200 prior. metrum-ai/llm-perfdata's
  10 H200 rows are multi-source-curated (each row is a different
  upstream report). This dataset gives 55 H200 rows from a SINGLE
  vLLM-MIG campaign with consistent methodology — the first time
  Aurelius can fit a per-(quant, request_rate) latency / throughput
  surface for H200 from one source. Also adds **NVFP4** quantisation
  to the corpus (previously absent — the metrum-ai ledger does not
  include NVFP4).
- **Available signals.** `ttft` (mean+median+p99), `tpot`
  (mean+median+p99), `itl` (mean+median+p99), `throughput`
  (request+output-token+peak-output-token+total-token), `concurrency`
  (request_rate), `input_tokens`, `output_tokens`, `gpu_type`
  (constant: NVIDIA H200 SXM, 141 GB HBM3e MIG), `model_id`,
  `engine` (constant: vLLM), `request_arrival` (request_rate as a
  closed-loop arrival proxy — NOT a real arrival trace),
  `request_completion` (num_successful + duration_s),
  `failure_label` (num_failed; >0 only for one rr=8 FP8 cell).
- **Statistical sample strength = strong.** 275 rows; (quant,
  request_rate) cross-tab has 12-14 rows per cell for AWQ / BF16 /
  FP8 / GPTQ. NVFP4 has only 1 row per cell — flagged as
  `insufficient_sample_groups` in `statistical_rollups.json`.
  Promotion = **`promoted_for_performance_priors`** (+
  `promoted_for_constraint_aware_evaluation`,
  `promoted_for_training_priors`).
- **Missing signals.** `e2e_latency`, `queue_state` / `queue_wait` /
  `queue_depth`, `gpu_utilization` / `memory_pressure`,
  `batch_size`, `engine_version`, `kv_cache_size` / `cache_hit`,
  `kernel_duration`, `timeout_label` / `sla_label`, `autoscaling` /
  `replica_count`, **`energy_per_request` / `carbon_intensity` /
  `cost_per_token` / `cost_per_request`** (entire economic
  signal-set absent — Round-6 confirms the Round-5 negative result
  that public HF datasets do not currently close the operational ×
  economic join gap).
- **Recommended Aurelius uses.**
  - **H200 single-source latency / throughput priors.** Per
    (quantization, request_rate) cell — first single-campaign H200
    cell that exists in the corpus. Use TTFT_p99 + TPOT_p99 + ITL_p99
    for the H200 placement engine's risk priors.
  - **Quantisation-aware throughput priors.** Five quantisations on
    the same hardware × same model family gives a cleaner
    quantisation-effect signal than the multi-source metrum-ai
    ledger.
  - **NVFP4 placement breadth.** First NVFP4 coverage in the corpus
    (5 rows, RedHatAI/gemma-4-31B-it-NVFP4 only). Use for placement
    breadth only — n=1 per cell forbids p99 claims.
  - **Concurrency-saturation prior.** Mean_ttft jumps from
    O(50 ms) at rr=1 to O(34,000 ms) at rr=16 for certain models
    (e.g. dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4) — direct
    measurement of where the serving stack falls over the latency
    cliff.
- **Prohibited uses.**
  - **Real arrival / queue scheduling.** request_rate is a
    closed-loop concurrency level, not a real arrival trace.
  - **Generalisation outside H200 SXM MIG + vLLM.** Single GPU
    class, single engine; cross-engine / cross-GPU extrapolation is
    unsafe.
  - **Subgroup p95 / p99 claims for quant=nvfp4.** Only 5 rows
    across all 5 request rates (1 row per cell) — explicitly
    flagged in `insufficient_sample_groups`.
  - **Treating rr=16 highest-concurrency cells as stable.**
    Backpressure-saturated outliers (mean_ttft up to 34 s for some
    models); fit non-saturation priors using rr ≤ 8 cells only.
  - **Goodput / $ denominator calibration.** No cost / energy /
    carbon fields — economic signals MUST come from operator policy
    + public pricing prior + grid carbon intensity.
  - **Pilot-grade SLA truth.** Tier 4 benchmark, not pilot
    telemetry.
- **Bounded ingest layout.** One raw CSV (~41 KiB) lives under
  `data/external/hf/ssakethch__h200-quantization-benchmarks/raw/`
  and is **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, and the 5-row stratified fixture
  (~5 KiB, covering all 5 quantisations) ARE committed.
  `analysis_sample.jsonl` (full 275 rows, ~250 KiB) is gitignored;
  `committed_normalized_sample.jsonl` is NOT written because
  license=unspecified does not permit redistribution
  (`committed_normalized_sample_reason_skipped` =
  `license_unspecified_no_redistribution_promise`). Regenerable
  via `scripts/ingest_hf_h200_quantization.py`.
- **Round-6 negative-result finding (economic priority).** The
  Round-6 audit inspected all 9 round-5 discovery-only candidates.
  One qualified (this dataset); eight were rejected:
  `sairamn/gcp-cloud-billing-cost` (synthetic billing — `Total Cost
  (INR)` pattern, sequential resource IDs, no provenance),
  `ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1`
  and `ClarusC64/datacenter-power-load-coherence-risk-v0.1`
  (synthetic text-classification eval tasks, n<1K),
  `Phipper/pe-energy-infrastructure-training-data` (private-equity
  finance LLM SFT data, not infrastructure telemetry),
  `uohna/llm_inference_energy_combined.parquet` (empty repository),
  `metrum-ai/llm-perf-dashboard` (HF 404 — repository not found),
  `crozai/vllm-benchmark-coding` (ShareGPT-derived workload input
  fixtures, duplicate of the existing `sharegpt_aiperf` ingester),
  `intellistream/sage-agent-benchmark` (agent capability eval,
  no infrastructure signals). NONE of these added economic
  signals — the Round-5 economic-overlay gap stands. Aurelius'
  goodput/$ denominator remains operator-policy +
  public-pricing-prior + ElectricityMaps/ENTSO-E carbon intensity
  (already integrated). Records under
  `data/external/hf_discovery/round6_broadened_discovery_audit_summary.json`.

#### Round-7 broadened discovery + H200 cross-source methodology drift (2026-06-02)

- **Round-7 broadened HF discovery — 13 discovery-only rejection
  records, ZERO new ingest.** Re-ran ~30 search-term groups against
  the public HF datasets API (`vllm benchmark`, `sglang benchmark`,
  `inference benchmark`, `mlperf`, `tpot`, `ttft`, `queue depth`,
  `prefix cache`, `kv cache`, `gpu telemetry`, `placement trace`,
  `scheduler trace`, `gpu pricing`, `cost aware`, `spot price`,
  `energy trace`, `carbon intensity`, `datacenter telemetry`, …).
  Surfaced 13 newly-appearing candidates (none in the existing
  79-candidate registry); ZERO qualified for bounded ingest. Records
  persist under
  `data/external/hf_discovery/round7_broadened_discovery_audit_summary.json`
  + `data/external/hf_discovery/hf_dataset_candidates.json` so they
  won't be re-discovered. Breakdown: 1 `gated_blocked`
  (`core12345/real_GPU_exp_placement_trace`, 9.94 GB
  `Qwen3-235B-A22B-FP8-traces.tar.gz` with `gated=auto`), 1
  `reject_synthetic_estimates`
  (`odyn-network/benchmark-dataset-different-gpu-workload` — README
  declares `math_engine` VRAM estimates, NOT measurements), 4
  `reject_irrelevant_domain` (LTX video diffusion, MINT clinical-QA,
  retrieval prompts, TTS speech-synthesis), 3 `duplicate_existing`
  (ShareGPT-derived workload-shape inputs for Isabella5, fabric, vrvrv
  benchmarker), 3 `reject_low_value` / `reject_raw_artifacts_only`
  (wseaton/prefix-cache-bench prompts, st192011/KVCaches raw .bin,
  h4shk4t/fast-kv-compaction-cache .pt checkpoint), and 1
  `reject_empty_repository` (`bldeaw/guardrails-load-test-results`,
  usedStorage=0). NONE of the 13 carry economic signals.
- **Round-7 negative-result finding (economic priority).** This is
  the THIRD CONSECUTIVE ROUND (5, 6, 7) confirming the same finding:
  the public HF dataset space does NOT currently close the
  operational × economic join gap. Round 7 was DESIGNED to falsify
  the Round-5 / Round-6 finding (different search-term groups,
  different time-window cohort, broader coverage); it failed to
  falsify. The Aurelius goodput/$ denominator REMAINS
  operator-policy + public-pricing-prior + ElectricityMaps /
  ENTSO-E carbon intensity (already integrated).
- **H200 cross-source methodology drift audit.** Bounded comparison
  between the two H200 measurement campaigns in the federated
  corpus: `ssakethch/h200-quantization-benchmarks @ throughput`
  (275 rows, single-source vLLM H200 SXM MIG-partitioned, real per-cell
  TTFT / TPOT / ITL p50 / p99 + throughput) and the 10 H200 rows in
  `metrum-ai/llm-perfdata @ multi_source_curated_v1` (multi-source
  curated, mixed engines). The metrum H200 slice has only ONE row
  with TTFT+TPOT (SGLang / Llama-3.1-70B / BF16, 8 GPUs, c=10;
  metrum's own source_notes flag TPOT=0.042 ms as "extremely low"),
  ONE row with tokens_per_sec (vLLM / Llama-3.1-8B / FP8, 8 GPUs,
  64,915 tok/s aggregate), and 8 "Target" placeholder rows without
  measurements. Per-GPU normalization: metrum 64,915 / 8 = 8,114
  tok/s per full H200; ssakethch Llama-3.1-8B FP8 at request_rate=4
  on a single MIG-partitioned H200 SXM is 1,596 tok/s per-replica
  (~5× per-replica vs per-GPU gap is consistent with MIG-partition-
  fraction × concurrency, NOT a methodology drift). The bounded
  conclusion is that the two sources are MUTUALLY COMPLEMENTARY
  (ssakethch = depth on single-source vLLM MIG H200; metrum-ai =
  curated breadth across engines / models / vendors) but NOT
  directly cross-comparable per-row. Consumers MUST NOT cross-compare
  ssakethch per-replica tokens/s directly with metrum aggregate
  tokens/s without explicit (MIG partition fraction, num_gpus,
  concurrency, engine) normalization. Recorded in
  `data/external/hf_discovery/h200_cross_source_methodology_audit.json`.

#### Round-8 broadened discovery — license=None failure mode surfaces (2026-06-02)

- **Round-8 broadened HF discovery — 11 discovery-only rejection
  records, ZERO new ingest.** Re-ran ~40 deliberately NEW search-term
  groups against the public HF datasets API (`codecarbon`, `scaphandre`,
  `agent runtime`, `opentelemetry`, `mcp telemetry`, `mlcommons`,
  `datacenter traces`, `cloud billing`, `inference-perf`, `energy
  consumption`, `carbon`, `dynamo`, `tensorrt`, `llmperf`, `anyscale`,
  `perfdata`, `cluster log`, `serverless`, `bedrock`, …) — none of
  which overlap the term groups exhausted in Rounds 5-7. Surfaced 11
  newly-appearing candidates (none in the existing 92-candidate
  registry); ZERO qualified for bounded ingest. Records persist under
  `data/external/hf_discovery/round8_broadened_discovery_audit_summary.json`
  + `data/external/hf_discovery/hf_dataset_candidates.json` so they
  won't be re-discovered.
- **NEW failure mode — license=None on real measurements.** The
  Round-8 sweep surfaced a category absent from Rounds 5-7: FOUR
  datasets carrying REAL infrastructure measurements but no declared
  license. Conservative redistribution policy (committed normalised
  sample requires a declared permissive license) blocks ingest, but
  the existence of these candidates is itself the actionable signal —
  unlike Rounds 5-7's synthetic / duplicate / wrong-domain failure
  modes, a license-clearance contact (or an operator-policy
  permission flow) could plausibly unblock them later.
  - `sasha/co2_models` — vision models (ViT / BEiT / ResNet) on
    CIFAR10 with CodeCarbon `emissions` (kgCO2e), `energy` (kWh),
    `region` (e.g. `virginia`), `gpu_count`, `gpu_model` (e.g.
    `1 x Tesla T4`). **THE FIRST HF candidate in Rounds 5-8 carrying
    simultaneous operational (duration, num_queries) + economic
    (emissions, energy, region) + infrastructure (gpu_model,
    gpu_count) signals together.** Adjacent (CV, not LLM) but the
    region × GPU × energy join keys are reusable as an energy / region
    prior — would directly inform the `gpu_hour_price_usd` and
    `carbon_g_per_kwh` scorer coefficients if the license question is
    resolved.
  - `ohdoking/energy_consumption_by_model_and_gpu` — per-prompt
    CodeCarbon energy / runtime / CO2 across 8 NVIDIA GPU classes
    (RTX 3070 / 3090 / 4090, RTX A4000 / A5000 / A6000, RTX 2000 /
    4000 Ada Gen) + CPU baseline. Multiple HF models (TinyLlama-1.1B
    etc.). 22 CSV files, 10K-100K rows. Would substantially broaden
    the existing `ejhusom/llm-inference-energy-consumption` coverage
    (currently laptop2 + workstation Ollama hosts only) to the
    consumer / workstation RTX tier.
  - `dadadada1/Inference-Performance-Dataset` — H100 token-level LLM
    inference (single-user). Schema: `model` (7b / 13b / 20b),
    `gpu_type` (H100), per-token mean / max / std latency,
    `prefill_time` (≈ TTFT), `avg_token_decode_time` (≈ TPOT),
    `token_throughput`, `early_sync_delay`, `decode_jitter_std`,
    `stall_ratio_95p`, `token_latency_slope`, `sync_cost_ratio`.
    100K-1M rows. UNIQUE in the corpus for **token-level JITTER +
    STALL + SYNC-COST telemetry** — signals not present in
    agent-perf-bench, optimum-benchmark, or any other entry.
  - `anon-betterbench/betterbench-inference-logs` — 1.8M parquet rows
    of inference logs (4.1 GB compressed / 15 GB uncompressed). Schema:
    `start_time` (float), `system_prompt`, `user_input`,
    `model_output`, `model_name`, `temperature`, `inference_time`.
    Real per-request arrival timestamp + e2e — a Tier-5
    `request_shape_trace` candidate. The `anon-` namespace suggests
    anonymised benchmark, not production — even if licensed, ceiling
    at Tier-5.
- **Round-8 hard rejections.**
  - `sairamn/gcp-cloud-billing-cost` — MIT-licensed 100K-1M rows of
    GCP-shaped billing data BUT `Resource ID` is `resource_1 …
    resource_999` (sequential synthetic IDs). Rejected as
    `reject_synthetic_economics` per the binding rule (same logic as
    the Round-4 `tarekmasryo/llm-system-ops-production-telemetry-sft-
    data` rejection).
  - `ClarusC64/datacenter-power-load-coherence-risk-v0.1` — MIT but
    every row carries `source_citation = "Synthetic"` and the card
    YAML marks `validation_status: pre_release`. Rejected as
    `reject_synthetic_estimates`. Also n<1K.
  - `deepanjalimishra99/datacenter-traces` — despite MIT license,
    3,674 downloads, and a name suggesting datacenter telemetry, the
    6,257 siblings are SPEC2017 / SimPoint fingerprint + simpoint
    traces (`bc/bc_web_stanford/fingerprint/bbfp.41`,
    `bc/bc_web_stanford/simpoints/opt.l`, …) — single-thread CPU-
    architecture simulation. Rejected as `reject_irrelevant_domain`.
  - `programasweights/paw-inference-logs` — schema `(spec, input,
    model_prediction, interpreter, program_id, source, ephemeral)`.
    Programs-As-Weights synthesised program execution, NOT serving
    infrastructure. Rejected as `reject_irrelevant_domain`.
  - `minhkhoi1026/opencl-llmperf` — Apache-2.0, 1,344 downloads, but
    schema is `(code, gsize, lsize, execution_time, input_sizes)`
    across 130 OpenCL benchmark configs (BlackScholes, DotProduct,
    MatVecMul). The "llm" refers to the modelling approach (training
    an LLM-based OpenCL kernel-runtime predictor), not the workload.
    Rejected as `reject_out_of_scope`.
  - `ICOS-AI/scaphandre_power_consumption` +
    `ICOS-AI/scaphandre_cpu_usage` — Apache-2.0 Scaphandre power-meter
    exports, 3-second sampling, ~1K-10K rows. REAL telemetry but
    schema is only `(timestamp, value)` with NO workload / model /
    GPU / request-id join key. Cannot be attributed to any LLM
    workload. Rejected as `reject_low_value_no_workload_context`.
- **Round-8 negative-result finding (economic priority).** This is
  the FOURTH CONSECUTIVE ROUND (5, 6, 7, 8) confirming the same
  finding on the ingest dimension: the public HF dataset space does
  NOT currently close the operational × economic join gap. Round 8
  WAS designed to falsify on a deliberately fresh angle set, and DID
  surface a new actionable failure category (license=None on real
  measurements) — but did not falsify on the ingestible-this-round
  dimension. The Aurelius `goodput/$` denominator REMAINS
  operator-policy + public-pricing-prior + ElectricityMaps / ENTSO-E
  carbon intensity (already integrated). Operator-policy-only scorer
  coefficients after Round 8: `gpu_hour_price_usd`,
  `kwh_per_request`, `carbon_g_per_kwh`,
  `spot_interruption_probability`, `egress_cost_per_gb`,
  `regional_price_usd_per_mwh`.
- **Recommended follow-ups.** (i) Contact owners of
  `sasha/co2_models`, `ohdoking/energy_consumption_by_model_and_gpu`,
  `dadadada1/Inference-Performance-Dataset`, and
  `anon-betterbench/betterbench-inference-logs` to request a
  permissive license declaration. (ii) Design an operator-policy
  permission flow that lets an operator confirm "I have explicit
  redistribution consent for this dataset" — would unblock the
  license=None category in general. (iii) The `sasha/co2_models`
  schema is the most economically valuable surfaced anywhere in
  Rounds 5-8 and should be the highest-priority license-clearance
  target.

#### `Lightcap/agent-runtime-telemetry-small` — inaugural `tool_runtime_trace`, Tier-3 cluster-scheduler-trace-family

- **Why this dataset matters for Aurelius.** First public Hugging Face
  dataset in the federated corpus that captures **real measured
  MCP / agent-runtime tool-call execution telemetry** — one row per
  tool call with measured `duration_ms`, terminal `status`, lifecycle
  `stage`, `error_type` for failures, UTC `created_at` / `updated_at`
  timestamps, plus content-addressed payload-size proxies
  (`args_fingerprint` sha256, `args_count`, `kwargs_key_count`,
  `result_payload_bytes`, `artifacts_bytes`). The export covers 2,262
  operations across 22 distinct tools and 8 days of one runtime owned
  by Faruk Alpay (Lightcap). License is **cc-by-4.0** so the bounded
  normalised sample is redistributable.
- **Closes the Round-5 defer.** PR-#141 Round-5 flagged Lightcap as
  `defer_high_value_different_trace_class` because no existing canonical
  type accepted tool-call execution telemetry without misclaiming
  serving signals. This PR introduces the **new `tool_runtime_trace`
  canonical type** in `aurelius/traces/hf_corpus/schemas.py`
  (`ToolRuntimeRecord`) + `aurelius/traces/hf_corpus/promotion.py`
  (allowed promotions = backtest + constraint_aware_evaluation +
  training_priors; **NOT** `dynamic_calibration` — there is no queue /
  replica / GPU-util signal to calibrate the safe utilization frontier
  against). Trust tier: **Tier 3** (real measured execution telemetry,
  job-trace shape — the "jobs" are tool calls, not GPU jobs).
- **Two configs ingested.**
  - `operations` — 2,262 × 33; promoted to
    `promoted_for_backtest` (+ `constraint_aware_evaluation`,
    `training_priors`) at moderate strength. The primary tool-runtime
    evidence: real per-call `duration_ms` + `status` + `error_type` +
    timestamps, with 22 distinct tools (largest = `surface_affinity`
    632 calls, `workflow_run` 324, `alignment_manifest` 252,
    `granite_timeseries_status` 181). Overall error rate ~5.48 %, with
    8 distinct error types (`RuntimeError` 65, `ValueError` 27,
    `DirectInputProvenanceError` 25, `RecoveredRunningOperation` 6,
    `SurfaceAffinityError` 3, `InternalError` 2, `TimeoutError` 1,
    `RemoteDisconnected` 1). Cancellation rate ~0.27 % (6 / 2,262).
    Latency distribution: p50 = 60.25 ms, p90 = 6.62 s, p95 = 19.73 s,
    p99 = 124.97 s, max = 900.59 s — a heavy-tailed real-production
    shape with errors ~6× slower than success at p99
    (518.16 s vs 120.93 s).
  - `tool_summary` — 32 pre-aggregated `(tool_name, status)` buckets;
    promoted to `promoted_for_schema_only` (fixture_only strength —
    32 aggregate rows do not support per-call distributional analysis).
    The per-bucket `avg_duration_ms` / `median_duration_ms` /
    `p95_duration_ms` are recorded in
    `statistical_rollups.json::per_tool_status_aggregates` so the
    routing-quality consumer doesn't have to recompute from
    `operations`. `field_quality=derived` on aggregate fields, as
    documented in the limitations.
- **What Aurelius can do with this dataset.** The constraint-aware
  engine + routing-quality forecasters can now consume:
  - **Per-tool error-rate priors** — `scenario_briefing` 100 % errors
    (6/6), `optimize_schedule` 56 % errors (5/9), `train_model` 50 %
    (1/2), `forecast_observables` 32 % (51/157), `state_decode` 32 %
    (7/22). These are first-class deferral / fallback / retry-budget
    priors for an agent orchestrator.
  - **Per-tool tail-latency priors** — used as timeout-budget priors
    for routing decisions. Stored in
    `statistical_rollups.json::numeric_distributions.duration_ms.per_tool`.
  - **Per-status latency cost-of-failure** — error operations p99
    (518 s) is ~4.3× the success p99 (121 s) → choosing a failing tool
    is expensive; routing-quality scorers should penalise tools with
    high `error_rate × p99_duration_ms`.
  - **args_fingerprint cache-reuse signal** — same `args_fingerprint`
    sha256 across operations indicates same input → potential
    same-result cache hit (proxy for prefix cache at the tool-call
    grain).
- **What this dataset is NOT.** Tool-runtime traces have **NO**
  `model_id`, **NO** `input_tokens` / `output_tokens`, **NO** GPU type,
  **NO** queue depth, **NO** replica count, **NO** KV-cache state,
  **NO** TTFT / TPOT — they do NOT inform LLM serving frontier or
  dynamic utilization frontier calibration. `duration_ms` is the
  closed tool-runtime end-to-end wall clock (including any nested
  LLM-call time inside the tool, but the LLM-call breakdown is not
  exposed). Distinct from `Exgentic/agent-llm-traces`, which is
  request_shape_trace with per-LLM-call spans + `gen_ai.*` semantic
  conventions: Exgentic captures the LLM-call layer; Lightcap captures
  the tool-call layer above it.
- **Bounded ingest layout.** Two raw parquet (`operations.parquet`
  270 KiB + `tool_summary.parquet` 8 KiB) live under
  `data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/` and
  are **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, 5-row fixture (≤ 7 KiB), and the
  committed `normalized_sample.jsonl` (operations: 2,262 rows ~ 3.0 MiB;
  tool_summary: 32 rows ~ 32 KiB) ARE committed — well under the
  100-MiB-per-file / 300-MiB-per-PR policy cap. cc-by-4.0 permits
  redistribution. `analysis_sample.jsonl` is **gitignored** —
  regenerable via `scripts/ingest_hf_lightcap_runtime_telemetry.py`.
- **Honesty + scope guarantees.** No production claim; no scheduler /
  controller / robust energy engine touched; no oracle as headline;
  no Tier-1 promotion; explicit "NOT GPU TTFT/TPOT, NOT LLM serving
  telemetry" caveat pinned in `limitations`; `field_quality` recorded
  for every accepted column (real for measured `duration_ms` /
  `status` / `error_type` / tokens-style payload-size proxies;
  derived for `created_at_s` / `updated_at_s` / `duration_s` /
  `is_error` / `is_cancelled`; derived for aggregate `duration_ms`
  in `tool_summary`); raw parquets gitignored; payload bodies are
  upstream-redacted (only fingerprints + counts + byte totals are
  exported by Lightcap itself).

#### `Lightcap/agent-runtime-telemetry-small` follow-up — `operation_events` + `audit_records` configs

The PR-#143 next-task list flagged the remaining two Lightcap configs
(`operation_events`: 9,903 lifecycle transitions; `audit_records`:
14,053 MCP audit rows) for a follow-on ingest. Both fit the existing
`tool_runtime_trace` canonical type without a new dataclass; this PR
extends `TOOL_RUNTIME_PAYLOAD_FIELDS` + `ToolRuntimeRecord` by 16 new
optional fields (8 per-event + 8 per-audit-record) so the canonical
schema covers both grains end-to-end.

- **`operation_events`** — 9,903 × 13; promoted to
  `promoted_for_backtest` (+ `constraint_aware_evaluation`,
  `training_priors`) at moderate strength. Each row is one lifecycle
  transition (`event_type ∈ {started, stage, completed, failed,
  reconciled}`, `stage ∈ {started, executing, execution_completed,
  completed, accepted, affinity_warning, affinity_rejected,
  artifacts_published, failed, startup_reconciled}`) timestamped by
  `event_time_utc`. The 2,262 operations average 4.4 events per
  operation (min 2, max 8). Per-event `duration_ms` is **DERIVED** —
  computed as ms-since-the-operation's earliest event — so the
  constraint-aware harness can read off per-stage transition latency
  directly without re-grouping by `operation_id`.
  - **Dispatch latency** (started → stage(executing)) — p50 = 19.23 ms,
    p95 = 399.44 ms, max = 35.9 s. This is the real agent-runtime
    delivery overhead from request acceptance to execution start; a
    first-class **routing-quality** prior.
  - **Affinity-warning latency** (affinity_warning stage) — p50 =
    806.16 ms, p95 = 129,218.98 ms, max = 151.7 s. Marks the operations
    where the runtime detected affinity issues; affinity_warning is
    ~42× slower than the executing stage at p50 → a clear lifecycle
    hotspot the routing layer should avoid when soft-routable.
  - **Artifacts-published latency** (artifacts_published stage) —
    p50 = 2.96 s, p95 = 32.1 s, max = 399.0 s. Post-execution result
    persistence overhead for the 694 operations that publish artifacts;
    relevant for deferral / batching decisions on artifact-heavy tools.
  - **Full lifecycle** (completed stage) — p50 = 125.06 ms, p95 = 19.4 s,
    max = 399.1 s. Cross-validates the operations' end-to-end
    `duration_ms` via `operation_id` join.
- **`audit_records`** — 14,053 × 17; promoted to
  `promoted_for_backtest` (+ `constraint_aware_evaluation`,
  `training_priors`) at strong strength. MCP-shell-layer audit log:
  7,012 `tool_requests` + 7,041 `tool_results` paired by `request_id`
  (mean 2.0 records per request). `duration_ms` is **REAL** but
  populated only on `tool_results` rows (50 % of audit_records);
  `tool_requests` rows have `duration_ms = null` because the request
  hasn't completed yet.
  - **MCP-shell-layer latency** (overall tool_results) — count=7,041;
    p50=4.74 ms, p90=58.08 ms, p95=400.13 ms, p99=2,457.75 ms,
    max=900,586.05 ms (~15 min). Heavy tail similar to the operations
    config but at a different measurement boundary: the audit-shell
    `duration_ms` captures the MCP request/response envelope, while
    operations' `duration_ms` captures the internal execution. Joining
    via `request_id` lets the constraint-aware harness compute
    **envelope-vs-execution overhead** (audit_records.duration_ms −
    operations.duration_ms = MCP shell layer cost).
  - **Per-tool MCP-shell failure rate** — overall error_rate = 8.58 %
    on tool_results (604 / 7,041). Higher than the operations-layer
    5.48 % because the audit shell also records some pre-execution
    rejections (status=`error` on tool_results without a corresponding
    runtime operation).
  - **Audit-shell payload-shape priors** — `payload_bytes` /
    `payload_key_count` / `payload_keys` (request body shape) plus
    `response_key_count` / `response_keys` (response shape) on
    tool_results. Distinct from operations' `args_*` / `result_*`
    payload proxies in that audit records the MCP envelope payload
    while operations records the internal request/result body.
- **Schema dimensions added (16 new payload fields).**
  - Per-event lifecycle (`operation_events`): `event_id`, `event_type`,
    `payload_bytes`, `payload_sha256`, `payload_key_count`,
    `payload_keys`, `payload_status`, `payload_stage`.
  - MCP audit-record (`audit_records`): `record_id`, `category`,
    `record_name`, `record_file`, `record_path_scope`, `kind`,
    `response_key_count`, `response_keys`.
  - All 16 added to `aurelius/traces/hf_corpus/schemas.py::TOOL_RUNTIME_PAYLOAD_FIELDS`
    and the `ToolRuntimeRecord` dataclass as `Optional[...]` fields.
    Existing `operations` + `tool_summary` configs are unaffected
    (their normalized rows leave the new fields null).
- **Join keys.**
  - `operation_events.operation_id` ↔ `operations.operation_id` — per
    operation, the events config exposes lifecycle stages; the
    operations config exposes the end-to-end duration. The harness can
    use the operations row's `duration_ms` to compute the
    post-completion stage fraction.
  - `audit_records.request_id` ↔ `operations.request_id` — per MCP
    request, the audit config exposes shell-layer timing; the
    operations config exposes runtime-layer timing. Subtracting yields
    the **MCP shell overhead prior** per tool / per status.
- **Bounded ingest layout.** Two additional raw parquets
  (`operation_events.parquet` 596 KiB + `audit_records.parquet` 2.14
  MiB) live under
  `data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/`
  and are **gitignored**. Per-config processed `summary.json`,
  `schema_profile.json`, `schema_mapping.json`,
  `statistical_rollups.json`, 5-row fixtures (≤ 6 KiB each), and the
  committed `normalized_sample.jsonl` (operation_events: 9,903 rows ≈
  6.7 MiB; audit_records: 14,053 rows ≈ 13.6 MiB) ARE committed —
  well under the 100-MiB-per-file / 300-MiB-per-PR policy cap.
  cc-by-4.0 permits redistribution. `analysis_sample.jsonl` is
  **gitignored** — regenerable via
  `scripts/ingest_hf_lightcap_runtime_telemetry.py`.
- **What this dataset is still NOT.** The follow-up does not change
  the trust ceiling. Tool-runtime traces have **NO** `model_id`,
  **NO** `input_tokens` / `output_tokens`, **NO** GPU type, **NO**
  queue depth, **NO** replica count, **NO** KV-cache state, **NO**
  TTFT / TPOT. The new "dispatch latency" signal from
  `operation_events.started → stage(executing)` is dispatch at the
  **agent-runtime** layer (started-event → executing-event gap), NOT
  cluster-scheduler queue wait. It is a routing-quality / lifecycle
  prior, not a GPU placement signal.
- **Honesty + scope guarantees.** No production claim; no scheduler
  / controller / robust energy engine touched; no oracle as headline;
  no Tier-1 promotion. The follow-up explicitly labels
  `operation_events.duration_ms` as **derived** (real timestamps,
  computed ms-since-started) and `audit_records.duration_ms` as
  **real** (raw measurement on tool_results rows). The
  `limitations` block in each summary pins the "NOT GPU TTFT/TPOT,
  NOT LLM serving telemetry" caveat and the agent-runtime-not-cluster-
  scheduler interpretation of dispatch latency. Tests:
  `tests/test_hf_lightcap_runtime_telemetry_ingest.py` (90 tests
  total: 30 prior + 60 follow-up).

### 7.2 Datasets evaluated but rejected / blocked

| dataset_id | trace_type | state | reason |
|---|---|---|---|
| `lmsys/chatbot_arena_conversations` | `request_shape_trace` | `gated_blocked` | HF gated:auto — requires Terms-of-Use acceptance even with HF_TOKEN |
| `anon8231489123/ShareGPT_Vicuna_unfiltered` | `request_shape_trace` | `candidate` (frontier_value=3) | text-only conversations; no infrastructure signals; existing ShareGPT ingester in `aurelius/traces/sharegpt_aiperf.py` already covers this role |
| `HuggingAGree/AcmeTrace` | `cluster_scheduler_trace` | `duplicate_existing` | re-upload of `Qinghao/AcmeTrace` (same 75 files); the Qinghao mirror is the canonical ingest target |
| `osteele/llm-calibration-db` | `latency_benchmark_trace` (+ telemetry candidate) | `gated_blocked` | HF `gated:manual` — requires manual approval from the dataset owner; `HF_TOKEN` is not authorised. Revisit if access granted. |
| `jaytonde05/iris-prefix-cache-benchmark` | `request_shape_trace` | `reject_low_value` | 20 synthetic prompts only (single `prompt: string` column, 57 KB total); no measured TTFT / cache-hit / GPU / queue / SLA. Existing `jaytonde05/prefixbench` already covers the synthetic prefix-cache role. |
| ~~`jaytonde05/prefixbench`~~ | ~~`candidate`~~ → **ingested 2026-06-01** | see §7.1 | — |
| ~~`semianalysisai/cc-traces-weka-no-subagents-051226`~~ | ~~`candidate`~~ → **ingested 2026-06-01** | see §7.1 | — |
| `tarekmasryo/llm-system-ops-production-telemetry-sft-data` | `telemetry_trace` (claimed) | `reject_low_value` | Self-declared SYNTHETIC despite the "production-telemetry" name — README: "Synthetic data… not real user data. cost_usd and token fields are synthetic estimates (not billing truth)". Tier-6; rejected to enforce the anti-dataset-spam rule. |
| `spiritbuun/turboquant-tcq-kv-cache` | `kernel_profile_trace` (claimed) | `reject_not_a_dataset` | Repository contains quantization codebooks (`.bin` / `.pt` artefacts), not a benchmark dataset. No measured latency / throughput / cache telemetry. |
| `hlarcher/inference-benchmarker` | `request_shape_trace` | `duplicate_existing` | ShareGPT-derived prompt fixtures used to drive the huggingface/inference-benchmarker tool. Workload-shape only; `aurelius/traces/sharegpt_aiperf.py` already covers this role. |
| `Boxoffice1280/Neurips2026_evaluating_accuracy_KV-cache_reuse_techniques` | `cache_residency_trace` | `license_restricted_no_redistribution` | License is cc-by-nc-nd-4.0 — Non-Commercial + No-Derivatives. Normalised samples are derivatives; committing any excerpt violates the No-Derivatives clause. HF metadata reference retained but no ingest. |
| `Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days-SVLSGPU_EvalData` | `request_shape_trace` | `duplicate_existing` | Combines BurstGPT + LMSYSChat prompt traces. The BurstGPT shape role is already covered by `lzzmm/BurstGPT/burstgpt_1_full`; LMSYSChat is request-shape only. license=None. |
| `MCP-1st-Birthday/smoltrace-cloud-cost-tasks` | `mixed_or_unknown_trace` | `reject_synthetic_agent_eval` | Synthetic MCP agent-evaluation task set (smoltrace). No measured infrastructure signals (latency / queue / GPU / cache). Tier 6. |
| `rbgo/llm-inference-benchmark` | `latency_benchmark_trace` | `license_unspecified_low_priority` | Single CSV with inference benchmark numbers, license=None. Without license clarity, committing a normalised sample is unsafe. Lower priority than `odyn-network/odyn-benchmarks` + `memoriant/dgx-spark-kv-cache-benchmark` (both Apache-2.0) which fill the same Tier-4 role. |
| `project-vajra/dev-staging-h100-dgx` | `kernel_profile_trace` | `license_unspecified_low_priority` | NCCL `all_reduce` / `send_recv` CSV traces (compressed `.xz`). Potentially useful as inter-GPU communication priors; license=None. Revisit if licence clarified or if Aurelius adds a multi-GPU placement / collective evaluator. |
| `Exgentic/agent-llm-traces` | `request_shape_trace` | `defer_high_value_large_size` | 1,781 OpenTelemetry agent traces across 6 benchmarks × 5 frameworks × 6 models (Claude / GPT / Gemini / DeepSeek / Kimi). Has span `start_time`/`end_time` + `gen_ai.usage.{input,output}_tokens` + `status.code`. 2.77 GB across 39 parquet files. cdla-permissive-2.0 (redistribution-friendly). HIGH VALUE for agent workload-shape + per-model duration priors, but timing is closed-API end-to-end latency (API + network + serving), NOT GPU-serving telemetry. Deferred to next-run for a targeted single-parquet bounded ingest once the request-shape ingester contract handles OpenTelemetry span lists. |
| `wseaton/prefix-cache-bench` | `request_shape_trace` | `reject_low_information_density` | Single `text` column with 500 prompt strings. Despite the name, contains NO measured cache / latency / queue / GPU signal — workload-shape fixture only. Duplicates the existing `sharegpt_aiperf` role at lower density. |
| `aintech/vdf_prefix-cache` | `mixed_or_unknown_trace` | `reject_misleading_name` | Despite "prefix-cache" in the name, this is a vector-DB VDF (vector-io) export — embedding vectors, not LLM prefix-cache telemetry. Tier 6. |
| `kshitijthakkar/moe-inference-benchmark` | `latency_benchmark_trace` | `defer_pending_schema_inspection` | Apache-2.0 MoE inference benchmark (n<1K rows). README returned HTTP 403 and datasets-server returned 404 during discovery. Deferred until HF auto-conversion completes OR a manual schema probe is done. |
| `kshitijthakkar/large-moe-inference-benchmark` | `latency_benchmark_trace` | `defer_pending_schema_inspection` | Companion "large" MoE benchmark. license=None; schema not yet accessible via datasets-server. Deferred paired with the small MoE benchmark. |
| `JohnGavin/llmtelemetry-metrics` | `mixed_or_unknown_trace` | `reject_no_infrastructure_signal` | `costs.parquet` + `sessions.parquet` with daily billing roll-up (`cost_id`, `project`, `date`, `daily_cost_usd`, `n_sessions`, `duration_min`). NO request-level latency / queue / GPU / cache. Project-level cost accounting, not infrastructure telemetry. |
| `abdallah1008/semantic-router-benchmark-data` | `request_shape_trace` | `reject_classification_labels_only` | Single JSONL with prompt + route-label pairs for training a semantic-router classifier. NO measured routing latency, throughput, model residency, or cache hit signal. The routing-quality term in the Aurelius objective needs measured-routing telemetry — this is router training labels only. |
| `Nathan-Maine/dgx-spark-kv-cache-benchmark` | `latency_benchmark_trace` | `duplicate_existing` | Same KV cache benchmark CSV as `memoriant/dgx-spark-kv-cache-benchmark` (already ingested as Tier-4 `v3_corrected`). Apache-2.0; near-duplicate of the same upstream Nathan-Maine work. |
| `fabric/inference-benchmarker` | `request_shape_trace` | `duplicate_existing` | ShareGPT-derived prompt fixtures used to drive the upstream huggingface/inference-benchmarker tool — identical role to `hlarcher/inference-benchmarker` (already rejected) and to the existing `sharegpt_aiperf` request-shape ingester. |
| `optimum-benchmark/llm-perf-leaderboard@openvino_cpu_unquantized_32vCPU_C7i` | `latency_benchmark_trace` (sub-config) | `reject_failure_only_no_measurements` | Every row is an isolated-process crash (`RuntimeError: Isolated process exited with non-zero code -6` in `report.traceback`); ZERO measured `report.prefill.latency.*` / `report.decode.latency.*` columns in the CSV header. The 9 working `optimum-benchmark` configs already cover the pytorch-cpu C7i baseline for cross-backend comparison. Re-add if a future openvino sub-run produces real latency. |
| `DistServe/2025-05-06T14-automatic-profiling` | `kernel_profile_trace` | `license_unspecified_low_priority` | vLLM CUDA-kernel profiling output for DeepSeek-R1-Distill-Llama-8B on H100, swept across batch_size × prompt_length (43 JSON files). Each file contains per-kernel `cuda_time_us` + per-layer breakdown (`LlamaDecoderLayer` / `RMSNorm` / `VocabParallelEmbedding`) + full vLLM `engine_args` context. **HIGH research value** as a kernel_profile_trace prior — but the dataset card declares no SPDX license. Without license clarity, committing a normalised sample would violate the corpus license-and-gating-recorded gate. Deferred; revisit if DistServe org adds a license. |
| `DistServe/test-amd-ci-profiler` | `kernel_profile_trace` | `license_unspecified_low_priority` | Companion DistServe AMD CI profiler output. Same license issue as the H100 profiling dump. Deferred pending license. |
| `DistServe/test-sample` | `mixed_or_unknown_trace` | `reject_out_of_scope` | `modality:imagefolder` + n<1K. Not an Aurelius-relevant trace; image data. |
| `deepanjalimishra99/datacenter-traces` | `mixed_or_unknown_trace` | `reject_out_of_scope` | DynamoRIO drmemtrace lz4-compressed binary memory traces from SPECrate / lectern / multiple workloads (bc, blender, etc.). Microarchitectural memory-trace data, NOT LLM serving / scheduler / GPU telemetry. Out of scope for Aurelius' constraint-aware LLM-serving decisions. License `mit` but data type is wrong. |
| `intellistream/sage-control-plane-llm-workloads` | `mixed_or_unknown_trace` | `insufficient_sample_no_license` | 3 rows of workload-configuration metadata (`workload_id`, `request_count`, `rate_per_second`, `arrival_pattern`, `model_distribution`, `priority_distribution`, `prompt_len_range`, `output_len_range`, `slo_deadlines`). Aurelius-relevant fields present, but 3 rows is below the fixture-only threshold AND no declared license. |
| `hlarcher/inference-benchmarker` (re-audit) | `request_shape_trace` | `duplicate_existing` (Round 3 re-confirm) | Re-confirmed in Round 3 — ShareGPT-derived prompt fixtures used to drive the huggingface/inference-benchmarker tool. Existing `sharegpt_aiperf` ingester covers this role. |
| `Nathan-Maine/dgx-spark-kv-cache-benchmark` (re-audit) | `latency_benchmark_trace` | `duplicate_existing` (Round 3 re-confirm) | Re-confirmed in Round 3 — same KV cache benchmark as `memoriant/dgx-spark-kv-cache-benchmark` (already promoted_for_training_priors). |
| `ssong1/llmperf-bedrock` | — | ~~candidate~~ → **ingested 2026-06-01** | see §7.1 (Tier-4 closed-managed-API LLMPerf priors, Round 3). |
| `ohdoking/energy_consumption_by_model_and_gpu` | `latency_benchmark_trace` | `license_unspecified_low_priority` | Energy-by-model-and-GPU benchmark CSV. License unspecified on the dataset card; without redistribution clarity, committing a normalised sample is unsafe. Defer pending license clarification; `ejhusom/llm-inference-energy-consumption` (cc-by-sa-4.0) covers the same role with safe redistribution. |
| `adityaupasani/llm-inference-energy-consumption` | `latency_benchmark_trace` | `duplicate_existing` | Near-identical fork of `ejhusom/llm-inference-energy-consumption`. License unspecified. Duplicate of the canonical SINTEF dataset already ingested in Round 4. |
| `Nayan10767/llm-inference-energy-consumption` | `latency_benchmark_trace` | `duplicate_existing` | Near-identical fork. License unspecified. Duplicate of the canonical SINTEF dataset. |
| `vgyhj/llm-inference-energy-consumption` | `latency_benchmark_trace` | `duplicate_existing` | Near-identical fork. License unspecified. Duplicate. |
| `nishant-k/speculative-decoding-benchmark-results` | `request_shape_trace` | `reject_low_information_density` | Speculative-decoding benchmark output. Each file is a HumanEval task → completion → pass/fail trace. NO measured latency / throughput / energy / GPU / queue signal — code-completion correctness only. Out of scope for the Aurelius constraint-aware engine. |
| `inference-optimization/speculators_benchmarks_tool_call` | `request_shape_trace` | `reject_low_information_density` | BFCL v4 tool-call evaluation tasks (function-call test cases). Workload-shape only — NO measured infrastructure signal. The existing `sharegpt_aiperf` ingester covers request-shape priors with comparable density. |
| `kshitijthakkar/large-moe-inference-benchmark` | `latency_benchmark_trace` | `gated_blocked` | HF gated:manual (companion to `kshitijthakkar/moe-inference-benchmark`). 38 rows, MoE-specific schema (model_id, prompt, tokens_generated, time_seconds, tokens_per_second, total_params, active_params). HF_TOKEN is NOT authorised. Re-confirmed `gated_blocked` in Round 4. |
| `ejhusom/llm-inference-energy-consumption` | — | ~~candidate~~ → **ingested 2026-06-01** | see §7.1 (Tier-4 cross-hardware-tier energy + timing, Round 4). |
| `sairamn/gcp-cloud-billing-cost` | `mixed_or_unknown_trace` | `reject_synthetic_economics` (Round 5) | GCP cloud-billing CSV (18.9 MB, 100K-1M rows). Schema looks economic-relevant on paper (Resource ID, Service Name, Usage Quantity, Region/Zone, CPU/Memory Utilization, Cost per Quantity ($), Total Cost (INR)) but inspection shows clearly SYNTHETIC values: uniform `resource_NNN` IDs, round-number cost columns, absurd network-data magnitudes, INR-conversion pattern. No upstream-source attribution. Synthetic chargeback/cost data; Tier-6. The Aurelius goodput/$ denominator MUST NOT consume synthetic billing rows. |
| `ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1` | `mixed_or_unknown_trace` | `reject_synthetic_ai_safety_eval` (Round 5) | ClarusC64 "coherence-risk" series — text-classification eval that detects when claimed carbon-aware scheduling decisions diverge from claimed emissions outcomes. NO measured infrastructure signal — AI-safety / claim-coherence evaluation, not telemetry. n<1K. |
| `ClarusC64/datacenter-power-load-coherence-risk-v0.1` | `mixed_or_unknown_trace` | `reject_synthetic_ai_safety_eval` (Round 5) | Same ClarusC64 family — datacenter power-load claim coherence vs. true outcome. AI-safety eval format, n<1K, no measured telemetry. |
| `Phipper/pe-energy-infrastructure-training-data` | `mixed_or_unknown_trace` | `reject_out_of_scope` (Round 5) | Private-equity / energy-infrastructure finance training data (297 DPO pairs + 804 SFT conversations + 2,308 Opus-distilled reasoning traces across PE deal analysis / financial modeling / regulatory / strategy). Despite the `energy-infrastructure` tag, this is finance domain text — NO measured infrastructure or telemetry signal. |
| `uohna/llm_inference_energy_combined.parquet` | `mixed_or_unknown_trace` | `reject_empty_dataset` (Round 5) | Empty dataset — only `.gitattributes` is committed in the repository tree; no actual parquet files. Despite the promising name, there is no data to ingest. |
| ~~`Lightcap/agent-runtime-telemetry-small`~~ | ~~`request_shape_trace` (claimed)~~ → **`tool_runtime_trace` (new type)** | ~~`defer_high_value_different_trace_class` (Round 5)~~ → **ingested 2026-06-01** | see §7.1 (inaugural `tool_runtime_trace`; Tier-3 cluster-scheduler-trace family, cc-by-4.0). Round-5 defer cleared by introducing the new canonical type. |
| `metrum-ai/llm-perf-dashboard` | `latency_benchmark_trace` | `defer_pending_inspection` (Round 5) | Companion dashboard dataset to `metrum-ai/llm-perfdata` (same author). Schema not yet inspected because the dashboard format is markdown / static-site oriented rather than tabular. Deferred to a follow-on run if the dashboard exports additional measurement rows beyond what llm-perfdata already covers. |
| `ssakethch/h200-quantization-benchmarks` | `latency_benchmark_trace` | `defer_pending_full_schema_probe` (Round 5) | Benchmark results for 40 quantized + non-quantized instruction-tuned LLMs on NVIDIA H200 MIG (Multi-Instance GPU). Potentially high-value (first H200-MIG coverage + first 40-model breadth quantization comparison) but the dataset card declares no SPDX license. Deferred — without redistribution clarity, committing a normalised sample would violate the corpus license-and-gating gate. Re-audit if author adds a license. |
| `crozai/vllm-benchmark-coding` | `request_shape_trace` | `duplicate_existing` (Round 5) | Coding-workload prompt fixtures designed for `vllm/benchmark_serving.py` — workload-shape only (prompt strings + token counts). NO measured TTFT / TPOT / throughput. `aurelius/traces/sharegpt_aiperf.py` and `hlarcher/inference-benchmarker` (already rejected) cover this role at higher density. |
| `intellistream/sage-agent-benchmark` | `request_shape_trace` | `reject_eval_only_no_telemetry` (Round 5) | AgentBench-style evaluation for tool-selection / task-planning / response-generation accuracy. NO measured latency / GPU / queue / cache signal — agent-capability scoring only. Out of scope for the constraint-aware engine (same rationale as `abdallah1008/semantic-router-benchmark-data`). |
| `metrum-ai/llm-perfdata` | — | ~~candidate~~ → **ingested 2026-06-01** | see §7.1 (Tier-4 multi-source curated GPU coverage breadth prior, Round 5). |
| `core12345/real_GPU_exp_placement_trace` | `mixed_or_unknown_trace` | `gated_blocked` (Round 7) | HF `gated=auto`. 9.94 GB `Qwen3-235B-A22B-FP8-traces.tar.gz` (no README, no license). Name suggests real-GPU placement traces (high-value for Aurelius placement/routing) but HF_TOKEN is not authorised. Revisit if access granted; do NOT unbounded-ingest a 9.94 GB compressed archive. |
| `odyn-network/benchmark-dataset-different-gpu-workload` | `mixed_or_unknown_trace` | `reject_synthetic_estimates` (Round 7) | GPU catalog × LLM workload VRAM benchmark by the same author as the already-ingested `odyn-network/odyn-benchmarks`. README explicitly self-declares the rows as `math_engine` VRAM ESTIMATES paired with `document_engine_recommended_vram_gb` and `llm_judge_verdict` audit columns: "Not suitable as Ground-truth hardware measurements." Synthetic capacity-planning data — NOT measured GPU performance. cc-by-4.0. |
| `BBuf/ltx-fp8-sglang-benchmark-results` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 7) | Lightricks LTX-2.0 / LTX-2.3 text-to-VIDEO diffusion benchmark on a single H100. Metrics are total_s / denoise_s / decode_s for VIDEO generation (8 inference steps, 5.04 s clip), NOT LLM serving TTFT / TPOT / throughput. license=None. |
| `Isabella5/sglang-seglen-benchmark` | `request_shape_trace` | `duplicate_existing` (Round 7) | 1.46 GB ShareGPT_V3 + swebench INPUT prompts for SGLang sequence-length benchmark harness. NOT benchmark RESULTS. Duplicates the existing `sharegpt_aiperf` workload-shape role. license=None. |
| `fabric/inference-benchmarker` | `request_shape_trace` | `duplicate_existing` (Round 7) | 2.02 GB ShareGPT-derived inputs for huggingface/inference-benchmarker — mirror of the previously-rejected `hlarcher/inference-benchmarker` rationale. Apache-2.0. |
| `vrvrv/vllm-benchmark-datasets` | `request_shape_trace` | `duplicate_existing` (Round 7) | 11 MB humaneval / spider / dataclaw / novita parquets — INPUT prompts only, no benchmark RESULTS. Duplicate of `sharegpt_aiperf`. Apache-2.0. |
| `ashwinnv/agent-telemetry-prompt-framing-mint-full1035-qwen32b` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 7) | Despite the "agent-telemetry" name, this is a CLINICAL-QA agent eval dataset (MINT medical QA paper replication, Qwen3-32B, 4,535 rows × 36 columns). The `agent_telemetry_mode` column refers to clinical-tools telemetry, NOT server / serving telemetry. NO latency / throughput / GPU / queue / energy fields. mit. |
| `juniworld/prompt_inference_traces` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 7) | 1.7 MB across 26 parquets: prompt (string) + domain_list + url_list. Federated domain / URL retrieval prompts, NOT inference latency / throughput / queue / cost / energy. Despite "inference_traces" in the name, no measured infrastructure signal. mit. |
| `efficient-speech/tts-serving-benchmark` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 7) | Text-to-speech serving benchmark INPUT dataset (HiFi-TTS + VCTK + LJ-Speech + Libri-TTS + Libri-Light + EMOV-DB, 178 MB total). Audio-domain benchmark inputs, not LLM-serving benchmark results. license=None. |
| `wseaton/prefix-cache-bench` | `request_shape_trace` | `reject_low_value` (Round 7) | 194 KB single `text` parquet (500 rows). INPUT prompts only — no measured TTFT / cache-hit / GPU. license=None. Existing `jaytonde05/prefixbench` covers the role at higher density. (Re-confirmed from the earlier audit pass.) |
| `bldeaw/guardrails-load-test-results` | `mixed_or_unknown_trace` | `reject_empty_repository` (Round 7) | usedStorage=0. Files listed in cardData are not actually published. Cannot be ingested. license=None. |
| `st192011/KVCaches` | `mixed_or_unknown_trace` | `reject_raw_artifacts_only` (Round 7) | 2.40 GB of raw `.bin` KV-cache binaries for three text prompts. No measured TTFT / TPOT / cache-hit / GPU / queue. Raw prefix-cache artifacts are not analyzable as a benchmark RESULTS dataset. apache-2.0. |
| `h4shk4t/fast-kv-compaction-cache` | `mixed_or_unknown_trace` | `reject_raw_artifacts_only` (Round 7) | 634 MB single `Qwen3-4B.pt` file. Model-checkpoint-shaped artifact, not a benchmark RESULTS dataset. license=None. |
| `sasha/co2_models` | `latency_benchmark_trace` | `inspect_manually_license_blocked` (Round 8) | REAL CodeCarbon per-run vision-model inference (ViT/BEiT/ResNet on CIFAR10) with `emissions` (kgCO2e), `energy` (kWh), `region`, `gpu_model`, `gpu_count`. **First HF candidate in Rounds 5-8 carrying simultaneous operational + economic + infrastructure signals.** Adjacent (CV, not LLM) but the region × GPU × energy join keys would directly inform `gpu_hour_price_usd` + `carbon_g_per_kwh` scorer coefficients. license=None blocks committed sample. Highest-priority license-clearance target. |
| `ohdoking/energy_consumption_by_model_and_gpu` | `latency_benchmark_trace` | `inspect_manually_license_blocked` (Round 8) | REAL per-prompt CodeCarbon energy / runtime / CO2 across 8 NVIDIA GPU classes (RTX 3070/3090/4090, RTX A4000/A5000/A6000, RTX 2000/4000 Ada Gen) + CPU baseline. Multiple HF models (TinyLlama-1.1B etc.). 22 CSV files, 10K-100K rows. Would broaden existing ejhusom coverage to the consumer/workstation RTX tier. license=None. |
| `dadadada1/Inference-Performance-Dataset` | `latency_benchmark_trace` | `inspect_manually_license_blocked` (Round 8) | H100 token-level LLM inference (single-user) with UNIQUE jitter/stall/sync-cost telemetry: `decode_jitter_std`, `stall_ratio_95p`, `sync_cost_ratio`, `token_latency_slope`. Plus `prefill_time` (≈ TTFT) + `avg_token_decode_time` (≈ TPOT) + `token_throughput`. 100K-1M rows × {7b,13b,20b} models. license=None. |
| `anon-betterbench/betterbench-inference-logs` | `request_shape_trace` | `inspect_manually_license_blocked` (Round 8) | 1.8M parquet rows of inference logs (4.1 GB compressed, 15 GB uncompressed). Schema: `start_time` (float), `system_prompt`, `user_input`, `model_output`, `model_name`, `temperature`, `inference_time`. Real per-request arrival + e2e — Tier-5 arrival prior candidate. The `anon-` namespace suggests anonymised benchmark, not production. license=None. |
| `sairamn/gcp-cloud-billing-cost` | `mixed_or_unknown_trace` | `reject_synthetic_economics` (Round 8) | MIT-licensed 100K-1M rows of GCP-shaped billing data. `Resource ID` is `resource_1 … resource_999` (sequential synthetic IDs diagnostic of fixture data; real Google Cloud invoice exports do NOT use that scheme). Service Name + Region / Zone catalog is real-shape but per-row costs are synthetic. Rejected per the binding "Do NOT treat synthetic cost fields as real economics" rule (same logic as the Round-4 `tarekmasryo` rejection). |
| `ClarusC64/datacenter-power-load-coherence-risk-v0.1` | `mixed_or_unknown_trace` | `reject_synthetic_estimates` (Round 8) | MIT but every row carries `source_citation = "Synthetic"` and the card YAML marks `validation_status: pre_release`. Real-shape datacenter power-load risk schema (`rack_power_kw`, `psu_margin`, `ups_buffer_minutes`, `voltage_variance_pct`, `load_spike_frequency`, `unexpected_resets`) but self-declared synthetic per row. Also n<1K. |
| `deepanjalimishra99/datacenter-traces` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 8) | MIT + 3,674 downloads + name suggests datacenter telemetry, BUT the 6,257 siblings are SPEC2017 / SimPoint fingerprint + simpoint traces (`bc/bc_web_stanford/fingerprint/bbfp.41`, `…/simpoints/opt.l`, …). Single-thread CPU-architecture simulation traces, NOT LLM serving / cluster scheduling. |
| `programasweights/paw-inference-logs` | `mixed_or_unknown_trace` | `reject_irrelevant_domain` (Round 8) | 1K-10K rows. Schema: `spec` / `input` / `model_prediction` / `model_version` / `interpreter` / `program_id` / `source` / `ephemeral`. Programs-As-Weights synthesised program execution logs, NOT serving infrastructure telemetry. license=None. |
| `minhkhoi1026/opencl-llmperf` | `kernel_profile_trace` | `reject_out_of_scope` (Round 8) | Apache-2.0 + 1,344 downloads. Schema: `(code, gsize, lsize, execution_time, input_sizes)` across 130 OpenCL benchmark configs (BlackScholes, DotProduct, MatVecMul, …). The "llm" in the name refers to the modelling approach (training an LLM-based OpenCL-kernel-runtime predictor), not the workload. `execution_time` is OpenCL kernel runtime, NOT LLM serving TTFT / TPOT. Out of scope for the LLM-serving federated corpus. |
| `ICOS-AI/scaphandre_power_consumption` | `telemetry_trace` | `reject_low_value_no_workload_context` (Round 8) | Apache-2.0. Schema is ONLY `(timestamp, power_consumption)`. 3-second Scaphandre sampling on ICOS Federated Learning edge infrastructure. ~1K-10K rows. REAL power telemetry but NO workload / model / GPU / request-id join key — cannot be attributed to any LLM workload. |
| `ICOS-AI/scaphandre_cpu_usage` | `telemetry_trace` | `reject_low_value_no_workload_context` (Round 8) | Apache-2.0 sibling of `scaphandre_power_consumption`. Schema: `(timestamp, cpu_usage)`. Same limitation: no workload join key. (Recorded so future re-discovery doesn't re-evaluate the sibling.) |

### 7.3 Datasets known in repo (non-HF or other ingest paths)

Already in `data/external/`:

- `azure_llm_2024` — Tier 3 cluster scheduler-adjacent (LLM inference
  arrivals). See `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`.
- `azure_llm_2023` — Tier 3.
- `burstgpt` — Tier 4 burst-shape proxy. See
  `docs/BURSTGPT_BACKTEST_RESULTS.md`.
- `alibaba_genai` — Tier 3 GenAI serving. See
  `docs/ALIBABA_GENAI_BACKTEST_RESULTS.md`.
- `alibaba_gpu` — Tier 3 GPU cluster. See
  `docs/ALIBABA_GPU_BACKTEST_RESULTS.md`.
- `mit_supercloud` — Tier 3 cluster trace. See
  `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`.
- `philly` — Tier 3 cluster trace. See `docs/PHILLY_BACKTEST_RESULTS.md`.
- `lmsys_chatbot_arena` — Tier 5 request shape (existing gated ingester
  at `aurelius/traces/lmsys_chatbot_arena.py`).
- `sharegpt_aiperf` — Tier 5 request shape (existing ingester at
  `aurelius/traces/sharegpt_aiperf.py`).

These are NOT in the new `hf_dataset_candidates.json` — they have their
own ingesters under `aurelius/traces/`. The HF discovery + ingestion
pipeline is additive.

## 8. Anti-dataset-spam rule (binding)

The system must NOT optimize for dataset count. A single dataset with
measured queue / SLA / timeout / GPU util / replica count is more
valuable than hundreds of conversation-only datasets. Discovery scores
favour information density:

- `frontier_value_score` is capped at 3 for `request_shape_trace`.
- `gated_blocked` overrides any positive score.
- `reject_low_value` triggers when `frontier_value_score == 1`.
- Multiple ShareGPT / LMSYS clones are deduplicated by classification +
  scoring; the highest-density variant wins.

## 9. Memory + re-discovery rules

Once a dataset appears in this registry it should NOT be re-discovered
in a new run unless:

- A configuration changes (new `--config-name`).
- The dataset itself is updated upstream (new `lastModified` >
  `ingestion_timestamp_s`).
- A new evaluator is added that needs a new available-signal set.

Re-running `scripts/discover_hf_aurelius_datasets.py` rebuilds
`hf_dataset_candidates.json` from scratch. Re-running
`scripts/ingest_hf_aurelius_dataset.py` for an existing
`(dataset_id, config_name)` pair overwrites the corresponding entry in
`canonical_corpus_registry.json`.

## 10. Next actions (documented for the next run)

- Expand the AcmeTrace `kalos_gpu_util_head` analysis sample beyond
  the current 6.7k-row (32 MiB) bound — the wide DCGM CSV (~2,344 host
  columns per row) only delivers `moderate` strength at 32 MiB. The
  full ~843 MiB file would push this to `strong` and unlock
  `promoted_for_dynamic_calibration` for the Kalos DCGM telemetry
  alongside the already-promoted IPMI power telemetry.
- Ingest the remaining AcmeTrace utilisation streams once budget allows:
  `kalos/FB_USED.csv` (1.15 GB — KV-cache memory pressure proxy),
  `kalos/PIPE_TENSOR_ACTIVE.csv` (972 MB — tensor-pipeline utilisation
  proxy), and `seren/CPU_D_Power.csv` (CPU power → energy-aware
  scheduler priors). Use the same wide-utilisation aggregation path
  in `scripts/ingest_hf_acmetrace.py`.
- Cross-validate the AcmeTrace Kalos+Seren job traces against the
  existing Tier-3 traces (Alibaba GPU / Philly / MIT Supercloud) —
  publish a cross-trace queue-wait distribution comparison under
  `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`.
- Revisit `osteele/llm-calibration-db` once manual gate approval is
  obtained — the dataset's `telemetry_samples` + `system_load_snapshots`
  + `inference_overhead_measurements` parquet files are exactly the
  Tier-2 telemetry shape Aurelius needs.
- Add a synthetic `telemetry_trace` smoke fixture so the dynamic-
  calibration evaluator has a positive test path that does not require
  any real telemetry trace to be present in CI.
- ~~Look for Odyn benchmarks (the seed was searched but the corresponding
  HF dataset namespace was not found in the May/June 2026 snapshot).~~
  **Done 2026-06-01** — `odyn-network/odyn-benchmarks` ingested as
  Tier-4 latency_benchmark_trace (4 configs, Apache-2.0). See the
  broadened-discovery audit above.
- Cross-validate Odyn `qwen_chat_streaming` TTFT_avg / TPOT_avg / e2e_avg
  surfaces against AgentPerfBench `trace_replay` for the overlapping
  model class (Qwen) — currently AgentPerfBench only carries Llama /
  Mistral, but a Qwen2.5-7B comparison would let the static frontier
  cross-reference 2 independent measurement campaigns.
- Use Memoriant `v3_corrected` `kv_buffer_mib` vs `cache_type` curve as
  a memory-pressure prior input to the cache/residency forecaster
  (`aurelius/forecasting/cache_prefix_reuse_forecaster.py`) — current
  forecaster assumes f16 KV cost; q8_0 / q4_0 give 47% / 72%
  memory savings that change the prewarming / eviction trade-off.
- Revisit the Intellistream vLLM-HUST leaderboard licence — if the
  dataset card adds a license, re-run ingest with
  `commit_normalized=True` to ship a redistributable normalised
  sample. Currently raw JSON is fetched on demand from HF; the
  fixture + summary covers the schema test but the analysis sample
  is regenerable-from-source only.
- Look for `osteele/llm-calibration-db` access escalation — still
  gated_blocked from PR #133.
- **Done 2026-06-01** — Round-2 broadened discovery: ingested
  `optimum-benchmark/llm-perf-leaderboard` as 9 Tier-4
  `latency_benchmark_trace` configs covering A100 / A10 / T4 / 32vCPU-C7i
  × pytorch / openvino × unquantized / awq / bnb / gptq / torchao. Real
  measured prefill (TTFT) + decode (TPOT) latency at p50/p90/p95/p99
  with per-request GPU/CPU/RAM energy (kWh) — the first public dataset
  in the federated corpus with measured per-request energy at this
  granularity, directly feeding the energy / carbon cost terms in the
  Aurelius objective.
- **Done 2026-06-01** — `Exgentic/agent-llm-traces` ingested as Tier-5
  `request_shape_trace` (config `swebench_claude_code_shard12`, 2,294
  spans, moderate strength, cdla-permissive-2.0 → 2.3 MiB committed
  normalised sample). The OpenTelemetry span-list is flattened to
  per-LLM-call request rows with closed-API `duration_ms`, real
  `input_tokens` / `output_tokens` / `status_code` / `finish_reasons` /
  `request_model`, plus payload-size proxies + `input_messages_hash`.
  Closed-API timing caveat is pinned in `limitations` and enforced by
  `tests/test_hf_agent_llm_traces_ingest.py::test_no_gpu_serving_signals_claimed`.
- Ingest additional `Exgentic/agent-llm-traces` shards on follow-on runs
  to add cross-harness coverage (the SWE-bench/claude_code shard alone
  has only 2 azure-hosted models). Suggested next shards:
  `train-00022-of-00039` (smallest, openai_solo × tau2_airline, OpenAI
  azure/gpt-4.1) and `train-00009-of-00039` (large, 136 MB raw, likely
  ≥10k spans → strong strength).
- Cross-validate `Exgentic/agent-llm-traces` per-request
  `input_tokens` / `output_tokens` joint distribution against
  `sammshen/lmcache-agentic-traces` (the only other agent-workload
  request-shape source in the corpus) — publish a workload-shape
  comparison under
  `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md` to calibrate which
  agent harness produces the heaviest input-token tail.
- Cross-validate `optimum-benchmark/llm-perf-leaderboard` mean_ttft_ms
  / mean_tpot_ms surfaces against `agent-perf-bench/AgentPerfBench`
  `trace_replay` for matched (model_family, batch_size, sequence_length)
  triples — both are Tier-4 latency benchmarks and a cross-reference
  audit would calibrate which measurement campaign is the stronger
  prior for which (GPU, model) cell.
- Use `optimum-benchmark/llm-perf-leaderboard` per-request
  `prefill_energy_gpu_kwh` + `decode_energy_gpu_kwh` × regional CO2
  g/kWh from the existing `caiso_pjm_prices` / WattTime ingester to
  produce a carbon-aware placement prior (model × GPU × quantization →
  gCO2 per request). This would be the first end-to-end energy →
  carbon prior the constraint-aware engine can consume.
- Probe `kshitijthakkar/moe-inference-benchmark` +
  `kshitijthakkar/large-moe-inference-benchmark` once the HF
  auto-conversion completes — these would be the first MoE-specific
  serving latency priors in the corpus (current latency benchmarks
  are dense-only). **Round-3 re-check 2026-06-01**: still gated:manual;
  HF_TOKEN not authorised. Path forward unchanged.
- **Done 2026-06-01** — Round-3 broadened discovery: ingested
  `ssong1/llmperf-bedrock` as the first Tier-4 **closed managed-API**
  `latency_benchmark_trace` in the corpus (Apache-2.0, 350 individual
  LLMPerf requests against AWS Bedrock claude-instant-v1, p25/p50/p75/
  p90/p95/p99 TTFT + ITL + e2e quantiles per run). Fills the
  managed-API vs. GPU-direct routing-prior gap. See §7.1 entry
  "ssong1/llmperf-bedrock — Tier-4 closed-managed-API LLMPerf priors".
- **Round 3 negative-result snapshot.** Two `DistServe` profiling
  dumps + two out-of-scope (`DistServe/test-sample` image data,
  `deepanjalimishra99/datacenter-traces` DynamoRIO drmemtrace) + one
  insufficient-sample (`intellistream/sage-control-plane-llm-workloads`
  3-row workload-config metadata) were rejected. Records persist in
  `data/external/hf_discovery/round3_broadened_discovery_audit_summary.json`
  + `data/external/hf_discovery/hf_dataset_candidates.json` so they
  won't be re-discovered in future runs.
- Cross-validate `ssong1/llmperf-bedrock` p50 TTFT (region=default
  1.219 s vs region=ap-northeast-1 0.584 s for identical model + prompt)
  against any future managed-API benchmark to calibrate the
  regional-egress-vs-latency tradeoff in the Aurelius residency /
  routing engine.
- If the DistServe org adds an SPDX license to either profiling dump,
  promote one of them (target: `H100_llama8b_pp1_tp1/profiling_bs10_pl128.json`
  — smallest representative file) as a Tier-4 `kernel_profile_trace`.
  Per-kernel `cuda_time_us` + `LlamaDecoderLayer` breakdown is exactly
  the GPU-direct counterpart to the closed-API Bedrock prior.
- **Done 2026-06-01** — Round-5 broadened discovery (economic-priority
  pass): ingested `metrum-ai/llm-perfdata` as the first Tier-4
  multi-source curated GPU coverage breadth prior, closing the
  previously-empty H100 / H200 / B200 / L40S / RTX-4090 / AMD MI300X /
  MI355X / Intel Gaudi 3 cells and the SGLang / vLLM-ROCm engine cells.
  MIT-licensed; weak strength → promoted_for_training_priors only.
  See §7.1 entry "metrum-ai/llm-perfdata — Round-5 broadened-discovery
  multi-source curated GPU-coverage breadth prior".
- **Round-5 negative-result snapshot (economic priority).** After
  ~80 targeted economic + operational HF searches (`gpu pricing`,
  `cloud gpu pricing`, `spot gpu price`, `chargeback`, `cost per
  token`, `cost per request`, `inference cost`, `gpu energy`,
  `kwh inference`, `carbon intensity datacenter`, `electricity
  price workload`, `cost aware scheduling`, `spot instance trace`,
  `region cost latency`, …), NO public HF dataset was found that
  joins (operational TTFT / TPOT / throughput / queue × measured
  GPU-hour cost OR measured kWh-per-request × verifiable
  provenance). The shapes that surfaced were either synthetic
  billing (`sairamn/gcp-cloud-billing-cost`), AI-safety
  claim-coherence evals (`ClarusC64/*-coherence-risk-v0.1`
  family), finance training data
  (`Phipper/pe-energy-infrastructure-training-data`), empty
  (`uohna/llm_inference_energy_combined.parquet`), deferred
  high-value-different-trace-class
  (`Lightcap/agent-runtime-telemetry-small`), or
  license-blocker (`ssakethch/h200-quantization-benchmarks`). The
  Aurelius goodput/$ denominator therefore remains
  **operator-policy + public-pricing prior + regional grid carbon
  intensity** (from ElectricityMaps / ENTSO-E, already
  integrated). This is a USEFUL NEGATIVE RESULT recorded in
  `data/external/hf_discovery/round5_broadened_discovery_audit_summary.json`
  under `economic_priority_summary` so that future runs don't
  re-run the same search.
- **Done 2026-06-01** — Lightcap follow-up resolved: chose option (a),
  added a new `tool_runtime_trace` canonical type
  (`aurelius/traces/hf_corpus/schemas.py::ToolRuntimeRecord`) +
  promotion rules + Tier-3 trust-tier mapping. Ingested
  `Lightcap/agent-runtime-telemetry-small` two configs (operations
  2,262 rows moderate strength → `promoted_for_backtest` +
  `constraint_aware_evaluation` + `training_priors`; tool_summary
  32 aggregate rows fixture-only → `promoted_for_schema_only`). See
  §2 (new canonical type added), §7.1 entry "Lightcap/agent-runtime-
  telemetry-small — inaugural tool_runtime_trace, Tier-3 cluster-
  scheduler-trace-family", and `scripts/{ingest,register}_hf_lightcap_runtime_telemetry.py`.
- **Done 2026-06-01 (follow-up)** — Lightcap operation_events +
  audit_records configs ingested into the same `tool_runtime_trace`
  canonical type. operation_events (9,903 lifecycle events × 2,262
  operations, moderate strength) → `promoted_for_backtest` +
  `constraint_aware_evaluation` + `training_priors`. Per-event
  `duration_ms` is computed as ms-since-the-operation's `started`
  event (field_quality=`derived`), exposing dispatch / execution /
  completion-stage latency priors directly: started→executing p50=19
  ms (real dispatch latency at the agent-runtime layer); affinity_warning
  p50=806 ms (stage-specific overhead); full lifecycle p50=125 ms /
  p95=19 s. audit_records (14,053 rows × 17 cols, strong strength)
  → `promoted_for_backtest` + `constraint_aware_evaluation` +
  `training_priors`. MCP-shell-layer `duration_ms` is REAL on
  `category='tool_results'` rows (7,041 / 14,053; p50=4.7 ms / p95=400
  ms / p99=2.5 s / max=900 s; error_rate=8.6%) — a separate
  measurement boundary from operations' runtime-layer timing. Joins:
  `operation_id` (operation_events ↔ operations); `request_id`
  (audit_records ↔ operations). See `scripts/{ingest,register}_hf_lightcap_runtime_telemetry.py`,
  `data/external/hf_discovery/canonical_corpus_registry.json` (52
  entries; +2 new), candidates JSON `focused_audit_2026_06_01d` block,
  and `tests/test_hf_lightcap_runtime_telemetry_ingest.py` (90 tests,
  all green).
- **Next: cross-validate Lightcap tail-latency profile against any
  future tool-runtime trace.** Lightcap's p99 of 125 s + max of
  900 s tells the constraint-aware engine that real production
  agent workloads have heavy tails that must be deferred / preempted
  rather than retried. A second tool_runtime_trace source would
  calibrate whether this tail shape is broadly representative or
  Lightcap-specific.
- **Cross-validate `metrum-ai/llm-perfdata` densest cells against
  matched corpus measurements.** Where metrum-ai records (gpu,
  engine) cells that also appear elsewhere — (A100, vLLM, FP16) →
  agent-perf-bench + odyn cross-reference; (H100, vLLM, BF16) →
  no overlap yet; (L40S, vLLM, BF16) → no overlap yet — publish a
  delta-from-corpus-baseline table under
  `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md` so future
  consumers know how much weight to give curated-vs-measured rows.
- **Re-audit `ssakethch/h200-quantization-benchmarks` once it
  declares a license.** ~~Holding only on the license-and-gating-
  recorded gate.~~ Done 2026-06-01 — see Round-6 entry below.
- **Done 2026-06-01** — Round-6 broadened discovery (focused
  follow-on of the 9 round-5 discovery-only candidates): ingested
  `ssakethch/h200-quantization-benchmarks` as Tier-4
  `latency_benchmark_trace` (275 rows × 21 cols, strong strength).
  First single-source measured H200 SXM MIG vLLM benchmark in the
  corpus + first NVFP4 quantisation coverage. Promotion =
  `promoted_for_performance_priors` (+ `constraint_aware_evaluation`
  + `training_priors`). License is **unspecified** (no `license:`
  field in HF card YAML) → `committed_normalized_sample.jsonl` is
  NOT committed (`committed_normalized_sample_reason_skipped =
  license_unspecified_no_redistribution_promise`); only the 5-row
  stratified fixture (~5 KiB, covering all 5 quantisations) is
  committed. Eight round-5 candidates rejected as discovery-only:
  `sairamn/gcp-cloud-billing-cost` (synthetic billing),
  `ClarusC64/ai-load-carbon-aware-scheduling-coherence-risk-v0.1` +
  `ClarusC64/datacenter-power-load-coherence-risk-v0.1` (synthetic
  ML eval tasks), `Phipper/pe-energy-infrastructure-training-data`
  (finance LLM SFT, not infrastructure),
  `uohna/llm_inference_energy_combined.parquet` (empty repository),
  `metrum-ai/llm-perf-dashboard` (HF 404),
  `crozai/vllm-benchmark-coding` (ShareGPT-derived input fixtures,
  duplicate of `sharegpt_aiperf`),
  `intellistream/sage-agent-benchmark` (agent capability eval, no
  infrastructure signals). See §7.1 entry "ssakethch/h200-
  quantization-benchmarks — Round-6 first measured-source H200 SXM",
  `scripts/ingest_hf_h200_quantization.py`,
  `data/external/hf_discovery/round6_broadened_discovery_audit_summary.json`,
  and `tests/test_hf_h200_quantization_ingest.py` (34 tests, all
  green).
- **Round-6 negative-result snapshot (economic priority).** The
  Round-6 audit re-confirmed the Round-5 finding: NONE of the 9
  round-5 candidates carry economic signals. The single ingested
  Round-6 dataset (H200 quantization) is operational-only — no
  cost / energy / carbon fields. The Aurelius goodput/$ denominator
  remains operator-policy + public-pricing prior + ElectricityMaps
  / ENTSO-E carbon intensity. The H200 ingest still adds real
  alpha to the constraint-aware engine via H200-specific latency /
  throughput priors and the NVFP4 quantisation coverage cell — the
  negative-result-on-economics does NOT make the ingest itself a
  negative result. Recorded in
  `data/external/hf_discovery/round6_broadened_discovery_audit_summary.json`
  under `economic_priority_summary`.
- **Done 2026-06-02** — Round-7 broadened HF discovery audit (no new
  ingest): re-ran ~30 search-term groups against the public HF API,
  surfaced 13 newly-appearing candidates (none in the existing
  79-candidate registry), and rejected ALL 13 as discovery-only with
  explicit reasons. ZERO ingestible candidates. Breakdown: 1
  `gated_blocked` (`core12345/real_GPU_exp_placement_trace`, 9.94 GB
  `Qwen3-235B-A22B-FP8-traces.tar.gz` with `gated=auto`), 1
  `reject_synthetic_estimates`
  (`odyn-network/benchmark-dataset-different-gpu-workload` — README
  declares `math_engine` VRAM estimates, NOT measurements), 4
  `reject_irrelevant_domain` (LTX video diffusion, MINT clinical-QA,
  retrieval prompts, TTS speech-synthesis), 3 `duplicate_existing`
  (ShareGPT-derived workload-shape inputs), 3 `reject_low_value` /
  `reject_raw_artifacts_only` (prompt-only parquet, raw .bin KV-cache
  binaries, raw model checkpoint), and 1 `reject_empty_repository`.
  See §7.4.1 narrative and
  `data/external/hf_discovery/round7_broadened_discovery_audit_summary.json`.
- **Round-7 negative-result snapshot (economic priority).** THIRD
  CONSECUTIVE ROUND (5, 6, 7) confirming the same finding: NONE of
  the 13 Round-7 candidates carry economic signals. Round 7 was
  DESIGNED to falsify the Round-5 / Round-6 finding (different
  search-term groups, broader coverage); it failed to falsify. This
  strengthens confidence in the negative result — the Aurelius
  goodput/$ denominator REMAINS operator-policy + public-pricing-prior
  + ElectricityMaps / ENTSO-E carbon intensity (already integrated).
- **Done 2026-06-02** — H200 cross-source methodology drift audit.
  Bounded comparison between `ssakethch/h200-quantization-benchmarks
  @ throughput` (275 single-source vLLM H200 SXM MIG rows, real
  per-cell TTFT/TPOT/ITL p50/p99) and the 10 H200 rows in
  `metrum-ai/llm-perfdata @ multi_source_curated_v1` (multi-source
  curated, mixed engines; only 1 row with TTFT+TPOT measurements,
  1 with tokens/sec, 8 "Target" placeholder rows). Bounded
  conclusion: the two sources are MUTUALLY COMPLEMENTARY (ssakethch
  depth vs metrum-ai breadth) but NOT directly cross-comparable
  per-row. Per-GPU normalization shows ssakethch single-MIG vs
  metrum 8×H200 aggregate; the ~5× per-replica-vs-per-GPU gap is
  consistent with MIG-partition-fraction × concurrency, NOT a
  methodology drift. metrum's SGLang TPOT=0.042 ms cell is flagged
  in metrum's own source_notes as "extremely low" — likely a unit /
  definition mismatch. Recorded in
  `data/external/hf_discovery/h200_cross_source_methodology_audit.json`
  and §7.4.2 narrative.
- **Next: Continue Round-8+ HF discovery only when new
  high-priority candidates appear.** The known-shape gaps remain:
  (a) a public Tier-2 telemetry export with ACTUAL queue / GPU-util
  / replica state alongside latency (the CARA analysis-tier still
  dominates this); (b) a measured-source dataset joining operational
  telemetry with economic signals (THREE rounds confirm this gap is
  not closed by the public HF ecosystem); (c) a public SGLang +
  H200 measurement campaign (ssakethch is vLLM-only; metrum-ai's
  SGLang H200 row is too thin and unit-suspect to ground a
  cross-engine prior). Pilot telemetry (Tier 1) remains the only
  path to production calibration — no HF dataset closes that gate.

## 11. License + auth

- `HF_TOKEN` is read from the environment by `HFAPIClient`. The token is
  never logged and never written to summary / registry JSON. Gated
  datasets without access are marked `gated_blocked` and skipped.
- Promoted datasets must record a non-`None` `license` string and a
  resolved `gated` boolean. Datasets with `license = None` fail the
  `license_and_gating_recorded` gate.
- This PR does NOT commit any HF token to git, settings.json, env
  examples, or test fixtures.

## 12. Operator redistribution policy (license=None datasets)

> **Default behaviour: deny-all.** With the committed policy file
> shipping zero grants, every `license = None` HF dataset remains
> blocked from ingestion — same posture as before this section landed.
> The framework adds the *structural* ability for an operator to
> deliberately opt in per-dataset, with provenance, expiry, and an
> explicit scope; it does NOT relax the default.

### 12.1 Why the policy exists

Rounds 5-8 of the broadened HF discovery audit (see §7) surfaced a
recurring failure category: real per-run / per-prompt LLM-serving or
energy-attribution measurements published on Hugging Face with
`license = None` on the dataset card YAML. Round-8 alone catalogued
four such candidates carrying together-rare operational + economic +
infrastructure signals:

- `sasha/co2_models` — first HF candidate in Rounds 5-8 with
  simultaneous operational (`duration`, `num_queries`), economic
  (`emissions`, `energy`, `region`), and infrastructure (`gpu_model`,
  `gpu_count`) signals on the same row. CV (ViT / BEiT / ResNet) not
  LLM-serving, but the region × GPU × energy join keys are reusable as
  an energy / region prior.
- `ohdoking/energy_consumption_by_model_and_gpu` — per-prompt
  CodeCarbon energy + CO2 + runtime across 8 NVIDIA GPU classes
  (RTX 3070/3090/4090, A4000/A5000/A6000, RTX 2000/4000 Ada) + CPU
  baseline.
- `dadadada1/Inference-Performance-Dataset` — H100 token-level LLM
  inference with `decode_jitter_std`, `stall_ratio_95p`,
  `sync_cost_ratio` — token-level jitter / stall / sync-cost telemetry
  not present anywhere else in the corpus.
- `anon-betterbench/betterbench-inference-logs` — 1.8M parquet rows
  with `start_time` + `inference_time` (Tier-5 request-shape with
  timestamped arrival).

Under the conservative committed redistribution rule, `license = None`
blocks the federated corpus's committed normalised sample. The
discovery pipeline currently records these as
`inspect_manually_license_blocked` and stops there.

### 12.2 What the policy module adds

`aurelius/ingestion/operator_redistribution_policy.py` defines the
`OperatorPolicyLedger` data structure and a decision API. The
canonical policy file is committed at
`data/external/hf_discovery/operator_redistribution_policy.json` with
zero grants and `policy_default = "deny_all"`. A standalone audit
script at `scripts/audit_hf_operator_policy_status.py` emits
`data/external/hf_discovery/operator_policy_status.json` documenting
the current per-dataset decision for every license-blocked candidate.

The decision API exposes:

```
ledger.permits_redistribution(dataset_id, scope, *, now_iso=None)
    -> PolicyDecision(permitted, reason_code, reason_detail,
                      matched_grant_dataset_id)
```

A grant entry permits a request **only if** every check passes:

1. `granted = true`,
2. `granted_by` non-empty,
3. `granted_at_iso` non-empty,
4. `expires_at_iso` absent or in the future at the moment of the
   decision,
5. requested `scope` listed in the grant's `allowed_scopes` array, and
6. requested `scope` itself in the closed set
   `SUPPORTED_SCOPES = {committed_normalized_sample, bounded_ingestion,
   schema_only}`.

Otherwise the decision is `permitted = False` with one of the closed
reason codes (`no_grant_recorded`, `grant_explicitly_denies`,
`grant_missing_provenance`, `grant_expired`,
`requested_scope_not_in_allowed_scopes`,
`requested_scope_not_in_supported_scopes`).

### 12.3 Safety invariants

- The loader rejects any policy file with `policy_default` other than
  `"deny_all"` — the policy file cannot be edited to silently widen
  redistribution.
- Duplicate `dataset_id` entries are rejected at load time.
- `granted = true` with empty `granted_by` or empty `granted_at_iso`
  is rejected at load time.
- Unsupported scopes in `allowed_scopes` are rejected at load time.
- No HF API call, no `HF_TOKEN` read, no data download.
- The committed policy file ships with zero grants, so behaviour of
  every existing ingestion / discovery script is unchanged.

### 12.4 Default-policy status snapshot

Under the policy file committed in this PR:

| Dataset (Round-8 license-blocked) | Default decision | Reason |
|---|---|---|
| `anon-betterbench/betterbench-inference-logs` | denied | `no_grant_recorded` |
| `dadadada1/Inference-Performance-Dataset` | denied | `no_grant_recorded` |
| `ohdoking/energy_consumption_by_model_and_gpu` | denied | `no_grant_recorded` |
| `sasha/co2_models` | denied | `no_grant_recorded` |

Reproduce:

```
python3 scripts/audit_hf_operator_policy_status.py \
  --now-iso 2026-06-02T00:00:00Z
```

Output: `data/external/hf_discovery/operator_policy_status.json`.

### 12.5 How an operator would unblock one dataset

An operator who has independently verified that they may redistribute
a particular `license = None` HF dataset (because they own it, have an
explicit redistribution permission from the owner, or operate under an
internal data-use agreement) can edit the policy file and add a single
grant entry:

```json
{
  "dataset_id": "sasha/co2_models",
  "granted": true,
  "granted_by": "<operator identifier, e.g. infra-eng@example.org>",
  "granted_at_iso": "2026-06-02T00:00:00Z",
  "allowed_scopes": ["committed_normalized_sample"],
  "notes": "Owner email-confirmed redistribution permission on YYYY-MM-DD; archived under data-clearance/<ticket>.",
  "expires_at_iso": "2027-06-02T00:00:00Z"
}
```

`policy_default` must remain `deny_all` — the loader refuses any other
value. The framework intentionally has no "allow everything" mode; an
operator must opt in **per dataset**.

### 12.6 What the policy does NOT do

- It does NOT call out to the HF API or contact the dataset owner.
- It does NOT relax any existing safety rule. Even with an operator
  grant, downstream ingestion scripts continue to enforce the bounded
  sampling, schema profile, fixture, checksum, and analysis-tier
  rules in §5.
- It does NOT promote any new dataset into the canonical corpus on
  its own — promotion still requires schema_profile + schema_mapping
  + tests + a recommended Aurelius use case + an analysis-sample
  policy record.
- It does NOT close the operational × economic join gap by itself.
  The Rounds 5-8 negative result on economic signals stands; the
  policy framework only changes whether a future operator can opt
  into ingesting one of the four already-identified license-blocked
  candidates.

### 12.7 RedistributionGate — first consumer of the ledger

The Round-8 policy framework (§12.1–§12.6) added a deny-by-default
consent record and an audit script, but it left the *consumer side*
unwritten: the ledger's `permits_redistribution(...)` API was unused
by any sample-commit path. An operator who recorded a grant would
see no behavioural effect.

The `RedistributionGate` milestone closes that loop. The module
`aurelius/ingestion/redistribution_gate.py` defines one canonical
decision function — `decide_redistribution(*, dataset_id, license_str,
scope, ledger, now_iso=None)` — that fuses the two permitted paths
into a single auditable `RedistributionGateDecision`:

1. **Declared permissive license** on the HF card (apache-2.0, mit,
   cc-by-4.0 / cc-by-3.0 / cc-by-2.0, cc0-1.0, cdla-permissive-2.0,
   cdla-permissive-1.0, odc-by-1.0, bsd-3-clause, bsd-2-clause) →
   PERMIT. The ledger is NOT consulted. `reason_code =
   permitted_declared_permissive_license`.
2. **Declared but not in the closed permissive allow-list** (`"other"`,
   `"openrail"`, bare `"cc"`, GPL, custom research licenses, …) →
   DENY. The ledger is NOT consulted. Operator grants cannot override
   an upstream owner's declared restrictive license. `reason_code =
   denied_declared_non_permissive_license`.
3. **License is None / empty / whitespace** → consult the ledger.
   Propagate the ledger's decision verbatim. With the committed
   default policy file shipping zero grants, every license=None
   dataset DENIES with `reason_code = no_grant_recorded`.

The closed permissive allow-list is in
`PERMISSIVE_LICENSE_TAGS`. Each tag maps to the same canonical
`license_status` label already used by
`scripts/commit_hf_gap_normalized_samples.py` (e.g.
`permissive_apache_2_0`, `permissive_mit`,
`permissive_cc_by_4_0`) so the two paths agree on the permissive
cases — wiring the gate in to replace the hard-coded TARGETS table
in a future PR cannot regress the four already-committed normalised
samples.

**Audit artifact.** `data/external/hf_discovery/redistribution_gate_audit.json`
(written by `scripts/audit_hf_redistribution_gate.py`) records the
gate decision for every candidate in the discovery registry. Under
the committed default policy with zero grants:

| Bucket | Candidates | Outcome |
|---|---|---|
| `permissive_apache_2_0` | 26 | permitted |
| `permissive_mit` | 8 | permitted |
| `permissive_cc_by_4_0` | 4 | permitted |
| `permissive_cc0_1_0` | 2 | permitted |
| `permissive_cc_by_2_0` | 1 | permitted |
| `permissive_cdla_2` | 1 | permitted |
| `declared_non_permissive` | 12 | denied (`denied_declared_non_permissive_license`) |
| `unspecified_no_committed_sample` | 45 | denied (`no_grant_recorded`) |
| **Total** | **99** | **42 permitted, 57 denied** |

The four Round-8 license-blocked candidates
(`sasha/co2_models`, `ohdoking/energy_consumption_by_model_and_gpu`,
`dadadada1/Inference-Performance-Dataset`,
`anon-betterbench/betterbench-inference-logs`) all sit in the
`no_grant_recorded` bucket — identical to the §12.4 default-policy
status snapshot and to the pre-gate behaviour. If/when an operator
records a `committed_normalized_sample` grant for one of them, the
gate flips that dataset to permitted *without* widening any other
candidate's decision.

**Safety invariants pinned by tests** (`tests/test_hf_redistribution_gate.py`):

- `classify_license(None) == classify_license("") == classify_license(" ") ==
  "unspecified_no_committed_sample"` — empty/missing licenses never
  drift into the permissive bucket via whitespace normalisation.
- An operator grant for a declared restrictive license (e.g.
  `"other"`) is IGNORED — the gate denies before consulting the
  ledger. This pins the invariant that operator grants record
  consent under `license=None` only, never override an upstream
  restriction.
- Every permissive-licensed candidate in the discovery registry
  remains permitted; if a future PR widens or shrinks
  `PERMISSIVE_LICENSE_TAGS`, the audit-rollup test fails.
- The audit JSON is deterministic for fixed inputs
  (`now_iso`, `git_sha`, candidate registry, policy file) — a
  changed audit JSON in a diff means a real input changed.
- The new module + script + audit JSON contain no `hf_<token>`
  literal, no `os.environ.get("HF_TOKEN")` call, no
  `huggingface_hub` import.

**Backwards compatibility.** The existing
`scripts/commit_hf_gap_normalized_samples.py` is unchanged in this
milestone. The gate exists side-by-side with its hard-coded TARGETS
table; both produce identical outcomes on the permissive cases.
Future PRs may wire the gate in to replace the TARGETS table; until
they do, neither default-policy outcome changes and the
`telemetry_gap_normalized_sample_commit_summary.json` artifact in
§12.4 remains valid.

### 12.8 RedistributionGate — second consumer wires the script's TARGETS table

The previous milestone (§12.7) wired the gate into the audit script
(`scripts/audit_hf_redistribution_gate.py`) but explicitly left
`scripts/commit_hf_gap_normalized_samples.py` carrying its own
hard-coded `license_redistribution_status` / `commit_sample` fields.
This milestone closes that loop: the commit script's TARGETS table
no longer pre-classifies redistribution decisions. Each entry now
holds only `dataset_id`, `config_name`, the raw HF license tag
(`license_tag`), and a human-curated provenance string
(`license_source`). The script asks
`aurelius.ingestion.redistribution_gate.decide_redistribution` for
both the canonical `license_redistribution_status` label and the
permit/deny decision.

**Behavioural equivalence on the four already-committed samples.**
Under the committed default policy (deny-all, zero grants):

| Dataset | license_tag | Gate verdict | reason_code | Commit decision |
|---|---|---|---|---|
| `lzzmm/BurstGPT` | `cc-by-4.0` | permitted | `permitted_declared_permissive_license` | COMMITTED |
| `lsliwko/google-cluster-data-2019-sorted-by-timestamp` | `cc-by-4.0` | permitted | `permitted_declared_permissive_license` | COMMITTED |
| `sammshen/lmcache-agentic-traces` | `mit` | permitted | `permitted_declared_permissive_license` | COMMITTED |
| `semianalysisai/cc-traces-weka-no-subagents-051226` | `apache-2.0` | permitted | `permitted_declared_permissive_license` | COMMITTED |
| `jaytonde05/prefixbench` | `None` | denied | `no_grant_recorded` | SKIPPED (gate denied: no_grant_recorded) |

`total_committed_bytes` is unchanged at 30,366,604 bytes (sum of the
four committed `normalized_sample.jsonl` files); every
`committed_normalized_sample_sha256` recorded in the per-dataset
`summary.json` files is unchanged. The script's `materialize` function
now detects when the gitignored `analysis_sample.jsonl` source is
missing (e.g. a fresh checkout) but the committed
`normalized_sample.jsonl` is present and whose sha256 matches the
recorded value — in that case it idempotently reuses the existing
committed sample instead of reporting SKIPPED. This is what makes the
script safe to re-run in CI to refresh the rollup's gate-derived
metadata.

**New fields the script records.** Each per-dataset `summary.json` now
also carries `redistribution_gate_reason_code`,
`redistribution_gate_reason_detail`,
`redistribution_gate_permitted`, and
`redistribution_gate_operator_grant_dataset_id` (None except when an
operator grant matched). The rollup
(`telemetry_gap_normalized_sample_commit_summary.json`,
`doc_version = telemetry_gap_normalized_sample_commit_v2`) now also
carries `redistribution_gate_scope`,
`redistribution_gate_policy_default`, and
`redistribution_gate_policy_grant_count` at the top level, plus the
per-dataset gate fields on each row. Downstream tooling that pivots on
the gate's closed reason-code set (e.g. operator dashboards) can now
read the same tokens the audit script uses, without re-classifying
licenses.

**Operator-grant smoke test pinned.** A new test
(`test_operator_grant_for_prefixbench_would_permit` in
`tests/test_hf_gap_commit_script_gate_wiring.py`) builds an in-memory
ledger with a `committed_normalized_sample` grant for
`jaytonde05/prefixbench`, calls the script's `evaluate_target` for
that target, and asserts the gate flips to PERMITTED with
`reason_code = permitted_operator_grant`. Nothing is written to disk;
the test verifies the wiring is real (the script consults the ledger,
not a hard-coded `unspecified → deny` table). With the committed
default policy file shipping zero grants this has no observable
effect on the committed artifacts, but it pins the operator-grant
path so a future regression that re-introduces hard-coded denials
fails this test instead of silently breaking operator workflows.

**Forbidden duplications pinned.** The wiring test file also asserts
that the script:

- does NOT carry `license_redistribution_status` or `commit_sample`
  keys in any TARGETS entry (those are computed, not declared),
- does NOT re-declare `PERMISSIVE_LICENSE_TAGS` or any equivalent
  permissive-status table inside the script (only the gate may
  classify licenses),
- does NOT import `huggingface_hub` or `datasets` (sample-commit
  decisions are pure-Python and read only gitignored local files),
- contains no `hf_<token>` literal and no `HF_TOKEN = "hf_..."`
  assignment.

**Tests landed.** 27 new tests in
`tests/test_hf_gap_commit_script_gate_wiring.py` covering: TARGETS
schema is minimal (no pre-classified status); script imports the
canonical gate; `evaluate_target` is a pure function returning a
`RedistributionGateDecision`; under the default policy every TARGETS
entry produces the expected `(permitted, license_status, reason_code)`
triple (5 cases, one per dataset); the rollup carries the new
gate metadata; per-dataset summary.json files carry
`redistribution_gate_reason_code`; the committed
`license_redistribution_status` matches `classify_license(license_tag)`
on every committed dataset; an in-memory operator grant flips the
prefixbench verdict; safety invariants (no HF_TOKEN literal, no HF
SDK import, no duplicated permissive allow-list); and `materialize`
is idempotent on the existing committed samples.

The pre-existing 60 tests in
`tests/test_hf_gap_normalized_samples.py` continue to pass on the
same committed artifacts, and the pre-existing 34 tests in
`tests/test_hf_redistribution_gate.py` continue to pass — including
`test_classify_license_agrees_with_commit_script_targets`, which
pinned the gate's verdicts on the script's licenses before the
wiring landed and now serves as the cross-file backstop that the
two paths stay consistent.

### 12.9 RedistributionGate — third consumer wires per-dataset ingestion (agent-llm-traces)

The second-consumer milestone (§12.8) wired the canonical gate into
`scripts/commit_hf_gap_normalized_samples.py` (the gap-closure commit
script) but explicitly left the per-dataset ingestion scripts
(`scripts/ingest_hf_*.py`) carrying their own hard-coded
`license_redistribution_status` strings. This milestone closes that
loop for the first per-dataset ingest:
`scripts/ingest_hf_agent_llm_traces.py` (Exgentic/agent-llm-traces,
`request_shape_trace`, Tier 5, cdla-permissive-2.0). The script's
summary writer no longer carries
`"license_redistribution_status": "permissive_cdla_2"` inline; it
declares `LICENSE_TAG = "cdla-permissive-2.0"` and `LICENSE_SOURCE`
at module level and asks
`aurelius.ingestion.redistribution_gate.decide_redistribution` for
the canonical status label, the permit/deny decision, and the
reason code.

**Behavioural equivalence on the already-committed normalised sample.**
Under the committed default policy (deny-all, zero grants):

| Dataset | license_tag | Gate verdict | reason_code | committed_normalized_sample |
|---|---|---|---|---|
| `Exgentic/agent-llm-traces` (`swebench_claude_code_shard12`) | `cdla-permissive-2.0` | permitted | `permitted_declared_permissive_license` | unchanged (2,294 rows, 2,322,517 bytes, sha256 a63d93df…) |

`committed_normalized_sample_sha256`, `committed_normalized_sample_bytes`,
and `committed_normalized_sample_rows` in the per-dataset summary.json
are byte-for-byte identical to the v1 values. The only fields that
change are the new gate-derived ones plus the existing
`license_redistribution_status`, which the gate now produces (and
which `classify_license("cdla-permissive-2.0")` happens to return as
`"permissive_cdla_2"` — identical to the v1 hard-coded string).

**New fields the script records.** The per-dataset `summary.json`
gains:

- `redistribution_gate_reason_code` —
  `permitted_declared_permissive_license` under the default policy.
- `redistribution_gate_reason_detail` — free-form audit trail that
  mentions both the raw license tag (`cdla-permissive-2.0`) and the
  canonical status code (`permissive_cdla_2`).
- `redistribution_gate_permitted` — `true` under the default policy.
- `redistribution_gate_operator_grant_dataset_id` — `null` (no
  ledger consultation: declared permissive license path).
- `redistribution_gate_scope` — `committed_normalized_sample`.

The cross-dataset rollup
`data/external/hf_discovery/agent_llm_traces_ingest_summary.json`
gains `redistribution_gate_scope`,
`redistribution_gate_policy_default`, and
`redistribution_gate_policy_grant_count` at the top level plus the
per-dataset gate fields on every row;
`doc_version` bumps from `exgentic_agent_llm_traces_ingest_summary_v1`
to `exgentic_agent_llm_traces_ingest_summary_v2` so downstream
tooling can pivot on the new fields.

**Operator-grant smoke test pinned.** A new test
(`test_evaluate_redistribution_operator_grant_for_none_license_permits`
in `tests/test_hf_agent_llm_traces_gate_wiring.py`) builds an
in-memory ledger with a `committed_normalized_sample` grant for
`Exgentic/agent-llm-traces`, calls the script's
`evaluate_redistribution` with `license_tag=None`, and asserts the
gate flips to PERMITTED with
`reason_code = permitted_operator_grant`. Nothing is written to
disk; the test verifies the wiring actually consults the ledger
(the script is not carrying a hard-coded permit for this dataset).
The complementary `test_evaluate_redistribution_none_tag_default_ledger_denies`
pins the opposite direction: under the default ledger, a `None` tag
denies with `no_grant_recorded` — so the wiring would catch a
regression that silently widened redistribution if the dataset's
declared license were ever removed.

**Forbidden duplications pinned.** The wiring test file also
asserts that the script:

- does NOT re-declare `PERMISSIVE_LICENSE_TAGS` or any equivalent
  permissive-status table (only the gate may classify licenses),
- does NOT hard-code the `"permissive_cdla_2"` status string in any
  code path outside docstrings (the gate produces it via
  `decide_redistribution`),
- contains no `hf_<token>` literal and no `HF_TOKEN = "hf_..."`
  assignment.

**Tests landed.** 17 new tests in
`tests/test_hf_agent_llm_traces_gate_wiring.py` covering: license
constants exist at module level; script imports the canonical gate;
no duplicated permissive allow-list / hard-coded status string;
`evaluate_redistribution` is a pure function returning a
`RedistributionGateDecision`; default-tag/empty-ledger PERMITS;
`license_tag=None`/empty-ledger DENIES; operator grant flips the
None-tag verdict to PERMIT; the committed summary.json carries the
new gate metadata; the committed `license_redistribution_status`
matches `classify_license(s["license"])`; the rollup carries the
v2 metadata + per-row gate fields; no HF_TOKEN literal;
`audit_one` accepts `ledger` as a keyword-only optional argument;
`_load_ledger` falls back to `OperatorPolicyLedger.empty()` when
the policy file is missing (fresh-checkout self-sufficiency rail).

The pre-existing 20 tests in
`tests/test_hf_agent_llm_traces_ingest.py` continue to pass on the
updated committed summary, and the pre-existing 34 tests in
`tests/test_hf_redistribution_gate.py` continue to pass —
including `test_classify_license_agrees_with_commit_script_targets`,
which pinned the gate's verdicts on every license tag the second
consumer ships and remains the cross-file backstop that the second
and third consumers stay consistent with each other and with the
audit script.

**Next.** Extend the same pattern to the remaining per-dataset
ingestion scripts that still hard-code license verdicts:
`scripts/ingest_hf_h200_quantization.py` (license = `None` on the
HF card),
`scripts/ingest_hf_llm_energy_consumption.py` (`cc-by-sa-4.0`),
`scripts/ingest_hf_latency_benchmarks.py` (mixed Apache-2.0 / `None`),
and `scripts/ingest_hf_optimum_benchmark.py` (`None`). Each
follows the same recipe: lift the raw license tag to a module
constant, call `decide_redistribution`, record the gate fields on
the per-dataset summary, bump the rollup `doc_version` to `v2`. As
with this PR, the operator-grant path remains opt-in: an operator
recording a per-dataset grant in
`operator_redistribution_policy.json` flips the verdict without a
code change. Pilot telemetry (Tier 1) remains the only path to
production calibration; no HF dataset closes that gate.
