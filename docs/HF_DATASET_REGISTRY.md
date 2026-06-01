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
- Ingest `Exgentic/agent-llm-traces` next (1,781 OpenTelemetry agent
  traces, cdla-permissive-2.0, 2.77 GB across 39 parquet files). Plan:
  download the smallest parquet file (~70 MiB) and normalise the span
  list into a `request_shape_trace` extended with `duration_ms`,
  `input_tokens`, `output_tokens`, `status_code` per LLM call. This
  fills the agent-task duration / token-usage gap currently covered
  only by `sammshen/lmcache-agentic-traces`.
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
  are dense-only).

## 11. License + auth

- `HF_TOKEN` is read from the environment by `HFAPIClient`. The token is
  never logged and never written to summary / registry JSON. Gated
  datasets without access are marked `gated_blocked` and skipped.
- Promoted datasets must record a non-`None` `license` string and a
  resolved `gated` boolean. Datasets with `license = None` fail the
  `license_and_gating_recorded` gate.
- This PR does NOT commit any HF token to git, settings.json, env
  examples, or test fixtures.
