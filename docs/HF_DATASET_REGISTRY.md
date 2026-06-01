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

### 7.2 Datasets evaluated but rejected / blocked

| dataset_id | trace_type | state | reason |
|---|---|---|---|
| `lmsys/chatbot_arena_conversations` | `request_shape_trace` | `gated_blocked` | HF gated:auto — requires Terms-of-Use acceptance even with HF_TOKEN |
| `anon8231489123/ShareGPT_Vicuna_unfiltered` | `request_shape_trace` | `candidate` (frontier_value=3) | text-only conversations; no infrastructure signals; existing ShareGPT ingester in `aurelius/traces/sharegpt_aiperf.py` already covers this role |
| ~~`jaytonde05/prefixbench`~~ | ~~`candidate`~~ → **ingested 2026-06-01** | see §7.1 | — |
| ~~`semianalysisai/cc-traces-weka-no-subagents-051226`~~ | ~~`candidate`~~ → **ingested 2026-06-01** | see §7.1 | — |

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

- Re-run `scripts/audit_cara_swissai_telemetry.py` against CARA
  `train.jsonl` + `train_queue_details.jsonl` with a larger per-file
  budget (50-100 MiB) so the analysis sample reaches `strong` strength
  and CARA can be promoted to
  `promoted_for_dynamic_calibration`.
- Add a `cache_residency_trace` ingest path for
  `jaytonde05/prefixbench` (flatten nested `metadata.prefix_group`).
- Inspect (without ingesting) the SemiAnalysis WEKA traces under a
  manually approved bounded budget; the dataset's KV-block-hash
  structure could populate the first real Tier 3-4 cache-residency
  evidence inside the federated corpus.
- Add a synthetic `telemetry_trace` smoke fixture so the dynamic-
  calibration evaluator has a positive test path before any real
  telemetry trace lands.
- Look for Odyn benchmarks (the seed was searched but the corresponding
  HF dataset namespace was not found in the May 2026 snapshot).

## 11. License + auth

- `HF_TOKEN` is read from the environment by `HFAPIClient`. The token is
  never logged and never written to summary / registry JSON. Gated
  datasets without access are marked `gated_blocked` and skipped.
- Promoted datasets must record a non-`None` `license` string and a
  resolved `gated` boolean. Datasets with `license = None` fail the
  `license_and_gating_recorded` gate.
- This PR does NOT commit any HF token to git, settings.json, env
  examples, or test fixtures.
