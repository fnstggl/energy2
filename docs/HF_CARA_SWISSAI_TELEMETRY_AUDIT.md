# HF Telemetry-Candidate Audit — CARA + SwissAI

> **Focused HF audit. Discovery / data-engine PR — no production claims,
> no controllers modified, no robust energy engine touched.** Pilot
> telemetry (Tier 1) remains the only production calibration source. This
> document records every PHASE 0-9 finding required by the audit charter
> for `asdwb/cara_latency_prediction` and `eth-easl/swissai-serving-trace`.
>
> **Read first:**
> - `docs/HF_DATASET_REGISTRY.md` (federated corpus design + trust hierarchy)
> - `docs/RESULTS.md`
> - `docs/PUBLIC_TRACE_BACKTESTS.md`
> - `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`
> - `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`

## 0. Safety + scope

- **Branch:** `feature/hf-cara-swissai-telemetry-audit`
- **Raw downloads** for both datasets are gitignored
  (`data/external/hf/*/raw/*` + `data/external/hf/*/*/processed/analysis_sample.jsonl`).
- **HF_TOKEN** is read from the environment by the audit script and is
  never printed, logged, committed, or echoed.
- **No production scheduler** behaviour, robust energy engine, controllers,
  or frontier modules are modified by this PR.

## 1. Metadata + file inventory

### A. `asdwb/cara_latency_prediction`

| field | value |
|---|---|
| URL | <https://huggingface.co/datasets/asdwb/cara_latency_prediction> |
| license | apache-2.0 |
| gated / private | false / false |
| last_modified | 2026-04-11T16:33:03Z |
| size_categories | 100K<n<1M |
| total_storage | 4.0 GB |
| siblings | `test.jsonl` (49.1 MB · 45 k rows), `test_queue_details.jsonl` (95.3 MB · 45 k rows), `train.jsonl` (392.3 MB · 359 k rows), `train_queue_details.jsonl` (812.4 MB · 359 k rows) |
| configs | none declared on HF card; the audit treats each file as one config |

CARA is a sweep collected on the CloudLab research cluster (March 18-19,
2026) covering **18 model instances across 4 GPU types**:

| model | gpu | instances | TP | vLLM |
|---|---|---|---|---|
| Qwen2.5-72B | A100 80GB × 2 | 2 | 2 | cara_v_11 |
| Qwen2.5-14B | V100 32GB × 4 | 3 | 4 | cara_v_11 |
| Qwen2.5-7B  | A30  24GB     | 5 | 1 | cara_v_11 |
| Qwen2.5-3B  | A30  24GB     | 3 | 1 | cara_v_11 |
| Qwen2.5-3B  | P100 16GB     | 5 | 1 | cara_v_11_p100 |

### B. `eth-easl/swissai-serving-trace`

| field | value |
|---|---|
| URL | <https://huggingface.co/datasets/eth-easl/swissai-serving-trace> |
| license | **other** (not OSI-permissive; redistribution requires ToS check) |
| gated / private | false / false |
| last_modified | 2026-04-28T09:51:24Z |
| size_categories | 10M<n<100M |
| total_storage | 21.4 GB |
| siblings | `trace.jsonl` (7.0 GB · 16.3M rows), `qwen3-32b-buckets.jsonl` (4.6 GB · 4.0M rows), `qwen3-32b-bucket-reuse.jsonl` (3.7 GB), `llama3-70b_bucket-reuse.jsonl` (4.9 GB), `qwen380b_thinking_bucket-reuse.jsonl` (385 MB), `qwen380b_instruct_bucket-reuse.jsonl` (307 MB), `data.jsonl` (235 MB), `apertus-70b-bucket-reuse.jsonl` (40 MB), `datasys_trace.py` (loader script) |

This PR audits 5 (dataset, config) units. The remaining bucket-reuse
files (llama3-70b, apertus-70b, qwen380b instruct/thinking) are
**deferred** — same schema as `qwen3-32b-bucket-reuse`; ingesting them
would only add per-model variants, not new evidence kinds.

## 2. Audit units + schema profile / mapping artefacts

Per the audit charter PHASE 2, every unit has a `schema_profile.json` +
`schema_mapping.json` written to disk and validated by
`tests/test_hf_cara_swissai_audit.py`:

| (dataset, config) | raw file | analysis chunk | profile + mapping path |
|---|---|---|---|
| `asdwb/cara_latency_prediction` / `test_flat` | `test.jsonl` | 10 MiB head (9,605 rows) | `data/external/hf/asdwb__cara_latency_prediction/test_flat/processed/{schema_profile,schema_mapping}.json` |
| `asdwb/cara_latency_prediction` / `test_queue_details` | `test_queue_details.jsonl` | 10 MiB head (4,876 rows) | `data/external/hf/asdwb__cara_latency_prediction/test_queue_details/processed/{schema_profile,schema_mapping}.json` |
| `eth-easl/swissai-serving-trace` / `trace` | `trace.jsonl` | 10 MiB head (25,409 rows) | `data/external/hf/eth-easl__swissai-serving-trace/trace/processed/{schema_profile,schema_mapping}.json` |
| `eth-easl/swissai-serving-trace` / `qwen3_32b_buckets` | `qwen3-32b-buckets.jsonl` | 10 MiB head (19,130 rows) | `data/external/hf/eth-easl__swissai-serving-trace/qwen3_32b_buckets/processed/{schema_profile,schema_mapping}.json` |
| `eth-easl/swissai-serving-trace` / `qwen3_32b_bucket_reuse` | `qwen3-32b-bucket-reuse.jsonl` | 10 MiB head (16,593 rows) | `data/external/hf/eth-easl__swissai-serving-trace/qwen3_32b_bucket_reuse/processed/{schema_profile,schema_mapping}.json` |

Every observed raw column / 1-level-nested key is classified as
`accepted` (mapped to a normalised field with field_quality +
aurelius_signal_category + usable_for) or `rejected` (no mapping →
ingestion refuses normalisation). All five units pass with **zero
rejected columns**.

## 3. Trust assessment + per-dataset checklist

### A. CARA (`asdwb/cara_latency_prediction`)

| Question | Answer |
|---|---|
| Real request arrivals? | ✅ yes — `prediction_timestamp` (Unix s) per request |
| Real completions? | ✅ yes — `completion_timestamp` (Unix s) per request |
| Actual measured latency? | ✅ yes — `actual_e2e_latency`, `actual_ttft`, `actual_tpot` (client-measured) |
| Queue wait / queue state? | ✅ yes — `num_running`, `num_waiting`, `num_active_decode_seqs`, `pending_prefill_tokens`, `pending_decode_tokens`, and the full nested `schedule_state.running_requests[]` / `waiting_requests[]` arrays in the `*_queue_details.jsonl` companions |
| Scheduler / routing state? | ✅ yes — `token_budget_per_iter`, `prefill_chunk_size`, `max_num_seqs`, `num_preempted`, EMA throughput counters |
| Model / server / GPU identity? | ✅ yes — `instance_id` (CloudLab host:port), `instance_type` (`qwen2.5-3b_p100` style) |
| Status / error / timeout labels? | ➖ partial — `num_preempted` is the cumulative preemption count; no explicit per-request status |
| Token counts? | ✅ yes — `num_prompt_tokens`, `num_predicted_output_tokens`, `actual_output_tokens` (derived) |
| Cache / reuse / residency? | ✅ yes — `kv_cache_utilization`, `kv_free_blocks`, `kv_evictions_per_s` |
| Replica / autoscaling / GPU util? | ➖ no GPU utilization gauge or replica count; the instance count is implicit (one row per request per instance) |
| Constraint-aware replay supported? | ✅ yes — request-level arrival + completion timestamps + queue state are sufficient for bounded replay |
| Dynamic frontier calibration supported? | ➖ partially — queue state + measured latency are present, but full frontier calibration also needs measured GPU utilisation and replica scale; analysis sample size is currently `moderate` (9,605 rows in test_flat), which is **insufficient** for the `dynamic_calibration` promotion gate (requires `strong` = 10k+ rows). Re-running the audit against `train.jsonl` (359k rows, 392 MB) would clear this gate. |
| Priors supported? | ✅ yes — latency / throughput / queue priors are all directly computable |
| Fields missing for Tier 1 pilot equivalence? | GPU utilisation (DCGM-style), replica scale-out / scale-in events, autoscaler decisions, per-request SLA labels, network / region context, energy / power telemetry, retry / timeout outcomes |

**Classification:** `partial_serving_telemetry` →
`canonical_trace_type = telemetry_trace`,
`trust_tier = tier_2_public_telemetry_traces`. **NOT** Tier 1 (CloudLab
research cluster ≠ production pilot).

**Scoring:**

| metric | score |
|---|---|
| telemetry_richness_score | 8 / 10 |
| production_similarity_score | 6 / 10 (research cluster, only Qwen2.5, no GPU util / autoscaling) |
| constraint_aware_value_score | 8 / 10 |
| dynamic_calibration_value_score | 7 / 10 (limited by missing GPU util + replica scale) |
| cache_residency_value_score | 6 / 10 (kv-utilisation + evictions present, no prefix-id) |
| overall_priority_score | **7.5 / 10** (highest among HF datasets we've audited) |

### B. SwissAI (`eth-easl/swissai-serving-trace`)

| Question | Answer |
|---|---|
| Real request arrivals? | ✅ yes — `created_at` (ISO-8601) |
| Real completions? | ✅ yes — `finished_at` (ISO-8601) |
| Actual measured latency? | ➖ derived only — `latency = finished_at - created_at`; no TTFT / TPOT / e2e split |
| Queue wait / queue state? | ❌ no |
| Scheduler / routing state? | ❌ no |
| Model / server / GPU identity? | ➖ model only (Qwen3-32B, Llama3-70B, Apertus-70B, Qwen3-80B-instruct/thinking) — no GPU, no replica, no instance id |
| Status / error / timeout labels? | ✅ yes — `status` (e.g. DEFAULT, ERROR) per request |
| Token counts? | ➖ `reported_token_input` / `reported_token_output` are frequently **-1 (unavailable)** — high missing rate in head-sample |
| Cache / reuse / residency? | ✅ yes — `bucket_ids` (16-token deterministic buckets, Qwen3-32B-tokenized) + per-request `reuse_percentage = reused_buckets / total_buckets` in the bucket-reuse files |
| Replica / autoscaling / GPU util? | ❌ no |
| Constraint-aware replay supported? | ➖ workload-shape replay only — no queue / GPU / per-token latency to drive a scheduler decision |
| Dynamic frontier calibration supported? | ❌ no — no queue depth, no GPU utilisation, no per-tick replica count |
| Priors supported? | ✅ workload shape, derived e2e latency distribution, cache hit / bucket-reuse priors |
| Fields missing for Tier 1 pilot equivalence? | TTFT / TPOT, queue state, GPU type / utilisation, autoscaler events, replica count, real per-request token counts (most rows are -1) |

**Classification per config:**

- `trace` → `request_shape_trace` (Tier 5, `tier_5_request_shape_traces`)
- `qwen3-32b-buckets` → `cache_residency_trace` (Tier 4)
- `qwen3-32b-bucket-reuse` → `cache_residency_trace` (Tier 4)

**SwissAI is `request_service_trace` + `cache_residency_trace`. NOT
`full_serving_telemetry`.**

**Scoring:**

| metric | score |
|---|---|
| telemetry_richness_score | 3 / 10 |
| production_similarity_score | 7 / 10 (real production traffic at SwissAI / DataSys) |
| constraint_aware_value_score | 3 / 10 |
| dynamic_calibration_value_score | 1 / 10 |
| cache_residency_value_score | 8 / 10 (bucket-reuse percentages are exactly the cache-residency signal Aurelius needs) |
| overall_priority_score | **5.0 / 10** (high cache-residency value; low everywhere else) |

## 4. Statistical sample size policy

| (dataset, config) | sampling_method | fixture_rows | analysis_rows | strength | stratification_keys | weakest subgroup p99 status |
|---|---|---|---|---|---|---|
| CARA / test_flat | stratified | 5 | 9,605 | `moderate` | `[instance_type]` | all 5 instance_type subgroups have ≥1,000 rows; no `INSUFFICIENT_SAMPLE_P99` flags |
| CARA / test_queue_details | stratified | 5 | 4,876 | `moderate` | `[instance_type]` | same |
| SwissAI / trace | head | 5 | 25,409 | `strong` | `[model_id, status]` | reported_token counts are -1 in most rows; latency derivable from timestamps |
| SwissAI / qwen3-32b-buckets | head | 5 | 19,130 | `strong` | `[model_id, status]` | one model only |
| SwissAI / qwen3-32b-bucket-reuse | head | 5 | 16,593 | `strong` | `[]` | reuse_percentage distribution well-populated |

The `moderate` strength of CARA test_flat is enough to clear the
`constraint_aware_evaluation` + `backtest` promotion gates but
**deliberately not enough** for `dynamic_calibration` — the gate
(see `aurelius/traces/hf_corpus/promotion.py::PROMOTION_TAG_MIN_SAMPLE_STRENGTH`)
requires `strong` = ≥10,000 rows. The audit honestly downgrades and
records the reason in the registry entry.

CARA `train.jsonl` (392 MB, 359k rows) would unlock `strong` strength
and the dynamic-calibration tag in a follow-up audit; that ingest is
deliberately deferred from this PR because the analysis sample would
exceed the per-file 10 MiB bounded-download budget by ~40×.

## 5. Bounded ingest results + promotion decisions

| (dataset, config) | trace_type | trust | sample strength | promotion state | promotion tags |
|---|---|---|---|---|---|
| CARA / test_flat | telemetry_trace | Tier 2 | moderate | `promoted_for_constraint_aware_evaluation` | `[constraint_aware_evaluation, backtest]` (dynamic_calibration **downgraded** — requires strong) |
| CARA / test_queue_details | telemetry_trace | Tier 2 | moderate | `promoted_for_constraint_aware_evaluation` | `[constraint_aware_evaluation, backtest]` (dynamic_calibration downgraded) |
| SwissAI / trace | request_shape_trace | Tier 5 | strong | `promoted_for_training_priors` | `[training_priors]` |
| SwissAI / qwen3_32b_buckets | cache_residency_trace | Tier 4 | strong | `promoted_for_cache_residency_evaluation` | `[cache_residency_evaluation, training_priors]` |
| SwissAI / qwen3_32b_bucket_reuse | cache_residency_trace | Tier 4 | strong | `promoted_for_cache_residency_evaluation` | `[cache_residency_evaluation, training_priors]` |

All 5 audited units pass all 9 promotion gates (schema, fixture,
bounded_size, license_and_gating, canonical_trace_type, signals,
limitations, at_least_one_use_case, analysis_sample_policy).

## 6. Alpha opportunity assessment

### CARA → constraint-aware + future dynamic calibration

The per-subgroup p99 latency surface from the 9,605-row analysis sample
demonstrates a **9× p99 spread** for the same Qwen2.5-3B model across
two GPU types — exactly the kind of placement-quality signal Aurelius'
constraint-aware engine consumes:

| instance_type | n | p50 (s) | p95 (s) | p99 (s) |
|---|---:|---:|---:|---:|
| `qwen2.5-3b_a30` | 1,666 | 1.64 | 6.36 | **9.38** |
| `qwen2.5-3b_p100` | 2,666 | 14.90 | 52.47 | **83.33** |
| `qwen2.5-7b_a30` | 2,665 | 5.43 | 15.27 | 19.37 |
| `qwen2.5-14b_v100` | 1,531 | 2.20 | 10.97 | 16.02 |
| `qwen2.5-72b_a100` | 1,077 | 8.03 | 28.07 | 43.26 |

**Direct Aurelius decisions this can improve:**

- **Placement quality** — the priors above quantify per-(model, GPU) tail
  latency, the exact input to the placement scorer.
- **Latency / queue / timeout risk** — `num_running`, `num_waiting`,
  `kv_cache_utilization`, `pending_prefill_tokens`, EMA decode/prefill
  throughput give the at-decision-time queue state.
- **Routing quality** — the same request (token shape, model) routed to
  different `instance_type`s lets us fit a routing prior.
- **Cache / residency** — `kv_cache_utilization` + `kv_evictions_per_s`
  are the eviction signals Aurelius' residency engine could shadow-fit.

**Modules that could consume CARA:**

- `aurelius/frontier/dynamic_calibration.py` — when the audit is re-run
  on `train.jsonl` for `strong` strength.
- `aurelius/frontier/dynamic_estimator.py` — for at-decision-time queue
  / KV utilisation features.
- `aurelius/constraints/frontier_integration.py` — placement scorer
  priors.

**What evaluation should be run:** a bounded per-(instance_type) latency
prior fit + a routing-prior smoke evaluation. **Not** a full
constraint-aware backtest until `train.jsonl` is ingested. **Not** a
production-claim run — CARA is CloudLab research, not pilot telemetry.

**Expected maximum upside:** **high** for placement / routing priors;
**medium** for dynamic frontier calibration (limited by missing GPU
utilisation + replica count); **none** for any production-savings claim
(Tier 2 ≠ Tier 1).

### SwissAI → cache-residency + workload-shape priors

The bucket-reuse files give per-request `reuse_percentage = reused / total`
over deterministic 16-token buckets. Sample stats from the bounded head:

- Mean reuse_percentage in the audited bucket-reuse head was low
  (most early entries showed 0% reuse), but the file extends to ~3.7 GB,
  so a `strong` sample from a follow-up bounded ingest of the larger
  per-model files will surface the long-tail reuse distribution.

**Direct Aurelius decisions this can improve:**

- **Residency / cache** — the bucket-reuse signal is exactly the
  cache-hit prior Aurelius' residency engine uses.
- **Routing (affinity)** — when the same input bucket-id appears across
  requests we can prior the affinity-routing gain.
- **Workload shape** — `trace.jsonl` is a high-fidelity arrival pattern
  for Qwen3-32B / Llama3-70B / Apertus-70B / Qwen3-80B production
  traffic.

**Modules:** `aurelius/residency/`, request-shape replay layer in
`aurelius/traces/replay.py`.

**Evaluation:** a bounded cache-hit-rate distribution + per-request
reuse-percentage histogram. **Not** a constraint-aware backtest (no
queue / latency / GPU state). **Not** a dynamic-frontier-calibration run
(no Tier 2 telemetry signals).

**Expected maximum upside:** **medium-high** for cache / residency
priors; **low** for everything else.

## 7. Bounded evaluations run

This PR runs the routed smoke evaluators **only via the existing
`scripts/run_hf_corpus_evaluations.py` harness**, which respects the
trace-type → evaluator → required-signals routing. CARA is routed to
`telemetry_calibration_smoke_v1` (queue / GPU util / SLA / queue_wait
signal-gated; CARA has `queue_depth` ✓). SwissAI bucket-reuse is routed
to `cache_residency_prior_smoke_v1`. SwissAI `trace` falls back to
`request_shape_prior_smoke_v1`.

No new evaluator is introduced in this PR. Production-feasible bounded
evaluation of the CARA priors against `aurelius/frontier/dynamic_*` is
the documented next task once the analysis sample reaches `strong`
strength.

## 8. Promotion decisions, rejections + deferrals

- **Promoted (5/5):** all 5 audited units cleared every gate. CARA's
  dynamic_calibration tag was downgraded because the head-sample only
  reaches `moderate` strength; this is recorded in
  `decision.reasons` and surfaced in the registry entry.
- **Deferred (5):** SwissAI `apertus-70b-bucket-reuse.jsonl`,
  `llama3-70b_bucket-reuse.jsonl`, `qwen380b_instruct_bucket-reuse.jsonl`,
  `qwen380b_thinking_bucket-reuse.jsonl`, `data.jsonl` — same schema as
  already-audited files; ingest in a follow-up only if a per-model
  reuse-percentage breakdown is needed. SwissAI `datasys_trace.py` is a
  Python loader, not a data file.
- **Deferred (CARA train splits):** `train.jsonl` (392 MB) +
  `train_queue_details.jsonl` (812 MB) — ingesting these would unlock
  CARA's `promoted_for_dynamic_calibration` tag. Deferred because of the
  10 MiB-per-file bounded-download budget in this audit script.

## 9. Tests

- 38 new tests in `tests/test_hf_cara_swissai_audit.py` covering: schema
  profile (flat + nested + lists + sentinels), schema mapping
  classification, per-subgroup insufficient-sample flagging,
  stratification, extended canonical record schemas, ingestion
  mappings, sample-strength enforcement (strong / moderate / weak /
  fixture_only), auth-blocked short-circuit, audit-artefact paths,
  raw-file gitignore enforcement, canonical registry round-trip.
- Existing 71 HF tests still pass after the schema / promotion-state
  extensions.

## 10. Next actions (documented for the next run)

- Run the audit script against `train.jsonl` + `train_queue_details.jsonl`
  with a per-file budget of 50-100 MiB to push CARA from `moderate` →
  `strong` strength + unlock `promoted_for_dynamic_calibration`.
- Add a `telemetry_calibration_smoke_v1` variant that consumes CARA's
  measured TTFT / TPOT / e2e + at-decision-time queue state and
  produces a placement-prior surface (without claiming production
  savings).
- Extend the audit to the other SwissAI per-model bucket-reuse files
  (`llama3-70b`, `apertus-70b`, `qwen380b_instruct`,
  `qwen380b_thinking`) only if per-model residency priors are required.
