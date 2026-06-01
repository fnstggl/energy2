# Aurelius Telemetry Gap Discovery — public-dataset audit

> **Audit-only PR. No forecaster is trained. No scheduler is modified.
> No production claim is made.** This is a discovery + signal-coverage
> audit against the remaining gaps in
> `docs/FORECAST_LEVERAGE_AUDIT.md` and
> `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md`. Pilot telemetry (Tier 1)
> remains the only production-equivalent source.
>
> **Read first:**
> - `docs/FORECAST_LEVERAGE_AUDIT.md` (which forecasts are gated on what data)
> - `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` (Tier-2 baseline already ingested)
> - `docs/HF_DATASET_REGISTRY.md` (trust hierarchy + canonical trace types)
> - `data/external/hf_discovery/aurelius_gap_closure_audit.json`
>   (machine-readable signal × dataset × forecast matrix)

## ⓘ Update 2026-06-01 — telemetry-gap ingest landed

5 of the top-10 ingest-now datasets have now been bounded-ingested via
`scripts/ingest_hf_gap_datasets.py` (audit-only; raw + analysis_sample
gitignored, only schema_profile + schema_mapping + summary +
statistical_rollups + tiny fixtures committed). Cross-dataset rollup:
`data/external/hf_discovery/telemetry_gap_ingest_summary.json`.
Tests: `tests/test_hf_gap_ingest.py` (35 tests, all green).

| Dataset | Bytes sampled | Rows | Strength | Promotion state | Key signals unlocked |
|---|---:|---:|---|---|---|
| `semianalysisai/cc-traces-weka-no-subagents-051226` | 80 MiB (raw) / 0.4 MB (normalized) | 761 | weak | `promoted_for_training_priors` | `kv_block_hashes`, `migration_or_cache_loss_proxy`, `prefix_reuse`, `ttft`, `latency` |
| `sammshen/lmcache-agentic-traces` | 398 MB (one shard) / 0.7 MB (normalized) | 4,976 | moderate | `promoted_for_training_priors` | `routing_proxy`, `cache_reuse`, `prefix_reuse`, `arrivals` |
| `lzzmm/BurstGPT` | 20 MiB (raw) / 8.0 MB (normalized) | 59,999 | strong | `promoted_for_training_priors` | `arrivals`, `capacity_proxy`, `autoscaling_proxy`, `customer_traffic_mix` |
| `lsliwko/google-cluster-data-2019-sorted-by-timestamp` | 53 MB (one gz shard) / 21 MB (normalized) | 60,000 | strong | `promoted_for_backtest` + `promoted_for_constraint_aware_evaluation` | `autoscaling_proxy`, `model_load_event`, `model_unload_event`, `migration_or_cache_loss_proxy`, `routing_proxy` |
| `jaytonde05/prefixbench` | 80 MB (full, 4 jsonl) / 1.8 MB (normalized) | 4,000 | moderate | `promoted_for_cache_residency_evaluation` + `promoted_for_training_priors` | `cache_reuse`, `prefix_reuse` |

**Net signal closures from this ingest** (signals that were NONE or
WEAK before, MODERATE+ now):

| Signal | Before | After ingest | Source |
|---|---|---|---|
| kv_block_hashes | NONE | MODERATE | CC-traces (per-request hash list) |
| migration_or_cache_loss_proxy | NONE | MODERATE | CC-traces + Google Cluster |
| model_load_event | NONE | MODERATE (Borg) | Google Cluster SCHEDULE |
| model_unload_event | NONE | MODERATE (Borg) | Google Cluster EVICT/KILL/FAIL/FINISH |
| autoscaling_proxy | NONE | MODERATE | BurstGPT + Google Cluster |
| capacity_proxy | WEAK | STRONG | BurstGPT (strong) + Google Cluster (strong) |
| arrival_patterns (additive) | STRONG | STRONG+ | + BurstGPT + CC-traces + LMCache + Google Cluster |
| customer_traffic_mix | NONE | MODERATE | BurstGPT (Conversation vs API Log Type) |
| routing_proxy | NONE | MODERATE | LMCache (session_id) + Google Cluster (machine_id) |
| cache_reuse (additive) | STRONG | STRONG+ | + CC-traces + LMCache + PrefixBench |
| prefix_reuse (additive) | STRONG | STRONG+ | + CC-traces + LMCache + PrefixBench |

**Per-dataset worth-ingesting verdict:**

- ✅ `lzzmm/BurstGPT` — strong-strength real Azure arrivals. Closes
  capacity + autoscaling proxies. Worth ingesting.
- ✅ `lsliwko/google-cluster-data-2019` — strong-strength Borg events;
  the only public source that gives autoscaling / migration / model-
  load / model-unload labels (job-level Borg, not LLM serving — still
  a useful PROXY). Highest gap-closing value. Worth ingesting.
- ✅ `sammshen/lmcache-agentic-traces` — moderate-strength curated
  agentic sessions. Worth ingesting.
- ✅ `jaytonde05/prefixbench` — moderate synthetic prefix corpus. Worth
  ingesting for replay-driven cache evaluation.
- ⚠️ `semianalysisai/cc-traces-weka-no-subagents-051226` — only weak
  strength at the 80 MiB head because each session is ~12 MB. The
  signal value (KV block hashes!) is uniquely high but the strength
  is below the `moderate` threshold required for
  `promoted_for_cache_residency_evaluation`. A follow-up should
  bounded-ingest a larger window (e.g. 300 MiB head ≈ 25 sessions =
  ~2,700 requests = moderate) OR sample session-level then expand
  per session-row to per-request rows with a turn cap to maximise
  session diversity per byte. Currently promoted only for training
  priors. Worth ingesting; needs strength expansion in next PR.

**Gaps that remain irreducibly pilot-only** — unchanged from §7
below. The 12 signals listed in §7 still cannot be sourced from any
HF dataset and require Tier 1 customer telemetry.

---

## 0. Scope, evidence trail + safety

- This document is the human-readable rollup of
  `data/external/hf_discovery/aurelius_gap_closure_audit.json`. Every
  number below has a key in that JSON.
- The audit used the HuggingFace API exclusively (search, dataset_info,
  card download, sample-row peek for files < 50 MB). No new dataset
  was downloaded into the repository — only schema + small-head probes
  on `/tmp`. No artefacts under `data/external/hf/*/raw/` were
  produced.
- HF_TOKEN was consumed via environment variable only. No token text
  is recorded in any committed artefact.
- **No scheduler, controller, frontier engine, residency engine, or
  robust energy engine was touched.**

## 1. Headline

| Question | Answer |
|---|---|
| Total HF candidates discovered across 175 search terms | **466** |
| Candidates deep-inspected (schema + README + sample row) | **40** |
| Candidates with ≥1 telemetry-gap signal match | **59** |
| Candidates recommended for ingest (NOW + LATER) | **15** |
| Gaps newly closed at STRONG strength | 3 (arrival, cache_reuse, prefix_reuse) |
| Gaps newly closed at MODERATE strength | 5 (autoscaling, capacity, migration, routing, thermal-as-CFD) |
| Gaps still WEAK / NONE after full ingest | **12** (all are pilot-telemetry-only) |
| Search-quality status | **plateaued** — round 3 dominated by term-match false positives |

## 2. Coverage update vs the gap list

Current strong coverage (from prior audits, unchanged):

> TTFT, TPOT, E2E latency, queue depth, queue state, waiting/running
> requests, KV utilization, cache reuse, prefix reuse, request
> timestamps, GPU type, workload shape, arrival patterns.

What this audit changes for the 20 remaining gaps:

| Gap | Before | After proposed ingest | New evidence |
|---|---|---|---|
| replica counts | NONE | NONE | — |
| autoscaling events | NONE | **MODERATE (job-level)** | Google Cluster 2019 collection + instance events |
| production SLA labels | NONE | NONE | — (synthetic only) |
| timeout labels | NONE | NONE | — (synthetic only) |
| customer traffic mix | NONE | **MODERATE** | BurstGPT (Conversation vs API Log Type) + CC-traces (Claude Code session segments) |
| multi-region routing | NONE | NONE | — |
| cloud cost telemetry | NONE | WEAK | tarekmasryo synthetic + MCP-1st-Birthday synthetic |
| GPU utilization | NONE | NONE | — |
| GPU memory utilization | NONE | WEAK | DGX Spark `rss_gb` (30 rows) |
| thermal telemetry | NONE | WEAK | PhysicsNeMo CFD simulation + ClarusC64 synthetic |
| power telemetry | NONE | WEAK | ClarusC64 synthetic |
| model load events | NONE | **MODERATE** | Google Cluster 2019 task SCHEDULE events |
| model unload events | NONE | **MODERATE** | Google Cluster 2019 EVICT/KILL/FAIL/FINISH |
| cold-start latency | NONE | NONE | — |
| prewarm events | NONE | NONE | — |
| migration outcomes | NONE | **MODERATE** | CC-traces KV hashes (cache-loss proxy) + Google Cluster 2019 evict events |
| cross-region placement | NONE | WEAK | Google Cluster 2019 (anonymized machine cells) |
| serving fleet inventory | NONE | **MODERATE** | Google Cluster 2019 machine events |
| cluster scaling decisions | NONE | **MODERATE** | Google Cluster 2019 collection capacity changes |
| infrastructure scheduling signals | NONE | **MODERATE** | Google Cluster 2019 scheduling_class + priority + alloc_collection |

Of the 20 gaps, ingestion of the proposed 10 datasets meaningfully
moves **8** to MODERATE+. The other 12 remain irreducibly Tier-1
(pilot telemetry only) — see §7.

## 3. Top 25 datasets discovered

Ranked by Aurelius usefulness score (0-10) and gap-closure impact.

| # | Dataset | License | Size | Trust | Score | Best gap closed |
|---|---|---|---:|---|---:|---|
| 1 | `asdwb/cara_latency_prediction` | apache-2.0 | 1.3 GB | tier 2 | 9.0 | TTFT/TPOT/queue/E2E (already ingested) |
| 2 | `semianalysisai/cc-traces-weka-no-subagents-051226` | apache-2.0 | 2.8 GB | tier 3 | 9.0 | **cache + arrival + migration** |
| 3 | `agent-perf-bench/AgentPerfBench` | apache-2.0 | 1.5 MB | tier 2 | 7.0 | heterogeneous placement (already ingested) |
| 4 | `sammshen/lmcache-agentic-traces` | mit | 2.4 GB | tier 3 | 8.5 | **cache residency (agentic)** |
| 5 | `lzzmm/BurstGPT` | LICENSE file | 392 MB | tier 3 | 7.5 | **arrival / capacity / autoscaling** |
| 6 | `lsliwko/google-cluster-data-2019-sorted-by-timestamp` | derived | 117 GB | tier 3 | 8.0 | **autoscaling / migration / fleet inventory** |
| 7 | `eth-easl/swissai-serving-trace` | other | 21 GB | tier 3-5 | 6.5 | cache + arrival (already ingested) |
| 8 | `jaytonde05/prefixbench` | unspecified | 80 MB | tier 4 | 5.5 | cache residency (synthetic prefixes) |
| 9 | `Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days...` | unspecified | 7 MB | tier 3 | 5.0 | arrival + prefix replay |
| 10 | `hlarcher/inference-benchmarker` | apache-2.0 | 466 MB | tier 5 | 5.0 | workload corpus |
| 11 | `odyn-network/odyn-benchmarks` | apache-2.0 | 8 MB | tier 5 | 5.0 | vLLM + Ray Serve profiles |
| 12 | `project-vajra/dev-staging-h100-dgx` (+ 80 peers) | unspecified | ~500 MB total | tier 5 | 4.5 | per-(GPU SKU × model × kernel) priors |
| 13 | `memoriant/dgx-spark-kv-cache-benchmark` | apache-2.0 | 13 KB | tier 5 | 3.5 | DGX Spark GB10 hardware coverage |
| 14 | `Nathan-Maine/dgx-spark-kv-cache-benchmark` | apache-2.0 | 13 KB | tier 5 | 3.5 | duplicate of #13 |
| 15 | `Boxoffice1280/Neurips2026_evaluating_accuracy_KV-cache_reuse_techniques` | cc-by-nc-nd-4.0 | 2.0 GB | tier 4 | 3.5 | KV-cache reuse benchmark (license blocks redistribution) |
| 16 | `Alexsssu/BurstGPT_Compressed_Files` | unspecified | 306 MB | tier 3 | 4.0 | duplicate BurstGPT subset (archives) |
| 17 | `rbgo/llm-inference-benchmark` | unspecified | 91 KB | tier 5 | 3.0 | cross-library TTFT/TPS priors |
| 18 | `nvidia/PhysicsNeMo-Datacenter-CFD` | apache-2.0 | 187 GB | tier 5 | 2.5 | hot-aisle thermal CFD (simulation only) |
| 19 | `ClarusC64/datacenter-job-queue-resource-coherence-risk-v0.1` | mit | 9 KB | tier 6 | 2.0 | synthetic SLO-breach eval |
| 20 | `ClarusC64/datacenter-thermal-performance-coherence-risk-v0.1` | mit | 9 KB | tier 6 | 2.0 | synthetic thermal eval |
| 21 | `ClarusC64/datacenter-power-load-coherence-risk-v0.1` | mit | 9 KB | tier 6 | 2.0 | synthetic power eval |
| 22 | `ClarusC64/datacenter-node-failure-cascade-risk-v0.1` | mit | 9 KB | tier 6 | 1.5 | synthetic fleet-failure eval |
| 23 | `ClarusC64/datacenter-water-cooling-demand-coherence-risk-v0.1` | mit | 9 KB | tier 6 | 1.5 | synthetic cooling eval |
| 24 | `tarekmasryo/llm-system-ops-production-telemetry-sft-data` | cc-by-4.0 | 27 MB | tier 6 | 3.0 | synthetic LLMOps multi-table |
| 25 | `spiritbuun/turboquant-tcq-kv-cache` | apache-2.0 | 605 KB | tier 5 | 2.0 | TCQ KV codebooks (research artifact) |

The full per-dataset record with `signals_present`, `signals_missing`,
`field_names_sample`, `forecast_support`, and trust-tier rationale is
in `data/external/hf_discovery/aurelius_gap_closure_audit.json` under
`datasets[]`.

## 4. Top 10 datasets to ingest immediately

| Order | Dataset | Why | Ingest budget |
|---|---|---|---|
| 1 | `lzzmm/BurstGPT` | Largest pure-arrival production trace (Microsoft Azure). Has retry / failure splits. | full (390 MB) |
| 2 | `semianalysisai/cc-traces-weka-no-subagents-051226` | Real Claude Code production traffic with per-request KV block hashes. Replayable. | bounded 50-100 MiB |
| 3 | `sammshen/lmcache-agentic-traces` | 787 multi-turn agentic sessions, ≥10K context. | bounded 50-100 MiB |
| 4 | `lsliwko/google-cluster-data-2019-sorted-by-timestamp` | Only public dataset with SUBMIT/SCHEDULE/EVICT/FAIL/FINISH/KILL lifecycle. | bounded sample (~1 GB) |
| 5 | `jaytonde05/prefixbench` | Adds eviction_pressure + multiturn_agent_branching for replay-driven cache eval. | full (80 MB) |
| 6 | `Alexsssu/BurstGPT_LMSYSChat_withPrompt_2Days...` | Enables joint arrival + prefix replay studies. | full (7 MB) |
| 7 | `hlarcher/inference-benchmarker` | Third-party workload corpus for cross-validation. | bounded 1 split |
| 8 | `odyn-network/odyn-benchmarks` | vLLM + Ray Serve prompt profiles. | full (8 MB) |
| 9 | `project-vajra/* (selective subset)` | Per-(GPU SKU × model × kernel) latency profiles for H100/A100/H200. | ~50 MB selective |
| 10 | `memoriant/dgx-spark-kv-cache-benchmark` | Rare DGX Spark GB10 unified-memory benchmark. | full (13 KB) |

Detailed ingest order + rationale is encoded in
`aurelius_gap_closure_audit.json::recommended_ingest_order`.

## 5. Gaps after proposed ingest

### 5.1 Newly fully covered (STRONG)

These move from `WEAK / NONE` to `STRONG` after ingesting the top 10:

| Gap | New strength | Source |
|---|---|---|
| arrival_patterns | STRONG | BurstGPT + CC-traces + LMCache agentic + Google Cluster 2019 |
| prefix_reuse | STRONG | CC-traces (KV hashes) + LMCache agentic + SwissAI |
| cache_reuse | STRONG | CC-traces + LMCache agentic + SwissAI |
| capacity_forecast | STRONG | BurstGPT + Google Cluster 2019 + LMCache agentic |

### 5.2 Partially closed (MODERATE)

| Gap | New strength | Source / caveat |
|---|---|---|
| autoscaling | MODERATE | Google Cluster 2019 events are **job-level Borg**, not GPU-replica. The loop arrival→replica→latency requires pilot telemetry. |
| migration_forecast | MODERATE | CC-traces give cache-loss proxy via KV hashes; Google Cluster 2019 gives evict events. |
| routing_forecast | MODERATE | CC-traces + LMCache + CARA instance routing (no live routing telemetry). |
| model_load_events | MODERATE | Google Cluster 2019 SCHEDULE — Borg semantics ≠ LLM model load. |
| model_unload_events | MODERATE | Google Cluster 2019 EVICT/KILL/FAIL/FINISH — same caveat. |
| serving_fleet_inventory | MODERATE | Google Cluster 2019 machine events — Borg, not LLM serving. |
| cluster_scaling_decisions | MODERATE | Google Cluster 2019 — Borg. |
| customer_traffic_mix | MODERATE | BurstGPT Log Type + CC-traces session segments. |
| thermal_forecast | MODERATE | PhysicsNeMo CFD (offline only — needs DCGM at runtime). |

### 5.3 Still WEAK / NONE (irreducibly pilot-only)

The following gaps **cannot** be closed by any HF dataset surveyed and
require Tier 1 pilot telemetry:

| Gap | Why HF cannot close it |
|---|---|
| production_SLA_labels | No HF dataset has measured per-request deadline + outcome. CARA has `num_preempted`; SwissAI has `status`; both are proxies. |
| timeout_labels | No measured per-request timeout outcome. |
| replica_counts (LLM serving) | No HF dataset records per-tick vLLM replica scale-out / scale-in. |
| autoscaling_events (LLM serving) | Google Cluster 2019 is Borg-level; LLM-serving HPA events are not on HF. |
| GPU_utilization (DCGM SM-busy %) | Absent from every HF dataset. |
| GPU_memory_utilization (DCGM VRAM) | Only DGX Spark `rss_gb` (30 rows) — insufficient. |
| thermal_telemetry (DCGM live) | Only CFD simulation; no measured T_core / T_HBM. |
| power_telemetry (DCGM live) | Only synthetic ClarusC64 power eval. |
| cold_start_latency | No HF dataset measures model load start → first inference ready. |
| prewarm_events | No HF dataset records preloaded-but-idle replica state. |
| migration_outcome (full) | Only cache-loss proxy via CC-traces KV hashes; no measured latency_blip. |
| cloud_cost_telemetry | Only synthetic; pilot billing required. |
| cross-region routing decision | Anonymized in Google Cluster 2019; LLM-serving multi-region routing not on HF. |

## 6. Estimated forecasting coverage after ingestion

| Forecast | Before this audit | After proposed ingest |
|---|---|---|
| queue_forecast | STRONG | STRONG (unchanged) |
| ttft_forecast | STRONG | STRONG (unchanged) |
| tpot_forecast | STRONG | STRONG (unchanged) |
| e2e_forecast | STRONG | STRONG (unchanged) |
| gpu_placement | STRONG | STRONG (CARA + AgentPerfBench + Vajra) |
| cache_residency | STRONG | **STRONG+** (CC-traces + LMCache + PrefixBench + SwissAI) |
| cold_start | NONE | WEAK (proxy via Google Cluster) — **pilot still required** |
| sla_risk | WEAK | WEAK — **pilot still required** |
| timeout_risk | WEAK | WEAK — **pilot still required** |
| autoscaling | NONE | MODERATE (BurstGPT + Google Cluster) |
| capacity_forecast | WEAK | STRONG |
| arrival_forecast | STRONG | STRONG+ |
| thermal_forecast | NONE | WEAK (CFD only) — **pilot DCGM required** |
| cost_forecast | NONE | WEAK (synthetic only) — **pilot billing required** |
| migration_forecast | NONE | MODERATE (CC-traces + Google Cluster) |
| routing_forecast | NONE | MODERATE (CC-traces + CARA + LMCache) |

## 7. Exact signals still requiring real customer telemetry

This list is binding for the
`docs/PILOT_TELEMETRY_CONTRACT.md` / `docs/PILOT_READINESS_AUDIT.md`
workstream — these signals **cannot** be delivered by any HF
dataset surveyed and must come from pilot telemetry:

1. Per-request `production_SLA_label` (deadline budget + outcome).
2. Per-request `timeout_label` (timed_out vs completed_within_budget).
3. vLLM/SGLang serving `replica_count` time series.
4. Autoscaler event log (`scale_up` / `scale_down` + rationale).
5. DCGM `GPU_utilization` (SM busy %) time series.
6. DCGM `GPU_memory_utilization` (per-process VRAM used) time series.
7. DCGM thermal sensor time series (T_core, T_HBM).
8. DCGM `power_draw` time series (W per GPU).
9. `model_load_event` (start, complete, bytes from disk / network).
10. `model_unload_event` (start, complete).
11. `cold_start_latency` (model load start → first inference ready).
12. `prewarm_event` (model preloaded but not yet serving).
13. `migration_outcome` (from-replica, to-replica, cache_loss_pct,
    latency_blip).
14. Cross-region routing decision (request → region/zone chosen +
    alternatives).
15. Customer / tenant `traffic_mix` labels (per-tenant SLA tier).
16. Per-request `cloud_cost` telemetry (per-region $/hr × utilization).

## 8. Recommended next ingestion order

The order in §4 is the recommended ingest sequence. For each, the
ingestion plan should:

- Honour the per-file 10-50 MiB bounded-download budget already
  established in `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` §5.
- Land schema_profile + schema_mapping + statistical_rollups under
  `data/external/hf/<author>__<dataset>/<config>/processed/` exactly
  as CARA + SwissAI do today.
- Add a `canonical_corpus_registry.json` entry with the right
  `trust_tier`, `canonical_trace_type`, and `promotion_tags` from
  `aurelius/traces/hf_corpus/promotion.py`.
- For redistribution-restrictive licenses (Boxoffice1280
  cc-by-nc-nd-4.0; SwissAI "other"), record `_license_restriction`
  metadata before any analysis-tier expansion.

## 9. Search-quality plateau evidence

The audit stopped expanding the search universe after round 3 because:

- **Round 1** (64 broad terms) surfaced 226 candidates and 59 with
  signal matches.
- **Round 2** (60 narrower terms) added 240 new candidates but the
  *signal-relevant* additions plateaued — most additions were Borg /
  Philly / cluster term-match false positives (usernames, restaurants,
  city of Philadelphia).
- **Round 3** (39 targeted terms) returned only 9 new datasets worth
  deep-inspecting, and 5 of those were ClarusC64 synthetic evals (the
  same author).
- Direct probes of 40 likely-existing IDs (Azure, Alibaba GPU, Philly,
  Google Cluster 2011, Splitwise, Vidur, MIT Supercloud) **all
  returned NotFound** — confirming that the canonical infrastructure
  traces remain hosted off-HF (GitHub releases, FigShare, Zenodo,
  authors' webpages).
- Searches for `dcgm`, `nvidia smi`, `gpu utilization`, `prometheus
  metrics`, `azure trace`, `philly trace`, `alibaba cluster`,
  `supercloud`, `borg`, `cluster-trace`, and `node trace` all returned
  **zero** telemetry-relevant hits.

The pattern is clear: HuggingFace hosts (a) LLM benchmark + workload
shape data well, (b) some real serving traces (CARA, SwissAI,
CC-traces), and (c) a single legacy cluster trace (Google 2019 mirror).
It does **not** host DCGM exports, replica-count series, or pilot SLA
labels.

## 10. Non-goals + safety

- No forecaster is trained in this PR.
- No scheduler / controller / frontier engine / residency engine /
  robust energy engine code is touched.
- No HF dataset is treated as a Tier 1 pilot — every candidate is
  classified at most Tier 2-5 per `docs/HF_DATASET_REGISTRY.md`.
- No production-savings number is quoted.
- The 175 search terms + 40 deep inspections + 466 candidates are
  recorded in `/tmp/hf_audit/out/` during the audit run but only the
  consolidated `data/external/hf_discovery/aurelius_gap_closure_audit.json`
  is committed.
