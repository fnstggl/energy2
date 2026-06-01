# Frontier Discovery Audit v1 — Missing Economic-Alpha Signals

> **Discovery / audit-only PR.** No model trained, no production code
> modified, no data ingested, no savings claimed, no constants invented,
> no synthetic data used. `HF_TOKEN` read from env only, never committed.
>
> **Read first:** `docs/FRONTIER_SIGNAL_HYPOTHESES.md`,
> `docs/ECONOMIC_ML_ALPHA_V1.md`, `docs/ECONOMIC_OVERLAY_LAYER_V1.md`,
> `docs/HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md`.
>
> Evidence: `data/external/hf_discovery/frontier_v1/{frontier_dataset_registry,
> frontier_field_matrix, economic_frontier_priority_ranking}.json`.

## 1. Mission & method

Discover NEW operational signals/datasets that could unlock economic-alpha
forecasting beyond the already-validated **cache_reuse / TTFT / peak_VRAM**.
Method: metadata-only HF discovery — 62 search probes + 23 curated infra
seeds (org-enumerated), 399 unique datasets surfaced, **160 deep-inspected**
(card + README + siblings + features), classified across 6 frontier signal
categories + baseline ops/econ signals, scored on forecastability (0–100)
and economic relevance (Very High/High/Medium/Low/Reject).

`scripts/discover_frontier_signals.py` downloads no data and trains nothing.

## 2. Headline finding (the honest negative result)

**Public HF datasets do not contain the missing economic-alpha signals.**
Of 160 inspected datasets:

| Economic relevance | count |
|---|---:|
| High | **3** (all already ingested in prior PRs) |
| Low | 47 |
| Reject | 110 |
| Very High / new | **0** |

- The 3 "High" datasets are `ssong1/llmperf-bedrock` (TTFT/TPOT/throughput/
  error_rate — already ingested PR #137), and `memoriant/` +
  `Nathan-Maine/dgx-spark-kv-cache-benchmark` (KV memory curves — already
  ingested PR #134). **No new high-relevance economic-frontier dataset.**
- **42 datasets matched "cold-start" — ALL 42 are RL fine-tuning
  "cold-start" SFT/reasoning corpora** (msmarco, math8k, GRPO, multimodal
  reasoning), **zero are serving cold-start telemetry.** The term "cold
  start" on HF overwhelmingly means RL warm-up data, not startup latency.
- **autoscaling: 0 datasets.** No public HF dataset exposes scale-up/down,
  warm-pool, or oscillation telemetry.
- **migration: 1** (`eth-easl/swissai-serving-trace` — a reuse *proxy*,
  already ingested; not real migration labels).
- **queueing: 1** (`Qinghao/AcmeTrace` — real job-level `queue_wait`,
  already ingested).

## 3. New serving-systems datasets found (and why they don't unlock alpha)

The infra-seed enumeration *did* surface genuinely new serving-systems
datasets — but they are micro-benchmarks/profiling, not production
telemetry with the missing labels:

| Dataset | What it is | License | Verdict |
|---|---|---|---|
| `project-vajra/dev-staging-*` (9 SKUs) | NCCL collective profiling (`all_reduce.csv.xz`, `send_recv.csv.xz`) + prefill/decode compute profiling; Sarathi/Vajra lineage | None | topology/compute priors, **not** queue/migration/cold-start; `license=None` |
| `project-vajra/predictions-*` | model-fit predictions over the above | None | derived artifacts, not telemetry |
| `intellistream/sage-control-plane-{benchmark,workloads,llm-workloads}` | control-plane workload configs (1 small parquet, 7 rows) | None | workload specs, no per-request queue/migration telemetry |
| `intellistream/sagellm-benchmark-results` | serving leaderboard JSONs | None | aggregate results, no per-request trace |
| `Isabella5/sglang-seglen-benchmark` | SGLang eviction/prefix request-shape JSONs | None | request shapes (cache-reuse adjacent), no serving telemetry |
| `crozai/mbicanic/vllm-benchmark-coding` | vLLM coding-benchmark request set (~37k rows) | None | request shapes, no economic-frontier labels |
| `metrum-ai/llm-perfdata` | TTFT + throughput perf data | mit | small latency prior (Low); overlaps existing latency corpus |

These reinforce, not overturn, the conclusion: real serving-systems data
on HF is **micro-profiling and request-shape**, with `license=None`
dominant, and **none carries production queue / migration / cold-start /
autoscaling labels**.

## 4. Phase 4 — forecastability audit

`economic_frontier_priority_ranking.json`. Forecastability rewards real
(incl. compressed `.csv.xz`) data files + rows + frontier/ops signals,
and penalises gated / no-data / `license=None`. Top forecastable +
non-reject datasets are all **already-ingested** latency/memory corpora
(`llmperf-bedrock` 49, `dgx-spark-kv-cache` 27). Every genuinely new
serving dataset scored low because it exposes config-only cards or
request-shape JSON without the target labels.

Per-category public-data status (the actionable matrix):

| Frontier category | Best public signal today | Status |
|---|---|---|
| Queueing | AcmeTrace `queue_wait` (job-level), CARA queue features | **partial / trainable as proxy** |
| Memory pressure | Optimum `peak_vram` (realized, shadow-ready), CARA KV evictions | **partial / partly realized** |
| Serving stability | Optimum/Odyn/llmperf error_rate, AcmeTrace FAILED/TIMEOUT | **weak proxy, no per-request labels** |
| Cold start | none (RL "cold-start" datasets are unrelated) | **blocked_by_pilot_telemetry** |
| Migration | CC-traces KV-loss proxy (not realized) | **blocked_by_pilot_telemetry** |
| Autoscaling | none (Google-Cluster/Borg job-level proxy only) | **blocked_by_pilot_telemetry** |

## 5. Phase 5 — economic-impact ranking (hypothesis leverage)

Independent of availability (see `FRONTIER_SIGNAL_HYPOTHESES.md`):

- **Very High leverage:** queue-wait/admission (flips sla_met), timeout/
  failure risk (flips goodput→0).
- **High:** cold-start cost, migration/cache-loss cost, KV-eviction.
- **Medium:** autoscaling timing, p95/p99 tail, peak-VRAM (already realized).
- **Low/Reject:** the 110 NLP-eval / image / RL-finetuning false positives.

The defining tension: the **highest-leverage un-forecasted signals
(timeout, cold-start, migration, autoscaling) have the least public data.**

## 6. Phase 6 — priority outputs

From `economic_frontier_priority_ranking.json`:

- **Top datasets by priority** (all already-ingested or low): `ssong1/
  llmperf-bedrock`, `memoriant/dgx-spark-kv-cache-benchmark`, `metrum-ai/
  llm-perfdata`. No new dataset clears the bar.
- **Top signals most likely to produce new alpha:** (1) per-request
  queue_wait, (2) timeout/failure label, (3) KV-eviction rate, (4)
  cold-start model-load seconds, (5) migration cache-loss seconds.
- **Most likely to improve scheduling:** queue_wait, queue_depth, replica
  saturation, admission delay, p99 tail.
- **Most likely to improve migration decisions:** cache_loss_pct,
  prefix_reuse_destruction, reroute_latency, warmup_after_migration,
  migration_veto_label.
- **Most likely to improve goodput/$:** queue_wait (num), timeout risk
  (num), cold-start cost (denom), migration cost (denom), KV-eviction
  (cache_value).

All five top alpha signals are **blocked_by_pilot_telemetry or partial-
proxy-only** in public data.

## 7. Phase 7 — actionable recommendations

1. **What new signals exist?** As public HF data: essentially none beyond
   the already-covered latency/memory/cache-reuse. New serving-systems
   *datasets* exist (project-vajra NCCL/compute profiling, sage control-
   plane, sglang eviction shapes) but expose no new economic-frontier
   labels.
2. **Trainable today?** Only the already-known: cache_reuse (shadow-ready),
   peak_VRAM (shadow-ready), TTFT (promising). A queue-wait *proxy*
   forecaster on AcmeTrace job-level data is the one incremental option.
3. **Require pilot telemetry?** cold-start (server-class model-load
   seconds), migration (cache-loss seconds + reroute/warmup), autoscaling
   (all), per-request timeout labels, real per-request cache_hit.
4. **Require simulator priors?** cold-start & migration *cost* — already
   handled by the Economic ML Alpha v1 sensitivity sweep
   (`cold_start_migration_sensitivity.json`); no public data changes this.
5. **Likely economic-alpha sources?** Queue-wait and timeout-risk
   forecasting (Very High leverage) — but both need pilot telemetry to go
   beyond AcmeTrace's job-level proxy.
6. **Aurelius forecasting priorities:** (a) harden the shadow-ready
   cache_reuse + peak_VRAM models with a *second* dataset (cross-dataset
   generalization gap from ML Alpha v1); (b) build a queue-wait proxy
   forecaster on AcmeTrace + CARA; (c) treat cold-start/migration/
   autoscaling as pilot-gated.
7. **Ignore:** the 42 RL "cold-start" datasets, the 110 NLP/image Reject
   set, and `license=None` micro-benchmarks for redistribution.
8. **Next forecasting roadmap:** no new public dataset unlocks a new
   forecaster. The roadmap is **pilot-telemetry acquisition**, not more
   public discovery — public-data discovery has **plateaued** for the
   economic frontier (consistent with `AURELIUS_TELEMETRY_GAP_DISCOVERY.md`).

## 8. Honest limitations & production-readiness

- **Metadata-only.** Signal detection is regex over card/README/feature
  names; a dataset with rich hidden telemetry but a thin card could be
  under-credited (e.g. project-vajra's `.csv.xz` compute profiles were
  detected as data files but their internal schema was not opened).
- **Inspection budget 160 of 399** unique discoveries; ranked by infra
  relevance + seeds, so the long tail is NLP noise (verified on the
  inspected head).
- **No production readiness implied.** This is discovery only; nothing
  here is trained, ingested, or wired into any scheduler/scorer.
- **Search-term ambiguity** ("cold start", "serving", "scheduler",
  "decode") pulls heavy NLP/RL noise; the audit's value is the curated
  infra-seed enumeration + the honest negative result.

## 9. Final verdict

The frontier discovery **confirms** the Economic ML Alpha Audit v1:
the un-forecasted high-leverage signals (cold-start, migration, queueing
beyond proxies, autoscaling, per-request timeout) **remain blocked by the
absence of public production serving telemetry**. No new public HF dataset
unlocks a new economic-alpha forecaster. The next move is pilot telemetry,
plus cross-dataset hardening of the two already-shadow-ready models.
