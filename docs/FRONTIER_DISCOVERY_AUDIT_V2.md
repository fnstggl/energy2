# Frontier Discovery Audit v2 — Systems-Research Artifact Deep Search

> **Discovery / audit-only PR.** No model trained, no production code modified,
> no data ingested, no savings claimed, no constants invented, no synthetic
> data used. No public / conference / observability artifact is treated as
> pilot telemetry. Missing cold-start / migration / cache-loss values are
> never silently zeroed. This audit reads the systems-research ecosystem;
> it downloads nothing into production.
>
> **Read first:** `docs/ECONOMIC_ML_ALPHA_V1.md`,
> `docs/FRONTIER_DISCOVERY_AUDIT_V1.md`, `docs/FRONTIER_SIGNAL_HYPOTHESES.md`,
> `docs/PILOT_TELEMETRY_CONTRACT.md`, `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md`.
>
> **Evidence:** `data/external/frontier_discovery_v2/*.json`
> (`phase0_known_gaps`, `artifacts_llm_serving`, `artifacts_cloud_cluster_traces`,
> `artifacts_serverless_coldstart_conference`, `observability_metrics`,
> `phase5_artifact_registry`, `phase6_economic_value_scoring`,
> `phase7_verdict`).

## 0. Mission & method

The Economic ML Alpha Audit (v1) found the highest-leverage *un-forecasted*
signals are cold-start duration, model-load duration, autoscaling events,
migration events, cache-loss events, cache-hit rate, warm-pool occupancy,
queue-state evolution, timeout/failure labels, routing instability, replica
contention, and tail-latency amplification — and that **none of them are
present as measured labels in public Hugging Face datasets**. v1 was
HF-focused. **This audit searches the broader systems-research ecosystem**:
USENIX/ACM conference artifacts (OSDI, NSDI, SOSP, EuroSys, ATC, Middleware,
SoCC, MLSys), artifact-evaluation repos, Zenodo archives, GitHub release
assets, cloud/cluster trace releases, and open-source observability stacks.

Method: a metadata + schema fan-out across four streams (LLM-serving
artifacts; cloud/cluster traces; serverless cold-start + conference
artifacts; open-source observability metrics). Schemas were verified by
fetching the actual GitHub READMEs, arXiv pages, Zenodo landing pages, proto
files, and official metrics docs. Every artifact carries an honest
`{measured | aggregated | simulated | workload | live-metrics-only |
measured-unreleased | gated}` classification, and `row_count` is left null
wherever a source did not publish it (we did not invent counts).

The binding KPI (unchanged) is:

```
sla_safe_goodput_per_dollar = (output_tokens if sla_met else 0)
                            / (gpu_cost + energy_cost + migration_cost
                               + cold_start_cost − cache_value)
```

A signal earns economic alpha only by forecasting an uncertain **upstream**
quantity that flips `sla_met` (numerator) or avoids a denominator cost.

## 1. Headline finding

**v2 confirms v1's core negative but is not a flat repeat of it.** The four
highest-leverage GPU-serving signals — server-class **cold-start/model-load**,
inference **migration**, **autoscaling** events, and per-request
**timeout/failure** labels — still have **no measured public dataset**. Every
system that measures them (ServerlessLLM, HydraServe, Splitwise/DynamoLLM,
Llumnix, HeteroScale) releases **code and figures, not data**.

But the systems ecosystem yields three things the HF-only v1 sweep missed:

1. **Mooncake (FAST'25) is a second, production-derived cache-reuse trace.**
   Its per-request `hash_ids` (prefix blocks, `block_size=512`) let cache-hit
   rate be **simulated** from real Kimi traffic — a genuine *second source*
   to cross-validate the `cache_reuse_pct` model that ML Alpha v1 flagged as
   **shadow-ready but single-dataset (SwissAI-only)**. This is the one
   concrete public-data action that attacks a binding v1 caveat.
2. **Measured cold-start data exists — for serverless functions, not GPUs.**
   The Huawei Cloud 2025 release (`sir-lab/data-release`, EuroSys'25) ships
   ~11.9M cold-start **events** with a component breakdown
   (`podAllocationCost`, `deployCodeCost`, `deployDependencyCost`,
   `schedulingCost`) — a structural template + calibration distribution for
   the cold-start cost model, but **FaaS, not GPU weight-load**.
3. **A named-metric reconstruction map** that turns v1's vague
   "blocked_by_pilot_telemetry" into an actionable pilot contract: a
   Ray-Serve-wrapped vLLM/SGLang/Triton pilot can capture cold-start,
   autoscaling, queue-state, cache-hit, and tail-latency **off-the-shelf**;
   only per-request **migration** has no metric in any stack.

## 2. Phase 0 — known gaps (extracted from v1)

`phase0_known_gaps.json`. Every missing signal from the v1 audits, with its
v1 status, KPI path, and best prior public signal. The `blocked_by_missing_
labels` / `blocked_by_pilot_telemetry` set going into v2:

| Signal | v1 status | best v1 public signal | measured rows v1 |
|---|---|---|---:|
| cold-start (server-class model-load) | blocked_by_missing_labels | ejhusom consumer-Ollama (not server-class) | 0 |
| migration (cache-loss/reroute/warmup/veto) | blocked_by_missing_labels | CC-traces KV-hash *proxy* (not realized) | 0 |
| autoscaling (all) | blocked_by_pilot_telemetry | Google/Borg job-level proxy | 0 |
| timeout/failure (per-request) | blocked (weak proxy) | Optimum/Odyn/llmperf error_rate (aggregate) | aggregate |
| cache-loss | blocked (proxy not realized) | CC-traces KV-hash | 0 |
| warm-pool occupancy | blocked_by_pilot_telemetry | none | 0 |
| queue-state evolution | partial / proxy | AcmeTrace job-level `queue_wait`, CARA | job-level |
| cache-hit rate | partial realized (single-dataset) | SwissAI `cache_reuse_pct` (shadow-ready) | 60,000 |

## 3. Phase 1–2 — conference + paper-family artifacts

`artifacts_llm_serving.json`, `artifacts_serverless_coldstart_conference.json`.
Deep-searched HydraServe, Mooncake, Llumnix, Sarathi-Serve, DistServe,
Splitwise, Vidur, AlpaServe, Helix, SGLang, vLLM, CacheBlend, Dynamo,
ServerlessLLM, and the serverless cold-start family (Serverless-in-the-Wild,
FaaSNet, SeBS, IceBreaker), plus Azure/Zenodo artifact releases.

**The structural result: released LLM-serving artifacts are workload traces,
simulators, code, or aggregated logs — not operational telemetry datasets.**

| Project | What's released | Operational labels? |
|---|---|---|
| **Mooncake** (FAST'25) | `{timestamp, input_length, output_length, hash_ids[]}` JSONL, real Kimi-derived | **No** — but `hash_ids` give prefix-reuse → cache-hit **simulatable** |
| Azure LLM 2023/24, LMM 2025 | `TIMESTAMP, ContextTokens, GeneratedTokens` CSV | No — arrival + tokens only |
| Splitwise (Zenodo) | Azure traces + KV-transfer prototype + sim | No — migration is **code**, not an event log |
| **Llumnix** (OSDI'24) | reproducible AE + `history_log.zip` | **Aggregated** P99/Mean only — no per-migration event dataset |
| DistServe, Sarathi-Serve | code + benchmark scripts | No — datasets prepared locally |
| **Vidur** (MLSys'24) | sim + committed workload traces | **Simulated** queue/GPU/latency CDFs |
| AlpaServe, Helix | simulators + prototypes | No — Azure workload as input |
| **ServerlessLLM** (OSDI'24) | code + README speedup table | **Measured-unreleased** (model-load + migration) |
| **HydraServe** (NSDI'26) | code | **Measured-unreleased** (cold-start breakdown) |
| CacheBlend (LMCache, EuroSys'25) | code | No — technique, public RAG benchmarks |
| FaaSNet (ATC'21) | per-request cold-start latency | **Measured** but FaaS + Google-Form gated |
| SeBS (Middleware'21, Zenodo) | cold/warm exec CSVs | **Measured** but FaaS, summary granularity |
| IceBreaker (ASPLOS'22, Zenodo) | "Scripts and data" ZIP | **Unverified** contents |

The two most on-topic GPU systems — **ServerlessLLM** (model-load + live
migration) and **HydraServe** (cold-start breakdown) — are exactly the ones
that release no data. This is the dominant failure mode: *measured behind
closed doors, published as plots.*

## 4. Phase 3 — cloud / cluster trace search

`artifacts_cloud_cluster_traces.json`. Schemas verified verbatim from the
Azure / Google / Alibaba / Huawei repos.

**LLM-serving traces (Azure):** `AzureLLMInferenceDataset2023` (Splitwise),
`2024` (DynamoLLM), `AzureLMMInferenceDataset2025` (ModServe) — all three
columns/four-columns are arrival timestamp + token counts. **Zero operational
signals** (no latency, failure, queue, GPU, cache, cold-start). The only true
LLM-serving public releases carry no frontier labels.

**Cluster traces (proxies, non-LLM):**
- **Google ClusterData 2019 (Borg)** — `instance_events` enum includes an
  explicit **`QUEUE`** event (scheduler-ineligible = real queue signal),
  **`EVICT`** (preemption = best public migration proxy), and **`FAIL`/`KILL`**
  (failure proxy); `tail_cpu_usage_distribution` is a p91–p99 tail of CPU
  *utilization* (not request latency). 2011 has EVICT/FAIL/KILL but no QUEUE.
- **Alibaba GPU v2020 (MLaaS, NSDI'22)** — GPU utilization (`gpu_wrk_util`,
  `avg/max_gpu_wrk_mem`), derivable queueing delay (the paper's "long
  queueing delays"), `Failed` status. GPU *training*, not serving.
- **Alibaba GPU v2023 (openb/FGD)** — K8s scheduler-sim snapshots (the GPU
  trace already in the in-repo lineage).
- **Alibaba microservices v2021/v2022** — per-call response time `rt` (the
  closest public **per-request latency**) + call-graph routing topology
  (`um`→`dm`); v2022's paper is about autoscaling but ships **no scale-event
  labels**.

**Serverless cold-start (measured, FaaS):**
- **Huawei Cloud 2025 (`sir-lab/data-release`, EuroSys'25)** — the standout.
  Per-cold-start-event rows with `totalCost_cold_start` decomposed into
  `podAllocationCost / deployCodeCost / deployDependencyCost / schedulingCost`
  (µs granularity); a day-30 per-request table; per-minute time-series with
  **p90/p95/p99 quantiles** on 19 metrics incl. `num_cold_starts`, `num_pods`
  (warm-pool/autoscaling proxy); the cold-start `requestID` links the request
  that *triggered the scaling decision*. **But FaaS** — `deployCodeCost` is
  code/dependency download, **not LLM weight load**.
- **Huawei 2023 (SoCC'23)** — aggregated means + `Instances` (scaling proxy).
- **Azure Functions 2019 ("Serverless in the Wild", ATC'20)** — the
  canonical cold-start *paper*, but its released data **explicitly excludes
  cold-start time** and is aggregated, not per-invocation. A common
  miscitation; verified here.
- **`maxday/lambda-perf`** — daily JSON of per-invocation AWS Lambda
  cold-start ms by runtime/memory/arch (CPU, synthetic).
- **Alibaba GPU v2025 (disaggregated DLRM, NSDI'25)** — 7,386 GPU instances
  with per-instance `creation_time/scheduled_time/deletion_time` → scale
  events **inferrable**; instance-level, not per-request.

**Migration as data:** the only measured migration dataset found is the
**CSAP SNU VM Live Migration** set (40k+ samples, SoCC'17) — VM-class, 2017,
and **email-gated**, i.e. not inference migration and not an open download.

## 5. Phase 4 — open-source observability (the named-metric map)

`observability_metrics.json`. Metric names verified against official docs /
source for vLLM, SGLang, Ray Serve, KServe, BentoML, TGI, LMCache, Mooncake,
Triton+TensorRT-LLM. These are **live runtime telemetry a pilot must scrape —
not downloadable datasets** — but they answer *what a pilot can reconstruct*.

| Signal | Exposed where (exact metric) | Not exposed |
|---|---|---|
| Queue depth / wait | `vllm:num_requests_waiting`, `vllm:request_queue_time_seconds`, `tgi_queue_size`, `nv_inference_queue_duration_us`, `ray_serve_deployment_queued_queries` | — |
| Cache-hit rate | `sglang:cache_hit_rate`, `vllm:prefix_cache_hits`/`_queries`, `lmcache:retrieve_hit_rate` | TGI, KServe, BentoML, Ray Serve |
| KV eviction | `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds` | everyone else (usage only) |
| TTFT | `vllm:time_to_first_token_seconds`, `sglang:...`, `nv_inference_first_response_histogram_ms` | **TGI (no native TTFT)** |
| Tail latency (e2e) | histograms in vLLM/SGLang/TGI/Triton/Ray Serve | — |
| Success / failure | `nv_inference_request_failure`, `ray_serve_deployment_error_counter_total`, `vllm:request_success_total{finished_reason}` | explicit *timeout* label (none) |
| GPU hardware util | `nv_gpu_utilization/memory/power` (**Triton only**) | all others → DCGM exporter |
| **Cold-start / model-load** | **Ray Serve only**: `ray_serve_replica_startup_latency_ms`, `ray_serve_multiplexed_model_load_latency_ms`, `ray_serve_replica_initialization_latency_ms` | vLLM/SGLang/TGI/Triton/BentoML/KServe |
| **Autoscaling** | **Ray Serve**: `ray_serve_autoscaling_{target,desired}_replicas`, `ray_serve_autoscaling_total_requests`; KServe via Knative KPA | inference engines |
| **Request migration** | **nowhere** — only `vllm:num_preemptions_total` (preemption) + KV-transfer bytes (`lmcache:num_remote_{read,write}_bytes`, Triton `disaggregated_serving_metrics{kv_cache_transfer_ms}`, Mooncake transfer stats) | a true per-migration counter |

**Decisive observation:** of the 12 signals, ~7–8 are reconstructable from
off-the-shelf metrics on a vLLM/SGLang/Triton deployment; **cold-start and
autoscaling are reconstructable only via the orchestration layer (Ray
Serve)**; and **request migration is the single signal with no metric in any
stack** — it needs custom instrumentation. This sharpens the
`docs/PILOT_TELEMETRY_CONTRACT.md` into named fields.

## 6. Phase 5–6 — registry + economic-value scoring

`phase5_artifact_registry.json` (dedup'd index + the 12-signal public-
availability matrix), `phase6_economic_value_scoring.json` (each artifact
scored on impact to the 8 forecasting targets, classed High/Medium/Low/No
Value by **realizable** alpha, not raw richness).

| Class | Artifacts |
|---|---|
| **High** | Mooncake FAST'25 traces (cache cross-validation); live-observability map (as a *pilot instrumentation* contract, not data) |
| **Medium** | Huawei 2025 cold-start (FaaS prior + cost-model structure); Alibaba GPU v2025 (inference autoscaling proxy) |
| **Low** | Google 2019, Alibaba microservices, maxday/lambda-perf, SeBS, Llumnix logs, Vidur, Azure LLM, Azure Blob 2020 |
| **No Value (as data)** | ServerlessLLM, HydraServe/HeteroScale/Chiron/SageServe (measured-unreleased); CSAP (gated); IceBreaker (unverified) |

The scoring rule mirrors v1's honesty: an artifact that merely restates an
already-ingested signal (Google EVICT, Azure LLM workload) scores Low; one
that unblocks a binding v1 caveat (Mooncake → cache cross-dataset) scores
High; simulator output (Vidur) scores Low because training on it would learn
the simulator, not reality.

## 7. Phase 7 — final verdict

`phase7_verdict.json`. The seven mission questions:

1. **What missing signals were found?** Cold-start *duration* as measured
   public data — **FaaS only** (Huawei 2025, lambda-perf, SeBS), not GPU
   model-load. Autoscaling and warm-pool as **proxies** (Huawei `num_pods`,
   Alibaba GPU v2025 lifecycle). Cache-hit as a **new reconstruction**
   (Mooncake `hash_ids`). Migration only as VM-class gated data (CSAP) +
   preemption proxies (Google EVICT).

2. **Which artifacts contain measured labels?** Huawei 2025 (cold-start
   events), lambda-perf, SeBS, FaaSNet (gated), Azure Functions Invocation
   2021, Google 2011/2019 (lifecycle events), Alibaba GPU v2020 + microservices
   v2021/22, CSAP (gated), Llumnix (aggregated). **Not** Azure LLM (workload),
   Mooncake (workload + reuse structure), Vidur/Splitwise/AlpaServe/Helix
   (simulated/code), ServerlessLLM/HydraServe (measured-unreleased).

3. **Which signals remain unavailable publicly?** Server-class GPU
   model-load/cold-start duration; per-migration cache-loss seconds for
   inference; labeled LLM autoscaling scale-events; per-request LLM
   timeout/failure labels; GPU warm-pool occupancy; routing-instability
   labels. **All six: none.**

4. **Which signals appear reconstructable from public traces?** Cache-hit
   (Mooncake simulation — **new**); job-level queue-state (Google 2019 QUEUE,
   Alibaba GPU v2020); migration *preemption* proxy (Google EVICT); tail
   latency (Alibaba microservices `rt`); autoscaling *inferred* (Alibaba GPU
   v2025 lifecycle); a cross-domain FaaS cold-start prior (Huawei/lambda-perf).

5. **Which signals require pilot telemetry?** GPU model-load/cold-start
   (Ray Serve `replica_startup`/`multiplexed_model_load`); LLM autoscaling
   (Ray Serve `autoscaling_*` / Knative KPA); per-request **migration**
   (custom instrumentation — no metric anywhere); per-request timeout labels;
   real per-request cache-hit (SGLang/vLLM/LMCache); GPU warm-pool occupancy.
   **Refinement vs v1:** these are reconstructable from a Ray-Serve-wrapped
   vLLM/SGLang pilot's *standard* metrics — only migration needs custom code.

6. **Which artifacts to ingest next?** (1) **Mooncake** — second cache-reuse
   set to cross-validate the v1 single-dataset shadow-ready model (bounded,
   public; cache-hit labelled as *simulated*). (2) **Huawei 2025** — cold-start
   cost-model template + distribution (labelled cross-domain FaaS,
   calibration-only, never server-class ground truth). (3) **Alibaba GPU
   v2025** — inference autoscaling proxy. Defer Vidur-Bench / microservices;
   ignore the simulators, IceBreaker (unverified), CSAP (gated), and the
   code/figures-only systems.

7. **Estimated incremental economic-alpha by category** (qualitative tiers;
   **no percentage savings claimed** per `docs/RESULTS.md` §8):

   | Category | Leverage | Realizable now | Note |
   |---|---|---|---|
   | Cache hit/reuse | High | **High** | Mooncake → cross-dataset validation of a caveated shadow-ready model; largest realizable increment |
   | Queue-state | Very High | Medium | job-level proxies extend; serving-level pilot, now named |
   | Tail-latency / TTFT | Medium | Medium | measured latency exists; cause-attribution pilot |
   | Memory-pressure | High | Medium | peak_VRAM already shadow-ready; no new public data |
   | Cold-start | High | Low (GPU) / Medium (FaaS prior) | structure + prior improve sweep; GPU headline pilot-gated |
   | Migration | High | **Very Low** | no inference data anywhere; only custom-instrumentation signal |
   | Autoscaling | Medium-High | Low | proxies + Ray Serve/Knative metrics |

## 8. Relationship to v1 and to the telemetry-gap thesis

v1 (HF-only) concluded public-data discovery had **plateaued** for the
economic frontier. v2 (systems-ecosystem) partially revises that: there *are*
new artifacts the HF sweep could not see — a second production cache-reuse
trace (Mooncake), measured FaaS cold-start data with cost decomposition
(Huawei), a GPU-inference autoscaling proxy (Alibaba GPU v2025), and a
named-metric observability map. **But the revision is bounded:** none of these
supplies the *measured GPU-serving* cold-start/migration/autoscaling/timeout
labels that flip the headline KPI. The defining v1 tension holds — *the
highest-leverage un-forecasted signals have the least public data* — and the
next move remains **pilot telemetry**, now with (a) a concrete public-data
cross-validation step (Mooncake) and (b) an actionable named-metric pilot
contract, rather than another public-discovery pass.

## 9. Honest limitations

- **Metadata + schema, not byte-level.** Several artifacts (Huawei Google-Drive
  archives, IceBreaker ZIP, gated CSAP/FaaSNet) were classified from their
  papers/READMEs/landing pages, **not** by opening the files. These carry
  `confidence: med/low` and `public_downloadable` is `conditional`/`unknown`
  where access is gated. No file was downloaded into the repo.
- **Simulator vs measured.** Vidur, SplitwiseSim, AlpaServe, Helix produce
  rich *signal categories* but as **simulated** output; they are scored Low
  precisely to avoid the v1 "train on a constant/synthetic and fabricate a
  signal" failure mode.
- **Observability ≠ dataset.** The Phase-4 metric map is *live runtime*
  telemetry; it tells a pilot what to scrape, not what is publicly available
  today. Request migration has **no** off-the-shelf metric.
- **No production readiness implied.** Nothing here is trained, ingested, or
  wired into any scheduler/scorer/overlay. This is discovery only, behind the
  `docs/RESULTS.md` §8 production-claim gate.

## 10. Files

- `docs/FRONTIER_DISCOVERY_AUDIT_V2.md` (this file).
- `data/external/frontier_discovery_v2/phase0_known_gaps.json`
- `data/external/frontier_discovery_v2/artifacts_llm_serving.json`
- `data/external/frontier_discovery_v2/artifacts_cloud_cluster_traces.json`
- `data/external/frontier_discovery_v2/artifacts_serverless_coldstart_conference.json`
- `data/external/frontier_discovery_v2/observability_metrics.json`
- `data/external/frontier_discovery_v2/phase5_artifact_registry.json`
- `data/external/frontier_discovery_v2/phase6_economic_value_scoring.json`
- `data/external/frontier_discovery_v2/phase7_verdict.json`

No optimizer, ingester, controller, simulator, or benchmark code is modified.
No existing artifact is rewritten.
