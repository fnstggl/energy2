# Frontier Discovery Research Audit — Adjacent Workload Classes

> **Discovery-stage research audit.** This document is an early-signal scan of
> AI workload classes adjacent to LLM serving where Aurelius may have new
> max-safe-utilization alpha. **Nothing here is a production feature, a new
> optimizer, an ML training phase, or a savings claim.** Simulator /
> public-trace evidence in the cited dependencies is directional only — this
> audit does not claim production savings, and no number is allowed until
> the §8 production-claim gate in `docs/RESULTS.md` is satisfied.
>
> The robust energy engine, the static Safe Utilization Frontier Controller,
> the Dynamic Safe Frontier Estimator v1, the Dynamic Serving Frontier
> Calibration harness, and the Training Safe Utilization Frontier v1 are all
> **unchanged** by this audit. This PR adds documentation, a summary JSON, and
> a schema test — **no controllers, no ingestion, no benchmark mutations**.

- **Read first:**
  - `docs/RESULTS.md` (canonical KPI + claim rules)
  - `docs/PUBLIC_TRACE_BACKTESTS.md` (dataset roles + ingester contract)
  - `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`
  - `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`
  - `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`
  - `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`
  - `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`
  - `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`

## 1. Scope (binding)

- **Discovery only.** This audit does **not** add controllers, ingesters, or
  benchmarks. It identifies which adjacent workload classes plausibly carry a
  *constraint-safe max-utilization frontier* that Aurelius does not yet
  exploit, and ranks them.
- **No new datasets downloaded.** This audit reuses *already-committed*
  summaries (BurstGPT, Azure LLM 2023/2024, Philly, Alibaba GPU v2023, Alibaba
  GenAI 2026, MIT Supercloud bounded real sample). External datasets are
  named with their public URLs but are **not** downloaded here.
- **No oracle as headline.** Wherever an oracle / clairvoyant baseline exists
  in a cited summary, it is treated as analysis-only per `docs/RESULTS.md` §3.
- **Negative results are reported.** Workload classes where the public-trace
  evidence is too weak, the signal-to-noise is too low, or the schedulers are
  already aggressive get explicit `NOT ENOUGH DATA` / `LOW EXPECTED ALPHA`
  flags. The audit is asymmetric: a `BUILD NOW` recommendation requires both
  (a) a robust public trace already in-repo or known-feasible at bounded scale
  and (b) a credible frontier variable + safety-constraint pair.

## 2. Workload classes — assessment matrix

The eight workload classes from the audit charter, scored on a 1–5 scale:

- **Feasibility** — can a *bounded* discovery run get reliable signal from a
  public trace this quarter? Higher = easier.
- **Expected alpha** — how plausible is a new max-safe-utilization alpha
  beyond what the existing serving / training frontier already captures?
  Higher = more new alpha.
- **Complexity** — implementation effort for a v1 frontier module + test
  harness. Higher = more effort.

The "alpha" scoring is conservative: a class only scores ≥ 4 if (i) the
existing serving / training frontier evidence does NOT already cover the
lever and (ii) at least one public trace can plausibly evidence the lever
without ML training.

### 2.1 Class 1 — Batch inference

| field | value |
|---|---|
| Best public trace | **Azure LLM 2024 (week-long)** — `data/external/azure_llm_2024/` already ingested. BurstGPT (`data/external/burstgpt/`) as a secondary bursty proxy. AIPerf BurstGPT replay docs (`https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/burst-gpt-trace.md`) as a request-generator pattern, not a trace. |
| Sample size / duration | Azure LLM 2024: **44,107,694 rows · 9 days · 12,960 ticks @ 60 s** (`docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`). BurstGPT: **17,689 rows · 34 min** in the canonical replay window (`docs/BURSTGPT_BACKTEST_RESULTS.md`). |
| Signals available | Arrival timestamp, prompt tokens, output tokens, log-type (conv/code), variant burstiness, daily + weekly periodicity. |
| Missing signals | Per-request deadline / SLO label, batch boundaries, model/service id, session/cache key, latency / TTFT / elapsed (Azure 2024), explicit failure flag (Azure 2024 derives failure from `GeneratedTokens == 0`). |
| Candidate frontier variable | (batch-window-seconds, batch-concurrency, target-rho, deadline-slack-seconds). Multi-dimensional — **not** a scalar rho. |
| Safety constraints | Deadline-miss rate ≤ pre-registered ceiling; queue p99 ≤ ceiling; no SLA regression vs the strongest non-batch baseline; cost/SLA-compliant-token monotone-non-worsening. |
| Likely economic lever | Shift batches into low-utilization windows + run hotter rho than interactive serving (batch has more deadline slack than interactive). |
| Feasibility (1–5) | **5** — Azure LLM 2024 already ingested; existing `aurelius/traces/backtest.py` reuses cleanly. |
| Expected alpha (1–5) | **4** — batch is the natural neighbour of serving; the slack between interactive p99 budget and a batch-deadline budget is the alpha source, not present in serving rho today. |
| Implementation complexity (1–5) | **2** — small adapter over existing replay; deadline labelling is the new piece. |

### 2.2 Class 2 — Embedding generation

| field | value |
|---|---|
| Best public trace | **No native embedding trace.** Azure LLM 2024 as a *token-arrival proxy* (committed); Azure Functions 2019 (`https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md`) as a fan-out arrival proxy for short tasks — **not ingested** in this audit. AIPerf (`https://github.com/ai-dynamo/aiperf`) is a request-generator harness, not a trace. |
| Sample size / duration | Proxy only. Azure Functions 2019 is *known-large* (≈ 1 billion invocations across 14 days per the published README) but is **not** committed here — bounded ingest would be required. |
| Signals available | (proxy) arrival rate, prompt/input tokens; (Functions) per-app invocation counts, duration distribution. |
| Missing signals | Per-embedding latency, memory pressure, embedding-throughput per replica, real failure mode. |
| Candidate frontier variable | (batch-size, target-rho, replica-concurrency, memory-headroom). |
| Safety constraints | OOM ceiling, end-to-end latency p99, queue depth. |
| Likely economic lever | Bigger safe batches per replica + lower replica count at a higher safe rho. |
| Feasibility (1–5) | **3** — proxy is only directionally honest; no measured embedding latency exists in any committed trace. |
| Expected alpha (1–5) | **3** — likely real but evidentially hard to verify on a public trace without invented numbers. |
| Implementation complexity (1–5) | **3** — needs a new `goodput_unit = embeddings_completed`, an embedding-cost model, and an OOM safety gate (would re-tread `docs/RESULTS.md` §5 unit-labelling). |

### 2.3 Class 3 — Data processing / ETL / feature engineering

| field | value |
|---|---|
| Best public trace | **Azure Functions 2019** for arrival shape of short-task elastic compute (not committed; bounded ingest feasible). MIT Supercloud bounded real sample (`data/external/mit_supercloud/`, already committed) for CPU/GPU pressure. Azure Functions Invocation Trace **2021** (`https://github.com/Azure/AzurePublicDataset`) listed in source notes but **not present** in `data/external/`. |
| Sample size / duration | MIT Supercloud bounded: **10,000 jobs · 55.9 days · 7/10,000 GPU-util matches** (`docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`). Azure Functions 2019 (per its README) is ≈ 1 B invocations across 14 days — full ingest is **not** in scope for this audit. |
| Signals available | (MIT) queue wait, duration, gpu_count, status; (Functions 2019, if ingested) per-app invocation arrivals, duration histograms. |
| Missing signals | Per-job deadline / freshness target, job-graph dependencies, IO pressure, downstream consumer ack. |
| Candidate frontier variable | (concurrency-cap, batch-window, max-queue-delay, deferral-budget). |
| Safety constraints | Completion-rate ceiling, queue-delay ceiling, freshness-target compliance. |
| Likely economic lever | Defer non-critical ETL to cheap-power / low-utilization windows; raise concurrency when downstream consumers tolerate it. |
| Feasibility (1–5) | **3** — MIT is committed but is a *training-class* trace, not a clean ETL trace; Azure Functions 2019 would need bounded ingest. |
| Expected alpha (1–5) | **3** — likely real but heavily overlaps the existing Training Frontier v1 levers (`docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`). |
| Implementation complexity (1–5) | **3** — bounded Azure Functions ingester + a freshness-target safety gate. |

### 2.4 Class 4 — Vector indexing / RAG indexing

| field | value |
|---|---|
| Best public trace | **No native trace.** Azure Functions 2019 as ingestion-event arrival proxy (not committed); Alibaba GPU v2023 (`data/external/alibaba_gpu/`, committed) as resource-packing proxy if indexing is GPU-assisted. |
| Sample size / duration | Alibaba GPU v2023: **6,282 jobs · 149.3 days · 6,212 GPUs** (`docs/ALIBABA_GPU_BACKTEST_RESULTS.md`-summary). |
| Signals available | (Alibaba GPU) GPU type distribution, fragmentation, packing density, queue wait. |
| Missing signals | **Per-indexing-job memory pressure** (no GPU-mem column; `docs/PUBLIC_TRACE_BACKTESTS.md` §3c states this explicitly), IO throughput, embedding pipeline lag, vector-store-side throughput limits. |
| Candidate frontier variable | (concurrent-indexers, batch-vector-count, memory-headroom, IO-budget). |
| Safety constraints | OOM, IO saturation, queue blowup. |
| Likely economic lever | Run more indexers in parallel when memory + IO headroom allows; lower replica count on quiet windows. |
| Feasibility (1–5) | **2** — no committed trace measures vector-store / IO throughput; the "indexing" framing on Alibaba GPU is heavily proxied. |
| Expected alpha (1–5) | **2** — the existing fragmentation-packing lever already covers most of what a public trace can evidence here. |
| Implementation complexity (1–5) | **4** — a believable v1 would need a vector-store IO model that no public trace currently grounds. |

### 2.5 Class 5 — Synthetic data generation

| field | value |
|---|---|
| Best public trace | **BurstGPT** (committed) for LLM request/token distributions, **Azure LLM 2024** (committed) for multi-day token throughput, **Alibaba GenAI 2026** (`data/external/alibaba_genai/`, committed) for model-load / LoRA / pipeline cold-start. |
| Sample size / duration | BurstGPT: 17,689 requests · 34 min canonical window. Azure LLM 2024: 44.1 M rows · 9 days. Alibaba GenAI 2026 application layer: **26,392 requests · 23.0 days · 79 distinct models · 16.5 % LoRA fraction · e2e p99 = 106 s** (`data/external/alibaba_genai/processed/alibaba_genai_backtest_summary.json`). |
| Signals available | Arrival timestamps, prompt/output tokens, real e2e latency (GenAI), measured base-model cold-start medians (GenAI). |
| Missing signals | Per-job deadline, downstream-consumer SLA, eval-pass criterion. |
| Candidate frontier variable | (batch-window-seconds, prewarm-set-size, target-rho, deadline-slack-seconds) — overlaps the model-residency lever. |
| Safety constraints | Failure-rate ceiling, cold-start tail-latency ceiling, deadline-miss ceiling. |
| Likely economic lever | Deadline-flexible generation can run *much* hotter than interactive serving and can pre-warm aggressively when the prewarm horizon matches the deadline horizon. |
| Feasibility (1–5) | **5** — three committed traces already cover the signal mix. |
| Expected alpha (1–5) | **4** — deadline flexibility is the largest unexploited slack relative to interactive serving rho (≈ 0.65). |
| Implementation complexity (1–5) | **2** — extends the existing serving + GenAI residency machinery; no new dataset ingest required. |

### 2.6 Class 6 — Evaluation workloads / eval harnesses

| field | value |
|---|---|
| Best public trace | **LMSYS Chatbot Arena conversations** (`https://huggingface.co/datasets/lmsys/chatbot_arena_conversations`) for conversation shape; **ShareGPT** via AIPerf (`https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-share-gpt-dataset`) for canonical eval shapes; Azure LLM 2024 / BurstGPT (both committed) as throughput proxies. **LMSYS + ShareGPT are not committed in this audit** — bounded ingest required. |
| Sample size / duration | LMSYS Chatbot Arena (per the HF dataset card): ≈ 33 K conversations, multi-turn — bounded ingest feasible. ShareGPT (per AIPerf docs): ≈ 52 K conversations packaged as a benchmark dataset. |
| Signals available | Multi-turn prompt/response token counts, conversation length distribution, model id. |
| Missing signals | Per-eval-suite deadline, per-eval pass/fail criterion, real serving latency under eval load. |
| Candidate frontier variable | (eval-batch-window-hours, concurrency, target-rho, deadline-slack). |
| Safety constraints | Eval-suite completion deadline; SLA-safety floor stays at interactive when eval shares a fleet with interactive (mixed-fleet veto). |
| Likely economic lever | Eval workloads are the **most deadline-flexible class** the audit touches — typical eval runs tolerate hours/days of slack, far more than any interactive class. The frontier is plausibly close to rho ≈ 1 on a dedicated fleet, far above the interactive 0.65. |
| Feasibility (1–5) | **4** — LMSYS / ShareGPT are bounded-ingest-feasible (datasets are < 1 GB packed) and the existing replay pipeline already accepts `NormalizedLLMRequest`. |
| Expected alpha (1–5) | **5** — the deadline-slack-vs-rho slope is steepest for evals; this is the workload class most likely to carry *new* alpha beyond serving. |
| Implementation complexity (1–5) | **2** — a `NormalizedLLMRequest` adapter + a deadline-aware backtest variant + a safety floor that prevents eval-fleet contention from leaking into interactive. |

### 2.7 Class 7 — RLHF pipelines

| field | value |
|---|---|
| Best public trace | **No native public RLHF trace.** Composite proxy: MIT Supercloud bounded (training proxy, committed), Alibaba GPU v2023 (GPU packing, committed), Azure LLM 2024 / BurstGPT (rollout-inference proxy, committed). |
| Sample size / duration | Per-stage sizes are the cited traces; no end-to-end RLHF run exists in any committed dataset. |
| Signals available | (training proxy) queue wait, packing density; (inference proxy) arrival shape, tokens; (no real reward-model latency, no PPO-step coupling). |
| Missing signals | Reward-model evaluation latency, RL-update synchronisation, off-policy buffer lag, fine-tune step / rollout-inference handoff timing. |
| Candidate frontier variable | Per-stage sub-frontier composed (training rho × rollout rho × eval rho) — no published trace measures all three on the same job. |
| Safety constraints | Gang-failure on training stage, queue p99 on rollout, deadline on eval. |
| Likely economic lever | Cross-stage pipelining and asynchrony; theoretically large, evidentially nearly invisible from any public trace. |
| Feasibility (1–5) | **1** — no public end-to-end RLHF trace exists; every connection between stages would be a faked join (`docs/PUBLIC_TRACE_BACKTESTS.md` "classified linkage quality" principle). |
| Expected alpha (1–5) | **2** — extrapolation rather than evidence; the existing training + serving frontiers already cover the dominant levers. |
| Implementation complexity (1–5) | **5** — composing three sub-frontiers without faking the joins is most of the work and adds little publishable evidence. |

### 2.8 Class 8 — Agent swarms / agent workloads

| field | value |
|---|---|
| Best public trace | **LMSYS / ShareGPT** as conversation/task-shape proxy; AIPerf agentic benchmark dataset (`https://docs.nvidia.com/aiperf/reference/benchmark-datasets`) if available; Azure LLM 2024 / BurstGPT for request burstiness; Azure Functions 2019 for fan-out / function-like task shape. None of the agentic-specific datasets are committed. |
| Sample size / duration | Proxy only; LMSYS / ShareGPT as in §2.6. |
| Signals available | Conversation length, multi-turn shape (proxy), request burstiness. |
| Missing signals | Tool-call latencies, per-step memory state, retry/loop behaviour, end-to-end multi-step deadline. |
| Candidate frontier variable | (concurrent-agents, per-step-rho, multi-step-deadline, retry-budget). |
| Safety constraints | End-to-end multi-step latency p99, retry-loop divergence, queue blowup on bursty fan-out. |
| Likely economic lever | Burst-spike scheduling vs steady-state agent pool sizing. |
| Feasibility (1–5) | **3** — proxy-grade; no committed agent-step-resolved trace exists. |
| Expected alpha (1–5) | **3** — agent deadlines are closer to interactive than to batch, narrowing the slack relative to serving rho. |
| Implementation complexity (1–5) | **4** — a believable v1 needs a multi-step deadline model + retry-cost accounting. |

## 3. Comparison vs current evidence (per the audit charter §5)

| frontier | current evidence | source | what's already captured |
|---|---|---|---|
| Serving (rho) | **Strong alpha** (committed) | `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md` (+25.75 % vs `sla_aware`), `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md` (dynamic recovers 73 % of oracle gap), `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md` | interactive rho ≈ 0.65 → ≈ 0.75–0.85 safe peak in shadow mode, with a per-window dynamic estimator |
| Training | **Ties / directional** | `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`, `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`, `docs/PHILLY_BACKTEST_RESULTS.md`, `docs/ALIBABA_GPU_BACKTEST_RESULTS.md` | multi-dimensional candidate (packing / backfill / reservation / gang-strictness / heterogeneity / price-aware routing) on committed summaries — sibling, not extension, of the rho controller |
| Residency / model affinity | **Safety + latency win, modest KPI alpha** | `docs/MODEL_RESIDENCY_DECISION_ENGINE_RESULTS.md`, `docs/ALIBABA_GENAI_BACKTEST_RESULTS.md` | prewarm + cache-affinity routing wins **safety** alpha; not a primary KPI multiplier on its own |

The discovery question is *what's left*. Against the table above, the
adjacent classes carry plausibly-new KPI alpha only where the *deadline
slack* or the *per-class safety floor* differs materially from interactive
serving. That argues — on the existing evidence — for **batch inference**
and **evaluation workloads** as the two strongest "build next" candidates,
both for the same reason: they tolerate much longer deadlines than
interactive serving, which is where the residual rho headroom lives.

## 4. Ranked recommendation

The ranking aggregates feasibility × expected-alpha and *penalises*
complexity that would require new ML training or new full-trace ingest.

### 4.1 Build now

1. **Evaluation workloads / eval harnesses** (§2.6) — *highest new
   expected-alpha-per-effort*. Deadline slack is the steepest lever the audit
   identifies; LMSYS Chatbot Arena + ShareGPT are bounded-ingest-feasible
   (<1 GB packed); the existing `NormalizedLLMRequest` contract accepts an
   adapter cleanly. **v1 deliverable would be:** a deadline-aware eval-class
   backtest harness + a safety floor that prevents eval-fleet contention
   from leaking into interactive workloads. **Do not** start without first
   committing a bounded ingester for LMSYS or ShareGPT (≤ 200 MB sample)
   and verifying the deadline-slack-vs-rho slope on the committed Azure 2024
   trace as a sanity check.
2. **Batch inference frontier** (§2.1) — *highest feasibility*. Azure LLM
   2024 is already ingested; the frontier variable is a small multi-axis
   sweep over the existing replay; the safety floor is a deadline-miss
   ceiling that the existing per-tick KPI already supports. **v1 deliverable
   would be:** a batch-class frontier candidate sweep + a deadline-slack
   safety gate, sibling to the existing serving Safe Utilization Frontier
   Controller. Opt-in, shadow only, no controller defaulted.

### 4.2 Investigate later

3. **Synthetic data generation** (§2.5) — *strong feasibility, alpha
   plausibly large.* Wait for the eval-class v1 to land first; many of the
   levers will be shared (deadline-flex + prewarm), so a generation-class
   v1 should re-use the eval-class candidate space rather than re-invent it.
4. **Data processing / ETL** (§2.3) — *real alpha, but heavily overlaps
   training frontier.* Investigate after a second training-class trace
   beyond MIT Supercloud has been validated, so the ETL frontier is not
   confused with the training one.
5. **Embedding generation** (§2.2) — *useful but evidentially weak.* Worth
   revisiting once a public embedding trace with measured latency exists.
6. **Agent swarms** (§2.8) — *proxy-grade.* Worth revisiting when an
   agent-step-resolved public trace exists.

### 4.3 Not enough data

7. **Vector indexing / RAG indexing** (§2.4) — no committed trace measures
   the IO / memory pressure that the frontier candidate would need; the
   "indexing" framing on Alibaba GPU is heavy proxy. **Do not build now.**

### 4.4 Low expected alpha

8. **RLHF pipelines** (§2.7) — no public end-to-end RLHF trace; any v1
   would compose three sub-frontiers across three unrelated traces, which
   is exactly the *faked-join* failure mode the
   `docs/PUBLIC_TRACE_BACKTESTS.md` "classified linkage quality" principle
   prohibits. **Do not build now.** Revisit if a public RLHF trace appears.

## 5. Datasets to ingest next (bounded only)

The following are *candidates* for a future bounded-ingest PR — **none are
downloaded by this audit**:

| dataset | proposed bound | purpose |
|---|---|---|
| **LMSYS Chatbot Arena conversations** (HF) | head-bounded sample, ≤ 200 MB | eval-class deadline-flex shape |
| **ShareGPT** (via AIPerf) | packaged sample, ≤ 100 MB | eval-class conversation shape |
| **Azure Functions 2019** | head-bounded, ≤ 500 MB invocation count subset | short-task arrival shape for ETL / embedding fan-out |
| **Alibaba GenAI 2026 LoRA detail** | already committed | (covered) |

Each candidate would need: a `NormalizedLLMRequest` (or task-shape) adapter,
a fixture-scale CSV under `tests/fixtures/`, an ingest script under
`scripts/`, a discovery summary under `data/external/<dataset>/processed/`,
and the `docs/PUBLIC_TRACE_BACKTESTS.md` dataset-role row honestly
classified.

## 6. What NOT to build yet

- **Do not** add ML training. The risk scores in the existing dynamic
  estimator are deterministic heuristics (`docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md` §5);
  any new class should be deterministic-first as well.
- **Do not** add a new controller default. Every new class should land
  opt-in, shadow only, with `executable_in_real_cluster=False` at
  construction, mirroring `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` and
  `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`.
- **Do not** ingest large datasets without a bound. The MIT Supercloud
  precedent (`docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`) — 2.96 MB
  of a ~1–2 TB archive — is the template.
- **Do not** claim production savings on any of these classes. Every v1
  must live behind the `docs/RESULTS.md` §8 production-claim gate.
- **Do not** use an oracle / clairvoyant baseline as a headline. Oracle is
  analysis-only per `docs/RESULTS.md` §3.
- **Do not** fold a secondary KPI into the primary. The KPI stays
  `sla_safe_goodput_per_infrastructure_dollar` per `docs/RESULTS.md` §1.
- **Do not** invent missing signals. Where a trace lacks deadlines, the
  field stays `None` and the candidate either runs without that gate or
  surfaces `INSUFFICIENT_TELEMETRY` — mirroring the training-frontier
  pattern (`docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md` §1).
- **Do not** start the RLHF or vector-indexing class until a public trace
  with the missing signals exists.

## 7. Commands run (this audit)

```
git status
git log --oneline -10
git fetch origin main
ls data/external/
ls data/external/azure_llm_2024/processed/
ls data/external/burstgpt/processed/
ls data/external/mit_supercloud/processed/
ls data/external/alibaba_genai/processed/
ls data/external/philly/processed/
ls data/external/alibaba_gpu/processed/
ls data/external/frontier/
wc -l tests/fixtures/azure_llm_2024_sample.csv tests/fixtures/burstgpt_sample.csv tests/fixtures/azure_llm_sample.csv
python3 -c 'json.load(...)'  # read existing summaries for committed sample sizes
pytest tests/test_frontier_discovery_research.py -q
```

No dataset was downloaded by this audit. No simulator run was triggered.

## 8. Files changed

- `docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md` (this file, added).
- `data/external/frontier/frontier_discovery_research_summary.json` (added).
- `tests/test_frontier_discovery_research.py` (added).

No optimizer, ingester, controller, simulator, or benchmark code is
modified. No existing artifact is rewritten.

## 9. Honesty / scope

- This audit is **discovery-stage signal scanning**, not a benchmark.
- Sample sizes and signals are sourced from the **committed** processed
  summaries (BurstGPT, Azure LLM 2023/2024, Philly, Alibaba GPU v2023,
  Alibaba GenAI 2026, MIT Supercloud bounded) and from the publicly
  documented sizes of named-but-uncommitted datasets (LMSYS, ShareGPT,
  Azure Functions 2019/2021). The latter are flagged "not committed".
- The "expected alpha" column is a **researcher estimate, not a measured
  result**. It is conservative and intentionally avoids quoting a percentage.
- Every "BUILD NOW" recommendation explicitly requires a bounded
  data-ingest step *first*, exactly as the MIT Supercloud bounded sample
  precedent did before the Training Frontier v1 validation re-ran.
- Reading `docs/RESULTS.md` §3 (oracle/clairvoyant baselines are
  analysis-only) and §8 (production-claim gate) is binding on any v1 that
  comes out of this audit.
