# Public Trace Backtests — framework + dataset roles

> Read `docs/RESULTS.md` (the canonical reporting standard) and
> `docs/BACKTESTS.md` (the frozen CAISO/PJM/ERCOT energy backtest) first.
>
> **Simulator/benchmark results are directional only — NOT production savings.**
> A public trace is *replayed serving traffic*, **not** customer telemetry. No
> number here may be quoted as a production saving until the `docs/RESULTS.md`
> §8 production-claim gate is satisfied.

This document describes the public-trace ingestion framework and the role each
dataset plays. **Only BurstGPT is implemented** in this phase
(`CANONICAL_TRACE_BACKTEST_BURSTGPT_V1`). The other datasets are documented
roadmap roles — **not ingested here** (see Non-goals).

This is a **public-trace benchmark phase, NOT an ML training phase.** No neural
forecasting, no model training, no robust-energy-engine changes, no simulator
constant tuning.

## 1. The framework

```
aurelius/traces/
  schema.py     # NormalizedLLMRequest contract + validation + summary stats
  burstgpt.py   # BurstGPT ingester (only dataset implemented)
  replay.py     # NormalizedLLMRequest -> ArrivalTick (simulator arrivals)
  backtest.py   # provisioning policies + serving-physics replay + canonical KPI
scripts/
  ingest_burstgpt.py        # download -> validate -> normalize -> stats -> processed trace
  run_burstgpt_backtest.py  # replay -> policies -> KPI -> results doc + summary JSON
```

Every ingester normalizes its raw rows into the **same** `NormalizedLLMRequest`
record, so the replay/backtest layers are dataset-agnostic. A future dataset is
added by implementing the `schema.TraceSource` interface (a `normalize()` that
maps raw columns onto `NormalizedLLMRequest`) — nothing downstream changes.

### NormalizedLLMRequest (the cross-dataset contract)

| field | meaning |
|---|---|
| `request_id` | stable per-request id |
| `timestamp_s` | arrival time (seconds) |
| `session_id` | conversation/session id when the source has one, else `None` |
| `model` | model label |
| `prompt_tokens` | input tokens |
| `output_tokens` | output tokens |
| `total_tokens` | prompt + output |
| `elapsed_s` | end-to-end response time when the source has it, else `None` — **NOT TTFT** |
| `log_type` | source usage-mode label |
| `is_failure` | `output_tokens == 0` OR (when an elapsed column exists) invalid/missing elapsed |
| `cache_affinity_key` | prefix/session locality **proxy** (NOT a measured KV hit rate) |

### How a trace becomes "simulator arrivals"

The Aurelius `ClusterSimulator` drives arrivals synthetically (diurnal + Markov
bursts) with a *constant* per-request token proxy. To replay a **real** trace,
`replay.requests_to_arrival_ticks` bins normalized requests into fixed-duration
`ArrivalTick`s that preserve real per-tick RPS, prompt/output tokens, model mix,
session/cache-affinity reuse, log-type mix and failures. The backtest then runs
these arrivals through the **unchanged** serving physics
(`aurelius/simulation/cluster/serving.py`) and scores the canonical KPI
(`aurelius/benchmarks/economics.py`). The serving physics, calibration
constants, and cost basis are identical across all policies — only the
provisioning/routing decision differs.

### GPU cluster traces — a second contract (`NormalizedGPUJob`)

GPU cluster traces (Alibaba v2023) describe **jobs requesting GPUs on a
heterogeneous fleet**, not token-level serving requests. They normalize into
`schema.NormalizedGPUJob` (+ `NormalizedGPUUtilizationSample` when a dataset has
utilization — Alibaba v2023 does not) and are scored by a **bin-packing**
backtest (`aurelius/traces/gpu_packing.py`) with **executable** packing
baselines, not the serving-physics replay. Same canonical KPI
(`docs/RESULTS.md` §1); `goodput_unit = completed_gpu_job_work` (token_equivalent
= effective_GPU × duration), labelled honestly.

## 2. Dataset roles

| Dataset | Role | Status |
|---|---|---|
| **BurstGPT** | LLM inference traffic replay — real arrival/burst shape, request/response token counts, failure rows for the interactive serving scenarios. | **Implemented** (`CANONICAL_TRACE_BACKTEST_BURSTGPT_V1`) |
| **Azure LLM inference traces (2023)** | Second, independent LLM inference trace — input/output token demand + arrival timing, to test whether inference alpha generalizes beyond BurstGPT. | **Implemented** (`CANONICAL_TRACE_BACKTEST_AZURE_LLM_V1`) |
| **Azure LLM Inference Dataset 2024 (week-long, multi-service)** | The **week-long** May 10–18 2024 conv+code trace used by DynamoLLM (HPCA 2025): real multi-day demand cycles + a **forecast-robustness / alpha-survival** experiment and lever attribution. | **Implemented** (`CANONICAL_TRACE_BACKTEST_AZURE_LLM_2024_WEEK_V1`) |
| **Azure LMM (multimodal) inference traces (2025)** | Multimodal token demand (image + text). | Roadmap — **not ingested** (LLM path landed first; do not claim multimodal support) |
| **Alibaba GPU cluster trace (v2023)** | Fragmentation / heterogeneous GPU scheduling — whole-GPU + fractional (`gpu_milli`) packing onto a heterogeneous fleet, with **executable** packing baselines (`first_fit`/`best_fit`/FFD/`greedy_packing`). | **Implemented** (`CANONICAL_TRACE_BACKTEST_ALIBABA_GPU_V2023_FRAGMENTATION_V1`) |
| **Philly (Microsoft) traces** | Training / fine-tuning GPU jobs — multi-tenant **temporal** job scheduling (queueing, backfill, fragmentation, fairness, retry/failure) on a fixed fleet. | **Implemented** (`CANONICAL_TRACE_BACKTEST_PHILLY_TRAINING_V1`) |
| **Alibaba GenAI 2026 (GenTD26)** | Top-down **multi-layer** stable-diffusion serving — application (requests + e2e latency), middleware (gateway queues), scheduler (pipeline/cold-start), infrastructure (GPU util/memory). Layers ingested with **classified linkage quality** (no faked joins). | **Implemented** (`CANONICAL_TRACE_BACKTEST_ALIBABA_GENAI_2026_V1`) |
| **MIT Supercloud** | Slurm scheduler + nvidia-smi GPU utilization + node-data snapshots + labelled DNN workloads — used to **validate Training Safe Utilization Frontier v1** on a third training-class trace (sibling of Philly + Alibaba GPU v2023). | **Implemented** (read-only ingestion + Training Frontier validation; `aurelius/traces/mit_supercloud.py`, `scripts/ingest_mit_supercloud.py`, `scripts/run_mit_supercloud_training_frontier.py`, `docs/MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md`). **Bounded real-S3 download** now available via `scripts/download_mit_supercloud_bounded.py` (~98 MB scheduler + ~80 KB labels + tres-mapping + a 50 MB head sample of node-data + optional budget-capped GPU sample); full ~1–2 TB at https://dcc.mit.edu/data + s3://mit-supercloud-dataset/datacenter-challenge/202201/ is **NOT committed**. See `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`. |

Known sources (for the future ingestion PRs — **do not download/ingest here**):
- Azure: https://github.com/Azure/AzurePublicDataset (`AzureLLMInferenceTrace`)
- Alibaba: https://github.com/alibaba/clusterdata
- Philly: https://github.com/msr-fiddle/philly-traces
- MIT Supercloud: https://github.com/MITLLSupercloud/ll-supercloud-datacenter-datasets

## 3. BurstGPT specifics

- Source: https://github.com/HPMLL/BurstGPT/tree/main/data — **`BurstGPT_1.csv`**.
- **Discovered schema** (verified against the raw file):
  `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type`.
  The published `BurstGPT_1.csv` carries **no Session ID column and no
  Elapsed-time column**, even though the project README documents them for a
  fuller schema. The ingester maps those columns *when present* and degrades
  honestly when absent:
  - no Session ID ⇒ `session_id = None`, `cache_affinity_key = "model:<model>"`
    (a **model-level** prefix-locality proxy — weak evidence of true prompt
    sharing, and explicitly **not** a KV cache hit rate);
  - no Elapsed-time column ⇒ `elapsed_s = None` and elapsed cannot mark
    failures, so only `Response tokens == 0` flags a failure.
- BurstGPT elapsed time (when a file provides it) is **end-to-end final response
  time, NOT TTFT.** No TTFT is measured from BurstGPT. The backtest's SLA budget
  is a standard interactive SLO decomposition (a TTFT p99 budget + a
  per-output-token budget), applied identically to every policy.
- BurstGPT's absolute arrival rate is low; the backtest replays a contiguous
  window scaled (`--scale-rps`) to a busy interactive tier, **preserving the
  real burst shape**, and reports a load-regime sensitivity sweep so the result
  is transparently regime-dependent.

See `docs/BURSTGPT_BACKTEST_RESULTS.md` for the canonical run, policies, and
results.

## 3b. Azure LLM specifics

- Source: https://github.com/Azure/AzurePublicDataset —
  `AzureLLMInferenceTrace_conv.csv` / `_code.csv` (2023) and the `_1week`
  variants (2024).
- **Discovered schema** (verified against the raw files):
  `TIMESTAMP,ContextTokens,GeneratedTokens` — **exactly three columns**.
  `TIMESTAMP` is absolute sub-second; `ContextTokens` = input/prompt tokens;
  `GeneratedTokens` = output tokens.
- Azure provides **far less** than BurstGPT. Honest degradation:
  - **no model / service id** ⇒ `model = "azure-llm"`;
  - **no request / session id, no prefix info** ⇒ `session_id = None`,
    `cache_affinity_key = None`. Real cache affinity is **unavailable**, so the
    backtest **omits `cache_affinity_baseline`** (not applicable) and
    `constraint_aware` gets **zero** cache benefit;
  - **no latency / TTFT / elapsed** ⇒ `elapsed_s = None`. This is a
    **token-demand and arrival replay, NOT a measured-latency replay**; no TTFT
    is measured from Azure;
  - **no failure column** ⇒ a row is a failure only if `GeneratedTokens == 0`.
- The two file variants (`conv`, `code`) are the only logical-workload signal;
  the variant is recorded as `log_type`.
- Azure conv is **much smoother** than BurstGPT (peak/mean RPS ≈ 1.5× vs ≈ 75×),
  which is the key contrast: see `docs/AZURE_LLM_BACKTEST_RESULTS.md`. The
  inference alpha *vs the reactive `sla_aware` headline* generalizes, but
  `constraint_aware`'s clean win over **every** baseline does not — on smooth
  load a leaner static/queue baseline is cheaper and CA's value is tail-latency
  safety, reported honestly.

See `docs/AZURE_LLM_BACKTEST_RESULTS.md` for the canonical run and results.

## 3f. Azure LLM Inference Dataset 2024 (week-long, multi-service) specifics

`CANONICAL_TRACE_BACKTEST_AZURE_LLM_2024_WEEK_V1`.

- Source: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md —
  the two **week-long** files on Azure blob storage (the actual 2024 dataset, not
  2023): `AzureLLMInferenceTrace_code_1week.csv` and
  `AzureLLMInferenceTrace_conv_1week.csv`. Licensed **CC-BY**; cite **DynamoLLM
  (HPCA 2025)**, Stojkovic et al. (arxiv 2408.00741).
- **Discovered schema (validated, NOT assumed from 2023):**
  `TIMESTAMP,ContextTokens,GeneratedTokens`. The 2024 `TIMESTAMP` carries a
  `+00:00` UTC offset + 6 fractional digits (distinct from the 2023 `.NET`
  7-digit form) — the ingester handles both.
- **Scale:** ~44.1M rows (code 16.8M + conv 27.3M), May 10–18 2024 (code is
  7 days from May 10; conv 7 days from May 12; combined union 9 days). Streamed
  in one memory-bounded pass (`azure_llm.stream_week_aggregate`).
- **Missing fields (same honest degradation as 2023):** no model/service id, no
  session/cache/prefix key, **no latency/TTFT/elapsed**. Token-demand + arrival
  replay only; **no TTFT claimed**; `cache_affinity_baseline` omitted and
  `constraint_aware` gets **zero** cache benefit. `log_type` = conv/code variant.
- **Demand pattern:** bursty (CV ≈ 0.65, peak/mean ≈ 2.85), strongly **periodic**
  (lag-1-day autocorrelation ≈ 0.68) and **multi-regime** (weekday/weekend RPS
  ≈ 2.7×). The absolute rate is low (peak ≈ 6 replicas at 1×); the canonical
  replays the real arrival SHAPE at documented busy-tier multipliers (1×/10×/50×).
- **Forecast robustness:** a forecast-driven autoscaler is run under
  `oracle_future` (analysis-only), `seasonal_time_of_day`, `moving_average`,
  `ewma`, `noisy_forecast`, `no_forecast_reactive` — only the demand estimate
  differs; **no future leakage except oracle**. `alpha_survival = alpha(mode) /
  alpha(oracle)`.
- **Headline result + attribution:** at 10× `constraint_aware` is an
  `ALPHA_WIN` vs the `sla_aware` headline (+~26% goodput/$), but the attribution
  shows the alpha is a **utilization / target-rho cost-efficiency** effect, NOT a
  forecasting one: holding the utilization target fixed, the demand-forecasting
  lever contributes <0.3% (oracle ceiling) and realistic forecasters retain only
  ~24% (EWMA) — `seasonal_time_of_day` and a 15%-noisy forecast are
  net-**negative**. Residency/affinity = 0 and prewarming is not modelled (no
  model/session id). See `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`.
- **Safe-utilization frontier audit + controller:** the
  `docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md` audit established that
  `constraint_aware` at rho ≈ 0.65 is SAFE but conservative — the
  anticipatory safe peak on the audit is `anticipatory@0.75`. The
  **Safe Utilization Frontier Controller v1**
  (`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`,
  `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`) turns that audit into a
  recommendation-only controller that estimates the safe frontier, vetoes
  unsafe points (timeout / queue / latency / telemetry gates) and selects
  the highest SLA-safe goodput/$ point — **simulator / shadow-mode only**,
  real-cluster execution disabled by default. The committed
  `constraint_aware` engine default (rho ≈ 0.65) is **unchanged** by this
  controller.

## 3c. Alibaba GPU v2023 specifics

- Source: https://github.com/alibaba/clusterdata — `cluster-trace-gpu-v2023/csv/`
  (`openb_pod_list_default.csv` pods + `openb_node_list_gpu_node.csv` fleet).
- **Discovered schema** (verified against the raw files):
  pods `name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time`;
  nodes `sn,cpu_milli,memory_mib,gpu,model`. `gpu_milli` = thousandths of a GPU
  (sharing); models are heterogeneous (T4/V100/P100/A10/G2/G3).
- **Missing (stated, not invented):** no GPU utilization time-series, no
  GPU-memory column, no per-pod node placement in the default pod list, no
  deadline/user columns. `NormalizedGPUUtilizationSample` is therefore empty for
  this dataset.
- This is a **bin-packing / fragmentation** backtest, not a serving replay. The
  headline baseline is the strongest **packing** baseline (`best_fit`/FFD/
  `greedy_packing`), **never** FIFO (`docs/RESULTS.md` §3). `constraint_aware`
  adds heterogeneous GPU-type **price-aware** placement on top of best-fit
  consolidation: it wins economic alpha when the fleet has spare cheap capacity
  to route to, and ties best-fit under saturation — reported with a
  fleet-contention sweep. See `docs/ALIBABA_GPU_BACKTEST_RESULTS.md`.

## 4. Reproduce

```bash
# BurstGPT — ingest (downloads BurstGPT_1.csv to data/external/burstgpt/raw):
python scripts/ingest_burstgpt.py
# BurstGPT — canonical backtest (busy interactive tier, real burst shape):
python scripts/run_burstgpt_backtest.py \
    --csv data/external/burstgpt/raw/BurstGPT_1.csv \
    --start-s 0 --duration-s 600000 --scale-rps 300 --tick-seconds 60

# Azure LLM (2023) — ingest (downloads AzureLLMInferenceTrace_conv.csv):
python scripts/ingest_azure_llm.py --workload conv
# Azure LLM (2023) — canonical backtest (busy interactive tier, real arrival shape):
python scripts/run_azure_llm_backtest.py \
    --csv data/external/azure_llm/raw/AzureLLMInferenceTrace_conv.csv \
    --scale-rps 12 --tick-seconds 15

# Azure LLM 2024 (week-long, multi-service) — ingest (downloads the two _1week
# files, ~0.7-1.1 GB each, to data/external/azure_llm_2024/raw; streamed):
python scripts/ingest_azure_llm_2024.py
# Azure LLM 2024 — canonical week-long backtest + forecast robustness + attribution:
python scripts/run_azure_llm_2024_backtest.py --primary-scale 10

# Alibaba GPU v2023 — ingest (downloads pod list + GPU node inventory):
python scripts/ingest_alibaba_gpu.py
# Alibaba GPU v2023 — canonical fragmentation/packing backtest:
python scripts/run_alibaba_gpu_backtest.py

# Philly — ingest (prints LFS download steps; uses fixture if full trace absent):
python scripts/ingest_philly.py
# Philly — canonical temporal training-job scheduling backtest:
python scripts/run_philly_backtest.py

# Alibaba GenAI 2026 — ingest all layers (downloads small layer files):
python scripts/ingest_alibaba_genai.py
# Alibaba GenAI 2026 — canonical multi-layer serving backtest:
python scripts/run_alibaba_genai_backtest.py --source-dir data/external/alibaba_genai/raw
```

Raw trace files are **downloaded, not committed** (`.gitignore`-d under
`data/external/*/raw/`). Unit tests use the fixtures
(`tests/fixtures/burstgpt_sample.csv`, `tests/fixtures/azure_llm_sample.csv`,
`tests/fixtures/alibaba_gpu/`, `tests/fixtures/philly_sample/`,
`tests/fixtures/alibaba_genai_sample/`) and never require the full files;
full-trace backtests are integration-only and are skipped if the raw file is
absent.

## 3e. Alibaba GenAI 2026 (GenTD26) specifics

- Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI
  — a top-down stable-diffusion serving trace; all layer files are small
  (< 5 MB compressed).
- **Layers + normalized records:** application →
  `NormalizedGenAIRequest` (`lora_request_trace.csv`: real `exec_time_seconds`
  e2e latency, model id, type, status, prompt size, `num_lora`); middleware →
  `NormalizedGatewayQueueSample` (`qps`/`queue_size`/`queue_rt`); scheduler →
  `NormalizedSchedulerPipelineEvent` (`pipeline_*`/`model_predict`/
  `basemodel`/`controlnet`/`lora_update` load latencies); infrastructure →
  `NormalizedInfraSample` (`pod_gpu_duty_cycle`/`pod_gpu_memory_used`/
  `pod_memory_util`).
- **Cross-layer linkage is classified from data, never faked:** the application
  layer is **`no_join`** to every metric layer (incompatible anonymized time
  bases — 2024 requests vs a 2022 metric epoch — and no `container_ip` in
  requests); the metric layers join to each other by `container_ip`
  (`container_join`). **No request→GPU causality is claimed.**
- The backtest is a request-level serving replay (application layer only);
  pipeline cold-start latencies are used as a **distribution calibration**
  (medians), not a per-request join. goodput_unit = **completed_requests**.
  The decisive lever is **model-affinity / prewarm** (87 base models, ~23 s
  measured base-model load): `constraint_aware` cuts mean cold-start ~23 s → ~3 s,
  winning both economic alpha and tail-latency safety vs the `sla_aware`
  headline. See `docs/ALIBABA_GENAI_BACKTEST_RESULTS.md`. The Philly full trace is a ~1 GB git-LFS
tarball (~6.6 GB extracted), so its canonical run is a **fixture-scale
demonstration** unless the tarball is downloaded.

## 3d. Philly specifics

- Source: https://github.com/msr-fiddle/philly-traces — `cluster_job_log` (JSON
  list) + `cluster_machine_list` (CSV), shipped as one ~1 GB git-LFS
  `trace-data.tar.gz`.
- **Discovered schema** (verified against the official analysis notebook): job =
  `{status (Pass/Killed/Failed), vc, jobid, submitted_time, user, attempts}`;
  each attempt = `{start_time, end_time, detail[{ip, gpus[]}]}`; times are
  `%Y-%m-%d %H:%M:%S`. GPU count = `sum(len(detail.gpus))` of the first attempt.
- **Missing (stated, not invented):** no GPU model/price (only per-GPU memory →
  a `GPU-<mem>` label), no CPU/host-mem request, no deadline; the
  `cluster_gpu_util` CSV is not parsed in this PR. `is_failed = status ∈
  {Failed, Killed}`. goodput_unit = `gpu_seconds_work` (NOT inference tokens).
- This is a **temporal scheduler-pressure** backtest (`gpu_scheduling.py`):
  queueing, backfill, fragmentation (jobs blocked despite aggregate free GPUs),
  size-class fairness, retry/failure (trace-observed). Headline = strongest
  scheduling baseline (best_fit / topology_aware / …), **never** FIFO. Because
  Philly has no GPU price signal, constraint_aware's heterogeneous-pricing lever
  is inactive: it ties the strongest baseline and wins big vs naive FIFO (whose
  strict head-of-line blocking collapses the queue) — a throughput/fairness
  safety win. See `docs/PHILLY_BACKTEST_RESULTS.md`.

## Non-goals

- **Implemented so far:** BurstGPT + Azure LLM 2023 + **Azure LLM 2024
  (week-long, multi-service + forecast robustness)** (LLM inference replay) +
  Alibaba GPU v2023 (GPU bin-packing/fragmentation) + Philly (temporal GPU
  training-job scheduling) + Alibaba GenAI 2026 (multi-layer GenAI serving).
- **Training Safe Utilization Frontier v1** (sibling of the serving frontier
  controller) reuses the Philly + Alibaba GPU v2023 committed backtest
  summaries — see `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md` +
  `docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md`.
- **MIT Supercloud Training Frontier validation** is now in place — see
  `docs/MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md` +
  `aurelius/traces/mit_supercloud.py`. Raw archive (~1 TB) lives at
  https://dcc.mit.edu/data and is **NOT committed**; the small synthetic
  fixture under `tests/fixtures/mit_supercloud_sample/` exercises every
  code path.
- No Azure **LMM/multimodal** or MIT ingestion yet (this PR did **not** ingest
  Azure LMM/multimodal).
- No ML training, no neural forecasting.
- No robust-energy-engine changes; no simulator constant tuning to force wins.
- No production-savings claims.
- Public traces are **not** customer telemetry. BurstGPT's Session/cache key is
  **not** a real KV cache hit rate; Azure has **no** cache/session/latency
  signal at all (token-demand + arrival replay only).
