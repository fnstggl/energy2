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

## 2. Dataset roles

| Dataset | Role | Status |
|---|---|---|
| **BurstGPT** | LLM inference traffic replay — real arrival/burst shape, request/response token counts, failure rows for the interactive serving scenarios. | **Implemented** (`CANONICAL_TRACE_BACKTEST_BURSTGPT_V1`) |
| **Azure LLM inference traces** | Second, independent LLM inference trace — input/output token demand + arrival timing, to test whether inference alpha generalizes beyond BurstGPT. | **Implemented** (`CANONICAL_TRACE_BACKTEST_AZURE_LLM_V1`) |
| **Azure LMM (multimodal) inference traces (2025)** | Multimodal token demand (image + text). | Roadmap — **not ingested** (LLM path landed first; do not claim multimodal support) |
| **Alibaba GPU cluster traces** | Fragmentation / heterogeneous GPU scheduling — utilization, placement, multi-tenant behavior to calibrate the packing baselines (`first_fit`/`best_fit`/FFD). | Roadmap — not ingested |
| **Alibaba GPU cluster traces** | Fragmentation / heterogeneous GPU scheduling — utilization, placement, multi-tenant behavior to calibrate the packing baselines (`first_fit`/`best_fit`/FFD). | Roadmap — not ingested |
| **Philly (Microsoft) traces** | Training / fine-tuning GPU jobs — multi-tenant job scheduling + topology-aware placement + RESERVE_CAPACITY crowding. | Roadmap — not ingested |
| **MIT Supercloud** | Utilization / power / monitoring calibration — to calibrate the simulator's utilization, power and thermal priors against real datacenter monitoring. | Roadmap — not ingested |

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

## 4. Reproduce

```bash
# BurstGPT — ingest (downloads BurstGPT_1.csv to data/external/burstgpt/raw):
python scripts/ingest_burstgpt.py
# BurstGPT — canonical backtest (busy interactive tier, real burst shape):
python scripts/run_burstgpt_backtest.py \
    --csv data/external/burstgpt/raw/BurstGPT_1.csv \
    --start-s 0 --duration-s 600000 --scale-rps 300 --tick-seconds 60

# Azure LLM — ingest (downloads AzureLLMInferenceTrace_conv.csv):
python scripts/ingest_azure_llm.py --workload conv
# Azure LLM — canonical backtest (busy interactive tier, real arrival shape):
python scripts/run_azure_llm_backtest.py \
    --csv data/external/azure_llm/raw/AzureLLMInferenceTrace_conv.csv \
    --scale-rps 12 --tick-seconds 15
```

Raw trace files are **downloaded, not committed** (`.gitignore`-d under
`data/external/*/raw/`). Unit tests use the fixtures
(`tests/fixtures/burstgpt_sample.csv`, `tests/fixtures/azure_llm_sample.csv`)
and never require the full files; full-trace backtests are integration-only and
are skipped if the raw file is absent.

## Non-goals

- **Implemented so far:** BurstGPT + Azure LLM (LLM inference replay only).
- No Azure **LMM/multimodal**, Alibaba, Philly, or MIT ingestion yet.
- No ML training, no neural forecasting.
- No robust-energy-engine changes; no simulator constant tuning to force wins.
- No production-savings claims.
- Public traces are **not** customer telemetry. BurstGPT's Session/cache key is
  **not** a real KV cache hit rate; Azure has **no** cache/session/latency
  signal at all (token-demand + arrival replay only).
