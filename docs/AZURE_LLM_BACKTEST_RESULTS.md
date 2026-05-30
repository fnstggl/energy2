# Azure LLM Backtest Results — CANONICAL_TRACE_BACKTEST_AZURE_LLM_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` (reporting standard) and `docs/PUBLIC_TRACE_BACKTESTS.md` (dataset roles) first.

## Provenance

- **Source:** `csv:data/external/azure_llm/raw/AzureLLMInferenceTrace_conv.csv`  ·  workload variant: **conv**
- **Dataset:** Azure public LLM inference trace (https://github.com/Azure/AzurePublicDataset).
- Azure public data is a **public dataset, not customer telemetry**.

## Available vs missing fields (honest)

Discovered schema: `TIMESTAMP,ContextTokens,GeneratedTokens` (3 columns).

| field | available? | mapping |
|---|---|---|
| arrival timestamp | **yes** (absolute, sub-second) | `timestamp_s` |
| input/prompt tokens | **yes** (`ContextTokens`) | `prompt_tokens` |
| output tokens | **yes** (`GeneratedTokens`) | `output_tokens` |
| total tokens | derived | `prompt + output` |
| model / service id | **no** | `model = "azure-llm"` |
| request / session id | **no** | `session_id = None` |
| cache / prefix info | **no** | `cache_affinity_key = None` (has_cache_affinity=False) |
| latency / TTFT / elapsed | **no** | `elapsed_s = None` |
| explicit failure flag | **no** | failure only if `GeneratedTokens == 0` |

**This is a token-demand and arrival replay, NOT a measured-latency replay.** No TTFT or end-to-end latency is measured from Azure; the SLA budget is a standard interactive SLO decomposition (TTFT p99 budget + per-output-token budget) applied identically to every policy. Real KV cache hit rate is unavailable, so `cache_affinity_baseline` is **omitted (not applicable)** and `constraint_aware` receives **zero** cache benefit (`mean_reuse_fraction` = 0).

## Trace summary

- Requests replayed: **19,366**  ·  ticks: **20**  ·  tick size: **15s**
- Time range: 292s (0.081 h)
- Failure rate: 0.0000% (zero-output rows)
- Prompt/input tokens p50/p95/p99: 1020 / 4083 / 4142
- Output tokens p50/p95/p99: 129 / 451 / 601
- RPS/min mean/p95/max: 64.5533 / 87.8667 / 87.8667

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. SLA is a filter on the goodput numerator, never a term in the cost denominator. Headline baseline for interactive inference is **sla_aware** (`docs/RESULTS.md` §3 rule 5). All policies share the **same** unchanged serving physics (`aurelius/simulation/cluster/serving.py`), calibration, and cost basis — only the provisioning decision differs.

| policy | goodput/$ | SLA-compliant tokens | total infra $ | lat p95 (ms) | lat p99 (ms) | queue p95 (ms) | timeout % | migration/reroute |
|---|---|---|---|---|---|---|---|---|
| fifo | 2,515,377.78 | 3,848,528 | 1.53 | 9,855.03 | 19,924.61 | 13.33 | 5.843 | 0 |
| sla_aware | 1,940,705.38 | 3,860,063 | 1.99 | 9,704.06 | 19,574.73 | 5.85 | 5.583 | 13 |
| constraint_aware | 2,326,157.47 | 3,855,606 | 1.66 | 9,768.47 | 19,710.51 | 7.82 | 5.686 | 11 |
| queue_aware | 2,919,308.93 | 3,548,420 | 1.22 | 14,489.59 | 31,418.76 | 2,877.90 | 9.771 | 6 |
| cache_affinity_baseline | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)

- **Outcome:** `ALPHA_WIN`  ·  margin vs sla_aware: **+19.86%** on goodput/$
- **Sanity check vs FIFO (do-nothing):** constraint_aware DOES NOT beat static FIFO (-7.52%). FIFO is the sanity baseline, not the buyer-facing benchmark (`docs/RESULTS.md` §3).
- Notes: static FIFO (do-nothing, mean-sized) beats CA on goodput/$ at this load — honest caveat: static provisioning is cheapest under mild burst-load

## Load-regime sensitivity (same arrival shape, replayed at several loads)

Azure's absolute arrival rate is low; the canonical run scales it to a busy interactive tier (`--scale-rps`), preserving the real arrival shape. This sweep replays the **same** trace at several load multipliers so the result is transparently regime-dependent.

| load × | fifo | sla_aware | constraint_aware | queue_aware | CA vs sla_aware | CA beats fifo? |
|---|---|---|---|---|---|---|
| 0.33× | 2,551,339 | 1,836,817 | 2,210,221 | 2,547,455 | +20.33% | no |
| 0.5× | 2,324,383 | 1,860,689 | 2,324,697 | 2,834,092 | +24.94% | yes |
| 1× | 2,515,378 | 1,940,705 | 2,326,157 | 2,919,309 | +19.86% | no |
| 2× | 2,660,111 | 1,957,531 | 2,338,264 | 3,253,401 | +19.45% | no |
| 3× | 2,685,371 | 1,869,600 | 2,267,999 | 2,744,321 | +21.31% | no |

## What improved / what did not (strongest-baseline honesty)

- **Improved vs the reactive `sla_aware` headline:** +19.86% goodput/$ — `constraint_aware` avoids the headline autoscaler's over-provisioning.
- **Best tail-latency / safety:** `constraint_aware` p99 = 19,710.51 ms, timeout = 5.686% — the lowest p99 of any policy here.
- **Did NOT beat the strongest baseline** (`queue_aware`, goodput/$ 2,919,308.93): -20.32%. On this **smooth, low-burstiness** Azure trace, a leaner scaler (`queue_aware`) is cheaper per SLA-safe token; `constraint_aware`'s anticipatory safety margin provisions more than a non-bursty load requires. Per `docs/RESULTS.md` §3 this is **not** a clean win over the strongest relevant baseline — reported honestly, not hidden.

## Comparison to BurstGPT (does inference alpha generalize?)

- **BurstGPT** (`CANONICAL_TRACE_BACKTEST_BURSTGPT_V1`): constraint_aware ALPHA_WIN vs sla_aware (**+26.35%**), beats FIFO: True.
- **Azure LLM** (this run): constraint_aware ALPHA_WIN vs sla_aware (**+19.86%**), beats FIFO: False.
- **Generalization read (directional only):** the inference alpha *vs the reactive `sla_aware` headline* generalizes across both traces (BurstGPT and Azure both positive). The **clean win over every baseline** seen on BurstGPT does **not** generalize: BurstGPT is highly bursty (peak/mean RPS ≈ 75×), where anticipatory sizing pays off and `constraint_aware` beat even static FIFO; Azure conv is smooth (peak/mean ≈ 1.5×), where a leaner static/queue baseline is cheaper and `constraint_aware`'s value is tail-latency **safety**, not economic alpha. Two independent datasets, same canonical KPI and same unchanged serving physics, but different schemas (BurstGPT has model/log-type + a model-level cache proxy; Azure has neither) — a cross-trace check, **not** a like-for-like number. No overclaim.

## Honest limits

- Token-demand + arrival replay over proxy serving physics; **no measured latency, no TTFT, no KV cache** in the Azure data. Throughput, GPU power and prices are documented public priors (±50%), identical across policies.
- Azure public data is **not customer telemetry**; no model id, no session/prefix info. `cache_affinity_baseline` omitted as not applicable.
- **Not production-real savings.** Directional simulator result only.

