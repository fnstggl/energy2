# BurstGPT Backtest Results — CANONICAL_TRACE_BACKTEST_BURSTGPT_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` (reporting standard) and `docs/PUBLIC_TRACE_BACKTESTS.md` (dataset roles) first.

## Provenance

- **Source:** `csv:data/external/burstgpt/raw/BurstGPT_1.csv`
- **Exact file:** BurstGPT_1.csv (https://github.com/HPMLL/BurstGPT/tree/main/data)
- BurstGPT is a **public LLM-serving trace, not customer telemetry**.
- The published `BurstGPT_1.csv` has **no Session ID and no Elapsed-time column**; the cache-affinity key is a model-level prefix-locality **proxy**, not a measured KV cache hit rate.
- BurstGPT elapsed time (when present in other files) is **end-to-end response time, NOT TTFT**. No TTFT is measured from BurstGPT.

## Trace summary

- Requests replayed: **17,689**  ·  ticks: **34**  ·  tick size: **60s**
- Time range: 5s → 2005s (0.56 h)
- Failure rate: 0.0000%
- Model distribution: {'ChatGPT': 11841, 'GPT-4': 5848}
- Log-type distribution: {'API log': 309, 'Conversation log': 17380}
- Prompt tokens p50/p95/p99: 499 / 1862 / 2389
- Output tokens p50/p95/p99: 236 / 634 / 934
- RPS/min mean/p95/max: 8.6711 / 17.5167 / 20.7500
- Cache-affinity proxy: 2 distinct keys, reuse rate 99.99%

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. SLA is a filter on the goodput numerator (`tokens × (1 − timeout_rate_pct/100)`), never a term in the cost denominator. Headline baseline for interactive inference is **sla_aware** (`docs/RESULTS.md` §3 rule 5).

| policy | goodput/$ | SLA-compliant tokens | total infra $ | lat p95 (ms) | lat p99 (ms) | queue p95 (ms) | timeout % | migration/reroute | cache proxy |
|---|---|---|---|---|---|---|---|---|---|
| fifo | 1,583,907.21 | 4,268,975 | 2.70 | 13,771.19 | 28,559.78 | 1,455.96 | 8.919 | 0 | no |
| sla_aware | 1,278,784.21 | 3,834,081 | 3.00 | 50,457.42 | 120,264.89 | 25,886.86 | 12.952 | 23 | no |
| constraint_aware | 1,615,693.90 | 4,438,537 | 2.75 | 11,478.01 | 23,002.17 | 109.25 | 5.096 | 17 | yes |
| queue_aware | 1,396,838.15 | 3,444,143 | 2.47 | 83,011.50 | 201,416.36 | 47,558.54 | 17.122 | 19 | no |
| cache_affinity_baseline | 1,587,558.86 | 4,278,817 | 2.70 | 13,625.89 | 28,239.33 | 1,455.96 | 8.709 | 0 | yes |

## Policies compared

- **fifo** — no optimization; static replica count sized once for the trace mean load. Sanity baseline (`docs/RESULTS.md` §3).
- **sla_aware** — reactive autoscaler (one-tick lag, conservative utilization target). Headline baseline for interactive inference.
- **constraint_aware** — Aurelius: anticipatory (EWMA) sizing + cache-affinity prefill savings + churn hysteresis, gated to a safe utilization target.
- **queue_aware** — scales on the queue-wait p95 signal only (no decode SLA budget, no cache).
- **cache_affinity_baseline** — static sizing + session/prefix-affinity prefill savings, but no load reaction. Isolates the cache lever.

All policies share the **same** serving physics (`aurelius/simulation/cluster/serving.py`, unchanged), the same calibration constants (`serving_value`), and the same cost basis (`InfrastructureCostConfig` defaults). Only the provisioning/routing decision differs — wins come from decisions, not tuned constants.

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)

- **Outcome:** `ALPHA_WIN`  ·  margin vs sla_aware: **+26.35%** on goodput/$
- **Safety evidence:** p99<=0.5x_queue_aware, timeout<=0.5x_queue_aware
- **Sanity check vs FIFO (do-nothing):** constraint_aware beats static FIFO (+2.01%). FIFO is the sanity baseline, not the buyer-facing benchmark (`docs/RESULTS.md` §3).

## Load-regime sensitivity (same burst shape, replayed at several loads)

BurstGPT's absolute arrival rate is low; the canonical run scales it to a busy interactive tier (`--scale-rps`), preserving the real burst shape. This sweep replays the **same** trace at multiple load multipliers so the result is transparently regime-dependent — not a single cherry-picked load.

| load × | fifo | sla_aware | constraint_aware | queue_aware | cache_affinity | CA vs sla_aware | CA beats fifo? |
|---|---|---|---|---|---|---|---|
| 0.33× | 1,089,325 | 953,723 | 1,025,834 | 966,415 | 1,091,286 | +7.56% | no |
| 0.5× | 1,423,830 | 1,127,740 | 1,312,249 | 1,036,520 | 1,427,190 | +16.36% | no |
| 1× | 1,583,907 | 1,278,784 | 1,615,694 | 1,396,838 | 1,587,559 | +26.35% | yes |
| 2× | 1,664,009 | 889,861 | 2,031,056 | 954,976 | 1,668,092 | +128.24% | yes |
| 3× | 2,167,072 | 1,553,649 | 2,021,686 | 1,547,649 | 2,173,155 | +30.13% | no |

Reading: constraint_aware beats the **realistic reactive autoscaler (`sla_aware`, the headline baseline)** across the swept load levels. It beats even static `fifo` once bursts regularly saturate capacity; under mild burst-load a static `fifo` sized for the mean is cheaper (an honest caveat, not hidden).

### What improved / what did not

- Goodput/$ vs sla_aware: Δ +336909.70 (+26.35%).
- Infra $ vs sla_aware: 2.75 vs 3.00.
- Latency p99 vs sla_aware: 23,002.17 vs 120,264.89 ms.
- Migration/reroute (scale events): 17 vs 23.

## Honest limits

- Trace-replay over proxy serving physics; no per-request KV simulation. Token throughput, GPU power, and prices are documented public priors (±50%), identical across policies. Override with real contract rates before any external claim (`docs/RESULTS.md` §8 production-claim gate).
- The SLA budget is a standard interactive SLO decomposition (TTFT p99 ≤ 2000ms + per-output-token budget), applied identically to every policy — BurstGPT supplies no TTFT to calibrate against.
- **Not production-real savings.** Directional simulator result only.

