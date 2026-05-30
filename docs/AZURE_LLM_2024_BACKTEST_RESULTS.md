# Azure LLM 2024 Backtest — CANONICAL_TRACE_BACKTEST_AZURE_LLM_2024_WEEK_V1

> **Simulator benchmark result — directional only, NOT production savings** (`docs/RESULTS.md` §8). Token-demand + arrival replay, **NOT** a measured-latency replay: Azure 2024 provides token counts + timestamps only (no latency/TTFT, no model/service id, no session/cache key). **No TTFT is claimed.** Read `docs/RESULTS.md` + `docs/PUBLIC_TRACE_BACKTESTS.md` first.

## Provenance & exact files used

- **Dataset:** Azure LLM Inference Dataset **2024** (week-long, multi-service).
- **Source:** `raw:data/external/azure_llm_2024/raw (multi-service week: code, conv)`
- **Exact files used:**
  - `data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_code_1week.csv`
  - `data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_conv_1week.csv`
- **Citation (CC-BY):** DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency, HPCA 2025, Stojkovic et al. (arxiv.org/abs/2408.00741); dataset CC-BY (github.com/Azure/AzurePublicDataset)
- **Discovered schema:** `TIMESTAMP,ContextTokens,GeneratedTokens` (verified against the actual 2024 files; the 2024 TIMESTAMP carries a `+00:00` UTC offset and 6 fractional digits — distinct from the 2023 `.NET` 7-digit form).

### Available vs missing fields (honest)

| field | available? | mapping |
|---|---|---|
| arrival timestamp | **yes** (absolute, sub-second, UTC) | `timestamp_s` |
| input/prompt tokens | **yes** (`ContextTokens`) | `prompt_tokens` |
| output tokens | **yes** (`GeneratedTokens`) | `output_tokens` |
| total tokens | derived | `prompt + output` |
| model / service id | **no** | `model = "azure-llm"` (single label) |
| workload variant | **yes** (file: conv/code) | `log_type` |
| session / cache / prefix | **no** | `cache_affinity_key = None` |
| latency / TTFT / elapsed | **no** | `elapsed_s = None` (no TTFT claimed) |
| explicit failure flag | **no** | failure iff `GeneratedTokens == 0` |

## Trace summary (full week-long trace)

- **Rows ingested:** 44,107,694 (variant distribution: {'code': 16803695, 'conv': 27303999})
- **Time range (UTC):** 2024-05-10T00:00:00.009930+00:00 → 2024-05-18T23:59:59.995460+00:00
- **Duration:** 9.0 days (216.0 h), 12960 ticks @ 60.0s
- **Failures (zero-output):** 0 (0.0%) · out-of-order rows: 0
- **Prompt tokens p50/p95/p99/max:** 1487.0 / 6138.0 / 7683.0 / 7999.0
- **Output tokens p50/p95/p99/max:** 20.0 / 386.0 / 632.0 / 5000.0
- **Total tokens p50/p95/p99:** 1528.0 / 6284.0 / 7695.0
- **RPS/min mean/p95/p99/max:** 56.722858 / 134.3 / 145.666667 / 161.816667
- **Burstiness:** peak/mean 2.8528 · p99/mean 2.568 · CV 0.6538
- **Day/night mean RPS:** 64.725262 / 48.720453 · **weekday/weekend:** 71.674423 / 26.819726
- **Missing fields:** model/service id, session/conversation id, cache/prefix key, latency/TTFT/elapsed, explicit failure flag (derived: GeneratedTokens==0)

## Demand-pattern analysis (Task 5)

- **Classification:** `bursty, periodic_daily, multi_regime_weekday_weekend`
- CV 0.6538 · peak/mean 2.8528 · p99/mean 2.568
- Autocorrelation lag-1 (min): 0.9936 · lag-1-day: 0.6762
- Weekday/weekend RPS ratio: 2.672
- **Forecastable pattern present:** True (strong daily + weekly seasonality)

## Base backtest — primary scale 10.0× (real arrival shape; busy-tier multiplier)

> Headline baseline = **sla_aware** (`docs/RESULTS.md` §3 rule 5). The absolute Azure rate is low (peak ≈ 6 replicas at 1×); the canonical replays the real shape at documented multipliers (see sweep) — only the provisioning decision differs across policies.

| policy | goodput/$ | SLA-compliant tokens | infra $ | GPU-hours | lat p95 (ms) | lat p99 (ms) | queue p99 (ms) | timeout % | scale events |
|---|---|---|---|---|---|---|---|---|---|
| fifo | 1,288,234.72 | 13,623,545,911 | 10,575.36 | 5,184.0 | 192,656.96 | 479,641.41 | 264,464.4 | 31.908 | 0 |
| sla_aware | 2,032,039.55 | 30,498,109,395 | 15,008.62 | 7,357.17 | 4,651.28 | 9,482.11 | 33.98 | 6.605 | 10,152 |
| queue_aware | 2,490,662.91 | 20,671,515,888 | 8,299.604 | 4,068.43 | 61,338.98 | 151,316.37 | 78,398.79 | 24.184 | 8,696 |
| constraint_aware | 2,555,324.54 | 30,214,182,940 | 11,824.01 | 5,796.08 | 4,826.61 | 9,953.86 | 0.63 | 7.639 | 8,830 |
| utilization_aware | 3,238,462.77 | 28,888,966,253 | 8,920.58 | 4,372.83 | 5,685.57 | 12,349.03 | 48.42 | 12.103 | 8,897 |
| naive_overprovisioning | 997,155.77 | 30,757,070,382 | 30,844.8 | 15,120.0 | 4,431.07 | 8,915.92 | 0.0 | 5.55 | 0 |
| oracle_forecast_ANALYSIS_ONLY *(analysis-only)* | 2,422,788.92 | 30,391,943,928 | 12,544.198 | 6,149.12 | 4,712.11 | 9,635.26 | 0.5 | 7.053 | 9,702 |

- **constraint_aware vs sla_aware:** `ALPHA_WIN` (+25.7517% on goodput/$). Beats FIFO sanity baseline: True (+98.3586%).

### Load-regime sweep (goodput/$; real shape at multipliers)

| scale | fifo | sla_aware | constraint_aware | CA vs sla_aware % | oracle alpha>0 |
|---|---|---|---|---|---|
| 1.0× | 1,681,607.31 | 1,776,169.99 | 2,180,502.0 | +22.764 | True |
| 10.0× | 1,288,234.72 | 2,032,039.55 | 2,555,324.54 | +25.752 | True |
| 50.0× | 1,349,842.96 | 2,055,527.29 | 2,590,107.58 | +26.007 | True |

## Forecast robustness / alpha survival (Task 4)

> Single forecast-driven autoscaler; only the demand estimate differs. **No future leakage except `oracle_future` (analysis-only).** alpha = KPI(mode) − KPI(no_forecast_reactive); alpha_survival = alpha(mode)/alpha(oracle_future).

| forecast mode | goodput/$ | timeout % | p99 (ms) | GPU-hours | scale events | RPS MAE | RPS MAPE | token MAE | alpha vs no-forecast | survival |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle_future *(analysis-only)* | 2,422,788.92 | 7.053 | 9,635.26 | 6,149.12 | 9,702 | 0.0 | 0.0 | 0.0 | — | — |
| seasonal_time_of_day | 1,031,606.27 | 40.294 | 675,612.42 | 3,172.95 | 6,925 | 297.037 | 0.6296 | 26.71 | -1,385,092.88 | -227.446 |
| moving_average | 2,417,593.48 | 7.234 | 10,121.46 | 6,147.72 | 2,830 | 28.1364 | 0.0621 | 2.77 | 894.32 | 0.1469 |
| ewma | 2,418,131.24 | 7.215 | 9,887.14 | 6,148.65 | 5,365 | 26.0889 | 0.058 | 2.76 | 1,432.09 | 0.2352 |
| noisy_forecast | 2,335,418.2 | 8.977 | 25,642.62 | 6,165.07 | 11,301 | 66.366 | 0.1183 | 8.75 | -81,280.96 | -13.3471 |
| no_forecast_reactive | 2,416,699.15 | 7.295 | 9,958.19 | 6,148.9 | 9,701 | 30.0417 | 0.0669 | 3.42 | — | — |

- **Oracle alpha (forecasting ceiling):** 6,089.77 goodput/$ (positive: True).

## Attribution — where does the alpha come from? (research question)

**Dominant lever: utilization/target_rho (cost-efficiency).** constraint_aware's 25.752% win over the headline is decomposed below; the forecasting lever is isolated by holding the utilization target fixed.

| lever | measure | value | note |
|---|---|---|---|
| forecasting demand (isolated) | oracle ceiling % of KPI / best realistic % | 0.252% / 0.0593% | oracle alpha 6,089.77; best survival {'mode': 'ewma', 'survival_ratio': 0.2352}; seasonal/noisy net-NEGATIVE → fragile |
| autoscaling timing | CA vs reactive (sla_aware) % | 25.752 | vs static FIFO: 98.359% |
| queue management | queue_aware vs reactive % | 22.57 | — |
| utilization | utilization_aware vs reactive % | 59.37 | hot target_rho is cheapest but risks tail latency |
| residency / affinity | contribution | 0.0 | NOT APPLICABLE — Azure 2024 has no model/service id, session id, or cache/prefix key; cache_affinity_baseline omitted and constraint_aware receives ZERO cache benefit. |
| prewarming | — | n/a | NOT MODELLED — this single-model autoscaling harness has no model cold-start/prewarm step (Azure exposes no model id); prewarm timing is not a factor on this trace. |

**constraint_aware WINS** vs the sla_aware headline (+25.75% goodput/$). The forecast experiment shows demand-forecasting IS a real lever (oracle alpha +6089.77 goodput/$), but only ~24% survives the best realistic forecaster — so most of the theoretical forecasting alpha is lost to forecast error on this trace. **Attribution (decomposed):** holding the utilization target FIXED, the demand-forecasting lever itself contributes only ~0.059% (best realistic forecaster, ewma) and some forecasters (seasonal time-of-day, 15%-noisy) are net-NEGATIVE — so forecasting *accuracy* is NOT where the win comes from. The dominant lever is **utilization / target-rho cost-efficiency**: utilization_aware (rho 0.85) alone is +59.37% vs the reactive headline, and constraint_aware's 25.752% win is mostly running hotter (rho 0.65 + anticipatory EWMA trim + hysteresis) while staying SLA-safe — an **autoscaling-timing / utilization** effect on a strongly periodic (daily+weekly) demand curve. Residency/affinity contributes **0** (no model/session/cache id) and prewarming is **not modelled** (no model-load step) — neither is a factor on this trace.

## What improved / what did not

- constraint_aware vs sla_aware: `ALPHA_WIN` (+25.75% goodput/$).
- Demand is strongly forecastable (bursty, periodic_daily, multi_regime_weekday_weekend; lag-1-day autocorr 0.6762), yet demand-forecasting is NOT where the alpha comes from: with the utilization target held fixed the forecasting ceiling (oracle) is only +6089.77 goodput/$ and realistic forecasters retain ~24% at best (EWMA), while seasonal-time-of-day and 15%-noisy forecasts are net-NEGATIVE.
- The win is a UTILIZATION / target-rho cost-efficiency effect (running hotter while staying SLA-safe), i.e. autoscaling-timing — NOT forecasting accuracy, residency, cache, or prewarming (the latter two are not applicable: no model/session/cache id).
- naive_overprovisioning is the cost-floor anti-pattern (cheap per GPU-hour idle, poor goodput/$); utilization_aware is cheapest but risks tail latency — neither is the buyer-facing headline.

## Honesty / claim discipline

- **No production-savings claim.** Directional simulator/backtest only (`docs/RESULTS.md` §8 gate unmet).
- **No TTFT claim** — Azure 2024 exposes no latency; the SLA budget is a standard interactive SLO decomposition applied identically to all policies.
- **No cache-affinity claim** — no session/prefix key; `cache_affinity_baseline` omitted, constraint_aware gets zero cache benefit.
- Load multipliers replay the real arrival SHAPE; no simulator constant was tuned and no oracle is used as a headline baseline.

