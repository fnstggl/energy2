# Frontier Signal Hypotheses — Economic-Alpha Map

> **Discovery / audit-only.** No model is trained, no production code is
> modified, no savings are claimed, no constants are invented. This document
> is the Phase-1 hypothesis map for the Frontier Discovery Audit v1; the
> evidence sits in `data/external/hf_discovery/frontier_v1/*.json` and the
> verdicts in `docs/FRONTIER_DISCOVERY_AUDIT_V1.md`.
>
> **Read first:** `docs/ECONOMIC_ML_ALPHA_V1.md`,
> `docs/ECONOMIC_OVERLAY_LAYER_V1.md`,
> `docs/HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md`.

## Frame: where a signal can move SLA-safe goodput/$

The binding KPI is

```
sla_safe_goodput_per_dollar = (output_tokens if sla_met else 0)
                             / (gpu_cost + energy_cost + migration_cost
                                + cold_start_cost − cache_value)
```

A new signal earns economic alpha only if it changes the **numerator**
(whether a request stays SLA-safe → goodput) or a **denominator** term
(a cost we can avoid). The Economic ML Alpha Audit v1 already showed the
cost terms are deterministic given inputs, so alpha must come from
**forecasting an uncertain upstream signal** that flips an SLA-safe
decision or avoids a cost. The hypotheses below are scored on that test.

Directionality legend: ↑ = larger signal raises the term; "avoidable" =
a forecast lets the scheduler dodge the cost.

## A. Cold starts

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| model_load_duration_s | adds to e2e → can break SLA; adds GPU-seconds | cold_start_cost (denom) ↑; sla_met (num) | ↑ load → ↑ cost, ↓ goodput | med | server-class vLLM/TGI load timing per (model,GPU) |
| weight_transfer_latency_s | same as load; first-request penalty | cold_start_cost ↑ | ↑ | med | per-(model,GPU,storage) transfer timing |
| graph_capture_latency_s | CUDA-graph capture stall on first call | cold_start_cost ↑ | ↑ | low | engine init telemetry |
| compile_latency_s | torch.compile / TRT build stall | cold_start_cost ↑ | ↑ | low | engine compile timing |
| warmup_duration_s | warm-pool sizing trade-off | cold_start_cost vs idle GPU cost | avoidable | med | warm-pool occupancy + first-token timing |
| scale_from_zero_delay_s | serverless first-request SLA breach | sla_met ↓ | ↑ | med | autoscaler scale-from-zero events |

**Why it matters:** a forecast of cold-start cost per candidate lets the
scheduler keep a model warm exactly when the predicted penalty × arrival
rate exceeds the idle-GPU cost — a direct goodput/$ lever. **Blocked**:
the ML Alpha Audit found 0 measured server-class load-duration rows.

## B. Migration / rerouting

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| cache_loss_pct | KV/prefix lost on migration → re-prefill | migration_cost ↑; cache_value ↓ | ↑ | med | per-migration KV block-hash survival (CC-traces has a *proxy*) |
| reroute_latency_s | in-flight requests stalled | migration_cost ↑; sla_met ↓ | ↑ | low | routing-layer shift timestamps |
| prefix_reuse_destruction | locality lost → TTFT inflation | cache_value ↓; ttft ↑ | ↑ | med | pre/post-migration reuse |
| warmup_after_migration_s | cold replica after shift | cold_start_cost ↑ | ↑ | low | post-migration first-token timing |
| traffic_shift_instability | oscillation → repeated penalties | migration_cost ↑ | ↑ | low | routing decision log |
| migration_veto_label | label: was the migration worth it | direct decision target | n/a | med | operator post-hoc migration outcomes |

**Why it matters:** migration penalty forecasting is the cache-aware
routing lever — only migrate when predicted (cache_value − migration_cost)
> 0. **Blocked**: a real cache-loss *proxy* exists (CC-traces KV hashes)
but is not realized as a per-migration label in any public serving trace.

## C. Queueing

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| queue_wait_s | adds to e2e → breaks SLA | sla_met ↓ | ↑ | **high** | per-request enqueue/dequeue timestamps |
| admission_delay_s | scheduler hold | sla_met ↓ | ↑ | high | admission-control log |
| queue_depth / num_waiting | leading indicator of wait | predicts sla_met | ↑ | high | sampled queue depth |
| replica_saturation | concurrency collapse → tail blow-up | sla_met ↓ | ↑ | med | per-replica in-flight counts |
| routing_contention | proxy bottleneck | e2e ↑ | ↑ | low | proxy/router metrics |

**Why it matters:** queue-wait is the single largest controllable SLA
risk and is **partially available** — CARA (queue features) and AcmeTrace
(real `queue_wait`) exist. The ML Alpha Audit's TTFT signal is the closest
realized proxy. Highest-confidence frontier with the most public support.

## D. Memory pressure

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| peak_vram_gb | placement feasibility; OOM risk | sla_met (OOM→fail) | ↑ | **high (realized)** | per-(model,GPU,quant) peak VRAM |
| kv_eviction_rate | evictions → re-prefill → ttft ↑ | cache_value ↓; ttft ↑ | ↑ | med | KV evictions/s (CARA has it) |
| oom_event_label | hard SLA failure | sla_met ↓ | ↑ | med | OOM crash labels |
| fragmentation | usable-VRAM shrink | placement | ↑ | low | allocator stats |
| active_token_packing | batch efficiency | throughput ↑ | ↓ packing → ↓ goodput | med | per-iter packing telemetry |

**Why it matters:** peak_VRAM is **already shadow-ready** (ML Alpha Audit,
Optimum). KV-eviction is the bridge between memory pressure and cache
value. Memory pressure is the best-covered frontier after queueing.

## E. Serving stability

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| p95/p99 inflation | tail breaks SLA | sla_met ↓ | ↑ | med | per-request latency distribution |
| timeout_risk | request abandoned | sla_met ↓ (goodput→0) | ↑ | med | timeout labels |
| retry_amplification | load multiplies under stress | gpu_cost ↑; sla_met ↓ | ↑ | low | retry counts |
| overload_event | systemic SLA collapse | sla_met ↓ | ↑ | med | overload labels |

**Why it matters:** timeout/failure forecasting flips goodput to 0 — the
biggest single numerator swing. **Partially blocked**: Optimum/Odyn/
llmperf-bedrock expose error_rate/failure counts but no per-request
timeout *labels*; AcmeTrace has job-level FAILED/TIMEOUT states (proxy).

## F. Autoscaling

| Signal | KPI path | Term | Direction | Confidence | Telemetry required |
|---|---|---|---|---|---|
| scale_up_latency_s | new replica too late → SLA breach | sla_met ↓; cold_start_cost ↑ | ↑ | med | autoscaler scale-up timing |
| scale_down_churn | premature down → re-cold-start | cold_start_cost ↑ | ↑ | low | scale-down + re-up events |
| warm_pool_occupancy | idle GPU cost vs cold-start avoidance | gpu_cost vs cold_start_cost | trade-off | med | pool occupancy time-series |
| oscillation_frequency | thrash → repeated penalties | cold_start_cost ↑ | ↑ | low | scaling event log |

**Why it matters:** autoscaling ties cold-start + queueing together
(scale early enough to avoid queue blow-up without paying idle GPU).
**Fully blocked**: **no public HF dataset exposes autoscaling telemetry**
(Frontier Discovery v1 found 0). Google-Cluster/Borg gives a job-level
*proxy* only.

## Summary of hypothesis confidence vs public-data availability

| Category | Economic leverage (hypothesis) | Public-data availability today |
|---|---|---|
| Queueing | **Very High** | Partial (CARA, AcmeTrace) |
| Memory pressure | High | Partial→Good (Optimum peak_VRAM realized; CARA KV evictions) |
| Serving stability (timeout) | **Very High** | Weak (proxies only; no per-request labels) |
| Cold start | High | **None measured** (RL "cold-start" datasets are unrelated) |
| Migration | High | **None measured** (CC-traces proxy not realized) |
| Autoscaling | Medium-High | **None** |

The pattern is stark: the categories with the **highest economic leverage
that are still un-forecasted** (timeout, cold-start, migration,
autoscaling) are exactly the ones with **no measured public data** —
they remain `blocked_by_pilot_telemetry`, confirming the Economic ML
Alpha Audit v1 verdict.
