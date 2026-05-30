# Alibaba GPU Backtest Results — CANONICAL_TRACE_BACKTEST_ALIBABA_GPU_V2023_FRAGMENTATION_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` and `docs/PUBLIC_TRACE_BACKTESTS.md` first.

## Provenance

- **Source:** `pod:data/external/alibaba_gpu/raw/openb_pod_list_default.csv node:data/external/alibaba_gpu/raw/openb_node_list_gpu_node.csv`
- **Dataset:** Alibaba `cluster-trace-gpu-v2023` (https://github.com/alibaba/clusterdata) — pod list `openb_pod_list_default.csv` + GPU node inventory `openb_node_list_gpu_node.csv`.
- Alibaba public data is a **public dataset, not customer telemetry**.

## Available vs missing fields (honest)

Pod schema: `name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time`. Node schema: `sn,cpu_milli,memory_mib,gpu,model`.

- **Missing (stated):** gpu_utilization_timeseries, gpu_memory_gb, per_pod_node_placement, deadline, user_or_group.
- `gpu_milli` = thousandths of a GPU (sharing); `num_gpu` = whole GPUs. `gpu_spec` empty in the default pod list → `gpu_type=None`.
- **No GPU utilization time-series** in this dataset → `NormalizedGPUUtilizationSample` list is empty (0 samples).
- **No GPU-memory column** → `gpu_memory_gb=None`. **No per-pod node placement** in the default pod list → placement is what the backtest computes. **No deadline / user** columns.

## Trace summary

- Jobs: **6,282** (5,195 GPU jobs, 1,087 CPU-only)  ·  failed: 0
- Time range: 12898342s (149.3 days)
- Status distribution: {'Pending': 897, 'Running': 5193, 'Succeeded': 192}
- num_gpu distribution: {0: 1087, 1: 5176, 2: 7, 4: 1, 8: 11}
- Job duration s p50/p95/p99: 613 / 19189 / 170469
- Queue wait s p50/p95/p99 (trace-observed): 3.0 / 245.0 / 1217.0
- Fleet: **1,213 GPU nodes**, **6,212 GPUs**, by model {'A10': 2, 'G2': 4392, 'G3': 312, 'P100': 265, 'T4': 842, 'V100M16': 195, 'V100M32': 204}
- GPU demand / capacity ratio: **0.6776**
- GPU utilization samples: **0** (no utilization series in v2023).

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. **goodput_unit = `completed_gpu_job_work` (token_equivalent = effective_GPU × duration)** — explicitly NOT inference output tokens. A job that cannot be placed is **stranded** (the SLA-violation analogue). Infra cost bills every **active** node (≥1 placed job) for the trace window at a documented per-GPU-type price, so fragmentation/spreading (more under-filled active nodes) costs more per unit work. Same packing physics, fleet, prices and window for all policies — only the placement decision differs.

**Headline baseline = `best_fit` (a real PACKING baseline, NOT fifo).** FIFO is the do-nothing sanity baseline only (`docs/RESULTS.md` §3).

| policy | goodput/$ | placed | stranded | active nodes | GPU util % | fragmentation | infra $ |
|---|---|---|---|---|---|---|---|
| fifo | 5.27 | 5185 | 10 | 1208 | 66.5 | 0.2863 | 31,980,345 |
| first_fit | 7.54 | 5195 | 0 | 891 | 90.4 | 0.0911 | 24,140,721 |
| best_fit *(headline)* | 7.77 | 5195 | 0 | 1023 | 89.7 | 0.0970 | 23,422,098 |
| first_fit_decreasing | 7.48 | 5195 | 0 | 896 | 89.7 | 0.0835 | 24,344,301 |
| greedy_packing | 7.55 | 5195 | 0 | 1038 | 87.5 | 0.0896 | 24,110,256 |
| constraint_aware **(CA)** | 8.24 | 5195 | 0 | 1004 | 89.8 | 0.0998 | 22,095,961 |

## Packing baselines are EXECUTABLE (not analysis-only)

`first_fit`, `best_fit`, `first_fit_decreasing` (FFD) and `greedy_packing` are run as real placement algorithms over the normalized trace/fleet — closing the prior analysis-only gap. `fifo` is a naive round-robin spread (no consolidation) and is the sanity baseline only.

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)

- **Outcome:** `ALPHA_WIN`  ·  margin vs `best_fit`: **+6.00%** on goodput/$
- **Sanity vs FIFO:** constraint_aware beats naive FIFO (+56.43%).

### What improved / what did not

- vs `best_fit` (strongest packing baseline): goodput/$ 8.24 vs 7.77 (+6.00%). constraint_aware adds **heterogeneous GPU-type price-awareness** (route to the cheapest adequate GPU) + big-job reservation on top of best-fit consolidation — infra $ 22,095,961 vs 23,422,098.
- vs naive `fifo` spread: +56.43% — consolidation avoids powering ~204 extra under-filled nodes.
- Where it can lose / tie: when the fleet is homogeneous (no cheaper GPU type to exploit) or under-subscribed, best-fit already packs near-optimally and constraint_aware's edge shrinks to a tie — reported honestly.

## Fleet-contention sensitivity (deterministic node subsets)

Replays the same job set onto progressively smaller fleets so fragmentation/stranding pressure rises transparently (no single cherry-picked fleet). `node_fraction=1.0` is the full fleet.

| node × | fleet GPUs | fifo gpd | best_fit gpd | constraint_aware gpd | CA vs headline | CA stranded | fifo stranded |
|---|---|---|---|---|---|---|---|
| 1× | 6,212 | 5.27 | 7.77 | 8.24 | +6.00% | 0 | 10 |
| 0.7× | 4,303 | 7.63 | 8.24 | 8.24 | -0.02% | 402 | 414 |
| 0.5× | 3,017 | 10.62 | 11.46 | 11.45 | -0.10% | 1813 | 1589 |
| 0.35× | 2,073 | 14.65 | 14.64 | 14.59 | -0.33% | 2794 | 2637 |

## Honest limits

- **Static** fractional bin-packing (openb-style); no temporal migration/churn (churn = 0). Goodput is `completed_gpu_job_work` (token_equivalent), NOT inference tokens. Durations are partly censored at the trace window.
- GPU-hour prices per model are documented public priors (±50%), identical across policies. Override before any external claim (`docs/RESULTS.md` §8).
- Alibaba public data is **not customer telemetry**. No GPU utilization, GPU-memory, deadline, or user columns exist in v2023.
- **Not production-real savings.** Directional simulator result only.

