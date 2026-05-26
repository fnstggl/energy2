# Simulator Realism + Benchmark Validation Report

_Generated 2026-05-26 18:55:45Z · seed=42 · steps=24 · **[SANDBOX]**_

> All numbers are **simulator-only, uncalibrated** directional results. Not production savings. See the realism audit verdict below.

**Realism audit overall verdict: `NOT_PRODUCTION_REALISTIC_YET`**

## 1. Realism audit (per-subsystem)

| Subsystem | Verdict | Calibration |
|---|---|---|
| serving | `REALISTIC_ENOUGH_FOR_DEV` | low |
| migration | `REALISTIC_ENOUGH_FOR_DEV` | low |
| telemetry | `NEEDS_REAL_TELEMETRY` | — |
| actions | `REALISTIC_ENOUGH_FOR_DEV` | — |
| energy | `REALISTIC_ENOUGH_FOR_DEV` | low |
| robustness | `REALISTIC_ENOUGH_FOR_DEV` | — |

Headline findings:
- [telemetry] Is telemetry always perfect at the canonical ClusterState level? — ClusterState
- All calibration parameters are uncalibrated priors (none measured on real hardware). Simulator evidence is directional only — not production savings.

## 2. Mean / median delta vs each baseline

Energy-cost delta = `baseline_cost − constraint_aware_cost` per scenario (positive = constraint_aware cheaper). Engine net-savings is penalty-adjusted (migration/cache/SLA/topology/thermal/forecast/churn).

| Baseline | Mean cost delta ($) | Median cost delta ($) |
|---|---|---|
| FIFO | -0.5747 | -0.3514 |
| current_price_only | -0.9527 | -0.5094 |
| greedy_energy | -0.9961 | -0.5094 |
| SLA-aware | -0.5747 | -0.3514 |

Engine-computed constraint_aware net savings across scenarios: mean=1.0433, median=0.0000.

Packing baselines (first-fit / best-fit / FFD / clairvoyant) are reported per packing scenario inside the benchmark JSON `packing_frontier` block; they are analysis-only and never a deployable comparison.

## 3. Per-scenario comparison (constraint_aware)

| scenario | policy | cost $ | net savings | goodput(tok) | p99 ms | queue p95 ms | SLA viol | migrations | churn | thermal | topology | cache hit | FIFO cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carbon_cheap_price_expensive | constraint_aware | 2.752 | 0.000 | 165429210 | 17449 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 2.246 |
| clean_batch_shift_arbitrage | constraint_aware | 5.858 | 26.574 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 4.204 |
| da_rt_basis_blowout | constraint_aware | 6.133 | 6.949 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 4.695 |
| degraded_topology_telemetry | constraint_aware | 3.682 | -0.737 | 90101936 | 15643 | 60000 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 3.682 |
| dram_bound_inference | constraint_aware | 1.869 | 0.000 | 145844703 | 1081053 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.869 |
| energy_price_arbitrage_multiregion | constraint_aware | 8.395 | -2.989 | 225252064 | 246910 | 665 | 600 | 0 | 0.0000 | 0 | 0.807 | 0.373 | 7.398 |
| fragmentation_stranded_capacity | constraint_aware | 4.118 | 0.000 | 189126024 | 72369 | 4 | 300 | 0 | 0.0000 | 0 | 0.878 | 0.140 | 3.767 |
| kv_exhaustion_preemption_storm | constraint_aware | 0.932 | 0.000 | 65203115 | 2632394 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.077 | 0.932 |
| latency_critical_no_energy_shift | constraint_aware | 5.276 | -0.228 | 177047592 | 902903 | 231662 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 3.899 |
| latency_tail_kvcache_pressure | constraint_aware | 3.243 | 0.000 | 247495860 | 15816 | 5 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.273 | 3.243 |
| low_confidence_energy_telemetry | constraint_aware | 4.047 | 0.000 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 3.095 |
| migration_trap_erased_savings | constraint_aware | 3.193 | -1.226 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 2.454 |
| moe_hotspot_nic_saturation | constraint_aware | 4.076 | 0.000 | 20161943 | 13414020 | 239730 | 300 | 0 | 0.0000 | 0 | 0.280 | 0.559 | 4.076 |
| partial_utilization_telemetry | constraint_aware | 1.856 | 0.000 | 165617291 | 3340797 | 222437 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 1.344 |
| prefix_affinity_energy_arbitrage | constraint_aware | 6.245 | 0.460 | 290632106 | 26313 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.897 | 4.736 |
| proxy_bottleneck_ingress | constraint_aware | 3.072 | 0.000 | 249972882 | 1110035 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 2.197 |
| queue_surge_latency_sensitive | constraint_aware | 5.398 | 0.000 | 418688109 | 1096933 | 239730 | 600 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 3.572 |
| rack_density_liquid_cooled | constraint_aware | 39.074 | 0.000 | 1766530990 | 14209 | 0 | 300 | 0 | 0.0000 | 0 | 0.341 | 0.559 | 39.074 |
| rack_density_overload_air | constraint_aware | 39.086 | 0.000 | 1730396395 | 14957 | 0 | 300 | 0 | 0.0000 | 63 | 0.331 | 0.559 | 39.074 |
| scheduler_bound_inference | constraint_aware | 0.988 | 0.000 | 46097013 | 1095846 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.453 | 0.988 |
| startup_heavy_migration_trtllm | constraint_aware | 6.245 | 0.460 | 287698325 | 63509 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.823 | 4.736 |
| tensor_parallel_topology_collapse | constraint_aware | 2.021 | 0.000 | 16477225 | 12014987 | 239730 | 300 | 0 | 0.0000 | 0 | 0.248 | 0.559 | 2.021 |
| thermal_hotspot_mixed_cluster | constraint_aware | 3.502 | 0.000 | 200904019 | 23739 | 65 | 300 | 0 | 0.0000 | 18 | 0.602 | 0.559 | 3.511 |
| topology_fragmentation_h100 | constraint_aware | 7.307 | 0.000 | 142362928 | 294315 | 37 | 300 | 0 | 0.0000 | 0 | 0.645 | 0.559 | 6.963 |
| underutilization_stranded_capacity | constraint_aware | 3.938 | 0.000 | 42013889 | 739919 | 31624 | 300 | 0 | 0.0000 | 0 | 0.845 | 0.140 | 3.586 |
| unsafe_aggressive_consolidation | constraint_aware | 13.344 | -2.136 | 25198864 | 11856087 | 239730 | 300 | 0 | 0.0000 | 161 | 0.272 | 0.559 | 13.344 |

## 4. Safety regressions

None — constraint_aware did not increase hard SLA violations vs FIFO in any scenario.

## 5. Where constraint_aware performs well / poorly

Performs well (improves a binding KPI without SLA regression): carbon_cheap_price_expensive, clean_batch_shift_arbitrage, da_rt_basis_blowout, latency_critical_no_energy_shift, low_confidence_energy_telemetry, migration_trap_erased_savings, partial_utilization_telemetry, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, rack_density_overload_air, startup_heavy_migration_trtllm, thermal_hotspot_mixed_cluster, underutilization_stranded_capacity

Performs poorly (net loss — tail worse with no throughput/thermal/cost relief, or an SLA regression): energy_price_arbitrage_multiregion, fragmentation_stranded_capacity, topology_fragmentation_h100

Known regression: `energy_price_arbitrage_multiregion` — since the energy/carbon realism upgrade, constraint_aware no longer beats greedy_energy by a 5× p99 margin; both saturate. Tracked as a calibration target (see test_serving_realism.py xfail).

## 6. What remains simulator-only / needs real telemetry

- Every calibration parameter is an uncalibrated prior (none measured on real hardware). All KPI numbers are directional.
- Canonical `ClusterState` hardcodes `confidence='high'`/`is_partial=False`; the missing/stale telemetry path is only weakly exercised end-to-end (`telemetry` subsystem verdict = `NEEDS_REAL_TELEMETRY`).
- Next calibration step: run a read-only shadow pilot against real Prometheus/DCGM/K8s telemetry to calibrate the priors and exercise the degraded-telemetry path with real confidence degradation.
