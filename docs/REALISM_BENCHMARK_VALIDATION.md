# Simulator Realism + Benchmark Validation Report

_Generated 2026-05-28 17:08:03Z · seed=42 · steps=24 · **[SANDBOX]**_

> All numbers are **simulator-only, uncalibrated** directional results. Not production savings. See the realism audit verdict below.

> **Primary KPI:** `sla_safe_goodput_per_infrastructure_dollar` (Section 2). Raw energy cost is **not** the primary metric; it is a diagnostic. Secondary KPIs (p99, queue, thermal, topology, …) are constraints/vetoes, never folded into the primary KPI.

**Realism audit overall verdict: `NOT_PRODUCTION_REALISTIC_YET`**

## 1. Realism audit (per-subsystem)

| Subsystem | Verdict | Calibration |
|---|---|---|
| serving | `REALISTIC_ENOUGH_FOR_DEV` | low |
| migration | `REALISTIC_ENOUGH_FOR_DEV` | low |
| telemetry | `REALISTIC_ENOUGH_FOR_DEV` | — |
| actions | `REALISTIC_ENOUGH_FOR_DEV` | — |
| energy | `REALISTIC_ENOUGH_FOR_DEV` | low |
| robustness | `REALISTIC_ENOUGH_FOR_DEV` | — |

Headline findings:
- All calibration parameters are uncalibrated priors (none measured on real hardware). Simulator evidence is directional only — not production savings.

## 2. Primary KPI: SLA-safe goodput per infrastructure dollar

This is the canonical benchmark metric. Higher is better. The denominator is `gpu_infra_cost + energy_cost + network_cost`; the numerator is tokens that met their workload's SLO (queue `timeout_rate_pct` filter). Secondary KPIs are NOT folded in — they're tracked separately below as constraints / diagnostics. GPU infra cost typically dominates electricity by 50–200×.

Per-policy aggregates across all scenarios:

| Policy | Mean goodput / $ | Median goodput / $ |
|---|---|---|
| FIFO | 414803.6616 | 459570.4658 |
| current_price_only | 414670.8699 | 449752.3827 |
| greedy_energy | 407800.6395 | 449752.3827 |
| SLA-aware | 414803.6616 | 459570.4658 |
| constraint_aware | 414912.9125 | 454858.0190 |

Scenarios where constraint_aware **loses** the canonical KPI to a baseline (honest, not hidden):
- vs FIFO: energy_price_arbitrage_multiregion, latency_critical_no_energy_shift, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, startup_heavy_migration_trtllm
- vs current_price_only: energy_price_arbitrage_multiregion, latency_critical_no_energy_shift, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, startup_heavy_migration_trtllm, unsafe_aggressive_consolidation
- vs greedy_energy: energy_price_arbitrage_multiregion, latency_critical_no_energy_shift, proxy_bottleneck_ingress, queue_surge_latency_sensitive, unsafe_aggressive_consolidation
- vs SLA-aware: energy_price_arbitrage_multiregion, latency_critical_no_energy_shift, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, startup_heavy_migration_trtllm

Per-scenario primary KPI (SLA-safe goodput per $):

| scenario | FIFO | current_price_only | greedy_energy | SLA-aware | constraint_aware |
|---|---|---|---|---|---|
| carbon_cheap_price_expensive | 466,101 | 466,101 | 466,101 | 466,101 | 465,834 |
| clean_batch_shift_arbitrage | 458,321 | 450,074 | 450,074 | 458,321 | 459,235 |
| da_rt_basis_blowout | 456,085 | 442,536 | 442,536 | 456,085 | 456,985 |
| degraded_topology_telemetry | 230,226 | 225,143 | 225,143 | 230,226 | 230,226 |
| dram_bound_inference | 376,142 | 376,142 | 376,142 | 376,142 | 376,142 |
| energy_price_arbitrage_multiregion | 338,274 | 402,882 | 274,801 | 338,274 | 228,634 |
| fragmentation_stranded_capacity | 295,886 | 295,886 | 295,886 | 295,886 | 295,886 |
| kv_exhaustion_preemption_storm | 361,584 | 361,584 | 361,584 | 361,584 | 361,584 |
| latency_critical_no_energy_shift | 495,968 | 495,968 | 495,968 | 495,968 | 472,692 |
| latency_tail_kvcache_pressure | 424,896 | 424,896 | 424,896 | 424,896 | 424,896 |
| low_confidence_energy_telemetry | 463,453 | 449,431 | 449,431 | 463,453 | 463,453 |
| migration_trap_erased_savings | 466,467 | 451,983 | 451,983 | 466,467 | 467,347 |
| moe_hotspot_nic_saturation | 34,515 | 34,515 | 34,515 | 34,515 | 34,515 |
| partial_utilization_telemetry | 478,725 | 478,725 | 478,725 | 478,725 | 478,725 |
| prefix_affinity_energy_arbitrage | 867,411 | 849,608 | 824,206 | 867,411 | 821,767 |
| proxy_bottleneck_ingress | 495,960 | 495,960 | 495,960 | 495,960 | 452,731 |
| queue_surge_latency_sensitive | 477,029 | 477,029 | 477,029 | 477,029 | 439,173 |
| rack_density_liquid_cooled | 696,917 | 696,917 | 696,917 | 696,917 | 696,917 |
| rack_density_overload_air | 667,543 | 667,543 | 667,543 | 667,543 | 680,129 |
| scheduler_bound_inference | 237,644 | 237,644 | 237,644 | 237,644 | 237,644 |
| startup_heavy_migration_trtllm | 859,845 | 843,426 | 818,284 | 859,845 | 813,348 |
| tensor_parallel_topology_collapse | 42,463 | 42,463 | 42,463 | 42,463 | 42,463 |
| thermal_hotspot_mixed_cluster | 565,817 | 565,817 | 565,817 | 565,817 | 830,781 |
| topology_fragmentation_h100 | 460,820 | 460,820 | 460,820 | 460,820 | 460,820 |
| underutilization_stranded_capacity | 45,426 | 45,426 | 45,426 | 45,426 | 74,432 |
| unsafe_aggressive_consolidation | 21,379 | 42,924 | 42,924 | 21,379 | 21,379 |

## 3. Mean / median delta vs each baseline (secondary — raw cost only)

These are the **legacy** raw-cost deltas, retained for diagnostic purposes only. They are NOT the primary KPI: a policy can be cheap on raw energy AND lose on `sla_safe_goodput_per_infra_dollar` (see Section 2).

Energy-cost delta = `baseline_cost − constraint_aware_cost` per scenario (positive = constraint_aware cheaper). Engine net-savings is penalty-adjusted (migration/cache/SLA/topology/thermal/forecast/churn).

| Baseline | Mean cost delta ($) | Median cost delta ($) |
|---|---|---|
| FIFO | -0.2797 | 0.0000 |
| current_price_only | -0.6577 | -0.0234 |
| greedy_energy | -0.7010 | -0.0234 |
| SLA-aware | -0.2797 | 0.0000 |

Engine-computed constraint_aware net savings across scenarios: mean=0.2615, median=0.0000.

Packing baselines (first-fit / best-fit / FFD / clairvoyant) are reported per packing scenario inside the benchmark JSON `packing_frontier` block; they are analysis-only and never a deployable comparison.

## 4. Per-scenario comparison (constraint_aware)

| scenario | policy | goodput/$ (PRIMARY) | $/SLA-tok | SLA-compliant goodput | infra $ | GPU $ | energy $ | raw cost $ | raw tokens | p99 ms | queue p95 ms | SLA viol | migrations | churn | thermal | topology | cache hit | telemetry conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carbon_cheap_price_expensive | constraint_aware | 465,834 | 2.147e-06 | 45,764,025 | 98.24 | 96.00 | 2.241 | 2.241 | 91528059 | 14730 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.85 |
| clean_batch_shift_arbitrage | constraint_aware | 459,235 | 2.178e-06 | 46,008,283 | 100.18 | 96.00 | 4.185 | 4.185 | 92016576 | 14744 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.85 |
| da_rt_basis_blowout | constraint_aware | 456,985 | 2.188e-06 | 46,008,283 | 100.68 | 96.00 | 4.678 | 4.678 | 92016576 | 14744 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.85 |
| degraded_topology_telemetry | constraint_aware | 230,226 | 4.344e-06 | 45,050,961 | 195.68 | 192.00 | 3.682 | 3.682 | 90101936 | 15643 | 60000 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 0.34 (partial) |
| dram_bound_inference | constraint_aware | 376,142 | 2.659e-06 | 72,922,344 | 193.87 | 192.00 | 1.869 | 1.869 | 145844703 | 1081053 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 |
| energy_price_arbitrage_multiregion | constraint_aware | 228,634 | 4.374e-06 | 111,437,017 | 487.40 | 480.00 | 7.403 | 7.403 | 222874064 | 19747 | 179 | 600 | 0 | 0.0000 | 0 | 0.827 | 0.373 | 0.65 |
| fragmentation_stranded_capacity | constraint_aware | 295,886 | 3.380e-06 | 100,532,433 | 339.77 | 336.00 | 3.767 | 3.767 | 196806844 | 11997 | 0 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.140 | 0.22 |
| kv_exhaustion_preemption_storm | constraint_aware | 361,584 | 2.766e-06 | 35,049,137 | 96.93 | 96.00 | 0.932 | 0.932 | 65203115 | 2632394 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.077 | 0.99 |
| latency_critical_no_energy_shift | constraint_aware | 472,692 | 2.116e-06 | 88,523,791 | 187.28 | 182.00 | 5.276 | 5.276 | 177047592 | 902903 | 231662 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 |
| latency_tail_kvcache_pressure | constraint_aware | 424,896 | 2.354e-06 | 123,747,924 | 291.24 | 288.00 | 3.243 | 3.243 | 247495860 | 15816 | 5 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.273 | 0.62 |
| low_confidence_energy_telemetry | constraint_aware | 463,453 | 2.158e-06 | 45,925,649 | 99.09 | 96.00 | 3.095 | 3.095 | 91851314 | 14744 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.29 (partial) |
| migration_trap_erased_savings | constraint_aware | 467,347 | 2.140e-06 | 46,008,283 | 98.45 | 96.00 | 2.446 | 2.446 | 92016576 | 14744 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.85 |
| moe_hotspot_nic_saturation | constraint_aware | 34,515 | 2.897e-05 | 10,080,964 | 292.08 | 288.00 | 4.076 | 4.076 | 20161943 | 13414020 | 239730 | 300 | 0 | 0.0000 | 0 | 0.280 | 0.559 | 1.00 |
| partial_utilization_telemetry | constraint_aware | 478,725 | 2.089e-06 | 46,601,134 | 97.34 | 96.00 | 1.344 | 1.344 | 93202280 | 20686 | 845 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 0.29 (partial) |
| prefix_affinity_energy_arbitrage | constraint_aware | 821,767 | 1.217e-06 | 229,474,498 | 279.25 | 273.00 | 6.245 | 6.245 | 290632106 | 26313 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.897 | 0.99 |
| proxy_bottleneck_ingress | constraint_aware | 452,731 | 2.209e-06 | 124,986,434 | 276.07 | 273.00 | 3.072 | 3.072 | 249972882 | 1110035 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 |
| queue_surge_latency_sensitive | constraint_aware | 439,173 | 2.277e-06 | 231,607,066 | 527.37 | 522.00 | 5.371 | 5.371 | 414438830 | 1001541 | 239730 | 600 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.63 |
| rack_density_liquid_cooled | constraint_aware | 696,917 | 1.435e-06 | 1,632,927,143 | 2343.07 | 2304.00 | 39.074 | 39.074 | 1766530990 | 14209 | 0 | 300 | 0 | 0.0000 | 0 | 0.341 | 0.559 | 0.28 |
| rack_density_overload_air | constraint_aware | 680,129 | 1.470e-06 | 1,593,601,496 | 2343.09 | 2304.00 | 39.086 | 39.086 | 1730396395 | 14957 | 0 | 300 | 0 | 0.0000 | 63 | 0.331 | 0.559 | 0.34 |
| scheduler_bound_inference | constraint_aware | 237,644 | 4.208e-06 | 23,048,502 | 96.99 | 96.00 | 0.988 | 0.988 | 46097013 | 1095846 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.453 | 1.00 |
| startup_heavy_migration_trtllm | constraint_aware | 813,348 | 1.229e-06 | 227,123,423 | 279.25 | 273.00 | 6.245 | 6.245 | 287698325 | 63509 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.823 | 0.99 |
| tensor_parallel_topology_collapse | constraint_aware | 42,463 | 2.355e-05 | 8,238,607 | 194.02 | 192.00 | 2.021 | 2.021 | 16477225 | 12014987 | 239730 | 300 | 0 | 0.0000 | 0 | 0.248 | 0.559 | 1.00 |
| thermal_hotspot_mixed_cluster | constraint_aware | 830,781 | 1.204e-06 | 162,419,183 | 195.50 | 192.00 | 3.502 | 3.502 | 200904019 | 23739 | 65 | 300 | 0 | 0.0000 | 18 | 0.602 | 0.559 | 0.34 |
| topology_fragmentation_h100 | constraint_aware | 460,820 | 2.170e-06 | 135,924,754 | 294.96 | 288.00 | 6.963 | 6.963 | 271849521 | 14809 | 0 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 |
| underutilization_stranded_capacity | constraint_aware | 74,432 | 1.344e-05 | 19,786,409 | 265.83 | 262.00 | 3.832 | 3.832 | 36304195 | 568799 | 19369 | 300 | 0 | 0.0000 | 0 | 0.848 | 0.140 | 0.81 |
| unsafe_aggressive_consolidation | constraint_aware | 21,379 | 4.678e-05 | 12,599,426 | 589.34 | 576.00 | 13.344 | 13.344 | 25198864 | 11856087 | 239730 | 300 | 0 | 0.0000 | 161 | 0.272 | 0.559 | 1.00 |

## 5. Safety regressions

None — constraint_aware did not increase hard SLA violations vs FIFO in any scenario.

## 6. Where constraint_aware performs well / poorly

Performs well (improves a binding KPI without SLA regression): latency_critical_no_energy_shift, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, rack_density_overload_air, startup_heavy_migration_trtllm, thermal_hotspot_mixed_cluster, underutilization_stranded_capacity

Performs poorly (net loss — tail worse with no throughput/thermal/cost relief, or an SLA regression): energy_price_arbitrage_multiregion

`greedy_energy` headline property (RESTORED): on `energy_price_arbitrage_multiregion`, greedy_energy's aggressive migration blows up p99 >5× past constraint_aware (now deterministic across pytest and a plain interpreter — the prior xfail was a YAML/builtin scenario-drift determinism bug, not a model regression; see test_scenario_source_parity.py).

Honest open weakness (energy scenario): constraint_aware is still the most EXPENSIVE policy on raw energy cost here and does not beat current_price_only (which is cheaper AND has fewer SLA violations). Root cause: the engine still applies some queue-relief scaling to BATCH workloads (which tolerate queueing), wasting energy. The constraint-dominance guard reduces but does not eliminate this; a full fix needs workload-class (priority_tier/latency_sensitive) propagated into the canonical InferenceServiceState so the engine can apply the spec's workload-aware priorities. Reported, not hidden.

## 7. What remains simulator-only / needs real telemetry

- Every calibration parameter is an uncalibrated prior (none measured on real hardware). All KPI numbers are directional.
- Telemetry truth (Mission 1, FIXED): the canonical `ClusterState` now derives provenance confidence + `is_partial` from the simulator's per-subsystem tiers, so degraded-telemetry scenarios report low/partial confidence and the engine force-KEEPs (telemetry subsystem verdict graduated to REALISTIC_ENOUGH_FOR_DEV). The tiers themselves remain uncalibrated heuristics.
- Next calibration step: run a read-only shadow pilot against real Prometheus/DCGM/K8s telemetry to calibrate the priors and the confidence model (C = R·F·K·S·N) against measured staleness/coverage/noise, and propagate workload class into InferenceServiceState for workload-aware action selection.
