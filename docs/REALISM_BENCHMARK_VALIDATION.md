# Simulator Realism + Benchmark Validation Report

_Generated 2026-05-27 02:17:11Z · seed=42 · steps=24 · **[SANDBOX]**_

> All numbers are **simulator-only, uncalibrated** directional results. Not production savings. See the realism audit verdict below.

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

## 2. Mean / median delta vs each baseline

Energy-cost delta = `baseline_cost − constraint_aware_cost` per scenario (positive = constraint_aware cheaper). Engine net-savings is penalty-adjusted (migration/cache/SLA/topology/thermal/forecast/churn).

| Baseline | Mean cost delta ($) | Median cost delta ($) |
|---|---|---|
| FIFO | -0.4829 | 0.0000 |
| current_price_only | -0.8609 | -0.1853 |
| greedy_energy | -0.9042 | -0.1853 |
| SLA-aware | -0.4829 | 0.0000 |

Engine-computed constraint_aware net savings across scenarios: mean=1.0505, median=0.0000.

Packing baselines (first-fit / best-fit / FFD / clairvoyant) are reported per packing scenario inside the benchmark JSON `packing_frontier` block; they are analysis-only and never a deployable comparison.

## 3. Per-scenario comparison (constraint_aware)

| scenario | policy | cost $ | net savings | goodput(tok) | p99 ms | queue p95 ms | SLA viol | migrations | churn | thermal | topology | cache hit | telemetry conf | FIFO cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carbon_cheap_price_expensive | constraint_aware | 2.752 | 0.000 | 165429210 | 17449 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 | 2.246 |
| clean_batch_shift_arbitrage | constraint_aware | 5.858 | 26.574 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 | 4.204 |
| da_rt_basis_blowout | constraint_aware | 6.133 | 6.949 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 | 4.695 |
| degraded_topology_telemetry | constraint_aware | 3.682 | -0.737 | 90101936 | 15643 | 60000 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 0.34 (partial) | 3.682 |
| dram_bound_inference | constraint_aware | 1.869 | 0.000 | 145844703 | 1081053 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 | 1.869 |
| energy_price_arbitrage_multiregion | constraint_aware | 8.298 | -2.800 | 221312266 | 375194 | 907 | 600 | 0 | 0.0000 | 0 | 0.806 | 0.373 | 0.55 | 7.398 |
| fragmentation_stranded_capacity | constraint_aware | 3.767 | 0.000 | 196806844 | 11997 | 0 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.140 | 0.22 | 3.767 |
| kv_exhaustion_preemption_storm | constraint_aware | 0.932 | 0.000 | 65203115 | 2632394 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.077 | 0.99 | 0.932 |
| latency_critical_no_energy_shift | constraint_aware | 5.276 | -0.228 | 177047592 | 902903 | 231662 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 | 3.899 |
| latency_tail_kvcache_pressure | constraint_aware | 3.243 | 0.000 | 247495860 | 15816 | 5 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.273 | 0.62 | 3.243 |
| low_confidence_energy_telemetry | constraint_aware | 3.095 | 0.000 | 91851314 | 14744 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.29 (partial) | 3.095 |
| migration_trap_erased_savings | constraint_aware | 3.193 | -1.226 | 164986559 | 20022 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.99 | 2.454 |
| moe_hotspot_nic_saturation | constraint_aware | 4.076 | 0.000 | 20161943 | 13414020 | 239730 | 300 | 0 | 0.0000 | 0 | 0.280 | 0.559 | 1.00 | 4.076 |
| partial_utilization_telemetry | constraint_aware | 1.344 | 0.000 | 93202280 | 20686 | 845 | 300 | 0 | 0.0000 | 0 | 0.600 | 0.559 | 0.29 (partial) | 1.344 |
| prefix_affinity_energy_arbitrage | constraint_aware | 6.245 | 0.460 | 290632106 | 26313 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.897 | 0.99 | 4.736 |
| proxy_bottleneck_ingress | constraint_aware | 3.072 | 0.000 | 249972882 | 1110035 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 | 2.197 |
| queue_surge_latency_sensitive | constraint_aware | 5.371 | 0.000 | 414438830 | 1001541 | 239730 | 600 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 0.63 | 3.572 |
| rack_density_liquid_cooled | constraint_aware | 39.074 | 0.000 | 1766530990 | 14209 | 0 | 300 | 0 | 0.0000 | 0 | 0.341 | 0.559 | 0.28 | 39.074 |
| rack_density_overload_air | constraint_aware | 39.086 | 0.000 | 1730396395 | 14957 | 0 | 300 | 0 | 0.0000 | 63 | 0.331 | 0.559 | 0.34 | 39.074 |
| scheduler_bound_inference | constraint_aware | 0.988 | 0.000 | 46097013 | 1095846 | 239730 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.453 | 1.00 | 0.988 |
| startup_heavy_migration_trtllm | constraint_aware | 6.245 | 0.460 | 287698325 | 63509 | 60000 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.823 | 0.99 | 4.736 |
| tensor_parallel_topology_collapse | constraint_aware | 2.021 | 0.000 | 16477225 | 12014987 | 239730 | 300 | 0 | 0.0000 | 0 | 0.248 | 0.559 | 1.00 | 2.021 |
| thermal_hotspot_mixed_cluster | constraint_aware | 3.502 | 0.000 | 200904019 | 23739 | 65 | 300 | 0 | 0.0000 | 18 | 0.602 | 0.559 | 0.34 | 3.511 |
| topology_fragmentation_h100 | constraint_aware | 6.963 | 0.000 | 271849521 | 14809 | 0 | 300 | 0 | 0.0000 | 0 | 1.000 | 0.559 | 1.00 | 6.963 |
| underutilization_stranded_capacity | constraint_aware | 3.832 | 0.000 | 36304195 | 568799 | 19369 | 300 | 0 | 0.0000 | 0 | 0.848 | 0.140 | 0.81 | 3.586 |
| unsafe_aggressive_consolidation | constraint_aware | 13.344 | -2.136 | 25198864 | 11856087 | 239730 | 300 | 0 | 0.0000 | 161 | 0.272 | 0.559 | 1.00 | 13.344 |

## 4. Safety regressions

None — constraint_aware did not increase hard SLA violations vs FIFO in any scenario.

## 5. Where constraint_aware performs well / poorly

Performs well (improves a binding KPI without SLA regression): carbon_cheap_price_expensive, clean_batch_shift_arbitrage, da_rt_basis_blowout, latency_critical_no_energy_shift, migration_trap_erased_savings, prefix_affinity_energy_arbitrage, proxy_bottleneck_ingress, queue_surge_latency_sensitive, rack_density_overload_air, startup_heavy_migration_trtllm, thermal_hotspot_mixed_cluster, underutilization_stranded_capacity

Performs poorly (net loss — tail worse with no throughput/thermal/cost relief, or an SLA regression): energy_price_arbitrage_multiregion

`greedy_energy` headline property (RESTORED): on `energy_price_arbitrage_multiregion`, greedy_energy's aggressive migration blows up p99 >5× past constraint_aware (now deterministic across pytest and a plain interpreter — the prior xfail was a YAML/builtin scenario-drift determinism bug, not a model regression; see test_scenario_source_parity.py).

Honest open weakness (energy scenario): constraint_aware is still the most EXPENSIVE policy on raw energy cost here and does not beat current_price_only (which is cheaper AND has fewer SLA violations). Root cause: the engine still applies some queue-relief scaling to BATCH workloads (which tolerate queueing), wasting energy. The constraint-dominance guard reduces but does not eliminate this; a full fix needs workload-class (priority_tier/latency_sensitive) propagated into the canonical InferenceServiceState so the engine can apply the spec's workload-aware priorities. Reported, not hidden.

## 6. What remains simulator-only / needs real telemetry

- Every calibration parameter is an uncalibrated prior (none measured on real hardware). All KPI numbers are directional.
- Telemetry truth (Mission 1, FIXED): the canonical `ClusterState` now derives provenance confidence + `is_partial` from the simulator's per-subsystem tiers, so degraded-telemetry scenarios report low/partial confidence and the engine force-KEEPs (telemetry subsystem verdict graduated to REALISTIC_ENOUGH_FOR_DEV). The tiers themselves remain uncalibrated heuristics.
- Next calibration step: run a read-only shadow pilot against real Prometheus/DCGM/K8s telemetry to calibrate the priors and the confidence model (C = R·F·K·S·N) against measured staleness/coverage/noise, and propagate workload class into InferenceServiceState for workload-aware action selection.
