# Simulator Realism + Benchmark Validation Report

_Generated 2026-05-28 20:16:33Z · seed=42 · steps=24 · **[SANDBOX]**_

> All numbers are **simulator-only, uncalibrated** directional results. Not production savings. See the realism audit verdict below.

> **Primary KPI:** `sla_safe_goodput_per_infrastructure_dollar`. Per-workload comparison uses the *workload-relevant strong baseline*, not FIFO. FIFO is the sanity-only baseline. Telemetry-failsafe scenarios are scored on KEEP-correctness, not alpha.

> ML forecasting is a later phase, after the optimizer has the right objective and workload-aware decision rules. Simulator results remain not production savings claims.

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

## A. Overall policy comparison

Median is the headline aggregate (robust to scenario heterogeneity); mean is shown as a secondary number. Telemetry-failsafe scenarios are excluded from these economic aggregates and reported separately (see end of section A).

| Policy | Mean goodput/$ | Median goodput/$ | CA ALPHA_WIN when this=headline | CA SAFETY_WIN when this=headline | SLA regressions vs FIFO |
|---|---|---|---|---|---|
| FIFO | 417,934 | 458,321 | 1 | 0 | 0 |
| current_price_only | 418,615 | 450,074 | 0 | 0 | 0 |
| greedy_energy | 410,849 | 450,074 | 0 | 0 | 0 |
| SLA-aware | 417,934 | 458,321 | 2 | 0 | 0 |
| constraint_aware | 418,058 | 452,731 | 0 | 0 | 0 |

**Alpha/safety counters:** alpha_wins=3 · safety_wins=0 · correct_keeps=3 · economic_losses=6 · SLA_regressions=0 · catastrophic_baseline_avoidances=3

Telemetry-failsafe scenarios (KEEP-correctness, not alpha): degraded_topology_telemetry, low_confidence_energy_telemetry, partial_utilization_telemetry

## B. Per-workload-type comparison

| Workload type | Scenarios | Policy | Mean goodput/$ | Median goodput/$ |
|---|---|---|---|---|
| batch_training | 6 | FIFO | 441,011 | 459,570 |
| batch_training | 6 | current_price_only | 445,733 | 451,028 |
| batch_training | 6 | greedy_energy | 424,386 | 451,028 |
| batch_training | 6 | SLA-aware | 441,011 | 459,570 |
| batch_training | 6 | constraint_aware | 423,143 | 460,027 |
| inference_critical | 3 | FIFO | 427,483 | 424,896 |
| inference_critical | 3 | current_price_only | 427,483 | 424,896 |
| inference_critical | 3 | greedy_energy | 427,483 | 424,896 |
| inference_critical | 3 | SLA-aware | 427,483 | 424,896 |
| inference_critical | 3 | constraint_aware | 419,724 | 424,896 |
| inference_standard | 14 | FIFO | 405,998 | 426,585 |
| inference_standard | 14 | current_price_only | 405,093 | 426,585 |
| inference_standard | 14 | greedy_energy | 401,483 | 426,585 |
| inference_standard | 14 | SLA-aware | 405,998 | 426,585 |
| inference_standard | 14 | constraint_aware | 415,522 | 407,657 |

## C. Per-scenario outcome

Headline-baseline column is the *workload-relevant strong baseline*, not FIFO. Outcome compares constraint_aware against that headline.

| scenario | workload type | intent | goodput_unit | headline baseline | rationale | outcome | margin % | loss reasons | notes |
|---|---|---|---|---|---|---|---|---|---|
| carbon_cheap_price_expensive | batch_training | energy_arbitrage | token_equivalent | current_price_only | strongest_safe_relevant_baseline:current_price_only | TIE | -0.06 | — | within tie band, no material safety edge |
| clean_batch_shift_arbitrage | batch_training | energy_arbitrage | token_equivalent | sla_aware | strongest_safe_relevant_baseline:sla_aware | TIE | +0.20 | — | within tie band, no material safety edge |
| da_rt_basis_blowout | batch_training | energy_arbitrage | token_equivalent | sla_aware | strongest_safe_relevant_baseline:sla_aware | TIE | +0.20 | — | within tie band, no material safety edge |
| degraded_topology_telemetry | telemetry_fail_safe | safety_keep | telemetry_correct_keeps | fifo | telemetry_failsafe_correctness | KEEP_CORRECT | +0.00 | — | telemetry-failsafe scenario; KEEP matched FIFO and no SLA regression |
| dram_bound_inference | inference_standard | fragmentation_packing | tokens | fifo | no_packing_baseline_computed_for_this_run | TIE | +0.00 | — | within tie band, no material safety edge |
| energy_price_arbitrage_multiregion | batch_training | energy_arbitrage | token_equivalent | current_price_only | strongest_safe_relevant_baseline:current_price_only | LOSS | -43.25 | missing_candidate_action, missing_forecast_lookahead | constraint_aware emitted no relevant action type |
| fragmentation_stranded_capacity | inference_standard | fragmentation_packing | tokens | fifo | no_packing_baseline_computed_for_this_run | TIE | +0.00 | — | within tie band, no material safety edge |
| kv_exhaustion_preemption_storm | inference_critical | memory_pressure_relief | tokens | sla_aware | interactive_workload_prefers_sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| latency_critical_no_energy_shift | inference_critical | energy_arbitrage | tokens | current_price_only | strongest_safe_relevant_baseline:current_price_only | LOSS | -4.69 | missing_candidate_action, missing_forecast_lookahead | constraint_aware emitted no relevant action type |
| latency_tail_kvcache_pressure | inference_critical | memory_pressure_relief | tokens | sla_aware | interactive_workload_prefers_sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| low_confidence_energy_telemetry | telemetry_fail_safe | safety_keep | telemetry_correct_keeps | fifo | telemetry_failsafe_correctness | KEEP_CORRECT | +0.00 | — | telemetry-failsafe scenario; KEEP matched FIFO and no SLA regression |
| migration_trap_erased_savings | batch_training | energy_arbitrage | token_equivalent | sla_aware | strongest_safe_relevant_baseline:sla_aware | TIE | +0.19 | — | within tie band, no material safety edge |
| moe_hotspot_nic_saturation | inference_standard | topology_fit | tokens | sla_aware | interactive_workload_prefers_sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| partial_utilization_telemetry | telemetry_fail_safe | safety_keep | telemetry_correct_keeps | fifo | telemetry_failsafe_correctness | KEEP_CORRECT | +0.00 | — | telemetry-failsafe scenario; KEEP matched FIFO and no SLA regression |
| prefix_affinity_energy_arbitrage | inference_standard | energy_arbitrage | tokens | sla_aware | strongest_safe_relevant_baseline:sla_aware | LOSS | -5.26 | missing_candidate_action | constraint_aware emitted no relevant action type |
| proxy_bottleneck_ingress | inference_standard | queue_relief | tokens | sla_aware | interactive_workload_prefers_sla_aware | LOSS | -8.72 | missing_candidate_action | constraint_aware emitted no relevant action type |
| queue_surge_latency_sensitive | inference_standard | queue_relief | tokens | sla_aware | interactive_workload_prefers_sla_aware | LOSS | -7.94 | missing_candidate_action | constraint_aware emitted no relevant action type |
| rack_density_liquid_cooled | inference_standard | thermal_spread | tokens | sla_aware | interactive_workload_prefers_sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| rack_density_overload_air | inference_standard | thermal_spread | tokens | sla_aware | interactive_workload_prefers_sla_aware | ALPHA_WIN | +1.89 | — | constraint_aware beat headline by +1.89% |
| scheduler_bound_inference | inference_standard | fragmentation_packing | tokens | fifo | no_packing_baseline_computed_for_this_run | TIE | +0.00 | — | within tie band, no material safety edge |
| startup_heavy_migration_trtllm | inference_standard | energy_arbitrage | tokens | sla_aware | strongest_safe_relevant_baseline:sla_aware | LOSS | -5.41 | missing_candidate_action | constraint_aware emitted no relevant action type |
| tensor_parallel_topology_collapse | inference_standard | topology_fit | tokens | sla_aware | interactive_workload_prefers_sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| thermal_hotspot_mixed_cluster | inference_standard | thermal_spread | tokens | sla_aware | interactive_workload_prefers_sla_aware | ALPHA_WIN | +46.83 | — | constraint_aware beat headline by +46.83% |
| topology_fragmentation_h100 | batch_training | topology_fit | token_equivalent | sla_aware | strongest_safe_relevant_baseline:sla_aware | TIE | +0.00 | — | within tie band, no material safety edge |
| underutilization_stranded_capacity | inference_standard | fragmentation_packing | tokens | fifo | no_packing_baseline_computed_for_this_run | ALPHA_WIN | +63.85 | — | constraint_aware beat headline by +63.85% |
| unsafe_aggressive_consolidation | inference_standard | fragmentation_packing | tokens | fifo | no_packing_baseline_computed_for_this_run | TIE | +0.00 | — | within tie band, no material safety edge |

## D. Baseline strength per scenario

Per-policy goodput/$ for every scenario, so reviewers can see which baseline was strongest and whether the headline selection was reasonable.

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

## E. Telemetry confidence (constraint_aware engine)

Telemetry truth signal from the engine assessments. Telemetry-failsafe scenarios are expected to show partial-confidence and force-KEEP behavior.

| scenario | mean confidence | partial |
|---|---|---|
| carbon_cheap_price_expensive | 0.85 | no |
| clean_batch_shift_arbitrage | 0.85 | no |
| da_rt_basis_blowout | 0.85 | no |
| degraded_topology_telemetry | 0.34 | yes |
| dram_bound_inference | 1.00 | no |
| energy_price_arbitrage_multiregion | 0.65 | no |
| fragmentation_stranded_capacity | 0.22 | no |
| kv_exhaustion_preemption_storm | 0.99 | no |
| latency_critical_no_energy_shift | 0.99 | no |
| latency_tail_kvcache_pressure | 0.62 | no |
| low_confidence_energy_telemetry | 0.29 | yes |
| migration_trap_erased_savings | 0.85 | no |
| moe_hotspot_nic_saturation | 1.00 | no |
| partial_utilization_telemetry | 0.29 | yes |
| prefix_affinity_energy_arbitrage | 0.99 | no |
| proxy_bottleneck_ingress | 1.00 | no |
| queue_surge_latency_sensitive | 0.63 | no |
| rack_density_liquid_cooled | 0.28 | no |
| rack_density_overload_air | 0.34 | no |
| scheduler_bound_inference | 1.00 | no |
| startup_heavy_migration_trtllm | 0.99 | no |
| tensor_parallel_topology_collapse | 1.00 | no |
| thermal_hotspot_mixed_cluster | 0.34 | no |
| topology_fragmentation_h100 | 1.00 | no |
| underutilization_stranded_capacity | 0.81 | no |
| unsafe_aggressive_consolidation | 1.00 | no |

## F. What remains simulator-only / needs real telemetry

- Every calibration parameter is an uncalibrated prior (none measured on real hardware). All KPI numbers are directional.
- Telemetry truth (Mission 1, FIXED): the canonical `ClusterState` derives provenance confidence + `is_partial` from the simulator's per-subsystem tiers, so degraded-telemetry scenarios report low/partial confidence and the engine force-KEEPs (telemetry subsystem verdict graduated to REALISTIC_ENOUGH_FOR_DEV). The tiers themselves remain uncalibrated heuristics.
- ML forecasting is a later phase. The current optimizer relies on the engine's workload-aware decision rules; once those land, a calibrated forecaster will get layered on top. Simulator results remain not production savings claims.
- Next calibration step: run a read-only shadow pilot against real Prometheus/DCGM/K8s telemetry to calibrate the priors and the confidence model (C = R·F·K·S·N) against measured staleness/coverage/noise.
