# Topology / Communication / Placement Realism Upgrade

Status: **simulator-only**. All outputs carry `is_sandbox=True` and are excluded
from economic claims; real clusters remain `recommendation_only`. This document
is deliberately conservative — it does **not** claim production accuracy. It
claims the simulator's topology and *communication dynamics* are now
operationally believable: topology-aware placement materially matters, bad
placement can collapse throughput, synchronization-heavy jobs become hard to
move, and communication penalties can outweigh energy savings. Builds on the
KV-cache (#77), migration (#78), and thermal (#79) realism layers.

Every uncertain value is a **tunable, source-tagged prior**, not a universal
constant. None are measured on a live cluster.

---

## 1. Topology architecture diff

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/topology.py` | Pure, deterministic (rng-seeded) functions: latency-bandwidth message cost `T = α + m/B_eff`, small-message amplification, congestion bandwidth degradation, ring/tree collective approximations + collective amplification, MoE all-to-all hotspot amplification, distance-ladder placement-quality score, communication penalty `P`, communication throughput factor, synchronization-stall penalty, communication-induced p95/p99 tail multipliers, NIC saturation, topology risk, telemetry-confidence tiers + score discount, topology-aware migration veto. |
| `aurelius/simulation/cluster/topology_model.py` | 20 explicit mutable state models: `GPUFabricState`, `NVLinkDomainState`, `NVSwitchState`, `PCIeFabricState`, `NUMAState`, `SocketLocalityState`, `RackLocalityState`, `NodeFabricState`, `InterconnectCongestionState`, `CommunicationPressureState`, `CollectiveLoadState`, `TopologyRiskState`, `PlacementAffinityState`, `FabricTelemetryConfidence`, `TopologyHealthState`, `CommunicationLatencyState`, `SynchronizationPenaltyState`, `CrossRegionFabricState`, `NICCongestionState`, `TopologyMigrationRiskState` (+ the `WorkloadTopologyState` composite). |

Changed modules:

| File | Change |
|---|---|
| `calibration.py` | `TOPOLOGY_PARAMS` (24) + `FABRIC_REGIMES` (9 regimes, NVSwitch→cross-region) + `TOPOLOGY_DISTANCE_LADDER` + `NVLINK_GENERATIONS` (A100/H100/future) + `WORKLOAD_COMM_PROFILES` (8 families), each with provenance/confidence; `topology_value()`, `resolve_fabric_regime()`, `resolve_nvlink_generation()`, `nvlink_generation_for_model()`, `resolve_comm_profile()`, `fabric_regime_table()`, `nvlink_generation_table()`, `workload_comm_profile_table()`; `calibration_table()` now spans 5 groups. |
| `model.py` | `SimGPU` gains `fabric: GPUFabricState`; `SimNode` gains `node_fabric: NodeFabricState`; `SimWorkload` gains `comm_profile`, `comm_message_bytes`, `topology: WorkloadTopologyState`. |
| `engine.py` | `_build_node` constructs per-GPU/per-node fabric state from the topology class; new `_update_topology` evolves per-node congestion/NIC/telemetry and per-workload collective load / pressure / sync penalty / risk / latency each tick (own RNG → preserves other layers' replay); `_compute_topology_score` rewritten on the distance ladder + telemetry discount; `_update_queues` applies the communication throughput factor + sync slowdown + communication-induced p95/p99 tail amplification; `_migration_veto` gains a topology-aware cross-domain veto; 14 new topology KPIs. |
| `scenarios.py` | New `tensor_parallel_topology_collapse`, `moe_hotspot_nic_saturation`, `degraded_topology_telemetry`. |
| `report.py`, `constraint_runner.py` | `TickKPI`/`AggregatedKPI` carry topology KPIs. |

---

## 2. Fabric regime diagram (distance ladder)

Each GPU pair in a workload maps to a regime; the **worst (bottleneck) hop**
paces the collective. Distances are a tunable ordinal ladder, NOT hop counts.

```
 d  regime         B_eff       α (lat)    example boundary crossed
 ─  ────────────   ─────────   ────────   ─────────────────────────────
 0  intra_gpu      2000 GB/s   0.1 µs     same GPU (on-device HBM)
 1  nvswitch        450 GB/s   0.8 µs     same NVSwitch domain (all-to-all)
 1  nvlink          300 GB/s   1.0 µs     same NVLink domain (partial mesh)
 2  pcie_root        50 GB/s   3.0 µs     same PCIe root complex (Gen5)
 3  socket           32 GB/s   5.0 µs     same NUMA socket (Gen4-class)
 3  node             20 GB/s   8.0 µs     cross-socket (UPI/QPI + PCIe staging)
 4  rack             25 GB/s   2.0 µs     same rack, different node (IB NDR/HDR)
 5  cross_rack      12.5 GB/s   5.0 µs     different rack (spine, oversubscribed)
 6  cross_region    1.25 GB/s 10000 µs    different region (WAN, ms-class)
```

`B_eff` degrades convexly under congestion toward a floor; `α` is amplified for
small messages and under congestion (queueing). Links are **not** interchangeable
pipes — they differ by orders of magnitude in both latency and bandwidth.

---

## 3. Communication calibration table (selected)

`calibration.TOPOLOGY_PARAMS` (24 entries, all with source/source_type/
confidence/calibration_notes). Highlights:

| param | value | source_type | meaning |
|---|---|---|---|
| `small_message_bytes` | 262144 | inferred | NCCL latency↔bandwidth crossover |
| `small_message_amp_max` | 3.0 | inferred | max small-message latency amplification |
| `congestion_onset` | 0.60 | inferred | link load where B_eff starts degrading |
| `congestion_convexity` | 2.0 | inferred | how fast B_eff collapses past onset |
| `congestion_bw_floor` | 0.20 | inferred | goodput floor under collapse |
| `nic_congestion_onset` | 0.55 | inferred | NIC incast/queueing onset |
| `collective_amp_max` | 6.0 | inferred | max collective amplification at large N |
| `sync_penalty_max` | 0.50 | inferred | max straggler-stall throughput slowdown |
| `comm_tail_max` | 10.0 | inferred | max comm-latency tail amplification |
| `topology_throughput_penalty_max` | 0.85 | inferred | max throughput collapse (bad TP placement) |
| `tp_instability_score` | 0.45 | inferred | quality below which TP/sync-heavy is unstable |
| `moe_hotspot_amp` | 2.5 | inferred | MoE all-to-all hotspot amplification |
| `topology_telemetry_missing_risk` | 0.5 | heuristic | risk inflation when topology telemetry missing |
| `migration_veto_distance` | 4.0 | inferred | distance rung that pins comm-heavy jobs |
| `collective_jitter_frac` | 0.08 | heuristic | collective-latency jitter (seeded) |

The great majority are HEURISTIC/INFERRED priors. NVLink/PCIe/IB bandwidth
regimes are DOCUMENTED order-of-magnitude anchors. Inspect everything via
`calibration_table()`, `fabric_regime_table()`, `nvlink_generation_table()`,
`workload_comm_profile_table()`.

---

## 4. Workload sensitivity table

`calibration.WORKLOAD_COMM_PROFILES`:

| profile | λ (comm weight) | collective | sync-heavy | NVLink affinity | tail sens. |
|---|---|---|---|---|---|
| tensor_parallel | 1.00 | all_reduce | yes | 1.0 | 0.9 |
| all_reduce_training | 0.85 | all_reduce | yes | 0.8 | 0.85 |
| moe_expert | 0.70 | all_to_all | yes | 0.7 | 0.8 |
| pipeline_parallel | 0.40 | p2p | yes | 0.6 | 0.6 |
| embedding | 0.30 | p2p | no | 0.4 | 0.4 |
| retrieval | 0.25 | p2p | no | 0.3 | 0.4 |
| batch_inference | 0.15 | none | no | 0.2 | 0.4 |
| comm_light_inference | 0.05 | none | no | 0.1 | 0.3 |

Tensor-parallel strongly prefers NVLink/NVSwitch and becomes unstable off it;
batch inference has low average sensitivity but still tails under poor topology.
Profiles are inferred from a scenario's `comm_profile`, else from
`workload_type` / `communication_intensity` (existing scenarios keep working).

---

## 5. Fabric penalty comparison (regime → quality → throughput)

Placement quality and the throughput factor by bottleneck regime, for a
tensor-parallel (λ=1.0) vs a batch-inference (λ=0.15) workload:

| regime | dist | quality | TP throughput× | batch throughput× |
|---|---|---|---|---|
| nvswitch | 1 | 1.000 | **1.000** | 1.000 |
| nvlink | 1 | 0.811 | 0.839 | 0.976 |
| pcie_root | 2 | 0.310 | 0.414 | 0.912 |
| socket | 3 | 0.229 | 0.345 | 0.902 |
| node (x-socket) | 3 | 0.172 | 0.296 | 0.894 |
| rack (IB) | 4 | 0.293 | 0.399 | 0.910 |
| cross_rack | 5 | 0.164 | **0.290** | 0.893 |
| cross_region | 6 | 0.034 | 0.179 | 0.877 |

A tensor-parallel job loses **~71%** of its throughput cross-rack; the same
placement costs batch inference only **~11%**. Workloads do **not** all benefit
equally from NVLink.

---

## 6. Collective amplification analysis

Ring all-reduce `T_ring ≈ 2(N−1)/N·(m/B_eff) + w·2(N−1)·α`; tree
`T_tree ≈ log2(N)·α + m/B_eff`. For an 8 MiB all-reduce across N=16:

| regime | all-reduce latency | amplification vs p2p |
|---|---|---|
| nvswitch | 0.06 ms | 3.03× |
| rack (IB) | 0.69 ms | 2.04× |
| cross_rack | 1.41 ms | 2.08× |

MoE all-to-all additionally multiplies by the congestion hotspot factor (up to
`moe_hotspot_amp` = 2.5×), so the MoE scenario reaches `collective_amplification_max`
≈ 6.0 under saturation. Collectives are **not** free, and latency-bound small-N
collectives are dominated by the α term.

---

## 7. Before / after topology behavior comparison

| behavior | before (old single link-type penalty) | after |
|---|---|---|
| placement model | one "best link type" → fixed 0.1/0.3/0.5 penalty by low/med/high | distance-ladder quality from 9 calibrated regimes, worst-hop paced |
| TP cross-rack | mild throughput nudge | throughput collapses (~71%), p99 runs away |
| NVSwitch co-location | indistinguishable from PCIe for "high" comm | quality 1.0, ~17% (sync only) vs ~83% cross-rack |
| collectives | not modeled | ring/tree cost + amplification (2–6×) |
| small messages | not modeled | latency amplification up to 3× |
| congestion | not modeled | convex B_eff degradation + NIC incast |
| sync-heavy jobs | not modeled | straggler stalls (up to 50% slowdown) |
| comm-induced tails | not modeled | p95/p99 amplified, p99 faster than p95 |
| telemetry gaps | assumed ideal | confidence tiers; missing → discounted score |
| migration | thermal/PDB/governor vetoes only | + topology cross-domain veto for comm-heavy jobs |

Scenario numbers (fixed seed): TP cross-rack → quality 0.248, comm penalty
83%, sync 35%, risk 0.55, 1 instability flag. Same layout co-located on one
NVSwitch node → quality 1.000, comm penalty 17%. MoE on PCIe under surge →
fabric congestion 1.00, NIC saturation 0.70, collective amplification 6.0,
comm penalty 82%.

---

## 8. Unsafe-placement / unsafe-migration examples now rejected

- **TP split across racks**: throughput collapses and p99 explodes; the
  topology score (0.25) flags it as unstable (`collective_instability`).
- **Cross-region migration of a comm-heavy training job**: vetoed
  (`topology_cross_domain`) even when the destination is cheaper energy — the
  move would break fabric locality. Batch-inference jobs migrate freely.
- **Migration under low topology telemetry**: the cross-domain veto distance
  drops (4→3), so an otherwise-allowed cross-socket move is blocked — missing
  topology is treated as *possibly bad*, never ideal.
- **Consolidating a sync-heavy job onto a congested PCIe node**: synchronization
  stalls + congestion erase the consolidation benefit.

---

## 9. Topology realism gap report

What is now realistic (qualitatively): distinct fabric regimes; latency +
bandwidth as separate dimensions; small-message amplification; ring/tree
collective scaling; topology-distance placement scoring; fabric congestion and
NIC incast; synchronization-stall penalties; communication-induced tail
blow-up; topology telemetry uncertainty; topology-aware migration vetoes;
workload-specific sensitivity.

What is still a **proxy** (NOT production-accurate):
- Regime bandwidths/latencies are vendor-doc order-of-magnitude anchors, not
  per-cluster measurements.
- The distance ladder and quality mapping are ordinal engineering heuristics.
- Collective costs are the standard ring/tree approximations, not a fitted NCCL
  model (no protocol selection, no chunking, no SHARP/in-network reduction).
- Congestion is a per-node scalar load, not a routed flow/contention simulation.
- The communication penalty `P` and throughput factor are believable summations,
  not regression-fitted to hardware.

---

## 10. Remaining limitations

- No explicit topology graph routing; cross-node distance uses node/rack/region
  membership, not measured paths.
- A fast same-rack InfiniBand regime can score *above* a slow cross-socket
  "node" regime (bandwidth-driven) — defensible but a ladder-vs-bandwidth
  tension worth calibrating.
- Per-node congestion is aggregated from resident workloads' comm weight ×
  utilization scaled by fabric headroom; it does not model per-flow fairness.
- Telemetry tiers are config-driven scenario inputs, not inferred from a real
  `nvidia-smi topo` / NCCL map parse.
- All magnitudes need calibration against real all-reduce/all-to-all sweeps,
  message-size sweeps, and congestion-collapse tests before any quantitative
  claim.

---

## 11. Honest production-readiness assessment

This layer makes topology **operationally meaningful** in simulation: bad
placement can collapse throughput, communication penalties can outweigh energy
savings, synchronization-heavy jobs become hard to migrate, and topology
degradation amplifies p95/p99 faster than the mean. That is sufficient to
exercise and stress topology-aware orchestration logic.

It is **not** a calibrated network model and must not be used for quantitative
production claims. Real clusters remain `recommendation_only`; simulator outputs
remain `is_sandbox=True`. Every constant is a tunable, source-tagged prior —
replace the HEURISTIC/INFERRED values with measured numbers before trusting any
absolute figure. Determinism is preserved under fixed seeds (the topology layer
uses a dedicated RNG so it does not perturb the thermal/serving/migration
streams).
