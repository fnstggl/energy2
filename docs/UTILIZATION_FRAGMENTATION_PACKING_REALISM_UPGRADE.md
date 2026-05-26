# GPU Utilization / Fragmentation / Bin-Packing Realism Upgrade

Status: **simulator-only**. All outputs carry `is_sandbox=True` and are excluded
from economic claims; real clusters remain `recommendation_only`. This document
is deliberately conservative — it does **not** claim production accuracy. It
claims the simulator's utilization and packing *dynamics* are now operationally
believable: "free GPUs" are often unusable, consolidation has nonlinear risk,
aggressive packing can destabilize workloads, batching benefits flatten, and
utilization is multidimensional rather than a scalar occupancy metric. Builds on
the KV-cache (#77), migration (#78), thermal (#79), and topology (#80) layers.

Every uncertain value is a **tunable, source-tagged prior**, not a universal
target. None are measured on a live cluster.

---

## 1. Utilization architecture diff

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/utilization.py` | Pure, rng-seeded functions: multidimensional utilization `U_gpu = min(U_sm, U_mem, U_sched, U_pcie)`, roofline token ceiling, continuous-batching gain with diminishing returns, KV/VRAM headroom, multidimensional + topology-aware fragmentation, stranded capacity, saturating consolidation benefit + nonlinear risk, queue amplification, GPU-sharing interference, cross-node shard penalty, bin-packing risk, utilization telemetry confidence. |
| `aurelius/simulation/cluster/utilization_model.py` | 23 explicit mutable state models (GPUUtilizationState, SMUtilizationState, MemoryBandwidthState, DRAMPressureState, SchedulerPressureState, PCIePressureState, KVHeadroomState, MemoryHeadroomState, BatchingEfficiencyState, ContinuousBatchingState, FragmentationState, StrandedCapacityState, ConsolidationRiskState, WorkloadFlexibilityState, PackingDensityState, QueueAmplificationState, TopologyFeasibilityState, CrossNodeShardState, ResourceDomainState, SchedulabilityState, GPUSharingState, BinPackingRiskState, UtilizationTelemetryConfidence + composites). |

Changed modules:

| File | Change |
|---|---|
| `calibration.py` | `UTILIZATION_PARAMS` (31) + `WORKLOAD_CLASS_PROFILES` (8 classes) + `FLEXIBILITY_CLASSES` (3) + `RESOURCE_DOMAINS` (9), each source-tagged; `utilization_value()`, `resolve_workload_class()`, `flexibility_multiplier()`, `workload_class_table()`; `calibration_table()` now spans 6 groups. |
| `model.py` | `SimGPU` gains `utilization: GPUUtilizationState`; `SimWorkload` gains `workload_class`, `flexibility`, `sharing_policy`, `sharing_tenants`, `admissible_domains`, `output_len_cv`, `vram_requirement_bytes`, `util: WorkloadUtilizationState`. |
| `engine.py` | per-GPU/per-workload util state at build; new `_update_utilization` tick step (own RNG → preserves other layers' replay) computing per-GPU dimensions + per-workload packing/batching/consolidation + per-region fragmentation/stranded/density; `_update_queues` applies the roofline throughput cap + cross-node shard + sharing interference + packing queue amplification; `_migration_veto` gains packing-unsafe + fragmented-destination vetoes; 21 new utilization KPIs. |
| `scenarios.py` | New `dram_bound_inference`, `scheduler_bound_inference`, `fragmentation_stranded_capacity`, `unsafe_aggressive_consolidation`, `partial_utilization_telemetry`. |
| `report.py`, `constraint_runner.py` | `TickKPI`/`AggregatedKPI` carry utilization KPIs. |

---

## 2. Fragmentation model diagram

```
 region GPUs ──► per-rack free blocks ──► schedulable check
   │                                          │
   │  demand_d = largest multi-GPU job        │  free GPU is schedulable iff:
   ▼                                          ▼    • VRAM headroom OK (≤ 1-reserve)
 PackingDensity = allocated/total       • rack has a free block ≥ demand
   │                                          │    (topology-local placement)
   ▼                                          ▼
 F = 1 - schedulable_free / free        Stranded = topology_isolated
 F_topo = 1 - Σmin(free_d,dem_d)/Σfree_d           + vram_isolated + …
   │                                          │
   └──────────────► BinPackingRisk = 0.5·F + 0.3·density + 0.2·demand
```

Emergent: a region with single-GPU free fragments scattered across racks shows
**fragmentation 1.0 and 3 stranded GPUs** while a 4-GPU job cannot be placed.

---

## 3. Packing-risk model

```
 ConsolidationRisk R = r1·cross_domain + r2·queue + r3·inv_temp_margin
                      + r4·kv_pressure + r5·scheduler_pressure
   r1=0.30  r2=0.25  r3=0.15  r4=0.15  r5=0.15   (weights sum to 1.0)

 packing_unsafe  ⇔  R ≥ 0.55   → migration veto "packing_unsafe_consolidation"
 low-flexibility job → fragmented destination → veto "packing_fragmented_destination"
```

ConsolidationBenefit = `B_max·(1 - exp(-k·fraction))` (B_max=0.6, k=3) — saturating:
benefit at fraction 0.2 / 0.5 / 0.9 = **0.27 / 0.47 / 0.56** (diminishing returns).

---

## 4. Workload flexibility matrix

| class | util target | flexibility | topology sens. | batching sens. | sla class |
|---|---|---|---|---|---|
| training | 0.90 | low | 0.9 | 0.4 | batch |
| fine_tuning | 0.80 | medium | 0.7 | 0.5 | standard |
| comm_heavy | 0.70 | low | 0.95 | 0.4 | standard |
| batch_inference | 0.65 | high | 0.2 | 0.9 | batch |
| standard_inference | 0.55 | medium | 0.4 | 0.7 | standard |
| embeddings | 0.55 | medium | 0.3 | 0.6 | standard |
| latency_critical_inference | 0.50 | low | 0.6 | 0.5 | latency_critical |
| memory_heavy | 0.45 | medium | 0.4 | 0.5 | standard |

Flexibility multipliers: low 0.2 · medium 0.6 · high 1.0. Low-flexibility jobs
(training / comm-heavy / latency-critical) are pinned; high-flexibility (batch /
async) move freely.

---

## 5. Consolidation safety table

| signal | safe-ish regime | unsafe regime |
|---|---|---|
| compute util | moderate, memory headroom remains | DRAM_ACTIVE high while SM low (paradox) |
| topology | preserved (local) | cross-node sharding |
| queue | p95 stable | queue growth accelerating (amp → 8×) |
| thermal | no throttling | throttling emerging (weak airflow / dense) |
| KV | headroom remains | near the safe-occupancy ceiling |
| scheduler | below capacity | active-seq past capacity (sched-bound) |
| risk R | < 0.55 | ≥ 0.55 → packing veto |

---

## 6. Utilization bottleneck comparison (caps)

| dimension | cap at light load | cap at heavy load |
|---|---|---|
| memory bandwidth | dram 0.4 → **1.00** | dram 0.9 → 0.83, 1.4 → **0.54** |
| scheduler | 128 seqs → **1.00** | 512 → 0.50, 1024 → **0.25** |

`U_gpu = min(U_sm, U_sm·mem_cap, U_sm·sched_cap, U_sm·pcie_cap)`. A compute-bound
workload (all caps 1.0) is unchanged; a memory- or scheduler-bound workload is
pinned below its compute rate.

---

## 7. Before / after realism comparison

| behavior | before | after |
|---|---|---|
| GPU utilization | one scalar (`utilization_pct`) | multidimensional `min(SM, DRAM, sched, PCIe, KV)` |
| free GPUs | universally schedulable | fragmentation + VRAM + topology gating; stranded islands |
| batching gain | implicit / linear-ish | `1 + a·CV` with diminishing returns (cv0.1→1.85, cv1.0→6.07, favorable→15+; capped) |
| consolidation | linear, always safe | saturating benefit + nonlinear risk + veto |
| memory pressure | conflated with compute | separate DRAM-bandwidth bottleneck → utilization paradox |
| scheduler | not modeled | scheduler capacity bottleneck (95% throughput penalty at surge) |
| VRAM | 100% assumed safe | ~5% reserve; >95% occupancy suppresses admission |
| packing queues | not modeled | superlinear queue amplification (bounded 8×) |
| GPU sharing | not modeled | MIG/time-slice/fractional interference |
| telemetry gaps | assumed clean | confidence tiers; missing ≠ schedulable |

---

## 8. Unsafe-packing examples now rejected

- **DRAM-bound "idle" GPU** (`dram_bound_inference`): SM 0.46 but DRAM_ACTIVE high
  → 4 GPUs flagged utilization-paradox; effective util 0.33, ~25% throughput
  penalty. Looks idle, is NOT a safe packing target.
- **Scheduler saturation** (`scheduler_bound_inference`): concurrency past
  scheduler capacity → scheduler-bound, ~95% throughput penalty, queue
  amplification 8×.
- **Stranded islands** (`fragmentation_stranded_capacity`): fragmentation 1.0,
  3 stranded GPUs; a 4-GPU job is unschedulable despite free capacity.
- **Unsafe consolidation** (`unsafe_aggressive_consolidation`): risk reaches
  0.56 → migration vetoed (`packing_unsafe_consolidation`).

---

## 9. Stranded-capacity examples

A region with one 4-GPU node (fully used by a 4-GPU job) plus three 2-GPU nodes
each half-used by a 1-GPU job: **3 free GPUs, all stranded** (no rack has a free
block ≥ 4). The cluster reports free capacity yet rejects the large job. Free
GPUs are not universally schedulable.

---

## 10. Queue amplification analysis

Queue amplification is driven by **per-replica oversubscription** (batch
occupancy), NOT raw GPU allocation — a fully-allocated-but-not-oversubscribed
cluster is healthy. Below the onset (0.70) it is neutral (1.0); past it it rises
convexly, **bounded to 8×** so it compounds with — rather than swamps — the
serving layer's own saturation amplifier. The scheduler-bound scenario reaches
the 8× cap under surge.

---

## 11. Remaining realism gaps

- Fragmentation uses rack-block heuristics, not a real bin-packing/scheduler
  simulation; `topology_fragmentation_score` can read 0 when each domain's free
  count ≤ demand even though co-location is impossible (the scalar
  `fragmentation_score` captures that case).
- DRAM-bandwidth / scheduler / PCIe caps are dimensionless operational
  heuristics, not measured FLOP/byte rooflines or engine-specific scheduler
  limits.
- Consolidation risk weights and the unsafe threshold are policy heuristics.
- GPU-sharing interference is a scalar penalty, not a co-tenancy contention
  model.
- Utilization telemetry tiers are config-driven scenario inputs, not parsed from
  real DCGM staleness.
- All magnitudes need calibration against real DCGM `GPU_UTIL`/`DRAM_ACTIVE`
  distributions, throughput-vs-batch curves, and scheduler saturation tests
  before any quantitative claim.

---

## 12. Honest production-readiness assessment

This layer makes utilization a **multidimensional systems problem** in
simulation: fragmentation materially affects scheduling, aggressive consolidation
can destabilize workloads, topology-aware packing matters, the utilization
paradox (high resource use, low throughput) emerges, and free GPUs are often
unusable. That is sufficient to exercise and stress packing / consolidation
orchestration logic.

It is **not** a calibrated scheduler/allocator model and must not be used for
quantitative production claims. Real clusters remain `recommendation_only`;
simulator outputs remain `is_sandbox=True`. Every constant is a tunable,
source-tagged prior — replace the HEURISTIC/INFERRED values with measured numbers
before trusting any absolute figure. Determinism is preserved under fixed seeds
(the utilization layer uses a dedicated RNG; the default well-provisioned,
compute-bound case is neutral — throughput factor 1.0, queue amplification 1.0).
