# Migration / Rerouting / Drain / Cold-Start Realism Upgrade

Status: **simulator-only**. All outputs carry `is_sandbox=True` and are excluded
from economic claims. Real/customer environments remain `recommendation_only`.
This document is deliberately conservative — it does **not** claim production
accuracy; it claims the simulator's migration *dynamics* are now operationally
believable. Builds on the KV-cache realism layer (PR #77).

---

## 1. Architectural diff

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/migration.py` | Pure, deterministic (rng-seeded) functions: K8s drain, heavy-tailed engine-specific cold start, rerouting + proxy saturation, cache-loss penalty, batching-under-churn, tail uplift, composite `migration_cost`, governor veto, phased-rollout helpers. |
| `aurelius/simulation/cluster/migration_model.py` | 15 explicit mutable state models (MigrationState, DrainState, PodEvictionState, WarmPoolState, StartupState, ReplicaWarmupState, RouteShiftState, ColdStartState, ProxyQueueState, BatchCohortState, TrafficShiftState, RolloutState, PDBConstraintState, TailInstabilityState, …) aggregated into `WorkloadMigrationState`. |

Changed modules:

| File | Change |
|---|---|
| `calibration.py` | `MIGRATION_PARAMS` (21 params) + `ENGINE_STARTUP_PROFILES` (vLLM, TensorRT-LLM, SGLang, Triton, Ray Serve), each with `source/source_type/confidence/calibration_notes`; `migration_value()`, `resolve_engine_profile()`, `engine_profile_table()`; `calibration_table()` now spans serving + kv_cache + migration. |
| `model.py` | `SimWorkload` gains `engine_runtime`, `warm_pool_size`, `pdb_min_available`, `migration: WorkloadMigrationState`; `SimQueue` gains `proxy_saturation`, `batch_efficiency`. |
| `engine.py` | New `_update_migration()` (decay/bookkeeping, runs after `_update_kv_cache`); `_update_queues()` applies proxy saturation to queue wait, churn to batching, startup penalty + tail uplift to TTFT; `migrate_workload()` rebuilt around `_apply_migration_cost()` with PDB + optional governor veto; new public `safe_migrate_workload`, `migrate_workload_phased`, `can_migrate`, `set_warm_pool`, `set_pdb`; `add_replica()` applies scale-up cost + scale-from-zero amplification; `TickMetrics` gains 13 migration KPIs. |
| `scenarios.py` | New scenarios `startup_heavy_migration_trtllm`, `proxy_bottleneck_ingress` (plus existing KV scenarios). |
| `report.py`, `constraint_runner.py` | `TickKPI`/`AggregatedKPI` carry migration KPIs through the benchmark. |

Tick pipeline:

```
… → _update_thermal → _update_kv_cache → _update_migration → _update_queues → _update_cost_accounting
```

---

## 2. Migration-state model diagram

```
   migrate_workload(target)                       SimWorkload.migration
   │  PDB veto? (available==0) ──────► BLOCKED     : WorkloadMigrationState
   │  governor veto? (optional) ─────► BLOCKED      ┌──────────────────────────┐
   │                                                │ DrainState   T_evict+grace│
   ▼                                                │   +rebind (heavy-tail)    │
   _apply_migration_cost()                          │ StartupState  cold-start  │
   │  C_mig = T_transfer + T_warmup + T_requeue      │   decomposition (engine)  │
   │        + T_cacheloss + T_batchloss + T_tail     │ ColdStartState  freq /    │
   │                                                │   scale_from_zero         │
   ├─ startup_penalty_ms ──► ReplicaWarmupState ───►│ warmup ticks + penalty    │
   ├─ tail_mult ───────────► TailInstabilityState ─►│ p95/p99 uplift            │
   ├─ η_batch ─────────────► BatchCohortState ─────►│ batching efficiency       │
   ├─ churn++ ─────────────► RouteShiftState ──────►│ reroute_count, churn_rate │
   └─ drain_s ─────────────► DrainState/Eviction ──►│ PDBConstraintState        │
                                                    │ ProxyQueueState (offered) │
                                                    │ WarmPoolState / Rollout / │
                                                    │ TrafficShiftState         │
                                                    └──────────────────────────┘

   TTFT = queue_wait·proxy_saturation
        + (prefill·(1−prefix_savings) + contention + kv_stall)·kv_pressure_mult
        + cache_route_penalty + recompute_penalty + migration_startup_penalty
   p95/p99 ← ×(serving tail) ×(migration tail_uplift)
   throughput ← ×(cache-aware η_batch) ×(churn η_batch)
```

---

## 3. Cold-start calibration table

Engine cold-start decomposition (mean seconds/stage; `engine_profile_table()`):

| engine | node | pull | load | gpu_xfer | warmup | compile-heavy | total |
|---|---|---|---|---|---|---|---|
| vllm | 0 | 15 | 25 | 10 | 15 | no | 65 |
| sglang | 0 | 15 | 25 | 10 | 20 | no | 70 |
| triton | 0 | 18 | 28 | 12 | 25 | no | 83 |
| ray_serve | 0 | 15 | 25 | 10 | 30 | no | 80 |
| **tensorrt-llm** | 0 | 20 | 30 | 15 | **180** | **yes** | **245** |

Each stage is sampled as a right-skewed lognormal (median = anchor,
`coldstart_lognormal_sigma=0.6`) → heavy-tailed, not Gaussian. With probability
`coldstart_firstcompile_prob=0.15` the warmup hits the first-compile path
(×`coldstart_firstcompile_mult=4`, ×1.5 again for compile-heavy engines) →
bimodal startup. Sources are engine docs / public startup reports (`inferred`,
low confidence); these are operational anchors, **not** measured per-cluster.

---

## 4. Source-confidence table

21 migration params, all with provenance (`MIGRATION_PARAMS`):

| confidence | count | examples |
|---|---|---|
| medium | 4 | drain_grace_seconds (K8s 30s default), rollout_hold_ticks, warm_pool_idle_power_frac |
| low | 17 | proxy capacity/convexity, cold-start shape, tail uplift, churn sensitivity, governor thresholds |

Engine profiles: all `inferred`, low confidence (1 medium anchor: the K8s grace
default). **No value is MEASURED on a live cluster.** Everything is overridable
per run via `serving_config` (e.g. `{"drain_grace_seconds": 60,
"proxy_capacity_rps_per_replica": 20}`).

---

## 5. Realism-gap report

Now modelled (was missing/instant before):
- K8s drain (`T_evict+T_grace+T_rebind`), heavy-tailed grace, PDB blocking.
- Engine-specific, heavy-tailed, bimodal cold starts (TensorRT-LLM multi-minute).
- Rerouting = `max(proxy, rtt, accept)`; proxy/ingress saturation that dominates
  queue wait independent of replica count.
- Prompt-length-scaled cache-loss `ΔT_prefill`; batching degradation under churn.
- Migration **tail** uplift (p95/p99), not p50-only.
- Composite `C_mig`; scale-up / scale-from-zero amplification.
- Migration governor (veto) + phased canary rollout + rollback.
- Warm pools (trade idle energy for startup safety).

Still a proxy (the gap):
- Sub-tick costs at hourly granularity surface as ≥1 warmup tick + a one-shot
  TTFT penalty + tail uplift, **not** a within-tick timeline.
- Collapse-region magnitudes (multi-minute TTFT p99 under cold-start storms) are
  directionally right but **uncalibrated**.
- Phased rollout is exposed as engine methods/tests, not yet a benchmark policy.
- Orphaned-queue mechanic (workload migrated away from its queue's region) is a
  crude stand-in for replica relocation, inherited from the base sim.

---

## 6. Before / after KPI comparison

`startup_heavy_migration_trtllm` (compile-heavy, migratable), 24 ticks, seed 42:

| policy | reroutes | cold starts | startup s (max) | TTFT p99 (ms) | energy | SLA viol |
|---|---|---|---|---|---|---|
| fifo (stay put) | 0 | 0 | 0 | **94** | 2.58 | 300 |
| **greedy_energy** | 2 | 2 | 200 | **600,495** | 1.94 | 70 |
| constraint_aware | 0 | 2* | 1872 | 311,902 | 3.94 | 300 |

`proxy_bottleneck_ingress` (high RPS, small replica set):

| KPI | value |
|---|---|
| proxy saturation (max) | 100× (capped) |
| overload events | 21 |
| p99 latency | ~914,000 ms |

Naive energy-greedy rerouting saves ~25% energy but drowns TTFT p99 by
**>6000×** vs staying put. The proxy bottleneck pins latency at the overload cap
regardless of replica count. *(\*constraint_aware's cold starts here come from
scaling replicas — startup-heavy engines make even autoscaling expensive, a
realistic tradeoff the sim now exposes.)*

**Before this upgrade** migration was instantaneous and free; energy arbitrage
looked unconditionally profitable and replica count alone set throughput.

---

## 7. Newly failing unrealistic strategies

- **Abrupt energy-greedy rerouting** — pays drain + multi-minute cold start +
  tail uplift; loses badly on TTFT/p99 (demonstrated).
- **Migrate compile-heavy engines for cheap power** — TensorRT-LLM cold start
  (~245s mean, heavy-tailed) dwarfs the energy saving.
- **Scale-from-zero under load** — amplified TTFT + queue spike (no warm replica
  to absorb the queue).
- **"Add replicas to fix throughput"** when the proxy is the bottleneck —
  replica count past proxy capacity does nothing.
- **Aggressive churn** — fragments decode cohorts (η_batch floor) and worsens
  latency.

## 8. Migration-risk analysis

The composite `C_mig` makes migration risk legible: `T_transfer`/`T_warmup`
dominate for compile-heavy engines; `T_requeue` (drain + reroute) is irreducible;
`T_cacheloss` scales with prompt length and warm-cache value; `T_batchloss` and
`T_tail` capture instability. The governor blocks migration under queue pressure,
strong cache affinity, p95/rollout instability, PDB unavailability, incomplete
warmup, or a startup-heavy / scale-from-zero path — encoding "do-nothing is often
safest". Warm pools and phased rollouts are the mitigations the sim now rewards.

## 9. Remaining realism limitations

1. No value is MEASURED against a live cluster; all are priors.
2. Sub-tick timelines collapse to warmup-tick + one-shot penalty granularity.
3. Collapse-region magnitudes are uncalibrated (directional only).
4. Phased rollout is not yet wired as a first-class benchmark policy.
5. Orphaned-queue mechanic is a crude replica-relocation proxy.

## 10. Honest production-readiness assessment

**Believable, not validated.** Migration now carries realistic, legible cost and
risk: naive migration can lose badly, cold starts (especially compile-heavy
engines) materially hurt TTFT, proxy bottlenecks can dominate, and rollout
strategy / warm pools / migration restraint become valuable. Estimated savings
are **substantially more trustworthy** as directional signals because the
dominant downside risks are no longer invisible.

It is **not** a validated quantitative predictor. Before any economic claim, the
`low`-confidence parameters (drain/cold-start distributions, proxy capacity,
tail uplift, governor thresholds) must be calibrated against real telemetry:
kubectl-drain traces, per-engine startup histograms, ingress saturation curves,
and rollout p99 instability windows. Until then: treat outputs as believable
scenario dynamics, not production forecasts. Do **not** overclaim realism.
