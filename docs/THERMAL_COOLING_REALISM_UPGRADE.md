# Thermal / Cooling / Power Realism Upgrade

Status: **simulator-only**. All outputs carry `is_sandbox=True` and are excluded
from economic claims; real clusters remain `recommendation_only`. This document
is deliberately conservative — it does **not** claim production accuracy; it
claims the simulator's thermal *dynamics* are now operationally believable.
Builds on the KV-cache (#77) and migration (#78) realism layers.

---

## 1. Thermal architecture diff

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/thermal.py` | Pure, deterministic (rng-seeded) functions: saturating board power, lumped-capacitance temperature ODE, rack kW density regimes, hotspot formation/persistence, inlet temperature, continuous thermal + power slowdown, telemetry-confidence tiers, thermal migration veto. |
| `aurelius/simulation/cluster/thermal_model.py` | Explicit mutable state models (GPUThermalState, RackThermalState, AirflowState, HotspotState, RackDensityState, ThermalThrottleState, PowerThrottleState, CoolingZoneState, AmbientBoundaryState, ThermalTelemetryConfidence, ThermalViolationState, ThermalMigrationRiskState, ThermalInertiaState, CoolingRecoveryState). |

Changed modules:

| File | Change |
|---|---|
| `calibration.py` | `THERMAL_PARAMS` (18) + `GPU_POWER_CLASSES` (H100 SXM/PCIe, A100, L40S, L4) + `COOLING_REGIMES` (air/liquid/hybrid/hot-aisle/weak-airflow), each with provenance/confidence; `thermal_value()`, `resolve_cooling_regime()`, `power_class_for_model()`, `cooling_regime_table()`, `power_class_table()`; `calibration_table()` now spans 4 groups. |
| `model.py` | `SimGPU` gains continuous slowdown fields + `thermal: GPUThermalState`; `SimNode` gains `cooling_regime`, `event_heat_c`, `rack_thermal: RackThermalState`. |
| `engine.py` | `_update_gpu_state` uses the saturating board-power curve; `_update_thermal` rewritten as a lumped-capacitance ODE with rack-aggregated kW density, persistent hotspots, cooling regimes, inlet recirculation, and continuous slowdown; `_update_queues` applies `(1 − s_thermal − s_power)` to throughput and a continuous TPOT throttle; `migrate_workload` gains a thermal veto on hot destinations; 11 new thermal KPIs. |
| `scenarios.py` | New `rack_density_overload_air` + `rack_density_liquid_cooled`. |
| `report.py`, `constraint_runner.py` | `TickKPI`/`AggregatedKPI` carry thermal KPIs. |

---

## 2. Rack / cooling-zone state diagram

```
   region.ambient_temp_c (slow boundary) + node.event_heat_c (cooling fault)
        │
        ▼
   RackThermalState (per node, aggregated per rack_id)
   ┌──────────────────────────────────────────────────────────────┐
   │ RackDensityState  rack_kw = Σ board_power + airflow/recirc      │
   │   regime ∈ {normal <20kW | elevated | critical ≥30kW}·regime    │
   │      │  (thresholds × cooling_regime.density_mult)              │
   │      ▼                                                          │
   │ HotspotState  risk=f(density,airflow,sustained,regime)          │
   │   severity_{t+1} = max(persistence·severity_t, risk)  (lingers) │
   │      │                                                          │
   │      ▼  inlet = ambient + recirc·severity + N(0,var·regime)     │
   │ AirflowState  quality↓ under critical density, recovers slowly  │
   └──────────────────────────────────────────────────────────────┘
        │ inlet °C
        ▼   per GPU (by GPU_POWER_CLASS):
   GPUThermalState
     T_{t+1} = T_t + a·P − b·(T_t − inlet) + ε     (b = b_air·regime.beta_mult)
     s_thermal = ramp(T, onset→max)·thermal_slowdown_max     (continuous)
     s_power   = ramp(P, 0.9·cap→cap)·power_slowdown_max
        │
        ▼
   throughput ×= (1 − s_thermal − s_power);  TPOT ×= 1/(1 − s_thermal − s_power)
```

---

## 3. Cold... (thermal) calibration table

18 thermal params with full provenance (`THERMAL_PARAMS`). Key values:

| name | value | source_type | confidence |
|---|---|---|---|
| thermal_alpha | 0.039 °C/W/tick | inferred | low |
| thermal_beta_air | 0.30 /tick | inferred | low |
| thermal_noise_c | 0.4 °C | heuristic | low |
| power_curve_k | 4.0 | inferred | low |
| power_idle_frac | 0.30 | inferred | medium |
| rack_density_elevated_kw | 20 | inferred | medium |
| rack_density_critical_kw | 30 | inferred | medium |
| hotspot_persistence | 0.85 | inferred | low |
| thermal_slowdown_max | 0.30 | inferred | medium |
| power_slowdown_max | 0.30 | inferred | low |
| thermal_migration_hot_veto_c | 78 °C | inferred | medium |

GPU power classes (`power_class_table()`): TDP/throttle temps are datasheet
values (documented); per-class `alpha` is inferred and calibrated so each class
settles ~50°C above inlet at full power under air.

---

## 4. Cooling-regime comparison table

`cooling_regime_table()` — multipliers relative to air:

| regime | beta (recovery) | hotspot | density tolerance | inlet variance |
|---|---|---|---|---|
| air | 1.0 | 1.0 | 1.0 | 1.0 |
| hot_aisle_containment | 1.2 | 0.8 | 1.3 | 0.85 |
| hybrid | 1.5 | 0.6 | 1.6 | 0.7 |
| **liquid** | **2.2** | **0.35** | **2.5** | **0.4** |
| weak_airflow | 0.6 | 1.8 | 0.6 | 1.6 |

Liquid improves heat transfer, density tolerance, and hotspot variance — but
`hotspot_mult=0.35 > 0`, so it does **not** eliminate thermal risk.

## 5. Source-confidence table

| confidence | count | notes |
|---|---|---|
| medium | 5 | idle-power frac, density thresholds, slowdown max, hot-veto temp, GPU TDPs |
| low | 13 | thermal coefficients, curve shapes, hotspot/airflow dynamics, regime multipliers |

**No value is MEASURED on a live cluster.** Datasheet TDP/throttle temps are
documented; everything governing *dynamics* is an inferred/heuristic prior. The
density thresholds (~20/30 kW) and the cooling-regime multipliers are
operational heuristics, NOT universal constants.

## 6. Before / after hotspot behavior comparison

`rack_density_overload_air` vs `rack_density_liquid_cooled` (same dense 8-node
H100 rack, 32 GPUs @ 90% util, 24 ticks, seed 42):

| KPI | air | liquid |
|---|---|---|
| max GPU temp (°C) | **85.1** | 51.5 |
| rack density (kW) | 30.1 | 30.1 |
| hotspot severity (max) | **0.72** | 0.07 |
| thermal slowdown (mean %) | 0.18 | 0.0 |
| thermal throttle events | **142** | 0 |
| thermal excursions | **44** | 0 |
| dominant constraint | thermal | none |

Same power draw; the air rack forms persistent hotspots, overheats, and throttles
while the liquid rack stays cool. **Before this upgrade** temperature was an
instantaneous EMA with a binary throttle, density was per-node, hotspots did not
persist, and cooling regime was not modelled — dense placements looked stable.

## 7. Throttle-behavior comparison

`thermal_hotspot_mixed_cluster` (cooling-fault event, 24 ticks): the affected
rack rises to **95°C** with a **continuous** mean slowdown of ~5.8% and 66
throttle events / 18 excursions, recovering gradually after the event ends
(thermal inertia). Old model: a binary `thermal_throttle_active` flag flipping at
83°C with instant recovery. New model: continuous `s_thermal` ramped from the
per-class onset to the max, feeding both throughput loss and TPOT inflation.

## 8. Newly failing unsafe strategies

- **Dense air-cooled consolidation** — packing a rack past ~30 kW forms hotspots
  and throttles; the same layout is safe only under liquid cooling.
- **Migration into a hot zone** — the thermal governor vetoes it
  (`thermal_hot_destination`); migrating in would land on heat.
- **"Utilization is low, so it's cool"** — power saturates with utilization and
  heat has inertia; sustained load keeps racks hot after util drops.
- **Treating missing thermal telemetry as safe** — low telemetry confidence
  *lowers* the migration veto threshold (conservative).

## 9. Thermal realism gap report

Now modelled: saturating board power; per-class thermal response; thermal
inertia + gradual recovery; rack-aggregated kW density regimes; persistent
hotspots + recirculation; cooling regimes (air/liquid/hybrid/weak); continuous
thermal+power slowdown; inlet variance; thermal telemetry tiers; thermal
migration veto.

Still a proxy: 1st-order lumped model (no spatial CFD / per-row airflow field);
at hourly ticks GPUs are near thermal equilibrium each tick (inertia spans a few
ticks, not seconds); density thresholds and regime multipliers are priors;
hotspot risk is a scalar, not a spatial map; humidity / facility-water-temp /
chiller dynamics are not modelled.

## 10. Remaining limitations

1. No value MEASURED on a live cluster; datasheet TDPs aside, all dynamics are priors.
2. Lumped-capacitance, not CFD; no spatial thermal field within a rack.
3. Hourly tick granularity compresses sub-tick thermal transients.
4. Cooling-regime multipliers and density thresholds are operational heuristics.
5. Thermal-aware consolidation is enforced via the migration veto + density
   metrics, not a full bin-packing thermal optimizer.

## 11. Honest production-readiness assessment

**Believable, not validated.** Thermal behaviour now exhibits the right
qualitative couplings — power saturates with utilization, heat accumulates with
inertia and recovers gradually, dense racks form persistent hotspots and
throttle, cooling regimes materially change outcomes, and missing telemetry is
treated conservatively. Dense placements can fail, throttling materially cuts
throughput, and thermal-aware orchestration (spreading, hot-zone vetoes) becomes
operationally meaningful rather than cosmetic. Estimated savings are
**substantially more trustworthy** because thermal risk is no longer invisible.

It is **not** a validated quantitative predictor. Before any economic or
safety claim, the `low`-confidence parameters (thermal coefficients, density
thresholds, hotspot dynamics, regime multipliers) must be calibrated against
real DCGM thermal telemetry, rack power/inlet sensors, and throttle-vs-temp
curves. Until then: treat outputs as believable scenario dynamics, not
production forecasts. Do **not** overclaim realism.
