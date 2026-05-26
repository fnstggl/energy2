# Energy / Carbon / Arbitrage Realism Upgrade

Status: **simulator-only**. All outputs carry `is_sandbox=True` and are excluded
from economic claims; real clusters remain `recommendation_only`. This document
is deliberately conservative — it does **not** claim production energy or carbon
savings. It claims energy-aware scheduling is now modeled as *constrained
optimization* (DA/RT basis, LMP congestion/loss, carbon-forecast uncertainty,
workload-flexibility limits, net-vs-gross accounting, forecast buffers,
diminishing returns) rather than clean price minimization. Builds on the
KV-cache (#77), migration (#78), thermal (#79), topology (#80), and utilization
(#81) layers.

Every uncertain value is a **tunable, source-tagged prior**. Basis + spikes are
OFF by default (RT == DA), preserving deterministic pricing for non-energy
scenarios.

---

## 1. Price / carbon model architecture

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/energy.py` | Pure, rng-seeded functions: two-settlement bill, mean-reverting + heteroskedastic DA/RT basis with congestion jumps, LMP (energy+congestion+loss), heavy-tailed RT spike, carbon intensity + forecast uncertainty, weighted objective, risk-adjusted savings, net-vs-gross savings, super-linear churn penalty, required-margin (telemetry-inflated), shift windows, telemetry confidence. |
| `aurelius/simulation/cluster/energy_model.py` | Mutable state models: DayAheadState, RealTimeState, BasisState, LMPComponentState, CongestionState, CarbonForecastState, ForecastUncertaintyState, SpareCapacityState, EnergyTelemetryConfidence, NetSavingsState, ChurnState, ShiftWindowState + RegionEnergyState / WorkloadEnergyState composites. |

Changed modules: `calibration.py` (23 `ENERGY_PARAMS` + 3 `ENERGY_FLEX_PROFILES`,
all source-tagged; `energy_value()`, `resolve_energy_flex()`, `energy_flex_table()`;
`calibration_table()` now spans 7 groups); `model.py` (`SimRegion` gains
day_ahead/realtime prices + `energy_state`; `SimWorkload` gains `workload_class`-
adjacent energy fields + `energy` state + `alpha_cost`/`beta_carbon`);
`engine.py` (`_update_energy` tick step with a dedicated RNG; DA/RT settlement in
cost accounting; EnergyState exposes distinct DA/RT; energy net-savings veto +
churn; continuous per-tick net-savings evaluation; energy congestion / grid-stress
events; 13 new KPIs); `scenarios.py` (6 new scenarios); benchmark report/runner KPIs.

---

## 2. DA/RT settlement implementation

```
bill = q_DA · p_DA + (q_RT − q_DA) · p_RT
```

- `current_energy_price` / `day_ahead_price` = the **day-ahead planning signal**.
- `realtime_price` = the **realized settlement price** = DA + basis + congestion +
  loss + spike (== DA when basis/spikes/congestion are all off).
- Cost accounting settles realized GPU energy at the **real-time** price, so a
  day-ahead planner (e.g. `current_price_only`) can be wrong under RT.
- `EnergyState.day_ahead_price_per_mwh` and `real_time_price_per_mwh` are now
  distinct; `price_per_mwh` exposes the realized RT price.

Emergent (`da_rt_basis_blowout`): DA mean **78**, RT mean **137** ($/MWh) after a
destination congestion event → the energy governor rejects the DA-cheap move.

---

## 3. LMP component model

```
p(t, node) = energy_component + congestion_component + loss_component
```

- energy_component = the system-wide day-ahead price;
- congestion_component = 0 in the base case, + `lmp_congestion_event_adder`
  (~80 $/MWh) during a constrained-interface event;
- loss_component = `lmp_loss_frac` (~3%) of the energy component.

Regional/nodal arbitrage relies on the congestion + loss components, which are
NOT clean: a destination congestion event can erase the spread.

---

## 4. Carbon forecast uncertainty model

```
CI_forecast(t,r) = CI_actual(t,r) · (1 + N(0, error_std))
```

- regional, diurnal (solar/wind cycle), with `carbon_forecast_error_std` (~15%)
  and `carbon_provider_disagreement` (~12%);
- forecast carbon is NOT ground truth and providers disagree;
- objective = `α·cost + β·carbon`; `carbon_price_correlation` is low — carbon-cheap
  windows are NOT price-cheap windows.

Emergent (`carbon_cheap_price_expensive`): moving to the clean (price-expensive)
region has **negative price gross** but **positive carbon value**, so with β>0 the
net is positive — carbon optimization ≠ price optimization.

---

## 5. Workload flexibility matrix

| flexibility | max shift window | spatial shift | requires locality | examples |
|---|---|---|---|---|
| high | 24 h | yes | no | batch/offline inference, embeddings, non-urgent fine-tuning |
| medium | 2 h | yes | yes | standard inference, retryable/async jobs, lower-priority fine-tuning |
| low | 0 h | no | yes | latency-critical inference, cache/topology-sensitive, training, TP jobs |

Nothing is infinitely deferrable; deadline pressure rises as deferred work
approaches its window.

---

## 6. Energy action veto table

An energy-motivated cross-region move is allowed only if it clears the required
margin on BOTH net and risk-adjusted savings:

```
act_only_if  risk_adjusted_savings > required_margin  AND  net_savings > required_margin
net_savings = gross_energy + gross_carbon − migration − cache − cold_start
              − queue − sla − topology − thermal − forecast_error − churn
```

| veto | trigger |
|---|---|
| `energy_not_worth_it` | net or risk-adjusted savings below the required margin (tiny spread / migration trap / forecast buffer / churn) |
| `thermal_hot_destination` | destination thermally unsafe (thermal layer) |
| `topology_cross_domain` | comm-/sync-heavy job across fabric domains (topology layer) |
| `packing_unsafe_consolidation` | consolidation risk unsafe (utilization layer) |
| `cache_affinity_strong` / queue / PDB | cache/serving/migration governors |

Low energy-telemetry confidence inflates the required margin (×2), biasing toward
no-op (missing forecast ≠ safe opportunity).

---

## 7. Net-vs-gross savings report

Reported per tick and aggregated per policy: `gross_savings_sum` vs
`net_savings_sum`, plus `energy_actions_rejected`, `energy_migration_vetoes`,
`churn_penalty_max`. Net is ALWAYS reported — energy savings are never shown
without penalties. If net ≤ 0 the recommendation is KEEP / no-op.

| scenario | gross | net | outcome |
|---|---|---|---|
| clean_batch_shift_arbitrage | 0.705 | **0.611** | shift allowed (safe net savings) |
| da_rt_basis_blowout | 0.437 | 0.111 (but move rejected) | DA planner wrong under RT |
| migration_trap_erased_savings | 0.013 | **−0.028** | KEEP / no-op |
| low_confidence_energy_telemetry | 0.054 | ~0 | no-op (inflated margin) |

---

## 8. Benchmark comparison vs current_price_only and greedy_energy

`current_price_only` and `greedy_energy` now **plan on the day-ahead price** but
**settle at the real-time price**. Under an RT basis blowout they migrate to a
DA-cheap region and pay the realized RT premium — gross looks good, the settled
`total_energy_cost` does not. The constraint-aware path can decline the move via
the energy net-savings veto (available through `safe_migrate_workload`). The
benchmark must NOT be tuned so constraint-aware always wins — it wins only when it
preserves SLA/queue/cache/topology/energy safety while capturing real net savings.

---

## 9. Scenarios where energy optimization works

- `clean_batch_shift_arbitrage`: a flexible batch job + a large, stable DA spread,
  no basis → shifting captures positive net savings (gross ≫ migration cost).

## 10. Scenarios where energy optimization is blocked

- `da_rt_basis_blowout`: DA-cheap destination, RT blows out under congestion → move
  rejected.
- `migration_trap_erased_savings`: small spread, migration cost makes net ≤ 0 →
  KEEP.
- `low_confidence_energy_telemetry`: missing telemetry inflates the margin → no-op.
- `latency_critical_no_energy_shift`: low-flexibility latency-critical inference is
  not shifted.

## 11. Scenarios where current_price_only wins gross but loses operationally

`da_rt_basis_blowout`: a DA planner sees a cheap destination (gross positive) but
settles at the blown-out RT price; the realized cost rises. The net-savings model
shows the loss the gross number hides.

## 12. Scenarios where constraint-aware captures safe savings

`clean_batch_shift_arbitrage`: the energy gate ALLOWS the move (net > margin) when
flexibility and headroom exist — the one place a safe net saving is real.

---

## 13. Remaining realism gaps

- The basis is a tunable OU + jump prior, NOT calibrated to a specific ISO; LMP
  component magnitudes are heuristic.
- The continuous net-savings KPI evaluates against the cheapest **day-ahead**
  region, so carbon-weighted moves are visible via the API but not in the
  price-driven continuous KPI.
- Migration cost is monetized as a fraction of one tick's source energy cost (a
  believable proxy, not a fitted cost); cache/cold-start/topology/thermal
  penalties are represented through the existing layers' governors rather than
  re-monetized here.
- The benchmark policy wiring still uses `migrate_workload` (the energy veto is
  enforced via `safe_migrate_workload`); policies are not re-plumbed.
- Carbon-price correlation, spike/basis parameters, and forecast errors all need
  calibration against real market data before any quantitative claim.

---

## 14. What claims are now supported

- "In simulator scenarios with explicit assumptions, constraint-aware Aurelius
  captures safe net savings when workload flexibility and operational headroom
  exist." (e.g. `clean_batch_shift_arbitrage`.)
- The simulator models DA/RT basis risk, LMP congestion/loss, carbon-forecast
  uncertainty, flexibility limits, forecast buffers, churn, and net-vs-gross
  accounting — so energy arbitrage is a constrained problem.

## 15. What claims remain simulator-only

- NO universal energy or carbon savings; NO production savings from this evidence.
- DA/RT basis is NOT negligible; carbon optimization does NOT always reduce cost;
  not all workloads are shiftable; migration is NOT free; gross ≠ net.
- All magnitudes are tunable, source-tagged priors. Real clusters remain
  `recommendation_only`; outputs remain `is_sandbox=True`. Determinism is
  preserved under fixed seeds (dedicated energy RNG; default DA == RT).
