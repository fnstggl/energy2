# Aurelius Carbon Accounting — Audit & Implementation Note

Status: implemented in `aurelius/carbon/` (this PR). Read this before trusting any
carbon number Aurelius emits.

## Phase 1 — Pre-implementation audit (current state, traced through real code paths)

| Concern | Finding (before this PR) |
|---|---|
| Carbon ingest | `ingestion/grid_apis/watttime.py` fetches **MOER (marginal)**, converts `lbs/MWh → gCO2/kWh` via `×453.592/1000`, fails loudly on missing creds. Historical only — **no forecast MOER**. Carried its **own BA map** disagreeing with `region_registry` (us-east `PJM` vs `PJM_DOM`, us-south `ERCOT` vs `ERCOT_HOUSTON`, us-north `MISO` vs `MISO_INDIANAPOLIS`). |
| Carbon optimization | `optimization/objective.py` adds `beta*carbon_cost` as a **soft term only**. `OptimizationConfig.carbon_objective` / `carbon_threshold_gco2_per_kwh` were **dead config** — never read by any optimizer. **No hard carbon constraint existed.** |
| Silent 400 fallback | `objective.py:173`, `scheduler.py:565` (MILP), `backtesting/evaluator.py` (`carbon_fallback=400.0`). The realized evaluator **silently substituted 400 gCO2/kWh for missing hours and counted them as real**. |
| Carbon reporting | `reporting/savings_report.py` reported `carbon_reduction_*` but **did not gate on carbon coverage** and did not separate forecast vs realized. Savings were **incidental to cost optimization** (nothing optimized carbon). |
| Synthetic carbon | `cli simulate` → `simulation/replay.py` used `generate_carbon_scenario` (synthetic) and printed carbon savings **unlabeled**. |
| Baselines w/ carbon | **None** — all 7 baselines ignore `carbon_data`. |
| Migration | **Price-only** (`scheduler._segment_forecast_cost`, `_try_*_migration`). No migration carbon overhead/payback. |
| Region coverage | Only `us-west/east/south/north` have WattTime zones; others had no mapping and **no `carbon_unavailable` marker**. |

Pre-PR verdict: **carbon-awareness was mostly reporting/scaffolding** on top of a real WattTime
ingest client, and it could **silently fabricate savings** from the 400-fallback.

## What this PR adds (`aurelius/carbon/`)

- `accounting.py` — authoritative `emissions_kgco2()` formula + `WorkloadCarbonRecord` with every
  required provenance field + `CARBON_CALCULATION_VERSION`. Unit-tested kW/MW, h/5-min, g/kg.
- `regions.py` — ONE authoritative `region → WattTime BA` map, derived from `region_registry`, with
  `carbon_unavailable` status, validation, and an optimizer/evaluator consistency assertion.
- `constraints.py` — **hard** carbon constraints with explainable rejection (`rejected_by`,
  `rejection_reason`, `carbon_constraint_value`, `candidate_value`).
- `scoring.py` + `candidate.py` — carbon-aware candidate scoring. **Option A (normalize cost &
  carbon)** chosen by default; **Option B (explicit carbon price → USD)** also provided. Dollars and
  kg are never added directly without an explicit `carbon_price_usd_per_tonne`.
- `migration.py` — carbon-aware migration with `migration_carbon_overhead_mode ∈ {unknown,
  estimated, measured}`. A move is only "carbon-saving" if `emissions_saved_by_destination >
  emissions_added_by_migration`. Savings are **not reported as real when overhead mode is unknown**.
- `replay.py` — coverage-gated realized carbon replay (NO 400 fallback) + per-baseline savings
  (FIFO / price-only / greedy / random / no-migration / carbon-disabled / carbon-optimal-oracle).
- `forecast_realized.py` — stores forecast vs realized MOER/emissions/savings separately.

## Surgical honesty fixes to existing code

- `watttime.py` — now consumes the authoritative `regions.py` map (no divergent BA map) and gains
  `fetch_forecast_carbon()` (WattTime `/v3/forecast`).
- `backtesting/evaluator.py` — `carbon_fallback` defaults to `None`: missing MOER is **not**
  silently replaced. Carbon is summed only over covered hours; `carbon_data_coverage_pct` and
  `carbon_complete` are surfaced.
- `reporting/savings_report.py` — adds `carbon_claim_validity` + per-baseline carbon savings, and
  labels carbon `unverified` when coverage is incomplete.
- `optimization/objective.py` / `scheduler.py` — the decision-time fallback MOER is now a named,
  documented constant (`DECISION_FALLBACK_MOER_GCO2_PER_KWH`) and is **decision-time only**; realized
  reporting never uses it. Numeric value preserved so cost benchmarks are unaffected.
- `cli.py simulate` — prints a `SYNTHETIC CARBON` banner; synthetic carbon is labeled
  `carbon_data_source=synthetic`, `carbon_claim_validity=not_real_world_verified`.

## Carbon pipeline (final)

```
WattTime /v3/forecast  -> forecast MOER -> scheduler/candidate decision (carbon constraints + scoring)
WattTime /v3/historical-> historical MOER -> realized replay (coverage-gated, no 400 fallback)
baseline schedules     -> baseline emissions (authoritative formula)
optimized schedule     -> optimized emissions (authoritative formula)
comparison             -> per-baseline savings (realized) + forecast-vs-realized tracking
```
