# Aurelius Results Standard — Canonical Benchmark / Reporting Reference

> **This document is the canonical results standard for Aurelius.**
>
> Any future Claude Code prompt (or human change) that touches optimizer,
> forecasting, simulator, benchmark, or report behavior **must read this file
> first** and preserve its reporting standard unless this document is itself
> updated through a deliberate revision. ML forecasting work — when it lands —
> must optimize and report against this standard, not against FIFO alone and
> not against raw energy cost alone.
>
> Simulator results are not production claims; live customer-telemetry
> calibration is required before any external savings number.

---

## 1. Primary KPI

**SLA-safe goodput per infrastructure dollar.**

```
sla_safe_goodput_per_infrastructure_dollar =
    sla_compliant_goodput
    / (gpu_infra_cost + energy_cost + network_cost)
```

- **Units:** `tokens-equivalent / $`. Higher is better.
- **Computed by:** `aurelius/benchmarks/economics.py::compute_sla_safe_goodput_per_infra_dollar`
  (and per-scenario in `compute_economic_kpi`). Cost basis is configurable via
  `InfrastructureCostConfig`.

### What counts as SLA-compliant goodput

For each tick, summed across all queues:

```
sla_compliant_tokens_for_queue =
    tokens_served_in_tick × max(0, 1 − timeout_rate_pct / 100)
```

- `timeout_rate_pct` is the simulator's per-queue measure of work whose p99
  exceeded the workload's configured `latency_sla_p99_ms` (or, for batch /
  offline / training workloads, work that missed its deadline / freshness
  target). At ≥ 50 % timeout the queue's contribution is **hard-excluded**:
  no partial credit.
- Telemetry-failsafe scenarios are scored on *correct KEEP* behavior, not
  goodput — see §5 and §6.

### What does NOT count

- Tokens served while the workload was violating its SLO (timed-out share
  excluded per the formula above).
- Tokens served by a workload that was scaled / migrated unsafely such that
  another workload's SLOs were broken — those losses appear in the receiving
  queue's `timeout_rate_pct`.
- Speculative throughput from over-provisioning that increased billable
  GPU-hours without commensurate SLA-compliant token gain (the denominator
  grows with active GPU-hours, so the ratio penalises this).
- Tokens served on TELEMETRY-failsafe scenarios where the engine's correct
  action is KEEP — these scenarios are scored on safety, not alpha.
- **No business-value weights, no revenue weights, no SLA-penalty dollars
  folded into the primary KPI.** SLA is a *filter on the numerator*, never a
  subtraction term in the denominator.

---

## 2. Secondary KPIs (diagnostics / vetoes — NEVER folded into the primary)

Secondary KPIs are constraints, vetoes, and diagnostics. They explain the
primary KPI and gate unsafe actions. They are **never** added as hidden
weighted terms to the headline metric.

| KPI | Source | Use |
|---|---|---|
| p95 / p99 latency | simulator + serving model | safety floor; latency-class headline filter |
| TTFT (p50/p95/p99) | simulator | interactive-SLO diagnostic |
| TPOT (p50/p95/p99) | simulator | interactive-SLO diagnostic |
| queue wait (p95) | simulator | queue-relief diagnostic; gate for SCALE_REPLICAS on interactive classes |
| migration count | runner / engine | churn diagnostic; cost denominator if `network_cost_per_migration` is configured |
| churn score | engine | safety diagnostic |
| cache hit rate (prefix / KV) | simulator | KV-pressure relief diagnostic |
| thermal throttling (ticks, events, hotspot severity) | simulator | thermal safety gate |
| topology quality (mean / min) | simulator | placement / fabric diagnostic |
| carbon (g CO₂ / kWh) | simulator | carbon-adjusted reporting (secondary only) |
| tokens / joule | simulator | efficiency diagnostic |
| utilization (effective / SM / DRAM-active) | simulator | underutilisation + fragmentation diagnostic |
| cost / token | `economics.py` | reciprocal of a goodput-per-dollar variant — diagnostic only |
| goodput / GPU-hour | `economics.py` | per-hour efficiency diagnostic |
| SLA violations (count) | simulator | hard SLA gate |

---

## 3. Baseline policy

**FIFO is the sanity baseline.** It catches "did anything change at all?" and
"did the engine make things worse than doing nothing?" It is **not** the
buyer-facing benchmark.

For each scenario, the report must compare `constraint_aware` against the
strongest *relevant* baseline for that workload class / intent:

| Scenario class | Required headline candidates |
|---|---|
| Energy / flexible batch / arbitrage | `current_price_only`, `greedy_energy`, `sla_aware` (energy-safe) |
| Online inference (latency-sensitive) | `sla_aware`, queue-aware autoscale / prewarm, no-unsafe-migration |
| Batch inference / embeddings | throughput-cost, bin-packing (`first_fit`, `best_fit`, `first_fit_decreasing`), `current_price_only` if region-shiftable |
| Training / fine-tuning | topology-aware placement, utilization / bin-packing, completion-time / deadline baseline |
| Thermal | thermal-aware spreading (if implemented), `sla_aware` no-hot-node |
| Topology / communication | topology-aware placement, same-node / NVSwitch-preferred |
| Telemetry-degraded | fail-safe KEEP / conservative no-op |

Headline selection rules (implemented in
`aurelius/benchmarks/per_workload.py::select_headline_baseline`):

1. Explicit `metadata.headline_baseline_override` is honored verbatim.
2. `is_telemetry_failsafe` → `fifo` (scored by KEEP-correctness, not alpha).
3. `optimization_intent == fragmentation_packing` → packing baseline; if none
   computed in this run, emit the honest disclaimer
   `"no_packing_baseline_computed_for_this_run"` rather than silently picking
   FIFO.
4. `optimization_intent == energy_arbitrage` → strongest safe of
   `current_price_only` / `greedy_energy` / `sla_aware`.
5. `inference_critical` / `inference_standard` → `sla_aware` if safe vs FIFO,
   else `fifo`.
6. `batch_inference` / `batch_training` / `embedding_offline` / `fine_tuning`
   → strongest safe of `metadata.relevant_baselines`.
7. Fallback: strongest safe non-oracle candidate else `fifo`.

A candidate is **safety-disqualified** if its `sla_violations > fifo` or its
`p99 > fifo × 1.5`. If all non-FIFO candidates fail safety, the headline is
`fifo` with rationale `"headline_baseline_disqualified_for_safety"`.

**Oracle / clairvoyant baselines** (e.g. `clairvoyant_lower_bound` in
`aurelius/benchmarks/packing.py`) are **analysis-only**. They quantify how
much room a heuristic has left, never serve as a deployable comparison, and
must not be used as the headline.

> Do not claim a win unless `constraint_aware` beats the strongest relevant
> baseline for the scenario / workload class. Matching FIFO is **not** a win.

---

## 4. Workload-type reporting

Every benchmark report must include the following tables, in this order:

| Table | Contents |
|---|---|
| **A. Overall policy** | per-policy mean + median goodput/$, win/loss counts, SLA regressions, alpha/safety counters |
| **B. Per workload type** | per-workload-type headline baseline, CA goodput/$, baseline goodput/$, margin %, cost/token delta, goodput delta, p99 delta, queue-p95 delta, SLA delta, migration/churn delta, outcome |
| **C. Per scenario** | scenario, workload type, optimization intent, headline baseline + rationale, CA goodput/$, baseline goodput/$, margin %, raw / GPU-infra / energy cost deltas, goodput delta, SLA delta, p99 delta, queue-p95 delta, win-or-loss reason, next-fix-if-loss |
| **D. Baseline strength** | per scenario, the full candidate set (FIFO, current_price_only, greedy_energy, SLA-aware, plus packing / topology / thermal candidates where relevant) with their goodput/$, plus the chosen headline and why |
| **E. Telemetry confidence** | per scenario, engine confidence + partial-flag, so KEEP-correct outcomes can be inspected |
| **F. Alpha / safety summary** | counts of ALPHA_WIN, SAFETY_WIN, KEEP_CORRECT, TIE, LOSS, SLA_regression, catastrophic_baseline_avoidance |

Every per-scenario row must surface:

- `scenario_name`
- `workload_type` (one of the names in §5)
- `optimization_intent` (see `OPTIMIZATION_INTENTS` in `per_workload.py`)
- `goodput_unit` (see §5)
- `headline_baseline_name` + `headline_baseline_rationale`
- `outcome` (one of: `ALPHA_WIN`, `SAFETY_WIN`, `TIE`, `LOSS`, `KEEP_CORRECT`)
- `reason_for_win_or_loss` (`loss_reasons` for LOSS rows; `safety_evidence`
  for SAFETY_WIN rows; empty for clean ALPHA_WIN / TIE)
- `next_fix_if_loss` (when outcome is LOSS)

Reports must **not** average telemetry-failsafe scenarios into the economic
mean / median — they live in a separate subsection so KEEP-correct behavior
does not distort the economic averages.

---

## 5. Goodput units

`goodput_unit` is a required field on every scenario row.

| Workload type | Goodput unit | Notes |
|---|---|---|
| `inference_critical` / `critical_interactive_inference` | `tokens` | SLA-compliant output tokens |
| `inference_standard` / `standard_interactive_inference` | `tokens` | SLA-compliant output tokens |
| `batch_inference` | `tokens` | output tokens completed before deadline / freshness target |
| `batch_training` / `training` | `token_equivalent` | tokens are a proxy for completed steps / samples / job-progress; **must be labelled** `token_equivalent` so readers do not conflate it with inference output tokens. When the simulator gains a step / sample / job-progress signal, switch to that and label accordingly. |
| `fine_tuning` | `token_equivalent` | same caveat as training. |
| `embedding_offline` / `embeddings_offline` | `tokens` | embeddings completed or tokens processed before deadline / freshness target. If the simulator gains an embedding-count signal, switch to `embeddings_completed` and label. |
| `telemetry_fail_safe` / `telemetry_degraded` | `telemetry_correct_keeps` | success metric is correct KEEP / no-unsafe-action, **not** higher goodput. Scored under §6 "correct KEEP", separately from alpha. |
| `communication_heavy` | `token_equivalent` | tokens are a proxy until a comm-volume / collective-step signal is available. |
| `mixed_cluster` / `mixed` | `tokens` | fall back to inference-token convention; secondary KPIs document the mix. |

Implementations must use the helper in
`aurelius/benchmarks/per_workload.py::_default_goodput_unit` (or an explicit
override via the scenario metadata) so the unit is always present and
unambiguous.

---

## 6. Alpha vs safety

`constraint_aware` has two distinct kinds of value, and the report must
separate them.

### Alpha wins — economic

- Higher SLA-safe goodput per infrastructure dollar than the headline baseline.
- Lower cost / SLA-compliant token.
- Better effective GPU utilization at the same SLA / safety floor.
- More SLA-safe serving capacity per dollar.

### Safety wins — risk reduction

- Prevents catastrophic p99 / TTFT / queue blow-up that a naive baseline
  would cause (e.g. `greedy_energy` migrating into a queue collapse).
- Avoids bad migrations: cold-cache thrash, destination instability,
  network-cost spikes.
- Fails safe under bad telemetry: forces KEEP when provenance is `low` and
  `is_partial`.
- Reduces thermal throttling without burning extra GPU-hours.
- Vetoes "fake savings" — actions where raw energy cost dropped but
  SLA-safe goodput collapsed harder.

### Correct KEEP — valuable safety, not alpha

In telemetry-failsafe scenarios the correct engine output is KEEP. Reports
**must** credit this as a `KEEP_CORRECT` outcome but **must not** count it
as economic alpha. KEEP_CORRECT lives in its own counter and its own report
subsection.

### Outcome classifier (binding)

Implemented in `aurelius/benchmarks/per_workload.py::analyze_outcome`. The
ordering of decision rules matters — do not change without revising this
document:

1. **KEEP_CORRECT** if `is_telemetry_failsafe AND ca.sla_violations ≤
   fifo.sla_violations AND |margin_pct| ≤ 1 %` (KEEP correctly preserved
   safety floor).
2. **SAFETY_WIN** if `|margin_pct| ≤ 1 %` (within the alpha tie band) AND a
   materially-better safety condition holds against the strongest
   non-headline baseline: `p99 ≤ 0.5 ×`, `sla_violations ≤ 0.5 ×`, OR
   `thermal_throttle_ticks ≤ 0.5 ×`.
3. **ALPHA_WIN** if `margin_pct > +1 %`.
4. **TIE** if `|margin_pct| ≤ 1 %` and no safety evidence.
5. **LOSS** if `margin_pct < −1 %`. Loss rows must populate `loss_reasons`
   (list — multi-cause allowed). Codes: `missing_candidate_action`,
   `wrong_workload_classification`, `over_conservative_gate`,
   `under_modeled_action_effect`, `missing_forecast_lookahead`,
   `telemetry_fail_safe`, `scenario_not_applicable`, `simulator_limitation`.

### Economic losses + SLA regressions

- `economic_loss_count` is the number of scenarios where `constraint_aware`
  lost to the headline by `> 1 %`. Each must carry a `loss_reasons` list and
  a `next_fix_if_loss` string.
- `sla_regression_count` is the number of scenarios where `constraint_aware`
  produced more SLA violations than FIFO. Any non-zero value blocks "win"
  claims for that scenario class.

---

## 7. Required report tables — exact schemas

The renderer in `aurelius/benchmarks/per_workload.py::CrossScenarioReport.to_markdown`
emits these schemas. Tables A, B, C, D, F are mandatory; E is required when
telemetry-failsafe scenarios are in the result set.

### A. Overall policy comparison

```
| policy | mean goodput/$ | median goodput/$ | wins vs headline | losses vs headline | SLA regressions | notes |
```

Plus an "Alpha/safety counters" line:

```
alpha_wins=N · safety_wins=N · correct_keeps=N · economic_losses=N
            · sla_regressions=N · catastrophic_baseline_avoidances=N
```

### B. Per workload type

```
| workload_type | headline_baseline | CA goodput/$ | baseline goodput/$ | margin % |
| cost/token Δ | goodput Δ | p99 Δ | queue p95 Δ | SLA Δ | migration/churn Δ | result |
```

`result ∈ { ALPHA_WIN, SAFETY_WIN, TIE, LOSS, KEEP_CORRECT }`.

### C. Per scenario

```
| scenario | workload_type | optimization_intent | goodput_unit |
| headline_baseline | headline_rationale | CA goodput/$ | baseline goodput/$ | margin % |
| raw cost Δ | GPU infra cost Δ | energy cost Δ |
| goodput Δ | SLA Δ | p99 Δ | queue p95 Δ |
| outcome | reason_for_win_or_loss | next_fix_if_loss |
```

### D. Baseline strength (per scenario)

```
| scenario |  fifo goodput/$ | current_price_only goodput/$ | greedy_energy goodput/$ |
| sla_aware goodput/$ | packing baselines goodput/$ (if applicable) |
| chosen headline | rationale |
```

Packing scenarios must list `first_fit`, `best_fit`,
`first_fit_decreasing` (and `clairvoyant_lower_bound` as analysis-only).
Topology scenarios must list any topology-aware baseline that ran.

### E. Telemetry confidence

```
| scenario | mean engine confidence | partial flag |
```

### F. Loss-reason table + alpha/safety summary

```
| scenario | outcome | loss_reasons | next_fix_if_loss |
```

Plus:

```
ALPHA_WIN: N    SAFETY_WIN: N    TIE: N    LOSS: N    KEEP_CORRECT: N
sla_regressions: N    catastrophic_baseline_avoidances: N
```

---

## 8. Claim rules

### Do NOT write

- "production savings"
- "guaranteed savings"
- "enterprise-ready autonomous optimization"
- "hyperscaler-validated economics"
- "production-proven"

Tests (`tests/test_per_workload_reporting.py`) scan generated markdown for
these substrings outside an explicit *negation* context. Any unhedged
occurrence is a CI failure.

### Allowed wording

- "Simulator benchmark result."
- "Requires live telemetry calibration."
- "Economic alpha shown only in simulator."
- "Safety win, not economic win."
- "Directional only — not production savings."
- "Not yet production-real savings."

### Production-claim gate

No production-savings number may be quoted until **all** of:

- the canonical KPI has been measured against a real customer telemetry feed,
- the per-class relief / safety priors in `_predict_scale_yield_ok` are
  calibrated against the customer's workload mix,
- the cost basis in `InfrastructureCostConfig` is the customer's actual
  procurement rate (not the public-list default),
- at least one shadow-pilot cycle has completed under
  `recommendation_only` with no SLA regression vs the customer's current
  scheduler.

---

## 9. Future ML / forecasting rule

When ML forecasting work lands (queue-surge forecasting, thermal forecasting,
DA/RT price forecasting, carbon forecasting, cache-hit forecasting,
topology / placement forecasting), it must:

1. **Primary objective:** maximise SLA-safe goodput per infrastructure
   dollar (the canonical KPI defined in §1). Not raw energy cost. Not FIFO
   delta. Not predicted token throughput.
2. **Secondary objectives:** the diagnostic KPIs in §2. These are constraints
   and safety floors, not weighted terms folded into the primary objective.
3. **Benchmark comparison:** the workload-relevant headline baseline per §3,
   not FIFO alone. Forecasters must beat the strongest baseline an operator
   would already use, on the canonical KPI.
4. **Reporting:** alpha wins and safety wins must remain separated per §6.
   Forecasting wins that come from speculative scaling must be recorded as
   alpha; forecasting wins that come from avoiding bad migrations under
   uncertain price / queue / thermal signals must be recorded as safety.
5. **Honest failures:** forecasters that improve forecast accuracy but do
   not improve the canonical KPI must be reported as **no economic alpha**
   even if their MAE / RMSE moves favourably. Forecast-quality metrics live
   under §2 (diagnostics), never under §1.
6. **Calibration before claims:** no forecaster's measured improvement may
   be quoted as a production saving until the §8 production-claim gate is
   satisfied for that forecaster's workload mix.

ML work is **not** in scope while the optimizer's workload-aware decision
rules are still being calibrated. ML forecasting is a later phase, after
the optimizer has the right objective and the right per-class action gates.

---

## 10. Reading rule

> Any future Claude Code prompt — or human change — that touches optimizer,
> forecasting, simulator, benchmark, or reporting behavior **must read
> `docs/RESULTS.md` first and preserve its reporting standard** unless this
> file itself is updated through a deliberate revision in the same change.
>
> Reviewers must reject changes that:
> - silently make FIFO the headline baseline for a workload class where a
>   relevant strong baseline exists,
> - fold a secondary KPI into the primary KPI as a weighted term,
> - introduce a workload-value or revenue weight,
> - drop the per-workload-type or per-scenario tables,
> - quote a production-savings number,
> - or weaken simulator realism penalties to make a benchmark win.
>
> If the standard itself needs to change (e.g., the canonical KPI is
> superseded, or a new workload class needs a new baseline), the change
> **must** update this file in the same PR and explain the reasoning in the
> PR body.
