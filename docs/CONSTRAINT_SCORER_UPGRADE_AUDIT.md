# Constraint-Aware Scorer Upgrade Audit + Shadow Scorer v1

> **Audit + shadow PR.** No production scheduler / residency / routing /
> robust-energy behaviour is changed. No real execution is enabled. No
> production savings are claimed. HF / public-trace data is not pilot
> telemetry. Oracle is never headline. No invented economic constants.
>
> **Read first:**
> - `docs/RESULTS.md`
> - `docs/BENCHMARK_BASELINE_AUDIT.md`
> - `docs/PLACEMENT_PRIOR_AUDIT.md` (precedent for the audit shape +
>   adapter contract)
> - `docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md` (the cache forecaster
>   wired in variant D / E)
> - `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md` (the TTFT p50
>   shadow prior wired in variant C / E)
> - `docs/CARA_QUEUE_FORECASTER_RESULTS.md`
> - `docs/HF_DATASET_REGISTRY.md` (Tier 2-5 trust hierarchy of HF priors)

## 1. Problem statement

The audit `docs/PLACEMENT_PRIOR_AUDIT.md` traced the existing
`score_residency_candidate` path and surfaced the *headline gap*:

> GPU type is not used as a latency prior anywhere in the goodput/$
> scorer. The 9× p99 TTFT spread observed across GPU types in CARA is
> invisible to the current scorer.

The follow-on cache / prefix-reuse forecaster
(`docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`) landed
`final_status = diagnostic_only` partly because:

> The residency scorer cannot today consume cache value
> (`docs/PLACEMENT_PRIOR_AUDIT.md::scoring_inputs`), so even at ≥5%
> shadow alpha the forecaster would map to
> `blocked_by_scorer_limitations` pending a future scorer-side PR.

**This PR is that scorer-side PR.** It builds a *shadow-only* upgraded
scorer that can express the missing terms (per-(model, GPU) latency,
prefill cost avoided, migration cache-loss penalty, per-GPU $/hr,
energy per request) and runs an honest A/B/C/D/E evaluation to decide
whether those terms move SLA-safe goodput/$ decisions.

## 2. Binding rules

- **Signal hierarchy.** Every economic input is classified as one of:
  - **Level 1 — measured**: pilot telemetry / hardware measurement /
    per-request observation / explicit operator-configurable policy.
    Source + units recorded; **no fitted coefficient allowed**.
  - **Level 2 — derived**: a transparent formula on Level-1 and
    Level-3 inputs. Formula must be shown; **no learned utility
    coefficient**.
  - **Level 3 — prior**: bounded benchmark / public-trace / forecasted
    values. Tagged `value_quality = "prior"`; never silently treated
    as production truth.
  - **Level 4 — prohibited or uncalibrated**: invented utility
    coefficient (e.g. `CACHE_WEIGHT = 0.15`, `MIGRATION_PENALTY = 3`),
    or a $-denominated term whose operator coefficient is missing.
    Excluded from headline.
- **No invented economic constants.** Every $-denominated coefficient
  traces to (a) measured telemetry, (b) measured benchmark data, (c)
  hardware measurement, or (d) an explicit
  `OperatorPricingPolicy` field. Terms requiring an operator
  coefficient the policy does not supply are reported
  `not_computable_without_operator_policy` and excluded from headline.
- **No utility-weighted composite.** The primary KPI is the single
  quotient `SLA-safe goodput / $`. No `0.4×latency + 0.3×cache + …`
  form exists anywhere.
- **Shadow-only.** Every record carries `shadow_only = True`,
  `executable_in_real_cluster = False`, `no_control_action_taken = True`.
- **Production safety floor.** The production scorer is the safety
  floor for hard vetoes. Hard vetoes (region/thermal/topology/memory/
  queue-max) make the candidate infeasible even if the shadow scorer
  prefers it.

## 3. Audit (PHASE 1) — every scorer input × variant

Full machine-readable trace + per-input classification + signal level
lives in
`data/external/forecasting/constraint_scorer_upgrade/scorer_path_audit.json`.

22 economic / latency / SLA inputs were inventoried. Before/after
counts by signal level (`level_counts_existing` /
`level_counts_upgraded`):

| level | existing scorer | upgraded shadow scorer |
|---|---:|---:|
| level_1_measured | 3 | 3 |
| level_1_operator_policy | 3 | 3 |
| level_2_derived | 3 | 10 |
| level_3_prior | 0 | 5 |
| level_4_prohibited_or_uncalibrated | 2 | 0 |
| missing | 12 | 2 |

Headline reclassifications (full list in the JSON):

- **TTFT / TPOT** (was `heuristic` — the 2.0 s
  `service_time_proxy_s` heuristic): now `derived_from_optimum_benchmark`
  (Level 2 — `service_time_s = ttft_prior + output_tokens /
  decode_throughput`).
- **prefill cost / decode cost / cache-hit value / prefix reuse**: were
  `missing`. Now `derived_from_per_gpu_prior` /
  `forecasted_cache_prefix_reuse_v1_proxy` (Level 2 derived from a
  Level-3 cache forecaster prior).
- **migration cache-loss penalty**: was `missing`. Now
  `derived_from_per_gpu_prior` (Level 2 —
  `prompt_tokens / prefill_throughput_tok_s`).
- **cloud cost**: was `static_global_default` (Level 1 operator policy
  but only a single global value). Now
  `operator_policy_or_operator_global_default` (Level 1 operator
  policy per GPU type, with safe fallback to the existing global
  default).
- **energy per request / energy cost**: were `missing`. Now
  `derived_from_optimum_or_acmetrace` (kWh from Level-3 prior) +
  `derived_iff_operator_energy_price_supplied` (USD requires Level-1
  operator coefficient or term reports uncalibrated).
- **SLA risk**: was `derived_binary`. Now
  `derived_latency_margin_indicator` — a *quantitative latency margin*
  in seconds (no fitted sigmoid coefficient).

## 4. Coefficient calibration audit (PHASE 1)

Every $-denominated coefficient in the shadow scorer (see
`scorer_path_audit.json::dollar_coefficient_calibration_sources`):

| coefficient | source | when supplied | when missing | invented? |
|---|---|---|---|---|
| `per_gpu_hour_price_usd` | `OperatorPricingPolicy.gpu_hour_price_per_type` OR `SafetyContext.gpu_hour_price` (existing operator global default) | Level-1 operator policy | falls back to existing operator global default | **No** |
| `energy_price_per_kwh_usd` | `OperatorPricingPolicy.energy_price_per_kwh_usd` (operator supplied; Aurelius already ingests real CAISO/PJM/ERCOT price data via `aurelius.forecasting.price_model`) | Level-1 operator policy | term reported uncalibrated; excluded from headline | **No** |
| `carbon_price_per_kg_usd` | `OperatorPricingPolicy.carbon_price_per_kg_usd` | Level-1 operator policy | term reported uncalibrated; excluded from headline | **No** |
| `cache_value_weight` / `MIGRATION_PENALTY` / `UTILITY_SCORE` / composite weights | **DOES NOT EXIST IN THIS MODULE** | n/a | n/a | **No** — none of these scalars exist anywhere in this PR. Cache value is DERIVED at scoring time from `predicted_reuse × prefill_throughput × operator_$/hr`. |

## 5. Modules introduced (PHASE 3)

- **`aurelius/forecasting/constraint_scorer_features.py`** — pure
  feature pipeline:
  - `OperatorPricingPolicy` — Level-1 operator policy slot (default
    empty / `None`).
  - `OptimumPriorTable` — Level-3 prior fitted from committed
    `optimum-benchmark/llm-perf-leaderboard` fixtures (9 GPU ×
    quantization configs, 1,400+ benchmark rows).
  - `GPUPowerPrior` — Level-3 prior from `Qinghao/AcmeTrace` IPMI
    Watts (real Tier-2 cluster telemetry, treated as cross-cluster
    prior at the integration layer).
  - `ScorerPriors` — bundle exposing the priors + the operator policy.
  - `SCORER_TERM_CATALOG` / `term_coverage_for_scorer` /
    `signal_level_for` — per-term × per-variant coverage matrix used
    by the audit + tests.
- **`aurelius/forecasting/constraint_shadow_scorer.py`** —
  `ConstraintShadowScorer.score(...)`. Returns the same
  `CandidateScore` shape as the production scorer plus a
  `breakdown` dict with `value_quality_by_term`, `term_formulae`,
  `uncalibrated_terms`, per-term sources. Five named variants
  (`A_existing`, `B_shadow_default_priors`,
  `C_shadow_with_ttft_p50_prior`, `D_shadow_with_cache_prefill`,
  `E_shadow_full`) flip a small set of feature flags.
- **`scripts/run_constraint_scorer_upgrade_audit.py`** — emits the
  audit JSON + term-coverage matrix.
- **`scripts/run_constraint_shadow_scorer_eval.py`** — runs the
  evaluation against a synthetic-but-realistic fleet (8 candidates × 4
  GPU types × 5 models × 80 requests).

## 6. Term formulae (PHASE 2)

Recorded in `CandidateScore.components["term_formulae"]` for every
shadow record:

```
expected_latency_s = queue_wait_s
                   + model_load_penalty_s
                   + adapter_load_penalty_s
                   + migration_cache_loss_penalty_s
                   + service_time_s
                   - cache_prefill_savings_s

service_time_s     = max(production_heuristic,
                         ttft_prior + output_tokens / decode_throughput)

migration_cache_loss_penalty_s
                   = prompt_tokens / prefill_throughput_tok_s
                   (when request.current_route != candidate.location_key)

cache_prefill_savings_s
                   = predicted_reuse_pct × prompt_tokens / prefill_throughput_tok_s

energy_kwh_per_request
                   = prefill_energy_kwh
                   + decode_energy_kwh_per_64_tokens × (output_tokens / 64)
                   (Optimum cell);  fallback: AcmeTrace mean_w × service_s / 3600 / 1000

expected_cost_usd  = (expected_latency_s / 3600) × per_gpu_hour_price_usd
                   + memory_pressure_cost
                   - cache_hit_value_usd   [only if operator-calibrated]
                   + energy_cost_per_request_usd   [only if operator-calibrated]

sla_met            = expected_latency_s <= sla_s   (production binary contract)
sla_latency_margin_s = sla_s - expected_latency_s  (quantitative indicator)

goodput_per_dollar = (1 if sla_met else 0) / expected_cost_usd
```

Every input on the right-hand side is either Level 1 (measurement /
operator policy) or Level 3 (prior). No learned utility coefficient
appears.

## 7. PHASE 4 — Evaluation result

### 7.1 Two-pass evaluation

The eval driver runs every variant twice:

- **Pass 1 — priors-only (BINDING headline).** No operator pricing
  policy is supplied. Every dollar coefficient that needs the operator
  to fill it in is reported uncalibrated; the upgraded scorer falls
  back to the production global default `gpu_hour_price` (which IS
  Level-1 operator policy, just at single-value granularity). This is
  the honest *"do the priors alone improve SLA-safe goodput/$?"*
  result.
- **Pass 2 — operator pricing policy supplied.** Illustrative
  per-GPU `$/hr` and `energy_price_per_kwh_usd` are passed in
  (every value labelled `operator_supplied`). Reported separately so
  reviewers can partition prior-driven improvement from
  operator-policy-driven improvement.

### 7.2 Pass 1 result (binding)

| variant | top-1 change rate | ranking change rate | SLA-safe goodput/$ delta vs A | SLA-safe count |
|---|---:|---:|---:|---:|
| A_existing | 0.000 | 0.000 | +0.00% | 32 / 80 |
| B_shadow_default_priors | 0.600 | 0.600 | **−11.35%** | 32 / 80 |
| C_shadow_with_ttft_p50_prior | 0.525 | 0.600 | **−11.35%** | 32 / 80 |
| D_shadow_with_cache_prefill | 0.600 | 0.600 | **−10.04%** | 32 / 80 |
| E_shadow_full | 0.525 | 0.600 | **−10.03%** | 32 / 80 |

**Binding final status: `diagnostic_only`.**

Honest interpretation: the Level-3 priors alone make the latency
estimate longer (Optimum / CARA service-time priors are higher than
the production 2.0 s heuristic floor for big models, and migration
penalties add seconds), which widens the cost denominator without a
compensating $-savings term. Cache savings can't translate to a
dollar improvement until the operator supplies a per-GPU price spread
(see §7.3).

SLA-safe count is unchanged across all variants: no candidate that
previously met SLA stops meeting it. This is the safety contract: the
shadow scorer never tightens the existing binary `sla_met` decision.

### 7.3 Pass 2 result (operator pricing supplied)

| variant | top-1 change rate | ranking change rate | SLA-safe goodput/$ delta vs A | SLA-safe count |
|---|---:|---:|---:|---:|
| A_existing | 0.000 | 0.000 | +0.00% | 32 / 80 |
| B_shadow_default_priors | 0.500 | 0.600 | +71.79% | 32 / 80 |
| C_shadow_with_ttft_p50_prior | 0.463 | 0.600 | +90.41% | 32 / 80 |
| D_shadow_with_cache_prefill | 0.500 | 0.600 | +74.14% | 32 / 80 |
| E_shadow_full | 0.463 | 0.600 | +93.19% | 32 / 80 |

**Pass 2 status: `shadow_ready_for_integration_review`** (mechanical
classification under the >5%-with-no-regression rule).

**This is NOT a production savings claim.** The +93% delta in Pass 2
is dominated by the per-GPU `$/hr` spread (a Level-1 operator-policy
input), not by the ML priors. Per-prior contribution (variant deltas
vs B, with operator policy):

- TTFT p50 prior (C − B): +18.6 pp
- Cache prefill savings (D − B): +2.4 pp
- Both (E − B):       +21.4 pp

So even *with* operator pricing, the cache forecaster contributes ~2
pp of goodput/$ improvement, and the TTFT p50 prior contributes ~19
pp. The other ~72 pp is the per-GPU price spread.

### 7.4 Partitioning the improvement (per the mission spec)

- Forecaster contribution (priors alone, no operator policy): **−10%**
  → does NOT improve SLA-safe goodput/$ on this fleet.
- Operator-policy contribution (per-GPU `$/hr` + energy_price):
  **≈+72 pp** of the headline.
- Forecasters add modest economic value ONLY when the underlying
  $-denominated coefficients are operator-calibrated.

## 8. PHASE 5 — Promotion decision

Per the mission ladder:

| condition | status |
|---|---|
| `pass_priors_only` improvement < 2% | **`diagnostic_only`** |
| `pass_with_operator_pricing` improvement > 5% | `shadow_ready_for_integration_review` (mechanical) |

**Binding decision: the upgraded scorer is `diagnostic_only`
pending:**

1. Operator-supplied per-GPU `$/hr` policy AND
2. Operator-supplied `energy_price_per_kwh_usd` AND
3. A pilot environment to validate that the prior translates to
   real economic savings (no public/HF trace can substitute for the
   operator's actual fleet pricing).

The two-pass result is recorded so a future PR that wires real
operator pricing has a clear reference point.

## 9. PHASE 6 — Subgroup audit

Per-GPU top-1 change rates surface where the upgraded scorer most
heavily reroutes. In Pass 2 (operator-policy):

- `t4`, `p100` (cheap GPUs) attract more routes → top-1 shifts to
  them for low-SLA-pressure requests.
- `a100` (expensive GPU) attracts fewer routes.

No SLA-safe count regression in any subgroup. Full subgroup table:
`shadow_scorer_eval.json::pass_*.aggregates.E_shadow_full.subgroup_by_gpu`.

## 10. What stays pilot-only

- Real measured energy per request from the production cluster
  (Optimum is a cross-hardware prior, not pilot data).
- Real per-GPU power draw on the production cluster (AcmeTrace is a
  cross-cluster prior).
- Real measured `cache_hit` per request (no HF dataset provides
  this; SwissAI is a `reuse_percentage` proxy).
- Cold-start latency per (model, GPU, cluster).
- Operator-supplied `energy_price_per_kwh_usd` from the actual
  utility bill or live spot feed.
- Operator-supplied per-GPU `$/hr` from the cloud invoice or
  internal chargeback policy.

## 11. Non-goals (binding)

- No production scheduler / residency / routing controller is modified
  (`tests/test_constraint_shadow_scorer_eval.py::test_no_production_module_modified`).
- No real execution. The shadow scorer's `executable_in_real_cluster`
  is pinned to `False` and the
  `ShadowScorerConfig` dataclass is frozen.
- No production-savings claim.
- No oracle as headline.
- No FIFO as headline.
- HF / public-trace data is NEVER labelled pilot telemetry.
- No invented economic constants — see §4.
- No utility-weighted composite — primary KPI is the single
  `goodput/$` quotient.

## 12. Reproducibility

```bash
# 1. Audit — writes scorer_path_audit.json + term_coverage_matrix.json.
python3 scripts/run_constraint_scorer_upgrade_audit.py

# 2. Evaluation — writes shadow_scorer_eval.json. Runs BOTH passes
#    (priors-only + operator-policy-supplied).
python3 scripts/run_constraint_shadow_scorer_eval.py
```
