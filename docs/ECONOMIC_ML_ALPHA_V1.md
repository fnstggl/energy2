# Economic ML Alpha Audit v1

> **Research / shadow PR.** No production scheduler / scorer / residency /
> frontier / overlay module is modified. No real execution. No production-
> savings claim. No oracle headline. No FIFO headline. No invented constants.
> Public / HF / market-overlay data is never treated as pilot telemetry.
> Missing cold-start / migration / cache-hit values are never silently zeroed.
>
> **Read first:**
> - `docs/ECONOMIC_OVERLAY_LAYER_V1_ANALYSIS.md` (the 187,505-row corpus)
> - `docs/ECONOMIC_OVERLAY_LAYER_V1.md` (the deterministic overlay formulas)
> - `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md` (operator-policy-only coefficients)
> - `docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`,
>   `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md`
> - `data/external/forecasting/economic_ml_alpha_v1/*.json`

## 1. Goal & headline

Determine whether Aurelius can find **real offline economic alpha** by
training modular component forecasters on the analysis-tier Economic
Overlay corpus, measured against the *strongest deterministic baseline*
for each target — not by maximising ML metrics, and not by training one
opaque black box.

**Headline finding.** The economic cost / goodput targets are
**deterministic transforms** of operational inputs × a public GPU-price
prior, so the overlay formula *is* their ground truth — ML cannot create
alpha by re-predicting them. Real offline alpha exists only where ML
forecasts a genuinely uncertain **upstream** signal:

| Target | Binding holdout | Best model | Δ vs strongest baseline | Status |
|---|---|---|---:|---|
| **cache_reuse_pct** | time | HGB | **+29.8% MAE** | **shadow_ready_for_integration_review**¹ |
| **peak_vram_gb** | high_tail | HGB | **+19.4% MAE** | **shadow_ready_for_integration_review**¹ |
| ttft_s | by_dataset | HGB | +2.4% MAE | promising_needs_validation |
| e2e_latency_s | by_dataset | HGB | −3660% (loses) | diagnostic_only |
| tpot_s | by_dataset | HGB | −21640% (loses) | diagnostic_only |
| energy_kwh | by_dataset | linear | −8.6% (loses) | diagnostic_only |
| high_reuse (clf) | time | logistic | no AUROC gain | diagnostic_only |
| estimated_gpu_cost_usd | by_dataset | HGB | −11067% (loses) | **diagnostic_only_deterministic_formula** |
| cold_start_cost_usd | — | — | — | **blocked_by_missing_labels** |
| migration_cost_usd | — | — | — | **blocked_by_missing_labels** |

¹ **Caveat (binding):** both shadow-ready targets are **single-dataset**
(cache_reuse → SwissAI only; peak_vram → Optimum only). They passed
temporal / tail holdouts but there is **no cross-dataset generalization
evidence**. A second dataset or pilot telemetry is required before
integration. This is recorded in `summary.json::shadow_ready_caveats`.

## 2. Rows used

187,505 overlay rows from the analysis-tier corpus
(`data/external/economic_overlay/analysis_corpus/*.jsonl`, gitignored):
CARA 60,000 · SwissAI 60,000 · AcmeTrace-power 40,000 · AcmeTrace-jobs
19,907 · CC-traces 5,000 · Optimum 2,598.

## 3. Phase 0 — exact overlap audit

`overlap_audit.json`. Answers to the 13 mission questions:

| # | Question | Answer |
|---|---|---:|
| 1 | rows with GPU cost | 187,505 |
| 2 | rows with prefill/decode cost | 127,598 |
| 3 | rows with cache value | 60,000 |
| 4 | rows with energy/carbon physical | 42,598 |
| 5 | rows with all four | **0** |
| 6 | all four + TTFT/TPOT/E2E | 0 |
| 7 | all four + queue | 0 |
| 8 | all four + cache reuse | 0 |
| 9 | all four + energy/kWh | 0 |
| 10 | cold-start cost | **0** |
| 11 | migration cost | **0** |
| 12 | real cache_hit | **0** |
| 13 | dominant datasets | gpu_cost: all 5; cache_value: SwissAI; energy/carbon: AcmeTrace+Optimum; latency: CARA+CC+Optimum |

**Critical structural finding:** the four economic terms never co-occur
in a single row (Q5 = 0) because each source supplies a *different*
slice — SwissAI gives cache value, AcmeTrace/Optimum give energy, CARA
gives latency. So an "all-overlap" joint economic model is **impossible**
on this corpus; **modular per-target training is the only honest path**
(mission decision rule).

**Signal-variability finding:** SwissAI `ttft_s`/`tpot_s`/`e2e_latency_s`
are **injected constants** (single value per dataset, 0.40 / 0.02 / 2.96),
and AcmeTrace-power `e2e` is a constant 15 s window. These are excluded
from latency training — training on a constant would fabricate a
degenerate "signal". Genuinely variable latency = CARA (60k real
per-request) + Optimum (2,598 real benchmark) + CC-traces (5k, TTFT/E2E
only).

## 4. Phase 1 — cold-start / migration realism-prior audit

`realism_prior_audit.json`. Every cold-start and migration term is
classified `measured | derived | prior | scenario_prior | simulator_prior
| missing`, each with `{value, source, source_type, confidence, formula,
calibration_notes, production_ready:false}`.

**Verdict:**
- **cold_start**: `blocked_by_missing_labels`. The only model-load signal
  is ejhusom's single-machine consumer-Ollama `load_duration_ns` (not in
  the analysis corpus, not server-class). AcmeTrace / Google Cluster give
  job SCHEDULE events, not model-load seconds. → simulator_prior sweep only.
- **migration**: `blocked_by_missing_labels`. A genuine cache-loss **proxy**
  exists in CC-traces (`migration_or_cache_loss_proxy`, KV block-hash loss)
  but it is **not realized** in the overlay rows (cache_loss_pct is unset).
  → simulator_prior sweep only.

These targets are **excluded from the headline goodput/$** and appear only
in the Phase-7 sensitivity sweep.

## 5. Phase 2 — target catalog

`target_catalog.json`. Per target: row_count, source datasets,
value_quality distribution, missingness, valid features, leaky/blocked
features, recommended model family, strongest baseline, holdouts,
`trainable_now`. Cost / goodput targets carry
`deterministic_ground_truth: true` + `ml_status: diagnostic_only`.
cold_start / migration carry `trainable_now: false`.

## 6. Phase 3 — feature contract

`aurelius/forecasting/economic_ml_features.py`. Strict per-target feature
resolution with a hard `LeakageError`:
- **Always blocked**: every `estimated_*` cost, `sla_safe_goodput*`,
  `sla_met`, `estimated_*_seconds`, and the provenance dicts.
- **Per-target blocks**: predicting `ttft_s` cannot see `tpot_s` / `e2e` /
  `throughput` / `output_tokens`; predicting `cache_reuse_pct` cannot see
  `cache_reuse_pct`; predicting `energy_kwh` cannot see `gpu_power_w`; etc.
- **`output_tokens`** is decision-time-unsafe for latency/cost targets it
  drives and is blocked for them (use `predicted_output_tokens` in
  production).
- Categorical encoding is deterministic from observed values only (no
  invented categories). Every feature row preserves `value_quality`.

## 7. Phases 4–5 — modular forecasters + holdouts

`aurelius/forecasting/economic_ml_forecaster.py` +
`scripts/run_economic_ml_alpha_v1.py`. Separate small models per target
(never one black box). Per target: deterministic baseline
(`GroupMedian(model,gpu)` for latency/resource, `PerModel/GlobalRate` for
cache) vs `HistGradientBoosting`, `RandomForest`, `Linear/Logistic`, with a
fallback-to-baseline wrapper that tracks fallback rate.

**Holdouts run:** random (decorative), by_dataset, by_gpu, time, high_tail.
**Binding selection:** first applicable of `time → by_dataset → high_tail`.
Random holdout is **never** binding (asserted by tests).

Illustrative spread (cache_reuse_pct MAE): baseline 0.60 → HGB 0.42 on the
**time** holdout (+29.8%); random holdout +25.4% (decorative). TTFT: HGB
beats `GroupMedian` by only +2.4% on by_dataset and *loses* on by_gpu
(−6.5%) — cross-hardware latency generalization is hard, hence
`promising_needs_validation` not shadow-ready.

## 8. Phase 6 — economic alpha evaluation

`economic_alpha_eval.json`. Variants A–I; primary baseline =
`B_overlay_deterministic_formula` (the strongest baseline and the ground
truth for cost targets); primary KPI = SLA-safe goodput/$ + economic regret.

Per-overlay-class goodput/$ (reported separately, never combined):
- `measured_same_record`: 0 rows (no operator $/hr).
- `cross_dataset_joined`: 144,907 rows, mean 29,699.
- `scenario_prior`: 42,598 rows, mean 98.7.

**Economic-alpha conclusion:** the only channel to economic alpha on this
corpus is forecasting upstream **latency** (TTFT +2.4% binding) and
**cache reuse** (+29.8% binding). Cost/goodput targets are deterministic
(`F`/`G` variants are diagnostic_only — ML over a derived target loses
badly to its own formula, −11067% on gpu_cost). No variant uses oracle or
FIFO as a headline; no production savings are claimed.

## 9. Phase 7 — cold-start / migration sensitivity (simulator_prior only)

`cold_start_migration_sensitivity.json`, labelled
`simulator_prior_only — NEVER headline`. Transparent sourced formulas:

```
cache_loss_prefill_penalty_s = prompt_tokens / prefill_throughput_tok_s * cache_loss_pct
migration_cost_usd = cache_loss_prefill_penalty_s * gpu_price/3600*gpu_count
                   + reroute_delay_s * gpu_price/3600*gpu_count
                   + warmup_delay_s  * gpu_price/3600*gpu_count
cold_start_cost_usd = model_load_duration_s * gpu_price/3600 * gpu_count
```

Every swept parameter carries `{source, source_type, confidence,
production_ready:false}` — `cache_loss_pct` (proxy, CC-traces),
`prefill_throughput_tok_s` (prior, Optimum), `gpu_price` (prior,
afhubbard), while `reroute_delay_s` / `warmup_delay_s` / `model_load_
duration_s` are `simulator_prior` with **no measured source**. 45-row
migration sweep + 12-row cold-start sweep. Findings: naive migration loses
badly as `cache_loss_pct → 1` at low prefill throughput; warm pools only
matter when `model_load_duration_s × price` dominates — and that parameter
**requires pilot telemetry** to calibrate.

## 10. Final status per target

```
cache_reuse_pct                       shadow_ready_for_integration_review (single-dataset caveat)
peak_vram_gb                          shadow_ready_for_integration_review (single-dataset caveat)
ttft_s                                promising_needs_validation
e2e_latency_s, tpot_s, energy_kwh     diagnostic_only (ML loses cross-dataset)
high_reuse                            diagnostic_only
estimated_gpu_cost_usd                diagnostic_only_deterministic_formula
cold_start_cost_usd, migration_cost_usd, migration_veto_label,
  timeout_or_failure_risk             blocked_by_missing_labels
```

**Is any model shadow-ready?** Yes — cache_reuse_pct and peak_vram_gb beat
their strongest baselines by >5% on a binding (time / high-tail) holdout
with no calibration regression. **But** both are single-dataset, so the
shadow-ready status is gated on a cross-dataset / pilot validation before
any integration review proceeds. TTFT is promising (modular latency
forecasting), not yet shadow-ready.

## 11. What needs pilot telemetry

`summary.json::pilot_telemetry_needed`: server-class `model_load_duration_s`
+ migration seconds, `reroute_delay_s`, `warmup_delay_s`,
`tail_latency_uplift_after_migration`, `migration_veto_label`, real
per-request `cache_hit`, operator per-GPU `$/hr`, operator energy tariff,
operator carbon price. A second independent dataset is needed to
cross-validate the shadow-ready cache_reuse / peak_vram models.

## 12. Files & reproducibility

New: `aurelius/forecasting/economic_ml_features.py`,
`aurelius/forecasting/economic_ml_forecaster.py`,
`scripts/run_economic_ml_alpha_v1.py`,
`data/external/forecasting/economic_ml_alpha_v1/{overlap_audit,
realism_prior_audit, target_catalog, trained_models, economic_alpha_eval,
cold_start_migration_sensitivity, summary}.json`,
`docs/ECONOMIC_ML_ALPHA_V1.md`, and 4 test files.

```bash
# requires the gitignored analysis corpus (regen via the overlay scripts if absent)
python3 scripts/run_economic_ml_alpha_v1.py
pytest tests/test_economic_ml_features.py tests/test_economic_ml_forecaster.py \
       tests/test_economic_ml_realism_priors.py tests/test_economic_ml_alpha_eval.py -q
```

## 13. Tests

46 tests across 4 files prove: no production module modified; leakage
blocker fires; value_quality preserved; exact overlap audit exists;
cold-start/migration missing labels are not zeroed; simulator_prior-only
targets cannot become headline (excluded from trainable, listed in blocked);
deterministic overlay baseline is the primary baseline and ML is compared
against it; random holdout is not binding; time/by-dataset/high-tail
holdouts run; deterministic cost targets are diagnostic_only; no
oracle/FIFO headline; no production-savings claim; pilot-telemetry needs
listed; single-dataset shadow-ready targets are caveated.
