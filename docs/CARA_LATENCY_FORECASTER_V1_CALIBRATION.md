# CARA Latency Forecaster v1 — Calibration + Tail-Safety Results

> **Forecasting safety / calibration PR.** No ML model is wired into
> any controller. No scheduler defaults are changed. No external savings
> number is quoted. The goal is to make the forecaster **safe enough to
> trust in shadow mode**, not to beat the prior PR's headline.
>
> **Read first:**
> - `docs/RESULTS.md`
> - `docs/FORECAST_LEVERAGE_AUDIT.md`
> - `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md`
> - `docs/CARA_LATENCY_FORECASTER_V1_RESULTS.md`
> - `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`

## 1. Scope (binding)

- **Inputs:** the CARA train_flat ingestion (76,825 rows, gitignored
  analysis sample). The forecaster v1 (PR #126) artefacts under
  `data/external/forecasting/cara_latency_forecaster_v1/` are read
  for the baseline reproduction.
- **Outputs:** a calibration module
  (`aurelius/forecasting/cara_latency_calibration.py`), a driver
  script that re-runs every (target × quantile × holdout) cell with
  calibration + tail-safety + subgroup audit, and the binding final
  decision table (PHASE H).
- **Out of scope:** scheduler changes, controller execution, frontier
  module wiring, any external savings claim. A negative finding is
  acceptable and reported honestly.
- **Safety floor:** the deterministic per_instance_type p{q} baseline
  is the **strongest realistic baseline** under the mission spec. The
  calibration framework either improves on it under the binding
  thresholds, or the cell stays diagnostic_only / baseline_fallback.

## 2. Calibration methods (PHASE B)

All four classes live in
`aurelius/forecasting/cara_latency_calibration.py`. Every `fit()` only
accepts `(X_cal, y_cal)` — the test holdout's labels are never seen,
enforced by signature inspection
(`test_calibrators_do_not_see_test_labels_by_construction`).

| Class | Calibrated prediction | Coverage handle | Where it's strong |
|---|---|---|---|
| `ConservativeMultiplierCalibration` | `m * raw_pred`, m = q-th quantile of `y_cal / raw_pred`, clamped `m >= 1.0` | scales with predicted magnitude | when residual / prediction is multiplicative |
| `QuantileResidualCalibration` | `raw_pred + r_q`, r_q = q-th quantile of `y_cal - raw_pred` (clamped `>= 0` by default) | additive shift | when residuals are stationary across magnitude |
| `SplitConformalUpperBound` | `raw_pred + q_level(s)`, q_level = `ceil((n+1)*alpha)/n` quantile of `s = y_cal - raw_pred` | distribution-free under exchangeability | when target finite-sample coverage is binding |
| `BaselineFallbackGate` | `max(ml_pred, baseline_pred)` or `baseline_pred` when ML is `< ood_tolerance_x * baseline_pred` | explicit safety floor | when ML may hallucinate low-latency on OOD input |

The driver applies all four to every (target × quantile × holdout)
cell, then picks a **preferred calibrator** by quantile:

- **p50:** `conservative_multiplier` (multiplicative residual fits the
  CARA median-latency distribution best).
- **p95 / p99:** `split_conformal_with_baseline_floor`
  (SplitConformalUpperBound wrapped by BaselineFallbackGate). The floor
  ensures the calibrated prediction never drops below the strongest
  baseline.

## 3. Tail-safety metrics (PHASE C)

Each cell reports, per the JSON summary:

```
raw_model_metrics:                  pinball_loss, alpha_pct_vs_baseline
strongest_baseline_metrics:         pinball_loss
calibration_variants[*]:            pinball_loss, empirical_coverage, alpha_pct
calibration_diagnostics[*]:         multiplier / residual_quantile / q_level
tail_safety_metrics:                target_coverage, empirical_coverage,
                                    coverage_error, undercoverage_rate,
                                    overprediction_ratio, mean_conservatism,
                                    p95_residual, p99_residual
fallback_used_rate_on_test:         fraction of rows that hit the floor
```

The mission spec's binding **promotion thresholds** are pre-registered
in `PROMOTION_THRESHOLDS` and emitted into the JSON for auditability:

| target / quantile | time-holdout pinball improvement | min empirical coverage | max undercoverage rate |
|---|---:|---:|---:|
| TTFT p50 | 10.0% (+ random & by_instance ≥ 10%) | n/a | n/a |
| TTFT p95 | 10.0% | 0.93 | 0.07 |
| TTFT p99 | 5.0% | 0.975 | 0.025 |
| E2E p95 | 5.0% | 0.93 | 0.07 |
| E2E p99 | 5.0% | 0.975 | 0.025 |

## 4. Subgroup safety audit (PHASE D)

Every cell carries per-subgroup audits for `instance_type`, `gpu_type`,
`model_size`, `prompt_token_bin`, `queue_depth_bin`, `kv_util_bin`.
Per-subgroup status is one of:

- `PASS` — improvement ≥ -5%, coverage ≥ target − 0.02, row_count ≥ 100.
- `INSUFFICIENT_SAMPLE` — row_count < 100 (p95/p99).
- `REGRESSION` — improvement < -5% vs baseline.
- `UNDERCOVERED` — coverage below threshold.
- `FALLBACK_REQUIRED` — fallback gate fired on this subgroup.

A cell is rejected_regression if **any** major subgroup is REGRESSION
on the time-holdout cell.

## 5. PHASE E ordering (binding)

The `classify_tail_status` function applies gates in this order. The
first failure determines the final status; later gates never overwrite
an earlier failure.

1. **Leakage check.** If any feature pipeline could have emitted a
   leakage column, mark `rejected_regression`.
2. **No test-label calibration.** Signature inspection guarantees this
   structurally.
3. **Time-holdout pinball improvement** vs threshold.
4. **Time-holdout empirical coverage** (p95/p99 only).
5. **Time-holdout undercoverage rate** (p95/p99 only).
6. **Subgroup regression / undercoverage** on time-holdout.
7. **Fallback fired on > 25% of time-holdout rows** → `baseline_fallback`.

Random and by_instance_type holdouts are diagnostic-only at this layer:
they cannot promote a model, but they can downgrade one (e.g. if the
random holdout regresses, the cell drops to `diagnostic_only`).

## 6. Final decision table (PHASE H)

| target | quantile | raw time-holdout α% | calibrated time-holdout α% | time-holdout coverage | fallback fired (on time) | **final status** |
|---|---|---:|---:|---:|---:|---|
| `actual_ttft_s` | p50 | +41.54% | +41.60% | 0.432 | n/a | **`shadow_ready`** |
| `actual_ttft_s` | p95 | +5.90% | +19.52% | 0.954 | 0.428 | `diagnostic_only` (subgroup undercoverage) |
| `actual_ttft_s` | p99 | -31.46% (regression) | +10.92% | 0.984 | 0.673 | `baseline_fallback` |
| `actual_e2e_latency_s` | p50 | +2.65% | +2.62% | 0.508 | n/a | `diagnostic_only` (no p50 E2E threshold) |
| `actual_e2e_latency_s` | p95 | +1.29% | +0.20% | 0.954 | 0.428 | `diagnostic_only` (time α < 5%) |
| `actual_e2e_latency_s` | p99 | -2.12% | +0.22% | 0.992 | 0.502 | `diagnostic_only` (time α < 5%) |

### What changed vs PR #126

- **TTFT p50:** was already `candidate_for_shadow_integration`. The
  calibration module re-validates it as `shadow_ready` under the
  stricter PHASE E pipeline (time + random + by_instance pinball ≥ 10%,
  no leakage, calibration did not increase error by > 5%). **Status
  promoted from `candidate_for_shadow_integration` →
  `shadow_ready`.**
- **TTFT p95:** was `diagnostic_only` in PR #126 (raw time α was +7%,
  below the 10% threshold). After split-conformal-with-baseline-floor
  calibration the time α is +19.5% and global coverage is 0.954
  (above 0.93). Global gates pass — but the subgroup safety audit
  catches **at least one major subgroup below the coverage threshold**,
  blocking promotion. Honest stay at `diagnostic_only`.
- **TTFT p99:** was `promising_needs_validation` in PR #126 with a
  raw -17% time regression. Calibration moves the headline α to +10.9%
  on time, BUT the fallback gate fires on **67% of time-holdout rows**
  — the model's raw predictions routinely drop below the baseline, so
  the conformal floor is doing the prediction work. The framework
  labels this `baseline_fallback`: the system should ship the
  per_instance_type p99 baseline directly, not the ML model.
- **E2E p95 / p99:** were `diagnostic_only` in PR #126. The
  calibration framework reproduces this finding (time α stays
  ~+0.2%, well below the 5% threshold). The deliberate exclusion of
  `actual_output_tokens` as a feature is the binding ceiling.
- **E2E p50:** has no documented promotion threshold (TTFT p50 is the
  only p50 cell the mission spec calls out). Stays
  `diagnostic_only`.

## 7. PHASE F artefacts

- **Code:**
  - `aurelius/forecasting/cara_latency_calibration.py`
  - `scripts/run_cara_latency_calibration_tail_safety.py`
- **JSON summary:**
  - `data/external/forecasting/cara_latency_forecaster_v1/calibration_tail_safety_summary.json`
- **Tests:**
  - `tests/test_cara_latency_calibration.py` (28 unit tests)
  - `tests/test_cara_latency_tail_safety.py` (20 artefact + invariant tests)
- **Docs:**
  - `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md` (this file)

## 8. PHASE I summary

- **One model promoted:** TTFT p50 → `shadow_ready` (eligible for
  shadow wiring into `aurelius/frontier/dynamic_estimator.py` priors
  path **only**, never the controller execution path).
- **One model gracefully demoted with explicit signal:** TTFT p99 →
  `baseline_fallback` (the framework correctly tells the system to
  ship the deterministic baseline because the ML model regresses on
  time-shifted data).
- **One model held back by subgroup safety:** TTFT p95 → still
  `diagnostic_only` despite cleared global thresholds, because of a
  subgroup-coverage failure.
- **Three models stay diagnostic_only:** E2E p50/p95/p99, blocked by
  feature-set design (no `actual_output_tokens`).

The calibration layer adds three real improvements:

1. **Honest tail safety.** Split-conformal-with-floor never lets the
   ML model under-predict below the baseline, so latency-risk
   estimates can be trusted.
2. **Explicit fallback semantics.** When fallback fires on more than
   25% of time-holdout rows the cell is labelled `baseline_fallback`
   — a clear signal to ship the deterministic baseline instead.
3. **Subgroup-first promotion.** A cell that looks good on the
   aggregate but regresses in a major subgroup is blocked from
   shadow promotion.

## 9. Non-goals (binding)

- No scheduler / robust energy engine / controller change.
- No ML model wired into live routing.
- No external-savings number quoted.
- Promoting TTFT p50 to `shadow_ready` does **not** authorise wiring it
  into any controller's execution path. Shadow integration is a
  *separate* PR with its own pre-registered gates.
