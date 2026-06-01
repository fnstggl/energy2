# CARA TTFT Tail Forecasting with Queue Features — Results

> **Shadow/research forecasting experiment.** No scheduler, controller,
> or robust-energy-engine change. No production-savings claim. A negative
> result is acceptable and reported honestly.
>
> **Read first:** `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md`,
> `docs/CARA_QUEUE_WAIT_FORECASTER_V1_RESULTS.md`,
> `docs/FORECAST_LEVERAGE_AUDIT.md`, `docs/RESULTS.md`.

## 1. Hypothesis

PR #127 left TTFT p95 at `diagnostic_only` (subgroup undercoverage) and
TTFT p99 at `baseline_fallback` (fallback fired on 67% of time-holdout
rows). The leverage audit hypothesised that **tail latency is
queue-driven**, so adding a queue-wait forecast as a feature might
improve the TTFT tail. This experiment tests that hypothesis.

## 2. Leakage-safe method

For each TTFT holdout (random / by_instance / time):

1. Train a queue-wait forecaster (HGB quantile p50/p95/p99) on the TTFT
   **train** fold only.
2. **Cross-fit (2-fold)** within the TTFT train fold to produce
   **out-of-fold** queue predictions for the train rows — so the TTFT
   model never trains on in-sample (optimistic) queue features.
3. Predict queue p50/p95/p99 + uncertainty for the TTFT **test** fold
   using a queue model that never saw the TTFT test labels.
4. Append 5 queue features: `predicted_queue_p50`, `predicted_queue_p95`,
   `predicted_queue_p99`, `queue_forecast_uncertainty` (= p99 − p50),
   `queue_pressure_score` (= p95).
5. Retrain TTFT p95/p99 with the augmented feature set; apply the same
   split-conformal + baseline-floor calibration + subgroup audit.
6. Compare against the per-instance-type baseline AND the prior
   calibrated TTFT p95/p99 from PR #127.

`queue_features_are_out_of_fold = True` and
`queue_target_is_derived_proxy = True` are recorded in the summary JSON.

## 3. Final decision table (PHASE H)

| target | quantile | prior status | prior time α | new time α (with queue) | Δ | new coverage | new fallback | **final decision** |
|---|---|---|---:|---:|---:|---:|---:|---|
| `actual_ttft_s` | p95 | `diagnostic_only` | +19.52% | +19.40% | **-0.12%** | 0.954 | 0.428 | `diagnostic_only` |
| `actual_ttft_s` | p99 | `baseline_fallback` | +10.92% | +9.73% | **-1.19%** | 0.984 | 0.634 | `baseline_fallback` |

## 4. Honest negative finding

**Adding queue-forecast features does NOT improve TTFT p95/p99 tail
safety.** The time-holdout pinball improvement is essentially unchanged
(p95: -0.12 pp; p99: -1.19 pp) and both quantiles keep their prior
status:

- **TTFT p95 stays `diagnostic_only`** — the queue features add no
  signal that lifts the time-holdout improvement to the 10% threshold,
  and the subgroup undercoverage that blocked it in PR #127 is unmoved.
- **TTFT p99 stays `baseline_fallback`** — fallback still fires on 63%
  of time-holdout rows (> 25% threshold). The conformal floor is doing
  the prediction work; queue features don't change that.

## 5. Why queue features don't help

The CARA derived queue-wait proxy is small (p95 = 0.21 s) and is itself
**not forecastable beyond the instance-type prior** (see
`docs/CARA_QUEUE_WAIT_FORECASTER_V1_RESULTS.md`). Moreover, the queue
forecast is built from the **same decision-time scheduler-state
features** (`num_running`, `pending_*_tokens`, `kv_cache_utilization`,
EMA throughput) that the TTFT model already uses directly. Adding a
queue forecast derived from those features is largely redundant — it
carries no information the TTFT model didn't already have.

The leverage-audit hypothesis ("tail latency is queue-driven") is
**not supported** on CARA: tail TTFT is GPU/model-driven (the 9× p99
spread across GPU types from PR #126), not queue-driven. CARA's research
cluster simply doesn't run hot enough to build a queue.

## 6. What remains blocked

- **TTFT p95 → shadow_ready_tail_candidate** requires either (a) more
  recent CARA data to fix the time-holdout subgroup undercoverage, or
  (b) a **measured** queue-wait signal from pilot telemetry (not a
  derived proxy).
- **TTFT p99 → shadow_ready_tail_candidate** requires fixing the
  time-holdout regression at the source; calibration + queue features
  cannot repair it.

Both paths point to the same conclusion as the leverage audit: the next
real unlock is **pilot telemetry** (measured queue wait, SLA labels,
GPU utilisation), not another feature-engineering pass on CARA.

## 7. Non-goals

No scheduler / controller change. No model promoted. No production claim.
Queue features are out-of-fold and the queue target is a labelled derived
proxy, never a measured queue wait.
