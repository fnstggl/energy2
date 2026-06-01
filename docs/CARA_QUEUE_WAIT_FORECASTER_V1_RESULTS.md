# CARA Queue-Wait Forecaster v1 — Results

> **Shadow/research forecasting PR.** No scheduler, controller, or
> robust-energy-engine change. No production-savings claim. No model is
> wired into any control loop. A negative result here is acceptable and
> reported honestly.
>
> **Read first:** `docs/FORECAST_LEVERAGE_AUDIT.md`,
> `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md`,
> `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md`, `docs/RESULTS.md`.

## 1. Target definition (honest)

CARA carries **no measured queue-wait label**. Its `num_waiting` field is
~always 0 (1.09% nonzero on `train_queue_details`) because vLLM
continuous batching absorbs load — there is no meaningful "waiting queue
length" to forecast.

We therefore forecast a **derived proxy**, explicitly labelled:

- **`derived_queue_wait_s`** (`field_quality = derived`):
  `(completion_timestamp_s − prediction_timestamp_s) − actual_e2e_latency_s`,
  clamped `>= 0`. This is the gap between total scheduling-to-completion
  wall-clock and the client-measured serving latency — the time the
  request spent **not** actively being served (dispatch / queue delay).
  Empirically always non-negative on CARA. Distribution: p50 = 0.07 s,
  p95 = 0.21 s, p99 = 0.43 s.

This is **NOT** a measured queue wait. The summary JSON sets
`target_definition.is_real = False`, `is_derived = True`,
`measured_queue_wait_available = False`.

A second target, `queue_pressure_score` (`field_quality = synthetic`),
is a hand-built decision-time score; it is retained as a diagnostic only
(it is trivially predictable from the same features and is not a primary
forecast target).

The timestamp + e2e fields used to **construct** the target are in
`QUEUE_LEAKAGE_TARGET_FIELDS` and are **never** used as features.

## 2. Dataset + features

- **Source:** `data/external/hf/asdwb__cara_latency_prediction/train_queue_details/processed/analysis_sample.jsonl`
  (38,509 rows, gitignored, regenerable via the analysis-tier audit script).
- **Features:** the same 24-numeric + 8-categorical decision-time feature
  set as the latency forecaster (scheduler state at decision time is
  exactly what predicts queue pressure).

## 3. Baselines

- `per_instance_type_queue_p{q}` — **strongest realistic baseline**.
- `per_model_gpu_queue_p{q}` — secondary.
- `num_waiting_baseline` — `base_service_time × num_waiting`
  (degenerates to ~0 on CARA, the honest naive operator baseline).
- `queue_depth_extrapolation` — `per_running_service_time × num_running`.

## 4. Models + calibration

HistGradientBoosting quantile regressors (p50/p95/p99), calibrated with
`SplitConformalUpperBound` + a `per_instance_type` baseline floor (p95/p99).
Three holdouts: random / by_instance_type / time. Time-holdout is the
binding gate.

## 5. Final decision table (PHASE H)

| target | quantile | time improvement | empirical coverage | fallback rate | **final status** |
|---|---|---:|---:|---:|---|
| `derived_queue_wait_s` | p50 | +0.35% | 0.497 | 0.000 | `diagnostic_only` (time < 10%) |
| `derived_queue_wait_s` | p95 | -2.14% | 0.955 | 0.597 | `diagnostic_only` (time < 10%) |
| `derived_queue_wait_s` | p99 | -22.63% | 0.996 | 0.598 | `diagnostic_only` (time < 5%) |

## 6. Honest negative finding

**The queue-wait forecaster does not beat the per-instance-type baseline
on the time-holdout at any quantile.** Why:

1. The derived dispatch delay is small (p50 = 0.07 s) and dominated by
   per-instance fixed overhead, which the `per_instance_type_queue_p{q}`
   baseline already captures.
2. `num_waiting` is ~always 0, so there is no genuine queueing signal to
   learn — the residual variance the ML model could exploit is mostly
   instance-specific noise.
3. On the **time-holdout** the p95/p99 ML models *regress* (-2% / -23%),
   consistent with the same temporal non-stationarity that blocked the
   TTFT tail models in PR #127.

This is a **valuable negative result**: it tells us that CARA's dispatch
delay is not a forecastable queue signal beyond the instance-type prior.
A real queue-wait forecaster needs **measured queue wait** — which only
pilot telemetry can supply (see `docs/FORECAST_LEVERAGE_AUDIT.md` gap
analysis: `queue_wait` is `pilot_telemetry_only`).

## 7. What this unblocks

Even as a negative result, the queue forecaster:

- Establishes the honest **derived_queue_wait_s** target + leakage-safe
  pipeline, reusable when real queue telemetry lands.
- Provides the out-of-fold queue predictions consumed by the TTFT-tail
  experiment (`docs/CARA_TTFT_TAIL_WITH_QUEUE_FEATURES.md`).
- Confirms that tail latency in CARA is **not** primarily queue-driven —
  it is GPU/model-driven (the 9× p99 spread across GPU types found in
  PR #126).

## 8. Non-goals

No scheduler / controller / robust-energy-engine change. No queue model
wired into routing or admission. No production claim. The
`derived_queue_wait_s` proxy is never presented as a measured queue wait.
