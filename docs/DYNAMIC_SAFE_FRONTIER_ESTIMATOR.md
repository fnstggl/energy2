# Dynamic Safe Frontier Estimator — v1

> **Opt-in. Disabled by default. Shadow / simulator only.** Simulator / shadow-mode evidence only — **NOT production savings** (`docs/RESULTS.md` §8). The static frontier controller and the `constraint_aware` engine default rho (0.65) are **unchanged** by this module.

This document specifies the **Dynamic Safe Frontier Estimator v1** — a telemetry-driven companion to the static Safe Utilization Frontier Controller (`aurelius/frontier/`). The dynamic estimator turns recent observed telemetry into a per-tick safe-utilization estimate and emits a recommendation-only `DynamicFrontierDecision` (RAISE / KEEP / LOWER / INSUFFICIENT_TELEMETRY).

- **Read first:** `docs/RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/PILOT_TELEMETRY_CONTRACT.md`.

## 1. Scope (binding)

- **Opt-in only.** The static frontier controller remains the committed default. The dynamic estimator is wired in via the existing `constraint_aware` × frontier integration shim (`aurelius/constraints/frontier_integration.py`) and is **disabled by default** there too.
- **No production mutation.** Decisions are recommendation-only at construction (`executable_in_real_cluster=False`). Real-cluster execution requires the same explicit opt-in as the static path (`execute_frontier_decision(..., allow_real_execution=True)` plus a non-stub executor).
- **No future leakage.** The estimator may read only the telemetry window the caller passes in. Streaming-replay validation rolls the window forward one tick at a time so the t-th decision sees only t' ≤ t telemetry.
- **No ML training in v1.** Risk scores come from **deterministic / statistical** heuristics (EWMA, slopes, coefficient-of-variation, Erlang-C tails). They are documented as such and are not learned.
- **Robust energy engine is unchanged.** This module touches only `aurelius/frontier/` and uses the existing physics in `aurelius/traces/backtest.py` for benchmarks.
- **No new datasets** are ingested for product use; the benchmark uses the already-integrated Azure 2024 trace.
- **No optimizer / safety / robust-energy-engine constant** is tuned to force a benchmark win.

## 2. Architecture

```
recent telemetry window  ──►  validate_dynamic_window
                              │
                              ▼
                       estimate_dynamic_frontier
                       (Erlang-C calibration + risk)
                              │
                              ▼            (candidate sweep + risks)
                     DynamicFrontierEstimate
                              │
                              ▼
                      choose_dynamic_rho
                       (deadband, hysteresis,
                        churn suppression)
                              │
                              ▼
                     DynamicFrontierDecision
                       (RAISE / KEEP / LOWER /
                        INSUFFICIENT_TELEMETRY)
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  dynamic_estimate_to_frontier_decision     │
        │  ──►  static-compatible FrontierDecision   │
        │  (consumed by constraint_aware integration)│
        └────────────────────────────────────────────┘
                              │
                              ▼
          execute_frontier_decision (shadow / simulator;
                       real execution disabled by default)
```

## 3. Models — `aurelius/frontier/dynamic_models.py`

| dataclass | purpose |
|---|---|
| `ServingTelemetryTick` | one observed tick of LLM-serving telemetry; **every numeric field is `Optional` and stays `None` when missing** (no zero-fill) |
| `DynamicFrontierCandidate` | predicted outcome at one candidate rho; carries `predicted_sla_risk_probability` and `predicted_queue_blowup_probability` in `[0, 1]` |
| `DynamicFrontierEstimate` | window → frontier estimate: current rho, recommended rho, slope, risk-at-current, headroom, candidate sweep |
| `DynamicFrontierDecision` | recommendation-only RAISE / KEEP / LOWER / INSUFFICIENT_TELEMETRY |

## 4. Telemetry window builder — `aurelius/frontier/dynamic_telemetry.py`

Accepts:

1. `aurelius/traces/replay.ArrivalTick` (used by the Azure 2024 streaming-replay benchmark).
2. plain dict telemetry (pilot bring-up).
3. native `ServingTelemetryTick`.

`validate_dynamic_window(window, min_ticks, required_fields, min_field_coverage)` returns a structured `TelemetryWindowValidation`. The default required fields are `("observed_rps", "queue_p99_ms", "active_replicas")`.

## 5. Deterministic risk estimator — `aurelius/frontier/risk.py`

| function | signal |
|---|---|
| `estimate_sla_risk` | weighted combination of (a) timeout-EMA proximity to threshold, (b) timeout-EMA slope, (c) rho-jump penalty |
| `estimate_queue_blowup_risk` | (a) queue-p99-EMA proximity to threshold, (b) queue-p99 slope, (c) rho-jump penalty, (d) RPS burstiness (CV) |
| `estimate_required_headroom` | RPS CV → headroom in `[0.05, 0.35]` |
| `estimate_churn_risk` | per-tick scale-events + churn deltas |

Risk fields are documented `[0, 1]` probability-like scores. They are **deterministic / statistical**, **not trained ML**. Every component is transparent and the weights are pre-registered.

## 6. Estimator — `aurelius/frontier/dynamic_estimator.py`

Algorithm:

1. Validate window (`min_ticks=8`, default required fields).
2. Estimate current rho from `mean_utilization` EMA.
3. Build candidate set: global grid (`[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`) ∪ local grid around current (step 0.05, range ±0.10), clamped to `[profile.min_rho, profile.max_rho]`.
4. For each candidate:
   - Predict queue p99 via workload-calibrated Erlang-C tail:
     `C_queue = q99_observed · (1 − ρ_observed)`, predicted `q99(R) = C_queue / (1 − R)`.
   - Same calibration for timeout share and latency p99 (when present).
   - Predict GPU-hours and goodput/$ via replica scaling `replicas ~ ρ_observed / R`.
   - Estimate SLA-risk and queue-blowup-risk.
   - Classify SAFE / UNSAFE / INSUFFICIENT_TELEMETRY (hard thresholds + soft risk gates).
5. Estimate frontier slope (∆goodput/$ per +0.01 rho near current).
6. Pick best safe candidate by predicted goodput/$.
7. Apply optional conservative margin (step back from a candidate adjacent to UNSAFE).
8. Return `DynamicFrontierEstimate` with full evidence.

## 7. Controller — `aurelius/frontier/dynamic_controller.py`

Rules, in order:

1. Estimator fallback → INSUFFICIENT_TELEMETRY.
2. `risk_at_current_rho >= lower_rho_risk_threshold (default 0.75)` → LOWER_RHO.
3. Deadband: `|Δρ| ≤ deadband_rho (0.05)` AND ΔKPI ≤ `deadband_kpi_pct (2 %)` → KEEP_RHO.
4. Churn suppression: workload exhibits `churn_high` or `scale_events_high` → suppress RAISE_RHO.
5. Hysteresis: previous action vs proposed flip — flip with magnitude below `deadband_rho · hysteresis_multiplier (2.0)` → suppress.
6. Direction: recommended > current → RAISE_RHO; < → LOWER_RHO; = → KEEP_RHO.

## 8. Shadow logging + outcome — `aurelius/frontier/dynamic_shadow.py`

`DynamicFrontierShadowLog` is JSONL round-trippable. `DynamicFrontierOutcome` carries `rho_error`, per-metric prediction errors, and `was_safe`. `compare_prediction_to_observed(...)` joins a log with the realized metrics — the substrate for any future pilot calibration.

## 9. Integration with the static path — `aurelius/frontier/dynamic_adapter.py`

`dynamic_estimate_to_frontier_decision(decision, candidate_points=...)` maps a `DynamicFrontierDecision` into a static-compatible `FrontierDecision`:

| dynamic action | static action |
|---|---|
| `RAISE_RHO` | `RECOMMEND_RHO` |
| `LOWER_RHO` | `LOWER_RHO` |
| `KEEP_RHO` | `KEEP_RHO` |
| `INSUFFICIENT_TELEMETRY` | `INSUFFICIENT_TELEMETRY` |

The returned `FrontierDecision` is recommendation-only at construction. The existing `constraint_aware` × frontier integration shim consumes either static or dynamic decisions through the same execute hook.

## 10. Azure 2024 streaming-replay benchmark (summary)

`scripts/run_azure_2024_dynamic_frontier.py` runs **offline streaming replay** on the Azure LLM 2024 trace. Each per-tick decision sees only telemetry from t' ≤ t. Results land in `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md` and `data/external/azure_llm_2024/processed/azure_2024_dynamic_frontier_summary.json`.

The benchmark compares:

- `constraint_aware_static` (rho 0.65)
- `static_frontier_controller` (rho 0.75 — committed Azure 2024 winner)
- `sla_aware` (rho 0.50) and `utilization_aware` (rho 0.85)
- `dynamic_frontier_estimator` at rolling windows of 30 / 60 / 180 minutes
- `oracle_realized_optimal_ANALYSIS_ONLY` — post-hoc upper bound; **NEVER** the headline baseline (it sees the future)

The benchmark reports per-window goodput/$, timeout %, queue p99, GPU-hours, mean rho, action distribution (RAISE / KEEP / LOWER counts), convergence tick, and frontier-recovery percentage (how much of the alpha between `constraint_aware` and the oracle the dynamic estimator recovered).

## 11. Hard rules / non-goals

- ❌ Dynamic estimator is **not** default.
- ❌ Static frontier controller is **not** removed.
- ❌ `constraint_aware` engine default rho (≈ 0.65) is **not** changed.
- ❌ No ML training in v1.
- ❌ No new datasets ingested.
- ❌ Robust energy engine is **not** modified.
- ❌ No production mutation.
- ❌ No future leakage.
- ❌ No claims of production savings.
- ❌ No safety-gate weakening.
- ❌ No constants tuned per trace to force benchmark wins.

## 12. Remaining gaps before any pilot or default-on promotion

- **Pilot telemetry calibration.** The Erlang-C calibration constant (`C_queue`, `C_timeout`) is workload-specific and noisy on short windows. Pilot data should also calibrate the deadband / hysteresis defaults.
- **Per-tenant safety thresholds.** `max_timeout_pct=10`, `max_queue_p99_ms=2000` are pre-registered defaults from the Azure 2024 audit. Per-tenant SLAs must override these explicitly.
- **Real executor.** Real-mode execution remains a deliberate stub. Promoting it requires the binding-boundary work described in `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` §"Real-mode execution boundary".
- **Trained risk model.** A pilot-data-calibrated logistic / quantile regression for the risk scores is plausible but **out of scope for v1**. Any future learned component must keep the same interface (`probability ∈ [0, 1]`, reason codes, confidence label) so the controller logic stays unchanged.
