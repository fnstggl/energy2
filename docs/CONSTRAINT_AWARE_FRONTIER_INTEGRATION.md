# `constraint_aware` × Safe Utilization Frontier Controller — Integration v1

> **Opt-in. LLM-serving only. Disabled by default. Real-cluster execution disabled by default.** Simulator / shadow-mode evidence only — **NOT production savings** (`docs/RESULTS.md` §8).

This document specifies how the Safe Utilization Frontier Controller v1 (`aurelius/frontier/`) is wired into the `constraint_aware` policy (`aurelius/traces/backtest.py`) as a guarded, opt-in utilization-target selector.

- **Read first:** `docs/RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/PILOT_TELEMETRY_CONTRACT.md`.

## 1. Scope (binding)

The integration:

- is **opt-in** — controlled by `FrontierIntegrationConfig(enabled=...)`; the default is `enabled=False`;
- applies to **LLM serving workloads only** — by `allowed_workload_types` (defaults: `inference_standard`, `interactive_inference`, `llm_serving`, `standard_interactive_inference`, `critical_interactive_inference`);
- runs in **shadow / simulator mode** by default — `shadow_only=True`, `allow_real_execution=False`;
- **selects the utilization (rho) target only**. It does **not** replace energy arbitrage, queue/SLA gates, residency gates, topology gates, thermal gates, telemetry-confidence gates, or migration safety gates — those continue to run downstream of the rho selection;
- **falls back to the existing `constraint_aware` default rho (0.65)** on any ineligibility, low confidence, missing telemetry, unsafe controller recommendation, or estimator / controller error;
- does **not** modify the robust energy engine;
- does **not** make the controller globally default;
- does **not** apply to training, batch GPU scheduling, packing, Philly, or Alibaba GPU traces.

The single hard invariant: **with `enabled=False`, every existing benchmark / test / call path produces byte-for-byte identical output** (asserted by `tests/test_constraint_aware_frontier_integration.py::test_default_constraint_aware_unchanged_byte_for_byte`).

## 2. Architecture

```
constraint_aware
    │
    ├─ if frontier_integration.enabled and eligible:
    │     ├─ build WorkloadFrontierProfile from workload_metadata
    │     ├─ estimate_frontier(profile, telemetry_window)
    │     ├─ choose_safe_utilization_target(profile, points, current_rho=0.65)
    │     ├─ accept selected_rho iff action ∈ {RECOMMEND_RHO, KEEP_RHO}
    │     │      AND selected_point.safety_status == SAFE
    │     └─ otherwise → fallback to default rho 0.65 with reason
    │
    └─ existing constraint_aware sizing / hysteresis / SLA trim continues
       on the resulting rho — every existing safety gate still runs
```

The integration only changes one input to `_size_for_target`: the target rho.

## 3. `FrontierIntegrationConfig`

Defined in `aurelius/constraints/frontier_integration.py`.

| field | default | notes |
|---|---|---|
| `enabled` | `False` | master switch; default preserves existing behaviour |
| `allowed_workload_types` | `{inference_standard, interactive_inference, llm_serving, standard_interactive_inference, critical_interactive_inference}` | strict allow-list |
| `min_telemetry_confidence` | `"medium"` | required label on `workload_metadata.telemetry_confidence` |
| `candidate_rhos` | `(0.45, 0.55, 0.65, 0.75, 0.85, 0.95)` | rho grid the controller may consider |
| `max_timeout_pct` | `10.0` | safety veto threshold (timeout) |
| `max_queue_p99_ms` | `2000.0` | safety veto threshold (queue p99) |
| `max_latency_p99_ms` | `None` | optional |
| `conservative_margin_enabled` | `False` | step back from boundary when adjacent point is unsafe |
| `min_telemetry_window_ticks` | `8` | adapter falls back to default rho below this |
| `fallback_to_existing_on_error` | `True` | swallow estimator / controller exceptions and fall back |
| `shadow_only` | `True` | recommendation-only |
| `allow_simulator_execution` | `False` | simulator-mutable on adapter result |
| `allow_real_execution` | `False` | construction with `True` + `shadow_only=True` is a `ValueError` |

## 4. Eligibility — `is_frontier_eligible(...)`

Eligible only when **all** of:

- `config.enabled` is `True`;
- `workload_metadata.workload_id` and `.workload_type` are present;
- `workload_type ∈ config.allowed_workload_types`;
- `workload_type` is not training / fine_tuning / offline_batch / philly_training_job / alibaba_gpu_packing_job / batch_inference, and `workload_metadata.is_training` is not truthy;
- `workload_metadata.telemetry_confidence ≥ config.min_telemetry_confidence`;
- `service_state.telemetry_window_ticks ≥ config.min_telemetry_window_ticks`;
- at least one of `service_state.request_metrics_present` or `.queue_metrics_present` is true;
- at least one SLA / timeout budget is declared (`latency_sla_ms`, `timeout_sla_pct`, or `queue_p99_sla_ms`);
- `service_state.degraded_telemetry` and `.failsafe_active` are not set.

Failing eligibility is **never silent** — `EligibilityResult.reason` is an explicit string and `missing_fields` enumerates the missing keys.

## 5. Adapter — `select_constraint_aware_rho(...)`

Returns a `FrontierAdapterResult` carrying either the frontier-selected rho or the engine default `0.65`, plus the full decision metadata (action, reason, fallback reason, expected deltas, confidence, safety vetoes). The fallback branches:

| branch | result |
|---|---|
| not eligible | default rho; `fallback_reason=ineligible: <reason>` |
| empty telemetry window | default rho |
| `estimate_frontier` raises | default rho; `fallback_reason=estimator_error:<exc>` |
| `choose_safe_utilization_target` raises | default rho; `fallback_reason=controller_error:<exc>` |
| controller returns `INSUFFICIENT_TELEMETRY` | default rho |
| controller returns `LOWER_RHO` | default rho (the engine's hysteresis + SLA trim handle bounded relief; the adapter never promotes a lower rho silently — opt-in is conservative-by-default) |
| selected point is not `SAFE` | default rho |
| `0.0 < rho ≤ 1.0` fails | default rho |
| otherwise | `selected_rho = decision.selected_rho`, `used_frontier=True` |

## 6. Engine integration — `aurelius/traces/backtest.py`

`_run_policy(...)` accepts four new optional kwargs (all default `None`):

- `frontier_integration: FrontierIntegrationConfig | None`
- `frontier_workload_metadata: dict | None`
- `frontier_service_state: dict | None`
- `frontier_counters: FrontierIntegrationCounters | None`

Only the `constraint_aware` branch consults them. When `frontier_integration is None` **or** `enabled is False`, the policy uses `target_rho=0.65` exactly as before. When enabled + eligible + safe, it uses `target_rho=result.selected_rho`. The downstream `_constraint_trim`, cache-affinity prefill, SLA / queue gates, and `evaluate_tick` are unchanged.

`run_backtest(...)` forwards the same kwargs (all default `None`). All other callers of `run_backtest` (`scripts/run_burstgpt_backtest.py`, `scripts/run_azure_llm_backtest.py`, `scripts/run_azure_llm_2024_backtest.py`, every existing test) are unaffected.

## 7. Reporting fields exposed

`FrontierAdapterResult.to_dict()` carries every required reporting field:

- `frontier_enabled` / `frontier_eligible` (under `eligibility`)
- `selected_rho` / `previous_rho` (via the decision)
- `frontier_action` / `frontier_reason` (via the decision)
- `frontier_fallback_reason`
- `frontier_expected_goodput_per_dollar_delta`
- `frontier_expected_gpu_hour_delta`
- `frontier_expected_sla_risk_delta`
- `frontier_safety_vetoes`
- `frontier_confidence`

`FrontierIntegrationCounters` aggregates per-workload:

- `frontier_used_count`, `frontier_fallback_count`
- `frontier_ineligible_count`, `frontier_low_confidence_count`
- `frontier_unsafe_recommendation_count`, `frontier_lower_rho_count`
- `frontier_error_count`

Fallback is never hidden — when enabled but unused, the report carries an explicit reason.

## 8. Benchmark evidence (simulator / shadow only)

| benchmark | constraint_aware (current) | constraint_aware + frontier opt-in | Δ % | selected rho |
|---|---|---|---|---|
| **Azure LLM 2024** (committed week-long audit) | 2,555,324.54 goodput/$ | **2,886,960.51 goodput/$** | **+12.978 %** | 0.75 |
| BurstGPT (fixture, 25× scale) | 212,102.93 | 215,215.70 | +1.468 % | 0.75 |
| Azure LLM 2023 (fixture, 25× scale) | 1,740,426.47 | 1,740,426.47 | +0.000 % (fallback: telemetry window below minimum) | 0.65 |

**Cross-trace integration safety check:** 0 regressions across the applicable LLM serving traces (2 wins + 1 safe tie). See `docs/CROSS_TRACE_CONSTRAINT_FRONTIER_INTEGRATION_SAFETY.md` and `data/external/frontier/cross_trace_constraint_frontier_integration_safety_summary.json`.

The Azure 2024 number reproduces the committed `frontier_controller_v1` result (`data/external/azure_llm_2024/processed/azure_2024_frontier_controller_summary.json`) within tolerance — same data, same controller, same safety gates.

## 9. Honesty / non-goals

- The `constraint_aware` engine default rho (≈ 0.65) is **unchanged**. No global default change is supported by the evidence — the safe rho **varies by workload, SLA, and telemetry**.
- **rho = 0.75 is not a universal constant.** It is the safe peak on Azure LLM 2024 under the audit's pre-registered safety gates. Other workloads will require different rhos.
- The frontier controller does **not replace** `constraint_aware`. It only selects the utilization target; every existing SLA / queue / latency / energy / cache / topology / residency gate continues to run.
- The robust energy engine is **not modified**.
- This is **simulator / shadow-mode evidence**. Public-trace results are directional — **NOT production savings** (`docs/RESULTS.md` §8). Pilot telemetry is required to calibrate the safe rho per workload before any production claim.
- Real-cluster execution is **disabled by default**. Constructing `FrontierIntegrationConfig(enabled=True, shadow_only=True, allow_real_execution=True)` raises `ValueError`; real execution requires an explicit `shadow_only=False` opt-in **and** a real executor passed in by the caller (the controller itself ships only a no-op stub).
- No ML model training, no new datasets, no oracle baseline used as the headline.

## 10. Remaining gaps before any production / default-on promotion

- **Pilot telemetry calibration.** Each customer / workload needs its own frontier audit to establish the safe rho band — public-trace ranges are directional only.
- **GenAI 2026 path.** GenAI's `constraint_aware` sizer (`_size_for_sla`) probes-up-to-SLA instead of using a fixed rho target; integrating the adapter there would require a different shim than the rho-target path used here.
- **Customer-specific safety thresholds.** `max_timeout_pct=10`, `max_queue_p99_ms=2000` are pre-registered defaults from the Azure 2024 audit. Per-tenant SLAs must override these explicitly.
- **Production executor.** Real-mode execution remains a deliberate stub. Promoting it requires the binding-boundary work in `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` §"Real-mode execution boundary".
