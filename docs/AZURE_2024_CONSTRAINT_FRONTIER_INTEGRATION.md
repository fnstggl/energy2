# Azure LLM 2024 — `constraint_aware` × Frontier-Integration Benchmark

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Reuses the COMMITTED Azure 2024 safe-utilization audit JSON (`data/external/azure_llm_2024/processed/azure_2024_safe_utilization_frontier.json`) — no re-simulation, no tuned constants. The committed Azure 2024 baseline JSON is read-only. Real-cluster execution is **disabled by default**.

- **Read first:** `docs/RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`.

## 1. Configuration

- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Safety thresholds (pre-registered):** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Min telemetry confidence:** `medium`
- **Conservative margin:** False
- **Real-cluster execution:** disabled by default (`shadow_only=True`, `allow_real_execution=False`).

## 2. Result

| label | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safe | frontier_used | action |
|---|---|---|---|---|---|---|---|---|
| `constraint_aware_current` | 0.65 | 2,555,324.54 | 7.64 | 0.6300 | 5,796.10 | SAFE | False | `—` |
| `constraint_aware_frontier_opt_in` | 0.75 | 2,886,960.51 | 9.46 | 2.82 | 5,037.80 | SAFE | True | `RECOMMEND_RHO` |

- **Δ goodput/$ (frontier_opt_in vs current):** **+12.978%** (2,555,324.54 → 2,886,960.51)
- **Δ GPU-hours:** -758.30 (-13.083%)
- **Δ timeout %:** 1.83 (absolute)
- **Δ queue p99 (ms):** 2.19 (absolute)
- **Selected rho:** 0.75 (previous: 0.65)
- **Frontier action:** `RECOMMEND_RHO`
- **Frontier reason:** highest SLA-safe goodput/$ at rho 0.75 (predicted 2,886,960.51 across all SLA / queue / latency / telemetry gates)

## 3. Baseline preservation

- `constraint_aware_current` goodput/$: **2,555,324.54** (committed Azure 2024 baseline preserved within ±1.0 %).
- `frontier_controller_v1` committed result (`azure_2024_frontier_controller_summary.json`): **2,886,960.51**.
- `constraint_aware_frontier_opt_in` reproduces the committed controller result (Δ +0.000%).

## 4. Counters

| counter | value |
|---|---|
| `frontier_used_count` | 1 |
| `frontier_fallback_count` | 0 |
| `frontier_ineligible_count` | 0 |
| `frontier_low_confidence_count` | 0 |
| `frontier_unsafe_recommendation_count` | 0 |
| `frontier_lower_rho_count` | 0 |
| `frontier_error_count` | 0 |

## 5. Honesty / scope

- The `constraint_aware` engine default rho **was not changed**. The integration is **opt-in**, **LLM-serving-only**, **disabled by default**, and **falls back to the existing engine** on any failure / ineligibility / unsafe recommendation.
- The committed Azure 2024 audit JSON is **read-only** in this benchmark. The committed `constraint_aware` baseline goodput/$ is preserved within 1 % tolerance (asserted by tests).
- This is **directional simulator/backtest evidence** — NOT production savings. Pilot telemetry is required to calibrate the safe rho per workload before any production claim.

