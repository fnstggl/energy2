# Azure LLM 2024 — Dynamic Safe Frontier Estimator Results

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Streaming replay of the Azure 2024 trace with the Dynamic Safe Frontier Estimator (`aurelius/frontier/dynamic_estimator.py`); each per-tick decision sees only the telemetry from t' ≤ t (no future leakage). The robust energy engine is **unchanged**; the static frontier controller and committed Azure 2024 artifacts are **read-only**. Real-cluster execution is disabled by default.

- **Read first:** `docs/RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/PILOT_TELEMETRY_CONTRACT.md`.

## 1. Configuration

- **Tick seconds:** 60.0
- **Primary scale:** 100.0×
- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Safety thresholds (pre-registered):** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Rolling windows:** [30, 60, 180] min
- **No future leakage** — each decision sees t' ≤ t only.

## 2. Source

- **Trace:** `/home/user/energy2/tests/fixtures/azure_llm_2024_sample.csv` (1,560 ticks @ 60s; 93,540 s total)

## 3. Streaming-replay results

| label | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | mean rho | scale ev | safe |
|---|---|---|---|---|---|---|---|---|
| `sla_aware` | 0.5000 | 1,036,721.02 | 3.58 | 281.59 | 30.90 | 0.515313 | 406 | ✅ |
| `constraint_aware_static` | 0.6500 | 1,139,319.79 | 3.73 | 314.92 | 28.07 | 0.523308 | 202 | ✅ |
| `static_frontier_controller` | 0.7500 | 1,181,075.23 | 3.85 | 349.06 | 27.03 | 0.529979 | 112 | ✅ |
| `utilization_aware` | 0.8500 | 1,198,981.49 | 4.09 | 470.54 | 26.53 | 0.536787 | 60 | ✅ |
| `dynamic_frontier_estimator_w30m` | dynamic(0.531 mean) | 1,182,973.13 | 3.87 | 356.64 | 26.98 | 0.530646 | 106 | ✅ |
| `dynamic_frontier_estimator_w60m` | dynamic(0.531 mean) | 1,182,973.13 | 3.87 | 356.64 | 26.98 | 0.530646 | 106 | ✅ |
| `dynamic_frontier_estimator_w180m` | dynamic(0.531 mean) | 1,182,973.13 | 3.87 | 356.64 | 26.98 | 0.530646 | 106 | ✅ |
| `oracle_realized_optimal_ANALYSIS_ONLY` | 0.8500 | 1,198,981.49 | 4.09 | 470.54 | 26.53 | 0.536787 | 60 | ✅ |

## 4. Cross-window deltas

| window | dynamic goodput/$ | Δ vs CA static | Δ vs FC static | Δ vs oracle | optimality gap | alpha retained vs oracle | convergence tick | safe |
|---|---|---|---|---|---|---|---|---|
| 30 min | 1,182,973.13 | +3.832% | +0.161% | -1.335% | +1.335% | +73.168% | 10 | ✅ |
| 60 min | 1,182,973.13 | +3.832% | +0.161% | -1.335% | +1.335% | +73.168% | 10 | ✅ |
| 180 min | 1,182,973.13 | +3.832% | +0.161% | -1.335% | +1.335% | +73.168% | 10 | ✅ |

**Verdict:** **`DYNAMIC_BEATS_STATIC_CA`**

**Frontier recovery:** PARTIAL — dynamic estimator recovered 73.2% of the alpha between constraint_aware and the oracle.

## 5. Action distribution by window

| window | RAISE | KEEP | LOWER | INSUFFICIENT |
|---|---|---|---|---|
| 30 min | 284 | 1064 | 204 | 0 |
| 60 min | 284 | 1064 | 204 | 0 |
| 180 min | 284 | 1064 | 204 | 0 |

## 6. Honesty / scope

- The Dynamic Safe Frontier Estimator is **opt-in** and **disabled by default**. The static frontier controller remains the committed default; this benchmark is a measurement.
- **No production mutation.** Decisions are recommendation-only (`executable_in_real_cluster=False`).
- **No future leakage.** Each per-tick decision sees only the telemetry from t' ≤ t in its rolling window.
- **No ML training in v1.** Risk scores are deterministic / statistical heuristics (EWMA, slopes, CV, Erlang-C tails). See `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`.
- The oracle row is **analysis-only**. It sees the entire trace ahead of time and is not a real-time baseline — it is a ceiling for the recovery question.
- This is **directional simulator / shadow-mode evidence** — NOT production savings. Pilot telemetry is required to calibrate the safe rho per workload before any production claim.
- The robust energy engine, the static frontier controller, the committed Azure 2024 audit / backtest / controller / integration / full-trace JSON are **NOT modified** by this benchmark.

