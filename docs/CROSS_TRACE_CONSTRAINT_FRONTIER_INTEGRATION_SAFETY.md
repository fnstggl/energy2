# Cross-Trace `constraint_aware` × Frontier-Integration Safety Check

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Compares the unchanged `constraint_aware` policy against itself with the opt-in frontier integration enabled. Real-cluster execution is **disabled by default**. The robust energy engine is **unchanged**.

- **Read first:** `docs/RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`.

## 1. Configuration

- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Safety thresholds:** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Regression tolerance:** ±1.0 % goodput/$ / + absolute timeout %
- **Min telemetry confidence:** `medium`
- **Real-cluster execution:** disabled by default.

## 2. Per-trace integration safety

| trace | applicable | current goodput/$ | opt_in goodput/$ | Δ % | Δ timeout % | selected rho | frontier_used | action | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `burstgpt` | ✅ | 212,102.93 | 215,215.70 | +1.468% | +0.513 | 0.75 | True | `RECOMMEND_RHO` | **INTEGRATION_WIN** |
| `azure_llm_2023` | ✅ | 1,740,426.47 | 1,740,426.47 | +0.000% | +0.000 | 0.65 | False | `—` | **SAFE_TIE** |
| `azure_llm_2024_week` | ✅ | 2,555,324.54 | 2,886,960.51 | +12.978% | +1.826 | 0.75 | True | `RECOMMEND_RHO` | **INTEGRATION_WIN** |
| `alibaba_genai_2026` | ❌ | — | — | — | — | — | — | — | _excluded — GenAI 2026's constraint_aware uses _size_for_sla (probe-up-t..._ |
| `alibaba_gpu_v2023` | ❌ | — | — | — | — | — | — | — | _excluded — Bin-packing / fragmentation trace. No continuous serving rho..._ |
| `microsoft_philly` | ❌ | — | — | — | — | — | — | — | _excluded — Training-job scheduling trace. Deadlines / job-progress domi..._ |

## 3. Excluded traces

- **`alibaba_genai_2026`** — GenAI 2026's constraint_aware uses _size_for_sla (probe-up-to-SLA), not a fixed rho target — the v1 integration adapter applies only to the rho-target sizer in aurelius/traces/backtest.py.
- **`alibaba_gpu_v2023`** — Bin-packing / fragmentation trace. No continuous serving rho target; frontier integration is structurally not applicable.
- **`microsoft_philly`** — Training-job scheduling trace. Deadlines / job-progress dominate, not serving rho; frontier integration is structurally not applicable.

## 4. Synthesis

- Applicable traces: **3**; excluded: **3**
- Safe ties: **1** | integration wins: **2** | regressions: **0**
- Safe-or-win %: **100.0** | regression %: **0.0**
- **Any regression?** False

## 5. Fallback explanations

- **`burstgpt`** — frontier used. Action: `RECOMMEND_RHO`. Selected rho: 0.75.
- **`azure_llm_2023`** — fell back to `constraint_aware` default rho. Reason: `ineligible: telemetry_window_ticks=1 below required 8`
- **`azure_llm_2024_week`** — frontier used. Action: `RECOMMEND_RHO`. Selected rho: 0.75.

## 6. Honesty / scope

- The `constraint_aware` engine default rho is **unchanged**. The frontier integration is **opt-in**, **LLM-serving-only**, **disabled by default**, and **falls back to the existing engine** on any ineligibility / unsafe recommendation / estimator or controller error.
- Alibaba GPU v2023 (bin-packing / fragmentation) and Microsoft Philly (training-job scheduling) are structurally outside the frontier-integration scope and are documented as **NOT APPLICABLE**.
- Real-cluster execution is **disabled by default**; pilot telemetry is required to calibrate the safe rho per workload before any production claim.

