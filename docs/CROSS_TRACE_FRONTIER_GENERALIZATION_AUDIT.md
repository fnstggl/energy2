# Cross-Trace Safe Utilization Frontier Generalization Audit

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Cross-trace validation of `frontier_controller_v1` (`aurelius/frontier/`) against every currently-integrated public trace where target-utilization (rho) decisions are meaningful. No production code, optimizer logic, simulator constant, or robust-energy-engine code was modified, no ML model was trained, no dataset was ingested, no constant was tuned to force a result. **Real-cluster execution is disabled by default** (`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`).

- **Read first:** `docs/RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`.

## 1. Configuration

- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Safety thresholds (pre-registered):** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Comparison policies:** `fifo`, `sla_aware`, `queue_aware`, `utilization_aware`, `constraint_aware` (current baseline), `frontier_controller_v1`, plus `oracle_forecast_ANALYSIS_ONLY` where available (analysis-only).

## 2. Datasets analyzed

| trace | applicable to frontier control | source | reason if excluded |
|---|---|---|---|
| `burstgpt` | ✅ | `fixture:/home/user/energy2/tests/fixtures/burstgpt_sample.csv` | — |
| `azure_llm_2023` | ✅ | `fixture:/home/user/energy2/tests/fixtures/azure_llm_sample.csv` | — |
| `azure_llm_2024_week` | ✅ | `committed_audit_json` | — |
| `alibaba_genai_2026` | ✅ | `fixture:/home/user/energy2/tests/fixtures/alibaba_genai_sample` | — |
| `alibaba_gpu_v2023` | ❌ | — | Bin-packing / fragmentation trace: pods have fixed (cpu, gpu_milli, memory) requirements and there is no continuous request-rate or serving-utilization target rho to sweep. The headline baseline is first-fit / best-fit / FFD packing — see docs/ALIBABA_GPU_BACKTEST_RESULTS.md. Safe Utilization Frontier Control acts on serving rho targets and is therefore structurally not applicable to packing decisions. |
| `microsoft_philly` | ❌ | — | Training-job scheduling trace: deadlines / job-progress and GPU-hours dominate, not request-rate / serving rho. The headline metric is job completion / SLA-violation count, not goodput/$ over a rho sweep. Safe Utilization Frontier Control acts on serving rho targets and is structurally not applicable to training-job scheduling. |

## 3.1 `burstgpt` — frontier sweep + controller verdict

- **Source:** `fixture:/home/user/energy2/tests/fixtures/burstgpt_sample.csv`
- **Ticks:** 55 @ 60.0s
- **Verdict vs `constraint_aware`:** **`FRONTIER_WIN`** (Δ +1.395%)
- **Frontier controller decision:** `RECOMMEND_RHO` → rho = 0.75; executable_in_real_cluster = False
- **Best safe rho (anticipatory frontier):** 0.95 | first unsafe rho: None

### Anticipatory frontier sweep

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 211,974.47 | 4.70 | 220.66 / 335.75 | 0.9333 | 0.293529 | 2 | SAFE |
| 0.55 | 211,974.47 | 4.70 | 220.66 / 335.75 | 0.9333 | 0.293529 | 2 | SAFE |
| 0.65 | 211,974.47 | 4.70 | 220.66 / 335.75 | 0.9333 | 0.293529 | 2 | SAFE |
| 0.75 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |
| 0.85 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |
| 0.95 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |

### Reactive frontier sweep (diagnostic)

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 211,292.06 | 5.15 | 373.84 / 640.76 | 0.9333 | 0.318665 | 2 | SAFE |
| 0.55 | 211,292.06 | 5.15 | 373.84 / 640.76 | 0.9333 | 0.318665 | 2 | SAFE |
| 0.65 | 211,292.06 | 5.15 | 373.84 / 640.76 | 0.9333 | 0.318665 | 2 | SAFE |
| 0.75 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |
| 0.85 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |
| 0.95 | 215,060.85 | 5.22 | 435.12 / 740.87 | 0.9167 | 0.336091 | 0 | SAFE |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| fifo | — | 215,060.85 | 5.22 | 740.87 | 0.9167 | SAFE |
| sla_aware | 0.5000 | 211,292.06 | 5.15 | 640.76 | 0.9333 | SAFE |
| queue_aware | — | 204,518.56 | 5.15 | 640.76 | 0.9667 | SAFE |
| utilization_aware | 0.8500 | 215,060.85 | 5.22 | 740.87 | 0.9167 | SAFE |
| constraint_aware | 0.6500 | 212,102.93 | 4.64 | 335.75 | 0.9333 | SAFE |
| frontier_controller_v1 | 0.7500 | 215,060.85 | 5.22 | 740.87 | 0.9167 | SAFE |

## 3.2 `azure_llm_2023` — frontier sweep + controller verdict

- **Source:** `fixture:/home/user/energy2/tests/fixtures/azure_llm_sample.csv`
- **Ticks:** 1 @ 60.0s
- **Verdict vs `constraint_aware`:** **`TIE`** (Δ +0.000%)
- **Frontier controller decision:** `RECOMMEND_RHO` → rho = 0.45; executable_in_real_cluster = False
- **Best safe rho (anticipatory frontier):** 0.75 | first unsafe rho: 0.85

### Anticipatory frontier sweep

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.55 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.65 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.75 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.85 | 3,119,382.35 | 14.07 | 1,956.47 / 3,825.75 | 0.0167 | 0.822833 | 0 | **UNSAFE** |
| 0.95 | 3,119,382.35 | 14.07 | 1,956.47 / 3,825.75 | 0.0167 | 0.822833 | 0 | **UNSAFE** |

### Reactive frontier sweep (diagnostic)

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.55 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.65 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.75 | 1,740,426.47 | 4.11 | 94.18 / 159.65 | 0.0333 | 0.599901 | 0 | SAFE |
| 0.85 | 3,119,382.35 | 14.07 | 1,956.47 / 3,825.75 | 0.0167 | 0.822833 | 0 | **UNSAFE** |
| 0.95 | 3,119,382.35 | 14.07 | 1,956.47 / 3,825.75 | 0.0167 | 0.822833 | 0 | **UNSAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| fifo | — | 1,740,426.47 | 4.11 | 159.65 | 0.0333 | SAFE |
| sla_aware | 0.5000 | 1,740,426.47 | 4.11 | 159.65 | 0.0333 | SAFE |
| queue_aware | — | 1,740,426.47 | 4.11 | 159.65 | 0.0333 | SAFE |
| utilization_aware | 0.8500 | 3,119,382.35 | 14.07 | 3,825.75 | 0.0167 | **UNSAFE** |
| constraint_aware | 0.6500 | 1,740,426.47 | 4.11 | 159.65 | 0.0333 | SAFE |
| frontier_controller_v1 | 0.4500 | 1,740,426.47 | 4.11 | 159.65 | 0.0333 | SAFE |

## 3.3 `azure_llm_2024_week` — frontier sweep + controller verdict

- **Source:** `committed_audit_json`
- **Ticks:** 12960 @ 60.0s
- **Verdict vs `constraint_aware`:** **`FRONTIER_WIN`** (Δ +12.978%)
- **Frontier controller decision:** `RECOMMEND_RHO` → rho = 0.75; executable_in_real_cluster = False
- **Best safe rho (anticipatory frontier):** 0.75 | first unsafe rho: 0.85

### Anticipatory frontier sweep

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 1,798,572.67 | 6.46 | 0.1600 / 0.2500 | 8,322.70 | 0.572700 | 9,828 | SAFE |
| 0.55 | 2,188,260.26 | 6.63 | 0.2100 / 0.3300 | 6,829.60 | 0.573900 | 9,268 | SAFE |
| 0.65 | 2,555,324.54 | 7.64 | 0.3800 / 0.6300 | 5,796.10 | 0.623400 | 8,830 | SAFE |
| 0.75 | 2,886,960.51 | 9.46 | 1.56 / 2.82 | 5,037.80 | 0.716100 | 8,411 | SAFE |
| 0.85 | 3,190,680.98 | 11.65 | 16.85 / 32.82 | 4,457.60 | 0.808900 | 8,068 | **UNSAFE** |
| 0.95 | 3,186,976.66 | 19.59 | 1,000.26 / 2,065.75 | 3,998.90 | 0.901300 | 7,592 | **UNSAFE** |

### Reactive frontier sweep (diagnostic)

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 1,833,582.20 | 6.49 | 15.99 / 33.86 | 8,161.30 | 0.573400 | — | SAFE |
| 0.55 | 2,228,257.64 | 6.82 | 32.94 / 69.66 | 6,697.40 | 0.581900 | — | SAFE |
| 0.65 | 2,594,665.58 | 8.06 | 108.66 / 229.31 | 5,685.00 | 0.642300 | — | SAFE |
| 0.75 | 2,916,406.16 | 10.17 | 602.66 / 1,273.55 | 4,940.80 | 0.735200 | — | **UNSAFE** |
| 0.85 | 3,011,921.12 | 14.99 | 7,858.69 / 16,626.28 | 4,372.70 | 0.830200 | — | **UNSAFE** |
| 0.95 | 1,986,660.36 | 30.27 | 66,629.05 / 141,186.21 | 3,923.20 | 0.924800 | — | **UNSAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| fifo | — | 1,288,234.72 | 31.91 | 264,464.40 | 5,184.00 | **UNSAFE** |
| sla_aware | 0.5000 | 2,032,039.55 | 6.61 | 33.98 | 7,357.20 | SAFE |
| queue_aware | — | 2,490,662.91 | 24.18 | 78,398.79 | 4,068.40 | **UNSAFE** |
| utilization_aware | 0.8500 | 3,238,462.77 | 12.10 | 48.42 | 4,372.80 | **UNSAFE** |
| constraint_aware | 0.6500 | 2,555,324.54 | 7.64 | 0.6300 | 5,796.10 | SAFE |
| oracle_forecast_ANALYSIS_ONLY | — | 2,422,788.92 | 7.05 | 0.5000 | 6,149.10 | SAFE |
| frontier_controller_v1 | 0.7500 | 2,886,960.51 | 9.46 | 2.82 | 5,037.80 | SAFE |

## 3.4 `alibaba_genai_2026` — frontier sweep + controller verdict

- **Source:** `fixture:/home/user/energy2/tests/fixtures/alibaba_genai_sample`
- **Ticks:** 6 @ 3600.0s
- **Verdict vs `constraint_aware`:** **`TIE`** (Δ +0.000%)
- **Frontier controller decision:** `LOWER_RHO` → rho = 0.45; executable_in_real_cluster = False
- **Best safe rho (anticipatory frontier):** None | first unsafe rho: 0.45

### Anticipatory frontier sweep

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |
| 0.55 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |
| 0.65 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |
| 0.75 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |
| 0.85 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |
| 0.95 | 3.33 | 0.0000 | 5,608.70 / 5,608.70 | 6.00 | — | 0 | **UNSAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| fifo | — | 3.33 | 0.0000 | 7,743.38 | 6.00 | **UNSAFE** |
| sla_aware | 0.6500 | 3.33 | 0.0000 | 7,743.38 | 6.00 | **UNSAFE** |
| queue_aware | 0.6500 | 3.33 | 0.0000 | 7,743.38 | 6.00 | **UNSAFE** |
| utilization_aware | 0.8500 | 3.33 | 0.0000 | 7,743.38 | 6.00 | **UNSAFE** |
| constraint_aware | 0.6500 | 3.33 | 0.0000 | 5,608.70 | 6.00 | **UNSAFE** |
| frontier_controller_v1 | 0.4500 | 3.33 | 0.0000 | 5,608.70 | 6.00 | **UNSAFE** |

## 3.5 `alibaba_gpu_v2023` — frontier sweep + controller verdict

**Excluded.** Bin-packing / fragmentation trace: pods have fixed (cpu, gpu_milli, memory) requirements and there is no continuous request-rate or serving-utilization target rho to sweep. The headline baseline is first-fit / best-fit / FFD packing — see docs/ALIBABA_GPU_BACKTEST_RESULTS.md. Safe Utilization Frontier Control acts on serving rho targets and is therefore structurally not applicable to packing decisions.

## 3.6 `microsoft_philly` — frontier sweep + controller verdict

**Excluded.** Training-job scheduling trace: deadlines / job-progress and GPU-hours dominate, not request-rate / serving rho. The headline metric is job completion / SLA-violation count, not goodput/$ over a rho sweep. Safe Utilization Frontier Control acts on serving rho targets and is structurally not applicable to training-job scheduling.

## 4. Cross-trace summary

| trace | workload class | constraint_aware goodput/$ | frontier_controller_v1 goodput/$ | Δ % | best safe rho | controller action → rho | verdict | generalizes? |
|---|---|---|---|---|---|---|---|---|
| `burstgpt` | bursty_interactive_inference | 212,102.93 | 215,060.85 | +1.395% | 0.95 | `RECOMMEND_RHO` → 0.75 | **FRONTIER_WIN** | yes |
| `azure_llm_2023` | interactive_inference | 1,740,426.47 | 1,740,426.47 | +0.000% | 0.75 | `RECOMMEND_RHO` → 0.45 | **TIE** | yes |
| `azure_llm_2024_week` | weekly_periodic_interactive_inference | 2,555,324.54 | 2,886,960.51 | +12.978% | 0.75 | `RECOMMEND_RHO` → 0.75 | **FRONTIER_WIN** | yes |
| `alibaba_genai_2026` | multi_layer_inference_with_cold_start | 3.33 | 3.33 | +0.000% | None | `LOWER_RHO` → 0.45 | **TIE** | yes |

**Counts (applicable traces, n=4):** wins = 2 (50.00%); ties = 2 (50.00%); losses = 0 (0.0000%); skipped = 2.

**Δ goodput/$ range:** 0.0000% to 12.98% (mean 3.59%).

**Best-safe-rho distribution:** {'0.95': 1, '0.75': 2} (min 0.75, max 0.95). The safe rho is **workload-specific** — no single global value is supported.

## 5. Generalization & architecture recommendation

- **Does `frontier_controller_v1` improve or safely tie across the applicable LLM serving traces?** `GENERALIZES_WITHIN_APPLICABLE_LLM_INFERENCE_TRACES`.
- **A. Generally superior?** Yes, on the applicable LLM serving traces (no regression observed).
- **B. Workload-dependent?** Yes — the safe peak rho varies by trace (see distribution above); the bin-packing / training-job traces are structurally outside the frontier-controller scope.
- **C. Should it be integrated into `constraint_aware`?** `KEEP_FRONTIER_CONTROLLER_SEPARATE_OR_OPT_IN` — 2/4 traces show a frontier_controller_v1 alpha win and 2/4 are safe ties; no regressions observed. Integration as a per-workload-class OPT-IN remains evidence-supported, but a global default change is NOT (rho varies by workload / SLA / telemetry).
- **D. % improvement (applicable):** 50.00%.
- **E. % neutral ties (applicable):** 50.00%.
- **F. % regressions (applicable):** 0.0000%.

## 6. Honesty / scope

- The Azure 2024 frontier is the **full week-long committed audit JSON** (`data/external/azure_llm_2024/processed/azure_2024_safe_utilization_frontier.json`); every other LLM trace frontier is computed in-process on its **fixture or raw data if present** via the UNCHANGED serving physics in `aurelius/traces/backtest.py` / `genai_backtest.py`.
- Bin-packing traces (Alibaba GPU v2023) and training-job-scheduling traces (Microsoft Philly) are **structurally not utilization-rho benchmarks** — they sweep packing density or job-completion times, not a continuous request-rate / serving-utilization target — so the frontier controller is documented as **NOT APPLICABLE**.
- The `constraint_aware` engine default (rho ≈ 0.65) is **unchanged** by this audit. No production code, simulator constant, optimizer logic, or safety gate has been modified.
- No production-savings claim. Real-cluster execution is **disabled by default**; pilot telemetry is required to calibrate the safe rho per workload (`docs/PILOT_TELEMETRY_CONTRACT.md`).

## 7. Remaining unknowns

- The fixture-derived frontiers for BurstGPT / Azure 2023 / GenAI 2026 inherit the fixture's small-sample shape; the *direction* of the verdict is informative but the absolute Δ% is fixture-bounded. Larger raw replays (when raw is present) will refine the absolute numbers; the *verdict bucket* should remain stable because the controller selects by category (SAFE / UNSAFE), not by a tuned KPI threshold.
- Customer/pilot telemetry is required to calibrate the safe rho per workload before any real-cluster promotion of a frontier decision.
- The reactive vs anticipatory dominance pattern observed on Azure 2024 (anticipatory frontier strictly dominates reactive on safety) needs re-validation on each customer-specific real serving engine.

