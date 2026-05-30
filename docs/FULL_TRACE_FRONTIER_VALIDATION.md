# Full-Trace Safe Utilization Frontier Validation Audit

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Validates the previously fixture-bound SAFE_TIE verdicts on Azure LLM 2023 and BurstGPT by running the same audit on the **raw, full traces**. Reuses the UNCHANGED serving physics in `aurelius/traces/backtest.py` and the UNCHANGED frontier controller / integration in `aurelius/frontier/` + `aurelius/constraints/frontier_integration.py`. No optimizer / robust-energy-engine constant is changed; no constant is tuned to force a result; no ML model is trained; no safety gate is weakened. The committed Azure 2024 artifacts are **read-only**.

- **Read first:** `docs/RESULTS.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`, `docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`, `docs/CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md`, `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`.

## 1. Configuration

- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Safety thresholds (pre-registered):** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Tie band:** ±1.0% goodput/$
- **constraint_aware default rho:** 0.65 (unchanged)

## 2. Datasets

| trace | raw file | rows | time span | source |
|---|---|---|---|---|
| `burstgpt` | `BurstGPT_1.csv` | 1,404,294 | 5,269,968 s (~60.99 d) | `/home/user/energy2/data/external/burstgpt/raw/BurstGPT_1.csv` |
| `azure_llm_2023_conv` | `AzureLLMInferenceTrace_conv.csv` | 19,366 | 3,502 s (~0.04 d) | `/home/user/energy2/data/external/azure_llm/raw/AzureLLMInferenceTrace_conv.csv` |
| `azure_llm_2023_code` | `AzureLLMInferenceTrace_code.csv` | 8,819 | 3,436 s (~0.04 d) | `/home/user/energy2/data/external/azure_llm/raw/AzureLLMInferenceTrace_code.csv` |
| `azure_llm_2024_week` | `azure_2024_safe_utilization_frontier.json` | — | 604,800 s (~7.00 d) | `data/external/azure_llm_2024/processed/azure_2024_safe_utilization_frontier.json` |

## 3.burstgpt — frontier sweep + controller verdict

- **Verdict vs `constraint_aware`:** **`SAFE_TIE`** (Δ goodput/$ -0.093 %)
- **n_ticks:** 87833; **n_requests:** 1,404,294; **time span:** 5,269,968 s
- **Best safe rho (anticipatory):** 0.95; first unsafe: None
- **Frontier-optimal rho:** 0.75 (50,629.76 goodput/$)
- **Lowest-cost safe rho:** 0.75 (1,463.88 GPU-h)
- **`constraint_aware` rho in safe set:** True

### Anticipatory frontier sweep (full trace)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 50,624.53 | 1.66 | 53.10 | 46.24 / 65.24 | 1,464.07 | 0.093745 | 20 | 20.00 | **SAFE** |
| 0.55 | 50,628.84 | 1.66 | 53.10 | 46.38 / 65.45 | 1,463.92 | 0.093764 | 4 | 4.00 | **SAFE** |
| 0.65 | 50,629.34 | 1.66 | 53.10 | 46.38 / 65.46 | 1,463.90 | 0.093782 | 2 | 2.00 | **SAFE** |
| 0.75 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.85 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.95 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |

### Reactive frontier sweep (full trace, diagnostic)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 50,627.26 | 1.66 | 53.10 | 46.36 / 65.43 | 1,463.97 | 0.093777 | 10 | 10.00 | **SAFE** |
| 0.55 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.65 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.75 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.85 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |
| 0.95 | 50,629.76 | 1.66 | 53.10 | 46.42 / 65.52 | 1,463.88 | 0.093786 | 0 | 0.0000 | **SAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| `fifo` | — | 50,629.76 | 1.66 | 65.52 | 1,463.88 | SAFE |
| `sla_aware` | 0.5000 | 50,629.31 | 1.66 | 65.47 | 1,463.90 | SAFE |
| `queue_aware` | — | 49,980.74 | 1.66 | 46.76 | 1,481.75 | SAFE |
| `constraint_aware` | 0.6500 | 50,677.09 | 1.60 | 65.46 | 1,463.90 | SAFE |
| `utilization_aware` | 0.8500 | 50,629.76 | 1.66 | 65.52 | 1,463.88 | SAFE |
| `frontier_controller_v1` | 0.7500 | 50,629.76 | 1.66 | 65.52 | 1,463.88 | SAFE |

### Root cause of SAFE_TIE

- **Code:** `B_constraint_aware_already_on_frontier`
- **Evidence:** best safe KPI 50,629.76 vs constraint_aware KPI 50,677.09 → Δ -0.093% (≤ tie band 1.0%)

## 3.azure_llm_2023_conv — frontier sweep + controller verdict

- **Verdict vs `constraint_aware`:** **`SAFE_TIE`** (Δ goodput/$ +0.000 %)
- **n_ticks:** 59; **n_requests:** 19,366; **time span:** 3,502 s
- **Best safe rho (anticipatory):** 0.95; first unsafe: None
- **Frontier-optimal rho:** 0.65 (1,904,272.18 goodput/$)
- **Lowest-cost safe rho:** 0.65 (0.9833 GPU-h)
- **`constraint_aware` rho in safe set:** True

### Anticipatory frontier sweep (full trace)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 1,064,059.92 | 5.31 | 100.00 | 183.37 / 298.06 | 1.78 | 0.505736 | 14 | 14.00 | **SAFE** |
| 0.55 | 1,655,857.70 | 6.37 | 100.00 | 421.45 / 720.51 | 1.13 | 0.602604 | 14 | 14.00 | **SAFE** |
| 0.65 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.75 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.85 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.95 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |

### Reactive frontier sweep (full trace, diagnostic)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 1,158,934.57 | 5.54 | 100.00 | 234.66 / 389.61 | 1.63 | 0.525018 | 22 | 22.00 | **SAFE** |
| 0.55 | 1,842,761.33 | 6.50 | 100.00 | 456.26 / 782.69 | 1.02 | 0.612705 | 2 | 2.00 | **SAFE** |
| 0.65 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.75 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.85 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |
| 0.95 | 1,904,272.18 | 6.54 | 100.00 | 466.86 / 801.67 | 0.9833 | 0.615332 | 0 | 0.0000 | **SAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| `fifo` | — | 1,904,272.18 | 6.54 | 801.67 | 0.9833 | SAFE |
| `sla_aware` | 0.5000 | 1,504,543.92 | 6.17 | 638.78 | 1.25 | SAFE |
| `queue_aware` | — | 1,330,562.63 | 6.04 | 534.00 | 1.42 | SAFE |
| `constraint_aware` | 0.6500 | 1,904,272.18 | 6.54 | 801.67 | 0.9833 | SAFE |
| `utilization_aware` | 0.8500 | 1,904,272.18 | 6.54 | 801.67 | 0.9833 | SAFE |
| `frontier_controller_v1` | 0.6500 | 1,904,272.18 | 6.54 | 801.67 | 0.9833 | SAFE |

### Root cause of SAFE_TIE

- **Code:** `B_constraint_aware_already_on_frontier`
- **Evidence:** best safe KPI 1,904,272.18 vs constraint_aware KPI 1,904,272.18 → Δ +0.000% (≤ tie band 1.0%)

## 3.azure_llm_2023_code — frontier sweep + controller verdict

- **Verdict vs `constraint_aware`:** **`SAFE_TIE`** (Δ goodput/$ +0.000 %)
- **n_ticks:** 58; **n_requests:** 8,819; **time span:** 3,436 s
- **Best safe rho (anticipatory):** 0.95; first unsafe: None
- **Frontier-optimal rho:** 0.45 (124,427.99 goodput/$)
- **Lowest-cost safe rho:** 0.45 (0.9667 GPU-h)
- **`constraint_aware` rho in safe set:** True

### Anticipatory frontier sweep (full trace)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.55 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.65 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.75 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.85 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.95 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |

### Reactive frontier sweep (full trace, diagnostic)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.55 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.65 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.75 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.85 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |
| 0.95 | 124,427.99 | 0.1852 | 45.39 | 5.57 / 6.68 | 0.9667 | 0.115666 | 0 | 0.0000 | **SAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| `fifo` | — | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |
| `sla_aware` | 0.5000 | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |
| `queue_aware` | — | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |
| `constraint_aware` | 0.6500 | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |
| `utilization_aware` | 0.8500 | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |
| `frontier_controller_v1` | 0.4500 | 124,427.99 | 0.1852 | 6.68 | 0.9667 | SAFE |

### Root cause of SAFE_TIE

- **Code:** `B_constraint_aware_already_on_frontier`
- **Evidence:** best safe KPI 124,427.99 vs constraint_aware KPI 124,427.99 → Δ +0.000% (≤ tie band 1.0%)

## 3.azure_llm_2024_week — frontier sweep + controller verdict

- **Verdict vs `constraint_aware`:** **`FRONTIER_WIN`** (Δ goodput/$ +12.978 %)
- **n_ticks:** 12960; **n_requests:** —; **time span:** 604,800 s
- **Best safe rho (anticipatory):** 0.75; first unsafe: 0.85
- **Frontier-optimal rho:** 0.75 (2,886,960.51 goodput/$)
- **Lowest-cost safe rho:** 0.75 (5,037.80 GPU-h)
- **`constraint_aware` rho in safe set:** True

### Anticipatory frontier sweep (full trace)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 1,798,572.67 | 6.46 | 100.00 | 0.1600 / 0.2500 | 8,322.70 | 0.572700 | 9,828 | 22,983 | **SAFE** |
| 0.55 | 2,188,260.26 | 6.63 | 100.00 | 0.2100 / 0.3300 | 6,829.60 | 0.573900 | 9,268 | 18,684 | **SAFE** |
| 0.65 | 2,555,324.54 | 7.64 | 100.00 | 0.3800 / 0.6300 | 5,796.10 | 0.623400 | 8,830 | 15,934 | **SAFE** |
| 0.75 | 2,886,960.51 | 9.46 | 100.00 | 1.56 / 2.82 | 5,037.80 | 0.716100 | 8,411 | 13,744 | **SAFE** |
| 0.85 | 3,190,680.98 | 11.65 | 100.00 | 16.85 / 32.82 | 4,457.60 | 0.808900 | 8,068 | 12,220 | **UNSAFE** |
| 0.95 | 3,186,976.66 | 19.59 | 100.00 | 1,000.26 / 2,065.75 | 3,998.90 | 0.901300 | 7,592 | 10,856 | **UNSAFE** |

### Reactive frontier sweep (full trace, diagnostic)

| rho | goodput/$ | timeout % | sla viol. % | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | safety |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 1,833,582.20 | 6.49 | 100.00 | 15.99 / 33.86 | 8,161.30 | 0.573400 | 10,396 | 31,682 | **SAFE** |
| 0.55 | 2,228,257.64 | 6.82 | 100.00 | 32.94 / 69.66 | 6,697.40 | 0.581900 | 9,941 | 25,851 | **SAFE** |
| 0.65 | 2,594,665.58 | 8.06 | 100.00 | 108.66 / 229.31 | 5,685.00 | 0.642300 | 9,610 | 21,959 | **SAFE** |
| 0.75 | 2,916,406.16 | 10.17 | 100.00 | 602.66 / 1,273.55 | 4,940.80 | 0.735200 | 9,181 | 19,009 | **UNSAFE** |
| 0.85 | 3,011,921.12 | 14.99 | 100.00 | 7,858.69 / 16,626.28 | 4,372.70 | 0.830200 | 8,896 | 16,802 | **UNSAFE** |
| 0.95 | 1,986,660.36 | 30.27 | 100.00 | 66,629.05 / 141,186.21 | 3,923.20 | 0.924800 | 8,571 | 14,993 | **UNSAFE** |

### Policy comparison

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|---|
| `fifo` | — | 1,288,234.72 | 31.91 | 264,464.40 | 5,184.00 | **UNSAFE** |
| `sla_aware` | 0.5000 | 2,032,039.55 | 6.61 | 33.98 | 7,357.20 | SAFE |
| `queue_aware` | — | 2,490,662.91 | 24.18 | 78,398.79 | 4,068.40 | **UNSAFE** |
| `utilization_aware` | 0.8500 | 3,238,462.77 | 12.10 | 48.42 | 4,372.80 | **UNSAFE** |
| `constraint_aware` | 0.6500 | 2,555,324.54 | 7.64 | 0.6300 | 5,796.10 | SAFE |
| `frontier_controller_v1` | 0.7500 | 2,886,960.51 | 9.46 | 2.82 | 5,037.80 | SAFE |

## 4. Cross-trace synthesis

| trace | constraint_aware goodput/$ | frontier_controller goodput/$ | Δ % | verdict | root cause | best safe rho |
|---|---|---|---|---|---|---|
| `burstgpt` | 50,677.09 | 50,629.76 | -0.093 % | **SAFE_TIE** | B_constraint_aware_already_on_frontier | 0.95 |
| `azure_llm_2023_conv` | 1,904,272.18 | 1,904,272.18 | +0.000 % | **SAFE_TIE** | B_constraint_aware_already_on_frontier | 0.95 |
| `azure_llm_2023_code` | 124,427.99 | 124,427.99 | +0.000 % | **SAFE_TIE** | B_constraint_aware_already_on_frontier | 0.95 |
| `azure_llm_2024_week` | 2,555,324.54 | 2,886,960.51 | +12.978 % | **FRONTIER_WIN** | — | 0.75 |

**Counts (applicable, n=4):** wins = 1 | safe-ties = 3 | regressions = 0 | excluded = 0.

## 5. Generalization

- **Was the previous SAFE_TIE caused by fixture limitations, or does the controller add little value on those workloads?** **The previous SAFE_TIE PERSISTS on the full raw trace — the `constraint_aware` operating point already sits at or near the safe-utilization frontier on these workloads (root-causes attached per trace).**
- **Does `frontier_controller_v1` truly generalize across LLM serving traces?** PARTIAL — the controller wins on 1 trace and safely ties on the others; the bulk of its measured value comes from a single workload class.
- **Is Azure 2024 unique?** YES — Azure 2024 is the only applicable trace where the controller produces a measurable goodput/$ uplift.

## 6. Honesty / scope

- This is a **measurement-only** validation audit. The `constraint_aware` engine default rho (≈ 0.65) is **unchanged**, the frontier integration remains opt-in and disabled by default, and real-cluster execution remains disabled by default.
- The committed Azure 2024 audit JSON / backtest summary / frontier-controller summary / integration summary are **read-only** in this audit — no committed artifact is overwritten.
- This is **directional simulator / shadow-mode evidence** — NOT production savings. Pilot telemetry is required to calibrate the safe rho per workload before any production claim.

