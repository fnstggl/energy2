# Azure LLM 2024 — Safe Utilization Frontier Controller v1 Results

> **Simulator / shadow-mode benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Real-cluster execution is **disabled by default** (`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`). This controller selects the highest SLA-safe goodput/$ point across a candidate rho grid, subject to timeout / queue / latency / telemetry-confidence safety gates. No optimizer constant was tuned, no robust energy engine code was touched, no ML model was trained, and no dataset was ingested.

- **Read first:** `docs/RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`, `docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md`, `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`.

## 1. Configuration

- **Controller version:** `frontier_controller_v1`
- **Default execution mode:** `shadow` (real execution disabled by default = True)
- **Safety thresholds:** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms (mirrors the Azure 2024 frontier audit).
- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Workload telemetry confidence:** `medium`

## 2. Frontier sweep (anticipatory — the safer dominant frontier)

| rho | predicted goodput/$ | predicted goodput | timeout % | queue p95/p99 (ms) | GPU-h | mean rho | safety |
|---|---|---|---|---|---|---|---|
| 0.45 | 1,798,572.67 | 30,536,843,128 | 6.46 | 0.1600 / 0.2500 | 8,322.70 | 0.5727 | SAFE |
| 0.55 | 2,188,260.26 | 30,487,533,415 | 6.63 | 0.2100 / 0.3300 | 6,829.60 | 0.5739 | SAFE |
| 0.65 | 2,555,324.54 | 30,214,182,940 | 7.64 | 0.3800 / 0.6300 | 5,796.10 | 0.6234 | SAFE |
| 0.75 | 2,886,960.51 | 29,669,714,655 | 9.46 | 1.56 / 2.82 | 5,037.80 | 0.7161 | SAFE |
| 0.85 | 3,190,680.98 | 29,014,795,698 | 11.65 | 16.85 / 32.82 | 4,457.60 | 0.8089 | UNSAFE (timeout_exceeds_threshold) |
| 0.95 | 3,186,976.66 | 25,998,577,937 | 19.59 | 1,000.26 / 2,065.75 | 3,998.90 | 0.9013 | UNSAFE (timeout_exceeds_threshold, queue_p99_exceeds_threshold) |

## 3. Reactive sweep (diagnostic — not the controller's frontier)

| rho | goodput/$ | timeout % | queue p95/p99 (ms) | GPU-h | safety |
|---|---|---|---|---|---|
| 0.45 | 1,833,582.20 | 6.49 | 15.99 / 33.86 | 8,161.30 | SAFE |
| 0.55 | 2,228,257.64 | 6.82 | 32.94 / 69.66 | 6,697.40 | SAFE |
| 0.65 | 2,594,665.58 | 8.06 | 108.66 / 229.31 | 5,685.00 | SAFE |
| 0.75 | 2,916,406.16 | 10.17 | 602.66 / 1,273.55 | 4,940.80 | **UNSAFE** |
| 0.85 | 3,011,921.12 | 14.99 | 7,858.69 / 16,626.28 | 4,372.70 | **UNSAFE** |
| 0.95 | 1,986,660.36 | 30.27 | 66,629.05 / 141,186.21 | 3,923.20 | **UNSAFE** |

## 4. Controller decision

- **Action:** `RECOMMEND_RHO`
- **Previous rho (constraint_aware default):** 0.65
- **Selected rho:** 0.75
- **Reason:** highest SLA-safe goodput/$ at rho 0.75 (predicted 2,886,960.51 across all SLA / queue / latency / telemetry gates)
- **Expected goodput/$ delta vs current:** 331,635.97
- **Expected GPU-hour delta vs current:** -758.30
- **Expected SLA risk delta:** 0.0000
- **Confidence:** medium
- **Execution mode (recommendation-only):** `shadow` · executable_in_real_cluster = False

### Conservative-margin variant (transparent)

- Selected rho: 0.65 (action `KEEP_RHO`); the controller can be configured to step back from the safety boundary when the next-higher rho is unsafe. This is a transparent operator control, not a hidden default.

## 5. Policy comparison (committed Azure 2024 evidence + controller)

| policy | rho | goodput/$ | timeout % | queue p99 (ms) | GPU-h | safe |
|---|---|---|---|---|---|---|
| sla_aware | 0.5000 | 2,032,039.55 | 6.61 | 33.98 | 7,357.20 | SAFE |
| utilization_aware | — | 3,238,462.77 | 12.10 | 48.42 | 4,372.80 | **UNSAFE** |
| constraint_aware | 0.6500 | 2,555,324.54 | 7.64 | 0.6300 | 5,796.10 | SAFE |
| oracle_forecast_ANALYSIS_ONLY | — | 2,422,788.92 | 7.05 | 0.5000 | 6,149.10 | SAFE |
| frontier_controller_v1 | 0.7500 | 2,886,960.51 | 9.46 | 2.82 | 5,037.80 | SAFE |

- **frontier_controller_v1 vs constraint_aware:** 12.98% goodput/$ (constraint_aware baseline 2,555,324.54).
- **frontier_controller_v1 vs sla_aware:** 42.07% goodput/$ (sla_aware baseline 2,032,039.55).

## 6. Preservation of the committed Azure 2024 baseline

- `constraint_aware` at rho ≈ 0.65: 2,555,324.54 goodput/$ (reproduced within 1.0% of the committed Azure 2024 benchmark — see `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`).
- `sla_aware`: 2,032,039.55 goodput/$ (unchanged).
- The controller does **not** mutate any committed result; it reads the audit frontier and chooses a safe rho. The frontier-audit doc and the canonical Azure 2024 backtest doc remain authoritative for their respective claims.

## 7. Shadow log + execution check

- Shadow log decisions recorded: **2** (executed = 0; modes = ['shadow']).
- Simulator-mode effect: mutated=True, selected_rho=0.75, notes=['simulator mode: set workload azure_llm_2024_week rho_target to 0.75'] (local simulated state only — no production write).
- Real-mode execution: **disabled by default** (`allow_real_execution=False`); even with the flag, no real executor ships in `aurelius.frontier.execution` — `not_implemented_real_executor`.

## 8. Claim discipline

- Simulator / public-trace evidence only — **not production savings** (`docs/RESULTS.md` §8).
- The safe rho is **workload- and SLA-specific**; `rho = 0.75` is **not** a global constant. A different workload mix, SLO, real hardware, or trace will move the safe peak.
- Pilot telemetry is required to calibrate the safe rho per workload before any production-savings claim.
- Real-cluster execution remains disabled by default; the controller recommends only. The committed `constraint_aware` engine default (rho ≈ 0.65) is **unchanged** by this controller.
- The product thesis is *maximum sustainable usage across constraints*. This controller chooses the best safe KPI point — not the highest utilization.

