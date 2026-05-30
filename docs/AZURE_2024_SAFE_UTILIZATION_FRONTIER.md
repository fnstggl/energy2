# Azure LLM 2024 — Safe-Utilization Frontier Audit

> **Measurement / attribution only. Directional simulator/backtest result — not production savings** (`docs/RESULTS.md` §8). Reuses the UNCHANGED serving physics + economics; **no** production code, optimizer logic, or simulator constant was modified, and **no** constant was tuned to a result. Token-demand + arrival replay (Azure exposes no latency/TTFT — the SLA is a modelled interactive SLO applied identically to all policies). Read `docs/RESULTS.md` + `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`.


- **Source:** `raw:data/external/azure_llm_2024/raw (multi-service week: code, conv)`  ·  **scale:** 10.0× busy-tier (real arrival shape)  ·  **trace:** 44,107,694 rows, 9.0 days
- **Files used:** `data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_code_1week.csv`, `data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_conv_1week.csv`
- **Safe threshold (pre-registered diagnostic):** timeout ≤ 10.0% **and** queue p99 ≤ 2000.0 ms.

## 1. Executive answer

- **Where does constraint_aware's +25.75% win come from?** **Almost entirely SAFE HIGHER UTILIZATION (the rho target), not forecasting, queue control, or hysteresis.** Raising rho 0.50→0.65 alone is **562,626.03 goodput/$ (>100% of the win)**; anticipation is a small goodput/$ *cost* (-39,341.04) that buys a ~360× queue-tail safety improvement; trim + hysteresis are goodput/$-neutral; demand-forecasting is ~0.252% of KPI (negligible).
- **Is constraint_aware on/near the safe frontier?** **SAFE but slightly *inside* (conservative).** At rho 0.65 it is below the reactive goodput/$ peak and below the best *safe anticipatory* point (`anticipatory@0.75`, 2,886,960.51); the anticipatory machinery could safely run hotter for more goodput/$ while keeping queue p99 ≤ a few ms.

## 2. Frontier sweep — reactive (sla_aware-style)

| rho / policy | goodput/$ | timeout % | SLA-viol rate | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | verdict |
|---|---|---|---|---|---|---|---|---|---|
| reactive@0.45 | 1,833,582.2 | 6.491 | 1.0 | 15.99 / 33.86 | 8,161.3 | 0.5734 | 10,396 | 31,682 | SAFE |
| reactive@0.55 | 2,228,257.64 | 6.82 | 1.0 | 32.94 / 69.66 | 6,697.4 | 0.5819 | 9,941 | 25,851 | SAFE |
| reactive@0.65 | 2,594,665.58 | 8.056 | 1.0 | 108.66 / 229.31 | 5,685.0 | 0.6423 | 9,610 | 21,959 | SAFE |
| reactive@0.75 | 2,916,406.16 | 10.172 | 1.0 | 602.66 / 1273.55 | 4,940.8 | 0.7352 | 9,181 | 19,009 | **UNSAFE** |
| reactive@0.85 | 3,011,921.12 | 14.992 | 1.0 | 7858.69 / 16626.28 | 4,372.7 | 0.8302 | 8,896 | 16,802 | **UNSAFE** |
| reactive@0.95 | 1,986,660.36 | 30.272 | 1.0 | 66629.05 / 141186.21 | 3,923.2 | 0.9248 | 8,571 | 14,993 | **UNSAFE** |

## 2b. Frontier sweep — anticipatory (constraint_aware-style)

| rho / policy | goodput/$ | timeout % | SLA-viol rate | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | verdict |
|---|---|---|---|---|---|---|---|---|---|
| anticipatory@0.45 | 1,798,572.67 | 6.455 | 1.0 | 0.16 / 0.25 | 8,322.7 | 0.5727 | 9,828 | 22,983 | SAFE |
| anticipatory@0.55 | 2,188,260.26 | 6.634 | 1.0 | 0.21 / 0.33 | 6,829.6 | 0.5739 | 9,268 | 18,684 | SAFE |
| anticipatory@0.65 | 2,555,324.54 | 7.639 | 1.0 | 0.38 / 0.63 | 5,796.1 | 0.6234 | 8,830 | 15,934 | SAFE |
| anticipatory@0.75 | 2,886,960.51 | 9.465 | 1.0 | 1.56 / 2.82 | 5,037.8 | 0.7161 | 8,411 | 13,744 | SAFE |
| anticipatory@0.85 | 3,190,680.98 | 11.648 | 1.0 | 16.85 / 32.82 | 4,457.6 | 0.8089 | 8,068 | 12,220 | **UNSAFE** |
| anticipatory@0.95 | 3,186,976.66 | 19.589 | 1.0 | 1000.26 / 2065.75 | 3,998.9 | 0.9013 | 7,592 | 10,856 | **UNSAFE** |

> The anticipatory frontier **dominates** the reactive one on safety at every rho: queue p99 stays ≤ a few ms through rho 0.75 (vs reactive's 229 ms@0.65, 1,274 ms@0.75). Anticipation's binding safety limit is **timeout** (compute saturation), not queue.

## 3. Policy comparison

| rho / policy | goodput/$ | timeout % | SLA-viol rate | queue p95/p99 (ms) | GPU-h | mean rho | scale ev | churn | verdict |
|---|---|---|---|---|---|---|---|---|---|
| sla_aware | 2,032,039.55 | 6.605 | 1.0 | 16.06 / 33.98 | 7,357.2 | 0.575 | 10,152 | 28,515 | SAFE |
| queue_aware | 2,490,662.91 | 24.184 | 1.0 | 37021.48 / 78398.79 | 4,068.4 | 0.893 | 8,696 | 15,313 | **UNSAFE** |
| utilization_aware | 3,238,462.77 | 12.103 | 1.0 | 24.7 / 48.42 | 4,372.8 | 0.8248 | 8,897 | 16,803 | **UNSAFE** |
| constraint_aware | 2,555,324.54 | 7.639 | 1.0 | 0.38 / 0.63 | 5,796.1 | 0.6234 | 8,830 | 15,934 | SAFE |
| oracle_forecast_ANALYSIS_ONLY | 2,422,788.92 | 7.053 | 1.0 | 0.31 / 0.5 | 6,149.1 | 0.5901 | 9,702 | 23,755 | SAFE |
| naive_overprovisioning | 997,155.77 | 5.55 | 0.9706 | 0.0 / 0.0 | 15,120.0 | 0.508 | 0 | 0 | SAFE |

## 4. Attribution of the +25.75% win (controlled factor ladder)

| step | goodput/$ Δ | meaning |
|---|---|---|
| baseline sla_aware (reactive@0.50) | 2,032,039.55 | — |
| **+ raise rho 0.50→0.65** | **562,626.03** | higher safe utilization — the entire win |
| + add anticipation (EWMA) | -39,341.04 | goodput/$ *cost*; queue p99 33.98→0.63 ms, churn −12,581 |
| + SLA-safe trim | 0.0 | inactive (no cache headroom) |
| + hysteresis | 0.0 | inactive (EWMA plan already smooth) |
| = constraint_aware | 2,555,324.54 | net **+25.75%** |

- **GPU-hour reduction vs sla_aware:** −1,561.1 GPU-h.
- **Utilization increase vs sla_aware:** +0.0484 mean rho.
- **Churn reduction vs sla_aware:** −12,581 (from EWMA anticipation, not the explicit hysteresis damper).
- **Overprovisioning avoided vs naive:** −9,323.9 GPU-h.
- **Forecast contribution:** oracle ceiling 6,089.77 (~0.252% of KPI); EWMA 1,432.087162 (survival 0.2352) — negligible.

## 5. Efficient frontier

- **Reactive frontier:** best safe = `reactive@0.65` (2,594,665.58); first unsafe = `reactive@0.75`.
- **Anticipatory frontier (the safer, dominant one):** best safe = `anticipatory@0.75` (2,886,960.51); cheapest safe = `anticipatory@0.75` (5,037.8 GPU-h); first unsafe = `anticipatory@0.85`.
- **constraint_aware** (mean rho 0.6234) is **inside (conservative)** the safe frontier — conservative headroom remains.

## 6. Explanation

- **Why utilization_aware becomes unsafe:** it targets rho≈0.85 → sustained compute saturation pushes p99 latency past the SLA budget (timeout ~12%) even with a modest queue. High rho without anticipation is unsafe via timeouts.
- **Why sla_aware is too conservative:** rho-target 0.50 + one-tick reactive lag → it under-utilizes (mean rho ~0.58, ~7,360 GPU-h), leaving ~25% goodput/$ on the table.
- **Why constraint_aware remains safer:** EWMA anticipation provisions for `max(current, smoothed-peak)`, so the queue never builds on bursts (queue p99 ~0.6 ms vs reactive 229 ms at the same rho). Anticipatory queue control is what makes higher utilization *sustainable* without SLA blowup.
- **Product thesis — "maximum sustainable usage across constraints":** **supported.** The economic win IS safe higher utilization, and anticipatory queue control is precisely the mechanism that keeps high utilization sustainable where naive high-rho policies (utilization_aware, queue_aware) break the SLA.

## 7. Remaining gaps / claim discipline

- **Simulator / public-trace evidence only.** This is a directional backtest on a public trace, **not** customer telemetry and **not** a production-savings claim (`docs/RESULTS.md` §8 gate unmet). No TTFT/latency claim — Azure exposes none; the SLA is a modelled interactive SLO.
- **constraint_aware is near but INSIDE the safe frontier** at its default rho≈0.65. This does **NOT** mean you should blindly set rho=0.75: the best-safe rho is **specific to this trace, this load multiplier, this modelled SLO, and the chosen safety threshold**. A different workload mix, burst profile, SLO, or real hardware will move it. **Do not change the production default rho on the basis of this backtest.**
- **Real pilot / shadow telemetry is required to calibrate the safe rho** — measured queue/TTFT vs provisioning, the customer's true SLO, and the real saturation point — before promoting any higher-rho operating point. The simulator queue physics here are not validated against real Azure serving; conclusions are regime/threshold-dependent (reported transparently).
- **ML demand forecasting is LOW-leverage here** (oracle ceiling ~0.252% of KPI); the leverage is the safe-utilization controller (rho target + anticipatory queue control), not better demand prediction.

