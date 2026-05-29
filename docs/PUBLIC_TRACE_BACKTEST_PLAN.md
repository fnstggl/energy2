# Public Trace Backtest Plan (CANONICAL_TRACE_BACKTEST_V1 — future)

> **Decision for this PR: do NOT ingest public traces yet.** This file records
> the plan only. Current priority is optimizer correctness + real ICP shadow
> telemetry calibration, not public-trace ingestion. Public traces do **not**
> block pilot readiness.
>
> Read `docs/RESULTS.md` (reporting standard) and `docs/BACKTESTS.md` (the frozen
> CAISO/PJM/ERCOT energy backtest) first. Any future trace backtest must report
> against the canonical KPI (SLA-safe goodput per infrastructure dollar) and
> must not be quoted as a production-savings number.

## Why public traces (later, as CANONICAL_TRACE_BACKTEST_V1)

| Trace | What it provides | How Aurelius would use it |
|---|---|---|
| **BurstGPT** | Real LLM-serving arrival/concurrency, request/response token counts, and failure traces. | Drives realistic queue/TTFT/p99 dynamics and proxy/ingress saturation for the **interactive** scenarios (queue surge, proxy bottleneck, prefix affinity) instead of synthetic Poisson arrivals. |
| **Azure LLM inference traces** | Input/output token demand + traffic shape for replay. | Replay real token-demand for batch-inference / embedding goodput and for cache-affinity (prefix reuse) realism. |
| **Alibaba GPU cluster traces** | GPU utilization / placement / multi-tenant cluster behavior. | Calibrates utilization / fragmentation / bin-packing realism and the packing baselines (`first_fit` / `best_fit` / FFD). |
| **Philly (Microsoft) traces** | Multi-tenant training job scheduling + utilization. | Calibrates training/fine-tuning completion-time + topology-aware placement and the RESERVE_CAPACITY (batch-vs-critical crowding) decision. |

Known sources (for the future ingestion PR — **do not download yet**):
- BurstGPT: https://github.com/HPMLL/BurstGPT
- Azure LLM inference trace: https://github.com/Azure/AzurePublicDataset (`AzureLLMInferenceTrace`)
- Alibaba cluster trace (GPU): https://github.com/alibaba/clusterdata
- Philly traces: https://github.com/msr-fiddle/philly-traces

## Why not now

1. The current gap is **optimizer correctness** (candidate generation, gating,
   next-best safe energy search) and **real ICP shadow telemetry** calibration —
   not more synthetic/public arrival data.
2. The canonical CAISO/PJM/ERCOT energy backtest (`docs/BACKTESTS.md`) is already
   a frozen, deterministic benchmark for the energy path.
3. Public-trace ingestion adds large data dependencies + a download/ETL surface
   that would slow pilot readiness without changing the optimizer's decision
   quality.

## Plan when it lands (CANONICAL_TRACE_BACKTEST_V1)

1. Add a fixed, versioned slice of each trace (committed fixture IDs or a
   pinned download manifest), exactly as `docs/BACKTESTS.md` fixes the energy
   data windows.
2. Replay arrivals/tokens into the cluster simulator's queue/serving model
   (replacing synthetic Poisson) for the interactive scenarios.
3. Keep the same baselines + the same canonical KPI + the same
   `docs/RESULTS.md` reporting (per-workload scorecard, alpha vs safety,
   honest losses).
4. Freeze a golden summary, the same way the energy backtest is frozen.
5. Calibrate the simulator priors against the traces; **only then** may any
   number approach a production claim, and still subject to the §8
   production-claim gate in `docs/RESULTS.md`.

Until then: simulator/benchmark results remain directional only — **not
production savings**.
