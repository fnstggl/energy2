# Aurelius Public-Trace Benchmark Rollup

> **Simulator / public-trace benchmark rollup. Directional only — NOT production
> savings** (`docs/RESULTS.md` §8). This document aggregates every committed
> public-trace and frozen-synthetic benchmark in the repository under the
> canonical reporting standard (`docs/RESULTS.md`). It contains no production
> savings claim, no tuning of constants, no optimizer modifications, and no
> weakening of safety gates. Live customer-telemetry calibration is required
> before any external savings number.
>
> **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`,
> `docs/BACKTESTS.md`.

## 1. Executive summary

Across **8 SLA-safe-eligible public-trace and frozen-synthetic backtests**:

- **Median SLA-safe goodput/$ improvement vs strongest realistic safe
  baseline: +9% (mean +19%, weighted mean by request/job count +26%).**
- **6 wins, 2 safe ties, 0 unsafe regressions** at a ±1% tie band.
- LLM-serving subset (4 traces, 44.1M requests, 9+ days):
  **median +23%**, weighted mean +26%, max +89% (Alibaba GenAI 2026).
- GPU packing + training subset (3 traces, 16,315 jobs, 205+ days):
  **median +0% — SAFE TIE** with the strongest packing baselines; +56% median
  vs naive FIFO. Interpretation: `constraint_aware` already sits on or near the
  safe utilization frontier for these workloads.
- Energy / flexible workload (canonical frozen 1000-job CAISO/PJM/ERCOT
  backtest): **+11% vs `current_price_only`** at 0 deadline misses (where
  `robust_energy_standalone` has lower energy cost but 143 deadline misses —
  UNSAFE).
- **Frontier audits:** static Safe Utilization Frontier Controller wins
  on Azure 2024 (+13% over `constraint_aware`, -13% GPU-hours), ties on
  every other applicable LLM-inference trace; dynamic estimator retains
  73.2% of the oracle alpha on Azure 2024; calibration's aspirational 95%
  oracle-alpha-capture target was **NOT** reached (final 91.07%).
- **No production-savings claim.** Every number here is simulator or
  public-trace replay.

## 2. Headline recommendation

Three tiers — each must be paired with the caveats listed in `outreach_use`
of `data/external/benchmark_rollup/public_trace_benchmark_rollup.json`.

### 2a. Conservative

> Across 8 public-trace and frozen-synthetic backtests, Aurelius improves
> SLA-safe goodput per infrastructure dollar by a median of ~9% (mean ~19%)
> vs the strongest realistic safe baselines, with 6 wins, 2 safe ties, and
> 0 unsafe regressions. Simulator-only — directional, not production
> savings.

### 2b. Strong-but-honest with "up to"

> Aurelius shows up to +89% SLA-safe goodput/$ on real LLM-serving traces
> vs the strongest realistic reactive autoscaler (Alibaba GenAI 2026,
> primarily a model-affinity / prewarm effect), and +26% on Azure's 44.1M
> request 9-day 2024 LLM-inference trace with 21% lower GPU-hours at
> parity SLA safety; LLM-serving median across 4 traces is +23%, no unsafe
> regressions. Simulator-only.

### 2c. Technical-doc / per-workload

> - LLM-serving public traces (n=4, 44.1M requests, 9+ days):
>   **median +23% SLA-safe goodput/$** vs reactive `sla_aware` baselines,
>   with **-21% GPU-hours** on the largest trace (Azure LLM 2024
>   week-long) at parity SLA safety.
> - GPU packing / training scheduling (n=3, 16.3k jobs, 205+ days):
>   `constraint_aware` **ties the strongest safe packing baselines**
>   (best_fit / FFD / topology_aware) and **wins +16% to +62% vs naive
>   FIFO** that suffers head-of-line blocking. The training subset is
>   already at or near its safe frontier.
> - Energy / flexible workload (canonical 1000-job CAISO/PJM/ERCOT
>   backtest): **+11% SLA-safe goodput/$ vs `current_price_only`** at 0
>   deadline misses. The raw-energy-cheapest baseline
>   (`robust_energy_standalone`) misses 143 deadlines — UNSAFE.

## 3. Benchmark inventory

Inventory file: `data/external/benchmark_rollup/benchmark_inventory.json`.

| id | workload class | trace kind | n requests / jobs | duration | summary JSON |
|---|---|---|---|---|---|
| `burstgpt_v1` | llm_serving | public trace replay | 17,689 req | 0.02 d | `data/external/burstgpt/processed/burstgpt_backtest_summary.json` |
| `azure_llm_2023_conv` | llm_serving | public trace replay | 19,366 req | 0.003 d | `data/external/azure_llm/processed/azure_llm_backtest_summary.json` |
| `azure_llm_2024_week` | llm_serving | public trace replay (full) | 44,107,694 req | 9.0 d | `data/external/azure_llm_2024/processed/azure_llm_2024_backtest_summary.json` |
| `alibaba_genai_2026` | llm_serving | public trace replay | 26,392 req | — | `data/external/alibaba_genai/processed/alibaba_genai_backtest_summary.json` |
| `alibaba_gpu_v2023` | gpu_packing | public trace replay | 6,282 jobs | 149.3 d | `data/external/alibaba_gpu/processed/alibaba_gpu_backtest_summary.json` |
| `philly_training` | training_gpu_scheduling | fixture-scale demo | 33 jobs | 0.007 d | `data/external/philly/processed/philly_backtest_summary.json` |
| `mit_supercloud_bounded` | training_gpu_scheduling | bounded real S3 sample | 10,000 jobs | 55.9 d | `data/external/mit_supercloud/processed/mit_supercloud_real_scheduler_frontier_summary.json` |
| `mit_supercloud_fixture` | training_gpu_scheduling | fixture (superseded) | 10 jobs | — | `data/external/mit_supercloud/processed/mit_supercloud_training_frontier_summary.json` |
| `canonical_energy_backtest` | energy_flexible_workload | frozen synthetic on real prices | 1,000 jobs | 26.0 d | `aurelius/benchmarks/golden/canonical_energy_backtest.json` |
| `azure_2024_safe_utilization_frontier` | frontier_static | frontier audit | — | — | `data/external/azure_llm_2024/processed/azure_2024_safe_utilization_frontier.json` |
| `azure_2024_dynamic_frontier` | frontier_dynamic | telemetry-replay estimator | — | — | `data/external/azure_llm_2024/processed/azure_2024_dynamic_frontier_summary.json` |
| `azure_2024_dynamic_frontier_calibration` | frontier_dynamic_calibration | 3-pass shadow eval | — | — | `data/external/azure_llm_2024/processed/azure_2024_dynamic_frontier_calibration_summary.json` |
| `cross_trace_frontier_generalization` | frontier_generalization | meta-audit | 4 applicable, 2 skipped | — | `data/external/frontier/cross_trace_frontier_generalization_summary.json` |
| `full_trace_frontier_validation` | frontier_generalization | full-trace meta-audit | 4 applicable | — | `data/external/frontier/full_trace_frontier_validation_summary.json` |
| `cross_trace_constraint_frontier_integration_safety` | frontier_integration_safety | safety audit | 3 applicable, 3 skipped | — | `data/external/frontier/cross_trace_constraint_frontier_integration_safety_summary.json` |
| `azure_2024_constraint_frontier_integration` | frontier_integration | integration audit | — | — | `data/external/azure_llm_2024/processed/azure_2024_constraint_frontier_integration_summary.json` |
| `alibaba_genai_ablation` | llm_serving_ablation | full-trace ablation | 26,392 req | — | `data/external/alibaba_genai/processed/alibaba_genai_ablation_summary.json` |
| `alibaba_genai_residency_decision` | residency_cold_start | small-sample per-request | 60 req | — | `data/external/alibaba_genai/processed/genai_residency_decision_summary.json` |
| `model_residency_audit` | residency_cold_start | readiness audit | — | — | `data/external/alibaba_genai/processed/model_residency_audit_summary.json` |

**Excluded from aggregate headline:**
- `mit_supercloud_fixture` — superseded by the bounded real S3 sample
  (`mit_supercloud_bounded`).
- `alibaba_genai_residency_decision` — n=60 small-sample per-request
  replay; goodput/$ identical across policies (shared fixed pool); the
  economic payoff at scale is carried by the
  `alibaba_genai_ablation` full-trace result (already included via the
  `alibaba_genai_2026` headline).
- Azure LMM / multimodal trace — **not ingested** in this phase
  (roadmap only).
- ML / neural forecasting results — **no ML training is in scope**
  (`docs/RESULTS.md` §9 — later phase).

## 4. Fair baseline table

Per `docs/RESULTS.md` §3, the strongest *realistic* baseline is workload-class
specific. Oracle / clairvoyant baselines are analysis-only and never the
headline.

| trace | naive baseline | strongest realistic SAFE baseline | rationale |
|---|---|---|---|
| BurstGPT | `fifo` | `cache_affinity_baseline` | uses trace's session/cache key proxy — strongest safe non-CA option; `sla_aware` is also reactive but exhibits worse p99/timeout on this trace |
| Azure LLM 2023 (conv) | `fifo` | `sla_aware` | reactive autoscaler — no cache/session signal in Azure 2023. Static FIFO beats CA on goodput/$ at this mild burst load — reported honestly. |
| Azure LLM 2024 (week) | `fifo` | `sla_aware` | `utilization_aware` has higher goodput/$ but timeout 12.1% > 10% SLA gate (UNSAFE); `sla_aware` is the strongest realistic SAFE reactive baseline |
| Alibaba GenAI 2026 | `fifo` | `sla_aware` | doc headline; SD-serving with model-affinity-decisive lever |
| Alibaba GPU v2023 | `fifo` | `best_fit` | packing baseline per `docs/RESULTS.md` §3 (packing rule); FFD/greedy_packing match within rounding |
| Philly training | `fifo` | `best_fit` | scheduling baseline; matches greedy_packing/topology_aware exactly on this trace |
| MIT Supercloud bounded | `fifo` (UNSAFE — queue p99 56h, starvation 47%) | `best_fit` | matches FFD / topology_aware / utilization_aware exactly |
| Canonical energy | `fifo` | `current_price_only` | the only SAFE non-CA energy baseline; `robust_energy_standalone`, `greedy_energy`, `sla_aware` all miss 119–143 deadlines |

Oracle baselines present in the artifacts (excluded from headline):
- `oracle_forecast_ANALYSIS_ONLY` on Azure 2024 (goodput/$ 2,422,789)
- Realized optimal frontier rho on Azure 2024 dynamic frontier oracle
- `clairvoyant_lower_bound` analysis points on packing benchmarks

## 5. Workload-specific results

### 5a. LLM serving (n=4)

| trace | n_requests | strongest realistic safe | CA goodput/$ | baseline goodput/$ | margin % | safety | GPU-hours Δ |
|---|---:|---|---:|---:|---:|---|---:|
| BurstGPT | 17,689 | `cache_affinity_baseline` | 1,615,694 | 1,587,559 | **+1.77%** | p99 ≤ 0.5× queue_aware; timeout ≤ 0.5× | +0.02 |
| Azure LLM 2023 conv | 19,366 | `sla_aware` | 2,326,157 | 1,940,705 | **+19.86%** | parity p99 & timeout | -0.16 |
| Azure LLM 2024 week | 44,107,694 | `sla_aware` | 2,555,325 | 2,032,040 | **+25.75%** | p99 & timeout ≤ 0.5× FIFO and queue_aware | **-1,561** (-21.2%) |
| Alibaba GenAI 2026 | 26,392 | `sla_aware` | 9.84 | 5.19 | **+89.46%** | e2e p99 ≤ 0.5× queue_aware & utilization_aware | -248 (-21.7%) |

**LLM-serving aggregates:**

- median margin: **+22.81%** · mean: +34.21%
- weighted mean by request count: **+25.78%**
- wins/ties/losses (±1% band): **4/0/0**
- best: Alibaba GenAI 2026 (+89.46%; attribution ~62% affinity/prewarm, ~38% sizing)
- worst: BurstGPT (+1.77% vs cache_affinity_baseline; +26.35% vs `sla_aware`)
- largest single trace: Azure LLM 2024 week-long, 44.1M rows, 9 days

### 5b. Training / GPU scheduling (n=3)

| trace | n_jobs | strongest realistic safe | CA goodput/$ | baseline goodput/$ | margin % | vs FIFO % | safety |
|---|---:|---|---:|---:|---:|---:|---|
| Alibaba GPU v2023 | 6,282 | `best_fit` | 8.238 | 7.7715 | **+6.00%** | +56.43% | matches packing baselines on safety |
| Philly | 33 (fixture) | `best_fit` | 1,362.98 | 1,362.98 | **+0.00%** TIE | +62.26% | matches all packing baselines |
| MIT Supercloud bounded | 10,000 | `best_fit` | 314.83 | 314.83 | **+0.00%** TIE | +16.35% | CA SAFE at every capacity point; FIFO UNSAFE (queue p99 56h, starvation 47%) |

**Training/GPU aggregates:**

- median margin: **+0.00%** (TIE) · mean: +2.00%
- vs naive FIFO median: **+56.43%**
- wins/ties/losses: **1/2/0**
- interpretation: `constraint_aware` is **already on or near the safe
  training/packing frontier**. Wins come from price-aware GPU routing on
  Alibaba v2023 (only trace with heterogeneous GPU prices); on Philly +
  MIT no price signal is published, so CA matches best_fit exactly.

### 5c. Energy / flexible workload (n=1)

Canonical frozen 1000-job CAISO/PJM/ERCOT backtest (golden snapshot
`aurelius/benchmarks/golden/canonical_energy_backtest.json`).

| policy | goodput/$ | energy cost ($) | infra cost ($) | deadline misses | safety |
|---|---:|---:|---:|---:|---|
| `fifo` | 0.166 | 70,347 | 105,241 | 0 | SAFE |
| `current_price_only` | 0.304 | 22,133 | 57,453 | 0 | **SAFE** — strongest realistic baseline |
| `greedy_energy` | 0.281 | 20,773 | 56,094 | 119 | UNSAFE |
| `robust_energy_standalone` | 0.301 | 14,561 | 49,882 | 143 | UNSAFE (the standalone engine ignores warmup) |
| `sla_aware` | 0.298 | 15,061 | 50,314 | 143 | UNSAFE on this scoring |
| `constraint_aware_with_energy_adapter` | **0.337** | 16,486 | 51,726 | **0** | **SAFE — wrapper eliminates all 143 deadline misses + +11% goodput/$ alpha** |

- **+11.07% vs `current_price_only`** (the strongest SAFE baseline).
- **+103.46% vs `fifo`** (naive sanity baseline).
- Wrapper accepts 698/1000 engine picks and 141 current-price-only picks;
  rejects 161 destinations as fallback-home for safety; 137 rejections
  flagged `ineligible_critical_interactive_inference`, 143 flagged
  `sla_unsafe_deadline_with_warmup`.
- This is the **SAFETY_WIN with alpha** pattern: standalone engine has
  lower raw energy cost but is warmup-blind; wrapper recovers safety
  AND modestly improves the canonical KPI.

### 5d. Frontier audits

#### Static Safe Utilization Frontier Controller vs `constraint_aware`

| trace | CA goodput/$ | static-frontier goodput/$ | Δ % | GPU-hours Δ % | verdict |
|---|---:|---:|---:|---:|---|
| BurstGPT (fixture) | 212,103 | 215,061 | +1.39% | (within tolerance) | FRONTIER_WIN |
| Azure LLM 2023 | 1,740,426 | 1,740,426 | +0.00% | — | TIE |
| Azure LLM 2024 week | 2,555,325 | 2,886,961 | **+12.98%** | **-13.08%** | FRONTIER_WIN |
| Alibaba GenAI 2026 | 3.33 | 3.33 | +0.00% | — | TIE |
| BurstGPT (full raw) | 50,677 | 50,630 | -0.09% | — | SAFE_TIE |
| Azure LLM 2023 conv (full) | 1,904,272 | 1,904,272 | +0.00% | — | SAFE_TIE |
| Azure LLM 2023 code (full) | 124,428 | 124,428 | +0.00% | — | SAFE_TIE |

Generalization verdict (`cross_trace_frontier_generalization_summary.json`
synthesis): `GENERALIZES_WITHIN_APPLICABLE_LLM_INFERENCE_TRACES`;
architecture recommendation `KEEP_FRONTIER_CONTROLLER_SEPARATE_OR_OPT_IN`.

#### Dynamic frontier estimator (Azure 2024 week-long)

| label | goodput/$ | rho | safe |
|---|---:|---:|---|
| `sla_aware` | 1,036,721 | 0.50 | SAFE |
| `constraint_aware_static` | 1,139,320 | 0.65 | SAFE |
| `static_frontier_controller` | 1,181,075 | 0.75 | SAFE |
| `utilization_aware` (oracle ceiling, target rho 0.85) | 1,198,981 | 0.85 | SAFE |
| **`dynamic_frontier_estimator` (w30/60/180m)** | **1,182,973** | dynamic 0.531 mean | SAFE |

- Dynamic estimator retains **73.2%** of the oracle alpha between
  `constraint_aware_static` and the realized optimal frontier rho.
- Verdict: `DYNAMIC_BEATS_STATIC_CA`.

#### Dynamic frontier calibration (3-pass shadow eval at scale 100×)

- Final oracle-alpha capture: **91.07%** (aspirational target 95% — NOT reached).
- False-safe rate: 0.45% · false-unsafe rate: 14.11%.
- Closing the remaining gap is a pilot-telemetry calibration task,
  not a constant-tuning task (per `docs/AZURE_2024_DYNAMIC_FRONTIER_CALIBRATION_RESULTS.md`).

#### Cross-trace constraint × frontier integration safety

- Applicable traces (LLM-inference, frontier integration enabled):
  BurstGPT, Azure LLM 2023, Azure LLM 2024 week.
- Skipped traces (workload class not applicable): Alibaba GenAI 2026,
  Alibaba GPU v2023, Philly.
- Verdict counts: 2 INTEGRATION_WIN · 1 SAFE_TIE · 0 regressions.
- **Any regression: false. Safe-or-win: 100%.**

## 6. Aggregate metrics (across the 8 safe-eligible benchmarks)

| metric | vs strongest realistic safe baseline | vs naive FIFO baseline |
|---|---:|---:|
| **median margin** | **+8.54%** | +59.34% |
| mean margin | +19.24% | +98.51% |
| weighted mean by request/job count | **+25.77%** | (see per-class) |
| min margin | +0.00% | -7.52% (Azure 2023, honest caveat) |
| max margin | +89.46% | +456.71% |
| wins / ties / losses (±1% band) | **6 / 2 / 0** | n/a (FIFO sanity only) |

Total real requests covered: **~44.18M.** Total real jobs covered:
**~16,315.** Approximate days covered: **~215 d** in the safe-eligible
rollup (270 d including frontier audits).

## 7. Win / tie / loss counts

Vs **strongest realistic safe baseline** at a ±1% tie band:

- **6 wins:** BurstGPT, Azure LLM 2023 conv, Azure LLM 2024 week,
  Alibaba GenAI 2026, Alibaba GPU v2023, canonical energy backtest.
- **2 ties:** Philly (fixture-scale), MIT Supercloud bounded.
- **0 losses.**
- **0 SLA-violation regressions** (no benchmark increases
  `constraint_aware` timeout/SLA-violation count above the strongest
  realistic safe baseline).

## 8. Safety outcomes

- `constraint_aware` is SAFE on **8/8** safe-eligible benchmarks.
- No unsafe-baseline result is used as the headline comparator.
- Unsafe baselines explicitly excluded from the headline:
  - Azure LLM 2024 `utilization_aware` (timeout 12.10% > 10% gate)
  - Energy backtest `robust_energy_standalone` (143 deadline misses)
  - Energy backtest `greedy_energy` (119 deadline misses)
  - Energy backtest `sla_aware` on this trace's scoring (143 deadline misses)
- Naive FIFO is itself UNSAFE on MIT Supercloud bounded (queue p99 ~56h,
  starvation 47%) — `constraint_aware` is a SAFETY_WIN there in addition
  to a +16% economic uplift.

## 9. Statistical coverage

- **8 safe-headline traces · 19 total committed benchmark / audit
  artifacts** (8 + 6 frontier audits + 2 ablations / diagnostics +
  3 other).
- **~44.18M real requests** across LLM-serving replays.
- **~16.3k real jobs** across GPU packing + training scheduling.
- **~215 days of real demand** in the safe-eligible rollup
  (Alibaba GPU v2023 149 d + MIT Supercloud bounded 55.9 d +
  canonical energy 26 d + Azure LLM 2024 week 9 d + others < 1 d each).
- **Public trace names:** BurstGPT (HPMLL) · Azure LLM Inference 2023
  (Microsoft Azure) · Azure LLM Inference Dataset 2024 (DynamoLLM HPCA
  2025) · Alibaba GenAI 2026 (GenTD26) · Alibaba cluster-trace-gpu-v2023
  · Microsoft Philly · MIT Supercloud datacenter challenge · CAISO/PJM/
  ERCOT day-ahead and real-time prices.
- **Raw vs fixture status:** see `statistical_coverage.raw_vs_fixture_status`
  in `public_trace_benchmark_rollup.json`. Philly canonical and the v1
  MIT result are FIXTURE-scale and labelled as such; MIT bounded is a
  REAL S3 sample (~3 MB downloaded of ~1-2 TB full archive).
- **Production-ready:** NO. No result here is a production-savings
  claim. `docs/RESULTS.md` §8 production-claim gate is not satisfied for
  any trace.

## 10. What number to use in outreach

- **Conservative:** *"Across 8 public-trace and frozen-synthetic
  backtests, Aurelius improves SLA-safe goodput per infrastructure
  dollar by a median of ~9% (mean ~19%) vs the strongest realistic safe
  baselines, with 6 wins, 2 safe ties, 0 unsafe regressions. Simulator
  result — directional, not production savings."*
- **Strong-but-honest:** *"Up to +89% SLA-safe goodput/$ on real LLM-
  serving traces; LLM-serving median across 4 traces is +23%, with no
  unsafe regressions. On Azure's 44.1M-request 9-day 2024 LLM-inference
  trace, +26% SLA-safe goodput/$ at 21% lower GPU-hours, with parity
  SLA safety. Simulator result — directional, not production savings."*
- **Technical:** per workload-class (§5 + §2c).

## 11. What number NOT to use

- The **+89% Alibaba GenAI** headline as a stand-alone claim — it is
  workload-specific (model-affinity-heavy stable-diffusion serving) and
  must be paired with the LLM-serving median.
- The **+457% Alibaba GenAI vs FIFO** number — FIFO is a sanity
  baseline, not a buyer-facing comparator.
- The **dynamic frontier estimator's 91% oracle-alpha capture** as a
  production figure — the 95% aspirational target was **not** reached
  and closing the gap requires pilot calibration.
- Any number from the **residency decision engine small-sample
  (n=60) per-request replay** as economic alpha — it's a diagnostic.
- The **`robust_energy_standalone`** lower energy cost — UNSAFE, 143
  deadline misses.
- The **`utilization_aware`** higher goodput/$ on Azure 2024 — UNSAFE,
  timeout 12.10% > 10% gate.
- Any **production-savings** phrasing — `docs/RESULTS.md` §8 gate is
  not met.

## 12. Limitations / no production-savings claim

- Simulator / public-trace results only. **NOT production savings.**
- No customer telemetry calibration, no customer cost basis, no
  shadow-pilot validation. The `docs/RESULTS.md` §8 production-claim
  gate is **not satisfied** for any of these traces.
- BurstGPT cache-affinity proxy is **model-level**, not a real KV-cache
  hit rate. Azure 2023/2024 have **no** cache/session/latency signal at
  all (token-demand and arrival replay only).
- Philly canonical is **fixture-scale** (full ~1 GB LFS trace not
  committed); MIT Supercloud is a **bounded real S3 sample** (~3 MB of
  ~1–2 TB).
- Canonical energy backtest uses a **synthetic 1000-job workload mix**;
  not customer-derived.
- Oracle / clairvoyant baselines are present for analysis only and are
  **never** used as the headline comparator.
- This document does not modify any optimizer constant, gate, or
  baseline. Existing benchmark summary artifacts are unchanged.

---

**Artifacts:**

- Inventory JSON: `data/external/benchmark_rollup/benchmark_inventory.json`
- Rollup JSON: `data/external/benchmark_rollup/public_trace_benchmark_rollup.json`
- Tests: `tests/test_public_trace_benchmark_rollup.py`
