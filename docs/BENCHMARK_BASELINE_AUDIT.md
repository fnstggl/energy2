# Aurelius Benchmark Baseline Audit

> **Audit only. No optimizer changes, no benchmark logic changes, no simulator
> re-runs.** This document traces every baseline implementation in every
> committed Aurelius benchmark to determine whether the "strongest realistic
> baseline" the rollup compares against is a *modern* scheduler/inference stack
> or merely a stronger version of FIFO. Simulator / public-trace evidence
> only — directional only, NOT production savings (`docs/RESULTS.md` §8).
>
> **Read first:** `docs/RESULTS.md`, `docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md`,
> `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/BACKTESTS.md`.

## 0. TL;DR

- **Continuous batching is shared serving physics, not a policy lever.** Every
  LLM-serving policy (`fifo`, `sla_aware`, `queue_aware`, `utilization_aware`,
  `constraint_aware`, etc.) gets the simulator's `batching_efficiency` model
  for free (`aurelius/simulation/cluster/serving.py:114`). Aurelius vs
  `sla_aware` is **autoscaler-vs-autoscaler**, not "with continuous batching
  vs without."
- **No baseline models KV-aware admission, real prefix caching, model-
  affinity routing, speculative decoding, chunked prefill, or
  disaggregated prefill/decode.** A real Azure / OpenAI / Anthropic stack
  (vLLM PagedAttention, TensorRT-LLM, SGLang, Mooncake, DistServe, custom
  hyperscaler inference) has these. **None of the implemented baselines do.**
- **GenAI 2026 is under-baselined.** Only `constraint_aware` has
  `affinity=True` (`aurelius/traces/genai_backtest.py:194`). The +89.46%
  headline is +89% against a baseline with no model-affinity routing,
  despite 87 distinct base models. The ablation explicitly attributes
  ~62% of the gain to affinity. Expected fair margin if a
  `sla_aware_with_affinity` baseline existed: ~+34%.
- **Training / GPU packing is FAIRLY baselined.** `best_fit` / `FFD` /
  `topology_aware` / `greedy_packing` are standard production scheduler
  baselines (Slurm / Kubernetes / YARN equivalents). Aurelius TIES them on
  Philly + MIT, wins +6% on Alibaba GPU v2023 (heterogeneous-fleet price-
  aware lever). Honest.
- **Energy backtest is FAIRLY baselined.** `current_price_only` is the
  strongest SAFE non-Aurelius baseline; `robust_energy_standalone` IS the
  Aurelius energy engine itself (a self-comparison). +11% vs
  `current_price_only` at 0 deadline misses is honest.

**Final question answered (§7):** Of the headline median ~+9% (rollup): vs
a realistic *hyperscaler-equivalent* baseline (NOT IMPLEMENTED in any
benchmark; estimated directionally), the conservative residual is roughly
**+3% to +15% on LLM-serving traces**, **+0% to +6% on GPU packing**,
**+0% (tie) on training**, **+3% to +11% on energy**. The +89% GenAI
headline is the most affected — it collapses to ~+20–40% if a baseline
had model-affinity routing.

## 1. Methodology

For each baseline used in each benchmark:

1. **Locate** the implementation file + function + lines.
2. **Trace** the code to enumerate features actually present in the
   decision path.
3. **Mark** each feature `Y` / `N` / `UNKNOWN` with a citation
   (file:line) for every `Y` and a reason for every `N`.
4. **Classify** the baseline as `naive` / `modern` / `advanced` /
   `oracle_only`.
5. **Judge** whether the rollup's "strongest realistic baseline" is the
   right pick: was it the strongest *implemented* alternative? would a
   *real* production stack have features none of these baselines model?
6. **Score** trust: baseline strength (1–10), production similarity
   (1–10), headline credibility (1–10).
7. **Generate** outreach guidance: vs FIFO claims? vs modern scheduler
   claims? vs hyperscaler claims? enterprise outreach?

Full machine-readable matrix:
`data/external/benchmark_rollup/baseline_capability_matrix.json`.

## 2. Per-benchmark capability matrix (summary)

> Y = feature present in decision code · N = absent · `~` = partial.
> Full evidence (file:line citations) in the JSON matrix.

### LLM serving — shared `aurelius/traces/backtest.py` `_run_policy`

| baseline | dyn batch | cont batch | autoscale | queue-aw | latency-aw | timeout-aw | cache | affinity | placement | migr | energy | carbon | deadline | util-aw | frontier | dyn-frontier | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fifo` | N | N | N | N | N | N | N | N | N | N | N | N | N | Y | N | N | **naive** |
| `cache_affinity_baseline` | N | N | N | N | N | N | **Y** | ~ | N | N | N | N | N | Y | N | N | **modern** |
| `sla_aware` | N | N | **Y** | N | ~ | N | N | N | N | N | N | N | N | Y | N | N | **modern** |
| `queue_aware` | N | N | **Y** | **Y** | ~ | N | N | N | N | N | N | N | N | N | N | N | **modern** |
| `utilization_aware` | N | N | **Y** | N | N | N | N | N | N | N | N | N | N | Y | N | N | **modern** |
| `naive_overprovisioning` | N | N | N | N | N | N | N | N | N | N | N | N | N | Y | N | N | **naive** |
| `oracle_forecast_ANALYSIS_ONLY` | N | N | Y | N | N | N | N | N | N | N | N | N | N | Y | N | N | **oracle_only** |
| `constraint_aware` | N | N | **Y** | ~ | **Y** | **Y** | **Y** | ~ | N | **Y** | N | N | N | Y | opt-in | opt-in | **advanced** |

**Continuous batching note:** every row above has `N` for continuous-batching
**as a policy lever**, but every row gets the simulator's
`serving.batching_efficiency` model identically
(`aurelius/simulation/cluster/serving.py:114`). The comparison is
autoscaler-vs-autoscaler on top of a shared batching physics, not
"with vs without batching."

### GenAI serving — `aurelius/traces/genai_backtest.py` `_run_policy`

| baseline | autoscale | queue-aw | latency-aw | timeout-aw | affinity | placement | util-aw | class |
|---|---|---|---|---|---|---|---|---|
| `fifo` | N (peak static) | N | N | N | **N** | N | Y | **naive** |
| `sla_aware` | Y | N | Y | Y | **N** | N | N | **modern** |
| `queue_aware` | Y | N | N | N | **N** | N | Y | **modern** |
| `utilization_aware` | Y | N | N | N | **N** | N | Y | **modern** |
| `constraint_aware` | Y (EWMA-peak) | N | Y | Y | **Y** | N | Y | **advanced** |

**Critical:** `affinity` flag is set to `(policy == "constraint_aware")` —
hard-coded at `aurelius/traces/genai_backtest.py:194`. No other baseline
gets affinity routing. This is the +62% Shapley term per the ablation
(`aurelius/traces/genai_ablation.py:227+`).

### GPU packing — `aurelius/traces/gpu_packing.py` `_select_node`

| baseline | placement-aw | util-aw | affinity (GPU type) | migration | class |
|---|---|---|---|---|---|
| `fifo` | ~ (round-robin spread) | N | N | N | **naive** |
| `first_fit` | Y | ~ | N | N | **modern** |
| `best_fit` | Y (tight + active-first) | Y | N | N | **modern** |
| `first_fit_decreasing` | Y (FFD job order + first_fit) | Y | N | N | **modern** |
| `greedy_packing` | Y (FFD + best_fit) | Y | N | N | **modern** |
| `topology_aware` | Y (right-size node to job) | Y | N | N | **modern** |
| `utilization_aware` | Y (max-utilised first) | Y | N | N | **modern** |
| `constraint_aware` | Y | Y | Y (cheapest GPU type) | ~ | **advanced** |

### Training / GPU scheduling — `aurelius/traces/gpu_scheduling.py`

| baseline | placement-aw | backfill | gang | retry-aware | starvation-aware | class |
|---|---|---|---|---|---|---|
| `fifo` | N (strict HoL) | **N** | N | N | N | **naive** |
| `first_fit` / `best_fit` / `FFD` / `greedy_packing` / `topology_aware` / `utilization_aware` | Y (variant per policy) | **Y** | ~ | N | ~ | **modern** |
| `constraint_aware` | Y | Y | ~ | N | ~ | **advanced** (CA's heterogeneous-pricing lever is INACTIVE on Philly/MIT since no GPU price published) |

### Energy / flexible workload — `aurelius/backtesting/baselines.py` + `aurelius/optimization/scheduler.py`

| baseline | energy-aw | carbon-aw | deadline-aw | migration-aw | placement-aw | warmup-aware | safety class |
|---|---|---|---|---|---|---|---|
| `fifo` | N | N | N | N | N | N | safe-cheap-overprovisioned, **naive** |
| `current_price_only` | Y (price@start) | N | ~ (no slack use) | N | Y | N | **SAFE modern** |
| `greedy_energy` | Y (cheapest hour in window) | N | ~ | N | Y | N | **UNSAFE** (119 misses), **modern** |
| `robust_energy_standalone` (Aurelius engine) | Y | Y | ~ (deadline as constraint) | ~ (network cost) | Y | **N** | **UNSAFE** (143 misses), **advanced** |
| `sla_aware` (engine + safety revert) | Y | Y | Y (revert) | ~ | Y | ~ | **UNSAFE** (143 misses on warmup-aware scoring), **advanced** |
| `constraint_aware_with_energy_adapter` | Y | Y | Y (warmup-aware) | Y | Y | **Y** | **SAFE** (0 misses), **advanced** |

### Frontier audits

- **Static frontier controller** (`aurelius/frontier/controller.py:120`):
  estimates a rho-grid Pareto frontier with safety vetoes; selects highest
  SLA-safe goodput/$. Self-comparison only (CA-default vs CA-with-controller).
- **Dynamic frontier estimator** (`aurelius/frontier/dynamic_*.py`):
  telemetry-replay window-based rho selection. Self-comparison only.
  Retains 73% of oracle alpha on Azure 2024.
- **Dynamic frontier calibration**: oracle-alpha-capture diagnostic. Pure
  quality metric; not a goodput/$ headline.
- **Eval / batch frontier v1** (newly merged): recommendation-only
  controllers that sweep a candidate grid. NOT external baselines; no
  `sla_aware` / `queue_aware` comparison.

## 3. Baseline classification (full)

| benchmark | baseline | class | strongest-realistic in rollup? |
|---|---|---|---|
| Azure LLM 2024 | `fifo` | naive | — |
| Azure LLM 2024 | `sla_aware` | modern | **SELECTED** |
| Azure LLM 2024 | `queue_aware` | modern | alternative |
| Azure LLM 2024 | `utilization_aware` | modern | UNSAFE — disqualified |
| Azure LLM 2024 | `naive_overprovisioning` | naive | — |
| Azure LLM 2024 | `oracle_forecast_ANALYSIS_ONLY` | oracle_only | excluded |
| Azure LLM 2024 | `constraint_aware` | advanced | — (target) |
| BurstGPT | `cache_affinity_baseline` | modern | **SELECTED** |
| BurstGPT | `sla_aware` | modern | alternative |
| Azure LLM 2023 | `sla_aware` | modern | **SELECTED** (fifo beats CA on this trace — labelled honestly) |
| Alibaba GenAI 2026 | `sla_aware` | modern (no affinity) | **SELECTED** — but UNDER-BASELINED for this trace |
| Alibaba GenAI 2026 | `utilization_aware` | modern | alternative (also no affinity) |
| Alibaba GPU v2023 | `best_fit` | modern | **SELECTED** |
| Philly | `best_fit` | modern | **SELECTED** |
| MIT Supercloud bounded | `best_fit` | modern | **SELECTED** |
| Energy canonical | `current_price_only` | modern | **SELECTED** |
| Energy canonical | `robust_energy_standalone` | advanced | UNSAFE — disqualified |
| Static/dynamic frontier | `constraint_aware` (self) | advanced | self-comparison |
| Dynamic frontier calibration | oracle | oracle_only | diagnostic only |
| Eval / batch frontier v1 | candidate-grid sweep | advanced | no external baseline |

## 4. Was the selection fair?

| benchmark | selected | alternative | was_selection_fair | confidence |
|---|---|---|---|---|
| Azure LLM 2024 | `sla_aware` | `queue_aware` (+2.6%) | **PARTIAL_YES** — fair vs implemented options; no hyperscaler baseline exists | MEDIUM |
| BurstGPT | `cache_affinity_baseline` | `sla_aware` | **YES** | MEDIUM |
| Azure LLM 2023 conv | `sla_aware` | (fifo beats CA — disclosed) | **YES_WITH_HONEST_CAVEAT** | MEDIUM |
| Alibaba GenAI 2026 | `sla_aware` (no affinity) | `utilization_aware` (still no affinity); MISSING `sla_aware_with_affinity` baseline | **NO — UNDER-BASELINED** | HIGH |
| Alibaba GPU v2023 | `best_fit` | `greedy_packing` (identical) | **YES** | HIGH |
| Philly | `best_fit` | `topology_aware`/`greedy_packing` (identical) | **YES** | MEDIUM (fixture-scale) |
| MIT Supercloud bounded | `best_fit` | any other safe packing (identical) | **YES** | HIGH |
| Canonical energy | `current_price_only` | `robust_energy_standalone` (UNSAFE) | **YES given safety gate** | HIGH |
| Frontier audits | self | — | YES for self-audit | HIGH |

## 5. Benchmark trust scores

| benchmark | baseline_strength | production_similarity | headline_credibility | notes |
|---|---:|---:|---:|---|
| Azure LLM 2024 week | 5 | 4 | 6 | sla_aware is modern-but-basic; real hyperscaler has KV-aware admission + prefix caching not modelled |
| BurstGPT | 6 | 4 | 6 | cache_affinity_baseline is right comparator; model-level proxy is weak vs real KV cache |
| Azure LLM 2023 conv | 5 | 4 | 5 | static FIFO beats CA — honest |
| Alibaba GenAI 2026 | 3 | 4 | 4 | **UNDER-BASELINED** — no baseline models model-affinity routing |
| Alibaba GPU v2023 | 8 | 6 | 7 | best_fit is standard scheduler baseline |
| Philly | 7 | 4 | 4 | fixture-scale (n=33); TIE outcome is honest |
| MIT Supercloud bounded | 8 | 6 | 7 | real 10k-job Slurm sample; TIE is honest |
| Canonical energy | 6 | 4 | 6 | current_price_only is fair; synthetic workload mix limits production similarity |

## 6. Outreach guidance per benchmark

> Columns: vs **F** = vs FIFO claims · vs **M** = vs modern-scheduler claims
> · vs **H** = vs hyperscaler-style-scheduler claims · **E** = enterprise
> outreach claims.

| benchmark | vs F | vs M | vs H | E |
|---|---|---|---|---|
| Azure LLM 2024 week | **YES** (+98%) | **YES** (+26% vs sla_aware, +3% vs queue_aware) | **NO** — no KV-aware/prefix-cache baseline | **YES with caveats** — pair with attribution disclosure |
| BurstGPT | MARGINAL (fifo wins by 2% in some configs) | **YES** (+1.77% vs cache_affinity, +26% vs sla_aware) | **NO** | MARGINAL — lead with LLM-serving median |
| Azure LLM 2023 conv | **NO** (fifo beats CA) | **YES** (+19.86% vs sla_aware) | NO | WEAK — only as part of LLM median |
| Alibaba GenAI 2026 | **YES** (+457%) | **PARTIALLY** — none have affinity; use +44% vs utilization_aware not +89% vs sla_aware | **NO** — under-baselined | **USE WITH STRONG DISCLOSURE** about missing affinity baseline |
| Alibaba GPU v2023 | YES (+56%) | **YES** (+6% vs best_fit) | PARTIALLY (Borg-style would erase most) | **YES** — heterogeneous-fleet price story |
| Philly | YES (+62%) | **NO** (TIE) | NO | "matches strongest safe packing baseline — already on frontier" |
| MIT Supercloud bounded | YES + SAFETY | **NO** (TIE) | NO | "safe-frontier-tied + safety win vs naive FIFO" |
| Canonical energy | YES (+103%) | **YES** (+11%) | PARTIALLY | **YES** — SAFETY_WIN with alpha |

## 7. Final question — answer

> **If a sophisticated Azure / OpenAI / Anthropic-style inference stack
> already uses dynamic batching, queue-aware scheduling, and autoscaling,
> how much of Aurelius' reported improvement remains after comparing
> against the strongest implemented realistic baseline?**

### Short answer

**Roughly half to two-thirds of the headline survives — but a hyperscaler-
equivalent baseline is NOT implemented in any current Aurelius benchmark.**
The honest range against the strongest *implemented* modern baseline is
**+10% to +25% on LLM-serving traces**; against a hypothetical real
hyperscaler stack the conservative residual is **+3% to +15% on LLM
serving**, **+0% to +6% on GPU packing**, **+0% (tie) on training**,
**+3% to +11% on energy**.

### What already holds

- **Continuous batching** is in the shared serving physics — every policy
  gets it for free. The CA-vs-sla_aware comparisons are
  autoscaler-vs-autoscaler, not "with continuous batching vs without."
- **Queue-aware scheduling** exists as the `queue_aware` baseline and as
  an input to `constraint_aware`'s `_constraint_trim`. CA wins **only
  +2.6%** vs `queue_aware` on Azure 2024 — that is the conservative floor.
- **Autoscaling** exists as `sla_aware` (rho=0.50), `queue_aware`
  (queue-target), `utilization_aware` (rho=0.85). CA's +26% vs `sla_aware`
  on Azure 2024 is the headline; CA's +3% vs `queue_aware` is the floor.

### What does NOT hold against a real hyperscaler stack

- **KV-aware admission control**: no baseline implements per-request
  KV-cache-pressure-aware admission. vLLM PagedAttention, TensorRT-LLM,
  custom Anthropic/OpenAI inference stacks do.
- **Prefix caching as a scheduler input**: `cache_affinity_baseline` uses
  a static prior (`MAX_PREFILL_SAVINGS=0.25 × reuse_fraction`), not a
  real per-request prefix-cache router. vLLM / SGLang / Mooncake have
  real prefix-cache-aware admission.
- **Model-affinity routing**: ONLY `constraint_aware` on GenAI has
  `affinity=True`. Real multi-model serving (Replicate, RunPod, Hugging
  Face Endpoints, Anthropic batch model multiplexing) has this as a
  primary feature. On GenAI this is the +62% Shapley share.
- **Speculative decoding, prompt-aware batching, chunked prefill,
  disaggregated prefill/decode (DistServe, Sarathi-Serve, Mooncake)**:
  NOT modelled. Aurelius's serving physics is a single shared
  continuous-batching model.
- **Custom hyperscaler autoscalers** calibrated to per-tenant SLOs:
  the implemented `sla_aware` / `queue_aware` are generic HPA-style.

### Estimated residual alpha against a hypothetical hyperscaler baseline (directional only)

| benchmark | headline vs strongest implemented | conservative residual vs hyperscaler-equivalent | rationale |
|---|---:|---:|---|
| Azure LLM 2024 week | +25.75% (vs `sla_aware`) | **+5–15%** | +2.6% vs `queue_aware` is the floor; +5–15% accounts for the static frontier controller's +12.98% being available to CA via opt-in (which a hyperscaler stack would also adopt) |
| BurstGPT | +1.77% (vs `cache_affinity`) | **+0–3%** | a real KV-cache-aware vLLM stack would erase most of the static cache-prior baseline gap |
| Azure LLM 2023 conv | +19.86% (vs `sla_aware`) | **+0–10%** | static fifo already beats CA on this trace; adaptive static-vs-reactive switching would likely match |
| Alibaba GenAI 2026 | +89.46% (vs `sla_aware`) | **+20–40%** | sizing Shapley share is 38% of 89% ≈ +34%; a real SD serving stack has model-affinity routing |
| Alibaba GPU v2023 | +6.0% (vs `best_fit`) | **+0–6%** | Borg-style heterogeneous bin-packing would erase most; CoreWeave-style multi-cloud price arbitrage would NOT erase the price-aware lever |
| Philly | 0% (TIE) | **+0%** (TIE) | already TIES standard packing baselines |
| MIT Supercloud | 0% (TIE) | **+0%** (TIE) | same |
| Canonical energy | +11.07% (vs `current_price_only`) | **+3–11%** | a sophisticated commercial DR product would be between `current_price_only` and `greedy_energy + safety reverts`; residual depends on commercial product's warmup-awareness |

### Honest summary for outreach

> *"On public-trace simulator benchmarks, Aurelius improves SLA-safe
> goodput/$ by a median of ~9% vs the strongest implemented modern
> baselines (reactive HPA-style autoscalers, packing baselines). None of
> the implemented baselines model a hyperscaler-grade inference stack
> (KV-aware admission, real prefix caching, model-affinity routing,
> speculative decoding) — against such a stack, the conservative
> estimated residual is roughly +3–15% on LLM-serving and +3–11% on
> energy / flexible workload, with TIEs on training/packing. Real
> production-savings number requires a customer shadow-pilot
> (`docs/RESULTS.md` §8)."*

### What to build next to close the gap

1. Implement a **vLLM-style baseline** with PagedAttention KV-aware
   admission + real prefix-cache routing — would tighten BurstGPT and
   Azure baselines.
2. Implement a **model-affinity baseline for GenAI** (`sla_aware_with_affinity`)
   — would tighten the +89% headline to ~+34%.
3. Implement a **Borg-style heterogeneous-fleet bin-packing baseline** —
   would tighten the +6% Alibaba GPU headline.
4. Run on a **customer-telemetry shadow-pilot** — the only way to satisfy
   the `docs/RESULTS.md` §8 production-claim gate.

---

**Artifacts:**

- Matrix JSON: `data/external/benchmark_rollup/baseline_capability_matrix.json`
- Tests: `tests/test_baseline_capability_matrix.py`
- Companion: `docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md`
