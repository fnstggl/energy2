# Model Residency / Cold-Start — Readiness Audit

> **Measurement + gap-analysis only.** This audit adds **no optimizer behavior,
> no simulator constant, no robust energy engine change, and no new dataset.**
> It answers: *how much model-residency / cold-start intelligence does Aurelius
> actually have today, how much of the public-trace alpha depends on it, and
> what is missing before pilot-ready shadow mode?*
>
> **Directional simulator/trace result — NOT production savings**
> (`docs/RESULTS.md` §8). Machine-readable companion:
> `data/external/alibaba_genai/processed/model_residency_audit_summary.json`.

Read first: `docs/RESULTS.md`, `docs/MODEL_RESIDENCY_COLD_START_SPEC.md`,
`docs/PILOT_TELEMETRY_CONTRACT.md`, `docs/ALIBABA_GENAI_BACKTEST_RESULTS.md`,
`docs/ALIBABA_GENAI_ABLATION_RESULTS.md`.

---

## 0. Headline

| Question | Answer |
|---|---|
| How much residency/cold-start intelligence exists? | **Cache (prefix) affinity:** partial *read-only* real path (vLLM `prefix_cache_hit_rate` → `PRESERVE_AFFINITY`/`PREWARM_REPLICA` candidate actions). **Model/adapter residency + cold-start latency:** simulated / trace-calibrated only — **no real model-load telemetry.** |
| How much public-trace alpha depends on it? | On Alibaba GenAI 2026, **~62% of the +89.5% goodput/$ win** is the model-affinity/prewarm lever (Shapley); with-affinity goodput/$ **9.84** vs without **7.05**; mean cold-start **2.9 s vs 23.6 s** (modelled). |
| Readiness verdict | **`TRACE_BACKTESTED_APPROXIMATION`** — not production-ready, not yet shadow-pilot-ready for *model* residency. |

---

## 1. Implementation inventory

Classification key: **REAL** (acts on/reads a real cluster/connector) · **SIM**
(cluster-simulator physics only) · **TRACE** (public-trace backtest only) ·
**BENCH** (benchmark/heuristic cost approximation) · **SPEC** (docs only) ·
**MISSING**.

### 1a. Real connector / state (what is actually read from live systems)

| Capability | Path | Class | Note |
|---|---|---|---|
| Prefix-cache hit rate | `aurelius/connectors/vllm.py:155` → `state/models.py::InferenceServiceState.prefix_cache_hit_rate` | **REAL** | Read from `vllm:gpu_prefix_cache_hit_rate`. This is the one real residency-adjacent signal. |
| KV-cache usage | `vllm.py` → `InferenceServiceState.kv_cache_usage` | **REAL** | From `vllm:gpu_cache_usage_perc`. |
| TTFT / TPOT / e2e latency | `vllm.py`, DCGM/K8s | **REAL** | Cannot decompose cold-start TTFT from warm TTFT. |
| `model_loaded_before_request` / model-load timestamps | — | **MISSING** | Not exposed by vLLM/Triton/K8s/DCGM connectors. |
| `adapter_id`/`lora_id` residency | — | **MISSING** | No adapter tracking in connectors or state. |

### 1b. Constraint engine (recommendation-only actions)

| Capability | Path | Class | Note |
|---|---|---|---|
| PRESERVE_AFFINITY decision | `aurelius/constraints/engine.py:264–282` | **REAL (rec-only)** | Gated on real `prefix_cache_hit_rate ≥ 0.70` (`_PREFIX_AFFINITY_PRESERVE_HIT_RATE`); blocks a cross-region move that would cold-route. Recommendation-only. |
| PREWARM_REPLICA candidate | `engine.py:367,393` | **REAL (rec-only)** | Offers a ready replica for critical-interactive workloads; gated by SLA + cost model. **No separate prewarm constant — prewarm ≡ affinity routing.** |
| Cold-start / cache-warmup cost terms | `aurelius/constraints/cost_model.py:94,205,602` | **BENCH** | `cold_start_p99_penalty_ms`, `cache_warmup_penalty_ms` are **heuristic** constants; fed by optional inputs (`prefix_cache_hit_rate=None` unless provided). |

### 1c. Cluster simulator physics

| Capability | Path | Class |
|---|---|---|
| Decomposed cold-start (T_node/pull/load/gpu/warmup, heavy-tailed, first-compile) | `aurelius/simulation/cluster/migration.py::cold_start_seconds` (183–215) | **SIM** |
| Prefix hit rate / prefill savings / cold-route penalty | `aurelius/simulation/cluster/kv_cache.py::prefix_hit_rate`, `prefill_savings_frac`, `cold_route_penalty_ms` | **SIM** |
| Warm-pool / replica-warmup / cold-start state | `migration_model.py::WarmPoolState`, `ReplicaWarmupState`, `ColdStartState`; `cache_model.py::CacheAffinityState` | **SIM** |
| Calibration constants (engine startup profiles, warm-pool idle power, prefix sigmoid) | `aurelius/simulation/cluster/calibration.py` (`MIGRATION_PARAMS`, `KV_CACHE_PARAMS`) | **SIM** (heuristic/inferred, not measured) |

### 1d. Public-trace backtests

| Capability | Path | Class |
|---|---|---|
| GenAI affinity cold-start model (`switch_rate × basemodel/LoRA/ControlNet load`) | `aurelius/traces/genai_backtest.py::_effective_service_s` (100–117); `affinity` flag set only for `constraint_aware` (194) | **TRACE** |
| GenAI Shapley attribution (affinity vs sizing) | `aurelius/traces/genai_ablation.py::attribute` | **TRACE** |
| BurstGPT cache-affinity proxy (`reuse_fraction` → prefill savings) | `traces/backtest.py` (MAX_PREFILL_SAVINGS=0.25), `traces/replay.py::reuse_fraction`, `traces/burstgpt.py::_cache_affinity_key` | **TRACE** (model-level proxy, not a KV hit rate) |

### 1e. Spec

| Capability | Path | Class |
|---|---|---|
| Residency spec + telemetry contract | `docs/MODEL_RESIDENCY_COLD_START_SPEC.md`, `docs/PILOT_TELEMETRY_CONTRACT.md`, `tests/test_model_residency_spec.py` | **SPEC** |

**Inventory verdict:** the only *real* residency-adjacent signal is the vLLM
**prefix-cache hit rate** (cache affinity), which already drives recommendation-
only `PRESERVE_AFFINITY`/`PREWARM_REPLICA` candidates. **Model/adapter residency
and cold-start latency are entirely simulated or trace-calibrated.** No code
reads `model_loaded_before_request` or a model-load timestamp from any live
engine.

---

## 2. Benchmark / ablation table (how much alpha depends on affinity/prewarm)

Source: committed `alibaba_genai_*_summary.json` (full trace, 26,392 requests).
goodput_unit = `completed_requests`. Cold-start values are **modelled**, not
per-request measured.

| config (GenAI 2026) | goodput/$ | SLA-compliant | e2e p99 (s) | mean cold-start (s) | replica GPU-hrs |
|---|---|---|---|---|---|
| **constraint_aware (with affinity/prewarm)** | **9.84** | 26,392 | 53.4 | **2.9** | 894 |
| constraint_aware (without affinity) | 7.05 | 26,392 | 66.4 | 23.6 | 1,247 |
| sla_aware (headline, no affinity) | 5.19 | 17,794 | 1,219 | 23.6 | 1,142 |
| fifo (no affinity) | 1.77 | 26,392 | 53.5 | 23.6 | 4,977 |
| fifo + affinity | 3.18 | 26,391 | 35.9 | 2.9 | 2,765 |

- **CA vs sla_aware gain:** **+89.5%**; Shapley share **affinity/prewarm ≈ 62%**,
  anticipatory sizing ≈ 38%, interaction ≈ 0%.
- **Cold-start avoided (modelled):** ~**20.7 s** mean per request (23.6 → 2.9 s).
- **Warm-pool GPU-hour cost:** **not separately metered** — affinity *reduces*
  required replica-hours (894 vs 1,247) rather than charging an explicit
  warm-pool line item. A real warm-pool cost requires the §5 next-fixes.
- **Cold-start p50/p95/p99:** **not per-request measured.** Only a calibrated
  pipeline-layer distribution exists (basemodel load p50 ≈ 22.7 s, LoRA ≈ 4.4 s,
  ControlNet ≈ 3.9 s).
- **Residency hit-rate:** **not measured** — the model uses
  `switch_rate ≈ distinct_models/N` as a proxy (GenTD26 app↔infra is `no_join`).

| other datasets | residency/affinity measurability |
|---|---|
| **BurstGPT** | cache-affinity **proxy only** (model-level `cache_affinity_key`, no session id); `cache_affinity_baseline` vs `fifo` ≈ **+0.2%** goodput/$ — negligible. Not a measured KV hit rate. |
| **Azure LLM** | **NOT APPLICABLE** — no model/session/cache fields; `reuse_fraction = 0` (no invented benefit). |
| **Canonical energy adapter** | has a *job-migration* cold-start/warmup gate (`migration_cost_hours`), relevant to cold-start-on-migration for **jobs**, **not** model residency. |

---

## 3. Spec conformance matrix

Against `docs/MODEL_RESIDENCY_COLD_START_SPEC.md`. Status ∈
{implemented, partial, missing, n/a}.

| requirement | status | evidence (file/function/test) | public-trace evidence | pilot telemetry required | next fix |
|---|---|---|---|---|---|
| `model_id` tracking | **partial** | `state/models.py::InferenceServiceState.service_id` | GenAI `service_id`, BurstGPT model | per-request `model_id` | map service_id→model_id; emit per request |
| `adapter_id`/`lora_id` tracking | **missing** | — | GenAI `num_lora` (count only) | `adapter_id` field | add adapter id to state + connector |
| `model_loaded_before_request` | **missing** | — (simulated only) | none (no_join) | required bool | instrument serving engine load events |
| `adapter_loaded_before_request` | **missing** | — | none | required bool | same as above for adapters |
| model load start/end timestamps | **missing** | sim `migration.cold_start_seconds` only | GenAI pipeline-layer *distribution* | load_start/end ts | engine-level load hooks |
| adapter load start/end timestamps | **missing** | sim only | GenAI `lora_update_latency` *distribution* | load_start/end ts | engine-level adapter hooks |
| residency hit rate | **missing (measured)** / partial (proxy) | trace `reuse_fraction`, GenAI `switch_rate` proxy | proxy only | hit/miss per request | derive from real load events |
| cold-start rate | **missing (measured)** | trace/sim only | calibrated, not measured | load-event count | derive from real load events |
| cold-start p95/p99 | **missing (measured)** | `cost_model.cold_start_p99_penalty_ms` (heuristic) | pipeline-layer dist (p50/p95) | per-request load latency | measure load latency |
| warm-pool cost | **partial** | sim `WarmPoolState`, `warm_pool_idle_power_frac` | implicit (replica-hours) | held GPU-hours | meter held-resident idle GPU-hours |
| no-substitution safety | **partial** | implicit — no action substitutes a model (actions are region/scale/affinity) | n/a | n/a | add explicit gate + test asserting it |
| preserve-affinity decision | **implemented (rec-only)** | `constraints/engine.py:264–282` (gated on real `prefix_cache_hit_rate`) | — | hit rate (have it) | extend from prefix → model/adapter residency |
| prewarm recommendation | **implemented (rec-only)** | `constraints/engine.py:367,393` (PREWARM_REPLICA) | — | demand signal | add residency-aware prewarm trigger |
| shadow-mode logging | **partial** | `aurelius/shadow/` (energy/scheduling DecisionRecords, rec-only) | — | decision log | add residency decision + counterfactual log |

---

## 4. Dataset coverage table

| dataset / source | model_id | adapter/LoRA | real e2e | cold-start latency | per-request residency hit | measure vs simulate |
|---|---|---|---|---|---|---|
| **Alibaba GenAI 2026** | yes | num_lora (count) | yes (`exec_time_seconds`) | **calibrated** distribution (pipeline layer) | **no** (app↔infra `no_join`) | cold-start *calibrated* + affinity *simulated*; residency **not measured** |
| **BurstGPT** | yes | no | elapsed (not TTFT) | no | model-level **proxy** only | affinity **proxy**, ~+0.2% effect |
| **Azure LLM** | no | no | no | no | no | **NOT APPLICABLE** |
| **Live vLLM connector** | service_id | no | yes (TTFT/TPOT/e2e) | **no** (not in `/metrics`) | **no** | `prefix_cache_hit_rate`+`kv_cache_usage` REAL; residency/cold-start **not** |

**Bottom line:** *no available dataset or live connector can currently measure
per-request model-residency hit/miss or cold-start latency.* GenTD26 can
**calibrate** the cold-start magnitude (it is real, ~22.7 s base-model load), and
the simulator/trace can **model** the affinity benefit — but neither
**measures** residency on a real serving path.

---

## 5. Readiness verdict

### `TRACE_BACKTESTED_APPROXIMATION` — not production-ready.

Justification (conservative):
- The affinity/prewarm economic result (+62% of the GenAI win) is real **as a
  trace-calibrated backtest**, grounded in measured cold-start magnitudes — above
  `SIMULATOR_APPROXIMATION`.
- But it is **below `SHADOW_PILOT_READY_READ_ONLY`** for *model* residency: there
  is no live `model_loaded_before_request` / model-load-latency signal, no
  per-request residency hit-rate, no metered warm-pool cost, and no residency
  decision log to compare against a counterfactual. The one real lever
  (prefix-cache affinity preservation/prewarm) is a *different, narrower*
  capability than model/adapter residency.

### What is needed to reach `SHADOW_PILOT_READY_READ_ONLY`

1. **Real load telemetry.** Instrument (or scrape) a serving engine
   (vLLM/Triton/SGLang) for **model + adapter load start/end events**, and emit
   `model_loaded_before_request` / `adapter_loaded_before_request` per request —
   the `docs/PILOT_TELEMETRY_CONTRACT.md` §2 fields that are currently absent.
2. **Cross-layer join key.** Achieve at least `container_join` between the
   request stream and the GPU/container metrics (request carries
   `container_id`+`gpu_id`), so residency can be **attributed**, not proxied.
3. **Measured derived metrics.** Compute residency hit rate, cold-start rate, and
   cold-start p50/p95/p99 from the real load events (replace the
   `switch_rate`/`reuse_fraction` proxies).
4. **Explicit warm-pool cost meter.** Charge held-resident idle GPU-hours as a
   line item (today it is implicit in replica-hours).
5. **Explicit no-substitution gate + test.** Assert (and test) that no
   residency/affinity action ever changes the requested model/adapter — today
   this holds by construction but is not enforced/asserted.
6. **Residency shadow log + counterfactual.** Extend the existing
   `aurelius/shadow/` recommendation-only runner to log
   prewarm/preserve-affinity *recommendations* with predicted penalty/cost and
   compare to the observed counterfactual — recommendation-only, no cluster
   mutation, per `docs/MODEL_RESIDENCY_COLD_START_SPEC.md` §5.

Only after 1–6, **and** the `docs/RESULTS.md` §8 production-claim gate (real
customer telemetry, calibrated priors, customer cost basis, ≥1 clean shadow
cycle), may any residency number move toward a production claim.

---

## 6. Precise next engineering tasks (ordered)

1. **(connector)** Add a model/adapter **load-event reader** (engine logs or a
   sidecar) → new optional fields on `InferenceServiceState`:
   `model_loaded_before_request`, `adapter_id`, `adapter_loaded_before_request`,
   `model_load_latency_s`. *Read-only; no optimizer change.*
2. **(telemetry)** Emit the `PILOT_TELEMETRY_CONTRACT.md` §2 request record with a
   real request↔container join key; add the §4 `linkage_quality` classifier to
   the live path.
3. **(metrics)** Add a residency-metrics computation (hit rate, cold-start rate,
   cold-start p50/p95/p99, warm-pool held GPU-hours) as **diagnostics** — never
   folded into the primary KPI.
4. **(safety)** Add an explicit `no_substitution` assertion + unit test on the
   action set.
5. **(shadow)** Add a residency recommendation log (prewarm / preserve-affinity)
   to `aurelius/shadow/` with counterfactual comparison, recommendation-only.
6. **(benchmark)** Keep reporting affinity/prewarm contribution **separately**
   (the existing `genai_ablation` Shapley method) per
   `docs/MODEL_RESIDENCY_COLD_START_SPEC.md` §7.

All six are observation/measurement/safety tasks; none requires new optimizer
*decision* logic beyond exposing metrics.

---

## 7. Claim discipline

- No production-savings claim is made here. All economic numbers are
  **directional trace-backtest** results (`docs/RESULTS.md` §8); the affinity
  lever is **modelled/calibrated, not measured on a live serving path**.
- "prewarm" and "model-affinity" are the **same** mechanism in the current
  implementation (warm-pool routing) — restated from the ablation, not a new
  claim.
- Production-claim gate (`docs/RESULTS.md` §8) is **not** met for model
  residency.
