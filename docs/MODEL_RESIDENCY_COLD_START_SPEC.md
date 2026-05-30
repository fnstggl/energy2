# Model Residency / Cold-Start Intelligence — Specification

> **Status:** Specification + telemetry contract only. **This document changes
> no optimizer behavior, no simulator constant, and no robust energy engine.**
> It defines *what Aurelius should track, score, and eventually optimize* for
> model residency / cold-start risk. Implementation is a later, separately-gated
> phase.
>
> **Simulator/benchmark results referenced here are directional only — not
> production savings** (`docs/RESULTS.md` §8). No production-savings number may
> be quoted until the `docs/RESULTS.md` §8 production-claim gate is satisfied
> against real customer telemetry.

Read first: `docs/RESULTS.md` (reporting standard), `docs/PUBLIC_TRACE_BACKTESTS.md`
(dataset roles), `docs/ALIBABA_GENAI_BACKTEST_RESULTS.md` (the GenAI serving
backtest), `docs/ALIBABA_GENAI_ABLATION_RESULTS.md` (the attribution this spec is
grounded in). Telemetry field-level detail lives in
`docs/PILOT_TELEMETRY_CONTRACT.md`.

---

## 0. Why this spec exists (grounding)

The Alibaba GenAI 2026 (GenTD26) multi-layer serving backtest showed
`constraint_aware` at **+89.5% SLA-safe goodput per infrastructure dollar** vs
the `sla_aware` headline. The ablation
(`docs/ALIBABA_GENAI_ABLATION_RESULTS.md`) decomposed that gain (Shapley over the
2×2 sizing×affinity corners):

| component | share of the +89.5% gain |
|---|---|
| **model-affinity / prewarm** | **~62%** |
| anticipatory SLA sizing | ~38% |
| interaction | ~0% (near-additive) |

The affinity lever cut **mean cold-start ~23.6 s → ~2.9 s** and lifted *every*
sizing strategy consistently (+33–80%). The cold-start magnitude itself is a
real, measured quantity from the trace's pipeline layer (median base-model load
**~22.7 s**, LoRA load **~4.4 s**, ControlNet load **~3.9 s**), across **~80
distinct base models**.

**Conclusion that motivates this spec:** model residency, warm pools, and
cold-start avoidance are **first-class economic levers**, not micro-optimizations.
Aurelius therefore needs a formal way to *observe* residency/cold-start in real
deployments before it may *act* on it. This document is that contract.

> Caveat carried from the ablation: in the *current simulator*, "prewarm" and
> "model-affinity" are modelled by the **same** cold-start-amortization mechanism.
> This spec deliberately treats them as **distinct real-world actions** (see §1,
> §4) because in production they are separable (prewarm = proactively load before
> demand; affinity = route demand to an already-warm replica). The spec must not
> be read as claiming the simulator separated them.

---

## 1. Core concepts (normative definitions)

| Term | Definition |
|---|---|
| **Model residency** | A base model's weights are resident in a replica's GPU memory and ready to serve **without** a load. A request whose target model is resident is a *residency hit*. |
| **Adapter / LoRA residency** | A LoRA (or other adapter) for the requested `(base_model, adapter_id)` is resident and merged/attachable without a load. Independent of base-model residency. |
| **Warm pool** | The set of replicas currently holding a given `(model_id[, adapter_id])` resident, deliberately kept warm to absorb demand. A warm pool has a **carrying cost** (GPU-hours of reserved/held capacity). |
| **Cold start** | Serving a request that requires loading its base model (and/or adapter) into GPU memory before inference can begin. The request pays **model load latency** (+ **adapter load latency**) on top of inference time. |
| **Model load latency** | Wall-clock time to load a base model's weights from storage into GPU memory and make it serveable (the dominant cold-start component; ~22.7 s median in GenTD26). |
| **Adapter load latency** | Wall-clock time to load/attach an adapter (LoRA/ControlNet) to an already-resident base model (~4 s median in GenTD26). |
| **Cache affinity** | The property that routing a request to a replica that recently served the same `(model_id[, adapter_id]` / prefix) avoids a (re)load and/or reuses KV/prefix cache. The *affinity action* is the routing decision that preserves it. |
| **Prewarm action** | A **proactive** load of a model/adapter into a warm pool *before* the predicted demand arrives, so the first matching request is a residency hit. (Recommendation-only in pilot.) |
| **Preserve-affinity action** | A routing/placement decision that keeps a request (or future requests of a tenant/model) on an already-warm replica instead of moving it — trading routing/energy flexibility for cold-start/cache avoidance. (Recommendation-only in pilot.) |
| **Cold-start risk** | The expected cold-start penalty for a (model/adapter, time-window) = `P(request needs a load) × E[load latency]`, in latency-seconds and in SLA-violation probability. The quantity the prewarm/affinity decision rules are scored against. |

Boundary rule (binding): **none** of these actions may ever change *which model
or adapter the user requested*. Residency/affinity acts on *where/when* a model
is loaded and a request is routed — never on *substituting* the served model
(see §4 and §5).

---

## 2. Required telemetry for pilots

Full field-level schema, types, units, and null-handling are in
`docs/PILOT_TELEMETRY_CONTRACT.md`. Summary of the **required** per-request and
per-sample fields:

**Per request (application + middleware join):**
`request_id`, `timestamp`, `tenant_id`/`workload_id`, `model_id`,
`adapter_id`/`lora_id` (if present), `endpoint_id`, `region`, `node_id`,
`gpu_id`, `container_id`, `model_loaded_before_request` (bool),
`adapter_loaded_before_request` (bool), `model_load_start`/`model_load_end`
(timestamps, null if no load), `adapter_load_start`/`adapter_load_end`,
`queue_wait`, `TTFT`, `TPOT`, `e2e_latency`, `status`/`error`.

**Per infrastructure sample (container/GPU time-series):**
`timestamp`, `node_id`, `gpu_id`, `container_id`, `gpu_utilization`,
`gpu_memory_used`, `gpu_memory_total`, `power` (if available),
`cache_hit`/`prefix_hit` (if available).

**Cross-layer key requirement (binding honesty rule):** residency/cold-start
attribution **requires** a real join key between the request and the
container/GPU that served it (e.g. `container_id` + `gpu_id` present on the
request, OR a request→placement event). Where that key is **absent** (as in the
public GenTD26 trace, whose application and infrastructure layers are `no_join`),
the pilot **MUST** mark residency metrics as *calibration-only / unattributed*
and **MUST NOT** claim per-request request→GPU causality. See
`docs/PUBLIC_TRACE_BACKTESTS.md` §3e and the GenAI backtest doc for the precedent.

---

## 3. Derived metrics

Computed from §2 telemetry. Each MUST be reported with its denominator and the
window over which it is computed.

| Metric | Definition |
|---|---|
| **Model residency hit rate** | `requests with model_loaded_before_request=true / total requests`, per model / tenant / endpoint / window. |
| **Adapter residency hit rate** | `requests with adapter_loaded_before_request=true / (requests using an adapter)`. |
| **Cold-start rate** | `requests that triggered a model and/or adapter load / total requests`. |
| **Cold-start latency p50/p95/p99** | percentiles of `(model_load + adapter_load)` latency over cold-start requests. |
| **Warm-pool cost** | GPU-hours (× price) of capacity held resident-but-idle to keep models warm, per model / window. The denominator side of every prewarm decision. |
| **Cold-start avoided latency** | counterfactual: `cold_start_rate_without_prewarm − cold_start_rate_with_prewarm` × `E[load latency]`, in latency-seconds; reported as a **range** when the counterfactual is estimated, not measured. |
| **SLA violations attributable to cold start** | SLA-violating requests whose `e2e_latency − load_latency ≤ SLO` (i.e. would have met SLO without the cold start), per `docs/RESULTS.md` SLA-filter conventions. |
| **goodput/$ with and without prewarm** | the canonical KPI (`docs/RESULTS.md` §1) computed for the prewarm-on vs prewarm-off counterfactual, reported separately (never blended). |
| **Model popularity half-life** | time for a model's request share to decay to half its peak — drives evict timing (§4). |
| **Residency churn score** | rate of load/evict events per model per window; high churn = thrash, a safety/cost diagnostic (analogous to migration churn in `docs/RESULTS.md` §2). |

All of the above are **diagnostics / inputs to decision rules** — they are
**never** folded into the primary KPI as weighted terms (`docs/RESULTS.md` §1, §2).

---

## 4. Decision rules (SPEC ONLY — not implemented here)

These define *intended* behavior for a future, separately-gated implementation.
**No optimizer code in this PR implements them.** Each is recommendation-only in
pilot (§5).

1. **Prewarm rule.** Recommend prewarming `(model_id[, adapter_id])` into a warm
   pool when the **expected cold-start penalty × request probability** over the
   lookahead window **exceeds the warm-pool carrying cost**:
   `P(request for model in window) × E[load_latency_penalty_$] > warm_pool_cost_$`.
   Penalty is measured in SLA-safe-goodput-dollars lost to cold-start, not raw
   latency.
2. **Preserve-affinity rule.** Recommend keeping a request/tenant on an
   already-warm replica when the **cold-start / cache penalty of moving it
   exceeds the energy/routing savings** of moving it
   (`cold_start_or_cache_penalty_$ > energy_or_routing_savings_$`). This is the
   residency analogue of the existing constraint-aware "don't migrate into a
   cold/expensive destination" gate — stated for models, not jobs.
3. **Evict rule.** Recommend evicting a model from a warm pool when its
   **popularity decays below the warm-pool cost threshold** (popularity
   half-life from §3 makes the expected future hit value `< warm_pool_cost_$`).
   Eviction MUST respect an anti-thrash cooldown to bound residency churn.
4. **No-substitution rule (binding safety gate).** A residency/affinity decision
   **MUST NEVER** serve a different model or adapter than the user requested,
   degrade output quality, or silently reroute to a "close enough" model. If the
   requested model cannot be served within SLO, that is a capacity/SLA event to
   surface — **never** a silent substitution. This rule is non-negotiable and
   overrides rules 1–3.

Scoring objective (binding): every rule is scored against **SLA-safe goodput per
infrastructure dollar** (`docs/RESULTS.md` §1), with cold-start avoided latency,
warm-pool cost, and churn as the §2/§3 diagnostics — **not** as hidden weighted
terms in the primary KPI.

---

## 5. Pilot / shadow-mode requirements

1. **Recommendation-only by default.** All prewarm / preserve-affinity / evict
   decisions are **logged recommendations**; the pilot performs **no real
   cluster mutation** (no real model load/evict, no real reroute) unless the
   operator explicitly promotes a recommendation out of shadow mode.
2. **No production cluster mutation in shadow mode.** Consistent with the
   existing `recommendation_only` shadow-pilot posture referenced in
   `docs/RESULTS.md` §8.
3. **Log every recommended decision** with: inputs (the §3 metrics it used), the
   rule that fired (§4), the predicted penalty/cost, and a decision id.
4. **Counterfactual comparison when possible.** When the deployment later
   reveals the observed outcome (a matching request did/did not arrive; a load
   did/did not occur), compare the recommendation to the observed counterfactual
   and record hit/miss. When no counterfactual is observable, mark the
   recommendation **unverified** — do not assume it was correct.
5. **No production-savings claims without real telemetry calibration.** Any
   residency/cold-start economic number stays **directional / simulator-only**
   until the `docs/RESULTS.md` §8 production-claim gate is met (real customer
   telemetry feed, calibrated priors, customer cost basis, ≥1 clean shadow-pilot
   cycle with no SLA regression). Allowed wording is `docs/RESULTS.md` §8.

---

## 6. Integration points (observation surfaces)

How the §2 telemetry is expected to be sourced. **No connector code is added in
this PR**; this lists the intended surfaces so the contract is implementable.

| Surface | Residency / cold-start signal it exposes |
|---|---|
| **vLLM** | model load events, LoRA load/attach, prefix-cache hit rate, running/waiting queue, KV-cache usage, per-request TTFT/TPOT. |
| **Triton (TensorRT-LLM / Python backends)** | model (un)load events, model-control API state (which models loaded), instance/GPU placement, per-model inference stats. |
| **SGLang** | RadixAttention prefix-cache hit rate, model/adapter residency, scheduler queue + running batch. |
| **Ray Serve** | replica/deployment autoscaling + warm-replica state, per-deployment model identity, request routing. |
| **Kubernetes / DCGM / Prometheus** | pod→node→GPU placement (`container_id`/`gpu_id`/`node_id`), GPU utilization + memory (DCGM `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED/FREE`), power (`DCGM_FI_DEV_POWER_USAGE`), scraped via Prometheus. This is the surface that supplies the **cross-layer join key** the public GenTD26 trace lacked. |

---

## 7. Benchmark standard (how future benchmarks must report this)

Binding additions to the `docs/RESULTS.md` reporting discipline for any future
benchmark or pilot that touches residency / cold-start:

1. **Separate attribution is mandatory.** A residency/cold-start benchmark MUST
   report the **model-affinity/prewarm contribution separately** from
   queue-awareness, utilization-awareness, and energy contributions — using the
   same factorial / Shapley method as `docs/ALIBABA_GENAI_ABLATION_RESULTS.md`
   (vary the affinity knob orthogonally to the sizing/energy knobs). A single
   blended "constraint_aware won" number is **not** acceptable for this lever.
2. **Prewarm vs affinity must be labelled honestly.** If a benchmark's mechanism
   does not separate proactive prewarm from reactive affinity, it MUST say so
   (as the ablation did) and report them as one combined lever — never imply two
   independent wins from one mechanism.
3. **Counterfactual framing.** "Cold-start avoided" and "goodput/$ with vs
   without prewarm" MUST be reported as explicit counterfactual pairs, never as a
   standalone savings number.
4. **Primary KPI unchanged.** The headline remains SLA-safe goodput per
   infrastructure dollar (`docs/RESULTS.md` §1); residency metrics (§3) are
   diagnostics, never folded into it.
5. **Claim gate.** §8 of `docs/RESULTS.md` governs; no production-savings claim
   until the gate is met.

### Suggested `docs/RESULTS.md` cross-reference (non-binding)

`docs/RESULTS.md` MAY add, when residency optimization is implemented, a one-line
pointer under §2 (secondary KPIs) listing `model residency hit rate`,
`cold-start rate`, and `warm-pool cost` as residency diagnostics, and under §6
(alpha vs safety) noting that **cold-start avoidance is an alpha lever** (cheaper
SLA-safe serving) while **preserve-affinity under uncertain demand is a safety
lever** (avoiding cold-cache/thrash) — mirroring the existing migration
alpha/safety split. This spec does **not** edit `docs/RESULTS.md`; it records the
intended reference so a future, deliberate revision can adopt it.

---

## 8. Non-goals (this document)

- **No new optimizer behavior** is implemented; §4 rules are specification only.
- **No forecasting model** is trained (popularity half-life is a *defined
  metric*, not a learned model here).
- **No new public dataset** is ingested.
- **No existing gate is weakened**; the no-substitution rule (§4.4) *adds* a
  safety constraint.
- **No robust-energy-engine or simulator-constant change.**
- **No production-savings claim.** Directional / simulator-only until the
  `docs/RESULTS.md` §8 gate is met.
