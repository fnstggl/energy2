# Model Residency Decision Engine v1

> **Recommendation-only in real/customer mode. Simulator execution is for
> backtests only.** The engine **never** changes which model/adapter the user
> requested — it recommends only *placement / routing / prewarm / evict*. It
> mutates **no** real cluster, router, or serving engine, and calls **no**
> Kubernetes write API. Public-trace results are **directional simulator /
> backtest evidence — not production savings** (`docs/RESULTS.md` §8).

Read first: `docs/RESULTS.md` (primary KPI + claim rules),
`docs/MODEL_RESIDENCY_COLD_START_SPEC.md` (concepts, decision rules §4,
shadow posture §5), `docs/PILOT_TELEMETRY_CONTRACT.md` (telemetry),
`docs/MODEL_RESIDENCY_READINESS_AUDIT.md` (where this fits),
`docs/ALIBABA_GENAI_ABLATION_RESULTS.md` (the affinity/prewarm lever this
operationalises).

---

## 0. What this is

A first-class, testable decision layer that answers, per request:

> *given a request for model X (optionally adapter A), route it to the
> node/GPU/container where X (and A) is already resident — unless another
> constraint makes that unsafe — and recommend prewarm/evict when that improves
> SLA-safe goodput per infrastructure dollar.*

It operationalises the **model-affinity / prewarm lever** that the GenAI 2026
ablation attributed **~62%** of the +89.5% goodput/$ gain to
(`docs/ALIBABA_GENAI_ABLATION_RESULTS.md`), but as **explicit per-request
routing** rather than a tick-level cold-start amortization.

**Code:**
- `aurelius/residency/models.py` — `ModelResidencyRequest`, `ModelLocationState`,
  `ModelLoadProfile`, `ResidencyDecision`, `ResidencyAction`.
- `aurelius/residency/decision.py` — `score_residency_candidate`,
  `choose_residency_decision`, `SafetyContext`, `CandidateScore`.
- `aurelius/residency/sim.py` — simulator-only execution (`SIMULATOR_MODE` /
  `REAL_MODE`).
- `aurelius/residency/backtest.py` + `scripts/run_genai_residency_decision_backtest.py`
  — per-request GenAI backtest.

---

## 1. Decision actions

| action | meaning |
|---|---|
| `ROUTE_TO_RESIDENT_MODEL` | route to a replica where the requested model (and adapter) is resident — highest SLA-safe goodput/$. |
| `PRESERVE_AFFINITY` | the request's current warm replica is (within the KPI band) best; stay rather than churn to a cold/cheaper replica. |
| `PREWARM_MODEL` | no warm SLA-safe replica (or a cold replica is materially better and warm-pool economics justify it): load the base model at the target. |
| `PREWARM_ADAPTER` | base model resident at target but the adapter is not; load/attach the adapter. |
| `KEEP_CURRENT_ROUTE` | current route is already optimal; nothing to change. |
| `REJECT_UNSAFE_ROUTE` | no candidate can meet SLA / passes hard safety gates. Surface a capacity/SLA event — **never** silently substitute a model. |
| `EVICT_CANDIDATE` | memory pressure blocks every route; recommend evicting a low-value resident model to admit the requested one (advisory; anti-thrash cooldown). |
| `INSUFFICIENT_TELEMETRY` | telemetry confidence too low / required load profile missing; the engine does **not** guess. |

---

## 2. Scoring (`score_residency_candidate`)

For each candidate location the engine estimates:

```
expected_latency =
      queue_wait
    + model_load_penalty   (0 if model resident; else p95 for safety-critical,
                            else p50; UNKNOWN ⇒ candidate not safely scorable)
    + adapter_load_penalty (0 if adapter resident; else p95/p50 as above)
    + service_time_proxy   (from output tokens × seconds_per_token, else a
                            documented decode/serve proxy)

expected_cost =
      incremental_gpu_cost  (expected_latency × gpu_hour_price)
    + memory_pressure_cost  (surcharge as the GPU approaches capacity)
    [+ warm_pool_cost on a PREWARM; + eviction_cost on an EVICT]

sla_met               = expected_latency ≤ SLA(request)
goodput_per_dollar    = (1 if sla_met else 0) / expected_cost
```

`goodput_per_dollar` is the canonical **SLA-safe goodput per infrastructure
dollar** (`docs/RESULTS.md` §1) at single-request granularity. It is the
**objective**; the diagnostics (cold-start saved, queue/latency/cost deltas) are
reported, never folded in as weighted terms.

**Honest-data rules (binding):**
- A **resident** model/adapter has load penalty **0**.
- A **non-resident** model/adapter with an **unknown** load latency is **not
  treated as 0** — the candidate is marked not-safely-scorable (`missing_load_latency`).
- **Missing memory** telemetry never permits a route (it is not "unlimited").
- Missing telemetry lowers confidence and can force `INSUFFICIENT_TELEMETRY`.

---

## 3. Decision procedure (`choose_residency_decision`)

1. **Telemetry gate.** No locations, or none meeting `min_telemetry_confidence`
   → `INSUFFICIENT_TELEMETRY`.
2. **Score** every candidate; split into feasible (passes hard safety vetoes)
   and infeasible.
3. **No feasible candidate** → if a candidate is blocked *only* by memory and
   holds an evictable resident model → `EVICT_CANDIDATE`; else if required
   load/telemetry is missing → `INSUFFICIENT_TELEMETRY`; else `REJECT_UNSAFE_ROUTE`.
4. **SLA filter.** Keep SLA-met candidates. If none → `REJECT_UNSAFE_ROUTE`.
5. **Select best by goodput/$**, preferring affinity (resident / current route)
   only **within a small KPI tie band** (`kpi_tie_band`, default 1%) — so a cold
   node wins only when goodput/$ *materially* improves (spec: "do not route to a
   lower queue if it causes a larger cold-start penalty unless KPI improves").
6. **Classify:** best is warm & is current → `KEEP_CURRENT_ROUTE` /
   `PRESERVE_AFFINITY`; best is a different warm replica → `ROUTE_TO_RESIDENT_MODEL`;
   best is cold → `PREWARM_MODEL` / `PREWARM_ADAPTER` if (no warm SLA-safe
   replica) or (warm-pool economics justify it), else affinity fallback to the
   best warm replica.

---

## 4. Safety gates (vetoes, never KPI weights — Task 6)

`SafetyContext` carries the gates. A blocked candidate records the veto in
`ResidencyDecision.safety_vetoes`:

| gate | veto code | rule |
|---|---|---|
| GPU memory headroom | `insufficient_memory_headroom` | never route/prewarm a model that does not fit in free memory. |
| SLA latency | (drops `sla_met`) | a candidate whose expected latency exceeds the request SLA is not SLA-safe. |
| thermal risk | `thermal_risk` | veto a location above `max_thermal_risk`. |
| topology score | `low_topology_score` | veto below `min_topology_score` (when provided). |
| region | `region_not_allowed` | respect `allowed_regions` (request or context). |
| telemetry confidence | `low_telemetry_confidence` | a location below `min_telemetry_confidence` is untrusted. |
| queue ceiling | `queue_wait_exceeds_max` | optional hard queue cap. |
| no-substitution | — | **structural**: a residency hit is only credited for the *requested* model/adapter; the decision has no field to express a substitute. |

These gates **add** safety; they do not weaken any existing gate, and they are
never summed into the primary KPI (`docs/RESULTS.md` §1–§2).

---

## 5. Execution modes (Task 4)

| mode | behavior |
|---|---|
| **real / customer** (`sim.REAL_MODE`) | **recommendation-only**: `apply_residency_decision` is a strict no-op (`mutated=False`). `ResidencyDecision.executable_in_real_cluster` is `False` and the model refuses to be constructed with it `True`. No cluster / router / Kubernetes / serving-engine write. |
| **simulator** (`sim.SIMULATOR_MODE`) | for backtests only: mutates **simulated** `ModelLocationState` — `PREWARM_*` loads, `EVICT_CANDIDATE` evicts then admits, `ROUTE_*`/`KEEP`/`PRESERVE` place the request (cold route still pays the load). |

`tests/test_residency_decision.py` proves real mode never mutates and simulator
mode mutates deterministically.

---

## 6. GenAI backtest (Task 5)

`scripts/run_genai_residency_decision_backtest.py` replays the GenAI 2026 trace
one request at a time through a simulated GPU pool and compares
`fifo_round_robin`, `sla_aware_least_queue`, `sla_aware_naive_prewarm`,
`affinity_only`, and `residency_engine`. It reports residency hit-rate, adapter
hit-rate, cold-start p50/p95/p99, prewarm / route-to-resident / eviction counts,
warm-pool GPU-hours, SLA violations, and goodput/$. Results:
`docs/MODEL_RESIDENCY_DECISION_ENGINE_RESULTS.md`. The **existing tick-based
ablation is preserved unchanged** and referenced for the full-trace economics.

---

## 7. Non-goals / claim discipline

- No autonomous production routing; no real cluster mutation; no Kubernetes
  write API; no ML training; no new datasets; no robust-energy-engine change; no
  simulator-constant tuning; no synthetic workload-value weights.
- No model substitution — ever.
- **No production-savings claim.** All economic numbers are directional
  simulator/backtest evidence until the `docs/RESULTS.md` §8 production-claim
  gate is met.
