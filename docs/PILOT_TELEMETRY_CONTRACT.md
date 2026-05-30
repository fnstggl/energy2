# Pilot Telemetry Contract — Model Residency / Cold-Start

> **Status:** Telemetry contract (schema) only. **No connector, optimizer,
> simulator, or energy-engine code is changed by this document.** It specifies
> the fields a pilot deployment must emit so Aurelius can *observe* (and later,
> under separate gating, *act on*) model-residency / cold-start risk.
>
> Companion to `docs/MODEL_RESIDENCY_COLD_START_SPEC.md`. Read `docs/RESULTS.md`
> (reporting + claim rules) first. **Directional / simulator-only until the
> `docs/RESULTS.md` §8 production-claim gate is met — not production savings.**

This contract is grounded in the GenAI 2026 result + ablation
(`docs/ALIBABA_GENAI_BACKTEST_RESULTS.md`, `docs/ALIBABA_GENAI_ABLATION_RESULTS.md`),
which attributed **~62%** of the +89.5% goodput/$ win to model-affinity/prewarm.
To reproduce that lever on real systems, the residency/cold-start fields below
are **required**; the public GenTD26 trace was missing the request→GPU join key
(its application and infrastructure layers are `no_join`), which is exactly the
gap this contract closes.

---

## 1. Conventions

- **Encoding:** newline-delimited JSON or columnar (Parquet/CSV) with the field
  names below. One stream per layer (request / infra), joinable by the keys in §4.
- **Timestamps:** RFC 3339 UTC or epoch milliseconds; document which. Sub-second
  precision required for load-latency fields.
- **Durations:** seconds (float) unless suffixed `_ms`.
- **Null handling:** explicit `null`; unify source `"NULL"`/`"None"`/`""` to
  `null` (the GenTD26 convention). A field that is *structurally* absent (the
  source cannot emit it) MUST be recorded as absent in the layer-coverage report,
  not silently zero-filled.
- **Anonymization:** ids MAY be hashed; hashing MUST be stable within a pilot so
  joins hold.
- **PII:** prompt/response **content** is NOT required and SHOULD NOT be sent;
  only sizes/metadata.

---

## 2. Request layer (per-request record) — REQUIRED fields

| field | type | unit / values | required | notes |
|---|---|---|---|---|
| `request_id` | string | — | **yes** | unique per request |
| `timestamp` | ts | UTC | **yes** | request arrival |
| `tenant_id` / `workload_id` | string | — | **yes** | billing / workload owner |
| `model_id` | string | — | **yes** | base model requested (never substituted) |
| `adapter_id` / `lora_id` | string | — | if present | adapter requested; null = base only |
| `endpoint_id` | string | — | **yes** | serving endpoint / deployment |
| `region` | string | — | **yes** | placement region |
| `node_id` | string | — | **yes\*** | serving node (\*join key, see §4) |
| `gpu_id` | string | — | **yes\*** | serving GPU (\*join key) |
| `container_id` | string | — | **yes\*** | serving container/pod (\*join key) |
| `model_loaded_before_request` | bool | — | **yes** | residency hit for base model |
| `adapter_loaded_before_request` | bool | — | if adapter | residency hit for adapter |
| `model_load_start` | ts | UTC | if cold | null when residency hit |
| `model_load_end` | ts | UTC | if cold | — |
| `adapter_load_start` | ts | UTC | if cold adapter | — |
| `adapter_load_end` | ts | UTC | if cold adapter | — |
| `queue_wait` | float | s | **yes** | gateway/scheduler wait |
| `TTFT` | float | s | yes (gen) | time-to-first-token (autoregressive) |
| `TPOT` | float | s/token | yes (gen) | time-per-output-token |
| `e2e_latency` | float | s | **yes** | end-to-end (incl. any load) |
| `status` / `error` | string | OK / code | **yes** | success / failure class |

`model_load_*` / `adapter_load_*` are the fields that make cold-start
**measurable** rather than inferred. If a runtime cannot emit explicit load
timestamps, it MUST at minimum emit `model_loaded_before_request` so cold-start
**rate** (not latency) is still computable; the layer-coverage report must note
the missing latency.

---

## 3. Infrastructure layer (per-sample time-series) — REQUIRED fields

| field | type | unit | required | notes |
|---|---|---|---|---|
| `timestamp` | ts | UTC | **yes** | sample time |
| `node_id` | string | — | **yes** | join key |
| `gpu_id` | string | — | **yes** | join key |
| `container_id` | string | — | **yes** | join key |
| `gpu_utilization` | float | % (0–100) | **yes** | DCGM `DCGM_FI_DEV_GPU_UTIL` |
| `gpu_memory_used` | float | bytes | **yes** | DCGM `DCGM_FI_DEV_FB_USED` |
| `gpu_memory_total` | float | bytes | **yes** | for residency headroom |
| `power` | float | W | if available | DCGM `DCGM_FI_DEV_POWER_USAGE` |
| `cache_hit` / `prefix_hit` | float | rate 0–1 | if available | vLLM/SGLang prefix-cache |

`gpu_memory_used`/`_total` bound how many models/adapters can be resident at once
— the capacity constraint behind the warm-pool / evict decisions
(`docs/MODEL_RESIDENCY_COLD_START_SPEC.md` §4).

---

## 4. Cross-layer join + linkage-quality reporting (binding)

Residency/cold-start attribution requires linking a request to the
container/GPU that served it. Every pilot MUST emit a **linkage-quality
classification** per layer pair, exactly as the GenAI ingester does
(`aurelius/traces/alibaba_genai.py::classify_linkage`):

- `exact_join` — shared `request_id` / batch key across layers.
- `container_join` — shared `container_id` (+ `gpu_id`) and overlapping time.
- `time_join` — only timestamps align (same clock base).
- `no_join` — usable only independently.

**Honesty gate (binding):** if the request↔infra linkage is `no_join` or
`time_join` only (as in the public GenTD26 trace), the pilot **MUST** label
residency metrics *calibration-only / unattributed* and **MUST NOT** claim
per-request request→GPU causality. A pilot intending to *act* on residency
(beyond shadow recommendations) MUST achieve at least `container_join`.

---

## 5. Derived-metric outputs (what the pilot computes from §2–§3)

The pilot emits the `docs/MODEL_RESIDENCY_COLD_START_SPEC.md` §3 derived metrics,
each with `{value, denominator, window, linkage_quality}`:
`model_residency_hit_rate`, `adapter_residency_hit_rate`, `cold_start_rate`,
`cold_start_latency_p50/p95/p99`, `warm_pool_cost`, `cold_start_avoided_latency`
(range), `sla_violations_attributable_to_cold_start`,
`goodput_per_dollar_with_prewarm`, `goodput_per_dollar_without_prewarm`,
`model_popularity_half_life`, `residency_churn_score`.

`goodput_per_dollar_*` MUST use the canonical KPI
(`aurelius/benchmarks/economics.py`); residency metrics are **diagnostics** and
MUST NOT be folded into it (`docs/RESULTS.md` §1–§2).

---

## 6. Shadow-mode posture (binding)

- All residency actions are **recommendation-only** by default; **no real
  cluster mutation** (`docs/MODEL_RESIDENCY_COLD_START_SPEC.md` §5).
- Each recommendation is logged with inputs, rule, predicted cost/penalty, and a
  decision id; compared to the observed counterfactual when one becomes available,
  else marked `unverified`.
- **No production-savings claim** is permitted from pilot telemetry until the
  `docs/RESULTS.md` §8 gate is satisfied. Use the §8 allowed wording
  ("directional only — not production savings", "requires live telemetry
  calibration").

---

## 7. Minimal conformance checklist

A pilot is **residency-observability conformant** when it emits:
1. the REQUIRED §2 request fields (incl. `model_loaded_before_request`),
2. the REQUIRED §3 infra fields (incl. `gpu_memory_used/total`),
3. a per-pair `linkage_quality` (§4),
4. the §5 derived metrics with denominators + windows,
5. under the §6 shadow-mode posture, with no production-savings claim.

It is **residency-actuation conformant** (a prerequisite for promoting any
recommendation out of shadow mode) only when, additionally, the request↔infra
linkage is at least `container_join` and the `docs/RESULTS.md` §8 production-claim
gate is satisfied.
