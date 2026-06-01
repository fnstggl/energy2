# Placement Prior Audit — TTFT p50 Shadow Prior Integration

> **Audit / shadow PR.** Reads the existing goodput/$ placement / routing
> path; does not change scheduler defaults; does not enable real
> execution; does not modify the robust energy engine; does not claim
> production savings. A negative finding ("TTFT p50 is not economically
> important yet") is the explicit result.
>
> **Read first:** `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md`,
> `docs/CARA_LATENCY_FORECASTER_V1_RESULTS.md`,
> `docs/FORECAST_LEVERAGE_AUDIT.md`,
> `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`, `docs/RESULTS.md`.

## 1. Scope (binding)

- Trace the existing goodput/$ scoring path and catalogue every input.
- Classify each input as measured / forecasted / static_prior /
  heuristic / constant / proxy / missing.
- Add the **TTFT p50 shadow prior** as an **optional** refinement of the
  existing scorer, never as a parallel economic scorer.
- Measure offline whether the prior changes rankings / top-1 placements
  / projected goodput per dollar / SLA risk / safety.
- Apply the binding promotion rule from the mission spec.

## 2. Existing scoring path (traced)

The audit pins the entry point as
`aurelius/residency/decision.py::score_residency_candidate`. Machine-
readable trace + signature snapshot live in
`data/external/forecasting/placement_prior_audit/scoring_path_audit.json`.

For every (request, candidate) pair the scorer computes:

```
expected_latency_s = queue_wait + model_penalty + adapter_penalty + service_s
expected_cost      = (expected_latency / 3600) * gpu_hour_price + memory_pressure_cost
sla_met            = expected_latency <= sla
goodput_per_dollar = (1 if sla_met else 0) / expected_cost
```

Inputs (15 total; full list in JSON):

| input | classification | enters where |
|---|---|---|
| TTFT | heuristic | `SafetyContext.service_time_proxy_s` (default **2.0 s**) or `seconds_per_token * output_tokens` |
| TPOT | heuristic | same as TTFT (no separation) |
| E2E latency | derived | `queue_wait + load_penalty + service_s` |
| queue depth | measured | `ModelLocationState.queue_depth` |
| queue wait | measured_or_proxy | `loc.estimated_queue_wait_s` or `queue_depth * service_s` |
| **GPU type** | **missing** | encoded only in `location_key` string; never parsed |
| **model size** | **missing** | encoded only in `request.model_id` string; never parsed |
| throughput | missing | EMA decode/prefill from CARA not consumed |
| KV cache state | proxy | only `has_model` / `has_adapter` booleans |
| cache reuse | missing | SwissAI bucket-reuse signals not wired in |
| residency / cold-start | static_prior | `ModelLoadProfile.model_load_penalty_s` |
| cost | derived | constant `gpu_hour_price = 3.0` × latency |
| energy / carbon | missing | only batch scheduler, not serving placement |
| SLA risk | derived_binary | hard threshold, no probability |
| timeout risk | missing | only in `frontier/risk.py`, not in residency scorer |

**Headline gap:** GPU type is not used as a latency prior anywhere in
the goodput/$ scorer. The 9× p99 TTFT spread observed across GPU types
in CARA (`docs/CARA_LATENCY_FORECASTER_V1_RESULTS.md`) is invisible to
the current scorer.

## 3. Hook chosen (and parallel scorer avoided)

The cleanest existing hook is
`aurelius/residency/decision.py::_service_time_s` which reads
`SafetyContext.service_time_proxy_s` and
`SafetyContext.seconds_per_token`. We add a thin **adapter**
`aurelius/forecasting/ttft_shadow_prior.py` exposing:

- `TTFTShadowPrior` — a per-(model_size, gpu_type, prompt_token_bin)
  median-TTFT lookup, fit from CARA train_flat.
- `refine_service_time_proxy_s(ctx, ...) -> (refined_ctx, record)` —
  returns a **copy** of the context with
  `service_time_proxy_s = max(static, predicted_ttft_p50)`. The MAX
  clamp is the binding safety floor: the prior can only widen, never
  tighten, the latency estimate. Default is `apply_to_scorer=False`
  (shadow / logging only).

**No parallel economic scorer is introduced.** The adapter wraps the
existing `SafetyContext` and lets the upstream
`score_residency_candidate` produce the score exactly as before — with
or without the refined service-time. No new gpd formula, no new
ranking logic, no new control path.

The mission's "do not use p95/p99 ML tails for control" rule is enforced
by `test_ttft_shadow_prior_module_does_not_use_p95_p99_for_control` —
the module exposes only a p50 predictor.

## 4. Offline shadow eval — does the prior change rankings?

`scripts/run_ttft_shadow_prior_eval.py` fits the prior on the CARA
train_flat **train** holdout, scores 2,000 test requests under both
policies, and writes
`data/external/forecasting/placement_prior_audit/ttft_shadow_prior_eval.json`.

Per-GPU median TTFT from CARA train (61,460 rows):

| GPU | median TTFT (s) |
|---|---:|
| a30 | 0.036 |
| v100 | 0.049 |
| a100 | 0.078 |
| p100 | 0.157 |

### Binding policy (`max(static, predicted)`, the contract)

| metric | value |
|---|---:|
| top-1 placement change rate | **0.0000** |
| ranking change rate | 0.0000 |
| tie-break rate | 0.0000 |
| latency-estimate change rate | 0.0000 |
| projected goodput/$ delta | **+0.00%** |
| projected SLA-met delta | +0.00 pp |
| projected expected-latency delta | +0.00% |
| safety regressions | 0 |

The CARA-derived p50 prior (sub-second medians) is dominated by the
**2.0 s static service-time proxy**. The MAX clamp leaves
`service_time_proxy_s` at 2.0 for every candidate, every request — no
change in any score, no change in any ranking.

### Diagnostic policy (without the MAX clamp — *not the binding shape*)

| metric | value |
|---|---:|
| top-1 placement change rate | 0.0000 |
| ranking change rate | **1.0000** |
| tie-break rate | **1.0000** |
| latency-estimate change rate | **1.0000** |
| safety regressions | 0 |

Without the safety floor, the prior changes every per-candidate latency
estimate (the predicted TTFT differs per GPU type, propagating through
`queue_wait = queue_depth × service_s`) and breaks the baseline's
indifference at the top — the baseline considers all 5 candidates
equal under the static 2.0 s proxy, the prior strictly orders them.

**But top-1 still doesn't change** because the baseline ties resolve
alphabetically to `qwen2.5-3b_a30`, and the prior also picks A30
(lowest median TTFT). The prior provides information; the existing
tie-break rule happens to agree.

## 5. Promotion decision (PHASE H)

Binding promotion rule (mission spec):

- < 2 % projected goodput/$ improvement → `diagnostic_only`
- 2 – 5 % → `promising_needs_validation`
- ≥ 5 % with zero safety regressions → `shadow_ready_for_integration_review`

| field | value |
|---|---|
| projected goodput/$ delta | +0.00 % |
| safety regression count | 0 |
| top-1 change rate | 0 |
| **final status** | **`diagnostic_only`** |
| reason | "no top-1 placement change; prior does not affect rankings" |

## 6. Honest finding

**TTFT p50 is not economically important yet under the existing scorer.**

Two structural reasons:

1. **The static 2.0 s service-time proxy dwarfs the sub-second TTFT
   prior.** Even the slowest CARA GPU (P100 median 0.157 s) is an order
   of magnitude below the safety floor. The MAX clamp — which is the
   binding safety floor — leaves every prediction at 2.0 s.
2. **The existing scorer has no per-(GPU, model) cost surface.** A
   single global `gpu_hour_price = 3.0` means cost is purely
   proportional to latency; the per-GPU TTFT differences translate into
   *negligible* cost differences ($0.0001-scale per request).

What this also confirms:

- The leverage audit's **"TTFT p50 → constraint-aware engine"** path
  cannot fire on the existing scorer alone — the integration shape must
  change first (lower service-time proxy, per-GPU cost surface, or
  tighter SLA budgets that surface TTFT differences).
- The forecaster itself stays `shadow_ready` for **logging**, just not
  for ranking-aware integration today.

## 7. What this unblocks for the next forecast PR

- Establishes the binding adapter contract: prior refines an existing
  `SafetyContext` field; the MAX clamp is the safety floor; default is
  not-applied.
- Surfaces the **scorer-side gaps** (no per-GPU cost; no per-(GPU,
  model) TTFT prior; static service-time proxy) that need to land
  *before* a latency prior matters. None of those changes are in scope
  for this PR — they are documented as the next milestone.
- Confirms the `TTFTShadowPrior` lookup table for later reuse
  (`data/external/forecasting/placement_prior_audit/ttft_shadow_prior_table.json`).

## 8. Non-goals (binding)

No production scheduler change. No real execution. No oracle / FIFO
headline. No production-savings claim. TTFT p95/p99 ML tails are not
exposed by this module for any control purpose — only the p50 prior is
shadow-wired.
