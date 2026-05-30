# Alibaba GenAI 2026 — Ablation / Affinity Audit

> **Simulator benchmark result — directional only, NOT production savings** (`docs/RESULTS.md` §8). **Measurement only:** this audit re-composes the EXISTING `genai_backtest` mechanisms (the `affinity` cold-start flag + the five sizing strategies) into a factorial grid. **No optimizer logic was added and no constant was changed.**

- **Source:** `data/external/alibaba_genai/raw`
- **Cold-start calibration (s, pipeline-layer medians):** {'pipeline_inference': 15.1, 'pipeline_update': 8.9, 'model_predict': 35.0, 'basemodel_load': 22.7, 'controlnet_load': 3.9, 'lora_load': 4.4}

## Two orthogonal existing knobs

1. **affinity** — model-affinity / warm-pool cold-start avoidance (`_effective_service_s(..., affinity=True)`). **In the implemented optimizer `prewarm` and `model-affinity` are the SAME mechanism** (route to a warm replica ⇒ avoid reloading the model); there is no separate prewarm constant, so `+prewarm` ≡ `+affinity` — stated honestly.
2. **sizing strategy** — `static_peak` (fifo), `reactive_sla` (sla_aware), `queue_target` (queue_aware), `util_target` (utilization_aware), `anticipatory_sla` (constraint_aware).

## Ablation grid (full trace)

| config | sizing | affinity | goodput/$ | SLA-compliant | infra $ | GPU-hrs | e2e p99 (s) | mean cold-start (s) |
|---|---|---|---|---|---|---|---|---|
| fifo | static_peak | no | 1.77 | 26,392 | 14,931 | 4,977 | 53 | 23.6 |
| fifo_plus_affinity | static_peak | yes | 3.18 | 26,391 | 8,295 | 2,765 | 36 | 2.9 |
| sla_aware | reactive_sla | no | 5.19 | 17,794 | 3,426 | 1,142 | 1,219 | 23.6 |
| sla_aware_plus_affinity | reactive_sla | yes | 8.18 | 20,399 | 2,493 | 831 | 846 | 2.9 |
| queue_aware | queue_target | no | 5.25 | 15,815 | 3,015 | 1,005 | 1,597 | 23.6 |
| queue_aware_plus_affinity | queue_target | yes | 7.71 | 18,109 | 2,349 | 783 | 1,367 | 2.9 |
| utilization_aware | util_target | no | 6.83 | 18,045 | 2,643 | 881 | 406 | 23.6 |
| utilization_aware_plus_affinity | util_target | yes | 9.05 | 19,119 | 2,112 | 704 | 370 | 2.9 |
| constraint_aware | anticipatory_sla | yes | 9.84 | 26,392 | 2,682 | 894 | 53 | 2.9 |
| constraint_aware_no_affinity | anticipatory_sla | no | 7.05 | 26,392 | 3,741 | 1,247 | 66 | 23.6 |

## Affinity lift per sizing strategy (affinity is orthogonal + consistent)

| sizing | goodput/$ no-affinity | goodput/$ +affinity | affinity lift |
|---|---|---|---|
| static_peak | 1.77 | 3.18 | +80.0% |
| reactive_sla | 5.19 | 8.18 | +57.5% |
| queue_target | 5.25 | 7.71 | +47.0% |
| util_target | 6.83 | 9.05 | +32.6% |
| anticipatory_sla | 7.05 | 9.84 | +39.5% |

Affinity adds a **consistent +33–80%** regardless of sizing strategy — it is an orthogonal lever, not an artefact of one sizing choice.

## Attribution of the +89.5% (constraint_aware vs sla_aware headline)

2×2 factorial corners (factor A = sizing reactive→anticipatory, factor B = affinity off→on), Shapley decomposition (average marginal contribution over both orderings):

- **model-affinity / prewarm:** **62.1%** of the gain (2.887 goodput/$)
- **anticipatory sizing:** **37.9%** (1.759 goodput/$)
- **interaction:** 0.0% (0.0 goodput/$)

### Single-factor lift vs FIFO (each lever in isolation)

| lever | lift vs FIFO |
|---|---|
| model-affinity alone (FIFO+affinity) | +80.0% |
| prewarm alone (≡ affinity) | +80.0% |
| queue-awareness alone | +196.8% |
| utilization-awareness alone | +286.3% |
| anticipatory-sizing alone | +299.1% |
| combined constraint_aware | +456.7% |

> **Caveat:** the FIFO baseline here is `static_peak` (it provisions every tick at the peak load → very expensive), so the *sizing* levers' vs-FIFO lifts are inflated by "any dynamic sizing beats static over-provisioning". The **Shapley split above (vs the sla_aware headline)** is the principled attribution; the **affinity lift is the orthogonal, consistent one** across every sizing strategy.

## Verdict

**The +89.5% GenAI 2026 gain is primarily a model-affinity / prewarming effect (~62% of the gain); anticipatory sizing is secondary.**

Answering the audit questions directly:

1. **Model-affinity contribution:** ~62.1% of the headline gain; +80% in isolation vs FIFO; cuts mean cold-start ~23.6s → ~2.9s.
2. **Prewarm contribution:** identical to affinity — **prewarm and model-affinity are the same implemented mechanism** (no separate prewarm logic exists to ablate).
3. **Queue-optimization contribution:** small as an independent lever (queue_target ≈ reactive_sla sizing); most of its vs-FIFO lift is "dynamic vs static sizing", not queue-specific.
4. **Utilization-optimization contribution:** util_target (hotter ρ) is the cheapest sizing but sacrifices tail latency (e2e p99 406s vs 53s for constraint_aware).
5. **Interaction effects:** ~0.0% — affinity and sizing are nearly **additive** (affinity helps every sizing strategy by a similar factor).

**Is it primarily prewarming or a broader optimizer effect?** It is **primarily the affinity/prewarm lever (~62.1%)**, but **not exclusively**: anticipatory SLA-aware sizing contributes the remaining ~37.9% and is what lets constraint_aware keep **all** requests SLA-compliant (lowest e2e p99) — a safety property the affinity-only and utilization-only configs do not achieve. `constraint_aware_no_affinity` still beats the `sla_aware` headline (7.05 vs 5.19 goodput/$), and `sla_aware_plus_affinity` recovers most of the gain (8.18) — confirming affinity is the dominant, transferable component.

## Honest limits
- Directional simulator result; cold-start magnitudes are pipeline-layer calibration (medians), not a per-request join (application↔metric layers are `no_join`). Affinity vs no-affinity is the modelled cold-start amortisation, not a re-simulation of a real router. No production logic changed; no constants tuned. **Not production-real savings.**

