# Safe Utilization Frontier Controller — v1 (simulator / shadow-mode)

> **Status:** Simulator and shadow-mode controller phase. **Real-cluster
> execution is DISABLED BY DEFAULT.** Real execution requires an explicit
> caller opt-in flag (`allow_real_execution=True`) **and** a real executor
> passed in by the caller — neither ships in this package.
>
> **No production-savings claim** (`docs/RESULTS.md` §8). Directional
> simulator/backtest evidence only. Pilot telemetry is required to
> calibrate the safe rho per workload before any production claim.
>
> **No ML model is trained.** **No new datasets are ingested.** **The
> robust energy engine is not modified.** **No optimizer constant is tuned
> to force a benchmark win.**

Read first: `docs/RESULTS.md`, `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`,
`docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md`,
`docs/PILOT_TELEMETRY_CONTRACT.md`.

---

## 1. Product thesis

Aurelius finds and maintains the **maximum sustainable** infrastructure
usage across the combined SLA, latency, queue, thermal, topology, energy,
residency, and telemetry-confidence constraints — **not** the maximum raw
utilization.

The Azure LLM 2024 audit (`docs/AZURE_2024_SAFE_UTILIZATION_FRONTIER.md`)
established three findings:

1. `constraint_aware` at the default rho ≈ 0.65 is **SAFE but
   conservative** — it sits *inside* the safe frontier.
2. The safe peak on the anticipatory frontier (the safer dominant frontier
   per the audit) is **`anticipatory@0.75`**, ~13% goodput/$ above
   `constraint_aware`'s default operating point.
3. The win comes from **safe higher utilization** (lower GPU-hours at
   sustainable load), **not** ML demand forecasting (oracle ceiling ~0.25%
   of KPI on that trace).

The Safe Utilization Frontier Controller turns that audit into a
first-class controller: estimate the safe frontier, pick the best safe
goodput/$ point, recommend it, and execute *only* in simulator/backtest by
default.

---

## 2. Module layout

```
aurelius/frontier/
  models.py        # WorkloadFrontierProfile, FrontierPoint, FrontierDecision,
                    # FrontierAction, SafetyStatus
  safety.py        # SafetyConfig, is_frontier_point_safe,
                    # classify_point_safety
  estimator.py     # FrontierEstimatorConfig, estimate_frontier,
                    # estimate_frontier_from_points
  controller.py    # FrontierControllerConfig, choose_safe_utilization_target
  execution.py     # execute_frontier_decision, EXECUTION_MODES,
                    # RealExecutionDisabledError
  shadow.py        # FrontierShadowDecisionLog, FrontierShadowLog,
                    # JSONL writer/reader helpers
```

Plus the Azure 2024 driver: `scripts/run_azure_2024_frontier_controller.py`.

---

## 3. Selection algorithm

`choose_safe_utilization_target(profile, frontier_points, current_rho, cfg)`:

1. **Telemetry-confidence gate.** If `profile.telemetry_confidence` is
   below `cfg.min_telemetry_confidence`, or if every candidate point is
   `INSUFFICIENT_TELEMETRY` → return `INSUFFICIENT_TELEMETRY`.
2. **Current-rho unsafe → LOWER_RHO.** If the point at `current_rho` is
   UNSAFE, recommend the next-lower SAFE rho (or the workload floor).
3. **Filter to SAFE points.** Drop UNSAFE / INSUFFICIENT_TELEMETRY points.
4. **No safe points → LOWER_RHO.** Recommend the smallest tested rho (or
   INSUFFICIENT_TELEMETRY if telemetry is the dominant gap).
5. **Pick max goodput/$.** Among SAFE points, choose the one with the
   highest `predicted_goodput_per_dollar`.
6. **Conservative margin (opt-in).** If the best safe point is *adjacent*
   to a first-unsafe point and `cfg.conservative_margin=True`, step back
   to the next-lower safe rho. Transparent, configurable, off by default.
7. **Deadband.** If the selected rho is within `cfg.deadband_rho` of the
   current rho AND the KPI delta is within `cfg.deadband_kpi_pct` →
   `KEEP_RHO` (avoid churn on noise).

The controller **never** picks the highest rho blindly. It picks the
highest KPI among safe points; safety is a veto, not a weight.

---

## 4. Safety gates

`SafetyConfig` carries explicit, pre-registered thresholds. Defaults
mirror the Azure 2024 audit:

| gate | default | unit | semantics |
|---|---|---|---|
| `max_timeout_pct` | 10.0 | % | timeout share over the window |
| `max_queue_p99_ms` | 2000.0 | ms | queue p99 |
| `max_queue_p95_ms` | None | ms | opt-in; None disables |
| `max_latency_p99_ms` | None | ms | opt-in; None disables |
| `max_latency_p95_ms` | None | ms | opt-in; None disables |
| `min_telemetry_confidence` | `"low"` | — | low / medium / high |
| `max_thermal_risk` | None | 0..1 | opt-in |
| `min_topology_score` | None | 0..1 | opt-in |
| `max_memory_pressure` | None | 0..1 | opt-in |
| `max_scale_events` | None | count | opt-in |
| `max_churn_score` | None | float | opt-in |

A configured gate with **missing telemetry** does NOT auto-pass — the
point is marked `INSUFFICIENT_TELEMETRY`, not `SAFE`. A configured gate
that **breaches** the threshold marks the point `UNSAFE` (hard breach
wins over missing telemetry). SLA violations are NEVER folded into the
KPI score (`docs/RESULTS.md` §1-§2).

---

## 5. Execution interface

`execute_frontier_decision(decision, mode, *, executor, simulated_state,
allow_real_execution)`:

| mode | mutates state? | notes |
|---|---|---|
| `shadow` | **no** | recommendation logged; nothing mutates |
| `simulator` | yes (simulated state only) | updates `simulated_state[workload_id]` |
| `real_disabled` | **no** | production write path *exists in signature only*; disabled |
| `real_enabled` | **no by default** | requires `allow_real_execution=True` AND a real `executor` AND no safety vetoes; ships as `not_implemented_real_executor` |

Hard rules (asserted by tests):

- `FrontierDecision.executable_in_real_cluster` is `False` at construction
  and the controller never sets it true.
- `execute_frontier_decision(..., mode="real_enabled",
  allow_real_execution=False)` raises `RealExecutionDisabledError`.
- `execute_frontier_decision(..., mode="real_enabled",
  allow_real_execution=True, executor=None)` returns
  `mutated=False, notes=["not_implemented_real_executor"]`.
- `shadow` and `real_disabled` modes mutate nothing.

There is **no** Kubernetes write API, **no** router write, **no**
serving-engine write in this package.

---

## 6. Shadow logging

`FrontierShadowLog` and `FrontierShadowDecisionLog` provide append-only
JSONL logging of recommendations + their expected deltas. The log carries
`executed`, `execution_mode`, and `safety_vetoes` so a reviewer can
verify shadow posture after the fact. Shadow / real_disabled modes
disallow `executed=True` at construction.

---

## 7. Workload sensitivity

The safe rho is **workload- and SLA-specific**. The Azure 2024 audit's
`anticipatory@0.75` safe-peak (rho = 0.75) is **not** a global constant. A different
workload mix, SLO budget, burst profile, hardware tier, real serving
engine, or trace will move it. The controller therefore:

- accepts a per-workload `WorkloadFrontierProfile`,
- accepts a per-workload `SafetyConfig`,
- accepts a per-workload candidate rho grid clamped to
  `[profile.min_rho, profile.max_rho]`,
- writes the chosen safe rho + the full frontier + every veto into the
  shadow log so the reviewer can see *why* a rho was chosen.

ML demand forecasting is **NOT required for v1**. The Azure 2024
attribution showed forecasting contributes ~0.25% of KPI on that trace;
the leverage is the safe-utilization controller itself.

---

## 8. Pilot calibration requirements

Real customer telemetry must measure (per workload):

- timeout / SLA-violation rate vs provisioning,
- queue p95 / p99 vs provisioning,
- latency p95 / p99 vs provisioning (when the runtime exposes them —
  Azure 2024 does not, GenTD26 does not),
- telemetry confidence / partial-flag,
- thermal / topology / memory headroom,
- workload SLA budget (model_id-level),
- scale event / churn counts.

Until those signals exist (`docs/PILOT_TELEMETRY_CONTRACT.md`), the
controller stays in shadow / simulator mode. The committed
`constraint_aware` engine default (rho ≈ 0.65) is **unchanged**.

---

## 9. Non-goals (v1)

- **Do NOT** change the committed `constraint_aware` default rho.
- **Do NOT** modify the robust energy engine.
- **Do NOT** tune optimizer constants to force a benchmark win.
- **Do NOT** train ML models or ingest new datasets.
- **Do NOT** quote production-savings numbers.
- **Do NOT** use oracle as a headline baseline.
- **Do NOT** mutate production infrastructure.

---

## 10. Claim discipline

- Simulator / public-trace evidence only — **not production savings**
  (`docs/RESULTS.md` §8 gate unmet).
- Real-cluster execution is disabled by default.
- Pilot telemetry is required before any safe-rho promotion.
- The product thesis (`maximum sustainable usage across constraints`) is
  *supported* by the Azure 2024 audit, but the customer-specific safe
  rho remains an open empirical question.
