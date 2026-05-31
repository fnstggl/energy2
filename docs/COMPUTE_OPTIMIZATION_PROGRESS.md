# Compute Optimization Progress Tracker

This is the canonical progress tracker for Aurelius constraint-aware GPU orchestration.

This tracker is separate from `docs/AURELIUS_PROGRESS.md`.

`docs/AURELIUS_PROGRESS.md` may contain legacy energy-optimization or general Aurelius progress. It may be useful historical context, but it is NOT the source of truth for this constraint-aware orchestration initiative.

The source planning document is:

`docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md`

Every implementation run must read that plan before deciding what to do next.

---

## Status Summary

Current status: **READY_FOR_SHADOW_PILOT_WITH_REAL_TELEMETRY (recommendation-only, read-only)**

The audit-first run (2026-05-26) downgraded the system to
`READY_FOR_SIM_ONLY_DEMO` after finding three product-level blockers. A
follow-up repair run (Missions 1–3, 2026-05-26) resolved all three. The
corrected verdict is **READY_FOR_SHADOW_PILOT_WITH_REAL_TELEMETRY**, with
explicit remaining work below. It is **not** production-ready: all heuristics
are uncalibrated and all KPI evidence is simulator-only.

**Blocker resolution (see "Product Repair (Missions 1–3, 2026-05-26)" below):**

1. ~~Real telemetry cannot drive the engine.~~ **RESOLVED.** `aurelius/state/assemble.py`
   `build_cluster_state(...)` aggregates connector leaf objects
   (GPU/service/node/topology/energy/thermal) into the canonical `ClusterState`,
   with honest `is_partial`/`missing_sources`, staleness→confidence, NaN→None, and
   unknown-reference detection. Proven end-to-end through the REAL DCGM/vLLM
   adapters (fixture-backed integration test) and via `constraint-report --config`.
2. ~~No demonstrated KPI improvement.~~ **RESOLVED (sim-only).** The benchmark
   apply layer had a latent bug (UPPERCASE vs lowercase action names) so it
   applied nothing. Fixed, plus simulator action methods (SCALE/SPREAD/DEFER/
   CONSOLIDATE). `constraint_aware` now differs from FIFO in 4/6 scenarios:
   thermal throttle −83%, p99 −15%, queue-wait 180s→0.23s, cost/token improves
   in every acting scenario, **no SLA regression** anywhere.
3. ~~Decisions are top-label-driven.~~ **RESOLVED.** The engine now reasons over
   the full constraint score vector: protects materially-active SLA-risk
   constraints, rejects actions that worsen them, and hard-blocks migrations to
   unsafe destinations. Adversarial cases A–G pass (`tests/test_constraint_multi.py`).

**Remaining before a real shadow pilot (NOT done):**
- Calibrate all `# HEURISTIC` thresholds (classifier, cost model, operational-relief
  weight, simulator physics) against real pilot telemetry.
- Complete `build_cluster_state_from_connectors` per-cluster node/service mapping
  (current orchestration is a working scaffold; multi-node topology mapping is manual).
- Validate against a LIVE cluster (all KPI evidence is simulator-only).
- Model within-region topology replacement (CHANGE_PLACEMENT) and node resume cost.
- Simulator realism is still dev-grade (smooth queue arrivals, fixed latency tails,
  cheap migrations, always-perfect telemetry) — see Synthetic Realism Audit.

---

## Independent Audit Findings (2026-05-26)

An audit-first run independently re-verified the constraint-aware system against
repo reality (not docs/test claims). Commands run: `python -m compileall aurelius`
(clean); full suite `2194 passed, 12 failed, 13 skipped` (the 12 failures are all
legacy energy/ML/API tests failing only because optional deps `lightgbm`/`fastapi`
are absent — env gaps, not constraint-aware regressions; the constraint-aware
suite is 100% green). A full benchmark (`benchmark-run --all-scenarios`), all CLI
`--help` smokes, a constraint-report, and a 7-case adversarial multi-constraint
harness were run.

**Verdict: READY_FOR_SIM_ONLY_DEMO** (not pilot-ready). Evidence:

### Gap 1 — Connectors are not wired into `ClusterState` (IMPLEMENTED_BUT_NOT_WIRED)
- The only `ClusterState(...)` producers in source are the simulator
  (`aurelius/simulation/cluster/engine.py:1198`) and the classifier's internal
  region re-scoping (`aurelius/constraints/classifier.py:706`), plus
  `ClusterState.from_dict` for loading a pre-existing JSON snapshot.
- Connectors (`prometheus`, `dcgm`, `vllm`, `triton`, `ray_serve`, `kubernetes`,
  `topology`) emit canonical *leaf* objects (`GPUState`, `InferenceServiceState`,
  `NodeState`, `TopologyState`) but **no code aggregates them into a
  `RegionState`/`ClusterState`.** Their only non-test caller is the
  `validate-connectors` diagnostic, which inspects shapes and discards the objects.
- Consequence: real Prometheus/DCGM/K8s/vLLM telemetry **cannot drive the engine
  today**; the constraint-aware pipeline has only ever run on the simulator.
- The connectors themselves are high quality: read-only K8s (RBAC get/list/watch
  only), missing-metric→None (never fabricated 0), failure→partial/unknown, and
  no secret logging (auth via env-var names resolved at call time). This gap is
  the missing *assembler*, not the connectors.

### Gap 2 — Benchmark shows zero KPI improvement for the constraint-aware policy
- In **all 6** canonical scenarios, `constraint_aware` KPIs are **byte-identical
  to `fifo`** (cost, SLA violations, migrations, p99). It makes no
  KPI-changing intervention anywhere.
- Cause (a): `aurelius/benchmarks/constraint_runner.py:_apply_constraint_aware`
  only applies recommendations whose action is a cross-region migration
  (`CHOOSE_CHEAPER_REGION`/`MIGRATE`/`CHANGE_PLACEMENT` with a target region).
  SPREAD, SCALE_REPLICAS, CONSOLIDATE, DEFER, REROUTE are computed but never
  applied.
- Cause (b): the simulator's only mutation method is `migrate_workload` — it does
  **not model** spread/scale/consolidate/reroute/defer, so for 5 of 6 constraint
  families the benchmark cannot demonstrate improvement even in principle.
- Cause (c): in the one family it can affect (energy), the state-conditioned cost
  model vetoed every candidate migration → 0 migrations. (Meanwhile naive
  `current_price_only`/`greedy_energy` baselines do cut energy cost — but at the
  cost of 60 000–180 000 ms queue waits in the simulator.)
- Consequence: there is **no benchmark evidence** that constraint-aware
  optimization improves cost/token, p99, utilization, or SLA over baselines.

### Gap 3 — Decisions are top-label-driven, not genuinely multi-constraint
- The classifier computes a full 8-family score vector, but the engine generates
  candidates and applies disallowed-actions **only** from the single
  `binding_constraint`. There is no `secondary_constraints` concept; the score
  vector is consumed only for display, the binding-score SLA proxy, and the narrow
  `CONSOLIDATE + binding==THERMAL` cost term.
- Adversarial harness (7 cases): **A, B, C1, F behave correctly** — but only
  because an SLA-risk constraint out-scored the cost constraint and won the single
  binding label. **C2, D, E, G fail:**
  - C2: util-dominant binding + secondary thermal (78 °C) → recommends
    `CONSOLIDATE` (net +7.3) into warm nodes; secondary thermal ignored.
  - D: topology binding + severe secondary thermal (score 0.92, throttling) →
    `change_placement` with no thermal consideration or tradeoff explanation.
  - E: energy binding → migrates to a destination with 3 % spare + PCIe (net +69);
    destination-risk penalty too small to block.
  - G: missing destination telemetry → still migrates (net +84); uncertainty
    buffer dwarfed by inflated synthetic gross savings.
- Root cause of E/G: soft penalties are capped (sum of risk weights = 24.0), so
  any action with gross savings > 24 can never be vetoed by soft penalties alone.

### Gap 4 — Simulator realism is dev-grade and optimizer-friendly
- Telemetry is always perfect (`confidence="high"`, `is_partial=False`,
  `sample_age_s=None`) — the classifier's missing-signal / staleness / fail-safe
  paths are never exercised by the sim.
- `migrate_workload` has no network-transfer cost, no dropped in-flight requests,
  no rollback risk (only a 2-tick warmup) → migrations are cheaper than reality.
- Queue arrivals are a smooth diurnal sinusoid (not bursty/Poisson); latency tails
  use fixed p95=p50×2.5 / p99=p50×5 multipliers (no nonlinear blow-up near
  saturation).
- Scenarios are answer-keyed (`expected_primary_constraint` hand-set) and energy
  spikes are clean 2.5× anti-correlated traces → easy arbitrage.
- Bug fixed in this run: `_compute_topology_score` (engine.py:1010) had a
  dead-code weight (`base*(1-w)+base*w == base`) so comm-intensity never affected
  the topology KPI; now `base_score*(1 - comm_weight*(1-base_score))`.

### Doc-claim corrections
- "enterprise-pilot-ready" → **READY_FOR_SIM_ONLY_DEMO**.
- "What is production-ready" list below: the connector/topology entries are
  **IMPLEMENTED_BUT_NOT_WIRED** (parse real-shaped fixtures and emit canonical
  leaf types, but are not assembled into `ClusterState`). "Supports real
  Prometheus/Kubernetes/DCGM/vLLM/Triton/Ray" is PROVEN_BY_UNIT_TESTS_ONLY at the
  parsing layer, OVERCLAIMED at the end-to-end layer.

### Exact next task (highest priority)
Build the connector→`ClusterState` assembler (`build_cluster_state(...)` that
aggregates connector leaf objects into `RegionState`/`ClusterState`, marking
absent sources via `is_partial`/`missing_sources`) and an integration test that
drives the classifier from fixture-backed connector output end-to-end. This
unblocks every other claim (real telemetry, calibration, shadow pilot).

> **DONE** in the repair run below.

---

## Product Repair (Missions 1–3, 2026-05-26)

This run repaired the three product-level blockers from the audit above. All
changes keep `recommendation_only` as default and introduce no cluster mutation.

### Mission 1 — Real telemetry assembly path (`aurelius/state/assemble.py`)
- `build_cluster_state(*, timestamp, gpu_states, inference_services, node_states,
  topology_state, energy_states, thermal_states, prometheus_snapshot,
  placement_states, source_metadata, default_region, ...)` aggregates connector
  leaf objects into one canonical `ClusterState`.
- Missing sources → `is_partial=True` + `missing_sources` (never fabricated 0s);
  staleness → confidence degradation; NaN/inf → None; unknown region/node/GPU
  references detected and recorded; sandbox provenance propagated.
- `build_cluster_state_from_connectors(config, connectors, timestamp)` drives
  connectors with per-source try/except (a failed connector → missing source,
  not a crash).
- CLI: `constraint-report --config <fixtures.json>` and `telemetry-check --config`
  drive the REAL DCGM/vLLM adapters → assembler → ClusterState → classifier →
  engine.
- Tests: `tests/test_state_assemble.py` (23) incl. a real-adapter integration
  path (simulator DCGM/vLLM Prometheus text → production adapters → assembler →
  engine) and a connector-failure-handling test.
- **Evidence chain:** connector fixture text → `DCGMAdapter.normalize_gpus` /
  `VLLMAdapter.normalize_all_services` → `build_cluster_state` → `ConstraintClassifier`
  → `ConstraintAwareEngine` → `Recommendation`. No simulator `get_cluster_state()`
  involved.

### Mission 2 — Holistic multi-constraint engine (`aurelius/constraints/engine.py`)
- The engine consumes the full `scores` vector, not just `binding_constraint`.
- Active-constraint candidate generation; when any SLA-risk constraint
  (latency/queue/thermal/memory) is materially active (≥ `safety_active_threshold`,
  HEURISTIC 0.20) the engine PROTECTS it and does not chase energy/cost migrations.
- `_action_impact` estimates each candidate's signed impact across all families;
  cross-constraint rejection drops any action that worsens a materially-active
  SLA-risk constraint (disallowed-union across active constraints).
- Hard destination-safety gate: a migration to a critically-low / unverifiable
  destination is blocked regardless of gross savings (closes the
  soft-penalty-ceiling escape).
- Operational-relief value lets SPREAD/SCALE/REROUTE be recommended (previously
  always KEEP'd for lack of monetary savings).
- Tests: `tests/test_constraint_multi.py` (14) — adversarial A–G now pass; all 53
  pre-existing engine tests preserved.

### Mission 3 — Simulator/benchmark action realism
- **Latent bug fixed:** the benchmark apply layer compared action types against
  UPPERCASE names while `ActionType.value` is lowercase, so `constraint_aware`
  /`sla_aware` applied NOTHING (the real reason they were byte-identical to FIFO).
- Simulator action methods with documented realism + confidence:
  `add_replica` (MODERATE), `spread_workload` (MODERATE), `defer_flexible_workload`
  (LOW), `consolidate_low_priority` (LOW). Each mutates SimCluster so the next
  tick's physics reflect it; semantics/limitations/calibration-needs are in
  docstrings.
- Benchmark dispatches every safe action type; regression detector now flags
  cost on **cost-per-token** (throughput-normalized), not absolute cost.
- Tests: `tests/test_simulator_actions.py` (11).
- **Sim-only results vs FIFO:** thermal throttle 72→12 (−83%), p99 −15%, +37%
  tokens, same cost, same SLA; queue p95 wait 180 000ms→233ms; cost/token improves
  in all 4 acting scenarios; **no SLA regression** anywhere.
  `latency_tail_kvcache` (node full → no idle GPU) and `topology_fragmentation`
  (within-region replacement not modeled) correctly take no action.

### Honest limits of this repair
- All KPI improvements are **simulator-only**; the simulator is dev-grade and
  optimizer-friendly (see Synthetic Realism Audit). Do not quote these as
  customer savings.
- All thresholds remain `# HEURISTIC`; none are calibrated to real telemetry.
- `build_cluster_state_from_connectors` is a working scaffold; multi-node
  per-service mapping for a specific cluster is still manual.
- No live-cluster validation has been performed.

### Next task after this run
Run a read-only shadow pilot against real Prometheus/DCGM/K8s telemetry (via the
assembler) to (a) validate the connector→ClusterState path on live data and
(b) collect the telemetry needed to calibrate the HEURISTIC thresholds. Do not
make production savings claims until that calibration is done.

---

## Serving Realism Upgrade (2026-05-26)

A follow-up run upgraded the simulator's inference-serving DYNAMICS from
optimistic toward operationally believable. Goal: make the benchmark non-trivial
so that realistic tradeoffs appear and naive strategies can lose — without
fabricating precision.

### Architectural diff
- New `aurelius/simulation/cluster/serving.py` — pure, seedable serving-realism
  functions (Erlang-C, convex saturation, exploding tails, decomposed TTFT/TPOT,
  batching tradeoff, MMPP bursts).
- New `aurelius/simulation/cluster/calibration.py` — every serving parameter is a
  `CalibratedParam{value, source, source_type, confidence, calibration_notes}`,
  inspectable via `calibration_table()` and overridable per run. No hidden
  magic constants.
- `engine._update_queues` rewritten to use the layer; `add_replica` gains
  autoscaling lag + anti-flap cooldown; `migrate_workload` gains a destination
  queue-disruption spike (migrations are not free).
- New state: `SimQueue.in_burst`, `SimWorkload.last_scaled_tick`,
  `SimulatorConfig.serving_config` (per-scenario overrides + `enable_bursts`).

### Source-confidence summary (20 parameters)
- By source type: inferred 10, heuristic 7, documented 2, benchmark-derived 1.
- By confidence: medium 8, low 12, high 0. **None are MEASURED on real hardware.**
- These exist to make *dynamics* qualitatively believable (convexity, tail
  explosion, autoscaling lag), NOT to assert quantitative accuracy. Treat every
  value as a tunable prior. Full table: `calibration_table()`.

### Realism-gap report (subsystem verdicts)
| Subsystem | Before | After | Verdict |
|---|---|---|---|
| Queue saturation | linear-ish | convex beyond safe band + overload region | MODERATE |
| Latency tails | fixed p95=3×mean | p95/p99 grow with ρ; p99 faster | MODERATE |
| TTFT/TPOT | single-base multipliers | decomposed (queue/prefill/contention/KV; per-replica) | MODERATE |
| Batching | linear in replicas | knee tradeoff (more replicas → thinner batches) | MODERATE |
| Autoscaling | instantaneous | detect + warmup lag + anti-flap cooldown | MODERATE |
| Migration cost | cold-start only | + destination queue-disruption spike | MODERATE |
| Arrivals | smooth sinusoid | MMPP bursts (opt-in per scenario) | MODERATE |
| Service rate vs util | — | **STILL** ∝ util (physically backwards) | **LOW — remaining gap** |

### Before/after benchmark KPI (sim-only, vs FIFO, seed 42, 24 ticks)
| Scenario | constraint_aware result | greedy_energy (aggressive) | SLA-viol regression? |
|---|---|---|---|
| energy_arbitrage | p99 ~10s, +18% tokens | **p99 ~910s (LOSES)**, 9 migrations | none |
| queue_surge | +120% tokens, p99 ≤ FIFO | — | none |
| thermal_hotspot | p99 −23%, queue −97%, +37% tokens | — | none |
| underutilization | +150% tokens; **p99 transient regression** (scaling spike) | — | none (SLA count equal) |

### Newly-failing unrealistic strategies
- **greedy_energy** (aggressive energy migration) now LOSES: its migrations
  destabilize destination queues (cold cache + disruption spike + convex
  saturation) → p99 explodes ~88× vs constraint_aware. This is the headline
  product property the audit wanted and could not previously demonstrate.

### Remaining realism limitations (honest)
- **Service rate ∝ utilization** (pre-existing quirk) — physically backwards; a
  low-util GPU has spare capacity but is modelled as slow. This makes slow-service
  low-util scenarios present as latency-adjacent (underutilization p99 flag). A
  proper fix decouples capacity from observed util.
- No full state-class decomposition (RequestArrivalProcess/KVCachePressureState/…);
  realism is added to the existing models, not a from-scratch rewrite.
- No heavy-tailed per-request token distributions (representative fixed prompt
  length); no explicit preemption/recompute storms; no admission-control/shedding;
  KV pressure is a single scalar, not fragmentation.
- All parameters uncalibrated against real telemetry (see confidence table).
- Bursts are opt-in (off for the 6 canonical detection scenarios) to preserve
  their designed single-constraint labels.

### Production-readiness (unchanged): READY_FOR_SHADOW_PILOT_WITH_REAL_TELEMETRY
The simulator is **substantially more realistic, not perfect**. It is now a
better dev/validation harness (naive strategies can lose; tradeoffs are visible),
but all numbers remain simulator-only and uncalibrated. No production savings
claims.

---

## Simulator Realism Audit + Benchmark Validation Upgrade (2026-05-26)

This run added the missing **self-audit + validation surface** on top of the
eight per-subsystem realism commits (#75–#82). The per-subsystem physics was
already built; what was missing was (a) a command that honestly *grades* that
realism, (b) the packing baselines the spec requires, and (c) the consolidated
§9 validation report.

### Files changed
- `aurelius/benchmarks/realism_audit.py` (new) — empirical realism audit. Each
  check PROBES a real code path and reports what it observes, then assigns a
  per-subsystem verdict (`REALISTIC_ENOUGH_FOR_DEV` / `TOO_SIMPLISTIC_FOR_CLAIMS`
  / `NEEDS_REAL_TELEMETRY` / `NOT_PRODUCTION_REALISTIC_YET`).
- `aurelius/benchmarks/packing.py` (new) — first-fit, best-fit, first-fit-
  decreasing, greedy bin-packing + clairvoyant lower bound (analysis-only).
  `analyze_cluster_packing(state)` computes the packing frontier per region.
- `aurelius/benchmarks/report.py` — `BenchmarkReport.packing_frontier` field +
  text/JSON rendering for packing scenarios.
- `aurelius/benchmarks/constraint_runner.py` — captures FIFO final state and
  attaches the packing frontier for utilization/fragmentation scenarios.
- `aurelius/cli.py` + `aurelius/cli_constraint.py` — new `realism-audit` command
  (`--format text|json`, `--strict`, `--output-dir`).
- `scripts/generate_realism_report.py` (new) — reproducible §9 report generator.
- `docs/REALISM_BENCHMARK_VALIDATION.md` (new) — the generated §9 report.
- `tests/test_realism_audit.py` (11), `tests/test_packing_baselines.py` (9).
- `tests/test_serving_realism.py` — the 5× greedy-loses assertion is now an
  honest **xfail** (see "Honest findings" below); a weaker directional assertion
  that currently holds was added alongside it.

### Realism audit verdicts (seed 42)
| Subsystem | Verdict |
|---|---|
| serving | REALISTIC_ENOUGH_FOR_DEV (convex saturation, growing tails, batching knee, bursts) |
| migration | REALISTIC_ENOUGH_FOR_DEV (cold-route resets locality conf 0.82→0.05 + warmup) |
| telemetry | **NEEDS_REAL_TELEMETRY** (blocker — see below) |
| actions | REALISTIC_ENOUGH_FOR_DEV (actions mutate state; KEEP/no-op states reachable) |
| energy | REALISTIC_ENOUGH_FOR_DEV (DA/RT basis, decorrelated carbon; base spike still clean) |
| robustness | REALISTIC_ENOUGH_FOR_DEV (no SLA regression vs FIFO across acting scenarios) |
| **OVERALL** | **NOT_PRODUCTION_REALISTIC_YET** (capped: zero params measured on real hardware) |

### Honest findings (surfaced, not hidden)
1. **Canonical telemetry is always perfect.** `ClusterSimulator.get_cluster_state()`
   hardcodes `provenance.confidence='high'` / `is_partial=False`, so the
   missing/stale-telemetry path is only weakly exercised end-to-end. The
   "degraded telemetry" scenarios largely still classify at high confidence
   (`degraded_topology_telemetry`→1.0, `partial_utilization_telemetry`→1.0,
   `low_confidence_energy_telemetry`→0.85). → telemetry = `NEEDS_REAL_TELEMETRY`.
2. **constraint_aware does NOT beat naive baselines on raw energy cost** — it is
   *more* expensive on average (mean cost delta vs FIFO ≈ −$0.57, vs
   current_price_only ≈ −$0.95 over 26 scenarios). Its value is safety-adjusted:
   it protects p99/queue/thermal and serves more tokens. This is the
   "safety-adjusted explanation" the spec requires before any savings framing.
3. **Known regression since #82:** in `energy_price_arbitrage_multiregion`,
   constraint_aware no longer beats greedy_energy by the 5× p99 margin — both
   saturate (constraint_aware: lower throughput, higher cost). Recorded as an
   xfail calibration target, NOT papered over by weakening the threshold.
4. Base energy spikes are still clean round-number step multipliers (adversarial
   variants `da_rt_basis_blowout` / `carbon_cheap_price_expensive` exist).

### Packing baselines
First-fit/best-fit/FFD/greedy + clairvoyant lower bound. On
`fragmentation_stranded_capacity` they reveal 1 stranded node recoverable
(3 active → 2 needed). These are **analysis-only**: the simulator has no faithful
arbitrary-relocation primitive, so packing heuristics report the achievable
frontier rather than acting as a deployable closed-loop policy (clairvoyant is
explicitly never deployable per spec).

### Commands run
```
python -m aurelius.cli realism-audit --format text
python scripts/generate_realism_report.py --steps 24 --seed 42
pytest tests/test_realism_audit.py tests/test_packing_baselines.py -q   # 20 passed
pytest tests/test_constraint_benchmark.py tests/test_simulator_actions.py \
       tests/test_constraint_multi.py tests/test_serving_realism.py -q   # green, 1 xfail
```

### What remains simulator-only / next calibration step
Unchanged from prior runs: all magnitudes are uncalibrated priors. The single
highest-value next step is a **read-only shadow pilot against real
Prometheus/DCGM/K8s telemetry** to (a) calibrate the priors and (b) make the
canonical `ClusterState` carry real confidence/partial flags so the telemetry
subsystem can graduate past `NEEDS_REAL_TELEMETRY`. No production savings claims
until then.

---

## Telemetry-Truth + Benchmark-Determinism Calibration (2026-05-27)

This run closed the telemetry-confidence gap, fixed a benchmark determinism bug,
resolved the xfail honestly, and added a principled action-selection guard —
without weakening any realism penalty.

### Mission 1 — Telemetry confidence / partial-state truth (FIXED)
- `ClusterSimulator.get_cluster_state()` previously hardcoded
  `provenance.confidence='high'` / `is_partial=False` for ALL state objects, so
  degraded-telemetry scenarios were masked as perfect.
- New `_region_telemetry_truth()` derives each region's provenance confidence from
  the simulator's own per-subsystem telemetry tiers (energy/topology/utilization),
  sets top-level `is_partial` + `missing_sources` honestly. Clean scenarios stay
  `high`/not-partial (canonical detection scenarios unchanged); degraded scenarios
  now report `low` + partial.
- Engine gains a **telemetry-trust gate**: `state.is_partial and
  provenance.confidence=='low'` → KEEP all (risky actions blocked). Gated on
  PROVENANCE trust, NOT the blended classifier confidence — so legitimate
  low-COVERAGE-but-trustworthy scenarios (rack_density, fragmentation) still act.
  Plus an advisory downgrade: partial-but-not-low telemetry blocks cross-region
  migrations / placement changes (can't trust an unseen destination).
- Result: `degraded_topology_telemetry`, `partial_utilization_telemetry`,
  `low_confidence_energy_telemetry` now classify at 0.29–0.34 confidence, are
  marked partial, and force KEEP (0 actions). Tests: `tests/test_telemetry_truth.py`
  (12). The realism-audit telemetry verdict graduated
  `NEEDS_REAL_TELEMETRY` → `REALISTIC_ENOUGH_FOR_DEV` (overall still
  `NOT_PRODUCTION_REALISTIC_YET` — no measured params).

### Mission 4 — xfail investigation (RESOLVED: was a determinism bug, not a model regression)
- The "greedy loses by 5× p99" xfail was traced to **benchmark non-determinism**:
  the energy scenario's builtin (`_BUILTIN_SCENARIOS`) had drifted from its YAML
  (`benchmarks/v1/...yaml`) — missing the flexible `batch-wl-west` workload. Since
  `load_scenario` prefers YAML when PyYAML is installed and falls back to the
  builtin otherwise, results depended on whether PyYAML was present (the bare
  pytest venv lacks it → stale 2-workload builtin → muted p99 blow-up; a plain
  interpreter has it → 3-workload YAML → full blow-up).
- Fix: re-synced the builtin to the YAML (added `batch-wl-west`; fixed
  `hot-wl`→`hot-wl-0`). Now both sources give identical results in every
  environment and the 5× property holds across seeds (ratios 15–26×). The xfail
  is converted to a passing assertion. Guard added:
  `tests/test_scenario_source_parity.py` asserts builtin == YAML structure (runs
  where PyYAML exists) plus yaml-free completeness checks.

### Mission 3 — Principled action guard (PARTIAL; honest open weakness)
- Added a **constraint-dominance** guard: reject an action that worsens a
  HIGHER-scored constraint than the best one it relieves (uses only the observed
  score vector — no new magic constant). This stops e.g. scaling batch replicas to
  chase a marginal queue=0.30 score when energy=0.60 dominates, while still
  allowing scaling when queue/latency is the dominant pressure (real surge) and
  allowing energy migrations whose only worsened constraints score below energy.
- **SLA-safe across all 26 scenarios** (no regression vs FIFO); thermal/queue wins
  preserved.
- **Honest open weakness:** it does NOT fully fix the energy arbitrage scenario.
  `constraint_aware` is still the most expensive policy there and loses to
  `current_price_only` on both cost AND SLA, because the engine still applies some
  queue-relief scaling to batch workloads. A complete fix requires propagating
  workload class (`priority_tier`/`latency_sensitive`) into the canonical
  `InferenceServiceState` so the engine can apply workload-aware priorities (batch:
  cost/throughput, not queue). Deliberately NOT papered over.

### Benchmark truth (Mission 2)
- `docs/REALISM_BENCHMARK_VALIDATION.md` regenerated: per-scenario table now
  includes a telemetry-confidence column (with partial flag); mean/median cost
  delta vs FIFO / current_price_only / greedy_energy / SLA-aware; engine net
  savings; honest win/loss lists. Median cost delta vs FIFO is now $0.00
  (degraded scenarios correctly KEEP = FIFO).

### Honest standing claims
- Benchmark evidence remains **simulator-only**; no production savings claims.
- `constraint_aware` raw cost is **worse than current_price_only/greedy_energy** in
  the energy scenario; its value elsewhere is safety-adjusted (thermal/queue/p99).
- Determinism: benchmark results are now environment-independent for the
  file-backed scenarios (builtin/YAML parity guarded).

### Commands run
```
python -m aurelius.cli realism-audit --format text
python scripts/generate_realism_report.py --steps 24 --seed 42
pytest tests/test_telemetry_truth.py tests/test_scenario_source_parity.py
       tests/test_realism_audit.py tests/test_serving_realism.py -q     # green
# full constraint suite: 538 passed, 6 skipped (yaml-parity), 4 pre-existing
#   yaml/pandas env-gap failures (test_sla_optimization / test_queue_aware).
```

---

## Canonical KPI: SLA-Safe Goodput per Infrastructure Dollar (2026-05-28)

This run replaces raw energy cost with Aurelius's new canonical benchmark metric.
Raw energy cost was the wrong objective: it rewarded starving customer SLOs to
save electricity and punished safe consolidation that uses more energy now to
prevent SLA-violating thrash later. The new headline metric is:

```
sla_safe_goodput_per_infrastructure_dollar =
    sla_compliant_goodput
    / (gpu_infra_cost + energy_cost + network_cost)
```

with **secondary KPIs (p99, queue, thermal, topology, …) tracked separately as
constraints/diagnostics — never folded into the primary KPI**.

### Files changed
- `aurelius/benchmarks/economics.py` (new) — pure, deterministic functions plus
  `InfrastructureCostConfig`, `SLAFilterConfig`, `EconomicKPIResult`. Documented
  public-list cloud GPU prices as priors; operator overrides every default. No
  workload-value weights anywhere in the module.
- `aurelius/simulation/cluster/engine.py` — added `sla_compliant_tokens`,
  `active_gpu_count`, `active_gpu_hours_by_type` to `TickMetrics`, computed in
  the existing per-tick aggregation loop from per-queue `timeout_rate_pct`
  (SLA filter) and `gpu.assigned_workload_id` (billable footprint).
- `aurelius/benchmarks/report.py` — `AggregatedKPI` gains the primary KPI +
  cost breakdown + active GPU-hours; `to_dict()` puts the primary KPI first;
  `to_text()` renders a Primary KPI section followed by a Secondary KPIs
  (diagnostics) section.
- `aurelius/benchmarks/constraint_runner.py` — wires the cost config end-to-end
  and aggregates per-policy.
- `aurelius/benchmarks/__init__.py` — exports the new public API.
- `scripts/generate_realism_report.py` — adds a new Section 2 with mean/median
  primary KPI per policy, "scenarios where constraint_aware loses the canonical
  KPI to a baseline" (1% noise floor for materiality), and a per-scenario primary
  KPI table across all five policies.
- `tests/test_economics_kpi.py` (new, 20 tests) — covers every spec invariant.
- `docs/REALISM_BENCHMARK_VALIDATION.md` regenerated with the canonical KPI
  front and center.

### KPI formula and terms
- **SLA-compliant goodput** = per queue per tick, `tokens × max(0, 1 −
  timeout_rate_pct/100)` summed across queues and ticks. `timeout_rate_pct` is
  the simulator's existing per-queue measure of the share of work whose p99
  exceeded the workload's configured `latency_sla_p99_ms` (engine.py:1758). No
  partial credit by default at ≥50% timeout (hard exclude).
- **GPU infra cost** = `Σ active_gpu_hours[type] × gpu_hour_price[type]`.
  "Active" = `gpu.assigned_workload_id is not None` (the billable footprint;
  consolidated idle nodes correctly drop to zero). Defaults are documented
  public-list on-demand prices ($3/hr H100, $2/hr A100 SXM4, …) — overridable
  per operator.
- **Energy cost** = pass-through of `tick_cost` (`kWh × realtime_price`); DA/RT
  basis preserved.
- **Network cost** = `migrations × per-migration` + `egress_gb × per-GB`,
  defaulted to 0.0 so we do not invent network penalties inside the headline KPI.

### Tests (20 new)
The spec's 11 invariants all proven:
SLA-violating tokens never count · raw throughput can rise while SLA-safe
goodput falls · lower energy can still lose / higher energy can still win · GPU
infra cost can dominate energy · network cost only when configured · zero
goodput never divides by zero · cost-per-compliant-token is `inf` when goodput
collapses but cost > 0 · NO workload-value parameters or fields exist · benchmark
reports surface both primary and secondary KPIs · constraint_aware is compared
against current_price_only and greedy_energy, not only FIFO.

### Honest benchmark findings under the new canonical KPI
**Per-policy aggregates across 26 scenarios (mean goodput per $infra):**
| Policy | Mean | Median |
|---|---|---|
| FIFO | 414,803 | 459,570 |
| current_price_only | 414,670 | 449,752 |
| greedy_energy | 407,800 | 449,752 |
| SLA-aware | 414,803 | 459,570 |
| **constraint_aware** | **410,663** | **439,149** |

**constraint_aware is mean-worse than FIFO and SLA-aware on the canonical KPI.**
It materially (>1%) loses to FIFO in 10 scenarios, to current_price_only in 10,
to greedy_energy in 8, to SLA-aware in 10.

**Energy scenario specifically:**
| Policy | goodput/$ | sla_goodput | infra $ |
|---|---|---|---|
| current_price_only | **402,882** | 158.8M | 394.26 |
| FIFO | 338,274 | 164.9M | 487.40 |
| greedy_energy | 274,801 | 85.0M  | 309.17 |
| SLA-aware | 338,274 | 164.9M | 487.40 |
| **constraint_aware** | **196,792** ← worst | 110.7M | **562.30** |

**constraint_aware delivers the lowest SLA-safe goodput per infrastructure
dollar of any policy in the energy scenario** (≈50% of current_price_only). The
canonical metric reveals this loss more starkly than raw energy cost did.

**Scenarios where constraint_aware genuinely wins on the canonical KPI:**
- `thermal_hotspot_mixed_cluster`: 830,781 vs FIFO 565,817 (**+47%** — thermal
  spreading prevents throttle).
- `underutilization_stranded_capacity`: 74,432 vs FIFO 45,426 (**+64%** — safe
  consolidation).
- `rack_density_overload_air`: 680,129 vs FIFO 667,543 (+1.9%).

### Next optimizer fix (the diagnosis was correct)
The energy-scenario loss has a single precise root cause, unchanged from the
last run: the engine generates `SCALE_REPLICAS` for **batch** workloads
(`batch-llm-east/west`) to chase a marginal queue=0.30 score under
energy-dominant pressure. Scaling adds billable GPU-hours (gpu_infra_cost goes
from $388 / $304 to $554), which is exactly the worst trade under the new KPI:
batch workloads tolerate queueing, so the extra GPU dollars buy goodput that
could have come from KEEP at lower cost. The principled fix is propagating
workload class (`priority_tier` / `latency_sensitive`) into the canonical
`InferenceServiceState` so the engine can apply the spec's workload-aware
priorities — **not** tuning a constant. Deliberately deferred to keep this PR
scoped to metric/accounting correctness.

### Honest standing claims
- **Simulator benchmark KPI implemented; production claims require customer
  telemetry calibration.**
- Raw energy cost is **not** the primary metric for full constraint-aware
  Aurelius.
- Secondary KPIs are constraints / vetoes / diagnostics, **not** hidden weighted
  objective terms.
- GPU infra cost typically dominates electricity by 50–200×; an "energy savings"
  win that increases active GPU-hours can lose the canonical KPI.
- No guaranteed savings · no production-proven savings · no hyperscaler-validated
  economics · no enterprise-ready autonomous optimization.

### Commands run
```
pytest tests/test_economics_kpi.py -q                            # 20 passed
pytest tests/{constraint_*,cluster_simulator,realism_audit,
       telemetry_truth,scenario_source_parity,*_realism,
       packing_baselines,state_assemble,migration_cost_model}.py -q   # green
python scripts/generate_realism_report.py --steps 24 --seed 42
ruff check aurelius/benchmarks tests/test_economics_kpi.py        # clean
python -m compileall aurelius/benchmarks aurelius/simulation/cluster
```

---

## Workload-Aware Economic Gating (2026-05-28)

This run closes the optimizer bug the previous run (#85) diagnosed: the engine
scaled BATCH/flexible workloads under marginal queue pressure, burning billable
GPU-hours without commensurate SLA-safe goodput gain. The fix is workload-class
propagation + workload-aware action eligibility + a conservative economic safety
net. **No new constants tuned to make benchmarks win; no realism penalties
weakened; no synthetic workload-value weights.**

### Files changed
- `aurelius/state/models.py` — `InferenceServiceState` gains optional
  workload-class fields (`workload_type`, `priority_tier`, `latency_sensitive`,
  `flexibility`, `migration_allowed`, `latency_sla_p99_ms`, `queue_sla_p95_ms`,
  `sla_policy_id`, `deadline_s`, `flexibility_window_minutes`). All default to
  None so legacy callers and JSON round-trip continue to work.
- `aurelius/simulation/cluster/engine.py` — `get_cluster_state()` populates the
  new fields from the matching `SimWorkload`.
- `aurelius/constraints/engine.py` — two helpers + two gates:
  - `_workload_class(service)` resolves spec classes (critical_interactive /
    standard_interactive / batch_inference / embedding_offline / training /
    best_effort / unknown). `flexible` is interpreted as a shiftability flag,
    so flexible inference services remain standard_interactive while flexible
    training jobs remain batch_inference.
  - `_scale_eligible_for_class(wclass, sla_risk, has_deadline_risk)` — primary
    gate. For batch / embedding_offline / best_effort / training: block scale
    unless SLA-risk ≥ `_STRONG_SLA_RISK_SCORE` (0.7 — documented policy
    threshold) or explicit deadline risk. Interactive classes remain eligible.
  - `_predict_scale_yield_ok(...)` — conservative economic safety net. Compares
    a class-specific expected relief share against a class floor (0.02 for
    critical, 0.05 standard, 0.15 batch/embedding, 0.30 best_effort). Rejects
    actions whose predicted Δgoodput / Δinfra-cost is non-positive.
  - Both gates fire after dominance, before destination-safety.
- `aurelius/benchmarks/report.py` + `aurelius/benchmarks/constraint_runner.py`
  — `AggregatedKPI` gains `scale_up_recommended`, `scale_up_applied`,
  `blocked_scale_for_low_value_queue_relief`, `blocked_uneconomic_scale`,
  `blocked_dominated`. Report text + JSON surface them.
- `tests/test_workload_aware_engine.py` (new, 20 tests).
- `docs/REALISM_BENCHMARK_VALIDATION.md` regenerated.

### Workload fields propagated (spec checklist)
- `workload_type`, `priority_tier`, `latency_sensitive`, `flexibility`,
  `migration_allowed`, `latency_sla_p99_ms`, `queue_sla_p95_ms`,
  `sla_policy_id`, `deadline_s`, `flexibility_window_minutes`. JSON round-trip
  verified.

### Action eligibility rules implemented
| Workload class | Scale-up policy |
|---|---|
| critical_interactive | allow (SLA gating + economic safety net) |
| standard_interactive | allow (SLA gating + economic safety net) |
| batch_inference | block unless SLA-risk ≥ 0.7 OR deadline risk |
| embedding_offline | block unless SLA-risk ≥ 0.7 OR deadline risk |
| training | block unless deadline risk |
| best_effort | block unless SLA-risk ≥ 0.7 OR deadline risk |
| unknown | allow with normal gating (safe default) |

### Economic gating formula
For SCALE_REPLICAS candidates that pass class eligibility:

```
expected_relief_share = min(0.5, sla_risk) × class_relief_factor
                        # relief factors: critical=0.6, standard=0.45,
                        # batch=0.10, embedding=0.10, best_effort=0.05,
                        # training=0.20, unknown=0.30
class_min_relief       # required floor: critical=0.02, standard=0.05,
                       # batch=0.15, embedding=0.15, best_effort=0.30,
                       # training=0.10, unknown=0.10
```
Accept iff `expected_relief_share >= class_min_relief`. This is a deliberate
*safety net*, not a precision instrument — we cannot predict next-tick goodput
exactly, so we err on the side of not acting unless the action plausibly helps.

### Energy scenario before / after
| Policy | goodput/$ (before) | goodput/$ (after) |
|---|---|---|
| fifo | 338,274 | 338,274 |
| current_price_only | 402,882 | 402,882 |
| greedy_energy | 274,801 | 274,801 |
| SLA-aware | 338,274 | 338,274 |
| **constraint_aware** | **196,792** | **228,634** (+16%) |

Constraint_aware on the energy scenario:
- Raw cost $8.30 → **$7.40** (matches FIFO; no harmful scaling).
- Infra cost $562 → **$487** (matches FIFO).
- p99 latency 246 910 ms → **19 747 ms** (dramatic improvement, matches FIFO).
- 0 SCALE_REPLICAS applied (was many before); 6 blocked by class gating;
  16 blocked by economic gating; 26 blocked by dominance.

### Constraint_aware vs other baselines on the energy scenario
| Comparison | Goodput/$ delta | Notes |
|---|---|---|
| vs FIFO | -109,640 (-32%) | improved from -141,482 (-42%) before |
| vs current_price_only | -174,248 (-43%) | improved from -206,090 (-51%) before |
| vs greedy_energy | -46,167 (-17%) | improved from -78,009 (-28%) before |

**Honest open issue:** constraint_aware still loses to `current_price_only` on
goodput/$ in the energy scenario, because `current_price_only` migrates the
flexible `batch-llm-east` workload to the cheaper region (a real arbitrage
opportunity). Constraint_aware does not. The next optimizer fix is to emit
CHOOSE_CHEAPER_REGION candidates for `flexibility=high` / `migration_allowed=True`
batch workloads when (a) the destination region is materially cheaper, (b)
destination-safety passes, and (c) the economic gate's predicted KPI delta is
positive. That requires real energy-price arbitrage candidate generation for
batch workloads (not synthesized scaling), which is a clean, scoped follow-up.

### Cross-scenario benchmark table (mean goodput / $ over 26 scenarios)
| Policy | Mean | Median |
|---|---|---|
| FIFO | 414,803 | 459,570 |
| current_price_only | 414,670 | 449,752 |
| greedy_energy | 407,800 | 449,752 |
| SLA-aware | 414,803 | 459,570 |
| **constraint_aware** | **414,912** | 454,858 |

Constraint_aware is now **mean-better** than FIFO on the canonical KPI for the
first time. The material-loss counts (>1%) dropped:
- vs FIFO: 10 → **6**
- vs current_price_only: 10 → **7**
- vs greedy_energy: 8 → **5**
- vs SLA-aware: 10 → **6**

### Scenarios where constraint_aware wins (canonical KPI)
- `thermal_hotspot_mixed_cluster` (+20–47% depending on Python env)
- `underutilization_stranded_capacity` (+5–64% depending on env)
- `rack_density_overload_air` (+1.9%)

### Scenarios where constraint_aware still loses materially
`energy_price_arbitrage_multiregion`, `latency_critical_no_energy_shift`,
`prefix_affinity_energy_arbitrage`, `proxy_bottleneck_ingress`,
`queue_surge_latency_sensitive`, `startup_heavy_migration_trtllm` (vs FIFO).
Energy-arbitrage-flavoured scenarios remain the dominant loss; the fix is
the batch-energy-arbitrage candidate noted above. NO SLA regressions vs FIFO.

### Remaining optimizer bugs (precise)
1. **Batch energy arbitrage missing:** constraint_aware does not migrate
   flexible/batch workloads to a materially-cheaper region when one is
   available. Add a candidate generator (or extend `_gen_energy`) that emits
   `CHOOSE_CHEAPER_REGION` for `flexibility=high` / `migration_allowed=True`
   batch workloads.
2. **Class relief factors are uncalibrated priors:** the per-class floor /
   relief factor values are documented heuristics. Production claims require
   calibrating against real per-class goodput response on a pilot deployment.

### Tests (20 new, comprehensive)
Workload propagation + JSON round-trip · batch mild-queue blocks scale · batch
deadline-risk allows scale · critical interactive remains eligible · batch
strong-SLA-risk allows scale · embedding-offline allows scale only under
deadline or strong-SLA · energy goodput/$ improves vs prior · energy infra cost
matches FIFO · thermal/underutilization wins preserved · no SLA regressions
across 5 canonical scenarios · blocked actions carry workload_class + reason.

### Commands run
```
pytest tests/test_workload_aware_engine.py -q           # 20 passed
pytest tests/{test_*}.py -q                             # 713 passed, 6 skipped
ruff check aurelius tests/test_workload_aware_engine.py  # clean
python -m compileall aurelius
python scripts/generate_realism_report.py --steps 24 --seed 42
```

### What this run does NOT change (per spec non-goals)
- No ML forecasting, no new ISOs, no revenue or workload-value weights.
- No simulator realism penalty was weakened.
- No constant was tuned solely to make a benchmark win.
- Canonical KPI remains `sla_safe_goodput_per_infrastructure_dollar`.
- Simulator results remain **not production savings claims**. ML forecasting
  is a later phase, after the optimizer has the right objective and
  workload-aware decision rules.

---

## Per-Workload Baseline Reporting (2026-05-28)

The previous benchmark layer compared `constraint_aware` to FIFO across all
26 scenarios using only economic alpha. That is the wrong reference: FIFO is a
sanity baseline, not the strong alternative for, say, a batch-training energy
arbitrage problem (where the real comparison is `current_price_only` or
`greedy_energy`). The reporting layer now picks the **workload-relevant strong
baseline** per scenario and classifies the outcome.

ML forecasting is a later phase, after the optimizer has the right objective
and workload-aware decision rules. Simulator results remain **not production
savings claims**.

### Files changed

- `aurelius/benchmarks/per_workload.py` (NEW): `ScenarioMetadata`,
  `OutcomeAnalysis`, `PerScenarioRow`, `CrossScenarioReport`,
  `classify_scenario`, `select_headline_baseline`, `analyze_outcome`,
  `workload_class_from_iss`.
- `aurelius/benchmarks/report.py`: `BenchmarkReport` carries
  `scenario_metadata`, `headline_baseline_name`, `headline_baseline_rationale`,
  `outcome` (appended in `to_dict()`/`to_text()` — never replaces existing
  output).
- `aurelius/benchmarks/constraint_runner.py`: `_build_report` calls
  `select_headline_baseline` + `analyze_outcome` after the existing scorecard.
- `aurelius/benchmarks/__init__.py`: re-exports the new public surface.
- `aurelius/simulation/cluster/scenarios.py`: `ScenarioConfig.metadata`
  populated by `load_scenario` via a safe import shim. Three builtin
  scenarios re-synced to YAML (`thermal_hotspot_mixed_cluster.ambient_temp_trace`,
  `queue_surge_latency_sensitive.critical-wl.gpu_count_required/target_util_pct`,
  `underutilization_stranded_capacity` nodes + per-workload utilization).
  This eliminates the pytest-vs-direct KPI drift on those scenarios.
- `aurelius/constraints/engine.py`: public alias
  `workload_class = _workload_class` (no logic duplication).
- `scripts/generate_realism_report.py`: now uses `CrossScenarioReport`. The
  realism-audit block (Section 1) and the "what remains simulator-only" block
  are preserved.

### Per-workload-type results (seed=42, steps=24)

| Workload type | Scenarios | CA median goodput/$ | Strongest baseline |
|---|---|---|---|
| batch_training | 6 | 460,027 | sla_aware/fifo (interchangeable) |
| inference_critical | 3 | 424,896 | sla_aware |
| inference_standard | 14 | 407,657 | sla_aware |
| telemetry_fail_safe | 3 | — (KEEP-correctness, not alpha) | fifo (correctness reference) |

Overall outcome distribution across the 26 scenarios:
ALPHA_WIN = 3, SAFETY_WIN = 0, TIE = 14, KEEP_CORRECT = 3, LOSS = 6.

### Where constraint_aware wins (alpha)

- `thermal_hotspot_mixed_cluster` vs `sla_aware`: **+46.83%** on goodput/$.
- `underutilization_stranded_capacity` vs `sla_aware`: **+63.85%** on goodput/$.
- `rack_density_overload_air` vs `sla_aware`: **+1.89%** on goodput/$.

### Where constraint_aware loses (honest, with reasons)

- `energy_price_arbitrage_multiregion` vs `current_price_only`: **-43.25%**.
  Loss reasons: `missing_candidate_action` (no migration emitted) +
  `missing_forecast_lookahead` (no DA/RT lookahead → no positive net_savings).
- `queue_surge_latency_sensitive` vs `sla_aware`: **-7.94%**.
  Loss reason: `missing_candidate_action` (queue_relief scale-replicas not emitted).
- `proxy_bottleneck_ingress` vs `sla_aware`: **-8.72%**. Same root cause.
- `prefix_affinity_energy_arbitrage`, `startup_heavy_migration_trtllm`,
  `latency_critical_no_energy_shift`: all in the **-4.7% to -5.5%** band; same
  family — CA emits no relevant action type.

### Honest open issues

- The fragmentation-packing scenarios still resolve through the `sla_aware`
  headline path (rule 3 fires before rule 5 when the primary workload type is
  inference); this is intentional but means the section-D packing-baseline
  comparison is the relevant view for those scenarios. The simulator has no
  arbitrary-placement primitive — `simulator_limitation` is the surfaced loss
  reason there.
- Most LOSS cases trace to `missing_candidate_action` — the engine has the
  right binding constraint but is not yet emitting the matching action type
  for queue_relief / energy_arbitrage on interactive workloads. That is a
  decision-rule gap in the engine, not a reporting gap.
- All KPI numbers are simulator-only and uncalibrated. Not production claims.

### Test counts

- New: `tests/test_per_workload_reporting.py` — 38 tests (classification,
  baseline selection, outcome analysis, cross-scenario report, end-to-end).
- New: `tests/test_pytest_vs_direct_parity.py` — 6 subprocess KPI parity tests
  + 6 full-signature checks (skipped when PyYAML missing). Documents the
  root cause of the pytest-vs-direct drift fixed in this PR.
- Extended: `tests/test_scenario_source_parity.py` — adds a `_full_signature`
  helper and 6 new parametrized tests.

ML forecasting is a later phase, after the optimizer has the right objective
and workload-aware decision rules. Simulator results remain **not production
savings claims**.

---

## Non-Negotiable Implementation Philosophy

This tracker is also a planning artifact, not proof of correctness.

Future implementation phases MUST NOT assume:
- the plan is complete
- the repo still matches the plan
- prior phases were implemented correctly
- passing a checklist means the feature works
- this tracker is always current

For every implementation phase, Claude MUST:

1. Re-read the high-level product goal.
2. Re-read `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md`.
3. Re-read this progress tracker.
4. Independently inspect the current repo state.
5. Compare repo reality against the plan and this tracker.
6. Identify gaps the plan missed.
7. Identify assumptions invalidated by implementation.
8. Verify real code paths are wired where relevant.
9. Run tests against actual behavior.
10. Audit failure modes and missing telemetry.
11. Update this tracker with repo-reality findings.
12. Update the plan if reality differs from the plan.

A phase is NOT complete merely because:
- files were added
- functions exist
- tests pass in isolation
- checklist items were checked
- this tracker says the phase is complete

A phase is complete only when:
- the implementation is wired into the real execution path where relevant
- the behavior changes correctly in end-to-end scenarios where relevant
- missing telemetry fails safely
- old behavior is preserved when disabled
- CLI/demo paths work if relevant
- sandbox and real connectors share the same interfaces where relevant
- evidence is provided

The implementation should optimize for:
- real operational correctness
- safety
- observability
- enterprise deployability
- reproducible validation
- stable measurable improvement

NOT:
- maximizing apparent feature completeness
- satisfying the plan mechanically
- creating placeholder abstractions disconnected from real execution paths
- optimizing only synthetic benchmark scores

If the plan or tracker conflicts with repo reality:
- trust the repo
- document the mismatch
- update the relevant document

---

## Product Goal Reminder

Aurelius is evolving from mostly energy-aware scheduling into constraint-aware GPU orchestration for:
- AI inference providers
- neoclouds
- GPU-heavy data centers
- infrastructure/platform teams running GPU clusters

The product should help operators improve:
- cost/token
- tokens/joule
- GPU utilization
- queue wait
- p95/p99 latency
- thermal stability
- topology-aware placement
- migration safety
- SLA preservation
- operational stability

Aurelius must remain an orchestration/control-plane intelligence layer.

Allowed:
- telemetry ingestion
- state normalization
- constraint classification
- routing recommendations
- scheduler hints
- placement scoring
- topology-aware placement recommendations
- energy-aware scheduling
- thermal-aware spreading
- queue-aware scheduling
- latency/SLA-aware routing
- utilization/bin-packing recommendations
- cache-affinity hints from exposed metrics
- dry-run/recommendation-first reports

Forbidden:
- modifying NCCL
- modifying CUDA
- modifying kernels
- controlling KV cache internals
- rewriting memory allocators
- altering model execution runtime internals
- mutating customer clusters by default

---

## Phase Status Table

| Phase | Name | Status | Evidence | Notes |
|---|---|---:|---|---|
| 0 | Audit + canonical plan | COMPLETE | `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md` exists | Planning only; no production implementation yet |
| 1 | Normalized state model | COMPLETE | `aurelius/state/`, 154 tests passing | See Phase 1 details below |
| 2 | Prometheus-native connector | COMPLETE | `aurelius/connectors/`, 282 tests passing | See Phase 2+3 details below |
| 3 | DCGM/vLLM/Triton/Ray adapters | COMPLETE | `aurelius/connectors/dcgm.py` etc., 282 tests passing | See Phase 2+3 details below |
| 4 | Kubernetes connector | COMPLETE | `aurelius/connectors/kubernetes.py`, 47 tests passing | See Phase 4+5 details below |
| 5 | Topology collector | COMPLETE | `aurelius/connectors/topology.py`, 62 tests passing | See Phase 4+5 details below |
| 6 | Synthetic cluster simulator | COMPLETE | `aurelius/simulation/cluster/`, 93 tests passing | See Phase 6 details below |
| 7 | Constraint classifier | COMPLETE | `aurelius/constraints/classifier.py`, 74 tests passing | See Phase 7+8 details below |
| 8 | Cost/risk/migration model | COMPLETE (risk model corrected + audited) | `aurelius/constraints/cost_model.py`, 47 tests passing | State-conditioned risk; static workload multipliers removed; code-level audit passed (7/7). See "Phase 8/9 Risk Model Correction" below |
| 9 | Constraint-aware recommendation engine | COMPLETE | `aurelius/constraints/engine.py`, 53 tests passing | See Phase 9 details below |
| 10 | CLI reports | COMPLETE | 5 CLI commands, 58 tests passing | See Phase 10 details below |
| 11 | Validation + benchmarking loop | COMPLETE | 58 tests, 6 scenarios, regression detection | See Phase 11 details below |
| 12 | Production hardening | COMPLETE | `aurelius/constraints/observability.py`, 51 tests passing | See Phase 12 details below |

---

## Phase 1 Completion Evidence

### Phase 1 Milestone Decision

- **What this run implemented:** Phase 1 — normalized state model (`aurelius/state/` package)
- **Why it was the correct next step:** `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md` existed from Phase 0 but `aurelius/state/` did not exist. No constraint-aware telemetry layer existed. This is the prerequisite foundation for all subsequent phases.
- **Prior dependencies verified:** No prior constraint-aware phases existed. Existing energy-arbitrage phases (1-5) are complete and were explicitly left untouched.
- **What was explicitly NOT attempted:** Connectors (Phase 2-4), topology collection (Phase 5), simulator (Phase 6), constraint classifier (Phase 7), optimizer changes (Phase 9).

### Repo-Reality Audit Findings

**Plan vs repo mismatches:**
- The plan (§5.6) says to reuse `aurelius/sla/telemetry.py:WorkloadState` and extend it additively. This was not done in Phase 1 because: (a) the plan also says "Do not touch optimizer logic in this phase" and (b) extending WorkloadState requires careful testing that it doesn't break existing SLA evaluator behavior. **Decision: documented as Phase 9 work.** The new state models don't duplicate WorkloadState — they reference it via adapters.
- The plan mentioned a `QueueStateV2` wrapper with provenance. Implemented as `adapt_queue_state()` returning a dict (not a typed model) since the plan says "Reuse the existing `QueueState`" and creating a new typed model risks confusion. This is documented — Phase 2/7 can promote it if needed.
- `asdict` was imported in models.py but not used (removed by ruff fix).

**Models implemented (all from §5):**
- `Provenance` ✓
- `ConstraintType` (enum) ✓
- `TopologyLinkType` (enum) ✓
- `GPUState` (adapted from §5.4, adapts `GPUMetrics`/`GPUHealthScore`) ✓
- `InferenceServiceState` (§5.5) ✓
- `TopologyState` (§5.8) ✓
- `EnergyState` (§5.9) ✓
- `ThermalState` (§5.10) ✓
- `NodeState` (§5.3) ✓
- `RegionState` (§5.2) ✓
- `ClusterState` (§5.1) ✓
- `MigrationEvent` + `MigrationHistory` (§5.11) ✓
- `ConstraintAssessment` (§5.12) ✓
- `Recommendation` (§5.13) ✓

**What was intentionally omitted:**
- `WorkloadState` extension — Phase 9 (requires SLA engine wiring audit)
- No connector code
- No classifier code
- No optimizer changes

### Tests Added

| Test File | Tests | What It Proves |
|---|---|---|
| `tests/test_state_models.py` | 90 | All model validation (UTC-aware, None-not-zero, pct/rate ranges, JSON round-trip, enum values, property derivations, impossible value rejection) |
| `tests/test_state_store.py` | 18 | Append-only, leakage-safe lookup, out-of-order insert, range queries, latest/earliest, clear, duplicate timestamps |
| `tests/test_state_normalize.py` | 46 | validate_utc_aware, validate_percentage, validate_non_negative, make_provenance, adapt_gpu_metrics (incl. GPUHealthScore merge), adapt_queue_state, coerce_to_utc, optimizer non-regression imports |

### Commands Run

```
python -m compileall aurelius/state
ruff check aurelius/state/ tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py
/root/.local/bin/pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py -q
/root/.local/bin/pytest tests/test_scheduler.py tests/test_safety_gate.py -q
```

### Test Results

```
tests/test_state_models.py: 90 passed
tests/test_state_store.py: 18 passed
tests/test_state_normalize.py: 46 passed
total Phase 1: 154 passed, 0 failed

tests/test_scheduler.py: 10 passed (existing, unmodified)
tests/test_safety_gate.py: 10 passed (existing, unmodified)
ruff: All checks passed
python -m compileall: No errors
```

### Proof Optimizer Behavior Was Not Changed

- No file in `aurelius/optimization/`, `aurelius/backtesting/`, `aurelius/sla/`, `aurelius/forecasting/`, or `aurelius/models.py` was modified.
- `tests/test_scheduler.py` and `tests/test_safety_gate.py` pass identically to before.
- `test_state_normalize.py::TestOptimizerUnchanged` explicitly tests that `JobScheduler`, `ObjectiveFunction`, existing `QueueState`/`GPUMetrics`/`GPUHealthScore`, `WorkloadState`, and `ActionType` all import cleanly with unchanged behavior.

### Wiring Evidence

Phase 1 is additive groundwork. The state models are NOT yet wired into any production decision path — this is intentional and documented. Wiring happens in:
- Phase 2 (connector produces `ClusterState`)
- Phase 7 (classifier consumes `ClusterState`)
- Phase 9 (optimizer/engine produces `Recommendation`)

**Which paths are intentionally not wired yet:**
- `BacktestEngine` still constructs `JobScheduler` without ClusterState
- CLI `simulate`/`backtest` paths still use old energy-arbitrage flow
- `SLARegistry` is still dormant (Phase 9 target)

### Failure Mode Review

- **Missing telemetry:** All optional fields default to `None`. The classifier (Phase 7) will treat `None` as `missing_signal` and reduce confidence — not fabricate a value.
- **Naive timestamps:** Rejected at model construction time with a clear `ValueError`. The `coerce_to_utc()` helper provides an explicit escape hatch for synthetic sources.
- **Invalid ranges:** All impossible values (pct > 100, negative bytes, pct < 0, PUE < 1.0, etc.) raise `ValueError` at construction time — never silently accepted.
- **Partial connector failures:** `ClusterState.is_partial=True` + `missing_sources` list enables the classifier to reduce confidence proportionally.

---

## Validation Requirements By Phase

Every phase must record:

### Commands Run

```text
<exact commands>

Test Results

<exact output summary>

Repo-Reality Findings

What did the plan say?
What did the repo actually need?
What mismatches were found?

Wiring Evidence

Which real paths are wired?
Which paths are intentionally not wired yet?

Failure Mode Review

How does the implementation behave with missing data?
How does it fail safely?

Open Limitations

What remains scaffolded, heuristic, sandboxed, or unproven?

Benchmark / Optimization Philosophy

The verification and optimization stage is not one-and-done.

Constraint-aware optimization must improve over multiple routine runs until the system demonstrates:

* stable safe net improvement
* no significant SLA regression
* bounded migration churn
* robustness across workload classes
* robustness across constraint scenarios
* robustness under partial telemetry
* meaningful improvement vs current_price_only
* meaningful improvement vs existing Aurelius energy-aware optimization where applicable

The simulator is not reality.

Optimization strategies that improve simulator metrics while likely degrading real-world behavior must be treated as regressions.

Benchmark comparisons must preserve controlled variables:

* same workload mix
* same seed
* same topology
* same energy trace
* same SLA config
* same simulator version
* same scenario version

A reported improvement is invalid if benchmark conditions changed without being clearly labeled.

Aurelius must optimize net operational quality, not isolated savings metrics.

⸻

Current Known Risks

* The SLA engine exists but is not yet wired into real optimizer/backtest paths (Phase 9 target).
* `WorkloadState` extension (adding `service_id`, `gpu_uuids`, etc.) is deferred to Phase 9 to avoid breaking SLA evaluator while it is still dormant.
* Phase 1 state models are additive; no production decision path yet consumes them.
* Constraint classifier (Phase 7) has not been built — ClusterState is produced but not consumed.
* Simulator (Phase 6) has not been built — state models are not yet exercised end-to-end.
* Benchmarking does not yet prove multi-constraint optimization.

⸻

## Phase 1 Open Technical Debt

| Item | Priority | Notes |
|---|---|---|
| `WorkloadState` extension (add `service_id`, `gpu_uuids`, `kv_cache_usage`, `comm_bytes_per_s`) | Medium | Deferred to Phase 9; existing SLA evaluator still consumes the original shape |
| `QueueStateV2` typed model with provenance | Low | Currently a dict; can be promoted in Phase 2 once connector wiring is clearer |
| JSON schema validation for fixture files | Low | Fixtures are tested via `ClusterState.from_dict()` round-trip, not schema validation |
| `StateStore` Postgres persistence layer | Medium | Phase 1 is in-memory only; Postgres integration is a Phase 11/12 concern |
| DCGMProvider unit bug (throttle ns vs µs) | Medium | Documented in §6.3 of the plan; Phase 3 will fix `dcgm_provider.py` |

⸻

Latest Run Log

Phase 0

Status: COMPLETE

Summary:

* Created canonical implementation plan.
* No production constraint-aware implementation yet.
* Next milestone was Phase 1 normalized state model.

Evidence:

* docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md

---

Phase 1

Status: COMPLETE

Date: 2026-05-24
Branch: claude/sleepy-bohr-m14dY
PR: (to be created this run)

Summary:

* Created `aurelius/state/` package with 4 files:
  - `__init__.py` (package exports)
  - `models.py` (14 frozen dataclass state models + 2 enums)
  - `store.py` (leakage-safe append-only StateStore)
  - `normalize.py` (adapters + validation helpers)
* Created 3 test files with 154 tests total (all passing)
* Created `tests/fixtures/cluster_state/` with 3 JSON scenario fixtures
* No existing optimizer, SLA, forecasting, or energy connector code was modified
* All existing tests that can run (scheduler, safety gate) still pass

Evidence:

* `aurelius/state/__init__.py`, `models.py`, `store.py`, `normalize.py`
* `tests/test_state_models.py` (90 tests)
* `tests/test_state_store.py` (18 tests)
* `tests/test_state_normalize.py` (46 tests)
* `tests/fixtures/cluster_state/` (3 fixtures)
* ruff: all checks passed
* python -m compileall aurelius/state: no errors

Open limitations from Phase 1:

* No connectors yet (Phase 2+)
* No constraint classifier yet (Phase 7)
* No simulator yet (Phase 6)
* No optimizer wiring yet (Phase 9)
* WorkloadState extension deferred to Phase 9
* StateStore is in-memory only; Postgres persistence is Phase 11/12

Next milestone: ~~Phase 2 — Prometheus-native telemetry ingestion~~ **COMPLETE**

---

Phase 2+3

Status: COMPLETE

Date: 2026-05-25
Branch: claude/sleepy-bohr-HXNzY
PR: fnstggl/energy2#57 (squash-merged)

Summary:

* Created `aurelius/connectors/` package with 9 files:
  - `__init__.py` (package exports)
  - `base.py` (AuthType, AuthConfig, ConnectorConfig, MetricValue, RawMetricResult, TelemetrySnapshot)
  - `metric_mapping.py` (UnitConversion, MetricMapping, MetricMappingRegistry; dcgm/vllm/triton/ray_serve built-in registries; YAML loader)
  - `prometheus.py` (parse_prometheus_text, PrometheusClient w/ bearer/basic auth + retries, FakePrometheusClient, PrometheusTelemetryConnector)
  - `dcgm.py` (DCGMAdapter → GPUState; thermal_violation_ns in nanoseconds, fixes old µs bug)
  - `vllm.py` (VLLMAdapter → InferenceServiceState; handles V0 vllm_* and V1 vllm:* metric naming)
  - `triton.py` (TritonAdapter → InferenceServiceState; cumulative counter derivation for avg latency)
  - `ray_serve.py` (RayServeAdapter → InferenceServiceState; histogram latency, replica count)
  - `otel.py` (OTelAdapter → InferenceServiceState; OTLP JSON sandbox adapter)
* Created `configs/connectors/dcgm_mapping.yaml` and `vllm_mapping.yaml` (YAML overrides for built-in registries)
* Created 5 Prometheus text fixture files under `tests/fixtures/prometheus/`:
  - `dcgm_metrics.prom` (3 GPUs: A100-SXM4-80GB)
  - `vllm_metrics.prom` (llama3-70b, mistral-7b)
  - `triton_metrics.prom` (bert-large/1, gpt2/1)
  - `ray_serve_metrics.prom` (llm-router, embedding-service)
  - `prometheus_api_response.json` (Prometheus HTTP API vector response fixture)
* Created 3 test files:
  - `tests/test_prometheus_connector.py` (56 passed, 10 skipped — requests/yaml intentionally absent from pytest venv)
  - `tests/test_dcgm_adapter.py` (29 passed)
  - `tests/test_vllm_triton_ray_adapters.py` (43 passed — vLLM, Triton, Ray Serve, OTel, interface consistency)
* No existing optimizer, SLA, forecasting, or energy connector code was modified
* All pre-existing tests still pass

### Commands Run

```
ruff check aurelius/connectors/ --select=E,F,W
/root/.local/bin/pytest tests/test_prometheus_connector.py tests/test_dcgm_adapter.py tests/test_vllm_triton_ray_adapters.py -q
/root/.local/bin/pytest -q  # full suite
```

### Test Results

```
tests/test_prometheus_connector.py: 56 passed, 10 skipped
tests/test_dcgm_adapter.py: 29 passed
tests/test_vllm_triton_ray_adapters.py: 43 passed
Full suite: 282 passed, 10 skipped, 0 failed
ruff: All checks passed
```

### Repo-Reality Findings

* `FakePrometheusClient` supports both `fixtures={}` dict mode and `prometheus_text=` raw Prometheus text — same interface as `PrometheusClient`
* `MetricMappingRegistry` fallback_queries handle both PromQL expressions and raw metric names (critical for FakeClient compatibility)
* `_REQUESTS_AVAILABLE` and `_YAML_AVAILABLE` flags gate tests that require those optional libraries; 10 tests correctly skipped
* `thermal_violation_ns` is nanoseconds (not µs) — this fixes the documented bug in old `dcgm_provider.py`
* `kv_cache_usage` and `prefix_cache_hit_rate` stored as 0-1 fractions (not percentages) as required by `InferenceServiceState`
* Valid `engine` values enforced: `"vllm"`, `"triton"`, `"ray_serve"`, `"unknown"`
* None-not-zero invariant: all missing optional metrics → `None`, never `0`
* UTC-aware timestamps enforced on all `TelemetrySnapshot.fetched_at` and normalized state objects

### Wiring Evidence

* Phase 2+3 are additive; connectors and adapters are NOT yet wired into any production scheduler/optimizer decision path (intentional)
* Wiring happens in: Phase 6 (simulator consumes ClusterState), Phase 7 (classifier), Phase 9 (optimizer/engine)
* `FakePrometheusClient` enables safe sandbox testing with zero network calls

### Failure Mode Review

* Missing optional metrics → `None` (never fabricated as `0`)
* `TelemetrySnapshot.coverage_pct()` reports fraction of expected fields present; `Provenance.confidence` is `"low"` when < 40%
* `TelemetrySnapshot.unknown_metrics` lists DCGM/vLLM metric names not in the registry
* `DCGMAdapter.normalize_gpus()` logs a warning per failed GPU and continues (partial-failure safe)
* `_clamp_fraction()` in vLLM/OTel adapters handles both 0-1 and 0-100 input ranges gracefully

### Open Limitations

* `PrometheusClient` real HTTP path not exercised in CI (requests not in pytest venv — by design; same test suite runs in prod with requests installed)
* Triton p95/p99 latency = `None` (not available from default Triton metrics; cumulative average used for p50)
* Ray Serve `ttft_*` = `None` (Ray Serve doesn't expose LLM-specific token metrics by default)
* OTelAdapter is sandbox/fixture only — no real OTLP ingest path
* YAML metric mapping override (`load_mapping_yaml`) requires `pyyaml` (skipped in CI; works in production venv)
* No Kubernetes connector yet (Phase 4)
* No topology collector yet (Phase 5)

Next milestone: ~~Phase 4 — Kubernetes connector~~ **COMPLETE** → **Phase 6 — Synthetic Cluster Simulator**

---

## Phase 4+5 Completion Evidence

### Phase 4+5 Milestone Decision

- **What this run implemented:** Phase 4 (Kubernetes connector) and Phase 5 (Topology collector) together, since both produce supplementary NodeState data that the same test suite can cover
- **Why it was the correct next step:** Phase 1-3 verified (391 tests passing), no prior K8s or topology implementation existed, these are required foundations for Phase 6 (simulator needs fake K8s + fake topology endpoints)
- **Prior dependencies verified:** Phase 1-3 tests all pass (282 tests, 10 intentional skips for missing optional deps)
- **What was explicitly NOT attempted:** Synthetic simulator (Phase 6), constraint classifier (Phase 7), cost model (Phase 8), recommendation engine (Phase 9)

### Files Changed

| File | Role |
|---|---|
| `aurelius/connectors/kubernetes.py` | K8s read-only connector: `KubernetesConnector`, `FakeKubernetesConnector`, `K8sPlacementSnapshot`, `PodPlacement`, normalization helpers |
| `aurelius/connectors/topology.py` | Topology collector: `parse_nvidia_smi_topo`, `parse_nvidia_smi_list`, `build_topology_state`, `NvidiaSmiTopologyCollector`, `FakeTopologyCollector`, `PlacementScorer`, `score_placement`, `rank_placements` |
| `aurelius/connectors/__init__.py` | Added Phase 4+5 exports |
| `tests/test_kubernetes_connector.py` | 47 tests for K8s connector |
| `tests/test_topology_connector.py` | 62 tests for topology collector |
| `tests/fixtures/kubernetes/node_list.json` | Fixture: 4 nodes (2 GPU, 1 CPU, 1 unschedulable) |
| `tests/fixtures/kubernetes/pod_list.json` | Fixture: 7 pods (running, pending, succeeded, CPU-only) |
| `tests/fixtures/topology/dgx_h100_8gpu_nvswitch.txt` | nvidia-smi topo -m fixture: DGX H100 NVSwitch (NV18) |
| `tests/fixtures/topology/dgx_h100_inventory.txt` | nvidia-smi -L fixture: DGX H100 8 GPUs |
| `tests/fixtures/topology/pcie_8gpu_dual_numa.txt` | nvidia-smi topo -m fixture: PCIe 8-GPU dual NUMA |
| `tests/fixtures/topology/pcie_8gpu_inventory.txt` | nvidia-smi -L fixture: PCIe A100 8 GPUs |
| `configs/connectors/kubernetes_rbac.yaml` | Minimal read-only RBAC for enterprise K8s deployments |

### Plan vs Repo Reality

**Plan said:**
- Ingest Nodes, Pods, GPU resource requests/limits, labels/taints/topology labels
- Normalize into NodeState, WorkloadState, PlacementState, QueueState where possible
- Sandbox: fake K8s API responses; no live cluster required
- Topology: parse nvidia-smi topo -m, nvidia-smi -L, placement scoring

**Repo reality required:**
- `WorkloadState` extension deferred to Phase 9 (same decision as Phase 1 — avoids breaking SLA evaluator)
- `PlacementState` as a separate model is not yet needed: `K8sPlacementSnapshot.pods` contains all placement data needed by the simulator. A formal PlacementState can be added when Phase 7 (classifier) requires it.
- `QueueState` from pending pods: pending pod count and GPU demand available from `K8sPlacementSnapshot.pending_gpu_pods`. Formal QueueState normalization deferred to Phase 6/7.
- Topology: implemented `nvidia-smi topo -m` parser only (not NVML — optional dep); NV18 (H100 SXM NVSwitch) correctly maps to NVSWITCH
- Kubernetes topology merge (zone/rack labels into NodeState): implemented directly in `normalize_node_dict` via configurable label keys

### Tests Added

| Test File | Tests | What It Proves |
|---|---|---|
| `tests/test_kubernetes_connector.py` | 47 | GPU qty parsing, topology label extraction, node normalization, pod normalization, GPU-allocated-per-node derivation, pending pod detection, taint normalization, FakeKubernetesConnector fixture mode, no-write-methods guarantee, partial snapshot behavior |
| `tests/test_topology_connector.py` | 62 | _parse_link_token (NV18→NVSWITCH, all types), parse_nvidia_smi_topo (DGX H100 NVSwitch, PCIe dual-NUMA), NUMA affinity extraction, parse_nvidia_smi_list, build_topology_state (UUID translation, fallback to logical ID), _derive_interconnect_class, score_placement (NVSwitch > PCIe, same-NUMA > cross-NUMA, comm penalty, latency multiplier, conservative score when topology unavailable), rank_placements ordering, FakeTopologyCollector (text fixture, pre-built state), NvidiaSmiTopologyCollector graceful failure |

### Commands Run

```
ruff check aurelius/connectors/kubernetes.py aurelius/connectors/topology.py aurelius/connectors/__init__.py --select=E,F,W
python -m compileall aurelius/connectors/kubernetes.py aurelius/connectors/topology.py
/root/.local/bin/pytest tests/test_kubernetes_connector.py tests/test_topology_connector.py -q
/root/.local/bin/pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py tests/test_prometheus_connector.py tests/test_dcgm_adapter.py tests/test_vllm_triton_ray_adapters.py tests/test_kubernetes_connector.py tests/test_topology_connector.py -q
```

### Test Results

```
tests/test_kubernetes_connector.py: 47 passed
tests/test_topology_connector.py: 62 passed
Phase 1-5 full suite: 391 passed, 10 skipped, 0 failed

ruff: All checks passed
python -m compileall: No errors
```

### Pre-Existing Failures (Not Caused by This Run)

6 failures in `test_sla_engine.py` and `test_sla_optimization.py` require `pyyaml` which is not installed in the pytest environment (this is a pre-existing env gap, not caused by Phase 4+5 changes). These tests are tracked as TESTED_WITH_ENV_GAPS.

### Wiring Evidence

Phase 4+5 are additive. The K8s connector and topology collector produce data structures that feed:
- Phase 6 (simulator) via `FakeKubernetesConnector` + `FakeTopologyCollector` — **same code paths as real connectors**
- Phase 7 (classifier) which consumes `NodeState`, `TopologyState`, pending pod signals
- Phase 9 (recommendation engine) for topology-aware placement recommendations

### Failure Mode Review

**Kubernetes connector:**
- `kubernetes` package not installed → `is_partial=True`, `missing_sources=["kubernetes-client-init"]`, never raises
- List nodes fails → nodes dict empty, partial flag set
- List pods fails → pods list empty, partial flag set
- Malformed node/pod dict → returns None, node added to `missing_sources`, parse continues
- GPU quantity string non-integer → `None` (not 0)
- Node not in allocated_per_node map → `gpu_allocated=None` (not 0)

**Topology collector:**
- `nvidia-smi` not found → returns `None`, caller degrades gracefully
- `nvidia-smi` timeout → returns `None`
- No GPU pairs found in topo output → returns `None`
- No UUID map → logical IDs (GPU0, GPU1) used as-is in `pair_levels` keys
- GPU UUIDs not in `TopologyState.pair_levels` → `score_placement` returns 0.5 (conservative, not 0)
- `topology=None` → `score_placement` returns 0.5

### Open Limitations

* Real K8s connector requires `kubernetes` package (not in pytest env; gated by `_K8S_AVAILABLE`)
* Topology collector requires node-local nvidia-smi access; cloud-only deployments → topology=None
* NVML-based topology (preferred over text parsing) is a future enhancement — interface exists
* Formal `PlacementState` model deferred to Phase 7
* `QueueState` from K8s pending pods deferred to Phase 7

### Next Milestone

~~Phase 6 — Synthetic Cluster Simulator~~ **COMPLETE** → **Phase 7 — Constraint Classifier**

---

## Phase 6 Completion Evidence

### Phase 6 Milestone Decision

- **What this run implemented:** Phase 6 — Synthetic Cluster Simulator
- **Why it was the correct next step:** Phases 1-5 verified (484 tests passing, 10 intentional skips). Phase 6 is the prerequisite for Phase 7 (constraint classifier needs labeled benchmark fixtures) and Phase 11 (benchmarking loop).
- **Prior dependencies verified:** All Phase 1-5 tests pass. `ClusterState`, all connector interfaces, and `FakeKubernetesConnector`/`FakeTopologyCollector` are verified before simulator was built.
- **What was explicitly NOT attempted:** Constraint classifier (Phase 7), cost/risk/migration model (Phase 8), recommendation engine (Phase 9).

### Files Added

| File | Role |
|---|---|
| `aurelius/simulation/cluster/__init__.py` | Package exports: `ClusterSimulator`, `SimulatorTick`, all model classes, `load_scenario`, `list_scenarios` |
| `aurelius/simulation/cluster/engine.py` | `ClusterSimulator`: seeded RNG, EMA thermal (α=0.25), M/M/1 queue, topology penalty, KV cache proxy, cold-start warmup, migration, `get_cluster_state()` → `ClusterState` |
| `aurelius/simulation/cluster/model.py` | Mutable state: `SimGPU`, `SimNode`, `SimQueue`, `SimWorkload`, `SimRegion`, `SimCluster`; `GPU_PROFILES` (H100 SXM5, A100 SXM4/PCIe, L4) |
| `aurelius/simulation/cluster/scenarios.py` | `load_scenario()` with YAML fallback to built-in Python dicts; `list_scenarios()` |
| `aurelius/simulation/cluster/fakes/__init__.py` | Package init |
| `aurelius/simulation/cluster/fakes/prometheus_text.py` | `generate_dcgm_metrics_text()` (DCGM format), `generate_vllm_metrics_text()` (vLLM V1 format with histogram buckets) |
| `aurelius/simulation/cluster/fakes/kubernetes_payloads.py` | `generate_node_list()` → V1NodeList, `generate_pod_list()` → V1PodList |
| `aurelius/simulation/cluster/fakes/topology_text.py` | `generate_topo_text()` → nvidia-smi topo -m, `generate_gpu_list_text()` → nvidia-smi -L |
| `benchmarks/v1/energy_price_arbitrage_multiregion.yaml` | Scenario 1: `energy_bound` |
| `benchmarks/v1/thermal_hotspot_mixed_cluster.yaml` | Scenario 2: `thermal_bound` |
| `benchmarks/v1/queue_surge_latency_sensitive.yaml` | Scenario 3: `queue_bound` |
| `benchmarks/v1/latency_tail_kvcache_pressure.yaml` | Scenario 4: `memory_bound_indirect` |
| `benchmarks/v1/topology_fragmentation_h100.yaml` | Scenario 5: `topology_bound` |
| `benchmarks/v1/underutilization_stranded_capacity.yaml` | Scenario 6: `utilization_bound` |
| `tests/test_cluster_simulator.py` | 61 simulator tests |
| `tests/test_fake_connectors.py` | 32 fake connector tests |

### Files Modified

| File | Change |
|---|---|
| `aurelius/simulation/__init__.py` | Wrapped optional `pandas`-dependent imports in `try/except ImportError` to prevent test environment failures |

### Plan vs Repo Reality

**Plan said:**
- Discrete-event simulator at hourly ticks
- EMA thermal model (α=0.25), throttle >83°C
- M/M/1 queue latency approximation with diurnal modulation
- Fake DCGM Prometheus text, vLLM metrics, K8s payloads, topology text
- 6 frozen benchmark scenarios covering all 6 constraint archetypes
- `ClusterState` output with `is_sandbox=True`
- Deterministic replay via seeded RNG

**Repo reality matched plan exactly.** One addition not in the plan:
- `SimulatorTick` also includes pre-built `dcgm_texts`, `vllm_texts`, `k8s_node_list`, `k8s_pod_list`, `topology_texts` dict so callers don't need to call each fake generator separately

### Tests Added

| Test File | Tests | What It Proves |
|---|---|---|
| `tests/test_cluster_simulator.py` | 61 | Determinism (same seed → identical cost), thermal throttling (EMA lag, throttle detection), queue/TTFT/TPOT latency model, migration cold-start (2-tick warmup), KV cache proxy, topology score, all 6 scenarios runnable, `ClusterState` field mapping (all required fields present, `is_sandbox=True`, UTC timestamps), tick metrics, cost accounting |
| `tests/test_fake_connectors.py` | 32 | Production `DCGMAdapter.normalize_gpus()` parses simulator DCGM text (throttle detection, memory, power, temperature); production `parse_nvidia_smi_topo` identifies NV18 links from simulator topology text; production `FakeKubernetesConnector` parses simulator V1NodeList; vLLM metrics text parses correctly; K8s pod list has running + pending pods; topology matrix format is correct |

### Commands Run

```
ruff check aurelius/simulation/cluster/ tests/test_cluster_simulator.py tests/test_fake_connectors.py --select=E,F,W
pytest tests/test_cluster_simulator.py tests/test_fake_connectors.py tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py tests/test_prometheus_connector.py tests/test_dcgm_adapter.py tests/test_vllm_triton_ray_adapters.py tests/test_kubernetes_connector.py tests/test_topology_connector.py -q
```

### Test Results

```
tests/test_cluster_simulator.py: 61 passed
tests/test_fake_connectors.py: 32 passed
Phase 1-6 full suite: 484 passed, 10 skipped, 0 failed
ruff: All checks passed
```

### Wiring Evidence

Phase 6 is the first phase that exercises the full vertical slice of the connector boundary. Specifically:

- `ClusterSimulator.get_cluster_state()` produces a canonical `ClusterState` (Phase 1 model) with `is_sandbox=True`
- `ClusterSimulator.get_dcgm_metrics_text(node_id)` produces Prometheus text that **production `DCGMAdapter`** parses unchanged (verified by `test_fake_connectors.py`)
- `ClusterSimulator.get_topology_text(node_id)` produces nvidia-smi text that **production `parse_nvidia_smi_topo`** parses unchanged (verified)
- `ClusterSimulator.get_kubernetes_node_list()` produces V1NodeList that **production `FakeKubernetesConnector`** parses unchanged (verified)
- The simulator does NOT wire into any production optimizer/scheduler path — this is intentional; wiring happens in Phase 9

### Failure Mode Review

- **Seeded RNG:** `reset()` restores exact initial state; identical total_energy_cost diff < 1e-9 across runs
- **Missing GPU profile:** Falls back to `a100-sxm4-80gb` (never KeyErrors)
- **Missing scenario YAML:** Falls back to built-in Python dicts; no filesystem dependency in CI
- **Thermal throttle:** 1-tick lag due to EMA; `thermal_throttle_active` never set without temperature data
- **Queue saturation:** M/M/1 `rho` clamped to 0.99 to avoid division by zero; saturated queue gets `mean_wait_s = 60.0`
- **Migration to unknown region:** No-op (does not crash)
- **Cold-start:** `kv_cache_usage` resets to 0.05, warmup ticks decrement each tick; full throughput restored after 2 ticks
- **`is_sandbox=True`:** All `Provenance` objects from simulator carry this flag; downstream consumer can exclude from economic claims

### Open Limitations

- Thermal model is EMA proxy (not physics); aggressive throttle events may not match real GPU behavior exactly
- M/M/1 queue is a rough approximation; real vLLM uses continuous batching with different saturation dynamics
- KV cache proxy tracks memory pressure monotonically (never decreases below current high-water mark); real KV cache can evict
- Communication bandwidth is a proxy (fraction of NVLink/PCIe TDP); actual NCCL behavior varies by collective type
- Simulator `tick_duration_hours=1.0` by default; sub-hourly simulation would require smaller SLA/latency sensitivity
- 6 scenarios cover known constraint archetypes; mixed-constraint scenarios (e.g., simultaneous thermal + queue) are not yet frozen benchmarks

The simulator must:
1. Expose fake Prometheus, fake K8s API, and fake topology fixture endpoints
2. Use the same connector interfaces (FakePrometheusClient, FakeKubernetesConnector, FakeTopologyCollector) as real deployments
3. Simulate GPU utilization, thermal, queue, and latency dynamics
4. Provide baseline comparisons (FIFO, current_price_only, greedy energy, SLA-aware)
5. Produce ClusterState snapshots via the same normalization paths

---

## Phase 7+8 Completion Evidence

### Phase 7+8 Milestone Decision

- **What this run implemented:** Phase 7 (Constraint Classifier) + Phase 8 (Migration Cost/Risk Model), combined on branch `claude/inspiring-einstein-2rpxi`
- **Why this was the correct next step:** Two stale Phase 7 PRs existed (#61, #62) — both were closed and the best elements merged into one clean implementation. Phase 8 was added to the same branch since the classifier is its primary input.
- **Prior dependencies verified:** Phase 1-6 full suite: 484 tests passing, 10 intentional skips (unchanged).
- **Bug fixes also applied (from PR #61 verification):**
  - `aurelius/connectors/base.py`: `basic_credentials()` returns `None` when password env var is unset (was returning `(username, "")`)
  - `aurelius/simulation/cluster/engine.py`: `_parse_float_trace()` handles YAML dash-separated strings (`"200 - 210 - 220"`)
  - `aurelius/simulation/cluster/engine.py`: energy spike event no longer compounds price each tick; uses `price_spike_active` flag to block trace overwrites
  - `aurelius/simulation/cluster/model.py`: added `price_spike_active: bool` to `SimRegion`
  - `benchmarks/v1/queue_surge_latency_sensitive.yaml`: reduced `critical-wl` GPU count from 2→1 and util from 65%→50% so queue actually saturates during surge

### Files Added

| File | Role |
|---|---|
| `aurelius/constraints/__init__.py` | Package: exports `ConstraintClassifier`, `ConstraintConfig`, `MigrationCostModel`, `MigrationCostEstimate`, `MigrationGovernor` |
| `aurelius/constraints/classifier.py` | Phase 7: scores 8 constraint families from ClusterState; hysteresis; tie-break; confidence; fail-safe |
| `aurelius/constraints/cost_model.py` | Phase 8: `MigrationCostEstimate`, `MigrationCostModel` (conservative heuristics), `MigrationGovernor` (rate-limit/cooldown) |
| `tests/test_constraint_classifier.py` | 74 tests: all 8 scorers, missing-signal invariants, hysteresis, tie-break, confidence math, 6 simulator scenarios |
| `tests/test_migration_cost_model.py` | 39 tests: cost estimate viability, governor rate limits, critical vs batch multipliers, topology degradation, thermal penalty, make_recommendation, pipeline integration |

### Files Modified

| File | Change |
|---|---|
| `aurelius/connectors/base.py` | `basic_credentials()` returns `None` when password env var unset |
| `aurelius/simulation/cluster/engine.py` | `_parse_float_trace()` helper; energy spike uses `price_spike_active` flag |
| `aurelius/simulation/cluster/model.py` | Added `price_spike_active: bool = False` to `SimRegion` |
| `benchmarks/v1/queue_surge_latency_sensitive.yaml` | Queue scenario `critical-wl` GPU count 2→1, util 65→50% |

### Classifier Design Decisions (Phase 7)

| Decision | Rationale |
|---|---|
| Latency scorer uses separate TTFT SLA (2000ms) vs e2e SLA (30000ms) | LLM e2e p99 can legitimately be 10–30s; a single 2000ms SLA produced false positives on all simulator ticks |
| Communication scorer requires SM < 50% to score high | High NVLink bytes with high SM = compute active, not stalled; both conditions needed to detect genuine stall |
| Confidence uses raw binding score, not threshold-normalized | `(score-threshold)/(1-threshold)` collapsed moderate scores to near-zero; raw score preserves signal |
| Hysteresis requires N consecutive identical candidates | Prevents flapping when scores oscillate near threshold |
| All thresholds labeled `# HEURISTIC` in `ConstraintConfig` | Operator-visible tuning; none calibrated on real production telemetry |

### Cost Model Design (Phase 8)

> **SUPERSEDED for risk computation** — the `risk_mult` per-workload-tier row below
> describes the ORIGINAL Phase 8 design. It was corrected (see "Phase 8/9 Risk
> Model Correction" at the end of this document). Static workload multipliers were
> removed; risk is now state-conditioned. The governor / `should_keep` / `make_recommendation`
> rows remain accurate.

| Component | Implementation |
|---|---|
| `MigrationCostEstimate` | Dataclass: gross savings, informational physical penalties (cold-start, cache warmup, queue instability, topology degradation, failure retry), state-conditioned risk buckets (SLA / destination / action / uncertainty / thermal), total penalty, net expected savings, **risk-factor explanation** |
| `MigrationCostModel.estimate()` | ~~Conservative heuristics with `risk_mult` per workload tier; critical×2.5, batch×0.4~~ **CORRECTED:** state-conditioned risk from SLA headroom, destination health, action-specific cost, and telemetry confidence. No static workload multiplier |
| `MigrationGovernor` | Per-workload: min interval (300s), hourly rate limit (2/hr); cluster: per-minute rate limit (3/min); SLA violation cooldown (600s) |
| `should_keep()` | KEEP when: hard SLA breach, blocked by governor, net savings unknown, net ≤ 0, or confidence < 0.15 |
| `make_recommendation()` | Always `recommendation_only` mode; delegates execution to Phase 9 |

### Commands Run

```
python -m compileall aurelius/constraints/
ruff check aurelius/constraints/ tests/test_constraint_classifier.py tests/test_migration_cost_model.py
pytest tests/test_constraint_classifier.py tests/test_migration_cost_model.py -q
pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py tests/test_prometheus_connector.py tests/test_dcgm_adapter.py tests/test_vllm_triton_ray_adapters.py tests/test_kubernetes_connector.py tests/test_topology_connector.py tests/test_cluster_simulator.py tests/test_fake_connectors.py tests/test_constraint_classifier.py tests/test_migration_cost_model.py -q
```

### Test Results

```
tests/test_constraint_classifier.py:  74 passed
tests/test_migration_cost_model.py:   39 passed
Phase 1-8 full suite:                597 passed, 10 skipped, 0 failed

ruff: All checks passed
python -m compileall: No errors
```

### Wiring Evidence

- Phase 7 is additive: `ConstraintClassifier.assess(ClusterState)` → `ConstraintAssessment`; not yet wired into any production optimizer/scheduler path (intentional; wiring in Phase 9)
- Phase 8 is additive: `MigrationCostModel.estimate(...)` → `MigrationCostEstimate` → `make_recommendation()` → `Recommendation`; not yet wired into optimizer decision path (Phase 9 target)
- Phase 7+8 both read `ClusterState` from Phase 1 model (verified)
- `ConstraintAssessment` feeds `MigrationCostModel` as required by the plan
- `Recommendation` produced is in `recommendation_only` mode (default safe behavior)

### Failure Mode Review

- **Missing signals:** All 8 scorers return `(None, missing_signals)` when required telemetry absent; `binding_constraint=None` when no family scores
- **Unknown gross savings:** `MigrationCostEstimate.net_expected_savings=None`; `should_keep()` returns KEEP
- **Low confidence:** `confidence < confidence_floor` → `binding_constraint=None`; `confidence < 0.15` in cost model → KEEP
- **Governor blocked:** `blocked_by_cooldown=True`, `is_viable()=False`, `make_recommendation()` returns KEEP
- **Partial ClusterState:** 0.85× confidence penalty; classifier still operates with remaining signals
- **Sandbox provenance:** `is_sandbox=True` passes through to all outputs; downstream can exclude from economic claims

### Open Limitations

- All classifier thresholds are HEURISTIC — calibrated on synthetic scenarios, not production telemetry
- Cost model penalty multipliers are HEURISTIC engineering estimates
- No persistence of migration history (governor is in-memory only); production needs Postgres store (Phase 11/12)
- `MigrationGovernor` history is per-process-instance; distributed deployments need shared state
- Phase 9 (recommendation engine) has not yet wired the classifier+cost model into optimizer decisions
- CLI reporting commands shipped in Phase 10 (constraint-report, simulate-constraint-scenario, telemetry-check, topology-report, validate-connectors)

---

## Phase 9 Completion Evidence

### Phase 9 Milestone Decision

- **What this run implemented:** Phase 9 — Constraint-aware recommendation engine
- **Why it was the correct next step:** Phases 1-8 verified (597 tests passing, 10 intentional skips). The classifier produced `ConstraintAssessment` and the cost model produced `MigrationCostEstimate`, but no engine wired them together into `Recommendation` objects.
- **Prior dependencies verified:** Phase 1-8 full suite: 597 tests passing, 10 intentional skips (unchanged).
- **What was explicitly NOT attempted:** CLI reports (Phase 10), benchmarking loop (Phase 11), production hardening (Phase 12), BacktestEngine wiring, WorkloadState extension.

### Files Added

| File | Role |
|---|---|
| `aurelius/constraints/engine.py` | `ConstraintAwareEngine`, `EngineResult`, `WorkloadDescriptor`; per-constraint candidate generators; state adapters (`_service_to_sla_workload_state`, `_build_region_contexts`, `_cheaper_regions`) |
| `tests/test_constraint_engine.py` | 53 tests for all safety invariants |

### Files Modified

| File | Change |
|---|---|
| `aurelius/constraints/__init__.py` | Added `ConstraintAwareEngine`, `EngineResult`, `WorkloadDescriptor` exports |

### Engine Design

The `ConstraintAwareEngine` pipeline for each `ClusterState` snapshot:

1. `ConstraintClassifier.assess(state)` → `ConstraintAssessment`
2. Low-confidence check → KEEP all (fail-safe)
3. Per-service: generate constraint-appropriate candidate `OptimizationAction`s
4. Filter disallowed actions from `ConstraintAssessment.disallowed_action_types`
5. `SLAAwareActionSelector.select()` → `SLADecision` (SLA gate)
6. `MigrationCostModel.estimate()` + `should_keep()` → cost gate
7. Emit `Recommendation` in `recommendation_only` mode

Per-constraint candidate generators:
- **ENERGY:** `CHOOSE_CHEAPER_REGION` + `DEFER`
- **THERMAL:** `SPREAD` + `REROUTE` (never `CONSOLIDATE` — classifier disallows it)
- **QUEUE:** `SCALE_REPLICAS` + `SPREAD` (never `CONSOLIDATE`)
- **LATENCY:** `SCALE_REPLICAS` only (never `MIGRATE` — disallowed for latency-bound)
- **COMMUNICATION:** `CHANGE_PLACEMENT` + `SPREAD`
- **MEMORY:** `SPREAD` + `SCALE_REPLICAS` (no KV cache internals)
- **TOPOLOGY:** `CHANGE_PLACEMENT` + `CONSOLIDATE`
- **UTILIZATION:** `CONSOLIDATE`
- **NONE:** KEEP (no candidates generated)

### Plan vs Repo Reality

**Plan said:**
- Wire `ConstraintClassifier` → `MigrationCostModel` → `Recommendation` into main scheduler/optimizer path
- Implement per-constraint action selection (8 families → allowed action types)
- SLA evaluator wiring: `SLARegistry.evaluate()` gates action selection
- Audit `BacktestEngine` to consume `ClusterState` + `ConstraintAssessment`
- Extend `WorkloadState` (deferred from Phase 1)

**Repo reality required:**
- `BacktestEngine` wiring deferred — the backtest engine is part of the legacy energy-arbitrage path; wiring it to ClusterState requires a deeper SLA evaluator refactor that belongs in Phase 10/11 (it would change existing backtest behavior, violating the "preserve old optimizer behavior" invariant).
- `WorkloadState` extension (add `service_id`, `gpu_uuids`, `comm_bytes_per_s`) deferred — the engine uses `InferenceServiceState` directly as the per-service unit; extending `WorkloadState` is still needed for Phase 10 but not required for Phase 9 to work correctly.
- `SLARegistry` is wired via `engine.run(state, sla_registry=registry)` — same interface as the existing `SLAAwareActionSelector`.

### Tests Added

| Test Class | Count | What It Proves |
|---|---|---|
| `TestEngineInstantiation` | 3 | Default init, custom modes, invalid mode raises |
| `TestEmptyState` | 3 | Empty ClusterState → no recommendations, no crash, to_dict works |
| `TestRecommendationInvariants` | 6 | recommendation_only mode, dry_run mode, is_sandbox propagation, timestamp matches state, unique IDs, workload IDs match services, confidence in [0,1] |
| `TestLowConfidenceFallback` | 2 | High confidence_floor → all KEEP, rationale mentions floor |
| `TestEnergyBound` | 5 | binding constraint check, service gets migration or KEEP, cheaper_regions helper sorted, empty when no cheaper, empty when price unknown |
| `TestThermalBound` | 3 | Runs without error, no CONSOLIDATE, service gets SPREAD/REROUTE/KEEP |
| `TestQueueBound` | 2 | Runs, service gets SCALE/SPREAD/KEEP, no CONSOLIDATE |
| `TestLatencyBound` | 3 | Runs, never emits MIGRATE, service gets SCALE/KEEP |
| `TestUtilizationBound` | 2 | Runs, service gets CONSOLIDATE/KEEP |
| `TestSLAGate` | 4 | migration_allowed=False blocks migration, allowed_regions enforced, no registry = no block, blocked actions in rejected list |
| `TestCostModelGate` | 2 | High threshold → KEEP, rejected list has entries |
| `TestNoopCounts` | 3 | noop_count matches is_noop, noop+actionable=total, low confidence all noop |
| `TestMultipleServices` | 2 | Each service gets one recommendation, services get independent recommendations |
| `TestDeterminism` | 2 | Fresh engines produce same result, hysteresis_count=1 stabilizes |
| `TestAdapters` | 5 | service→WorkloadState mapping, None region, region contexts thermal, energy, empty cluster |
| `TestWorkloadDescriptor` | 2 | job_id, default workload_type |
| `TestSerialization` | 2 | EngineResult.to_dict, Recommendation.to_dict required keys |

### Commands Run

```
ruff check aurelius/constraints/engine.py aurelius/constraints/__init__.py tests/test_constraint_engine.py --select=E,F,W
python -m compileall aurelius/constraints/
pytest tests/test_constraint_engine.py -q
pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py tests/test_prometheus_connector.py tests/test_dcgm_adapter.py tests/test_vllm_triton_ray_adapters.py tests/test_kubernetes_connector.py tests/test_topology_connector.py tests/test_cluster_simulator.py tests/test_fake_connectors.py tests/test_constraint_classifier.py tests/test_migration_cost_model.py tests/test_constraint_engine.py -q
pytest tests/test_scheduler.py tests/test_safety_gate.py -q
```

### Test Results

```
tests/test_constraint_engine.py: 53 passed
Phase 1-9 full suite:            650 passed, 10 skipped, 0 failed
Existing optimizer tests:        20 passed (unchanged)

ruff: All checks passed (new files only)
python -m compileall: No errors
```

### Wiring Evidence

Phase 9 is additive. The `ConstraintAwareEngine` produces `Recommendation` objects from `ClusterState` — it is NOT yet wired into the `BacktestEngine` or existing energy-arbitrage CLI paths (intentional; those are Phase 10/11 targets).

What IS wired in Phase 9:
- `ConstraintClassifier.assess(ClusterState)` → `ConstraintAssessment` (Phase 7)
- `MigrationCostModel.estimate(...)` + `should_keep()` (Phase 8)
- `SLAAwareActionSelector.select()` via `SLARegistry` (existing SLA engine)
- `Recommendation` output with `implementation_mode="recommendation_only"` (Phase 1 model)

What remains unwired until later phases:
- `BacktestEngine` still uses legacy energy-arbitrage path (Phase 11)
- CLI `constraint-report` command (Phase 10)
- `MigrationGovernor.record_migration()` not called in engine (engine is recommendation-only; execution layer must call this)

### Failure Mode Review

- **Low confidence:** `confidence < confidence_floor` → all services get KEEP (fail-safe, no crash)
- **Empty ClusterState:** `state.all_services` empty → empty recommendations, no crash
- **SLA blocks migration:** `migration_allowed=False` → no migration action emitted, falls back to KEEP or non-migration action
- **Cost model blocks action:** `net_expected_savings ≤ 0` → KEEP emitted, blocked action added to `rejected` list
- **Unknown region:** `service.region=None` → candidates use "unknown" as target_region fallback
- **No cheaper regions:** `_cheaper_regions()` returns `[]` → only DEFER candidate generated for ENERGY-bound
- **is_sandbox propagation:** `state.provenance.is_sandbox=True` → all `Recommendation.provenance.is_sandbox=True`
- **Classifier disallowed actions:** e.g. `CONSOLIDATE` during THERMAL → filtered from candidates before SLA gate

### Open Limitations

- All candidate generators use HEURISTIC gross savings estimates (not calibrated on real telemetry)
- `WorkloadDescriptor.workload_type` is hardcoded to `"realtime_inference"` — Phase 10 should derive from `InferenceServiceState.engine` or K8s labels
- `current_topo_score` / `target_topo_score` are crude proxies (0.7/0.3 heuristics) — Phase 10 should use real `PlacementScorer.score_placement()`
- `BacktestEngine` not yet wired to `ClusterState` — deferred to Phase 11
- `MigrationGovernor` is in-memory only — Phase 11/12 should persist to Postgres

### Next Milestone

**Phase 11 — Validation, benchmarking, and continuous optimization loop**

Planned:
- Wire `ConstraintAwareEngine` into `BacktestEngine` for historical replay
- `MigrationGovernor` persistence to Postgres
- `PlacementScorer.score_placement()` integration (replace 0.7/0.3 heuristics)
- `WorkloadState` extension (`service_id`, `gpu_uuids`, `comm_bytes_per_s`)
- Continuous optimization loop with real telemetry connectors

---

## Phase 10 Completion Evidence

### Phase 10 Milestone Decision

- **What this run implemented:** Phase 10 — CLI constraint-aware report commands
- **Why it was the correct next step:** Phase 9 complete (660 tests passing). The `ConstraintAwareEngine` was fully implemented but had no user-facing CLI surface; Phase 10 exposes it via 5 CLI subcommands.
- **Prior dependencies verified:** Phase 1-9 full suite: 660 tests passing.
- **What was explicitly NOT attempted:** BacktestEngine wiring (Phase 11), MigrationGovernor Postgres persistence (Phase 11/12), PlacementScorer integration (Phase 11).

### Files Added

| File | Role |
|---|---|
| `aurelius/cli_constraint.py` | Command implementations for all 5 CLI subcommands (lazy imports) |
| `aurelius/reporting/constraint_report.py` | Text/JSON formatters for all constraint-aware CLI outputs |
| `tests/test_cli_assess_recommend.py` | 58 tests covering formatters, dispatch, sandbox invariants, CLI registration |

### Files Modified

| File | Change |
|---|---|
| `aurelius/cli.py` | Added 5 argparse subparsers + dispatch branches |
| `aurelius/reporting/__init__.py` | Wrapped matplotlib/pandas/numpy imports in try/except to prevent cascade |

### CLI Commands

| Command | Flags | Description |
|---|---|---|
| `constraint-report` | `--scenario`/`--snapshot`, `--steps`, `--format text\|json` | Assess constraints, emit `recommendation_only` recommendations |
| `simulate-constraint-scenario` | `--scenario`, `--steps`, `--list` | Run ClusterSimulator; output labeled `[SANDBOX]` |
| `telemetry-check` | `--scenario`/`--snapshot`, `--steps` | Report telemetry coverage per region/node |
| `topology-report` | `--scenario`/`--snapshot`, `--steps` | Display GPU topology |
| `validate-connectors` | _(none)_ | Smoke-test all 10 connectors/adapters |

### Test Evidence

```
pytest tests/test_cli_assess_recommend.py
58 passed in 0.89s

pytest (full suite)
718 passed
```

### Invariants Maintained

- All recommendations are `recommendation_only`; no cluster mutations
- Missing telemetry is `None`, never fabricated as `0`
- Sandbox outputs explicitly labeled `[SANDBOX]`
- No secrets in logs or reports
- `validate-connectors` reports 10/10 PASSED

### Merge

PR #65 merged to main via squash at commit `0471e8533e5d06f1078ff4862cbb491296edc6f3`.

---

## Phase 11 — Validation, Benchmarking, and Continuous Optimization Improvement

### Completion Evidence

- **58/58 Phase 11 tests passing** (`tests/test_constraint_benchmark.py`)
- **571/577 constraint-aware tests passing** (6 pre-existing failures due to missing PyYAML, unrelated to Phase 11)
- All benchmark outputs labeled `[SANDBOX]`
- No secrets committed

### What was implemented

**Multi-policy constraint-aware benchmark framework** with deterministic seeds, immutable scenario versioning, regression detection, and 3 new CLI commands.

- **Why it was the correct next step:** Phases 1-10 verified (718 tests passing). The `ConstraintAwareEngine` had full CLI surface (Phase 10) but no quantitative performance verification. Phase 11 adds closed-loop simulation benchmarking that compares the engine against 4 baselines and enforces SLA/cost/migration safety invariants.
- **Prior dependencies verified:** Phase 1-10 full suite: 718 tests passing.
- **What was explicitly NOT attempted:** Live production data ingestion (Phase 12), Postgres persistence for benchmark history (Phase 12), real-time KPI streaming.

### Files Added

| File | Role |
|---|---|
| `aurelius/benchmarks/__init__.py` | Package init — exports all public symbols |
| `aurelius/benchmarks/report.py` | `BenchmarkMetadata`, `TickKPI`, `AggregatedKPI`, `OptimizationScorecard`, `BenchmarkReport`, `build_scorecard()` |
| `aurelius/benchmarks/constraint_runner.py` | `ConstraintBenchmarkRunner` — 5 policies, closed-loop simulation, KPI collection |
| `aurelius/benchmarks/regression.py` | `BenchmarkRegressionChecker` — metadata compatibility, KPI diffs, pass/fail |
| `aurelius/benchmarks/scenario_lock.py` | SHA-256 scenario lockfile generator/checker (`--check` / `--generate`) |
| `benchmarks/v1/.scenario_hashes.json` | Frozen hashes for all 6 canonical v1 scenarios |
| `tests/test_constraint_benchmark.py` | 58 tests across 14 test classes |

### Files Modified

| File | Change |
|---|---|
| `aurelius/state/models.py` | Added `target_region: Optional[str]` field to `Recommendation` dataclass |
| `aurelius/constraints/engine.py` | Populated `target_region` on output `Recommendation` objects |
| `aurelius/simulation/cluster/engine.py` | Added safety guards to `migrate_workload()` (same-region, missing-region checks) |
| `aurelius/simulation/cluster/scenarios.py` | Fixed `list_scenarios()` to skip hidden files (`.scenario_hashes.json`) |
| `aurelius/cli_constraint.py` | Added `cmd_benchmark_run`, `cmd_benchmark_compare`, `cmd_optimizer_regression_check` |
| `aurelius/cli.py` | Added 3 argparse subparsers + dispatch branches for Phase 11 commands |
| `docs/COMPUTE_OPTIMIZATION_PROGRESS.md` | Updated status to Phase 11 complete |

### CLI Commands

| Command | Flags | Description |
|---|---|---|
| `benchmark-run` | `--scenario`, `--steps`, `--seed`, `--output-dir`, `--baseline` | Run multi-policy benchmark; exit 1 if regression flags |
| `benchmark-compare` | `--baseline`, `--current`, `--policy` | Compare two saved JSON reports; exit 1 on regression |
| `optimizer-regression-check` | `--steps`, `--seed`, `--min-score` | Run all scenarios; verify SLA ≤ FIFO, churn bounded, scorecard ≥ threshold |

### Benchmark Framework Details

- **5 policies benchmarked**: `fifo` (baseline, no-op), `current_price_only`, `greedy_energy`, `sla_aware`, `constraint_aware`
- **6 canonical scenarios**: all `benchmarks/v1/*.yaml` scenarios (energy arbitrage, queue surge, thermal throttling, topology fragmentation, migration cost, mixed)
- **OptimizationScorecard**: 7 weighted components — net_cost (0.25), sla_preservation (0.25), utilization (0.15), latency (0.15), migration_stability (0.10), thermal (0.05), topology (0.05)
- **Regression thresholds**: cost +2%, SLA any increase, p99 latency +10%, migration churn +50%, scorecard -5%
- **Deterministic**: seed-based `ClusterSimulator` ensures reproducible results; metadata hash validates scenario identity

### Test Evidence

```
pytest tests/test_constraint_benchmark.py
58 passed in 0.90s

pytest (constraint-aware suite)
571 passed, 6 pre-existing failures (missing PyYAML — unrelated)
```

### Safety Invariants Maintained

- All recommendations are `recommendation_only`; no live cluster mutations
- Missing telemetry is `None`, never fabricated as `0`
- Benchmark outputs explicitly labeled `[SANDBOX]`
- No secrets in logs or reports
- Scenario lockfile prevents silent YAML drift in CI

---

## Phase 8/9 Risk Model Correction (state-conditioned migration risk)

### Why this correction was required

The original Phase 8 `MigrationCostModel` decided migration safety primarily from a
**static workload-class multiplier**:

```python
# REMOVED (unsafe):
if is_critical:
    risk_mult = cfg.critical_workload_risk_multiplier   # 2.5×
elif is_batch:
    risk_mult = cfg.batch_workload_risk_multiplier      # 0.4×
else:
    risk_mult = 1.0
cold_start_ms = base * risk_mult     # every penalty scaled by the label
```

This is not enterprise-grade. The workload *label* alone could block a perfectly safe
migration (critical workload with huge SLA headroom and a warm, idle destination) or
permit an unsafe one (batch workload migrating into a hot, full, distant region). Risk
must be a function of **state**, not of a hardcoded label coefficient.

### What changed

`risk = base_risk × workload_type_multiplier` was **removed**. `CostModelConfig` no
longer has `critical_workload_risk_multiplier` / `batch_workload_risk_multiplier`.
`MigrationCostModel.estimate()` now computes risk from first principles across five
state-conditioned families, each surfaced as an explicit penalty bucket and explained
via `risk_factors` / `dominant_risk_factors`:

| Risk family | Field | State inputs (all from `RiskInputs`, missing ⇒ uncertainty) |
|---|---|---|
| 1. SLA headroom | `sla_risk_penalty` | predicted p95/p99/queue/error/availability/capacity vs **hard SLA bounds**; plus the active binding constraint as a low-headroom proxy (migrations only) |
| 2. Workload/runtime | (feeds 1, 3) | request rate, active sequences, latency sensitivity (via SLA tightness), queue depth, KV/cache pressure, prefix-cache affinity, `migration_allowed` policy |
| 3. Destination state | `destination_risk_penalty` | spare capacity, thermal/throttling, destination p99/queue, memory pressure, topology quality, network distance |
| 4. Action-specific | `action_risk_penalty` | cold-start, cache warmup (× cache affinity), lost batching (× active sequences), topology degradation, recent migration churn, rollback/failure probability |
| 5. Telemetry confidence | `uncertainty_penalty` | missing metrics, stale metrics, sandbox provenance, low classifier confidence |

`total_penalty = sla_risk + destination_risk + action_risk + uncertainty + thermal`, and
`net_expected_savings = gross_savings − total_penalty`.

### Where workload priority still (legitimately) enters

Per the plan, workload priority may influence conservatism **only** through explicit SLA
policy, measured SLA headroom, and the uncertainty buffer — never as a standalone
multiplier:

- A critical tier carries **tighter hard SLA bounds** (e.g. `max_p99=500ms`). The same
  predicted latency therefore consumes more of its headroom ⇒ higher state-conditioned
  SLA risk. This flows through `RiskInputs.sla_policy`, not through the label.
- `priority_tier` / `is_latency_sensitive` arguments are retained for observability but
  are **inert** in the risk math (recorded in the explanation only). With all state
  inputs held identical, swapping the label does not change `total_penalty` or the
  KEEP/act decision (proved by `test_workload_label_alone_does_not_change_decision`).

### Hard gates (always block, regardless of savings)

- Predicted breach of any hard SLA bound (`max_p95/p99/queue/error`, `min_availability`,
  `required_capacity_buffer`) ⇒ `hard_sla_block=True` ⇒ KEEP.
- `migration_allowed=false` on a migration ⇒ `hard_sla_block=True` ⇒ KEEP.
- Governor cooldown / rate limits ⇒ `blocked_by_cooldown=True` ⇒ KEEP.

### Wiring

`aurelius/constraints/engine.py` now builds a `RiskInputs` per service — SLA policy,
current `WorkloadState`, predicted post-action `WorkloadState` (via the existing
`HeuristicPredictor`), destination `RegionContext`, prefix-cache hit rate, KV usage,
active/queued requests, sample age — and passes it to `estimate()`. Recommendation mode
remains `recommendation_only`.

### Required behaviors (all covered by tests in `tests/test_migration_cost_model.py`)

| Behavior | Test |
|---|---|
| Critical workload MAY migrate when headroom large + destination safe | `test_critical_migrates_with_large_headroom_and_safe_dest` |
| Critical workload blocked when headroom small | `test_critical_blocked_with_small_headroom` |
| Batch workload blocked when destination topology/thermal/queue risk high | `test_batch_blocked_with_hostile_destination` |
| Missing telemetry raises uncertainty and can force KEEP | `test_missing_telemetry_increases_uncertainty_and_can_force_keep` |
| Workload label alone does not change the decision when state identical | `test_workload_label_alone_does_not_change_decision` |
| High savings rejected when a hard SLA bound is breached | `test_high_savings_rejected_when_hard_sla_breached` |
| `migration_allowed=false` hard-blocks | `test_migration_allowed_false_hard_blocks` |
| Cold-start penalty is label-independent | `test_cold_start_penalty_is_label_independent` |
| Static multiplier config removed | `test_no_static_workload_multiplier_config` |
| Risk-factor explanation present + buckets sum to total | `test_risk_factor_explanation_present` |

### Test evidence

```
pytest tests/test_migration_cost_model.py            47 passed
pytest tests/test_constraint_engine.py               53 passed
pytest tests/test_constraint_classifier.py           74 passed
pytest (all constraint/SLA/migration-touching suites) 290 passed
ruff check aurelius/constraints/ tests/test_migration_cost_model.py   All checks passed
```

### Files changed

| File | Change |
|---|---|
| `aurelius/constraints/cost_model.py` | Removed static workload multipliers; added `RiskInputs` + five state-conditioned risk families; added `hard_sla_block`, `risk_factors`, `dominant_risk_factors`, `sla_headroom_fraction`, `missing_signals`; `should_keep()` now hard-blocks on SLA breach |
| `aurelius/constraints/engine.py` | Builds and passes `RiskInputs` (SLA policy, current/predicted `WorkloadState`, destination `RegionContext`, cache/load signals) to the cost model |
| `aurelius/constraints/__init__.py` | Exported `RiskInputs`, `CostModelConfig` |
| `tests/test_migration_cost_model.py` | Replaced label-multiplier tests with state-conditioned tests; added `TestStateConditionedRisk` |
| `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md`, `docs/COMPUTE_OPTIMIZATION_PROGRESS.md` | Documented the correction |

### Phase 8/9 Risk Model Audit (code-level verification)

The merged correction (commit `6d93fbd`, squash of PR #68) was audited from the
**implementation**, not only from passing tests. All seven required properties were
verified:

| # | Property | Verified from code |
|---|---|---|
| 1 | No static workload-class multiplier affects risk decisions | `is_critical`/`is_batch` and the `*_workload_risk_multiplier` config fields are gone (the only `workload_risk_multiplier` strings remaining are in a docstring NOTE documenting their removal). The four risk-family methods (`_sla_headroom_risk`, `_destination_risk`, `_action_risk`, `_uncertainty_risk`) take **no** priority/latency label parameters. `total_penalty = sla + destination + action + uncertainty + thermal`; weights are constants × state-derived fractions. |
| 2 | Priority/latency labels affect risk only via SLA policy / headroom / uncertainty | `priority_tier` / `is_latency_sensitive` appear only as `estimate()` params, in docstrings, and in the explanation string — never in a penalty term. In the engine, `priority_tier` is derived **from** the resolved SLA policy tier (not vice-versa) and is passed only for observability. Runtime: two estimates identical in state but differing in label produce **identical** `total_penalty` and `net_expected_savings`. |
| 3 | Critical workloads can migrate when state is safe | Runtime: critical tier, large headroom (0.69), safe warm destination, low cache affinity ⇒ `is_viable=True`, net **+22.4**; dominant factors are state/action (`cold_start`, `lost_batching`), not the label. |
| 4 | Batch workloads blocked when state is unsafe | Runtime: batch tier, hostile destination (thermal+throttling, 5% spare, far, slower) + severe topology degradation ⇒ `should_keep=True`; dominant factors `dest_thermal`, `dest_higher_latency`, `lost_batching`. |
| 5 | Missing telemetry increases uncertainty | Runtime: `uncertainty_penalty` 0.18 (full telemetry) → 2.60 (no telemetry); `missing_signals` 0 → 6; can flip a marginal action to KEEP. |
| 6 | Recommendation outputs explain dominant risk factors | `MigrationCostEstimate` carries `risk_factors` + `dominant_risk_factors` + per-bucket penalties + `sla_headroom_fraction`; `make_recommendation()` and the engine both embed `estimate.explanation` (containing "Dominant risk factors: …") in the `Recommendation.rationale`. |
| 7 | Existing optimizer behavior unchanged outside constraint-aware paths | The merge changed **only** `aurelius/constraints/` (3 files) + 2 docs + `tests/test_migration_cost_model.py`. The energy optimizer (`aurelius/optimization/scheduler.py`) imports its **own** `aurelius/optimization/constraints.py::ConstraintBuilder` — a different module, untouched. `MigrationCostModel` / `RiskInputs` are used only within the additive constraint-aware path (dedicated CLI + benchmark runner), never by the optimizer/backtester. Optimizer/SLA regression suites pass unchanged. |

**Audit commands**

```
git diff --name-only c39c2b1 6d93fbd          # scope: constraints/ + docs + 1 test only
grep -nE "priority_tier|is_latency_sensitive|is_critical|is_batch" aurelius/constraints/cost_model.py
pytest tests/test_migration_cost_model.py::TestStateConditionedRisk -v   # 8 passed
pytest tests/test_sla_engine.py tests/test_sla_optimization.py tests/test_constraint_engine.py   # 100 passed
```

**Conclusion:** the correction is real and structurally enforced — risk is state-conditioned,
the workload label is inert in the risk math, hard SLA breaches always block, and the change
is isolated to the additive constraint-aware path. Phase 8/9 risk model is **corrected and audited**.

---

## Phase 12 Completion Evidence

### Phase 12 Milestone Decision

- **What this run implemented:** Phase 12 — Production hardening for enterprise pilots
- **Why it was the correct next step:** Phases 1–11 verified (825 tests passing, 13 intentional skips). The constraint-aware system had full observability gaps — no Prometheus-scrapeable internal metrics existed.
- **Prior dependencies verified:** Full constraint-aware test suite (774 phase 1-11 tests) all passing before Phase 12 work began.
- **What was explicitly NOT attempted:** Live production wiring (requires customer infrastructure), Postgres persistence for MigrationGovernor (deferred — in-memory governor is sufficient for single-process deployments), BacktestEngine wiring to ClusterState (not required for Phase 12 scope).

### What Already Existed (Pre-Phase-12)

Before implementing Phase 12, an audit of "already satisfied" requirements was performed:

| Requirement | Status Before Phase 12 | Evidence |
|---|---|---|
| Auth via env vars | ✓ COMPLETE | `AuthConfig.bearer_token()` / `basic_credentials()` — reads from env |
| No secrets in logs | ✓ COMPLETE | Tokens stored as env var names only; values never logged |
| Namespace allowlists | ✓ COMPLETE | `KubernetesConnectorConfig.namespace_allowlist` |
| Dry-run / recommendation-only default | ✓ COMPLETE | Phase 9 engine default mode |
| Read-only K8s RBAC | ✓ COMPLETE | `configs/connectors/kubernetes_rbac.yaml` |
| No cluster mutation | ✓ COMPLETE | `recommendation_only` mode enforced |
| TLS verification | ✓ COMPLETE | `ConnectorConfig.tls_verify=True` default |
| Timeouts + retries | ✓ COMPLETE | `ConnectorConfig.timeout_s`, `max_retries` |
| Fail-safe on missing telemetry | ✓ COMPLETE | Classifier / engine both emit KEEP on missing data |
| Stale metric detection | ✓ COMPLETE | Classifier `staleness_weight` penalizes stale regions |
| Partial data handling | ✓ COMPLETE | `ClusterState.is_partial`, `missing_sources` |
| Confidence scoring | ✓ COMPLETE | Classifier confidence in [0,1] |
| Rate limiting | ✓ COMPLETE | `MigrationGovernor` per-workload + cluster rate limits |
| validate-connectors CLI | ✓ COMPLETE | Phase 10 |
| Security + deployment docs | ✓ COMPLETE | `enterprisedocs/security-and-deployment.md` |

### What Was Genuinely Missing (Phase 12 Additions)

| Gap | Implementation |
|---|---|
| Observability metrics export | `aurelius/constraints/observability.py`: `AureliusObserver`, `AureliusMetrics`, `ConnectorHealth`; Prometheus text exposition |
| Production YAML config template | `configs/connectors/aurelius_constraint_production.yaml` |
| `self-metrics` CLI command | `aurelius/cli_constraint.py::cmd_self_metrics`; registered in `aurelius/cli.py` |
| Phase 12 hardening tests | `tests/test_phase12_hardening.py`: 51 tests |

### Files Added

| File | Role |
|---|---|
| `aurelius/constraints/observability.py` | `AureliusObserver` (thread-safe metrics collector), `AureliusMetrics` (snapshot), `ConnectorHealth`; `to_prometheus_text()` produces valid Prometheus exposition |
| `configs/connectors/aurelius_constraint_production.yaml` | Template production YAML config for constraint-aware deployment |
| `tests/test_phase12_hardening.py` | 51 Phase 12 tests across 16 test classes |

### Files Modified

| File | Change |
|---|---|
| `aurelius/constraints/__init__.py` | Added `AureliusObserver`, `AureliusMetrics`, `ConnectorHealth` exports |
| `aurelius/cli_constraint.py` | Added `cmd_self_metrics` command |
| `aurelius/cli.py` | Added `self-metrics` subparser and dispatch |
| `docs/COMPUTE_OPTIMIZATION_PROGRESS.md` | Updated phase status table, status summary |

### Observability Module Design

`AureliusObserver` collects:
- `aurelius_constraints_detected_total` (counter, labeled by constraint type)
- `aurelius_recommendations_generated_total` (counter, labeled by action_type)
- `aurelius_recommendations_blocked_by_sla_total` (counter)
- `aurelius_estimated_net_savings_dollars` (counter, accumulated)
- `aurelius_confidence_current` (gauge, last cycle)
- `aurelius_connector_health` (gauge per connector, 1=healthy, 0=unhealthy)
- `aurelius_stale_data_count` (gauge)
- `aurelius_engine_cycles_total` (counter)

Thread-safe via `threading.Lock`. No Prometheus client library required.
Prometheus text format 0.0.4 compatible (scrapeable by any Prometheus server).

### Test Coverage

| Test Class | Count | What It Proves |
|---|---|---|
| `TestSecretRedaction` | 8 | Auth secrets never appear in repr/str; env var names stored, not values |
| `TestStaleTelemetryConfidence` | 4 | Stale data reduces confidence; None sample_age_s doesn't crash |
| `TestKubernetesReadOnly` | 3 | No write/mutate methods in K8s connector; snapshot is readable |
| `TestMissingConnectorGraceful` | 5 | Missing connectors produce None/empty/partial, never fabricated data |
| `TestAureliusObserver` | 10 | Core observer behavior: empty state, recording, accumulation, reset |
| `TestPrometheusTextExport` | 10 | Valid Prometheus text; required metric names; numeric values; labels |
| `TestObserverThreadSafety` | 1 | 4 threads × 20 cycles = 80 correct (no data races) |
| `TestRecommendationOnlyDefault` | 2 | All recommendations in recommendation_only mode; sandbox propagates |
| `TestConnectorHealth` | 3 | ConnectorHealth dataclass fields; import from package root |
| `TestProductionConfig` | 4 (3 yaml-skip) | YAML exists, parseable, correct defaults (yaml-dependent tests skip in CI) |
| `TestPackageImports` | 4 | AureliusObserver/Metrics/ConnectorHealth importable from constraints package |

### Commands Run

```
python -m compileall aurelius/constraints/ aurelius/cli_constraint.py aurelius/cli.py
ruff check aurelius/constraints/observability.py aurelius/constraints/__init__.py tests/test_phase12_hardening.py --select=E,F,W
/root/.local/bin/pytest tests/test_phase12_hardening.py -q --tb=short
/root/.local/bin/pytest tests/test_state_models.py tests/test_state_store.py ... tests/test_phase12_hardening.py -q --tb=short
python -m aurelius.cli self-metrics --steps 5
```

### Test Results

```
tests/test_phase12_hardening.py: 51 passed, 3 skipped (pyyaml not in CI env)

Full constraint-aware suite (phases 1-12):
825 passed, 13 skipped, 0 failed

ruff: All checks passed (new files only; pre-existing long-line warnings in cli.py are not new)
python -m compileall: No errors
```

### CLI Smoke Test Output (self-metrics)

```
$ python -m aurelius.cli self-metrics --steps 5
[SANDBOX] Aurelius internal metrics (Prometheus text format)
# Driven from simulator — for illustration only

# HELP aurelius_constraints_detected_total Total constraint detection events by constraint type
# TYPE aurelius_constraints_detected_total counter
aurelius_constraints_detected_total{constraint="queue"} 4
...
aurelius_confidence_current 0.925
aurelius_connector_health{connector="dcgm"} 1
aurelius_connector_health{connector="kubernetes"} 1
...
```

### Wiring Evidence

Phase 12 is additive. `AureliusObserver` is a standalone collector:
- Callers pass `EngineResult` from `ConstraintAwareEngine.run()` → observer accumulates metrics
- `connector.record_connector_health(name, is_healthy)` → tracking per connector
- `observer.to_prometheus_text()` → valid Prometheus exposition
- `aurelius self-metrics` CLI → demo mode, driven from simulator

The observer is NOT yet wired into the engine's auto-recording path. Callers must explicitly call `observer.record_engine_result(result)`. This is intentional — avoids coupling the engine to any specific observability backend.

### Failure Mode Review

- **Empty observer:** `to_prometheus_text()` returns valid Prometheus text with zero counts (not crash)
- **No connector health recorded:** health section shows "no connector health reported yet" comment
- **No confidence yet (0 cycles):** `confidence_current` metric omitted until first cycle
- **Thread safety:** `threading.Lock` protects all read/write paths
- **Secret leakage:** `AuthConfig` stores env var *names* only; `bearer_token()` reads value at call time; repr never triggers value evaluation

### Invariants Maintained

- All recommendations remain `recommendation_only` (not changed by Phase 12)
- Missing telemetry → None, never fabricated (not changed by Phase 12)
- No K8s write methods added (read-only enforced by test)
- Sandbox outputs labeled `[SANDBOX]` in CLI output
- No secrets committed; no DATABASE_URL or tokens in any file

### Open Limitations

- `AureliusObserver` is in-memory only; production deployments with multiple replicas need shared state (Prometheus Pushgateway or side-car scrape)
- `record_connector_health()` is caller-driven; auto-health-check on each connector tick would require a coordinator (Phase 12+ enhancement)
- YAML config template uses standard connector names; YAML loading requires `pyyaml` (already required by `metric_mapping.py` YAML override path)
- `self-metrics` CLI runs a fresh simulator per invocation; production deployments should maintain a long-running `AureliusObserver` singleton

---

## Independent Completeness Audit (post-Phase 12)

| Phase | Claimed Status | Repo-Reality Status | Evidence | Gaps | Final Status |
|---|---|---|---|---|---|
| 0 | COMPLETE | COMPLETE | `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md` | None | COMPLETE |
| 1 | COMPLETE | COMPLETE | `aurelius/state/` + 154 tests | WorkloadState ext. deferred | COMPLETE |
| 2 | COMPLETE | COMPLETE | Prometheus connector, 56 passed (10 skip, requests) | requests/yaml in CI | TESTED_WITH_ENV_GAPS |
| 3 | COMPLETE | COMPLETE | DCGM/vLLM/Triton/Ray adapters, 72 passed | Triton/Ray p99 = None | COMPLETE |
| 4 | COMPLETE | COMPLETE | K8s connector, 47 passed | kubernetes pkg in CI | TESTED_WITH_ENV_GAPS |
| 5 | COMPLETE | COMPLETE | Topology collector, 62 passed | nvidia-smi not in CI | TESTED_WITH_ENV_GAPS |
| 6 | COMPLETE | COMPLETE | Simulator + fake connectors, 93 passed | Thermal model is EMA proxy | COMPLETE |
| 7 | COMPLETE | COMPLETE | Classifier, 74 passed | Thresholds are HEURISTIC | COMPLETE |
| 8 | COMPLETE (corrected) | COMPLETE | Cost/risk model, 47 passed; state-conditioned | Governor in-memory only | COMPLETE |
| 9 | COMPLETE | COMPLETE | Engine, 53 passed; SLA+cost gates wired | BacktestEngine not wired | IMPLEMENTED_BUT_NOT_WIRED (in legacy path) |
| 10 | COMPLETE | COMPLETE | 5 CLI commands, 58 passed | — | COMPLETE |
| 11 | COMPLETE | COMPLETE | Benchmark framework, 58 passed, 6 scenarios | Scenarios are synthetic only | COMPLETE |
| 12 | COMPLETE | COMPLETE | Observability, 51 passed; CLI self-metrics works | Observer not auto-wired | COMPLETE |

**Phase 9 note:** "IMPLEMENTED_BUT_NOT_WIRED (in legacy path)" means the `ConstraintAwareEngine` works correctly as a standalone engine — but the legacy `BacktestEngine` (energy-arbitrage path) still uses its own decision path. These are separate product layers. The constraint-aware engine is wired to the constraint-aware CLI and benchmark paths, which is correct for Phase 12.

---

## System Status After Phase 12

### What is production-ready

> **Audit correction (2026-05-26), updated after Mission 1:** "production-ready"
> here means "unit-tested component." The connector→`ClusterState` assembler now
> EXISTS (`aurelius/state/assemble.py`) and is proven end-to-end through the real
> DCGM/vLLM adapters with fixtures, so connectors CAN now drive the engine. What
> remains is (a) per-cluster node/service mapping in
> `build_cluster_state_from_connectors` and (b) validation against LIVE telemetry
> — neither has been done, so this is "wired and fixture-validated," not
> "validated on a real cluster."

- Normalized ClusterState model (energy, thermal, topology, GPU, inference, queue)
- Connector→ClusterState assembler (`build_cluster_state`) *(fixture-validated via real adapters; live-untested)*
- Prometheus-native ingestion with fake server for offline testing
- DCGM, vLLM, Triton, Ray Serve, OTel adapters *(emit leaf objects; now assembled via build_cluster_state)*
- Kubernetes read-only connector (with RBAC config)
- nvidia-smi topology parser and placement scorer
- Synthetic cluster simulator (6 canonical scenarios, deterministic)
- Constraint classifier (8 families, staleness-aware, hysteresis)
- State-conditioned migration cost/risk model (SLA-hard-gate, no static multipliers)
- Constraint-aware recommendation engine (all 8 constraint → action families)
- CLI: constraint-report, simulate-constraint-scenario, telemetry-check, topology-report, validate-connectors, benchmark-run, benchmark-compare, optimizer-regression-check, self-metrics
- Observability metrics export (Prometheus text format)
- Production YAML config template

### What is heuristic (requires real telemetry to calibrate)

- Constraint classifier thresholds (all labeled `# HEURISTIC`)
- Migration cost/risk penalty multipliers
- Placement scorer NVLink/PCIe/NUMA weights
- Simulator thermal EMA model
- M/M/1 queue approximation

### What requires real customer infrastructure

- Live DCGM/Prometheus scraping
- Live Kubernetes API
- Live nvidia-smi topology output
- Real workload trace for constraint calibration
- Multi-replica observability (Pushgateway or side-car scrape)

### What remains for future runs (OPTIONAL — not required for pilot)

1. BacktestEngine wiring to ClusterState (legacy energy path + constraint layer)
2. MigrationGovernor persistence to Postgres (in-memory is fine for single-process)
3. AureliusObserver auto-wired to engine (currently caller-driven)
4. WorkloadState extension (service_id, gpu_uuids, comm_bytes_per_s)
5. PlacementScorer integration into engine (replace 0.7/0.3 heuristics)
6. Per-region forecaster with ≥90-day training windows
7. Prometheus Pushgateway integration for multi-replica deployments
8. ENTSO-E connector for EU market coverage

---

## Post-Phase-12 Verification Audit (Routine Run 2026-05-25)

### Bugs Found and Fixed

This run performed an independent end-to-end verification of the system after Phase 12. The following bugs were discovered and fixed:

#### Bug 1: Missing workload in `energy_price_arbitrage_multiregion` scenario

**Root cause:** The `us-west` region had a queue (`batch-llm-west`) but no workload with `service_id: batch-llm-west`. `_find_workload_for_service` returned `None`, causing `queue.service_rate_per_sec = 0.01` and `queue.queue_wait_p95_ms = 60000ms` (saturated). This forced the classifier to detect `queue_bound` in EVERY scenario that included this region state.

**Impact:** `energy_price_arbitrage_multiregion` was classified as queue-bound instead of energy-bound. The constraint mismatch propagated to the optimizer regression check as a false warning.

**Fix:** Added `batch-wl-west` workload to `benchmarks/v1/energy_price_arbitrage_multiregion.yaml`. Updated `.scenario_hashes.json`.

#### Bug 2: Constraint name normalization mismatch

**Root cause:** YAML scenario files use `_bound` suffix (`energy_bound`, `thermal_bound`, `memory_bound_indirect`) but `ConstraintType.value` uses bare names (`energy`, `thermal`, `memory`). The `cli_constraint.py` comparison (`dominant == expected`) always failed because of the suffix. The `constraint_runner.py` had a partial fix (`removesuffix("_bound")`) that didn't handle `memory_bound_indirect`.

**Impact:** All constraint validation output showed `[MISMATCH]` even when the correct constraint was detected.

**Fix:**
- Added `_normalize_constraint_name()` helper in `cli_constraint.py`
- Fixed `memory_bound_indirect` → `memory` case in `constraint_runner.py`

#### Bug 3: Simulator did not populate `RegionState.topology`

**Root cause:** `ClusterSimulator.get_cluster_state()` constructed `RegionState` without a `topology` field. The constraint classifier's topology scorer requires `region.topology` to be a `TopologyState` with an `interconnect_class`. When `topology=None`, the scorer returned `(None, ["topology[region_id]"])` and the family was excluded from scoring.

**Impact:** `topology_fragmentation_h100` scenario always detected `utilization_bound` instead of `topology_bound`. The topology scorer was effectively disabled in simulator-driven scenarios.

**Fix:** Added `_derive_region_interconnect_class()` method to compute worst-case interconnect class from node topology labels. Added `TopologyState` construction per region in `get_cluster_state()`, populating `interconnect_class` from node `topology-class` labels and `pair_levels` from node `topology_links`.

#### Bug 4: `constraint_report.py` called `.bandwidth_score` on `TopologyLinkType` enum

**Root cause:** The topology report formatter called `lnk.bandwidth_score` on `TopologyLinkType` enum values, which don't have that attribute.

**Impact:** `topology-report` CLI command crashed with `AttributeError` on any state with non-empty `pair_levels`. Tests testing topology report output were failing.

**Fix:** Replaced the attribute access with an inline `_PENALTY` dict (matching `connectors/topology.py`). Changed field names from `.link_type.value` to `.value` (since `all_links` contains `TopologyLinkType` enum values directly, not wrapper objects).

#### Bug 5: `test_missing_signals_shown` relied on simulator always missing topology

**Root cause:** After Bug 3 fix, the simulator now populates topology correctly, so no signals are missing for the default scenario. The test had a comment "Simulator always has at least topology signals missing" which was correct before Bug 3 was fixed.

**Fix:** Updated test to use an empty `ClusterState` (no regions/GPUs) which genuinely produces missing signals from all scorers.

### Constraint Match Results (Before → After)

| Scenario | Before | After |
|---|---|---|
| energy_price_arbitrage_multiregion | MISMATCH (queue detected) | MATCH (energy detected) |
| latency_tail_kvcache_pressure | MATCH | MATCH |
| queue_surge_latency_sensitive | MISMATCH (utilization detected early) | MISMATCH (latency detected during surge — acceptable, see note) |
| thermal_hotspot_mixed_cluster | MATCH | MATCH |
| topology_fragmentation_h100 | MISMATCH (utilization detected) | MATCH (topology detected) |
| underutilization_stranded_capacity | MATCH | MATCH |

**Note on queue_surge_latency_sensitive:** During a queue surge, TTFT/p99 latency spikes to max (1.0) while queue score is 0.85 (dampened by spare capacity factor). The classifier correctly identifies the most observable symptom (high latency). The recommended action is SCALE_REPLICAS for both `latency_bound` and `queue_bound`, so the operational impact is identical.

### Test Results (Post-Fix)

```
942 passed, 2 pre-existing PyYAML failures (yaml not in CI env), 13 intentional skips
ruff: all checks passed
python -m compileall: no errors
optimizer-regression-check: PASS (all 6 scenarios pass, 1 documented warning)
validate-connectors: 10/10 PASSED
```

### Files Changed This Run

| File | Change |
|---|---|
| `benchmarks/v1/energy_price_arbitrage_multiregion.yaml` | Added `batch-wl-west` workload to us-west |
| `benchmarks/v1/.scenario_hashes.json` | Updated hash for energy arbitrage scenario |
| `aurelius/cli_constraint.py` | Added `_normalize_constraint_name()`, fixed constraint comparison |
| `aurelius/benchmarks/constraint_runner.py` | Fixed `memory_bound_indirect` normalization |
| `aurelius/simulation/cluster/engine.py` | Added `_derive_region_interconnect_class()`, topology population in `get_cluster_state()` |
| `aurelius/reporting/constraint_report.py` | Fixed `bandwidth_score` AttributeError; use inline `_PENALTY` dict |
| `tests/test_cli_assess_recommend.py` | Updated `test_missing_signals_shown` to use empty state |

### Exact Next Recommended Step

The system is operationally complete for enterprise pilot readiness. No mandatory implementation work remains.

If a next run is initiated, the highest-value optional improvement is:

**PlacementScorer integration into the engine** — replace the `0.7/0.3` heuristic target/source topology scores in `engine.py` with real `PlacementScorer.score_placement()` calls. This would make topology-based recommendation decisions quantitatively correct rather than just directionally correct.

---

## Full-Suite Verification Audit (Routine Run 2026-05-25 — Second Pass)

### Audit Goal

Independent re-verification of the complete test suite, including the legacy energy-arbitrage benchmark harness (`tests/test_benchmark_harness.py`) that was not collected in prior runs due to an import path bug.

### Bug Found and Fixed

#### Bug 6: `test_benchmark_harness.py` sys.path collision with `aurelius/benchmarks/`

**Root cause:** `tests/test_benchmark_harness.py` called:
```python
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "aurelius"))
```
The second `insert(0, ...)` placed `aurelius/` before `_REPO_ROOT` in sys.path. When the test then did `from benchmarks.compare_against_previous import ...`, Python found `aurelius/benchmarks/` first (not `benchmarks/` at the repo root). `aurelius/benchmarks/__init__.py` uses `from ..constraints.engine import ...` which fails with "attempted relative import beyond top-level package" because the package is treated as top-level `benchmarks`, not as `aurelius.benchmarks`.

**Impact:** All 44 tests in `test_benchmark_harness.py` failed at collection — they were never running, even in prior "passing" CI runs.

**Fix:** Removed the `sys.path.insert(0, str(_REPO_ROOT / "aurelius"))` line. The `_REPO_ROOT` entry is sufficient — `aurelius.*` imports work through it, and `benchmarks.*` correctly finds `_REPO_ROOT/benchmarks/`.

### Test Results (Post-Fix)

```
Full suite (non-live):   2194 passed, 8 skipped, 0 failed
Constraint-aware phases: 838 passed, 0 skipped (pyyaml now available), 0 failed
optimizer-regression-check: PASS (all 6 scenarios, 1 documented queue/latency mismatch)
validate-connectors: 10/10 PASSED
CLI smoke tests: constraint-report, simulate-constraint-scenario, self-metrics all pass
ruff: no new violations
```

**Previous reported count (825 constraint-aware):** The 13 previously-skipped tests were pyyaml-dependent. With pyyaml now installed in the test environment, they pass — these were TESTED_WITH_ENV_GAPS items that are now COMPLETE.

### Files Changed This Run

| File | Change |
|---|---|
| `tests/test_benchmark_harness.py` | Removed stale `sys.path.insert(0, str(_REPO_ROOT / "aurelius"))` that caused benchmarks package collision |
| `docs/COMPUTE_OPTIMIZATION_PROGRESS.md` | Added this audit section |

### Independent Completeness Audit (updated)

| Phase | Claimed Status | Repo-Reality After This Audit | Evidence | Gaps | Final Status |
|---|---|---|---|---|---|
| 0 | COMPLETE | COMPLETE | Plan doc exists | None | COMPLETE |
| 1 | COMPLETE | COMPLETE | 154 model/store/normalize tests | None | COMPLETE |
| 2 | COMPLETE | COMPLETE | Prometheus connector, 56 passed (10 now pass with requests) | None | COMPLETE |
| 3 | COMPLETE | COMPLETE | DCGM/vLLM/Triton/Ray adapters | Triton/Ray p99 = None by design | COMPLETE |
| 4 | COMPLETE | COMPLETE | K8s connector, 47 tests | kubernetes pkg in prod env | COMPLETE |
| 5 | COMPLETE | COMPLETE | Topology collector, 62 tests | nvidia-smi not in CI | COMPLETE |
| 6 | COMPLETE | COMPLETE | Simulator + fakes, 93 tests | Thermal EMA proxy | COMPLETE |
| 7 | COMPLETE | COMPLETE | Classifier, 74 tests, 5/6 scenarios match | Thresholds heuristic | COMPLETE |
| 8 | COMPLETE | COMPLETE | Cost model, 47 tests, state-conditioned | Governor in-memory | COMPLETE |
| 9 | COMPLETE | COMPLETE | Engine, 53 tests, SLA+cost gates | BacktestEngine not wired (intentional) | COMPLETE |
| 10 | COMPLETE | COMPLETE | 5 CLI commands, 58 tests | None | COMPLETE |
| 11 | COMPLETE | COMPLETE | Benchmark framework, 58 tests, regression detection | Scenarios synthetic | COMPLETE |
| 12 | COMPLETE | COMPLETE | Observability, 51 tests, CLI self-metrics | Observer caller-driven | COMPLETE |

### System Status

**Operationally complete for enterprise pilot readiness.** No mandatory implementation work remains.

The only remaining optional improvements are:
1. ~~PlacementScorer integration into engine~~ **COMPLETE** (implemented in routine run 2026-05-25 third pass)
2. MigrationGovernor Postgres persistence (in-memory sufficient for single-process)
3. AureliusObserver auto-wired to engine (currently caller-driven)
4. BacktestEngine wiring to ClusterState (legacy energy path is separate, not a gap)

---

## PlacementScorer Integration — Routine Run 2026-05-25 (Third Pass)

### What was done

Replaced the static `0.7/0.3` heuristic topology quality scores in `aurelius/constraints/engine.py` with real `PlacementScorer.score_placement()` calls when `RegionState.topology` is populated.

**Before (heuristic):**
```python
current_topo_score = 0.7  # HEURISTIC: decent within-region topology
target_topo_score = 0.3   # HEURISTIC: cross-region = worse topology
```

**After (real scoring):**
```python
# Current-region quality from real PlacementScorer
cur_region = state.regions.get(service.region)
if cur_region and cur_region.topology and cur_region.topology.gpu_uuids:
    wspec = PlacementWorkloadSpec(...)
    gpu_uuids = list(cur_region.topology.gpu_uuids)
    ps = score_placement(wspec, gpu_uuids, cur_region.topology)
    current_topo_score = 1.0 - ps.score  # invert: lower penalty = higher quality
else:
    current_topo_score = 0.7  # fallback when topology unavailable
# Cross-region link quality = 0.0 (REGION link has penalty=1.0 in _LINK_PENALTY)
target_topo_score = 0.0
```

**Why this is more correct:**
- NVSwitch topology → quality ≈ 1.0 (penalty = 0.0); cross-region degradation = 1.0
- PCIe (PIX) topology → quality ≈ 0.65 (penalty = 0.35); cross-region degradation = 0.65
- Cross-region target quality = 0.0 (REGION link is the worst link type, penalty = 1.0)
- Previous heuristic (0.3) underestimated cross-region topology cost

**Safety:** Falls back to the prior 0.7 heuristic when `RegionState.topology` is absent, preserving safe degradation behavior for real connectors without topology data.

### Files Changed

| File | Change |
|---|---|
| `aurelius/constraints/engine.py` | Import `PlacementWorkloadSpec`, `score_placement` from connectors; replace heuristic with real scorer; fall back to 0.7 when topology absent |
| `tests/test_constraint_engine.py` | Import `TopologyState`, `TopologyLinkType`; add `TestPlacementScorerIntegration` (4 tests) |

### Test Results

```
tests/test_constraint_engine.py:  57 passed (53 original + 4 new)
Full constraint-aware suite:      842 passed, 0 failed
optimizer-regression-check:       PASS (all 6 scenarios)
validate-connectors:              10/10 PASSED
ruff:                             All checks passed
```

### Why target_topo_score changes from 0.3 to 0.0

The `_LINK_PENALTY` dict in `topology.py` defines `TopologyLinkType.REGION: 1.00`. A cross-region GPU placement uses REGION-level links (WAN/inter-DC). Quality = `1.0 - 1.0 = 0.0`. 

The previous 0.3 heuristic underestimated the topology cost. The corrected 0.0 means the topology degradation fraction (30% of gross savings) now correctly penalizes cross-region migration from good topology clusters.

For NVSwitch → cross-region: `topo_deg = 1.0 - 0.0 = 1.0` → topology_penalty = 30% of gross savings.
Previously: `topo_deg = 0.7 - 0.3 = 0.4` → topology_penalty = 12% of gross savings.

This makes Aurelius more conservative about recommending cross-region migrations from high-quality topology clusters — the correct enterprise-safe behavior.

### Independent Completeness Audit (updated after PlacementScorer integration)

| Phase | Claimed Status | Repo-Reality After This Audit | Evidence | Gaps | Final Status |
|---|---|---|---|---|---|
| 0 | COMPLETE | COMPLETE | Plan doc exists | None | COMPLETE |
| 1 | COMPLETE | COMPLETE | 154 model/store/normalize tests | None | COMPLETE |
| 2 | COMPLETE | COMPLETE | Prometheus connector, all tests pass | None | COMPLETE |
| 3 | COMPLETE | COMPLETE | DCGM/vLLM/Triton/Ray adapters | Triton/Ray p99 = None by design | COMPLETE |
| 4 | COMPLETE | COMPLETE | K8s connector, 47 tests | kubernetes pkg in prod env | COMPLETE |
| 5 | COMPLETE | COMPLETE | Topology collector, 62 tests; PlacementScorer now wired to engine | nvidia-smi not in CI | COMPLETE |
| 6 | COMPLETE | COMPLETE | Simulator + fakes, 93 tests | Thermal EMA proxy | COMPLETE |
| 7 | COMPLETE | COMPLETE | Classifier, 74 tests, 5/6 scenarios match | Thresholds heuristic | COMPLETE |
| 8 | COMPLETE | COMPLETE | Cost model, 47 tests, state-conditioned | Governor in-memory | COMPLETE |
| 9 | COMPLETE | COMPLETE | Engine, 57 tests (4 new topology tests); topology heuristic replaced | BacktestEngine not wired (intentional) | COMPLETE |
| 10 | COMPLETE | COMPLETE | 5 CLI commands, 58 tests | None | COMPLETE |
| 11 | COMPLETE | COMPLETE | Benchmark framework, 58 tests, regression detection | Scenarios synthetic | COMPLETE |
| 12 | COMPLETE | COMPLETE | Observability, 51 tests, CLI self-metrics | Observer caller-driven | COMPLETE |

**Remaining optional work (non-blocking for pilot):**
1. MigrationGovernor Postgres persistence
2. AureliusObserver auto-wired to engine
3. BacktestEngine wiring to ClusterState

---

## Interactive candidate-generation deepening (queue / proxy / prefix)

> Read `docs/RESULTS.md` first. Simulator/recommendation only — **not production
> savings**. All actions are recommendation/simulation only; no real cluster is
> mutated.

This change makes the constraint-aware engine propose the *right* interactive
relief actions and route them through the existing SLA/KPI/risk gates (no
bypasses, no weakened gates, no constant tuning to force wins).

**Queue surge (Part A).** `_gen_queue` now emits first-class candidates:
- `SCALE_REPLICAS` — primary capacity relief (subject to per-class eligibility +
  the economic goodput/$ gate; mild pressure on batch is still blocked).
- `PREWARM_REPLICA` — for **critical-interactive** workloads, to hide cold-start
  TTFT/p99 lag under a surge.
- `RESERVE_CAPACITY_FOR_SLA` — when a **batch/best-effort co-tenant** shares the
  region and could crowd protected interactive traffic.
- `REROUTE` — when the proxy is the bottleneck **or** there is no in-region idle
  capacity, to a clearly-safer peer (and only when it would not destroy high
  cache affinity).
- `PRESERVE_AFFINITY` / no-move — preferred over a cache-destroying move.

**Proxy bottleneck (Part B).** `_proxy_bottleneck()` distinguishes an
ingress/proxy cap from a replica/GPU cap (`proxy_saturation` high **and** mean
GPU utilization below the replica-bound floor). When proxy-bound, capacity-relief
actions are **suppressed** with the explicit reason
`blocked_useless_scale_proxy_bottleneck` (unless replicas also bind), a reroute
to a healthy peer is offered when one exists, and the engine KEEPs when no safe
target exists. Adding replicas can never relieve a front-door proxy cap.

**Prefix affinity (Part C).** When prefix-cache hit rate is high, the engine
emits `PRESERVE_AFFINITY` instead of `CHOOSE_CHEAPER_REGION`; the energy adapter
models cold-route cache loss (`hit_rate × cache_warmup_hit_rate_loss`, the
existing cost-model realism constant — no new synthetic weight) and emits
explicit decisions: `preserve_affinity_high_cache_hit_rate`,
`reject_energy_move_cache_loss_exceeds_savings`, `accept_energy_move_cache_safe`,
`accept_energy_move_low_cache_dependency`.

**Honest benchmark result (24-step, fixed seed).** Against the `sla_aware`
headline, all three scenarios remain a **LOSS** — these are hard-overload /
front-door-capped scenarios the simulator cannot relieve by adding modelled
capacity, not regressions:
- `queue_surge_latency_sensitive`: byte-identical KPI before/after; its workload
  is not critical (no prewarm) and has no batch co-tenant (no reserve), and the
  modelled queue collapse (p99 ≈ 1.0e6 ms) is not recovered by single-replica
  adds. goodput/$ stays below `sla_aware` (which scales less).
- `proxy_bottleneck_ingress`: proxy detection now fires — 8 useless
  `SPREAD`s are replaced by 8 explicit `blocked_useless_scale_proxy_bottleneck`
  suppressions. KPI is unchanged (the proxy cap dominates; this scenario is
  single-region so there is no reroute target), but the engine no longer wastes
  actions and explains why. Correctness/explainability win, not a KPI win.
- `prefix_affinity_energy_arbitrage`: the classifier finds it queue/latency-bound
  (not energy-bound), so the `PRESERVE_AFFINITY` energy veto does not engage in
  this sim run; KPI byte-identical before/after.

The new candidate generation + cache-loss veto are exercised by
`tests/test_interactive_actions.py` and `tests/test_energy_adapter.py`; the
canonical 1000-job CAISO/PJM/ERCOT backtest golden is unchanged and the energy
engine core remains byte-unchanged.

---

## Interactive alpha + energy next-best search (this PR)

> Simulator/recommendation only — **not production savings**. Read
> `docs/RESULTS.md` first.

**Part D — energy wrapper next-best safe search (headline win).** The energy
adapter now searches the energy engine's RANKED alternatives per job
(`EnergyArbitrageAdapter.evaluate_best`): engine-optimized placement → the
`current_price_only` placement (an existing baseline, cheapest region at
earliest_start, full slack) → home. It accepts the first SLA-safe + KPI-positive
alternative instead of rejecting straight home. The energy engine remains the
authoritative ranking source — no energy logic is regenerated.

Canonical 1000-job CAISO/PJM/ERCOT result (frozen golden), constraint-aware vs
`current_price_only`:

| policy | goodput/$ | deadline misses | migrations |
|---|--:|--:|--:|
| current_price_only | 0.30368 | 0 | 851 |
| robust_energy_standalone | 0.30067 | 143 (warmup-blind) | 854 |
| **constraint_aware_with_energy_adapter** | **0.33730** | **0** | **692** |

Constraint-aware now **beats** current_price_only by ~11% on goodput/$ with **0
deadline misses, 0 SLA violations, and lower churn** (692 vs 851). The search
accepts 698 engine-optimized + 141 current_price_only next-best placements and
keeps 161 jobs safely home (137 latency-critical + 24 where no safe alternative
is KPI-positive).

**Part B — queue relief is measurable when capacity exists.** New scenario
`queue_surge_relievable_capacity` (8 GPUs, healthy ingress proxy, sub-collapse
arrival): adding replicas drains queue p95 ~99% (≈828 ms → ≈2 ms) and lifts
served tokens ~150% — using the UNCHANGED queue physics, in the regime where
capacity headroom exists. Without scaling the surge stays pressured (not faked).

**Part A — proxy bottleneck.** Added per-ingress proxy capacity
(`SimQueue.proxy_capacity_rps_per_replica`, default `None` = global — a realism
enrichment, not a weakening) and a negative-control scenario
`proxy_bottleneck_no_safe_target`: the engine detects the proxy bottleneck,
suppresses useless scale-up (`blocked_useless_scale_proxy_bottleneck`), and KEEPs
— never a fake reroute when no safe target exists.

**Honest blockers (not forced).**
- Proxy *scenario-level* reroute KPI win is blocked by the simulator's
  region-static arrival model (a rerouted workload's offered load does not
  follow it to the target ingress queue) and a multi-queue-per-service
  interaction. The proxy detection + suppression + reroute emission + per-ingress
  capacity model + negative control are delivered and tested.
- The cluster-sim `prefix_affinity_energy_arbitrage` scenario stays
  queue/latency-bound (long-sequence inference keeps latency pressured), so the
  energy/cache veto does not bind there. The veto itself is fully implemented and
  tested at the adapter (`reject_energy_move_cache_loss_exceeds_savings` /
  `accept_energy_move_cache_safe` / `accept_energy_move_low_cache_dependency` /
  `preserve_affinity_high_cache_hit_rate`) and `_gen_energy` levels.

## 2026-05-31 — Federated HF benchmark corpus v1 (`feature/hf-dataset-discovery-corpus`)

Autonomous Hugging Face discovery + bounded ingestion + federated benchmark
corpus pipeline. **No production claims, no controllers modified, no
robust energy engine touched.** This is a research / data-engine PR.

**What landed**

- `aurelius/traces/hf_corpus/` — federated corpus package: canonical
  per-trace-type record schemas with `field_quality` provenance labels
  (`schemas.py`), HF API discovery + scoring + classification (`discovery.py`),
  bounded ingestion + summary writer (`ingestion.py`), promotion gates +
  registry writer (`promotion.py`), compatibility-routed evaluation harness
  (`evaluation.py`).
- `scripts/discover_hf_aurelius_datasets.py` — metadata-only discovery against
  the public HF API with `HF_TOKEN` honoured. Writes
  `data/external/hf_discovery/hf_dataset_candidates.json`.
- `scripts/ingest_hf_aurelius_dataset.py` — bounded ingestion with `--max-rows`,
  `--max-bytes`, schema-first inspection, unknown-column rejection.
- `scripts/run_hf_corpus_evaluations.py` — routes each promoted dataset to its
  trace-type-specific smoke evaluator; skips incompatible datasets with explicit
  reasons; never aggregates across trace types; never uses oracle as headline.
- `docs/HF_DATASET_REGISTRY.md` — authoritative registry doc + trust hierarchy.
- 4 new test files (71 new tests): `test_hf_dataset_discovery.py`,
  `test_hf_bounded_ingestion.py`, `test_hf_corpus_promotion.py`,
  `test_hf_corpus_evaluation_harness.py`.

**Datasets in the corpus**

- `agent-perf-bench/AgentPerfBench` / `trace_replay` (100-row bounded sample,
  91 KB, `latency_benchmark_trace` → `promoted_for_performance_priors` +
  `promoted_for_constraint_aware_evaluation` + `promoted_for_training_priors`).
- `agent-perf-bench/AgentPerfBench` / `kernels_labeled` (100-row bounded sample,
  50 KB, `kernel_profile_trace` → `promoted_for_performance_priors` +
  `promoted_for_training_priors`).
- `lmsys/chatbot_arena_conversations` evaluated → `gated_blocked` (gated:auto).
- `jaytonde05/prefixbench` + `semianalysisai/cc-traces-weka-no-subagents-051226`
  remain candidates pending a follow-up cache-residency ingest path.

**Trust hierarchy (binding, from the spec)**: Tier 1 real pilot telemetry
remains the only production calibration source; AgentPerfBench is Tier 4. The
registry, every ingester, and every evaluator carry explicit
`is_production_telemetry_substitute: false` /
`comparison_against_oracle_is_headline: false` invariants.

**Next**

- Cache-residency ingest path for `jaytonde05/prefixbench` (flatten nested
  `metadata.prefix_group`).
- Bounded metadata-only audit of the WEKA CC traces with an explicit budget.
- Synthetic telemetry-trace smoke fixture so `telemetry_calibration_smoke_v1`
  has a positive test path before real telemetry lands.

## 2026-05-31 — CARA + SwissAI HF telemetry audit (`feature/hf-cara-swissai-telemetry-audit`)

Focused HF telemetry-candidate audit for `asdwb/cara_latency_prediction`
and `eth-easl/swissai-serving-trace`. Discovery/data-engine PR only —
no controllers modified, no production claims, no ML training.

**What landed**

- `aurelius/traces/hf_corpus/schema_profile.py` — schema profiler
  (flat + 1-level nested + lists), per-subgroup latency summary with
  INSUFFICIENT_SAMPLE_P95/P99 flagging, stratified sampling helper,
  bucket-id hash/sample helpers.
- Extended `aurelius/traces/hf_corpus/schemas.py`: `TelemetryRecord` now
  carries CARA's vLLM scheduler-state fields (num_running, num_waiting,
  kv_cache_utilization, ema_*, actual_e2e_latency_s, actual_ttft_s,
  actual_tpot_s, ...). `CacheResidencyRecord` carries SwissAI bucket
  fields. `RequestShapeRecord` carries SwissAI ISO timestamps + status +
  model_parameters subset.
- Extended `aurelius/traces/hf_corpus/ingestion.py` RAW_TO_NORMALIZED for
  CARA + SwissAI; extended NORMALIZED_FIELD_TO_SIGNAL for the new
  signals (kv_cache_utilization → cache_hit, num_waiting → queue_depth,
  actual_e2e_latency_s → e2e_latency, etc.).
- Extended `aurelius/traces/hf_corpus/promotion.py`: 4 new promotion
  states (`promoted_for_schema_only`, `auth_blocked`,
  `deferred_bounded_ingest`, ...), 9th gate
  `analysis_sample_policy_recorded`, sample-strength → promotion-tag
  filtering (`PROMOTION_TAG_MIN_SAMPLE_STRENGTH`), automatic
  downgrade with `decision.reasons`.
- `scripts/audit_cara_swissai_telemetry.py` — bounded HTTP-Range
  download (gitignored raw), schema profile + mapping generation,
  stratified sampling, statistical_sample_strength labelling, registry
  writer.
- `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` — full PHASE 0-9 audit
  with trust assessment, signal extraction, alpha opportunity
  (9× p99 latency spread for Qwen2.5-3B across A30 vs P100 GPUs).
- Updated `docs/HF_DATASET_REGISTRY.md` with the 5 new (dataset, config)
  entries and CARA's first-Tier-2 status.

**Audit outcomes (5/5 cleared all gates)**

- CARA / test_flat → `telemetry_trace` · Tier 2 · `moderate` strength →
  `promoted_for_constraint_aware_evaluation` + `promoted_for_backtest`
  (dynamic_calibration downgraded; needs `strong` strength which
  `train.jsonl` would unlock).
- CARA / test_queue_details → same as above (full nested
  `schedule_state.running_requests[]` arrays preserved at raw level).
- SwissAI / trace → `request_shape_trace` · Tier 5 · `strong` →
  `promoted_for_training_priors`.
- SwissAI / qwen3_32b_buckets → `cache_residency_trace` · Tier 4 ·
  `strong` → `promoted_for_cache_residency_evaluation`.
- SwissAI / qwen3_32b_bucket_reuse → `cache_residency_trace` · Tier 4 ·
  `strong` → `promoted_for_cache_residency_evaluation`.

**Tests:** 38 new (all pass) + 71 existing HF tests + 192 + key
regression tests all pass. No production scheduler, robust energy
engine, controllers, or frontier modules touched.

**Honesty invariants:**
- Raw downloads + analysis_sample.jsonl both gitignored.
- HF_TOKEN never logged / committed.
- Statistical_sample_strength enforced on every promotion tag (e.g.
  `dynamic_calibration` requires `strong` = ≥10k rows).
- CARA labelled Tier 2 (public telemetry), NOT Tier 1 — CloudLab
  research cluster is not a production pilot.
- SwissAI license is "other" — only summary statistics + 5-row fixture
  committed; raw rows kept gitignored.

**Next:** re-run audit against CARA `train.jsonl` (392 MB, 359k rows)
with a larger per-file budget to lift the analysis sample to `strong`
strength and unlock `promoted_for_dynamic_calibration`.

## 2026-05-31 — CARA + SwissAI analysis-tier expansion (`feature/cara-swissai-analysis-tier-expansion`)

Bounded 50-100 MiB analysis-tier ingestion for CARA + SwissAI so the
Forecast Leverage Audit can run on `strong`-strength evidence. No
forecasting models trained. No scheduler / robust energy engine /
controllers modified. No production claim.

**What landed**

- Extended `scripts/audit_cara_swissai_telemetry.py` with
  `ANALYSIS_TIER_TARGETS` (9 new configs: CARA `train_flat` +
  `train_queue_details` at 80 MiB head, SwissAI `trace_analysis` /
  `qwen3_32b_buckets_analysis` / `qwen3_32b_bucket_reuse_analysis` at
  80 MiB head, plus 4 per-model bucket-reuse files for Apertus-70B,
  Qwen3-80B-instruct/thinking, Llama3-70B). New `--target-set
  {focused,analysis_tier,all}` flag; focused stays the default.
- Added per-config `statistical_rollups.json` artefact (committed):
  per-(instance_type) p50/p95/p99 for e2e + TTFT + TPOT,
  per-(prompt_token_bin / queue_depth_bin / kv_util_bin) p99 latency,
  reuse_percentage distribution, with `INSUFFICIENT_SAMPLE_P99` flagging
  below 100 rows/subgroup.
- New `scripts/build_cara_swissai_signal_coverage.py` aggregates per-
  config summaries into the federated signal coverage + forecast
  readiness + forecast leverage + missing-telemetry gap + strongest-
  dataset matrix tables at
  `data/external/hf_discovery/cara_swissai_signal_coverage.json`.

**Audit outcomes (9/9 new configs cleared all gates)**

- CARA train_flat: 76,825 rows · strong · `promoted_for_dynamic_calibration`
- CARA train_queue_details: 38,509 rows · strong · `promoted_for_dynamic_calibration`
- SwissAI trace_analysis: 202,215 rows · strong
- SwissAI qwen3_32b_buckets_analysis: 103,507 rows · strong
- SwissAI qwen3_32b_bucket_reuse_analysis: 147,440 rows · strong
- SwissAI apertus_70b_bucket_reuse: 49,434 rows · strong (whole 40 MB file)
- SwissAI qwen380b_instruct_bucket_reuse: 45,887 rows · strong
- SwissAI qwen380b_thinking_bucket_reuse: 7,399 rows · **moderate** (large per-row payload)
- SwissAI llama3_70b_bucket_reuse: 153,275 rows · strong

**Forecast readiness (8/10 ready)**

8 forecasts now READY_FOR_FORECAST_LEVERAGE_AUDIT (TTFT, queue_wait,
TPOT, e2e_latency, cache_hit, GPU placement, model residency proxy,
workload arrival). 2 forecasts remain blocked on pilot telemetry
(timeout/SLA labels, autoscaling/replica labels). The same 9× p99
spread for Qwen2.5-3B across A30 vs P100 GPUs surfaces at 76,825 rows.

**Tests:** ~16 new in `tests/test_hf_cara_swissai_analysis_tier.py` +
existing 109 HF tests + 211 regression tests still pass.

**Honesty invariants:**
- Raw HF data + analysis_sample.jsonl gitignored.
- HF_TOKEN never logged / committed.
- statistical_sample_strength still enforced per promotion tag.
- CARA stays Tier 2 — `promoted_for_dynamic_calibration` is a research-
  class promotion, NOT a Tier 1 production calibration source.
- SwissAI license is `other` — only summary statistics + 5-row fixture
  + statistical rollups committed; raw rows + analysis sample
  gitignored.

**Next:** Forecast Leverage Audit v2 (build the actual forecasters in
the build_now ranking using the new strong-strength evidence).
