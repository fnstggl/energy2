# Compute Optimization Progress Tracker

This is the canonical progress tracker for Aurelius constraint-aware GPU orchestration.

This tracker is separate from `docs/AURELIUS_PROGRESS.md`.

`docs/AURELIUS_PROGRESS.md` may contain legacy energy-optimization or general Aurelius progress. It may be useful historical context, but it is NOT the source of truth for this constraint-aware orchestration initiative.

The source planning document is:

`docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md`

Every implementation run must read that plan before deciding what to do next.

---

## Status Summary

Current status: **PHASE 0 COMPLETE / PHASE 1 NOT STARTED**

Phase 0 produced:
- the canonical architecture plan
- the implementation phase map
- connector/API research
- simulator/benchmarking strategy
- trust-boundary rules
- anti-checklist rules
- anti-simulator-overfitting rules

Phase 1 has not begun yet.

The next expected milestone is:

**Phase 1 — Normalized state model**

Expected Phase 1 files:
- `aurelius/state/__init__.py`
- `aurelius/state/models.py`
- `aurelius/state/store.py`
- `aurelius/state/normalize.py`

Expected Phase 1 tests:
- `tests/test_state_models.py`
- `tests/test_state_store.py`
- `tests/test_state_normalize.py`

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
| 1 | Normalized state model | NOT_STARTED | None yet | Next milestone |
| 2 | Prometheus-native connector | NOT_STARTED | None yet | Depends on Phase 1 |
| 3 | DCGM/vLLM/Triton/Ray adapters | NOT_STARTED | None yet | Depends on Phase 2 |
| 4 | Kubernetes connector | NOT_STARTED | None yet | Depends on Phase 1/2 |
| 5 | Topology collector | NOT_STARTED | None yet | Depends on Phase 1 |
| 6 | Synthetic cluster simulator | NOT_STARTED | None yet | Depends on state/connectors |
| 7 | Constraint classifier | NOT_STARTED | None yet | Depends on Phase 1 and simulator fixtures |
| 8 | Cost/risk/migration model | NOT_STARTED | None yet | Depends on classifier + SLA/state models |
| 9 | Constraint-aware recommendation engine | NOT_STARTED | None yet | Requires SLA wiring audit |
| 10 | CLI reports | NOT_STARTED | None yet | Depends on classifier/engine |
| 11 | Validation + benchmarking loop | NOT_STARTED | None yet | Multi-run continuous improvement |
| 12 | Production hardening | NOT_STARTED | None yet | Final enterprise pilot readiness |

---

## Phase 1 Expected Scope

Phase 1 should create the canonical normalized cluster state layer.

Expected additions:
- normalized dataclasses
- provenance model
- UTC-aware timestamp validation
- `None`-not-zero missing telemetry behavior
- range validation
- JSON-compatible serialization/deserialization if practical
- in-memory append-only snapshot store
- leakage-safe `last_known_at_or_before()` lookup
- adapters from existing models where possible

Expected non-goals:
- no Prometheus connector
- no DCGM/vLLM/Triton/Ray adapter
- no Kubernetes connector
- no topology collector
- no simulator
- no classifier
- no optimizer change
- no execution/mutation behavior

Phase 1 should preserve existing optimizer behavior.

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

* The SLA engine exists but may not yet be wired into real optimizer/backtest paths.
* Existing telemetry scaffolding may be synthetic or fixture-based.
* Constraint-aware orchestration is not yet implemented.
* Simulator does not yet exist as a full cluster digital twin.
* Benchmarking does not yet prove multi-constraint optimization.
* Phase completion claims must be verified against repo reality.

⸻

Latest Run Log

Phase 0

Status: COMPLETE

Summary:

* Created canonical implementation plan.
* No production constraint-aware implementation yet.
* Next milestone is Phase 1 normalized state model.

Evidence:

* docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md

Open limitations:

* No normalized aurelius/state/ model yet.
* No constraint classifier yet.
* No simulator yet.
* No Prometheus-native ingestion yet.
* No constraint-aware engine yet.
