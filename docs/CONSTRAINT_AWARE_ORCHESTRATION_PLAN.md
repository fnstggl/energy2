# Aurelius — Constraint-Aware GPU Orchestration: Canonical Implementation Plan

This document is a planning artifact, not proof of correctness.

Future implementation phases MUST NOT assume:
- the plan is complete,
- the repo still matches the plan,
- prior phases were implemented correctly,
- passing a checklist means the feature works.

For every implementation phase, the agent must:

1. Re-read the high-level product goal
2. Re-read this document
3. Independently inspect the current repo state
4. Compare repo reality against the plan
5. Identify gaps the plan missed
6. Identify assumptions invalidated by implementation
7. Verify real code paths are wired
8. Run tests against actual behavior
9. Audit failure modes and missing telemetry
10. Update this document if reality differs from the plan

A phase is NOT complete merely because:
- files were added,
- functions exist,
- tests pass in isolation,
- or checklist items were checked.

A phase is complete only when:
- the implementation is wired into the real execution path,
- the behavior changes correctly in end-to-end scenarios,
- missing telemetry fails safely,
- old behavior is preserved when disabled,
- CLI/demo paths work if relevant,
- sandbox and real connectors share the same interfaces,
- and evidence is provided.

The implementation should optimize for:
- real operational correctness,
- safety,
- observability,
- and enterprise deployability,

NOT:
- maximizing apparent feature completeness,
- satisfying the plan mechanically,
- or creating placeholder abstractions disconnected from real execution paths.

This document itself may be incomplete or wrong.

Even if the files outlined in the phase are added, other supporting files and tests may be necessary to add to achieve the actual goal of the phase. Think critically about what is necessary and implement it if it does not currently exist.

Future phases are expected to critically evaluate and revise it when necessary.

---

> **Document status:** Phase 0 (audit + research + plan). No production code has been written for constraint-aware orchestration as of this document. The repo audit below reflects the state on the date noted in §2.
>
> **Provenance of external facts:** Connector details in §6 were researched against official documentation (URLs inline). Where a doc was ambiguous or unverified it is explicitly flagged `[UNVERIFIED]` or `[AMBIGUOUS]`. Do not treat any metric name or API signature as correct without re-verifying against the cited URL at implementation time — upstream projects rename metrics (see the vLLM V0→V1 split and DCGM `CLOCK_THROTTLE_REASONS`→`CLOCKS_EVENT_REASONS` rename documented in §6).

---

## 1. Executive Summary

### What Aurelius is today

Aurelius is a mature (~36k LOC in `aurelius/`, 45 test files, ~798 passing tests at last verified run) **energy-aware batch-compute optimizer**. Its core loop:

1. Ingests real wholesale electricity prices (CAISO OASIS, PJM Data Miner, ENTSO-E), carbon intensity (ElectricityMaps, WattTime), and — more recently — synthetic/fixture GPU telemetry (DCGM) and queue state.
2. Represents work as batch `Job` objects with deadlines, region options, power profiles, and migration costs.
3. Runs a `JobScheduler` that decides **when** (time-shift), **where** (region routing), **how fast** (power throttle), and **whether to migrate mid-run** to minimize a single scalarized objective: `α·energy + β·carbon + γ·risk + δ·SLA_penalty + data_transfer + queue_delay + gpu_health`.
4. Validates with leakage-free walk-forward backtesting against `current_price_only` and other baselines, reports savings with bootstrap confidence intervals, and can optionally execute through gated, dry-run-by-default adapters (Kubernetes / Slurm / AWS Batch).

### What Aurelius is becoming

Aurelius is evolving into a **constraint-aware GPU orchestration control plane**. The defining new capability:

> **Detect the current *binding constraint* of a GPU/inference cluster, then optimize scheduling, routing, placement, deferral, spreading, bin-packing, and migration decisions around that constraint while respecting SLAs.**

The conceptual shift is from *"add every signal as a cost term and minimize the weighted sum"* to *"first classify which resource/condition is actually limiting the cluster right now (energy, thermal, queue, latency, communication, memory pressure, topology, utilization), then choose the optimization strategy appropriate to that constraint."* Energy-cost optimization becomes **one mode among several**, selected when energy is the binding constraint — not the universal objective.

### What Aurelius is NOT becoming

Aurelius remains an **orchestration / control-plane intelligence layer**. It is explicitly **not** a resource manager, runtime, or inference engine. The following are out of scope and must never be implemented:

- direct KV cache ownership or management
- memory allocator changes
- NCCL replacement or collective-algorithm modification
- CUDA / runtime manipulation
- model execution changes
- inference kernel / compiler changes
- direct mutation of customer clusters **by default** (live execution stays opt-in, gated, dry-run-first)

Aurelius **detects** memory/cache pressure, communication bottlenecks, and topology fragmentation and **recommends safe orchestration responses** (placement, routing, spreading, deferral, scaling hints). It never reaches into the inference data plane. This boundary is already structurally enforced today (see §3 and §8) and must be preserved.

### The three architectural facts that shape this plan

1. **The scaffolding already exists, the classifier does not.** The objective function (`aurelius/optimization/objective.py`) already carries `queue_delay_cost` and `gpu_health_cost` terms (default-off via zero weights), and the SLA engine (`aurelius/sla/`) already has the full constraint-aware *action vocabulary* (`MIGRATE`, `REROUTE`, `DEFER`, `SCALE_REPLICAS`, `CONSOLIDATE`, `SPREAD`, `CHOOSE_CHEAPER_REGION`) plus thermal/capacity-aware risk scoring. **What is missing is the binding-constraint classifier and the normalized ClusterState it would consume.** This plan builds the classifier on top of existing scaffolding rather than greenfield.

2. **The trust boundary is already enforced.** Execution adapters are real but default to `dry_run`; live execution requires a signed policy bundle, a kill-switch check, and a fail-closed quantile safety gate. "No mutation by default" is not aspirational — it is the current default (see §8).

3. **The SLA engine is built but not wired into the real optimizer path.** `JobScheduler` accepts `sla_registry`/`region_contexts`/`current_states` and has complete correction logic, but no CLI/backtest path passes them. This is the single biggest "exists but inert" gap and a priority wiring target (see §3).

---

## 2. Phase 0 Codebase Baseline Snapshot

> **This section describes the repo state at the time Phase 0 was run. It may become outdated after later phases. Future phases must inspect the live repo directly and must not rely on this section as authoritative.**

Snapshot taken on branch `claude/loving-hopper-uzfZq`, repo `fnstggl/energy2`, working dir `/home/user/energy2`. Most recent relevant commit: `fc8dafe Add SLA ingestion + SLA-aware optimization correction engine` (the SLA layer is the newest addition). Package version `1.0.0`, Python `>=3.10`.

### 2.1 Optimizer decision paths (the core)

| File | Role |
|---|---|
| `aurelius/models.py` (504 lines) | Core dataclasses: `Job`, `EnergyPrice`, `CarbonIntensity`, `ScheduleSegment`, `ScheduleDecision`, `SimulationResult`, `QueueState`, `OptimizationConfig`, `GPUMetrics`, `GPUHealthScore`. Workload-type tables (`WORKLOAD_DEFAULT_*`). |
| `aurelius/optimization/scheduler.py` (1123 lines) | `JobScheduler` — the optimizer. Methods: `solve(method=...)` dispatching to `_solve_greedy`, `_solve_local_search`, `_solve_milp` (PuLP), and migration variants `greedy_migrate`, `greedy_migrate_dp`, `local_search_migrate(_dp)`. `_find_best_slot` enumerates time×region×power. Migration DP (`_try_optimal_migrations`) and receding-horizon `replan_remainder` (MPC). **SLA-aware correction hooks already present** (`_sla_policy_for`, `_evaluate_sla_candidate`, `_sla_adjusted_score`, `_sla_forbids_migration`) — active only when `sla_registry` is passed. |
| `aurelius/optimization/objective.py` (339 lines) | `ObjectiveFunction.calculate(...)` → `ObjectiveComponents`. Scalarized objective: `α·energy + β·carbon + γ·risk + δ·SLA_penalty + data_transfer + queue_delay + gpu_health`. Has `_lookup_last_known` (leakage-safe). `queue_delay_cost` and `gpu_health_cost` terms exist, gated by zero-weight config defaults. |
| `aurelius/optimization/constraints.py` (367 lines) | `ConstraintBuilder` — feasibility checks: earliest_start, deadline, region validity, power bounds, regional power caps (`would_violate_power_cap`). Returns `ConstraintViolation` list. |

**Optimizer entry-point flow (the REAL path):**
```
CLI `backtest` (aurelius/cli.py: cmd_backtest)
  → BacktestEngine(method, train_days, eval_days, config, forecaster_cls, ...)   [aurelius/backtesting/engine.py]
    → per fold: scheduler = JobScheduler(config)
              scheduler.solve(eval_jobs, opt_prices, forecast_carbon_data,
                              method=..., queue_data=queue_data,
                              gpu_health_data=gpu_health_data)
  → SavingsReport.generate(rounds) → text/JSON, optional HTML
```
`queue_data` and `gpu_health_data` ARE threaded through; `sla_registry` is NOT (defaults to `None`).

The `simulate` CLI path is separate and goes `cmd_simulate → SimulationReplay.run → ScenarioComparator.compare → JobScheduler.solve` (FIFO + peak-blind baselines vs optimized).

### 2.2 Scheduler models / solving strategies

- Greedy (priority/deadline sort → per-job best slot enumeration over `[1.0, 0.75, 0.5]` power × feasible start hours × region options).
- Local search (greedy seed → single-job re-placement until no >0.1% improvement or time limit).
- MILP (PuLP `PULP_CBC_CMD`, binary `x[job,time,region]`, full-power only) — falls back to local search if PuLP missing.
- Migration post-processing: single-split heuristic and exact DP over `(useful_hours_done, region, num_migrations)`; `replan_remainder` for MPC re-planning of in-flight jobs.

### 2.3 SLA engine

| File | Role |
|---|---|
| `aurelius/sla/schema.py` | `HardSLA` (blocking constraints: regions, latency p95/p99, queue wait, availability, error/timeout rate, migration governance, capacity buffer), `SoftSLA` (penalizing preferences), `SLAPolicy`, `PriorityTier` (critical/latency_sensitive/standard/flexible/batch with safest→cheapest defaults), `OptimizationAggressiveness`, `apply_tier_defaults`. |
| `aurelius/sla/evaluator.py` | `evaluate_action_against_sla(...)` → `SLAEvaluation` (allowed, violated_hard_constraints, soft_penalty_score, `RiskBreakdown` incl. `thermal_risk_penalty`/`capacity_risk_penalty`, corrected_action). `block_on_unknown` for fail-closed. |
| `aurelius/sla/actions.py` | `ActionType` enum (MIGRATE, REROUTE, DEFER, SCALE_REPLICAS, CONSOLIDATE, SPREAD, CHOOSE_CHEAPER_REGION, CHOOSE_LOWER_CARBON_REGION, CHANGE_PLACEMENT, KEEP), `OptimizationAction`. |
| `aurelius/sla/telemetry.py` | `WorkloadState` (latency, queue, util, error/timeout/availability, capacity_buffer, energy/carbon), `RegionContext` (spare_capacity, baseline latency/queue, `thermally_stressed`, `throttling`, `network_rtt_ms`), `TelemetryProvider` Protocol, `StaticTelemetryProvider`, `HeuristicPredictor` (explicitly marked placeholder, `# HEURISTIC` constants). |
| `aurelius/sla/loader.py` | `SLALoader` (load_file/load_dir/load_text, JSON+YAML), `SLARegistry` (resolve by workload id → type → default), `policy_from_dict`, `SLAValidationError`. |
| `aurelius/sla/selector.py` | `SLAAwareActionSelector`, `SLADecision`, `ScoredAction`. |
| `aurelius/sla/report.py` | `SLAReport`. |

### 2.4 Telemetry / ingestion modules

| File | Role | Real vs synthetic |
|---|---|---|
| `aurelius/ingestion/dcgm_provider.py` | `DCGMProvider`: parse Prometheus text, live Prometheus `/api/v1/query`, direct dcgm-exporter `/metrics` scrape, CSV, `.prom` fixtures, `generate_fixture`. `score_gpu_health` → util/thermal/throttle/ECC penalties; `aggregate_region_health`; `to_dict_lookup()` → `{region:{ts:penalty}}`. Bearer/basic auth, TLS toggle. | Live path real; default usage is fixture/synthetic. **See §6.4 for metric-name correctness issues.** |
| `aurelius/ingestion/queue_provider.py` | `QueueProvider`: CSV + fixture, leakage-safe `get_wait_hours`, `to_dict_lookup()` → `{region:{ts:est_wait_hours}}`. | CSV real; default synthetic. |
| `aurelius/ingestion/energy_prices.py`, `market_data_provider.py`, `region_registry.py`, `grid_apis/*` | Real ISO/TSO connectors: `caiso.py` (OASIS, no key), `pjm.py` (Data Miner, `PJM_API_KEY`), `ercot.py`, `entsoe.py` (`ENTSOE_API_KEY`), `eia.py`, `electricitymaps.py`, `watttime.py`, `csv_importer.py`, `market_registry.py`. `region_registry.py` maps canonical regions → ISO + EM zone + carbon zones + cloud aliases + `Confidence`. | **Production-grade**, with sandbox-rejection discipline. |
| `aurelius/ingestion/workload_traces.py`, `job_logs.py` | Customer workload CSV ingestion; synthetic job generation. | Mixed. |

### 2.5 Simulation modules

| File | Role |
|---|---|
| `aurelius/simulation/workload_simulator.py` | `WorkloadSimulator.generate(workload_type, gpu_type, n_jobs, seed)` — seeded, GPU-typed `Job` generation with per-type power/runtime/SLA profiles. GPU power table (h100 700W … t4 70W). |
| `aurelius/simulation/replay.py` | `SimulationReplay.run(config)` — orchestrates synthetic price/carbon + forecasting + optimization + comparison + DB save. `SimulationConfig` has `random_seed`. |
| `aurelius/simulation/compare.py` | `ScenarioComparator` — FIFO + peak-blind ASAP baselines vs optimized; `ComparisonResult`. |
| `aurelius/simulation/metrics.py` | `MetricsCalculator` / `ScheduleMetrics` — energy cost, compute cost (vCPU estimate), carbon, peak power, makespan, region distribution. **No latency/queue-wait/SLA-violation/thermal-throttle/migration-count metrics yet.** |

> **Key gap:** This "simulator" is a price/carbon/job generator + schedule comparator. It is **not** a live discrete-event cluster simulator with queues, arrivals, thermal dynamics, topology, or fake connector endpoints. The Synthetic Cluster Simulator in §9 is a major new build.

### 2.6 Execution / shadow / safety (the trust boundary)

| File | Role |
|---|---|
| `aurelius/execution/base.py` | `Executor` ABC, `ExecutionConfig(mode="dry_run"...)` (default dry-run), `ExecutionResult`, `log_execution_audit`, `is_kill_switch_active()` (`AURELIUS_KILL_SWITCH`). |
| `aurelius/execution/{kubernetes,slurm,aws_batch}.py` | Real submission adapters, each gated by dry-run/kill-switch/guardrails. |
| `aurelius/execution/policy.py` | Signed policy bundles (Ed25519 + HMAC fallback), `authorize_execution()` — live mode requires valid, unexpired, enabled, constraint-matched policy. |
| `aurelius/execution/constraints.py` | `batch_optimized` vs `latency_safe` profiles; `apply_constraint_filter` (substitutes baseline for violators, does not mutate decisions). |
| `aurelius/execution/post_execution.py` | Observability-only realized-outcome recording (read-only, failure-tolerant). |
| `aurelius/execution/shadow_runner.py`, `aurelius/shadow/*` | Read-only replay of decisions vs realized prices; `DecisionRecord`, `RealizedSavingsCalculator` (RT prices only post-hoc — leakage-safe), `DecisionRecorder` (JSONL). |
| `aurelius/safety/quantile_gate.py` | `QuantileSafetyGate` — fail-closed (missing forecast/p50/p90/baseline → BLOCK); filters risky decisions; structured audit logs. |
| `aurelius/learning/{locking,promotion}.py` | fcntl lock; honest candidate-vs-active model promotion on leakage-free holdout. |

### 2.7 CLI / reporting / ROI

- `aurelius/cli.py` (1773 lines, **argparse**). 11 commands: `simulate`, `generate-data`, `backtest`, `show-schema`, `sla-report`, `robustness-test`, `roi`, `db`, `shadow run|realize|report`. `aurelius/__main__.py` → `cli.main()`. Entry point `aurelius = "aurelius.cli:main"`.
- `aurelius/reporting/savings_report.py` — `SavingsReport.generate(rounds, n_bootstrap, primary_baseline)` → dict with totals, 95% bootstrap CIs, per-baseline comparison, per-fold results, methodology. Tracks latency violations + queue delays.
- `aurelius/reporting/html_report.py` — `render_html_report(dict)` → self-contained HTML with base64 matplotlib charts.
- `aurelius/roi/calculator.py` — `ROICalculator` projects savings from benchmark rates (`BENCHMARK_SAVINGS_RATES` per workload type).
- `aurelius/api/app.py` — FastAPI: `/health`, `POST /simulate`, `GET /simulations[/{id}]`. `AURELIUS_API_KEY` header auth.

### 2.8 Benchmark harness

- `benchmarks/run_benchmark.py` (1161 lines): drives workload×region cells, compares optimizer vs all baselines, writes `benchmarks/results/*.json`. `--quick` for CI smoke (results invalid for claims).
- `benchmarks/baseline_matrix.yaml`: `current_price_only` (**primary**), `fifo`, `peak_blind_asap`, `latency_first`, `closest_region`, `fixed_primary_region`, `round_robin`, `oracle` (diagnostic only). Implementations in `aurelius/backtesting/baselines.py`.
- `benchmarks/{workload,region}_matrix.yaml`, `benchmark_config.yaml` (horizons [24,36,48,72], train_days 30, eval_days 7, `regression_threshold_pct: 2.0`).
- `benchmarks/compare_against_previous.py`: regression checker (exit 1 on >threshold regression).
- Determinism via `seed=42`. **No versioned immutable scenario tree (`benchmarks/v1/`) and no config/SLA/workload-mix hashing in result metadata** (see §9).

### 2.9 Config & dependencies

- `OptimizationConfig` is a dataclass built **inline from CLI flags** — no YAML config file is loaded for the optimizer. SLA configs (`configs/sla_examples/*.yaml|json`) are loaded **only** by `cmd_sla_report`.
- `tests/conftest.py`: 12 fixtures (`sample_regions`, `t0`, price/carbon CSV + df + dict, `sample_jobs`, `opt_config`, `make_decision`). Fixtures `data/fixtures/dcgm_metrics_healthy.prom`, `dcgm_metrics_degraded.prom`, `sample_customer_workload_trace.csv`.
- Declared deps (`aurelius/requirements.txt`, `pyproject.toml`): fastapi, uvicorn, pydantic, supabase, pulp, numpy, pandas, scikit-learn, lightgbm, jinja2, matplotlib, sqlalchemy, psycopg2-binary (optional), PyYAML (used), requests (used, transitively present), boto3 (used by aws_batch, not declared). **NOT declared:** `kubernetes`, `prometheus-api-client`, `opentelemetry-*`, `nvidia-ml-py`. New connectors must add these as **optional** extras.
- CI (`.github/workflows/ci.yml`): `lint` (ruff, blocking), `typecheck` (mypy, non-blocking), `test` (pytest, excludes live), `benchmark-smoke` (`--quick`), `docker-build`.

### 2.10 What must remain untouched

These are working, validated, and load-bearing. Changes here require explicit justification and regression tests proving identical behavior when constraint-aware mode is OFF:

- The scalarized objective math in `objective.py` (energy/carbon/risk accounting) — backtested savings depend on it.
- Leakage-safe lookups (`_lookup_last_known`, `get_wait_hours`, `get_health_penalty`, shadow `RealizedSavingsCalculator`).
- The energy/carbon ISO connectors and `region_registry.py` confidence discipline.
- The execution trust boundary (dry-run default, signed policy, kill switch, quantile gate).
- Existing baselines and the benchmark regression contract.
- The forecasting stack (`forecasting/`, `ml/`) and its leakage audits.

---

## 3. Existing SLA Integration Audit

**Question by question (verified against `aurelius/sla/*`, `aurelius/optimization/scheduler.py`, `aurelius/cli.py`, `aurelius/backtesting/engine.py`):**

| Aspect | Status | Evidence |
|---|---|---|
| **Loaded from config?** | **Partially.** `SLALoader.load_file/load_dir/load_text` parse JSON/YAML, validate (`SLAValidationError`), apply tier defaults. But this loader is invoked **only** by `cmd_sla_report` (`aurelius/cli.py`). No other path loads SLA config. | `aurelius/sla/loader.py`; `cmd_sla_report` is the sole caller. |
| **Passed into real optimizer paths?** | **NO.** `JobScheduler.__init__` accepts `sla_registry`, `region_contexts`, `current_states`, `sla_block_on_unknown`, but `BacktestEngine` constructs `JobScheduler(self.config)` with no SLA args, and `SimulationReplay`/`ScenarioComparator` likewise. `sla_registry` defaults to `None` → `sla_enabled` is False → all SLA hooks are inert in production paths. | `aurelius/backtesting/engine.py` (`JobScheduler(self.config)`); `scheduler.py` `sla_enabled` property. |
| **Only used in tests/demo?** | **Effectively yes.** The SLA engine is exercised by `tests/test_sla_engine.py`, `tests/test_sla_optimization.py`, and the standalone `cmd_sla_report` (a before/after evaluator over hand-supplied candidate actions). It does not influence `simulate`/`backtest`/`shadow` decisions. | `cmd_sla_report` uses `SLAAwareActionSelector` on a scenario JSON; no feedback into the scheduler. |
| **Enforcing hard constraints correctly?** | **The engine does; the system does not (yet).** `evaluate_action_against_sla` correctly returns `allowed=False` on hard violations (region/residency/latency/queue/availability/capacity/migration-governance) and `_find_best_slot` excludes blocked candidates **when SLA is active**. But because SLA is never active in real paths, no hard constraint is currently enforced end-to-end. Hard SLA is also distinct from the older per-`Job` `allowed_regions`/`forbidden_regions` (enforced in `Job.__post_init__`) and `sla_penalty_per_hour` (a soft cost in the objective), which DO take effect. | `scheduler.py:_find_best_slot` (lines ~373–384); `evaluator.py`. |
| **Affecting rankings or only logging?** | **In the engine: affects rankings.** When active, `_sla_adjusted_score` folds `risk_score + soft_penalty_score` into the candidate score (scale-free, as a fraction of objective magnitude), and `_log_sla_correction` emits explainable logs. **In production today: neither** — it is dormant. | `scheduler.py:_sla_adjusted_score`, `_log_sla_correction`. |

**Conclusion.** The SLA layer is well-designed and test-covered but **dormant in every real decision path**. It is the highest-leverage, lowest-risk wiring target: connecting `SLARegistry` + `RegionContext` + `WorkloadState` into `BacktestEngine`/`JobScheduler` activates hard-constraint blocking and soft/risk-aware ranking that the constraint-aware engine (§7) depends on. This wiring is Phase 1 work (normalized state) culminating in Phase 9 (recommendation engine). Until then, the constraint classifier must not assume SLA is enforced — it must treat SLA enforcement as a capability it switches on.

**Important nuance for later phases:** there are effectively **two SLA representations** in the repo that must be reconciled, not duplicated:
1. `Job`-level fields (`sla_class`, `sla_penalty_per_hour`, `allowed_regions`, `forbidden_regions`, `max_delay_hours`) — already enforced/priced.
2. The `aurelius/sla/` policy engine (`HardSLA`/`SoftSLA`/tiers) — richer, dormant.
The constraint-aware design should make the `aurelius/sla/` engine the **single source of truth** and derive/migrate the `Job`-level fields from it, rather than maintaining both independently.

---

## 4. Target Architecture

### 4.1 Text diagram

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                  TELEMETRY CONNECTORS                     │
                          │  (each has a REAL impl + a FAKE sandbox impl, same iface) │
                          │                                                           │
   energy/carbon ───────► │  PrometheusConnector   DCGMConnector    vLLMConnector     │
   (existing, real)       │  TritonConnector       RayServeConnector OTelConnector    │
   GPU/inference/k8s ───► │  KubernetesConnector   TopologyCollector EnergyConnector  │
   (new)                  │  WeatherConnector(existing) ...                           │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │ raw provider payloads
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │              NORMALIZATION LAYER                          │
                          │  connector payloads → canonical ClusterState (§5)         │
                          │  units normalized, provenance + confidence + staleness    │
                          │  attached, missing fields = None (never fabricated)       │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │ ClusterState snapshot (immutable, timestamped)
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │          STATE STORE / SNAPSHOTS                          │
                          │  append-only, leakage-safe "last known ≤ T" lookups,      │
                          │  replayable, hashable (for benchmark metadata)            │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │            CONSTRAINT CLASSIFIER (§7)                     │
                          │  scores each constraint family; emits ranked              │
                          │  ConstraintAssessment with binding constraint +           │
                          │  confidence + missing-data flags                          │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │ binding constraint + confidence
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │          COST / RISK / MIGRATION MODEL                    │
                          │  per-constraint cost terms, migration penalty modeling,   │
                          │  reuses existing ObjectiveFunction + SLA RiskBreakdown     │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │             SLA-AWARE OPTIMIZER                           │
                          │  existing JobScheduler + sla_registry (wired), strategy   │
                          │  selected by binding constraint; HARD SLA blocks,         │
                          │  SOFT/risk penalties rank, fail-safe to KEEP              │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │ Recommendation[] (ranked, explained)
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │          RECOMMENDATION / REPORTING LAYER                 │
                          │  dry-run by default; CLI reports, JSON, HTML; audit log;  │
                          │  every rec carries constraint, confidence, expected       │
                          │  effect, SLA status, and "why"                            │
                          └───────────────────────────┬─────────────────────────────┘
                                                       │ (opt-in only)
                                                       ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │      OPTIONAL FUTURE EXECUTION ADAPTERS (gated)           │
                          │  existing Executor ABC: dry_run default, signed policy,   │
                          │  kill switch, quantile gate, read-only K8s by default     │
                          └─────────────────────────────────────────────────────────┘
```

### 4.2 Design principles

1. **Same interface, sandbox or real.** Every connector implements one Protocol with a real implementation and a fake (simulator-backed) implementation. Aurelius core never branches on "am I in sim or prod" — only the connector wiring differs (§9). This mirrors the existing `TelemetryProvider` Protocol and `StaticTelemetryProvider` pattern.
2. **Normalize early, decide late.** Connectors produce raw payloads; the normalization layer produces a single canonical `ClusterState`. The classifier, cost model, and optimizer only ever see `ClusterState` — never raw Prometheus JSON or DCGM text.
3. **Missing data is first-class.** Every model field that can be absent is `Optional` and defaults to `None`. The classifier and SLA engine already treat unknowns explicitly (`block_on_unknown`, `unknown_metrics`). Aurelius never fabricates a value to fill a gap.
4. **Detect, then optimize.** The classifier runs first and selects the optimization strategy. Energy optimization is the strategy chosen when energy is binding; it is not the default for all conditions.
5. **Recommend by default.** The output is a ranked list of explained `Recommendation`s. Execution is the existing opt-in, gated path. The classifier/optimizer never call execution adapters directly.
6. **Reuse, don't reinvent.** The cost/risk/migration model reuses `ObjectiveFunction` and the SLA `RiskBreakdown`; the optimizer is the existing `JobScheduler` with `sla_registry` wired; the state store reuses leakage-safe lookup patterns.

---

## 5. Canonical Data Model Design

All models are proposed as **frozen Python dataclasses** (matching the repo's no-pydantic-in-core convention; pydantic is used only at the API boundary). New models live in `aurelius/state/models.py` (new module). Every field that can be absent is `Optional[...] = None` — **a `None` means "not observed", never "zero"**. Each snapshot carries provenance + freshness so the classifier can reason about staleness and confidence.

**Shared value objects (defined once, reused):**

```python
@dataclass(frozen=True)
class Provenance:
    source: str               # connector name, e.g. "dcgm-exporter", "prometheus", "simulator"
    fetched_at: datetime      # UTC; when this value was collected
    confidence: str           # "high" | "medium" | "low" (reuse ingestion.region_registry.Confidence)
    is_sandbox: bool = False  # True if from simulator/sandbox; REJECTED from savings/SLA claims
    sample_age_s: Optional[float] = None  # observed_at→now staleness; None if unknown
```

Validation rule applied to every model: `timestamp` must be UTC-aware; any percentage field validated to `0–100`; any rate/byte/duration field validated `>= 0`; unknown → `None` (never coerced to 0). Models never raise on missing optional data; the normalization layer logs and flags.

> **Naming note:** `QueueState` and `GPUMetrics` already exist in `aurelius/models.py` with specific shapes. To avoid collision and churn, the new canonical models live under `aurelius/state/` and the normalization layer adapts the existing `QueueState`/`GPUMetrics`/`GPUHealthScore` into them. Do **not** rename the existing models in Phase 1.

### 5.1 `ClusterState` — root snapshot
The single object the classifier consumes.

| Field | Type | Validation | Source connector |
|---|---|---|---|
| `timestamp` | `datetime` | UTC-aware, required | — (assigned at normalization) |
| `regions` | `dict[str, RegionState]` | non-empty in prod; may be empty in degraded mode | aggregation |
| `provenance` | `Provenance` | required | normalization layer |
| `snapshot_id` | `str` | uuid4 | normalization layer |
| `config_hash` | `Optional[str]` | sha256 of active config | controller |
| `is_partial` | `bool` | True if any connector failed | normalization layer |
| `missing_sources` | `list[str]` | connector names that failed/were stale | normalization layer |

### 5.2 `RegionState`
| Field | Type | Validation | Source |
|---|---|---|---|
| `region` | `str` | canonical id (`region_registry`) | — |
| `nodes` | `dict[str, NodeState]` | — | K8s + DCGM |
| `services` | `dict[str, InferenceServiceState]` | — | vLLM/Triton/Ray |
| `queues` | `dict[str, QueueState]` | by cluster/pool | queue/K8s/Ray |
| `energy` | `Optional[EnergyState]` | — | energy connectors |
| `thermal` | `Optional[ThermalState]` | — | weather/DCGM-derived |
| `topology` | `Optional[TopologyState]` | — | topology collector |
| `spare_capacity_pct` | `Optional[float]` | 0–100 | derived (K8s allocatable − requested) |
| `provenance` | `Provenance` | required | — |

### 5.3 `NodeState`
| Field | Type | Validation | Source |
|---|---|---|---|
| `node_id` | `str` | host/node name | K8s `node.metadata.name` / DCGM `Hostname` |
| `region` | `str` | — | mapping |
| `gpus` | `dict[str, GPUState]` | keyed by gpu_uuid | DCGM/NVML |
| `gpu_capacity` | `Optional[int]` | `>=0` | K8s `node.status.capacity["nvidia.com/gpu"]` (string→int) |
| `gpu_allocatable` | `Optional[int]` | `>=0` | K8s `node.status.allocatable["nvidia.com/gpu"]` |
| `gpu_allocated` | `Optional[int]` | `>=0` | derived from pod limits |
| `labels` | `dict[str,str]` | — | K8s `node.metadata.labels` (incl. `nvidia.com/gpu.product`) |
| `taints` | `list[dict]` | — | K8s `node.spec.taints` |
| `schedulable` | `Optional[bool]` | — | K8s `!node.spec.unschedulable` |
| `provenance` | `Provenance` | required | — |

### 5.4 `GPUState`
Adapts the existing `GPUMetrics` + `GPUHealthScore`. **See §6.4 for which fields are reliably populated by default dcgm-exporter.**

| Field | Type | Validation | Source (DCGM metric) |
|---|---|---|---|
| `gpu_uuid` | `str` | required | DCGM label `UUID` |
| `node_id`, `region` | `str` | — | mapping |
| `gpu_index` | `Optional[int]` | `>=0` | label `gpu` |
| `gpu_type` | `Optional[str]` | — | label `modelName` / GFD label |
| `util_pct` | `Optional[float]` | 0–100 | `DCGM_FI_DEV_GPU_UTIL` |
| `mem_used_mb` | `Optional[float]` | `>=0` | `DCGM_FI_DEV_FB_USED` |
| `mem_free_mb` | `Optional[float]` | `>=0` | `DCGM_FI_DEV_FB_FREE` |
| `mem_total_mb` | `Optional[float]` | `>=0` | `DCGM_FI_DEV_FB_TOTAL` if present, else `used+free+reserved` |
| `power_w` | `Optional[float]` | `>=0` | `DCGM_FI_DEV_POWER_USAGE` |
| `temp_c` | `Optional[float]` | — | `DCGM_FI_DEV_GPU_TEMP` |
| `mem_temp_c` | `Optional[float]` | — | `DCGM_FI_DEV_MEMORY_TEMP` |
| `sm_clock_mhz` | `Optional[float]` | `>=0` | `DCGM_FI_DEV_SM_CLOCK` |
| `clocks_event_reasons` | `Optional[int]` | bitmask | `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (alias `..._CLOCK_THROTTLE_REASONS`) |
| `ecc_sbe_total` | `Optional[int]` | `>=0` | `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` **(disabled by default)** |
| `ecc_dbe_total` | `Optional[int]` | `>=0` | `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` **(disabled by default)** |
| `xid_last` | `Optional[int]` | — | `DCGM_FI_DEV_XID_ERRORS` (last value, enabled) |
| `xid_count_window` | `Optional[int]` | `>=0` | `DCGM_EXP_XID_ERRORS_COUNT` **(disabled by default)** |
| `power_violation_ns` | `Optional[int]` | `>=0` | `DCGM_FI_DEV_POWER_VIOLATION` **(disabled by default; nanoseconds)** |
| `thermal_violation_ns` | `Optional[int]` | `>=0` | `DCGM_FI_DEV_THERMAL_VIOLATION` **(disabled by default; nanoseconds)** |
| `pcie_tx_bytes_per_s` | `Optional[float]` | `>=0` | `DCGM_FI_PROF_PCIE_TX_BYTES` |
| `nvlink_tx_bytes_per_s` | `Optional[float]` | `>=0` | `DCGM_FI_PROF_NVLINK_TX_BYTES` (disabled by default) |
| `tensor_active_ratio` | `Optional[float]` | 0–1 | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` |
| `mig_instance_id` | `Optional[str]` | — | label `GPU_I_ID` |
| `health_penalty` | `Optional[float]` | 0–1 | derived (`score_gpu_health`) |
| `is_schedulable` | `Optional[bool]` | — | derived |
| `provenance` | `Provenance` | required | — |

### 5.5 `InferenceServiceState`
The inference-server view (vLLM/Triton/Ray). This is new and central to queue/latency/memory-bound classification.

| Field | Type | Validation | Source |
|---|---|---|---|
| `service_id` | `str` | required | label `model_name`/`model`/`deployment` |
| `region`, `node_id` | `Optional[str]` | — | mapping |
| `engine` | `str` | `"vllm"\|"triton"\|"ray_serve"\|"unknown"` | connector |
| `requests_running` | `Optional[float]` | `>=0` | vLLM `num_requests_running` / Ray `ray_serve_replica_processing_queries` |
| `requests_waiting` | `Optional[float]` | `>=0` | vLLM `num_requests_waiting` / Triton `nv_inference_pending_request_count` / Ray `ray_serve_deployment_queued_queries` |
| `p95_latency_ms` | `Optional[float]` | `>=0` | histogram quantile of vLLM `e2e_request_latency_seconds` / Triton `nv_inference_request_duration_us` / Ray `ray_serve_deployment_processing_latency_ms` |
| `p99_latency_ms` | `Optional[float]` | `>=0` | same histograms (0.99) |
| `ttft_ms` | `Optional[float]` | `>=0` | vLLM `time_to_first_token_seconds` / Triton `nv_inference_first_response_histogram_ms` (opt-in) |
| `queue_time_ms` | `Optional[float]` | `>=0` | vLLM `request_queue_time_seconds` / Triton `nv_inference_queue_duration_us` |
| `kv_cache_usage` | `Optional[float]` | 0–1 | vLLM `kv_cache_usage_perc` (V1) / `gpu_cache_usage_perc` (V0) |
| `prefix_cache_hit_rate` | `Optional[float]` | 0–1 | vLLM `gpu_prefix_cache_hit_rate` (V0) or `prefix_cache_hits/prefix_cache_queries` (V1) |
| `preemptions_total` | `Optional[float]` | `>=0` | vLLM `num_preemptions_total` |
| `error_rate_pct` | `Optional[float]` | 0–100 | derived from request_success/failure counters |
| `replicas` | `Optional[int]` | `>=0` | Ray autoscaling / K8s |
| `tokens_per_s` | `Optional[float]` | `>=0` | derived from token counters |
| `provenance` | `Provenance` | required | — |

### 5.6 `WorkloadState`
**Reuse the existing `aurelius/sla/telemetry.py:WorkloadState`** (latency, queue, util, error/timeout/availability, capacity_buffer, energy/carbon, cost_per_token, tokens_per_joule). Extend it (additively, default `None`) with: `service_id`, `gpu_uuids: list[str]`, `kv_cache_usage`, `comm_bytes_per_s`, `migration_count_last_hour` (already present). Do not fork it — the SLA engine already consumes it.

### 5.7 `QueueState`
**Reuse the existing `aurelius/models.py:QueueState`** (timestamp, region, cluster_id, gpu_type, available_gpus, queue_depth_jobs, est_wait_hours). The normalization layer populates it from K8s pending pods, Ray queued queries, vLLM waiting requests, or a queue CSV. Add `provenance` via a thin wrapper in `aurelius/state/`.

### 5.8 `TopologyState`
New. Captures intra-node/inter-node GPU interconnect for topology-bound classification and placement.

| Field | Type | Validation | Source |
|---|---|---|---|
| `node_id` | `str` | required | — |
| `gpu_uuids` | `list[str]` | — | NVML/DCGM |
| `pair_levels` | `dict[tuple[str,str], str]` | values ∈ {`NV#`,`PIX`,`PXB`,`PHB`,`NODE`,`SYS`} | `nvidia-smi topo -m` parse OR NVML `nvmlDeviceGetTopologyCommonAncestor` |
| `nvlink_present` | `Optional[bool]` | — | NVML `nvmlDeviceGetNvLinkState` |
| `numa_affinity` | `dict[str,int]` | gpu_uuid→NUMA node | topo matrix affinity columns |
| `interconnect_class` | `Optional[str]` | `"nvlink_full"\|"nvlink_partial"\|"pcie"\|"cross_numa"\|"unknown"` | derived |
| `provenance` | `Provenance` | required (often `confidence="medium"`, see §6) | — |

> Topology is the **lowest-confidence** signal: `nvidia-smi topo -m` has no machine-readable form and requires node-local access; NVML requires the GPU host. In sandbox/cloud-only deployments topology may be entirely `None`. The classifier must degrade gracefully (§7).

### 5.9 `EnergyState`
| Field | Type | Validation | Source |
|---|---|---|---|
| `region` | `str` | — | — |
| `price_per_mwh` | `Optional[float]` | `>=0` | ISO connectors (existing) |
| `price_percentile` | `Optional[float]` | 0–100 | derived vs region history |
| `carbon_gco2_per_kwh` | `Optional[float]` | `>=0` | EM/WattTime (existing) |
| `pue` | `Optional[float]` | `>=1.0` | config/weather-derived |
| `power_cap_kw` | `Optional[float]` | `>=0` | config (`region_power_caps`) |
| `power_draw_kw` | `Optional[float]` | `>=0` | sum of GPU `power_w` |
| `provenance` | `Provenance` | required | — |

### 5.10 `ThermalState`
| Field | Type | Validation | Source |
|---|---|---|---|
| `region` / `node_id` | `str` | — | — |
| `max_gpu_temp_c` | `Optional[float]` | — | max DCGM `GPU_TEMP` |
| `mean_gpu_temp_c` | `Optional[float]` | — | mean DCGM `GPU_TEMP` |
| `throttling_gpu_count` | `Optional[int]` | `>=0` | count where `clocks_event_reasons` has HW/SW thermal bit set |
| `ambient_temp_c` | `Optional[float]` | — | weather connector (existing `fetch_weather_data`) |
| `cooling_headroom_pct` | `Optional[float]` | 0–100 | derived (weather/PUE proxy) |
| `provenance` | `Provenance` | required | — |

> **Honesty note:** Aurelius has **no direct facility/DCIM telemetry** (CRAC units, PDU, chilled water). `cooling_headroom_pct` is a proxy derived from ambient weather + GPU temps + PUE. It must be labeled low-confidence and never presented as measured cooling capacity (see §6.13).

### 5.11 `MigrationHistory`
Tracks recent migrations for churn detection and migration-governance SLA enforcement.

| Field | Type | Validation | Source |
|---|---|---|---|
| `workload_id` | `str` | — | — |
| `events` | `list[MigrationEvent]` | each: `(from_region, to_region, ts, reason, cost_hours)` | recommendation/exec log |
| `count_last_hour` | `int` | `>=0` | derived |
| `count_last_24h` | `int` | `>=0` | derived |
| `provenance` | `Provenance` | required | — |

### 5.12 `ConstraintAssessment`
Output of the classifier (§7). The pivotal new object.

| Field | Type | Validation | Source |
|---|---|---|---|
| `timestamp`, `region` (or `scope`) | `datetime`/`str` | — | classifier |
| `scores` | `dict[ConstraintType, float]` | each 0–1 | classifier |
| `binding_constraint` | `Optional[ConstraintType]` | None if no signal/low confidence | classifier |
| `confidence` | `float` | 0–1 | classifier |
| `missing_signals` | `list[str]` | telemetry needed but absent | classifier |
| `safe_actions` | `list[ActionType]` | from §8 matrix | classifier |
| `disallowed_actions` | `list[ActionType]` | from §8 matrix | classifier |
| `rationale` | `str` | human explanation | classifier |
| `provenance` | `Provenance` | required | — |

`ConstraintType` enum: `ENERGY`, `THERMAL`, `QUEUE`, `LATENCY`, `COMMUNICATION`, `MEMORY`, `TOPOLOGY`, `UTILIZATION`, `NONE`.

### 5.13 `Recommendation`
The product output. Recommendation-only by default.

| Field | Type | Validation | Source |
|---|---|---|---|
| `recommendation_id` | `str` | uuid4 | engine |
| `workload_id` / `scope` | `str` | — | engine |
| `action` | `OptimizationAction` | reuse `aurelius/sla/actions.py` | engine |
| `binding_constraint` | `ConstraintType` | — | classifier |
| `expected_effect` | `dict` | e.g. `{metric: delta}`; signed | cost model |
| `confidence` | `float` | 0–1 | engine |
| `sla_status` | `str` | `"satisfied"\|"corrected"\|"blocked"\|"unknown"` | SLA evaluator |
| `sla_evaluation` | `Optional[dict]` | `SLAEvaluation.to_dict()` | SLA evaluator |
| `migration_penalty` | `Optional[float]` | `>=0` | migration model |
| `net_benefit` | `Optional[float]` | signed; effect − penalties | cost model |
| `rationale` | `str` | why this, why now | engine |
| `is_noop` | `bool` | True = KEEP (fail-safe) | engine |
| `provenance` | `Provenance` | required | — |

---

## 6. Connector Plan

**Universal rules for every connector:**
- One Protocol per data domain in `aurelius/connectors/base.py`, with a **real** implementation and a **fake** (simulator-backed) implementation that returns byte-identical payload shapes (§9). Aurelius core depends only on the Protocol.
- Auth/config from env vars (extend `.env.example`); secrets never logged (§13).
- **Read-only.** No connector mutates the customer cluster.
- **Failure mode = safe degradation.** On timeout/error/missing config, return an empty/partial payload, log a structured warning, set `is_partial=True` and add to `missing_sources`. Never raise into the optimizer. This mirrors the existing `PriceProvider` contract ("returns empty df on failure — never raises on missing data") and `DCGMProvider.from_prometheus_live` (returns empty provider, logs warning).
- New third-party libs are **optional extras** (`pip install aurelius[k8s]`, `[otel]`, `[nvml]`), lazily imported, with graceful ImportError handling (the repo already does this for PuLP and boto3).

### 6.1 Prometheus HTTP API (foundational — most signals arrive via Prometheus)
- **Source/API:** `GET`/`POST {PROMETHEUS_URL}/api/v1/query` (instant), `/api/v1/query_range` (range), `/api/v1/series`, `/api/v1/labels`, `/api/v1/targets`. Docs: https://prometheus.io/docs/prometheus/latest/querying/api/
- **Exact response shape to parse:** `{"status":"success","data":{"resultType":"vector|matrix|scalar|string","result":[...]}}`. Vector result item: `{"metric":{<labels>}, "value":[<unix_ts_float>, "<value_string>"]}`. **Sample values are JSON strings — cast to float.** Range → `resultType:"matrix"`, items use `"values":[[ts,"v"],...]`. Recent Prometheus may add a `histograms` field alongside `values` — parse defensively. `[AMBIGUOUS: native-histogram shape is version-dependent.]`
- **Auth:** Prometheus server natively supports **basic auth + TLS only** (`web.config.file`); bearer/OAuth is typically terminated by a fronting proxy or managed service. Support: basic auth (`PROMETHEUS_USERNAME`/`PASSWORD`) and arbitrary `Authorization` header (`PROMETHEUS_BEARER_TOKEN`) — **the existing `DCGMProvider` already implements exactly this.**
- **API key needed?** No (customer-operated). **Customer cluster access?** Network reachability to their Prometheus only (read).
- **Sandbox equivalent:** Fake Prometheus HTTP server (or in-process responder) returning recorded/synthetic `/api/v1/query` JSON for the simulator's metric set (§9).
- **Client:** No official query client. Use `requests` + the HTTP API directly (matches existing code). `prometheus-api-client` (community) is optional. **Do NOT use `prometheus-client` — that is instrumentation/exposition only, not query.**
- **Metrics/fields:** whatever the customer scrapes — DCGM, vLLM, Triton, Ray, node-exporter. Aurelius issues PromQL per signal.
- **Failure mode:** unreachable/4xx/5xx → empty result, flag stale, classifier treats affected signals as `None`.

### 6.2 Raw `/metrics` scraping (when Prometheus is absent)
- **Source:** direct HTTP GET of an exporter's `/metrics` (Prometheus text exposition format). Used today by `DCGMProvider.from_prometheus_live` against `DCGM_EXPORTER_URL` (`:9400/metrics`), and applicable to vLLM (`:8000/metrics`), Triton (`:8002/metrics`), Ray agent (`--metrics-export-port`).
- **Parser:** the repo already has `parse_prometheus_text` for DCGM; generalize to a shared text-exposition parser handling `# HELP`/`# TYPE`, labels, histograms (`_bucket`/`_sum`/`_count`), and counters with `_total`.
- **Auth/key/access:** none/customer-operated/network read. **Sandbox:** fake `/metrics` text endpoints (§9).
- **Failure mode:** as §6.1.

### 6.3 DCGM / dcgm-exporter (GPU health, thermal, memory, power, comm)
- **Source/API:** dcgm-exporter on **`:9400/metrics`** (Prometheus text), or via Prometheus (§6.1). Docs: https://github.com/NVIDIA/dcgm-exporter and https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html
- **Auth/key/access:** none / customer-operated / network read (no public API key; DCGM runs on-prem with the GPUs).
- **Sandbox equivalent:** fake dcgm-exporter `/metrics` text — the repo already ships `data/fixtures/dcgm_metrics_{healthy,degraded}.prom` and `DCGMProvider.generate_fixture(seed=42)`.
- **Metrics enabled BY DEFAULT** (verified against `etc/default-counters.csv`): `DCGM_FI_DEV_SM_CLOCK`, `MEM_CLOCK`, `MEMORY_TEMP`, `GPU_TEMP`, `POWER_USAGE`, `TOTAL_ENERGY_CONSUMPTION` (mJ), `PCIE_REPLAY_COUNTER`, `GPU_UTIL`, `MEM_COPY_UTIL`, `ENC_UTIL`, `DEC_UTIL`, `XID_ERRORS` (last value), `FB_FREE`, `FB_USED`, `FB_RESERVED`, `NVLINK_BANDWIDTH_TOTAL`, remapped-rows, `DCGM_FI_DRIVER_VERSION` (label), `DCGM_FI_PROF_{GR_ENGINE_ACTIVE, PIPE_TENSOR_ACTIVE, DRAM_ACTIVE, PCIE_TX_BYTES, PCIE_RX_BYTES}`.
- **CRITICAL — metrics DISABLED by default** (must instruct customers to enable, or treat as `None`): all **ECC** (`ECC_SBE_VOL_TOTAL`/`ECC_DBE_VOL_TOTAL`/agg), throttle **durations** (`POWER_VIOLATION`/`THERMAL_VIOLATION`, in **nanoseconds**), retired pages, **NVLink error counters**, `DCGM_FI_PROF_{SM_ACTIVE,SM_OCCUPANCY,PIPE_FP*}`, and the exporter-computed `DCGM_EXP_*` (`XID_ERRORS_COUNT`, `CLOCK_EVENTS_COUNT`, `GPU_HEALTH_STATUS`). **`DCGM_FI_DEV_FB_TOTAL` is NOT emitted by default** — derive total from `FB_USED+FB_FREE+FB_RESERVED`.
- **Throttle reason field renamed:** `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` → **`DCGM_FI_DEV_CLOCKS_EVENT_REASONS`** (field ID 112; old name kept as deprecated alias). Bitmask constants renamed `DCGM_CLOCKS_THROTTLE_REASON_*` → `DCGM_CLOCKS_EVENT_REASON_*`. Thermal bits: `SW_THERMAL=0x20`, `HW_THERMAL=0x40`, `HW_POWER_BRAKE=0x80`, `SW_POWER_CAP=0x04`, `HW_SLOWDOWN=0x08`. **Support both names.**
- **Labels (exact capitalization):** `gpu`, `UUID` (capitalized), `pci_bus_id`, `device`, `modelName`, `Hostname` (capitalized); under k8s also `container`/`namespace`/`pod`; under MIG also `GPU_I_ID`/`GPU_I_PROFILE`. Key identity on `gpu`+`UUID` (+`GPU_I_ID` for MIG).
- **⚠ Existing-code correctness issues to FIX in Phase 3** (`aurelius/ingestion/dcgm_provider.py`): (a) the constant `_DCGM_MEM_TOTAL = "DCGM_FI_DEV_FB_FREE"` is mislabeled and total derivation ignores `FB_RESERVED`; (b) it reads ECC/violation metrics that are **disabled by default** without flagging them as likely-`None`; (c) `power_throttle_us`/`thermal_throttle_us` field names imply microseconds but `POWER_VIOLATION`/`THERMAL_VIOLATION` are **nanoseconds**; (d) uses the deprecated `CLOCK_THROTTLE_REASONS` name only. These must be corrected and unit-tested against the real default metric set.
- **Failure mode:** missing metrics → fields `None`; classifier downgrades thermal/memory confidence.

### 6.4 vLLM `/metrics` (queue depth, latency, KV-cache, preemptions)
- **Source/API:** Prometheus metrics on the OpenAI-compatible server, **same port as the API (`:8000` default), path `/metrics`**, prefix `vllm:`, primary label `model_name`. Docs: https://docs.vllm.ai/en/stable/usage/metrics/ and V1 design https://docs.vllm.ai/en/latest/design/v1/metrics.html
- **CRITICAL V0→V1 naming split — the connector MUST branch:** KV cache `vllm:gpu_cache_usage_perc` (V0) → **`vllm:kv_cache_usage_perc`** (V1); prefix-cache `vllm:gpu_prefix_cache_hit_rate` (V0 gauge) → compute from **`vllm:prefix_cache_hits`/`vllm:prefix_cache_queries`** (V1 counters); per-output-token `time_per_output_token_seconds` (V0) → `inter_token_latency_seconds` (V1). `cpu_cache_usage_perc`/`num_requests_swapped` removed in V1. Detect engine version and branch; treat absent variant as `None`.
- **Key metrics:** `vllm:num_requests_running` (gauge), `vllm:num_requests_waiting` (queue depth), `vllm:kv_cache_usage_perc` (1.0=100%), `vllm:time_to_first_token_seconds` (hist), `vllm:e2e_request_latency_seconds` (hist), `vllm:request_queue_time_seconds` (hist), `vllm:num_preemptions_total`, `vllm:{prompt,generation}_tokens_total`. **Counters expose with `_total` suffix** even when docs list the base name.
- **Auth/key/access:** none / customer-operated / network read. **Sandbox:** fake vLLM `/metrics` text (§9).
- **Failure mode:** as §6.2.

### 6.5 NVIDIA Triton `/metrics`
- **Source/API:** **`:8002/metrics`** default, prefix `nv_`. Docs: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html
- **Key metrics:** `nv_inference_request_success`/`_failure`/`_count`, `nv_inference_pending_request_count` (gauge, queue depth, per-model), latency **counters in microseconds** `nv_inference_{request,queue,compute_input,compute_infer,compute_output}_duration_us`; optional **summaries** `nv_inference_*_summary_us` (off by default, `--metrics-config summary_latencies=true`); `nv_inference_first_response_histogram_ms` (TTFT, opt-in); GPU `nv_gpu_utilization` (0–1), `nv_gpu_memory_{total,used}_bytes`, `nv_gpu_power_{usage,limit}`, `nv_energy_consumption`. Labels: `model`,`version`; GPU metrics `gpu_uuid`.
- **Auth/key/access:** none / customer-operated / network read. **Sandbox:** fake Triton `/metrics`.
- **Gotchas:** summaries + first-response histogram are opt-in; GPU metrics require DCGM-capable GPU and `--allow-gpu-metrics`; latency is µs. Treat opt-in metrics as `None` unless present.
- **Failure mode:** as §6.2.

### 6.6 Ray Serve metrics
- **Source/API:** per-node Ray metrics agent in Prometheus format; enable with `ray start --metrics-export-port=<port>` (**docs example 8080 — not a guaranteed default; set explicitly**). Service discovery file `/tmp/ray/prom_metrics_service_discovery.json`; dashboard SD at `:8265/api/prometheus/sd`. Docs: https://docs.ray.io/en/latest/serve/monitoring.html and https://docs.ray.io/en/latest/cluster/metrics.html
- **Key metrics (prefix `ray_serve_`, counters with `_total`):** `ray_serve_deployment_queued_queries` (queue depth), `ray_serve_num_ongoing_requests_at_replicas`, `ray_serve_replica_processing_queries`, `ray_serve_deployment_processing_latency_ms` (hist), `ray_serve_num_router_requests_total`, `ray_serve_num_http_requests_total`, autoscaling `ray_serve_autoscaling_{target,desired}_replicas`. Core/cluster: `ray_node_gpus_utilization` (labels `GpuDeviceName`,`GpuIndex`), `ray_node_gram_{used,available}`, `ray_resources{Name="GPU"}`, `ray_cluster_{active,pending,failed}_nodes`. **`ray_serve_replica_pending_queries` and `ray_node_gpus` do NOT exist in current docs** — use the names above.
- **Auth/key/access:** none / customer-operated / network read. **Sandbox:** fake Ray metrics endpoint.
- **Failure mode:** as §6.2.

### 6.7 OpenTelemetry Collector / OTLP
- **Source/API:** OTLP **gRPC `:4317`**, **HTTP `:4318`** (POST `/v1/metrics`, `/v1/traces`, `/v1/logs`; `application/x-protobuf` or `application/json`). Data model: Resource → ScopeMetrics → Metric (Gauge/Sum/Histogram/ExponentialHistogram/Summary). Docs: https://opentelemetry.io/docs/specs/otlp/ and https://opentelemetry.io/docs/specs/otel/metrics/data-model/
- **CRITICAL architecture fact:** **OTLP is push-only; there is no standard way to "scrape" OTLP, and the Python SDK ships exporters only (no receiver/server).** Therefore Aurelius does **not** implement an OTLP scraper. The supported pattern is **OTLP → Collector → Prometheus (remote-write or prometheus exporter) → Aurelius reads via §6.1**. Document this clearly; do not invent an OTLP pull client.
- **Semantic conventions** (for resource attribution if/when relevant): host `host.name`/`host.id`; k8s `k8s.node.name`/`k8s.pod.name`/`k8s.namespace.name`; GPU is modeled under **`hw.*`** (`hw.gpu.utilization`, `hw.gpu.memory.usage`, `hw.status` with `hw.type="gpu"`) — **Development status, expect churn, and these differ from DCGM names.** `[UNVERIFIED-STABILITY: hw.gpu.* conventions are experimental.]`
- **Auth/key/access:** collector-dependent; none required by Aurelius. **Sandbox:** fake Prometheus endpoint fed by a simulated collector export.
- **Python:** `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-{grpc,http}` (export only). **Failure mode:** rely on §6.1 once landed in Prometheus.

### 6.8 Kubernetes API (nodes, pods, GPU capacity, pending queue)
- **Source/API:** official PyPI **`kubernetes`** client. `config.load_incluster_config()` in-cluster else `config.load_kube_config()`. `CoreV1Api().list_node()` → `V1NodeList`, `list_pod_for_all_namespaces()` → `V1PodList`, `read_node(name)`. Docs: https://github.com/kubernetes-client/python (`kubernetes/docs/CoreV1Api.md`).
- **GPU capacity:** `node.status.capacity["nvidia.com/gpu"]` / `node.status.allocatable["nvidia.com/gpu"]` (**string quantities → int**). Labels `node.metadata.labels` (GFD: `nvidia.com/gpu.product`, `.memory`, `.count`). Taints `node.spec.taints`. Pod GPU requests appear in **`limits` only** (`pod.spec.containers[*].resources.limits["nvidia.com/gpu"]`; `resources` may be `None`).
- **Gotcha:** the pagination kwarg is **`_continue=`** (Python reserved word), not `continue`.
- **Metrics-server** (`metrics.k8s.io` via `CustomObjectsApi.list_cluster_custom_object(group="metrics.k8s.io", version="v1beta1", plural="nodes"|"pods")`) exposes **CPU/memory only — NOT GPU**. GPU utilization comes from DCGM/Prometheus, not metrics-server.
- **Auth/key/access:** **in-cluster ServiceAccount token** (`/var/run/secrets/kubernetes.io/serviceaccount/{token,ca.crt,namespace}`) or kubeconfig. **Customer cluster access required (read-only).** Minimal RBAC ClusterRole: `get,list,watch` on `nodes`, `nodes/status`, `pods` (+ `metrics.k8s.io` `nodes,pods` if used). Provide this YAML in docs.
- **API key needed?** No (token-based). **Sandbox:** fake K8s API payloads — synthetic `V1NodeList`/`V1PodList` dicts (§9).
- **Failure mode:** unreachable/forbidden → empty node/pod lists, flag stale; classifier loses spare-capacity and pending-queue signals → lower confidence.
- **Trust boundary:** **read-only by default.** Any write (cordon/drain/scale) is out of scope for the connector and belongs only to the gated execution layer.

### 6.9 `nvidia-smi topo -m` (intra-node topology)
- **Source/API:** shell `nvidia-smi topo -m`. **No JSON/CSV/XML output exists.** Legend (verified, https://docs.nvidia.com/deploy/nvidia-smi/): `X`=self, `NV#`=# bonded NVLinks (fastest), `PIX`=single PCIe switch, `PXB`=multiple PCIe switches, `PHB`=PCIe host bridge, `NODE`=PCIe + intra-NUMA bridges, `SYS`=cross-NUMA/SMP interconnect (slowest). Trailing columns: `CPU Affinity`, `NUMA Affinity`, optionally `GPU NUMA ID` (driver-version dependent — parse by header name, not offset).
- **Auth/key/access:** **node-local shell access required** (or NVML on the host). Not available from a cloud-only/control-plane-only deployment.
- **Recommended:** prefer **NVML** over text scraping — `nvmlDeviceGetTopologyCommonAncestor(d1,d2,&level)` returns `nvmlGpuTopologyLevel_t` (INTERNAL=0, SINGLE=10, MULTIPLE=20, HOSTBRIDGE=30, NODE=40, SYSTEM=50). Python binding: **`nvidia-ml-py`** (imports as `pynvml`) — NOT the third-party `pynvml`/`py3nvml` forks.
- **Sandbox equivalent:** fake `nvidia-smi topo -m` text output for synthetic node topologies (§9).
- **Failure mode:** no node access → `TopologyState=None`; topology-bound classification disabled, others unaffected.
- **Caveat `[AMBIGUOUS]`:** the NVTAGS doc lists different numeric weights and calls PXB "bridges" vs nvidia-smi's "switches" — use the **nvidia-smi legend** as canonical for parsing.

### 6.10 DCGM topology / NVML (optional future layer)
- **Source/API:** DCGM Python bindings ship inside the DCGM OS package (NOT PyPI) under a versioned path (e.g. `/usr/share/datacenter-gpu-manager-4/bindings/python3/`); modules `dcgm_fields`, `pydcgm`, helper `DcgmReader`. NVML topology via `nvidia-ml-py`. Docs: https://github.com/NVIDIA/DCGM
- **Auth/key/access:** node-local. **Sandbox:** synthetic topology (§9).
- **Status:** **future/optional.** MVP uses Prometheus+dcgm-exporter (§6.3) and, where node access exists, `nvidia-smi topo -m`/NVML (§6.9). Do not build the DCGM-binding path until a customer requires it.
- **Failure mode:** absent → rely on §6.3/§6.9.

### 6.11 ISO/RTO energy APIs (existing — keep)
- **Source/API:** CAISO OASIS (no key), PJM Data Miner (`PJM_API_KEY`), ENTSO-E (`ENTSOE_API_KEY`), ERCOT. Already implemented in `aurelius/ingestion/grid_apis/*` with `region_registry` confidence mapping. Carbon: ElectricityMaps (`ELECTRICITYMAPS_API_KEY`, sandbox flag), WattTime (`WATTTIME_USERNAME/PASSWORD`).
- **Auth/key/access:** API keys per provider (see `.env.example`); no customer cluster access. **Sandbox:** EM sandbox endpoint (already flags `is_sandbox=True` and rejects from claims) + CSV fixtures.
- **Failure mode:** existing — `empty_price_df()`/`empty_carbon_df()` on failure; `ProviderConfigError` only when a key is required and absent.
- **Action:** normalize these into `EnergyState`; otherwise **untouched**.

### 6.12 Weather / cooling proxies (existing — keep)
- **Source/API:** Iowa Environmental Mesonet ASOS METAR (no key), Open-Meteo. `scripts/fetch_weather_data.py`, `aurelius/forecasting/features.py` weather features.
- **Auth/key/access:** none / no cluster access. **Sandbox:** CSV fixtures.
- **Use:** populate `ThermalState.ambient_temp_c` and a low-confidence `cooling_headroom_pct` proxy. **Honest limitation:** weather is NOT cooling-system telemetry; it is a predictive proxy (per the team's own roadmap note "GPU temperature is NOT a substitute for weather", and the inverse holds — weather is not a substitute for DCIM).
- **Failure mode:** absent → `ThermalState.ambient_*=None`.

### 6.13 Optional DCIM / BMS (future)
- **Source/API:** facility DCIM/BMS (Redfish, SNMP, vendor APIs) for PDU power, CRAC/chilled-water, rack inlet temps. **No implementation exists and none is planned for MVP.**
- **Auth/key/access:** facility-specific; deep customer integration. **Sandbox:** synthetic rack/cooling model in the simulator (§9).
- **Status:** **explicitly future.** Until then, thermal/energy-capacity signals are proxies (GPU temps + weather + configured `power_cap_kw`), and the plan must not claim measured cooling/PDU headroom.
- **Failure mode:** absent (the default) → proxies only, flagged low-confidence.

---

## 7. Constraint Classifier Design

**Purpose:** given a `ClusterState` (or a per-region/per-service slice), score each constraint family in `[0,1]`, identify the **binding constraint** (highest score above a confidence floor), and emit a `ConstraintAssessment` (§5.12) that tells the optimizer which strategy to use and which actions are safe/unsafe.

**Module:** new `aurelius/constraints/classifier.py` with one scorer function per family and a `ConstraintClassifier.assess(state) -> ConstraintAssessment`.

**Cross-cutting rules (apply to every constraint):**
- **Scores are normalized 0–1**, monotone in "how binding". A score is computed only from signals that are present.
- **Confidence rule:** `confidence = (fraction of required signals present) × (1 − staleness_penalty) × min(provenance confidence weight)`. `high=1.0, medium=0.7, low=0.4`. Sandbox provenance → confidence usable for sim validation but **flagged `is_sandbox`** and never used for customer claims.
- **Missing-data behavior:** if a family's *required* signals are absent, its score is `None` (not 0) and it is excluded from "binding" selection; the family is listed in `missing_signals`. The classifier **never fabricates** a binding constraint from absent data. If no family clears the confidence floor (default 0.5), `binding_constraint=NONE` and the only safe action is `KEEP` (fail-safe no-op).
- **Hysteresis / anti-flapping:** a family must exceed the binding threshold for N consecutive snapshots (configurable, default 2) before it flips the binding constraint, to prevent recommendation churn. This is also a benchmark stability requirement (§9).
- **Tie-break order** (when scores are close, within 0.05): SLA-risk-reducing constraints first (LATENCY, MEMORY, COMMUNICATION, THERMAL) before cost constraints (ENERGY, UTILIZATION), because mis-serving SLA is worse than missing savings. Documented and tested.
- **All thresholds are config, not magic numbers**, defined in a new `ConstraintConfig` dataclass with conservative defaults and `# HEURISTIC` markers (matching the SLA engine's discipline).

### 7.1 Energy-bound
- **Input signals:** `EnergyState.price_per_mwh`, `price_percentile`, `power_draw_kw` vs `power_cap_kw`, carbon (secondary). Cross-region price spread.
- **Scoring:** `score = w1·clamp(price_percentile/100) + w2·clamp(power_draw/power_cap) + w3·clamp(cross_region_spread_normalized)`. High when current region price is in a high percentile AND cheaper regions exist AND/or approaching a power cap.
- **Confidence:** requires price for ≥2 regions for the spread term; price percentile needs region history (existing forecasting/history). Without multi-region price, score from percentile + cap only, lower confidence.
- **Missing-data behavior:** no price → `None` (energy-bound undetectable). This is the **existing, validated** signal — degrade to today's energy optimizer when energy is binding.
- **Safe recommendations:** `DEFER` (time-shift to cheaper hours), `CHOOSE_CHEAPER_REGION`/`REROUTE`/`MIGRATE` (subject to SLA + migration penalty), power throttle (for flexible workloads), `CHOOSE_LOWER_CARBON_REGION`.
- **Unsafe/disallowed:** migrating latency-critical/realtime_inference workloads for marginal price gains; any action violating residency/migration HARD SLA.
- **Simulator scenario:** `energy_price_arbitrage_multiregion.yaml` — anti-correlated regional price traces (mirrors the real CAISO/PJM anti-correlation that drives existing savings) + flexible batch mix; expected: defer/route reduces cost vs `current_price_only` without SLA violations.

### 7.2 Thermal-bound
- **Input signals:** `ThermalState.max_gpu_temp_c`, `throttling_gpu_count`, GPU `clocks_event_reasons` thermal bits (`SW_THERMAL`/`HW_THERMAL`/`HW_POWER_BRAKE`), `ambient_temp_c`, `cooling_headroom_pct` (proxy).
- **Scoring:** `score = w1·clamp((max_temp − T_safe)/(T_crit − T_safe)) + w2·clamp(throttling_gpu_count/total_gpus) + w3·thermal_throttle_bit_fraction`. `T_safe=70°C`, `T_crit=95°C` (reuse `dcgm_provider` thresholds).
- **Confidence:** high if DCGM temps + throttle reasons present; **medium/low if relying on weather proxy only.** Throttle-duration counters are disabled-by-default (§6.3), so prefer temp + `clocks_event_reasons` bits.
- **Missing-data behavior:** no GPU temp → score `None`; do not infer thermal stress from ambient weather alone (flag low confidence if attempted).
- **Safe recommendations:** `SPREAD` (de-densify hot nodes), `REROUTE`/`MIGRATE` away from thermally-stressed nodes/regions, `DEFER` non-urgent work, reduce placement density. (The SLA `HeuristicPredictor` already inflates p99 for `thermally_stressed`/`throttling` destinations — reuse.)
- **Unsafe/disallowed:** consolidating onto hot nodes; any fan/power-limit/clock manipulation (that is runtime control — out of bounds); recommending a destination that is itself throttling.
- **Simulator scenario:** `thermal_hotspot_mixed_cluster.yaml` — a subset of nodes runs hot/throttling under load; expected: spread/reroute reduces throttling and protects p99 without violating residency.

### 7.3 Queue-bound
- **Input signals:** `InferenceServiceState.requests_waiting`, `QueueState.queue_depth_jobs`/`est_wait_hours`, K8s pending-pod count, Ray `deployment_queued_queries`, `NodeState.gpu_allocatable − gpu_allocated`.
- **Scoring:** `score = w1·clamp(est_wait_hours/wait_ref) + w2·clamp(queue_depth/depth_ref) + w3·(1 − spare_capacity_fraction)`. High when queues are deep AND spare capacity is low in this region but exists elsewhere.
- **Confidence:** requires at least one queue signal; cross-region routing benefit requires queue+capacity for ≥2 regions.
- **Missing-data behavior:** no queue signal → `None`. This reuses the **existing, validated** queue-aware path (`QueueProvider`, `queue_delay_cost`).
- **Safe recommendations:** `REROUTE`/`MIGRATE` to a less-congested region with capacity, `SPREAD`, `SCALE_REPLICAS` (hint, if autoscaling is customer-controlled), `DEFER` deferrable batch out of the congestion window.
- **Unsafe/disallowed:** routing into a region whose spare capacity is below SLA `required_capacity_buffer_pct`; scaling beyond customer-authorized limits; consolidating during a queue surge.
- **Simulator scenario:** `queue_surge_latency_sensitive.yaml` — arrival burst on one region with a latency-sensitive SLA; expected: reroute/spread cuts queue wait and p95 without breaching capacity buffer.

### 7.4 Latency-bound
- **Input signals:** `InferenceServiceState.p95/p99_latency_ms`, `ttft_ms`, `queue_time_ms` vs SLA `max_p95/p99_latency_ms`; trend (rising tail).
- **Scoring:** `score = max(headroom_penalty(p99, sla_p99), headroom_penalty(p95, sla_p95), headroom_penalty(queue_time, sla_queue))`. Reuse the SLA evaluator's `_headroom_penalty` (0 below 50% of limit → 1 at limit). Distinguish from queue-bound: latency-bound is when **served** latency tails approach SLA even if queue depth is modest (e.g. KV-cache contention, slow tokens).
- **Confidence:** requires latency histograms + an SLA limit to compare against. If no SLA limit, use absolute thresholds at lower confidence.
- **Missing-data behavior:** no latency telemetry → `None`. **Do not** claim SLA-safe placement on unknown latency (the SLA engine already enforces this via `block_on_unknown`).
- **Safe recommendations:** `SPREAD`/`SCALE_REPLICAS` hints to add serving capacity, `REROUTE` to a lower-latency region (account for `network_rtt_ms`), avoid disruptive migrations of the very workload that is latency-critical.
- **Unsafe/disallowed:** migrating a latency-critical workload (cold-start tail risk — the predictor inflates p99 25% on migration); any KV-cache or batching-policy change (runtime/inference-plane — out of bounds).
- **Simulator scenario:** `latency_tail_kvcache_pressure.yaml` — rising p99/TTFT with high KV-cache usage under steady arrivals; expected: scale/spread recommendation reduces tail; migration is correctly NOT recommended for the critical service.

### 7.5 Communication-bound
- **Input signals:** `GPUState.nvlink_tx_bytes_per_s`/`pcie_tx_bytes_per_s` (PROF metrics), low `DCGM_FI_PROF_SM_ACTIVE` with high PCIe traffic (compute stalled on transfer — note SM_ACTIVE is disabled-by-default), `TopologyState.interconnect_class`, multi-GPU job spanning `PHB`/`SYS` links.
- **Scoring:** `score = w1·clamp(comm_bytes/compute_bytes_ratio) + w2·(interconnect_penalty)` where `interconnect_penalty` is high when a collective-heavy multi-GPU workload is placed across `SYS`/`NODE`/`PHB` rather than `NV#`/`PIX`.
- **Confidence:** **low by default** — PCIe/NVLink PROF metrics and SM_ACTIVE may be disabled; topology may be unavailable. Often `None`.
- **Missing-data behavior:** no comm/topology signal → `None`. Communication-bound is the **hardest to detect reliably**; the classifier must be conservative and prefer `NONE` over a low-confidence COMMUNICATION call.
- **Safe recommendations:** topology-aware **placement** (co-locate communicating GPUs on NVLink/PIX), `MIGRATE` a comm-heavy job to a node with better interconnect, avoid scattering a collective job across NUMA boundaries.
- **Unsafe/disallowed (HARD RULE):** **Aurelius can detect and optimize placement/topology, but must NOT touch NCCL, CUDA, or collective algorithms.** No changing NCCL env/topology hints, no ring/tree algorithm selection, no comm kernel tuning. Detection + placement recommendation only.
- **Simulator scenario:** `topology_fragmentation_h100.yaml` — a multi-GPU collective workload placed across poor interconnect vs good; expected: placement recommendation moves it onto NVLink-connected GPUs, modeled as reduced effective comm cost — **without** any collective-algorithm change.

### 7.6 Memory-bound (indirect)
- **Input signals:** `GPUState.mem_used_mb/mem_total_mb` ratio, `InferenceServiceState.kv_cache_usage` (vLLM), `preemptions_total` rising (vLLM preemption = KV pressure), OOM/eviction events if surfaced.
- **Scoring:** `score = w1·clamp(mem_used/mem_total) + w2·clamp(kv_cache_usage) + w3·clamp(preemption_rate/ref)`. High when framebuffer/KV cache near full and preemptions rising.
- **Confidence:** requires FB metrics (enabled by default) and/or vLLM KV metrics. Medium-high when both present.
- **Missing-data behavior:** no memory/KV signal → `None`.
- **Safe recommendations (CONSTRAINED):** `SPREAD` to reduce per-node memory pressure, `REROUTE`/`MIGRATE` to a node/region with more free framebuffer, `DEFER` admission of new memory-heavy work, `SCALE_REPLICAS` hint. **All are orchestration-level.**
- **Unsafe/disallowed (HARD RULE):** **Aurelius only DETECTS memory/cache pressure and recommends safe orchestration actions. It must NOT directly manage KV cache internals** — no KV cache sizing, eviction policy, block allocation, `gpu_memory_utilization`/`--max-model-len` tuning, paged-attention parameters, or any allocator change. Those belong to the inference engine.
- **Simulator scenario:** `memory_pressure_kvcache.yaml` — KV cache usage climbs toward 1.0 with rising preemptions on one service; expected: spread/reroute/defer relieves pressure; NO KV-internal action is emitted (asserted in the test).

### 7.7 Topology-bound
- **Input signals:** `TopologyState.pair_levels`/`interconnect_class`/`numa_affinity`, fragmentation (free GPUs scattered across nodes such that a multi-GPU job cannot get NVLink-local placement), gang-scheduling needs.
- **Scoring:** `score = w1·fragmentation_index + w2·(demand_for_tight_coupling)`. High when there is demand for tightly-coupled multi-GPU placement but free GPUs are topologically scattered.
- **Confidence:** **medium/low** — depends on topology availability (§6.9), often `None` in cloud-only deployments.
- **Missing-data behavior:** no topology → `None`; topology-bound disabled, others unaffected.
- **Safe recommendations:** topology-aware bin-packing/placement hints (pack a job onto NVLink-connected GPUs on one node), defragmentation via consolidation **of compatible workloads**, prefer placement that preserves future tight-coupling capacity.
- **Unsafe/disallowed:** consolidations that violate capacity buffers or co-locate antagonistic workloads; any interconnect/driver manipulation.
- **Simulator scenario:** reuse `topology_fragmentation_h100.yaml` from §7.5 but with the *placement/bin-packing* lens; expected: packing recommendation increases NVLink-local placements and reduces fragmentation index.

### 7.8 Utilization-bound
- **Input signals:** `GPUState.util_pct`/`tensor_active_ratio`, `gpu_allocated` vs actual `util_pct` (allocated-but-idle), region/cluster mean utilization, stranded capacity.
- **Scoring:** `score = w1·clamp((target_util − observed_util)/target_util)` for **under**-utilization (waste) and a separate high-util saturation signal. Detect allocated-but-idle GPUs (low `util_pct` on allocated nodes).
- **Confidence:** requires util metrics (enabled by default) and ideally allocation data (K8s). Medium without K8s.
- **Missing-data behavior:** no util → `None`.
- **Safe recommendations:** `CONSOLIDATE` idle/under-utilized workloads to free nodes (subject to SLA capacity buffer and the SLA `HeuristicPredictor`'s consolidation risk inflation), `SPREAD` if over-saturated, `DEFER`/admission control, surface stranded-capacity reports.
- **Unsafe/disallowed:** consolidating to the point of breaching `required_capacity_buffer_pct` or inflating queues/p99 beyond SLA; consolidating latency-critical services aggressively.
- **Simulator scenario:** `underutilization_stranded_capacity.yaml` — many half-idle allocated GPUs; expected: consolidation raises mean utilization and frees nodes without SLA breach; `CONSOLIDATE` is suppressed where it would breach capacity buffers.

---

## 8. Trust-Boundary Matrix

Aurelius **detects** all constraints and **recommends** orchestration-level actions. It never enters the inference/runtime data plane. Customer trust level reflects how comfortable an operator is letting Aurelius act on that constraint (recommend → shadow → gated live).

| Constraint | Safe Aurelius actions (recommend/dry-run; gated live later) | Unsafe / disallowed actions (NEVER) | Customer trust level | MVP priority |
|---|---|---|---|---|
| **Energy-bound** | Defer/time-shift, choose-cheaper-region, reroute, migrate (SLA-checked), power-throttle flexible jobs, lower-carbon routing | Direct grid/market interaction; migrating latency-critical for marginal price; bypassing residency SLA | High (already validated in backtests) | **P0** (reuse existing) |
| **Queue-bound** | Reroute/spread to regions with capacity, defer batch, scale-replica hints | Scaling beyond authorized limits; routing below capacity-buffer SLA; consolidating during surge | High (queue-aware path exists) | **P0** |
| **Latency-bound** | Spread/scale hints, reroute to lower-latency region (RTT-aware) | Migrating the latency-critical workload itself; **any KV-cache/batching/runtime change** | Medium-High | **P1** |
| **Thermal-bound** | Spread off hot nodes, reroute/migrate away from throttling, defer | **Fan/power-limit/clock/thermal-runtime control**; consolidating onto hot nodes | Medium | **P1** |
| **Utilization-bound** | Consolidate idle workloads (capacity-buffer-safe), spread if saturated, stranded-capacity reports | Consolidation breaching capacity buffer/SLA; aggressive consolidation of latency-critical | Medium | **P1** |
| **Memory-bound (indirect)** | **Aurelius only detects memory/cache pressure and recommends safe orchestration actions (spread, reroute, defer admission, scale hints). It must not directly manage KV cache internals.** | **Direct KV cache ownership/sizing/eviction, allocator changes, `gpu_memory_utilization`/paged-attention/`--max-model-len` tuning** | Medium-Low (newer signal) | **P2** |
| **Communication-bound** | **Aurelius can detect and optimize placement/topology, but must not touch NCCL/CUDA/collective algorithms.** Topology-aware placement, migrate comm-heavy job to better-interconnect node | **NCCL replacement/tuning, CUDA/collective-algorithm changes, comm-kernel manipulation** | Low (hard to detect, intrusive if wrong) | **P3** |
| **Topology-bound** | Topology-aware bin-packing/placement hints, defrag via safe consolidation | Interconnect/driver manipulation; capacity-breaching consolidation | Low (telemetry often unavailable) | **P3** |

**Global invariants (all rows):** recommendation-only by default; HARD SLA blocks any action; missing telemetry → fail-safe `KEEP`; live execution only through the existing signed-policy/kill-switch/quantile-gate path; Kubernetes access read-only by default.

---

## 9. Synthetic Cluster Simulator Design

> **The simulator is not reality. The simulator is an approximation used for validation and development.** Its purpose is to develop and regression-test the classifier and optimizer when real GPU clusters are unavailable — not to produce savings claims. Numbers from the simulator are `is_sandbox=True` and are **rejected** from any economic/SLA claim (reusing the existing ElectricityMaps-sandbox rejection discipline).

**Module:** new `aurelius/simulation/cluster/` (distinct from the existing energy-focused `aurelius/simulation/`). The existing `WorkloadSimulator` (seeded GPU-typed job generation) is reused as the workload source.

### 9.1 What the simulator models
A discrete-event (hourly tick, with optional sub-hour for queue/latency) model of:
- **Regions** (reuse canonical region ids + `region_registry`), each with **nodes → racks/zones → GPUs**.
- **GPUs** with type (h100/a100/…), framebuffer, power curve, thermal model, NVLink/PCIe topology.
- **Queues** per region/pool with arrival processes (Poisson/bursty/diurnal — reuse `QueueProvider` patterns).
- **Workloads** from `WorkloadSimulator` (training/fine-tuning/batch/realtime mix), each with a `WorkloadState` evolving over time.
- **Energy prices** (replay real CSV traces from `data/` or synthetic scenarios — reuse `generate_price_scenario`).
- **Topology** per node (NVLink groups, PCIe, NUMA) → fake `nvidia-smi topo -m`.
- **SLA policies** (reuse `aurelius/sla/` configs).
- **Thermal behavior:** GPU temp as f(utilization, ambient, node density, cooling proxy); throttling when temp > threshold (sets `clocks_event_reasons` bits and degrades effective throughput).
- **Migration costs:** checkpoint/transfer/warmup time (reuse `Job.migration_cost_hours`); cold-start latency penalty on the destination.
- **Cache-affinity proxy:** KV-cache usage as f(concurrent requests, model size, available framebuffer); prefix-cache hit rate as f(request locality) — a **proxy**, not a real cache.
- **Communication pressure:** effective throughput penalty when a multi-GPU job spans poor interconnect — a **proxy** for collective overhead, NOT a NCCL model.
- **Baseline schedulers** (FIFO, current_price_only, round-robin, etc.) and the **Aurelius scheduler** (constraint-aware), run on identical environments.

### 9.2 The simulator exposes FAKE versions of the SAME connectors
This is the central design constraint: **Aurelius must not know whether it is connected to the simulator or a real customer stack.** Each connector Protocol (§6) has a simulator-backed implementation that emits **byte-compatible payloads**:
- **Fake Prometheus HTTP API** — an in-process responder (or a real local HTTP server for integration tests) answering `/api/v1/query` and `/api/v1/query_range` with the exact JSON envelope (`status/data/resultType/result`, string sample values) from §6.1, populated from simulator state.
- **Fake raw `/metrics` endpoints** — Prometheus text exposition for fake **dcgm-exporter** (`:9400`, exact `DCGM_FI_*` names + label capitalization, only the default-enabled set unless the scenario opts in), **vLLM** (`:8000`, `vllm:` names with correct V0 or V1 variant per scenario), **Triton** (`:8002`, `nv_*`), **Ray** (`ray_serve_*`).
- **Fake Kubernetes API payloads** — synthetic `V1NodeList`/`V1PodList`-shaped dicts with `nvidia.com/gpu` capacity/allocatable (string quantities), labels, taints, pending pods.
- **Fake `nvidia-smi topo -m` output** — exact legend/matrix text (§6.9) for the scenario's node topologies.
- **Fake energy API payloads** — reuse existing CSV replay + EM/WattTime sandbox-shaped responses.

These fakes live in `aurelius/simulation/cluster/fakes/` and are the SAME classes the connector tests use. The "exact connector code that production would use" (per the brief) is validated because the production connector parses the fake's output without modification.

### 9.3 Anti-overfitting risks and how Aurelius must avoid exploiting the simulator
The simulator can be gamed; the optimizer must not. Explicitly enumerated risks and mitigations:

- **Simulator overfitting:** tuning classifier/optimizer thresholds to the specific synthetic distributions. *Mitigation:* hold-out scenario families never used for tuning; report sim and (where available) real-replay results side by side; treat sim-only gains as unproven.
- **Unrealistic optimization artifacts:** the optimizer exploits modeling shortcuts (e.g. instantaneous migration, perfectly anti-correlated prices, zero cold-start). *Mitigation:* migrations always cost time + cold-start latency; thermal/queue dynamics have inertia; prices are replayed from real traces where possible; randomized perturbations on each seed family.
- **Pathological migration behavior:** chasing tiny gains with constant migrations. *Mitigation:* migration penalty modeling + SLA `max_migrations_per_hour` + classifier hysteresis (§7) + a **migration-churn metric** treated as a regression if it spikes.
- **Synthetic telemetry limitations:** fake KV-cache/comm/thermal are proxies, not physics. *Mitigation:* label all sim outputs `is_sandbox`; never publish sim numbers as savings; require real-replay corroboration before claims.
- **Exploiting unrealistic assumptions:** *Mitigation core rule below.*

> **Core anti-gaming rule:** Optimization strategies that improve simulator metrics while likely degrading real-world operational behavior are treated as **regressions**, not improvements. A change that increases sim savings but increases migration churn, SLA risk, or relies on a known sim idealization must be rejected.

### 9.4 Validation philosophy (must be implemented, not just stated)
- **Realism checks:** automated assertions that simulator outputs stay within plausible physical ranges (GPU temp ≤ 105°C, util 0–100, KV usage 0–1, latency monotone in load, power within rating). A scenario failing realism checks is invalid for benchmarking.
- **Replay validation:** where real telemetry traces exist (e.g. recorded DCGM `.prom`, real price CSVs, `sample_customer_workload_trace.csv`), replay them through the SAME connectors and confirm the classifier/optimizer behave sensibly; compare to sim behavior.
- **Baseline comparison:** every optimizer run is compared against the full baseline set (§10) on the identical environment; gains are reported relative to `current_price_only` (the honest primary baseline) and the strongest applicable baseline, never only FIFO.
- **Stability requirements:** repeated runs with the same seed are bit-identical (replay mode); across a seed family, variance is reported with confidence intervals (reuse `SavingsReport` bootstrap CIs); migration churn and SLA-violation counts must not regress.
- **Anti-overfitting guardrails:** a tuning/validation scenario split; a rule that any threshold change must improve a held-out scenario family, not just the tuning set; CI fails if sim-only gains are presented as claims.

### 9.5 Benchmarking & controlled-variable experimental rules
**Goal: isolate optimizer effects — NOT change the environment until results improve.** When comparing optimizer behavior, the workload mix, arrival patterns, topology, energy traces, thermal conditions, queue behavior, migration assumptions, SLA policies, random seeds, and simulator parameters MUST remain identical across baseline runs, Aurelius runs, and optimizer revisions — unless the experiment explicitly tests changing one of those variables (a "controlled variable" experiment changes exactly one thing).

The 10 implementation requirements (all must be built in Phase 11):

1. **Deterministic seeds.** Every scenario takes an explicit `seed` and supports a deterministic replay mode (same inputs → same outputs). The existing harness already uses `seed=42`; extend to all simulator subsystems (arrivals, thermal noise, topology).
2. **Immutable benchmark scenarios.** Canonical scenarios are versioned and frozen under:
   ```
   benchmarks/
     v1/
       queue_surge_latency_sensitive.yaml
       thermal_hotspot_mixed_cluster.yaml
       topology_fragmentation_h100.yaml
       energy_price_arbitrage_multiregion.yaml
       memory_pressure_kvcache.yaml
       underutilization_stranded_capacity.yaml
   ```
   These files must not silently change. A CI check hashes `benchmarks/v1/**` and fails if a frozen scenario is modified without a version bump.
3. **Controlled-variable experiments.** When evaluating optimizer changes, ONLY the optimizer logic changes; simulator/environment changes are separate experiments with their own scenario version.
4. **Benchmark metadata.** Every benchmark report records: `scenario_version`, `seed`, `simulator_version`, `optimizer_version`, `config_hash`, `sla_config_hash`, `workload_mix_hash`. (The current harness records run_ts/method/primary_baseline but **none of these hashes** — this is a concrete gap to close.)
5. **Regression protection.** If benchmark conditions changed (any metadata field differs), comparisons are invalidated or clearly labeled; apparent gains under changed conditions are NOT treated as optimizer improvements.
6. **Anti-benchmark-gaming rule.** Gains that exist only because the simulator, randomness, workload mix, topology, or telemetry assumptions changed are invalid regressions.
7. **Replay mode.** The simulator supports exact replay: same inputs → same outputs (deterministic event ordering, seeded RNG per subsystem).
8. **Benchmark stability checks.** Aurelius periodically reruns the canonical suite to detect optimizer regressions, unstable heuristics, migration churn, SLA regressions, and overfitting to synthetic scenarios. Wire into CI (extend the existing `benchmark-smoke` job) and a scheduled run.
9. **Scenario evolution policy.** New scenarios may be added; old canonical scenarios are preserved for historical comparison. `benchmarks/v1/` is frozen; new families go in `benchmarks/v2/` etc.
10. **Reporting.** Every benchmark comparison explicitly states what variables changed, what stayed fixed, whether the environment was identical, and whether the comparison is valid.

**The simulator itself is a versioned dependency.** `simulator_version` is recorded in every report; changing simulator behavior may invalidate historical optimizer comparisons, and such changes require a version bump and re-baselining (not silent edits).

---

## 10. Baseline vs Aurelius Validation Plan

### 10.1 Baselines
All implemented as comparable policies (extend `aurelius/backtesting/baselines.py` and `benchmarks/baseline_matrix.yaml`; several already exist). Each runs on the **identical** environment (§9.5).

| Baseline | Definition | Status |
|---|---|---|
| **FIFO / no optimization** | Jobs in submission order, default region, full power, no awareness | Exists (`fifo`, `peak_blind_asap`) |
| **current_price_only** | Pick cheapest region at `earliest_start`; no time-shift, no forecasting | Exists (**primary baseline**) |
| **greedy energy optimizer** | Current `JobScheduler` greedy/greedy_migrate with energy objective | Exists (this is "Aurelius v1") |
| **SLA-aware optimizer** | Greedy energy optimizer + `sla_registry` wired (HARD blocks, SOFT/risk ranking) | **New** (wiring per §3) |
| **constraint-aware optimizer** | Classifier-selected strategy per binding constraint + SLA-aware | **New** (the target) |

Each successive baseline isolates one capability so we can attribute gains: energy-only → +SLA-safety → +constraint-awareness.

### 10.2 Metrics
Extend `aurelius/simulation/metrics.py` (`ScheduleMetrics`) and `SavingsReport` — the current metrics are energy-only and lack most of these:

| Metric | Definition | Source |
|---|---|---|
| total cost ($) | energy + (optional compute) cost | exists |
| cost / token | total cost ÷ tokens served | **new** (needs token counters) |
| tokens / joule | tokens ÷ energy (J) | **new** (efficiency) |
| GPU utilization | mean/median `util_pct`, stranded-capacity % | **new** |
| p95 / p99 latency | served-request tail | **new** (needs inference metrics) |
| queue wait | mean/p95 `est_wait_hours`/`queue_time_ms` | partial (queue delay tracked) |
| SLA violations | count + rate of HARD-constraint breaches | **new** (needs SLA wired) |
| thermal throttling | throttled GPU-hours / throttle events | **new** |
| migrations | count + churn rate (per workload per hour) | partial (count derivable) |
| topology score | fraction of multi-GPU jobs on good interconnect | **new** |
| energy spend ($) | as today | exists |
| net savings after migration penalties | gross savings − migration cost (time+latency+energy) | partial (migration cost modeled) |

**Reporting rule:** every comparison reports the full metric vector for baseline AND Aurelius on the identical environment, with §9.5 metadata and validity flag. A cost win that worsens p99/SLA/throttling/churn is flagged, not celebrated.

---

## 11. Phase-by-Phase Implementation Plan

> Each phase: files to add, files to modify, tests to add, success criteria, risks, and what NOT to build. **Every phase begins by re-reading the product goal and this document, then inspecting live repo state (per the header).** Phases are ordered so each delivers a wired, testable increment. Energy-optimizer behavior must be preserved (regression-tested) whenever constraint-aware mode is OFF.

### Phase 1 — Normalized state model
- **Add:** `aurelius/state/__init__.py`, `aurelius/state/models.py` (§5 models + `Provenance`), `aurelius/state/store.py` (append-only, leakage-safe `last_known ≤ T` snapshot store, hashable), `aurelius/state/normalize.py` (adapters from existing `GPUMetrics`/`QueueState`/`GPUHealthScore` + connector payloads → `ClusterState`).
- **Modify:** nothing in the optimizer yet (additive). Optionally `aurelius/__init__.py` exports.
- **Tests:** `tests/test_state_models.py` (validation: UTC, ranges, None-not-zero), `tests/test_state_store.py` (leakage-safe lookup parity with `_lookup_last_known`), `tests/test_state_normalize.py` (existing fixtures → ClusterState).
- **Success criteria:** `ClusterState` can be built from existing `.prom`/CSV fixtures; round-trips; missing fields are `None`; store passes the same leakage tests as `QueueProvider`.
- **Risks:** model churn vs existing `models.py`; mitigate by adapting, not renaming.
- **Do NOT build:** the classifier, connectors, or any optimizer change yet.

### Phase 2 — Prometheus-native connector
- **Add:** `aurelius/connectors/__init__.py`, `aurelius/connectors/base.py` (Protocols + `ConnectorResult` with provenance), `aurelius/connectors/prometheus.py` (`requests`-based instant/range query, exact §6.1 response parsing incl. string→float, defensive histogram handling, basic+bearer auth, TLS toggle — generalize the auth code already in `dcgm_provider`), `aurelius/connectors/metrics_text.py` (shared Prometheus text-exposition parser generalized from `parse_prometheus_text`).
- **Modify:** `aurelius/ingestion/dcgm_provider.py` to delegate parsing to the shared parser (no behavior change); `pyproject.toml` optional extras (`[prometheus]`).
- **Tests:** `tests/test_prometheus_connector.py` (parse real-shaped JSON fixtures incl. vector/matrix/scalar, string values, missing/error envelopes, auth header construction, failure→empty).
- **Success criteria:** connector returns normalized series from fixture JSON; failure modes return empty + flagged, never raise.
- **Risks:** PromQL correctness per signal — defer signal-specific queries to Phase 3.
- **Do NOT build:** signal-specific scrapers (next phase); live network calls in unit tests.

### Phase 3 — DCGM / vLLM / Triton / Ray / OpenTelemetry adapters
- **Add:** `aurelius/connectors/dcgm.py` (use §6.3 exact metrics; FIX the existing FB_TOTAL/units/throttle-name/disabled-by-default issues; flag disabled metrics as `None`), `aurelius/connectors/vllm.py` (V0/V1 branch per §6.4), `aurelius/connectors/triton.py` (§6.5, µs latency, opt-in summaries→None), `aurelius/connectors/ray_serve.py` (§6.6). OTel handled as "read via Prometheus" (document; no scraper).
- **Modify:** `aurelius/state/normalize.py` to populate `GPUState`/`InferenceServiceState`/`ThermalState` from these; `.prom` fixtures extended with vLLM/Triton/Ray samples (`data/fixtures/`).
- **Tests:** one per connector parsing real-shaped fixtures; **explicit tests that disabled-by-default DCGM metrics yield `None` (not 0)**; vLLM V0 vs V1 fixture tests; a test asserting the production connector parses the simulator's fake `/metrics` byte-for-byte (cross-checks §9.2).
- **Success criteria:** each inference/GPU signal in §5 populated from fixtures; honest `None` for absent metrics; existing `test_dcgm_provider.py` still green.
- **Risks:** metric-name drift upstream — re-verify against §6 URLs at implementation time; histogram quantile computation correctness.
- **Do NOT build:** any OTLP receiver; NVML/DCGM-binding topology (Phase 5/§6.10).

### Phase 4 — Kubernetes connector
- **Add:** `aurelius/connectors/kubernetes.py` (`load_incluster_config` else `load_kube_config`; `list_node`/`list_pod_for_all_namespaces`; parse `nvidia.com/gpu` capacity/allocatable string→int, labels, taints, pod GPU limits, pending pods; `_continue` pagination; **read-only**). `deploy/rbac-readonly.yaml` (the minimal ClusterRole from §6.8).
- **Modify:** `normalize.py` → `NodeState`/spare-capacity/pending-queue; `pyproject.toml` `[k8s]` extra; `.env.example` (kubeconfig/in-cluster note).
- **Tests:** `tests/test_kubernetes_connector.py` with fake `V1NodeList`/`V1PodList` payloads (the §9.2 fakes); capacity/allocatable parsing; missing `resources`→None; failure→empty.
- **Success criteria:** node/pod/GPU-capacity/pending-queue populate `ClusterState` from fake payloads; documented read-only RBAC; no write calls anywhere in the connector.
- **Risks:** cluster-version API differences; metrics-server GPU absence (documented — use DCGM).
- **Do NOT build:** any mutating call (cordon/drain/scale) — those are execution-layer and gated.

### Phase 5 — Topology collector
- **Add:** `aurelius/connectors/topology.py` (parse `nvidia-smi topo -m` by header name per §6.9; optional NVML path via `nvidia-ml-py` behind `[nvml]` extra; derive `interconnect_class`).
- **Modify:** `normalize.py` → `TopologyState`; `pyproject.toml` `[nvml]` extra.
- **Tests:** `tests/test_topology_connector.py` with fake `topo -m` text fixtures (NVLink/PCIe/SYS matrices, variable affinity columns); NVML mocked; absent→`TopologyState=None`.
- **Success criteria:** topology parsed into `pair_levels`/`interconnect_class`; graceful `None` when node access absent.
- **Risks:** text-format variability across driver versions; NVML import collisions (pin `nvidia-ml-py`).
- **Do NOT build:** DCGM-binding topology (§6.10); any interconnect manipulation.

### Phase 6 — Synthetic cluster simulator
- **Add:** `aurelius/simulation/cluster/` — `engine.py` (discrete-event), `model.py` (regions/nodes/racks/GPUs/queues/thermal/topology/migration/cache-affinity/comm proxies), `fakes/` (fake Prometheus HTTP, fake `/metrics` for dcgm/vllm/triton/ray, fake K8s payloads, fake `nvidia-smi topo -m`, fake energy), `scenarios.py` (loader for `benchmarks/v*/`). Reuse `WorkloadSimulator`.
- **Modify:** wire fakes to the §6 connector Protocols so the same connectors consume them; `benchmarks/v1/*.yaml` (the six frozen scenarios from §7/§9).
- **Tests:** `tests/test_cluster_simulator.py` (determinism/replay: same seed → identical state stream; realism checks; each scenario produces the intended binding-constraint signature), `tests/test_fake_connectors.py` (production connector parses fake output identically to real fixtures).
- **Success criteria:** Aurelius cannot distinguish sim from real at the connector boundary; replay is bit-identical; scenarios exhibit the targeted constraints.
- **Risks:** sim idealization → overfitting (mitigations §9.3); scope creep.
- **Do NOT build:** a physically accurate NCCL/thermal/cache model — these are explicitly proxies.

### Phase 7 — Constraint classifier
- **Add:** `aurelius/constraints/__init__.py`, `aurelius/constraints/classifier.py` (`ConstraintClassifier`, per-family scorers, `ConstraintConfig` with `# HEURISTIC` thresholds), `aurelius/constraints/types.py` (`ConstraintType`, `ConstraintAssessment` if not in `state/models.py`).
- **Modify:** nothing in optimizer yet (classifier is read-only over `ClusterState`).
- **Tests:** `tests/test_constraint_classifier.py` — one per constraint family using the matching simulator scenario; missing-signal→`None`/`NONE`; hysteresis; tie-break order; confidence math; **no fabricated binding constraint from absent data**.
- **Success criteria:** each scenario yields the expected binding constraint with calibrated confidence; absent telemetry → `NONE` + `missing_signals`; deterministic.
- **Risks:** threshold calibration/overfitting (use held-out scenarios); over-confident classification on sparse data.
- **Do NOT build:** the strategy selection/optimizer change yet; per-constraint actions beyond emitting `safe_actions`/`disallowed_actions`.

### Phase 8 — Cost / risk / migration model
- **Add:** `aurelius/constraints/cost_model.py` — per-constraint cost terms + migration penalty model (checkpoint+transfer+warmup+cold-start latency + destination energy), reusing `ObjectiveFunction` and the SLA `RiskBreakdown`. `net_benefit = expected_effect − migration_penalty − sla_risk`.
- **Modify:** `aurelius/sla/telemetry.py` `HeuristicPredictor` only if needed (additively); keep `# HEURISTIC` markers.
- **Tests:** `tests/test_cost_risk_migration.py` — migration penalties make marginal migrations net-negative; churn penalized; SLA risk folds in; parity with existing objective when constraint-aware off.
- **Success criteria:** marginal/pathological migrations score net-negative; numbers reconcile with existing objective on energy-only cases.
- **Risks:** double-counting between objective and risk terms — unit-test additivity.
- **Do NOT build:** new ML predictors (heuristic + existing models only this phase).

### Phase 9 — Constraint-aware recommendation engine
- **Add:** `aurelius/constraints/engine.py` — `ConstraintAwareEngine`: ClusterState → classifier → strategy selection → SLA-aware `JobScheduler` (with `sla_registry` WIRED) → ranked `Recommendation[]` (fail-safe `KEEP` on low confidence/missing data/blocked SLA).
- **Modify (CRITICAL WIRING):** `aurelius/backtesting/engine.py` and the relevant CLI paths to pass `sla_registry`/`region_contexts`/`current_states` into `JobScheduler` (closing the §3 gap), behind a config flag defaulting to OFF (preserve existing behavior). `OptimizationConfig` gains `constraint_aware: bool=False` and `sla_config_path: Optional[str]`.
- **Tests:** `tests/test_constraint_engine.py` (end-to-end per scenario: correct strategy, SLA-safe, fail-safe on missing data, **memory-bound emits NO KV-internal action**, **communication-bound emits NO NCCL/CUDA action** — assert the disallowed-action invariants from §8); `tests/test_sla_wiring.py` (SLA now blocks/ranks in the real path); regression test that `constraint_aware=False` reproduces current backtest numbers bit-for-bit.
- **Success criteria:** wired into the real path; behavior changes correctly per binding constraint; OLD energy-optimizer behavior preserved exactly when flag OFF; HARD SLA enforced; trust-boundary invariants asserted by tests.
- **Risks:** regression in validated savings — gate behind flag + regression tests; SLA misconfiguration → over-blocking (use `block_on_unknown=False` default).
- **Do NOT build:** any execution/mutation; any inference-plane action.

### Phase 10 — CLI reports
- **Add:** `cmd_assess` (print `ConstraintAssessment` for a ClusterState/snapshot/scenario) and `cmd_recommend` (constraint-aware recommendations, dry-run) in `aurelius/cli.py`; reporting in `aurelius/reporting/` for recommendations (text + JSON + HTML, reusing `render_html_report` patterns) with constraint/confidence/SLA-status/why columns.
- **Modify:** `aurelius/cli.py` arg parsing; `SavingsReport`/metrics for the §10.2 metric vector.
- **Tests:** `tests/test_cli_assess_recommend.py` (CLI runs against a simulator scenario, emits valid report; `--output` JSON schema; no secrets in output).
- **Success criteria:** an operator can run `aurelius assess`/`recommend` against the simulator (or a real Prometheus) and get an explained, dry-run recommendation report.
- **Risks:** CLI sprawl — reuse existing argparse + reporting patterns.
- **Do NOT build:** live execution from these commands (dry-run only).

### Phase 11 — Validation, benchmarking, and continuous optimization
- **Add:** `benchmarks/v1/*.yaml` (frozen scenarios, §9.5), `benchmarks/run_constraint_benchmark.py` (controlled-variable runner emitting full §9.5 metadata incl. all hashes + `simulator_version`/`scenario_version`), `benchmarks/scenario_lock.py` (CI hash-check of `benchmarks/v1/**`), realism/replay/stability checks.
- **Modify:** `benchmarks/compare_against_previous.py` to validate metadata identity and label invalid comparisons; `aurelius/reporting/savings_report.py` for the new metric vector + metadata; CI `benchmark-smoke` job to also run a frozen constraint scenario and the scenario-lock check.
- **Tests:** `tests/test_benchmark_metadata.py` (hashes present, deterministic), `tests/test_scenario_immutability.py` (frozen scenarios unchanged), `tests/test_benchmark_stability.py` (replay determinism; churn/SLA non-regression).
- **Success criteria:** baseline-vs-Aurelius comparisons run on identical environments with full metadata; CI fails on silent scenario change or sim-only "gains"; stability/churn tracked.
- **Risks:** flaky nondeterminism — enforce per-subsystem seeding; benchmark runtime — keep `--quick` for CI, full for scheduled.
- **Do NOT build:** any claim from sandbox numbers (sim is `is_sandbox`).

### Phase 12 — Production hardening
- **Add:** connector health checks (`aurelius/connectors/health.py`), staleness detection in the state store, confidence scoring surfaced in every recommendation, audit logging for assessments/recommendations (reuse `log_execution_audit` patterns), secret-redaction in logs, dry-run/no-op fail-safe enforcement.
- **Modify:** connectors for timeouts/retries/circuit-breaking; `.env.example` + docs for read-only K8s RBAC; `enterprisedocs/` security/deployment notes.
- **Tests:** `tests/test_production_hardening.py` (stale telemetry → fail-safe no-op + low confidence; no secrets in logs; connector health endpoints; dry-run default).
- **Success criteria:** the §13 checklist is met and tested; safe under partial/stale telemetry; recommendation-only default verified end-to-end.
- **Risks:** false sense of safety — verify with adversarial/partial-telemetry tests.
- **Do NOT build:** default-on live execution; anything that mutates a customer cluster without the gated signed-policy path.

---

## 12. Test Strategy

Follow existing conventions: **pytest**, `tests/conftest.py` fixtures, fixture files under `data/fixtures/`, live tests skipped via `pytest.mark.skipif` when env vars absent, determinism via `seed=42`. New tests mirror the per-module pattern (`tests/test_<module>.py`).

- **Unit tests** — every new model (validation: UTC, ranges, None-not-zero), every connector parser (real-shaped fixtures), every classifier scorer (per-family), the cost/risk/migration model, the recommendation engine. Pure functions where possible.
- **Integration tests** — connectors → `normalize` → `ClusterState` → classifier → engine → `Recommendation`, end-to-end on simulator scenarios. The `backtest`/`recommend` CLI paths.
- **Sandbox connector tests** — assert the **production** connector parses the **simulator's fake** output byte-for-byte (the §9.2 invariant). This is the proof that "Aurelius can't tell sim from real."
- **Simulator scenario tests** — each `benchmarks/v1/*.yaml` produces its intended binding-constraint signature; realism checks pass; replay is deterministic.
- **SLA safety tests** — HARD constraints block in the real path (post-wiring); `block_on_unknown` behavior; fail-safe `KEEP` on missing data; the **trust-boundary invariants** (memory-bound emits no KV-internal action; communication-bound emits no NCCL/CUDA action) asserted explicitly.
- **Regression tests (critical)** — `constraint_aware=False` and `sla_registry=None` reproduce **current** backtest/scheduler numbers bit-for-bit on existing data, proving the old energy optimizer is preserved when constraint-aware mode is off. Run the existing suite (must stay green) plus a golden-output comparison on a fixed seed/dataset.
- **Missing-telemetry / failure-mode tests** — every connector failure returns empty+flagged (never raises); partial `ClusterState` degrades confidence and falls back to `KEEP`; stale telemetry detected.
- **Determinism/stability tests** — replay bit-identical; churn/SLA-violation non-regression across runs.

CI: keep `lint` (ruff) blocking, `test` excluding live, extend `benchmark-smoke` with a frozen constraint scenario + scenario-lock hash check.

---

## 13. Production Hardening Requirements

- **Recommendation-only default.** `constraint_aware` produces `Recommendation`s only. No execution adapter is invoked by the classifier/engine. Live execution remains the existing opt-in, signed-policy, kill-switch, quantile-gated path (`aurelius/execution/`), default `dry_run`.
- **Read-only Kubernetes access.** The K8s connector performs only `get/list/watch`; ship the minimal read-only ClusterRole (`deploy/rbac-readonly.yaml`). No cordon/drain/scale in any connector.
- **No secrets in logs.** Redact bearer tokens / passwords / kubeconfig / API keys in all logs and reports (audit existing logging too). Add a `tests/test_no_secret_logging.py`.
- **Stale telemetry detection.** Every `ClusterState`/signal carries `sample_age_s`; signals older than a configurable TTL are treated as missing (→ `None`), lowering confidence and biasing toward `KEEP`.
- **Confidence scoring.** Every `ConstraintAssessment` and `Recommendation` carries a `confidence` derived from signal completeness + provenance + staleness; low confidence → conservative/no-op.
- **Fail-safe no-op behavior.** Missing/contradictory/stale data, low confidence, or blocked SLA → `Recommendation(is_noop=True, action=KEEP)`. Never recommend a disruptive action on weak evidence.
- **Connector health checks.** `aurelius/connectors/health.py` exposes per-connector reachability/auth/freshness status; surfaced in reports and the API `/health`.
- **Dry-run mode.** Default everywhere; explicit, audited opt-in to anything else.
- **Audit logs.** Structured JSON for every assessment and recommendation (reuse `log_execution_audit` format): inputs' provenance, binding constraint, action, SLA status, confidence, rationale.
- **Sandbox rejection.** Any `is_sandbox=True` provenance is excluded from savings/SLA claims (reuse the ElectricityMaps-sandbox discipline).

---

## 14. Open Questions / Assumptions

**Assumptions:**
1. Customers expose telemetry via Prometheus in the common case; raw `/metrics` scraping and K8s are secondary paths. (Aligns with the team's `API-NEEDED/PROMETHEUS_DCGM.md`.)
2. The `aurelius/sla/` policy engine becomes the single SLA source of truth; `Job`-level SLA fields are derived from it over time (not maintained in parallel).
3. Energy/queue/GPU-health signals (already validated) are the trustworthy P0 base; thermal/memory/topology/communication are progressively lower-confidence.
4. The existing execution trust boundary (dry-run/signed-policy/kill-switch/quantile-gate) is the only path to mutation and is not weakened.
5. New third-party libs (`kubernetes`, `nvidia-ml-py`, `opentelemetry-*`, `prometheus-api-client`) are optional extras, lazily imported.

**Open questions (resolve during the relevant phase, with the user where needed):**
1. **Per-GPU vs region-level control.** DCGM gives per-GPU observability, but placement granularity depends on the customer's scheduler (K8s node selectors / Slurm GRES / Ray labels). How granular should recommendations be by default? (Affects §5 `GPUState` usage and §7 placement actions.)
2. **Binding-constraint vs multi-constraint.** Is a single binding constraint sufficient, or do real clusters need a small ranked set (e.g. thermal AND queue simultaneously)? `ConstraintAssessment` supports ranked `scores`; the engine currently acts on the top one. Validate against scenarios.
3. **SLA defaults when none provided.** If a customer supplies no SLA config, does the classifier assume `best_effort` (cost-aggressive) or a conservative default? Leaning conservative (fail-safe), but confirm.
4. **Cost/token & tokens/joule** require token counters and an energy attribution model per service — feasible from vLLM/Triton token + DCGM energy metrics, but the attribution (which GPU's energy → which service) is non-trivial when services share GPUs. Flag as approximate.
5. **Topology/communication telemetry availability.** In cloud/managed-K8s deployments node-local `nvidia-smi`/NVML may be unavailable. Confirm how often topology will actually be `None` for target ICP (neoclouds likely have it; managed K8s may not).
6. **Real-replay corpus.** What real telemetry traces (DCGM `.prom`, inference `/metrics`, K8s snapshots) can we record from a pilot to validate the classifier beyond the simulator? Needed before any constraint-aware savings claim.
7. **vLLM version detection.** How do we reliably detect V0 vs V1 to branch metric names — from a `vllm:` metric presence probe, or a configured version? (Affects §6.4 connector.)
8. **`simulator_version` semantics.** Define what constitutes a behavior-changing simulator edit requiring a version bump + re-baseline (vs a non-behavioral refactor).

**Ambiguities flagged in research (do not treat as settled):**
- Prometheus native-histogram response shape is version-dependent (§6.1).
- OTel `hw.gpu.*` semantic conventions are Development-status and differ from DCGM names (§6.7).
- `nvidia-smi topo -m` has no machine-readable form and column set varies by driver; NVTAGS vs nvidia-smi legend wording differs (§6.9).
- DCGM disabled-by-default metrics mean ECC/throttle-duration/NVLink-error/SM_ACTIVE telemetry may simply be absent in a given cluster (§6.3) — the classifier must assume they can be `None`.

---

## 15. Final Implementation Checklist

A later coding agent can follow this. **Each item is "done" only under the header's definition of complete (wired, behavior-changing, fail-safe, old-behavior-preserved, evidence provided) — not merely "file exists."**

**Per-phase preamble (every phase):**
- [ ] Re-read the product goal and this document.
- [ ] Inspect live repo state; diff reality vs this plan; note drift in this doc.
- [ ] Confirm prior phases are actually wired (not just present).

**Phase 1 — state model:** [ ] `aurelius/state/{models,store,normalize}.py` · [ ] None-not-zero validation · [ ] leakage-safe store parity tests · [ ] existing models adapted, not renamed.

**Phase 2 — Prometheus connector:** [ ] `connectors/{base,prometheus,metrics_text}.py` · [ ] exact §6.1 parsing (string→float, defensive histograms) · [ ] basic+bearer auth, TLS toggle · [ ] failure→empty+flagged · [ ] `dcgm_provider` delegates to shared parser (no behavior change).

**Phase 3 — inference/GPU adapters:** [ ] `connectors/{dcgm,vllm,triton,ray_serve}.py` · [ ] DCGM fixes (FB_TOTAL, ns units, CLOCKS_EVENT_REASONS, disabled→None) · [ ] vLLM V0/V1 branch · [ ] disabled-metric→None tests · [ ] OTel documented as via-Prometheus (no scraper).

**Phase 4 — Kubernetes:** [ ] `connectors/kubernetes.py` read-only · [ ] `nvidia.com/gpu` capacity/allocatable string→int · [ ] `_continue` pagination · [ ] `deploy/rbac-readonly.yaml` · [ ] no mutating calls anywhere.

**Phase 5 — topology:** [ ] `connectors/topology.py` (topo-m parse by header + optional NVML via `nvidia-ml-py`) · [ ] absent→`None` · [ ] `interconnect_class` derived.

**Phase 6 — simulator:** [ ] `aurelius/simulation/cluster/**` + `fakes/**` · [ ] fakes wired to §6 connectors · [ ] deterministic replay · [ ] realism checks · [ ] `benchmarks/v1/*.yaml` (6 scenarios) · [ ] production connector parses fake output byte-for-byte.

**Phase 7 — classifier:** [ ] `constraints/classifier.py` (8 families) · [ ] missing→`None`/`NONE` · [ ] hysteresis + tie-break · [ ] confidence math · [ ] no fabricated binding constraint · [ ] per-scenario tests.

**Phase 8 — cost/risk/migration:** [ ] `constraints/cost_model.py` · [ ] migration penalty makes marginal/churny migrations net-negative · [ ] additivity with existing objective tested.

**Phase 9 — engine + SLA wiring (CRITICAL):** [ ] `constraints/engine.py` · [ ] `sla_registry` wired into `BacktestEngine`/`JobScheduler` real path · [ ] `constraint_aware`/`sla_config_path` config flags (default OFF) · [ ] **regression: flag-OFF reproduces current numbers bit-for-bit** · [ ] memory-bound emits no KV-internal action (asserted) · [ ] communication-bound emits no NCCL/CUDA action (asserted) · [ ] HARD SLA enforced in real path.

**Phase 10 — CLI reports:** [ ] `cmd_assess`/`cmd_recommend` (dry-run) · [ ] text/JSON/HTML reports with constraint/confidence/SLA/why · [ ] §10.2 metric vector · [ ] no secrets in output.

**Phase 11 — validation/benchmarking:** [ ] frozen `benchmarks/v1/` + scenario-lock CI · [ ] full metadata (scenario/sim/optimizer version, config/SLA/workload-mix hashes) · [ ] controlled-variable runner · [ ] `compare_against_previous` validates metadata identity · [ ] sim numbers flagged `is_sandbox` · [ ] stability/churn checks.

**Phase 12 — hardening:** [ ] connector health checks · [ ] staleness→fail-safe · [ ] confidence everywhere · [ ] no-secret-logging test · [ ] audit logs · [ ] dry-run default verified · [ ] read-only K8s verified.

**Global invariants (must hold at every phase):**
- [ ] Recommendation-only by default; no cluster mutation outside the gated execution path.
- [ ] Missing telemetry fails safe (`KEEP`), never fabricated.
- [ ] Old energy-optimizer behavior preserved exactly when constraint-aware mode is OFF.
- [ ] Sandbox/simulator outputs never used for savings/SLA claims.
- [ ] No KV-cache / allocator / NCCL / CUDA / kernel / runtime / model-execution changes — ever.
- [ ] Sandbox and real connectors share the same Protocol/interface.
- [ ] Every new external fact re-verified against the cited official doc URL at implementation time.

---

*End of Phase 0 plan. This document is a starting point, not a contract; later phases must verify it against the live repo and revise it.*




