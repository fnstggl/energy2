# Aurelius Full System Audit

**Date:** 2026-05-23
**Auditor:** Autonomous senior engineering auditor (Claude Code)
**Baseline main SHA at audit start:** `bb455ad3b6d27d89a2267f9fb58f13845306921d`
**Open PRs:** none (`gh pr list` equivalent via GitHub MCP returned `[]`)

This audit is intentionally skeptical. Where a claim is partial, it says
partial. Where code drift broke a documented command, it says so and the bug
was fixed (per the audit's bug-fix rule). It does not flatter the repo.

---

## TL;DR verdict

- **Sell a Tier 1 pilot now?** Conditionally yes — Tier 1 region/time
  optimization is the most solid part (real data, leakage-free backtest, 1217
  passing tests). But sell it on the *shadow-validation* promise, not on a fixed
  savings number, because the headline "25% mean" is not reproduced by the
  default/quick tooling and the shadow ML path was silently broken until this
  audit.
- **Execute a Tier 1 pilot now?** Yes, after this audit's fixes — the
  `shadow run → realize → report` workflow now actually runs end-to-end with a
  customer trace and the ML forecaster (it did not before — three drift bugs).
- **Live production routing now?** No. The safety gate is not wired into the
  decision path, and no live execution/scheduler-adapter path is exercised.
- **Claim a "continuous learning data moat" now?** No. Storage to collect the
  data now exists (this audit added it), but realized customer outcomes do not
  yet drive model selection. It is a data-collection backbone, not a closed loop.

---

## Phase 0 — Repo state

| Item | Result |
|------|--------|
| Branch | audited on `main` @ `bb455ad`, fixes on `claude/nice-davinci-8FS7d` |
| Open PRs | none to merge |
| Packaging | `aurelius/pyproject.toml` + `aurelius/requirements.txt` (note: packaging files live under `aurelius/`, not repo root) |
| Docs reviewed | AURELIUS_PROGRESS.md, PILOT_READINESS_AUDIT.md, ROI_METHODOLOGY.md, local_prod_like_run.md, README.md, docker/* , .env.example |
| Stale doc finding | PILOT_READINESS_AUDIT.md claims **"917 tests passing"**; actual is **1217 passed, 1 skipped** (live excluded). Test-count claim is stale (understated). |

---

## Phase 1 — Claims audit

Legend: **PASS** (implemented, tested, reproducible) · **PARTIAL** (works but
not production-complete) · **FIXTURE_ONLY** · **BLOCKED** (needs
credential/customer infra) · **FALSE_OR_STALE**.

| # | Claim | Status | Evidence / note |
|---|-------|--------|-----------------|
| 1 | Tier 1 region/time optimization | **PASS** | 1217 tests pass; real CAISO/PJM/ERCOT data; leakage-free backtest verified |
| 2 | Tier 2 queue-aware optimization | **FIXTURE_ONLY** | `--queue-file` path + fixtures; no live queue ingestion writer |
| 3 | Tier 3 DCGM/GPU telemetry | **FIXTURE_ONLY** | `.prom` fixtures + `--gpu-file`; needs customer Prometheus |
| 4 | Customer workload trace ingestion | **PASS** (backtest) / **was BROKEN** (shadow) | `JobLogIngester.load_from_csv` works; `backtest --jobs-file` works. **`shadow run --jobs-file` crashed** (`JobLogIngester(regions=)`) — FIXED this audit |
| 5 | Shadow mode run→realize→report | **PARTIAL → PASS after fixes** | 3 drift bugs (see Phase 5) silently degraded or crashed it; now runs end-to-end with ML forecaster + customer trace |
| 6 | ROI calculator | **PASS** | `python -m aurelius.cli roi --monthly-cost 1000000` works; honest caveats; no 60% claim in output |
| 7 | Benchmark harness | **PASS (harness)** | `run_benchmark.py --quick` runs; **but `--quick` uses `seasonal_naive` and returns −4.3% mean — it is a harness smoke test, NOT a validation of the 25% figure** |
| 8 | Oracle diagnostics | **PASS** | `--oracle` flag; documented ceilings; correctly labeled "never present as real savings" |
| 9 | Safety gate | **PARTIAL** | Unit-tested (`tests/test_safety_gate.py`); **NOT wired into shadow/backtest decision path**; inline `_run_validation()` in `quantile_gate.py` is stale (asserts "missing forecast passes" but code fails-closed) |
| 10 | Leakage audit | **PASS** | `tests/test_leakage_audit.py` 8/8 pass; strict `< decision_time` train guards verified in engine + shadow runner |
| 11 | Market data provider admissibility | **PASS** | Sandbox (Electricity Maps) hard-blocked from benchmark claims; tested |
| 12 | Postgres persistence | **PASS** | `TimeSeriesStore` (SQLAlchemy) Postgres/SQLite/no-op; `DATABASE_URL=sqlite:... row_counts()` works |
| 13 | Daily learning loop | **PARTIAL → runs after fixes** | **Was fully broken** against current `BacktestEngine` API (constructor kwargs `price_df/regions/n_folds/forecaster_cls` removed; `run()` returns a list). FIXED. Even fixed, its internal eval gives ~1% and smoke gives −6.4% — *not* the 25% headline; promotion logic is unsound (see Phase 2) |
| 14 | API service | **PASS (smoke)** | FastAPI app importable; auth via `AURELIUS_API_KEY` (not load-tested here) |
| 15 | Docker / docker-compose | **PARTIAL** | Dockerfile + compose present; not built/run in this audit |
| 16 | CI / lint / mypy / pytest | **PASS (lint+tests)** | `ruff check` clean; 1217 tests pass. mypy not run here |
| 17 | ENTSO-E status | **BLOCKED** | `entsoe.py` provider exists; no API token → EU prices unavailable |
| 18 | SOC2 / security / procurement docs | **PARTIAL** | `enterprisedocs/security-and-deployment.md` exists — *documentation*, not SOC2 certification |
| 19 | Scheduler adapters K8s / Slurm / Ray / AWS Batch | **PARTIAL + 1 FALSE_OR_STALE** | `kubernetes.py`, `slurm.py`, `aws_batch.py` present (adapters, not live-validated). **No `ray` adapter exists** — any "Ray" claim is FALSE_OR_STALE |
| 20 | No secrets in repo/history | **PASS** | `.env.example` placeholders only; git history scan found no embedded keys. (Live API keys are present in the *session environment* only, never committed) |

---

## Phase 2 — Continuous learning loop audit (the important section)

Files: `scripts/daily_learning_loop.py`, `aurelius/database/`, `aurelius/shadow/`,
`aurelius/forecasting/`, `aurelius/backtesting/`, tests as listed.

**Headline finding:** the learning loop is a **dry-run-quality orchestration
script with an unsound promotion criterion**, not a production learning loop.
Before this audit it did not even run against the current engine API.

Answers to the 22 questions (strict):

1. **Ingest new real market data automatically?** Partial. `fetch_latest_prices`
   pulls last-3-days from CAISO/PJM/ERCOT when creds exist (observed: fetched 72
   CAISO + 73 PJM rows live). Only 3 days/run; relies on cron, not a scheduler.
2. **Append to durable storage safely?** Yes for CSV store (dedup on
   timestamp+region) and now Postgres `energy_prices` (unique constraint).
3. **Deduplicate by timestamp/region/source?** Yes (CSV: ts+region; DB: ts+region+source).
4. **Leakage-free evaluation?** Yes structurally (BacktestEngine strict train split; verified).
5. **Train candidate models?** Yes — `train_candidate_model` fits a forecaster and pickles it.
6. **Compare candidate vs active?** **No, not honestly.** `compare_models`
   compares the *current run's* freshly-trained-in-engine backtest savings against
   a number stored in `active_metadata.json`. The pickled candidate/active model
   artifacts are **never loaded for inference**. It is run-over-run drift of an
   in-engine model, not candidate-vs-active on the same held-out set.
7. **Reject regressions?** Mechanically yes (0.5pp threshold), but on the unsound
   metric in (6), so the rejection is not meaningful.
8. **Promote only if better?** Same caveat — promotion copies the candidate
   `.pkl` to active, but "better" is measured against a stored scalar, not a real comparison.
9. **Persist model artifacts/version metadata?** Yes, to local disk
   `data/models/*.pkl` + `*_metadata.json`. No object storage, no real versioning.
10. **Persist benchmark artifacts?** Yes — JSON/TXT in `benchmarks/results/` and
    optionally `benchmark_runs` table.
11. **Record shadow decisions?** Yes — JSONL always; now also `decision_events`
    (DB) with customer_id/pilot_id (added this audit).
12. **Later compare predicted vs realized savings?** Yes — `shadow realize` +
    `shadow report`; persisted to `realized_outcomes` (added this audit).
13. **Track forecast error over time?** Now partially — the loop's new
    `read_realized_outcomes_summary` computes mean |realized−predicted| pp from the
    store. It is reported, **not** acted upon.
14. **Customer-specific workload-trace learning?** No. Traces are ingested per run
    and not persisted as a learnable entity; models are global, not per-customer.
15. **Customer-specific policy/baseline learning?** No.
16. **Queue/DCGM learning from customer telemetry?** No. Telemetry tables/interface
    exist (added this audit) but no live writer and no learning consumer.
17. **Daily reports a human can inspect?** Yes — `reports/learning_loop/learning_loop_<date>.json`.
18. **Run from cron/systemd/Railway without overlapping?** Partially — a
    `scripts/learning_loop_cron.sh` exists, but **no locking** in the Python loop.
19. **Locking/idempotency?** **No lock.** Idempotency only at the dedup level.
    Concurrent runs could race on the CSV store and `data/models/`.
20. **Rollback if a promoted model later performs worse?** **No.** No rollback path.
21. **Retention policy / audit trail for enterprise buyers?** **No** retention
    policy; audit trail is now better (append-only event tables) but not documented
    as a policy.
22. **What's missing for a real compounding advantage?** Realized, customer-scoped
    outcomes must *drive model selection* (close gap G1 in DATA_MOAT_ARCHITECTURE.md);
    safety-gate decisions must be recorded; locking + rollback; per-customer models.

**Verdict: PARTIAL — data-collection backbone now exists; the loop is not yet a
real self-improving system.** A dry-run script is not a production learning loop,
and that is exactly what this is.

---

## Phase 3 — Runnable command audit

All run from repo root with deps from `aurelius/requirements-dev.txt`.

| Command | Result | Note |
|---------|--------|------|
| `pytest tests/ (live excluded)` | **PASS** | 1217 passed, 1 skipped, 150s |
| `ruff check aurelius/ scripts/ tests/` | **PASS** | All checks passed |
| `run_benchmark.py --quick` | **PASS (harness)** | seasonal_naive, mean −4.3% — smoke only, not a savings claim |
| `cli roi --monthly-cost 1000000` | **PASS** | p50 annualized projection; no 60% claim |
| `shadow run --jobs-file <trace> --forecaster ml_quantile_recovery` | **FAIL (by design)** | `ml_quantile_recovery` is **not a valid shadow `--forecaster` choice** (only `ml_quantile`/`seasonal_naive`). The repo's *best* forecaster is not available in shadow mode. |
| `shadow run --jobs-file <trace> --forecaster ml_quantile` | **was FAIL → PASS after fix** | crashed on `JobLogIngester(regions=)`; then ML path crashed (`predict(recent_context=)`); then tz crash. All FIXED. Also: bundled fixture trace (Jan 14–16) is incompatible with default decision-time + 30-day train (price data starts Jan 1) → "no schedulable jobs" unless `--decision-time 2026-01-13 --train-days 12` |
| `shadow realize` / `shadow report` | **PASS after fix** | full workflow runs; demo numbers noisy (MAPE 67%, one workload realized −69%) due to tiny fixture/short training |
| `DATABASE_URL=sqlite TimeSeriesStore().row_counts()` | **PASS** | returns counts incl. new event tables |
| `daily_learning_loop.py --dry-run` | **was FAIL (exit 1) → PASS (exit 0) after fix** | `BacktestEngine` API drift in eval + smoke; FIXED; now exits 0, eval ~1%, smoke −6.4% |

---

## Phase 4 — Pilot readiness verdict

| Category | Status | Evidence | Blocker | Next action |
|----------|--------|----------|---------|-------------|
| Tier 1 first pilot | **READY (conditioned)** | tests + leakage + shadow workflow (post-fix) | savings are validated only on synthetic/short fixtures end-to-end | run shadow on a real ≥60-day customer trace + matching RT settlement |
| Tier 2 queue-aware pilot | **NOT READY** | fixture-only | no live queue ingestion | wire a K8s/Slurm queue exporter → `telemetry_snapshots` |
| Tier 3 GPU telemetry pilot | **NOT READY** | fixture-only | needs customer DCGM/Prometheus | wire Prometheus reader → `telemetry_snapshots` |
| Continuous learning loop | **PARTIAL** | loop runs post-fix | promotion metric unsound; no rollback/lock | close G1 (outcomes drive selection) |
| Data moat / compounding advantage | **PARTIAL (backbone only)** | new event tables + customer_id | outcomes don't drive learning | close G1; persist gate + features |
| Enterprise procurement readiness | **PARTIAL** | enterprisedocs/ present | no SOC2 cert, no retention policy | retention policy + security review |
| Deployment readiness | **PARTIAL** | Docker/compose/API present | not built/load-tested here; no object storage for artifacts | build image in CI; add artifact store |
| Sales/demo readiness | **READY (with honesty)** | ROI CLI + shadow report | quick benchmark ≠ headline; fixtures didn't work OOTB | ship a working demo fixture+command |
| EU expansion | **BLOCKED** | entsoe.py skeleton | no ENTSO-E token | register token; validate connector |
| Long-term hyperscaler readiness | **NOT READY** | region-level only | no node/topology control, no Ray adapter | out of current scope |

### Final verdict

- **A. Ready to *sell* a first Tier 1 pilot now?** Yes, if sold as a
  shadow-mode validation engagement ("we will prove savings on *your* trace"),
  not as a guaranteed 25%/60%.
- **B. Ready to *execute* a first Tier 1 pilot now?** Yes — but only because this
  audit fixed three shadow-mode drift bugs. Before the fixes, the documented
  pilot command crashed and the ML forecaster silently fell back to naive.
- **C. Ready for live production routing now?** No. Safety gate not wired in; no
  live execution path validated.
- **D. Ready to claim "continuous learning data moat" now?** No. Backbone exists;
  realized outcomes do not yet drive model improvement.
- **E. What must be fixed before claiming each?** B: validate on a real customer
  trace. C: wire + persist the safety gate, validate one scheduler adapter end to
  end. D: close G1 (realized outcomes → model selection), add rollback + locking,
  per-customer models.

---

## Phase 5 — Bugs fixed (all within the allowed "code/doc drift" scope)

One branch, one PR. Fixes only; no new product features, no ML/optimizer changes.

1. **`aurelius/cli.py` `shadow run`** — `JobLogIngester(regions=regions)` →
   `JobLogIngester()` (constructor takes no `regions`; matches the working
   `backtest` path). Unblocks `shadow run --jobs-file`.
2. **`aurelius/shadow/runner.py` `_build_ml_forecast`** — (a) tz crash
   `pd.Timestamp(x, tz="UTC")` on a tz-aware value → use existing `_to_utc_ts`;
   (b) stale `predict(recent_context=, forecast_start=, forecast_end=)` → current
   per-region `predict(region, timestamps, recent_prices)` API. The ML forecaster
   now actually runs in shadow mode instead of silently degrading to naive.
3. **`scripts/daily_learning_loop.py`** — `run_evaluation` and
   `run_benchmark_smoke_test` used the **removed** `BacktestEngine(price_df=,
   regions=, n_folds=, forecaster_cls=…)` constructor and `engine.run(jobs)` →
   `result.savings_vs_current_price`. Rewritten to the current
   `BacktestEngine(method=…, price_forecaster_cls=…)` + `run(jobs, price_df,
   carbon_df)` returning `list[BacktestRound]`, with a `_mean_savings_vs_cpo`
   aggregator. Fixed smoke-test job time-window (jobs now span the full dataset so
   folds are produced). Moved the `aurelius.database` import after the
   `sys.path.insert` so DB persistence isn't dead when run as a script.

**Data-collection storage layer** (per the appended Data Moat work order, with
explicit acceptance criteria — infrastructure, not a product feature):
- New append-only tables `decision_events`, `realized_outcomes`,
  `telemetry_snapshots` (customer_id/pilot_id/run_id, gate columns, data_source_hash).
- New `TimeSeriesStore` methods: `record_decisions`, `get_decisions`,
  `record_realized_outcomes`, `get_realized_outcomes`, `record_telemetry`.
- `shadow run`/`shadow realize` persist to the store when `DATABASE_URL` is set
  (`--customer-id`/`--pilot-id` args); JSONL remains the always-on record.
- Daily loop reads realized outcomes back (`read_realized_outcomes_summary`).
- 19 new SQLite unit tests in `tests/test_database_store.py` (71 total pass).
- `docs/DATA_MOAT_ARCHITECTURE.md` documents the architecture and the open gaps.

**Bugs found but NOT fixed (out of scope — documented for follow-up):**
- Safety gate not integrated into the decision path; `quantile_gate._run_validation`
  inline assertions are stale vs the fail-closed code.
- Learning-loop promotion criterion is unsound (compares in-engine model vs a
  stored scalar; saved artifacts never loaded). Fixing this is a design change.
- `tests/test_daily_learning_loop.py` accepts `status in ("ok","error")` and exit
  code `0 or 1` — i.e. it passed even while the loop was fully broken. False
  confidence; the tests should assert success once the loop is trusted.
- `ml_quantile_recovery` (the repo's best forecaster) is not selectable in shadow mode.

---

## Phase 6 — Final output

1. **Current main SHA:** `bb455ad3b6d27d89a2267f9fb58f13845306921d` (fixes on
   branch `claude/nice-davinci-8FS7d`, PR opened).
2. **Open PR status:** none at audit start; this audit opens one.
3. **Tests:** 1217 passed, 1 skipped (live excluded), then 71/71 DB tests +
   81/81 shadow+loop tests after changes; ruff clean.
4. **Commands:** see Phase 3 — all the prescribed audit commands now run; three
   were broken by drift and were fixed.
5. **Continuous learning loop verdict:** PARTIAL. Orchestration runs; promotion is
   unsound; outcomes don't yet drive learning. Not a real moat yet.
6. **First pilot readiness verdict:** Tier 1 is sellable/executable as a
   shadow-validation engagement (post-fix). Not ready for live routing.
7. **What is actually missing:** (a) outcomes-driven model selection + rollback +
   locking; (b) safety gate wired into and recorded from the decision path;
   (c) a working out-of-the-box shadow demo (fixture + decision-time that match);
   (d) live queue/GPU ingestion writers; (e) object storage for artifacts;
   (f) ENTSO-E token for EU; (g) honest doc refresh (test count, Ray claim).
8. **Top 5 next tasks (priority order):**
   1. Close G1: make the daily loop evaluate the saved candidate against realized
      outcomes / a held-out window and promote on that, with rollback.
   2. Wire the `QuantileSafetyGate` into the shadow/backtest decision path and
      persist `gate_status`/`gate_reason`.
   3. Ship a working shadow demo (fixture trace + `--decision-time` that produce
      decisions OOTB) and add `ml_quantile_recovery` to shadow `--forecaster`.
   4. Tighten `tests/test_daily_learning_loop.py` to assert success (no
      "ok or error"), and add a real candidate-vs-active comparison test.
   5. Validate one scheduler adapter (K8s) + one telemetry ingestion path end to
      end, persisting to `telemetry_snapshots`.
9. **Continue or pause the routine?** Continue — but pause any messaging that
   says "learning loop live" or "data moat complete" until G1 is closed.
10. **Backend / frontend / infra / sales?** **Backend + infra.** The gaps are all
    backend (loop correctness, gate integration, persistence wiring) and infra
    (object storage, deployment, ingestion). No frontend work is warranted now.
    Sales can proceed in parallel on the shadow-validation framing.
