# Aurelius Progress Tracker

## Current Status
- Phase: PHASE_4 → PHASE_5 ready
- Milestone: Phase 4 — Reporting and Pilot Readiness
- Status: MERGED

## Last Run
- Date: 2026-04-25
- Branch: claude/bold-dirac-ZZwHC
- PR URL: https://github.com/fnstggl/energy2/pull/9
- PR Status: MERGED (squash)
- Merge Status: MERGED
- Main Commit SHA: fbef7095f4bbc32ab610bfaf53d12f7d2e771988

## Tests
- Unit: 432 passed, 0 failed (full suite)
- Phase 4 new tests: 60 (tests/test_phase4_reporting.py)
- Pre-existing tests: 372 (all still pass)
- Skipped: 7 (live API tests requiring credentials)
- Result: ALL PASSING

## Phase 4 Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| SavingsReport.generate(backtest_result) → dict with cost savings, carbon reduction, latency violations, utilization, queue delay | DONE | aurelius/reporting/savings_report.py |
| All metrics include lower/upper CI bounds | DONE | 95% bootstrap CI via _bootstrap_ci() |
| Methodology section explains leakage-free computation | DONE | SavingsReport._methodology_section() |
| render_html_report(savings_report) → str using Jinja2 | DONE | aurelius/reporting/html_report.py |
| Produces self-contained HTML with charts (matplotlib embedded as base64) | DONE | _chart_savings_by_fold() + _chart_baseline_comparison() |
| GET /simulations already implemented | DONE | aurelius/api/app.py (pre-existing) |
| GET /simulations/{run_id} already implemented | DONE | aurelius/api/app.py (pre-existing) |
| API key auth middleware (X-API-Key header) | DONE | require_api_key() dependency |
| Unauthenticated requests return 401 | DONE | Tested in TestAPIAuthMiddleware |
| /health works without auth | DONE | No dependency on require_api_key |
| DataLeakageError and assert_no_leakage() | DONE | aurelius/validation/leakage_audit.py (pre-existing, re-tested) |

## What Was Completed This Run

### New Files
- aurelius/reporting/__init__.py — exports SavingsReport, ConfidenceInterval, render_html_report
- aurelius/reporting/savings_report.py — SavingsReport.generate() with bootstrap CI, all required metrics
- aurelius/reporting/html_report.py — render_html_report() with Jinja2 + embedded base64 matplotlib charts
- tests/test_phase4_reporting.py — 60 adversarial tests (unit + integration + API auth)

### Modified Files
- aurelius/api/app.py — Added require_api_key() FastAPI dependency for all endpoints except /health
- aurelius/pyproject.toml — Added jinja2>=3.1.0, matplotlib>=3.7.0 to core; httpx>=0.24.0 to dev
- aurelius/requirements.txt — Added jinja2 and matplotlib as core deps

## Adversarial Review Findings and Fixes

| Issue Found | Fix Applied |
|-------------|-------------|
| `from datetime import timedelta` was a local import inside _latency_violations | Moved to module-level |
| jinja2, matplotlib, httpx missing from dependency files | Added to pyproject.toml and requirements.txt |
| Jinja2 autoescape=False with baseline name strings in template | Changed to autoescape=True |
| 6 ruff lint errors (import ordering, unused imports) in new files | Auto-fixed with `ruff --fix` |

## Known Risks for Phase 5

- Docker build not verified locally (no daemon) — CI validates on GitHub Actions
- Bootstrap CI with 1 fold produces degenerate interval lower=upper=estimate (correct, tested)
- HTML report with 1000+ folds will have very long table (usable, not a correctness issue)
- AURELIUS_API_KEY unset → API is open with warning log (correct dev/test behavior)
- No learning loop data yet — PostExecutionRecorder records not wired to forecast corrections
- No drift detector yet
- No daily retraining cron script yet
- forecast_corrections_v1.json bias correction artifact not yet populated from real runs

## What Remains for Phase 4
NONE — all Phase 4 acceptance criteria are met.

## Phase 5 Next Steps
Phase 5: Learning Loop / Data Moat requires:
1. aurelius/simulation/replay.py — Wire PostExecutionRecorder.record() for every decision
2. aurelius/forecasting/price_model.py — Load forecast_corrections_v1.json on init and apply bias correction
3. aurelius/ml/train_offline.py — Add --min-records N guard to prevent noisy models
4. scripts/learning_loop_cron.sh — Fully automated daily loop: pull data → append → train → validate → swap if improved
5. aurelius/execution/post_execution.py — Add realized_energy_price lookup from real grid APIs
6. aurelius/monitoring/drift_detector.py — DriftDetector.check() flagging when error exceeds 2× baseline

Acceptance criteria for Phase 5:
- After 30 days of dry-run shadow mode, forecast_corrections_v1.json shows non-zero bias estimates
- Retraining with corrections reduces p50 MAPE by ≥ 5% on next 7-day holdout
- Every backtest/simulation/dry-run records run_id, job_id, forecast snapshot, realized price/carbon, savings
- DriftDetector alerts when model needs retraining

## Next Task
Start Phase 5 sprint. Exact scope:
1. Wire PostExecutionRecorder into replay.py and shadow_runner.py
2. Create aurelius/monitoring/drift_detector.py with DriftDetector.check()
3. Add bias correction loading to price_model.py and carbon_model.py
4. Add --min-records guard to train_offline.py
5. Create scripts/learning_loop_cron.sh
6. Update post_execution.py with realized price lookup
7. Write unit + integration tests for all Phase 5 components
