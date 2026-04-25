# Aurelius Progress Tracker

## Current Status
- Phase: PHASE_2 → PHASE_3 ready
- Milestone: Phase 2 — Complete ML Forecasting
- Status: MERGED

## Last Run
- Date: 2026-04-25
- Branch: claude/bold-dirac-OmIoy
- PR URL: https://github.com/fnstggl/energy2/pull/7
- PR Status: MERGED (squash)
- Merge Status: MERGED
- Main Commit SHA: 05c0d0ab6b0e63dc4557b21687b6c93e30d6b5a8

## Tests
- Unit: 310 passed, 0 failed (full suite)
- Integration: covered in test_phase2_completion.py (40 new tests)
- E2E: retrain_forecasters CLI smoke test + model promotion test pass
- Result: ALL PASSING

## Phase 2 Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| MAPE < 15% on held-out price series | PASS | 3.01% on 30-day holdout |
| p90 coverage >= 88% on 30-day holdout | PASS | 89.7% with calibration (scale=1.096) |
| Model artifacts versioned on disk | PASS | ModelStore + joblib save/load |
| features.py: build_features(df, weather_df) | DONE | aurelius/forecasting/features.py |
| calibration.py: calibrate_quantile() | DONE | aurelius/forecasting/calibration.py |
| price_model.py: save/load/validate_coverage | DONE | PriceQuantileForecaster |
| carbon_model.py: same | DONE | CarbonQuantileForecaster |
| Bias correction from forecast_corrections_v1.json | DONE | _load_corrections() on __init__ |
| train_offline.py: LightGBM for savings + risk | DONE | trainers.py lgbm variants |
| scripts/retrain_forecasters.py --start/--end | DONE | scripts/ + aurelius/scripts/ |

## What Was Completed This Run

### New Files
- aurelius/forecasting/features.py — build_features(df, weather_df=None) pandas interface
- aurelius/forecasting/calibration.py — calibrate_quantile() binary-search calibration
- scripts/retrain_forecasters.py — Full retraining CLI with --start/--end
- aurelius/scripts/__init__.py + aurelius/scripts/retrain_forecasters.py — module invocation
- tests/test_phase2_completion.py — 40 new Phase 2 tests

### Modified Files
- aurelius/forecasting/price_model.py — save/load, validate_coverage, bias correction
- aurelius/forecasting/carbon_model.py — same
- aurelius/ml/trainers.py — LightGBM savings and risk models with fallback
- aurelius/ml/train_offline.py — LightGBM by default, --min-records guard, --no-lgbm flag

## What Remains for Phase 2
NONE — all Phase 2 acceptance criteria are met.

## Phase 3 Next Steps
Phase 3: Production-Like Simulation Environment requires:
1. aurelius/models.py — Add workload_type, gpu_type, gpu_count, sla_penalty_per_hour, data_transfer_gb, pue fields to Job and OptimizationConfig
2. aurelius/ingestion/workload_traces.py — load_workload_csv() returning list[Job]
3. aurelius/simulation/workload_simulator.py — GPU-typed realistic workload generation
4. aurelius/optimization/objective.py — SLA penalty, data transfer, PUE cost terms
5. aurelius/execution/shadow_runner.py — ShadowRunner.run() recording decisions vs realized prices
6. docker/Dockerfile + docker/docker-compose.yml — multi-stage build, non-root, postgres
7. .github/workflows/ci.yml — ruff, mypy, pytest, Docker build

## Known Risks
- p90 coverage >= 88% requires calibration step after training (raw coverage ~65% on synthetic data)
- Real price/carbon data requires API keys — CSV fallback provided
- LightGBM falls back to bucketed stats when < 50 labelled records (correct behavior)
- Phase 3 Docker/CI is a substantial lift requiring careful dependency management

## Next Task
Start Phase 3 sprint. Exact scope:
- Audit aurelius/models.py for existing Job/OptimizationConfig fields
- Add workload_type, gpu_type, gpu_count, sla_penalty_per_hour, data_transfer_gb, pue to Job
- Create aurelius/ingestion/workload_traces.py with load_workload_csv()
- Create aurelius/simulation/workload_simulator.py with GPU/workload profiles
- Modify aurelius/optimization/objective.py to include SLA/PUE/transfer costs
- Create aurelius/execution/shadow_runner.py with ShadowRunner
- Create docker/Dockerfile (multi-stage, Python 3.11, non-root)
- Create docker/docker-compose.yml (aurelius-api + postgres)
- Create .github/workflows/ci.yml (ruff + mypy + pytest + docker build)
- All with full unit + integration + E2E test coverage
