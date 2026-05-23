# Aurelius Progress Tracker

===============================================================================
CURRENT PRODUCT STATUS
===============================================================================

Aurelius currently has a strong infrastructure foundation,
but is NOT yet considered a fully validated production-grade
multi-signal AI infrastructure optimizer.

IMPORTANT:

Historical “MERGED” phases below represent FOUNDATION IMPLEMENTATION HISTORY,
NOT proof of:
- production correctness
- benchmark superiority
- validated economic savings
- production deployment readiness
- customer/pilot readiness
- reproducible real-world optimization superiority

All systems remain subject to:
- adversarial validation
- leakage audits
- benchmark comparison
- regression testing
- production-similar evaluation
- real workload validation
- safety validation
- reproducibility verification

The ACTIVE roadmap is now the:
PRODUCTION MAXIMUM-SAVINGS ROADMAP.

The original implementation phases established:
- real data ingestion foundations
- leakage-free backtesting foundations
- ML forecasting foundations
- shadow execution infrastructure
- reporting systems
- learning loop infrastructure

However, the current system still likely lacks:
- fully validated superiority over current_price_only
- production-scale benchmark validation
- weather-aware optimization
- cooling/PUE-aware optimization
- queue-aware optimization
- production-scale GPU telemetry integration
- standardized benchmark harness
- validated multi-region anti-correlation gains
- large-scale migration validation
- long-duration shadow validation
- reproducible pilot-grade economic validation

The current mission is:

Transform Aurelius from:
“working infrastructure foundation”
into:
“production-grade multi-signal infrastructure optimizer.”

===============================================================================
STRATEGIC ICP
===============================================================================

PRIMARY ICP:
- neoclouds
- GPU cloud operators
- inference providers
- enterprise GPU fleets
- HPC clusters
- self-hosted AI infrastructure operators
- data-center operators

SECONDARY / LONG-TERM ICP:
- hyperscalers
- foundation model companies only when infrastructure-level scheduling
  control exists

Aurelius should prioritize customers who expose:
- electricity cost
- GPU availability
- queue depth
- cooling/PUE
- workload placement
- migration/checkpointing
- utilization
- SLA constraints

===============================================================================
PRIMARY PRODUCT OBJECTIVE
===============================================================================

Build a production-grade multi-signal AI infrastructure optimizer that:
- minimizes real-world compute cost
- preserves SLA/safety/latency correctness
- improves utilization efficiency
- safely optimizes across geography and time

Primary benchmark:
- current_price_only

Aurelius must optimize against strong realistic baselines,
NOT weak strawman baselines.

60% savings is an aspirational stretch target,
NOT a current claim.

No savings claim is valid unless proven using:
- leakage-free backtesting
- real historical data
- production-similar workloads
- identical information constraints across baselines
- reproducible benchmark outputs

===============================================================================
FOUNDATION IMPLEMENTATION HISTORY
===============================================================================

These phases represent historical implementation milestones only.

They do NOT imply:
- maximum savings achieved
- production deployment readiness
- pilot validation complete
- benchmark superiority proven
- infrastructure intelligence completeness

-------------------------------------------------------------------------------
Foundation Phase 1 — Real Data + Leakage-Free Backtesting
-------------------------------------------------------------------------------

Status: MERGED

Implemented:
- real data ingestion foundations
- leakage-free walk-forward foundations
- baseline comparison infrastructure
- historical storage foundations

-------------------------------------------------------------------------------
Foundation Phase 2 — Real ML Forecasting
-------------------------------------------------------------------------------

Status: MERGED

Implemented:
- LightGBM forecasting foundations
- quantile forecasting
- calibration infrastructure
- artifact versioning
- holdout evaluation foundations

-------------------------------------------------------------------------------
Foundation Phase 3 — Production-Like Shadow Environment
-------------------------------------------------------------------------------

Status: MERGED

Implemented:
- shadow execution infrastructure
- workload simulation foundations
- dry-run adapters
- Docker/CI foundations
- reporting infrastructure

-------------------------------------------------------------------------------
Foundation Phase 4 — Reporting & Pilot Readiness
-------------------------------------------------------------------------------

Status: MERGED

Implemented:
- savings reporting
- confidence interval foundations
- methodology reporting
- API/reporting foundations

-------------------------------------------------------------------------------
Foundation Phase 5 — Learning Loop / Data Moat
-------------------------------------------------------------------------------

Status: MERGED

Implemented:
- post-execution recording
- drift detection
- retraining loop foundations
- artifact correction infrastructure
- benchmark recording foundations

===============================================================================
LAST VERIFIED IMPLEMENTATION RUN
===============================================================================

Date:
2026-04-25

Branch:
claude/bold-dirac-YjQXK

PR URL:
https://github.com/fnstggl/energy2/pull/10

PR Status:
MERGED (squash)

Merge Status:
MERGED

Main Commit SHA:
4c944a6d9b397355d6e7664a5dfec0bd0e7e3cb8

===============================================================================
LAST VERIFIED TEST STATUS
===============================================================================

Unit:
483 passed, 0 failed

Phase 5 tests:
53

Pre-existing tests:
430

Skipped:
7 live API tests requiring credentials

Result:
ALL PASSING

IMPORTANT:
Passing tests do NOT guarantee:
- production correctness
- benchmark superiority
- economic reproducibility
- production-scale robustness

===============================================================================
KNOWN RISKS / REMAINING GAPS
===============================================================================

Known likely gaps include:
- no production-scale benchmark harness yet
- no standardized oracle diagnostic suite
- no validated superiority over current_price_only
- no production-scale weather-aware optimization
- no production-scale queue-aware optimization
- no production-scale DCGM integration
- no long-duration shadow validation
- no production-scale workload traces
- no proven anti-correlation strategy across many regions
- no verified large-scale migration economics
- JSONL growth unbounded
- limited EU validation
- benchmark regression tracking incomplete

===============================================================================
ACTIVE PRODUCTION MAXIMUM-SAVINGS ROADMAP
===============================================================================

-------------------------------------------------------------------------------
ROADMAP PHASE 1 — BENCHMARK & PILOT VALIDATION
-------------------------------------------------------------------------------

Goal:
Build trustworthy, reproducible, leakage-free benchmark infrastructure.

Required:
- benchmark harness
- workload matrix
- region matrix
- horizon matrix
- oracle diagnostics
- regression detection
- benchmark reproducibility
- benchmark archival

Required workload types:
- training
- fine_tuning
- llm_batch_inference
- data_processing
- scheduled_batch
- realtime_inference
- background_maintenance

Required forecast horizons:
- 24h
- 36h
- 48h
- 72h
- 168h where valid

Required benchmark outputs:
- savings vs current_price_only
- savings vs all baselines
- carbon impact
- SLA violations
- migration count
- downside events
- confidence intervals
- regression vs previous run

Required benchmark folder:

benchmarks/
  workload_matrix.yaml
  region_matrix.yaml
  baseline_matrix.yaml
  benchmark_config.yaml
  run_all_workloads.sh
  run_all_regions.sh
  run_oracle_diagnostics.sh
  compare_against_previous.py

The benchmark system must fail if:
- future leakage detected
- synthetic data used for real claims
- current_price_only missing
- constraint violations occur
- savings regress unexpectedly

-------------------------------------------------------------------------------
ROADMAP PHASE 2 — MULTI-REGION EXPANSION
-------------------------------------------------------------------------------

Expand and validate:
- CAISO
- PJM
- ERCOT
- NYISO
- MISO
- SPP
- ISO-NE
- ENTSO-E

Goal:
Determine whether savings bottlenecks are:
- forecasting
- region correlation
- migration overhead
- workload inflexibility
- optimizer weakness

-------------------------------------------------------------------------------
ROADMAP PHASE 3 — WEATHER & COOLING INTELLIGENCE
-------------------------------------------------------------------------------

Integrate:
- Open-Meteo
- NOAA
- Meteostat or equivalent

Weather should estimate:
- cooling load
- heat-wave risk
- PUE penalty
- future grid stress
- future price spikes

GPU temperature is NOT a substitute for weather.

DCGM gives current GPU state.
Weather predicts future cooling/grid conditions.

-------------------------------------------------------------------------------
ROADMAP PHASE 4 — GPU TELEMETRY & DCGM
-------------------------------------------------------------------------------

Integrate:
- NVIDIA DCGM
- dcgm-exporter
- Prometheus

Required metrics:
- GPU utilization
- memory utilization
- GPU power draw
- GPU temperature
- thermal throttling
- NVLink throughput
- PCIe throughput
- ECC errors
- GPU health

-------------------------------------------------------------------------------
ROADMAP PHASE 5 — QUEUE-AWARE OPTIMIZATION
-------------------------------------------------------------------------------

Integrate:
- Kubernetes
- Slurm
- Ray
- AWS Batch
- CSV fallback

Use:
- queue depth
- wait time
- available GPUs
- preemption risk
- SLA deadlines
- job priority

-------------------------------------------------------------------------------
ROADMAP PHASE 6 — FULL MULTI-SIGNAL OPTIMIZER
-------------------------------------------------------------------------------

Build optimizer modes:
- no migration
- delay scheduling
- region switching
- chunked migration
- checkpoint-aware migration
- queue-aware placement
- weather-aware placement
- carbon-constrained placement
- oracle diagnostic mode
- full multi-signal optimizer

Optimization objective should include:
- electricity cost
- cooling/PUE cost
- queue delay cost
- SLA penalties
- migration/checkpointing cost
- carbon cost
- utilization inefficiency
- risk penalties
- data transfer cost

-------------------------------------------------------------------------------
ROADMAP PHASE 7 — PRODUCTION SHADOW VALIDATION
-------------------------------------------------------------------------------

Run production-similar dry-run shadow mode using:
- real/latest market data
- realistic workload traces
- real scheduling constraints

Record:
- forecast snapshots
- realized prices
- realized savings
- realized downside
- SLA outcomes
- migration overhead

-------------------------------------------------------------------------------
ROADMAP PHASE 8 — CONTINUOUS LEARNING & SELF-IMPROVEMENT
-------------------------------------------------------------------------------

Every run should:
1. ingest new data
2. retrain candidate models
3. benchmark candidate models
4. compare against active models
5. reject regressions
6. run leakage audits
7. generate reports
8. update progress tracker

===============================================================================
API-NEEDED REQUIREMENT
===============================================================================

Maintain:

API-NEEDED/

Whenever external providers/APIs are required.

Examples:
- API-NEEDED/EIA.md
- API-NEEDED/PJM.md
- API-NEEDED/CAISO.md
- API-NEEDED/ERCOT.md
- API-NEEDED/ELECTRICITYMAPS.md
- API-NEEDED/WATTTIME.md
- API-NEEDED/OPEN_METEO.md
- API-NEEDED/PROMETHEUS_DCGM.md

===============================================================================
LAST VERIFIED IMPLEMENTATION RUN (UPDATED)
===============================================================================

Date:
2026-05-23

Branch:
claude/youthful-feynman-DfckO

PR URL:
(pending — see push)

PR Status:
IN PROGRESS

Main Commit SHA:
(see git log)

===============================================================================
LAST VERIFIED TEST STATUS (UPDATED)
===============================================================================

Unit + integration:
750 passed, 0 failed (4 skipped), 200 warnings

New tests (Phase 3):
23 new tests in tests/test_weather_features.py:
  - TestBuildWeatherLookup (5 tests): lookup construction, multi-region, NaN→0
  - TestBuildFeatureMatrixWithWeather (5 tests): columns, backward compat
  - TestBuildFeatureMatrixForPredictWithWeather (2 tests): passthrough, fallback
  - TestPriceQuantileForecasterBackwardCompat (2 tests): no-weather v2.0
  - TestPriceQuantileForecasterWithWeather (6 tests): v3.0, determinism
  - TestWeatherLeakageSafety (2 tests): train<eval_start, missing→empty
  - TestBacktestEngineWeatherIntegration (1 test): end-to-end smoke

Pre-existing tests:
701 (all Phase 1-2 benchmark, migration, spread_risk, ML forecaster, etc.)

Skipped:
4 live API tests requiring credentials

Result:
ALL PASSING (724 tests)

===============================================================================
FIRST OFFICIAL BENCHMARK RESULTS
===============================================================================

Run date: 2026-05-22
Data: CAISO Q1 2026 DA + PJM Q1 2026 DA + combined DA-plan/RT-settle
Optimizer: greedy_migrate, seasonal_naive forecaster
Train window: 30 days | Eval window: 7 days
All 21 cells: 0% missing price hours

Savings vs current_price_only (THE honest benchmark):

  background_maintenance @ caiso_pjm_da_rt:   55.9%
  background_maintenance @ us-west-only:       58.7%
  background_maintenance @ us-east-only:       30.2%
  scheduled_batch @ caiso_pjm_da_rt:           38.0%
  data_processing @ caiso_pjm_da_rt:           36.5%
  llm_batch_inference @ caiso_pjm_da_rt:       31.9%
  llm_batch_inference @ us-west-only:          24.6%
  llm_batch_inference @ us-east-only:          22.2%
  fine_tuning @ caiso_pjm_da_rt:               17.5%
  training @ us-west-only:                      3.6%
  training @ us-east-only:                      3.2%
  training @ caiso_pjm_da_rt:                   0.8%  ← known bottleneck
  fine_tuning @ us-east-only:                  -1.2%  ← optimizer below cpo (known)
  realtime_inference @ caiso_pjm_da_rt:         4.5%

  Mean across all 21 cells: 21.6%

IMPORTANT — interpretation:
- These numbers are VALID (leakage-free, real data, 0% missing hours)
- 60% savings is an aspirational stretch target — not yet proven at scale
- Negative/near-zero for training@caiso_pjm_da_rt is a known bottleneck:
  the seasonal_naive forecaster + greedy_migrate doesn't fully capture
  long-horizon price valleys for training workloads
- fine_tuning@us-east-only -1.2% is within noise (50 jobs, 5 folds)
  but warrants investigation in next phase

===============================================================================
CURRENT ACTIVE MILESTONE
===============================================================================

ROADMAP PHASE 2 — ML QUANTILE FORECASTER + MULTI-SIGNAL OPTIMIZATION

Status: COMPLETE (2026-05-22)

Completed tasks:
✓ 1. ML forecaster v2.0 with volatility regime features (spike_flag, momentum, std)
✓ 2. Fixed predict-time forward-fill bug (zero-fill → forward-fill from last known price)
✓ 3. Improved hyperparameters: 200 estimators, LR=0.05, num_leaves=63
✓ 4. Extended context window: 192h → 336h for better lag_168h coverage
✓ 5. WattTime carbon data fetched: Q1 2026 and Summer 2025 (CAISO only on free plan)
✓ 6. Carbon auto-detection in benchmark runner (co-located CSV files)
✓ 7. fetch_watttime_carbon.py script added
✓ 8. Head-to-head benchmark run: seasonal_naive vs ml_quantile v2 (7 workloads × 3-region)
✓ 9. Adversarial audit: zero-fill artifact found and fixed before publishing results
✓ 10. ML benchmark archived: benchmarks/results/benchmark_ml_quantile_v2_3region_q12026_20260522.json
✓ 11. 25 new tests added (all passing)
✓ 12. 692 total tests passing

PRIOR PHASE COMPLETE:
ROADMAP PHASE 1 — BENCHMARK & PILOT VALIDATION: COMPLETE
✓ 1. Audit benchmark correctness (adversarial review passed)
✓ 2. Create standardized benchmark harness (benchmarks/ directory)
✓ 3. Add oracle diagnostics (run_oracle_diagnostics.sh + --oracle flag)
✓ 4. Add API-NEEDED documentation (3 providers still needed, 4 credentials confirmed)
✓ 5. Run baseline benchmark suite (21 cells, 0% missing)
✓ 6. Save benchmark outputs for regression comparison
✓ 7. Benchmark smoke test integrated into CI (GitHub Actions: benchmark-smoke job)
✓ 8. ERCOT as third region added (data/q12026_3region_dam.csv + rt.csv)
✓ 9. Summer2025 3-region combo added (data/summer2025/3region_dam.csv + rt.csv)
✓ 10. Oracle diagnostics run — forecasting bottleneck quantified
✓ 11. ML quantile forecaster option wired into benchmark runner (--forecaster ml_quantile)
✓ 12. All 667 tests passing

===============================================================================
PHASE 1 FINAL BENCHMARK RESULTS
===============================================================================

Run date: 2026-05-22

--- Q1 2026 results (caiso_pjm_da_rt, 2-region) ---

  background_maintenance:  55.9%
  scheduled_batch:         38.0%
  data_processing:         36.5%
  llm_batch_inference:     31.9%
  fine_tuning:             17.5%
  realtime_inference:       4.5%
  training:                 0.8%  ← known bottleneck (forecasting gap confirmed)
  Mean (21 cells total):   21.6%

--- NEW: Q1 2026 results (caiso_pjm_ercot_da_rt, 3-region) ---

  background_maintenance:  36.3%
  data_processing:         24.5%
  llm_batch_inference:     21.9%
  scheduled_batch:         18.6%
  fine_tuning:             11.2%
  realtime_inference:       8.9%
  training:                 3.3%
  Mean (7 cells):          17.8%

--- NEW: ERCOT standalone (us-south-only, Q1 2026) ---

  data_processing:         33.0%
  background_maintenance:  33.5%
  scheduled_batch:         29.4%
  llm_batch_inference:     27.2%
  fine_tuning:             10.0%
  realtime_inference:       5.5%
  training:                 2.7%
  Mean (7 cells):          20.2%

--- NEW: Summer 2025 results (3-region, Jun-Aug 2025) ---

  data_processing:         31.9%
  llm_batch_inference:     29.8%
  fine_tuning:             28.8%
  scheduled_batch:         26.4%
  background_maintenance:  25.2%
  training:                16.2%  ← summer season dramatically better than winter
  realtime_inference:       1.4%
  Mean (7 cells):          22.8%

IMPORTANT — interpretation:
- All results are leakage-free, real data, 0% missing price hours
- 60% savings is an aspirational stretch target — not yet proven at scale
- Summer2025 training (16.2%) >> Q1 2026 training (3.3%) — ERCOT winter spikes
  are the structural challenge for seasonal_naive forecasting
- 3-region combos show lower savings vs 2-region for some workloads — the
  seasonal_naive forecaster struggles with 3-way anti-correlation optimization

===============================================================================
ORACLE DIAGNOSTIC FINDINGS (2026-05-22)
===============================================================================

Q1 2026 / caiso_pjm_ercot_da_rt:
  training:      seasonal_naive 3.3%   oracle_ceiling 37.7%   gap 34.4pp
  fine_tuning:   seasonal_naive 11.2%  oracle_ceiling 61.4%   gap 50.2pp
  llm_batch:     seasonal_naive 21.9%  oracle_ceiling 59.0%   gap 37.1pp

Summer 2025 / summer2025_3region:
  training:      seasonal_naive 16.2%  oracle_ceiling 25.8%   gap  9.6pp
  fine_tuning:   seasonal_naive 28.8%  oracle_ceiling 39.5%   gap 10.7pp
  llm_batch:     seasonal_naive 29.8%  oracle_ceiling 33.7%   gap  3.9pp

Key conclusion:
- ML forecasting gap is LARGE for winter Q1 data (34-50pp for training/fine_tuning)
- ML forecasting gap is SMALL for summer data (4-11pp)
- Winter ERCOT volatility drives the forecasting bottleneck
- ML quantile forecaster is the HIGHEST-LEVERAGE next milestone
- Summer season validates that structural savings are real when forecasting is adequate

Quick mode (10d train) note:
- Quick mode with ERCOT winter data shows -17% for training (expected — short window)
- Quick mode is NOT valid for savings claims (documented behavior)
- Full mode (30d train) gives correct results

===============================================================================
ROADMAP PHASE 2 — ML QUANTILE FORECASTER + MULTI-SIGNAL OPTIMIZATION
===============================================================================

Status: COMPLETE (2026-05-22)

Acceptance criterion MET:
  ml_quantile v2 > seasonal_naive by ≥5pp for training@caiso_pjm_ercot_da_rt
  Actual result: +11.7pp (3.3% → 15.0%) ✓

What was implemented:
  1. ML forecaster v2.0:
     - Volatility regime features (rolling_std_24h/168h, volatility_ratio_24h,
       spike_flag, price_momentum_6h/24h) for ERCOT winter spike detection
     - Fixed predict-time feature computation: forward-fill from last known price
       instead of zero-fill (zero-fill corrupted momentum/spike features)
     - Improved hyperparameters: 200 estimators, LR=0.05, num_leaves=63
     - Context window extended: 192h → 336h for full lag_168h coverage
  
  2. WattTime carbon integration:
     - Fetched Q1 2026 and Summer 2025 MOER data for CAISO (us-west)
     - Carbon data auto-detected from co-located CSV files in benchmark runner
     - Carbon signal used in optimizer objective (beta=0.3 weight)
     - Limitation: WattTime free plan only covers CAISO_NP15/CAISO_NORTH;
       PJM and ERCOT require a paid plan (documented in API-NEEDED)
     - data/watttime_carbon_q12026.csv: 1571 hourly rows (us-west, Q1 2026)
     - data/summer2025/watttime_carbon_summer2025.csv: 1681 hourly rows
  
  3. Script: scripts/fetch_watttime_carbon.py
  
  4. Benchmark runner improvements:
     - --carbon-file option added
     - Auto-detection of co-located carbon CSVs
     - carbon_regions in result dict for auditability

Head-to-head benchmark results (Q1 2026, 3-region, 5 folds):

  Workload                 | seasonal_naive | ml_quantile v2 | delta
  -------------------------|----------------|----------------|--------
  training                 |  3.3%         | 15.0%          | +11.7pp ✓
  fine_tuning              | 11.2%         | 13.4%          | +2.2pp
  llm_batch_inference      | 21.9%         | 33.6%          | +11.7pp
  data_processing          | 24.5%         | 37.7%          | +13.2pp
  scheduled_batch          | 18.6%         | 25.3%          | +6.7pp
  background_maintenance   | 36.3%         | 40.3%          | +4.0pp
  realtime_inference        |  8.9%         | 10.0%          | +1.1pp
  Mean                     | 17.8%         | 25.0%          | +7.2pp

Benchmark artifact: benchmarks/results/benchmark_ml_quantile_v2_3region_q12026_20260522.json

Adversarial findings fixed during implementation:
  - Zero-fill bug: predict-time volatility features used np.zeros(n_predict)
    for the future window, creating artificial "prices drop to $0" momentum
    signals that inflated savings to 53.3% (leakage-adjacent artifact, NOT
    true leakage). Fixed to forward-fill from last_known_price. Post-fix
    results: 15.0% training (honest, verified).
  - The 53.3% result was explicitly NOT saved as a benchmark artifact.

Oracle gap analysis (post-fix):
  training@3region:    ml_v2 15.0% vs oracle 37.7% → 22.7pp remaining
  fine_tuning@3region: ml_v2 13.4% vs oracle 61.4% → 48pp remaining
  Remaining gap for training: likely needs weather/temperature features
  Remaining gap for fine_tuning: likely needs weather + longer delay windows

Forecast quality (ml_quantile v2 vs v1):
  - MAPE: 3.2% (v1) → 8.3% (v2) [point accuracy slightly worse]
  - p90_coverage: 0.29 (v1) → 0.67 (v2) [calibration dramatically better]
  - Higher MAPE but better savings: model captures price structure better
    (rank correlation improved, even if point prediction is slightly noisier)

Tests added: 25 new tests in tests/test_ml_forecaster_v2.py
  - TestVolatilityRegimeFeatures (7 tests)
  - TestBuildFeatureMatrixVolatility (5 tests)
  - TestPriceModelConfigV2 (5 tests)
  - TestPriceQuantileForecasterV2 (5 tests)
  - TestCarbonCSVLoading (2 tests)
  - TestMLForecasterBenchmarkAcceptance (1 test)

WattTime carbon data limitations (documented):
  - Free plan: CAISO_NP15 and CAISO_NORTH only
  - PJM (us-east) and ERCOT (us-south): 403 INVALID_SCOPE errors
  - Paid WattTime plan required for PJM/ERCOT carbon coverage
  - Carbon optimization is CAISO-only until paid plan is available

The system remains:
- leakage-free (adversarial checks passed)
- benchmark-driven (real data, 5 folds, 0% missing price hours)
- economically honest (forward-fill bug found and fixed before publishing)
- reproducible (seed=42, deterministic, archived JSON)

===============================================================================

===============================================================================
ROADMAP PHASE 3 — WEATHER & COOLING INTELLIGENCE
===============================================================================

Status: INFRASTRUCTURE COMPLETE — ACCEPTANCE CRITERION NOT MET

Run date: 2026-05-23
Branch: claude/brave-mccarthy-QG2QF

Summary:
  Full weather feature infrastructure implemented and tested (24 weather tests,
  724 total tests passing). The primary acceptance criterion — training@
  caiso_pjm_ercot_da_rt ≥ 20.0% — was NOT met. Weather features regress the
  primary metric (15.0% → 11.1%). Root cause is structural: the January 2026
  cold snap falls entirely within training windows; all 5 eval folds run in
  February-March mild weather where temperature features add noise without
  signal. The infrastructure is correct and useful; the Q1 2026 backtesting
  fold structure simply cannot demonstrate weather value for this metric.

What was implemented:
  1. Weather data acquisition (scripts/fetch_weather_data.py):
     - Iowa Environmental Mesonet (IEM) ASOS METAR: free, no API key
     - Stations: KSFO (CAISO), KDCA (PJM), KHOU (ERCOT)
     - Fetched: data/weather_q12026.csv (7,635 rows, Q1 2026)
     - Fetched: data/summer2025/weather_summer2025.csv (8,859 rows, Summer 2025)
     - Open-Meteo ERA5 archive unavailable (503); IEM ASOS used instead

  2. Weather features added to ML forecaster (aurelius/forecasting/):
     - WEATHER_FEATURE_COLS: temperature_c, hdd_f, cdd_f, wind_speed_ms,
       temp_rolling_24h_c, temp_delta_24h_c
     - build_weather_lookup(): O(1) (timestamp, region) lookup dict
     - add_weather_features(): graceful join, missing → 0.0 (no crash)
     - build_feature_matrix() / build_feature_matrix_for_predict(): optional
       weather_lookup parameter (backward-compatible, optional)
     - PriceModelConfig: include_weather_features=True (default), plus new
       min_child_samples and reg_lambda regularization params
     - PriceQuantileForecaster.fit()/predict(): optional weather_df parameter
       with graceful fallback to price-only mode
     - Model version: v3.0 when weather active, v2.0 when not

  3. Backtest engine (aurelius/backtesting/engine.py):
     - BacktestEngine: weather_df parameter
     - _build_ml_forecast(): leakage-safe weather split (train < eval_start)
     - Backward-compatible via inspect.signature() for custom forecaster subclasses
     - Weather for prediction: full range (correct — weather is exogenous)

  4. Benchmark runner (benchmarks/run_benchmark.py):
     - --forecaster ml_quantile_weather: weather-enhanced v3.0
     - --forecaster ml_quantile: v2.0 (no weather, baseline preserved)
     - Auto-detection of co-located weather CSVs
     - ml_quantile (no weather) preserves exact 15.0% baseline

  5. Tests (tests/test_weather_features.py): 23 new tests, all passing
     - TestBuildWeatherLookup (5), TestBuildFeatureMatrixWithWeather (5)
     - TestBuildFeatureMatrixForPredictWithWeather (2)
     - TestPriceQuantileForecasterBackwardCompat (2)
     - TestPriceQuantileForecasterWithWeather (6)
     - TestWeatherLeakageSafety (2), TestBacktestEngineWeatherIntegration (1)

Benchmark results (2026-05-23):

  Primary metric: training@caiso_pjm_ercot_da_rt
    ml_quantile v2.0 (no weather):   15.0%   ← BASELINE PRESERVED
    ml_quantile_weather v3.0:        11.1%   ← REGRESSION (-3.9pp)
    Oracle ceiling (v2.0 run):       29.9%   ← remaining gap: 14.9pp

  Summer 2025: training@summer2025_3region
    ml_quantile v2.0 (no weather):    8.5%
    ml_quantile_weather v3.0:         8.2%   ← negligible change (-0.3pp)

  Full ml_quantile v2.0 benchmark (all workloads, all regions — 2026-05-23):
    background_maintenance@caiso_pjm_ercot_da_rt:  40.3%  ✓
    data_processing@caiso_pjm_ercot_da_rt:         37.7%  ✓
    llm_batch_inference@caiso_pjm_ercot_da_rt:     33.6%  ✓
    scheduled_batch@caiso_pjm_ercot_da_rt:         25.3%  ✓
    fine_tuning@caiso_pjm_ercot_da_rt:             13.4%  ⚠
    training@caiso_pjm_ercot_da_rt:                15.0%  ⚠
    realtime_inference@caiso_pjm_ercot_da_rt:      10.0%  ⚠
    (all 42 cells in benchmark artifact)

  Benchmark artifact: benchmarks/results/benchmark_ml_quantile_v3_weather_q12026_20260523.json

Root cause analysis (why weather regressed the primary metric):
  The January 2026 ERCOT cold snap (Jan 7-14, Houston min -4.4°C, 36 hours
  below freezing, max hdd_f=41) falls entirely within the 30-day training
  window for ALL 5 evaluation folds. The evaluation folds run February-March
  where ERCOT weather is mild (hdd_f ≈ 0-5). Weather features therefore:
    - Learn "high HDD → high ERCOT price" from January cold snap (in training)
    - Get mild weather signals during evaluation (no predictive signal)
    - Add noise to the joint 3-region LightGBM model that hurts CAISO/PJM
      predictions (feature stealing in shared model capacity)
  Tested configurations that all regressed:
    - All 3 regions' weather (11.1%)
    - ERCOT-only weather (9.3%)
    - ERCOT-only weather + regularization (num_leaves=31, mcs=50, λ=0.5) (2.6%)
  Note: single-region ERCOT benchmark improved with weather (-7.8% → -0.9%)
  confirming the weather feature logic is correct; the multi-region joint model
  is where the structural mismatch causes regression.

Why the acceptance criterion cannot be met with this dataset/configuration:
  The 30-day-training / 7-day-eval fold structure places the ONLY cold snap
  event in the training set. No eval fold contains a cold snap. Weather features
  would help if: (a) cold snaps appeared in eval windows, or (b) eval windows
  contained post-cold-snap "recovery" price elevations (they do not — ERCOT
  prices normalize within 7 days of a cold snap). Separate-model-per-region
  (ERCOT with weather, CAISO/PJM without) estimated to reach ~17.3% — still
  below the 20% acceptance criterion.

What works as intended (verified):
  ✓ Weather lookup joins correctly by (timestamp, region) UTC-floored hour
  ✓ Missing weather → 0.0 (graceful degradation, no crash)
  ✓ Leakage-safe split: train weather < eval_start, predict gets full range
  ✓ Backward-compatible with custom forecaster subclasses (inspect.signature)
  ✓ Single-region ERCOT benchmark improves from -7.8% to -0.9% with weather
  ✓ All 724 tests pass (701 pre-existing + 23 new weather tests)
  ✓ No regression in v2.0 baseline metrics (15.0% preserved exactly)

Acceptance criterion status:
  REQUIRED: training@caiso_pjm_ercot_da_rt ≥ 20.0% with weather features
  ACHIEVED:  11.1% (ml_quantile_weather) — criterion NOT met
  Honest finding: Q1 2026 fold structure makes this metric inaccessible to
  weather-based improvements. Infrastructure delivered; metric deferred.

===============================================================================
ROADMAP PHASE 3 EXTENSION — PER-REGION FORECASTER ARCHITECTURE
===============================================================================

Run date: 2026-05-23
Branch: claude/youthful-feynman-DfckO

Summary:
  PerRegionForecaster (v4.0) implemented, tested (26 new tests), and benchmarked.
  The per-region approach does NOT improve the Q1 2026 primary benchmark metric.
  Root cause identified and documented below. Infrastructure is correct and
  valuable for longer training windows (≥90 days per region).

What was implemented:
  1. PerRegionForecasterConfig dataclass:
     - base_config: PriceModelConfig (applied to all regions by default)
     - weather_regions: list of regions that receive weather features;
       default ["us-south"] so ERCOT gets weather and CAISO/PJM don't
     - region_configs: optional per-region PriceModelConfig overrides
       (ERCOT override: n_estimators=250, num_leaves=127 for spike patterns)

  2. PerRegionForecaster class (aurelius/forecasting/price_model.py):
     - Identical fit()/predict() interface to PriceQuantileForecaster
       → drop-in replacement in BacktestEngine, no engine changes required
     - fit(): groups price records by region, trains one separate
       PriceQuantileForecaster per region; weather passed only to regions
       in weather_regions (ERCOT), all others remain price-only
     - predict(): dispatches to per-region sub-forecaster; unknown/unfitted
       regions return flat fallback (no crash)
     - Backward-compatible: accepts bare PriceModelConfig for ease of use

  3. Benchmark runner update (benchmarks/run_benchmark.py):
     - Added ml_quantile_perregion to --forecaster choices
     - Weather data auto-loaded for perregion mode (for ERCOT weather features)
     - ERCOT gets higher-capacity model config (250 trees, 127 leaves)
     - Forecast quality collection extended to perregion mode

  4. Tests (tests/test_per_region_forecaster.py): 26 new tests, all passing
     - TestPerRegionForecasterConfig (5): config construction, backward compat
     - TestPerRegionForecasterFit (7): per-region training, weather routing,
       region config override, isolation between region models
     - TestPerRegionForecasterPredict (5): dispatch, fallback, p90≥p50
     - TestPerRegionForecasterDeterminism (2): same seed same output
     - TestPerRegionForecasterLeakage (2): fit only on training data
     - TestPerRegionForecasterMetadata (3): metadata, is_fitted
     - TestPerRegionForecasterBacktestIntegration (2): end-to-end with engine

Benchmark results (2026-05-23, Q1 2026, 30-day training, 5 folds):

  training@caiso_pjm_ercot_da_rt:
    ml_quantile v2.0 (joint, no weather, BASELINE):  15.0%  ← PRESERVED
    ml_quantile_perregion v4.0 (ERCOT gets weather): -10.9%  ← REGRESSION

  Root cause — per-region data starvation on 30-day windows:
    Joint model (2160 records per fold across 3 regions):
      Effective samples per LightGBM leaf ≈ 0.17 → well-generalized
    Per-region model (720 records per fold for each region):
      Effective samples per LightGBM leaf ≈ 0.06 → 3× more overfit
    With 1/3 the training data, per-region models fail to learn robust
    price-hour rank ordering from 30-day windows. The January cold snap
    patterns dominate ERCOT training data and produce noisy February-March
    predictions. CAISO/PJM models also have less statistical power.
    Additionally, the joint model learns CROSS-REGION calibration ("when
    CAISO is expensive AND ERCOT is cheap, route to ERCOT"), which per-region
    models cannot capture — this cross-region correlation is the core signal
    for multi-region optimization.

  Cross-region calibration loss (key finding):
    The 15.0% joint model savings come partly from correctly ranking all 3
    regions' price levels relative to each other. Per-region models produce
    forecasts on independent scales without cross-region recalibration, which
    causes the optimizer to systematically misroute jobs across regions.

  When per-region architecture WOULD help (conditions required):
    - Training window ≥ 90 days per region (≥ 2160 records per region)
    - Enough data for each region to learn its own patterns independently
    - Cross-region correlation captured via separate calibration layer
    - Appropriate hyperparameters per region (not just copying joint config)
    - Summer 2025 data (90 days) would be a better test case

Honest acceptance criterion status (Phase 3 Extension):
  REQUIRED: training@caiso_pjm_ercot_da_rt ≥ 17% with per-region architecture
  ACHIEVED: -10.9% — criterion NOT met
  Infrastructure is correct and tested; regression is a data-quantity constraint.
  ml_quantile v2.0 joint model (15.0%) remains the best validated forecaster.
  The per-region artifact is saved: benchmark_perregion_training_3region_q12026_20260523.json

Next exact task (after this PR merges):
  The per-region architecture is NOT the right next improvement for 30-day windows.
  The highest-leverage next milestones are:
  1. ROADMAP PHASE 4 — GPU Telemetry & DCGM:
     Implement DCGM-compatible fixture tests, Prometheus mock endpoint,
     GPU health / thermal / utilization features. Foundation infrastructure
     (no live cluster required). Expected to unlock Tier 3 control-level
     optimization and improve pilot appeal to GPU fleet operators.
  2. ROADMAP PHASE 5 — Queue-Aware Optimization:
     Implement queue depth / wait time / GPU availability features from
     Kubernetes/Slurm/Ray CSV trace inputs. Multi-signal objective function
     improvement (electricity + queue delay + SLA).
  3. ALTERNATIVE — Extended Training Window Benchmark:
     Re-run per-region benchmark with --train-days 90 (using summer 2025 data)
     to validate whether per-region outperforms joint when data is sufficient.
     If yes, perregion becomes the default for production deployments.

Tests after Phase 3 extension: 750 passed, 4 skipped (no regressions)
  - 724 pre-existing tests
  - 26 new tests in tests/test_per_region_forecaster.py

===============================================================================
ELECTRICITY MAPS CONTRIB AUDIT + MARKET-DATA PROVIDER ABSTRACTION
===============================================================================

What was audited
- Full inspection of the public electricitymaps/electricitymaps-contrib repo
  (AGPL-3.0): config/zones/*.yaml, config/data_centers/data_centers.json,
  electricitymap/contrib/parsers/*.py, DATA_SOURCES.md, license files.
- Clone lived only in /tmp and was NOT committed to Aurelius.
- Full audit written to docs/ELECTRICITYMAPS_CONTRIB_AUDIT.md.

Key audit findings
- The contrib repo is a CARBON-INTENSITY / GENERATION-MIX project, not a price
  project. NO US ISO zone (CISO/PJM/ERCOT/NYISO/MISO/SPP/ISO-NE) binds a price
  parser — only carbon/production/consumption/exchange.
- The 84 zones that DO have a price parser are European (ENTSO-E) or other
  international operators, and all return zonal/bidding-zone/country prices,
  never true nodal LMP.
- EU prices come from ENTSO-E Transparency, which Aurelius already reads
  directly — so the repo adds nothing for prices beyond confirming the source.
- Highest-value asset: config/data_centers/data_centers.json (cloud region ->
  grid zone). A verified subset was adapted (clean-room, factual) into the
  region registry.

What was implemented (this branch)
- aurelius/ingestion/market_data_provider.py — provenance-aware abstraction:
  MarketPricePoint, CarbonPoint, ProviderCapability, MarketDataProvider, plus
  Provenance/MarketType/Signal vocab, a benchmark-admissibility gate
  (assert_benchmark_admissible / filter_benchmark_admissible), and converters to
  the canonical DataFrame schema. Sits ABOVE grid_apis/base.py (no duplication).
- aurelius/ingestion/region_registry.py — canonical region -> ISO/TSO source
  region + Electricity Maps zone + carbon-provider zones + cloud-region aliases,
  each with a confidence level (unimplemented ISOs marked LOW). Single source of
  truth for EM zone maps.
- aurelius/ingestion/grid_apis/electricitymaps.py — added ElectricityMapsProvider
  (implements MarketDataProvider) with sandbox mode (ELECTRICITYMAPS_SANDBOX),
  is_sandbox propagation, token redaction in repr/logs, registry-backed zone
  map, and explicit refusal to serve US prices / nodal LMP. Legacy
  ElectricityMapsCarbonProvider preserved for backward compatibility.
- Tests: tests/test_market_data_provider.py, tests/test_region_registry.py,
  tests/test_electricitymaps_provider.py (39 new tests; HTTP fully mocked).

Was the Electricity Maps repo used only as reference?
- Yes. No AGPL source code was vendored, imported, or copy-pasted. Only factual
  identifiers (zone keys, EIC codes, source URLs) and cloud-region geography
  were used, re-expressed in Aurelius' own schema. Documented in the audit.

What remains
- Wire MarketDataProvider capability discovery into optimizer/backtester source
  selection (prefer source-of-truth ISO before EM fallback).
- Call assert_benchmark_admissible() inside the savings/benchmark harness.
- Implement SPP/MISO/NYISO/ISO-NE price providers to lift their registry
  confidence from LOW (currently carbon-only).
- Migrate any remaining hard-coded EM zone maps in cli.py to the registry.

Production API access still required for real benchmark claims?
- YES. Electricity Maps sandbox/randomized data is connector/schema-test only
  and is hard-blocked from benchmark/savings paths. Real economic claims still
  require real, unrandomized historical data from the source-of-truth ISO/TSO
  (and a paid EM plan for deeper carbon history). This branch adds plumbing and
  guardrails, not validated savings.
