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
claude/youthful-feynman-2pFK8

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
798 passed, 0 failed (4 skipped), 200 warnings

New tests (Phase 5 — Queue-Aware Optimization):
48 new tests in tests/test_queue_aware.py
  - TestQueueStateModel (2)
  - TestQueueProviderFromCSV (4)
  - TestQueueProviderGetWaitHours (6)
  - TestQueueProviderToDictLookup (2)
  - TestQueueProviderGenerateFixture (7)
  - TestLookupLastKnown (5)
  - TestObjectiveFunctionQueueDelay (6)
  - TestSchedulerQueueAwareRouting (6)
  - TestOptimizationConfigQueue (4)
  - TestBacktestEngineQueueIntegration (5)
  - TestQueueProviderRoundTrip (1)

Pre-existing tests:
750 (all Phase 1-4 benchmark, migration, spread_risk, ML forecaster, etc.)

Skipped:
4 live API tests requiring credentials

Result:
ALL PASSING (798 total)

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

Tests after Phase 3 extension: 750 passed, 4 skipped (no regressions)
  - 724 pre-existing tests
  - 26 new tests in tests/test_per_region_forecaster.py

===============================================================================
ROADMAP PHASE 5 — QUEUE-AWARE OPTIMIZATION
===============================================================================

Status: COMPLETE (2026-05-23)
Branch: claude/youthful-feynman-2pFK8

Summary:
  Full queue-aware optimization infrastructure implemented, tested (48 new
  tests), benchmarked, and integrated end-to-end. The optimizer now penalizes
  placements in congested regions via a queue delay cost in the multi-signal
  objective function. Backward-compatible: zero cost config = unchanged behaviour.

What was implemented:

  1. Data model (aurelius/models.py):
     - QueueState dataclass: timestamp, region, cluster_id, gpu_type,
       available_gpus, queue_depth_jobs, est_wait_hours
     - OptimizationConfig: queue_delay_cost_per_gpu_hour field (default 0.0)
       plus to_dict() inclusion

  2. QueueProvider (aurelius/ingestion/queue_provider.py):
     - from_csv(): loads canonical queue CSV schema (6 columns)
     - from_dataframe(): construct from DataFrame (used in backtesting engine)
     - get_wait_hours(region, timestamp): leakage-safe "last known before T" lookup
     - to_dict_lookup(): returns {region: {timestamp: est_wait_hours}} —
       plugs directly into objective function and scheduler
     - generate_fixture(): reproducible synthetic queue data with realistic
       business-hours congestion patterns (seed-deterministic)
     - to_dataframe() / save_csv(): export and round-trip support
     - Multi-cluster aggregation: weighted-mean wait time across clusters

  3. Objective function (aurelius/optimization/objective.py):
     - ObjectiveComponents: added queue_delay_cost: float field
     - _lookup_last_known() helper: leakage-safe last-value lookup
     - calculate(): queue_data parameter (optional); computes
       queue_cost = est_wait_h * queue_delay_cost_per_gpu_hour * gpu_count
     - calculate_job_cost() / compare_options(): queue_data propagated
     - New total formula:
       alpha*energy + beta*carbon + gamma*risk + delta*SLA + data_transfer + queue_delay

  4. Scheduler (aurelius/optimization/scheduler.py):
     - solve(): queue_data parameter (optional)
     - _solve_greedy() / _find_best_slot(): queue_data threaded through
     - _solve_local_search(): queue_data threaded through
     - Optimizer naturally routes away from congested regions because
       queue delay adds to the placement objective score

  5. Backtesting engine (aurelius/backtesting/engine.py):
     - __init__(): queue_df parameter (optional DataFrame)
     - _run_fold(): builds queue_data lookup from queue_df with proper
       leakage-safe split (only rows with timestamp < eval_start used)
     - Passes queue_data to scheduler.solve() per fold

  6. Benchmark runner (benchmarks/run_benchmark.py):
     - --queue-file: path to queue CSV (relative to repo root)
     - --queue-delay-cost: $/GPU-hour opportunity cost (default 0.0)
     - Auto-loads and logs queue signal stats
     - Rebuilds OptimizationConfig with queue cost when provided

  7. Synthetic fixtures:
     - data/queue_q12026_3region.csv: 6,477 rows (Q1 2026, CAISO/PJM/ERCOT)
       us-west: mean 1.93h wait, us-east: mean 3.21h wait, us-south: mean 0.64h
     - data/summer2025/queue_summer2025_3region.csv: 6,621 rows (Summer 2025)

Adversarial findings (all verified correct):
  ✓ No queue data / zero config → zero queue_delay_cost (backward compat)
  ✓ Future queue data (dt(100)) not used when scheduling at dt(0) (leakage safe)
  ✓ Queue-aware optimizer routes to us-south (0h wait) over us-west (5h wait)
    even when us-west energy price is lower, when queue cost dominates
  ✓ test_high_queue_cost_overrides_price_advantage: $320 queue cost >> $4
    energy savings → correctly switches to us-east (passing test)
  ✓ Multi-cluster weighted aggregation verified
  ✓ Business-hours congestion pattern verified (12-20 UTC higher than 0-8 UTC)
  ✓ Far-future queue data (2030) not used in 2026 backtesting folds

Queue fixture data note:
  SYNTHETIC: These fixtures are generated programmatically for testing/demo.
  They are NOT from any real customer queue system. Real queue traces must
  come from the customer's Kubernetes/Slurm/Ray cluster for production use.
  Synthetic queue data MUST NOT be used for savings benchmark claims.

Benchmark demo (quick mode, llm_batch_inference, queue-aware):
  --queue-file data/queue_q12026_3region.csv --queue-delay-cost 2.0
  Result: 33.0% vs current_price_only [folds=7]
  Note: This is a pricing-only benchmark; the queue delay component adds
  routing intelligence but the savings % still reflects energy price savings
  (queue cost is an additional optimizer signal, not a new savings source).

Tests: 48 new tests in tests/test_queue_aware.py
  - TestQueueStateModel (2): construction, optional gpu_type
  - TestQueueProviderFromCSV (4): load, bad column, defaults, regions
  - TestQueueProviderGetWaitHours (6): exact match, last-known, no-data,
    unknown region, leakage safety, multi-cluster aggregation
  - TestQueueProviderToDictLookup (2): format, mirrors get_wait_hours
  - TestQueueProviderGenerateFixture (7): records, determinism, seed diff,
    non-negative, congestion pattern, base_wait_hours, save/reload
  - TestLookupLastKnown (5): empty, exact, last-before, all-after, latest
  - TestObjectiveFunctionQueueDelay (6): no data, zero config, calculation,
    total inclusion, zero-gpu-count default, backward compat
  - TestSchedulerQueueAwareRouting (6): routes-to-clear, no-queue, high-cost,
    zero-cost-ignores, backward-compat, objective-components
  - TestOptimizationConfigQueue (4): default zero, custom, to_dict, backward
  - TestBacktestEngineQueueIntegration (5): none, stores, run-with, run-without,
    leakage-safe-in-fold
  - TestQueueProviderRoundTrip (1): DataFrame round-trip

Total tests: 798 passed, 4 skipped (no regressions)

Enterprise value of queue-aware optimization:
  Tier 2 control-level optimization (cluster/queue placement) is now
  implemented with a clear CSV trace ingestion path. A neocloud pilot can
  provide their queue state logs and immediately get queue-aware routing
  without any code changes — just supply --queue-file and --queue-delay-cost.
  The control pathway: any customer who tracks GPU availability + queue depth
  in a CSV export from Kubernetes/Slurm/Ray/AWS Batch can plug it in.

Next exact task (after this PR merges):
  ROADMAP PHASE 4 — GPU Telemetry & DCGM:
  Implement DCGM-compatible fixture tests, Prometheus mock endpoint,
  GPU health / thermal / utilization features. Foundation infrastructure
  (no live cluster required). Exposes Tier 3 control-level optimization
  (specific GPU/node placement based on health, temperature, utilization).
  ALTERNATIVE: Extended Training Window Benchmark (summer 2025, 90-day windows)
  to validate whether per-region forecaster outperforms joint model with
  sufficient data. If yes, perregion becomes the default forecaster.

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

===============================================================================
ROADMAP FORECASTER v5.0 — PRICE CONTEXT FEATURES (COLD-SNAP RECOVERY)
===============================================================================

Status: INFRASTRUCTURE DELIVERED — ACCEPTANCE CRITERION NOT MET
Run date: 2026-05-23
Branch: claude/youthful-feynman-oBAKP

Summary:
  Price context feature infrastructure (v5.0) implemented, leakage-tested, and
  benchmarked on Q1 2026 3-region data. The primary acceptance criterion —
  ml_quantile_v5 ≥ ml_quantile_v2 with no regressions — was NOT met. The rank
  features help fine_tuning (+2.7pp) but regress background_maintenance (-8pp)
  and realtime_inference (-14.7pp). Root cause identified: lag_336h adds noise
  for short-horizon workloads in the joint model on 30-day training windows.
  include_rank_features remains False by default; v5.0 is an opt-in benchmark.

Motivation:
  Oracle diagnostics identified a 22-48pp forecasting gap (v2.0 actual vs
  oracle ceiling):
    training@3region:    ml_v2 15.0% vs oracle 29.9% → 14.9pp gap
    fine_tuning@3region: ml_v2 12.6% vs oracle 46.8% → 34.2pp gap
  Root cause: January 2026 ERCOT cold snap (lag_168h = $2000/MWh) anchors the
  v2.0 model to predict high ERCOT prices for 7 days after recovery to $20-50/MWh.
  Target: encode "current price vs last week's price for this region."

What was implemented:
  1. compute_price_rank_features() — v5.0 cold-snap recovery features:
     - price_momentum_168h: (price - lag_168h) / |lag_168h|, clipped [-1, 5]
       Cold-snap recovery: -0.95 = "95% cheaper than last week"
       Spike onset: +5.0 = "≥5× more expensive than last week"
       Neutral fallback: 0.0 when lag_168h context unavailable
     - price_vs_lag168_abs: price / lag_168h, clipped [0, 10]
       Complementary absolute ratio (0.05 = 5% of last week's price)
  
  2. _compute_per_region_lag_168h() — cross-region contamination fix:
     compute_lagged_features() builds one ts→val dict for ALL regions combined;
     in joint training, ERCOT's cold-snap $2000 overwrites the lag lookup for
     CAISO and PJM. Per-region lookup dicts prevent this contamination.
     Each region's lag_168h is computed against its own historical prices only.
  
  3. PRICE_RANK_FEATURE_COLS = ["price_momentum_168h", "price_vs_lag168_abs"]
  
  4. build_feature_matrix(): include_rank_features=False parameter; when True,
     uses _compute_per_region_lag_168h() for correct multi-region training.
     Placeholder branch: momentum=0.0 (neutral), ratio=1.0 (neutral).
  
  5. build_feature_matrix_for_predict(): include_rank_features=False parameter
     (single-region context at predict time → no contamination risk).
  
  6. PriceModelConfig.include_rank_features=False (opt-in, not default).
  
  7. PriceQuantileForecaster.__init__(): when _use_rank=True, lag_hours expanded
     to [1, 6, 24, 168, 336] (adds 2-week bi-weekly lag for broader context).
  
  8. benchmarks/run_benchmark.py: ml_quantile_v5 forecaster option added.
     Uses include_rank_features=True explicitly.
  
  9. tests/test_forecaster_v5.py: 44 tests for new feature design:
     - TestComputePriceRankFeatures (14 tests)
     - TestPriceRankFeatureCols (3 tests)
     - TestBuildFeatureMatrixWithRankFeatures (6 tests)
     - TestBuildFeatureMatrixForPredictWithRankFeatures (3 tests)
     - TestPriceModelConfigV5 (5 tests)
     - TestPriceQuantileForecasterV5 (7 tests)
     - TestRankFeaturesLeakageSafety (3 tests)
     - TestV5BenchmarkAcceptance (2 tests)

Design iterations (why price_momentum_168h was chosen over other approaches):

  Attempt 1 — Rolling percentile rank features (range_position, below_p10):
    Root cause: at predict time, future values forward-filled with last_known_price
    → trailing 168h window for k≥168h prediction = all last_known_price
    → range_position = 0, below_p10 = 1 always → degenerate features
    Training-prediction distribution shift → model confused → REJECTED

  Attempt 2 — Per-region percentile features (with regions parameter):
    Root cause: same forward-fill degeneration for long-horizon predictions
    → still degenerate at k≥168h → REJECTED

  Attempt 3 — price_momentum_168h (current implementation):
    Derived from time-based lag_168h (computed by compute_lagged_features
    via timestamp lookup, not row-shift). For k<168h: lag_168h reaches into
    real context. For k≥168h: both current and lag_168h = last_known_price
    → momentum = 0 (neutral, not degenerate). Graceful degradation. ADOPTED.

Benchmark results (2026-05-23, Q1 2026, 5 folds, 30-day training):

  ml_quantile_v5 vs ml_quantile_v2 (all deltas vs v2.0 baseline):
    fine_tuning:          15.3% vs 12.6%  (+2.7pp ✓ improvement)
    llm_batch_inference:  33.7% vs 33.7%  ( 0.0pp  no change)
    scheduled_batch:      26.3% vs 25.4%  (+0.9pp ✓ slight improvement)
    data_processing:      36.4% vs 37.9%  (-1.5pp  slight regression)
    training:             -1.6% vs 15.0%  (-16.6pp ✗ regression + 12% missing hours)
    background:           37.8% vs 45.8%  ( -8.0pp ✗ regression)
    realtime_inference:  -11.9% vs  2.8%  (-14.7pp ✗ big regression)
    Mean:                 19.4% vs 24.7%  ( -5.3pp ✗ overall regression)

  Note on training workload: 12% missing price hours flagged (us-west/us-east
  data gaps March 15-16, 2026) — savings figure not reliable for that workload.

  Oracle ceilings (for reference):
    training ceiling:    29.9%
    fine_tuning ceiling: 46.8%
    llm_batch ceiling:   42.7%

Root cause analysis (why v5.0 regressed):

  1. lag_336h noise for short horizons:
     v5.0 adds lag_336h (2-week lag). For realtime_inference (4h window) and
     background_maintenance (168h window), the 2-week lag introduces noise
     because price regimes change on 2-week timescales. LightGBM with only
     30 days of training data (720 per-region hourly records per fold) cannot
     learn useful signal from lag_336h — it overfits to the cold-snap regime.
     Estimated contribution: ~8pp of regression in realtime_inference.

  2. Residual training-prediction distribution shift:
     df["lag_168h"] in the training matrix is contaminated (ERCOT overwrites
     CAISO/PJM in global ts→val dict). The rank features use per-region
     lag_168h (correct) while df["lag_168h"] in the same matrix uses the
     contaminated global lag. These two features encode DIFFERENT reference
     prices, creating an inconsistency the LightGBM model must navigate.
     Fix would require per-region lag for ALL lag features (not just rank).

  3. Feature count vs training data ratio:
     v2.0: ~12 features, 720 per-region training records per fold
     v5.0: ~15 features (+price_momentum_168h, +price_vs_lag168_abs, +lag_336h)
     Higher feature-to-data ratio → overfitting with 30-day windows
     
When v5.0 features WOULD help (conditions required):
  - Training windows ≥ 60 days (≥ 1440 per-region records) to absorb lag_336h
  - OR: exclude lag_336h from v5.0 feature set (use rank features with v2.0 lags)
  - OR: per-region architecture with ≥ 90 days (eliminates contamination)
  - Dataset that includes cold-snap recovery WITHIN evaluation windows
    (not just in training): Q2 2026 or winter 2025-2026 would be needed

Acceptance criterion status:
  REQUIRED: ml_quantile_v5 ≥ ml_quantile_v2 (24.7% mean) with no regressions
  ACHIEVED: 19.4% mean (-5.3pp) — criterion NOT met
  v2.0 (24.7% mean) remains the best validated joint forecaster.

Tests: 834 passed, 0 failed, 4 skipped (834 total, up from 798)
  - 790 pre-existing tests (all preserved)
  - 44 new tests in tests/test_forecaster_v5.py

Benchmark artifacts:
  benchmarks/results/benchmark_ml_quantile_v5_3region_q12026_20260523.json

LAST VERIFIED TEST STATUS (v5.0 branch):
  Unit + integration: 834 passed, 0 failed, 4 skipped
  All pre-existing tests preserved: YES
  v2.0 baseline preserved: 24.7% mean (CONFIRMED)

Next exact task:
  Option A: Extended Training Window Benchmark (60-day windows)
    Tests whether rank features improve over v2.0 with more data.
    If yes, v5.0 becomes default. Required: Summer 2025 90-day dataset or
    configurable train_days parameter in benchmark runner.
  Option B: ROADMAP PHASE 4 — GPU Telemetry & DCGM
    Implement fixture-backed Prometheus metrics, DCGM data model, Tier 3
    optimizer interface. Foundation infrastructure (no live cluster needed).
