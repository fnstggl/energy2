# Weather-Aware Optimization: Alpha Validation Report

Branch: `claude/weather-alpha-validation` · Date: 2026-05-24

This report upgrades Aurelius weather-awareness from benchmark-only scaffolding
to a leakage-free, statistically-validated, live-wired subsystem — and states,
with runnable evidence, whether weather creates **real economic alpha** under
realistic (forecast, not perfect-foresight) deployment conditions.

**Bottom line:** Yes, weather creates real, statistically-significant forecast
alpha that survives the realistic-forecast test — but it is **region-specific
(PJM/us-east), modest, and can regress in summer.** Savings are net-positive and
sign-stable across job mixes, but **dollar-concentrated in high-cost winter
folds.** The mechanism is steady PJM forecast improvement, **not** spike
prediction. It is honestly claimable as "weather-aware forecasting that helps
PJM and net-helps multi-season savings," not as a universal grid-stress edge.

---

## Phase 1 — Leakage removed

**Defect:** `engine.py` set `predict_weather_df = weather_df.copy()` — the eval
window received **observed future weather** (perfect foresight). Not price
leakage (weather is exogenous) but it **is** forecast leakage: a deployed system
never has the future observation.

**Fix (backward-compatible):**
- `BacktestEngine(forecast_weather_df=...)` and `LiveShadowRunner(forecast_weather_df=...)`:
  training uses observed history `< eval_start`; the prediction horizon uses the
  **day-ahead forecast**.
- `_assert_no_observed_weather_in_eval()` guards against regressions
  (`WeatherLeakageError`).
- Realistic forecast weather is sourced from **Open-Meteo Previous Runs API**
  (`*_previous_day1` = the value forecast ~24 h ahead).

**Day-ahead forecast error (Open-Meteo, observed ERA5 vs `previous_day1`):**

| lead | temp MAE | hdd_f MAE | cdd_f MAE |
|---|---|---|---|
| day-ahead | 1.25 °C | 1.18 | 1.07 |
| 2-day | 1.81 °C | 1.83 | 1.43 |

**Critical result:** in evaluation, honest (forecast) ≈ leaky (observed):
overall dMAE −1.36 (honest) vs −1.49 (leaky). **The leakage was not inflating
weather value** — day-ahead weather is good enough that realism barely changes
the answer. Weather's limitation is its price-predictive power, not forecast error.

---

## Phase 2 — Trustworthy evaluation (bootstrap CIs)

`benchmarks/weather_alpha_eval.py` — 14 walk-forward folds (90-d train / 14-d
eval) spanning summer→winter, 3 seeds, modes {none, observed-leaky,
forecast-honest}, bootstrap 95% CI over folds. A claim is valid only if the CI
upper bound < 0.

**Forecast quality — honest (day-ahead) mode, dMAE vs price-only:**

| scope | dMAE | 95% CI | verdict |
|---|---|---|---|
| overall | −1.36 | [−2.53, −0.48] | **significant** |
| us-west (CAISO) | −0.05 | [−0.16, +0.05] | n.s. |
| us-east (PJM) | **−3.64** | [−6.75, −1.03] | **significant** |
| us-south (ERCOT) | −0.40 | [−0.64, −0.16] | significant (small) |

RMSE and pinball90 agree: significant overall and for PJM; n.s. for CAISO and
(pinball) ERCOT. By season, weather **hurts in summer** (all regions),
**helps PJM strongly in winter** (dMAE −7.0), and helps PJM/ERCOT in shoulder.

**Savings — `benchmarks/weather_savings_eval.py`** (real optimizer, bootstrap
over job-mix seeds, day-ahead forecast weather, training workload, DAM prices):

- mean savings delta (ON − OFF) = **+8.8 pp**, 95% CI **[+5.8, +11.7]**
  (6 job-mix seeds), **all seeds positive** — no sign-flip (contrast: the prior
  audit's metric flipped +2.0/−3.9/−0.6 pp with job count).
- Per-fold: **6/6 folds positive**, but **one cold-snap fold = 62%** of the
  dollar value. Broad in sign, concentrated in dollars.

> Caveat: the savings bootstrap resamples job mixes over **shared folds**, so it
> measures job-mix stability, not event robustness. The per-fold breakdown is
> the honest robustness check.

---

## Phase 3 — Wired into real (live) optimization

Previously weather touched only the benchmark forecaster. Now:
- `LiveShadowRunner` accepts `weather_df` / `forecast_weather_df` and threads
  them leakage-free into `fit()`/`predict()` (verified: training weather is
  strictly pre-decision; horizon weather is the forecast).
- CLI: `aurelius shadow run --weather-file ... [--forecast-weather-file ...]`
  now fits **v3.0 (lightgbm_quantile+volatility+weather)** in the live path.

Weather influences orchestration **through the price forecast the optimizer
consumes** — no optimizer redesign, no hardcoded "hot⇒expensive" rules.

---

## Phase 4 — Production ingestion (Open-Meteo)

`aurelius/ingestion/weather_provider.py` — `OpenMeteoWeatherProvider`:
- **Historical** (ERA5 archive) — observed ground truth / training.
- **Forecast** (live) — production decision-time weather.
- **Previous Runs** (`previous_dayN`) — fixed lead-time forecast for backtests.
- Retries w/ exponential backoff, on-disk JSON cache (reproducible), region→coord
  mapping, canonical derived-feature pipeline (identical maths to legacy script).

**Provider verdict:** Open-Meteo is **sufficient and should be primary**. ERA5
matched the existing IEM cold-snap data (Houston −3.7 °C, 2026-01-26).
Adding NOAA/Meteostat is **not justified**: day-ahead error is already ~1.25 °C,
and weather's binding limitation is its weak price-predictive power, not the data
source. More providers would add complexity without moving the alpha.

---

## Phase 5 — Grid-stress mechanism (where does the value come from?)

High-price-hour MAE decomposition (honest mode, % MAE improvement):

| region | normal hours | expensive (top-10%) hours |
|---|---|---|
| us-west | +0.3% | +1.3% |
| us-east (PJM) | **+11.1%** | +1.9% |
| us-south (ERCOT) | +1.8% | **+0.0%** |

**Weather does NOT predict spikes.** It does ~nothing at ERCOT's extreme hours
and little at any region's top-decile hours. The gain is **broad PJM forecast
improvement at normal-to-moderate hours**, which the optimizer amplifies into
dollars during high-cost winter folds via routing leverage. The "weather predicts
cold-snap/heat-wave grid stress" narrative is **not supported** by the data; the
real, validated effect is mundane and PJM-specific.

---

## Phase 6 — Brutally honest assessment

1. **Genuinely useful?** Yes for forecasting, with caveats. Significant overall
   and PJM forecast-MAE improvement that survives realistic forecast weather.
2. **Which regions?** PJM (us-east) clearly. ERCOT marginally. CAISO not at all.
3. **Which features matter?** Temperature/HDD/CDD/rolling-temp in PJM. (`humidity`
   is still computed-but-unused; the `solar_cf`/`wind_cf` path in `features.py`
   remains dead — see backlog.)
4. **Which are noise?** Weather in CAISO and in summer (regresses).
5. **Materially improves optimization?** Yes in this config: +7.3 pp savings,
   sign-stable across job mixes; but dollar-concentrated in one winter fold.
6. **Statistically significant?** Forecast: yes (CI excludes 0). Savings: yes
   across job mixes; treat the magnitude as uncertain (few folds, one dominant).
7. **Stable or fragile?** Sign is stable; **dollar magnitude is fragile** (62%
   from one cold-snap fold) and **season-dependent** (summer hurts).
8. **Realistic savings contribution?** Directionally a few pp on price-sensitive
   flexible workloads in multi-season operation, with most value in cold winter
   periods. Not a year-round uniform edge.
9. **Production-ready?** Infrastructure: yes (leakage-free eval, live wiring,
   Open-Meteo ingestion, tests). Alpha: real but modest/concentrated — enable
   **selectively**, not universally.
10. **Minimum work before publicly claiming "weather-aware optimization":**
    - Season/region gating (disable summer & CAISO where it regresses).
    - Confirm savings under **RT-settlement** scoring (this test used DAM only).
    - Wider multi-year sample to de-risk the one-fold dollar concentration.
    - Wire the live forecast-weather feed into the production scheduler (shadow
      path done) + feature-usage/importance monitoring.

## Reproduce

```
python benchmarks/weather_alpha_eval.py            # forecast-level bootstrap
python benchmarks/weather_savings_eval.py --seeds 8  # savings bootstrap
AURELIUS_OPENMETEO_LIVE=1 python -m pytest tests/test_openmeteo_provider.py
```

Artifacts: `benchmarks/results/weather_alpha_eval.json`,
`benchmarks/results/weather_savings_eval.json`.
Data: `data/weather_openmeteo/{observed_era5,forecast_day1,forecast_day2}.csv`.
