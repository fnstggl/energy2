# Economic Overlay Layer v1 — Analysis-Tier Scale-Up

> **Overlay / shadow PR.** No production scheduler / residency / scorer /
> robust-energy behaviour is changed. No ML is trained. No production
> savings are claimed. No oracle / FIFO headline. HF / public-list-price /
> market-LMP data is NEVER labelled pilot telemetry or operator truth.
> Credentials are read from the environment only and never printed or
> committed.
>
> **Read first:**
> - `docs/ECONOMIC_OVERLAY_LAYER_V1.md` (the v1 smoke-test layer this scales)
> - `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md` (operator-policy-only coefficients)
> - `docs/HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md` (which public datasets exist)
> - `data/external/economic_overlay/economic_overlay_analysis_summary.json`
> - `data/external/economic_overlay/economic_overlay_analysis_eval.json`
> - `data/external/economic_overlay/market_fetch_manifest.json`

## 1. Goal

Scale the v1 Economic Overlay from a 35-row smoke test to an
**analysis-tier corpus** large enough to train economic ML targets
offline, joining real operational traces to real / derived / scenario
economic signals — without inventing a single constant.

## 2. Rows joined

**187,505 overlay records** across 6 source configs:

| Source | Config | Rows | Operational signals used |
|---|---|---:|---|
| `asdwb/cara_latency_prediction` | train_flat (head) | 60,000 | TTFT, TPOT, e2e, KV util, output tokens |
| `eth-easl/swissai-serving-trace` | llama3-70b + qwen3-32b bucket-reuse | 60,000 | reuse_percentage, bucket count |
| `Qinghao/AcmeTrace` | GPU_AB_Power (IPMI) | 40,000 | derived GPU power (Sys−CPU−Mem), 15 s window |
| `Qinghao/AcmeTrace` | trace_kalos jobs | 19,907 | queue wait, job duration, gpu_num |
| `semianalysisai/cc-traces-weka` | traces_3000mib | 5,000 | TTFT prior, prefix reuse, cache-loss proxy |
| `optimum-benchmark/llm-perf-leaderboard` | 7 CUDA/CPU configs | 2,598 | measured TTFT/TPOT + per-request kWh + VRAM |

Bounded raw heads fetched into `data/external/hf/<safe>/raw/` (all
gitignored; ~1.04 GiB total): CARA 280 MiB, CARA-queue 170 MiB,
SwissAI-llama3 150 MiB, SwissAI-qwen3 130 MiB, AcmeTrace power/util
120 MiB each, AcmeTrace jobs 8.5 MiB, Optimum CSVs full. The full
187 k-row overlay corpus lives in
`data/external/economic_overlay/analysis_corpus/*.jsonl` (gitignored);
only a bounded ≤800-row sample per source + the summary/eval/manifest
JSON are committed.

## 3. API coverage

| Provider | Status | Rows | value_quality | Window / region |
|---|---|---:|---|---|
| **PJM** Data Miner DA LMP | ✅ measured | 337 | measured | 14 d, us-east |
| **CAISO** OASIS DA LMP | ✅ measured | 168 | measured | 7 d, us-west |
| **ERCOT** SPP DA | ✅ measured | 336 | measured | 14 d, us-south |
| **WattTime** MOER carbon | ⏭️ skipped (HTTP 403) | — | scenario_prior fallback | account not authorized for these regions |
| `afhubbard/gpu-prices` | ✅ measured (public list) | 10,313 | prior_public_list_price | 5 daily snapshots, 12+ clouds |

Three of four energy/carbon market providers are **live-measured this
run**. WattTime returned HTTP 403 for the supplied account (not
authorized for the requested balancing authorities) and was skipped
per operator instruction — carbon intensity falls back to a
`scenario_prior` midpoint. Full per-provider record (no credential
values) in `market_fetch_manifest.json`.

### Missing credentials / access (no values)
- **WattTime**: account authenticated but returns 403 on `/login` for
  these regions — needs region registration / plan upgrade on the
  WattTime side. Recorded as `skipped_403`.

All other credentials (PJM_API_KEY, ERCOT_API_KEY + ERCOT_USERNAME +
ERCOT_PASSWORD, HF_TOKEN) were present and worked. CAISO needs no key.

## 4. Economic-field coverage (187,505 rows)

| Field | Rows with value | Coverage | Dominant value_quality |
|---|---:|---:|---|
| `estimated_gpu_cost_usd` | 187,505 | 100.0 % | derived (from `prior` GPU price) |
| `sla_safe_goodput_per_dollar` | 187,505 | 100.0 % | derived |
| `estimated_decode_cost_usd` | 127,598 | 68.1 % | derived |
| `estimated_prefill_cost_usd` | 127,598 | 68.1 % | derived |
| `estimated_cache_value_usd` | 60,000 | 32.0 % | derived (SwissAI reuse × prior) |
| `estimated_energy_cost_usd` | 42,598 | 22.7 % | scenario_prior (market price, scenario region) |
| `estimated_carbon_kg` | 42,598 | 22.7 % | scenario_prior |
| `estimated_carbon_cost_usd` | **0** | 0 % | **missing — operator-policy-only** |
| `estimated_cold_start_cost_usd` | 0 | 0 % | missing (no measured model-load duration in these sources) |
| `estimated_migration_cost_usd` | 0 | 0 % | missing (no cache-loss proxy realized in these sources) |

### measured / derived / prior / scenario / missing breakdown (key inputs)

| Input | breakdown |
|---|---|
| `gpu_price_usd_per_hour` | prior 187,134 · prior_fuzzy_match 371 |
| `energy_kwh` | derived_from_power_prior 40,000 · measured 2,598 (Optimum) · missing 144,907 |
| `electricity_price_usd_per_kwh` | scenario_prior 187,505 (region-less traces) |
| `carbon_intensity_g_per_kwh` | scenario_prior 187,505 (WattTime 403 → midpoint) |
| `estimated_carbon_cost_usd` | missing 187,505 (no operator carbon price) |

Subgroups (GPU): A100 93,769 · H100 65,000 · P100 16,582 · V100 10,011
· A10 1,424 · CPU-C7i 371 · T4 348.

## 5. Three result classes (never combined)

The classifier keys on the cost terms **actually realized** in each
row's `goodput/$` denominator — a looked-up-but-unused scenario price
does NOT taint a row:

| Class | Rows | Meaning |
|---|---:|---|
| `measured_same_record` | **0** | needs operator-supplied `$/hr` — none given |
| `cross_dataset_joined` | **144,907** | cost driven by the public GPU-price prior (CARA, SwissAI, CC-traces, AcmeTrace jobs); no scenario term realized |
| `scenario_prior` | **42,598** | energy cost realized from a market price applied under a scenario region assignment (Optimum, AcmeTrace power) |

This is the core honesty result: **77 % of the corpus's economic
headline is driven by the measured public GPU price prior**, not by
scenario energy assumptions. Only the 23 % of rows that carry an
energy term inherit the scenario-region label.

## 6. A–H evaluation (primary KPI: SLA-safe goodput / $)

| Variant | goodput/$ rows | mean goodput/$ | class split (joined / scenario) |
|---|---:|---:|---|
| A — existing scorer baseline (no overlay) | 0 | — | 187,505 / 0 |
| B — + public GPU price | 187,505 | 22,974.64 | 187,505 / 0 |
| C — + energy/carbon only | 42,598 | 3,266,554 | 144,907 / 42,598 |
| D — + cache value | 187,505 | 22,974.64 | 187,505 / 0 |
| E — + full overlay | 187,505 | 22,974.63 | 144,907 / 42,598 |
| F — full + TTFT p50 prior | 187,505 | 22,974.63 | 144,907 / 42,598 |
| G — full + cache/prefix prior | 187,505 | 23,227.18 | 144,907 / 42,598 |
| H — full + both priors | 187,505 | 23,227.21 | 144,907 / 42,598 |

Notes:
- **A → B is the calibration unlock**: the baseline scorer computes
  goodput/$ on 0 rows (no public-data inputs); adding the GPU price
  prior makes it computable on all 187,505.
- **C** computes goodput/$ only where an energy cost exists (42,598)
  and — with no GPU cost in that variant — divides goodput by the tiny
  energy cost, hence the large mean. Reported separately, never merged.
- **Cache/prefix prior (G/H) lifts mean goodput/$ by ~1.1 %** vs E
  (23,227 vs 22,975) — the cache-value term shrinks the cost
  denominator on the 60 k SwissAI rows. The TTFT prior alone (F) does
  not move it (those rows already have measured/own TTFT).
- **Ranking-change rate = top-1-change rate = 0.0** for every variant:
  the overlay is *additive* to the existing scorer; it fills $-terms,
  it does not re-rank candidates.

### Mean realized cost terms (variant E, $)
- `estimated_gpu_cost_usd`: mean 27.89 (n=187,505) — dominated by
  multi-GPU AcmeTrace jobs (gpu_count up to node size × long duration).
- `estimated_decode_cost_usd`: mean 0.0142 (n=127,598)
- `estimated_prefill_cost_usd`: mean 8.6e-4 (n=127,598)
- `estimated_cache_value_usd`: mean 6.3e-4 (n=60,000)
- `estimated_energy_cost_usd`: mean 4.1e-4 (n=42,598)
- `estimated_carbon_kg`: mean 4.5e-3 kg (n=42,598) — physical quantity;
  carbon **cost** stays missing (operator-policy-only).

## 7. Formulas (unchanged from v1; see `economic_overlay.py`)

```
gpu_cost          = gpu_price_$/hr × gpu_count × gpu_seconds / 3600
prefill_cost      = ttft_s × (gpu_price_$/hr / 3600) × gpu_count
decode_cost       = tpot_s × output_tokens × (gpu_price_$/hr / 3600) × gpu_count
energy_kwh        = measured | gpu_power_w × gpu_seconds / 3_600_000 (derived_from_power_prior)
energy_cost       = energy_kwh × electricity_$/kWh
carbon_kg         = energy_kwh × carbon_g/kWh / 1000      (physical)
carbon_cost       = carbon_kg × operator.carbon_$/kg      (operator-policy-only → missing)
cache_value       = reuse_pct × ttft_s × (gpu_price_$/hr / 3600) × gpu_count
sla_met           = e2e_latency_s ≤ sla_s
goodput/$         = (output_tokens if sla_met else 0)
                    / (gpu_cost + energy_cost + migration + cold_start − cache_value)
```

No utility weights. No invented `$/hr`, `$/kWh`, or `$/kg`. Energy
price = real PJM/CAISO/ERCOT LMP (÷1000 from $/MWh) under a scenario
region assignment for region-less traces. Carbon intensity = WattTime
scenario midpoint (live 403). GPU price = `afhubbard/gpu-prices` public
list price (`prior`), `prior_fuzzy_match` for nearest-family fallback.

## 8. Promotion decision

```
final_status = economic_overlay_ready
ready_for_economic_ml_target_training = True   (goodput/$ coverage 100% ≥ 50%)
carbon_cost_requires_operator_carbon_price_per_kg_usd = True
```

**Is this ready for economic ML target training?** Yes, *offline*:
- `sla_safe_goodput_per_dollar` and `estimated_gpu_cost_usd` are
  computed on 100 % of 187,505 rows; decode/prefill on 68 %; cache
  value on 32 %; energy/carbon on 23 %.
- Targets are deterministic and fully traceable (per-field
  `value_quality` + `formula`).
- **Caveat (binding):** every $-term is a public-data prior / scenario,
  NOT pilot truth. A model trained on these targets must be validated
  against operator telemetry before any production decision. The
  energy term carries a scenario region assignment; carbon cost is
  uncomputable without an operator carbon price.

Not `shadow_ready_for_integration_review`: the overlay is additive and
does not change candidate ranking (0.0 change rate), and the energy
slice depends on scenario-region assumptions.

## 9. What remains pilot-only

Unchanged from `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md §10` and v1 §11:
- Operator fleet-actual per-GPU `$/hr` (public list price is a `prior`).
- Operator `energy_price_per_kwh_usd` tariff (market LMP ≠ contract).
- Operator `carbon_price_per_kg_usd` (carbon cost stays missing).
- Real per-request `cache_hit` (SwissAI reuse_percentage is a proxy).
- Cold-start latency per (model, GPU, cluster) — 0 % coverage here.
- Migration cache-loss seconds per (model, GPU) — 0 % coverage here.
- Operator region / zone for a real (not scenario) energy/carbon join.
- `memory_pressure_pricing_policy`.

## 10. What can / cannot be claimed externally

**CAN:** "Aurelius computes per-record SLA-safe goodput/$ over a
187 k-row public-data overlay spanning CARA / SwissAI / Optimum /
AcmeTrace / CC-traces, joined to live PJM + CAISO + ERCOT energy
markets and a 12-cloud GPU price index, with every value labelled
measured / derived / prior / scenario / missing." "The corpus is large
and traceable enough to train economic ML targets offline."

**CANNOT:** production cost savings; operator invoice / chargeback
truth; carbon cost from public data; region-specific routing claims
(energy is scenario-region); any claim that the overlay re-ranks
candidates (it does not).

## 11. Reproducibility

```bash
# 1. Bounded raw fetch (gitignored output, ~1 GiB):
HF_TOKEN=… python3 scripts/fetch_economic_overlay_analysis_sources.py
# 2. Live market + GPU-price overlays (creds from env; no values logged):
PJM_API_KEY=… ERCOT_API_KEY=… ERCOT_USERNAME=… ERCOT_PASSWORD=… \
  python3 scripts/fetch_economic_overlay_market_data.py
# 3. Build the 187k-row overlay corpus + summary:
python3 scripts/build_economic_overlay_analysis.py
# 4. A-H eval:
python3 scripts/run_economic_overlay_analysis_eval.py
# 5. Tests:
pytest tests/test_economic_overlay_analysis.py tests/test_economic_overlay_*.py -q
```

## 12. Tests

`tests/test_economic_overlay_analysis.py` (24 tests) proves: no secret
literals or `hf_`/`Bearer` tokens in any committed file; scripts read
credentials from env only; no invented constants in the analysis
scripts; energy + carbon are labelled scenario_prior; the market
manifest records all four providers' success/failure with known status
values; ≥2 energy markets measured; total rows ≥ 50 k and CARA +
SwissAI each ≥ 50 k; subgroup counts reported across ≥4 GPU types;
economic coverage non-zero; carbon cost is operator-policy-only (0
rows); the A–H eval ran all 8 variants with a known promotion state, no
oracle/FIFO headline, three classes reported separately, and the
refined classifier does not collapse everything to scenario_prior; raw
+ full corpus stay gitignored; committed samples bounded ≤100 MB/file,
≤300 MB total. The v1 suite
(`tests/test_economic_overlay_{sources,formulas,joining,eval}.py`, 66
tests) remains green after the additive classifier + perf changes.
