# Cache / Prefix-Reuse Forecaster v1 — shadow-only research

> **Shadow-only PR.** No production scheduler, residency, or routing
> behavior is changed. No robust energy engine code is touched. No
> production savings are claimed. HF datasets are not treated as pilot
> telemetry. The CC-traces 3000 MiB strength expansion is committed only
> as a small normalized sample + summary; the 2.7 GB raw download is
> gitignored.
>
> **Read first:**
> - `docs/FORECAST_LEVERAGE_AUDIT.md` (engine ranking)
> - `docs/HF_DATASET_REGISTRY.md` (trust hierarchy + canonical trace types)
> - `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md` (Tier-2 telemetry baseline)
> - `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md` (gap-ingest audit)
> - `docs/CARA_QUEUE_FORECASTER_RESULTS.md` (analogous shadow forecaster)
> - `docs/PLACEMENT_PRIOR_AUDIT.md` (the scoring-path gap this forecaster
>   would integrate against)

## 1. Mission + non-goals

**Goal.** Determine whether a cache / prefix-reuse forecast improves
Aurelius's economic decisions for cache-aware routing, residency,
migration veto, and cold-start avoidance — not just predictive accuracy.

**Non-goals (binding).**

- No scheduler / residency / routing behavior change.
- No real execution. The shadow adapter, when wired in a future PR, must
  default to logging-only.
- No oracle as headline. Comparisons use the strongest realistic
  baseline (`per_model_reuse_rate`).
- HF data is benchmark / public-trace, never pilot telemetry.
- CC-traces 3000 MiB raw data is never committed.

## 2. Datasets used

| Dataset | Config(s) | Rows used | License | Trust tier | Role |
|---|---|---:|---|---|---|
| `eth-easl/swissai-serving-trace` | `qwen3_32b_bucket_reuse`, `qwen380b_instruct_bucket_reuse`, `qwen380b_thinking_bucket_reuse`, `llama3_70b_bucket_reuse`, `apertus_70b_bucket_reuse` | 67,190 (bounded 10 MiB head per config) | other (SwissAI research) | Tier 4 | **Primary training** — `reuse_percentage` labels |
| `semianalysisai/cc-traces-weka-no-subagents-051226` | `traces_3000mib` (Phase 0 expansion) | 136,118 (3000 MiB bounded raw, 5,000 row committed normalized sample) | apache-2.0 | Tier 5 | Training+validation for `intra_session_reuse` derived label (decision: `use_for_training`, see §3) |
| `sammshen/lmcache-agentic-traces` | `train_shard4` | 4,976 (committed normalized sample) | mit | Tier 5 | Cross-dataset structural prior only — no reuse label |
| `jaytonde05/prefixbench` | `prefixbench_all` | 5 fixture rows | unspecified | Tier 4 | Synthetic prior only — never headline |

The bounded 10 MiB SwissAI heads are regenerable via
`scripts/regen_swissai_bucket_reuse_samples.py` (raw + analysis_sample
remain gitignored).

## 3. Phase 0 — CC-traces 3000 MiB strength expansion

Per the mission spec's adaptive expansion rule, a bounded 3000 MiB
sample of `semianalysisai/cc-traces-weka-no-subagents-051226` was
fetched and flattened into per-request rows. Full audit:
`data/external/forecasting/cache_prefix_reuse_v1/cc_traces_strength_expansion.json`.

| Metric | 80 MiB committed | 3000 MiB Phase 0 |
|---|---:|---:|
| Sessions | 7 | **949** |
| Requests | 761 | **136,118** |
| Unique KV block hashes | 758 | **134,862** |
| Repeated KV block hashes | 3 | **912** |
| Intra-session reuse examples | 740 | **129,299** |
| Cache-loss proxy examples | 0 | **1,145** |
| Models covered | 3 | 4 (haiku-4-5, opus-4-6, opus-4-7, sonnet-4-6) |
| Request types covered | `s` (1) | `s`, `n` (2) |
| TTFT coverage | 100% | 98.76% |
| api_time coverage | 100% | 100% |
| think_time coverage | 99.61% | 99.66% |

**Decision: `use_for_training`** — the 3000 MiB sample passes every
gate: unique-hash ratio 177.9× (> 3× threshold), 136,118 requests
(> 2,500 threshold), 912 repeated hashes (> 0), 129,299 intra-session
reuse examples (> 0). CC-traces is now eligible for training, but
because it has no `reuse_percentage` label it is reported separately
from SwissAI / PrefixBench / LMCache headline results (see §5.3).

**Commit policy (enforced by tests).** Raw 2.77 GB `traces.jsonl` is
gitignored. The per-request flattened `analysis_sample.jsonl` (68 MB)
is gitignored. Only the bounded committed `normalized_sample.jsonl`
(2.5 MB / 5,000 rows / cap 8 MiB) and the Phase 0 summary JSON are
committed. No raw prompt or completion text is ever written — only
hashed block-id summaries + numeric metadata.

## 4. Feature contract + leakage rules

`aurelius/forecasting/cache_prefix_features.py` enforces a strict
leakage contract.

### 4.1 Targets

| Target | Type | Definition | Source dataset |
|---|---|---|---|
| `reuse_percentage` | continuous [0,100] | `reused_buckets / total_buckets × 100` | SwissAI bucket_reuse |
| `high_reuse` | binary | `reuse_percentage >= 50%` (threshold pre-registered) | derived from SwissAI |
| `intra_session_reuse` | binary | per-row indicator: same `block_hashes_hash` as the prior turn in the session OR `block_hashes_count` grew without resetting | derived from CC-traces |

### 4.2 Leakage blocklist (`LEAKAGE_TARGET_FIELDS`)

```
reuse_percentage, reused_buckets, reused_bucket_count,
actual_e2e_latency_s, actual_ttft_s, ttft_s, api_time_s,
tpot_s, actual_tpot_s, actual_output_tokens, output_tokens,
cache_hit, prefix_hit, completion_timestamp_s, completion_timestamp
```

A `LeakageError` is raised at feature-spec build time if any of these
appear as a feature. The test
`tests/test_cache_prefix_features.py::test_leakage_blocker_raises`
enforces this on every commit.

### 4.3 Predict-time features (allowed)

| Feature | Source | Field quality |
|---|---|---|
| `bucket_count` (= `total_buckets`) | SwissAI / CC-traces (`block_hashes_count`) | real |
| `input_tokens` | CC-traces / SwissAI | real (decision-time observable) |
| `predicted_output_tokens` (= `max_tokens`) | PrefixBench / SwissAI synthetic | real |
| `turn_index` | CC-traces session position | real |
| `session_turns_so_far` | rolling count of prior turns | derived |
| `session_requests_total` | CC-traces `requests` length | real (session-level constant) |
| `request_arrival_delta_s` | CC-traces per-request gap | real |
| `think_time_s` | CC-traces inter-turn gap | real |
| `pre_gap_s` | LMCache inter-turn gap | real |
| `rolling_per_model_reuse_pct` | running mean over prior rows in chrono order | derived (decision-time-safe) |
| `rolling_per_hash_seen_count` | how many prior rows share this `bucket_ids_hash` | derived (decision-time-safe) |
| `rolling_per_session_mean_block_count` | running mean of block counts in session | derived |
| `rolling_session_last_block_count` | last turn's block count in session | derived |
| `model_id` (one-hot) | all | real |
| `request_type` (one-hot) | CC-traces | real |
| `bucket_size_bin` | pre-registered bins on `bucket_count` | derived |
| `input_token_bin` | pre-registered bins on `input_tokens` | derived |
| `hour_of_day` | parsed from `created_at_iso` | derived |

Pre-registered bin boundaries are constants (never fitted from holdout
data).

Rolling features are computed in ascending `decision_timestamp_s`
order; row *i* only sees rows 0..i-1.

## 5. Baselines + ML candidates

### 5.1 Baselines

| Baseline | Rule |
|---|---|
| `global_reuse_rate` | Predicts the training-set base rate for every row |
| `per_model_reuse_rate` (**strongest realistic**) | Per-`model_id` training-set mean; unseen models fall back to global mean |
| `per_session_history` | Within-session running mean (uses `rolling_per_model_reuse_pct` at predict time) |
| `recency_frequency_seen` | Branches on whether `rolling_per_hash_seen_count >= 1` (binary feature → conditional reuse rate) |
| `prefix_group` | PrefixBench-only — per-`prefix_group` mean reuse |
| `residency_aware_routing` | Existing `aurelius.residency.decision.score_residency_candidate` is used as the integration-target baseline; see §6 for why it cannot today express cache value |

### 5.2 ML candidates

| Model | Target |
|---|---|
| `LogisticReuseClassifier` | `high_reuse` binary |
| `HistGradientBoostingReuseClassifier` | `high_reuse` binary |
| `RandomForestReuseClassifier` (capped at 50k rows for runtime) | `high_reuse` binary |
| `HistGradientBoostingReuseRegressor` | `reuse_percentage` continuous |
| `FallbackToBaselineWrapper` | Wraps any ML model; falls back to a baseline below a confidence threshold |

All ML candidates are research-class until the promotion gates pass.

### 5.3 Holdouts

| Holdout | Status |
|---|---|
| `random_holdout` | always run (decorative; usually overfit-friendly) |
| `time_holdout` | run when timestamps are present (binding for promotion) |
| `holdout_by_model_<model_id>` | run; the held-out model is the alphabetically last (`qwen3_32b`) |
| `holdout_by_session` | run on CC-traces (SwissAI bucket_reuse has no per-row session id) |

The promotion classifier uses **`time_holdout`** as the binding
economic metric (most realistic generalisation test), falling back to
worst-of-`by_model` if `time_holdout` is unavailable. Random-holdout
results are reported but never headline.

## 6. Phase C — economic-decision evaluation (shadow proxy)

The mission spec is explicit:

> The goal is not simply prediction accuracy. The goal is to determine
> whether cache/prefix-reuse forecasts improve Aurelius economic
> decisions.

### 6.1 Predictive metrics

For classification targets: AUROC, AUPRC, Brier score, expected
calibration error. For regression: MAE, RMSE. All metrics are
deterministic (no random seed needed) and reported per holdout per
model.

### 6.2 Shadow economic proxy

Until the residency scorer can express cache value (§6.3), the
forecaster cannot drive the production economic KPI directly. The
shadow economic proxy models two cache-aware routing decisions:

- **Prefill-savings proxy.** Sum of `y_true × routed` where `routed =
  (y_score >= 0.5)`. Improvement vs the strongest realistic baseline =
  100 × (model − baseline) / baseline.
- **Migration-veto FP/FN proxy.** False-positive = veto migration when
  reuse is actually low; false-negative = allow migration when reuse is
  actually high (cache lost).

Both proxies carry `result_quality = "shadow_proxy"` so reports cannot
accidentally claim a measured result.

### 6.3 Why integration is `blocked_by_scorer_limitations`

`aurelius/residency/decision.py::score_residency_candidate` collapses
TTFT + TPOT into a single `_service_time_s` heuristic and does NOT
consume cache reuse, prefill savings, or migration cache-loss value —
see `docs/PLACEMENT_PRIOR_AUDIT.md::scoring_inputs`. Until a future
scorer-side PR adds these hooks, the cache forecaster cannot drive a
residency / routing / migration-veto decision, even if shadow metrics
clear the >5% bar.

Per the mission spec's PHASE C rule, this PR therefore:

1. documents the missing hook (this section + summary.json
   `scorer_limitation_note`),
2. computes a shadow economic proxy only,
3. classifies as `blocked_by_scorer_limitations` if shadow proxy
   exceeds 5% — otherwise the standard
   `diagnostic_only` / `promising_needs_validation` ladder applies,
4. recommends a future scorer-side PR.

### 6.4 Promotion thresholds

| Binding economic improvement | Status |
|---|---|
| < 2% | `diagnostic_only` |
| 2 – 5% | `promising_needs_validation` |
| ≥ 5% AND scorer supports cache value AND no calibration/subgroup regression | `shadow_ready_for_integration_review` |
| ≥ 2% AND scorer does NOT support cache value | `blocked_by_scorer_limitations` |
| leakage detected | `rejected_regression` |

## 7. Headline result (this PR)

The driver
`scripts/run_cache_prefix_reuse_forecaster_v1.py` was run on the full
ingested corpus. Headline numbers from
`data/external/forecasting/cache_prefix_reuse_v1/summary.json`:

| Holdout | Best ML | Economic improvement (vs `per_model_reuse_rate`) |
|---|---|---:|
| `random_holdout` | `hist_gradient_boosting` | +9.14% (decorative) |
| `time_holdout` (**binding**) | `hist_gradient_boosting` | **−2.01%** |
| `holdout_by_model_qwen3_32b` | `hist_gradient_boosting` | −9.69% |

CC-traces results (separate report — derived intra-session-reuse label,
NOT headline):

| Holdout | Best ML | Economic improvement |
|---|---|---:|
| `random_holdout` | `hist_gradient_boosting` | −0.09% |
| `holdout_by_session` | `hist_gradient_boosting` | −0.10% |

**Final status: `diagnostic_only`** — the binding time-holdout shows
the ML model regresses against the strongest realistic baseline.
Random-holdout alpha is overfit to within-distribution structure and
does not generalize.

**Shadow integration is NOT justified** in this PR. The forecaster is
kept as a research artifact + diagnostic prior. Per mission spec
PHASE D, no shadow adapter is created.

## 8. Subgroup audit

The `holdout_by_model_qwen3_32b` holdout registers a Brier-score
regression at the model-level subgroup (the held-out model). No
high-volume `bucket_size_bin` / `input_token_bin` subgroup regression
is reported under the random or time holdouts. Full subgroup audit
lives in
`summary.json::swissai_results.per_holdout[].subgroup_audit`.

## 9. Cross-dataset generalisation

- **SwissAI → SwissAI** (cross-model): −9.69% by-model holdout
  demonstrates the model does NOT generalize across model families —
  the per-model `reuse_percentage` distribution differs sharply
  between qwen3-32b, qwen380b, llama3-70b, and apertus-70b.
- **SwissAI → CC-traces** is not directly comparable: CC-traces lacks
  `reuse_percentage` labels and uses the derived `intra_session_reuse`
  proxy instead. The CC-traces ML candidate underperforms the
  per-model baseline by a fraction of a percent, consistent with the
  baseline already absorbing most of the signal in CC-traces (most
  same-session turns repeat the previous block-hash).
- **PrefixBench → SwissAI** is NOT run as a headline. PrefixBench is
  synthetic and serves only as a generalisation diagnostic; the
  fixture has 5 rows in this PR.

## 10. What remains pilot-only

| Capability | Reason |
|---|---|
| Production cache-hit calibration | No HF dataset provides measured per-request `cache_hit`; SwissAI's `reuse_percentage` is the bucket overlap, not a wall-clock cache hit |
| Cold-start latency forecasting | No HF dataset measures cold-start latency end-to-end |
| Migration-veto closed loop | The residency scorer does not today consume cache value — future scorer-side PR required |
| KV-eviction rate calibration | CARA exposes `kv_evictions_per_s` as a proxy; pilot Prometheus / vLLM exports are the binding source |

## 11. Reproducibility

```bash
# 1. Phase 0 — CC-traces 3000 MiB expansion (downloads + flattens + decides).
python3 scripts/expand_cc_traces_3000mib.py

# 2. Regenerate the bounded SwissAI bucket-reuse heads (5 configs, 10 MiB each).
python3 scripts/regen_swissai_bucket_reuse_samples.py

# 3. Materialize the gitignored CC-traces analysis_sample.jsonl from the
#    cached 3000 MiB raw (needed only if you keep the raw file).
python3 scripts/materialize_cc_traces_analysis_sample.py

# 4. Train + evaluate + write summary.json + data_readiness_audit.json.
python3 scripts/run_cache_prefix_reuse_forecaster_v1.py
```

All raw downloads land under `data/external/hf/<safe_dataset>/raw/` and
are gitignored. The per-request `analysis_sample.jsonl` files are
gitignored. Committed normalized samples ≤ 50 MB each; total committed
across this PR ≤ 150 MB.

## 12. Audit checklist

- [x] No scheduler / residency / routing controller behavior changed.
- [x] No robust energy engine code touched.
- [x] No production claim made.
- [x] No oracle as headline.
- [x] HF data not treated as pilot telemetry.
- [x] CC-traces 3000 MiB raw NOT committed (gitignored).
- [x] CC-traces `analysis_sample.jsonl` NOT committed (gitignored).
- [x] Committed normalized samples ≤ 8 MiB each; ≤ 50 MB committed per
      dataset; total ≤ 150 MB across this PR.
- [x] No raw prompt / completion text committed.
- [x] Leakage blocklist enforced at feature-spec build time.
- [x] Time-holdout is the binding promotion metric.
- [x] Random-holdout is decorative (never headline).
- [x] CC-traces results reported separately from SwissAI / PrefixBench /
      LMCache because CC-traces uses a derived label.
