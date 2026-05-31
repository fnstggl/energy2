# Eval Workload Frontier v1 + Batch Inference Frontier v1 — Results

> **Simulator / public-trace evidence only — directional, NOT production
> savings** (`docs/RESULTS.md` §8). Both frontiers are **opt-in**, **shadow
> only**, with `executable_in_real_cluster=False` at construction. The
> serving Safe Utilization Frontier Controller, the Dynamic Safe Frontier
> Estimator v1, the Dynamic Serving Frontier Calibration harness, the
> Training Safe Utilization Frontier v1, the robust energy engine, and
> every committed serving / training / residency benchmark artifact are
> **unchanged**. **No oracle / clairvoyant baseline is used as a headline.**

These are the first two builds out of the
`docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md` ranking — Eval (alpha 5, feas 4)
and Batch Inference (feas 5, alpha 4). Both ship as siblings of the
existing serving rho controller (do **not** import or mutate it) and the
Training Safe Utilization Frontier (`docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`).

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`,
  `docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md`,
  `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`,
  `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`,
  `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`.

## 1. Scope (binding)

- **Opt-in.** No default scheduler / controller is changed. The serving
  rho controller's `constraint_aware` default rho (0.65) is **unchanged**
  byte-for-byte (`tests/test_batch_inference_frontier.py::test_serving_rho_default_unchanged`).
- **Shadow only.** `EvalWorkloadFrontierDecision.executable_in_real_cluster`
  and `BatchInferenceFrontierDecision.executable_in_real_cluster` are
  `False` at construction; constructing with `True` raises.
- **Sibling, not subclass.** `aurelius/frontier/eval_workload_*.py` and
  `aurelius/frontier/batch_inference_*.py` do NOT import the serving rho
  controller modules (`controller.py`, `dynamic_controller.py`,
  `estimator.py`, `dynamic_estimator.py`). Tests assert the import ban.
- **Synthetic deadlines, labelled.** Eval-class traces (ShareGPT) and
  serving traces (Azure LLM 2024) carry **no real deadline**. The deadline
  knob is a *synthetic scenario*; every workload profile carries an
  explicit `synthetic_scenario_label`.
- **Char-derived token estimates are PROXIES.** The ShareGPT ingester
  reports `token_count_source = "char_div_4_proxy"` on every record. The
  eval-frontier estimator uses these as the work-volume input and labels
  the result as a proxy in the summary JSON.
- **Bounded ingest only.** The ShareGPT raw download is HTTP-Range-capped
  (default 50 MB) — the same bounded-ingest pattern that
  `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md` established.
- **No production claim.** The `docs/RESULTS.md` §8 production-claim gate
  has NOT been satisfied for either frontier. No production savings number
  is allowed, and the summary JSONs assert
  `production_claim=false / ml_training=false /
  modifies_serving_rho_controller=false / uses_oracle_as_headline=false /
  executable_in_real_cluster=false`.

## 2. Phase A — Bounded ingestion

### 2.1 ShareGPT (eval-class)

- **Source:** `https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/old/sg_52k.json`
- **Bounded HTTP-Range cap:** default 50 MB; configurable via `--max-bytes`.
- **Raw schema (verified, NOT assumed):**
  `[{"id": str, "conversations": [{"from": "human|gpt|chatgpt|system",
  "value": str}, ...]}, ...]`. The ingester REJECTS unknown record / turn
  keys (`tests/test_sharegpt_aiperf_ingest.py::test_unknown_record_key_rejected`).
- **Normalization target:** `aurelius/traces/eval_schema.py::EvalWorkloadRequest`
  (a separate contract from `NormalizedLLMRequest` — ShareGPT has no
  timestamp, no model id, no real tokens, so forcing it into the serving
  contract would mean inventing fields).
- **Missing signals (preserved as `None`):** `timestamp_s`, `model_id`,
  `language`, `prompt_tokens_real`, `response_tokens_real`, `e2e_latency_s`,
  `deadline_s`.
- **Derived/proxy:** `prompt_tokens_est = round(prompt_chars / 4)`,
  `response_tokens_est = round(response_chars / 4)`, `turn_count`,
  `role_sequence_signature`. Every record is tagged
  `token_count_source = "char_div_4_proxy"`.
- **Honesty flag:** every record carries `provenance =
  "sharegpt_52k_head_sample_v1"`.
- **Fixture:** `tests/fixtures/sharegpt_aiperf_sample/sg_head_fixture.json`
  (3 hand-crafted records; the ingester / summary tests pin the schema
  against it).
- **Processed summary:** `data/external/sharegpt_aiperf/processed/sharegpt_aiperf_ingest_summary.json`
  (≤ 100 MB user-spec cap; current 500-record sample is ≤ ~600 KB).
- **Raw head sample:** `data/external/sharegpt_aiperf/raw/sg_52k_head.json`
  — **gitignored**; only the bounded-download manifest is committed.

### 2.2 LMSYS Chatbot Arena (eval-class, gated)

- **Source:** `https://huggingface.co/datasets/lmsys/chatbot_arena_conversations`
  — **gated** (`gated: auto` on the HF API). Auto-download requires
  accepting the LMSYS terms-of-use AND supplying `HF_TOKEN`.
- **Status: BLOCKED_GATED_DATASET.** `aurelius/traces/lmsys_chatbot_arena.py::download_gated`
  REFUSES to proceed without a token (`LMSYSGatedAccessError`); the script
  exits non-zero with the gated-access banner. Tests pin this behavior
  (`tests/test_lmsys_chatbot_arena_ingest.py::test_cli_no_token_exits_nonzero`).
- **Schema fields (transcribed from the dataset card):** `question_id`,
  `model_a`, `model_b`, `winner`, `judge`, `conversation_a/b`, `turn`,
  `language`, `tstamp`, `openai_moderation`, `toxic_chat_tag`. The
  ingester's `normalize_record(rec, side="a"|"b")` maps one row + one side
  onto `EvalWorkloadRequest` — when a user supplies a local file the
  adapter works without changes.

### 2.3 Azure Functions 2019

- **Status: DEFERRED_BOUNDED_INGEST.** The dataset is multi-file and
  ~25 GB raw — too large to bounded-ingest for the v1 use case. Will be
  revisited when an ETL / embedding-fan-out frontier is built. URL noted
  in the dataset registry; no script committed.

### 2.4 Azure LLM 2024 (already committed)

The existing committed `tests/fixtures/azure_llm_2024_sample.csv`
(5,880 rows, 1,560 ticks @ 60 s) is REUSED as the synthetic batch-flex
scenario backing the Batch Inference Frontier v1. The full week-long file
(~44.1 M rows, 9 days) is not re-ingested in this PR.

## 3. Phase A acceptance — Azure 2024 deadline-slack sanity sweep

The user-spec Phase A acceptance gate is the sanity question:

> Is there a visible deadline-slack-vs-rho slope? Does higher rho improve
> goodput/$ before deadline misses explode? Does the result remain positive
> under conservative assumptions?

Run on the committed Azure 2024 sample fixture (5,880 requests, 100×
time-rescale to the audit's primary busy tier; mean RPS ≈ 6.1):

| target rho | deadline_slack_s = 0 | slack = 60 s | slack = 300 s | slack = 900 s | slack = 3600 s |
|---|---|---|---|---|---|
| 0.45 | UNSAFE (100% miss) | SAFE goodput/$ ≈ 919,514 | SAFE | SAFE | SAFE |
| 0.55 | UNSAFE (100% miss) | SAFE goodput/$ ≈ 1,199,167 | SAFE | SAFE | SAFE |
| 0.65–0.95 | UNSAFE | SAFE goodput/$ ≈ 1,199,167 | SAFE | SAFE | SAFE |

**Verdict (gate passes):**

1. **deadline-slack-vs-rho slope is visible.** At `slack=0` every candidate
   is UNSAFE (deadline_miss_rate = 100 % since the existing serving SLA
   already mandates a TTFT + per-token budget that exceeds the
   "no-slack" budget). At `slack ≥ 60 s` the frontier flips to SAFE.
2. **Higher rho improves goodput/$ before deadline misses explode.**
   Between rho=0.45 and rho=0.55, goodput/$ jumps **+30.4 %**
   (919,514 → 1,199,167) with deadline_miss_rate staying at 0 %. Above
   rho=0.55 the existing serving-replay machinery hits the trim ceiling
   (constraint_trim caps the effective replica count when SLA is met) —
   that is *consistent* with the committed serving frontier evidence that
   the safe peak on this trace is around rho ≈ 0.85 at the audit's
   load multiplier.
3. **Result is positive under conservative assumptions** (anticipatory
   mode, prefill_savings=0, the standard interactive SLA budget, the
   pre-registered safety ceilings deadline_miss ≤ 2 % / timeout ≤ 10 % /
   queue p99 ≤ 2000 ms).

The sanity sweep is pinned by
`tests/test_batch_inference_frontier.py::test_azure_2024_phase_a_sanity_deadline_slack_slope`,
so the gate cannot silently regress.

## 4. Phase B — Eval Workload Frontier v1

### 4.1 Architecture

```
EvalWorkloadProfile
   (workload_id, trace_source, synthetic_scenario_label,
    dedicated_fleet, deadline_*, interactive_baseline_*)
                │
                ▼
EvalWorkloadFrontierCandidate
   (eval_batch_window_hours, concurrency, target_rho,
    deadline_slack_hours, dedicated_fleet)
                │
                ▼
estimate_eval_workload_frontier
   (deterministic structural model:
    fleet tokens/s = concurrency × per_replica × R × eff(R);
    completion_h = total_tokens / fleet_tokens_per_s;
    gpu_hours = concurrency × completion_h;
    cost = gpu_hours × ($_per_h + power_kW × $_per_kWh);
    deadline_miss_pct = fraction of per-request projected times
                        exceeding deadline_slack budget.)
                │
                ▼
EvalWorkloadFrontierPoint (KPI + deadline_miss + completion_h +
                           interactive deltas + categorical safety)
                │
                ▼
EvalWorkloadSafetyConfig + classify_eval_point_safety
   - max_deadline_miss_rate_pct (default 1.0)
   - max_eval_suite_completion_hours
   - enforce_mixed_fleet_veto: candidate w/ dedicated_fleet=False
     REQUIRES interactive_baseline_p99_ms + timeout_pct on the profile;
     missing → INSUFFICIENT_TELEMETRY; predicted deltas above
     tolerance → UNSAFE.
                │
                ▼
choose_eval_workload_frontier_target
   (highest-goodput safe point; mixed-fleet veto → ISOLATE_FROM_INTERACTIVE;
    UNSAFE current → LOWER_EVAL_CONCURRENCY; deadband → KEEP)
                │
                ▼
EvalWorkloadFrontierDecision
   (recommendation-only; executable_in_real_cluster=False at
    construction; execute shim is shadow-only by default)
```

### 4.2 Modules

| file | role |
|---|---|
| `aurelius/traces/eval_schema.py` | `EvalWorkloadRequest`, `chars_to_token_estimate`, `role_sequence_signature`, `EvalWorkloadSummary` |
| `aurelius/traces/sharegpt_aiperf.py` | bounded HTTP-Range ingester + lenient JSON parser + `normalize_record` |
| `aurelius/traces/lmsys_chatbot_arena.py` | gated-download helper + `normalize_record` adapter |
| `aurelius/frontier/eval_workload_models.py` | `EvalWorkloadProfile`, `EvalWorkloadFrontierCandidate`, `EvalWorkloadFrontierPoint`, `EvalWorkloadFrontierDecision` + categorical enums |
| `aurelius/frontier/eval_workload_safety.py` | `EvalWorkloadSafetyConfig` + `classify_eval_point_safety` |
| `aurelius/frontier/eval_workload_estimator.py` | structural model + `estimate_eval_workload_frontier` |
| `aurelius/frontier/eval_workload_controller.py` | `choose_eval_workload_frontier_target` + shadow-only `execute_eval_workload_frontier_decision` (stub for real mode) |
| `scripts/ingest_sharegpt_aiperf.py` | bounded ingest CLI |
| `scripts/ingest_lmsys_chatbot_arena.py` | gated ingest CLI (`HF_TOKEN` required) |
| `scripts/run_eval_workload_frontier.py` | shadow-mode sweep driver |

### 4.3 v1 sweep — ShareGPT bounded sample

- **Source:** `data/external/sharegpt_aiperf/raw/sg_52k_head.json` (50 MB
  HTTP-Range bounded download, 500 records loaded into the estimator).
- **Candidate grid:** rho ∈ {0.55, 0.65, 0.75, 0.85, 0.95} ×
  concurrency ∈ {1, 2, 4, 8} × deadline_slack_hours ∈ {0.5, 1, 4, 24} =
  80 candidates.
- **Profile:** `synthetic_scenario_label = "sharegpt_eval_overnight_v1"`,
  `dedicated_fleet=True`, `deadline_slack_hours_baseline=4`,
  `deadline_miss_rate_sla_pct=1`, `eval_suite_completion_deadline_hours=24`,
  `telemetry_confidence="low"`.

**Recommendation (this run):** `RECOMMEND_EVAL_FRONTIER` at
rho=0.95, concurrency=1, slack_h=0.5, dedicated_fleet=True; predicted
goodput/$ ≈ 4.11 M, predicted completion_h ≈ 0.26, predicted
deadline_miss_rate = 0.0 %. **Eval workloads tolerate hours of slack** —
the safety floor stays inside the eval-suite completion deadline at every
candidate row. The structural model is CONSERVATIVE (it uses the same
public-benchmark per-replica throughput prior the canonical backtest uses
and a saturating efficiency model that maxes at 1.0).

**Mixed-fleet veto sanity:** the same sweep run with `dedicated_fleet=False`
AND no `interactive_baseline_p99_ms / timeout_pct` on the profile yields
`INSUFFICIENT_TELEMETRY`; with baselines present and rho=0.95 the
structural interactive-degradation model raises a hard UNSAFE veto. Both
behaviors are pinned by
`tests/test_eval_workload_frontier.py::test_mixed_fleet_without_baselines_is_insufficient_telemetry`
and `..._high_rho_unsafe_with_baselines`.

## 5. Phase B — Batch Inference Frontier v1

### 5.1 Architecture

```
BatchInferenceWorkloadProfile
   (workload_id, trace_source, synthetic_scenario_label,
    deadline_*, queue_*, interactive_baseline_*)
                │
                ▼
BatchInferenceFrontierCandidate
   (batch_window_seconds, batch_concurrency, target_rho,
    deadline_slack_seconds)
                │
                ▼
estimate_batch_inference_frontier
   (reuses UNCHANGED serving physics in aurelius/traces/backtest.py:
    Erlang-C wait + tail multipliers + decomposed TTFT/TPOT +
    engine timeout formula. Sizers locally re-implement the reactive
    and EWMA-anticipatory replay so we do not import the serving rho
    controller; deadline_miss_rate = fraction of ticks where the
    predicted p99 latency exceeds the candidate's deadline budget.)
                │
                ▼
BatchInferenceFrontierPoint
                │
                ▼
BatchInferenceSafetyConfig + classify_batch_point_safety
   - max_deadline_miss_rate_pct (default 2.0)
   - max_timeout_pct (default 10.0, sibling of serving)
   - max_queue_p99_ms (default 2000 ms, sibling of serving)
   - enforce_interactive_baseline_floor: queue_p99 + timeout must stay
     within the interactive baseline + tolerance when present.
                │
                ▼
choose_batch_inference_frontier_target
   (highest-goodput safe point; UNSAFE current →
    LOWER_BATCH_PRESSURE; deadband → KEEP)
                │
                ▼
BatchInferenceFrontierDecision
   (recommendation-only; executable_in_real_cluster=False at
    construction; execute shim is shadow-only by default)
```

### 5.2 Modules

| file | role |
|---|---|
| `aurelius/frontier/batch_inference_models.py` | profile / candidate / point / decision + categorical enums |
| `aurelius/frontier/batch_inference_safety.py` | `BatchInferenceSafetyConfig` + `classify_batch_point_safety` |
| `aurelius/frontier/batch_inference_estimator.py` | wraps `aurelius/traces/backtest.py` physics with local sizers + deadline-miss accounting |
| `aurelius/frontier/batch_inference_controller.py` | `choose_batch_inference_frontier_target` + shadow-only execute stub |
| `scripts/run_batch_inference_frontier.py` | shadow-mode sweep driver |

### 5.3 v1 sweep — Azure 2024 batch-flex scenario

- **Source:** `tests/fixtures/azure_llm_2024_sample.csv` (5,880 rows,
  100× time-rescale, 60 s ticks). Same fixture and same load multiplier
  the committed Dynamic Safe Frontier Estimator audit uses
  (`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`).
- **Candidate grid:** rho ∈ {0.45..0.95 in 0.10 steps} ×
  deadline_slack_seconds ∈ {0, 30, 60, 300, 900, 3600} = 36 candidates.
- **Profile:** `synthetic_scenario_label =
  "azure_llm_2024_batch_flex_scenario_v1"`,
  `deadline_miss_rate_sla_pct=2`, `queue_wait_sla_p99_ms=2000`,
  `telemetry_confidence="medium"`.

**Recommendation (this run):** `RECOMMEND_BATCH_FRONTIER` at
rho=0.55, deadline_slack=30 s; predicted goodput/$ ≈ 1,199,167; predicted
deadline_miss_rate = 0 %; queue p99 ≈ 343 ms.

The selected rho (0.55) reflects the fact that the existing
`_constraint_trim` machinery in the serving replay aggressively trims
replicas downward once SLA is safe — so the "effective" rho at scaled
load is roughly identical above rho=0.55 (the serving replay has the
final say on replica count). Above this point goodput/$ does not improve
materially; below it the slope is steep. This is consistent with the
committed Azure 2024 dynamic-frontier audit
(`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`).

**Sanity gate (pinned in tests):** at deadline_slack=0 every candidate is
UNSAFE; at deadline_slack=60 s at least one candidate is SAFE; goodput/$
at rho=0.55 is ≥ goodput/$ at rho=0.45. See
`tests/test_batch_inference_frontier.py::test_azure_2024_phase_a_sanity_deadline_slack_slope`.

## 6. Honesty / negative results

- **The eval-class frontier's per-candidate KPI is a STRUCTURAL prediction**
  (transparent per-replica throughput × R × efficiency), NOT a calibrated
  measurement. The committed processed summary and the run script label
  this explicitly. Calibration against pilot telemetry is the next step
  before any production claim — `docs/RESULTS.md` §8 is not yet satisfied.
- **The batch frontier's goodput/$ plateau above rho ≈ 0.55** at the
  Azure-2024 100× sample load is a known artifact of the existing
  `_constraint_trim` machinery; it is NOT new alpha relative to the
  committed serving rho controller. The new value comes from the
  **deadline-flexibility scenario** (slack ≥ 60 s lets every candidate
  stay SAFE under deadline-miss accounting), not from higher rho per se.
- **ShareGPT token counts are char/4 PROXIES.** Reports MUST label any
  derived goodput/$ as a proxy-grade number.
- **LMSYS Chatbot Arena is gated.** No automatic ingest is performed; the
  ingester refuses to proceed without `HF_TOKEN`.
- **Azure Functions 2019 is DEFERRED.** No bounded ingest is performed.
- **No oracle / clairvoyant baseline is used as a headline.** Neither
  v1 frontier consults a future-aware estimator; both are deterministic
  over the candidate grid.
- **No production savings claimed.**

## 7. Tests (pin invariants)

- `tests/test_sharegpt_aiperf_ingest.py` — fixture schema + unknown-key
  rejection + truncated-JSON head-parse + proxy-token labelling +
  missing-field honesty + processed-summary size guard.
- `tests/test_lmsys_chatbot_arena_ingest.py` — gated-refusal + banner
  contents + schema-field pinning + `normalize_record` mapping + CLI
  non-zero exit without `HF_TOKEN`.
- `tests/test_eval_workload_frontier.py` — no-serving-controller-import
  ban + enum/range validation + empty-set → INSUFFICIENT_TELEMETRY +
  highest-goodput safe-point selection + UNSAFE current →
  LOWER_EVAL_CONCURRENCY + mixed-fleet veto (missing baselines →
  INSUFFICIENT, present + high-rho → UNSAFE, isolate when all unsafe are
  mixed-fleet) + deadline-miss cap → UNSAFE + decision
  `executable_in_real_cluster=False` + execute shim shadow-only.
- `tests/test_batch_inference_frontier.py` — same invariants as above PLUS
  the Azure 2024 deadline-slack sanity gate AND the
  `ca_target_rho = 0.65` byte-for-byte serving-default-unchanged check.

## 8. What this PR does NOT do

- Does not change any default scheduler or controller.
- Does not enable real cluster execution.
- Does not claim production savings.
- Does not use an oracle / clairvoyant baseline as a headline.
- Does not ingest LMSYS Chatbot Arena automatically (gated).
- Does not ingest Azure Functions 2019 (deferred).
- Does not introduce ML training.
- Does not modify the robust energy engine.
- Does not modify committed serving / training / residency / Azure 2024
  Dynamic Frontier Calibration artifacts.
