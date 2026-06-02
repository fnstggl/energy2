# Frontier Signal Bounded Ingestion v1

> **Shadow / research ingest.** No production scheduler / scorer / residency /
> frontier / overlay behavior is changed. No real execution. No production
> savings claimed. No invented constants. Public / artifact data is **never**
> treated as pilot telemetry. FaaS cold-start is **never** silently converted
> into GPU model-load; autoscaling proxies are **never** silently converted
> into measured serving autoscaling. Raw downloads are gitignored; only bounded
> derived numeric samples + processed JSON are committed.
>
> **Read first:** `docs/FRONTIER_DISCOVERY_AUDIT_V2.md`,
> `docs/ECONOMIC_ML_ALPHA_V1.md`, `docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`,
> `docs/HF_DATASET_REGISTRY.md`.
>
> **Evidence:** `data/external/frontier_ingest_v1/{source_audit,signal_strength,
> ingest_manifest}.json`, `data/external/frontier_signals/<source>/processed/*`.
> **Companion:** `docs/ECONOMIC_ML_ALPHA_FRONTIER_V1.md`.

## 1. Mission

Bounded-ingest the three highest-value Frontier Discovery v2 next-ingest
candidates and normalize each into a **separate canonical trace role** — then
refresh the Economic ML Alpha audit (companion doc).

| # | Source | Role | Main question |
|---|---|---|---|
| 1 | **Mooncake** (FAST'25) | `cache_residency_trace` | Does `cache_reuse_pct` remain shadow-ready beyond SwissAI? |
| 2 | **Huawei FaaS 2025** (EuroSys'25) | `cold_start_prior_trace` | Can public FaaS cold-start breakdowns improve simulator priors *without* pretending they are GPU model-load? |
| 3 | **Alibaba GPU v2025** (NSDI'25) | `autoscaling_queue_proxy_trace` | Can this add useful autoscaling / queue-risk *proxy* labels? |

## 2. Phase 0 — source verification

`source_audit.json`. All three verified, schema-confirmed, classified, and
deemed bounded-ingest-safe.

| Source | License | Class | Bounded ingest |
|---|---|---|---|
| Mooncake | repo (citation-requested); values are author-anonymized integers, no raw text | `workload_only + derived_reuse_structure` | full (4 files, ~13 MB, under 100 MB cap) |
| Huawei FaaS 2025 | **CC BY 4.0** (attribution) | `measured (FaaS cold-start)` | **bounded slice** — see §4 |
| Alibaba GPU v2025 | clusterdata terms (citation) | `proxy (instance-lifecycle)` | full (2.1 MB CSV) |

## 3. Phase 1–2 — exact schemas + normalization

### 3.1 Mooncake → `cache_residency_trace`

Raw JSONL row: `{timestamp:int(ms), input_length:int, output_length:int,
hash_ids:[int…]}` (block_size 512). **No** model_id / session_id / request_id /
measured latency / measured cache-hit. Reuse is **derived** by simulating a
global (infinite) prefix cache in arrival order: `cache_reuse_pct = reused_blocks
/ total_blocks × 100`, `high_reuse = cache_reuse_pct ≥ 50`, `cache_hit_proxy =
1 if any block reused`. Fields carry `field_quality` (`measured_anonymized` for
timestamps/tokens/blocks; `derived_proxy` for all reuse fields) and an explicit
`limitations` note that this is **not** identical to SwissAI's measured
`reuse_percentage`.

- Normalized rows: **63,240** (conversation 12,031 · synthetic 3,993 · toolagent
  23,608 · arxiv 23,608). Mean derived reuse **57.0%**; high-reuse rate **66.0%**;
  any-reuse **96.5%** — consistent with the paper's "up to 50% cache hit ratio."

### 3.2 Huawei FaaS 2025 → `cold_start_prior_trace`

Raw cold-start schema (verified): `day, time, clusterName, funcName, userID,
requestID(hash), totalCost_cold_start, podAllocationCost, deployCodeCost,
deployDependencyCost, schedulingCost, podID`. Mapped to `cold_start_latency_s`,
`pod_allocation_s`, `deploy_code_s`, `deploy_dependency_s`, `scheduling_s`,
`function_id`, `pool_*` (parsed from podID), `platform=huawei_yuanrong_faas`.
`requestID` hash is **dropped** from the committed sample. Every row carries
`field_quality.for_gpu_llm_cold_start = "prior_proxy_only"` and a `limitation`
that `deploy_code/deploy_dependency` are code/dependency download — **not**
model-weight load.

- Normalized rows: **239,405** cold-start events (+2,418 trigger/runtime rows).
- Measured cost structure (calibration prior): scheduling **51.1%**, deploy_code
  **23.4%**, deploy_dependency **15.5%**, pod_allocation **10.1%** of total;
  cold-start latency p50 **2.33 s**, p90 **9.34 s**, p99 **42.1 s**.

### 3.3 Alibaba GPU v2025 → `autoscaling_queue_proxy_trace`

Raw schema: instance-lifecycle records (`instance_sn, role(CN/HN), app_name,
cpu/gpu/rdma/memory/disk request+limit, max_instance_per_node, creation_time,
scheduled_time, deletion_time`). **No** per-request rows, **no** measured
queue-wait, **no** utilization, **no** failure. Mapped to `scheduler_delay_s =
scheduled_time − creation_time` (**derived proxy**, NOT serving queue-wait),
`instance_lifetime_s`, `gpu_count` (allocation), with `queue_wait_s = utilization
= failure_or_timeout_state = None` and an autoscaling proxy = per-app
create/delete counts (**inferred**, not measured events).

- Normalized rows: **23,871** instances (16,485 CN + 7,386 GPU/HN), 156 apps.
  `scheduler_delay_s` proxy: p50 0 s, p90 59 s, p99 1,650 s (heavy tail).

### 3.4 Per-source processed artifacts

Each source has `processed/{schema_profile, schema_mapping, summary,
statistical_rollups}.json` + a bounded `normalized_sample.jsonl` (≤ 5,000 rows,
≤ 8 MiB; 4.6–5.3 MB committed each). The full normalized
`analysis_sample.jsonl` is **gitignored** (used only by the ML refresh).

## 4. Bounded-ingest policy (binding) + the Huawei exception

Policy: start at 100 MB (or full if smaller), expand to 300 MB only if
high-value and subgroup coverage is weak, **never unbounded**.

- Mooncake (13 MB) and Alibaba (2.1 MB) are ingested **in full** — both under
  the 100 MB start cap.
- **Huawei is the bounded exception.** The smallest cold-start region file is
  `R1.zip` = **467 MB** (> 300 MB cap) and the quantiles zips are 0.9–5.5 GB —
  all over cap and non-range-able. We therefore **did not** download the full
  file. Instead: a bounded **100 MB range-download** of `R1.zip`, then a
  raw-DEFLATE inflate of the first fully-contained ZIP member (`R1/day_28.csv`),
  recovering **239,405 complete cold-start rows** (the truncated tail row is
  discarded). The 117 KB trigger/runtime CSV is ingested in full. Total raw
  bytes downloaded ≈ **120 MB**.

## 5. Signal strength (Phase 3)

`signal_strength.json`. Per source: rows, unique entities, coverage, target
labels, measured/proxy/simulated breakdown, and `suitable_for` flags.

| Source | measured | proxy | absent | suitable_for (honest) |
|---|---|---|---|---|
| Mooncake | timestamps, tokens, block counts | reuse_pct, high_reuse, cache_hit_proxy | cache_hit, model_id, session_id | cache-reuse **proxy** training; cross-dataset validation **limited** |
| Huawei | cold-start cost breakdown (FaaS) | (GPU prior) | GPU model-load | cold-start **simulator prior** only; **no** GPU ML |
| Alibaba | lifecycle, allocations | scheduler_delay, autoscaling-count | queue-wait, utilization, failure | autoscaling/queue **proxy** only |

## 6. Commit policy (enforced by tests)

- Raw files under `data/external/frontier_signals/raw/` — **gitignored**.
- `analysis_sample.jsonl` (full normalized) — **gitignored**.
- Committed: `normalized_sample.jsonl` (≤ 5,000 rows, ≤ 8 MiB) + the four
  processed JSONs + the three `frontier_ingest_v1` JSONs.
- **No** raw prompt/response text exists in any source; **no** secrets/tokens;
  Huawei `requestID` hashes and SwissAI raw `bucket_ids` are **not** committed.

## 7. Reproducibility

```bash
python3 scripts/ingest_frontier_signals_v1.py --download   # bounded download + normalize
python3 scripts/audit_frontier_signal_strength_v1.py       # Phase 3
python3 scripts/regen_swissai_bucket_reuse_samples.py      # SwissAI rows (ML companion)
python3 scripts/run_economic_ml_alpha_frontier_v1.py       # Phase 4 (companion doc)
pytest tests/test_frontier_signal_ingest_v1.py tests/test_frontier_signal_strength_v1.py \
       tests/test_economic_ml_alpha_frontier_v1.py -q
```

## 8. What remains pilot-only

Server-class GPU `model_load_duration_s` (Huawei is FaaS), measured serving
autoscaling events (Alibaba is instance-lifecycle proxy), per-request migration
/ cache-loss seconds, and real per-request measured `cache_hit` (Mooncake reuse
is a derived proxy). These stay `blocked_by_pilot_telemetry`, consistent with
Economic ML Alpha v1 and Frontier Discovery v2.
