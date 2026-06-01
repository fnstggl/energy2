# HF Economic-Signal Discovery Audit

> **Discovery / audit-only PR.** No training, no scheduler / scorer
> behaviour change, no raw-data ingest, no production claim. `HF_TOKEN`
> is read from the environment and never written to any committed
> artefact. Tier-1 pilot telemetry remains the only production
> calibration source.
>
> **Read first:**
> - `docs/HF_DATASET_REGISTRY.md` (trust hierarchy + canonical trace types)
> - `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md` (operational-gap audit)
> - `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md` (which $-coefficients are
>   operator-policy-only)
> - `docs/FORECAST_LEVERAGE_AUDIT.md`
> - `data/external/hf_discovery/economic_signal_discovery_audit.json`
>   (machine-readable audit — every claim below has a JSON key)

## 0. Scope

This audit answers a single question:

> Is there a public Hugging Face dataset that combines **operational
> AI-infra behaviour** (queue / TTFT / TPOT / throughput / GPU
> utilization / power / cache reuse / autoscaling / placement) with
> **economic signals** (GPU-hour price / cloud cost / billing /
> energy per request / kWh / carbon intensity)?

If a paired (A) + (B) dataset exists, the constraint-aware scorer
(`docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md`) could replace operator-
policy `$/hr` with measured public pricing. If not, the gap stays
operator-policy-only and the headline `diagnostic_only` status from
PR #139 stands.

## 1. Method

- Live HuggingFace dataset API only (`/api/datasets`), called via
  stdlib `urllib`. `HF_TOKEN` sent in the `Authorization` header; no
  query-string tokens; no token text written anywhere.
- 90 search terms total (40 primary + 17 combination + 33 single-word
  probes — see `data/external/hf_discovery/economic_signal_discovery_audit.json::search_terms`).
- For each candidate: fetched the `/api/datasets/{id}` detail card, the
  raw README (best-effort, 128 KiB cap), `cardData.dataset_info`
  features, and the siblings list. No file content was downloaded.
- Signals classified with context-aware regex: economic terms only
  count when the haystack also matches a compute / AI / datacenter
  context token (`gpu`, `cluster`, `aws`, `caiso`, `codecarbon`, …).
  This drops the dominant false positive — banking "chargeback" /
  invoice-OCR / utility "kwh billing" datasets — that any uncontextual
  search picks up.
- A small `SEED_SIGNAL_OVERRIDES` table re-asserts the verified
  operational + economic signals for the seeds we already ingested
  (`Qinghao/AcmeTrace`, `optimum-benchmark/llm-perf-leaderboard`,
  `ejhusom/llm-inference-energy-consumption`,
  `memoriant/dgx-spark-kv-cache-benchmark`, plus the two new pricing
  datasets) when the HF card is too thin for the text detector. The
  override is recorded per-candidate (`seed_override_applied` +
  `seed_override_evidence`) so the audit stays auditable.

Driver: `scripts/discover_hf_economic_signals.py`. Output:
`data/external/hf_discovery/economic_signal_discovery_audit.json`.

## 2. Headline

| Question | Answer |
|---|---|
| Search terms run | **90** |
| Total unique HF datasets surfaced | **143** |
| Datasets inspected (card + README + schema) | **143** |
| Paired ops+pricing datasets (A_PLUS) | **0** |
| Ops + energy/kWh (A) | **2** (already in registry) |
| Economics-only joinable (B) | **8** (2 are new high-value finds) |
| Ops-only (C) | **2** (already in registry) |
| Synthetic-economics rejects (D) | 0 |
| Irrelevant (F) | 131 |
| `HF_TOKEN` leaked into any committed file | **No** |
| Raw dataset content committed | **No** |

**Headline finding.** No public HF dataset combines real GPU-hour
pricing with operational request/queue/latency telemetry in the same
record. The strongest new economic addition is
`afhubbard/gpu-prices` — cross-cloud GPU rental snapshots from 12+
providers (AWS / GCP / Azure / Lambda Labs / RunPod / Vast.ai / …),
CC-BY-4.0, twice-daily, with `(timestamp, provider, gpu_type, region,
price_per_hour, is_spot)`. It is a **join-overlay candidate**, not a
standalone calibration source.

The `optimum-benchmark/llm-perf-leaderboard` and
`ejhusom/llm-inference-energy-consumption` paths to A-class were
already ingested in PRs #135 and #138 — they remain the only public
HF sources of measured per-request energy + latency at any
meaningful row count.

## 3. Special audit (mission spec §"Special audit")

1. **Did we find any public dataset with real GPU-hour pricing plus
   request/queue/latency telemetry?**  
   **No.** `afhubbard/gpu-prices` and `labofsahil/aws-pricing-dataset`
   are pricing-only — no operational telemetry. `Qinghao/AcmeTrace`
   has queue / utilization / IPMI power but no pricing. The two are
   **joinable** by `gpu_type` and `timestamp` but live in different
   datasets.
2. **Did we find any public dataset with cloud billing/chargeback
   plus workload telemetry?**  
   **No.** Every chargeback/invoice-OCR candidate that survived the
   compute-context filter was a banking or telecom dataset, not a
   cloud-compute one.
3. **Did we find any public dataset with energy per request plus
   GPU/model latency?**  
   **Yes — two, both already in the registry:**
   - `optimum-benchmark/llm-perf-leaderboard` — measured prefill
     (TTFT) + decode (TPOT) p50/p90/p95/p99 + per-request CodeCarbon
     kWh (CPU/RAM/GPU/total) across A100 / A10 / T4 / Sapphire-Rapids
     × quantization. (PR #135, no declared license — conservative
     redistribution policy.)
   - `ejhusom/llm-inference-energy-consumption` — per-request Ollama
     timing (total / load / prompt / response_duration_ns) +
     per-request CodeCarbon kWh, laptop2 vs workstation × Alpaca vs
     CodeFeedback × gemma:7b / codellama:7b / codellama:70b.
     CC-BY-SA-4.0. (PR #138.)
4. **Which datasets can calibrate the scorer without invented
   constants?**  
   Only the same two A-class datasets above can supply *measured*
   kWh-per-request priors. Pricing remains operator-policy-only:
   even with `afhubbard/gpu-prices` overlaid, the operator's
   internal chargeback rate / fleet-actual `$/hr` are different
   numbers from the public list price.
5. **Which required coefficients remain operator-policy-only?**
   - `energy_price_per_kwh_usd` (live utility / spot feed)
   - `carbon_price_per_kg_usd` (internal carbon price)
   - `per_gpu_hour_price_usd` *fleet-actual*, not public list price
   - `internal_chargeback_rate`
   - `reserved_capacity_amortization`
   - `datacenter_pue_for_the_operator_facility`

This matches the
`docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md::§4` calibration table —
no `Level 1 operator policy` slot is closeable from public data.

## 4. Top economic candidates (ranked by ESV)

> ESV = expected scorer value = 0.45·economic_signal_quality +
> 0.20·operational_signal_quality + 0.15·joinability +
> 0.10·license_safety + 0.10·production_similarity (0-10 scale).

| Rank | Dataset | Class | ESV | EQ | OQ | License | Action |
|---:|---|---|---:|---:|---:|---|---|
| 1 | `afhubbard/gpu-prices` | B | 6.60 | 10 | 0 | cc-by-4.0 | **join_overlay_candidate** |
| 2 | `optimum-benchmark/llm-perf-leaderboard` | A | 6.50 | 6 | 8 | none-declared | already ingested |
| 3 | `labofsahil/aws-pricing-dataset` | B | 5.85 | 9 | 0 | mit | metadata_only |
| 4 | `ejhusom/llm-inference-energy-consumption` | A | 5.50 | 4 | 8 | cc-by-sa-4.0 | already ingested |
| 5 | `memoriant/dgx-spark-kv-cache-benchmark` | C | 4.40 | 2 | 6 | apache-2.0 | already ingested |
| 6 | `anonymoususermargin/ercot-rtcb-v1` | B | 3.65 | 5 | 0 | mit | metadata_only |
| 7 | `grm/caiso-curtailment` | B | 3.25 | 5 | 0 | none | metadata_only |
| 8 | `ariefansclub/han-humanoid-energy-consumption-estimator-v1` | B | 2.70 | 2 | 0 | mit | metadata_only |
| 9 | `Nayan10767/llm-inference-energy-consumption` | B | 2.40 | 2 | 0 | cc-by-sa-4.0 | metadata_only (mirror) |
| 10 | `adityaupasani/llm-inference-energy-consumption` | B | 2.40 | 2 | 0 | cc-by-sa-4.0 | metadata_only (mirror) |
| 11 | `tulipa762/electricity_load_diagrams` | B | 2.10 | 2 | 0 | unknown | metadata_only |

Full ranked list (top 20): see
`economic_signal_discovery_audit.json::top_20_economic_candidates`.

## 5. Datasets with both ops + economics in the same record

**Same-dataset paired:**

| Dataset | Operational | Economic | Strength | Source PR |
|---|---|---|---|---|
| `optimum-benchmark/llm-perf-leaderboard` | TTFT + TPOT (p50/p90/p95/p99), throughput, peak VRAM | kWh per request (CPU/RAM/GPU/total), gpu_type | measured | #135 |
| `ejhusom/llm-inference-energy-consumption` | Ollama timing (~ TTFT/TPOT/e2e) + throughput | kWh per request (CPU/GPU/total) | measured | #138 |

Both are *already* in the federated corpus. **No new dataset adds
same-record ops + economics.**

**Cross-dataset joinable** (B-class economics overlaid on existing
operational traces):

| B-side dataset | Joins to | Join keys | Note |
|---|---|---|---|
| `afhubbard/gpu-prices` | `Qinghao/AcmeTrace`, `optimum-benchmark`, `CARA`, `AgentPerfBench` | `gpu_type`, `timestamp` (`region` where available) | strongest new overlay — see §6 |
| `labofsahil/aws-pricing-dataset` | `Qinghao/AcmeTrace`, `optimum-benchmark`, `CARA` | `gpu_type` (via EC2 instance_type), `region` | static AWS list price |
| `anonymoususermargin/ercot-rtcb-v1` | `ejhusom/llm-inference-energy-consumption`, energy/price model | `timestamp`, ERCOT bidding zone | Texas grid only |
| `grm/caiso-curtailment` | energy/price model | `timestamp`, CAISO zone | curtailment, not full price — license=None |

## 6. Strongest candidate for scorer calibration — `afhubbard/gpu-prices`

| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/afhubbard/gpu-prices |
| License | CC-BY-4.0 |
| Gated | no |
| Coverage | 12+ public clouds (AWS, GCP, Azure, Lambda Labs, RunPod, Vast.ai, DataCrunch, Cudo Compute, TensorDock, Vultr, Oracle, Nebius, CloudRift) |
| Frequency | twice daily, Hive-partitioned Parquet by `dt=YYYY-MM-DD` |
| Schema | `timestamp, provider, instance_type, gpu_type, gpu_count, gpu_memory_gb, vcpus, ram_gb, region, price_per_hour, is_spot, available, availability_zone` |
| Field quality | measured (provider list prices via `gpuhunt` scraper) |
| Joinable to existing Aurelius traces | `Qinghao/AcmeTrace` (gpu_type), `optimum-benchmark` (gpu_type), `CARA` (gpu_type), `AgentPerfBench` (gpu_type) |
| Why it does NOT make Aurelius A_PLUS | No operational telemetry — pricing snapshots only. Joins to ops traces are *cross-dataset overlays*, not same-record pairs. |
| Risk to scorer headline | LOW — public list price ≠ operator fleet-actual `$/hr`. Treated as a `Level 3 prior` per §4 below. |

Recommended action (per the mission ladder): **join_overlay_candidate**
— do NOT promote to `ingest_now` until the operator confirms the
public list price is an acceptable proxy. The constraint scorer's
`OperatorPricingPolicy.gpu_hour_price_per_type` slot remains
authoritative; the public price feed at best populates the *missing*
slot with a `value_quality = "prior"` tag.

## 7. Datasets rejected and why

- **131 of 143 candidates classified F (irrelevant).** Dominant
  rejection reasons (recorded per-candidate in
  `audit.candidates[*].classification`):
  - **No compute context** — banking/payments chargeback ("billing",
    "invoice", "chargeback"), utility kWh billing, supply-chain
    routing, etc. The context-aware filter drops these.
  - **Talk only, no telemetry** — papers / instruction-tuning sets /
    chatbot conversations that mention "cost" or "GPU" without ever
    measuring either.
  - **README-only repos** — e.g. `exalsius/gpu-prices` ships nothing
    but a README; no parquet/csv data.
  - **Energy-market-only without operator-cluster join key** — many
    EIA / utility consumption datasets where neither GPU nor
    cluster context applies.
- **No D-class (synthetic economics) candidates** survived the
  rejection filter — `tarekmasryo/llm-system-ops-production-telemetry-sft-data`
  and `MCP-1st-Birthday/smoltrace-cloud-cost-tasks` were both
  rejected in the prior broadened-discovery audit (PR #134) and did
  not resurface here under the economic-term searches.

## 8. Field quality + trust tier per candidate

Per-candidate fields recorded in the JSON
(`audit.candidates[*]`):

- `field_quality` ∈ `{measured, derived, proxy, synthetic, missing}`
- `trust_tier` ∈
  - `tier_2_paired_ops_economics` (A_PLUS)
  - `tier_3_ops_plus_energy` (A)
  - `tier_4_economics_only_joinable` (B)
  - `tier_4_ops_only` (C)
  - `tier_5_metadata_or_other` (F)
  - `tier_6_synthetic` (D / synthetic)
- `license`, `gated_status`, `row_count`, `storage_size_bytes`
- `operational_signals_present`, `economic_signals_present`
- `join_keys_available`, `timestamp_available`, `region_available`,
  `gpu_type_available`, `price_or_cost_available`,
  `energy_available`, `carbon_available`
- `can_pair_with_existing_traces` — boolean per
  `{CARA, AcmeTrace, Optimum, BurstGPT, Google Cluster, SwissAI,
  AgentPerfBench}`
- `recommended_action` ∈ `{ingest_now, join_overlay_candidate,
  metadata_only, reject_synthetic, reject_no_economics,
  reject_no_ops, license_blocked, gated_blocked}`
- `scores.{economic_signal_quality, operational_signal_quality,
  joinability, license_safety, production_similarity, uniqueness,
  expected_scorer_value}` (each 0-10)

## 9. Operator-policy-only coefficients (binding)

Every $-denominated coefficient in the constraint-aware scorer that
public HF data cannot supply (matches
`docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md::§4`):

| Coefficient | Why public HF can't supply it |
|---|---|
| `energy_price_per_kwh_usd` | Live utility / spot-market price by zone; even ERCOT / CAISO datasets are 5-minute LMPs, not the operator's contracted tariff. |
| `carbon_price_per_kg_usd` | Internal carbon shadow price; not a public quantity. |
| `per_gpu_hour_price_usd` (fleet-actual) | Public list price ≠ negotiated invoice / reserved-capacity amortisation. `afhubbard/gpu-prices` is a *prior* at best. |
| `internal_chargeback_rate` | Per-org policy. |
| `reserved_capacity_amortization` | Per-contract amortisation schedule. |
| `datacenter_pue_for_the_operator_facility` | Facility-specific; public PUE benchmarks (e.g. Uptime Institute) are aggregates. |

## 10. Does this change the scorer roadmap?

**No.** The two-pass result in
`docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md::§7` stands:

- **Pass 1 (priors only):** `diagnostic_only`. No A_PLUS or A
  candidate added by this audit changes that — the two A-class
  datasets are the same ones already wired into the Optimum prior.
- **Pass 2 (operator policy supplied):**
  `shadow_ready_for_integration_review`. The audit *does* unlock
  one new option: a future operator-overlay PR could use
  `afhubbard/gpu-prices` as the *fallback* `Level 3 prior` when no
  operator policy is supplied, instead of falling all the way back
  to the single global default. That is a follow-on, not part of
  this PR.

The +93% goodput/$ headline from PR #139 is still dominated by the
per-GPU price spread (~72 pp), and that spread is still
operator-policy-only.

## 11. Tests

`tests/test_hf_economic_signal_discovery.py` enforces (39 tests, all
green):

- `economic_signal_discovery_audit.json` exists and validates against
  the documented schema (top-level keys, per-candidate keys, scores
  block, special_audit keys).
- Every candidate has `scores.economic_signal_quality` ∈ [0, 10].
- No file under `data/external/hf_discovery/` contains the
  `HF_TOKEN` literal.
- `git ls-files` lists no raw download from `afhubbard/gpu-prices`,
  `labofsahil/aws-pricing-dataset`, or any other dataset under
  `data/external/hf/`.
- No D-class (synthetic-economics) candidate is labelled
  `ingest_now`.
- Every `ingest_now` candidate records both `license` and
  `gated_status`.
- Every `join_overlay_candidate` carries a non-empty
  `join_keys_available`.
- The docs reference every operator-policy-only coefficient name
  from the JSON.

## 12. Reproducibility

```bash
HF_TOKEN=hf_... python3 scripts/discover_hf_economic_signals.py
# writes: data/external/hf_discovery/economic_signal_discovery_audit.json
pytest tests/test_hf_economic_signal_discovery.py -q
```

The script accepts `--limit-per-query`, `--max-inspect`, and
`--output` for tuning. Discovery NEVER downloads file contents — only
the per-dataset JSON card + the README (best-effort, 128 KiB cap).
