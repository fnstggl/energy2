# Frontier Scavenger Audit V1 — non-HuggingFace public LLM-inference telemetry sweep

> **Audit-only PR. No forecaster is trained. No production code is
> modified. No savings is claimed. Figures are NOT treated as raw data.
> Simulated/benchmark traces are NOT called production telemetry.**
>
> This is a final deep internet scavenger for *downloadable* public
> datasets/artifacts that carry server-class LLM/AI **inference**
> operational telemetry — cold-start, model-load, migration, cache /
> KV-reuse, queue, autoscaling, latency, GPU utilization / power /
> thermal, and economic fields. It deliberately **excludes Hugging Face**
> (covered by `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md`,
> `docs/HF_DATASET_REGISTRY.md`) and sweeps systems-conference artifact
> pages, GitHub, cloud trace releases, and arXiv instead.
>
> **Read first:**
> - `docs/FORECAST_LEVERAGE_AUDIT.md` — which forecasts are gated on what data
> - `docs/AURELIUS_TELEMETRY_GAP_DISCOVERY.md` — HF Tier-2..6 baseline
> - `docs/HF_DATASET_REGISTRY.md` — trust hierarchy (binding)
>
> Machine-readable companion:
> `data/external/frontier_scavenger_v1/candidate_registry.json`.

## 0. TL;DR final verdict

The single **strongest** public trace that closes the cold-start /
model-load gap **already exists in the corpus**: Alibaba
`cluster-trace-v2026-GenAI` (GenTD26) → `data/external/alibaba_genai`.
It carries measured `model_load_latency`, GPU duty-cycle, GPU memory,
queue size / queue RT, QPS, pure-inference and e2e latency from a real
GenAI serving cluster.

The scavenger surfaced **3 net-new downloadable candidates not yet
ingested**, plus one partially-ingested:

| Candidate | What it adds | Closes a gap? | Verdict |
|---|---|---|---|
| **Mooncake FAST25 traces** | production KV-block-hash + arrivals (Kimi) | additive to KV/prefix-reuse (already STRONG) + arrivals; **no** latency/GPU | `exact_dataset_found` (narrow) |
| **Azure LMM 2025** (multimodal) | first images-per-request arrival dimension | additive arrival shape only | `partial_dataset_found` |
| **Vidur profiling CSVs** | measured kernel latency A100/A40/H100 | additive Tier-4 kernel-cost prior | `partial_dataset_found` |
| **AcmeTrace** (NSDI'24) | GPU power (IPMI) + util (DCGM) + thermal-adjacent | closest GPU-power source, but **training-class** | `exact_dataset_found` (partially ingested) |

The gaps that **remain unclosed by any downloadable public artifact**:

- **Cold-start / model-load LATENCY labels for serverless LLM** —
  measured by ServerlessLLM (OSDI'24) and HydraServe (NSDI'26) but
  released only as figures → `measured_unreleased` / `gated`.
- **GPU power / thermal for INFERENCE serving** — measured by POLCA
  (arXiv'23 + ASPLOS'24), DynamoLLM (HPCA'25), TAPAS (ASPLOS'25) but
  released only as figures / internal Azure traces → `measured_unreleased`.
  (Acme/MIT-SuperCloud/Philly give *training-class* power/util only.)
- **Autoscaler / replica-action labels for LLM serving** → `not_found`
  in any public downloadable trace.
- **Migration / rerouting event logs** — Llumnix (OSDI'24) describes
  them but ships code/figures only → `code_or_figures_only`; Mooncake
  `hash_ids` remain the only downloadable cache-loss / migration *proxy*.

**Bottom line:** the corpus is already at the public frontier for
cold-start (Alibaba GenAI) and KV-reuse. The remaining Aurelius targets
(serverless cold-start latency, inference GPU power/thermal, autoscaler
actions) are **measured-but-unreleased or not-found** — they cannot be
closed by ingestion and must stay on pilot-telemetry (Tier 1).

## 1. Method & scope

- **Surfaces swept:** USENIX OSDI/NSDI/ATC/FAST artifact + presentation
  pages; ACM SOSP/EuroSys/SoCC/ASPLOS/HPCA/ISCA/MLSys papers; GitHub
  repos/releases/data folders (Azure, Alibaba, Microsoft, kvcache-ai,
  InternLM, ServerlessLLM, AlibabaPAI); Microsoft Azure Public Dataset;
  arXiv PDFs/HTML; author/project pages.
- **Verification per candidate (task contract):** (1) is there real
  downloadable data, not just figures? (2) which files? (3) what
  schema/columns? (4) raw / aggregated / simulated / benchmark /
  production-derived? (5) measured labels vs formulas/plots? (6)
  bounded-ingestible safely? (7) license/access? (8) exact URL? (9)
  closes which Aurelius target?
- **Trust hierarchy (binding, from `HF_DATASET_REGISTRY.md`):** Tier 1
  pilot telemetry only is production-equivalent; everything below is
  directional. Benchmark/simulator output is never production telemetry.

## 2. Candidate ledger

### 2.1 Net-new downloadable (NOT yet ingested)

#### Mooncake FAST25 traces — `exact_dataset_found` (narrow)
- **URL:** https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces (+ root `mooncake_trace.jsonl`, released 2024-07-09)
- **Files:** `conversation_trace.jsonl`, `synthetic_trace.jsonl`, `toolagent_trace.jsonl`, original `mooncake_trace.jsonl`.
- **Schema (verified by fetching record 0-1):**
  `{"timestamp": int, "input_length": int, "output_length": int, "hash_ids": [int,...]}`
  e.g. `{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0,1,2,...,13]}`.
- **Class:** production-derived (real Kimi/Moonshot arrivals; prompt text
  removed; KV-block hashes remapped — no user content).
- **Measured labels:** arrivals, input/output token counts, **KV-block
  hash list** → prefix-reuse & cache-loss proxy.
- **Absent:** TTFT/TPOT/e2e, queue, GPU util/power/mem, cold-start,
  model-load, autoscaling, SLO, cost.
- **Bounded-ingestible:** yes — small JSONL; commit only schema_profile +
  normalized sample (timestamps, token counts, hash-list length / reuse
  stats), no raw content.
- **License:** Apache-2.0 (repo LICENSE) → redistribution-permissive.
- **Closes:** *additive* to `kv_prefix_reuse_forecast` (already STRONG)
  and `arrival_forecast`. An independent production KV-hash trace beyond
  CC-traces / LMCache. Does **not** close cold-start / GPU / power /
  autoscaling.

#### Azure LMM Inference 2025 (multimodal, SoCC'25) — `partial_dataset_found`
- **URL:** https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md → `data/AzureLMMInferenceTrace_multimodal.csv.gz`
- **Schema:** `TIMESTAMP, NumImages, ContextTokens, GeneratedTokens` (Oct 15-22 2024).
- **Class:** production-derived; token/image counts only.
- **Closes:** *additive* arrival/workload-shape priors; first
  **multimodal** (images-per-request) dimension. No latency/GPU/economic.
- **License:** CC-BY. **Bounded-ingestible:** yes (gz CSV).

#### Vidur profiling CSVs (MLSys'24) — `partial_dataset_found`
- **URL:** https://github.com/microsoft/vidur/tree/main/data/profiling (subfolders `compute/`, `network/`)
- **Files:** `compute/<gpu>/<model>/mlp.csv`, `.../attention.csv`, network collective profiles.
- **Schema:** operator/kernel config (batch, seq, TP) → measured kernel
  latency (ms), per GPU type (A100/A40/H100).
- **Class:** **benchmark / micro-profiling** — measured on real GPUs,
  then used to drive a *simulator*. Pre-collected CSVs shipped for
  profiled combos; new models need GPU profiling. **NOT production
  telemetry** — Tier-4 at best.
- **Closes:** additive kernel-cost / compute-time priors. **License:** MIT.

### 2.2 Already in corpus (confirmed at frontier — no re-ingest)

| Candidate | Local path | Note |
|---|---|---|
| Alibaba `cluster-trace-v2026-GenAI` (GenTD26) | `data/external/alibaba_genai` | **Strongest** public cold-start/model-load + GPU-util + queue + inference-latency trace. Production raw sampling. |
| Alibaba `cluster-trace-gpu-v2023` | `data/external/alibaba_gpu` | Scheduling lifecycle + GPU inventory; **no util/power series** in v2023. |
| Azure LLM Inference 2023 (Splitwise) / 2024 (DynamoLLM) | `data/external/azure_llm`, `azure_llm_2024` | Token+timestamp only; energy/latency content of those papers is *unreleased* (see §2.4). |
| AcmeTrace (NSDI'24) | `data/external/economic_overlay/...acmetrace...` | GPU power (IPMI) + util (DCGM) + thermal-adjacent; **training-class**, partially ingested for economic overlay. |

> **AcmeTrace** (https://github.com/InternLM/AcmeTrace) is the closest
> public **GPU power/thermal** source (real DCGM+IPMI+Prometheus), but it
> is a **training** cluster, not inference serving — usable only as a
> directional GPU power/util prior, never inference truth.

### 2.3 Code / figures only — `code_or_figures_only`

- **Llumnix** (OSDI'24, `alibaba/llm-scheduling-artifact`) — live request
  migration. Artifact = code to reproduce figures on synthetic / Azure
  arrivals; migration outcomes are **plots, not an event-log dataset**.
- **AlpaServe / DistServe / Sarathi-Serve** (OSDI'23/'24) — serving
  systems + (Sarathi) Vidur-based simulation. Latency/throughput are
  *reproduce-on-run*, not shipped as measured traces. (No run performed —
  out of scope.)

### 2.4 Measured-but-unreleased — `measured_unreleased`

- **ServerlessLLM** (OSDI'24, `ServerlessLLM/ServerlessLLM`) — multi-tier
  checkpoint loading + live migration; **cold-start / model-load latency
  is the exact label Aurelius lacks**, but it lives only in paper figures
  and a hardcoded README table. Repo = code only (verified: no `data/`,
  no benchmark CSVs).
- **DynamoLLM** (HPCA'25) — GPU-frequency × parallelism × request-length
  **energy** profiling (TP2/4/8, 800-1980 MHz) is in the paper; the
  *released* artifact is only the Azure 2024 token trace.
- **POLCA** (arXiv 2308.12908) + **"Characterizing Power Management
  Opportunities for LLMs in the Cloud"** (ASPLOS'24) — production Azure
  GPU/server power traces analyzed in-paper; **not released** as a
  dataset.
- **TAPAS** (ASPLOS'25) — thermal/power-aware scheduling on Azure; no
  released thermal/power artifact located.

### 2.5 Gated — `gated_contact_authors_needed`

- **HydraServe** (NSDI'26, arXiv 2502.15524) — minimizes serverless LLM
  cold-start; evaluated on "real-world datasets"; no public artifact
  dataset located → would require contacting authors / the NSDI'26
  artifact (not yet posted at audit date).

### 2.6 Not found — `not_found`

- **Autoscaler / replica-action labels for LLM serving** — no public
  downloadable trace carries a replica-count time-series with explicit
  autoscaler-action labels. Closest proxies: arrivals (BurstGPT / Azure /
  Mooncake), Alibaba GenAI queue/QPS, and Google/Alibaba cluster
  SCHEDULE/EVICT events (cluster-scheduler proxy, already ingested).
- **Zenodo / figshare / Dataverse sweep** — LLM-serving artifacts there
  are dominated by code / reproduction bundles / simulator configs; no
  net-new measured *inference*-serving telemetry beyond the GitHub-hosted
  candidates above.

## 3. Aurelius target × frontier-availability matrix

| Aurelius ML target | Best public downloadable source | Status |
|---|---|---|
| TTFT / TPOT / E2E latency | CARA (Tier-2, ingested) + Alibaba GenAI inference latency | **covered** |
| Queue depth / wait | CARA queue_details + Alibaba GenAI queue_size/rt | **covered** |
| KV / prefix reuse | SwissAI/CC-traces/LMCache (ingested) + **Mooncake hash_ids** (new) | **covered, additive** |
| Arrivals / workload shape | BurstGPT/Azure 2023-24 (ingested) + **Azure LMM 2025**, **Mooncake** (new) | **covered, additive** |
| **Cold-start / model-load latency** | **Alibaba GenAI `model_load_latency` (ingested)** | **covered (only public source)** |
| Cold-start *serverless* (load + GPU activation) | ServerlessLLM / HydraServe | **measured-unreleased / gated** |
| Migration / rerouting outcomes | — (Mooncake hash_ids = cache-loss proxy only) | **code/figures only** |
| **GPU power / thermal (inference)** | — (POLCA/DynamoLLM/TAPAS unreleased; Acme = training) | **measured-unreleased** |
| GPU power / thermal (training proxy) | AcmeTrace / MIT SuperCloud / Philly (ingested) | **covered (directional, training-class)** |
| Kernel / compute-cost priors | **Vidur profiling CSVs** (new, benchmark) | **covered, additive (Tier-4)** |
| **Autoscaler / replica actions** | — | **not found** |
| Energy price / carbon | CAISO/PJM/ERCOT/ElectricityMaps (ingested) | **covered** |

## 4. Recommended bounded actions (audit-only; no obligation)

1. **Mooncake FAST25 traces** — worth a bounded ingest as an independent
   production KV-hash + arrivals trace (Apache-2.0, tiny JSONL). Commit
   only schema_profile + normalized sample (timestamps, token counts,
   hash-list reuse stats); no raw content. Strength: additive to an
   already-STRONG signal — *nice-to-have, not gap-closing*.
2. **Azure LMM 2025** — optional; only meaningful if multimodal arrival
   shape becomes a modelled dimension. CC-BY, gz CSV.
3. **Vidur profiling CSVs** — optional Tier-4 kernel-cost prior; must be
   labelled benchmark/simulator, never production.
4. **AcmeTrace** — already partially ingested; if a dedicated GPU-power
   prior is wanted, expand the existing economic-overlay sample — keep
   the training-class caveat binding.
5. **Do NOT** attempt to manufacture cold-start-serverless / inference
   power / autoscaler datasets from the unreleased papers — those targets
   stay on pilot telemetry (Tier 1).

## 5. Safety properties of this audit

- No dataset was downloaded or ingested by this PR; deliverables are a
  registry + this doc.
- No forecaster trained, no scheduler / production code modified, no
  savings claimed.
- Every "measured" claim is qualified by whether the data is
  downloadable; figures and README tables are explicitly flagged as
  **not raw data**.
- Simulator/benchmark sources (Vidur, AlpaServe/DistServe/Sarathi) are
  labelled as such and never called production telemetry.
- Production-derived arrival traces (Azure, Mooncake) are not conflated
  with measured latency/GPU telemetry.

---
*Companion machine-readable registry:
`data/external/frontier_scavenger_v1/candidate_registry.json` (16
candidates; verdicts: 5 exact, 2 partial, 2 code/figures-only, 4
measured-unreleased, 1 gated, 2 not-found).*
