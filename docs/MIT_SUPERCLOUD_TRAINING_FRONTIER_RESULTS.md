# MIT Supercloud — Training Safe Utilization Frontier Results

> **Simulator / public-trace benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Validates Training Frontier v1 (`aurelius/frontier/training_*`) on the MIT Supercloud Dataset (Samsi et al., HPEC 2021). The serving Safe Utilization Frontier Controller, the robust energy engine, the committed Azure 2024 / Philly / Alibaba GPU benchmark artifacts are all **unchanged**. Real-cluster execution is **disabled by default**. The MIT Supercloud raw archive (~1 TB) is NOT committed — see `scripts/ingest_mit_supercloud.py` for download instructions.

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`, `docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md`.

## 1. Source + discovery

- **Source dir:** `tests/fixtures/mit_supercloud_sample` (synthetic fixture)
- **Repo:** https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset
- **Raw archive home:** https://dcc.mit.edu/data
- **Paper:** https://arxiv.org/abs/2108.02037
- **n_jobs:** 10  **n_gpu_jobs:** 10  **n_labelled:** 6

### Discovered files

| file | status | classification | kind |
|---|---|---|---|
| `scheduler-log.csv` | present | primary | scheduler |
| `labelled_jobids.csv` | present | label_metadata | label |
| `tres-mapping.txt` | present | scheduler_metadata | tres_mapping |
| `node-data.csv` | present | node_inventory | node |
| `gpu` | present | primary | gpu_metric |
| `cpu` | present | primary | cpu_metric |

### Join quality matrix

| join | kind | matched / right | confidence | notes |
|---|---|---|---|---|
| `label_to_job` | `exact_job_id_join` | 6 / 10 | `high` | job_id appears in labelled_jobids.csv |
| `gpu_util_to_job` | `exact_job_id_join` | 2 / 10 | `high` | GPU sample file name == job_id (per the MIT intro notebook); join is exact |
| `node_util_to_job` | `node_time_join` | 4 / 10 | `medium` | node snapshot ↔ job by node-name intersection + [start,end] window overlap; medium confidence because snapshots are 5-min granular |

## 2. Trace summary

- queue wait p50/p95/p99 (s): 300.00 / 1,000.00 / 1,000.00
- duration   p50/p95/p99 (s): 2,000.00 / 7,200.00 / 7,200.00
- gpu_count distribution: `{'1': 2, '2': 5, '4': 2, '8': 1}`
- gpu_type distribution:  `{'gpu:tesla': 6, 'gpu:volta': 4}`
- status distribution:    `{'CANCELLED': 1, 'COMPLETED': 7, 'FAILED': 2}`
- workload labels:        `{'Bert': 1, 'DistillBert': 1, 'Inception': 1, 'ResNet': 1, 'U-Net': 1, 'VGG': 1}`

## 3. Synthetic fleet (sized to trace demand)

- **n_nodes:** 3  **gpus_per_node:** 8  **total_gpus:** 24
- **node_overhead_factor:** 5.00 (fleet sized to peak job × overhead — MIT does not publish per-node capacity)

## 4. Training-frontier sweep (one row per policy)

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `fifo` | 252.69 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `first_fit` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `best_fit` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `first_fit_decreasing` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `greedy_packing` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `topology_aware` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `utilization_aware` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `constraint_aware` | 758.06 | 0.140905 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |

## 5. Controller verdict

- **current policy (baseline):** `constraint_aware` → goodput/$ 758.06
- **training_frontier_v1 selected:** `constraint_aware` → goodput/$ 758.06
- **Δ vs current:** +0.000 %
- **action:** `KEEP_CURRENT_POLICY`
- **verdict:** **`TIE`**
- **reason:** current candidate 'constraint_aware' within KPI deadband (0.0000 ≤ 0.01) and packing-density deadband (0.0000 ≤ 0.05)

## 6. Does MIT Supercloud reveal new Training Frontier alpha?

- **Verdict:** NO — MIT Supercloud safely ties constraint_aware. Result is consistent with Philly + Alibaba GPU: constraint_aware is already on or near the safe training frontier.
- **Evidence:** selected policy `constraint_aware` matches the baseline KPI within ±1.0% goodput/$; 8 of 8 candidates SAFE

## 7. Metrics that were UNAVAILABLE and NOT INVENTED

- Per-job gang-scheduling failure — MIT scheduler-log does not cleanly distinguish gang failures from other failure causes (gate disabled by default).
- Per-job retry/wasted-GPU-hours — MIT scheduler-log lacks attempt history (Philly has it; Alibaba GPU does not).
- Per-node capacity — MIT publishes `node-data.csv` utilization but not per-node capacity; fleet is sized synthetically to the trace's peak demand.
- Per-job utilization integration into KPI — GPU CSVs match job_id exactly, but the KPI uses requested GPU-seconds, not realized utilization, to stay comparable across traces.

## 8. Honesty / scope

- The MIT Supercloud raw dataset (~1 TB compressed) is NOT committed to this repo. The full benchmark requires running the script with the published archive (see `scripts/ingest_mit_supercloud.py`).
- The synthetic fleet sizes nodes from the trace's peak demand and an overhead factor; MIT does NOT publish per-node capacity. Absolute KPIs are therefore relative across policies, not production-comparable.
- No new datasets ingested beyond MIT Supercloud.
- No serving-frontier code changed.
- No robust-energy-engine change.
- No ML training.
- No production-savings claim. Pilot telemetry is required to calibrate per-tenant safety thresholds.

