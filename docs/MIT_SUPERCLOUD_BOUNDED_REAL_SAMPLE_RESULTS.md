# MIT Supercloud — Bounded Real-Sample Training Frontier Results

> **Simulator / public-trace benchmark. Directional only — NOT production savings** (`docs/RESULTS.md` §8). Re-runs Training Safe Utilization Frontier v1 on the **bounded real MIT Supercloud Slurm sample** (downloaded from the public S3 bucket — see §1), replacing the tiny 10-GPU-job synthetic fixture used in the v1 PR. The serving Safe Utilization Frontier Controller, the robust energy engine, the committed Azure 2024 / Philly / Alibaba GPU benchmark artifacts, and the v1 fixture-mode MIT summary are all **unchanged**. Real-cluster execution is **disabled by default**. The MIT Supercloud raw archive is bounded-downloaded; the full ~1–2 TB dataset is **NOT** committed.

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`, `docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md`, `docs/TRAINING_SAFE_UTILIZATION_FRONTIER_RESULTS.md`, `docs/MIT_SUPERCLOUD_TRAINING_FRONTIER_RESULTS.md` (v1 fixture-mode result).

## 1. S3 paths used + bounded download

- **Bucket:** https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201/
- **Local raw dir:** `/home/user/energy2/data/external/mit_supercloud/raw`
- **Total downloaded:** 2.96 MB (full dataset is ~1–2 TB; NOT downloaded)

| file | s3 path | size (B) | downloaded | sample policy |
|---|---|---|---|---|
| `LICENSE` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/LICENSE` | 285 | ✅ | `full_file` |
| `README.md` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/README.md` | 7,616 | ✅ | `full_file` |
| `32585007376605-r8939293-n208530.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/32585007376605-r8939293-n208530.csv` | 157,396 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `63281038950145-r2652301-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/63281038950145-r2652301-n43543.csv` | 167,170 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `32482686106814-r8937440-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/32482686106814-r8937440-n43543.csv` | 166,659 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `75173921402868-r1682297-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/75173921402868-r1682297-n43543.csv` | 166,274 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `63435844151296-r8642123-n911952.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/63435844151296-r8642123-n911952.csv` | 131,547 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `6575560256580-r1485405-n685852.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/6575560256580-r1485405-n685852.csv` | 166,985 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `90841990794011-r7217787-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/90841990794011-r7217787-n43543.csv` | 158,652 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `19439987563587-r629115-n830961.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/19439987563587-r629115-n830961.csv` | 140,316 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `79192399314227-r9192091-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/79192399314227-r9192091-n43543.csv` | 141,982 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `5399923315118-r2998125-n208530.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/5399923315118-r2998125-n208530.csv` | 156,553 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `54381036425027-r9102715-n830961.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/54381036425027-r9102715-n830961.csv` | 148,474 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `43238601009423-r4822976-n208530.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/43238601009423-r4822976-n208530.csv` | 151,560 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `31001519108118-r4229531-n386398.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/31001519108118-r4229531-n386398.csv` | 143,129 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `78299339974881-r6272977-n851693.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/78299339974881-r6272977-n851693.csv` | 167,406 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `46449034615364-r3741709-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/46449034615364-r3741709-n43543.csv` | 151,103 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `84849662026062-r1682297-n43543.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/84849662026062-r1682297-n43543.csv` | 166,603 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `8622029461267-r1485405-n685852.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/8622029461267-r1485405-n685852.csv` | 151,853 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `70906003034474-r4179716-n851693.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/70906003034474-r4179716-n851693.csv` | 133,826 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `18895765180183-r3226521-n685852.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/18895765180183-r3226521-n685852.csv` | 165,110 | ✅ | `gpu_sample_n=20_max_mb=30` |
| `68581317360449-r2582019-n976057.csv` | `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/0000/68581317360449-r2582019-n976057.csv` | 163,679 | ✅ | `gpu_sample_n=20_max_mb=30` |

## 2. Real Slurm sample summary

- **n_jobs:** 10,000  **n_gpu_jobs:** 10,000  **n_labelled:** 0
- time span: 55.9 d
- queue wait p50/p95/p99 (s): 167.00 / 234,074.00 / 412,686.00
- duration   p50/p95/p99 (s): 9,013.00 / 182,921.00 / 614,130.00
- gpu_count distribution: `{'1': 8965, '10': 1, '16': 53, '2': 796, '3': 12, '4': 132, '6': 7, '8': 34}`
- gpu_type distribution:  `{'gpu:volta': 10000}`
- status distribution:    `{'CANCELLED': 1535, 'COMPLETED': 6669, 'FAILED': 1144, 'NODE_FAIL': 15, 'OUT_OF_MEMORY': 298, 'TIMEOUT': 339}`
- top-10 workload labels: `{}`

## 3. Join quality matrix

| join | kind | matched / right | confidence | notes |
|---|---|---|---|---|
| `label_to_job` | `exact_job_id_join` | 0 / 10000 | `none` | job_id appears in labelled_jobids.csv |
| `gpu_util_to_job` | `exact_job_id_join` | 7 / 10000 | `high` | GPU sample file name == job_id (per the MIT intro notebook); join is exact |
| `node_util_to_job` | `node_time_join` | 0 / 10000 | `none` | node snapshot ↔ job by node-name intersection + [start,end] window overlap; medium confidence because snapshots are 5-min granular |

## 4. Bounded GPU utilization sample coverage

- sampled files: 20  (65,966 util rows)
- matched job_ids (any): 7  of 10,000  → 0.0700 %
- matched GPU job_ids: 7  of 10,000  → 0.0700 %
- realized GPU utilization p50/p95/p99: 0.0000 / 94.00 / 95.00

## 5. Training-frontier capacity sensitivity sweep

- MIT does NOT publish per-node capacity. The fleet is synthesized at three pre-registered sizing points (small / medium / large) so the verdict is reported against capacity, not against a single tuned fleet.

| fleet | n_nodes × gpus/node | total_gpus | controller verdict | selected policy | Δ vs current | action |
|---|---|---|---|---|---|---|
| `small` | 20 × 16 | 320 | **TIE** | `constraint_aware` | +0.000% | `KEEP_CURRENT_POLICY` |
| `medium` | 39 × 16 | 624 | **TIE** | `constraint_aware` | +0.000% | `KEEP_CURRENT_POLICY` |
| `large` | 78 × 16 | 1,248 | **TIE** | `constraint_aware` | +0.000% | `KEEP_CURRENT_POLICY` |

### small fleet — full per-policy table

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `fifo` | 270.58 | 0.314025 | 203,793.00 | 47.06 | 32.35 | 0.0000 | **UNSAFE** |
| `first_fit` | 314.82 | 0.322912 | 26,291.00 | 4.11 | 10.45 | 38.16 | **SAFE** |
| `best_fit` | 314.83 | 0.322912 | 26,110.00 | 3.86 | 10.79 | 42.51 | **SAFE** |
| `first_fit_decreasing` | 314.82 | 0.322912 | 26,291.00 | 4.11 | 10.45 | 38.16 | **SAFE** |
| `greedy_packing` | 314.83 | 0.322912 | 26,110.00 | 3.86 | 10.79 | 42.51 | **SAFE** |
| `topology_aware` | 314.83 | 0.322912 | 26,110.00 | 3.86 | 10.79 | 42.51 | **SAFE** |
| `utilization_aware` | 314.83 | 0.322912 | 26,110.00 | 3.86 | 10.79 | 42.51 | **SAFE** |
| `constraint_aware` | 314.83 | 0.322912 | 26,110.00 | 3.86 | 10.79 | 42.51 | **SAFE** |

### medium fleet — full per-policy table

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `fifo` | 160.56 | 0.164527 | 44,635.00 | 7.35 | 17.22 | 0.0000 | **UNSAFE** |
| `first_fit` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.10 | 7.12 | **SAFE** |
| `best_fit` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.16 | 7.12 | **SAFE** |
| `first_fit_decreasing` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.10 | 7.12 | **SAFE** |
| `greedy_packing` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.16 | 7.12 | **SAFE** |
| `topology_aware` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.16 | 7.12 | **SAFE** |
| `utilization_aware` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.16 | 7.12 | **SAFE** |
| `constraint_aware` | 162.68 | 0.165596 | 7,434.00 | 0.0600 | 4.16 | 7.12 | **SAFE** |

### large fleet — full per-policy table

| policy | goodput/$ | occupancy | queue p99 (s) | starv % | frag block % | backfill % | safety |
|---|---|---|---|---|---|---|---|
| `fifo` | 81.33 | 0.082798 | 7,237.00 | 0.0000 | 8.56 | 0.0000 | **SAFE** |
| `first_fit` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `best_fit` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `first_fit_decreasing` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `greedy_packing` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `topology_aware` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `utilization_aware` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |
| `constraint_aware` | 96.13 | 0.082798 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **SAFE** |

## 6. Headline alpha-finding

- **Verdict:** NO — even on the real bounded MIT sample, Training Frontier safely TIES `constraint_aware` at every capacity point. Consistent with the Philly + Alibaba GPU + v1 fixture result: `constraint_aware` is already on or near the safe training frontier on this trace family.
- **Evidence:** wins=0 ties=3 losses=0 across the small / medium / large capacity sweep

## 7. Metrics that remain UNAVAILABLE and NOT INVENTED

- Per-job gang-scheduling failure — MIT scheduler-log does not cleanly label gang failures (gate disabled by default).
- Per-job retry / wasted-GPU-hours — MIT scheduler-log lacks attempt history (Philly has it; MIT does not).
- Per-node capacity — MIT publishes node utilization (`node-data.csv`) but not per-node capacity; fleet is synthesized over three sizing points (small / medium / large).
- Realized utilization in KPI — GPU CSVs match job_id exactly, but the KPI uses requested GPU-seconds to stay comparable across traces. Realized utilization is reported separately as `gpu_sample_coverage` and NOT folded into goodput/$.
- Full ~1–2 TB dataset — bounded download only; full archive lives at https://dcc.mit.edu/data and s3://mit-supercloud-dataset/datacenter-challenge/202201/, NOT committed.

## 8. Honesty / scope

- The MIT Supercloud raw archive is bounded-downloaded: slurm-log + labels + tres-mapping + LICENSE + README (~98 MB) plus an HTTP-Range-GET head sample of `node-data.csv` (default ~50 MB of the ~2.1 GB full file). The full ~1–2 TB dataset is **NOT** committed and **NOT** downloaded.
- Per-node capacity is NOT published by MIT. The fleet is synthesized at three pre-registered sizing points and the verdict is reported per-point — never on a single tuned fleet.
- The serving frontier code is **unchanged**.
- The robust energy engine is **unchanged**.
- The committed v1 fixture-mode MIT summary (`mit_supercloud_training_frontier_summary.json`) is **unchanged**; this PR writes new sibling JSON.
- No new datasets beyond MIT Supercloud.
- No ML training.
- No production-savings claim. Pilot telemetry is required to calibrate per-tenant safety thresholds.

