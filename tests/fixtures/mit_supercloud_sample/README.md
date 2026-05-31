# MIT Supercloud — small synthetic fixture

This is a **synthetic** fixture that mirrors the schema documented in
the MIT Supercloud README + intro notebook
(https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset). It is
**not** a copy of the real published dataset — the real ~1 TB archive
lives at https://dcc.mit.edu/data and is not committed to this repo.

The fixture exists so the ingestion + training-frontier code paths
can be exercised by unit tests with zero external network access. The
synthetic values are pre-registered to cover:

- the `tres_req` parser (CPU / mem / `gpu:tesla` / `gpu:volta`)
- the labelled-job join (every-other job in `labelled_jobids.csv`)
- queue wait / duration / GPU-seconds derivation
- mixed `COMPLETED` / `FAILED` / `CANCELLED` statuses
- the node-data + GPU-utilization layers (one per layer)

The values are NOT representative of MIT Supercloud's actual workload
distribution — for the real frontier audit, point the ingestion script
at the published archive (see `scripts/ingest_mit_supercloud.py`).
