# Aurelius — Constraint-Aware Orchestration Progress Tracker

## Overview

This is the canonical progress tracker for the constraint-aware GPU orchestration
initiative. It is separate from `docs/AURELIUS_PROGRESS.md`, which tracks the
legacy energy-arbitrage system (Phases 1-5 of that initiative).

---

## Current System Status

| Phase | Name | Status | Evidence |
|-------|------|--------|----------|
| 1 | Normalized State Model | WIRED_BUT_UNTESTED → COMPLETE (pending test run) | aurelius/state/ |
| 2 | Prometheus-native Telemetry Ingestion | IMPLEMENTED | aurelius/connectors/prometheus/ |
| 3 | Production Metric Adapters | IMPLEMENTED | aurelius/connectors/adapters/ |
| 4 | Kubernetes Placement Connector | IMPLEMENTED | aurelius/connectors/kubernetes/ |
| 5 | Topology Collection and Scoring | NOT_STARTED | — |
| 6 | Synthetic Cluster Simulator | NOT_STARTED | — |
| 7 | Binding Constraint Classifier | NOT_STARTED | — |
| 8 | Cost/Risk Model | NOT_STARTED | — |
| 9 | Constraint-Aware Optimizer | NOT_STARTED | — |
| 10 | Reporting and Validation CLI | NOT_STARTED | — |
| 11 | Continuous Validation / Benchmarking | NOT_STARTED | — |
| 12 | Production Hardening | NOT_STARTED | — |

---

## Repo-Reality Audit Findings

### What existed before this run
- `aurelius/` package: energy arbitrage optimizer (Phases 1-5 in AURELIUS_PROGRESS.md)
- `aurelius/models.py`: Job, EnergyPrice, CarbonIntensity, ScheduleDecision, etc.
- `aurelius/optimization/`: scheduler, constraints, objective — energy optimizer
- No constraint-aware state layer existed
- No `CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md` existed
- No `COMPUTE_OPTIMIZATION_PROGRESS.md` existed

### What was created in this run
- `docs/CONSTRAINT_AWARE_ORCHESTRATION_PLAN.md` — canonical architecture plan
- `aurelius/state/__init__.py` — state package
- `aurelius/state/models.py` — all Pydantic v2 state models
- `aurelius/state/store.py` — leakage-safe ClusterStateStore
- `aurelius/state/normalize.py` — normalization utilities
- `aurelius/connectors/__init__.py`
- `aurelius/connectors/prometheus/__init__.py`
- `aurelius/connectors/prometheus/client.py` — generic Prometheus HTTP client
- `aurelius/connectors/prometheus/mappings.py` — DCGM/vLLM/Triton/Ray mappings
- `aurelius/connectors/prometheus/connector.py` — high-level connector
- `aurelius/connectors/adapters/__init__.py`
- `aurelius/connectors/adapters/dcgm.py` — DCGM text format parser
- `aurelius/connectors/adapters/vllm.py` — vLLM metrics parser
- `aurelius/connectors/adapters/triton.py` — Triton metrics parser
- `aurelius/connectors/adapters/ray_serve.py` — Ray Serve metrics parser
- `aurelius/connectors/adapters/otel.py` — OTel OTLP JSON parser
- `aurelius/connectors/kubernetes/__init__.py`
- `aurelius/connectors/kubernetes/client.py` — read-only K8s client
- `aurelius/connectors/kubernetes/parser.py` — node/pod/workload parser
- `tests/test_state_models.py`
- `tests/test_state_store.py`
- `tests/test_state_normalize.py`
- `tests/test_prometheus_connector.py`
- `tests/test_metric_adapters.py`
- `tests/test_kubernetes_connector.py`
- `tests/fixtures/cluster_state/minimal_cluster.json`
- `tests/fixtures/prometheus/dcgm_metrics.txt`
- `tests/fixtures/prometheus/vllm_metrics.txt`
- `tests/fixtures/prometheus/triton_metrics.txt`
- `tests/fixtures/prometheus/prometheus_query_response.json`
- `tests/fixtures/kubernetes/nodes_list.json`
- `tests/fixtures/kubernetes/pods_list.json`

### Plan vs Repo mismatches
- Plan referenced `QueueState` and `GPUMetrics` as existing models to adapt.
  Reality: existing `aurelius/models.py` has no `QueueState` or `GPUMetrics`.
  Resolution: new models created from scratch, compatible with existing `Job`/`EnergyPrice` types.
- Plan did not specify whether to use dataclasses or Pydantic.
  Reality: Pydantic v2 was already a dependency and provides superior validation.
  Resolution: used Pydantic v2 for all new state models.

---

## Validation Status

| Component | Unit Tests | Integration | Simulator | Production-Ready |
|-----------|-----------|-------------|-----------|-----------------|
| State models (Phase 1) | YES | N/A | N/A | Needs Phase 6 sim |
| State store | YES | N/A | N/A | Heuristic |
| Normalize utilities | YES | N/A | N/A | Heuristic |
| Prometheus client | YES (sandbox) | ENV_GAP (no live Prometheus) | N/A | Heuristic |
| DCGM adapter | YES (fixture) | ENV_GAP | N/A | Heuristic |
| vLLM adapter | YES (fixture) | ENV_GAP | N/A | Heuristic |
| Triton adapter | YES (fixture) | ENV_GAP | N/A | Heuristic |
| Ray Serve adapter | YES (fixture) | ENV_GAP | N/A | Heuristic |
| OTel adapter | YES (fixture) | ENV_GAP | N/A | Heuristic |
| Kubernetes connector | YES (fixture) | ENV_GAP (no live cluster) | N/A | Heuristic |

### Environment Gaps (acceptable)
- `PROMETHEUS_BEARER_TOKEN` not present — live Prometheus tests use sandbox mode
- No live Kubernetes cluster — K8s tests use fixture JSON responses
- No DCGM/vLLM/Triton/Ray instances — adapter tests use fixture text files
- These gaps are documented and acceptable for Phase 1-4

---

## Benchmark / Regression History

No benchmarks yet. Requires Phase 6 (Simulator) to benchmark optimizer comparisons.

---

## Exact Next Recommended Milestone

**Phase 5: Topology Collection and Scoring**

Priority: Build topology parser for `nvidia-smi topo -m` output and scoring engine.

Dependencies met:
- Phase 1 state models include `TopologyState`, `TopologyLink`, `TopologyNode` ✓
- Phase 4 Kubernetes connector provides node/zone/rack labels ✓

What needs to be built:
1. `aurelius/topology/parser.py` — nvidia-smi topo -m parser
2. `aurelius/topology/scoring.py` — placement scoring function
3. `aurelius/topology/collector.py` — TopologyProvider interface + mock
4. Topology-aware placement recommendation
5. Fixtures: dgx_h100_nvswitch, pcie_8gpu, fragmented cluster
6. Tests proving NVLink > PCIe placement scores

Blockers: None — foundations are ready.

---

## Open Technical Debt

| Item | Severity | Notes |
|------|----------|-------|
| No live Prometheus integration test | LOW | ENV_GAP, acceptable for Phase 1-4 |
| No live Kubernetes integration test | LOW | ENV_GAP, acceptable for Phase 1-4 |
| Triton histogram latencies are mean approximations (not true p95/p99) | MEDIUM | Real p95 requires histogram quantile from Prometheus; adapter notes this |
| Ray Serve error rate derived from counter totals (not rate) | MEDIUM | In prod, use rate() in Prometheus query instead of raw total |
| DCGM NVLink bytes are raw counters not rates in fixture | LOW | In prod, Prometheus scrape interval provides rates |
| Kubernetes `_parse_gpu_count` counts both requests AND limits, may double-count | LOW | Logic is conservative; GPU count = max(requests, limits) would be better |
| No topology-aware placement yet | N/A | Phase 5 |
| No constraint classifier yet | N/A | Phase 7 |
| No migration penalty model yet | N/A | Phase 8 |
| No optimizer changes yet | N/A | Phase 9 |

---

## Optimizer Behavior Preservation

The existing energy arbitrage optimizer (`aurelius/optimization/`) was not
modified in Phases 1-4. All new code is additive:

- `aurelius/state/` — new package, no imports from optimization/
- `aurelius/connectors/` — new package, no imports from optimization/
- No changes to `aurelius/models.py`
- No changes to `aurelius/optimization/scheduler.py`
- No changes to `aurelius/optimization/constraints.py`
- No changes to `aurelius/optimization/objective.py`

Existing test suite (483 tests in AURELIUS_PROGRESS.md) should remain unaffected.

---

## Test Commands

```bash
# Phase 1 state models
pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py -q

# Phase 2 Prometheus connector
pytest tests/test_prometheus_connector.py -q

# Phase 3 adapters
pytest tests/test_metric_adapters.py -q

# Phase 4 Kubernetes connector
pytest tests/test_kubernetes_connector.py -q

# All new tests
pytest tests/test_state_models.py tests/test_state_store.py tests/test_state_normalize.py \
       tests/test_prometheus_connector.py tests/test_metric_adapters.py \
       tests/test_kubernetes_connector.py -q

# Compile check
python -m compileall aurelius/state aurelius/connectors

# Ruff lint
ruff check aurelius/state aurelius/connectors
```

---

*Last updated: Phase 1-4 implementation run*
