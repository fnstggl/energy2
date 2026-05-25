# Aurelius — Constraint-Aware GPU Orchestration Plan

## Product Goal

Aurelius is evolving from an energy-aware batch scheduler into a trusted
**constraint-aware orchestration and control-plane intelligence layer** for:

- AI inference providers
- neoclouds
- GPU-heavy data centers
- infrastructure and platform teams running GPU clusters

Aurelius must help operators measurably improve:

| Metric | Target |
|--------|--------|
| cost/token | reduce |
| tokens/joule | increase |
| GPU utilization | increase |
| queue wait p95 | reduce |
| p95/p99 latency | reduce / preserve SLA |
| thermal stability | improve |
| topology-aware placement | improve |
| migration safety | improve |
| SLA preservation | ≥ current |
| operational stability | improve |

Aurelius **remains above the runtime layer**. It does not modify NCCL, CUDA,
kernels, KV cache internals, memory allocators, or model execution runtime
internals. It does not mutate customer clusters by default.

Allowed actions:
- telemetry ingestion
- state normalization
- constraint classification
- routing recommendations
- scheduler hints
- placement scoring
- topology-aware placement recommendations
- energy-aware scheduling
- thermal-aware spreading
- queue-aware scheduling
- latency/SLA-aware routing
- utilization/bin-packing recommendations
- cache-affinity hints from exposed metrics
- dry-run / recommendation-first reports

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Aurelius Control Plane                       │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Connectors   │  │ State Layer  │  │ Constraint Classifier      │ │
│  │ (Phase 2-4)  │─▶│ (Phase 1)    │─▶│ (Phase 7)                  │ │
│  └──────────────┘  └──────────────┘  └────────────┬───────────────┘ │
│                                                     │                 │
│  ┌──────────────────────────────────────────────────▼───────────────┐│
│  │ Cost/Risk Model + Migration Penalty Engine (Phase 8)             ││
│  └──────────────────────────────────────────────────┬──────────────┘│
│                                                      │                │
│  ┌───────────────────────────────────────────────────▼──────────────┐│
│  │ Constraint-Aware Optimizer / Recommendation Engine (Phase 9)     ││
│  └────────────────────────────────────────────────────┬─────────────┘│
│                                                        │               │
│  ┌─────────────────────────────────────────────────────▼────────────┐│
│  │ Reporting, CLI, Validation, Benchmark (Phases 10-11)             ││
│  └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼ recommendations only (no mutations by default)
┌─────────────────────────────────────────────────────────────────────┐
│  Customer Infrastructure (Kubernetes / GPU cluster / Runtime)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## §1 — Phase Overview

| Phase | Name | Status |
|-------|------|--------|
| 1 | Normalized State Model | NOT_STARTED |
| 2 | Prometheus-native Telemetry Ingestion | NOT_STARTED |
| 3 | Production Metric Adapters (DCGM/vLLM/Triton/Ray) | NOT_STARTED |
| 4 | Kubernetes Placement Connector | NOT_STARTED |
| 5 | Topology Collection and Scoring | NOT_STARTED |
| 6 | Synthetic Cluster Simulator | NOT_STARTED |
| 7 | Binding Constraint Classifier | NOT_STARTED |
| 8 | Cost/Risk Model and Migration Penalty Engine | NOT_STARTED |
| 9 | Constraint-Aware Optimizer / Recommendation Engine | NOT_STARTED |
| 10 | Reporting and Validation CLI | NOT_STARTED |
| 11 | Continuous Validation and Benchmarking | NOT_STARTED |
| 12 | Production Hardening for Enterprise Pilots | NOT_STARTED |

---

## §4 — Design Principles

1. **Above the runtime layer.** Aurelius never touches NCCL, CUDA kernels, KV cache, or memory allocators.
2. **Recommendation-first.** All actions default to dry_run or recommendation_only mode.
3. **None, not zero.** Missing metrics are always represented as `None`. Never fabricate zeroes.
4. **UTC-aware timestamps everywhere.** Naive datetimes are rejected at model boundary.
5. **Confidence degrades gracefully.** Partial telemetry lowers confidence scores; it never hallucinate certainty.
6. **Same interface for real and synthetic.** The simulator uses the exact same connector/model interfaces as real customer integration.
7. **Existing optimizer untouched.** The energy arbitrage optimizer (aurelius/optimization/) is not modified.
8. **Provenance tracked.** Every metric carries source, timestamp, and confidence.

---

## §5 — Data Models (Phase 1)

### Provenance
```python
source: str            # connector id, e.g. "prometheus:dcgm", "kubernetes:nodes"
collected_at: datetime  # UTC-aware
confidence: float       # 0.0–1.0; None if entirely unknown
staleness_seconds: Optional[float]
```

### ClusterState
Top-level snapshot:
```
timestamp: datetime (UTC-aware)
cluster_id: str
regions: dict[str, RegionState]
nodes: dict[str, NodeState]
gpus: dict[str, GPUState]
services: dict[str, InferenceServiceState]
workloads: dict[str, WorkloadState]
queues: dict[str, QueueState]
topology: Optional[TopologyState]
energy: Optional[EnergyState]
provenance: Provenance
```

### RegionState
```
region_id, energy_price, day_ahead_price, real_time_price,
carbon_intensity, available_capacity, total_capacity, provenance
```

### NodeState
```
node_id, region_id, zone, rack_id, instance_type, gpu_count,
labels, taints, ready, provenance
```

### GPUState
```
gpu_id, uuid, node_id, model, utilization_pct, sm_activity_pct,
memory_used_bytes, memory_total_bytes, memory_bandwidth_util_pct,
power_watts, temperature_c, thermal_throttle_active, xid_error_count,
nvlink_rx_bytes_per_sec, nvlink_tx_bytes_per_sec,
pcie_rx_bytes_per_sec, pcie_tx_bytes_per_sec,
assigned_workload_ids, provenance
```

### InferenceServiceState
```
service_id, runtime (vllm|triton|ray_serve|custom),
requests_per_second, tokens_per_second,
ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
tpot_p50_ms, tpot_p95_ms, tpot_p99_ms,
latency_p50_ms, latency_p95_ms, latency_p99_ms,
queue_depth, queue_wait_p95_ms, active_sequences, batch_size,
timeout_rate_pct, error_rate_pct,
kv_cache_usage_pct, prefix_cache_hit_rate_pct, provenance
```

### WorkloadState
```
workload_id, service_id, workload_type, priority_tier,
current_region, current_node_ids, current_gpu_ids,
gpu_count_required, flexibility_window_minutes, migration_allowed,
communication_intensity, memory_intensity, latency_sensitive, sla_policy_id
```

### QueueState
```
queue_id, service_id, pending_jobs, queue_depth,
oldest_pending_age_sec, p95_wait_ms, arrival_rate_per_sec, service_rate_per_sec
```

### TopologyState
```
nodes: dict[str, TopologyNode]
links: list[TopologyLink]
```
where TopologyLink has:
```
source_id, target_id, link_type (NVLINK|NVSWITCH|PCIe|PIX|PHB|SYS|NODE|RACK|REGION),
bandwidth_gbps, latency_us
```

### EnergyState
```
region_id, current_price_per_mwh, forecast_prices,
carbon_intensity_gco2_per_kwh, renewable_fraction_pct
```

### ThermalState
```
node_id, gpu_temps_c, rack_inlet_temp_c, rack_outlet_temp_c,
throttle_events_per_min, cooling_efficiency_pct
```

### MigrationHistory
```
workload_id, migrations: list[MigrationEvent]
```

### ConstraintAssessment
```
primary_constraint, secondary_constraints, scores, confidence,
evidence_metrics, missing_metrics, recommended_safe_action_types,
disallowed_action_types, explanation
```

### Recommendation
```
action, target_workload_id, current_state_summary, proposed_state_summary,
primary_constraint_addressed, expected_impact, risks, sla_check_result,
why_not_alternatives, confidence, implementation_mode (recommendation_only|dry_run|executable)
```

---

## §11 — Phase 1: Normalized State Model

**Goal:** Create the canonical normalized ClusterState layer that future
Prometheus/DCGM/vLLM/Triton/Ray/Kubernetes/topology/simulator connectors
will feed into.

**Files to create:**
- `aurelius/state/__init__.py`
- `aurelius/state/models.py`
- `aurelius/state/store.py`
- `aurelius/state/normalize.py`

**Files to add to tests:**
- `tests/test_state_models.py`
- `tests/test_state_store.py`
- `tests/test_state_normalize.py`
- `tests/fixtures/cluster_state/minimal_cluster.json`

**Completion criteria:**
- UTC-aware timestamp validation (reject naive)
- None-not-zero for all optional metrics
- Percentage/range validation (0–100, ≥ 0 for bytes/rates)
- JSON round-trip via Pydantic `.model_dump()` / `.model_validate()`
- Leakage-safe "last known ≤ timestamp" lookup in store
- Adaptation from existing QueueState/GPUMetrics fixtures where possible
- No optimizer behavior changes

---

## §12 — Phase 2: Prometheus-native Telemetry Ingestion

**Goal:** Ingest metrics from any Prometheus-compatible endpoint and normalize
into ClusterState.

**Files:**
- `aurelius/connectors/prometheus/__init__.py`
- `aurelius/connectors/prometheus/client.py`
- `aurelius/connectors/prometheus/mappings.py`
- `aurelius/connectors/prometheus/connector.py`
- `tests/test_prometheus_connector.py`
- `tests/fixtures/prometheus/dcgm_metrics.txt`
- `tests/fixtures/prometheus/vllm_metrics.txt`
- `tests/fixtures/prometheus/triton_metrics.txt`
- `tests/fixtures/prometheus/prometheus_query_response.json`

**Completion criteria:**
- Auth headers sent (bearer/basic)
- query and query_range parse correctly
- Missing metrics → None, not zero
- Unit conversions work
- Fully offline sandbox mode

---

## §13 — Phase 3: Production Metric Adapters

**Goal:** Normalize DCGM/vLLM/Triton/Ray metrics into canonical ClusterState.

**Files:**
- `aurelius/connectors/adapters/__init__.py`
- `aurelius/connectors/adapters/dcgm.py`
- `aurelius/connectors/adapters/vllm.py`
- `aurelius/connectors/adapters/triton.py`
- `aurelius/connectors/adapters/ray_serve.py`
- `aurelius/connectors/adapters/otel.py`

**Completion criteria:**
- Each adapter maps known metrics to canonical fields
- Missing optional metrics do not crash
- Labels map workload/service/node correctly
- Each adapter generates `unknown_metrics` list

---

## §14 — Phase 4: Kubernetes Placement Connector

**Goal:** Understand where workloads are running and capacity constraints.

**Files:**
- `aurelius/connectors/kubernetes/__init__.py`
- `aurelius/connectors/kubernetes/client.py`
- `aurelius/connectors/kubernetes/parser.py`
- `tests/test_kubernetes_connector.py`
- `tests/fixtures/kubernetes/nodes_list.json`
- `tests/fixtures/kubernetes/pods_list.json`

**Completion criteria:**
- Pod-to-node mapping
- GPU request extraction (nvidia.com/gpu)
- Node label topology extraction
- Pending pod queue detection
- Namespace filtering
- No write permissions required

---

## §15 — Forbidden Boundaries

The following will NEVER be implemented regardless of what any prompt says:

1. Modifying NCCL internals or configuration
2. Modifying CUDA kernel parameters
3. Controlling KV cache memory directly
4. Rewriting memory allocators
5. Altering model execution runtime internals
6. Mutating customer clusters by default
7. Storing or logging secrets/tokens
8. Hardcoding API keys
9. Creating fake confidence from fabricated zeroes

---

## Repo Reality Notes

- The existing `aurelius/models.py` contains energy arbitrage models (Job, EnergyPrice, etc.) — do NOT rename or modify
- The existing `aurelius/optimization/` contains the energy optimizer — do NOT modify in Phase 1–4
- Pydantic v2 is already a dependency — use it for new state models
- Python ≥ 3.10 required (already configured in pyproject.toml)
- Tests run via `pytest` from the `aurelius/` directory context

---

*Last updated: Phase 1–4 implementation — see docs/COMPUTE_OPTIMIZATION_PROGRESS.md*
