# SLA Ingestion + SLA-Aware Optimization

Aurelius can ingest **hard** and **soft** Service Level Agreements (SLAs),
attach them to workloads/services/regions, and let them **materially change**
optimization decisions. SLAs are not just stored or logged — a hard SLA
violation **blocks** an optimization action, and a soft SLA violation
**reduces** an action's score so cheaper-but-riskier moves lose to safe ones.

This document covers what is supported, how policies are loaded, how they
affect optimization, the example configs, how to run the tests, and how to read
the SLA report.

---

## 1. What SLAs are supported

### Hard constraints (blocking)
A violation makes an action **disallowed**.

| Field | Meaning |
|---|---|
| `allowed_regions` | Action's target region must be in this set |
| `forbidden_regions` | Target region must not be in this set |
| `data_residency_region` | Target region must equal this region |
| `max_p95_latency_ms` | Predicted p95 latency ceiling |
| `max_p99_latency_ms` | Predicted p99 latency ceiling |
| `max_queue_wait_ms` | Predicted queue wait ceiling |
| `min_availability_pct` | Predicted availability floor |
| `max_error_rate_pct` | Predicted error-rate ceiling |
| `max_timeout_rate_pct` | Predicted timeout-rate ceiling |
| `migration_allowed` | If `false`, any region-changing action is blocked |
| `max_migrations_per_hour` | Cap on migrations within the rolling hour |
| `no_migration_windows` | `[[start_iso, end_iso], ...]` windows where migration is blocked |
| `required_capacity_buffer_pct` | Predicted capacity buffer floor at destination |

### Soft constraints (penalizing)
A violation does **not** block — it adds to `soft_penalty_score`, lowering the
action's rank.

| Field | Meaning |
|---|---|
| `preferred_regions` | Leaving this set is penalized |
| `target_cost_per_token` | Exceeding it is penalized |
| `target_tokens_per_joule` | Falling below it is penalized |
| `target_gpu_utilization_pct` | Deviating from it is penalized |
| `preferred_carbon_intensity` | Exceeding it (gCO2/kWh) is penalized |
| `preferred_energy_price_percentile` | Exceeding it is penalized |
| `preferred_latency_headroom_pct` | Less headroom than this is penalized |
| `max_acceptable_savings_tradeoff_pct` | How much savings you'll sacrifice to honor `preferred_regions` |
| `optimization_aggressiveness` | `conservative` \| `balanced` \| `aggressive` |

---

## 2. Priority tiers

Each policy has a tier supplying **default** hard/soft behavior. Explicit
fields in a policy always override the tier defaults; unspecified fields are
filled from the tier. A field still `None` after merging is simply **not
enforced** — Aurelius never invents a guarantee it wasn't given.

| Tier | Posture |
|---|---|
| `critical` | **Safest.** No migrations (`migration_allowed=false`), large capacity buffer, tight latency, conservative |
| `latency_sensitive` | Tight latency, migrations rate-limited |
| `standard` | Balanced cost vs SLA risk |
| `flexible` | Tolerates moderate SLA risk for savings |
| `batch` | **Most cost-optimized.** Aggressive migration allowed, only reliability floors |

See `aurelius/sla/schema.py::TIER_DEFAULTS` for the exact numbers.

---

## 3. How policies are loaded

Configs are JSON or YAML. Top level is `{"policies": [...]}` (optionally with
`"default": "<policy name>"` and `"enabled": true|false`), or a bare single
policy, or a bare list.

```python
from aurelius.sla import SLALoader

registry = SLALoader.load_file("configs/sla_examples/inference_critical.yaml")
registry = SLALoader.load_dir("configs/sla_examples")        # merge a directory
policy   = registry.resolve(workload_id="inference-prod",
                            workload_type="realtime_inference")
```

**Validation is strict.** Malformed configs raise `SLAValidationError` listing
*all* problems (unknown fields, out-of-range percentages, `p99 < p95`,
regions in both `allowed` and `forbidden`, residency outside `allowed`, etc.).
Malformed configs are never silently accepted.

**Resolution order** for a workload: exact `applies_to_workloads` id →
`applies_to_workload_types` → registry default → `None` (no policy, legacy
behavior).

---

## 4. How SLAs affect optimization

### The correction engine
`aurelius.sla.evaluate_action_against_sla(action, workload, current_state,
predicted_state, sla_policy)` returns an `SLAEvaluation`:

- `allowed: bool` — `False` if any hard constraint is violated
- `violated_hard_constraints: list[str]`
- `soft_penalty_score: float`
- `risk_score: float` (+ a `risk_breakdown` of migration / latency / queue /
  availability / capacity / thermal penalties)
- `corrected_action` — the safe fallback (keep current placement) when blocked
- `explanation: str`

### The selector
`SLAAwareActionSelector.select(...)` ranks candidate actions by:

```
score = expected_savings
        - migration_penalty
        - latency_risk_penalty
        - queue_risk_penalty
        - availability_risk_penalty
        - soft_sla_penalty
subject to: hard_sla_constraints satisfied
```

If the highest-savings action violates a hard SLA, the selector chooses the
**next-best SLA-safe** action, or the **no-op** (keep current placement) if
nothing safe beats keeping put. It also implements the soft preferred-region
tradeoff: a pricier preferred-region option wins over a cheaper non-preferred
one when the savings gap is within `max_acceptable_savings_tradeoff_pct`.

### Wired into the real optimizer (`JobScheduler`)
`JobScheduler` accepts an optional `sla_registry` (+ `region_contexts`,
`current_states`). When a policy resolves for a job:

- region placements that violate **hard** constraints are **excluded** from the
  candidate search (`_find_best_slot`);
- **soft + risk** penalties are folded into each candidate's objective score, so
  safe-but-slightly-pricier placements outrank risky cheap ones;
- mid-job **migrations are suppressed** when `migration_allowed=false`;
- explainable logs are emitted (`SLA correction for <job>: unconstrained
  placement=X -> SLA-aware placement=Y`, plus the blocking violations).

**Default behavior is preserved.** With no `sla_registry` (or a disabled one),
the scheduler behaves exactly as before — SLAs never change a decision unless
enforcement is enabled.

---

## 5. Telemetry: what is real vs placeholder

The engine reasons over a **current** `WorkloadState` and a **predicted**
post-action `WorkloadState`. Supported inputs: p95/p99 latency, queue depth &
wait, GPU utilization, region, energy price, carbon intensity, error rate,
timeout rate, migration count, availability, capacity buffer, priority tier.

- `WorkloadState` / `RegionContext` are **real** data carriers — populate them
  from your telemetry (Prometheus/DCGM via `aurelius/ingestion/dcgm_provider.py`,
  the queue provider, latency histograms). Implement the `TelemetryProvider`
  protocol to wire a live backend.
- `HeuristicPredictor` is a **conservative placeholder** (no trained model). It
  applies pessimistic deltas — migration inflates p99 and adds cold-start queue
  wait; low spare capacity / thermal stress raise tail-latency risk;
  consolidation raises utilization but lengthens queues. Every assumption is
  marked `# HEURISTIC` / `# TODO`. Intent: the SLA gate fails **safe** rather
  than approving risk on optimistic numbers.

**Unknown metrics**: if an SLA constrains a metric you have no telemetry for,
the default is to **not block** (and report it under `unknown_metrics`) so
Aurelius never *claims* an SLA is met on missing data. Pass
`block_on_unknown=True` for strict fail-closed behavior.

---

## 6. Example configs

Under `configs/sla_examples/`:

- `inference_critical.yaml` — critical real-time inference (residency + latency
  hard SLAs, no migration).
- `batch_training.yaml` — batch/training (aggressive, cost-optimized).
- `standard_service.json` — balanced default + an EU-residency policy with a
  `no_migration_windows` holiday freeze.

---

## 7. How to run the tests

```bash
# SLA engine: schema, loader/validator, evaluator (one test per SLA type),
# selector (with vs without SLA, aggressiveness, tiers, tradeoff, thermal).
pytest tests/test_sla_engine.py -q

# SLA wired into the real JobScheduler + the before/after report.
pytest tests/test_sla_optimization.py -q
```

Each SLA type has a test asserting it **changes behavior** when relevant, and
the scheduler tests run the optimizer both **unconstrained** and **SLA-enabled**
to prove the region/migration choice differs.

---

## 8. SLA-aware optimization report

```bash
# Built-in illustrative scenario:
python -m aurelius.cli sla-report --config configs/sla_examples --demo

# Your own scenario:
python -m aurelius.cli sla-report --config configs/sla_examples \
    --scenario my_scenario.json --output report.json
```

Example output:

```
Workload: inference-prod
Unconstrained action: migrate_workload -> ercot
Unconstrained expected savings: 18.4%
SLA-aware action: keep us-east
SLA-aware expected savings: 0.0%
Blocked because:
  - region 'ercot' not in allowed_regions ['us-east', 'us-west']
  - predicted p99_latency_ms 12600.0 > limit 500.0
  - migration_allowed=false but action migrates the workload
Savings sacrificed for SLA safety: 18.4%

Workload: batch-job
Unconstrained action: migrate_workload -> ercot
SLA-aware action: migrate_workload -> ercot
SLA-aware expected savings: 18.4%
No hard SLA violations
```

**Scenario file schema** (`--scenario`):

```json
{
  "region_contexts": {
    "ercot": {"baseline_p99_latency_ms": 6000, "spare_capacity_pct": 10,
              "energy_price": 10, "thermally_stressed": true}
  },
  "workloads": [
    {
      "id": "inference-prod",
      "workload_type": "realtime_inference",
      "policy": "inference-prod",
      "current_state": {"region": "us-east", "p99_latency_ms": 1500,
                        "availability_pct": 99.99, "capacity_buffer_pct": 50},
      "candidate_actions": [
        {"action_type": "migrate_workload", "target_region": "ercot",
         "expected_savings_pct": 18.4}
      ]
    }
  ]
}
```

---

## 9. Guardrails / limitations

- Aurelius does **not** claim SLA guarantees beyond what telemetry supports —
  unknown metrics are reported, not assumed satisfied.
- No fake production integrations are bundled; `StaticTelemetryProvider` is for
  tests/backtests, and `HeuristicPredictor` is explicitly a heuristic.
- Default (no SLA config) behavior is byte-for-byte the prior optimizer
  behavior; SLA enforcement is opt-in.
- **TODOs**: replace `HeuristicPredictor` deltas with a learned latency/queue
  predictor once per-region telemetry history is available (M/M/c queueing for
  `scale_replicas`, cold-start curves for migration). Marked in
  `aurelius/sla/telemetry.py`.
```
