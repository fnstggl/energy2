"""Model-residency / cold-start telemetry — read-only ingestion + shadow logging.

This package implements the *observation* substrate described in
``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` and
``docs/PILOT_TELEMETRY_CONTRACT.md``: data models for model/adapter residency
telemetry, read-only ingestion adapters (JSONL/CSV, vLLM, K8s/Prometheus/DCGM
join helper), honest derived metrics, and a recommendation-only shadow log.

Hard boundaries (enforced by design, asserted by tests):

- **Read-only.** Nothing here mutates a real cluster, loads/evicts a real
  model, calls a Kubernetes write API, or reroutes real traffic.
- **No autonomous prewarming.** The shadow log emits *recommendations only*
  (``executed=False`` by default); promoting a recommendation is out of scope.
- **No ML training, no optimizer-constant tuning, no robust-energy-engine
  change.**
- **Missing data is never treated as zero.** Optional fields default to
  ``None`` (unknown) and are excluded from metric denominators — never silently
  zero-filled (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1 null-handling).
- **No production-savings claim.** All numbers are directional / pilot-telemetry
  diagnostics until the ``docs/RESULTS.md`` §8 production-claim gate is met.
"""

from .models import (
    CONFIDENCE_LEVELS,
    EVENT_TYPES,
    ModelResidencyEvent,
    ModelResidencySnapshot,
    RequestResidencyObservation,
    ResidencySchemaError,
    parse_timestamp,
)

__all__ = [
    "ModelResidencyEvent",
    "ModelResidencySnapshot",
    "RequestResidencyObservation",
    "ResidencySchemaError",
    "EVENT_TYPES",
    "CONFIDENCE_LEVELS",
    "parse_timestamp",
]
