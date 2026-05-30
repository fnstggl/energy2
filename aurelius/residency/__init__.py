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

from .decision import (  # noqa: E402
    CandidateScore,
    SafetyContext,
    choose_residency_decision,
    score_residency_candidate,
)
from .models import (
    CONFIDENCE_LEVELS,
    EVENT_TYPES,
    PRIORITY_CLASSES,
    RESIDENCY_ACTIONS,
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyEvent,
    ModelResidencyRequest,
    ModelResidencySnapshot,
    RequestResidencyObservation,
    ResidencyAction,
    ResidencyDecision,
    ResidencySchemaError,
    parse_timestamp,
)
from .sim import (  # noqa: E402
    REAL_MODE,
    SIMULATOR_MODE,
    RealModeMutationError,
    apply_residency_decision,
)

__all__ = [
    "ModelResidencyEvent",
    "ModelResidencySnapshot",
    "RequestResidencyObservation",
    "ResidencySchemaError",
    "EVENT_TYPES",
    "CONFIDENCE_LEVELS",
    "parse_timestamp",
    # decision engine
    "ModelResidencyRequest",
    "ModelLocationState",
    "ModelLoadProfile",
    "ResidencyDecision",
    "ResidencyAction",
    "RESIDENCY_ACTIONS",
    "PRIORITY_CLASSES",
    "choose_residency_decision",
    "score_residency_candidate",
    "SafetyContext",
    "CandidateScore",
    # simulator execution
    "apply_residency_decision",
    "SIMULATOR_MODE",
    "REAL_MODE",
    "RealModeMutationError",
]
