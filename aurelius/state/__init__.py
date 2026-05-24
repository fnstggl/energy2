"""Canonical normalized cluster state layer for Aurelius constraint-aware orchestration.

This package provides the vendor-neutral internal state model consumed by the
constraint classifier, cost/risk model, and recommendation engine. It is
additive groundwork only — no optimizer logic is changed in this phase.

Connectors (Prometheus, DCGM, vLLM, Kubernetes, topology) are not built here;
they will normalize their outputs into ClusterState in later phases.
"""

from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    MigrationEvent,
    MigrationHistory,
    NodeState,
    Provenance,
    Recommendation,
    RegionState,
    ThermalState,
    TopologyLinkType,
    TopologyState,
)
from aurelius.state.normalize import (
    adapt_gpu_metrics,
    adapt_queue_state,
    make_provenance,
    validate_non_negative,
    validate_percentage,
    validate_utc_aware,
)
from aurelius.state.store import StateStore

__all__ = [
    # Models
    "ClusterState",
    "ConstraintAssessment",
    "ConstraintType",
    "EnergyState",
    "GPUState",
    "InferenceServiceState",
    "MigrationEvent",
    "MigrationHistory",
    "NodeState",
    "Provenance",
    "Recommendation",
    "RegionState",
    "ThermalState",
    "TopologyLinkType",
    "TopologyState",
    # Normalize utilities
    "adapt_gpu_metrics",
    "adapt_queue_state",
    "make_provenance",
    "validate_percentage",
    "validate_non_negative",
    "validate_utc_aware",
    # Store
    "StateStore",
]
